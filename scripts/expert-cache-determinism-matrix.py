#!/usr/bin/env python3

# Matrix runner for the expert-cache determinism check.
#
# Runs the three deterministic token-hash rows from the Phase 0 plan against
# the MTP-Quality model, each with a fresh llama-server launched with the
# preset performance args (batch 4096, ubatch 2048, fit on fit-target 256,
# etc.). Every row issues one completion at temperature 0 / top-k 1 / seed 42
# / ignore_eos for a fixed number of tokens and records the SHA-256 of the
# generated token IDs.
#
# Rows:
#   A  baseline, no expert cache            -exc 0
#   B  expert cache on                     -exc 64M -excp 64
#   E  expert cache on + MTP draft         -exc 64M -excp 64 + MTP flags
#   F  expert cache on + routing predictor -exc 64M -excp 64 + routing-predictor flags
#
# All four must produce the same SHA-256 for determinism to hold. The runner
# prints a comparison table and exits nonzero if any row fails or if any hash
# differs from the reference.
#
# Optional: set REFERENCE_HASH to a historical hash for this exact prompt,
# seed and token count (with cache off) to additionally check byte-for-byte
# reproducibility against that reference.
#
# Usage:
#   python scripts/expert-cache-determinism-matrix.py

import json
import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DET = os.path.join(SCRIPT_DIR, "expert-cache-determinism.py")

# If set, every row must also match this hash (cache-off reference).
REFERENCE_HASH = ""

ROWS = [
    {
        "name": "A",
        "desc": "baseline (no expert cache)",
        "exc": "0",
        "excp": 0,
        "extra": "",
    },
    {
        "name": "B",
        "desc": "expert cache on",
        "exc": "64M",
        "excp": 64,
        "extra": "",
    },
    {
        "name": "E",
        "desc": "expert cache on + MTP draft",
        "exc": "64M",
        "excp": 64,
        "extra": (
            "--spec-type draft-mtp --spec-draft-n-max 2 "
            "--mtp-dynamic-offload"
        ),
    },
    {
        "name": "F",
        "desc": "expert cache on + routing predictor",
        "exc": "64M",
        "excp": 64,
        "extra": "--routing-predictor-horizon 8 --routing-predictor-stats",
    },
]


def run_row(row: dict) -> dict:
    print(f"\n===== Row {row['name']} : {row['desc']} =====", flush=True)
    json_out = os.path.join(SCRIPT_DIR, f"determinism-row-{row['name']}.json")
    cmd = [
        sys.executable, DET,
        "--exc", row["exc"],
        "--excp", str(row["excp"]),
        "--json-out", json_out,
    ]
    if row["extra"]:
        cmd += ["--extra-args", row["extra"]]
    proc = subprocess.run(cmd, capture_output=False, text=True)
    if proc.returncode != 0:
        return {"name": row["name"], "desc": row["desc"], "sha256": "RUN_ERROR", "tok_s": 0.0}
    if not os.path.isfile(json_out):
        return {"name": row["name"], "desc": row["desc"], "sha256": "NO_OUTPUT", "tok_s": 0.0}
    with open(json_out, "r", encoding="utf-8") as f:
        rec = json.load(f)
    return {
        "name": row["name"],
        "desc": row["desc"],
        "sha256": rec.get("sha256_tokens", "UNKNOWN"),
        "tok_s": rec.get("tok_s", 0.0),
    }


def main() -> int:
    print("expert-cache determinism matrix", flush=True)
    results = [run_row(row) for row in ROWS]

    print("\n===== Results =====", flush=True)
    print(f"{'Row':<4}{'desc':<28}{'tok/s':>8}  sha256", flush=True)
    hashes = []
    ok = True
    for r in results:
        print(f"{r['name']:<4}{r['desc']:<28}{r['tok_s']:>8.2f}  {r['sha256']}", flush=True)
        if r["sha256"] in ("RUN_ERROR", "NO_OUTPUT", "UNKNOWN"):
            ok = False
        else:
            hashes.append(r["sha256"])

    if hashes and len(set(hashes)) == 1:
        print(f"\nPASS: all rows produced identical token hash {hashes[0]}", flush=True)
    else:
        print("\nFAIL: token hashes differ between rows", flush=True)
        ok = False

    if REFERENCE_HASH and hashes and REFERENCE_HASH not in hashes:
        print(f"\nFAIL: hash {hashes[0]} does not match reference {REFERENCE_HASH}", flush=True)
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())