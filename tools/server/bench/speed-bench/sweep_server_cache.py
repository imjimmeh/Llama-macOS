#!/usr/bin/env python3
"""Run automated grid sweep across batch, ubatch, exc, and excp on llama-server."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SWEEP_CONFIGS: list[dict[str, Any]] = [
    # 1. Control Baseline
    {"name": "Control (exc=0, b=4096, ub=2048)", "exc": 0, "excp": 32, "b": 4096, "ub": 2048, "pe": None},

    # 2. Batch / Ubatch Sweep with Dynamic 128M (excp=32)
    {"name": "Dynamic 128M (b=4096, ub=2048, excp=32)", "exc": 128, "excp": 32, "b": 4096, "ub": 2048, "pe": None},
    {"name": "Dynamic 128M (b=2048, ub=1024, excp=32)", "exc": 128, "excp": 32, "b": 2048, "ub": 1024, "pe": None},
    {"name": "Dynamic 128M (b=2048, ub=512, excp=32)",  "exc": 128, "excp": 32, "b": 2048, "ub": 512,  "pe": None},
    {"name": "Dynamic 128M (b=1024, ub=512, excp=32)",  "exc": 128, "excp": 32, "b": 1024, "ub": 512,  "pe": None},
    {"name": "Dynamic 128M (b=512,  ub=512, excp=32)",  "exc": 128, "excp": 32, "b": 512,  "ub": 512,  "pe": None},

    # 3. Rebalance Period Sweep (b=4096, ub=2048, exc=128M)
    {"name": "Dynamic 128M (excp=16, b=4096, ub=2048)", "exc": 128, "excp": 16, "b": 4096, "ub": 2048, "pe": None},
    {"name": "Dynamic 128M (excp=64, b=4096, ub=2048)", "exc": 128, "excp": 64, "b": 4096, "ub": 2048, "pe": None},

    # 4. Cache Size Sweep (excp=32, b=4096, ub=2048)
    {"name": "Dynamic 64M  (excp=32, b=4096, ub=2048)", "exc": 64,  "excp": 32, "b": 4096, "ub": 2048, "pe": None},
    {"name": "Dynamic 256M (excp=32, b=4096, ub=2048)", "exc": 256, "excp": 32, "b": 4096, "ub": 2048, "pe": None},

    # 5. Hybrid Pinned Manifest Sweep
    {"name": "Hybrid 128M (pinned_v2_128m, excp=32)",    "exc": 128, "excp": 32, "b": 4096, "ub": 2048, "pe": "tools/results/expert-cache/bundle-v2/pinned_bundle_v2_128mb.json"},
    {"name": "Hybrid 256M (pinned_v2_256m, excp=32)",    "exc": 256, "excp": 32, "b": 4096, "ub": 2048, "pe": "tools/results/expert-cache/bundle-v2/pinned_bundle_v2_256mb.json"},
]


def run_sweep(output_dir: Path, limit: int = 5, osl: int = 2048) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_results: list[dict[str, Any]] = []

    print(f"Starting parameter sweep ({len(SWEEP_CONFIGS)} configurations)...", file=sys.stderr)

    for idx, cfg in enumerate(SWEEP_CONFIGS, start=1):
        name = cfg["name"]
        prefix = f"sweep-{idx:02d}-{name.replace(' ', '_').replace('=', '').replace('(', '').replace(')', '').replace(',', '')}"
        print(f"\n=======================================================", file=sys.stderr)
        print(f"[{idx}/{len(SWEEP_CONFIGS)}] Running config: {name}", file=sys.stderr)
        print(f"=======================================================", file=sys.stderr)

        cmd = [
            sys.executable, "tools/server/bench/speed-bench/run_repeat_bench.py",
            "--prefix", prefix,
            "--output-dir", str(output_dir),
            "--cache-mib", str(cfg["exc"]),
            "--cache-period", str(cfg["excp"]),
            "--batch-size", str(cfg["b"]),
            "--ubatch-size", str(cfg["ub"]),
            "--iterations", "2",
            "--limit", str(limit),
            "--osl", str(osl),
            "--category", "coding",
            "--ctx-size", "128000",
            "--sps", "0.0",
        ]
        if cfg["pe"]:
            cmd.extend(["--pinned-experts", str(cfg["pe"])])

        t0 = time.time()
        res = subprocess.run(cmd)
        elapsed = time.time() - t0

        if res.returncode != 0:
            print(f"Config {name} returned error code {res.returncode}", file=sys.stderr)
            continue

        p1_path = output_dir / f"{prefix}-pass-1.json"
        p2_path = output_dir / f"{prefix}-pass-2.json"

        p1_speed, p2_speed = None, None
        p1_lat, p2_lat = None, None

        if p1_path.exists():
            d1 = json.loads(p1_path.read_text(encoding="utf-8"))
            p1_speed = d1["summary"][0]["avg_pred_t_s"]
            p1_lat = d1["summary"][0]["avg_latency"]

        if p2_path.exists():
            d2 = json.loads(p2_path.read_text(encoding="utf-8"))
            p2_speed = d2["summary"][0]["avg_pred_t_s"]
            p2_lat = d2["summary"][0]["avg_latency"]

        record = {
            "config": name,
            "exc": cfg["exc"],
            "excp": cfg["excp"],
            "batch": cfg["b"],
            "ubatch": cfg["ub"],
            "pinned": cfg["pe"] is not None,
            "pass_1_speed": p1_speed,
            "pass_2_speed": p2_speed,
            "pass_1_latency": p1_lat,
            "pass_2_latency": p2_lat,
            "elapsed_s": elapsed,
        }
        summary_results.append(record)

        # Print incremental summary table
        print_table(summary_results)

    summary_file = output_dir / "sweep_summary.json"
    summary_file.write_text(json.dumps(summary_results, indent=2), encoding="utf-8")
    print(f"\nFinal sweep summary written to {summary_file}", file=sys.stderr)
    return summary_results


def print_table(records: list[dict[str, Any]]) -> None:
    ctrl_p2 = records[0]["pass_2_speed"] if records and records[0].get("pass_2_speed") else None

    print("\n" + "=" * 110)
    print(f"{'Configuration':<45} | {'Pass 1 (tok/s)':<14} | {'Pass 2 (tok/s)':<14} | {'Pass 2 Delta vs Ctrl':<20} | {'Pass 2 Latency':<12}")
    print("-" * 110)

    for r in records:
        name = r["config"]
        p1 = f"{r['pass_1_speed']:.2f}" if r["pass_1_speed"] else "n/a"
        p2 = f"{r['pass_2_speed']:.2f}" if r["pass_2_speed"] else "n/a"
        lat = f"{r['pass_2_latency']:.2f}s" if r["pass_2_latency"] else "n/a"

        if ctrl_p2 and r["pass_2_speed"]:
            delta = (r["pass_2_speed"] - ctrl_p2) / ctrl_p2 * 100.0
            delta_str = f"{delta:+6.2f}%"
        else:
            delta_str = "baseline"

        print(f"{name:<45} | {p1:<14} | {p2:<14} | {delta_str:<20} | {lat:<12}")

    print("=" * 110 + "\n")


if __name__ == "__main__":
    out = Path("tools/results/expert-cache/server-speed-bench/sweep-grid")
    run_sweep(out, limit=5, osl=2048)
