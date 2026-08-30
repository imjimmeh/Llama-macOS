#!/usr/bin/env python3
"""Run comparison between baseline upstream llama.cpp and expert-cache tuned llama.cpp."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

BASELINE_SERVER = Path("G:/code/AI/llama.cpp/build/bin/Release/llama-server.exe")
TUNED_SERVER = Path("G:/code/AI/llamacpptuned/llama.cpp/build/bin/Release/llama-server.exe")
MODEL_PATH = Path("C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf")


def run_benchmark(
    server_path: Path,
    prefix: str,
    output_dir: Path,
    cache_mib: int = 0,
    cache_period: int = 32,
    batch_size: int = 1024,
    ubatch_size: int = 512,
    limit: int = 5,
    osl: int = 2048,
) -> dict[str, Any]:
    cmd = [
        sys.executable, "tools/server/bench/speed-bench/run_repeat_bench.py",
        "--server", str(server_path),
        "--model", str(MODEL_PATH),
        "--prefix", prefix,
        "--output-dir", str(output_dir),
        "--cache-mib", str(cache_mib),
        "--cache-period", str(cache_period),
        "--batch-size", str(batch_size),
        "--ubatch-size", str(ubatch_size),
        "--iterations", "2",
        "--limit", str(limit),
        "--osl", str(osl),
        "--category", "coding",
        "--ctx-size", "128000",
        "--sps", "0.0",
    ]

    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0

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

    return {
        "prefix": prefix,
        "server": str(server_path),
        "cache_mib": cache_mib,
        "cache_period": cache_period,
        "batch": batch_size,
        "ubatch": ubatch_size,
        "pass_1_speed": p1_speed,
        "pass_2_speed": p2_speed,
        "pass_1_latency": p1_lat,
        "pass_2_latency": p2_lat,
        "elapsed_s": elapsed,
    }


def main() -> int:
    out_dir = Path("tools/results/expert-cache/server-speed-bench/baseline-comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================", file=sys.stderr)
    print("Running Baseline llama.cpp (Upstream G:/code/AI/llama.cpp)", file=sys.stderr)
    print("==================================================================", file=sys.stderr)

    baseline_res = run_benchmark(
        server_path=BASELINE_SERVER,
        prefix="baseline-upstream-llamacpp",
        output_dir=out_dir,
        cache_mib=0,
        batch_size=1024,
        ubatch_size=512,
    )

    print("\n==================================================================", file=sys.stderr)
    print("Running Tuned llama.cpp with Dynamic Expert Cache (128M)", file=sys.stderr)
    print("==================================================================", file=sys.stderr)

    tuned_res = run_benchmark(
        server_path=TUNED_SERVER,
        prefix="tuned-expert-cache-128m",
        output_dir=out_dir,
        cache_mib=128,
        cache_period=128,
        batch_size=1024,
        ubatch_size=512,
    )

    records = [baseline_res, tuned_res]
    summary_file = out_dir / "comparison_summary.json"
    summary_file.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print("\n" + "=" * 110)
    print(f"{'Server Implementation':<35} | {'Pass 1 Speed':<14} | {'Pass 2 Speed':<14} | {'Speedup vs Baseline':<20} | {'Pass 2 Latency':<12}")
    print("-" * 110)

    base_p2 = baseline_res["pass_2_speed"]
    for r in records:
        lbl = "Baseline Upstream llama.cpp" if "baseline" in r["prefix"] else "Tuned (Dynamic Cache 128M)"
        p1 = f"{r['pass_1_speed']:.2f} tok/s" if r['pass_1_speed'] else "n/a"
        p2 = f"{r['pass_2_speed']:.2f} tok/s" if r['pass_2_speed'] else "n/a"
        lat = f"{r['pass_2_latency']:.2f}s" if r['pass_2_latency'] else "n/a"
        if base_p2 and r["pass_2_speed"]:
            diff = (r["pass_2_speed"] - base_p2) / base_p2 * 100.0
            diff_str = f"{diff:+6.2f}%" if r != baseline_res else "baseline (0.00%)"
        else:
            diff_str = "n/a"
        print(f"{lbl:<35} | {p1:<14} | {p2:<14} | {diff_str:<20} | {lat:<12}")

    print("=" * 110 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
