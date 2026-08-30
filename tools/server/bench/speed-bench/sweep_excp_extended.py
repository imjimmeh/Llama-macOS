#!/usr/bin/env python3
"""Run extended excp sweep (32, 64, 128, 256) on the winning batch/cache configuration."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

EXCP_CONFIGS: list[dict[str, Any]] = [
    {"name": "Dynamic 128M (b=1024, ub=512, excp=32)",  "exc": 128, "excp": 32,  "b": 1024, "ub": 512},
    {"name": "Dynamic 128M (b=1024, ub=512, excp=64)",  "exc": 128, "excp": 64,  "b": 1024, "ub": 512},
    {"name": "Dynamic 128M (b=1024, ub=512, excp=128)", "exc": 128, "excp": 128, "b": 1024, "ub": 512},
    {"name": "Dynamic 128M (b=1024, ub=512, excp=256)", "exc": 128, "excp": 256, "b": 1024, "ub": 512},
]


def run_extended_sweep(output_dir: Path, limit: int = 5, osl: int = 2048) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_results: list[dict[str, Any]] = []

    print(f"Starting extended excp sweep ({len(EXCP_CONFIGS)} configurations)...", file=sys.stderr)

    for idx, cfg in enumerate(EXCP_CONFIGS, start=1):
        name = cfg["name"]
        prefix = f"excp-sweep-{idx:02d}-{name.replace(' ', '_').replace('=', '').replace('(', '').replace(')', '').replace(',', '')}"
        print(f"\n=======================================================", file=sys.stderr)
        print(f"[{idx}/{len(EXCP_CONFIGS)}] Running config: {name}", file=sys.stderr)
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
            "pass_1_speed": p1_speed,
            "pass_2_speed": p2_speed,
            "pass_1_latency": p1_lat,
            "pass_2_latency": p2_lat,
            "elapsed_s": elapsed,
        }
        summary_results.append(record)

        print("\n" + "=" * 105)
        print(f"{'Configuration':<45} | {'Pass 1 (tok/s)':<14} | {'Pass 2 (tok/s)':<14} | {'Pass 2 Latency':<12}")
        print("-" * 105)
        for r in summary_results:
            p1 = f"{r['pass_1_speed']:.2f}" if r['pass_1_speed'] else "n/a"
            p2 = f"{r['pass_2_speed']:.2f}" if r['pass_2_speed'] else "n/a"
            lat = f"{r['pass_2_latency']:.2f}s" if r['pass_2_latency'] else "n/a"
            print(f"{r['config']:<45} | {p1:<14} | {p2:<14} | {lat:<12}")
        print("=" * 105 + "\n")

    summary_file = output_dir / "excp_extended_summary.json"
    summary_file.write_text(json.dumps(summary_results, indent=2), encoding="utf-8")
    print(f"\nExtended sweep summary written to {summary_file}", file=sys.stderr)
    return summary_results


if __name__ == "__main__":
    out = Path("tools/results/expert-cache/server-speed-bench/sweep-excp-extended")
    run_extended_sweep(out, limit=5, osl=2048)
