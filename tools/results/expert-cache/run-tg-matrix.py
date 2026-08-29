#!/usr/bin/env python3
"""Run alternating cache-off/cache-on TG measurements and retain raw JSONL."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="GGUF model path")
    parser.add_argument("--bench", type=Path, default=Path("build/bin/Release/llama-bench.exe"))
    parser.add_argument("--output-dir", type=Path, default=Path("tools/results/expert-cache"))
    parser.add_argument("--prefix", help="Output filename prefix; defaults to the current UTC timestamp")
    parser.add_argument("--runs", type=int, default=5, help="Alternating control/cache pairs")
    parser.add_argument("--cache-mib", type=int, default=128)
    parser.add_argument("--cache-period", type=int, default=256)
    parser.add_argument("--max-swaps", type=int, default=-1)
    parser.add_argument("--n-gen", type=int, default=128)
    parser.add_argument("--load-mode", default="mlock")
    parser.add_argument("--fit-target", type=int, default=256, help="GPU-memory fit target; zero disables fitting")
    parser.add_argument("--gpu-layers", type=int)
    parser.add_argument("--cpu-moe-layers", type=int)
    parser.add_argument("--cache-first", action="store_true", help="run cache before control in each pair")
    parser.add_argument("--pinned-experts", type=Path, help="static pinned-expert manifest")
    parser.add_argument(
        "--route-ready-sidecar",
        action="store_true",
        help="run the fixed Compact route-ready sidecar measurement matrix",
    )
    return parser.parse_args()


def bench_command(args: argparse.Namespace, cache_enabled: bool) -> list[str]:
    command = [
        str(args.bench),
        "-m", str(args.model),
        "-p", "0",
        "-n", str(args.n_gen),
        "-r", "1",
        "-t", "14",
        "-b", "4096",
        "-ub", "2048",
        "-ctk", "q8_0",
        "-ctv", "q8_0",
        "-fa", "on",
        "-lm", args.load_mode,
    ]
    if cache_enabled and args.pinned_experts is not None:
        command.extend(("-pe", str(args.pinned_experts)))
    if args.fit_target:
        command.extend(("-fitt", str(args.fit_target)))
    if args.gpu_layers is not None:
        command.extend(("-ngl", str(args.gpu_layers)))
    if args.cpu_moe_layers is not None:
        command.extend(("-ncmoe", str(args.cpu_moe_layers)))
    command.extend([
        "-exc", str(args.cache_mib if cache_enabled else 0),
        "-o", "jsonl",
    ])
    if cache_enabled:
        command.extend(("-excp", str(args.cache_period), "-excm", str(args.max_swaps)))
    return command


def run_order(cache_first: bool) -> tuple[bool, bool]:
    return (True, False) if cache_first else (False, True)


def main() -> int:
    args = parse_args()
    if args.route_ready_sidecar:
        if args.runs != 5:
            raise ValueError("--route-ready-sidecar requires --runs 5")
        if args.cache_mib != 128:
            raise ValueError("--route-ready-sidecar requires --cache-mib 128")
        if args.cache_period != 256:
            raise ValueError("--route-ready-sidecar requires --cache-period 256")

        if args.output_dir == Path("tools/results/expert-cache"):
            args.output_dir /= "route-ready-sidecar"

    if args.runs < 1:
        raise ValueError("--runs must be positive")
    if args.n_gen < 1:
        raise ValueError("--n-gen must be positive")
    if args.fit_target < 0:
        raise ValueError("--fit-target must not be negative")
    if args.max_swaps < -1:
        raise ValueError("--max-swaps must be at least -1")
    if not args.bench.is_file():
        raise FileNotFoundError(f"benchmark executable not found: {args.bench}")
    if not args.model.is_file():
        raise FileNotFoundError(f"model not found: {args.model}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or datetime.now(UTC).strftime("%Y-%m-%d-tg-matrix")
    environment = os.environ | {"GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL": "0"}

    for run in range(1, args.runs + 1):
        for cache_enabled in run_order(args.cache_first):
            label = "cache" if cache_enabled else "control"
            output_path = output_dir / f"{prefix}-{label}-{run}.jsonl"
            command = bench_command(args, cache_enabled)
            print(f"running {label} {run}/{args.runs}: {output_path}", file=sys.stderr)
            completed = subprocess.run(command, capture_output=True, text=True, env=environment)
            output_path.write_text(completed.stdout, encoding="utf-8", newline="\n")
            if completed.returncode:
                print(completed.stderr, file=sys.stderr, end="")
                return completed.returncode
            print(next(line for line in completed.stdout.splitlines() if line.startswith("{")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
