#!/usr/bin/env python3
"""Run alternating cache-off/cache-on TG measurements and retain raw JSONL."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--matrix", action="store_true",
                        help="run A/B/C three-way matrix (A=-exc 0, B=reserved empty, C=static manifest)")
    parser.add_argument("--manifest", type=Path,
                        help="v3 manifest for the C row (implies --matrix)")
    parser.add_argument("--placement-report", type=Path,
                        help="geometry/placement JSON emitted per capacity before the matrix")
    parser.add_argument("--geometry-tool", type=Path,
                        help="test-moe-geometry-report executable used for the placement report")
    parser.add_argument(
        "--route-ready-sidecar",
        action="store_true",
        help="run the fixed Compact route-ready sidecar measurement matrix",
    )
    parser.add_argument("--hetero-concurrent", choices=(0, 1), type=int, default=0)
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


def matrix_command(args: argparse.Namespace, config: str) -> list[str]:
    """Build the llama-bench command for one A/B/C matrix row.

    A: cache off. B: cache reserved but no useful residency (-excm 0, no
    manifest). C: same reservation plus a static v3 manifest. Only -exc/-pe
    and the swap knobs differ; every other flag is held identical."""
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
    if args.fit_target:
        command.extend(("-fitt", str(args.fit_target)))
    if args.gpu_layers is not None:
        command.extend(("-ngl", str(args.gpu_layers)))
    if args.cpu_moe_layers is not None:
        command.extend(("-ncmoe", str(args.cpu_moe_layers)))
    if config == "A":
        command.extend(["-exc", "0", "-o", "jsonl"])
        return command
    command.extend(["-exc", str(args.cache_mib), "-o", "jsonl"])
    command.extend(("-excp", str(args.cache_period), "-excm", "0"))
    if config == "C":
        command.extend(("-pe", str(args.pinned_experts)))
    return command


def matrix_order(pair: int, cache_first: bool) -> list[str]:
    """Alternate the A/B/C traversal direction per pair so order bias cancels.
    Odd pairs go A->B->C, even pairs go C->B->A (or the reverse when
    --cache-first is set)."""
    forward = pair % 2 == 1
    if cache_first:
        forward = not forward
    return ["A", "B", "C"] if forward else ["C", "B", "A"]


def order_label(pair: int, cache_first: bool) -> str:
    return "control_first" if matrix_order(pair, cache_first)[0] == "A" else "static_first"


def build_matrix_index(capacity_mib: int, rows: list[dict]) -> dict:
    return {
        "capacity_mib": capacity_mib,
        "runs": [
            {
                "config": row["config"],
                "run": row["pair"],
                "order": row["order"],
                "jsonl": row["stdout"],
                "stderr": row["stderr"],
            }
            for row in rows
        ],
    }


def bench_environment(args: argparse.Namespace) -> dict[str, str]:
    return os.environ | {
        "GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL": "0",
        "GGML_EXPERT_CACHE_HETERO_CONCURRENT": str(args.hetero_concurrent),
    }


def run_matrix(args, output_dir, prefix, environment):
    """Execute the A/B/C three-way matrix, writing one JSONL/stderr per
    child process plus a matrix index JSON."""
    placement_path = None
    if args.geometry_tool and args.geometry_tool.is_file():
        placement_path = output_dir / f"{prefix}-placement.json"
        geom_cmd = [str(args.geometry_tool), "--placement-json",
                    "-m", str(args.model), "-fitt", str(args.fit_target)]
        print(f"placement report: {placement_path}", file=sys.stderr)
        g = subprocess.run(geom_cmd, capture_output=True, text=True, env=environment)
        placement_path.write_text(g.stdout, encoding="utf-8")
        if g.returncode:
            print(g.stderr, file=sys.stderr, end="")
            placement_path = None
    rows = []
    for pair in range(1, args.runs + 1):
        order = matrix_order(pair, args.cache_first)
        for idx, config in enumerate(order):
            tag = f"p{pair}-{config}"
            stdout_path = output_dir / f"{prefix}-{tag}.jsonl"
            stderr_path = output_dir / f"{prefix}-{tag}.stderr"
            command = matrix_command(args, config)
            print(f"matrix {tag} ({idx+1}/3): {stdout_path}", file=sys.stderr)
            completed = subprocess.run(command, capture_output=True, text=True, env=environment)
            stdout_path.write_text(completed.stdout, encoding="utf-8", newline="\n")
            stderr_path.write_text(completed.stderr, encoding="utf-8", errors="replace")
            if completed.returncode:
                print(completed.stderr, file=sys.stderr, end="")
                return completed.returncode
            json_line = next(line for line in completed.stdout.splitlines() if line.startswith("{"))
            print(json_line)
            rows.append({
                "config": config,
                "pair": pair,
                "order": order_label(pair, args.cache_first),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            })
    index = build_matrix_index(args.cache_mib, rows)
    if placement_path:
        index["placement_report"] = str(placement_path)
    index_path = output_dir / f"{prefix}.index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"matrix index: {index_path}", file=sys.stderr)
    return 0

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
    environment = bench_environment(args)
    prefix = args.prefix or datetime.now(UTC).strftime("%Y-%m-%d-tg-matrix")

    if args.matrix or args.manifest is not None:
        if args.manifest is not None:
            args.pinned_experts = args.manifest
            args.matrix = True
        if args.pinned_experts is None:
            raise ValueError("--matrix requires --manifest or --pinned-experts")
        if args.max_swaps not in (0, -1):
            raise ValueError("--matrix forces --max-swaps 0 for B and C rows")
        return run_matrix(args, output_dir, prefix, environment)

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
