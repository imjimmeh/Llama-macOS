#!/usr/bin/env python3
"""Run repeated SPEED-Bench prompts against a live llama-server instance."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, default=Path("build/bin/Release/llama-server.exe"))
    parser.add_argument("--model", type=Path, default=Path("C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf"))
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--bench", default="qualitative")
    parser.add_argument("--category", default="coding")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--osl", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=Path("tools/results/expert-cache/server-speed-bench"))
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--cache-mib", type=int, default=0)
    parser.add_argument("--cache-period", type=int, default=32)
    parser.add_argument("--pinned-experts", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--ubatch-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--ctx-size", type=int, default=128000)
    parser.add_argument("--fit-target", type=int, default=256)
    parser.add_argument("--sps", type=float, default=0.0)
    parser.add_argument("--spec-type", default=None)
    parser.add_argument("--spec-match", type=int, default=24)
    parser.add_argument("--spec-min", type=int, default=24)
    parser.add_argument("--spec-max", type=int, default=48)
    return parser.parse_args()


def wait_for_server(port: int, timeout_s: float = 120.0) -> bool:
    start = time.time()
    url = f"http://localhost:{port}/health"
    print(f"Waiting for llama-server on port {port}...", file=sys.stderr)
    while time.time() - start < timeout_s:
        try:
            resp = requests.get(url, timeout=2.0)
            if resp.status_code == 200:
                print(f"llama-server is ready (status=200)", file=sys.stderr)
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def build_server_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(args.server),
        "-m", str(args.model),
        "--port", str(args.port),
        "-c", str(args.ctx_size),
        "-np", "1",
        "-t", str(args.threads),
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "-ctk", "q8_0",
        "-ctv", "q8_0",
        "-fa", "on",
        "-lm", "mmap",
        "-fitt", str(args.fit_target),
        "--jinja",
        "-sps", str(args.sps),
        "--no-context-shift",
    ]
    if args.pinned_experts is not None:
        cmd.extend(["-pe", str(args.pinned_experts)])
    if args.cache_mib > 0:
        cmd.extend([
            "-exc", f"{args.cache_mib}M",
            "-excp", str(args.cache_period),
            "-excs",
        ])
    if args.spec_type:
        cmd.extend([
            "--spec-type", args.spec_type,
            "--spec-ngram-mod-n-match", str(args.spec_match),
            "--spec-ngram-mod-n-min", str(args.spec_min),
            "--spec-ngram-mod-n-max", str(args.spec_max),
        ])
    return cmd


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL"] = "0"

    server_cmd = build_server_command(args)
    print("Launching server command:", " ".join(server_cmd), file=sys.stderr)

    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
    )

    try:
        if not wait_for_server(args.port, timeout_s=180.0):
            print("Server failed to become ready in time", file=sys.stderr)
            server_proc.kill()
            out, err = server_proc.communicate()
            print("Server STDERR:\n", err, file=sys.stderr)
            return 1

        results_summary: list[dict[str, Any]] = []

        for i in range(1, args.iterations + 1):
            out_json = args.output_dir / f"{args.prefix}-pass-{i}.json"
            bench_cmd = [
                sys.executable, "tools/server/bench/speed-bench/speed_bench.py",
                "--url", f"localhost:{args.port}",
                "--bench", args.bench,
                "--category", args.category,
                "--limit", str(args.limit),
                "--osl", str(args.osl),
                "--concurrency", "1",
                "--output", str(out_json),
            ]
            print(f"\n>>> Running iteration {i}/{args.iterations}: {out_json}", file=sys.stderr)
            res = subprocess.run(bench_cmd, capture_output=True, text=True)
            print(res.stdout)
            if res.returncode != 0:
                print(f"speed_bench failed on pass {i}:", res.stderr, file=sys.stderr)
                break

            if out_json.exists():
                data = json.loads(out_json.read_text(encoding="utf-8"))
                for r in data.get("results", []):
                    results_summary.append({
                        "pass": i,
                        "id": r.get("id"),
                        "category": r.get("category"),
                        "predicted_per_second": r.get("predicted_per_second"),
                        "prompt_per_second": r.get("prompt_per_second"),
                        "completion_tokens": r.get("completion_tokens"),
                        "latency_s": r.get("latency_s"),
                    })

    finally:
        print("\nStopping llama-server...", file=sys.stderr)
        server_proc.terminate()
        try:
            out, err = server_proc.communicate(timeout=15.0)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            out, err = server_proc.communicate()

        log_path = args.output_dir / f"{args.prefix}-server.log"
        log_path.write_text(f"=== STDOUT ===\n{out}\n=== STDERR ===\n{err}\n", encoding="utf-8")
        print(f"Server log written to {log_path}", file=sys.stderr)

        # Print relevant stderr telemetry
        for line in err.splitlines():
            if any(k in line for k in ["Route-Ready", "expert_cache", "Admission", "Histogram", "Zero-Copy", "avoided"]):
                print(line)

    summary_path = args.output_dir / f"{args.prefix}-summary.json"
    summary_path.write_text(json.dumps(results_summary, indent=2), encoding="utf-8")
    print(f"\nRun summary written to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
