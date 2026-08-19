#!/usr/bin/env python3
"""
benchmark_grid.py - Systematically benchmarks llama.cpp across EXC and FFN-Split grid configurations.
Compares upstream baseline against llamacpptuned.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_benchmark_command(binary_path, args_list, desc=""):
    cmd = [str(binary_path)] + args_list
    print(f"\n=======================================================")
    print(f"[*] {desc}")
    print(f"[*] Command: {' '.join(cmd)}")
    print(f"=======================================================")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        dt = time.time() - t0
        print(proc.stdout)
        return proc.returncode, proc.stdout, dt
    except Exception as e:
        print(f"Error executing command: {e}")
        return -1, str(e), 0.0


def parse_llama_bench_output(output):
    """
    Parses markdown table output from llama-bench.
    Returns dict with pp and tg throughput in t/s.
    """
    results = {}
    lines = output.splitlines()
    for line in lines:
        if "pp" in line and "|" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 2:
                test_name = parts[-2]
                val_str = parts[-1].split("±")[0].strip()
                try:
                    results["pp_tps"] = float(val_str)
                    results["pp_test"] = test_name
                except ValueError:
                    pass
        elif "tg" in line and "|" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 2:
                test_name = parts[-2]
                val_str = parts[-1].split("±")[0].strip()
                try:
                    results["tg_tps"] = float(val_str)
                    results["tg_test"] = test_name
                except ValueError:
                    pass
    return results


def main():
    parser = argparse.ArgumentParser(description="Grid Benchmark Harness for llama-bench")
    parser.add_argument("-m", "--model", default=r"C:\Users\jimme\.lmstudio\models\mudler\Qwen3.6-35B-A3B-APEX-GGUF\Qwen3.6-35B-A3B-APEX-Compact.gguf", help="Model path")
    parser.add_argument("-p", "--n-prompt", type=int, default=4096, help="Prompt token count")
    parser.add_argument("-n", "--n-gen", type=int, default=1024, help="Generation token count")
    parser.add_argument("-t", "--threads", type=int, default=14, help="Thread count")
    parser.add_argument("-fitt", "--fit-target", type=int, default=512, help="Fit target VRAM margin in MiB")
    parser.add_argument("-ctk", "--cache-type-k", default="q5_1", help="K cache type")
    parser.add_argument("-ctv", "--cache-type-v", default="q5_1", help="V cache type")
    parser.add_argument("-r", "--repetitions", type=int, default=1, help="Repetitions per test")
    parser.add_argument("-o", "--output-json", default="benchmark_grid_results.json", help="Output JSON path")
    parser.add_argument("--quick", action="store_true", help="Quick mode (p=512, n=128)")

    args = parser.parse_args()

    if args.quick:
        args.n_prompt = 512
        args.n_gen = 128

    upstream_bench = Path(r"G:\code\AI\llama.cpp\build\bin\Release\llama-bench.exe")
    tuned_bench = Path(r"G:\code\AI\llamacpptuned\llama.cpp\build\bin\Release\llama-bench.exe")

    exc_values = [0, 128, 256, 512, 1024]
    ffn_split_values = [0.0, 0.2, 0.3, 0.35, 0.4, 0.5]

    all_results = {
        "model": args.model,
        "n_prompt": args.n_prompt,
        "n_gen": args.n_gen,
        "threads": args.threads,
        "upstream": {},
        "grid": []
    }

    # 1. Benchmark Upstream Baseline
    if upstream_bench.exists():
        print("\n>>> Running Upstream Baseline Benchmark <<<")
        up_args = [
            "-m", args.model,
            "-p", str(args.n_prompt),
            "-n", str(args.n_gen),
            "-t", str(args.threads),
            "-fitt", str(args.fit_target),
            "-fa", "on",
            "-ctk", args.cache_type_k,
            "-ctv", args.cache_type_v,
            "-r", str(args.repetitions),
            "-o", "md"
        ]
        rc, out, dt = run_benchmark_command(upstream_bench, up_args, "Upstream llama.cpp Baseline")
        parsed = parse_llama_bench_output(out)
        all_results["upstream"] = {
            "returncode": rc,
            "duration_s": dt,
            "metrics": parsed
        }
    else:
        print(f"Warning: Upstream binary {upstream_bench} not found.")

    # 2. Benchmark Grid on llamacpptuned
    print("\n>>> Running llamacpptuned Grid Benchmark <<<")
    for exc in exc_values:
        for ffn in ffn_split_values:
            desc = f"Tuned: EXC={exc}M, FFN-Split={ffn}"
            tuned_args = [
                "-m", args.model,
                "-p", str(args.n_prompt),
                "-n", str(args.n_gen),
                "-t", str(args.threads),
                "-fitt", str(args.fit_target),
                "-fa", "on",
                "-ctk", args.cache_type_k,
                "-ctv", args.cache_type_v,
                "-exc", str(exc),
                "--ffn-split", str(ffn),
                "-r", str(args.repetitions),
                "-o", "md"
            ]
            rc, out, dt = run_benchmark_command(tuned_bench, tuned_args, desc)
            parsed = parse_llama_bench_output(out)
            grid_entry = {
                "exc_mib": exc,
                "ffn_split": ffn,
                "returncode": rc,
                "duration_s": dt,
                "metrics": parsed
            }
            all_results["grid"].append(grid_entry)

            # Save progress incrementally
            with open(args.output_json, "w") as f:
                json.dump(all_results, f, indent=2)

    # 3. Print Final Markdown Comparison Table
    print("\n\n" + "=" * 80)
    print("                      FINAL GRID BENCHMARK SUMMARY TABLE                       ")
    print("=" * 80)
    up_pp = all_results.get("upstream", {}).get("metrics", {}).get("pp_tps", 0.0)
    up_tg = all_results.get("upstream", {}).get("metrics", {}).get("tg_tps", 0.0)
    print(f"Upstream Reference: PP = {up_pp:.2f} t/s | TG = {up_tg:.2f} t/s\n")

    print("| EXC (MiB) | FFN-Split | PP Speed (t/s) | TG Speed (t/s) | PP vs Upstream | TG vs Upstream |")
    print("|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------------:|")

    for g in all_results["grid"]:
        exc = g["exc_mib"]
        ffn = g["ffn_split"]
        pp = g.get("metrics", {}).get("pp_tps", 0.0)
        tg = g.get("metrics", {}).get("tg_tps", 0.0)
        pp_rel = f"{(pp / up_pp * 100.0):.1f}%" if up_pp > 0 else "N/A"
        tg_rel = f"{(tg / up_tg * 100.0):.1f}%" if up_tg > 0 else "N/A"
        print(f"| {exc:9d} | {ffn:9.2f} | {pp:14.2f} | {tg:14.2f} | {pp_rel:14s} | {tg_rel:14s} |")
    print("=" * 80)


if __name__ == "__main__":
    main()
