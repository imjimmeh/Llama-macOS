#!/usr/bin/env python3
"""
benchmark_mtp_suite.py - Empirical performance profiling and comparative benchmark suite
for llama.cpp MTP Dynamic Offload, Expert Cache, and Baseline inference.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, desc=""):
    print(f"[*] Running: {desc or ' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    return res.returncode, res.stdout, dt


def main():
    parser = argparse.ArgumentParser(description="llama.cpp MTP Comparative Benchmark Harness")
    parser.add_argument("-m", "--model", default=r"G:\ai\models\Qwen3.6-35B-A3B-APEX-MTP-Quality.gguf", help="Path to GGUF model")
    parser.add_argument("-p", "--prompt-len", type=int, default=64, help="Prompt token length")
    parser.add_argument("-n", "--gen-len", type=int, default=16, help="Generation token length")
    parser.add_argument("-ngl", "--gpu-layers", type=int, default=12, help="Number of GPU layers to offload")
    parser.add_argument("-exc", "--expert-cache-mb", type=int, default=1024, help="Expert cache size in MiB")
    parser.add_argument("-t", "--threads", type=int, default=14, help="Number of CPU threads")
    parser.add_argument("-o", "--output-json", default="benchmark_mtp_results.json", help="Output JSON results file")

    args = parser.parse_args()

    bin_dir = Path(r"G:\code\AI\llamacpptuned\llama.cpp\build\bin\Release")
    bench_exe = bin_dir / "test-benchmark-mtp.exe"

    if not bench_exe.exists():
        print(f"Error: {bench_exe} not found. Please compile test-benchmark-mtp.")
        sys.exit(1)

    print("==========================================================================")
    print("         llama.cpp MTP Comparative Performance Benchmark Suite             ")
    print("==========================================================================")
    print(f"Model:        {args.model}")
    print(f"Prompt Len:   {args.prompt_len}")
    print(f"Gen Len:      {args.gen_len}")
    print(f"GPU Layers:   {args.gpu_layers}")
    print(f"Expert Cache: {args.expert_cache_mb} MiB")
    print(f"Threads:      {args.threads}")
    print("==========================================================================")

    cmd = [
        str(bench_exe),
        "-m", args.model,
        "-p", str(args.prompt_len),
        "-n", str(args.gen_len),
        "-ngl", str(args.gpu_layers),
        "-exc", str(args.expert_cache_mb),
        "-t", str(args.threads),
    ]

    rc, out, dt = run_command(cmd, desc="Comparative C++ Benchmark Execution")
    print(out)

    if rc != 0:
        print(f"Benchmark failed with return code {rc}")
        sys.exit(rc)


if __name__ == "__main__":
    main()
