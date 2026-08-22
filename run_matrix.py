#!/usr/bin/env python3
"""Matrix A/B/C alternating benchmark rounds, gen t/s + 5H counters."""
import os, subprocess, csv, time

MODEL = r"C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf"
PRED = r"tools/training_data/model.bin"
EXE = r"./build/bin/Release/llama-bench.exe"
OUT_DIR = r"g:/tmp/matrix"
N = 5

def base():
    return ["-m", MODEL, "-p", "32", "-n", "64",
            "-fitt", "256", "-exc", "256", "-excp", "64",
            "--routing-predictor-stats",
            "-r", "1", "-o", "csv"]

def run(label, extra_argv, env_extra, idx):
    env = os.environ.copy()
    env["GGML_OP_OFFLOAD_MIN_BATCH"] = "1"
    for k, v in env_extra.items():
        env[k] = v
    out_csv = os.path.join(OUT_DIR, f"{label}_{idx}.csv")
    cmd = [EXE] + base() + extra_argv
    t0 = time.time()
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, env=env, check=False)
    print(f"{label}#{idx} -> {out_csv} ({time.time()-t0:.1f}s)")

def matrix_A(i):
    # baseline: cache present, no predictor model
    return [], {}

def matrix_B(i):
    # full predictor + cache, but force CPU to skip swap
    return ["--routing-predictor-model", PRED,
            "--routing-predictor-variant", "low-rank-mlp"], \
           {"GGML_EXPERT_EXEC_FORCE_CPU": "1"}

def matrix_C(i):
    # full predictor + cache + GPU slot swap
    return ["--routing-predictor-model", PRED,
            "--routing-predictor-variant", "low-rank-mlp"], \
           {}

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for i in range(N):
        for label, fn in [("A", matrix_A), ("B", matrix_B), ("C", matrix_C)]:
            argv, env = fn(i)
            run(label, argv, env, i)

if __name__ == "__main__":
    main()
