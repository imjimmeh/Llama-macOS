import subprocess
import json
import math
import sys
import os

EXE_PATH = os.path.abspath("build/bin/Release/test-moe-heterogeneous-bench.exe")
MODEL_PATH = "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf"

TIERS = [
    {"name": "Control (0 MiB)", "cache_mb": 0, "manifest": ""},
    {"name": "Pinned 64 MiB", "cache_mb": 64, "manifest": "pinned_experts_64mb.json"},
    {"name": "Pinned 128 MiB", "cache_mb": 128, "manifest": "pinned_experts_128mb.json"},
    {"name": "Pinned 256 MiB", "cache_mb": 256, "manifest": "pinned_experts_256mb.json"},
    {"name": "Pinned 512 MiB", "cache_mb": 512, "manifest": "pinned_experts_512mb.json"},
    {"name": "Pinned 1024 MiB", "cache_mb": 1024, "manifest": "pinned_experts_1024mb.json"},
]

def run_single(tier_name, cache_mb, manifest, n_prompt=16, n_gen=64, isolate_dense=True):
    cmd = [
        EXE_PATH,
        "-m", MODEL_PATH,
        "-t", "14",
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "--tier-name", tier_name,
        "--cache-mb", str(cache_mb),
        "--json"
    ]
    if manifest:
        cmd.extend(["--manifest", manifest])
    if not isolate_dense:
        cmd.append("--fit")

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    out = proc.stdout
    if "__JSON_RESULT_START__" not in out:
        print(f"Error in tier {tier_name}: {proc.stderr}\n{out}", file=sys.stderr)
        return None

    json_str = out.split("__JSON_RESULT_START__")[1].split("__JSON_RESULT_END__")[0].strip()
    return json.loads(json_str)

def compute_stats(values):
    n = len(values)
    if n == 0:
        return {"mean": 0, "median": 0, "stddev": 0, "ci95": 0, "p95": 0, "min": 0, "max": 0}
    s_val = sorted(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    stddev = math.sqrt(variance)
    ci95 = 1.96 * (stddev / math.sqrt(n))
    median = s_val[n // 2] if n % 2 == 1 else (s_val[n // 2 - 1] + s_val[n // 2]) / 2.0
    p95_idx = min(int(math.ceil(0.95 * n)) - 1, n - 1)
    p95 = s_val[p95_idx]
    return {
        "mean": mean,
        "median": median,
        "stddev": stddev,
        "ci95": ci95,
        "p95": p95,
        "min": s_val[0],
        "max": s_val[-1]
    }

def main():
    num_pairs = 5 if len(sys.argv) < 2 else int(sys.argv[1])
    print(f"=== Epic 14: Statistical Multi-Tier Deployment Benchmark ({num_pairs} A/B Pairs per Tier) ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Protocol: Fresh process per trial, alternating A/B sequence\n")

    results_by_tier = {t["name"]: [] for t in TIERS}

    for tier_idx in range(1, len(TIERS)):
        tier = TIERS[tier_idx]
        control = TIERS[0]
        print(f"--- Running A/B Trials: Control vs {tier['name']} ({num_pairs} pairs) ---")
        
        for pair_idx in range(num_pairs):
            print(f"  Pair {pair_idx+1:02d}/{num_pairs:02d}: ", end="", flush=True)
            res_ctrl = run_single(control["name"], control["cache_mb"], control["manifest"])
            if res_ctrl:
                results_by_tier[control["name"]].append(res_ctrl["tg_tps"])
                print(f"[Ctrl: {res_ctrl['tg_tps']:.2f} t/s] ", end="", flush=True)

            res_tier = run_single(tier["name"], tier["cache_mb"], tier["manifest"])
            if res_tier:
                results_by_tier[tier["name"]].append(res_tier["tg_tps"])
                print(f"[{tier['name']}: {res_tier['tg_tps']:.2f} t/s] ", flush=True)

    print("\n" + "=" * 110)
    print(f"{'STATISTICAL DEPLOYMENT SUMMARY (N=' + str(num_pairs) + ' Pairs)':^110}")
    print("=" * 110)
    print(f"| {'Tier':<22} | {'Mean (t/s)':<12} | {'Median':<10} | {'Stddev':<10} | {'95% CI':<14} | {'P95 (t/s)':<10} | {'Speedup':<10} |")
    print("|" + "-" * 24 + "|" + "-" * 14 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 16 + "|" + "-" * 12 + "|" + "-" * 12 + "|")

    ctrl_stats = compute_stats(results_by_tier[TIERS[0]["name"]])
    ctrl_mean = ctrl_stats["mean"] if ctrl_stats["mean"] > 0 else 1.0

    print(f"| {TIERS[0]['name']:<22} | {ctrl_stats['mean']:>10.3f}   | {ctrl_stats['median']:>8.3f}   | {ctrl_stats['stddev']:>8.3f}   | +- {ctrl_stats['ci95']:>6.3f}     | {ctrl_stats['p95']:>8.3f}   | {'1.000x':>10} |")

    for tier in TIERS[1:]:
        st = compute_stats(results_by_tier[tier["name"]])
        speedup = st["mean"] / ctrl_mean
        print(f"| {tier['name']:<22} | {st['mean']:>10.3f}   | {st['median']:>8.3f}   | {st['stddev']:>8.3f}   | +- {st['ci95']:>6.3f}     | {st['p95']:>8.3f}   | {speedup:>9.3f}x |")

    print("=" * 110 + "\n")

if __name__ == "__main__":
    main()
