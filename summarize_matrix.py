#!/usr/bin/env python3
"""Matrix summariser: one statistic, deltas computed from the same data.

Usage: python summarize_matrix.py <dir> [--stat median|mean]
Prints per-matrix stats table plus pairwise deltas derived from the same
per-run values that produced the displayed statistic.
"""
import argparse
import csv
import os
import statistics
import sys

COUNTER_KEYS = [
    "avg_ts",
    "expert_cache_requests",
    "expert_cache_zero_copy_hits",
    "expert_cache_gpu_slot_executions",
    "expert_cache_cpu_fallbacks",
    "expert_cache_used_ready",
    "expert_cache_used_in_flight",
    "expert_cache_used_miss",
    "expert_cache_already_resident",
    "expert_cache_wasted_prefetch_bytes",
    "expert_cache_in_flight_wait_us",
    "expert_cache_eligible_ops",
    "expert_cache_mul_mat_id_inputs",
    "routing_predictor_predictions_generated",
    "routing_predictor_predictions_used",
]

FLOAT_KEYS = {"avg_ts"}


def gen_row(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    for r in rows[1:]:
        if len(r) == len(hdr) and r[hdr.index("n_gen")] not in ("0", ""):
            return dict(zip(hdr, r))
    return None


def load_matrix(directory, label, n):
    recs = []
    for i in range(n):
        p = os.path.join(directory, f"{label}_{i}.csv")
        if not os.path.exists(p):
            continue
        r = gen_row(p)
        if r:
            recs.append(r)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--labels", default="A,B,C")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--stat", choices=["median", "mean"], default="median")
    ap.add_argument(
        "--deltas", default="C-A,C-B,B-A",
        help="comma-separated pairs like X-Y to report")
    args = ap.parse_args()

    stat_fn = statistics.median if args.stat == "median" else statistics.mean
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    delta_pairs = []
    for d in args.deltas.split(","):
        d = d.strip()
        if "-" in d:
            a, b = d.split("-", 1)
            delta_pairs.append((a.strip(), b.strip()))

    # Collect raw values once; derive every reported number from them.
    values = {}  # label -> key -> list
    for label in labels:
        recs = load_matrix(args.directory, label, args.runs)
        for k in COUNTER_KEYS:
            vals = []
            for r in recs:
                v = float(r[k]) if k in FLOAT_KEYS else int(r[k])
                vals.append(v)
            values[(label, k)] = vals

    def fmt(k, v):
        return f"{v:10.3f}" if k in FLOAT_KEYS else f"{int(round(v)):10d}"

    print(f"== {args.directory}  statistic={args.stat}  runs={args.runs} ==")
    for k in COUNTER_KEYS:
        cells = " ".join(
            f"{label}:{fmt(k, stat_fn(values[(label, k)]))}" for label in labels)
        print(f"  {k:48s} {cells}")
    if delta_pairs:
        print(f"  deltas ({args.stat} of same data):")
        for a, b in delta_pairs:
            parts = []
            for k in COUNTER_KEYS:
                d = stat_fn(values[(a, k)]) - stat_fn(values[(b, k)])
                sign = "+" if d >= 0 else "-"
                if k in FLOAT_KEYS:
                    parts.append(f"ts={sign}{abs(d):.3f}" if k == "avg_ts"
                                 else f"{k}={sign}{abs(d):.1f}")
                else:
                    parts.append(f"{k}={sign}{abs(int(round(d)))}")
            print(f"    {a}-{b}: {' '.join(parts)}")


if __name__ == "__main__":
    main()
