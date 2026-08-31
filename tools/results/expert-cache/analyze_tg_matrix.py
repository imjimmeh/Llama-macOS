#!/usr/bin/env python3
"""Prefix-generic paired analysis for A/B/C TG matrices.

A matrix index (written by run-tg-matrix.py --matrix) lists one record per
child process:

    {"capacity_mib": 128,
     "runs": [{"config": "A"|"B"|"C", "run": 1, "order": "control_first",
               "jsonl": "...", "stderr": "..."}]}

Configs:
    A - cache off              (-exc 0)
    B - reserved empty         (-exc <cap> -excm 0, no manifest)
    C - static oracle manifest (-exc <cap> -excm 0 -pe manifest)

Attribution (never report C/A as a cache execution gain):
    placement cost     = B / A - 1
    execution benefit  = C / B - 1
    net benefit        = C / A - 1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

BASELINE = "A"
RESERVED = "B"
STATIC = "C"

FULL_HITS = "expert_cache_route_ready_full_hits"
FALLBACKS = "expert_cache_route_ready_fallbacks"
FAST_REJECTS = "expert_cache_route_ready_fast_rejects"
ROUTE_ID_US = "expert_cache_route_ready_route_id_us"
TIMED_H2D = "expert_cache_bytes_ram_to_gpu"
BYTES_AVOIDED = "expert_cache_bytes_avoided"
RESIDENT_HIST_KEYS = [f"expert_cache_route_ready_resident_bundles_{k}" for k in range(9)]


def tg_rate(record: dict) -> float:
    """Tokens/second decode rate from one llama-bench JSONL record."""
    if record.get("avg_ts") is not None:
        return float(record["avg_ts"])
    avg_ns = float(record["avg_ns"])
    n_gen = int(record["n_gen"])
    return n_gen / (avg_ns / 1e9)


def load_matrix(index_path: Path) -> dict:
    """Load a matrix index and its per-run JSONL/stderr payloads."""
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    runs = []
    for entry in index["runs"]:
        record = json.loads(Path(entry["jsonl"]).read_text(encoding="utf-8").splitlines()[0])
        stderr_path = entry.get("stderr")
        stderr = ""
        if stderr_path and Path(stderr_path).is_file():
            stderr = Path(stderr_path).read_text(encoding="utf-8", errors="replace")
        runs.append({
            "config": entry["config"],
            "run": int(entry["run"]),
            "order": entry.get("order", "control_first"),
            "record": record,
            "stderr": stderr,
        })
    return {"capacity_mib": index.get("capacity_mib"), "runs": runs}


def by_config(matrix: dict) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for entry in matrix["runs"]:
        out[entry["config"]][entry["run"]] = entry
    return dict(out)


def paired_deltas(matrix: dict, treatment: str, control: str) -> list[float]:
    """Percent delta (treatment vs control) for every shared run index."""
    configs = by_config(matrix)
    if treatment not in configs or control not in configs:
        return []
    common = sorted(set(configs[treatment]) & set(configs[control]))
    return [
        (tg_rate(configs[treatment][r]["record"]) / tg_rate(configs[control][r]["record"]) - 1.0)
        * 100.0
        for r in common
    ]


def _mean_or_none(values):
    return sum(values) / len(values) if values else None


def compare(matrix: dict, treatment: str, control: str) -> dict | None:
    deltas = paired_deltas(matrix, treatment, control)
    if not deltas:
        return None
    configs = by_config(matrix)
    n = len(deltas)
    mean = statistics.fmean(deltas)
    ci_low = ci_high = None
    stdev = None
    if n >= 2:
        stdev = statistics.stdev(deltas)
        if stdev > 0:
            lo, hi = stats.t.interval(0.95, n - 1, loc=mean, scale=stats.sem(deltas))
            ci_low, ci_high = float(lo), float(hi)
    def rates(config):
        return [tg_rate(e["record"]) for e in configs.get(config, {}).values()]
    def telem(config, field):
        return [e["record"].get(field, 0) for e in configs.get(config, {}).values()]
    return {
        "treatment": treatment,
        "control": control,
        "n_pairs": n,
        "deltas_pct": deltas,
        "mean_delta_pct": mean,
        "median_delta_pct": statistics.median(deltas),
        "stdev_delta_pct": stdev,
        "positive_pairs": sum(1 for d in deltas if d > 0),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "treatment_mean_tg": _mean_or_none(rates(treatment)),
        "control_mean_tg": _mean_or_none(rates(control)),
        "full_hits_mean": _mean_or_none(telem(treatment, FULL_HITS)),
        "fallback_calls_mean": _mean_or_none(telem(treatment, FALLBACKS)),
        "fast_rejects_mean": _mean_or_none(telem(treatment, FAST_REJECTS)),
        "route_id_us_mean": _mean_or_none(telem(treatment, ROUTE_ID_US)),
        "timed_h2d_bytes_mean": _mean_or_none(telem(treatment, TIMED_H2D)),
        "bytes_avoided_mean": _mean_or_none(telem(treatment, BYTES_AVOIDED)),
    }


def placement_attribution(matrix: dict) -> dict:
    configs = by_config(matrix)
    def mean_rate(config):
        runs = configs.get(config, {})
        return statistics.fmean([tg_rate(e["record"]) for e in runs.values()]) if runs else None
    a = mean_rate(BASELINE)
    b = mean_rate(RESERVED)
    c = mean_rate(STATIC)
    return {
        "baseline_tg": a,
        "reserved_empty_tg": b,
        "static_tg": c,
        "placement_cost_pct": (b / a - 1.0) * 100.0 if a and b else None,
        "execution_benefit_pct": (c / b - 1.0) * 100.0 if b and c else None,
        "net_benefit_pct": (c / a - 1.0) * 100.0 if a and c else None,
    }


def resident_histogram(matrix: dict, config: str = STATIC) -> dict:
    configs = by_config(matrix)
    hist = defaultdict(int)
    for entry in configs.get(config, {}).values():
        record = entry["record"]
        for key in RESIDENT_HIST_KEYS:
            value = int(record.get(key, 0) or 0)
            if value:
                hist[key.rsplit("_", 1)[1]] += value
    return dict(hist)


def gate_checks(matrix: dict) -> dict:
    """Evidence for Gate 2: nonzero static full hits, zero timed H2D,
    and reserved-empty carrying no useful residency."""
    configs = by_config(matrix)
    static = configs.get(STATIC, {})
    reserved = configs.get(RESERVED, {})
    static_full_hits = sum(int(e["record"].get(FULL_HITS, 0) or 0) for e in static.values())
    static_h2d = sum(int(e["record"].get(TIMED_H2D, 0) or 0) for e in static.values())
    reserved_full_hits = sum(int(e["record"].get(FULL_HITS, 0) or 0) for e in reserved.values())
    return {
        "static_full_hits_total": static_full_hits,
        "static_has_full_hits": static_full_hits > 0,
        "static_timed_h2d_bytes_total": static_h2d,
        "static_has_zero_timed_h2d": static_h2d == 0,
        "reserved_empty_full_hits_total": reserved_full_hits,
        "reserved_empty_has_zero_full_hits": reserved_full_hits == 0,
    }


def placement_summary(matrix: dict, placement_report: Path | None) -> dict:
    """Full GPU layer count and split layers from a geometry placement JSON."""
    if placement_report is None or not Path(placement_report).is_file():
        return {}
    report = json.loads(Path(placement_report).read_text(encoding="utf-8"))
    layers = report.get("layers", [])
    return {
        "n_layers": len(layers),
        "gpu_layers": sum(1 for l in layers if l.get("placement") == "gpu"),
        "cpu_layers": sum(1 for l in layers if l.get("placement") == "cpu"),
        "split_layers": sum(1 for l in layers if l.get("placement") == "split"),
    }


def analyze(index_path: Path, placement_report: Path | None = None) -> dict:
    matrix = load_matrix(index_path)
    return {
        "capacity_mib": matrix["capacity_mib"],
        "comparisons": {
            "static_vs_baseline": compare(matrix, STATIC, BASELINE),
            "static_vs_reserved": compare(matrix, STATIC, RESERVED),
            "reserved_vs_baseline": compare(matrix, RESERVED, BASELINE),
        },
        "attribution": placement_attribution(matrix),
        "gates": gate_checks(matrix),
        "resident_histogram": resident_histogram(matrix, STATIC),
        "placement": placement_summary(matrix, placement_report),
    }


def format_report(result: dict) -> str:
    lines = []
    lines.append(f"capacity: {result['capacity_mib']} MiB")
    attr = result["attribution"]
    if attr["baseline_tg"]:
        lines.append(f"baseline TG:            {attr['baseline_tg']:.3f}")
    if attr["reserved_empty_tg"]:
        lines.append(f"reserved-empty TG:      {attr['reserved_empty_tg']:.3f}"
                     f"  (placement cost {attr['placement_cost_pct']:+.2f}%)")
    if attr["static_tg"]:
        lines.append(f"static oracle TG:       {attr['static_tg']:.3f}"
                     f"  (execution benefit {attr['execution_benefit_pct']:+.2f}%,"
                     f" net {attr['net_benefit_pct']:+.2f}%)")
    for label, comp in result["comparisons"].items():
        if comp is None:
            lines.append(f"{label}: no paired samples")
            continue
        ci = "n/a" if comp["ci95_low"] is None else f"[{comp['ci95_low']:+.2f}%, {comp['ci95_high']:+.2f}%]"
        lines.append(
            f"{label}: mean {comp['mean_delta_pct']:+.2f}% / median {comp['median_delta_pct']:+.2f}%"
            f" / sd {comp['stdev_delta_pct'] if comp['stdev_delta_pct'] is not None else float('nan'):.2f}"
            f" / positive {comp['positive_pairs']}/{comp['n_pairs']} / 95% CI {ci}"
        )
        lines.append(
            f"    full hits {comp['full_hits_mean']} / fallbacks {comp['fallback_calls_mean']}"
            f" / fast rejects {comp['fast_rejects_mean']} / route-id us {comp['route_id_us_mean']}"
            f" / timed H2D bytes {comp['timed_h2d_bytes_mean']}"
        )
    gates = result["gates"]
    lines.append(f"gates: static full hits={gates['static_full_hits_total']}"
                 f" (pass={gates['static_has_full_hits']})"
                 f", static timed H2D={gates['static_timed_h2d_bytes_total']}"
                 f" (zero={gates['static_has_zero_timed_h2d']})"
                 f", reserved-empty full hits={gates['reserved_empty_full_hits_total']}"
                 f" (zero={gates['reserved_empty_has_zero_full_hits']})")
    if result["resident_histogram"]:
        lines.append(f"static resident histogram (per classification): {result['resident_histogram']}")
    if result["placement"]:
        lines.append(f"placement: {result['placement']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path, help="matrix index JSON from run-tg-matrix.py")
    parser.add_argument("--placement-report", type=Path, default=None,
                        help="placement JSON from test-moe-geometry-report --placement-json")
    parser.add_argument("--json", action="store_true", help="emit raw JSON instead of a report")
    args = parser.parse_args()

    result = analyze(args.index, args.placement_report)
    if args.json:
        json.dump(result, sys.stdout, indent=2)
        print()
    else:
        print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
