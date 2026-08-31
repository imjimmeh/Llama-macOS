#!/usr/bin/env python3
"""Held-out full-hit residency oracle analyzer.

Subcommands:
  geometry      validate model and bundle geometry
  characterize  emit per-layer concentration and coverage tables
  oracle        emit one v3 manifest plus predicted benefit
  sweep         emit 32,64,96,128,160,192,256,384,512 MiB frontier
  compare       score a manifest against an independent held-out trace
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --- Trace loading ---

def load_trace(path: Path) -> dict[str, Any]:
    """Load an attested JSONL trace. Raises ValueError if not attested."""
    header = None
    routes = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj.get("_header"):
                header = obj
            else:
                routes.append(obj)
    if header is None:
        raise ValueError(f"no header in trace: {path}")
    if not header.get("callback_matches_canonical", False):
        raise ValueError(f"trace not attested (callback_matches_canonical=false): {path}")
    return {"header": header, "routes": routes}


# --- Trace statistics ---

def count_expert_freq(routes: list[dict]) -> dict[int, Counter]:
    """Per-layer expert frequency counts."""
    freq: dict[int, Counter] = defaultdict(Counter)
    for r in routes:
        layer = r["layer"]
        for e in r["experts"]:
            freq[layer][e] += 1
    return dict(freq)


def individual_coverage(routes: list[dict], top_n: int) -> dict[int, float]:
    """Per-layer fraction of routes where ALL top_n most-frequent experts appear in the route."""
    freq = count_expert_freq(routes)
    result = {}
    for layer, counter in freq.items():
        top_experts = {e for e, _ in counter.most_common(top_n)}
        layer_routes = [r for r in routes if r["layer"] == layer]
        if not layer_routes:
            result[layer] = 0.0
            continue
        covered = sum(1 for r in layer_routes if set(r["experts"]).issuperset(top_experts))
        result[layer] = covered / len(layer_routes)
    return result


def full_hit_probability(routes: list[dict], resident_set: set[int]) -> float:
    """Fraction of routes fully covered by resident_set."""
    if not routes:
        return 0.0
    covered = sum(1 for r in routes if resident_set.issuperset(r["experts"]))
    return covered / len(routes)


def shannon_entropy(freq: Counter) -> float:
    """Shannon entropy in bits."""
    total = sum(freq.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in freq.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def normalized_entropy(freq: Counter) -> float:
    """Normalized entropy [0, 1]."""
    n = len(freq)
    if n <= 1:
        return 0.0
    return shannon_entropy(freq) / math.log2(n)


def gini_coefficient(values: list[float]) -> float:
    """Gini coefficient from a list of non-negative values."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    total = sum(s)
    if total == 0:
        return 0.0
    cumulative = 0.0
    gini_sum = 0.0
    for i, v in enumerate(s):
        cumulative += v
        gini_sum += (2 * (i + 1) - n - 1) * v
    return gini_sum / (n * total)


def has_duplicate_route_ids(experts: list[int]) -> bool:
    return len(experts) != len(set(experts))


# --- RouteTrace dataclass ---

@dataclass
class RouteTrace:
    """Aggregated route data for one layer."""
    routes: list[dict]
    layer: int
    saved_us: float
    bundle_bytes: int


# --- Oracle ---

def greedy_oracle(
    traces: list[RouteTrace],
    capacity_bytes: int,
    top_k: int = 8,
) -> dict[str, Any]:
    """Greedy full-hit oracle: maximize 8/8 full hits within capacity.

    Uses route-set greedy: for each unique route, compute the bytes needed
    to complete it (missing experts * bundle_bytes) and the value (saved_us).
    Greedily select routes by value/cost ratio, adding missing experts.
    """
    # Collect all routes per layer
    layer_routes: dict[int, list[dict]] = defaultdict(list)
    layer_saved: dict[int, float] = {}
    layer_bundle_bytes: dict[int, int] = {}
    for t in traces:
        layer_routes[t.layer].extend(t.routes)
        layer_saved[t.layer] = t.saved_us
        layer_bundle_bytes[t.layer] = t.bundle_bytes

    selected: list[dict] = []
    used_bytes = 0
    resident: dict[int, set[int]] = defaultdict(set)

    # Greedy: at each step, find the route with the best value/cost ratio
    # that can be completed within the remaining budget
    while True:
        best_score = -1.0
        best_route_info = None

        for layer, routes in layer_routes.items():
            saved = layer_saved.get(layer, 0)
            bb = layer_bundle_bytes.get(layer, 0)
            if bb == 0:
                continue

            for r in routes:
                route_experts = set(r["experts"])
                missing = route_experts - resident[layer]
                if not missing:
                    continue  # already fully covered

                cost = len(missing) * bb
                if used_bytes + cost > capacity_bytes:
                    continue

                # Value: saved_us for one new full hit
                value = saved
                score = value / cost if cost > 0 else float('inf')

                # Tie-break: (layer, route_experts) for determinism
                if score > best_score:
                    best_score = score
                    best_route_info = (layer, missing, cost, route_experts)

        if best_route_info is None:
            break

        layer, missing, cost, route_experts = best_route_info
        for expert in sorted(missing):
            resident[layer].add(expert)
            used_bytes += layer_bundle_bytes[layer]
            selected.append({
                "layer": layer,
                "expert": expert,
                "bundle_bytes": layer_bundle_bytes[layer],
            })

    # Count predicted full hits
    predicted_full_hits = 0
    total_routes = 0
    for layer, routes in layer_routes.items():
        res = resident.get(layer, set())
        for r in routes:
            total_routes += 1
            if res.issuperset(r["experts"]):
                predicted_full_hits += 1

    return {
        "bundles": selected,
        "predicted_full_hits": predicted_full_hits,
        "total_routes": total_routes,
        "hit_rate": predicted_full_hits / total_routes if total_routes > 0 else 0.0,
        "used_bytes": used_bytes,
        "capacity_bytes": capacity_bytes,
    }


# --- Analyzer subcommands ---

def cmd_geometry(geom_path: Path) -> dict[str, Any]:
    """Validate model and bundle geometry."""
    with open(geom_path) as f:
        geom = json.load(f)
    return {
        "expert_count": geom["expert_count"],
        "top_k": geom["top_k"],
        "n_layers": geom["n_layers"],
        "all_equal": geom.get("all_repeated_moe_blocks_equal", False),
        "bundle_bytes_range": _bundle_bytes_range(geom),
    }


def _bundle_bytes_range(geom: dict) -> dict[str, int]:
    layers = geom["layers"]
    sizes = [l["complete_bundle_bytes"] for l in layers]
    return {"min": min(sizes), "max": max(sizes), "mean": sum(sizes) // len(sizes)}


def cmd_characterize(trace_path: Path, geom_path: Path) -> dict[str, Any]:
    """Emit per-layer concentration and coverage tables."""
    trace = load_trace(trace_path)
    with open(geom_path) as f:
        geom = json.load(f)

    routes = trace["routes"]
    freq = count_expert_freq(routes)
    layer_data = []

    for layer_info in geom["layers"]:
        layer = layer_info["layer"]
        counter = freq.get(layer, Counter())
        if not counter:
            continue

        total = sum(counter.values())
        sorted_freq = counter.most_common()
        values = [c for _, c in sorted_freq]

        layer_data.append({
            "layer": layer,
            "expert_freq": dict(sorted_freq[:16]),
            "entropy": shannon_entropy(counter),
            "normalized_entropy": normalized_entropy(counter),
            "gini": gini_coefficient(values),
            "total_expert_hits": total,
            "unique_experts": len(counter),
            "bundle_bytes": layer_info["complete_bundle_bytes"],
        })

    return {"layers": layer_data, "n_routes": len(routes)}


def cmd_oracle(
    trace_path: Path,
    geom_path: Path,
    capacity_mib: int = 128,
    held_out: bool = False,
) -> dict[str, Any]:
    """Emit one v3 manifest plus predicted benefit."""
    trace = load_trace(trace_path)
    with open(geom_path) as f:
        geom = json.load(f)

    routes = trace["routes"]
    if held_out:
        train, test = split_routes(routes, train_frac=0.5)
        routes_for_oracle = train
        routes_for_eval = test
        partition = "held_out"
    else:
        routes_for_oracle = routes
        routes_for_eval = routes
        partition = "combined"

    # Build RouteTrace per layer
    traces = []
    for layer_info in geom["layers"]:
        layer = layer_info["layer"]
        layer_routes = [r for r in routes_for_oracle if r["layer"] == layer]
        if not layer_routes:
            continue
        traces.append(RouteTrace(
            routes=layer_routes,
            layer=layer,
            saved_us=1000.0,  # placeholder until real latency data
            bundle_bytes=layer_info["complete_bundle_bytes"],
        ))

    capacity_bytes = capacity_mib * 1024 * 1024
    result = greedy_oracle(traces, capacity_bytes)

    # Build v3 manifest
    manifest = {
        "format": 3,
        "admission": "8of8",
        "model": {
            "sha256": geom.get("model_sha256", ""),
            "top_k": geom["top_k"],
            "expert_count": geom["expert_count"],
        },
        "cache_bytes": capacity_bytes,
        "bundles": [
            {"layer": b["layer"], "expert": b["expert"],
             "projections": ["gate", "up", "down"]}
            for b in result["bundles"]
        ],
    }

    # Evaluate on held-out if applicable
    eval_result = None
    if held_out and routes_for_eval:
        resident_by_layer: dict[int, set[int]] = defaultdict(set)
        for b in result["bundles"]:
            resident_by_layer[b["layer"]].add(b["expert"])
        eval_full_hits = 0
        eval_total = 0
        for r in routes_for_eval:
            layer = r["layer"]
            res = resident_by_layer.get(layer, set())
            eval_total += 1
            if res.issuperset(r["experts"]):
                eval_full_hits += 1
        eval_result = {
            "full_hits": eval_full_hits,
            "total_routes": eval_total,
            "hit_rate": eval_full_hits / eval_total if eval_total > 0 else 0.0,
        }

    return {
        "manifest": manifest,
        "predicted_full_hits": result["predicted_full_hits"],
        "total_routes": result["total_routes"],
        "hit_rate": result["hit_rate"],
        "used_bytes": result["used_bytes"],
        "partition": partition,
        "held_out_eval": eval_result,
        "predicted_saved_ms_per_1k": result["predicted_full_hits"] * 1.0,
    }


def cmd_sweep(trace_path: Path, geom_path: Path) -> dict[str, Any]:
    """Emit frontier across capacities."""
    capacities = [32, 64, 96, 128, 160, 192, 256, 384, 512]
    results = []
    for cap in capacities:
        r = cmd_oracle(trace_path, geom_path, capacity_mib=cap)
        results.append({
            "capacity_mib": cap,
            "bundles": len(r["manifest"]["bundles"]),
            "predicted_full_hits": r["predicted_full_hits"],
            "hit_rate": r["hit_rate"],
            "used_bytes": r["used_bytes"],
        })
    return {"capacities": results}


def cmd_compare(manifest: dict, trace_path: Path, geom_path: Path) -> dict[str, Any]:
    """Score a manifest against an independent trace."""
    trace = load_trace(trace_path)
    with open(geom_path) as f:
        geom = json.load(f)

    routes = trace["routes"]
    resident_by_layer: dict[int, set[int]] = defaultdict(set)
    for b in manifest.get("bundles", []):
        resident_by_layer[b["layer"]].add(b["expert"])

    full_hits = 0
    total = 0
    for r in routes:
        total += 1
        layer = r["layer"]
        res = resident_by_layer.get(layer, set())
        if res.issuperset(r["experts"]):
            full_hits += 1

    return {
        "full_hits": full_hits,
        "total_routes": total,
        "hit_rate": full_hits / total if total > 0 else 0.0,
        "resident_bundles": len(manifest.get("bundles", [])),
    }


def split_routes(routes: list[dict], train_frac: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Split routes by sequence_index for held-out protocol."""
    by_seq: dict[int, list[dict]] = defaultdict(list)
    for r in routes:
        by_seq[r.get("sequence_index", 0)].append(r)
    seqs = sorted(by_seq.keys())
    split_idx = int(len(seqs) * train_frac)
    train_seqs = set(seqs[:split_idx])
    train = []
    test = []
    for seq, rs in by_seq.items():
        if seq in train_seqs:
            train.extend(rs)
        else:
            test.extend(rs)
    return train, test


# --- CLI ---

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    g = sub.add_parser("geometry")
    g.add_argument("geom", type=Path)

    c = sub.add_parser("characterize")
    c.add_argument("trace", type=Path)
    c.add_argument("geom", type=Path)

    o = sub.add_parser("oracle")
    o.add_argument("trace", type=Path)
    o.add_argument("geom", type=Path)
    o.add_argument("--capacity-mib", type=int, default=128)
    o.add_argument("--held-out", action="store_true")

    s = sub.add_parser("sweep")
    s.add_argument("trace", type=Path)
    s.add_argument("geom", type=Path)

    cmp = sub.add_parser("compare")
    cmp.add_argument("manifest", type=Path)
    cmp.add_argument("trace", type=Path)
    cmp.add_argument("geom", type=Path)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return 1

    if args.cmd == "geometry":
        result = cmd_geometry(args.geom)
    elif args.cmd == "characterize":
        result = cmd_characterize(args.trace, args.geom)
    elif args.cmd == "oracle":
        result = cmd_oracle(args.trace, args.geom, args.capacity_mib, args.held_out)
    elif args.cmd == "sweep":
        result = cmd_sweep(args.trace, args.geom)
    elif args.cmd == "compare":
        with open(args.manifest) as f:
            manifest = json.load(f)
        result = cmd_compare(manifest, args.trace, args.geom)
    else:
        parser.print_help()
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
