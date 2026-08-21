#!/usr/bin/env python3
"""
Phase 5A: Route Trace Analyzer

Analyzes expert routing traces to measure predictability across different
prediction horizons (H layers ahead).

Usage:
    python route-trace-analyzer.py <trace_file> [--output <output_dir>]

Output:
    - Recall curves for different prediction strategies
    - Cross-layer correlation matrix
    - Ready-recall analysis (if timing data available)
"""

import argparse
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")


# Trace entry structure (must match C++ definition)
# uint64_t token_id
# int32_t layer
# int32_t n_experts
# int32_t expert_ids[64]
# uint64_t timestamp_us
ENTRY_FORMAT = '<Qii64iQ'
ENTRY_SIZE = struct.calcsize(ENTRY_FORMAT)
MAGIC = 0x52545243  # "RTRC"


class RouteTraceEntry:
    def __init__(self, token_id: int, layer: int, n_experts: int, expert_ids: List[int], timestamp_us: int):
        self.token_id = token_id
        self.layer = layer
        self.n_experts = n_experts
        self.expert_ids = expert_ids[:n_experts]  # trim padding
        self.timestamp_us = timestamp_us

    def __repr__(self):
        return f"RouteTraceEntry(token={self.token_id}, layer={self.layer}, experts={self.expert_ids})"


def load_trace(trace_path: str) -> List[RouteTraceEntry]:
    """Load binary trace file."""
    entries = []
    
    with open(trace_path, 'rb') as f:
        # Read header
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: 0x{magic:08X} (expected 0x{MAGIC:08X})")
        
        version = struct.unpack('<I', f.read(4))[0]
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")
        
        # Read entries
        while True:
            data = f.read(ENTRY_SIZE)
            if len(data) < ENTRY_SIZE:
                break
            
            fields = struct.unpack(ENTRY_FORMAT, data)
            token_id = fields[0]
            layer = fields[1]
            n_experts = fields[2]
            expert_ids = list(fields[3:3+64])
            timestamp_us = fields[67]
            
            entries.append(RouteTraceEntry(token_id, layer, n_experts, expert_ids, timestamp_us))
    
    print(f"Loaded {len(entries)} trace entries")
    return entries


def compute_recall(predicted: Set[int], actual: Set[int], k: int) -> float:
    """Compute recall@k: fraction of actual experts in top-k predicted."""
    if not actual:
        return 0.0
    
    predicted_k = set(list(predicted)[:k])
    intersection = predicted_k & actual
    return len(intersection) / len(actual)


def compute_precision(predicted: Set[int], actual: Set[int], k: int) -> float:
    """Compute precision@k: fraction of top-k predicted that are actual."""
    if not predicted:
        return 0.0
    
    predicted_k = set(list(predicted)[:k])
    intersection = predicted_k & actual
    return len(intersection) / len(predicted_k)


def analyze_previous_token_same_layer(entries: List[RouteTraceEntry]) -> Dict[int, Dict[str, float]]:
    """
    Predictor A: Previous token, same layer.
    
    When generating token t, we already know the complete expert route taken by token t-1.
    Predict: route(t-1, L) -> route(t, L)
    """
    print("\n=== Predictor A: Previous Token, Same Layer ===")
    
    # Group by layer
    layer_traces = defaultdict(list)
    for entry in entries:
        layer_traces[entry.layer].append(entry)
    
    results = {}
    
    for layer in sorted(layer_traces.keys()):
        traces = layer_traces[layer]
        
        # Sort by token_id
        traces.sort(key=lambda e: e.token_id)
        
        recalls = {8: [], 10: [], 12: [], 16: []}
        precisions = {8: [], 10: [], 12: [], 16: []}
        
        for i in range(1, len(traces)):
            prev_experts = set(traces[i-1].expert_ids)
            curr_experts = set(traces[i].expert_ids)
            
            for k in recalls.keys():
                recalls[k].append(compute_recall(prev_experts, curr_experts, k))
                precisions[k].append(compute_precision(prev_experts, curr_experts, k))
        
        results[layer] = {
            'recall@8': np.mean(recalls[8]) if recalls[8] else 0.0,
            'recall@10': np.mean(recalls[10]) if recalls[10] else 0.0,
            'recall@12': np.mean(recalls[12]) if recalls[12] else 0.0,
            'recall@16': np.mean(recalls[16]) if recalls[16] else 0.0,
            'precision@8': np.mean(precisions[8]) if precisions[8] else 0.0,
            'precision@10': np.mean(precisions[10]) if precisions[10] else 0.0,
            'precision@12': np.mean(precisions[12]) if precisions[12] else 0.0,
            'precision@16': np.mean(precisions[16]) if precisions[16] else 0.0,
        }
    
    return results


def analyze_cross_layer(entries: List[RouteTraceEntry], horizons: List[int]) -> Dict[int, Dict[int, Dict[str, float]]]:
    """
    Predictor B: Cross-layer prediction.
    
    Predict: route(t, L) -> route(t, L+H) for different horizons H.
    """
    print("\n=== Predictor B: Cross-Layer Prediction ===")
    
    # Group by token
    token_traces = defaultdict(list)
    for entry in entries:
        token_traces[entry.token_id].append(entry)
    
    results = {}
    
    for horizon in horizons:
        recalls = {8: [], 10: [], 12: [], 16: []}
        precisions = {8: [], 10: [], 12: [], 16: []}
        
        for token_id, traces in token_traces.items():
            # Group by layer
            layer_map = {t.layer: t for t in traces}
            
            for layer_src in sorted(layer_map.keys()):
                layer_dst = layer_src + horizon
                if layer_dst not in layer_map:
                    continue
                
                src_experts = set(layer_map[layer_src].expert_ids)
                dst_experts = set(layer_map[layer_dst].expert_ids)
                
                for k in recalls.keys():
                    recalls[k].append(compute_recall(src_experts, dst_experts, k))
                    precisions[k].append(compute_precision(src_experts, dst_experts, k))
        
        results[horizon] = {
            'recall@8': np.mean(recalls[8]) if recalls[8] else 0.0,
            'recall@10': np.mean(recalls[10]) if recalls[10] else 0.0,
            'recall@12': np.mean(recalls[12]) if recalls[12] else 0.0,
            'recall@16': np.mean(recalls[16]) if recalls[16] else 0.0,
            'precision@8': np.mean(precisions[8]) if precisions[8] else 0.0,
            'precision@10': np.mean(precisions[10]) if precisions[10] else 0.0,
            'precision@12': np.mean(precisions[12]) if precisions[12] else 0.0,
            'precision@16': np.mean(precisions[16]) if precisions[16] else 0.0,
        }
    
    return results


def print_results(results: Dict, title: str):
    """Print results in a formatted table."""
    print(f"\n{title}")
    print("=" * 80)
    
    if isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
        # Single-level dict (e.g., layer -> metrics)
        print(f"{'Key':<10} {'Recall@8':<10} {'Recall@10':<10} {'Recall@12':<10} {'Recall@16':<10}")
        print("-" * 80)
        for key, metrics in sorted(results.items()):
            print(f"{key:<10} {metrics['recall@8']:<10.3f} {metrics['recall@10']:<10.3f} "
                  f"{metrics['recall@12']:<10.3f} {metrics['recall@16']:<10.3f}")
    else:
        # Two-level dict (e.g., horizon -> layer -> metrics)
        for key1, inner in sorted(results.items()):
            print(f"\n{key1}:")
            print(f"{'Layer':<10} {'Recall@8':<10} {'Recall@10':<10} {'Recall@12':<10} {'Recall@16':<10}")
            print("-" * 80)
            for key2, metrics in sorted(inner.items()):
                print(f"{key2:<10} {metrics['recall@8']:<10.3f} {metrics['recall@10']:<10.3f} "
                      f"{metrics['recall@12']:<10.3f} {metrics['recall@16']:<10.3f}")


def plot_recall_curves(results: Dict[int, Dict[str, float]], output_dir: Path):
    """Plot recall vs horizon."""
    if not HAS_MATPLOTLIB:
        return
    
    horizons = sorted(results.keys())
    recalls = {
        8: [results[h]['recall@8'] for h in horizons],
        10: [results[h]['recall@10'] for h in horizons],
        12: [results[h]['recall@12'] for h in horizons],
        16: [results[h]['recall@16'] for h in horizons],
    }
    
    plt.figure(figsize=(10, 6))
    for k, values in recalls.items():
        plt.plot(horizons, values, marker='o', label=f'Recall@{k}')
    
    plt.xlabel('Prediction Horizon (layers ahead)')
    plt.ylabel('Recall')
    plt.title('Cross-Layer Prediction: Recall vs Horizon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'recall_vs_horizon.png', dpi=150)
    print(f"Saved plot: {output_dir / 'recall_vs_horizon.png'}")


def plot_layer_recall(results: Dict[int, Dict[str, float]], output_dir: Path):
    """Plot recall vs layer for previous-token predictor."""
    if not HAS_MATPLOTLIB:
        return
    
    layers = sorted(results.keys())
    recalls = {
        8: [results[l]['recall@8'] for l in layers],
        10: [results[l]['recall@10'] for l in layers],
        12: [results[l]['recall@12'] for l in layers],
        16: [results[l]['recall@16'] for l in layers],
    }
    
    plt.figure(figsize=(12, 6))
    for k, values in recalls.items():
        plt.plot(layers, values, marker='o', label=f'Recall@{k}')
    
    plt.xlabel('Layer')
    plt.ylabel('Recall')
    plt.title('Previous-Token Same-Layer Prediction: Recall vs Layer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'recall_vs_layer.png', dpi=150)
    print(f"Saved plot: {output_dir / 'recall_vs_layer.png'}")


def main():
    parser = argparse.ArgumentParser(description='Analyze expert routing traces')
    parser.add_argument('trace_file', help='Path to binary trace file')
    parser.add_argument('--output', '-o', default='.', help='Output directory for plots')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trace
    entries = load_trace(args.trace_file)
    if not entries:
        print("No entries found in trace file")
        return
    
    # Analyze previous-token same-layer predictor
    prev_token_results = analyze_previous_token_same_layer(entries)
    print_results(prev_token_results, "Previous-Token Same-Layer Prediction")
    plot_layer_recall(prev_token_results, output_dir)
    
    # Analyze cross-layer predictor
    horizons = [1, 2, 4, 8, 12, 16]
    cross_layer_results = analyze_cross_layer(entries, horizons)
    print_results(cross_layer_results, "Cross-Layer Prediction")
    plot_recall_curves(cross_layer_results, output_dir)
    
    print("\n=== Summary ===")
    print(f"Total entries: {len(entries)}")
    print(f"Unique tokens: {len(set(e.token_id for e in entries))}")
    print(f"Unique layers: {len(set(e.layer for e in entries))}")
    
    # Go/No-Go decision
    print("\n=== Go/No-Go Decision ===")
    avg_recall_8 = np.mean([cross_layer_results[h]['recall@8'] for h in [4, 8]])
    if avg_recall_8 > 0.70:
        print(f"✓ PASS: Average Recall@8 for H=4,8 is {avg_recall_8:.3f} (>0.70)")
        print("  Routing is predictable enough to proceed with prefetch pipeline.")
    else:
        print(f"✗ FAIL: Average Recall@8 for H=4,8 is {avg_recall_8:.3f} (<0.70)")
        print("  Routing may be too unpredictable. Consider oracle simulator before proceeding.")


if __name__ == '__main__':
    main()
