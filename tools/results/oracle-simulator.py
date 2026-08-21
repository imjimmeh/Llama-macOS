#!/usr/bin/env python3
"""
Phase 5B: Oracle Simulator

Simulates perfect prediction (oracle) to determine theoretical maximum benefit
of routing lookahead pipeline.

Usage:
    python oracle-simulator.py <trace_file> [--pcie-bandwidth <GB/s>] [--cache-capacity <MiB>]

Output:
    - Theoretical speedup for different prediction horizons
    - Bottleneck analysis (latency vs bandwidth)
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


# Trace entry structure
ENTRY_FORMAT = '<Qii64iQ'
ENTRY_SIZE = struct.calcsize(ENTRY_FORMAT)
MAGIC = 0x52545243


class RouteTraceEntry:
    def __init__(self, token_id: int, layer: int, n_experts: int, expert_ids: List[int], timestamp_us: int):
        self.token_id = token_id
        self.layer = layer
        self.n_experts = n_experts
        self.expert_ids = expert_ids[:n_experts]
        self.timestamp_us = timestamp_us


def load_trace(trace_path: str) -> List[RouteTraceEntry]:
    """Load binary trace file."""
    entries = []
    
    with open(trace_path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: 0x{magic:08X}")
        
        version = struct.unpack('<I', f.read(4))[0]
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")
        
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


def estimate_expert_size(n_layers: int = 40, n_experts_per_layer: int = 128, 
                         hidden_dim: int = 2048, intermediate_dim: int = 5120) -> int:
    """
    Estimate size of one expert in bytes.
    
    For Qwen 35B MoE:
    - Each expert has gate, up, down projections
    - gate: [hidden_dim, intermediate_dim] = [2048, 5120]
    - up: [hidden_dim, intermediate_dim] = [2048, 5120]
    - down: [intermediate_dim, hidden_dim] = [5120, 2048]
    - Assuming Q4_K_M quantization (~4.5 bits per param)
    """
    params_per_expert = (hidden_dim * intermediate_dim * 2 + intermediate_dim * hidden_dim)
    bytes_per_param = 4.5 / 8  # Q4_K_M
    expert_bytes = int(params_per_expert * bytes_per_param)
    
    return expert_bytes


def simulate_baseline(entries: List[RouteTraceEntry], expert_size: int, 
                     pcie_bandwidth_gbs: float) -> Dict[str, float]:
    """
    Simulate baseline (no prefetch): every expert access is a miss.
    
    Returns:
        - total_transfer_time_us: total time spent on DMA transfers
        - total_tokens: number of tokens processed
        - time_per_token_us: average time per token
    """
    # Group by token
    token_traces = defaultdict(list)
    for entry in entries:
        token_traces[entry.token_id].append(entry)
    
    total_transfers = 0
    total_bytes = 0
    
    for token_id, traces in token_traces.items():
        # Count unique experts per token
        unique_experts = set()
        for trace in traces:
            for expert_id in trace.expert_ids:
                unique_experts.add((trace.layer, expert_id))
        
        total_transfers += len(unique_experts)
        total_bytes += len(unique_experts) * expert_size
    
    # Calculate transfer time
    pcie_bandwidth_bytes_per_us = pcie_bandwidth_gbs * 1e9 / 1e6  # GB/s -> bytes/us
    total_transfer_time_us = total_bytes / pcie_bandwidth_bytes_per_us
    
    n_tokens = len(token_traces)
    time_per_token_us = total_transfer_time_us / n_tokens if n_tokens > 0 else 0
    
    return {
        'total_transfers': total_transfers,
        'total_bytes': total_bytes,
        'total_transfer_time_us': total_transfer_time_us,
        'total_tokens': n_tokens,
        'time_per_token_us': time_per_token_us,
        'bytes_per_token': total_bytes / n_tokens if n_tokens > 0 else 0,
    }


def simulate_oracle(entries: List[RouteTraceEntry], horizon: int, expert_size: int,
                   pcie_bandwidth_gbs: float, cache_capacity_bytes: int,
                   layer_compute_time_us: float = 100.0) -> Dict[str, float]:
    """
    Simulate oracle with perfect prediction at given horizon.
    
    Oracle knows exactly which experts will be needed H layers ahead and
    initiates prefetch early enough that transfer completes before execution.
    
    Args:
        horizon: prediction horizon (0 = no lookahead, -1 = whole-token oracle)
        expert_size: size of one expert in bytes
        pcie_bandwidth_gbs: PCIe bandwidth in GB/s
        cache_capacity_bytes: cache capacity in bytes
        layer_compute_time_us: estimated compute time per layer in microseconds
    
    Returns:
        - fully_hidden_hits: experts that arrived before needed
        - misses: experts that didn't arrive in time
        - wasted_prefetches: experts prefetched but not used
        - time_per_token_us: average time per token
    """
    # Group by token
    token_traces = defaultdict(list)
    for entry in entries:
        token_traces[entry.token_id].append(entry)
    
    pcie_bandwidth_bytes_per_us = pcie_bandwidth_gbs * 1e9 / 1e6
    
    total_fully_hidden = 0
    total_partially_hidden = 0
    total_misses = 0
    total_wasted = 0
    total_transfer_time_us = 0
    
    for token_id, traces in sorted(token_traces.items()):
        # Sort by layer
        traces.sort(key=lambda e: e.layer)
        
        # Track cache state
        cache_state = {}  # (layer, expert_id) -> timestamp_when_resident
        prefetch_queue = []  # list of (target_layer, expert_id, completion_time, layer_exec_time)
        
        n_layers = len(traces)
        
        for i, trace in enumerate(traces):
            current_layer = trace.layer
            current_time_us = i * layer_compute_time_us
            
            # Determine prediction horizon (FUTURE layers only)
            if horizon == -1:
                # Whole-token oracle: know all future layers
                future_layers = [t.layer for t in traces if t.layer > current_layer]
            elif horizon == 0:
                # No lookahead: no prefetching
                future_layers = []
            else:
                # Fixed horizon: predict layers current_layer+1 to current_layer+horizon
                future_layers = [t.layer for t in traces 
                                if current_layer < t.layer <= current_layer + horizon]
            
            # Issue prefetches for predicted future experts
            for pred_layer in future_layers:
                pred_trace = next((t for t in traces if t.layer == pred_layer), None)
                if pred_trace is None:
                    continue
                
                for expert_id in pred_trace.expert_ids:
                    key = (pred_layer, expert_id)
                    
                    # Skip if already in cache or already prefetching
                    if key in cache_state:
                        continue
                    if any(pq[1] == key for pq in prefetch_queue):
                        continue
                    
                    # Calculate transfer time
                    transfer_time_us = expert_size / pcie_bandwidth_bytes_per_us
                    
                    # Calculate when we'll reach this layer
                    layers_ahead = pred_layer - current_layer
                    layer_execution_time_us = current_time_us + layers_ahead * layer_compute_time_us
                    
                    # Transfer completion time
                    completion_time = current_time_us + transfer_time_us
                    
                    # Enqueue prefetch
                    prefetch_queue.append((pred_layer, key, completion_time, layer_execution_time_us))
            
            # Process prefetch queue: move completed transfers to cache
            still_queued = []
            for target_layer, key, completion_time, layer_exec_time_us in prefetch_queue:
                if completion_time <= layer_exec_time_us:
                    # Transfer completed before layer execution
                    cache_state[key] = completion_time
                else:
                    still_queued.append((target_layer, key, completion_time, layer_exec_time_us))
            prefetch_queue = still_queued
            
            # Check if current layer's experts are resident
            for expert_id in trace.expert_ids:
                key = (current_layer, expert_id)
                
                if key in cache_state:
                    # Fully hidden hit (arrived before we needed it)
                    total_fully_hidden += 1
                else:
                    # Check if prefetching and will arrive soon
                    prefetch_match = next((pq for pq in prefetch_queue if pq[1] == key), None)
                    if prefetch_match:
                        # Partially hidden hit (we have to wait)
                        total_partially_hidden += 1
                        wait_time = prefetch_match[2] - current_time_us
                        total_transfer_time_us += wait_time
                        cache_state[key] = prefetch_match[2]
                    else:
                        # Miss: synchronous transfer
                        total_misses += 1
                        transfer_time_us = expert_size / pcie_bandwidth_bytes_per_us
                        total_transfer_time_us += transfer_time_us
                        cache_state[key] = current_time_us + transfer_time_us
    
    # Count wasted prefetches (prefetched but never used)
    # This is approximated by checking cache state at end
    total_wasted = len(cache_state) - total_fully_hidden - total_partially_hidden - total_misses
    total_wasted = max(0, total_wasted)
    
    total_requests = total_fully_hidden + total_partially_hidden + total_misses
    n_tokens = len(token_traces)
    time_per_token_us = total_transfer_time_us / n_tokens if n_tokens > 0 else 0
    
    return {
        'fully_hidden_hits': total_fully_hidden,
        'partially_hidden_hits': total_partially_hidden,
        'misses': total_misses,
        'wasted_prefetches': total_wasted,
        'total_requests': total_requests,
        'total_transfer_time_us': total_transfer_time_us,
        'total_tokens': n_tokens,
        'time_per_token_us': time_per_token_us,
        'hit_rate': (total_fully_hidden + total_partially_hidden) / total_requests if total_requests > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Oracle simulator for routing lookahead')
    parser.add_argument('trace_file', help='Path to binary trace file')
    parser.add_argument('--pcie-bandwidth', type=float, default=8.0, 
                       help='PCIe bandwidth in GB/s (default: 8.0 for PCIe 2.0 x8)')
    parser.add_argument('--cache-capacity', type=float, default=4096,
                       help='Cache capacity in MiB (default: 4096)')
    parser.add_argument('--layer-compute-time', type=float, default=100.0,
                       help='Layer compute time in microseconds (default: 100)')
    parser.add_argument('--output-dir', type=str, default='tools/results',
                       help='Output directory for plots (default: tools/results)')
    args = parser.parse_args()
    
    # Load trace
    print(f"Loading trace from {args.trace_file}...")
    entries = load_trace(args.trace_file)
    
    if not entries:
        print("Error: No entries in trace file")
        sys.exit(1)
    
    # Estimate expert size
    n_layers = max(e.layer for e in entries) + 1
    expert_size = estimate_expert_size(n_layers=n_layers)
    print(f"Estimated expert size: {expert_size / 1024 / 1024:.2f} MiB")
    
    # Run baseline simulation
    print("\n=== Baseline (No Prefetch) ===")
    baseline = simulate_baseline(entries, expert_size, args.pcie_bandwidth)
    print(f"Total transfers: {baseline['total_transfers']}")
    print(f"Total bytes: {baseline['total_bytes'] / 1024 / 1024 / 1024:.2f} GiB")
    print(f"Bytes per token: {baseline['bytes_per_token'] / 1024:.2f} MiB")
    print(f"Time per token: {baseline['time_per_token_us']:.2f} us")
    
    # Run oracle simulations for different horizons
    print("\n=== Oracle Simulations ===")
    horizons = [0, 1, 2, 4, 8, 12, 16, -1]
    results = []
    
    for horizon in horizons:
        cache_capacity_bytes = int(args.cache_capacity * 1024 * 1024)
        result = simulate_oracle(
            entries, horizon, expert_size, args.pcie_bandwidth,
            cache_capacity_bytes, args.layer_compute_time
        )
        results.append((horizon, result))
        
        speedup = baseline['time_per_token_us'] / result['time_per_token_us'] if result['time_per_token_us'] > 0 else 0
        
        horizon_str = "whole-token" if horizon == -1 else f"H={horizon}"
        print(f"{horizon_str:12s} | "
              f"Fully Hidden: {result['fully_hidden_hits']:6d} | "
              f"Partially Hidden: {result['partially_hidden_hits']:6d} | "
              f"Misses: {result['misses']:6d} | "
              f"Hit Rate: {result['hit_rate']:.3f} | "
              f"Time/Token: {result['time_per_token_us']:.2f} us | "
              f"Speedup: {speedup:.2f}x")
    
    # Plot results if matplotlib available
    if HAS_MATPLOTLIB:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Hit rate vs horizon
        horizon_labels = ["whole-token" if h == -1 else f"H={h}" for h, _ in results]
        hit_rates = [r['hit_rate'] for _, r in results]
        fully_hidden_rates = [r['fully_hidden_hits'] / r['total_requests'] if r['total_requests'] > 0 else 0 
                             for _, r in results]
        
        ax1.plot(horizon_labels, hit_rates, 'b-o', label='Total Hit Rate')
        ax1.plot(horizon_labels, fully_hidden_rates, 'g-s', label='Fully Hidden Rate')
        ax1.set_xlabel('Prediction Horizon')
        ax1.set_ylabel('Hit Rate')
        ax1.set_title('Oracle Prediction Hit Rate vs Horizon')
        ax1.legend()
        ax1.grid(True)
        ax1.set_ylim(0, 1.1)
        
        # Plot 2: Time per token vs horizon
        times = [r['time_per_token_us'] for _, r in results]
        speedups = [baseline['time_per_token_us'] / t if t > 0 else 0 for t in times]
        
        ax2_twin = ax2.twinx()
        ax2.plot(horizon_labels, times, 'r-o', label='Time/Token (us)')
        ax2_twin.plot(horizon_labels, speedups, 'm-s', label='Speedup')
        ax2.set_xlabel('Prediction Horizon')
        ax2.set_ylabel('Time per Token (us)', color='r')
        ax2_twin.set_ylabel('Speedup', color='m')
        ax2.set_title('Oracle Simulation: Time per Token and Speedup')
        ax2.grid(True)
        
        # Add legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plot_path = output_dir / 'oracle_simulation.png'
        plt.savefig(plot_path, dpi=150)
        print(f"\nSaved plot: {plot_path}")
    
    # Bottleneck analysis
    print("\n=== Bottleneck Analysis ===")
    best_result = results[-1][1]  # Whole-token oracle
    best_speedup = baseline['time_per_token_us'] / best_result['time_per_token_us'] if best_result['time_per_token_us'] > 0 else 0
    
    if best_speedup < 1.1:
        print("WARNING: Even perfect prediction gives <10% speedup")
        print("This suggests PCIe bandwidth is the bottleneck, not latency")
        print("Consider: reducing expert size, increasing cache capacity, or using faster interconnect")
    elif best_speedup > 1.3:
        print(f"SUCCESS: Perfect prediction gives {best_speedup:.1f}x speedup")
        print("This confirms latency is the bottleneck and prefetching is worthwhile")
        print("Next step: implement heuristic predictor (Phase 5C)")
    else:
        print(f"MODERATE: Perfect prediction gives {best_speedup:.2f}x speedup")
        print("Some benefit from prefetching, but may not justify complexity")


if __name__ == '__main__':
    main()
