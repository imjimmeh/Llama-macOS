#!/usr/bin/env python3
"""
Collect router features for predictor training.

This script processes route traces and extracts router logits (or synthesizes them)
to create training data for the routing predictor.

Usage:
    python collect_router_features.py --input route_trace.bin --output training_data.bin
"""

import argparse
import struct
import numpy as np
from pathlib import Path


def load_route_trace(trace_path):
    """Load route trace from binary file."""
    with open(trace_path, 'rb') as f:
        # Read magic and version
        magic, version = struct.unpack('<II', f.read(8))
        assert magic == 0x52545243, f"Invalid trace magic: {magic:x}"
        assert version == 1, f"Unsupported trace version: {version}"
        
        # Read entries
        entries = []
        while True:
            data = f.read(4 + 4 + 4 + 64*4 + 8)  # token_id, layer, n_experts, expert_ids[64], timestamp
            if not data or len(data) < 4 + 4 + 4 + 64*4 + 8:
                break
            token_id, layer, n_experts = struct.unpack('<IIi', data[:12])
            expert_ids = struct.unpack('<64i', data[12:12+64*4])
            timestamp = struct.unpack('<Q', data[12+64*4:])[0]
            entries.append({
                'token_id': token_id,
                'layer': layer,
                'n_experts': n_experts,
                'expert_ids': expert_ids[:n_experts],
                'timestamp': timestamp
            })
    
    return entries


def synthesize_router_logits(expert_ids, num_experts=64):
    """
    Synthesize router logits from expert IDs.
    
    In a real implementation, we'd extract actual logits from the model.
    For now, create a one-hot-like distribution with the selected experts.
    """
    logits = np.zeros(num_experts, dtype=np.float32)
    
    # Set high logits for selected experts
    for eid in expert_ids:
        if 0 <= eid < num_experts:
            logits[eid] = 5.0  # High logit for selected experts
    
    # Add small random noise to non-selected experts
    for i in range(num_experts):
        if logits[i] == 0:
            logits[i] = np.random.uniform(-2.0, 0.0)
    
    return logits


def generate_training_data(entries, sample_rate=8, num_experts=64):
    """Generate training samples from route entries."""
    samples = []
    
    # Group entries by token_id
    token_entries = {}
    for entry in entries:
        tid = entry['token_id']
        if tid not in token_entries:
            token_entries[tid] = {}
        token_entries[tid][entry['layer']] = entry
    
    # Sample tokens
    token_ids = sorted(token_entries.keys())
    sampled_tokens = token_ids[::sample_rate]
    
    for tid in sampled_tokens:
        layers = token_entries[tid]
        
        # For each layer L, collect features and future routes
        for layer_l in sorted(layers.keys()):
            entry_l = layers[layer_l]
            
            # Features: router logits at layer L
            features = synthesize_router_logits(entry_l['expert_ids'], num_experts)
            
            # Labels: future routes at L+4, L+6, L+8, L+10, L+12
            future_routes = []
            for h in [4, 6, 8, 10, 12]:
                layer_future = layer_l + h
                if layer_future in layers:
                    future_experts = layers[layer_future]['expert_ids'][:8]
                    # Pad to 8 experts
                    future_experts = list(future_experts) + [-1] * (8 - len(future_experts))
                    future_routes.append(future_experts)
                else:
                    future_routes.append([-1] * 8)
            
            samples.append({
                'layer': layer_l,
                'token_id': tid,
                'features': features,
                'future_routes': future_routes
            })
    
    return samples


def save_training_data(samples, output_path, num_experts=64):
    """Save training samples to binary file."""
    with open(output_path, 'wb') as f:
        # Write header
        magic = 0x52504453  # "RPDS"
        version = 1
        f.write(struct.pack('<II', magic, version))
        
        # Write samples
        for sample in samples:
            f.write(struct.pack('<ii', sample['layer'], sample['token_id']))
            f.write(sample['features'].tobytes())  # num_experts floats
            for future_experts in sample['future_routes']:
                f.write(struct.pack('<8i', *future_experts))
    
    print(f"Saved {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect router features for predictor training")
    parser.add_argument("--input", type=str, required=True, help="Input route trace file")
    parser.add_argument("--output", type=str, required=True, help="Output training data file")
    parser.add_argument("--sample-rate", type=int, default=8, help="Sample 1-in-N tokens")
    parser.add_argument("--num-experts", type=int, default=64, help="Number of experts")
    args = parser.parse_args()
    
    print(f"Loading route trace from {args.input}")
    entries = load_route_trace(args.input)
    print(f"Loaded {len(entries)} entries")
    
    print(f"Generating training data (sample rate: 1/{args.sample_rate})")
    samples = generate_training_data(entries, args.sample_rate, args.num_experts)
    
    print(f"Saving to {args.output}")
    save_training_data(samples, args.output, args.num_experts)
    
    print("Done")


if __name__ == "__main__":
    main()
