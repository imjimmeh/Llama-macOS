#!/usr/bin/env python3
"""
Generate synthetic routing traces for oracle simulator testing.

Simulates realistic MoE routing patterns with:
- Cross-layer correlation (experts at layer L predict experts at L+H)
- Temporal correlation (similar experts used across tokens)
- Top-K routing (K=8 experts per layer)

Usage:
    python generate-synthetic-trace.py --output trace.bin --tokens 1000 --layers 40 --experts 128
"""

import argparse
import struct
import random
import numpy as np
from pathlib import Path

# Trace format (must match C++ definition)
ENTRY_FORMAT = '<Qii64iQ'  # token_id, layer, n_experts, expert_ids[64], timestamp_us
MAGIC = 0x52545243  # "RTRC"
VERSION = 1


def generate_trace(n_tokens: int, n_layers: int, n_experts: int, top_k: int = 8,
                   correlation: float = 0.7, output_path: str = "trace.bin"):
    """
    Generate synthetic routing trace with realistic correlation patterns.
    
    Args:
        n_tokens: Number of tokens to generate
        n_layers: Number of MoE layers
        n_experts: Number of experts per layer
        top_k: Number of experts selected per layer
        correlation: Cross-layer correlation (0= random, 1= perfect correlation)
        output_path: Output binary file path
    """
    
    entries = []
    timestamp_us = 0
    
    # Generate base routing patterns for each layer
    # Each layer has a "preferred" subset of experts
    layer_preferences = []
    for layer in range(n_layers):
        # Each layer prefers ~30% of experts
        n_preferred = int(n_experts * 0.3)
        preferred = random.sample(range(n_experts), n_preferred)
        layer_preferences.append(preferred)
    
    # Generate per-token routing
    for token_id in range(n_tokens):
        # Each token has a "topic" that influences routing
        topic_experts = random.sample(range(n_experts), n_experts // 4)
        
        prev_layer_experts = []  # Track previous layer's experts for correlation
        
        for layer in range(n_layers):
            # Mix of:
            # - Layer preferences (40%)
            # - Topic influence (30%)
            # - Random (30%)
            candidates = []
            
            # Layer preferences
            n_from_layer = int(top_k * 0.4)
            candidates.extend(random.sample(layer_preferences[layer], min(n_from_layer, len(layer_preferences[layer]))))
            
            # Topic influence
            n_from_topic = int(top_k * 0.3)
            candidates.extend(random.sample(topic_experts, min(n_from_topic, len(topic_experts))))
            
            # Random fill
            while len(candidates) < top_k:
                expert = random.randint(0, n_experts - 1)
                if expert not in candidates:
                    candidates.append(expert)
            
            # Apply cross-layer correlation
            if correlation > 0 and layer > 0 and prev_layer_experts:
                # Some experts from previous layer carry over
                n_carry = int(top_k * correlation * 0.3)
                for i in range(min(n_carry, len(prev_layer_experts))):
                    if i < len(candidates):
                        candidates[i] = prev_layer_experts[i]
            
            # Ensure unique
            candidates = list(set(candidates))[:top_k]
            while len(candidates) < top_k:
                expert = random.randint(0, n_experts - 1)
                if expert not in candidates:
                    candidates.append(expert)
            
            # Pad to 64
            expert_ids = candidates + [0] * (64 - len(candidates))
            
            # Timestamp: ~1ms per layer (simulating compute time)
            timestamp_us += 1000
            
            entry = (token_id, layer, top_k, expert_ids, timestamp_us)
            entries.append(entry)
            
            # Track for next layer correlation
            prev_layer_experts = candidates
    
    # Write binary file
    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', MAGIC))
        f.write(struct.pack('<I', VERSION))
        
        # Entries
        for entry in entries:
            token_id, layer, n_experts, expert_ids, timestamp = entry
            f.write(struct.pack(ENTRY_FORMAT, token_id, layer, n_experts, *expert_ids, timestamp))
    
    print(f"Generated {len(entries)} trace entries ({n_tokens} tokens × {n_layers} layers)")
    print(f"Output: {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic routing trace')
    parser.add_argument('--output', '-o', default='synthetic-trace.bin', help='Output file path')
    parser.add_argument('--tokens', type=int, default=1000, help='Number of tokens')
    parser.add_argument('--layers', type=int, default=40, help='Number of MoE layers')
    parser.add_argument('--experts', type=int, default=128, help='Number of experts per layer')
    parser.add_argument('--top-k', type=int, default=8, help='Experts selected per layer')
    parser.add_argument('--correlation', type=float, default=0.7, help='Cross-layer correlation (0-1)')
    args = parser.parse_args()
    
    generate_trace(
        n_tokens=args.tokens,
        n_layers=args.layers,
        n_experts=args.experts,
        top_k=args.top_k,
        correlation=args.correlation,
        output_path=args.output
    )
