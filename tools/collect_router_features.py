#!/usr/bin/env python3
"""
Collect router features for predictor training.

Parses route trace files (v1 or v2) and emits supervised training rows for
the low-rank MLP routing predictor (variant B). Each row pairs a layer's
router logits with the expert ids observed `--horizon` layers later within
the same token.

Route trace format:
  header : uint32 magic (0x52545243 "RTRC"), uint32 version
  v1 entry: token_id(u64) layer(i32) n_experts(i32) expert_ids[64](i32) ts(u64)
  v2 entry: <v1 entry> + n_logits(i32) + float[n_logits]   (logits blob)

Output format (training_data.bin):
  header : uint32 magic (0x52504453 "RPDS"), uint32 version(2),
           int32 num_experts, int32 horizon
  row    : int32 layer, int32 token_id,
           float[num_experts] logits,
           int8[num_experts] future_mask   (1 at future-selected ids, else 0)

Usage:
    python collect_router_features.py --input route_trace.bin --output training_data.bin --horizon 8
"""

import argparse
import struct
import numpy as np
from pathlib import Path

TRACE_MAGIC = 0x52545243  # "RTRC"
ENTRY_FIXED_SIZE = 8 + 4 + 4 + 64 * 4 + 8  # token_id, layer, n_experts, ids[64], ts
OUT_MAGIC = 0x52504453  # "RPDS"
OUT_VERSION = 2


def load_route_trace(trace_path):
    """Load route trace (v1 or v2). Returns list of entry dicts."""
    with open(trace_path, 'rb') as f:
        magic, version = struct.unpack('<II', f.read(8))
        assert magic == TRACE_MAGIC, f"Invalid trace magic: {magic:x}"
        assert version in (1, 2), f"Unsupported trace version: {version}"

        entries = []
        while True:
            data = f.read(ENTRY_FIXED_SIZE)
            if not data or len(data) < ENTRY_FIXED_SIZE:
                break
            token_id, layer, n_experts = struct.unpack('<Qii', data[:16])
            expert_ids = struct.unpack('<64i', data[16:16 + 64 * 4])
            logits = None
            if version == 2:
                (n_logits,) = struct.unpack('<i', f.read(4))
                if n_logits > 0:
                    logits = np.frombuffer(f.read(4 * n_logits), dtype=np.float32).copy()
            entries.append({
                'token_id': token_id,
                'layer': layer,
                'n_experts': n_experts,
                'expert_ids': expert_ids[:n_experts],
                'logits': logits,
            })

    return entries


def synthesize_router_logits(expert_ids, num_experts=64):
    """One-hot-like fallback when a trace entry lacks real logits."""
    logits = np.zeros(num_experts, dtype=np.float32)
    for eid in expert_ids:
        if 0 <= eid < num_experts:
            logits[eid] = 5.0
    for i in range(num_experts):
        if logits[i] == 0:
            logits[i] = np.random.uniform(-2.0, 0.0)
    return logits


def generate_training_data(entries, horizon=8, num_experts=64, sample_rate=1):
    """Pair each layer's logits with the ids observed `horizon` layers later."""
    samples = []

    # Group entries by token_id -> {layer: entry}
    token_entries = {}
    for entry in entries:
        tid = entry['token_id']
        token_entries.setdefault(tid, {})[entry['layer']] = entry

    token_ids = sorted(token_entries.keys())
    sampled_tokens = token_ids[::sample_rate]

    for tid in sampled_tokens:
        layers = token_entries[tid]
        for layer_l in sorted(layers.keys()):
            layer_future = layer_l + horizon
            if layer_future not in layers:
                continue

            entry_l = layers[layer_l]
            if entry_l['logits'] is not None and len(entry_l['logits']) == num_experts:
                features = entry_l['logits']
            else:
                features = synthesize_router_logits(entry_l['expert_ids'], num_experts)

            future_ids = layers[layer_future]['expert_ids']
            future_mask = np.zeros(num_experts, dtype=np.int8)
            for eid in future_ids:
                if 0 <= eid < num_experts:
                    future_mask[eid] = 1

            samples.append({
                'layer': layer_l,
                'token_id': tid,
                'features': features,
                'future_mask': future_mask,
            })

    return samples


def save_training_data(samples, output_path, num_experts=64, horizon=8):
    """Save training rows to binary file."""
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<II', OUT_MAGIC, OUT_VERSION))
        f.write(struct.pack('<ii', num_experts, horizon))
        for sample in samples:
            f.write(struct.pack('<ii', sample['layer'], sample['token_id']))
            f.write(sample['features'].astype(np.float32).tobytes())
            f.write(sample['future_mask'].tobytes())

    print(f"Saved {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect router features for predictor training")
    parser.add_argument("input", type=str, help="Input route trace file")
    parser.add_argument("output", type=str, help="Output training data file")
    parser.add_argument("--horizon", type=int, default=8, help="Layers ahead to pair as the target")
    parser.add_argument("--sample-rate", type=int, default=1, help="Sample 1-in-N tokens")
    parser.add_argument("--num-experts", type=int, default=64, help="Number of experts")
    args = parser.parse_args()

    print(f"Loading route trace from {args.input}")
    entries = load_route_trace(args.input)
    print(f"Loaded {len(entries)} entries")

    print(f"Generating training data (horizon={args.horizon}, sample rate: 1/{args.sample_rate})")
    samples = generate_training_data(entries, args.horizon, args.num_experts, args.sample_rate)

    print(f"Saving to {args.output}")
    save_training_data(samples, args.output, args.num_experts, args.horizon)

    print("Done")


if __name__ == "__main__":
    main()
