#!/usr/bin/env python3
"""
Train a low-rank MLP routing predictor (variant B) and emit an LRPD binary
that the C++ loader (ggml-routing-predictor.cpp load_model) reads.

Input:  training_data.bin produced by collect_router_features.py
        header: magic "RPDS"(0x52504453), version(2), int32 num_experts, int32 horizon
        row:    int32 layer, int32 token_id,
                float32[num_experts] router_logits,
                int8[num_experts]   future_mask (1 at future-selected ids)

Output: model.bin in LRPD format
        header: magic "LRPD"(0x4C525044), version(2),
                int32 input_dim, int32 rank, int32 num_experts
        body:   float32 down_weight[rank*input_dim],
                float32 down_bias[rank],
                float32 output_weight[num_experts*rank],
                float32 output_bias[num_experts]

Model: x -> GELU(W_down x + b_down) -> W_out h + b_out -> logits
       top-k of softmax(logits) predicts the experts at layer L+horizon.

Usage:
    python train_routing_predictor.py training_data.bin --output model.bin --rank 32 --epochs 200
"""

import argparse
import struct
import numpy as np

RPDS_MAGIC = 0x52504453  # "RPDS"
LRPD_MAGIC = 0x4C525044  # "LRPD"
LRPD_VERSION = 2


def load_training_data(path):
    with open(path, 'rb') as f:
        magic, version = struct.unpack('<II', f.read(8))
        assert magic == RPDS_MAGIC, f"bad magic {magic:#x}"
        assert version == 2, f"unsupported version {version}"
        num_experts, horizon = struct.unpack('<ii', f.read(8))
        row_size = 8 + num_experts * 4 + num_experts  # layer+token, floats, int8 mask
        rest = f.read()
    n_rows = len(rest) // row_size
    X = np.zeros((n_rows, num_experts), dtype=np.float32)
    Y = np.zeros((n_rows, num_experts), dtype=np.float32)
    for i in range(n_rows):
        off = i * row_size
        # layer, token_id (skip)
        logits = np.frombuffer(rest, dtype=np.float32, count=num_experts, offset=off + 8)
        mask = np.frombuffer(rest, dtype=np.int8, count=num_experts, offset=off + 8 + num_experts * 4)
        X[i] = logits
        Y[i] = mask.astype(np.float32)
    return X, Y, num_experts, horizon


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def train(X, Y, rank, epochs, lr, seed=0):
    rng = np.random.default_rng(seed)
    n, input_dim = X.shape
    num_experts = Y.shape[1]

    # Init: small random
    Wd = (rng.standard_normal((rank, input_dim)) * 0.02).astype(np.float32)
    bd = np.zeros(rank, dtype=np.float32)
    Wo = (rng.standard_normal((num_experts, rank)) * 0.02).astype(np.float32)
    bo = np.zeros(num_experts, dtype=np.float32)

    # Targets: distribution putting mass on future-selected experts
    # If a row has k positives, target prob = 1/k on each; else uniform.
    pos = Y.sum(axis=1, keepdims=True)
    pos = np.where(pos > 0, pos, 1.0)
    T = Y / pos
    # Rows with no positives: uniform target (keeps gradient bounded)
    none_mask = (Y.sum(axis=1) == 0).astype(np.float32)[:, None]
    T = T * (1 - none_mask) + (1.0 / num_experts) * none_mask

    batch = min(256, n)
    for ep in range(epochs):
        perm = rng.permutation(n)
        total_loss = 0.0
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            xb = X[idx]
            tb = T[idx]

            h = gelu(xb @ Wd.T + bd)               # [b, rank]
            logits = h @ Wo.T + bo                 # [b, num_experts]
            p = softmax(logits, axis=1)            # [b, num_experts]

            # cross-entropy loss against target distribution
            loss = -np.sum(tb * np.log(p + 1e-9)) / len(idx)
            total_loss += loss * len(idx)

            # gradient
            glogits = (p - tb) / len(idx)          # [b, num_experts]
            gWo = glogits.T @ h                    # [num_experts, rank]
            gbo = glogits.sum(axis=0)              # [num_experts]
            gh = glogits @ Wo                      # [b, rank]
            gpre = gh * (1.0 - np.tanh(0.7978845608 * (xb @ Wd.T + bd + 0.044715 * (xb @ Wd.T + bd) ** 3)) ** 2)
            gWd = gpre.T @ xb                      # [rank, input_dim]
            gbd = gpre.sum(axis=0)                 # [rank]

            Wo -= lr * gWo
            bo -= lr * gbo
            Wd -= lr * gWd
            bd -= lr * gbd

        if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0:
            k = min(8, num_experts)
            topk = np.argpartition(-logits, k - 1, axis=1)[:, :k]
            hits = 0.0
            for i in range(n):
                if Y[i].sum() == 0:
                    continue
                hits += len(set(topk[i].tolist()) & set(np.where(Y[i] > 0)[0].tolist())) / min(k, int(Y[i].sum()))
            denom = max(1, int((Y.sum(axis=1) > 0).sum()))
            print(f"  epoch {ep+1:4d}  loss={total_loss/n:.4f}  recall@8={hits/denom:.3f}")

    return Wd, bd, Wo, bo


def save_model(path, Wd, bd, Wo, bo, num_experts, rank):
    input_dim = Wd.shape[1]
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', LRPD_MAGIC, LRPD_VERSION))
        f.write(struct.pack('<iii', input_dim, rank, num_experts))
        # down_weight [rank, input_dim] row-major = Wd
        f.write(Wd.astype(np.float32).tobytes())
        f.write(bd.astype(np.float32).tobytes())
        # output_weight [num_experts, rank] row-major = Wo
        f.write(Wo.astype(np.float32).tobytes())
        f.write(bo.astype(np.float32).tobytes())
    print(f"Saved LRPD model to {path}")
    print(f"  input_dim={input_dim} rank={rank} num_experts={num_experts}")


def main():
    ap = argparse.ArgumentParser(description="Train low-rank MLP routing predictor")
    ap.add_argument("input", help="training_data.bin from collect_router_features.py")
    ap.add_argument("--output", default="model.bin", help="output LRPD model path")
    ap.add_argument("--rank", type=int, default=32, help="low-rank trunk dimension")
    ap.add_argument("--epochs", type=int, default=200, help="training epochs")
    ap.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    args = ap.parse_args()

    print(f"Loading training data from {args.input}")
    X, Y, num_experts, horizon = load_training_data(args.input)
    print(f"  {X.shape[0]} samples, num_experts={num_experts}, horizon={horizon}")

    if X.shape[0] == 0:
        print("No training samples; aborting")
        return 1

    print(f"Training (rank={args.rank}, epochs={args.epochs}, lr={args.lr})")
    Wd, bd, Wo, bo = train(X, Y, args.rank, args.epochs, args.lr)

    save_model(args.output, Wd, bd, Wo, bo, num_experts, args.rank)
    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
