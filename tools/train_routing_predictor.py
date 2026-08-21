#!/usr/bin/env python3
"""
Phase 5D: Train low-rank routing predictor model

This script trains a lightweight predictor that learns to predict expert routing
decisions from hidden states. The model uses a low-rank architecture to minimize
computational overhead during inference.

Architecture:
- Input: hidden_state (256 dims)
- Low-rank projection: 256 -> 32 -> 256
- Output: expert_logits (num_experts)
- Loss: cross-entropy on top-k experts

Usage:
    python train_routing_predictor.py <trace_file> --output model.bin
"""

import argparse
import struct
import numpy as np
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available, cannot train model")


# Sample structure (must match C++ definition)
SAMPLE_FORMAT = '<ii256f8fi'  # layer, token_id, hidden_state[256], expert_ids[8], n_experts
SAMPLE_SIZE = struct.calcsize(SAMPLE_FORMAT)


class RoutingDataset(Dataset):
    """Dataset for routing prediction training"""
    
    def __init__(self, trace_file):
        self.samples = []
        self._load_trace(trace_file)
    
    def _load_trace(self, trace_file):
        """Load binary trace file"""
        with open(trace_file, 'rb') as f:
            while True:
                data = f.read(SAMPLE_SIZE)
                if len(data) < SAMPLE_SIZE:
                    break
                
                fields = struct.unpack(SAMPLE_FORMAT, data)
                layer = fields[0]
                token_id = fields[1]
                hidden_state = np.array(fields[2:258], dtype=np.float32)
                expert_ids = np.array(fields[258:266], dtype=np.int64)
                n_experts = fields[266]
                
                # Only use samples with valid experts
                if n_experts > 0:
                    self.samples.append({
                        'layer': layer,
                        'token_id': token_id,
                        'hidden_state': hidden_state,
                        'expert_ids': expert_ids[:n_experts]
                    })
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'hidden_state': torch.tensor(sample['hidden_state'], dtype=torch.float32),
            'expert_ids': torch.tensor(sample['expert_ids'], dtype=torch.long)
        }


class LowRankPredictor(nn.Module):
    """
    Low-rank routing predictor
    
    Architecture:
    - Input projection: 256 -> rank
    - Output projection: rank -> 256
    - Final layer: 256 -> num_experts
    """
    
    def __init__(self, input_dim=256, rank=32, num_experts=128):
        super().__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.num_experts = num_experts
        
        # Low-rank bottleneck
        self.down_proj = nn.Linear(input_dim, rank, bias=False)
        self.up_proj = nn.Linear(rank, input_dim, bias=False)
        
        # Output projection
        self.output_proj = nn.Linear(input_dim, num_experts)
        
        # Activation
        self.act = nn.GELU()
    
    def forward(self, hidden_state):
        """
        Args:
            hidden_state: [batch, input_dim]
        
        Returns:
            logits: [batch, num_experts]
        """
        # Low-rank bottleneck
        x = self.down_proj(hidden_state)
        x = self.act(x)
        x = self.up_proj(x)
        x = self.act(x)
        
        # Output logits
        logits = self.output_proj(x)
        return logits


def train_model(dataset, num_experts=128, epochs=10, batch_size=256, lr=1e-3, rank=32):
    """Train the predictor model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Create model
    model = LowRankPredictor(input_dim=256, rank=rank, num_experts=num_experts).to(device)
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in loader:
            hidden_states = batch['hidden_state'].to(device)
            expert_ids = batch['expert_ids'].to(device)
            
            # Forward pass
            logits = model(hidden_states)
            
            # Compute loss on top-1 expert (simplified)
            # In practice, you might want to use top-k or weighted loss
            target = expert_ids[:, 0]  # Use first expert as target
            loss = criterion(logits, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item() * hidden_states.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == target).sum().item()
            total_samples += hidden_states.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, accuracy={accuracy:.4f}")
    
    return model


def save_model(model, output_path):
    """Save model in binary format for C++ loading"""
    
    # Extract weights
    down_weight = model.down_proj.weight.detach().cpu().numpy()  # [rank, input_dim]
    up_weight = model.up_proj.weight.detach().cpu().numpy()      # [input_dim, rank]
    output_weight = model.output_proj.weight.detach().cpu().numpy()  # [num_experts, input_dim]
    output_bias = model.output_proj.bias.detach().cpu().numpy()  # [num_experts]
    
    # Write binary format
    with open(output_path, 'wb') as f:
        # Header
        magic = 0x4C525044  # "LRPD" (Low-Rank Predictor)
        version = 1
        f.write(struct.pack('<II', magic, version))
        
        # Dimensions
        input_dim = model.input_dim
        rank = model.rank
        num_experts = model.num_experts
        f.write(struct.pack('<III', input_dim, rank, num_experts))
        
        # Weights (row-major)
        f.write(down_weight.astype(np.float32).tobytes())
        f.write(up_weight.astype(np.float32).tobytes())
        f.write(output_weight.astype(np.float32).tobytes())
        f.write(output_bias.astype(np.float32).tobytes())
    
    print(f"Model saved to {output_path}")
    print(f"  Input dim: {input_dim}")
    print(f"  Rank: {rank}")
    print(f"  Num experts: {num_experts}")
    print(f"  Total parameters: {down_weight.size + up_weight.size + output_weight.size + output_bias.size}")


def main():
    if not HAS_TORCH:
        print("Error: PyTorch is required for training")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='Train low-rank routing predictor')
    parser.add_argument('trace_file', help='Path to hidden state trace file')
    parser.add_argument('--output', '-o', default='predictor.bin', help='Output model path')
    parser.add_argument('--num-experts', type=int, default=128, help='Number of experts')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--rank', type=int, default=32, help='Low-rank dimension')
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.trace_file}...")
    dataset = RoutingDataset(args.trace_file)
    
    if len(dataset) == 0:
        print("Error: No samples in dataset")
        sys.exit(1)
    
    # Train model
    print(f"\nTraining model...")
    model = train_model(
        dataset,
        num_experts=args.num_experts,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rank=args.rank
    )
    
    # Save model
    print(f"\nSaving model...")
    save_model(model, args.output)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
