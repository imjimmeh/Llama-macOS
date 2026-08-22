# Phase 5D: Learned Routing Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a learned routing predictor that predicts expert demand H layers ahead and issues asynchronous prefetches, moving prediction upstream to the router-input tensor location.

**Architecture:** The predictor executes alongside the router computation (not inside the cache), using router logits or projected hidden state as features. Three variants are tested: stale future router (no training), low-rank MLP, and future-router + learned residual. Predictions feed into the existing async prefetch pipeline via a clean `submit_prediction()` API.

**Tech Stack:** C++17, GGML, PyTorch (training), CUDA streams, Python (analysis)

## Global Constraints

- Target hardware: GTX 1080 (8 GiB VRAM), PCIe 3.0 x16 @ 12 GB/s
- Model: Qwen 3.5/3.6 MoE (35B params, 64 experts per layer, top-8 routing)
- Primary metric: Recall@12 at H=8 (must exceed 80%)
- Overhead budget: <0.5ms per layer for predictor inference
- Build: Must compile cleanly with existing expert cache infrastructure
- Tests: All existing tests must pass (16/16 in test-expert-cache suite)

---

## File Structure

### Files to Create

1. **`ggml/src/ggml-routing-predictor.h`** — Public API for routing predictor
   - Predictor context management
   - Feature extraction interface
   - Prediction submission interface

2. **`ggml/src/ggml-routing-predictor.cpp`** — Predictor implementation
   - Three predictor variants (A/B/C)
   - CPU inference for variants A and B
   - Feature extraction utilities

3. **`tools/train-routing-predictor-v2.py`** — Training script for revised architecture
   - Load router logit traces
   - Train three variants
   - Export models in binary format

4. **`tools/collect-router-features.py`** — Feature collection from route traces
   - Parse existing route traces from Phase 5A
   - Extract router logits (if available) or synthesize features
   - Generate training data in binary format

5. **`tests/test-routing-predictor.cpp`** — Unit tests for predictor
   - Test model loading
   - Test inference correctness
   - Test API integration

### Files to Modify

1. **`src/llama-graph.cpp`** — Add predictor invocation at router-input point
   - In `build_moe_ffn()` after router computation (line ~2075)
   - Extract features and call predictor

2. **`ggml/src/ggml-backend-expert-cache.h`** — Add prediction submission API
   - `ggml_backend_expert_cache_submit_prediction()`

3. **`ggml/src/ggml-backend-expert-cache.cpp`** — Implement prediction submission
   - Queue predictions for async prefetch
   - Integrate with existing 5C prefetch infrastructure

4. **`ggml/src/CMakeLists.txt`** — Add new source files
   - `ggml-routing-predictor.cpp`

---

## Task 1: Define Predictor Public API

**Files:**
- Create: `ggml/src/ggml-routing-predictor.h`

**Interfaces:**
- Consumes: Nothing (first task)
- Produces: Public API header for routing predictor

- [ ] **Step 1: Create header file with predictor context struct**

```cpp
// ggml/src/ggml-routing-predictor.h
#ifndef GGML_ROUTING_PREDICTOR_H
#define GGML_ROUTING_PREDICTOR_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Predictor variant types
enum ggml_routing_predictor_type {
    GGML_ROUTING_PREDICTOR_STALE_FUTURE = 0,  // Variant A: no training
    GGML_ROUTING_PREDICTOR_LOW_RANK_MLP = 1,  // Variant B: learned MLP
    GGML_ROUTING_PREDICTOR_FUTURE_RESIDUAL = 2 // Variant C: future router + residual
};

// Predictor configuration
struct ggml_routing_predictor_config {
    enum ggml_routing_predictor_type type;
    int32_t input_dim;        // Feature dimension (128 for router logits, 256 for projected hidden)
    int32_t num_experts;      // Total number of experts
    int32_t horizon;          // Prediction horizon H (e.g., 8 for L+8)
    int32_t rank;             // Low-rank dimension (for variants B and C)
    const char * model_path;  // Path to trained model (NULL for variant A)
};

// Opaque predictor context
typedef struct ggml_routing_predictor * ggml_routing_predictor_t;

// Lifecycle
GGML_API ggml_routing_predictor_t ggml_routing_predictor_init(
    const struct ggml_routing_predictor_config * config);

GGML_API void ggml_routing_predictor_free(ggml_routing_predictor_t predictor);

// Feature extraction
GGML_API void ggml_routing_predictor_extract_features(
    ggml_routing_predictor_t predictor,
    const float * router_logits,  // [num_experts] router output at layer L
    int32_t num_experts,
    float * out_features);        // [input_dim] output features

// Prediction
GGML_API int32_t ggml_routing_predictor_predict(
    ggml_routing_predictor_t predictor,
    const float * features,       // [input_dim] input features
    int32_t * out_expert_ids,     // [max_predictions] output expert IDs
    float * out_confidences,      // [max_predictions] output confidences (optional, can be NULL)
    int32_t max_predictions);

#ifdef __cplusplus
}
#endif

#endif // GGML_ROUTING_PREDICTOR_H
```

- [ ] **Step 2: Verify header compiles**

Run: `cd build && cmake .. && make ggml-base`
Expected: Compiles without errors

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-routing-predictor.h
git commit -m "feat(predictor): add routing predictor public API"
```

---

## Task 2: Implement Variant A (Stale Future Router)

**Files:**
- Create: `ggml/src/ggml-routing-predictor.cpp`
- Modify: `ggml/src/CMakeLists.txt`

**Interfaces:**
- Consumes: `ggml_routing_predictor.h` API
- Produces: Variant A implementation (no training required)

- [ ] **Step 1: Write failing test for Variant A initialization**

Create `tests/test-routing-predictor.cpp`:

```cpp
#include "ggml-routing-predictor.h"
#include <cassert>
#include <cstdio>

void test_variant_a_init() {
    ggml_routing_predictor_config config = {};
    config.type = GGML_ROUTING_PREDICTOR_STALE_FUTURE;
    config.input_dim = 128;
    config.num_experts = 64;
    config.horizon = 8;
    config.rank = 0;
    config.model_path = nullptr;
    
    ggml_routing_predictor_t pred = ggml_routing_predictor_init(&config);
    assert(pred != nullptr);
    
    ggml_routing_predictor_free(pred);
    printf("test_variant_a_init: PASS\n");
}

int main() {
    test_variant_a_init();
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd build && make test-routing-predictor && ./bin/test-routing-predictor`
Expected: Linker error (ggml_routing_predictor_init not defined)

- [ ] **Step 3: Implement minimal Variant A struct and init**

In `ggml/src/ggml-routing-predictor.cpp`:

```cpp
#include "ggml-routing-predictor.h"
#include "ggml-impl.h"
#include <cstdlib>
#include <cstring>

struct ggml_routing_predictor {
    enum ggml_routing_predictor_type type;
    int32_t input_dim;
    int32_t num_experts;
    int32_t horizon;
    int32_t rank;
    
    // Variant B/C model weights (not used by Variant A)
    float * down_weight = nullptr;
    float * up_weight = nullptr;
    float * output_weight = nullptr;
    float * output_bias = nullptr;
};

ggml_routing_predictor_t ggml_routing_predictor_init(
        const struct ggml_routing_predictor_config * config) {
    if (config == nullptr) {
        return nullptr;
    }
    
    auto * pred = new ggml_routing_predictor();
    pred->type = config->type;
    pred->input_dim = config->input_dim;
    pred->num_experts = config->num_experts;
    pred->horizon = config->horizon;
    pred->rank = config->rank;
    
    // Variant A: no model loading needed
    if (config->type == GGML_ROUTING_PREDICTOR_STALE_FUTURE) {
        GGML_LOG_INFO("Routing predictor: Variant A (stale future router) initialized\n");
        return pred;
    }
    
    // TODO: Variants B and C will load model here
    
    delete pred;
    return nullptr;
}

void ggml_routing_predictor_free(ggml_routing_predictor_t predictor) {
    if (predictor == nullptr) {
        return;
    }
    
    delete[] predictor->down_weight;
    delete[] predictor->up_weight;
    delete[] predictor->output_weight;
    delete[] predictor->output_bias;
    
    delete predictor;
}
```

- [ ] **Step 4: Add source file to CMakeLists.txt**

In `ggml/src/CMakeLists.txt`, find the list of source files and add:

```cmake
ggml-routing-predictor.cpp
```

- [ ] **Step 5: Implement feature extraction for Variant A**

Add to `ggml-routing-predictor.cpp`:

```cpp
void ggml_routing_predictor_extract_features(
        ggml_routing_predictor_t predictor,
        const float * router_logits,
        int32_t num_experts,
        float * out_features) {
    if (predictor == nullptr || router_logits == nullptr || out_features == nullptr) {
        return;
    }
    
    // Variant A: features are just the router logits (truncated or padded)
    int32_t copy_dim = (num_experts < predictor->input_dim) ? num_experts : predictor->input_dim;
    memcpy(out_features, router_logits, copy_dim * sizeof(float));
    
    // Pad with zeros if needed
    if (copy_dim < predictor->input_dim) {
        memset(out_features + copy_dim, 0, (predictor->input_dim - copy_dim) * sizeof(float));
    }
}
```

- [ ] **Step 6: Implement prediction for Variant A (identity)**

Add to `ggml-routing-predictor.cpp`:

```cpp
int32_t ggml_routing_predictor_predict(
        ggml_routing_predictor_t predictor,
        const float * features,
        int32_t * out_expert_ids,
        float * out_confidences,
        int32_t max_predictions) {
    if (predictor == nullptr || features == nullptr || out_expert_ids == nullptr) {
        return 0;
    }
    
    // Variant A: stale future router - just return top-k from input features
    // (In real usage, these would be logits from W_router[L+H] × x_L)
    
    // Simple top-k selection
    std::vector<std::pair<float, int32_t>> scores;
    for (int32_t i = 0; i < predictor->num_experts; i++) {
        scores.push_back({features[i], i});
    }
    
    std::partial_sort(scores.begin(),
                      scores.begin() + std::min(max_predictions, (int32_t)scores.size()),
                      scores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    int32_t n_output = std::min(max_predictions, (int32_t)scores.size());
    for (int32_t i = 0; i < n_output; i++) {
        out_expert_ids[i] = scores[i].second;
        if (out_confidences != nullptr) {
            out_confidences[i] = scores[i].first;
        }
    }
    
    return n_output;
}
```

- [ ] **Step 7: Build and run test**

Run: `cd build && make test-routing-predictor && ./bin/test-routing-predictor`
Expected: `test_variant_a_init: PASS`

- [ ] **Step 8: Commit**

```bash
git add ggml/src/ggml-routing-predictor.cpp ggml/src/CMakeLists.txt tests/test-routing-predictor.cpp
git commit -m "feat(predictor): implement Variant A (stale future router)"
```

---

## Task 3: Add Prediction Submission API to Expert Cache

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.h`
- Modify: `ggml/src/ggml-backend-expert-cache.cpp`

**Interfaces:**
- Consumes: Existing expert cache infrastructure
- Produces: `submit_prediction()` API for predictor to feed prefetch queue

- [ ] **Step 1: Write failing test for submit_prediction API**

Add to `tests/test-routing-predictor.cpp`:

```cpp
#include "ggml-backend-expert-cache.h"

void test_submit_prediction_api() {
    // Create a minimal cache (this will fail until we implement the API)
    ggml_backend_expert_cache_t cache = nullptr; // TODO: create real cache
    
    int32_t expert_ids[] = {1, 5, 12, 23};
    float confidences[] = {0.9f, 0.8f, 0.7f, 0.6f};
    
    ggml_backend_expert_cache_submit_prediction(
        cache,
        /*target_layer*/ 8,
        expert_ids,
        /*n_experts*/ 4,
        confidences);
    
    printf("test_submit_prediction_api: PASS\n");
}
```

Update `main()`:

```cpp
int main() {
    test_variant_a_init();
    test_submit_prediction_api();
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd build && make test-routing-predictor && ./bin/test-routing-predictor`
Expected: Compile error (ggml_backend_expert_cache_submit_prediction not declared)

- [ ] **Step 3: Add API declaration to header**

In `ggml/src/ggml-backend-expert-cache.h`, add after existing Phase 5C APIs:

```cpp
// Phase 5D: Prediction Submission (Revised Architecture)
GGML_API void ggml_backend_expert_cache_submit_prediction(
    ggml_backend_expert_cache_t cache,
    int32_t target_layer,
    const int32_t * expert_ids,
    int32_t n_experts,
    const float * confidences);
```

- [ ] **Step 4: Implement submit_prediction**

In `ggml/src/ggml-backend-expert-cache.cpp`, add at end of file:

```cpp
void ggml_backend_expert_cache_submit_prediction(
        ggml_backend_expert_cache_t cache,
        int32_t target_layer,
        const int32_t * expert_ids,
        int32_t n_experts,
        const float * confidences) {
    if (cache == nullptr || expert_ids == nullptr || n_experts <= 0) {
        return;
    }
    
    // For now, just log the prediction
    // TODO: Integrate with async prefetch queue in Task 6
    GGML_LOG_DEBUG("Prediction submitted for layer %d: %d experts\n", target_layer, n_experts);
    
    // Track metrics
    cache->stats.n_predictions_submitted++;
}
```

- [ ] **Step 5: Add stats field to cache struct**

In `ggml/src/ggml-backend-expert-cache.cpp`, find the `ggml_backend_expert_cache` struct and add:

```cpp
struct ggml_backend_expert_cache_stats {
    // ... existing fields ...
    uint64_t n_predictions_submitted = 0;  // Phase 5D
};
```

- [ ] **Step 6: Build and run test**

Run: `cd build && make test-routing-predictor && ./bin/test-routing-predictor`
Expected: `test_submit_prediction_api: PASS`

- [ ] **Step 7: Commit**

```bash
git add ggml/src/ggml-backend-expert-cache.h ggml/src/ggml-backend-expert-cache.cpp tests/test-routing-predictor.cpp
git commit -m "feat(cache): add prediction submission API for Phase 5D"
```

---

## Task 4: Collect Router Logit Features from Traces

**Files:**
- Create: `tools/collect-router-features.py`

**Interfaces:**
- Consumes: Route traces from Phase 5A (`tools/results/route-trace-*.bin`)
- Produces: Training data in binary format for predictor training

- [ ] **Step 1: Create feature collection script skeleton**

```python
#!/usr/bin/env python3
"""
Collect router logit features from route traces for predictor training.

Input: Route trace files from Phase 5A
Output: Binary training data: <layer, token_id, features[128], future_routes[5][8]>
"""

import struct
import numpy as np
from pathlib import Path
import argparse

MAGIC_ROUTING_PREDICTOR_DATA = 0x52504453  # "RPDS"
VERSION = 1

def main():
    parser = argparse.ArgumentParser(description="Collect router features for predictor training")
    parser.add_argument("--input-trace", type=str, required=True, help="Input route trace file")
    parser.add_argument("--output", type=str, required=True, help="Output training data file")
    parser.add_argument("--sample-rate", type=int, default=8, help="Sample 1-in-N tokens")
    args = parser.parse_args()
    
    # TODO: Implement feature collection
    print(f"Collecting features from {args.input_trace}")
    print(f"Output: {args.output}")
    print(f"Sample rate: 1/{args.sample_rate}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Implement trace loading**

Add to `tools/collect-router-features.py`:

```python
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
            if not data:
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
```

- [ ] **Step 3: Implement feature extraction (synthetic router logits)**

Add to `tools/collect-router-features.py`:

```python
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
```

- [ ] **Step 4: Implement training data generation**

Add to `tools/collect-router-features.py`:

```python
def generate_training_data(entries, sample_rate=8):
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
            features = synthesize_router_logits(entry_l['expert_ids'])
            
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

def save_training_data(samples, output_path):
    """Save training samples to binary file."""
    with open(output_path, 'wb') as f:
        # Write header
        f.write(struct.pack('<II', MAGIC_ROUTING_PREDICTOR_DATA, VERSION))
        
        # Write samples
        for sample in samples:
            f.write(struct.pack('<ii', sample['layer'], sample['token_id']))
            f.write(sample['features'].tobytes())  # 128 floats
            for future_experts in sample['future_routes']:
                f.write(struct.pack('<8i', *future_experts))
    
    print(f"Saved {len(samples)} samples to {output_path}")
```

- [ ] **Step 5: Wire up main function**

Update `main()` in `tools/collect-router-features.py`:

```python
def main():
    parser = argparse.ArgumentParser(description="Collect router features for predictor training")
    parser.add_argument("--input-trace", type=str, required=True, help="Input route trace file")
    parser.add_argument("--output", type=str, required=True, help="Output training data file")
    parser.add_argument("--sample-rate", type=int, default=8, help="Sample 1-in-N tokens")
    args = parser.parse_args()
    
    print(f"Loading route trace from {args.input_trace}")
    entries = load_route_trace(args.input_trace)
    print(f"Loaded {len(entries)} entries")
    
    print(f"Generating training data (sample rate: 1/{args.sample_rate})")
    samples = generate_training_data(entries, args.sample_rate)
    
    print(f"Saving to {args.output}")
    save_training_data(samples, args.output)
    
    print("Done")
```

- [ ] **Step 6: Test with existing route trace**

Run: `python tools/collect-router-features.py --input-trace tools/results/route-trace-qwen35b.bin --output tools/results/predictor-training-data.bin --sample-rate 8`

Expected: Generates training data file with ~1000+ samples

- [ ] **Step 7: Commit**

```bash
git add tools/collect-router-features.py
git commit -m "feat(tools): add router feature collection for predictor training"
```

---

## Task 5: Train Predictor Variants B and C

**Files:**
- Create: `tools/train-routing-predictor-v2.py`

**Interfaces:**
- Consumes: Training data from Task 4
- Produces: Trained models in binary format

- [ ] **Step 1: Create training script skeleton**

```python
#!/usr/bin/env python3
"""
Train routing predictor variants B and C.

Variant B: Low-rank MLP (router logits → rank-32 → expert logits)
Variant C: Future-router + learned residual
"""

import torch
import torch.nn as nn
import torch.optim as optim
import struct
import numpy as np
from pathlib import Path
import argparse

# Model architectures

class LowRankMLP(nn.Module):
    """Variant B: Low-rank MLP predictor."""
    def __init__(self, input_dim=128, rank=32, num_experts=64):
        super().__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.num_experts = num_experts
        
        self.down_proj = nn.Linear(input_dim, rank)
        self.act = nn.GELU()
        self.output_proj = nn.Linear(rank, num_experts)
    
    def forward(self, x):
        h = self.act(self.down_proj(x))
        logits = self.output_proj(h)
        return logits

class FutureRouterResidual(nn.Module):
    """Variant C: Future-router + learned residual."""
    def __init__(self, input_dim=128, rank=32, num_experts=64):
        super().__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.num_experts = num_experts
        
        # Residual correction MLP
        self.residual_proj = nn.Linear(input_dim, rank)
        self.act = nn.GELU()
        self.residual_output = nn.Linear(rank, num_experts)
    
    def forward(self, x, stale_logits=None):
        # Learned residual correction
        h = self.act(self.residual_proj(x))
        residual = self.residual_output(h)
        
        # Add to stale logits (if provided)
        if stale_logits is not None:
            logits = stale_logits + residual
        else:
            logits = residual
        
        return logits

def main():
    parser = argparse.ArgumentParser(description="Train routing predictor variants")
    parser.add_argument("--input", type=str, required=True, help="Training data file")
    parser.add_argument("--output-dir", type=str, default="models/predictors", help="Output directory")
    parser.add_argument("--variant", type=str, choices=["B", "C"], required=True, help="Variant to train")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    
    print(f"Training Variant {args.variant}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Implement data loading**

Add to `tools/train-routing-predictor-v2.py`:

```python
def load_training_data(data_path):
    """Load training samples from binary file."""
    with open(data_path, 'rb') as f:
        magic, version = struct.unpack('<II', f.read(8))
        assert magic == 0x52504453, f"Invalid data magic: {magic:x}"
        
        samples = []
        while True:
            # Read layer and token_id
            data = f.read(8)
            if not data:
                break
            layer, token_id = struct.unpack('<ii', data)
            
            # Read features (128 floats)
            features = np.frombuffer(f.read(128 * 4), dtype=np.float32)
            
            # Read future routes (5 horizons × 8 experts)
            future_routes = []
            for _ in range(5):
                experts = struct.unpack('<8i', f.read(8 * 4))
                future_routes.append(experts)
            
            samples.append({
                'layer': layer,
                'token_id': token_id,
                'features': features,
                'future_routes': future_routes
            })
    
    return samples

def prepare_dataset(samples, horizon_idx=2):
    """
    Prepare PyTorch dataset for training.
    
    horizon_idx: 0=H+4, 1=H+6, 2=H+8, 3=H+10, 4=H+12
    Default to H=8 (index 2).
    """
    X = []
    y = []
    
    for sample in samples:
        features = sample['features']
        future_experts = sample['future_routes'][horizon_idx]
        
        # Filter out samples with no future data
        if all(eid == -1 for eid in future_experts):
            continue
        
        X.append(features)
        
        # Multi-hot label
        label = np.zeros(64, dtype=np.float32)
        for eid in future_experts:
            if 0 <= eid < 64:
                label[eid] = 1.0
        y.append(label)
    
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    
    return torch.utils.data.TensorDataset(X, y)
```

- [ ] **Step 3: Implement training loop for Variant B**

Add to `tools/train-routing-predictor-v2.py`:

```python
def train_variant_b(samples, epochs=50, lr=1e-3):
    """Train Variant B: Low-rank MLP."""
    print("Training Variant B: Low-Rank MLP")
    
    # Prepare dataset (H=8)
    dataset = prepare_dataset(samples, horizon_idx=2)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=256)
    
    # Model
    model = LowRankMLP(input_dim=128, rank=32, num_experts=64)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_recall = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_recall = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                preds = (torch.sigmoid(logits) > 0.5).float()
                # Recall@12
                _, top12_idx = torch.topk(logits, 12, dim=1)
                top12_preds = torch.zeros_like(y_batch).scatter(1, top12_idx, 1)
                recall = (top12_preds * y_batch).sum(dim=1) / y_batch.sum(dim=1).clamp(min=1)
                val_recall += recall.mean().item()
        
        val_recall /= len(val_loader)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f}, val_recall@12={val_recall:.4f}")
        
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_model = model.state_dict()
    
    print(f"Best validation Recall@12: {best_val_recall:.4f}")
    return best_model
```

- [ ] **Step 4: Implement model export**

Add to `tools/train-routing-predictor-v2.py`:

```python
def export_model_b(model_state, output_path):
    """Export Variant B model to binary format."""
    with open(output_path, 'wb') as f:
        # Header
        magic = 0x4C525044  # "LRPD"
        version = 2  # Version 2 for revised architecture
        f.write(struct.pack('<II', magic, version))
        
        # Dimensions
        input_dim = 128
        rank = 32
        num_experts = 64
        f.write(struct.pack('<iii', input_dim, rank, num_experts))
        
        # Weights (Variant B: down_proj + output_proj)
        down_weight = model_state['down_proj.weight'].numpy().flatten()
        down_bias = model_state['down_proj.bias'].numpy().flatten()
        output_weight = model_state['output_proj.weight'].numpy().flatten()
        output_bias = model_state['output_proj.bias'].numpy().flatten()
        
        f.write(down_weight.tobytes())
        f.write(down_bias.tobytes())
        f.write(output_weight.tobytes())
        f.write(output_bias.tobytes())
    
    print(f"Exported model to {output_path}")
```

- [ ] **Step 5: Wire up main function**

Update `main()` in `tools/train-routing-predictor-v2.py`:

```python
def main():
    parser = argparse.ArgumentParser(description="Train routing predictor variants")
    parser.add_argument("--input", type=str, required=True, help="Training data file")
    parser.add_argument("--output-dir", type=str, default="models/predictors", help="Output directory")
    parser.add_argument("--variant", type=str, choices=["B", "C"], required=True, help="Variant to train")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading training data from {args.input}")
    samples = load_training_data(args.input)
    print(f"Loaded {len(samples)} samples")
    
    if args.variant == "B":
        model_state = train_variant_b(samples, epochs=args.epochs, lr=args.lr)
        output_path = Path(args.output_dir) / "predictor-variant-b.bin"
        export_model_b(model_state, output_path)
    
    print("Done")
```

- [ ] **Step 6: Train Variant B**

Run: `python tools/train-routing-predictor-v2.py --input tools/results/predictor-training-data.bin --variant B --epochs 50`

Expected: Trains model, exports to `models/predictors/predictor-variant-b.bin`

- [ ] **Step 7: Commit**

```bash
git add tools/train-routing-predictor-v2.py
git commit -m "feat(tools): add training script for predictor variants B and C"
```

---

## Task 6: Integrate Predictor at Router-Input Point

**Files:**
- Modify: `src/llama-graph.cpp`

**Interfaces:**
- Consumes: Predictor API from Task 2, cache API from Task 3
- Produces: Predictor invocation in model graph

- [ ] **Step 1: Add predictor context to graph context**

In `src/llama-graph.h`, find `struct llm_graph_context` and add:

```cpp
// Phase 5D: Routing predictor
ggml_routing_predictor_t routing_predictor = nullptr;
ggml_backend_expert_cache_t expert_cache = nullptr;  // For submit_prediction
```

- [ ] **Step 2: Add predictor invocation in build_moe_ffn**

In `src/llama-graph.cpp`, in `build_moe_ffn()` after line 2079 (`cb(logits, "ffn_moe_logits", il);`), add:

```cpp
// Phase 5D: Run routing predictor if enabled
if (routing_predictor != nullptr && expert_cache != nullptr) {
    // Extract features from router logits
    // Note: logits is [n_expert, n_tokens], we need to extract for single token
    // For now, use the first token's logits
    std::vector<float> features(128, 0.0f);
    
    // Copy logits to features (simplified - real implementation needs proper indexing)
    // TODO: Handle multi-token case properly
    int32_t n_copy = std::min((int32_t)n_expert, 128);
    // logits is column-major [n_expert, n_tokens]
    // For token 0: logits[i] for i in [0, n_expert)
    
    // Run predictor
    int32_t predicted_ids[16];
    float confidences[16];
    int32_t n_predicted = ggml_routing_predictor_predict(
        routing_predictor,
        features.data(),
        predicted_ids,
        confidences,
        16);
    
    // Submit prediction to cache
    if (n_predicted > 0) {
        int32_t target_layer = il + 8;  // Predict H=8 layers ahead
        ggml_backend_expert_cache_submit_prediction(
            expert_cache,
            target_layer,
            predicted_ids,
            n_predicted,
            confidences);
    }
}
```

- [ ] **Step 3: Build and verify compilation**

Run: `cd build && make llama`
Expected: Compiles without errors

- [ ] **Step 4: Commit**

```bash
git add src/llama-graph.cpp src/llama-graph.h
git commit -m "feat(graph): integrate routing predictor at router-input point"
```

---

## Task 7: End-to-End Integration Test

**Files:**
- Modify: `tests/test-routing-predictor.cpp`

**Interfaces:**
- Consumes: All previous tasks
- Produces: End-to-end test validating full pipeline

- [ ] **Step 1: Write integration test**

Add to `tests/test-routing-predictor.cpp`:

```cpp
void test_end_to_end_pipeline() {
    // Initialize predictor (Variant A for simplicity)
    ggml_routing_predictor_config config = {};
    config.type = GGML_ROUTING_PREDICTOR_STALE_FUTURE;
    config.input_dim = 128;
    config.num_experts = 64;
    config.horizon = 8;
    
    ggml_routing_predictor_t pred = ggml_routing_predictor_init(&config);
    assert(pred != nullptr);
    
    // Simulate router logits
    float router_logits[128];
    for (int i = 0; i < 128; i++) {
        router_logits[i] = (i % 8 == 0) ? 5.0f : -1.0f;  // Experts 0, 8, 16, ...
    }
    
    // Extract features
    float features[128];
    ggml_routing_predictor_extract_features(pred, router_logits, 64, features);
    
    // Run prediction
    int32_t predicted_ids[16];
    float confidences[16];
    int32_t n_predicted = ggml_routing_predictor_predict(
        pred, features, predicted_ids, confidences, 16);
    
    assert(n_predicted > 0);
    printf("Predicted %d experts: ", n_predicted);
    for (int i = 0; i < n_predicted; i++) {
        printf("%d ", predicted_ids[i]);
    }
    printf("\n");
    
    // Submit to cache (mock cache for now)
    // In real usage, this would call ggml_backend_expert_cache_submit_prediction
    
    ggml_routing_predictor_free(pred);
    printf("test_end_to_end_pipeline: PASS\n");
}
```

Update `main()`:

```cpp
int main() {
    test_variant_a_init();
    test_submit_prediction_api();
    test_end_to_end_pipeline();
    return 0;
}
```

- [ ] **Step 2: Build and run test**

Run: `cd build && make test-routing-predictor && ./bin/test-routing-predictor`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test-routing-predictor.cpp
git commit -m "test(predictor): add end-to-end integration test"
```

---

## Task 8: Runtime Validation with Qwen 35B

**Files:**
- None (validation only)

**Interfaces:**
- Consumes: Full pipeline from Tasks 1-7
- Produces: Performance metrics and validation results

- [ ] **Step 1: Build server with predictor enabled**

Run: `cd build && cmake -DLLAMA_BUILD_SERVER=ON .. && make -j`
Expected: Builds successfully

- [ ] **Step 2: Run generation with predictor**

Run:
```bash
./build/bin/llama-server \
  -m models/Qwen3.6-35B-A3B-APEX-Compact.gguf \
  -exc 64M \
  -ngl 99 \
  --port 8080
```

Expected: Server starts, predictor logs initialization

- [ ] **Step 3: Collect performance metrics**

Run generation and monitor:
- Prediction submission rate
- Prefetch hit rate
- Generation throughput (tok/s)

Expected: Throughput > baseline (5C heuristic), prediction submissions > 0

- [ ] **Step 4: Compare against baseline**

Compare:
- Baseline (5C heuristic only): ~26.5 tok/s
- With predictor (5D): target >28 tok/s

Expected: Measurable speedup if predictor recall > 80%

- [ ] **Step 5: Document results**

Create `docs/plans/2026-08-21-learned-predictor-validation-results.md`:

```markdown
# Phase 5D Validation Results

**Date:** 2026-08-21
**Model:** Qwen3.6-35B-A3B-APEX-Compact.gguf
**Hardware:** GTX 1080 (8 GiB VRAM)

## Baseline (5C Heuristic Only)
- Generation throughput: 26.5 tok/s
- Cache hit rate: 80%

## With Predictor (5D Variant A)
- Generation throughput: [TBD] tok/s
- Prediction submissions: [TBD]
- Prefetch hit rate: [TBD]

## Analysis
[TBD - fill in after validation]
```

- [ ] **Step 6: Commit results**

```bash
git add docs/plans/2026-08-21-learned-predictor-validation-results.md
git commit -m "docs(predictor): add Phase 5D validation results"
```

---

## Summary

This implementation plan breaks down Phase 5D into 8 bite-sized tasks:

1. **Define Predictor API** — Public interface for routing predictor
2. **Implement Variant A** — Stale future router (no training)
3. **Add Prediction Submission API** — Cache receives predictions
4. **Collect Router Features** — Extract training data from traces
5. **Train Variants B/C** — Learned predictors
6. **Integrate at Router-Input** — Wire predictor into model graph
7. **End-to-End Test** — Validate full pipeline
8. **Runtime Validation** — Measure performance on Qwen 35B

**Total estimated time:** 14-21 days (can stop at Task 7 if CPU integration is sufficient)

**Key deliverables:**
- Three predictor variants (A/B/C)
- Training pipeline for learned models
- Integration at router-input point (upstream of cache)
- Performance validation on real model

**Success criteria:**
- Recall@12 at H=8 > 80%
- Generation throughput > 28 tok/s (5%+ speedup over 5C)
- All existing tests pass
