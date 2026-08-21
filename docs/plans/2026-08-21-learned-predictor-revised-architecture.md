# Phase 5D: Learned Routing Predictor (Revised Architecture)

**Date:** 2026-08-21  
**Status:** Planning  
**Branch:** `expert-cache`  
**Target:** GTX 1080 (8 GiB VRAM) with Qwen 35B MoE

---

## Executive Summary

The current Phase 5D implementation is blocked because the learned predictor requires hidden state data, but the `mul_mat_id` execution path only has access to expert weights. The architectural insight from recent research (SpecPrefetch, PROBE) is:

**Don't drag hidden state down into the cache scheduler. Move prediction upstream to where the router input exists.**

This revised plan relocates the predictor from inside the expert cache to alongside the router computation, using router logits or projected hidden state as features instead of raw hidden state.

---

## Key Architectural Changes

### Current (Blocked) Architecture

```
router_input x_L
    ↓
native router → actual experts L
    ↓
MUL_MAT_ID (cache only sees expert weights here)
    ↓
[blocked: no hidden state access]
```

### Revised Architecture

```
router_input x_L
    │
    ├── native router ──────► actual experts L
    │
    └── predictor ──────────► predicted experts L+H
                                │
                                ▼
                         submit_prefetch_prediction()
                                │
                                ▼
                         async prefetch queue
```

**Key insight:** The predictor executes when `x_L` exists (at router input time), not when `MUL_MAT_ID` runs.

---

## Implementation Phases

### Phase 5D.1: Locate Router-Input Tensor (1-2 days)

**Objective:** Find the tensor feeding the router GEMV during graph construction.

**Tasks:**
- Search `src/llama-graph.cpp` for Qwen MoE router computation
- Identify the exact tensor in `build_moe_ffn()` that feeds `ggml_mul_mat(gate_inp, cur)`
- Verify tensor dimensions: should be `[n_embd, n_tokens]`
- Add instrumentation point immediately after router input is available

**Current location identified:**
```cpp
// src/llama-graph.cpp:2075
logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
```

The `cur` tensor at line 2044 is the router input. This is where the predictor should execute.

**Deliverable:** Code location confirmed, tensor reference identified.

---

### Phase 5D.2: Instrument Feature Collection (2-3 days)

**Objective:** Collect predictor inputs and future route labels without D2H bottleneck.

**Feature Options (cheapest to most expensive):**

**Option A: Router Logits (128 floats)** — RECOMMENDED FOR FIRST EXPERIMENT
```cpp
// Already computed during forward pass
logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
// Copy logits[128] to CPU for prediction
```
- Already available after router computation
- Compressed semantic representation
- No projection needed
- ~512 bytes per layer per token

**Option B: Fixed Random Projection (256 floats)**
```cpp
// Generate R ∈ ℝ^{256 × D} once at startup
// Compute z_L = R × x_L on-device
// Copy z_L[256] to CPU
```
- Requires adding projection tensor to graph
- ~1 KB per layer per token
- Reproducible with fixed seed

**Label Collection:**
- Collect actual routes at L+4, L+6, L+8, L+10, L+12
- Binary format: `<layer, token_id, features[128 or 256], future_routes[5][8]>`
- Sample 1-in-8 tokens to reduce storage (50k tokens × 40 layers × 128 floats ≈ 1 GB)

**Deliverable:** Trace collector producing training data.

---

### Phase 5D.3: Train Three Predictor Variants (3-4 days)

**Objective:** Compare three approaches, pick winner by Recall@K.

#### Variant A — Stale Future Router (no training)

```cpp
predicted_logits[L+H] = W_router[L+H] × x_L
```

**Implementation:**
- Use existing router weights from future layer
- Zero training cost
- Baseline: "how much is already encoded in earlier hidden states?"

**Expected result:** Moderate recall (60-70%) with zero training.

#### Variant B — Low-Rank MLP

```cpp
x_L → [D×32] → GELU → [32×E] → logits
```

**Architecture:**
- Input: router logits (128 floats) OR projected hidden state (256 floats)
- Hidden: rank 32
- Output: expert logits (128 experts)
- Multi-horizon heads: shared trunk, separate H=4/6/8/10/12 heads

**Parameters:**
```
128 × 32 = 4k (input projection)
32 × 128 × 5 = 20k (5 horizon heads)
Total: ~24k parameters
```

**Training:**
- Loss: BCE + ranking loss (not cross-entropy)
- Metric: Recall@12 at H=8
- Optimizer: AdamW, lr=1e-3
- Early stopping on validation recall

**Expected result:** High recall (80-90%) with minimal training.

#### Variant C — Future-Router + Residual (RECOMMENDED)

```cpp
predicted = W_router[L+H] × x_L + Δ_θ(x_L)
```

**Architecture:**
- Frozen prior: stale future router (Variant A)
- Learned residual: tiny MLP correction
- Strong structural prior + adaptive correction

**Parameters:**
```
Residual MLP: 128 → 32 → 128 = ~8k parameters
Total: ~8k trainable parameters
```

**Expected result:** Best of both worlds — strong prior + learned correction.

---

### Phase 5D.4: Evaluate Predictors (1-2 days)

**Objective:** Pick winner based on prefetch utility, not classification accuracy.

**Metrics:**
- Recall@12 at H=8 (can we fit 12 experts in cache?)
- Ready-recall (do predictions arrive before execution?)
- Precision (how many wasted prefetches?)

**Evaluation:**
- Run oracle simulator with predicted routes vs. perfect routes
- Compare: `prediction_recall × ready_rate × bytes_transferred`
- Target: >80% Recall@12 at H=8 to justify integration

**Decision criteria:**
- **Proceed** if Recall@12 > 80% at H=8
- **Iterate** if 60% < Recall@12 < 80% (tune architecture)
- **Stop** if Recall@12 < 60% (prediction signal too weak)

**Deliverable:** Selected model + performance projection.

---

### Phase 5D.5: Integrate Predictor at Router-Input Point (2-3 days)

**Objective:** Run predictor immediately after router input is computed.

**Architecture:**
```cpp
// In build_moe_ffn() after router computation
logits = build_lora_mm(gate_inp, cur); // native router

if (routing_predictor_enabled) {
    // Extract features (router logits or projected hidden state)
    // Run predictor for L+H
    // Call submit_prefetch_prediction(target_layer, expert_ids, confidences)
}
```

**Cache API addition:**
```cpp
void ggml_backend_expert_cache_submit_prediction(
    ggml_backend_expert_cache_t cache,
    int32_t target_layer,
    const int32_t * expert_ids,
    int32_t n_experts,
    const float * confidences);
```

**Key design:**
- Predictor lives in model graph code (alongside router)
- Cache just receives predictions via `submit_prediction()`
- Separation of concerns: predictor doesn't know about cache internals

**Deliverable:** Predictor integrated, feeding cache prefetch queue.

---

### Phase 5D.6: CPU Async Integration (2-3 days)

**Objective:** Validate end-to-end with CPU predictor thread.

**Architecture:**
```
GPU: router_input → feature extraction → z_L
        │
        └── async D2H (z_L only, ~512 bytes)
        
CPU predictor thread:
    z_L arrives → MLP → predicted IDs → prefetch queue
    
CUDA prefetch stream:
    host expert → GPU slot (async DMA)
```

**Implementation:**
- Tiny D2H copy of features (not full hidden state)
- Background CPU thread runs predictor
- Async prefetch to cache via existing 5C infrastructure
- **Nothing waits for prediction** — if too late, drop it

**Metrics:**
- `prediction_generated`, `prediction_too_late`, `prediction_used`
- `fully_hidden`, `partially_hidden`, `wasted_bytes`

**Deliverable:** Working CPU async predictor with telemetry.

---

### Phase 5D.7: GPU Predictor (Optional, if profitable) (3-4 days)

**Objective:** Move predictor to GPU if CPU latency is bottleneck.

**Architecture:**
```
compute_stream:    ──── block L ────────────────────────►
predictor_stream:       └── predictor H=8 ──► IDs
prefetch_stream:                              └── H2D experts
```

**Implementation:**
- Dedicated CUDA stream for predictor
- Device-side top-K selection
- Async ID copy to pinned host memory (don't sync compute stream)
- Event-based dependencies between streams

**Key constraint:**
- Do NOT synchronize compute stream to read IDs on CPU
- Keep predicted IDs on GPU, drive device-side cache lookup

**Deliverable:** GPU predictor with zero compute-stream synchronization.

---

## Training Data Format

### Binary Trace Format

```cpp
struct routing_predictor_sample {
    int32_t layer;              // source layer L
    int32_t token_id;           // token index
    float features[128];        // router logits OR projected hidden state
    int32_t future_routes[5][8]; // experts at L+4, L+6, L+8, L+10, L+12
};
```

**Magic:** `0x52504453` ("RPDS" = Routing Predictor Data Samples)  
**Version:** 1

### Data Collection Strategy

**Storage budget:**
- 50k tokens × 40 layers × 128 floats × 4 bytes = 1 GB (features only)
- With labels: ~1.5 GB total
- Sample 1-in-8 tokens: ~190 MB

**Split strategy (avoid leakage):**
- Train: 70% of prompts (whole conversations)
- Validation: 15% of unseen prompts
- Test: 15% of unseen prompts

**Do NOT randomly split samples** — adjacent tokens/layers are heavily correlated.

---

## Training Configuration

### Loss Function

```python
loss = BCE_loss + λ × ranking_loss

# BCE: multi-label classification
BCE = -Σ[y_e × log(σ(s_e)) + (1-y_e) × log(1-σ(s_e))]

# Ranking: encourage correct ordering
ranking_loss = Σ max(0, margin - (s_positive - s_negative))
```

### Evaluation Metrics

```python
Recall@K = |predicted_top_K ∩ actual_top_8| / |actual_top_8|

# Primary metric: Recall@12 at H=8
# Secondary: Recall@16 at H=8 (cache capacity headroom)
```

### Hyperparameters

```python
optimizer = AdamW(lr=1e-3, weight_decay=1e-4)
batch_size = 256
epochs = 50 (with early stopping patience=5)
scheduler = CosineAnnealingLR(T_max=epochs)
```

---

## Success Criteria

### Phase 5D.1-5D.4 (Offline)
- ✅ Router-input tensor located and instrumented
- ✅ Training data collected (1+ GB)
- ✅ Three variants trained and evaluated
- ✅ Winner selected with Recall@12 > 80% at H=8

### Phase 5D.5-5D.6 (CPU Integration)
- ✅ Predictor integrated at router-input point
- ✅ CPU async thread running predictor
- ✅ Telemetry shows: prediction_used > 50% of tokens
- ✅ Measured speedup > 10% over baseline (5C heuristic)

### Phase 5D.7 (GPU, Optional)
- ✅ GPU predictor adds <0.1ms overhead per layer
- ✅ Zero compute-stream synchronization
- ✅ Speedup approaches oracle (within 60% of theoretical max)

---

## Risk Mitigation

### Risk 1: Router Logits Carry Insufficient Information

**Mitigation:** Variant A (stale future router) tests this immediately. If recall < 60%, switch to Variant B (learned MLP) or Option B features (projected hidden state).

### Risk 2: CPU Predictor Too Slow

**Mitigation:** Phase 5D.7 moves predictor to GPU. GTX 1080 can run two GEMVs in <0.1ms.

### Risk 3: Prediction Signal Too Weak at H=8

**Mitigation:** Oracle simulator (5B) already confirmed H=8 gives 9.52x speedup with perfect prediction. If learned predictor achieves 80% recall, we get ~7.6x speedup.

### Risk 4: Training Data Leakage

**Mitigation:** Split by prompts, not samples. Evaluate by category (coding, conversation, reasoning) to detect overfitting.

---

## Timeline

| Sub-phase | Duration | Dependencies | Risk |
|-----------|----------|--------------|------|
| 5D.1 Locate router input | 1-2 days | None | Low |
| 5D.2 Instrument features | 2-3 days | 5D.1 | Low |
| 5D.3 Train 3 variants | 3-4 days | 5D.2 | Medium (training) |
| 5D.4 Evaluate & select | 1-2 days | 5D.3 | Low |
| 5D.5 Integrate at router | 2-3 days | 5D.4 | Medium (graph changes) |
| 5D.6 CPU async | 2-3 days | 5D.5 | Low |
| 5D.7 GPU predictor | 3-4 days | 5D.6 success | High (stream sync) |
| **Total** | **14-21 days** | | |

**Note:** Can stop at 5D.6 if CPU async is sufficient. 5D.7 is optional optimization.

---

## Comparison with Current 5D

| Aspect | Current 5D | Revised 5D |
|--------|-----------|-----------|
| **Predictor location** | Inside cache (blocked) | Alongside router (upstream) |
| **Features** | Hidden state (256 floats) | Router logits (128 floats) |
| **Model variants** | One MLP | Three variants (A/B/C) |
| **Training target** | H=1 only | Multi-horizon (H=4/6/8/10/12) |
| **Metric** | Classification accuracy | Recall@K for prefetching |
| **Integration** | Requires hidden state pass-through | Uses existing router computation |
| **Status** | Blocked | Unblocked, ready to implement |

---

## Immediate Next Steps

1. **Start with 5D.1:** Confirm router-input tensor location in `src/llama-graph.cpp`
2. **Implement 5D.2:** Add feature collection instrumentation
3. **Collect training data:** Generate 1+ GB of routing samples
4. **Train Variant A first:** Stale future router (zero training cost, immediate baseline)
5. **Evaluate and iterate:** Compare all three variants, select winner

---

## References

- SpecPrefetch: https://github.com/wei390/SpecPrefetch
- PROBE: https://www.researchgate.net/publication/400369736
- ACM Survey on MoE Inference: https://doi.org/10.1145/3794845

---

## Conclusion

This revised architecture unblocks Phase 5D by moving the predictor to where router input exists, using cheaper features (router logits), and testing multiple predictor variants. The key insight is separation of concerns: the predictor lives alongside the router, and the cache just receives predictions via a clean API.

**Expected outcome:** 10-30% speedup over 5C heuristic predictor, approaching the 9.52x oracle ceiling from 5B.
