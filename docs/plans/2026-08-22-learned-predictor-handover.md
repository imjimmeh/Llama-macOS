# Phase 5D: Learned Routing Predictor - Handover Document

**Date**: 2026-08-22  
**Status**: Partial Implementation (Variant A only, not integrated into inference path)  
**Author**: imjimmeh  
**Reviewer**: Claude Sonnet (architectural guidance)

---

## Executive Summary

Phase 5D implements a learned routing predictor for MoE expert prefetching. The predictor runs upstream of the expert cache, consuming router logits at layer L to predict which experts will be needed at layer L+H (horizon H=8 by default). This enables async prefetch of predicted experts to hide PCIe DMA latency.

**Current state**: Variant A (stale future router) is implemented and passes unit tests, but is not wired into the inference path. Variants B/C require trained model files that don't exist yet. The predictor initializes once per MoE layer per benchmark repetition (bug: should be once per context). Stats retrieval is a stub.

---

## 1. Architectural Context

### 1.1 Problem Statement

In CPU/GPU offloaded MoE inference, expert weights reside in host RAM and transfer over PCIe on every token. The expert cache (Vectors 1-7) reduces transfers for hot experts, but still suffers from:
- Router ID synchronization: scheduler waits for D2H copy of selected IDs before cache lookup
- No lookahead: cache reacts to current layer's routing, not future layers

### 1.2 Solution Architecture

```
router input x_L
|
+-------- native router -------► actual experts L
|
+-------- predictor -----------► predicted experts L+H
|
▼
async prefetch queue
```

The predictor executes when `x_L` exists (during graph construction), not when `MUL_MAT_ID` runs. It produces predicted expert IDs for layer L+H, which feed into an async prefetch queue that stages experts into the cache before they're needed.

### 1.3 Three Predictor Variants

| Variant | Description | Training Required | Input |
|---------|-------------|-------------------|-------|
| **A: Stale Future Router** | Use `W_router[L+H] × x_L` directly | No | Router logits at L |
| **B: Low-Rank MLP** | `x_L → rank-32 → expert logits L+H` | Yes (~147k params) | Router logits or projected hidden state |
| **C: Future Router + Residual** | `W_router[L+H]×x_L + Δ_θ(x_L)` | Yes (residual only) | Router logits + learned correction |

**Reviewer recommendation**: Start with Variant A (zero training cost), then try Variant C (strong structural prior + tiny trainable params).

---

## 2. Current Implementation Status

### 2.1 Files Modified

| File | Change | Status |
|------|--------|--------|
| `ggml/include/ggml-routing-predictor.h` | New header: predictor types, config, lifecycle API | ✅ Complete |
| `ggml/src/ggml-routing-predictor.cpp` | Implementation: init, free, load_model, predict, extract_features | ✅ Complete |
| `ggml/include/ggml-backend.h` | Added `ggml_routing_predictor_stats` struct + `ggml_backend_sched_get_routing_predictor_stats()` declaration | ✅ Complete |
| `ggml/src/ggml-backend.cpp` | Stub implementation of stats retrieval (returns false) | ⚠️ Stub |
| `include/llama.h` | Added `routing_predictor_horizon`, `routing_predictor_stats` to `llama_context_params` | ✅ Complete |
| `src/llama-cparams.h` | Added routing predictor fields to `llama_cparams` | ✅ Complete |
| `common/arg.cpp` | Added CLI args `--routing-predictor-horizon`, `--routing-predictor-stats` | ✅ Complete |
| `src/llama-graph.cpp` | Predictor initialization at line 1501 | ⚠️ Bug: initializes per-layer |
| `tools/llama-bench/llama-bench.cpp` | Added CLI parsing, stats capture, CSV output fields | ✅ Complete |
| `tests/test-routing-predictor.cpp` | Unit tests for all variants | ✅ All pass |
| `docs/plans/2026-08-22-fix-routing-predictor-issues.md` | Fix plan document | ✅ Complete |

### 2.2 What Works

1. **CLI integration**: `--routing-predictor-horizon N` and `--routing-predictor-stats` flags register and appear in `llama-bench --help`
2. **Unit tests**: All 5 test cases pass (init, extract_features, predict, null handling, different dims)
3. **Variant A initialization**: Predictor initializes successfully with `input_dim=256, num_experts=256, horizon=8`
4. **Build**: All targets compile without errors (warnings only)

### 2.3 What Doesn't Work

1. **Stats retrieval is a stub**: `ggml_backend_sched_get_routing_predictor_stats()` returns false without populating stats. All CSV output shows zeros.
2. **Predictor initializes per-layer**: Should initialize once per context, but currently initializes once per MoE layer per benchmark repetition (~20 times for 4 layers × 5 reps).
3. **Not wired into inference**: Predictor is initialized but never called during graph execution. No eval callback registered.
4. **No trained models**: Variants B/C require `.bin` model files that don't exist.
5. **No prefetch integration**: Predictor output doesn't feed into async prefetch queue.

---

## 3. Known Issues and Bugs

### 3.1 Critical: Stats Retrieval Stub

**File**: `ggml/src/ggml-backend.cpp:2410-2421`

```cpp
bool ggml_backend_sched_get_routing_predictor_stats(
    ggml_backend_sched_t sched,
    struct ggml_routing_predictor_stats * out_stats) {
    
    if (!sched || !out_stats) return false;
    
    // Access the graph result to get metrics
    // This requires exposing the graph result through the scheduler
    // For now, return false to indicate not implemented
    // TODO: Implement proper metrics retrieval
    return false;
}
```

**Impact**: All routing predictor stats are zero in benchmarks. Cannot measure prediction accuracy, hit rate, or prefetch effectiveness.

**Fix direction**: Store predictor stats in `ggml_backend_sched` or `llm_graph_result`, populate during eval callback, retrieve via this function.

### 3.2 High: Per-Layer Initialization

**File**: `src/llama-graph.cpp:1501`

```cpp
res->routing_predictor = ggml_routing_predictor_init(&config);
```

**Impact**: Predictor initializes once per MoE layer per benchmark repetition. For 4 MoE layers × 5 reps = 20 initializations. Should be once per context lifetime.

**Fix direction**: Move initialization to `llama_new_context_with_model()` or similar one-time setup. Store in `llama_context` or `llm_graph_result` with proper lifetime management.

### 3.3 High: No Inference Integration

**Current state**: Predictor is initialized but never called. No eval callback registered to run prediction during graph execution.

**Fix direction**: Register eval callback in `llama_context` that:
1. Detects when router logits tensor is available (layer L)
2. Calls `ggml_routing_predictor_predict()` with router logits
3. Submits predicted expert IDs to prefetch queue via `ggml_backend_sched_submit_prediction()`

### 3.4 Medium: No Trained Models

**Current state**: Variants B/C require `.bin` model files. No training pipeline exists.

**Fix direction**:
1. Create data collection script (collect router logits + future expert labels)
2. Train Variant B/C models (PyTorch or direct C++ training)
3. Export to binary format matching `load_model()` expectations

---

## 4. Remaining Work

### 4.1 Immediate Fixes (1-2 days)

1. **Implement stats retrieval**
   - Add stats fields to `ggml_backend_sched` or `llm_graph_result`
   - Populate during eval callback (predictions_generated, predictions_used, etc.)
   - Return via `ggml_backend_sched_get_routing_predictor_stats()`

2. **Fix initialization lifetime**
   - Move `ggml_routing_predictor_init()` to context creation
   - Store predictor in `llama_context` or `llm_graph_result`
   - Free in context destructor

3. **Wire eval callback**
   - Register callback in `llama_context` after `model.build_graph()`
   - Detect router logits tensor (parse `blk.N.ffn_moe_logits` name)
   - Call predictor, submit predictions to scheduler

### 4.2 Variant A Integration (2-3 days)

4. **Async prefetch queue**
   - Add `ggml_backend_sched_submit_prediction()` API
   - Maintain async prefetch queue in scheduler
   - When predicted experts arrive, stage into cache via pinned DMA

5. **Benchmark with real MoE model**
   - Run Qwen3.6-35B-A3B-APEX-Compact.gguf with `-exc 64 -excp 64 --routing-predictor-horizon 8`
   - Measure: prediction recall, prefetch hit rate, decode throughput delta

### 4.3 Training Pipeline (1-2 weeks)

6. **Data collection**
   - Instrument graph to collect router logits at layer L
   - Record actual expert selections at layer L+H (H=4,6,8,10,12)
   - Store as `(router_logits_L, expert_ids_L+H)` pairs
   - Split by whole prompts (not individual tokens) to avoid leakage

7. **Train Variant B (Low-Rank MLP)**
   - Architecture: `router_logits → rank-32 → expert_logits`
   - Loss: BCE + ranking loss
   - Metric: Recall@8, Recall@12, Recall@16 (not exact-match)
   - Target: H=8 (8 layers ahead to hide DMA latency)

8. **Train Variant C (Future Router + Residual)**
   - Architecture: `W_router[L+H]×x_L + low_rank_correction(x_L)`
   - Freeze future router weights, train only residual
   - Smaller trainable params (~16k vs 147k for Variant B)

### 4.4 Advanced Integration (2-4 weeks)

9. **CUDA predictor stream**
   - Run predictor on low-priority auxiliary CUDA stream
   - Avoid synchronizing main compute stream to read predicted IDs
   - Keep predicted IDs on GPU, drive device-side cache lookup

10. **Multi-horizon prediction**
    - Train shared rank-32 trunk + horizon-specific heads (H=4,6,8,10,12)
    - Compare recall × ready_rate × bytes_transferred across horizons
    - Select optimal H based on DMA latency vs prediction accuracy tradeoff

11. **Router logits as features**
    - Test Predictor 1: `x_L` (projected hidden state)
    - Test Predictor 2: `router_logits_L` (128 floats, very cheap)
    - Test Predictor 3: top-K IDs + weights at L
    - Test Predictor 4: concatenation of above

---

## 5. Key Design Decisions (from Reviewer)

### 5.1 Don't Drag Hidden State into Cache

**Problem**: Original implementation tried to run predictor inside expert cache code, but cache only sees expert weight tensors in `MUL_MAT_ID` path, not hidden state.

**Solution**: Move prediction upstream, close to router computation. Predictor consumes router input `x_L` during graph construction, not during `MUL_MAT_ID` execution.

### 5.2 Use Router Logits, Not Hidden State

**Rationale**: Router logits are only 128 floats and encode compressed semantic routing representation. Much cheaper than copying full hidden state (4096 dims) to CPU.

**Implementation**: Variant A uses router logits directly. Variants B/C can use router logits or projected hidden state.

### 5.3 Target H=8, Not H=1

**Rationale**: One 16.88 MiB expert takes ~1467 µs to transfer. At 200 µs/layer compute, need ~8 layers to hide DMA latency. H=1 predictions arrive too late to be useful.

**Metric**: Optimize for `recall@8 × ready_rate × bytes_transferred`, not exact-match accuracy.

### 5.4 Multi-Label, Not Classification

**Rationale**: Qwen activates 8 experts out of 128. Target is `y ∈ {0,1}^128`, not single class.

**Loss**: `L = L_BCE + λ L_rank`

**Metric**: Recall@8, Recall@10, Recall@12, Recall@16. If target experts are `{1, 7, 19, 24, 38, 55, 77, 101}` and top-12 contains all 8, that's perfect for prefetching even with 4 extras.

### 5.5 Train on Whole Prompts, Not Tokens

**Rationale**: Adjacent layers/tokens are heavily correlated. Random split of `(token, layer)` samples causes data leakage.

**Solution**: Split by whole prompts/conversations:
- Train: 70% prompts
- Validation: 15% unseen prompts
- Test: 15% unseen prompts

Evaluate by category (coding, conversation, reasoning, creative, factual).

---

## 6. Testing and Verification

### 6.1 Current Tests

```bash
# Unit tests (all pass)
cmake --build build --config Release --target test-routing-predictor
build/bin/Release/test-routing-predictor.exe

# CLI integration (flags register)
build/bin/Release/llama-bench.exe --help 2>&1 | grep routing-predictor
# Output:
#   --routing-predictor-horizon <N>               lookahead layers for routing predictor (default: 8)
#   --routing-predictor-stats                     print routing predictor performance statistics on exit
```

### 6.2 Required Tests

1. **Stats retrieval test**: Initialize predictor, run inference, verify stats are nonzero
2. **Lifetime test**: Create context, verify predictor initialized once, destroy context, verify no leak
3. **Integration test**: Run with real MoE model, verify predictions generated and used
4. **Benchmark test**: Compare baseline vs. expert cache vs. expert cache + routing predictor

### 6.3 Benchmark Configuration

```bash
# Baseline (no expert cache, no predictor)
llama-bench -m <model> -p 512 -n 128 -fitt 256

# Expert cache only
llama-bench -m <model> -p 512 -n 128 -fitt 256 -exc 64 -excp 64

# Expert cache + routing predictor (Variant A)
llama-bench -m <model> -p 512 -n 128 -fitt 256 -exc 64 -excp 64 --routing-predictor-horizon 8 --routing-predictor-stats
```

**Model**: `Qwen3.6-35B-A3B-APEX-Compact.gguf` (35B total, 3B active, 64 experts/layer, 8 active/token)  
**Hardware**: GTX 1080 (8 GB VRAM) + CPU (14 threads)

---

## 7. File Locations

### 7.1 Core Implementation

- `ggml/include/ggml-routing-predictor.h` - Public API
- `ggml/src/ggml-routing-predictor.cpp` - Implementation
- `ggml/include/ggml-backend.h:364-373` - Stats struct
- `ggml/include/ggml-backend.h:428` - Stats retrieval declaration
- `ggml/src/ggml-backend.cpp:2410-2421` - Stats retrieval stub

### 7.2 Integration

- `include/llama.h:417,420` - Context params fields
- `src/llama-cparams.h:73-74` - Internal params fields
- `common/arg.cpp:2800-2815` - CLI argument registration
- `src/llama-graph.cpp:1490-1505` - Predictor initialization (buggy)
- `tools/llama-bench/llama-bench.cpp` - Benchmark integration

### 7.3 Tests and Docs

- `tests/test-routing-predictor.cpp` - Unit tests
- `docs/plans/2026-08-21-routing-lookahead-pipeline.md` - Original plan
- `docs/plans/2026-08-22-fix-routing-predictor-issues.md` - Fix plan
- `docs/plans/2026-08-22-learned-predictor-handover.md` - This document

### 7.4 Related Files

- `ggml/src/ggml-backend-expert-cache.cpp` - Expert cache (prefetch target)
- `src/llama-graph.cpp` - Graph construction (router logits source)
- `common/expert-cache-profile.cpp` - Profile persistence (related)

---

## 8. Next Steps (Priority Order)

1. **Fix stats retrieval** (1 day) - Implement `ggml_backend_sched_get_routing_predictor_stats()` properly
2. **Fix initialization lifetime** (0.5 day) - Move to context creation, free in destructor
3. **Wire eval callback** (1-2 days) - Register callback, detect router logits, call predictor
4. **Benchmark Variant A** (1 day) - Run with real MoE model, measure baseline impact
5. **Create training pipeline** (1 week) - Data collection, Variant B/C training
6. **Integrate prefetch queue** (2-3 days) - Async staging of predicted experts
7. **Benchmark Variants B/C** (2-3 days) - Compare against Variant A and baseline

---

## 9. Risks and Mitigations

### 9.1 Predictor Adds Latency

**Risk**: Running predictor on CPU adds latency to graph execution.

**Mitigation**: 
- Run on separate thread (async)
- Use router logits (128 floats) not hidden state (4096 floats)
- Variant A has zero compute cost (just matrix multiply with frozen weights)

### 9.2 Predictions Arrive Too Late

**Risk**: Prediction for layer L+H arrives after layer L+H executes.

**Mitigation**:
- Target H=8 (8 layers × 200 µs = 1600 µs budget)
- Predictor compute: ~10 µs (tiny MLP)
- Prefetch DMA: ~1467 µs per expert
- If prediction arrives late, drop it (never wait)

### 9.3 Low Prediction Accuracy

**Risk**: Predictor accuracy too low to justify prefetch overhead.

**Mitigation**:
- Start with Variant A (uses actual future router weights, should be high accuracy)
- Measure Recall@8, not exact-match
- Even 50% recall with 0% wasted bytes is beneficial
- Fall back to reactive cache if predictions are worse than random

### 9.4 Training Data Leakage

**Risk**: Model memorizes routing patterns from adjacent tokens/layers.

**Mitigation**:
- Split by whole prompts, not individual tokens
- Evaluate on unseen prompts
- Test across categories (coding, conversation, reasoning)

---

## 10. Contact and Context

**Original plan**: `docs/plans/2026-08-21-routing-lookahead-pipeline.md`  
**Fix plan**: `docs/plans/2026-08-22-fix-routing-predictor-issues.md`  
**Reviewer guidance**: Archived in conversation history (2026-08-21 session)

**Key insight from reviewer**: 
> "The main issue in your current 5D design is where you're trying to run the predictor. Right now you've implemented the learned model inside the expert-cache code, but the cache only sees the expert weight tensor in the `MUL_MAT_ID` path. It does not see the hidden state that produced the router decision. That's why integration stalled."

**Reviewer recommendation**: 
> "I would not try to drag hidden state all the way down into the cache scheduler. I'd move prediction upstream, close to the router computation itself."

---

## 11. Appendix: Predictor Binary Format

For Variants B/C, model files use this binary format:

```
Magic: 0x4C525044 ("LRPD")
Version: 1 (uint32)
Input dim: 256 (int32)
Num experts: 256 (int32)
Horizon: 8 (int32)
Rank: 32 (int32)
Weights: [rank × input_dim] floats (down_weight)
Bias: [rank] floats (down_bias)
Output weights: [num_experts × rank] floats (output_weight)
Output bias: [num_experts] floats (output_bias)
```

For Variant C, additional residual weights follow the same pattern.

---

**End of handover document**
