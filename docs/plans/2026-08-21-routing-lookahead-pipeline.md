# Routing Lookahead Pipeline Implementation Plan

**Date:** 2026-08-21  
**Status:** Planning  
**Branch:** `expert-cache`  
**Target:** GTX 1080 (8 GiB VRAM) with Qwen 35B MoE

---

## Executive Summary

Current measurements show **169.7 GiB RAM→GPU transfer** over 5×128 tokens, dropping generation from 26.5 to 9.19 tok/s. The problem is not cache hit rate—it is **exposed DMA latency**.

This plan implements a **Routing Lookahead Pipeline** that predicts expert demand H layers ahead and issues asynchronous DMA transfers on a dedicated CUDA stream, allowing transfers to overlap with compute.

**Key insight:** We need to determine the **minimum prediction horizon** that provides meaningful benefit before building complex predictors. An oracle simulator will answer: "If we knew the exact future experts, how much faster could we go?"

---

## Phase 5A: Offline Route Predictability Study

**Objective:** Measure how predictable expert routing is, without any runtime modifications.

**Duration:** 2-3 days  
**Risk:** Low (instrumentation only, no behavioral changes)

### Deliverables

1. **Route trace collector**
   - Instrument `ggml-backend.cpp` to dump per-token, per-layer routing decisions
   - Record: token_id, layer, router_input_hidden_state, router_logits, top-k experts, expert_weights, timestamp
   - Output format: binary trace file + human-readable summary

2. **Predictability analysis scripts**
   - Compute Recall@K for multiple prediction strategies:
     - Previous token, same layer: `route(t-1, L) → route(t, L)`
     - Cross-layer: `route(t, L) → route(t, L+H)` for H ∈ {1, 2, 4, 8, 12, 16}
   - Generate plots:
     ```
     Recall
     100% ┤\
      90% ┤ \
      80% ┤  \____
      70% ┤       \___
         └──────────────
           1  2  4  8 16
           lookahead layers
     ```
   - Compute **ready recall**: fraction of predicted experts that arrive before execution reaches the layer

3. **Cross-layer correlation matrix**
   - For each layer L and horizon H, compute:
     - Recall@8, Recall@10, Recall@12, Recall@16
     - Precision@8, Precision@12, Precision@16
     - Bytes prefetched vs bytes useful vs bytes wasted

### Code Changes

**File: `ggml/src/ggml-backend.cpp`**
- Add instrumentation hooks in `mul_mat_id` execution path (around line 1735-1850)
- Record router decisions to trace buffer
- Flush trace to disk periodically

**File: `tools/results/route-trace-analyzer.py`** (new)
- Parse binary trace files
- Compute recall/precision metrics
- Generate visualization plots

**File: `tests/test-route-trace.cpp`** (new)
- Unit tests for trace collection
- Verify trace format correctness

### Success Criteria

- ✅ Trace collector adds <1% overhead to generation
- ✅ Can generate 1000+ token traces for analysis
- ✅ Recall curves show clear trend: how much prediction quality degrades with horizon
- ✅ Identify **minimum useful horizon** (e.g., "H=8 gives 80% recall, H=16 gives 60%")

### Go/No-Go Decision

**Proceed to 5B if:**
- Recall@H≥4 > 70% (meaningful prediction signal exists)
- Ready recall analysis shows transfers can complete in time

**Stop if:**
- Recall drops below 50% for all H > 2 (routing is essentially random)
- Ready recall < 30% (PCIe bandwidth is the bottleneck, not latency)

---

## Phase 5B: Trace-Driven Oracle Simulator

**Objective:** Simulate the entire prefetch pipeline with perfect future knowledge to determine theoretical maximum benefit.

**Duration:** 3-4 days  
**Risk:** Low (simulation only, no runtime changes)

### Deliverables

1. **Oracle simulator**
   - Input: route trace from 5A, expert sizes, measured PCIe throughput, per-layer execution times
   - Simulate:
     - Perfect prediction (100% recall)
     - Constrained prediction (oracle L+1, L+2, L+4, L+8, L+16, whole-token)
     - Dedicated CUDA transfer stream
     - Cache admission/eviction policy
     - Transfer scheduling with bandwidth constraints
   - Output:
     ```
     Baseline (no prefetch):     26.5 tok/s
     Oracle L+1:                 27.1 tok/s
     Oracle L+4:                 30.2 tok/s
     Oracle L+8:                 34.7 tok/s
     Oracle L+16:                35.5 tok/s
     Whole-token oracle:         36.0 tok/s
     ```

2. **Bottleneck analysis**
   - Determine if we're latency-bound or bandwidth-bound
   - If oracle whole-token gives <10% improvement → **stop, hardware cannot benefit**
   - If oracle L+8 gives >20% improvement → **prediction accuracy is worth pursuing**

3. **Sensitivity analysis**
   - Vary PCIe bandwidth (simulate PCIe 3.0 x16 vs 2.0 x8)
   - Vary cache capacity (2 GiB, 4 GiB, 6 GiB)
   - Identify optimal cache size for given bandwidth

### Code Changes

**File: `tools/results/oracle-simulator.py`** (new)
- Parse route traces
- Simulate transfer pipeline with perfect knowledge
- Compute theoretical speedup

**File: `tools/results/measure-pcie-bandwidth.cpp`** (new)
- Benchmark actual PCIe throughput for expert-sized transfers
- Record: transfer_size vs throughput

### Success Criteria

- ✅ Simulator reproduces baseline (no prefetch) within 5% of measured performance
- ✅ Oracle simulations show clear trend: speedup vs prediction horizon
- ✅ Identify **target horizon** for predictor (e.g., "aim for H=8")

### Go/No-Go Decision

**Proceed to 5C if:**
- Oracle L+8 gives >15% speedup
- Analysis shows we're latency-bound, not bandwidth-bound

**Stop if:**
- Oracle whole-token gives <10% speedup (PCIe is the bottleneck)
- Simulator shows diminishing returns beyond H=2

---

## Phase 5C: Async DMA Pipeline + Heuristic Predictor

**Objective:** Implement dedicated CUDA transfer stream and heuristic predictor (previous token + cross-layer transition tables).

**Duration:** 5-7 days  
**Risk:** Medium (first runtime changes, but correctness-preserving)

### Deliverables

1. **Dedicated CUDA transfer stream**
   - Create separate `cudaStream_t` for expert prefetch transfers
   - Modify `ggml_backend_expert_cache_prefetch()` to use async DMA
   - Add transfer state tracking:
     ```cpp
     enum prefetch_state {
         PREFETCH_EMPTY,
         PREFETCH_IN_FLIGHT,
         PREFETCH_RESIDENT
     };
     struct prefetch_slot {
         prefetch_state state;
         cudaEvent_t ready_event;
         int32_t layer;
         int32_t expert_id;
     };
     ```

2. **Heuristic predictor**
   - Maintain transition table: `P(expert_e | previous_token_route)`
   - Maintain cross-layer table: `P(expert_e at L+H | expert_f at L)`
   - At layer L, issue prefetches for L+H using transition probabilities
   - Filter by expected value:
     ```cpp
     V(e) = P(e) * T_stall - (1 - P(e)) * T_waste - T_eviction
     if (V(e) > 0 && ETA_transfer < ETA_layer) {
         enqueue_prefetch(e);
     }
     ```

3. **Prefetch-aware execution**
   - Modify `mul_mat_id` path to check prefetch state:
     ```cpp
     if (slot.state == PREFETCH_RESIDENT) {
         execute;  // fully hidden hit
     } else if (slot.state == PREFETCH_IN_FLIGHT) {
         cudaEventSynchronize(slot.ready_event);  // partially hidden
         execute;
     } else {
         synchronous_miss();  // fallback
     }
     ```

4. **Metrics**
   - Track: fully_hidden_hits, partially_hidden_hits, misses, wasted_prefetches
   - Compare against baseline (no prefetch) and oracle (from 5B)

### Code Changes

**File: `ggml/src/ggml-backend-expert-cache.cpp`**
- Add `cudaStream_t prefetch_stream` to `ggml_backend_expert_cache` struct
- Add `std::vector<prefetch_slot> prefetch_slots` for tracking in-flight transfers
- Modify `ggml_backend_expert_cache_prefetch()` to:
  - Allocate prefetch slot
  - Issue `cudaMemcpyAsync` on `prefetch_stream`
  - Record `cudaEvent_t` for completion
- Add `ggml_backend_expert_cache_wait_prefetch()` to synchronize before use

**File: `ggml/src/ggml-backend-expert-cache.h`**
- Add API:
  ```cpp
  GGML_API void ggml_backend_expert_cache_prefetch_async(
      ggml_backend_expert_cache_t cache,
      const struct ggml_tensor * tensor,
      const int32_t * expert_ids,
      int32_t n_experts,
      int32_t target_layer);
  
  GGML_API bool ggml_backend_expert_cache_is_prefetch_ready(
      ggml_backend_expert_cache_t cache,
      const struct ggml_tensor * tensor,
      int32_t expert_id);
  ```

**File: `ggml/src/ggml-backend.cpp`**
- Modify `mul_mat_id` path (around line 1812-1882) to:
  - Check prefetch state before synchronous copy
  - Wait on prefetch event if in-flight
  - Record metrics

**File: `ggml/src/ggml-backend-expert-cache.cpp`** (new section)
- Implement heuristic predictor:
  ```cpp
  struct routing_predictor {
      std::unordered_map<int32_t, std::vector<float>> token_to_expert_prob;
      std::unordered_map<int32_t, std::vector<float>> layer_to_layer_prob;
      
      std::vector<int32_t> predict(
          int32_t current_layer,
          int32_t horizon,
          const std::vector<int32_t>& current_route);
  };
  ```

### Success Criteria

- ✅ Async DMA pipeline adds <2% overhead when prefetches are not used
- ✅ Heuristic predictor achieves >60% of oracle recall (from 5B)
- ✅ Measured speedup >5% over baseline
- ✅ Metrics show: fully_hidden_hits > 50% of total requests

### Go/No-Go Decision

**Proceed to 5D if:**
- Heuristic predictor gives >5% speedup
- Async pipeline works correctly (no deadlocks, no correctness issues)

**Iterate on 5C if:**
- Speedup <5% but oracle showed potential (predictor needs tuning)
- Async pipeline has correctness issues (fix before proceeding)

---

## Phase 5D: Learned Low-Rank Routing Predictor

**Objective:** Train tiny external model to predict future expert routes.

**Status:** Implementation complete but **not integrated** into execution path.

**Limitation:** The learned predictor requires hidden state data from the forward pass, but the `mul_mat_id` execution path only has access to the expert weights tensor (`input`), not the hidden state that feeds the router. Integration would require passing hidden state through the scheduler/graph execution infrastructure, which is a deeper architectural change.

**Current deliverables:**
- API declarations in `ggml-backend-expert-cache.h`
- C++ implementation in `ggml-backend-expert-cache.cpp` (model loading, inference)
- Python training script in `tools/train_routing_predictor.py`
- **Not integrated** into `ggml-backend.cpp` execution path

**Path forward:** Test heuristic predictor (Phase 5C) first. If it shows meaningful speedup, the learned predictor can be integrated later with additional infrastructure changes.

**Duration:** 7-10 days (original estimate)  
**Risk:** Medium-High

### Deliverables

1. **Training data collector**
   - Extend 5A trace to include hidden states:
     ```cpp
     struct routing_sample {
         int32_t layer;
         std::vector<float> hidden_state;  // router input
         std::vector<int32_t> selected_experts;  // router output
     };
     ```
   - Collect 100k+ samples across diverse prompts

2. **Predictor architecture**
   - Shared low-rank trunk + horizon-specific heads:
     ```
     hidden_state (D=2048 for Qwen 35B)
        ↓
     low-rank projection (D → r=32)
        ↓
     heads:
        H=2:  r → E (128 experts)
        H=4:  r → E
        H=8:  r → E
        H=12: r → E
     ```
   - Total parameters: ~200k (tiny, trains in minutes)

3. **Training script**
   - Loss: cross-entropy on expert logits
   - Optimizer: AdamW, lr=1e-3
   - Early stopping on validation recall@8

4. **Inference integration**
   - Export predictor to ONNX or custom C++ kernel
   - Run predictor at layer L to predict L+H
   - Integrate with async DMA pipeline from 5C

### Code Changes

**File: `tools/train-routing-predictor.py`** (new)
- Load trace data
- Define predictor architecture (PyTorch)
- Train and export model

**File: `ggml/src/ggml-routing-predictor.cpp`** (new)
- Load predictor model
- Run inference: `predict(hidden_state, layer) → expert_probs`
- Integrate with expert cache

**File: `ggml/src/ggml-backend.cpp`**
- At each layer L:
  - Extract router input hidden state
  - Run predictor for L+H
  - Issue prefetches for top-K predicted experts

### Success Criteria

- ✅ Predictor achieves >80% recall@8 for H=8
- ✅ Predictor inference adds <0.5ms per layer
- ✅ Measured speedup >10% over baseline
- ✅ Speedup approaches oracle (within 50% of theoretical max)

### Go/No-Go Decision

**Proceed to 5E if:**
- Learned predictor gives >10% speedup
- Predictor is stable across diverse prompts

**Fallback to 5C if:**
- Predictor overfits or is unstable
- Speedup <5% (heuristic may be sufficient)

---

## Phase 5E: Adaptive Horizon Scheduler

**Objective:** Dynamically select prediction horizon based on runtime conditions.

**Duration:** 3-4 days  
**Risk:** Low (builds on 5D)

### Deliverables

1. **Runtime monitor**
   - Track:
     - PCIe throughput (GB/s)
     - Per-layer execution time (µs)
     - Prefetch queue depth
     - Cache state (resident, in-flight, empty)

2. **Horizon selector**
   - At each layer, compute:
     ```cpp
     transfer_time = expert_bytes / observed_pcie_bandwidth;
     available_time = predicted_layer_time[target] - now;
     
     if (available_time > transfer_time * safety_factor) {
         use_horizon = max_H;  // aggressive
     } else {
         use_horizon = min_H;  // conservative
     }
     ```

3. **Layer-specific tuning**
   - Early layers (0-10): H=12 (lots of time)
   - Middle layers (11-30): H=8
   - Late layers (31-39): H=4 (less time)

### Code Changes

**File: `ggml/src/ggml-backend-expert-cache.cpp`**
- Add runtime monitor:
  ```cpp
  struct runtime_monitor {
      double measured_pcie_bandwidth;
      std::vector<double> layer_execution_times;
      int32_t prefetch_queue_depth;
      
      int32_t select_horizon(int32_t current_layer, int32_t target_layer);
  };
  ```

**File: `ggml/src/ggml-backend.cpp`**
- Integrate horizon selector into prefetch decision

### Success Criteria

- ✅ Adaptive scheduler gives >5% speedup over fixed horizon
- ✅ No performance regressions across diverse workloads

---

## Phase 5F: MTP-Assisted Whole-Token Lookahead

**Objective:** Use MTP speculative tokens as routing hints for whole-token prefetch.

**Duration:** 5-7 days  
**Risk:** High (complex interaction with MTP, but potentially highest reward)

### Deliverables

1. **MTP routing predictor**
   - During MTP draft phase, predict routes for speculative tokens:
     ```
     t+1 = "return"  →  predict experts(t+1, L0..L39)
     t+2 = " result" →  predict experts(t+2, L0..L39)
     ```
   - Begin prefetching high-confidence experts before target model reaches them

2. **Verification-aware prefetch**
   - Only prefetch experts for tokens that survive MTP verification
   - Discard prefetches for rejected speculative tokens

3. **Integration with existing MTP**
   - Modify MTP draft loop to issue prefetches
   - Coordinate with target model execution

### Code Changes

**File: `src/llama-model.cpp`** (MTP draft loop)
- After drafting speculative token t+k:
  - Run routing predictor for t+k
  - Issue prefetches for predicted experts

**File: `ggml/src/ggml-backend.cpp`**
- Add MTP-aware prefetch API:
  ```cpp
  GGML_API void ggml_backend_expert_cache_prefetch_token(
      ggml_backend_expert_cache_t cache,
      int32_t token_id,
      int32_t max_layers);
  ```

### Success Criteria

- ✅ MTP-assisted prefetch gives >15% speedup over layer-only lookahead
- ✅ No correctness issues (MTP verification still works)
- ✅ Prefetch discard rate <20% (most predictions survive verification)

---

## Risk Mitigation

### Risk 1: PCIe Bandwidth is the Bottleneck

**Mitigation:** Phase 5B oracle simulator will reveal if we're bandwidth-bound. If oracle whole-token gives <10% speedup, **stop the project**—no amount of prediction will help.

### Risk 2: Routing is Unpredictable

**Mitigation:** Phase 5A predictability study will reveal if routing is essentially random. If recall drops below 50% for all H > 2, **stop**—prediction won't help.

### Risk 3: Async DMA Pipeline Complexity

**Mitigation:** Phase 5C is the first runtime change. If async pipeline has correctness issues (deadlocks, race conditions), **iterate** before proceeding. Add extensive logging and validation.

### Risk 4: Predictor Overfits

**Mitigation:** Phase 5D uses cross-validation and early stopping. If predictor is unstable, **fallback** to heuristic (5C).

### Risk 5: MTP Integration Breaks Verification

**Mitigation:** Phase 5F is last. If MTP-assisted prefetch breaks verification, **disable** it and keep layer-only lookahead.

---

## Success Metrics

### Primary Metrics

1. **Token generation speed (tok/s)**
   - Baseline: 26.5 tok/s (no prefetch)
   - Target: 35+ tok/s (30%+ speedup)

2. **Prefetch effectiveness**
   - Fully hidden hits: >50% of requests
   - Partially hidden hits: >30% of requests
   - Misses: <20% of requests
   - Wasted prefetches: <10% of prefetches

3. **Prediction accuracy**
   - Recall@8 for H=8: >80%
   - Ready recall: >70% (predicted experts arrive in time)

### Secondary Metrics

1. **PCIe utilization**
   - Transfer bandwidth: >80% of theoretical max
   - Overhead: <5% of total transfer time

2. **Cache efficiency**
   - Hit rate: >80% (including prefetch hits)
   - Eviction rate: <10% of resident experts

3. **Overhead**
   - Predictor inference: <0.5ms per layer
   - Async pipeline management: <1% of total time

---

## Timeline

| Phase | Duration | Dependencies | Risk |
|-------|----------|--------------|------|
| 5A: Predictability Study | 2-3 days | None | Low |
| 5B: Oracle Simulator | 3-4 days | 5A | Low |
| 5C: Async DMA + Heuristic | 5-7 days | 5B | Medium |
| 5D: Learned Predictor | 7-10 days | 5C | Medium-High |
| 5E: Adaptive Scheduler | 3-4 days | 5D | Low |
| 5F: MTP Integration | 5-7 days | 5E | High |
| **Total** | **25-35 days** | | |

---

## Conclusion

This plan prioritizes **measurement before implementation**. Phases 5A and 5B will reveal whether the problem is solvable before we invest in complex predictors.

**Key decision points:**
- After 5A: Is routing predictable enough?
- After 5B: Is there theoretical headroom?
- After 5C: Does async pipeline work?
- After 5D: Does learned predictor help?

If any phase shows the approach won't work, **stop and pivot**. The oracle simulator (5B) is the single most important experiment—it tells us the theoretical maximum before we build anything.

**Expected outcome:** If all phases succeed, we achieve 30%+ speedup (26.5 → 35+ tok/s) on GTX 1080 with Qwen 35B MoE.
