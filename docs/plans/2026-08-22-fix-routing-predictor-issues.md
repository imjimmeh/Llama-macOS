# Fix Routing Predictor Integration Issues (2026-08-22)

## Context

Phase 5D routing predictor is now integrated via eval callback (2026-08-22), but has 5 known limitations that prevent production use. This plan addresses each issue in priority order.

---

## Issue 1: No Cleanup (Memory Leak) 🔴 **CRITICAL**

**Problem:** `ggml_routing_predictor_free()` is never called, leaking memory on context destruction.

**Impact:** Memory leak every time a model is loaded/unloaded.

**Fix Location:** `src/llama-graph.h` - `llm_graph_result` destructor

### Tasks

1. **Add destructor to `llm_graph_result`**
   - File: `src/llama-graph.h`
   - Add: `virtual ~llm_graph_result();` (currently `= default`)
   - File: `src/llama-graph.cpp`
   - Implement destructor:
     ```cpp
     llm_graph_result::~llm_graph_result() {
         if (routing_predictor) {
             ggml_routing_predictor_free(routing_predictor);
             routing_predictor = nullptr;
         }
     }
     ```

2. **Verify no double-free**
   - Check that `routing_predictor` is only freed in destructor
   - Ensure `nullptr` assignment prevents double-free

**Effort:** 30 minutes  
**Risk:** Low  
**Verification:** Run `test-routing-predictor` under Valgrind/ASan

---

## Issue 2: Hardcoded H=8  **HIGH**

**Problem:** Horizon is fixed at 8 layers ahead. Different models/hardware may need different horizons.

**Impact:** Cannot tune for different transfer speeds or layer compute times.

**Fix Location:** CLI parameter + config propagation

### Tasks

1. **Add CLI parameter**
   - File: `common/common.h` - Add to `gpt_params`:
     ```cpp
     int32_t routing_predictor_horizon = 8;
     ```
   - File: `common/common.cpp` - Add argument parser:
     ```cpp
     gpt_params_add_opt("--routing-predictor-horizon", params.routing_predictor_horizon);
     ```
   - File: `common/arg.cpp` - Add help text

2. **Propagate through config chain**
   - `llama_context_params` → `llama_cparams` → `llm_graph_params`
   - File: `include/llama.h` - Add to `llama_context_params`
   - File: `src/llama.cpp` - Copy in `llama_init_from_model()`
   - File: `src/llama-graph.h` - Add to `llm_graph_params`

3. **Use in predictor initialization**
   - File: `src/llama-graph.cpp` line ~1478
   - Change: `config.horizon = 8;` → `config.horizon = params.cparams.routing_predictor_horizon;`

4. **Update test**
   - File: `tests/test-routing-predictor.cpp`
   - Test multiple horizon values (4, 8, 12)

**Effort:** 2 hours  
**Risk:** Medium (config propagation touches multiple files)  
**Verification:** Run with `--routing-predictor-horizon 12`, verify logs show H=12

---

## Issue 3: No Metrics 🟡 **HIGH**

**Problem:** Cannot measure prediction effectiveness without counters.

**Impact:** Cannot determine if predictor is actually helping or hurting performance.

**Fix Location:** `llm_graph_result` + callback + stats API

### Tasks

1. **Add metrics struct to `llm_graph_result`**
   - File: `src/llama-graph.h`
   - Add:
     ```cpp
     struct routing_predictor_metrics {
         int64_t predictions_generated = 0;
         int64_t predictions_used = 0;
         int64_t predictions_too_late = 0;
         int64_t predictions_wrong = 0;
         int64_t experts_fully_hidden = 0;
         int64_t experts_partially_hidden = 0;
         int64_t experts_missed = 0;
         int64_t bytes_wasted = 0;
     };
     routing_predictor_metrics predictor_metrics;
     ```

2. **Increment counters in callback**
   - File: `src/llama-graph.cpp` - `routing_predictor_callback()`
   - Increment `predictions_generated` on each call
   - (Other counters need expert cache cooperation - see below)

3. **Add stats retrieval API**
   - File: `ggml/include/ggml-backend.h`
     ```cpp
     GGML_API bool ggml_backend_sched_get_routing_predictor_stats(
         ggml_backend_sched_t sched,
         struct ggml_routing_predictor_stats * out_stats);
     ```
   - File: `ggml/src/ggml-backend.cpp` - Implement

4. **Expert cache cooperation** (deferred to Issue 4)
   - When prefetch completes, check if prediction was used
   - Update `predictions_used` vs `predictions_too_late`
   - Track `experts_fully_hidden` vs `partially_hidden`

5. **CLI flag to print stats**
   - File: `common/common.cpp`
   - Add `--routing-predictor-stats` flag
   - Print on exit or periodic interval

**Effort:** 4 hours (partial - full metrics need expert cache changes)  
**Risk:** Medium  
**Verification:** Run with stats flag, verify counters increment

---

## Issue 4: Synchronous D2H Stall 🟠 **MEDIUM**

**Problem:** `ggml_backend_tensor_get()` in callback blocks compute stream.

**Impact:** Defeats the purpose of async prefetch - adds latency to every MoE layer.

**Fix Location:** Callback + async D2H infrastructure

### Tasks

1. **Allocate pinned host buffer for logits**
   - File: `src/llama-graph.h` - Add to `llm_graph_result`:
     ```cpp
     float * pinned_logits_buffer = nullptr;
     size_t pinned_logits_size = 0;
     ```
   - File: `src/llama-graph.cpp` - Allocate in constructor:
     ```cpp
     cudaHostAlloc(&pinned_logits_buffer, n_experts * sizeof(float), cudaHostAllocDefault);
     ```

2. **Use async copy in callback**
   - File: `src/llama-graph.cpp` - `routing_predictor_callback()`
   - Replace `ggml_backend_tensor_get()` with:
     ```cpp
     cudaMemcpyAsync(pinned_logits_buffer, tensor->data, 
                     n_experts * sizeof(float), cudaMemcpyDeviceToHost,
                     compute_stream);
     cudaStreamSynchronize(compute_stream);  // Still sync, but faster with pinned
     ```

3. **True async (requires CUDA stream management)**
   - Create dedicated `predictor_stream`
   - Use CUDA events to synchronize:
     ```cpp
     cudaEvent_t logits_ready;
     cudaMemcpyAsync(pinned_buffer, tensor->data, ..., compute_stream);
     cudaEventRecord(logits_ready, compute_stream);
     cudaStreamWaitEvent(predictor_stream, logits_ready);
     // Run predictor on predictor_stream
     ```
   - File: `ggml/src/ggml-backend.cpp` - Expose stream access API

4. **Eliminate CPU prediction entirely (long-term)**
   - Move predictor to CUDA kernel
   - Run on auxiliary stream
   - No D2H needed at all

**Effort:** 8 hours (pinned buffer), 16 hours (true async), 24 hours (CUDA predictor)  
**Risk:** High (CUDA stream management is complex)  
**Verification:** Nsight Systems trace showing no compute stream stall

---

## Issue 5: Variant A Only 🟢 **LOW**

**Problem:** Only stale future router is implemented. Learned variants need model loading.

**Impact:** Cannot test if learned predictor improves accuracy over stale router.

**Fix Location:** Predictor initialization + model loading

### Tasks

1. **Add model path to config**
   - File: `ggml/include/ggml-routing-predictor.h`
   - Add `const char * model_path` to `ggml_routing_predictor_config` (already exists)

2. **Implement model loading for Variant B**
   - File: `ggml/src/ggml-routing-predictor.cpp`
   - Add `load_learned_model()` function
   - Read binary format from `tools/train_routing_predictor.py`
   - Populate weight matrices

3. **Implement Variant C (residual)**
   - Load both future router weights and residual correction
   - Forward pass: `W_router[L+H] * x_L + residual_mlp(x_L)`

4. **CLI parameter for model path**
   - File: `common/common.cpp`
   - Add `--routing-predictor-model <path>`

5. **Training pipeline**
   - File: `tools/train_routing_predictor.py` - Already exists
   - Need to collect real routing traces with router logits (not hidden state)
   - Update trace format to include router logits

**Effort:** 12 hours  
**Risk:** Medium  
**Verification:** Train on real traces, compare Variant A vs B vs C recall

---

## Priority Order and Dependencies

```
Issue 1 (Cleanup) ──────────────────────────────────────► 30 min
       │
Issue 2 (Horizon) ──────────────────────────────────────► 2 hours
       │
Issue 3 (Metrics) ──────────────────────────────────────► 4 hours (partial)
       │
Issue 4 (Async D2H) ────────────────────────────────────► 8-24 hours
       │
Issue 5 (Variants B/C) ─────────────────────────────────► 12 hours
```

**Recommended execution order:**
1. Issue 1 (fix memory leak - critical)
2. Issue 2 (make horizon configurable - enables tuning)
3. Issue 3 (add basic metrics - enables measurement)
4. Issue 4 (async D2H - removes sync stall)
5. Issue 5 (learned variants - improves accuracy)

**Total effort:** ~26-42 hours depending on async D2H approach

---

## Verification Plan

After each issue is fixed:

1. **Build:** `cmake --build . --target llama --config Release`
2. **Unit tests:** `test-routing-predictor.exe`
3. **Integration test:** Run Qwen 35B MoE with expert cache enabled
4. **Metrics:** Verify counters/stats are correct
5. **Performance:** Compare tok/s before/after each fix

**Final acceptance criteria:**
- No memory leaks (Valgrind/ASan clean)
- Configurable horizon via CLI
- Metrics show >80% prediction recall at H=8
- No compute stream stalls in Nsight trace
- Variant B/C recall > Variant A recall on real traces

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Config propagation breaks existing code | Add defaults, test all CLI tools |
| Async D2H introduces race conditions | Use CUDA events, extensive testing |
| Learned variants overfit training data | Split by prompts, not tokens |
| Metrics overhead slows inference | Use atomic counters, sample at 1% |

---

## Success Metrics

- **Memory:** Zero leaks on context destroy
- **Performance:** No regression vs current Variant A
- **Accuracy:** Variant C recall > Variant B > Variant A
- **Usability:** CLI parameters work as documented
- **Observability:** Metrics show clear prediction effectiveness
