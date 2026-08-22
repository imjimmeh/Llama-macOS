# Learned Routing Predictor: Finish Variant A End-to-End + Training Pipeline

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the learned routing predictor actually useful end-to-end: fix the callback bugs, wire predictions into real prefetch, produce accuracy/efficiency stats in llama-bench CSV, and complete the trace -> train -> load pipeline for variant B.

**Architecture:** The expert cache (ggml-backend-expert-cache.cpp) becomes the single source of truth for prediction state: submissions land in a per-layer pending queue, settle when the layer executes at MUL_MAT_ID interception, and immediately trigger async prefetch of predicted experts' weights. All counters live in `ggml_routing_predictor_stats` inside each expert cache; the sched getter aggregates them. The training side extends route traces with router logits (v2), and the existing Python scripts gain v2 parsing plus LRPD model output that the C++ loader already reads.

**Tech Stack:** C++17 / ggml backend sched + expert cache, llama.cpp graph context eval callback, Python 3 (numpy) for tooling, llama-bench for measurement, scripts/expert-cache-determinism-matrix.py for determinism.

**Target hardware/model:** Qwen3.6-35B-A3B-APEX-Compact.gguf (64 experts, 8 active), GTX 1080 8GB + CPU 14 threads.

---

## Background: verified current state (read this first)

The handover doc (`docs/plans/2026-08-22-learned-predictor-handover.md`) is stale on several points. Verified reality:

- Predictor init happens **once per graph build** in the `llm_graph_context` constructor (`src/llama-graph.cpp:1491-1524`), gated on `cparams.expert_cache_size > 0 && hparams.n_expert > 0`. Config: type=STALE_FUTURE, input_dim=num_experts=n_expert, horizon from cparams, rank=32, model_path=nullptr. Pinned logits buffer allocated alongside. Freed in reset path `src/llama-graph.cpp:1294-1306`.
- Eval callback IS registered (`src/llama-context.cpp:1401-1410`) but it **clobbers any user cb_eval** (BUG A1).
- Callback impl `src/llama-graph.cpp:3978-4059`: matches "ffn_moe_logits" tensors, sync D2H via `ggml_backend_tensor_get`, predicts, submits via `ggml_backend_sched_submit_prediction`. Bugs:
  - A2: dangling pointer - fallback `std::vector<float> logits` declared inside an if-block (lines 4003-4007), used after block ends.
  - A3: target layer hardcoded `current_layer + 8`, ignores config horizon.
  - A4: hardcoded depth 16 arrays; counts prefill tokens too.
- Submit chain dead-ends: `ggml_backend_expert_cache_submit_prediction` (`ggml/src/ggml-backend-expert-cache.cpp:2251-2288`) only bumps `n_predictions_submitted`.
- Stats getter is a stub returning false (`ggml/src/ggml-backend.cpp:2410-2421`) though declared (`ggml/include/ggml-backend.h:428`) and fully consumed by llama-bench CSV columns (`tools/llama-bench/llama-bench.cpp:1599,1796-1804`).
- Dead metrics struct `routing_predictor_metrics` in `src/llama-graph.h:930-949` - only `predictions_generated` ever incremented. Delete it.
- Reusable Phase 5C infra in the cache: `prefetch_async` (1395, dedupes, falls back to sync without CUDA), `register_bundle`/`prefetch_layer` (1214/1713), `is_prefetch_ready`/`wait_prefetch` (1517/1554), route-trace enable/record/flush/disable (1897/1930/1966/1981), hidden-state trace (2002/2052).
- Reactive Markov predictor already hooked at MUL_MAT_ID interception (`ggml/src/ggml-backend.cpp:1811-1830`); actual requested experts available there (bitset built 1757-1764) - this is where settle accounting hooks in.
- Route trace entries record ids only, no logits (`ggml-backend-expert-cache.h:278-284`) - variant B training needs a v2 format with logits.
- Tests: `tests/test-routing-predictor.cpp` (unit, 5 tests), `tests/test-expert-cache.cpp::test_route_trace` (line 377) shows how to build a cache without CUDA - reuse that pattern.

Design decisions baked into tasks below:
- D1: single source of truth = per-cache `ggml_routing_predictor_stats`; sched getter aggregates; delete dead res->predictor_metrics.
- D2: submit-time immediate prefetch of predicted experts' gate/up/down weights.
- D3: one pending entry per target layer; latest submission replaces.
- D4: too_late detection via executed-layer cursor advanced at interception.
- D5: settle accounting classifies hits as fully/partially hidden or missed/wrong.
- D6: horizon + depth stored as llm_graph_result fields; depth = min(num_experts, n_expert_used*2) capped 32; count only decode-token submissions.
- D7: eval callback chaining instead of clobbering.
- D8: trace v2 adds logits; env-var trigger; LRPD binary output from trainer.

Build/test commands assume a configured build dir named `build`; adjust to your setup. Configure once:

```sh
cmake -B build -DGGML_CUDA=ON
```

---

### Task 1: Fix callback bugs A2/A3/A4 (dangling ptr, horizon, depth, prefill counting)

**Files:**
- Modify: `src/llama-graph.h:930-949` (add fields to llm_graph_result)
- Modify: `src/llama-graph.cpp:1491-1524` (set new fields at init)
- Modify: `src/llama-graph.cpp:3978-4059` (callback body)

**Step 1: Add config fields to llm_graph_result**

In `src/llama-graph.h`, inside `struct llm_graph_result` near the routing_predictor members:

```cpp
int predictor_horizon = 0;
int predictor_depth   = 0;
```

**Step 2: Populate them at init**

In the constructor init block (`src/llama-graph.cpp:1491-1524`), after config setup:

```cpp
res->predictor_horizon = config.horizon;
res->predictor_depth   = std::min<int>(config.num_experts, hparams.n_expert_used * 2);
res->predictor_depth   = std::min(res->predictor_depth, 32);
```

**Step 3: Rewrite the buggy parts of the callback**

In `src/llama-graph.cpp:3978-4059`:

- Hoist the fallback buffer out of the if-block so `logits_ptr` always points at memory alive through `ggml_backend_tensor_get`:

```cpp
std::vector<float> logits_fallback;
const float * logits_ptr = nullptr;
if (pinned && n_logits <= res->pinned_logits_capacity) {
    logits_ptr = (const float *) res->pinned_logits_buffer;
} else {
    logits_fallback.resize(n_logits);
    logits_ptr = logits_fallback.data();
}
```

- Replace hardcoded `+ 8` and depth 16:

```cpp
const int target_layer = current_layer + res->predictor_horizon;
float predicted_confidences[32];
int32_t predicted_ids[32];
int n_pred = ggml_routing_predictor_predict(res->routing_predictor,
        logits_ptr, predicted_ids, predicted_confidences, res->predictor_depth);
```

- Move the `predictions_generated` increment after the single-token gate so prefill does not count:

```cpp
if (n_tokens != 1) {
    return true;
}
// ... extract, predict ...
res->predictor_metrics.predictions_generated += n_pred > 0;
```

(keep using res->predictor_metrics here until Task 5 moves counters into the cache).

**Step 4: Build and run unit tests**

Run: `cmake --build build --target test-routing-predictor test-expert-cache && ./build/bin/test-routing-predictor && ./build/bin/test-expert-cache`
Expected: all pass (these bugs are not unit-covered; verification is compile + no regression).

**Step 5: Commit**

```sh
git add src/llama-graph.h src/llama-graph.cpp
git commit -m "routing-predictor : fix dangling logits ptr, honor horizon/depth, count decode only"
```

---

### Task 2: Prediction queue in the expert cache

One pending entry per target layer; latest submission replaces (D3).

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.h` (struct + method decls)
- Modify: `ggml/src/ggml-backend-expert-cache.cpp:2251-2288` (submit stores entry)
- Test: `tests/test-expert-cache.cpp`

**Step 1: Write the failing test**

Append to `tests/test-expert-cache.cpp` (reuse the cache-construction pattern from `test_route_trace`, line 377):

```cpp
static void test_prediction_queue() {
    auto cache = make_test_cache(); // same helper style as test_route_trace
    const int32_t ids[4] = {1, 5, 9, 20};
    ggml_backend_expert_cache_submit_prediction(cache.get(), 10, ids, 4, nullptr);
    // second submit for same layer replaces
    const int32_t ids2[2] = {3, 7};
    ggml_backend_expert_cache_submit_prediction(cache.get(), 10, ids2, 2, nullptr);
    GGML_ASSERT(cache->pending_predictions.size() == 1);
    GGML_ASSERT(cache->pending_predictions[10].n_experts == 2);
    GGML_ASSERT(cache->pending_predictions[10].expert_ids[0] == 3);
}
```

Register it in main() next to the other tests.

**Step 2: Run it to verify it fails**

Run: `cmake --build build --target test-expert-cache && ./build/bin/test-expert-cache`
Expected: FAIL - compile error, `pending_predictions` does not exist.

**Step 3: Implement**

In `ggml/src/ggml-backend-expert-cache.h`, add:

```cpp
struct ggml_expert_cache_pending_prediction {
    int target_layer = -1;
    int n_experts    = 0;
    int32_t expert_ids[64];
};

// members in ggml_backend_expert_cache_context:
std::map<int, ggml_expert_cache_pending_prediction> pending_predictions;
int executed_layer_cursor = -1;

void submit_prediction(int target_layer, const int32_t * expert_ids, int n_experts);
void begin_step();
bool settle_prediction(int layer, const int32_t * actual_ids, int n_actual, size_t pool_stride);
```

In `ggml/src/ggml-backend-expert-cache.cpp`, replace the body of `ggml_backend_expert_cache_submit_prediction` (2251-2288) with storage + keep the debug log + stats bump:

```cpp
auto & e = cache->pending_predictions[target_layer];
e.target_layer = target_layer;
e.n_experts = std::min(n_experts, 64);
memcpy(e.expert_ids, expert_ids, e.n_experts * sizeof(int32_t));
cache->stats.n_predictions_submitted++;
```

Add `begin_step()` clearing the cursor:

```cpp
void ggml_backend_expert_cache_begin_step(ggml_backend_expert_cache_context * cache) {
    cache->executed_layer_cursor = -1;
}
```

Declare both in `ggml/include/ggml-backend.h` next to the existing submit declaration (line ~439):

```cpp
GGML_API void ggml_backend_expert_cache_begin_step(ggml_backend_expert_cache_context * cache);
```

**Step 4: Run tests to verify they pass**

Run: `./build/bin/test-expert-cache`
Expected: PASS including new test.

**Step 5: Commit**

```sh
git add ggml/include/ggml-backend.h ggml/src/ggml-backend-expert-cache.h ggml/src/ggml-backend-expert-cache.cpp tests/test-expert-cache.cpp
git commit -m "expert-cache : store submitted predictions in per-layer queue"
```

---

### Task 3: Settle accounting at layer execution

When a layer actually executes at MUL_MAT_ID interception, classify what happened to its pending prediction (D4/D5).

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.cpp` (implement settle)
- Modify: `ggml/src/ggml-backend.cpp:1787-1830` (call it in interception block)
- Test: `tests/test-expert-cache.cpp`

**Step 1: Write the failing test**

```cpp
static void test_settle_accounting() {
    auto cache = make_test_cache();
    const int32_t pred[4] = {1, 5, 9, 20};
    ggml_backend_expert_cache_submit_prediction(cache.get(), 10, pred, 4, nullptr);
    // simulate: 1 resident, 5 in-flight prefetch, 9+20 never prefetched
    // actual execution requested {5, 9, 33}
    const int32_t actual[3] = {5, 9, 33};
    bool ok = ggml_backend_expert_cache_settle_prediction(cache.get(), 10, actual, 3, 4096*4096*2 /*stride*/);
    GGML_ASSERT(ok);
    const auto & s = cache->stats.routing_predictor;
    GGML_ASSERT(s.predictions_used == 1);
    GGML_ASSERT(s.too_late == 0);
    GGML_ASSERT(s.experts_partially_hidden == 1); // 5 was in-flight
    GGML_ASSERT(s.experts_missed == 2);           // 9, 20 never ready
    GGML_ASSERT(s.predictions_wrong == 1);        // 33 not predicted
}
```

Also assert `too_late` increments when settling a layer <= cursor.

**Step 2: Run to verify failure**

Run: `cmake --build build --target test-expert-cache && ./build/bin/test-expert-cache`
Expected: FAIL - `settle_prediction` undeclared.

**Step 3: Implement settle in the cache**

```cpp
bool ggml_backend_expert_cache_settle_prediction(ggml_backend_expert_cache_context * cache,
        int layer, const int32_t * actual_ids, int n_actual, size_t pool_stride) {
    auto it = cache->pending_predictions.find(layer);
    auto & s = cache->stats.routing_predictor;
    if (it == cache->pending_predictions.end()) {
        return false;
    }
    if (layer <= cache->executed_layer_cursor) {
        s.too_late++;
    }
    cache->executed_layer_cursor = layer;
    const auto & p = it->second;
    s.predictions_used++;
    std::set<int32_t> actual_set(actual_ids, actual_ids + n_actual);
    for (int i = 0; i < p.n_experts; i++) {
        const int32_t id = p.expert_ids[i];
        if (!actual_set.count(id)) {
            continue;
        }
        // classify readiness against prefetch slots / residency
        if (find_slot(cache, id, layer) >= 0) {
            s.experts_fully_hidden++;
        } else if (has_inflight_prefetch(cache, id, layer)) {
            s.experts_partially_hidden++;
        } else {
            s.experts_missed++;
        }
    }
    for (int32_t id : actual_set) {
        bool predicted = false;
        for (int i = 0; i < p.n_experts; i++) {
            predicted |= p.expert_ids[i] == id;
        }
        if (!predicted) {
            s.predictions_wrong++;
            s.bytes_wasted += (int64_t) pool_stride;
        }
    }
    cache->pending_predictions.erase(it);
    return true;
}
```

Notes:
- `find_slot` exists (used by sync prefetch, line ~1343). If no equivalent inflight check exists, track it by scanning `prefetch_slots` for matching tensor/layer state != READY; add a tiny helper `has_inflight_prefetch`.
- `pool_stride` = bytes per expert weight bundle for one layer; compute at call site from the moe weights (gate+up+down sizes) or approximate as `n_ff * n_embd * 2 * sizeof(f16)` initially; refine later.

**Step 4: Hook into interception**

In `ggml/src/ggml-backend.cpp` inside the MUL_MAT_ID interception block, right after the used_ids bitset is built (~line 1764) and before/next-to the reactive record_prediction call (~1811):

```cpp
if (cache && cache->stats.n_predictions_submitted > 0) {
    ggml_backend_expert_cache_settle_prediction(cache, il,
            (const int32_t *) ids_host.data(), n_used, pool_stride_bytes);
}
```

Where `ids_host` is the already-synced ids buffer and `pool_stride_bytes` computed once per graph from the expert weight tensors.

**Step 5: Run tests**

Run: `./build/bin/test-expert-cache`
Expected: PASS.

**Step 6: Commit**

```sh
git add ggml/src/ggml-backend-expert-cache.cpp ggml/src/ggml-backend.cpp tests/test-expert-cache.cpp
git commit -m "expert-cache : settle predictions at layer execution with accuracy accounting"
```

---

### Task 4: Prediction-driven prefetch at submit time

Predicted experts' weights start loading immediately (D2). Reuses `prefetch_async` which dedupes and falls back to sync without CUDA.

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.cpp` (submit triggers prefetch)
- Test: `tests/test-expert-cache.cpp`

**Step 1: Write the failing test**

```cpp
static void test_submit_triggers_prefetch() {
    auto cache = make_test_cache();
    register_test_bundles(cache.get(), /*n_layers*/ 16); // gate/up/down fakes like test_route_trace
    const int32_t ids[2] = {1, 5};
    ggml_backend_expert_cache_submit_prediction(cache.get(), 12, ids, 2, nullptr);
    GGML_ASSERT(cache->stats.n_speculative_prefetches > 0);
}
```

**Step 2: Verify fail**

Run: `cmake --build build --target test-expert-cache && ./build/bin/test-expert-cache`
Expected: FAIL - counter still zero.

**Step 3: Implement**

At the end of the new submit body (Task 2), resolve the target layer's bundle and issue prefetches exactly like `prefetch_layer` (1713) does:

```cpp
for (int t = 0; t < 3; t++) {
    ggml_tensor * w = get_bundle_tensor(cache, target_layer, t); // gate/up/down lookup as in prefetch_layer
    if (!w) continue;
    prefetch_async(cache, w, e.expert_ids, e.n_experts, target_layer);
}
```

If `prefetch_layer` internals are not directly callable per-tensor, factor a small helper out of it rather than duplicating logic (DRY).

**Step 4: Verify pass**

Run: `./build/bin/test-expert-cache`
Expected: PASS.

**Step 5: Commit**

```sh
git add ggml/src/ggml-backend-expert-cache.cpp tests/test-expert-cache.cpp
git commit -m "expert-cache : prefetch predicted experts at submit time"
```

---

### Task 5: Real stats getter + sched aggregation + delete dead metrics

**Files:**
- Modify: `ggml/src/ggml-backend.cpp:2410-2421` (implement getter)
- Modify: `ggml/include/ggml-backend.h` (declare sched-level aggregate if missing)
- Modify: `src/llama-graph.h:930-949`, `src/llama-graph.cpp` (delete routing_predictor_metrics; move generated-count readout)

**Step 1: Write the failing test**

In `tests/test-expert-cache.cpp`:

```cpp
static void test_stats_getter() {
    auto cache = make_test_cache();
    ggml_routing_predictor_stats s = {};
    GGML_ASSERT(!ggml_backend_expert_cache_get_routing_predictor_stats(cache.get(), &s));
    const int32_t ids[1] = {1};
    ggml_backend_expert_cache_submit_prediction(cache.get(), 3, ids, 1, nullptr);
    GGML_ASSERT(ggml_backend_expert_cache_get_routing_predictor_stats(cache.get(), &s));
    GGML_ASSERT(s.predictions_generated == 0); // generated counted upstream, not here
    GGML_ASSERT(s.too_late == 0);
}
```

For the sched-level getter, verify via llama-bench smoke later (Task 11); unit coverage stays at cache level.

**Step 2: Verify fail**

Run: `cmake --build build --target test-expert-cache && ./build/bin/test-expert-cache`
Expected: FAIL - getter returns false today.

**Step 3: Implement**

Cache-level getter copies the field set from `ggml_routing_predictor_stats` (declared `ggml/include/ggml-backend.h:364-373`). Sched-level `ggml_backend_sched_get_routing_predictor_stats` sums across `sched->expert_caches[b]`, copying the loop pattern of `ggml_backend_sched_get_expert_cache_stats` (`ggml/src/ggml-backend.cpp:2423+`). Add `predictions_generated` accumulation: the graph callback knows the count; simplest correct path is to have the callback forward its decode-only count into the primary cache via a small setter, e.g. `ggml_backend_expert_cache_add_predictions_generated(cache, n)`, called right after predict succeeds. Then the sched getter is the single aggregation point (D1).

Delete from `src/llama-graph.h` the `routing_predictor_metrics` struct and its member; update the two use sites in `src/llama-graph.cpp` (increment site from Task 1 becomes the setter call above; remove any other references).

**Step 4: Verify pass + bench compiles**

Run: `./build/bin/test-expert-cache && cmake --build build --target llama-bench`
Expected: tests PASS, llama-bench builds (its CSV helpers now hit real data).

**Step 5: Commit**

```sh
git add ggml/include/ggml-backend.h ggml/src/ggml-backend.cpp src/llama-graph.h src/llama-graph.cpp tests/test-expert-cache.cpp
git commit -m "routing-predictor : implement stats getters, single source of truth in expert cache"
```

---

### Task 6: Eval callback chaining (fix A1 clobbering)

**Files:**
- Modify: `src/llama-graph.h` (add prev_cb fields to llm_graph_result)
- Modify: `src/llama-context.cpp:1395-1410` (capture then chain)
- Modify: `src/llama-graph.cpp:3978` (forward at end of routing_predictor_callback)

**Step 1: No unit harness exists for context callbacks; verification = build + determinism run in Task 11.**

**Step 2: Implement**

In `struct llm_graph_result`:

```cpp
ggml_backend_sched_eval_callback prev_cb = nullptr;
void * prev_cb_user_data = nullptr;
```

In `src/llama-context.cpp`, replace the unconditional overwrite:

```cpp
res->prev_cb           = cparams.cb_eval;
res->prev_cb_user_data = cparams.cb_eval_user_data;
cparams.cb_eval = llm_graph_context::routing_predictor_callback;
cparams.cb_eval_user_data = res.get();
ggml_backend_sched_set_eval_callback(lctx.sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
```

(match the actual param names/types present at lines 1395-1410.)

In `routing_predictor_callback` (src/llama-graph.cpp:3978), before every `return`:

```cpp
if (res->prev_cb) {
    return res->prev_cb(t, ask, res->prev_cb_user_data);
}
return true;
```

**Step 3: Build**

Run: `cmake --build build --target llama-cli llama-bench`
Expected: clean build.

**Step 4: Commit**

```sh
git add src/llama-graph.h src/llama-context.cpp src/llama-graph.cpp
git commit -m "routing-predictor : chain instead of clobbering user eval callback"
```

---

### Task 7: Route trace v2 with router logits + env-var trigger

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.h:278-284` (entry struct + version)
- Modify: `ggml/src/ggml-backend-expert-cache.cpp:1897-1981` (record logits, bump version)
- Modify: `ggml/src/ggml-backend.cpp` (env var trigger at cache creation)
- Test: `tests/test-expert-cache.cpp`

**Step 1: Write the failing test**

Extend the route-trace test pattern:

```cpp
static void test_route_trace_v2_logits() {
    auto cache = make_test_cache();
    ggml_backend_expert_cache_enable_route_trace(cache.get(), path.c_str());
    float logits[8] = {0}; logits[3] = 5.f;
    ggml_backend_expert_cache_record_route_trace(cache.get(), 42, 2, ids, 8, logits);
    ggml_backend_expert_cache_flush_route_trace(cache.get());
    // reopen file, assert version == 2 and entry payload contains the logits blob
}
```

**Step 2: Verify fail**

Run: `cmake --build build --target test-expert-cache && ./build/bin/test-expert-cache`
Expected: FAIL - signature/format mismatch.

**Step 3: Implement**

- Bump trace file version constant to 2. New entry layout: existing header fields + `int32 n_logits` + `float logits[n_logits]` appended after expert_ids. Keep old readers out of scope; collect script is updated in Task 8.
- `record_route_trace` gains a `const float * logits, int n_logits` parameter (nullable => writes zeros-length blob for compat).
- Env trigger in `ggml/src/ggml-backend.cpp` where each `expert_caches[b]` is constructed:

```cpp
const char * trace_path = getenv("GGML_EXPERT_ROUTE_TRACE");
if (trace_path && trace_path[0]) {
    ggml_backend_expert_cache_enable_route_trace(cache, trace_path);
}
```

Flush/disable stays manual (atexit or server shutdown is fine for now; document that traces flush every 10000 entries anyway).

**Step 4: Verify pass**

Run: `./build/bin/test-expert-cache`
Expected: PASS.

**Step 5: Commit**

```sh
git add ggml/src/ggml-backend-expert-cache.h ggml/src/ggml-backend-expert-cache.cpp ggml/src/ggml-backend.cpp tests/test-expert-cache.cpp
git commit -m "expert-cache : route trace v2 captures router logits, env-var trigger"
```

---

### Task 8: Update collect_router_features.py for v2

**Files:**
- Modify: `tools/collect_router_features.py`

**Step 1: Inspect current parser** (read the script; it converts route_trace.bin -> training_data.bin per its docstring).

**Step 2: Update**

- Accept version 2 records; parse optional logits blob.
- Output rows: `{token_id, layer, logits: float[n_logits], future_ids: ids[layer+horizon]}` pairing each layer's logits with the ids observed `horizon` layers later within the same token (this is the supervised target for variant B).
- Keep CLI shape identical; add `--horizon` (default 8).

**Step 3: Smoke test**

Generate a tiny trace from the unit test (Task 7 writes one), run:

Run: `python tools/collect_router_features.py trace.bin out.bin --horizon 8`
Expected: exits 0, non-empty out.bin, row count = entries with a valid layer+horizon partner.

**Step 4: Commit**

```sh
git add tools/collect_router_features.py
git commit -m "tools : parse route trace v2 with logits in feature collector"
```

---

### Task 9: Trainer emits LRPD binary the C++ loader reads

**Files:**
- Modify: `tools/train_routing_predictor.py`

**Step 1: Read the loader contract first**

`ggml/src/ggml-routing-predictor.cpp:70` `load_model` defines the exact expected layout. Match it byte-for-byte: magic `LRPD` (0x4C525044), header `{input_dim, num_experts, horizon, rank}`, then down_weight/bias, output_weight/bias (verify order/dtypes/shapes in code before writing anything).

**Step 2: Extend trainer**

- Input: training_data.bin from Task 8.
- Model: low-rank MLP (numpy): x -> down (rank trunk) -> gelu -> up (num_experts). Init per handover appendix: W_down = W_router[L+H] - W_router[L] delta fit, or plain regression onto future one-hot targets when router weights unavailable.
- Loss: softmax cross-entropy against future top-n_expert_used ids; report train recall@K.
- Save LRPD via struct.pack little-endian floats.

**Step 3: Round-trip test**

Train on synthetic data (random logits -> planted linear map), then load the .bin with a tiny C++ test or by running `test-routing-predictor` extended with a load-and-predict case:

Add to `tests/test-routing-predictor.cpp`:

```cpp
static void test_load_lprd_and_predict() {
    // write minimal valid LRPD file with known weights
    // init LOW_RANK_MLP with model_path, load_model, predict, assert argmax matches hand-computed value
}
```

Run: `cmake --build build --target test-routing-predictor && ./build/bin/test-routing-predictor`
Expected: FAIL first (no file handling change needed? if loader already works, this test may pass immediately - then it is a characterization test; keep it either way).

**Step 4: Commit**

```sh
git add tools/train_routing_predictor.py tests/test-routing-predictor.cpp
git commit -m "tools : train low-rank MLP predictor, emit LRPD binary"
```

---

### Task 10: CLI for model path + variant selection

**Files:**
- Modify: `common/arg.cpp:2800-2815` (new arg)
- Modify: `include/llama.h:417-420` (param field)
- Modify: `src/llama-cparams.h:73-74` (cparams field)
- Modify: `src/llama-context.cpp` (propagate)
- Modify: `src/llama-graph.cpp:1491-1524` (use type/model_path)
- Modify: `tools/llama-bench/llama-bench.cpp` (CLI vector + CSV column `predictor_model` tag)

**Step 1: Implement plumbing**

- `--routing-predictor-model PATH` (string, default empty) and `--routing-predictor-variant {stale-future,low-rank-mlp,future-residual}` (default stale-future) in arg.cpp next to the existing pair.
- Thread through llama_context_params -> cparams -> config in the graph constructor:

```cpp
config.type      = cparams.routing_predictor_variant;
config.model_path = cparams.routing_predictor_model.empty() ? nullptr : cparams.routing_predictor_model.c_str();
```

- On `load_model` failure: log warning, fall back to STALE_FUTURE with model_path=nullptr (never crash a benchmark run).

**Step 2: Bench wiring**

Mirror the existing horizon/stats vectors in llama-bench (lines 359-360, 1119-1122, 1258-1262, 1381-1382): add `routing_predictor_model` string vector (empty default) and include in the params tuple + CSV output column.

**Step 3: Build + smoke**

Run: `cmake --build build --target llama-bench llama-server && ./build/bin/llama-bench -m <model> -p 0 -n 32 --routing-predictor-horizon 8 --routing-predictor-stats`
Expected: runs; CSV contains stats columns with nonzero predictions_generated on decode.

**Step 4: Commit**

```sh
git add common/arg.cpp include/llama.h src/llama-cparams.h src/llama-context.cpp src/llama-graph.cpp tools/llama-bench/llama-bench.cpp
git commit -m "routing-predictor : CLI for model path and variant selection with safe fallback"
```

---

### Task 11: Benchmark + determinism validation

**Files:**
- Modify: `scripts/expert-cache-determinism-matrix.py` (add row F)

**Step 1: Add matrix row F**

Following the existing row pattern (A baseline / B cache / E MTP):

```python
Row("F", label="cache+predictor", extra_args=["--routing-predictor-horizon", "8", "--routing-predictor-stats"]),
```

(passthrough mechanism already exists via --extra-args; match however rows B/E declare args.)

**Step 2: Run determinism matrix**

Run: `python scripts/expert-cache-determinism-matrix.py --model models/Qwen3.6-35B-A3B-APEX-Compact.gguf`
Expected: SHA-256 of generated tokens identical across rows A/B/F. Any divergence = predictor path perturbing numerics => investigate before proceeding (likely callback ordering; must be fixed).

**Step 3: Run perf comparison**

Three configs, same prompt set:

```sh
./build/bin/llama-bench -m <model> -p 0 -n 512 -t 14 -ngl 99
./build/bin/llama-bench -m <model> -p 0 -n 512 -t 14 -ngl 99 -exc 64 -excp 64
./build/bin/llama-bench -m <model> -p 0 -n 512 -t 14 -ngl 99 -exc 64 -excp 64 --routing-predictor-horizon 8 --routing-predictor-stats
```

Record: tok/s, and from CSV stats columns compute recall@K proxy = experts_fully_hidden+partially_hidden vs missed, ready_rate, bytes_wasted delta vs row B.

**Step 4: Record results**

Append a results section to `docs/plans/2026-08-22-learned-predictor-handover.md` (numbers + hashes). Success criteria: determinism holds; row C shows nonzero fully_hidden/partially_hidden; tok/s not worse than row B beyond noise.

**Step 5: Commit**

```sh
git add scripts/expert-cache-determinism-matrix.py docs/plans/2026-08-22-learned-predictor-handover.md
git commit -m "scripts : add predictor row to determinism matrix, record validation results"
```

---

## Out of scope (explicitly deferred)

- Variant C (future-residual) runtime tuning - loader/forward exist; needs trained model first.
- Async D2H of logits (removing sync tensor_get) - pinned buffer already mitigates; measure first.
- Vulkan/Metal predictor backend placement - currently CPU-side predict is fine at these sizes.
- Hidden-state trace usage - too large; logits v2 supersedes it.

## Risk notes

- Task 3 settle hook sits in the hot interception path; keep it branch-cheap (early-out when no pending entry).
- Trace v2 grows files by ~n_logits*4 bytes/entry; cap trace duration during collection runs.
- GTX 1080 has no async-copy-friendly sm>60 features relied upon elsewhere; prefetch_async already falls back to sync - behavior is correct, just less overlapped.
