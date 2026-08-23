# Expert-Cache Zero-Copy Decode Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the slot lifecycle, prefetch routing, staging-memory, VRAM-budget, and predictor bugs in the single-token zero-copy expert-cache path so its results are bit-identical to the generic path.

**Architecture:** Two files carry the change surface: `ggml/src/ggml-backend-expert-cache.cpp` (slot pools, prefetch pipeline, staging ring, predictor) and `ggml/src/ggml-backend.cpp` (`compute_splits` zero-copy consumer + same-backend route discovery). We introduce an explicit `EMPTY -> LOADING -> RESIDENT` slot state so `find_slot` never returns in-flight data, funnel all predictive DMA through the target layer's registered bundle, replace the append-only `prefetch_slots` vector with a generation-keyed map, and enforce one global byte budget across all slot pools.

**Tech Stack:** C++17, ggml backend abstraction, CUDA events/streams (guarded by `GGML_USE_CUDA`), existing test harness `tests/test-expert-cache.cpp` (plain `require()` asserts, built via CMake `llama_build_and_test`).

## Global Constraints

- Repo rule: NEVER run `git commit` or `git push`. At each commit step, propose the message to the user and wait for explicit approval.
- No unicode arrows/emdash in code or comments; ASCII only.
- Comments: 1-2 lines max, ASD-STE100 plain wording, no fixed-column wrapping.
- All CUDA-specific code stays inside `#if defined(GGML_USE_CUDA)`; the non-CUDA branch must still compile (CI builds without CUDA).
- Do not change public behavior of the multi-token/prefill sparse-transfer path except where a task explicitly says so.
- Verification model for smoke tests: Qwen3.6-35B-A3B-APEX-Compact.gguf on GTX 1080; unit tests must pass without any GPU (CPU backend).
- Working tree already contains uncommitted fixes for review items #1 (`remapped_ids[i] = slot`) and #5 (restore loop). This plan does NOT re-do those; it covers the remaining verified defects only.

---

### Task 1: Slot lifecycle - find_slot must not return in-flight slots

The root bug: `alloc_slot_idx()` publishes `(tensor, expert_id) -> slot` into `pool->key_to_slot` immediately, but the data arrives later (async H2D). `find_slot()` then reports residency before the copy completes. The zero-copy consumer at `ggml-backend.cpp:1988` trusts `find_slot >= 0` as "ready", which is wrong for in-flight slots.

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.h` (add state enum field to slot entry struct)
- Modify: `ggml/src/ggml-backend-expert-cache.cpp:872-1003` (`find_slot`, `alloc_slot_idx`, plus every site that publishes or consumes `key_to_slot`)
- Test: `tests/test-expert-cache.cpp`

**Interfaces:**
- Consumes: existing `ggml_expert_cache_key`, `ggml_expert_cache_slot_entry` structs; `ggml_backend_expert_cache_find_slot`, `ggml_backend_expert_cache_alloc_slot_idx`.
- Produces:
  - `enum class ggml_expert_slot_state { EMPTY, LOADING, RESIDENT };`
  - New field `ggml_expert_slot_state state` on the slot entry struct (default `EMPTY`).
  - New function `void ggml_backend_expert_cache_mark_resident(ggml_backend_expert_cache_t cache, const struct ggml_tensor * tensor, int32_t expert_id);` - flips `LOADING -> RESIDENT` after a fill completes.
  - `find_slot` returns `-1` unless state is `RESIDENT`.
  - New function `int32_t ggml_backend_expert_cache_find_or_loading_slot(...)` with the same signature as `find_slot`, returning the slot even when `LOADING` (needed by Task 2's dedupe logic).

- [ ] **Step 1: Write the failing test**

Add to `tests/test-expert-cache.cpp` (reuse the harness style of `test_slot_pools_and_remapping`):

```cpp
static void test_loading_slot_not_visible_as_resident() {
    // alloc publishes a mapping; until mark_resident, find_slot must be -1
    // and the remap helper must not report a hit for that expert.
    struct ggml_init_params params = { 4 * ggml_tensor_overhead(), NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 8);
    int32_t remapped[2] = { -1, -1 };
    bool hit[2] = { false, false };
    const int32_t ids_in[2] = { 3, 5 };

    int32_t s = ggml_backend_expert_cache_alloc_slot_idx(cache, w, 3, NULL, 0);
    assert(s >= 0);
    // no data written yet: resident lookup fails, loading lookup succeeds
    assert(ggml_backend_expert_cache_find_slot(cache, w, 3) == -1);
    assert(ggml_backend_expert_cache_find_or_loading_slot(cache, w, 3) == s);

    ggml_backend_expert_cache_mark_resident(cache, w, 3);
    assert(ggml_backend_expert_cache_find_slot(cache, w, 3) == s);

    // remap helper: resident expert hits, unknown expert misses
    ggml_backend_expert_cache_remap_ids(cache, w, ids_in, 2, remapped, hit);
    assert(hit[0] == true && remapped[0] == s);
    assert(hit[1] == false);

    ggml_free(ctx);
}
```

Register it in `main()` next to the other test calls:

```cpp
test_loading_slot_not_visible_as_resident();
```

Note: `cache` and `backend` are file-scope helpers created by the existing tests; follow whatever setup pattern `test_slot_pools_and_remapping` uses (it creates them at lines 83-102). `remap_ids` refers to the exported remap function already used by that test (`ggml_backend_expert_cache_*_remapped_ids` - use the exact symbol it links against).

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target test-expert-cache && .\build\bin\test-expert-cache.exe`
Expected: FAIL (compile error: `find_or_loading_slot` / `mark_resident` undefined).

- [ ] **Step 3: Implement minimal changes**

In the header, add the enum and extend the slot entry struct with `state`. In `ggml-backend-expert-cache.cpp`:

1. Every construction `pool->slots[s] = { tensor, expert_id, layer, SEG_PROBATIONARY, clock, 0, 1 }` (three sites: empty-slot path line ~919, eviction path ~993, plus any other) gains trailing `, GGML_EXPERT_SLOT_LOADING` in aggregate-init order matching the new field position.
2. `find_slot`: after locating `key_to_slot`, return `-1` if `state != RESIDENT`.
3. New `find_or_loading_slot`: identical body minus the state check.
4. New `mark_resident`: look up key, set `slots[s].state = RESIDENT`.
5. Audit all `alloc_slot_idx` callers: `prefetch_async` (~1479), `prefetch` (~1393), reactive fill in `ggml-backend.cpp` (~2028), seeding paths (~1970, ~2073). Each must call `mark_resident` exactly when its copy has been enqueued such that stream ordering makes data visible to the consumer node BEFORE the graph executes. For backend-ordered fills (`ggml_backend_tensor_set_async` onto the consuming backend) call `mark_resident` immediately after enqueue - the enqueue-before-consume ordering guarantees correctness. For `prefetch_stream` copies (CUDA branch, Task 2), do NOT mark resident there; Task 2 wires completion-event-driven marking.

- [ ] **Step 4: Run tests**

Run: `cmake --build build --target test-expert-cache && .\build\bin\test-expert-cache.exe`
Expected: PASS (all pre-existing tests too - they exercise alloc+fill+find sequences and will catch a missed `mark_resident`).

Also rebuild everything to catch compile breaks in other TUs:

Run: `cmake --build build --target test-moe-latency-oracle test-expert-cache-profile`
Expected: PASS

- [ ] **Step 5: Propose commit (DO NOT run git commit)**

Message to propose:

```
ggml : expert cache slot lifecycle (EMPTY/LOADING/RESIDENT)

find_slot no longer reports in-flight prefills as resident;
fills must call mark_resident once their copy is ordered
before consumption.
```

Wait for user approval before committing.

---

### Task 2: Prefetch failure rollback + event-driven residency + dedup map

`prefetch_async` currently: creates the CUDA event AFTER `alloc_slot_idx` published the mapping (event-create failure leaks a phantom slot), skips on missing pinned buffer while keeping the mapping, and pushes records into an append-only vector where stale entries can shadow newer generations. It also never marks slots RESIDENT (Task 1 left that to this task for the CUDA path).

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.cpp:127-135` (struct), `1424-1563` (`prefetch_async`), `1577-1660` (lookup/wait helpers)
- Test: `tests/test-expert-cache.cpp`

**Interfaces:**
- Consumes: Task 1's `mark_resident`, `find_or_loading_slot`, slot `state` field.
- Produces:
  - Field change: `std::vector<ggml_expert_cache_prefetch_slot> prefetch_slots` becomes `std::unordered_map<ggml_expert_cache_key, ggml_expert_cache_prefetch_slot, ggml_expert_cache_key_hash, ggml_expert_cache_key_eq> prefetch_slots` (keyed map holds exactly the current generation).
  - `ggml_expert_cache_prefetch_slot` gains `bool rolled_back` handling via erasure instead: failed setups erase the map entry.
  - All four consumers of the old vector (`was_prefetched` ~1581, `is_prefetch_ready` ~1610, `wait_prefetch` ~1643, `has_inflight_prefetch` static helper ~2553) switch to map lookup by `{tensor, expert_id}`.
  - On CUDA-path completion detection (`is_prefetch_ready` seeing `cudaSuccess`, `wait_prefetch` after sync): call `ggml_backend_expert_cache_mark_resident(cache, tensor, expert_id)` then ERASE the map entry (residency now lives in the pool).
  - Non-CUDA branch: `mark_resident` immediately after `tensor_set_async` enqueue, then erase entry.

- [ ] **Step 1: Write the failing test**

```cpp
static void test_prefetch_record_lifecycle() {
    // After issuing a non-CUDA prefetch, was_prefetched is true; after
    // marking resident the record is gone from the pending map and
    // find_slot sees the slot.
    struct ggml_init_params params = { 8 * ggml_tensor_overhead(), NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 8);
    // fill host source with known bytes
    std::vector<uint8_t> src(ggml_nbytes(w) / w->ne[2], 0xAB);
    w->buffer = /* host buffer owning src, mirroring existing tests */ nullptr;

    const int32_t ids[1] = { 6 };
    ggml_backend_expert_cache_prefetch_async(cache, w, ids, 1, 0);
    assert(ggml_backend_expert_cache_was_prefetched(cache, w, 6));
    int32_t s = ggml_backend_expert_cache_find_or_loading_slot(cache, w, 6);
    assert(s >= 0);
    assert(ggml_backend_expert_cache_find_slot(cache, w, 6) == -1); // still loading

    ggml_backend_expert_cache_wait_prefetch(cache, w, 6);
    assert(ggml_backend_expert_cache_find_slot(cache, w, 6) == s);  // resident now
    ggml_free(ctx);
}
```

(Adapt buffer setup to however existing tests obtain a host-backed tensor; `test_slot_pools_and_remapping` shows the pattern.)

Second assertion pair, appended to the same test - failure rollback cannot happen without forcing an allocation failure, so instead test the dedupe invariant directly: calling `prefetch_async` twice for the same expert must not grow the record count:

```cpp
    ggml_backend_expert_cache_prefetch_async(cache, w, ids, 1, 0);
    size_t n_before = ggml_backend_expert_cache_prefetch_slot_count(cache);
    ggml_backend_expert_cache_prefetch_async(cache, w, ids, 1, 0);
    assert((size_t)ggml_backend_expert_cache_prefetch_slot_count(cache) <= n_before);
```

(`prefetch_slot_count` already exists at line ~1577 and keeps working over the map.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target test-expert-cache && .\build\bin\test-expert-cache.exe`
Expected: FAIL - `wait_prefetch` does not currently make the slot resident-visible via `find_slot`.

- [ ] **Step 3: Implement**

In `prefetch_async`:

1. Move the dedupe check first: `auto dup = cache->prefetch_slots.find(key); if (dup != end && dup->second.state == IN_FLIGHT) continue;`
2. Reorder: allocate slot, create CUDA event, acquire staging buffer - ALL BEFORE anything is published. If event creation fails OR staging acquisition fails: free the slot back (erase nothing needed yet since publication happens last) and continue WITHOUT ever inserting into `key_to_slot`. Concretely: perform `alloc_slot_idx` LAST, right before the memcpy+copy issue; on any earlier failure just `continue` (no rollback needed because nothing was published). On failure after `alloc_slot_idx` (only possible inside memcpy region - none), remove the key from `pool->key_to_slot` and decrement `used_slots`/segment counters symmetrically.
3. Insert into `cache->prefetch_slots[key] = {...}` after successful issue.
4. Completion transitions per "Produces" above.

Non-CUDA branch: same reorder, then `mark_resident` + erase immediately (stream ordering guarantees visibility).

- [ ] **Step 4: Run tests**

Run: `cmake --build build --target test-expert-cache && .\build\bin\test-expert-cache.exe`
Expected: PASS

- [ ] **Step 5: Smoke on GPU (manual, requires GTX 1080 + model)**

Run: server or llama-bench decode with `--expert-cache` enabled as in prior Phase 5 runs; confirm `n_zero_copy_hits > 0` and no crash/hang over 128 generated tokens.

- [ ] **Step 6: Propose commit (DO NOT run git commit)**

```
ggml : expert cache prefetch lifecycle and dedupe

Failed prefetch setup no longer leaves phantom slots;
prefetch records are keyed per expert and retired on
completion instead of accumulating.
```

---

### Task 3: Predictive prefetch must use the target layer's tensors

Two call sites predict experts for `layer + 1` but pass the CURRENT layer's weight tensor to `prefetch_async`, which ignores `target_layer` entirely and uses the supplied tensor for both pool selection and source data - so it uploads current-layer weights under next-layer keys. `submit_prediction` (line 2544) already does the correct thing via `bundle_registrations[target_layer]`; route discovery duplicates the broken pattern at `ggml-backend.cpp:1739-1742` and `compute_splits` at `1939-1947`.

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.cpp:2496-2551` (`submit_prediction`)
- Delete: direct `predict_experts` + `prefetch_async` pairs in `ggml/src/ggml-backend.cpp:1938-1947` and `1735-1742`
- Test: `tests/test-expert-cache.cpp`

**Interfaces:**
- Consumes: `bundle_registrations[layer] = {gate, up, down}` populated by `ggml_backend_expert_cache_register_bundle` (~1252); Task 2's keyed `prefetch_slots`.
- Produces: `submit_prediction(cache, target_layer, expert_ids, n, confidences)` becomes the ONLY predictive-DMA entry point. Route discovery and compute_splits keep calling `record_prediction` (learning) but replace their local `predict_experts + prefetch_async` blocks with `predict_experts` + `submit_prediction(cache, layer + 1, predicted, n_predicted, NULL)`.

- [ ] **Step 1: Write the failing test**

```cpp
static void test_prediction_targets_bundle_tensors() {
    // register bundles for layers 0 and 1 with distinguishable tensors,
    // submit a prediction targeting layer 1, verify slots were allocated
    // for the LAYER 1 tensors (not layer 0's).
    struct ggml_init_params params = { 16 * ggml_tensor_overhead(), NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * l0w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 8);
    struct ggml_tensor * l1w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 8);

    ggml_backend_expert_cache_register_bundle(cache, 0, l0w, NULL, NULL);
    ggml_backend_expert_cache_register_bundle(cache, 1, l1w, NULL, NULL);

    const int32_t preds[2] = { 2, 7 };
    ggml_backend_expert_cache_submit_prediction(cache, 1, preds, 2, NULL);

    // layer 0 untouched, layer 1 has loading/resident slots
    assert(ggml_backend_expert_cache_find_or_loading_slot(cache, l0w, 2) == -1);
    assert(ggml_backend_expert_cache_find_or_loading_slot(cache, l1w, 2) >= 0);
    assert(ggml_backend_expert_cache_find_or_loading_slot(cache, l1w, 7) >= 0);
    ggml_free(ctx);
}
```

Check the exact signature of `register_bundle` in the header before writing the call (argument order gate/up/down).

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target test-expert-cache && .\build\bin\test-expert-cache.exe`
Expected: FAIL - today `submit_prediction` targets the right bundle, but nothing exercises it here; more importantly the sched call sites bypass it. If the test passes immediately, strengthen it by asserting `find_or_loading_slot(l0w, ...)` stays `-1` after driving the sched-side helper - otherwise rely on Step 3's call-site deletion for the real fix.

- [ ] **Step 3: Implement**

1. In `compute_splits` (ggml-backend.cpp ~1939-1947): replace the `prefetch_async(cache, input, ...)` call with `submit_prediction(cache, next_layer, predicted_experts, n_predicted, NULL)` guarded on bundle presence (submit_prediction already no-ops when no bundle is registered).
2. Same replacement in `route_discovery` (~1739-1742).
3. In `submit_prediction`, add the host-source guard currently only in `prefetch_async`: skip bundles whose tensors are not host-buffer-backed (they cannot be staged) - mirror lines 1436-1439.

- [ ] **Step 4: Run tests + smoke**

Run: `cmake --build build --target test-expert-cache && .\build\bin\test-expert-cache.exe`
Expected: PASS

GPU smoke as Task 2 Step 5; additionally check log stats: `n_from_pred` should rise relative to baseline while outputs stay identical.

- [ ] **Step 5: Propose commit (DO NOT run git commit)**

```
ggml : route predictions through target-layer bundles

Predictive prefetch now resolves gate/up/down tensors from
bundle_registrations[target_layer] instead of uploading the
current layer under future keys.
```

---

### Task 4: Per-node remapped IDs tensor (shared router IDs)

Zero-copy writes slot indices into the ORIGINAL router IDs tensor via `ggml_backend_tensor_set_async(split_backend, ids_tensor, remapped_ids...)` (ggml-backend.cpp:2088-2092). Gate/up/down MUL_MAT_ID nodes can share that ids tensor while drawing from different slot pools whose logical expert N maps to different physical slots - one shared rewrite cannot serve three mappings. Also, rewriting the shared graph tensor corrupts the values other nodes read.

Fix: never mutate `ids_tensor`. Build/cache a small remapped-ids tensor per (MUL_MAT_ID node, weight tensor) pair, redirect `node->src[2]` alongside `node->src[0]`, and restore both after the split (extend `restored_nodes` to pairs covering src[0] and src[2]).

**Files:**
- Modify: `ggml/src/ggml-backend.cpp:1964-2118` (zero-copy block) and `2347-2350` (restore loop)
- Modify: `ggml/src/ggml-backend.hpp` or sched private struct (wherever `expert_ids_scratch` etc. live - search `remapped_ids_scratch`) for a per-node remap-tensor cache
- Test: `tests/test-expert-cache.cpp` (unit-level) + end-to-end matrix below

**Interfaces:**
- Consumes: existing scratch vectors; Task 1 states.
- Produces:
  - In the sched struct: `std::unordered_map<const ggml_tensor *, ggml_tensor *> expert_remap_ids_cache;` keyed by the original `ids_tensor` pointer PER weight tensor context - since two weight tensors sharing ids need different remaps, key by `{node->src[0]-original-input_cpy, ids_tensor}` composite or simply create one remap tensor per zero-copy invocation cached on the input_cpy pointer.
  - Restore loop extended: `restored_nodes` becomes `std::vector<std::tuple<ggml_tensor *, ggml_tensor *, ggml_tensor *>>` of `{node, orig_src0, orig_src2}`.

- [ ] **Step 1: Write the failing test (end-to-end, CPU backend)**

This needs an actual scheduler-level graph, following the style of `tests/test-moe-latency-oracle.cpp` (which drives real graphs). Add `tests/test-expert-cache-shared-ids.cpp`:

Build a tiny graph: one shared router-ids tensor feeding THREE MUL_MAT_ID nodes (gate/up/down shapes: e.g. (8,32)x(8,4,4), (8,32)x(8,4,4), (16,32)... keep down a different shape to force a second pool), register the cache, warm experts of pool A, run through `ggml_backend_sched`, and compare outputs against the same graph executed with the cache detached. Bit-identical is the assertion.

Skeleton (fill exact API usage from oracle test):

```cpp
// two pools: gate/up share shape, down differs
// shared ids tensor: ids = {3} single token
// expected: outputs equal cache-off run within exact equality
static void test_shared_ids_two_pools() {
    // build graph A: full weights, run, snapshot outputs
    // build graph B: same logical weights routed via expert cache with
    //   slot pools warmed for gate/up but NOT down (down takes miss path)
    // compare
}
```

Because the plan must not hand-wave: implement by copying `test-moe-latency-oracle.cpp`'s graph-construction main flow, replacing its single MUL_MAT_ID with three nodes sharing `ids`, and adding a reference run with `GGML_EXPERT_CACHE_DISABLE=1` (verify that env knob exists; if absent, construct the reference by running the same graph before registering the cache in a fresh process section of the same binary).

Run: `cmake --build build --target test-expert-cache-shared-ids && .\build\bin\test-expert-cache-shared-ids.exe`
Expected: FAIL today (outputs diverge or garbage when gate/up remaps differ from down remaps).

- [ ] **Step 2: Implement per-node remap tensors**

In the zero-copy success branch (ggml-backend.cpp ~2082-2117):

1. Remove the `ggml_backend_tensor_set_async(split_backend, ids_tensor, remapped_ids.data(), ...)`.
2. Obtain-or-create remap tensor: shape `{ids ne[0], ne[1], 1}`, type I32, allocated on split_backend via a small persistent per-sched ggml context (allocate once per unique `ne` combination; reuse across steps since contents are rewritten each step).
3. Copy `remapped_ids` into it each invocation (`ggml_backend_tensor_set_async`), then `node->src[2] = remap_tensor;` and record `{node, input_cpy, ids_tensor}` into `restored_nodes`.
4. Extend restore loop to restore both srcs.
5. Guard: only take zero-copy when `node->src[2]` is not itself consumed by another op in the same graph view being computed (in practice MUL_MAT_ID ids have no other consumers; assert this during development with a debug scan, drop after verification).

- [ ] **Step 3: Run tests**

Run: `cmake --build build --target test-expert-cache test-expert-cache-shared-ids && .\build\bin\test-expert-cache.exe && .\build\bin\test-expert-cache-shared-ids.exe`
Expected: PASS both.

- [ ] **Step 4: End-to-end equivalence matrix (GPU smoke)**

Run decode with cache on vs off (-n 64, fixed seed, temperature 0):
`llama-cli -m <model> -p "The capital of France is" -n 64 -t 8 --temp 0 --seed 42` with and without the expert-cache flag.
Expected: identical token streams. Log the comparison output.

- [ ] **Step 5: Propose commit (DO NOT run git commit)**

```
ggml : per-node remapped ids for zero-copy MoE decode

Shared router-id tensors are no longer mutated; each
MUL_MAT_ID node gets its own slot-index ids tensor, restored
after the split.
```

---

### Task 5: Truly pinned staging memory + correct event ownership

`get_pinned_buffer` uses `_aligned_malloc`/`posix_memalign` - pageable RAM despite the name - so every staging memcpy goes through pageable bounce buffering. And `stage_commit` records the guard event on `cache->backend` while the CUDA prefetch path issues its copy on `prefetch_stream`; `stage_acquire` may then reuse a block whose prefetch-stream copy has not completed.

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.cpp:22-26` (constants area), `1281-1368` (staging functions), struct fields near line 195
- Test: covered by Task 2's GPU smoke + a targeted unit assertion

**Interfaces:**
- Produces:
  - Pinned allocation: on Windows `_aligned_malloc` is replaced by `cudaHostAlloc(&ptr, size, cudaHostAllocDefault)` when compiled with CUDA; fall back to current aligned malloc otherwise. Grow/free path mirrors this (`cudaFreeHost`). Track a bool `pinned_is_cuda` next to the buffer fields.
  - Staging ring events become CUDA events recorded on `prefetch_stream` in the CUDA path. Concretely: `stage_commit` gains a parameter or internal branch - simplest correct form: give the cache TWO event sets? No: keep ONE ring, but make `stage_commit` record on whichever stream the caller just used. Add `enum stage_stream { STAGE_BACKEND, STAGE_PREFETCH }` parameter; CUDA prefetch path passes `STAGE_PREFETCH`, all `tensor_set_async` callers pass `STAGE_BACKEND` (current behavior). Under the hood store `cudaEvent_t` per entry when CUDA, else the existing ggml events.
  - `stage_acquire` synchronize path: for `STAGE_PREFETCH` entries use `cudaEventSynchronize`; ggml-event fallback unchanged for non-CUDA.

- [ ] **Step 1: Unit assertion (CPU-only portion)**

Non-CUDA builds keep aligned malloc; the observable unit-level fix is limited. Assert in `tests/test-expert-cache.cpp` that repeated acquire/commit/acquire cycles do not lose track of in-flight flags:

```cpp
static void test_staging_ring_reuse() {
    // hammer acquire+commit 100x across two fake tensors; must not crash
    // and must report waits when wrapping
    struct ggml_init_params params = { 4 * ggml_tensor_overhead(), NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 8);
    struct ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, 8, 8, 8);
    for (int i = 0; i < 100; i++) {
        void * p = ggml_backend_expert_cache_stage_acquire(cache, i & 1 ? b : a, i % 3, 512);
        assert(p != NULL);
        ggml_backend_expert_cache_stage_commit(cache, i & 1 ? b : a, i % 3);
    }
    ggml_free(ctx);
}
```

(Q4_K choice exercises stride padding; adjust type if the header lacks it in scope.)

- [ ] **Step 2: Run test, then implement**

Run test first (expected PASS already on CPU - it guards regressions), then implement the pinned alloc + stream-parameterized commit as specified above. Keep the diff tight: one new enum param threaded to the 4 `stage_commit` call sites (lines ~1407, 1512, 1541, 2063).

- [ ] **Step 3: GPU verification**

Rebuild with CUDA, rerun Task 4's equivalence matrix plus a 10-minute soak decode watching for `cudaErrorIllegalAddress` or corrupted logits (the classic symptom of unpinned async-copy buffers being paged out mid-DMA).

- [ ] **Step 4: Propose commit (DO NOT run git commit)**

```
ggml : pin expert cache staging memory, own prefetch-stream events

Staging ring now uses cudaHostAlloc pages and records its
guard event on the stream that issued the copy.
```

---

### Task 6: Hard global VRAM budget for slot pools

Today: first pool carves from the backing buffer; later pools silently allocate EXTRA backend buffers (`pool_bytes` each, uncounted in `cache->used`), and the failure fallback aliases `ptensor->data = cache->tensor->data` (lines 318-321) - overlapping the flat cache. A "64 MiB" setting can consume far more.

Fix: one allocator accounting ALL pool bytes (carved + extra buffers) against `capacity`. When budget exhausted, `get_or_create_pool` returns NULL (callers already handle NULL). Remove the aliasing fallback entirely.

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.cpp:262-350` (`get_or_create_pool`)
- Test: `tests/test-expert-cache.cpp`

**Interfaces:**
- Produces: new cache field `size_t pool_bytes_total;` incremented for carved AND separately-buffered pools; `get_or_create_pool` refuses creation when `pool_bytes_total + pool_bytes > capacity`. Existing NULL-return contract preserved.

- [ ] **Step 1: Failing test**

```cpp
static void test_global_pool_budget() {
    // capacity fits ONE pool of the given shape; a second distinct-shape
    // pool request must fail cleanly (NULL) rather than aliasing memory
    // (existing test harness sets capacity; craft two shapes accordingly)
    auto * p1 = ggml_backend_expert_cache_get_or_create_pool(cache, shape_a_tensor);
    assert(p1 != NULL);
    auto * p2 = ggml_backend_expert_cache_get_or_create_pool(cache, shape_b_huge_tensor);
    assert(p2 == NULL); // budget exhausted, no aliasing fallback
}
```

Size `shape_b_huge_tensor` so its pool exceeds remaining capacity given the harness cache's configured capacity.

- [ ] **Step 2: Verify failure**

Run: `cmake --build build --target test-expert-cache && .\build\bin\test-expert-cache.exe`
Expected: FAIL - today p2 comes back non-NULL (aliased or extra-buffer).

- [ ] **Step 3: Implement**

In `get_or_create_pool`:
1. Compute `pool_bytes` as today.
2. Early-out: `if (cache->pool_bytes_total + pool_bytes > cache->capacity) return NULL;`
3. Carved path: increment `pool_bytes_total += pool_bytes`.
4. Extra-buffer path: same increment; on `pbuffer == NULL` RETURN NULL (delete lines 318-321 aliasing fallback).
5. Pool destruction/free path (search `ggml_backend_buffer_free(pool->buffer)` sites): decrement `pool_bytes_total`.

- [ ] **Step 4: Tests pass + smoke**

Run: full test-expert-cache suite. Then GPU smoke: run with a deliberately tiny `--expert-cache-capacity 16MiB` (or equivalent env) and confirm process starts, falls back gracefully to generic path for oversized shapes, and total VRAM delta measured via `nvidia-smi` stays under roughly capacity + model overhead.

- [ ] **Step 5: Propose commit (DO NOT run git commit)**

```
ggml : enforce global expert cache pool budget

All slot-pool bytes count against capacity; exhausted budgets
return NULL instead of aliasing the flat cache buffer.
```

---

### Task 7 (P2, optional stretch): Transition table conditioned on source expert

Current table `[from][to][dst]` counts identically regardless of the previous layer's expert - it models marginal P(dst), not transition. Minimal correct upgrade without a dense 4D blow-up: sparse per-source maps.

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.cpp:198-206` (struct fields), `1670-1789` (init, record, predict)

**Interfaces:**
- Produces: `std::unordered_map<uint64_t, int32_t> transition_sparse;` with key = hash(from_layer, to_layer, src_expert, dst_expert) (pack into uint64: from_layer<<48 | to_layer<<40 | src<<20 | dst, bounded by predictor_max dims). `record_prediction` increments per (prev_expert, curr_expert) PAIR. `predict_experts(from,to)` iterates `current_experts[from]` sources and sums only dst counters for those sources.

- [ ] **Step 1: Failing test**

```cpp
static void test_transition_conditioned_on_source() {
    // layer0 uses expert 1 repeatedly with dst=5 -> strong (1,5) count
    // layer0 also saw expert 2 once with dst=9
    // predicting from source {2} alone must rank 9 above 5
    feed via record_prediction: step1 layer0 {1}, layer1 {5};
    repeat x10; then layer0 {2}, layer1 {9};
    reset current_experts to {2}; predict from 0 to 1;
    assert top prediction == 9
}
```

(Drive `record_prediction(layer, ids, n)` twice to simulate two layers; consult how `current_experts` advances - it stores per-layer lists, so call order layer0 then layer1 populates both.)

- [ ] **Step 2: Implement, run, propose**

Same TDD cycle. Commit proposal:

```
ggml : condition expert transition counts on source expert

Sparse (src,dst) counters replace the marginal destination
histogram; predictions score only observed sources.
```

---

### Task 8 (P3): Fine-grained router-ID sync (review item #12, stage A)

Full design and rationale: `docs/superpowers/plans/2026-08-23-expert-cache-async-ids-design.md`. This task implements stage A only: replace the per-matrix device-wide drains with a single event-scoped wait, deduped across gate/up/down. Stage B (GPU compaction kernel into pinned host memory) is gated on measuring this first - do NOT build it in this task.



Today (`ggml/src/ggml-backend.cpp`): `ggml_backend_synchronize(input_backend)` at :1861 runs once per eligible expert input (3 drains per MoE layer), and the ids read at :1879-1880 adds a second drain via `ggml_backend_tensor_get_async` + `ggml_backend_synchronize`. CUDA buffer ops run on `cudaStreamPerThread` + synchronize (`ggml-cuda.cu:782-796`), so each is a full device drain.



**Files:**

- Modify: `ggml/src/ggml-backend.cpp:1853-1893` (sync block in `compute_splits`)

- Modify: `ggml/src/ggml-backend-expert-cache.h` / `.cpp` (new event helper + counters)

- Test: `tests/test-expert-cache.cpp` (unit) + GPU bench gate



**Interfaces:**

- Consumes: backend events already used by the sched (`sched->events[split_backend_id][cur_copy]`, `ggml_backend_event_record/wait/synchronize`); `prev_ids_tensor` dedup.

- Produces:

  - New cache-side helpers (header):

    - `void ggml_backend_expert_cache_route_wait_begin(ggml_backend_expert_cache_t cache);`

    - `void ggml_backend_expert_cache_route_wait_end(ggml_backend_expert_cache_t cache, uint64_t wait_us);` - feeds a new stat `n_route_sync_us_total` so the bench gate has numbers.

  - The compute_splits sync block collapses to at most ONE host wait per split for all expert inputs sharing an unchanged ids tensor.



- [ ] **Step 1: Write the failing test**



Unit-level (CPU backend; asserts the dedup logic, not CUDA timing):



```cpp

static void test_route_sync_dedup() {

    // second call with same ids tensor pointer must be a no-op:

    // model by asserting the existing prev_ids_tensor behavior through

    // the refactored helper - extract the "should we re-read ids" decision

    // into a testable function if not already

    bool first = ggml_backend_expert_cache_route_should_read(NULL, tensor_a);

    assert(first);

}

```



If the read decision stays inline in compute_splits, instead assert observable stats: after Task 8, `record_probe_sync` accumulates strictly less time than before on the same synthetic workload (compare logged totals across two runs of `test-expert-cache-profile`). Accept either form; prefer extracting a pure helper.



- [ ] **Step 2: Implement**



In `compute_splits`:

1. Hoist one `ggml_backend_event_t route_event = NULL;` per split iteration.

2. Replace :1861 `ggml_backend_synchronize(input_backend)` with nothing - it is only needed to order the weight-source reads below against prior graph work, which `tensor_set_async` stream ordering already provides (verify with a comment citing the enqueue-before-consume invariant from Task 1).

3. For the ids read (:1877-1880): when `ids_tensor != prev_ids_tensor`, record the event on `ids_backend` immediately after issuing `ggml_backend_tensor_get_async` into a PINNED staging slice (reuse Task 5's ring or a dedicated small cudaHostAlloc buffer), then `ggml_backend_event_synchronize(route_event)` ONCE, right before the bitset build. On non-CUDA backends keep today's get+synchronize.

4. Gate/up/down sharing an ids tensor skip both read and wait entirely (already handled by `prev_ids_tensor` - extend it to also skip the event wait).



- [ ] **Step 3: Correctness tests**



Run: `cmake --build build --target test-expert-cache test-expert-cache-shared-ids && .\build\bin\test-expert-cache.exe && .\build\bin\test-expert-cache-shared-ids.exe`

Expected: PASS (bit-equality matrix still holds - the route set must be identical to raw-id parsing).



- [ ] **Step 4: Bench gate (GTX 1080, REQUIRED before any stage-B work)**



Run llama-bench PP and single-token TG with cache enabled, before/after this change; record `n_route_sync_us_total` and `record_probe_sync` averages.

Decision rule from the design doc: if TG does not improve AND probe_sync did not drop materially, the drain was not the bottleneck - stop here, stage B (compaction kernel) does not pay for itself. Log the numbers in the commit proposal body.



- [ ] **Step 5: Propose commit (DO NOT run git commit)**



```

ggml : scope expert route id waits to one event per split



Router id reads now wait on a single recorded event instead

of draining the device once per expert matrix; unchanged ids

tensors skip the wait entirely.

```



 ## Deferred (documented, NOT in this plan)

 


- Review item #12 stage B (GPU compaction kernel writing used-set bitset to pinned host memory): designed in docs/superpowers/plans/2026-08-23-expert-cache-async-ids-design.md, gated on Task 8 Step 4 measurements.

- Multi-token zero-copy: blocked behind Tasks 1-4 proving the single-token path correct.
- begin_step semantics ("tokens" vs "decode calls" naming, route-trace granularity): cosmetic/logging concern, batch with the PP re-baseline work.

## Self-Review Notes

- Review item #1 (missing `remapped_ids[i] = slot`): ALREADY FIXED in uncommitted working tree (verified at ggml-backend.cpp:2003, 2018, 2031). Not planned.
- Review item #5 (restore loop): ALREADY FIXED (ggml-backend.cpp:2347-2350); Task 4 extends it to src[2].
- Items #2 #3 #4 #6 #7 #8 #9 #10 #11 verified present in working tree; Tasks 1-7 map 1:1 (#2+#3+#9 -> Tasks 1-2; #4 -> Task 3; #6 -> Task 4; #7+#8 -> Task 5; #10 -> Task 6; #11 -> Task 7).
- Item #12: stage A folded in as Task 8 with a measurement gate; stage B stays deferred pending those numbers (design: docs/superpowers/plans/2026-08-23-expert-cache-async-ids-design.md).
