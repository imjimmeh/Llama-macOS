# Route-Ready CPU-Base GPU-Sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every route-ready cache miss execute the unchanged cache-off CPU MoE bundle, while a complete resident Gate/Up/Down bundle executes through a persistent GPU slot-pool sidecar without scheduler placement retry or expert-weight upload.

**Architecture:** The scheduler always allocates registered host-backed MoE bundles on their normal cache-off CPU placement. After the route-ID producer finishes, one read-only classifier selects exactly one whole-bundle executor: CPU graph nodes on any miss, or an independent persistent GPU sidecar on a complete hit. The sidecar copies only activations and remapped route IDs, writes the complete Down result back to the CPU-allocated Down node, and skips the contiguous original CPU bundle range. It never rewrites scheduler graph source pointers, reassigns backends, resets the scheduler, or reallocates the active graph.

**Tech Stack:** C++17, GGML scheduler, GGML graph views, GGML CPU and CUDA backends, expert-cache slot pools, `test-expert-cache`, `test-backend-ops`, `llama-bench`, `llama-server`, Python 3, Windows PowerShell.

## Global Constraints

- Do not commit, push, create a branch, worktree, or PR without explicit user approval.
- Keep every added code comment ASCII-only, concise, and limited to non-obvious invariants.
- Do not modify `ggml-alloc.c` or try to make gallocr placement changes transactionally reusable during graph execution.
- Do not keep any cache-miss `sched_reset()`, `alloc_graph()`, graph-input snapshot, deny-list, or backend reassignment path inside `ggml_backend_sched_graph_compute_async()`.
- Cache capacity must not promote a host-backed registered MoE bundle to GPU placement. Cache registration is not residency and is not a placement request.
- Route-ready GPU execution requires every requested expert to be `RESIDENT` for every required projection of the complete bundle. `LOADING`, absent, stale, unsupported, partial, or over-capacity states use CPU.
- A miss must submit no current-route host-expert RAM-to-GPU copy, must not wait for a fill, and must preserve the cache-off greedy output.
- No mixed CPU/GPU expert subsets, partial reductions, CPU correction adds, zero-gated rows, or per-node CPU interception.
- Do not add CUDA kernels, threads, a background service, a second scheduler, or a new model-specific graph form.
- Keep `GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL=0` for route-ready acceptance runs.
- Add an `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` entry after every decision gate. Update `EXPERT_CACHE.md` only after the relevant runtime behavior passes all gates.
- Run specific tests while debugging. Before completion, run the focused cache target, `test-backend-ops`, `llama-bench` build, TG1 JSONL smoke, deterministic greedy server replay, and the conditional five-pair matrix.

---

## File Structure

| File | Responsibility after this plan |
| --- | --- |
| `ggml/include/ggml-backend.h` | Public cumulative `n_route_ready_dispatches` and `n_route_ready_classifications` telemetry fields. |
| `ggml/src/ggml-backend.cpp` | CPU-base scheduler placement, immutable route dispatch records, producer-prefix execution, binary full-hit-or-CPU choice, sidecar handoff, and scheduler-owned telemetry. |
| `ggml/src/ggml-backend-moe-hetero.h` | Narrow API for a persistent full-hit route-ready sidecar. It must not expose mixed routing or CPU miss execution. |
| `ggml/src/ggml-backend-moe-hetero.cpp` | Persistent activation, ID, and result buffers plus GPU full-bundle sidecar execution. |
| `ggml/src/ggml-backend-expert-cache.h` and `.cpp` | Canonical cache access accounting and an epoch counter for the no-cache-mutation-during-compute canary. |
| `tests/test-expert-cache.cpp` | Synthetic CPU-reference tests for base misses, complete hits, graph reuse, ground-truth admission, and mutation invariants. |
| `tests/test-backend-ops.cpp` | Backend-copy smoke coverage for GPU-sidecar result handoff when CUDA is present. |
| `tools/llama-bench/llama-bench.cpp` | Atomic JSONL export of route-ready admission counters. |
| `tools/results/expert-cache/run-tg-matrix.py` | Compact-preset TG matrix extension only when the warm-sidecar ceiling passes. |
| `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` | Accepted/rejected gates, command lines, raw JSONL paths, deterministic hashes, and matrix summary. |

---

### Task 1: Remove the Failed Re-entry Prototype and Restore a Testable Baseline

**Files:**
- Modify: `ggml/src/ggml-backend.cpp:908-1052,1421-2060,2240-3595`
- Modify: `tests/test-expert-cache.cpp:160-639,1804-1806`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: the completed producer-prefix route dispatch behavior in `test_route_ready_producer_at_split_cursor()` and `ggml_graph_view()`.
- Produces: a scheduler with no active-compute allocation retry, no diagnostic prints, restored all-hit test registration, and a passing baseline focused test target.

- [ ] **Step 1: Preserve the current failure as an optimization-log rejection, not as retained code**

Append an entry containing all of the following facts:

```text
Rejected design: cache-off placement rebuild after route-ready miss.
Observed behavior: unseeded cross-split dispatch classified a miss, then a reset/reallocation CPU retry terminated before returning GGML status.
Rejected mechanism: changing scheduler backend assignments after gallocr allocation and reusing original graph tensor descriptors.
Reason: ggml_gallocr_reserve_n() can replace virtual backend buffers while tensors with non-null data are treated as externally allocated on the next graph allocation.
Decision: do not repair the retry. CPU placement is selected before allocation and full-hit GPU execution moves to a separate persistent sidecar.
```

Do not claim a byte-identical allocator diagnosis. State it as an evidence-backed architectural rejection.

- [ ] **Step 2: Delete re-entry-only scheduler state and helpers**

Remove these members and every helper/call site that exists only for the retry:

```cpp
std::vector<const ggml_tensor *> route_ready_deny_weights;
std::vector<ggml_backend_sched_copied_source> copied_sources;
std::vector<ggml_backend_sched_input_snapshot> route_ready_input_snapshots;
bool route_ready_retrying;
```

Delete the associated deny, bundle-deny, copied-source save/restore, and graph-input snapshot/restore helpers. Restore the ordinary scheduler source-copy lifecycle: source pointers are not retained in new retry bookkeeping and `ggml_backend_sched_synchronize()` does not reset the scheduler merely because a side vector is non-empty.

Delete all temporary `fprintf(stderr, "route ...")` instrumentation from scheduler code and tests. Restore the normal all-hit test call in `main()`:

```cpp
test_route_ready_ids_use_gpu_slots();
test_route_ready_producer_at_split_cursor();
```

Remove the incomplete `test_route_ready_normal_miss_preserves_normal_execution()` fixture completely; Task 2 replaces it with the CPU-base contract.

- [ ] **Step 3: Delete retry control flow without weakening producer-prefix execution**

Restore `ggml_backend_sched_graph_compute_async()` to one ordinary allocation/compute path:

```cpp
if (!sched->is_reset && !sched->is_alloc) {
    ggml_backend_sched_reset(sched);
}

if (!sched->is_alloc && !ggml_backend_sched_alloc_graph(sched, graph)) {
    return GGML_STATUS_ALLOC_FAILED;
}

return ggml_backend_sched_compute_splits(sched);
```

Keep the Task 1 producer gate:

```cpp
if (dispatch->producer_node_idx >= cur_j) {
    const ggml_cgraph producer_view = ggml_graph_view(
        &split->graph, cur_j, dispatch->producer_node_idx + 1);
    const ggml_status status = ggml_backend_graph_compute_async(
        split_backend, &producer_view);
    if (status != GGML_STATUS_SUCCESS) {
        return status;
    }
}
```

A route-ready miss must no longer set a retry flag or change placement. Until Task 2, it must leave graph nodes unmodified and continue ordinary execution.

- [ ] **Step 4: Restore all-hit telemetry assertions and run the focused target**

In both `test_route_ready_ids_use_gpu_slots()` and `test_route_ready_producer_at_split_cursor()`, assert the full all-hit accounting contract:

```cpp
require(stats.n_route_ready_dispatches == 1);
require(stats.n_route_ready_classifications == 1);
require(stats.n_route_ready_actions == 1);
require(stats.n_zero_copy_hits == 6);
require(stats.bytes_ram_to_gpu == 0);
```

Run:

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: the original graph census, stale-ID deferral, all-hit GPU route execution, producer-at-cursor execution, graph reuse, and every pre-existing cache test pass. No route debug line appears.

---

### Task 2: Make Cache-Off CPU Placement the Only Miss Placement

**Files:**
- Modify: `ggml/src/ggml-backend.cpp:1443-1650,1809-1964,2230-3210`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: `ggml_backend_sched_register_expert_bundle()`, host `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` tensors, and the original graph planner.
- Produces: `test_route_ready_bundle_miss_uses_cpu_base_graph()` and a placement rule that leaves registered host-backed bundle nodes on CPU unless Task 4 skips them for a complete GPU sidecar hit.

- [ ] **Step 1: Write the failing CPU-base miss regression**

Add `test_route_ready_bundle_miss_uses_cpu_base_graph()` beside the all-hit fixture. Reuse its CPU-reference Gate/Up/Down fixture exactly:

```cpp
// CPU WEIGHTS: Gate, Up, Down
// route_ids = ggml_dup(route_input), pinned to CPU
// registered bundle, no slot seeds
```

The test computes twice on the same graph, with no cache seed between calls. It must assert after each compute:

```cpp
require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
require(output_matches_cpu_reference);
require(stats.n_route_ready_dispatches == 1);
require(stats.n_route_ready_classifications == 1);
require(stats.n_route_ready_actions == 0);
require(stats.n_zero_copy_hits == 0);
require(stats.bytes_ram_to_gpu == 0);
```

For the second compute, preserve output equality and assert that route-ready action and zero-copy counters remain zero. This proves graph reuse and proves a miss is not made sticky or retried.

- [ ] **Step 2: Run the test before changing placement**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: the new miss test fails because cache registration still makes the scheduler choose a GPU consumer placement that is unsafe without complete slot residency.

- [ ] **Step 3: Add the narrow CPU-base placement gate**

Add one scheduler-local predicate next to `ggml_backend_sched_can_offload_host_weight()`:

```cpp
static bool ggml_backend_sched_is_registered_host_expert_weight(
        ggml_backend_sched_t sched,
        int backend_id,
        const ggml_tensor * tensor) {
    const ggml_backend_expert_cache_t cache = sched->expert_caches[backend_id];
    return cache != nullptr &&
        tensor != nullptr &&
        tensor->buffer != nullptr &&
        ggml_backend_buffer_is_host(tensor->buffer) &&
        ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_expert_cache_has_tensor(cache, tensor);
}
```

Where passes 2 and 3 would move a `MUL_MAT_ID` node to a non-CPU backend, reject only a node whose `src[0]` satisfies that predicate. Do not reject generic host tensors, dense layers, unregistered weights, prompt-processing work, or explicitly preallocated GPU weights.

The result is intentional: cache registration preserves host placement. It is not evidence of a resident slot and must never by itself cause a GPU split.

- [ ] **Step 4: Make route dispatch classification read-only on every CPU-base path**

Keep `classify_route_dispatch()` responsible for exactly one route-ID read and all projection slot checks. On an incomplete bundle:

```cpp
dispatch.complete_bundle_hit = false;
dispatch.classified = true;
return;
```

Do not call a deny helper, rewrite `src[0]` or `src[2]`, alter `node->op`, reset the scheduler, allocate a graph, invoke a CPU interception, or skip the CPU bundle. The normal CPU split computes it.

- [ ] **Step 5: Update the seeded fixture to prove CPU-base placement**

Before the sidecar exists, a seeded registered bundle must still execute its original CPU graph. Change the immediate post-compute assertions in `test_route_ready_ids_use_gpu_slots()` to:

```cpp
require(stats.n_route_ready_dispatches == 1);
require(stats.n_route_ready_classifications == 1);
require(stats.n_route_ready_actions == 0);
require(stats.n_zero_copy_hits == 0);
require(stats.bytes_ram_to_gpu == 0);
```

Retain CPU-reference output and same-graph reuse assertions. This is the required stable state between Task 2 and Task 4: slot residency is observed, but it does not alter scheduler placement or execute GPU work.

- [ ] **Step 6: Verify CPU-base behavior**

Run the Task 2 command again.

Expected: seeded and unseeded registered bundles complete through the unchanged CPU graph with CPU-reference output. Both classify one dispatch; neither produces a route-ready action, a zero-copy hit, an expert-weight upload, a retry, or a failure. Append command output and the result to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`.

---

### Task 3: Define and Test a Persistent Full-Hit GPU Sidecar

**Files:**
- Modify: `ggml/src/ggml-backend-moe-hetero.h`
- Modify: `ggml/src/ggml-backend-moe-hetero.cpp`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `tests/test-backend-ops.cpp`

**Interfaces:**
- Consumes: `ggml_moe_bundle_plan`, complete slot ownership from `ggml_backend_expert_cache_reserve_bundle_slots()`, `ggml_backend_expert_cache_get_slot_tensor()`, and CPU-allocated original bundle outputs.
- Produces:

```cpp
struct ggml_moe_route_ready_sidecar;
typedef struct ggml_moe_route_ready_sidecar * ggml_moe_route_ready_sidecar_t;

ggml_moe_route_ready_sidecar_t ggml_moe_route_ready_sidecar_new(
    ggml_backend_t gpu_backend,
    ggml_backend_t cpu_backend,
    int64_t d_model,
    int64_t d_ff,
    int32_t top_k,
    bool is_fused);

void ggml_moe_route_ready_sidecar_free(
    ggml_moe_route_ready_sidecar_t sidecar);

enum ggml_status ggml_moe_route_ready_sidecar_execute_full_hit(
    ggml_moe_route_ready_sidecar_t sidecar,
    const struct ggml_moe_bundle_plan * bundle,
    ggml_backend_expert_cache_t cache,
    const int32_t * route_ids,
    int32_t n_route_ids,
    struct ggml_backend_expert_cache_stats * stats);
```

The executor is only valid for a complete hit. It must return `GGML_STATUS_FAILED` before touching output if any required slot or sidecar shape is unavailable.

- [ ] **Step 1: Write a direct full-hit sidecar regression**

Add `test_route_ready_sidecar_full_hit_matches_cpu()` using the existing two-expert CPU reference. It creates a sidecar once, seeds both experts for Gate/Up/Down, and invokes `ggml_moe_route_ready_sidecar_execute_full_hit()` with `{0, 1}`.

Assert:

```cpp
require(status == GGML_STATUS_SUCCESS);
for (size_t i = 0; i < expected.size(); ++i) {
    require(fabsf(actual[i] - expected[i]) < 1e-5f);
}
require(stats.n_zero_copy_hits == 6);
require(stats.bytes_ram_to_gpu == 0);
```

Add the negative sibling in the same fixture: leave one Down slot unseeded, initialize a sentinel output, invoke the API, and assert:

```cpp
require(status == GGML_STATUS_FAILED);
for (const float value : actual) {
    require(value == sentinel);
}
require(stats.n_zero_copy_hits == 0);
```

- [ ] **Step 2: Run the new regression before implementation**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache test-backend-ops
build/bin/Release/test-expert-cache.exe
```

Expected: compilation fails because the sidecar API does not exist.

- [ ] **Step 3: Add a full-hit-only sidecar beside the existing mixed serial executor**

Do not route new work through `ggml_backend_moe_hetero_execute_serial()` or `ggml_backend_moe_hetero_execute_concurrent()`. Those functions include CPU-miss and mixed-route behavior and are not the route-ready contract.

Create the opaque sidecar in `ggml-backend-moe-hetero.cpp`. Its constructor allocates all persistent resources once:

```text
GPU: input activation, Gate IDs, Up IDs when separate, Down IDs, Gate output, Up output, activation, Down output, graph context, backend buffer.
CPU: one host result staging tensor or supported direct GPU-to-CPU tensor-copy target.
```

Build one internal GPU graph matching the constructor's `is_fused` value. The graph must reference sidecar-owned activation and ID tensors. Before each execution, bind only the sidecar graph's weight source descriptors to the current bundle's slot tensors; those descriptors are not scheduler graph nodes and are restored before return.

`execute_full_hit()` must perform this exact sequence:

```text
1. Validate route count equals top_k and all bundle fields are structurally valid.
2. Resolve and reserve every Gate/Up/Down slot before any GPU tensor write.
3. Translate original expert IDs to each projection's slot IDs.
4. Copy the CPU base bundle's input activation into persistent GPU input storage.
5. Upload only the three remapped ID arrays.
6. Execute the sidecar GPU graph.
7. Copy the complete sidecar Down output to bundle->down_node, which remains CPU allocated.
8. Record one zero-copy hit per unique (projection, expert), record sidecar GPU execution, and release reserved slots after GPU completion.
```

The function must never copy original expert weights, allocate a context/buffer per call, calculate a CPU contribution, merge partial results, mutate a scheduler node, or issue a cache fill.

- [ ] **Step 4: Add backend-copy smoke coverage**

In `tests/test-backend-ops.cpp`, add a CUDA-gated test that copies a known GPU F32 sidecar-output tensor to a CPU tensor and verifies element equality. Skip only when no GPU backend is available.

This test isolates the final sidecar handoff from MoE graph behavior.

- [ ] **Step 5: Verify the isolated executor**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache test-backend-ops
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-backend-ops.exe
```

Expected: sidecar all-hit output matches the CPU reference; incomplete slots fail before touching output; direct GPU-to-CPU handoff is correct; no existing hetero test regresses.

---

### Task 4: Select the Sidecar Only for a Complete Contiguous CPU Bundle

**Files:**
- Modify: `ggml/src/ggml-backend.cpp:884-906,1882-1964,2288-3220,3410-3435`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: CPU-base placement from Task 2 and full-hit sidecar API from Task 3.
- Produces: one scheduler-owned `ggml_moe_route_ready_sidecar_t` per compatible bundle shape and a binary dispatch action with exactly two outcomes: skip the original CPU bundle after sidecar success, or execute it unchanged.

- [ ] **Step 1: Extend the failing full-hit scheduler test for CPU-base placement**

Update `test_route_ready_ids_use_gpu_slots()` so route IDs are CPU-pinned, matching normal cache-off placement. Seed all three projections, then assert:

```cpp
require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
require(output_matches_cpu_reference);
require(stats.n_route_ready_dispatches == 1);
require(stats.n_route_ready_classifications == 1);
require(stats.n_route_ready_actions == 1);
require(stats.n_zero_copy_hits == 6);
require(stats.bytes_ram_to_gpu == 0);
```

Add a graph-reuse second compute with the same assertions except cumulative counters. The first run after Task 2 should fail because the CPU base graph does not yet invoke a GPU sidecar.

- [ ] **Step 2: Store a structurally complete executable range in the dispatch record**

Extend `ggml_backend_sched_route_bundle_dispatch` with:

```cpp
int first_bundle_node_idx = -1;
int last_bundle_node_idx = -1;
ggml_moe_route_ready_sidecar_t sidecar = nullptr;
```

When constructing a dispatch, accept it only if the route producer and every required Gate/Up/Down projection are found and the range `[first_bundle_node_idx, last_bundle_node_idx]` contains only the recognized Gate/Up/Down nodes, the directly derived activation node, or shape-preserving views. Any unrelated operation makes the dispatch unsupported and ordinary CPU execution continues.

Build or reuse the sidecar only after this structural validation. Create it during scheduler graph allocation, never after route IDs have been read. Free it in `ggml_backend_sched_free()`.

- [ ] **Step 3: Keep miss classification read-only and make full-hit execution binary**

After the existing producer-inclusive graph view completes and `classify_route_dispatch()` establishes a complete hit, invoke the sidecar. On success:

```cpp
const enum ggml_status status = ggml_moe_route_ready_sidecar_execute_full_hit(
    dispatch.sidecar,
    &to_hetero_plan(*dispatch.bundle),
    cache,
    dispatch.ids.data(),
    (int32_t) dispatch.ids.size(),
    &sched->route_census_stats);
if (status != GGML_STATUS_SUCCESS) {
    return status;
}

sched->route_census_stats.n_route_ready_actions++;
cur_j = dispatch.last_bundle_node_idx + 1;
```

The graph-view cursor skips the entire original CPU bundle range only after sidecar success. It must not use `node->op = GGML_OP_NONE`, node source replacement, a remapped-ID scheduler buffer, a scheduler reset, or a backend reassignment.

On `complete_bundle_hit == false`, advance only through the already-executed producer and allow the ordinary CPU graph view to execute every bundle node. If the sidecar returns failure despite a classifier hit, fail the compute; do not fall back after partially writing the Down output.

- [ ] **Step 4: Add partial, stale, and unsupported range regressions**

Add three fixtures:

```cpp
static void test_route_ready_partial_bundle_uses_cpu_base_graph();
static void test_route_ready_stale_ids_use_cpu_base_graph();
static void test_route_ready_non_contiguous_bundle_uses_cpu_base_graph();
```

Each must compare to the CPU reference and assert `n_route_ready_actions == 0`, `n_zero_copy_hits == 0`, and `bytes_ram_to_gpu == 0`. The partial fixture leaves one projection slot absent. The stale fixture reuses an old route tensor after an intervening graph generation. The non-contiguous fixture inserts an unrelated node between the activation and Down node.

- [ ] **Step 5: Verify binary execution and record the gate**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache test-backend-ops
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-backend-ops.exe
```

Expected: only a complete resident contiguous bundle invokes the sidecar; all other cases run the original CPU path; every output matches its CPU reference; same-graph reuse succeeds. Append this full binary-dispatch result to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`.

---

### Task 5: Teach Boundary-Time Admission From Actual Route IDs

**Files:**
- Modify: `ggml/src/ggml-backend.cpp:classify_route_dispatch()`
- Modify: `ggml/src/ggml-backend-expert-cache.h`
- Modify: `ggml/src/ggml-backend-expert-cache.cpp`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: post-producer route IDs, `ggml_backend_expert_cache_record_access_count()`, periodic `ggml_backend_expert_cache_begin_step()`, and canonical registered bundle weights.
- Produces: one-step-lag complete-bundle residency based on actual Gate/Up/Down access frequency. Current-token miss execution remains CPU-base.

- [ ] **Step 1: Write the failing miss-then-hit regression**

Add `test_route_ready_dispatch_learns_bundle_residency()`. Use the CPU-base fixture with no seeds and configure cache period to one. Compute twice without direct calls to seed, claim-slot, rebalance, or prefetch.

Assert after the first compute:

```cpp
require(first_output_matches_cpu_reference);
require(stats.n_route_ready_classifications == 1);
require(stats.n_route_ready_actions == 0);
require(stats.n_zero_copy_hits == 0);
require(stats.bytes_ram_to_gpu == 0);
```

Assert after the second compute:

```cpp
require(second_output_matches_cpu_reference);
require(stats.n_route_ready_classifications == 2);
require(stats.n_route_ready_actions == 1);
require(stats.n_zero_copy_hits == 6);
require(stats.bytes_ram_to_gpu == 0);
```

- [ ] **Step 2: Run the regression before recording route access**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: the second compute remains a CPU miss because route-ready classification has not recorded actual expert usage for the rebalancer.

- [ ] **Step 3: Normalize weights and record access after the safe route read**

Add one local helper that resolves a dispatch consumer's source to its canonical registered host weight. It must return null for a scheduler copy, view, unknown tensor, or tensor belonging to another bundle; it must never use name parsing.

After `classify_route_dispatch()` has read IDs and deduplicated requested experts, but before marking classification complete, run this loop for every required projection:

```cpp
for (const ggml_tensor * weights : canonical_bundle_weights) {
    if (weights == nullptr ||
        !ggml_backend_expert_cache_can_store(cache, weights->nb[2])) {
        continue;
    }

    for (const int32_t expert_id : requested_experts) {
        ggml_backend_expert_cache_record_access_count(
            cache,
            weights,
            expert_id,
            1,
            GGML_EXPERT_CACHE_PHASE_TG);
    }
}
```

This code records data only. It must not allocate a slot, invoke a rebalance, call `process_jit_swaps()`, block on a cache event, or change current-token dispatch.

- [ ] **Step 4: Verify one-step re-admission**

Run the Task 5 command again.

Expected: the first compute remains an untouched CPU miss; the next `begin_step()` rebalances from canonical actual access and the second compute is a full sidecar hit. Record the result in the optimization log.

---

### Task 6: Fence Slot Mutation to Compute Boundaries

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.h`
- Modify: `ggml/src/ggml-backend-expert-cache.cpp`
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: every slot-map mutation site: seed, allocation, promotion, eviction, rebalance, JIT swap, and cache clear.
- Produces:

```cpp
uint64_t ggml_backend_expert_cache_get_residency_epoch(
    ggml_backend_expert_cache_t cache);

uint64_t ggml_backend_sched_expert_cache_epoch(
    ggml_backend_sched_t sched,
    int backend_index);
```

The epoch changes after a committed slot-map or slot-generation mutation. The scheduler records it only after `begin_step()` and validates it after all split execution completes.

- [ ] **Step 1: Write the failing no-mid-compute-mutation test**

Add `test_route_ready_compute_keeps_residency_epoch_stable()`. Configure one full-hit CPU-base sidecar fixture, capture the cache epoch immediately before `ggml_backend_sched_graph_compute()`, then assert:

```cpp
require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
require(ggml_backend_sched_expert_cache_epoch(sched, 0) == epoch_before);
```

Add the valid companion: a seed or period-one rebalance between computes advances the epoch before the next compute begins.

- [ ] **Step 2: Run the test before epoch accounting exists**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: compilation fails because the epoch query and scheduler test helper do not exist.

- [ ] **Step 3: Increment epoch only at committed cache mutation points**

Add `uint64_t residency_epoch` to the expert-cache implementation. Increment it after, not before, each successful mutation that changes an observable slot binding or its contents. This includes successful seed completion, slot promotion, eviction/reuse, rebalance commit, and JIT swap commit. Do not increment it for read-only lookup, classification, access recording, failed reservation, or a no-op rebalance.

At the beginning of `ggml_backend_sched_compute_splits()`, after `ggml_backend_expert_cache_begin_step()` and before route classification, snapshot each non-null cache epoch. After the final split has completed and before returning success, assert equality with its snapshot. Keep this assertion enabled only when `GGML_EXPERT_CACHE_DEBUG_EPOCH` is nonzero; production behavior stays unchanged.

- [ ] **Step 4: Move route-ready cache mutation out of execution**

If the epoch test reports a mutation during a complete hit or a CPU miss, move that exact operation to the next `begin_step()` boundary or reject it for route-ready dispatch. Do not add a wait, retry, or sidecar-local cache mutation.

- [ ] **Step 5: Verify the fence and update documentation**

Run the Task 6 focused test command with:

```powershell
$env:GGML_EXPERT_CACHE_DEBUG_EPOCH = "1"
build/bin/Release/test-expert-cache.exe
```

Expected: full hit and miss are epoch-stable during compute; a between-compute seed/rebalance advances the epoch before the next run. Append the result to the optimization log. Do not update `EXPERT_CACHE.md` yet.

---

### Task 7: Export Telemetry, Prove Real-Model Eligibility, and Gate Measurement

**Files:**
- Modify: `ggml/include/ggml-backend.h`
- Modify: `tools/llama-bench/llama-bench.cpp`
- Modify: `tools/results/expert-cache/run-tg-matrix.py`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`
- Conditionally modify: `EXPERT_CACHE.md`

**Interfaces:**
- Consumes: cumulative scheduler counters and the finalized binary-dispatch behavior.
- Produces JSONL fields:

```text
expert_cache_route_ready_dispatches
expert_cache_route_ready_classifications
```

- [ ] **Step 1: Add compile-time test coverage for exported counters**

Keep the scheduler tests from Tasks 1-5 as the source-of-truth counter checks. Add a llama-bench regression or focused parser assertion that a JSON object containing both values parses them as integers and that subtraction preserves their delta:

```cpp
result.expert_cache_route_ready_dispatches = after.n_route_ready_dispatches - before.n_route_ready_dispatches;
result.expert_cache_route_ready_classifications = after.n_route_ready_classifications - before.n_route_ready_classifications;
```

- [ ] **Step 2: Export both fields atomically**

Append both counters to all four llama-bench lists in the same edit:

```cpp
get_fields();
get_values();
integer_field_names;
subtract_expert_cache_stats();
```

Use only these output names:

```text
expert_cache_route_ready_dispatches
expert_cache_route_ready_classifications
```

Do not add per-node logs, a dashboard field, or a trace parser.

- [ ] **Step 3: Build and perform a Compact TG1 smoke row**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache test-backend-ops llama-bench llama-server
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-backend-ops.exe
$env:GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL = "0"
build/bin/Release/llama-bench.exe -m "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" -p 0 -n 1 -r 1 --no-warmup -t 14 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -fa on -lm mlock -fitt 256 -exc 128 -excp 256 -o jsonl
```

Record the JSONL path and these fields:

```text
expert_cache_route_census_plans
expert_cache_route_ready_dispatches
expert_cache_route_ready_classifications
expert_cache_route_ready_actions
expert_cache_route_census_cpu_host_nodes
expert_cache_route_census_non_cpu_host_nodes
expert_cache_eligible_ops
expert_cache_cpu_backend_bypasses
```

If the row has no useful route plans, dispatches, or classifications, stop performance work and record the placement reason. Do not compensate by changing global offload policy.

- [ ] **Step 4: Run deterministic greedy cache-off versus sidecar replay**

Use the Compact preset's fixed single-sequence, greedy 256-token server request. Run a fresh `-exc 0` process and a fresh enabled-cache process with identical model, prompt, seed, context, GPU layers, and all non-cache options. Retain both token hashes and cache telemetry JSON.

Expected acceptance condition:

```text
token hash equal
route-ready miss or full-hit execution selected per current route
expert RAM-to-GPU weight bytes during timed generation equal zero
```

If hashes differ, stop and fix the first divergent sidecar/miss fixture before running performance measurements.

- [ ] **Step 5: Establish the warm-sidecar ceiling before policy work**

Only if TG1 classification is nonzero and deterministic replay passes, extend `tools/results/expert-cache/run-tg-matrix.py` with a `--route-ready-sidecar` mode. It must preserve the Compact preset, alternate five fresh-process cache-off/cache-on TG pairs, disable `GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL`, and retain one raw JSONL file per run.

Run the matrix with the same settings as `G:\qwen3.6-35b-a3b-presets-exc-latest.ini`.

Retain ground-truth admission and consider generation-stamped ghost slots only if all are true:

```text
at least one route-ready full-hit action occurs during timed TG
expert RAM-to-GPU weight bytes remain zero
cache-on greedy hash matches cache-off
median TG exceeds cache-off beyond measured variation
```

Otherwise append the raw result, reject further sidecar policy work, and keep the CPU-base miss behavior. Update `EXPERT_CACHE.md` only when all conditions pass.

---

## Plan Coverage Review

- **Unsafe re-entry removal:** Task 1 removes every retry-owned state transition and diagnostic artifact.
- **Cache-off-equivalent miss:** Task 2 proves the CPU base graph completes ordinary misses without upload, retry, or sticky state.
- **Complete full-hit GPU execution:** Tasks 3 and 4 isolate persistent sidecar ownership, validate every projection before mutation, and skip CPU only after a complete successful result.
- **No mixed reductions:** Sidecar and scheduler tests use binary whole-bundle dispatch only.
- **Next-compute learning:** Task 5 records actual route accesses after safe producer synchronization and proves period-one admission.
- **Residency safety:** Task 6 makes mid-compute slot mutation observable and rejects it instead of adding another retry.
- **Runtime evidence:** Task 7 builds the public telemetry path, classifies real-model eligibility, proves deterministic output, and gates matrix work on measured warm full-hit value.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-28-route-ready-cpu-base-sidecar.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task and review each task before starting the next.
2. **Inline Execution** - execute tasks in this session with explicit review checkpoints.

Choose one before implementation. This plan requires cleanup of the current incomplete re-entry edits first; do not start sidecar work on top of them.
