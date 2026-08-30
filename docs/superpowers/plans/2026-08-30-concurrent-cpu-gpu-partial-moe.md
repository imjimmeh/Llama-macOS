# Concurrent CPU-GPU TG1 Partial-MoE Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> Project constraint: execute inline. Do not create a worktree or dispatch subagents. Before every source commit, obtain explicit user approval.

**Goal:** Replace the staged concurrent stub with a persistent TG1 partial-hit executor that overlaps GPU-resident expert routes with CPU miss routes, measures every 1/8 through 7/8 mask, and admits only masks proven faster than CPU-base execution.

**Architecture:** Keep the current three-way route-ready policy: CPU-base for rejected partial bundles, the existing serial path as the 7/8 A/B reference, and the unchanged 8/8 sidecar. Add a rebindable `ggml_moe_partial_executor` shared by compatible route-ready dispatches. It preallocates fixed K=1..7 GPU graphs, M=1..7 CPU graphs, event objects, CUDA-host buffers, and route-output buffers before decode; each execution consumes exactly one immutable eight-route residency snapshot.

**Tech Stack:** C++17, GGML scheduler, GGML CPU and CUDA backends, backend events and host buffer types, CUDA backend proc-address extension for elapsed event timing, `test-expert-cache`, `test-moe-partial-hit-bench`, `llama-bench`, Python 3, Nsight Systems, Windows PowerShell.

## Global Constraints

- Scope is TG1 only: `route_ids->ne[0] == 8` and `route_ids->ne[1] == 1`. Reject PP and any other route count without changing their existing path.
- Do not change cache capacity, manifest loading, fit target, placement policy, promotion, rebalance, or full-hit sidecar behavior.
- Preserve the current default policy: 0-6/8 CPU-base, 7/8 serial heterogeneous, 8/8 full-hit sidecar.
- `GGML_EXPERT_CACHE_HETERO_CONCURRENT=1` is development-only, defaults off, and initially replaces only the 7/8 serial branch.
- The executor hot path must not initialize/free a GGML context, allocate/free a backend buffer, construct a heap vector, or upload expert weights.
- Use `ggml_backend_dev_host_buffer_type()` plus `ggml_backend_alloc_ctx_tensors_from_buft()` for exchange buffers. Require `ggml_backend_buffer_get_type(exchange_buffer) == requested_host_buft`; otherwise return `NOT_ADMITTED`.
- `ggml-base` compiles `ggml-backend-moe-hetero.cpp` without `GGML_USE_CUDA`. Do not add CUDA headers or direct CUDA runtime calls to that file. Use existing generic backend APIs and a CUDA proc-address capability where true device elapsed time is required.
- Keep the original bundle output unweighted. Existing graph nodes retain router weighting and reduction.
- A pre-launch validation failure returns `NOT_ADMITTED` and executes the existing CPU-base path in the same graph invocation. A failure after queueing GPU work drains only its recorded event, releases slots, poisons the executor for later calls, and returns `GGML_STATUS_FAILED`; it must not run a half-executed fallback.
- Every `llama-bench` telemetry field is appended in all four places: field names, integer type classification, values, and `subtract_expert_cache_stats()`.
- Before changing production admission below 7/8, reproduce and resolve the documented CTest-only `test_route_ready_cross_split_sidecar` failure. Record the cause in `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`.

## File Structure

| File | Responsibility |
| --- | --- |
| `ggml/src/ggml-backend-moe-hetero.h` | Fixed-size snapshot, executor result, activation-source contract, and opaque executor API. Remove the unused concurrent-stub API. |
| `ggml/src/ggml-backend-moe-hetero.cpp` | Persistent executor construction/destruction, descriptor rebinding, K/M graphs, event-driven execution, row scatter, failure cleanup, and executor telemetry. |
| `ggml/include/ggml-backend.h` | New partial-executor statistic fields and the optional backend event-elapsed proc typedef. |
| `ggml/src/ggml-backend.cpp` | Executor catalog ownership, single route partition snapshot, 7/8 feature-gated dispatch, CPU-base fallback reuse, stat printing, and stat aggregation. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Register the event elapsed callback backed by `cudaEventElapsedTime`; no new synchronization behavior. |
| `tests/test-expert-cache.cpp` | Real executor correctness, permutation, event ordering, failure, telemetry, and scheduler feature-gate regressions. |
| `tests/test-moe-partial-hit-bench.cpp` | Persistent CPU-base/serial/concurrent mask harness with raw per-iteration CSV output. |
| `tools/llama-bench/llama-bench.cpp` | Export partial executor counters, bytes, and timings atomically with existing cache telemetry. |
| `tools/results/expert-cache/run-tg-matrix.py` | Explicit serial/concurrent child-process environment selection. |
| `tools/results/expert-cache/test_run_tg_matrix.py` | Unit coverage for the runner environment contract. |
| `EXPERT_CACHE.md` | Production admission rule, executor boundary, validation evidence, and known limitations. |
| `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` | Cross-split diagnosis, raw benchmark paths, mask table, deterministic generation hashes, and A/B/C matrix results. |

---

### Task 1: Record and Stabilize the Cross-Split Correctness Gate

**Files:**
- Modify: `tests/test-expert-cache.cpp:2010-2134`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: existing `test_route_ready_cross_split_sidecar()` and the two-backend route-ready scheduler fixture.
- Produces: a deterministic cross-split regression with an explicitly recorded cause category: test construction, slot lifetime, stream/event order, tensor layout, or arithmetic mismatch.

- [ ] **Step 1: Reproduce the documented CTest-only failure before touching partial admission**

Run the focused executable and CTest invocation repeatedly. Preserve every failing stdout/stderr transcript under `tools/results/expert-cache/` before modifying the fixture.

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
ctest --test-dir build -C Release -R "^test-expert-cache$" --output-on-failure --repeat until-fail:50
```

Expected before a fix: either a reproducible numeric/order failure or a documented non-reproduction after 50 CTest repetitions. Do not describe the failure as intermittent without retaining the command and outcome.

- [ ] **Step 2: Make the test expose the actual failing state rather than masking it**

Replace the loose second-compute admission assertion with exact assertions for the fixture's deliberately seeded state. Keep the existing async-seeding histogram assertion only if the test intentionally leaves seeding asynchronous; otherwise call `ggml_backend_sched_expert_cache_sync(sched)` before the second compute and assert the exact mask/action counters. Compare every element of `actual` with `expected` on every repeated execution.

```cpp
for (int iteration = 0; iteration < 32; ++iteration) {
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    ggml_backend_tensor_set(route_input, ids, 0, sizeof(ids));
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-5f);
    }
}
```

- [ ] **Step 3: Apply the smallest category-specific correction**

Use this decision rule:

```text
Test construction: make seed completion and input ownership deterministic; retain independent cross-split producer coverage.
Slot lifetime: reserve before queueing and release only after the recorded GPU-use event completes.
Stream/event order: record the producer/consumer event at the producer boundary and replace the broad sync with that event.
Tensor layout: preserve the source tensor's nb[] and construct only same-layout row views.
Arithmetic mismatch: keep CPU-base placement, correct the sidecar/executor math, and retain an elementwise parity regression.
```

Do not weaken tolerances or change production admission to make the test pass.

- [ ] **Step 4: Verify and log the gate**

Run the executable once and CTest 50 times again. Append the exact category, reproduction command, result, and retained log paths to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`.

```powershell
build/bin/Release/test-expert-cache.exe
ctest --test-dir build -C Release -R "^test-expert-cache$" --output-on-failure --repeat until-fail:50
```

Expected: every repeated run passes with the same numerical result.

- [ ] **Step 5: Request approval before committing Task 1**

Do not create a commit until the user explicitly approves the focused cross-split fix and log entry.

### Task 2: Define the Fixed TG1 Partial Executor Contract

**Files:**
- Modify: `ggml/src/ggml-backend-moe-hetero.h:14-90`
- Modify: `ggml/include/ggml-backend.h:365-451`
- Modify: `tests/test-expert-cache.cpp`

**Interfaces:**
- Consumes: `ggml_cache_route_bundle`, `ggml_moe_bundle_plan`, and current scheduler CPU-base placement.
- Produces: one snapshot structure passed exactly once to the executor and an executor result that lets the scheduler distinguish safe rejection from queued-work failure.

- [ ] **Step 1: Write the failing API and snapshot tests**

Add a small eight-route fixture to `test-expert-cache.cpp`. It must construct a mixed snapshot with non-contiguous hits, validate that duplicate/missing route positions are rejected before queueing, and call the new API. The initial build must fail because the types and functions do not exist.

```cpp
struct ggml_moe_partial_route_snapshot snapshot = {};
snapshot.n_hits = 7;
snapshot.n_misses = 1;
// Fill routes 0, 2, 3, 4, 5, 6, 7 as hits and route 1 as the miss.
require(ggml_moe_partial_route_snapshot_is_valid(&snapshot, 8));
snapshot.misses[0].route = 7;
require(!ggml_moe_partial_route_snapshot_is_valid(&snapshot, 8));
```

- [ ] **Step 2: Run the focused target and verify the red state**

```powershell
cmake --build build --config Release --target test-expert-cache
```

Expected: compile failure naming the missing partial snapshot/executor symbols.

- [ ] **Step 3: Add the narrow C-compatible header contract**

Replace `ggml_backend_moe_hetero_execute_concurrent()` rather than retaining a forwarding stub. Keep `ggml_backend_moe_hetero_execute_serial()` unchanged as the temporary reference path.

```cpp
enum { GGML_MOE_PARTIAL_MAX_ROUTES = 8 };

enum ggml_moe_partial_executor_result {
    GGML_MOE_PARTIAL_EXECUTOR_SUCCESS,
    GGML_MOE_PARTIAL_EXECUTOR_NOT_ADMITTED,
    GGML_MOE_PARTIAL_EXECUTOR_LAUNCH_FAILED,
};

struct ggml_moe_partial_route_snapshot {
    int32_t n_hits;
    int32_t n_misses;
    struct ggml_cache_route_bundle hits[GGML_MOE_PARTIAL_MAX_ROUTES];
    struct ggml_cache_route_bundle misses[GGML_MOE_PARTIAL_MAX_ROUTES];
};

struct ggml_moe_partial_activation {
    const void * host_data;
    size_t nbytes;
};

struct ggml_moe_partial_executor;
typedef struct ggml_moe_partial_executor * ggml_moe_partial_executor_t;

GGML_API bool ggml_moe_partial_route_snapshot_is_valid(
    const struct ggml_moe_partial_route_snapshot * snapshot,
    int32_t top_k);
GGML_API ggml_moe_partial_executor_t ggml_moe_partial_executor_new(
    ggml_backend_t gpu_backend,
    ggml_backend_t cpu_backend,
    const struct ggml_expert_bundle_weights * template_weights,
    int64_t d_model,
    int64_t d_ff,
    int32_t top_k,
    bool is_fused);
GGML_API void ggml_moe_partial_executor_free(
    ggml_moe_partial_executor_t executor);
GGML_API enum ggml_moe_partial_executor_result ggml_moe_partial_executor_execute(
    ggml_moe_partial_executor_t executor,
    const struct ggml_moe_bundle_plan * bundle,
    ggml_backend_expert_cache_t cache,
    const struct ggml_moe_partial_route_snapshot * snapshot,
    const struct ggml_moe_partial_activation * activation,
    struct ggml_backend_expert_cache_stats * stats);
```

Add `GGML_MOE_PARTIAL_EXECUTOR_*` stats fields to `ggml_backend_expert_cache_stats`; retain existing aggregate hetero fields for compatibility. The new fields are detailed in Task 7.

- [ ] **Step 4: Implement only validation and result classification**

`ggml_moe_partial_route_snapshot_is_valid()` must require `n_hits + n_misses == top_k`, `0 < n_hits < top_k`, each route in `[0, top_k)`, each hit marked `bundle_resident`, and each route position written exactly once. Use a stack array only:

```cpp
bool written[GGML_MOE_PARTIAL_MAX_ROUTES] = {};
for (int32_t group = 0; group < 2; ++group) {
    const int32_t count = group == 0 ? snapshot->n_hits : snapshot->n_misses;
    const ggml_cache_route_bundle * routes = group == 0 ? snapshot->hits : snapshot->misses;
    for (int32_t i = 0; i < count; ++i) {
        const int32_t route = routes[i].route;
        if (route < 0 || route >= top_k || written[route]) return false;
        written[route] = true;
    }
}
return std::all_of(written, written + top_k, [](bool value) { return value; });
```

`execute()` returns `NOT_ADMITTED` before queueing work for invalid snapshots, unsupported dimensions/types, missing slot descriptors, absent event/copy support, a non-host down output, or an exchange buffer that is not genuinely GPU-host-pinned.

- [ ] **Step 5: Verify the contract**

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: new invalid-snapshot assertions pass; existing cache tests remain unchanged.

- [ ] **Step 6: Request approval before committing Task 2**

Do not commit without explicit user approval.

### Task 3: Build Rebindable Persistent GPU and CPU Execution State

**Files:**
- Modify: `ggml/src/ggml-backend-moe-hetero.cpp`
- Modify: `ggml/src/ggml-backend.cpp:884-915,1333-1445,3339-3362`
- Modify: `tests/test-expert-cache.cpp`

**Interfaces:**
- Consumes: the Task 2 executor API and registered bundle weights returned by `ggml_backend_expert_cache_get_bundle_weights()`.
- Produces: a scheduler-owned catalog keyed by `(gpu_backend, cpu_backend, d_model, d_ff, top_k, is_fused, projection types)` and dispatch pointers to immutable persistent executors.

- [ ] **Step 1: Write the failing persistence test**

Create one executor and assert that it owns all seven GPU variants, all seven CPU variants, the pinned exchange buffer, and the complete event set immediately after construction. Construct a second compatible route-ready dispatch and assert that the scheduler catalog reuses the same executor pointer. Expose identities only through a test-only `#ifdef GGML_TEST` accessor or the fixture; do not add a production debug API.

The test must also require a true CUDA host buffer:

```cpp
ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(gpu_device);
require(host_buft != nullptr);
require(ggml_backend_buffer_get_type(exchange_buffer) == host_buft);
```

- [ ] **Step 2: Run the red test**

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: failure because the executor does not yet allocate persistent variants.

- [ ] **Step 3: Allocate all state in `ggml_moe_partial_executor_new()`**

Reject any `top_k != GGML_MOE_PARTIAL_MAX_ROUTES`. Use `std::array` only for fixed route state. Construct and allocate the following before returning success:

```text
GPU: one input, gate IDs, up IDs when unfused, down IDs, output, graph, and buffer for each K=1..7.
GPU: one canonical [d_model, 8] merge buffer and one [d_model, 8] CPU-upload buffer.
Host: one GPU-host-buffer-type allocation containing x[d_model], CPU miss IDs for M=1..7, and CPU miss output[M][d_model].
CPU: one graph and preallocated output for each M=1..7.
Events: activation-copy start/end, GPU graph start/end, CPU-result H2D start/end, scatter start/end, and final-output completion.
Descriptors: stable copies for GPU slot gate/up/down or gate_up, plus CPU host gate/up/down or gate_up.
```

Build all seven GPU graphs from stable descriptor copies with fixed `K`; build all seven CPU graphs with fixed `M`. Do not mutate `ne[]` in any graph tensor after construction. At execution, copy current slot/host tensor descriptors into the stable descriptors, clear `src`, `view_src`, and `extra` exactly as the full-hit sidecar does, and unbind them after completion.

- [ ] **Step 4: Add catalog ownership in the scheduler**

Add an executor pointer to `ggml_backend_sched_route_ready_dispatch`. Build or reuse the executor during `ggml_backend_sched_build_route_ready_dispatches()` after retrieving that bundle's registered weights. Store unique executors in a scheduler catalog and free each unique object in `ggml_backend_sched_free()`.

Do not allocate an executor from `ggml_backend_sched_graph_compute_async()` or from a decode branch.

- [ ] **Step 5: Verify persistence and CPU/GPU graph construction**

```powershell
cmake --build build --config Release --target test-expert-cache test-backend-ops
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-backend-ops.exe
```

Expected: the new persistence test and existing sidecar tests pass. No code path creates contexts or buffers after executor construction.

- [ ] **Step 6: Request approval before committing Task 3**

Do not commit without explicit user approval.

### Task 4: Implement the Exact-K GPU and Exact-M CPU FFN Paths

**Files:**
- Modify: `ggml/src/ggml-backend-moe-hetero.cpp`
- Modify: `tests/test-expert-cache.cpp`

**Interfaces:**
- Consumes: persistent `gpu[K]`, `cpu[M]`, stable descriptors, and a valid Task 2 snapshot.
- Produces: unweighted packed GPU hit output and unweighted packed CPU miss output, with no route reclassification or second slot lookup.

- [ ] **Step 1: Write the failing 1/8 through 7/8 execution tests**

Seed exactly the desired complete bundles in the cache, call `ggml_backend_expert_cache_partition_bundle_routes()` once, copy its result into a `ggml_moe_partial_route_snapshot`, then call the executor. Compare the canonical unweighted Down result with the ordinary CPU bundle graph. Assert route execution counters separately.

```cpp
require(stats.hetero_partial_exec_by_hits[n_hits] == 1);
require(stats.hetero_partial_gpu_routes_executed == (uint64_t) n_hits);
require(stats.hetero_partial_cpu_routes_executed == (uint64_t) (8 - n_hits));
require(stats.hetero_partial_weight_h2d_bytes == 0);
```

Use these masks at minimum: `CGGGGGGG`, `GGGGGGGC`, `CCGGGGGG`, `GGGCGGGC`, `GGGGCCCC`, `GCGCGCGC`, and `CCGGGGCC`.

- [ ] **Step 2: Run the red test**

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: the new direct-executor tests fail because no exact-K/M graph is executed.

- [ ] **Step 3: Populate metadata once and run only the selected routes**

Use the snapshot as the sole source of route, expert, and slot values. Do not call `find_slot()` or repartition after validation.

```cpp
for (int32_t i = 0; i < snapshot->n_hits; ++i) {
    const ggml_cache_route_bundle & route = snapshot->hits[i];
    hit_gate_slots[i] = bundle->is_fused ? route.gate_up_slot : route.gate_slot;
    hit_up_slots[i] = bundle->is_fused ? route.gate_up_slot : route.up_slot;
    hit_down_slots[i] = route.down_slot;
}
for (int32_t i = 0; i < snapshot->n_misses; ++i) {
    miss_expert_ids[i] = snapshot->misses[i].expert;
}
```

Upload only `K` slot IDs and execute `gpu[K].graph`. Copy only `M` expert IDs into `cpu[M].ids` and execute `cpu[M].graph`. Both paths produce unweighted Down rows.

- [ ] **Step 4: Add pre-launch parity safeguards**

Before queueing work, require F32 TG1 input/output layouts, an exact `[d_model, 8, 1]` canonical output, expected Gate/Up/Down shapes, all required hit slots nonnegative, and host-resident model weights for the CPU graph. A failed check returns `NOT_ADMITTED` before reserve/queue operations.

- [ ] **Step 5: Verify all masks and canonical ordering**

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: every direct mask has CPU-reference-equivalent route output and final weighted/reduced output; a 6/8 test reports exactly six GPU and two CPU executions, independent of hit order.

- [ ] **Step 6: Request approval before committing Task 4**

Do not commit without explicit user approval.

### Task 5: Add Event-Driven Overlap, Pinned Transfers, and Canonical Scatter

**Files:**
- Modify: `ggml/include/ggml-backend.h`
- Modify: `ggml/src/ggml-backend-moe-hetero.cpp`
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Modify: `tests/test-expert-cache.cpp`

**Interfaces:**
- Consumes: exact-K/M state, CUDA-host exchange buffer, GPU event objects, and the existing async copy API.
- Produces: a concurrent path with no backend-wide synchronization, targeted event timings, and a canonical host Down result after GPU-side row scatter.

- [ ] **Step 1: Write the failing no-global-sync and activation-order tests**

Wrap `gpu_backend->iface.synchronize` using the existing `test_count_synchronize` helper. Execute a 7/8 partial bundle through the direct executor and require zero calls while the executor runs. Alternate the activation input on consecutive calls and compare each final result with its ordinary CPU reference; this protects against consuming stale activation.

```cpp
test_synchronize_calls = 0;
gpu_backend->iface.synchronize = test_count_synchronize;
require(ggml_moe_partial_executor_execute(executor, &bundle, cache, &snapshot, &activation, &stats)
        == GGML_MOE_PARTIAL_EXECUTOR_SUCCESS);
gpu_backend->iface.synchronize = test_original_synchronize;
require(test_synchronize_calls == 0);
require(stats.hetero_partial_weight_h2d_bytes == 0);
```

- [ ] **Step 2: Run the red test**

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: the current serial implementation calls backend synchronization or the executor does not yet produce an output.

- [ ] **Step 3: Add precise backend event elapsed support without changing the backend vtable**

Declare this optional proc typedef in `ggml-backend.h` next to existing proc-address typedefs:

```cpp
typedef bool (*ggml_backend_event_elapsed_us_t)(
    ggml_backend_event_t start,
    ggml_backend_event_t end,
    uint64_t * elapsed_us);
```

In `ggml-cuda.cu`, expose `"ggml_backend_event_elapsed_us"` from `ggml_backend_cuda_reg_get_proc_address()`. Implement it with `cudaEventElapsedTime`, return `false` for invalid events/CUDA errors, and convert milliseconds to rounded microseconds. In the core executor, resolve it through `ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu_backend))`; executor creation returns null when the capability is unavailable.

- [ ] **Step 4: Queue the concurrent timeline**

Use this exact ordering. `activation.host_data != nullptr` is the normal CPU-base placement path; it avoids a D2H activation transfer entirely.

```text
1. Validate snapshot and bind descriptors.
2. Reserve every hit bundle slot.
3. Copy host activation into persistent pinned x when host_data exists.
4. Otherwise queue async D2H x, record activation_done, and wait only on that event before CPU work.
5. Queue GPU input/ID uploads and gpu[K].graph; record gpu_hits_done. Do not synchronize.
6. Run cpu[M].graph on pinned host x while the GPU graph is running.
7. Queue the contiguous M-row H2D upload into gpu_cpu_miss_output; record cpu_results_uploaded.
8. Synchronize gpu_hits_done and cpu_results_uploaded individually for measured join waits.
9. Queue K GPU-to-GPU row copies and M GPU-to-GPU row copies into gpu_canonical_output.
10. Queue async D2H of gpu_canonical_output into bundle->down_node; wait only on final_output_done.
11. Release hit slots after gpu_hits_done has completed and unbind descriptors.
```

Do not use `ggml_backend_synchronize()` anywhere in this function. Do not use the current CUDA scatter helper: it is conditionally compiled out of `ggml-base`. Use same-layout stack row descriptors with `ggml_backend_tensor_copy_async()` instead. At construction, reject a backend with no `cpy_tensor_async`; this prevents the generic copy fallback from inserting broad synchronizations.

```cpp
struct ggml_tensor src_row = *executor->gpu[snapshot->n_hits].output;
struct ggml_tensor dst_row = *executor->gpu_canonical_output;
src_row.ne[1] = 1;
dst_row.ne[1] = 1;
src_row.data = (uint8_t *) src_row.data + (size_t) packed_row * src_row.nb[1];
dst_row.data = (uint8_t *) dst_row.data + (size_t) route * dst_row.nb[1];
ggml_backend_tensor_copy_async(executor->gpu_backend, executor->gpu_backend, &src_row, &dst_row);
```

The descriptors are temporary stack copies; no allocated graph tensor is modified.

- [ ] **Step 5: Enforce genuine pinned exchange memory**

Allocate the host context with the GPU device's host buffer type. After allocation, verify exact type identity and emit one debug-only line during executor creation:

```cpp
const bool pinned = ggml_backend_buffer_get_type(executor->host_buffer) == host_buft;
GGML_LOG_DEBUG("%s: exchange buffer %s pinned=%d\n", __func__,
    ggml_backend_buffer_name(executor->host_buffer), (int) pinned);
if (!pinned) {
    return false;
}
```

If allocation falls back to ordinary CPU memory, free the partial executor and leave the scheduler on CPU-base/serial behavior. Do not label `malloc()` or aligned allocation as pinned.

- [ ] **Step 6: Implement conservative post-launch failure cleanup**

After a successful GPU graph submission, every failure path must synchronize only `final_output_done` if recorded, otherwise `gpu_hits_done` if recorded, then release reservations, unbind descriptors, set `executor->poisoned = true`, and return `LAUNCH_FAILED`. Never execute a CPU fallback after the GPU graph has consumed a partial bundle.

- [ ] **Step 7: Verify event and numerical behavior**

```powershell
cmake --build build --config Release --target test-expert-cache test-backend-ops
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-backend-ops.exe
```

Expected: direct 7/8 execution makes zero backend-wide synchronize calls, runs one CPU and seven GPU routes, has zero weight H2D bytes, and matches the ordinary bundle output after router weighting/reduction.

- [ ] **Step 8: Request approval before committing Task 5**

Do not commit without explicit user approval.

### Task 6: Route the Existing 7/8 Branch Through the Feature Gate

**Files:**
- Modify: `ggml/src/ggml-backend.cpp:895-915,2916-3088,3339-3362`
- Modify: `tests/test-expert-cache.cpp`

**Interfaces:**
- Consumes: one route-ID read, a single fixed snapshot, the persistent executor pointer, and existing serial/CPU-base/full-hit paths.
- Produces: default serial 7/8 behavior and opt-in concurrent 7/8 behavior without changing full-hit sidecar or 0-6 fallback behavior.

- [ ] **Step 1: Write failing scheduler feature-gate tests**

Extend the eight-route scheduler fixture to execute the same 7/8 mask with `GGML_EXPERT_CACHE_HETERO_CONCURRENT=0` and `=1`. Verify equal canonical/final output. In the off case assert the existing serial counter behavior; in the on case assert `hetero_partial_exec_by_hits[7] == 1`, 7 GPU routes, 1 CPU route, and no weight H2D.

Use `_putenv_s` on Windows and restore the prior value after each case.

- [ ] **Step 2: Run the red state**

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: the on case still reaches the serial function or the requested telemetry is absent.

- [ ] **Step 3: Snapshot once in the route-ready dispatcher**

For the TG1/top-k-eight branch, replace per-token vectors with stack state:

```cpp
std::array<int32_t, GGML_MOE_PARTIAL_MAX_ROUTES> route_ids = {};
struct ggml_moe_partial_route_snapshot snapshot = {};
ggml_backend_tensor_get(bundle.route_ids, route_ids.data(), 0, route_ids.size() * sizeof(int32_t));
ggml_backend_expert_cache_partition_bundle_routes(
    cache, bundle.layer, route_ids.data(), GGML_MOE_PARTIAL_MAX_ROUTES, 1,
    snapshot.hits, &snapshot.n_hits, snapshot.misses, &snapshot.n_misses);
```

Record route-ready accesses from the same `route_ids` data. Do not call the partition helper again inside the concurrent executor.

- [ ] **Step 4: Preserve the three-way dispatch with an explicit admission constant**

Add this internal constant in `ggml-backend.cpp`:

```cpp
static constexpr int32_t GGML_MOE_PARTIAL_MIN_GPU_HITS = 7;
```

Dispatch rules are:

```cpp
if (snapshot.n_hits == top_k) {
    // unchanged full-hit sidecar
} else if (snapshot.n_hits >= GGML_MOE_PARTIAL_MIN_GPU_HITS &&
           snapshot.n_hits < top_k && concurrent_enabled) {
    // Task 5 executor
} else if (snapshot.n_hits == top_k - 1) {
    // unchanged serial reference path
} else {
    // unchanged CPU-base path
}
```

If the concurrent executor returns `NOT_ADMITTED`, execute the existing CPU-base branch in the same invocation. Do not invoke serial as a hidden fallback when the feature gate requested concurrent mode. Do not touch `GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL`; the matrix script continues to force it to `0`.

- [ ] **Step 5: Preserve producer-before-consumer ordering**

Keep the current producer-inclusive prefix view and its synchronization before route-ID read. In the concurrent partial branch, skip the original Gate/Up/activation/Down range only after `ggml_moe_partial_executor_execute()` returns success. A `NOT_ADMITTED` result must leave the original graph nodes untouched and let the CPU-base branch execute them with valid activation data.

- [ ] **Step 6: Verify gate behavior**

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: default environment retains serial 7/8; feature-enabled environment reaches only concurrent 7/8; 1-6 stay CPU-base; 8/8 still uses the original sidecar.

- [ ] **Step 7: Request approval before committing Task 6**

Do not commit without explicit user approval.

### Task 7: Export Complete Partial-Executor Telemetry

**Files:**
- Modify: `ggml/include/ggml-backend.h:426-451`
- Modify: `ggml/src/ggml-backend-moe-hetero.cpp`
- Modify: `ggml/src/ggml-backend.cpp:3536-3640`
- Modify: `tools/llama-bench/llama-bench.cpp:1560-1606,1740-1804,1830-1873,1950-2005`
- Modify: `tests/test-expert-cache.cpp`

**Interfaces:**
- Consumes: Task 5 event start/end pairs and exact execution counts.
- Produces: cumulative scheduler stats and per-benchmark delta columns that prove what ran and where time/bytes went.

- [ ] **Step 1: Write failing telemetry assertions**

For a concurrent 7/8 fixture, assert nonzero invocation, seven-GPU/one-CPU route counts, one `partial_exec_by_hits[7]` increment, `total_partial_us > 0`, and exact zero for `weight_h2d_bytes`. Assert no activation D2H bytes when the fixture exposes host activation.

- [ ] **Step 2: Add the new stat fields and update them at one ownership point**

Append these fields after existing hetero telemetry:

```cpp
uint64_t hetero_partial_exec_by_hits[9];
uint64_t hetero_partial_gpu_routes_executed;
uint64_t hetero_partial_cpu_routes_executed;
size_t hetero_partial_activation_d2h_bytes;
size_t hetero_partial_cpu_result_h2d_bytes;
size_t hetero_partial_weight_h2d_bytes;
uint64_t hetero_partial_partition_us;
uint64_t hetero_partial_activation_d2h_us;
uint64_t hetero_partial_gpu_hit_submit_us;
uint64_t hetero_partial_gpu_hit_elapsed_us;
uint64_t hetero_partial_cpu_miss_compute_us;
uint64_t hetero_partial_cpu_result_h2d_us;
uint64_t hetero_partial_join_wait_gpu_us;
uint64_t hetero_partial_join_wait_cpu_us;
uint64_t hetero_partial_scatter_us;
uint64_t hetero_partial_total_us;
```

Increment them only for successful concurrent executions. Update `ggml_backend_sched_print_expert_cache_stats()` to print all seven execution buckets, the GPU/CPU route totals, transfer bytes, and the hard `weight_h2d_bytes == 0` invariant.

- [ ] **Step 3: Export every field atomically from `llama-bench`**

Create scalar fields `expert_cache_partial_exec_1_hit` through `_7_hit`, plus fields whose suffixes exactly match the remaining new statistic names. Append them to all four export sites. Update the existing delta helper with loops for both arrays:

```cpp
for (int k = 0; k < 9; ++k) {
    delta.hetero_partial_exec_by_hits[k] = after.hetero_partial_exec_by_hits[k] - before.hetero_partial_exec_by_hits[k];
}
```

Add a focused `get_fields().size() == get_values(...).size()` assertion to the appropriate existing llama-bench test or create a small local helper test alongside the existing export tests. Do not rely on CSV column position alone.

- [ ] **Step 4: Verify telemetry and export**

```powershell
cmake --build build --config Release --target test-expert-cache llama-bench
build/bin/Release/test-expert-cache.exe
build/bin/Release/llama-bench.exe --help
```

Expected: fixture assertions pass and the new names appear in JSONL/CSV field metadata without shifting existing values.

- [ ] **Step 5: Request approval before committing Task 7**

Do not commit without explicit user approval.

### Task 8: Replace the Synthetic Overlap Benchmark with the Real Executor Harness

**Files:**
- Modify: `tests/test-moe-partial-hit-bench.cpp:137-840`
- Modify: `tests/CMakeLists.txt:359-365` only if the target needs an explicit test dependency

**Interfaces:**
- Consumes: Task 2 snapshot API, real persistent executor, real serial reference, and CPU-base graph.
- Produces: raw latency samples and a summary table for identical input/expert routes across CPU-base, serial, and concurrent execution.

- [ ] **Step 1: Write a failing output-contract test for the benchmark**

Remove the `std::async` pseudo-concurrent implementation from the benchmark. Add command-line parsing for:

```text
--warmup N       default 100
--reps N         default 1000
--output PATH    required for timed measurements
--serial-only    run CPU-base and serial only
```

Make the test fail until it writes one CSV row per timed sample with these columns:

```text
hits,misses,mode,iteration,latency_us
```

- [ ] **Step 2: Prebuild all benchmark modes outside the timed loop**

Build once per fixture:

```text
CPU-base graph for all eight host routes.
Persistent serial scratch and cache snapshot per mask.
One Task 3 partial executor.
A deterministic input activation and the same eight expert IDs for all modes.
```

For each mask K=1..7, preload exactly K complete bundles into static slots, build the snapshot once, run 100 warmups per mode, then time 1000 calls per mode. Do not construct a GGML context, buffer, vector, `std::future`, or executor inside an iteration.

- [ ] **Step 3: Compute and print the decision table**

Calculate median and P95 from raw samples and print this exact table shape:

```text
| GPU hits | CPU misses | CPU-base median us | CPU-base P95 us | Serial median us | Serial P95 us | Concurrent median us | Concurrent P95 us | Best |
| -------: | ---------: | -----------------: | ---------------: | ---------------: | ------------: | -------------------: | -------------------: | ---- |
```

`Best` is CPU-base, serial, or concurrent by median; it is not the production admission decision.

- [ ] **Step 4: Verify benchmark correctness and artifact generation**

```powershell
cmake --build build --config Release --target test-moe-partial-hit-bench
build/bin/Release/test-moe-partial-hit-bench.exe --warmup 2 --reps 4 --output tools/results/expert-cache/partial-smoke.csv
```

Expected: every mask matches the canonical CPU output, the CSV has `7 * 3 * 4` sample rows, and the summary includes median/P95 columns.

- [ ] **Step 5: Request approval before committing Task 8**

Do not commit without explicit user approval.

### Task 9: Measure 7/8 Overlap Before Broadening Policy

**Files:**
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`
- Create: `tools/results/expert-cache/2026-08-30-partial-mask-latency.csv` as generated evidence

**Interfaces:**
- Consumes: real executor telemetry and the Task 8 raw benchmark.
- Produces: a retained 1/8 through 7/8 latency table and profiler evidence that 7/8 GPU/CPU work overlaps.

- [ ] **Step 1: Build the benchmark binaries**

```powershell
cmake --build build --config Release --target test-expert-cache test-moe-partial-hit-bench llama-bench
```

- [ ] **Step 2: Run the mandatory mask matrix**

```powershell
build/bin/Release/test-moe-partial-hit-bench.exe --warmup 100 --reps 1000 --output tools/results/expert-cache/2026-08-30-partial-mask-latency.csv
```

Expected: retained raw samples for 1/8 through 7/8 and a printed median/P95 table for CPU-base, serial, and concurrent with the identical route set for each row.

- [ ] **Step 3: Capture the 7/8 timeline**

```powershell
nsys profile --trace=cuda,osrt --force-overwrite=true --output tools/results/expert-cache/2026-08-30-partial-7of8 `
  build/bin/Release/test-moe-partial-hit-bench.exe --warmup 100 --reps 1000 --output tools/results/expert-cache/2026-08-30-partial-mask-latency.csv
```

Inspect the trace for overlapping CPU miss computation and GPU hit graph execution. Record the observed overlap and the executor timing counters; do not infer overlap merely from lower wall-clock time.

- [ ] **Step 4: Apply the first decision gate**

Concurrent 7/8 is eligible to replace serial only if its median is faster than both serial and CPU-base and its P95 does not regress versus CPU-base. If either condition fails, keep the concurrent gate experimental and retain serial as production policy. Record the table and decision in the optimization log.

- [ ] **Step 5: Request approval before committing measurement evidence**

Generated evidence and log updates are not committed without explicit user approval.

### Task 10: Set the Measured Admission Threshold and Verify Whole-Model Behavior

**Files:**
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `tools/results/expert-cache/run-tg-matrix.py`
- Modify: `tools/results/expert-cache/test_run_tg_matrix.py`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: Task 9 raw table, resolved cross-split gate, and current dynamic 128 MiB Compact configuration.
- Produces: an evidence-selected minimum hit count, reproducible serial/concurrent fresh-process matrices, deterministic generation comparison, and production telemetry.

- [ ] **Step 1: Choose the threshold mechanically from the retained table**

Select the smallest K in `[1, 7]` whose concurrent median is at least 5 percent faster than CPU-base and whose P95 is not slower than CPU-base. Require the Task 1 cross-split regression to pass before selecting K below 7. If no K meets both rules, leave `GGML_MOE_PARTIAL_MIN_GPU_HITS` at 7 and leave concurrent development-gated.

Replace the constant only with the actual selected integer and add a test that asserts K-1 stays CPU-base while K reaches concurrent execution.

- [ ] **Step 2: Make the matrix runner set both heterogeneous switches explicitly**

Add this argument:

```python
parser.add_argument("--hetero-concurrent", choices=(0, 1), type=int, default=0)
```

Build the child environment in a named helper:

```python
def bench_environment(args: argparse.Namespace) -> dict[str, str]:
    return os.environ | {
        "GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL": "0",
        "GGML_EXPERT_CACHE_HETERO_CONCURRENT": str(args.hetero_concurrent),
    }
```

Use it for every `subprocess.run()` call. Add a Python unit test for both values so a parent shell cannot accidentally select the wrong policy.

- [ ] **Step 3: Run deterministic generation for every proposed production threshold**

Use the Compact model and the same prompt, seed, temperature zero, top-k one, and at least 256 generated tokens for cache-off and cache-on. Save both raw token streams and SHA-256 hashes under `tools/results/expert-cache/`.

```powershell
$env:GGML_EXPERT_CACHE_HETERO_CONCURRENT = "1"
# Run the existing Compact server preset twice with identical --seed, --temp 0, --top-k 1, prompt, and 256-token limit.
# Save token-only output before calculating hashes.
Get-FileHash tools/results/expert-cache/concurrent-cache-off.tokens -Algorithm SHA256
Get-FileHash tools/results/expert-cache/concurrent-cache-on.tokens  -Algorithm SHA256
```

Require coherent output, no NaN/Inf/CUDA error, and exact hashes when the platform produces identical arithmetic. If hashes differ, record the first differing token and its logit margin before considering the threshold eligible.

- [ ] **Step 4: Run paired fresh-process TG matrices with unchanged placement**

Use the same model placement and dynamic configuration in every run: `-fitt 256 -exc 128 -excp 32`, no manifest or promotion-policy changes.

```powershell
python tools/results/expert-cache/run-tg-matrix.py `
  --model "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" `
  --runs 5 --cache-mib 128 --cache-period 32 --fit-target 256 `
  --hetero-concurrent 0 --prefix 2026-08-30-partial-serial

python tools/results/expert-cache/run-tg-matrix.py `
  --model "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" `
  --runs 5 --cache-mib 128 --cache-period 32 --fit-target 256 `
  --hetero-concurrent 1 --prefix 2026-08-30-partial-concurrent
```

If K below 7 was selected, run a third matrix after the exact threshold change with the same arguments and prefix `2026-08-30-partial-threshold-K`. Do not use the 1024 MiB auto-fit profile for this decision.

- [ ] **Step 5: Verify telemetry reaches the real model**

From the cache-on JSONL, report classified mask histogram, admitted mask histogram, executor invocation count, GPU routes, CPU routes, and `expert_cache_partial_weight_h2d_bytes == 0`. Reject a synthetic-only win when real-model telemetry shows no partial executor invocations.

- [ ] **Step 6: Apply the production decision**

Enable concurrent partial execution by default only when all of these are true:

```text
The selected mask has a >=5 percent CPU-base median advantage.
P95 is not worse than CPU-base.
The cross-split regression is deterministic.
The 256-token deterministic run is coherent.
The full TG matrix does not regress against serial policy.
Real-model telemetry proves the selected masks were admitted and executed.
```

Otherwise retain default-off `GGML_EXPERT_CACHE_HETERO_CONCURRENT`, serial 7/8 production behavior, and document the negative result.

- [ ] **Step 7: Verify runner and focused tests**

```powershell
python -m unittest tools/results/expert-cache/test_run_tg_matrix.py
cmake --build build --config Release --target test-expert-cache llama-bench llama-server
build/bin/Release/test-expert-cache.exe
```

Expected: runner environment tests pass, source compiles, and direct executor/scheduler tests remain green.

- [ ] **Step 8: Request approval before committing Task 10**

Do not commit without explicit user approval.

### Task 11: Final Documentation and Verification

**Files:**
- Modify: `EXPERT_CACHE.md`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: selected threshold decision, raw latency CSV, profiler trace, generation token artifacts, and TG matrix JSONL.
- Produces: documentation that states exactly which masks are admitted, why, and what remains rejected.

- [ ] **Step 1: Update `EXPERT_CACHE.md`**

Replace the current one-miss description with the final measured state:

```text
0 through K-1 hits: CPU-base.
K through 7 hits: concurrent partial executor, only if Task 10 enabled it.
8 hits: unchanged full-hit GPU sidecar.
```

Describe the TG1-only shape gate, persistent K/M graphs, host-buffer requirement, event-based joins, zero current-token weight H2D invariant, and the feature gate/default state. Do not claim PP or multi-token support.

- [ ] **Step 2: Update `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`**

Append:

```text
Cross-split diagnosis and final regression command.
All seven CPU-base/serial/concurrent median and P95 rows.
Raw CSV and Nsight trace paths.
Selected threshold and the explicit 5 percent/P95 rule result.
Generation token hashes or first divergence analysis.
Serial/concurrent/threshold TG JSONL paths and paired summary.
Mask/admission/executed-route telemetry from the real model.
A statement that cache size, fit target, manifest, promotion, and sidecar were held fixed.
```

Record a negative result if no threshold was promoted; it is still required evidence.

- [ ] **Step 3: Run final focused verification**

```powershell
cmake --build build --config Release --target test-expert-cache test-backend-ops test-moe-partial-hit-bench llama-bench llama-server
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-backend-ops.exe
build/bin/Release/test-moe-partial-hit-bench.exe --warmup 100 --reps 1000 --output tools/results/expert-cache/2026-08-30-partial-mask-latency-final.csv
python -m unittest tools/results/expert-cache/test_run_tg_matrix.py
ctest --test-dir build -C Release --output-on-failure
```

Expected: all tests pass, including the now-stable cross-split regression; the final raw partial matrix, profiler evidence, and whole-model matrix are retained.

- [ ] **Step 4: Request approval before final commit(s)**

Present the exact source, test, benchmark, and documentation diffs. Create only approved atomic commits; do not push.

## Acceptance Matrix

| Criterion | Evidence |
| --- | --- |
| No steady-state allocations | Persistent-construction test plus executor source path contains no context/buffer/vector allocation. |
| Exact route split | Direct 1/8 through 7/8 tests assert GPU and CPU execution counters independently. |
| Actual overlap | Nsight Systems trace and Task 7 event timings for 7/8. |
| No broad synchronization | Direct executor test intercepts zero `gpu_backend->iface.synchronize` calls. |
| Zero weight H2D | Direct tests and real-model JSONL report `expert_cache_partial_weight_h2d_bytes == 0`. |
| Correct ordering and parity | Alternating-activation regression plus canonical route/final-output comparisons. |
| Cross-split safety | Task 1 repeated CTest regression passes before lower-mask admission. |
| Measured admission | Retained 100-warmup/1000-sample raw CSV with median and P95 decision rule. |
| No end-to-end regression | Same-placement fresh-process serial/concurrent TG matrices. |
| Accurate docs | `EXPERT_CACHE.md` and optimization log cite final threshold and raw evidence paths. |
