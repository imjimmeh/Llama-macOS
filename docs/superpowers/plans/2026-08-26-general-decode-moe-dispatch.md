# General Decode MoE Route-Aware Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make expert-cache dispatch depend on the current decode-time MoE route for every supported microbatch form: normal parallel generation, speculative verification, MTP target/draft graphs, and future decode batches.

**Architecture:** Preserve the existing slot-pool `MUL_MAT_ID` execution path as the GPU full-hit mechanism. Add scheduler-owned route plans that are discovered from original graph `MUL_MAT_ID` dependencies, execute routing before dispatch, and select either full-bundle GPU slot-pool execution or the unchanged CPU operation for the current route. CPU fallback may enqueue one bounded, completion-safe fill for a later route; it never waits for that fill. Mixed CPU/GPU expert execution is excluded until the full-hit path proves value.

**Tech Stack:** C++17, GGML graph and scheduler APIs, backend event APIs, CUDA backend events and asynchronous tensor copies, CMake/CTest, llama-bench, llama-server, Python deterministic token-hash harness.

## Global Constraints

- General decode microbatches are the scope. Do not specialize behavior around `n_tokens == 1` or MTP alone.
- Do not globally lower `GGML_OP_OFFLOAD_MIN_BATCH` or force host-resident MoE `MUL_MAT_ID` operations to CUDA. Existing Compact and MTP forced-routing measurements reject that mechanism.
- The phase-one dispatch table is binary: complete current-route bundle hit -> GPU slot-pool execution; every other state -> existing CPU operation. There is no partial CPU/GPU merge.
- Route decisions must be based on the current graph generation, logical sequence identity, source tensor identities, and current selected-expert IDs. Do not use tensor display names as identity.
- `LOADING` entries are never cache hits. Publish `RESIDENT` only after nonblocking completion observation. Eviction/reuse waits for fill and consumer-use completion.
- CPU fallback and route-plan discovery must not wait for a fill or introduce a synchronous router-ID read in the normal decode compute stream.
- Background filling is opt-in, backend-capability-gated, one complete bundle at a time, and fail-closed. No valid budget or event capability means no fill.
- Reactive cache misses remain a correctness fallback while existing behavior is retained. New route dispatch remains disabled by default until measurement validates it.
- Every source or behavior change requires a focused test result, deterministic server result, control-versus-enabled benchmark, and a dated append to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`, including rejected and no-effect results.
- Throughput claims require five alternating fresh-process control/enabled pairs with fixed model placement, persistence disabled, identical binary/model/options, raw rows, median, mean, standard deviation, TG and PP separately, and causal cache telemetry.
- MTP comparisons must report target calls, accepted drafts, target tokens, dynamic-promotion state, and VRAM placement. Do not compare rows whose placement differs.
- Commits are allowed because the user explicitly requested them; each commit uses an `Assisted-by: OpenAI Codex` trailer. Never push or create a PR.

## Implementation checkpoint (2026-08-26)

- Tasks 1 and 2 are complete: route census, benchmark schema, nonblocking event query, completion-gated slot lookup, and consumer-use event recording.
- Task 3 is partial: shared route-plan grouping and host-visible route capture exist, but the scheduler does not yet execute a router checkpoint before backend dispatch.
- Task 4 is partial: the zero-copy path now requires an already complete full hit; a missing route uses the existing copied-tensor fallback. CPU-on-miss dispatch is not implemented.
- Tasks 5 and 6 remain conditional: no bounded CPU-fallback fill queue or workload-separated route dispatcher has been retained.
- Current Compact TG remains CPU-routed with zero cache requests. The forced placement diagnostic remains rejected.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `ggml/include/ggml-backend.h` | Public aggregate expert-cache telemetry and backend-event query API. |
| `ggml/src/ggml-backend-impl.h` | Optional backend-device event-query callback contract. |
| `ggml/src/ggml-backend.cpp` | Original-graph route census, scheduler route-plan discovery, phase execution, dispatch, and telemetry aggregation. |
| `ggml/src/ggml-backend-expert-cache.h` | Cache-private route-plan, slot lifecycle, fill reservation, and consumer-use interfaces. |
| `ggml/src/ggml-backend-expert-cache.cpp` | Completion-safe slots, complete-bundle reservation, bounded fill queue, and canonical source resolution. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | CUDA event-query implementation and no other cache policy. |
| `src/llama-context.cpp` | Logical graph/sequence identity, graph lifecycle, expert-bundle registration, and opt-in scheduler configuration. |
| `src/llama-graph.cpp` | Only change if structural graph discovery cannot unambiguously recognize supported `MUL_MAT_ID` route forms. |
| `src/llama-cparams.h`, `common/common.h`, `common/arg.cpp` | Disabled-by-default public route-dispatch option if Task 3 proves a runtime option is needed. |
| `tests/test-expert-cache.cpp` | Scheduler fixture, route identity, slot lifecycle, route dispatch, and fill-budget tests. |
| `tests/test-expert-cache-profile.cpp` | Canonical source/profile seed behavior where route plans resolve persisted entries. |
| `tests/test-backend-ops.cpp` | CPU/CUDA `MUL_MAT_ID` batch-shape and remapped-ID numerical equivalence. |
| `tests/test-mtp-dynamic-offload.cpp` | MTP promotion and cache-route separation. |
| `tests/unit/test_speculative.py` | Target/draft parity integration coverage. |
| `tools/llama-bench/llama-bench.cpp` | Telemetry schema, aggregation delta, and benchmark serialization. |
| `tools/server/server-context.cpp` | Optional server timing output for retained route-dispatch counters. |
| `EXPERT_CACHE.md` | Current retained behavior and link to this plan. |
| `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` | Dated, append-only evidence and retain/reject decisions. |

---

### Task 1: Add a behavior-preserving route census

**Files:**
- Modify: `ggml/include/ggml-backend.h`
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `tools/llama-bench/llama-bench.cpp`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: original `ggml_cgraph`, scheduler backend assignments, existing `ggml_backend_expert_cache_stats`, and registered expert bundles.
- Produces: aggregate route-census counters and a sampled developer trace that classify original-graph and split-graph MoE `MUL_MAT_ID` nodes without reading IDs from device memory.

- [ ] **Step 1: Write failing scheduler-census tests**

Add a synthetic graph fixture with host expert weights and route IDs. Exercise three classifications: CPU-routed host weights, non-CPU-routed host weights, and non-CPU-routed device weights. Assert that the census records original graph counts separately from cache-eligible split inputs, preserves node outputs, and emits no route-ID transfer or synchronization counter.

```cpp
static void test_expert_cache_route_census_classifies_original_graph() {
    // Build one graph with three independent MUL_MAT_ID nodes.
    // Assert CPU-host, GPU-host, and GPU-device classifications are distinct.
    // Run cache-off and census-enabled; assert output bytes match.
}
```

- [ ] **Step 2: Run the focused test and confirm red**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: the new route-census assertions fail because current statistics only count cache-intercepted split inputs.

- [ ] **Step 3: Add original-graph route census and aggregate fields**

In `ggml_backend_sched_split_graph()`, structurally scan original graph nodes for registered-expert `GGML_OP_MUL_MAT_ID` forms before split-input transformation. For every supported node, record:

```text
operation count
assigned backend
source buffer usage and host/device residency
op->ne[2] batch size
layer and canonical source identity
split-input visibility
cache eligibility or exact bypass classification
```

Store sampled per-plan diagnostics in scheduler-private state. Do not call `ggml_backend_tensor_get_async()`, `ggml_backend_synchronize()`, or mutate assignment while collecting this data. Extend `ggml_backend_expert_cache_stats` with aggregate counters only; append matching field name, type, serialization, and subtraction entries in llama-bench as one atomic schema change.

- [ ] **Step 4: Run focused verification**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache llama-bench
build/bin/Release/test-expert-cache.exe
build/bin/Release/llama-bench.exe --help
```

Expected: original and split counts remain distinguishable; all telemetry fields have one declared type and one serialized value; the fixture output is unchanged.

- [ ] **Step 5: Collect the decision evidence before changing dispatch**

Run fixed-placement, fresh-process observations for normal parallel decode, speculative verification, MTP target, and MTP draft. Record the batch-size histogram, assignment, source residency, route-plan visibility, placement, and route-ID transfer counters. Run deterministic cache-off/census-enabled completions and append the complete result to the optimization log.

**Decision gate:** if a workload has no useful GPU-eligible route plan, do not implement cache dispatch for that workload. If normal decode is CPU-routed, retain it as the CPU baseline rather than forcing CUDA.

---

### Task 2: Make slot readiness and reuse completion-safe

**Files:**
- Modify: `ggml/include/ggml-backend.h`
- Modify: `ggml/src/ggml-backend-impl.h`
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `ggml/src/ggml-backend-expert-cache.h`
- Modify: `ggml/src/ggml-backend-expert-cache.cpp`
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Modify: every backend device-interface initializer affected by the new optional callback
- Modify: `tests/test-expert-cache.cpp`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: slot `EMPTY`/`LOADING`/`RESIDENT` state, existing backend event APIs, staging events, and scheduler split completion.
- Produces: optional `ggml_backend_event_query()` readiness contract; cache-private fill and consumer event ownership; `find_slot()` that returns only proven-resident slots.

- [ ] **Step 1: Write failing lifecycle tests**

Replace the current immediate-promotion expectation with tests that use a manually controlled asynchronous test backend event. Assert all four transitions:

```cpp
static void test_expert_cache_loading_slot_never_hits_before_completion();
static void test_expert_cache_duplicate_fill_attaches_to_loading_slot();
static void test_expert_cache_eviction_waits_for_last_gpu_consumer();
static void test_expert_cache_failed_bundle_fill_restores_all_mappings();
```

The first test must prove that a transfer enqueue followed by `promote_slot()` does not make the slot a hit until the event query reports completion. The third must prove that eviction cannot overwrite a consumed GPU slot before its consumer event completes.

- [ ] **Step 2: Run the focused test and confirm red**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: current `promote_slot()` immediately publishes `RESIDENT`, so the pre-completion assertion fails.

- [ ] **Step 3: Add optional nonblocking event query and centralized transitions**

Add this public backend event function and a matching optional device-interface callback:

```cpp
GGML_API bool ggml_backend_event_query(ggml_backend_event_t event);
```

CUDA implements it with nonblocking event status. Backends without event-query capability report that route-cache fills are unsupported rather than synchronizing. Centralize these transitions in cache-private helpers:

```text
reserve_bundle -> LOADING
poll_bundle_completion -> RESIDENT
record_bundle_consumer -> consumer event owned by slots
release_or_evict -> wait for fill and consumer completion
cancel_bundle_fill -> remove every reservation and mapping
```

Change `ggml_backend_expert_cache_promote_slot()` semantics or replace it with explicit completion polling so enqueue and publication cannot be confused. Record the consumer event after the final GPU expert projection that consumes the slot-pool source.

- [ ] **Step 4: Run focused verification**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache test-expert-cache-profile
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-expert-cache-profile.exe
```

Expected: loading slots are not hits, duplicate claims issue one fill, failed fills leave no mapping, and eviction waits for live consumers.

- [ ] **Step 5: Record the safety baseline**

Run deterministic cache-off and cache-enabled server completions at fixed placement. Append lifecycle counters, unsupported-backend behavior, and deterministic results to the optimization log. Do not claim a TG gain from this safety change.

---

### Task 3: Discover route plans and execute the router before dispatch

**Files:**
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `ggml/src/ggml-backend-expert-cache.h`
- Modify: `ggml/src/ggml-backend-expert-cache.cpp`
- Modify: `src/llama-context.cpp`
- Modify: `src/llama-graph.cpp` only if structural discovery cannot distinguish registered expert bundles from adapter `MUL_MAT_ID` forms
- Modify: `src/llama-cparams.h`, `common/common.h`, `common/arg.cpp` if an opt-in public flag is required
- Modify: `tests/test-expert-cache.cpp`
- Modify: `tests/test-mtp-dynamic-offload.cpp`
- Modify: `tests/unit/test_speculative.py`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: original graph `MUL_MAT_ID` nodes, `src[2]` route tensors, expert-bundle registration, graph generation, and logical sequence identity.
- Produces: scheduler-private immutable route plans and graph checkpoints that make current IDs available before choosing the MoE execution backend.

- [ ] **Step 1: Write failing route-plan tests**

Add scheduler tests for a graph containing merged gate/up plus down and one graph containing separate gate, up, and down. Assert that each plan contains only registered expert projections, groups nodes sharing the same route-ID tensor, preserves layer separation for same-shaped weights, and rejects an adapter-style unregistered `MUL_MAT_ID`.

```cpp
static void test_route_plan_groups_complete_moe_bundle();
static void test_route_plan_rejects_unregistered_mul_mat_id();
static void test_route_plan_rejects_stale_graph_identity();
```

Add integration assertions that normal parallel decode, speculative target/draft contexts, and MTP graphs produce separate logical graph identities.

- [ ] **Step 2: Run the focused tests and confirm red**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache test-mtp-dynamic-offload
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-mtp-dynamic-offload.exe
python -m pytest tests/unit/test_speculative.py
```

Expected: current scheduler has no route-plan/checkpoint ownership and the new assertions fail.

- [ ] **Step 3: Implement structural discovery and checkpointed execution**

Discover plans from original graph dependencies rather than split inputs. A supported plan must have a registered bundle and a shared current route tensor. Store this identity exactly:

```cpp
struct ggml_backend_moe_route_plan {
    uint64_t graph_generation;
    uint64_t logical_sequence_id;
    int32_t  layer;
    const ggml_tensor * route_ids;
    ggml_backend_expert_bundle_identity bundle;
    int64_t n_tokens;
};
```

Use graph views or scheduler-owned phase checkpoints to execute all dependencies through `route_ids` before reading its values. After that checkpoint, form the deduplicated current route union and resplit only the remaining graph region. Do not use tensor names as identities. Do not add a broad callback that synchronizes every graph node. If a route cannot be structurally recognized, select ordinary scheduler execution and count `unsupported`.

Wire graph generation and logical sequence identity from `llama_context` graph lifecycle. Keep MTP target and draft contexts distinct. Add a disabled-by-default `--expert-cache-route-dispatch` option only after the scheduler checkpoint has a working internal contract.

- [ ] **Step 4: Run focused verification**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache test-mtp-dynamic-offload llama-server
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-mtp-dynamic-offload.exe
python -m pytest tests/unit/test_speculative.py
```

Expected: supported plans pause only at router checkpoints, unsupported forms retain ordinary execution, and target/draft parity is unchanged.

- [ ] **Step 5: Record checkpoint cost and route lead**

Run the route census from Task 1 with route dispatch disabled and checkpoint tracing enabled. Record router-ready to expert-consumer lead time, checkpoint count, backend assignment, and all bypass reasons. Reject the route-dispatch path for any workload where checkpoint cost or absent lead prevents a useful full-hit decision.

---

### Task 4: Dispatch complete current-route hits to slot pools and all other routes to CPU

**Files:**
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `ggml/src/ggml-backend-expert-cache.h`
- Modify: `ggml/src/ggml-backend-expert-cache.cpp`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `tests/test-backend-ops.cpp`
- Modify: `tools/llama-bench/llama-bench.cpp`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: completed route plans, completion-safe slot pools, canonical bundle identities, current route union, and normal CPU `MUL_MAT_ID` scheduling.
- Produces: `gpu_full_hit` or `cpu_fallback` for a complete route plan, plus telemetry that explains every decision.

- [ ] **Step 1: Write failing numerical-equivalence and dispatch tests**

Add tests for current route batches with one, several, and many decode tokens. For each shape, run cache-off CPU output, a complete warm bundle full hit, and one missing selected expert. Assert:

```cpp
static void test_route_dispatch_full_hit_uses_remapped_slot_pool();
static void test_route_dispatch_any_missing_bundle_member_uses_cpu();
static void test_route_dispatch_preserves_batch_order_and_output();
```

The mixed-ID test must prove that one missing expert selects CPU for the whole phase-one route plan; it must not upload the miss or partially rewrite `node->src[0]`.

- [ ] **Step 2: Run focused tests and confirm red**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache test-backend-ops
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-backend-ops.exe
```

Expected: current code fills misses immediately and executes the cache-eligible operation on GPU, so CPU-fallback assertions fail.

- [ ] **Step 3: Implement binary route dispatch**

After a route checkpoint, resolve canonical source tensors and test every selected expert against every projection in the complete bundle. Apply exactly one decision:

```text
all selected projections RESIDENT -> upload remapped IDs, rewrite supported nodes to slot pools, execute GPU phase, record consumer event
otherwise                       -> leave original nodes/placement on CPU, execute CPU phase unchanged
```

For a full hit, remap every route ID and restore all graph pointers after compute. For CPU fallback, submit no current-route host-to-device expert transfer. Preserve the existing reactive path outside opt-in route dispatch until its replacement is fully validated.

- [ ] **Step 4: Run focused verification**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache test-backend-ops llama-server
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-backend-ops.exe
python scripts/expert-cache-determinism.py --exc 0 --json-out tools/results/expert-cache/route-dispatch-control.json
python scripts/expert-cache-determinism.py --exc 64M --extra-args "--expert-cache-route-dispatch" --json-out tools/results/expert-cache/route-dispatch-enabled.json
```

Expected: hashes match for fixed placement; full hits use slot pools; any miss executes CPU; unsupported routes retain default scheduling.

- [ ] **Step 5: Establish the warm-bundle ceiling**

Benchmark cache-off CPU fallback against fixed-placement full-bundle warm hits for every workload that passed Task 3. Use five alternating fresh-process pairs. Keep this feature only where GPU full hits beat CPU fallback beyond observed variance and telemetry confirms avoided current-route H2D traffic.

---

### Task 5: Add bounded, nonblocking complete-bundle fills behind CPU fallback

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.h`
- Modify: `ggml/src/ggml-backend-expert-cache.cpp`
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `ggml/include/ggml-backend.h`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `tools/llama-bench/llama-bench.cpp`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: CPU-fallback route plans, complete-bundle slot reservation, event-query capability, staging arena, and canonical source tensors.
- Produces: one backend-owned in-flight fill bundle at most; nonblocking fill counters and useful-after-fill attribution.

- [ ] **Step 1: Write failing fill-budget tests**

Add tests for these exact outcomes:

```cpp
static void test_cpu_fallback_enqueues_at_most_one_complete_bundle_fill();
static void test_fill_rejects_without_evicting_resident_bundle();
static void test_duplicate_cpu_fallback_coalesces_loading_bundle();
static void test_incomplete_fill_never_stalls_next_cpu_fallback();
```

Use a manually controlled event backend. Verify that a full queue, incomplete event, unsupported event query, or insufficient complete-bundle capacity records rejection and leaves current CPU output and resident mappings unchanged.

- [ ] **Step 2: Run the focused tests and confirm red**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: current prefetch/fill code has no complete-bundle queue, byte budget, or CPU-fallback integration.

- [ ] **Step 3: Implement one-bundle best-effort filling**

Reserve all required projection slots before submitting a copy. Reject if the complete bundle cannot fit, the backend lacks event query, another bundle is already loading, or reservation would evict a live resident. Deduplicate requests against `LOADING` reservations. Submit copies through existing asynchronous backend APIs, retain source identity until completion, and publish only after every projection completes.

Do not add public tuning knobs in this task. The fixed initial policy is one complete bundle in flight per backend and zero decode-stream waits. Record submitted, rejected, coalesced, completed, cancelled, bytes, completion latency, and useful-after-fill outcomes.

- [ ] **Step 4: Run focused verification**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache test-expert-cache-profile llama-server
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-expert-cache-profile.exe
python scripts/expert-cache-determinism.py --exc 64M --extra-args "--expert-cache-route-dispatch" --json-out tools/results/expert-cache/route-fill-enabled.json
```

Expected: every fallback remains nonblocking and token-identical; fills become useful only on later full-hit route plans.

- [ ] **Step 5: Decide whether fills pay for themselves**

Benchmark CPU fallback alone against CPU fallback plus bounded fills for workloads that passed Task 4. Require useful-after-fill hits, reduced current-route H2D traffic, and a TG gain beyond paired-run variance. If fills add noise or have no useful hits, remove the fill queue and retain full-hit dispatch only.

---

### Task 6: Validate all decode microbatch integrations and retain only evidence-backed modes

**Files:**
- Modify: `tests/test-expert-cache.cpp`
- Modify: `tests/test-mtp-dynamic-offload.cpp`
- Modify: `tests/unit/test_speculative.py`
- Modify: `tools/llama-bench/llama-bench.cpp`
- Modify: `tools/server/server-context.cpp`
- Modify: `EXPERT_CACHE.md`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: retained route dispatch and fill behavior from prior tasks.
- Produces: workload-separated telemetry and documented retained/rejected modes.

- [ ] **Step 1: Write failing workload-separation tests**

Add assertions that normal parallel decode, speculative target verification, speculative draft execution, MTP target execution, and MTP draft execution use distinct logical graph/sequence identities. Assert MTP dynamic promotion never turns promoted device weights into host-cache candidates and that cache profile/reservation changes are reported as placement changes.

- [ ] **Step 2: Run focused tests and confirm red**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache test-mtp-dynamic-offload llama-bench
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-mtp-dynamic-offload.exe
python -m pytest tests/unit/test_speculative.py
```

Expected: current telemetry does not fully distinguish target, draft, MTP, and normal decode route-plan ownership.

- [ ] **Step 3: Serialize workload-separated telemetry**

Ensure llama-bench aggregation and output fields remain schema-aligned across `get_fields()`, `get_field_type()`, `get_values()`, and `subtract_expert_cache_stats()`. Extend `server_slot::print_timings()` only for retained counters that explain server-side route dispatch. Emit separate counters for workload label, batch histogram, CPU fallback, GPU full hit, fill state, useful-after-fill hit, and bypass reason. A zero-request row remains valid evidence and must not be converted into a cache performance claim.

- [ ] **Step 4: Run deterministic integration verification**

For every retained mode, run fixed-placement greedy controls with cache disabled, route dispatch enabled without fills, and route dispatch plus fills. Run MTP target/draft controls separately and record draft acceptance, target calls, token hashes, placement, and promotion state.

Expected: matching hashes within a fixed placement row; any changed placement invalidates the comparison rather than being reported as an improvement.

- [ ] **Step 5: Run the final benchmark matrix and document the decision**

Use five alternating fresh-process pairs for each retained workload/mode. Report normal decode, parallel decode, speculative verification, MTP target, MTP draft, and PP separately. Append every raw row and the retain/reject decision to the optimization log. Update `EXPERT_CACHE.md` with only retained behavior and link rejected mechanisms to dated evidence.

---

## Deferred Work: Mixed CPU/GPU Expert Execution

Do not implement this in the first route-aware dispatch change. It is justified only when all of these observations are true:

1. full-hit GPU dispatch beats CPU fallback beyond variance;
2. partial misses dominate the remaining CPU-fallback workload;
3. route-plan identities and completion ownership are stable;
4. a projection-level equivalence test can partition IDs, run CPU and GPU subsets, preserve expert order, and merge outputs without changing tokens.

A later design may partition a route into resident and missing IDs, execute resident experts from GPU slots and missing experts on CPU, then merge each projection result. That is a separate design because gate/up/down ordering, merged gate/up forms, output accumulation, memory ownership, and synchronization are correctness-critical.

## Final Verification and Documentation Pass

- [ ] Build focused retained targets:

```powershell
cmake --build build --config Release --target test-expert-cache test-expert-cache-profile test-backend-ops test-mtp-dynamic-offload llama-bench llama-server
```

- [ ] Run each focused executable and the speculative parity test.
- [ ] Re-run deterministic controls for every retained mode; compare hashes only inside fixed-placement rows.
- [ ] Re-run every retained alternating benchmark matrix and include cache-off, CPU fallback, GPU full-hit, and bounded-fill rows where applicable.
- [ ] Append the full evidence, rejected paths, placement records, and final decision to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` without editing earlier sections.
- [ ] Update `EXPERT_CACHE.md` only with verified current behavior. Do not present this proposed design or a planned mixed path as implemented.
- [ ] Do not push or create a PR. Commit only user-authorized changes with the required `Assisted-by: OpenAI Codex` trailer.
