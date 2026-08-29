# Route-Ready Token Generation Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore cache-enabled Compact TG to the cache-off deployment baseline without reintroducing stale route-ID CUDA faults, then admit GPU slot-pool execution only after each route-ID producer has completed.

**Architecture:** Keep the current `route_ids_pending` safety invariant during graph construction. Refactor the scheduler so it executes the graph prefix ending at a route-ID producer, waits for that prefix, reads the now-valid route IDs once, applies cache remapping to the dependent FFN bundle, and executes the remaining graph view. Use existing `ggml_graph_view()` execution mechanics rather than creating a second scheduler or changing CUDA kernels. Decouple cache configuration from global op-level placement so inactive cache capacity cannot displace profitable full GPU MoE layers.

**Tech Stack:** C++17, ggml scheduler and backend events, CUDA backend, `test-expert-cache`, `llama-bench`, `llama-server`, PowerShell on Windows.

## Global Constraints

- Preserve the stale route-ID safety rule: no host route read or slot remap may occur before the graph operation producing that ID tensor has completed.
- Preserve graph reuse: all temporary `src[0]`, `src[2]`, and `op` mutations must be restored after each scheduler compute.
- Cache-off output, cache-on output, and route-ready output must produce the same fixed greedy token sequence.
- Never force host MoE weights onto CUDA merely because cache capacity exists. CPU miss routes must not upload expert weights during timed TG.
- Do not add a CUDA kernel or a background subsystem in this plan.
- Keep cache-specific scheduling limited to single-token `GGML_OP_MUL_MAT_ID` routes. Prompt processing must retain its existing bulk placement.
- Benchmark TG separately from PP. A row with prompt and generation combined is not accepted as TG evidence.
- Use five alternating fresh-process control/enabled pairs for throughput claims, with fixed model, model checksum, prompt, seed, placement, 14 threads, Q8_0 KV cache, Flash Attention, mlock, `--fit-target 256`, and `--parallel 1`.
- Do not commit, push, or create a PR without explicit user approval. Append all accepted and rejected benchmark rows to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`.

---

## File Structure

- `ggml/src/ggml-backend.cpp` owns graph placement, split execution, route readiness, cache remapping, and restoration. It is the only production scheduler file modified by this plan.
- `src/llama-context.cpp` owns context scheduler configuration. It must not make nonzero cache capacity globally force op-level offload.
- `common/fit.cpp` owns auto-fit layer placement. It must preserve complete GPU MoE layers when cache capacity is inactive for the evaluated route.
- `tests/test-expert-cache.cpp` owns deterministic scheduler/cache correctness coverage, including stale-ID safety and route-ready GPU remapping.
- `tests/test-fit-debug.cpp` is the existing exact-preset initialization smoke harness. It records selected placement only; it is not used as a throughput test.
- `tools/llama-bench/llama-bench.cpp` already exports the required expert-cache counters. Do not add columns unless the route-ready decision cannot be derived from the existing `probe_sync_us`, `probe_host_us`, requests, hits, misses, and bytes fields.
- `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` is the permanent benchmark decision record.

## Task 1: Establish a Comparable TG Decision Baseline

**Files:**
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`
- Inspect: `G:/qwen3.6-35b-a3b-presets-exc-latest.ini`
- Inspect: `tools/llama-bench/llama-bench.cpp:1744-1790,1966-1984`

**Interfaces:**
- Consumes: the existing `llama-bench` expert-cache statistics fields and the Compact preset.
- Produces: a dated control matrix that later tasks use as the sole performance acceptance baseline.

- [ ] **Step 1: Record the immutable command configuration**

Write the model absolute path, model SHA-256, binary commit, `-t 14`, `-b 4096`, `-ub 2048`, Q8_0 K/V, Flash Attention, mlock, `-fitt 256`, prompt/seed, and the exact cache arguments into a new dated log section before running a row.

- [ ] **Step 2: Run the cache-off control five times**

Run five fresh processes with TG only:

```powershell
build/bin/Release/llama-bench.exe -m "<compact-model>" -p 0 -n 128 -r 1 -t 14 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -fa on -lm mlock -fitt 256 -exc 0 -o jsonl
```

Save each raw JSONL row under `tools/results/expert-cache/2026-08-28-route-ready-control-<N>.jsonl`.

- [ ] **Step 3: Run the cache-enabled placement control five times**

Alternate each cache-off process with this cache-enabled process. Do not enable `GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL` in this task:

```powershell
build/bin/Release/llama-bench.exe -m "<compact-model>" -p 0 -n 128 -r 1 -t 14 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -fa on -lm mlock -fitt 256 -exc 128 -excp 256 -o jsonl
```

Save each raw row under `tools/results/expert-cache/2026-08-28-route-ready-cache-<N>.jsonl`.

- [ ] **Step 4: Record the decision inputs**

For every row, record TG mean, median, standard deviation, `expert_cache_eligible_ops`, requests, zero-copy hits, misses, `probe_sync_us`, `probe_host_us`, rebalances, RAM-to-GPU bytes, and full GPU MoE layer count. Record the cache-off/cache-on output hash from a fixed greedy server completion separately.

- [ ] **Step 5: Apply the admission gate**

Proceed to Task 2 only when the cache-enabled row is slower than cache-off or reports zero eligible TG operations. If the enabled row already beats cache-off outside variance, record that result and stop this plan; the reported regression is not reproduced under controlled placement.

## Task 2: Define Route-Ready Scheduler Correctness Tests

**Files:**
- Modify: `tests/test-expert-cache.cpp:96-160,1314-1318`
- Inspect: `ggml/src/ggml-backend.cpp:878-894,1179-1252,2477-2626`

**Interfaces:**
- Consumes: `ggml_backend_sched_split_graph()`, `ggml_backend_sched_graph_compute_async()`, the existing expert-cache test fixture, and cache statistics.
- Produces: tests that distinguish an unsafe eager route read from a valid route-ready remap.

- [ ] **Step 1: Convert the stale-ID placement regression into an execution-order regression**

Rename `test_pending_route_ids_stay_on_cpu()` to `test_pending_route_ids_defer_cache_remap()`. Before graph execution, assert that a cache-managed `MUL_MAT_ID` with non-leaf route IDs has not had `src[0]` or `src[2]` changed. The test must then execute the producer prefix and assert that the cache remap happens only after that prefix completes.

- [ ] **Step 2: Write the failing route-ready integration test**

Add `test_route_ready_ids_use_gpu_slots()` beside the existing pending-ID test. Construct one CPU backend and one available non-CPU backend, create a route-ID tensor with a non-`GGML_OP_NONE` producer, register a complete Gate/Up/Down slot bundle, and execute one TG1 graph through the production scheduler.

The test must assert all of the following after graph compute:

```cpp
require(status == GGML_STATUS_SUCCESS);
require(output_matches_cpu_reference);
require(stats.n_zero_copy_hits > 0);
require(stats.n_requests > 0);
require(stats.n_bytes_ram_to_gpu == 0);
```

Use a separately constructed all-CPU `MUL_MAT_ID` graph as the numerical reference. The cache graph must reuse the original graph safely on a second compute call, proving that node restoration remains correct.

- [ ] **Step 3: Run the focused test before implementation**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: the new route-ready test fails because the current safety guard keeps the node on CPU and records no GPU slot hit. Existing tests, including stale-ID safety, pass.

- [ ] **Step 4: Add the deterministic CLI/server check definition**

Add a test comment immediately above the new test stating the externally verified contract: the same fixed prompt, temperature 0, top-k 1, seed 42, and `ignore_eos` must yield the same 256-token SHA-256 with `-exc 0` and route-ready cache enabled. Do not add a model-dependent unit test to `test-expert-cache`.

## Task 3: Execute Route Producers Before Cache Remapping

**Files:**
- Modify: `ggml/src/ggml-backend.cpp:878-894,1008-1024,2069-2635`
- Test: `tests/test-expert-cache.cpp`

**Interfaces:**
- Consumes: `ggml_backend_sched_route_bundle_plan`, `ggml_graph_view()`, `ggml_backend_graph_compute_async()`, `ggml_backend_synchronize()`, and existing slot-remap state in `remapped_ids_buf`.
- Produces: a private per-split `ggml_backend_sched_route_ready_action` that safely gates cache remapping on a completed route-ID producer.

- [ ] **Step 1: Introduce the private action record**

Add this private scheduler-local record next to `active_hetero_bundle` in `ggml_backend_sched_compute_splits()`:

```cpp
struct ggml_backend_sched_route_ready_action {
    int producer_node_idx;
    int first_consumer_node_idx;
    int last_consumer_node_idx;
    ggml_tensor * route_ids;
    std::vector<ggml_tensor *> cache_nodes;
};
```

One action represents one route-ID tensor and all complete-cache `MUL_MAT_ID` consumers that share it in one split. Do not create one action per Gate, Up, and Down projection.

- [ ] **Step 2: Build actions without reading IDs**

Replace the eager `cache_can_store && is_decode_route` route-ID D2H block with action discovery. For each cache-eligible decode node:

1. locate its `route_ids` producer index in `split->graph.nodes`;
2. locate every eligible consumer sharing the same `route_ids` pointer;
3. set `producer_node_idx` to the producer index and consumer bounds to the first/last sharing consumer index;
4. do not call `ggml_backend_tensor_get_async()`, `ggml_backend_synchronize()`, `find_slot()`, or mutate graph nodes during this discovery pass.

A leaf route-ID tensor has no producer index. It may use the existing immediate remap path because its values are ready before split execution.

- [ ] **Step 3: Re-admit cache candidates, but defer their GPU execution**

Remove `route_ids_pending` as a graph-placement veto for a registered cache tensor that is a single-token `GGML_OP_MUL_MAT_ID`. The graph may assign that candidate to the accelerator so gallocr can reserve its non-weight inputs, but it must not execute the original CUDA operation before a route-ready action has completed.

For every pending candidate, action discovery must either create one valid route-ready action or classify the candidate as unsupported. Before evaluating any graph view, intercept every unsupported pending candidate through the existing CPU fallback and save its original node for restoration. This gives the GPU graph no path that can consume stale IDs.

The runtime safety condition is:

```cpp
const bool route_ready = action.producer_node_idx >= 0 &&
    action.producer_node_idx < action.first_consumer_node_idx &&
    route_producer_view_has_completed;
```

No cache node may execute on GPU based solely on cache capacity. Continue requiring a registered cache tensor, `GGML_OP_MUL_MAT_ID`, and `route_ids->ne[1] == 1`.

- [ ] **Step 4: Execute prefix, then remap once**

In the existing graph-view execution path, run a graph view from the current node index through `producer_node_idx + 1`, then synchronize only `split_backend`. Mark that action's producer view complete, read `route_ids` once to host, build requested-expert counts once, and apply the existing slot lookup/remapped-ID upload to every node in `cache_nodes`.

Use the existing `save_node_for_restore()` for every mutated or CPU-intercepted consumer. Preserve `node->src[0]`, `node->src[2]`, and `node->op` restoration at the end of the split.

- [ ] **Step 5: Keep incomplete routes safe**

For a complete slot hit, rewrite each consumer to its corresponding slot tensor and remapped IDs, then continue executing the graph after the producer prefix.

For incomplete slots, execute the current CPU fallback before the consumer graph view and set the original GPU node to `GGML_OP_NONE` through `save_node_for_restore()`. Do not upload missing expert weights. Do not execute a prefix twice.

- [ ] **Step 6: Run the focused correctness target**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: stale-ID safety, route-ready GPU slot execution, duplicate routes, shared IDs, partial-hit scatter, and second-compute graph reuse all pass.

## Task 4: Collapse Cache Work to One FFN Bundle Action

**Files:**
- Modify: `ggml/src/ggml-backend.cpp:2127-2475`
- Test: `tests/test-expert-cache.cpp`

**Interfaces:**
- Consumes: route-ready actions from Task 3 and `ggml_backend_sched_route_bundle_plan` fields `gate_node`, `up_node`, `gate_up_node`, and `down_node`.
- Produces: exactly one host route read and one hit/miss classification per route-ID tensor per layer.

- [ ] **Step 1: Write the failing shared-ID accounting test**

Add `test_route_ready_bundle_reads_shared_ids_once()`. Create a non-leaf route-ID producer shared by separate Gate, Up, and Down `MUL_MAT_ID` nodes. After one scheduler compute, assert the cache reports one route-ready request group and three remapped projections, not three independent route reads.

Use a scheduler-local counter only if existing `n_route_census_split_inputs` and request counters cannot distinguish one ID read from three. If a new counter is required, add `n_route_ready_actions` to `ggml_backend_expert_cache_stats`, update its scheduler aggregation, `llama-bench` field/type/value lists, and the subtraction helper together.

- [ ] **Step 2: Make the action own all same-ID cache nodes**

Move requested-expert counting, access-frequency recording, slot lookup, and remapped-ID vector creation out of the per-node loop. Run them once per action. Apply the result to the three registered tensor slot pools while retaining distinct Gate, Up, and Down slot indices.

- [ ] **Step 3: Preserve complete-bundle admission**

A GPU action is a hit only when every projection required by the FFN bundle has a ready slot for every selected expert. If one projection is unavailable, do not remap a subset of that bundle through the normal slot path.

- [ ] **Step 4: Run the focused test**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: one route-ready action per shared-ID bundle, all prior cache tests pass, and no test changes output after a second graph compute.

## Task 5: Decouple Cache Capacity From Global Placement

**Files:**
- Modify: `src/llama-context.cpp:280`
- Modify: `common/fit.cpp:745-747`
- Modify: `tests/test-fit-debug.cpp:101-122`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: explicit route-ready candidate placement from Task 3 and the existing VRAM target accounting in `common/fit.cpp`.
- Produces: cache-off-equivalent general op placement, while allowing auto-fit to use free VRAM for complete or partial model layers.

- [ ] **Step 1: Write the placement smoke condition**

In `test-fit-debug.cpp`, run context initialization twice with otherwise identical Compact parameters:

```cpp
params.expert_cache_size = 0;
```

and:

```cpp
params.expert_cache_size = 128ULL * 1024 * 1024;
params.pinned_experts_manifest.clear();
```

Print the resolved layer-placement summary for both runs. The acceptance condition is that enabling unused cache capacity does not reduce the count of complete GPU-resident MoE layers unless the 128 MiB reservation alone makes the previous placement physically impossible.

- [ ] **Step 2: Remove cache capacity as an unconditional op-offload reason**

Change context configuration to:

```cpp
cparams.op_offload = params.op_offload;
```

The route-ready cache candidate logic from Task 3 remains the only cache-specific reason a decode `MUL_MAT_ID` can be placed on the accelerator.

- [ ] **Step 3: Let auto-fit use remaining VRAM**

Replace:

```cpp
const bool allow_split_moe_layers = (cparams == nullptr || cparams->expert_cache_size == 0);
```

with:

```cpp
const bool allow_split_moe_layers = true;
```

The fit allocator already deducts expert-cache reservation from the device target. Layer splitting must therefore be governed by available bytes, not by cache capacity. Complete cache bundle atomicity remains a runtime scheduler invariant from Task 4; it is not an auto-fit placement rule.

- [ ] **Step 4: Run initialization and cache tests**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache test-fit-debug llama-bench llama-server
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-fit-debug.exe
```

Expected: both contexts initialize, cache tests pass, cache-enabled no-manifest placement either matches the cache-off full-layer count or accounts for an unavoidable 128 MiB capacity reduction, and no cache node reaches CUDA before route readiness.

## Task 6: Validate Performance and Determinism

**Files:**
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`
- Create: `tools/results/expert-cache/2026-08-28-route-ready-*.jsonl`

**Interfaces:**
- Consumes: the Task 1 controls and route-ready implementation from Tasks 3-5.
- Produces: retain, revise, or reject evidence for route-ready cache execution.

- [ ] **Step 1: Re-run the Task 1 alternating matrix**

Use the same command, model checksum, fresh-process ordering, and output location convention. Add a fifth row with route-ready cache enabled. Keep heterogeneous execution disabled for this matrix.

- [ ] **Step 2: Run deterministic server completions**

For cache-off and route-ready cache-on, use temperature 0, top-k 1, seed 42, `ignore_eos`, one completion per fresh server process, and 256 generated tokens. Hash the complete emitted token sequence. Reject the implementation if hashes differ, output becomes incoherent, or CUDA reports an illegal access.

- [ ] **Step 3: Apply the throughput acceptance rule**

Retain route-ready cache execution only if all are true:

```text
cache-on output hash == cache-off output hash
cache-on TG requests > 0
cache-on eligible TG operations > 0
cache-on RAM-to-GPU expert-weight bytes == 0
cache-on TG median > cache-off TG median beyond the paired-run variance
```

If the implementation is correct but does not beat cache-off placement, retain the safety tests and record the route-ready path as rejected for this GTX 1080 deployment regime. Do not continue to heterogeneous executor allocation reuse or rebalancer tuning in this plan.

- [ ] **Step 4: Append the complete decision record**

Append raw-row paths, exact command lines, placement summaries, medians, means, standard deviations, all cache telemetry, deterministic hashes, and the retain/revise/reject decision to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`.

## Verification Matrix

| Contract | Verification |
|---|---|
| No stale route IDs reach CUDA | `test_pending_route_ids_defer_cache_remap()` plus deterministic cache-enabled server completion |
| Route-ready full hit uses GPU slots | `test_route_ready_ids_use_gpu_slots()` and nonzero zero-copy hits |
| Gate/Up/Down share one route read | `test_route_ready_bundle_reads_shared_ids_once()` |
| Graph reuse is safe | Second compute in `test_route_ready_ids_use_gpu_slots()` |
| Missing experts do not upload weights | TG telemetry reports zero expert RAM-to-GPU bytes |
| Inactive cache does not harm placement | `test-fit-debug.exe` placement summaries and Task 1 controls |
| Deployment benefit is real | Five alternating fresh-process TG-only pairs and deterministic output hashes |
