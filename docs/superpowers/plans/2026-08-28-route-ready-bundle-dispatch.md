# Route-Ready Bundle Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Admit GPU expert-slot execution only for complete, route-ready MoE bundles while preserving the exact cache-off greedy token sequence for every cache miss and unsupported layout.

**Architecture:** Treat a route-ID producer plus its Gate/Up/Down projections as one scheduler dispatch unit, even when its nodes occupy different scheduler splits. The dispatch unit reads resolved route IDs once after its producer completes, validates every projection slot before mutating any node, and otherwise preserves the cache-off GPU graph path. It does not use CPU fallback merely because cache capacity is configured.

**Tech Stack:** C++17, ggml scheduler, CUDA backend, `test-expert-cache`, `test-fit-debug`, `llama-bench`, `llama-server`, Python 3, Windows PowerShell.

## Global Constraints

- Do not commit, push, or create a PR without explicit user approval.
- Do not add a CUDA kernel, a worker thread, or a new background subsystem.
- No route-ID host read, slot lookup, or graph mutation may happen before the route-ID producer has completed on its backend.
- Cache-off and cache-on must emit the identical fixed greedy 256-token sequence before cache-on performance can be retained.
- A complete bundle hit requires a resident slot for every selected expert in every required Gate/Up/Down projection.
- A partial bundle hit must not remap any projection through the normal GPU slot path.
- A cache miss must not upload host expert weights during TG. It must preserve the cache-off execution backend unless a separately verified model-specific fallback is bitwise equivalent.
- Prompt processing retains its existing scheduler placement and bulk behavior.
- Use `tools/results/expert-cache/run-tg-matrix.py` for repeated fresh-process TG matrices. It must set `GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL=0`.
- Record every accepted or rejected experiment in `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`; update `EXPERT_CACHE.md` only for behavior that passes the gates below.

---

## Current Evidence and Decision Gate

Current evidence, captured on the Compact model with `GGML_SCHED_DEBUG=2`, shows a route producer followed by `ffn_moe_gate`, `ffn_moe_up`, and `ffn_moe_down` in the original graph, but the current per-split action receives one projection while the bundle plan requires three. The current cache-on server completion hash differs from cache-off:

```text
cache off: f7aeb2f94774622d2dda6c5b38ca3e617d6b92843952c2ec8620d3a0402d0177
cache on : 23d89910b54ffc245b9c82dae8913665df877e7506a2ee3e6107ca2c77d74b5c
```

The current implementation is therefore rejected as a deployment candidate. The tasks below define the evidence required before each implementation change. If a condition is false, record the result and take the stated fallback path instead of continuing implementation.

## File Structure

- `ggml/src/ggml-backend.cpp` owns graph placement, split execution, route bundle discovery, route-ready dispatch state, cache remapping, and graph restoration.
- `tests/test-expert-cache.cpp` owns synthetic scheduler correctness tests for stale IDs, cross-split bundle grouping, all-hit remapping, misses, and graph reuse.
- `tests/test-fit-debug.cpp` owns the exact-preset initialization and placement smoke harness.
- `tools/llama-bench/llama-bench.cpp` exports scheduler/cache telemetry. Add a field only when existing counters cannot prove one route read and three projection remaps.
- `tools/results/expert-cache/run-tg-matrix.py` owns repeatable fresh-process TG measurement and raw JSONL retention.
- `EXPERT_CACHE.md` describes only accepted cache scheduling behavior.
- `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` records the evidence and disposition of every gate.

---

### Task 1: Prove the Production Bundle Topology

**Files:**
- Inspect: `ggml/src/ggml-backend.cpp:1176-1248,2169-2514,2685-2814`
- Inspect: `tools/llama-bench/llama-bench.cpp:2479-2718`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: `ggml_backend_sched_discover_route_plans()`, `ggml_backend_sched_split_graph()`, and `GGML_SCHED_DEBUG=2` assignment output.
- Produces: a recorded map of each Compact decode bundle's route producer and its Gate/Up/Down split locations.

- [ ] **Step 1: Capture one cache-enabled TG1 scheduler trace without changing source**

The full scheduler assignment dump is too large for an interactive terminal. Redirect both streams to retained files, then extract only the route nodes:

```powershell
$trace = "tools/results/expert-cache/2026-08-28-route-topology.stdout.log"
$traceErr = "tools/results/expert-cache/2026-08-28-route-topology.stderr.log"
$env:GGML_SCHED_DEBUG = "2"
$env:GGML_LOG_LEVEL = "5"
$env:GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL = "0"
& build/bin/Release/llama-bench.exe -m "<compact-model>" -p 0 -n 1 -r 1 --no-warmup -t 14 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -fa on -lm mlock -fitt 256 -exc 128 -excp 256 -o jsonl 1> $trace 2> $traceErr
if ($LASTEXITCODE -ne 0) { throw "route topology trace failed: $LASTEXITCODE" }
Get-Content $trace, $traceErr | Select-String -Pattern "ffn_moe_(topk|gate-|up-|swiglu|down-)"
```

Retain both raw logs and the filtered output path in the optimization log. For one representative layer, record the graph indices, assigned backend, and scheduler split number for:

```text
ffn_moe_topk-<layer>
ffn_moe_gate-<layer>
ffn_moe_up-<layer>
ffn_moe_swiglu-<layer>
ffn_moe_down-<layer>
```

- [ ] **Step 2: Apply the topology gate**

Proceed with Task 2 only if the trace proves one of these two layouts:

```text
A. Every Gate/Up/Down consumer for a route ID is in one split.
B. The consumers span two or more splits, but all are identifiable from one canonical route-ID pointer and the scheduler can order their splits.
```

If neither is true because the route IDs or projection nodes are copied into identities the route census cannot relate, do not implement dispatch. First write a focused plan to normalize those graph identities from scheduler copy metadata.

- [ ] **Step 3: Record the chosen implementation branch**

Append one explicit statement to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`:

```text
Topology branch A: use a complete per-split bundle action.
```

or:

```text
Topology branch B: use a cross-split route bundle dispatch record.
```

Current evidence indicates branch B. Do not continue with per-projection action recovery if the trace again shows one projection per split.

### Task 2: Establish the Cache-Off Equivalent Miss Path

**Files:**
- Modify: `tests/test-expert-cache.cpp`
- Inspect: `ggml/src/ggml-backend.cpp:2255-2514,2516-2814`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: existing `ggml_backend_sched_route_bundle_plan`, `save_node_for_restore()`, and normal split execution.
- Produces: a tested rule that a cache miss leaves the normal cache-off graph execution path intact.

- [ ] **Step 1: Add a failing miss-path scheduler test**

Add `test_route_ready_bundle_miss_preserves_normal_execution()` beside `test_route_ready_ids_use_gpu_slots()`.

Build the existing non-leaf route-ID Gate/Up/Down fixture, but do not seed any projection slots. Make a separate all-CPU graph with the same tensors as the numerical reference. Assert:

```cpp
require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
require(output_matches_cpu_reference);
require(stats.n_zero_copy_hits == 0);
require(stats.bytes_ram_to_gpu == 0);
require(original_gate_src0_is_restored);
require(original_up_src0_is_restored);
require(original_down_src0_is_restored);
```

The test must not assert CPU fallback. Its contract is that a miss cannot leave remapped IDs, slot tensors, or disabled graph nodes behind.

- [ ] **Step 2: Run the new test before changing production code**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected before the fix: the test fails because an incomplete bundle is routed through a CPU-intercept or partial-action path rather than the original unmutated graph path.

- [ ] **Step 3: Apply the miss-equivalence gate**

Only replace the current CPU-miss interception if both checks are true:

```text
- The cache-off GPU graph executes host-backed expert weights without expert RAM -> GPU uploads.
- A fresh cache-on server completion differs from cache-off only when the CPU-miss interception is enabled.
```

Prove the first condition from telemetry and the second by running the same fixed 256-token server request with the interception disabled in a local diagnostic build.

If either condition is false, do not assume the original GPU graph is an equivalent fallback. Record the result, restore cache-off placement for cache-on capacity, and stop this recovery plan pending a model-specific numerical-equivalence design.

- [ ] **Step 4: Implement the minimal miss behavior after the gate passes**

In the route-ready evaluation path, replace the incomplete-slot result with an unmutated normal execution result:

```cpp
if (!complete_bundle_hit) {
    // Do not alter src[0], src[2], or op. The normal split executor runs this bundle.
    dispatch.miss = true;
    return route_dispatch_result::normal_graph;
}
```

Do not call `ggml_backend_moe_hetero_execute_serial()` for this result. Do not mark projection nodes `GGML_OP_NONE`.

- [ ] **Step 5: Re-run the focused scheduler test**

Run the Task 2 command again. Expected: PASS, with no zero-copy hits, no expert upload bytes, and restored original node sources.

### Task 3: Build One Route Dispatch Per Complete Bundle

**Files:**
- Modify: `ggml/src/ggml-backend.cpp:878-929,1176-1248,2169-2814`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `ggml/include/ggml-backend.h` only if existing counters cannot prove one dispatch per route ID
- Modify: `tools/llama-bench/llama-bench.cpp` only if the header counter is added

**Interfaces:**
- Consumes: `ggml_backend_sched_route_bundle_plan`, `ggml_graph_view()`, `ggml_backend_graph_compute_async()`, `ggml_backend_synchronize()`, `remapped_ids_buf`, and `save_node_for_restore()`.
- Produces: `ggml_backend_sched_route_bundle_dispatch`, private to the scheduler, with one canonical route-ID pointer and every projection location required by its bundle.

- [ ] **Step 1: Add the failing cross-split bundle accounting test**

Add `test_route_ready_bundle_reads_shared_ids_once()`.

Create a non-leaf route-ID producer shared by Gate, Up, and Down `GGML_OP_MUL_MAT_ID` nodes. Force scheduler split boundaries between at least two projections using explicit backend placement and a cross-backend dependency. Seed all selected experts in all three projection pools. After one compute, assert:

```cpp
require(output_matches_cpu_reference);
require(stats.n_zero_copy_hits == 6);
require(stats.n_route_ready_actions == 1);
require(stats.n_requests == 6);
require(stats.bytes_ram_to_gpu == 0);
```

Run the graph a second time and assert the same output. The test fails if three per-projection actions read and classify the same route IDs independently.

- [ ] **Step 2: Add a private cross-split dispatch record**

Only implement this step if Task 1 selected branch B. Add a private scheduler-local record near `active_hetero_bundle`:

```cpp
struct ggml_backend_sched_route_bundle_dispatch {
    const ggml_backend_sched_route_bundle_plan * bundle = nullptr;
    ggml_tensor * route_ids = nullptr;
    int producer_split = -1;
    int producer_node_idx = -1;
    std::vector<std::pair<int, int>> consumer_locations;
    std::vector<int32_t> ids;
    bool producer_complete = false;
    bool complete_bundle_hit = false;
    bool classified = false;
};
```

Populate it by scanning every scheduler split after `ggml_backend_sched_split_graph()` has produced the split list. Resolve consumer locations by pointer equality against `bundle->gate_node`, `bundle->up_node`, `bundle->gate_up_node`, and `bundle->down_node`.

If Task 1 selected branch A, retain the same record without `consumer_locations`; require every bundle projection in the action's single split before creation.

- [ ] **Step 3: Apply the complete-bundle admission gate**

Create a dispatch only when all required conditions hold:

```cpp
const bool has_producer = dispatch.producer_split >= 0 &&
    dispatch.producer_node_idx >= 0;
const size_t expected_consumers = dispatch.bundle->is_fused ? 2 : 3;
const bool has_full_bundle = dispatch.consumer_locations.size() == expected_consumers;
const bool is_tg1 = dispatch.route_ids->ne[1] == 1;
```

Unsupported pending candidates must stay unmutated on the normal graph path. Do not create an action for a subset of a bundle.

- [ ] **Step 4: Classify a ready route once**

When the producer split reaches `producer_node_idx + 1`, execute the prefix graph view, synchronize only that split backend, then perform this sequence exactly once:

```cpp
ggml_backend_tensor_get(dispatch.route_ids, dispatch.ids.data(), 0, ids_bytes);
collect_unique_requested_experts(dispatch.ids, n_expert, requested_experts);
dispatch.complete_bundle_hit = all_bundle_slots_resident(
    cache, dispatch.bundle, requested_experts);
dispatch.classified = true;
```

`all_bundle_slots_resident()` must check each selected expert against Gate, Up, and Down slots before mutating any consumer. It must not allocate, seed, upload, or rebalance cache entries.

- [ ] **Step 5: Remap only after complete admission**

Only if `dispatch.complete_bundle_hit` is true, remap each projection at its scheduled execution point:

```cpp
save_node_for_restore(node);
node->src[0] = projection_slot_tensor;
node->src[2] = upload_remapped_ids_once_per_projection(
    split_backend, dispatch.ids, projection_slot_indices);
```

Increment `n_route_ready_actions` once after the full bundle is classified as a hit. Record zero-copy hits per unique expert per projection. Preserve existing restoration of `src[0]` and `src[2]` after compute.

- [ ] **Step 6: Run the focused route bundle tests**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: stale-ID deferral, all-hit remapping, shared-ID one-action accounting, miss behavior, partial-hit scatter, and graph reuse pass.

### Task 4: Restore Placement Equivalence for Inactive Capacity

**Files:**
- Modify: `ggml/src/ggml-backend.cpp:1008-1021,1302-1709`
- Modify: `src/llama-context.cpp:280`
- Modify: `common/fit.cpp:745-747`
- Modify: `tests/test-fit-debug.cpp`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: route bundle dispatch support from Task 3 and `common_init_from_params()` placement output.
- Produces: cache capacity that does not alter general placement until a route-ready complete bundle is admitted.

- [ ] **Step 1: Add the placement assertion to the smoke harness**

Extend `test-fit-debug.cpp` to initialize the Compact preset twice, once with:

```cpp
params.expert_cache_size = 0;
```

and once with:

```cpp
params.expert_cache_size = 128ULL * 1024 * 1024;
params.pinned_experts_manifest.clear();
```

For both runs, print and capture the count of complete GPU-resident MoE layers and the resolved host/GPU locations for one representative Gate/Up/Down bundle.

- [ ] **Step 2: Apply the placement gate**

Only retain route-specific accelerator placement if the cache-enabled no-manifest configuration has both properties:

```text
- Complete GPU-resident MoE layer count matches cache-off, except for a documented 128 MiB physical reservation limit.
- An unsupported or not-yet-classified route bundle does not move a host expert tensor or change its normal execution backend.
```

If either property fails, restore the `route_ids_pending` placement guard for unsupported bundles. Do not use `ggml_backend_expert_cache_has_tensor()` alone as proof of a cache hit; it means registration, not residency.

- [ ] **Step 3: Keep existing global placement decoupling**

Retain these changes only after Step 2 passes:

```cpp
cparams.op_offload = params.op_offload;
const bool allow_split_moe_layers = true;
```

If the gate fails because these two changes expose a separate fit regression, revert only the failed change and record the exact placement evidence. Do not reintroduce cache capacity as a global `op_offload` reason.

- [ ] **Step 4: Build and run placement checks**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache test-fit-debug llama-bench llama-server
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-fit-debug.exe
```

Expected: tests pass and the recorded placement gate passes. If placement differs without a physical-reservation explanation, stop before Task 5.

### Task 5: Validate Determinism Before Throughput

**Files:**
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`
- Create: `tools/results/expert-cache/2026-08-28-route-ready-dispatch-*.jsonl`
- Create: `tools/results/expert-cache/2026-08-28-route-ready-dispatch-*.json`

**Interfaces:**
- Consumes: the Task 3 dispatch and Task 4 placement result.
- Produces: a deterministic cache-on/cache-off disposition before any performance claim.

- [ ] **Step 1: Run fresh deterministic control and cache servers**

For each server process, set:

```text
GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL=0
```

Use identical Compact model, `-t 14`, Q8_0 K/V, Flash Attention, mlock, `-fitt 256`, and a 4,096-token context. Submit the same `/completion` body to each fresh server:

```json
{
  "prompt": "Continue this deterministic benchmark record with concise cache-locality facts:\\n1.",
  "n_predict": 256,
  "temperature": 0,
  "top_k": 1,
  "seed": 42,
  "ignore_eos": true,
  "return_tokens": true,
  "stream": false
}
```

Save both raw responses. Hash `tokens` using compact JSON serialization:

```python
hashlib.sha256(json.dumps(response["tokens"], separators=(",", ":")).encode()).hexdigest()
```

- [ ] **Step 2: Apply the determinism gate**

Proceed to Task 6 only if all are true:

```text
cache-on token count == 256
cache-off token count == 256
cache-on SHA-256 == cache-off SHA-256
no CUDA illegal access or scheduler failure occurred
```

If the hashes differ, do not benchmark throughput. Restore the last known cache-off-equivalent miss path, record both hashes and stop implementation until a focused numerical-path test identifies the first divergent token.

- [ ] **Step 3: Add the final deterministic result to documentation**

Only if the gate passes, replace the current rejected deterministic note in `EXPERT_CACHE.md` with the exact verified command shape and hash contract. Always append raw response paths and hashes to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`.

### Task 6: Measure Cache-Hit Throughput Only When Residency Is Possible

**Files:**
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`
- Use: `tools/results/expert-cache/run-tg-matrix.py`

**Interfaces:**
- Consumes: deterministic cache-on behavior from Task 5 and existing cache telemetry.
- Produces: retain or reject evidence for this GTX 1080 deployment regime.

- [ ] **Step 1: Prove the chosen measurement can produce cache residency**

Before a five-pair matrix, run one diagnostic TG measurement and inspect:

```text
expert_cache_route_ready_actions > 0
expert_cache_requests > 0
expert_cache_zero_copy_hits > 0
```

The existing no-manifest `-n 128 -excp 256` configuration is not sufficient by itself because it does not reach the rebalance period. Use one of these explicitly isolated methods:

```text
A. A longer non-timed warmup that crosses the 256-token rebalance boundary, then a separate timed TG128 measurement.
B. A controlled test-only slot seed that creates a complete bundle before timed decode.
C. A separate experiment with a shorter rebalance period, recorded as a different configuration.
```

Do not use a pinned manifest in the no-manifest parity matrix.

- [ ] **Step 2: Apply the residency gate**

Only run the five-pair throughput matrix if the diagnostic has positive route-ready actions, requests, and zero-copy hits with zero expert RAM-to-GPU bytes.

If not, record that the configuration cannot exercise cache hits. Retain safety tests but reject performance tuning for this matrix; do not interpret cache capacity overhead as a cache-hit result.

- [ ] **Step 3: Run the alternating TG matrix after the gate passes**

Run:

```powershell
python tools/results/expert-cache/run-tg-matrix.py --model "<compact-model>" --prefix "2026-08-28-route-ready-dispatch" --runs 5 --cache-mib 128 --cache-period 256
```

Capture control/cache TG-only rows, preserving the tool's raw JSONL files. Calculate mean, median, sample standard deviation, paired differences, eligible ops, requests, zero-copy hits, misses, route-ready actions, probe times, rebalances, and expert RAM-to-GPU bytes.

- [ ] **Step 4: Apply the deployment acceptance gate**

Retain the route-ready path for this GTX 1080 only if every condition is true:

```text
cache-on deterministic token hash == cache-off hash
cache-on TG requests > 0
cache-on TG eligible operations > 0
cache-on zero-copy hits > 0
cache-on expert RAM-to-GPU bytes == 0
cache-on median TG throughput exceeds cache-off beyond paired-run variance
```

If correctness passes but throughput does not, retain only the safety and determinism tests and record the route-ready performance path as rejected for this hardware. Do not proceed to rebalancer tuning, heterogeneous executor reuse, or new cache allocation work.

- [ ] **Step 5: Write the complete disposition record**

Append model and binary SHA-256, command lines, environment, route topology branch, placement summaries, raw artifact paths, token hashes, all matrix rows, calculated statistics, and one final decision:

```text
retain
```

```text
revise
```

or:

```text
reject for GTX 1080 deployment
```

## Verification Matrix

| Contract | Evidence | Stop condition |
| --- | --- | --- |
| Route IDs are not stale | Route producer prefix completes before D2H in focused tests | Any CUDA error or read before producer completion |
| One route read covers a full bundle | Cross-split shared-ID test reports one action and three remaps | A split sees only a partial bundle and attempts remap |
| Misses preserve cache-off semantics | Miss-path test plus equal server token hashes | CPU interception or altered normal node state changes tokens |
| Inactive capacity preserves placement | `test-fit-debug.exe` and no-manifest placement log | Cache capacity alone changes unsupported bundle placement |
| Timed cache hit is real | Positive actions, requests, and zero-copy hits before matrix | No-manifest TG128 never reaches residency |
| Deployment benefit is real | Five alternating fresh-process pairs | Median improvement is inside paired variance or negative |
