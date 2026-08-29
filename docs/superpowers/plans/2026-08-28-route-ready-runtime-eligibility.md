# Route-Ready Runtime Eligibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine why the completed route-ready dispatcher does not admit a real Compact TG bundle, then enable learned slot residency and measure it without weakening stale-ID safety, miss equivalence, or cache-off deterministic output.

**Architecture:** First prove the dispatcher reaches and classifies a real bundle with low-volume counters, rather than infer topology from a full scheduler trace. Only after classification is proven, record route access frequency for every projection in a complete bundle so the existing periodic rebalancer can populate slots. A new `llama-bench` warmup option may be added only if the dispatcher records access but the TG128 measurement cannot cross its configured rebalance period.

**Tech Stack:** C++17, ggml scheduler and expert cache, `test-expert-cache`, `llama-bench`, `llama-server`, Python 3, Windows PowerShell.

## Global Constraints

- Do not commit, push, create a branch, or create a PR without explicit user approval.
- Do not add CUDA kernels, threads, background services, or a second scheduler.
- Preserve the completed route-ready dispatcher and its current invariants: producer prefix completes before route-ID read; all Gate/Up/Down slots must be ready before any remap; incomplete bundles leave the normal graph untouched.
- Do not re-enable the CPU route fallback for an incomplete route-ready bundle.
- Do not use `ggml_backend_expert_cache_has_tensor()` as a residency test. It proves only bundle registration.
- Keep `GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL=0` in all route-ready acceptance experiments.
- A cache-on result is deployable only if its fixed greedy 256-token sequence matches cache-off exactly and expert RAM-to-GPU weight bytes during the timed interval are zero.
- Use `tools/results/expert-cache/run-tg-matrix.py` for matrix execution. Extend it instead of writing an inline benchmark script.
- Update `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` after every conditional gate. Update `EXPERT_CACHE.md` only after a gate passes.

## Current Evidence

- The handover verifies the route-ready dispatcher, complete-bundle admission, normal miss behavior, non-allocating slot lookup, source restoration, focused cache tests, build targets, and placement smoke test.
- The no-manifest Compact TG128 baseline still has zero eligible operations and zero requests. No real-model cache-hit or deterministic-server claim is currently valid.
- `ggml_backend_sched_route_bundle_dispatch` construction requires a valid TG1 bundle, a producer node in a split, and every expected projection location.
- `classify_route_dispatch()` reads route IDs and checks slots but currently does not call `ggml_backend_expert_cache_record_access_count()`. Therefore a classified miss cannot teach the existing periodic rebalancer which Gate/Up/Down slots to load.
- `llama-bench` snapshots expert-cache statistics after its one-token warmup and before the timed generation loop. It has no option to perform a same-process untimed decode warmup long enough to cross `-excp 256` before TG128 measurement.

---

### Task 1: Close the Producer-Prefix Execution Edge Case

**Files:**
- Modify: `tests/test-expert-cache.cpp:160-310`
- Modify: `ggml/src/ggml-backend.cpp:2867-2930` only if the new regression fails
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: `ggml_backend_sched_route_bundle_dispatch`, `ggml_graph_view()`, `ggml_backend_graph_compute_async()`, and `ggml_backend_synchronize()`.
- Produces: a regression proving the producer executes even when it is the first unexecuted node in its split.

- [ ] **Step 1: Add the failing producer-at-cursor regression**

Add `test_route_ready_producer_at_split_cursor()` next to `test_route_ready_ids_use_gpu_slots()`.

Reuse the existing CPU-reference Gate/Up/Down fixture. Arrange the graph so the route-ID producer is the first unexecuted node before a route-ready dispatch classification. Assert:

```cpp
require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
require(output_matches_cpu_reference);
require(stats.n_route_ready_actions == 1);
require(stats.n_zero_copy_hits == 6);
```

The route-ID producer must be non-leaf. The test must not prefill its output or make it an input tensor; otherwise it does not test prefix execution.

- [ ] **Step 2: Run the focused regression before changing scheduler code**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected before a fix, if this edge is reachable: the route IDs are read before their producer has run, producing a failed status, mismatched CPU reference, or no route-ready hit.

- [ ] **Step 3: Apply the cursor gate**

Only change the producer-view condition if Step 2 fails and the failure is isolated to `producer_node_idx == cur_j`.

The required behavior is to execute the producer-inclusive graph view for both cases:

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
ggml_backend_synchronize(split_backend);
classify_route_dispatch(*dispatch);
```

If the test passes, do not modify this execution branch. Record that the equality case is already unreachable or safe in the tested graph topology.

- [ ] **Step 4: Re-run the full focused cache target**

Run the Task 1 command again. Expected: all cache tests pass, including stale-ID deferral, route-ready all-hit execution, graph reuse, and the new cursor regression.

### Task 2: Add Low-Volume Dispatch Admission Telemetry

**Files:**
- Modify: `ggml/include/ggml-backend.h:418-420`
- Modify: `ggml/src/ggml-backend.cpp:1792-1825,2235-2297,3419-3500`
- Modify: `tools/llama-bench/llama-bench.cpp` field, value, integer-type, and subtraction lists together
- Modify: `tests/test-expert-cache.cpp`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: the existing `ggml_backend_expert_cache_stats` aggregate and route bundle dispatch lifecycle.
- Produces: `n_route_ready_dispatches` and `n_route_ready_classifications` in every `llama-bench` JSONL row.

- [ ] **Step 1: Add failing telemetry assertions**

In the existing route-ready all-hit test, add:

```cpp
require(stats.n_route_ready_dispatches == 1);
require(stats.n_route_ready_classifications == 1);
require(stats.n_route_ready_actions == 1);
```

Add a normal-miss variant that has a dispatch and classification but no slots:

```cpp
require(stats.n_route_ready_dispatches == 1);
require(stats.n_route_ready_classifications == 1);
require(stats.n_route_ready_actions == 0);
```

- [ ] **Step 2: Run the test to confirm the counters do not yet exist**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: compilation failure because the fields are absent.

- [ ] **Step 3: Add exactly two scheduler-owned counters**

Append these fields adjacent to `n_route_ready_actions`:

```cpp
uint64_t n_route_ready_dispatches;
uint64_t n_route_ready_classifications;
```

Increment `n_route_ready_dispatches` only after `route_bundle_dispatches.push_back()` succeeds. Increment `n_route_ready_classifications` immediately after `classify_route_dispatch()` completes its one route-ID read and slot classification, whether the result is a complete hit or a normal miss.

Do not increment either field in the legacy per-split route path. These counters describe only complete-bundle dispatches.

- [ ] **Step 4: Export counters atomically**

Append both JSON fields to all four `llama-bench` structures together:

```cpp
get_fields()
get_values()
integer field type list
subtract_expert_cache_stats()
```

Use these exact output names:

```text
expert_cache_route_ready_dispatches
expert_cache_route_ready_classifications
```

Do not add dashboard output, logging, or per-node debug printing.

- [ ] **Step 5: Verify counters and aggregation**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache llama-bench
build/bin/Release/test-expert-cache.exe
```

Then run one Compact TG1 JSONL smoke row with `-exc 128 -excp 256` and confirm both fields appear in JSON, even when their values are zero.

### Task 3: Classify the Real-Model Eligibility Failure Before Changing Placement

**Files:**
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`
- Inspect: `ggml/src/ggml-backend.cpp:1021-1034,1792-1825,2235-2297,2405-2446`
- Inspect: `ggml/src/ggml-backend-expert-cache.cpp:695-725,915-937,1571-1585`

**Interfaces:**
- Consumes: Task 2 counters, existing route census fields, and a TG1/TG128 JSONL run.
- Produces: one evidence-backed cause category and the only permitted next implementation branch.

- [ ] **Step 1: Run one small real-model classification row**

Run this command with a fresh process and retain the JSONL row:

```powershell
$env:GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL = "0"
build/bin/Release/llama-bench.exe -m "<compact-model>" -p 0 -n 1 -r 1 --no-warmup -t 14 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -fa on -lm mlock -fitt 256 -exc 128 -excp 256 -o jsonl
```

Record these fields:

```text
expert_cache_route_census_plans
expert_cache_route_ready_dispatches
expert_cache_route_ready_classifications
expert_cache_route_ready_actions
expert_cache_route_census_cpu_host_nodes
expert_cache_route_census_non_cpu_host_nodes
expert_cache_route_census_non_host_nodes
expert_cache_eligible_ops
expert_cache_cpu_backend_bypasses
```

- [ ] **Step 2: Select exactly one evidence branch**

| Observed result | Cause category | Permitted next change |
| --- | --- | --- |
| `route_census_plans > 0`, `dispatches == 0` | split optimization or tensor-copy identity prevents locating the producer or all consumers | Add a focused pointer-normalization test. Only then normalize canonical route IDs through scheduler copy/view metadata. |
| `dispatches > 0`, `classifications == 0` | producer dispatch does not reach the classifier | Fix only the failed producer/split ordering proven by Task 1 or a new focused regression. |
| `classifications > 0`, `cpu_host_nodes == 0` | evaluated model placement leaves no cacheable CPU-host MoE weights | Do not alter cache dispatch. Start a separate model-placement investigation. |
| `classifications > 0`, `cpu_host_nodes > 0`, `eligible_ops == 0` | the relevant consumer split has no accelerator cache or the cache is not selected for the host weight | Add a focused backend-to-cache selection regression; do not change global `op_offload`. |
| `classifications > 0`, `eligible_ops > 0`, `actions == 0` | route classification works but no complete resident bundle exists | Proceed to Task 4. |
| `actions > 0` | real route-ready GPU execution is active | Skip Task 4 and proceed directly to determinism validation in Task 5. |

Append the selected branch and raw row path to the optimization log. Do not make two branches of code changes in one pass.

### Task 4: Teach the Existing Rebalancer From Route-Ready Misses

**Files:**
- Modify: `tests/test-expert-cache.cpp`
- Modify: `ggml/src/ggml-backend.cpp:2235-2297`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: `ggml_backend_expert_cache_record_access_count()`, the existing periodic `ggml_backend_expert_cache_begin_step()`, and full bundle classification.
- Produces: TG frequency entries for every requested expert in every projection of a classified, cacheable route-ready bundle.

- [ ] **Step 1: Write a failing learned-residency regression**

Add `test_route_ready_dispatch_learns_bundle_residency()`.

Use the route-ready Gate/Up/Down fixture with no seeded slots and a cache period of one step. Compute once to classify a miss. Compute again so the existing rebalancer runs before classification. Assert:

```cpp
require(first_output_matches_cpu_reference);
require(second_output_matches_cpu_reference);
require(second_stats.n_route_ready_classifications == 2);
require(second_stats.n_route_ready_actions == 1);
require(second_stats.n_zero_copy_hits == 6);
```

The first compute may take the normal graph path. The second compute must use a complete bundle hit. The test must not call seed, claim-slot, or rebalance APIs directly.

- [ ] **Step 2: Run the regression before implementation**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected before the fix: the second compute still has zero route-ready actions because `classify_route_dispatch()` records no route access frequency.

- [ ] **Step 3: Record access only after a safe route read**

Only if Task 3 selected the classified-but-no-resident-slots branch, add the following after requested experts have been collected and before `dispatch.classified = true`:

```cpp
for (const ggml_tensor * consumer : bundle_consumers) {
    ggml_backend_expert_cache_t consumer_cache = cache_for_consumer(consumer);
    if (consumer_cache == nullptr ||
        !ggml_backend_expert_cache_can_store(consumer_cache, consumer->src[0]->nb[2])) {
        continue;
    }
    for (int32_t expert_id : requested_experts) {
        ggml_backend_expert_cache_record_access_count(
            consumer_cache,
            consumer->src[0],
            expert_id,
            1,
            GGML_EXPERT_CACHE_PHASE_TG);
    }
}
```

`bundle_consumers` must contain Gate/Up/Down for separate bundles or GateUp/Down for fused bundles. Do not record access before the route producer synchronization. Do not allocate, claim, or upload a slot in this code path.

- [ ] **Step 4: Re-run focused cache tests**

Run the Task 4 command again. Expected: first miss is safe, the next periodic rebalance makes all projection slots resident, and the second compute reaches one full bundle action with six zero-copy hits.

### Task 5: Add a Same-Process Untimed TG Warmup Only If Needed

**Files:**
- Modify: `tools/llama-bench/llama-bench.cpp`
- Modify: `tools/results/expert-cache/run-tg-matrix.py`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: `test_gen()`, `llama_memory_clear()`, `get_expert_cache_stats()`, `subtract_expert_cache_stats()`, and the existing TG matrix runner.
- Produces: `--warmup-gen N`, which warms one loaded context before the timed interval and excludes warmup cache telemetry from reported row metrics.

- [ ] **Step 1: Apply the warmup necessity gate**

Implement this task only if all are true:

```text
Task 4 produces a full slot hit after a periodic rebalance.
The deployment configuration remains -excp 256.
A no-manifest TG128 fresh process still has zero route-ready actions because it cannot cross the rebalance period before timed measurement.
```

If `-excp 1` or a supported deployment preload mechanism already gives a full hit without timed expert uploads, do not add a benchmark option. Record that existing controls are sufficient.

- [ ] **Step 2: Write a parser regression**

Add parser coverage for:

```text
--warmup-gen 256
```

It must store `256` in `cmd_params::warmup_gen`, reject negative values, and preserve the current default of `0`.

- [ ] **Step 3: Add the option and execute it before timing**

Add this field to `cmd_params`:

```cpp
int32_t warmup_gen;
```

Add this option:

```cpp
{"--warmup-gen"}, "N", "untimed generation tokens before each measured run (default: 0)"
```

After standard one-token warmup and before `stats_before`, execute:

```cpp
if (params.warmup_gen > 0 && !test_gen(ctx, params.warmup_gen, t.n_threads)) {
    fprintf(stderr, "%s: error: failed to run generation warmup\n", __func__);
    llama_free(ctx);
    llama_model_free(lmodel);
    exit(1);
}
llama_memory_clear(llama_get_memory(ctx), false);
const ggml_backend_expert_cache_stats stats_before = get_expert_cache_stats(ctx);
```

This preserves learned cache slots while clearing sequence state and excludes warmup uploads, rebalances, and hit counters from the reported timed row through the existing subtract helper.

- [ ] **Step 4: Extend the reusable matrix runner**

Add optional `--warmup-gen N` to `run-tg-matrix.py`. When nonzero, append:

```python
["--warmup-gen", str(args.warmup_gen)]
```

to both cache-off and cache-on commands. Include the selected warmup value in the emitted command summary.

- [ ] **Step 5: Build and smoke test the warmup behavior**

Run:

```powershell
cmake --build build --config Release --target llama-bench
build/bin/Release/llama-bench.exe --help
```

Then run one cache-on TG128 row with:

```powershell
--warmup-gen 256 -exc 128 -excp 256
```

Proceed only if the timed JSONL interval has positive classifications, positive route-ready actions, positive zero-copy hits, and zero expert RAM-to-GPU weight bytes.

### Task 6: Validate Determinism and Deployment Throughput

**Files:**
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`
- Modify: `EXPERT_CACHE.md` only if every acceptance gate passes
- Create: `tools/results/expert-cache/2026-08-28-route-ready-runtime-*.jsonl`
- Create: `tools/results/expert-cache/2026-08-28-route-ready-runtime-*.json`

**Interfaces:**
- Consumes: proven real-model route-ready actions from Tasks 3-5.
- Produces: a retain, revise, or reject decision for the GTX 1080 Compact deployment.

- [ ] **Step 1: Run deterministic fresh-server completions**

Use fresh cache-off and cache-on servers with the same model, prompt, temperature 0, top-k 1, seed 42, `ignore_eos`, `return_tokens`, and `n_predict: 256`. Hash compact JSON token arrays:

```python
hashlib.sha256(json.dumps(response["tokens"], separators=(",", ":")).encode()).hexdigest()
```

Only continue if both runs return 256 tokens and hashes match exactly.

- [ ] **Step 2: Run five alternating fresh-process TG pairs**

Only after the deterministic gate passes, run:

```powershell
python tools/results/expert-cache/run-tg-matrix.py --model "<compact-model>" --prefix "2026-08-28-route-ready-runtime" --runs 5 --cache-mib 128 --cache-period 256 --warmup-gen 256
```

If Task 5 was not necessary, omit `--warmup-gen` and record why.

- [ ] **Step 3: Apply the deployment gate**

Retain route-ready runtime execution only if all conditions are true:

```text
cache-on token hash == cache-off token hash
cache-on route-ready classifications > 0
cache-on route-ready actions > 0
cache-on zero-copy hits > 0
cache-on expert RAM-to-GPU bytes == 0 during the timed interval
cache-on median TG tok/s exceeds cache-off beyond paired-run variance
```

If correctness passes but speed does not, retain the scheduler safety tests and record `reject for GTX 1080 deployment`. Do not tune rebalancing, prefetch, or slot allocation beyond this plan.

- [ ] **Step 4: Record final evidence**

Append exact commands, model and binary SHA-256, warmup configuration, raw paths, all new dispatch counters, existing eligibility counters, hashes, paired statistics, and the disposition to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`.

Update `EXPERT_CACHE.md` only for a retained behavior.

## Verification Matrix

| Contract | Evidence | Stop condition |
| --- | --- | --- |
| Producer is ready before D2H | Task 1 cursor regression | Producer-first route is read before its graph view executes |
| Dispatcher reaches real model | Task 2 counters plus Task 3 TG1 row | Dispatches or classifications remain zero without an evidence-backed cause |
| Misses train the existing cache | Task 4 period-one regression | Second compute cannot produce a complete hit after recorded access |
| Warmup is excluded from timed data | Task 5 stats snapshot/subtraction | Timed JSONL includes warmup upload or rebalance statistics |
| Cache-on semantics match cache-off | Task 6 fresh-server token hashes | Any token divergence, incoherent output, or CUDA fault |
| Runtime is useful | Task 6 five-pair TG matrix | No hit, timed upload, or median gain inside variance |
