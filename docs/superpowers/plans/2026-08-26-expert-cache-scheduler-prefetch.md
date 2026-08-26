# Expert Cache Scheduler Prefetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the expert cache safe to use asynchronously, establish trustworthy scheduler-level evidence, and evaluate a bounded carry-forward prefetch path without forcing host MoE decode operations onto CUDA.

**Architecture:** Preserve the current scheduler-owned, per-accelerator cache and its fixed-capacity slot pools. First make slot lifecycle publication event-backed and observable. Then move route observation across decode graph boundaries so prefetch has lead time, keeping the current reactive miss path authoritative. Admission is optional, deadline-aware, and fail-closed: a late or uncertain prefetch must leave resident cache state unchanged.

**Tech Stack:** C++17, GGML scheduler/backend APIs, CUDA backend events and async tensor copies, CMake/CTest, llama-bench, llama-server, Python deterministic token-hash harness.
> **Status (2026-08-26):** This is the dated single-token carry-forward and lifecycle plan. Retained and partial work remains useful history, but it does not define general decode-time route-aware dispatch. New decode performance work follows `docs/superpowers/specs/2026-08-26-general-decode-moe-dispatch-design.md` and `docs/superpowers/plans/2026-08-26-general-decode-moe-dispatch.md`.


## Global Constraints

- Do not force host-resident MoE `MUL_MAT_ID` decode work to CUDA. `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` records this as a rejected GTX 1080 path.
- Preserve output correctness: cache-enabled greedy decode must be token-hash-identical to the matching cache-disabled control when model placement is identical.
- Keep `ggml_backend_sched_compute_splits()` free of new synchronous route-ID reads or backend synchronizations on the decode compute stream.
- The normal reactive cache miss path remains the correctness fallback for every prefetch rejection, stale route, disabled option, unsupported graph, and full cache.
- Add no CUDA-only implementation to `ggml-base`. Backend-specific CUDA work belongs behind an existing backend dispatch boundary; delete dead CUDA-only cache code rather than retaining a second inactive path.
- Cache state must distinguish `EMPTY`, `LOADING`, and `RESIDENT`; only completed DMA may publish `RESIDENT`.
- Every source or behavior change requires: focused test result, deterministic server result, control-versus-enabled benchmark, and a dated append to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`, including rejected/no-effect results.
- Benchmark on the established GTX 1080 configuration: one fresh process per row, `--parallel 1`, fixed model placement, 14 threads, batch 4096, ubatch 2048, q8_0 K/V cache, Flash Attention, mlock, no-mmap, no-context-shift, `--fit on --fit-target 256`.
- For throughput claims, use at least five alternating fresh-process control/enabled pairs. Report all raw rows, median, mean, standard deviation, TG and PP separately, cache telemetry, and exact binary/model/options.
- Never commit, push, or create a PR as part of this work. The repository requires explicit human approval for each commit.

---

## Benchmark-and-log protocol used by every task

The task-specific logging step below is mandatory even if the implementation is removed. Append a new dated section at the end of `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`; never overwrite historical results.

Each section must contain:

1. hypothesis and exact source symbols changed;
2. build/test commands plus PASS/FAIL output;
3. exact `llama-server`/`llama-bench` binary, GGUF path, model SHA-256, GPU/driver, command arguments, cache profile, and fresh-process policy;
4. deterministic control/enabled token hashes from `scripts/expert-cache-determinism.py`;
5. all ten or more alternating throughput rows, summary statistics, PP/TG telemetry, DMA bytes, slot-state counters, and route/prefetch counters where applicable;
6. decision: retain, revise, revert, or reject; and the reason.

Use these fixed correctness commands after building the changed targets:

```powershell
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-expert-cache-profile.exe
python scripts/expert-cache-determinism.py --exc 0 --json-out tools/results/expert-cache/control.json
python scripts/expert-cache-determinism.py --exc 64M --excp 64 --json-out tools/results/expert-cache/enabled.json
```

The token hashes in the two JSON records must match before any throughput comparison. If a change needs an opt-in prefetch option, include that identical option only in the enabled row and log it verbatim. If model placement changes between rows, the comparison is invalid; record that fact and stop the comparison rather than claiming correctness or performance.

### Task 1: Establish scheduler-level cache correctness and observability

**Files:**
- Modify: `tests/test-expert-cache.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `ggml/include/ggml-backend.h`
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `tools/llama-bench/llama-bench.cpp`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: current `ggml_backend_expert_cache_stats`, `ggml_backend_sched_get_expert_cache_stats()`, and `ggml_backend_sched_compute_splits()`.
- Produces: complete benchmark-visible cache deltas and a scheduler integration fixture that distinguishes no-cache, cold-miss, warm-hit, mixed-route, and unsupported-route behavior.

- [ ] **Step 1: Write failing stats-aggregation and scheduler-fixture tests**

Add a synthetic scheduler fixture using the CPU backend and an enumerated non-CPU backend that executes an indirect `GGML_OP_MUL_MAT_ID` split; skip only when the build exposes no non-CPU backend. Assert the fixture can run cold, warm, all-hit, all-miss, and mixed-ID cases without changing output bytes. Add an aggregation assertion for every public cache counter used by this work: `n_staging_waits`, `probe_sync_us`, `probe_host_us`, `probe_upload_us`, slot-loading/resident counters added in Task 2, and later route-prefetch counters.

```cpp
static void test_expert_cache_scheduler_cold_warm_equivalence() {
    // Build one indirect MUL_MAT_ID graph with host expert weights.
    // Run it once with cache disabled, once cold, then once warm.
    // Assert the output bytes are equal and warm telemetry contains a hit.
}

static void test_expert_cache_stats_are_aggregated() {
    // Seed distinct non-zero fields in two scheduler cache instances.
    // Assert scheduler aggregation returns the exact component-wise sum.
}
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: the new scheduler fixture or counter assertions fail because the current test path is helper-level and not all public stats are aggregated.

- [ ] **Step 3: Implement the minimal test fixture and complete stats plumbing**

Keep the fixture in `tests/test-expert-cache.cpp`; do not introduce a model-dependent test. Add only the test backend behavior needed to exercise scheduler splitting/copy execution. In `ggml_backend_sched_get_expert_cache_stats()`, sum every field declared in `ggml_backend_expert_cache_stats`. In llama-bench, add matching fields to `get_fields()`, `get_field_type()`, `get_values()`, and `subtract_expert_cache_stats()` as one atomic schema change.

- [ ] **Step 4: Run focused verification**

```powershell
cmake --build build --config Release --target test-expert-cache llama-bench
build/bin/Release/test-expert-cache.exe
build/bin/Release/llama-bench.exe --help
```

Expected: cache fixture passes; every emitted cache telemetry key has a matching declared type and value.

- [ ] **Step 5: Run benchmark protocol and append log entry**

Run the global deterministic controls and five alternating fresh-process `llama-bench` control/enabled pairs for PP512/TG128. Append the full result section. This is a baseline/instrumentation entry; do not claim a performance gain from telemetry alone.

### Task 2: Make slot loading completion-aware and coalesce duplicate route requests

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.cpp`
- Modify: `ggml/src/ggml-backend-expert-cache.h`
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `ggml/include/ggml-backend.h`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: `ggml_expert_cache_slot_state`, `ggml_expert_cache_slot_entry`, `ggml_backend_tensor_set_async()`, slot remapping, and scheduler split execution.
- Produces: an internal completion-backed slot transition API and deduplicated requested-expert handling. `find_slot()` remains RESIDENT-only.

- [ ] **Step 1: Write failing lifecycle tests**

Add tests that request the same expert more than once in one route tensor and across two immediate route tensors. Assert one load is issued per unique key, a loading slot is not remapped as a hit before its completion event, a completed slot is remapped as a hit, and eviction waits for the last consumer event.

```cpp
static void test_expert_cache_loading_slot_is_not_a_hit() {
    // Allocate a slot, begin an async fill without completing its event.
    // Assert find_slot() returns -1 and the same key does not enqueue another fill.
}

static void test_expert_cache_duplicate_ids_issue_one_fill() {
    // Route [3, 3, 3, 7, 7].
    // Assert two unique fill attempts, never five.
}
```

- [ ] **Step 2: Run focused tests and confirm failure**

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: current code either reports repeated misses or cannot prove deferred publication/eviction ordering.

- [ ] **Step 3: Implement explicit ownership and transition rules**

Add a per-slot fill-completion event and generation number. Reserve a `LOADING` slot before submitting DMA. While loading, a duplicate key attaches to that reservation rather than submitting another DMA. Transition it to `RESIDENT` only after the completion event is observed on the consumer ordering path. Record a consumer event after the rewired node is scheduled; require it before eviction, reuse, cache synchronization, or free. Centralize transition logic in cache-private helpers rather than publishing raw state mutations at each call site.

Ensure direct host DMA, pinned staging DMA, periodic rebalance, JIT swaps, and reactive scheduler fills all use the same transition helper. If an async operation cannot be given a completion event on the active backend, retain the synchronous-safe fallback and record that fallback in telemetry.

- [ ] **Step 4: Run focused verification**

```powershell
cmake --build build --config Release --target test-expert-cache test-expert-cache-profile
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-expert-cache-profile.exe
```

Expected: duplicate fills are coalesced; no loading slot is used as resident; teardown and eviction tests pass.

- [ ] **Step 5: Run benchmark protocol and append log entry**

Run the global correctness controls plus alternating PP512/TG128 throughput pairs. Log DMA submissions, coalesced duplicate requests, loading-to-resident transitions, waits, eviction waits, raw measurements, and retain/reject decision.

### Task 3: Remove or complete inactive CUDA map code based on an executable contract

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.cpp`
- Modify: `ggml/src/ggml-backend-expert-cache.h`
- Modify: `ggml/include/ggml-backend.h`
- Modify: `ggml/src/gg-cuda/expert-cache-remap.cu`
- Modify: `ggml/src/gg-cuda/expert-cache-remap.cuh`
- Modify: `ggml/src/CMakeLists.txt`
- Modify: `ggml/src/gg-cuda/CMakeLists.txt`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `EXPERT_CACHE.md`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: the Task 2 event contract and the current explicit remapped-ID upload path.
- Produces: exactly one supported remapping implementation. The initial default is CPU remap plus explicit ID upload because it is the active, tested path.

- [ ] **Step 1: Write a failing build/behavior guard**

Add a test or compile-visible assertion that the active scheduler remap path is explicit-ID upload and that no public GPU-map function is declared without a scheduler caller. Build the existing expert-cache test target with CUDA dependencies enabled.

- [ ] **Step 2: Run the guard and confirm failure**

```powershell
cmake --build build --config Release --target test-expert-cache llama-bench llama-server
build/bin/Release/test-expert-cache.exe
```

Expected: the guard exposes currently declared but unwired GPU map/partition state or backend-inappropriate CUDA code in `ggml-base`.

- [ ] **Step 3: Delete the inactive GPU-map path**

Delete unused GPU map structures, wrappers, CUDA remap source/header, stats, and public APIs if no scheduler-owned contract is implemented in this task. Retain host-side remap/upload because it is the active path. Update `EXPERT_CACHE.md` and append a dated correction section to the optimization log for any historical device-map status wording that no longer describes the tree; do not alter historical results.

Do not replace this deletion with a speculative GPU compaction kernel. A future GPU remap design requires an observed bottleneck and a dedicated scheduler dispatch API.

- [ ] **Step 4: Run focused verification**

```powershell
cmake --build build --config Release --target test-expert-cache llama-bench llama-server
build/bin/Release/test-expert-cache.exe
```

Expected: active remapping tests still pass, CUDA and base builds are clean, and no emitted telemetry advertises an unavailable GPU map.

- [ ] **Step 5: Run benchmark protocol and append log entry**

Run the global correctness controls and alternating throughput pairs. Append the result, explicitly identifying this as a simplification/correctness experiment. Retain only if tests and output equivalence pass; report no performance claim unless variance-separated results justify one.

### Task 4: Add a bounded, decode-only carry-forward route snapshot and prefetch experiment

**Files:**
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `ggml/src/ggml-backend-expert-cache.cpp`
- Modify: `ggml/src/ggml-backend-expert-cache.h`
- Modify: `ggml/include/ggml-backend.h`
- Modify: `common/arg.cpp`
- Modify: `common/common.h`
- Modify: `src/llama-context.cpp`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `tools/llama-bench/llama-bench.cpp`
- Modify: `EXPERT_CACHE.md`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: Task 2 completion-backed slots, registered layer bundles, `ggml_backend_sched_graph_compute_async()`, and host remapped route IDs.
- Produces: an opt-in `--expert-cache-prefetch` decode-only mode with a scheduler-owned two-generation route snapshot table. It reports submitted, ready, stale, late, duplicate, rejected, and useful prefetches.

- [ ] **Step 1: Write failing route-snapshot tests**

Add deterministic unit/scheduler tests for: a ready prior-step route schedules one prefetch; an incomplete snapshot never blocks compute; stale sequence/graph-generation snapshots are discarded; duplicated IDs are deduplicated; disabled mode produces no prefetch; and unsupported/ambiguous operation forms retain the reactive path.

```cpp
static void test_expert_cache_prefetch_uses_ready_previous_route() {
    // Previous route: [1, 1, 4].
    // At next decode boundary, expect one bundle request for experts 1 and 4.
}

static void test_expert_cache_prefetch_never_waits_for_route_snapshot() {
    // Keep the snapshot completion event unsignaled.
    // Assert graph submission proceeds and prefetch_rejected_late increments.
}
```

- [ ] **Step 2: Run focused tests and confirm failure**

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: no production route-snapshot ownership or decode prefetch mode exists.

- [ ] **Step 3: Implement the smallest safe pipeline**

Create private scheduler state holding one in-flight and one ready snapshot per decode layer: sequence ID, graph generation, canonical registered bundle identity, deduplicated IDs, byte estimate, and completion event. At a completed decode graph boundary, asynchronously copy only the route IDs needed for the next opportunity. At the next boundary, nonblockingly poll readiness; stale, absent, or incomplete snapshots are discarded without a compute-stream wait.

When ready, invoke existing prefetch through Task 2's loading reservation path. Limit the first version to one layer and one complete gate/up/down bundle per boundary with hard byte and in-flight-DMA caps. Preserve the ordinary reactive route/miss path as fallback. Do not enable broad op-form recognition: begin with demonstrated `MUL_MAT_ID` identity, log unsupported decode lowering, and only add a classifier after a reproducing test.

Add `--expert-cache-prefetch` as a disabled-by-default opt-in. Add matching benchmark field/result serialization and a concise `EXPERT_CACHE.md` explanation of its experimental status and fallback semantics.

- [ ] **Step 4: Run focused verification**

```powershell
cmake --build build --config Release --target test-expert-cache llama-bench llama-server
build/bin/Release/test-expert-cache.exe
python scripts/expert-cache-determinism.py --exc 0 --json-out tools/results/expert-cache/prefetch-control.json
python scripts/expert-cache-determinism.py --exc 64M --excp 64 --extra-args "--expert-cache-prefetch" --json-out tools/results/expert-cache/prefetch-enabled.json
```

Expected: token hashes match for identical placement; unsupported graphs report a nonblocking bypass rather than a route synchronization stall.

- [ ] **Step 5: Run benchmark protocol and append log entry**

Benchmark cache-off, cache-on without prefetch, and cache-on with `--expert-cache-prefetch` in alternating fresh processes. Record PP512 and TG128 separately, route-snapshot readiness/late rate, bundle coverage, DMA queue bytes, useful-prefetch hits, cache misses, and full raw rows. If TG remains inactive (`eligible_ops == 0`) or the feature cannot achieve pre-consumer lead time, log the rejection and remove the option/code rather than preserving an inert path.

### Task 5: Add deadline-aware admission only after the snapshot path proves usable lead time

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.cpp`
- Modify: `ggml/src/ggml-backend-expert-cache.h`
- Modify: `ggml/include/ggml-backend.h`
- Modify: `common/arg.cpp`
- Modify: `common/common.h`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `tools/llama-bench/llama-bench.cpp`
- Modify: `EXPERT_CACHE.md`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: Task 4 ready route snapshots, Task 2 transfer-completion events, existing TG frequency counters, slot pool capacity, and bundle registrations.
- Produces: an opt-in `--expert-cache-prefetch-admission` policy. It either reserves a complete bundle before its deadline or records a rejection and changes no resident mapping.

- [ ] **Step 1: Write failing admission tests**

Test four exact outcomes: reject a candidate whose estimated completion misses its deadline; reject when byte/staging credits are exhausted; retain an existing resident instead of evicting it for a rejected prediction; and admit a complete bundle only when all projections and credits fit.

```cpp
static void test_expert_cache_admission_preserves_resident_on_late_prefetch() {
    // Resident expert 2 is useful; predicted expert 7 cannot meet deadline.
    // Assert slot for 2 is unchanged and late_rejections increments.
}
```

- [ ] **Step 2: Run focused tests and confirm failure**

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: current prefetch/rebalance has no deadline or DMA-credit decision boundary.

- [ ] **Step 3: Implement a conservative admission ledger**

Add an internal candidate record with bundle identity, bytes, predicted deadline, observed transfer ETA, confidence/reuse score, and decision reason. Estimate transfer ETA from completed cache transfers only; until enough samples exist, reject speculative admission. Use hard budgets for in-flight bytes, staging entries, and bundles per boundary. Reserve a slot and DMA credit first; publish no slot mapping until Task 2 completion. Keep periodic TG frequency as a tie-breaker, not the sole policy.

Expose only counters and a disabled-by-default flag; do not add an auto-tuning layer or persistent learned model in this task.

- [ ] **Step 4: Run focused verification**

```powershell
cmake --build build --config Release --target test-expert-cache llama-bench llama-server
build/bin/Release/test-expert-cache.exe
python scripts/expert-cache-determinism.py --exc 64M --excp 64 --extra-args "--expert-cache-prefetch --expert-cache-prefetch-admission" --json-out tools/results/expert-cache/admission-enabled.json
```

Expected: all rejection paths preserve output equivalence and resident mappings.

- [ ] **Step 5: Run benchmark protocol and append log entry**

Compare no cache, cache only, prefetch only, and prefetch-plus-admission. Log candidate count, admission/rejection reasons, deadline slack, ETA samples, credit pressure, useful-after-arrival hits, retained-resident fallbacks, raw PP/TG rows, and decision. Reject and remove the policy if it does not demonstrate usable prefetch lead time or produces no variance-separated benefit.

### Task 6: Canonicalize expert identity only if alias mismatch is reproduced

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.cpp`
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `src/llama-context.cpp`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: registered gate/up/down bundles, scheduler graph copies/views, and Task 4's route snapshot identity.
- Produces: only if required, an internal `ExpertIdentity { int32_t layer; uint64_t lineage_id; int32_t expert_id; }` with an alias resolver. Public pointer-shaped APIs remain wrappers.

- [ ] **Step 1: Write the reproducing alias test before implementation**

Construct canonical host weights and a fit-promoted/copy alias that represents the same expert source. Register the canonical bundle, execute the graph with the alias, and assert a resident/prefetched expert resolves to the same slot. Add a negative test where a same-shape tensor from another layer must not resolve.

- [ ] **Step 2: Run focused test and make the decision**

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

If the test already passes without new identity state, append the negative finding to the optimization log and stop this task: do not add lineage machinery without a demonstrated mismatch.

- [ ] **Step 3: Implement only if the reproducer fails**

Add a private alias registry seeded at bundle registration and graph materialization. Resolve slot lookup/remap and bundle residency through the canonical identity; retain each slot's authoritative source pointer for DMA. Fail closed on any layer, shape, type, stride, source, or lineage conflict. Migrate only lookup/remap first; do not rework allocation and rebalance maps until tests require it.

- [ ] **Step 4: Run focused verification**

```powershell
cmake --build build --config Release --target test-expert-cache test-expert-cache-profile
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-expert-cache-profile.exe
```

Expected: original and approved alias share a slot; cross-layer/same-shape aliases remain isolated.

- [ ] **Step 5: Run benchmark protocol and append log entry**

If implementation occurred, run the global correctness controls and alternating throughput pairs, then append evidence. If it was not needed, append the reproducer and no-change decision; the attempted investigation still belongs in the log.

## Final verification and documentation pass

- [ ] Run the focused targets changed by retained work:

```powershell
cmake --build build --config Release --target test-expert-cache test-expert-cache-profile llama-bench llama-server
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-expert-cache-profile.exe
```

- [ ] Re-run the exact deterministic controls for every retained cache mode and compare token hashes only within fixed placement rows.
- [ ] Re-run the complete alternating benchmark matrix for each retained mode; include cache-off control, cache-only, and each retained opt-in variant.
- [ ] Confirm every source change, rejected attempt, correctness result, raw benchmark row, and retention/reversion decision is appended to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`.
- [ ] Update `EXPERT_CACHE.md` only with current retained behavior, exact configuration semantics, known limitations, and links to the dated log sections. Historical results remain in the optimization log.
- [ ] Do not commit or push. Present the changed-file list and the complete evidence bundle to the human for review.
