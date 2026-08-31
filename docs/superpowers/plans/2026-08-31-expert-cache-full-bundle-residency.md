# Expert-Cache Full-Bundle Residency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, with current-head measurements, whether static complete `(layer, expert)` residency can produce a repeatable TG1 improvement for Qwen3.6-35B-A3B-APEX-Compact; implement a bundle-atomic dynamic policy only if the static upper bound wins net of fit displacement.

**Architecture:** Preserve the existing native graph for every incomplete TG1 route and for all PP or multi-token routes. Extend the existing TG profiler into an attested offline data producer, use a strict bundle-only static manifest to make resident state controllable, and calculate a held-out greedy full-hit oracle outside the scheduler. The runtime cache remains mutable only after a static manifest has produced a repeatable net win; dynamic policy reuses the same bundle transaction and never introduces partial execution.

**Tech Stack:** C++17, GGML scheduler and expert cache, CUDA backend, Python 3 standard library plus existing SciPy analysis dependency, CTest, `llama-bench`, `llama-server`, `tools/results/expert-cache/run-tg-matrix.py`, Qwen3.6-35B-A3B-APEX-Compact GGUF.

## Global Constraints

- TG1 production policy is exact: `0-7/8 -> native graph`, `8/8 -> persistent GPU sidecar`; do not add partial CPU/GPU execution.
- PP, speculative verification, and MTP remain native graph execution for this epic.
- No timed expert-weight host-to-GPU transfer. `expert_cache_bytes_ram_to_gpu == 0` is a hard acceptance invariant.
- Do not add GPU route compaction, a new background service, a learned routing model, or an exact solver before the greedy oracle proves inadequate.
- All route analysis consumes canonical router IDs, not cache-hit telemetry or unverified callback reads.
- Use the Compact preset settings in `G:/qwen3.6-35b-a3b-presets-exc-latest.ini` and use `tools/results/expert-cache/run-tg-matrix.py` for repeated alternating TG measurements.
- Every benchmark attempt, rejected hypothesis, raw result path, and gate outcome is appended to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`. Update `EXPERT_CACHE.md` only when a runtime behavior or retained disposition changes.
- Keep source, comments, JSON field names, and documentation ASCII-only.
- Do not commit or push unless the user explicitly authorizes that exact commit or push. The checkpoints below are review boundaries, not standing authorization.
- Treat all historical TG deltas, old manifest projections, and pre-existing cache-size conclusions as historical until reproduced on the current head.

---

## Decision Funnel

```text
geometry and placement truth
        |
canonical TG1 route trace + attestation
        |
strict complete-bundle loading + live residency census
        |
held-out static full-hit upper bound
        |
current-head paired cache-off / reserved / static matrices
        |
static net win >= 3-5%?
        |                         \
       yes                         no
        |                           \
bundle-atomic dynamic policy      document upper bound and stop
        |
dynamic achieves >= 70-80% of static full-hit benefit?
        |                         \
       yes                         no
        |                           \
retain policy                     improve residency only, not execution
```

The negative branch is a successful delivery. Do not implement dynamic admission after any failed static gate.

## Artifact and Interface Contract

### Attested route trace

The profiler writes one JSON object per complete TG1 route, plus one JSON header object. The trace is valid only if its header attests the exact model, placement configuration, and canonical-read comparison.

```json
{"schema":1,"kind":"header","model_sha256":"...","model_name":"Qwen3.6-35B-A3B-APEX-Compact","fit_target_mib":128,"cache_mib":128,"top_k":8,"workload":"coding","canonical_route_hash":"...","callback_route_hash":"...","callback_matches_canonical":true}
{"schema":1,"kind":"route","request_id":"coding-001","sequence_index":184,"layer":27,"top_k":8,"experts":[17,81,4,109,31,205,7,56]}
```

`callback_matches_canonical` is false if any `(sequence_index, layer, route index)` differs. The oracle refuses a trace with `callback_matches_canonical == false`.

### Complete-bundle manifest v3

A manifest enumerates bundles, never projections. The loaded tuple is atomically admitted as the registered layout dictates: `gate + up + down` for separate projections or `gate_up + down` for fused projections.

```json
{
  "format": 3,
  "admission": "8of8",
  "model": {
    "sha256": "...",
    "top_k": 8,
    "expert_count": 256,
    "layout_fingerprint": "..."
  },
  "cache_bytes": 134217728,
  "bundles": [
    {"layer": 24, "expert": 17, "projections": ["gate", "up", "down"]}
  ]
}
```

The loader rejects duplicate tuples, an unsupported format or admission value, a mismatched model identity or layout, an incorrect projection list, an unregistered layer, an out-of-range expert, and a manifest whose complete bundles exceed capacity. It publishes no partial bundle if any component cannot become resident.

### Offline oracle objective

For a selected resident bundle set `S`, route `r`, bundle bytes `b(layer, expert)`, and measured full-hit saving `saved_us(layer)`:

```text
value(S) = sum_r(fully_covered(r, S) * saved_us(r.layer))
subject to sum_(layer, expert in S) b(layer, expert) <= capacity
```

The first implementation uses both greedy candidate forms and takes the better deterministic result:

```text
single bundle: marginal newly-covered saved_us / bundle_bytes
route set:     newly-covered saved_us / missing-route-bundle-bytes
```

It uses only `8of8` coverage. It never awards a `7of8` route.

## Task 1: Lock the TG1 Execution Contract and Capture Current-Head Baseline

**Files:**
- Modify: `ggml/src/ggml-backend.cpp:3062-3209`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `tests/test-moe-partial-hit-bench.cpp`
- Modify: `tools/results/expert-cache/run-tg-matrix.py`
- Modify: `tools/results/expert-cache/test_run_tg_matrix.py`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`
- Modify: `EXPERT_CACHE.md` only if the retained production-policy wording differs from the verified code

**Interfaces:**
- Consumes: `ggml_backend_expert_cache_partition_bundle_routes()`, `ggml_moe_route_ready_sidecar_execute_full_hit()`, and the native `ggml_graph_view()` bundle fallback.
- Produces: one explicit production admission predicate: `n_tokens == 1 && n_hits == top_k && n_misses == 0` is the only sidecar branch.

- [ ] **Step 1: Write the production-policy regression first**

  Add a scheduler test beside `test_route_ready_sidecar_full_hit()` that prepares all nine hit masks. Assert masks `0..7` execute the unchanged CPU bundle and record no route-ready action. Assert mask `8` executes exactly one sidecar action and has CPU-reference-equivalent output.

  ```cpp
  require(stats.n_route_ready_actions == (n_hits == 8 ? 1 : 0));
  require(stats.n_route_ready_full_hits == (n_hits == 8 ? 1 : 0));
  require(stats.n_route_ready_native_fallbacks == (n_hits < 8 ? 1 : 0));
  require(tensor_close(output, cpu_reference));
  ```

- [ ] **Step 2: Run the focused test and record the red state**

  Run: `ctest --test-dir build -C Release -R "test-expert-cache" --output-on-failure`

  Expected before the code cutover: the new mask-7 assertion exposes any remaining production serial or concurrent route admission.

- [ ] **Step 3: Remove dead or reachable partial-production admission**

  Delete the `ggml_backend_moe_hetero_execute_serial()` production branch from the route-ready dispatcher and any now-unused `hetero_scratch` allocation reachable only through it. Keep direct partial executor tests and microbenchmarks as non-production regression evidence; do not route scheduler TG1 calls to them.

- [ ] **Step 4: Verify the policy and preserve the latency evidence**

  Run:

  ```powershell
  cmake --build build --config Release --target test-expert-cache test-moe-partial-hit-bench
  ctest --test-dir build -C Release -R "test-expert-cache" --output-on-failure
  .\build\bin\Release\test-moe-partial-hit-bench.exe --csv tools/results/expert-cache/2026-08-31-full-bundle-policy-mask-latency.csv
  ```

  Expected: only 8/8 is admitted by the scheduler; direct partial benchmark artifacts remain available but cannot imply production admission.

- [ ] **Step 5: Reproduce the current-head zero-hit control**

  Run the 28-pair 128 MiB matrix with no manifest, period 128, and the Compact preset. Persist every raw control/cache JSONL and stdout block under a new dated prefix. The runner must save stderr to a same-stem `.stderr.txt` file before this task is complete.

  Report per pair: TG tok/s, full-hit calls, fast rejects, `route_ready_prefix_sync_us`, `route_ready_route_id_us`, `route_ready_partition_us`, native fallback time, requested cache bytes, and expert RAM-to-GPU bytes.

- [ ] **Step 6: Append the baseline disposition**

  Add the command, build identity, raw paths, mask histogram, placement configuration, and paired mean/median/95% interval to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`. Do not infer a cache benefit from a row with zero full hits.

- [ ] **Step 7: Request commit authorization**

  Proposed checkpoint: `ggml: restrict TG1 cache admission to full bundles`.

## Task 2: Produce Exact MoE Geometry and Actual Fit Placement Reports

**Files:**
- Modify: `tests/test-moe-tg-profiler.cpp`
- Modify: `tests/CMakeLists.txt` only if the profiler requires a new target name
- Create: `tools/results/expert-cache/test_geometry_report.py`
- Create: `tools/results/expert-cache/geometry-schema-v1.json`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: `llama_model_get_tensor()`, `ggml_tensor`, `ggml_backend_buffer_is_host()`, existing model initialization in `test-moe-tg-profiler.cpp`.
- Produces: a versioned JSON geometry report and placement report for the exact model/configuration, consumed by the offline analyzer.

- [ ] **Step 1: Write parser-level tests for the report schema**

  `test_geometry_report.py` constructs one separate-projection and one fused-projection fixture. Require calculated bytes to include scales and require a layer to be marked unequal if any per-expert projection bytes, shape, quantization type, or scale bytes differ.

  ```python
  self.assertEqual(row["complete_bundle_bytes"], 576 * 1024 + 576 * 1024 + 800 * 1024 + 4096)
  self.assertFalse(report["all_repeated_moe_blocks_equal"])
  self.assertIn("down_scale", row["scales"])
  ```

- [ ] **Step 2: Run the schema tests to establish the red state**

  Run: `python tools/results/expert-cache/test_geometry_report.py`

  Expected: fail because the profiler has no JSON report mode.

- [ ] **Step 3: Add `--geometry-json` and `--placement-json` modes**

  Enumerate every registered MoE block and emit, per layer:

  ```text
  layer, expert_count, top_k, layout, gate/gate_up/up/down shapes,
  gate/up/down scale shapes, type, per-projection bytes,
  scale bytes, complete_bundle_bytes, complete_bank_bytes
  ```

  Determine bytes from the live `ggml_tensor` shape/type and include all APEX scale tensors registered for the block. The placement report enumerates every model layer as `gpu`, `cpu`, or `split`, records its host-resident MoE tensors, and includes the exact config fingerprint. It must distinguish full GPU layers from an MoE-only tensor placement.

- [ ] **Step 4: Add a compact model smoke invocation**

  Run the profiler in geometry-only mode with cache sizes `0`, `32`, `64`, `128`, `192`, `256`, `384`, and `512` MiB using the same `--fit` target as the benchmark preset. Save one geometry/placement report per capacity.

- [ ] **Step 5: Verify report invariants**

  Run the schema test again, then compare every repeated MoE block. The report must explicitly state `all_repeated_moe_blocks_equal: true` or list each varying layer and field. Record the measured bundle size rather than the historical 1.95 MiB estimate.

- [ ] **Step 6: Append geometry and placement evidence**

  Append the report paths and a table of bundle bytes, complete-bank MiB, host-MoE layers, and full GPU layers per capacity to the optimization log.

- [ ] **Step 7: Request commit authorization**

  Proposed checkpoint: `tooling: report MoE bundle geometry and fit placement`.

## Task 3: Capture and Attest Canonical TG1 Router Decisions

**Files:**
- Modify: `tests/test-moe-tg-profiler.cpp:66-163,274-317`
- Modify: `ggml/src/ggml-backend.cpp:3067-3093` only if post-decode profiler capture cannot produce canonical IDs
- Modify: `ggml/src/ggml-backend-expert-cache.h` and `.cpp` only if a scheduler-owned record-only trace sink is required
- Create: `tools/results/expert-cache/test_route_trace.py`
- Create: `tools/results/expert-cache/residency-workloads.json`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: `MUL_MAT_ID` `src[2]` route IDs and the producer completion ordering already used by the route-ready dispatcher.
- Produces: attested TG1-only route JSONL with `request_id`, `sequence_index`, `layer`, `top_k`, and route-order expert IDs.

- [ ] **Step 1: Write trace validation tests before changing capture**

  Create fixtures for repeated IDs, invalid IDs, a device-backed ID tensor, and a multi-token ID tensor. Require exactly one trace row for every `(TG1 token, layer)`, preserve route order, and reject a non-eight route before it enters oracle input.

  ```python
  self.assertEqual(route["experts"], [17, 81, 4, 109, 31, 205, 7, 56])
  self.assertEqual(route["top_k"], 8)
  self.assertEqual(len(rows_for_token_layer), 1)
  self.assertTrue(header["callback_matches_canonical"])
  ```

- [ ] **Step 2: Run the trace tests and the existing profiler diagnostic**

  Run the trace parser tests and `test-moe-tg-profiler.exe` with `MOE_PROF_DIAG=1`. Record whether callback reads contain zeros or invalid expert IDs. Do not use these routes for analysis yet.

- [ ] **Step 3: Implement dual capture with a strict attestation**

  Retain the callback capture only as a comparison source. After each `llama_decode()` returns, copy each remembered route tensor after its producer has completed and compare it to the callback capture. Include both hashes in the JSON header.

  If post-decode copies are not valid for every target placement, add a scheduler trace sink in the fast-reject path only after native `bundle_view` completion and synchronization. It is decode-gated, record-only, has no cache mutation, and writes no trace for `ne[1] != 1`. Its IDs are the canonical capture source; it must not read IDs before the producer runs.

- [ ] **Step 4: Define the reproducible workload corpus**

  Add `residency-workloads.json` with five named records: `coding`, `reasoning`, `general_chat`, `long_form`, and `multi_turn`. Each record specifies a UTF-8 prompt file, request ID prefix, sampling parameters, and generated-token target. The checked-in prompt files must be deterministic, non-sensitive, and collectively produce at least 10,000 TG1 route events per workload. `multi_turn` records separate request IDs and sequence indices for each turn.

- [ ] **Step 5: Capture the six required datasets**

  Produce one attested trace per named workload and one combined trace. Capture TG1 only, use fresh processes, and record PP/speculative/multi-token observations in separate files with `kind` values that the oracle rejects. Each accepted workload trace contains at least 10,000 TG1 route rows, not merely 10,000 generated tokens.

- [ ] **Step 6: Verify trace fidelity**

  For the same fixed greedy workload and placement, compare canonical trace IDs against the callback trace, route-ready mask telemetry when classification occurs, and the deterministic server token hash. If any captured expert is outside the model expert range, any route has wrong `top_k`, callback mismatch exists, or cache-on hash differs from cache-off, invalidate that trace and use only the canonical sink after fixing its ordering.

- [ ] **Step 7: Append the trace attestation**

  Record every trace header, prompt fingerprint, model fingerprint, mismatch count, and accepted/rejected status in the optimization log.

- [ ] **Step 8: Request commit authorization**

  Proposed checkpoint: `profiler: export attested TG1 expert routes`.

## Task 4: Measure Layer Cacheability and Actual Full-Hit Latency

**Files:**
- Modify: `tests/test-moe-tg-profiler.cpp`
- Modify: `tests/test-moe-oracle-bench.cpp`
- Modify: `tests/test-expert-cache.cpp`
- Create: `tools/results/expert-cache/test_layer_latency_report.py`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: attested route traces, the native bundle fallback, and `ggml_moe_route_ready_sidecar_execute_full_hit()`.
- Produces: layer-specific native and full-hit distribution data (`median`, `P95`, `mean`) and `saved_us(layer)`.

- [ ] **Step 1: Write the latency-report test**

  Verify percentile calculation, zero-sample rejection, and that the saved value is `native_median_us - sidecar_median_us`; the report must preserve negative savings rather than clamp it.

- [ ] **Step 2: Add deterministic single-layer 8/8 fixtures**

  Extend `test-expert-cache.cpp` so one layer has exactly the eight bundles for one known route pinned resident. Run the same route once via native fallback and once via the full-hit sidecar. Assert CPU-reference equivalence, one sidecar action only in the resident case, zero timed expert H2D, and stable route IDs.

- [ ] **Step 3: Run the focused red/green cycle**

  Run `test-expert-cache` before the sidecar instrumentation, then after it. The test must fail until its native and sidecar sample collection is wired.

- [ ] **Step 4: Add per-layer real-model sampling**

  In the profiler, record native bundle elapsed time and full-hit sidecar elapsed time separately for each layer. Time completed execution, not asynchronous submission. Force one complete static route at a time with `--max-swaps 0`; do not use a global all-experts GPU tensor.

- [ ] **Step 5: Capture distributions**

  Measure enough repeated TG1 observations to report `median`, `P95`, and `mean` for each host-eligible layer. Produce `saved_us(layer)` from the measured end-to-end full-hit and native paths. A layer with non-positive saving gets zero oracle value and remains eligible only for a documented diagnostic reason.

- [ ] **Step 6: Verify against the old microbenchmark**

  Run `test-moe-oracle-bench` as a sanity check only. Do not use its historical per-expert constant as oracle input if the end-to-end layer data differs.

- [ ] **Step 7: Request commit authorization**

  Proposed checkpoint: `benchmark: measure per-layer native and sidecar latency`.

## Task 5: Build the Offline Full-Hit Oracle and Cacheability Reports

**Files:**
- Create: `tools/results/expert-cache/analyze_residency.py`
- Create: `tools/results/expert-cache/test_analyze_residency.py`
- Modify: `tools/results/expert-cache/run-tg-matrix.py`
- Modify: `tools/results/expert-cache/test_run_tg_matrix.py`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: geometry JSON, placement JSON, attested TG1 JSONL traces, and layer latency JSON.
- Produces: per-layer cacheability JSON/CSV, a strict v3 manifest, and one oracle sweep result for each capacity.

- [ ] **Step 1: Write unit tests for trace statistics and coverage**

  Cover expert frequencies, top `1/2/4/8/16/32/64` coverage, Shannon entropy, normalized entropy, Gini coefficient, duplicate route IDs, and exact full-route containment. A test fixture must prove that high individual expert coverage does not imply an 8/8 hit.

  ```python
  self.assertAlmostEqual(result[12].full_hit_probability[32], 0.312)
  self.assertEqual(result[12].full_hit_probability[8], 0.001)
  self.assertGreater(result[14].gini, result[13].gini)
  ```

- [ ] **Step 2: Write oracle optimality and budget tests**

  Add synthetic routes where single-bundle greedy gets stuck but route-set greedy unlocks a valuable eight-expert route. Require the chosen set to remain within bytes, contain only complete bundles, use `8of8` scoring, and deterministically break score ties by `(layer, expert)`.

- [ ] **Step 3: Implement the analyzer**

  Add subcommands:

  ```text
  geometry       validate model and bundle geometry
  characterize   emit per-layer concentration and coverage tables
  oracle         emit one v3 manifest plus predicted benefit
  sweep          emit 32,64,96,128,160,192,256,384,512 MiB frontier
  compare        score a manifest against an independent held-out trace
  ```

  `characterize` emits the requested resident-count coverage table for `8,12,16,24,32,48,64,96,128`. `sweep` reports complete bundles, participating layers, predicted full hits, hit rate, predicted saved milliseconds per 1000 tokens, and bundles by layer. The analyzer refuses non-attested or model-incompatible traces.

- [ ] **Step 4: Use a held-out protocol**

  Select bundles on one trace partition or workload set and score them on an independent partition/workload set. The output labels every number `training`, `held_out`, or `combined`; no in-sample value is presented as the static upper bound.

- [ ] **Step 5: Generate the layer heatmap/table**

  Emit one table per layer with route entropy, experts required for `10/25/50/75/90%` full-hit probability, bundle bytes, native latency, sidecar latency, `saved_us`, and oracle-assigned capacity. Explicitly state whether equal-sized layers differ materially in full-hit usefulness.

- [ ] **Step 6: Apply Gate 1: static reachability**

  If all held-out sweeps through 512 MiB predict too few 8/8 hits to recover measured overhead and reach the 3-5% target, stop this epic: archive results, update both expert-cache documents with the upper bound, and do not begin Tasks 6-11. Otherwise generate only the promising `64/128/192/256` MiB v3 manifests.

- [ ] **Step 7: Request commit authorization**

  Proposed checkpoint: `tooling: add held-out full-hit residency oracle`.

## Task 6: Make Static Manifests Strict and Bundle-Atomic

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.h`
- Modify: `ggml/src/ggml-backend-expert-cache.cpp:2059-2350`
- Modify: `ggml/src/ggml-backend.cpp:3889-3956`
- Modify: `common/expert-cache-profile.h`
- Modify: `common/expert-cache-profile.cpp`
- Modify: `common/common.cpp:1438-1452`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `tests/test-expert-cache-profile.cpp`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: registered `ggml_moe_bundle_plan` layouts and manifest v3.
- Produces: `ggml_expert_bundle_key { int32_t layer; int32_t expert; }` internal cache identity and an all-or-nothing bundle transaction reusable by static seeding and later dynamic promotion.

- [ ] **Step 1: Write manifest rejection and atomicity tests**

  Add cases for bad format, `7of8`, model fingerprint mismatch, wrong fused/separate projection list, duplicate tuple, one missing registered projection, capacity overflow, a component stuck in `LOADING`, and a valid fused bundle. Require a failed load to leave the resident set byte-for-byte unchanged.

  ```cpp
  require(!ggml_backend_sched_load_pinned_manifest(sched, invalid_manifest.c_str()));
  require(before_resident_bundle_count == after_resident_bundle_count);
  require(ggml_backend_expert_cache_is_bundle_resident(cache, layer, expert));
  ```

- [ ] **Step 2: Write profile coherence tests**

  Extend `test-expert-cache-profile.cpp` so exported profile state, profile reload, and a v3 manifest all yield the same complete-bundle count and active slot-pool residency. This test decides whether the historical legacy-only `seed()` concern still exists on the checked-out head; do not change seeding until the test demonstrates a difference.

- [ ] **Step 3: Run focused tests to establish the red state**

  Run:

  ```powershell
  cmake --build build --config Release --target test-expert-cache test-expert-cache-profile
  ctest --test-dir build -C Release -R "test-expert-cache|test-expert-cache-profile" --output-on-failure
  ```

- [ ] **Step 4: Replace text scanning with strict v3 parsing**

  Parse the complete document before mutating cache state. Validate model/layout fields against registered bundles, validate all tuples and exact projection sets, calculate total complete-bundle bytes, and stage all component slots. On any error, release staged slots and return failure. After every staged component reaches `RESIDENT`, publish the bundle; pin it only after publication.

- [ ] **Step 5: Implement one bundle transaction**

  Centralize static loader, profile seed, async promotion, and eviction around one internal key and state transition:

  ```text
  EMPTY -> LOADING(all components reserved) -> RESIDENT(all components ready)
  RESIDENT -> EMPTY(all components released together)
  ```

  No public operation may publish gate-only, up-only, or down-only residency. Keep slot-pool storage implementation details private.

- [ ] **Step 6: Export verifiable loader outcomes**

  Surface parsed, rejected, staged, published, pinned, and resident-complete-bundle counts in the saved benchmark artifact. Prefer JSONL columns if they describe a stable counter; otherwise persist structured per-run loader JSON next to the raw JSONL and stderr. Do not rely on interactive stderr text as evidence.

- [ ] **Step 7: Verify a real static manifest**

  Load one promising manifest with `--max-swaps 0`, then assert from a live per-layer census that its requested layers have the expected complete resident-bundle counts before decode begins. Confirm malformed manifests fail rather than silently consuming partial capacity.

- [ ] **Step 8: Request commit authorization**

  Proposed checkpoint: `expert-cache: enforce complete-bundle static residency`.

## Task 7: Benchmark Static Oracle Manifests and Separate Fit Displacement

**Files:**
- Modify: `tools/results/expert-cache/run-tg-matrix.py`
- Modify: `tools/results/expert-cache/test_run_tg_matrix.py`
- Create: `tools/results/expert-cache/analyze_tg_matrix.py`
- Create: `tools/results/expert-cache/test_analyze_tg_matrix.py`
- Modify: `scripts/expert-cache-determinism.py`
- Modify: `scripts/expert-cache-determinism-matrix.py`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`
- Modify: `EXPERT_CACHE.md`

**Interfaces:**
- Consumes: strict manifests, profiler placement reports, saved JSONL/stderr/loader artifacts, and `run-tg-matrix.py` alternating fresh processes.
- Produces: paired analysis for cache-off, reserved-empty, and static-oracle configurations at each capacity.

- [ ] **Step 1: Write runner and analyzer tests**

  Verify each matrix creates the three rows below with identical model, fit, threads, KV type, flash attention, prompt, seed, batching, and speculative settings:

  For the 128 MiB matrix, require these exact rows:

  ```text
  A: -exc 0
  B: -exc 128M --max-swaps 0 with no useful residency
  C: -exc 128M --max-swaps 0 -pe tools/results/expert-cache/manifests/oracle-128m.json
  ```

  The runner repeats the same A/B/C construction for the selected 64, 192, and 256 MiB manifests.

  Require one stdout, one stderr, one JSONL, one loader result, and one placement report per child process. Require analyzer pairing by capacity, workload, order, and run index.

- [ ] **Step 2: Implement generic paired analysis**

  Replace the hardcoded `bundle-v2/analyze_v2.py` assumptions with a prefix-generic analyzer. Report per comparison: mean and median paired TG delta, standard deviation, positive pairs, 95% Student-t interval, wall time, full-hit calls, fallback calls, fast rejects, route ID D2H time/bytes, timed expert H2D bytes, static resident histogram, and placement metadata.

- [ ] **Step 3: Add placement attribution**

  The runner invokes the geometry/placement report in the exact cache/fit configuration before each capacity matrix. The analyzer reports:

  ```text
  baseline TG
  reserved-empty TG
  placement cost = reserved-empty / baseline - 1
  execution benefit = static / reserved-empty - 1
  net benefit = static / baseline - 1
  ```

  It also reports full GPU layer count and changed layer placement from the JSON reports. Never label `static / baseline` a cache execution gain.

- [ ] **Step 4: Run current-head static matrices**

  For each promising capacity at minimum `64`, `128`, `192`, and `256` MiB, run paired fresh-process A/B/C matrices. Retain raw artifacts, record cache-first and control-first order, and use enough pairs to calculate the configured interval. Do not reuse one process for multiple capacities.

- [ ] **Step 5: Validate deterministic output**

  Extend the determinism matrix with a v3 manifest row. For identical deterministic server input, require the cache-off, reserved-empty, and static-manifest token hash and content hash to match. Capture final `-excs` output to prove full hits did not introduce timed expert H2D.

- [ ] **Step 6: Apply Gate 2: static net value**

  Continue only if at least one capacity produces repeatable positive TG improvement after displacement, preferably `>= 3-5%`, with nonzero 8/8 sidecar calls, zero timed expert H2D, and passing deterministic output. If not, write the negative conclusion: held-out oracle frontier, static measured frontier, fit-placement cost, and maintenance recommendation. Stop Tasks 8-11.

- [ ] **Step 7: Request commit authorization**

  Proposed checkpoint: `bench: validate static full-bundle residency against fit cost`.

## Task 8: Add Bundle-Atomic Dynamic Residency Only After Gate 2 Passes

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.cpp`
- Modify: `ggml/src/ggml-backend-expert-cache.h`
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `tests/test-expert-cache-profile.cpp`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: the Task 6 internal bundle transaction.
- Produces: dynamic promotion and eviction that change only complete bundle state and leave scheduler execution policy unchanged.

- [ ] **Step 1: Write lifecycle tests first**

  Cover `LOADING -> RESIDENT`, failed upload rollback, concurrent use reservation, eviction after use, minimum-age protection, and max-promotion cap. Assert cache state never contains a usable partial bundle and that eviction releases every registered projection together.

- [ ] **Step 2: Run the focused cache tests to prove absence**

  Run `test-expert-cache` and `test-expert-cache-profile`; capture which lifecycle assertions fail before implementation.

- [ ] **Step 3: Reuse the Task 6 transaction for runtime mutation**

  Replace per-projection admission and eviction calls in the dynamic path with candidate bundle staging, all-component upload, readiness polling, atomic publication, reservation-aware use, and atomic release. The scheduler continues to choose native graph until `is_bundle_resident()` is true for every route expert.

- [ ] **Step 4: Verify mutation safety**

  Run the focused suite with the existing residency-epoch debug guard. Require no epoch mutation during graph compute, no null-buffer assertion, and zero timed expert H2D.

- [ ] **Step 5: Request commit authorization**

  Proposed checkpoint: `expert-cache: make dynamic residency bundle-atomic`.

## Task 9: Add Full-Hit-Oriented Admission, Allocation, and Hysteresis

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.cpp`
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `tests/test-expert-cache.cpp`
- Create: `tools/results/expert-cache/test_dynamic_policy.py`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: recent canonical TG1 route windows, measured `saved_us(layer)`, resident bundle sets, and bundle transaction API.
- Produces: candidates scored by marginal newly-coverable 8/8 route value per bundle byte, with unrestricted per-layer allocation.

- [ ] **Step 1: Write scoring tests**

  Create synthetic layers where individual LFU selects a popular expert but route-set scoring selects the eight experts that unlock a full route. Assert dynamic admission assigns zero bundles to low-value layers when a high-value layer has better marginal value. Assert `7of8` increments no score.

- [ ] **Step 2: Implement bounded route-window accounting**

  Keep only TG1 canonical route windows, per-expert frequency, route-set frequency, and current complete resident sets. PP/multi-token routes cannot affect the score. Compute candidates for an individual missing bundle and for a complete missing set; choose the greatest positive value per byte.

- [ ] **Step 3: Implement hysteresis and bounded change rate**

  Make a candidate replace a victim only when:

  ```text
  candidate_score > victim_score * promotion_margin
  candidate age and victim age satisfy the configured minimum residency age
  promotions_this_period < maximum_promotions_per_period
  ```

  Use named defaults and expose them for benchmark configuration. Apply frequency decay at the period boundary, not during decode execution.

- [ ] **Step 4: Add a decision replay before real mutation**

  On held-out traces, run the policy in pure decision-replay mode against the static oracle start state. Record would-be complete bundles and predicted 8/8 value. It must beat the legacy raw-LFU replay and achieve a pre-registered substantial fraction of the static oracle before a live mutation A/B is attempted.

- [ ] **Step 5: Verify no allocation fairness rule remains**

  Add a test where the valid outcome is `Layer 12: 0`, `Layer 13: 20`, `Layer 21: 31`, `Layer 32: 14`. Reject any `capacity / layer_count` partitioning in dynamic admission.

- [ ] **Step 6: Request commit authorization**

  Proposed checkpoint: `expert-cache: score dynamic bundles by full-hit value`.

## Task 10: Sweep Dynamic Periods, Persist Compatible Bundle Profiles, and Expose Telemetry

**Files:**
- Modify: `ggml/include/ggml-backend.h`
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `ggml/src/ggml-backend-expert-cache.cpp`
- Modify: `tools/llama-bench/llama-bench.cpp`
- Modify: `common/expert-cache-profile.h`
- Modify: `common/expert-cache-profile.cpp`
- Modify: `common/common.cpp`
- Modify: `tools/server/server-context.cpp`
- Modify: `tests/test-expert-cache.cpp`
- Modify: `tests/test-expert-cache-profile.cpp`
- Modify: `tools/results/expert-cache/run-tg-matrix.py`
- Modify: `tools/results/expert-cache/test_run_tg_matrix.py`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: bundle keys and dynamic scores from Tasks 8-9.
- Produces: persisted bundle-level candidates and stable global/per-layer telemetry in benchmark JSONL and server output.

- [ ] **Step 1: Write counter and serialization tests first**

  Add tests for per-layer route count, full-hit count, resident bundles, promotions, evictions, bytes resident/promoted/evicted, promotion H2D bytes, timed H2D bytes, and every route classification. Add profile tests for model identity, GGUF hash, expert count, quantization/layout fingerprint, cache capacity, and top-k mismatch rejection.

- [ ] **Step 2: Add stats in lockstep**

  For every new stat, update all required sites together: `ggml_backend_expert_cache_stats`, scheduler aggregation, `subtract_expert_cache_stats()`, `get_fields()`, field type list, `get_values()`, `cmd_params` defaults, and all three positional `cmd_params_instance` construction sites. Add JSONL schema assertions in a Python test so field/value order cannot drift.

- [ ] **Step 3: Persist bundle candidates, not projection entries**

  Version the profile format. Persist `(layer, expert)`, bundle score, observed frequency, model/layout fingerprint, and cache capacity. On load, rank candidates within current capacity, admit only complete bundles through Task 6 transaction, and allow runtime evidence to replace stale entries. Do not seed partial data.

- [ ] **Step 4: Add residency telemetry**

  Expose global resident complete bundles, resident bundles per layer, candidates, promotions, evictions, bundle hit/miss counts, 8/8 rate, sidecar calls, fast rejects, route classifications, resident/promoted/evicted bytes, promotion H2D bytes, and timed H2D bytes. Per layer expose route count, full hits and rate, resident bundles, promotions, evictions, and estimated saved microseconds.

- [ ] **Step 5: Sweep promotion periods**

  Run `16`, `32`, `64`, `128`, `256`, and `512` token periods with the selected cache capacity. Report full-hit rate, promotion count, uploaded bytes, TG, placement metadata, and net benefit. Optimize net TG benefit, not hit rate.

- [ ] **Step 6: Verify profile and telemetry behavior**

  Run focused cache/profile tests, one fresh process warm-start smoke, and a current-head dynamic matrix. Require saved JSONL to show timed inference H2D bytes exactly zero.

- [ ] **Step 7: Request commit authorization**

  Proposed checkpoints: `expert-cache: persist scored bundle residency` and `telemetry: report full-hit bundle residency`.

## Task 11: Compare Dynamic Policy to Static Oracle and Close the TG1 Decision

**Files:**
- Modify: `tools/results/expert-cache/analyze_tg_matrix.py`
- Modify: `tools/results/expert-cache/run-tg-matrix.py`
- Modify: `scripts/expert-cache-determinism-matrix.py`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`
- Modify: `EXPERT_CACHE.md`

**Interfaces:**
- Consumes: winning static manifest, dynamic period results, paired raw artifacts, and per-layer telemetry.
- Produces: retained capacity/policy or an evidence-backed rejection.

- [ ] **Step 1: Run static-oracle and dynamic paired matrices**

  Use the same workload distributions, cache capacity, fit target, process order balance, and benchmark command. Compare cache-off, reserved-empty, static manifest, and dynamic policy.

- [ ] **Step 2: Calculate oracle fraction**

  Report:

  ```text
  dynamic_full_hit_benefit / static_full_hit_benefit
  dynamic_net_TG_delta / static_net_TG_delta
  ```

  Explain a zero or negative denominator rather than dividing by it. Target 70-80% of static full-hit benefit only when the static denominator is positive and repeatable.

- [ ] **Step 3: Re-run determinism and correctness gates**

  Execute the deterministic server matrix and the full focused cache suite. Require no malformed output, slash loops, stale route IDs, stale node state, null-buffer assertion, or timed expert H2D.

- [ ] **Step 4: Publish the conclusion**

  Update `EXPERT_CACHE.md` with retained policy/capacity or negative disposition. Append all raw paths, paired statistics, placement decomposition, full-hit rate, oracle fraction, and stop decision to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`.

- [ ] **Step 5: Request commit authorization**

  Proposed checkpoint: `bench: compare dynamic bundle policy to static oracle`.

## Task 12: Preserve Correctness Boundaries and Investigate MTP Separately

**Files:**
- Modify: `tests/test-expert-cache.cpp`
- Modify: `tests/test-expert-cache-profile.cpp`
- Modify: `tests/test-backend-ops.cpp`
- Modify: `tests/test-benchmark-mtp.cpp` or create a narrowly scoped MTP regression in the existing test target
- Modify: `src/llama-context.cpp:617-622` only after an MTP reproducer identifies the fault
- Modify: `scripts/benchmark_mtp_suite.py`
- Modify: `docs/plans/2026-08-20-mtp-expert-cache-unified.md`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: existing native fallback/sidecar test fixtures and the MTP skip-registration rule.
- Produces: a regression matrix proving cache safety across modes; a separate MTP assertion root-cause record and fix if independently justified.

- [ ] **Step 1: Add the regression matrix**

  Cover cache off, empty cache, partial residency, full residency, fit on/off, speculative on/off, single-turn, multi-turn, APEX scales, fused and unfused MoE, LOADING-to-RESIDENT, eviction during/after use, PP, TG1, and multi-token verification. Each incomplete bundle must use the native graph.

- [ ] **Step 2: Run focused correctness tests**

  Run `test-expert-cache`, `test-expert-cache-profile`, and `test-backend-ops` after every scheduler/cache state-machine change. Keep real-model deterministic server replay separate from synthetic tests.

- [ ] **Step 3: Reproduce the MTP assertion in isolation**

  Use the MTP preset with cache off/on and a minimal generated-token count. Capture the null-buffer assertion stack, cache state, graph type, tensor buffer, and registration status. Do not apply any TG1 residency change as a speculative MTP fix.

- [ ] **Step 4: Fix only the demonstrated MTP fault**

  Write the reproducer before the change. Preserve the MTP registration skip rule unless evidence proves that exact rule is wrong. Verify cache-on MTP no longer asserts and does not alter TG1 or PP residency behavior.

- [ ] **Step 5: Record independent MTP disposition**

  Document whether the MTP path is safe. Do not claim or benchmark MTP cache acceleration in this epic.

- [ ] **Step 6: Request commit authorization**

  Proposed checkpoint: `tests: preserve expert-cache fallback and MTP safety`.

## Final Acceptance Checklist

- [ ] Geometry report proves or disproves equal-size repeated MoE bundles and includes every required APEX scale.
- [ ] Six attested datasets exist: five workload classes plus combined; each accepted workload has at least 10,000 TG1 route records.
- [ ] Layer concentration, complete-route coverage, full-hit probability frontier, and measured per-layer latency tables are retained as artifacts.
- [ ] Offline manifest selection uses held-out 8/8-only value per byte and never `7of8` credit.
- [ ] Every static capacity sweep includes complete bundles, layer allocation, predicted full hits, predicted saved time, and actual fit placement.
- [ ] Every real static matrix includes cache-off, reserved-empty, and oracle-manifest rows; reports placement cost and execution benefit separately.
- [ ] Every timed matrix has zero expert-weight RAM-to-GPU bytes and passing deterministic output.
- [ ] Dynamic code exists only if static residency has a repeatable net TG win; it is complete-bundle atomic, layer-unconstrained, hysteretic, bounded, and profile-compatible.
- [ ] Dynamic policy outcome is compared to static oracle; if it misses the target, improve residency only, not partial execution.
- [ ] MTP is safe or has a separate evidence-backed issue record; it is not accelerated by this epic.
- [ ] `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` and `EXPERT_CACHE.md` contain the final positive or negative conclusion and every result path.
