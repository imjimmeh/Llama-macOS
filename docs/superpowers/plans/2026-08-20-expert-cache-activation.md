# Expert Cache Activation and Fast-Bypass Plan

## Goal

Make the GTX 1080 Compact-model cache path either execute correctly or add no decode overhead. Do not select cache capacity or FFN split values by benchmark tuning. The cache implementation must expose enough telemetry to prove which outcome occurs.

## Verified diagnosis

- `--expert-cache 256` means **256 bytes**, not 256 MiB. `common/arg.cpp:2743-2766` treats a value without a unit as bytes. The preset likewise uses unqualified cache values.
- At `-exc 128`, the server saved the profile but neither seeded it nor printed an expert-cache line after a 128-token request. `server-context.cpp:645-652` prints that line whenever `n_requests > 0`; therefore the active path made zero cache requests.
- At `-exc 256M`, the same profile loaded and seeded 512 experts. The 128-token request still made zero cache requests, with both `--ffn-split 0` and `--ffn-split 0.35`. Capacity is therefore not the only blocker.
- The dispatch guard in `ggml_backend_sched_compute_splits()` only examines `split->graph.nodes[0]` (`ggml-backend.cpp:1662-1669`). It does not search for a `GGML_OP_MUL_MAT_ID` that consumes the current copied host-expert tensor later in that split. This makes cache activation depend on incidental scheduler node order.
- The guarded block unconditionally synchronizes the input backend and then copies router IDs to the host and synchronizes again (`ggml-backend.cpp:1674-1694`) before it knows whether the cache can store one expert. A tiny cache can therefore add synchronization and CPU bookkeeping while falling back to the normal copy.

## Constraints

- Preserve bit-identical logits and token output for cache off, cold cache, and warm cache.
- Preserve the normal full-tensor copy when a host-expert tensor has no eligible `MUL_MAT_ID` consumer or cache capacity cannot hold one expert.
- No CUDA-only API in shared scheduler code. The scheduler uses backend events and generic async tensor APIs.
- Keep existing public cache semantics: an unqualified `SIZE` remains bytes. Add an explicit warning instead of silently changing unit behavior.
- Do not reintroduce transition prediction or speculative prefetch. Neither has a production caller.

## Task 1: Test and fix dispatch-node selection

**Files**
- Modify: `ggml/src/ggml-backend.cpp:1640-2016`
- Modify: `tests/test-expert-cache.cpp`

1. Add a file-local helper that scans the current split graph for `GGML_OP_MUL_MAT_ID` nodes whose `src[0]` is the current `input_cpy`. It returns the matching node(s), not merely `nodes[0]`.
2. Use the helper before entering the cache-specific path. The normal input-copy path remains unchanged when no node matches.
3. Restore every rewritten node after graph execution through the existing `restored_nodes` mechanism.
4. Extend the CUDA-backed cache integration test with a split where an unrelated operation precedes the matching expert `MUL_MAT_ID`. Seed the cache, execute one decode-equivalent graph, and assert the matching node used the slot-pool tensor and IDs were remapped. The existing first-node case must still pass.
5. Run the focused test red before the helper change, then build and rerun `test-expert-cache`.

## Task 2: Bypass unserviceable cache before synchronization

**Files**
- Modify: `ggml/include/ggml-backend.h`
- Modify: `ggml/src/ggml-backend-expert-cache.h`
- Modify: `ggml/src/ggml-backend-expert-cache.cpp`
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `tests/test-expert-cache.cpp`

1. Add a narrow internal/public cache query that answers whether a cache can store an expert of a given byte size. It must compare against real capacity and avoid allocation, eviction, mutation, or a CUDA call.
2. After identifying a matching `MUL_MAT_ID`, compute `expert_size` and call the query before `ggml_backend_synchronize(input_backend)` or router-ID D2H.
3. If the answer is false, skip all cache-specific code and perform the existing full-tensor copy. The cache must not request router IDs, increment hit/miss counters, allocate a slot, or wait for DMA.
4. Add a regression test with cache capacity below one expert. Assert no slot allocation and that the source data reaches the original input-copy destination unchanged.
5. Add a CLI warning in `common/arg.cpp` when `--expert-cache` has no unit and is smaller than 1 MiB. The warning must state the interpreted byte count and give a unit-suffixed example. It must not alter parsed values.

## Task 3: Make activation observable in normal benchmarks

**Files**
- Modify: `ggml/include/ggml-backend.h`
- Modify: `ggml/src/ggml-backend-expert-cache.cpp`
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `tools/llama-bench/llama-bench.cpp`
- Modify: `tools/server/server-context.cpp`

1. Add aggregate counters for eligible host-expert operations and capacity-bypassed operations. They answer whether zero requests means no eligible graph node or a deliberately skipped tiny cache.
2. Extend llama-bench's existing before/after expert-cache snapshot and its JSONL/CSV fields with the two counters. Do not reset global state.
3. Make the server print cache telemetry whenever the scheduler owns an expert cache, including zero-request executions. Report requests, eligible operations, capacity bypasses, and hit rate only when requests are nonzero.
4. Add a llama-bench schema test/smoke command covering `-exc 0`, `-exc 256`, and `-exc 256M`. The unqualified case must report capacity bypasses rather than request traffic; the unit-suffixed case must report eligible operations and requests after Task 1.

## Task 4: Measure only the execution path that now runs

**Files**
- Modify only if Task 3 cannot separate routing wait, host bookkeeping, and cache DMA: `ggml_backend_expert_cache_stats`, `ggml-backend.cpp`, `tools/llama-bench/llama-bench.cpp`.

1. Run fresh-process `llama-bench` measurements with the fixed binary for `-exc 0`, `-exc 256`, and `-exc 256M`; use the same model, `-fitt 256`, Q8_0 KV, Flash Attention, 14 threads, PP512, and TG128/TG256.
2. Accept a cache improvement only when the unit-suffixed cache has nonzero eligible operations and requests, stable output, and a throughput gain beyond run-to-run variation over `-exc 0`.
3. If active cached decode is still slower, add phase timing around router-ID D2H, host ID processing/remap, cache H2D, and its required backend waits. Use the resulting dominant phase to choose the next isolated optimization.
4. The first candidate is removal of the redundant `input_backend` synchronization only if traces show the subsequent ID-read synchronization fully subsumes it. Add deterministic cold/warm output equivalence coverage before making that change.

## Verification matrix

```text
Focused correctness:   test-expert-cache
Build:                 cmake --build build --config Release --target test-expert-cache llama-bench llama-server
Tiny-cache regression: -exc 256 -> warning, zero router-ID/cache traffic, no decode regression
Active-cache smoke:    -exc 256M + existing coder profile -> profile seeds, cache telemetry has requests
Output contract:        same fixed prompt/seed, cache off/cold/warm token sequences identical
Performance contract:   fresh-process PP512 + TG128/TG256, cache-enabled result beats cache-off only beyond observed variance
```

## Non-goals

- Do not retain the current cache profile as evidence of speed: it can seed slots while dispatch remains inactive.
- Do not tune `--ffn-split`, cache capacity, cache period, or GPU offload choices until the corrected cache path has telemetry.
- Do not implement speculative prefetch, transition prediction, or a CUDA-specific router-ID path without Task 4 evidence.
