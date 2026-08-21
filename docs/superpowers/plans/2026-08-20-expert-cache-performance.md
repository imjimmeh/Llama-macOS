# Expert Cache Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the Compact model's measured performance advantage over the rebuilt baseline by selecting the proven no-cache/no-split profile now and making future cache decisions depend on measured cache value.

**Architecture:** `llama-bench` will snapshot aggregate `ggml_backend_expert_cache_stats` after warmup and after the measured repetitions, then emit their difference in every output format. This exposes the transfer savings and bookkeeping cost that determine whether caching helps a given workload. The Compact/GTX 1080 preset will explicitly disable expert cache and FFN partitioning until an isolated real-request replay proves a cache configuration wins.

**Tech Stack:** C++17, ggml scheduler API, llama-bench, CMake/CTest, Windows PowerShell, jq.

## Global Constraints

- Target model: `Qwen3.6-35B-A3B-APEX-Compact.gguf` on GTX 1080 (SM 6.1).
- Baseline configuration: `-exc 0 --ffn-split 0`.
- Keep batch size 4096, ubatch size 2048, 14 threads, Q8_0 KV cache, Flash Attention, mlock, fit target 256 MiB.
- Run each cache/FFN point in a fresh process. The multi-value llama-bench grid reproduced a CUDA error while changing cache state within one process.
- Do not alter scheduler ordering, route-ID synchronization, cache admission, or FFN graph construction in this plan.
- Do not accept a cache result merely because it beats an older cache revision. It must beat the no-cache/no-split control beyond measured variance for the intended workload.
- Do not create commits automatically. The user owns all commits.

---

### Task 1: Emit measured expert-cache telemetry

**Files:**
- Modify: `tools/llama-bench/llama-bench.cpp:1501-1757, 2401-2552`

**Interfaces:**
- Consumes: `llama_context_get_sched(const llama_context *)` from `include/llama.h:585`.
- Consumes: `ggml_backend_sched_get_expert_cache_stats(ggml_backend_sched_t, int, ggml_backend_expert_cache_stats *)` from `ggml/include/ggml-backend.h:391`.
- Produces: Stable llama-bench result fields for all nineteen members of `ggml_backend_expert_cache_stats`.

- [ ] **Step 1: Run the output-contract test and confirm red**

Run:

```powershell
build/bin/Release/llama-bench.exe -m "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" -p 512 -n 128 -r 1 -o jsonl -fitt 256 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -t 14 -fa on -lm mlock -exc 0 --ffn-split 0 | jq -e 'has("expert_cache_requests") and has("expert_cache_dma_wait_ns")'
```

Expected: `jq` exits nonzero because llama-bench does not yet serialize the expert-cache telemetry fields.

- [ ] **Step 2: Confirm the source statistics are already covered**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected: PASS. `test_slot_pools_and_remapping()` already asserts a zero-copy statistic. No new lower-level cache test is required because Task 1 changes benchmark serialization, not expert-cache accounting.

- [ ] **Step 3: Add an explicit llama-bench statistics snapshot helper**

In `tools/llama-bench/llama-bench.cpp`, add a file-local helper near `test`:

```cpp
static ggml_backend_expert_cache_stats get_expert_cache_stats(const llama_context * ctx) {
    ggml_backend_expert_cache_stats stats = {};
    ggml_backend_sched_t sched = llama_context_get_sched(ctx);
    if (sched) {
        ggml_backend_sched_get_expert_cache_stats(sched, -1, &stats);
    }
    return stats;
}
```

Add a second file-local helper that subtracts each unsigned counter in an after snapshot from its before snapshot. Assert no field decreases. The helper must subtract all of these fields:

```text
n_requests
n_hits
n_zero_copy_hits
n_d2d_fallback_hits
n_speculative_prefetches
n_misses
n_evictions
n_rebalances
n_jit_swaps
bytes_ram_to_gpu
bytes_avoided
n_cpu_id_remaps
n_gpu_id_resolutions
staging_memcpy_bytes
direct_pinned_dma_bytes
n_map_updates
map_update_bytes
dma_ns
dma_wait_ns
```

Do not reset global scheduler statistics. A before/after difference excludes warmup without adding a mutable public reset API.

- [ ] **Step 4: Store and serialize the measured difference**

Extend `struct test` with one `ggml_backend_expert_cache_stats expert_cache_stats` member. After warmup, before the repetition loop at `tools/llama-bench/llama-bench.cpp:2467`, capture `stats_before`. After the repetition loop at line 2538, set `t.expert_cache_stats = subtract(get_expert_cache_stats(ctx), stats_before)`.

Append these stable output names to `test::get_fields()` and append matching decimal values in identical order to `test::get_values()`:

```text
expert_cache_requests
expert_cache_hits
expert_cache_zero_copy_hits
expert_cache_d2d_fallback_hits
expert_cache_speculative_prefetches
expert_cache_misses
expert_cache_evictions
expert_cache_rebalances
expert_cache_jit_swaps
expert_cache_bytes_ram_to_gpu
expert_cache_bytes_avoided
expert_cache_cpu_id_remaps
expert_cache_gpu_id_resolutions
expert_cache_staging_memcpy_bytes
expert_cache_direct_pinned_dma_bytes
expert_cache_map_updates
expert_cache_map_update_bytes
expert_cache_dma_ns
expert_cache_dma_wait_ns
```

Classify all nineteen fields as `INT` in `test::get_field_type()`. Emit zeroes for `-exc 0`; do not conditionally omit fields. This preserves one schema for CSV, JSON, JSONL, Markdown, and SQL output.

- [ ] **Step 5: Run the unit test and build llama-bench**

Run:

```powershell
cmake --build build --config Release --target test-expert-cache llama-bench
build/bin/Release/test-expert-cache.exe
```

Expected: PASS. The cache unit test proves the raw counters used by the benchmark are updated deterministically.

- [ ] **Step 6: Smoke-test the public result schema**

Run the no-cache control and verify all telemetry values are zero:

```powershell
build/bin/Release/llama-bench.exe -m "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" -p 512 -n 128 -r 1 -o jsonl -fitt 256 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -t 14 -fa on -lm mlock -exc 0 --ffn-split 0 | jq -e '.expert_cache_requests == 0 and .expert_cache_dma_wait_ns == 0'
```

Then run the cache case and retain the complete JSONL result for comparison:

```powershell
build/bin/Release/llama-bench.exe -m "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" -p 512 -n 128 -r 10 -o jsonl -fitt 256 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -t 14 -fa on -lm mlock -exc 256 -excp 64 --ffn-split 0 | jq -se 'map(select(.expert_cache_requests > 0)) | length > 0 and all(.[]; .expert_cache_hits <= .expert_cache_requests)'
```

Expected: both commands exit zero. The cache case contains at least one request-bearing result, reports a complete nonnegative telemetry set, and no request-bearing result has more hits than requests.

### Task 2: Correct the Compact/GTX 1080 preset

**Files:**
- Modify: `G:/qwen3.6-35b-a3b-presets-exc.ini:153-176`

**Interfaces:**
- Consumes: the existing `[qwen3.6-35B-apex-compact]` preset.
- Produces: an explicit no-cache/no-split configuration for this profile, independent of `[*]` defaults.

- [ ] **Step 1: Establish the red control from the current preset**

Run a fresh-process benchmark with the current production values and record PP/TG plus telemetry:

```powershell
build/bin/Release/llama-bench.exe -m "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" -p 512 -n 128 -r 10 -o jsonl -fitt 256 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -t 14 -fa on -lm mlock -exc 256 -excp 64 --ffn-split 0.35
```

Expected: lower TG than the no-cache/no-split control, matching the reproduced regression direction.

- [ ] **Step 2: Override inherited tuning in the Compact section**

In `[qwen3.6-35B-apex-compact]`, replace:

```ini
exc = 256M
```

with:

```ini
exc = 0
ffn-split = 0
```

Do not change the global `[*]` values. Other models may have different capacity and locality constraints.

- [ ] **Step 3: Verify the corrected preset through the actual launch path**

Launch the application that consumes this preset with its normal command, then capture its model-load configuration and timing output. Confirm it reports `expert-cache = 0` and `ffn-split = 0` for the Compact model.

Run the matching isolated benchmark command:

```powershell
build/bin/Release/llama-bench.exe -m "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" -p 512 -n 128 -r 10 -o jsonl -fitt 256 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -t 14 -fa on -lm mlock -exc 0 --ffn-split 0
```

Expected: it reproduces the known tuned-binary control advantage over the rebuilt baseline and reports zero cache telemetry.

### Task 3: Create a cache-admission evidence report

**Files:**
- Create: `tools/results/qwen36-compact-expert-cache-gtx1080.jsonl`
- Create: `tools/results/qwen36-compact-expert-cache-gtx1080.md`

**Interfaces:**
- Consumes: the JSONL schema from Task 1 and deterministic output hashes collected from real server replay.
- Produces: a versioned decision record identifying either a cache configuration that wins or a decision to retain `exc = 0`.

- [ ] **Step 1: Collect synthetic ablation data in isolated processes**

Run one process for every point in this matrix:

```text
ffn-split: 0
expert-cache: 0, 64, 128, 192, 256 MiB
expert-cache-period: 0, 64, 128 for nonzero cache sizes
prompt/generation: pp512, tg64, tg128, tg256, tg512
repetitions: 10
```

Use the Task 1 benchmark command and change only `-exc`, `-excp`, `-p`, and `-n`. Keep raw JSONL records instead of averaging them in an external spreadsheet.

- [ ] **Step 2: Collect deterministic real-request replay data**

For each candidate that is not slower than `exc=0` in the synthetic TG128 result, run a fresh llama-server process with:

```text
temperature = 0
top-k = 1
fixed seed
parallel = 1
fixed prompt sequence
```

Measure cold TG64, warm TG128/TG256, and steady TG512. Hash every full token sequence. Reject any candidate whose output hash differs from `exc=0`.

- [ ] **Step 3: Write the decision record from raw evidence**

For every row, report:

```text
build commit and number
model checksum/path
full runtime parameters
PP/TG mean and standard deviation
requests, hits, zero-copy hits, D2D fallback hits, speculative prefetches, misses, evictions, rebalances, and JIT swaps
bytes RAM-to-GPU, bytes avoided, staging/direct DMA bytes, and map-update bytes
CPU remaps, GPU resolutions, and map updates
dma_ns and dma_wait_ns
output hash
```

Select a nonzero cache setting only when it exceeds the no-cache/no-split control beyond observed variance in the intended replay workload. Otherwise retain `exc = 0` for Compact/GTX 1080.

#### Focused decode-placement result (2026-08-21)

A temporary scheduler experiment forced host-resident Compact `GGML_OP_MUL_MAT_ID` operations onto CUDA when one expert fit in the cache. It was rejected and removed. On the target TG128 workload, 256 MiB produced 14.68 tok/s with 424,960 requests and 77,918 hits; 1024 MiB produced 15.99 tok/s with 496,640 requests and 253,838 hits. The restored normal route measured 26.47 tok/s with no cache requests.

This disproves cache capacity as a sufficient admission condition for Compact decode on the GTX 1080. Do not implement scheduler placement changes from Task 4 unless a trace demonstrates a design that removes the CPU/GPU boundary cost for the routed projections. The full Task 3 matrix and deterministic replay remain required before selecting any cache configuration.

### Task 4: Gate any router-ID overlap redesign

**Files:**
- No source changes in this task.
- Evidence source: `ggml/src/ggml-backend.cpp:1674-1694`.

**Interfaces:**
- Consumes: Task 3 telemetry and an Nsight Systems trace of the CUDA backend.
- Produces: a binary go/no-go decision for a separate scheduler design and implementation plan.

- [ ] **Step 1: Capture a decode trace for a winning or near-winning cache workload**

Trace the router-ID `ggml_backend_tensor_get_async()` plus `ggml_backend_synchronize(ids_backend)` region and the subsequent cache transfer/compute region for TG256.

- [ ] **Step 2: Apply the explicit threshold**

Proceed to a separate design only if router-ID synchronization accounts for at least 5% of decode wall time and the trace shows transferable overlap with independent CUDA work. Otherwise close this line of work and keep the measured cache policy from Task 3.

- [ ] **Step 3: Preserve correctness before any future asynchronous implementation**

A later implementation must first extend `tests/test-expert-cache.cpp` with deterministic sequences that cover all-hit, all-miss, and mixed-hit remapping; retain the existing direct assertions for remapped IDs and zero-copy counters. It must then prove identical CPU/CUDA model output hashes for cold and warm cache states before performance claims are accepted.
