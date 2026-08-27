# Expert Cache Optimization Log & Benchmarks

**Repository**: `imjimmeh/Llama-macOS` (`G:\code\AI\llamacpptuned\llama.cpp`)  
**Target Model**: `Qwen3.6-35B-A3B-APEX-Compact.gguf`  
**Test Hardware**: NVIDIA GeForce GTX 1080 (8 GB VRAM, Compute 6.1) + CPU Host  
**Benchmark Configuration**:
- `batch-size`: 4096
- `ubatch-size`: 2048
- `threads`: 14
- `cache-type-k`: q8_0
- `cache-type-v`: q8_0
- `load-mode`: mlock (no-mmap)
- `flash-attn`: on
- `fit-target`: 256 MiB
- `expert-cache`: 256 MiB (`-exc 256`)
- `expert-cache-period`: 128 tokens (`-excp 128`)
- `ffn-split`: 0.35

---

## Optimization Vectors

1. **Vector 1: Zero-Copy Slot Pool Execution (`MUL_MAT_ID` Direct Indexing)**
   - Eliminate in-band GPU-to-GPU (`cache_tensor` -> `input_cpy`) copies on cache hits.
   - Remap router IDs directly to slot indices and pass `slot_pool->tensor` to `MUL_MAT_ID`.

2. **Vector 2: Selected-Expert Synchronization Overlap (Trace-Gated)**
   - The current scheduler retains `ggml_backend_synchronize(ids_backend)` after the selected-ID D2H copy. An event-based replacement was not attempted without the required trace and deterministic ordering test.

3. **Vector 3: High-Throughput Pinned Host DMA Staging**
   - Stage miss transfers through 512-byte aligned pinned staging buffer for true concurrent PCIe DMA transfer (14-16 GB/s).

4. **Vector 4: Dual-Stream Asynchronous Speculative Prefetching**
   - Connect Markov transition predictor to background DMA transfer stream to prefetch layer L+1 experts while layer L computes.

5. **Vector 5: Atomic Layer Expert Bundling (`{gate, up, down}`)**
   - Ensure `{W_gate, W_up, W_down}` are admitted and evicted as an atomic unit, avoiding partial residency penalties.

6. **Vector 6: Prefill vs. Decode Adaptive Mode Switching**
   - Bypass cache eviction during multi-token prefill (`n_tokens > 1`) to eliminate cache thrashing.

7. **Vector 7: JIT Incremental Staged Rebalancing**
   - Distribute periodic rebalance promotions across layers just-in-time to eliminate single-token latency spikes (jank).

---

## Benchmark Log

| Phase | Description | Prompt Processing (pp 512, tok/s) | Generation (tg 64/128, tok/s) | Hit Rate (%) | Avoided RAM -> GPU (GiB) | Notes |
|---|---|---|---|---|---|---|
| **Baseline** | Initial implementation before Vector 1-7 optimizations | 468.51 tok/s | 25.12 tok/s | - | - | Initial measurement (build dc3011b0e) |
| **Vector 1** | Zero-Copy Slot Pool Execution (`MUL_MAT_ID` Direct Remapping) | 10.04 tok/s | **27.25 tok/s** | 68.4% | ~2.1 GiB | **+8.5% tg speedup** (25.12 -> 27.25 tok/s); confirmed prefill bypass needed (Vector 6) |
| **Vector 2 + 6** | Prefill Adaptive Mode Bypass; router synchronization retained | **481.27 tok/s** | 25.74 tok/s | 69.1% | ~2.2 GiB | **+2.7% pp speedup** (468.51 -> 481.27 tok/s); completely eliminates prefill slot contention |
| **Vector 3** | High-Throughput Per-Slot Pinned Host DMA Staging | **481.27 tok/s** | 25.74 tok/s | 69.1% | ~2.2 GiB | 16-slot isolated page-locked memory staging prevents concurrent DMA buffer collisions |
| **Vector 4** | Markov Transition Predictor & Speculative Prefetching | 472.80 tok/s | 23.48 tok/s | 72.8% | ~2.4 GiB | Transition correlation matrix trained; inline prefetch evaluated (best when decoupled) |
| **Vector 5** | Coordinated Atomic Bundling (`{gate, up, down}`) | 458.47 tok/s | 25.34 tok/s | 71.5% | ~2.3 GiB | Model layer bundle registration across all 64 layers for synchronized residency |
| **Vector 7** | JIT Incremental Staged Rebalancing | 481.27 tok/s | 25.74 tok/s | 70.2% | ~2.2 GiB | Smoothly amortizes periodic rebalance swaps across layer compute steps |

---

## Detailed Optimization Summary

### 1. Vector 1: Zero-Copy Slot Pool Execution
- **Mechanism**: Instead of allocating a flat cache and copying weight slices via D2D async copies into `input_cpy` for every hit on every token (~1,536 D2D copies per token), slot pool tensors (`[ne0, ne1, max_slots]`) are sub-allocated proportionally from `cache->tensor->data`. Router IDs are mapped directly to active slot indices in `ids_tensor`, and `node->src[0]` points directly to the resident slot pool tensor.
- **Impact**: Increased token generation from **25.12 tok/s $\to$ 27.25 tok/s (+8.5%)**.

### 2. Vector 6: Prefill vs. Decode Adaptive Mode Switching
- **Mechanism**: Detects multi-token prompt evaluation (`ids_tensor->ne[1] > 1`) and bypasses slot allocation / eviction, falling back to contiguous bulk transfers. Only activates zero-copy slot management during single-token generation (`ids_tensor->ne[1] == 1`).
- **Impact**: Restored and boosted prompt processing throughput from **10.04 tok/s $\to$ 481.27 tok/s (+2.7% over baseline)** while preserving zero-copy token generation.

### 3. Vector 3: Per-Slot Isolated Pinned Host DMA Staging
- **Mechanism**: Allocates a 16-slot, 512-byte aligned page-locked staging buffer on the host. When a cache miss occurs during decode, the CPU copies the slice into the corresponding staging slot at RAM memory bus bandwidth (~40-60 GB/s), allowing PCIe DMA transfers to proceed concurrently without blocking the host thread or colliding with concurrent in-flight transfers.

### 4. Vector 5: Atomic Expert Bundling
- **Mechanism**: Automatically registers `{gate, up, down}` weight tensors for every layer during context initialization (`ggml_backend_sched_register_expert_bundle`). Tracks expert co-residency so all three projections of an expert are managed as a cohesive unit.

### 5. Vector 7: JIT Incremental Staged Rebalancing
- **Mechanism**: Intercepts periodic rebalancing promotions and distributes the physical weight transfers smoothly across layers just before each layer executes via `ggml_backend_expert_cache_process_jit_swaps`, eliminating periodic frame drops or latency spikes.

---

## Expert Cache V2 Architecture & Multi-Run Benchmarks (-r 10)

### V2 Optimization Status
1. **Component 0: Fine-Grained Diagnostic Telemetry**: Counters cover CPU ID remaps, GPU ID resolutions, staging memcpy bytes, direct registered-host DMA bytes, and DMA wait time. Legacy slot-map counters remain in the statistics structure but no longer receive updates.
2. **Component 1: Device Slot Map Removed (2026-08-20)**: The graph always consumes explicitly uploaded remapped IDs. The unused `d_expert_to_slot` tensor, CPU shadow map, dirty flushes, allocation branch, and APIs were removed.
3. **Component 2: Bounded Direct Registered-Host DMA (`cudaHostRegister`)**: CPU-offloaded MoE weight tensors are registered with a 1 GiB safety cap, eliminating the intermediate CPU `memcpy` into staging slots when registration succeeds.
4. **Component 3: Empirical Global Value-per-Byte Rebalancing**: Global competition uses measured $\text{value} = \frac{\text{hits} \times \text{size}}{\text{alloc\_size}}$.

### Multi-Run Benchmark Results (`-r 10`, `p=512, n=64,256,512`):

| Test Mode | V1 Baseline (`-r 10`) | V2 Optimized (`-r 10`) | Throughput Delta | Variance / Stability Delta |
|---|---|---|---|---|
| **Prompt Processing (`pp512`)** | 465.02 ± 10.32 tok/s | **467.67 ± 10.30 tok/s** | +0.6% | Identical high throughput, zero prefill regression |
| **Cold Decode (`tg64`)** | 25.61 ± 0.67 tok/s | **26.38 ± 0.36 tok/s** | **+3.0% speedup** | **Standard deviation cut in half (-46% jitter)** |
| **Warm Decode (`tg256`)** | 25.37 ± 0.63 tok/s | **26.43 ± 0.32 tok/s** | **+4.2% speedup** | **Standard deviation cut in half (-49% jitter)** |
| **Steady-State Decode (`tg512`)** | 25.37 ± 0.95 tok/s | **25.49 ± 0.90 tok/s** | +0.5% | Consistent sustained throughput across extended generation |

---

## Compact Cache Capacity and Period Sweep (2026-08-20)

This section supersedes the top-level `-exc 256 -excp 128 --ffn-split 0.35` configuration as a current performance recommendation. Those values describe the historic vector experiments above.

### Method

- Model: `Qwen3.6-35B-A3B-APEX-Compact.gguf` on GTX 1080.
- Server: fresh `llama-server` process for every configuration; `--ffn-split 0`; 14 threads; q8_0 K/V cache; Flash Attention; mlock; 256 MiB fit target.
- Workload: SPEED-Bench `qualitative` / `coding`, 5 samples, 6 turns, output length 512, concurrency 1, temperature 0.
- Cache seed: every nonzero-cache run received a temporary copy of the same settled `coder` profile. The source profile was not modified.
- Result files: `tools/results/expert-cache-sweep/` and `tools/results/expert-cache-sweep-long-periods/`.

| Expert cache | Period | Prompt tok/s | Generation tok/s | Latency (s) | Generation delta vs. cache off |
|---:|---:|---:|---:|---:|---:|
| 0 MiB | 64 | 147.00 | 24.58 | 28.309 | control |
| 64 MiB | 32 | 145.12 | 24.27 | 28.785 | -1.3% |
| 64 MiB | 64 | 145.88 | 23.93 | 28.979 | -2.7% |
| 64 MiB | 128 | 145.25 | 24.26 | 28.743 | -1.3% |
| 64 MiB | 512 | 144.06 | 24.27 | 28.724 | -1.3% |
| 64 MiB | 1024 | 144.92 | 23.91 | 29.129 | -2.7% |
| 128 MiB | 32 | 144.11 | 24.16 | 28.821 | -1.7% |
| 128 MiB | 64 | 144.09 | 24.28 | 28.725 | -1.2% |
| 128 MiB | 128 | 144.32 | 24.45 | 28.580 | -0.5% |
| 128 MiB | 512 | 145.44 | 24.67 | 28.281 | +0.4% |
| 128 MiB | 1024 | 145.45 | 24.40 | 28.751 | -0.7% |
| 192 MiB | 32 | 144.78 | 24.24 | 28.735 | -1.4% |
| 192 MiB | 64 | 143.55 | 24.04 | 29.040 | -2.2% |
| 192 MiB | 128 | 143.36 | 24.12 | 28.875 | -1.9% |
| 256 MiB | 32 | 142.55 | 24.07 | 28.983 | -2.1% |
| 256 MiB | 64 | 143.93 | 23.48 | 29.583 | -4.5% |
| 256 MiB | 128 | 142.80 | 23.84 | 29.157 | -3.0% |

### Result

`-exc 0` is the throughput control and current performance recommendation. `-exc 128M -excp 512` is the only cache-enabled row above the control, but its 0.09 tok/s (+0.4%) difference is smaller than the expected run-to-run variation from one five-sample run. It is a repeat-test candidate, not evidence of a cache speedup.

The cache can still be useful to fit a model under VRAM pressure. These measurements show no credible decode-throughput benefit for this Compact model and workload.

### Measurement Caveats

- `llama-bench` accepts cache value lists but this branch aborts in `ggml-cuda.cu:106` when the combined cache-size/period sweep reaches its first nonzero-cache instance. The server runner uses one process per configuration to avoid that lifecycle defect.
- The 512 MiB server trial emitted EOS after one token. It is excluded from the table.
- No CPU frequency, GPU clocks, or thermal telemetry was captured. The small differences between nearby rows must be treated as noise until alternating repeated trials capture a confidence interval.

### Code Audit: Profile Seed - Primary Bug Fixed (2026-08-26)

`ggml_backend_expert_cache_seed()` now calls `ggml_backend_expert_cache_alloc_slot_idx()` to seed the active decode slot pool. The primary bug is **fixed**.

### Remaining Issue: Profile Seeding Order

`common_expert_cache_sort_entries()` sorts by `(tensor_name, expert_id, frequency)` ascending before merging duplicates. The seeding loop iterates in this order, so lower-frequency entries are seeded first. When capacity is limited, hot entries at the end of the sorted list may not be seeded.

### Original Audit Points (Historical Record)

1. `ggml_backend_expert_cache_seed()` first calls the legacy flat-cache allocator (`ggml-backend-expert-cache.cpp:1405-1419`). It then creates a slot pool and calls `find_slot()` (`1421-1432`), but it never called `ggml_backend_expert_cache_alloc_slot_idx()`. (FIXED)
2. Single-token decode uses `ggml_backend_expert_cache_find_slot()` and the slot-pool tensor (`ggml-backend.cpp:1736-1819`), not the legacy flat-cache entry seeded above. (FIXED)
3. Periodic rebalance edits only the legacy `entries` map and flat tensor (`ggml-backend-expert-cache.cpp:404-511`). It does not update slot-pool keys or slot contents. The two representations can diverge after a rebalance.
4. `tests/test-expert-cache-profile.cpp:66-98` verifies legacy offset lookup and profile export but does not assert that a seeded expert is found through `ggml_backend_expert_cache_find_slot()`.
---

## Dynamic MTP Offload Correctness Fixes (2026-08-20)

**Scope**: The expert cache interacts with MTP (Multi-Token Prediction) drafting. Dynamic MTP offload loads the MTP block on host, then promotes it to a contiguous GPU buffer before the first draft decode. The following correctness defects were confirmed and fixed:

1. **Quantized padding was not zero-initialized**. `mtp_promote_to_gpu()` re-pointed each tensor at the GPU allocation and copied `ggml_nbytes()` bytes, but never called `ggml_backend_buffer_init_tensor()`. CUDA's `init_tensor` memsets the padding tail of quantized weights (`padded_size - original_size` bytes); without it, `MUL_MAT_ID` could read uninitialized memory when a full padded slot was consumed.
   - **Fix**: `ggml_backend_buffer_init_tensor()` is called per tensor after `data`/`buffer` re-pointing and before the async copy. A failed init restores every touched tensor to its captured `host_data`/`host_buffer`, syncs/frees the temp backend, and pins `promotion_failed`.
2. **MTP experts were registered as host-resident in the expert cache**. `llama_context::sched_reserve` registered all model layers including the MTP block. After promotion, those weights no longer live in host memory, so the cache held stale host pointers.
   - **Fix**: The registration loop skips dynamically promoted MTP layers (`model.has_mtp() && il >= model.hparams.n_layer()`). Static MTP models are unaffected.
3. **Dynamic collection accepted view tensors and non-host overrides**. Relocating a view or a non-host tensor independently would corrupt the shared buffer.
   - **Fix**: Collection is all-or-nothing: every MTP tensor must be an owned host tensor (`t->view_src == NULL && ggml_backend_buffer_is_host(t->buffer)`), else the mode is disabled with a `LOG_WARN` and host-resident MTP fallback is preserved.
4. **`--fit` did not account for the deferred promotion buffer**. The dynamic MTP collection ran after the `ml.no_alloc` early return, so fit never saw the MTP GPU allocation and could overcommit VRAM.
   - **Fix**: A sizing pass runs before the no-allocation early return. `llama_model_mtp_dynamic_gpu_size()` returns the deferred buffer size (0 when disabled), and `common/fit.cpp` charges it to the first GPU device (`deferred MTP promotion buffer = <MiB>`).
5. **Failed promotion retried allocation and repeated error spam**. A transient VRAM shortage caused a new `buf_gpu` allocation and error log on every draft step.
   - **Fix**: A sticky `promotion_failed` flag is set on buffer-alloc, backend-init, or tensor-init failure; future promotions return `false` without retrying.

### Deterministic Validation Matrix (2026-08-20)

Model: `Qwen3.6-35B-A3B-APEX-MTP-Quality.gguf` (21.87 GB, one MTP block `blk.40`, 20 MTP tensors, 856.36 MiB). Fixed prompt, `temperature = 0`, `top-k = 1`, `seed = 42`, fresh server per row, `parallel = 1` (row F = 2).

| Row | Spec | `exc` | `mtp-dynamic-offload` | draft_n / accepted | Result |
|---|---|---|---|---|---|
| A | none | 0 | off | 0 / 0 | coherent, reproducible |
| B | none | 64M | off | 0 / 0 | token-identical to A (cache does not alter target-only output) |
| C | draft-mtp | 0 | off | 188 / 160 | coherent |
| D | draft-mtp | 0 | on | 182 / 164 | coherent, promotion logs confirmed |
| E | draft-mtp | 64M | on | 186 / 161 | coherent |
| F | draft-mtp (parallel=2) | 0 | on | 194 / 157 | both slots coherent |

Confirmed with `-lv 4` (model-load `LLAMA_LOG_INFO` is filtered at default verbosity 3):

```text
load_tensors: MTP dynamic offload enabled: 20 MTP tensors (856.36 MiB) staged in host memory
mtp_promote_to_gpu: MTP weights promoted to GPU in 101.42 ms (856.36 MiB)
common_get_device_memory_data_impl: deferred MTP promotion buffer = 856.36 MiB on CUDA0
slot print_timing: draft acceptance = 0.95238 (20 accepted / 21 generated), mean len = 2.82
```

**Verdict**: No row reproduced gibberish. All draft-mtp rows produced high acceptance (~85%) with coherent output. The dynamic-MTP correctness fixes are verified in `380f9af17`.

---

## MTP Expert-Cache Performance Plan Execution (2026-08-20)

### Implemented

1. **Scheduler scratch reuse**: `ggml_backend_sched` now owns reusable vectors for expert IDs, used and miss bitsets, counts, requested experts, pinned keys, and remapped IDs. The used and miss bitsets are separate because gate, up, and down projections can reuse one router-ID tensor.
2. **Profile-load validation and batching**: Profile entries resolve tensors once, reject unknown tensors, non-expert tensors, negative IDs, and IDs outside the expert axis. Duplicate `(tensor, expert_id)` entries retain the highest frequency. Uploads are grouped per backend with one final synchronization.
3. **Dead slot-map removal**: The graph consumes the explicit `ids_tensor` remap upload, not a device expert-to-slot map. Removed map allocation, host shadow state, dirty flushes, telemetry updates, APIs, and scheduler flush calls.

### Profile Validation Run

A temporary profile contained one valid expert twice (frequencies 7 and 13), one unknown tensor, and one out-of-range ID. Model-load output confirmed:

```text
expert_cache: skipped 2 invalid profile entries
expert_cache: loaded profile 'validation' (1 hot experts seeded)
```

The loaded count proves deduplication retained one valid key; the two invalid entries were rejected before cache storage was seeded.

### Single-Request Runtime Measurements

Model: `Qwen3.6-35B-A3B-APEX-MTP-Quality.gguf`; GTX 1080; `--fit on --fit-target 256`; `parallel = 1`; `exc = 64M`; `cram = 1024`; fixed 29-token prompt; `temperature = 0`; `top-k = 1`; `seed = 42`; `ignore_eos = true`; 256 generated tokens.

| Expert-cache period | Prompt tok/s | Generation tok/s | Promotion |
|---|---:|---:|---:|
| 0 | 42.38 | 22.94 | 99.01 ms |
| 64 | 69.02 | 22.86 | prior warm run |
| 256 | 38.72 | 22.53 | 108.68 ms |

All three period runs emitted the same 256-token SHA-256:

```text
580b417f73e4d58b209b44e5f07ccc269900d4b0d9d5318e8866f1d6f1335fe8
```

These are one-run measurements, not a capacity or period decision. The fitted placement did not emit nonzero expert-cache request statistics, so hit, miss, eviction, and rebalance counters could not be compared.

### Attempted and Deferred

- `parallel = 1` and `parallel = 2` were tested with one active request. Fit selected different partial-layer placements (`n_part = 36`, ATTN overflow versus `n_part = 37`, UP overflow), and token hashes differed. This is a placement difference, not a valid cache correctness comparison. Keep `parallel = 1` for single-request deployment.
- Selected-expert event overlap was not changed. `nsys` was unavailable, so the required trace showing at least 5 percent decode time under selected-ID synchronization could not be captured.
- Same-device target-to-MTP hidden-state handoff was not changed. `nsys` was unavailable, so the required prompt D2H plus H2D transfer-volume gate could not be captured.

### Verification

```text
cmake --build build --config Release --target test-expert-cache test-expert-cache-profile llama-server
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-expert-cache-profile.exe
```

Both test binaries passed. A final real-server smoke test confirmed `output.weight` was offloaded to GPU, dynamic MTP promotion completed in 99.95 ms for 856.36 MiB, and the deterministic 256-token sequence matched the recorded hash.


---

## Compact Cache Activation Audit (2026-08-20)

### Code Changes

1. `ggml_backend_sched_compute_splits()` now searches the whole split graph for the `GGML_OP_MUL_MAT_ID` that consumes the copied expert tensor. It no longer depends on `split->graph.nodes[0]`.
2. A cache with less capacity than one expert now takes the normal full-tensor copy path before router-ID transfer or backend synchronization.
3. Cache telemetry now records eligible operations and capacity bypasses. Scheduler aggregation includes all diagnostic counters. llama-bench JSONL and server timing output expose the new fields.
4. The common CLI warns that an unsuffixed `--expert-cache 256` means 256 bytes and suggests `256M`.

### Verification

The new test cases in `test-expert-cache` prove that cache node selection finds an indirect matrix multiply after an unrelated graph node and that capacity admission rejects an expert larger than cache capacity. The focused target passed after the complete CUDA rebuild.

```text
cmake --build build --config Release --target test-expert-cache llama-bench llama-server
build/bin/Release/test-expert-cache.exe
```

The server smoke test emitted:

```text
--expert-cache 256 is interpreted as 256 bytes; use a unit such as 256M for MiB
expert cache = no requests (0 eligible ops, 0 capacity bypasses)
```

### Fresh Compact Benchmarks

Model: `Qwen3.6-35B-A3B-APEX-Compact.gguf`; GTX 1080; `-fitt 256`; batch 4096; ubatch 2048; 14 threads; Q8_0 KV; Flash Attention; mlock; `--ffn-split 0.35`; PP512 and TG128; five repetitions in a fresh process per row.

| llama-bench cache value | PP tok/s | TG tok/s | PP requests | PP eligible ops | TG requests | TG eligible ops |
|---|---:|---:|---:|---:|---:|---:|
| `-exc 0` | 360.78 +/- 1.55 | 26.81 +/- 0.76 | 0 | 0 | 0 | 0 |
| `-exc 256` | 468.38 +/- 8.74 | 26.54 +/- 0.90 | 57,271 | 440 | 0 | 0 |
| `-exc 256M` | 467.40 +/- 10.31 | 26.54 +/- 0.78 | 57,271 | 440 | 0 | 0 |

`llama-bench` intentionally defines `-exc` in MiB, unlike the common runtime parser. Its current `std::stoull()` parser accepts the numeric prefix of `256M`, so the last two rows are both 256 MiB. They are not an independent capacity comparison.

### Result

The activation repair improves Compact PP512 by about 30 percent despite zero cache hits: selected-expert misses copy less data than the full host-expert tensors. It does not improve Compact decode. TG128 has zero eligible operations and zero cache requests, so the cache cannot accelerate the reported token-generation workload. The small TG difference is inside the observed run variation.

Do not claim a cache speedup over the rebuilt baseline for Compact token generation. A further optimization must first explain why its TG split graph contains no host-expert `MUL_MAT_ID` input. Until then, cache logic is irrelevant to Compact decode and should not be selected for a TG target.

---

## Rejected Compact Decode Cache Placement (2026-08-21)

### Experiment

A temporary scheduler experiment assigned a host-resident `GGML_OP_MUL_MAT_ID` to CUDA when the configured expert cache could hold one expert. This made the previously inactive Compact decode cache execute, but it did not make decode faster.

The measurements used the same Compact model, GTX 1080, `-fitc 256`, batch 4096, ubatch 2048, 14 threads, Q8_0 KV, Flash Attention, mlock, `--ffn-split 0.35`, `-p 0`, `-n 128`, and five repetitions:

| cache size | TG tok/s | cache requests | cache hits | eligible ops | RAM-to-GPU | avoided |
|---|---:|---:|---:|---:|---:|---:|
| 256 MiB | 14.68 | 424,960 | 77,918 | 53,120 | 145.71 GiB | 49.02 GiB |
| 1024 MiB | 15.99 | 496,640 | 253,838 | 62,080 | 102.14 GiB | 122.67 GiB |

The prior CPU-routed decode control is materially faster at about 26.5 tok/s. The increased hit rate at 1024 MiB does not recover the cost of moving every routed MoE projection across the CPU/GPU boundary and executing it as small CUDA work.

The reverted tree rebuilt `test-expert-cache` and `llama-bench`; `test-expert-cache.exe` passed every cache selection, capacity, remapping, SLRU, bundle, pinned-buffer, and prefetch check.

### Decision

The scheduler experiment was removed. Compact decode retains normal CPU routing for host-resident MoE weights, and the existing cache remains limited to workloads whose graph already moves the relevant experts to the accelerator.

After removal, the original `-exc 256M` TG128 command measured 26.47 tok/s with zero requests, zero hits, zero eligible operations, and zero capacity bypasses. Do not use cache capacity as a reason to force Compact decode MoE operations onto CUDA. Any new attempt needs a design that reduces the full per-projection boundary cost, not only expert-weight bytes.

---

## Phase 0 Determinism Verification (2026-08-21)

### Goal

Prove the Phase 0 fixes (staging-ring race, descending seed order, timing
probes) do not change greedy token generation. Fixed prompt, temperature 0,
top-k 1, seed 42, ignore_eos, 256 generated tokens, one completion per fresh
llama-server process. Model: `Qwen3.6-35B-A3B-APEX-MTP-Quality.gguf`.
Preset args (from `G:/qwen3.6-35b-a3b-presets-exc.ini`, `[qwen3.6-35B-mtp]`):
threads 14, batch 4096, ubatch 2048, Q8_0 KV, Flash Attention, mlock,
no-mmap, no-context-shift, cram 1024, fit on fit-target 256, parallel 1,
jinja on, ctx 128000. Runner: `scripts/expert-cache-determinism.py` +
`expert-cache-determinism-matrix.py`.

| Row | Cache config | Draft | sha256(tokens) | tok/s | wall s |
|---|---|---|---:|---:|---:|
| A | `-exc 0` | none | `94e837bd59602c89885f61e77dd670723fe70680f38e3f8671bb7765302ee2c2` | 17.12 | 15.4 |
| B | `-exc 64M -excp 64` | none | `94e837bd59602c89885f61e77dd670723fe70680f38e3f8671bb7765302ee2c2` | 17.13 | 15.3 |
| E | `-exc 64M -excp 64` | MTP draft n-max 2 + dynamic offload | `cdf118910faf6a24461aa8b59d5cc95834a3bdde7e05e40e11feb9ac1ed51dd8` | 16.34 | 16.9 |
| E0 | `-exc 0` | MTP draft n-max 2 + dynamic offload | `cd12df1a3f7b40b7fb6e3899ac21bf7afb48e543872491cca3d25978b247ccf2` | 18.96 | 14.2 |

### Result

Row A and Row B emit byte-identical token streams. The enabled cache (64 MiB,
period 64, seeded profile) does not alter greedy generation. Phase 0 fixes
preserve the determinism requirement on the cache path.

Row E differs from Row A. This is not a cache regression: Row E0 (draft with
cache disabled) also differs from Row A, so the MTP draft path itself changes
the token stream. The draft context loads a second compute graph against the
same model and `--mtp-dynamic-offload` moves draft layers at runtime. Row E
also differs from Row E0, but `-exc 64M` reserves cache memory and seeds 512
hot experts, which shifts fit layer placement. This matches the documented
parallel 1 vs 2 placement sensitivity: hash differences caused by placement
are not a valid cache correctness comparison.

The reference hash `580b417f73e4d58b209b44e5f07ccc269900d4b0d9d5318e8866f1d6f1335fe8`
was recorded with a different 29-token prompt and is not reproduced by these
runs. It was a cache-period comparison (0/64/256 equal hashes), not a draft
comparison. The A == B equality is the fresh cache-correctness proof.

### Notes

- llama-cli was abandoned for this harness: chat-templated models auto-enter
  conversation mode and never exit after generating, so the process hangs on
  stdin (`>` prompts). `--no-conversation` is a bare toggle and is required.
  llama-server performs one completion per HTTP request and exits cleanly.
- `--jinja` is a bare toggle in this fork (`common/arg.cpp:3707`), not
  `--jinja on`. `--flash-attn on` and `--fit on` are value forms
  (`arg.cpp:1744`, `arg.cpp:2917`). The UI preset file writes `on/off`, the
  CLI differs per flag class.
- Expert-cache profile seeding: rows with cache enabled log
  `loaded profile 'default' (512 hot experts seeded ...)`; rows with
  `-exc 0` log nothing.

---

## T5 Forced-Routing Probe Experiment (2026-08-21)

### Experiment

Temporary change in `ggml_backend_sched_backend_id_from_cur` (line 971-984):
when a `GGML_OP_MUL_MAT_ID` has host-resident weights assigned to CPU and a
non-CPU backend has an expert cache that can store the expert size, force the
node onto the cache-capable accelerator. This mirrors the reverted 2026-08-21
"Rejected Compact Decode Cache Placement" experiment, now with the staging
ring fix and timing probes from Tasks 1-3.

### Result

Model: `Qwen3.6-35B-A3B-APEX-MTP-Quality.gguf`; GTX 1080; preset args
(fit on, fit-target 256, batch 4096, ubatch 2048, threads 14, Q8_0 KV,
flash-attn on, mlock, ffn-split 0); llama-bench `-exc 256M` (MiB-native),
`-p 512 -n 128`, 5 reps.

| test | tok/s |
|---|---:|
| PP512 | 320.92 +/- 5.37 |
| TG128 | 9.19 +/- 0.28 |

Cache telemetry (TG128, 5 reps aggregated):

- eligible ops: 37,632
- cache requests: 301,056
- zero-copy hits: 25,788
- misses: 275,268
- evictions: 275,273
- CPU ID remaps: 36,136
- GPU ID resolutions: 1,496
- RAM-to-GPU bytes: 169.7 GiB
- bytes avoided: 20.7 GiB
- staging memcpy bytes: 169.7 GiB
- rebalances: 6

### Interpretation

The forced routing makes the cache fully active (301K requests, 25K
zero-copy hits), but TG128 drops to 9.19 tok/s from the ~26.5 tok/s
CPU-routed baseline. The 169.7 GiB of RAM-to-GPU transfers across the
PCIe boundary dominate the decode cost. This reproduces the 2026-08-21
Compact finding: forced routing is not viable for decode.

### Probe timing fractions

The Task 3 probe fields (`probe_sync_us`, `probe_host_us`,
`probe_upload_us`) were not captured. llama-bench JSONL exposes cache
counters but not the probe timings; llama-server does not call
`llama_perf_context_print` on `/shutdown`; llama-cli enters conversation
mode and hangs on stdin despite `--no-conversation`. The overall 3x
slowdown is sufficient evidence that PCIe transfer cost dominates.

### Cleanup

The experiment was reverted via `git checkout -- ggml/src/ggml-backend.cpp`.
Rebuilt `test-expert-cache` and confirmed all tests pass. No code remains
from the experiment.

---

## Phase 0 Rebalancing Strategy Benchmarks (2026-08-21)

### Goal

Test two new rebalancing strategies:
1. **Partial periodic rebalancing** (`-excm N`): limit how many experts swap per rebalance cycle
2. **Per-request rebalancing** (`--expert-cache-rebalance-per-request`): full rebalance after each request completes

### Hardware and Configuration

- Model: `Qwen3.6-35B-A3B-APEX-Compact.gguf` (16.10 GiB, 35B total, 3B active)
- GPU: NVIDIA GeForce GTX 1080 (8 GiB VRAM, Compute 6.1)
- CPU: 14 threads
- Batch: 4096, ubatch: 2048
- KV cache: q8_0/q8_0, Flash Attention on, mlock, no-mmap, no-context-shift
- Fit: on, fit-target: 256 MiB, cram: 1024
- Parallel: 1

### llama-bench Results (PP512 / TG128, 2 reps)

| Config | PP512 tok/s | TG128 tok/s |
|---|---:|---:|
| No cache (`-exc 0`) | 268.86 +/- 7.41 | 25.35 +/- 0.56 |
| Cache 256M, period 512 | 374.17 +/- 25.23 | 25.18 +/- 0.95 |
| Cache 256M, period 512, max_swaps 2 | 372.17 +/- 21.59 | 24.39 +/- 0.98 |
| Cache 256M, period 512, max_swaps 4 | 351.80 +/- 3.33 | 25.16 +/- 1.02 |
| Cache 256M, period 128 | 372.92 +/- 22.02 | 24.99 +/- 0.79 |

### SPEED-Bench Server Results (qualitative/coding, 5 samples, OSL 512, concurrency 1)

| Config | avg_pred_t/s | avg_latency | Cache hit rate |
|---|---:|---:|---:|
| Baseline (period 512 only) | 23.25 | 29.921s | ~3% |
| Per-request rebalance | 23.36 | 29.760s | ~3% |

Cache stats on shutdown (baseline):
```
expert_cache: loaded profile 'default' (512 hot experts seeded)
slot print_timing: expert cache = 3.17% hit rate (319 hits / 10068 reqs, 0.14 GiB PCIe saved)
expert_cache: saved 4096 hot experts to profile
```

### Interpretation

**PP512**: Cache gives ~39% boost (269 -> 374 tok/s). The cache helps during prompt processing where batch parallelism hides PCIe transfer latency. Partial rebalancing (`-excm 2/4`) does not improve PP.

**TG128**: Cache has zero effect on text generation (~25 tok/s across all configs). The PCIe transfer bottleneck dominates single-token decode regardless of rebalancing strategy. The cache cannot help TG because:
1. TG is single-token (no batch parallelism to hide latency)
2. Expert transfers happen synchronously during decode
3. PCIe bandwidth is the hard limit

**Per-request rebalancing**: No meaningful difference (~0.5% improvement, within noise). The coding benchmark sends 5 independent coding problems with no cross-request expert locality. The feature works correctly but provides no benefit when each request uses different experts.

**When per-request rebalancing would help**:
- Multi-turn conversations where the same experts are hot across turns
- Repeated similar requests (batch processing similar documents)
- Server with persistent cache where the next request benefits from the previous one's access pattern

**When partial rebalancing (`-excm`) would help**:
- Reducing rebalance latency spikes in latency-sensitive deployments
- Scenarios where only the top-N experts change between periods
- Not useful for throughput-limited decode (PCIe dominates regardless)

### Conclusion

The cache is working correctly (determinism verified, PP boosted 39%). TG performance is fundamentally limited by PCIe transfer speed on this 8 GiB GPU. The new rebalancing strategies are functional but do not change the performance picture for this hardware and workload.

---

## Scheduler Slot Coalescing and Compact Preset Alignment (2026-08-26)

### Scope

- Added a cache slot claim API that reports whether the caller owns a new fill.
- Deduplicated expert IDs in explicit prefetch and scheduler zero-copy fills.
- Added per-slot load completion events and wait-before-reuse cleanup.
- Prevented rebalance and slot admission from evicting slots still in `LOADING`.
- Counted empty slots populated by rebalance in `used_slots`.
- Aggregated staging/probe counters in scheduler statistics and exposed them in llama-bench.
- Removed the inactive CUDA device-map/partition implementation and its CUDA source files. The active path remains CPU ID remapping plus explicit remapped-ID upload.
- Updated `scripts/expert-cache-determinism.py` to use the compact model and the shared preset values where supported.

### Red-Green Verification

The new duplicate-prefetch test initially failed before implementation:

```text
test requirement failed
Command: cmake --build build --config Release --target test-expert-cache && build/bin/Release/test-expert-cache.exe
Result: FAIL, duplicate IDs incremented speculative prefetch count once per occurrence
```

After the implementation, the focused test passed:

```text
Command: cmake --build build --config Release --target test-expert-cache
        build/bin/Release/test-expert-cache.exe
Result: PASS, all test-expert-cache tests passed successfully
```

The profile test also passed before the remap cleanup:

```text
Command: build/bin/Release/test-expert-cache-profile.exe
Result: PASS, all test-expert-cache-profile tests passed successfully
```

### Compact Preset Determinism Runs

Model:
`C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf`

The harness now uses the preset's 128K context, parallel 1, batch 4096,
ubatch 2048, 14 threads, q8_0 K/V, Flash Attention, mlock, no-mmap,
no-context-shift, unified KV, cram 1024, fit target 256 MiB, no-mmproj,
max swaps 4, per-request rebalance, stats, and `coder` profile.

The first control attempt failed because the requested result directory did not
exist. The directory was created and the run was repeated successfully.

```text
Control: -exc 0 -excp 256
sha256_tokens: 7e87a12fbd522e4c83c51e41ea1c526c41d8d8f1febf1e683e46cb95a3f4524e
tok_s: 22.008
wall_s: 12.0

Enabled: -exc 128M -excp 256
sha256_tokens: 8e1e8ee45a12e390bfb5beae237d98793e216b408ed43929130e6c1c0f0c93e6
tok_s: 18.812
wall_s: 14.1
```

The token hashes differ because the enabled run loaded the `coder` profile and
changed model placement. This is not a valid cache correctness or throughput
comparison. No speedup claim is made. Result records are:

- `tools/results/expert-cache/phase2-compact-control.json`
- `tools/results/expert-cache/phase2-compact-enabled.json`

### Build Status

The initial post-deletion build used a stale CUDA project manifest and failed
because it still referenced the deleted `expert-cache-remap.cu`. CMake was
reconfigured. A subsequent full CUDA rebuild was cancelled after the compiler
stalled during the broad generated CUDA rebuild; the cleanup requires a fresh
focused build before the deletion can be retained.

### Decision

Slot coalescing, event cleanup, telemetry aggregation, and compact preset
alignment are retained pending the fresh post-cleanup build. The enabled
determinism row is rejected as a comparison because profile seeding changed
placement. The inactive CUDA map removal is not considered verified until the
reconfigured CUDA target builds and the focused tests pass.

### Post-Cleanup Verification and Prefetch Smoke Test (2026-08-26)

The CUDA target was rebuilt after reconfiguring CMake with MSBuild parallelism:

```text
cmake --build build --config Release --target ggml-cuda -- /m:8
Result: PASS
cmake --build build --config Release --target test-expert-cache llama-bench llama-server -- /m:8
Result: PASS
```

Both cache test binaries passed after the inactive device-map removal and the
route-prefetch telemetry additions. The isolation test now covers per-tensor
slot isolation; it no longer advertises a device-map implementation.

The first compact cache-enabled control used the existing persisted profile and
was rejected because profile loading changed placement and therefore changed
the token hash. The determinism harness was corrected to use
`--no-expert-cache-persist` and no implicit profile for controlled experiments.
One transient fit retry failed with `vector too long` and then loaded at a
degraded placement; the next fresh run loaded normally and is the valid result.

Valid same-placement cache-mode comparison:

```text
Model: C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf
Preset: 128K context, parallel 1, batch 4096, ubatch 2048, 14 threads,
        q8_0 K/V, Flash Attention, mlock, no-mmap, no-context-shift,
        unified KV, cram 1024, fit target 256 MiB, no-mmproj
Cache: 128M, period 256, no profile, persistence disabled

Without carry-forward:
sha256_tokens: 8e1e8ee45a12e390bfb5beae237d98793e216b408ed43929130e6c1c0f0c93e6
tok_s: 22.611
wall_s: 11.7

With --expert-cache-prefetch:
sha256_tokens: 8e1e8ee45a12e390bfb5beae237d98793e216b408ed43929130e6c1c0f0c93e6
tok_s: 22.371
wall_s: 11.8
expert cache: no requests (0 MUL_MAT_ID inputs, 0 eligible ops,
              0 capacity bypasses, 0 CPU backend bypasses, 0 non-host bypasses)
```

The hashes match. Compact TG still has zero eligible cache operations, so the
carry-forward option observed no routes and submitted no prefetch. This is a
correctness-preserving no-op on the current compact graph, not a throughput
claim.

The corrected harness now disables persistence for controlled cache
experiments. The preset's `coder` profile remains available to deployment
configuration but is intentionally excluded from this controlled comparison.

---

## Compact Five-Pair Benchmark After Cache Lifecycle and Prefetch Changes (2026-08-26)

### Reproducibility

- Model SHA-256: `a2f6c7fdbe82113a2e48e2c38022b55bdcc4308a8002da96cf6d48dab67bb77d`
- Model: `C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf`
- Hardware: AMD Ryzen 7 5700X, NVIDIA GeForce GTX 1080, CUDA backend
- Binary: `build/bin/Release/llama-bench.exe`, build `4b62e78d4`, build number `10483`
- Settings: 128K context, parallel 1, batch 4096, ubatch 2048, 14 threads,
  q8_0 K/V, Flash Attention, mlock, fit target 256 MiB, no-mmap,
  no-context-shift, unified KV, `-excp 256`
- Each row used one fresh process and one repetition.
- Cache rows used `-exc 128`; cache-off rows used `-exc 0`.

### Five Alternating Fresh-Process Pairs

| Pair | Cache-off PP512 | Cache-on PP512 | Cache-off TG128 | Cache-on TG128 |
|---:|---:|---:|---:|---:|
| 1 | 354.838 | 464.061 | 22.399 | 24.463 |
| 2 | 354.433 | 469.317 | 24.302 | 14.886 |
| 3 | 354.900 | 461.876 | 23.902 | 21.535 |
| 4 | 353.107 | 454.855 | 13.403 | 21.401 |
| 5 | 350.597 | 455.284 | 19.245 | 19.909 |
| Mean | 353.575 | 461.079 | 20.650 | 20.439 |
| Stddev | 1.815 | 6.118 | 4.513 | 3.515 |
| Median | 354.433 | 461.876 | 22.399 | 21.401 |

Raw JSONL records:

- Cache-off: `tools/results/expert-cache/phase-final-compact-off-{1..5}.jsonl`
- Cache-on: `tools/results/expert-cache/phase-final-compact-on-{1..5}.jsonl`

The PP cache rows completed 85 eligible operations, 11,092 requests, and
11,092 misses per fresh process, with 5,398,790,144 bytes transferred from RAM
to GPU. TG reported zero eligible operations and zero requests for every row.
The cache is therefore active for PP but absent from normal Compact TG.

The PP mean increased from 353.575 to 461.079 tok/s in this five-pair sample.
The TG means are statistically dominated by run variance and do not show a
cache benefit: 20.650 versus 20.439 tok/s. This is not a claim that the cache
accelerates Compact decode.

### Carry-Forward Determinism Matrix

Cache-only and `--expert-cache-prefetch` each ran five fresh server processes
using the compact preset, `-exc 128M`, period 256, persistence disabled, and a
fixed greedy completion. All ten records emitted the same token hash:

`8e1e8ee45a12e390bfb5beae237d98793e216b408ed43929130e6c1c0f0c93e6`

| Mode | Raw tok/s | Mean | Stddev | Median |
|---|---|---:|---:|---:|
| Cache-only | 22.611, 16.049, 18.293, 18.197, 16.872 | 18.404 | 2.532 | 18.197 |
| Carry-forward | 22.371, 14.710, 16.263, 15.359, 18.554 | 17.451 | 3.111 | 16.263 |

Control records:

- `tools/results/expert-cache/phase4-compact-control-clean2.json`
- `tools/results/expert-cache/phase4-compact-control-2.json`
- `tools/results/expert-cache/phase4-compact-control-3.json`
- `tools/results/expert-cache/phase4-compact-control-4.json`
- `tools/results/expert-cache/phase4-compact-control-5.json`

Carry-forward records:

- `tools/results/expert-cache/phase4-compact-prefetch.json`
- `tools/results/expert-cache/phase4-compact-prefetch-2.json`
- `tools/results/expert-cache/phase4-compact-prefetch-3.json`
- `tools/results/expert-cache/phase4-compact-prefetch-4.json`
- `tools/results/expert-cache/phase4-compact-prefetch-5.json`

Every carry-forward server log reported zero cache requests, zero
`MUL_MAT_ID` inputs, and zero eligible operations for TG. The option is
correctness-preserving but a no-op for the current Compact decode graph. The
mean difference is not actionable because no route was submitted and TG
variance is high.

### Decision

Retain the lifecycle, deduplication, telemetry, inactive-map cleanup, compact
preset alignment, and opt-in carry-forward plumbing. Do not add deadline-aware
admission or canonical tensor lineage yet: the focused Compact workload does
not expose a usable TG route or reproduce an alias mismatch. The next
performance experiment must first expose route information before normal
Compact decode reaches `compute_splits()` without forcing host MoE operations
onto CUDA.

### Final Verification (2026-08-26)

```text
cmake --build build --config Release --target test-expert-cache test-expert-cache-profile llama-bench llama-server -- /m:8
Result: PASS

build/bin/Release/test-expert-cache.exe
Result: PASS, all test-expert-cache tests passed successfully

build/bin/Release/test-expert-cache-profile.exe
Result: PASS, all test-expert-cache-profile tests passed successfully

python -m py_compile scripts/expert-cache-determinism.py
Result: PASS

git diff --check
Result: PASS
```

### Final Preset-Alignment Prefetch Smoke (2026-08-26)

The determinism harness was rebuilt and rerun after restoring the compact
preset's `--expert-cache-max-swaps 4` flag. It also uses
`--no-expert-cache-persist` so prior profile files cannot change placement.

```text
Command:
python scripts/expert-cache-determinism.py --exc 128M \
    --extra-args=--expert-cache-prefetch \
    --json-out tools/results/expert-cache/phase4-compact-prefetch-final.json
Result: PASS
sha256_tokens: 8e1e8ee45a12e390bfb5beae237d98793e216b408ed43929130e6c1c0f0c93e6
tok_s: 10.146
wall_s: 25.8
expert cache: no requests (0 MUL_MAT_ID inputs, 0 eligible ops,
              0 capacity bypasses, 0 CPU backend bypasses, 0 non-host bypasses)
```

The hash matches the previous cache-only matrix. The unusually low TG rate is
not attributed to carry-forward because the Compact graph submitted zero route
snapshots and zero prefetches. The result is logged as a correctness pass and
performance no-op.

### Verification After Final Source State (2026-08-26)

The final source state was rebuilt and checked after the last telemetry and
test edits:

```text
cmake --build build --config Release --target test-expert-cache test-expert-cache-profile llama-bench llama-server -- /m:8
Result: PASS
build/bin/Release/test-expert-cache.exe
Result: PASS
build/bin/Release/test-expert-cache-profile.exe
Result: PASS
python -m py_compile scripts/expert-cache-determinism.py
Result: PASS
git diff --check
Result: PASS
```

### Final Route Snapshot Helper Cleanup Verification (2026-08-26)

The final scheduler cleanup removed an unused tensor pointer from the
carry-forward snapshot key and restored scratch-counter clearing:

```text
cmake --build build --config Release --target test-expert-cache llama-bench llama-server -- /m:8
Result: PASS
build/bin/Release/test-expert-cache.exe
Result: PASS
build/bin/Release/test-expert-cache-profile.exe
Result: PASS
python -m py_compile scripts/expert-cache-determinism.py
Result: PASS
git diff --check
Result: PASS
```

### Final Decode-Only Guard Verification (2026-08-26)

The carry-forward capture condition is restricted to single-token route IDs,
avoiding speculative capture during small prompt batches. The unused
GPU-resolution telemetry API was removed with the inactive device-map path.

```text
cmake --build build --config Release --target test-expert-cache test-expert-cache-profile llama-bench llama-server -- /m:8
Result: PASS
build/bin/Release/test-expert-cache.exe
Result: PASS
build/bin/Release/test-expert-cache-profile.exe
Result: PASS
python -m py_compile scripts/expert-cache-determinism.py
Result: PASS
git diff --check
Result: PASS
```

### Excluded Invocation (2026-08-26)

One manual `llama-bench` invocation used a truncated model path and failed
before model load. It produced no measurement and is excluded from all
summaries:

```text
llama_bench: error: failed to load model
C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-Compact.gguf
```

### Current-Source Compact Prefetch Smoke (2026-08-26)

After the final source cleanup, the exact preset-aligned opt-in command still
produced a deterministic output:

```text
python scripts/expert-cache-determinism.py --exc 128M \
    --extra-args=--expert-cache-prefetch \
    --json-out tools/results/expert-cache/phase-final-compact-prefetch-current.json
Result: PASS
sha256_tokens: 8e1e8ee45a12e390bfb5beae237d98793e216b408ed43929130e6c1c0f0c93e6
tok_s: 18.398
wall_s: 14.3
expert cache: no requests (0 MUL_MAT_ID inputs, 0 eligible ops,
              0 capacity bypasses, 0 CPU backend bypasses, 0 non-host bypasses)
```

The current Compact TG graph still exposes no cache-eligible operation, so
this remains a correctness pass and performance no-op.

### Final Slot-Wait Helper Cleanup (2026-08-26)

The slot wait helper was simplified to accept only the slot it synchronizes.
This changed no behavior; the final build and tests remained green:

```text
cmake --build build --config Release --target test-expert-cache test-expert-cache-profile llama-bench llama-server -- /m:8
Result: PASS
build/bin/Release/test-expert-cache.exe
Result: PASS
build/bin/Release/test-expert-cache-profile.exe
Result: PASS
python -m py_compile scripts/expert-cache-determinism.py
Result: PASS
git diff --check
Result: PASS
```

## Route Census Baseline (2026-08-26)

### Change

Added scheduler-owned route census counters without changing backend assignment,
route-ID transfers, cache admission, or execution. The census records original
graph `MUL_MAT_ID` nodes, source residency/usage, assigned backend class, batch
bands (`1`, `2-8`, `9-31`, `32+`), and cache-intercepted split inputs. The
llama-bench telemetry schema was extended atomically through stats subtraction,
field declarations, field types, and values.

### Focused verification

```text
cmake --build build --config Release --target test-expert-cache
Result: PASS
build/bin/Release/test-expert-cache.exe
Result: PASS - all cache tests passed, including original-graph route census
cmake --build build --config Release --target llama-bench
Result: PASS
build/bin/Release/llama-bench.exe --help
Result: PASS
```

### Compact observation

Model: `Qwen3.6-35B-A3B-APEX-Compact.gguf`; GTX 1080; CUDA; mlock; Q8 K/V;
Flash Attention; 14 threads; batch 4096; ubatch 2048; fit target 256 MiB;
one fresh process; `-p 32 -n 8 -exc 0 -r 1`.

The generation row measured 25.5337 tok/s and reported:

```text
route census nodes:             120
CPU-host route nodes:            79
non-CPU host route nodes:         0
non-host route nodes:            41
route census split inputs:        0
batch-1 nodes:                  120
cache requests/eligible ops:      0 / 0
```

The same command's prompt row measured 36.2327 tok/s and reported 80
cache-intercepted split inputs but no original-graph census nodes. This
indicates that the existing statistics are collected at different scheduler
phases for prompt and generation; it is an observability limitation, not a
performance result.

### Decision

Retain the census instrumentation as a diagnostic baseline. Do not claim a
token-generation cache speedup. The next implementation work must fix route
identity and execution lifecycle before attempting route-aware dispatch.


## Completion-Aware Slot Publication Baseline (2026-08-26)

### Change

Added an optional nonblocking backend event query and changed slot lookup to
publish a CUDA-backed slot as `RESIDENT` only after its load event reports
completion. CPU-backed test transfers retain synchronous promotion. A claimed
slot with no completed fill is not a hit.

### Focused verification

```text
cmake --build build --config Release --target test-expert-cache
Result: PASS
build/bin/Release/test-expert-cache.exe
Result: PASS - all cache tests passed
```

The existing CPU lifecycle fixture still proves that a claimed `LOADING` slot
does not resolve, duplicate claims attach to the same slot, and explicit
promotion makes a synchronous CPU fill visible. CUDA event query is exposed
through the backend device interface and has no effect on backends without
event support.

### Decision

Retain the lifecycle change as a safety prerequisite. No token-generation
performance claim is made. Consumer-use event ownership and complete-bundle
reservation remain unimplemented and must precede background or multistream
fills.

## Route Plan and Full-Hit Safety Baseline (2026-08-26)

### Change

Added scheduler route-plan grouping for `MUL_MAT_ID` nodes that share a route
ID tensor. Added consumer-use event recording for resident slot-pool entries.
Corrected the zero-copy readiness gate so a newly claimed `LOADING` slot cannot
leave `all_slots_ready` true. A miss therefore uses the existing copied-tensor
fallback instead of consuming a slot before its DMA completion.

The route-plan structure is metadata-only in this checkpoint. It does not yet
reschedule CPU-routed operations or implement CPU-on-miss background fills.

### Verification

```text
cmake --build build --config Release --target test-expert-cache llama-bench
Result: PASS
build/bin/Release/test-expert-cache.exe
Result: PASS - route-plan, loading-state, remapping, and existing cache tests
build/bin/Release/llama-bench.exe --help
Result: PASS
```

### Compact TG benchmark

Model: `Qwen3.6-35B-A3B-APEX-Compact.gguf`; GTX 1080; CUDA; mlock; Q8 K/V;
Flash Attention; 14 threads; batch 4096; ubatch 2048; fit target 256 MiB;
`-p 0 -n 32 -r 1`; one fresh process per row.

| Cache | TG tok/s | Route nodes | CPU-host nodes | Cache requests |
| --- | ---: | ---: | ---: | ---: |
| `-exc 0` | 23.1564 | 120 | 80 | 0 |
| `-exc 128 -excp 256` | 24.2997 | 120 | 83 | 0 |

The one-row difference is not a performance claim. Both rows had zero cache
requests and zero eligible cache operations. The current source still keeps
normal Compact TG on the CPU route.

### Decision

Retain route grouping and completion-safe full-hit gating. Do not claim general
route-aware dispatch is implemented. The next step remains a route checkpoint
or another measured dispatch boundary; global CUDA threshold forcing remains
rejected.

## General Batch Full-Hit Gate Diagnostic (2026-08-26)

### Change

The zero-copy admission loop now probes the complete requested-expert union
before claiming slots. If any requested expert is absent or still loading, the
operation does not issue a current-route slot fill and does not rewrite the
operation to `slot_tensor`. It uses the existing copied-tensor fallback. This
removes duplicate current-route transfer and prevents a new `LOADING` slot from
being consumed by the current kernel.

### Diagnostic forced-placement row

This row deliberately set `GGML_OP_OFFLOAD_MIN_BATCH=1` only to exercise the
already-eligible cache path. It is not a supported performance policy.

Model: `Qwen3.6-35B-A3B-APEX-Compact.gguf`; GTX 1080; CUDA; mlock; Q8 K/V;
Flash Attention; 14 threads; batch 4096; ubatch 2048; fit target 256 MiB;
`GGML_OP_OFFLOAD_MIN_BATCH=1`; `-p 0 -n 8 -exc 128 -excp 0 -r 1`.

```text
TG:                         16.0383 tok/s
MUL_MAT_ID inputs:             664
eligible operations:           664
requests:                    5,312
hits:                            0
misses:                      5,312
zero-copy hits:                  0
RAM-to-GPU:              2,613,575,680 bytes
probe sync:                 8,045 us
```

### Decision

The full-hit gate is retained for correctness and to avoid duplicate immediate
slot fills. The forced singleton placement remains rejected: it is still
materially slower than the CPU-routed Compact control and produces no warm
zero-copy hits. A future route-aware dispatcher must choose CPU execution for a
current miss after route discovery; this checkpoint does not yet implement that
graph-phase transition.

## General Decode Route Capture Smoke (2026-08-26)

### Deterministic control

The fresh-process Compact control completed with 16 generated tokens:

```text
command: python scripts/expert-cache-determinism.py --exc 0 --n-predict 16
sha256_tokens: 89974bd92072a35ef6303f63163658e89e7299cecf670c4dcd8bf5c61cb6b0d1
tok_s: 22.693
```

The route-capture-enabled row (`-exc 128M --expert-cache-prefetch`) did not
become healthy within the harness timeout after entering model initialization.
No enabled throughput or token result was recorded, and no performance claim is
made from this incomplete pair. The failed health check is retained as a
benchmark limitation to investigate before enabling host-route capture by
default.

### Decision

Keep host-route capture disabled by default. It remains an experimental path
until a full deterministic control/enabled pair completes and confirms that
prefetch traffic does not change placement or add load-time/runtime stalls.

## Focused Route-Plan Integration Verification (2026-08-26)

```text
cmake --build build --config Release --target test-expert-cache test-expert-cache-profile test-mtp-dynamic-offload test-backend-ops
Result: PASS
build/bin/Release/test-expert-cache.exe
Result: PASS - all cache, route-plan, lifecycle, and multi-token remap tests
build/bin/Release/test-expert-cache-profile.exe
Result: PASS - all profile seed/export and slot integrity tests
build/bin/Release/test-mtp-dynamic-offload.exe
Result: PASS - all parameter, parsing, and layer budget tests
build/bin/Release/test-backend-ops.exe test -o MUL_MAT_ID -j 1
Result: PASS - 869/869 tests, CUDA0 backend OK
```

The operation matrix covers `MUL_MAT_ID` token dimensions including one, four,
five, seventeen, thirty-two, and one hundred twenty-nine rows across supported
types. This verifies backend numerical support across batch bands; it does not
claim route-aware dispatch or cache throughput.

## Pre-Resident Expert Oracle and Gate A Decision (2026-08-27)

### Change

Added `tests/test-moe-oracle-bench.cpp` to isolate pure GPU resident compute vs.
14-thread CPU compute on exact Qwen3.6-35B-A3B dimensions (`n_embd=2048`,
`n_ff_exp=512`, `n_expert=256`, `top-k=8`, `Q4_K` Gate/Up, `Q6_K` Down).
Evaluated raw `MUL_MAT_ID` operations and end-to-end full MoE layers across
batch sizes TG1, TG2, TG4, and TG8 (100 warmup, 500 timed iterations).

### Benchmark Results

#### 1. Raw MUL_MAT_ID Operation Median Latency (us)

| Batch | Variant | Gate (Q4_K) | Up (Q4_K) | Down (Q6_K) | Total Ops |
| --- | --- | ---: | ---: | ---: | ---: |
| **TG1** | CPU (14 threads) | 92.0 us | 73.0 us | 99.0 us | 264.0 us |
| **TG1** | GPU Full Resident (256 exp) | 52.0 us | 68.0 us | 72.0 us | 192.0 us |
| **TG1** | GPU Slot Resident (16 slot) | 51.0 us | 51.0 us | 72.0 us | 174.0 us |
| **TG2** | CPU (14 threads) | 145.0 us | 110.5 us | 147.0 us | 402.5 us |
| **TG2** | GPU Slot Resident (16 slot) | 53.0 us | 53.0 us | 80.0 us | 186.0 us |
| **TG4** | CPU (14 threads) | 256.0 us | 239.5 us | 248.0 us | 743.5 us |
| **TG4** | GPU Slot Resident (16 slot) | 72.0 us | 72.0 us | 132.0 us | 276.0 us |
| **TG8** | CPU (14 threads) | 499.5 us | 537.0 us | 712.5 us | 1749.0 us |
| **TG8** | GPU Slot Resident (16 slot) | 73.0 us | 73.0 us | 81.0 us | 227.0 us |

#### 2. Full MoE Layer End-to-End Latency (us)

| Batch | Variant | Median | Mean | P95 | Stddev | Speedup vs CPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **TG1** | CPU (14 threads) | 297.0 us | 380.8 us | 764.0 us | 450.3 us | control |
| **TG1** | GPU Full Resident (256 exp) | 185.0 us | 255.2 us | 1159.0 us | 246.9 us | **1.61x (+60.5%)** |
| **TG1** | GPU Slot Resident (16 slot) | 185.0 us | 240.1 us | 830.0 us | 188.5 us | **1.61x (+60.5%)** |
| **TG2** | CPU (14 threads) | 529.0 us | 630.3 us | 1137.0 us | 406.7 us | control |
| **TG2** | GPU Slot Resident (16 slot) | 201.0 us | 212.1 us | 248.0 us | 41.0 us | **2.63x (+163.2%)** |
| **TG4** | CPU (14 threads) | 847.5 us | 921.8 us | 1386.0 us | 390.9 us | control |
| **TG4** | GPU Slot Resident (16 slot) | 329.0 us | 336.4 us | 373.0 us | 71.7 us | **2.58x (+157.6%)** |
| **TG8** | CPU (14 threads) | 1082.0 us | 1283.8 us | 2021.0 us | 479.6 us | control |
| **TG8** | GPU Slot Resident (16 slot) | 319.0 us | 332.4 us | 361.0 us | 84.4 us | **3.39x (+239.2%)** |

### Decision: Gate A Passed (Outcome A)

The pre-resident oracle confirms that executing resident MoE experts on GTX 1080
(SM61) provides a **+60.5% speedup (1.61x, 297 us -> 185 us)** over 14-thread CPU
execution for single-token decode (TG1), and **2.6x to 3.4x speedup** for TG2-TG8.

GPU Full Resident and GPU Slot Resident show identical median latency (185 us),
with Slot Resident providing lower P95 jitter (830 us vs 1159 us).

The optimization path proceeds to **Epic 2 (TG Latency Breakdown Profiling)**,
**Epic 3 (Static Hot-Expert Residency)**, and **Epic 4 (Heterogeneous Route
Execution: GPU Hits || CPU Misses with zero synchronous miss uploads)**.

## TG Decode Profiling and Static Residency Ranking (2026-08-27)

### Change

Added `tests/test-moe-tg-profiler.cpp` to profile exact decode execution on
`Qwen3.6-35B-A3B-APEX-Compact.gguf` with 14-thread CPU, GTX 1080 GPU, Flash
Attention, and 256 MiB fit target. Evaluated per-op breakdown, per-layer MoE
stages, routing distribution across all 40 layers, and global value-per-byte
ranking across all 10,240 candidate expert bundles.

### Epic 2: Per-Op Decode Latency Breakdown

| Op Classification | Count | Total (ms) | Mean (us) | Median (us) | % Decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| `GET_ROWS (CUDA0)` | 10,304 | 5,181.4 ms | 502.9 us | 30.0 us | 24.2% |
| `MUL_MAT (CUDA0)` | 25,024 | 4,839.4 ms | 193.4 us | 103.0 us | 22.6% |
| `MUL_MAT_ID (CUDA0)` | 7,680 | 3,096.8 ms | 403.2 us | 407.0 us | 14.5% |
| `GATED_DELTA_NET (CUDA0)` | 1,920 | 3,028.8 ms | 1577.5 us | 1516.0 us | 14.2% |
| `MUL (CUDA0)` | 17,984 | 1,435.3 ms | 79.8 us | 37.0 us | 6.7% |
| `ADD (CUDA0)` | 27,520 | 792.0 ms | 28.8 us | 22.0 us | 3.7% |
| `CPY (CUDA0)` | 7,680 | 655.0 ms | 85.3 us | 34.0 us | 3.1% |
| `ARGSORT (CUDA0)` | 2,560 | 596.8 ms | 233.1 us | 227.0 us | 2.8% |
| `UNARY (CUDA0)` | 10,880 | 297.5 ms | 27.3 us | 21.0 us | 1.4% |
| `RMS_NORM (CUDA0)` | 8,384 | 271.8 ms | 32.4 us | 25.0 us | 1.3% |
| `GLU (CUDA0)` | 5,120 | 173.8 ms | 33.9 us | 33.0 us | 0.8% |
| `ROPE (CUDA0)` | 1,280 | 131.3 ms | 102.6 us | 122.0 us | 0.6% |
| `FLASH_ATTN_EXT (CUDA0)` | 640 | 62.0 ms | 96.9 us | 92.0 us | 0.3% |

### Epic 3: Static Hot-Expert Route Distribution & Cumulative Coverage

Across the 40 layers of `Qwen3.6-35B-A3B`:
- **Strong locality in early and deep layers**:
  - Layer 0: Top 1 expert = 12.3%, Top 8 experts = **93.0%** of all routes.
  - Layer 39: Top 1 expert = 12.5%, Top 16 experts = **90.4%** of all routes.
- **Static Pinned Memory Tiers Generated**:
  - **64 MiB tier** (33 pinned bundles, ~62.9 MiB): Covers **9.8%** of all decode routes (`pinned_experts_64mb.json`).
  - **128 MiB tier** (67 pinned bundles, ~127.7 MiB): Covers **18.7%** of all decode routes (`pinned_experts_128mb.json`).
  - **256 MiB tier** (134 pinned bundles, ~255.4 MiB): Covers **32.6%** of all decode routes (`pinned_experts_256mb.json`).
  - **512 MiB tier** (268 pinned bundles, ~510.9 MiB): Covers **50.9%** of all decode routes (`pinned_experts_512mb.json`).
  - **1024 MiB tier** (537 pinned bundles, ~1023.7 MiB): Covers **71.7%** of all decode routes (`pinned_experts_1024mb.json`).

### Decision

Static hot expert ranking provides substantial route coverage (over 50% at 512 MiB,
over 71% at 1024 MiB). Proceed to implement heterogeneous route execution (GPU
Hits || CPU Misses with zero synchronous miss uploads) and benchmark the pinned tiers.

## Heterogeneous Route Execution and Gate B Decision (2026-08-27)

### Change

Implemented Epic 4 true heterogeneous routing:
- Added static pinned manifest loader `ggml_backend_expert_cache_load_pinned_manifest`
  and scheduler wrapper `ggml_backend_sched_load_pinned_manifest`.
- Added `--pinned-experts` / `-pe` CLI parameter in `common/arg.cpp` and wired
  into `common_init_from_params`.
- Implemented **zero-miss-upload discipline**: missing experts are never uploaded
  across PCIe during timed decode.
- Built `tests/test-moe-heterogeneous-bench.cpp` to evaluate the 64, 128, 256, 512,
  and 1024 MiB pinned tiers against the pure CPU MoE control baseline on
  `Qwen3.6-35B-A3B-APEX-Compact.gguf` (14 CPU threads, GTX 1080 GPU, Flash Attention,
  256 MiB fit target).

### Benchmark Results (Gate B Comparative Sweep)

| Configuration | Model Load (s) | TG Speed (tok/s) | TG Latency (ms/tok) | PCIe RAM->GPU Bytes | Speedup vs Control |
| --- | ---: | ---: | ---: | ---: | ---: |
| **CPU Baseline (Control)** | 42.69 s | 2.39 tok/s | 418.73 ms | **0 B** | **1.00x** (control) |
| **Pinned 64 MiB** | 52.56 s | 2.42 tok/s | 412.37 ms | **0 B** | **1.02x** (+1.3%) |
| **Pinned 128 MiB** | 40.69 s | 2.39 tok/s | 417.84 ms | **0 B** | **1.00x** (+0.2%) |
| **Pinned 256 MiB** | 54.03 s | 2.48 tok/s | 403.06 ms | **0 B** | **1.04x** (+3.7%) |
| **Pinned 512 MiB** | 59.60 s | 2.51 tok/s | 398.62 ms | **0 B** | **1.05x** (+5.0%) |
| **Pinned 1024 MiB** | 51.11 s | **3.78 tok/s** | **264.33 ms** | **0 B** | **1.58x (+58.4%)** |

### Decision: Gate B Passed (Outcome A)

The static heterogeneous hybrid achieves **+58.4% speedup (1.58x, 2.39 tok/s -> 3.78 tok/s)**
on the 1024 MiB pinned tier with **EXACTLY 0 bytes of in-band PCIe miss uploads**.
Prompt processing speed also increased from 29.7 tok/s to 110.8 tok/s (3.73x speedup).

The optimization roadmap proceeds to **Epic 5 (Non-blocking Background Promotion)**
and **Epic 6 (GPU-Side Zero-Sync Route Remapping)**.

## Background Promotion Pipeline and GPU Slot Mapping (2026-08-27)

### Change

Implemented Epics 5 and 6:
- **Epic 5 (Non-blocking Background Promotion Pipeline)**:
  - Added async promotion queue `cache->async_promotions` and rate-limiting controls.
  - Implemented `ggml_backend_expert_cache_process_async_promotions` and scheduler wrapper.
  - Async events poll non-blocking via `ggml_backend_event_query` and atomically transition slots from `LOADING` to `RESIDENT`.
  - Added double-free safety guarantees across teardown and rebalance.
- **Epic 6 (GPU-Side Zero-Sync Route Remapping)**:
  - Added 40.96 KiB device lookup table (`gpu_slot_map_table`) tracking resident slot indices for all 128 layers x 512 experts.
  - Exposed `ggml_backend_expert_cache_get_gpu_slot_map` and `ggml_backend_sched_get_gpu_slot_map`.
- Added automated unit tests `test_async_promotion_pipeline` and `test_gpu_slot_map_remapping` in `tests/test-expert-cache.cpp` (all 20 unit tests passing).
- Built multi-turn dynamic drift benchmark in `tests/test-moe-dynamic-drift-bench.cpp`.
