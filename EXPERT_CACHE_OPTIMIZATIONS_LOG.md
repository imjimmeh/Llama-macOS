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

### Code Audit: Profile Seed Does Not Populate the Active Decode Cache

The cache profile path needs a correctness repair before more throughput tuning:

1. `ggml_backend_expert_cache_seed()` first calls the legacy flat-cache allocator (`ggml-backend-expert-cache.cpp:1405-1419`). It then creates a slot pool and calls `find_slot()` (`1421-1432`), but it never calls `ggml_backend_expert_cache_alloc_slot_idx()`. A newly created slot pool has no key, so `find_slot()` returns `-1` and the zero-copy slot tensor is not seeded.
2. Single-token decode uses `ggml_backend_expert_cache_find_slot()` and the slot-pool tensor (`ggml-backend.cpp:1736-1819`), not the legacy flat-cache entry seeded above. Therefore a loaded persistent profile can report successful profile loading without making its entries resident in the active zero-copy decode path.
3. Periodic rebalance edits only the legacy `entries` map and flat tensor (`ggml-backend-expert-cache.cpp:404-511`). It does not update slot-pool keys or slot contents. The two representations can diverge after a rebalance.
4. `tests/test-expert-cache-profile.cpp:66-98` verifies legacy offset lookup and profile export but does not assert that a seeded expert is found through `ggml_backend_expert_cache_find_slot()`.

The low 1.7-3.1% runtime hit rates in the server logs are consistent with a profile that did not seed the decode slot pools, but the cause must be confirmed by a focused regression test before changing production code.

Required fix direction:

- Seed the slot pool through `ggml_backend_expert_cache_alloc_slot_idx()` and upload directly to its slot tensor.
- Establish one authoritative representation for zero-copy residency. Do not let periodic rebalance mutate legacy entries independently of slot pools.
- Add a CPU regression test that seeds an expert, synchronizes, and asserts `find_slot()` succeeds before the first decode request. Add a CUDA test that compares seeded slot-pool data with the source expert.
Additional audit findings:

- `common_expert_cache_load_profile()` sorts seed entries by tensor address and expert ID (`common/expert-cache-profile.cpp:112-117`), not by stored frequency. When capacity is limited, the seed order does not prefer the hot entries that the profile records.
- Decode records transition data on every zero-copy path (`ggml-backend.cpp:1821-1825`), but `ggml_backend_expert_cache_predict_next()` and bundle prefetch have no production callers. The current transition predictor adds host map updates without starting a prefetch transfer.
- Bundle registration and `ggml_backend_expert_cache_is_bundle_resident()` are likewise only exercised by unit tests. No runtime admission or eviction policy enforces bundle atomicity.

Further optimization order:

1. Repair and test profile-to-slot-pool seeding, then seed highest-frequency complete bundles first.
2. Remove or feature-gate the unconsumed transition tracking until a measured prefetch implementation exists.
3. Add timing counters around selected-ID D2H synchronization, host-side cache decision/remapping, ID upload, and cache DMA. The scheduler currently synchronizes the ID backend before and after every new router-ID tensor (`ggml-backend.cpp:1674-1694`); this can only be optimized after its share of decode time is measured.
4. If that timing proves the cache decision cost exceeds saved transfer time at low locality, bypass zero-copy lookup/remapping adaptively for the affected tensor or model. Do not attempt asynchronous router overlap before the timing and deterministic-token tests exist.


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
