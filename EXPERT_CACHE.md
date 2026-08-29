# Expert Cache Subsystem in llama.cpp

The **Expert Cache** provides high-performance heterogeneous inference for Mixture-of-Experts (MoE) models (such as Qwen MoE, DeepSeek, Mixtral, Gemma MoE, and OLMoE) when offloaded across CPU host memory and GPU/accelerator VRAM.

---

## 1. Current Status & Implementation Matrix (Updated 2026-08-29)

```text
CURRENT IMPLEMENTATION STATUS:
- Route-ready full-hit GPU sidecar: Implemented, tested, and measured.
- Route-ready one-miss serial heterogeneous execution: Implemented and tested.
- Other incomplete route-ready bundles: CPU-base execution.
- Multi-token prompt processing CPU backend decoupling: Implemented and tested.
- Windows WDDM and pre-Hopper GPU stability safeguards: Implemented.
- Timed decode expert-weight upload: 0 bytes in the active route-ready paths.
- Route-ready admission telemetry: full-hit, fallback, and 0/8-8/8 mask counters exposed to llama-bench.
- Fallback bundle ordering: the down node runs on the host after its activation input exists, never before.
- Event-driven dual-device concurrency: Staged.
```

For the Compact model's `top_k = 8` single-token route-ready workload, production dispatch is intentionally selective:
```text
0-6 hits / 2-8 misses -> CPU-base execution
7 hits   / 1 miss     -> serial GPU-hit plus CPU-miss heterogeneous execution
8 hits   / 0 misses   -> full GPU route-ready sidecar execution
```

The partial-hit oracle covers every 0-8 hit mask. Runtime profiling found the 1-6-hit heterogeneous cases slower than the CPU-base path, so they are not admitted by the production route-ready dispatcher.

- **Full-Hit Sidecar**: Uses persistent GPU graph storage and stable weight descriptors. Input, remapped IDs, graph execution, and output download are queued on the sidecar backend stream and synchronized once after the download.
- **One-Miss Heterogeneous Path**: Retains GPU execution for seven resident routes and CPU execution for one miss. It synchronizes the GPU stream after CPU miss work and before reading the GPU result, which preserves graph-replay ordering.
- **Prompt Processing (PP) Scaling**: Multi-token prompt batches use dynamic backend buffer allocation (`ggml_backend_alloc_ctx_tensors`), supporting arbitrary context lengths (tested to 32k+ tokens with 0 pool exhaustion).
- **Native Tooling**: `llama-bench` supports `-pe / --pinned-experts <path0,path1,...>` alongside `-exc`, `-excp`, and `-fitt`.
- **Background Promotion Pipeline**: Non-blocking asynchronous promotion worker streams emerging hot experts from host pinned RAM into device slot pools without stalling active decode steps.

### 1.1 Regression Safeguards (2026-08-29)

- Cache residency can place a registered decode `MUL_MAT_ID` on an accelerator without changing general `op_offload` placement. Prompt processing retains normal placement.
- A route-ready action executes the route-ID producer prefix, synchronizes its split backend, reads IDs once, and remaps every complete Gate/Up/Down slot bundle before any dependent GPU `MUL_MAT_ID` runs. The dispatch record follows the producer and all consumers across scheduler splits.
- Full route-ready hits execute through the GPU sidecar. A bundle with exactly one missing route uses the serial heterogeneous path. Other incomplete bundles remain on the CPU-base path. None of these paths uploads expert weights during timed decode.
- The full-hit sidecar test asserts one backend synchronization, four backend-stream uploads for an unfused bundle, one backend-stream output download, CPU-reference-equivalent output, and zero expert RAM-to-GPU bytes.
- Sidecar error handling synchronizes queued uploads before slot release and resets failed GPU-buffer initialization for a retry. Regression coverage includes a failed graph submission, a repeated allocation failure, and two route-ready bundles in one split separated by non-bundle nodes.


---

## 2. Core Architecture & High-Performance Principles

```
         Host RAM (CPU)                        Accelerator (e.g. CUDA / Metal / Vulkan)
+----------------------------+                +-----------------------------------------+
| Host Weights (All Experts) |                | Unified Expert Cache (Strict Hard Cap)  |
| [Exp 0][Exp 1]...[Exp N]   |                | [Slot Pool 0: gate/up][Slot Pool 1:down]|
+--------------+-------------+                +--------------------+--------------------+
               |                                                   |
               | Background DMA Stream (Non-blocking)              | Hit: Direct Zero-Copy Indexing
               | (Page-Locked Pinned Host Buffer)                  | (MUL_MAT_ID on slot_tensor)
               v                                                   v
        +-----------------------------------------------------------------+
        | Device Working Tensor (node->src[0] = slot_tensor)             |
        | [Slot 0 (Exp 4)][Slot 1 (Exp 12)][Slot 2 (Exp 31)]...           |
        +-----------------------------------------------------------------+
```

### 2.1 The Zero-Miss-Upload Discipline (Crucial Invariant)
In earlier naive forced-routing experiments, un-cached experts were synchronously transferred across the PCIe bus during the critical decode path (causing up to 169.7 GiB of miss uploads and dropping decode speed to 9.19 tok/s).

Under the **Zero-Miss-Upload Discipline**:
1. **Never upload weights in-band during decode**: When a route-ready bundle does not meet the GPU admission threshold, production keeps the bundle on the CPU-base path.
2. **Transfer activations instead of weights**: The admitted one-miss path transfers partial hidden state instead of expert weights.
3. **Zero timed expert-weight upload**: The admitted full-hit and one-miss paths record zero expert RAM-to-GPU bytes.

### 2.2 Understanding Layer-Fit vs. Expert-Cache Trade-Offs
- **Full GPU Layer**: Holds all 256 experts in VRAM. 100% of routes compute on GPU at ~300 GB/s with 0 CPU fallbacks.
- **Cached Layer**: Holds 2 to 16 experts in VRAM. Top-k routes with un-cached experts fall back to host CPU memory.
- **Rule of Thumb**: Allocate full GPU layers first. Use Expert Cache only for:
  - Remaining VRAM headroom that is too small to fit another full layer.
  - Giant models (DeepSeek 671B, Qwen 236B) where a single layer exceeds total GPU VRAM.

### 2.3 Static Hot-Expert Value-Per-Byte Ranking
Because expert routing follows strong temporal and semantic locality, a small fraction of experts account for the majority of activations:
$$\text{value\_per\_byte} = \frac{P(\text{route}) \times (T_{\text{CPU}} - T_{\text{GPU}})}{\text{bundle\_bytes}}$$

The built-in profiler ranks all candidate expert bundles and produces pinned manifests:
- `pinned_experts_1024mb.json`: 537 bundles (~1023.7 MiB) -> **71.7% route coverage** (+42.0% TG speedup in isolation)
- `pinned_experts_512mb.json`: 268 bundles (~510.9 MiB) -> **50.9% route coverage**
- `pinned_experts_256mb.json`: 134 bundles (~255.4 MiB) -> **32.6% route coverage**
- `pinned_experts_128mb.json`: 67 bundles (~127.7 MiB) -> **18.7% route coverage**
- `pinned_experts_64mb.json`: 33 bundles (~62.9 MiB) -> **9.8% route coverage**

These coverage values are historical aggregate individual-route coverage from the older arbitrary-partial-hit executor. They do not predict the current production dispatcher, which admits only 7/8-hit and 8/8-hit bundles. Regenerate manifests against complete route co-occurrence before using them with the current path.

### 2.4 Non-Blocking Background Promotion Pipeline
- Candidate expert promotions are dispatched asynchronously on dedicated background CUDA streams.
- Completion is polled via non-blocking `ggml_backend_event_query()`.
- Active decode steps never wait for background DMA transfers; tokens compute on CPU until promotion confirms completion.
- Promotion rate is bounded (e.g. 1-2 bundles per rebalance epoch) to prevent PCIe bus contention.

### 2.5 GPU-Side Zero-Sync Route Remapping
- A compact 40.96 KiB device lookup table (`gpu_slot_map_table`) stores the resident slot index for every layer and expert:
  $$\text{gpu\_slot\_map}[L, E] \in [0, \text{max\_slots}-1] \cup \{-1\}$$
- When all top-k experts for a layer are resident, the GPU executes `MUL_MAT_ID` directly with zero host CPU inspection and zero synchronization bubbles.

---

## 3. CLI Options and Configuration

| Parameter | CLI Flag | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Pinned Manifest** | `-pe <path>`, `--pinned-experts <path>` | `LLAMA_ARG_PINNED_EXPERTS` | `""` | Path to static pinned experts JSON manifest (e.g. `pinned_experts_1024mb.json`). |
| **Expert Cache Size** | `-exc <size>`, `--expert-cache <size>` | `LLAMA_ARG_EXPERT_CACHE` | `0` (disabled) | Size of VRAM cache (e.g. `1024M`, `512M`, `'auto'`). |
| **Rebalance Period** | `-excp N`, `--expert-cache-period N` | `LLAMA_ARG_EXPERT_CACHE_PERIOD` | `64` | Token interval between dynamic rebalance swaps. |
| **Max Swaps** | `-excm N`, `--expert-cache-max-swaps N` | `LLAMA_ARG_EXPERT_CACHE_MAX_SWAPS` | `-1` | Maximum experts swapped per rebalance step. |
| **Cache Stats** | `-excs`, `--expert-cache-stats` | `LLAMA_ARG_EXPERT_CACHE_STATS` | `false` | Print expert cache hit rate and avoided bytes on exit. |
| **Profile Name** | `-excr <name>`, `--expert-cache-profile <name>` | `LLAMA_ARG_EXPERT_CACHE_PROFILE` | `""` | Name of profile for persistent hot-expert caching. |
| **Persistence** | `--expert-cache-persist`, `--no-expert-cache-persist` | `LLAMA_ARG_EXPERT_CACHE_PERSIST` | `true` | Auto-save/load expert cache profiles to disk. |

---

## 4. Empirical Benchmark Sweep Results (Preliminary Baseline Context)

Hardware: NVIDIA GeForce GTX 1080 (SM61, 8 GB VRAM) | CPU: 14 Threads | Model: `Qwen3.6-35B-A3B-APEX-Compact.gguf`

### Regime A: Scientific Isolation (Fixed Dense Offload, 0 Full MoE Layers on GPU)
| Configuration | Model Load (s) | TG Speed (tok/s) | TG Latency (ms/tok) | PCIe RAM->GPU Bytes | Speedup vs Control |
|---|---:|---:|---:|---:|---:|
| **CPU Baseline (Control)** | 42.69 s | 2.59 tok/s | 386.10 ms | **0 B** | **1.00x** (control) |
| **Pinned 64 MiB** | 52.56 s | 2.62 tok/s | 381.68 ms | **0 B** | **1.01x (+1.2%)** |
| **Pinned 128 MiB** | 40.69 s | 2.65 tok/s | 377.36 ms | **0 B** | **1.02x (+2.3%)** |
| **Pinned 256 MiB** | 54.03 s | **3.67 tok/s** | **272.48 ms** | **0 B** | **1.42x (+42.0%)** |
| **Pinned 512 MiB** | 59.60 s | 3.55 tok/s | 281.69 ms | **0 B** | **1.37x (+37.1%)** |
| **Pinned 1024 MiB** | 51.11 s | **3.47 tok/s** | **288.18 ms** | **0 B** | **1.34x (+34.0%)** |

### Regime B: Auto-Fit Deployment Reality (`--fit -fitt 256`)
| Configuration | Pinned Manifest | Cache Size | PP Throughput (`pp512`) | TG Throughput (`tg128`) | Analysis |
|:---|:---|---:|---:|---:|:---|
| **Control Baseline** | none | 0 MiB | 334.38 tok/s | **18.36 ± 0.87 tok/s** | 11 full MoE layers offloaded to GPU. 0 CPU fallback. |
| **Pinned 64 MiB** | `pinned_experts_64mb.json` | 0 MiB | 342.35 tok/s | **18.06 tok/s** | 11 full layers + 33 pinned experts. |
| **Pinned 128 MiB** | `pinned_experts_128mb.json` | 0 MiB | **337.61 tok/s** | **18.56 ± 0.58 tok/s** | 11 full layers + 67 pinned experts (within statistical variance of control). |
| **Pinned 256 MiB** | `pinned_experts_256mb.json` | 0 MiB | 336.79 tok/s | **18.12 ± 0.13 tok/s** | 11 full layers + 134 pinned experts. |
| **Pinned 1024 MiB** | `pinned_experts_1024mb.json` | 0 MiB | 340.04 tok/s | **17.71 tok/s** | 8 full layers + 1024M pinned experts (displaces 3 full layers). |
| **Dynamic Cache 128 MiB** | none | 128 MiB | 324.55 tok/s | **16.39 tok/s** | 8 full layers + 128M dynamic LRU cache. |
| **Hybrid Preset** | `pinned_experts_1024mb.json` | 128 MiB | 326.85 tok/s | **16.18 tok/s** | 8 full layers + hybrid dynamic/pinned cache. |

### Regime C: Active Route-Ready Sidecar (Post-Parity, 2026-08-29)

This is the retained explicit-placement deployment result after restoring cacheless upstream scheduler parity.

- Model: `Qwen3.6-35B-A3B-APEX-Compact.gguf`; GTX 1080; 14 CPU threads.
- Workload: fresh `llama-bench` process per row with `-p 0 -n 512 -r 1 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -fa on -lm mmap -ngl 99 -ncmoe 40 -fitt 0`.
- Control: `-exc 0`.
- Cache: `-exc 3072 -excp 65536 -excm 0 -pe tools/results/expert-cache/active-sidecar/pinned_layer03_all_256_3g.json`.
- Manifest: 1024 static entries for expert IDs 0-255 in layers 0-3. It is a 3072 MiB tier with dynamic swaps disabled.
- Method: twenty alternating pairs, ten control-first and ten cache-first.

| Pairs | Control mean (tok/s) | Cache mean (tok/s) | Mean paired delta | Median paired delta | Positive pairs |
|:---|---:|---:|---:|---:|---:|
| 20 | 18.804934 | 20.340449 | **+8.176%** | **+8.330%** | **20/20** |

The combined paired-delta 95% Student-t interval is +7.467% to +8.885%. Cache-enabled rows averaged 20,340 requests and zero-copy hits, 894 route-ready actions, 80 dispatches, 20,480 classifications, and zero expert RAM-to-GPU bytes. Every control row recorded zero for those counters.

Raw result pairs are under `tools/results/expert-cache/post-parity-cache-matrix/` with prefixes `2026-08-29-post-parity-explicit-control-first-n512` and `2026-08-29-post-parity-explicit-cache-first-n512`.

### Regime D: Static Expert Profiles With Automatic Fit (Post-Parity, 2026-08-29)

The 3 GiB manifest holding all 256 experts for layers 0-3 was rerun under automatic fit: `-fitt 256 -exc 3072 -excp 65536 -excm 0 -pe tools/results/expert-cache/active-sidecar/pinned_layer03_all_256_3g.json`. Ten alternating fresh-process TG512 pairs measured 23.258087 tok/s cache-off and 10.835326 tok/s cache-on: **-53.367%** mean paired delta, **-52.658%** median, 0/10 positive pairs, 95% interval -55.071% to -51.662%. It engaged correctly, averaging 24,720 zero-copy hits, 1,091 route-ready actions, 80 dispatches, 20,480 classifications, and zero RAM-to-GPU bytes. Do not use this static profile with automatic fit. Raw rows use `2026-08-29-post-parity-fit256-static1024-*` in `tools/results/expert-cache/post-parity-cache-matrix/`.

The intended 1,024 MiB profile was also measured with `-fitt 256 -exc 1024 -excp 65536 -excm 0 -pe pinned_experts_1024mb.json`. Ten pairs measured 20.988435 tok/s cache-off and 21.130657 tok/s cache-on: +1.092% mean paired delta, -1.142% median, 5/10 positive pairs, and a -6.003% to +8.188% interval. The manifest loaded and seeded 666.77 MiB into 1,370 slots across four pools, but none of 17,408 classified route bundles reached the current 7/8-hit or 8/8-hit admission threshold. The legacy manifest spreads 537 entries across all 40 layers, only 8-18 expert IDs per layer. Its historical 71.7% figure is individual-route coverage from the older heterogeneous executor, not complete-bundle coverage for the current selective dispatcher. The zero request counter records zero executed cache tensors, not zero route lookups. Do not retain this profile without regenerating it for current automatic placement and bundle-level admission. Raw rows use `2026-08-29-post-parity-fit256-static1024m-*` in `tools/results/expert-cache/post-parity-cache-matrix/`.

### Regime E: Automatic-Fit Dynamic Expert Cache (Post-Parity, 2026-08-29)

The post-parity automatic-fit target-256 Compact TG512 matrix used `-p 0 -n 512 -r 1 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -fa on -lm mmap -fitt 256` with fresh processes. Each period has five control-first and five cache-first pairs. The cache rows use `-exc 128 -excm -1`; no pinned manifest.

| Rebalance period | Pairs | Control mean (tok/s) | Cache mean (tok/s) | Mean paired delta | Median paired delta | Positive pairs | 95% paired interval |
|:---|---:|---:|---:|---:|---:|---:|---:|
| 32 | 10 | 22.040287 | 22.764980 | **+3.305%** | **+3.637%** | 9/10 | +2.049% to +4.561% |
| 256 | 10 | 22.108924 | 22.349791 | +1.105% | +1.345% | 8/10 | -0.842% to +3.051% |

Retain the 128 MiB dynamic cache with `-excp 32` for this model, GPU, and automatic-fit workload. It averaged 249 requests, 249 zero-copy hits, 66 route-ready actions, 56 dispatches, 14,336 classifications, and zero RAM-to-GPU bytes. Period 256 is not retained because its paired interval includes zero.

Raw result pairs are under `tools/results/expert-cache/post-parity-cache-matrix/` with prefixes `2026-08-29-post-parity-fit256-period32-*` and `2026-08-29-post-parity-fit256-period256-*`.


---

## 5. Verification Tools & Acceptance Criteria

### Test Executables
- `build/bin/Release/test-moe-partial-hit-bench.exe`: Standalone partial-hit heterogeneous execution oracle and latency curve benchmark (Epic 1 / Milestone 1-2).
- `build/bin/Release/test-expert-cache.exe`: Comprehensive unit test suite covering 20 invariants (slot pools, non-blocking query, SLRU admission, pinned staging ring, async promotions, GPU slot mapping).
- `build/bin/Release/test-moe-oracle-bench.exe`: Gate A microbenchmark comparing isolated resident GPU vs CPU compute.
- `build/bin/Release/test-moe-tg-profiler.exe`: Op-level profiler and value-per-byte pinned manifest generator.
- `build/bin/Release/test-moe-heterogeneous-bench.exe`: Gate B comparative sweep runner benchmarking memory tiers.
- `build/bin/Release/test-moe-dynamic-drift-bench.exe`: Multi-turn topic drift benchmark evaluating dynamic background promotions.

### Partial-Hit Heterogeneous Acceptance Criteria Matrix
| Hit Mask | GPU Routes Executed | CPU Routes Executed | Numerical Tolerance | Status |
| :---: | :---: | :---: | :---: | :---: |
| 0/8 | 0 | 8 | Exact CPU Reference | Pending Verification |
| 1/8 | 1 | 7 | NMSE <= 0.0002, MeanRel <= 0.005 | Pending Verification |
| 2/8 | 2 | 6 | NMSE <= 0.0002, MeanRel <= 0.005 | Pending Verification |
| 3/8 | 3 | 5 | NMSE <= 0.0002, MeanRel <= 0.005 | Pending Verification |
| 4/8 | 4 | 4 | NMSE <= 0.0002, MeanRel <= 0.005 | Pending Verification |
| 5/8 | 5 | 3 | NMSE <= 0.0002, MeanRel <= 0.005 | Pending Verification |
| 6/8 | 6 | 2 | NMSE <= 0.0002, MeanRel <= 0.005 | Pending Verification |
| 7/8 | 7 | 1 | NMSE <= 0.0002, MeanRel <= 0.005 | Pending Verification |
| 8/8 | 8 | 0 | Exact GPU Slot Reference | Pending Verification |

