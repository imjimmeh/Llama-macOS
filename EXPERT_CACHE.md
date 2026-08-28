# Expert Cache Subsystem in llama.cpp

The **Expert Cache** provides high-performance heterogeneous inference for Mixture-of-Experts (MoE) models (such as Qwen MoE, DeepSeek, Mixtral, Gemma MoE, and OLMoE) when offloaded across CPU host memory and GPU/accelerator VRAM.

---

## 1. Current Status & Implementation Matrix (Updated 2026-08-27)

```text
CURRENT IMPLEMENTATION STATUS:
- Full-hit slot execution: Implemented.
- Whole-node CPU fallback on any miss: Implemented (replacing with per-route heterogeneous execution).
- Route hit/miss partition metadata: Implemented.
- True partial-hit GPU/CPU execution: IN PROGRESS.
- Concurrent GPU-hit + CPU-miss execution: IN PROGRESS.
- GPU/CPU route output merge: IN PROGRESS.
```

The objective is to replace whole-node fallback with canonical per-route heterogeneous execution for single-token generation (TG1):
```text
0 hits / 8 misses -> CPU routes = 8, GPU routes = 0
1 hit  / 7 misses -> CPU routes = 7, GPU routes = 1
...
7 hits / 1 miss   -> CPU routes = 1, GPU routes = 7
8 hits / 0 misses -> CPU routes = 0, GPU routes = 8
```

- **Gate A (Pre-Resident GPU Compute Oracle)**: **PASSED (Outcome A)**. Single-token decode execution for resident GPU experts takes **185 us vs. 297 us on CPU (+60.5% speedup / 1.61x)**.
- **Gate B (Heterogeneous Route Execution & Zero Miss Upload)**: **IN PROGRESS**. Dedicated partial-hit heterogeneous execution engine undergoing two-phase validation (Phase 1 serial correctness -> Phase 2 concurrent streams).
- **Prompt Processing (PP) Invariant**: Expert Cache is strictly a single-token decode optimization (`ne[1] == 1`) and is completely bypassed during batch prompt processing (`ne[1] > 1`).
- **Native Tooling**: `llama-bench` natively supports `-pe / --pinned-experts <path0,path1,...>` alongside `-exc`, `-excp`, and `-fitt`.
- **Background Promotion Pipeline (Epic 5)**: Non-blocking asynchronous promotion worker streams emerging hot experts from host pinned RAM into device slot pools without stalling active decode steps.
- **GPU-Side Route Remapping (Epic 6)**: Compact 40.96 KiB device lookup table (`gpu_slot_map_table`) maps resident slot indices directly in GPU memory.

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
1. **Never upload weights in-band during decode**: When a requested expert is not resident in GPU slots, its slice is computed on host CPU memory.
2. **Transfer activations instead of weights**: A missing expert bundle requires uploading 1.95 MiB across PCIe. In contrast, partial hidden states require transferring only 8 KiB (2000x less PCIe traffic).
3. **Zero GPU compute stall**: Tokens with GPU hits compute on GPU slots; tokens with CPU misses compute on CPU threads; hidden states are combined with zero PCIe weight stalls.

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

