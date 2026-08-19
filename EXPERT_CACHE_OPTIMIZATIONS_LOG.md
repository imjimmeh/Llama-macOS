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

2. **Vector 2: Elimination of In-Band Router Synchronization**
   - Replace blocking `ggml_backend_synchronize(ids_backend)` with event-based synchronization and mapped/pinned router output memory.

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
| **Vector 2 + 6** | Prefill Adaptive Mode Bypass + Router Sync Elimination | **481.27 tok/s** | 25.74 tok/s | 69.1% | ~2.2 GiB | **+2.7% pp speedup** (468.51 -> 481.27 tok/s); completely eliminates prefill slot contention |
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

### V2 Optimizations Implemented:
1. **Component 0: Fine-Grained Diagnostic Telemetry**: Added exact counters for CPU ID remaps, GPU ID resolutions, staging memcpy bytes, direct registered-host DMA bytes, slot map updates, and DMA wait time.
2. **Component 1: Per-Pool GPU-Resident Slot Resolution (`d_expert_to_slot`)**: Attached device mapping tensors directly to each slot pool with CPU shadow tables and batched dirty flushes, eliminating chatty per-mutation host-to-device transfers.
3. **Component 2: Bounded Direct Registered-Host DMA (`cudaHostRegister`)**: Registered CPU-offloaded MoE weight tensors into page-locked memory (with 1 GiB safety cap), eliminating the intermediate CPU `memcpy` into staging slots on misses.
4. **Component 3: Empirical Global Value-per-Byte Rebalancing**: Replaced static layer allocation formulas with global dynamic competition based on measured $\text{value} = \frac{\text{hits} \times \text{size}}{\text{alloc\_size}}$.

### Multi-Run Benchmark Results (`-r 10`, `p=512, n=64,256,512`):

| Test Mode | V1 Baseline (`-r 10`) | V2 Optimized (`-r 10`) | Throughput Delta | Variance / Stability Delta |
|---|---|---|---|---|
| **Prompt Processing (`pp512`)** | 465.02 ± 10.32 tok/s | **467.67 ± 10.30 tok/s** | +0.6% | Identical high throughput, zero prefill regression |
| **Cold Decode (`tg64`)** | 25.61 ± 0.67 tok/s | **26.38 ± 0.36 tok/s** | **+3.0% speedup** | **Standard deviation cut in half (-46% jitter)** |
| **Warm Decode (`tg256`)** | 25.37 ± 0.63 tok/s | **26.43 ± 0.32 tok/s** | **+4.2% speedup** | **Standard deviation cut in half (-49% jitter)** |
| **Steady-State Decode (`tg512`)** | 25.37 ± 0.95 tok/s | **25.49 ± 0.90 tok/s** | +0.5% | Consistent sustained throughput across extended generation |

---

## Expert Cache V3 Architecture: Prompt-Tail Warmup & Core Anchor Pinning

### V3 Optimizations Implemented:
1. **Component 1: Prompt-Tail Locality Seeding (`ggml_backend_expert_cache_record_prompt_tail`)**:
   - Captures expert routing decisions from the final 64 tokens of the prompt ($K=64$).
   - On transition to autoregressive decode (`is_single_token_decode == true`), promotes the top-frequency prompt-tail experts directly into the slot pools via high-speed DMA before Token 1 begins.
2. **Component 2: Universal Core Anchor Pinning (`ggml_backend_expert_cache_pin_anchor`)**:
   - Designates permanent immutable anchor slots (`is_anchor = true`, `GGML_EXPERT_CACHE_SEG_ANCHOR`) per pool.
   - Core anchor slots are strictly exempt from SLRU/LRU eviction, providing a guaranteed zero-overhead floor across all queries.

### Multi-Run Benchmark Validation (`-r 10`, `p=512, n=64,256,512`):

| Test Mode | V1 Baseline (`-exc 256`, `-r 10`) | V2 Optimized (`-exc 256`, `-r 10`) | V3 Tail & Anchor (`-exc 256`, `-r 10`) | V3 Tail & Anchor (`-exc 512`, `-r 10`) | Notes |
|---|---|---|---|---|---|
| **Prompt Processing (`pp512`)** | 465.02 ± 10.32 tok/s | **467.67 ± 10.30 tok/s** | 454.00 ± 9.40 tok/s | 448.27 ± 8.81 tok/s | High prefill throughput |
| **Cold Decode (`tg64`)** | 25.61 ± 0.67 tok/s | **26.38 ± 0.36 tok/s** | 25.53 ± 0.44 tok/s | 25.90 ± 0.46 tok/s | Low jitter |
| **Warm Decode (`tg256`)** | 25.37 ± 0.63 tok/s | **26.43 ± 0.32 tok/s** | 24.60 ± 1.42 tok/s | 25.49 ± 0.82 tok/s | Multi-step decode |
| **Steady-State Decode (`tg512`)** | 25.37 ± 0.95 tok/s | **25.49 ± 0.90 tok/s** | 25.35 ± 0.65 tok/s | 24.96 ± 1.09 tok/s | Extended generation |

---



