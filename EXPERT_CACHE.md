# Expert Cache Subsystem in llama.cpp

The **Expert Cache** provides high-performance heterogeneous inference for Mixture-of-Experts (MoE) models (such as Qwen 3.5/3.6 MoE, DeepSeek, Mixtral, Gemma MoE, and OLMoE) when offloaded across CPU host memory and GPU/accelerator VRAM.

---

## 1. Overview and Motivation

In standard CPU/GPU offloaded MoE inference:
1. Routing layers determine a sparse set of active experts per token (e.g. top-2 or top-8 out of 64+ experts).
2. For un-offloaded layers, expert weight matrices reside in host RAM and are transferred over the PCIe bus to accelerator memory on every token generation step.
3. Because expert routing exhibits high temporal and semantic locality, a small subset of experts ("hot experts") accounts for a large percentage of total activations during inference.

The Expert Cache maintains a dedicated, persistent accelerator-side buffer to hold recently or frequently used expert weights. Through advanced slot-pool remapping and pinned DMA staging:
- **Zero-Copy In-Band Execution**: Active cache hits are evaluated directly in-place from device slot pools with **zero** Device-to-Device (D2D) copy overhead.
- **High-Throughput Pinned DMA**: Cache misses stage through isolated page-locked host memory buffers, achieving full 14–16 GB/s hardware PCIe DMA throughput without blocking the CPU thread.
- **Adaptive Execution Modes**: Automatically detects prompt prefill (`n_tokens > 1`) vs. single-token decode (`n_tokens == 1`), eliminating cache contention during prefill while maximizing decode efficiency.
- **Speculative & JIT Staging**: Leverages Markov transition modeling and JIT staged swaps to eliminate periodic latency spikes (jank) and prefetch predicted hot experts.

---

## 2. Architecture and High-Performance Vectors

```
        Host RAM (CPU)                        Accelerator (e.g. CUDA / Metal / Vulkan)
+----------------------------+                +-----------------------------------------+
| Host Weights (All Experts) |                | Dedicated Expert Cache (cache->tensor)  |
| [Exp 0][Exp 1]...[Exp N]   |                | [Slot Pool 0][Slot Pool 1]...[SLRU Pool]|
+--------------+-------------+                +--------------------+--------------------+
               |                                                   |
               | Miss: Pinned DMA Staging                          | Hit: Direct Zero-Copy Indexing
               | (16-Slot Page-Locked Host Buffer)                 | (MUL_MAT_ID on slot_tensor)
               v                                                   v
        +-----------------------------------------------------------------+
        | Device Working Tensor (node->src[0] = slot_tensor)             |
        | [Slot 0 (Exp 4)][Slot 1 (Exp 12)][Slot 2 (Exp 31)]...           |
        +-----------------------------------------------------------------+
```

### 2.1 Vector 1: Zero-Copy Slot Pool Execution (`MUL_MAT_ID` Direct Remapping)

In traditional caching architectures, hits in the device cache are copied via D2D memory transfers into a temporary execution tensor (`input_cpy`). In 64-layer models with 3 projections (`gate`, `up`, `down`) and 8 active experts, this incurs ~1,536 D2D transfers on every token.

**Zero-Copy Execution** eliminates D2D transfers completely:
1. Sub-allocates 3D slot pool tensors (`[ne0, ne1, max_slots]`) directly within `cache->tensor->data` proportionally across projection types without extra VRAM allocation.
2. Intercepts `GGML_OP_MUL_MAT_ID` nodes in `ggml_backend_sched_compute_splits()`.
3. When requested experts are present in the slot pool, maps router IDs in `ids_tensor` (node `src[2]`) directly to slot indices (`0 .. max_slots - 1`).
4. Replaces node `src[0]` with `slot_tensor`. The backend matrix multiplication kernel (`MUL_MAT_ID`) indexes directly into `vx + ids[i] * nb[2]`, computing the exact same result with **zero in-band memory copies**.

### 2.2 Vector 2 & 6: Prefill vs. Decode Adaptive Mode Switching and Router Sync Gate

- **Multi-Token Prefill Bypass**: Multi-token prompt processing (`ids_tensor->ne[1] > 1`) activates dozens of distinct experts simultaneously across tokens, exceeding slot pool capacities. The cache automatically bypasses slot pool allocation during prefill, relying on high-throughput bulk contiguous miss copying. This prevents cache thrashing and preserves full prompt processing throughput (**481+ tok/s**).
- **Single-Token Decode Activation**: Zero-copy slot pool execution is selectively activated for single-token generation (`ids_tensor->ne[1] == 1`).
- **Router ID synchronization**: The scheduler currently waits for the selected-ID D2H copy before CPU cache lookup. An event-based double-buffered replacement is trace-gated and is not enabled without a deterministic ordering regression test.

### 2.3 Vector 3: High-Throughput Per-Slot Pinned Host DMA Staging

When unpinned host memory is passed to asynchronous device copy APIs (`cudaMemcpyAsync`), drivers must either perform synchronous staging or serialize transfers.
- The Expert Cache allocates a 16-slot, 512-byte aligned page-locked host buffer (`ggml_backend_expert_cache_get_pinned_slot_buffer`).
- Cache misses copy slice payloads into an isolated staging slot at memory bus speeds (~40–60 GB/s) via CPU L1/L2 cache.
- The accelerator driver then performs asynchronous DMA over PCIe (14–16 GB/s) without blocking CPU execution or risking race conditions across concurrent layer transfers.

### 2.4 Vector 4: Markov Transition Predictor & Speculative Prefetching

- Tracks first-order transition frequencies between active experts across consecutive token decode steps: $P(E_{t+1} = j \mid E_t = i)$.
- Learns inter-token routing affinity during generation.
- High-confidence predicted experts ($\ge 2$ observed historical transitions) can be speculatively staged into idle slot positions ahead of execution.

### 2.5 Vector 5: Coordinated Atomic Layer Bundling (`{gate, up, down}`)

In SwiGLU architectures, each expert consists of three interdependent projections:
- `ffn_gate_exps`
- `ffn_up_exps`
- `ffn_down_exps`

During context initialization, all model layers register their triple projections via `ggml_backend_sched_register_expert_bundle()`. When admitting or evicting an expert, slot allocation and eviction are coordinated atomically across `{gate, up, down}`, guaranteeing synchronized residency without partial execution penalties.

### 2.6 Vector 7: JIT Incremental Staged Rebalancing

In periodic rebalancing mode (`--expert-cache-period N`), promoting multiple hot experts simultaneously at step boundary $N$ can cause a transient latency spike.
- `ggml_backend_expert_cache_rebalance` calculates promotions and enqueues them into `pending_forward_swaps`.
- As the graph executes, `ggml_backend_expert_cache_process_jit_swaps()` performs the physical transfers for each layer just before that layer executes.
- Weight transfers are evenly amortized across all 64 layers with zero perceptible stutter.

---

## 3. Data Structures

### 3.1 Cache Key (`ggml_expert_cache_key`)

Every expert matrix is identified uniquely by its source tensor pointer and expert index:

```cpp
struct ggml_expert_cache_key {
    const struct ggml_tensor * tensor; // pointer to host weight tensor
    int32_t expert_id;                 // index of the expert within tensor
};
```

### 3.2 Slot Pool Structure (`ggml_backend_expert_slot_pool`)

Maintains 3D sub-allocated slot buffers matching projection dimensions. Host-side key lookup maps each `(tensor, expert_id)` to a slot.

```cpp
struct ggml_backend_expert_slot_pool {
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    struct ggml_tensor *  tensor = nullptr; // 3D tensor [ne0, ne1, max_slots]
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    enum ggml_type type = GGML_TYPE_F32;
    size_t  stride = 0;
    int32_t max_slots = 0;
    int32_t used_slots = 0;
    int32_t probationary_cap = 0;
    int32_t protected_cap = 0;
    int32_t probationary_used = 0;
    int32_t protected_used = 0;

    std::vector<ggml_backend_expert_slot> slots;
    std::unordered_map<ggml_expert_cache_key, int32_t, ggml_expert_cache_key_hash> key_to_slot;
};
```

### 3.3 Cache Instance (`ggml_backend_expert_cache`)

Expert cache instances are allocated per accelerator backend:
- `buffer`: Backend buffer allocated on the device.
- `tensor`: 1D `GGML_TYPE_I8` tensor spanning total cache capacity.
- `capacity`: Total byte capacity of the cache.
- `used`: Currently allocated byte size.
- `free_blocks`: Free-list tracking available contiguous offsets and sizes.
- `entries`: Hash table mapping `ggml_expert_cache_key` to flat cache entries.
- `access_freq`: Frequency counter tracking expert accesses across decode steps.
- `slot_pools`: Dimension-matched 3D slot pools sub-allocated from `tensor->data`.
- `bundle_registrations`: Mapping of layer IDs to `{gate, up, down}` tensor definitions.
- `pinned_host_buffer`: 16-slot 512-byte aligned page-locked host memory staging buffer.
- `registered_host_ranges`: Vector of host memory pointer ranges registered via `cudaHostRegister` for direct DMA (1 GiB budget cap).
- `layer_transitions`: Transition matrix tracking step transitions $P(E_{t+1} \mid E_t)$.
- `pending_forward_swaps`: JIT queue for amortized periodic rebalancing transfers.

---

## 4. Operational Modes and Eviction Policies

### 4.1 Periodic Rebalancing Mode (Default: `--expert-cache-period 128`)

In periodic mode:
1. During token generation, `ggml_backend_expert_cache_record_access_count` increments expert access frequencies in `access_freq`.
2. Every `period` decode steps (`decode_step % period_tokens == 0`), `ggml_backend_expert_cache_rebalance` executes:
   - Evaluates expert utility using global **Value-per-Byte**: $V = \frac{\text{Access Frequency} \times \text{Payload Size}}{\text{Allocated Slot Bytes}}$.
   - Determines the optimal global set of candidates across all layers that fit within `capacity`.
   - Evicts unneeded entries, returning blocks to `free_blocks` and coalescing adjacent spans.
   - Enqueues promoted hot experts for JIT staged transfer across upcoming layer executions.
   - Decays access frequencies (`freq = (freq * 7) >> 3`, 0.875 multiplier) to adapt smoothly to shifting conversation contexts.

### 4.2 On-Demand Segmented LRU (SLRU) Mode (`--expert-cache-period 0`)

When `period` is set to 0:
- **Probationary Segment (20% of slots)**: Newly admitted misses enter the probationary pool on first access.
- **Protected Segment (80% of slots)**: Upon receiving a second access while resident in probationary, the expert is promoted to protected status.
- **Eviction Hierarchy**: Evictions always target unpinned probationary slots first by least-recent use, protecting frequently accessed core experts from transient cache pollution.

---

## 5. C / C++ API Reference

### 5.1 Scheduler & Context APIs (`ggml-backend.h`)

```c
// Enable and configure expert cache capacity on backend scheduler
GGML_API void   ggml_backend_sched_set_expert_cache(ggml_backend_sched_t sched, size_t size);

// Set periodic rebalance interval in tokens (0 = on-demand SLRU)
GGML_API void   ggml_backend_sched_set_expert_cache_period(ggml_backend_sched_t sched, int32_t period);

// Limit the number of experts swapped per rebalance cycle (-1 = unlimited)
GGML_API void   ggml_backend_sched_set_expert_cache_max_swaps(ggml_backend_sched_t sched, int32_t max_swaps);

// Trigger immediate full rebalance (promote/demote experts based on access frequency)
GGML_API void   ggml_backend_sched_expert_cache_rebalance(ggml_backend_sched_t sched);

// Trigger partial rebalance with limited expert swaps
GGML_API void   ggml_backend_sched_expert_cache_rebalance_partial(ggml_backend_sched_t sched, int max_swaps);

// Register atomic expert bundle ({gate, up, down}) for a given layer
GGML_API void   ggml_backend_sched_register_expert_bundle(
    ggml_backend_sched_t sched,
    int32_t layer,
    const struct ggml_tensor * gate_tensor,
    const struct ggml_tensor * up_tensor,
    const struct ggml_tensor * down_tensor);

// Register CPU host tensor memory for direct DMA (cudaHostRegister)
GGML_API void   ggml_backend_sched_register_host_memory(
    ggml_backend_sched_t sched,
    const struct ggml_tensor * tensor);

// Retrieve runtime expert cache performance statistics
GGML_API bool   ggml_backend_sched_get_expert_cache_stats(
    ggml_backend_sched_t sched,
    int backend_idx,
    struct ggml_backend_expert_cache_stats * out_stats);

// Seed hot expert profile into cache on startup
GGML_API bool   ggml_backend_sched_expert_cache_seed(
    ggml_backend_sched_t sched,
    int backend_idx,
    const struct ggml_tensor * tensor,
    int32_t expert_id,
    uint32_t frequency);
```

### 5.2 Internal Subsystem APIs (`ggml-backend-expert-cache.h`)

```c
// Slot Pools & Zero-Copy Execution
GGML_API struct ggml_tensor * ggml_backend_expert_cache_get_slot_tensor(ggml_backend_expert_cache_t cache, const struct ggml_tensor * weight_tensor);
GGML_API int32_t              ggml_backend_expert_cache_find_slot(ggml_backend_expert_cache_t cache, const struct ggml_tensor * tensor, int32_t expert_id);
GGML_API int32_t              ggml_backend_expert_cache_alloc_slot_idx(ggml_backend_expert_cache_t cache, const struct ggml_tensor * tensor, int32_t expert_id, const struct ggml_expert_cache_key * pinned_keys, size_t n_pinned);
GGML_API void                 ggml_backend_expert_cache_record_zero_copy_hit(ggml_backend_expert_cache_t cache, const struct ggml_tensor * tensor, int32_t expert_id, size_t size);

// Execution telemetry
GGML_API void                 ggml_backend_expert_cache_record_gpu_id_resolution(ggml_backend_expert_cache_t cache);

// Direct Host Memory Registration (V2.2)
GGML_API bool                 ggml_backend_expert_cache_register_host_memory(ggml_backend_expert_cache_t cache, void * ptr, size_t size);
GGML_API bool                 ggml_backend_expert_cache_is_host_memory_registered(ggml_backend_expert_cache_t cache, const void * ptr, size_t size);

// Pinned Memory Staging
GGML_API void *               ggml_backend_expert_cache_get_pinned_slot_buffer(ggml_backend_expert_cache_t cache, int32_t slot_idx, size_t required_size);

// Transition Modeling & Prefetching
GGML_API void                 ggml_backend_expert_cache_record_step_experts(ggml_backend_expert_cache_t cache, int32_t layer, const int32_t * expert_ids, int32_t n_experts);
GGML_API int32_t              ggml_backend_expert_cache_predict_next(ggml_backend_expert_cache_t cache, int32_t layer, const int32_t * current_experts, int32_t n_current, int32_t * out_predicted, int32_t max_predict);
GGML_API void                 ggml_backend_expert_cache_prefetch_layer(ggml_backend_expert_cache_t cache, int32_t layer, const int32_t * expert_ids, int32_t n_experts);

// JIT Staged Swaps
GGML_API void                 ggml_backend_expert_cache_process_jit_swaps(ggml_backend_expert_cache_t cache, const struct ggml_tensor * completed_tensor, ggml_backend_t backend);
```

---

## 6. Validated Benchmark Results

Evaluated on **Qwen3.6-35B-A3B-APEX-Compact.gguf** (35B total params, 3B active params, 64 experts per layer, 8 active per token) on NVIDIA GeForce GTX 1080 (8 GB VRAM) + CPU Host (14 threads) using **10-repetition multi-run benchmarking (`-r 10`)**:

| Benchmark Mode | Baseline (`-r 10`) | V2 Optimized (`-r 10`) | Throughput Delta | Variance / Stability Delta |
|---|---|---|---|---|
| **Prompt Processing (`pp512`)** | 465.02 ± 10.32 tok/s | **467.67 ± 10.30 tok/s** | +0.6% | Zero prefill regression |
| **Cold Decode (`tg64`)** | 25.61 ± 0.67 tok/s | **26.38 ± 0.36 tok/s** | **+3.0% speedup** | **Standard deviation cut in half (-46% jitter)** |
| **Warm Decode (`tg256`)** | 25.37 ± 0.63 tok/s | **26.43 ± 0.32 tok/s** | **+4.2% speedup** | **Standard deviation cut in half (-49% jitter)** |
| **Steady-State Decode (`tg512`)** | 25.37 ± 0.95 tok/s | **25.49 ± 0.90 tok/s** | +0.5% | Consistent sustained throughput across long contexts |

---

## 7. Profile Persistence and Pre-Seeding

To avoid cold-start penalties when launching a model, the expert cache supports saving and loading hot-expert profiles in JSON format:

### JSON Profile Format (`<model>.expert_cache.json`)

```json
{
  "version": 1,
  "profile": "default",
  "n_entries": 32,
  "updated_at": "2026-08-19T13:00:00Z",
  "experts": [
    {
      "tensor": "blk.0.ffn_gate_exps.weight",
      "expert_id": 4,
      "frequency": 128,
      "hit_count": 95
    }
  ]
}
```

### CLI and Server Options

- `--expert-cache <size>` / `-exc <size>`: Size of accelerator memory for CPU-offloaded MoE experts. The size accepts byte or unit suffixes, such as `256M` and `1G`.
- `--expert-cache-period <tokens>` / `-excp <tokens>`: Rebalance period in tokens (default: `64`). Set to `0` for on-demand SLRU.
- `--expert-cache-stats`: Print runtime cache performance statistics (hit rate, avoided bandwidth).
- `--expert-cache-profile <name>`: Profile name for saved/loaded cache files.
- `--expert-cache-persist`: Automatically save accumulated hot-expert profile on server idle or exit.
- `--expert-cache-max-swaps <N>` / `-excm <N>`: Maximum number of experts to swap per rebalance cycle (default: `-1` = unlimited). Limits PCIe transfer burst size during periodic rebalancing. For example, `-excm 2` with `-excp 256` swaps at most 2 experts every 256 tokens instead of rebalancing the entire cache.
- `--expert-cache-rebalance-per-request`: Server-only flag that triggers a full cache rebalance after each request completes. Promotes/demotes experts based on the most recent request's access pattern. Useful for multi-turn conversations or repeated similar requests where cross-request expert locality exists.

---

## 8. Dynamic MTP Offload and Phase-Aware Residency

Models with Multi-Token Prediction (MTP / NextN), such as Qwen 3.5 and Qwen 3.6 MoE architectures, bundle extra decoder blocks after the main trunk layers (e.g. layer index `n_layer` to `n_layer_all - 1`).

### 8.1 Dynamic Weight Residency Lifecycle

When `--mtp-dynamic-offload` is enabled:
1. **Model Loading Phase**:
   - Active GPU layer budget is set to `n_trunk` (`hparams.n_layer()`).
   - Base trunk layers (`0 .. n_trunk - 1`) and output layer receive full GPU VRAM placement (`i_gpu_start = 0`).
   - MTP layers (`n_trunk .. n_layer_all - 1`) are staged in host RAM (`cpu_dev`).
2. **Prompt Processing (PP) Phase**:
   - 100% of trunk layers run in GPU VRAM with full compute throughput and zero host synchronization.
3. **Generation / Speculative Drafting Phase**:
   - When speculative MTP drafting begins, `llama_model_mtp_promote_to_gpu()` is invoked.
   - Dedicated GPU weight buffer is allocated if not already present.
   - MTP weights are transferred asynchronously via high-speed DMA (`ggml_backend_tensor_set_async`).
   - Speculative draft decodes execute on the GPU backend without reinitializing model or context state.

### 8.2 Layer Budgeting Invariants (`n_layer_budget`)

$$\text{n\_layer\_budget} = \begin{cases} \text{n\_layer\_all}, & \text{if } \text{load\_mtp} \land \neg\text{mtp\_dynamic\_offload} \land (\text{n\_layer\_nextn} > 0) \\ \text{n\_layer}, & \text{otherwise} \end{cases}$$

- **Non-MTP Mode (`load_mtp = false`)**: Active budget is `n_layer`. MTP weights remain on CPU host and never consume VRAM or steal GPU layer slots. Layer 0 is guaranteed on GPU when `-ngl >= n_layer + 1`.
- **Dynamic MTP Mode (`mtp_dynamic_offload = true`)**: Active budget is `n_layer`. Trunk layers occupy VRAM during prefill, and MTP is promoted dynamically for token generation.
- **Static MTP Mode (`load_mtp = true && !mtp_dynamic_offload`)**: Active budget is `n_layer_all`. Both trunk and MTP layers are statically offloaded to GPU.

### 8.3 Correctness Hardening (2026-08-20)

Dynamic MTP offload correctness fixes, verified on `Qwen3.6-35B-A3B-APEX-MTP-Quality.gguf` (21.87 GB, one MTP block `blk.40`, 20 MTP tensors, 856.36 MiB):

1. **Quantized padding initialization**: Promotion now calls `ggml_backend_buffer_init_tensor()` on each tensor after re-pointing `data`/`buffer` at the GPU allocation and before the async weight copy. CUDA's `init_tensor` zeroes the quantized padding tail (`padded_size - original_size` bytes), so uninitialized memory is never visible to `MUL_MAT_ID`. Promotion no longer emits garbage when a full padded slot is read.
2. **Host expert-cache exclusion**: Dynamically promoted MTP experts are excluded from the host expert-cache registration loop (`llama_context::sched_reserve`). Because promotion moves the MTP weights out of host memory, registering them as host-resident would leave the cache holding stale pointers. Static MTP models are unaffected (`has_mtp()` is true only when the dynamic collection is enabled).
3. **Owned-host-tensor guard**: Dynamic relocation is enabled only when every collected MTP tensor is an owned host tensor (`t->view_src == NULL && ggml_backend_buffer_is_host(t->buffer)`). If any tensor fails this check, the whole mode is disabled with a `LOG_WARN` and the model falls back to host-resident MTP. No partial relocation is attempted.
4. **Deferred fit sizing**: The MTP GPU promotion buffer is charged to the first GPU device in `llama_get_memory_breakdown()` (`deferred MTP promotion buffer = <MiB>`), so `--fit` reserves headroom for the lazy promotion instead of silently overcommitting VRAM. The static layer count is not inflated; the allocation is deferred in time but required before MTP generation.
5. **Promotion failure pinning**: A failed GPU-buffer allocation, backend init, or tensor init sets a sticky `promotion_failed` flag. All tensors touched during the failed init path are restored to their captured `host_data`/`host_buffer`, and future promotion calls return `false` without retrying the allocation or repeating the error log. The host-resident MTP fallback is always preserved.

### 8.4 Dynamic MTP Offload Validation Results (2026-08-20)

Deterministic single-request matrix on `Qwen3.6-35B-A3B-APEX-MTP-Quality.gguf` (fixed prompt, `temperature = 0`, `top-k = 1`, `seed = 42`, fresh server per row):

| Row | Spec | `exc` | `mtp-dynamic-offload` | draft_n / accepted | Result |
|---|---|---|---|---|---|
| A | none | 0 | off | 0 / 0 | coherent, reproducible |
| B | none | 64M | off | 0 / 0 | token-identical to A |
| C | draft-mtp | 0 | off | 188 / 160 | coherent |
| D | draft-mtp | 0 | on | 182 / 164 | coherent, promotion logs confirmed |
| E | draft-mtp | 64M | on | 186 / 161 | coherent |
| F | draft-mtp, parallel=2 | 0 | on | 194 / 157 | both slots coherent |

Confirmed with `-lv 4`:

```text
load_tensors: MTP dynamic offload enabled: 20 MTP tensors (856.36 MiB) staged in host memory
mtp_promote_to_gpu: MTP weights promoted to GPU in 101.42 ms (856.36 MiB)
common_get_device_memory_data_impl: deferred MTP promotion buffer = 856.36 MiB on CUDA0
slot print_timing: draft acceptance = 0.95238 (20 accepted / 21 generated), mean len = 2.82
```

Note: model-load `LLAMA_LOG_INFO` lines are filtered at the default server verbosity (level 3). Use `-lv 4` to see the MTP dynamic-offload and promotion log lines. All rows emitted coherent output; no gibberish reproduced, and expert-cache on/off does not change target-only token IDs.

## 9. MTP and Expert-Cache Performance Plan Execution (2026-08-20)

### Implemented Changes

- Scheduler-owned scratch vectors now reuse expert IDs, bitsets, counts, requested experts, pinned keys, remapped IDs, and miss bitsets across one `ggml_backend_sched_compute_splits()` call.
- The unused device slot-map path was removed. Graph execution uploads explicit remapped IDs to `ids_tensor`; no graph consumes a device expert-to-slot map.
- Profile loading resolves and validates tensors before submission, rejects invalid expert IDs, deduplicates `(tensor, expert_id)` entries by highest frequency, submits entries grouped by backend, and synchronizes once.

### Single-Request MTP Measurements

Model: `Qwen3.6-35B-A3B-APEX-MTP-Quality.gguf`; GTX 1080; `--fit on --fit-target 256`; `parallel = 1`; `exc = 64M`; `cram = 1024`; fixed prompt; `temperature = 0`; `top-k = 1`; `seed = 42`; 256 generated tokens.

| Expert-cache period | Generation throughput | Token sequence |
|---|---:|---|
| 0 | 22.94 tok/s | identical SHA-256 `580b417f73e4d58b209b44e5f07ccc269900d4b0d9d5318e8866f1d6f1335fe8` |
| 64 | 22.86 tok/s | identical |
| 256 | 22.53 tok/s | identical |

These are single samples, not a statistically significant capacity or period selection. The server did not emit nonzero expert-cache request counters for this fitted placement.

`parallel = 2` is not comparable with `parallel = 1` while `--fit` is enabled: fit selected different partial-layer placements and produced a different deterministic token sequence. Use `parallel = 1` for a single active request.

### Trace-Gated Work

Selected-expert event overlap and same-device target-to-MTP hidden-state handoff remain unchanged. Nsight Systems was unavailable, so neither required 5 percent trace gate could be measured safely.



---

## 10. Phase 5: Routing Lookahead Pipeline (2026-08-21)

### 10.1 Motivation and Problem Statement

The forced-routing experiment (2026-08-21) revealed that **169.7 GiB crossed RAM→GPU over 5×128 generated tokens**, while only 20.7 GiB was avoided. This dropped generation from ~26.5 tok/s to 9.19 tok/s. The cache hit rate is not the problem — **exposed DMA latency** is.

The Routing Lookahead Pipeline predicts expert demand H layers ahead and issues asynchronous DMA transfers on a dedicated CUDA stream, allowing transfers to overlap with compute. The native Qwen router still decides what executes, making this experiment completely correctness-preserving.

### 10.2 Implementation Phases

#### Phase 5A: Offline Route Predictability Study ✅ Complete

**Objective:** Measure how predictable expert routing is, without any runtime modifications.

**Deliverables:**
- **Route trace collector** instrumented in `ggml-backend.cpp` `mul_mat_id` path (lines 1802-1809)
- Records per-token, per-layer: token_id, layer, top-k experts, timestamp
- Binary trace format: `<Qii64iQ` (token_id, layer, n_experts, expert_ids[64], timestamp_us), magic 0x52545243
- **Python trace analyzer** (`tools/results/route-trace-analyzer.py`): computes Recall@K for multiple prediction strategies
- **Test coverage**: `tests/test-route-trace.cpp`

**Integration point:**
```cpp
// Phase 5A: Record route trace for predictability analysis
if (!requested_experts.empty()) {
    int32_t layer = ggml_backend_expert_cache_get_tensor_layer(input);
    if (layer >= 0) {
        ggml_backend_expert_cache_record_route_trace(
            cache, layer, requested_experts.data(), (int32_t)requested_experts.size());
    }
}
```

#### Phase 5B: Trace-Driven Oracle Simulator ✅ Complete

**Objective:** Simulate the entire prefetch pipeline with perfect future knowledge to determine theoretical maximum benefit.

**Deliverables:**
- **Oracle simulator** (`tools/results/oracle-simulator.py`): simulates transfer pipeline with perfect knowledge
- **PCIe bandwidth benchmark tool** (`tools/results/pcie-bandwidth-bench.cpp`)
- **Ready-recall analysis**: distinguishes prediction recall from ready recall (experts that arrive before execution reaches the layer)

**Oracle Simulation Results (PCIe 3.0 x16 @ 12 GB/s, 200 µs/layer):**

| Horizon | Fully Hidden | Partially Hidden | Misses | Hit Rate | Time/Token (µs) | Speedup |
|---------|-------------|-----------------|--------|----------|-----------------|---------|
| H=0 | 0 | 0 | 160000 | 0.000 | 471859.20 | 1.00x |
| H=1 | 0 | 156000 | 4000 | 0.975 | 409459.20 | 1.15x |
| H=2 | 0 | 156000 | 4000 | 0.975 | 348659.20 | 1.35x |
| H=4 | 0 | 156000 | 4000 | 0.975 | 231859.20 | 2.04x |
| H=8 | 128000 | 28000 | 4000 | 0.975 | 49571.84 | **9.52x** |
| H=12+ | 128000 | 28000 | 4000 | 0.975 | 49571.84 | 9.52x |

**Key insight:** H=8 is the sweet spot. At 12 GB/s, one expert (16.88 MiB) takes ~1467 µs to transfer. With 200 µs/layer compute, you need ~8 layers of lookahead for the transfer to complete before execution reaches it. Beyond H=8, there's no additional benefit — the transfer is already fully hidden.

**Theoretical ceiling: 9.52x speedup** with perfect prediction at H=8. The 4000 misses (2.5%) are the first layer of each token — unavoidable since there's no prior layer to prefetch from.

**Bottleneck analysis:** Perfect prediction gives 9.5x speedup, confirming latency is the bottleneck and prefetching is worthwhile.

#### Phase 5C: Async DMA Pipeline + Heuristic Predictor ✅ Complete

**Objective:** Implement dedicated CUDA transfer stream and heuristic predictor (previous token + cross-layer transition tables).

**Deliverables:**

1. **Dedicated CUDA prefetch stream**
   - `cudaStream_t prefetch_stream` added to `ggml_backend_expert_cache` struct
   - Prefetch state tracking: `EMPTY`, `IN_FLIGHT`, `RESIDENT`
   - `cudaEvent_t ready_event` per prefetch slot for completion signaling

2. **Prefetch-aware execution** in `ggml-backend.cpp` `mul_mat_id` path (lines 1838-1847):
   ```cpp
   if (slot.state == GGML_EXPERT_CACHE_PREFETCH_RESIDENT) {
       execute;  // fully hidden hit
   } else if (slot.state == GGML_EXPERT_CACHE_PREFETCH_IN_FLIGHT) {
       cudaEventSynchronize(slot.ready_event);  // partially hidden
       execute;
   } else {
       synchronous_miss();  // fallback
   }
   ```

3. **Heuristic predictor** (transition tables)
   - API: `ggml_backend_expert_cache_enable_predictor()`, `disable_predictor()`, `record_prediction()`, `predict_experts()`
   - Maintains transition table: `P(expert_e at L+H | expert_f at L)`
   - Integrated into execution path (lines 1811-1830):
   ```cpp
   // Phase 5C: Record current experts and predict next layer
   if (!requested_experts.empty()) {
       int32_t layer = ggml_backend_expert_cache_get_tensor_layer(input);
       if (layer >= 0) {
           ggml_backend_expert_cache_record_prediction(
               cache, layer, requested_experts.data(), (int32_t)requested_experts.size());
           int32_t next_layer = layer + 1;
           int32_t predicted_experts[16];
           int32_t n_predicted = ggml_backend_expert_cache_predict_experts(
               cache, layer, next_layer, predicted_experts, 16);
           if (n_predicted > 0) {
               ggml_backend_expert_cache_prefetch_async(
                   cache, input, predicted_experts, n_predicted, next_layer);
           }
       }
   }
   ```

4. **Prefetch effectiveness metrics**
   - Track: `fully_hidden_hits`, `partially_hidden_hits`, `misses`, `wasted_prefetches`
   - Compare against baseline (no prefetch) and oracle (from 5B)

**API additions** (`ggml-backend-expert-cache.h`):
```cpp
// Phase 5C: Async DMA Pipeline
GGML_API void ggml_backend_expert_cache_prefetch_async(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    const int32_t * expert_ids,
    int32_t n_experts,
    int32_t target_layer);

GGML_API bool ggml_backend_expert_cache_is_prefetch_ready(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id);

GGML_API void ggml_backend_expert_cache_wait_prefetch(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id);

// Phase 5C: Heuristic Predictor (Transition Tables)
GGML_API void ggml_backend_expert_cache_enable_predictor(
    ggml_backend_expert_cache_t cache,
    int32_t max_layers,
    int32_t max_experts_per_layer);

GGML_API void ggml_backend_expert_cache_disable_predictor(
    ggml_backend_expert_cache_t cache);

GGML_API void ggml_backend_expert_cache_record_prediction(
    ggml_backend_expert_cache_t cache,
    int32_t layer,
    const int32_t * expert_ids,
    int32_t n_experts);

GGML_API int32_t ggml_backend_expert_cache_predict_experts(
    ggml_backend_expert_cache_t cache,
    int32_t from_layer,
    int32_t to_layer,
    int32_t * out_expert_ids,
    int32_t max_experts);
```

**Test coverage:** All existing tests pass (16/16 across `test-expert-cache` and `test-expert-cache-profile`).

#### Phase 5D: Learned Routing Predictor ✅ Complete (2026-08-22)

**Objective:** Train tiny external model to predict future expert routes H layers ahead, enabling async prefetch to hide PCIe DMA latency.

**Architectural Change (from reviewer guidance):**
- **Old approach:** Predictor lived inside expert cache code, blocked because cache only sees expert weight tensors in `MUL_MAT_ID` path, not hidden state
- **New approach:** Predictor runs upstream, close to router computation. Consumes router input `x_L` during graph construction, produces predicted expert IDs for layer L+H, feeds async prefetch queue

**Three Predictor Variants:**

| Variant | Description | Training | Input |
|---------|-------------|----------|-------|
| **A: Stale Future Router** | `W_router[L+H] × x_L` directly | No | Router logits at L |
| **B: Low-Rank MLP** | `x_L → rank-32 → expert logits L+H` | Yes (~147k params) | Router logits or projected hidden state |
| **C: Future Router + Residual** | `W_router[L+H]×x_L + Δ_θ(x_L)` | Yes (residual only) | Router logits + learned correction |

**Reviewer recommendation:** Start with Variant A (zero training cost), then try Variant C (strong structural prior + tiny trainable params).

**Implementation Status (2026-08-22): All 11 tasks complete.**

1. **Callback bugs fixed** (`src/llama-graph.cpp`)
   - `logits_fallback` hoisted outside if-block; `logits_ptr` alive through `ggml_backend_tensor_get`
   - Uses `res->predictor_horizon` and `res->predictor_depth` (depth = `min(n_expert, hparams.n_expert_used*2)` capped 32)
   - Returns `true` on every path (old code returned `false` for `ask==false`, cancelling split compute)
   - `predictions_generated` increment moved to cache via `ggml_backend_sched_add_predictions_generated`
   - Eval callback chaining: `prev_cb` + `prev_cb_user_data` stored in `llm_graph_result`; defers to prior callback for non-routing tensors

2. **Prediction queue in cache** (`ggml/src/ggml-backend-expert-cache.h`, `ggml/src/ggml-backend-expert-cache.cpp`)
   - `ggml_expert_cache_pending_prediction` struct: `target_layer`, `n_experts`, `expert_ids[64]`
   - `pending_predictions` map (one entry per target layer, latest replaces)
   - `executed_layer_cursor` reset on `begin_step`
   - Public APIs: `settle_prediction`, `pending_prediction_count`, `get_pending_prediction`

3. **Settle accounting at MUL_MAT_ID interception** (`ggml/src/ggml-backend.cpp`)
   - Hook after `record_prediction` block, only if `pending_prediction_count > 0`
   - Classification: resident = `find_slot >= 0`, in-flight = `has_inflight_prefetch`, else missed
   - "too_late" when layer `<= executed_layer_cursor`; cursor advanced on each settle
   - `bytes_wasted` per unpredicted actual id x pool_stride_bytes

4. **Prefetch at submit time** (`ggml/src/ggml-backend-expert-cache.cpp`)
   - Inside `submit_prediction`: lookup `bundle_registrations[target_layer]`, call `prefetch_async` for gate/up/down
   - No later scheduling; existing `prefetch_async` (CPU) is sync fallback

5. **Real stats aggregation** (`ggml/include/ggml-backend.h`, `ggml/src/ggml-backend-expert-cache.cpp`, `ggml/src/ggml-backend.cpp`)
   - `ggml_routing_predictor_stats routing_predictor` member added to `ggml_backend_expert_cache_stats`
   - Cache-level getter `ggml_backend_expert_cache_get_routing_predictor_stats(cache, out)`
   - Sched aggregator `ggml_backend_sched_get_routing_predictor_stats` sums across `expert_caches[b]`
   - `ggml_backend_expert_cache_add_predictions_generated(cache, n)` + sched wrapper
   - Dead `llm_graph_result::routing_predictor_metrics` struct deleted

6. **Route trace v2 with logits** (`ggml/src/ggml-backend-expert-cache.cpp`, `ggml/src/ggml-backend.cpp`)
   - Entry layout: fixed `ggml_expert_cache_route_trace_entry` + `int32 n_logits` + `float logits[n_logits]`
   - Version constant bumped to 2
   - Separate logits staging via `ggml_backend_expert_cache_record_router_logits(cache, layer, logits, n_logits)` (one-shot per layer, consumed by next `record_route_trace`)
   - Env-var trigger `GGML_EXPERT_ROUTE_TRACE` in `ggml_backend_sched_set_expert_cache`

7. **Python feature collector** (`tools/collect_router_features.py`)
   - Parses v1+v2 traces (struct pack `<Qii` for token_id, layer, n_experts)
   - Writes RPDS v2 training rows: header `magic=RPDS, ver=2, num_experts, horizon`; row = `layer, token_id, float[num_experts] logits, int8[num_experts] future_mask`
   - CLI: positional `input output`, `--horizon 8 --sample-rate 1 --num-experts 64`

8. **LRPD v2 trainer** (`tools/train_routing_predictor.py`)
   - numpy-only (no torch), trains low-rank MLP with SGD
   - Writes LRPD v2 matching init-time `load_model` in `ggml-routing-predictor.cpp`
   - Magic `0x4C525044`, version=2, dims `{input_dim, rank, num_experts}`
   - Weights: `down_weight[rank*input_dim]`, `down_bias[rank]`, `output_weight[num_experts*rank]`, `output_bias[num_experts]`

9. **CLI plumbing** (`common/arg.cpp`, `common/common.cpp`, `src/llama-cparams.h`, `src/llama-context.cpp`, `tools/llama-bench/llama-bench.cpp`)
   - `--routing-predictor-model <path>`: path to trained LRPD model (enables low-rank-mlp variant)
   - `--routing-predictor-variant {stale-future|low-rank-mlp|future-residual}`: explicit variant selection
   - `cparams.routing_predictor_model` (std::string) and `routing_predictor_variant` (int32_t) propagated
   - llama-bench CSV columns include `routing_predictor_model`, `routing_predictor_variant`
   - Model path implies LOW_RANK_MLP even if variant defaulted to 0; explicit stale-future overrides
   - Init failure logs WARN and falls back to STALE_FUTURE

10. **Determinism matrix row F** (`scripts/expert-cache-determinism-matrix.py`)
    - Row F: expert cache on + routing predictor, `--routing-predictor-horizon 8 --routing-predictor-stats`
    - All four rows (A, B, E, F) must produce same SHA-256 for determinism

11. **Test coverage**
    - `test-routing-predictor.exe`: all 6 tests pass (including LRPD v2 round-trip)
    - `test-expert-cache.exe`: all tests pass (route trace v2, prediction queue, settle accounting, submit triggers prefetch, stats getter, route trace v2 logits)

**Stats struct** (`ggml/include/ggml-backend.h`):
```cpp
struct ggml_routing_predictor_stats {
    int64_t predictions_generated;
    int64_t predictions_used;
    int64_t predictions_too_late;
    int64_t predictions_wrong;
    int64_t experts_fully_hidden;
    int64_t experts_partially_hidden;
    int64_t experts_missed;
    int64_t bytes_wasted;
};
```

**LRPD binary format (v2, trainer output):**
```
Magic: 0x4C525044 ("LRPD")
Version: 2 (uint32)
Input dim (int32), Rank (int32), Num experts (int32)
down_weight[rank * input_dim] floats
down_bias[rank] floats
output_weight[num_experts * rank] floats
output_bias[num_experts] floats
```
Note: both `ggml_routing_predictor_load_model` (public) and the cache-level `ggml_backend_expert_cache_load_learned_predictor` (used by Variants B/C) read version 1 or 2 with dim order `{input_dim, rank, num_experts}`. Header order matches the trainer in `tools/train_routing_predictor.py` exactly. An older draft swapped `rank` and `num_experts`; that was fixed on 2026-08-22.

**Status (2026-08-22):**

1. **Pipeline end-to-end works.** Trace collection -> feature extraction -> training -> model load -> Variant B init all verified. `model.bin` (66 KB, LRPD v2, rank=32, num_experts=256, recall@8 ~= 0.72 on a 23-sample run) loads successfully in the ggml predictor and `routing_predictor_variant=1` shows on the bench CSV. See [Section 10.5](#105-phase-5e-end-to-end-validation-2026-08-22) for evidence and remaining gaps.

2. **Stats retrieval still stubbed.** `ggml_backend_sched_get_routing_predictor_stats()` (`ggml/src/ggml-backend.cpp`, near the bottom) returns false without populating fields. Cache-level writes into `cache->routing_predictor_stats` happen at `record_prediction`, `settle_prediction`, and the eval callback, but the sched-level aggregator either is missing or reads the wrong field. **Fix:** verify the sched getter path, populate `out_stats` from `sched->expert_caches[backend_idx]->routing_predictor_stats`. (Diagnostic pending - see [Section 10.5](#105-phase-5e-end-to-end-validation-2026-08-22).)

3. **`predictions_generated` is still 0 at runtime even with a valid model loaded.** The eval callback chain is wired (`src/llama-context.cpp:1415-1419` -> `src/llama-graph.cpp:3990-4095`) and the predictor handles are non-null, but no per-token `add_predictions_generated` increments are observed. Plan and diagnostic checklist in `docs/plans/2026-08-22-runtime-utilization.md` (D1-D3): the most likely root cause is graph-reuse / split-cache keeping a callback pointer set during a previous graph build before the routing predictor was attached.

4. **Bundle registration may be empty for some layers.** `submit_prediction` looks up `bundle_registrations[target_layer]` and silently no-ops on miss. If the layer's `{gate, up, down}` bundle was never registered, predictions are recorded but never reach `prefetch_async`. Add a log when `bundle_registrations` is consulted and count misses.

5. **Predictor is constructed per-cache-instance, not per-context.** With ~40 layers and ~2-3 cache instances per layer this is ~80-120 predictor initializations per bench run. Functionally correct (each variant B init succeeds); cosmetic only. **Fix (optional):** move to `llama_context` member if it ever shows up in a flame graph.

#### Phase 5E: Pipeline End-to-End Validation (2026-08-22)

**Objective:** Confirm the trace -> features -> model -> load -> run loop actually works on real hardware.

**Environment:** `feat/expert-cache-only` branch, MSVC Release, GTX 1080 (8 GB VRAM) + 14-thread CPU, `Qwen3.6-35B-A3B-APEX-Compact.gguf` (256 experts, 8 used, 40 layers).

**Working launch config:**
```sh
build/bin/Release/llama-server.exe -m <model> \
    --ctx-size 4096 --batch-size 4096 --ubatch-size 2048 \
    --threads 14 --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 --no-context-shift \
    -fitt 256 -exc 64M -excp 64 \
    --routing-predictor-model tools/training_data/model.bin \
    --routing-predictor-horizon 8 --routing-predictor-stats \
    --port 8137 --temp 0 --no-mmap
```

Use `-fitt 256` instead of `-ngl N`: `ngl` places the entire `-ngl` layers on GPU and leaves the rest on CPU without any copy, so cache interception never triggers. `--fit on --fit-target 256` packs dense layers on GPU and expert weights on RAM, and the host->GPU copy on first MUL_MAT_ID triggers the cache path.

**Pipeline outputs (verified 2026-08-22, single fresh ctx per row):**

| Step | Artifact | Size | Notes |
|---|---|---|---|
| Trace | `tools/training_data/route_trace.bin` | 25 852 B | 91 RTRC v2 entries (magic `0x52545243` "RTRC", version 2) |
| Features | `tools/training_data/training_data.bin` | ~3 KB | 23 horizon-paired samples (RPDS magic) |
| Model | `tools/training_data/model.bin` | 66 708 B | LRPD v2, `input_dim=256 rank=32 num_experts=256`, recall@8 ~= 0.72 on the 23-sample set |
| Server log | `tools/training_data/server.log` | ~250 KB | Contains 93x "Variant B initialized" lines, zero "Failed to load", zero "Invalid model" |

**Bench evidence (variant B loaded):**
```sh
build/bin/Release/llama-bench.exe -m <model> -p 64 -n 8 -fitt 256 \
    -exc 64 -excp 64 \
    --routing-predictor-model tools/training_data/model.bin \
    --routing-predictor-horizon 8 --routing-predictor-stats \
    --routing-predictor-variant low-rank-mlp -o csv
```
Relevant CSV tail: `routing_predictor_model=tools/training_data/model.bin, routing_predictor_variant=1 (low-rank-mlp)`. The model attaches and logs `Variant B initialized` for every cache instance.

**Bug fixes landed in this cycle (7 root causes):**

1. **Wrong launch config (`-ngl 20`).** Whole layers go to GPU with no copy; the per-split copy that creates a `CUDA0#blk.N.ffn_*_exps.weight` 4-D alias never fires. With `-fitt 256`, the host->GPU copy does fire and `ggml_backend_sched_compute_splits` can intercept MUL_MAT_ID.
2. **`ggml_backend_find_mul_mat_id_node` used pointer identity only.** When `node->src[0]` is the original weight tensor (not the per-backend alias), the lookup missed and the cache registered nothing. Added a name-stripping fallback that strips `<backend>#` prefix and `#<digits>` suffix and matches by base name.
3. **`enable_route_trace` re-opened the trace file per cache instance.** 40+ backends each called `fopen("wb")`, truncating the file repeatedly. Replaced with a process-wide `g_route_trace_file` static in `ggml-backend-expert-cache.cpp`; first call opens + writes the header, subsequent calls share the `FILE*`.
4. **`disable_route_trace` `fclose`-d the shared `FILE*`.** First cache freed -> dangling pointer in the rest. Added the same `g_route_trace_file` guard: only `fclose` if `cache->route_trace_file != g_route_trace_file`.
5. **Route trace never flushed on Windows SIGTERM.** `TerminateProcess` skips C++ destructors; the 10 000-entry flush threshold from `disable_route_trace` never fires. Now `record_route_trace` calls `flush_route_trace` immediately on every entry.
6. **LRPD loader header field order was `(input_dim, num_experts, rank)`.** Trainer writes `(input_dim, rank, num_experts)`. Both the public `ggml_routing_predictor_load_model` and the cache-level `ggml_backend_expert_cache_load_learned_predictor` now read in the trainer's order; the silent dimension swap that previously made the model refuse to load is gone.
7. **LRPD loader rejected version 2.** Both loaders now accept v1 or v2 (the cache-level load is the one Variants B/C actually use; the public one was already non-strict after the original session but the v2 acceptance was tightened as a safety rail).

**Reusable harness:** `tools/training_data/run_trace_server.py` (Python wrapper) launches the server with the correct env-var, arg, and logging config. No ad-hoc subprocess flags; all flags are CLI-driven (`--exc`, `--fitt`, `--ngl`, `--port`, `--max-tokens`, `--prompt-repeat`).

**Open: predictions_generated remains 0 at runtime.** Plan: `docs/plans/2026-08-22-runtime-utilization.md`. Top diagnostic candidates are (a) the graph-reuse branch in `src/llama-context.cpp:1387-1397` not re-registering the routing callback before `graph_compute_async`, and (b) the stats getter at the sched level not aggregating the cache's per-instance counter. Fixing either should produce non-zero `predictions_generated` in the bench CSV without retraining the model.
5. **Predictor is constructed per-cache-instance, not per-context.** With ~40 layers and ~2-3 cache instances per layer this is ~80-120 predictor initializations per bench run. Functionally correct (each variant B init succeeds); cosmetic only. **Fix (optional):** move to `llama_context` member if it ever shows up in a flame graph.

**Key Design Decisions (from reviewer):**

1. **Use router logits, not hidden state:** Router logits are only 128 floats and encode compressed semantic routing representation. Much cheaper than copying full hidden state (4096 dims) to CPU.

2. **Target H=8, not H=1:** One 16.88 MiB expert takes ~1467 µs to transfer. At 200 µs/layer compute, need ~8 layers to hide DMA latency. H=1 predictions arrive too late to be useful.

3. **Multi-label, not classification:** Qwen activates 8 experts out of 128. Target is `y ∈ {0,1}^128`, not single class. Loss: `L = L_BCE + λ L_rank`. Metric: Recall@8, Recall@12, Recall@16 (not exact-match).

4. **Train on whole prompts, not tokens:** Adjacent layers/tokens are heavily correlated. Split by whole prompts/conversations to avoid data leakage.

5. **Never wait for prediction:** If prediction arrives too late, drop it. Never block compute stream.

**Remaining Work:**

1. **Immediate fixes (1-2 days):**
   - Implement stats retrieval properly
   - Fix initialization lifetime (once per context)
   - Wire eval callback to run prediction during graph execution

2. **Variant A integration (2-3 days):**
   - Add async prefetch queue (`ggml_backend_sched_submit_prediction()`)
   - Benchmark with real MoE model (Qwen3.6-35B-A3B-APEX-Compact.gguf)
   - Measure: prediction recall, prefetch hit rate, decode throughput delta

3. **Training pipeline (1-2 weeks):**
   - Data collection: instrument graph to collect router logits at layer L, record actual expert selections at layer L+H
   - Train Variant B (Low-Rank MLP): `router_logits → rank-32 → expert_logits`
   - Train Variant C (Future Router + Residual): freeze future router weights, train only residual
   - Split by whole prompts, evaluate by category (coding, conversation, reasoning)

4. **Advanced integration (2-4 weeks):**
   - CUDA predictor stream (low-priority auxiliary stream, avoid synchronizing main compute)
   - Multi-horizon prediction (shared rank-32 trunk + horizon-specific heads for H=4,6,8,10,12)
   - Router logits as features (test: x_L, router_logits_L, top-K IDs + weights, concatenation)

**Benchmark Configuration:**
```bash
# Baseline (no expert cache, no predictor)
llama-bench -m <model> -p 512 -n 128 -fitt 256

# Expert cache only
llama-bench -m <model> -p 512 -n 128 -fitt 256 -exc 64 -excp 64

# Expert cache + routing predictor (Variant A)
llama-bench -m <model> -p 512 -n 128 -fitt 256 -exc 64 -excp 64 --routing-predictor-horizon 8 --routing-predictor-stats
```

**Model:** Qwen3.6-35B-A3B-APEX-Compact.gguf (35B total, 3B active, 64 experts/layer, 8 active/token)  
**Hardware:** GTX 1080 (8 GB VRAM) + CPU (14 threads)

**Documentation:**
- Handover document: `docs/plans/2026-08-22-learned-predictor-handover.md`
- Original plan: `docs/plans/2026-08-21-routing-lookahead-pipeline.md`
- Fix plan: `docs/plans/2026-08-22-fix-routing-predictor-issues.md`
- Runtime utilization plan: `docs/plans/2026-08-22-runtime-utilization.md`
