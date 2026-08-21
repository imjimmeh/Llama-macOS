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


