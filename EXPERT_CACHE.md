# Expert Cache Subsystem in llama.cpp

The **Expert Cache** provides high-performance heterogeneous inference for Mixture-of-Experts (MoE) models (such as Qwen 3.5/3.6 MoE, DeepSeek, Mixtral, Gemma MoE, and OLMoE) when offloaded across CPU host memory and GPU/accelerator VRAM.

---

## 1. Overview and Motivation

In standard CPU/GPU offloaded MoE inference:
1. Routing layers determine a sparse set of active experts per token (e.g. top-2 or top-8 out of 64+ experts).
2. For un-offloaded layers, expert weight matrices reside in host RAM and are transferred over the PCIe bus to accelerator memory on every token generation step.
3. Because expert routing exhibits high temporal and semantic locality, a small subset of experts ("hot experts") accounts for a large percentage of total activations during inference.

The Expert Cache maintains a dedicated, persistent accelerator-side buffer to hold recently or frequently used expert weights. When an active expert hits the cache:
- The weight transfer becomes an ultra-fast **Device-to-Device (D2D)** memory copy rather than a PCIe Host-to-Device (H2D) transfer.
- PCIe bandwidth consumption and latency are substantially reduced.
- Compute stream execution runs uninterrupted without in-band PCIe saturation.

---

## 2. Architecture and Data Structures

The expert cache is architecture-agnostic. It operates entirely at the tensor/operator layer without model-specific assumptions.

```
       Host RAM (CPU)                        Accelerator (e.g. CUDA / Metal / Vulkan)
+---------------------------+                +-----------------------------------------+
| Expert Weights (All)      |                | Expert Cache Buffer (cache->tensor)     |
| [Exp 0][Exp 1]...[Exp N]  |                | [Slot 0][Slot 1][Slot 2]...[Free Space] |
+-------------+-------------+                +--------------------+--------------------+
              |                                                   |
              | Miss: Host-to-Device (PCIe)                       | Hit: Device-to-Device (D2D)
              v                                                   v
       +-----------------------------------------------------------------+
       | Device Working Buffer (input_cpy for GGML_OP_MUL_MAT_ID)       |
       +-----------------------------------------------------------------+
```

### 2.1 Cache Key (`ggml_expert_cache_key`)

Every expert matrix is identified uniquely by its source tensor pointer and expert index:

```cpp
struct ggml_expert_cache_key {
    const struct ggml_tensor * tensor; // pointer to host weight tensor
    int32_t expert_id;                 // index of the expert within tensor
};
```

### 2.2 Expert Bundle Key (`ggml_expert_bundle_key`)

For layer-wide atomic caching, expert components across gate, up, and down projections can be registered as an atomic unit:

```cpp
struct ggml_expert_bundle_key {
    int32_t layer;
    int32_t expert_id;
};
```

### 2.3 Cache Entry (`ggml_backend_expert_cache_entry`)

Each resident expert in the cache tracks:
- `offset`: Byte offset within `cache->tensor`.
- `size`: Exact byte size of the expert matrix payload (`tensor->nb[2]`).
- `alloc_size`: 512-byte aligned slot allocation size.
- `last_used`: Logical clock timestamp of the last access (for LRU/SLRU eviction).
- `hit_count`: Cumulative cache hit count.

### 2.4 Cache Instance (`ggml_backend_expert_cache`)

Expert cache instances are allocated per accelerator backend:
- `buffer`: Backend buffer allocated on the device.
- `tensor`: 1D `GGML_TYPE_I8` tensor spanning the cache capacity.
- `capacity`: Total byte capacity of the cache.
- `used`: Currently allocated byte size.
- `free_blocks`: Free-list tracking available contiguous offsets and sizes.
- `entries`: Hash table mapping `ggml_expert_cache_key` to cache entries.
- `access_freq`: Frequency counter tracking expert accesses across decode steps.
- `slot_pools`: 3D tensor slot pools `[ne00, ne01, max_slots]` matching tensor stride.
- `bundle_registrations`: Mapping of layer IDs to `{gate, up, down}` tensor definitions.
- `pinned_host_buffer`: 512-byte aligned host memory staging buffer for PCIe DMA.
- `layer_transitions`: Transition matrix tracking step transitions $P(E_{t+1} \mid E_t)$.

---

## 3. Operational Modes and Eviction Policies

### 3.1 Periodic Rebalancing Mode (Default: `--expert-cache-period 64`)

In periodic mode:
1. During token generation, `ggml_backend_expert_cache_record_access_count` increments expert access frequencies in `access_freq`.
2. Every `period` decode steps (`decode_step % period_tokens == 0`), `ggml_backend_expert_cache_rebalance` executes:
   - Evaluates expert access frequencies across all layers.
   - Determines the optimal set of top-k experts that fit within `capacity`.
   - Evicts unneeded entries, returning blocks to `free_blocks` and coalescing adjacent spans.
   - Allocates slots for newly promoted hot experts and issues asynchronous H2D transfers into `cache->tensor`.
   - Smoothly decays access frequencies (`freq = (freq * 7) >> 3`, 0.875 multiplier) to adapt dynamically to shifting conversational contexts.

### 3.2 On-Demand LRU / SLRU Mode (`--expert-cache-period 0`)

When `period` is set to 0:
- When a cache miss occurs during single-token decoding, the fetched expert in `input_cpy` is copied D2D into `cache->tensor`.
- If the cache is full, least-recently-used (LRU) unpinned entries are evicted to make room.
- Protected entries with multi-hit history are prioritized over newly admitted probationary entries.

---

## 4. Execution Pipeline inside `ggml-backend.cpp`

During graph evaluation for `GGML_OP_MUL_MAT_ID`:
1. **Decode Step Start**: `ggml_backend_expert_cache_begin_step(cache)` advances the clock and triggers periodic rebalance when due.
2. **Access Tracking and Partitioning**:
   - Inspects `ids_tensor` (node `src[2]`) to extract requested expert IDs.
   - Records access counts in `access_freq`.
   - Partitions requested experts into **hits** (present in `cache->entries`) and **misses**.
3. **Hit Processing (Zero In-Band Overhead)**:
   - Performs ultra-fast D2D asynchronous copies from `cache->tensor` at `cache_offset` directly into `input_cpy` at `expert_offset`.
   - Copies exactly `expert_size` bytes per expert payload.
   - Records hit statistics.
4. **Miss Processing**:
   - Groups contiguous miss ranges and transfers them from host RAM to `input_cpy` over PCIe.
   - If on-demand mode is enabled, registers newly loaded experts into the cache via D2D copy.
5. **Kernel Execution**: Accelerator executes `MUL_MAT_ID` on `input_cpy`.

---

## 5. Profile Persistence and Pre-Seeding

To avoid cold-start penalties when starting the server or reloading a model, the expert cache supports saving and loading hot-expert profiles in JSON format:

### JSON Profile Format (`<model>.expert_cache.json`)

```json
{
  "version": 1,
  "profile": "default",
  "n_entries": 32,
  "updated_at": "2026-08-18T21:00:00Z",
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

- `--expert-cache-size <bytes>`: Size in bytes allocated for the expert cache on accelerator backends (e.g. `2147483648` for 2 GiB).
- `--expert-cache-period <tokens>`: Rebalance period in tokens (default: `64`). Set to `0` for on-demand LRU.
- `--expert-cache-stats`: Print runtime cache performance statistics (hit rate, avoided bandwidth).
- `--expert-cache-profile <name>`: Profile name for saved/loaded cache files.
- `--expert-cache-persist`: Automatically save accumulated hot-expert profile on server idle or exit.

---

## 6. Memory Safety and Invariants

To prevent weight corruption and activation divergence (such as repetitive token degeneration):
- **Exact Payload Transfers**: Cache buffer slots and D2D copies must always use `expert_size` (`tensor->nb[2]`). Host-to-device MMQ padding offsets are strictly confined to the continuous host tensor and destination `input_cpy` buffers.
- **Defragmentation and Free-Block Coalescing**: All evictions during rebalancing return `{offset, alloc_size}` to `free_blocks` and invoke `ggml_backend_expert_cache_coalesce_free` before allocating new entries.
- **512-Byte Boundary Alignment**: All slot allocations are padded to `GGML_EXPERT_CACHE_ALIGN` (512 bytes) to satisfy backend DMA, DirectStorage, and SIMD alignment requirements.
- **Platform-Safe Pinned Deallocation**: Pinned host memory uses `_aligned_free` on Windows and `free` on POSIX systems.

---

## 7. Dynamic MTP Offload and Phase-Aware Residency

Models with Multi-Token Prediction (MTP / NextN), such as Qwen 3.5 and Qwen 3.6 MoE architectures, bundle extra decoder blocks after the main trunk layers (e.g. layer index `n_layer` to `n_layer_all - 1`).

### 7.1 Problem: Compute Residency vs. Weight Residency

In standard execution:
1. **Prompt Processing (PP)**: Only the base trunk layers (`0 .. n_layer - 1`) are executed in the forward graph. The MTP block is completely idle during prompt processing.
2. **Static Offload Bottleneck**: Standard layer offloading assigns layers from the top down (`n_layer_all` down to 0). On memory-constrained accelerators (such as 8 GB GPUs), placing idle MTP blocks in VRAM forces active trunk layers (especially Layer 0) onto CPU host memory.
3. **PCIe Ping-Pong**: Evicting Layer 0 to CPU forces host-device activation synchronization on every token, cutting inference speed in half.

```
Conventional Static Offload (Suboptimal):
GPU VRAM:  [ Layer 1..31 ] [ Output ] [ MTP Layer 32 (IDLE during PP) ]
Host RAM:  [ Layer 0 (ACTIVE during PP -> Forces PCIe Bottleneck)    ]

Dynamic MTP Offload (Optimized):
PP Phase:  GPU VRAM: [ Layer 0..31 (100% Trunk) ] [ Output ]
           Host RAM: [ MTP Layer 32 (Staged in RAM)        ]
TG Phase:  GPU VRAM: [ Layer 0..31 ] [ Output ] [ MTP Layer 32 (Promoted via DMA) ]
```

### 7.2 Dynamic Weight Residency Lifecycle

When `--mtp-dynamic-offload` is enabled:
1. **Model Loading Phase**:
   - Active GPU layer budget is set to `n_trunk` (`hparams.n_layer()`).
   - Base trunk layers (`0 .. n_trunk - 1`) and the output layer receive full GPU VRAM placement (`i_gpu_start = 0`).
   - MTP layers (`n_trunk .. n_layer_all - 1`) are staged in host RAM (`cpu_dev`).
   - MTP tensor metadata, host pointers, and GPU buffer sizes are registered in `llama_model::impl::mtp_state`.
2. **Prompt Processing (PP) Phase**:
   - 100% of trunk layers run in GPU VRAM with full compute throughput and zero host synchronization.
3. **Generation / Speculative Drafting Phase**:
   - When speculative MTP drafting begins, `llama_model_mtp_promote_to_gpu()` is invoked.
   - Dedicated GPU weight buffer is allocated if not already present.
   - MTP weights are transferred asynchronously via high-speed DMA (`ggml_backend_tensor_set_async`).
   - Tensor data pointers (`tensor->data`, `tensor->buffer`) and layer device mappings (`dev_layer[il]`) are dynamically rebound to the GPU backend.
   - Speculative draft decodes execute on the GPU backend without reinitializing model or context state.

### 7.3 Layer Budgeting Invariants (`n_layer_budget`)

To guarantee correct device placement across all execution modes:

$$\text{n\_layer\_budget} = \begin{cases} \text{n\_layer\_all}, & \text{if } \text{load\_mtp} \land \neg\text{mtp\_dynamic\_offload} \land (\text{n\_layer\_nextn} > 0) \\ \text{n\_layer}, & \text{otherwise} \end{cases}$$

- **Non-MTP Mode (`load_mtp = false`)**: Active budget is `n_layer`. MTP weights remain on CPU host and never consume VRAM or steal GPU layer slots. Layer 0 is guaranteed on GPU when `-ngl >= n_layer + 1`.
- **Dynamic MTP Mode (`mtp_dynamic_offload = true`)**: Active budget is `n_layer`. Trunk layers occupy VRAM during prefill, and MTP is promoted dynamically for token generation.
- **Static MTP Mode (`load_mtp = true && !mtp_dynamic_offload`)**: Active budget is `n_layer_all`. Both trunk and MTP layers are statically offloaded to GPU.

### 7.4 CLI Options and Environment Variables

- `--mtp-dynamic-offload`, `--no-mtp-dynamic-offload`: Enable or disable dynamic MTP staging and promotion (default: disabled).
  - Environment variable: `LLAMA_ARG_MTP_DYNAMIC_OFFLOAD`

### 7.5 C API Reference

```c
// Check if model contains MTP layers and has dynamic offload enabled
LLAMA_API bool llama_model_has_mtp(const struct llama_model * model);

// Check if MTP layers currently reside in GPU VRAM
LLAMA_API bool llama_model_mtp_is_gpu_resident(const struct llama_model * model);

// Asynchronously promote MTP weights from host memory to GPU VRAM
LLAMA_API bool llama_model_mtp_promote_to_gpu(const struct llama_model * model, struct llama_context * ctx);

// Demote MTP weights back to host RAM and restore CPU device bindings
LLAMA_API bool llama_model_mtp_demote_to_host(const struct llama_model * model);
```

### 7.6 Synergy with Expert Cache

Dynamic MTP offloading works seamlessly alongside the Expert Cache:
- Base trunk MoE layers continue to utilize the Expert Cache buffer for frequent routed experts.
- MTP decoder blocks (which include routed and shared expert matrices) execute on GPU during drafting without displacing or corrupting trunk expert cache entries.

