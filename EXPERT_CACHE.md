# Expert Cache Subsystem in llama.cpp

The **Expert Cache** provides high-performance heterogeneous inference for Mixture-of-Experts (MoE) models (such as Qwen 3.5/3.6 MoE, DeepSeek, Mixtral, Gemma MoE, and OLMoE) when offloaded across CPU and GPU/accelerator memory.

---

## 1. Overview and Motivation

In standard CPU/GPU offloaded MoE inference:
1. Routing layers determine a sparse set of active experts per token (e.g. top-2 or top-8 out of 64+ experts).
2. For un-offloaded layers, expert weight matrices reside in host RAM and are transferred over the PCIe bus to accelerator memory on every token generation step.
3. Because expert routing exhibits high temporal and semantic locality, a small subset of experts ("hot experts") accounts for a large percentage of total activations during inference.

The Expert Cache maintains a dedicated, persistent accelerator-side buffer to hold recently or frequently used expert weights. When an active expert hits the cache:
- The weight transfer becomes an ultra-fast **Device-to-Device (D2D)** memory copy rather than a PCIe Host-to-Device (H2D) transfer.
- PCIe bandwidth consumption and latency are substantially reduced.

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

This ensures universal support across all MoE architectures (gate, up, down projections) regardless of layer depth or expert count.

### 2.2 Cache Entry (`ggml_backend_expert_cache_entry`)

Each resident expert in the cache tracks:
- `offset`: Byte offset within `cache->tensor`.
- `size`: Exact byte size of the expert matrix payload (`tensor->nb[2]`).
- `alloc_size`: 512-byte aligned slot allocation size.
- `last_used`: Logical clock timestamp of the last access (for LRU eviction).
- `hit_count`: Cumulative cache hit count.

### 2.3 Cache Instance (`ggml_backend_expert_cache`)

Expert cache instances are allocated per accelerator backend:
- `buffer`: Backend buffer allocated on the device.
- `tensor`: 1D `GGML_TYPE_I8` tensor spanning the cache capacity.
- `capacity`: Total byte capacity of the cache.
- `used`: Currently allocated byte size.
- `free_blocks`: Free-list tracking available contiguous offsets and sizes.
- `entries`: Hash table mapping `ggml_expert_cache_key` to cache entries.
- `access_freq`: Frequency counter tracking expert accesses across decode steps.

---

## 3. Operational Modes

### 3.1 Periodic Rebalancing Mode (Default: `--expert-cache-period 64`)

In periodic mode:
1. During token generation, `ggml_backend_expert_cache_record_access_count` increments expert access frequencies in `access_freq`.
2. Every `period` decode steps (`decode_step % period_tokens == 0`), `ggml_backend_expert_cache_rebalance` executes:
   - Evaluates expert access frequencies across all layers.
   - Determines the optimal set of top-k experts that fit within `capacity`.
   - Evicts unneeded entries, returning blocks to `free_blocks` and coalescing adjacent spans.
   - Allocates slots for newly promoted hot experts and issues asynchronous H2D transfers into `cache->tensor`.
   - Decays access frequencies (`freq >>= 1`) to adapt dynamically to shifting topic contexts.

### 3.2 On-Demand LRU Mode (`--expert-cache-period 0`)

When `period` is set to 0:
- When a cache miss occurs during single-token decoding, the fetched expert in `input_cpy` is copied D2D into `cache->tensor`.
- If the cache is full, least-recently-used (LRU) entries are evicted to make room.

---

## 4. Execution Pipeline inside `ggml-backend.cpp`

During graph evaluation for `GGML_OP_MUL_MAT_ID`:
1. **Decode Step Start**: `ggml_backend_expert_cache_begin_step(cache)` advances the clock and triggers periodic rebalance when due.
2. **Access Tracking and Partitioning**:
   - Inspects `ids_tensor` (node `src[2]`) to extract requested expert IDs.
   - Records access counts in `access_freq`.
   - Partitions requested experts into **hits** (present in `cache->entries`) and **misses**.
3. **Hit Processing**:
   - Performs D2D asynchronous copies from `cache->tensor` at `cache_offset` directly into `input_cpy` at `expert_offset`.
   - Copies exactly `expert_size` bytes per expert payload.
   - Records hit statistics.
4. **Miss Processing**:
   - Groups contiguous miss ranges and transfers them from host RAM to `input_cpy` over PCIe.
   - If on-demand LRU mode is enabled, registers newly loaded experts into the cache via D2D copy.
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

### CLI and Server Usage

- `--expert-cache-size <bytes>`: Size in bytes allocated for the expert cache on accelerator backends (e.g. `2147483648` for 2 GiB).
- `--expert-cache-period <tokens>`: Rebalance period in tokens (default: `64`). Set to `0` for on-demand LRU.
- `--expert-cache-stats`: Print runtime cache performance statistics (hit rate, avoided bandwidth).
- `--expert-cache-profile <name>`: Profile name for saved/loaded cache files.
- `--expert-cache-persist`: Automatically save accumulated hot-expert profile on server idle.

---

## 6. Memory Safety and Invariants

To prevent weight corruption and activation divergence (such as repetitive token degeneration):
- **Exact Payload Transfers**: Cache buffer slots and D2D copies must always use `expert_size` (`tensor->nb[2]`). Host-to-device MMQ padding offsets are strictly confined to the continuous host tensor and destination `input_cpy` buffers.
- **Defragmentation and Free-Block Coalescing**: All evictions during rebalancing return `{offset, alloc_size}` to `free_blocks` and invoke `ggml_backend_expert_cache_coalesce_free` before allocating new entries.
- **512-Byte Boundary Alignment**: All slot allocations are padded to `GGML_EXPERT_CACHE_ALIGN` (512 bytes) to satisfy backend DMA and SIMD alignment requirements.
