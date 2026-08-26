# Expert Cache Subsystem in llama.cpp

The **Expert Cache** provides high-performance heterogeneous inference for Mixture-of-Experts (MoE) models (such as Qwen MoE, DeepSeek, Mixtral, Gemma MoE, and OLMoE) when offloaded across CPU host memory and GPU/accelerator VRAM.

---

## 1. Overview and Motivation

In standard CPU/GPU offloaded MoE inference:
1. Routing layers determine a sparse set of active experts per token (e.g. top-2 or top-8 out of 64+ experts).
2. For un-offloaded layers, expert weight matrices reside in host RAM and are transferred over the PCIe bus to accelerator memory on every token generation step.
3. Because expert routing exhibits high temporal and semantic locality, a small subset of experts ("hot experts") accounts for a large percentage of total activations during inference.

The Expert Cache maintains a dedicated, persistent accelerator-side buffer to hold recently or frequently used expert weights. Through slot-pool remapping and pinned DMA staging:
- **Zero-Copy In-Band Execution**: Active cache hits are evaluated directly in-place from device slot pools with **zero** Device-to-Device (D2D) copy overhead.
- **High-Throughput Pinned DMA**: Cache misses stage through a 32-entry genuinely page-locked host memory buffer (`cudaHostAlloc`), achieving full 14-16 GB/s hardware PCIe DMA throughput without blocking the CPU thread.
- **Universal Zero-Copy Execution**: Automatically activates zero-copy slot execution for any token batch size where the unique requested experts fit within available slot capacity.
- **Admission Hysteresis & Anti-Thrashing Guard**: Employs a 2-strike ghost filter and eviction cooldown to eliminate PCIe thrashing from transient one-off experts.
- **Automated Memory Planning**: Integrates with `--fit` via `--expert-cache auto` to automatically size the dynamic expert cache from remaining GPU VRAM headroom.

---

## 2. Architecture and High-Performance Vectors

```
        Host RAM (CPU)                        Accelerator (e.g. CUDA / Metal / Vulkan)
+----------------------------+                +-----------------------------------------+
| Host Weights (All Experts) |                | Unified Expert Cache (Strict Hard Cap)  |
| [Exp 0][Exp 1]...[Exp N]   |                | [Slot Pool 0: gate/up][Slot Pool 1:down]|
+--------------+-------------+                +--------------------+--------------------+
               |                                                   |
               | Miss: Pinned DMA Staging                          | Hit: Direct Zero-Copy Indexing
               | (32-Slot Page-Locked Host Arena)                  | (MUL_MAT_ID on slot_tensor)
               v                                                   v
        +-----------------------------------------------------------------+
        | Device Working Tensor (node->src[0] = slot_tensor)             |
        | [Slot 0 (Exp 4)][Slot 1 (Exp 12)][Slot 2 (Exp 31)]...           |
        +-----------------------------------------------------------------+
```

### 2.1 Vector 1: Zero-Copy Slot Pool Execution (`MUL_MAT_ID` Direct Remapping)

In traditional caching architectures, hits in the device cache are copied via D2D memory transfers into a temporary execution tensor (`input_cpy`). In 64-layer models with 3 projections (`gate`, `up`, `down`) and 8 active experts, this incurs ~1,536 D2D transfers on every token.

**Zero-Copy Execution** eliminates D2D transfers completely:
1. Sub-allocates 3D slot pool tensors (`[ne0, ne1, max_slots]`) directly within a single pre-allocated device buffer.
2. Intercepts `GGML_OP_MUL_MAT_ID` nodes in `ggml_backend_sched_compute_splits()`.
3. Maps router IDs in `ids_tensor` (node `src[2]`) directly to slot indices (`0 .. max_slots - 1`).
4. Replaces node `src[0]` with `slot_tensor`. The backend matrix multiplication kernel (`MUL_MAT_ID`) indexes directly into `vx + ids[i] * nb[2]`, computing the exact same result with **zero in-band memory copies**.

### 2.2 Vector 2: Universal Batch Execution

- **Batch-Independent Zero-Copy**: Zero-copy slot pool execution activates for any token batch size (single-token decode, speculative verification, or MTP draft batches) as long as the unique requested experts fit within available slot capacity.
- **Bulk Miss Fallback**: When requested distinct experts exceed slot capacity (e.g. large prompt prefill activating all 64 experts simultaneously), the cache automatically routes through bulk contiguous transfers, preserving full prompt processing throughput.

### 2.3 Vector 3: True Pinned Host DMA Staging Ring

When unpinned host memory is passed to asynchronous device copy APIs (`cudaMemcpyAsync`), drivers must either perform synchronous staging or serialize transfers.
- The Expert Cache allocates a 32-slot, 512-byte aligned page-locked host buffer (`cudaHostAlloc` with write-combining under CUDA).
- Cache misses copy slice payloads into an isolated staging slot at memory bus speeds (~40-60 GB/s) via CPU L1/L2 cache.
- The accelerator driver then performs asynchronous DMA over PCIe (14-16 GB/s) without blocking CPU execution or risking race conditions across concurrent layer transfers.

### 2.4 Vector 4: Admission Hysteresis & Anti-Thrashing Guard

To prevent cache pollution from transient one-off expert activations:
- **Two-Strike Ghost Filter**: When the cache is fully utilized, the first sighting of a missing expert is recorded in a lightweight ghost table without evicting an existing slot. Only if the expert is requested again within a 128-token window is it admitted to the probationary cache segment.
- **Eviction Readmission Cooldown**: Evicted experts require 8 fresh misses before being allowed to evict an existing resident slot, eliminating ping-pong oscillation.

### 2.5 Vector 5: Coordinated Atomic Layer Bundling (`{gate, up, down}`)

In SwiGLU architectures, each expert consists of three interdependent projections:
- `ffn_gate_exps`
- `ffn_up_exps`
- `ffn_down_exps`

During context initialization, all model layers register their triple projections via `ggml_backend_sched_register_expert_bundle()`. When checking residency or prefetching, operations are coordinated across `{gate, up, down}` to guarantee synchronized layer execution.

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

### 3.2 Slot Pool Structure (`ggml_expert_cache_slot_pool`)

Maintains 3D sub-allocated slot buffers matching projection dimensions:

```cpp
struct ggml_expert_cache_slot_pool {
    struct ggml_context * ctx = nullptr;
    struct ggml_tensor *  tensor = nullptr; // 3D tensor [ne0, ne1, max_slots]
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    enum ggml_type type = GGML_TYPE_F32;
    size_t  stride = 0;
    size_t  buffer_offset = 0;
    int32_t max_slots = 0;
    int32_t used_slots = 0;
    int32_t probationary_cap = 0;
    int32_t protected_cap = 0;
    int32_t probationary_used = 0;
    int32_t protected_used = 0;

    std::vector<ggml_expert_cache_slot_entry> slots;
    std::unordered_map<ggml_expert_cache_key, int32_t, ggml_expert_cache_key_hash> key_to_slot;
    std::unordered_map<ggml_expert_cache_key, uint64_t, ggml_expert_cache_key_hash> ghost_sightings;
    std::unordered_map<ggml_expert_cache_key, uint32_t, ggml_expert_cache_key_hash> eviction_miss_counts;
};
```

### 3.3 Cache Instance (`ggml_backend_expert_cache`)

Expert cache instances are allocated per accelerator backend:
- `buffer`: Single device buffer enforcing strict hard VRAM cap.
- `capacity`: Total byte capacity of the cache.
- `slot_pools`: Unified 3D slot pools sub-allocated from the device buffer.
- `tg_access_freq`: Frequency counter tracking expert accesses during token generation.
- `pinned_host_buffer`: 32-entry page-locked host memory staging arena (`cudaHostAlloc`).
- `registered_host_ranges`: Vector of host memory pointer ranges registered via `cudaHostRegister` for direct DMA (1 GiB budget cap).

---

## 4. Usage and CLI Flags

| Flag | Description | Default |
| :--- | :--- | :--- |
| `-exc SIZE`, `--expert-cache SIZE` | Size of VRAM cache (e.g. `1024M`, `1.5G`, `'auto'`) | `0` (disabled) |
| `-excp N`, `--expert-cache-period N` | Token interval between periodic rebalancing swaps (0 = on-demand SLRU) | `64` |
| `-excm N`, `--expert-cache-max-swaps N` | Maximum experts swapped per rebalance step (-1 = unlimited) | `-1` |
| `-excs`, `--expert-cache-stats` | Print telemetry and hit-rate statistics on exit | `false` |
| `-excr NAME`, `--expert-cache-profile NAME` | Name of profile for persistent hot-expert caching | `""` |
| `-excf PATH`, `--expert-cache-file PATH` | Explicit path to JSON file for persisting cache profile | `""` |
