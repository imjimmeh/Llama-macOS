# Expert Cache Subsystem in llama.cpp

The **Expert Cache** provides high-performance heterogeneous inference for Mixture-of-Experts (MoE) models (such as Qwen MoE, DeepSeek, Mixtral, Gemma MoE, and OLMoE) when offloaded across CPU host memory and GPU/accelerator VRAM.

---

# Current Implementation Status (2026-08-26)

This document describes the current expert-cache source. The subsystem is functional for cache-eligible accelerator splits, but GTX 1080 / Compact measurements show no credible decode speedup over cache-disabled operation. The cache remains useful for VRAM-pressure fitting; throughput requires workload-specific evidence.

**Key decisions:**
- Cache disabled (`-exc 0`) is the throughput control for Qwen3.6-35B-A3B-APEX-Compact on GTX 1080.
- Device slot map (`d_expert_to_slot`) was removed; graph execution uses an explicit `ids_tensor` remap upload.
- Unified slot pools replace legacy flat-cache entries as the authoritative residency representation.
- Profile seeding populates slot pools through `alloc_slot_idx()`.
- Forced host-MoE CUDA routing was rejected: Compact forced-cache runs reached 14.68-15.99 TG tok/s against an approximately 26.5 tok/s CPU-routed control, and an MTP Quality forced run reached 9.19 TG tok/s with 169.7 GiB RAM-to-GPU traffic.
- The current carry-forward prefetch is opt-in and a no-op for Compact normal decode because no cache-eligible operation reaches it.
- General decode-time route-aware CPU/GPU dispatch is proposed, not implemented. See `docs/superpowers/specs/2026-08-26-general-decode-moe-dispatch-design.md` and `docs/superpowers/plans/2026-08-26-general-decode-moe-dispatch.md`.

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

**Status: IMPLEMENTED** (commit `367ea5347`)

Current legacy fallback handling can copy cache-hit slices into `input_cpy`. The slot-pool path avoids those in-band D2D copies only after an operation is already assigned to a cache-capable non-CPU backend.

**Zero-Copy Execution** for an eligible full route union:
1. Sub-allocates 3D slot pool tensors (`[ne0, ne1, max_slots]`) within a pre-allocated device buffer.
2. Intercepts a host-weight `GGML_OP_MUL_MAT_ID` split input in `ggml_backend_sched_compute_splits()`.
3. Maps router IDs in `ids_tensor` (`node->src[2]`) to slot indices (`0 .. max_slots - 1`).
4. Replaces `node->src[0]` with `slot_tensor`. The backend `MUL_MAT_ID` indexes the resident slot tensor without an in-band D2D copy.

### 2.2 Vector 2: Universal Slot-Pool Batch Execution

**Status: IMPLEMENTED, SUBJECT TO PLACEMENT**

- **Batch-Independent Slot Remapping**: Once a host-weight `MUL_MAT_ID` is assigned to a non-CPU cache backend, zero-copy slot-pool execution can handle any token batch whose unique requested-expert union fits available slots.
- **Not General Decode Dispatch**: CPU-routed MoE operations are not cache-eligible, regardless of batch capacity. The current implementation does not decide CPU versus GPU after current route IDs are known.
- **Legacy Fallback**: A union that cannot be made slot-ready uses the existing copied-tensor fallback. It does not use CPU-on-miss or mixed CPU/GPU expert execution.

### 2.3 Vector 3: True Pinned Host DMA Staging Ring

**Status: IMPLEMENTED** (commit `65e52abe6`)

When unpinned host memory is passed to asynchronous device copy APIs (`cudaMemcpyAsync`), drivers can stage or serialize the transfer.
- The CUDA cache path allocates a 32-slot, 512-byte aligned page-locked host buffer with `cudaHostAlloc` when available.
- Cache misses can copy slice payloads into isolated staging slots before device DMA.
- Reuse of an occupied staging slot waits for its recorded event. This protects staging memory lifetime; it is not a bounded fill-job queue or a proof that asynchronous slot publication is safe.

### 2.4 Vector 4: Admission Hysteresis and Anti-Thrashing Guard

**Status: IMPLEMENTED FOR FULL POOLS**

The cache mitigates transient admissions when a pool is full:
- **Two-Strike Ghost Filter**: The first missing expert is recorded in a ghost table. A second sighting within a 128-token window may enter probationary cache storage.
- **Eviction Readmission Cooldown**: An evicted expert requires eight fresh misses before it can evict a resident again.
- **Scope**: Empty pools still admit immediately. There is no bounded fill-job count, in-flight byte budget, or per-route fill budget.

### 2.5 Vector 5: Registered Expert Bundles

**Status: IMPLEMENTED REGISTRATION; ATOMIC DISPATCH PROPOSED**

SwiGLU expert projections can include:
- `ffn_gate_exps`
- `ffn_up_exps`
- `ffn_down_exps`

Context initialization registers related tensors through `ggml_backend_sched_register_expert_bundle()`. Current residency and prefetch helpers can query or request registered members. CUDA-backed slot publication now uses a completion event before lookup reports a hit. Atomic complete-bundle admission, consumer-use ownership, CPU-on-miss fallback, and route-aware GPU dispatch remain proposed in the general decode route-aware dispatch design.

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
    // Per-tensor slot maps: tensor pointer -> vector of slot indices for each expert_id (P0 fix)
    std::unordered_map<const struct ggml_tensor *, std::vector<int32_t>> slot_maps;
```

### 3.3 Cache Instance (`ggml_backend_expert_cache`)

Expert cache instances are allocated per accelerator backend:
- `buffer`: Single device buffer enforcing strict hard VRAM cap.
- `capacity`: Total byte capacity of the cache.
- `slot_pools`: Unified 3D slot pools sub-allocated from the device buffer.
- `tg_access_freq`: Frequency counter tracking expert accesses during token generation.
- `pp_access_freq`: Frequency counter tracking expert accesses during prefill (PP).
- `pinned_host_buffer`: 32-entry page-locked host memory staging arena (`cudaHostAlloc`).
- `registered_host_ranges`: Vector of host memory pointer ranges registered via `cudaHostRegister` for direct DMA (1 GiB budget cap).
- `bundle_registrations`: Expert bundle registrations for coordinated layer swaps.
- `pending_layer_swaps`: JIT staged layer swap queues.
- `staging_events` / `staging_in_flight`: Staging ring keyed by (tensor, slot_idx).
- `clock`: Monotonic clock for slot LRU ordering.
- `decode_step`: Current decode step counter.
- `period_tokens`: Token interval between periodic rebalancing swaps.
- `max_swaps`: Maximum experts swapped per rebalance step.
- `stats`: cache hit/miss, DMA, staging, probe, and route-prefetch telemetry (`ggml_backend_expert_cache_stats`).

---

## 4. Usage and CLI Flags

| Flag | Description | Default |
| :--- | :--- | :--- |
| `-exc SIZE`, `--expert-cache SIZE` | Size of VRAM cache (e.g. `1024M`, `1.5G`, `'auto'`) | `0` (disabled) |
| `-excp N`, `--expert-cache-period N` | Token interval between periodic rebalancing swaps (0 = on-demand SLRU) | `64` |
| `-excm N`, `--expert-cache-max-swaps N` | Maximum experts swapped per rebalance step (-1 = unlimited) | `-1` |
| `-excs`, `--expert-cache-stats` | Print telemetry and hit-rate statistics on exit | `false` |
| `-excr NAME`, `--expert-cache-profile NAME` | Name of profile for persistent hot-expert caching | `""` |
| `--expert-cache-prefetch` | Enable bounded decode carry-forward route prefetch (experimental, disabled by default) | `false` |

---

## 5. Implementation Status: Optimization Vectors

| Vector | Description | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Vector 1** | Zero-Copy Slot Pool Execution | **IMPLEMENTED** | Applies after host-weight `MUL_MAT_ID` reaches a non-CPU cache backend |
| **Vector 2** | Universal Slot-Pool Batch Execution | **IMPLEMENTED** | Any eligible route union that fits slots; not general CPU/GPU dispatch |
| **Vector 3** | Pinned Host DMA Staging Ring | **IMPLEMENTED** | CUDA pinned staging when available; staging safety is not a fill queue |
| **Vector 4** | Admission Hysteresis | **IMPLEMENTED** | Full-pool `ghost_sightings` plus `eviction_miss_counts` |
| **Vector 5** | Registered Expert Bundles | **IMPLEMENTED** | Registration/residency helpers exist; atomic route dispatch is proposed |
| **V2: Device Slot Map** | `d_expert_to_slot` GPU-resident lookup table | **REMOVED** | Explicit remapped IDs are active |
| **V2: Direct Page DMA** | Host registration for direct host-to-GPU DMA | **IMPLEMENTED** | `registered_host_ranges` has a 1 GiB registration cap |
| **General Decode Route Dispatch** | Current-route CPU/GPU choice for all decode microbatches | **PROPOSED** | See the dated design and implementation plan |


### 5.1 Carry-Forward Route Prefetch (Experimental)

`--expert-cache-prefetch` stores prior single-token route snapshots and prefetches at most one valid layer bundle on a later scheduler call. Snapshots are discarded on graph reset or step mismatch. It remains disabled by default, does not force host-resident MoE operations to CUDA, and is currently a correctness-preserving no-op for Compact normal decode.

The option is not the general route-aware dispatch design. It has no complete-bundle CPU-on-miss fallback, current-route decision boundary, bounded fill queue, route-generation identity, consumer-use ownership, or useful-after-fill admission policy. CUDA slot lookup now waits for load-event completion when the backend provides event queries.
---

## 6. Known Issues

### 6.1 Profile Seeding Order (Minor)

`common_expert_cache_sort_entries()` sorts by `(tensor_name, expert_id, frequency)` ascending before merging. The seeding loop iterates in this order, so lower-frequency entries are seeded first. When capacity is limited, hot entries at the end may not be seeded. The primary bug (seed not calling `alloc_slot_idx()`) is **fixed**. This ordering issue is a minor optimization gap.

### 6.2 GTX 1080 / Compact Profile: No Throughput Win

Benchmark sweep shows `-exc 0` (cache disabled) is the throughput control for Qwen3.6-35B-A3B-APEX-Compact on GTX 1080. No cache configuration beats the control beyond measurement noise. Cache remains useful for VRAM-pressure fitting via `--fit auto` but not for decode acceleration on this hardware.

### 6.3 Forced Routing Rejected

T5 forced-routing probe (TG128 dropped to 9.19 tok/s from ~26.5 tok/s baseline with 169.7 GiB RAM-to-GPU transfers). PCIe transfer cost dominates single-token decode regardless of rebalancing strategy.

### 6.4 Compact Carry-Forward Status

The Compact model's normal decode graph currently reports zero cache-eligible
operations, so carry-forward has no route data to submit on the GTX 1080.
The option is correctness-preserving but currently a no-op for this model.
See the dated measurements in `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`.

### 6.5 General Decode Route-Aware Dispatch: Proposed

The next performance design targets all decode-time MoE microbatches, including normal parallel generation, speculative verification, MTP target/draft graphs, and future batch forms. It will execute a complete current-route bundle from GPU slots only on a proven full hit; every incomplete route remains on the CPU path and may request a bounded background fill for later work.

This is not implemented. The design, invariants, telemetry, decision gates, and implementation order are documented in:

- `docs/superpowers/specs/2026-08-26-general-decode-moe-dispatch-design.md`
- `docs/superpowers/plans/2026-08-26-general-decode-moe-dispatch.md`
