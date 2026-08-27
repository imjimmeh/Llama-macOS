#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Expert cache key: identifies an expert within a weight tensor
struct ggml_expert_cache_key {
    const struct ggml_tensor * tensor;
    int32_t expert_id;
};

// Expert bundle key: identifies an expert across an entire layer (gate, up, down)
struct ggml_expert_bundle_key {
    int32_t layer;
    int32_t expert_id;
};

// SLRU cache segments
enum ggml_expert_cache_segment {
    GGML_EXPERT_CACHE_SEG_PROBATIONARY = 0,
    GGML_EXPERT_CACHE_SEG_PROTECTED    = 1,
};

// Slot lifecycle states (Phase 2 & 4)
enum ggml_expert_cache_slot_state {
    GGML_EXPERT_CACHE_SLOT_EMPTY     = 0,
    GGML_EXPERT_CACHE_SLOT_LOADING   = 1,
    GGML_EXPERT_CACHE_SLOT_RESIDENT  = 2,
};

// Canonical MoE route descriptor (Epic 2)
struct ggml_cache_route_bundle {
    int32_t token;
    int32_t route;
    int32_t expert;
    int32_t gate_slot;
    int32_t up_slot;
    int32_t down_slot;
    int32_t gate_up_slot;
    bool    bundle_resident;
    bool    is_bundle_hit; // legacy alias
};
typedef struct ggml_cache_route_bundle ggml_moe_route_desc;

enum ggml_moe_bundle_kind {
    GGML_MOE_BUNDLE_SEPARATE_GATE_UP = 0,
    GGML_MOE_BUNDLE_FUSED_GATE_UP    = 1,
};

struct ggml_moe_bundle_plan {
    int32_t layer;
    enum ggml_moe_bundle_kind kind;
    const struct ggml_tensor * route_ids;
    struct ggml_tensor * gate_node;
    struct ggml_tensor * up_node;
    struct ggml_tensor * gate_up_node;
    struct ggml_tensor * down_node;
    struct ggml_tensor * layer_input;
    struct ggml_tensor * canonical_route_output;
    bool valid;
};

// Expert tensor map metadata
struct ggml_expert_map_meta {
    uint32_t offset;
    uint32_t n_expert;
};



typedef struct ggml_backend_expert_cache * ggml_backend_expert_cache_t;
GGML_API struct ggml_tensor * ggml_backend_find_mul_mat_id_node(
    const struct ggml_cgraph * graph,
    const struct ggml_tensor * input);

GGML_API bool ggml_backend_expert_cache_can_store(
    ggml_backend_expert_cache_t cache,
    size_t expert_size);


// Lifecycle
GGML_API ggml_backend_expert_cache_t ggml_backend_expert_cache_new(
    ggml_backend_t backend,
    size_t capacity);

GGML_API void ggml_backend_expert_cache_free(
    ggml_backend_expert_cache_t cache);

// Period & Mode settings
GGML_API void ggml_backend_expert_cache_set_period(
    ggml_backend_expert_cache_t cache,
    int32_t period);

GGML_API void ggml_backend_expert_cache_set_max_swaps(
    ggml_backend_expert_cache_t cache,
    int32_t max_swaps);

GGML_API int32_t ggml_backend_expert_cache_get_period(
    ggml_backend_expert_cache_t cache);

GGML_API void ggml_backend_expert_cache_begin_step(
    ggml_backend_expert_cache_t cache);

GGML_API void ggml_backend_expert_cache_rebalance(
    ggml_backend_expert_cache_t cache,
    int max_swaps);

enum ggml_expert_cache_phase {
    GGML_EXPERT_CACHE_PHASE_TG = 0,
    GGML_EXPERT_CACHE_PHASE_PP = 1,
};

// Access recording & SLRU touch
GGML_API void ggml_backend_expert_cache_record_access(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id);

GGML_API void ggml_backend_expert_cache_record_access_count(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id,
    uint32_t count,
    enum ggml_expert_cache_phase phase);

GGML_API void ggml_backend_expert_cache_process_jit_swaps(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * completed_tensor,
    ggml_backend_t backend);

GGML_API void ggml_backend_expert_cache_touch(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id);

// Telemetry
GGML_API void ggml_backend_expert_cache_record_hit(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id,
    size_t bytes_avoided);

GGML_API void ggml_backend_expert_cache_record_zero_copy_hit(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id,
    size_t bytes_avoided);

GGML_API void ggml_backend_expert_cache_record_miss(
    ggml_backend_expert_cache_t cache,
    size_t bytes_ram_to_gpu);
GGML_API void ggml_backend_expert_cache_record_eligible(
    ggml_backend_expert_cache_t cache);

GGML_API void ggml_backend_expert_cache_record_capacity_bypass(
    ggml_backend_expert_cache_t cache);
GGML_API void ggml_backend_expert_cache_record_cpu_backend_bypass(
    ggml_backend_expert_cache_t cache);
GGML_API void ggml_backend_expert_cache_record_mul_mat_id_input(
    ggml_backend_expert_cache_t cache);
GGML_API void ggml_backend_expert_cache_record_non_host_weight_bypass(
    ggml_backend_expert_cache_t cache);


// Legacy flat pool & offset lookup
GGML_API struct ggml_tensor * ggml_backend_expert_cache_get_tensor(
    ggml_backend_expert_cache_t cache);

GGML_API size_t ggml_backend_expert_cache_find_offset(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id);

// Phase 1: Zero-Copy Slot Pools & ID Remapping
GGML_API struct ggml_tensor * ggml_backend_expert_cache_get_slot_tensor(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * weight_tensor);

GGML_API int32_t ggml_backend_expert_cache_find_slot(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id);

GGML_API int32_t ggml_backend_expert_cache_alloc_slot_idx(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id,
    const struct ggml_expert_cache_key * pinned_keys,
    size_t n_pinned);

// Remap array of router expert IDs to slot indices.
// Returns number of hits.
GGML_API int32_t ggml_backend_expert_cache_remap_ids(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    const int32_t * original_ids,
    int32_t n_ids,
    int32_t * out_remapped_ids,
    bool * out_is_hit);

GGML_API void ggml_backend_expert_cache_record_cpu_id_remap(
    ggml_backend_expert_cache_t cache);

GGML_API void ggml_backend_expert_cache_record_staging_memcpy(
    ggml_backend_expert_cache_t cache,
    size_t bytes);

GGML_API void ggml_backend_expert_cache_record_direct_dma(
    ggml_backend_expert_cache_t cache,
    size_t bytes);

// Claim a slot and report whether the caller owns a new fill.
GGML_API int32_t ggml_backend_expert_cache_claim_slot_idx(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id,
    const struct ggml_expert_cache_key * pinned_keys,
    size_t n_pinned,
    bool * out_needs_load);

// Legacy byte-based slot allocation
GGML_API size_t ggml_backend_expert_cache_alloc_slot(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id,
    size_t size,
    const struct ggml_expert_cache_key * pinned_keys,
    size_t n_pinned);

// Phase 3: Expert Bundles
GGML_API void ggml_backend_expert_cache_register_bundle(
    ggml_backend_expert_cache_t cache,
    int32_t layer,
    const struct ggml_tensor * gate_tensor,
    const struct ggml_tensor * up_tensor,
    const struct ggml_tensor * down_tensor);

GGML_API void ggml_backend_expert_cache_register_fused_bundle(
    ggml_backend_expert_cache_t cache,
    int32_t layer,
    const struct ggml_tensor * gate_up_tensor,
    const struct ggml_tensor * down_tensor);

GGML_API bool ggml_backend_expert_cache_is_bundle_resident(
    ggml_backend_expert_cache_t cache,
    int32_t layer,
    int32_t expert_id);

GGML_API bool ggml_backend_expert_cache_is_layer_fully_resident(
    ggml_backend_expert_cache_t cache,
    int32_t layer);

GGML_API int32_t ggml_backend_expert_cache_partition_bundle_routes(
    ggml_backend_expert_cache_t cache,
    int32_t layer,
    const int32_t * router_ids,
    int32_t n_routes,
    int32_t n_tokens,
    struct ggml_cache_route_bundle * out_hit_routes,
    int32_t * out_n_hits,
    struct ggml_cache_route_bundle * out_miss_routes,
    int32_t * out_n_misses);

GGML_API struct ggml_tensor * ggml_backend_expert_cache_get_device_bundle_map(
    ggml_backend_expert_cache_t cache);

GGML_API struct ggml_tensor * ggml_backend_expert_cache_get_device_slot_map(
    ggml_backend_expert_cache_t cache);

GGML_API uint32_t ggml_backend_expert_cache_get_map_id(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor);

GGML_API bool ggml_backend_expert_cache_get_bundle_tensors(
    ggml_backend_expert_cache_t cache,
    int32_t layer,
    const struct ggml_tensor ** out_gate,
    const struct ggml_tensor ** out_up,
    const struct ggml_tensor ** out_down,
    const struct ggml_tensor ** out_gate_up);

GGML_API bool ggml_backend_expert_cache_has_tensor(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor);


GGML_API void ggml_backend_expert_cache_record_all_hit_resolution(
    ggml_backend_expert_cache_t cache);


GGML_API void ggml_backend_expert_cache_record_probe_layer(
    ggml_backend_expert_cache_t cache);

GGML_API void ggml_backend_expert_cache_record_probe_sync(
    ggml_backend_expert_cache_t cache,
    uint64_t us);

GGML_API void ggml_backend_expert_cache_record_probe_host(
    ggml_backend_expert_cache_t cache,
    uint64_t us);

GGML_API void ggml_backend_expert_cache_record_route_snapshot(
    ggml_backend_expert_cache_t cache);

GGML_API void ggml_backend_expert_cache_record_route_prefetch_submitted(
    ggml_backend_expert_cache_t cache);

GGML_API void ggml_backend_expert_cache_record_route_prefetch_duplicate(
    ggml_backend_expert_cache_t cache);

GGML_API void ggml_backend_expert_cache_record_route_prefetch_stale(
    ggml_backend_expert_cache_t cache);

GGML_API void ggml_backend_expert_cache_record_route_prefetch_rejected(
    ggml_backend_expert_cache_t cache);

GGML_API void ggml_backend_expert_cache_record_probe_upload(
    ggml_backend_expert_cache_t cache,
    uint64_t us);


// V2.2: Bounded Direct Host Page Registration
GGML_API bool ggml_backend_expert_cache_register_host_memory(
    ggml_backend_expert_cache_t cache,
    void * ptr,
    size_t size);

GGML_API bool ggml_backend_expert_cache_is_host_memory_registered(
    ggml_backend_expert_cache_t cache,
    const void * ptr,
    size_t size);

// Phase 5: Pinned Host Memory & Staging
GGML_API void * ggml_backend_expert_cache_get_pinned_buffer(
    ggml_backend_expert_cache_t cache,
    size_t required_size);

GGML_API void * ggml_backend_expert_cache_stage_acquire(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t slot_idx,
    size_t required_size);

GGML_API void ggml_backend_expert_cache_stage_commit(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t slot_idx);



GGML_API int32_t ggml_backend_expert_cache_get_tensor_layer(
    const struct ggml_tensor * tensor);

GGML_API void ggml_backend_expert_cache_prefetch(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    const int32_t * expert_ids,
    int32_t n_experts);

GGML_API void ggml_backend_expert_cache_prefetch_layer(
    ggml_backend_expert_cache_t cache,
    int32_t layer,
    const int32_t * expert_ids,
    int32_t n_experts);

// Stats & Seeding
GGML_API void ggml_backend_expert_cache_get_stats(
    ggml_backend_expert_cache_t cache,
    struct ggml_backend_expert_cache_stats * stats);

GGML_API void ggml_backend_expert_cache_print_stats(
    ggml_backend_expert_cache_t cache);

GGML_API size_t ggml_backend_expert_cache_export_entries(
    ggml_backend_expert_cache_t cache,
    struct ggml_backend_expert_cache_export_entry * out_entries,
    size_t max_entries);

GGML_API bool ggml_backend_expert_cache_seed(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id,
    uint32_t frequency);

GGML_API void ggml_backend_expert_cache_sync_staging(
    ggml_backend_expert_cache_t cache);

GGML_API void ggml_backend_expert_cache_promote_slot(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id,
    int32_t slot);
GGML_API bool ggml_backend_expert_cache_record_slot_use(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id,
    int32_t slot);

GGML_API void ggml_backend_expert_cache_sync(
    ggml_backend_expert_cache_t cache);

// Epic 4: Static Pinned Expert Manifests & Heterogeneous Routing
GGML_API bool ggml_backend_expert_cache_load_pinned_manifest(
    ggml_backend_expert_cache_t cache,
    const char * manifest_json_path);

GGML_API bool ggml_backend_expert_cache_is_pinned(
    ggml_backend_expert_cache_t cache,
    int32_t layer,
    int32_t expert_id);

GGML_API void ggml_backend_expert_cache_set_zero_miss_upload(
    ggml_backend_expert_cache_t cache,
    bool enabled);

GGML_API bool ggml_backend_expert_cache_get_zero_miss_upload(
    ggml_backend_expert_cache_t cache);

// Epic 5: Non-blocking Background Promotion Pipeline
GGML_API size_t ggml_backend_expert_cache_process_async_promotions(
    ggml_backend_expert_cache_t cache,
    size_t max_promotions);

GGML_API void ggml_backend_expert_cache_set_max_async_promotions(
    ggml_backend_expert_cache_t cache,
    int32_t max_promotions);

// Epic 6: GPU-Side Zero-Sync Route Remapping
GGML_API const int32_t * ggml_backend_expert_cache_get_gpu_slot_map(
    ggml_backend_expert_cache_t cache,
    int32_t layer);

#ifdef __cplusplus
}
#endif
