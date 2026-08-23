#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
// Slot lifecycle state machine: EMPTY -> LOADING -> RESIDENT.
// Typed via C++ enum class for safe field access in the slot entry struct.
// C callers only need the integer constants; they see them via the regular
// enum below.
enum class ggml_expert_slot_state {
    EMPTY = 0,
    LOADING = 1,
    RESIDENT = 2,
};
#endif

// C-visible companion enum with the same numeric values so C TUs can refer
// to the states by name. Not used by C++ code (which uses the enum class).
enum ggml_expert_slot_state_c {
    GGML_EXPERT_SLOT_STATE_EMPTY = 0,
    GGML_EXPERT_SLOT_STATE_LOADING = 1,
    GGML_EXPERT_SLOT_STATE_RESIDENT = 2,
};

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

// Access recording & SLRU touch
GGML_API void ggml_backend_expert_cache_record_access(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id);

GGML_API void ggml_backend_expert_cache_record_access_count(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id,
    uint32_t count);

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

// Same as find_slot but also returns LOADING entries; used for dedupe.
GGML_API int32_t ggml_backend_expert_cache_find_or_loading_slot(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id);

// Mark a previously LOADING slot as RESIDENT (data is now visible to consumers).
GGML_API void ggml_backend_expert_cache_mark_resident(
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

GGML_API bool ggml_backend_expert_cache_is_bundle_resident(
    ggml_backend_expert_cache_t cache,
    int32_t layer,
    int32_t expert_id);


GGML_API void ggml_backend_expert_cache_record_gpu_id_resolution(
    ggml_backend_expert_cache_t cache);

GGML_API void ggml_backend_expert_cache_record_probe_layer(
    ggml_backend_expert_cache_t cache);

GGML_API void ggml_backend_expert_cache_record_probe_sync(
    ggml_backend_expert_cache_t cache,
    uint64_t us);

GGML_API void ggml_backend_expert_cache_record_probe_host(
    ggml_backend_expert_cache_t cache,
    uint64_t us);

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

// The guard event is recorded on the stream that issued the staging copy.
enum ggml_expert_cache_stage_stream {
    GGML_EXPERT_CACHE_STAGE_BACKEND,
    GGML_EXPERT_CACHE_STAGE_PREFETCH,
};

GGML_API void ggml_backend_expert_cache_stage_commit(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t slot_idx,
    enum ggml_expert_cache_stage_stream stream = GGML_EXPERT_CACHE_STAGE_BACKEND);



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

GGML_API void ggml_backend_expert_cache_sync(
    ggml_backend_expert_cache_t cache);

// Phase 5A: Route Trace Collection for Predictability Analysis
// v1: fixed-size entry without logits; v2 appends a variable-length logits blob
struct ggml_expert_cache_route_trace_entry {
    uint64_t token_id;
    int32_t layer;
    int32_t n_experts;
    int32_t expert_ids[64];  // max 64 experts per layer (top-K routing)
    uint64_t timestamp_us;
};

GGML_API void ggml_backend_expert_cache_enable_route_trace(
    ggml_backend_expert_cache_t cache,
    const char * output_path,
    size_t max_entries);

// logits may be staged via record_router_logits; entries without staging
// get an empty blob that v2 readers skip
GGML_API void ggml_backend_expert_cache_record_route_trace(
    ggml_backend_expert_cache_t cache,
    int32_t layer,
    const int32_t * expert_ids,
    int32_t n_experts);

GGML_API void ggml_backend_expert_cache_flush_route_trace(
    ggml_backend_expert_cache_t cache);

// Stage router logits for a layer; attached to that layer's next trace entry
GGML_API void ggml_backend_expert_cache_record_router_logits(
    ggml_backend_expert_cache_t cache,
    int32_t layer,
    const float * logits,
    int32_t n_logits);

GGML_API void ggml_backend_expert_cache_disable_route_trace(
    ggml_backend_expert_cache_t cache);

// Phase 5C: Async DMA Pipeline
enum ggml_expert_cache_prefetch_state {
    GGML_EXPERT_CACHE_PREFETCH_EMPTY = 0,
    GGML_EXPERT_CACHE_PREFETCH_IN_FLIGHT = 1,
    GGML_EXPERT_CACHE_PREFETCH_RESIDENT = 2,
};

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

// True if an async DMA is in flight for this tensor/expert
GGML_API bool ggml_backend_expert_cache_has_inflight_prefetch(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id);

// True if this tensor/expert slot was placed by a completed predictor prefetch
GGML_API bool ggml_backend_expert_cache_was_prefetched(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id);

// Phase 5I: attribute slot executions to their source
GGML_API void ggml_backend_expert_cache_record_gpu_slot_from_prediction(ggml_backend_expert_cache_t cache);
GGML_API void ggml_backend_expert_cache_record_gpu_slot_reactive(ggml_backend_expert_cache_t cache);
GGML_API int32_t ggml_backend_expert_cache_prefetch_slot_count(ggml_backend_expert_cache_t cache);

// Phase 5H accessor hooks. used_ready/used_in_flight/used_miss classify slot
// readiness at node arrival for ALL executions (reactive + predicted); they do
// NOT measure predictor hit/miss. See record_gpu_slot_from_prediction/_reactive
// for attribution.
GGML_API void ggml_backend_expert_cache_record_gpu_slot_execution(ggml_backend_expert_cache_t cache);
GGML_API void ggml_backend_expert_cache_record_cpu_fallback(ggml_backend_expert_cache_t cache);
GGML_API void ggml_backend_expert_cache_record_used_ready(ggml_backend_expert_cache_t cache);
GGML_API void ggml_backend_expert_cache_record_used_in_flight(ggml_backend_expert_cache_t cache);
GGML_API void ggml_backend_expert_cache_record_used_miss(ggml_backend_expert_cache_t cache);
GGML_API void ggml_backend_expert_cache_record_already_resident(ggml_backend_expert_cache_t cache);
GGML_API void ggml_backend_expert_cache_record_in_flight_wait_us(ggml_backend_expert_cache_t cache, uint64_t us);



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

// Phase 5D: Learned Predictor (Low-Rank Model)
struct ggml_expert_cache_hidden_state_sample {
    int32_t layer;
    int32_t token_id;
    float hidden_state[256];  // Reduced dimension hidden state
    int32_t expert_ids[8];    // Top-8 experts for this layer
    int32_t n_experts;
};

GGML_API void ggml_backend_expert_cache_enable_hidden_state_trace(
    ggml_backend_expert_cache_t cache,
    const char * output_path,
    size_t max_samples);

GGML_API void ggml_backend_expert_cache_disable_hidden_state_trace(
    ggml_backend_expert_cache_t cache);

GGML_API void ggml_backend_expert_cache_record_hidden_state(
    ggml_backend_expert_cache_t cache,
    int32_t layer,
    int32_t token_id,
    const float * hidden_state,
    int32_t hidden_dim,
    const int32_t * expert_ids,
    int32_t n_experts);

GGML_API bool ggml_backend_expert_cache_load_learned_predictor(
    ggml_backend_expert_cache_t cache,
    const char * model_path);

GGML_API int32_t ggml_backend_expert_cache_predict_with_learned_model(
    ggml_backend_expert_cache_t cache,
    int32_t layer,
    const float * hidden_state,
    int32_t hidden_dim,
    int32_t * out_expert_ids,
    int32_t max_experts);

// Phase 5D: Prediction Submission (Revised Architecture)
// Submit predicted expert IDs for a future layer to trigger async prefetch
GGML_API void ggml_backend_expert_cache_submit_prediction(
    ggml_backend_expert_cache_t cache,
    int32_t target_layer,
    const int32_t * expert_ids,
    int32_t n_experts,
    const float * confidences);

// Phase 5D: Pending prediction entry (one per target layer; latest replaces)
struct ggml_expert_cache_pending_prediction {
    int32_t target_layer = -1;
    int32_t n_experts    = 0;
    int32_t expert_ids[64];
};

// Settle a pending prediction against the experts a layer actually requested.
// Returns true if a pending entry existed for this layer.
GGML_API bool ggml_backend_expert_cache_settle_prediction(
    ggml_backend_expert_cache_t cache,
    int32_t layer,
    const int32_t * actual_ids,
    int32_t n_actual,
    size_t pool_stride);

// Number of pending prediction entries (distinct target layers).
GGML_API int32_t ggml_backend_expert_cache_pending_prediction_count(
    ggml_backend_expert_cache_t cache);

// Copy the pending prediction for target_layer into out_ids (max max_ids).
// Returns n_experts stored, or 0 if none pending.
GGML_API int32_t ggml_backend_expert_cache_get_pending_prediction(
    ggml_backend_expert_cache_t cache,
    int32_t target_layer,
    int32_t * out_ids,
    int32_t max_ids);

// Routing predictor stats (single source of truth lives in the cache).
GGML_API bool ggml_backend_expert_cache_get_routing_predictor_stats(
    ggml_backend_expert_cache_t cache,
    struct ggml_routing_predictor_stats * out_stats);

// Accumulate decode-only predictions_generated count from the graph callback.
GGML_API void ggml_backend_expert_cache_add_predictions_generated(
    ggml_backend_expert_cache_t cache,
    int32_t n);





#ifdef __cplusplus
}
#endif
