#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-expert-cache.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_moe_hetero_scratch {
    // GPU Tensor Descriptors (backed by persistent device buffer outside gallocr)
    struct ggml_context * ctx_gpu;
    ggml_backend_buffer_t buf_gpu;
    struct ggml_tensor * hit_down_tensor;      // [d_model, max_routes, 1]
    struct ggml_tensor * cpu_upload_tensor;    // [d_model, max_routes, 1]
    struct ggml_tensor * merged_route_tensor;  // [d_model, max_routes, 1]
    struct ggml_tensor * gpu_hit_indices;      // [max_routes] (i32)
    struct ggml_tensor * gpu_miss_indices;     // [max_routes] (i32)

    // Pinned Host Memory
    void * host_pinned_buf;
    size_t host_pinned_cap;
    struct ggml_context * ctx_host;
    struct ggml_tensor * host_x_tensor;        // [d_model, 1, 1]
    struct ggml_tensor * host_down_tensor;     // [d_model, max_routes, 1]

    int64_t d_model;
    int64_t d_ff;
    int32_t max_routes;
    bool initialized;
};

typedef struct ggml_moe_hetero_scratch * ggml_moe_hetero_scratch_t;

enum { GGML_MOE_PARTIAL_MAX_ROUTES = 8 };

enum ggml_moe_partial_executor_result {
    GGML_MOE_PARTIAL_EXECUTOR_SUCCESS,
    GGML_MOE_PARTIAL_EXECUTOR_NOT_ADMITTED,
    GGML_MOE_PARTIAL_EXECUTOR_LAUNCH_FAILED,
};

struct ggml_moe_partial_route_snapshot {
    int32_t n_hits;
    int32_t n_misses;
    struct ggml_cache_route_bundle hits[GGML_MOE_PARTIAL_MAX_ROUTES];
    struct ggml_cache_route_bundle misses[GGML_MOE_PARTIAL_MAX_ROUTES];
};

struct ggml_moe_partial_activation {
    const void * host_data;
    size_t nbytes;
};

struct ggml_moe_partial_executor;
typedef struct ggml_moe_partial_executor * ggml_moe_partial_executor_t;

struct ggml_moe_route_ready_sidecar;
typedef struct ggml_moe_route_ready_sidecar * ggml_moe_route_ready_sidecar_t;

GGML_API ggml_moe_route_ready_sidecar_t ggml_moe_route_ready_sidecar_new(
    ggml_backend_t gpu_backend,
    ggml_backend_t cpu_backend,
    int64_t d_model,
    int64_t d_ff,
    int32_t top_k,
    bool is_fused);

GGML_API void ggml_moe_route_ready_sidecar_free(
    ggml_moe_route_ready_sidecar_t sidecar);

GGML_API enum ggml_status ggml_moe_route_ready_sidecar_execute_full_hit(
    ggml_moe_route_ready_sidecar_t sidecar,
    const struct ggml_moe_bundle_plan * bundle,
    ggml_backend_expert_cache_t cache,
    const int32_t * route_ids,
    int32_t n_route_ids,
    struct ggml_backend_expert_cache_stats * stats);

GGML_API ggml_moe_hetero_scratch_t ggml_moe_hetero_scratch_init(
    ggml_backend_t gpu_backend,
    int64_t d_model,
    int64_t d_ff,
    int32_t max_routes);

GGML_API void ggml_moe_hetero_scratch_free(
    ggml_moe_hetero_scratch_t scratch);

// Phase 1: Truly serial execution engine
GGML_API enum ggml_status ggml_backend_moe_hetero_execute_serial(
    ggml_backend_t gpu_backend,
    ggml_backend_t cpu_backend,
    const struct ggml_moe_bundle_plan * bundle,
    ggml_backend_expert_cache_t cache,
    const int32_t * ids_data,
    int32_t top_k,
    ggml_moe_hetero_scratch_t scratch,
    struct ggml_backend_expert_cache_stats * stats);

GGML_API bool ggml_moe_partial_route_snapshot_is_valid(
    const struct ggml_moe_partial_route_snapshot * snapshot,
    int32_t top_k);

GGML_API ggml_moe_partial_executor_t ggml_moe_partial_executor_new(
    ggml_backend_t gpu_backend,
    ggml_backend_t cpu_backend,
    const struct ggml_expert_bundle_weights * template_weights,
    int64_t d_model,
    int64_t d_ff,
    int32_t top_k,
    bool is_fused);

GGML_API void ggml_moe_partial_executor_free(
    ggml_moe_partial_executor_t executor);

GGML_API enum ggml_moe_partial_executor_result ggml_moe_partial_executor_execute(
    ggml_moe_partial_executor_t executor,
    const struct ggml_moe_bundle_plan * bundle,
    ggml_backend_expert_cache_t cache,
    const struct ggml_moe_partial_route_snapshot * snapshot,
    const struct ggml_moe_partial_activation * activation,
    struct ggml_backend_expert_cache_stats * stats);

#ifdef GGML_TEST
struct ggml_moe_partial_executor_test_state {
    bool gpu_variants[GGML_MOE_PARTIAL_MAX_ROUTES - 1];
    bool cpu_variants[GGML_MOE_PARTIAL_MAX_ROUTES - 1];
    bool has_merge_buffer;
    bool has_cpu_upload_buffer;
    ggml_backend_buffer_t exchange_buffer;
    struct ggml_tensor * gpu_outputs[GGML_MOE_PARTIAL_MAX_ROUTES - 1];
    struct ggml_tensor * cpu_outputs[GGML_MOE_PARTIAL_MAX_ROUTES - 1];
    bool events[9];
};

GGML_API bool ggml_moe_partial_executor_get_test_state(
    ggml_moe_partial_executor_t executor,
    struct ggml_moe_partial_executor_test_state * state);
#endif

#ifdef __cplusplus
}
#endif
