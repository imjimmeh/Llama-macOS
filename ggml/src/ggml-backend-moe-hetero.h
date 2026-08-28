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

// Phase 2: Event-driven concurrent execution engine
GGML_API enum ggml_status ggml_backend_moe_hetero_execute_concurrent(
    ggml_backend_t gpu_backend,
    ggml_backend_t cpu_backend,
    const struct ggml_moe_bundle_plan * bundle,
    ggml_backend_expert_cache_t cache,
    const int32_t * ids_data,
    int32_t top_k,
    ggml_moe_hetero_scratch_t scratch,
    struct ggml_backend_expert_cache_stats * stats);

#ifdef __cplusplus
}
#endif
