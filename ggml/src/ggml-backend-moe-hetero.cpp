#include "ggml-backend-moe-hetero.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"

#if defined(GGML_USE_CUDA)
#include "ggml-cuda/moe-hetero.cuh"
#endif

#include <array>

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

ggml_moe_hetero_scratch_t ggml_moe_hetero_scratch_init(
        ggml_backend_t gpu_backend,
        int64_t d_model,
        int64_t d_ff,
        int32_t max_routes) {

    ggml_moe_hetero_scratch_t scratch = new ggml_moe_hetero_scratch();
    scratch->d_model = d_model;
    scratch->d_ff = d_ff;
    scratch->max_routes = max_routes;

    // 1. Allocate GPU persistent scratch tensors
    struct ggml_init_params params_gpu = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    scratch->ctx_gpu = ggml_init(params_gpu);
    scratch->hit_down_tensor     = ggml_new_tensor_3d(scratch->ctx_gpu, GGML_TYPE_F32, d_model, max_routes, 1);
    scratch->cpu_upload_tensor   = ggml_new_tensor_3d(scratch->ctx_gpu, GGML_TYPE_F32, d_model, max_routes, 1);
    scratch->merged_route_tensor = ggml_new_tensor_3d(scratch->ctx_gpu, GGML_TYPE_F32, d_model, max_routes, 1);
    scratch->gpu_hit_indices     = ggml_new_tensor_1d(scratch->ctx_gpu, GGML_TYPE_I32, max_routes);
    scratch->gpu_miss_indices    = ggml_new_tensor_1d(scratch->ctx_gpu, GGML_TYPE_I32, max_routes);

    scratch->buf_gpu = ggml_backend_alloc_ctx_tensors(scratch->ctx_gpu, gpu_backend);

    // 2. Allocate Pinned Host Memory
    const size_t host_x_bytes = (size_t)d_model * sizeof(float);
    const size_t host_down_bytes = (size_t)d_model * (size_t)max_routes * sizeof(float);
    scratch->host_pinned_cap = host_x_bytes + host_down_bytes + 1024;
    scratch->host_pinned_buf = malloc(scratch->host_pinned_cap);

    struct ggml_init_params params_host = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    scratch->ctx_host = ggml_init(params_host);
    scratch->host_x_tensor = ggml_new_tensor_3d(scratch->ctx_host, GGML_TYPE_F32, d_model, 1, 1);
    scratch->host_down_tensor = ggml_new_tensor_3d(scratch->ctx_host, GGML_TYPE_F32, d_model, max_routes, 1);

    scratch->initialized = true;
    return scratch;
}

void ggml_moe_hetero_scratch_free(ggml_moe_hetero_scratch_t scratch) {
    if (scratch == nullptr) return;

    if (scratch->buf_gpu) {
        ggml_backend_buffer_free(scratch->buf_gpu);
        scratch->buf_gpu = nullptr;
    }
    if (scratch->ctx_gpu) {
        ggml_free(scratch->ctx_gpu);
        scratch->ctx_gpu = nullptr;
    }
    if (scratch->ctx_host) {
        ggml_free(scratch->ctx_host);
        scratch->ctx_host = nullptr;
    }
    if (scratch->host_pinned_buf) {
        free(scratch->host_pinned_buf);
        scratch->host_pinned_buf = nullptr;
    }
    delete scratch;
}

enum {
    GGML_MOE_PARTIAL_VARIANT_COUNT = GGML_MOE_PARTIAL_MAX_ROUTES - 1,
    GGML_MOE_PARTIAL_EVENT_COUNT = 9,
    GGML_MOE_PARTIAL_EVENT_ACTIVATION_START = 0,
    GGML_MOE_PARTIAL_EVENT_ACTIVATION_END,
    GGML_MOE_PARTIAL_EVENT_GPU_GRAPH_START,
    GGML_MOE_PARTIAL_EVENT_GPU_GRAPH_END,
    GGML_MOE_PARTIAL_EVENT_CPU_UPLOAD_START,
    GGML_MOE_PARTIAL_EVENT_CPU_UPLOAD_END,
    GGML_MOE_PARTIAL_EVENT_SCATTER_START,
    GGML_MOE_PARTIAL_EVENT_SCATTER_END,
    GGML_MOE_PARTIAL_EVENT_FINAL_OUTPUT,
};

struct ggml_moe_partial_gpu_variant {
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    struct ggml_tensor * input = nullptr;
    struct ggml_tensor * gate_ids = nullptr;
    struct ggml_tensor * up_ids = nullptr;
    struct ggml_tensor * down_ids = nullptr;
    struct ggml_tensor * output = nullptr;
    struct ggml_cgraph * graph = nullptr;
    struct ggml_tensor gate_slot = {};
    struct ggml_tensor up_slot = {};
    struct ggml_tensor down_slot = {};
};

struct ggml_moe_partial_cpu_variant {
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    struct ggml_tensor * input = nullptr;
    struct ggml_tensor * ids = nullptr;
    struct ggml_tensor * output = nullptr;
    struct ggml_cgraph * graph = nullptr;
    struct ggml_tensor gate = {};
    struct ggml_tensor up = {};
    struct ggml_tensor down = {};
};

struct ggml_moe_partial_executor {
    ggml_backend_t gpu_backend;
    ggml_backend_t cpu_backend;
    struct ggml_expert_bundle_weights template_weights;
    int64_t d_model;
    int64_t d_ff;
    int32_t top_k;
    bool is_fused;
    std::array<struct ggml_moe_partial_gpu_variant, GGML_MOE_PARTIAL_VARIANT_COUNT> gpu = {};
    std::array<struct ggml_moe_partial_cpu_variant, GGML_MOE_PARTIAL_VARIANT_COUNT> cpu = {};
    struct ggml_context * merge_ctx = nullptr;
    ggml_backend_buffer_t merge_buffer = nullptr;
    struct ggml_tensor * merge = nullptr;
    struct ggml_tensor * cpu_upload = nullptr;
    struct ggml_context * exchange_ctx = nullptr;
    ggml_backend_buffer_t exchange_buffer = nullptr;
    struct ggml_tensor * exchange_activation = nullptr;
    struct ggml_tensor * exchange_hit_ids = nullptr;
    struct ggml_tensor * exchange_miss_ids = nullptr;
    struct ggml_tensor * exchange_miss_output = nullptr;
    std::array<ggml_backend_event_t, GGML_MOE_PARTIAL_EVENT_COUNT> events = {};
    ggml_backend_event_elapsed_us_t event_elapsed_us = nullptr;
    bool poisoned = false;
};

struct ggml_moe_route_ready_sidecar {
    ggml_backend_t gpu_backend;
    ggml_backend_t cpu_backend;
    int64_t d_model;
    int64_t d_ff;
    int32_t top_k;
    bool is_fused;
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    struct ggml_tensor * input = nullptr;
    struct ggml_tensor * gate_ids = nullptr;
    struct ggml_tensor * up_ids = nullptr;
    struct ggml_tensor * down_ids = nullptr;
    struct ggml_tensor * output = nullptr;
    struct ggml_cgraph * graph = nullptr;
    struct ggml_tensor gate_slot = {};
    struct ggml_tensor up_slot = {};
    struct ggml_tensor down_slot = {};
    std::vector<ggml_cache_route_bundle> routes;
    std::vector<ggml_cache_route_bundle> misses;
    std::vector<int32_t> gate_slots;
    std::vector<int32_t> up_slots;
    std::vector<int32_t> down_slots;
};

ggml_moe_route_ready_sidecar_t ggml_moe_route_ready_sidecar_new(
        ggml_backend_t gpu_backend,
        ggml_backend_t cpu_backend,
        int64_t d_model,
        int64_t d_ff,
        int32_t top_k,
        bool is_fused) {
    if (gpu_backend == nullptr || cpu_backend == nullptr || d_model <= 0 || d_ff <= 0 || top_k <= 0) {
        return nullptr;
    }

    auto * sidecar = new ggml_moe_route_ready_sidecar {
        gpu_backend, cpu_backend, d_model, d_ff, top_k, is_fused,
    };
    sidecar->routes.resize(top_k);
    sidecar->misses.resize(top_k);
    sidecar->gate_slots.resize(top_k);
    sidecar->up_slots.resize(top_k);
    sidecar->down_slots.resize(top_k);
    return sidecar;
}

void ggml_moe_route_ready_sidecar_free(ggml_moe_route_ready_sidecar_t sidecar) {
    if (sidecar == nullptr) {
        return;
    }
    if (sidecar->buffer != nullptr) {
        ggml_backend_buffer_free(sidecar->buffer);
    }
    if (sidecar->ctx != nullptr) {
        ggml_free(sidecar->ctx);
    }
    delete sidecar;
}

static void ggml_moe_route_ready_sidecar_copy_slot(
        struct ggml_tensor * destination,
        const struct ggml_tensor * source) {
    *destination = *source;
    memset(destination->src, 0, sizeof(destination->src));
    destination->view_src = nullptr;
    destination->extra = nullptr;
}

static void ggml_moe_route_ready_sidecar_unbind_slot(struct ggml_tensor * slot) {
    slot->buffer = nullptr;
    slot->data = nullptr;
}

static void ggml_moe_partial_gpu_variant_free(struct ggml_moe_partial_gpu_variant * variant) {
    if (variant->buffer != nullptr) {
        ggml_backend_buffer_free(variant->buffer);
        variant->buffer = nullptr;
    }
    if (variant->ctx != nullptr) {
        ggml_free(variant->ctx);
        variant->ctx = nullptr;
    }
}

static void ggml_moe_partial_cpu_variant_free(struct ggml_moe_partial_cpu_variant * variant) {
    if (variant->buffer != nullptr) {
        ggml_backend_buffer_free(variant->buffer);
        variant->buffer = nullptr;
    }
    if (variant->ctx != nullptr) {
        ggml_free(variant->ctx);
        variant->ctx = nullptr;
    }
}

static bool ggml_moe_partial_executor_init_gpu_variant(
        ggml_moe_partial_executor_t executor,
        struct ggml_moe_partial_gpu_variant * variant,
        int32_t n_routes) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    variant->ctx = ggml_init(params);
    if (variant->ctx == nullptr) {
        return false;
    }

    ggml_moe_route_ready_sidecar_copy_slot(
        &variant->gate_slot, executor->is_fused ? executor->template_weights.gate_up : executor->template_weights.gate);
    if (!executor->is_fused) {
        ggml_moe_route_ready_sidecar_copy_slot(&variant->up_slot, executor->template_weights.up);
    }
    ggml_moe_route_ready_sidecar_copy_slot(&variant->down_slot, executor->template_weights.down);

    variant->input = ggml_new_tensor_3d(variant->ctx, GGML_TYPE_F32, executor->d_model, 1, 1);
    variant->gate_ids = ggml_new_tensor_2d(variant->ctx, GGML_TYPE_I32, n_routes, 1);
    variant->up_ids = executor->is_fused ? variant->gate_ids :
        ggml_new_tensor_2d(variant->ctx, GGML_TYPE_I32, n_routes, 1);
    variant->down_ids = ggml_new_tensor_2d(variant->ctx, GGML_TYPE_I32, n_routes, 1);

    struct ggml_tensor * gate = nullptr;
    struct ggml_tensor * up = nullptr;
    if (executor->is_fused) {
        struct ggml_tensor * gate_up = ggml_mul_mat_id(variant->ctx, &variant->gate_slot, variant->input, variant->gate_ids);
        gate = ggml_view_3d(variant->ctx, gate_up, executor->d_ff, n_routes, 1, gate_up->nb[1], gate_up->nb[2], 0);
        up = ggml_view_3d(variant->ctx, gate_up, executor->d_ff, n_routes, 1, gate_up->nb[1], gate_up->nb[2],
            executor->d_ff * gate_up->nb[0]);
    } else {
        gate = ggml_mul_mat_id(variant->ctx, &variant->gate_slot, variant->input, variant->gate_ids);
        up = ggml_mul_mat_id(variant->ctx, &variant->up_slot, variant->input, variant->up_ids);
    }
    variant->output = ggml_mul_mat_id(
        variant->ctx, &variant->down_slot, ggml_swiglu_split(variant->ctx, gate, up), variant->down_ids);
    variant->graph = ggml_new_graph(variant->ctx);
    ggml_build_forward_expand(variant->graph, variant->output);
    variant->buffer = ggml_backend_alloc_ctx_tensors(variant->ctx, executor->gpu_backend);

    ggml_moe_route_ready_sidecar_unbind_slot(&variant->gate_slot);
    if (!executor->is_fused) {
        ggml_moe_route_ready_sidecar_unbind_slot(&variant->up_slot);
    }
    ggml_moe_route_ready_sidecar_unbind_slot(&variant->down_slot);
    return variant->buffer != nullptr;
}

static bool ggml_moe_partial_executor_init_cpu_variant(
        ggml_moe_partial_executor_t executor,
        struct ggml_moe_partial_cpu_variant * variant,
        int32_t n_routes) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    variant->ctx = ggml_init(params);
    if (variant->ctx == nullptr) {
        return false;
    }

    ggml_moe_route_ready_sidecar_copy_slot(
        &variant->gate, executor->is_fused ? executor->template_weights.gate_up : executor->template_weights.gate);
    if (!executor->is_fused) {
        ggml_moe_route_ready_sidecar_copy_slot(&variant->up, executor->template_weights.up);
    }
    ggml_moe_route_ready_sidecar_copy_slot(&variant->down, executor->template_weights.down);

    variant->input = ggml_new_tensor_3d(variant->ctx, GGML_TYPE_F32, executor->d_model, 1, 1);
    variant->ids = ggml_new_tensor_2d(variant->ctx, GGML_TYPE_I32, n_routes, 1);

    struct ggml_tensor * gate = nullptr;
    struct ggml_tensor * up = nullptr;
    if (executor->is_fused) {
        struct ggml_tensor * gate_up = ggml_mul_mat_id(variant->ctx, &variant->gate, variant->input, variant->ids);
        gate = ggml_view_3d(variant->ctx, gate_up, executor->d_ff, n_routes, 1, gate_up->nb[1], gate_up->nb[2], 0);
        up = ggml_view_3d(variant->ctx, gate_up, executor->d_ff, n_routes, 1, gate_up->nb[1], gate_up->nb[2],
            executor->d_ff * gate_up->nb[0]);
    } else {
        gate = ggml_mul_mat_id(variant->ctx, &variant->gate, variant->input, variant->ids);
        up = ggml_mul_mat_id(variant->ctx, &variant->up, variant->input, variant->ids);
    }
    variant->output = ggml_mul_mat_id(
        variant->ctx, &variant->down, ggml_swiglu_split(variant->ctx, gate, up), variant->ids);
    variant->graph = ggml_new_graph(variant->ctx);
    ggml_build_forward_expand(variant->graph, variant->output);
    variant->buffer = ggml_backend_alloc_ctx_tensors(variant->ctx, executor->cpu_backend);

    ggml_moe_route_ready_sidecar_unbind_slot(&variant->gate);
    if (!executor->is_fused) {
        ggml_moe_route_ready_sidecar_unbind_slot(&variant->up);
    }
    ggml_moe_route_ready_sidecar_unbind_slot(&variant->down);
    return variant->buffer != nullptr;
}

static bool ggml_moe_partial_executor_init_merge_buffers(ggml_moe_partial_executor_t executor) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    executor->merge_ctx = ggml_init(params);
    if (executor->merge_ctx == nullptr) {
        return false;
    }
    executor->merge = ggml_new_tensor_3d(
        executor->merge_ctx, GGML_TYPE_F32, executor->d_model, GGML_MOE_PARTIAL_MAX_ROUTES, 1);
    executor->cpu_upload = ggml_new_tensor_3d(
        executor->merge_ctx, GGML_TYPE_F32, executor->d_model, GGML_MOE_PARTIAL_MAX_ROUTES, 1);
    executor->merge_buffer = ggml_backend_alloc_ctx_tensors(executor->merge_ctx, executor->gpu_backend);
    return executor->merge_buffer != nullptr;
}

static bool ggml_moe_partial_executor_init_exchange(ggml_moe_partial_executor_t executor) {
    ggml_backend_dev_t gpu_device = ggml_backend_get_device(executor->gpu_backend);
    ggml_backend_buffer_type_t host_buft = gpu_device ? ggml_backend_dev_host_buffer_type(gpu_device) : nullptr;
    if (host_buft == nullptr) {
        return false;
    }
    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    executor->exchange_ctx = ggml_init(params);
    if (executor->exchange_ctx == nullptr) {
        return false;
    }
    executor->exchange_activation = ggml_new_tensor_3d(executor->exchange_ctx, GGML_TYPE_F32, executor->d_model, 1, 1);
    executor->exchange_hit_ids = ggml_new_tensor_2d(
        executor->exchange_ctx, GGML_TYPE_I32, 3 * GGML_MOE_PARTIAL_MAX_ROUTES, 1);
    executor->exchange_miss_ids = ggml_new_tensor_2d(
        executor->exchange_ctx, GGML_TYPE_I32, GGML_MOE_PARTIAL_MAX_ROUTES, 1);
    executor->exchange_miss_output = ggml_new_tensor_2d(
        executor->exchange_ctx, GGML_TYPE_F32, executor->d_model, GGML_MOE_PARTIAL_MAX_ROUTES);
    executor->exchange_buffer = ggml_backend_alloc_ctx_tensors_from_buft(executor->exchange_ctx, host_buft);
    if (executor->exchange_buffer == nullptr) {
        return false;
    }

    const bool pinned = ggml_backend_buffer_get_type(executor->exchange_buffer) == host_buft;
    GGML_LOG_DEBUG("%s: exchange buffer %s pinned=%d\n", __func__,
        ggml_backend_buffer_name(executor->exchange_buffer), (int) pinned);
    return pinned;
}

static bool ggml_moe_partial_executor_init_events(ggml_moe_partial_executor_t executor) {
    ggml_backend_dev_t gpu_device = ggml_backend_get_device(executor->gpu_backend);
    if (gpu_device == nullptr) {
        return false;
    }
    for (ggml_backend_event_t & event : executor->events) {
        event = ggml_backend_event_new(gpu_device);
        if (event == nullptr) {
            return false;
        }
    }
    return true;
}

static bool ggml_moe_route_ready_sidecar_bind_weights(
        ggml_moe_route_ready_sidecar_t sidecar,
        ggml_backend_expert_cache_t cache,
        const struct ggml_expert_bundle_weights & weights) {
    struct ggml_tensor * slot_gate = weights.gate ? ggml_backend_expert_cache_get_slot_tensor(cache, weights.gate) : nullptr;
    struct ggml_tensor * slot_up = weights.up ? ggml_backend_expert_cache_get_slot_tensor(cache, weights.up) : nullptr;
    struct ggml_tensor * slot_gate_up = weights.gate_up ? ggml_backend_expert_cache_get_slot_tensor(cache, weights.gate_up) : nullptr;
    struct ggml_tensor * slot_down = weights.down ? ggml_backend_expert_cache_get_slot_tensor(cache, weights.down) : nullptr;
    if (slot_down == nullptr || (sidecar->is_fused ? slot_gate_up == nullptr : slot_gate == nullptr || slot_up == nullptr)) {
        return false;
    }

    ggml_moe_route_ready_sidecar_copy_slot(&sidecar->gate_slot, sidecar->is_fused ? slot_gate_up : slot_gate);
    if (!sidecar->is_fused) {
        ggml_moe_route_ready_sidecar_copy_slot(&sidecar->up_slot, slot_up);
    }
    ggml_moe_route_ready_sidecar_copy_slot(&sidecar->down_slot, slot_down);
    return true;
}

static void ggml_moe_route_ready_sidecar_unbind_weights(ggml_moe_route_ready_sidecar_t sidecar) {
    ggml_moe_route_ready_sidecar_unbind_slot(&sidecar->gate_slot);
    if (!sidecar->is_fused) {
        ggml_moe_route_ready_sidecar_unbind_slot(&sidecar->up_slot);
    }
    ggml_moe_route_ready_sidecar_unbind_slot(&sidecar->down_slot);
}

static bool ggml_moe_route_ready_sidecar_validate_bundle(
        ggml_moe_route_ready_sidecar_t sidecar,
        const struct ggml_moe_bundle_plan * bundle,
        const struct ggml_expert_bundle_weights & weights) {
    if (!bundle->valid || bundle->route_ids == nullptr ||
        bundle->route_ids->ne[0] != sidecar->top_k || bundle->route_ids->ne[1] != 1 ||
        bundle->layer_input->type != GGML_TYPE_F32 || bundle->down_node->type != GGML_TYPE_F32 ||
        bundle->layer_input->ne[0] != sidecar->d_model || bundle->layer_input->ne[1] != 1 || bundle->layer_input->ne[2] != 1 ||
        bundle->down_node->ne[0] != sidecar->d_model || bundle->down_node->ne[1] != sidecar->top_k || bundle->down_node->ne[2] != 1 ||
        bundle->layer_input->buffer == nullptr || bundle->down_node->buffer == nullptr ||
        !ggml_backend_buffer_is_host(bundle->layer_input->buffer) ||
        !ggml_backend_buffer_is_host(bundle->down_node->buffer) ||
        weights.down == nullptr || weights.down->ne[0] != sidecar->d_ff || weights.down->ne[1] != sidecar->d_model ||
        bundle->down_node->src[0] != weights.down) {
        return false;
    }

    if (sidecar->is_fused) {
        return weights.gate_up != nullptr && bundle->gate_up_node != nullptr &&
            weights.gate_up->ne[0] == sidecar->d_model && weights.gate_up->ne[1] == 2 * sidecar->d_ff &&
            bundle->gate_up_node->src[0] == weights.gate_up &&
            bundle->gate_up_node->src[1] == bundle->layer_input;
    }

    return weights.gate != nullptr && weights.up != nullptr &&
        bundle->gate_node != nullptr && bundle->up_node != nullptr &&
        weights.gate->ne[0] == sidecar->d_model && weights.gate->ne[1] == sidecar->d_ff &&
        weights.up->ne[0] == sidecar->d_model && weights.up->ne[1] == sidecar->d_ff &&
        bundle->gate_node->src[0] == weights.gate && bundle->up_node->src[0] == weights.up &&
        bundle->gate_node->src[1] == bundle->layer_input && bundle->up_node->src[1] == bundle->layer_input;
}


static bool ggml_moe_route_ready_sidecar_init(
        ggml_moe_route_ready_sidecar_t sidecar,
        ggml_backend_expert_cache_t cache,
        const struct ggml_expert_bundle_weights & weights) {
    if (sidecar->ctx != nullptr) {
        return sidecar->buffer != nullptr;
    }
    if (!ggml_moe_route_ready_sidecar_bind_weights(sidecar, cache, weights)) {
        return false;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    sidecar->ctx = ggml_init(params);
    if (sidecar->ctx == nullptr) {
        ggml_moe_route_ready_sidecar_unbind_weights(sidecar);
        return false;
    }

    sidecar->input = ggml_new_tensor_3d(sidecar->ctx, GGML_TYPE_F32, sidecar->d_model, 1, 1);
    sidecar->gate_ids = ggml_new_tensor_2d(sidecar->ctx, GGML_TYPE_I32, sidecar->top_k, 1);
    sidecar->up_ids = sidecar->is_fused ? sidecar->gate_ids :
        ggml_new_tensor_2d(sidecar->ctx, GGML_TYPE_I32, sidecar->top_k, 1);
    sidecar->down_ids = ggml_new_tensor_2d(sidecar->ctx, GGML_TYPE_I32, sidecar->top_k, 1);

    struct ggml_tensor * gate = nullptr;
    struct ggml_tensor * up = nullptr;
    if (sidecar->is_fused) {
        struct ggml_tensor * gate_up = ggml_mul_mat_id(sidecar->ctx, &sidecar->gate_slot, sidecar->input, sidecar->gate_ids);
        gate = ggml_view_3d(sidecar->ctx, gate_up, sidecar->d_ff, sidecar->top_k, 1, gate_up->nb[1], gate_up->nb[2], 0);
        up = ggml_view_3d(sidecar->ctx, gate_up, sidecar->d_ff, sidecar->top_k, 1, gate_up->nb[1], gate_up->nb[2], sidecar->d_ff * gate_up->nb[0]);
    } else {
        gate = ggml_mul_mat_id(sidecar->ctx, &sidecar->gate_slot, sidecar->input, sidecar->gate_ids);
        up = ggml_mul_mat_id(sidecar->ctx, &sidecar->up_slot, sidecar->input, sidecar->up_ids);
    }
    sidecar->output = ggml_mul_mat_id(sidecar->ctx, &sidecar->down_slot, ggml_swiglu_split(sidecar->ctx, gate, up), sidecar->down_ids);
    sidecar->graph = ggml_new_graph(sidecar->ctx);
    ggml_build_forward_expand(sidecar->graph, sidecar->output);
    sidecar->buffer = ggml_backend_alloc_ctx_tensors(sidecar->ctx, sidecar->gpu_backend);
    ggml_moe_route_ready_sidecar_unbind_weights(sidecar);
    if (sidecar->buffer == nullptr) {
        ggml_free(sidecar->ctx);
        sidecar->ctx = nullptr;
        sidecar->input = nullptr;
        sidecar->gate_ids = nullptr;
        sidecar->up_ids = nullptr;
        sidecar->down_ids = nullptr;
        sidecar->output = nullptr;
        sidecar->graph = nullptr;
        return false;
    }
    return true;
}

enum ggml_status ggml_moe_route_ready_sidecar_execute_full_hit(
        ggml_moe_route_ready_sidecar_t sidecar,
        const struct ggml_moe_bundle_plan * bundle,
        ggml_backend_expert_cache_t cache,
        const int32_t * route_ids,
        int32_t n_route_ids,
        struct ggml_backend_expert_cache_stats * stats) {
    if (sidecar == nullptr || bundle == nullptr || cache == nullptr || route_ids == nullptr ||
        bundle->layer_input == nullptr || bundle->down_node == nullptr ||
        n_route_ids != sidecar->top_k || bundle->is_fused != sidecar->is_fused) {
        return GGML_STATUS_FAILED;
    }

    struct ggml_expert_bundle_weights weights = {};
    if (!ggml_backend_expert_cache_get_bundle_weights(cache, bundle->layer, &weights) ||
        !ggml_moe_route_ready_sidecar_validate_bundle(sidecar, bundle, weights)) {
        return GGML_STATUS_FAILED;
    }

    auto & routes = sidecar->routes;
    auto & misses = sidecar->misses;
    int32_t n_hits = 0;
    int32_t n_misses = 0;
    ggml_backend_expert_cache_partition_bundle_routes(
        cache, bundle->layer, route_ids, n_route_ids, 1,
        routes.data(), &n_hits, misses.data(), &n_misses);
    if (n_hits != n_route_ids || n_misses != 0 ||
        !ggml_moe_route_ready_sidecar_init(sidecar, cache, weights) ||
        !ggml_moe_route_ready_sidecar_bind_weights(sidecar, cache, weights)) {
        return GGML_STATUS_FAILED;
    }
    if (ggml_nbytes(bundle->layer_input) != ggml_nbytes(sidecar->input) ||
        ggml_nbytes(bundle->down_node) != ggml_nbytes(sidecar->output)) {
        ggml_moe_route_ready_sidecar_unbind_weights(sidecar);
        return GGML_STATUS_FAILED;
    }
    ggml_backend_expert_cache_reserve_bundle_slots(cache, bundle->layer, routes.data(), n_hits);
    auto & gate_slots = sidecar->gate_slots;
    auto & up_slots = sidecar->up_slots;
    auto & down_slots = sidecar->down_slots;
    for (int32_t i = 0; i < n_route_ids; ++i) {
        gate_slots[i] = sidecar->is_fused ? routes[i].gate_up_slot : routes[i].gate_slot;
        up_slots[i] = sidecar->is_fused ? routes[i].gate_up_slot : routes[i].up_slot;
        down_slots[i] = routes[i].down_slot;
        if (gate_slots[i] < 0 || up_slots[i] < 0 || down_slots[i] < 0) {
            ggml_backend_expert_cache_release_bundle_slots(cache, bundle->layer, routes.data(), n_hits, nullptr);
            ggml_moe_route_ready_sidecar_unbind_weights(sidecar);
            return GGML_STATUS_FAILED;
        }
    }

    ggml_backend_tensor_set_async(
        sidecar->gpu_backend, sidecar->input, bundle->layer_input->data, 0, ggml_nbytes(bundle->layer_input));
    ggml_backend_tensor_set_async(
        sidecar->gpu_backend, sidecar->gate_ids, gate_slots.data(), 0, n_route_ids * sizeof(int32_t));
    if (!sidecar->is_fused) {
        ggml_backend_tensor_set_async(
            sidecar->gpu_backend, sidecar->up_ids, up_slots.data(), 0, n_route_ids * sizeof(int32_t));
    }
    ggml_backend_tensor_set_async(
        sidecar->gpu_backend, sidecar->down_ids, down_slots.data(), 0, n_route_ids * sizeof(int32_t));
    const enum ggml_status status = ggml_backend_graph_compute_async(sidecar->gpu_backend, sidecar->graph);
    if (status == GGML_STATUS_SUCCESS) {
        GGML_ASSERT(ggml_are_same_layout(sidecar->output, bundle->down_node));
        ggml_backend_tensor_get_async(
            sidecar->gpu_backend, sidecar->output, bundle->down_node->data, 0, ggml_nbytes(sidecar->output));
        for (int32_t i = 0; i < n_route_ids; ++i) {
            bool seen = false;
            for (int32_t j = 0; j < i; ++j) {
                if (route_ids[j] == route_ids[i]) {
                    seen = true;
                    break;
                }
            }
            if (seen) {
                continue;
            }
            ggml_backend_expert_cache_record_zero_copy_hit(cache, weights.down, route_ids[i], weights.down->nb[2]);
            if (stats != nullptr) {
                stats->n_zero_copy_hits++;
            }
            if (sidecar->is_fused) {
                ggml_backend_expert_cache_record_zero_copy_hit(cache, weights.gate_up, route_ids[i], weights.gate_up->nb[2]);
                if (stats != nullptr) {
                    stats->n_zero_copy_hits++;
                }
            } else {
                ggml_backend_expert_cache_record_zero_copy_hit(cache, weights.gate, route_ids[i], weights.gate->nb[2]);
                ggml_backend_expert_cache_record_zero_copy_hit(cache, weights.up, route_ids[i], weights.up->nb[2]);
                if (stats != nullptr) {
                    stats->n_zero_copy_hits += 2;
                }
            }
        }
    }
    ggml_backend_synchronize(sidecar->gpu_backend);
    ggml_moe_route_ready_sidecar_unbind_weights(sidecar);
    ggml_backend_expert_cache_release_bundle_slots(cache, bundle->layer, routes.data(), n_hits, nullptr);
    return status;
}

// Phase 1: Truly serial execution engine
enum ggml_status ggml_backend_moe_hetero_execute_serial(
        ggml_backend_t gpu_backend,
        ggml_backend_t cpu_backend,
        const struct ggml_moe_bundle_plan * bundle,
        ggml_backend_expert_cache_t cache,
        const int32_t * ids_data,
        int32_t top_k,
        ggml_moe_hetero_scratch_t scratch,
        struct ggml_backend_expert_cache_stats * stats) {

    if (!gpu_backend || !cpu_backend || !bundle || !cache || !ids_data || top_k <= 0) {
        fprintf(stderr, "[ERROR] hetero_execute_serial: invalid args: gpu=%p cpu=%p bundle=%p cache=%p ids=%p top_k=%d\n",
            (void*)gpu_backend, (void*)cpu_backend, (void*)bundle, (void*)cache, (void*)ids_data, top_k);
        return GGML_STATUS_FAILED;
    }

    struct ggml_tensor * gate_node = bundle->gate_node;
    struct ggml_tensor * up_node   = bundle->up_node;
    struct ggml_tensor * gu_node   = bundle->gate_up_node;
    struct ggml_tensor * down_node = bundle->down_node;
    struct ggml_tensor * input_x   = bundle->layer_input;

    if (!down_node || !input_x) {
        fprintf(stderr, "[ERROR] hetero_execute_serial: down_node=%p input_x=%p layer=%d\n",
            (void*)down_node, (void*)input_x, bundle->layer);
        return GGML_STATUS_FAILED;
    }

    const int64_t d_model = down_node->ne[0];
    const int64_t d_ff    = gu_node ? (gu_node->ne[0] / 2) : (gate_node ? gate_node->ne[0] : 0);

    const int32_t n_tokens = (int32_t)(bundle->route_ids ? bundle->route_ids->ne[1] : 1);
    if (n_tokens > 1) {
        struct ggml_expert_bundle_weights bw = {};
        if (!ggml_backend_expert_cache_get_bundle_weights(cache, bundle->layer, &bw)) {
            fprintf(stderr, "[ERROR] hetero_execute_serial: failed to get bundle weights for layer=%d\n", bundle->layer);
            return GGML_STATUS_FAILED;
        }

        struct ggml_init_params params_cpu = {
            /*.mem_size   =*/ 16 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * ctx_cpu = ggml_init(params_cpu);

        struct ggml_tensor * cpu_x = ggml_new_tensor_3d(ctx_cpu, GGML_TYPE_F32, d_model, 1, n_tokens);
        struct ggml_tensor * cpu_ids = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_I32, top_k, n_tokens);

        struct ggml_tensor * cpu_gate = nullptr;
        struct ggml_tensor * cpu_up   = nullptr;

        if (bundle->is_fused) {
            struct ggml_tensor * cpu_gu = ggml_mul_mat_id(ctx_cpu, bw.gate_up, cpu_x, cpu_ids);
            cpu_gate = ggml_view_3d(ctx_cpu, cpu_gu, d_ff, top_k, n_tokens, cpu_gu->nb[1], cpu_gu->nb[2], 0);
            cpu_up   = ggml_view_3d(ctx_cpu, cpu_gu, d_ff, top_k, n_tokens, cpu_gu->nb[1], cpu_gu->nb[2], d_ff * cpu_gu->nb[0]);
        } else {
            cpu_gate = ggml_mul_mat_id(ctx_cpu, bw.gate, cpu_x, cpu_ids);
            cpu_up   = ggml_mul_mat_id(ctx_cpu, bw.up,   cpu_x, cpu_ids);
        }

        struct ggml_tensor * cpu_silu = ggml_silu(ctx_cpu, cpu_gate);
        struct ggml_tensor * cpu_act  = ggml_mul(ctx_cpu, cpu_silu, cpu_up);
        struct ggml_tensor * cpu_down = ggml_mul_mat_id(ctx_cpu, bw.down, cpu_act, cpu_ids);

        ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, cpu_backend);

        ggml_backend_tensor_get(bundle->layer_input, cpu_x->data, 0, (size_t)d_model * n_tokens * sizeof(float));
        ggml_backend_tensor_get(bundle->route_ids, cpu_ids->data, 0, (size_t)top_k * n_tokens * sizeof(int32_t));

        struct ggml_cgraph * gf_cpu = ggml_new_graph(ctx_cpu);
        ggml_build_forward_expand(gf_cpu, cpu_down);
        ggml_backend_graph_compute(cpu_backend, gf_cpu);

        ggml_backend_tensor_set(bundle->down_node, cpu_down->data, 0, (size_t)d_model * top_k * n_tokens * sizeof(float));
        ggml_backend_buffer_free(buf_cpu);
        ggml_free(ctx_cpu);

        if (stats) {
            stats->hetero_layers++;
            stats->hetero_full_miss_layers++;
            stats->hetero_cpu_routes += top_k * n_tokens;
            stats->hetero_d2h_activation_bytes += (size_t)d_model * n_tokens * sizeof(float);
            stats->hetero_h2d_result_bytes += (size_t)d_model * top_k * n_tokens * sizeof(float);
        }
        return GGML_STATUS_SUCCESS;
    }

    // Lazy initialization of persistent scratch if needed
    if (scratch == nullptr || !scratch->initialized) {
        fprintf(stderr, "[ERROR] hetero_execute_serial: uninitialized scratch %p\n", (void*)scratch);
        return GGML_STATUS_FAILED;
    }

    // 1. Partition routes into hit_routes and miss_routes
    std::vector<ggml_cache_route_bundle> hit_routes(top_k);
    std::vector<ggml_cache_route_bundle> miss_routes(top_k);
    int32_t n_hits = 0;
    int32_t n_misses = 0;

    ggml_backend_expert_cache_partition_bundle_routes(
        cache,
        bundle->layer,
        ids_data,
        top_k,
        1,
        hit_routes.data(),
        &n_hits,
        miss_routes.data(),
        &n_misses);

    // Reserve slots for in-flight execution
    ggml_backend_expert_cache_reserve_bundle_slots(cache, bundle->layer, hit_routes.data(), n_hits);

    const char * debug_env = getenv("GGML_EXPERT_CACHE_DEBUG_HETERO");
    if (debug_env && atoi(debug_env) > 0) {
        fprintf(stderr, "[DEBUG_HETERO] layer=%d hits=%d misses=%d gpu_routes=%d cpu_routes=%d\n",
            bundle->layer, n_hits, n_misses, n_hits, n_misses);
    }

    const size_t in_bytes = (size_t)d_model * sizeof(float);
    const size_t out_bytes = (size_t)d_model * (size_t)top_k * sizeof(float);

    struct ggml_expert_bundle_weights bw = {};
    if (!ggml_backend_expert_cache_get_bundle_weights(cache, bundle->layer, &bw)) {
        fprintf(stderr, "[ERROR] hetero_execute_serial: failed to get bundle weights for layer=%d\n", bundle->layer);
        ggml_backend_expert_cache_release_bundle_slots(cache, bundle->layer, hit_routes.data(), n_hits, nullptr);
        return GGML_STATUS_FAILED;
    }

    struct ggml_tensor * slot_gate = bw.gate ? ggml_backend_expert_cache_get_slot_tensor(cache, bw.gate) : nullptr;
    struct ggml_tensor * slot_up   = bw.up   ? ggml_backend_expert_cache_get_slot_tensor(cache, bw.up)   : nullptr;
    struct ggml_tensor * slot_gu   = bw.gate_up ? ggml_backend_expert_cache_get_slot_tensor(cache, bw.gate_up) : nullptr;
    struct ggml_tensor * slot_down = bw.down ? ggml_backend_expert_cache_get_slot_tensor(cache, bw.down) : nullptr;

    // Case 1: Full Hit (all top_k routes resident on GPU)
    if (n_misses == 0) {
        if (!slot_down || (!slot_gu && (!slot_gate || !slot_up))) {
            fprintf(stderr, "[ERROR] hetero_execute_serial: full hit missing slot tensors layer=%d\n", bundle->layer);
            ggml_backend_expert_cache_release_bundle_slots(cache, bundle->layer, hit_routes.data(), n_hits, nullptr);
            return GGML_STATUS_FAILED;
        }

        std::vector<int32_t> hit_gate_slots(n_hits);
        std::vector<int32_t> hit_up_slots(n_hits);
        std::vector<int32_t> hit_down_slots(n_hits);
        for (int32_t i = 0; i < n_hits; i++) {
            hit_gate_slots[i] = bundle->is_fused ? hit_routes[i].gate_up_slot : hit_routes[i].gate_slot;
            hit_up_slots[i]   = bundle->is_fused ? hit_routes[i].gate_up_slot : hit_routes[i].up_slot;
            hit_down_slots[i] = hit_routes[i].down_slot;
        }

        struct ggml_init_params params_gpu = {
            /*.mem_size   =*/ 16 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * ctx_gpu = ggml_init(params_gpu);

        struct ggml_tensor * inp_x        = ggml_new_tensor_3d(ctx_gpu, GGML_TYPE_F32, d_model, 1, 1);
        struct ggml_tensor * gpu_gate_ids = ggml_new_tensor_2d(ctx_gpu, GGML_TYPE_I32, n_hits, 1);
        struct ggml_tensor * gpu_up_ids   = bundle->is_fused ? gpu_gate_ids : ggml_new_tensor_2d(ctx_gpu, GGML_TYPE_I32, n_hits, 1);
        struct ggml_tensor * gpu_down_ids = ggml_new_tensor_2d(ctx_gpu, GGML_TYPE_I32, n_hits, 1);
        struct ggml_tensor * gpu_gate     = nullptr;
        struct ggml_tensor * gpu_up       = nullptr;

        if (bundle->is_fused) {
            struct ggml_tensor * gpu_gu = ggml_mul_mat_id(ctx_gpu, slot_gu, inp_x, gpu_gate_ids);
            gpu_gate = ggml_view_3d(ctx_gpu, gpu_gu, d_ff, n_hits, 1, gpu_gu->nb[1], gpu_gu->nb[2], 0);
            gpu_up   = ggml_view_3d(ctx_gpu, gpu_gu, d_ff, n_hits, 1, gpu_gu->nb[1], gpu_gu->nb[2], d_ff * gpu_gu->nb[0]);
        } else {
            gpu_gate = ggml_mul_mat_id(ctx_gpu, slot_gate, inp_x, gpu_gate_ids);
            gpu_up   = ggml_mul_mat_id(ctx_gpu, slot_up,   inp_x, gpu_up_ids);
        }

        struct ggml_tensor * gpu_act  = ggml_swiglu_split(ctx_gpu, gpu_gate, gpu_up);
        struct ggml_tensor * gpu_down = ggml_mul_mat_id(ctx_gpu, slot_down, gpu_act, gpu_down_ids);

        struct ggml_cgraph * gf_gpu = ggml_new_graph(ctx_gpu);
        ggml_build_forward_expand(gf_gpu, gpu_down);

        ggml_backend_buffer_t buf_gpu_exec = ggml_backend_alloc_ctx_tensors(ctx_gpu, gpu_backend);
        ggml_backend_tensor_copy(input_x, inp_x);
        ggml_backend_tensor_set(gpu_gate_ids, hit_gate_slots.data(), 0, n_hits * sizeof(int32_t));
        if (!bundle->is_fused) {
            ggml_backend_tensor_set(gpu_up_ids, hit_up_slots.data(), 0, n_hits * sizeof(int32_t));
        }
        ggml_backend_tensor_set(gpu_down_ids, hit_down_slots.data(), 0, n_hits * sizeof(int32_t));

        ggml_backend_graph_compute(gpu_backend, gf_gpu);
        ggml_backend_synchronize(gpu_backend);

        // Copy directly to down_node on GPU
        ggml_backend_tensor_copy(gpu_down, down_node);

        ggml_backend_buffer_free(buf_gpu_exec);
        ggml_free(ctx_gpu);
        ggml_backend_expert_cache_release_bundle_slots(cache, bundle->layer, hit_routes.data(), n_hits, nullptr);

        if (stats) {
            stats->hetero_layers++;
            stats->hetero_full_hit_layers++;
            stats->hetero_gpu_routes += n_hits;
            if (n_hits <= 8) stats->hetero_hit_histogram[n_hits]++;
        }

        return GGML_STATUS_SUCCESS;
    }

    // Case 2: Full Miss (all top_k routes on CPU)
    if (n_hits == 0) {
        // Single D2H of layer input x
        ggml_backend_tensor_get(input_x, scratch->host_x_tensor->data, 0, in_bytes);

        // Execute full bundle on CPU using host weights
        struct ggml_init_params params_cpu = {
            /*.mem_size   =*/ 16 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        struct ggml_context * ctx_cpu = ggml_init(params_cpu);

        struct ggml_tensor * cpu_x = ggml_new_tensor_3d(ctx_cpu, GGML_TYPE_F32, d_model, 1, 1);
        memcpy(cpu_x->data, scratch->host_x_tensor->data, in_bytes);

        struct ggml_tensor * cpu_ids = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_I32, top_k, 1);
        memcpy(cpu_ids->data, ids_data, top_k * sizeof(int32_t));

        struct ggml_tensor * cpu_gate = nullptr;
        struct ggml_tensor * cpu_up   = nullptr;

        if (bundle->is_fused) {
            struct ggml_tensor * cpu_gu = ggml_mul_mat_id(ctx_cpu, bw.gate_up, cpu_x, cpu_ids);
            cpu_gate = ggml_view_3d(ctx_cpu, cpu_gu, d_ff, top_k, 1, cpu_gu->nb[1], cpu_gu->nb[2], 0);
            cpu_up   = ggml_view_3d(ctx_cpu, cpu_gu, d_ff, top_k, 1, cpu_gu->nb[1], cpu_gu->nb[2], d_ff * cpu_gu->nb[0]);
        } else {
            cpu_gate = ggml_mul_mat_id(ctx_cpu, bw.gate, cpu_x, cpu_ids);
            cpu_up   = ggml_mul_mat_id(ctx_cpu, bw.up,   cpu_x, cpu_ids);
        }

        struct ggml_tensor * cpu_silu = ggml_silu(ctx_cpu, cpu_gate);
        struct ggml_tensor * cpu_act  = ggml_mul(ctx_cpu, cpu_silu, cpu_up);
        struct ggml_tensor * cpu_down = ggml_mul_mat_id(ctx_cpu, bw.down, cpu_act, cpu_ids);

        struct ggml_cgraph * gf_cpu = ggml_new_graph(ctx_cpu);
        ggml_build_forward_expand(gf_cpu, cpu_down);
        ggml_backend_graph_compute(cpu_backend, gf_cpu);

        // Upload computed down output directly to down_node on GPU (safe at this scheduled lifetime)
        ggml_backend_tensor_set(down_node, cpu_down->data, 0, out_bytes);

        ggml_free(ctx_cpu);
        ggml_backend_expert_cache_release_bundle_slots(cache, bundle->layer, hit_routes.data(), n_hits, nullptr);

        if (stats) {
            stats->hetero_layers++;
            stats->hetero_full_miss_layers++;
            stats->hetero_cpu_routes += n_misses;
            stats->hetero_d2h_activation_bytes += in_bytes;
            stats->hetero_h2d_result_bytes += out_bytes;
            stats->hetero_hit_histogram[0]++;
        }

        return GGML_STATUS_SUCCESS;
    }

    // Case 3: Mixed Partial Hit (0 < n_hits < top_k) - Strictly Serial Execution
    // -------------------------------------------------------------------------
    // Step 3a: Execute GPU Hit Routes
    // -------------------------------------------------------------------------
    if (!slot_down || (!slot_gu && (!slot_gate || !slot_up))) {
        fprintf(stderr, "[ERROR] hetero_execute_serial: mixed hit missing slot tensors layer=%d\n", bundle->layer);
        ggml_backend_expert_cache_release_bundle_slots(cache, bundle->layer, hit_routes.data(), n_hits, nullptr);
        return GGML_STATUS_FAILED;
    }

    std::vector<int32_t> hit_gate_slots(n_hits);
    std::vector<int32_t> hit_up_slots(n_hits);
    std::vector<int32_t> hit_down_slots(n_hits);
    std::vector<int32_t> hit_route_indices(n_hits);
    for (int32_t i = 0; i < n_hits; i++) {
        hit_gate_slots[i]    = bundle->is_fused ? hit_routes[i].gate_up_slot : hit_routes[i].gate_slot;
        hit_up_slots[i]      = bundle->is_fused ? hit_routes[i].gate_up_slot : hit_routes[i].up_slot;
        hit_down_slots[i]    = hit_routes[i].down_slot;
        hit_route_indices[i] = hit_routes[i].route;
    }

    std::vector<int32_t> miss_experts(n_misses);
    std::vector<int32_t> miss_route_indices(n_misses);
    for (int32_t i = 0; i < n_misses; i++) {
        miss_experts[i]       = miss_routes[i].expert;
        miss_route_indices[i] = miss_routes[i].route;
    }

    // Build GPU sub-graph for exactly n_hits routes
    struct ggml_init_params params_gpu = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx_gpu = ggml_init(params_gpu);

    struct ggml_tensor * inp_x        = ggml_new_tensor_3d(ctx_gpu, GGML_TYPE_F32, d_model, 1, 1);
    struct ggml_tensor * gpu_gate_ids = ggml_new_tensor_2d(ctx_gpu, GGML_TYPE_I32, n_hits, 1);
    struct ggml_tensor * gpu_up_ids   = bundle->is_fused ? gpu_gate_ids : ggml_new_tensor_2d(ctx_gpu, GGML_TYPE_I32, n_hits, 1);
    struct ggml_tensor * gpu_down_ids = ggml_new_tensor_2d(ctx_gpu, GGML_TYPE_I32, n_hits, 1);
    struct ggml_tensor * gpu_gate     = nullptr;
    struct ggml_tensor * gpu_up       = nullptr;

    if (bundle->is_fused) {
        struct ggml_tensor * gpu_gu = ggml_mul_mat_id(ctx_gpu, slot_gu, inp_x, gpu_gate_ids);
        gpu_gate = ggml_view_3d(ctx_gpu, gpu_gu, d_ff, n_hits, 1, gpu_gu->nb[1], gpu_gu->nb[2], 0);
        gpu_up   = ggml_view_3d(ctx_gpu, gpu_gu, d_ff, n_hits, 1, gpu_gu->nb[1], gpu_gu->nb[2], d_ff * gpu_gu->nb[0]);
    } else {
        gpu_gate = ggml_mul_mat_id(ctx_gpu, slot_gate, inp_x, gpu_gate_ids);
        gpu_up   = ggml_mul_mat_id(ctx_gpu, slot_up,   inp_x, gpu_up_ids);
    }

    struct ggml_tensor * gpu_act  = ggml_swiglu_split(ctx_gpu, gpu_gate, gpu_up);
    struct ggml_tensor * gpu_down = ggml_mul_mat_id(ctx_gpu, slot_down, gpu_act, gpu_down_ids);

    struct ggml_cgraph * gf_gpu = ggml_new_graph(ctx_gpu);
    ggml_build_forward_expand(gf_gpu, gpu_down);

    ggml_backend_buffer_t buf_gpu_exec = ggml_backend_alloc_ctx_tensors(ctx_gpu, gpu_backend);
    ggml_backend_tensor_copy(input_x, inp_x);
    ggml_backend_tensor_set(gpu_gate_ids, hit_gate_slots.data(), 0, n_hits * sizeof(int32_t));
    if (!bundle->is_fused) {
        ggml_backend_tensor_set(gpu_up_ids, hit_up_slots.data(), 0, n_hits * sizeof(int32_t));
    }
    ggml_backend_tensor_set(gpu_down_ids, hit_down_slots.data(), 0, n_hits * sizeof(int32_t));

    ggml_backend_graph_compute_async(gpu_backend, gf_gpu);

    // -------------------------------------------------------------------------
    // Step 3b: Execute CPU Miss Routes (Explicit 2D Slices)
    // -------------------------------------------------------------------------
    // 1 D2H of layer input x
    ggml_backend_tensor_get(input_x, scratch->host_x_tensor->data, 0, in_bytes);

    struct ggml_init_params params_cpu = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx_cpu = ggml_init(params_cpu);

    struct ggml_tensor * cpu_x = ggml_new_tensor_3d(ctx_cpu, GGML_TYPE_F32, d_model, 1, 1);
    memcpy(cpu_x->data, scratch->host_x_tensor->data, in_bytes);

    struct ggml_tensor * cpu_ids = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_I32, n_misses, 1);
    memcpy(cpu_ids->data, miss_experts.data(), n_misses * sizeof(int32_t));

    struct ggml_tensor * cpu_gate = nullptr;
    struct ggml_tensor * cpu_up   = nullptr;

    if (bundle->is_fused) {
        struct ggml_tensor * cpu_gu = ggml_mul_mat_id(ctx_cpu, bw.gate_up, cpu_x, cpu_ids);
        cpu_gate = ggml_view_3d(ctx_cpu, cpu_gu, d_ff, n_misses, 1, cpu_gu->nb[1], cpu_gu->nb[2], 0);
        cpu_up   = ggml_view_3d(ctx_cpu, cpu_gu, d_ff, n_misses, 1, cpu_gu->nb[1], cpu_gu->nb[2], d_ff * cpu_gu->nb[0]);
    } else {
        cpu_gate = ggml_mul_mat_id(ctx_cpu, bw.gate, cpu_x, cpu_ids);
        cpu_up   = ggml_mul_mat_id(ctx_cpu, bw.up,   cpu_x, cpu_ids);
    }

    struct ggml_tensor * cpu_silu = ggml_silu(ctx_cpu, cpu_gate);
    struct ggml_tensor * cpu_act  = ggml_mul(ctx_cpu, cpu_silu, cpu_up);
    struct ggml_tensor * cpu_down = ggml_mul_mat_id(ctx_cpu, bw.down, cpu_act, cpu_ids);

    struct ggml_cgraph * gf_cpu = ggml_new_graph(ctx_cpu);
    ggml_build_forward_expand(gf_cpu, cpu_down);
    ggml_backend_graph_compute(cpu_backend, gf_cpu);

    // Copy CPU miss results into scratch host_down_tensor
    memcpy(scratch->host_down_tensor->data, cpu_down->data, (size_t)n_misses * d_model * sizeof(float));

#if defined(GGML_USE_CUDA)
    ggml_backend_tensor_set(scratch->cpu_upload_tensor, scratch->host_down_tensor->data, 0, (size_t)n_misses * d_model * sizeof(float));
    ggml_backend_tensor_set(scratch->gpu_hit_indices, hit_route_indices.data(), 0, n_hits * sizeof(int32_t));
    ggml_backend_tensor_set(scratch->gpu_miss_indices, miss_route_indices.data(), 0, n_misses * sizeof(int32_t));
#endif

    // -------------------------------------------------------------------------
    // Step 3d: Scatter Merge into down_node
    // -------------------------------------------------------------------------
#if defined(GGML_USE_CUDA)
    if (down_node->buffer != nullptr && !ggml_backend_buffer_is_host(down_node->buffer)) {
        ggml_cuda_moe_scatter_tg1(
            (uint8_t *)down_node->data,
            (const float *)gpu_down->data,
            (const float *)scratch->cpu_upload_tensor->data,
            (const int32_t *)scratch->gpu_hit_indices->data,
            (const int32_t *)scratch->gpu_miss_indices->data,
            n_hits,
            n_misses,
            (int)d_model,
            down_node->nb[1],
            0);
        ggml_backend_synchronize(gpu_backend);
    } else
#endif
    {
        std::vector<float> gpu_down_host((size_t)n_hits * d_model);
        ggml_backend_synchronize(gpu_backend);
        ggml_backend_tensor_get(gpu_down, gpu_down_host.data(), 0, (size_t)n_hits * d_model * sizeof(float));
        for (int32_t k = 0; k < n_hits; k++) {
            const int32_t r = hit_route_indices[k];
            ggml_backend_tensor_set(down_node, gpu_down_host.data() + (size_t)k * d_model, (size_t)r * down_node->nb[1], d_model * sizeof(float));
        }
        for (int32_t m = 0; m < n_misses; m++) {
            const int32_t r = miss_route_indices[m];
            ggml_backend_tensor_set(
                down_node,
                (const float *) scratch->host_down_tensor->data + (size_t) m * d_model,
                (size_t) r * down_node->nb[1],
                d_model * sizeof(float));
        }
    }

    // Cleanup local execution graphs
    ggml_backend_buffer_free(buf_gpu_exec);
    ggml_free(ctx_gpu);
    ggml_free(ctx_cpu);

    // Release slot reservations
    ggml_backend_expert_cache_release_bundle_slots(cache, bundle->layer, hit_routes.data(), n_hits, nullptr);

    // Update telemetry
    if (stats) {
        stats->hetero_layers++;
        stats->hetero_partial_layers++;
        stats->hetero_gpu_routes += n_hits;
        stats->hetero_cpu_routes += n_misses;
        stats->hetero_d2h_activation_bytes += in_bytes;
        stats->hetero_h2d_result_bytes += (size_t)n_misses * d_model * sizeof(float);
        if (n_hits <= 8) stats->hetero_hit_histogram[n_hits]++;
    }

    return GGML_STATUS_SUCCESS;
}

bool ggml_moe_partial_route_snapshot_is_valid(
        const struct ggml_moe_partial_route_snapshot * snapshot,
        int32_t top_k) {
    if (snapshot == nullptr || top_k <= 1 || top_k > GGML_MOE_PARTIAL_MAX_ROUTES ||
        snapshot->n_hits <= 0 || snapshot->n_misses <= 0 ||
        snapshot->n_hits > GGML_MOE_PARTIAL_MAX_ROUTES ||
        snapshot->n_misses > GGML_MOE_PARTIAL_MAX_ROUTES ||
        snapshot->n_hits + snapshot->n_misses != top_k) {
        return false;
    }

    bool written[GGML_MOE_PARTIAL_MAX_ROUTES] = {};
    const struct ggml_cache_route_bundle * route_groups[] = {
        snapshot->hits,
        snapshot->misses,
    };
    const int32_t route_counts[] = {
        snapshot->n_hits,
        snapshot->n_misses,
    };
    for (int32_t group = 0; group < 2; ++group) {
        for (int32_t i = 0; i < route_counts[group]; ++i) {
            const struct ggml_cache_route_bundle & route = route_groups[group][i];
            if (route.route < 0 || route.route >= top_k || written[route.route] ||
                (group == 0 && !route.bundle_resident)) {
                return false;
            }
            written[route.route] = true;
        }
    }

    for (int32_t route = 0; route < top_k; ++route) {
        if (!written[route]) {
            return false;
        }
    }
    return true;
}

ggml_moe_partial_executor_t ggml_moe_partial_executor_new(
        ggml_backend_t gpu_backend,
        ggml_backend_t cpu_backend,
        const struct ggml_expert_bundle_weights * template_weights,
        int64_t d_model,
        int64_t d_ff,
        int32_t top_k,
        bool is_fused) {
    if (gpu_backend == nullptr || cpu_backend == nullptr || template_weights == nullptr ||
        d_model <= 0 || d_ff <= 0 || top_k != GGML_MOE_PARTIAL_MAX_ROUTES ||
        template_weights->is_fused != is_fused || template_weights->down == nullptr ||
        template_weights->down->ne[0] != d_ff || template_weights->down->ne[1] != d_model ||
        (is_fused ?
            (template_weights->gate_up == nullptr || template_weights->gate_up->ne[0] != d_model ||
                template_weights->gate_up->ne[1] != 2 * d_ff) :
            (template_weights->gate == nullptr || template_weights->up == nullptr ||
                template_weights->gate->ne[0] != d_model || template_weights->gate->ne[1] != d_ff ||
                template_weights->up->ne[0] != d_model || template_weights->up->ne[1] != d_ff))) {
        return nullptr;
    }

    auto * executor = new ggml_moe_partial_executor {
        gpu_backend,
        cpu_backend,
        *template_weights,
        d_model,
        d_ff,
        top_k,
        is_fused,
    };
    ggml_backend_dev_t gpu_device = ggml_backend_get_device(gpu_backend);
    ggml_backend_reg_t gpu_reg = gpu_device ? ggml_backend_dev_backend_reg(gpu_device) : nullptr;
    executor->event_elapsed_us = gpu_reg ?
        (ggml_backend_event_elapsed_us_t) ggml_backend_reg_get_proc_address(gpu_reg, "ggml_backend_event_elapsed_us") :
        nullptr;
    if (executor->event_elapsed_us == nullptr ||
        !ggml_moe_partial_executor_init_merge_buffers(executor) ||
        !ggml_moe_partial_executor_init_exchange(executor) ||
        !ggml_moe_partial_executor_init_events(executor)) {
        ggml_moe_partial_executor_free(executor);
        return nullptr;
    }
    for (int32_t i = 0; i < GGML_MOE_PARTIAL_VARIANT_COUNT; ++i) {
        if (!ggml_moe_partial_executor_init_gpu_variant(executor, &executor->gpu[i], i + 1) ||
            !ggml_moe_partial_executor_init_cpu_variant(executor, &executor->cpu[i], i + 1)) {
            ggml_moe_partial_executor_free(executor);
            return nullptr;
        }
    }
    return executor;
}

void ggml_moe_partial_executor_free(ggml_moe_partial_executor_t executor) {
    if (executor == nullptr) {
        return;
    }
    for (ggml_backend_event_t event : executor->events) {
        ggml_backend_event_free(event);
    }
    for (struct ggml_moe_partial_gpu_variant & variant : executor->gpu) {
        ggml_moe_partial_gpu_variant_free(&variant);
    }
    for (struct ggml_moe_partial_cpu_variant & variant : executor->cpu) {
        ggml_moe_partial_cpu_variant_free(&variant);
    }
    if (executor->merge_buffer != nullptr) {
        ggml_backend_buffer_free(executor->merge_buffer);
    }
    if (executor->merge_ctx != nullptr) {
        ggml_free(executor->merge_ctx);
    }
    if (executor->exchange_buffer != nullptr) {
        ggml_backend_buffer_free(executor->exchange_buffer);
    }
    if (executor->exchange_ctx != nullptr) {
        ggml_free(executor->exchange_ctx);
    }
    delete executor;
}

#ifdef GGML_TEST
bool ggml_moe_partial_executor_get_test_state(
        ggml_moe_partial_executor_t executor,
        struct ggml_moe_partial_executor_test_state * state) {
    if (executor == nullptr || state == nullptr) {
        return false;
    }
    *state = {};
    for (int32_t i = 0; i < GGML_MOE_PARTIAL_VARIANT_COUNT; ++i) {
        const struct ggml_moe_partial_gpu_variant & gpu = executor->gpu[i];
        const struct ggml_moe_partial_cpu_variant & cpu = executor->cpu[i];
        state->gpu_variants[i] = gpu.ctx != nullptr && gpu.buffer != nullptr &&
            gpu.input != nullptr && gpu.gate_ids != nullptr && gpu.up_ids != nullptr && gpu.down_ids != nullptr &&
            gpu.output != nullptr && gpu.graph != nullptr;
        state->cpu_variants[i] = cpu.ctx != nullptr && cpu.buffer != nullptr &&
            cpu.input != nullptr && cpu.ids != nullptr && cpu.output != nullptr && cpu.graph != nullptr;
        state->gpu_outputs[i] = gpu.output;
        state->cpu_outputs[i] = cpu.output;

    }
    state->has_merge_buffer = executor->merge != nullptr && executor->merge_buffer != nullptr;
    state->has_cpu_upload_buffer = executor->cpu_upload != nullptr && executor->merge_buffer != nullptr;
    state->exchange_buffer = executor->exchange_buffer;
    for (int32_t i = 0; i < GGML_MOE_PARTIAL_EVENT_COUNT; ++i) {
        state->events[i] = executor->events[i] != nullptr;
    }
    return true;
}
#endif

static void ggml_moe_partial_executor_bind_gpu(
        ggml_moe_partial_executor_t executor,
        struct ggml_moe_partial_gpu_variant * variant,
        struct ggml_tensor * const slot_tensors[3]) {
    ggml_moe_route_ready_sidecar_copy_slot(&variant->gate_slot, slot_tensors[0]);
    if (!executor->is_fused) {
        ggml_moe_route_ready_sidecar_copy_slot(&variant->up_slot, slot_tensors[1]);
    }
    ggml_moe_route_ready_sidecar_copy_slot(&variant->down_slot, slot_tensors[2]);
}

static void ggml_moe_partial_executor_unbind_gpu(
        ggml_moe_partial_executor_t executor,
        struct ggml_moe_partial_gpu_variant * variant) {
    ggml_moe_route_ready_sidecar_unbind_slot(&variant->gate_slot);
    if (!executor->is_fused) {
        ggml_moe_route_ready_sidecar_unbind_slot(&variant->up_slot);
    }
    ggml_moe_route_ready_sidecar_unbind_slot(&variant->down_slot);
}

static void ggml_moe_partial_executor_bind_cpu(
        ggml_moe_partial_executor_t executor,
        struct ggml_moe_partial_cpu_variant * variant,
        const struct ggml_expert_bundle_weights * weights) {
    ggml_moe_route_ready_sidecar_copy_slot(&variant->gate, executor->is_fused ? weights->gate_up : weights->gate);
    if (!executor->is_fused) {
        ggml_moe_route_ready_sidecar_copy_slot(&variant->up, weights->up);
    }
    ggml_moe_route_ready_sidecar_copy_slot(&variant->down, weights->down);
}

static void ggml_moe_partial_executor_unbind_cpu(
        ggml_moe_partial_executor_t executor,
        struct ggml_moe_partial_cpu_variant * variant) {
    ggml_moe_route_ready_sidecar_unbind_slot(&variant->gate);
    if (!executor->is_fused) {
        ggml_moe_route_ready_sidecar_unbind_slot(&variant->up);
    }
    ggml_moe_route_ready_sidecar_unbind_slot(&variant->down);
}

enum ggml_moe_partial_executor_result ggml_moe_partial_executor_execute(
        ggml_moe_partial_executor_t executor,
        const struct ggml_moe_bundle_plan * bundle,
        ggml_backend_expert_cache_t cache,
        const struct ggml_moe_partial_route_snapshot * snapshot,
        const struct ggml_moe_partial_activation * activation,
        struct ggml_backend_expert_cache_stats * stats) {
    if (executor == nullptr || bundle == nullptr || cache == nullptr || activation == nullptr ||
        !ggml_moe_partial_route_snapshot_is_valid(snapshot, executor ? executor->top_k : 0) ||
        bundle->route_ids == nullptr || bundle->is_fused != executor->is_fused ||
        bundle->layer_input->type != GGML_TYPE_F32 || bundle->down_node->type != GGML_TYPE_F32 ||
        bundle->layer_input->ne[0] != executor->d_model || bundle->layer_input->ne[1] != 1 ||
        bundle->layer_input->ne[2] != 1 || bundle->layer_input->nb[0] != sizeof(float) ||
        bundle->layer_input->nb[1] != (size_t) executor->d_model * sizeof(float) ||
        bundle->layer_input->nb[2] != (size_t) executor->d_model * sizeof(float) ||
        bundle->down_node->ne[0] != executor->d_model || bundle->down_node->ne[1] != executor->top_k ||
        bundle->down_node->ne[2] != 1 || bundle->down_node->nb[0] != sizeof(float) ||
        bundle->down_node->nb[1] != (size_t) executor->d_model * sizeof(float) ||
        bundle->down_node->nb[2] != (size_t) executor->d_model * executor->top_k * sizeof(float) ||
        bundle->route_ids->type != GGML_TYPE_I32 || bundle->route_ids->ne[0] != executor->top_k ||
        bundle->route_ids->ne[1] != 1 || bundle->route_ids->ne[2] != 1 ||
        bundle->route_ids->nb[0] != sizeof(int32_t) ||
        bundle->route_ids->nb[1] != (size_t) executor->top_k * sizeof(int32_t) ||
        ggml_nbytes(bundle->layer_input) != activation->nbytes) {
        return GGML_MOE_PARTIAL_EXECUTOR_NOT_ADMITTED;
    }

    const struct ggml_tensor * gate = bundle->is_fused ? bundle->gate_up_node : bundle->gate_node;
    const struct ggml_tensor * up = bundle->is_fused ? nullptr : bundle->up_node;
    if (gate == nullptr || gate->type != GGML_TYPE_F32 || gate->ne[0] != (bundle->is_fused ? 2 * executor->d_ff : executor->d_ff) ||
        gate->ne[1] != executor->top_k || gate->ne[2] != 1 ||
        (!bundle->is_fused && (up == nullptr || up->type != GGML_TYPE_F32 ||
            up->ne[0] != executor->d_ff || up->ne[1] != executor->top_k || up->ne[2] != 1))) {
        return GGML_MOE_PARTIAL_EXECUTOR_NOT_ADMITTED;
    }

    const struct ggml_expert_bundle_weights * template_weights = &executor->template_weights;
    struct ggml_expert_bundle_weights weights = {};
    if (!ggml_backend_expert_cache_get_bundle_weights(cache, bundle->layer, &weights) ||
        weights.is_fused != executor->is_fused || weights.down == nullptr ||
        weights.down->type != template_weights->down->type ||
        weights.down->ne[0] != executor->d_ff || weights.down->ne[1] != executor->d_model ||
        weights.down->buffer == nullptr || !ggml_backend_buffer_is_host(weights.down->buffer) ||
        (executor->is_fused ?
            (weights.gate_up == nullptr || weights.gate_up->type != template_weights->gate_up->type ||
                weights.gate_up->ne[0] != executor->d_model || weights.gate_up->ne[1] != 2 * executor->d_ff ||
                weights.gate_up->buffer == nullptr || !ggml_backend_buffer_is_host(weights.gate_up->buffer)) :
            (weights.gate == nullptr || weights.up == nullptr ||
                weights.gate->type != template_weights->gate->type ||
                weights.up->type != template_weights->up->type ||
                weights.gate->ne[0] != executor->d_model || weights.gate->ne[1] != executor->d_ff ||
                weights.up->ne[0] != executor->d_model || weights.up->ne[1] != executor->d_ff ||
                weights.gate->buffer == nullptr || !ggml_backend_buffer_is_host(weights.gate->buffer) ||
                weights.up->buffer == nullptr || !ggml_backend_buffer_is_host(weights.up->buffer)))) {
        return GGML_MOE_PARTIAL_EXECUTOR_NOT_ADMITTED;
    }

    struct ggml_tensor * slot_tensors[] = {
        ggml_backend_expert_cache_find_slot_tensor(cache, executor->is_fused ? weights.gate_up : weights.gate),
        executor->is_fused ? nullptr : ggml_backend_expert_cache_find_slot_tensor(cache, weights.up),
        ggml_backend_expert_cache_find_slot_tensor(cache, weights.down),
    };
    if (slot_tensors[0] == nullptr || slot_tensors[2] == nullptr ||
        (!executor->is_fused && slot_tensors[1] == nullptr)) {
        return GGML_MOE_PARTIAL_EXECUTOR_NOT_ADMITTED;
    }

    int32_t hit_gate_slots[GGML_MOE_PARTIAL_MAX_ROUTES] = {};
    int32_t hit_up_slots[GGML_MOE_PARTIAL_MAX_ROUTES] = {};
    int32_t hit_down_slots[GGML_MOE_PARTIAL_MAX_ROUTES] = {};
    int32_t miss_expert_ids[GGML_MOE_PARTIAL_MAX_ROUTES] = {};
    for (int32_t i = 0; i < snapshot->n_hits; ++i) {
        const struct ggml_cache_route_bundle & route = snapshot->hits[i];
        hit_gate_slots[i] = executor->is_fused ? route.gate_up_slot : route.gate_slot;
        hit_up_slots[i] = executor->is_fused ? route.gate_up_slot : route.up_slot;
        hit_down_slots[i] = route.down_slot;
        const int32_t slots[] = { hit_gate_slots[i], hit_up_slots[i], hit_down_slots[i] };
        for (int32_t projection = 0; projection < 3; ++projection) {
            if (slot_tensors[projection] != nullptr &&
                (slots[projection] < 0 || slots[projection] >= slot_tensors[projection]->ne[2])) {
                return GGML_MOE_PARTIAL_EXECUTOR_NOT_ADMITTED;
            }
        }
    }
    for (int32_t i = 0; i < snapshot->n_misses; ++i) {
        miss_expert_ids[i] = snapshot->misses[i].expert;
        if (miss_expert_ids[i] < 0 || miss_expert_ids[i] >= weights.down->ne[2]) {
            return GGML_MOE_PARTIAL_EXECUTOR_NOT_ADMITTED;
        }
    }

    ggml_backend_buffer_t output_buffer = bundle->down_node->view_src ?
        bundle->down_node->view_src->buffer : bundle->down_node->buffer;
    ggml_backend_dev_t gpu_device = ggml_backend_get_device(executor->gpu_backend);
    ggml_backend_buffer_type_t host_buft = gpu_device ? ggml_backend_dev_host_buffer_type(gpu_device) : nullptr;
    if (output_buffer == nullptr || !ggml_backend_buffer_is_host(output_buffer) ||
        executor->exchange_buffer == nullptr || executor->exchange_activation == nullptr ||
        executor->exchange_hit_ids == nullptr || executor->exchange_miss_ids == nullptr ||
        executor->exchange_miss_output == nullptr ||
        host_buft == nullptr || ggml_backend_buffer_get_type(executor->exchange_buffer) != host_buft ||
        gpu_device == nullptr || gpu_device->iface.event_new == nullptr ||
        executor->gpu_backend->iface.set_tensor_async == nullptr ||
        executor->gpu_backend->iface.get_tensor_async == nullptr ||
        executor->gpu_backend->iface.cpy_tensor_async == nullptr) {
        return GGML_MOE_PARTIAL_EXECUTOR_NOT_ADMITTED;
    }

    struct ggml_moe_partial_gpu_variant * gpu = &executor->gpu[snapshot->n_hits - 1];
    struct ggml_moe_partial_cpu_variant * cpu = &executor->cpu[snapshot->n_misses - 1];
    if (executor->poisoned || gpu->graph == nullptr || cpu->graph == nullptr ||
        gpu->output == nullptr || cpu->output == nullptr) {
        return executor->poisoned ? GGML_MOE_PARTIAL_EXECUTOR_LAUNCH_FAILED : GGML_MOE_PARTIAL_EXECUTOR_NOT_ADMITTED;
    }

    const void * activation_host = activation->host_data;
    if (activation_host == nullptr && ggml_backend_buffer_is_host(bundle->layer_input->buffer)) {
        activation_host = bundle->layer_input->data;
    }
    if (activation_host == nullptr) {
        return GGML_MOE_PARTIAL_EXECUTOR_NOT_ADMITTED;
    }

    int32_t * hit_ids_base = (int32_t *) executor->exchange_hit_ids->data;
    memcpy(executor->exchange_activation->data, activation_host, activation->nbytes);
    memcpy(hit_ids_base, hit_gate_slots, (size_t) snapshot->n_hits * sizeof(int32_t));
    memcpy(hit_ids_base + GGML_MOE_PARTIAL_MAX_ROUTES, hit_up_slots, (size_t) snapshot->n_hits * sizeof(int32_t));
    memcpy(hit_ids_base + 2 * GGML_MOE_PARTIAL_MAX_ROUTES, hit_down_slots, (size_t) snapshot->n_hits * sizeof(int32_t));
    memcpy(executor->exchange_miss_ids->data, miss_expert_ids, (size_t) snapshot->n_misses * sizeof(int32_t));
    ggml_backend_tensor_set(cpu->input, executor->exchange_activation->data, 0, activation->nbytes);
    ggml_backend_tensor_set(cpu->ids, executor->exchange_miss_ids->data, 0,
        (size_t) snapshot->n_misses * sizeof(int32_t));

    ggml_backend_expert_cache_reserve_bundle_slots(cache, bundle->layer, snapshot->hits, snapshot->n_hits);
    ggml_moe_partial_executor_bind_gpu(executor, gpu, slot_tensors);
    ggml_moe_partial_executor_bind_cpu(executor, cpu, &weights);

    const int64_t t_start = ggml_time_us();
    ggml_backend_tensor_set_async(executor->gpu_backend, gpu->input, executor->exchange_activation->data, 0,
        activation->nbytes);
    ggml_backend_tensor_set_async(executor->gpu_backend, gpu->gate_ids, hit_ids_base, 0,
        (size_t) snapshot->n_hits * sizeof(int32_t));
    if (!executor->is_fused) {
        ggml_backend_tensor_set_async(executor->gpu_backend, gpu->up_ids,
            hit_ids_base + GGML_MOE_PARTIAL_MAX_ROUTES, 0, (size_t) snapshot->n_hits * sizeof(int32_t));
    }
    ggml_backend_tensor_set_async(executor->gpu_backend, gpu->down_ids,
        hit_ids_base + 2 * GGML_MOE_PARTIAL_MAX_ROUTES, 0, (size_t) snapshot->n_hits * sizeof(int32_t));
    const int64_t t_submit_done = ggml_time_us();
    ggml_backend_event_record(executor->events[GGML_MOE_PARTIAL_EVENT_GPU_GRAPH_START], executor->gpu_backend);
    if (ggml_backend_graph_compute_async(executor->gpu_backend, gpu->graph) != GGML_STATUS_SUCCESS) {
        ggml_moe_partial_executor_unbind_gpu(executor, gpu);
        ggml_moe_partial_executor_unbind_cpu(executor, cpu);
        ggml_backend_expert_cache_release_bundle_slots(cache, bundle->layer, snapshot->hits, snapshot->n_hits, nullptr);
        return GGML_MOE_PARTIAL_EXECUTOR_LAUNCH_FAILED;
    }
    ggml_backend_event_record(executor->events[GGML_MOE_PARTIAL_EVENT_GPU_GRAPH_END], executor->gpu_backend);

    const int64_t t_cpu_start = ggml_time_us();
    if (ggml_backend_graph_compute(executor->cpu_backend, cpu->graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_event_synchronize(executor->events[GGML_MOE_PARTIAL_EVENT_GPU_GRAPH_END]);
        ggml_moe_partial_executor_unbind_gpu(executor, gpu);
        ggml_moe_partial_executor_unbind_cpu(executor, cpu);
        ggml_backend_expert_cache_release_bundle_slots(cache, bundle->layer, snapshot->hits, snapshot->n_hits, nullptr);
        executor->poisoned = true;
        return GGML_MOE_PARTIAL_EXECUTOR_LAUNCH_FAILED;
    }
    const int64_t t_cpu_done = ggml_time_us();

    ggml_backend_tensor_get(cpu->output, executor->exchange_miss_output->data, 0,
        (size_t) snapshot->n_misses * executor->d_model * sizeof(float));
    ggml_backend_event_record(executor->events[GGML_MOE_PARTIAL_EVENT_CPU_UPLOAD_START], executor->gpu_backend);
    ggml_backend_tensor_set_async(executor->gpu_backend, executor->cpu_upload, executor->exchange_miss_output->data, 0,
        (size_t) snapshot->n_misses * executor->d_model * sizeof(float));
    ggml_backend_event_record(executor->events[GGML_MOE_PARTIAL_EVENT_CPU_UPLOAD_END], executor->gpu_backend);

    const int64_t t_join_gpu_start = ggml_time_us();
    ggml_backend_event_synchronize(executor->events[GGML_MOE_PARTIAL_EVENT_GPU_GRAPH_END]);
    const int64_t t_join_gpu_done = ggml_time_us();
    ggml_backend_event_synchronize(executor->events[GGML_MOE_PARTIAL_EVENT_CPU_UPLOAD_END]);
    const int64_t t_join_cpu_done = ggml_time_us();

    const int64_t t_scatter_start = ggml_time_us();
    ggml_backend_event_record(executor->events[GGML_MOE_PARTIAL_EVENT_SCATTER_START], executor->gpu_backend);
    for (int32_t i = 0; i < snapshot->n_hits; ++i) {
        struct ggml_tensor src_row = *gpu->output;
        struct ggml_tensor dst_row = *executor->merge;
        src_row.ne[1] = 1;
        dst_row.ne[1] = 1;
        src_row.nb[2] = src_row.nb[1];
        src_row.nb[3] = src_row.nb[2];
        dst_row.nb[2] = dst_row.nb[1];
        dst_row.nb[3] = dst_row.nb[2];
        src_row.data = (uint8_t *) src_row.data + (size_t) i * src_row.nb[1];
        dst_row.data = (uint8_t *) dst_row.data + (size_t) snapshot->hits[i].route * dst_row.nb[1];
        ggml_backend_tensor_copy_async(executor->gpu_backend, executor->gpu_backend, &src_row, &dst_row);
    }
    for (int32_t i = 0; i < snapshot->n_misses; ++i) {
        struct ggml_tensor src_row = *executor->cpu_upload;
        struct ggml_tensor dst_row = *executor->merge;
        src_row.ne[1] = 1;
        dst_row.ne[1] = 1;
        src_row.nb[2] = src_row.nb[1];
        src_row.nb[3] = src_row.nb[2];
        dst_row.nb[2] = dst_row.nb[1];
        dst_row.nb[3] = dst_row.nb[2];
        src_row.data = (uint8_t *) src_row.data + (size_t) i * src_row.nb[1];
        dst_row.data = (uint8_t *) dst_row.data + (size_t) snapshot->misses[i].route * dst_row.nb[1];
        ggml_backend_tensor_copy_async(executor->gpu_backend, executor->gpu_backend, &src_row, &dst_row);
    }
    ggml_backend_event_record(executor->events[GGML_MOE_PARTIAL_EVENT_SCATTER_END], executor->gpu_backend);
    const int64_t t_scatter_done = ggml_time_us();

    ggml_backend_tensor_get_async(executor->gpu_backend, executor->merge, bundle->down_node->data, 0,
        ggml_nbytes(bundle->down_node));
    ggml_backend_event_record(executor->events[GGML_MOE_PARTIAL_EVENT_FINAL_OUTPUT], executor->gpu_backend);
    ggml_backend_event_synchronize(executor->events[GGML_MOE_PARTIAL_EVENT_FINAL_OUTPUT]);
    const int64_t t_done = ggml_time_us();

    uint64_t gpu_hit_elapsed = 0;
    executor->event_elapsed_us(
        executor->events[GGML_MOE_PARTIAL_EVENT_GPU_GRAPH_START],
        executor->events[GGML_MOE_PARTIAL_EVENT_GPU_GRAPH_END],
        &gpu_hit_elapsed);
    ggml_moe_partial_executor_unbind_gpu(executor, gpu);
    ggml_moe_partial_executor_unbind_cpu(executor, cpu);
    ggml_backend_expert_cache_release_bundle_slots(cache, bundle->layer, snapshot->hits, snapshot->n_hits, nullptr);

    if (stats != nullptr) {
        stats->hetero_partial_exec_by_hits[snapshot->n_hits]++;
        stats->hetero_partial_gpu_routes_executed += snapshot->n_hits;
        stats->hetero_partial_cpu_routes_executed += snapshot->n_misses;
        stats->hetero_partial_gpu_hit_submit_us += (uint64_t) (t_submit_done - t_start);
        stats->hetero_partial_gpu_hit_elapsed_us += gpu_hit_elapsed;
        stats->hetero_partial_cpu_miss_compute_us += (uint64_t) (t_cpu_done - t_cpu_start);
        stats->hetero_partial_cpu_result_h2d_bytes += (size_t) snapshot->n_misses * executor->d_model * sizeof(float);
        stats->hetero_partial_join_wait_gpu_us += (uint64_t) (t_join_gpu_done - t_join_gpu_start);
        stats->hetero_partial_join_wait_cpu_us += (uint64_t) (t_join_cpu_done - t_join_gpu_done);
        stats->hetero_partial_scatter_us += (uint64_t) (t_scatter_done - t_scatter_start);
        stats->hetero_partial_total_us += (uint64_t) (t_done - t_start);
    }
    return GGML_MOE_PARTIAL_EXECUTOR_SUCCESS;
}

