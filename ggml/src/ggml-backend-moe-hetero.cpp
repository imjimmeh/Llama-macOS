#include "ggml-backend-moe-hetero.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"

#if defined(GGML_USE_CUDA)
#include "ggml-cuda/moe-hetero.cuh"
#endif

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

// Phase 2: Event-driven concurrent execution engine (staged for Milestone 6)
enum ggml_status ggml_backend_moe_hetero_execute_concurrent(
        ggml_backend_t gpu_backend,
        ggml_backend_t cpu_backend,
        const struct ggml_moe_bundle_plan * bundle,
        ggml_backend_expert_cache_t cache,
        const int32_t * ids_data,
        int32_t top_k,
        ggml_moe_hetero_scratch_t scratch,
        struct ggml_backend_expert_cache_stats * stats) {

    // Fall back to verified serial implementation during initial Phase 1 gating
    return ggml_backend_moe_hetero_execute_serial(gpu_backend, cpu_backend, bundle, cache, ids_data, top_k, scratch, stats);
}

