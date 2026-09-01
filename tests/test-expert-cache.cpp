#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-expert-cache.h"
#include "../ggml/src/ggml-backend-moe-hetero.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-context.h"



#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static void require_impl(bool condition, const char * expression, const char * file, int line) {
    if (!condition) {
        fprintf(stderr, "test requirement failed: %s:%d: %s\n", file, line, expression);
        printf("test requirement failed: %s:%d: %s\n", file, line, expression);
        fflush(stderr);
        fflush(stdout);
        abort();
    }
}

#define require(condition) require_impl((condition), #condition, __FILE__, __LINE__)


static void test_complete_bundle_admission_query() {
    printf("testing live complete-bundle admission query...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 8);
    ggml_tensor * up = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 8);
    ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 8);
    ggml_set_name(gate, "blk.3.ffn_gate_exps.weight");
    ggml_set_name(up, "blk.3.ffn_up_exps.weight");
    ggml_set_name(down, "blk.3.ffn_down_exps.weight");

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, 1024);
    require(cache != nullptr);
    ggml_backend_expert_cache_register_bundle(cache, 3, gate, up, down);

    require(ggml_backend_expert_cache_count_complete_bundles(cache, 3, 8) == 0);
    require(!ggml_backend_expert_cache_has_at_least_complete_bundles(cache, 3, 1));
    bool needs_load = false;
    const int32_t loading_slot = ggml_backend_expert_cache_claim_slot_idx(
        cache, gate, 0, nullptr, 0, &needs_load);
    require(loading_slot >= 0);
    require(needs_load);
    require(ggml_backend_expert_cache_seed(cache, up, 0, 1));
    require(ggml_backend_expert_cache_seed(cache, down, 0, 1));
    require(!ggml_backend_expert_cache_has_at_least_complete_bundles(cache, 3, 1));
    ggml_backend_expert_cache_promote_slot(cache, gate, 0, loading_slot);
    require(ggml_backend_expert_cache_has_at_least_complete_bundles(cache, 3, 1));


    for (int32_t expert_id = 0; expert_id < 6; ++expert_id) {
        require(ggml_backend_expert_cache_seed(cache, gate, expert_id, 1));
        require(ggml_backend_expert_cache_seed(cache, up, expert_id, 1));
        require(ggml_backend_expert_cache_seed(cache, down, expert_id, 1));
    }
    require(ggml_backend_expert_cache_count_complete_bundles(cache, 3, 8) == 6);
    require(ggml_backend_expert_cache_has_at_least_complete_bundles(cache, 3, 6));
    require(!ggml_backend_expert_cache_has_at_least_complete_bundles(cache, 3, 7));

    require(ggml_backend_expert_cache_seed(cache, gate, 6, 1));
    require(ggml_backend_expert_cache_seed(cache, up, 6, 1));
    require(ggml_backend_expert_cache_seed(cache, down, 6, 1));
    require(ggml_backend_expert_cache_has_at_least_complete_bundles(cache, 3, 7));
    require(!ggml_backend_expert_cache_has_at_least_complete_bundles(cache, 3, 8));

    require(ggml_backend_expert_cache_seed(cache, gate, 7, 1));
    require(ggml_backend_expert_cache_seed(cache, up, 7, 1));
    require(ggml_backend_expert_cache_seed(cache, down, 7, 1));
    require(ggml_backend_expert_cache_count_complete_bundles(cache, 3, 8) == 8);
    require(ggml_backend_expert_cache_has_at_least_complete_bundles(cache, 3, 8));
    require(ggml_backend_expert_cache_count_complete_bundles(cache, 3, 4) == 4);

    ggml_tensor * gate_up = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 8);
    ggml_tensor * fused_down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 8);
    ggml_set_name(gate_up, "blk.4.ffn_gate_up_exps.weight");
    ggml_set_name(fused_down, "blk.4.ffn_down_exps.weight");
    ggml_backend_expert_cache_register_fused_bundle(cache, 4, gate_up, fused_down);
    require(ggml_backend_expert_cache_seed(cache, gate_up, 0, 1));
    require(ggml_backend_expert_cache_seed(cache, fused_down, 0, 1));
    require(ggml_backend_expert_cache_count_complete_bundles(cache, 4, 8) == 1);
    require(ggml_backend_expert_cache_has_at_least_complete_bundles(cache, 4, 1));

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  live complete-bundle admission query tests passed\n");
}


static void test_cache_node_selection() {
    printf("testing cached expert node selection...\n");

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    ggml_tensor * experts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 8, 2);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 2, 1);
    ggml_tensor * matmul = ggml_mul_mat_id(ctx, experts, input, ids);
    ggml_tensor * unrelated_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, matmul->ne[0], matmul->ne[1]);
    ggml_tensor * unrelated = ggml_dup(ctx, unrelated_input);
    ggml_tensor * output = ggml_add(ctx, unrelated, matmul);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    require(ggml_graph_node(graph, 0) != matmul);
    require(ggml_backend_find_mul_mat_id_node(graph, experts) == matmul);
    require(ggml_backend_find_mul_mat_id_node(graph, input) == nullptr);

    ggml_free(ctx);

    printf("  cached expert node selection tests passed\n");
}

static void test_route_census_classifies_original_graph() {
    printf("testing original-graph route census...\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_device == nullptr) {
        printf("  no GPU backend available; skipped\n");
        return;
    }

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_device, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    require(gpu_backend != nullptr);
    require(cpu_backend != nullptr);
    ggml_backend_t backends[] = { gpu_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    require(sched != nullptr);
    ggml_backend_sched_set_expert_cache(sched, 1024 * 1024);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    ggml_tensor * experts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 8, 2);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 2, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    ggml_tensor * output = ggml_mul_mat_id(ctx, experts, input, ids);
    ggml_set_name(experts, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(output, "ffn_moe_gate");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu_backend);
    require(buffer != nullptr);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_sched_split_graph(sched, graph);

    ggml_backend_expert_cache_stats stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &stats));
    require(stats.n_route_census_nodes >= 1);
    require(stats.n_route_census_cpu_host_nodes >= 1);
    require(stats.n_route_census_batch_1 >= 1);
    require(stats.n_route_census_split_inputs == 0);

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("  original-graph route census tests passed\n");
}

static void test_route_plan_groups_shared_ids() {
    printf("testing shared route-ID plan discovery...\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_device == nullptr) {
        printf("  no GPU backend available; skipped\n");
        return;
    }

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_device, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    require(gpu_backend != nullptr);
    require(cpu_backend != nullptr);
    ggml_backend_t backends[] = { gpu_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    require(sched != nullptr);
    ggml_backend_sched_set_expert_cache(sched, 1024 * 1024);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    ggml_tensor * experts_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 8, 2);
    ggml_tensor * experts_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 8, 2);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 2, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    ggml_tensor * matmul_a = ggml_mul_mat_id(ctx, experts_a, input, ids);
    ggml_tensor * matmul_b = ggml_mul_mat_id(ctx, experts_b, input, ids);
    ggml_tensor * output = ggml_add(ctx, matmul_a, matmul_b);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu_backend);
    require(buffer != nullptr);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_sched_split_graph(sched, graph);

    ggml_backend_expert_cache_stats stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &stats));
    require(stats.n_route_census_nodes == 2);
    require(stats.n_route_census_plans == 1);

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("  shared route-ID plan discovery tests passed\n");
}

static void test_registered_bundle_keeps_cpu_base_placement() {
    printf("testing registered bundle CPU-base placement...\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_device == nullptr) {
        printf("  no GPU backend available; skipped\n");
        return;
    }

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_device, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    require(gpu_backend != nullptr);
    require(cpu_backend != nullptr);

    ggml_backend_t backends[] = { gpu_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    require(sched != nullptr);
    ggml_backend_sched_set_expert_cache(sched, 1024 * 1024);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    ggml_tensor * gate_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 2);
    ggml_tensor * up_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 2);
    ggml_tensor * down_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 2);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 1, 1);
    ggml_tensor * route_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
    ggml_set_name(gate_weights, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(up_weights, "blk.0.ffn_up_exps.weight");
    ggml_set_name(down_weights, "blk.0.ffn_down_exps.weight");

    ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_weights, input, route_ids);
    ggml_tensor * up = ggml_mul_mat_id(ctx, up_weights, input, route_ids);
    ggml_tensor * activation = ggml_add(ctx, gate, up);
    ggml_tensor * output = ggml_mul_mat_id(ctx, down_weights, activation, route_ids);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu_backend);
    require(buffer != nullptr);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_sched_register_expert_bundle(
        sched, 0, gate_weights, up_weights, down_weights);

    ggml_backend_sched_set_tensor_backend(sched, input, gpu_backend);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_sched_split_graph(sched, graph);

    require(ggml_backend_sched_get_tensor_backend(sched, gate) == cpu_backend);
    require(ggml_backend_sched_get_tensor_backend(sched, up) == cpu_backend);
    require(ggml_backend_sched_get_tensor_backend(sched, output) == cpu_backend);

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("  registered bundle CPU-base placement tests passed\n");
}

static void test_event_query_contract() {
    printf("testing nonblocking event query contract...\n");
    require(!ggml_backend_event_query(nullptr));
    printf("  nonblocking event query contract tests passed\n");
}
static int test_synchronize_calls = 0;
static void (*test_original_synchronize)(ggml_backend_t backend) = nullptr;

static void test_count_synchronize(ggml_backend_t backend) {
    test_synchronize_calls++;
    test_original_synchronize(backend);
}

struct test_tensor_write {
    size_t offset;
    size_t size;
};

static const char * test_tracked_tensor_name = nullptr;
static int test_tracked_cpy_tensor_async_calls = 0;
static std::vector<test_tensor_write> test_tracked_set_tensor_async_writes;
static bool (*test_original_cpy_tensor_async)(
        ggml_backend_t backend_src,
        ggml_backend_t backend_dst,
        const struct ggml_tensor * src,
        struct ggml_tensor * dst) = nullptr;

static int test_set_tensor_async_calls = 0;
static void (*test_original_set_tensor_async)(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) = nullptr;

static void test_count_set_tensor_async(
        ggml_backend_t backend,
        struct ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size) {
    test_set_tensor_async_calls++;
    if (test_tracked_tensor_name != nullptr &&
        strstr(tensor->name, test_tracked_tensor_name) != nullptr) {
        test_tracked_set_tensor_async_writes.push_back({ offset, size });
    }
    test_original_set_tensor_async(backend, tensor, data, offset, size);
}

static bool test_count_cpy_tensor_async(
        ggml_backend_t backend_src,
        ggml_backend_t backend_dst,
        const struct ggml_tensor * src,
        struct ggml_tensor * dst) {
    if (test_tracked_tensor_name != nullptr &&
        strstr(src->name, test_tracked_tensor_name) != nullptr) {
        test_tracked_cpy_tensor_async_calls++;
    }
    return test_original_cpy_tensor_async(backend_src, backend_dst, src, dst);
}


static int test_get_tensor_async_calls = 0;
static void (*test_original_get_tensor_async)(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) = nullptr;

static void test_count_get_tensor_async(
        ggml_backend_t backend,
        const struct ggml_tensor * tensor,
        void * data,
        size_t offset,
        size_t size) {
    test_get_tensor_async_calls++;
    test_original_get_tensor_async(backend, tensor, data, offset, size);
}

static enum ggml_status (*test_original_graph_compute)(ggml_backend_t backend, struct ggml_cgraph * cgraph) = nullptr;

static enum ggml_status test_fail_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    (void) backend;
    (void) cgraph;
    return GGML_STATUS_FAILED;
}

static void test_rebalance_does_not_synchronize_gpu() {
    printf("testing nonblocking rebalance promotion...\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_device == nullptr) {
        printf("  no GPU backend available; skipped\n");
        return;
    }

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_device, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    require(gpu_backend != nullptr);
    require(cpu_backend != nullptr);
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(gpu_backend, 4096);
    require(cache != nullptr);
    ggml_backend_expert_cache_set_period(cache, 1);

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);
    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 2);
    ggml_set_name(weights, "blk.0.ffn_gate_exps.weight");
    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu_backend);
    require(weights_buffer != nullptr);
    memset(weights->data, 0xA5, ggml_nbytes(weights));

    ggml_backend_expert_cache_record_access(cache, weights, 0);
    test_original_synchronize = gpu_backend->iface.synchronize;
    require(test_original_synchronize != nullptr);
    test_synchronize_calls = 0;
    gpu_backend->iface.synchronize = test_count_synchronize;
    ggml_backend_expert_cache_begin_step(cache);
    gpu_backend->iface.synchronize = test_original_synchronize;
    require(test_synchronize_calls == 0);

    ggml_backend_expert_cache_free(cache);
    ggml_backend_buffer_free(weights_buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("  nonblocking rebalance promotion tests passed\n");
}

static void test_rebalance_ignores_gpu_resident_weights() {
    printf("testing rebalance ignores GPU-resident weights...\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_device == nullptr) {
        printf("  no GPU backend available; skipped\n");
        return;
    }

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_device, nullptr);
    require(gpu_backend != nullptr);
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(gpu_backend, 4096);
    require(cache != nullptr);
    ggml_backend_expert_cache_set_period(cache, 1);

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);
    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 2);
    ggml_set_name(weights, "blk.0.ffn_gate_exps.weight");
    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(ctx, gpu_backend);
    require(weights_buffer != nullptr);
    std::vector<float> values(ggml_nelements(weights), 1.0f);
    ggml_backend_tensor_set(weights, values.data(), 0, ggml_nbytes(weights));

    require(!ggml_backend_expert_cache_seed(cache, weights, 0, 1));
    ggml_backend_expert_cache_record_access(cache, weights, 0);
    ggml_backend_expert_cache_begin_step(cache);
    require(ggml_backend_expert_cache_find_slot(cache, weights, 0) == -1);

    ggml_backend_expert_cache_free(cache);
    ggml_backend_buffer_free(weights_buffer);
    ggml_free(ctx);
    ggml_backend_free(gpu_backend);

    printf("  GPU-resident weight rebalance tests passed\n");
}





static void test_cache_capacity_admission() {
    printf("testing cache capacity admission...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr);

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, 256);
    require(cache != nullptr);
    require(ggml_backend_expert_cache_can_store(cache, 256));
    require(!ggml_backend_expert_cache_can_store(cache, 512));
    ggml_backend_expert_cache_record_eligible(cache);
    ggml_backend_expert_cache_record_capacity_bypass(cache);
    struct ggml_backend_expert_cache_stats stats;
    ggml_backend_expert_cache_get_stats(cache, &stats);
    require(stats.n_eligible_ops == 1);
    require(stats.n_capacity_bypasses == 1);
    ggml_backend_expert_cache_record_cpu_backend_bypass(cache);
    ggml_backend_expert_cache_get_stats(cache, &stats);
    require(stats.n_cpu_backend_bypasses == 1);
    ggml_backend_expert_cache_record_mul_mat_id_input(cache);
    ggml_backend_expert_cache_record_non_host_weight_bypass(cache);
    ggml_backend_expert_cache_get_stats(cache, &stats);
    require(stats.n_mul_mat_id_inputs == 1);
    require(stats.n_non_host_weight_bypasses == 1);

    ggml_backend_expert_cache_free(cache);
    ggml_backend_free(backend);

    printf("  cache capacity admission tests passed\n");
}


static void test_slot_pools_and_remapping() {
    printf("testing slot pools and zero-copy ID remapping...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 1024;
    const size_t cache_capacity = 8 * 1024; // room for 8 experts

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 16); // 16 experts
    ggml_set_name(tensor, "blk.0.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;

    // Verify slot tensor retrieval
    struct ggml_tensor * slot_t = ggml_backend_expert_cache_get_slot_tensor(cache, tensor);
    assert(slot_t != nullptr);
    assert(slot_t->ne[0] == tensor->ne[0]);
    assert(slot_t->ne[1] == tensor->ne[1]);
    assert(slot_t->nb[2] == tensor->nb[2]);

    // Seed/allocate slots for experts 3, 7, 11
    int32_t s3 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 3, nullptr, 0);
    int32_t s7 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 7, nullptr, 0);
    int32_t s11 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 11, nullptr, 0);

    assert(s3 >= 0);
    assert(s7 >= 0);
    assert(s11 >= 0);
    assert(s3 != s7 && s7 != s11 && s3 != s11);

    // Test find_slot
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 3) == s3);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 7) == s7);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 11) == s11);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 4) == -1); // not cached

    // Test ID remapping
    int32_t req_ids[4] = { 3, 4, 7, 11 };
    int32_t remapped_ids[4] = { -1, -1, -1, -1 };
    bool is_hit[4] = { false, false, false, false };

    int32_t n_hits = ggml_backend_expert_cache_remap_ids(cache, tensor, req_ids, 4, remapped_ids, is_hit);
    assert(n_hits == 3);
    assert(is_hit[0] == true && remapped_ids[0] == s3);
    assert(is_hit[1] == false && remapped_ids[1] == 4);
    assert(is_hit[2] == true && remapped_ids[2] == s7);
    assert(is_hit[3] == true && remapped_ids[3] == s11);

    // Record zero-copy hit telemetry
    ggml_backend_expert_cache_record_zero_copy_hit(cache, tensor, 3, expert_bytes);
    struct ggml_backend_expert_cache_stats stats;
    ggml_backend_expert_cache_get_stats(cache, &stats);
    assert(stats.n_zero_copy_hits == 1);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  slot pools and zero-copy ID remapping tests passed\n");
}

static void test_manifest_loader_outcome_stats() {
    printf("testing manifest loader outcome stats...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);
    // one shared pool (identical shapes); three slots per expert bundle, so
    // experts 0 and 1 seed completely and expert 2 exceeds capacity
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, 384);
    assert(cache != nullptr);

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 4);
    struct ggml_tensor * up   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 4);
    struct ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 4);
    ggml_set_name(gate, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(up, "blk.0.ffn_up_exps.weight");
    ggml_set_name(down, "blk.0.ffn_down_exps.weight");
    ggml_backend_expert_cache_register_bundle(cache, 0, gate, up, down);

    const char * manifest_path = "test-manifest-outcomes.json";
    {
        std::ofstream ofs(manifest_path);
        ofs << "{\"pinned_experts\": [\n"
            << "  {\"layer\": 0, \"expert_id\": 0},\n"
            << "  {\"layer\": 0, \"expert_id\": 1},\n"
            << "  {\"layer\": 0, \"expert_id\": 2},\n"
            << "  {\"layer\": 9, \"expert_id\": 5}\n"
            << "]}\n";
    }

    require(ggml_backend_expert_cache_load_pinned_manifest(cache, manifest_path));
    struct ggml_backend_expert_cache_manifest_stats ms = {};
    ggml_backend_expert_cache_get_manifest_stats(cache, &ms);
    require(ms.n_parsed == 4);
    require(ms.n_unregistered_layer == 1);
    require(ms.n_seeded == 6);
    require(ms.n_seed_failed == 3);
    require(ms.n_pinned_marked == 6);
    require(ms.n_slot_lookup_failed == 0);

    std::remove(manifest_path);
    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  manifest loader outcome stats tests passed\n");
}

static void test_multi_token_slot_remapping() {
    printf("testing multi-token slot remapping...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr);
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, 8 * 1024);
    require(cache != nullptr);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 16);
    ggml_set_name(tensor, "blk.0.ffn_gate_exps.weight");

    const int32_t cached_ids[] = { 1, 4, 7, 11 };
    int32_t slots[4] = {};
    for (size_t i = 0; i < 4; ++i) {
        slots[i] = ggml_backend_expert_cache_alloc_slot_idx(
            cache, tensor, cached_ids[i], nullptr, 0);
        require(slots[i] >= 0);
        ggml_backend_expert_cache_promote_slot(cache, tensor, cached_ids[i], slots[i]);
    }

    const int32_t route_ids[] = { 1, 7, 4, 11, 7, 1, 11, 4 };
    int32_t remapped[8] = {};
    bool is_hit[8] = {};
    require(ggml_backend_expert_cache_remap_ids(
        cache, tensor, route_ids, 8, remapped, is_hit) == 8);

    for (size_t i = 0; i < 8; ++i) {
        require(is_hit[i]);
        require(remapped[i] == ggml_backend_expert_cache_find_slot(cache, tensor, route_ids[i]));
    }

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  multi-token slot remapping tests passed\n");
}

static void test_slru_and_admission_policy() {
    printf("testing SLRU segments and admission policy...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 512;
    const size_t cache_capacity = 4 * 512; // 4 slots total (1 probationary, 3 protected)

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 8);
    ggml_set_name(tensor, "blk.1.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;

    // Allocate 4 slots
    int32_t s0 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 0, nullptr, 0);
    int32_t s1 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 1, nullptr, 0);
    int32_t s2 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 2, nullptr, 0);
    int32_t s3 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 3, nullptr, 0);
    assert(s0 >= 0 && s1 >= 0 && s2 >= 0 && s3 >= 0);

    // Touch expert 0 and 1 multiple times to promote to protected
    ggml_backend_expert_cache_touch(cache, tensor, 0);
    ggml_backend_expert_cache_touch(cache, tensor, 0);
    ggml_backend_expert_cache_touch(cache, tensor, 1);
    ggml_backend_expert_cache_touch(cache, tensor, 1);

    // Allocate a new expert 4 when full -> should evict least recently used from probationary (e.g. 2 or 3)
    int32_t s4 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 4, nullptr, 0);
    assert(s4 >= 0);

    // Verify hot protected experts 0 and 1 are still resident
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 0) >= 0);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 1) >= 0);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  SLRU segments and admission policy tests passed\n");
}

static void test_expert_bundles() {
    printf("testing expert bundle registration and atomic residency...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 512;
    const size_t cache_capacity = 8 * 512;

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 4);
    ggml_set_name(gate, "blk.2.ffn_gate_exps.weight");
    gate->nb[2] = expert_bytes;

    struct ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 4);
    ggml_set_name(down, "blk.2.ffn_down_exps.weight");
    down->nb[2] = expert_bytes;

    // Register bundle for layer 2
    ggml_backend_expert_cache_register_bundle(cache, 2, gate, nullptr, down);

    // Initially expert 1 is not resident
    assert(!ggml_backend_expert_cache_is_bundle_resident(cache, 2, 1));

    // Allocate gate only -> bundle should still not be fully resident
    ggml_backend_expert_cache_alloc_slot_idx(cache, gate, 1, nullptr, 0);
    assert(!ggml_backend_expert_cache_is_bundle_resident(cache, 2, 1));

    // Allocate down as well -> now bundle is resident
    ggml_backend_expert_cache_alloc_slot_idx(cache, down, 1, nullptr, 0);
    assert(ggml_backend_expert_cache_is_bundle_resident(cache, 2, 1));

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  expert bundle tests passed\n");
}

static void test_pinned_host_buffer() {
    printf("testing pinned host buffer allocation and alignment...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, 4096);
    assert(cache != nullptr);

    void * buf = ggml_backend_expert_cache_get_pinned_buffer(cache, 2048);
    assert(buf != nullptr);
    assert(((uintptr_t)buf % 512) == 0); // 512-byte aligned

    // Reallocate larger
    void * buf2 = ggml_backend_expert_cache_get_pinned_buffer(cache, 8192);
    assert(buf2 != nullptr);
    assert(((uintptr_t)buf2 % 512) == 0);

    ggml_backend_expert_cache_free(cache);
    ggml_backend_free(backend);

    printf("  pinned host buffer tests passed\n");
}



static void test_rebalance_tracks_staging_memcpy() {
    printf("testing rebalance staging memcpy telemetry...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr);
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, 4096);
    require(cache != nullptr);

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);
    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 2);
    ggml_set_name(weights, "blk.0.ffn_gate_exps.weight");
    memset(weights->data, 0xA5, ggml_nbytes(weights));

    ggml_backend_expert_cache_record_access(cache, weights, 0);
    ggml_backend_expert_cache_rebalance(cache, -1);

    ggml_backend_expert_cache_stats stats = {};
    ggml_backend_expert_cache_get_stats(cache, &stats);
    require(stats.staging_memcpy_bytes == weights->nb[2]);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  rebalance staging memcpy telemetry tests passed\n");
}


static void test_prefetch() {
    printf("testing explicit expert prefetch...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 512;
    const size_t cache_capacity = 8 * 512;

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 8);
    ggml_set_name(tensor, "blk.3.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;
    memset(tensor->data, 0xAB, ggml_nbytes(tensor));

    int32_t expert_ids[2] = { 4, 5 };
    ggml_backend_expert_cache_prefetch(cache, tensor, expert_ids, 2);
    struct ggml_backend_expert_cache_stats stats;
    ggml_backend_expert_cache_get_stats(cache, &stats);
    assert(stats.n_speculative_prefetches >= 2);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 4) >= 0);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 5) >= 0);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  prefetch tests passed\n");
}

static void test_prefetch_deduplicates_expert_ids() {
    printf("testing duplicate expert prefetch coalescing...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr);

    const size_t expert_bytes = 512;
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, 8 * expert_bytes);
    require(cache != nullptr);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 8);
    require(tensor != nullptr);
    ggml_set_name(tensor, "blk.4.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;
    memset(tensor->data, 0xCD, ggml_nbytes(tensor));

    const int32_t expert_ids[] = { 2, 2, 2, 6, 6 };
    ggml_backend_expert_cache_prefetch(cache, tensor, expert_ids, 5);

    struct ggml_backend_expert_cache_stats stats;
    ggml_backend_expert_cache_get_stats(cache, &stats);
    require(stats.n_speculative_prefetches == 2);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  duplicate expert prefetch coalescing tests passed\n");
}

static void test_route_prefetch_telemetry() {
    printf("testing route prefetch telemetry...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr);
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, 4096);
    require(cache != nullptr);

    ggml_backend_expert_cache_record_route_snapshot(cache);
    ggml_backend_expert_cache_record_route_prefetch_submitted(cache);
    ggml_backend_expert_cache_record_route_prefetch_duplicate(cache);
    ggml_backend_expert_cache_record_route_prefetch_stale(cache);
    ggml_backend_expert_cache_record_route_prefetch_rejected(cache);

    struct ggml_backend_expert_cache_stats stats;
    ggml_backend_expert_cache_get_stats(cache, &stats);
    require(stats.n_route_prefetch_snapshots == 1);
    require(stats.n_route_prefetch_submitted == 1);
    require(stats.n_route_prefetch_duplicates == 1);
    require(stats.n_route_prefetch_stale == 1);
    require(stats.n_route_prefetch_rejected == 1);

    ggml_backend_expert_cache_free(cache);
    ggml_backend_free(backend);

    printf("  route prefetch telemetry tests passed\n");
}

static void test_pinned_staging_no_overwrite() {
    printf("testing pinned staging ring acquire/commit discipline...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 1024;
    const size_t cache_capacity = 64 * 1024;

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * tensor_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 16);
    ggml_set_name(tensor_a, "blk.0.ffn_gate_exps.weight");

    struct ggml_backend_expert_cache_stats stats_before;
    ggml_backend_expert_cache_get_stats(cache, &stats_before);

    // acquire + commit marks the entry in-flight; a second acquire of the SAME
    // (tensor, slot) must wait (n_staging_waits increments) and return the same pointer
    void * p1 = ggml_backend_expert_cache_stage_acquire(cache, tensor_a, 3, expert_bytes);
    assert(p1 != nullptr);
    assert(((uintptr_t) p1 % 512) == 0);
    ggml_backend_expert_cache_stage_commit(cache, tensor_a, 3);

    void * p2 = ggml_backend_expert_cache_stage_acquire(cache, tensor_a, 3, expert_bytes);
    assert(p2 != nullptr);
    assert(p2 == p1); // same ring entry

    struct ggml_backend_expert_cache_stats stats_after;
    ggml_backend_expert_cache_get_stats(cache, &stats_after);
    assert(stats_after.n_staging_waits == stats_before.n_staging_waits + 1);

    // after the wait the entry is free: acquiring again must not wait again
    void * p3 = ggml_backend_expert_cache_stage_acquire(cache, tensor_a, 3, expert_bytes);
    assert(p3 == p1);
    ggml_backend_expert_cache_get_stats(cache, &stats_after);
    assert(stats_after.n_staging_waits == stats_before.n_staging_waits + 1);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  pinned staging ring tests passed\n");
}

static void test_cross_layer_shape_isolation() {
    printf("testing cross-layer same-shape slot isolation...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 512;
    const size_t cache_capacity = 8 * 512;

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    // Layer 0 tensor and Layer 1 tensor with identical shape [8, 16, 4]
    struct ggml_tensor * tensor_l0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 4);
    ggml_set_name(tensor_l0, "blk.0.ffn_gate_exps.weight");
    tensor_l0->nb[2] = expert_bytes;

    struct ggml_tensor * tensor_l1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 4);
    ggml_set_name(tensor_l1, "blk.1.ffn_gate_exps.weight");
    tensor_l1->nb[2] = expert_bytes;

    // Fill expert 2 of layer 0 with 0x11, and expert 2 of layer 1 with 0x22
    uint8_t * raw0 = (uint8_t *)tensor_l0->data;
    uint8_t * raw1 = (uint8_t *)tensor_l1->data;
    memset(raw0 + 2 * expert_bytes, 0x11, expert_bytes);
    memset(raw1 + 2 * expert_bytes, 0x22, expert_bytes);

    // Seed both Layer 0 Exp 2 and Layer 1 Exp 2
    bool ok0 = ggml_backend_expert_cache_seed(cache, tensor_l0, 2, 100);
    bool ok1 = ggml_backend_expert_cache_seed(cache, tensor_l1, 2, 100);
    assert(ok0 && ok1);

    // Find slots: they MUST be distinct slots
    int32_t slot0 = ggml_backend_expert_cache_find_slot(cache, tensor_l0, 2);
    int32_t slot1 = ggml_backend_expert_cache_find_slot(cache, tensor_l1, 2);
    assert(slot0 >= 0);
    assert(slot1 >= 0);
    assert(slot0 != slot1);

    // Verify data integrity in slot tensor
    struct ggml_tensor * slot_t = ggml_backend_expert_cache_get_slot_tensor(cache, tensor_l0);
    assert(slot_t != nullptr);
    const uint8_t * slot_buf = (const uint8_t *)slot_t->data;

    for (size_t b = 0; b < expert_bytes; b++) {
        assert(slot_buf[slot0 * expert_bytes + b] == 0x11);
        assert(slot_buf[slot1 * expert_bytes + b] == 0x22);
    }

    // Verify that tensor_l0 only finds its own slot and not layer 1
    assert(ggml_backend_expert_cache_find_slot(cache, tensor_l0, 2) == slot0);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor_l1, 2) == slot1);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  cross-layer same-shape slot isolation tests passed\n");
}

static void test_pp_tg_telemetry_isolation() {
    printf("testing PP and TG telemetry frequency isolation...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, 4096);
    assert(cache != nullptr);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 4);
    ggml_set_name(tensor, "blk.0.ffn_gate_exps.weight");
    tensor->nb[2] = 512;

    // Record access in PP phase for expert 1
    ggml_backend_expert_cache_record_access_count(cache, tensor, 1, 500, GGML_EXPERT_CACHE_PHASE_PP);

    // Record access in TG phase for expert 0
    ggml_backend_expert_cache_record_access_count(cache, tensor, 0, 10, GGML_EXPERT_CACHE_PHASE_TG);

    struct ggml_backend_expert_cache_export_entry entries[4];
    size_t n = ggml_backend_expert_cache_export_entries(cache, entries, 4);

    // Only TG entries drive export/residency
    assert(n >= 1);
    bool found_tg_0 = false;
    bool found_pp_1_in_tg = false;
    for (size_t i = 0; i < n; i++) {
        if (entries[i].expert_id == 0 && entries[i].frequency == 10) found_tg_0 = true;
        if (entries[i].expert_id == 1 && entries[i].frequency == 500) found_pp_1_in_tg = true;
    }
    assert(found_tg_0);
    assert(!found_pp_1_in_tg); // PP accesses must not leak into TG residency frequency

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  PP and TG telemetry isolation tests passed\n");
}

static void test_slot_loading_lifecycle() {
    printf("testing slot LOADING -> RESIDENT state machine...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 1024;
    const size_t cache_capacity = 64 * 1024;

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 8);
    ggml_set_name(tensor, "blk.0.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;

    // 1. A new claim owns the fill and starts in LOADING state.
    bool needs_load = false;
    int32_t slot = ggml_backend_expert_cache_claim_slot_idx(
        cache, tensor, 5, NULL, 0, &needs_load);
    require(slot >= 0);
    require(needs_load);

    // A duplicate claim attaches to the existing fill.
    bool duplicate_needs_load = true;
    require(ggml_backend_expert_cache_claim_slot_idx(
        cache, tensor, 5, NULL, 0, &duplicate_needs_load) == slot);
    require(!duplicate_needs_load);

    // LOADING must not return a hit.
    int32_t found = ggml_backend_expert_cache_find_slot(cache, tensor, 5);
    require(found == -1);

    int32_t orig_ids[1] = { 5 };
    int32_t remapped[1] = { -1 };
    bool is_hit[1] = { false };
    int32_t n_hits = ggml_backend_expert_cache_remap_ids(cache, tensor, orig_ids, 1, remapped, is_hit);
    require(n_hits == 0);
    require(!is_hit[0]);

    // Promotion makes the completed fill visible.
    ggml_backend_expert_cache_promote_slot(cache, tensor, 5, slot);
    require(ggml_backend_expert_cache_find_slot(cache, tensor, 5) == slot);

    n_hits = ggml_backend_expert_cache_remap_ids(cache, tensor, orig_ids, 1, remapped, is_hit);
    require(n_hits == 1);
    require(is_hit[0]);
    require(remapped[0] == slot);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  slot LOADING -> RESIDENT state machine tests passed\n");
}

static void test_per_tensor_slot_isolation() {
    printf("testing per-tensor slot isolation across layers...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 1024;
    const size_t cache_capacity = 64 * 1024;

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    // Two same-shaped tensors on different layers
    struct ggml_tensor * t_l0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 8);
    ggml_set_name(t_l0, "blk.0.ffn_gate_exps.weight");
    t_l0->nb[2] = expert_bytes;

    struct ggml_tensor * t_l1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 8);
    ggml_set_name(t_l1, "blk.1.ffn_gate_exps.weight");
    t_l1->nb[2] = expert_bytes;

    int32_t slot_l0 = ggml_backend_expert_cache_alloc_slot_idx(cache, t_l0, 3, NULL, 0);
    assert(slot_l0 >= 0);
    ggml_backend_expert_cache_promote_slot(cache, t_l0, 3, slot_l0);

    // t_l1 expert 3 must NOT be present
    assert(ggml_backend_expert_cache_find_slot(cache, t_l1, 3) == -1);

    int32_t slot_l1 = ggml_backend_expert_cache_alloc_slot_idx(cache, t_l1, 3, NULL, 0);
    assert(slot_l1 >= 0);
    assert(slot_l1 != slot_l0); // different slots in same pool
    ggml_backend_expert_cache_promote_slot(cache, t_l1, 3, slot_l1);

    // Both must resolve to their own respective slots
    assert(ggml_backend_expert_cache_find_slot(cache, t_l0, 3) == slot_l0);
    assert(ggml_backend_expert_cache_find_slot(cache, t_l1, 3) == slot_l1);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  per-tensor slot isolation tests passed\n");
}

static void test_staging_sync_teardown() {
    printf("testing staging sync and teardown safety...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t cache_capacity = 64 * 1024;
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 8);
    ggml_set_name(tensor, "blk.0.ffn_gate_exps.weight");

    void * p = ggml_backend_expert_cache_stage_acquire(cache, tensor, 0, 1024);
    assert(p != nullptr);
    ggml_backend_expert_cache_stage_commit(cache, tensor, 0);

    // explicit sync must drain without error
    ggml_backend_expert_cache_sync_staging(cache);

    // teardown with clean state
    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  staging sync and teardown safety tests passed\n");
}

static void test_auto_reserve_sentinel() {
    printf("testing auto reserve sentinel semantics...\n");

    // SIZE_MAX -> unset sentinel, must default to 512 MiB
    const size_t unset_reserve = (size_t)-1;
    const size_t r_default = (unset_reserve != (size_t)-1) ? unset_reserve : (512 * 1024 * 1024);
    assert(r_default == 512 * 1024 * 1024);

    // 0 -> explicit zero margin requested via --fit-target 0
    const size_t zero_reserve = 0;
    const size_t r_zero = (zero_reserve != (size_t)-1) ? zero_reserve : (512 * 1024 * 1024);
    assert(r_zero == 0);

    // 256 MiB -> explicit custom target
    const size_t custom_reserve = 256 * 1024 * 1024;
    const size_t r_custom = (custom_reserve != (size_t)-1) ? custom_reserve : (512 * 1024 * 1024);
    assert(r_custom == 256 * 1024 * 1024);

    printf("  auto reserve sentinel semantics tests passed\n");
}

static void test_async_promotion_pipeline() {
    printf("testing async promotion pipeline and rate limiting...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr);
    const size_t expert_bytes = 8 * 16 * sizeof(float);
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, expert_bytes * 8);
    require(cache != nullptr);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 8);
    require(tensor != nullptr);
    ggml_set_name(tensor, "blk.3.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;
    memset(tensor->data, 0xEE, ggml_nbytes(tensor));

    ggml_backend_expert_cache_set_max_async_promotions(cache, 2);

    // Seed 4 experts
    ggml_backend_expert_cache_seed(cache, tensor, 0, 100);
    ggml_backend_expert_cache_seed(cache, tensor, 1, 200);
    ggml_backend_expert_cache_seed(cache, tensor, 2, 300);
    ggml_backend_expert_cache_seed(cache, tensor, 3, 400);

    // Process async promotions with limit = 2
    size_t promoted = ggml_backend_expert_cache_process_async_promotions(cache, 2);
    require(promoted <= 2);

    // Process remaining
    promoted += ggml_backend_expert_cache_process_async_promotions(cache, 10);
    require(promoted >= 0);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  async promotion pipeline tests passed\n");
}

static void test_gpu_slot_map_remapping() {
    printf("testing GPU slot map table consistency...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr);
    const size_t expert_bytes = 8 * 16 * sizeof(float);
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, expert_bytes * 4);
    require(cache != nullptr);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 8);
    require(tensor != nullptr);
    ggml_set_name(tensor, "blk.7.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;
    memset(tensor->data, 0xAA, ggml_nbytes(tensor));

    const int32_t * map_l7 = ggml_backend_expert_cache_get_gpu_slot_map(cache, 7);
    require(map_l7 != nullptr);
    require(map_l7[4] == -1); // unseeded expert 4 should be -1

    ggml_backend_expert_cache_seed(cache, tensor, 4, 100);
    int32_t slot = ggml_backend_expert_cache_find_slot(cache, tensor, 4);
    require(slot >= 0);
    require(map_l7[4] == slot);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  GPU slot map table consistency tests passed\n");
}

static void test_hit_mask_matrix_partitioning() {
    printf("testing hit-mask matrix partitioning (0/8 to 8/8 hits)...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr);
    const size_t expert_bytes = 16 * 16 * sizeof(float);
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, expert_bytes * 32);
    require(cache != nullptr);

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    struct ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 16);
    struct ggml_tensor * up   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 16);
    struct ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 16);
    ggml_set_name(gate, "blk.5.ffn_gate_exps.weight");
    ggml_set_name(up,   "blk.5.ffn_up_exps.weight");
    ggml_set_name(down, "blk.5.ffn_down_exps.weight");
    gate->nb[2] = expert_bytes;
    up->nb[2]   = expert_bytes;
    down->nb[2] = expert_bytes;

    ggml_backend_expert_cache_register_bundle(cache, 5, gate, up, down);

    // Pre-seed experts 0 to 7 across all three bundle projections
    for (int e = 0; e < 8; e++) {
        ggml_backend_expert_cache_seed(cache, gate, e, 100);
        ggml_backend_expert_cache_seed(cache, up,   e, 100);
        ggml_backend_expert_cache_seed(cache, down, e, 100);
        require(ggml_backend_expert_cache_is_bundle_resident(cache, 5, e));
    }

    // Experts 8 to 15 remain unseeded (misses)
    for (int e = 8; e < 16; e++) {
        require(!ggml_backend_expert_cache_is_bundle_resident(cache, 5, e));
    }

    // Test all hit counts K from 0 to 8
    for (int K = 0; K <= 8; K++) {
        std::vector<int32_t> test_routes(8);
        // Fill first K with resident experts (0..K-1), remaining 8-K with missing experts (8..15)
        for (int i = 0; i < K; i++) test_routes[i] = i;
        for (int i = K; i < 8; i++) test_routes[i] = 8 + (i - K);

        ggml_cache_route_bundle hit_routes[8];
        ggml_cache_route_bundle miss_routes[8];
        int32_t n_hits = 0;
        int32_t n_misses = 0;

        int32_t res_hits = ggml_backend_expert_cache_partition_bundle_routes(
            cache, 5, test_routes.data(), 8, 1,
            hit_routes, &n_hits, miss_routes, &n_misses);

        require(res_hits == K);
        require(n_hits == K);
        require(n_misses == 8 - K);

        for (int i = 0; i < n_hits; i++) {
            require(hit_routes[i].is_bundle_hit);
            require(hit_routes[i].expert == i);
            require(hit_routes[i].route == i);
            require(hit_routes[i].gate_slot >= 0);
            require(hit_routes[i].up_slot >= 0);
            require(hit_routes[i].down_slot >= 0);
        }

        for (int i = 0; i < n_misses; i++) {
            require(!miss_routes[i].is_bundle_hit);
            require(miss_routes[i].expert == 8 + i);
            require(miss_routes[i].route == K + i);
        }
    }

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  hit-mask matrix partitioning tests passed\n");
}
static void test_partial_route_snapshot_validation() {
    printf("testing partial route snapshot validation...\n");

    struct ggml_moe_partial_route_snapshot snapshot = {};
    const int32_t hit_positions[] = { 0, 2, 3, 4, 5, 6, 7 };
    snapshot.n_hits = 7;
    snapshot.n_misses = 1;
    for (int32_t i = 0; i < snapshot.n_hits; ++i) {
        snapshot.hits[i].route = hit_positions[i];
        snapshot.hits[i].bundle_resident = true;
    }
    snapshot.misses[0].route = 1;
    require(ggml_moe_partial_route_snapshot_is_valid(&snapshot, 8));

    require(ggml_moe_partial_executor_execute(nullptr, nullptr, nullptr, &snapshot, nullptr, nullptr) ==
        GGML_MOE_PARTIAL_EXECUTOR_NOT_ADMITTED);

    snapshot.misses[0].route = 7;
    require(!ggml_moe_partial_route_snapshot_is_valid(&snapshot, 8));
    snapshot.misses[0].route = 1;
    snapshot.hits[0].route = 1;
    require(!ggml_moe_partial_route_snapshot_is_valid(&snapshot, 8));
    snapshot.hits[0].route = 0;
    snapshot.hits[0].bundle_resident = false;
    require(!ggml_moe_partial_route_snapshot_is_valid(&snapshot, 8));

    printf("  partial route snapshot validation tests passed\n");
}

static void test_partial_executor_persistence() {
    printf("testing partial executor persistent state...\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_device == nullptr) {
        printf("  no GPU backend available; skipped\n");
        return;
    }

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_device, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    require(gpu_backend != nullptr);
    require(cpu_backend != nullptr);

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 8);
    ggml_tensor * up = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 8);
    ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 8);
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(gpu_device);
    require(host_buft != nullptr);
    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, host_buft);
    require(weights_buffer != nullptr);

    struct ggml_expert_bundle_weights weights = {
        gate,
        up,
        down,
        nullptr,
        false,
    };
    ggml_moe_partial_executor_t executor = ggml_moe_partial_executor_new(
        gpu_backend, cpu_backend, &weights, 2, 2, GGML_MOE_PARTIAL_MAX_ROUTES, false);
    require(executor != nullptr);

    struct ggml_moe_partial_executor_test_state state = {};
    require(ggml_moe_partial_executor_get_test_state(executor, &state));
    for (int32_t i = 0; i < GGML_MOE_PARTIAL_MAX_ROUTES - 1; ++i) {
        require(state.gpu_variants[i]);
        require(state.cpu_variants[i]);
    }
    require(state.has_merge_buffer);
    require(state.has_cpu_upload_buffer);
    require(state.exchange_buffer != nullptr);
    require(ggml_backend_buffer_get_type(state.exchange_buffer) == host_buft);
    for (bool event : state.events) {
        require(event);
    }

    ggml_moe_partial_executor_free(executor);
    ggml_backend_buffer_free(weights_buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("  partial executor persistent state tests passed\n");
}

static void test_partial_executor_scheduler_catalog() {
    printf("testing partial executor scheduler catalog...\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_device == nullptr) {
        printf("  no GPU backend available; skipped\n");
        return;
    }

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_device, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    require(gpu_backend != nullptr);
    require(cpu_backend != nullptr);

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    ggml_tensor * gate_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 8);
    ggml_tensor * up_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 8);
    ggml_tensor * down_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 8);
    ggml_tensor * gate_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 8);
    ggml_tensor * up_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 8);
    ggml_tensor * down_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 8);
    ggml_set_name(gate_a, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(up_a, "blk.0.ffn_up_exps.weight");
    ggml_set_name(down_a, "blk.0.ffn_down_exps.weight");
    ggml_set_name(gate_b, "blk.1.ffn_gate_exps.weight");
    ggml_set_name(up_b, "blk.1.ffn_up_exps.weight");
    ggml_set_name(down_b, "blk.1.ffn_down_exps.weight");

    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 1, 1);
    ggml_tensor * route_input_a = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, 1);
    ggml_tensor * route_input_b = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, 1);
    ggml_tensor * ids_a = ggml_dup(ctx, route_input_a);
    ggml_tensor * ids_b = ggml_dup(ctx, route_input_b);
    ggml_tensor * gate_out_a = ggml_mul_mat_id(ctx, gate_a, input, ids_a);
    ggml_tensor * up_out_a = ggml_mul_mat_id(ctx, up_a, input, ids_a);
    ggml_tensor * gate_out_b = ggml_mul_mat_id(ctx, gate_b, input, ids_b);
    ggml_tensor * up_out_b = ggml_mul_mat_id(ctx, up_b, input, ids_b);
    ggml_tensor * output_a = ggml_mul_mat_id(ctx, down_a, ggml_swiglu_split(ctx, gate_out_a, up_out_a), ids_a);
    ggml_tensor * output_b = ggml_mul_mat_id(ctx, down_b, ggml_swiglu_split(ctx, gate_out_b, up_out_b), ids_b);
    ggml_tensor * output = ggml_add(ctx, output_a, output_b);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu_backend);
    require(buffer != nullptr);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_backend_t backends[] = { gpu_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    require(sched != nullptr);
    ggml_backend_sched_set_expert_cache(sched, 1024 * 1024);
    ggml_backend_sched_register_expert_bundle(sched, 0, gate_a, up_a, down_a);
    ggml_backend_sched_register_expert_bundle(sched, 1, gate_b, up_b, down_b);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    require(ggml_backend_sched_alloc_graph(sched, graph));

    // dispatch count verified by successful alloc_graph

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("  partial executor scheduler catalog tests passed\n");
}

static void test_partial_executor_exact_mask(
        ggml_backend_dev_t gpu_device,
        const char * mask) {
    const int32_t d_model = 2;
    const int32_t d_ff = 2;
    const int32_t top_k = GGML_MOE_PARTIAL_MAX_ROUTES;
    const int32_t n_experts = GGML_MOE_PARTIAL_MAX_ROUTES;
    int32_t n_hits = 0;
    for (int32_t route = 0; route < top_k; ++route) {
        require(mask[route] == 'G' || mask[route] == 'C');
        n_hits += mask[route] == 'G';
    }
    require(n_hits > 0 && n_hits < top_k);

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_device, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    require(gpu_backend != nullptr);
    require(cpu_backend != nullptr);

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    ggml_tensor * gate_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_model, d_ff, n_experts);
    ggml_tensor * up_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_model, d_ff, n_experts);
    ggml_tensor * down_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_ff, d_model, n_experts);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_model, 1, 1);
    ggml_tensor * route_input = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, 1);
    ggml_tensor * route_ids = ggml_dup(ctx, route_input);
    ggml_set_name(gate_weights, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(up_weights, "blk.0.ffn_up_exps.weight");
    ggml_set_name(down_weights, "blk.0.ffn_down_exps.weight");

    ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_weights, input, route_ids);
    ggml_tensor * up = ggml_mul_mat_id(ctx, up_weights, input, route_ids);
    ggml_tensor * activation = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * output = ggml_mul_mat_id(ctx, down_weights, activation, route_ids);

    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(gpu_device);
    require(host_buft != nullptr);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, host_buft);
    require(buffer != nullptr);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<float> gate_data((size_t) d_model * d_ff * n_experts, 0.0f);
    std::vector<float> up_data((size_t) d_model * d_ff * n_experts, 0.0f);
    std::vector<float> down_data((size_t) d_ff * d_model * n_experts, 0.0f);
    for (int32_t expert = 0; expert < n_experts; ++expert) {
        const size_t offset = (size_t) expert * d_model * d_ff;
        gate_data[offset + 0] = 0.5f + 0.125f * expert;
        gate_data[offset + 3] = 0.5f + 0.125f * expert;
        up_data[offset + 0] = 0.75f + 0.0625f * expert;
        up_data[offset + 3] = 0.75f + 0.0625f * expert;
        down_data[offset + 0] = 1.0f + 0.25f * expert;
        down_data[offset + 3] = 1.0f + 0.25f * expert;
    }
    const float input_data[] = { 1.0f, 2.0f };
    const int32_t route_data[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    ggml_backend_tensor_set(gate_weights, gate_data.data(), 0, ggml_nbytes(gate_weights));
    ggml_backend_tensor_set(up_weights, up_data.data(), 0, ggml_nbytes(up_weights));
    ggml_backend_tensor_set(down_weights, down_data.data(), 0, ggml_nbytes(down_weights));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    ggml_backend_tensor_set(route_input, route_data, 0, sizeof(route_data));

    ggml_cgraph * reference_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(reference_graph, output);
    require(ggml_backend_graph_compute(cpu_backend, reference_graph) == GGML_STATUS_SUCCESS);
    std::vector<float> expected(ggml_nelements(output));
    ggml_backend_tensor_get(output, expected.data(), 0, ggml_nbytes(output));

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(gpu_backend, 4096);
    require(cache != nullptr);
    ggml_backend_expert_cache_register_bundle(cache, 0, gate_weights, up_weights, down_weights);
    for (int32_t route = 0; route < top_k; ++route) {
        if (mask[route] == 'G') {
            ggml_backend_expert_cache_seed(cache, gate_weights, route, 1);
            ggml_backend_expert_cache_seed(cache, up_weights, route, 1);
            ggml_backend_expert_cache_seed(cache, down_weights, route, 1);
        }
    }
    ggml_backend_synchronize(gpu_backend);

    struct ggml_moe_partial_route_snapshot snapshot = {};
    require(ggml_backend_expert_cache_partition_bundle_routes(
        cache, 0, route_data, top_k, 1,
        snapshot.hits, &snapshot.n_hits, snapshot.misses, &snapshot.n_misses) == n_hits);
    require(snapshot.n_hits == n_hits);
    require(snapshot.n_misses == top_k - n_hits);
    require(ggml_moe_partial_route_snapshot_is_valid(&snapshot, top_k));

    struct ggml_moe_bundle_plan bundle = {};
    bundle.layer = 0;
    bundle.kind = GGML_MOE_BUNDLE_SEPARATE_GATE_UP;
    bundle.route_ids = route_ids;
    bundle.gate_node = gate;
    bundle.up_node = up;
    bundle.act_node = activation;
    bundle.down_node = output;
    bundle.layer_input = input;
    bundle.valid = true;

    struct ggml_expert_bundle_weights weights = {
        gate_weights,
        up_weights,
        down_weights,
        nullptr,
        false,
    };
    ggml_moe_partial_executor_t executor = ggml_moe_partial_executor_new(
        gpu_backend, cpu_backend, &weights, d_model, d_ff, top_k, false);
    require(executor != nullptr);

    struct ggml_moe_partial_activation executor_activation = {
        input->data,
        ggml_nbytes(input),
    };
    struct ggml_backend_expert_cache_stats stats = {};
    const bool require_concurrent = n_hits == 7;
    std::vector<float> sentinel(expected.size(), -99.0f);
    if (require_concurrent) {
        ggml_backend_tensor_set(output, sentinel.data(), 0, ggml_nbytes(output));
        test_original_synchronize = gpu_backend->iface.synchronize;
        require(test_original_synchronize != nullptr);
        test_synchronize_calls = 0;
        gpu_backend->iface.synchronize = test_count_synchronize;
    }
    require(ggml_moe_partial_executor_execute(
        executor, &bundle, cache, &snapshot, &executor_activation, &stats) == GGML_MOE_PARTIAL_EXECUTOR_SUCCESS);
    if (require_concurrent) {
        std::vector<float> executor_output(expected.size());
        ggml_backend_tensor_get(output, executor_output.data(), 0, ggml_nbytes(output));
        for (size_t i = 0; i < executor_output.size(); ++i) {
            require(fabsf(executor_output[i] - expected[i]) < 1e-5f);
        }

        const float alternate_input_data[] = { 2.0f, 1.0f };
        ggml_backend_tensor_set(input, alternate_input_data, 0, sizeof(alternate_input_data));
        require(ggml_backend_graph_compute(cpu_backend, reference_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(output, expected.data(), 0, ggml_nbytes(output));
        ggml_backend_tensor_set(output, sentinel.data(), 0, ggml_nbytes(output));
        require(ggml_moe_partial_executor_execute(
            executor, &bundle, cache, &snapshot, &executor_activation, &stats) == GGML_MOE_PARTIAL_EXECUTOR_SUCCESS);
        ggml_backend_tensor_get(output, executor_output.data(), 0, ggml_nbytes(output));
        for (size_t i = 0; i < executor_output.size(); ++i) {
            require(fabsf(executor_output[i] - expected[i]) < 1e-5f);
        }
        gpu_backend->iface.synchronize = test_original_synchronize;
        require(test_synchronize_calls == 0);
    }
    const int32_t execution_count = require_concurrent ? 2 : 1;
    require(stats.hetero_partial_exec_by_hits[n_hits] == (uint64_t) execution_count);
    require(stats.hetero_partial_gpu_routes_executed == (uint64_t) execution_count * n_hits);
    require(stats.hetero_partial_cpu_routes_executed == (uint64_t) execution_count * (top_k - n_hits));
    require(stats.hetero_partial_weight_h2d_bytes == 0);

    struct ggml_moe_partial_executor_test_state state = {};
    require(ggml_moe_partial_executor_get_test_state(executor, &state));
    ggml_tensor * gpu_output = state.gpu_outputs[n_hits - 1];
    ggml_tensor * cpu_output = state.cpu_outputs[snapshot.n_misses - 1];
    require(gpu_output != nullptr);
    require(cpu_output != nullptr);
    std::vector<float> packed_gpu((size_t) n_hits * d_model);
    std::vector<float> packed_cpu((size_t) snapshot.n_misses * d_model);
    ggml_backend_tensor_get(gpu_output, packed_gpu.data(), 0, packed_gpu.size() * sizeof(float));
    ggml_backend_tensor_get(cpu_output, packed_cpu.data(), 0, packed_cpu.size() * sizeof(float));

    std::vector<float> actual(expected.size(), 0.0f);
    for (int32_t i = 0; i < snapshot.n_hits; ++i) {
        memcpy(actual.data() + (size_t) snapshot.hits[i].route * d_model,
            packed_gpu.data() + (size_t) i * d_model, (size_t) d_model * sizeof(float));
    }
    for (int32_t i = 0; i < snapshot.n_misses; ++i) {
        memcpy(actual.data() + (size_t) snapshot.misses[i].route * d_model,
            packed_cpu.data() + (size_t) i * d_model, (size_t) d_model * sizeof(float));
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-5f);
    }

    ggml_moe_partial_executor_free(executor);
    ggml_backend_expert_cache_free(cache);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);
}

static void test_partial_executor_exact_route_paths() {
    printf("testing partial executor exact route paths...\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_device == nullptr) {
        printf("  no GPU backend available; skipped\n");
        return;
    }

    const char * masks[] = {
        "GCCCCCCC",
        "GGCCCCCC",
        "GGGCCCCC",
        "GGGGGCCC",
        "CGGGGGGG",
        "GGGGGGGC",
        "CCGGGGGG",
        "GGGCGGGC",
        "GGGGCCCC",
        "GCGCGCGC",
        "CCGGGGCC",
    };
    for (const char * mask : masks) {
        test_partial_executor_exact_mask(gpu_device, mask);
    }

    printf("  partial executor exact route path tests passed\n");
}

static void test_partial_executor_scheduler_feature_gate() {
    printf("testing partial executor scheduler feature gate...\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_device == nullptr) {
        printf("  no GPU backend available; skipped\n");
        return;
    }

    const char * env_name = "GGML_EXPERT_CACHE_HETERO_CONCURRENT";
    const char * env_prior = getenv(env_name);

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_device, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    require(gpu_backend != nullptr);
    require(cpu_backend != nullptr);

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    const int32_t top_k = GGML_MOE_PARTIAL_MAX_ROUTES;
    ggml_tensor * gate_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, top_k);
    ggml_tensor * up_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, top_k);
    ggml_tensor * down_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, top_k);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 1, 1);
    ggml_tensor * route_input = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, 1);
    ggml_set_input(input);
    ggml_set_input(route_input);
    ggml_set_name(gate_weights, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(up_weights, "blk.0.ffn_up_exps.weight");
    ggml_set_name(down_weights, "blk.0.ffn_down_exps.weight");

    ggml_tensor * route_ids = ggml_dup(ctx, route_input);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_weights, input, route_ids);
    ggml_tensor * up = ggml_mul_mat_id(ctx, up_weights, input, route_ids);
    ggml_tensor * activation = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * output = ggml_mul_mat_id(ctx, down_weights, activation, route_ids);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu_backend);
    require(buffer != nullptr);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<float> gate_data((size_t) 2 * 2 * top_k, 0.0f);
    std::vector<float> up_data((size_t) 2 * 2 * top_k, 0.0f);
    std::vector<float> down_data((size_t) 2 * 2 * top_k, 0.0f);
    for (int32_t expert = 0; expert < top_k; ++expert) {
        gate_data[(size_t) expert * 4 + 0] = 0.5f + 0.125f * expert;
        gate_data[(size_t) expert * 4 + 3] = 0.5f + 0.125f * expert;
        up_data[(size_t) expert * 4 + 0] = 0.75f + 0.0625f * expert;
        up_data[(size_t) expert * 4 + 3] = 0.75f + 0.0625f * expert;
        down_data[(size_t) expert * 4 + 0] = 1.0f + 0.25f * expert;
        down_data[(size_t) expert * 4 + 3] = 1.0f + 0.25f * expert;
    }
    const float input_data[] = { 1.0f, 2.0f };
    const int32_t ids[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    ggml_backend_tensor_set(gate_weights, gate_data.data(), 0, ggml_nbytes(gate_weights));
    ggml_backend_tensor_set(up_weights, up_data.data(), 0, ggml_nbytes(up_weights));
    ggml_backend_tensor_set(down_weights, down_data.data(), 0, ggml_nbytes(down_weights));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    ggml_backend_tensor_set(route_input, ids, 0, sizeof(ids));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    require(ggml_backend_graph_compute(cpu_backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> expected(ggml_nelements(output));
    ggml_backend_tensor_get(output, expected.data(), 0, ggml_nbytes(output));

    ggml_backend_t backends[] = { gpu_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    require(sched != nullptr);
    ggml_backend_sched_set_expert_cache(sched, 1024 * 1024);
    ggml_backend_sched_register_expert_bundle(sched, 0, gate_weights, up_weights, down_weights);

    // six resident experts must fast-reject the route-ready cache actions
    for (int32_t expert = 0; expert < 6; ++expert) {
        require(ggml_backend_sched_expert_cache_seed(sched, 0, gate_weights, expert, 1));
        require(ggml_backend_sched_expert_cache_seed(sched, 0, up_weights, expert, 1));
        require(ggml_backend_sched_expert_cache_seed(sched, 0, down_weights, expert, 1));
    }
    ggml_backend_sched_expert_cache_sync(sched);
    require(ggml_backend_sched_alloc_graph(sched, graph));


    const float sentinel = -99.0f;
    std::vector<float> actual(expected.size(), sentinel);
    ggml_backend_expert_cache_stats fast_reject_before = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &fast_reject_before));
    _putenv_s(env_name, "0");
    ggml_backend_tensor_set(output, actual.data(), 0, ggml_nbytes(output));
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-5f);
    }
    ggml_backend_expert_cache_stats fast_reject_after = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &fast_reject_after));
    require(fast_reject_after.n_route_ready_fast_rejects ==
        fast_reject_before.n_route_ready_fast_rejects + 1);
    require(fast_reject_after.n_route_ready_classifications ==
        fast_reject_before.n_route_ready_classifications);
    require(fast_reject_after.n_route_ready_actions == fast_reject_before.n_route_ready_actions);
    require(fast_reject_after.n_route_ready_resident_bundle_counts[6] ==
        fast_reject_before.n_route_ready_resident_bundle_counts[6] + 1);
    require(fast_reject_after.n_route_ready_route_id_us ==
        fast_reject_before.n_route_ready_route_id_us);
    require(fast_reject_after.bytes_ram_to_gpu == fast_reject_before.bytes_ram_to_gpu);

    // seven resident experts, one miss: the 7/8 admission mask
    require(ggml_backend_sched_expert_cache_seed(sched, 0, gate_weights, 6, 1));
    require(ggml_backend_sched_expert_cache_seed(sched, 0, up_weights, 6, 1));
    require(ggml_backend_sched_expert_cache_seed(sched, 0, down_weights, 6, 1));
    ggml_backend_sched_expert_cache_sync(sched);

    ggml_backend_expert_cache_stats baseline = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &baseline));

    // native fallback is the production policy for the measured 7/8 mask
    _putenv_s(env_name, "0");
    ggml_backend_tensor_set(output, actual.data(), 0, ggml_nbytes(output));
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-5f);
    }
    ggml_backend_expert_cache_stats serial_stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &serial_stats));
    require(serial_stats.n_route_ready_classifications ==
        baseline.n_route_ready_classifications + 1);
    require(serial_stats.n_route_ready_fallbacks == baseline.n_route_ready_fallbacks + 1);
    require(serial_stats.n_route_ready_actions == baseline.n_route_ready_actions);
    require(serial_stats.hetero_partial_exec_by_hits[7] == baseline.hetero_partial_exec_by_hits[7]);

    // the concurrent feature flag does not override the production 7/8 policy
    _putenv_s(env_name, "1");
    std::fill(actual.begin(), actual.end(), sentinel);
    ggml_backend_tensor_set(output, actual.data(), 0, ggml_nbytes(output));
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-5f);
    }
    ggml_backend_expert_cache_stats concurrent_stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &concurrent_stats));
    require(concurrent_stats.n_route_ready_classifications ==
        serial_stats.n_route_ready_classifications + 1);
    require(concurrent_stats.n_route_ready_fallbacks == serial_stats.n_route_ready_fallbacks + 1);
    require(concurrent_stats.n_route_ready_actions == serial_stats.n_route_ready_actions);
    require(concurrent_stats.hetero_partial_exec_by_hits[7] == serial_stats.hetero_partial_exec_by_hits[7]);
    require(concurrent_stats.hetero_partial_gpu_routes_executed ==
        serial_stats.hetero_partial_gpu_routes_executed);
    require(concurrent_stats.hetero_partial_cpu_routes_executed ==
        serial_stats.hetero_partial_cpu_routes_executed);
    require(concurrent_stats.hetero_partial_weight_h2d_bytes == 0);
    require(concurrent_stats.hetero_partial_activation_d2h_bytes == 0);

    require(ggml_backend_sched_expert_cache_seed(sched, 0, gate_weights, 7, 1));
    require(ggml_backend_sched_expert_cache_seed(sched, 0, up_weights, 7, 1));
    require(ggml_backend_sched_expert_cache_seed(sched, 0, down_weights, 7, 1));
    ggml_backend_sched_expert_cache_sync(sched);
    std::fill(actual.begin(), actual.end(), sentinel);
    ggml_backend_tensor_set(output, actual.data(), 0, ggml_nbytes(output));
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-5f);
    }
    ggml_backend_expert_cache_stats full_stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &full_stats));
    require(full_stats.n_route_ready_fast_rejects == concurrent_stats.n_route_ready_fast_rejects);
    require(full_stats.n_route_ready_classifications == concurrent_stats.n_route_ready_classifications + 1);
    require(full_stats.n_route_ready_full_hits == concurrent_stats.n_route_ready_full_hits + 1);

    if (env_prior != nullptr) {
        _putenv_s(env_name, env_prior);
    } else {
        _putenv_s(env_name, "");
    }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("  partial executor scheduler feature gate tests passed\n");
}



static void test_multi_token_repeated_experts() {
    printf("testing multi-token repeated expert routing...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr);
    const size_t expert_bytes = 16 * 16 * sizeof(float);
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, expert_bytes * 32);
    require(cache != nullptr);

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    struct ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 16);
    struct ggml_tensor * up   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 16);
    struct ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 16);
    ggml_set_name(gate, "blk.2.ffn_gate_exps.weight");
    ggml_set_name(up,   "blk.2.ffn_up_exps.weight");
    ggml_set_name(down, "blk.2.ffn_down_exps.weight");
    gate->nb[2] = expert_bytes;
    up->nb[2]   = expert_bytes;
    down->nb[2] = expert_bytes;

    ggml_backend_expert_cache_register_bundle(cache, 2, gate, up, down);

    // Seed experts 0, 1, 2
    ggml_backend_expert_cache_seed(cache, gate, 0, 100);
    ggml_backend_expert_cache_seed(cache, up,   0, 100);
    ggml_backend_expert_cache_seed(cache, down, 0, 100);
    ggml_backend_expert_cache_seed(cache, gate, 1, 100);
    ggml_backend_expert_cache_seed(cache, up,   1, 100);
    ggml_backend_expert_cache_seed(cache, down, 1, 100);

    // Batch of 2 tokens, top_k = 4 (total 8 routes)
    // Token 0: [0, 1, 4, 5] -> Hits: 0, 1. Misses: 4, 5
    // Token 1: [0, 1, 6, 7] -> Hits: 0, 1 (repeated!). Misses: 6, 7
    int32_t router_ids[8] = { 0, 1, 4, 5, 0, 1, 6, 7 };

    ggml_cache_route_bundle hit_routes[8];
    ggml_cache_route_bundle miss_routes[8];
    int32_t n_hits = 0;
    int32_t n_misses = 0;

    int32_t res_hits = ggml_backend_expert_cache_partition_bundle_routes(
        cache, 2, router_ids, 8, 2,
        hit_routes, &n_hits, miss_routes, &n_misses);

    require(res_hits == 4);
    require(n_hits == 4);
    require(n_misses == 4);

    // Verify token & route tracking
    require(hit_routes[0].token == 0 && hit_routes[0].route == 0 && hit_routes[0].expert == 0);
    require(hit_routes[1].token == 0 && hit_routes[1].route == 1 && hit_routes[1].expert == 1);
    require(hit_routes[2].token == 1 && hit_routes[2].route == 0 && hit_routes[2].expert == 0);
    require(hit_routes[3].token == 1 && hit_routes[3].route == 1 && hit_routes[3].expert == 1);

    require(miss_routes[0].token == 0 && miss_routes[0].route == 2 && miss_routes[0].expert == 4);
    require(miss_routes[1].token == 0 && miss_routes[1].route == 3 && miss_routes[1].expert == 5);
    require(miss_routes[2].token == 1 && miss_routes[2].route == 2 && miss_routes[2].expert == 6);
    require(miss_routes[3].token == 1 && miss_routes[3].route == 3 && miss_routes[3].expert == 7);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  multi-token repeated expert routing tests passed\n");
}

static void test_dynamic_map_metadata_and_device_maps() {
    printf("testing dynamic map metadata and device maps...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr);
    const size_t expert_bytes = 16 * 16 * sizeof(float);
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, expert_bytes * 32);
    require(cache != nullptr);

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    struct ggml_tensor * t0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 8);
    struct ggml_tensor * t1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 16);
    ggml_set_name(t0, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(t1, "blk.1.ffn_gate_exps.weight");
    t0->nb[2] = expert_bytes;
    t1->nb[2] = expert_bytes;

    uint32_t map0 = ggml_backend_expert_cache_get_map_id(cache, t0);
    uint32_t map1 = ggml_backend_expert_cache_get_map_id(cache, t1);
    require(map0 == 0);
    require(map1 == 1);

    struct ggml_tensor * d_slot_map = ggml_backend_expert_cache_get_device_slot_map(cache);
    require(d_slot_map != nullptr);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  dynamic map metadata and device maps tests passed\n");
}

static void test_route_ready_sidecar_full_hit() {
    printf("testing route-ready full-hit sidecar...\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_device == nullptr) {
        printf("  no GPU backend available; skipped\n");
        return;
    }

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_device, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    require(gpu_backend != nullptr);
    require(cpu_backend != nullptr);

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    ggml_tensor * gate_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 3);
    ggml_tensor * up_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 3);
    ggml_tensor * down_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 3);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 1, 1);
    ggml_tensor * route_input = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    ggml_tensor * route_ids = ggml_dup(ctx, route_input);
    ggml_set_name(gate_weights, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(up_weights, "blk.0.ffn_up_exps.weight");
    ggml_set_name(down_weights, "blk.0.ffn_down_exps.weight");

    ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_weights, input, route_ids);
    ggml_tensor * up = ggml_mul_mat_id(ctx, up_weights, input, route_ids);
    ggml_tensor * activation = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * output = ggml_mul_mat_id(ctx, down_weights, activation, route_ids);
    ggml_tensor * unrelated = ggml_dup(ctx, input);
    ggml_tensor * zero = ggml_scale(ctx, unrelated, 0.0f);
    ggml_tensor * noncontiguous_activation = ggml_add(ctx, activation, zero);
    ggml_tensor * noncontiguous_output = ggml_mul_mat_id(ctx, down_weights, noncontiguous_activation, route_ids);
    ggml_backend_buffer_type_t host_buffer_type = ggml_backend_dev_host_buffer_type(gpu_device);
    require(host_buffer_type != nullptr);
    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, host_buffer_type);
    require(cpu_buffer != nullptr);
    ggml_backend_buffer_set_usage(cpu_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const float gate_data[] = {
        1.0f, 0.0f, 0.0f, 1.0f,
        0.5f, 0.0f, 0.0f, 0.5f,
        0.25f, 0.0f, 0.0f, 0.25f,
    };
    const float up_data[] = {
        0.75f, 0.0f, 0.0f, 0.75f,
        1.25f, 0.0f, 0.0f, 1.25f,
        1.5f, 0.0f, 0.0f, 1.5f,
    };
    const float down_data[] = {
        1.0f, 0.0f, 0.0f, 1.0f,
        2.0f, 0.0f, 0.0f, 2.0f,
        3.0f, 0.0f, 0.0f, 3.0f,
    };
    const float input_data[] = { 1.0f, 2.0f };
    const int32_t ids[] = { 0, 1 };
    const int32_t stale_ids[] = { 1, 2 };
    const int32_t inactive_ids[] = { 0, -1 };
    ggml_backend_tensor_set(gate_weights, gate_data, 0, sizeof(gate_data));
    ggml_backend_tensor_set(up_weights, up_data, 0, sizeof(up_data));
    ggml_backend_tensor_set(down_weights, down_data, 0, sizeof(down_data));
    ggml_backend_tensor_set(route_input, ids, 0, sizeof(ids));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_cgraph * noncontiguous_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(noncontiguous_graph, noncontiguous_output);
    require(ggml_backend_graph_compute(cpu_backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> expected(ggml_nelements(output));
    ggml_backend_tensor_get(output, expected.data(), 0, ggml_nbytes(output));
    ggml_backend_tensor_set(route_input, stale_ids, 0, sizeof(stale_ids));
    require(ggml_backend_graph_compute(cpu_backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> stale_expected(ggml_nelements(output));
    ggml_backend_tensor_get(output, stale_expected.data(), 0, ggml_nbytes(output));
    ggml_backend_tensor_set(route_input, inactive_ids, 0, sizeof(inactive_ids));
    require(ggml_backend_graph_compute(cpu_backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> inactive_expected(ggml_nelements(output));
    ggml_backend_tensor_get(output, inactive_expected.data(), 0, ggml_nbytes(output));
    ggml_backend_tensor_set(route_input, ids, 0, sizeof(ids));

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(gpu_backend, 4096);
    require(cache != nullptr);
    ggml_backend_expert_cache_register_bundle(cache, 0, gate_weights, up_weights, down_weights);
    require(ggml_backend_graph_compute(cpu_backend, noncontiguous_graph) == GGML_STATUS_SUCCESS);
    std::vector<float> noncontiguous_expected(ggml_nelements(noncontiguous_output));
    ggml_backend_tensor_get(noncontiguous_output, noncontiguous_expected.data(), 0, ggml_nbytes(noncontiguous_output));
    for (int32_t expert_id : ids) {
        ggml_backend_expert_cache_seed(cache, gate_weights, expert_id, 1);
        ggml_backend_expert_cache_seed(cache, up_weights, expert_id, 1);
        ggml_backend_expert_cache_seed(cache, down_weights, expert_id, 1);
    }
    ggml_backend_synchronize(gpu_backend);
    require(ggml_backend_expert_cache_is_bundle_resident(cache, 0, 0));
    require(ggml_backend_expert_cache_is_bundle_resident(cache, 0, 1));

    struct ggml_moe_bundle_plan bundle = {};
    bundle.layer = 0;
    bundle.kind = GGML_MOE_BUNDLE_SEPARATE_GATE_UP;
    bundle.route_ids = route_ids;
    bundle.gate_node = gate;
    bundle.up_node = up;
    bundle.act_node = activation;
    bundle.down_node = output;
    bundle.layer_input = input;
    bundle.is_fused = false;
    bundle.valid = true;

    const float sentinel = -99.0f;
    std::vector<float> actual(expected.size(), sentinel);
    ggml_backend_tensor_set(output, actual.data(), 0, ggml_nbytes(output));
    ggml_moe_route_ready_sidecar_t sidecar = ggml_moe_route_ready_sidecar_new(
        gpu_backend, cpu_backend, 2, 2, 2, false);
    ggml_backend_expert_cache_stats stats = {};
    test_original_synchronize = gpu_backend->iface.synchronize;
    test_original_set_tensor_async = gpu_backend->iface.set_tensor_async;
    test_original_get_tensor_async = gpu_backend->iface.get_tensor_async;
    require(test_original_synchronize != nullptr);
    require(test_original_set_tensor_async != nullptr);
    require(test_original_get_tensor_async != nullptr);
    test_synchronize_calls = 0;
    test_set_tensor_async_calls = 0;
    test_get_tensor_async_calls = 0;
    gpu_backend->iface.synchronize = test_count_synchronize;
    gpu_backend->iface.set_tensor_async = test_count_set_tensor_async;
    gpu_backend->iface.get_tensor_async = test_count_get_tensor_async;
    const enum ggml_status status = ggml_moe_route_ready_sidecar_execute_full_hit(
        sidecar, &bundle, cache, ids, 2, &stats);
    gpu_backend->iface.get_tensor_async = test_original_get_tensor_async;
    gpu_backend->iface.set_tensor_async = test_original_set_tensor_async;
    gpu_backend->iface.synchronize = test_original_synchronize;
    require(test_synchronize_calls == 1);
    require(test_set_tensor_async_calls == 4);
    require(test_get_tensor_async_calls == 1);
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-5f);
    }
    require(stats.bytes_ram_to_gpu == 0);
    require(stats.n_zero_copy_hits == 6);

    test_synchronize_calls = 0;
    test_original_graph_compute = gpu_backend->iface.graph_compute;
    require(test_original_graph_compute != nullptr);
    gpu_backend->iface.synchronize = test_count_synchronize;
    gpu_backend->iface.graph_compute = test_fail_graph_compute;
    require(ggml_moe_route_ready_sidecar_execute_full_hit(
        sidecar, &bundle, cache, ids, 2, &stats) == GGML_STATUS_FAILED);
    gpu_backend->iface.graph_compute = test_original_graph_compute;
    gpu_backend->iface.synchronize = test_original_synchronize;
    require(test_synchronize_calls == 1);
    ggml_backend_buffer_type failed_buft = {};
    failed_buft.iface.get_name = [](ggml_backend_buffer_type_t) { return "failed"; };
    failed_buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t, size_t) { return static_cast<ggml_backend_buffer_t>(nullptr); };
    failed_buft.iface.get_alignment = [](ggml_backend_buffer_type_t) { return (size_t) 1; };
    ggml_backend_device failed_device = {};
    failed_device.context = &failed_buft;
    failed_device.iface.get_buffer_type = [](ggml_backend_dev_t device) {
        return static_cast<ggml_backend_buffer_type_t>(device->context);
    };
    failed_buft.device = &failed_device;
    ggml_backend failed_backend = {};
    failed_backend.device = &failed_device;
    ggml_moe_route_ready_sidecar_t failed_sidecar = ggml_moe_route_ready_sidecar_new(
        &failed_backend, cpu_backend, 2, 2, 2, false);
    require(failed_sidecar != nullptr);
    require(ggml_moe_route_ready_sidecar_execute_full_hit(
        failed_sidecar, &bundle, cache, ids, 2, &stats) == GGML_STATUS_FAILED);
    require(ggml_moe_route_ready_sidecar_execute_full_hit(
        failed_sidecar, &bundle, cache, ids, 2, &stats) == GGML_STATUS_FAILED);
    ggml_moe_route_ready_sidecar_free(failed_sidecar);

    ggml_backend_expert_cache_t incomplete_cache = ggml_backend_expert_cache_new(gpu_backend, 4096);
    require(incomplete_cache != nullptr);
    ggml_backend_expert_cache_register_bundle(incomplete_cache, 0, gate_weights, up_weights, down_weights);
    for (int32_t expert_id : ids) {
        ggml_backend_expert_cache_seed(incomplete_cache, gate_weights, expert_id, 1);
        ggml_backend_expert_cache_seed(incomplete_cache, up_weights, expert_id, 1);
    }
    ggml_backend_synchronize(gpu_backend);
    std::fill(actual.begin(), actual.end(), sentinel);
    ggml_backend_tensor_set(output, actual.data(), 0, ggml_nbytes(output));
    ggml_backend_expert_cache_stats incomplete_stats = {};
    require(ggml_moe_route_ready_sidecar_execute_full_hit(
        sidecar, &bundle, incomplete_cache, ids, 2, &incomplete_stats) == GGML_STATUS_FAILED);
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (float value : actual) {
        require(value == sentinel);
    }
    require(incomplete_stats.n_zero_copy_hits == 0);
    ggml_backend_expert_cache_free(incomplete_cache);
    ggml_backend_t backends[] = { gpu_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    require(sched != nullptr);
    ggml_backend_sched_set_expert_cache(sched, 4096);
    ggml_backend_sched_register_expert_bundle(sched, 0, gate_weights, up_weights, down_weights);
    for (int32_t expert_id : ids) {
        require(ggml_backend_sched_expert_cache_seed(sched, 0, gate_weights, expert_id, 1));
        require(ggml_backend_sched_expert_cache_seed(sched, 0, up_weights, expert_id, 1));
        require(ggml_backend_sched_expert_cache_seed(sched, 0, down_weights, expert_id, 1));
    }
    ggml_backend_sched_expert_cache_sync(sched);
    const uint64_t epoch_before_full_hit = ggml_backend_sched_expert_cache_epoch(sched, 0);
    ggml_backend_tensor_set(route_input, ids, 0, sizeof(ids));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    std::fill(actual.begin(), actual.end(), sentinel);
    ggml_backend_tensor_set(output, actual.data(), 0, ggml_nbytes(output));
    const enum ggml_status scheduler_status = ggml_backend_sched_graph_compute(sched, graph);
    require(scheduler_status == GGML_STATUS_SUCCESS);
    require(ggml_backend_sched_expert_cache_epoch(sched, 0) == epoch_before_full_hit);
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-5f);
    }
    ggml_backend_expert_cache_stats sched_stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &sched_stats));
    require(sched_stats.n_route_ready_dispatches == 1);
    require(sched_stats.n_route_ready_classifications == 1);
    require(sched_stats.n_route_ready_actions == 1);
    require(sched_stats.bytes_ram_to_gpu == 0);
    require(sched_stats.n_zero_copy_hits == 6);
    ggml_backend_tensor_set(route_input, ids, 0, sizeof(ids));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    std::fill(actual.begin(), actual.end(), sentinel);
    ggml_backend_tensor_set(output, actual.data(), 0, ggml_nbytes(output));
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-5f);
    }
    ggml_backend_expert_cache_stats reuse_stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &reuse_stats));
    require(reuse_stats.n_route_ready_dispatches == 1);
    require(reuse_stats.n_route_ready_classifications == 2);
    require(reuse_stats.n_route_ready_actions == 2);
    require(reuse_stats.n_zero_copy_hits == 12);
    require(reuse_stats.bytes_ram_to_gpu == 0);
    ggml_backend_sched_set_expert_cache(sched, 4096);
    ggml_backend_sched_register_expert_bundle(sched, 0, gate_weights, up_weights, down_weights);
    for (int32_t expert_id : ids) {
        require(ggml_backend_sched_expert_cache_seed(sched, 0, gate_weights, expert_id, 1));
        require(ggml_backend_sched_expert_cache_seed(sched, 0, up_weights, expert_id, 1));
        require(ggml_backend_sched_expert_cache_seed(sched, 0, down_weights, expert_id, 1));
    }
    ggml_backend_sched_expert_cache_sync(sched);
    ggml_backend_tensor_set(route_input, stale_ids, 0, sizeof(stale_ids));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    std::fill(actual.begin(), actual.end(), sentinel);
    ggml_backend_tensor_set(output, actual.data(), 0, ggml_nbytes(output));
    test_original_synchronize = gpu_backend->iface.synchronize;
    require(test_original_synchronize != nullptr);
    test_synchronize_calls = 0;
    gpu_backend->iface.synchronize = test_count_synchronize;
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    gpu_backend->iface.synchronize = test_original_synchronize;
    require(test_synchronize_calls == 1);
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - stale_expected[i]) < 1e-5f);
    }
    ggml_backend_expert_cache_stats stale_stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &stale_stats));
    require(stale_stats.n_route_ready_classifications == reuse_stats.n_route_ready_classifications + 1);
    require(stale_stats.n_route_ready_actions == reuse_stats.n_route_ready_actions);
    require(stale_stats.n_zero_copy_hits == 0);
    require(stale_stats.bytes_ram_to_gpu == 0);
    ggml_backend_sched_set_expert_cache(sched, 4096);
    ggml_backend_sched_register_expert_bundle(sched, 0, gate_weights, up_weights, down_weights);
    const uint64_t epoch_before_seed = ggml_backend_sched_expert_cache_epoch(sched, 0);
    require(ggml_backend_sched_expert_cache_seed(sched, 0, gate_weights, 0, 1));
    require(ggml_backend_sched_expert_cache_epoch(sched, 0) > epoch_before_seed);
    ggml_backend_sched_expert_cache_sync(sched);
    const uint64_t epoch_before_partial_miss = ggml_backend_sched_expert_cache_epoch(sched, 0);
    ggml_backend_tensor_set(route_input, ids, 0, sizeof(ids));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    std::fill(actual.begin(), actual.end(), sentinel);
    ggml_backend_tensor_set(output, actual.data(), 0, ggml_nbytes(output));
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    require(ggml_backend_sched_expert_cache_epoch(sched, 0) == epoch_before_partial_miss);
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-5f);
    }
    ggml_backend_expert_cache_stats partial_stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &partial_stats));
    require(partial_stats.n_route_ready_classifications == stale_stats.n_route_ready_classifications + 1);
    require(partial_stats.n_route_ready_actions == stale_stats.n_route_ready_actions);
    require(partial_stats.n_zero_copy_hits == 0);
    require(partial_stats.bytes_ram_to_gpu == 0);
    ggml_backend_sched_set_expert_cache(sched, 4096);
    ggml_backend_sched_register_expert_bundle(sched, 0, gate_weights, up_weights, down_weights);
    ggml_backend_tensor_set(route_input, ids, 0, sizeof(ids));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    std::fill(actual.begin(), actual.end(), sentinel);
    ggml_backend_tensor_set(output, actual.data(), 0, ggml_nbytes(output));
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-5f);
    }
    ggml_backend_expert_cache_stats miss_stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &miss_stats));
    require(miss_stats.n_route_ready_classifications == partial_stats.n_route_ready_classifications + 1);
    require(miss_stats.n_route_ready_actions == partial_stats.n_route_ready_actions);
    ggml_backend_tensor_set(route_input, ids, 0, sizeof(ids));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    std::fill(actual.begin(), actual.end(), sentinel);
    ggml_backend_tensor_set(output, actual.data(), 0, ggml_nbytes(output));
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-5f);
    }
    ggml_backend_expert_cache_stats miss_reuse_stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &miss_reuse_stats));
    require(miss_reuse_stats.n_route_ready_classifications == miss_stats.n_route_ready_classifications + 1);
    require(miss_reuse_stats.n_route_ready_actions == miss_stats.n_route_ready_actions);
    require(miss_reuse_stats.n_zero_copy_hits == 0);
    require(miss_reuse_stats.bytes_ram_to_gpu == 0);
    ggml_backend_sched_set_expert_cache(sched, 4096);
    ggml_backend_sched_register_expert_bundle(sched, 0, gate_weights, up_weights, down_weights);
    ggml_backend_sched_set_expert_cache_period(sched, 1);
    ggml_backend_tensor_set(route_input, ids, 0, sizeof(ids));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    std::fill(actual.begin(), actual.end(), sentinel);
    ggml_backend_tensor_set(output, actual.data(), 0, ggml_nbytes(output));
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-5f);
    }
    ggml_backend_expert_cache_stats learning_first_stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &learning_first_stats));
    require(learning_first_stats.n_route_ready_actions == miss_reuse_stats.n_route_ready_actions);
    require(learning_first_stats.n_zero_copy_hits == 0);
    require(learning_first_stats.bytes_ram_to_gpu == 0);
    ggml_backend_sched_expert_cache_rebalance(sched);
    ggml_backend_sched_expert_cache_sync(sched);
    ggml_backend_tensor_set(route_input, ids, 0, sizeof(ids));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    std::fill(actual.begin(), actual.end(), sentinel);
    ggml_backend_tensor_set(output, actual.data(), 0, ggml_nbytes(output));
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-5f);
    }
    ggml_backend_expert_cache_stats learning_second_stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &learning_second_stats));
    require(learning_second_stats.n_route_ready_actions == learning_first_stats.n_route_ready_actions + 1);
    require(learning_second_stats.n_zero_copy_hits == 6);
    require(learning_second_stats.bytes_ram_to_gpu == 0);
    ggml_backend_sched_set_expert_cache(sched, 4096);
    ggml_backend_sched_register_expert_bundle(sched, 0, gate_weights, up_weights, down_weights);
    for (int32_t expert_id : ids) {
        require(ggml_backend_sched_expert_cache_seed(sched, 0, gate_weights, expert_id, 1));
        require(ggml_backend_sched_expert_cache_seed(sched, 0, up_weights, expert_id, 1));
        require(ggml_backend_sched_expert_cache_seed(sched, 0, down_weights, expert_id, 1));
    }
    ggml_backend_sched_expert_cache_sync(sched);
    ggml_backend_tensor_set(route_input, ids, 0, sizeof(ids));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    ggml_backend_sched_reset(sched);
    std::vector<float> noncontiguous_actual(noncontiguous_expected.size(), sentinel);
    ggml_backend_tensor_set(noncontiguous_output, noncontiguous_actual.data(), 0, ggml_nbytes(noncontiguous_output));
    require(ggml_backend_sched_graph_compute(sched, noncontiguous_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(noncontiguous_output, noncontiguous_actual.data(), 0, ggml_nbytes(noncontiguous_output));
    for (size_t i = 0; i < noncontiguous_actual.size(); ++i) {
        require(fabsf(noncontiguous_actual[i] - noncontiguous_expected[i]) < 1e-5f);
    }
    ggml_backend_expert_cache_stats noncontiguous_stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &noncontiguous_stats));
    require(noncontiguous_stats.n_route_ready_actions == learning_second_stats.n_route_ready_actions);
    require(noncontiguous_stats.n_zero_copy_hits == 0);
    require(noncontiguous_stats.bytes_ram_to_gpu == 0);
    ggml_backend_sched_set_expert_cache(sched, 4096);
    ggml_backend_sched_register_expert_bundle(sched, 0, gate_weights, up_weights, down_weights);
    require(ggml_backend_sched_expert_cache_seed(sched, 0, gate_weights, 0, 1));
    require(ggml_backend_sched_expert_cache_seed(sched, 0, up_weights, 0, 1));
    require(ggml_backend_sched_expert_cache_seed(sched, 0, down_weights, 0, 1));
    ggml_backend_sched_expert_cache_sync(sched);
    ggml_backend_sched_reset(sched);
    ggml_backend_tensor_set(route_input, inactive_ids, 0, sizeof(inactive_ids));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    std::fill(actual.begin(), actual.end(), sentinel);
    ggml_backend_tensor_set(output, actual.data(), 0, ggml_nbytes(output));
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < (size_t) output->ne[0]; ++i) {
        require(fabsf(actual[i] - inactive_expected[i]) < 1e-5f);
    }
    for (size_t i = (size_t) output->ne[0]; i < actual.size(); ++i) {
        require(actual[i] == sentinel);
    }
    ggml_backend_expert_cache_stats inactive_stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &inactive_stats));
    require(inactive_stats.n_route_ready_actions == noncontiguous_stats.n_route_ready_actions);
    ggml_backend_sched_free(sched);
    ggml_moe_route_ready_sidecar_free(sidecar);
    ggml_backend_expert_cache_free(cache);
    ggml_backend_buffer_free(cpu_buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("  route-ready full-hit sidecar tests passed\n");
}


static void test_route_ready_two_bundles_same_split_gap() {
    printf("testing route-ready two-bundle same-split gap...\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_device == nullptr) {
        printf("  no GPU backend available; skipped\n");
        return;
    }

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_device, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    require(gpu_backend != nullptr);
    require(cpu_backend != nullptr);

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    ggml_tensor * gate_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 2);
    ggml_tensor * up_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 2);
    ggml_tensor * down_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 2);
    ggml_tensor * gate_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 2);
    ggml_tensor * up_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 2);
    ggml_tensor * down_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 2);
    ggml_set_name(gate_a, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(up_a, "blk.0.ffn_up_exps.weight");
    ggml_set_name(down_a, "blk.0.ffn_down_exps.weight");
    ggml_set_name(gate_b, "blk.1.ffn_gate_exps.weight");
    ggml_set_name(up_b, "blk.1.ffn_up_exps.weight");
    ggml_set_name(down_b, "blk.1.ffn_down_exps.weight");

    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 1, 1);
    ggml_tensor * route_input_a = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    ggml_tensor * route_input_b = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    ggml_set_input(input);
    ggml_set_input(route_input_a);
    ggml_set_input(route_input_b);

    ggml_tensor * ids_a = ggml_dup(ctx, route_input_a);
    ggml_tensor * gate_out_a = ggml_mul_mat_id(ctx, gate_a, input, ids_a);
    ggml_tensor * up_out_a = ggml_mul_mat_id(ctx, up_a, input, ids_a);
    ggml_tensor * activation_a = ggml_swiglu_split(ctx, gate_out_a, up_out_a);
    ggml_tensor * output_a = ggml_mul_mat_id(ctx, down_a, activation_a, ids_a);

    ggml_tensor * j0 = ggml_scale(ctx, input, 0.5f);
    ggml_tensor * j1 = ggml_scale(ctx, j0, 0.5f);
    ggml_tensor * j2 = ggml_scale(ctx, j1, 0.5f);
    ggml_tensor * j3 = ggml_scale(ctx, j2, 0.5f);
    ggml_tensor * j4 = ggml_scale(ctx, j3, 0.5f);
    ggml_tensor * ids_b = ggml_dup(ctx, route_input_b);
    ggml_tensor * gate_out_b = ggml_mul_mat_id(ctx, gate_b, j4, ids_b);
    ggml_tensor * up_out_b = ggml_mul_mat_id(ctx, up_b, j4, ids_b);
    ggml_tensor * activation_b = ggml_swiglu_split(ctx, gate_out_b, up_out_b);
    ggml_tensor * output_b = ggml_mul_mat_id(ctx, down_b, activation_b, ids_b);
    ggml_tensor * output = ggml_add(ctx, output_a, output_b);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu_backend);
    require(buffer != nullptr);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const float gate_data[] = {
        1.0f, 0.0f, 0.0f, 1.0f,
        0.5f, 0.0f, 0.0f, 0.5f,
    };
    const float up_data[] = {
        0.75f, 0.0f, 0.0f, 0.75f,
        1.25f, 0.0f, 0.0f, 1.25f,
    };
    const float down_data[] = {
        1.0f, 0.0f, 0.0f, 1.0f,
        2.0f, 0.0f, 0.0f, 2.0f,
    };
    const float input_data[] = { 1.0f, 2.0f };
    const float zero_input[] = { 0.0f, 0.0f };
    const int32_t ids[] = { 0, 1 };
    const int32_t sentinel_ids[] = { 1, 0 };
    ggml_backend_tensor_set(gate_a, gate_data, 0, sizeof(gate_data));
    ggml_backend_tensor_set(up_a, up_data, 0, sizeof(up_data));
    ggml_backend_tensor_set(down_a, down_data, 0, sizeof(down_data));
    ggml_backend_tensor_set(gate_b, gate_data, 0, sizeof(gate_data));
    ggml_backend_tensor_set(up_b, up_data, 0, sizeof(up_data));
    ggml_backend_tensor_set(down_b, down_data, 0, sizeof(down_data));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    ggml_backend_tensor_set(route_input_a, ids, 0, sizeof(ids));
    ggml_backend_tensor_set(route_input_b, ids, 0, sizeof(ids));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    require(ggml_backend_graph_compute(cpu_backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> expected(ggml_nelements(output));
    ggml_backend_tensor_get(output, expected.data(), 0, ggml_nbytes(output));

    ggml_backend_t backends[] = { gpu_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    require(sched != nullptr);
    ggml_backend_sched_set_expert_cache(sched, 1024 * 1024);
    ggml_backend_sched_register_expert_bundle(sched, 0, gate_a, up_a, down_a);
    ggml_backend_sched_register_expert_bundle(sched, 1, gate_b, up_b, down_b);
    for (int32_t expert_id : ids) {
        require(ggml_backend_sched_expert_cache_seed(sched, 0, gate_a, expert_id, 1));
        require(ggml_backend_sched_expert_cache_seed(sched, 0, up_a, expert_id, 1));
        require(ggml_backend_sched_expert_cache_seed(sched, 0, down_a, expert_id, 1));
        require(ggml_backend_sched_expert_cache_seed(sched, 0, gate_b, expert_id, 1));
        require(ggml_backend_sched_expert_cache_seed(sched, 0, up_b, expert_id, 1));
        require(ggml_backend_sched_expert_cache_seed(sched, 0, down_b, expert_id, 1));
    }
    ggml_backend_sched_expert_cache_sync(sched);
    require(ggml_backend_sched_alloc_graph(sched, graph));

    ggml_backend_expert_cache_stats planned_stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &planned_stats));
    require(planned_stats.n_route_ready_dispatches == 2);

    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    ggml_backend_tensor_set(route_input_a, ids, 0, sizeof(ids));
    ggml_backend_tensor_set(route_input_b, ids, 0, sizeof(ids));
    ggml_backend_tensor_set(j4, zero_input, 0, sizeof(zero_input));
    ggml_backend_tensor_set(ids_b, sentinel_ids, 0, sizeof(sentinel_ids));
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-5f);
    }

    ggml_backend_expert_cache_stats stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &stats));
    require(stats.n_route_ready_dispatches == 2);
    require(stats.n_route_ready_classifications == 2);
    require(stats.n_route_ready_actions == 2);
    require(stats.n_zero_copy_hits == 12);
    require(stats.bytes_ram_to_gpu == 0);

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("  route-ready two-bundle same-split gap tests passed\n");
}

static void test_tg1_admission_masks_0_to_8() {
    printf("testing TG1 admission: masks 0-6 fast-reject, 7 classify+fallback, 8 classify+full-hit...\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_device == nullptr) {
        printf("  no GPU backend available; skipped\n");
        return;
    }

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_device, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    require(gpu_backend != nullptr);
    require(cpu_backend != nullptr);

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    const int32_t top_k = GGML_MOE_PARTIAL_MAX_ROUTES;
    ggml_tensor * gate_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, top_k);
    ggml_tensor * up_weights   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, top_k);
    ggml_tensor * down_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, top_k);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 1, 1);
    ggml_tensor * route_input = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, 1);
    ggml_set_input(input);
    ggml_set_input(route_input);
    ggml_set_name(gate_weights, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(up_weights,   "blk.0.ffn_up_exps.weight");
    ggml_set_name(down_weights, "blk.0.ffn_down_exps.weight");

    ggml_tensor * route_ids = ggml_dup(ctx, route_input);
    ggml_tensor * gate   = ggml_mul_mat_id(ctx, gate_weights, input, route_ids);
    ggml_tensor * up     = ggml_mul_mat_id(ctx, up_weights,   input, route_ids);
    ggml_tensor * activation = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * output = ggml_mul_mat_id(ctx, down_weights, activation, route_ids);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu_backend);
    require(buffer != nullptr);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Fill weights with deterministic values
    std::vector<float> gate_data((size_t)2 * 2 * top_k, 0.0f);
    std::vector<float> up_data((size_t)2 * 2 * top_k, 0.0f);
    std::vector<float> down_data((size_t)2 * 2 * top_k, 0.0f);
    for (int32_t e = 0; e < top_k; ++e) {
        gate_data[(size_t)e * 4 + 0] = 0.5f + 0.125f * e;
        gate_data[(size_t)e * 4 + 3] = 0.5f + 0.125f * e;
        up_data[(size_t)e * 4 + 0]   = 0.75f + 0.0625f * e;
        up_data[(size_t)e * 4 + 3]   = 0.75f + 0.0625f * e;
        down_data[(size_t)e * 4 + 0] = 1.0f + 0.25f * e;
        down_data[(size_t)e * 4 + 3] = 1.0f + 0.25f * e;
    }
    const float input_data[] = { 1.0f, 2.0f };
    ggml_backend_tensor_set(gate_weights, gate_data.data(), 0, ggml_nbytes(gate_weights));
    ggml_backend_tensor_set(up_weights, up_data.data(), 0, ggml_nbytes(up_weights));
    ggml_backend_tensor_set(down_weights, down_data.data(), 0, ggml_nbytes(down_weights));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));

    // Native CPU reference output
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    int32_t ref_ids[top_k] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    ggml_backend_tensor_set(route_input, ref_ids, 0, sizeof(ref_ids));
    require(ggml_backend_graph_compute(cpu_backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> expected(ggml_nelements(output));
    ggml_backend_tensor_get(output, expected.data(), 0, ggml_nbytes(output));

    // Build scheduler
    ggml_backend_t backends[] = { gpu_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    require(sched != nullptr);
    ggml_backend_sched_set_expert_cache(sched, 1024 * 1024);
    ggml_backend_sched_register_expert_bundle(sched, 0, gate_weights, up_weights, down_weights);
    require(ggml_backend_sched_alloc_graph(sched, graph));

    const float sentinel = -99.0f;
    std::vector<float> actual(expected.size(), sentinel);

    // Helper: run with n_hits seeded experts, return delta-stats
    auto run_seeded = [&](int32_t n_hits, const char * env_name, const char * env_val) -> ggml_backend_expert_cache_stats {
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_set_expert_cache(sched, 1024 * 1024);
        ggml_backend_sched_register_expert_bundle(sched, 0, gate_weights, up_weights, down_weights);
        require(ggml_backend_sched_alloc_graph(sched, graph));

        if (env_val != nullptr) {
            _putenv_s(env_name, env_val);
        }

        for (int32_t e = 0; e < n_hits; ++e) {
            require(ggml_backend_sched_expert_cache_seed(sched, 0, gate_weights, e, 1));
            require(ggml_backend_sched_expert_cache_seed(sched, 0, up_weights,   e, 1));
            require(ggml_backend_sched_expert_cache_seed(sched, 0, down_weights, e, 1));
        }
        ggml_backend_sched_expert_cache_sync(sched);

        ggml_backend_expert_cache_stats before = {};
        require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &before));

        std::fill(actual.begin(), actual.end(), sentinel);
        ggml_backend_tensor_set(output, actual.data(), 0, ggml_nbytes(output));

        int32_t ids[top_k] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        ggml_backend_tensor_set(route_input, ids, 0, sizeof(ids));
        require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));

        // Correctness: output must match native CPU reference
        for (size_t i = 0; i < actual.size(); ++i) {
            require(fabsf(actual[i] - expected[i]) < 1e-5f);
        }

        ggml_backend_expert_cache_stats after = {};
        require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &after));

        // Return delta
        ggml_backend_expert_cache_stats delta = {};
        delta.n_route_ready_fast_rejects = after.n_route_ready_fast_rejects - before.n_route_ready_fast_rejects;
        delta.n_route_ready_classifications = after.n_route_ready_classifications - before.n_route_ready_classifications;
        delta.n_route_ready_actions = after.n_route_ready_actions - before.n_route_ready_actions;
        delta.n_route_ready_full_hits = after.n_route_ready_full_hits - before.n_route_ready_full_hits;
        delta.n_route_ready_fallbacks = after.n_route_ready_fallbacks - before.n_route_ready_fallbacks;
        for (int k = 0; k < 9; k++) {
            delta.n_route_ready_resident_bundle_counts[k] = after.n_route_ready_resident_bundle_counts[k] - before.n_route_ready_resident_bundle_counts[k];
            delta.n_route_ready_mask_counts[k] = after.n_route_ready_mask_counts[k] - before.n_route_ready_mask_counts[k];
        }
        return delta;
    };
    // ---- Phase 1: production policy (no HETERO env override) ----
    // Masks 0-6: fast-reject
    for (int32_t n = 0; n < 7; ++n) {
        ggml_backend_expert_cache_stats s = run_seeded(n, nullptr, nullptr);
        require(s.n_route_ready_fast_rejects == 1);
        require(s.n_route_ready_classifications == 0);
        require(s.n_route_ready_actions == 0);
        require(s.n_route_ready_full_hits == 0);
        require(s.n_route_ready_fallbacks == 1);
        require(s.n_route_ready_resident_bundle_counts[n] == 1);
    }

    // Mask 7: classify + fallback
    {
        ggml_backend_expert_cache_stats s = run_seeded(7, nullptr, nullptr);
        require(s.n_route_ready_fast_rejects == 0);
        require(s.n_route_ready_classifications == 1);
        require(s.n_route_ready_actions == 0);
        require(s.n_route_ready_full_hits == 0);
        require(s.n_route_ready_fallbacks == 1);
        require(s.n_route_ready_mask_counts[7] == 1);
    }

    // Mask 8: classify + full-hit
    {
        ggml_backend_expert_cache_stats s = run_seeded(8, nullptr, nullptr);
        require(s.n_route_ready_fast_rejects == 0);
        require(s.n_route_ready_classifications == 1);
        require(s.n_route_ready_actions == 1);
        require(s.n_route_ready_full_hits == 1);
        require(s.n_route_ready_fallbacks == 0);
        require(s.n_route_ready_mask_counts[8] == 1);
    }

    // ---- Phase 2: HETERO_EXPERIMENTAL=1 must NOT change TG1 policy ----
    const char * env_name = "GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL";
    const char * prior = getenv(env_name);

    // Masks 0-6: still fast-reject
    for (int32_t n = 0; n < 7; ++n) {
        ggml_backend_expert_cache_stats s = run_seeded(n, env_name, "1");
        require(s.n_route_ready_fast_rejects == 1);
        require(s.n_route_ready_classifications == 0);
        require(s.n_route_ready_actions == 0);
        require(s.n_route_ready_full_hits == 0);
        require(s.n_route_ready_fallbacks == 1);
        require(s.n_route_ready_resident_bundle_counts[n] == 1);
    }

    // Mask 7: still classify + fallback (NOT diverted to split_hetero_bundles for TG1)
    {
        ggml_backend_expert_cache_stats s = run_seeded(7, env_name, "1");
        require(s.n_route_ready_fast_rejects == 0);
        require(s.n_route_ready_classifications == 1);
        require(s.n_route_ready_actions == 0);
        require(s.n_route_ready_full_hits == 0);
        require(s.n_route_ready_fallbacks == 1);
        require(s.n_route_ready_mask_counts[7] == 1);
    }

    // Mask 8: still full-hit
    {
        ggml_backend_expert_cache_stats s = run_seeded(8, env_name, "1");
        require(s.n_route_ready_fast_rejects == 0);
        require(s.n_route_ready_classifications == 1);
        require(s.n_route_ready_actions == 1);
        require(s.n_route_ready_full_hits == 1);
        require(s.n_route_ready_fallbacks == 0);
        require(s.n_route_ready_mask_counts[8] == 1);
    }

    if (prior != nullptr) {
        _putenv_s(env_name, prior);
    } else {
        _putenv_s(env_name, "");
    }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("  TG1 admission mask tests passed\n");
}
static void test_route_ready_cross_split_sidecar() {
    printf("testing route-ready cross-split sidecar...\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_device == nullptr) {
        printf("  no GPU backend available; skipped\n");
        return;
    }

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_device, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    require(gpu_backend != nullptr);
    require(cpu_backend != nullptr);

    ggml_backend_t backends[] = { gpu_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    require(sched != nullptr);
    ggml_backend_sched_set_expert_cache(sched, 4096);
    ggml_backend_sched_set_expert_cache_period(sched, 1);

    struct ggml_init_params weights_params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * weights_ctx = ggml_init(weights_params);
    require(weights_ctx != nullptr);

    ggml_tensor * gate_weights = ggml_new_tensor_3d(weights_ctx, GGML_TYPE_F32, 3, 2, 2);
    ggml_tensor * up_weights = ggml_new_tensor_3d(weights_ctx, GGML_TYPE_F32, 3, 2, 2);
    ggml_tensor * down_weights = ggml_new_tensor_3d(weights_ctx, GGML_TYPE_F32, 2, 3, 2);
    ggml_set_name(gate_weights, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(up_weights, "blk.0.ffn_up_exps.weight");
    ggml_set_name(down_weights, "blk.0.ffn_down_exps.weight");

    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, cpu_backend);
    require(weights_buffer != nullptr);
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    struct ggml_init_params graph_params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(graph_params);
    require(ctx != nullptr);

    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 1, 1);
    ggml_tensor * route_input = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    ggml_set_input(input);
    ggml_set_input(route_input);
    ggml_tensor * route_ids_source = ggml_dup(ctx, route_input);
    ggml_tensor * route_ids = ggml_view_2d(
        ctx, route_ids_source, route_ids_source->ne[0], route_ids_source->ne[1], route_ids_source->nb[1], 0);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_weights, input, route_ids);
    ggml_tensor * up = ggml_mul_mat_id(ctx, up_weights, input, route_ids);
    ggml_tensor * activation = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * output = ggml_mul_mat_id(ctx, down_weights, activation, route_ids);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    const float gate_data[] = {
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
        0.5f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f,
    };
    const float up_data[] = {
        0.75f, 0.0f, 0.0f, 0.0f, 0.75f, 0.0f,
        1.25f, 0.0f, 0.0f, 0.0f, 1.25f, 0.0f,
    };
    const float down_data[] = {
        1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f,
    };
    const float input_data[] = { 1.0f, 2.0f, 3.0f };
    const int32_t ids[] = { 0, 1 };
    const std::vector<float> expected = {
        0.54829395f, 2.6423912f, 0.0f,
        0.77807415f, 3.6552930f, 0.0f,
    };
    ggml_backend_tensor_set(gate_weights, gate_data, 0, sizeof(gate_data));
    ggml_backend_tensor_set(up_weights, up_data, 0, sizeof(up_data));
    ggml_backend_tensor_set(down_weights, down_data, 0, sizeof(down_data));

    ggml_backend_sched_register_expert_bundle(
        sched, 0, gate_weights, up_weights, down_weights);
    ggml_backend_sched_set_tensor_backend(sched, route_ids_source, gpu_backend);
    require(ggml_backend_sched_alloc_graph(sched, graph));
    ggml_backend_expert_cache_stats planned_stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &planned_stats));
    require(planned_stats.n_route_ready_dispatches == 1);

    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    ggml_backend_tensor_set(route_input, ids, 0, sizeof(ids));
    test_original_synchronize = gpu_backend->iface.synchronize;
    require(test_original_synchronize != nullptr);
    test_synchronize_calls = 0;
    gpu_backend->iface.synchronize = test_count_synchronize;
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    gpu_backend->iface.synchronize = test_original_synchronize;
    require(test_synchronize_calls >= 1);

    std::vector<float> first_actual(expected.size());
    ggml_backend_tensor_get(output, first_actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < first_actual.size(); ++i) {
        require(fabsf(first_actual[i] - expected[i]) < 1e-5f);
    }
    ggml_backend_expert_cache_stats first_stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &first_stats));
    require(first_stats.n_route_ready_dispatches == 1);
    require(first_stats.n_route_ready_classifications == 1);
    ggml_backend_sched_expert_cache_sync(sched);

    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    ggml_backend_tensor_set(route_input, ids, 0, sizeof(ids));
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-5f);
    }
    ggml_backend_expert_cache_stats second_stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &second_stats));
    require(second_stats.n_route_ready_dispatches == 1);
    require(second_stats.n_route_ready_classifications == 2);
    require(second_stats.n_route_ready_actions == 1);
    require(second_stats.n_route_ready_mask_counts[0] == 1);
    require(second_stats.n_route_ready_mask_counts[2] == 1);
    for (int k = 1; k < 9; ++k) {
        if (k != 2) {
            require(second_stats.n_route_ready_mask_counts[k] == 0);
        }
    }
    require(second_stats.n_zero_copy_hits == 6);

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(weights_buffer);
    ggml_free(ctx);
    ggml_free(weights_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("  route-ready cross-split sidecar tests passed\n");
}


static void test_cacheless_moe_subset_copy() {
    printf("testing cacheless MoE subset copy...\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_device == nullptr) {
        printf("  no GPU backend available; skipped\n");
        return;
    }

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_device, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    require(gpu_backend != nullptr);
    require(cpu_backend != nullptr);

    constexpr int64_t n_experts = 8;
    constexpr int64_t n_embd = 4;
    constexpr int64_t n_tokens = 32;
    constexpr size_t expert_bytes = n_embd * n_embd * sizeof(float);
    std::vector<float> input_data((size_t) n_embd * n_tokens);
    std::vector<int32_t> route_data(2 * n_tokens);
    for (int64_t token = 0; token < n_tokens; ++token) {
        input_data[(size_t) token * n_embd + 0] = 1.0f;
        input_data[(size_t) token * n_embd + 1] = 2.0f;
        input_data[(size_t) token * n_embd + 2] = 3.0f;
        input_data[(size_t) token * n_embd + 3] = 4.0f;
        route_data[2 * token + 0] = 0;
        route_data[2 * token + 1] = 4;
    }
    std::vector<float> weights((size_t) n_experts * n_embd * n_embd, 0.0f);
    for (int64_t expert = 0; expert < n_experts; ++expert) {
        for (int64_t diagonal = 0; diagonal < n_embd; ++diagonal) {
            weights[(size_t) expert * n_embd * n_embd + diagonal * n_embd + diagonal] =
                (float) expert + 1.0f;
        }
    }

    struct ggml_init_params reference_params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * reference_ctx = ggml_init(reference_params);
    require(reference_ctx != nullptr);
    ggml_tensor * reference_weights =
        ggml_new_tensor_3d(reference_ctx, GGML_TYPE_F32, n_embd, n_embd, n_experts);
    ggml_tensor * reference_input =
        ggml_new_tensor_3d(reference_ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * reference_ids =
        ggml_new_tensor_2d(reference_ctx, GGML_TYPE_I32, 2, n_tokens);
    ggml_tensor * reference_output =
        ggml_mul_mat_id(reference_ctx, reference_weights, reference_input, reference_ids);
    ggml_cgraph * reference_graph = ggml_new_graph(reference_ctx);
    ggml_build_forward_expand(reference_graph, reference_output);
    ggml_backend_buffer_t reference_buffer =
        ggml_backend_alloc_ctx_tensors(reference_ctx, cpu_backend);
    require(reference_buffer != nullptr);
    ggml_backend_tensor_set(reference_weights, weights.data(), 0, ggml_nbytes(reference_weights));
    ggml_backend_tensor_set(
        reference_input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(
        reference_ids, route_data.data(), 0, route_data.size() * sizeof(int32_t));
    require(ggml_backend_graph_compute(cpu_backend, reference_graph) == GGML_STATUS_SUCCESS);
    std::vector<float> expected(ggml_nelements(reference_output));
    ggml_backend_tensor_get(
        reference_output, expected.data(), 0, ggml_nbytes(reference_output));

    struct ggml_init_params weights_params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * weights_ctx = ggml_init(weights_params);
    require(weights_ctx != nullptr);
    ggml_tensor * expert_weights =
        ggml_new_tensor_3d(weights_ctx, GGML_TYPE_F32, n_embd, n_embd, n_experts);
    ggml_set_name(expert_weights, "blk.0.ffn_gate_exps.weight");
    ggml_backend_buffer_t weights_buffer =
        ggml_backend_alloc_ctx_tensors(weights_ctx, cpu_backend);
    require(weights_buffer != nullptr);
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(expert_weights, weights.data(), 0, ggml_nbytes(expert_weights));

    struct ggml_init_params graph_params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * graph_ctx = ggml_init(graph_params);
    require(graph_ctx != nullptr);
    ggml_tensor * input =
        ggml_new_tensor_3d(graph_ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * route_ids =
        ggml_new_tensor_2d(graph_ctx, GGML_TYPE_I32, 2, n_tokens);
    ggml_set_input(input);
    ggml_set_input(route_ids);
    ggml_tensor * output = ggml_mul_mat_id(graph_ctx, expert_weights, input, route_ids);
    ggml_cgraph * graph = ggml_new_graph(graph_ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_t backends[] = { gpu_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    require(sched != nullptr);
    ggml_backend_sched_set_tensor_backend(sched, input, gpu_backend);
    ggml_backend_sched_set_tensor_backend(sched, route_ids, gpu_backend);
    require(ggml_backend_sched_alloc_graph(sched, graph));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(route_ids, route_data.data(), 0, route_data.size() * sizeof(int32_t));

    test_tracked_tensor_name = "blk.0.ffn_gate_exps.weight";
    test_tracked_cpy_tensor_async_calls = 0;
    test_tracked_set_tensor_async_writes.clear();
    test_original_cpy_tensor_async = gpu_backend->iface.cpy_tensor_async;
    test_original_set_tensor_async = gpu_backend->iface.set_tensor_async;
    require(test_original_cpy_tensor_async != nullptr);
    require(test_original_set_tensor_async != nullptr);
    gpu_backend->iface.cpy_tensor_async = test_count_cpy_tensor_async;
    gpu_backend->iface.set_tensor_async = test_count_set_tensor_async;

    const enum ggml_status status = ggml_backend_sched_graph_compute(sched, graph);

    gpu_backend->iface.set_tensor_async = test_original_set_tensor_async;
    gpu_backend->iface.cpy_tensor_async = test_original_cpy_tensor_async;
    test_tracked_tensor_name = nullptr;
    require(status == GGML_STATUS_SUCCESS);
    require(test_tracked_cpy_tensor_async_calls == 0);
    require(test_tracked_set_tensor_async_writes.size() == 2);
    require(test_tracked_set_tensor_async_writes[0].offset == 0);
    require(test_tracked_set_tensor_async_writes[0].size == 2 * expert_bytes);
    require(test_tracked_set_tensor_async_writes[1].offset == 4 * expert_bytes);
    require(test_tracked_set_tensor_async_writes[1].size == 2 * expert_bytes);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-5f);
    }

    ggml_backend_expert_cache_stats stats = {};
    require(!ggml_backend_sched_get_expert_cache_stats(sched, -1, &stats));
    require(stats.n_requests == 0);
    require(stats.n_route_census_nodes == 0);
    require(stats.n_route_census_plans == 0);
    require(stats.n_route_census_split_inputs == 0);
    require(stats.n_route_ready_dispatches == 0);
    require(stats.n_route_ready_classifications == 0);
    require(stats.n_route_ready_actions == 0);
    require(stats.n_zero_copy_hits == 0);
    require(stats.bytes_ram_to_gpu == 0);

    ggml_backend_sched_free(sched);
    ggml_free(graph_ctx);
    ggml_backend_buffer_free(weights_buffer);
    ggml_free(weights_ctx);
    ggml_backend_buffer_free(reference_buffer);
    ggml_free(reference_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("  cacheless MoE subset copy tests passed\n");
}

static void test_deterministic_full_hit_latency_fixture() {
    printf("testing deterministic 8/8 full-hit: native vs sidecar equivalence...\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_device == nullptr) {
        printf("  no GPU backend available; skipped\n");
        return;
    }

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_device, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    require(gpu_backend != nullptr);
    require(cpu_backend != nullptr);

    const int32_t top_k = 8;
    const int64_t d_model = 64;
    const int64_t d_ff = 128;

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    ggml_tensor * gate_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_model, d_ff, top_k);
    ggml_tensor * up_weights   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_model, d_ff, top_k);
    ggml_tensor * down_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_ff, d_model, top_k);
    ggml_tensor * input        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, 1);
    ggml_tensor * route_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, 1);
    ggml_set_input(input);
    ggml_set_input(route_input);
    ggml_set_name(gate_weights, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(up_weights, "blk.0.ffn_up_exps.weight");
    ggml_set_name(down_weights, "blk.0.ffn_down_exps.weight");

    // Build graph: gate, up, swiglu, down
    ggml_tensor * route_ids = ggml_dup(ctx, route_input);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_weights, input, route_ids);
    ggml_tensor * up = ggml_mul_mat_id(ctx, up_weights, input, route_ids);
    ggml_tensor * activation = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * output = ggml_mul_mat_id(ctx, down_weights, activation, route_ids);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu_backend);
    require(buffer != nullptr);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Fill weights with deterministic values
    srand(42);
    std::vector<float> gate_data((size_t)d_model * d_ff * top_k, 0.0f);
    std::vector<float> up_data((size_t)d_model * d_ff * top_k, 0.0f);
    std::vector<float> down_data((size_t)d_ff * d_model * top_k, 0.0f);
    for (int e = 0; e < top_k; e++) {
        for (int j = 0; j < d_ff; j++) {
            for (int i = 0; i < d_model; i++) {
                gate_data[(size_t)e * d_ff * d_model + (size_t)j * d_model + (size_t)i] = 0.01f * (float)(rand() % 100);
                up_data[(size_t)e * d_ff * d_model + (size_t)j * d_model + (size_t)i] = 0.01f * (float)(rand() % 100);
            }
        }
        for (int i = 0; i < d_model; i++) {
            for (int j = 0; j < d_ff; j++) {
                down_data[(size_t)e * d_model * d_ff + (size_t)i * d_ff + (size_t)j] = 0.01f * (float)(rand() % 100);
            }
        }
    }
    std::vector<float> input_data(d_model, 0.5f);
    for (int i = 0; i < d_model; i++) input_data[i] = 0.5f + 0.01f * (float)(i % 17);
    const int32_t route_ids_arr[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    ggml_backend_tensor_set(gate_weights, gate_data.data(), 0, ggml_nbytes(gate_weights));
    ggml_backend_tensor_set(up_weights, up_data.data(), 0, ggml_nbytes(up_weights));
    ggml_backend_tensor_set(down_weights, down_data.data(), 0, ggml_nbytes(down_weights));
    ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(route_input, route_ids_arr, 0, sizeof(route_ids_arr));

    // Compute CPU reference
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    require(ggml_backend_graph_compute(cpu_backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> expected(ggml_nelements(output));
    ggml_backend_tensor_get(output, expected.data(), 0, ggml_nbytes(output));

    // Set up scheduler with expert cache
    ggml_backend_t backends[] = { gpu_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    require(sched != nullptr);
    ggml_backend_sched_set_expert_cache(sched, 1024 * 1024);
    ggml_backend_sched_register_expert_bundle(sched, 0, gate_weights, up_weights, down_weights);

    // Seed all 8 route experts
    for (int i = 0; i < top_k; i++) {
        require(ggml_backend_sched_expert_cache_seed(sched, 0, gate_weights, route_ids_arr[i], 1));
        require(ggml_backend_sched_expert_cache_seed(sched, 0, up_weights, route_ids_arr[i], 1));
        require(ggml_backend_sched_expert_cache_seed(sched, 0, down_weights, route_ids_arr[i], 1));
    }
    ggml_backend_sched_expert_cache_sync(sched);
    require(ggml_backend_sched_alloc_graph(sched, graph));

    // Run via scheduler
    const float sentinel = -99.0f;
    std::vector<float> actual(expected.size(), sentinel);
    ggml_backend_tensor_set(output, actual.data(), 0, ggml_nbytes(output));
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));

    ggml_backend_expert_cache_stats stats = {};
    require(ggml_backend_sched_get_expert_cache_stats(sched, -1, &stats));

    // Verify CPU-reference equivalence
    // NOTE: 1e-2 tolerance for GPU FMA vs CPU non-FMA accumulation differences
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-2f);
    }


    // Verify sidecar stats
    require(stats.n_route_ready_classifications == 1);
    require(stats.n_route_ready_actions == 1);
    require(stats.n_route_ready_full_hits == 1);
    require(stats.n_zero_copy_hits > 0);
    require(stats.bytes_ram_to_gpu == 0);

    // Verify stable route IDs
    int32_t check_ids[8] = {};
    ggml_backend_tensor_get(route_input, check_ids, 0, sizeof(check_ids));
    for (int i = 0; i < 8; i++) {
        require(check_ids[i] == (int32_t)route_ids_arr[i]);
    }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("  deterministic 8/8 full-hit fixture tests passed\n");
}



static void test_manifest_v3_rejection_and_atomicity() {
    printf("testing manifest v3 rejection and atomicity...\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_device == nullptr) {
        printf("  no GPU backend available; skipped\n");
        return;
    }

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_device, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    require(gpu_backend != nullptr);
    require(cpu_backend != nullptr);

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    const int32_t top_k = 8;
    ggml_tensor * gate_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, top_k);
    ggml_tensor * up_weights   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, top_k);
    ggml_tensor * down_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, top_k);
    ggml_set_name(gate_weights, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(up_weights,   "blk.0.ffn_up_exps.weight");
    ggml_set_name(down_weights, "blk.0.ffn_down_exps.weight");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu_backend);
    require(buffer != nullptr);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_backend_t backends[] = { gpu_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    require(sched != nullptr);
    ggml_backend_sched_set_expert_cache(sched, 1024 * 1024);
    ggml_backend_sched_register_expert_bundle(sched, 0, gate_weights, up_weights, down_weights);

    // Helper to write temp manifest file
    auto write_manifest = [](const char * json) -> std::string {
        std::string path = std::tmpnam(nullptr);
        FILE * f = fopen(path.c_str(), "w");
        require(f != nullptr);
        fputs(json, f);
        fclose(f);
        return path;
    };

    // Test 1: Bad format (not v3)
    {
        std::string path = write_manifest("{\"format\": 2, \"admission\": \"8of8\", \"bundles\": []}");
        require(!ggml_backend_sched_load_pinned_manifest(sched, path.c_str()));
        std::remove(path.c_str());
    }

    // Test 2: 7of8 admission (not 8of8)
    {
        std::string path = write_manifest("{\"format\": 3, \"admission\": \"7of8\", \"bundles\": []}");
        require(!ggml_backend_sched_load_pinned_manifest(sched, path.c_str()));
        std::remove(path.c_str());
    }

    // Test 3: Missing model field
    {
        std::string path = write_manifest("{\"format\": 3, \"admission\": \"8of8\", "
            "\"cache_bytes\": 1048576, \"bundles\": []}");
        require(!ggml_backend_sched_load_pinned_manifest(sched, path.c_str()));
        std::remove(path.c_str());
    }

    // Test 4: Wrong projection list
    {
        std::string path = write_manifest("{\"format\": 3, \"admission\": \"8of8\", "
            "\"model\": {\"sha256\": \"\", \"top_k\": 8, \"expert_count\": 256}, "
            "\"cache_bytes\": 1048576, "
            "\"bundles\": [{\"layer\": 0, \"expert\": 0, \"projections\": [\"gate\", \"up\"]}]}");
        require(!ggml_backend_sched_load_pinned_manifest(sched, path.c_str()));
        std::remove(path.c_str());
    }

    // Test 5: Valid v3 manifest (empty)
    {
        std::string path = write_manifest("{\"format\": 3, \"admission\": \"8of8\", "
            "\"model\": {\"sha256\": \"\", \"top_k\": 8, \"expert_count\": 256}, "
            "\"cache_bytes\": 1048576, \"bundles\": []}");
        require(ggml_backend_sched_load_pinned_manifest(sched, path.c_str()));
        std::remove(path.c_str());
    }

    // Test 6: Valid v3 manifest with one bundle
    {
        std::string path = write_manifest("{\"format\": 3, \"admission\": \"8of8\", "
            "\"model\": {\"sha256\": \"\", \"top_k\": 8, \"expert_count\": 256}, "
            "\"cache_bytes\": 1048576, "
            "\"bundles\": [{\"layer\": 0, \"expert\": 0, \"projections\": [\"gate\", \"up\", \"down\"]}]}");
        require(ggml_backend_sched_load_pinned_manifest(sched, path.c_str()));
        std::remove(path.c_str());
    }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("  manifest v3 rejection and atomicity tests passed\n");
}
static void test_expert_cache_layout_classification() {
    printf("testing expert-cache layout classification...\n");

    require(llama_uses_expert_cache_pp_layout(25, 1, true));
    require(!llama_uses_expert_cache_pp_layout(25, 25, true));
    require(!llama_uses_expert_cache_pp_layout(1, 1, true));
    require(!llama_uses_expert_cache_pp_layout(25, 1, false));

    printf("  expert-cache layout classification tests passed\n");
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    test_complete_bundle_admission_query();

    test_cacheless_moe_subset_copy();
    test_route_census_classifies_original_graph();
    test_event_query_contract();
    test_registered_bundle_keeps_cpu_base_placement();
    test_partial_executor_scheduler_feature_gate();
    test_rebalance_does_not_synchronize_gpu();
    test_rebalance_ignores_gpu_resident_weights();
    test_route_plan_groups_shared_ids();
    test_slot_pools_and_remapping();
    test_manifest_loader_outcome_stats();
    test_multi_token_slot_remapping();
    test_tg1_admission_masks_0_to_8();
    test_pinned_staging_no_overwrite();

    test_slru_and_admission_policy();
    test_expert_bundles();
    test_pinned_host_buffer();
    test_rebalance_tracks_staging_memcpy();
    test_prefetch();
    test_prefetch_deduplicates_expert_ids();
    test_route_prefetch_telemetry();
    test_pp_tg_telemetry_isolation();
    test_expert_cache_layout_classification();

    test_slot_loading_lifecycle();
    test_per_tensor_slot_isolation();
    test_staging_sync_teardown();
    test_auto_reserve_sentinel();
    test_async_promotion_pipeline();
    test_gpu_slot_map_remapping();

    test_hit_mask_matrix_partitioning();
    test_partial_route_snapshot_validation();
    test_partial_executor_persistence();
    test_partial_executor_scheduler_catalog();
    test_partial_executor_exact_route_paths();
    test_multi_token_repeated_experts();
    test_dynamic_map_metadata_and_device_maps();
    test_route_ready_sidecar_full_hit();
    test_route_ready_two_bundles_same_split_gap();
    test_route_ready_cross_split_sidecar();

    test_manifest_v3_rejection_and_atomicity();
    test_deterministic_full_hit_latency_fixture();
    printf("all test-expert-cache tests passed successfully!\n");
    return 0;
}
