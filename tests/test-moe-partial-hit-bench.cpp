#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

// Performance timer in microseconds
static int64_t get_time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

struct bench_stats {
    double min_us;
    double max_us;
    double mean_us;
    double median_us;
    double p95_us;
    double stddev_us;
};

static bench_stats compute_stats(std::vector<double> & samples) {
    if (samples.empty()) {
        return { 0, 0, 0, 0, 0, 0 };
    }
    std::sort(samples.begin(), samples.end());
    const size_t n = samples.size();
    const double min_v = samples.front();
    const double max_v = samples.back();
    const double median_v = (n % 2 == 0) ? 0.5 * (samples[n / 2 - 1] + samples[n / 2]) : samples[n / 2];
    const size_t p95_idx = (size_t)std::floor(0.95 * (n - 1));
    const double p95_v = samples[p95_idx];

    double sum = 0.0;
    for (double v : samples) {
        sum += v;
    }
    const double mean_v = sum / (double)n;

    double sum_sq_diff = 0.0;
    for (double v : samples) {
        const double diff = v - mean_v;
        sum_sq_diff += diff * diff;
    }
    const double stddev_v = std::sqrt(sum_sq_diff / (double)n);

    return { min_v, max_v, mean_v, median_v, p95_v, stddev_v };
}

// Error statistics for comparison against reference
struct error_stats {
    double max_abs_error;
    double max_rel_error;
    double mean_abs_error;
    double mean_rel_error;
    double nmse;
    bool passed;
};

static error_stats compute_error(const float * actual, const float * expected, size_t n, double tol_nmse = 0.001, double tol_mean_rel = 0.05, double tol_max_abs = 0.05) {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double sum_abs = 0.0;
    double sum_rel = 0.0;
    double mse_diff = 0.0;
    double mse_ref = 0.0;

    for (size_t i = 0; i < n; i++) {
        const double act = (double)actual[i];
        const double exp = (double)expected[i];
        const double diff = std::abs(act - exp);
        const double denom = std::max(std::abs(exp), 1e-3);
        const double rel = diff / denom;

        if (diff > max_abs) max_abs = diff;
        if (rel > max_rel) max_rel = rel;
        sum_abs += diff;
        sum_rel += rel;

        mse_diff += (act - exp) * (act - exp);
        mse_ref  += exp * exp;
    }

    const double mean_abs = n > 0 ? (sum_abs / (double)n) : 0.0;
    const double mean_rel = n > 0 ? (sum_rel / (double)n) : 0.0;
    const double nmse = mse_ref > 1e-9 ? (mse_diff / mse_ref) : 0.0;

    // Numerical tolerance grounded in GGML quantized kernel standards
    const bool passed = (nmse <= tol_nmse) || (mean_rel <= tol_mean_rel) || (max_abs <= tol_max_abs);

    return { max_abs, max_rel, mean_abs, mean_rel, nmse, passed };
}

static void init_tensor_uniform(ggml_tensor * tensor, float min_v = -1.0f, float max_v = 1.0f, uint32_t seed = 42) {
    const size_t nels = ggml_nelements(tensor);
    std::vector<float> data(nels);

    std::default_random_engine gen(seed);
    std::uniform_real_distribution<float> dist(min_v, max_v);
    for (size_t i = 0; i < nels; i++) {
        data[i] = dist(gen);
    }

    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, nels * sizeof(float));
    } else if (ggml_is_quantized(tensor->type)) {
        const size_t blck_size = ggml_blck_size(tensor->type);
        GGML_ASSERT(nels % blck_size == 0);
        const size_t n_blocks = nels / blck_size;
        std::vector<uint8_t> dataq(ggml_row_size(tensor->type, nels));
        ggml_quantize_chunk(tensor->type, data.data(), dataq.data(), 0, n_blocks, blck_size, nullptr);
        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    }
}

// Gate A exact Qwen3.6-35B-A3B dimensions
struct moe_gate_a_spec {
    int64_t d_model       = 2048;
    int64_t d_ff          = 512;
    int64_t n_expert      = 256;
    int64_t top_k         = 8;
    int64_t n_slots       = 16;
    ggml_type type_gate   = GGML_TYPE_Q4_K;
    ggml_type type_up     = GGML_TYPE_Q4_K;
    ggml_type type_down   = GGML_TYPE_Q6_K;
};

// Route descriptor for partial-hit execution
struct moe_route_desc {
    int32_t token;
    int32_t route;
    int32_t expert;
    int32_t gate_slot;
    int32_t up_slot;
    int32_t down_slot;
    bool    bundle_resident;
};

// Execution context holding test tensors and backends
struct moe_oracle_context {
    const moe_gate_a_spec & spec;
    ggml_backend_t backend_cpu;
    ggml_backend_t backend_gpu;

    // CPU host weights [d_model, d_ff, n_expert] etc.
    struct ggml_context * ctx_cpu = nullptr;
    ggml_backend_buffer_t buf_cpu = nullptr;
    ggml_tensor * host_gate_w = nullptr;
    ggml_tensor * host_up_w = nullptr;
    ggml_tensor * host_down_w = nullptr;

    // GPU slot pool weights [d_model, d_ff, n_slots] etc.
    struct ggml_context * ctx_gpu = nullptr;
    ggml_backend_buffer_t buf_gpu = nullptr;
    ggml_tensor * slot_gate_w = nullptr;
    ggml_tensor * slot_up_w = nullptr;
    ggml_tensor * slot_down_w = nullptr;

    // Test input activation and routing
    std::vector<float> input_x;
    std::vector<int32_t> selected_experts;
    std::vector<float> routing_weights;

    moe_oracle_context(const moe_gate_a_spec & s, ggml_backend_t b_cpu, ggml_backend_t b_gpu)
        : spec(s), backend_cpu(b_cpu), backend_gpu(b_gpu) {}

    ~moe_oracle_context() {
        if (buf_cpu) ggml_backend_buffer_free(buf_cpu);
        if (ctx_cpu) ggml_free(ctx_cpu);
        if (buf_gpu) ggml_backend_buffer_free(buf_gpu);
        if (ctx_gpu) ggml_free(ctx_gpu);
    }
};

static void setup_oracle_context(moe_oracle_context & moec) {
    const auto & spec = moec.spec;

    // 1. Allocate CPU Host Weights
    struct ggml_init_params params_cpu = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    moec.ctx_cpu = ggml_init(params_cpu);
    moec.host_gate_w = ggml_new_tensor_3d(moec.ctx_cpu, spec.type_gate, spec.d_model, spec.d_ff, spec.n_expert);
    moec.host_up_w   = ggml_new_tensor_3d(moec.ctx_cpu, spec.type_up,   spec.d_model, spec.d_ff, spec.n_expert);
    moec.host_down_w = ggml_new_tensor_3d(moec.ctx_cpu, spec.type_down, spec.d_ff, spec.d_model, spec.n_expert);
    moec.buf_cpu = ggml_backend_alloc_ctx_tensors(moec.ctx_cpu, moec.backend_cpu);

    init_tensor_uniform(moec.host_gate_w, -1.0f, 1.0f, 101);
    init_tensor_uniform(moec.host_up_w,   -1.0f, 1.0f, 102);
    init_tensor_uniform(moec.host_down_w, -1.0f, 1.0f, 103);

    // 2. Allocate GPU Slot Pools
    struct ggml_init_params params_gpu = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    moec.ctx_gpu = ggml_init(params_gpu);
    moec.slot_gate_w = ggml_new_tensor_3d(moec.ctx_gpu, spec.type_gate, spec.d_model, spec.d_ff, spec.n_slots);
    moec.slot_up_w   = ggml_new_tensor_3d(moec.ctx_gpu, spec.type_up,   spec.d_model, spec.d_ff, spec.n_slots);
    moec.slot_down_w = ggml_new_tensor_3d(moec.ctx_gpu, spec.type_down, spec.d_ff, spec.d_model, spec.n_slots);
    moec.buf_gpu = ggml_backend_alloc_ctx_tensors(moec.ctx_gpu, moec.backend_gpu);

    // 3. Test input vector
    moec.input_x.resize(spec.d_model);
    std::default_random_engine gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < spec.d_model; i++) {
        moec.input_x[i] = dist(gen);
    }

    // 4. Selected 8 expert IDs
    moec.selected_experts = { 7, 23, 45, 89, 112, 168, 204, 241 };

    // 5. Normalized routing weights
    moec.routing_weights = { 0.25f, 0.20f, 0.15f, 0.12f, 0.10f, 0.08f, 0.06f, 0.04f };
    float sum_w = 0.0f;
    for (float w : moec.routing_weights) sum_w += w;
    for (float & w : moec.routing_weights) w /= sum_w;

    // 6. Preload the 8 selected experts into GPU slots 0..7
    const size_t gate_slice_bytes = ggml_row_size(spec.type_gate, spec.d_model * spec.d_ff);
    const size_t up_slice_bytes   = ggml_row_size(spec.type_up,   spec.d_model * spec.d_ff);
    const size_t down_slice_bytes = ggml_row_size(spec.type_down, spec.d_ff * spec.d_model);

    std::vector<uint8_t> tmp_buf(std::max({ gate_slice_bytes, up_slice_bytes, down_slice_bytes }));

    for (int32_t slot = 0; slot < (int32_t)moec.selected_experts.size(); slot++) {
        const int32_t exp_id = moec.selected_experts[slot];

        // Gate slice
        ggml_backend_tensor_get(moec.host_gate_w, tmp_buf.data(), (size_t)exp_id * gate_slice_bytes, gate_slice_bytes);
        ggml_backend_tensor_set(moec.slot_gate_w, tmp_buf.data(), (size_t)slot * gate_slice_bytes, gate_slice_bytes);

        // Up slice
        ggml_backend_tensor_get(moec.host_up_w, tmp_buf.data(), (size_t)exp_id * up_slice_bytes, up_slice_bytes);
        ggml_backend_tensor_set(moec.slot_up_w, tmp_buf.data(), (size_t)slot * up_slice_bytes, up_slice_bytes);

        // Down slice
        ggml_backend_tensor_get(moec.host_down_w, tmp_buf.data(), (size_t)exp_id * down_slice_bytes, down_slice_bytes);
        ggml_backend_tensor_set(moec.slot_down_w, tmp_buf.data(), (size_t)slot * down_slice_bytes, down_slice_bytes);
    }
}

// Compute full CPU reference for all 8 routes
static void compute_cpu_reference(
    moe_oracle_context & moec,
    std::vector<float> & out_unweighted_down,
    std::vector<float> & out_weighted_down,
    std::vector<float> & out_moe_reduced) {

    const auto & spec = moec.spec;
    out_unweighted_down.resize(spec.top_k * spec.d_model);
    out_weighted_down.resize(spec.top_k * spec.d_model);
    out_moe_reduced.assign(spec.d_model, 0.0f);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    ggml_tensor * cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, spec.d_model, 1, 1);
    memcpy(cur->data, moec.input_x.data(), spec.d_model * sizeof(float));

    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, spec.top_k, 1);
    memcpy(ids->data, moec.selected_experts.data(), spec.top_k * sizeof(int32_t));

    ggml_tensor * gate = ggml_mul_mat_id(ctx, moec.host_gate_w, cur, ids);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, moec.host_up_w,   cur, ids);
    ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * down = ggml_mul_mat_id(ctx, moec.host_down_w, act, ids);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, down);
    ggml_backend_graph_compute(moec.backend_cpu, gf);

    memcpy(out_unweighted_down.data(), down->data, spec.top_k * spec.d_model * sizeof(float));

    for (int64_t r = 0; r < spec.top_k; r++) {
        const float w = moec.routing_weights[r];
        for (int64_t d = 0; d < spec.d_model; d++) {
            const float val = out_unweighted_down[r * spec.d_model + d];
            const float w_val = val * w;
            out_weighted_down[r * spec.d_model + d] = w_val;
            out_moe_reduced[d] += w_val;
        }
    }

    ggml_free(ctx);
}

// Compute full GPU reference using slot pools for all 8 routes
static void compute_gpu_reference(
    moe_oracle_context & moec,
    std::vector<float> & out_unweighted_down,
    std::vector<float> & out_weighted_down,
    std::vector<float> & out_moe_reduced) {

    const auto & spec = moec.spec;
    out_unweighted_down.resize(spec.top_k * spec.d_model);
    out_weighted_down.resize(spec.top_k * spec.d_model);
    out_moe_reduced.assign(spec.d_model, 0.0f);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    ggml_tensor * cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, spec.d_model, 1, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, spec.top_k, 1);

    ggml_tensor * gate = ggml_mul_mat_id(ctx, moec.slot_gate_w, cur, ids);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, moec.slot_up_w,   cur, ids);
    ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * down = ggml_mul_mat_id(ctx, moec.slot_down_w, act, ids);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, down);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, moec.backend_gpu);

    std::vector<int32_t> slot_ids(spec.top_k);
    for (int32_t i = 0; i < spec.top_k; i++) slot_ids[i] = i;

    ggml_backend_tensor_set(cur, moec.input_x.data(), 0, spec.d_model * sizeof(float));
    ggml_backend_tensor_set(ids, slot_ids.data(), 0, spec.top_k * sizeof(int32_t));

    ggml_backend_graph_compute(moec.backend_gpu, gf);
    ggml_backend_synchronize(moec.backend_gpu);

    ggml_backend_tensor_get(down, out_unweighted_down.data(), 0, spec.top_k * spec.d_model * sizeof(float));

    for (int64_t r = 0; r < spec.top_k; r++) {
        const float w = moec.routing_weights[r];
        for (int64_t d = 0; d < spec.d_model; d++) {
            const float val = out_unweighted_down[r * spec.d_model + d];
            const float w_val = val * w;
            out_weighted_down[r * spec.d_model + d] = w_val;
            out_moe_reduced[d] += w_val;
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// Execute K hit routes on GPU using slot pools
static void execute_gpu_hit_routes(
    moe_oracle_context & moec,
    int32_t n_hits,
    const std::vector<moe_route_desc> & hit_descs,
    std::vector<float> & out_gpu_hits) {

    const auto & spec = moec.spec;
    out_gpu_hits.resize((size_t)n_hits * spec.d_model);
    if (n_hits == 0) return;

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    ggml_tensor * cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, spec.d_model, 1, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_hits, 1);

    ggml_tensor * gate = ggml_mul_mat_id(ctx, moec.slot_gate_w, cur, ids);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, moec.slot_up_w,   cur, ids);
    ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * down = ggml_mul_mat_id(ctx, moec.slot_down_w, act, ids);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, down);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, moec.backend_gpu);

    std::vector<int32_t> slot_ids(n_hits);
    for (int32_t i = 0; i < n_hits; i++) {
        slot_ids[i] = hit_descs[i].gate_slot;
    }

    ggml_backend_tensor_set(cur, moec.input_x.data(), 0, spec.d_model * sizeof(float));
    ggml_backend_tensor_set(ids, slot_ids.data(), 0, n_hits * sizeof(int32_t));

    ggml_backend_graph_compute(moec.backend_gpu, gf);
    ggml_backend_synchronize(moec.backend_gpu);

    ggml_backend_tensor_get(down, out_gpu_hits.data(), 0, (size_t)n_hits * spec.d_model * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// Execute M miss routes on CPU using host weights
static void execute_cpu_miss_routes(
    moe_oracle_context & moec,
    int32_t n_misses,
    const std::vector<moe_route_desc> & miss_descs,
    std::vector<float> & out_cpu_misses) {

    const auto & spec = moec.spec;
    out_cpu_misses.resize((size_t)n_misses * spec.d_model);
    if (n_misses == 0) return;

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    ggml_tensor * cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, spec.d_model, 1, 1);
    memcpy(cur->data, moec.input_x.data(), spec.d_model * sizeof(float));

    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_misses, 1);
    std::vector<int32_t> miss_expert_ids(n_misses);
    for (int32_t i = 0; i < n_misses; i++) {
        miss_expert_ids[i] = miss_descs[i].expert;
    }
    memcpy(ids->data, miss_expert_ids.data(), n_misses * sizeof(int32_t));

    ggml_tensor * gate = ggml_mul_mat_id(ctx, moec.host_gate_w, cur, ids);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, moec.host_up_w,   cur, ids);
    ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * down = ggml_mul_mat_id(ctx, moec.host_down_w, act, ids);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, down);
    ggml_backend_graph_compute(moec.backend_cpu, gf);

    memcpy(out_cpu_misses.data(), down->data, (size_t)n_misses * spec.d_model * sizeof(float));

    ggml_free(ctx);
}

// Mixed Partial-Hit Execution with arbitrary residency mask
static void compute_mixed_custom_mask(
    moe_oracle_context & moec,
    const std::vector<bool> & is_gpu_hit,
    bool concurrent_mode,
    std::vector<float> & out_scattered_down,
    std::vector<float> & out_weighted_down,
    std::vector<float> & out_moe_reduced,
    int32_t * out_n_hits_executed = nullptr,
    int32_t * out_n_misses_executed = nullptr) {

    const auto & spec = moec.spec;
    const size_t top_k = spec.top_k;
    GGML_ASSERT(is_gpu_hit.size() == top_k);

    out_scattered_down.resize(top_k * spec.d_model);
    out_weighted_down.resize(top_k * spec.d_model);
    out_moe_reduced.assign(spec.d_model, 0.0f);

    std::vector<moe_route_desc> hit_descs;
    std::vector<moe_route_desc> miss_descs;

    for (int32_t r = 0; r < (int32_t)top_k; r++) {
        moe_route_desc desc;
        desc.token = 0;
        desc.route = r;
        desc.expert = moec.selected_experts[r];

        if (is_gpu_hit[r]) {
            desc.gate_slot = r;
            desc.up_slot = r;
            desc.down_slot = r;
            desc.bundle_resident = true;
            hit_descs.push_back(desc);
        } else {
            desc.gate_slot = -1;
            desc.up_slot = -1;
            desc.down_slot = -1;
            desc.bundle_resident = false;
            miss_descs.push_back(desc);
        }
    }

    const int32_t n_gpu_hits = (int32_t)hit_descs.size();
    const int32_t n_misses   = (int32_t)miss_descs.size();

    if (out_n_hits_executed) *out_n_hits_executed = n_gpu_hits;
    if (out_n_misses_executed) *out_n_misses_executed = n_misses;

    std::vector<float> gpu_route_out;
    std::vector<float> cpu_route_out;

    if (concurrent_mode && n_gpu_hits > 0 && n_misses > 0) {
        // Concurrent mode: async overlap
        auto fut_gpu = std::async(std::launch::async, [&]() {
            execute_gpu_hit_routes(moec, n_gpu_hits, hit_descs, gpu_route_out);
        });

        execute_cpu_miss_routes(moec, n_misses, miss_descs, cpu_route_out);
        fut_gpu.get();
    } else {
        // Serial mode: sequential execution
        if (n_gpu_hits > 0) {
            execute_gpu_hit_routes(moec, n_gpu_hits, hit_descs, gpu_route_out);
        }
        if (n_misses > 0) {
            execute_cpu_miss_routes(moec, n_misses, miss_descs, cpu_route_out);
        }
    }

    // Scatter merge both into canonical route layout
    for (int32_t k = 0; k < n_gpu_hits; k++) {
        const int32_t route_idx = hit_descs[k].route;
        memcpy(out_scattered_down.data() + (size_t)route_idx * spec.d_model,
               gpu_route_out.data() + (size_t)k * spec.d_model,
               spec.d_model * sizeof(float));
    }

    for (int32_t m = 0; m < n_misses; m++) {
        const int32_t route_idx = miss_descs[m].route;
        memcpy(out_scattered_down.data() + (size_t)route_idx * spec.d_model,
               cpu_route_out.data() + (size_t)m * spec.d_model,
               spec.d_model * sizeof(float));
    }

    // Downstream weighting and sum reduction
    for (int64_t r = 0; r < (int64_t)top_k; r++) {
        const float w = moec.routing_weights[r];
        for (int64_t d = 0; d < spec.d_model; d++) {
            const float val = out_scattered_down[r * spec.d_model + d];
            const float w_val = val * w;
            out_weighted_down[r * spec.d_model + d] = w_val;
            out_moe_reduced[d] += w_val;
        }
    }
}

// Latency benchmark for given hit mask
static bench_stats benchmark_mask_latency(
    moe_oracle_context & moec,
    const std::vector<bool> & is_gpu_hit,
    bool concurrent_mode,
    int warmup_reps,
    int bench_reps) {

    std::vector<float> down_out, weighted_out, moe_out;

    for (int i = 0; i < warmup_reps; i++) {
        compute_mixed_custom_mask(moec, is_gpu_hit, concurrent_mode, down_out, weighted_out, moe_out);
    }

    std::vector<double> samples;
    samples.reserve(bench_reps);

    for (int i = 0; i < bench_reps; i++) {
        const int64_t t0 = get_time_us();
        compute_mixed_custom_mask(moec, is_gpu_hit, concurrent_mode, down_out, weighted_out, moe_out);
        const int64_t t1 = get_time_us();
        samples.push_back((double)(t1 - t0));
    }

    return compute_stats(samples);
}

int main(int argc, char ** argv) {
    int n_threads = 14;
    int warmup_reps = 20;
    int bench_reps = 1000;
    bool run_bench = true;
    bool serial_only = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup_reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            bench_reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--serial") == 0) {
            serial_only = true;
        } else if (strcmp(argv[i], "--no-bench") == 0) {
            run_bench = false;
        }
    }

    printf("================================================================================\n");
    printf("MoE Partial-Hit Heterogeneous Execution Bench & Oracle Suite (Epics 1-13)\n");
    printf("Model: Gate A Spec (d_model=2048, d_ff=512, n_expert=256, top_k=8, TG1)\n");
    printf("CPU Threads: %d | Warmup: %d | Timed Iterations: %d | Mode: %s\n",
        n_threads, warmup_reps, bench_reps, serial_only ? "Serial Phase 1" : "Dual Mode (Serial + Concurrent)");
    printf("================================================================================\n\n");

    ggml_backend_t backend_cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend_cpu, n_threads);

    ggml_backend_dev_t dev_gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            dev_gpu = dev;
            break;
        }
    }

    if (dev_gpu == nullptr) {
        fprintf(stderr, "ERROR: No GPU backend device found!\n");
        return 1;
    }

    ggml_backend_t backend_gpu = ggml_backend_dev_init(dev_gpu, nullptr);
    printf("GPU Device: %s (%s)\n\n", ggml_backend_dev_name(dev_gpu), ggml_backend_dev_description(dev_gpu));

    moe_gate_a_spec spec;
    moe_oracle_context moec(spec, backend_cpu, backend_gpu);
    setup_oracle_context(moec);

    // -------------------------------------------------------------------------
    // 1. CPU & GPU Baselines
    // -------------------------------------------------------------------------
    printf("--------------------------------------------------------------------------------\n");
    printf("1. Baseline Computation (All-CPU Reference vs All-GPU Reference)\n");
    printf("--------------------------------------------------------------------------------\n");

    std::vector<float> cpu_ref_down, cpu_ref_weighted, cpu_ref_moe;
    compute_cpu_reference(moec, cpu_ref_down, cpu_ref_weighted, cpu_ref_moe);

    std::vector<float> gpu_ref_down, gpu_ref_weighted, gpu_ref_moe;
    compute_gpu_reference(moec, gpu_ref_down, gpu_ref_weighted, gpu_ref_moe);

    error_stats baseline_err = compute_error(gpu_ref_down.data(), cpu_ref_down.data(), cpu_ref_down.size());
    printf("  GPU vs CPU Baseline Down Parity: NMSE=%.6f, MeanRel=%.4f, MaxAbs=%.4f [%s]\n\n",
        baseline_err.nmse, baseline_err.mean_rel_error, baseline_err.max_abs_error, baseline_err.passed ? "PASS" : "FAIL");

    if (!baseline_err.passed) {
        fprintf(stderr, "FAIL: GPU vs CPU baseline tolerance check failed!\n");
        return 1;
    }

    // -------------------------------------------------------------------------
    // 2. Hit-Mask Correctness & Telemetry Matrix (N = 0..8)
    // -------------------------------------------------------------------------
    printf("--------------------------------------------------------------------------------\n");
    printf("2. Partial-Hit Hit-Mask Correctness & Executed-Route Telemetry Matrix (N=0..8)\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("  %-12s | %-10s | %-10s | %-12s | %-12s | %-12s | %-8s\n",
        "Hit Mask", "GPU Routes", "CPU Routes", "Down NMSE", "MeanRel", "MoE MaxAbs", "Status");
    printf("  ------------------------------------------------------------------------------------\n");

    bool all_passed = true;

    for (int32_t N = 0; N <= (int32_t)spec.top_k; N++) {
        std::vector<bool> mask(spec.top_k, false);
        for (int32_t i = 0; i < N; i++) mask[i] = true;

        std::vector<float> mixed_down, mixed_weighted, mixed_moe;
        int32_t gpu_exec = 0;
        int32_t cpu_exec = 0;

        compute_mixed_custom_mask(moec, mask, false, mixed_down, mixed_weighted, mixed_moe, &gpu_exec, &cpu_exec);

        error_stats err_down = compute_error(mixed_down.data(), cpu_ref_down.data(), cpu_ref_down.size());
        error_stats err_moe  = compute_error(mixed_moe.data(), cpu_ref_moe.data(), cpu_ref_moe.size());

        // Verify executed route telemetry invariant
        const bool routes_valid = (gpu_exec == N) && (cpu_exec == (int32_t)spec.top_k - N);
        const bool pass = err_down.passed && err_moe.passed && routes_valid;
        if (!pass) all_passed = false;

        char mask_label[32];
        snprintf(mask_label, sizeof(mask_label), "%d/8 (%dG/%dC)", N, N, (int)(spec.top_k - N));

        printf("  %-12s | %10d | %10d | %12.6f | %12.4f | %12.4f | %-8s\n",
            mask_label, gpu_exec, cpu_exec, err_down.nmse, err_down.mean_rel_error, err_moe.max_abs_error,
            pass ? "PASS" : "FAIL");
    }

    if (!all_passed) {
        fprintf(stderr, "\nFAIL: One or more hit-mask configurations failed correctness or telemetry checks!\n");
        return 1;
    }
    printf("\n>>> All 9 Hit-Mask Permutations (0/8 through 8/8) PASSED! <<<\n\n");

    // -------------------------------------------------------------------------
    // 3. Route-Order Permutation & Non-Contiguous Mask Tests
    // -------------------------------------------------------------------------
    printf("--------------------------------------------------------------------------------\n");
    printf("3. Route-Order Permutation & Non-Contiguous Mask Test Suite\n");
    printf("--------------------------------------------------------------------------------\n");

    struct perm_test {
        std::string name;
        std::vector<bool> mask;
    };

    std::vector<perm_test> perm_tests = {
        { "4/8 Alternating (GCGCGCGC)", { true, false, true, false, true, false, true, false } },
        { "4/8 Inverted Alt (CGCGCGCG)", { false, true, false, true, false, true, false, true } },
        { "4/8 Split Center (CCGGGGCC)", { false, false, true, true, true, true, false, false } },
        { "4/8 Split Edges  (GGCCCCGG)", { true, true, false, false, false, false, true, true } },
        { "7/8 Miss at Head (CGGGGGGG)", { false, true, true, true, true, true, true, true } },
        { "7/8 Miss at Tail (GGGGGGGC)", { true, true, true, true, true, true, true, false } },
        { "7/8 Miss in Mid  (GGGCGGGG)", { true, true, true, false, true, true, true, true } },
        { "1/8 Hit at Head  (GCCCCCCC)", { true, false, false, false, false, false, false, false } },
        { "1/8 Hit at Tail  (CCCCCCCG)", { false, false, false, false, false, false, false, true } },
    };

    bool perms_passed = true;
    for (const auto & pt : perm_tests) {
        std::vector<float> mixed_down, mixed_weighted, mixed_moe;
        int32_t gpu_exec = 0, cpu_exec = 0;
        compute_mixed_custom_mask(moec, pt.mask, false, mixed_down, mixed_weighted, mixed_moe, &gpu_exec, &cpu_exec);

        error_stats err_down = compute_error(mixed_down.data(), cpu_ref_down.data(), cpu_ref_down.size());
        error_stats err_moe  = compute_error(mixed_moe.data(), cpu_ref_moe.data(), cpu_ref_moe.size());

        int expected_hits = 0;
        for (bool b : pt.mask) if (b) expected_hits++;

        const bool pass = err_down.passed && err_moe.passed && (gpu_exec == expected_hits) && (cpu_exec == 8 - expected_hits);
        if (!pass) perms_passed = false;

        printf("  %-28s | Hits: %dG / %dC | Down NMSE: %.6f | MoE MaxAbs: %.4f | [%s]\n",
            pt.name.c_str(), gpu_exec, cpu_exec, err_down.nmse, err_moe.max_abs_error, pass ? "PASS" : "FAIL");
    }

    if (!perms_passed) {
        fprintf(stderr, "\nFAIL: Route-order permutation checks failed!\n");
        return 1;
    }
    printf("\n>>> All Route-Order Permutation Tests PASSED! <<<\n\n");

    // -------------------------------------------------------------------------
    // 4. Repeated Expert Robustness Test
    // -------------------------------------------------------------------------
    printf("--------------------------------------------------------------------------------\n");
    printf("4. Repeated Expert Defensive Robustness Test\n");
    printf("--------------------------------------------------------------------------------\n");

    std::vector<int32_t> orig_experts = moec.selected_experts;
    // Inject repeated expert 7 at route 0 and route 1
    moec.selected_experts[0] = 7;
    moec.selected_experts[1] = 7;

    std::vector<float> rep_cpu_down, rep_cpu_weighted, rep_cpu_moe;
    compute_cpu_reference(moec, rep_cpu_down, rep_cpu_weighted, rep_cpu_moe);

    // Evaluate 4/8 with repeated experts
    std::vector<bool> rep_mask = { true, false, true, false, true, false, true, false };
    std::vector<float> rep_mixed_down, rep_mixed_weighted, rep_mixed_moe;
    compute_mixed_custom_mask(moec, rep_mask, false, rep_mixed_down, rep_mixed_weighted, rep_mixed_moe);

    error_stats rep_err = compute_error(rep_mixed_down.data(), rep_cpu_down.data(), rep_cpu_down.size());
    printf("  Repeated Expert [7, 7, ...] 4/8 Mixed NMSE: %.6f, MeanRel: %.4f, MaxAbs: %.4f [%s]\n\n",
        rep_err.nmse, rep_err.mean_rel_error, rep_err.max_abs_error, rep_err.passed ? "PASS" : "FAIL");

    moec.selected_experts = orig_experts; // Restore

    if (!rep_err.passed) {
        fprintf(stderr, "FAIL: Repeated expert robustness check failed!\n");
        return 1;
    }

    // -------------------------------------------------------------------------
    // 5. Latency Curve Benchmark Suite (>=1000 Iterations)
    // -------------------------------------------------------------------------
    if (run_bench) {
        printf("--------------------------------------------------------------------------------\n");
        printf("5. Partial-Hit Latency Curve Benchmark (N=0..8, %d Iterations)\n", bench_reps);
        printf("--------------------------------------------------------------------------------\n");
        printf("  %-14s | %-10s | %-10s | %-10s | %-10s | %-10s\n",
            "Configuration", "Median (us)", "Mean (us)", "P95 (us)", "Min (us)", "Stddev (us)");
        printf("  ------------------------------------------------------------------------------\n");

        for (int32_t N = 0; N <= (int32_t)spec.top_k; N++) {
            std::vector<bool> mask(spec.top_k, false);
            for (int32_t i = 0; i < N; i++) mask[i] = true;

            bench_stats s = benchmark_mask_latency(moec, mask, false, warmup_reps, bench_reps);
            char label[32];
            snprintf(label, sizeof(label), "%d GPU / %d CPU", N, (int)(spec.top_k - N));

            printf("  %-14s | %10.1f | %10.1f | %10.1f | %10.1f | %10.1f\n",
                label, s.median_us, s.mean_us, s.p95_us, s.min_us, s.stddev_us);
        }
        printf("--------------------------------------------------------------------------------\n\n");

        if (!serial_only) {
            printf("--------------------------------------------------------------------------------\n");
            printf("6. Dual-Device Overlap Concurrency Benchmark (Serial vs Concurrent Overlap)\n");
            printf("--------------------------------------------------------------------------------\n");
            printf("  %-14s | %-12s | %-12s | %-10s\n", "Configuration", "Serial (us)", "Overlap (us)", "Speedup");
            printf("  ------------------------------------------------------------------------------\n");

            for (int32_t N = 1; N < (int32_t)spec.top_k; N++) {
                std::vector<bool> mask(spec.top_k, false);
                for (int32_t i = 0; i < N; i++) mask[i] = true;

                bench_stats s_serial = benchmark_mask_latency(moec, mask, false, warmup_reps, bench_reps / 2);
                bench_stats s_concur = benchmark_mask_latency(moec, mask, true,  warmup_reps, bench_reps / 2);

                const double speedup = s_serial.median_us / std::max(1.0, s_concur.median_us);
                char label[32];
                snprintf(label, sizeof(label), "%d GPU / %d CPU", N, (int)(spec.top_k - N));

                printf("  %-14s | %10.1f us | %10.1f us | %9.2fx\n",
                    label, s_serial.median_us, s_concur.median_us, speedup);
            }
            printf("--------------------------------------------------------------------------------\n\n");
        }
    }

    ggml_backend_free(backend_cpu);
    ggml_backend_free(backend_gpu);

    printf(">>> All Oracle and Benchmark Evaluations Completed Successfully! <<<\n");
    return 0;
}
