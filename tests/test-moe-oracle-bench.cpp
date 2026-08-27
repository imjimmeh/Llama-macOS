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

static void init_tensor_uniform(ggml_tensor * tensor, float min_v = -1.0f, float max_v = 1.0f) {
    const size_t nels = ggml_nelements(tensor);
    std::vector<float> data(nels);

    std::default_random_engine gen(42);
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

// Exact Qwen3.6-35B-A3B dimensions
struct qwen_moe_spec {
    int64_t n_embd       = 2048;
    int64_t n_ff_exp     = 512;
    int64_t n_expert     = 256;
    int64_t n_expert_used= 8;
    ggml_type type_gate  = GGML_TYPE_Q4_K;
    ggml_type type_up    = GGML_TYPE_Q4_K;
    ggml_type type_down  = GGML_TYPE_Q6_K;
};

enum oracle_variant {
    ORACLE_CPU = 0,
    ORACLE_GPU_FULL = 1,
    ORACLE_GPU_SLOT = 2,
};

static const char * oracle_variant_name(oracle_variant v) {
    switch (v) {
        case ORACLE_CPU:      return "CPU (14-thread)";
        case ORACLE_GPU_FULL: return "GPU Full Resident (256 exp)";
        case ORACLE_GPU_SLOT: return "GPU Slot Resident (16 slot)";
        default: return "Unknown";
    }
}

struct op_bench_result {
    std::string op_name;
    int64_t n_tokens;
    oracle_variant variant;
    bench_stats stats;
};

// Benchmark single MUL_MAT_ID operation (Gate, Up, or Down)
static op_bench_result benchmark_single_mul_mat_id(
    const std::string & op_name,
    int64_t k,
    int64_t m,
    int64_t n_expert,
    int64_t n_expert_used,
    int64_t n_tokens,
    ggml_type type_w,
    oracle_variant variant,
    ggml_backend_t backend,
    int warmup_reps,
    int bench_reps) {

    const int64_t n_weights_mats = (variant == ORACLE_GPU_SLOT) ? 16 : n_expert;

    struct ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // w: [k, m, n_weights_mats]
    ggml_tensor * w = ggml_new_tensor_3d(ctx, type_w, k, m, n_weights_mats);
    ggml_set_name(w, "w_exps");

    // cur: [k, (b ? 1 : n_used), n_tokens]
    // For gate/up: cur is [n_embd, 1, n_tokens]
    // For down: cur is [n_ff_exp, n_expert_used, n_tokens]
    const int64_t cur_ne1 = (op_name == "down") ? n_expert_used : 1;
    ggml_tensor * cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, cur_ne1, n_tokens);
    ggml_set_name(cur, "cur");

    // ids: [n_expert_used, n_tokens]
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, n_tokens);
    ggml_set_name(ids, "ids");

    ggml_tensor * out = ggml_mul_mat_id(ctx, w, cur, ids);
    ggml_set_name(out, "out");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // Allocate buffer on the specified backend
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // Initialize weights and inputs
    init_tensor_uniform(w);
    init_tensor_uniform(cur);

    // Initialize deterministic router IDs
    std::vector<int32_t> id_data(n_expert_used * n_tokens);
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t u = 0; u < n_expert_used; u++) {
            if (variant == ORACLE_GPU_SLOT) {
                // Slot mapped IDs (0..15)
                id_data[t * n_expert_used + u] = (int32_t)(u % n_weights_mats);
            } else {
                // Global expert IDs (e.g. 7, 14, 23, 42...)
                id_data[t * n_expert_used + u] = (int32_t)((u * 17 + 7) % n_weights_mats);
            }
        }
    }
    ggml_backend_tensor_set(ids, id_data.data(), 0, id_data.size() * sizeof(int32_t));

    // Warmup
    for (int i = 0; i < warmup_reps; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);

    // Timed iterations
    std::vector<double> samples;
    samples.reserve(bench_reps);

    for (int i = 0; i < bench_reps; i++) {
        const int64_t t0 = get_time_us();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        const int64_t t1 = get_time_us();
        samples.push_back((double)(t1 - t0));
    }

    bench_stats stats = compute_stats(samples);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    return { op_name, n_tokens, variant, stats };
}

// Story 1.3: Full MoE Layer Oracle Benchmark
struct layer_bench_result {
    int64_t n_tokens;
    oracle_variant variant;
    bench_stats full_layer_stats;
};

static layer_bench_result benchmark_full_moe_layer(
    const qwen_moe_spec & spec,
    int64_t n_tokens,
    oracle_variant variant,
    ggml_backend_t backend,
    int warmup_reps,
    int bench_reps) {

    const int64_t n_weights_mats = (variant == ORACLE_GPU_SLOT) ? 16 : spec.n_expert;

    struct ggml_init_params params = {
        /*.mem_size   =*/ 128 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // Input hidden states
    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.n_embd, n_tokens);
    ggml_set_name(inp, "inp");

    // 1. RMS Norm
    ggml_tensor * norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, spec.n_embd);
    ggml_tensor * cur = ggml_rms_norm(ctx, inp, 1e-6f);
    cur = ggml_mul(ctx, cur, norm_w);

    // 2. Router
    ggml_tensor * gate_inp_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.n_embd, spec.n_expert);
    ggml_tensor * logits = ggml_mul_mat(ctx, gate_inp_w, cur); // [n_expert, n_tokens]

    // Selected experts and routing weights
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, spec.n_expert_used, n_tokens);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.n_expert_used, n_tokens);

    // 3. Gate & Up projections
    ggml_tensor * gate_w = ggml_new_tensor_3d(ctx, spec.type_gate, spec.n_embd, spec.n_ff_exp, n_weights_mats);
    ggml_tensor * up_w   = ggml_new_tensor_3d(ctx, spec.type_up,   spec.n_embd, spec.n_ff_exp, n_weights_mats);

    // cur reshape for mul_mat_id: [n_embd, 1, n_tokens]
    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, cur, spec.n_embd, 1, n_tokens);

    ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_w, cur_3d, ids); // [n_ff_exp, n_expert_used, n_tokens]
    ggml_tensor * up   = ggml_mul_mat_id(ctx, up_w,   cur_3d, ids); // [n_ff_exp, n_expert_used, n_tokens]

    // 4. SwiGLU activation: silu(gate) * up
    ggml_tensor * act = ggml_swiglu_split(ctx, gate, up); // [n_ff_exp, n_expert_used, n_tokens]

    // 5. Down projection
    ggml_tensor * down_w = ggml_new_tensor_3d(ctx, spec.type_down, spec.n_ff_exp, spec.n_embd, n_weights_mats);
    ggml_tensor * experts = ggml_mul_mat_id(ctx, down_w, act, ids); // [n_embd, n_expert_used, n_tokens]

    // 6. Weighting & Combine
    // weights: [1, n_expert_used, n_tokens]
    ggml_tensor * weights_3d = ggml_reshape_3d(ctx, weights, 1, spec.n_expert_used, n_tokens);
    ggml_tensor * weighted_experts = ggml_mul(ctx, experts, weights_3d);

    // Sum reduction across used experts
    ggml_tensor * moe_out = nullptr;
    for (int64_t i = 0; i < spec.n_expert_used; i++) {
        ggml_tensor * exp_view = ggml_view_2d(ctx, weighted_experts, spec.n_embd, n_tokens, weighted_experts->nb[2], i * weighted_experts->nb[1]);
        if (moe_out == nullptr) {
            moe_out = exp_view;
        } else {
            moe_out = ggml_add(ctx, moe_out, exp_view);
        }
    }

    // Residual add
    ggml_tensor * out = ggml_add(ctx, inp, moe_out);
    ggml_set_name(out, "layer_out");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // Initialize all tensors
    init_tensor_uniform(inp);
    init_tensor_uniform(norm_w);
    init_tensor_uniform(gate_inp_w);
    init_tensor_uniform(gate_w);
    init_tensor_uniform(up_w);
    init_tensor_uniform(down_w);
    init_tensor_uniform(weights, 0.05f, 0.25f);

    std::vector<int32_t> id_data(spec.n_expert_used * n_tokens);
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t u = 0; u < spec.n_expert_used; u++) {
            if (variant == ORACLE_GPU_SLOT) {
                id_data[t * spec.n_expert_used + u] = (int32_t)(u % n_weights_mats);
            } else {
                id_data[t * spec.n_expert_used + u] = (int32_t)((u * 17 + 7) % n_weights_mats);
            }
        }
    }
    ggml_backend_tensor_set(ids, id_data.data(), 0, id_data.size() * sizeof(int32_t));

    // Warmup
    for (int i = 0; i < warmup_reps; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);

    // Timed iterations
    std::vector<double> samples;
    samples.reserve(bench_reps);

    for (int i = 0; i < bench_reps; i++) {
        const int64_t t0 = get_time_us();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        const int64_t t1 = get_time_us();
        samples.push_back((double)(t1 - t0));
    }

    bench_stats stats = compute_stats(samples);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    return { n_tokens, variant, stats };
}

int main(int argc, char ** argv) {
    int n_threads = 14;
    int warmup_reps = 100;
    int bench_reps = 1000;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup_reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            bench_reps = atoi(argv[++i]);
        }
    }

    printf("================================================================================\n");
    printf("MoE Pre-Resident Expert Oracle Microbenchmark (Epic 1 / Story 1.2 & 1.3)\n");
    printf("Model: Qwen3.6-35B-A3B (n_embd=2048, n_ff_exp=512, n_expert=256, top-k=8)\n");
    printf("Threads: %d | Warmup: %d | Reps: %d\n", n_threads, warmup_reps, bench_reps);
    printf("================================================================================\n\n");

    // Initialize backends
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

    qwen_moe_spec spec;

    const std::vector<int64_t> batch_sizes = { 1, 2, 4, 8 };
    const std::vector<oracle_variant> variants = { ORACLE_CPU, ORACLE_GPU_FULL, ORACLE_GPU_SLOT };

    printf("--------------------------------------------------------------------------------\n");
    printf("Story 1.2: Raw MUL_MAT_ID Operation Microbenchmarks\n");
    printf("--------------------------------------------------------------------------------\n");

    std::vector<op_bench_result> op_results;

    for (int64_t n_tokens : batch_sizes) {
        printf("\n>>> Batch Size: TG%d (n_tokens=%lld)\n", (int)n_tokens, (long long)n_tokens);

        for (oracle_variant var : variants) {
            ggml_backend_t b = (var == ORACLE_CPU) ? backend_cpu : backend_gpu;

            // Gate: [2048, 512, n_expert], Q4_K
            op_bench_result r_gate = benchmark_single_mul_mat_id(
                "gate", spec.n_embd, spec.n_ff_exp, spec.n_expert, spec.n_expert_used,
                n_tokens, spec.type_gate, var, b, warmup_reps, bench_reps);
            op_results.push_back(r_gate);

            // Up: [2048, 512, n_expert], Q4_K
            op_bench_result r_up = benchmark_single_mul_mat_id(
                "up", spec.n_embd, spec.n_ff_exp, spec.n_expert, spec.n_expert_used,
                n_tokens, spec.type_up, var, b, warmup_reps, bench_reps);
            op_results.push_back(r_up);

            // Down: [512, 2048, n_expert], Q6_K
            op_bench_result r_down = benchmark_single_mul_mat_id(
                "down", spec.n_ff_exp, spec.n_embd, spec.n_expert, spec.n_expert_used,
                n_tokens, spec.type_down, var, b, warmup_reps, bench_reps);
            op_results.push_back(r_down);

            printf("  %-28s | Gate: %7.1f us | Up: %7.1f us | Down: %7.1f us | Total: %7.1f us\n",
                oracle_variant_name(var),
                r_gate.stats.median_us,
                r_up.stats.median_us,
                r_down.stats.median_us,
                r_gate.stats.median_us + r_up.stats.median_us + r_down.stats.median_us);
        }
    }

    printf("\n--------------------------------------------------------------------------------\n");
    printf("Story 1.3: Full MoE Layer End-to-End Oracle Benchmark\n");
    printf("--------------------------------------------------------------------------------\n");

    std::vector<layer_bench_result> layer_results;

    for (int64_t n_tokens : batch_sizes) {
        printf("\n>>> Full Layer Batch Size: TG%d (n_tokens=%lld)\n", (int)n_tokens, (long long)n_tokens);

        for (oracle_variant var : variants) {
            ggml_backend_t b = (var == ORACLE_CPU) ? backend_cpu : backend_gpu;

            layer_bench_result lr = benchmark_full_moe_layer(
                spec, n_tokens, var, b, warmup_reps, bench_reps);
            layer_results.push_back(lr);

            printf("  %-28s | Median: %7.1f us | Mean: %7.1f us | P95: %7.1f us | Stddev: %5.1f us\n",
                oracle_variant_name(var),
                lr.full_layer_stats.median_us,
                lr.full_layer_stats.mean_us,
                lr.full_layer_stats.p95_us,
                lr.full_layer_stats.stddev_us);
        }
    }

    printf("\n================================================================================\n");
    printf("Go / No-Go Gate A Decision Summary (TG1 Single-Token Decode Focus)\n");
    printf("================================================================================\n");

    // Find TG1 layer results
    double cpu_layer_tg1 = 0.0;
    double gpu_full_layer_tg1 = 0.0;
    double gpu_slot_layer_tg1 = 0.0;

    for (const auto & lr : layer_results) {
        if (lr.n_tokens == 1) {
            if (lr.variant == ORACLE_CPU) cpu_layer_tg1 = lr.full_layer_stats.median_us;
            if (lr.variant == ORACLE_GPU_FULL) gpu_full_layer_tg1 = lr.full_layer_stats.median_us;
            if (lr.variant == ORACLE_GPU_SLOT) gpu_slot_layer_tg1 = lr.full_layer_stats.median_us;
        }
    }

    if (cpu_layer_tg1 > 0.0 && gpu_full_layer_tg1 > 0.0) {
        const double speedup_full = cpu_layer_tg1 / gpu_full_layer_tg1;
        const double speedup_slot = cpu_layer_tg1 / gpu_slot_layer_tg1;

        printf("TG1 Full Layer Latencies:\n");
        printf("  CPU baseline (14 threads):       %7.1f us\n", cpu_layer_tg1);
        printf("  GPU Full Resident (256 experts): %7.1f us  (Speedup: %.2fx, %+.1f%%)\n",
            gpu_full_layer_tg1, speedup_full, (speedup_full - 1.0) * 100.0);
        printf("  GPU Slot Resident (16 slots):    %7.1f us  (Speedup: %.2fx, %+.1f%%)\n\n",
            gpu_slot_layer_tg1, speedup_slot, (speedup_slot - 1.0) * 100.0);

        printf("Gate A Decision:\n");
        if (speedup_full >= 1.20 || speedup_slot >= 1.20) {
            printf("  [OUTCOME A / PASS] Resident GPU is >= 20%% faster than CPU (Speedup: %.2fx).\n", std::max(speedup_full, speedup_slot));
            printf("  -> Recommendation: Proceed directly to Epic 3 (Static Hot-Expert Residency) and Epic 4 (Heterogeneous Route Execution).\n");
        } else if (speedup_full >= 1.10 || speedup_slot >= 1.10) {
            printf("  [MODERATE PASS] Resident GPU is 10-20%% faster than CPU (Speedup: %.2fx).\n", std::max(speedup_full, speedup_slot));
            printf("  -> Recommendation: Proceed to Epic 3 & Epic 7 (Specialized SM61 DP4A GEMV Kernel).\n");
        } else if (speedup_full >= 0.95) {
            printf("  [OUTCOME B / MARGINAL] Resident GPU is within 10%% of CPU (Speedup: %.2fx).\n", std::max(speedup_full, speedup_slot));
            printf("  -> Recommendation: Generic CUDA MUL_MAT_ID lacks efficiency on Pascal SM61. Jump to Epic 7 (Dedicated SM61 DP4A Fused MoE Kernel).\n");
        } else {
            printf("  [OUTCOME C / SLOWER] Resident GPU is slower than CPU (Speedup: %.2fx).\n", std::max(speedup_full, speedup_slot));
            printf("  -> Recommendation: Stop TG dynamic expert caching on this hardware.\n");
        }
    }
    printf("================================================================================\n");

    ggml_backend_free(backend_cpu);
    ggml_backend_free(backend_gpu);

    return 0;
}
