#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-expert-cache.h"
#include "../ggml/src/ggml-backend-moe-hetero.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

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

// Exact Qwen3.6-35B-A3B TG1 dimensions
struct moe_gate_a_spec {
    int64_t d_model       = 2048;
    int64_t d_ff          = 512;
    int64_t n_expert      = 256;
    int32_t top_k         = 8;
    ggml_type type_gate   = GGML_TYPE_Q4_K;
    ggml_type type_up     = GGML_TYPE_Q4_K;
    ggml_type type_down   = GGML_TYPE_Q6_K;
};

// All execution state is built once before any timed iteration.
struct bench_fixture {
    const moe_gate_a_spec & spec;
    ggml_backend_t backend_cpu;
    ggml_backend_t backend_gpu;

    // host expert weights
    struct ggml_context * ctx_weights = nullptr;
    ggml_backend_buffer_t buf_weights = nullptr;
    ggml_tensor * host_gate_w = nullptr;
    ggml_tensor * host_up_w = nullptr;
    ggml_tensor * host_down_w = nullptr;

    // CPU-base graph for all eight host routes
    struct ggml_context * ctx_base = nullptr;
    ggml_backend_buffer_t buf_base = nullptr;
    ggml_tensor * base_cur = nullptr;
    ggml_tensor * base_ids = nullptr;
    ggml_tensor * base_down = nullptr;
    struct ggml_cgraph * base_graph = nullptr;

    // serial scratch and shared partial executor
    ggml_moe_hetero_scratch_t scratch = nullptr;
    ggml_moe_partial_executor_t executor = nullptr;

    // shared plan tensors: canonical down output, layer input, route ids,
    // and dummy gate/up/activation outputs for plan validation
    struct ggml_context * ctx_plan = nullptr;
    ggml_backend_buffer_t buf_plan = nullptr;
    ggml_tensor * down_node = nullptr;
    ggml_tensor * layer_input = nullptr;
    ggml_tensor * route_ids = nullptr;
    ggml_tensor * gate_out = nullptr;
    ggml_tensor * up_out = nullptr;
    ggml_tensor * act_out = nullptr;
    struct ggml_moe_bundle_plan plan = {};

    std::vector<float> input_x;
    std::vector<int32_t> selected_experts;

    bench_fixture(const moe_gate_a_spec & s, ggml_backend_t b_cpu, ggml_backend_t b_gpu)
        : spec(s), backend_cpu(b_cpu), backend_gpu(b_gpu) {}

    ~bench_fixture() {
        if (executor) ggml_moe_partial_executor_free(executor);
        if (scratch) ggml_moe_hetero_scratch_free(scratch);
        if (buf_base) ggml_backend_buffer_free(buf_base);
        if (ctx_base) ggml_free(ctx_base);
        if (buf_plan) ggml_backend_buffer_free(buf_plan);
        if (ctx_plan) ggml_free(ctx_plan);
        if (buf_weights) ggml_backend_buffer_free(buf_weights);
        if (ctx_weights) ggml_free(ctx_weights);
    }
};

static void setup_fixture(bench_fixture & fx) {
    const auto & spec = fx.spec;

    struct ggml_init_params params_noalloc = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    fx.ctx_weights = ggml_init(params_noalloc);
    fx.host_gate_w = ggml_new_tensor_3d(fx.ctx_weights, spec.type_gate, spec.d_model, spec.d_ff, spec.n_expert);
    fx.host_up_w   = ggml_new_tensor_3d(fx.ctx_weights, spec.type_up,   spec.d_model, spec.d_ff, spec.n_expert);
    fx.host_down_w = ggml_new_tensor_3d(fx.ctx_weights, spec.type_down, spec.d_ff, spec.d_model, spec.n_expert);
    fx.buf_weights = ggml_backend_alloc_ctx_tensors(fx.ctx_weights, fx.backend_cpu);
    init_tensor_uniform(fx.host_gate_w, -1.0f, 1.0f, 101);
    init_tensor_uniform(fx.host_up_w,   -1.0f, 1.0f, 102);
    init_tensor_uniform(fx.host_down_w, -1.0f, 1.0f, 103);

    fx.ctx_base = ggml_init(params_noalloc);
    fx.base_cur = ggml_new_tensor_3d(fx.ctx_base, GGML_TYPE_F32, spec.d_model, 1, 1);
    fx.base_ids = ggml_new_tensor_2d(fx.ctx_base, GGML_TYPE_I32, spec.top_k, 1);
    ggml_tensor * base_gate = ggml_mul_mat_id(fx.ctx_base, fx.host_gate_w, fx.base_cur, fx.base_ids);
    ggml_tensor * base_up   = ggml_mul_mat_id(fx.ctx_base, fx.host_up_w,   fx.base_cur, fx.base_ids);
    ggml_tensor * base_act  = ggml_swiglu_split(fx.ctx_base, base_gate, base_up);
    fx.base_down = ggml_mul_mat_id(fx.ctx_base, fx.host_down_w, base_act, fx.base_ids);
    fx.base_graph = ggml_new_graph(fx.ctx_base);
    ggml_build_forward_expand(fx.base_graph, fx.base_down);
    fx.buf_base = ggml_backend_alloc_ctx_tensors(fx.ctx_base, fx.backend_cpu);

    fx.ctx_plan = ggml_init(params_noalloc);
    fx.down_node   = ggml_new_tensor_2d(fx.ctx_plan, GGML_TYPE_F32, spec.d_model, spec.top_k);
    fx.layer_input = ggml_new_tensor_3d(fx.ctx_plan, GGML_TYPE_F32, spec.d_model, 1, 1);
    fx.route_ids   = ggml_new_tensor_2d(fx.ctx_plan, GGML_TYPE_I32, spec.top_k, 1);
    fx.gate_out    = ggml_new_tensor_2d(fx.ctx_plan, GGML_TYPE_F32, spec.d_ff, spec.top_k);
    fx.up_out      = ggml_new_tensor_2d(fx.ctx_plan, GGML_TYPE_F32, spec.d_ff, spec.top_k);
    fx.act_out     = ggml_new_tensor_2d(fx.ctx_plan, GGML_TYPE_F32, spec.d_ff, spec.top_k);
    fx.buf_plan = ggml_backend_alloc_ctx_tensors(fx.ctx_plan, fx.backend_cpu);

    // deterministic input and the same eight expert IDs for every mode
    fx.input_x.resize(spec.d_model);
    std::default_random_engine gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < spec.d_model; i++) {
        fx.input_x[i] = dist(gen);
    }
    fx.selected_experts = { 7, 23, 45, 89, 112, 168, 204, 241 };

    ggml_backend_tensor_set(fx.layer_input, fx.input_x.data(), 0, spec.d_model * sizeof(float));
    ggml_backend_tensor_set(fx.route_ids, fx.selected_experts.data(), 0, spec.top_k * sizeof(int32_t));
    ggml_backend_tensor_set(fx.base_cur, fx.input_x.data(), 0, spec.d_model * sizeof(float));
    ggml_backend_tensor_set(fx.base_ids, fx.selected_experts.data(), 0, spec.top_k * sizeof(int32_t));

    fx.plan.layer = 0;
    fx.plan.kind = GGML_MOE_BUNDLE_SEPARATE_GATE_UP;
    fx.plan.route_ids = fx.route_ids;
    fx.plan.gate_node = fx.gate_out;
    fx.plan.up_node = fx.up_out;
    fx.plan.act_node = fx.act_out;
    fx.plan.down_node = fx.down_node;
    fx.plan.layer_input = fx.layer_input;
    fx.plan.canonical_route_output = fx.down_node;
    fx.plan.is_fused = false;
    fx.plan.valid = true;
}


// One expert cache per mask so exactly K bundles are resident.
struct mask_cache {
    ggml_backend_expert_cache_t cache = nullptr;
    struct ggml_moe_partial_route_snapshot snapshot = {};
    std::vector<ggml_cache_route_bundle> hits;
    std::vector<ggml_cache_route_bundle> misses;

    ~mask_cache() {
        if (cache) ggml_backend_expert_cache_free(cache);
    }
};

static bool build_mask_cache(bench_fixture & fx, const moe_gate_a_spec & spec, int32_t n_hits, mask_cache & mc) {
    mc.cache = ggml_backend_expert_cache_new(fx.backend_gpu, 64 * 1024 * 1024);
    if (mc.cache == nullptr) {
        return false;
    }
    ggml_backend_expert_cache_register_bundle(mc.cache, 0, fx.host_gate_w, fx.host_up_w, fx.host_down_w);
    for (int32_t i = 0; i < n_hits; i++) {
        const int32_t expert = fx.selected_experts[i];
        ggml_backend_expert_cache_seed(mc.cache, fx.host_gate_w, expert, 1);
        ggml_backend_expert_cache_seed(mc.cache, fx.host_up_w, expert, 1);
        ggml_backend_expert_cache_seed(mc.cache, fx.host_down_w, expert, 1);
    }
    ggml_backend_synchronize(fx.backend_gpu);

    mc.hits.resize(spec.top_k);
    mc.misses.resize(spec.top_k);
    mc.snapshot = {};
    ggml_backend_expert_cache_partition_bundle_routes(
        mc.cache, 0, fx.selected_experts.data(), spec.top_k, 1,
        mc.snapshot.hits, &mc.snapshot.n_hits, mc.snapshot.misses, &mc.snapshot.n_misses);
    return mc.snapshot.n_hits == n_hits && mc.snapshot.n_misses == spec.top_k - n_hits &&
        ggml_moe_partial_route_snapshot_is_valid(&mc.snapshot, spec.top_k);
}

static void run_cpu_base(bench_fixture & fx) {
    ggml_backend_graph_compute(fx.backend_cpu, fx.base_graph);
}

static void run_serial(bench_fixture & fx, mask_cache & mc) {
    const enum ggml_status ec = ggml_backend_moe_hetero_execute_serial(
        fx.backend_gpu, fx.backend_cpu, &fx.plan, mc.cache,
        fx.selected_experts.data(), fx.spec.top_k, fx.scratch, nullptr);
    GGML_ASSERT(ec == GGML_STATUS_SUCCESS);
}

static void run_concurrent(bench_fixture & fx, mask_cache & mc) {
    struct ggml_moe_partial_activation activation = {
        fx.input_x.data(),
        (size_t) fx.spec.d_model * sizeof(float),
    };
    const enum ggml_moe_partial_executor_result result = ggml_moe_partial_executor_execute(
        fx.executor, &fx.plan, mc.cache, &mc.snapshot, &activation, nullptr);
    GGML_ASSERT(result == GGML_MOE_PARTIAL_EXECUTOR_SUCCESS);
}

static bool verify_mode(bench_fixture & fx, mask_cache & mc, int32_t n_hits, const char * mode) {
    std::vector<float> reference(ggml_nbytes(fx.base_down) / sizeof(float));
    ggml_backend_tensor_get(fx.base_down, reference.data(), 0, ggml_nbytes(fx.base_down));

    if (strcmp(mode, "cpu_base") == 0) {
        run_cpu_base(fx);
    } else if (strcmp(mode, "serial") == 0) {
        run_serial(fx, mc);
    } else {
        run_concurrent(fx, mc);
    }

    std::vector<float> actual(ggml_nbytes(fx.down_node) / sizeof(float));
    const ggml_tensor * source = strcmp(mode, "cpu_base") == 0 ? fx.base_down : fx.down_node;
    ggml_backend_tensor_get(source, actual.data(), 0, ggml_nbytes(fx.down_node));

    const error_stats err = compute_error(actual.data(), reference.data(), actual.size());
    printf("    %d/8 %-10s vs CPU-base: NMSE=%.6f MeanRel=%.4f MaxAbs=%.4f [%s]\n",
        n_hits, mode, err.nmse, err.mean_rel_error, err.max_abs_error, err.passed ? "PASS" : "FAIL");
    return err.passed;
}

enum bench_mode { MODE_CPU_BASE, MODE_SERIAL, MODE_CONCURRENT };

static void run_mode_timed(bench_fixture & fx, mask_cache & mc, bench_mode mode,
                           int warmup_reps, int bench_reps, FILE * csv,
                           int32_t n_hits, int32_t n_misses, const char * mode_name,
                           std::vector<double> & samples) {
    for (int i = 0; i < warmup_reps; i++) {
        switch (mode) {
            case MODE_CPU_BASE:   run_cpu_base(fx);          break;
            case MODE_SERIAL:     run_serial(fx, mc);        break;
            case MODE_CONCURRENT: run_concurrent(fx, mc);    break;
        }
    }
    samples.reserve(bench_reps);
    for (int i = 0; i < bench_reps; i++) {
        const int64_t t0 = get_time_us();
        switch (mode) {
            case MODE_CPU_BASE:   run_cpu_base(fx);          break;
            case MODE_SERIAL:     run_serial(fx, mc);        break;
            case MODE_CONCURRENT: run_concurrent(fx, mc);    break;
        }
        const int64_t t1 = get_time_us();
        const double latency = (double)(t1 - t0);
        samples.push_back(latency);
        if (csv) {
            fprintf(csv, "%d,%d,%s,%d,%.0f\n", n_hits, n_misses, mode_name, i, latency);
        }
    }
}

int main(int argc, char ** argv) {
    int n_threads = 14;
    int warmup_reps = 100;
    int bench_reps = 1000;
    bool run_bench = true;
    bool serial_only = false;
    const char * output_path = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup_reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            bench_reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--serial-only") == 0) {
            serial_only = true;
        } else if (strcmp(argv[i], "--no-bench") == 0) {
            run_bench = false;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 1;
        }
    }

    if (run_bench && output_path == nullptr) {
        fprintf(stderr, "ERROR: --output PATH is required for timed measurements\n");
        return 1;
    }

    printf("================================================================================\n");
    printf("MoE Partial-Hit Real Executor Bench (CPU-base vs Serial vs Concurrent)\n");
    printf("Model: Gate A Spec (d_model=2048, d_ff=512, n_expert=256, top_k=8, TG1)\n");
    printf("CPU Threads: %d | Warmup: %d | Timed Iterations: %d | Modes: %s\n",
        n_threads, warmup_reps, bench_reps, serial_only ? "CPU-base + Serial" : "CPU-base + Serial + Concurrent");
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
    bench_fixture fx(spec, backend_cpu, backend_gpu);
    setup_fixture(fx);
    fx.scratch = ggml_moe_hetero_scratch_init(backend_gpu, spec.d_model, spec.d_ff, spec.top_k);
    if (fx.scratch == nullptr) {
        fprintf(stderr, "ERROR: serial scratch init failed\n");
        return 1;
    }

    struct ggml_expert_bundle_weights template_weights = {
        fx.host_gate_w,
        fx.host_up_w,
        fx.host_down_w,
        nullptr,
        false,
    };
    fx.executor = ggml_moe_partial_executor_new(
        backend_gpu, backend_cpu, &template_weights, spec.d_model, spec.d_ff, spec.top_k, false);
    if (fx.executor == nullptr) {
        fprintf(stderr, "ERROR: partial executor init failed\n");
        return 1;
    }

    // baseline: CPU-base output is the canonical reference for every mask
    run_cpu_base(fx);

    FILE * csv = nullptr;
    if (run_bench) {
        csv = fopen(output_path, "w");
        if (csv == nullptr) {
            fprintf(stderr, "ERROR: cannot open %s for writing\n", output_path);
            return 1;
        }
        fprintf(csv, "hits,misses,mode,iteration,latency_us\n");
    }

    bool all_passed = true;
    const bench_mode modes[] = { MODE_CPU_BASE, MODE_SERIAL, MODE_CONCURRENT };
    const char * mode_names[] = { "cpu_base", "serial", "concurrent" };
    const int n_modes = serial_only ? 2 : 3;

    struct summary_row {
        int32_t hits;
        int32_t misses;
        bench_stats stats[3];
    };
    std::vector<summary_row> summary;

    for (int32_t n_hits = 1; n_hits < spec.top_k; n_hits++) {
        printf("  Mask %d/8 (%d GPU hits / %d CPU misses)\n", n_hits, n_hits, spec.top_k - n_hits);
        mask_cache mc;
        if (!build_mask_cache(fx, spec, n_hits, mc)) {
            fprintf(stderr, "FAIL: mask %d residency/snapshot build\n", n_hits);
            return 1;
        }

        for (int m = 0; m < n_modes; m++) {
            if (!verify_mode(fx, mc, n_hits, mode_names[m])) {
                all_passed = false;
            }
        }

        summary_row row = { n_hits, spec.top_k - n_hits, {} };
        if (run_bench) {
            for (int m = 0; m < n_modes; m++) {
                std::vector<double> samples;
                run_mode_timed(fx, mc, modes[m], warmup_reps, bench_reps, csv,
                    n_hits, spec.top_k - n_hits, mode_names[m], samples);
                row.stats[m] = compute_stats(samples);
            }
        }
        summary.push_back(row);
        printf("\n");
    }

    if (csv) {
        fclose(csv);
        printf("Raw samples written to %s\n\n", output_path);
    }

    printf("================================================================================\n");
    printf("Decision Table (median/P95, microseconds; Best by median, not admission policy)\n");
    printf("================================================================================\n");
    printf("| GPU hits | CPU misses | CPU-base median us | CPU-base P95 us | Serial median us | Serial P95 us | Concurrent median us | Concurrent P95 us | Best |\n");
    printf("| -------: | ---------: | -----------------: | ---------------: | ---------------: | ------------: | -------------------: | -------------------: | ---- |\n");
    for (const summary_row & row : summary) {
        const char * best = "cpu_base";
        double best_median = row.stats[MODE_CPU_BASE].median_us;
        if (!serial_only && row.stats[MODE_CONCURRENT].median_us > 0 && row.stats[MODE_CONCURRENT].median_us < best_median) {
            best = "concurrent";
            best_median = row.stats[MODE_CONCURRENT].median_us;
        }
        if (row.stats[MODE_SERIAL].median_us > 0 && row.stats[MODE_SERIAL].median_us < best_median) {
            best = "serial";
            best_median = row.stats[MODE_SERIAL].median_us;
        }
        printf("| %8d | %10d | %18.1f | %16.1f | %16.1f | %13.1f | %20.1f | %19.1f | %-4s |\n",
            row.hits, row.misses,
            row.stats[MODE_CPU_BASE].median_us, row.stats[MODE_CPU_BASE].p95_us,
            row.stats[MODE_SERIAL].median_us, row.stats[MODE_SERIAL].p95_us,
            serial_only ? 0.0 : row.stats[MODE_CONCURRENT].median_us,
            serial_only ? 0.0 : row.stats[MODE_CONCURRENT].p95_us,
            best);
    }
    printf("\n");

    ggml_backend_free(backend_cpu);
    ggml_backend_free(backend_gpu);

    if (!all_passed) {
        fprintf(stderr, "FAIL: one or more mask/mode correctness checks failed!\n");
        return 1;
    }
    printf(">>> All Masks Match the Canonical CPU-base Output! <<<\n");
    return 0;
}
