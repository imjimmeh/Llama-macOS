// MoE mapped-host benchmark: deterministic expert-selection contract, CLI parsing,
// sample statistics, real host-resident Qwen APEX model loading, CPU/VRAM
// complete-FFN controls, and persistent direct mapped-host expert backing for
// the low-VRAM mapped-host oracle experiment.
//
// This target is CUDA-gated (see tests/CMakeLists.txt). Task 2 delivered argument
// parsing, deterministic case selection, statistics, and --self-test. Task 3 adds
// --inspect, the read-only model tensor bundle (qwen_apex_expert_bundle), persistent
// CPU/VRAM mutable weight controls, the complete routed FFN graph (Gate/GateUp
// MUL_MAT_ID -> APEX scale lookup/multiply -> Up -> SwiGLU -> Down -> APEX scale ->
// route weighting -> expert reduction), CPU-versus-VRAM correctness checks, and
// wall-clock timing. Task 4 adds the mapped-host storage mode: --mode, the
// eligibility probe, moe_mapped_host_range, direct GGUF page registration with a
// persistent staging fallback, weight binding via mapped device pointers
// (no-alloc ctx + aliased CUDA buffer), and the mapped-host case runner with
// correctness checks and telemetry fields.
// Task 5 adds the full result schema (CUDA event timing with per-sample sync
// discipline, registration telemetry), the timing_started guard, the pre-write
// schema assert, and the Gate 1 evaluation with its JSON object and stdout
// summary.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"
#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cuda_runtime.h> // cudaHostRegister / cudaHostAlloc / cudaHostGetDevicePointer

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <sys/stat.h>
#    include <unistd.h> // sysconf(_SC_PAGESIZE)
#endif

// ---------------------------------------------------------------------------
// Shared interfaces (contract created by Task 2, consumed by Tasks 3-5)
// ---------------------------------------------------------------------------

enum class moe_storage_mode { cpu, vram, mapped_host };
enum class moe_mapping_kind  { direct_gguf, mapped_staging, unsupported };

// Which storage mode(s) a run executes. The default (no --mode) is cpu+vram
// (the Task 3 behavior); 'all' runs cpu, vram, and mapped_host in that order.
// 'all' is NOT the default; the Gate 1 evaluation needs mapped rows, so the
// gate-smoke run passes --mode all explicitly.
enum class moe_run_mode {
    cpu_vram,
    cpu,
    vram,
    mapped_host,
    all,
};

// ---------------------------------------------------------------------------
// Task 4 interfaces (persistent direct mapped-host expert backing)
// ---------------------------------------------------------------------------

// One registered host-memory range backing a single expert weight tensor.
// direct_gguf: the range is the mmap'd GGUF pages containing t->data,
// registered in place with cudaHostRegister (zero copy, zero staging).
// mapped_staging: the range is a persistent cudaHostAlloc copy made once at
// setup (fallback when direct registration fails, e.g. Windows file mappings).
// All CUDA calls run on the device ordinal captured by the eligibility probe.
struct moe_mapped_host_range {
    const ggml_tensor * source = nullptr; // bundle tensor this range backs
    void * registered_base = nullptr;     // page-aligned host base (cudaHostRegister / cudaHostAlloc)
    void * device_base = nullptr;         // mapped device pointer for registered_base
    size_t source_offset = 0;             // source->data - registered_base (direct) or 0 (staging)
    size_t registered_bytes = 0;          // bytes handed to cudaHostRegister (direct only)
    size_t staging_bytes = 0;             // bytes copied into the staging allocation (staging only)
    double staging_copy_us = 0.0;         // setup-time staging memcpy duration
    moe_mapping_kind kind = moe_mapping_kind::unsupported;

    bool is_usable() const {
        return kind != moe_mapping_kind::unsupported && device_base != nullptr;
    }

    // Device pointer the CUDA kernels read the tensor bytes through.
    void * device_data() const {
        return is_usable() ? (char *) device_base + source_offset : nullptr;
    }

    // Waits for any kernel that could still read the registered pages, then
    // releases the registration / staging allocation. Idempotent.
    void reset();
};

struct moe_shape_case {
    const char * name;
    int64_t n_tokens;
    int64_t n_selected_experts;
};

struct moe_expert_selection {
    std::vector<int32_t> ids;        // n_tokens * n_selected_experts expert IDs
    std::vector<float>    weights;   // one route weight per (token, expert) pair
    int64_t unique_experts;          // distinct expert IDs across the selection
};

struct moe_sample_stats {
    double min_us;
    double median_us;
    double p95_us;
    double mean_us;
    double stddev_us;
};

// ---------------------------------------------------------------------------
// Task 3 interfaces (model bundle, FFN oracle, storage controls)
// ---------------------------------------------------------------------------

// Read-only descriptors of the real model tensors for one MoE layer. Never
// written; the benchmark copies the bytes into per-mode mutable controls.
struct qwen_apex_expert_bundle {
    const ggml_tensor * gate = nullptr;        // blk.N.ffn_gate_exps.weight
    const ggml_tensor * up = nullptr;          // blk.N.ffn_up_exps.weight (null when fused)
    const ggml_tensor * down = nullptr;        // blk.N.ffn_down_exps.weight
    const ggml_tensor * gate_scale = nullptr;  // blk.N.ffn_gate_exps.scale (optional)
    const ggml_tensor * up_scale = nullptr;    // blk.N.ffn_up_exps.scale (optional)
    const ggml_tensor * down_scale = nullptr;  // blk.N.ffn_down_exps.scale (optional)
    bool fused_gate_up = false;                // accepted only when ne[2] == 256 and up == nullptr
};

// One complete routed-FFN graph instance for a single (shape, pattern) case.
// The weight tensors live in the persistent per-mode control; the case context
// holds fresh activation, ID, route-weight, intermediate, and output tensors.
struct moe_ffn_oracle {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * output = nullptr;        // [n_embd, n_tokens] F32 after reduction
    ggml_tensor * inp = nullptr;           // [n_embd, 1, n_tokens] F32 activations
    ggml_tensor * ids = nullptr;           // [n_selected_experts, n_tokens] I32
    ggml_tensor * route_weights = nullptr; // [n_selected_experts, n_tokens] F32
    ggml_backend_buffer_t mutable_buffer = nullptr;
};

// Persistent per-mode mutable copies of the real expert bytes. The CPU control
// is a host copy; the VRAM control is a device copy made once during setup,
// before any warmup, and reported separately.
struct moe_ffn_control {
    moe_storage_mode mode = moe_storage_mode::cpu;
    ggml_backend_t backend = nullptr;
    ggml_context * weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * gate_scale = nullptr;
    ggml_tensor * up_scale = nullptr;
    ggml_tensor * down_scale = nullptr;
    int64_t setup_bytes = 0;        // expert-weight + scale bytes copied into the control
    int64_t setup_h2d_bytes = 0;    // host -> device bytes (VRAM control only)
    int64_t setup_copy_us = 0;
    double registration_us = 0.0;       // Task 5: setup-time registration wall time
    // Task 4: mapped-host backing for the expert weight tensors (gate/up/down;
    // up stays unused in fused mode). The mapping kind is uniform per control
    // (direct-first policy with an all-or-nothing staging fallback).
    moe_mapped_host_range mappings[3] = {};
    int64_t mapping_count = 0;      // number of registered weight tensors
};

// One benchmark result record per (storage mode, shape, pattern).
struct moe_case_result {
    moe_storage_mode mode = moe_storage_mode::cpu;
    std::string shape_name;
    std::string pattern_name;
    int64_t n_tokens = 0;
    int64_t n_selected_experts = 0;
    int64_t unique_experts = 0;
    int64_t samples = 0;
    moe_sample_stats wall;
    // Task 5: CUDA event-elapsed stats (all zeros for CPU rows, which never
    // create events) and the paired CPU control median (CPU rows carry their
    // own wall median).
    moe_sample_stats cuda;
    double cpu_median_us = 0.0;
    int64_t setup_expert_weight_h2d_bytes = 0;
    double max_abs_error = 0.0;
    double mean_relative_error = 0.0;
    double nmse = 0.0;
    bool correctness_pass = false;
    // Task 5: timed-loop sync discipline. explicit_sync_count counts the
    // completion-event synchronizes (one per CUDA sample); backend_wide_sync_count
    // counts ggml_backend_synchronize calls inside the timed loop and is 0 by
    // construction (the timed loop never calls it). The schema assert enforces
    // both invariants before the JSON file is written.
    int64_t explicit_sync_count = 0;
    int64_t backend_wide_sync_count = 0;
    // Task 5 mapped-host telemetry (every row carries the full schema; non-mapped
    // rows use the neutral "none"/"cpu_mode"|"vram_mode" values).
    moe_mapping_kind mapping_kind = moe_mapping_kind::unsupported;
    double registration_us = 0.0;              // setup-time registration wall time
    int64_t direct_registered_bytes = 0;       // bytes pinned by cudaHostRegister
    int64_t staging_bytes = 0;                 // bytes copied into staging allocations
    int64_t timed_expert_weight_h2d_bytes = 0; // must stay 0: kernels read host RAM in place
    int64_t timed_expert_weight_d2h_bytes = 0; // must stay 0: no D2H in the timed loop
    std::string mapped_host_status;            // "ok" | "unsupported" | "cpu_mode" | "vram_mode"
};

// ---------------------------------------------------------------------------
// Task 5: timing_started guard and CUDA event timing plumbing
// ---------------------------------------------------------------------------

// Armed right before the first warmup begins (after every setup weight copy
// and registration). Any expert-weight copy attempted after this point is a
// regression and fails the run; the attempted byte count is recorded so the
// pre-write schema assert (timed_expert_weight_h2d_bytes == 0) can report it.
static bool g_timing_started = false;
static int64_t g_guard_violation_h2d_bytes = 0;

static bool guard_expert_weight_copy(int64_t attempted_bytes) {
    if (!g_timing_started) {
        return true;
    }
    g_guard_violation_h2d_bytes = attempted_bytes;
    fprintf(stderr,
        "ERROR: expert-weight copy attempted after timing started (%lld bytes); "
        "timed_expert_weight_h2d_bytes set to the attempted count\n",
        (long long) attempted_bytes);
    return false;
}

// ggml_backend_event_elapsed_us is exported only through the backend reg proc
// address table (see ggml_backend_event_elapsed_us_t in ggml-backend.h);
// resolved once in main from the CUDA device's backend reg.
static ggml_backend_event_elapsed_us_t g_event_elapsed_us = nullptr;

// Correctness tolerances, approved 2026-08-31 for real q3_K tensors: NMSE is
// the scale-invariant gate; mean_rel uses a 5%-of-RMS denominator floor and a
// 5% threshold, so CPU-vs-CUDA quantization kernel-path noise (~1.6% RMS,
// NMSE 2.5e-4 observed) does not fail the gate while a routing or indexing
// bug (order-of-one errors) still trips it.
static constexpr double k_max_nmse = 0.002;
static constexpr double k_max_mean_relative_error = 0.05;

static const char * storage_mode_name(moe_storage_mode mode) {
    switch (mode) {
        case moe_storage_mode::cpu:         return "cpu";
        case moe_storage_mode::vram:        return "vram";
        case moe_storage_mode::mapped_host: return "mapped_host";
    }
    return "unknown";
}

static const char * moe_run_mode_name(moe_run_mode mode) {
    switch (mode) {
        case moe_run_mode::cpu_vram:    return "cpu+vram";
        case moe_run_mode::cpu:         return "cpu";
        case moe_run_mode::vram:        return "vram";
        case moe_run_mode::mapped_host: return "mapped_host";
        case moe_run_mode::all:         return "all";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// Deterministic case selection
// ---------------------------------------------------------------------------

static const moe_shape_case known_shapes[] = {
    { "tg1",   1,   8 },
    { "tg4",   4,   8 },
    { "tg8",   8,   8 },
    { "pp128", 128, 8 },
    { "pp512", 512, 8 },
};

static const int64_t known_shape_count = (int64_t)(sizeof(known_shapes) / sizeof(known_shapes[0]));

enum class selection_pattern {
    distinct,
    repeated,
    both, // run both patterns in one invocation
};

static const char * selection_pattern_name(selection_pattern pattern) {
    switch (pattern) {
        case selection_pattern::distinct: return "distinct";
        case selection_pattern::repeated: return "repeated";
        case selection_pattern::both:      return "both";
    }
    return "unknown";
}

static const moe_shape_case * find_shape(const char * name) {
    for (int64_t i = 0; i < known_shape_count; i++) {
        if (strcmp(known_shapes[i].name, name) == 0) {
            return &known_shapes[i];
        }
    }
    return nullptr;
}

// Deterministic expert selection. Distinct: expert IDs vary by token and route via
// (token * 17 + route * 31 + 7) % 256. Repeated: the same expert IDs repeat for every
// token via (route * 31 + 7) % 256. Route weights follow route % 2.
static moe_expert_selection make_expert_selection(const moe_shape_case & shape, selection_pattern pattern) {
    moe_expert_selection sel;
    const int64_t n_pairs = shape.n_tokens * shape.n_selected_experts;
    sel.ids.reserve(n_pairs);
    sel.weights.reserve(n_pairs);
    for (int64_t token = 0; token < shape.n_tokens; token++) {
        for (int64_t route = 0; route < shape.n_selected_experts; route++) {
            const int32_t id = (pattern == selection_pattern::distinct)
                ? (int32_t)((token * 17 + route * 31 + 7) % 256)
                : (int32_t)((route * 31 + 7) % 256);
            sel.ids.push_back(id);
            sel.weights.push_back((float)(route % 2));
        }
    }
    std::vector<int32_t> sorted = sel.ids;
    std::sort(sorted.begin(), sorted.end());
    sel.unique_experts = (int64_t)(std::unique(sorted.begin(), sorted.end()) - sorted.begin());
    return sel;
}

// ---------------------------------------------------------------------------
// Sample statistics (reuses tests/test-moe-oracle-bench.cpp compute_stats():
// even/odd median and P95 index floor(0.95 * (n - 1)))
// ---------------------------------------------------------------------------

static moe_sample_stats compute_sample_stats(std::vector<double> & samples) {
    moe_sample_stats stats = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    if (samples.empty()) {
        return stats;
    }
    std::sort(samples.begin(), samples.end());
    const size_t n = samples.size();
    stats.min_us = samples.front();
    stats.median_us = (n % 2 == 0) ? 0.5 * (samples[n / 2 - 1] + samples[n / 2]) : samples[n / 2];
    const size_t p95_idx = (size_t)std::floor(0.95 * (n - 1));
    stats.p95_us = samples[p95_idx];

    double sum = 0.0;
    for (double v : samples) {
        sum += v;
    }
    stats.mean_us = sum / (double)n;

    double sum_sq_diff = 0.0;
    for (double v : samples) {
        const double diff = v - stats.mean_us;
        sum_sq_diff += diff * diff;
    }
    stats.stddev_us = std::sqrt(sum_sq_diff / (double)n);
    return stats;
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

struct bench_params {
    bool self_test = false;
    bool inspect = false;                // --inspect
    std::string model_path;              // -m / --model
    int64_t layer = 0;                   // --layer
    std::vector<std::string> shapes;     // --shapes; empty means all known shapes
    selection_pattern pattern = selection_pattern::distinct; // --pattern
    int64_t warmup = 100;                // --warmup
    int64_t reps = 1000;                 // --reps
    std::string json_path;               // --json <path>
    moe_run_mode mode = moe_run_mode::cpu_vram; // --mode (default: cpu+vram until Task 5)
};

static void print_usage(const char * prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\n");
    printf("  --self-test                 Run the deterministic case contract self-test and exit\n");
    printf("  --inspect                   Load the model, validate the layer tensor bundle, print\n");
    printf("                              tensor details, and exit\n");
    printf("  -m, --model <path>          Model path (required outside --self-test)\n");
    printf("  --layer <int>               MoE layer index to benchmark (default 0)\n");
    printf("  --shapes <list>             Comma-separated shapes: tg1,tg4,tg8,pp128,pp512\n");
    printf("                              (default: all five shapes)\n");
    printf("  --pattern <name>            Expert-ID pattern: distinct|repeated|both\n");
    printf("                              (default: distinct)\n");
    printf("  --mode <name>               Storage mode(s): cpu|vram|mapped_host|all\n");
    printf("                              (default: cpu+vram; 'all' runs cpu, vram, mapped_host in order)\n");
    printf("  --warmup <int>              Warmup iterations (default 100)\n");
    printf("  --reps <int>                Timed iterations, must be >= 1 (default 1000)\n");
    printf("  --json <path>               Write one JSON result record per control to <path>\n");
}

static bool parse_shapes(const char * list, std::vector<std::string> & out) {
    out.clear();
    const std::string rest(list);
    size_t start = 0;
    while (start <= rest.size()) {
        const size_t comma = rest.find(',', start);
        const std::string name = (comma == std::string::npos)
            ? rest.substr(start) : rest.substr(start, comma - start);
        if (name.empty() || find_shape(name.c_str()) == nullptr) {
            fprintf(stderr, "ERROR: unsupported shape '%s'\n", name.c_str());
            return false;
        }
        out.push_back(name);
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return !out.empty();
}

static bool parse_args(int argc, char ** argv, bench_params & params) {
    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (strcmp(arg, "--self-test") == 0) {
            params.self_test = true;
        } else if (strcmp(arg, "--inspect") == 0) {
            params.inspect = true;
        } else if ((strcmp(arg, "-m") == 0 || strcmp(arg, "--model") == 0) && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (strcmp(arg, "--layer") == 0 && i + 1 < argc) {
            params.layer = atoll(argv[++i]);
        } else if (strcmp(arg, "--shapes") == 0 && i + 1 < argc) {
            if (!parse_shapes(argv[++i], params.shapes)) {
                return false;
            }
        } else if (strcmp(arg, "--pattern") == 0 && i + 1 < argc) {
            const char * name = argv[++i];
            if (strcmp(name, "distinct") == 0) {
                params.pattern = selection_pattern::distinct;
            } else if (strcmp(name, "repeated") == 0) {
                params.pattern = selection_pattern::repeated;
            } else if (strcmp(name, "both") == 0) {
                params.pattern = selection_pattern::both;
            } else {
                fprintf(stderr, "ERROR: unsupported pattern '%s' (expected distinct|repeated|both)\n", name);
                return false;
            }
        } else if (strcmp(arg, "--warmup") == 0 && i + 1 < argc) {
            params.warmup = atoll(argv[++i]);
        } else if (strcmp(arg, "--reps") == 0 && i + 1 < argc) {
            params.reps = atoll(argv[++i]);
        } else if (strcmp(arg, "--json") == 0 && i + 1 < argc) {
            params.json_path = argv[++i];
        } else if (strcmp(arg, "--mode") == 0 && i + 1 < argc) {
            const char * name = argv[++i];
            if (strcmp(name, "cpu") == 0) {
                params.mode = moe_run_mode::cpu;
            } else if (strcmp(name, "vram") == 0) {
                params.mode = moe_run_mode::vram;
            } else if (strcmp(name, "mapped_host") == 0) {
                params.mode = moe_run_mode::mapped_host;
            } else if (strcmp(name, "all") == 0) {
                params.mode = moe_run_mode::all;
            } else {
                fprintf(stderr, "ERROR: unsupported --mode '%s' (expected cpu|vram|mapped_host|all)\n", name);
                return false;
            }
        } else {
            fprintf(stderr, "ERROR: unknown or incomplete argument '%s'\n", arg);
            print_usage(argv[0]);
            return false;
        }
    }

    if (params.shapes.empty()) {
        for (int64_t s = 0; s < known_shape_count; s++) {
            params.shapes.push_back(known_shapes[s].name);
        }
    }
    return true;
}

static bool validate_params(const bench_params & params) {
    if (params.reps <= 0) {
        fprintf(stderr, "ERROR: --reps must be >= 1 (got %lld)\n", (long long)params.reps);
        return false;
    }
    if (params.warmup < 0) {
        fprintf(stderr, "ERROR: --warmup must be >= 0 (got %lld)\n", (long long)params.warmup);
        return false;
    }
    if (!params.self_test && params.model_path.empty()) {
        fprintf(stderr, "ERROR: -m/--model is required outside --self-test\n");
        return false;
    }
    if (params.layer < 0) {
        fprintf(stderr, "ERROR: --layer must be >= 0 (got %lld)\n", (long long)params.layer);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Self-test: deterministic case contract
// ---------------------------------------------------------------------------

static bool run_self_test() {
    // Exact contract from the task brief.
    static const moe_shape_case expected_cases[] = {
        { "tg1",   1,   8 },
        { "tg4",   4,   8 },
        { "tg8",   8,   8 },
        { "pp128", 128, 8 },
        { "pp512", 512, 8 },
    };
    const int64_t n_cases = (int64_t)(sizeof(expected_cases) / sizeof(expected_cases[0]));

    int64_t failures = 0;
    printf("self-test: MoE mapped-host deterministic case contract\n");

    for (int64_t c = 0; c < n_cases; c++) {
        const moe_shape_case & shape = expected_cases[c];
        const int64_t n_pairs = shape.n_tokens * shape.n_selected_experts;

        for (int64_t p = 0; p < 2; p++) {
            const selection_pattern pattern = static_cast<selection_pattern>(p);
            const char * pattern_name = selection_pattern_name(pattern);
            const moe_expert_selection sel = make_expert_selection(shape, pattern);

            bool ok = true;

            // ids.size() == n_tokens * n_selected_experts
            if ((int64_t)sel.ids.size() != n_pairs) {
                fprintf(stderr, "  FAIL %s %s: ids.size()=%zu, expected %lld\n",
                    shape.name, pattern_name, sel.ids.size(), (long long)n_pairs);
                ok = false;
            }

            // every ID in [0, 255]
            for (int32_t id : sel.ids) {
                if (id < 0 || id > 255) {
                    fprintf(stderr, "  FAIL %s %s: expert ID %d out of [0, 255]\n",
                        shape.name, pattern_name, id);
                    ok = false;
                    break;
                }
            }

            // route weights have the same count
            if ((int64_t)sel.weights.size() != (int64_t)sel.ids.size()) {
                fprintf(stderr, "  FAIL %s %s: weights.size()=%zu != ids.size()=%zu\n",
                    shape.name, pattern_name, sel.weights.size(), sel.ids.size());
                ok = false;
            }

            // weights follow route % 2
            for (int64_t token = 0; token < shape.n_tokens && ok; token++) {
                for (int64_t route = 0; route < shape.n_selected_experts; route++) {
                    const float expected_weight = (float)(route % 2);
                    if (sel.weights[token * shape.n_selected_experts + route] != expected_weight) {
                        fprintf(stderr, "  FAIL %s %s: weight[%lld][%lld]=%f, expected %f\n",
                            shape.name, pattern_name, (long long)token, (long long)route,
                            sel.weights[token * shape.n_selected_experts + route], expected_weight);
                        ok = false;
                        break;
                    }
                }
            }

            // unique_experts is correct (independent recomputation)
            {
                std::vector<int32_t> sorted = sel.ids;
                std::sort(sorted.begin(), sorted.end());
                const int64_t expected_unique =
                    (int64_t)(std::unique(sorted.begin(), sorted.end()) - sorted.begin());
                if (sel.unique_experts != expected_unique) {
                    fprintf(stderr, "  FAIL %s %s: unique_experts=%lld, expected %lld\n",
                        shape.name, pattern_name,
                        (long long)sel.unique_experts, (long long)expected_unique);
                    ok = false;
                }
            }

            if (pattern == selection_pattern::distinct) {
                // all-distinct: per-token expert IDs are distinct
                bool per_token_distinct = true;
                for (int64_t token = 0; token < shape.n_tokens && per_token_distinct; token++) {
                    std::vector<int32_t> token_ids;
                    token_ids.reserve(shape.n_selected_experts);
                    for (int64_t route = 0; route < shape.n_selected_experts; route++) {
                        token_ids.push_back(sel.ids[token * shape.n_selected_experts + route]);
                    }
                    std::sort(token_ids.begin(), token_ids.end());
                    per_token_distinct =
                        (std::adjacent_find(token_ids.begin(), token_ids.end()) == token_ids.end());
                }
                if (!per_token_distinct) {
                    fprintf(stderr, "  FAIL %s distinct: duplicate expert ID within a token\n", shape.name);
                    ok = false;
                }
                if (shape.n_tokens >= 2 && shape.n_selected_experts >= 2) {
                    // Formula spot checks: id(token, route) = (token * 17 + route * 31 + 7) % 256
                    // id(0,0)=7, id(0,1)=38, id(1,0)=24, id(1,1)=55
                    const int64_t e00 = (int64_t)sel.ids[0];
                    const int64_t e01 = (int64_t)sel.ids[1];
                    const int64_t e10 = (int64_t)sel.ids[shape.n_selected_experts];
                    const int64_t e11 = (int64_t)sel.ids[shape.n_selected_experts + 1];
                    if (e00 != 7 || e01 != 38 || e10 != 24 || e11 != 55) {
                        fprintf(stderr, "  FAIL %s distinct: formula spot check got (%lld,%lld,%lld,%lld)\n",
                            shape.name, (long long)e00, (long long)e01, (long long)e10, (long long)e11);
                        ok = false;
                    }
                }
            } else {
                // repeated: the same expert IDs repeat for every token
                bool token_independent = true;
                for (int64_t token = 1; token < shape.n_tokens && token_independent; token++) {
                    for (int64_t route = 0; route < shape.n_selected_experts; route++) {
                        if (sel.ids[token * shape.n_selected_experts + route] != sel.ids[route]) {
                            token_independent = false;
                            break;
                        }
                    }
                }
                if (!token_independent) {
                    fprintf(stderr, "  FAIL %s repeated: expert IDs are not token-independent\n", shape.name);
                    ok = false;
                }
                // The 8 route IDs of the repeated formula are 8 distinct IDs in [0, 255]
                if (sel.unique_experts != shape.n_selected_experts) {
                    fprintf(stderr, "  FAIL %s repeated: unique_experts=%lld, expected %lld\n",
                        shape.name, (long long)sel.unique_experts, (long long)shape.n_selected_experts);
                    ok = false;
                }
            }

            printf("  %-6s %-8s : %s\n", shape.name, pattern_name, ok ? "OK" : "FAIL");
            if (!ok) {
                failures++;
            }
        }
    }

    const int64_t total = n_cases * 2;
    printf("self-test: %lld/%lld cases passed\n", (long long)(total - failures), (long long)total);
    return failures == 0;
}

// ---------------------------------------------------------------------------
// Model loading and tensor discovery (Task 3)
// ---------------------------------------------------------------------------

static const char * tensor_host_str(const ggml_tensor * t) {
    if (t == nullptr) {
        return "null";
    }
    return (t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer)) ? "yes" : "no";
}

static void print_tensor_info(const char * name, const ggml_tensor * t) {
    if (t == nullptr) {
        printf("  %-42s : null\n", name);
        return;
    }
    printf("  %-42s : type=%-8s ne=[%lld %lld %lld %lld] nb=[%zu %zu %zu %zu] nbytes=%zu host=%s\n",
        name, ggml_type_name(t->type),
        (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
        t->nb[0], t->nb[1], t->nb[2], t->nb[3],
        ggml_nbytes(t), tensor_host_str(t));
}

// Resolves and validates the six required tensor names for the selected layer,
// enforces host residency and scale presence, and accepts a fused GateUp only
// when it has ne[2] == 256 and no separate Up tensor exists.
static bool load_qwen_apex_bundle(struct llama_model * model, int64_t layer, qwen_apex_expert_bundle & bundle) {
    char name[128];

    snprintf(name, sizeof(name), "blk.%lld.ffn_gate_exps.weight", (long long)layer);
    bundle.gate = llama_model_get_tensor(model, name);
    snprintf(name, sizeof(name), "blk.%lld.ffn_up_exps.weight", (long long)layer);
    bundle.up = llama_model_get_tensor(model, name);
    snprintf(name, sizeof(name), "blk.%lld.ffn_down_exps.weight", (long long)layer);
    bundle.down = llama_model_get_tensor(model, name);
    // The APEX Compact GGUF stores per-expert scales as blk.N.ffn_<proj>_exps.scale
    // (production name is tn(LLM_TENSOR_FFN_*_EXPS, "scale"); the ".weight.scale"
    // variant is checked as a fallback for older exports)
    snprintf(name, sizeof(name), "blk.%lld.ffn_gate_exps.scale", (long long)layer);
    bundle.gate_scale = llama_model_get_tensor(model, name);
    if (bundle.gate_scale == nullptr) {
        snprintf(name, sizeof(name), "blk.%lld.ffn_gate_exps.weight.scale", (long long)layer);
        bundle.gate_scale = llama_model_get_tensor(model, name);
    }
    snprintf(name, sizeof(name), "blk.%lld.ffn_up_exps.scale", (long long)layer);
    bundle.up_scale = llama_model_get_tensor(model, name);
    if (bundle.up_scale == nullptr) {
        snprintf(name, sizeof(name), "blk.%lld.ffn_up_exps.weight.scale", (long long)layer);
        bundle.up_scale = llama_model_get_tensor(model, name);
    }
    snprintf(name, sizeof(name), "blk.%lld.ffn_down_exps.scale", (long long)layer);
    bundle.down_scale = llama_model_get_tensor(model, name);
    if (bundle.down_scale == nullptr) {
        snprintf(name, sizeof(name), "blk.%lld.ffn_down_exps.weight.scale", (long long)layer);
        bundle.down_scale = llama_model_get_tensor(model, name);
    }

    printf("Layer %lld expert projection tensors (model descriptors are read-only):\n", (long long)layer);
    print_tensor_info("blk.N.ffn_gate_exps.weight", bundle.gate);
    print_tensor_info("blk.N.ffn_up_exps.weight", bundle.up);
    print_tensor_info("blk.N.ffn_down_exps.weight", bundle.down);
    print_tensor_info("blk.N.ffn_gate_exps.scale", bundle.gate_scale);
    print_tensor_info("blk.N.ffn_up_exps.scale", bundle.up_scale);
    print_tensor_info("blk.N.ffn_down_exps.scale", bundle.down_scale);

    bool ok = true;

    if (bundle.down == nullptr) {
        fprintf(stderr, "ERROR: missing required tensor blk.%lld.ffn_down_exps.weight\n", (long long)layer);
        ok = false;
    }
    if (bundle.down_scale == nullptr) {
        printf("  note: no per-expert down scale tensor (scale-free export; production graph also skips w_s == nullptr)\n");
    }

    if (bundle.up == nullptr) {
        // Fused GateUp acceptance (task brief): the GateUp tensor is accepted as
        // fused only when it has ne[2] == 256 and no separate Up tensor exists.
        // Candidates in order: the canonical fused tensor blk.N.ffn_gate_up_exps.weight,
        // then blk.N.ffn_gate_exps.weight itself when it carries the fused layout.
        const ggml_tensor * fused = nullptr;
        const char * fused_name = nullptr;
        snprintf(name, sizeof(name), "blk.%lld.ffn_gate_up_exps.weight", (long long)layer);
        fused = llama_model_get_tensor(model, name);
        if (fused != nullptr) {
            fused_name = "blk.N.ffn_gate_up_exps.weight";
        } else if (bundle.gate != nullptr && bundle.gate->ne[2] == 256) {
            fused = bundle.gate;
            fused_name = "blk.N.ffn_gate_exps.weight";
        }
        if (fused == nullptr) {
            fprintf(stderr, "ERROR: missing blk.%lld.ffn_up_exps.weight and no fused GateUp candidate "
                "(neither blk.%lld.ffn_gate_up_exps.weight nor a gate tensor with ne[2] == 256)\n",
                (long long)layer, (long long)layer);
            ok = false;
        } else if (fused->ne[2] != 256) {
            fprintf(stderr, "ERROR: fused GateUp candidate %s has ne[2]=%lld; "
                "fused layout accepted only when ne[2] == 256\n", fused_name, (long long)fused->ne[2]);
            ok = false;
        } else {
            const ggml_tensor * fused_scale = nullptr;
            const char * fused_scale_name = nullptr;
            if (fused == bundle.gate) {
                fused_scale = bundle.gate_scale;
                fused_scale_name = "blk.N.ffn_gate_exps.scale";
            } else {
                snprintf(name, sizeof(name), "blk.%lld.ffn_gate_up_exps.scale", (long long)layer);
                fused_scale = llama_model_get_tensor(model, name);
                fused_scale_name = "blk.N.ffn_gate_up_exps.scale";
            }
            if (fused_scale == nullptr) {
                printf("  note: fused GateUp accepted without scale tensor (scale-free export)\n");
            }
            bundle.fused_gate_up = true;
            bundle.gate = fused;
            bundle.gate_scale = fused_scale;
            printf("  fused GateUp accepted: %s (ne[2]=256), no separate Up tensor\n", fused_name);
            print_tensor_info(fused_name, bundle.gate);
            print_tensor_info(fused_scale_name, bundle.gate_scale);
        }
    } else {
        if (bundle.gate == nullptr) {
            fprintf(stderr, "ERROR: missing required tensor blk.%lld.ffn_gate_exps.weight\n", (long long)layer);
            ok = false;
        }
        if (bundle.gate_scale == nullptr || bundle.up_scale == nullptr) {
            printf("  note: gate/up scale tensors absent (scale-free export; production graph also skips w_s == nullptr)\n");
        }
    }

    if (ok) {
        // Require every selected expert projection tensor (and its scale) to use a host buffer.
        const ggml_tensor * projections[] = {
            bundle.gate, bundle.up, bundle.down,
            bundle.gate_scale, bundle.up_scale, bundle.down_scale,
        };
        char projection_names[6][64];
        for (size_t i = 0; i < 6; i++) {
            const char * base = (bundle.fused_gate_up && (i == 0 || i == 3))
                ? "blk.N.ffn_gate_up_exps" : "blk.N.ffn_gate_exps";
            const char * suffix = (i == 0 || i == 1 || i == 2) ? ".weight" : ".scale";
            if (i == 1 || i == 4) {
                base = "blk.N.ffn_up_exps";
            } else if (i == 2 || i == 5) {
                base = "blk.N.ffn_down_exps";
            }
            snprintf(projection_names[i], sizeof(projection_names[i]), "%s%s", base, suffix);
        }
        for (size_t i = 0; i < sizeof(projections) / sizeof(projections[0]); i++) {
            const ggml_tensor * t = projections[i];
            if (t == nullptr) {
                continue; // up and up_scale are null in fused mode
            }
            if (t->buffer == nullptr || !ggml_backend_buffer_is_host(t->buffer)) {
                fprintf(stderr, "ERROR: %s is not host-resident (host=%s); the -cmoe-style "
                    "llm_ffn_exps_cpu_override() must keep it in a CPU buffer\n",
                    projection_names[i], tensor_host_str(t));
                ok = false;
            }
            if (t->type != GGML_TYPE_F32 && t != bundle.gate && t != bundle.up && t != bundle.down) {
                fprintf(stderr, "ERROR: %s has type %s; APEX scale tensors must be F32 for the graph multiply\n",
                    projection_names[i], ggml_type_name(t->type));
                ok = false;
            }
        }

        // Cross-tensor shape consistency.
        const int64_t n_embd = bundle.gate->ne[0];
        const int64_t n_expert = bundle.gate->ne[2];
        const int64_t n_ff = bundle.fused_gate_up ? bundle.gate->ne[1] / 2 : bundle.gate->ne[1];
        if (bundle.fused_gate_up) {
            if (bundle.gate->ne[1] % 2 != 0) {
                fprintf(stderr, "ERROR: fused GateUp ne[1]=%lld is not even\n", (long long)bundle.gate->ne[1]);
                ok = false;
            }
        }
        if (bundle.down->ne[2] != n_expert
                || (bundle.gate_scale != nullptr && bundle.gate_scale->ne[0] != n_expert)
                || (bundle.down_scale != nullptr && bundle.down_scale->ne[0] != n_expert)) {
            fprintf(stderr, "ERROR: expert-axis mismatch: gate ne[2]=%lld down ne[2]=%lld gate_scale ne[0]=%lld down_scale ne[0]=%lld\n",
                (long long)n_expert, (long long)bundle.down->ne[2],
                (long long)(bundle.gate_scale ? bundle.gate_scale->ne[0] : -1),
                (long long)(bundle.down_scale ? bundle.down_scale->ne[0] : -1));
            ok = false;
        }
        if (bundle.up != nullptr && (bundle.up->ne[0] != n_embd || bundle.up->ne[1] != n_ff || bundle.up->ne[2] != n_expert)) {
            fprintf(stderr, "ERROR: Up tensor shape [%lld %lld %lld] does not match gate [%lld %lld %lld]\n",
                (long long)bundle.up->ne[0], (long long)bundle.up->ne[1], (long long)bundle.up->ne[2],
                (long long)n_embd, (long long)n_ff, (long long)n_expert);
            ok = false;
        }
        if (bundle.up_scale != nullptr && bundle.up_scale->ne[0] != n_expert) {
            fprintf(stderr, "ERROR: up_scale ne[0]=%lld != n_expert %lld\n",
                (long long)bundle.up_scale->ne[0], (long long)n_expert);
            ok = false;
        }
        if (bundle.down->ne[0] != n_ff || bundle.down->ne[1] != n_embd) {
            fprintf(stderr, "ERROR: Down tensor shape [%lld %lld %lld] does not match [n_ff=%lld n_embd=%lld n_expert=%lld]\n",
                (long long)bundle.down->ne[0], (long long)bundle.down->ne[1], (long long)bundle.down->ne[2],
                (long long)n_ff, (long long)n_embd, (long long)n_expert);
            ok = false;
        }

        if (ok) {
            printf("  bundle OK: n_embd=%lld n_ff=%lld n_expert=%lld gate=%s up=%s down=%s %s\n",
                (long long)n_embd, (long long)n_ff, (long long)n_expert,
                ggml_type_name(bundle.gate->type),
                bundle.up ? ggml_type_name(bundle.up->type) : "null",
                ggml_type_name(bundle.down->type),
                bundle.fused_gate_up ? "(fused gate_up)" : "");
        }
    }

    return ok;
}

// ---------------------------------------------------------------------------
// CPU versus VRAM correctness metrics
// ---------------------------------------------------------------------------

struct moe_error_metrics {
    double max_abs_error;
    double mean_relative_error;
    double nmse;
    bool finite;
    bool passed;
};

// NMSE (scale-invariant) plus a mean-relative metric whose denominator floor is
// 5% of output RMS, so quantization-path noise on small outputs does not
// dominate. Thresholds approved 2026-08-31: nmse <= 2e-3, mean_rel <= 5e-2.
// Reject non-finite values (AND semantics).
static moe_error_metrics compute_ffn_error(const float * actual, const float * expected, size_t n) {
    double max_abs = 0.0;
    double mse_diff = 0.0;
    double mse_ref = 0.0;
    bool finite = true;

    for (size_t i = 0; i < n; i++) {
        const double act = (double)actual[i];
        const double exp = (double)expected[i];
        if (!std::isfinite(act) || !std::isfinite(exp)) {
            finite = false;
        }
        const double diff = std::fabs(act - exp);
        max_abs = std::max(max_abs, diff);
        mse_diff += diff * diff;
        mse_ref += exp * exp;
    }

    const double nmse = mse_ref > 1e-9 ? (mse_diff / mse_ref) : 0.0;
    double sum_rel = 0.0;
    if (n > 0 && mse_ref > 1e-9) {
        const double rms_ref = std::sqrt(mse_ref / (double)n);
        const double denom_floor = 0.05 * rms_ref;
        for (size_t i = 0; i < n; i++) {
            const double diff = std::fabs((double)actual[i] - (double)expected[i]);
            sum_rel += diff / std::max(std::fabs((double)expected[i]), denom_floor);
        }
    }
    const double mean_rel_out = n > 0 ? (sum_rel / (double)n) : 0.0;
    const bool passed = finite && nmse <= k_max_nmse && mean_rel_out <= k_max_mean_relative_error;

    return { max_abs, mean_rel_out, nmse, finite, passed };
}

// ---------------------------------------------------------------------------
// Complete routed FFN graph (Task 3)
// ---------------------------------------------------------------------------

// Gate/Up/Down MUL_MAT_ID followed by the production-equivalent APEX scale path
// from llm_graph_context::build_lora_mm_id() (src/llama-graph.cpp:1520-1534):
// scale [n_expert] -> reshape [1, n_expert, 1] -> repeat [1, n_expert, n_tokens]
// -> get_rows(ids) -> multiply.
static ggml_tensor * build_apex_scaled_mm_id(
        ggml_context * ctx,
        ggml_tensor * w,
        ggml_tensor * cur,
        ggml_tensor * ids,
        ggml_tensor * scale) {
    ggml_tensor * res = ggml_mul_mat_id(ctx, w, cur, ids);
    if (scale != nullptr) {
        const int64_t n_expert = scale->ne[0];
        const int64_t n_tokens = cur->ne[2];
        ggml_tensor * s = ggml_reshape_3d(ctx, scale, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx, s, ids);
        res = ggml_mul(ctx, res, s);
    }
    return res;
}

// Builds the complete routed FFN for one (shape, pattern) case:
//   Gate/GateUp MUL_MAT_ID -> APEX scale multiply -> Up MUL_MAT_ID -> APEX scale
//   multiply -> SwiGLU -> Down MUL_MAT_ID -> APEX scale multiply -> route
//   weighting -> expert reduction (production-equivalent views + adds).
// Weight tensors come from the persistent per-mode control; the returned oracle
// owns only the case context and its mutable buffer.
static bool build_ffn_oracle(
        const qwen_apex_expert_bundle & bundle,
        const moe_ffn_control & control,
        const moe_shape_case & shape,
        const moe_expert_selection & sel,
        moe_ffn_oracle & oracle) {
    const int64_t n_embd = bundle.gate->ne[0];
    const int64_t n_tokens = shape.n_tokens;
    const int64_t n_selected = shape.n_selected_experts;

    ggml_init_params ip = {
        /*.mem_size   =*/ 256 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    oracle.ctx = ggml_init(ip);
    if (oracle.ctx == nullptr) {
        fprintf(stderr, "ERROR: ggml_init failed for case graph\n");
        return false;
    }

    oracle.inp = ggml_new_tensor_3d(oracle.ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    oracle.ids = ggml_new_tensor_2d(oracle.ctx, GGML_TYPE_I32, n_selected, n_tokens);
    oracle.route_weights = ggml_new_tensor_2d(oracle.ctx, GGML_TYPE_F32, n_selected, n_tokens);
    ggml_set_name(oracle.inp, "inp");
    ggml_set_name(oracle.ids, "ids");
    ggml_set_name(oracle.route_weights, "route_weights");

    ggml_tensor * act = nullptr;
    if (bundle.fused_gate_up) {
        // One MUL_MAT_ID on the fused tensor, then split into gate/up halves.
        ggml_tensor * gate_up = build_apex_scaled_mm_id(oracle.ctx, control.gate, oracle.inp, oracle.ids, control.gate_scale);
        const int64_t n_ff = gate_up->ne[0] / 2;
        ggml_tensor * gate_view = ggml_view_3d(oracle.ctx, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2],
            gate_up->nb[1], gate_up->nb[2], 0);
        ggml_tensor * up_view = ggml_view_3d(oracle.ctx, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2],
            gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        act = ggml_swiglu_split(oracle.ctx, gate_view, up_view);
    } else {
        ggml_tensor * gate_out = build_apex_scaled_mm_id(oracle.ctx, control.gate, oracle.inp, oracle.ids, control.gate_scale);
        ggml_tensor * up_out = build_apex_scaled_mm_id(oracle.ctx, control.up, oracle.inp, oracle.ids, control.up_scale);
        act = ggml_swiglu_split(oracle.ctx, gate_out, up_out);
    }
    ggml_set_name(act, "swiglu");

    ggml_tensor * experts = build_apex_scaled_mm_id(oracle.ctx, control.down, act, oracle.ids, control.down_scale);
    ggml_set_name(experts, "experts");

    // Route weighting: weights [1, n_selected, n_tokens] broadcast over experts.
    ggml_tensor * route_3d = ggml_reshape_3d(oracle.ctx, oracle.route_weights, 1, n_selected, n_tokens);
    ggml_tensor * weighted = ggml_mul(oracle.ctx, experts, route_3d);
    ggml_set_name(weighted, "weighted_experts");

    // Expert reduction: production-equivalent views [n_embd, n_tokens] + adds.
    ggml_tensor * out = nullptr;
    for (int64_t i = 0; i < n_selected; i++) {
        ggml_tensor * view = ggml_view_2d(oracle.ctx, weighted, n_embd, n_tokens, weighted->nb[2], i * weighted->nb[1]);
        out = (out == nullptr) ? view : ggml_add(oracle.ctx, out, view);
    }
    oracle.output = out;
    ggml_set_name(oracle.output, "ffn_out");

    oracle.graph = ggml_new_graph(oracle.ctx);
    ggml_build_forward_expand(oracle.graph, oracle.output);

    oracle.mutable_buffer = ggml_backend_alloc_ctx_tensors(oracle.ctx, control.backend);
    if (oracle.mutable_buffer == nullptr) {
        fprintf(stderr, "ERROR: failed to allocate case graph tensors on %s backend\n", storage_mode_name(control.mode));
        ggml_free(oracle.ctx);
        oracle.ctx = nullptr;
        return false;
    }
    return true;
}

static void free_ffn_oracle(moe_ffn_oracle & oracle) {
    if (oracle.mutable_buffer != nullptr) {
        ggml_backend_buffer_free(oracle.mutable_buffer);
        oracle.mutable_buffer = nullptr;
    }
    if (oracle.ctx != nullptr) {
        ggml_free(oracle.ctx);
        oracle.ctx = nullptr;
    }
    oracle.graph = nullptr;
    oracle.output = nullptr;
    oracle.inp = nullptr;
    oracle.ids = nullptr;
    oracle.route_weights = nullptr;
}

// ---------------------------------------------------------------------------
// Persistent per-mode weight controls (Task 3)
// ---------------------------------------------------------------------------

static ggml_tensor * clone_tensor_meta(ggml_context * ctx, const ggml_tensor * t) {
    return ggml_new_tensor(ctx, t->type, GGML_MAX_DIMS, t->ne);
}

// Allocates fresh mutable copies of the bundle tensors on the given backend and
// copies the real model bytes once. The VRAM control records the host -> device
// bytes; the CPU control is a host -> host copy (zero H2D bytes).
static bool init_ffn_control(
        moe_storage_mode mode,
        ggml_backend_t backend,
        const qwen_apex_expert_bundle & bundle,
        moe_ffn_control & control) {
    control.mode = mode;
    control.backend = backend;

    ggml_init_params ip = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    control.weight_ctx = ggml_init(ip);
    if (control.weight_ctx == nullptr) {
        fprintf(stderr, "ERROR: ggml_init failed for %s weight control\n", storage_mode_name(mode));
        return false;
    }

    control.gate = clone_tensor_meta(control.weight_ctx, bundle.gate);
    control.down = clone_tensor_meta(control.weight_ctx, bundle.down);
    if (bundle.gate_scale != nullptr) {
        control.gate_scale = clone_tensor_meta(control.weight_ctx, bundle.gate_scale);
    }
    if (bundle.down_scale != nullptr) {
        control.down_scale = clone_tensor_meta(control.weight_ctx, bundle.down_scale);
    }
    if (bundle.up != nullptr) {
        control.up = clone_tensor_meta(control.weight_ctx, bundle.up);
    }
    if (bundle.up_scale != nullptr) {
        control.up_scale = clone_tensor_meta(control.weight_ctx, bundle.up_scale);
    }
    ggml_set_name(control.gate, "control.gate");
    if (control.up != nullptr) {
        ggml_set_name(control.up, "control.up");
    }
    ggml_set_name(control.down, "control.down");
    if (control.gate_scale != nullptr) {
        ggml_set_name(control.gate_scale, "control.gate_scale");
    }
    if (control.up_scale != nullptr) {
        ggml_set_name(control.up_scale, "control.up_scale");
    }
    if (control.down_scale != nullptr) {
        ggml_set_name(control.down_scale, "control.down_scale");
    }

    control.weight_buffer = ggml_backend_alloc_ctx_tensors(control.weight_ctx, backend);
    if (control.weight_buffer == nullptr) {
        fprintf(stderr, "ERROR: failed to allocate %s weight control buffer\n", storage_mode_name(mode));
        ggml_free(control.weight_ctx);
        control.weight_ctx = nullptr;
        return false;
    }

    const ggml_tensor * sources[] = { bundle.gate, bundle.up, bundle.down, bundle.gate_scale, bundle.up_scale, bundle.down_scale };
    ggml_tensor * targets[]  = { control.gate, control.up, control.down, control.gate_scale, control.up_scale, control.down_scale };

    control.setup_bytes = 0;
    const int64_t t0 = ggml_time_us();
    for (size_t i = 0; i < sizeof(sources) / sizeof(sources[0]); i++) {
        if (sources[i] == nullptr) {
            continue;
        }
        // Task 5 timing_started guard: no expert-weight copy may run after the
        // guard is armed (setup only). On violation the attempted count is
        // recorded and the run aborts.
        if (!guard_expert_weight_copy((int64_t) ggml_nbytes(sources[i]))) {
            return false;
        }
        ggml_backend_tensor_copy(sources[i], targets[i]);
        control.setup_bytes += (int64_t) ggml_nbytes(sources[i]);
    }
    const int64_t t1 = ggml_time_us();
    control.setup_copy_us = t1 - t0;
    control.setup_h2d_bytes = (mode == moe_storage_mode::vram) ? control.setup_bytes : 0;

    printf("  %-4s control: copied %lld bytes (%lld host->device) in %lld us\n",
        storage_mode_name(mode), (long long)control.setup_bytes,
        (long long)control.setup_h2d_bytes, (long long)control.setup_copy_us);
    return true;
}

static void free_ffn_control(moe_ffn_control & control) {
    if (control.mode == moe_storage_mode::mapped_host) {
        // Release the mapped ranges first: reset() synchronizes the device and
        // unregisters while no kernel can still read the pages, then the
        // aliased CUDA buffer is freed exactly once below (the mutable buffer
        // free path).
        for (int64_t i = 0; i < control.mapping_count; i++) {
            control.mappings[i].reset();
        }
        control.mapping_count = 0;
    }
    if (control.weight_buffer != nullptr) {
        ggml_backend_buffer_free(control.weight_buffer);
        control.weight_buffer = nullptr;
    }
    if (control.weight_ctx != nullptr) {
        ggml_free(control.weight_ctx);
        control.weight_ctx = nullptr;
    }
    control.gate = nullptr;
    control.up = nullptr;
    control.down = nullptr;
    control.gate_scale = nullptr;
    control.up_scale = nullptr;
    control.down_scale = nullptr;
}

// ---------------------------------------------------------------------------
// Mapped-host storage mode (Task 4)
// ---------------------------------------------------------------------------

// CUDA device ordinal captured once at startup; every mapped-host CUDA call in
// this test runs on this device (the first GPU backend device).
static int g_mapped_device = 0;

// Teardown ordering contract: reset() must never unregister while a kernel
// could still read the pages. The CUDA stream of the ggml backend context is
// not reachable through the public ggml API (there is no backend -> native
// stream accessor), so reset() uses the design's "otherwise" branch: one
// cudaDeviceSynchronize() at teardown. That sync runs after the last timed
// sample of the last case (ggml_backend_graph_compute already synchronizes
// after every compute) and outside any timed region, so the timed-loop
// invariant is untouched. After the sync, no kernel can still read the range,
// so cudaHostUnregister / cudaFreeHost is safe; no further graph computes are
// issued on this backend afterwards (reset() runs before the control's CUDA
// buffer and the backend are freed). Idempotent: a second call is a no-op.
void moe_mapped_host_range::reset() {
    if (kind == moe_mapping_kind::unsupported || registered_base == nullptr) {
        return; // nothing registered (or already released)
    }
    cudaSetDevice(g_mapped_device);
    const cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "  mapped-host: teardown sync error: %s\n", cudaGetErrorString(sync_err));
    }
    cudaError_t err = (kind == moe_mapping_kind::direct_gguf)
        ? cudaHostUnregister(registered_base)
        : cudaFreeHost(registered_base);
    if (err != cudaSuccess) {
        fprintf(stderr, "  mapped-host: release error for %s: %s\n",
            source != nullptr ? ggml_get_name(source) : "?",
            cudaGetErrorString(err));
        (void)cudaGetLastError(); // clear the sticky error
    }
    kind = moe_mapping_kind::unsupported;
    registered_base = nullptr;
    device_base = nullptr;
    source = nullptr;
    source_offset = 0;
    registered_bytes = 0;
    staging_bytes = 0;
    staging_copy_us = 0.0;
}

static size_t host_page_size() {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (size_t) si.dwPageSize;
#else
    return (size_t) sysconf(_SC_PAGESIZE);
#endif
}

static const char * mapping_kind_name(moe_mapping_kind kind) {
    switch (kind) {
        case moe_mapping_kind::direct_gguf:   return "direct_gguf";
        case moe_mapping_kind::mapped_staging: return "mapped_staging";
        case moe_mapping_kind::unsupported:   return "unsupported";
    }
    return "unknown";
}

// Direct registration: page-align t->data inside the mmap'd GGUF file and pin
// those pages in place (cudaHostRegister with Portable|Mapped|ReadOnly), then
// map them into the device address space with cudaHostGetDevicePointer. CUDA
// kernels then read the model's own file-mapped pages over PCIe: zero copy,
// zero staging. On failure the cudaGetErrorString diagnostics are printed, the
// range is marked unsupported, and the caller falls back to staging.
static moe_mapped_host_range make_direct_gguf_mapping(const ggml_tensor * t) {
    moe_mapped_host_range m;
    m.source = t;
    cudaSetDevice(g_mapped_device);

    const size_t page = host_page_size();
    const uintptr_t addr = (uintptr_t) t->data;
    const uintptr_t base_addr = addr & ~(uintptr_t)(page - 1);
    m.registered_base = (void *) base_addr;
    m.source_offset = (size_t)(addr - base_addr);
    const size_t nbytes = ggml_nbytes(t);
    m.registered_bytes = (m.source_offset + nbytes + page - 1) & ~(size_t)(page - 1);

    cudaError_t err = cudaHostRegister(m.registered_base, m.registered_bytes,
        cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        (void)cudaGetLastError(); // clear the sticky error
        fprintf(stderr, "  mapped-host: direct registration FAILED for %s: %s\n",
            ggml_get_name(t), cudaGetErrorString(err));
        fprintf(stderr, "  mapped-host:   base=%p offset=%zu requested=%zu bytes (OS page %zu)\n",
            m.registered_base, m.source_offset, m.registered_bytes, page);
        m.registered_base = nullptr;
        m.registered_bytes = 0;
        return m; // kind stays unsupported
    }

    err = cudaHostGetDevicePointer(&m.device_base, m.registered_base, 0);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "  mapped-host: cudaHostGetDevicePointer FAILED for %s: %s\n",
            ggml_get_name(t), cudaGetErrorString(err));
        (void)cudaHostUnregister(m.registered_base);
        m.registered_base = nullptr;
        m.registered_bytes = 0;
        return m;
    }

    m.kind = moe_mapping_kind::direct_gguf;
    printf("  mapped-host: direct registration OK for %s: base=%p offset=%zu registered=%zu bytes\n",
        ggml_get_name(t), m.registered_base, m.source_offset, m.registered_bytes);
    return m;
}

// Staging fallback: one persistent cudaHostAlloc copy of the tensor bytes,
// made once during setup (never in the timed loop), mapped into the device
// address space. Used for ALL weight tensors of a control when any direct
// registration fails, so the mapping kind is uniform per control.
static moe_mapped_host_range make_staging_mapping(const ggml_tensor * t) {
    moe_mapped_host_range m;
    m.source = t;
    cudaSetDevice(g_mapped_device);

    const size_t nbytes = ggml_nbytes(t);
    const int64_t t0 = ggml_time_us();
    cudaError_t err = cudaHostAlloc(&m.registered_base, nbytes,
        cudaHostAllocPortable | cudaHostAllocMapped);
    const int64_t t1 = ggml_time_us();
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "  mapped-host: cudaHostAlloc FAILED for %s: %s\n",
            ggml_get_name(t), cudaGetErrorString(err));
        return m; // kind stays unsupported
    }

    m.staging_bytes = nbytes;
    // Task 5 timing_started guard: the staging memcpy is a setup-only expert
    // weight copy; any attempt after the guard is armed aborts the run.
    if (!guard_expert_weight_copy((int64_t) nbytes)) {
        (void) cudaFreeHost(m.registered_base);
        m.registered_base = nullptr;
        m.staging_bytes = 0;
        return m; // kind stays unsupported
    }
    memcpy(m.registered_base, t->data, nbytes);
    const int64_t t2 = ggml_time_us();

    err = cudaHostGetDevicePointer(&m.device_base, m.registered_base, 0);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "  mapped-host: cudaHostGetDevicePointer FAILED for staging of %s: %s\n",
            ggml_get_name(t), cudaGetErrorString(err));
        (void)cudaFreeHost(m.registered_base);
        m.registered_base = nullptr;
        m.staging_bytes = 0;
        return m;
    }

    m.kind = moe_mapping_kind::mapped_staging;
    printf("  mapped-host: staging copy for %s: %zu bytes in %.0f us (alloc %.0f us)\n",
        ggml_get_name(t), m.staging_bytes, m.staging_copy_us, (double)(t1 - t0));
    return m;
}

// Eligibility probe, run once per run when a mapped-host mode is requested.
// The mapped path requires cudaDevAttrCanMapHostMemory on the device; the
// device flags must also include cudaDeviceMapHost (set once at startup, see
// main). Prints and emits mapped_host_status: "unsupported" on failure.
static bool probe_mapped_host_support() {
    cudaSetDevice(g_mapped_device);
    int can_map = 0;
    const cudaError_t err =
        cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, g_mapped_device);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "  mapped-host: eligibility probe FAILED: %s\n", cudaGetErrorString(err));
        printf("mapped_host_status: \"unsupported\"\n");
        return false;
    }
    if (can_map == 0) {
        printf("  mapped-host: device %d does not support mapping host memory "
            "(cudaDevAttrCanMapHostMemory = 0)\n", g_mapped_device);
        printf("mapped_host_status: \"unsupported\"\n");
        return false;
    }
    printf("  mapped-host: device %d supports host memory mapping\n", g_mapped_device);
    return true;
}

// Persistent mapped-host weight control. The expert weight tensors (gate/up/
// down; up is null in fused mode) are registered once at setup with the
// direct-first policy: every tensor tries direct GGUF page registration first
// and if ANY of them fails, ALL weight tensors of the control fall back to
// mapped staging (uniform mapping per control). The control tensors then get
// data = mapped device pointer while buffer stays the control's CUDA buffer as
// an alias (the moe-l2 trick: buffer type CUDA, data host RAM), so a direct
// ggml_backend_graph_compute(cuda_backend, graph) reads the weights over PCIe
// with zero H2D copy. Scale tensors (if any) stay as real mutable copies
// inside the CUDA buffer.
static bool init_mapped_host_control(
        ggml_backend_t backend,
        const qwen_apex_expert_bundle & bundle,
        moe_ffn_control & control) {
    control.mode = moe_storage_mode::mapped_host;
    control.backend = backend;

    ggml_init_params ip = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    control.weight_ctx = ggml_init(ip);
    if (control.weight_ctx == nullptr) {
        fprintf(stderr, "ERROR: ggml_init failed for the mapped-host weight control\n");
        return false;
    }

    control.gate = clone_tensor_meta(control.weight_ctx, bundle.gate);
    control.down = clone_tensor_meta(control.weight_ctx, bundle.down);
    if (bundle.gate_scale != nullptr) {
        control.gate_scale = clone_tensor_meta(control.weight_ctx, bundle.gate_scale);
    }
    if (bundle.down_scale != nullptr) {
        control.down_scale = clone_tensor_meta(control.weight_ctx, bundle.down_scale);
    }
    if (bundle.up != nullptr) {
        control.up = clone_tensor_meta(control.weight_ctx, bundle.up);
    }
    if (bundle.up_scale != nullptr) {
        control.up_scale = clone_tensor_meta(control.weight_ctx, bundle.up_scale);
    }
    ggml_set_name(control.gate, "control.gate");
    if (control.up != nullptr) {
        ggml_set_name(control.up, "control.up");
    }
    ggml_set_name(control.down, "control.down");
    if (control.gate_scale != nullptr) {
        ggml_set_name(control.gate_scale, "control.gate_scale");
    }
    if (control.up_scale != nullptr) {
        ggml_set_name(control.up_scale, "control.up_scale");
    }
    if (control.down_scale != nullptr) {
        ggml_set_name(control.down_scale, "control.down_scale");
    }

    // 1) Register the expert weight tensors (direct-first, uniform per control).
    const ggml_tensor * weight_sources[] = { bundle.gate, bundle.up, bundle.down };
    const char * weight_labels[] = { "gate", "up", "down" };
    // Task 5 timing_started guard: registration is a setup-only operation; any
    // attempt after the guard is armed aborts the run with the attempted bytes.
    int64_t weight_bytes_total = 0;
    for (size_t i = 0; i < 3; i++) {
        if (weight_sources[i] != nullptr) {
            weight_bytes_total += (int64_t) ggml_nbytes(weight_sources[i]);
        }
    }
    if (!guard_expert_weight_copy(weight_bytes_total)) {
        for (size_t j = 0; j < 3; j++) {
            control.mappings[j].reset();
        }
        control.mapping_count = 0;
        ggml_free(control.weight_ctx);
        control.weight_ctx = nullptr;
        return false;
    }
    const int64_t t_reg0 = ggml_time_us();
    bool direct_failed = false;
    for (size_t i = 0; i < 3; i++) {
        if (weight_sources[i] == nullptr) {
            continue;
        }
        control.mappings[i] = make_direct_gguf_mapping(weight_sources[i]);
        if (!control.mappings[i].is_usable()) {
            direct_failed = true;
        }
    }
    const int64_t t_reg1 = ggml_time_us();
    control.registration_us = (double) (t_reg1 - t_reg0);

    if (direct_failed) {
        printf("  mapped-host: direct GGUF page registration failed for at least one weight tensor;\n");
        printf("  mapped-host: falling back to mapped staging for ALL weight tensors of this control\n");
        for (size_t i = 0; i < 3; i++) {
            if (weight_sources[i] == nullptr) {
                continue;
            }
            if (control.mappings[i].is_usable()) {
                control.mappings[i].reset(); // unregister the direct registration first
            }
            control.mappings[i] = make_staging_mapping(weight_sources[i]);
            if (!control.mappings[i].is_usable()) {
                fprintf(stderr, "ERROR: mapped-host staging mapping failed for %s\n", weight_labels[i]);
                for (size_t j = 0; j < 3; j++) {
                    control.mappings[j].reset();
                }
                control.mapping_count = 0;
                ggml_free(control.weight_ctx);
                control.weight_ctx = nullptr;
                return false;
            }
        }
    }
    for (size_t i = 0; i < 3; i++) {
        if (weight_sources[i] != nullptr) {
            control.mapping_count++;
        }
    }

    // 2) Allocate the CUDA buffer for the control tensors (the alias target).
    control.weight_buffer = ggml_backend_alloc_ctx_tensors(control.weight_ctx, backend);
    if (control.weight_buffer == nullptr) {
        fprintf(stderr, "ERROR: failed to allocate the mapped-host weight control buffer\n");
        for (size_t i = 0; i < 3; i++) {
            control.mappings[i].reset();
        }
        control.mapping_count = 0;
        ggml_free(control.weight_ctx);
        control.weight_ctx = nullptr;
        return false;
    }

    // 3) Bind the weights to their mapped device pointers; buffer stays the
    //    CUDA alias (the weight slices inside the buffer are never touched).
    ggml_tensor * weight_targets[] = { control.gate, control.up, control.down };
    for (size_t i = 0; i < 3; i++) {
        if (weight_targets[i] == nullptr) {
            continue;
        }
        weight_targets[i]->data = control.mappings[i].device_data();
    }

    // 4) Copy the scale tensors (real mutable copies inside the CUDA buffer).
    const ggml_tensor * scale_sources[] = { bundle.gate_scale, bundle.up_scale, bundle.down_scale };
    ggml_tensor * scale_targets[] = { control.gate_scale, control.up_scale, control.down_scale };
    control.setup_bytes = 0;
    const int64_t t0 = ggml_time_us();
    for (size_t i = 0; i < 3; i++) {
        if (scale_sources[i] == nullptr) {
            continue;
        }
        // Task 5 timing_started guard: scale copies are setup-only.
        if (!guard_expert_weight_copy((int64_t) ggml_nbytes(scale_sources[i]))) {
            return false;
        }
        ggml_backend_tensor_copy(scale_sources[i], scale_targets[i]);
        control.setup_bytes += (int64_t) ggml_nbytes(scale_sources[i]);
    }
    const int64_t t1 = ggml_time_us();
    control.setup_h2d_bytes = 0; // expert weights are never copied; scale bytes (if any) are setup-only

    printf("  mapped control: kind=%s weights_registered=%lld setup_bytes=%lld host->device=%lld in %lld us\n",
        mapping_kind_name(control.mappings[0].kind), (long long)control.mapping_count,
        (long long)control.setup_bytes, (long long)control.setup_h2d_bytes, (long long)control.setup_copy_us);
    return true;
}

// ---------------------------------------------------------------------------
// Deterministic activations
// ---------------------------------------------------------------------------

// Fixed-seed LCG in [-1, 1); identical on every run and for every mode.
static void fill_deterministic_f32(std::vector<float> & data) {
    uint32_t state = 0x9E3779B9u;
    for (size_t i = 0; i < data.size(); i++) {
        state = state * 1664525u + 1013904223u;
        data[i] = ((float)(state >> 8) / (float)0xFFFFFFu) * 2.0f - 1.0f;
    }
}

// ---------------------------------------------------------------------------
// CPU / VRAM case execution (Task 3)
// ---------------------------------------------------------------------------

// Runs one correctness pass (a single graph compute followed by a host copy of
// the output) and, when do_timing is set, the warmup and timed loop. Wall-clock
// samples use ggml_time_us(). Task 5: VRAM computes are timed with one
// persistent ggml_backend_event_t pair per (mode, case) around each graph
// compute on the CUDA backend; only the completion event is synchronized per
// sample (explicit_sync_count++). ggml_backend_synchronize() is NEVER called
// inside the timed loop, so backend_wide_sync_count stays 0 during timed
// samples. CPU mode creates no events (cuda stats stay zero).
static bool run_cpu_or_vram_case(
        moe_storage_mode mode,
        ggml_backend_t backend,
        const moe_ffn_oracle & oracle,
        const std::vector<float> & inp,
        const moe_expert_selection & sel,
        int64_t warmup,
        int64_t reps,
        bool do_timing,
        std::vector<float> & out_once,
        moe_sample_stats & stats,
        moe_sample_stats & cuda_stats,
        int64_t & explicit_sync_count) {
    stats = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    cuda_stats = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    explicit_sync_count = 0;

    // Fill the case inputs on the mode's backend.
    ggml_backend_tensor_set(oracle.inp, inp.data(), 0, inp.size() * sizeof(float));
    ggml_backend_tensor_set(oracle.ids, sel.ids.data(), 0, sel.ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(oracle.route_weights, sel.weights.data(), 0, sel.weights.size() * sizeof(float));

    // Correctness run (before timing).
    if (ggml_backend_graph_compute(backend, oracle.graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: %s graph compute failed during the correctness run\n", storage_mode_name(mode));
        return false;
    }
    if (mode == moe_storage_mode::vram) {
        ggml_backend_synchronize(backend);
    }
    const size_t out_elems = ggml_nelements(oracle.output);
    out_once.resize(out_elems);
    ggml_backend_tensor_get(oracle.output, out_once.data(), 0, out_elems * sizeof(float));

    if (!do_timing) {
        return true;
    }

    // One persistent event pair per (mode, case); CPU mode never creates events.
    ggml_backend_event_t ev_start = nullptr;
    ggml_backend_event_t ev_end = nullptr;
    if (mode == moe_storage_mode::vram) {
        const ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ev_start = ggml_backend_event_new(dev);
        ev_end = ggml_backend_event_new(dev);
        if (ev_start == nullptr || ev_end == nullptr || g_event_elapsed_us == nullptr) {
            fprintf(stderr, "ERROR: cannot create CUDA timing events for the %s control\n",
                storage_mode_name(mode));
            if (ev_start != nullptr) {
                ggml_backend_event_free(ev_start);
            }
            if (ev_end != nullptr) {
                ggml_backend_event_free(ev_end);
            }
            return false;
        }
    }

    for (int64_t i = 0; i < warmup; i++) {
        ggml_backend_graph_compute(backend, oracle.graph);
    }
    if (mode == moe_storage_mode::vram) {
        ggml_backend_synchronize(backend); // warmup drain; outside the timed loop
    }

    std::vector<double> samples;
    samples.reserve((size_t) reps);
    std::vector<double> cuda_samples;
    cuda_samples.reserve(mode == moe_storage_mode::vram ? (size_t) reps : 0);
    for (int64_t i = 0; i < reps; i++) {
        const int64_t t0 = ggml_time_us();
        if (mode == moe_storage_mode::vram) {
            ggml_backend_event_record(ev_start, backend);
        }
        ggml_backend_graph_compute(backend, oracle.graph);
        if (mode == moe_storage_mode::vram) {
            ggml_backend_event_record(ev_end, backend);
            ggml_backend_event_synchronize(ev_end); // completion wait; explicit per sample
            explicit_sync_count++;
        }
        const int64_t t1 = ggml_time_us();
        samples.push_back((double) (t1 - t0));
        if (mode == moe_storage_mode::vram) {
            uint64_t elapsed_us = 0;
            if (!g_event_elapsed_us(ev_start, ev_end, &elapsed_us)) {
                fprintf(stderr, "ERROR: ggml_backend_event_elapsed_us failed for the %s control sample %lld\n",
                    storage_mode_name(mode), (long long) i);
                ggml_backend_event_free(ev_start);
                ggml_backend_event_free(ev_end);
                return false;
            }
            cuda_samples.push_back((double) elapsed_us);
        }
    }
    stats = compute_sample_stats(samples);
    if (mode == moe_storage_mode::vram) {
        cuda_stats = compute_sample_stats(cuda_samples);
    }
    if (ev_start != nullptr) {
        ggml_backend_event_free(ev_start);
    }
    if (ev_end != nullptr) {
        ggml_backend_event_free(ev_end);
    }
    return true;
}

// Runs one (shape, pattern) case on both controls: one CPU run and one VRAM run
// before any timing, the correctness comparison, then warmup + timed loops.
static bool run_case_pair(
        const qwen_apex_expert_bundle & bundle,
        const moe_ffn_control & cpu_control,
        const moe_ffn_control & vram_control,
        const moe_shape_case & shape,
        selection_pattern pattern,
        const bench_params & params,
        moe_case_result & cpu_result,
        moe_case_result & vram_result) {
    const moe_expert_selection sel = make_expert_selection(shape, pattern);
    const int64_t n_embd = bundle.gate->ne[0];
    const int64_t out_elems = n_embd * shape.n_tokens;

    std::vector<float> inp((size_t)out_elems);
    fill_deterministic_f32(inp);

    moe_ffn_oracle cpu_oracle;
    moe_ffn_oracle vram_oracle;
    if (!build_ffn_oracle(bundle, cpu_control, shape, sel, cpu_oracle)) {
        return false;
    }
    if (!build_ffn_oracle(bundle, vram_control, shape, sel, vram_oracle)) {
        free_ffn_oracle(cpu_oracle);
        return false;
    }

    bool ok = true;

    // Phase 1: one CPU run and one VRAM run before any timing.
    std::vector<float> cpu_once;
    std::vector<float> vram_once;
    moe_sample_stats cpu_pre;
    moe_sample_stats vram_pre;
    moe_sample_stats cpu_pre_cuda;
    moe_sample_stats vram_pre_cuda;
    int64_t cpu_pre_explicit = 0;
    int64_t vram_pre_explicit = 0;
    ok &= run_cpu_or_vram_case(moe_storage_mode::cpu, cpu_control.backend, cpu_oracle, inp, sel,
        0, 0, /*do_timing=*/false, cpu_once, cpu_pre, cpu_pre_cuda, cpu_pre_explicit);
    ok &= run_cpu_or_vram_case(moe_storage_mode::vram, vram_control.backend, vram_oracle, inp, sel,
        0, 0, /*do_timing=*/false, vram_once, vram_pre, vram_pre_cuda, vram_pre_explicit);

    const moe_error_metrics metrics =
        (ok && cpu_once.size() == (size_t)out_elems && vram_once.size() == (size_t)out_elems)
            ? compute_ffn_error(vram_once.data(), cpu_once.data(), (size_t)out_elems)
            : moe_error_metrics{ 0.0, 0.0, 0.0, false, false };

    // Phase 2: warmup + timed reps for both modes.
    moe_sample_stats cpu_stats;
    moe_sample_stats vram_stats;
    moe_sample_stats vram_cuda_stats;
    int64_t vram_explicit = 0;
    ok &= run_cpu_or_vram_case(moe_storage_mode::cpu, cpu_control.backend, cpu_oracle, inp, sel,
        params.warmup, params.reps, /*do_timing=*/true, cpu_once, cpu_stats, cpu_pre_cuda, cpu_pre_explicit);
    ok &= run_cpu_or_vram_case(moe_storage_mode::vram, vram_control.backend, vram_oracle, inp, sel,
        params.warmup, params.reps, /*do_timing=*/true, vram_once, vram_stats, vram_cuda_stats, vram_explicit);

    const bool passed = ok && metrics.passed;

    printf("  case %-5s %-8s : tokens=%lld selected=%lld unique=%lld\n",
        shape.name, selection_pattern_name(pattern),
        (long long)shape.n_tokens, (long long)sel.ids.size() / shape.n_tokens, (long long)sel.unique_experts);
    printf("    correctness (VRAM vs CPU): max_abs=%.6e mean_rel=%.6e nmse=%.6e finite=%s [%s]\n",
        metrics.max_abs_error, metrics.mean_relative_error, metrics.nmse,
        metrics.finite ? "yes" : "no", passed ? "PASS" : "FAIL");
    printf("    CPU  control: samples=%lld median=%8.1f us p95=%8.1f us mean=%8.1f us\n",
        (long long)params.reps, cpu_stats.median_us, cpu_stats.p95_us, cpu_stats.mean_us);
    printf("    VRAM control: samples=%lld median=%8.1f us p95=%8.1f us mean=%8.1f us\n",
        (long long)params.reps, vram_stats.median_us, vram_stats.p95_us, vram_stats.mean_us);

    cpu_result.mode = moe_storage_mode::cpu;
    cpu_result.shape_name = shape.name;
    cpu_result.pattern_name = selection_pattern_name(pattern);
    cpu_result.n_tokens = shape.n_tokens;
    cpu_result.n_selected_experts = shape.n_selected_experts;
    cpu_result.unique_experts = sel.unique_experts;
    cpu_result.samples = params.reps;
    cpu_result.wall = cpu_stats;
    cpu_result.cuda = cpu_pre_cuda; // zeros: CPU mode creates no events
    cpu_result.cpu_median_us = cpu_stats.median_us; // CPU row: its own median
    cpu_result.explicit_sync_count = cpu_pre_explicit; // 0: no events in CPU mode
    cpu_result.backend_wide_sync_count = 0;
    cpu_result.registration_us = 0.0;
    cpu_result.direct_registered_bytes = 0;
    cpu_result.staging_bytes = 0;
    cpu_result.timed_expert_weight_h2d_bytes = 0;
    cpu_result.timed_expert_weight_d2h_bytes = 0;
    cpu_result.mapping_kind = moe_mapping_kind::unsupported; // JSON: "none"
    cpu_result.mapped_host_status = "cpu_mode";
    cpu_result.setup_expert_weight_h2d_bytes = cpu_control.setup_h2d_bytes;
    cpu_result.max_abs_error = 0.0; // CPU is the reference for the comparison
    cpu_result.mean_relative_error = 0.0;
    cpu_result.nmse = 0.0;
    cpu_result.correctness_pass = passed;

    vram_result.mode = moe_storage_mode::vram;
    vram_result.shape_name = shape.name;
    vram_result.pattern_name = selection_pattern_name(pattern);
    vram_result.n_tokens = shape.n_tokens;
    vram_result.n_selected_experts = shape.n_selected_experts;
    vram_result.unique_experts = sel.unique_experts;
    vram_result.samples = params.reps;
    vram_result.wall = vram_stats;
    vram_result.cuda = vram_cuda_stats;
    vram_result.cpu_median_us = cpu_stats.median_us; // paired CPU control median
    vram_result.explicit_sync_count = vram_explicit;
    vram_result.backend_wide_sync_count = 0;
    vram_result.registration_us = 0.0;
    vram_result.direct_registered_bytes = 0;
    vram_result.staging_bytes = 0;
    vram_result.timed_expert_weight_h2d_bytes = 0;
    vram_result.timed_expert_weight_d2h_bytes = 0;
    vram_result.mapping_kind = moe_mapping_kind::unsupported; // JSON: "none"
    vram_result.mapped_host_status = "vram_mode";
    vram_result.setup_expert_weight_h2d_bytes = vram_control.setup_h2d_bytes;
    vram_result.max_abs_error = metrics.max_abs_error;
    vram_result.mean_relative_error = metrics.mean_relative_error;
    vram_result.nmse = metrics.nmse;
    vram_result.correctness_pass = passed;

    free_ffn_oracle(vram_oracle);
    free_ffn_oracle(cpu_oracle);
    return passed;
}

// ---------------------------------------------------------------------------
// Mapped-host case execution (Task 4)
// ---------------------------------------------------------------------------

// Runs one correctness pass and (when do_timing is set) the warmup and timed
// loop for the mapped-host oracle: inputs on CUDA, expert weights read by the
// kernels in place from host RAM through the mapped device pointers (zero H2D).
// The correctness run is the first CUDA compute on the mapped pointers; a CUDA
// error there (or in warmup) is caught via cudaGetLastError after the per-call
// synchronize inside ggml_backend_graph_compute, reported as unsupported with
// diagnostics, and stops the case. The check is deliberately NOT repeated
// inside the timed loop so the Task 5 timing stays clean (the correctness and
// warmup gates already failed fast on any mapping error). Task 5: timed
// samples use one persistent event pair around each graph compute; only the
// completion event is synchronized per sample (explicit_sync_count++), never
// ggml_backend_synchronize() (backend_wide_sync_count stays 0).
static bool run_mapped_host_case(
        ggml_backend_t backend,
        const moe_ffn_oracle & oracle,
        const std::vector<float> & inp,
        const moe_expert_selection & sel,
        int64_t warmup,
        int64_t reps,
        bool do_timing,
        std::vector<float> & out_once,
        moe_case_result & result) {
    result.wall = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    result.cuda = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    result.explicit_sync_count = 0;
    result.backend_wide_sync_count = 0;

    ggml_backend_tensor_set(oracle.inp, inp.data(), 0, inp.size() * sizeof(float));
    ggml_backend_tensor_set(oracle.ids, sel.ids.data(), 0, sel.ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(oracle.route_weights, sel.weights.data(), 0, sel.weights.size() * sizeof(float));

    // Correctness run (the first warmup call).
    if (ggml_backend_graph_compute(backend, oracle.graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: mapped-host graph compute failed during the correctness run\n");
        result.mapped_host_status = "unsupported";
        return false;
    }
    ggml_backend_synchronize(backend);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: mapped-host CUDA error after the correctness compute: %s\n",
            cudaGetErrorString(err));
        result.mapped_host_status = "unsupported";
        return false;
    }
    const size_t out_elems = ggml_nelements(oracle.output);
    out_once.resize(out_elems);
    ggml_backend_tensor_get(oracle.output, out_once.data(), 0, out_elems * sizeof(float));

    if (!do_timing) {
        return true;
    }

    // One persistent event pair for this (mode, case), alive for the whole
    // warmup + timed loop.
    const ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_event_t ev_start = ggml_backend_event_new(dev);
    ggml_backend_event_t ev_end = ggml_backend_event_new(dev);
    if (ev_start == nullptr || ev_end == nullptr || g_event_elapsed_us == nullptr) {
        fprintf(stderr, "ERROR: cannot create CUDA timing events for the mapped-host control\n");
        if (ev_start != nullptr) {
            ggml_backend_event_free(ev_start);
        }
        if (ev_end != nullptr) {
            ggml_backend_event_free(ev_end);
        }
        result.mapped_host_status = "unsupported";
        return false;
    }

    for (int64_t i = 0; i < warmup; i++) {
        ggml_backend_graph_compute(backend, oracle.graph);
    }
    ggml_backend_synchronize(backend); // warmup drain; outside the timed loop
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: mapped-host CUDA error during warmup: %s\n",
            cudaGetErrorString(err));
        ggml_backend_event_free(ev_start);
        ggml_backend_event_free(ev_end);
        result.mapped_host_status = "unsupported";
        return false;
    }

    std::vector<double> samples;
    samples.reserve((size_t) reps);
    std::vector<double> cuda_samples;
    cuda_samples.reserve((size_t) reps);
    for (int64_t i = 0; i < reps; i++) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_event_record(ev_start, backend);
        ggml_backend_graph_compute(backend, oracle.graph);
        ggml_backend_event_record(ev_end, backend);
        ggml_backend_event_synchronize(ev_end); // completion wait; explicit per sample
        result.explicit_sync_count++;
        const int64_t t1 = ggml_time_us();
        samples.push_back((double) (t1 - t0));
        uint64_t elapsed_us = 0;
        if (!g_event_elapsed_us(ev_start, ev_end, &elapsed_us)) {
            fprintf(stderr, "ERROR: ggml_backend_event_elapsed_us failed for the mapped-host control sample %lld\n",
                (long long) i);
            ggml_backend_event_free(ev_start);
            ggml_backend_event_free(ev_end);
            result.mapped_host_status = "unsupported";
            return false;
        }
        cuda_samples.push_back((double) elapsed_us);
    }
    result.wall = compute_sample_stats(samples);
    result.cuda = compute_sample_stats(cuda_samples);
    ggml_backend_event_free(ev_start);
    ggml_backend_event_free(ev_end);
    return true;
}

// Runs one (shape, pattern) mapped-host case: CPU reference run first, then
// the mapped oracle, the correctness comparison, then warmup + timed reps for
// both. NEVER falls back to VRAM compute for the mapped case: a CUDA error on
// the mapped path emits unsupported diagnostics, stops the case, and the
// process exits nonzero.
static bool run_mapped_case(
        const qwen_apex_expert_bundle & bundle,
        const moe_ffn_control & cpu_control,
        const moe_ffn_control & mapped_control,
        const moe_shape_case & shape,
        selection_pattern pattern,
        const bench_params & params,
        moe_case_result & mapped_result) {
    const moe_expert_selection sel = make_expert_selection(shape, pattern);
    const int64_t n_embd = bundle.gate->ne[0];
    const int64_t out_elems = n_embd * shape.n_tokens;

    std::vector<float> inp((size_t) out_elems);
    fill_deterministic_f32(inp);

    moe_ffn_oracle cpu_oracle;
    moe_ffn_oracle mapped_oracle;
    if (!build_ffn_oracle(bundle, cpu_control, shape, sel, cpu_oracle)) {
        return false;
    }
    if (!build_ffn_oracle(bundle, mapped_control, shape, sel, mapped_oracle)) {
        free_ffn_oracle(cpu_oracle);
        return false;
    }

    bool ok = true;

    // Phase 1: one CPU reference run and one mapped run before any timing.
    std::vector<float> cpu_once;
    std::vector<float> mapped_once;
    moe_sample_stats cpu_pre;
    moe_sample_stats cpu_pre_cuda;
    int64_t cpu_pre_explicit = 0;
    ok &= run_cpu_or_vram_case(moe_storage_mode::cpu, cpu_control.backend, cpu_oracle, inp, sel,
        0, 0, /*do_timing=*/false, cpu_once, cpu_pre, cpu_pre_cuda, cpu_pre_explicit);
    ok &= run_mapped_host_case(mapped_control.backend, mapped_oracle, inp, sel,
        0, 0, /*do_timing=*/false, mapped_once, mapped_result);

    const moe_error_metrics metrics =
        (ok && cpu_once.size() == (size_t) out_elems && mapped_once.size() == (size_t) out_elems)
            ? compute_ffn_error(mapped_once.data(), cpu_once.data(), (size_t) out_elems)
            : moe_error_metrics{ 0.0, 0.0, 0.0, false, false };

    // Phase 2: warmup + timed reps for both; skipped entirely when the mapped
    // path already failed (stop the case).
    bool timed_ran = false;
    moe_sample_stats cpu_stats;
    if (ok) {
        timed_ran = true;
        ok &= run_cpu_or_vram_case(moe_storage_mode::cpu, cpu_control.backend, cpu_oracle, inp, sel,
            params.warmup, params.reps, /*do_timing=*/true, cpu_once, cpu_stats, cpu_pre_cuda, cpu_pre_explicit);
        ok &= run_mapped_host_case(mapped_control.backend, mapped_oracle, inp, sel,
            params.warmup, params.reps, /*do_timing=*/true, mapped_once, mapped_result);
    }

    const bool passed = ok && metrics.passed;
    int64_t mapped_direct_bytes = 0;
    int64_t mapped_staging_bytes = 0;
    for (int64_t i = 0; i < mapped_control.mapping_count; i++) {
        mapped_direct_bytes += (int64_t) mapped_control.mappings[i].registered_bytes;
        mapped_staging_bytes += (int64_t) mapped_control.mappings[i].staging_bytes;
    }

    printf("  case %-5s %-8s : tokens=%lld selected=%lld unique=%lld\n",
        shape.name, selection_pattern_name(pattern),
        (long long) shape.n_tokens, (long long) sel.ids.size() / shape.n_tokens, (long long) sel.unique_experts);
    printf("    correctness (mapped vs CPU): max_abs=%.6e mean_rel=%.6e nmse=%.6e finite=%s [%s]\n",
        metrics.max_abs_error, metrics.mean_relative_error, metrics.nmse,
        metrics.finite ? "yes" : "no", passed ? "PASS" : "FAIL");
    printf("    CPU      control: samples=%lld median=%8.1f us p95=%8.1f us mean=%8.1f us\n",
        (long long) params.reps, cpu_stats.median_us, cpu_stats.p95_us, cpu_stats.mean_us);
    printf("    MAPPED   control: kind=%s registration_us=%.0f direct_bytes=%lld staging_bytes=%lld samples=%lld median=%8.1f us p95=%8.1f us mean=%8.1f us\n",
        mapping_kind_name(mapped_control.mapping_count > 0 ? mapped_control.mappings[0].kind : moe_mapping_kind::unsupported),
        mapped_control.registration_us,
        (long long) mapped_direct_bytes, (long long) mapped_staging_bytes,
        (long long) params.reps,
        mapped_result.wall.median_us, mapped_result.wall.p95_us, mapped_result.wall.mean_us);

    mapped_result.mode = moe_storage_mode::mapped_host;
    mapped_result.shape_name = shape.name;
    mapped_result.pattern_name = selection_pattern_name(pattern);
    mapped_result.n_tokens = shape.n_tokens;
    mapped_result.n_selected_experts = shape.n_selected_experts;
    mapped_result.unique_experts = sel.unique_experts;
    mapped_result.samples = timed_ran ? params.reps : 0;
    // wall/cuda/explicit_sync_count are written by run_mapped_host_case into
    // mapped_result directly (the timed path); do not clobber them here.
    mapped_result.setup_expert_weight_h2d_bytes = mapped_control.setup_h2d_bytes;
    mapped_result.max_abs_error = metrics.max_abs_error;
    mapped_result.mean_relative_error = metrics.mean_relative_error;
    mapped_result.nmse = metrics.nmse;
    mapped_result.correctness_pass = passed;
    mapped_result.mapping_kind = mapped_control.mapping_count > 0
        ? mapped_control.mappings[0].kind : moe_mapping_kind::unsupported;
    mapped_result.registration_us = mapped_control.registration_us;
    mapped_result.direct_registered_bytes = mapped_direct_bytes;
    mapped_result.staging_bytes = mapped_staging_bytes;
    mapped_result.timed_expert_weight_h2d_bytes = 0;
    mapped_result.timed_expert_weight_d2h_bytes = 0;
    mapped_result.cpu_median_us = cpu_stats.median_us;
    mapped_result.backend_wide_sync_count = 0;
    mapped_result.mapped_host_status =
        (mapped_control.mapping_count > 0 && mapped_control.mappings[0].is_usable()) ? "ok" : "unsupported";

    // Task 4/5 invariants: registrations and weight copies happen once during
    // setup, so the timed telemetry must stay zero (the schema assert re-checks
    // this before any JSON row is written).
    if (mapped_result.timed_expert_weight_h2d_bytes != 0
            || mapped_result.timed_expert_weight_d2h_bytes != 0
            || mapped_result.backend_wide_sync_count != 0) {
        fprintf(stderr, "ERROR: mapped-host timed telemetry invariant violated "
            "(timed_expert_weight_h2d_bytes=%lld timed_expert_weight_d2h_bytes=%lld backend_wide_sync_count=%lld)\n",
            (long long) mapped_result.timed_expert_weight_h2d_bytes,
            (long long) mapped_result.timed_expert_weight_d2h_bytes,
            (long long) mapped_result.backend_wide_sync_count);
        ok = false;
    }

    free_ffn_oracle(mapped_oracle);
    free_ffn_oracle(cpu_oracle);
    return passed;
}

// Runs one (shape, pattern) case on a single control (--mode cpu / --mode
// vram). When cpu_ref is provided (vram mode) the output is numerically
// compared against the CPU reference; otherwise (cpu mode) the case passes by
// successful compute.
static bool run_single_mode_case(
        const qwen_apex_expert_bundle & bundle,
        const moe_ffn_control & control,
        const moe_ffn_control * cpu_ref,
        const moe_shape_case & shape,
        selection_pattern pattern,
        const bench_params & params,
        moe_case_result & result) {
    const moe_expert_selection sel = make_expert_selection(shape, pattern);
    const int64_t n_embd = bundle.gate->ne[0];
    const int64_t out_elems = n_embd * shape.n_tokens;

    std::vector<float> inp((size_t) out_elems);
    fill_deterministic_f32(inp);

    moe_ffn_oracle oracle;
    if (!build_ffn_oracle(bundle, control, shape, sel, oracle)) {
        return false;
    }
    moe_ffn_oracle ref_oracle;
    if (cpu_ref != nullptr && !build_ffn_oracle(bundle, *cpu_ref, shape, sel, ref_oracle)) {
        free_ffn_oracle(oracle);
        return false;
    }

    bool ok = true;
    std::vector<float> out_once;
    std::vector<float> ref_once;
    moe_sample_stats pre;
    moe_sample_stats ref_pre;
    moe_sample_stats pre_cuda;
    moe_sample_stats ref_pre_cuda;
    int64_t pre_explicit = 0;
    int64_t ref_pre_explicit = 0;
    ok &= run_cpu_or_vram_case(control.mode, control.backend, oracle, inp, sel,
        0, 0, /*do_timing=*/false, out_once, pre, pre_cuda, pre_explicit);
    if (cpu_ref != nullptr) {
        ok &= run_cpu_or_vram_case(moe_storage_mode::cpu, cpu_ref->backend, ref_oracle, inp, sel,
            0, 0, /*do_timing=*/false, ref_once, ref_pre, ref_pre_cuda, ref_pre_explicit);
    }

    const moe_error_metrics metrics =
        (cpu_ref != nullptr && ok && out_once.size() == (size_t) out_elems && ref_once.size() == (size_t) out_elems)
            ? compute_ffn_error(out_once.data(), ref_once.data(), (size_t) out_elems)
            : moe_error_metrics{ 0.0, 0.0, 0.0, true, true };

    moe_sample_stats stats;
    moe_sample_stats cuda_stats;
    int64_t explicit_sync_count = 0;
    ok &= run_cpu_or_vram_case(control.mode, control.backend, oracle, inp, sel,
        params.warmup, params.reps, /*do_timing=*/true, out_once, stats, cuda_stats, explicit_sync_count);

    const bool passed = ok && metrics.passed;

    printf("  case %-5s %-8s : tokens=%lld selected=%lld unique=%lld\n",
        shape.name, selection_pattern_name(pattern),
        (long long) shape.n_tokens, (long long) sel.ids.size() / shape.n_tokens, (long long) sel.unique_experts);
    if (cpu_ref != nullptr) {
        printf("    correctness (%s vs CPU): max_abs=%.6e mean_rel=%.6e nmse=%.6e finite=%s [%s]\n",
            storage_mode_name(control.mode),
            metrics.max_abs_error, metrics.mean_relative_error, metrics.nmse,
            metrics.finite ? "yes" : "no", passed ? "PASS" : "FAIL");
    } else {
        printf("    correctness: single-mode run (no cross-mode comparison) [%s]\n",
            passed ? "PASS" : "FAIL");
    }
    printf("    %-6s control: samples=%lld median=%8.1f us p95=%8.1f us mean=%8.1f us\n",
        storage_mode_name(control.mode),
        (long long) params.reps, stats.median_us, stats.p95_us, stats.mean_us);

    result.mode = control.mode;
    result.shape_name = shape.name;
    result.pattern_name = selection_pattern_name(pattern);
    result.n_tokens = shape.n_tokens;
    result.n_selected_experts = shape.n_selected_experts;
    result.unique_experts = sel.unique_experts;
    result.samples = params.reps;
    result.wall = stats;
    result.cuda = cuda_stats;
    result.cpu_median_us = (control.mode == moe_storage_mode::cpu) ? stats.median_us : 0.0;
    result.explicit_sync_count = explicit_sync_count;
    result.backend_wide_sync_count = 0;
    result.registration_us = 0.0;
    result.direct_registered_bytes = 0;
    result.staging_bytes = 0;
    result.timed_expert_weight_h2d_bytes = 0;
    result.timed_expert_weight_d2h_bytes = 0;
    result.mapping_kind = moe_mapping_kind::unsupported; // JSON: "none"
    result.mapped_host_status = (control.mode == moe_storage_mode::cpu) ? "cpu_mode" : "vram_mode";
    result.setup_expert_weight_h2d_bytes = control.setup_h2d_bytes;
    result.max_abs_error = metrics.max_abs_error;
    result.mean_relative_error = metrics.mean_relative_error;
    result.nmse = metrics.nmse;
    result.correctness_pass = passed;

    free_ffn_oracle(ref_oracle);
    free_ffn_oracle(oracle);
    return passed;
}

// Result record for a mapped case that could not run at all (eligibility
// probe or control init failed): unsupported status, no samples.
static void fill_unsupported_mapped_result(
        moe_case_result & result,
        const moe_shape_case & shape,
        selection_pattern pattern,
        const moe_expert_selection & sel) {
    result.mode = moe_storage_mode::mapped_host;
    result.shape_name = shape.name;
    result.pattern_name = selection_pattern_name(pattern);
    result.n_tokens = shape.n_tokens;
    result.n_selected_experts = shape.n_selected_experts;
    result.unique_experts = sel.unique_experts;
    result.samples = 0;
    result.mapping_kind = moe_mapping_kind::unsupported;
    result.registration_us = 0.0;
    result.direct_registered_bytes = 0;
    result.staging_bytes = 0;
    result.timed_expert_weight_h2d_bytes = 0;
    result.timed_expert_weight_d2h_bytes = 0;
    result.explicit_sync_count = 0;
    result.backend_wide_sync_count = 0;
    result.mapped_host_status = "unsupported";
    result.correctness_pass = false;
}

// ---------------------------------------------------------------------------
// Task 5: Gate 1 evaluation (distinct top-8 basis)
// ---------------------------------------------------------------------------

// Per-shape medians/P95 from the distinct top-8 rows (pattern "distinct",
// n_selected_experts == 8), one entry per known shape present in the run.
struct gate_one_shape_entry {
    std::string shape;
    bool has_cpu = false;
    double cpu_median_us = 0.0;
    double cpu_p95_us = 0.0;
    bool has_vram = false;
    double vram_median_us = 0.0;
    double vram_p95_us = 0.0;
    bool has_mapped = false;
    double mapped_median_us = 0.0;
    double mapped_p95_us = 0.0;
};

struct gate_one_report {
    std::string status = "negative"; // strong | conditional | equivalent | negative | unsupported
    std::string mapping_kind = "none";
    int64_t timed_expert_weight_h2d_bytes_total = 0;
    std::vector<gate_one_shape_entry> per_shape;
};

// Returns the distinct top-8 result row for (shape, mode), or nullptr.
static const moe_case_result * find_distinct_row(
        const std::vector<moe_case_result> & results,
        const char * shape,
        moe_storage_mode mode) {
    for (const moe_case_result & r : results) {
        if (r.shape_name == shape && r.mode == mode
                && r.pattern_name == "distinct" && r.n_selected_experts == 8) {
            return &r;
        }
    }
    return nullptr;
}

// Gate 1: is the mapped-host path fast enough to ship? Evaluated on the
// distinct top-8 rows only:
//   strong      = mapped tg1 beats CPU tg1 on median AND p95, timed H2D == 0
//   equivalent  = mapped tg1 within 5% of CPU tg1 on median AND p95
//   batched_win = any of {tg4, tg8, pp128, pp512} mapped median <= 90% of CPU
//   conditional = equivalent && batched_win && timed H2D == 0
//   unsupported = no mapped-host row ran (or all mapped_host_status == "unsupported")
//   otherwise   = negative
static gate_one_report evaluate_gate_one(const std::vector<moe_case_result> & results) {
    gate_one_report report;

    // Per-shape table in known-shape order; only shapes with rows appear.
    for (int64_t s = 0; s < known_shape_count; s++) {
        const moe_case_result * cpu_row = find_distinct_row(results, known_shapes[s].name, moe_storage_mode::cpu);
        const moe_case_result * vram_row = find_distinct_row(results, known_shapes[s].name, moe_storage_mode::vram);
        const moe_case_result * mapped_row = find_distinct_row(results, known_shapes[s].name, moe_storage_mode::mapped_host);
        if (cpu_row == nullptr && vram_row == nullptr && mapped_row == nullptr) {
            continue;
        }
        gate_one_shape_entry entry;
        entry.shape = known_shapes[s].name;
        if (cpu_row != nullptr) {
            entry.has_cpu = true;
            entry.cpu_median_us = cpu_row->wall.median_us;
            entry.cpu_p95_us = cpu_row->wall.p95_us;
        }
        if (vram_row != nullptr) {
            entry.has_vram = true;
            entry.vram_median_us = vram_row->wall.median_us;
            entry.vram_p95_us = vram_row->wall.p95_us;
        }
        if (mapped_row != nullptr) {
            entry.has_mapped = true;
            entry.mapped_median_us = mapped_row->wall.median_us;
            entry.mapped_p95_us = mapped_row->wall.p95_us;
            report.timed_expert_weight_h2d_bytes_total += mapped_row->timed_expert_weight_h2d_bytes;
            if (report.mapping_kind == "none" && mapped_row->mapped_host_status == "ok") {
                report.mapping_kind = mapping_kind_name(mapped_row->mapping_kind);
            }
        }
        report.per_shape.push_back(entry);
    }

    // Unsupported: no mapped-host row ran, or every mapped-host row is
    // unsupported (probe/registration failure or a CUDA error on the path).
    bool any_mapped_row = false;
    bool any_mapped_ok = false;
    for (const moe_case_result & r : results) {
        if (r.mode != moe_storage_mode::mapped_host
                || r.pattern_name != "distinct" || r.n_selected_experts != 8) {
            continue;
        }
        any_mapped_row = true;
        if (r.mapped_host_status == "ok") {
            any_mapped_ok = true;
        }
    }
    if (!any_mapped_row || !any_mapped_ok) {
        report.status = "unsupported";
        return report;
    }

    const moe_case_result * cpu_tg1 = find_distinct_row(results, "tg1", moe_storage_mode::cpu);
    const moe_case_result * mapped_tg1 = find_distinct_row(results, "tg1", moe_storage_mode::mapped_host);

    const bool strong = mapped_tg1 != nullptr && cpu_tg1 != nullptr
        && mapped_tg1->timed_expert_weight_h2d_bytes == 0
        && mapped_tg1->wall.median_us < cpu_tg1->wall.median_us
        && mapped_tg1->wall.p95_us < cpu_tg1->wall.p95_us;

    const bool equivalent = mapped_tg1 != nullptr && cpu_tg1 != nullptr
        && mapped_tg1->wall.median_us <= cpu_tg1->wall.median_us * 1.05
        && mapped_tg1->wall.p95_us <= cpu_tg1->wall.p95_us * 1.05;

    // batched_win: any of {tg4, tg8, pp128, pp512} where the mapped median is
    // at least 10% faster than the CPU median (<= 90% of it).
    bool batched_win = false;
    static const char * batched_shapes[] = { "tg4", "tg8", "pp128", "pp512" };
    for (size_t b = 0; b < sizeof(batched_shapes) / sizeof(batched_shapes[0]); b++) {
        const moe_case_result * m = find_distinct_row(results, batched_shapes[b], moe_storage_mode::mapped_host);
        const moe_case_result * c = find_distinct_row(results, batched_shapes[b], moe_storage_mode::cpu);
        if (m != nullptr && c != nullptr && m->wall.median_us <= c->wall.median_us * 0.90) {
            batched_win = true;
            break;
        }
    }

    const bool h2d_zero = mapped_tg1 != nullptr && mapped_tg1->timed_expert_weight_h2d_bytes == 0;
    if (strong) {
        report.status = "strong";
    } else if (equivalent && batched_win && h2d_zero) {
        report.status = "conditional";
    } else if (equivalent) {
        report.status = "equivalent";
    } else {
        report.status = "negative";
    }
    return report;
}

// Human Gate 1 summary; the final stdout line is exactly "Gate 1: <status>".
static void print_gate_one_summary(const gate_one_report & report) {
    printf("--------------------------------------------------------------------------------\n");
    printf("Gate 1 summary (basis: distinct_top8)\n");
    printf("  %-6s %11s %11s %11s %11s %11s %11s\n",
        "shape", "cpu_med_us", "cpu_p95_us", "vram_med_us", "vram_p95_us", "map_med_us", "map_p95_us");
    for (const gate_one_shape_entry & e : report.per_shape) {
        printf("  %-6s %11.1f %11.1f %11.1f %11.1f %11.1f %11.1f\n",
            e.shape.c_str(),
            e.has_cpu ? e.cpu_median_us : 0.0,
            e.has_cpu ? e.cpu_p95_us : 0.0,
            e.has_vram ? e.vram_median_us : 0.0,
            e.has_vram ? e.vram_p95_us : 0.0,
            e.has_mapped ? e.mapped_median_us : 0.0,
            e.has_mapped ? e.mapped_p95_us : 0.0);
    }
    printf("  mapping_kind: %s | timed_expert_weight_h2d_bytes total: %lld\n",
        report.mapping_kind.c_str(), (long long) report.timed_expert_weight_h2d_bytes_total);
    printf("Gate 1: %s\n", report.status.c_str());
}

// ---------------------------------------------------------------------------
// JSON output (Task 5 full result schema + gate_one)
// ---------------------------------------------------------------------------

static bool ensure_parent_dirs(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos || slash == 0) {
        return true;
    }
    const std::string dir = path.substr(0, slash);
#if defined(_WIN32)
    size_t pos = 0;
    std::string cur;
    while (pos <= dir.size()) {
        const size_t sep = dir.find_first_of("/\\", pos);
        const size_t end = (sep == std::string::npos) ? dir.size() : sep;
        cur += dir.substr(pos, end - pos);
        if (!cur.empty() && GetFileAttributesA(cur.c_str()) == INVALID_FILE_ATTRIBUTES) {
            if (!CreateDirectoryA(cur.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS) {
                return false;
            }
        }
        if (sep == std::string::npos) {
            break;
        }
        cur += '\\';
        pos = sep + 1;
    }
    return true;
#else
    for (size_t i = 1; i < dir.size(); i++) {
        if (dir[i] == '/') {
            const std::string part = dir.substr(0, i);
            if (!part.empty()) {
                mkdir(part.c_str(), 0755); // ignore EEXIST
            }
        }
    }
    mkdir(dir.c_str(), 0755); // ignore EEXIST
    return true;
#endif
}

static bool write_json_results(
        const char * path,
        const bench_params & params,
        const std::vector<moe_case_result> & results,
        const gate_one_report & gate) {
    // Schema assert BEFORE any file I/O: every mapped row must carry zero timed
    // expert-weight H2D bytes and zero backend-wide synchronizes inside the
    // timed loop. On violation we print, return false (main exits nonzero), and
    // the file is never opened, so nothing is written.
    for (const moe_case_result & r : results) {
        if (r.mode != moe_storage_mode::mapped_host) {
            continue;
        }
        if (r.timed_expert_weight_h2d_bytes != 0 || r.backend_wide_sync_count != 0) {
            fprintf(stderr,
                "ERROR: schema assert failed before writing %s: mapped row %s/%s has "
                "timed_expert_weight_h2d_bytes=%lld backend_wide_sync_count=%lld "
                "(both must be 0); JSON file NOT written\n",
                path, r.shape_name.c_str(), r.pattern_name.c_str(),
                (long long) r.timed_expert_weight_h2d_bytes,
                (long long) r.backend_wide_sync_count);
            return false;
        }
    }

    if (!ensure_parent_dirs(path)) {
        fprintf(stderr, "ERROR: cannot create parent directories for %s\n", path);
        return false;
    }
    FILE * f = fopen(path, "w");
    if (f == nullptr) {
        fprintf(stderr, "ERROR: cannot open %s for writing\n", path);
        return false;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"format\": 2,\n");
    fprintf(f, "  \"experiment\": \"low-vram-moe-mapped-host-oracle\",\n");
    fprintf(f, "  \"model\": \"");
    for (char c : params.model_path) {
        if (c == '\\' || c == '"') {
            fputc('\\', f);
        }
        fputc(c, f);
    }
    fprintf(f, "\",\n");
    fprintf(f, "  \"layer\": %lld,\n", (long long)params.layer);
    fprintf(f, "  \"warmup\": %lld,\n", (long long)params.warmup);
    fprintf(f, "  \"reps\": %lld,\n", (long long)params.reps);
    int64_t mapped_host_count = 0;
    for (const moe_case_result & r : results) {
        if (r.mode == moe_storage_mode::mapped_host) {
            mapped_host_count++;
        }
    }
    fprintf(f, "  \"mapped_host_results\": %lld,\n", (long long)mapped_host_count);
    fprintf(f, "  \"results\": [\n");
    for (size_t i = 0; i < results.size(); i++) {
        const moe_case_result & r = results[i];
        fprintf(f, "    {\n");
        fprintf(f, "      \"mode\": \"%s\",\n", storage_mode_name(r.mode));
        fprintf(f, "      \"mapping_kind\": \"%s\",\n",
            r.mode == moe_storage_mode::mapped_host ? mapping_kind_name(r.mapping_kind) : "none");
        fprintf(f, "      \"shape\": \"%s\",\n", r.shape_name.c_str());
        fprintf(f, "      \"pattern\": \"%s\",\n", r.pattern_name.c_str());
        fprintf(f, "      \"n_tokens\": %lld,\n", (long long)r.n_tokens);
        fprintf(f, "      \"n_selected_experts\": %lld,\n", (long long)r.n_selected_experts);
        fprintf(f, "      \"unique_experts\": %lld,\n", (long long)r.unique_experts);
        fprintf(f, "      \"samples\": %lld,\n", (long long)r.samples);
        fprintf(f, "      \"wall_min_us\": %.3f,\n", r.wall.min_us);
        fprintf(f, "      \"wall_median_us\": %.3f,\n", r.wall.median_us);
        fprintf(f, "      \"wall_p95_us\": %.3f,\n", r.wall.p95_us);
        fprintf(f, "      \"wall_mean_us\": %.3f,\n", r.wall.mean_us);
        fprintf(f, "      \"wall_stddev_us\": %.3f,\n", r.wall.stddev_us);
        fprintf(f, "      \"cuda_median_us\": %.3f,\n", r.cuda.median_us);
        fprintf(f, "      \"cuda_p95_us\": %.3f,\n", r.cuda.p95_us);
        fprintf(f, "      \"cpu_median_us\": %.3f,\n", r.cpu_median_us);
        fprintf(f, "      \"registration_us\": %.3f,\n", r.registration_us);
        fprintf(f, "      \"direct_registered_bytes\": %lld,\n", (long long)r.direct_registered_bytes);
        fprintf(f, "      \"staging_bytes\": %lld,\n", (long long)r.staging_bytes);
        fprintf(f, "      \"setup_expert_weight_h2d_bytes\": %lld,\n", (long long)r.setup_expert_weight_h2d_bytes);
        fprintf(f, "      \"timed_expert_weight_h2d_bytes\": %lld,\n", (long long)r.timed_expert_weight_h2d_bytes);
        fprintf(f, "      \"timed_expert_weight_d2h_bytes\": %lld,\n", (long long)r.timed_expert_weight_d2h_bytes);
        fprintf(f, "      \"explicit_sync_count\": %lld,\n", (long long)r.explicit_sync_count);
        fprintf(f, "      \"backend_wide_sync_count\": %lld,\n", (long long)r.backend_wide_sync_count);
        fprintf(f, "      \"max_abs_error\": %.9g,\n", r.max_abs_error);
        fprintf(f, "      \"mean_relative_error\": %.9g,\n", r.mean_relative_error);
        fprintf(f, "      \"nmse\": %.9g,\n", r.nmse);
        fprintf(f, "      \"mapped_host_status\": \"%s\",\n", r.mapped_host_status.c_str());
        fprintf(f, "      \"correctness_pass\": %s\n", r.correctness_pass ? "true" : "false");
        fprintf(f, "    }%s\n", (i + 1 < results.size()) ? "," : "");
    }
    fprintf(f, "  ],\n");
    fprintf(f, "  \"gate_one\": {\n");
    fprintf(f, "    \"status\": \"%s\",\n", gate.status.c_str());
    fprintf(f, "    \"basis\": \"distinct_top8\",\n");
    fprintf(f, "    \"mapping_kind\": \"%s\",\n", gate.mapping_kind.c_str());
    fprintf(f, "    \"timed_expert_weight_h2d_bytes\": %lld,\n",
        (long long) gate.timed_expert_weight_h2d_bytes_total);
    fprintf(f, "    \"per_shape\": [\n");
    for (size_t s = 0; s < gate.per_shape.size(); s++) {
        const gate_one_shape_entry & e = gate.per_shape[s];
        fprintf(f, "      {\n");
        fprintf(f, "        \"shape\": \"%s\",\n", e.shape.c_str());
        if (e.has_cpu) {
            fprintf(f, "        \"cpu\": { \"wall_median_us\": %.3f, \"wall_p95_us\": %.3f },\n",
                e.cpu_median_us, e.cpu_p95_us);
        }
        if (e.has_vram) {
            fprintf(f, "        \"vram\": { \"wall_median_us\": %.3f, \"wall_p95_us\": %.3f },\n",
                e.vram_median_us, e.vram_p95_us);
        }
        if (e.has_mapped) {
            fprintf(f, "        \"mapped\": { \"wall_median_us\": %.3f, \"wall_p95_us\": %.3f }\n",
                e.mapped_median_us, e.mapped_p95_us);
        }
        fprintf(f, "      }%s\n", (s + 1 < gate.per_shape.size()) ? "," : "");
    }
    fprintf(f, "    ]\n");
    fprintf(f, "  }\n");
    fprintf(f, "}\n");

    if (fclose(f) != 0) {
        fprintf(stderr, "ERROR: failed to flush %s\n", path);
        return false;
    }
    printf("Wrote %zu result record(s) to %s\n", results.size(), path);
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    bench_params params;
    if (!parse_args(argc, argv, params)) {
        return 1;
    }
    if (!validate_params(params)) {
        return 1;
    }

    if (params.self_test) {
        return run_self_test() ? 0 : 1;
    }
    // Mapped registration requires the device flags to include cudaDeviceMapHost.
    // The flag must be set before any CUDA context exists in this process, so
    // this runs before llama_backend_init(). Any error is reported, not fatal:
    // the eligibility probe and per-tensor registration failures still degrade
    // gracefully to staging / unsupported.
    const cudaError_t set_flags_err = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (set_flags_err != cudaSuccess) {
        (void)cudaGetLastError();
        printf("  note: cudaSetDeviceFlags(cudaDeviceMapHost) failed: %s "
            "(continuing; mapped registration may be unavailable)\n",
            cudaGetErrorString(set_flags_err));
    }
    (void)cudaGetDevice(&g_mapped_device);

    llama_backend_init();

    // Real-model harness initialization (mirrors tests/test-moe-heterogeneous-bench.cpp
    // and tests/test-moe-geometry-report.cpp): mmap-enabled load, 14 CPU threads,
    // n_gpu_layers = 99, and MoE expert tensors forced to CPU host buffers.
    // fit_params is disabled so n_gpu_layers = 99 applies literally.
    common_params cparams;
    cparams.model.path = params.model_path;
    cparams.load_mode = LLAMA_LOAD_MODE_MMAP;
    cparams.n_gpu_layers = 99;
    cparams.fit_params = false;
    cparams.cpuparams.n_threads = 14;
    cparams.cpuparams_batch.n_threads = 14;
    // common_model_params_to_llama() requires a {nullptr, nullptr}-terminated
    // override list outside fit mode
    cparams.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
    cparams.tensor_buft_overrides.push_back({nullptr, nullptr});

    common_init_result_ptr init = common_init_from_params(cparams, /*model_only=*/ true);
    if (!init || !init->model()) {
        fprintf(stderr, "ERROR: failed to initialize model from %s\n", params.model_path.c_str());
        llama_backend_free();
        return 1;
    }
    struct llama_model * model = init->model();

    const int32_t n_layers = llama_model_n_layer(model);
    if (params.layer >= n_layers) {
        fprintf(stderr, "ERROR: --layer %lld out of range; model has %d layers\n",
            (long long)params.layer, n_layers);
        llama_backend_free();
        return 1;
    }

    qwen_apex_expert_bundle bundle;
    if (!load_qwen_apex_bundle(model, params.layer, bundle)) {
        llama_backend_free();
        return 1;
    }

    if (params.inspect) {
        printf("Inspect OK: layer %lld bundle resolved and host-resident\n", (long long)params.layer);
        llama_backend_free();
        return 0;
    }

    // Backends: CPU control and CUDA (VRAM) control.
    ggml_backend_t backend_cpu = ggml_backend_cpu_init();
    if (backend_cpu == nullptr) {
        fprintf(stderr, "ERROR: failed to initialize the CPU backend\n");
        llama_backend_free();
        return 1;
    }
    ggml_backend_cpu_set_n_threads(backend_cpu, 14);

    ggml_backend_dev_t dev_gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            dev_gpu = dev;
            break;
        }
    }
    if (dev_gpu == nullptr) {
        fprintf(stderr, "ERROR: no GPU backend device found\n");
        ggml_backend_free(backend_cpu);
        llama_backend_free();
        return 1;
    }
    ggml_backend_t backend_vram = ggml_backend_dev_init(dev_gpu, nullptr);
    if (backend_vram == nullptr) {
        fprintf(stderr, "ERROR: failed to initialize the GPU backend (%s)\n", ggml_backend_dev_name(dev_gpu));
        ggml_backend_free(backend_cpu);
        llama_backend_free();
        return 1;
    }
    printf("GPU device: %s (%s)\n", ggml_backend_dev_name(dev_gpu), ggml_backend_dev_description(dev_gpu));
    // Selected storage modes. The default (cpu_vram) keeps the Task 3
    // behavior; 'all' runs cpu, vram, and mapped_host in order. The mapped
    // path needs the CPU control as the correctness reference and the CUDA
    // backend for the graph compute (weights bind via mapped device pointers).
    const bool run_cpu    = params.mode == moe_run_mode::cpu || params.mode == moe_run_mode::cpu_vram
        || params.mode == moe_run_mode::all;
    const bool run_vram   = params.mode == moe_run_mode::vram || params.mode == moe_run_mode::cpu_vram
        || params.mode == moe_run_mode::all;
    const bool run_mapped = params.mode == moe_run_mode::mapped_host || params.mode == moe_run_mode::all;

    // Task 5: resolve the CUDA event elapsed function once (proc-address API;
    // see ggml_backend_event_elapsed_us_t). Required whenever a CUDA-timed mode
    // runs: the timed loop measures with events and never uses
    // ggml_backend_synchronize(), so a missing implementation is fatal here.
    if (run_vram || run_mapped) {
        g_event_elapsed_us = (ggml_backend_event_elapsed_us_t)
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev_gpu), "ggml_backend_event_elapsed_us");
        if (g_event_elapsed_us == nullptr) {
            fprintf(stderr, "ERROR: CUDA backend does not provide ggml_backend_event_elapsed_us; "
                "cannot time CUDA samples\n");
            ggml_backend_free(backend_vram);
            ggml_backend_free(backend_cpu);
            llama_backend_free();
            return 1;
        }
    }

    // Eligibility probe, once per run, before any mapped registration.
    bool mapped_supported = false;
    if (run_mapped) {
        mapped_supported = probe_mapped_host_support();
    }

    // Persistent per-mode weight controls; the real expert bytes are copied once,
    // before any warmup, and the setup copy is reported separately.
    moe_ffn_control cpu_control;
    moe_ffn_control vram_control;
    moe_ffn_control mapped_control;
    bool ok = true;
    if (run_cpu || run_vram || (run_mapped && mapped_supported)) {
        ok &= init_ffn_control(moe_storage_mode::cpu, backend_cpu, bundle, cpu_control);
    }
    if (run_vram) {
        ok &= init_ffn_control(moe_storage_mode::vram, backend_vram, bundle, vram_control);
    }
    if (run_mapped && mapped_supported) {
        ok &= init_mapped_host_control(backend_vram, bundle, mapped_control);
        if (!ok) {
            mapped_supported = false; // registration or allocation failure: unsupported
        }
    }
    if (!ok) {
        free_ffn_control(mapped_control);
        free_ffn_control(vram_control);
        free_ffn_control(cpu_control);
        ggml_backend_free(backend_vram);
        ggml_backend_free(backend_cpu);
        llama_backend_free();
        return 1;
    }

    printf("================================================================================\n");
    printf("MoE Mapped-Host Benchmark - CPU/VRAM/mapped-host complete-FFN controls\n");
    printf("================================================================================\n");
    printf("Model:  %s\n", params.model_path.c_str());
    printf("Layer:  %lld\n", (long long)params.layer);
    printf("Shapes: ");
    for (size_t i = 0; i < params.shapes.size(); i++) {
        printf("%s%s", i > 0 ? "," : "", params.shapes[i].c_str());
    }
    printf("\nMode: %s | Pattern: %s | Warmup: %lld | Reps: %lld | JSON: %s\n",
        moe_run_mode_name(params.mode),
        selection_pattern_name(params.pattern),
        (long long)params.warmup, (long long)params.reps,
        params.json_path.empty() ? "no" : params.json_path.c_str());
    printf("--------------------------------------------------------------------------------\n");
    // Task 5 timing_started guard: armed right before warmup begins. Every
    // setup weight copy and registration has completed above; the correctness
    // passes and the timed loops never copy expert weights, so any copy
    // attempted from here on is a regression that aborts the run.
    g_timing_started = true;
    std::vector<selection_pattern> patterns;
    if (params.pattern == selection_pattern::both) {
        patterns.push_back(selection_pattern::distinct);
        patterns.push_back(selection_pattern::repeated);
    } else {
        patterns.push_back(params.pattern);
    }

    std::vector<moe_case_result> results;
    results.reserve(params.shapes.size() * patterns.size() * 3);

    bool all_correct = true;
    for (const std::string & shape_name : params.shapes) {
        const moe_shape_case * shape = find_shape(shape_name.c_str());
        for (selection_pattern pattern : patterns) {
            if (params.mode == moe_run_mode::cpu) {
                moe_case_result cpu_result;
                const bool passed = run_single_mode_case(
                    bundle, cpu_control, nullptr, *shape, pattern, params, cpu_result);
                all_correct = all_correct && passed;
                results.push_back(cpu_result);
            } else if (params.mode == moe_run_mode::vram) {
                moe_case_result vram_result;
                const bool passed = run_single_mode_case(
                    bundle, vram_control, &cpu_control, *shape, pattern, params, vram_result);
                all_correct = all_correct && passed;
                results.push_back(vram_result);
            } else if (params.mode == moe_run_mode::mapped_host) {
                const moe_expert_selection sel = make_expert_selection(*shape, pattern);
                moe_case_result mapped_result;
                if (!mapped_supported) {
                    // Diagnostics already emitted by the probe / control init;
                    // record an unsupported result and exit nonzero.
                    fill_unsupported_mapped_result(mapped_result, *shape, pattern, sel);
                    all_correct = false;
                } else {
                    const bool passed = run_mapped_case(
                        bundle, cpu_control, mapped_control, *shape, pattern, params, mapped_result);
                    all_correct = all_correct && passed;
                }
                results.push_back(mapped_result);
            } else {
                // cpu_vram (default) and all: the CPU/VRAM pair.
                moe_case_result cpu_result;
                moe_case_result vram_result;
                bool passed = run_case_pair(
                    bundle, cpu_control, vram_control, *shape, pattern, params, cpu_result, vram_result);
                results.push_back(cpu_result);
                results.push_back(vram_result);
                if (params.mode == moe_run_mode::all) {
                    // Mapped-host case, in order after cpu and vram.
                    moe_case_result mapped_result;
                    const bool mapped_passed = mapped_supported
                        ? run_mapped_case(
                            bundle, cpu_control, mapped_control, *shape, pattern, params, mapped_result)
                        : false;
                    if (!mapped_supported) {
                        const moe_expert_selection sel = make_expert_selection(*shape, pattern);
                        fill_unsupported_mapped_result(mapped_result, *shape, pattern, sel);
                    }
                    passed = passed && mapped_passed;
                    results.push_back(mapped_result);
                }
                all_correct = all_correct && passed;
            }
        }
    }

    // Task 5: Gate 1 evaluation on the distinct top-8 rows feeds both the JSON
    // file (top-level gate_one object) and the final stdout summary.
    const gate_one_report gate = evaluate_gate_one(results);
    if (!params.json_path.empty()) {
        if (g_guard_violation_h2d_bytes != 0) {
            fprintf(stderr, "ERROR: timing_started guard tripped (%lld attempted expert-weight bytes); "
                "refusing to write %s\n",
                (long long) g_guard_violation_h2d_bytes, params.json_path.c_str());
            all_correct = false;
        } else {
            ok = write_json_results(params.json_path.c_str(), params, results, gate);
            all_correct = all_correct && ok;
        }
    }

    // Teardown order: mapped ranges first (reset() synchronizes and unregisters
    // while no kernel can still read the pages), then the control buffers, then
    // the backends, then the llama runtime.
    free_ffn_control(mapped_control);
    free_ffn_control(vram_control);
    free_ffn_control(cpu_control);
    ggml_backend_free(backend_vram);
    ggml_backend_free(backend_cpu);
    llama_backend_free();

    printf("================================================================================\n");
    printf("Summary: %s\n", all_correct ? "all cases passed numerical comparison" :
        "one or more cases failed (see errors above)");
    if (run_mapped && !mapped_supported) {
        printf("Mapped-host mode: unsupported (diagnostics above)\n");
    }
    print_gate_one_summary(gate); // last stdout line: "Gate 1: <status>"
    return all_correct ? 0 : 1;
}
