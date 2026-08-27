#include "llama.h"
#include "common.h"
#include "arg.h"
#include "sampling.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <string>
#include <vector>

struct heterogeneous_bench_result {
    std::string tier_name;
    std::string manifest_path;
    size_t cache_bytes = 0;
    double t_load_s = 0.0;
    double pp_tps = 0.0;
    double tg_tps = 0.0;
    double tg_ms_per_tok = 0.0;
    uint64_t n_requests = 0;
    uint64_t n_hits = 0;
    uint64_t n_misses = 0;
    uint64_t bytes_ram_to_gpu = 0;
    double hit_rate_pct = 0.0;
    double speedup_vs_baseline = 1.0;
};

static uint64_t get_time_us() {
    return (uint64_t) ggml_time_us();
}

static heterogeneous_bench_result run_tier_bench(
        const std::string & model_path,
        const std::string & tier_name,
        const std::string & manifest_path,
        size_t cache_bytes,
        int n_prompt,
        int n_gen,
        size_t fit_target_bytes,
        int n_threads) {

    heterogeneous_bench_result res;
    res.tier_name = tier_name;
    res.manifest_path = manifest_path;
    res.cache_bytes = cache_bytes;

    printf("\n================================================================================\n");
    printf("  Benchmarking Tier: %s (Manifest: %s | Cache: %zu MiB)\n",
        tier_name.c_str(), manifest_path.empty() ? "None (CPU Control)" : manifest_path.c_str(), cache_bytes / (1024 * 1024));
    printf("================================================================================\n");

    common_params params;
    params.model.path = model_path;
    params.n_gpu_layers = -1;
    params.fit_params = true;
    params.fit_params_target = std::vector<size_t>(llama_max_devices(), fit_target_bytes);
    params.fit_params_min_ctx = n_prompt + n_gen + 256;
    params.cpuparams.n_threads = n_threads;
    params.cpuparams_batch.n_threads = n_threads;
    params.n_ctx = n_prompt + n_gen + 256;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.expert_cache_size = cache_bytes;
    params.expert_cache_persist = false;
    params.pinned_experts_manifest = manifest_path;

    const uint64_t t_load0 = get_time_us();
    common_init_result_ptr init = common_init_from_params(params);
    if (!init || !init->model() || !init->context()) {
        fprintf(stderr, "ERROR: Failed to initialize model from %s\n", model_path.c_str());
        return res;
    }
    const uint64_t t_load1 = get_time_us();
    res.t_load_s = (t_load1 - t_load0) / 1e6;

    auto * model = init->model();
    auto * ctx = init->context();
    const auto * vocab = llama_model_get_vocab(model);

    // Warmup prompt
    std::vector<llama_token> prompt_tokens;
    prompt_tokens.push_back(llama_vocab_bos(vocab));
    for (int i = 1; i < n_prompt; ++i) {
        prompt_tokens.push_back((llama_token)(100 + (i % 5000)));
    }

    // 1. Prompt Processing
    const uint64_t t_pp0 = get_time_us();
    llama_batch pp_batch = llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());
    if (llama_decode(ctx, pp_batch) != 0) {
        fprintf(stderr, "ERROR: llama_decode failed during prompt processing\n");
        return res;
    }
    const uint64_t t_pp1 = get_time_us();
    res.pp_tps = (double)n_prompt / ((t_pp1 - t_pp0) / 1e6);

    // Reset perf counters before TG decode
    llama_perf_context_reset(ctx);

    // 2. Decode Generation
    llama_token cur_token = prompt_tokens.back();
    const uint64_t t_tg0 = get_time_us();

    for (int i = 0; i < n_gen; ++i) {
        llama_batch gen_batch = llama_batch_get_one(&cur_token, 1);
        if (llama_decode(ctx, gen_batch) != 0) {
            fprintf(stderr, "ERROR: llama_decode failed during generation step %d\n", i);
            break;
        }

        // Greedy argmax sample
        const float * logits = llama_get_logits_ith(ctx, 0);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        int max_id = 0;
        float max_logit = -1e9f;
        for (int v = 0; v < n_vocab; v++) {
            if (logits[v] > max_logit) {
                max_logit = logits[v];
                max_id = v;
            }
        }
        cur_token = (llama_token)max_id;
    }

    const uint64_t t_tg1 = get_time_us();
    const double tg_elapsed_s = (t_tg1 - t_tg0) / 1e6;
    res.tg_tps = (double)n_gen / tg_elapsed_s;
    res.tg_ms_per_tok = (tg_elapsed_s * 1000.0) / (double)n_gen;

    // Collect expert cache telemetry
    auto * sched = llama_context_get_sched(ctx);
    if (sched) {
        struct ggml_backend_expert_cache_stats stats = {};
        if (ggml_backend_sched_get_expert_cache_stats(sched, -1, &stats)) {
            res.n_requests = stats.n_requests;
            res.n_hits = stats.n_hits;
            res.n_misses = stats.n_misses;
            res.bytes_ram_to_gpu = stats.bytes_ram_to_gpu;
            res.hit_rate_pct = stats.n_requests > 0 ? ((double)stats.n_hits / (double)stats.n_requests) * 100.0 : 0.0;
        }
    }

    printf(">> Result: Decode: %.2f ms/tok (%.2f tok/s) | PP: %.2f tok/s | Hits: %llu / %llu (%.1f%%) | RAM->GPU PCIe: %llu bytes\n",
        res.tg_ms_per_tok, res.tg_tps, res.pp_tps,
        (unsigned long long)res.n_hits, (unsigned long long)res.n_requests,
        res.hit_rate_pct, (unsigned long long)res.bytes_ram_to_gpu);

    return res;
}

int main(int argc, char ** argv) {
    std::string model_path = "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf";
    int n_threads = 14;
    int n_prompt = 16;
    int n_gen = 64;
    size_t fit_target_bytes = 256 * 1024 * 1024; // 256 MiB VRAM margin

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            n_prompt = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_gen = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-fitt") == 0 && i + 1 < argc) {
            fit_target_bytes = (size_t)atoi(argv[++i]) * 1024 * 1024;
        }
    }

    printf("================================================================================\n");
    printf("Epic 4: True Heterogeneous Route Execution Benchmark & Gate B Evaluation\n");
    printf("Model: %s\n", model_path.c_str());
    printf("Threads: %d | Prompt: %d tokens | Generation: %d tokens | Fit Target: %zu MiB\n",
        n_threads, n_prompt, n_gen, fit_target_bytes / (1024 * 1024));
    printf("================================================================================\n");

    llama_backend_init();

    std::vector<heterogeneous_bench_result> results;

    // 1. Control Baseline: CPU routed MoE (0 MiB cache)
    results.push_back(run_tier_bench(model_path, "CPU Baseline (Control)", "", 0, n_prompt, n_gen, fit_target_bytes, n_threads));

    // 2. Static Pinned 64 MiB Tier
    results.push_back(run_tier_bench(model_path, "Pinned 64 MiB", "pinned_experts_64mb.json", 64 * 1024 * 1024, n_prompt, n_gen, fit_target_bytes, n_threads));

    // 3. Static Pinned 128 MiB Tier
    results.push_back(run_tier_bench(model_path, "Pinned 128 MiB", "pinned_experts_128mb.json", 128 * 1024 * 1024, n_prompt, n_gen, fit_target_bytes, n_threads));

    // 4. Static Pinned 256 MiB Tier
    results.push_back(run_tier_bench(model_path, "Pinned 256 MiB", "pinned_experts_256mb.json", 256 * 1024 * 1024, n_prompt, n_gen, fit_target_bytes, n_threads));

    // 5. Static Pinned 512 MiB Tier
    results.push_back(run_tier_bench(model_path, "Pinned 512 MiB", "pinned_experts_512mb.json", 512 * 1024 * 1024, n_prompt, n_gen, fit_target_bytes, n_threads));

    // 6. Static Pinned 1024 MiB Tier
    results.push_back(run_tier_bench(model_path, "Pinned 1024 MiB", "pinned_experts_1024mb.json", 1024 * 1024 * 1024, n_prompt, n_gen, fit_target_bytes, n_threads));

    // Compute speedup vs control baseline
    const double baseline_tps = results[0].tg_tps > 0 ? results[0].tg_tps : 1.0;
    for (auto & r : results) {
        r.speedup_vs_baseline = r.tg_tps / baseline_tps;
    }

    printf("\n\n========================================================================================================================\n");
    printf("                                            GATE B COMPARATIVE BENCHMARK SUMMARY                                        \n");
    printf("========================================================================================================================\n");
    printf("| %-24s | %-12s | %-12s | %-12s | %-10s | %-12s | %-12s |\n",
        "Configuration", "Load (s)", "TG (tok/s)", "TG (ms/tok)", "Hit Rate", "RAM->GPU Bytes", "Speedup");
    printf("|--------------------------|--------------|--------------|--------------|------------|----------------|--------------|\n");

    for (const auto & r : results) {
        printf("| %-24s | %10.2f s | %10.2f t/s | %10.2f ms | %8.1f%% | %12llu B | %11.2fx |\n",
            r.tier_name.c_str(),
            r.t_load_s,
            r.tg_tps,
            r.tg_ms_per_tok,
            r.hit_rate_pct,
            (unsigned long long)r.bytes_ram_to_gpu,
            r.speedup_vs_baseline);
    }
    printf("========================================================================================================================\n");

    // Gate B Decision Evaluation
    printf("\nGate B Evaluation Summary:\n");
    bool passed_gate_b = false;
    double max_speedup = 1.0;
    for (size_t i = 1; i < results.size(); ++i) {
        if (results[i].speedup_vs_baseline > max_speedup) {
            max_speedup = results[i].speedup_vs_baseline;
        }
        if (results[i].speedup_vs_baseline >= 1.05 && results[i].bytes_ram_to_gpu == 0) {
            passed_gate_b = true;
        }
    }

    if (passed_gate_b) {
        printf("  [OUTCOME A / PASS] Static heterogeneous hybrid demonstrates measurable TG throughput gain (Max Speedup: %.2fx, +%.1f%%) with ZERO in-band PCIe miss uploads.\n",
            max_speedup, (max_speedup - 1.0) * 100.0);
        printf("  -> Recommendation: Proceed to Epic 5 (Non-blocking Background Promotion) and Epic 6 (Zero-Sync Remapping).\n");
    } else {
        printf("  [OUTCOME B / NOTICE] Static heterogeneous hybrid speedup is marginal (Max Speedup: %.2fx). Review synchronization overhead in scheduler.\n", max_speedup);
    }
    printf("========================================================================================================================\n");

    llama_backend_free();
    return 0;
}
