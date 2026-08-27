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

struct turn_metrics {
    std::string domain_name;
    double tg_tps = 0.0;
    double tg_ms_per_tok = 0.0;
    uint64_t n_requests = 0;
    uint64_t n_hits = 0;
    uint64_t n_misses = 0;
    uint64_t bytes_ram_to_gpu = 0;
    size_t async_promotions = 0;
    double hit_rate_pct = 0.0;
};

struct run_metrics {
    std::string mode_name;
    std::vector<turn_metrics> turns;
    double total_tg_tps = 0.0;
    double avg_tg_ms_per_tok = 0.0;
    double avg_hit_rate_pct = 0.0;
    uint64_t total_ram_to_gpu = 0;
    size_t total_async_promotions = 0;
    double speedup_vs_baseline = 1.0;
};

static uint64_t get_time_us() {
    return (uint64_t) ggml_time_us();
}

static run_metrics run_dynamic_drift_bench(
        const std::string & model_path,
        const std::string & mode_name,
        const std::string & manifest_path,
        size_t cache_bytes,
        int period_tokens,
        int max_swaps,
        int n_threads,
        size_t fit_target_bytes) {

    run_metrics res;
    res.mode_name = mode_name;

    printf("\n================================================================================\n");
    printf("  Running Mode: %s\n", mode_name.c_str());
    printf("  Manifest: %s | Cache: %zu MiB | Period: %d tok | Max Swaps: %d\n",
        manifest_path.empty() ? "None" : manifest_path.c_str(),
        cache_bytes / (1024 * 1024), period_tokens, max_swaps);
    printf("================================================================================\n");

    common_params params;
    params.model.path = model_path;
    params.n_gpu_layers = -1;
    params.fit_params = true;
    params.fit_params_target = std::vector<size_t>(llama_max_devices(), fit_target_bytes);
    params.fit_params_min_ctx = 256;
    params.cpuparams.n_threads = n_threads;
    params.cpuparams_batch.n_threads = n_threads;
    params.n_ctx = 512;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.expert_cache_size = cache_bytes;
    params.expert_cache_period = period_tokens;
    params.expert_cache_max_swaps = max_swaps;
    params.expert_cache_persist = false;
    params.pinned_experts_manifest = manifest_path;

    common_init_result_ptr init = common_init_from_params(params);
    if (!init || !init->model() || !init->context()) {
        fprintf(stderr, "ERROR: Failed to initialize model\n");
        return res;
    }

    auto * model = init->model();
    auto * ctx = init->context();
    const auto * vocab = llama_model_get_vocab(model);
    auto * sched = llama_context_get_sched(ctx);

    struct turn_def {
        std::string name;
        int seed_offset;
    };

    std::vector<turn_def> domain_turns = {
        { "Turn 1: Coding / Algorithms", 100 },
        { "Turn 2: Creative Writing",    5000 },
        { "Turn 3: Formal Mathematics",  12000 },
    };

    const int n_prompt = 16;
    const int n_gen = 32;

    double total_gen_s = 0.0;
    int total_gen_tokens = 0;

    for (size_t t = 0; t < domain_turns.size(); ++t) {
        const auto & td = domain_turns[t];
        turn_metrics tm;
        tm.domain_name = td.name;

        // Construct distinct domain prompt tokens
        std::vector<llama_token> prompt_tokens;
        prompt_tokens.push_back(llama_vocab_bos(vocab));
        for (int i = 1; i < n_prompt; ++i) {
            prompt_tokens.push_back((llama_token)(td.seed_offset + i * 37));
        }

        // 1. Prompt processing
        llama_batch pp_batch = llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());
        llama_decode(ctx, pp_batch);

        // Reset perf counters before decode
        llama_perf_context_reset(ctx);

        // 2. Decode generation
        llama_token cur_token = prompt_tokens.back();
        const uint64_t t_tg0 = get_time_us();

        for (int i = 0; i < n_gen; ++i) {
            // Process any background promotion completions before step
            if (sched) {
                tm.async_promotions += ggml_backend_sched_process_async_promotions(sched, 2);
            }

            llama_batch gen_batch = llama_batch_get_one(&cur_token, 1);
            if (llama_decode(ctx, gen_batch) != 0) {
                break;
            }

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
        const double turn_gen_s = (t_tg1 - t_tg0) / 1e6;
        tm.tg_tps = (double)n_gen / turn_gen_s;
        tm.tg_ms_per_tok = (turn_gen_s * 1000.0) / (double)n_gen;
        total_gen_s += turn_gen_s;
        total_gen_tokens += n_gen;

        // Telemetry
        if (sched) {
            struct ggml_backend_expert_cache_stats stats = {};
            if (ggml_backend_sched_get_expert_cache_stats(sched, -1, &stats)) {
                tm.n_requests = stats.n_requests;
                tm.n_hits = stats.n_hits;
                tm.n_misses = stats.n_misses;
                tm.bytes_ram_to_gpu = stats.bytes_ram_to_gpu;
                tm.hit_rate_pct = stats.n_requests > 0 ? ((double)stats.n_hits / (double)stats.n_requests) * 100.0 : 0.0;
            }
        }

        printf("  >> [%s] Decode: %.2f ms/tok (%.2f tok/s) | Hits: %llu / %llu (%.1f%%) | RAM->GPU: %llu B | Promoted: %zu\n",
            tm.domain_name.c_str(), tm.tg_ms_per_tok, tm.tg_tps,
            (unsigned long long)tm.n_hits, (unsigned long long)tm.n_requests,
            tm.hit_rate_pct, (unsigned long long)tm.bytes_ram_to_gpu, tm.async_promotions);

        res.turns.push_back(tm);
    }

    res.total_tg_tps = (double)total_gen_tokens / total_gen_s;
    res.avg_tg_ms_per_tok = (total_gen_s * 1000.0) / (double)total_gen_tokens;

    uint64_t tot_req = 0, tot_hit = 0;
    for (const auto & tm : res.turns) {
        tot_req += tm.n_requests;
        tot_hit += tm.n_hits;
        res.total_ram_to_gpu += tm.bytes_ram_to_gpu;
        res.total_async_promotions += tm.async_promotions;
    }
    res.avg_hit_rate_pct = tot_req > 0 ? ((double)tot_hit / (double)tot_req) * 100.0 : 0.0;

    return res;
}

int main(int argc, char ** argv) {
    std::string model_path = "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf";
    int n_threads = 14;
    size_t fit_target_bytes = 256 * 1024 * 1024;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-fitt") == 0 && i + 1 < argc) {
            fit_target_bytes = (size_t)atoi(argv[++i]) * 1024 * 1024;
        }
    }

    printf("================================================================================\n");
    printf("Epic 5 & Epic 6: Dynamic Multi-Turn Topic Drift Benchmark & Evaluation\n");
    printf("Model: %s\n", model_path.c_str());
    printf("Threads: %d | Fit Target: %zu MiB\n", n_threads, fit_target_bytes / (1024 * 1024));
    printf("================================================================================\n");

    llama_backend_init();

    std::vector<run_metrics> all_runs;

    // 1. Control Baseline: Pure CPU MoE
    all_runs.push_back(run_dynamic_drift_bench(model_path, "1. Pure CPU MoE (Control)", "", 0, 0, 0, n_threads, fit_target_bytes));

    // 2. Static Manifest (1024 MiB Pinned)
    all_runs.push_back(run_dynamic_drift_bench(model_path, "2. Static Pinned (1024 MiB)", "pinned_experts_1024mb.json", 1024 * 1024 * 1024, 0, 0, n_threads, fit_target_bytes));

    // 3. Dynamic Background Promotion (1024 MiB Cache, Rate-limited background DMA)
    all_runs.push_back(run_dynamic_drift_bench(model_path, "3. Dynamic Background Promotion (1024 MiB)", "", 1024 * 1024 * 1024, 16, 2, n_threads, fit_target_bytes));

    const double baseline_tps = all_runs[0].total_tg_tps > 0 ? all_runs[0].total_tg_tps : 1.0;
    for (auto & r : all_runs) {
        r.speedup_vs_baseline = r.total_tg_tps / baseline_tps;
    }

    printf("\n\n========================================================================================================================\n");
    printf("                                    DYNAMIC TOPIC DRIFT COMPARATIVE SUMMARY                                             \n");
    printf("========================================================================================================================\n");
    printf("| %-38s | %-12s | %-12s | %-12s | %-14s | %-10s |\n",
        "Mode", "Total TG (t/s)", "Latency (ms)", "RAM->GPU", "Async Promoted", "Speedup");
    printf("|----------------------------------------|--------------|--------------|--------------|----------------|------------|\n");

    for (const auto & r : all_runs) {
        printf("| %-38s | %10.2f t/s | %10.2f ms | %10llu B | %14zu | %9.2fx |\n",
            r.mode_name.c_str(),
            r.total_tg_tps,
            r.avg_tg_ms_per_tok,
            (unsigned long long)r.total_ram_to_gpu,
            r.total_async_promotions,
            r.speedup_vs_baseline);
    }
    printf("========================================================================================================================\n");

    llama_backend_free();
    return 0;
}
