#include "llama.h"
#include "common.h"
#include "arg.h"
#include "sampling.h"
#include "speculative.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct benchmark_result {
    std::string mode_name;
    double t_load_ms = 0.0;
    double t_pp_ms = 0.0;
    double pp_tps = 0.0;
    double t_tg_ms = 0.0;
    double tg_tps = 0.0;
    double t_promo_ms = 0.0;
    int n_gpu_layers = 0;
    int n_prompt = 0;
    int n_gen = 0;
};

static uint64_t get_time_us() {
    return (uint64_t) ggml_time_us();
}

static benchmark_result run_single_benchmark(
        const std::string & model_path,
        int mode, // 0: Baseline Non-MTP, 1: Static MTP, 2: Dynamic MTP
        int n_prompt,
        int n_gen,
        int n_gpu_layers,
        size_t fit_target_bytes,
        size_t expert_cache_bytes,
        int n_threads) {

    benchmark_result res;
    res.n_prompt = n_prompt;
    res.n_gen = n_gen;

    if (mode == 0) {
        res.mode_name = "Baseline (Non-MTP)";
    } else if (mode == 1) {
        res.mode_name = "Static MTP Offload";
    } else {
        res.mode_name = "Dynamic MTP Offload";
    }

    printf("\n=================================================================\n");
    printf("  Executing: %s\n", res.mode_name.c_str());
    printf("=================================================================\n");

    common_params params;
    params.model.path = model_path;
    params.n_gpu_layers = n_gpu_layers;
    if (n_gpu_layers < 0) {
        params.fit_params = true;
        params.fit_params_target = std::vector<size_t>(llama_max_devices(), fit_target_bytes);
        params.fit_params_min_ctx = n_prompt + n_gen + 256;
    } else {
        params.fit_params = false;
    }

    params.cpuparams.n_threads = n_threads;
    params.cpuparams_batch.n_threads = n_threads;
    params.n_ctx = n_prompt + n_gen + 256;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.expert_cache_size = expert_cache_bytes;

    if (mode == 0) {
        params.speculative.types.clear();
        params.mtp_dynamic_offload = false;
    } else if (mode == 1) {
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.speculative.draft.n_max = 2;
        params.mtp_dynamic_offload = false;
        params.speculative.draft.mtp_dynamic_offload = false;
    } else {
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.speculative.draft.n_max = 2;
        params.mtp_dynamic_offload = true;
        params.speculative.draft.mtp_dynamic_offload = true;
    }

    const uint64_t t_load_start = get_time_us();
    common_init_result_ptr init = common_init_from_params(params);
    if (!init || !init->model() || !init->context()) {
        fprintf(stderr, "Failed to initialize model/context from %s\n", model_path.c_str());
        return res;
    }

    res.t_load_ms = (get_time_us() - t_load_start) / 1000.0;
    auto * model = init->model();
    auto * ctx = init->context();
    res.n_gpu_layers = params.n_gpu_layers;

    const auto * vocab = llama_model_get_vocab(model);

    // Warm up / create dummy prompt tokens
    std::vector<llama_token> prompt_tokens;
    prompt_tokens.reserve(n_prompt);
    prompt_tokens.push_back(llama_vocab_bos(vocab));
    for (int i = 1; i < n_prompt; ++i) {
        prompt_tokens.push_back((llama_token)(100 + (i % 5000)));
    }

    // 1. Prompt Processing
    const uint64_t t_pp_start = get_time_us();
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed during prompt processing\n");
        return res;
    }
    const uint64_t t_pp_end = get_time_us();
    res.t_pp_ms = (t_pp_end - t_pp_start) / 1000.0;
    res.pp_tps = (n_prompt / (res.t_pp_ms / 1000.0));

    printf(">> Prompt Processing: %d tokens in %.2f ms (%.2f t/s)\n",
           n_prompt, res.t_pp_ms, res.pp_tps);

    // 2. Promotion check for Dynamic MTP
    if (mode == 2) {
        const uint64_t t_promo_start = get_time_us();
        if (llama_model_has_mtp(model) && !llama_model_mtp_is_gpu_resident(model)) {
            llama_model_mtp_promote_to_gpu(model, ctx);
        }
        res.t_promo_ms = (get_time_us() - t_promo_start) / 1000.0;
        printf(">> Dynamic MTP GPU Promotion Latency: %.2f ms\n", res.t_promo_ms);
    }

    // 3. Token Generation
    struct common_sampler * smpl = common_sampler_init(model, params.sampling);
    const uint64_t t_tg_start = get_time_us();

    for (int i = 0; i < n_gen; ++i) {
        llama_token token = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, token, true);

        llama_batch gen_batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, gen_batch) != 0) {
            fprintf(stderr, "llama_decode failed during generation\n");
            break;
        }
    }
    const uint64_t t_tg_end = get_time_us();
    res.t_tg_ms = (t_tg_end - t_tg_start) / 1000.0;
    res.tg_tps = (n_gen / (res.t_tg_ms / 1000.0));

    printf(">> Token Generation:  %d tokens in %.2f ms (%.2f t/s)\n",
           n_gen, res.t_tg_ms, res.tg_tps);

    common_sampler_free(smpl);
    return res;
}

int main(int argc, char ** argv) {
    std::string model_path = "G:/ai/models/Qwen3.6-35B-A3B-APEX-MTP-Quality.gguf";
    int n_prompt = 64;
    int n_gen = 16;
    int n_gpu_layers = -1;
    size_t fit_target_bytes = 512 * 1024 * 1024ULL; // 512 MiB VRAM margin
    size_t expert_cache_bytes = 1024 * 1024 * 1024ULL; // 1 GiB Expert Cache
    int n_threads = 14;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "-p" && i + 1 < argc) {
            n_prompt = std::atoi(argv[++i]);
        } else if (arg == "-n" && i + 1 < argc) {
            n_gen = std::atoi(argv[++i]);
        } else if (arg == "-ngl" && i + 1 < argc) {
            n_gpu_layers = std::atoi(argv[++i]);
        } else if (arg == "-fitt" && i + 1 < argc) {
            fit_target_bytes = (size_t)std::atoll(argv[++i]) * 1024ULL * 1024ULL;
        } else if (arg == "-exc" && i + 1 < argc) {
            expert_cache_bytes = (size_t)std::atoll(argv[++i]) * 1024ULL * 1024ULL;
        } else if (arg == "-t" && i + 1 < argc) {
            n_threads = std::atoi(argv[++i]);
        }
    }

    llama_backend_init();

    printf("=================================================================\n");
    printf("  llama.cpp Comparative Benchmark: Baseline vs Dynamic vs Static\n");
    printf("=================================================================\n");
    printf("Model:        %s\n", model_path.c_str());
    printf("Prompt Len:   %d tokens\n", n_prompt);
    printf("Gen Len:      %d tokens\n", n_gen);
    if (n_gpu_layers >= 0) {
        printf("GPU Layers:   %d (explicit)\n", n_gpu_layers);
    } else {
        printf("GPU Fit:      Enabled (Target Margin: %zu MiB)\n", fit_target_bytes / (1024 * 1024));
    }
    printf("Expert Cache: %zu MiB\n", expert_cache_bytes / (1024 * 1024));
    printf("Threads:      %d\n", n_threads);
    printf("=================================================================\n");

    std::vector<benchmark_result> results;

    // Run Mode 0: Baseline (Non-MTP)
    results.push_back(run_single_benchmark(model_path, 0, n_prompt, n_gen, n_gpu_layers, fit_target_bytes, expert_cache_bytes, n_threads));

    // Run Mode 2: Dynamic MTP Offload
    results.push_back(run_single_benchmark(model_path, 2, n_prompt, n_gen, n_gpu_layers, fit_target_bytes, expert_cache_bytes, n_threads));

    // Run Mode 1: Static MTP Offload
    results.push_back(run_single_benchmark(model_path, 1, n_prompt, n_gen, n_gpu_layers, fit_target_bytes, expert_cache_bytes, n_threads));

    printf("\n\n==================================================================================================\n");
    printf("                                  COMPARATIVE BENCHMARK SUMMARY                                   \n");
    printf("==================================================================================================\n");
    printf("| %-22s | %-10s | %-12s | %-14s | %-12s | %-14s |\n",
           "Mode", "GPU Layers", "Load (s)", "PP Speed (t/s)", "Promo (ms)", "TG Speed (t/s)");
    printf("|------------------------|------------|--------------|----------------|--------------|----------------|\n");

    for (const auto & r : results) {
        printf("| %-22s | %10d | %10.2f s | %11.2f t/s | %10.2f ms | %11.2f t/s |\n",
               r.mode_name.c_str(),
               r.n_gpu_layers,
               r.t_load_ms / 1000.0,
               r.pp_tps,
               r.t_promo_ms,
               r.tg_tps);
    }
    printf("==================================================================================================\n");

    llama_backend_free();
    return 0;
}
