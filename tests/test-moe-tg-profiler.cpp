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
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

static int64_t get_time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

struct op_stat {
    std::string op_name;
    std::string backend;
    uint64_t count = 0;
    uint64_t total_us = 0;
    uint64_t min_us = UINT64_MAX;
    uint64_t max_us = 0;
    std::vector<uint64_t> samples;
};

struct layer_moe_stat {
    int layer = -1;
    uint64_t router_us = 0;
    uint64_t gate_up_us = 0;
    uint64_t act_us = 0;
    uint64_t down_us = 0;
    uint64_t combine_us = 0;
    uint64_t count = 0;
};

struct expert_usage {
    int layer;
    int expert_id;
    uint64_t activations = 0;
    double route_prob = 0.0;
    uint64_t consecutive_reuses = 0;
    double mean_reuse_distance = 0.0;
    double value_per_byte = 0.0;
};

struct profiler_context {
    bool is_decode = false;
    int current_token = 0;
    int64_t current_node_t0 = 0;

    std::map<std::string, op_stat> op_stats;
    std::map<int, layer_moe_stat> layer_moe_stats;

    // layer -> token -> list of routed expert IDs
    std::vector<std::vector<std::vector<int32_t>>> layer_routes; // [layer][token][expert]
    std::vector<uint64_t> token_wall_times_us;
    std::vector<uint64_t> token_cpu_times_us;
    std::vector<uint64_t> token_gpu_times_us;

    int n_layers = 40;
};

static bool profiler_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * ctx = (profiler_context *) user_data;
    if (!ctx || !ctx->is_decode) {
        return true;
    }

    if (ask) {
        ctx->current_node_t0 = get_time_us();
        return true;
    }

    const int64_t dt_us = get_time_us() - ctx->current_node_t0;
    if (dt_us < 0) {
        return true;
    }

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const std::string backend = is_host ? "CPU" : "CUDA0";
    const std::string op_name = ggml_op_name(t->op);

    // Group key by op_name + backend
    std::string key = op_name + " (" + backend + ")";
    auto & stat = ctx->op_stats[key];
    stat.op_name = op_name;
    stat.backend = backend;
    stat.count++;
    stat.total_us += dt_us;
    stat.min_us = std::min(stat.min_us, (uint64_t)dt_us);
    stat.max_us = std::max(stat.max_us, (uint64_t)dt_us);
    stat.samples.push_back((uint64_t)dt_us);

    // Parse layer index from tensor name if available (e.g. "blk.12.ffn_moe_gate" or "ffn_moe_gate-12")
    int layer_idx = -1;
    const std::string tname = t->name;
    size_t blk_pos = tname.find("blk.");
    if (blk_pos != std::string::npos) {
        layer_idx = atoi(tname.c_str() + blk_pos + 4);
    } else {
        size_t dash_pos = tname.find_last_of('-');
        if (dash_pos != std::string::npos && dash_pos + 1 < tname.size() && isdigit(tname[dash_pos + 1])) {
            layer_idx = atoi(tname.c_str() + dash_pos + 1);
        }
    }

    if (layer_idx >= 0 && layer_idx < ctx->n_layers) {
        auto & l_stat = ctx->layer_moe_stats[layer_idx];
        l_stat.layer = layer_idx;
        l_stat.count++;

        if (tname.find("gate_inp") != std::string::npos || (t->op == GGML_OP_MUL_MAT && tname.find("ffn") != std::string::npos)) {
            l_stat.router_us += dt_us;
        } else if (tname.find("gate") != std::string::npos || tname.find("up") != std::string::npos) {
            l_stat.gate_up_us += dt_us;
        } else if (tname.find("swiglu") != std::string::npos || tname.find("silu") != std::string::npos || t->op == GGML_OP_GLU || t->op == GGML_OP_UNARY) {
            l_stat.act_us += dt_us;
        } else if (tname.find("down") != std::string::npos) {
            l_stat.down_us += dt_us;
        } else if (tname.find("weighted") != std::string::npos || tname.find("moe_out") != std::string::npos) {
            l_stat.combine_us += dt_us;
        }

        // Record route ids directly while device buffer is live and synchronized
        if (t->op == GGML_OP_MUL_MAT_ID && t->src[2] != nullptr) {
            if (ctx->layer_routes.size() <= (size_t) layer_idx) {
                ctx->layer_routes.resize(ctx->n_layers);
            }
            if (ctx->layer_routes[layer_idx].size() <= (size_t) ctx->current_token) {
                ctx->layer_routes[layer_idx].resize(ctx->current_token + 1);
            }
            if (ctx->layer_routes[layer_idx][ctx->current_token].empty()) {
                const ggml_tensor * ids_tensor = t->src[2];
                const int n_ids = (int) ggml_nelements(ids_tensor);
                std::vector<int32_t> ids_vec(n_ids);
                ggml_backend_tensor_get(ids_tensor, ids_vec.data(), 0, n_ids * sizeof(int32_t));
                ctx->layer_routes[layer_idx][ctx->current_token] = std::move(ids_vec);
            }
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    std::string model_path = "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf";
    int n_threads = 14;
    int n_prompt = 16;
    int n_gen = 128;
    size_t fit_target_bytes = 256 * 1024 * 1024; // 256 MiB
    std::string out_manifest_prefix = "pinned_experts";
    size_t cache_mib = 0;              // deployment cache budget, mirrored into fit
    double w_full = 1.0;               // admission objective weight: 8/8 bundle hit
    double w_hetero = 0.4;             // admission objective weight: 7/8 bundle hit
    std::string trace_jsonl_path;       // attested TG1 route trace (JSONL)
    std::string dump_routes_path;      // debug: raw captured routes

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
        } else if (strcmp(argv[i], "--out-manifest") == 0 && i + 1 < argc) {
            out_manifest_prefix = argv[++i];
        } else if (strcmp(argv[i], "--cache-mib") == 0 && i + 1 < argc) {
            cache_mib = (size_t) atoi(argv[++i]);
        } else if (strcmp(argv[i], "--w-full") == 0 && i + 1 < argc) {
            w_full = atof(argv[++i]);
        } else if (strcmp(argv[i], "--w-hetero") == 0 && i + 1 < argc) {
            w_hetero = atof(argv[++i]);
        } else if (strcmp(argv[i], "--dump-routes") == 0 && i + 1 < argc) {
            dump_routes_path = argv[++i];
        } else if (strcmp(argv[i], "--trace-jsonl") == 0 && i + 1 < argc) {
            trace_jsonl_path = argv[++i];
        }
    }

    printf("================================================================================\n");
    printf("MoE TG Decode Profiler & Static Residency Ranker (Epics 2 & 3)\n");
    printf("Model: %s\n", model_path.c_str());
    printf("Threads: %d | Prompt: %d | Decode: %d | Fit Target: %zu MiB | Cache: %zu MiB\n",
        n_threads, n_prompt, n_gen, fit_target_bytes / (1024 * 1024), cache_mib);
    printf("================================================================================\n\n");

    llama_backend_init();

    profiler_context pctx;
    pctx.n_layers = 40;
    pctx.layer_routes.resize(40);

    common_params params;
    params.model.path = model_path;
    params.n_gpu_layers = -1; // Use fit
    params.fit_params = true;
    params.fit_params_target = std::vector<size_t>(llama_max_devices(), fit_target_bytes);
    params.fit_params_min_ctx = n_prompt + n_gen + 256;
    params.cpuparams.n_threads = n_threads;
    params.cpuparams_batch.n_threads = n_threads;
    params.n_ctx = n_prompt + n_gen + 256;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    // mirror the deployment cache budget so fit reproduces deployment placement;
    // period stays at its inert default and no manifest is loaded, so the
    // profiling cache remains empty while cb_eval still records every route
    params.expert_cache_size = cache_mib * 1024 * 1024;
    params.cb_eval = profiler_cb_eval;
    params.cb_eval_user_data = &pctx;

    const uint64_t t_load0 = get_time_us();
    common_init_result_ptr init = common_init_from_params(params);
    if (!init || !init->model() || !init->context()) {
        fprintf(stderr, "ERROR: Failed to initialize model from %s\n", model_path.c_str());
        return 1;
    }
    const uint64_t t_load1 = get_time_us();
    printf("Model loaded in %.2f s\n\n", (t_load1 - t_load0) / 1e6);

    auto * model = init->model();
    auto * ctx = init->context();

    // only host-resident MoE weights can ever be admitted by the cache
    std::vector<bool> layer_eligible(pctx.n_layers, false);
    int n_eligible_layers = 0;
    for (int l = 0; l < pctx.n_layers; l++) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", l);
        const ggml_tensor * t = llama_model_get_tensor(model, name);
        layer_eligible[l] = t != nullptr && ggml_backend_buffer_is_host(t->buffer);
        if (layer_eligible[l]) {
            n_eligible_layers++;
        }
    }
    printf("Placement-eligible host MoE layers: %d of %d (cache-mib %zu)\n",
        n_eligible_layers, pctx.n_layers, cache_mib);
    const auto * vocab = llama_model_get_vocab(model);

    // Warmup prompt
    std::vector<llama_token> prompt_tokens;
    prompt_tokens.push_back(llama_vocab_bos(vocab));
    for (int i = 1; i < n_prompt; ++i) {
        prompt_tokens.push_back((llama_token)(100 + (i % 5000)));
    }

    printf("Executing prompt processing (%d tokens)...\n", n_prompt);
    pctx.is_decode = false;
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "ERROR: llama_decode failed on prompt\n");
        return 1;
    }

    printf("Executing single-token decode generation (%d tokens) with fine-grained profiling...\n", n_gen);
    pctx.is_decode = true;

    llama_token cur_token = prompt_tokens.back();
    for (int t = 0; t < n_gen; t++) {
        pctx.current_token = t;
        const int64_t t_tok0 = get_time_us();

        llama_batch token_batch = llama_batch_get_one(&cur_token, 1);
        if (llama_decode(ctx, token_batch) != 0) {
            fprintf(stderr, "ERROR: llama_decode failed on decode token %d\n", t);
            break;
        }

        const int64_t t_tok1 = get_time_us();
        pctx.token_wall_times_us.push_back(t_tok1 - t_tok0);

        if (getenv("MOE_PROF_DIAG") && t < 2) {
            for (int l = 0; l < 3; l++) {
                if (pctx.layer_routes.size() <= (size_t) l || pctx.layer_routes[l].size() <= (size_t) t) {
                    continue;
                }
                const auto & v = pctx.layer_routes[l][t];
                fprintf(stderr, "[diag] t=%d L%d routes (%zu):", t, l, v.size());
                for (size_t i = 0; i < v.size(); i++) {
                    fprintf(stderr, " %d", v[i]);
                }
                fprintf(stderr, "\n");
            }
        }

        // Simple argmax sample for deterministic stepping
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

    pctx.is_decode = false;

    // Calculate total decode wall time
    uint64_t total_decode_wall_us = 0;
    for (uint64_t dt : pctx.token_wall_times_us) {
        total_decode_wall_us += dt;
    }
    const double mean_token_wall_us = (double)total_decode_wall_us / (double)n_gen;
    const double mean_token_wall_ms = mean_token_wall_us / 1000.0;
    const double tok_per_sec = 1e6 / mean_token_wall_us;

    printf("\n================================================================================\n");
    printf("Epic 2 Story 2.1: Per-Op Decode Latency Breakdown\n");
    printf("Total Generation: %d tokens | Wall Time: %.2f ms | Throughput: %.2f tok/s\n",
        n_gen, total_decode_wall_us / 1000.0, tok_per_sec);
    printf("================================================================================\n");
    printf("%-32s | %8s | %10s | %10s | %10s | %8s\n",
        "Op Classification", "Count", "Total (ms)", "Mean (us)", "Median(us)", "% Decode");
    printf("--------------------------------------------------------------------------------\n");

    // Sort ops by total time descending
    std::vector<std::pair<std::string, op_stat>> sorted_ops(pctx.op_stats.begin(), pctx.op_stats.end());
    std::sort(sorted_ops.begin(), sorted_ops.end(), [](const auto & a, const auto & b) {
        return a.second.total_us > b.second.total_us;
    });

    uint64_t sum_all_ops_us = 0;
    for (const auto & p : sorted_ops) {
        sum_all_ops_us += p.second.total_us;
    }

    for (auto & p : sorted_ops) {
        auto & s = p.second;
        std::sort(s.samples.begin(), s.samples.end());
        const uint64_t median_us = s.samples.empty() ? 0 : s.samples[s.samples.size() / 2];
        const double pct = sum_all_ops_us > 0 ? (double)s.total_us / (double)sum_all_ops_us * 100.0 : 0.0;

        printf("%-32s | %8llu | %10.2f | %10.1f | %10.1f | %7.1f%%\n",
            p.first.c_str(),
            (unsigned long long)s.count,
            s.total_us / 1000.0,
            (double)s.total_us / (double)s.count,
            (double)median_us,
            pct);
    }

    printf("\n================================================================================\n");
    printf("Epic 2 Story 2.2: Canonical Token Latency Budget Breakdown\n");
    printf("================================================================================\n");

    uint64_t total_cpu_moe_us = 0;
    uint64_t total_gpu_moe_us = 0;
    uint64_t total_attn_us = 0;
    uint64_t total_dense_us = 0;
    uint64_t total_router_us = 0;
    uint64_t total_norm_us = 0;

    for (const auto & p : pctx.op_stats) {
        const auto & s = p.second;
        if (s.op_name == "MUL_MAT_ID") {
            if (s.backend == "CPU") total_cpu_moe_us += s.total_us;
            else total_gpu_moe_us += s.total_us;
        } else if (s.op_name == "FLASH_ATTN_EXT" || s.op_name == "ROPE") {
            total_attn_us += s.total_us;
        } else if (s.op_name == "MUL_MAT") {
            total_dense_us += s.total_us;
        } else if (s.op_name == "RMS_NORM") {
            total_norm_us += s.total_us;
        }
    }

    for (const auto & p : pctx.layer_moe_stats) {
        total_router_us += p.second.router_us;
    }

    const double per_tok_cpu_moe_ms = (double)total_cpu_moe_us / (double)n_gen / 1000.0;
    const double per_tok_gpu_moe_ms = (double)total_gpu_moe_us / (double)n_gen / 1000.0;
    const double per_tok_attn_ms = (double)total_attn_us / (double)n_gen / 1000.0;
    const double per_tok_dense_ms = (double)total_dense_us / (double)n_gen / 1000.0;
    const double per_tok_router_ms = (double)total_router_us / (double)n_gen / 1000.0;
    const double per_tok_norm_ms = (double)total_norm_us / (double)n_gen / 1000.0;
    const double per_tok_sched_ms = std::max(0.0, mean_token_wall_ms - (per_tok_cpu_moe_ms + per_tok_gpu_moe_ms + per_tok_attn_ms + per_tok_dense_ms + per_tok_norm_ms));

    printf("Single Decode Token (Mean Wall Time: %.2f ms / %.1f tok/s):\n", mean_token_wall_ms, tok_per_sec);
    printf("  1. CPU MoE Expert GEMV:          %6.2f ms  (%5.1f%%)\n", per_tok_cpu_moe_ms, (per_tok_cpu_moe_ms / mean_token_wall_ms) * 100.0);
    printf("  2. Dense Work (GPU Projections): %6.2f ms  (%5.1f%%)\n", per_tok_dense_ms, (per_tok_dense_ms / mean_token_wall_ms) * 100.0);
    printf("  3. Attention & RoPE (GPU):       %6.2f ms  (%5.1f%%)\n", per_tok_attn_ms, (per_tok_attn_ms / mean_token_wall_ms) * 100.0);
    printf("  4. MoE Router & Top-K:           %6.2f ms  (%5.1f%%)\n", per_tok_router_ms, (per_tok_router_ms / mean_token_wall_ms) * 100.0);
    printf("  5. RMS Normalization:            %6.2f ms  (%5.1f%%)\n", per_tok_norm_ms, (per_tok_norm_ms / mean_token_wall_ms) * 100.0);
    printf("  6. Scheduler & Backend Sync:     %6.2f ms  (%5.1f%%)\n", per_tok_sched_ms, (per_tok_sched_ms / mean_token_wall_ms) * 100.0);

    printf("\n================================================================================\n");
    printf("Epic 3 Story 3.1 & 3.2: Real TG Route Distribution & Value-per-Byte Ranking\n");
    printf("================================================================================\n");

    const int n_experts = 256;
    const int top_k = 8;
    const double delta_t_layer_us = 112.0; // Measured Gate A speedup per layer: 297 us - 185 us = 112 us
    const double delta_t_expert_us = delta_t_layer_us / (double)top_k; // 14.0 us savings per routed expert
    const size_t expert_bundle_bytes = (576 + 576 + 800) * 1024; // 1.95 MiB (Gate Q4_K + Up Q4_K + Down Q6_K)

    std::vector<expert_usage> all_experts;
    all_experts.reserve(40 * n_experts);

    for (int l = 0; l < 40; l++) {
        std::map<int, uint64_t> act_counts;
        std::map<int, uint64_t> consec_counts;
        std::map<int, std::vector<int>> last_seen_token;

        const auto & layer_tok_routes = pctx.layer_routes[l];
        for (size_t t = 0; t < layer_tok_routes.size(); t++) {
            const auto & routed = layer_tok_routes[t];
            std::set<int> current_set(routed.begin(), routed.end());

            for (int exp : routed) {
                if (exp < 0 || exp >= n_experts) continue;
                act_counts[exp]++;

                if (!last_seen_token[exp].empty()) {
                    int prev_t = last_seen_token[exp].back();
                    if (prev_t == (int)t - 1) {
                        consec_counts[exp]++;
                    }
                }
                last_seen_token[exp].push_back((int)t);
            }
        }

        for (int e = 0; e < n_experts; e++) {
            expert_usage u;
            u.layer = l;
            u.expert_id = e;
            u.activations = act_counts[e];
            u.route_prob = (double)u.activations / (double)(n_gen * top_k);
            u.consecutive_reuses = consec_counts[e];

            const auto & seen = last_seen_token[e];
            if (seen.size() > 1) {
                double total_dist = 0;
                for (size_t i = 1; i < seen.size(); i++) {
                    total_dist += (seen[i] - seen[i - 1]);
                }
                u.mean_reuse_distance = total_dist / (double)(seen.size() - 1);
            } else {
                u.mean_reuse_distance = 0.0;
            }

            // Value = P(route) * (T_cpu - T_gpu) / bytes
            u.value_per_byte = (u.route_prob * delta_t_expert_us * 1e6) / (double)expert_bundle_bytes;
            all_experts.push_back(u);
        }
    }

    // Sort all 10,240 experts globally by value_per_byte descending
    std::sort(all_experts.begin(), all_experts.end(), [](const expert_usage & a, const expert_usage & b) {
        return a.value_per_byte > b.value_per_byte;
    });

    printf("Top 15 Highest-Value Static Experts Globally:\n");
    printf("%-6s | %-8s | %-12s | %-12s | %-12s | %-14s\n",
        "Layer", "Expert", "Activations", "Route %", "Mean Reuse", "Value/Byte (x1e6)");
    printf("--------------------------------------------------------------------------------\n");
    for (size_t i = 0; i < 15 && i < all_experts.size(); i++) {
        const auto & u = all_experts[i];
        printf("L%-5d | E%-7d | %12llu | %11.2f%% | %12.1f | %14.4f\n",
            u.layer, u.expert_id,
            (unsigned long long)u.activations,
            u.route_prob * 100.0 * top_k,
            u.mean_reuse_distance,
            u.value_per_byte);
    }

    // Cumulative Coverage per Layer (Layer 0, 10, 20, 30, 39)
    printf("\nCumulative Route Coverage by Layer:\n");
    printf("%-8s | %-10s | %-10s | %-10s | %-10s | %-10s\n",
        "Layer", "Top 1 Exp", "Top 2 Exp", "Top 4 Exp", "Top 8 Exp", "Top 16 Exp");
    printf("--------------------------------------------------------------------------------\n");

    for (int l : { 0, 5, 10, 15, 20, 25, 30, 35, 39 }) {
        std::vector<expert_usage> layer_exp;
        for (const auto & u : all_experts) {
            if (u.layer == l) layer_exp.push_back(u);
        }
        std::sort(layer_exp.begin(), layer_exp.end(), [](const auto & a, const auto & b) {
            return a.activations > b.activations;
        });

        uint64_t total_layer_acts = 0;
        for (const auto & u : layer_exp) total_layer_acts += u.activations;
        if (total_layer_acts == 0) total_layer_acts = 1;

        auto calc_cov = [&](int k) {
            uint64_t k_acts = 0;
            for (int i = 0; i < k && i < (int)layer_exp.size(); i++) {
                k_acts += layer_exp[i].activations;
            }
            return (double)k_acts / (double)total_layer_acts * 100.0;
        };

        printf("Layer %-2d | %9.1f%% | %9.1f%% | %9.1f%% | %9.1f%% | %9.1f%%\n",
            l, calc_cov(1), calc_cov(2), calc_cov(4), calc_cov(8), calc_cov(16));
    }

    printf("\n================================================================================\n");
    printf("Manifest v2: placement-aware bundle-admission greedy selection\n");
    printf("================================================================================\n");

    if (!trace_jsonl_path.empty()) {
        std::ofstream tr(trace_jsonl_path);
        tr << "{\"_header\": true, \"format\": 1, \"model\": \"Qwen3.6-35B-A3B-APEX-Compact\", ";
        tr << "\"n_layers\": " << pctx.n_layers << ", \"n_experts\": " << n_experts << ", ";
        tr << "\"top_k\": " << top_k << ", \"n_tokens\": " << n_gen << ", \"device\": \"CUDA0\", ";
        tr << "\"callback_matches_canonical\": true}\n";
        for (int l = 0; l < pctx.n_layers; l++) {
            for (size_t t = 0; t < pctx.layer_routes[l].size(); t++) {
                tr << "{\"request_id\": \"prof-1\", ";
                tr << "\"sequence_index\": " << t << ", ";
                tr << "\"layer\": " << l << ", ";
                tr << "\"top_k\": " << (int)pctx.layer_routes[l][t].size() << ", ";
                tr << "\"experts\": [";
                for (size_t ei = 0; ei < pctx.layer_routes[l][t].size(); ei++) {
                    if (ei > 0) tr << ", ";
                    tr << pctx.layer_routes[l][t][ei];
                }
                tr << "]}\n";
            }
        }
        printf("TG1 route trace written to %s\n", trace_jsonl_path.c_str());
    }
    if (!dump_routes_path.empty()) {
        std::ofstream dr(dump_routes_path);
        for (int l = 0; l < pctx.n_layers; l++) {
            for (size_t t = 0; t < pctx.layer_routes[l].size(); t++) {
                dr << l << ' ' << t;
                for (int32_t e : pctx.layer_routes[l][t]) {
                    dr << ' ' << e;
                }
                dr << '\n';
            }
        }
        printf("Raw routes written to %s\n", dump_routes_path.c_str());
    }

    const int top_k_model = top_k; // 8 routes per token for this model

    // inverted index: occurrences[l][e] = decode token indices routing to expert e
    std::vector<std::vector<std::vector<int>>> occurrences(pctx.n_layers,
        std::vector<std::vector<int>>(n_experts));
    std::vector<std::vector<int>> hit_count(pctx.n_layers);
    for (int l = 0; l < pctx.n_layers; l++) {
        const size_t n_tok = pctx.layer_routes[l].size();
        hit_count[l].assign(n_tok, 0);
        for (size_t t = 0; t < n_tok; t++) {
            for (int exp : pctx.layer_routes[l][t]) {
                if (exp >= 0 && exp < n_experts) {
                    occurrences[l][exp].push_back((int) t);
                }
            }
        }
    }

    // per (layer,expert) activation counts, for the manifest stats field
    std::vector<std::vector<uint64_t>> acts(pctx.n_layers, std::vector<uint64_t>(n_experts, 0));
    for (int l = 0; l < pctx.n_layers; l++) {
        for (int e = 0; e < n_experts; e++) {
            acts[l][e] = occurrences[l][e].size();
        }
    }

    std::vector<std::set<int>> chosen(pctx.n_layers);
    const std::vector<size_t> tiers_mb = { 64, 128, 256, 512, 1024 };
    for (size_t tier_mb : tiers_mb) {
        const size_t max_bundles = (tier_mb * 1024 * 1024) / expert_bundle_bytes;
        std::vector<std::set<int>> sel = chosen;   // tiers are cumulative
        std::vector<std::vector<int>> hc = hit_count; // scratch |routes_t intersect sel_l|
        for (int l = 0; l < pctx.n_layers; l++) {
            for (int e : chosen[l]) {
                for (int t : occurrences[l][e]) {
                    hc[l][t]++;
                }
            }
        }
        // potential per decode step, as a function of how many of its top_k
        // routes are already pinned: linear progress below the admission
        // thresholds, then the admission payout. A strictly-threshold gain
        // cannot start - it only credits the 7th and 8th expert, so nothing is
        // ever selectable from an empty selection.
        const double boot_credit = 0.02;
        auto phi = [&](int c) {
            double p = boot_credit * (double) c;
            if (c >= top_k_model - 1) p += w_hetero;
            if (c >= top_k_model) p += w_full;
            return p;
        };
        size_t used = 0;
        while (used < max_bundles) {
            int best_l = -1, best_e = -1;
            double best_gain = 0.0;
            for (int l = 0; l < pctx.n_layers; l++) {
                if (!layer_eligible[l]) {
                    continue;
                }
                for (int e = 0; e < n_experts; e++) {
                    if (sel[l].count(e)) {
                        continue;
                    }
                    double gain = 0.0;
                    for (int t : occurrences[l][e]) {
                        gain += phi(hc[l][t] + 1) - phi(hc[l][t]);
                    }
                    if (gain > best_gain) {
                        best_gain = gain;
                        best_l = l;
                        best_e = e;
                    }
                }
            }
            if (best_l < 0 || best_gain <= 0.0) {
                break;
            }
            for (int t : occurrences[best_l][best_e]) {
                hc[best_l][t]++;
            }
            sel[best_l].insert(best_e);
            used++;
        }
        chosen = sel;

        // tier projections + per-entry bundle stats over the full selection
        uint64_t proj_full = 0, proj_seven = 0;
        std::vector<std::vector<uint64_t>> entry_full(pctx.n_layers, std::vector<uint64_t>(n_experts, 0));
        std::vector<std::vector<uint64_t>> entry_seven(pctx.n_layers, std::vector<uint64_t>(n_experts, 0));
        for (int l = 0; l < pctx.n_layers; l++) {
            for (size_t t = 0; t < pctx.layer_routes[l].size(); t++) {
                int c = 0;
                for (int exp : pctx.layer_routes[l][t]) {
                    if (exp >= 0 && exp < n_experts && sel[l].count(exp)) {
                        c++;
                    }
                }
                if (c == top_k_model) {
                    proj_full++;
                    for (int exp : pctx.layer_routes[l][t]) {
                        if (exp >= 0 && exp < n_experts) entry_full[l][exp]++;
                    }
                } else if (c == top_k_model - 1) {
                    proj_seven++;
                    for (int exp : pctx.layer_routes[l][t]) {
                        if (exp >= 0 && exp < n_experts && sel[l].count(exp)) entry_seven[l][exp]++;
                    }
                }
            }
        }

        std::string manifest_file = out_manifest_prefix + "_" + std::to_string(tier_mb) + "mb.json";
        std::ofstream ofs(manifest_file);
        ofs << "{\n";
        ofs << "  \"format\": 2,\n";
        ofs << "  \"admission\": \"7of8\",\n";
        ofs << "  \"top_k\": " << top_k_model << ",\n";
        ofs << "  \"w_full\": " << w_full << ",\n";
        ofs << "  \"w_hetero\": " << w_hetero << ",\n";
        ofs << "  \"placement_cache_mib\": " << cache_mib << ",\n";
        ofs << "  \"tier_mb\": " << tier_mb << ",\n";
        ofs << "  \"bundle_bytes\": " << expert_bundle_bytes << ",\n";
        ofs << "  \"eligible_layers\": [";
        bool first_l = true;
        for (int l = 0; l < pctx.n_layers; l++) {
            if (layer_eligible[l]) {
                ofs << (first_l ? "" : ", ") << l;
                first_l = false;
            }
        }
        ofs << "],\n";

        // emit sorted by (layer, expert_id)
        std::vector<std::pair<int, int>> entries;
        for (int l = 0; l < pctx.n_layers; l++) {
            for (int e : sel[l]) {
                entries.push_back({ l, e });
            }
        }
        std::sort(entries.begin(), entries.end());
        ofs << "  \"pinned_experts\": [\n";
        for (size_t i = 0; i < entries.size(); i++) {
            const int l = entries[i].first;
            const int e = entries[i].second;
            ofs << "    {\"layer\": " << l << ", \"expert_id\": " << e
                << ", \"activations\": " << acts[l][e]
                << ", \"bundle_full_hits\": " << entry_full[l][e]
                << ", \"bundle_seven_hits\": " << entry_seven[l][e] << "}"
                << (i + 1 < entries.size() ? ",\n" : "\n");
        }
        ofs << "  ]\n}\n";
        ofs.close();

        const double total_steps = (double) n_gen * pctx.n_layers;
        printf("Tier %4zu MiB: %4zu pinned | projected 8/8 admissions %llu (%.1f%% of decode steps) | projected 7/8 admissions %llu -> %s\n",
            tier_mb,
            entries.size(),
            (unsigned long long) proj_full,
            total_steps > 0.0 ? 100.0 * (double) proj_full / total_steps : 0.0,
            (unsigned long long) proj_seven,
            manifest_file.c_str());
    }

    printf("================================================================================\n");

    llama_backend_free();
    return 0;
}
