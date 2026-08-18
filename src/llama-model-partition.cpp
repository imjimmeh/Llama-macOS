#include "llama-model-partition.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int64_t llama_ffn_partition_align(
        const ggml_tensor * up,
        const ggml_tensor * gate,
        const ggml_tensor * down,
        float fraction) {
    if (!up) {
        return 0;
    }

    const int64_t n_ff = up->ne[1];
    const int64_t raw  = (int64_t)(n_ff * fraction);

    int64_t blk = 1;
    if (up)   blk = std::max(blk, (int64_t)ggml_blck_size(up->type));
    if (gate) blk = std::max(blk, (int64_t)ggml_blck_size(gate->type));
    if (down) blk = std::max(blk, (int64_t)ggml_blck_size(down->type));

    const int64_t aligned = (raw / blk) * blk;
    if (aligned <= 0 || aligned >= n_ff) {
        return 0;
    }

    return aligned;
}

int64_t llama_ffn_partition_align(const llama_layer & layer, float fraction) {
    return llama_ffn_partition_align(layer.ffn_up, layer.ffn_gate, layer.ffn_down, fraction);
}

bool llama_ffn_can_partition(
        const ggml_tensor * up,
        const ggml_tensor * gate,
        const ggml_tensor * down,
        llm_ffn_gate_type type_gate) {
    if (!up || !gate || !down) {
        return false;
    }

    if (type_gate != LLM_FFN_PAR) {
        return false;
    }

    if (up->ne[1] != gate->ne[1]) {
        return false;
    }

    if (down->ne[0] != up->ne[1]) {
        return false;
    }

    if (up->type == GGML_TYPE_NVFP4 ||
        gate->type == GGML_TYPE_NVFP4 ||
        down->type == GGML_TYPE_NVFP4) {
        return false;
    }

    return true;
}

bool llama_ffn_can_partition(const llama_layer & layer, llm_ffn_gate_type type_gate) {
    if (layer.ffn_up_b || layer.ffn_gate_b || layer.ffn_down_b) {
        return false;
    }
    return llama_ffn_can_partition(layer.ffn_up, layer.ffn_gate, layer.ffn_down, type_gate);
}

std::unique_ptr<llama_ffn_partition_set> llama_ffn_partition_build(
        llama_model & model,
        float fraction,
        ggml_backend_dev_t accel_dev) {

    if (fraction <= 0.0f || accel_dev == nullptr) {
        return nullptr;
    }

    struct partition_candidate {
        const char * suffix;
        size_t il;
        ggml_tensor * up;
        ggml_tensor * gate;
        ggml_tensor * down;
        int64_t n_ff_accel;
    };

    std::vector<partition_candidate> candidates;
    const size_t n_layer = model.layers.size();

    auto check_and_add = [&](const char * suffix, size_t il, ggml_tensor * up, ggml_tensor * gate, ggml_tensor * down) {
        if (!llama_ffn_can_partition(up, gate, down, LLM_FFN_PAR)) {
            return;
        }

        // skip layers with clamping in hparams
        if (il < model.hparams.swiglu_clamp_shexp.size() && model.hparams.swiglu_clamp_shexp[il] > 1e-6f) {
            return;
        }

        // only partition tensors whose weights reside in host memory
        if (up->buffer && !ggml_backend_buffer_is_host(up->buffer)) {
            return;
        }

        const int64_t n_ff_accel = llama_ffn_partition_align(up, gate, down, fraction);
        if (n_ff_accel <= 0) {
            return;
        }

        candidates.push_back({ suffix, il, up, gate, down, n_ff_accel });
    };

    for (size_t il = 0; il < n_layer; ++il) {
        auto & layer = model.layers[il];

        // 1. Primary dense FFN branch
        check_and_add("ffn", il, layer.ffn_up, layer.ffn_gate, layer.ffn_down);

        // 2. Shared expert branch
        check_and_add("shexp", il, layer.ffn_up_shexp, layer.ffn_gate_shexp, layer.ffn_down_shexp);
    }

    if (candidates.empty()) {
        return nullptr;
    }

    auto result = std::make_unique<llama_ffn_partition_set>();

    // create metadata contexts
    const size_t n_tensors_accel = candidates.size() * 3;
    const size_t n_tensors_cpu   = candidates.size() * 1;

    struct ggml_init_params params_accel = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors_accel + 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    result->ctx_accel.reset(ggml_init(params_accel));

    struct ggml_init_params params_cpu = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors_cpu + 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    result->ctx_cpu.reset(ggml_init(params_cpu));

    // create tensor headers and partition objects
    struct target_entry {
        std::unique_ptr<llama_ffn_partition> part;
        const partition_candidate * cand;
    };
    std::vector<target_entry> targets;
    targets.reserve(candidates.size());

    for (const auto & cand : candidates) {
        auto part = std::make_unique<llama_ffn_partition>();
        part->n_ff_accel = cand.n_ff_accel;
        part->dev        = accel_dev;

        const int64_t n_ff_accel = cand.n_ff_accel;
        const int64_t n_ff       = cand.up->ne[1];
        const int64_t n_ff_cpu   = n_ff - n_ff_accel;

        char name[128];

        if (strcmp(cand.suffix, "shexp") == 0) {
            snprintf(name, sizeof(name), "blk.%zu.ffn_up_shexp.hsplit_gpu", cand.il);
            part->up_accel = ggml_new_tensor_2d(result->ctx_accel.get(), cand.up->type, cand.up->ne[0], n_ff_accel);
            ggml_set_name(part->up_accel, name);

            snprintf(name, sizeof(name), "blk.%zu.ffn_gate_shexp.hsplit_gpu", cand.il);
            part->gate_accel = ggml_new_tensor_2d(result->ctx_accel.get(), cand.gate->type, cand.gate->ne[0], n_ff_accel);
            ggml_set_name(part->gate_accel, name);

            snprintf(name, sizeof(name), "blk.%zu.ffn_down_shexp.hsplit_gpu", cand.il);
            part->down_accel = ggml_new_tensor_2d(result->ctx_accel.get(), cand.down->type, n_ff_accel, cand.down->ne[1]);
            ggml_set_name(part->down_accel, name);

            snprintf(name, sizeof(name), "blk.%zu.ffn_down_shexp.hsplit_cpu", cand.il);
            part->down_cpu = ggml_new_tensor_2d(result->ctx_cpu.get(), cand.down->type, n_ff_cpu, cand.down->ne[1]);
            ggml_set_name(part->down_cpu, name);
        } else {
            snprintf(name, sizeof(name), "blk.%zu.ffn_up.hsplit_gpu", cand.il);
            part->up_accel = ggml_new_tensor_2d(result->ctx_accel.get(), cand.up->type, cand.up->ne[0], n_ff_accel);
            ggml_set_name(part->up_accel, name);

            snprintf(name, sizeof(name), "blk.%zu.ffn_gate.hsplit_gpu", cand.il);
            part->gate_accel = ggml_new_tensor_2d(result->ctx_accel.get(), cand.gate->type, cand.gate->ne[0], n_ff_accel);
            ggml_set_name(part->gate_accel, name);

            snprintf(name, sizeof(name), "blk.%zu.ffn_down.hsplit_gpu", cand.il);
            part->down_accel = ggml_new_tensor_2d(result->ctx_accel.get(), cand.down->type, n_ff_accel, cand.down->ne[1]);
            ggml_set_name(part->down_accel, name);

            snprintf(name, sizeof(name), "blk.%zu.ffn_down.hsplit_cpu", cand.il);
            part->down_cpu = ggml_new_tensor_2d(result->ctx_cpu.get(), cand.down->type, n_ff_cpu, cand.down->ne[1]);
            ggml_set_name(part->down_cpu, name);
        }

        targets.push_back({ std::move(part), &cand });
    }

    // allocate backend buffers
    ggml_backend_buffer_type_t accel_buft = ggml_backend_dev_buffer_type(accel_dev);
    result->buf_accel.reset(ggml_backend_alloc_ctx_tensors_from_buft(result->ctx_accel.get(), accel_buft));
    if (!result->buf_accel) {
        LLAMA_LOG_ERROR("%s: failed to allocate accelerator buffer for FFN partitions\n", __func__);
        return nullptr;
    }

    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    result->buf_cpu.reset(ggml_backend_alloc_ctx_tensors_from_buft(result->ctx_cpu.get(), cpu_buft));
    if (!result->buf_cpu) {
        LLAMA_LOG_ERROR("%s: failed to allocate CPU buffer for FFN partitions\n", __func__);
        return nullptr;
    }

    // populate tensor data
    for (auto & target : targets) {
        const auto & cand = *target.cand;
        auto & part       = target.part;

        const int64_t n_ff_accel = part->n_ff_accel;
        const int64_t n_ff       = cand.up->ne[1];

        // up_accel
        {
            const size_t size = n_ff_accel * cand.up->nb[1];
            std::vector<uint8_t> tmp(size);
            ggml_backend_tensor_get(cand.up, tmp.data(), 0, size);
            ggml_backend_tensor_set(part->up_accel, tmp.data(), 0, size);
        }

        // gate_accel
        {
            const size_t size = n_ff_accel * cand.gate->nb[1];
            std::vector<uint8_t> tmp(size);
            ggml_backend_tensor_get(cand.gate, tmp.data(), 0, size);
            ggml_backend_tensor_set(part->gate_accel, tmp.data(), 0, size);
        }

        // down_accel and down_cpu (row-by-row packed copy)
        {
            const size_t down_nbytes = ggml_nbytes(cand.down);
            std::vector<uint8_t> down_src(down_nbytes);
            ggml_backend_tensor_get(cand.down, down_src.data(), 0, down_nbytes);

            const int64_t blk_size  = ggml_blck_size(cand.down->type);
            const int64_t type_size = ggml_type_size(cand.down->type);

            const int64_t gpu_row_bytes = (n_ff_accel / blk_size) * type_size;
            const int64_t cpu_row_bytes = ((n_ff - n_ff_accel) / blk_size) * type_size;

            std::vector<uint8_t> down_accel_data(ggml_nbytes(part->down_accel));
            std::vector<uint8_t> down_cpu_data(ggml_nbytes(part->down_cpu));

            for (int64_t r = 0; r < cand.down->ne[1]; ++r) {
                const uint8_t * src_row = down_src.data() + r * cand.down->nb[1];
                uint8_t * dst_gpu_row   = down_accel_data.data() + r * part->down_accel->nb[1];
                uint8_t * dst_cpu_row   = down_cpu_data.data() + r * part->down_cpu->nb[1];

                memcpy(dst_gpu_row, src_row, gpu_row_bytes);
                memcpy(dst_cpu_row, src_row + gpu_row_bytes, cpu_row_bytes);
            }

            ggml_backend_tensor_set(part->down_accel, down_accel_data.data(), 0, down_accel_data.size());
            ggml_backend_tensor_set(part->down_cpu, down_cpu_data.data(), 0, down_cpu_data.size());
        }

        result->partitions[cand.up] = std::move(part);
    }

    return result;
}

void llama_moe_partition::promote_expert(int32_t candidate_id, int32_t slot_id) {
    if (slot_id < 0 || slot_id >= n_expert_accel || candidate_id < 0 || candidate_id >= n_expert_total) {
        return;
    }

    int32_t old_expert = slot_to_expert[slot_id];
    if (old_expert == candidate_id) {
        return;
    }

    auto copy_slice = [&](ggml_tensor * dst_accel, const ggml_tensor * src_host) {
        if (!dst_accel || !src_host) {
            return;
        }
        const size_t slice_size = dst_accel->nb[2];
        const size_t src_offset = (size_t)candidate_id * src_host->nb[2];
        const size_t dst_offset = (size_t)slot_id * dst_accel->nb[2];

        std::vector<uint8_t> tmp(slice_size);
        ggml_backend_tensor_get(src_host, tmp.data(), src_offset, slice_size);
        ggml_backend_tensor_set(dst_accel, tmp.data(), dst_offset, slice_size);
    };

    if (gate_up_exps_accel) {
        copy_slice(gate_up_exps_accel, gate_up_exps_host);
        copy_slice(gate_up_exps_b_accel, gate_up_exps_b_host);
    } else {
        copy_slice(gate_exps_accel, gate_exps_host);
        copy_slice(gate_exps_b_accel, gate_exps_b_host);
        copy_slice(gate_exps_s_accel, gate_exps_s_host);
        copy_slice(up_exps_accel, up_exps_host);
        copy_slice(up_exps_b_accel, up_exps_b_host);
        copy_slice(up_exps_s_accel, up_exps_s_host);
    }

    copy_slice(down_exps_accel, down_exps_host);
    copy_slice(down_exps_b_accel, down_exps_b_host);
    copy_slice(down_exps_s_accel, down_exps_s_host);

    // Update slot mappings
    if (old_expert >= 0 && old_expert < n_expert_total) {
        expert_to_slot[old_expert] = -1;
    }
    slot_to_expert[slot_id] = candidate_id;
    expert_to_slot[candidate_id] = slot_id;

    // Update accelerator lookup tables
    if (expert_gpu_mask_table && expert_cpu_mask_table && expert_slot_table) {
        std::vector<float> gpu_mask(n_expert_total, 0.0f);
        std::vector<float> cpu_mask(n_expert_total, 1.0f);
        std::vector<int32_t> slot_table(n_expert_total, 0);
        for (int32_t e = 0; e < n_expert_total; ++e) {
            int32_t s = expert_to_slot[e];
            if (s >= 0) {
                gpu_mask[e]   = 1.0f;
                cpu_mask[e]   = 0.0f;
                slot_table[e] = s;
            } else {
                gpu_mask[e]   = 0.0f;
                cpu_mask[e]   = 1.0f;
                slot_table[e] = 0;
            }
        }
        ggml_backend_tensor_set(expert_gpu_mask_table, gpu_mask.data(), 0, gpu_mask.size() * sizeof(float));
        ggml_backend_tensor_set(expert_cpu_mask_table, cpu_mask.data(), 0, cpu_mask.size() * sizeof(float));
        ggml_backend_tensor_set(expert_slot_table, slot_table.data(), 0, slot_table.size() * sizeof(int32_t));
    }
}

void llama_moe_tracker::record_access(int32_t layer_idx, const int32_t * selected_experts, int32_t n_expert_used, const llama_moe_partition & part) {
    if (layer_idx < 0 || (size_t)layer_idx >= layers.size()) {
        return;
    }

    auto & st = layers[layer_idx];
    for (int32_t i = 0; i < n_expert_used; ++i) {
        int32_t exp = selected_experts[i];
        if (exp >= 0 && (size_t)exp < st.access_count.size()) {
            st.access_count[exp]++;
            if (part.expert_to_slot[exp] >= 0) {
                resident_hits++;
            } else {
                non_resident_misses++;
            }
        }
    }
}

void llama_moe_tracker::rebalance(llama_moe_partition_set & set) {
    tokens_in_epoch = 0;

    for (size_t il = 0; il < layers.size() && il < set.ordered_partitions.size(); ++il) {
        auto * part = set.ordered_partitions[il];
        if (!part || part->n_expert_accel <= 0) {
            continue;
        }

        auto & st = layers[il];
        const int64_t n_tot = part->n_expert_total;

        // 1. Decay EMA scores with current epoch accesses
        for (int64_t e = 0; e < n_tot; ++e) {
            st.scores[e] = st.scores[e] * ema_alpha + (float)st.access_count[e];
            st.access_count[e] = 0;
        }

        // 2. Find hottest non-resident candidate
        int32_t best_candidate = -1;
        float max_cand_score = -1.0f;
        for (int64_t e = 0; e < n_tot; ++e) {
            if (part->expert_to_slot[e] < 0 && st.scores[e] > max_cand_score) {
                max_cand_score = st.scores[e];
                best_candidate = (int32_t)e;
            }
        }

        // 3. Find coldest resident expert
        int32_t coldest_slot = -1;
        float min_resident_score = 1e9f;
        for (int32_t s = 0; s < part->n_expert_accel; ++s) {
            int32_t exp = part->slot_to_expert[s];
            if (exp >= 0 && (size_t)exp < st.scores.size()) {
                if (st.scores[exp] < min_resident_score) {
                    min_resident_score = st.scores[exp];
                    coldest_slot = s;
                }
            }
        }

        // 4. Check hysteresis promotion threshold
        if (best_candidate >= 0 && coldest_slot >= 0 && max_cand_score > (min_resident_score + hysteresis)) {
            part->promote_expert(best_candidate, coldest_slot);
            total_promotions++;
        }
    }
}

void llama_moe_tracker::print_stats() const {
    const uint64_t total_evals = resident_hits + non_resident_misses;
    const double hit_pct = total_evals > 0 ? (100.0 * (double)resident_hits / (double)total_evals) : 0.0;

    LLAMA_LOG_INFO("\n--- MoE Heterogeneous Expert Residency Telemetry ---\n");
    LLAMA_LOG_INFO("  Total tokens processed:       %llu\n", (unsigned long long)total_tokens);
    LLAMA_LOG_INFO("  Resident VRAM hits:           %llu (%.1f%%)\n", (unsigned long long)resident_hits, hit_pct);
    LLAMA_LOG_INFO("  Non-resident CPU misses:      %llu (%.1f%%)\n", (unsigned long long)non_resident_misses, 100.0 - hit_pct);
    LLAMA_LOG_INFO("  Background expert promotions: %llu\n", (unsigned long long)total_promotions);
    LLAMA_LOG_INFO("  Zero PCIe weight transfers on decode critical path.\n");
    LLAMA_LOG_INFO("----------------------------------------------------\n\n");
}

void llama_moe_partition_set::record_access(int32_t layer_idx, const int32_t * selected_experts, int32_t n_expert_used) {
    if (!tracker || layer_idx < 0 || (size_t)layer_idx >= ordered_partitions.size()) {
        return;
    }
    auto * part = ordered_partitions[layer_idx];
    if (part) {
        tracker->record_access(layer_idx, selected_experts, n_expert_used, *part);
    }
}

void llama_moe_partition_set::check_rebalance() {
    if (!tracker) {
        return;
    }
    tracker->total_tokens++;
    tracker->tokens_in_epoch++;
    if (tracker->rebalance_period > 0 && tracker->tokens_in_epoch >= tracker->rebalance_period) {
        tracker->rebalance(*this);
    }
}

void llama_moe_partition_set::print_stats() const {
    if (tracker) {
        tracker->print_stats();
    }
}

std::unique_ptr<llama_moe_partition_set> llama_moe_partition_build(
        llama_model & model,
        float fraction,
        size_t hot_vram_bytes,
        ggml_backend_dev_t accel_dev,
        int32_t rebalance_period) {

    if ((fraction <= 0.0f && hot_vram_bytes == 0) || accel_dev == nullptr) {
        return nullptr;
    }

    struct moe_candidate {
        size_t il;
        ggml_tensor * up;
        ggml_tensor * gate;
        ggml_tensor * down;
        ggml_tensor * gate_up;

        ggml_tensor * up_s;
        ggml_tensor * gate_s;
        ggml_tensor * down_s;

        ggml_tensor * up_b;
        ggml_tensor * gate_b;
        ggml_tensor * down_b;
        ggml_tensor * gate_up_b;

        int64_t n_expert_accel;
        int64_t n_expert_total;
        size_t  bytes_per_expert;
    };

    std::vector<moe_candidate> candidates;
    const size_t n_layer = model.layers.size();
    size_t total_bytes_per_exp_all_layers = 0;

    for (size_t il = 0; il < n_layer; ++il) {
        auto & layer = model.layers[il];

        ggml_tensor * up      = layer.ffn_up_exps;
        ggml_tensor * gate    = layer.ffn_gate_exps;
        ggml_tensor * down    = layer.ffn_down_exps;
        ggml_tensor * gate_up = layer.ffn_gate_up_exps;

        ggml_tensor * key = up ? up : gate_up;
        if (!key || !down) {
            continue;
        }

        // only partition tensors whose weights reside in host memory
        if (key->buffer && !ggml_backend_buffer_is_host(key->buffer)) {
            continue;
        }

        const int64_t n_expert = key->ne[2];
        size_t bpe = 0;
        if (gate_up) {
            bpe += gate_up->nb[2];
        } else {
            if (up)   bpe += up->nb[2];
            if (gate) bpe += gate->nb[2];
        }
        if (down) bpe += down->nb[2];
        if (layer.ffn_up_exps_s)   bpe += layer.ffn_up_exps_s->nb[2];
        if (layer.ffn_gate_exps_s) bpe += layer.ffn_gate_exps_s->nb[2];
        if (layer.ffn_down_exps_s) bpe += layer.ffn_down_exps_s->nb[2];

        total_bytes_per_exp_all_layers += bpe;

        candidates.push_back({
            il,
            up, gate, down, gate_up,
            layer.ffn_up_exps_s, layer.ffn_gate_exps_s, layer.ffn_down_exps_s,
            layer.ffn_up_exps_b, layer.ffn_gate_exps_b, layer.ffn_down_exps_b, layer.ffn_gate_up_exps_b,
            0, n_expert, bpe
        });
    }

    if (candidates.empty()) {
        return nullptr;
    }

    // Determine n_expert_accel per layer
    for (auto & cand : candidates) {
        if (hot_vram_bytes > 0 && total_bytes_per_exp_all_layers > 0) {
            cand.n_expert_accel = (int64_t)(hot_vram_bytes / total_bytes_per_exp_all_layers);
        } else {
            cand.n_expert_accel = (int64_t)(cand.n_expert_total * fraction);
        }
        if (cand.n_expert_accel <= 0) {
            cand.n_expert_accel = 1;
        }
        if (cand.n_expert_accel >= cand.n_expert_total) {
            cand.n_expert_accel = cand.n_expert_total - 1;
        }
    }

    auto result = std::make_unique<llama_moe_partition_set>();
    result->ordered_partitions.resize(n_layer, nullptr);

    // create accelerator metadata context (each layer has up to 8 tensors + 3 lookup tables)
    const size_t n_tensors_accel = candidates.size() * 11;
    struct ggml_init_params params_accel = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors_accel + 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    result->ctx_accel.reset(ggml_init(params_accel));

    struct target_moe_entry {
        std::unique_ptr<llama_moe_partition> part;
        const moe_candidate * cand;
    };
    std::vector<target_moe_entry> targets;
    targets.reserve(candidates.size());

    for (const auto & cand : candidates) {
        auto part = std::make_unique<llama_moe_partition>();
        part->n_expert_accel = cand.n_expert_accel;
        part->n_expert_total = cand.n_expert_total;
        part->layer_id       = (int32_t)cand.il;
        part->dev            = accel_dev;

        part->slot_to_expert.resize(cand.n_expert_accel);
        part->expert_to_slot.resize(cand.n_expert_total, -1);
        for (int32_t s = 0; s < cand.n_expert_accel; ++s) {
            part->slot_to_expert[s] = s;
            part->expert_to_slot[s] = s;
        }

        // Host references for dynamic swapping
        part->gate_exps_host      = cand.gate;
        part->up_exps_host        = cand.up;
        part->down_exps_host      = cand.down;
        part->gate_up_exps_host   = cand.gate_up;
        part->gate_exps_s_host    = cand.gate_s;
        part->up_exps_s_host      = cand.up_s;
        part->down_exps_s_host    = cand.down_s;
        part->gate_exps_b_host    = cand.gate_b;
        part->up_exps_b_host      = cand.up_b;
        part->down_exps_b_host    = cand.down_b;
        part->gate_up_exps_b_host = cand.gate_up_b;

        char name[128];
        const int64_t ne_accel = cand.n_expert_accel;

        // Lookup tables
        snprintf(name, sizeof(name), "blk.%zu.moe_gpu_mask_table", cand.il);
        part->expert_gpu_mask_table = ggml_new_tensor_2d(result->ctx_accel.get(), GGML_TYPE_F32, 1, cand.n_expert_total);
        ggml_set_name(part->expert_gpu_mask_table, name);

        snprintf(name, sizeof(name), "blk.%zu.moe_cpu_mask_table", cand.il);
        part->expert_cpu_mask_table = ggml_new_tensor_2d(result->ctx_accel.get(), GGML_TYPE_F32, 1, cand.n_expert_total);
        ggml_set_name(part->expert_cpu_mask_table, name);

        snprintf(name, sizeof(name), "blk.%zu.moe_slot_table", cand.il);
        part->expert_slot_table = ggml_new_tensor_2d(result->ctx_accel.get(), GGML_TYPE_I32, 1, cand.n_expert_total);
        ggml_set_name(part->expert_slot_table, name);

        if (cand.gate_up) {
            snprintf(name, sizeof(name), "blk.%zu.ffn_gate_up_exps.moe_gpu", cand.il);
            part->gate_up_exps_accel = ggml_new_tensor_3d(result->ctx_accel.get(), cand.gate_up->type, cand.gate_up->ne[0], cand.gate_up->ne[1], ne_accel);
            ggml_set_name(part->gate_up_exps_accel, name);
        } else {
            if (cand.up) {
                snprintf(name, sizeof(name), "blk.%zu.ffn_up_exps.moe_gpu", cand.il);
                part->up_exps_accel = ggml_new_tensor_3d(result->ctx_accel.get(), cand.up->type, cand.up->ne[0], cand.up->ne[1], ne_accel);
                ggml_set_name(part->up_exps_accel, name);
            }
            if (cand.gate) {
                snprintf(name, sizeof(name), "blk.%zu.ffn_gate_exps.moe_gpu", cand.il);
                part->gate_exps_accel = ggml_new_tensor_3d(result->ctx_accel.get(), cand.gate->type, cand.gate->ne[0], cand.gate->ne[1], ne_accel);
                ggml_set_name(part->gate_exps_accel, name);
            }
        }

        snprintf(name, sizeof(name), "blk.%zu.ffn_down_exps.moe_gpu", cand.il);
        part->down_exps_accel = ggml_new_tensor_3d(result->ctx_accel.get(), cand.down->type, cand.down->ne[0], cand.down->ne[1], ne_accel);
        ggml_set_name(part->down_exps_accel, name);

        // scales
        if (cand.up_s) {
            snprintf(name, sizeof(name), "blk.%zu.ffn_up_exps_s.moe_gpu", cand.il);
            part->up_exps_s_accel = ggml_new_tensor_3d(result->ctx_accel.get(), cand.up_s->type, cand.up_s->ne[0], cand.up_s->ne[1], ne_accel);
            ggml_set_name(part->up_exps_s_accel, name);
        }
        if (cand.gate_s) {
            snprintf(name, sizeof(name), "blk.%zu.ffn_gate_exps_s.moe_gpu", cand.il);
            part->gate_exps_s_accel = ggml_new_tensor_3d(result->ctx_accel.get(), cand.gate_s->type, cand.gate_s->ne[0], cand.gate_s->ne[1], ne_accel);
            ggml_set_name(part->gate_exps_s_accel, name);
        }
        if (cand.down_s) {
            snprintf(name, sizeof(name), "blk.%zu.ffn_down_exps_s.moe_gpu", cand.il);
            part->down_exps_s_accel = ggml_new_tensor_3d(result->ctx_accel.get(), cand.down_s->type, cand.down_s->ne[0], cand.down_s->ne[1], ne_accel);
            ggml_set_name(part->down_exps_s_accel, name);
        }

        targets.push_back({ std::move(part), &cand });
    }

    ggml_backend_buffer_type_t accel_buft = ggml_backend_dev_buffer_type(accel_dev);
    result->buf_accel.reset(ggml_backend_alloc_ctx_tensors_from_buft(result->ctx_accel.get(), accel_buft));
    if (!result->buf_accel) {
        LLAMA_LOG_ERROR("%s: failed to allocate accelerator buffer for MoE partitions\n", __func__);
        return nullptr;
    }

    auto copy_expert_slice = [](const ggml_tensor * src, ggml_tensor * dst, int64_t n_exp) {
        if (!src || !dst) {
            return;
        }
        const size_t size = (size_t)n_exp * src->nb[2];
        std::vector<uint8_t> tmp(size);
        ggml_backend_tensor_get(src, tmp.data(), 0, size);
        ggml_backend_tensor_set(dst, tmp.data(), 0, size);
    };

    // Initialize tracker
    result->tracker = std::make_unique<llama_moe_tracker>();
    result->tracker->rebalance_period = rebalance_period;
    result->tracker->layers.resize(n_layer);

    for (auto & target : targets) {
        const auto & cand = *target.cand;
        auto & part       = target.part;
        const int64_t ne  = cand.n_expert_accel;

        // Initialize lookup tables on device
        std::vector<float> gpu_mask(cand.n_expert_total, 0.0f);
        std::vector<float> cpu_mask(cand.n_expert_total, 1.0f);
        std::vector<int32_t> slot_table(cand.n_expert_total, 0);
        for (int32_t e = 0; e < cand.n_expert_total; ++e) {
            int32_t s = part->expert_to_slot[e];
            if (s >= 0) {
                gpu_mask[e]   = 1.0f; // is_gpu
                cpu_mask[e]   = 0.0f; // is_cpu
                slot_table[e] = s;    // resident slot index
            } else {
                gpu_mask[e]   = 0.0f; // is_gpu
                cpu_mask[e]   = 1.0f; // is_cpu
                slot_table[e] = 0;    // safe clamped slot
            }
        }
        ggml_backend_tensor_set(part->expert_gpu_mask_table, gpu_mask.data(), 0, gpu_mask.size() * sizeof(float));
        ggml_backend_tensor_set(part->expert_cpu_mask_table, cpu_mask.data(), 0, cpu_mask.size() * sizeof(float));
        ggml_backend_tensor_set(part->expert_slot_table, slot_table.data(), 0, slot_table.size() * sizeof(int32_t));

        // Copy initial resident slices
        if (cand.gate_up) {
            copy_expert_slice(cand.gate_up, part->gate_up_exps_accel, ne);
        } else {
            copy_expert_slice(cand.up,   part->up_exps_accel, ne);
            copy_expert_slice(cand.gate, part->gate_exps_accel, ne);
        }
        copy_expert_slice(cand.down, part->down_exps_accel, ne);

        copy_expert_slice(cand.up_s,   part->up_exps_s_accel, ne);
        copy_expert_slice(cand.gate_s, part->gate_exps_s_accel, ne);
        copy_expert_slice(cand.down_s, part->down_exps_s_accel, ne);

        // Setup tracker stats for this layer
        auto & st = result->tracker->layers[cand.il];
        st.scores.resize(cand.n_expert_total, 0.0f);
        st.access_count.resize(cand.n_expert_total, 0);

        ggml_tensor * key = cand.up ? cand.up : cand.gate_up;
        result->ordered_partitions[cand.il] = part.get();
        result->partitions[key] = std::move(part);
    }

    return result;
}
