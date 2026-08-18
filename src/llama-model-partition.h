#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "llama-graph.h"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

struct llama_layer;
struct llama_model;

struct llama_ffn_partition {
    int64_t n_ff_accel = 0;

    // persistent accelerator-resident copies (packed)
    ggml_tensor * up_accel   = nullptr;
    ggml_tensor * gate_accel = nullptr;
    ggml_tensor * down_accel = nullptr;

    // packed CPU complement for down (avoids non-contiguous stride issues)
    ggml_tensor * down_cpu   = nullptr;

    ggml_backend_dev_t dev = nullptr;
};

struct llama_ffn_partition_set {
    ggml_context_ptr ctx_accel;
    ggml_context_ptr ctx_cpu;

    ggml_backend_buffer_ptr buf_accel;
    ggml_backend_buffer_ptr buf_cpu;

    // Keyed by original host up-tensor (e.g. layer.ffn_up or layer.ffn_up_shexp)
    std::unordered_map<const ggml_tensor *, std::unique_ptr<llama_ffn_partition>> partitions;

    const llama_ffn_partition * find(const ggml_tensor * up) const {
        if (!up) {
            return nullptr;
        }
        auto it = partitions.find(up);
        return it != partitions.end() ? it->second.get() : nullptr;
    }
};

struct llama_moe_tracker;

struct llama_moe_partition {
    int64_t n_expert_accel = 0;
    int64_t n_expert_total = 0;
    int32_t layer_id       = -1;

    // Slot-to-expert and expert-to-slot mappings
    std::vector<int32_t> slot_to_expert; // [n_expert_accel]
    std::vector<int32_t> expert_to_slot; // [n_expert_total], -1 if in host RAM

    // Accelerator-resident expert lookup table tensors
    ggml_tensor * expert_gpu_mask_table = nullptr; // [1, n_expert_total] (F32: 1.0 if in VRAM, 0.0 if in RAM)
    ggml_tensor * expert_cpu_mask_table = nullptr; // [1, n_expert_total] (F32: 0.0 if in VRAM, 1.0 if in RAM)
    ggml_tensor * expert_slot_table     = nullptr; // [1, n_expert_total] (I32: resident slot index, 0 if non-resident)

    // Accelerator-resident expert tensors [n_embd, n_ff, n_expert_accel]
    ggml_tensor * gate_exps_accel    = nullptr;
    ggml_tensor * up_exps_accel      = nullptr;
    ggml_tensor * down_exps_accel    = nullptr;
    ggml_tensor * gate_up_exps_accel = nullptr;

    // Optional per-expert scale tensors
    ggml_tensor * gate_exps_s_accel = nullptr;
    ggml_tensor * up_exps_s_accel   = nullptr;
    ggml_tensor * down_exps_s_accel = nullptr;

    // Optional bias tensors
    ggml_tensor * gate_exps_b_accel    = nullptr;
    ggml_tensor * up_exps_b_accel      = nullptr;
    ggml_tensor * down_exps_b_accel    = nullptr;
    ggml_tensor * gate_up_exps_b_accel = nullptr;

    // References to source host tensors (for background dynamic promotion)
    const ggml_tensor * gate_exps_host      = nullptr;
    const ggml_tensor * up_exps_host        = nullptr;
    const ggml_tensor * down_exps_host      = nullptr;
    const ggml_tensor * gate_up_exps_host   = nullptr;
    const ggml_tensor * gate_exps_s_host    = nullptr;
    const ggml_tensor * up_exps_s_host      = nullptr;
    const ggml_tensor * down_exps_s_host    = nullptr;
    const ggml_tensor * gate_exps_b_host    = nullptr;
    const ggml_tensor * up_exps_b_host      = nullptr;
    const ggml_tensor * down_exps_b_host    = nullptr;
    const ggml_tensor * gate_up_exps_b_host = nullptr;

    ggml_backend_dev_t dev = nullptr;

    void promote_expert(int32_t candidate_id, int32_t slot_id);
};

struct llama_moe_tracker {
    struct layer_stats {
        std::vector<float>   scores;       // EMA frequency score per expert [n_expert_total]
        std::vector<int32_t> access_count; // Access count in current rebalance epoch
    };

    std::vector<layer_stats> layers;
    int32_t rebalance_period = 64;
    float   ema_alpha        = 0.9f;
    float   hysteresis       = 1.5f;
    int32_t tokens_in_epoch  = 0;

    // Telemetry counters
    uint64_t total_tokens        = 0;
    uint64_t resident_hits       = 0;
    uint64_t non_resident_misses = 0;
    uint64_t total_promotions    = 0;

    void record_access(int32_t layer_idx, const int32_t * selected_experts, int32_t n_expert_used, const llama_moe_partition & part);
    void rebalance(struct llama_moe_partition_set & set);
    void print_stats() const;
};

struct llama_moe_partition_set {
    ggml_context_ptr ctx_accel;
    ggml_backend_buffer_ptr buf_accel;

    // Keyed by original host up-tensor (ffn_up_exps or ffn_gate_up_exps)
    std::unordered_map<const ggml_tensor *, std::unique_ptr<llama_moe_partition>> partitions;
    std::vector<llama_moe_partition *> ordered_partitions; // indexed by layer_id

    std::unique_ptr<llama_moe_tracker> tracker;

    const llama_moe_partition * find(const ggml_tensor * up) const {
        if (!up) {
            return nullptr;
        }
        auto it = partitions.find(up);
        return it != partitions.end() ? it->second.get() : nullptr;
    }

    void record_access(int32_t layer_idx, const int32_t * selected_experts, int32_t n_expert_used);
    void check_rebalance();
    void print_stats() const;
};

#include "llama.h"

int64_t LLAMA_API llama_ffn_partition_align(
        const ggml_tensor * up,
        const ggml_tensor * gate,
        const ggml_tensor * down,
        float fraction);

int64_t LLAMA_API llama_ffn_partition_align(
        const llama_layer & layer,
        float fraction);

bool LLAMA_API llama_ffn_can_partition(
        const ggml_tensor * up,
        const ggml_tensor * gate,
        const ggml_tensor * down,
        llm_ffn_gate_type type_gate);

bool LLAMA_API llama_ffn_can_partition(
        const llama_layer & layer,
        llm_ffn_gate_type type_gate);

LLAMA_API std::unique_ptr<llama_ffn_partition_set> llama_ffn_partition_build(
        llama_model & model,
        float fraction,
        ggml_backend_dev_t accel_dev);

LLAMA_API std::unique_ptr<llama_moe_partition_set> llama_moe_partition_build(
        llama_model & model,
        float fraction,
        size_t hot_vram_bytes,
        ggml_backend_dev_t accel_dev,
        int32_t rebalance_period = 64);
