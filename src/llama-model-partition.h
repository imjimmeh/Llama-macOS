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
