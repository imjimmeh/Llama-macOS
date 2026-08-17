#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "llama-graph.h"

#include <cstdint>
#include <memory>
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

    std::vector<std::unique_ptr<llama_ffn_partition>> partitions;
};

#include "llama.h"

int64_t LLAMA_API llama_ffn_partition_align(
        const llama_layer & layer,
        float fraction);

bool LLAMA_API llama_ffn_can_partition(
        const llama_layer & layer,
        llm_ffn_gate_type type_gate);

LLAMA_API std::unique_ptr<llama_ffn_partition_set> llama_ffn_partition_build(
        llama_model & model,
        float fraction,
        ggml_backend_dev_t accel_dev);
