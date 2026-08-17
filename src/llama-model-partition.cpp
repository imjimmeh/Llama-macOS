#include "llama-model-partition.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int64_t llama_ffn_partition_align(const llama_layer & layer, float fraction) {
    if (!layer.ffn_up) {
        return 0;
    }

    const int64_t n_ff = layer.ffn_up->ne[1];
    const int64_t raw  = (int64_t)(n_ff * fraction);

    int64_t blk = 1;
    if (layer.ffn_up)   blk = std::max(blk, (int64_t)ggml_blck_size(layer.ffn_up->type));
    if (layer.ffn_gate) blk = std::max(blk, (int64_t)ggml_blck_size(layer.ffn_gate->type));
    if (layer.ffn_down) blk = std::max(blk, (int64_t)ggml_blck_size(layer.ffn_down->type));

    const int64_t aligned = (raw / blk) * blk;
    if (aligned <= 0 || aligned >= n_ff) {
        return 0;
    }

    return aligned;
}

bool llama_ffn_can_partition(const llama_layer & layer, llm_ffn_gate_type type_gate) {
    if (!layer.ffn_up || !layer.ffn_down || !layer.ffn_gate) {
        return false;
    }

    if (type_gate != LLM_FFN_PAR) {
        return false;
    }

    if (layer.ffn_up->ne[1] != layer.ffn_gate->ne[1]) {
        return false;
    }

    if (layer.ffn_down->ne[0] != layer.ffn_up->ne[1]) {
        return false;
    }

    if (layer.ffn_up->type == GGML_TYPE_NVFP4 ||
        layer.ffn_gate->type == GGML_TYPE_NVFP4 ||
        layer.ffn_down->type == GGML_TYPE_NVFP4) {
        return false;
    }

    if (layer.ffn_up_b || layer.ffn_gate_b || layer.ffn_down_b) {
        return false;
    }

    return true;
}

std::unique_ptr<llama_ffn_partition_set> llama_ffn_partition_build(
        llama_model & model,
        float fraction,
        ggml_backend_dev_t accel_dev) {

    if (fraction <= 0.0f || accel_dev == nullptr) {
        return nullptr;
    }

    const size_t n_layer = model.layers.size();
    auto result = std::make_unique<llama_ffn_partition_set>();
    result->partitions.resize(n_layer);

    size_t count = 0;
    for (size_t il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        if (!llama_ffn_can_partition(layer, LLM_FFN_PAR)) {
            continue;
        }

        // skip layers with clamping in hparams
        if (il < model.hparams.swiglu_clamp_shexp.size() && model.hparams.swiglu_clamp_shexp[il] > 1e-6f) {
            continue;
        }

        // only partition layers whose weights reside on CPU
        if (layer.ffn_up->buffer && !ggml_backend_buffer_is_host(layer.ffn_up->buffer)) {
            continue;
        }

        const int64_t n_ff_accel = llama_ffn_partition_align(layer, fraction);
        if (n_ff_accel <= 0) {
            continue;
        }

        auto part = std::make_unique<llama_ffn_partition>();
        part->n_ff_accel = n_ff_accel;
        part->dev        = accel_dev;

        result->partitions[il] = std::move(part);
        count++;
    }

    if (count == 0) {
        return nullptr;
    }

    // create metadata contexts
    const size_t n_tensors_accel = count * 3;
    const size_t n_tensors_cpu   = count * 1;

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

    // create tensor headers
    for (size_t il = 0; il < n_layer; ++il) {
        auto & part = result->partitions[il];
        if (!part) {
            continue;
        }

        const auto & layer = model.layers[il];
        const int64_t n_ff_accel = part->n_ff_accel;
        const int64_t n_ff       = layer.ffn_up->ne[1];
        const int64_t n_ff_cpu   = n_ff - n_ff_accel;

        char name[128];

        snprintf(name, sizeof(name), "blk.%zu.ffn_up.hsplit_gpu", il);
        part->up_accel = ggml_new_tensor_2d(result->ctx_accel.get(), layer.ffn_up->type, layer.ffn_up->ne[0], n_ff_accel);
        ggml_set_name(part->up_accel, name);

        snprintf(name, sizeof(name), "blk.%zu.ffn_gate.hsplit_gpu", il);
        part->gate_accel = ggml_new_tensor_2d(result->ctx_accel.get(), layer.ffn_gate->type, layer.ffn_gate->ne[0], n_ff_accel);
        ggml_set_name(part->gate_accel, name);

        snprintf(name, sizeof(name), "blk.%zu.ffn_down.hsplit_gpu", il);
        part->down_accel = ggml_new_tensor_2d(result->ctx_accel.get(), layer.ffn_down->type, n_ff_accel, layer.ffn_down->ne[1]);
        ggml_set_name(part->down_accel, name);

        snprintf(name, sizeof(name), "blk.%zu.ffn_down.hsplit_cpu", il);
        part->down_cpu = ggml_new_tensor_2d(result->ctx_cpu.get(), layer.ffn_down->type, n_ff_cpu, layer.ffn_down->ne[1]);
        ggml_set_name(part->down_cpu, name);
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
    for (size_t il = 0; il < n_layer; ++il) {
        auto & part = result->partitions[il];
        if (!part) {
            continue;
        }

        const auto & layer = model.layers[il];
        const int64_t n_ff_accel = part->n_ff_accel;
        const int64_t n_ff       = layer.ffn_up->ne[1];

        // up_accel
        {
            const size_t size = n_ff_accel * layer.ffn_up->nb[1];
            std::vector<uint8_t> tmp(size);
            ggml_backend_tensor_get(layer.ffn_up, tmp.data(), 0, size);
            ggml_backend_tensor_set(part->up_accel, tmp.data(), 0, size);
        }

        // gate_accel
        {
            const size_t size = n_ff_accel * layer.ffn_gate->nb[1];
            std::vector<uint8_t> tmp(size);
            ggml_backend_tensor_get(layer.ffn_gate, tmp.data(), 0, size);
            ggml_backend_tensor_set(part->gate_accel, tmp.data(), 0, size);
        }

        // down_accel and down_cpu (row-by-row packed copy)
        {
            const size_t down_nbytes = ggml_nbytes(layer.ffn_down);
            std::vector<uint8_t> down_src(down_nbytes);
            ggml_backend_tensor_get(layer.ffn_down, down_src.data(), 0, down_nbytes);

            const int64_t blk_size  = ggml_blck_size(layer.ffn_down->type);
            const int64_t type_size = ggml_type_size(layer.ffn_down->type);

            const int64_t gpu_row_bytes = (n_ff_accel / blk_size) * type_size;
            const int64_t cpu_row_bytes = ((n_ff - n_ff_accel) / blk_size) * type_size;

            std::vector<uint8_t> down_accel_data(ggml_nbytes(part->down_accel));
            std::vector<uint8_t> down_cpu_data(ggml_nbytes(part->down_cpu));

            for (int64_t r = 0; r < layer.ffn_down->ne[1]; ++r) {
                const uint8_t * src_row = down_src.data() + r * layer.ffn_down->nb[1];
                uint8_t * dst_gpu_row   = down_accel_data.data() + r * part->down_accel->nb[1];
                uint8_t * dst_cpu_row   = down_cpu_data.data() + r * part->down_cpu->nb[1];

                memcpy(dst_gpu_row, src_row, gpu_row_bytes);
                memcpy(dst_cpu_row, src_row + gpu_row_bytes, cpu_row_bytes);
            }

            ggml_backend_tensor_set(part->down_accel, down_accel_data.data(), 0, down_accel_data.size());
            ggml_backend_tensor_set(part->down_cpu, down_cpu_data.data(), 0, down_cpu_data.size());
        }
    }

    return result;
}
