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
