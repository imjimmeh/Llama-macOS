#include "../src/llama-model-partition.h"
#include "../src/llama-model.h"
#include "ggml.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

int main() {
    printf("running test-ffn-partition-align...\n");

    // create dummy ggml_context for tensor metadata
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    // Test 1: Standard F16 layer (block size 1)
    {
        llama_layer layer = {};
        layer.ffn_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4096, 14336);
        layer.ffn_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4096, 14336);
        layer.ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 14336, 4096);

        assert(llama_ffn_can_partition(layer, LLM_FFN_PAR) == true);
        assert(llama_ffn_can_partition(layer, LLM_FFN_SEQ) == false);

        int64_t n_ff_gpu = llama_ffn_partition_align(layer, 0.5f);
        assert(n_ff_gpu == 7168);

        n_ff_gpu = llama_ffn_partition_align(layer, 0.0f);
        assert(n_ff_gpu == 0);

        n_ff_gpu = llama_ffn_partition_align(layer, 1.0f);
        assert(n_ff_gpu == 0);
    }

    // Test 2: Q4_K layer (block size 256 for Q4_K)
    {
        llama_layer layer = {};
        layer.ffn_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 4096, 14336);
        layer.ffn_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 4096, 14336);
        layer.ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 14336, 4096);

        assert(llama_ffn_can_partition(layer, LLM_FFN_PAR) == true);

        // 14336 * 0.33 = 4730.88 -> aligned to 256 = 4608
        int64_t n_ff_gpu = llama_ffn_partition_align(layer, 0.33f);
        assert(n_ff_gpu > 0);
        assert(n_ff_gpu % 256 == 0);
        assert((14336 - n_ff_gpu) % 256 == 0);
        assert(n_ff_gpu == 4608);
    }

    // Test 3: Mixed quantization types (Q8_0 blk 32, Q4_K blk 256)
    {
        llama_layer layer = {};
        layer.ffn_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 4096, 14336);
        layer.ffn_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 4096, 14336);
        layer.ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 14336, 4096);

        assert(llama_ffn_can_partition(layer, LLM_FFN_PAR) == true);

        int64_t n_ff_gpu = llama_ffn_partition_align(layer, 0.35f);
        assert(n_ff_gpu % 256 == 0);
        assert((14336 - n_ff_gpu) % 256 == 0);
    }

    // Test 4: NVFP4 layer should be rejected
    {
        llama_layer layer = {};
        layer.ffn_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4, 4096, 14336);
        layer.ffn_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4, 4096, 14336);
        layer.ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4, 14336, 4096);

        assert(llama_ffn_can_partition(layer, LLM_FFN_PAR) == false);
    }

    // Test 5: Layer with bias should be rejected for partitioning
    {
        llama_layer layer = {};
        layer.ffn_up     = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4096, 14336);
        layer.ffn_gate   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4096, 14336);
        layer.ffn_down   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 14336, 4096);
        layer.ffn_down_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4096);

        assert(llama_ffn_can_partition(layer, LLM_FFN_PAR) == false);
    }

    ggml_free(ctx);
    printf("test-ffn-partition-align passed successfully.\n");
    return 0;
}
