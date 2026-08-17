#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void randomize_tensor(struct ggml_tensor * tensor) {
    float * p = (float *) tensor->data;
    const size_t ne = ggml_nelements(tensor);
    for (size_t i = 0; i < ne; ++i) {
        p[i] = ((float) rand() / (float) RAND_MAX) * 2.0f - 1.0f;
    }
}

int main() {
    printf("running test-ffn-split (numerical equivalence)...\n");

    const int64_t n_embd = 64;
    const int64_t n_ff   = 128;
    const int64_t n_tok  = 4;
    const int64_t n_gpu  = 48; // split point
    const int64_t n_cpu  = n_ff - n_gpu;

    // allocate host memory for test
    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    // input activations
    ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tok);
    randomize_tensor(cur);

    // full weights
    ggml_tensor * up   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);
    ggml_tensor * gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);
    ggml_tensor * down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff, n_embd);
    randomize_tensor(up);
    randomize_tensor(gate);
    randomize_tensor(down);

    // 1. Unpartitioned FFN graph
    // tmp = up * cur
    ggml_tensor * u = ggml_mul_mat(ctx, up, cur);
    // g = silu(gate * cur)
    ggml_tensor * g = ggml_mul_mat(ctx, gate, cur);
    ggml_tensor * act = ggml_silu(ctx, g);
    // par = u * act
    ggml_tensor * ffn_mid = ggml_mul(ctx, act, u);
    // out = down * ffn_mid
    ggml_tensor * out_ref = ggml_mul_mat(ctx, down, ffn_mid);

    // 2. Partitioned FFN graph
    // GPU partition
    ggml_tensor * up_gpu   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_gpu);
    ggml_tensor * gate_gpu = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_gpu);
    ggml_tensor * down_gpu = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_gpu, n_embd);

    // Copy GPU slices
    memcpy(up_gpu->data, up->data, n_gpu * up->nb[1]);
    memcpy(gate_gpu->data, gate->data, n_gpu * gate->nb[1]);
    for (int64_t r = 0; r < n_embd; ++r) {
        const float * src_r = (const float *) ((const uint8_t *) down->data + r * down->nb[1]);
        float * dst_r       = (float *) ((uint8_t *) down_gpu->data + r * down_gpu->nb[1]);
        memcpy(dst_r, src_r, n_gpu * sizeof(float));
    }

    ggml_tensor * u_gpu = ggml_mul_mat(ctx, up_gpu, cur);
    ggml_tensor * g_gpu = ggml_mul_mat(ctx, gate_gpu, cur);
    ggml_tensor * act_gpu = ggml_silu(ctx, g_gpu);
    ggml_tensor * ffn_mid_gpu = ggml_mul(ctx, act_gpu, u_gpu);
    ggml_tensor * out_gpu = ggml_mul_mat(ctx, down_gpu, ffn_mid_gpu);

    // CPU partition
    ggml_tensor * up_cpu   = ggml_view_2d(ctx, up, n_embd, n_cpu, up->nb[1], n_gpu * up->nb[1]);
    ggml_tensor * gate_cpu = ggml_view_2d(ctx, gate, n_embd, n_cpu, gate->nb[1], n_gpu * gate->nb[1]);
    ggml_tensor * down_cpu = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cpu, n_embd);

    // Copy CPU slice for packed down
    for (int64_t r = 0; r < n_embd; ++r) {
        const float * src_r = (const float *) ((const uint8_t *) down->data + r * down->nb[1]);
        float * dst_r       = (float *) ((uint8_t *) down_cpu->data + r * down_cpu->nb[1]);
        memcpy(dst_r, src_r + n_gpu, n_cpu * sizeof(float));
    }

    ggml_tensor * u_cpu = ggml_mul_mat(ctx, up_cpu, cur);
    ggml_tensor * g_cpu = ggml_mul_mat(ctx, gate_cpu, cur);
    ggml_tensor * act_cpu = ggml_silu(ctx, g_cpu);
    ggml_tensor * ffn_mid_cpu = ggml_mul(ctx, act_cpu, u_cpu);
    ggml_tensor * out_cpu = ggml_mul_mat(ctx, down_cpu, ffn_mid_cpu);

    // Sum
    ggml_tensor * out_split = ggml_add(ctx, out_gpu, out_cpu);

    // Compute both
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_ref);
    ggml_build_forward_expand(gf, out_split);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_status status = ggml_backend_graph_compute(backend, gf);
    assert(status == GGML_STATUS_SUCCESS);
    ggml_backend_free(backend);

    // Verify numerical equivalence
    const float * ref_data   = (const float *) out_ref->data;
    const float * split_data = (const float *) out_split->data;
    const size_t n_out = ggml_nelements(out_ref);

    float max_diff = 0.0f;
    for (size_t i = 0; i < n_out; ++i) {
        float diff = std::abs(ref_data[i] - split_data[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    printf("max absolute diff between unpartitioned and partitioned FFN: %e\n", max_diff);
    assert(max_diff < 1e-4f);

    ggml_free(ctx);
    printf("test-ffn-split passed successfully.\n");
    return 0;
}
