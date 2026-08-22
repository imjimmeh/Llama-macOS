// Pre-resident single-layer MoE latency oracle.
//
// Answers: if the needed experts were already resident in GPU memory,
// would executing this MoE layer on the GPU beat executing it on the CPU?
//
// Scenarios, all with qwen35moe A3B shapes and Q4_K weights:
//   1. cpu_full : MUL_MAT_ID over the host-resident 256-expert tensor (today's TG path)
//   2. gpu_full : same op, tensor pre-copied to VRAM once, zero H2D inside timing loop
//   3. gpu_slot : MUL_MAT_ID over an 8-expert slot tensor in VRAM with remapped ids
//                 (the shape of the expert-cache zero-copy fast path)
//
// Transfer cost is deliberately excluded: weights are staged before timing.
// Run: test-moe-latency-oracle [iters]

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct moe_dims {
    int64_t n_embd   = 2048;
    int64_t n_ff     = 512;
    int64_t n_expert = 256;
    int64_t n_used   = 8;
    int64_t n_tok    = 1;
};

struct bench_result {
    double avg_us;
    double min_us;
};

uint64_t rng_state = 42;

float frand() {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((rng_state >> 33) & 0xFFFFFF) / float(0x1000000) * 2.0f - 1.0f;
}


// Builds one MoE FFN core: up/gate/down projections through MUL_MAT_ID.
// n_resident controls the expert dimension of the weight tensor (slot vs full).
ggml_cgraph * build_layer(ggml_context * ctx, const moe_dims & d, int64_t n_resident) {
    ggml_tensor * w_up   = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, d.n_embd, d.n_ff, n_resident);
    ggml_tensor * w_gate = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, d.n_embd, d.n_ff, n_resident);
    ggml_tensor * w_down = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, d.n_ff, d.n_embd, n_resident);

    ggml_tensor * cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d.n_embd, 1, d.n_tok);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, d.n_used, d.n_tok);

    ggml_tensor * up   = ggml_mul_mat_id(ctx, w_up,   cur, ids);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, w_gate, cur, ids);
    ggml_tensor * act  = ggml_silu(ctx, gate);
    ggml_tensor * mid  = ggml_mul(ctx, act, up);
    ggml_tensor * out = ggml_mul_mat_id(ctx, w_down, mid, ids);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    return gf;
}

bench_result bench(ggml_backend_t backend, ggml_cgraph * gf, int iters) {
    // one untimed warmup call per timed call set below
    ggml_backend_graph_compute(backend, gf);

    double total_us = 0;
    double min_us = 1e30;
    for (int i = 0; i < iters; ++i) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_graph_compute(backend, gf);
        const int64_t t1 = ggml_time_us();
        const double us = double(t1 - t0);
        total_us += us;
        if (us < min_us) {
            min_us = us;
        }
    }
    return { total_us / iters, min_us };
}
void report(const char * name, bench_result r) {
    printf("%-10s avg %8.1f us   min %8.1f us\n", name, r.avg_us, r.min_us);
}

// fills weights with quantized random data, activations with random floats, ids with valid indices
void populate(ggml_backend_t backend, ggml_context * ctx, const moe_dims & d, int64_t n_resident) {
    // order matches creation order in build_layer
    ggml_tensor * w_up   = ggml_get_first_tensor(ctx);
    ggml_tensor * w_gate = ggml_get_next_tensor(ctx, w_up);
    ggml_tensor * w_down = ggml_get_next_tensor(ctx, w_gate);
    ggml_tensor * cur    = ggml_get_next_tensor(ctx, w_down);
    ggml_tensor * ids    = ggml_get_next_tensor(ctx, cur);
    std::vector<float> fsrc(d.n_embd);
    for (ggml_tensor * w : { w_up, w_gate, w_down }) {
        const int64_t nrows = ggml_nrows(w);
        const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_K, w->ne[0]);
        std::vector<uint8_t> qdst(row_bytes);
        size_t off = 0;
        for (int64_t r = 0; r < nrows; ++r) {
            for (int64_t i = 0; i < w->ne[0]; ++i) {
                fsrc[i] = frand();
            }
            ggml_quantize_chunk(GGML_TYPE_Q4_K, fsrc.data(), qdst.data(), 0, 1, w->ne[0], nullptr);
            ggml_backend_tensor_set(w, qdst.data(), off, row_bytes);
            off += row_bytes;
        }
    }

    {
        std::vector<float> tmp(ggml_nelements(cur));
        for (auto & v : tmp) {
            v = frand();
        }
        ggml_backend_tensor_set(cur, tmp.data(), 0, ggml_nbytes(cur));
    }

    {
        std::vector<int32_t> tmp(d.n_used * d.n_tok);
        for (int64_t t = 0; t < d.n_tok; ++t) {
            for (int64_t k = 0; k < d.n_used; ++k) {
                tmp[t * d.n_used + k] = int32_t((t * d.n_used + k) % n_resident);
            }
        }
        ggml_backend_tensor_set(ids, tmp.data(), 0, ggml_nbytes(ids));
    }
}

} // namespace

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 100;
    moe_dims d;

    printf("moe latency oracle: n_embd=%lld n_ff=%lld n_expert=%lld n_used=%lld n_tok=%lld iters=%d\n",
           (long long) d.n_embd, (long long) d.n_ff, (long long) d.n_expert,
           (long long) d.n_used, (long long) d.n_tok, iters);

    ggml_time_init();

    // scenario 1: CPU over the full host-resident expert tensor
    {
        ggml_init_params ip = { 64u << 20, nullptr, true };
        ggml_context_ptr ctx(ggml_init(ip));
        ggml_cgraph * gf = build_layer(ctx.get(), d, d.n_expert);
        ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get());
        populate(backend.get(), ctx.get(), d, d.n_expert);
        report("cpu_full", bench(backend.get(), gf, iters));
        ggml_backend_buffer_free(buf);
    }

    // scenario 2: GPU over the full device-resident expert tensor
    // scenario 3: GPU over an 8-expert device-resident slot tensor (cache fast path)
    ggml_backend_ptr gpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr));
    if (!gpu.get()) {
        printf("no GPU backend available; gpu_full/gpu_slot skipped\n");
        return 0;
    }
    {
        ggml_init_params ip = { 64u << 20, nullptr, true };
        ggml_context_ptr ctx(ggml_init(ip));
        ggml_cgraph * gf = build_layer(ctx.get(), d, d.n_expert);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx.get(), gpu.get());
        populate(gpu.get(), ctx.get(), d, d.n_expert);
        report("gpu_full", bench(gpu.get(), gf, iters));
        ggml_backend_buffer_free(buf);
    }
    {
        ggml_init_params ip = { 16u << 20, nullptr, true };
        ggml_context_ptr ctx(ggml_init(ip));
        ggml_cgraph * gf = build_layer(ctx.get(), d, d.n_used);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx.get(), gpu.get());
        populate(gpu.get(), ctx.get(), d, d.n_used);
        report("gpu_slot", bench(gpu.get(), gf, iters));
        ggml_backend_buffer_free(buf);
    }

    return 0;
}
