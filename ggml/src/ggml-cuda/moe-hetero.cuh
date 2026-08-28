#pragma once

#include "ggml.h"
#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// GPU Scatter Kernel for Partial-Hit Heterogeneous MoE Output (TG1)
void ggml_cuda_moe_scatter_tg1(
    uint8_t * dst_base,
    const float * gpu_hits,
    const float * cpu_misses_uploaded,
    const int32_t * hit_route_indices,
    const int32_t * miss_route_indices,
    int n_hits,
    int n_misses,
    int d_model,
    size_t route_stride_bytes,
    cudaStream_t stream);

// Legacy float stride wrapper
void ggml_cuda_moe_scatter_routes(
    float * dst,
    const float * gpu_hits,
    const float * cpu_misses_uploaded,
    const int32_t * hit_route_indices,
    const int32_t * miss_route_indices,
    int n_hits,
    int n_misses,
    int d_model,
    int64_t stride_route_floats,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

