#include "common.cuh"
#include "moe-hetero.cuh"

// Warp-level scatter kernel merging GPU hits and uploaded CPU misses into canonical route layout (TG1)
__global__ void moe_hetero_scatter_tg1_kernel(
        uint8_t * __restrict__ dst_base,
        const float * __restrict__ gpu_hits,
        const float * __restrict__ cpu_misses,
        const int32_t * __restrict__ hit_route_indices,
        const int32_t * __restrict__ miss_route_indices,
        const int n_hits,
        const int n_misses,
        const int d_model,
        const size_t route_stride_bytes) {

    const int item_idx = blockIdx.x;
    if (item_idx >= n_hits + n_misses) {
        return;
    }

    const bool is_gpu_hit = (item_idx < n_hits);
    const int local_idx = is_gpu_hit ? item_idx : (item_idx - n_hits);
    const int route_idx = is_gpu_hit ? hit_route_indices[local_idx] : miss_route_indices[local_idx];
    const float * src = is_gpu_hit ? (gpu_hits + (size_t)local_idx * d_model) : (cpu_misses + (size_t)local_idx * d_model);
    float * dst_route = reinterpret_cast<float *>(dst_base + (size_t)route_idx * route_stride_bytes);

    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        dst_route[i] = src[i];
    }
}

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
        cudaStream_t stream) {

    const int total_items = n_hits + n_misses;
    if (total_items <= 0 || dst_base == nullptr) {
        return;
    }

    const int block_size = 256;
    const int grid_size = total_items;

    moe_hetero_scatter_tg1_kernel<<<grid_size, block_size, 0, stream>>>(
        dst_base,
        gpu_hits,
        cpu_misses_uploaded,
        hit_route_indices,
        miss_route_indices,
        n_hits,
        n_misses,
        d_model,
        route_stride_bytes);
}

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
        cudaStream_t stream) {

    ggml_cuda_moe_scatter_tg1(
        (uint8_t *)dst,
        gpu_hits,
        cpu_misses_uploaded,
        hit_route_indices,
        miss_route_indices,
        n_hits,
        n_misses,
        d_model,
        (size_t)stride_route_floats * sizeof(float),
        stream);
}

