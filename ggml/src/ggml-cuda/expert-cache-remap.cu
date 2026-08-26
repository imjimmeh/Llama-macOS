#include "expert-cache-remap.cuh"
#include "common.cuh"

__global__ void expert_cache_partition_ids_cuda_kernel(
        const int32_t * __restrict__ ids_in,
        int32_t * __restrict__ remapped_ids_out,
        const int32_t * __restrict__ gpu_slot_map,
        int32_t * __restrict__ hit_mask_out,
        struct ggml_expert_cache_miss_desc * __restrict__ miss_descs_out,
        int32_t * __restrict__ miss_counter,
        const int32_t n_expert_used,
        const int32_t n_tokens) {
    const int32_t token = blockIdx.y;
    const int32_t route = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= n_tokens || route >= n_expert_used) return;

    const int32_t idx = token * n_expert_used + route;
    const int32_t expert_id = ids_in[idx];

    if (expert_id < 0) {
        remapped_ids_out[idx] = -1;
        if (hit_mask_out) hit_mask_out[idx] = 0;
        return;
    }

    const int32_t slot = gpu_slot_map ? gpu_slot_map[expert_id] : -1;
    remapped_ids_out[idx] = slot;

    if (slot >= 0) {
        if (hit_mask_out) hit_mask_out[idx] = 1;
    } else {
        if (hit_mask_out) hit_mask_out[idx] = 0;
        if (miss_counter && miss_descs_out) {
            const int32_t miss_pos = atomicAdd(miss_counter, 1);
            miss_descs_out[miss_pos].token_idx = token;
            miss_descs_out[miss_pos].route_idx = route;
            miss_descs_out[miss_pos].expert_id = expert_id;
        }
    }
}

void * ggml_cuda_expert_cache_gpu_map_create(int32_t n_expert) {
    if (n_expert <= 0) return nullptr;
    void * dev_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_ptr, (size_t)n_expert * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(dev_ptr, 0xFF, (size_t)n_expert * sizeof(int32_t))); // init with -1
    return dev_ptr;
}

void ggml_cuda_expert_cache_gpu_map_update(void * dev_map, const int32_t * host_entries, int32_t n_entries, void * stream) {
    if (!dev_map || !host_entries || n_entries <= 0) return;
    cudaStream_t s = stream ? (cudaStream_t)stream : cudaStreamPerThread;
    CUDA_CHECK(cudaMemcpyAsync(dev_map, host_entries, (size_t)n_entries * sizeof(int32_t), cudaMemcpyHostToDevice, s));
}

void ggml_cuda_expert_cache_gpu_map_free(void * dev_map) {
    if (!dev_map) return;
    CUDA_CHECK(cudaFree(dev_map));
}

void ggml_cuda_expert_cache_partition_ids(
        const int32_t * ids_in,
        int32_t * remapped_ids_out,
        const int32_t * gpu_slot_map,
        int32_t * hit_mask_out,
        struct ggml_expert_cache_miss_desc * miss_descs_out,
        int32_t * miss_counter_dev,
        int32_t n_expert_used,
        int32_t n_tokens,
        void * stream) {
    if (n_expert_used <= 0 || n_tokens <= 0) return;

    cudaStream_t s = stream ? (cudaStream_t)stream : cudaStreamPerThread;

    const int block_size = 64;
    dim3 grid((n_expert_used + block_size - 1) / block_size, n_tokens);
    dim3 block(block_size);

    expert_cache_partition_ids_cuda_kernel<<<grid, block, 0, s>>>(
        ids_in,
        remapped_ids_out,
        gpu_slot_map,
        hit_mask_out,
        miss_descs_out,
        miss_counter_dev,
        n_expert_used,
        n_tokens
    );
    CUDA_CHECK(cudaGetLastError());
}
