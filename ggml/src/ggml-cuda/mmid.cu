#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    int nex_prev = 0; // Number of columns for experts with a lower index.

    if constexpr (n_expert_used_template == 0) {
        // Pass 1: compute nex_prev
        for (int it = 0; it < n_tokens; ++it) {
            for (int iex = 0; iex < n_expert_used; ++iex) {
                const int expert_used = ids[it*si1 + iex];
                if (threadIdx.x == 0) {
                    nex_prev += expert_used < expert;
                }
            }
        }

        if (threadIdx.x == 0) {
            expert_bounds[expert] = nex_prev;
        }

        // Pass 2: write compact indices directly to global memory
        int it_compact = 0;
        for (int it = 0; it < n_tokens; ++it) {
            for (int iex = 0; iex < n_expert_used; ++iex) {
                const int expert_used = ids[it*si1 + iex];
                if (threadIdx.x == 0) {
                    if (expert_used == expert) {
                        const int dst_idx = nex_prev + it_compact;
                        ids_dst[dst_idx] = it*n_expert_used + iex;
                        if (write_inverse) {
                            ids_src1[it*n_expert_used + iex] = dst_idx;
                        } else {
                            ids_src1[dst_idx] = it*sis1 + iex % nchannels_y;
                        }
                        it_compact++;
                    }
                }
            }
        }

        if (threadIdx.x == 0 && expert == static_cast<int>(gridDim.x) - 1) {
            expert_bounds[gridDim.x] = nex_prev + it_compact;
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, "bad n_expert_used");
        const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used; // Padded to next higher power of 2.

        // Pass 1: count columns for experts with a lower index
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;
            const int iex = threadIdx.x % neu_padded;
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            nex_prev += expert_used < expert;
        }
        nex_prev = warp_reduce_sum<warp_size>(nex_prev);

        if (threadIdx.x == 0) {
            expert_bounds[expert] = nex_prev;
        }

        // Pass 2: write directly to global memory using warp prefix scan
        int it_compact = 0;
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;
            const int iex = threadIdx.x % neu_padded;
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;

            const int it_compact_add_self = (iex_used != -1) ? 1 : 0;

            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = 1; offset < warp_size; offset *= 2) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self + it_compact_add_lower, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                const int dst_idx = nex_prev + it_compact + it_compact_add_lower;
                ids_dst[dst_idx] = it*n_expert_used + iex_used;
                if (write_inverse) {
                    ids_src1[it*n_expert_used + iex_used] = dst_idx;
                } else {
                    ids_src1[dst_idx] = it*sis1 + iex_used % nchannels_y;
                }
            }

            const int warp_matches = __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
            it_compact += warp_matches;
        }

        if (threadIdx.x == 0 && expert == static_cast<int>(gridDim.x) - 1) {
            expert_bounds[gridDim.x] = nex_prev + it_compact;
        }
    }
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, 0, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}
