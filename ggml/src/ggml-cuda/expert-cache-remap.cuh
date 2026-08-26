#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_expert_cache_miss_desc {
    int32_t token_idx;
    int32_t route_idx;
    int32_t expert_id;
};

void * ggml_cuda_expert_cache_gpu_map_create(int32_t n_expert);
void   ggml_cuda_expert_cache_gpu_map_update(void * dev_map, const int32_t * host_entries, int32_t n_entries, void * stream);
void   ggml_cuda_expert_cache_gpu_map_free(void * dev_map);

void ggml_cuda_expert_cache_partition_ids(
    const int32_t * ids_in,
    int32_t * remapped_ids_out,
    const int32_t * gpu_slot_map,
    int32_t * hit_mask_out,
    struct ggml_expert_cache_miss_desc * miss_descs_out,
    int32_t * miss_counter_dev,
    int32_t n_expert_used,
    int32_t n_tokens,
    void * stream);

#ifdef __cplusplus
}
#endif
