#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_expert_cache_key {
    const struct ggml_tensor * tensor;
    int32_t expert_id;
};

typedef struct ggml_backend_expert_cache * ggml_backend_expert_cache_t;

GGML_API ggml_backend_expert_cache_t ggml_backend_expert_cache_new(
    ggml_backend_t backend,
    size_t capacity);

GGML_API void ggml_backend_expert_cache_free(
    ggml_backend_expert_cache_t cache);

GGML_API void ggml_backend_expert_cache_set_period(
    ggml_backend_expert_cache_t cache,
    int32_t period);

GGML_API int32_t ggml_backend_expert_cache_get_period(
    ggml_backend_expert_cache_t cache);

GGML_API void ggml_backend_expert_cache_begin_step(
    ggml_backend_expert_cache_t cache);

GGML_API void ggml_backend_expert_cache_record_access(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id);

GGML_API void ggml_backend_expert_cache_record_access_count(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id,
    uint32_t count);

GGML_API void ggml_backend_expert_cache_process_jit_swaps(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * completed_tensor,
    ggml_backend_t backend);

GGML_API struct ggml_tensor * ggml_backend_expert_cache_get_tensor(
    ggml_backend_expert_cache_t cache);

GGML_API size_t ggml_backend_expert_cache_find_offset(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id);

GGML_API void ggml_backend_expert_cache_touch(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id);

GGML_API void ggml_backend_expert_cache_record_hit(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id,
    size_t bytes_avoided);

GGML_API void ggml_backend_expert_cache_record_miss(
    ggml_backend_expert_cache_t cache,
    size_t bytes_ram_to_gpu);

GGML_API size_t ggml_backend_expert_cache_alloc_slot(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id,
    size_t size,
    const struct ggml_expert_cache_key * pinned_keys,
    size_t n_pinned);

GGML_API void ggml_backend_expert_cache_get_stats(
    ggml_backend_expert_cache_t cache,
    struct ggml_backend_expert_cache_stats * stats);

GGML_API void ggml_backend_expert_cache_print_stats(
    ggml_backend_expert_cache_t cache);

GGML_API size_t ggml_backend_expert_cache_export_entries(
    ggml_backend_expert_cache_t cache,
    struct ggml_backend_expert_cache_export_entry * out_entries,
    size_t max_entries);

GGML_API bool ggml_backend_expert_cache_seed(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t expert_id,
    uint32_t frequency);

#ifdef __cplusplus
}
#endif
