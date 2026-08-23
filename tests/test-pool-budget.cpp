// Standalone unit test for the global slot-pool byte budget added by
// Task 6 of docs/superpowers/plans/2026-08-23-expert-cache-zero-copy-fixes.md.
//
// The cache is configured with capacity 8 KiB. A small tensor fits a single
// pool well within the budget; a second, much larger tensor must be refused
// (NULL) instead of silently aliasing the flat cache buffer as the legacy
// fallback used to.

#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-expert-cache.h"
#include "ggml-cpu.h"

#include <assert.h>
#include <stdio.h>

static void test_global_pool_budget() {
    printf("testing global pool byte budget enforcement...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t cache_capacity = 8 * 1024;
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    size_t mem_size = 512 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    // Small tensor: stride fits well under cache_capacity.
    struct ggml_tensor * small = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 16);
    ggml_set_name(small, "blk.0.ffn_gate_exps.weight");

    // Huge tensor: stride alone dwarfs the configured cache capacity so any
    // single pool for this shape would exceed capacity.
    struct ggml_tensor * huge = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1024, 1024, 64);
    ggml_set_name(huge, "blk.1.ffn_down_exps.weight");

    struct ggml_tensor * slot_small = ggml_backend_expert_cache_get_slot_tensor(cache, small);
    assert(slot_small != nullptr);

    // Same small tensor resolves to the same pool - confirms the small pool
    // is allocated and capacity is the small shape's footprint.
    struct ggml_tensor * slot_small_again = ggml_backend_expert_cache_get_slot_tensor(cache, small);
    assert(slot_small_again == slot_small);

    // Budget exhausted: a second distinct-shape pool must be refused (NULL)
    // instead of aliasing the flat cache buffer.
    struct ggml_tensor * slot_huge = ggml_backend_expert_cache_get_slot_tensor(cache, huge);
    assert(slot_huge == nullptr);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  global pool byte budget test passed\n");
}

int main() {
    printf("running test-pool-budget...\n");
    test_global_pool_budget();
    printf("all test-pool-budget tests passed successfully!\n");
    return 0;
}
