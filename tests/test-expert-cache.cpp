#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-expert-cache.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void test_slot_pools_and_remapping() {
    printf("testing slot pools and zero-copy ID remapping...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 1024;
    const size_t cache_capacity = 8 * 1024; // room for 8 experts

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 16); // 16 experts
    ggml_set_name(tensor, "blk.0.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;

    // Verify slot tensor retrieval
    struct ggml_tensor * slot_t = ggml_backend_expert_cache_get_slot_tensor(cache, tensor);
    assert(slot_t != nullptr);
    assert(slot_t->ne[0] == tensor->ne[0]);
    assert(slot_t->ne[1] == tensor->ne[1]);
    assert(slot_t->nb[2] == tensor->nb[2]);

    // Seed/allocate slots for experts 3, 7, 11
    int32_t s3 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 3, nullptr, 0);
    int32_t s7 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 7, nullptr, 0);
    int32_t s11 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 11, nullptr, 0);

    assert(s3 >= 0);
    assert(s7 >= 0);
    assert(s11 >= 0);
    assert(s3 != s7 && s7 != s11 && s3 != s11);

    // Test find_slot
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 3) == s3);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 7) == s7);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 11) == s11);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 4) == -1); // not cached

    // Test ID remapping
    int32_t req_ids[4] = { 3, 4, 7, 11 };
    int32_t remapped_ids[4] = { -1, -1, -1, -1 };
    bool is_hit[4] = { false, false, false, false };

    int32_t n_hits = ggml_backend_expert_cache_remap_ids(cache, tensor, req_ids, 4, remapped_ids, is_hit);
    assert(n_hits == 3);
    assert(is_hit[0] == true && remapped_ids[0] == s3);
    assert(is_hit[1] == false && remapped_ids[1] == 4);
    assert(is_hit[2] == true && remapped_ids[2] == s7);
    assert(is_hit[3] == true && remapped_ids[3] == s11);

    // Record zero-copy hit telemetry
    ggml_backend_expert_cache_record_zero_copy_hit(cache, tensor, 3, expert_bytes);
    struct ggml_backend_expert_cache_stats stats;
    ggml_backend_expert_cache_get_stats(cache, &stats);
    assert(stats.n_zero_copy_hits == 1);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  slot pools and zero-copy ID remapping tests passed\n");
}

static void test_slru_and_admission_policy() {
    printf("testing SLRU segments and admission policy...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 512;
    const size_t cache_capacity = 4 * 512; // 4 slots total (1 probationary, 3 protected)

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 8);
    ggml_set_name(tensor, "blk.1.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;

    // Allocate 4 slots
    int32_t s0 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 0, nullptr, 0);
    int32_t s1 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 1, nullptr, 0);
    int32_t s2 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 2, nullptr, 0);
    int32_t s3 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 3, nullptr, 0);
    assert(s0 >= 0 && s1 >= 0 && s2 >= 0 && s3 >= 0);

    // Touch expert 0 and 1 multiple times to promote to protected
    ggml_backend_expert_cache_touch(cache, tensor, 0);
    ggml_backend_expert_cache_touch(cache, tensor, 0);
    ggml_backend_expert_cache_touch(cache, tensor, 1);
    ggml_backend_expert_cache_touch(cache, tensor, 1);

    // Allocate a new expert 4 when full -> should evict least recently used from probationary (e.g. 2 or 3)
    int32_t s4 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 4, nullptr, 0);
    assert(s4 >= 0);

    // Verify hot protected experts 0 and 1 are still resident
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 0) >= 0);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 1) >= 0);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  SLRU segments and admission policy tests passed\n");
}

static void test_expert_bundles() {
    printf("testing expert bundle registration and atomic residency...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 512;
    const size_t cache_capacity = 8 * 512;

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 4);
    ggml_set_name(gate, "blk.2.ffn_gate_exps.weight");
    gate->nb[2] = expert_bytes;

    struct ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 4);
    ggml_set_name(down, "blk.2.ffn_down_exps.weight");
    down->nb[2] = expert_bytes;

    // Register bundle for layer 2
    ggml_backend_expert_cache_register_bundle(cache, 2, gate, nullptr, down);

    // Initially expert 1 is not resident
    assert(!ggml_backend_expert_cache_is_bundle_resident(cache, 2, 1));

    // Allocate gate only -> bundle should still not be fully resident
    ggml_backend_expert_cache_alloc_slot_idx(cache, gate, 1, nullptr, 0);
    assert(!ggml_backend_expert_cache_is_bundle_resident(cache, 2, 1));

    // Allocate down as well -> now bundle is resident
    ggml_backend_expert_cache_alloc_slot_idx(cache, down, 1, nullptr, 0);
    assert(ggml_backend_expert_cache_is_bundle_resident(cache, 2, 1));

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  expert bundle tests passed\n");
}

static void test_pinned_host_buffer() {
    printf("testing pinned host buffer allocation and alignment...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, 4096);
    assert(cache != nullptr);

    void * buf = ggml_backend_expert_cache_get_pinned_buffer(cache, 2048);
    assert(buf != nullptr);
    assert(((uintptr_t)buf % 512) == 0); // 512-byte aligned

    // Reallocate larger
    void * buf2 = ggml_backend_expert_cache_get_pinned_buffer(cache, 8192);
    assert(buf2 != nullptr);
    assert(((uintptr_t)buf2 % 512) == 0);

    ggml_backend_expert_cache_free(cache);
    ggml_backend_free(backend);

    printf("  pinned host buffer tests passed\n");
}

static void test_transition_predictor_and_prefetch() {
    printf("testing transition predictor and speculative prefetch...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 512;
    const size_t cache_capacity = 8 * 512;

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 8);
    ggml_set_name(tensor, "blk.3.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;
    memset(tensor->data, 0xAB, ggml_nbytes(tensor));

    // Simulate transition pattern: {1, 2} frequently transitions to {4, 5}
    int32_t step1_experts[2] = { 1, 2 };
    int32_t step2_experts[2] = { 4, 5 };

    for (int rep = 0; rep < 5; rep++) {
        ggml_backend_expert_cache_record_step_experts(cache, 3, step1_experts, 2);
        ggml_backend_expert_cache_record_step_experts(cache, 3, step2_experts, 2);
    }

    // Predict next from {1, 2}
    int32_t predicted[4] = { -1, -1, -1, -1 };
    int32_t n_pred = ggml_backend_expert_cache_predict_next(cache, 3, step1_experts, 2, predicted, 4);
    assert(n_pred >= 2);
    bool has_4 = (predicted[0] == 4 || predicted[1] == 4);
    bool has_5 = (predicted[0] == 5 || predicted[1] == 5);
    assert(has_4 && has_5);

    // Test speculative prefetch
    ggml_backend_expert_cache_prefetch(cache, tensor, predicted, n_pred);
    struct ggml_backend_expert_cache_stats stats;
    ggml_backend_expert_cache_get_stats(cache, &stats);
    assert(stats.n_speculative_prefetches >= 2);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 4) >= 0);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 5) >= 0);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  transition predictor and prefetch tests passed\n");
}

static void test_core_anchor_pinning() {
    printf("testing core anchor pinning...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 1024;
    const size_t cache_capacity = 4 * expert_bytes; // room for only 4 experts

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 16);
    ggml_set_name(tensor, "blk.0.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;

    // Pin expert 2 as a permanent core anchor
    bool ok_pin = ggml_backend_expert_cache_pin_anchor(cache, tensor, 2);
    assert(ok_pin);
    int32_t slot2 = ggml_backend_expert_cache_find_slot(cache, tensor, 2);
    assert(slot2 >= 0);

    // Fill remaining slots with experts 4, 5, 6
    int32_t slot4 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 4, nullptr, 0);
    int32_t slot5 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 5, nullptr, 0);
    int32_t slot6 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 6, nullptr, 0);
    assert(slot4 >= 0 && slot5 >= 0 && slot6 >= 0);

    // Cause heavy evictions by allocating experts 7, 8, 9, 10
    for (int e = 7; e <= 10; e++) {
        int32_t se = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, e, nullptr, 0);
        assert(se >= 0);
    }

    // Anchor expert 2 MUST STILL BE RESIDENT!
    int32_t check_slot2 = ggml_backend_expert_cache_find_slot(cache, tensor, 2);
    assert(check_slot2 == slot2);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  core anchor pinning tests passed\n");
}

static void test_prompt_tail_warmup() {
    printf("testing prompt-tail warmup seeding...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 1024;
    const size_t cache_capacity = 8 * expert_bytes;

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 16);
    ggml_set_name(tensor, "blk.0.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;

    // Simulate 128 tokens with 2 experts each, where tail tokens heavily use experts 9 and 13
    std::vector<int32_t> ids_data(128 * 2, 1);
    for (int t = 64; t < 128; t++) {
        ids_data[t * 2 + 0] = 9;
        ids_data[t * 2 + 1] = 13;
    }

    ggml_backend_expert_cache_record_prompt_tail(cache, tensor, ids_data.data(), 2, 128, 64);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 9) < 0);

    // Warm up from prompt tail
    ggml_backend_expert_cache_warmup_from_prompt_tail(cache, backend);

    // Experts 9 and 13 should now be resident in slots
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 9) >= 0);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 13) >= 0);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  prompt-tail warmup seeding tests passed\n");
}

int main() {
    printf("running test-expert-cache (V3 features)...\n");

    test_slot_pools_and_remapping();
    test_slru_and_admission_policy();
    test_expert_bundles();
    test_pinned_host_buffer();
    test_transition_predictor_and_prefetch();
    test_core_anchor_pinning();
    test_prompt_tail_warmup();

    printf("all test-expert-cache tests passed successfully!\n");
    return 0;
}
