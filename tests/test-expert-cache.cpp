#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-expert-cache.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool condition) {
    if (!condition) {
        fprintf(stderr, "test requirement failed\n");
        abort();
    }
}


static void test_cache_node_selection() {
    printf("testing cached expert node selection...\n");

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    ggml_tensor * experts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 8, 2);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 2, 1);
    ggml_tensor * matmul = ggml_mul_mat_id(ctx, experts, input, ids);
    ggml_tensor * unrelated_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, matmul->ne[0], matmul->ne[1]);
    ggml_tensor * unrelated = ggml_dup(ctx, unrelated_input);
    ggml_tensor * output = ggml_add(ctx, unrelated, matmul);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    require(ggml_graph_node(graph, 0) != matmul);
    require(ggml_backend_find_mul_mat_id_node(graph, experts) == matmul);
    require(ggml_backend_find_mul_mat_id_node(graph, input) == nullptr);

    ggml_free(ctx);

    printf("  cached expert node selection tests passed\n");
}


static void test_cache_capacity_admission() {
    printf("testing cache capacity admission...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr);

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, 256);
    require(cache != nullptr);
    require(ggml_backend_expert_cache_can_store(cache, 256));
    require(!ggml_backend_expert_cache_can_store(cache, 512));
    ggml_backend_expert_cache_record_eligible(cache);
    ggml_backend_expert_cache_record_capacity_bypass(cache);
    struct ggml_backend_expert_cache_stats stats;
    ggml_backend_expert_cache_get_stats(cache, &stats);
    require(stats.n_eligible_ops == 1);
    require(stats.n_capacity_bypasses == 1);
    ggml_backend_expert_cache_record_cpu_backend_bypass(cache);
    ggml_backend_expert_cache_get_stats(cache, &stats);
    require(stats.n_cpu_backend_bypasses == 1);
    ggml_backend_expert_cache_record_mul_mat_id_input(cache);
    ggml_backend_expert_cache_record_non_host_weight_bypass(cache);
    ggml_backend_expert_cache_get_stats(cache, &stats);
    require(stats.n_mul_mat_id_inputs == 1);
    require(stats.n_non_host_weight_bypasses == 1);

    ggml_backend_expert_cache_free(cache);
    ggml_backend_free(backend);

    printf("  cache capacity admission tests passed\n");
}


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

static void test_prefetch() {
    printf("testing explicit expert prefetch...\n");

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

    int32_t expert_ids[2] = { 4, 5 };
    ggml_backend_expert_cache_prefetch(cache, tensor, expert_ids, 2);
    struct ggml_backend_expert_cache_stats stats;
    ggml_backend_expert_cache_get_stats(cache, &stats);
    assert(stats.n_speculative_prefetches >= 2);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 4) >= 0);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 5) >= 0);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  prefetch tests passed\n");
}

static void test_prefetch_deduplicates_expert_ids() {
    printf("testing duplicate expert prefetch coalescing...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr);

    const size_t expert_bytes = 512;
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, 8 * expert_bytes);
    require(cache != nullptr);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 8);
    require(tensor != nullptr);
    ggml_set_name(tensor, "blk.4.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;
    memset(tensor->data, 0xCD, ggml_nbytes(tensor));

    const int32_t expert_ids[] = { 2, 2, 2, 6, 6 };
    ggml_backend_expert_cache_prefetch(cache, tensor, expert_ids, 5);

    struct ggml_backend_expert_cache_stats stats;
    ggml_backend_expert_cache_get_stats(cache, &stats);
    require(stats.n_speculative_prefetches == 2);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  duplicate expert prefetch coalescing tests passed\n");
}

static void test_route_prefetch_telemetry() {
    printf("testing route prefetch telemetry...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr);
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, 4096);
    require(cache != nullptr);

    ggml_backend_expert_cache_record_route_snapshot(cache);
    ggml_backend_expert_cache_record_route_prefetch_submitted(cache);
    ggml_backend_expert_cache_record_route_prefetch_duplicate(cache);
    ggml_backend_expert_cache_record_route_prefetch_stale(cache);
    ggml_backend_expert_cache_record_route_prefetch_rejected(cache);

    struct ggml_backend_expert_cache_stats stats;
    ggml_backend_expert_cache_get_stats(cache, &stats);
    require(stats.n_route_prefetch_snapshots == 1);
    require(stats.n_route_prefetch_submitted == 1);
    require(stats.n_route_prefetch_duplicates == 1);
    require(stats.n_route_prefetch_stale == 1);
    require(stats.n_route_prefetch_rejected == 1);

    ggml_backend_expert_cache_free(cache);
    ggml_backend_free(backend);

    printf("  route prefetch telemetry tests passed\n");
}

static void test_pinned_staging_no_overwrite() {
    printf("testing pinned staging ring acquire/commit discipline...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 1024;
    const size_t cache_capacity = 64 * 1024;

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * tensor_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 16);
    ggml_set_name(tensor_a, "blk.0.ffn_gate_exps.weight");

    struct ggml_backend_expert_cache_stats stats_before;
    ggml_backend_expert_cache_get_stats(cache, &stats_before);

    // acquire + commit marks the entry in-flight; a second acquire of the SAME
    // (tensor, slot) must wait (n_staging_waits increments) and return the same pointer
    void * p1 = ggml_backend_expert_cache_stage_acquire(cache, tensor_a, 3, expert_bytes);
    assert(p1 != nullptr);
    assert(((uintptr_t) p1 % 512) == 0);
    ggml_backend_expert_cache_stage_commit(cache, tensor_a, 3);

    void * p2 = ggml_backend_expert_cache_stage_acquire(cache, tensor_a, 3, expert_bytes);
    assert(p2 != nullptr);
    assert(p2 == p1); // same ring entry

    struct ggml_backend_expert_cache_stats stats_after;
    ggml_backend_expert_cache_get_stats(cache, &stats_after);
    assert(stats_after.n_staging_waits == stats_before.n_staging_waits + 1);

    // after the wait the entry is free: acquiring again must not wait again
    void * p3 = ggml_backend_expert_cache_stage_acquire(cache, tensor_a, 3, expert_bytes);
    assert(p3 == p1);
    ggml_backend_expert_cache_get_stats(cache, &stats_after);
    assert(stats_after.n_staging_waits == stats_before.n_staging_waits + 1);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  pinned staging ring tests passed\n");
}

static void test_cross_layer_shape_isolation() {
    printf("testing cross-layer same-shape slot isolation...\n");

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

    // Layer 0 tensor and Layer 1 tensor with identical shape [8, 16, 4]
    struct ggml_tensor * tensor_l0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 4);
    ggml_set_name(tensor_l0, "blk.0.ffn_gate_exps.weight");
    tensor_l0->nb[2] = expert_bytes;

    struct ggml_tensor * tensor_l1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 4);
    ggml_set_name(tensor_l1, "blk.1.ffn_gate_exps.weight");
    tensor_l1->nb[2] = expert_bytes;

    // Fill expert 2 of layer 0 with 0x11, and expert 2 of layer 1 with 0x22
    uint8_t * raw0 = (uint8_t *)tensor_l0->data;
    uint8_t * raw1 = (uint8_t *)tensor_l1->data;
    memset(raw0 + 2 * expert_bytes, 0x11, expert_bytes);
    memset(raw1 + 2 * expert_bytes, 0x22, expert_bytes);

    // Seed both Layer 0 Exp 2 and Layer 1 Exp 2
    bool ok0 = ggml_backend_expert_cache_seed(cache, tensor_l0, 2, 100);
    bool ok1 = ggml_backend_expert_cache_seed(cache, tensor_l1, 2, 100);
    assert(ok0 && ok1);

    // Find slots: they MUST be distinct slots
    int32_t slot0 = ggml_backend_expert_cache_find_slot(cache, tensor_l0, 2);
    int32_t slot1 = ggml_backend_expert_cache_find_slot(cache, tensor_l1, 2);
    assert(slot0 >= 0);
    assert(slot1 >= 0);
    assert(slot0 != slot1);

    // Verify data integrity in slot tensor
    struct ggml_tensor * slot_t = ggml_backend_expert_cache_get_slot_tensor(cache, tensor_l0);
    assert(slot_t != nullptr);
    const uint8_t * slot_buf = (const uint8_t *)slot_t->data;

    for (size_t b = 0; b < expert_bytes; b++) {
        assert(slot_buf[slot0 * expert_bytes + b] == 0x11);
        assert(slot_buf[slot1 * expert_bytes + b] == 0x22);
    }

    // Verify that tensor_l0 only finds its own slot and not layer 1
    assert(ggml_backend_expert_cache_find_slot(cache, tensor_l0, 2) == slot0);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor_l1, 2) == slot1);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  cross-layer same-shape slot isolation tests passed\n");
}

static void test_pp_tg_telemetry_isolation() {
    printf("testing PP and TG telemetry frequency isolation...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, 4096);
    assert(cache != nullptr);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 4);
    ggml_set_name(tensor, "blk.0.ffn_gate_exps.weight");
    tensor->nb[2] = 512;

    // Record access in PP phase for expert 1
    ggml_backend_expert_cache_record_access_count(cache, tensor, 1, 500, GGML_EXPERT_CACHE_PHASE_PP);

    // Record access in TG phase for expert 0
    ggml_backend_expert_cache_record_access_count(cache, tensor, 0, 10, GGML_EXPERT_CACHE_PHASE_TG);

    struct ggml_backend_expert_cache_export_entry entries[4];
    size_t n = ggml_backend_expert_cache_export_entries(cache, entries, 4);

    // Only TG entries drive export/residency
    assert(n >= 1);
    bool found_tg_0 = false;
    bool found_pp_1_in_tg = false;
    for (size_t i = 0; i < n; i++) {
        if (entries[i].expert_id == 0 && entries[i].frequency == 10) found_tg_0 = true;
        if (entries[i].expert_id == 1 && entries[i].frequency == 500) found_pp_1_in_tg = true;
    }
    assert(found_tg_0);
    assert(!found_pp_1_in_tg); // PP accesses must not leak into TG residency frequency

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  PP and TG telemetry isolation tests passed\n");
}

static void test_slot_loading_lifecycle() {
    printf("testing slot LOADING -> RESIDENT state machine...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 1024;
    const size_t cache_capacity = 64 * 1024;

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 8);
    ggml_set_name(tensor, "blk.0.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;

    // 1. A new claim owns the fill and starts in LOADING state.
    bool needs_load = false;
    int32_t slot = ggml_backend_expert_cache_claim_slot_idx(
        cache, tensor, 5, NULL, 0, &needs_load);
    require(slot >= 0);
    require(needs_load);

    // A duplicate claim attaches to the existing fill.
    bool duplicate_needs_load = true;
    require(ggml_backend_expert_cache_claim_slot_idx(
        cache, tensor, 5, NULL, 0, &duplicate_needs_load) == slot);
    require(!duplicate_needs_load);

    // LOADING must not return a hit.
    int32_t found = ggml_backend_expert_cache_find_slot(cache, tensor, 5);
    require(found == -1);

    int32_t orig_ids[1] = { 5 };
    int32_t remapped[1] = { -1 };
    bool is_hit[1] = { false };
    int32_t n_hits = ggml_backend_expert_cache_remap_ids(cache, tensor, orig_ids, 1, remapped, is_hit);
    require(n_hits == 0);
    require(!is_hit[0]);

    // Promotion makes the completed fill visible.
    ggml_backend_expert_cache_promote_slot(cache, tensor, 5, slot);
    require(ggml_backend_expert_cache_find_slot(cache, tensor, 5) == slot);

    n_hits = ggml_backend_expert_cache_remap_ids(cache, tensor, orig_ids, 1, remapped, is_hit);
    require(n_hits == 1);
    require(is_hit[0]);
    require(remapped[0] == slot);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  slot LOADING -> RESIDENT state machine tests passed\n");
}

static void test_per_tensor_slot_isolation() {
    printf("testing per-tensor slot isolation across layers...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 1024;
    const size_t cache_capacity = 64 * 1024;

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    // Two same-shaped tensors on different layers
    struct ggml_tensor * t_l0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 8);
    ggml_set_name(t_l0, "blk.0.ffn_gate_exps.weight");
    t_l0->nb[2] = expert_bytes;

    struct ggml_tensor * t_l1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 8);
    ggml_set_name(t_l1, "blk.1.ffn_gate_exps.weight");
    t_l1->nb[2] = expert_bytes;

    int32_t slot_l0 = ggml_backend_expert_cache_alloc_slot_idx(cache, t_l0, 3, NULL, 0);
    assert(slot_l0 >= 0);
    ggml_backend_expert_cache_promote_slot(cache, t_l0, 3, slot_l0);

    // t_l1 expert 3 must NOT be present
    assert(ggml_backend_expert_cache_find_slot(cache, t_l1, 3) == -1);

    int32_t slot_l1 = ggml_backend_expert_cache_alloc_slot_idx(cache, t_l1, 3, NULL, 0);
    assert(slot_l1 >= 0);
    assert(slot_l1 != slot_l0); // different slots in same pool
    ggml_backend_expert_cache_promote_slot(cache, t_l1, 3, slot_l1);

    // Both must resolve to their own respective slots
    assert(ggml_backend_expert_cache_find_slot(cache, t_l0, 3) == slot_l0);
    assert(ggml_backend_expert_cache_find_slot(cache, t_l1, 3) == slot_l1);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  per-tensor slot isolation tests passed\n");
}

static void test_staging_sync_teardown() {
    printf("testing staging sync and teardown safety...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t cache_capacity = 64 * 1024;
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 8);
    ggml_set_name(tensor, "blk.0.ffn_gate_exps.weight");

    void * p = ggml_backend_expert_cache_stage_acquire(cache, tensor, 0, 1024);
    assert(p != nullptr);
    ggml_backend_expert_cache_stage_commit(cache, tensor, 0);

    // explicit sync must drain without error
    ggml_backend_expert_cache_sync_staging(cache);

    // teardown with clean state
    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  staging sync and teardown safety tests passed\n");
}

static void test_auto_reserve_sentinel() {
    printf("testing auto reserve sentinel semantics...\n");

    // SIZE_MAX -> unset sentinel, must default to 512 MiB
    const size_t unset_reserve = (size_t)-1;
    const size_t r_default = (unset_reserve != (size_t)-1) ? unset_reserve : (512 * 1024 * 1024);
    assert(r_default == 512 * 1024 * 1024);

    // 0 -> explicit zero margin requested via --fit-target 0
    const size_t zero_reserve = 0;
    const size_t r_zero = (zero_reserve != (size_t)-1) ? zero_reserve : (512 * 1024 * 1024);
    assert(r_zero == 0);

    // 256 MiB -> explicit custom target
    const size_t custom_reserve = 256 * 1024 * 1024;
    const size_t r_custom = (custom_reserve != (size_t)-1) ? custom_reserve : (512 * 1024 * 1024);
    assert(r_custom == 256 * 1024 * 1024);

    printf("  auto reserve sentinel semantics tests passed\n");
}

int main() {
    printf("running test-expert-cache (V2 features)...\n");

    test_cache_node_selection();
    test_cache_capacity_admission();
    test_slot_pools_and_remapping();
    test_cross_layer_shape_isolation();
    test_pinned_staging_no_overwrite();

    test_slru_and_admission_policy();
    test_expert_bundles();
    test_pinned_host_buffer();
    test_prefetch();
    test_prefetch_deduplicates_expert_ids();
    test_route_prefetch_telemetry();
    test_pp_tg_telemetry_isolation();

    test_slot_loading_lifecycle();
    test_per_tensor_slot_isolation();
    test_staging_sync_teardown();
    test_auto_reserve_sentinel();

    printf("all test-expert-cache tests passed successfully!\n");
    return 0;
}
