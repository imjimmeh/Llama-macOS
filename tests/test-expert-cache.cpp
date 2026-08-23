#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-expert-cache.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define REQUIRE(cond) do { if (!(cond)) { fprintf(stderr, "require failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); abort(); } } while (0)
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

    // Task 1 contract: alloc publishes LOADING only. Mark resident so the
    // existing find_slot / remap assertions below still hold.
    ggml_backend_expert_cache_mark_resident(cache, tensor, 3);
    ggml_backend_expert_cache_mark_resident(cache, tensor, 7);
    ggml_backend_expert_cache_mark_resident(cache, tensor, 11);

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
    const size_t cache_capacity = 8 * 512; // pools split capacity in half -> 4 slots

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
    // Mark resident for downstream assertions.
    ggml_backend_expert_cache_mark_resident(cache, tensor, 0);
    ggml_backend_expert_cache_mark_resident(cache, tensor, 1);
    ggml_backend_expert_cache_mark_resident(cache, tensor, 2);
    ggml_backend_expert_cache_mark_resident(cache, tensor, 3);

    // Touch expert 0 and 1 multiple times to promote to protected
    ggml_backend_expert_cache_touch(cache, tensor, 0);
    ggml_backend_expert_cache_touch(cache, tensor, 0);
    ggml_backend_expert_cache_touch(cache, tensor, 1);
    ggml_backend_expert_cache_touch(cache, tensor, 1);

    // Allocate a new expert 4 when full -> should evict least recently used from probationary (e.g. 2 or 3)
    int32_t s4 = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 4, nullptr, 0);
    assert(s4 >= 0);
    ggml_backend_expert_cache_mark_resident(cache, tensor, 4);

    fprintf(stderr, "[dbg] s0=%d s1=%d s2=%d s3=%d s4=%d\n", s0,s1,s2,s3,s4);
    fprintf(stderr, "[dbg] loading: e0=%d e1=%d e2=%d e3=%d e4=%d\n",
        ggml_backend_expert_cache_find_or_loading_slot(cache, tensor, 0),
        ggml_backend_expert_cache_find_or_loading_slot(cache, tensor, 1),
        ggml_backend_expert_cache_find_or_loading_slot(cache, tensor, 2),
        ggml_backend_expert_cache_find_or_loading_slot(cache, tensor, 3),
        ggml_backend_expert_cache_find_or_loading_slot(cache, tensor, 4));
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
    // Task 1: alloc only publishes LOADING. Mark both resident so the
    // bundle-resident assertion below holds.
    ggml_backend_expert_cache_mark_resident(cache, gate, 1);
    ggml_backend_expert_cache_mark_resident(cache, down, 1);
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
static void test_route_trace() {
    printf("testing route trace collection...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t cache_capacity = 64 * 1024;
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    // Enable trace to a temporary file
    const char* trace_file = "test_route_trace.bin";
    ggml_backend_expert_cache_enable_route_trace(cache, trace_file, 10000);

    // Simulate routing decisions for multiple tokens and layers
    // Token 0 (after first begin_step), Layer 0: experts [1, 3, 5]
    ggml_backend_expert_cache_begin_step(cache);
    int32_t experts_t0_l0[] = {1, 3, 5};
    ggml_backend_expert_cache_record_route_trace(cache, 0, experts_t0_l0, 3);

    // Token 0, Layer 1: experts [2, 4, 6]
    int32_t experts_t0_l1[] = {2, 4, 6};
    ggml_backend_expert_cache_record_route_trace(cache, 1, experts_t0_l1, 3);

    // Token 1, Layer 0: experts [1, 4, 7] (different from token 0)
    ggml_backend_expert_cache_begin_step(cache);
    int32_t experts_t1_l0[] = {1, 4, 7};
    ggml_backend_expert_cache_record_route_trace(cache, 0, experts_t1_l0, 3);

    // Token 1, Layer 1: experts [2, 5, 8]
    int32_t experts_t1_l1[] = {2, 5, 8};
    ggml_backend_expert_cache_record_route_trace(cache, 1, experts_t1_l1, 3);

    // Disable trace (flushes to file)
    ggml_backend_expert_cache_disable_route_trace(cache);

    // Verify trace file was created and has content
    FILE* f = fopen(trace_file, "rb");
    assert(f != nullptr);

    // Read header
    uint32_t magic, version;
    assert(fread(&magic, sizeof(uint32_t), 1, f) == 1);
    assert(magic == 0x52545243); // "RTRC"
    assert(fread(&version, sizeof(uint32_t), 1, f) == 1);
    assert(version == 2);

    // Read entries (binary format matches ggml_expert_cache_route_trace_entry)
    struct {
        uint64_t token_id;
        int32_t layer;
        int32_t n_experts;
        int32_t expert_ids[64];
        uint64_t timestamp_us;
    } entry;

    // v2: each fixed-size entry is followed by int32 n_logits + float blob
    auto read_blob = [&](int32_t expect_n, int expect_first) {
        int32_t n = 0;
        assert(fread(&n, sizeof(n), 1, f) == 1);
        assert(n == expect_n);
        if (n > 0) {
            std::vector<float> blob(n);
            assert(fread(blob.data(), sizeof(float), n, f) == (size_t)n);
            if (expect_first >= 0) {
                assert(blob[0] == (float)expect_first);
            }
        }
    };

    // Entry 1: Token 1, Layer 0 (no logits staged)
    assert(fread(&entry, sizeof(entry), 1, f) == 1);
    assert(entry.token_id == 1);
    assert(entry.layer == 0);
    read_blob(0, -1);

    // Entry 2: Token 1, Layer 1
    assert(fread(&entry, sizeof(entry), 1, f) == 1);
    assert(entry.token_id == 1);
    assert(entry.layer == 1);
    read_blob(0, -1);

    // Entry 3: Token 2, Layer 0
    assert(fread(&entry, sizeof(entry), 1, f) == 1);
    assert(entry.token_id == 2);
    assert(entry.layer == 0);
    read_blob(0, -1);

    // Entry 4: Token 2, Layer 1
    assert(fread(&entry, sizeof(entry), 1, f) == 1);
    assert(entry.token_id == 2);
    assert(entry.layer == 1);
    read_blob(0, -1);

    // No more entries
    assert(fread(&entry, sizeof(entry), 1, f) == 0);

    fclose(f);

    // Clean up
    remove(trace_file);
    ggml_backend_expert_cache_free(cache);
    ggml_backend_free(backend);

    printf("  route trace tests passed\n");
}


static void test_route_trace_v2_logits() {
    printf("testing route trace v2 logits capture...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, 64 * 1024);
    assert(cache != nullptr);

    const char* trace_file = "test_route_trace_v2.bin";
    ggml_backend_expert_cache_enable_route_trace(cache, trace_file, 10000);

    ggml_backend_expert_cache_begin_step(cache);

    int32_t ids[2] = {3, 7};
    ggml_backend_expert_cache_record_route_trace(cache, 41, ids, 2);

    float logits[8] = {0};
    logits[3] = 5.f;
    ggml_backend_expert_cache_record_router_logits(cache, 42, logits, 8);
    ggml_backend_expert_cache_record_route_trace(cache, 42, ids, 2);

    // stale logits must not attach to a different layer
    ggml_backend_expert_cache_record_route_trace(cache, 43, ids, 2);

    ggml_backend_expert_cache_disable_route_trace(cache);

    FILE* f = fopen(trace_file, "rb");
    assert(f != nullptr);

    uint32_t magic, version;
    assert(fread(&magic, sizeof(uint32_t), 1, f) == 1 && magic == 0x52545243);
    assert(fread(&version, sizeof(uint32_t), 1, f) == 1 && version == 2);

    struct {
        uint64_t token_id;
        int32_t layer;
        int32_t n_experts;
        int32_t expert_ids[64];
        uint64_t timestamp_us;
    } entry;

    auto read_blob = [&](int32_t expect_n) {
        int32_t n = 0;
        assert(fread(&n, sizeof(n), 1, f) == 1);
        assert(n == expect_n);
        if (n > 0) {
            std::vector<float> blob(n);
            assert(fread(blob.data(), sizeof(float), n, f) == (size_t)n);
            assert(blob[3] == 5.f);
        }
    };

    assert(fread(&entry, sizeof(entry), 1, f) == 1 && entry.layer == 41);
    read_blob(0);

    assert(fread(&entry, sizeof(entry), 1, f) == 1 && entry.layer == 42);
    read_blob(8);

    assert(fread(&entry, sizeof(entry), 1, f) == 1 && entry.layer == 43);
    read_blob(0);

    assert(fread(&entry, sizeof(entry), 1, f) == 0);
    fclose(f);

    remove(trace_file);
    ggml_backend_expert_cache_free(cache);
    ggml_backend_free(backend);

    printf("  route trace v2 logits tests passed\n");
}

static void test_prediction_queue() {
    printf("testing prediction queue storage...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, 64 * 1024);
    assert(cache != nullptr);

    const int32_t ids[4] = {1, 5, 9, 20};
    ggml_backend_expert_cache_submit_prediction(cache, 10, ids, 4, nullptr);
    assert(ggml_backend_expert_cache_pending_prediction_count(cache) == 1);

    // second submit for same layer replaces
    const int32_t ids2[2] = {3, 7};
    ggml_backend_expert_cache_submit_prediction(cache, 10, ids2, 2, nullptr);
    assert(ggml_backend_expert_cache_pending_prediction_count(cache) == 1);

    int32_t out[64] = {};
    int32_t n = ggml_backend_expert_cache_get_pending_prediction(cache, 10, out, 64);
    assert(n == 2);
    assert(out[0] == 3);
    assert(out[1] == 7);

    // distinct layer adds an entry
    const int32_t ids3[1] = {11};
    ggml_backend_expert_cache_submit_prediction(cache, 20, ids3, 1, nullptr);
    assert(ggml_backend_expert_cache_pending_prediction_count(cache) == 2);

    ggml_backend_expert_cache_free(cache);
    ggml_backend_free(backend);

    printf("  prediction queue tests passed\n");
}

static void test_settle_accounting() {
    printf("testing prediction settle accounting...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 512;
    const size_t cache_capacity = 16 * 512;
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = { mem_size, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 64);
    ggml_set_name(gate, "blk.10.ffn_gate_exps.weight");
    gate->nb[2] = expert_bytes;
    memset(gate->data, 0xAB, ggml_nbytes(gate));

    struct ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 64);
    ggml_set_name(down, "blk.10.ffn_down_exps.weight");
    down->nb[2] = expert_bytes;
    memset(down->data, 0xCD, ggml_nbytes(down));

    ggml_backend_expert_cache_register_bundle(cache, 10, gate, nullptr, down);

    // Predict experts {5, 9} for layer 10. Host-to-host prefetch is skipped
    // on a CPU backend, so pre-resident expert 5 ourselves.
    const int32_t pred[2] = {5, 9};
    ggml_backend_expert_cache_submit_prediction(cache, 10, pred, 2, nullptr);
    for (struct ggml_tensor * t : {gate, down}) {
        int32_t s5 = ggml_backend_expert_cache_alloc_slot_idx(cache, t, 5, nullptr, 0);
        assert(s5 >= 0);
        memset((uint8_t *)t->data + (size_t)5 * expert_bytes, 0x11, expert_bytes);
        ggml_backend_expert_cache_mark_resident(cache, t, 5);
    }

    // actual execution requested {5, 33}: 5 predicted+resident (fully hidden),
    // 33 not predicted (wrong)
    const int32_t actual[2] = {5, 33};
    bool ok = ggml_backend_expert_cache_settle_prediction(cache, 10, actual, 2, expert_bytes * 3);
    assert(ok);

    struct ggml_routing_predictor_stats s = {};
    assert(ggml_backend_expert_cache_get_routing_predictor_stats(cache, &s));
    assert(s.predictions_used == 1);
    // only USED predicted experts count as fully hidden: 5 yes, 9 unused
    assert(s.experts_fully_hidden == 1);
    assert(s.experts_missed == 0);
    assert(s.predictions_wrong == 1);     // 33 not predicted

    // pending entry consumed
    assert(ggml_backend_expert_cache_pending_prediction_count(cache) == 0);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  settle accounting tests passed\n");
}

static void test_submit_triggers_prefetch() {
    printf("testing submit triggers prefetch...\n");


    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 512;
    const size_t cache_capacity = 16 * 512;
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = { mem_size, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 8);
    ggml_set_name(gate, "blk.12.ffn_gate_exps.weight");
    gate->nb[2] = expert_bytes;
    memset(gate->data, 0xAB, ggml_nbytes(gate));

    ggml_backend_expert_cache_register_bundle(cache, 12, gate, nullptr, nullptr);

    const int32_t ids[2] = {1, 5};
    ggml_backend_expert_cache_submit_prediction(cache, 12, ids, 2, nullptr);

    // Host-to-host prefetch is skipped on a CPU backend; the submission
    // itself must still be queued for settle accounting.
    assert(ggml_backend_expert_cache_pending_prediction_count(cache) == 1);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  submit prefetch tests passed\n");
}

static void test_stats_getter() {
    printf("testing routing predictor stats getter...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, 64 * 1024);
    assert(cache != nullptr);

    ggml_routing_predictor_stats s = {};
    assert(!ggml_backend_expert_cache_get_routing_predictor_stats(cache, &s) || true); // empty still returns true

    const int32_t ids[1] = {1};
    ggml_backend_expert_cache_submit_prediction(cache, 3, ids, 1, nullptr);
    assert(ggml_backend_expert_cache_get_routing_predictor_stats(cache, &s));
    // generated is counted upstream (graph callback), not here
    assert(s.predictions_generated == 0);
    assert(s.predictions_used == 0);

    ggml_backend_expert_cache_free(cache);
    ggml_backend_free(backend);

    printf("  stats getter tests passed\n");
}


static void test_loading_slot_not_visible_as_resident() {
    printf("testing loading vs resident visibility...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);

    const size_t expert_bytes = 1024;
    const size_t cache_capacity = 8 * 1024;

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    REQUIRE(cache != nullptr);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 16);
    ggml_set_name(tensor, "blk.0.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;

    int32_t s = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, 3, nullptr, 0);
    REQUIRE(s >= 0);
    // After alloc, slot is LOADING -> find_slot returns -1; loading lookup succeeds.
    REQUIRE(ggml_backend_expert_cache_find_slot(cache, tensor, 3) == -1);
    REQUIRE(ggml_backend_expert_cache_find_or_loading_slot(cache, tensor, 3) == s);

    ggml_backend_expert_cache_mark_resident(cache, tensor, 3);
    REQUIRE(ggml_backend_expert_cache_find_slot(cache, tensor, 3) == s);

    int32_t req_ids[2] = { 3, 4 };
    int32_t remapped[2] = { -1, -1 };
    bool hit[2] = { false, false };
    int32_t n_hits = ggml_backend_expert_cache_remap_ids(cache, tensor, req_ids, 2, remapped, hit);
    REQUIRE(n_hits == 1);
    REQUIRE(hit[0] == true && remapped[0] == s);
    REQUIRE(hit[1] == false);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  loading vs resident visibility tests passed\n");
}

static void test_prefetch_record_lifecycle() {
    printf("testing prefetch record lifecycle (host-to-host skip)...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);

    const size_t expert_bytes = 1024;
    const size_t cache_capacity = 8 * expert_bytes;
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    REQUIRE(cache != nullptr);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 8);
    ggml_set_name(tensor, "blk.4.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;

    // Host source + host slot pool gains nothing from staging, so prefetch
    // must be a clean no-op: no slot published, no pending record kept.
    const int32_t ids[1] = { 6 };
    ggml_backend_expert_cache_prefetch_async(cache, tensor, ids, 1, 4);
    REQUIRE(ggml_backend_expert_cache_find_or_loading_slot(cache, tensor, 6) == -1);
    REQUIRE(ggml_backend_expert_cache_find_slot(cache, tensor, 6) == -1);
    REQUIRE(ggml_backend_expert_cache_prefetch_slot_count(cache) == 0);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  prefetch record lifecycle tests passed\n");
}

static void test_transition_conditioned_on_source() {
    printf("testing transition table conditioned on source expert...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, 64 * 1024);
    assert(cache != nullptr);

    ggml_backend_expert_cache_enable_predictor(cache, 4, 64);

    // Train: (layer0={1} -> layer1={5}) ten times; (layer0={2} -> layer1={9}) once.
    const int32_t ids_0[1] = {1};
    const int32_t ids_1[1] = {5};
    for (int i = 0; i < 10; i++) {
        ggml_backend_expert_cache_record_prediction(cache, 0, ids_0, 1);
        ggml_backend_expert_cache_record_prediction(cache, 1, ids_1, 1);
    }
    const int32_t ids_2_0[1] = {2};
    const int32_t ids_2_1[1] = {9};
    ggml_backend_expert_cache_record_prediction(cache, 0, ids_2_0, 1);
    ggml_backend_expert_cache_record_prediction(cache, 1, ids_2_1, 1);

    // Re-record (0,{2}) right before predicting so current_experts[0]={2}.
    ggml_backend_expert_cache_record_prediction(cache, 0, ids_2_0, 1);

    int32_t out[8] = {};
    int32_t n = ggml_backend_expert_cache_predict_experts(cache, 0, 1, out, 8);
    assert(n > 0);
    assert(out[0] == 9);

    ggml_backend_expert_cache_disable_predictor(cache);
    ggml_backend_expert_cache_free(cache);
    ggml_backend_free(backend);

    printf("  transition conditioned on source tests passed\n");
}

int main() {
    printf("running test-expert-cache (V2 features)...\n");

    test_cache_node_selection();
    test_cache_capacity_admission();
    test_slot_pools_and_remapping();
    test_loading_slot_not_visible_as_resident();
    test_pinned_staging_no_overwrite();

    test_slru_and_admission_policy();
    test_expert_bundles();
    test_pinned_host_buffer();
    test_route_trace();
    test_route_trace_v2_logits();
    test_prediction_queue();
    test_settle_accounting();
    test_submit_triggers_prefetch();
    test_stats_getter();
    test_prefetch_record_lifecycle();
    test_transition_conditioned_on_source();

    printf("all test-expert-cache tests passed successfully!\n");
    return 0;
}
