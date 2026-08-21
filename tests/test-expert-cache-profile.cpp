#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-expert-cache.h"
#include "ggml-cpu.h"
#include "expert-cache-profile.h"
#include "nlohmann/json.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

using json = nlohmann::json;

static void test_path_resolution() {
    printf("testing path resolution...\n");

    std::string path1 = common_expert_cache_get_file_path("models/qwen3.5.gguf", "coding", "");
    assert(path1 == "models/qwen3.5.coding.expert_cache.json");

    std::string path2 = common_expert_cache_get_file_path("models/qwen3.5.GGUF", "prose", "");
    assert(path2 == "models/qwen3.5.prose.expert_cache.json");

    std::string path3 = common_expert_cache_get_file_path("models/qwen3.5.gguf", "default", "");
    assert(path3 == "models/qwen3.5.expert_cache.json");

    std::string path4 = common_expert_cache_get_file_path("models/qwen3.5.gguf", "", "");
    assert(path4 == "models/qwen3.5.expert_cache.json");

    std::string path5 = common_expert_cache_get_file_path("models/qwen3.5.gguf", "coding", "custom/path.json");
    assert(path5 == "custom/path.json");

    printf("  path resolution tests passed\n");
}

static void test_expert_cache_seed_and_export() {
    printf("testing expert cache seed and export...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 1024;
    const size_t cache_capacity = 4 * 1024; // room for 4 padded experts

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);

    // create a mock host tensor for experts
    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 8); // 8 experts
    ggml_set_name(tensor, "blk.0.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;
    memset(tensor->data, 0x42, ggml_nbytes(tensor));

    // Seed expert 2 with freq 150 and expert 5 with freq 80
    bool ok2 = ggml_backend_expert_cache_seed(cache, tensor, 2, 150);
    assert(ok2);

    bool ok5 = ggml_backend_expert_cache_seed(cache, tensor, 5, 80);
    assert(ok5);

    // Verify offset lookup
    size_t off2 = ggml_backend_expert_cache_find_offset(cache, tensor, 2);
    size_t off5 = ggml_backend_expert_cache_find_offset(cache, tensor, 5);
    GGML_ASSERT(off2 != SIZE_MAX);
    GGML_ASSERT(off5 != SIZE_MAX);
    GGML_ASSERT(off2 != off5);

    const int32_t slot2 = ggml_backend_expert_cache_find_slot(cache, tensor, 2);
    const int32_t slot5 = ggml_backend_expert_cache_find_slot(cache, tensor, 5);
    GGML_ASSERT(slot2 >= 0);
    GGML_ASSERT(slot5 >= 0);

    ggml_backend_expert_cache_sync(cache);
    struct ggml_tensor * slot_tensor = ggml_backend_expert_cache_get_slot_tensor(cache, tensor);
    GGML_ASSERT(slot_tensor != nullptr);
    GGML_ASSERT(memcmp((const uint8_t *)slot_tensor->data + slot2 * expert_bytes, (const uint8_t *)tensor->data + 2 * expert_bytes, expert_bytes) == 0);
    GGML_ASSERT(memcmp((const uint8_t *)slot_tensor->data + slot5 * expert_bytes, (const uint8_t *)tensor->data + 5 * expert_bytes, expert_bytes) == 0);


    // Export entries
    struct ggml_backend_expert_cache_export_entry exported[8];
    size_t n_exported = ggml_backend_expert_cache_export_entries(cache, exported, 8);
    assert(n_exported >= 2);

    bool found2 = false;
    bool found5 = false;
    for (size_t i = 0; i < n_exported; i++) {
        if (exported[i].tensor == tensor && exported[i].expert_id == 2) {
            assert(exported[i].frequency == 150);
            found2 = true;
        }
        if (exported[i].tensor == tensor && exported[i].expert_id == 5) {
            assert(exported[i].frequency == 80);
            found5 = true;
        }
    }
    assert(found2);
    assert(found5);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  expert cache seed and export tests passed\n");
}

static void test_json_profile_io() {
    printf("testing json profile format serialization...\n");

    std::string test_file = "test_profile_tmp.json";

    json j;
    j["version"] = 1;
    j["profile"] = "coding";
    j["n_entries"] = 2;
    j["updated_at"] = "2026-08-18T16:00:00Z";

    json j_experts = json::array();
    json e1;
    e1["tensor"] = "blk.0.ffn_gate_exps.weight";
    e1["expert_id"] = 7;
    e1["frequency"] = 250;
    e1["hit_count"] = 180;
    j_experts.push_back(e1);

    json e2;
    e2["tensor"] = "blk.0.ffn_down_exps.weight";
    e2["expert_id"] = 7;
    e2["frequency"] = 250;
    e2["hit_count"] = 180;
    j_experts.push_back(e2);

    j["experts"] = j_experts;

    {
        std::ofstream out(test_file);
        assert(out.is_open());
        out << j.dump(2) << "\n";
    }

    // Verify file reading
    {
        std::ifstream in(test_file);
        assert(in.is_open());
        json j_read;
        in >> j_read;
        assert(j_read["version"] == 1);
        assert(j_read["profile"] == "coding");
        assert(j_read["experts"].size() == 2);
        assert(j_read["experts"][0]["expert_id"] == 7);
    }

    std::filesystem::remove(test_file);
    printf("  json profile format tests passed\n");
}

static void test_batched_prefill_access_and_export() {
    printf("testing batched prefill access recording and export...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 512;
    const size_t cache_capacity = 4 * 1024;

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

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 8);
    ggml_set_name(tensor, "blk.0.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;

    // Simulate multi-token batch prefill access:
    // Token 0: experts 3, 5
    // Token 1: experts 3, 7
    // Token 2: experts 3, 5
    int32_t batch_tokens_experts[3][2] = {
        { 3, 5 },
        { 3, 7 },
        { 3, 5 }
    };

    for (int t = 0; t < 3; t++) {
        for (int k = 0; k < 2; k++) {
            ggml_backend_expert_cache_record_access(cache, tensor, batch_tokens_experts[t][k]);
        }
    }

    struct ggml_backend_expert_cache_export_entry exported[8];
    size_t n_exported = ggml_backend_expert_cache_export_entries(cache, exported, 8);
    assert(n_exported >= 3);

    uint32_t freq3 = 0;
    uint32_t freq5 = 0;
    uint32_t freq7 = 0;
    for (size_t i = 0; i < n_exported; i++) {
        if (exported[i].tensor == tensor) {
            if (exported[i].expert_id == 3) freq3 = exported[i].frequency;
            if (exported[i].expert_id == 5) freq5 = exported[i].frequency;
            if (exported[i].expert_id == 7) freq7 = exported[i].frequency;
        }
    }

    assert(freq3 == 3);
    assert(freq5 == 2);
    assert(freq7 == 1);

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  batched prefill access tests passed\n");
}

static void test_slot_boundary_integrity() {
    printf("testing slot boundary integrity and padding isolation...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 1024;
    const size_t cache_capacity = 4 * 1024;

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

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 4);
    ggml_set_name(tensor, "blk.0.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;

    uint8_t * raw_data = (uint8_t *)tensor->data;
    for (int e = 0; e < 4; e++) {
        uint8_t pattern = (uint8_t)(0x11 * (e + 1));
        memset(raw_data + e * expert_bytes, pattern, expert_bytes);
    }

    for (int e = 0; e < 4; e++) {
        bool ok = ggml_backend_expert_cache_seed(cache, tensor, e, 100);
        assert(ok);
    }

    struct ggml_tensor * cache_t = ggml_backend_expert_cache_get_tensor(cache);
    assert(cache_t != nullptr);
    const uint8_t * cache_buf = (const uint8_t *)cache_t->data;

    for (int e = 0; e < 4; e++) {
        size_t off = ggml_backend_expert_cache_find_offset(cache, tensor, e);
        assert(off != SIZE_MAX);
        uint8_t expected_pattern = (uint8_t)(0x11 * (e + 1));
        for (size_t b = 0; b < expert_bytes; b++) {
            if (cache_buf[off + b] != expected_pattern) {
                fprintf(stderr, "corrupted byte at expert %d offset %zu: expected 0x%02x, got 0x%02x\n",
                    e, b, expected_pattern, cache_buf[off + b]);
                assert(false);
            }
        }
    }

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  slot boundary integrity tests passed\n");
}

static void test_multicycle_rebalance_integrity() {
    printf("testing multi-cycle rebalancing and data integrity across requests...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t expert_bytes = 512;
    const size_t cache_capacity = 3 * 512; // 3 slots

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);
    ggml_backend_expert_cache_set_period(cache, 4);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 8);
    ggml_set_name(tensor, "blk.0.ffn_gate_exps.weight");
    tensor->nb[2] = expert_bytes;

    uint8_t * raw_data = (uint8_t *)tensor->data;
    for (int e = 0; e < 8; e++) {
        for (size_t b = 0; b < expert_bytes; b++) {
            raw_data[e * expert_bytes + b] = (uint8_t)((e * 31 + b) & 0xFF);
        }
    }

    struct ggml_tensor * cache_t = ggml_backend_expert_cache_get_tensor(cache);
    const uint8_t * cache_buf = (const uint8_t *)cache_t->data;

    // Simulate 40 decode steps with changing hot experts across 10 rebalances
    for (int step = 1; step <= 40; step++) {
        int hot1 = (step / 8) % 8;
        int hot2 = (step / 8 + 1) % 8;
        int hot3 = (step / 8 + 2) % 8;

        ggml_backend_expert_cache_record_access(cache, tensor, hot1);
        ggml_backend_expert_cache_record_access(cache, tensor, hot2);
        ggml_backend_expert_cache_record_access(cache, tensor, hot3);

        ggml_backend_expert_cache_begin_step(cache);

        // Verify resident cache contents have exact ground-truth bytes
        for (int e = 0; e < 8; e++) {
            size_t off = ggml_backend_expert_cache_find_offset(cache, tensor, e);
            if (off != SIZE_MAX) {
                for (size_t b = 0; b < expert_bytes; b++) {
                    uint8_t expected = (uint8_t)((e * 31 + b) & 0xFF);
                    if (cache_buf[off + b] != expected) {
                        fprintf(stderr, "step %d: corrupted byte at resident expert %d byte %zu: expected 0x%02x got 0x%02x\n",
                            step, e, b, expected, cache_buf[off + b]);
                        assert(false);
                    }
                }
            }
        }
    }

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  multi-cycle rebalance integrity tests passed\n");
}

static void test_heterogeneous_expert_sizes_rebalance() {
    printf("testing heterogeneous expert sizes and free block coalescing...\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    const size_t cache_capacity = 4 * 1024;

    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    assert(cache != nullptr);
    ggml_backend_expert_cache_set_period(cache, 2);

    size_t mem_size = 16 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    // Tensor A: 512 bytes per expert (4 experts)
    struct ggml_tensor * tensorA = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 4);
    ggml_set_name(tensorA, "blk.0.ffn_gate_exps.weight");
    tensorA->nb[2] = 512;
    uint8_t * rawA = (uint8_t *)tensorA->data;
    for (int e = 0; e < 4; e++) {
        memset(rawA + e * 512, (uint8_t)(0x10 + e), 512);
    }

    // Tensor B: 1024 bytes per expert (4 experts)
    struct ggml_tensor * tensorB = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 4);
    ggml_set_name(tensorB, "blk.0.ffn_down_exps.weight");
    tensorB->nb[2] = 1024;
    uint8_t * rawB = (uint8_t *)tensorB->data;
    for (int e = 0; e < 4; e++) {
        memset(rawB + e * 1024, (uint8_t)(0xA0 + e), 1024);
    }

    struct ggml_tensor * cache_t = ggml_backend_expert_cache_get_tensor(cache);
    const uint8_t * cache_buf = (const uint8_t *)cache_t->data;

    // Access Tensor A experts and Tensor B experts across alternating steps
    for (int step = 1; step <= 12; step++) {
        if (step % 2 == 1) {
            ggml_backend_expert_cache_record_access_count(cache, tensorA, 0, 10);
            ggml_backend_expert_cache_record_access_count(cache, tensorA, 1, 10);
            ggml_backend_expert_cache_record_access_count(cache, tensorB, 0, 5);
        } else {
            ggml_backend_expert_cache_record_access_count(cache, tensorB, 1, 20);
            ggml_backend_expert_cache_record_access_count(cache, tensorB, 2, 20);
            ggml_backend_expert_cache_record_access_count(cache, tensorA, 2, 5);
        }
        ggml_backend_expert_cache_begin_step(cache);

        // Verify validity of all resident entries
        for (int e = 0; e < 4; e++) {
            size_t offA = ggml_backend_expert_cache_find_offset(cache, tensorA, e);
            if (offA != SIZE_MAX) {
                uint8_t exp = (uint8_t)(0x10 + e);
                for (size_t b = 0; b < 512; b++) {
                    assert(cache_buf[offA + b] == exp);
                }
            }
            size_t offB = ggml_backend_expert_cache_find_offset(cache, tensorB, e);
            if (offB != SIZE_MAX) {
                uint8_t exp = (uint8_t)(0xA0 + e);
                for (size_t b = 0; b < 1024; b++) {
                    assert(cache_buf[offB + b] == exp);
                }
            }
        }
    }

    ggml_backend_expert_cache_free(cache);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  heterogeneous expert sizes tests passed\n");
}
static void test_profile_sort_order() {
    printf("testing profile sort order (descending frequency)...\n");

    std::vector<common_expert_cache_profile_entry> entries = {
        { "blk.0.ffn_gate_exps.weight", 1, 5,  0 },
        { "blk.0.ffn_gate_exps.weight", 2, 90, 0 },
        { "blk.0.ffn_gate_exps.weight", 1, 40, 0 }, // duplicate: keep max freq
        { "blk.1.ffn_gate_exps.weight", 3, 20, 0 },
    };

    common_expert_cache_sort_entries(entries);

    assert(entries.size() == 3);
    assert(entries[0].expert_id == 2 && entries[0].frequency == 90); // hottest first
    assert(entries[1].expert_id == 1 && entries[1].frequency == 40); // merged duplicate
    assert(entries[2].expert_id == 3 && entries[2].frequency == 20);

    printf("  profile sort order tests passed\n");
}


int main() {
    printf("running test-expert-cache-profile...\n");

    test_path_resolution();
    test_expert_cache_seed_and_export();
    test_json_profile_io();
    test_profile_sort_order();

    test_batched_prefill_access_and_export();
    test_slot_boundary_integrity();
    test_multicycle_rebalance_integrity();
    test_heterogeneous_expert_sizes_rebalance();

    printf("all test-expert-cache-profile tests passed successfully!\n");
    return 0;
}

