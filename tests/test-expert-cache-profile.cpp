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
    assert(off2 != SIZE_MAX);
    assert(off5 != SIZE_MAX);
    assert(off2 != off5);

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

int main() {
    printf("running test-expert-cache-profile...\n");

    test_path_resolution();
    test_expert_cache_seed_and_export();
    test_json_profile_io();

    printf("all test-expert-cache-profile tests passed successfully!\n");
    return 0;
}
