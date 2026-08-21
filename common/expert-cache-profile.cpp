#include "expert-cache-profile.h"
#include "log.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <sstream>

using json = nlohmann::json;

std::string common_expert_cache_get_file_path(
        const std::string & model_path,
        const std::string & profile_name,
        const std::string & explicit_file_path) {
    if (!explicit_file_path.empty()) {
        return explicit_file_path;
    }

    std::string base = model_path.empty() ? "model" : model_path;

    // strip trailing .gguf if present
    const std::string gguf_ext = ".gguf";
    if (base.size() >= gguf_ext.size()) {
        std::string suffix = base.substr(base.size() - gguf_ext.size());
        for (char & c : suffix) c = (char)tolower((unsigned char)c);
        if (suffix == gguf_ext) {
            base = base.substr(0, base.size() - gguf_ext.size());
        }
    }

    if (!profile_name.empty() && profile_name != "default") {
        return base + "." + profile_name + ".expert_cache.json";
    }

    return base + ".expert_cache.json";
}

size_t common_expert_cache_load_profile(
        struct llama_context * ctx,
        const struct llama_model * model,
        const std::string & file_path) {
    if (ctx == nullptr || model == nullptr || file_path.empty()) {
        return 0;
    }

    std::error_code ec;
    if (!std::filesystem::exists(file_path, ec)) {
        return 0;
    }

    std::ifstream file(file_path);
    if (!file.is_open()) {
        LOG_WRN("%s: unable to open expert cache file '%s'\n", __func__, file_path.c_str());
        return 0;
    }

    json j;
    try {
        file >> j;
    } catch (const std::exception & err) {
        LOG_WRN("%s: failed to parse JSON in '%s': %s\n", __func__, file_path.c_str(), err.what());
        return 0;
    }

    if (!j.contains("experts") || !j["experts"].is_array()) {
        LOG_WRN("%s: invalid format in '%s': missing 'experts' array\n", __func__, file_path.c_str());
        return 0;
    }

    ggml_backend_sched_t sched = llama_context_get_sched(ctx);
    if (sched == nullptr) {
        return 0;
    }

    std::string profile = j.value("profile", "default");
    const auto & experts_json = j["experts"];

    struct expert_cache_seed_entry {
        const struct ggml_tensor * tensor;
        int32_t                    expert_id;
        uint32_t                   frequency;
    };

    std::vector<expert_cache_seed_entry> entries;
    entries.reserve(experts_json.size());

    size_t skipped_count = 0;
    for (const auto & item : experts_json) {
        if (!item.contains("tensor") || !item.contains("expert_id")) {
            skipped_count++;
            continue;
        }

        const std::string tensor_name = item["tensor"].get<std::string>();
        const int32_t expert_id = item["expert_id"].get<int32_t>();
        const uint32_t frequency = item.value("frequency", (uint32_t) 1);

        const struct ggml_tensor * tensor = llama_model_get_tensor(model, tensor_name.c_str());
        if (tensor == nullptr || tensor->ne[2] <= 1 || expert_id < 0 || expert_id >= tensor->ne[2]) {
            skipped_count++;
            continue;
        }

        entries.push_back({ tensor, expert_id, frequency });
    }

    std::sort(entries.begin(), entries.end(), [](const expert_cache_seed_entry & a, const expert_cache_seed_entry & b) {
        if (a.tensor != b.tensor) {
            return std::less<const struct ggml_tensor *>()(a.tensor, b.tensor);
        }
        return a.expert_id < b.expert_id;
    });

    size_t n_entries = 0;
    for (const auto & entry : entries) {
        if (n_entries > 0 && entries[n_entries - 1].tensor == entry.tensor && entries[n_entries - 1].expert_id == entry.expert_id) {
            entries[n_entries - 1].frequency = std::max(entries[n_entries - 1].frequency, entry.frequency);
            continue;
        }
        entries[n_entries++] = entry;
    }
    entries.resize(n_entries);
    std::sort(entries.begin(), entries.end(), [](const expert_cache_seed_entry & a, const expert_cache_seed_entry & b) {
        if (a.frequency != b.frequency) {
            return a.frequency < b.frequency;
        }
        if (a.tensor != b.tensor) {
            return std::less<const struct ggml_tensor *>()(a.tensor, b.tensor);
        }
        return a.expert_id < b.expert_id;
    });


    std::vector<bool> seeded(entries.size(), false);
    for (int b = 0; b < ggml_backend_sched_get_n_backends(sched); b++) {
        for (size_t i = 0; i < entries.size(); i++) {
            const auto & entry = entries[i];
            seeded[i] = ggml_backend_sched_expert_cache_seed(
                sched, b, entry.tensor, entry.expert_id, entry.frequency) || seeded[i];
        }
    }

    const size_t seeded_count = std::count(seeded.begin(), seeded.end(), true);
    if (skipped_count > 0) {
        LOG_WRN("expert_cache: skipped %zu invalid profile entries from '%s'\n", skipped_count, file_path.c_str());
    }
    if (seeded_count > 0) {
        ggml_backend_sched_expert_cache_sync(sched);
        LOG_INF("expert_cache: loaded profile '%s' (%zu hot experts seeded from '%s')\n",
            profile.c_str(), seeded_count, file_path.c_str());
    }

    return seeded_count;
}

bool common_expert_cache_save_profile(
        const struct llama_context * ctx,
        const struct llama_model * model,
        const std::string & file_path,
        const std::string & profile_name) {
    if (ctx == nullptr || model == nullptr || file_path.empty()) {
        return false;
    }

    ggml_backend_sched_t sched = llama_context_get_sched(ctx);
    if (sched == nullptr) {
        return false;
    }

    std::vector<struct ggml_backend_expert_cache_export_entry> entries(4096);
    size_t n_entries = ggml_backend_sched_expert_cache_export_entries(
        sched, -1, entries.data(), entries.size());

    if (n_entries == 0) {
        return true;
    }

    entries.resize(n_entries);
    entries.erase(
        std::remove_if(entries.begin(), entries.end(),
            [](const ggml_backend_expert_cache_export_entry & e) {
                return e.frequency == 0 && e.hit_count == 0;
            }),
        entries.end());

    if (entries.empty()) {
        return true;
    }

    std::sort(entries.begin(), entries.end(),
        [](const ggml_backend_expert_cache_export_entry & a, const ggml_backend_expert_cache_export_entry & b) {
            if (a.frequency != b.frequency) {
                return a.frequency > b.frequency;
            }
            return a.hit_count > b.hit_count;
        });

    const size_t max_profile_entries = 512;
    if (entries.size() > max_profile_entries) {
        entries.resize(max_profile_entries);
    }

    json j;
    j["version"] = 1;
    j["profile"] = profile_name.empty() ? "default" : profile_name;
    j["n_entries"] = entries.size();

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    struct tm tm_buf;
#if defined(_WIN32)
    gmtime_s(&tm_buf, &in_time_t);
#else
    gmtime_r(&in_time_t, &tm_buf);
#endif
    ss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%SZ");
    j["updated_at"] = ss.str();

    json j_experts = json::array();
    for (const auto & entry : entries) {
        if (entry.tensor == nullptr || entry.tensor->name[0] == '\0') {
            continue;
        }

        json item;
        item["tensor"]    = std::string(entry.tensor->name);
        item["expert_id"] = entry.expert_id;
        item["frequency"] = entry.frequency;
        item["hit_count"] = entry.hit_count;
        j_experts.push_back(item);
    }
    j["experts"] = j_experts;

    std::string tmp_path = file_path + ".tmp";
    {
        std::ofstream out(tmp_path);
        if (!out.is_open()) {
            LOG_WRN("%s: failed to open temporary file '%s' for writing\n", __func__, tmp_path.c_str());
            return false;
        }
        out << j.dump(2) << "\n";
    }

    std::error_code ec;
    std::filesystem::rename(tmp_path, file_path, ec);
    if (ec) {
        std::filesystem::remove(file_path, ec);
        std::filesystem::rename(tmp_path, file_path, ec);
        if (ec) {
            LOG_WRN("%s: failed to rename '%s' to '%s': %s\n",
                __func__, tmp_path.c_str(), file_path.c_str(), ec.message().c_str());
            return false;
        }
    }

    LOG_INF("expert_cache: saved %zu hot experts to '%s'\n", n_entries, file_path.c_str());
    return true;
}
