#pragma once

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

struct common_expert_cache_profile_entry {
    std::string tensor_name;
    int32_t     expert_id = 0;
    uint32_t    frequency = 0;
    uint64_t    hit_count = 0;
};

// Computes the path to the expert cache profile file.
// If explicit_file_path is non-empty, returns explicit_file_path.
// If profile_name is non-empty, returns "<model_path>.<profile_name>.expert_cache.json".
// Otherwise, returns "<model_path>.expert_cache.json".
std::string common_expert_cache_get_file_path(
    const std::string & model_path,
    const std::string & profile_name,
    const std::string & explicit_file_path);
// Sort seed entries for cache admission: deduplicate by (tensor_name, expert_id)
// keeping the maximum frequency, then order highest frequency first so hot
// experts are admitted before cache capacity fills.
void common_expert_cache_sort_entries(std::vector<common_expert_cache_profile_entry> & entries);


// Loads the expert cache profile JSON from disk and pre-seeds the expert cache.
// Returns the number of experts successfully seeded into cache.
size_t common_expert_cache_load_profile(
    struct llama_context * ctx,
    const struct llama_model * model,
    const std::string & file_path);

// Exports the current expert cache hotness and resident status to a JSON file on disk.
// Writes atomically using a temporary file.
bool common_expert_cache_save_profile(
    const struct llama_context * ctx,
    const struct llama_model * model,
    const std::string & file_path,
    const std::string & profile_name);
