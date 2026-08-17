#include "ggml-backend-expert-cache.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#define GGML_EXPERT_CACHE_ALIGN 512
#define GGML_EXPERT_CACHE_PAD(x) (((x) + GGML_EXPERT_CACHE_ALIGN - 1) & ~(GGML_EXPERT_CACHE_ALIGN - 1))

struct ggml_expert_cache_key_hash {
    size_t operator()(const ggml_expert_cache_key & k) const {
        return std::hash<const void *>()(k.tensor) ^
               (std::hash<int32_t>()(k.expert_id) + 0x9e3779b9 +
               (std::hash<const void *>()(k.tensor) << 6) +
               (std::hash<const void *>()(k.tensor) >> 2));
    }
};

struct ggml_expert_cache_key_eq {
    bool operator()(const ggml_expert_cache_key & a, const ggml_expert_cache_key & b) const {
        return a.tensor == b.tensor && a.expert_id == b.expert_id;
    }
};

struct ggml_backend_expert_cache_entry {
    const struct ggml_tensor * tensor;
    int32_t expert_id;

    size_t offset;
    size_t size;
    size_t alloc_size;

    uint64_t last_used;
    uint64_t hit_count;
};

struct ggml_backend_expert_cache_free_block {
    size_t offset;
    size_t size;
};

struct ggml_backend_expert_cache {
    ggml_backend_t backend;

    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
    struct ggml_tensor *  tensor;

    size_t capacity;
    size_t used;

    uint64_t clock;

    std::vector<ggml_backend_expert_cache_free_block> free_blocks;
    std::unordered_map<ggml_expert_cache_key, ggml_backend_expert_cache_entry, ggml_expert_cache_key_hash, ggml_expert_cache_key_eq> entries;

    struct ggml_backend_expert_cache_stats stats;
};

static void ggml_backend_expert_cache_coalesce_free(ggml_backend_expert_cache_t cache) {
    if (cache->free_blocks.size() <= 1) {
        return;
    }

    std::sort(cache->free_blocks.begin(), cache->free_blocks.end(),
        [](const ggml_backend_expert_cache_free_block & a, const ggml_backend_expert_cache_free_block & b) {
            return a.offset < b.offset;
        });

    std::vector<ggml_backend_expert_cache_free_block> merged;
    merged.reserve(cache->free_blocks.size());
    merged.push_back(cache->free_blocks[0]);

    for (size_t i = 1; i < cache->free_blocks.size(); i++) {
        auto & prev = merged.back();
        const auto & cur = cache->free_blocks[i];

        if (prev.offset + prev.size == cur.offset) {
            prev.size += cur.size;
        } else {
            merged.push_back(cur);
        }
    }

    cache->free_blocks = std::move(merged);
}

ggml_backend_expert_cache_t ggml_backend_expert_cache_new(
        ggml_backend_t backend,
        size_t capacity) {
    if (backend == NULL || capacity == 0) {
        return NULL;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(backend, capacity);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes on backend %s\n",
            __func__, capacity, ggml_backend_name(backend));
        return NULL;
    }

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead(),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (ctx == NULL) {
        ggml_backend_buffer_free(buffer);
        return NULL;
    }

    struct ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, capacity);
    if (ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer)) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
        return NULL;
    }

    ggml_backend_expert_cache_t cache = new ggml_backend_expert_cache();
    cache->backend = backend;
    cache->buffer = buffer;
    cache->ctx = ctx;
    cache->tensor = tensor;
    cache->capacity = capacity;
    cache->used = 0;
    cache->clock = 0;
    cache->free_blocks.push_back({ 0, capacity });
    memset(&cache->stats, 0, sizeof(cache->stats));

    return cache;
}

void ggml_backend_expert_cache_free(ggml_backend_expert_cache_t cache) {
    if (cache == NULL) {
        return;
    }
    if (cache->ctx) {
        ggml_free(cache->ctx);
    }
    if (cache->buffer) {
        ggml_backend_buffer_free(cache->buffer);
    }
    delete cache;
}

struct ggml_tensor * ggml_backend_expert_cache_get_tensor(ggml_backend_expert_cache_t cache) {
    GGML_ASSERT(cache != NULL);
    return cache->tensor;
}

size_t ggml_backend_expert_cache_find_offset(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id) {
    if (cache == NULL) {
        return SIZE_MAX;
    }

    ggml_expert_cache_key key = { tensor, expert_id };
    auto it = cache->entries.find(key);
    if (it != cache->entries.end()) {
        return it->second.offset;
    }
    return SIZE_MAX;
}

void ggml_backend_expert_cache_touch(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id) {
    if (cache == NULL) {
        return;
    }

    ggml_expert_cache_key key = { tensor, expert_id };
    auto it = cache->entries.find(key);
    if (it != cache->entries.end()) {
        it->second.last_used = ++cache->clock;
        it->second.hit_count++;
    }
}

void ggml_backend_expert_cache_record_hit(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id,
        size_t bytes_avoided) {
    if (cache == NULL) {
        return;
    }
    cache->stats.n_requests++;
    cache->stats.n_hits++;
    cache->stats.bytes_avoided += bytes_avoided;
    ggml_backend_expert_cache_touch(cache, tensor, expert_id);
}

void ggml_backend_expert_cache_record_miss(
        ggml_backend_expert_cache_t cache,
        size_t bytes_ram_to_gpu) {
    if (cache == NULL) {
        return;
    }
    cache->stats.n_requests++;
    cache->stats.n_misses++;
    cache->stats.bytes_ram_to_gpu += bytes_ram_to_gpu;
}

size_t ggml_backend_expert_cache_alloc_slot(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id,
        size_t size,
        const struct ggml_expert_cache_key * pinned_keys,
        size_t n_pinned) {
    if (cache == NULL || size == 0) {
        return SIZE_MAX;
    }

    ggml_expert_cache_key key = { tensor, expert_id };
    auto it = cache->entries.find(key);
    if (it != cache->entries.end()) {
        it->second.last_used = ++cache->clock;
        return it->second.offset;
    }

    const size_t alloc_size = GGML_EXPERT_CACHE_PAD(size);
    if (alloc_size > cache->capacity) {
        return SIZE_MAX;
    }

    // Try allocating from free blocks
    auto try_alloc = [&]() -> size_t {
        for (size_t i = 0; i < cache->free_blocks.size(); i++) {
            if (cache->free_blocks[i].size >= alloc_size) {
                size_t offset = cache->free_blocks[i].offset;
                cache->free_blocks[i].offset += alloc_size;
                cache->free_blocks[i].size -= alloc_size;
                if (cache->free_blocks[i].size == 0) {
                    cache->free_blocks.erase(cache->free_blocks.begin() + i);
                }
                cache->used += alloc_size;
                return offset;
            }
        }
        return SIZE_MAX;
    };

    size_t offset = try_alloc();
    if (offset != SIZE_MAX) {
        ggml_backend_expert_cache_entry entry = {
            /* .tensor     = */ tensor,
            /* .expert_id  = */ expert_id,
            /* .offset     = */ offset,
            /* .size       = */ size,
            /* .alloc_size = */ alloc_size,
            /* .last_used  = */ ++cache->clock,
            /* .hit_count  = */ 0,
        };
        cache->entries[key] = entry;
        return offset;
    }

    // Free memory is insufficient or fragmented; collect candidates for LRU eviction
    std::vector<ggml_expert_cache_key> evict_candidates;
    evict_candidates.reserve(cache->entries.size());

    for (const auto & kv : cache->entries) {
        bool pinned = false;
        if (pinned_keys != NULL && n_pinned > 0) {
            for (size_t p = 0; p < n_pinned; p++) {
                if (pinned_keys[p].tensor == kv.first.tensor && pinned_keys[p].expert_id == kv.first.expert_id) {
                    pinned = true;
                    break;
                }
            }
        }
        if (!pinned) {
            evict_candidates.push_back(kv.first);
        }
    }

    // Sort unpinned entries by last_used ascending (oldest first)
    std::sort(evict_candidates.begin(), evict_candidates.end(),
        [&](const ggml_expert_cache_key & a, const ggml_expert_cache_key & b) {
            return cache->entries[a].last_used < cache->entries[b].last_used;
        });

    for (const auto & evict_key : evict_candidates) {
        auto eit = cache->entries.find(evict_key);
        if (eit == cache->entries.end()) {
            continue;
        }

        const auto & evict_entry = eit->second;
        cache->free_blocks.push_back({ evict_entry.offset, evict_entry.alloc_size });
        cache->used -= evict_entry.alloc_size;
        cache->stats.n_evictions++;
        cache->entries.erase(eit);

        ggml_backend_expert_cache_coalesce_free(cache);

        offset = try_alloc();
        if (offset != SIZE_MAX) {
            ggml_backend_expert_cache_entry entry = {
                /* .tensor     = */ tensor,
                /* .expert_id  = */ expert_id,
                /* .offset     = */ offset,
                /* .size       = */ size,
                /* .alloc_size = */ alloc_size,
                /* .last_used  = */ ++cache->clock,
                /* .hit_count  = */ 0,
            };
            cache->entries[key] = entry;
            return offset;
        }
    }

    return SIZE_MAX;
}

void ggml_backend_expert_cache_get_stats(
        ggml_backend_expert_cache_t cache,
        struct ggml_backend_expert_cache_stats * stats) {
    if (cache == NULL || stats == NULL) {
        return;
    }
    *stats = cache->stats;
}

void ggml_backend_expert_cache_print_stats(ggml_backend_expert_cache_t cache) {
    if (cache == NULL) {
        return;
    }

    const double hit_rate = cache->stats.n_requests > 0 ?
        (100.0 * (double)cache->stats.n_hits / (double)cache->stats.n_requests) : 0.0;
    const double cap_mib = (double)cache->capacity / (1024.0 * 1024.0);
    const double used_mib = (double)cache->used / (1024.0 * 1024.0);
    const double ram_to_gpu_gib = (double)cache->stats.bytes_ram_to_gpu / (1024.0 * 1024.0 * 1024.0);
    const double avoided_gib = (double)cache->stats.bytes_avoided / (1024.0 * 1024.0 * 1024.0);

    GGML_LOG_INFO("\n");
    GGML_LOG_INFO("Expert Cache (%s):\n", ggml_backend_name(cache->backend));
    GGML_LOG_INFO("  capacity:             %8.2f MiB\n", cap_mib);
    GGML_LOG_INFO("  resident:             %8.2f MiB (%zu entries)\n", used_mib, cache->entries.size());
    GGML_LOG_INFO("  requests:             %" PRIu64 "\n", cache->stats.n_requests);
    GGML_LOG_INFO("  hits:                 %" PRIu64 "\n", cache->stats.n_hits);
    GGML_LOG_INFO("  misses:               %" PRIu64 "\n", cache->stats.n_misses);
    GGML_LOG_INFO("  hit rate:             %8.2f %%\n", hit_rate);
    GGML_LOG_INFO("  RAM -> GPU:           %8.2f GiB\n", ram_to_gpu_gib);
    GGML_LOG_INFO("  avoided RAM -> GPU:   %8.2f GiB\n", avoided_gib);
    GGML_LOG_INFO("  evictions:            %" PRIu64 "\n", cache->stats.n_evictions);
}
