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

struct ggml_expert_cache_swap_op {
    ggml_expert_cache_key evict_key;
    ggml_expert_cache_key load_key;
    size_t offset;
    size_t size;
    size_t alloc_size;
    bool   is_forward;
};

struct ggml_backend_expert_cache {
    ggml_backend_t backend;

    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
    struct ggml_tensor *  tensor;

    size_t capacity;
    size_t used;

    uint64_t clock;
    int32_t  period_tokens;
    uint64_t decode_step;

    std::vector<ggml_backend_expert_cache_free_block> free_blocks;
    std::unordered_map<ggml_expert_cache_key, ggml_backend_expert_cache_entry, ggml_expert_cache_key_hash, ggml_expert_cache_key_eq> entries;

    std::unordered_map<ggml_expert_cache_key, uint32_t, ggml_expert_cache_key_hash, ggml_expert_cache_key_eq> access_freq;
    std::vector<ggml_expert_cache_swap_op> pending_forward_swaps;

    struct ggml_backend_expert_cache_stats stats;
};

static int ggml_expert_cache_get_tensor_layer(const struct ggml_tensor * tensor) {
    if (tensor == NULL || tensor->name[0] == '\0') {
        return -1;
    }
    int layer = -1;
    if (sscanf(tensor->name, "blk.%d.", &layer) == 1) {
        return layer;
    }
    return -1;
}

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
    cache->period_tokens = 64;
    cache->decode_step = 0;
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

void ggml_backend_expert_cache_set_period(
        ggml_backend_expert_cache_t cache,
        int32_t period) {
    if (cache == NULL) {
        return;
    }
    cache->period_tokens = period;
}

int32_t ggml_backend_expert_cache_get_period(
        ggml_backend_expert_cache_t cache) {
    if (cache == NULL) {
        return 0;
    }
    return cache->period_tokens;
}

static void ggml_backend_expert_cache_rebalance(ggml_backend_expert_cache_t cache) {
    if (cache == NULL || cache->access_freq.empty()) {
        return;
    }

    struct candidate {
        ggml_expert_cache_key key;
        uint32_t freq;
        size_t size;
        size_t alloc_size;
    };
    std::vector<candidate> candidates;
    candidates.reserve(cache->access_freq.size());

    for (const auto & kv : cache->access_freq) {
        if (kv.second > 0 && kv.first.tensor != NULL) {
            const size_t expert_size = kv.first.tensor->nb[2];
            const size_t alloc_size = GGML_EXPERT_CACHE_PAD(expert_size);
            candidates.push_back({ kv.first, kv.second, expert_size, alloc_size });
        }
    }

    if (candidates.empty()) {
        return;
    }

    std::sort(candidates.begin(), candidates.end(), [](const candidate & a, const candidate & b) {
        if (a.freq != b.freq) {
            return a.freq > b.freq;
        }
        return a.key.expert_id < b.key.expert_id;
    });

    std::vector<candidate> desired;
    size_t desired_total_size = 0;
    for (const auto & cand : candidates) {
        if (desired_total_size + cand.alloc_size <= cache->capacity) {
            desired.push_back(cand);
            desired_total_size += cand.alloc_size;
        }
    }

    std::vector<ggml_expert_cache_key> to_evict;
    for (const auto & kv : cache->entries) {
        bool found = false;
        for (const auto & d : desired) {
            if (d.key.tensor == kv.first.tensor && d.key.expert_id == kv.first.expert_id) {
                found = true;
                break;
            }
        }
        if (!found) {
            to_evict.push_back(kv.first);
        }
    }

    std::vector<candidate> to_load;
    for (const auto & d : desired) {
        if (cache->entries.find(d.key) == cache->entries.end()) {
            to_load.push_back(d);
        }
    }

    if (to_evict.empty() && to_load.empty()) {
        for (auto & kv : cache->access_freq) {
            kv.second >>= 1;
        }
        return;
    }

    cache->stats.n_rebalances++;
    cache->pending_forward_swaps.clear();

    for (const auto & evict_key : to_evict) {
        auto eit = cache->entries.find(evict_key);
        if (eit != cache->entries.end()) {
            cache->free_blocks.push_back({ eit->second.offset, eit->second.alloc_size });
            cache->used -= eit->second.alloc_size;
            cache->stats.n_evictions++;
            cache->entries.erase(eit);
        }
    }
    ggml_backend_expert_cache_coalesce_free(cache);

    for (const auto & load_cand : to_load) {
        size_t slot_offset = ggml_backend_expert_cache_alloc_slot(
            cache, load_cand.key.tensor, load_cand.key.expert_id, load_cand.size, NULL, 0);
        if (slot_offset != SIZE_MAX) {
            const size_t expert_size = load_cand.size;
            const size_t src_off = (size_t)load_cand.key.expert_id * expert_size;
            ggml_backend_tensor_set_async(
                cache->backend,
                cache->tensor,
                (const uint8_t *)load_cand.key.tensor->data + src_off,
                slot_offset,
                expert_size);
        }
    }

    for (auto & kv : cache->access_freq) {
        kv.second >>= 1;
    }
}

void ggml_backend_expert_cache_begin_step(ggml_backend_expert_cache_t cache) {
    if (cache == NULL) {
        return;
    }
    cache->decode_step++;
    if (cache->period_tokens > 0 && (cache->decode_step % (uint64_t)cache->period_tokens == 0)) {
        ggml_backend_expert_cache_rebalance(cache);
    }
}

void ggml_backend_expert_cache_record_access(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id) {
    if (cache == NULL || tensor == NULL) {
        return;
    }
    ggml_expert_cache_key key = { tensor, expert_id };
    cache->access_freq[key]++;
}

void ggml_backend_expert_cache_record_access_count(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id,
        uint32_t count) {
    if (cache == NULL || tensor == NULL || count == 0) {
        return;
    }
    ggml_expert_cache_key key = { tensor, expert_id };
    cache->access_freq[key] += count;
}

void ggml_backend_expert_cache_process_jit_swaps(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * completed_tensor,
        ggml_backend_t backend) {
    if (cache == NULL || cache->pending_forward_swaps.empty() || completed_tensor == NULL) {
        return;
    }

    for (auto it = cache->pending_forward_swaps.begin(); it != cache->pending_forward_swaps.end(); ) {
        if (it->evict_key.tensor == completed_tensor) {
            auto eit = cache->entries.find(it->evict_key);
            if (eit != cache->entries.end()) {
                cache->entries.erase(eit);
                cache->stats.n_evictions++;
            }

            ggml_backend_expert_cache_entry new_entry = {
                it->load_key.tensor,
                it->load_key.expert_id,
                it->offset,
                it->size,
                it->alloc_size,
                ++cache->clock,
                0,
            };
            cache->entries[it->load_key] = new_entry;
            cache->stats.n_jit_swaps++;

            const size_t expert_size = it->load_key.tensor->nb[2];
            const size_t src_off = (size_t)it->load_key.expert_id * expert_size;
            ggml_backend_tensor_set_async(
                backend,
                cache->tensor,
                (const uint8_t *)it->load_key.tensor->data + src_off,
                it->offset,
                expert_size);

            it = cache->pending_forward_swaps.erase(it);
        } else {
            ++it;
        }
    }
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
    GGML_LOG_INFO("  period:               %d tokens\n", cache->period_tokens);
    GGML_LOG_INFO("  rebalances:           %" PRIu64 "\n", cache->stats.n_rebalances);
    GGML_LOG_INFO("  JIT swaps:            %" PRIu64 "\n", cache->stats.n_jit_swaps);
    GGML_LOG_INFO("  requests:             %" PRIu64 "\n", cache->stats.n_requests);
    GGML_LOG_INFO("  hits:                 %" PRIu64 "\n", cache->stats.n_hits);
    GGML_LOG_INFO("  misses:               %" PRIu64 "\n", cache->stats.n_misses);
    GGML_LOG_INFO("  hit rate:             %8.2f %%\n", hit_rate);
    GGML_LOG_INFO("  RAM -> GPU:           %8.2f GiB\n", ram_to_gpu_gib);
    GGML_LOG_INFO("  avoided RAM -> GPU:   %8.2f GiB\n", avoided_gib);
    GGML_LOG_INFO("  evictions:            %" PRIu64 "\n", cache->stats.n_evictions);
}

size_t ggml_backend_expert_cache_export_entries(
        ggml_backend_expert_cache_t cache,
        struct ggml_backend_expert_cache_export_entry * out_entries,
        size_t max_entries) {
    if (cache == NULL || out_entries == NULL || max_entries == 0) {
        return 0;
    }

    size_t count = 0;
    for (const auto & kv : cache->access_freq) {
        if (count >= max_entries) {
            break;
        }
        if (kv.first.tensor == NULL) {
            continue;
        }

        uint64_t hits = 0;
        auto eit = cache->entries.find(kv.first);
        if (eit != cache->entries.end()) {
            hits = eit->second.hit_count;
        }

        out_entries[count].tensor    = kv.first.tensor;
        out_entries[count].expert_id = kv.first.expert_id;
        out_entries[count].frequency = kv.second;
        out_entries[count].hit_count = hits;
        count++;
    }

    // Also include any resident entries that might have 0 access_freq in current window
    for (const auto & kv : cache->entries) {
        if (count >= max_entries) {
            break;
        }
        if (cache->access_freq.find(kv.first) == cache->access_freq.end()) {
            out_entries[count].tensor    = kv.first.tensor;
            out_entries[count].expert_id = kv.first.expert_id;
            out_entries[count].frequency = 0;
            out_entries[count].hit_count = kv.second.hit_count;
            count++;
        }
    }

    return count;
}

bool ggml_backend_expert_cache_seed(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id,
        uint32_t frequency) {
    if (cache == NULL || tensor == NULL || expert_id < 0 || tensor->data == NULL) {
        return false;
    }

    ggml_expert_cache_key key = { tensor, expert_id };
    cache->access_freq[key] = std::max(cache->access_freq[key], frequency);

    if (cache->entries.find(key) != cache->entries.end()) {
        return true;
    }

    const size_t expert_size = tensor->nb[2];
    size_t slot_offset = ggml_backend_expert_cache_alloc_slot(
        cache, tensor, expert_id, expert_size, NULL, 0);

    if (slot_offset == SIZE_MAX) {
        return false;
    }

    const size_t src_off = (size_t)expert_id * expert_size;
    ggml_backend_tensor_set_async(
        cache->backend,
        cache->tensor,
        (const uint8_t *)tensor->data + src_off,
        slot_offset,
        expert_size);

    return true;
}

