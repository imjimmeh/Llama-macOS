#include "ggml-backend-expert-cache.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"


#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <unordered_map>
#include <vector>

#define GGML_EXPERT_CACHE_ALIGN 512
#define GGML_EXPERT_CACHE_PAD(x) (((x) + GGML_EXPERT_CACHE_ALIGN - 1) & ~(GGML_EXPERT_CACHE_ALIGN - 1))
#define GGML_EXPERT_CACHE_STAGING_ENTRIES 32

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

struct ggml_expert_cache_pool_key {
    int64_t ne0;
    int64_t ne1;
    enum ggml_type type;
    size_t stride;

    bool operator==(const ggml_expert_cache_pool_key & o) const {
        return ne0 == o.ne0 && ne1 == o.ne1 && type == o.type && stride == o.stride;
    }
};

struct ggml_expert_cache_pool_key_hash {
    size_t operator()(const ggml_expert_cache_pool_key & k) const {
        return std::hash<int64_t>()(k.ne0) ^
               (std::hash<int64_t>()(k.ne1) << 1) ^
               (std::hash<int>()((int)k.type) << 2) ^
               (std::hash<size_t>()(k.stride) << 3);
    }
};


struct ggml_expert_cache_slot_entry {
    const struct ggml_tensor * tensor = nullptr;
    int32_t expert_id = -1;
    int32_t layer = -1;
    enum ggml_expert_cache_slot_state state = GGML_EXPERT_CACHE_SLOT_EMPTY;
    enum ggml_expert_cache_segment segment = GGML_EXPERT_CACHE_SEG_PROBATIONARY;
    uint64_t last_used = 0;
    uint64_t hit_count = 0;
    uint32_t access_in_window = 0;
    bool is_pinned = false;
    ggml_backend_event_t last_use_event = nullptr;
    ggml_backend_event_t load_event = nullptr;
};

struct ggml_expert_cache_slot_pool {
    struct ggml_context * ctx = nullptr;
    struct ggml_tensor *  tensor = nullptr; // 3D tensor: [ne0, ne1, max_slots]
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    enum ggml_type type = GGML_TYPE_F32;
    size_t stride = 0;
    size_t buffer_offset = 0;
    int32_t max_slots = 0;
    int32_t used_slots = 0;

    int32_t probationary_cap = 0;
    int32_t protected_cap = 0;
    int32_t probationary_used = 0;
    int32_t protected_used = 0;

    // Per-tensor slot maps: tensor pointer -> vector of slot indices for each expert_id (P0 fix)
    std::unordered_map<const struct ggml_tensor *, std::vector<int32_t>> slot_maps;
    std::vector<ggml_expert_cache_slot_entry> slots;

    // Admission hysteresis & anti-thrashing (Phase 5)
    std::unordered_map<ggml_expert_cache_key, uint64_t, ggml_expert_cache_key_hash, ggml_expert_cache_key_eq> ghost_sightings;
    std::unordered_map<ggml_expert_cache_key, uint32_t, ggml_expert_cache_key_hash, ggml_expert_cache_key_eq> eviction_miss_counts;
};

struct ggml_expert_bundle_reg {
    const struct ggml_tensor * gate = nullptr;
    const struct ggml_tensor * up   = nullptr;
    const struct ggml_tensor * down = nullptr;
};

struct ggml_pending_swap_entry {
    const struct ggml_tensor * tensor = nullptr;
    int32_t expert_id = -1;
    int32_t slot = -1;
};

struct ggml_expert_cache_async_promotion {
    const struct ggml_tensor * tensor = nullptr;
    int32_t layer = -1;
    int32_t expert_id = -1;
    int32_t slot = -1;
};

struct ggml_backend_expert_cache {
    ggml_backend_t backend;

    // Single pre-allocated device buffer enforcing strict hard VRAM cap
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
    struct ggml_tensor *  tensor; // 1D tensor view spanning the entire buffer

    size_t capacity;
    size_t pool_alloc_offset = 0;

    uint64_t clock = 0;
    int32_t  period_tokens = 64;
    int32_t  max_swaps = -1;
    uint64_t decode_step = 0;

    // Frequency counters: separated TG (decode) vs PP (prefill) (Phase 6)
    std::unordered_map<ggml_expert_cache_key, uint32_t, ggml_expert_cache_key_hash, ggml_expert_cache_key_eq> tg_access_freq;
    std::unordered_map<ggml_expert_cache_key, uint32_t, ggml_expert_cache_key_hash, ggml_expert_cache_key_eq> pp_access_freq;

    // Unified slot pools (the ONE and ONLY cache representation)
    std::unordered_map<ggml_expert_cache_pool_key, ggml_expert_cache_slot_pool, ggml_expert_cache_pool_key_hash> slot_pools;

    // Expert bundle registrations
    std::unordered_map<int32_t, ggml_expert_bundle_reg> bundle_registrations;

    // Per-layer budget tracking
    std::unordered_map<int32_t, int32_t> layer_slots;

    // JIT Staged layer swap queues (Phase 7)
    std::unordered_map<int32_t, std::vector<ggml_pending_swap_entry>> pending_layer_swaps;

    // Genuine CUDA Pinned Host DMA Staging Arena with Safe Lifetime (Phase 2 / P1)
    void * pinned_host_buffer = nullptr;
    size_t pinned_host_capacity = 0;
    bool   pinned_is_cuda = false;

    // Staging ring keyed by (tensor, slot_idx)
    std::vector<ggml_backend_event_t> staging_events;
    std::vector<bool>                 staging_in_flight;

    // Host Registered Direct DMA
    std::vector<std::pair<uintptr_t, size_t>> registered_host_ranges;
    size_t registered_host_bytes = 0;
    size_t max_registered_host_bytes = 1024 * 1024 * 1024; // 1 GiB safety cap

    // Epic 4: Static Pinned Expert Manifests & Heterogeneous Routing
    std::set<std::pair<int32_t, int32_t>> pinned_bundles;
    bool is_static_manifest_loaded = false;
    bool zero_miss_upload = true;

    // Epic 5: Non-blocking Background Promotion Pipeline
    std::vector<ggml_expert_cache_async_promotion> async_promotions;
    int32_t max_async_promotions = 2;

    // Epic 6: GPU-Side Zero-Sync Route Remapping (128 layers x 512 experts)
    std::vector<int32_t> gpu_slot_map_table;

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

static ggml_expert_cache_slot_pool * ggml_backend_expert_cache_get_or_create_pool(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * weight_tensor) {
    if (cache == NULL || weight_tensor == NULL) {
        return NULL;
    }

    const int64_t ne0 = weight_tensor->ne[0];
    const int64_t ne1 = weight_tensor->ne[1];
    const enum ggml_type type = weight_tensor->type;
    const size_t stride = weight_tensor->nb[2];

    if (stride == 0 || stride > cache->capacity) {
        return NULL;
    }

    ggml_expert_cache_pool_key pkey = { ne0, ne1, type, stride };
    auto it = cache->slot_pools.find(pkey);
    if (it != cache->slot_pools.end()) {
        return &it->second;
    }

    const size_t remaining_bytes = (cache->pool_alloc_offset < cache->capacity) ?
        (cache->capacity - cache->pool_alloc_offset) : 0;

    if (remaining_bytes < stride) {
        return NULL; // Strict VRAM cap: never allocate beyond capacity
    }

    // Determine target pool bytes
    size_t target_pool_bytes = remaining_bytes;
    if (!cache->bundle_registrations.empty()) {
        std::vector<ggml_expert_cache_pool_key> unique_keys;
        for (const auto & b : cache->bundle_registrations) {
            const struct ggml_tensor * t_list[3] = { b.second.gate, b.second.up, b.second.down };
            for (int ti = 0; ti < 3; ti++) {
                if (t_list[ti]) {
                    ggml_expert_cache_pool_key k = { t_list[ti]->ne[0], t_list[ti]->ne[1], t_list[ti]->type, (size_t)t_list[ti]->nb[2] };
                    if (std::find(unique_keys.begin(), unique_keys.end(), k) == unique_keys.end()) {
                        unique_keys.push_back(k);
                    }
                }
            }
        }
        size_t u = unique_keys.empty() ? 2 : unique_keys.size();
        target_pool_bytes = std::min(remaining_bytes, cache->capacity / u);
    } else {
        size_t needed_bytes = (weight_tensor->ne[2] > 0) ? ((size_t)weight_tensor->ne[2] * stride) : remaining_bytes;
        if (needed_bytes <= remaining_bytes) {
            target_pool_bytes = needed_bytes;
        } else if (remaining_bytes >= 2 * stride) {
            target_pool_bytes = remaining_bytes / 2;
        }
    }
    target_pool_bytes = std::min(target_pool_bytes, remaining_bytes);

    int32_t max_slots = (int32_t)(target_pool_bytes / stride);
    if (max_slots <= 0) {
        max_slots = 1;
    }
    const size_t pool_bytes = (size_t)max_slots * stride;
    if (pool_bytes > remaining_bytes) {
        max_slots = (int32_t)(remaining_bytes / stride);
        if (max_slots <= 0) {
            return NULL;
        }
    }

    struct ggml_init_params params = {
        /* .mem_size   = */ 4 * ggml_tensor_overhead(),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * pctx = ggml_init(params);
    if (pctx == NULL) {
        return NULL;
    }

    struct ggml_tensor * ptensor = ggml_new_tensor_3d(pctx, type, ne0, ne1, max_slots);
    if (ptensor == NULL) {
        ggml_free(pctx);
        return NULL;
    }

    ptensor->data = (uint8_t *)ggml_backend_buffer_get_base(cache->buffer) + cache->pool_alloc_offset;
    ptensor->buffer = cache->buffer;
    const size_t pool_offset = cache->pool_alloc_offset;
    cache->pool_alloc_offset += (size_t)max_slots * stride;

    ptensor->nb[0] = weight_tensor->nb[0];
    ptensor->nb[1] = weight_tensor->nb[1];
    ptensor->nb[2] = stride;
    ptensor->nb[3] = stride * max_slots;

    ggml_expert_cache_slot_pool pool;
    pool.ctx = pctx;
    pool.tensor = ptensor;
    pool.ne0 = ne0;
    pool.ne1 = ne1;
    pool.type = type;
    pool.stride = stride;
    pool.buffer_offset = pool_offset;
    pool.max_slots = max_slots;
    pool.used_slots = 0;
    pool.probationary_cap = std::max(1, (int32_t)(max_slots * 0.20));
    pool.protected_cap = max_slots - pool.probationary_cap;
    pool.probationary_used = 0;
    pool.protected_used = 0;
    pool.slots.resize(max_slots);

    cache->slot_pools[pkey] = std::move(pool);
    return &cache->slot_pools[pkey];
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
        /* .mem_size   = */ 8 * ggml_tensor_overhead(),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (ctx == NULL) {
        ggml_backend_buffer_free(buffer);
        return NULL;
    }

    struct ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, capacity);
    tensor->data = ggml_backend_buffer_get_base(buffer);
    tensor->buffer = buffer;

    ggml_backend_expert_cache_t cache = new ggml_backend_expert_cache();
    cache->backend = backend;
    cache->buffer = buffer;
    cache->ctx = ctx;
    cache->tensor = tensor;
    cache->capacity = capacity;
    cache->pool_alloc_offset = 0;
    cache->clock = 0;
    cache->period_tokens = 64;
    cache->max_swaps = -1;
    cache->decode_step = 0;
    cache->pinned_host_buffer = nullptr;
    cache->pinned_host_capacity = 0;
    cache->pinned_is_cuda = false;
    cache->gpu_slot_map_table.assign(128 * 512, -1);
    memset(&cache->stats, 0, sizeof(cache->stats));

    return cache;
}

static void ggml_expert_cache_wait_slot(ggml_expert_cache_slot_entry & slot) {
    if (slot.load_event != nullptr) {
        ggml_backend_event_synchronize(slot.load_event);
        ggml_backend_event_free(slot.load_event);
        slot.load_event = nullptr;
    }
    if (slot.last_use_event != nullptr) {
        ggml_backend_event_synchronize(slot.last_use_event);
        ggml_backend_event_free(slot.last_use_event);
        slot.last_use_event = nullptr;
    }
}

static bool ggml_expert_cache_poll_slot(
        ggml_backend_expert_cache_t cache,
        ggml_expert_cache_slot_entry & slot) {
    if (slot.state == GGML_EXPERT_CACHE_SLOT_RESIDENT) {
        return true;
    }
    if (slot.state != GGML_EXPERT_CACHE_SLOT_LOADING) {
        return false;
    }

    if (slot.load_event == nullptr) {
        return false;
    }

    if (!ggml_backend_event_query(slot.load_event)) {
        return false;
    }

    ggml_backend_event_free(slot.load_event);
    slot.load_event = nullptr;
    slot.state = GGML_EXPERT_CACHE_SLOT_RESIDENT;
    if (slot.layer >= 0 && slot.layer < 128 && slot.expert_id >= 0 && slot.expert_id < 512) {
        int32_t slot_idx = -1;
        auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, slot.tensor);
        if (pool) {
            auto smit = pool->slot_maps.find(slot.tensor);
            if (smit != pool->slot_maps.end() && (size_t)slot.expert_id < smit->second.size()) {
                slot_idx = smit->second[slot.expert_id];
            }
        }
        cache->gpu_slot_map_table[slot.layer * 512 + slot.expert_id] = slot_idx;
    }
    return true;
}

bool ggml_backend_expert_cache_can_store(
        ggml_backend_expert_cache_t cache,
        size_t expert_size) {
    return cache != NULL && expert_size > 0 && expert_size <= cache->capacity;
}

void ggml_backend_expert_cache_free(ggml_backend_expert_cache_t cache) {
    if (cache == NULL) {
        return;
    }

    ggml_backend_expert_cache_sync_staging(cache);

    for (auto & kv : cache->slot_pools) {
        for (auto & slot : kv.second.slots) {
            ggml_expert_cache_wait_slot(slot);
        }
        if (kv.second.ctx) {
            ggml_free(kv.second.ctx);
        }
    }
    cache->slot_pools.clear();

    if (cache->pinned_host_buffer) {
        if (cache->pinned_is_cuda) {
#if defined(GGML_USE_CUDA)
            cudaFreeHost(cache->pinned_host_buffer);
#endif
        } else {
#if defined(_WIN32)
            _aligned_free(cache->pinned_host_buffer);
#else
            free(cache->pinned_host_buffer);
#endif
        }
        cache->pinned_host_buffer = nullptr;
        cache->pinned_is_cuda = false;
    }

    for (size_t i = 0; i < cache->staging_events.size(); i++) {
        if (cache->staging_events[i] != NULL) {
            ggml_backend_event_free(cache->staging_events[i]);
        }
    }
    cache->staging_events.clear();

#if defined(GGML_USE_CUDA)
    for (const auto & range : cache->registered_host_ranges) {
        cudaHostUnregister((void *)range.first);
    }
    cache->registered_host_ranges.clear();
#endif

    cache->async_promotions.clear();

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

void ggml_backend_expert_cache_set_max_swaps(
        ggml_backend_expert_cache_t cache,
        int32_t max_swaps) {
    if (cache == NULL) {
        return;
    }
    cache->max_swaps = max_swaps;
}

int32_t ggml_backend_expert_cache_get_period(
        ggml_backend_expert_cache_t cache) {
    if (cache == NULL) {
        return 0;
    }
    return cache->period_tokens;
}

void ggml_backend_expert_cache_rebalance(ggml_backend_expert_cache_t cache, int max_swaps) {
    if (cache == NULL || cache->tg_access_freq.empty()) {
        return;
    }

    struct candidate {
        ggml_expert_cache_key key;
        uint32_t freq;
        size_t size;
        double value;
    };

    cache->stats.n_rebalances++;

    for (auto & pkv : cache->slot_pools) {
        auto & pool = pkv.second;
        if (pool.tensor == NULL || pool.max_slots <= 0) continue;

        std::vector<candidate> candidates;
        for (const auto & fkv : cache->tg_access_freq) {
            if (fkv.second > 0 && fkv.first.tensor != NULL) {
                if (fkv.first.tensor->ne[0] == pool.ne0 &&
                    fkv.first.tensor->ne[1] == pool.ne1 &&
                    fkv.first.tensor->type == pool.type &&
                    fkv.first.tensor->nb[2] == pool.stride) {
                    candidates.push_back({ fkv.first, fkv.second, pool.stride, (double)fkv.second });
                }
            }
        }

        if (candidates.empty()) continue;

        std::sort(candidates.begin(), candidates.end(), [](const candidate & a, const candidate & b) {
            if (a.freq != b.freq) return a.freq > b.freq;
            if (a.key.tensor != b.key.tensor) return a.key.tensor < b.key.tensor;
            return a.key.expert_id < b.key.expert_id;
        });

        const size_t desired_count = std::min((size_t)pool.max_slots, candidates.size());
        std::vector<candidate> desired(candidates.begin(), candidates.begin() + desired_count);

        std::vector<int32_t> slots_to_evict;
        for (int32_t s = 0; s < pool.max_slots; s++) {
            if (pool.slots[s].tensor == NULL) continue;
            bool keep = false;
            for (const auto & d : desired) {
                if (d.key.tensor == pool.slots[s].tensor && d.key.expert_id == pool.slots[s].expert_id) {
                    keep = true;
                    break;
                }
            }
            if (!keep && pool.slots[s].state != GGML_EXPERT_CACHE_SLOT_LOADING) {
                slots_to_evict.push_back(s);
            }
        }

        std::vector<candidate> to_load;
        for (const auto & d : desired) {
            auto smit = pool.slot_maps.find(d.key.tensor);
            int32_t slot = -1;
            if (smit != pool.slot_maps.end() && d.key.expert_id >= 0 && (size_t)d.key.expert_id < smit->second.size()) {
                slot = smit->second[d.key.expert_id];
            }
            if (slot < 0) {
                to_load.push_back(d);
            }
        }

        int swaps_done = 0;
        size_t evict_idx = 0;
        for (const auto & load_cand : to_load) {
            if (max_swaps >= 0 && swaps_done >= max_swaps) break;

            int32_t slot = -1;
            bool slot_was_empty = false;
            for (int32_t s = 0; s < pool.max_slots; s++) {
                if (pool.slots[s].tensor == NULL) {
                    slot = s;
                    slot_was_empty = true;
                    break;
                }
            }
            if (slot == -1 && evict_idx < slots_to_evict.size()) {
                slot = slots_to_evict[evict_idx++];
                auto & vict = pool.slots[slot];
                ggml_expert_cache_wait_slot(vict);
                if (vict.tensor != NULL) {
                    auto vit = pool.slot_maps.find(vict.tensor);
                    if (vit != pool.slot_maps.end() && vict.expert_id >= 0 && (size_t)vict.expert_id < vit->second.size()) {
                        vit->second[vict.expert_id] = -1;
                    }
                }
                if (vict.layer >= 0) cache->layer_slots[vict.layer]--;
                if (vict.segment == GGML_EXPERT_CACHE_SEG_PROBATIONARY) {
                    pool.probationary_used--;
                } else {
                    pool.protected_used--;
                }
                cache->stats.n_evictions++;
            }

            if (slot >= 0) {
                const int32_t layer = ggml_expert_cache_get_tensor_layer(load_cand.key.tensor);
                pool.slots[slot] = { load_cand.key.tensor, load_cand.key.expert_id, layer, GGML_EXPERT_CACHE_SLOT_LOADING, GGML_EXPERT_CACHE_SEG_PROBATIONARY, ++cache->clock, 0, 1, nullptr };
                if (slot_was_empty) {
                    pool.used_slots++;
                }

                auto & sm = pool.slot_maps[load_cand.key.tensor];
                if ((size_t)load_cand.key.expert_id >= sm.size()) {
                    sm.resize((size_t)load_cand.key.expert_id + 1, -1);
                }
                sm[load_cand.key.expert_id] = slot;

                pool.probationary_used++;
                if (layer >= 0) cache->layer_slots[layer]++;

                const size_t expert_size = pool.stride;
                const size_t src_off = (size_t)load_cand.key.expert_id * expert_size;
                const size_t dst_off = (size_t)slot * expert_size;

                void * pinned_buf = ggml_backend_expert_cache_stage_acquire(cache, load_cand.key.tensor, slot, expert_size);
                if (pinned_buf != NULL && load_cand.key.tensor->data != NULL) {
                    memcpy(pinned_buf, (const uint8_t *)load_cand.key.tensor->data + src_off, expert_size);
                    ggml_backend_tensor_set_async(cache->backend, pool.tensor, pinned_buf, dst_off, expert_size);
                    ggml_backend_expert_cache_stage_commit(cache, load_cand.key.tensor, slot);
                } else if (load_cand.key.tensor->data != NULL) {
                    ggml_backend_tensor_set_async(cache->backend, pool.tensor, (const uint8_t *)load_cand.key.tensor->data + src_off, dst_off, expert_size);
                }
                ggml_backend_expert_cache_promote_slot(cache, load_cand.key.tensor, load_cand.key.expert_id, slot);
                swaps_done++;
            }
        }
    }

    for (auto & kv : cache->tg_access_freq) {
        kv.second = (kv.second * 7) >> 3; // smooth 0.875 decay
    }
}

void ggml_backend_expert_cache_begin_step(ggml_backend_expert_cache_t cache) {
    if (cache == NULL) {
        return;
    }
    cache->decode_step++;
    if (cache->period_tokens > 0 && (cache->decode_step % (uint64_t)cache->period_tokens == 0)) {
        ggml_backend_expert_cache_rebalance(cache, cache->max_swaps);
    }
}

void ggml_backend_expert_cache_record_access(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id) {
    ggml_backend_expert_cache_record_access_count(cache, tensor, expert_id, 1, GGML_EXPERT_CACHE_PHASE_TG);
}

void ggml_backend_expert_cache_record_access_count(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id,
        uint32_t count,
        enum ggml_expert_cache_phase phase) {
    if (cache == NULL || tensor == NULL || count == 0) {
        return;
    }
    ggml_expert_cache_key key = { tensor, expert_id };
    if (phase == GGML_EXPERT_CACHE_PHASE_PP) {
        cache->pp_access_freq[key] += count;
    } else {
        cache->tg_access_freq[key] += count;
    }
}

void ggml_backend_expert_cache_process_jit_swaps(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * completed_tensor,
        ggml_backend_t backend) {
    if (cache == NULL || cache->pending_layer_swaps.empty()) {
        return;
    }

    const int32_t layer = completed_tensor ? ggml_expert_cache_get_tensor_layer(completed_tensor) : -1;
    if (layer >= 0) {
        auto it = cache->pending_layer_swaps.find(layer);
        if (it != cache->pending_layer_swaps.end()) {
            for (const auto & swap : it->second) {
                if (swap.tensor == NULL || swap.tensor->data == NULL) continue;
                auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, swap.tensor);
                if (pool == NULL) continue;

                const size_t expert_size = pool->stride;
                const size_t src_off = (size_t)swap.expert_id * expert_size;
                const size_t dst_off = (size_t)swap.slot * expert_size;

                void * pinned_buf = ggml_backend_expert_cache_stage_acquire(cache, swap.tensor, swap.slot, expert_size);
                if (pinned_buf != NULL) {
                    memcpy(pinned_buf, (const uint8_t *)swap.tensor->data + src_off, expert_size);
                    ggml_backend_tensor_set_async(backend ? backend : cache->backend, pool->tensor, pinned_buf, dst_off, expert_size);
                    ggml_backend_expert_cache_stage_commit(cache, swap.tensor, swap.slot);
                } else {
                    ggml_backend_tensor_set_async(backend ? backend : cache->backend, pool->tensor, (const uint8_t *)swap.tensor->data + src_off, dst_off, expert_size);
                }
            }
            cache->pending_layer_swaps.erase(it);
        }
    }
}

struct ggml_tensor * ggml_backend_expert_cache_get_tensor(ggml_backend_expert_cache_t cache) {
    if (cache == NULL) return NULL;
    return cache->tensor;
}

size_t ggml_backend_expert_cache_find_offset(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id) {
    if (cache == NULL || tensor == NULL || expert_id < 0) {
        return SIZE_MAX;
    }
    int32_t slot = ggml_backend_expert_cache_find_slot(cache, tensor, expert_id);
    if (slot >= 0) {
        auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, tensor);
        if (pool != NULL) {
            return pool->buffer_offset + (size_t)slot * pool->stride;
        }
    }
    return SIZE_MAX;
}

void ggml_backend_expert_cache_touch(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id) {
    if (cache == NULL || tensor == NULL || expert_id < 0) {
        return;
    }

    auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, tensor);
    if (pool != NULL) {
        auto smit = pool->slot_maps.find(tensor);
        if (smit != pool->slot_maps.end() && (size_t)expert_id < smit->second.size()) {
            int32_t s = smit->second[expert_id];
            if (s >= 0 && s < pool->max_slots && pool->slots[s].tensor == tensor && pool->slots[s].expert_id == expert_id) {
                auto & slot = pool->slots[s];
                slot.last_used = ++cache->clock;
                slot.hit_count++;
                slot.access_in_window++;

                if (slot.segment == GGML_EXPERT_CACHE_SEG_PROBATIONARY && slot.access_in_window >= 2) {
                    if (pool->protected_used < pool->protected_cap) {
                        slot.segment = GGML_EXPERT_CACHE_SEG_PROTECTED;
                        pool->probationary_used--;
                        pool->protected_used++;
                    } else {
                        int32_t lru_prot = -1;
                        uint64_t oldest = UINT64_MAX;
                        for (int32_t p = 0; p < pool->max_slots; p++) {
                            if (pool->slots[p].tensor != NULL && pool->slots[p].segment == GGML_EXPERT_CACHE_SEG_PROTECTED) {
                                if (pool->slots[p].last_used < oldest) {
                                    oldest = pool->slots[p].last_used;
                                    lru_prot = p;
                                }
                            }
                        }
                        if (lru_prot >= 0) {
                            pool->slots[lru_prot].segment = GGML_EXPERT_CACHE_SEG_PROBATIONARY;
                            slot.segment = GGML_EXPERT_CACHE_SEG_PROTECTED;
                        }
                    }
                }
            }
        }
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
    cache->stats.n_d2d_fallback_hits++;
    cache->stats.bytes_avoided += bytes_avoided;
    ggml_backend_expert_cache_touch(cache, tensor, expert_id);
}

void ggml_backend_expert_cache_record_zero_copy_hit(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id,
        size_t bytes_avoided) {
    if (cache == NULL) {
        return;
    }
    cache->stats.n_requests++;
    cache->stats.n_hits++;
    cache->stats.n_zero_copy_hits++;
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

void ggml_backend_expert_cache_record_eligible(
        ggml_backend_expert_cache_t cache) {
    if (cache != NULL) {
        cache->stats.n_eligible_ops++;
    }
}

void ggml_backend_expert_cache_record_capacity_bypass(
        ggml_backend_expert_cache_t cache) {
    if (cache != NULL) {
        cache->stats.n_capacity_bypasses++;
    }
}

void ggml_backend_expert_cache_record_cpu_backend_bypass(
        ggml_backend_expert_cache_t cache) {
    if (cache != NULL) {
        cache->stats.n_cpu_backend_bypasses++;
    }
}

void ggml_backend_expert_cache_record_mul_mat_id_input(
        ggml_backend_expert_cache_t cache) {
    if (cache != NULL) {
        cache->stats.n_mul_mat_id_inputs++;
    }
}

void ggml_backend_expert_cache_record_non_host_weight_bypass(
        ggml_backend_expert_cache_t cache) {
    if (cache != NULL) {
        cache->stats.n_non_host_weight_bypasses++;
    }
}

struct ggml_tensor * ggml_backend_expert_cache_get_slot_tensor(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * weight_tensor) {
    auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, weight_tensor);
    return pool != NULL ? pool->tensor : NULL;
}

int32_t ggml_backend_expert_cache_find_slot(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id) {
    if (cache == NULL || tensor == NULL || expert_id < 0) {
        return -1;
    }
    auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, tensor);
    if (pool == NULL) {
        return -1;
    }
    auto it = pool->slot_maps.find(tensor);
    if (it != pool->slot_maps.end() && (size_t) expert_id < it->second.size()) {
        const int32_t slot = it->second[expert_id];
        if (slot >= 0 && slot < pool->max_slots &&
            pool->slots[slot].tensor == tensor &&
            pool->slots[slot].expert_id == expert_id &&
            ggml_expert_cache_poll_slot(cache, pool->slots[slot])) {
            return slot;
        }
    }
    return -1;
}

int32_t ggml_backend_expert_cache_claim_slot_idx(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id,
        const struct ggml_expert_cache_key * pinned_keys,
        size_t n_pinned,
        bool * out_needs_load) {
    if (out_needs_load != nullptr) {
        *out_needs_load = false;
    }
    if (cache == NULL || tensor == NULL || expert_id < 0) {
        return -1;
    }

    auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, tensor);
    if (pool == NULL || pool->max_slots <= 0) {
        return -1;
    }

    auto smit = pool->slot_maps.find(tensor);
    if (smit != pool->slot_maps.end() && (size_t)expert_id < smit->second.size()) {
        int32_t s = smit->second[expert_id];
        if (s >= 0 && s < pool->max_slots && pool->slots[s].tensor == tensor && pool->slots[s].expert_id == expert_id) {
            pool->slots[s].last_used = ++cache->clock;
            return s;
        }
    }

    ggml_expert_cache_key key = { tensor, expert_id };
    const int32_t layer = ggml_expert_cache_get_tensor_layer(tensor);

    // 1. Look for an empty slot (unconstrained pool)
    for (int32_t s = 0; s < pool->max_slots; s++) {
        if (pool->slots[s].tensor == NULL) {
            pool->slots[s] = { tensor, expert_id, layer, GGML_EXPERT_CACHE_SLOT_LOADING, GGML_EXPERT_CACHE_SEG_PROBATIONARY, ++cache->clock, 0, 1, nullptr };
            auto & sm = pool->slot_maps[tensor];
            if ((size_t)expert_id >= sm.size()) {
                sm.resize((size_t)expert_id + 1, -1);
            }
            sm[expert_id] = s;
            pool->used_slots++;
            pool->probationary_used++;
            if (layer >= 0) cache->layer_slots[layer]++;

            pool->eviction_miss_counts.erase(key);
            if (out_needs_load != nullptr) {
                *out_needs_load = true;
            }
            return s;
        }
    }

    // 2. Pool is fully utilized: Apply Admission Hysteresis & Anti-Thrashing Guard (Phase 5)
    auto evit = pool->eviction_miss_counts.find(key);
    if (evit != pool->eviction_miss_counts.end()) {
        evit->second++;
        if (evit->second < 8) {
            return -1;
        }
        pool->eviction_miss_counts.erase(evit);
    }

    auto ghit = pool->ghost_sightings.find(key);
    const uint64_t ghost_window = 128;
    if (ghit == pool->ghost_sightings.end() || (cache->decode_step > ghit->second + ghost_window)) {
        pool->ghost_sightings[key] = cache->decode_step;
        return -1;
    }
    pool->ghost_sightings.erase(ghit);
    int32_t victim_slot = -1;
    uint64_t oldest_clock = UINT64_MAX;

    for (int32_t s = 0; s < pool->max_slots; s++) {
        const auto & slot = pool->slots[s];
        if (slot.tensor == NULL || slot.state == GGML_EXPERT_CACHE_SLOT_LOADING || slot.is_pinned) continue;

        bool is_pinned = false;
        if (pinned_keys != NULL && n_pinned > 0) {
            for (size_t p = 0; p < n_pinned; p++) {
                if (pinned_keys[p].tensor == slot.tensor && pinned_keys[p].expert_id == slot.expert_id) {
                    is_pinned = true;
                    break;
                }
            }
        }
        if (is_pinned) continue;

        if (slot.segment == GGML_EXPERT_CACHE_SEG_PROBATIONARY && slot.last_used < oldest_clock) {
            oldest_clock = slot.last_used;
            victim_slot = s;
        }
    }

    if (victim_slot == -1) {
        for (int32_t s = 0; s < pool->max_slots; s++) {
            const auto & slot = pool->slots[s];
            if (slot.tensor == NULL || slot.state == GGML_EXPERT_CACHE_SLOT_LOADING || slot.is_pinned) continue;

            bool is_pinned = false;
            if (pinned_keys != NULL && n_pinned > 0) {
                for (size_t p = 0; p < n_pinned; p++) {
                    if (pinned_keys[p].tensor == slot.tensor && pinned_keys[p].expert_id == slot.expert_id) {
                        is_pinned = true;
                        break;
                    }
                }
            }
            if (is_pinned) continue;

            if (slot.last_used < oldest_clock) {
                oldest_clock = slot.last_used;
                victim_slot = s;
            }
        }
    }

    if (victim_slot != -1) {
        auto & vict = pool->slots[victim_slot];
        ggml_expert_cache_key vkey = { vict.tensor, vict.expert_id };

        // Synchronize before overwriting slot data.
        ggml_expert_cache_wait_slot(vict);

        if (vict.tensor != NULL) {
            auto vit = pool->slot_maps.find(vict.tensor);
            if (vit != pool->slot_maps.end() && vict.expert_id >= 0 && (size_t)vict.expert_id < vit->second.size()) {
                vit->second[vict.expert_id] = -1;
            }
        }
        if (vict.layer >= 0) cache->layer_slots[vict.layer]--;
        if (vict.segment == GGML_EXPERT_CACHE_SEG_PROBATIONARY) {
            pool->probationary_used--;
        } else {
            pool->protected_used--;
        }

        pool->eviction_miss_counts[vkey] = 1;
        cache->stats.n_evictions++;

        pool->slots[victim_slot] = { tensor, expert_id, layer, GGML_EXPERT_CACHE_SLOT_LOADING, GGML_EXPERT_CACHE_SEG_PROBATIONARY, ++cache->clock, 0, 1, nullptr };
        auto & sm = pool->slot_maps[tensor];
        if ((size_t)expert_id >= sm.size()) {
            sm.resize((size_t)expert_id + 1, -1);
        }
        sm[expert_id] = victim_slot;
        pool->probationary_used++;
        if (layer >= 0) cache->layer_slots[layer]++;
        if (out_needs_load != nullptr) {
            *out_needs_load = true;
        }
        return victim_slot;
    }

    return -1;
}

int32_t ggml_backend_expert_cache_alloc_slot_idx(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id,
        const struct ggml_expert_cache_key * pinned_keys,
        size_t n_pinned) {
    return ggml_backend_expert_cache_claim_slot_idx(
        cache, tensor, expert_id, pinned_keys, n_pinned, nullptr);
}

int32_t ggml_backend_expert_cache_remap_ids(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        const int32_t * original_ids,
        int32_t n_ids,
        int32_t * out_remapped_ids,
        bool * out_is_hit) {
    if (cache == NULL || tensor == NULL || original_ids == NULL || n_ids <= 0) {
        return 0;
    }

    auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, tensor);
    int32_t n_hits = 0;

    const std::vector<int32_t> * p_sm = nullptr;
    if (pool != NULL) {
        auto smit = pool->slot_maps.find(tensor);
        if (smit != pool->slot_maps.end()) {
            p_sm = &smit->second;
        }
    }

    for (int32_t i = 0; i < n_ids; i++) {
        int32_t id = original_ids[i];
        if (id < 0) {
            if (out_remapped_ids) out_remapped_ids[i] = -1;
            if (out_is_hit) out_is_hit[i] = false;
            continue;
        }

        int32_t slot = -1;
        if (pool != NULL && p_sm != nullptr && (size_t)id < p_sm->size()) {
            int32_t s = (*p_sm)[id];
            if (s >= 0 && s < pool->max_slots && pool->slots[s].tensor == tensor && pool->slots[s].expert_id == id && pool->slots[s].state == GGML_EXPERT_CACHE_SLOT_RESIDENT) {
                slot = s;
            }
        }

        if (slot >= 0) {
            if (out_remapped_ids) out_remapped_ids[i] = slot;
            if (out_is_hit) out_is_hit[i] = true;
            ggml_backend_expert_cache_touch(cache, tensor, id);
            n_hits++;
        } else {
            if (out_remapped_ids) out_remapped_ids[i] = id;
            if (out_is_hit) out_is_hit[i] = false;
        }
    }

    return n_hits;
}

void ggml_backend_expert_cache_record_cpu_id_remap(ggml_backend_expert_cache_t cache) {
    if (cache == NULL) return;
    cache->stats.n_cpu_id_remaps++;
}

void ggml_backend_expert_cache_record_staging_memcpy(ggml_backend_expert_cache_t cache, size_t bytes) {
    if (cache == NULL || bytes == 0) return;
    cache->stats.staging_memcpy_bytes += bytes;
}

void ggml_backend_expert_cache_record_direct_dma(ggml_backend_expert_cache_t cache, size_t bytes) {
    if (cache == NULL || bytes == 0) return;
    cache->stats.direct_pinned_dma_bytes += bytes;
}

void ggml_backend_expert_cache_record_all_hit_resolution(ggml_backend_expert_cache_t cache) {
    if (cache == NULL) return;
    cache->stats.n_gpu_id_resolutions++;
}

void ggml_backend_expert_cache_record_probe_layer(ggml_backend_expert_cache_t cache) {
    if (cache == NULL) return;
    cache->stats.probe_n_layers++;
}

void ggml_backend_expert_cache_record_probe_sync(ggml_backend_expert_cache_t cache, uint64_t us) {
    if (cache == NULL) return;
    cache->stats.probe_sync_us += us;
}

void ggml_backend_expert_cache_record_probe_host(ggml_backend_expert_cache_t cache, uint64_t us) {
    if (cache == NULL) return;
    cache->stats.probe_host_us += us;
}

void ggml_backend_expert_cache_record_probe_upload(ggml_backend_expert_cache_t cache, uint64_t us) {
    if (cache == NULL) return;
    cache->stats.probe_upload_us += us;
}

void ggml_backend_expert_cache_record_route_snapshot(ggml_backend_expert_cache_t cache) {
    if (cache != NULL) {
        cache->stats.n_route_prefetch_snapshots++;
    }
}

void ggml_backend_expert_cache_record_route_prefetch_submitted(ggml_backend_expert_cache_t cache) {
    if (cache != NULL) {
        cache->stats.n_route_prefetch_submitted++;
    }
}

void ggml_backend_expert_cache_record_route_prefetch_duplicate(ggml_backend_expert_cache_t cache) {
    if (cache != NULL) {
        cache->stats.n_route_prefetch_duplicates++;
    }
}

void ggml_backend_expert_cache_record_route_prefetch_stale(ggml_backend_expert_cache_t cache) {
    if (cache != NULL) {
        cache->stats.n_route_prefetch_stale++;
    }
}

void ggml_backend_expert_cache_record_route_prefetch_rejected(ggml_backend_expert_cache_t cache) {
    if (cache != NULL) {
        cache->stats.n_route_prefetch_rejected++;
    }
}

bool ggml_backend_expert_cache_register_host_memory(
        ggml_backend_expert_cache_t cache,
        void * ptr,
        size_t size) {
    if (cache == NULL || ptr == NULL || size == 0) {
        return false;
    }

    if (cache->registered_host_bytes + size > cache->max_registered_host_bytes) {
        return false;
    }

#if defined(GGML_USE_CUDA)
    cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
    if (err == cudaSuccess) {
        cache->registered_host_ranges.push_back({(uintptr_t)ptr, size});
        cache->registered_host_bytes += size;
        return true;
    }
#endif
    return false;
}

bool ggml_backend_expert_cache_is_host_memory_registered(
        ggml_backend_expert_cache_t cache,
        const void * ptr,
        size_t size) {
    if (cache == NULL || ptr == NULL || size == 0) {
        return false;
    }
    uintptr_t p = (uintptr_t)ptr;
    for (const auto & range : cache->registered_host_ranges) {
        if (p >= range.first && (p + size) <= (range.first + range.second)) {
            return true;
        }
    }
    return false;
}

size_t ggml_backend_expert_cache_alloc_slot(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id,
        size_t size,
        const struct ggml_expert_cache_key * pinned_keys,
        size_t n_pinned) {
    GGML_UNUSED(size);
    int32_t slot = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, expert_id, pinned_keys, n_pinned);
    if (slot >= 0) {
        auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, tensor);
        if (pool != NULL) {
            return pool->buffer_offset + (size_t)slot * pool->stride;
        }
    }
    return SIZE_MAX;
}

void ggml_backend_expert_cache_register_bundle(
        ggml_backend_expert_cache_t cache,
        int32_t layer,
        const struct ggml_tensor * gate_tensor,
        const struct ggml_tensor * up_tensor,
        const struct ggml_tensor * down_tensor) {
    if (cache == NULL || layer < 0) {
        return;
    }
    cache->bundle_registrations[layer] = { gate_tensor, up_tensor, down_tensor };

    if (cache->is_static_manifest_loaded) {
        for (const auto & entry : cache->pinned_bundles) {
            if (entry.first == layer) {
                int32_t expert_id = entry.second;
                if (gate_tensor && gate_tensor->data != NULL) {
                    ggml_backend_expert_cache_seed(cache, gate_tensor, expert_id, 1000);
                    int32_t s = ggml_backend_expert_cache_find_slot(cache, gate_tensor, expert_id);
                    if (s >= 0) {
                        auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, gate_tensor);
                        if (pool && s < pool->max_slots) pool->slots[s].is_pinned = true;
                    }
                }
                if (up_tensor && up_tensor->data != NULL) {
                    ggml_backend_expert_cache_seed(cache, up_tensor, expert_id, 1000);
                    int32_t s = ggml_backend_expert_cache_find_slot(cache, up_tensor, expert_id);
                    if (s >= 0) {
                        auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, up_tensor);
                        if (pool && s < pool->max_slots) pool->slots[s].is_pinned = true;
                    }
                }
                if (down_tensor && down_tensor->data != NULL) {
                    ggml_backend_expert_cache_seed(cache, down_tensor, expert_id, 1000);
                    int32_t s = ggml_backend_expert_cache_find_slot(cache, down_tensor, expert_id);
                    if (s >= 0) {
                        auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, down_tensor);
                        if (pool && s < pool->max_slots) pool->slots[s].is_pinned = true;
                    }
                }
            }
        }
    }
}

bool ggml_backend_expert_cache_is_bundle_resident(
        ggml_backend_expert_cache_t cache,
        int32_t layer,
        int32_t expert_id) {
    if (cache == NULL || layer < 0 || expert_id < 0) {
        return false;
    }

    auto bit = cache->bundle_registrations.find(layer);
    if (bit == cache->bundle_registrations.end()) {
        return false;
    }

    const auto & reg = bit->second;
    if (reg.gate != NULL && ggml_backend_expert_cache_find_slot(cache, reg.gate, expert_id) < 0) {
        return false;
    }
    if (reg.up != NULL && ggml_backend_expert_cache_find_slot(cache, reg.up, expert_id) < 0) {
        return false;
    }
    if (reg.down != NULL && ggml_backend_expert_cache_find_slot(cache, reg.down, expert_id) < 0) {
        return false;
    }
    return true;
}

void * ggml_backend_expert_cache_get_pinned_buffer(
        ggml_backend_expert_cache_t cache,
        size_t required_size) {
    if (cache == NULL || required_size == 0) {
        return NULL;
    }

    if (cache->pinned_host_capacity < required_size) {
        // Synchronize all active staging events before freeing/resizing arena
        for (size_t i = 0; i < cache->staging_events.size(); i++) {
            if (cache->staging_in_flight[i] && cache->staging_events[i]) {
                ggml_backend_event_synchronize(cache->staging_events[i]);
                cache->staging_in_flight[i] = false;
            }
        }

        if (cache->pinned_host_buffer) {
            if (cache->pinned_is_cuda) {
#if defined(GGML_USE_CUDA)
                cudaFreeHost(cache->pinned_host_buffer);
#endif
            } else {
#if defined(_WIN32)
                _aligned_free(cache->pinned_host_buffer);
#else
                free(cache->pinned_host_buffer);
#endif
            }
            cache->pinned_host_buffer = nullptr;
            cache->pinned_is_cuda = false;
        }
        cache->pinned_host_capacity = GGML_EXPERT_CACHE_PAD(required_size);

#if defined(GGML_USE_CUDA)
        cudaError_t err = cudaHostAlloc(&cache->pinned_host_buffer, cache->pinned_host_capacity, cudaHostAllocPortable | cudaHostAllocWriteCombined);
        if (err == cudaSuccess) {
            cache->pinned_is_cuda = true;
        } else {
            cache->pinned_host_buffer = nullptr;
            cache->pinned_is_cuda = false;
#if defined(_WIN32)
            cache->pinned_host_buffer = _aligned_malloc(cache->pinned_host_capacity, GGML_EXPERT_CACHE_ALIGN);
#else
            posix_memalign(&cache->pinned_host_buffer, GGML_EXPERT_CACHE_ALIGN, cache->pinned_host_capacity);
#endif
        }
#elif defined(_WIN32)
        cache->pinned_host_buffer = _aligned_malloc(cache->pinned_host_capacity, GGML_EXPERT_CACHE_ALIGN);
        cache->pinned_is_cuda = false;
#else
        posix_memalign(&cache->pinned_host_buffer, GGML_EXPERT_CACHE_ALIGN, cache->pinned_host_capacity);
        cache->pinned_is_cuda = false;
#endif
    }
    return cache->pinned_host_buffer;
}

static size_t ggml_expert_cache_staging_index(
        const struct ggml_tensor * tensor,
        int32_t slot_idx) {
    const uint64_t h = std::hash<const void *>()(tensor) ^
                       ((uint64_t)(uint32_t) slot_idx * 2654435761ull);
    return (size_t)(h & (GGML_EXPERT_CACHE_STAGING_ENTRIES - 1));
}

void * ggml_backend_expert_cache_stage_acquire(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t slot_idx,
        size_t required_size) {
    if (cache == NULL || tensor == NULL || required_size == 0) {
        return NULL;
    }

    const size_t idx = ggml_expert_cache_staging_index(tensor, slot_idx);

    if (cache->staging_events.empty()) {
        cache->staging_events.resize(GGML_EXPERT_CACHE_STAGING_ENTRIES, NULL);
        cache->staging_in_flight.assign(GGML_EXPERT_CACHE_STAGING_ENTRIES, false);
        ggml_backend_dev_t dev = ggml_backend_get_device(cache->backend);
        for (size_t i = 0; i < GGML_EXPERT_CACHE_STAGING_ENTRIES; i++) {
            cache->staging_events[i] = ggml_backend_event_new(dev);
        }
    }

    if (cache->staging_in_flight[idx]) {
        if (cache->staging_events[idx] != NULL) {
            ggml_backend_event_synchronize(cache->staging_events[idx]);
        } else {
            ggml_backend_synchronize(cache->backend);
        }
        cache->staging_in_flight[idx] = false;
        cache->stats.n_staging_waits++;
    }

    const size_t slot_stride = GGML_EXPERT_CACHE_PAD(required_size);
    const size_t total_capacity = slot_stride * GGML_EXPERT_CACHE_STAGING_ENTRIES;
    void * base = ggml_backend_expert_cache_get_pinned_buffer(cache, total_capacity);
    if (base == NULL) {
        return NULL;
    }

    return (uint8_t *) base + idx * slot_stride;
}

void ggml_backend_expert_cache_stage_commit(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t slot_idx) {
    if (cache == NULL || tensor == NULL || cache->staging_events.empty()) {
        return;
    }
    const size_t idx = ggml_expert_cache_staging_index(tensor, slot_idx);
    if (cache->staging_events[idx] != NULL) {
        ggml_backend_event_record(cache->staging_events[idx], cache->backend);
    }
    cache->staging_in_flight[idx] = true;
}

void ggml_backend_expert_cache_prefetch(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        const int32_t * expert_ids,
        int32_t n_experts) {
    if (cache == NULL || tensor == NULL || expert_ids == NULL || n_experts <= 0) {
        return;
    }
    auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, tensor);
    if (pool == NULL || tensor->data == NULL) {
        return;
    }
    const size_t expert_size = pool->stride;
    std::vector<int32_t> unique_ids;
    unique_ids.reserve((size_t)n_experts);
    for (int32_t i = 0; i < n_experts; i++) {
        const int32_t eid = expert_ids[i];
        if (eid < 0 || (int64_t)eid >= tensor->ne[2]) continue;
        if (std::find(unique_ids.begin(), unique_ids.end(), eid) == unique_ids.end()) {
            unique_ids.push_back(eid);
        }
    }

    for (const int32_t eid : unique_ids) {
        bool needs_load = false;
        const int32_t slot = ggml_backend_expert_cache_claim_slot_idx(
            cache, tensor, eid, NULL, 0, &needs_load);
        if (slot < 0 || !needs_load) {
            continue;
        }

        const size_t src_off = (size_t)eid * expert_size;
        const size_t dst_off = (size_t)slot * expert_size;

        void * pinned_buf = ggml_backend_expert_cache_stage_acquire(cache, tensor, slot, expert_size);
        if (pinned_buf != NULL) {
            memcpy(pinned_buf, (const uint8_t *) tensor->data + src_off, expert_size);
            ggml_backend_tensor_set_async(
                cache->backend,
                pool->tensor,
                pinned_buf,
                dst_off,
                expert_size);
            ggml_backend_expert_cache_stage_commit(cache, tensor, slot);
        } else {
            ggml_backend_tensor_set_async(
                cache->backend,
                pool->tensor,
                (const uint8_t *) tensor->data + src_off,
                dst_off,
                expert_size);
        }
        ggml_backend_expert_cache_promote_slot(cache, tensor, eid, slot);
        cache->stats.n_speculative_prefetches++;
    }
}

int32_t ggml_backend_expert_cache_get_tensor_layer(const struct ggml_tensor * tensor) {
    return ggml_expert_cache_get_tensor_layer(tensor);
}

void ggml_backend_expert_cache_prefetch_layer(
        ggml_backend_expert_cache_t cache,
        int32_t layer,
        const int32_t * expert_ids,
        int32_t n_experts) {
    if (cache == NULL || layer < 0 || expert_ids == NULL || n_experts <= 0) {
        return;
    }
    auto bit = cache->bundle_registrations.find(layer);
    if (bit != cache->bundle_registrations.end()) {
        if (bit->second.gate) ggml_backend_expert_cache_prefetch(cache, bit->second.gate, expert_ids, n_experts);
        if (bit->second.up)   ggml_backend_expert_cache_prefetch(cache, bit->second.up,   expert_ids, n_experts);
        if (bit->second.down) ggml_backend_expert_cache_prefetch(cache, bit->second.down, expert_ids, n_experts);
    }
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

    size_t total_used_bytes = 0;
    size_t total_entries = 0;
    for (const auto & pkv : cache->slot_pools) {
        total_used_bytes += (size_t)pkv.second.used_slots * pkv.second.stride;
        total_entries += (size_t)pkv.second.used_slots;
    }

    const double hit_rate = cache->stats.n_requests > 0 ?
        (100.0 * (double)cache->stats.n_hits / (double)cache->stats.n_requests) : 0.0;
    const double cap_mib = (double)cache->capacity / (1024.0 * 1024.0);
    const double used_mib = (double)total_used_bytes / (1024.0 * 1024.0);
    const double ram_to_gpu_gib = (double)cache->stats.bytes_ram_to_gpu / (1024.0 * 1024.0 * 1024.0);
    const double avoided_gib = (double)cache->stats.bytes_avoided / (1024.0 * 1024.0 * 1024.0);

    GGML_LOG_INFO("\n");
    GGML_LOG_INFO("Expert Cache (%s):\n", ggml_backend_name(cache->backend));
    GGML_LOG_INFO("  capacity:             %8.2f MiB\n", cap_mib);
    GGML_LOG_INFO("  resident:             %8.2f MiB (%zu slots used across %zu pools)\n",
        used_mib, total_entries, cache->slot_pools.size());
    GGML_LOG_INFO("  period:               %d tokens\n", cache->period_tokens);
    GGML_LOG_INFO("  rebalances:           %" PRIu64 "\n", cache->stats.n_rebalances);
    GGML_LOG_INFO("  requests:             %" PRIu64 "\n", cache->stats.n_requests);
    GGML_LOG_INFO("  CPU backend bypasses: %" PRIu64 "\n", cache->stats.n_cpu_backend_bypasses);
    GGML_LOG_INFO("  MUL_MAT_ID inputs:    %" PRIu64 "\n", cache->stats.n_mul_mat_id_inputs);
    GGML_LOG_INFO("  non-host bypasses:    %" PRIu64 "\n", cache->stats.n_non_host_weight_bypasses);
    GGML_LOG_INFO("  hits (total):         %" PRIu64 "\n", cache->stats.n_hits);
    GGML_LOG_INFO("    zero-copy hits:     %" PRIu64 "\n", cache->stats.n_zero_copy_hits);
    GGML_LOG_INFO("    D2D fallback hits:  %" PRIu64 "\n", cache->stats.n_d2d_fallback_hits);
    GGML_LOG_INFO("  speculative prefetch: %" PRIu64 "\n", cache->stats.n_speculative_prefetches);
    GGML_LOG_INFO("  misses:               %" PRIu64 "\n", cache->stats.n_misses);
    GGML_LOG_INFO("  eligible ops:         %" PRIu64 "\n", cache->stats.n_eligible_ops);
    GGML_LOG_INFO("  capacity bypasses:    %" PRIu64 "\n", cache->stats.n_capacity_bypasses);
    GGML_LOG_INFO("  hit rate:             %8.2f %%\n", hit_rate);
    GGML_LOG_INFO("  RAM -> GPU:           %8.2f GiB\n", ram_to_gpu_gib);
    GGML_LOG_INFO("  avoided RAM -> GPU:   %8.2f GiB\n", avoided_gib);
    GGML_LOG_INFO("  evictions:            %" PRIu64 "\n", cache->stats.n_evictions);
    GGML_LOG_INFO("  CPU ID remaps:        %" PRIu64 "\n", cache->stats.n_cpu_id_remaps);
    GGML_LOG_INFO("  all-hit resolutions:  %" PRIu64 "\n", cache->stats.n_gpu_id_resolutions);
    GGML_LOG_INFO("  staging memcpy bytes: %zu (%8.2f MiB)\n", cache->stats.staging_memcpy_bytes, (double)cache->stats.staging_memcpy_bytes / (1024.0 * 1024.0));
    GGML_LOG_INFO("  direct DMA bytes:     %zu (%8.2f MiB)\n", cache->stats.direct_pinned_dma_bytes, (double)cache->stats.direct_pinned_dma_bytes / (1024.0 * 1024.0));
    GGML_LOG_INFO("  staging waits:        %" PRIu64 "\n", cache->stats.n_staging_waits);
    GGML_LOG_INFO("  route snapshots:      %" PRIu64 "\n", cache->stats.n_route_prefetch_snapshots);
    GGML_LOG_INFO("  route prefetches:     %" PRIu64 "\n", cache->stats.n_route_prefetch_submitted);
    GGML_LOG_INFO("  route duplicates:     %" PRIu64 "\n", cache->stats.n_route_prefetch_duplicates);
    GGML_LOG_INFO("  route stale:          %" PRIu64 "\n", cache->stats.n_route_prefetch_stale);
    GGML_LOG_INFO("  route rejected:       %" PRIu64 "\n", cache->stats.n_route_prefetch_rejected);
}

size_t ggml_backend_expert_cache_export_entries(
        ggml_backend_expert_cache_t cache,
        struct ggml_backend_expert_cache_export_entry * out_entries,
        size_t max_entries) {
    if (cache == NULL || out_entries == NULL || max_entries == 0) {
        return 0;
    }

    size_t count = 0;
    for (const auto & kv : cache->tg_access_freq) {
        if (count >= max_entries) break;
        if (kv.first.tensor == NULL) continue;

        uint64_t hits = 0;
        int32_t slot = ggml_backend_expert_cache_find_slot(cache, kv.first.tensor, kv.first.expert_id);
        if (slot >= 0) {
            auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, kv.first.tensor);
            if (pool && slot < (int32_t)pool->slots.size()) {
                hits = pool->slots[slot].hit_count;
            }
        }

        out_entries[count].tensor    = kv.first.tensor;
        out_entries[count].expert_id = kv.first.expert_id;
        out_entries[count].frequency = kv.second;
        out_entries[count].hit_count = hits;
        count++;
    }

    for (const auto & pkv : cache->slot_pools) {
        for (const auto & slot : pkv.second.slots) {
            if (count >= max_entries) break;
            if (slot.tensor == NULL) continue;

            ggml_expert_cache_key key = { slot.tensor, slot.expert_id };
            if (cache->tg_access_freq.find(key) == cache->tg_access_freq.end()) {
                out_entries[count].tensor    = slot.tensor;
                out_entries[count].expert_id = slot.expert_id;
                out_entries[count].frequency = 0;
                out_entries[count].hit_count = slot.hit_count;
                count++;
            }
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
    cache->tg_access_freq[key] = std::max(cache->tg_access_freq[key], frequency);

    bool needs_load = false;
    const int32_t slot = ggml_backend_expert_cache_claim_slot_idx(
        cache, tensor, expert_id, NULL, 0, &needs_load);
    if (!needs_load) {
        return slot >= 0;
    }

    if (slot < 0) {
        return false;
    }

    struct ggml_tensor * slot_tensor = ggml_backend_expert_cache_get_slot_tensor(cache, tensor);
    if (slot_tensor == NULL) {
        return false;
    }

    const size_t expert_size = tensor->nb[2];
    const size_t src_off = (size_t)expert_id * expert_size;
    const size_t dst_off = (size_t)slot * expert_size;

    ggml_backend_tensor_set_async(
        cache->backend,
        slot_tensor,
        (const uint8_t *)tensor->data + src_off,
        dst_off,
        expert_size);

    ggml_backend_expert_cache_promote_slot(cache, tensor, expert_id, slot);

    return true;
}

void ggml_backend_expert_cache_sync_staging(ggml_backend_expert_cache_t cache) {
    if (cache == NULL) {
        return;
    }
    for (size_t i = 0; i < cache->staging_events.size(); i++) {
        if (cache->staging_in_flight[i] && cache->staging_events[i] != NULL) {
            ggml_backend_event_synchronize(cache->staging_events[i]);
            cache->staging_in_flight[i] = false;
        }
    }
}

void ggml_backend_expert_cache_promote_slot(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id,
        int32_t slot) {
    if (cache == NULL || tensor == NULL || expert_id < 0 || slot < 0) {
        return;
    }
    auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, tensor);
    if (pool == NULL || slot >= pool->max_slots) {
        return;
    }
    auto & slot_entry = pool->slots[slot];
    if (slot_entry.tensor == tensor && slot_entry.expert_id == expert_id &&
        slot_entry.state == GGML_EXPERT_CACHE_SLOT_LOADING) {
        if (slot_entry.load_event == nullptr) {
            slot_entry.load_event = ggml_backend_event_new(ggml_backend_get_device(cache->backend));
        }
        if (slot_entry.load_event != nullptr) {
            ggml_backend_event_record(slot_entry.load_event, cache->backend);
            cache->async_promotions.push_back({ tensor, slot_entry.layer, expert_id, slot });
        } else if (ggml_backend_dev_type(ggml_backend_get_device(cache->backend)) ==
                GGML_BACKEND_DEVICE_TYPE_CPU) {
            slot_entry.state = GGML_EXPERT_CACHE_SLOT_RESIDENT;
            if (slot_entry.layer >= 0 && slot_entry.layer < 128 && expert_id >= 0 && expert_id < 512) {
                cache->gpu_slot_map_table[slot_entry.layer * 512 + expert_id] = slot;
            }
        }
    }
}
bool ggml_backend_expert_cache_record_slot_use(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id,
        int32_t slot) {
    if (cache == NULL || tensor == NULL || expert_id < 0 || slot < 0) {
        return false;
    }

    auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, tensor);
    if (pool == NULL || slot >= pool->max_slots) {
        return false;
    }

    auto & slot_entry = pool->slots[slot];
    if (slot_entry.tensor != tensor || slot_entry.expert_id != expert_id ||
        !ggml_expert_cache_poll_slot(cache, slot_entry)) {
        return false;
    }

    if (slot_entry.last_use_event == nullptr) {
        slot_entry.last_use_event = ggml_backend_event_new(ggml_backend_get_device(cache->backend));
    }
    if (slot_entry.last_use_event == nullptr) {
        return false;
    }

    ggml_backend_event_record(slot_entry.last_use_event, cache->backend);
    return true;
}



void ggml_backend_expert_cache_sync(ggml_backend_expert_cache_t cache) {
    if (cache == NULL) {
        return;
    }
    ggml_backend_synchronize(cache->backend);
}

bool ggml_backend_expert_cache_load_pinned_manifest(
        ggml_backend_expert_cache_t cache,
        const char * manifest_json_path) {
    if (cache == NULL || manifest_json_path == NULL || manifest_json_path[0] == '\0') {
        return false;
    }

    std::ifstream ifs(manifest_json_path);
    if (!ifs.is_open()) {
        fprintf(stderr, "%s: failed to open manifest '%s'\n", __func__, manifest_json_path);
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
    ifs.close();

    cache->pinned_bundles.clear();

    size_t pos = 0;
    while ((pos = content.find("\"layer\"", pos)) != std::string::npos) {
        size_t colon = content.find(':', pos);
        if (colon == std::string::npos) break;
        int layer = atoi(content.c_str() + colon + 1);

        size_t exp_pos = content.find("\"expert_id\"", colon);
        if (exp_pos == std::string::npos) break;
        size_t exp_colon = content.find(':', exp_pos);
        if (exp_colon == std::string::npos) break;
        int expert_id = atoi(content.c_str() + exp_colon + 1);

        cache->pinned_bundles.insert({ layer, expert_id });
        pos = exp_colon + 1;
    }

    cache->is_static_manifest_loaded = true;
    cache->zero_miss_upload = true;

    fprintf(stderr, "%s: loaded %zu static pinned experts from '%s'\n",
        __func__, cache->pinned_bundles.size(), manifest_json_path);

    for (const auto & entry : cache->pinned_bundles) {
        int32_t layer = entry.first;
        int32_t expert_id = entry.second;

        auto reg_it = cache->bundle_registrations.find(layer);
        if (reg_it != cache->bundle_registrations.end()) {
            if (reg_it->second.gate && reg_it->second.gate->data != NULL) {
                ggml_backend_expert_cache_seed(cache, reg_it->second.gate, expert_id, 1000);
                int32_t s = ggml_backend_expert_cache_find_slot(cache, reg_it->second.gate, expert_id);
                if (s >= 0) {
                    auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, reg_it->second.gate);
                    if (pool && s < pool->max_slots) pool->slots[s].is_pinned = true;
                }
            }
            if (reg_it->second.up && reg_it->second.up->data != NULL) {
                ggml_backend_expert_cache_seed(cache, reg_it->second.up, expert_id, 1000);
                int32_t s = ggml_backend_expert_cache_find_slot(cache, reg_it->second.up, expert_id);
                if (s >= 0) {
                    auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, reg_it->second.up);
                    if (pool && s < pool->max_slots) pool->slots[s].is_pinned = true;
                }
            }
            if (reg_it->second.down && reg_it->second.down->data != NULL) {
                ggml_backend_expert_cache_seed(cache, reg_it->second.down, expert_id, 1000);
                int32_t s = ggml_backend_expert_cache_find_slot(cache, reg_it->second.down, expert_id);
                if (s >= 0) {
                    auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, reg_it->second.down);
                    if (pool && s < pool->max_slots) pool->slots[s].is_pinned = true;
                }
            }
        }
    }

    return true;
}

bool ggml_backend_expert_cache_is_pinned(
        ggml_backend_expert_cache_t cache,
        int32_t layer,
        int32_t expert_id) {
    if (cache == NULL) return false;
    return cache->pinned_bundles.count({ layer, expert_id }) > 0;
}

void ggml_backend_expert_cache_set_zero_miss_upload(
        ggml_backend_expert_cache_t cache,
        bool enabled) {
    if (cache != NULL) {
        cache->zero_miss_upload = enabled;
    }
}

bool ggml_backend_expert_cache_get_zero_miss_upload(
        ggml_backend_expert_cache_t cache) {
    return cache != NULL ? cache->zero_miss_upload : true;
}

size_t ggml_backend_expert_cache_process_async_promotions(
        ggml_backend_expert_cache_t cache,
        size_t max_promotions) {
    if (cache == NULL || cache->async_promotions.empty()) {
        return 0;
    }

    size_t promoted_count = 0;
    auto it = cache->async_promotions.begin();
    while (it != cache->async_promotions.end()) {
        if (max_promotions > 0 && promoted_count >= max_promotions) {
            break;
        }

        auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, it->tensor);
        if (pool && it->slot >= 0 && it->slot < pool->max_slots) {
            auto & s = pool->slots[it->slot];
            if (s.tensor == it->tensor && s.expert_id == it->expert_id) {
                if (ggml_expert_cache_poll_slot(cache, s)) {
                    it = cache->async_promotions.erase(it);
                    promoted_count++;
                    continue;
                }
            } else {
                it = cache->async_promotions.erase(it);
                continue;
            }
        } else {
            it = cache->async_promotions.erase(it);
            continue;
        }
        ++it;
    }

    return promoted_count;
}

void ggml_backend_expert_cache_set_max_async_promotions(
        ggml_backend_expert_cache_t cache,
        int32_t max_promotions) {
    if (cache != NULL) {
        cache->max_async_promotions = max_promotions;
    }
}

const int32_t * ggml_backend_expert_cache_get_gpu_slot_map(
        ggml_backend_expert_cache_t cache,
        int32_t layer) {
    if (cache == NULL || layer < 0 || layer >= 128 || cache->gpu_slot_map_table.empty()) {
        return NULL;
    }
    return cache->gpu_slot_map_table.data() + (size_t)layer * 512;
}
