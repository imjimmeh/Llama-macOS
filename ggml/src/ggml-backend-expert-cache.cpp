#include "ggml-backend-expert-cache.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#if defined(GGML_USE_CUDA)
#include <cuda_runtime.h>
#endif

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
    enum ggml_expert_cache_segment segment = GGML_EXPERT_CACHE_SEG_PROBATIONARY;
    uint64_t last_used = 0;
    uint64_t hit_count = 0;
    uint32_t access_in_window = 0;
};

struct ggml_expert_cache_slot_pool {
    ggml_backend_buffer_t buffer = nullptr;
    struct ggml_context * ctx = nullptr;
    struct ggml_tensor *  tensor = nullptr; // 3D tensor: [ne0, ne1, max_slots]
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    enum ggml_type type = GGML_TYPE_F32;
    size_t stride = 0;
    int32_t max_slots = 0;
    int32_t used_slots = 0;

    int32_t probationary_cap = 0;
    int32_t protected_cap = 0;
    int32_t probationary_used = 0;
    int32_t protected_used = 0;


    std::vector<ggml_expert_cache_slot_entry> slots;
    std::unordered_map<ggml_expert_cache_key, int32_t, ggml_expert_cache_key_hash, ggml_expert_cache_key_eq> key_to_slot;
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

struct ggml_expert_bundle_reg {
    const struct ggml_tensor * gate = nullptr;
    const struct ggml_tensor * up   = nullptr;
    const struct ggml_tensor * down = nullptr;
};


struct ggml_backend_expert_cache {
    ggml_backend_t backend;

    // Legacy flat 1D backing buffer
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

    // Phase 1: Slot Pools
    size_t pool_alloc_offset = 0;
    std::unordered_map<ggml_expert_cache_pool_key, ggml_expert_cache_slot_pool, ggml_expert_cache_pool_key_hash> slot_pools;

    // Phase 3: Expert Bundles
    std::unordered_map<int32_t, ggml_expert_bundle_reg> bundle_registrations;

    // Phase 4: Per-layer budget tracking
    std::unordered_map<int32_t, int32_t> layer_slots;

    // Phase 5: Host Pinned Staging Buffer
    void * pinned_host_buffer = nullptr;
    size_t pinned_host_capacity = 0;
    // Phase 5b: staging ring keyed by (tensor, slot_idx); waits before reuse
    std::vector<ggml_backend_event_t> staging_events;
    std::vector<bool>                 staging_in_flight;



    // Phase 7: Host Registered Direct DMA
    std::vector<std::pair<uintptr_t, size_t>> registered_host_ranges;
    size_t registered_host_bytes = 0;
    size_t max_registered_host_bytes = 1024 * 1024 * 1024; // 1 GiB safety cap

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

    ggml_expert_cache_pool_key pkey = { ne0, ne1, type, stride };
    auto it = cache->slot_pools.find(pkey);
    if (it != cache->slot_pools.end()) {
        return &it->second;
    }

    // Proportional slots: share total capacity across potential tensor types (gate/up and down)
    const size_t pool_cap = cache->capacity >= 2 * stride ? cache->capacity / 2 : cache->capacity;
    int32_t max_slots = (int32_t)(pool_cap / stride);
    if (max_slots <= 0) {
        max_slots = 1;
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

    const size_t pool_bytes = max_slots * stride;
    ggml_backend_buffer_t pbuffer = NULL;

    if (cache->used == 0 && cache->pool_alloc_offset + pool_bytes <= cache->capacity && cache->tensor != NULL && cache->tensor->data != NULL) {
        ptensor->data = (uint8_t *)cache->tensor->data + cache->pool_alloc_offset;
        ptensor->buffer = cache->buffer;
        cache->pool_alloc_offset += pool_bytes;
    } else {
        pbuffer = ggml_backend_alloc_buffer(cache->backend, pool_bytes);
        if (pbuffer != NULL) {
            if (ggml_backend_tensor_alloc(pbuffer, ptensor, ggml_backend_buffer_get_base(pbuffer)) != GGML_STATUS_SUCCESS) {
                ggml_free(pctx);
                ggml_backend_buffer_free(pbuffer);
                return NULL;
            }
        } else {
            ptensor->data = cache->tensor->data;
            ptensor->buffer = cache->buffer;
        }
    }

    ptensor->nb[0] = weight_tensor->nb[0];
    ptensor->nb[1] = weight_tensor->nb[1];
    ptensor->nb[2] = stride;
    ptensor->nb[3] = stride * max_slots;

    const int32_t n_experts = weight_tensor->ne[2] > 0 ? (int32_t)weight_tensor->ne[2] : 64;

    ggml_expert_cache_slot_pool pool;
    pool.buffer = (pbuffer != cache->buffer) ? pbuffer : NULL;
    pool.ctx = pctx;
    pool.tensor = ptensor;
    pool.ne0 = ne0;
    pool.ne1 = ne1;
    pool.type = type;
    pool.stride = stride;
    pool.max_slots = max_slots;
    pool.used_slots = 0;
    pool.probationary_cap = std::max(1, (int32_t)(max_slots * 0.20));
    pool.protected_cap = max_slots - pool.probationary_cap;
    pool.probationary_used = 0;
    pool.protected_used = 0;
    pool.slots.resize(max_slots);


    cache->slot_pools[pkey] = pool;
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
bool ggml_backend_expert_cache_can_store(
        ggml_backend_expert_cache_t cache,
        size_t expert_size) {
    return cache != NULL && expert_size > 0 && expert_size <= cache->capacity;
}


void ggml_backend_expert_cache_free(ggml_backend_expert_cache_t cache) {
    if (cache == NULL) {
        return;
    }

    for (auto & kv : cache->slot_pools) {
        if (kv.second.ctx) {
            ggml_free(kv.second.ctx);
        }
        if (kv.second.buffer) {
            ggml_backend_buffer_free(kv.second.buffer);
        }
    }
    cache->slot_pools.clear();

    if (cache->pinned_host_buffer) {
#if defined(_WIN32)
        _aligned_free(cache->pinned_host_buffer);
#else
        free(cache->pinned_host_buffer);
#endif
        cache->pinned_host_buffer = nullptr;
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
        double value; // Value-per-byte score
    };
    std::vector<candidate> candidates;
    candidates.reserve(cache->access_freq.size());

    for (const auto & kv : cache->access_freq) {
        if (kv.second > 0 && kv.first.tensor != NULL) {
            const size_t expert_size = kv.first.tensor->nb[2];
            const size_t alloc_size = GGML_EXPERT_CACHE_PAD(expert_size);
            const double value = alloc_size > 0 ? ((double)kv.second * (double)expert_size) / (double)alloc_size : 0.0;
            candidates.push_back({ kv.first, kv.second, expert_size, alloc_size, value });
        }
    }

    if (candidates.empty()) {
        return;
    }

    std::sort(candidates.begin(), candidates.end(), [](const candidate & a, const candidate & b) {
        if (a.value != b.value) {
            return a.value > b.value;
        }
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
            kv.second = (kv.second * 7) >> 3; // smooth 0.875 decay
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
        kv.second = (kv.second * 7) >> 3; // smooth 0.875 decay
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
    if (cache == NULL || tensor == NULL) {
        return;
    }

    ggml_expert_cache_key key = { tensor, expert_id };
    auto it = cache->entries.find(key);
    if (it != cache->entries.end()) {
        it->second.last_used = ++cache->clock;
        it->second.hit_count++;
    }

    auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, tensor);
    if (pool != NULL) {
        auto sit = pool->key_to_slot.find(key);
        if (sit != pool->key_to_slot.end()) {
            int32_t s = sit->second;
            auto & slot = pool->slots[s];
            slot.last_used = ++cache->clock;
            slot.hit_count++;
            slot.access_in_window++;

            // SLRU Promotion: Promote from probationary to protected on second hit
            if (slot.segment == GGML_EXPERT_CACHE_SEG_PROBATIONARY && slot.access_in_window >= 2) {
                if (pool->protected_used < pool->protected_cap) {
                    slot.segment = GGML_EXPERT_CACHE_SEG_PROTECTED;
                    pool->probationary_used--;
                    pool->protected_used++;
                } else {
                    // Demote LRU protected to probationary to make room
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




// Phase 1: Slot Pools & ID Remapping
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
    if (cache == NULL || tensor == NULL) {
        return -1;
    }
    auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, tensor);
    if (pool == NULL) {
        return -1;
    }

    ggml_expert_cache_key key = { tensor, expert_id };
    auto it = pool->key_to_slot.find(key);
    if (it != pool->key_to_slot.end()) {
        return it->second;
    }
    return -1;
}

int32_t ggml_backend_expert_cache_alloc_slot_idx(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id,
        const struct ggml_expert_cache_key * pinned_keys,
        size_t n_pinned) {
    if (cache == NULL || tensor == NULL) {
        return -1;
    }

    auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, tensor);
    if (pool == NULL) {
        return -1;
    }

    ggml_expert_cache_key key = { tensor, expert_id };
    auto it = pool->key_to_slot.find(key);
    if (it != pool->key_to_slot.end()) {
        pool->slots[it->second].last_used = ++cache->clock;
        return it->second;
    }

    const int32_t layer = ggml_expert_cache_get_tensor_layer(tensor);

    // 1. Look for an empty slot
    for (int32_t s = 0; s < pool->max_slots; s++) {
        if (pool->slots[s].tensor == NULL) {
            pool->slots[s] = { tensor, expert_id, layer, GGML_EXPERT_CACHE_SEG_PROBATIONARY, ++cache->clock, 0, 1 };
            pool->key_to_slot[key] = s;
            pool->used_slots++;
            pool->probationary_used++;
            if (layer >= 0) cache->layer_slots[layer]++;

            return s;
        }
    }

    // 2. Evict an unpinned slot via SLRU + per-layer budget preference
    int32_t victim_slot = -1;
    uint64_t oldest_clock = UINT64_MAX;

    // Prefer evicting from probationary segment first
    for (int32_t s = 0; s < pool->max_slots; s++) {
        const auto & slot = pool->slots[s];
        if (slot.tensor == NULL) continue;

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

    // If all probationary are pinned, pick oldest protected
    if (victim_slot == -1) {
        for (int32_t s = 0; s < pool->max_slots; s++) {
            const auto & slot = pool->slots[s];
            if (slot.tensor == NULL) continue;

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
        const auto & vict = pool->slots[victim_slot];
        ggml_expert_cache_key vkey = { vict.tensor, vict.expert_id };
        if (vict.layer >= 0) cache->layer_slots[vict.layer]--;
        if (vict.segment == GGML_EXPERT_CACHE_SEG_PROBATIONARY) {
            pool->probationary_used--;
        } else {
            pool->protected_used--;
        }


        pool->key_to_slot.erase(vkey);
        cache->stats.n_evictions++;

        pool->slots[victim_slot] = { tensor, expert_id, layer, GGML_EXPERT_CACHE_SEG_PROBATIONARY, ++cache->clock, 0, 1 };
        pool->key_to_slot[key] = victim_slot;
        pool->probationary_used++;
        if (layer >= 0) cache->layer_slots[layer]++;


        return victim_slot;
    }

    return -1;
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

    for (int32_t i = 0; i < n_ids; i++) {
        int32_t id = original_ids[i];
        if (id < 0) {
            if (out_remapped_ids) out_remapped_ids[i] = -1;
            if (out_is_hit) out_is_hit[i] = false;
            continue;
        }

        int32_t slot = -1;
        if (pool != NULL) {
            ggml_expert_cache_key key = { tensor, id };
            auto it = pool->key_to_slot.find(key);
            if (it != pool->key_to_slot.end()) {
                slot = it->second;
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

void ggml_backend_expert_cache_record_gpu_id_resolution(ggml_backend_expert_cache_t cache) {
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

// Phase 3: Expert Bundles
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

// Phase 5: Host Pinned Staging Buffer
void * ggml_backend_expert_cache_get_pinned_buffer(
        ggml_backend_expert_cache_t cache,
        size_t required_size) {
    if (cache == NULL || required_size == 0) {
        return NULL;
    }

    if (cache->pinned_host_capacity < required_size) {
        if (cache->pinned_host_buffer) {
#if defined(_WIN32)
            _aligned_free(cache->pinned_host_buffer);
#else
            free(cache->pinned_host_buffer);
#endif
        }
        cache->pinned_host_capacity = GGML_EXPERT_CACHE_PAD(required_size);
#if defined(_WIN32)
        cache->pinned_host_buffer = _aligned_malloc(cache->pinned_host_capacity, GGML_EXPERT_CACHE_ALIGN);
#else
        posix_memalign(&cache->pinned_host_buffer, GGML_EXPERT_CACHE_ALIGN, cache->pinned_host_capacity);
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

    // never write over a copy that may still be read by the DMA engine
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
    for (int32_t i = 0; i < n_experts; i++) {
        int32_t eid = expert_ids[i];
        if (eid < 0 || (int64_t)eid >= tensor->ne[2]) continue;

        if (ggml_backend_expert_cache_find_slot(cache, tensor, eid) >= 0) {
            continue; // already resident
        }

        int32_t slot = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, eid, NULL, 0);
        if (slot >= 0) {
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
            cache->stats.n_speculative_prefetches++;
        }
        // else: do nothing (cannot allocate a slot)
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
    GGML_LOG_INFO("  CPU backend bypasses: %" PRIu64 "\n", cache->stats.n_cpu_backend_bypasses);
    GGML_LOG_INFO("  MUL_MAT_ID inputs:    %" PRIu64 "\n", cache->stats.n_mul_mat_id_inputs);
    GGML_LOG_INFO("  non-host bypasses:     %" PRIu64 "\n", cache->stats.n_non_host_weight_bypasses);
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
    GGML_LOG_INFO("  GPU ID resolutions:   %" PRIu64 "\n", cache->stats.n_gpu_id_resolutions);
    GGML_LOG_INFO("  staging memcpy bytes: %zu (%8.2f MiB)\n", cache->stats.staging_memcpy_bytes, (double)cache->stats.staging_memcpy_bytes / (1024.0 * 1024.0));
    GGML_LOG_INFO("  direct DMA bytes:     %zu (%8.2f MiB)\n", cache->stats.direct_pinned_dma_bytes, (double)cache->stats.direct_pinned_dma_bytes / (1024.0 * 1024.0));
    GGML_LOG_INFO("  map updates:          %" PRIu64 " (%zu bytes)\n", cache->stats.n_map_updates, cache->stats.map_update_bytes);
    GGML_LOG_INFO("  staging waits:        %" PRIu64 "\n", cache->stats.n_staging_waits);
    GGML_LOG_INFO("  probe layers:         %" PRIu64 "\n", cache->stats.probe_n_layers);
    GGML_LOG_INFO("  probe sync:           %8.2f ms (per layer %6.2f us)\n",
        (double)cache->stats.probe_sync_us / 1000.0,
        cache->stats.probe_n_layers > 0 ? (double)cache->stats.probe_sync_us / (double)cache->stats.probe_n_layers : 0.0);
    GGML_LOG_INFO("  probe host:           %8.2f ms (per layer %6.2f us)\n",
        (double)cache->stats.probe_host_us / 1000.0,
        cache->stats.probe_n_layers > 0 ? (double)cache->stats.probe_host_us / (double)cache->stats.probe_n_layers : 0.0);
    GGML_LOG_INFO("  probe upload:         %8.2f ms (per layer %6.2f us)\n",
        (double)cache->stats.probe_upload_us / 1000.0,
        cache->stats.probe_n_layers > 0 ? (double)cache->stats.probe_upload_us / (double)cache->stats.probe_n_layers : 0.0);

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

    const size_t expert_size = tensor->nb[2];
    const size_t slot_offset = ggml_backend_expert_cache_alloc_slot(
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

    const int32_t slot = ggml_backend_expert_cache_alloc_slot_idx(
        cache, tensor, expert_id, NULL, 0);

    if (slot < 0) {
        return false;
    }

    struct ggml_tensor * slot_tensor = ggml_backend_expert_cache_get_slot_tensor(cache, tensor);
    if (slot_tensor == NULL) {
        return false;
    }

    const size_t dst_off = (size_t)slot * expert_size;
    ggml_backend_tensor_set_async(
        cache->backend,
        slot_tensor,
        (const uint8_t *)tensor->data + src_off,
        dst_off,
        expert_size);

    return true;
}

void ggml_backend_expert_cache_sync(ggml_backend_expert_cache_t cache) {
    if (cache == NULL) {
        return;
    }
    ggml_backend_synchronize(cache->backend);
}

