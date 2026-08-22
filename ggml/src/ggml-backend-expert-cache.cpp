#include <errno.h>
#include <cstdio>
static FILE * g_route_trace_file = nullptr;

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
#include <map>
#include <vector>
#include <set>

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

// Phase 5C: Prefetch slot structure
struct ggml_expert_cache_prefetch_slot {
    enum ggml_expert_cache_prefetch_state state;
    void * ready_event;  // cudaEvent_t when CUDA available
    const struct ggml_tensor * tensor;
    int32_t expert_id;
    int32_t slot_idx;
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
    int32_t  max_swaps;
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

    // Phase 5A: Route Trace Collection
    bool route_trace_enabled = false;
    FILE * route_trace_file = nullptr;
    std::vector<ggml_expert_cache_route_trace_entry> route_trace_buffer;
    std::vector<float> route_trace_logits_buffer; // interleaved n_logits + blobs
    std::vector<float> route_trace_staged_logits; // last logits per staging
    int32_t route_trace_staged_layer = -1;
    int32_t route_trace_staged_n = 0;
    size_t route_trace_max_entries = 0;
    uint64_t route_trace_token_id = 0;

    // Phase 5C: Async DMA Pipeline
    void * prefetch_stream = nullptr;  // cudaStream_t when CUDA available
    std::vector<ggml_expert_cache_prefetch_slot> prefetch_slots;
    
    // Phase 5C: Heuristic Predictor (Transition Tables)
    bool predictor_enabled = false;
    int32_t predictor_max_layers = 0;
    int32_t predictor_max_experts = 0;
    // transition_table[from_layer][to_layer][expert_id] = count
    std::vector<std::vector<std::vector<int32_t>>> transition_table;
    // current_experts[layer] = list of expert IDs used
    std::vector<std::vector<int32_t>> current_experts;
    
    // Phase 5D: Learned Predictor
    bool hidden_state_trace_enabled = false;
    FILE * hidden_state_trace_file = nullptr;
    std::vector<ggml_expert_cache_hidden_state_sample> hidden_state_buffer;
    size_t hidden_state_max_samples = 0;
    bool learned_predictor_loaded = false;
    void * learned_model = nullptr;  // Opaque pointer to model data
    
    // Phase 5D: Pending predictions (one per target layer; latest replaces)
    std::map<int32_t, ggml_expert_cache_pending_prediction> pending_predictions;
    int32_t executed_layer_cursor = -1;

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
    cache->max_swaps = -1;
    cache->decode_step = 0;
    cache->free_blocks.push_back({ 0, capacity });

    // Phase 5C: Initialize async prefetch stream (CUDA only)
#if defined(GGML_USE_CUDA)
    cudaStream_t stream;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess) {
        cache->prefetch_stream = stream;
    }
#endif

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

    // Phase 5C: Cleanup async prefetch resources
    if (cache->prefetch_stream) {
        cudaStreamDestroy((cudaStream_t)cache->prefetch_stream);
        cache->prefetch_stream = nullptr;
    }
    for (auto& slot : cache->prefetch_slots) {
        if (slot.ready_event) {
            cudaEventDestroy((cudaEvent_t)slot.ready_event);
        }
    }
    cache->prefetch_slots.clear();
#endif

    // Phase 5A: Disable route tracing if enabled
    ggml_backend_expert_cache_disable_route_trace(cache);

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
void ggml_backend_expert_cache_rebalance(ggml_backend_expert_cache_t cache, int max_swaps = -1) {
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

    // Limit swaps if max_swaps is specified
    if (max_swaps >= 0) {
        if ((int)to_evict.size() > max_swaps) {
            to_evict.resize(max_swaps);
        }
        if ((int)to_load.size() > max_swaps) {
            to_load.resize(max_swaps);
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
    if (cache->route_trace_enabled) {
        cache->route_trace_token_id++;
    }
    cache->executed_layer_cursor = -1;
    if (cache->period_tokens > 0 && (cache->decode_step % (uint64_t)cache->period_tokens == 0)) {
        ggml_backend_expert_cache_rebalance(cache, cache->max_swaps);
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

// Phase 5C: Async DMA Pipeline

void ggml_backend_expert_cache_prefetch_async(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        const int32_t * expert_ids,
        int32_t n_experts,
        int32_t target_layer) {
    if (cache == NULL || tensor == NULL || expert_ids == NULL || n_experts <= 0) {
        return;
    }
    
    // Check if async prefetch is supported
    if (cache->prefetch_stream == nullptr) {
        // Fall back to synchronous prefetch
        ggml_backend_expert_cache_prefetch(cache, tensor, expert_ids, n_experts);
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
        
        // Skip if already resident
        if (ggml_backend_expert_cache_find_slot(cache, tensor, eid) >= 0) {
            cache->stats.n_already_resident++;
            continue;
        }
        
        // Check if already prefetching this expert
        bool already_prefetching = false;
        for (const auto& slot : cache->prefetch_slots) {
            if (slot.tensor == tensor && slot.expert_id == eid && 
                slot.state == GGML_EXPERT_CACHE_PREFETCH_IN_FLIGHT) {
                already_prefetching = true;
                break;
            }
        }
        if (already_prefetching) {
            continue;
        }
        
        // Allocate slot for this expert
        int32_t slot_idx = ggml_backend_expert_cache_alloc_slot_idx(cache, tensor, eid, NULL, 0);
        if (slot_idx < 0) {
            continue;
        }
        
        const size_t src_off = (size_t)eid * expert_size;
        const size_t dst_off = (size_t)slot_idx * expert_size;
        
#if defined(GGML_USE_CUDA)
        // Create CUDA event for this prefetch
        cudaEvent_t event;
        if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
            continue;
        }
        
        // Get pinned buffer for staging
        void * pinned_buf = ggml_backend_expert_cache_stage_acquire(cache, tensor, slot_idx, expert_size);
        if (pinned_buf != NULL) {
            // Copy to pinned buffer
            memcpy(pinned_buf, (const uint8_t *) tensor->data + src_off, expert_size);
            
            // Issue async copy on dedicated prefetch stream
            cudaMemcpyAsync(
                (uint8_t*)pool->tensor->data + dst_off,
                pinned_buf,
                expert_size,
                cudaMemcpyHostToDevice,
                (cudaStream_t)cache->prefetch_stream
            );
            
            // Record event on prefetch stream
            cudaEventRecord(event, (cudaStream_t)cache->prefetch_stream);
            
            ggml_backend_expert_cache_stage_commit(cache, tensor, slot_idx);
        } else {
            // No pinned buffer available, skip this prefetch
            cudaEventDestroy(event);
            continue;
        }
        
        // Track this prefetch
        ggml_expert_cache_prefetch_slot slot;
        slot.state = GGML_EXPERT_CACHE_PREFETCH_IN_FLIGHT;
        slot.ready_event = event;
        slot.tensor = tensor;
        slot.expert_id = eid;
        slot.slot_idx = slot_idx;
        cache->prefetch_slots.push_back(slot);
        
        cache->stats.n_prefetch_issued++;
#else
        // Non-CUDA backend: fall back to synchronous
        void * pinned_buf = ggml_backend_expert_cache_stage_acquire(cache, tensor, slot_idx, expert_size);
        if (pinned_buf != NULL) {
            memcpy(pinned_buf, (const uint8_t *) tensor->data + src_off, expert_size);
            ggml_backend_tensor_set_async(
                cache->backend,
                pool->tensor,
                pinned_buf,
                dst_off,
                expert_size);
            ggml_backend_expert_cache_stage_commit(cache, tensor, slot_idx);
        } else {
            ggml_backend_tensor_set_async(
                cache->backend,
                pool->tensor,
                (const uint8_t *) tensor->data + src_off,
                dst_off,
                expert_size);
        }
        cache->stats.n_speculative_prefetches++;
#endif
    }
}

static bool has_inflight_prefetch(ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor, int32_t expert_id);


bool ggml_backend_expert_cache_has_inflight_prefetch(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id) {
    return has_inflight_prefetch(cache, tensor, expert_id);
}

bool ggml_backend_expert_cache_is_prefetch_ready(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id) {
    if (cache == NULL || tensor == NULL) {
        return false;
    }
    
#if defined(GGML_USE_CUDA)
    // Check if this expert has an in-flight prefetch
    for (auto& slot : cache->prefetch_slots) {
        if (slot.tensor == tensor && slot.expert_id == expert_id) {
            if (slot.state == GGML_EXPERT_CACHE_PREFETCH_IN_FLIGHT) {
                // Check if the event is complete
                cudaError_t err = cudaEventQuery((cudaEvent_t)slot.ready_event);
                if (err == cudaSuccess) {
                    // Prefetch complete, mark as resident
                    slot.state = GGML_EXPERT_CACHE_PREFETCH_RESIDENT;
                    cache->stats.n_prefetch_hits++;
                    return true;
                } else if (err == cudaErrorNotReady) {
                    // Still in flight
                    return false;
                } else {
                    // Error occurred
                    return false;
                }
            } else if (slot.state == GGML_EXPERT_CACHE_PREFETCH_RESIDENT) {
                return true;
            }
        }
    }
#endif
    
    return false;
}

void ggml_backend_expert_cache_wait_prefetch(
        ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor,
        int32_t expert_id) {
    if (cache == NULL || tensor == NULL) {
        return;
    }
    
#if defined(GGML_USE_CUDA)
    // Find and wait for this prefetch
    for (auto& slot : cache->prefetch_slots) {
        if (slot.tensor == tensor && slot.expert_id == expert_id) {
            if (slot.state == GGML_EXPERT_CACHE_PREFETCH_IN_FLIGHT) {
                cudaEventSynchronize((cudaEvent_t)slot.ready_event);
                slot.state = GGML_EXPERT_CACHE_PREFETCH_RESIDENT;
                cache->stats.n_prefetch_waits++;
            }
            break;
        }
    }
#endif
}

// Phase 5C: Heuristic Predictor Implementation

void ggml_backend_expert_cache_enable_predictor(
        ggml_backend_expert_cache_t cache,
        int32_t max_layers,
        int32_t max_experts_per_layer) {
    if (cache == NULL || max_layers <= 0 || max_experts_per_layer <= 0) {
        return;
    }

    
    cache->predictor_enabled = true;
    cache->predictor_max_layers = max_layers;
    cache->predictor_max_experts = max_experts_per_layer;
    
    // Initialize transition table: [from_layer][to_layer][expert_id]
    cache->transition_table.resize(max_layers);
    for (int32_t i = 0; i < max_layers; i++) {
        cache->transition_table[i].resize(max_layers);
        for (int32_t j = 0; j < max_layers; j++) {
            cache->transition_table[i][j].resize(max_experts_per_layer, 0);
        }
    }
    
    // Initialize current experts tracking
    cache->current_experts.resize(max_layers);
    
    GGML_LOG_INFO("Expert cache predictor enabled: %d layers, %d experts/layer\n",
                  max_layers, max_experts_per_layer);
}

void ggml_backend_expert_cache_disable_predictor(
        ggml_backend_expert_cache_t cache) {
    if (cache == NULL) {
        return;
    }
    
    cache->predictor_enabled = false;
    cache->transition_table.clear();
    cache->current_experts.clear();
    
    GGML_LOG_INFO("Expert cache predictor disabled\n");
}

void ggml_backend_expert_cache_record_prediction(
        ggml_backend_expert_cache_t cache,
        int32_t layer,
        const int32_t * expert_ids,
        int32_t n_experts) {
    if (cache == NULL || !cache->predictor_enabled || layer < 0 || 
        layer >= cache->predictor_max_layers || expert_ids == NULL || n_experts <= 0) {
        return;
    }
    
    // Update transition table: for each expert used in this layer,
    // increment the transition count from previous layer's experts
    if (layer > 0 && !cache->current_experts[layer - 1].empty()) {
        for (int32_t prev_expert : cache->current_experts[layer - 1]) {
            for (int32_t i = 0; i < n_experts; i++) {
                int32_t curr_expert = expert_ids[i];
                if (curr_expert >= 0 && curr_expert < cache->predictor_max_experts) {
                    cache->transition_table[layer - 1][layer][curr_expert]++;
                }
            }
        }
    }
    
    // Update current experts for this layer
    cache->current_experts[layer].clear();
    for (int32_t i = 0; i < n_experts; i++) {
        if (expert_ids[i] >= 0 && expert_ids[i] < cache->predictor_max_experts) {
            cache->current_experts[layer].push_back(expert_ids[i]);
        }
    }
}

int32_t ggml_backend_expert_cache_predict_experts(
        ggml_backend_expert_cache_t cache,
        int32_t from_layer,
        int32_t to_layer,
        int32_t * out_expert_ids,
        int32_t max_experts) {
    if (cache == NULL || !cache->predictor_enabled || 
        from_layer < 0 || from_layer >= cache->predictor_max_layers ||
        to_layer < 0 || to_layer >= cache->predictor_max_layers ||
        out_expert_ids == NULL || max_experts <= 0) {
        return 0;
    }
    
    // If we have current experts for from_layer, use transition table
    if (!cache->current_experts[from_layer].empty()) {
        // Collect all predicted experts with their scores
        std::vector<std::pair<int32_t, int32_t>> expert_scores; // (expert_id, score)
        
        for (int32_t prev_expert : cache->current_experts[from_layer]) {
            for (int32_t expert_id = 0; expert_id < cache->predictor_max_experts; expert_id++) {
                int32_t score = cache->transition_table[from_layer][to_layer][expert_id];
                if (score > 0) {
                    // Check if expert already in list
                    bool found = false;
                    for (auto& p : expert_scores) {
                        if (p.first == expert_id) {
                            p.second += score;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        expert_scores.push_back({expert_id, score});
                    }
                }
            }
        }
        
        // Sort by score (descending)
        std::sort(expert_scores.begin(), expert_scores.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Return top experts
        int32_t count = 0;
        for (const auto& p : expert_scores) {
            if (count >= max_experts) break;
            out_expert_ids[count++] = p.first;
        }
        
        return count;
    }
    
    // No history available, return 0
    return 0;
}


void ggml_backend_expert_cache_record_gpu_slot_execution(ggml_backend_expert_cache_t cache) {
    if (cache) cache->stats.n_gpu_slot_executions++;
}
void ggml_backend_expert_cache_record_cpu_fallback(ggml_backend_expert_cache_t cache) {
    if (cache) cache->stats.n_cpu_fallbacks++;
}
void ggml_backend_expert_cache_record_used_ready(ggml_backend_expert_cache_t cache) {
    if (cache) cache->stats.n_used_ready++;
}
void ggml_backend_expert_cache_record_used_in_flight(ggml_backend_expert_cache_t cache) {
    if (cache) cache->stats.n_used_in_flight++;
}
void ggml_backend_expert_cache_record_used_miss(ggml_backend_expert_cache_t cache) {
    if (cache) cache->stats.n_used_miss++;
}
void ggml_backend_expert_cache_record_already_resident(ggml_backend_expert_cache_t cache) {
    if (cache) cache->stats.n_already_resident++;
}
void ggml_backend_expert_cache_record_in_flight_wait_us(ggml_backend_expert_cache_t cache, uint64_t us) {
    if (cache) cache->stats.in_flight_wait_us += us;
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

// Phase 5A: Route Trace Collection
void ggml_backend_expert_cache_enable_route_trace(
        ggml_backend_expert_cache_t cache,
        const char * output_path,
        size_t max_entries) {
    if (cache == NULL || output_path == NULL) {
        return;
    }
    // Process-wide guard: ggml_backend_sched_set_expert_cache is invoked once per
    // backend and re-invoked per slot load. Only the first call should open the
    // file and set the shared FILE*; later calls just mark this cache as enabled
    // so record_route_trace writes through the same handle.
    if (g_route_trace_file == nullptr) {
        cache->route_trace_file = fopen(output_path, "wb");
        g_route_trace_file = cache->route_trace_file;
        if (cache->route_trace_file == nullptr) {
            GGML_LOG_ERROR("Failed to open route trace file: %s\n", output_path);
            return;
        }
        const uint32_t magic = 0x52545243;
        const uint32_t version = 2;
        fwrite(&magic, sizeof(magic), 1, cache->route_trace_file);
        fwrite(&version, sizeof(version), 1, cache->route_trace_file);
        fflush(cache->route_trace_file);
    } else {
        // Reuse the file handle opened by the first caller.
        cache->route_trace_file = g_route_trace_file;
    }

    cache->route_trace_enabled = true;
    cache->route_trace_max_entries = max_entries > 0 ? max_entries : 1000000;
    // Do NOT clear route_trace_buffer here: it's per-cache and accumulates entries
    // from record_route_trace between flushes. Clearing on every slot init would
    // drop all entries before they can be flushed.
    cache->route_trace_token_id = 0;

    GGML_LOG_INFO("Route trace enabled: %s (max %zu entries)\n", output_path, cache->route_trace_max_entries);
}

void ggml_backend_expert_cache_record_router_logits(
        ggml_backend_expert_cache_t cache,
        int32_t layer,
        const float * logits,
        int32_t n_logits) {
    if (cache == NULL || !cache->route_trace_enabled || logits == NULL || n_logits <= 0) {
        return;
    }

    cache->route_trace_staged_layer = layer;
    cache->route_trace_staged_n = n_logits;
    cache->route_trace_staged_logits.assign(logits, logits + n_logits);
}

void ggml_backend_expert_cache_record_route_trace(
        ggml_backend_expert_cache_t cache,
        int32_t layer,
        const int32_t * expert_ids,
        int32_t n_experts) {
    if (cache == NULL || !cache->route_trace_enabled || expert_ids == NULL || n_experts <= 0) {
        return;
    }

    ggml_expert_cache_route_trace_entry entry;
    entry.token_id = cache->route_trace_token_id;
    entry.layer = layer;
    entry.n_experts = n_experts;
    entry.timestamp_us = ggml_time_us();

    // Copy expert IDs (cap at 64)
    const int32_t copy_count = std::min(n_experts, 64);
    for (int32_t i = 0; i < copy_count; i++) {
        entry.expert_ids[i] = expert_ids[i];
    }
    for (int32_t i = copy_count; i < 64; i++) {
        entry.expert_ids[i] = -1; // padding
    }

    cache->route_trace_buffer.push_back(entry);

    // v2: variable-length logits blob follows each fixed-size entry.
    const bool have_logits = cache->route_trace_staged_layer == layer &&
                             cache->route_trace_staged_n > 0 &&
                             !cache->route_trace_staged_logits.empty();
    if (have_logits) {
        const int32_t n = cache->route_trace_staged_n;
        cache->route_trace_logits_buffer.push_back(n);
        cache->route_trace_logits_buffer.insert(cache->route_trace_logits_buffer.end(),
                cache->route_trace_staged_logits.begin(), cache->route_trace_staged_logits.begin() + n);
        cache->route_trace_staged_layer = -1;
        cache->route_trace_staged_n = 0;
    } else {
        cache->route_trace_logits_buffer.push_back(0);
    }

    // Flush every entry: SIGTERM on Windows kills without running destructors,
    // so the buffer must be drained immediately to survive abrupt shutdown.
    ggml_backend_expert_cache_flush_route_trace(cache);
}

void ggml_backend_expert_cache_flush_route_trace(
        ggml_backend_expert_cache_t cache) {
    if (cache == NULL || cache->route_trace_file == nullptr || cache->route_trace_buffer.empty()) {
        return;
    }

    // Write each entry directly. FILE* is shared across caches and was opened
    // in "wb" mode by the first caller; after writing the 8-byte header the
    // file position is past the header, so subsequent writes append naturally.
    size_t logits_pos = 0;
    for (const auto & e : cache->route_trace_buffer) {
        GGML_ASSERT(logits_pos < cache->route_trace_logits_buffer.size());
        const int32_t n_logits = cache->route_trace_logits_buffer[logits_pos++];
        fwrite(&e, sizeof(e), 1, cache->route_trace_file);
        fwrite(&n_logits, sizeof(n_logits), 1, cache->route_trace_file);
        if (n_logits > 0) {
            fwrite(cache->route_trace_logits_buffer.data() + logits_pos,
                   sizeof(float), n_logits, cache->route_trace_file);
            logits_pos += n_logits;
        }
    }
    fflush(cache->route_trace_file);

    cache->route_trace_buffer.clear();
    cache->route_trace_logits_buffer.clear();
}



void ggml_backend_expert_cache_disable_route_trace(
        ggml_backend_expert_cache_t cache) {
    if (cache == NULL) {
        return;
    }

    cache->route_trace_enabled = false;

    // Flush remaining entries
    ggml_backend_expert_cache_flush_route_trace(cache);

    // Only close the file if it's not the process-wide shared handle.
    // Multiple caches share the same FILE* via g_route_trace_file; closing it
    // from one cache's disable leaves siblings with dangling pointers.
    if (cache->route_trace_file != nullptr && cache->route_trace_file != g_route_trace_file) {
        fclose(cache->route_trace_file);
    }
    cache->route_trace_file = nullptr;

    GGML_LOG_INFO("Route trace disabled\n");
}

// Phase 5D: Learned Predictor Implementation

void ggml_backend_expert_cache_enable_hidden_state_trace(
        ggml_backend_expert_cache_t cache,
        const char * output_path,
        size_t max_samples) {
    if (cache == NULL || output_path == NULL) {
        return;
    }
    
    cache->hidden_state_trace_file = fopen(output_path, "wb");
    if (cache->hidden_state_trace_file == NULL) {
        GGML_LOG_ERROR("Failed to open hidden state trace file: %s\n", output_path);
        return;
    }
    
    cache->hidden_state_trace_enabled = true;
    cache->hidden_state_max_samples = max_samples;
    cache->hidden_state_buffer.reserve(std::min(max_samples, (size_t)10000));
    
    GGML_LOG_INFO("Hidden state trace enabled: %s (max %zu samples)\n", 
                  output_path, max_samples);
}

void ggml_backend_expert_cache_disable_hidden_state_trace(
        ggml_backend_expert_cache_t cache) {
    if (cache == NULL) {
        return;
    }
    
    if (!cache->hidden_state_trace_enabled) {
        return;
    }
    
    // Flush buffer to file
    if (cache->hidden_state_trace_file != NULL && !cache->hidden_state_buffer.empty()) {
        fwrite(cache->hidden_state_buffer.data(), 
               sizeof(ggml_expert_cache_hidden_state_sample),
               cache->hidden_state_buffer.size(),
               cache->hidden_state_trace_file);
        cache->hidden_state_buffer.clear();
    }
    
    if (cache->hidden_state_trace_file != NULL) {
        fclose(cache->hidden_state_trace_file);
        cache->hidden_state_trace_file = NULL;
    }
    
    cache->hidden_state_trace_enabled = false;
    GGML_LOG_INFO("Hidden state trace disabled\n");
}

void ggml_backend_expert_cache_record_hidden_state(
        ggml_backend_expert_cache_t cache,
        int32_t layer,
        int32_t token_id,
        const float * hidden_state,
        int32_t hidden_dim,
        const int32_t * expert_ids,
        int32_t n_experts) {
    if (cache == NULL || !cache->hidden_state_trace_enabled || 
        hidden_state == NULL || expert_ids == NULL) {
        return;
    }
    
    if (cache->hidden_state_buffer.size() >= cache->hidden_state_max_samples) {
        // Buffer full, flush to file
        if (cache->hidden_state_trace_file != NULL) {
            fwrite(cache->hidden_state_buffer.data(),
                   sizeof(ggml_expert_cache_hidden_state_sample),
                   cache->hidden_state_buffer.size(),
                   cache->hidden_state_trace_file);
            cache->hidden_state_buffer.clear();
        }
    }
    
    ggml_expert_cache_hidden_state_sample sample;
    sample.layer = layer;
    sample.token_id = token_id;
    sample.n_experts = std::min(n_experts, (int32_t)8);
    
    // Copy hidden state (truncate or pad to 256 dims)
    int32_t copy_dim = std::min(hidden_dim, (int32_t)256);
    memcpy(sample.hidden_state, hidden_state, copy_dim * sizeof(float));
    if (copy_dim < 256) {
        memset(sample.hidden_state + copy_dim, 0, (256 - copy_dim) * sizeof(float));
    }
    
    // Copy expert IDs
    memcpy(sample.expert_ids, expert_ids, sample.n_experts * sizeof(int32_t));
    
    cache->hidden_state_buffer.push_back(sample);
}

struct learned_predictor_model {
    int32_t input_dim;
    int32_t rank;
    int32_t num_experts;
    std::vector<float> down_weight;   // [rank * input_dim]
    std::vector<float> up_weight;     // [input_dim * rank]
    std::vector<float> output_weight; // [num_experts * input_dim]
    std::vector<float> output_bias;   // [num_experts]
};

bool ggml_backend_expert_cache_load_learned_predictor(
        ggml_backend_expert_cache_t cache,
        const char * model_path) {
    if (cache == NULL || model_path == NULL) {
        return false;
    }
    
    FILE * f = fopen(model_path, "rb");
    if (f == NULL) {
        GGML_LOG_ERROR("Failed to open model file: %s\n", model_path);
        return false;
    }
    
    // Read header
    uint32_t magic, version;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 ||
        fread(&version, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return false;
    }
    
    if (magic != 0x4C525044 || (version != 1 && version != 2)) {
        GGML_LOG_ERROR("Invalid model file format (magic=%#x version=%u)\n", magic, version);
        fclose(f);
        return false;
    }
    
    // Read dimensions
    int32_t input_dim, rank, num_experts;
    if (fread(&input_dim, sizeof(int32_t), 1, f) != 1 ||
        fread(&rank, sizeof(int32_t), 1, f) != 1 ||
        fread(&num_experts, sizeof(int32_t), 1, f) != 1) {
        fclose(f);
        return false;
    }
    
    // Allocate model
    auto * model = new learned_predictor_model();
    model->input_dim = input_dim;
    model->rank = rank;
    model->num_experts = num_experts;
    
    // Read weights
    model->down_weight.resize(rank * input_dim);
    model->up_weight.resize(input_dim * rank);
    model->output_weight.resize(num_experts * input_dim);
    model->output_bias.resize(num_experts);
    
    if (fread(model->down_weight.data(), sizeof(float), rank * input_dim, f) != (size_t)(rank * input_dim) ||
        fread(model->up_weight.data(), sizeof(float), input_dim * rank, f) != (size_t)(input_dim * rank) ||
        fread(model->output_weight.data(), sizeof(float), num_experts * input_dim, f) != (size_t)(num_experts * input_dim) ||
        fread(model->output_bias.data(), sizeof(float), num_experts, f) != (size_t)num_experts) {
        GGML_LOG_ERROR("Failed to read model weights\n");
        delete model;
        fclose(f);
        return false;
    }
    
    fclose(f);
    
    // Free old model if exists
    if (cache->learned_model != nullptr) {
        delete static_cast<learned_predictor_model*>(cache->learned_model);
    }
    
    cache->learned_model = model;
    cache->learned_predictor_loaded = true;
    
    GGML_LOG_INFO("Loaded learned predictor model: input_dim=%d, rank=%d, num_experts=%d\n",
                  input_dim, rank, num_experts);
    
    return true;
}

int32_t ggml_backend_expert_cache_predict_with_learned_model(
        ggml_backend_expert_cache_t cache,
        int32_t layer,
        const float * hidden_state,
        int32_t hidden_dim,
        int32_t * out_expert_ids,
        int32_t max_experts) {
    if (cache == NULL || !cache->learned_predictor_loaded ||
        hidden_state == NULL || out_expert_ids == NULL) {
        return 0;
    }
    
    auto * model = static_cast<learned_predictor_model*>(cache->learned_model);
    
    // Prepare input (pad or truncate to input_dim)
    std::vector<float> input(model->input_dim, 0.0f);
    int32_t copy_dim = std::min(hidden_dim, model->input_dim);
    memcpy(input.data(), hidden_state, copy_dim * sizeof(float));
    
    // Forward pass: down_proj
    std::vector<float> hidden(model->rank, 0.0f);
    for (int32_t i = 0; i < model->rank; i++) {
        float sum = 0.0f;
        for (int32_t j = 0; j < model->input_dim; j++) {
            sum += model->down_weight[i * model->input_dim + j] * input[j];
        }
        // GELU activation (approximation)
        hidden[i] = sum * 0.5f * (1.0f + tanhf(0.7978845608f * (sum + 0.044715f * sum * sum * sum)));
    }
    
    // Forward pass: up_proj
    std::vector<float> reconstructed(model->input_dim, 0.0f);
    for (int32_t i = 0; i < model->input_dim; i++) {
        float sum = 0.0f;
        for (int32_t j = 0; j < model->rank; j++) {
            sum += model->up_weight[i * model->rank + j] * hidden[j];
        }
        // GELU activation
        reconstructed[i] = sum * 0.5f * (1.0f + tanhf(0.7978845608f * (sum + 0.044715f * sum * sum * sum)));
    }
    
    // Forward pass: output_proj
    std::vector<float> logits(model->num_experts);
    for (int32_t i = 0; i < model->num_experts; i++) {
        float sum = model->output_bias[i];
        for (int32_t j = 0; j < model->input_dim; j++) {
            sum += model->output_weight[i * model->input_dim + j] * reconstructed[j];
        }
        logits[i] = sum;
    }
    
    // Get top-k experts
    std::vector<std::pair<float, int32_t>> expert_scores;
    for (int32_t i = 0; i < model->num_experts; i++) {
        expert_scores.push_back({logits[i], i});
    }
    
    std::partial_sort(expert_scores.begin(),
                      expert_scores.begin() + std::min(max_experts, (int32_t)expert_scores.size()),
                      expert_scores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    int32_t n_output = std::min(max_experts, (int32_t)expert_scores.size());
    for (int32_t i = 0; i < n_output; i++) {
        out_expert_ids[i] = expert_scores[i].second;
    }
    
    return n_output;
}

// Phase 5D: Prediction Submission (Revised Architecture)
// Submit predicted expert IDs for a future layer to trigger async prefetch

void ggml_backend_expert_cache_submit_prediction(
        ggml_backend_expert_cache_t cache,
        int32_t target_layer,
        const int32_t * expert_ids,
        int32_t n_experts,
        const float * confidences) {
    if (!cache || !expert_ids || n_experts <= 0) {
        return;
    }

    // Track metrics
    cache->stats.n_predictions_submitted++;

    // Log prediction (debug level)
    if (getenv("GGML_PREDICTOR_DEBUG")) {
        fprintf(stderr, "[predictor] Layer %d: %d experts predicted: ", target_layer, n_experts);
        for (int32_t i = 0; i < n_experts && i < 8; i++) {
            fprintf(stderr, "%d", expert_ids[i]);
            if (confidences) {
                fprintf(stderr, "(%.2f)", confidences[i]);
            }
            if (i < n_experts - 1 && i < 7) {
                fprintf(stderr, ", ");
            }
        }
        if (n_experts > 8) {
            fprintf(stderr, ", ...");
        }
        fprintf(stderr, "\n");
    }

    // Queue the prediction for settle accounting; latest submission for a
    // layer replaces the previous one.
    auto & e = cache->pending_predictions[target_layer];
    e.target_layer = target_layer;
    e.n_experts = std::min(n_experts, (int32_t)64);
    memcpy(e.expert_ids, expert_ids, e.n_experts * sizeof(int32_t));

    // Prefetch the predicted experts' weights immediately (D2)
    auto bit = cache->bundle_registrations.find(target_layer);
    if (bit != cache->bundle_registrations.end()) {
        if (bit->second.gate) ggml_backend_expert_cache_prefetch_async(cache, bit->second.gate, e.expert_ids, e.n_experts, target_layer);
        if (bit->second.up)   ggml_backend_expert_cache_prefetch_async(cache, bit->second.up,   e.expert_ids, e.n_experts, target_layer);
        if (bit->second.down) ggml_backend_expert_cache_prefetch_async(cache, bit->second.down, e.expert_ids, e.n_experts, target_layer);
    }

}

static bool has_inflight_prefetch(ggml_backend_expert_cache_t cache,
        const struct ggml_tensor * tensor, int32_t expert_id) {
    for (const auto & slot : cache->prefetch_slots) {
        if (slot.tensor == tensor && slot.expert_id == expert_id &&
            slot.state == GGML_EXPERT_CACHE_PREFETCH_IN_FLIGHT) {
            return true;
        }
    }
    return false;
}

bool ggml_backend_expert_cache_settle_prediction(
        ggml_backend_expert_cache_t cache,
        int32_t layer,
        const int32_t * actual_ids,
        int32_t n_actual,
        size_t pool_stride) {
    if (!cache || !actual_ids || n_actual <= 0) {
        return false;
    }
    auto it = cache->pending_predictions.find(layer);
    if (it == cache->pending_predictions.end()) {
        return false;
    }

    auto & s = cache->stats.routing_predictor;
    if (layer <= cache->executed_layer_cursor) {
        s.predictions_too_late++;
    }
    cache->executed_layer_cursor = layer;

    const auto & p = it->second;
    s.predictions_used++;

    std::set<int32_t> actual_set(actual_ids, actual_ids + n_actual);

    // Classify predicted experts that were actually used
    auto bit = cache->bundle_registrations.find(layer);
    for (int i = 0; i < p.n_experts; i++) {
        const int32_t id = p.expert_ids[i];
        if (!actual_set.count(id)) {
            continue;
        }
        bool resident = false;
        bool inflight = false;
        if (bit != cache->bundle_registrations.end()) {
            const struct ggml_tensor * tensors[3] = { bit->second.gate, bit->second.up, bit->second.down };
            for (int t = 0; t < 3 && !resident; t++) {
                if (!tensors[t]) continue;
                if (ggml_backend_expert_cache_find_slot(cache, tensors[t], id) >= 0) {
                    resident = true;
                } else if (has_inflight_prefetch(cache, tensors[t], id)) {
                    inflight = true;
                }
            }
        }
        if (resident) {
            s.experts_fully_hidden++;
        } else if (inflight) {
            s.experts_partially_hidden++;
        } else {
            s.experts_missed++;
        }
    }

    // Experts the layer wanted but we did not predict
    for (int32_t id : actual_set) {
        bool predicted = false;
        for (int i = 0; i < p.n_experts; i++) {
            predicted |= p.expert_ids[i] == id;
        }
        if (!predicted) {
            s.predictions_wrong++;
        }
    }

    // Prefetched ids that never got used are pure PCIe waste
    for (int i = 0; i < p.n_experts; i++) {
        const int32_t id = p.expert_ids[i];
        if (!actual_set.count(id)) {
            cache->stats.wasted_prefetch_bytes += pool_stride;
        }
    }

    cache->pending_predictions.erase(it);
    return true;
}

int32_t ggml_backend_expert_cache_pending_prediction_count(
        ggml_backend_expert_cache_t cache) {
    if (!cache) {
        return 0;
    }
    return (int32_t) cache->pending_predictions.size();
}

int32_t ggml_backend_expert_cache_get_pending_prediction(
        ggml_backend_expert_cache_t cache,
        int32_t target_layer,
        int32_t * out_ids,
        int32_t max_ids) {
    if (!cache || !out_ids || max_ids <= 0) {
        return 0;
    }
    auto it = cache->pending_predictions.find(target_layer);
    if (it == cache->pending_predictions.end()) {
        return 0;
    }
    const int32_t n = std::min(it->second.n_experts, max_ids);
    memcpy(out_ids, it->second.expert_ids, n * sizeof(int32_t));
    return n;
}

bool ggml_backend_expert_cache_get_routing_predictor_stats(
        ggml_backend_expert_cache_t cache,
        struct ggml_routing_predictor_stats * out_stats) {
    if (!cache || !out_stats) {
        return false;
    }
    *out_stats = cache->stats.routing_predictor;
    return true;
}

void ggml_backend_expert_cache_add_predictions_generated(
        ggml_backend_expert_cache_t cache,
        int32_t n) {
    if (!cache || n <= 0) {
        return;
    }
    cache->stats.routing_predictor.predictions_generated += n;
}

