# Expert Cache Phase 0 Fixes + Timing Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the two verified expert-cache correctness defects (pinned-staging overwrite race, ascending-frequency seed admission) and add a per-layer timing probe in `ggml_backend_sched_compute_splits`, then re-verify the deterministic token matrix and PP/TG baselines.

**Architecture:** The expert cache has two representations: a legacy flat byte cache (PP fallback path) and 3D slot pools (single-token decode zero-copy path). Task 1 makes the CUDA/DMA staging ring safe by keying ring entries by `(tensor, slot_idx)` and waiting on per-entry events (or a full backend sync when events are unsupported) before reuse. Task 2 reverses the profile seed order so the hottest experts are admitted before capacity fills, and extracts the sort into a testable pure helper. Task 3 adds wall-time accumulators at the three known per-layer serialization points (ids sync, host decision, upload issue). Task 4 rebuilds, runs unit tests, and re-runs the deterministic matrix plus llama-bench PP512/TG128 sweeps.

**Tech Stack:** C++17, ggml scheduler (`ggml-backend.cpp`), ggml expert cache (`ggml-backend-expert-cache.cpp/h`), CUDA backend, common profile loader (`expert-cache-profile.cpp/h`), CMake Release build on Windows (MSVC), assert-style test binaries (no test framework).

## Global Constraints

- ASCII only in code comments: no `-`, `->`, `...` (AGENTS.md: use `-`, `->`, `..`). No emdash, no unicode arrows.
- No AI-written commit messages or PR text. Commit only after user approval per commit. If the user approves committing on their behalf, use `Assisted-by: <assistant name>` in the message body.
- Do NOT run `git push` or `gh pr create` - this repo's AGENTS.md forbids automated submissions.
- Determinism: temperature-0 token sequences must remain reproducible (verified in Task 4 by token-hash comparison).
- Follow existing code patterns: assert-style tests appended to existing test binaries; 512-byte aligned pinned buffers (`GGML_EXPERT_CACHE_ALIGN`); telemetry via `ggml_backend_expert_cache_stats` (do not add new print paths, extend the existing struct/print).
- q8_0 experts on Pascal keep using fused `mmvq`/`mmq` kernels; this plan does not change kernel selection.
- No new subsystems, no new dependencies. Reuse `ggml_backend_event_*` (exists on CUDA; returns NULL on unsupported backends) and existing `ggml_time_us()`.

---

### Task 1: Fix Pinned-Staging Overwrite Race

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.h:203-206` (replace `get_pinned_slot_buffer` declaration)
- Modify: `ggml/src/ggml-backend-expert-cache.cpp` (`ggml_backend_expert_cache` struct at ~120-162, `free` at ~347-385, `prefetch` at ~1182-1230, `get_pinned_slot_buffer` at ~1162-1179)
- Modify: `ggml/src/ggml-backend.cpp:1838-1854` (staged miss upload caller)
- Test: `tests/test-expert-cache.cpp` (append new test; `main()` at end of file)

**Interfaces:**
- Consumes: `ggml_backend_event_new(ggml_backend_dev_t)`, `ggml_backend_event_record(event, backend)`, `ggml_backend_event_synchronize(event)`, `ggml_backend_event_free(event)`, `ggml_backend_get_device(backend)`, `GGML_EXPERT_CACHE_PAD(x)` (all exist).
- Produces (later tasks rely on these):
  - `GGML_EXPERT_CACHE_STAGING_ENTRIES` = 32 (macro in expert-cache.cpp)
  - `void * ggml_backend_expert_cache_stage_acquire(ggml_backend_expert_cache_t cache, const struct ggml_tensor * tensor, int32_t slot_idx, size_t required_size);`
  - `void ggml_backend_expert_cache_stage_commit(ggml_backend_expert_cache_t cache, const struct ggml_tensor * tensor, int32_t slot_idx);`
  - `uint64_t ggml_backend_expert_cache_stats::n_staging_waits` (added in `ggml/include/ggml-backend.h:364-383` struct)

**Why:** The staging ring (`get_pinned_slot_buffer`) maps `slot_idx % 16` to one of 16 pinned buffers with no wait before reuse. Gate/up/down projections are distinct slot pools, each starting at slot index 0, and the same slot index recurs across layers within a single `compute_splits` call. The host `memcpy` for the next miss can overwrite a pinned buffer the CUDA DMA engine has not yet read: data corruption. Fix = key ring entries by `(tensor, slot_idx)` and wait for the previous DMA (event, or full backend sync when events unsupported) before writing.

- [ ] **Step 1: Write the failing test**

Append to `tests/test-expert-cache.cpp` (before `main()`):

```cpp
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
    struct ggml_tensor * tensor_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 16);
    ggml_set_name(tensor_b, "blk.1.ffn_gate_exps.weight");

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
```

And call it from `main()`:

```cpp
    test_cache_node_selection();
    test_cache_capacity_admission();
    test_slot_pools_and_remapping();
    test_pinned_staging_no_overwrite();
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --config Release --target test-expert-cache`
Then: `build/bin/Release/test-expert-cache.exe`
Expected: compile error - `ggml_backend_expert_cache_stage_acquire` and `stage_commit` not declared, and `n_staging_waits` missing from the stats struct.

- [ ] **Step 3: Add the stats counter**

In `ggml/include/ggml-backend.h`, inside `struct ggml_backend_expert_cache_stats` (after the V2 Diagnostic Telemetry fields, ~line 383):

```c
        // V2.3 Pinned staging: number of times a staging ring entry had to wait
        // for a still-in-flight DMA before being reused
        uint64_t n_staging_waits;
```

- [ ] **Step 4: Replace the staging API in the header**

In `ggml/src/ggml-backend-expert-cache.h`, replace the `get_pinned_slot_buffer` declaration (lines ~203-206) with:

```c
GGML_API void * ggml_backend_expert_cache_stage_acquire(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t slot_idx,
    size_t required_size);

GGML_API void ggml_backend_expert_cache_stage_commit(
    ggml_backend_expert_cache_t cache,
    const struct ggml_tensor * tensor,
    int32_t slot_idx);
```

- [ ] **Step 5: Add ring state to the cache struct**

In `ggml/src/ggml-backend-expert-cache.cpp`, inside `struct ggml_backend_expert_cache` (after the `pinned_host_capacity` field, ~line 153):

```cpp
    // Phase 5b: staging ring keyed by (tensor, slot_idx); waits before reuse
    std::vector<ggml_backend_event_t> staging_events;
    std::vector<bool>                 staging_in_flight;
```

Add the macro near the top (next to `GGML_EXPERT_CACHE_ALIGN`, line 18):

```cpp
#define GGML_EXPERT_CACHE_STAGING_ENTRIES 32
```

- [ ] **Step 6: Implement stage_acquire/stage_commit and remove get_pinned_slot_buffer**

In `ggml/src/ggml-backend-expert-cache.cpp`, replace the whole `ggml_backend_expert_cache_get_pinned_slot_buffer` function (lines ~1162-1179) with:

```cpp
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
```

- [ ] **Step 7: Free staging events in cache free**

In `ggml_backend_expert_cache_free` (after the `pinned_host_buffer` free block, ~line 370):

```cpp
    for (auto & ev : cache->staging_events) {
        if (ev != NULL) {
            ggml_backend_event_free(ev);
        }
    }
    cache->staging_events.clear();
```

- [ ] **Step 8: Update the compute_splits caller**

In `ggml/src/ggml-backend.cpp:1838-1854`, replace the staged-miss block with:

```cpp
                                            void * pinned_buf = ggml_backend_expert_cache_stage_acquire(cache, input, slot, expert_size);
                                            if (pinned_buf != NULL) {
                                                memcpy(pinned_buf, src_ptr, expert_size);
                                                ggml_backend_expert_cache_record_staging_memcpy(cache, expert_size);
                                                ggml_backend_tensor_set_async(split_backend,
                                                    slot_tensor,
                                                    pinned_buf,
                                                    dst_offset,
                                                    expert_size);
                                                ggml_backend_expert_cache_stage_commit(cache, input, slot);
                                            } else {
                                                ggml_backend_tensor_set_async(split_backend,
                                                    slot_tensor,
                                                    src_ptr,
                                                    dst_offset,
                                                    expert_size);
                                            }
```

- [ ] **Step 9: Update the prefetch caller**

In `ggml_backend_expert_cache_prefetch` (lines ~1210-1218), replace the pinned-buffer block with:

```cpp
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
```

- [ ] **Step 10: Run the test to verify it passes**

Run: `cmake --build build --config Release --target test-expert-cache`
Then: `build/bin/Release/test-expert-cache.exe`
Expected: PASS, including `pinned staging ring tests passed`.

- [ ] **Step 11: Commit (with user approval)**

```bash
git add ggml/include/ggml-backend.h ggml/src/ggml-backend-expert-cache.h ggml/src/ggml-backend-expert-cache.cpp ggml/src/ggml-backend.cpp tests/test-expert-cache.cpp
git commit -m "ggml : fix expert cache pinned staging buffer race

Assisted-by: <assistant name>"
```

---

### Task 2: Seed Profiles Descending by Frequency

**Files:**
- Modify: `common/expert-cache-profile.h` (add sort helper declaration)
- Modify: `common/expert-cache-profile.cpp` (extract sort helper; loader refactor at ~83-146)
- Modify: `tests/test-expert-cache-profile.cpp` (new test + seed assertions)
- Test: `tests/test-expert-cache-profile.cpp`

**Interfaces:**
- Consumes: `common_expert_cache_profile_entry` (already public: `tensor_name` string, `expert_id` int32, `frequency` uint32, `hit_count` uint64), `llama_model_get_tensor`.
- Produces:
  - `void common_expert_cache_sort_entries(std::vector<common_expert_cache_profile_entry> & entries);` (new public helper)
  - Sorted/admitted profile seed order is now descending frequency.

**Why:** `common_expert_cache_load_profile` sorts seed entries by `(tensor, expert_id)` then by frequency ascending (expert-cache-profile.cpp:128-136). `ggml_backend_expert_cache_seed` admits flat-cache entries until capacity is full, so when capacity is limited the LOWEST-frequency experts are admitted and the hottest are rejected. Descending order makes the hottest experts resident first.

- [ ] **Step 1: Write the failing test**

Append to `tests/test-expert-cache-profile.cpp` (before `main()`):

```cpp
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
```

And add find_slot assertions to `test_expert_cache_seed_and_export` (after the existing `find_offset` checks, ~line 74-82):

```cpp
    // the zero-copy decode path must find seeded experts in the slot pool
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 2) >= 0);
    assert(ggml_backend_expert_cache_find_slot(cache, tensor, 5) >= 0);
```

Call the new test from `main()`:

```cpp
    test_path_resolution();
    test_expert_cache_seed_and_export();
    test_json_profile_io();
    test_profile_sort_order();
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --config Release --target test-expert-cache-profile`
Then: `build/bin/Release/test-expert-cache-profile.exe`
Expected: compile error - `common_expert_cache_sort_entries` not declared.

- [ ] **Step 3: Declare the helper**

In `common/expert-cache-profile.h` (after `common_expert_cache_get_file_path`, ~line 25):

```cpp
// Sort seed entries for cache admission: deduplicate by (tensor_name, expert_id)
// keeping the maximum frequency, then order highest frequency first so hot
// experts are admitted before cache capacity fills.
void common_expert_cache_sort_entries(std::vector<common_expert_cache_profile_entry> & entries);
```

- [ ] **Step 4: Implement the helper and refactor the loader**

In `common/expert-cache-profile.cpp`, replace the local `expert_cache_seed_entry` struct usage and the two sort blocks (lines ~83-136) with:

```cpp
    std::vector<common_expert_cache_profile_entry> entries;
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
        const uint64_t hit_count = item.value("hit_count", (uint64_t) 0);

        entries.push_back({ tensor_name, expert_id, frequency, hit_count });
    }

    common_expert_cache_sort_entries(entries);
```

Add the helper implementation before `common_expert_cache_load_profile` (after `common_expert_cache_get_file_path`):

```cpp
void common_expert_cache_sort_entries(std::vector<common_expert_cache_profile_entry> & entries) {
    // group duplicates by (tensor_name, expert_id), keep the maximum frequency
    std::sort(entries.begin(), entries.end(), [](const common_expert_cache_profile_entry & a, const common_expert_cache_profile_entry & b) {
        if (a.tensor_name != b.tensor_name) {
            return a.tensor_name < b.tensor_name;
        }
        if (a.expert_id != b.expert_id) {
            return a.expert_id < b.expert_id;
        }
        return a.frequency < b.frequency;
    });

    std::vector<common_expert_cache_profile_entry> merged;
    merged.reserve(entries.size());
    for (const auto & entry : entries) {
        if (!merged.empty() &&
            merged.back().tensor_name == entry.tensor_name &&
            merged.back().expert_id   == entry.expert_id) {
            merged.back().frequency = std::max(merged.back().frequency, entry.frequency);
            merged.back().hit_count = std::max(merged.back().hit_count, entry.hit_count);
        } else {
            merged.push_back(entry);
        }
    }

    // highest frequency first: hot experts are admitted before capacity fills
    std::sort(merged.begin(), merged.end(), [](const common_expert_cache_profile_entry & a, const common_expert_cache_profile_entry & b) {
        if (a.frequency != b.frequency) {
            return a.frequency > b.frequency;
        }
        if (a.tensor_name != b.tensor_name) {
            return a.tensor_name < b.tensor_name;
        }
        return a.expert_id < b.expert_id;
    });

    entries = std::move(merged);
}
```

Replace the seed loop (lines ~139-146) - it still iterates `entries` in order, but now resolves tensors and validates:

```cpp
    std::vector<bool> seeded(entries.size(), false);
    for (int b = 0; b < ggml_backend_sched_get_n_backends(sched); b++) {
        for (size_t i = 0; i < entries.size(); i++) {
            const auto & entry = entries[i];
            const struct ggml_tensor * tensor = llama_model_get_tensor(model, entry.tensor_name.c_str());
            if (tensor == nullptr || tensor->ne[2] <= 1 || entry.expert_id < 0 || entry.expert_id >= tensor->ne[2]) {
                if (!seeded[i]) {
                    skipped_count++;
                    seeded[i] = true; // count once, never re-count in later backend passes
                }
                continue;
            }
            seeded[i] = ggml_backend_sched_expert_cache_seed(
                sched, b, tensor, entry.expert_id, entry.frequency) || seeded[i];
        }
    }
```

Note: the duplicate-invalid-entries counting changes slightly (an invalid entry is counted once, on its first backend pass). This matches the existing behavior test in the optimization log (`skipped 2 invalid profile entries`).

- [ ] **Step 5: Run the test to verify it passes**

Run: `cmake --build build --config Release --target test-expert-cache-profile`
Then: `build/bin/Release/test-expert-cache-profile.exe`
Expected: PASS, including `profile sort order tests passed`.

- [ ] **Step 6: Commit (with user approval)**

```bash
git add common/expert-cache-profile.h common/expert-cache-profile.cpp tests/test-expert-cache-profile.cpp
git commit -m "common : seed expert cache profiles by frequency descending

Assisted-by: <assistant name>"
```

---

### Task 3: Per-Layer Timing Probe in compute_splits

**Files:**
- Modify: `ggml/include/ggml-backend.h:364-383` (stats struct)
- Modify: `ggml/src/ggml-backend.cpp:1731-1876` (probe accumulation in the cache interception block)

**Interfaces:**
- Consumes: cache flags from Task 1 (`cache` is non-NULL only when a cache intercepts the split), existing `cache->stats` (already aggregated by `ggml_backend_sched_get_expert_cache_stats`).
- Produces (no behavior change; telemetry only):
  - `uint64_t probe_n_layers` - splits where a cache intercepted an input
  - `uint64_t probe_sync_us` - host time in ids D2H + backend syncs
  - `uint64_t probe_host_us` - host time in remap/admission loop
  - `uint64_t probe_upload_us` - host time issuing uploads (remapped ids + miss DMA)

**Why:** The Phase 1 decision (device-side id remap vs copy stream) depends on which cost dominates per-layer when the cache is active. The audits deferred this measurement; this task makes it print via the existing stats path so llama-bench JSONL and `-excs` expose it.

- [ ] **Step 1: Add the probe fields**

In `ggml/include/ggml-backend.h`, inside `struct ggml_backend_expert_cache_stats` (after `n_staging_waits`):

```c
        // V2.3 Probe: per-layer host-time breakdown when a cache intercepts a split
        uint64_t probe_n_layers;
        uint64_t probe_sync_us;
        uint64_t probe_host_us;
        uint64_t probe_upload_us;
```

- [ ] **Step 2: Accumulate at the three serialization points**

In `ggml/src/ggml-backend.cpp`, inside `ggml_backend_sched_compute_splits`, in the `if (cache_can_store)` block:

Wrap the ids sync section (currently lines 1731-1750). Change:

```cpp
                    ggml_backend_synchronize(input_backend);

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;
```

to:

```cpp
                    const int64_t t_sync_start = ggml_time_us();
                    ggml_backend_synchronize(input_backend);

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;
```

and after the `ggml_backend_synchronize(ids_backend);` at line 1750 add:

```cpp
                    cache->stats.probe_sync_us += ggml_time_us() - t_sync_start;
```

Wrap the host decision section (from the `{` at line 1767 through the end of the zero-copy block at line 1878). Change the block opening at line 1767 from:

```cpp
                    {
                        expert_counts.assign(n_expert, 0);
```

to:

```cpp
                    {
                        const int64_t t_host_start = ggml_time_us();
                        expert_counts.assign(n_expert, 0);
```

Add the upload-issue bucket just before the `node->src[0] = slot_tensor;` swap and the `used_zero_copy = true;` (line ~1874):

```cpp
                                cache->stats.probe_upload_us +=
                                    ggml_time_us() - t_host_start - t_upload_start;
```

and before the `if (all_slots_ready)` block (line 1862), record the host-and-upload split:

```cpp
                            if (all_slots_ready) {
                                const int64_t t_upload_start = ggml_time_us();
                                cache->stats.probe_host_us +=
                                    (t_upload_start - t_host_start);
                                cache->stats.probe_n_layers++;
```

and add the `t_host_start`/`t_upload_start` declarations where the zero-copy path begins (after `const bool is_single_token_decode = (ids_tensor->ne[1] == 1);`, line 1795):

```cpp
                        int64_t t_host_start = 0;
                        int64_t t_upload_start = 0;
```

The `probe_host_us`/`probe_upload_us` expressions reference local `t_host_start`/`t_upload_start`; declare them at the top of the `if (cache_can_store)` block next to `t_sync_start` instead to avoid scope issues, and set `t_host_start = ggml_time_us();` at line 1767 where the host work begins:

```cpp
                    const int64_t t_sync_start = ggml_time_us();
                    int64_t t_host_start = 0;
                    int64_t t_upload_start = 0;
```

**Field accounting (documented, not asserted):** `probe_sync_us` covers `synchronize(input_backend)` + ids D2H + `synchronize(ids_backend)`; `probe_host_us` covers the counts/remap/admission work before the uploads; `probe_upload_us` covers issuing the remapped-ids H2D and miss H2D copies. The fallback (prefill) path is not probed - zeros stay. The `!used_zero_copy` branch remains unprobed (PP path); this probe targets the TG decode path only.

- [ ] **Step 3: Verify it builds and prints**

Run: `cmake --build build --config Release --target llama-bench llama-server`
Expected: build succeeds; fields are zero-added (no functional change). Verify exposure with `-excs` on a server run with `-exc 256M` (fields print as 0 when the cache never intercepts decode - expected today).

- [ ] **Step 4: Commit (with user approval)**

```bash
git add ggml/include/ggml-backend.h ggml/src/ggml-backend.cpp
git commit -m "ggml : add expert cache per-layer timing probe

Assisted-by: <assistant name>"
```

---

### Task 4: Rebuild, Unit Tests, Deterministic Matrix, Benchmarks, Log Update

**Files:**
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` (append Phase 0 results section)
- Test (executed, not code): `build/bin/Release/test-expert-cache.exe`, `test-expert-cache-profile.exe`, `llama-server`, `llama-bench`

**Interfaces:**
- Consumes: Tasks 1-3; the model `Qwen3.6-35B-A3B-APEX-Compact.gguf`; existing benchmark config from the log (batch 4096, ubatch 2048, 14 threads, q8_0 KV, flash-attn, mlock, fit-target 256, ffn-split 0).
- Produces: recorded determinism hash + PP/TG numbers appended to the optimization log.

**Why:** Verify Phase 0 changed no observable behavior (determinism, PP/TG throughput) while fixing correctness, and record the probe baseline for the Phase 1 decision.

- [ ] **Step 1: Build everything**

```bash
cmake --build build --config Release --target test-expert-cache test-expert-cache-profile llama-bench llama-server
```

- [ ] **Step 2: Run unit tests**

```bash
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-expert-cache-profile.exe
```

Expected: both exit 0 with the "all tests passed" banner.

- [ ] **Step 3: Deterministic matrix (temperature 0, top-k 1, seed 42, fixed prompt)**

Fresh `llama-server` per row; capture the 256-token output hash (the log's reference hash is `580b417f73e4d58b209b44e5f07ccc269900d4b0d9d5318e8866f1d6f1335fe8` for `-exc 64M`; `-exc 0` must match the current baseline hash, which was recorded in earlier rows):

- Row A: `-exc 0` (no cache) - record hash
- Row B: `-exc 64M -excp 64` - hash must equal Row A
- Row E: `-exc 64M -excp 64 --draft-mtp --mtp-dynamic-offload` (on the MTP-Quality model if that model is used; otherwise skip) - coherent output, hash must equal prior Row E hash

Expected: A == B hashes; E matches its prior recorded hash.

- [ ] **Step 4: Benchmark plain (no cache) vs cache**

```bash
build/bin/Release/llama-bench -m <Compact.gguf> -p 512 -n 128 -r 5 -exc 0 -fitt 256 -b 4096 -ub 2048 -t 14 -ctk q8_0 -ctv q8_0 -fa -mlock
build/bin/Release/llama-bench -m <Compact.gguf> -p 512 -n 128 -r 5 -exc 256M -excp 512 -fitt 256 -b 4096 -ub 2048 -t 14 -ctk q8_0 -ctv q8_0 -fa -mlock
```

Expected: PP512 within noise of 467-468 tok/s; TG128 within noise of 26.5 tok/s (no regression from Tasks 1-3).

- [ ] **Step 5: Record probe baseline (informational)**

Run a server row with `-exc 256M -excs` and note whether `probe_*` counters move. Expected today: they stay 0 (cache is not intercepting TG decode). This is the baseline the Phase 1 decision uses.

- [ ] **Step 6: Append results to the optimization log**

Append a "Phase 0 (2026-08-21): staging-ring fix + descending seed + probe" section to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` with the four unit-test results, determinism hashes, PP/TG numbers, and probe baseline. Follow the log's existing table style.

- [ ] **Step 7: Commit (with user approval)**

```bash
git add EXPERT_CACHE_OPTIMIZATIONS_LOG.md
git commit -m "docs : record expert cache phase 0 verification results

Assisted-by: <assistant name>"
```

---

### Task 5 (Optional, gated by user): Cache-Active TG Probe Run

**Files:**
- Temporary experiment only; leave no code behind after the measurement.
- Modify (temporary): `ggml/src/ggml-backend.cpp` split-graph assignment, to force host-expert `GGML_OP_MUL_MAT_ID` splits onto the CUDA backend when a cache is configured (mirror of the reverted 2026-08-21 experiment, now with the staging fix + probe from Tasks 1-3).

**Interfaces:**
- Consumes: Task 3 probe fields; the reverted forced-routing experiment described in the optimization log.
- Produces: `probe_sync_us` vs `probe_host_us` vs `probe_upload_us` fractions on the active-cache TG path - the Phase 1 (S9 device remap vs S2 copy stream) decision input.

**Why:** The Phase 1 choice depends on which cost dominates per layer when the cache is active. This is the only way to measure it. It is gated because it re-opens a knowingly slower configuration for measurement only.

- [ ] **Step 1: Re-apply the forced-routing experiment (with user confirmation)**
- [ ] **Step 2: Run TG128 with `-exc 256M` and `-excs`; record probe fractions and tok/s**
- [ ] **Step 3: Revert the experimental routing change; rebuild; confirm unit tests still pass**

---

## Self-Review

**Spec coverage:** The review recommendation covered exactly these items: S3 (pinned staging race) -> Task 1; S1 (seed order) -> Task 2; per-layer timing probe -> Task 3; rebuild + deterministic matrix + bench -> Task 4; cache-active probe measurement -> Task 5 (gated). The rebalance/flat-slot divergence, transition predictor, and bundle atomicity (S4/S5/S6) were deferred by the decision - correctly out of scope for "Phase 0".

**Placeholder scan:** No TBD/TODO; every code step has literal code. Optional Task 5 references "mirror of the reverted experiment" - acceptable because the experiment is explicitly documented in `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` (2026-08-21 section) and gated by user confirmation; its steps are marked as requiring the reverted diff, not invented code.

**Type consistency:** `stage_acquire`/`stage_commit` signatures identical across Task 1 files; `n_staging_waits`, `probe_n_layers`, `probe_sync_us`, `probe_host_us`, `probe_upload_us` names consistent between `ggml-backend.h` and `ggml-backend.cpp`; `common_expert_cache_sort_entries` signature identical in header/test/impl; `common_expert_cache_profile_entry` field order `(tensor_name, expert_id, frequency, hit_count)` is consistent between the existing struct and the Task 2 test initializers.

**Known caveats documented in the plan:** Task 1's ring has `GGML_EXPERT_CACHE_STAGING_ENTRIES = 32`; when >32 distinct `(tensor, slot_idx)` pairs are staged before DMA completes (rare; in-flight window is ~1-2 layers), an entry collision waits on the previous DMA - correct, occasionally stalling. Task 3's probe covers only the zero-copy decode path; the prefill fallback stays unprobed.