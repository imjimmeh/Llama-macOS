# Implementation Plan: Persistent, Scalable N-Gram Speculative Memory

This plan implements the epic at `docs/superpowers/plans/2026-08-31-persistent-ngram-mod-speculative-memory.md`.

## Baseline Facts (from codebase inspection)

| Property | Current value |
|---|---|
| Entry type | `int32_t` (= `llama_token`), `EMPTY = -1` |
| Table storage | `std::vector<int32_t>`, 4M slots = 16 MB |
| Table size | Hardcoded: `4*1024*1024` at `speculative.cpp:1853` |
| Hash function | `res = res * 6364136223846793005ULL + tokens[i]`, mod table size |
| Collision handling | None (last-write-wins overwrite) |
| Hash bits stored | None - only `next_token` per slot |
| Thread safety | Single-threaded by construction (inference loop) |
| Adaptation | Reset on >25% occupancy or 5 consecutive low-acceptance rounds |
| Persistence | None |
| Size parsing | `arg.cpp:2750-2775` - K/M/G/T suffix parser (from expert-cache) |
| Model size API | `llama_model_size()` - tensor payload bytes |
| Tokenizer identity | `llama_model_meta_val_str()` - GGUF keys, no fingerprinting |
| Shared pool | Yes, across all server slots |
| CLI options | `--spec-ngram-mod-n-match/n-max/n-min` only |

**Critical constraint**: The current entry stores ONLY `next_token` (4 bytes). No hash bits are retained. This means:
- Rehashing on growth is impossible without changing the entry format
- Collision detection is impossible
- The entry format MUST change to support scalability

---

## Stage 0: Baseline Investigation (complete - this document)

Already done. Key answers to epic research questions:

1. **Entry representation**: `int32_t` per slot (just `next_token`). No fingerprints.
2. **16 MB reason**: Hardcoded `4*1024*1024` slots in constructor. Not compile-time, but not configurable either.
3. **Size derivation**: Runtime literal. Not derived from model or any config.
4. **Slot content**: Only `next_token`. Not a full hash.
5. **Rehashable**: No. Original hash bits are lost after insertion.
6. **Collision behaviour**: Silent overwrite. Last write wins.
7. **Collision measurement**: Not currently measurable.
8. **Synchronisation**: None needed - single inference thread by construction.
9. **Concurrent copy safety**: Safe because single-threaded.
10. **Binary persistence**: `ngram-cache` has raw binary dump (no header/version/checksum). Reusable pattern but format is unsuitable.
11. **Model/vocab identity**: `llama_model_meta_val_str()` for GGUF keys. `tokenizer.ggml.model` and `tokenizer.ggml.pre` are the relevant keys.
12. **Model bytes**: `llama_model_size()` returns tensor payload bytes (excludes GGUF header/metadata).
13. **CLI size parsing**: Yes, `arg.cpp:2750-2775` already handles K/M/G/T.
14. **SPEED-Bench metrics**: Yes, `common_speculative_print_stats()` already reports drafts/tokens/acceptance.
15. **Concurrency claim**: All speculative calls run on the inference loop thread.

---

## Stage 1: Configurable Entry Format + Capacity

**Goal**: Change the entry from bare `next_token` to a fingerprinted entry that supports future rehashing, then make the table size runtime-configurable.

### Task 1.1: New entry format with fingerprint

**Files**: `common/ngram-mod.h`, `common/ngram-mod.cpp`

Replace:
```cpp
using entry_t = int32_t;
static constexpr entry_t EMPTY = -1;
```

With a packed entry:
```cpp
struct ngram_mod_slot {
    uint32_t fingerprint;  // upper 32 bits of hash for collision detection
    int32_t  next_token;   // predicted token, EMPTY = -1

    bool is_empty() const { return next_token == EMPTY; }
    bool matches(uint32_t fp) const { return fingerprint == fp && !is_empty(); }
};

static constexpr int32_t EMPTY = -1;
```

This doubles entry size from 4 to 8 bytes per slot. 4M slots = 32 MB. The hash function currently computes `size_t res`; split into `fingerprint = (uint32_t)(res >> 32)` (or just the lower 32 bits) and use the modular index as before.

**Impact on existing behaviour**: The `idx()` function must be updated to also compute/store fingerprints. `get()` must check fingerprint before returning. `add()` must store both fingerprint and token. The `used` counter must use `is_empty()`.

**Verification**: Build, run `test-ngram-mod` (if it exists) or create a basic unit test. Verify existing spec-type ngram-mod behaviour is unchanged at 16 MB default.

### Task 1.2: Make table size configurable via CLI

**Files**: `common/common.h`, `common/arg.cpp`, `common/speculative.cpp`

Add to `common_params_speculative_ngram_mod`:
```cpp
size_t pool_size_bytes = 0; // 0 = default (16 MB)
```

Add CLI option in `arg.cpp`:
```cpp
{"--spec-ngram-mod-size"}, "SIZE",
"size of ngram-mod hash pool (e.g. 16M, 256M, 1G; default: 16M)"
```

Reuse the K/M/G/T parsing pattern from `arg.cpp:2750-2775`. Convert size to slot count: `slots = size_bytes / sizeof(ngram_mod_slot)`.

Update constructor at `speculative.cpp:1853`:
```cpp
// Before: mod(params.ngram_mod.n_match, 4*1024*1024)
// After:
size_t n_slots = params.ngram_mod.pool_size_bytes > 0
    ? params.ngram_mod.pool_size_bytes / sizeof(ngram_mod_slot)
    : 4 * 1024 * 1024;  // default ~32 MB with new entry format
```

**Verification**: `llama-server --spec-type ngram-mod --spec-ngram-mod-size 64M` allocates ~8M slots (64 MB). Default unchanged. Benchmark at 16M, 64M, 256M, 1G.

### Task 1.3: Add occupancy and collision telemetry

**Files**: `common/ngram-mod.h/.cpp`, `common/speculative.cpp`

Add counters to `common_ngram_mod`:
```cpp
size_t n_lookups = 0;
size_t n_hits = 0;       // get() returned non-empty AND fingerprint matched
size_t n_collisions = 0;  // slot occupied but fingerprint mismatch (overwrite happened)
size_t n_inserts = 0;
size_t n_overwrites = 0;  // add() replaced a non-empty slot
```

Add `get_with_stats()` or update `get()` to increment counters. Expose via `get_stats()` struct.

Update `common_speculative_print_stats()` in `speculative.cpp:2767+` to report these.

**Verification**: Run with `LLAMA_TRACE=1`, observe telemetry output.

---

## Stage 2: Percentage-Based Capacity

### Task 2.1: Add `--spec-ngram-mod-size N%` support

**Files**: `common/common.h`, `common/arg.cpp`, `common/speculative.cpp`

Extend the size parser to detect `%` suffix. If percentage, store as `double percentage = 0.0` in params.

In `speculative.cpp`, at init time (where model is available), resolve:
```cpp
double model_bytes = llama_model_size(model);
size_t resolved_size = (size_t)(model_bytes * percentage);
```

**Problem**: The `common_speculative_impl_ngram_mod` constructor currently does NOT receive the model pointer. Need to either:
- (a) Pass model size into the params struct before init, or
- (b) Add model_size to `common_params_speculative` set by server at startup.

Option (b) is cleaner. Add `size_t model_weight_bytes = 0` to `common_params_speculative`. Server sets it after loading the model.

**Verification**: `--spec-ngram-mod-size 10%` with a 24 GB model allocates ~2.4 GB.

### Task 2.2: Validate model-size calculation

Confirm that `llama_model_size()` returns the correct value for multi-split GGUF files. Test with the Qwen3.6-35B model.

---

## Stage 3: Persistence Format + Save/Load

### Task 3.1: Cache file format design

**Files**: New file `common/ngram-mod-cache.h` and `common/ngram-mod-cache.cpp`

Header:
```cpp
struct ngram_mod_cache_header {
    uint32_t magic;          // 0x4E474D44 = "NGMD"
    uint32_t version;        // 1
    uint32_t header_size;    // sizeof(header)
    uint32_t slot_size;      // sizeof(ngram_mod_slot)

    uint64_t capacity;       // number of slots
    uint64_t entry_count;    // non-empty slots

    uint32_t n_match;        // n-gram length used

    // Tokenizer/vocab compatibility
    uint32_t vocab_size;
    char     tokenizer_model[64];  // tokenizer.ggml.model
    char     tokenizer_pre[64];    // tokenizer.ggml.pre

    // Timestamps
    uint64_t created_ts;
    uint64_t saved_ts;

    // Stats
    uint64_t total_inserts;
    uint64_t total_lookups;
    uint64_t total_hits;
};
```

Payload: raw `ngram_mod_slot[capacity]` array, but ONLY non-empty slots can be saved. Two options:
- **(a) Save full array**: Simple, fast, large file. A 4 GB pool = 4 GB file.
- **(b) Save sparse**: Only non-empty slots (8 bytes each + index). More complex but much smaller.

**Decision**: Start with (a) - full array. Rationale:
- Simpler implementation
- Load is O(1) with a single memcpy/mmap
- For benchmarking correctness is paramount
- Sparse format is an optimization to add later if file size matters

The full array for a 20% cache on a 24 GB model = 4.8 GB file. Acceptable on SSD.

### Task 3.2: Checksum

Add xxHash3 (non-cryptographic, fast) checksum of the payload after writing. Store in header or append after payload. Verify on load.

Rationale: xxHash is already used in the ggml ecosystem. Full-payload hash of 4 GB takes ~100 ms on modern CPUs, acceptable at startup.

### Task 3.3: Tokenizer fingerprint

Compute a simple fingerprint from GGUF metadata:
```cpp
// Combine vocab_size + tokenizer.ggml.model + tokenizer.ggml.pre into a stable hash
// xxHash3 of: vocab_size || tokenizer_model_string || tokenizer_pre_string
```

Store in header. Compare on load.

### Task 3.4: Save function

**File**: `common/ngram-mod-cache.cpp`

```cpp
bool ngram_mod_cache_save(const ngram_mod_slot * slots, size_t capacity,
                          const ngram_mod_cache_header & meta,
                          const std::string & path);
```

Write path (crash-safe):
1. Open `path + ".tmp"`
2. Write header
3. Write full slot array
4. Compute and append checksum
5. Flush (`fflush` + `fsync` on POSIX, `FlushFileBuffers` on Windows)
6. Close
7. Atomic rename: `rename(path + ".tmp", path)` (POSIX) or `MoveFileEx` with `MOVEFILE_REPLACE_EXISTING` (Windows)

### Task 3.5: Load function

```cpp
bool ngram_mod_cache_load(const std::string & path,
                          ngram_mod_slot * & slots, size_t & capacity,
                          ngram_mod_cache_header & meta);
```

Validation order:
1. File exists and is readable
2. Read header, check magic, version, header_size
3. Check `slot_size == sizeof(ngram_mod_slot)` (format compatibility)
4. Check `capacity` is sane (non-zero, not absurdly large, multiple of power-of-two reasonable)
5. Check `n_match` matches current config
6. Check tokenizer fingerprint compatibility
7. Allocate `capacity * sizeof(ngram_mod_slot)` bytes
8. Read slot array
9. Verify checksum
10. If any check fails: log warning, return false, caller falls back to empty pool

**Security**: `capacity` field is attacker-controlled (untrusted file). Must bounds-check before allocation: `capacity <= MAX_REASONABLE_SLOTS` where MAX = e.g. `256 * 1024 * 1024` (2 GB of slots). Also check `capacity * sizeof(slot)` doesn't overflow.

### Task 3.6: Wire into `common_speculative_impl_ngram_mod`

**File**: `common/speculative.cpp`

Add to params:
```cpp
std::string cache_path;           // --spec-ngram-mod-cache PATH
int save_interval_sec = 0;        // --spec-ngram-mod-save-interval N (0 = only on shutdown)
```

Modify constructor:
1. If `cache_path` is non-empty and file exists: attempt load
2. If load succeeds: use loaded slots and capacity
3. If load fails or no file: allocate fresh pool

Add destructor:
```cpp
~common_speculative_impl_ngram_mod() {
    if (!cache_path.empty()) {
        ngram_mod_cache_save(mod.slots(), mod.size(), build_meta(), cache_path);
    }
}
```

Add periodic save (called from `process()` which is already called per-batch):
```cpp
bool process(const llama_batch & batch) override {
    // ... existing no-op ...
    // periodic save check
    if (save_interval_sec > 0 && dirty) {
        auto now = time(nullptr);
        if (now - last_save_ts >= save_interval_sec) {
            ngram_mod_cache_save(...);
            last_save_ts = now;
            dirty = false;
        }
    }
    return true;
}
```

**Verification**:
1. Start server with `--spec-ngram-mod-cache ./test.ngrammod`
2. Process some tokens
3. Stop server (graceful shutdown saves)
4. Restart with same options - confirm log shows "loaded N entries"
5. Verify speculative behaviour matches

---

## Stage 4: Crash-Safe Checkpointing + Periodic Save

### Task 4.1: Dirty tracking

Add `dirty` flag to `common_ngram_mod`. Set `dirty = true` on every `add()`. Clear after save.

### Task 4.2: Periodic save implementation

Already described in Task 3.6. Add save-interval option.

### Task 4.3: Checkpoint telemetry

On startup, log:
```
ngram-mod cache: loaded from ./test.ngrammod
  format version: 1
  capacity: 536870912 slots (4.000 GB)
  entries: 1234567
  tokenizer: compatible (qwen2.5)
  load time: 234 ms
```

On save, log:
```
ngram-mod cache: saved to ./test.ngrammod
  size: 4.000 GB
  entries: 1234567
  duration: 89 ms
```

**Verification**: Benchmark checkpoint time for 256 MB, 1 GB, 4 GB pools. Must be < 500 ms for 4 GB to avoid unacceptable stalls.

---

## Stage 5: Extended Telemetry

### Task 5.1: Per-lookup stats

Track in `common_ngram_mod`:
- `n_lookups` (every `get()` call)
- `n_hits` (get returned non-empty AND fingerprint matched)
- `n_miss_fingerprint` (slot occupied but fingerprint mismatched)
- `n_inserts` (every `add()` call)
- `n_overwrites` (add replaced a non-empty slot)

### Task 5.2: Derived metrics

Compute and report:
- Hit rate = `n_hits / n_lookups`
- Overwrite rate = `n_overwrites / n_inserts`
- Occupancy = `used / capacity`

### Task 5.3: Export to llama-bench

Add new columns to `tools/llama-bench/llama-bench.cpp`:
- `ngram_mod_hit_rate`
- `ngram_mod_overwrite_rate`
- `ngram_mod_occupancy`

Follow the pattern of existing `expert_cache_*` columns in llama-bench.

### Task 5.4: Export to server Prometheus metrics

Add to `tools/server/server-task.cpp`:
- `ngram_mod_hit_rate` gauge
- `ngram_mod_occupancy` gauge

---

## Stage 6: Dynamic Growth Decision Gate

**Before implementing growth, run the benchmark matrix (Stage 5) and evaluate.**

Benchmark matrix:
```
No spec decoding
ngram-mod 16 MB (current default)
ngram-mod 64 MB
ngram-mod 256 MB
ngram-mod 1 GB
ngram-mod 5% model size
ngram-mod 10% model size
ngram-mod 20% model size
```

For each: cold start + warm start (with persistence).

**Decision gate**: If performance continues improving through 1 GB+ without latency regression, proceed to Stage 7. If it plateaus at 256 MB, stop and document the optimum.

---

## Stage 7: Dynamic Growth (conditional on Stage 6 results)

### Task 7.1: Growth params

Add to `common_params_speculative_ngram_mod`:
```cpp
bool grow = false;                  // --spec-ngram-mod-grow
size_t max_size_bytes = 0;         // --spec-ngram-mod-max-size SIZE (0 = no max beyond system)
```

### Task 7.2: Growth trigger

Measure in `add()`: if `n_overwrites / n_inserts > threshold` (e.g. 0.3) over a window of N inserts, and accepted prediction rate > some threshold, and current size < max: grow.

Geometric growth: double the pool size each time.

### Task 7.3: Rehash implementation

Because we now have `fingerprint` stored in each slot:
1. Allocate new larger pool
2. For each non-empty slot in old pool: recompute `idx()` with new mask, insert into new pool
3. Brief pause (no new inserts during rehash)
4. Swap pointer
5. Free old pool

Since growth is rare (maybe a few times during a session) and the pool is single-threaded, a brief pause is acceptable.

### Task 7.4: Persist grown size

After growth, save includes the new capacity. On next restart with `--spec-ngram-mod-size auto`:
- If valid persisted cache exists: use persisted capacity (subject to max)
- If no cache: use initial configured size

Add `--spec-ngram-mod-size auto` option.

### Task 7.5: Concurrent checkpoint during growth

Not needed initially. Growth is rare and single-threaded. Just pause speculation briefly.

---

## Stage 8: Future Work (not in initial scope)

- Corpus pre-seeding tool (`llama-ngram-mod-build`)
- Admission policy (probationary table)
- Richer confidence metadata per entry
- Multiple continuation candidates
- Multi-head hashing
- Sparse cache file format (for large pools)

Done in follow-up work:

- mmap-backed persistence and hot/cold tiering: implemented in
  `docs/superpowers/plans/2026-08-31-persistent-ngram-mod-hot-cold-tiering.md`

---

## New CLI Options Summary

| Option | Type | Default | Stage |
|---|---|---|---|
| `--spec-ngram-mod-size SIZE` | SIZE (K/M/G/T/%) | 32M | 1 |
| `--spec-ngram-mod-cache PATH` | string | "" | 3 |
| `--spec-ngram-mod-save-interval N` | seconds | 0 | 4 |
| `--spec-ngram-mod-max-size SIZE` | SIZE (K/M/G/T/%) | 0 | 7 |
| `--spec-ngram-mod-grow` | flag | false | 7 |

## New Files

| File | Purpose | Stage |
|---|---|---|
| `common/ngram-mod-cache.h` | Persistence format + save/load declarations | 3 |
| `common/ngram-mod-cache.cpp` | Persistence implementation | 3 |

## Modified Files

| File | Changes | Stage |
|---|---|---|
| `common/ngram-mod.h` | New `ngram_mod_slot` struct, stats counters, capacity API | 1 |
| `common/ngram-mod.cpp` | Updated hash/lookup/insert with fingerprints, stats | 1 |
| `common/common.h` | New params fields (pool_size, cache_path, save_interval, etc.) | 1-7 |
| `common/arg.cpp` | New CLI options | 1-7 |
| `common/speculative.cpp` | Configurable capacity, persistence wiring, telemetry | 1-5 |
| `tools/llama-bench/llama-bench.cpp` | New benchmark columns | 5 |
| `tools/server/server-task.cpp` | Prometheus metrics | 5 |

## Backwards Compatibility

Default `--spec-type ngram-mod` with no new options MUST produce equivalent behaviour to current upstream. The new 8-byte entry format changes the internal representation, but observable behaviour (speculative draft quality, acceptance rate) should be identical at 16 MB default.

The one observable change: default pool is now 32 MB (4M slots * 8 bytes) instead of 16 MB (4M slots * 4 bytes), because we doubled the entry size. To preserve exact 16 MB default, either:
- (a) Reduce slot count to 2M (but this halves capacity)
- (b) Accept 32 MB as the new default (recommended - it's a benign improvement)

**Recommendation**: Accept 32 MB default. Document in changelog. If upstream maintainers object, it's trivial to reduce slot count.

## Verification Results (Stages 1-4)

Tested on: Qwen3.5-4B Q4_K_M (2.6 GB), CPU-only, GTX 1080 + Ryzen 7 5700X.

| Test | Result |
|------|--------|
| `--spec-ngram-mod-size 64M` | 8M slots (64 MB confirmed) |
| `--spec-ngram-mod-size 5%` | 17M slots (130.175 MB, correctly 5% of 2.6 GB model) |
| Default (`--spec-type ngram-mod`) | 32 MB pool (4M slots), backwards compatible |
| Cache file size | 67 MB for 64 MB pool (208 byte header + 8 byte checksum) |
| Header validation | magic=0x4E474D44, version=1, all fields validated |
| **Round-trip** | **Cold: 28 lookups / 0 hits (0%). Warm: 181 lookups / 153 hits (85%).** |
| Periodic save | Correctly saves on threshold |
| Telemetry | `lookups`/`hits`/`miss_fp`/`inserts`/`overwrites` all reported |
| Bug fix | `recount_used()` added to `ngram_mod_cache_load()` - reset cleared counter after fread |
- Export telemetry to llama-bench columns and Prometheus metrics
- Dynamic pool growth with `--spec-ngram-mod-grow` / `--spec-ngram-mod-max-size`

### Verification Strategy

After each stage:
1. Build in Release mode
2. Run `llama-bench` with `--spec-type ngram-mod` at default settings - no TG regression
3. Run `llama-server` with new options - verify startup logs, speculative behaviour
4. For persistence: save/load round-trip test, corrupt file test, incompatible vocab test
5. For growth: populate small pool, trigger growth, verify entries retained

Full benchmark matrix after Stage 5 using the run-tg-matrix.py pattern from expert-cache work.
