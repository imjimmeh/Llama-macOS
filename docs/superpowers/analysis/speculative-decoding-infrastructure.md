# Speculative Decoding Infrastructure in llama.cpp

## 1. Core Files and Locations

| File | Purpose |
|---|---|
| `common/common.h:170-183` | `enum common_speculative_type` — all spec type constants |
| `common/common.h:322-395` | `common_params_speculative_draft`, `common_params_speculative_ngram_mod`, `common_params_speculative_ngram_map`, `common_params_speculative_ngram_cache`, `common_params_speculative` |
| `common/speculative.h` | Public API: `common_speculative_*` functions, `common_speculative_draft_params`, `common_speculative_init_result` |
| `common/speculative.cpp` | Full implementation: all `common_speculative_impl_*` subclasses, init/draft/accept/print_stats |
| `common/ngram-mod.h` + `ngram-mod.cpp` | `struct common_ngram_mod` — the hash-table data structure for ngram-mod predictor |
| `common/ngram-cache.h` + `ngram-cache.cpp` | `common_ngram_cache` — 3-level n-gram cache (static/dynamic/context) |
| `common/ngram-map.h` + `ngram-map.cpp` | `common_ngram_map` — key→value m-gram map used by ngram-simple, ngram-map-k, ngram-map-k4v |
| `common/arg.cpp:4029-4468` | CLI option parsing for all `--spec-*` options |
| `tools/server/server-context.cpp` | Server integration: slot spec flow, batch construction, verify/accept loop |
| `tools/server/server-common.h:346-470` | `server_task_result` and `server_slot` speculative stats fields |
| `tools/server/server-schema.cpp:196-227` | Schema fields for speculative params (currently `#if 0` disabled) |
| `tools/server/server-task.cpp:82-83, 1552-1561` | Task response and Prometheus metrics for spec decode |

---

## 2. Enum: `common_speculative_type` (common.h:170-183)

```cpp
enum common_speculative_type {
    COMMON_SPECULATIVE_TYPE_NONE,          // no speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE,  // standalone draft model
    COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3,  // Eagle3 draft model
    COMMON_SPECULATIVE_TYPE_DRAFT_MTP,     // Multi-token prediction head
    COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH,  // DFlash draft
    COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK,  // DSpark (DFlash + Markov head)
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,  // simple self-speculative (linear scan)
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,   // key-only n-gram map
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, // key + 4-value n-gram map
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,     // MOD hash-table predictor
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,   // 3-level n-gram cache
    COMMON_SPECULATIVE_TYPE_COUNT          // sentinel (11)
};
```

---

## 3. Struct Definitions for ngram-mod Configuration

### `common_params_speculative_ngram_mod` (common.h:353-358)

```cpp
struct common_params_speculative_ngram_mod {
    int32_t n_match = 24;   // n-gram lookup length (the "n" in n-gram)
    int32_t n_max   = 64;   // max tokens to draft per verify cycle
    int32_t n_min   = 48;   // min tokens required or discard the draft
};
```

### `common_params_speculative` (common.h:371-395)

```cpp
struct common_params_speculative {
    std::vector<enum common_speculative_type> types = { COMMON_SPECULATIVE_TYPE_NONE };
    common_params_speculative_draft draft;
    common_params_speculative_ngram_mod ngram_mod;
    common_params_speculative_ngram_map ngram_simple;
    common_params_speculative_ngram_map ngram_map_k;
    common_params_speculative_ngram_map ngram_map_k4v;
    common_params_speculative_ngram_cache ngram_cache;
    bool has_dft() const;
    uint32_t need_n_rs_seq() const;
};
```

---

## 4. How `spec-type ngram-mod` Is Configured

### Via CLI (arg.cpp:4289-4318)

```
--spec-type ngram-mod           → pushes COMMON_SPECULATIVE_TYPE_NGRAM_MOD onto params.speculative.types
--spec-ngram-mod-n-match N      → params.speculative.ngram_mod.n_match = N  (default: 24)
--spec-ngram-mod-n-max N        → params.speculative.ngram_mod.n_max = N    (default: 64)
--spec-ngram-mod-n-min N        → params.speculative.ngram_mod.n_min = N    (default: 48)
```

Validation:
- `n_match`: 1–1024
- `n_max`: 0–1024
- `n_min`: 0–1024
- Warning if `n_match < 16`: "ngram_mod n_match=N is too small - poor quality is possible"

### Via `--spec-default` shorthand (arg.cpp:4754-4768)

```cpp
params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
params.speculative.ngram_mod.n_match = 24;
params.speculative.ngram_mod.n_min   = 48;
params.speculative.ngram_mod.n_max   = 64;
```

---

## 5. CLI Option Parsing for All Speculative Decoding Options

All in `common/arg.cpp`, lines 4029-4468. Key options:

| Option | Default | Target Field |
|---|---|---|
| `--spec-type <list>` | `none` | `params.speculative.types` |
| `--spec-draft-model/-md <file>` | unused | `params.speculative.draft.mparams.path` |
| `--spec-draft-hf/-hfd <repo>` | unused | `params.speculative.draft.mparams.hf_repo` |
| `--spec-draft-n-max N` | 3 | `params.speculative.draft.n_max` |
| `--spec-draft-n-min N` | 0 | `params.speculative.draft.n_min` |
| `--spec-draft-p-split P` | 0.1 | `params.speculative.draft.p_split` |
| `--spec-draft-p-min P` | 0.0 | `params.speculative.draft.p_min` |
| `--spec-draft-ngl/-ngld N` | auto | `params.speculative.draft.n_gpu_layers` |
| `--spec-draft-device/-devd` | default | `params.speculative.draft.devices` |
| `--spec-draft-threads/-td N` | same | `params.speculative.draft.cpuparams.n_threads` |
| `--spec-draft-type-k/-ctkd TYPE` | `f16` | `params.speculative.draft.cache_type_k` |
| `--spec-draft-type-v/-ctvd TYPE` | `f16` | `params.speculative.draft.cache_type_v` |
| `--spec-draft-backend-sampling` | enabled | `params.speculative.draft.backend_sampling` |
| `--spec-ngram-mod-n-match N` | 24 | `params.speculative.ngram_mod.n_match` |
| `--spec-ngram-mod-n-max N` | 64 | `params.speculative.ngram_mod.n_max` |
| `--spec-ngram-mod-n-min N` | 48 | `params.speculative.ngram_mod.n_min` |
| `--spec-ngram-simple-size-n N` | 12 | `params.speculative.ngram_simple.size_n` |
| `--spec-ngram-simple-size-m N` | 48 | `params.speculative.ngram_simple.size_m` |
| `--spec-ngram-simple-min-hits N` | 1 | `params.speculative.ngram_simple.min_hits` |
| `--spec-ngram-map-k-size-n N` | 12 | `params.speculative.ngram_map_k.size_n` |
| `--spec-ngram-map-k-size-m N` | 48 | `params.speculative.ngram_map_k.size_m` |
| `--spec-ngram-map-k-min-hits N` | 1 | `params.speculative.ngram_map_k.min_hits` |
| `--spec-ngram-map-k4v-size-n N` | 12 | `params.speculative.ngram_map_k4v.size_n` |
| `--spec-ngram-map-k4v-size-m N` | 48 | `params.speculative.ngram_map_k4v.size_m` |
| `--spec-ngram-map-k4v-min-hits N` | 1 | `params.speculative.ngram_map_k4v.min_hits` |
| `--spec-ngram-cache-static <path>` | — | `params.speculative.ngram_cache.lookup_cache_static` |
| `--spec-ngram-cache-dynamic <path>` | — | `params.speculative.ngram_cache.lookup_cache_dynamic` |
| `--spec-default` | — | Sets ngram-mod with n_match=24, n_min=48, n_max=64 |

Removed aliases:
- `--draft` / `--draft-max` → error: use `--spec-draft-n-max` or `--spec-ngram-mod-n-max`
- `--draft-min` → error: use `--spec-draft-n-min` or `--spec-ngram-mod-n-min`
- `--spec-ngram-size-n` → error: use `--spec-ngram-*-size-n` or `--spec-ngram-mod-n-match`

---

## 6. `common_ngram_mod` Data Structure (ngram-mod.h / ngram-mod.cpp)

```cpp
struct common_ngram_mod {
    using entry_t = int32_t;
    static constexpr entry_t EMPTY = -1;

    common_ngram_mod(uint16_t n, size_t size);  // n = n-gram length, size = table slots

    size_t  idx(const entry_t * tokens) const;  // hash n-gram → slot index
    void    add(const entry_t * tokens);          // store tokens[0..n-1] → tokens[n]
    entry_t get(const entry_t * tokens) const;   // lookup: returns tokens[n] or EMPTY

    void reset();
    size_t get_n()    const;   // returns n
    size_t get_used() const;   // count of occupied slots
    size_t size()       const; // total table slots
    size_t size_bytes() const; // size() * sizeof(entry_t)
};
```

**Hash function** (ngram-mod.cpp:15-24):
```cpp
size_t common_ngram_mod::idx(const entry_t * tokens) const {
    size_t res = 0;
    for (size_t i = 0; i < n; ++i) {
        res = res * 6364136223846793005ULL + tokens[i];
    }
    return res % entries.size();
}
```

This is a simple modular hash with multiplicative constant (LCG-style). The table stores a single predicted token per slot — no collision handling, so any collision overwrites the previous prediction. The default table size is `4 * 1024 * 1024` entries = 16 MB (set at speculative.cpp:1853).

---

## 7. `common_speculative_impl_ngram_mod` (speculative.cpp:1826-1999)

### Constructor
```cpp
common_speculative_impl_ngram_mod(
    const common_params_speculative & params,
    uint32_t n_seq)
    : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq)
    , params(params.ngram_mod)
    , mod(params.ngram_mod.n_match, 4*1024*1024)  // n_match-gram, 16MB table
    , verbose(std::getenv("LLAMA_TRACE") != nullptr)
```

### Per-sequence state
```cpp
struct seq_info {
    size_t i_last = 0;       // last position in prompt added to the mod
    size_t n_draft_last = 0; // length of last draft (for acceptance tracking)
    int n_low = 0;           // consecutive low-acceptance rounds (< 25%)
};
```

### Key methods

**`begin(seq_id, prompt)`** — Seeds the mod table from prompt tokens:
- Adds all n-grams from position 0 to `prompt.size() - n`
- Checks occupancy; if > 25% full, resets (collision quality degradation)

**`draft_one(seq_id, dparams)`** — Core drafting:
1. Incrementally adds new n-grams if `i_last + 32 < cur_len`
2. Seeds draft window: `result[0..n-2]` = last `n-1` context tokens, `result[n-1]` = `dparams.id_last`
3. Iterates `n_max` times calling `mod.get()`; stops on EMPTY or after `n_max`
4. If fewer than `n_min` tokens drafted, discards the entire draft
5. Records `n_draft_last` for acceptance tracking

**`accept(seq_id, n_accepted, is_other)`** — Adaptation logic:
- Computes acceptance fraction: `n_accepted / n_draft_last`
- If fraction < 0.25 for 5 consecutive rounds → resets the entire mod table (poor quality detected)
- No state persistence (no `get_state`/`set_state` overrides)

**`process(batch)`** — Currently a no-op (TODO comment: "implement")

---

## 8. Implementation Priority and Init (speculative.cpp:2460-2580)

The `common_speculative_init()` function creates implementations in priority order:

```
1. NGRAM_SIMPLE    ← highest priority (first impl to draft wins)
2. NGRAM_MAP_K
3. NGRAM_MAP_K4V
4. NGRAM_MOD       ← ngram-mod is 4th priority
5. NGRAM_CACHE
6. DRAFT_SIMPLE
7. DRAFT_EAGLE3    ← requires ctx_dft != nullptr
8. DRAFT_MTP       ← requires ctx_dft != nullptr
9. DRAFT_DFLASH    ← requires ctx_dft != nullptr
10. DRAFT_DSPARK   ← requires ctx_dft != nullptr
```

The `common_speculative_draft()` function iterates through impls in order. The first impl to produce a non-empty result for a sequence wins — subsequent impls skip that sequence. This means if ngram-simple or ngram-map produces a draft, ngram-mod never gets a chance.

---

## 9. Base `common_speculative_impl` (speculative.cpp:137-175)

```cpp
struct common_speculative_impl {
    const common_speculative_type type;
    uint32_t n_seq;

    // Stats counters
    size_t n_call_begin  = 0;
    size_t n_call_draft  = 0;
    size_t n_call_accept = 0;
    size_t n_gen_drafts  = 0;
    size_t n_acc_drafts  = 0;
    size_t n_gen_tokens  = 0;
    size_t n_acc_tokens  = 0;
    std::vector<size_t> n_acc_tokens_per_pos;

    // Timing
    int64_t t_begin_us  = 0;
    int64_t t_draft_us  = 0;
    int64_t t_accept_us = 0;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;
    virtual bool process(const llama_batch & batch) = 0;
    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;
    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;
    virtual bool get_state(...) const { return false; }
    virtual void set_state(...) {}
};
```

---

## 10. Stats Collection and Reporting

### Internal stats (speculative.cpp:2767-2811)

`common_speculative_print_stats(spec)` iterates all impls and logs via `SPC_TRC`:
```
statistics ngram-mod: #calls(b,g,a) = 100  100  100,
  #gen drafts = 80, #acc drafts = 60,
  #gen tokens = 4800, #acc tokens = 3600,
  #mean acc len = 46.000, #acc rate/pos = (1.0, 0.98, ...)
```

Fields reported:
- `n_call_begin`, `n_call_draft`, `n_call_accept` — call counts
- `n_gen_drafts`, `n_acc_drafts` — draft generation and acceptance counts
- `n_gen_tokens`, `n_acc_tokens` — token counts
- `n_acc_tokens_per_pos` — per-position acceptance rates (normalized by n_call_accept)
- `t_begin_us`, `t_draft_us`, `t_accept_us` — timing in ms (when `gen_perf=true`)

### Server-side stats (server-common.h:346-470, server-context.cpp:614-640)

`server_task_result` fields:
```cpp
uint64_t n_draft_tokens      = 0;  // total draft tokens generated
uint64_t n_draft_accepted    = 0;  // draft tokens accepted
uint64_t n_draft_verif_steps = 0;  // verification steps (decode calls with drafts)
```

`server_slot` fields:
```cpp
uint64_t n_draft_tokens      = 0;
uint64_t n_draft_accepted    = 0;
uint64_t n_draft_verif_steps = 0;
std::vector<uint64_t> n_accepted_per_pos;
```

At slot release (server-context.cpp:614-640), server computes and logs:
- `draft_ratio = n_draft_accepted / n_draft_tokens`
- `mean_acc_len = 1.0 + n_draft_accepted / n_draft_verif_steps`
- Per-position acceptance rates

Prometheus metrics (server-task.cpp:1552-1561):
```
spec_decode_num_drafts_total       → metrics.n_draft_tokens
spec_decode_num_accepted_tokens_total → metrics.n_draft_accepted
spec_decode_num_drafts_total       → metrics.n_draft_verif_steps
```

---

## 11. Server Integration Flow

### Initialization (server-context.cpp:1155-1275)

```
1. common_speculative_init_from_params(params, model_tgt, ctx_tgt)
   → loads draft model if has_dft(), creates draft context
   → stores in spec_init (common_speculative_init_result_ptr)

2. common_speculative_init(params_base.speculative, params_base.n_parallel)
   → creates common_speculative with impl vector
   → stored in spec (common_speculative_ptr)

3. Each slot: slot.spec = spec.get()
   → slot.can_speculate() = (spec != nullptr)
```

### Per-token generation loop (server-context.cpp:2965-3949)

```
A. BEFORE decode:
   1. Set draft params: common_speculative_get_draft_params(spec, slot.id)
      → .drafting = true, .n_max = ..., .id_last = sampled, .prompt = &spec_prompt, .result = &spec_draft

   2. common_speculative_draft(spec) — yields to task queue
      → iterates impls; first non-empty result wins
      → for ngram-mod: calls draft_one() which fills spec_draft

B. AFTER decode (with draft tokens in batch):
   1. common_sampler_sample_and_accept_n(smpl, ctx_tgt, spec_i_batch, spec_draft)
      → verifies draft tokens, returns accepted prefix

   2. n_accepted = accepted.size() - 1
   3. common_speculative_accept(spec, slot.id, n_accepted)
      → updates stats, calls impl->accept() for primary + all others (is_other=true)

   4. slot.stats.n_draft_accepted += n_accepted
   5. slot.stats.n_draft_verif_steps += 1

C. Batch construction (handle_last_sampled_token):
   if spec_draft is non-empty:
     → batch.add(sampled + all spec_draft tokens) with sequence IDs

D. Process callback:
   llama_set_batch_callback → common_speculative_process(spec, batch)
   → ngram-mod's process() is currently a no-op
```

### State serialization (for context checkpointing):
- `common_speculative_get_state(spec, slot.id, data)` — called during checkpoint creation
- `common_speculative_set_state(spec, slot.id, data)` — called during checkpoint restore
- ngram-mod does NOT override get_state/set_state (returns false/no-op)

---

## 12. `--spec-ngram-*` Options (Complete List)

### ngram-mod specific:
- `--spec-type ngram-mod`
- `--spec-ngram-mod-n-match N` (default: 24)
- `--spec-ngram-mod-n-max N` (default: 64)
- `--spec-ngram-mod-n-min N` (default: 48)

### ngram-simple:
- `--spec-type ngram-simple`
- `--spec-ngram-simple-size-n N` (default: 12)
- `--spec-ngram-simple-size-m N` (default: 48)
- `--spec-ngram-simple-min-hits N` (default: 1)

### ngram-map-k:
- `--spec-type ngram-map-k`
- `--spec-ngram-map-k-size-n N` (default: 12)
- `--spec-ngram-map-k-size-m N` (default: 48)
- `--spec-ngram-map-k-min-hits N` (default: 1)

### ngram-map-k4v:
- `--spec-type ngram-map-k4v`
- `--spec-ngram-map-k4v-size-n N` (default: 12)
- `--spec-ngram-map-k4v-size-m N` (default: 48)
- `--spec-ngram-map-k4v-min-hits N` (default: 1)

### ngram-cache:
- `--spec-type ngram-cache`
- `--spec-ngram-cache-static <path>`
- `--spec-ngram-cache-dynamic <path>`

### Shared / convenience:
- `--spec-default` → sets ngram-mod with n_match=24, n_min=48, n_max=64
- `--draft` / `--draft-max` → REMOVED (error message points to --spec-draft-n-max or --spec-ngram-mod-n-max)

---

## 13. `common_speculative_n_max` (speculative.cpp:2290-2324)

This function computes the maximum draft tokens across all enabled spec types:

```cpp
int32_t common_speculative_n_max(const common_params_speculative * spec) {
    // for NGRAM_MOD: max(0, spec->ngram_mod.n_max)
    // for DRAFT_*:   max(0, spec->draft.n_max)
    // for NGRAM_SIMPLE/MAP_K/MAP_K4V: size_m
    // for NGRAM_CACHE: 8 (hardcoded)
}
```

This value is used by the server to size output buffers and draft parameter limits.

---

## 14. Key Architectural Notes

1. **ngram-mod is a single-token-per-slot predictor**: Unlike ngram-map (which stores full m-gram sequences), ngram-mod only stores one next-token prediction per n-gram hash. It's the simplest but also the most collision-prone predictor.

2. **Fixed table size**: 4M entries × 4 bytes = 16 MB, hardcoded in the constructor. The `n_match` parameter controls n-gram length but not table capacity.

3. **Occupancy-based reset**: If > 25% of slots are occupied during `begin()`, the entire table is cleared. This is a quality gate — high occupancy means many collisions.

4. **Low-acceptance reset**: 5 consecutive rounds with < 25% acceptance triggers a full reset of the mod table. This adapts when the model's behavior shifts.

5. **No state persistence**: ngram-mod has no `get_state`/`set_state` implementation — it doesn't survive context checkpoints/replays. The server's `common_speculative_set_state` call on restore is a no-op for ngram-mod.

6. **Implementation priority**: ngram-mod (priority 4) will only be tried if ngram-simple, ngram-map-k, and ngram-map-k4v all fail to produce a draft. In practice with `--spec-type ngram-mod` alone, only ngram-mod is active.

7. **Server schema for runtime adjustment**: Currently `#if 0` disabled (server-schema.cpp:196-227). There's no runtime HTTP API to change spec params; only CLI args at startup.

8. **`--spec-default` enables ngram-mod only**: The commented-out ngram-map-k4v config suggests this is still experimental:
```cpp
//params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
//params.speculative.ngram_map_k4v.size_n = 8;
//params.speculative.ngram_map_k4v.size_m = 24;
//params.speculative.ngram_map_k4v.min_hits = 2;
```
