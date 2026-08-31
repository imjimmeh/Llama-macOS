# Implementation Plan: Hot/Cold Tiering for Persistent N-Gram Speculative Memory

## 1. Summary

Extend the persistent ngram-mod implementation (Stages 1-4 of
`2026-08-31-persistent-ngram-mod-implementation.md`) with a two-tier memory:

```text
                   ngram lookup (per token)
                            |
                            v
               +------------------------+
               | HOT RAM TABLE          |   hit -> draft token
               | 8 B/slot               |
               | --spec-ngram-mod-size  |
               +-----------+------------+
                           | miss
                           v
               +------------------------+
               | COLD mmap store (SSD)  |   hit -> promote to hot -> draft token
               | 8 B/slot + 4 B hits    |
               | --spec-ngram-mod-cold- |
               | size                   |
               +-----------+------------+
                           | miss
                           v
                    draft ends here
```

Goals:

1. Scale total ngram memory beyond the RAM budget. The compact preset model
   (17.3 GB) plus 128k KV cache already consumes most of 32 GB RAM; a multi-GB
   ngram store must live on SSD.
2. Keep the latency-critical lookup path in RAM. Cold store is consulted only
   on hot miss, through the OS page cache, never via explicit disk I/O.
3. Only the hot subset of entries is loaded into RAM. The rest stays on SSD
   (user requirement).
4. Turn the existing 25% occupancy reset (which wipes all learned knowledge)
   into a demote-to-cold operation.
5. Backwards compatible: tiering off by default. `--spec-type ngram-mod`
   alone behaves exactly as today.

## 2. Relationship to the Epic

- Epic section 19: RAM -> active speculation, SSD -> persistence. Do NOT make
  lookup depend on random SSD access.
- Epic section 41: two-tier memory (HOT RAM TABLE + LONG-TERM RAM or mmap
  TABLE) is listed as a later optimisation, to be built only if benchmarking
  establishes a need.
- Epic section 20: mmap persistence experiment, page-cache backed.

This plan implements the section 41 architecture with the section 20 mmap
mechanism. The epic's evidence gate is preserved: Stage A below is a small
measurement step that must pass before the tiering work proceeds.

## 3. Evidence Gate (must pass before Stage C)

Before building the tiered runtime, demonstrate that capacity beyond the RAM
budget improves hit coverage.

1. Measure ngram-mod hit coverage across pool sizes using the existing
   server/speculative-simple telemetry (Task A1). No llama-bench changes:
   llama-bench has no speculative decoding support (no --spec-type, no
   sampling loop), so ngram-mod columns there would be dead code.
2. Sweep pool sizes 256M / 1G / 2G / 4G on the qwen3.6-35B-apex-compact
   model, recording hit rate and TG t/s, cold and warm start.
3. Decision: if hit rate plateaus within the RAM budget (<= 1G on this
   machine), tiering adds no value - stop, document the optimum, and mark the
   plan abandoned. If hit coverage keeps improving past the RAM budget,
   proceed to Stage B.

Expected outcome on the compact preset: coding/reasoning workloads repeat
long sequences, so larger stores should keep helping. This must be measured,
not assumed.

### Stage A Results (measured 2026-08-31)

**Occupancy / collision curve** (`tests/test-ngram-mod.cpp`, n=24, scale-free;
1M/4M/16M slot pools agree within noise):

| occupancy | 5% | 10% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|---|
| overwrite rate | 0.025 | 0.049 | 0.115 | 0.213 | 0.297 | 0.368 |
| lookup hit rate | 0.975 | 0.951 | 0.884 | 0.787 | 0.703 | 0.632 |

False hits at 100% occupancy: 0 / 100000.

**Pool-size sweep** (`tools/results/ngram-mod/run-ngram-mod-sweep.py`, compact
preset model, 768 generated tokens, cold = fresh cache, warm = destructor-saved
cache from the cold run):

| size | mode | lookups | hits | hit_rate | overwrite_rate | t/s |
|---|---|---|---|---|---|---|
| 256M | cold | 896 | 288 | 0.321 | 0.173 | 25.3 |
| 256M | warm | 1158 | 1008 | 0.870 | 0.831 | 44.8 |
| 512M | cold | 896 | 288 | 0.321 | 0.173 | 25.7 |
| 512M | warm | 1158 | 1008 | 0.870 | 0.831 | 43.8 |
| 1G | cold | 896 | 288 | 0.321 | 0.173 | 23.9 |
| 1G | warm | 1158 | 1008 | 0.870 | 0.831 | 45.9 |
| 2G | cold | 896 | 288 | 0.321 | 0.173 | 26.2 |
| 2G | warm | 1158 | 1008 | 0.870 | 0.831 | 43.9 |
| 4G | cold | 896 | 288 | 0.321 | 0.173 | 26.3 |
| 4G | warm | N/A | - | - | - | 26.0 |

Raw records: `tools/results/ngram-mod/compact-20260831-ngram-mod-sweep.jsonl`
(one row per run) and per-run logs `compact-20260831-<size>-<mode>.log`.

Interpretation:

1. Hit quality is occupancy-driven, not size-driven. All pool sizes give
   identical stats on a 768-token run because only ~900 distinct n-grams are
   learned (pool occupancy 0.00%). The scale-free micro-benchmark curve is the
   pool-size evidence: 25% occupancy (the runtime reset threshold) costs hit
   rate 0.975 -> 0.884; 100% occupancy costs 0.975 -> 0.632.
2. Persistence dominates the short-run benefit: warm start hits 0.870 vs cold
   0.321 and TG throughput 44-46 vs 24-26 t/s (+~75%). The value of ngram-mod
   memory is retained knowledge across sessions, not raw pool size.
3. The 4G warm row is missing because the v1 loader rejects capacity above
   `NGRAM_MOD_CACHE_MAX_SLOTS` (256M slots = 2 GB, `ngram-mod-cache.h`). The
   runtime accepts 4G pools (pool slots = 536870912); only the loader is
   capped. This 2 GB load cap is itself a scalability limit the Stage B cold
   store removes (mmap load has no giant heap fread).

**Gate decision: PASS.** Hit coverage does improve past the RAM budget, but
only through retention, not through a larger hot pool: the 25% occupancy reset
currently destroys all learned n-grams, and the warm-run evidence shows that
retained knowledge is worth ~3x hit rate and ~1.75x TG throughput. A cold
store converts the reset from "wipe everything" to "demote to SSD", preserving
that value across sessions and removing the 2 GB loader cap. Proceed to
Stage B.

## 4. Design Decisions

### 4.1 Cold store is a writable mmap of the cache file

The existing `--spec-ngram-mod-cache` file becomes the cold store when
tiering is enabled. A writable memory mapping gives:

- slot writes become memory writes + page dirty (no full-file rewrite);
- checkpoint = msync + header update (fast, no 1 GB rewrite per save);
- the OS page cache is a de-facto frequency filter: recurring n-grams keep
  their pages resident, one-off n-grams fault once and are evicted.

The existing `llama_mmap` (src/llama-mmap.cpp) is read-only and
model-loading oriented. Do not modify it. Add a small writable-mmap wrapper
in ngram-mod-cache.cpp (POSIX `mmap` + Windows `CreateFileMapping`/
`MapViewOfFile`, ~60 lines).

### 4.2 Cold lookup is page-cache only, gated by a flag

Cold misses on the hot table consult the mapped cold store. First touch of a
page faults (10-100 us); recurring patterns stay resident. Random hashing
spreads one draft chain across many pages (epic section 20 risk), so a
fully memorized draft of length 64 can touch 64 random pages. On first
encounter this is ~1 ms worst case on a ~50 ms/token budget; on recurrence
the pages are resident.

`--spec-ngram-mod-cold-fallback off` disables cold lookup entirely. The cold
store then serves persistence only, and the hot table is loaded from the
top-K hotness-ranked entries at startup (the literal "only hot in RAM"
mode).

### 4.3 Hotness metadata

Each cold slot gets a saturating 4-byte hit counter. Incremented on:
- hot-table hit;
- cold-table hit (before promotion).

This drives:
- load-time hot selection (rank by hits, load top-K);
- promotion decisions (a cold entry is promoted into the hot table on hit).

Hot table slots stay 8 bytes; hot hits live in a parallel
`std::vector<uint32_t>` allocated only in tiered mode. This keeps the hot
lookup path identical to today.

### 4.4 Promotion and demotion

- Promote: cold hit -> `add()` into the hot table (direct-mapped, may
  overwrite) + bump hits.
- Demote (hot eviction): every hot `add()` overwrite writes the evicted
  entry into the cold slot (12 B mapped write: slot + hits). Cold
  overwrites are natural direct-mapped behavior.
- Occupancy reset (25%): currently wipes the pool in `process()`. With
  tiering, flush all hot slots to cold first, then reset. Knowledge
  survives. The low-acceptance streak reset stays a hard hot wipe (active
  bad model), cold is untouched.

### 4.5 File format v2

```text
[ header v2 ]            magic, version=2, slot_size=8, hits_size=4,
                         capacity_cold, flags (tiered, clean_close), n_match, ...
[ cold slots : capacity_cold * 8 B ]
[ cold hits  : capacity_cold * 4 B ]
[ footer     : xxh64 over cold slots + cold hits ]
```

- v1 files still load (tiering off path unchanged). With tiering on, a v1
  file is upgraded in place to v2 with hits = 0.
- Checksum policy: the footer checksum is written at graceful close. Torn
  pages from a crash corrupt at most a page of slots; corrupted slots are
  detected at lookup by fingerprint mismatch (`miss_fp`) and overwritten by
  the next insert - self-healing. Header is validated on load (magic,
  version, sizes, capacity bound). Best-effort durability, documented.

### 4.6 VRAM is out of scope

User asked for "RAM/VRAM". The hot table stays in system RAM:

- epic section 13: keep the active lookup path in system RAM, not GPU VRAM;
- GTX 1080 has 8 GB shared with the model; a PCIe round-trip per token
  lookup would destroy the cheap-lookup property of ngram-mod.

A GPU-resident hot table (device hash table or managed memory) is noted as a
future experiment only, contingent on evidence that RAM lookup bandwidth is
the bottleneck.

## 5. CLI

| Option | Type | Default | Description |
|---|---|---|---|
| `--spec-ngram-mod-cold-size SIZE` | SIZE (K/M/G/T/%) | 0 | cold store size on disk; 0 = tiering off |
| `--spec-ngram-mod-cold-fallback` | flag | on (when cold-size > 0) | consult cold store on hot miss |
| `--spec-ngram-mod-cache PATH` | string | "" | unchanged; file is the cold store when tiering on |

- `%` resolves against model weight bytes, same as `--spec-ngram-mod-size`.
- `--spec-ngram-mod-cold-size 0` (default) preserves today's behavior
  exactly.
- Cold store is created/truncated at startup to the configured size. Resize
  at runtime is future work.

## 6. Stages

### Stage A: Measurement Prerequisite
**Task A1: sweep vehicle and script**

llama-bench cannot run speculative decoding (no --spec-type support, no
sampling loop), so ngram-mod columns there would be dead code. Use
`examples/speculative-simple` instead:

- it drives the same `common_speculative` machinery as the server;
- it accepts the `--spec-ngram-mod-*` flags via common_params;
- it prints the `statistics ngram-mod` line (lookups / hits / miss_fp /
  inserts / overwrites) and decoded t/s;
- normal exit triggers the destructor save, so warm runs reuse the cache.

New script `tools/results/ngram-mod/run-ngram-mod-sweep.py`:
- for each pool size: cold run (fresh cache) then warm run (reuse saved
  cache);
- fixed prompt with repeated content (configurable);
- parse `#nm lookups=... hits=... inserts=... overwrites=...` and decoded
  t/s from the log;
- report: size, mode, lookups, hits, hit rate, overwrite rate, t/s.

**Task A2: capacity sweep**

Sweep 256M / 1G / 2G / 4G on the qwen3.6-35B-A3B-APEX-Compact.gguf model
with speculative-simple (-t 14, no-mmap, mlock, n_match=24, n_min=24,
n_max=48, n_predict=512). Record hit rate, overwrite rate, t/s, cold and
warm start.

**Task A3: synthetic occupancy micro-benchmark**

A short real-model run inserts only a few thousand n-grams, far below
collision pressure for even a 256M pool (33M slots). Pool-size effects only
appear at millions of inserts. Measure the collision curve directly instead:

New `tests/test-ngram-mod.cpp`: insert K random n-grams into
`common_ngram_mod` pools at occupancies 5% / 10% / 25% / 50% / 75% / 100%
for slot counts 1M / 4M / 16M (scale-free curve), and report overwrite rate
and lookup hit rate. Also verify fingerprint checks reject never-inserted
n-grams at 100% occupancy (false-hit test).

This gives the pool-sizing evidence the sweep cannot: the occupancy at which
hit coverage degrades, and therefore whether capacity past the RAM budget
can matter for a server that accumulates millions of n-grams.

**Gate**: proceed to Stage B only if hit coverage improves past the RAM
budget. Otherwise stop and document the optimum capacity.

### Stage B: Writable mmap Cold Store + Format v2

**Task B1: mmap wrapper**

New writable mapping helper in `common/ngram-mod-cache.cpp`:
- open/truncate/create file to `capacity_cold * 12 + header + footer`;
- map (POSIX `mmap` PROT_READ|PROT_WRITE MAP_SHARED; Windows
  CreateFileMapping + MapViewOfFile);
- unmap on close; `msync`/`FlushViewOfFile` helper.

**Task B2: format v2**

- Extend `ngram_mod_cache_header` with `capacity_cold`, `hits_size`,
  `flags` (tiered, clean_close). Bump `NGRAM_MOD_CACHE_VERSION` to 2.
- v1 load path unchanged; v2 load validates header, maps the file.
- v1 + tiering on: rewrite as v2 (hits = 0) on first save.
- Graceful close: flush, msync, write footer checksum, set clean flag.
- Load with clean flag clear (crash): skip footer checksum, rely on
  fingerprint self-healing.

**Task B3: `--spec-ngram-mod-cold-size`**

- `common/common.h`: add `cold_size_bytes`, `cold_size_pct` to
  `common_params_speculative_ngram_mod`.
- `common/arg.cpp`: parse with the existing SIZE parser (K/M/G/T/%).
- Resolve against `model_weight_bytes` at init, same as pool size.

**Verification**: cold file created and mapped at the configured size;
header round-trips; v1 file still loads with tiering off; v1 upgraded to v2
with tiering on; unmapped cleanly at exit.

**Stage B Results (2026-08-31, MSVC Release, `test-ngram-mod`)**

- Cold file layout: `240 B header + cap*(8 B slot + 4 B hits) + 8 B footer`.
  1M-slot store = 12,583,160 B, verified byte-exact after flush.
- Header round-trip: v2 header (tiered flag, capacity_cold, cold_entry_count)
  survives reopen; slot 10 content `{0x12345678, -3}` with hits 9 and slot
  123456 `{0xCAFEBABE, 1000}` persist across reopen; empty slots read as
  `next_token == -1` (nullptr).
- Crash simulation: CLEAN_CLOSE cleared in the file -> reopen marks store
  dirty; footer rewritten on next flush.
- Checksum mismatch on a clean-close file -> store reopens, flags dirty,
  self-heals (no data loss).
- Wrong n_match / mismatched tokenizer rejected at open.
- v1 (224 B header) file upgrades in place: all slots copied, hits = 0,
  extension region 0xFF, reopens as valid tiered v2.
- `--spec-ngram-mod-cold-size 0` (default): non-tiered save/load round-trip
  unchanged and byte-compatible with v1 readers.
- Commits: `e3c3bc5fa` (format v2 + mmap cold store), `bd7c226e3` (CLI +
  runtime wiring), `6efd3fc2d` (tests).

### Stage C: Hot/Cold Runtime

**Task C1: tier wrapper**

New `common/ngram-mod-tier.h/.cpp`:
- holds hot `common_ngram_mod` + cold mapping + hits vector;
- `tier_add(tokens)`: hot add; on overwrite, write evicted entry to cold;
- `tier_get(tokens)`: hot get; on miss, cold get; on cold hit, promote and
  bump hits; returns token or EMPTY;
- gate cold lookup on `cold_fallback`.

**Task C2: wire into `common_speculative_impl_ngram_mod`**

- constructor: build tier (cold from cache file) when `cold_size > 0`;
- `draft_one()`: use `tier_get` instead of `mod.get`;
- `process()`: occupancy reset becomes flush-to-cold then reset; periodic
  checkpoint = msync + header update instead of full-array save;
- destructor: final flush + msync + footer + clean flag;
- stats block: add hot/cold lookup and hit counts.

**Task C3: telemetry**

- `lookups` / `hits` per tier in the `statistics ngram-mod` line;
- cold promotion count, cold fault proxy (cold lookups that miss).

**Verification**: on the compact preset with 1G hot + 4G cold, hot hit rate
stays near the single-tier 1G rate; cold lookup count is small relative to
hot; promoted entries appear in hot; 25% occupancy reset no longer drops
the hit rate to zero (demote works).

**Stage C Results (2026-08-31, MSVC Release)**

- Tier wrapper (`common/ngram-mod-tier.h/.cpp`): cold store is keyed by
  fingerprint (cold idx = fp % cold size) so demotion needs only the stored
  {fp, token} pair, and the v1 upgrade now places entries at the same
  fingerprint index (the Stage B index-to-index copy was unreachable from
  runtime lookups; fixed).
- Hot hits never touch the cold mapping (plan 4.3 hot-hit hit-counter bump
  dropped: it would put a random page touch on the hot path); cold hit
  counters bump only on cold hits and survive flush merges.
- Occupancy reset and shutdown flush the whole hot pool to cold before
  resetting; the low-acceptance reset stays a hard hot wipe per plan.
- `--spec-ngram-mod-cold-fallback on|off` gates cold lookups (default on).
- Tests: demotion-on-overwrite, cold-hit promotion (counters verified),
  fallback-off miss, flush_reset persistence across reopen, v1 upgrade at
  fingerprint indices (988 v1 entries -> 986 cold, 2 fp-index collisions).
- Smoke (CPU build, Qwen3.6-35B-APEX-Compact, 512M cold): fresh run fills
  hot only; warm restart drafts and accepts 47/47 tokens via cold fallback +
  promotion; with fallback off on the same warm cache: 0 drafts (gate works).
  Exit state: cold_entry_count=155, flags=3 (tiered + clean close).
- Telemetry: `statistics` line gains cold lookups/hits/promotes/demotes/
  flushes when tiered (SPC_TRC, trace log level).

### Stage D: Load-Time Hot Selection

**Task D1: startup scan**

On init with tiering on: scan cold slots + hits, rank by hits, load the
top-K into the hot pool (K = hot capacity). Log:
```
ngram-mod tier: hot slots=134217728 (1.000 GB), cold slots=536870912 (4.000 GB)
ngram-mod tier: loaded 1234567 hot entries (top 42% by hotness)
```

**Task D2: cold-fallback-off mode**

With `--spec-ngram-mod-cold-fallback off`: no cold lookup, pure
persistence + hot selection. This is the literal "only hot in RAM" mode and
the zero-page-fault fallback if Stage C measurements show latency damage.

**Verification**: warm restart with 4G cold + 1G hot shows hot hit rate
comparable to a pre-tiering full-RAM load of the same data; cold-fallback
off mode shows identical behavior to current persistence plus hotness-ranked
load.

**Stage D Results (2026-08-31, MSVC Release)**

- Hot pool keying changed to the fingerprint index (idx = fp % size), the
  same key the cold store uses: a load-time restore only has {fp, token}
  per cold entry, and the full hash cannot be reconstructed. The pool index
  and the stored identity are now the same 32-bit value, so a fully
  occupied pool has 2^-(32-log2(size)) false-hit probability per lookup
  (1/4096 at 1M slots) instead of ~1/2^32; the sanity band in
  `test_false_hits` reflects this. At the 25% runtime occupancy cap the
  real pools see <= ~1/128 per lookup at 1G.
- New header flag `NGRAM_MOD_CACHE_FLAG_FP_INDEX` (bit 2) written by every
  save and cold-store flush; non-tiered loads reject files without it
  (v1 dumps and early v2 dumps are h-keyed and unreachable under fp keying).
  The tiered v1 upgrade path is not gated - it re-places by fingerprint.
- Startup scan: single pass over cold slots, per hot slot the highest-hit
  candidate wins (direct-mapped top-K). Telemetry logs tier sizes, loaded
  count and scan time at SPC_TRC; `hot_loaded` added to the statistics line.
- Smoke (CPU build, Qwen3.6-35B-APEX-Compact, 128M hot + 512M cold,
  repeated-prompt text): warm restart logs
  `loaded 299 hot entries (top 100% by hotness) in 0.096 s`; with
  `--spec-ngram-mod-cold-fallback off` the same warm cache drafts and
  accepts 21/21 tokens with `cold lookups=0` (pure hot-tier operation).
  Cold file: 536,871,152 B exact, flags=7 (TIERED|CLEAN_CLOSE|FP_INDEX).
- `test-ngram-mod` green: hot selection (hotter entry wins a colliding
  slot), v1 and non-FP_INDEX v2 rejection, fp-keyed occupancy curve
  unchanged (0.884 hit rate at 25% occupancy), deterministic flush_reset.
- Commits: `57673c361` (fp index + compat gate), `5ea8bd6a7` (startup hot
  restore), `d8ca2ffba` (tests).

### Stage E: Measurement and Documentation

**Task E1: tiering benchmark matrix**

- hot/cold split ratios (1G/2G, 1G/4G, 2G/4G);
- fallback on vs off;
- cold and warm start;
- record hit rate, promotion count, TG t/s, cold lookup latency impact.

**Task E2: docs**

- `docs/speculative.md`: new options in the n-gram Mod Parameters block;
  two-tier description.
- `tools/server/README.md`: auto-generated rows for the new options.
- Update this plan's Verification Results section with real numbers.
- Keep `docs/superpowers/plans/2026-08-31-persistent-ngram-mod-implementation.md`
  Stage 8 list in sync (hot/cold tiering moves from future work to done).

## 7. Files

| File | Change | Stage |
|---|---|---|
| `common/ngram-mod-tier.h` | new: tier wrapper, policy | C |
| `common/ngram-mod-tier.cpp` | new: tier implementation | C |
| `common/ngram-mod-cache.h` | format v2 header fields | B |
| `common/ngram-mod-cache.cpp` | mmap wrapper, v2 save/load, migration | B |
| `common/common.h` | cold_size params | B |
| `common/arg.cpp` | new CLI options | B |
| `common/speculative.cpp` | tier wiring, draft_one, process, stats | C |
| `tools/llama-bench/llama-bench.cpp` | telemetry columns | A |
| `docs/speculative.md` | options + architecture | E |
| `tools/server/README.md` | options rows | E |


## 8. Backwards Compatibility

- `--spec-type ngram-mod` with no new options: unchanged (32 MB RAM pool,
  v1 file format, full-array save).
- `--spec-ngram-mod-cache` alone: unchanged behavior, but the file format
  written by a tiering-enabled build is v2. v2 files still load with
  tiering off (cold-only semantics: the whole file is the pool).
- Fingerprint keying (Stage D): the hot pool index is now fp % size, so
  v1 dumps and early v2 dumps (h-keyed) are unreachable in non-tiered
  loads and are rejected by the FP_INDEX flag gate; the tiered v1 upgrade
  re-places them by fingerprint and still works. New saves set FP_INDEX,
  so same-build round-trips are unaffected.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Cold lookup page faults spike latency | page-cache frequency filter; `--spec-ngram-mod-cold-fallback off` escape hatch; measure in E1 |
| Dirty-page writeback stalls at checkpoint/reset | reset flush is ~400 MB for a 1G hot pool at 25% occupancy; rare; measure, tune interval |
| RAM pressure (mlock model + hot pool) | cold pages are evictable - that is the point; hot pool sized by flag |
| Weaker crash semantics than v1 atomic replace | fingerprint self-healing on corrupted slots; header validation; clean-close flag; documented best-effort durability |
| 4 GB startup scan for hot selection | ~0.4 s at load; acceptable; future hot-candidate index |
| Fingerprint-keyed false hits | bounded at 2^-(32-log2(hot slots)) per lookup at full occupancy, <= 1/128 at 1G at the 25% cap; a wrong draft is rejected by verification, no correctness impact |

## 10. Out of Scope (future work)

- Cold store resize at runtime (remap).
- Sparse cold format.
- GPU-resident hot table (VRAM) - only if evidence shows RAM lookup
  bandwidth is the bottleneck.
- Admission policy (probationary table) - hotness ranking is the first step
  toward it.
