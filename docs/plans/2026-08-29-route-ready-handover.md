# Route-Ready Expert Cache Handover (2026-08-29)

Companion to `docs/plans/2026-08-23-phase-5i-handover.md`.
Plan in flight: `docs/superpowers/plans/2026-08-29-bundle-admission-manifest-v2.md`.

Audience: the agent picking this up next. Read this whole file before touching code.

---

## 1. Where things stand (one paragraph)

The expert cache dispatcher works and is observable, but the cache still delivers no
reliable throughput win. This session found and fixed a correctness bug that had been
silently corrupting every route-ready **fallback** decode step (the cache was "gaining"
~3.3% while producing garbage), added admission telemetry so engagement can be read from
benchmark columns alone, and added manifest-load outcome accounting. The remaining plan
work (bundle-admission manifest v2 + validation matrix) is blocked on a separate, now
well-localised defect in the **profiler's route capture**, not in the cache.

---

## 2. Commits landed this session

- `60e52b380` `ggml: order fallback expert down node and account manifest load`
  - stale-activation fallback fix + synchronous producer-ids read
  - admission telemetry counters + scheduler print + llama-bench columns
  - manifest loader outcome accounting + `test_manifest_loader_outcome_stats`
  - files: `ggml/include/ggml-backend.h`, `ggml/src/ggml-backend.cpp`,
    `ggml/src/ggml-backend-expert-cache.{h,cpp}`, `tests/test-expert-cache.cpp`
- `a66ac1cf7` `tools: rank pinned experts by bundle-admission coverage`
  - greedy v2 selection, `--cache-mib`/`--w-full`/`--w-hetero`/`--dump-routes`,
    host-weight eligibility census, manifest format v2 keys
  - file: `tests/test-moe-tg-profiler.cpp`
- (docs commit, see this file's parent) `EXPERT_CACHE.md` +
  `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

Build that reproduces everything:
```
cmake --build build --config Release --target test-expert-cache test-moe-tg-profiler test-backend-ops
```

---

## 3. The bug that mattered (root cause, so it never comes back)

**Symptom.** Cache-on dynamic decode (`-fitt 256 -exc 128 -excp 32 -excm ...`, seed 42)
emitted corrupt mixed-language token soup; cache-off was coherent. Measured as a **+3.3%
TG gain** while being wrong.

**Cause.** In the route-ready fallback path (`ggml/src/ggml-backend.cpp`,
`ggml_backend_sched_compute_splits`), the bundle's `down` node was executed during the
**input-copy loop**, before its activation input (`swiglu`/`act`) had been computed in the
same split. It consumed a not-yet-written buffer, i.e. the previous decode step's
activation, and folded a stale expert output into the residual stream every fallback step.

**Fix.** Defer the `down` node into the **dispatch** phase: compute the valid prefix
(gate, up, act), then run `down` on the host, uploading rows individually and **skipping
negative route ids** so untouched rows keep their prior contents. A second, smaller bug:
the dispatch route-id read raced the producer-split copy and saw garbage ids on the first
compute; fixed with a synchronous producer-ids read (`ggml_backend_synchronize` before the
`tensor_get`, matching the scheduler's existing pattern at `ggml-backend.cpp:2657`).

**Invariant to preserve.** A node must never be host-executed before all of its inputs are
materialised. If you reintroduce any early execution in the input loop, you recreate this
bug. Gate any cache path to **decode-only**, and never remap stale route ids.

**Verification performed.**
- cache-on coherence restored; cache-off control unchanged
- `test-backend-ops.exe test -b CUDA0 -o MUL_MAT_ID`: 869/869 pass
- `tools/results/expert-cache/test_run_tg_matrix.py` (bench column alignment): pass
- `test_route_ready_sidecar_full_hit` incl. negative-route sentinels: pass

---

## 4. Telemetry now available (use these, do not guess)

`ggml_backend_expert_cache_stats`:
- `n_route_ready_full_hits`, `n_route_ready_fallbacks`, `n_route_ready_mask_counts[0..8]`

llama-bench columns: `expert_cache_route_ready_full_hits`,
`expert_cache_route_ready_fallbacks`. Admission histogram prints in
`ggml_backend_sched_print_expert_cache_stats`.

Manifest load outcomes (`ggml_backend_expert_cache_get_manifest_stats`):
`n_parsed`, `n_unregistered_layer`, `n_seed_failed`, `n_seeded`,
`n_slot_lookup_failed`, `n_pinned_marked`. The loader now prints
`parsed N entries (M unique)` and `seeded K tensor slots from N parsed entries`.

**Key gotcha (from this session's test):** `seed()` needs one slot per (tensor, expert)
pair, so a complete gate/up/down bundle costs **three** slots. `test_manifest_loader_outcome_stats`
sizes the cache at 384 bytes for exactly two full bundles; if you change shapes or the
shared-pool behaviour, recompute that capacity or the test silently changes meaning.

---

## 5. Current blocker: the profiler captures degenerate routes

This is NOT a cache bug. `tests/test-moe-tg-profiler.cpp` `profiler_cb_eval` records the
`MUL_MAT_ID` ids tensor (`ffn_moe_topk-N`) and reads it, but every value comes back as 0.

Evidence:
- `MOE_PROF_DIAG=1`: `ffn_moe_topk-N` is a `GGML_OP_VIEW` (offset 0, ne=8) over
  `ffn_moe_argsort-N` (`GGML_OP_ARGSORT`, device-resident, 256 i32). Reading the full
  256-element argsort after `llama_decode` returns is still all zeros.
- Same degeneracy at `--cache-mib 0`, so the cache path is not the cause.
- Consequence: each token's route vector is eight identical ids -> the greedy saturates at
  one expert per eligible layer (33 bundles) and reports an impossible ~82.5% projected
  8/8 admissions. **Do not ship or benchmark any v2 manifest until routes are real.**

**Proven, not hypothesised.** The read path committed in `ffd217396` (which produced the
valid `pinned_experts_*.json` manifests with 220 distinct expert ids) is **byte-identical**
to the mid-graph read that now returns all zeros. The profiler code did not change; the
graph did. `ffn_moe_topk` / `ffn_moe_argsort` are now **device-resident**
(`GGML_OP_VIEW` over `GGML_OP_ARGSORT`, `p_host=0`). The original capture relied on the
`ggml_backend_buffer_is_host` fast path being taken; when routing tensors were host
buffers that was valid. The route-ready / cacheless-parity dispatcher work moved MoE
routing tensors to the GPU, so the read now falls through to a device `tensor_get` that
returns the pre-argsort scratch (zeros). This is a profiler-side staleness bug, not a
cache bug.

What was tried:
1. Read mid-graph in the callback (original code) -> zeros (device path taken now).
2. Record the ids pointer, read AFTER `llama_decode` returns -> still zeros.
3. Read the ARGSORT producer directly (bypass the view) -> still zeros.
4. Confirmed view offset is 0 and argsort is 256 i32 -> not an offset or shape bug.
5. Attempted to sync a freshly `ggml_backend_dev_init`'d CUDA backend in the callback;
   reverted before commit (re-initing a device backend per token is not how you sync, and
   the block did not parse).

Why step 2 still failed: `llama_decode` submits the graph and the scheduler may free /
reuse the compute-buffer scratch backing `ffn_moe_argsort` before the profiler reads it,
and a device `tensor_get` on scratch that has been reset reads zeros. The scheduler itself
never reads these ids after the graph - it reads them synchronously *during* split
dispatch, right after submitting the producing split, against a backend handle it already
owns: `ggml_backend_tensor_get_async(ids_backend, ...)` then
`ggml_backend_synchronize(ids_backend)` (`ggml-backend.cpp:2657`).

Likely real fix, in priority order:
   (a) capture routes through the same synchronous read the cache already performs at
       decode time (the cache reads real ids in `compute_splits`; reuse that as the source
       of truth for the profiler instead of a graph callback), or
   (b) force the routing tensors host-resident for the profiling run (`-ngl 0` / a
       profiling-only placement that keeps `ffn_moe_argsort` on a host buffer) so the
       original memcpy path is valid again. Confirm against a known-good legacy manifest.

Two things already confirmed so the next agent does not rediscover them:
- The profiler does **not** parse `-ngl` / `-ngl0`: it hardcodes
  `params.n_gpu_layers = -1` (fit). Option (b) therefore requires first adding an `-ngl`
  argument (set `params.n_gpu_layers` and disable `fit_params`) so routing tensors land on
  host buffers. A run that only passes `-ngl0` still uses fit and still yields garbage ids
  (observed: 17 distinct values, e.g. `-1181548544`, i.e. raw scratch, not zeros-only).
- The known-good legacy `pinned_experts_1024mb.json` has **220 distinct expert ids**, so a
  valid capture is broad; anything near "all the same id" or "a handful of huge negatives"
  is broken. Use it as the correctness reference.

Cleanest fix is option (a): read the ids the same synchronous way the cache does
(`ggml-backend.cpp:2657`), against a backend handle you already own, rather than trusting
a graph callback on now-device-resident routing tensors. Do not rewrite the greedy; it is
correct given valid routes.

---

## 6. Known intermittent failure (pre-existing class)

`test_route_ready_cross_split_sidecar` fails its numeric comparison ~4/5 runs (currently
around `tests/test-expert-cache.cpp:2109`): compute-two's admitted GPU path disagrees with
compute-one's host path for the synthetic two-expert graph. Matches the documented
synthetic-graph GPU-path numeric fault class (see the shared-ids test note). Production
coherence passes. Needs a debugger pass on sidecar + hetero slot reads. Not caused by this
session's changes.

---

## 7. Remaining plan work (Task 3/4 of the v2 plan)

- Task 2 (loader accounting): DONE, committed (`60e52b380`), tested.
- Task 3 (generator v2): code DONE (`a66ac1cf7`), but **gated on the route-capture blocker
  in section 5** - it produces correct manifests only from correct routes.
- Dynamic 128 MiB period-32 re-measured after the fallback fix: **+18.3% paired mean
  (20.22 -> 23.91 t/s), positive in all 3 pairs, coherence verified.** See the
  optimization log "Dynamic 128 MiB Period-32 Re-Measurement After Fallback Fix". The old
  +3.305%-on-corrupt-output number is superseded; the win is now real.
  - **Strategic finding:** the cache run reports **Full-Hit (8/8) = 0, Fallbacks = 3482**,
    mask histogram `0:1618 1:638 2:623 3:354 4:181 5:52 6:16 7:1 8:0`. The entire +18% at
    this operating point comes from the partial-admission **fallback / CPU-base** path, not
    from any GPU sidecar full hit. Automatic-fit placement never forms a complete 7/8-or-8/8
    bundle here. This is precisely the gap bundle-admission manifest v2 (Task 3) targets, so
    unblocking section 5 is the highest-value next step: it should raise the full-hit share
    from ~0 and convert the sidecar path's known +8% (3 GiB) into a larger, additive win.
- Task 4 (validation matrix): blocked until v2 manifests are trustworthy.

Benchmark config (repo rule): mirror the `qwen3.6-35b-a3b-presets-exc-latest.ini` compact
preset; use `tools/results/expert-cache/run-tg-matrix.py`; keep
`GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL` off; load with `mmap`. Target is >=20 TK/s (control
sits right at that line; cache-on clears it). Throughput is `n_gen / (avg_ns/1e9)` from the
first JSON record of each run file; the admission histogram is in the plain-text telemetry
block appended after it.

---

## 8. Constraints (do not violate)

- **No subagents / no worktrees for this work.** Implement inline.
- **Never `git commit` / `git push` without explicit user approval** (AGENTS.md). This
  session had approval for these commits only.
- Any per-layer synchronous ids D2H read inside the split loop MUST be decode-gated; an
  ungated one regressed PP32 from ~64 to ~43 t/s by stalling the CUDA stream.
- Save graph nodes before mutating them; never remap stale route ids.
- Keep `EXPERT_CACHE.md` and `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` current - the log records
  every attempt, including ones you revert.
