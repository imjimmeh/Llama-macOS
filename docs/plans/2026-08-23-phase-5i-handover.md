# Phase 5I Handover (2026-08-23)

Active llama.cpp tuning branch: `feat/expert-cache-only`. Goal: prove
`expert_cache_gpu_slot_exec_from_prediction > 0` during decode so the matrix
finally compares predictor lookahead vs reactive rather than reactive vs
reactive.

## Where we are right now

Plumbing commits on top of `babd073e1` (5G):

```
ad60cc644 ggml : Phase 5I predictor engagement + execution attribution
51a0aab9c ggml : Phase 5I attribution per-expert (fix from_pred=0)
```

Working tree has **uncommitted** changes for the `GGML_OP_OFFLOAD_MIN_BATCH=1`
carve-out (review item 7). The carve-out is mid-debug and **does not work yet**.

**Proven (committed):**
- Predictive fills issue (`prefetch_issued=19338` for `-p 32 -n 16`).
- `gpu_slot_exec_from_prediction > 0` (549 for `-n 16` with predictor on vs
  486 with predictor off — the +13% delta is the predictor contribution).
- Sum invariant: `from_pred + reactive == n_valid` (routed experts served).
- Attribution counters in CSV, subtracted in llama-bench, aggregated in
  `ggml_backend_sched_get_expert_cache_stats`.

**Open and being debugged (uncommitted, build green):**
- `GGML_OP_OFFLOAD_MIN_BATCH` per-op carve-out. Currently fails because:
  - (a) decode `MUL_MAT_ID` for qwen35moe has `ne[1]==8`, not 1.
  - (b) `op->src[0]->extra` is **always NULL** in the offload gate — the
    sentinel set by `ggml_backend_expert_cache_get_or_create_pool` is not
    reaching the source tensor the gate sees.

## Hardware, model, command

```
CPU: AMD Ryzen 7 5700X (8 cores, 14 threads via -n 14 not used here)
GPU: NVIDIA GeForce GTX 1080, 8 GiB VRAM, SM 6.1
RAM: 32 GiB
OS:  Windows 11 10.0.26200
Terminal: Windows Terminal

Model: C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/
       Qwen3.6-35B-A3B-APEX-Compact.gguf
       qwen35moe 35B.A3B Q4_K, 40 layers, 256 experts, 8 used.

Predictor: tools/training_data/model.bin (LRPD v2), variant low-rank-mlp.

Canonical bench command (committed code, works):
  GGML_OP_OFFLOAD_MIN_BATCH=1 ./build/bin/Release/llama-bench.exe \
    -m "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" \
    -p 32 -n 16 -fitt 256 -exc 256 -excp 64 \
    --routing-predictor-model tools/training_data/model.bin \
    --routing-predictor-variant low-rank-mlp --routing-predictor-stats \
    -r 1 -o csv
```

Deployment preset: `G:\qwen3.6-35b-a3b-presets-exc.ini`, `[qwen3.6-35b-apex-compact]`
section: `fit=on, fit-target=256, exc=256M, fa=on, threads=14`. **No `-ngl`** —
`fit` alone decides placement.

## What's in each commit

### ad60cc644 — predictor engagement + execution attribution

Files: 9, +354/-27.

- `ggml/include/ggml-backend.h`: 3 new fields in
  `ggml_backend_expert_cache_stats`:
  - `n_gpu_slot_exec_from_prediction` (slot placed by completed prefetch)
  - `n_gpu_slot_exec_reactive` (slot placed reactively / LRU / seed)
  - `n_prefetch_src_not_host` (prefetch skipped: source weights not host)
- `ggml/src/ggml-backend-expert-cache.h`: 4 new APIs:
  - `ggml_backend_expert_cache_was_prefetched(cache, tensor, eid)` — bool
  - `ggml_backend_expert_cache_prefetch_slot_count(cache)` — int32
  - `ggml_backend_expert_cache_record_gpu_slot_from_prediction(cache)`
  - `ggml_backend_expert_cache_record_gpu_slot_reactive(cache)`
- `ggml/src/ggml-backend-expert-cache.cpp`:
  - Reworked `prefetch_async` backend-agnostic. Old `#if defined(GGML_USE_CUDA)`
    block was dead code when compiling into `ggml-base.dll` (objdump-verified
    no CUDA imports). New non-CUDA branch uses
    `ggml_backend_tensor_set_async(cache->backend, pool->tensor, ...)` +
    immediate `state=RESIDENT` marking + push into `prefetch_slots`.
  - Source-weight host check at top of `prefetch_async`: skip if
    `tensor->buffer` is non-host (device pointer would crash a CPU memcpy).
  - Host-to-host skip guard: skip if both source and pool buffer are host
    (NULL pool buffer = carved from device cache backing = proceed).
  - `was_prefetched` hardened: matching entry must also have `find_slot >= 0`
    so stale entries after eviction don't produce false positives.
  - Stats update: increment `n_prefetch_issued` in non-CUDA branch.
- `ggml/src/ggml-backend.cpp`:
  - Attribution gate relocated from `src/llama-graph.cpp:4077-4079` to
    `ggml-backend.cpp:1974-2125` (zero-copy classification site).
  - `ggml_backend_sched_get_expert_cache_stats` now aggregates all new
    fields across backends (was missing `wasted_prefetch_bytes`,
    `in_flight_wait_us`, `n_prefetch_issued`, `n_prefetch_src_not_host`).
- `tools/llama-bench/llama-bench.cpp:1651`: added `EXPERT_CACHE_SUBTRACT(
  n_gpu_slot_exec_reactive)`. (Omission was the "delta always 0" bug.)
- `summarize_matrix.py`: row-pair delta summariser (was already present;
  just promoted in this commit).

### 51a0aab9c — attribution per-expert (fix from_pred=0)

Files: 4, +87/-44.

- `ggml/src/ggml-backend.cpp:1988-2035`: attribution switched from
  all-or-nothing (`n_from_pred == n_valid` per decode step) to per-expert.
  Inside the zero-copy loop, `was_prefetched(eid)` increments from_pred and
  calls `record_gpu_slot_from_prediction`; every other routed expert
  calls `record_gpu_slot_reactive`. Strict gate after the loop removed.
- `ggml/src/ggml-backend-expert-cache.cpp`: reverted temporary
  wp-dbg/pa-ptr/zc-ptr diagnostic blocks (added then removed during
  diagnosis).
- `EXPERT_CACHE.md`: Phase 5I section added (root-cause narrative,
  outcome table, pointer-mismatch-disproven note).
- `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`: Phase 5I session block expanded.

## What was disproven

The previous session's "leading hypothesis" was tensor pointer mismatch —
bundle registrations in `src/llama-context.cpp:610-650` use
`model.layers[il].ffn_gate_exps` (host originals), but the graph's
MUL_MAT_ID src0 might be a fit-promoted copy. This session verified
**directly** with paired `[pa-ptr]` and `[zc-ptr]` fprintf probes:
bundle-registered tensors and graph src0 pointers are the same object
per layer (e.g. layer 11 both `00000210665CB910`, layer 12 both
`00000210665CBD60` / `00000210665CBED0` / `00000210665CBBF0`).

`[wp-dbg]` instrumentation in `was_prefetched` showed it returning TRUE
many times per step (e.g. `layer=11 exp=180 entry=yes resident=8`),
yet `from_pred=0`. The real cause was the strict all-or-nothing
attribution gate at the bottom of the zero-copy loop: the predictor
recalled ~22% of routed experts, never all 8, so the gate never closed.
Per-expert attribution fixed it.

## The carve-out work — current state (uncommitted)

### Goal

Replace the global `GGML_OP_OFFLOAD_MIN_BATCH=1` knob with a per-op
carve-out. The knob currently offloads MUL_MAT_ID during decode by force,
but it also over-broadly offloads unrelated small-batch ops.

### What's in place

- `ggml/src/ggml-backend-expert-cache.h:13-18`: header sentinel
  ```cpp
  #define GGML_EXPERT_CACHE_SLOT_TENSOR_EXTRA_VALUE \
      ((void *)(uintptr_t)0x4558504552545344ULL)
  static const void * const ggml_backend_expert_cache_slot_tensor_extra =
      GGML_EXPERT_CACHE_SLOT_TENSOR_EXTRA_VALUE;
  ```
- `ggml/src/ggml-backend-expert-cache.cpp:301-304`: every pool tensor
  gets `ptensor->extra = const_cast<void *>(ggml_backend_expert_cache_slot_tensor_extra)`.
- `ggml/src/ggml-cuda/ggml-cuda.cu:4`: includes
  `ggml-backend-expert-cache.h`.
- `ggml/src/ggml-cuda/ggml-cuda.cu:5327-5344`: carve-out (current debug
  version):
  ```cpp
  static bool ggml_backend_cuda_device_offload_op(...) {
      ggml_backend_cuda_device_context * dev_ctx = ...;
      static int dbg_total = 0;
      if (op->op == GGML_OP_MUL_MAT_ID && dbg_total++ < 24) {
          fprintf(stderr, "[carve-dbg] MUL_MAT_ID ne1=%lld src0_extra=%p sentinel=%p match=%d\n",
              (long long)op->ne[1],
              op->src[0] ? op->src[0]->extra : NULL,
              ggml_backend_expert_cache_slot_tensor_extra,
              op->src[0] && op->src[0]->extra == ggml_backend_expert_cache_slot_tensor_extra);
      }
      if (op->op == GGML_OP_MUL_MAT_ID && op->ne[1] == 1 &&
          op->src[0] != NULL && op->src[0]->extra == ggml_backend_expert_cache_slot_tensor_extra) {
          return true;
      }
      return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
  }
  ```

### What's wrong

Debug output (running without `MIN_BATCH=1`, all other args as canonical):
```
[carve-dbg] MUL_MAT_ID ne1=8 src0_extra=0000000000000000 sentinel=4558504552545344 match=0
```

Two issues:

1. **`ne[1]==8`, not 1.** Decode MUL_MAT_ID for qwen35moe folds top-k=8
   into a small batch dim. The carve-out gate `op->ne[1] == 1` never matches.
   Fix: change to `op->ne[1] <= 8` (or `op->ne[1] <= 16` for safety).

2. **`src0_extra == NULL`.** Pool tensor was supposed to have
   `extra = sentinel`, but the gate sees NULL. Either:
   - The sentinel is being overwritten between pool creation and graph
     build (something in `ggml_new_tensor_3d` resets `extra`).
   - `op->src[0]` is NOT the pool tensor — it might be a view or a
     `ggml_cont()` of it (which clears `extra`).
   - `op->src[0]` is the **original** weight tensor because the slot install
     didn't actually happen at this op (the carve-out test runs against
     pre-install MUL_MAT_ID ops, or against a different op path).

### How to debug

Add probes in:
- `ggml/src/ggml-backend-expert-cache.cpp` near `ptensor->extra = ...`
  (just after) to log `(void*)ptensor, (void*)ptensor->extra`. Run a
  predictor-enabled bench with `GGML_PREDICTOR_DEBUG=1`.
- `ggml/src/ggml-backend.cpp:2087-2100` near `node->src[0] = slot_tensor`
  to log `(void*)node->src[0], (void*)node->src[0]->extra,
  (void*)slot_tensor, (void*)slot_tensor->extra`. This is where the
  zero-copy branch swaps in the slot tensor.
- If the `slot_tensor->extra` matches sentinel but `node->src[0]->extra`
  is NULL after install, the install path or a `ggml_cont()`/`ggml_view_*`
  clears it. If both are NULL, the sentinel was never set (the assignment
  in pool creation isn't running, or is being clobbered).

The slot tensor is created in
`ggml/src/ggml-backend-expert-cache.cpp` `get_or_create_pool` around
lines 296-304. The `pctx` is local to that function — if anything inside
it copies the tensor (it doesn't today), `extra` may not survive. The
tensor returned via `pool->tensor` is the same `ptensor`, so that should
be fine.

### Plan B (if sentinel can't be made to survive)

Drop the `extra` sentinel. Use a different discriminator that does not
rely on `extra`:
- Check `op->src[0]->ne[2] > 1` (the expert weight tensor's 3rd dim is
  the expert count). Plain linear weights are 2D so `ne[2] == 1` there.
- Tighten to `op->op == GGML_OP_MUL_MAT_ID && op->ne[1] <= 8 &&
  op->src[0]->ne[2] > 1`.

This is structurally clean (no shared sentinel state) and matches the
real shape difference between MoE weights and other weight tensors. It
may have false positives if some non-MoE op also has src0 with
`ne[2] > 1`, but I haven't found any in llama.cpp.

## Files touched by uncommitted work

```
M ggml/include/ggml-backend.h                    (Phase 5I committed)
M ggml/src/ggml-backend.cpp                       (Phase 5I committed)
M ggml/src/ggml-backend-expert-cache.cpp          (Phase 5I committed + pool-tensor-extra in uncommitted work)
M ggml/src/ggml-backend-expert-cache.h            (Phase 5I committed + sentinel in uncommitted)
M ggml/src/ggml-cuda/ggml-cuda.cu                 (uncommitted: include + carve-out)
M src/llama-graph.cpp                             (Phase 5I committed)
M tools/llama-bench/llama-bench.cpp               (Phase 5I committed)
M EXPERT_CACHE.md                                  (Phase 5I committed)
M EXPERT_CACHE_OPTIMIZATIONS_LOG.md               (Phase 5I committed)
?? summarize_matrix.py                            (Phase 5I committed)
```

`git status --short` will show only the uncommitted items:
```
M ggml/src/ggml-backend-expert-cache.cpp
M ggml/src/ggml-backend-expert-cache.h
M ggml/src/ggml-cuda/ggml-cuda.cu
```

## How the predictor plumbing works (for the next agent)

### Submit path

1. `src/llama-graph.cpp:4029+`: `routing_predictor_callback` runs on every
   named tensor with `ne[1] == 1`. Detects MoE router logits tensor
   (`ffn_moe_logits-N`), reads logits via `tensor_get_async` (handles
   device pointer), calls `ggml_routing_predictor_predict` (LRPD v2
   model), calls `ggml_backend_sched_submit_prediction` (which forwards
   to every non-NULL expert cache).

2. `ggml/src/ggml-backend-expert-cache.cpp::ggml_backend_expert_cache_submit_prediction`:
   records `n_predictions_submitted`, queues a pending entry, then calls
   `prefetch_async` on each registered bundle tensor (gate, up, down)
   for that layer.

3. `prefetch_async`: source-weight host check → pool fetch → host-pool
   skip guard → per-expert loop:
   - CUDA branch (dead code in this build): pinned staging buffer +
     `cudaMemcpyAsync` + `cudaEvent_t`.
   - Non-CUDA branch (active): `ggml_backend_tensor_set_async(
     cache->backend, pool->tensor, pinned_buf or src, dst_off, expert_size)`.
     Stream-ordering on `cache->backend` (which is the GPU backend here)
     guarantees residency before the consumer MUL_MAT_ID node executes.
     Pushes `prefetch_slot{state=RESIDENT, ready_event=NULL}` into
     `prefetch_slots` and increments `n_prefetch_issued`.

### Settle path

`ggml/src/ggml-backend.cpp:1820+`: in `ggml_backend_sched_compute_splits`,
for each split input that is a host expert weight, find the
`MUL_MAT_ID` consumer (`find_mul_mat_id_node`). If the cache is non-NULL
and `ids_tensor->ne[1] == 1` (decode), enter the zero-copy classification
site at `ggml/src/ggml-backend.cpp:1974-2125`:

- Read `ids` via `tensor_get_async` (decodes which experts per layer).
- For each routed expert:
  - `find_slot(cache, input, eid)` — keyed by `(input, eid)`. Note that
    `input` here is the **host** weight tensor; pool fetches use the
    same key, so lookups match.
  - If `slot >= 0`: hit path. `was_prefetched(input, eid)` returns true
    if a `prefetch_slots` entry for this `(tensor, eid)` is RESIDENT or
    IN_FLIGHT AND `find_slot(...) >= 0` (resident in pool).
  - Else: `is_prefetch_ready`, `has_inflight_prefetch` + `wait_prefetch`,
    `alloc_slot_idx` (reactive fill).
- Attribution: per-expert. `was_prefetched` -> `from_pred++` and
  `record_gpu_slot_from_prediction`; else `record_gpu_slot_reactive`.
- If all routed experts have a slot: install slot tensor:
  `node->src[0] = slot_tensor`, `remapped_ids[i] = slot_idx`,
  `ggml_backend_tensor_set_async(split_backend, ids_tensor, ...)`.
- `record_gpu_slot_execution` once per node (counter for total
  slot-pool-attended executions; sum of `from_pred + reactive` should
  equal this within `n_valid` terms — they share the same per-node loop).

### Stats aggregation

`ggml_backend_sched_get_expert_cache_stats` (in `ggml-backend.cpp`)
walks `sched->n_backends` and sums every field across non-NULL
`expert_caches[b]`. All 5I fields are aggregated here.

### CSV flow

`tools/llama-bench/llama-bench.cpp`:
- `get_fields()` and `get_values()` MUST include all new columns in
  the same order. Mismatch = shifted CSV (Phase 5F.1 lesson).
- `EXPERT_CACHE_SUBTRACT(stat_name)` macros are called once per new
  counter in `subtract_expert_cache_stats` to produce per-test deltas.
  Omission = the field stays at 0 in CSV (Phase 5F/5I lesson).
- `EXPERT_CACHE_DISPATCH(column)` macros map `out` to the field for
  printing.

## CSV columns added in 5I (exact names, lock-step order)

In `ggml_backend_expert_cache_stats` (`ggml/include/ggml-backend.h:423-425`):
- `n_gpu_slot_exec_from_prediction`
- `n_gpu_slot_exec_reactive`
- `n_prefetch_src_not_host`

In llama-bench CSV: `expert_cache_gpu_slot_exec_from_prediction`,
`expert_cache_gpu_slot_exec_reactive`, `expert_cache_prefetch_src_not_host`.

When adding a new column in the future, update:
1. `ggml/include/ggml-backend.h` struct.
2. `ggml/src/ggml-backend-expert-cache.cpp` `record_*` functions.
3. `ggml/src/ggml-backend.cpp` `ggml_backend_sched_get_expert_cache_stats`.
4. `tools/llama-bench/llama-bench.cpp::test::get_fields()`.
5. `tools/llama-bench/llama-bench.cpp::test::get_values()`.
6. `tools/llama-bench/llama-bench.cpp::subtract_expert_cache_stats`
   (add `EXPERT_CACHE_SUBTRACT(field)` in lockstep).
7. `tools/llama-bench/llama-bench.cpp` printer EXPERT_CACHE_DISPATCH
   (one line per field).
8. If the field needs to influence per-test config: also wire through
   `cmd_params_instance` and `--expert-cache-stats` if not already.

## Predictor training data and config

The `tools/training_data/model.bin` is the trained LRPD v2 predictor
(input_dim=256, rank=32, num_experts=256, horizon=8). It was trained
with `tools/train_routing_predictor.py`; see Phase 5F/5F.1 notes in
`EXPERT_CACHE_OPTIMIZATIONS_LOG.md`. Predictor path:

```
src/llama-graph.cpp::routing_predictor_callback
  -> ggml_routing_predictor_predict(LRPD, ...)
  -> ggml_backend_sched_submit_prediction(sched, layer, ids, n, conf)
  -> ggml_backend_expert_cache_submit_prediction(cache, layer, ids, n, conf)
  -> prefetch_async(cache, bundle->gate|up|down, ids, n, layer)
```

Confidences are very low (~0.01). Predictor output quality is a
Phase 5J concern.

## Files for the next agent to read first

1. `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` lines 1285-1372 — current session
   block.
2. `EXPERT_CACHE.md` lines 783-855 — Phase 5I section with attribution
   outcome table.
3. `ggml/src/ggml-backend-expert-cache.cpp::ggml_backend_expert_cache_prefetch_async`
   — the new backend-agnostic predictive fill.
4. `ggml/src/ggml-backend.cpp::ggml_backend_sched_compute_splits`
   zero-copy branch (~lines 1820-2125) — the attribution site.
5. `src/llama-graph.cpp::llm_graph_context::routing_predictor_callback`
   (~lines 4029+) — the predictor entry point.

## Next-step checklist for the next agent

1. **Diagnose carve-out sentinel.** Run with `GGML_PREDICTOR_DEBUG=1` and
   log `(void*)ptensor->extra` in `get_or_create_pool` after the
   sentinel set. Then log `(void*)slot_tensor->extra` and
   `(void*)node->src[0]->extra` at the install site in
   `ggml-backend.cpp`. Identify where the sentinel is lost (if it is).
2. **If sentinel doesn't survive:** fall back to Plan B (ne[2] > 1).
3. **Once carve-out works:** remove all `[carve-dbg]` fprintf blocks and
   the sentinel machinery if Plan B was used.
4. **Run 5-way matrix A-E** per review item 9:
   - A: predictor off, reactive GPU slot (`--no-routing-predictor-model`).
   - B: predictor on, prefetch disabled.
   - C: predictor on, prefetch enabled, force CPU (`GGML_EXPERT_EXEC_FORCE_CPU=1`).
   - D: predictor on, prefetch enabled, GPU slot (current default).
   - E: oracle pre-resident, GPU slot (5G oracle).
   - Use `summarize_matrix.py` (repo root) to print pairwise deltas.
5. **Re-baseline PP** with predictor disabled, confirm ~60-64 t/s band
   (the Phase 5H regression check). If regressed, bisect against
   `babd073e1` (5G).
6. **Scope the matrix** to commit the carve-out as one commit
   (e.g. `ggml : Phase 5I per-op MUL_MAT_ID slot-pool carve-out`),
   `Assisted-by: Claude` trailer, no push.
7. **No commit without explicit human approval per AGENTS.md.** Same for
   push, PR, gh pr create. Always use `Assisted-by:` not `Co-authored-by:`.