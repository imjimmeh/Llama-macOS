# Learned predictor runtime utilization

Status: partially complete (Phase 5F done 2026-08-22)
Branch: `feat/expert-cache-only`
Outcome: `predictions_generated > 0` achieved (CSV fix + cb layer extraction + host-buffer guard + submit-all routing). `predictions_used > 0` blocked by Phase 5F.5 (cache does not engage during single-token gen; see EXPERT_CACHE_OPTIMIZATIONS_LOG.md Phase 5F.5 for hypotheses). Phase R3 bench matrix NOT run yet (blocked on Phase 5F.5).

## Goal

`predictions_generated > 0` and `predictions_used > 0` during real inference, with measurable prefetch hit rate and decode tok/s improvement over baseline (no predictor).

## Why now

Trace pipeline verified end-to-end:

- `route_trace.bin` (25852 B, 91 RTRC v2 entries)
- `training_data.bin` (23 samples)
- `model.bin` (LRPD v2, input_dim=256, rank=32, num_experts=256, 66 KB)
- Loader in `ggml-routing-predictor.cpp` now accepts v2 and reads `(input_dim, rank, num_experts)` in trainer order

Bench with model loaded:
- `routing_predictor_variant=1` (LOW_RANK_MLP)
- `predictions_generated=0, predictions_used=0, expert_cache_requests=0`

So: predictor init works, but the eval-time chain that should drive `predictions_generated` is not firing.

## Current wiring (verified, traced through source)

| # | Component | File:line | Notes |
|---|---|---|---|
| 1 | Predictor init | `src/llama-graph.cpp:1492-1511` | Variant B selected when model path set |
| 2 | Eval callback registered | `src/llama-context.cpp:1415-1419` | Overrides `cparams.cb_eval` (NULL by default) |
| 3 | Callback filter | `src/llama-graph.cpp:3990-4095` | Only matches tensor name `ffn_moe_logits` |
| 4 | Predict invocation | `src/llama-graph.cpp:4061-4069` | `ggml_routing_predictor_predict()` |
| 5 | Counter + submit | `src/llama-graph.cpp:4071-4091` | `add_predictions_generated` + `sched_submit_prediction` |
| 6 | Sched forward | `ggml/src/ggml-backend.cpp:2666-2684` | Forwards to cache's `submit_prediction` |
| 7 | Cache prefetch | `ggml/src/ggml-backend-expert-cache.cpp:2312-2358` | Looks up `bundle_registrations[layer]`, calls `prefetch_async` for gate/up/down |
| 8 | Stats getter | `ggml/src/ggml-backend.cpp` (around line 2410) | `ggml_backend_sched_get_routing_predictor_stats` returns 0s |

Steps 1, 2 log fine (93 Variant B init, all 40 layers present). Steps 3-8 must be the gap.

## Diagnostic (must do first; gated plan below)

Decision point: does the callback actually run?

### D1: One-shot call counter

Edit `src/llama-graph.cpp:3990-4095` — add a static call counter that prints the first call to stderr under `GGML_PREDICTOR_DEBUG`. Block goal: confirm `ggml_backend_sched_compute_splits` ever invokes it. Code shape:

```cpp
static int s_cb_calls = 0;
if (getenv("GGML_PREDICTOR_DEBUG") && ++s_cb_calls <= 5) {
    fprintf(stderr, "[cb] ask=%d name=%s n_pass=%d\n", ask,
        tensor && tensor->name ? tensor->name : "(null)", tensor ? (int)tensor->ne[1] : -1);
}
```

Build, run server with `GGML_PREDICTOR_DEBUG=1`, send 1 completion with 16 tokens, grep stderr.

### D2: Branching by outcome

- **D2.a**: 0 calls. Move up the stack.
  - D2.a.i: `alloc_graph` cached an old splitter that did not have our callback. Verify in `ggml_backend_sched_split_graph` whether splits persist across graph re-builds, and whether `compute_splits` is called every time `graph_compute_async` runs. (Reading `ggml-backend.cpp:2388-2401` shows: yes, called every time `!is_alloc`.) So compute_splits SHOULD run on first decode. Then the cb IS set on the sched. So... investigate `n_reused` path in `llama-context.cpp:1387-1397` — does that branch reuse an old `gf` whose splits were computed with no callback?
  - D2.a.ii: `cparams.cb_eval_user_data` mismatch. Check whether the `res` passed to the callback has a valid `routing_predictor`. If user passes a cb_eval, `prev_cb` chains correctly, but ours might miss the `routing_predictor_callback` registration because of guard.
  - D2.a.iii: Fork/cudart race condition. Server forking makes the res pointer stale.
- **D2.b**: >0 calls but never `ask=true` returning true for ffn_moe_logits. Then `is_moe_logits` filter is wrong. Possibly the qwen35moe code path uses `gate_up_exps`-based routing that emits the matrix-multiplied gate with a DIFFERENT name. Run with extra log line dumping `tensor->name` for every ask=true call, find which names appear.
- **D2.c**: ask=true returns true for ffn_moe_logits, but observation pass `ask=false` never reaches the predictor. Re-read `ggml-backend.cpp:2188-2193`: observation is gated on `need`. Need becomes true when ask was true. The while loop exits when `need==true`. So observation SHOULD run. (Unless the `t` variable in the observation call points to a different tensor than the one for which need==true.) Add observation hit log and confirm.
- **D2.d**: Observation runs but `n_predicted==0` from the predictor. Wrong input_dim? Wrong feature extraction? Check `extract_features` output statistics.

### D3: Cache submit ignores predictions

Even if predictions are GENERATED, prefetch must actually target the right cache slots. The cache has `bundle_registrations[target_layer]` lookup at `ggml-backend-expert-cache.cpp:2351`. Empty registry → no prefetch. Most likely cause: `register_bundle` was never called for the predicted layer. Verify with a log: count `bundle_registrations.size()`.

## Plan (after D1-D3 resolve)

### Phase R1: Minimal fix — make the chain fire

Most-likely fix path based on D2.a.i (splits reuse old callback):

1. In `src/llama-context.cpp:1395-1397` (graph-reuse branch), re-register the routing cb BEFORE the next `graph_compute_async` call. Currently the code only re-registers on a fresh graph. Move the predicate check OR register unconditionally.
2. Build, bench, expect non-zero `predictions_generated`.

Backup fixes:
- If D2.a.ii (user prev_cb): make `prev_cb` chain end-of-callback observable.
- If D2.b: add support for the alternate tensor name.
- If D2.c: refactor scheduler to track per-node need state instead of stale `t`.

### Phase R2: Settle accounting

Once predictions generate, they need to be CONSUMED.

1. Add `ggml_backend_expert_cache_settle_prediction` call site in `ggml_backend_sched_compute_splits`. Find the MUL_MAT_ID node for the predicted layer (already have `ggml_backend_find_mul_mat_id_node` after fix #2). For each expert in the MUL_MAT_ID's selected set, check `pending_predictions[layer]` and update `n_speculative_prefetches`/`n_speculative_hits` stats.
2. Add CSV columns for `n_speculative_prefetches`, `n_speculative_hits`.

### Phase R3: Bench matrix

```
build/bin/Release/llama-bench.exe -m <model> -p 512 -n 128 \
    -fitt 256 -exc 256 -excp 64 -o csv

build/bin/Release/llama-bench.exe -m <model> -p 512 -n 128 \
    -fitt 256 -exc 256 -excp 64 \
    --routing-predictor-model tools/training_data/model.bin \
    --routing-predictor-variant low-rank-mlp --routing-predictor-stats \
    -o csv

build/bin/Release/llama-bench.exe -m <model> -p 512 -n 128 \
    -fitt 256 -exc 256 -excp 64 \
    --routing-predictor-model tools/training_data/model.bin \
    --routing-predictor-horizon 4 --routing-predictor-stats \
    --routing-predictor-variant low-rank-mlp -o csv
```

Report:
- `avg_ts` per config
- `predictions_generated`, `predictions_used`
- `n_speculative_prefetches`, `n_speculative_hits`
- `expert_cache_*` rates

### Phase R4: If predictor accuracy is low

23 training samples is one run. Expand:
- `--max-tokens 200 --prompt-repeat 50` → larger trace
- Retrain at `--rank 64 --epochs 500`
- Repeat Phase R3 matrix

## Open questions

1. Is the cb registered with the right `user_data` on the FIRST compute path? (D2.a.i covers this.)
2. Does `qwen35moe` emit a `ffn_moe_logits` tensor? (Confirmed: yes via `build_moe_ffn` → `cb(logits, "ffn_moe_logits", il)` at `src/llama-graph.cpp:2143`.)
3. Stats getter is reading from where? If it reads from `sched->routing_predictor_stats` but the actual writes go to `cache->stats`, the getter returns 0 even when work is happening. Verify mapping in `ggml-backend.cpp:2410-2421`.

## Out of scope

- Replacing the heuristic predictor with the learned one (currently both share Variant B slot).
- Multi-token look-ahead (H > 8).
- Per-token confidence thresholding.
