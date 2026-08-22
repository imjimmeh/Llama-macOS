# Phase 5H - Opportunistic GPU Slot Execution

Status: planned (2026-08-22)
Parent: docs/plans/2026-08-22-runtime-utilization.md
Branch: feat/expert-cache-only

## Goal

Convert confirmed prediction consumption (`predictions_used=4/4`) and the validated
execution-shape win (oracle: GPU slot 142 us vs CPU 264 us, ~1.75x per MoE FFN) into
real end-to-end TG improvement. Success is `gpu_slot_executions > 0` plus measured
ready/late/wrong economics, not more predictor features.

## Starting position (verified)

- Zero-copy GPU execution already exists: `ggml-backend.cpp:1962-2062` finds slots,
  remaps ids, swaps `node->src[0] = slot_tensor`, restores after compute.
- It only fires when an expert weight enters `split->inputs`, i.e. cross-backend
  copies. Whole-layer CPU offload (`-ngl 20`) never satisfies this.
- Route discovery (commit babd073e1) scores/settles same-backend weights, decode-only.
  It cannot enable GPU execution for those nodes - they sit in CPU splits.
- Async DMA pipeline exists: dedicated stream + events
  (`ggml-backend-expert-cache.cpp:1421-1531`), readiness polling
  (`is_prefetch_ready`, :1533), sync wait (`wait_prefetch`, :1570).
- Learned predictor submits per-layer predictions with gate/up/down prefetches
  (`submit_prediction`, :2350-2356). Bundle registration tracks the three tensors
  separately (:121-125).

Consequence: 5H deploys in the config the cache was built for - MoE expert weights
pinned to host, their MUL_MAT_ID executed on the CUDA split (`-ncmoe N` /
`--ffn-split`), then instruments and tunes. No new execution subsystem.

## M0 - Config bring-up: prove the existing path engages

Run the verified bench with `-ngl 99 -ncmoe 28` (all non-MoE on GPU, last 28 MoE
layers' expert tensors on CPU; tune N to fit 8 GiB) plus predictor flags:

```
build/bin/Release/llama-bench.exe -m <model> -p 32 -n 64 \
  -fitt 256 -exc 256 -excp 64 -ngl 99 -ncmoe 28 \
  --routing-predictor-model tools/training_data/model.bin \
  --routing-predictor-variant low-rank-mlp --routing-predictor-stats -r 1 -o csv
```

Pass: gen row has `expert_cache_requests > 0`, `expert_cache_zero_copy_hits > 0`,
`predictions_used > 0`.

If the gate still misses, likely causes and fix sites:
- Weight buffer usage not WEIGHTS -> check `is_eligible` at `ggml-backend.cpp:1832`.
- `find_mul_mat_id_node` NULL (name/shape mismatch) -> `ggml-backend.cpp:1620-1678`,
  compare against `[diag_expert_src]` naming rules from 5F.7.
- Slot pool never seeded -> verify `alloc_slot_idx` eviction pressure at
  `ggml-backend-expert-cache.cpp:879-960`; 8 slots/shape suffice for horizon-1.

Deliverable: one bench line proving engagement. Everything else builds on it.

## M1 - Economics instrumentation

Stats struct: `include/ggml-backend.h` (`ggml_backend_expert_cache_stats`,
`ggml_routing_predictor_stats`). CSV: `tools/llama-bench/llama-bench.cpp`
`get_fields()` AND `get_values()` must both be updated (positional printer;
the 5F.1 `dma_wait_ns` bug is the cautionary tale).

New counters (cache stats):

```
n_gpu_slot_executions   // per MUL_MAT_ID node actually swapped to slot tensor
n_cpu_fallbacks         // node wanted zero-copy, could not get it
n_used_ready            // every requested id resident at arrival
n_used_in_flight        // >=1 id arrived while DMA in flight and we waited
n_used_miss             // >=1 id absent and not in flight -> CPU fallback
n_already_resident      // predicted id was resident before prefetch (dedup waste)
wasted_prefetch_bytes   // prefetched ids never used at settle time
```

Classification point: the per-id loop at `ggml-backend.cpp:1967-2043`. Before
`alloc_slot_idx` (:1991), each id is exactly one of:
- `find_slot >= 0` (:1974) -> READY (already counted as zero_copy_hit; keep)
- `is_prefetch_ready` true (:1980) -> IN_FLIGHT; wrap the transition in a timed
  wait and record `remaining_wait_us`
- else -> MISS candidate; see M2 for policy

Increment `n_gpu_slot_executions` once per successful swap (:2045-2061),
`n_cpu_fallbacks` when `used_zero_copy == false` after the single-token branch.

DMA timestamps: extend `ggml_expert_cache_prefetch_slot` (:128-134) with
`t_submit_us`, and set `t_finish_us` when `cudaEventQuery` first reports success
(:1548). Slack = layer start time - t_finish_us; log distribution, not just mean.

Prediction quality (routing_predictor stats): add
`USED_READY / USED_IN_FLIGHT / USED_MISS / PREDICTED_WRONG / PREDICTED_DUPLICATE`
breakdown fed from settle (`settle_prediction`, :2371-2439 already classifies
wrong/resident; extend it to emit the five-way split).

## M2 - Never-stall guarantee

Directive: speculative transfer must never block execution.

- In the zero-copy branch, treat MISS as fallback: if any requested id is neither
  resident nor ready, stop issuing DMAs for that node, leave `used_zero_copy=false`,
  take the generic path (:2064+). Slots allocated for partial hits remain valid for
  the next token. Do NOT call `wait_prefetch` anywhere in the decode path; audit its
  callers (currently none in the hot loop - keep it that way).
- The existing inline-miss behavior (:1991-2037) issues a fresh synchronous-ish DMA
  on the compute stream. Under M2 policy this becomes the cold-start cost only for
  tokens where ALL ids miss nothing in flight; acceptable, but count it
  (`n_cold_dma`) so the matrix can price it.
- Wrong/late predictions degrade to CPU execution with zero added latency beyond
  the wasted background bandwidth (priced in M4).

## M3 - Prefetch width control

Hardcoded widths to plumb into one knob (env `GGML_EXPERT_PREFETCH_WIDTH`, default 16):
- heuristic predictor callsite `ggml-backend.cpp:1929-1930` (`predict_experts(..., 16)`)
- route discovery helper `ggml/src/ggml-backend.cpp:1716-1719` (same constant)
- learned predictor submit path in `src/llama-graph.cpp` (~:4099 area, its own top-k)

Sweep 8/10/12/16 at M4; report recall vs wasted bytes vs TG. Expect an interior optimum
because wrong-prefetch bytes compete with useful DMA for PCIe.

## M4 - End-to-end matrix (before any further changes)

Four configurations, alternating runs (>=5 each) for significance:

```
A  predictor OFF, cache OFF or ON but no predictor     -> pure baseline TG
B  predictor ON, prefetch ON, force CPU execution      -> predictor+DMA overhead
C  predictor ON + opportunistic GPU slot execution     -> the product
D  preseeded/oracle residency                          -> ceiling incl. gap to 142 us
```

B needs one switch: env `GGML_EXPERT_EXEC_FORCE_CPU=1` guarding the
`node->src[0] = slot_tensor` swap at `ggml-backend.cpp:2057` (single line).
D: seed slot pools from the first K tokens' actual routes (seeding infra exists -
`ggml_backend_expert_cache_seed` legacy path; add slot-pool seeding beside it if
missing) so steady-state TG excludes warm-up.

Derived reads:
- C - B = GPU execution gain; B - A = prediction+DMA tax; D - C = scheduling gap.
- PCIe accounting: total bytes_ram_to_gpu + wasted_prefetch_bytes per token.
- Amdahl sanity: measure MoE share of token time (probe_host_us/probe_upload_us +
  diag from M1) and check measured speedup against 264->142 us bound.

## M5 - Staged bundle prefetch (gate/up first, down on confirm)

Only after M4 shows wasted bytes matter.

- `submit_prediction` (:2350-2356) currently prefetches gate+up+down together.
  Split: gate+up at prediction time; down enqueued from the zero-copy confirm
  point for the gate projection of the same layer (per-projection identification:
  `src[0]` name contains `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`;
  `get_tensor_layer` already extracts N from `blk.N.` names).
- Effect: wrong-prediction waste drops from 3 projections to 2; down gains the
  gate/up compute window as extra lead time.
- Risk: down arrives late more often -> `USED_IN_FLIGHT` rises; M1 counters price
  this directly. Keep both modes behind the width/env knob until measured.

## Acceptance criteria

1. `gpu_slot_executions > 0` during real TG in the M0 config.
2. Misses/late predictions fall back to CPU with no blocking wait
   (`wait_prefetch` absent from hot path; `n_used_miss` priced).
3. PP unchanged in the M0 config vs predictor-off (alternating runs).
4. Greedy output token-identical to predictor-off for a fixed prompt
   (remap permutes expert order, math is per-id independent; verify empirically).
5. Ready/in-flight/miss rates reported per bench run.
6. Actual PCIe bytes reported, including wasted prediction traffic.
7. Statistically significant TG gain of C over A (alternating predictor-off runs).
8. Gap to pre-resident oracle quantified (D vs C, and 142 us bound vs realized).

## Explicitly out of scope for 5H

- Learned predictor architecture changes (no new features, no retraining).
- Lifting `ids ne[1] == 1` gates (PP keeps the sparse batched path).
- GPU execution for whole-layer-CPU configs (ops sit in CPU splits; discovery
  stays scoring-only there).

## Risk register

- Slot-pool capacity: pool = capacity/2 per tensor shape (:273-290). Horizon-1
  lookahead needs 8 live slots per shape; `-exc 256` gives ~80+. Watch evictions
  (`n_evictions`) during the sweep; if thrash appears, raise `-exc` before tuning.
- Event object churn: `cudaEventCreateWithFlags` per prefetch (:1469). At 16
  wide x 40 layers this is fine, but reuse-from-a-small-ring is a known follow-up
  if profiling shows creation cost.
- `-ncmoe` regex pinning (`llama-bench.cpp:1374-1385`) applies to the LAST N
  blocks; confirm the pinned set matches layers the GPU split consumes, else
  eligibility checks silently decline (`non_host_weight_bypasses` will show it).
