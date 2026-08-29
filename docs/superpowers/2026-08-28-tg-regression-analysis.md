# Token Generation Throughput Regression Analysis

## Scope

Review of the Qwen3.6-35B-A3B-APEX-Compact single-token generation regression on the GTX 1080 / Ryzen 7 5700X system. The observed range is 15-18 TK/s, down from the prior 20-22 TK/s cache-off-equivalent range.

This document records evidence from the current working tree, the active preset, recent commit history, and `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`. It does not claim that a benchmark was rerun for this review.

## Conclusion

The latest 15.0 TK/s result is most strongly correlated with a correct safety change that prevents stale route IDs from reaching CUDA. It restores coherent output and prevents an illegal memory access, but currently forces the real MoE decode route away from GPU slot-pool execution.

The deployment-level 15-18 TK/s range also has an independent cause: merely enabling the expert cache changes auto-fit placement and scheduler behavior even when cache telemetry proves that no token-generation cache operation executes.

The next implementation target is not a CUDA kernel micro-optimization. It is a route-ready scheduler boundary that lets the route-ID producer execute before a cache-managed `MUL_MAT_ID` is remapped and dispatched. Cache enablement must also stop globally changing placement when no cache route is eligible.

## Evidence

### 1. Decode route-ID freshness guard

The uncommitted helper `ggml_backend_sched_can_offload_host_weight()` in `ggml/src/ggml-backend.cpp:1009-1024` computes:

```cpp
const bool route_ids_pending = has_cache && is_decode_mul_mat_id &&
    tensor->src[2]->op != GGML_OP_NONE;

return !route_ids_pending && ...;
```

Normal MoE route IDs are produced by graph operations such as top-k and therefore have `op != GGML_OP_NONE`. The helper is consulted during placement and split selection. Consequently, cache-managed decode `MUL_MAT_ID` nodes with route IDs produced in the current graph remain on CPU.

This guard is required for correctness. The optimization log records that stale or uninitialized IDs previously reached `ffn_moe_down-13`, causing an illegal CUDA memory access and slash-only output. It also records a coherent cache-enabled CLI completion at 15.0 TK/s after the guard was added. See `EXPERT_CACHE_OPTIMIZATIONS_LOG.md:1524-1530`.

The immediately preceding 2026-08-28 record reports 22.15 TK/s for `-exc 64` and 22.18 TK/s for `-exc 128 -excp 256 -pe pinned_experts_1024mb.json` with `GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL=1`. Those are not proof that removing the guard is safe; they establish a strong performance boundary across a necessary correctness change.

### 2. Cache enablement changes deployment placement

`src/llama-context.cpp:280` sets:

```cpp
cparams.op_offload = params.op_offload || (params.expert_cache_size > 0);
```

Thus a nonzero `-exc` forces op-level offload scheduling. `common/fit.cpp:745-747` separately disables partial MoE layer fitting whenever an expert cache exists:

```cpp
const bool allow_split_moe_layers = (cparams == nullptr || cparams->expert_cache_size == 0);
```

The active Compact preset contains both `exc = 128M` and a 1024 MiB pinned-expert manifest. These settings reserve VRAM and change the number and shape of full GPU-resident MoE layers, independently of whether the cache serves a decode request.

The documented deployment comparison is:

| Configuration | TG throughput | Interpretation |
|---|---:|---|
| Full-layer control | 18.26 TK/s | 11 full MoE layers on GPU |
| Dynamic cache, 128 MiB | 16.39 TK/s | Post-sync-fix result |
| Hybrid cache, 128 MiB plus 1024 MiB pinned | 16.18 TK/s | Full layers displaced by cache reservation |

Source: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md:1382-1401`.

Earlier cache-off/cache-on five-pair data averages 20.65 versus 20.44 TK/s, while all TG cache counters are zero. This means those rows do not measure cache acceleration; they measure placement and run variance.

### 3. Cache hot-path cost when it is active

In `ggml_backend_sched_compute_splits()` the current cache path performs work per cache-eligible `MUL_MAT_ID` node:

- synchronizes the input backend at `ggml-backend.cpp:2223-2225`;
- reads route IDs to host and synchronizes the IDs backend at `2247-2250`;
- counts IDs, records access frequency, resolves slots, and uploads remapped IDs;
- uses a CPU fallback that transfers activations to host, executes a one-node CPU graph, and uploads the result when a slot is unavailable;
- optionally builds heterogeneous bundle work.

The log identified the original miss path as up to 120 CPU/GPU synchronization boundaries per generated token: Gate, Up, and Down across 40 MoE layers. Removing one result-upload synchronization improved the dynamic 128 MiB row from 13.34 to 16.39 TK/s, but it did not remove the architecture-level boundary cost. See `EXPERT_CACHE_OPTIMIZATIONS_LOG.md:1351-1389`.

The optional heterogeneous full-hit executor also allocates and destroys a 16 MiB ggml context and a GPU backend buffer for each bundle, then synchronizes the GPU before copying the result. See `ggml/src/ggml-backend-moe-hetero.cpp:238-282`. It is experimental-only and should not be the first deployment optimization.

### 4. Deprioritized hypotheses

- The learned routing predictor is not present on the current `feat/expert-cache-without-prediction` branch. It is not a default TG cost.
- Prefetch is disabled by default and is not exposed through `llama-bench`; it cannot explain default benchmark rows.
- The CUDA `MUL_MAT_ID` implementation is stock in this branch. The evidence points to host scheduling and CPU/GPU boundary work, not a newly introduced CUDA kernel regression.
- Rebalancing every cache period can add latency spikes, but it is not the first explanation for the sustained 15-18 TK/s range.

## Recommended Direction

1. Preserve the freshness guard until a route-ready scheduler path exists.
2. Execute the graph prefix that produces route IDs, then remap and dispatch only dependent MoE nodes. Reuse graph views already used by heterogeneous execution; do not read IDs before their producer finishes.
3. Process Gate, Up, and Down as one FFN bundle: one route read, one hit/miss partition, and no per-projection CPU/GPU round trip.
4. Decouple nonzero cache capacity from global `op_offload` and auto-fit restrictions. Cache-off-equivalent placement must be preserved when no cache route is active.
5. Reuse heterogeneous execution contexts and device buffers only after the route-ready path is correct and measured.

## Required Measurement Matrix

Run at least five alternating fresh-process TG-only pairs using the same binary, model checksum, model placement, fixed prompt and seed, 14 threads, Q8_0 KV cache, Flash Attention, mlock, and fit target.

| Row | Purpose |
|---|---|
| `-exc 0`, no pinned manifest | Deployment placement control |
| `-exc 128M`, no pinned manifest | Isolate cache-enable placement effect |
| `-exc 128M` plus pinned manifest, hetero disabled | Isolate pinned-residency displacement |
| Same as preceding row with hetero enabled | Measure experimental heterogeneous overhead |
| Route-ready implementation | Measure safe cache execution against the same control |

Run generation separately from prompt processing. Capture `expert_cache_eligible_ops`, `expert_cache_requests`, `expert_cache_zero_copy_hits`, misses, `expert_cache_probe_sync_us`, `expert_cache_probe_host_us`, rebalances, RAM-to-GPU bytes, and the number of full GPU-resident MoE layers.

The first acceptance threshold is recovery of the cache-off placement baseline with deterministic output. A cache configuration is not an optimization until it exceeds that baseline outside measured run-to-run variance.
