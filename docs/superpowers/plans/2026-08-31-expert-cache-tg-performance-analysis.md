# Expert Cache TG Performance Analysis and Improvement Plan

**Date:** 2026-08-31

**Scope:** Qwen3.6-35B-A3B-APEX-Compact token generation on GTX 1080 with the expert-cache route-ready dispatcher.

## Problem

The supplied review is correct about `188f29386`: restoring `down_node->op = GGML_OP_NONE` after manually producing the node is a correctness fix, not a meaningful performance cost.

Its main performance diagnosis is stale. Current head `4344a5e76` has already removed the per-fallback temporary CPU context, graph, and buffer lifecycle. Incomplete bundles now execute the native CPU split-graph segment:

```cpp
ggml_graph_view(&split->graph, first_bundle_node_idx, last_bundle_node_idx + 1)
ggml_backend_graph_compute_async(split_backend, &bundle_view)
ggml_backend_synchronize(split_backend)
```

`ggml/src/ggml-backend.cpp:3174-3183`

That preserves APEX scale tensors, LoRA operations, and activation graph semantics. Do not reintroduce a manually rebuilt FFN fallback.

## Current Raw TG Performance

Fresh build and five alternating TG128 pairs on `4344a5e76`:

- Compact Qwen3.6-35B-A3B-APEX
- 14 threads, Q8_0 KV, Flash Attention, mlock
- batch 4096, ubatch 2048, `-fitt 256`
- cache-on: `-exc 128 -excp 128`
- cache-off: `-exc 0`

| Metric | Cache off | Cache on |
|---|---:|---:|
| Mean TG | **24.137 t/s** | **23.688 t/s** |
| Sample SD | 1.221 | 0.835 |
| Mean paired delta | - | **-1.70%** |
| Approx. 95% paired interval | - | **-8.03% to +4.62%** |

The sign is not established by five pairs.

More important than the mean: every cache-on run had exactly:

```text
3,483 classifications
0 full 8/8 hits
3,479 native CPU fallback bundles
4 serial 7/8 actions
0 timed expert-weight H2D bytes
mask: 0:3436 1:12 2:20 3:7 4:4 5:0 6:0 7:4 8:0
```

So this measurement contains zero full-hit sidecar benefit. It measures cache dispatch overhead against the ordinary scheduler.

The live MTP artifact reports 18.398 accepted predicted tokens/s at 82.0% draft acceptance, but it uses the MTP model and speculative verification batches. It is not comparable with Compact raw TG1 and has no cache-off control.

Fresh raw results are recorded in `EXPERT_CACHE_OPTIMIZATIONS_LOG.md:2755-2788`.

## Actual Remaining Cost

For 0-6/8 routes, the fallback is now mathematically native, but it is not scheduler-transparent. Each route-ready bundle still pays:

1. Prefix graph submission and synchronization before dispatch.
2. Route-ID host read, including producer synchronization when cross-split.
3. Route partitioning and route-census bookkeeping.
4. Three short-lived vectors: IDs, hits, misses.
5. Native bundle graph submission and synchronization.

Relevant path: `ggml/src/ggml-backend.cpp:3047-3184`.

The four 7/8 serial actions cannot explain a material whole-model regression. They are only 0.11% of classifications. Historic partial-mask results favored CPU-base for every partial mask, but that fixture predates the corrected native fallback, so do not remove 7/8 serial without remeasurement.

## Decision

Do not implement "8/8-only transparent misses" as the next source change. Most of it is already true for 0-6 hits, and the tested cache has no 8/8 opportunities.

First make the cache cheap when its resident set makes sidecar or serial admission impossible. Then decide whether this preset should enable dynamic expert cache at all.

## Gated Improvement Plan

### 1. Establish a Decisive Raw-TG Result First

Extend the current matrix from five to approximately 28 total alternating pairs, preserving the exact command settings and alternating order.

Current paired SD is 5.09%. About 28 pairs should reduce the estimated 95% paired interval to roughly +/-2%, if variance remains similar.

Collect per run:

- TG rate and paired delta.
- 0-8 route mask histogram.
- Full-hit and serial-action counts.
- Zero weight-H2D invariant.
- Automatic-fit placement and VRAM diagnostics.

**Decision gate**

- No 8/8 hits and a statistically credible negative cache delta: disable dynamic cache for the Compact raw-TG preset (`exc = 0`) after the separate MTP validation below.
- No 8/8 hits and an inconclusive delta: park performance changes until the dispatch-overhead measurement identifies a sufficiently large removable population.
- Meaningful 8/8 hits: retain sidecar and proceed to fast rejection.

Do not increase cache capacity merely to chase hits. Existing dynamic-capacity evidence shows VRAM displacement can dominate decode.

### 2. Add an Exact, Live Complete-Bundle Eligibility Query

**Files:**

- `ggml/src/ggml-backend-expert-cache.h`
- `ggml/src/ggml-backend-expert-cache.cpp`
- `ggml/src/ggml-backend.cpp`
- `tests/test-expert-cache.cpp`
- `ggml/include/ggml-backend.h`
- `tools/llama-bench/llama-bench.cpp`

Expose a query equivalent to:

```cpp
bool ggml_backend_expert_cache_has_at_least_complete_bundles(
    ggml_backend_expert_cache_t cache,
    int32_t layer,
    int32_t minimum);
```

A complete bundle means all required registered projections for one expert are `RESIDENT`:

- fused: `gate_up` and `down`;
- unfused: `gate`, `up`, and `down`.

It must poll LOADING slots through the existing event state machine before reporting. Do not use a stale rebalance-epoch snapshot.

Initial admission bound:

```text
complete bundle count < 7
-> 7/8 serial and 8/8 sidecar are impossible
-> skip route-ID read, partitioning, census access recording, and vector allocation
-> execute the native bundle view
```

This preserves current behavior exactly:

- Prefix graph evaluation remains mandatory.
- The native bundle still runs normally.
- No node operation is changed.
- A layer capable of 7/8 serial or 8/8 sidecar still uses current classification.

**TDD cases**

1. Six complete bundles: fast reject, native output equals ordinary CPU output, zero weight H2D.
2. Seven complete bundles: no fast reject; serial eligibility remains reachable.
3. Eight complete bundles: no fast reject; full-hit sidecar remains reachable.
4. A LOADING bundle whose event becomes ready: query sees the transition before deciding.
5. Multi-token `ne[1] > 1`: route-ready TG1 behavior remains bypassed.

### 3. Measure the Fast Reject, Then Simplify Allocation Only if Justified

Add decode-only counters for:

- fast-rejected route-ready bundles;
- complete-bundle-count distribution;
- time in prefix wait, route-ID read, partition/census, and native segment.

Use existing stats plumbing consistently: header struct, bench field list, value list, integer typing, and subtraction logic must move together.

After the fast-reject implementation:

- run `test-expert-cache`;
- run `test-backend-ops`;
- rerun the same 28-pair TG matrix;
- verify zero output corruption in the Compact server preset;
- update `EXPERT_CACHE.md` and the optimization log.

Only if profiling shows remaining classification allocations matter, replace TG1's three vectors with scheduler-owned or stack scratch. This is a small optimization, not the primary bet.

### 4. Re-evaluate 7/8 Serial Separately

Use the corrected native fallback and actual APEX execution path to compare:

```text
native CPU bundle vs 7/8 serial vs 7/8 concurrent
```

Do not infer this from the current whole-model matrix; only four serial actions occurred.

- Native faster with a credible margin: demote 7/8 to native fallback in production. Keep executor infrastructure only where it still has a real direct-test or development use.
- Serial faster: retain the current threshold.
- Difference inside noise: choose native fallback for lower lifecycle complexity.

### 5. Treat MTP and Speculative Verification as a Separate Workstream

Route-ready dispatch is TG1-only. MTP and speculative verification have `ne[1] > 1` and follow a different multi-token remap path.

Run the existing five-sample MTP qualitative workload cache-on versus cache-off with fixed deployment conditions. Record:

- accepted predicted tokens/s;
- draft acceptance;
- prompt throughput;
- malformed or repetitive output count;
- multi-token cache eligibility and full-residency counters.

Do not use raw TG1 results to disable cache for MTP without this control.

## Do Not Pursue Now

- GPU-side route classification or a CUDA compaction kernel. Prior route synchronization evidence was sub-1% of decode time; zero full hits make it irrelevant.
- Lowering partial-hit admission below 7. Historic latency evidence says CPU-base wins every partial mask.
- Pinned-profile or larger-cache tuning before proving that this configuration can generate full hits without worsening `--fit`.
- Removing synchronization that currently fixes cross-split GPU-to-CPU ordering.

## Immediate Next Action

The first source change worth implementing is the live `< 7 complete bundles` fast reject, but only after the longer paired matrix confirms the eligible population and separates raw TG1 from MTP behavior.
