# General Decode MoE Route-Aware Dispatch Design

**Status:** Proposed. No route-aware dispatch behavior exists in the current source.

**Goal:** Enable one general decode-time MoE execution policy for normal parallel generation, speculative verification, MTP verification, and future decode microbatches. For each routed expert operation, execute a complete resident expert bundle directly from GPU slot pools only when the current route is ready and fully resident; otherwise keep the operation on the existing CPU path and optionally fill the cache for later work.

## Problem

The current expert cache has a useful execution mechanism but no route-aware placement decision.

`llm_graph_context::build_moe_ffn()` creates router logits, `ggml_argsort_top_k()` selected-expert IDs, and gate/up/down `GGML_OP_MUL_MAT_ID` operations in one graph. The scheduler assigns backends before those IDs are computed. A cache can currently act only when a host `WEIGHTS` tensor is copied into a non-CPU split and is the source of a matching `MUL_MAT_ID`.

This has two separate consequences:

1. The existing zero-copy path is universal after an operation is cache-eligible. It gathers the requested union, allocates or finds slots, remaps all route IDs, and replaces `node->src[0]` with the slot-pool tensor. It does not impose a one-token limit.
2. Normal host-resident decode MoE operations commonly remain CPU-routed before the cache sees them. CUDA offload uses `MUL_MAT_ID.ne[2]` as its batch-size input and defaults to an offload floor of 32. Lowering that global floor forces singleton host MoE work to CUDA, which has already regressed token generation because PCIe traffic and small GPU work dominate.

The design must solve placement after the current route is known. It must not solve it by globally forcing host MoE operations to CUDA.

## Source-Grounded Baseline

### Existing slot-pool execution

`ggml_backend_sched_compute_splits()` currently enters slot-pool execution when the cache has an eligible host-weight `MUL_MAT_ID` split input and the unique requested-expert union fits the pool capacity. It uploads a remapped ID tensor, changes the node source to the pool tensor, executes the normal backend `MUL_MAT_ID`, and restores graph pointers after the split.

This already covers arbitrary decode microbatch sizes. Large prompt processing naturally bypasses the path when the distinct-expert union exceeds slot capacity.

### Existing miss behavior

A zero-copy miss claims a slot, enqueues host-to-device transfer into that
slot, and leaves the slot `LOADING` until the load event reports completion.
The current operation still uses the existing copied-tensor fallback when the
slot union cannot be made ready. There is no CPU-on-miss dispatch, no bounded
fill queue, and no mixed CPU/GPU expert merge.

### Existing lifecycle behavior

Slots have `EMPTY`, `LOADING`, and `RESIDENT` states. CUDA-backed lookup now
polls a recorded load event before publishing `RESIDENT`; CPU-backed test
transfers retain synchronous promotion. A `last_use_event` field exists but no
consumer-use event is published. Complete-bundle ownership and multistream
dispatch therefore remain unimplemented.

### Existing cache-policy behavior

The cache already implements SLRU-style probationary and protected segments plus a full-pool ghost entry and eviction cooldown:

- an empty pool admits immediately;
- a full pool requires a second sighting within the ghost window before an admission can evict;
- an evicted entry must receive eight fresh misses before it can evict again.

The implementation does not yet bound fill jobs, in-flight fill bytes, or fills per route plan.

## Scope

This design applies to all decode-time `MUL_MAT_ID` microbatches. Workload origin is telemetry, not dispatch policy:

- ordinary server generation with one or more active sequences;
- speculative target verification;
- MTP target verification and draft execution;
- future decode microbatch modes.

Prompt processing is out of scope except as a regression workload. The design does not add a model-specific CUDA kernel and does not use MTP as the only cache target.

## Terminology

- **Route:** the selected-expert ID tensor consumed by one or more `MUL_MAT_ID` operations.
- **Route union:** the sorted, deduplicated expert IDs selected for one route tensor and decode microbatch.
- **Expert bundle:** every weight tensor required to evaluate one routed expert for one MoE layer. A bundle must represent the actual graph form, including merged gate/up tensors where present.
- **Route plan:** immutable metadata for one logical graph generation, layer, route tensor, route union, expert bundle, batch shape, and source tensor identities.
- **Full hit:** every selected expert in every required projection is `RESIDENT` and safe to consume.
- **CPU fallback:** execute the unchanged CPU MoE operation for the full route plan. It is the correctness fallback for every incomplete, stale, unsupported, over-budget, or nonresident plan.
- **Fill work item:** a bounded, best-effort request to make one complete expert bundle resident after CPU fallback. It never blocks the current MoE consumer.

## Required Invariants

1. **Current-route correctness:** the dispatch decision uses IDs from the current logical graph generation, never a pointer-equivalent tensor from a previous graph or sequence.
2. **Complete-bundle GPU execution:** phase one sends an operation to the GPU only when every selected expert and every required projection is resident. Partial GPU execution is not part of the first implementation.
3. **Fail closed:** absent IDs, unsupported `MUL_MAT_ID` forms, unknown bundle identity, capacity pressure, incomplete transfer, stale route identity, or missing event capability all select CPU fallback without waiting.
4. **No reactive transfer wait:** CPU fallback must not wait for a host-to-device fill. A fill may continue only after CPU fallback has selected its immutable source data and reserved a safe destination slot.
5. **No premature residency:** `LOADING` entries are not hits. A fill becomes `RESIDENT` only after completion is observed through a backend event query or an equivalent nonblocking completion contract.
6. **No unsafe reuse:** an evicted or overwritten slot waits for both the fill event and the last GPU consumer event. CPU fallback does not need a GPU consumer event.
7. **Graph identity:** route plans carry graph generation, logical sequence identity, layer, source tensor identity, and route tensor identity. Pointer comparison alone is insufficient across graph copies, fit promotion, or multiple active sequences.
8. **No model-specific parsing:** route plans are discovered structurally from registered expert bundles and `MUL_MAT_ID` dependencies. Tensor display names are diagnostic data, not identity.
9. **No silent placement change:** route-aware dispatch is disabled by default. Every benchmark records model placement, cache capacity, dynamic MTP promotion state, and VRAM accounting.
10. **Output preservation:** for fixed placement, cache-off and every route-dispatch mode must produce identical greedy tokens.

## Proposed Architecture

### Route discovery and execution phases

The scheduler will discover supported route plans from the original graph by grouping registered expert `MUL_MAT_ID` nodes that share a route-ID source. It will not infer routes from split inputs alone; that would miss CPU-routed operations, which are the reason this work exists.

For a supported route plan, graph execution becomes two internal phases:

```text
Phase R: execute dependencies through selected-expert IDs
         -> make current route IDs available
         -> form route union and validate identity
         -> classify full hit, partial hit, miss, stale, unsupported, or over-budget

Phase E: full hit
         -> wait on already-completed residency only
         -> upload remapped IDs
         -> execute registered bundle from slot pools on GPU

         otherwise
         -> execute unchanged bundle on CPU
         -> optionally enqueue one nonblocking complete-bundle fill
```

The phase boundary is per route plan, not per model type. A normal parallel decode batch and an MTP verification batch use the same route-plan structure and decision table.

The first implementation may use graph views and scheduler-owned checkpoints to execute Phase R then resplit the remaining graph for Phase E. It must prove that all dependencies of the selected IDs are complete before the route union is inspected. It must not use the current cache path's synchronous ID read inside `compute_splits()` as a substitute for a route checkpoint.

### Dispatch table

| Route-plan condition | Current operation | Fill action | Telemetry result |
| --- | --- | --- | --- |
| Disabled | Existing scheduler placement | None | `disabled` |
| Unsupported graph form or unregistered bundle | CPU fallback | None | `unsupported` |
| Stale graph, sequence, or tensor identity | CPU fallback | None | `stale` |
| Route union exceeds slot capacity | CPU fallback | None | `union_over_capacity` |
| Any selected bundle member is `LOADING` or missing | CPU fallback | Attempt one bounded fill only for a valid complete bundle | `cpu_miss` or `cpu_loading` |
| Every selected bundle member is `RESIDENT` | GPU slot-pool execution with remapped IDs | None | `gpu_full_hit` |
| Fill budget, event capability, or slot reservation unavailable | CPU fallback | None | `fill_rejected` |

There is deliberately no phase-one `gpu_partial_hit` row. Mixed execution requires partitioned route IDs, separate CPU and GPU results, exact output merge semantics, and per-projection synchronization. It is deferred until full-hit dispatch demonstrates end-to-end value.

### Bounded fill queue

The fill queue is backend-owned and disabled unless the backend exposes asynchronous copies plus event creation, recording, waiting, and nonblocking completion query.

The initial queue has these fixed policy constraints:

- at most one complete bundle is in flight per backend;
- the queue reserves every projection slot before issuing its first copy;
- the entire bundle must fit the remaining cache capacity and staging budget before any copy is submitted;
- duplicate route plans attach to an existing `LOADING` reservation rather than submitting another fill;
- a rejected fill leaves every resident mapping unchanged;
- no queue worker synchronizes the CPU or decode compute stream;
- a queue entry is discarded on graph, sequence, source-identity, or route-plan generation mismatch.

The queue is not a generic transfer queue. It is a best-effort cache-population mechanism behind CPU fallback.

### Slot state machine

```text
EMPTY
  -> reserve complete bundle
  -> LOADING

LOADING
  -> duplicate request attaches; not a hit
  -> completion query succeeds for every projection
  -> RESIDENT
  -> cancellation or failed copy releases every reserved slot

RESIDENT
  -> full route hit may consume
  -> record consumer event after final GPU bundle use
  -> eviction waits for fill and consumer events
  -> EMPTY or LOADING after safe eviction/reuse
```

A slot is never exposed as `RESIDENT` merely because a transfer has been enqueued. For the initial one-stream CUDA implementation, a nonblocking completion query may observe a completed event before publishing the mapping. Backends without that contract retain CPU fallback and no background fills.

## Identity Contract

The route plan must identify the exact operation it authorizes:

```text
graph_generation
logical_sequence_id
layer
route_ids_tensor
registered bundle identity
original host source tensor for every projection
batch shape
sorted unique expert IDs
```

Slot maps remain keyed by canonical source tensor plus expert ID. Graph views and backend copies must resolve to that canonical source before lookup, remap, fill, or eviction. Same-shape tensors from separate layers must remain isolated.

## Telemetry Contract

The existing cache statistics remain cumulative. Route-aware dispatch adds counts and bytes sufficient to explain a result without requiring a trace parser:

- original-graph and split-graph routed `MUL_MAT_ID` counts;
- CPU-routed, GPU-host-weight, and GPU-device-weight route-plan counts;
- route-plan batch histogram and route-union histogram;
- full-hit GPU dispatches, CPU misses, CPU loading fallbacks, stale plans, unsupported forms, and over-capacity plans;
- fill submissions, coalesced fills, fill bytes, completion latency, useful-after-fill hits, rejected fills, and cancellation reasons;
- residency publication waits, consumer-use waits, and evictions blocked by live consumers;
- per-workload labels supplied by the caller: normal decode, speculative verification, MTP target, and MTP draft.

A developer-only sampled trace may include per-layer identity and timing. It must not copy route IDs or synchronize a backend solely to produce telemetry.

## Validation Matrix

Every retained implementation is tested in these graph forms:

| Workload | Batch condition | Required result |
| --- | --- | --- |
| Synthetic scheduler fixture | cold, warm all-hit, all-miss, mixed IDs | byte-identical output and exact route state counters |
| Normal generation | one and multiple active sequences | CPU fallback or GPU full-hit is selected from current route identity |
| Speculative verification | accepted and rejected draft batches | target output parity and correct batch classification |
| MTP | dynamic promotion on and off where supported | target/draft route plans remain isolated and placement is recorded |
| Prompt processing | large union | zero-copy bypass without a PP regression |

Every behavior change requires focused unit tests, deterministic greedy server output at fixed placement, and five alternating fresh-process benchmark pairs for a throughput claim. Report raw rows, median, mean, standard deviation, TG and PP separately, cache telemetry, transfer bytes, placement, and exact binary/model/options.

## Non-Goals

- Global `GGML_OP_OFFLOAD_MIN_BATCH=1` or unconditional host-MoE CUDA placement.
- MTP-only dispatch logic.
- A model-specific CUDA `MUL_MAT_ID` kernel.
- Mixed CPU/GPU expert execution in the first implementation.
- A learned router predictor, broad speculative prefetch, full-expert residency, deadline admission, or auto-tuning.
- A device slot-map or GPU compaction reimplementation.
- Using prompt-processing measurements as token-generation evidence.

## Decision Gates

1. If the route census finds no useful GPU-eligible decode route plans for a workload, stop cache-performance work for that workload.
2. If a complete warm bundle cannot beat CPU fallback beyond paired-run variance, do not implement fills or prefetch for that workload.
3. If route plans are not ready before the next consumer deadline, do not implement carry-forward fills.
4. If completion and consumer ownership cannot be proven on a backend, retain CPU fallback and disable its fill queue.
5. Mixed execution is considered only after full-hit GPU dispatch has a reproducible end-to-end win and partial misses dominate the remaining cost.

## Related Documents

- `EXPERT_CACHE.md` describes current implemented behavior and links here for proposed route-aware dispatch.
- `docs/superpowers/specs/2026-08-20-expert-cache-performance-design.md` is historical baseline evidence and remains valid for its stated Compact/GTX 1080 measurements.
- `docs/superpowers/plans/2026-08-26-expert-cache-scheduler-prefetch.md` documents prior scheduler-prefetch work. It is not evidence that general route-aware dispatch exists.
- `docs/superpowers/plans/2026-08-26-general-decode-moe-dispatch.md` is the implementation plan for this design.
