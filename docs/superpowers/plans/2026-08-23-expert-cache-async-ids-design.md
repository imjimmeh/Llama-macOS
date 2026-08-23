# Async Router-ID Consumption - Design (Review Item #12)

## Problem

The zero-copy/multi-token expert-cache paths resolve which experts a layer uses by pulling the router-id tensor to host and doing host-side set analysis. The current code in `ggml_backend_sched_compute_splits` does, for every eligible expert input in every split (`ggml/src/ggml-backend.cpp`):

```cpp
ggml_backend_synchronize(input_backend);                      // :1861
...
ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(),
                              0, ggml_nbytes(ids_tensor));    // :1879
ggml_backend_synchronize(ids_backend);                        // :1880
```

Three cost drivers, in impact order on a GTX 1080:

1. `ggml_backend_synchronize(ids_backend)` is a **device-wide drain**. CUDA's id-scope `tensor_get` runs `cudaMemcpyAsync(D2H, cudaStreamPerThread)` then `cudaStreamSynchronize` (`ggml-cuda.cu:790-796`); the `synchronize` wrapper drains everything queued on the device, not just the ids-producing work. During single-token decode this serializes the whole pipeline for a 32-byte read.
2. The D2H copy **depends on the fused topk/softmax node** for that layer, which is still queued in-stream when we reach the input-copy loop for the next layer. We wait for it cold rather than overlapping it with independent work.
3. The sync block runs **once per expert matrix** (gate, up, down) - three drains per MoE layer - even when the ids tensor is unchanged (`prev_ids_tensor` caches the read but not the sync).

Goal: consume the router output without a per-matrix full-device sync, so single-token decode latency stops being dominated by the ids round-trip.

## Constraints from the reviewed code

- ids tensor for `GGML_OP_MUL_MAT_ID` is I32, 2-D: `ne[0]` = top-k width (e.g. 8), `ne[1]` = tokens in batch (`ggml.c:3339-3340`). For single-token decode `ne[1] == 1`, so `ne[0]*ne[1]*4` bytes.
- The ids-producing fusion (`topk_moe_cuda`, `ggml-cuda.cu:2792-3358`) runs on the CUDA compute stream; the ids tensor is GPU-resident for offloaded MoE. The zero-copy consumer (`ggml-backend.cpp:1964-2118`) is host code needing the expert set before it can rewrite `remapped_ids` / drive `node->src[2]`.
- The host MUST know the used set before it can issue the zero-copy transformation. So we cannot fully remove the host round-trip; we can only make it narrow (wait for one kernel, not the device) and asynchronous where possible.
- Cache-off equivalence is the acceptance bar (existing tests + the Task 4 matrix): the route set must be bit-identical to reading raw ids.

## Options

### A. Fine-grained event wait (no kernel change) - cheap, partial win

Instead of draining the device, record a CUDA event on the compute stream right after the ids-producing node, then `cudaStreamWaitEvent`/`cudaEventSynchronize` on just that event:

- Replace the buffer-interface get + device-sync with: record `event` on the ids backend's compute stream (we already have backend events via `sched->events`), enqueue the D2H `cudaMemcpyAsync` on a dedicated copy stream with `cudaStreamWaitEvent(copyStream, event)`, then `cudaEventSynchronize(event_on_copy)`.
- Same kernel, same result, but the wait covers only the ids-producing chain, not unrelated queued work.
- Win is real but bounded: for a single-token decode with an otherwise idle pipeline, device-drain vs event-wait are nearly equal because the drain isn't waiting on anything else. Main benefit is multi-stream / batched overlap.
- Low risk, small diff.

### B. GPU compaction to pinned host memory + completion flag (recommended)

Keep the ids-producer in-stream and let it feed a tiny compaction kernel that writes the *used-expert set* directly into pinned/zero-copy host memory, so host reads a completed result instead of issuing a blocking D2H of raw ids.

New kernel `ggml_cuda_op_expert_ids_compact` (new `ggml-cuda/mmid-compact.cu`, guarded by `GGML_USE_CUDA`), launched on the compute stream after the topk/softmax node:

Inputs:
- `const int32_t * ids` - the MUL_MAT_ID ids tensor (device)
- `ne0`, `ne1` - ids dims
- `n_expert` - max expert id (from the weight tensor `ne[2]`)

Outputs, in a pinned host-mapped buffer allocated once per backend (extend `ggml_backend_expert_cache` or a small sched-side struct):
- `uint32_t * used_bitset` - dense bitset over `n_expert` bits (host-visible after completion)
- `int32_t * used_experts` / `uint32_t * n_used` - compacted deduped list + count
- `volatile uint32_t * done` - completion flag, `0` at issue, set `1` by the kernel on exit (via `__threadfence()` + store)

Host protocol:
1. Before backing up the first eligible input, record the current `done = 0` (per buffer slot).
2. Launch compaction on the compute stream (a graph node or a `ggml_backend_cuda_graph_compute` scheduled op). Because it reads only `ids` and writes host-mapped memory, it needs no dependency beyond stream order after the ids producer.
3. Host spin-waits on `done` (bounded retry, then fall back to `cudaEventSynchronize`), or better: `cudaStreamWaitEvent` on a copy stream + event, then the flag is already visibly ordered.

Why this wins on GTX 1080:
- No `cudaMemcpyAsync` D2H round-trip of potentially several expert-matrices' ids; one small host-mapped buffer is written in situ by the kernel.
- Dedup + bitset happen on GPU, exactly the union the host currently computes in the `for (i1...i0...)` loops (`ggml-backend.cpp:1884-1890, 1904-1911`). Host work shrinks to a scan of the compacted list.
- The wait is on one kernel's completion, not the device.

Zero-copy host memory is genuinely pinned (`cudaHostAlloc` + `cudaHostGetDevicePointer`, or managed memory) - which dovetails with Task 5's push to stop using pageable staging.

Risk: a new GPU kernel must be correct across GPU families (it is a trivial gather/bitmask - safe); the `done` flag needs a `__threadfence()` before the store so host reads see the bitset; host spin needs a bounded fallback to avoid burning CPU on prefill where the info is needed immediately anyway.

### C. Async copy of top-K into pinned host, consume later (no kernel)

Use `cudaMemcpyAsync(D2H, pinned)` enqueued right after the ids producer on a side stream, and let the host issue it without synchronizing at read time - only synchronize once the result is actually needed (just before the zero-copy rewrite for that layer). This is essentially Option A + deliberately late sync; similar benefit, no kernel. Weaker than B because host still does raw-id parsing and waits at the same logical point; it only wins by deferring the wait to coalesce multiple layers into one sync.

## Recommendation

Adopt **B** as the target architecture, but sequence it:

1. Land **A** first (fine-grained event wait + cache the sync via `prev_ids_tensor`) - it removes the "three drains per layer" pathology with a small diff and is independently verifiable against the Task 4 equivalence matrix.
2. Then implement **B**, reusing A's infrastructure: a pinned host-mapped result buffer per backend, compaction kernel, and the `done`-flag wait. Keep A's code path as the non-CUDA fallback (CPU backend never had the drain).
3. Keep host-side `prev_ids_tensor` dedup so unchanged ids skip both the read and the compaction relaunch.

## Integration points

- `ggml/src/ggml-backend.cpp:1853-1893` - replace the sync block; route through new helper `expert_route_begin(split_backend, ids_tensor, n_expert)`.
- `ggml/src/ggml-backend-expert-cache.h` - new pinned result-buffer struct + lifecycle (alloc in `new`, restore in `free`), plus the `done` flag.
- New `ggml/src/ggml-cuda/mmid-compact.cu/.cuh` + a scheduler node type or backend hook to run it in-stream (mirror how `ggml_cuda_op_topk_moe` is scheduled; needs ids dependency ordering).
- `tests/test-expert-cache.cpp` / `test-expert-cache-shared-ids.cpp` - assert the compacted used-set equals the host-computed set from raw ids for cold/hot/partial/in-flight cases; the equivalence matrix (Task 4) is the correctness gate.

## PoC / bench gate

Do not proceed past Option A on the GTX 1080 until measured: compare `-p 32 -n 16` and single-token `-n 1` `llama-bench` PP and TG with/without, watching `record_probe_sync` / `record_probe_host` counters. Target: TG improves and `probe_sync` drops; if TG is flat, the drain was not the bottleneck and B's kernel cost may not pay off - stop after A. This is the same discipline the Phase 5G thread applied (`tests/test-moe-latency-oracle.cpp`).

## Non-goals

- Multi-token zero-copy (blocked on Tasks 1-4 anyway).
- Removing the host round-trip entirely (impossible: host must route).