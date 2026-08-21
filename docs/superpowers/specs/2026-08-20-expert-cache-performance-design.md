# Expert Cache Performance Design

**Goal:** Restore a measured throughput advantage over the unmodified baseline for Qwen3.6-35B-A3B-APEX-Compact on the GTX 1080 while retaining expert caching only where it pays for its runtime cost.

## Evidence

A controlled `llama-bench` run used the Compact model, GTX 1080, 14 threads, batch 4096, ubatch 2048, Q8 KV cache, Flash Attention, mlock, fit target 256 MiB, `p=512`, `n=128`, and 10 repetitions.

| Binary / configuration | PP tok/s | TG tok/s |
| --- | ---: | ---: |
| Rebuilt baseline `849798132`, no cache/split | 467.40 +/- 9.36 | 25.79 +/- 0.86 |
| Tuned `f9d483a22`, `-exc 0 --ffn-split 0` | 473.85 +/- 10.71 | 27.68 +/- 0.30 |
| Tuned, `--ffn-split 0.35` only | 466.00 +/- 8.92 | 26.89 +/- 0.65 |
| Tuned, `-exc 256` only | 453.78 +/- 9.53 | 25.63 +/- 0.84 |
| Tuned, `-exc 256 -excp 64 --ffn-split 0.35` | 453.68 +/- 9.28 | 26.12 +/- 0.59 |

The branch without expert cache or FFN splitting is faster than the rebuilt baseline. The tuned operating point loses performance because FFN partitioning and cache bookkeeping cost more than the expert transfers avoided by this synthetic workload.

`src/llama-graph.cpp` builds independent CPU and GPU FFN branches and combines them for a nonzero `ffn_split`. `ggml/src/ggml-backend.cpp` downloads and synchronizes router IDs before cache lookup, then records accesses, maintains slot residency, and uploads remapped IDs. The expert cache must therefore demonstrate enough zero-copy hits and avoided transfer volume to amortize those operations.

## Scope

1. Keep `ffn-split = 0` for the affected Compact/GTX 1080 profile unless a controlled capacity sweep proves a positive end-to-end tradeoff.
2. Make each `llama-bench` result expose the cumulative expert-cache statistics used to explain its throughput result.
3. Run real request-replay measurements separately from synthetic `llama-bench` runs. Record cold, warm, and steady-state generation throughput, deterministic output hash, and cache telemetry.
4. Consider asynchronous router-ID overlap only after trace data shows that router synchronization materially limits decode throughput.

## Data contract

When `expert_cache_size > 0`, a llama-bench result will include these integer fields from `ggml_backend_expert_cache_stats`:

- `expert_cache_requests`
- `expert_cache_hits`
- `expert_cache_zero_copy_hits`
- `expert_cache_d2d_fallback_hits`
- `expert_cache_speculative_prefetches`
- `expert_cache_misses`
- `expert_cache_evictions`
- `expert_cache_rebalances`
- `expert_cache_jit_swaps`
- `expert_cache_bytes_ram_to_gpu`
- `expert_cache_bytes_avoided`
- `expert_cache_cpu_id_remaps`
- `expert_cache_gpu_id_resolutions`
- `expert_cache_staging_memcpy_bytes`
- `expert_cache_direct_pinned_dma_bytes`
- `expert_cache_map_updates`
- `expert_cache_map_update_bytes`
- `expert_cache_dma_ns`
- `expert_cache_dma_wait_ns`

For `expert_cache_size == 0`, the same fields are emitted as zero. This keeps CSV, JSON, JSONL, Markdown, and SQL result schemas stable across an ablation matrix.

The benchmark will snapshot the aggregate scheduler statistics after warmup and again after its repetitions, then emit the difference. Warmup traffic must not be reported as measured throughput telemetry.

## Validation gates

- Unit tests continue to cover slot allocation, remapped IDs, zero-copy hits, misses, SLRU admission, bundles, pinned buffers, and prediction through `tests/test-expert-cache.cpp`.
- A benchmark smoke run confirms every output format contains a complete and internally consistent telemetry schema.
- Real server replay uses `temperature = 0`, `top-k = 1`, fixed seed, fixed prompt sequence, `parallel = 1`, fresh process for each configuration, and output hashes to prove cache state does not alter tokens.
- An asynchronous router-ID implementation is allowed only if trace results show a material, repeatable decode cost and the new path preserves deterministic tokens for cold and warm cache hit/miss sequences.
- A cache configuration is accepted only if it beats `-exc 0 --ffn-split 0` beyond measured variation for the intended real workload. It is not sufficient to beat an older cache implementation.

## Non-goals

- Do not add a model-specific CUDA kernel.
- Do not enable or tune FFN splitting for the Compact/GTX 1080 profile without measured evidence.
- Do not implement speculative cache prefetch, router transfer overlap, or automatic cache enablement before the telemetry and replay benchmarks identify a compensating bottleneck.
