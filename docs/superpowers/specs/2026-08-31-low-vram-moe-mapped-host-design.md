# Low-VRAM MoE Mapped-Host Execution Design

## Status

Approved investigation design. This document authorizes research and an isolated benchmark only. It does not authorize scheduler, cache-policy, or production fallback changes.

## Objective

Determine on the target GTX 1080 whether a GPU can execute Qwen3.6 APEX MoE expert bundles by directly reading persistent mapped host memory, without timed expert-weight H2D copies and without degrading batched prompt processing.

The result must choose one of these outcomes:

- Strong positive: proceed to a production mapped-host path.
- Conditional positive: investigate a simple TG versus PP policy only if the evidence supports it.
- Negative: retain native CPU misses plus the existing full-hit VRAM sidecar and stop this execution-path line.

## Verified Current State

- Host-resident MoE `MUL_MAT_ID` weights are assigned to the CPU split by `ggml_backend_sched_backend_id_from_cur()` in `ggml/src/ggml-backend.cpp`.
- The existing cache is a VRAM slot-pool sidecar. It fills device slots from host tensors and preserves the native CPU bundle view for all production misses.
- TG1 route-ready dispatch is intentionally limited to complete 8/8 bundles. Multi-token prompt processing does not enter this dispatcher.
- Native fallback is a view of the existing CPU split graph, preserving activation, LoRA, and APEX scale operations.
- The APEX graph applies per-expert scales after `MUL_MAT_ID` in `build_lora_mm_id()` in `src/llama-graph.cpp`. The current route-ready sidecar has no scale nodes.
- The current APEX Compact measurement had no useful 8/8 sidecar work. The result must not be used to relax the fallback or admission contract.
- The current `tests/test-moe-oracle-bench.cpp` has exact Qwen geometry and quantization types, but initializes synthetic quantized weights and does not test mapped-host GPU reads or PP128/PP512.

## Reference Architecture Findings

### moe-l2

`moe-l2` is not evidence that direct mapped-host GPU reads are a winning path. Its design record reports that direct UVA dereference was slower than `cudaMemcpy` and that cuBLAS on UVA pointers was much slower. Its shipped path pins host experts, copies selected experts to VRAM, and applies a GPU LRU. That is a useful reference for registration lifetime and cache mechanics, but it violates this investigation's timed-H2D invariant.

Sources:

- https://github.com/yalun753/moe-l2
- https://raw.githubusercontent.com/yalun753/moe-l2/main/references/en/design-decisions_EN.md
- https://raw.githubusercontent.com/yalun753/moe-l2/main/references/llama.cpp-gpu-lru-cache/ggml-cuda.cu

### miltos22 WackMall

The WackMall tier path places hot experts on GPU and computes cold experts on CPU. It adds hot-store lookup and cold operators rather than mapped-host GPU computation. Its merged-request reports identify severe PP regressions; the research fork limits tiering to small token counts and leaves larger prompt graphs on the stock path. Reuse its bundle metadata and admission analysis only. Do not reuse its split hot/cold execution model.

Sources:

- https://github.com/ggml-org/llama.cpp/pull/26563
- https://github.com/ggml-org/llama.cpp/pull/26824
- https://raw.githubusercontent.com/miltos22/llama-wackMall/master/ARCHITECTURE.md

## Architecture Boundary

The first experiment is a falsification oracle, not a scheduler integration.

| Mode | Expert backing | Compute backend | Timed expert-weight H2D |
| --- | --- | --- | --- |
| CPU control | Original host tensor | CPU | 0 |
| VRAM control | Persistent device copy | CUDA | 0 |
| Candidate | Registered GGUF pages or persistent mapped host staging | CUDA direct mapped-host read | 0 |

The candidate must use CUDA mapped host addresses. A copy-on-route expert cache is expressly excluded from this experiment.

No change is permitted to:

- scheduler assignment;
- cache admission or the 8/8 threshold;
- native fallback semantics;
- sidecar behavior;
- `--fit` behavior;
- ngram-mod or MTP execution.

## Isolated Benchmark

Add `tests/test-moe-mapped-host-bench.cpp` as a benchmark target beside the current MoE oracle targets.

### Inputs and graph

- Load an actual Qwen3.6 APEX Compact GGUF supplied through a required model argument.
- Select host-resident Gate, Up, and Down expert tensors from one MoE layer, preserving actual types, dimensions, stride, fused or unfused layout, and APEX scale values.
- Build only the routed FFN: Gate and Up projections, production-equivalent scale lookup and application, SwiGLU, Down projection, route weighting, and reduction.
- Omit router, normalization, residual, attention, and the scheduler.
- Use deterministic activations, route IDs, and route weights.
- Cover 1, 2, 4, and 8 selected experts, with mixed and repeated expert selections.

### Shapes and iteration counts

Run each storage mode at TG1, TG4, TG8, PP128, and PP512 with 100 warmups and 1,000 timed iterations.

### Storage lifetime

The fixture owns every registered range, mapped pointer, staging allocation, and CUDA event.

1. Attempt direct registration of a page-aligned GGUF-backed range with `cudaHostRegister` using the mapped flag, then obtain its device-visible address.
2. Validate that the CUDA device can map host memory and that the registration is accepted on the Windows mmap allocation.
3. If direct registration is unsupported or unsafe, make a single persistent `cudaHostAllocMapped` staging copy during setup. Report the startup copy time and RAM cost separately.
4. Register or allocate before warmup. Release only after all associated stream work completes.
5. Never allocate, register, unregister, copy weights, grow vectors, construct graphs, or allocate backend buffers in the timed loop.

### Measurement contract

Report for every shape and storage mode:

- sample count, P50, P95, mean, and standard deviation wall latency;
- CUDA event elapsed time for the FFN graph;
- CPU wall time for the CPU control;
- selected and unique expert counts;
- registration and setup time;
- directly registered bytes and staging bytes;
- timed expert-weight H2D and D2H bytes;
- explicit synchronization count and backend-wide synchronization count.

Use CUDA events for device time and one completion synchronization per latency sample. Do not use a per-layer backend-wide synchronization.

### Correctness contract

Before timing every shape, compare:

- CPU versus VRAM;
- CPU versus mapped host;
- VRAM versus mapped host.

Use quantization-appropriate numeric tolerances. The test must exercise nontrivial APEX scale values, repeated IDs, mixed IDs, and all selected-expert counts so a missing scale path or a route-indexing error fails visibly.

## Decision Gate

### Strong positive

Mapped-host top-8 TG1 is faster than the CPU control at both P50 and P95, and timed expert-weight H2D bytes are exactly zero.

### Conditional positive

Mapped-host TG1 is within 5 percent of CPU at both P50 and P95, timed expert-weight H2D bytes are exactly zero, and at least one of TG4, TG8, PP128, or PP512 is at least 10 percent faster than CPU.

### Negative

Neither condition holds. Record the result, retain native CPU misses plus the existing full-hit VRAM sidecar, and stop the mapped-host production path. Do not replace it with a copy-on-route cache under this epic.

## Post-Gate Production Sequence

Only a strong or conditional positive result permits:

1. A model-lifetime host-registration facility with page-aligned range tracking, ref-counting, device compatibility, and stream-safe release.
2. Storage-aware CUDA `MUL_MAT_ID` addressing that preserves ordinary route grouping and batch semantics.
3. A two-tier expert store: atomic VRAM Gate/Up/Down bundle promotion over mapped-host GPU backing.
4. Telemetry for storage-tier execution, registered bytes, promotions, timed H2D bytes, and synchronization.
5. Deterministic TG, PP, fit, manual-placement, and ngram-mod validation.

The dynamic MTP null-buffer assertion is an independent defect. No speculative-compatibility claim is allowed until it has an evidence-backed root cause and fix.

## Non-Negotiable Invariants

- Timed mapped-host expert-weight H2D bytes equal zero.
- No per-token allocation, registration, graph construction, or buffer allocation.
- No per-expert graph construction or synchronization.
- No backend-wide synchronize per MoE layer.
- PP remains on the stock batched path until mapped-host `MUL_MAT_ID` has demonstrated preserved batch semantics.
- The current 8/8 admission and native fallback contracts remain unchanged during this investigation.

## Documentation and Run Records

Before benchmark implementation, create:

- `docs/superpowers/analysis/moe-low-vram-execution-options.md`
- `docs/superpowers/analysis/miltos-expert-tier-pp-regression.md`

After each benchmark attempt, update `EXPERT_CACHE.md` and `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` with the exact command, binary revision, environment, raw output paths, result table, and gate decision.
