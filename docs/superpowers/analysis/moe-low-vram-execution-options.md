# Low-VRAM MoE Execution Options

Status: Phase-zero evidence record (Task 1 of the mapped-host oracle epic).
Scope: Analysis only. This document does not change executable code, CMake,
cache behavior, scheduler behavior, fit behavior, MTP, or tests.

## 1. Purpose

The approved design (`docs/superpowers/specs/2026-08-31-low-vram-moe-mapped-host-design.md`)
requires an execution-options comparison for MoE expert bundles on the GTX 1080
target before any benchmark source is added. For each option this document
records weight location, compute backend, route handling, PP/TG1/TG>1 behavior,
H2D traffic, synchronization, VRAM and host RAM cost, PCIe dependency,
quantization and APEX-scale implications, complexity, and expected Pascal
suitability. Every local behavior cites a repository path and symbol; every
external claim cites a URL. Neither this document nor its companion
(`docs/superpowers/analysis/miltos-expert-tier-pp-regression.md`) proposes a
copy-on-route cache or a CPU/GPU partial executor as this experiment.

## 2. Verified current state (local evidence)

- Host-resident MoE `MUL_MAT_ID` weights are assigned to the CPU split.
  `ggml_backend_sched_backend_id_from_cur()` (`ggml/src/ggml-backend.cpp:1059`)
  walks the WEIGHTS-buffer sources of a tensor (`:1110-1112`); when a source's
  own backend is the last (CPU) backend and the buffer is host memory
  (`ggml_backend_buffer_is_host`, `:1112`), it tries an earlier backend only if
  the tensor is not a registered host expert weight
  (`ggml_backend_sched_is_registered_host_expert_weight()`,
  `ggml/src/ggml-backend.cpp:1022` declaration / `:1037` definition) and
  `ggml_backend_supports_op` plus `ggml_backend_offload_op` accept (`:1117-1121`);
  otherwise it returns the source backend, i.e. CPU (`:1125`).
  `EXPERT_CACHE_OPTIMIZATIONS_LOG.md:3052-3058` records the observed behavior on
  this machine: with MoE weights on host, `MUL_MAT_ID` lands on the CPU split and
  the expert cache never engages (`eligible_ops=0`, `requests=0`,
  `cpu_backend_bypasses=8064` under `-fitt 256` on the GTX 1080).
- The expert cache is created only for non-CPU backends:
  `ggml_backend_sched_set_expert_cache()` (`ggml/src/ggml-backend.cpp:3379-3395`)
  skips backends whose device type is CPU (`:3388`). CPU splits record a bypass
  instead (`ggml_backend_expert_cache_record_cpu_backend_bypass()`,
  `ggml/src/ggml-backend.cpp:2641`).
- The current cache is a VRAM slot-pool sidecar. Route-ready dispatch structures
  are built by `ggml_backend_sched_build_route_ready_dispatches()`
  (`ggml/src/ggml-backend.cpp:1336-1474`) with
  `ggml_moe_route_ready_sidecar_new()` (`:1451`) and the
  `ggml_backend_sched_route_ready_dispatch` struct (`:897-907`). Full 8/8
  bundles execute via `ggml_moe_route_ready_sidecar_execute_full_hit()`
  (`:3030`) when `n_tokens == 1 && n_hits == top_k && n_misses == 0` (`:3029`).
- TG1 admission is selective (`EXPERT_CACHE.md:24-29`): 0-6 hits -> CPU-base;
  7 hits -> native CPU bundle; 8 hits -> full GPU route-ready sidecar.
  `GGML_MOE_PARTIAL_MIN_GPU_HITS` is 7 (`ggml/src/ggml-backend.cpp:2268`); a
  resident-bundle count below 7 fast-rejects to the native CPU bundle view
  (`ggml_graph_view` + `ggml_backend_graph_compute_async` +
  `ggml_backend_synchronize`, `:2953-2969`), skipping the route-ID read and
  partition (`EXPERT_CACHE.md:196`).
- The APEX graph applies per-expert scales after `MUL_MAT_ID` in
  `build_lora_mm_id()` (`src/llama-graph.cpp:1520-1556`): `ggml_mul_mat_id` at
  `:1525`, then reshape/repeat/get_rows/mul of the scale vector at `:1527-1534`.
  The current route-ready sidecar has no scale nodes (design spec line 23);
  native fallback preserves activation, LoRA, and APEX scale operations (design
  spec line 22).
- The existing oracle microbenchmark `tests/test-moe-oracle-bench.cpp` has exact
  Qwen geometry and quantization types (`qwen_moe_spec`, `:94-102`, with
  `GGML_TYPE_Q4_K` Gate/Up and `GGML_TYPE_Q6_K` Down) and CPU/GPU variants
  (`oracle_variant`, `:104`), but initializes synthetic quantized weights and
  does not test mapped-host GPU reads or PP128/PP512 (design spec line 25). Its
  defaults are 100 warmups and 1,000 timed iterations (`main()`, `:348-351`).

## 3. Execution-option comparison

Summary table (detailed per-option records follow):

| Option | Expert weight location | Compute backend | Timed expert H2D | VRAM for experts | Production status |
| --- | --- | --- | --- | --- | --- |
| Native CPU MoE | host tensors (mmap-backed GGUF) | CPU | 0 | 0 | active baseline |
| Full-bundle VRAM sidecar | host + device slot pool | CUDA (8/8), CPU else | 0 (timed) | full 8/8 bundles per layer | active |
| Retired partial CPU/GPU executor | host + resident slots | CUDA + CPU overlapped | 0 (timed) | exact-K graphs + slots | retired (tests only) |
| WackMall hot/cold tiering | hot in VRAM slots, cold on host/mmap | CUDA (hot), CPU (cold) | hot promotions | S slots x 3 matrices | external, unmerged |
| moe-l2 selected-expert H2D LRU | host-pinned + VRAM LRU | CUDA | per-step misses | LRU slot pool | external reference |
| Direct mapped-host GPU reads (candidate) | registered GGUF pages / mapped staging | CUDA mapped read | 0 (timed) | 0 | candidate, unbuilt |

### 3.1 Native CPU MoE

- Weight location: original host tensors in WEIGHTS buffers; host residency is
  the branch condition at `ggml/src/ggml-backend.cpp:1112`
  (`ggml_backend_buffer_is_host`).
- Compute backend: CPU split, assigned by
  `ggml_backend_sched_backend_id_from_cur()` (`ggml/src/ggml-backend.cpp:1125`).
- Route handling: native `MUL_MAT_ID` consumes route IDs in-graph; no route-ID
  D2H read, no partition, no dispatch (`EXPERT_CACHE.md:196` describes the
  fast-reject shortcut that skips route-ID transfer, access recording, hit/miss
  vector resizing, and route partitioning).
- PP/TG1/TG>1: all shapes run the stock CPU split graph. Multi-token prompt
  batches use dynamic backend buffer allocation
  (`ggml_backend_alloc_ctx_tensors`, `EXPERT_CACHE.md:35`, tested to 32k+
  tokens with 0 pool exhaustion).
- H2D traffic: zero (no GPU expert compute).
- Synchronization: standard per-split compute + synchronize; no per-layer extra
  sync.
- VRAM cost: zero for experts; only non-expert layers per `-ngl` placement.
- Host RAM cost: full model pages resident/working set.
- PCIe dependency: none for expert compute.
- Quantization/APEX implications: all types and APEX scales, plus LoRA, handled
  by `build_lora_mm_id()` (`src/llama-graph.cpp:1520-1556`).
- Complexity: none (baseline).
- Expected Pascal suitability: baseline reference. On this machine the 28-pair
  fast-reject TG128 matrix measured control 24.732 tok/s vs cache 24.662 tok/s
  (`EXPERT_CACHE.md:199`), i.e. the CPU path is the reference the gate compares
  against.

### 3.2 Current full-bundle VRAM route-ready sidecar

- Weight location: host tensors remain; complete Gate/Up/Down bundles live in
  persistent device slot pools in `RESIDENT` state (`EXPERT_CACHE.md:194`).
- Compute backend: CUDA for 8/8 bundles via
  `ggml_moe_route_ready_sidecar_execute_full_hit()`
  (`ggml/src/ggml-backend.cpp:3030`); CPU split for everything else.
- Route handling: TG1-only; classified bundles read route IDs from device
  (`ggml_backend_tensor_get`, `:2979-2982`, with producer-split synchronize at
  `:2976-2983`) and partition via
  `ggml_backend_expert_cache_partition_bundle_routes()` (`:2998-3000`); a
  resident-bundle count below `GGML_MOE_PARTIAL_MIN_GPU_HITS` (7, `:2268`)
  fast-rejects to the native CPU bundle view (`:2953-2969`).
- PP/TG1/TG>1: TG1 8/8 -> sidecar; TG1 7/8 -> native CPU bundle; TG1 0-6 ->
  CPU-base (`EXPERT_CACHE.md:24-29`); multi-token prompt processing does not
  enter this dispatcher (TG1 gate `ne[1] == 1`, `EXPERT_CACHE.md:47`, `:196`).
- H2D traffic: zero timed decode expert-weight upload in the active route-ready
  paths (`EXPERT_CACHE.md:16`); the background promotion pipeline is
  non-blocking and outside timed decode (`EXPERT_CACHE.md:37`).
- Synchronization: route-ID producer split synchronize before the D2H read
  (`ggml/src/ggml-backend.cpp:2976-2983`); one synchronize after the sidecar
  output download (`EXPERT_CACHE.md:33`).
- VRAM cost: slot-pool capacity per layer; on the GTX 1080 the useful
  configuration is constrained by `-fitt`/`-ncmoe`
  (`EXPERT_CACHE_OPTIMIZATIONS_LOG.md:3052-3058`).
- Host RAM cost: original weights plus promotion staging.
- PCIe dependency: promotion pipeline only (async, outside the timed loop).
- Quantization/APEX implications: slot copies preserve quantized types; the
  sidecar has no APEX scale nodes (design spec line 23), so scaled paths rely on
  the native fallback for equivalence. The fast-reject matrix measured parity
  within noise (mean paired delta +0.03%, 95% CI -3.78% to +3.84%;
  `EXPERT_CACHE.md:199`).
- Complexity: high (slot pools, dispatch build, admission, telemetry);
  implemented, tested, and measured.
- Expected Pascal suitability: working on the GTX 1080, but the current APEX
  Compact measurement had no useful 8/8 sidecar work and must not be used to
  relax the fallback or admission contract (design spec line 24).

### 3.3 Retired partial CPU/GPU executor

- Weight location: resident slot bundles (GPU) plus host weights (CPU side).
- Compute backend: CUDA for selected routes, CPU for the remainder, overlapped
  via events.
- Route handling: fixed TG1 partial executor: seven persistent exact-K GPU
  graphs, seven exact-M CPU graphs, GPU-host-pinned exchange storage, and the
  required event set, constructed during scheduler graph allocation; direct
  execution covers every 1/8 through 7/8 mask with overlapped GPU/CPU routes and
  a canonical host result (`EXPERT_CACHE.md:20-21`).
- PP/TG1/TG>1: TG1-only; production demotes 7/8 to the native CPU bundle after
  the 2026-08-31 latency recheck; 0-6 stays CPU-base and 8/8 retains the GPU
  sidecar (`EXPERT_CACHE.md:20`, `:34`). PP and TG>1 never enter this executor.
- H2D traffic: zero timed expert-weight upload (none of the route-ready paths
  uploads expert weights during timed decode, `EXPERT_CACHE.md:43`).
- Synchronization: event-driven dual-device concurrency (`EXPERT_CACHE.md:20`);
  the executor remains reachable for focused tests and development via
  `GGML_EXPERT_CACHE_HETERO_CONCURRENT`
  (`ggml_expert_cache_hetero_concurrent_enabled()`,
  `ggml/src/ggml-backend.cpp:2270-2272`).
- VRAM cost: persistent exact-K graph storage, sidecar slots, and pinned
  exchange buffers.
- Host RAM cost: weights plus GPU-host-pinned exchange storage.
- PCIe dependency: exchange buffers cross host-pinned memory.
- Quantization/APEX implications: GPU graphs carry no scale nodes (sidecar
  limitation); output is the canonical host result (`EXPERT_CACHE.md:21`).
- Complexity: very high; retired for production.
- Expected Pascal suitability: not suitable. Measured 7/8 APEX TG1 latency:
  CPU-base median 138 us (P95 317 us) vs serial median 427 us (P95 696 us) and
  concurrent median 424 us (P95 660 us) (`EXPERT_CACHE.md:201`); runtime
  profiling found the 1-6-hit and 7/8 heterogeneous cases slower than the
  CPU-base path (`EXPERT_CACHE.md:31`).

### 3.4 WackMall hot/cold expert tiering (external reference)

- Weight location: hot experts in GPU slots (S per layer x 3 matrices), cold
  experts RAM-resident with mmap fallback (ARCHITECTURE.md sec 1-2).
- Compute backend: CUDA for hot (`mul_mat_id` on the `.hot` tensor), CPU for
  cold (`MUL_MAT_ID_COLD`/`MOE_COLD`, ARCHITECTURE.md sec 3).
- Route handling: sentinel LUT rewrite `ids_hot = get_rows(lut, ids)`; cold
  selections map to a zeroed sentinel and contribute exact zero; a `ggml_add`
  merges GPU and CPU partial results; dispatch follows the `src0` buffer
  (ARCHITECTURE.md sec 2-3, 9).
- PP/TG1/TG>1: tiered kernels engage only for graphs of at most TMAX tokens
  (default 16); larger prompt graphs keep the stock path while `MOE_COUNT`
  harvests router decisions from large batches (ARCHITECTURE.md sec 3c).
  PR #26563 states the tier engages only at `n_tokens == 1`. Merged-request
  threads report severe PP regressions (see
  `docs/superpowers/analysis/miltos-expert-tier-pp-regression.md`); the design
  spec (line 41) records that the research fork limits tiering to small token
  counts and leaves larger prompt graphs on the stock path.
- H2D traffic: hot expert promotions and evictions (copy or move mode,
  hash-verified swap handshake, throttled per-pair transfer queue; PR #26824
  items 4, 11, 13, 16).
- Synchronization: store mutation only after a full scheduler synchronize
  following graph compute (ARCHITECTURE.md sec 5, sec 9).
- VRAM cost: S slots x 3 matrices per layer with a 512 MiB flat reserve and
  auto-fit sizing (ARCHITECTURE.md sec 4).
- Host RAM cost: demand RAM pool plus speculative prefetch pool plus mmap
  fallback (ARCHITECTURE.md sec 6-7); madvise(DONTNEED) frees ~5.4 GiB VmRSS
  on 35B models (sec 6).
- PCIe dependency: promotions and mmap fallback reads.
- Quantization/APEX implications: per-expert scale handling (PR #26563); slot
  sizing derived from the actual quantization (ARCHITECTURE.md sec 4); `mmq
  mul_mat_id` `ncols_max` relaxed only for `.hot` tensors (sec 9); verified on
  Q2_M, Q5_K_P, Q4_K_M, IQ2_M models in the PR tables.
- Complexity: very high (graph rewrite, new CPU ops, RAM pool, prefetch,
  heatmap sidecar); never merged upstream.
- Expected Pascal suitability: none of the cited benchmarks was measured on a
  GTX 1080; no transfer of its values is claimed (see the dedicated regression
  document, section 5).

### 3.5 moe-l2 selected-expert H2D LRU (external reference)

- Weight location: all experts host-resident and pinned (selective pin via a
  router map; README "How it works"); activated experts are copied to VRAM and
  hot experts stay in a GPU LRU (design-decisions, host-buffer section).
- Compute backend: CUDA; per-expert get/set inside `ggml_cuda_mul_mat_id`
  (`references/llama.cpp-gpu-lru-cache/ggml-cuda.cu:2141-2231`).
- Route handling: per-step list of used expert indices; per-expert cache lookup;
  miss -> H2D copy + compute + write-back to the LRU
  (`ggml-cuda.cu:2151-2154`, symbols `ggml_cuda_expert_cache_get` /
  `ggml_cuda_expert_cache_set` / `ggml_cuda_expert_cache_maybe_init`).
- PP/TG1/TG>1: per-step granularity. The author's A2 on-demand packing variant
  measured 10.76 t/s on Qwen3.6-35B and was dropped because per-fetch overhead
  ate the GPU compute advantage; the final host-buffer approach measured
  46.8 t/s on an RTX 4090 (design-decisions, Phase 3).
- H2D traffic: timed selected-expert H2D per step on cache miss; this violates
  the mapped-host investigation's zero-timed-H2D invariant (design spec line
  150).
- Synchronization: cache D2D copies run on the CUDA backend compute stream,
  which the patch exposes to the scheduler (`ggml-cuda.cu:5877-5880`).
- VRAM cost: LRU slot pool only; ~2.1 GB for Qwen3.6-35B-A3B on an RTX 4090
  (design-decisions, Phase 3). On the 235B validation, a 512-slot cap gave 0
  hits during generation and 9.3 GB VRAM reached only a 50% hit rate
  (design-decisions, 235B validation).
- Host RAM cost: pinned experts (whole-pin vs selective-pin vs on-demand
  variants; README).
- PCIe dependency: every activated-expert copy crosses PCIe; the author
  measured UVA direct read slower than `cudaMemcpy` (section 4 below).
- Quantization/APEX implications: Q2_K-focused; IQ1_M MMQ routing fix
  (README bins-v0.7.0); APEX Compact behavior is not evidenced in the cited
  sources.
- Complexity: high (OpenAI-compatible proxy plus patched binaries plus per-
  expert `cudaHostRegister`/`cudaHostUnregister` range tracking,
  `ggml-cuda.cu:5698-5837`).
- Expected Pascal suitability: no cited benchmark was measured on a GTX 1080;
  the author ships sm_61-inclusive multi-arch binaries (README) but reports no
  GTX 1080 numbers; no transfer of values is claimed.

### 3.6 Direct mapped-host GPU reads (candidate, this investigation)

- Weight location: registered GGUF pages or a single persistent mapped host
  staging copy (design spec lines 55-59, 89-95).
- Compute backend: CUDA kernels dereference device-visible mapped host
  addresses; no timed weight copies.
- Route handling: deterministic route IDs in the isolated benchmark (design
  spec line 81); ordinary route grouping and batch semantics are a post-gate
  production requirement (design spec line 141).
- PP/TG1/TG>1: benchmark covers TG1, TG4, TG8, PP128, PP512 (design spec line
  85). PP remains on the stock batched path until mapped-host `MUL_MAT_ID` has
  demonstrated preserved batch semantics (design spec line 154).
- H2D traffic: zero timed expert-weight H2D (design spec lines 53-57, 150).
- Synchronization: CUDA events for device time; one completion synchronization
  per latency sample; no per-layer backend-wide synchronize (design spec lines
  110, 153).
- VRAM cost: zero for expert weights (benchmark graph buffers only).
- Host RAM cost: model pages or staging; startup registration/staging time and
  RAM cost are reported separately (design spec line 92).
- PCIe dependency: every expert-weight read traverses the CPU-GPU interconnect;
  mapped-memory accesses have higher latency and lower bandwidth than device
  memory (CUDA Programming Guide, mapped-memory section:
  https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#mapped-memory).
- Quantization/APEX implications: benchmark preserves actual types, dimensions,
  stride, fused or unfused layout, and APEX scale values from a real Qwen3.6
  APEX Compact GGUF (design spec lines 76-77, 112-120).
- Complexity: moderate for the isolated benchmark (single routed FFN, no
  scheduler); the production path is high and gated on a positive result
  (design spec lines 136-146).
- Expected Pascal suitability: the GTX 1080 is a UVA-capable discrete GPU and
  mapped host memory is supported on all current GPUs, but kernel access is
  interconnect-bound (CUDA Programming Guide, mapped-memory section:
  https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#mapped-memory). Windows
  uses the limited unified-memory paradigm (CUDA Programming Guide, limited
  unified-memory support section:
  https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#limited-unified-memory-support), so direct
  `cudaHostRegister` of the mmap-backed GGUF allocation is an open question the
  benchmark must validate (design spec lines 91-92). Expected outcome unknown
  until measured; that is the purpose of the oracle.

## 4. moe-l2 direct-UVA rejection record

The design directs this row's sourcing (design spec lines 29-37). The
authoritative record is moe-l2's design-decisions document
(https://raw.githubusercontent.com/yalun753/moe-l2/main/references/en/design-decisions_EN.md),
section "235B ultimate validation (Qwen3-235B-A22B, 2026-08-02)":

> UVA direct read (cudaHostRegister): 5.8 GB/s direct vs 25.1 GB/s cudaMemcpy
> (4x slower); cuBLAS on UVA pointers 11x slower

The design concludes (line 31): moe-l2 "is not evidence that direct mapped-host
GPU reads are a winning path."

moe-l2's shipped path differs from the candidate being measured:

- Shipped: experts stay CPU-resident and pinned; the scheduler copies only the
  activated experts to VRAM per step; hot experts stay in an LRU cache on the
  GPU to avoid PCIe round-trips (design-decisions, host-buffer section). Timed
  selected-expert H2D is the mechanism, not an accident.
- Candidate: zero timed expert-weight H2D (design spec line 150); the GPU reads
  persistent mapped host memory directly.

The moe-l2 record is useful for registration lifetime and cache mechanics
(design spec line 31) - e.g. per-expert `cudaHostRegister` ranges, overlap
merging, and eviction ordering (`ggml-cuda.cu:5698-5837`) - but its UVA
bandwidth measurement is a caution for the candidate, not a verdict: it was
measured in a different model and granularity context (235B Q2_K), while the
candidate benchmark measures whole-bundle FFN graphs on Qwen3.6 APEX Compact
geometry, not raw copy bandwidth.

## 5. Isolated mapped-host benchmark (first executable experiment)

`tests/test-moe-mapped-host-bench.cpp` does not exist in the tree yet
(verified by glob); the design mandates adding it beside the current MoE oracle
targets (design spec line 72). It is the first executable experiment of this
epic; nothing else may be built before it. The full contract (design spec lines
70-120):

- Modes: CPU control (original host tensor, CPU compute), VRAM control
  (persistent device copy, CUDA), and candidate (registered GGUF pages or
  persistent mapped host staging, CUDA direct mapped-host read). Timed
  expert-weight H2D is 0 in all three modes (lines 53-57).
- Shapes and counts: TG1, TG4, TG8, PP128, PP512; 100 warmups and 1,000 timed
  iterations per mode and shape (lines 83-85).
- Real-model tensor requirement: an actual Qwen3.6 APEX Compact GGUF supplied
  through a required model argument; host-resident Gate, Up, and Down expert
  tensors from one MoE layer preserving actual types, dimensions, stride,
  fused or unfused layout, and APEX scale values (lines 74-77). No synthetic
  weights.
- Graph: only the routed FFN (Gate and Up projections, production-equivalent
  scale lookup and application, SwiGLU, Down projection, route weighting,
  reduction); router, normalization, residual, attention, and scheduler are
  omitted (lines 78-80). Deterministic activations, route IDs, and route
  weights (line 81). Covers 1, 2, 4, and 8 selected experts with mixed and
  repeated selections (line 82).
- Storage lifetime: the fixture owns every registered range, mapped pointer,
  staging allocation, and CUDA event (line 87). First attempt direct
  `cudaHostRegister` of a page-aligned GGUF-backed range with the mapped flag,
  then obtain its device-visible address (line 90); validate that the CUDA
  device can map host memory and that the registration is accepted on the
  Windows mmap allocation (line 91); if direct registration is unsupported or
  unsafe, make a single persistent `cudaHostAllocMapped` staging copy during
  setup and report the startup copy time and RAM cost separately (line 92).
  Register or allocate before warmup; release only after all associated stream
  work completes (line 93). Never allocate, register, unregister, copy weights,
  grow vectors, construct graphs, or allocate backend buffers in the timed loop
  (line 95).
- Measurement contract: for every shape and storage mode, report sample count,
  P50, P95, mean, and standard deviation wall latency; CUDA event elapsed time
  for the FFN graph; CPU wall time for the CPU control; selected and unique
  expert counts; registration and setup time; directly registered bytes and
  staging bytes; timed expert-weight H2D and D2H bytes; explicit
  synchronization count and backend-wide synchronization count (lines 99-108).
  Use CUDA events for device time and one completion synchronization per
  latency sample; do not use a per-layer backend-wide synchronization (line
  110).
- Correctness contract: before timing every shape, compare CPU vs VRAM, CPU vs
  mapped host, and VRAM vs mapped host with quantization-appropriate numeric
  tolerances; exercise nontrivial APEX scale values, repeated IDs, mixed IDs,
  and all selected-expert counts so a missing scale path or route-indexing
  error fails visibly (lines 112-120).

Invariants during the experiment (design spec lines 61-68, 148-155): scheduler
assignment, cache admission and the 8/8 threshold, native fallback semantics,
sidecar behavior, `--fit` behavior, and ngram-mod or MTP execution are
unchanged. The current sidecar and native fallback remain untouched even if the
benchmark succeeds.

## 6. Decision gate (Gate 1)

Copied verbatim from the approved design
(`docs/superpowers/specs/2026-08-31-low-vram-moe-mapped-host-design.md`,
"Decision Gate", lines 122-134):

### Strong positive

Mapped-host top-8 TG1 is faster than the CPU control at both P50 and P95, and
timed expert-weight H2D bytes are exactly zero.

### Conditional positive

Mapped-host TG1 is within 5 percent of CPU at both P50 and P95, timed
expert-weight H2D bytes are exactly zero, and at least one of TG4, TG8, PP128,
or PP512 is at least 10 percent faster than CPU.

### Negative

Neither condition holds. Record the result, retain native CPU misses plus the
existing full-hit VRAM sidecar, and stop the mapped-host production path. Do
not replace it with a copy-on-route cache under this epic.

Only a strong or conditional positive permits the post-gate production sequence
(design spec lines 136-146). The dynamic MTP null-buffer assertion is an
independent defect; no speculative-compatibility claim is allowed until it has
an evidence-backed root cause and fix (design spec line 146;
`EXPERT_CACHE.md:203` records the abort at `ggml/src/ggml-backend.cpp:213`).

## 7. Excluded approaches

- Copy-on-route expert cache: expressly excluded from this experiment (design
  spec line 59). Neither document proposes it.
- CPU/GPU partial executor: not proposed as this experiment. The retired
  partial executor is recorded above as evidence only, and the negative gate
  retains native CPU misses plus the existing full-hit VRAM sidecar (design
  spec line 134).

## 8. Conclusion

Both analysis documents (this one and
`docs/superpowers/analysis/miltos-expert-tier-pp-regression.md`) identify the
same isolated mapped-host benchmark, `tests/test-moe-mapped-host-bench.cpp`, as
the sole next executable step.

## 9. Gate 1 outcome (2026-08-31)

The experiment was executed as specified. Result: NEGATIVE. On GTX 1080
(SM 6.1, PCIe 3, Windows WDDM), direct cudaHostRegister on GGUF mmap pages
works, but mapped-host expert reads measure 6.26 GB/s effective throughput,
giving top-8 TG1 a ~1.75 ms/token PCIe floor against 0.22-0.30 ms measured
CPU TG1. Measured medians: tg1 mapped 6997 us vs CPU 303 us; the gate
predicates reject at every shape. See
`EXPERT_CACHE_OPTIMIZATIONS_LOG.md` ("Mapped-Host GPU Expert Execution
Oracle - Gate 1 Negative Result") and
`tools/results/expert-cache/mapped-host-oracle/` for the raw records.
