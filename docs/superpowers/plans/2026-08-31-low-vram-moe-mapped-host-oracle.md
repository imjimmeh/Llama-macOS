# Low-VRAM MoE Mapped-Host Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether GPU direct reads from persistent mapped host Qwen3.6 APEX expert memory can beat native CPU execution on GTX 1080 without timed expert-weight H2D copies.

**Architecture:** Keep the scheduler, cache policy, `--fit`, native CPU fallback, and full-hit sidecar unchanged. Add two source-backed analysis documents and one isolated benchmark target that loads real host-resident Qwen expert tensors, constructs only the complete routed FFN, and compares CPU, ordinary VRAM, and mapped-host CUDA storage. The benchmark is the decision gate; no production integration follows from this plan unless it passes.

**Tech Stack:** C++17, ggml C API, llama model API, CUDA runtime API, CMake, Qwen3.6-35B-A3B-APEX-Compact GGUF, JSON result files.

## Global Constraints

- Target hardware is GTX 1080, 8 GB VRAM, Pascal SM61, PCIe 3.x class host link, and Windows mmap behavior.
- Load `Qwen3.6-35B-A3B-APEX-Compact.gguf` through mmap and force its MoE tensors to CPU host memory before selecting benchmark tensors.
- Use real model Gate, Up, Down, and APEX scale tensor bytes. Synthetic quantized weights are insufficient.
- Test TG1, TG4, TG8, PP128, and PP512 with 100 warmups and 1,000 timed samples each.
- GPU candidate means direct dereference of a persistent mapped-host address. It must have zero timed expert-weight H2D bytes. A copy-on-route cache is out of scope.
- Preserve ordinary batched `MUL_MAT_ID` grouping. Do not add a scheduler interceptor, alternate CPU FFN, per-expert graph, per-token allocation, or per-layer backend-wide synchronize.
- APEX scale lookup and multiplication must remain in the complete benchmarked FFN graph.
- `moe-l2` and WackMall are research sources, not code donors. `moe-l2` rejected direct UVA dereference in its own measurements; WackMall's CPU cold path regresses PP.
- Do not change production scheduler, cache, sidecar, fit, ngram-mod, or MTP sources in this plan.
- Use `llama_build`, not `llama_build_and_test`, because the actual local model and CUDA hardware are required.
- Do not run `git add`, `git commit`, or `git push`. Repository rules require separate explicit approval for every commit.
- This plan ends at Gate 1. A strong or conditional positive result requires a new approved production-integration plan. A negative result closes the execution-path investigation.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `docs/superpowers/analysis/moe-low-vram-execution-options.md` | Source-backed comparison of baseline, current sidecar, partial executor, WackMall, and mapped-host CUDA designs. Defines the smallest viable experiment and Gate 1. |
| `docs/superpowers/analysis/miltos-expert-tier-pp-regression.md` | Evidence-backed PP regression analysis for the WackMall hot/cold implementation. |
| `tests/test-moe-mapped-host-bench.cpp` | Isolated model-backed complete MoE FFN oracle, storage fixtures, correctness checks, sample statistics, JSON output, and gate evaluation. |
| `tests/CMakeLists.txt` | Builds `test-moe-mapped-host-bench` only when `GGML_CUDA` is enabled, linking the CUDA runtime required for page registration and mapped-pointer lookup. |
| `EXPERT_CACHE.md` | Records the architecture decision and the retained execution policy after the hardware run. |
| `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` | Records commands, revision, hardware, raw JSON, numerical results, and Gate 1 outcome for every attempt. |

## Shared Interfaces

Task 2 creates these types in `tests/test-moe-mapped-host-bench.cpp`:

```cpp
enum class moe_storage_mode { cpu, vram, mapped_host };
enum class moe_mapping_kind { direct_gguf, mapped_staging, unsupported };

struct moe_shape_case {
    const char * name;
    int64_t n_tokens;
    int64_t n_selected_experts;
};

struct moe_expert_selection {
    std::vector<int32_t> ids;
    std::vector<float> weights;
    int64_t unique_experts;
};

struct moe_sample_stats {
    double min_us;
    double median_us;
    double p95_us;
    double mean_us;
    double stddev_us;
};
```

Task 3 adds the model and complete-FFN interfaces:

```cpp
struct qwen_apex_expert_bundle {
    const ggml_tensor * gate;
    const ggml_tensor * up;
    const ggml_tensor * down;
    const ggml_tensor * gate_scale;
    const ggml_tensor * up_scale;
    const ggml_tensor * down_scale;
    bool fused_gate_up;
};

struct moe_ffn_oracle {
    ggml_context * ctx;
    ggml_cgraph * graph;
    ggml_tensor * output;
    ggml_tensor * ids;
    ggml_tensor * route_weights;
    ggml_backend_buffer_t mutable_buffer;
};
```

Task 4 adds mapped-host ownership:

```cpp
struct moe_mapped_host_range {
    const uint8_t * source;
    uint8_t * registered_base;
    void * device_base;
    size_t source_offset;
    size_t registered_bytes;
    size_t staging_bytes;
    moe_mapping_kind kind;

    bool is_usable() const;
    const void * device_data() const;
    void reset();
};
```

Task 5 emits a JSON object per `(storage_mode, shape, route_pattern)` with the timing, byte, synchronization, registration, and correctness fields named in the task.

### Task 1: Write the architecture analysis documents

**Files:**
- Create: `docs/superpowers/analysis/moe-low-vram-execution-options.md`
- Create: `docs/superpowers/analysis/miltos-expert-tier-pp-regression.md`
- Reference: `docs/superpowers/specs/2026-08-31-low-vram-moe-mapped-host-design.md`
- Reference: `EXPERT_CACHE.md:7-48, 192-203`
- Reference: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md:3030-3067`
- Reference: `ggml/src/ggml-backend.cpp:1104-1125, 2636-2660, 2952-3055, 3379-3394`
- Reference: `src/llama-graph.cpp:1520-1543`

**Interfaces:**
- Consumes: approved design specification and the cited local and external sources.
- Produces: the phase-zero evidence record that tasks 2-7 cite; it does not alter executable interfaces.

- [ ] **Step 1: Draft the execution-options comparison table**

  Write one row each for native CPU MoE, current full-bundle VRAM cache, retired partial CPU/GPU executor, WackMall hot/cold tiering, and direct mapped-host GPU reads. Every row must state weight location, compute backend, route handling, PP/TG1/TG>1 behavior, H2D traffic, synchronization, VRAM and host RAM cost, PCIe dependency, quantization and APEX-scale implications, complexity, and expected Pascal suitability.

  The `moe-l2` row must cite its source record that direct UVA dereference was rejected and distinguish its actual selected-expert H2D LRU from the candidate being measured.

- [ ] **Step 2: Draft the WackMall PP regression analysis**

  Cite upstream PRs `#26563` and `#26824` and `miltos22/llama-wackMall/ARCHITECTURE.md`. Attribute PP regression to the CPU cold path, hot/cold route fragmentation, graph rewrite and merge work, and the implementation's `TMAX` decode gate. Explicitly label unknown post-closure behavior as unknown. Do not claim its reported benchmark values apply to GTX 1080.

- [ ] **Step 3: Specify the smallest viable experiment and stop condition**

  State that `tests/test-moe-mapped-host-bench.cpp` will be the first executable experiment. Record the CPU, VRAM, and mapped-host modes, five shape cases, 100/1,000 sample rule, real-model tensor requirement, and zero timed expert-weight H2D invariant.

  Copy the exact strong, conditional, and negative Gate 1 criteria from the approved design. State that the current sidecar and native fallback remain untouched even if the benchmark succeeds.

- [ ] **Step 4: Review documents against sources**

  Verify every numerical or behavioral external claim has a URL, every local behavior has a repository path and symbol, and neither document proposes copy-on-route caching or a CPU/GPU partial executor as this experiment.

  Expected result: both analysis documents identify the same isolated mapped-host benchmark as the sole next executable step.

### Task 2: Add a CUDA-only benchmark target and deterministic case contract

**Files:**
- Create: `tests/test-moe-mapped-host-bench.cpp`
- Modify: `tests/CMakeLists.txt:359-367`
- Reference: `tests/test-moe-oracle-bench.cpp:34-68, 93-117, 348-507`
- Reference: `tests/test-moe-geometry-report.cpp:29-69`
- Reference: `ggml/include/ggml-cuda.h:22-43`

**Interfaces:**
- Consumes: Task 1's exact benchmark contract.
- Produces: `moe_shape_case`, `moe_expert_selection`, `moe_sample_stats`, argument parsing, and the `test-moe-mapped-host-bench` executable.

- [ ] **Step 1: Write the failing deterministic case self-test**

  Create `--self-test` before model loading. It must construct exactly these cases:

  ```cpp
  static const moe_shape_case expected_cases[] = {
      { "tg1",   1,   8 },
      { "tg4",   4,   8 },
      { "tg8",   8,   8 },
      { "pp128", 128, 8 },
      { "pp512", 512, 8 },
  };
  ```

  For each case, assert that `ids.size() == n_tokens * n_selected_experts`, all IDs are in `[0, 255]`, route weights have the same count, and `unique_experts` reports both an all-distinct pattern and a repeated-ID pattern correctly. Use deterministic patterns such as `(token * 17 + route * 31 + 7) % 256` and `route % 2`.

- [ ] **Step 2: Build and run the self-test red phase**

  Add this conditional target declaration immediately after the existing MoE benchmark declarations:

  ```cmake
  if (GGML_CUDA)
      find_package(CUDAToolkit REQUIRED)
      llama_build(test-moe-mapped-host-bench.cpp)
      target_link_libraries(test-moe-mapped-host-bench PRIVATE CUDA::cudart)
  endif()
  ```

  Run:

  ```powershell
  cmake --build build --config Release --target test-moe-mapped-host-bench
  build/bin/Release/test-moe-mapped-host-bench.exe --self-test
  ```

  Expected before implementation: the target or `--self-test` path fails because the source and contract do not exist.

- [ ] **Step 3: Implement parsing and contract validation**

  Implement `--self-test`, `-m/--model`, `--layer`, `--shapes`, `--pattern distinct|repeated`, `--warmup`, `--reps`, and `--json`. Default to the five required shapes, 100 warmups, and 1,000 repetitions. Reject zero repetitions, any shape outside the five required names, missing model in non-self-test mode, and a layer outside the actual model's MoE block count.

  Reuse the existing `compute_stats()` median and P95 implementation from `tests/test-moe-oracle-bench.cpp`; retain its inclusive `floor(0.95 * (n - 1))` P95 index.

- [ ] **Step 4: Build and run the self-test green phase**

  Run the same build and self-test commands. Expected result: exit code zero and JSON or text output listing all five shape names and both route patterns.

- [ ] **Step 5: Review target scope**

  Confirm the target is omitted from non-CUDA builds, is not registered as a CTest requiring a local model, and links no new production component.

### Task 3: Load real host-resident APEX tensors and build CPU/VRAM complete-FFN controls

**Files:**
- Modify: `tests/test-moe-mapped-host-bench.cpp`
- Reference: `tests/test-moe-heterogeneous-bench.cpp:59-105`
- Reference: `tests/test-moe-geometry-report.cpp:34-69`
- Reference: `common/arg.cpp:2720-2741`
- Reference: `src/llama-graph.cpp:1520-1543`
- Reference: `tests/test-moe-oracle-bench.cpp:225-345`

**Interfaces:**
- Consumes: Task 2 case data and CLI.
- Produces: `qwen_apex_expert_bundle`, `moe_ffn_oracle`, `load_qwen_apex_bundle()`, and `run_cpu_or_vram_case()`.

- [ ] **Step 1: Write the failing model-inspection assertion path**

  Add `--inspect` mode. After model initialization, require these names for the selected layer:

  ```cpp
  blk.<layer>.ffn_gate_exps.weight
  blk.<layer>.ffn_up_exps.weight
  blk.<layer>.ffn_down_exps.weight
  blk.<layer>.ffn_gate_exps.weight.scale
  blk.<layer>.ffn_up_exps.weight.scale
  blk.<layer>.ffn_down_exps.weight.scale
  ```

  Accept fused GateUp only when it has `ne[2] == 256` and no separate Up tensor. Require every selected expert projection tensor to use a host buffer and require its scale tensor to be present. Print type, `ne[]`, `nb[]`, `ggml_nbytes()`, and host-buffer status.

- [ ] **Step 2: Run the inspection red phase**

  Run:

  ```powershell
  build/bin/Release/test-moe-mapped-host-bench.exe --inspect -m C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf --layer 5
  ```

  Expected before implementation: failure because the test does not initialize the model, force MoE tensors to CPU, or resolve the required tensor names.

- [ ] **Step 3: Implement model loading and tensor discovery**

  Initialize `common_params` as the real-model test harness does, with mmap enabled, 14 CPU threads, `n_gpu_layers = 99`, and `params.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override())`. Use `common_init_from_params()` and `llama_model_get_tensor()`.

  Build the FFN graph from external expert tensor descriptors and fresh activation, ID, route-weight, activation, and output tensors. The CPU and VRAM controls must both run:

  ```text
  Gate or GateUp MUL_MAT_ID
  -> APEX get_rows(scale, ids) and multiply
  -> Up when unfused
  -> APEX get_rows(scale, ids) and multiply
  -> SwiGLU
  -> Down MUL_MAT_ID
  -> APEX get_rows(scale, ids) and multiply
  -> route weighting and expert reduction
  ```

  Keep model tensor descriptors read-only. Allocate CPU mutable tensors on the CPU backend and CUDA mutable tensors on the CUDA backend. Copy real expert bytes once into persistent VRAM control buffers before warmup; report that setup copy separately.

- [ ] **Step 4: Implement CPU versus VRAM correctness checks**

  For every required shape and each `distinct` and `repeated` route pattern, run CPU then VRAM once before timing. Copy both outputs to host and calculate max absolute error, mean relative error, and normalized mean square error. Reject a result when it contains a non-finite value or exceeds the tolerances already used by `tests/test-moe-partial-hit-oracle.cpp`:

  ```cpp
  static constexpr double k_max_nmse = 0.0002;
  static constexpr double k_max_mean_relative_error = 0.005;
  ```

  Print the three errors and the selected/unique expert counts for every case.

- [ ] **Step 5: Run the CPU/VRAM green phase**

  Run:

  ```powershell
  build/bin/Release/test-moe-mapped-host-bench.exe -m C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf --layer 5 --shapes tg1 --pattern distinct --warmup 1 --reps 1 --json tools/results/expert-cache/mapped-host-oracle/cpu-vram-smoke.json
  ```

  Expected result: CPU and VRAM controls pass numerical comparison, report actual projection and scale types, and write one valid JSON result record per control. The record must contain no mapped-host result yet.

### Task 4: Add persistent direct mapped-host expert backing

**Files:**
- Modify: `tests/test-moe-mapped-host-bench.cpp`
- Reference: `ggml/src/ggml-cuda/ggml-cuda.cu:1257-1321, 1868-1937, 4645-4677`
- Reference: `ggml/include/ggml-cuda.h:22-43`

**Interfaces:**
- Consumes: `qwen_apex_expert_bundle` and FFN graph builders from Task 3.
- Produces: `moe_mapped_host_range`, `make_direct_gguf_mapping()`, `make_staging_mapping()`, and `run_mapped_host_case()`.

- [ ] **Step 1: Write the failing mapped-host eligibility checks**

  Add `--mode mapped_host`. Before graph execution, require all of these conditions:

  ```cpp
  cudaDeviceGetAttribute(&can_map_host_memory, cudaDevAttrCanMapHostMemory, 0);
  cudaHostRegister(page_aligned_base, registered_bytes,
                   cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterReadOnly);
  cudaHostGetDevicePointer(&device_base, page_aligned_base, 0);
  ```

  Derive `page_aligned_base`, `source_offset`, and `registered_bytes` from each tensor's data pointer and the operating-system page size. Reject direct registration when the range cannot be page aligned or when registration fails. The test must print the CUDA error string and classify the direct mode as unavailable instead of continuing with an unregistered pointer.

- [ ] **Step 2: Run the mapped-host red phase**

  Run the Task 3 smoke command with `--mode mapped_host`. Expected before implementation: a clearly reported mapped-host setup failure, no benchmark samples, no timed expert-weight H2D, and a nonzero exit status.

- [ ] **Step 3: Implement direct registration and persistent staging fallback**

  Implement `moe_mapped_host_range::reset()` so it waits for the final stream event before `cudaHostUnregister()` or `cudaFreeHost()`. It must be idempotent.

  `make_direct_gguf_mapping()` registers the original page-aligned mmap range and returns a device pointer offset by `source_offset`.

  `make_staging_mapping()` calls `cudaHostAlloc()` with `cudaHostAllocPortable | cudaHostAllocMapped`, copies the complete projection bytes before warmup, and resolves the mapped device pointer with `cudaHostGetDevicePointer()`. It records the one-time copy duration and exact staging bytes. It must never call `cudaMemcpy` or `ggml_backend_tensor_set()` after warmup begins.

  Bind the mapped device address as the read-only source data pointer for the existing CUDA `GGML_OP_MUL_MAT_ID` graph. Do not introduce a custom quantized matmul kernel. Keep APEX scales in the same graph; copy their small immutable tensors to the CUDA control buffer during setup and record their setup bytes separately from expert-weight bytes.

  If the existing CUDA backend rejects the source descriptor or the graph produces a CUDA error, emit `mapped_host_status: "unsupported"`, preserve the direct-registration diagnostics, and stop the mapped-host case without modifying CUDA backend or scheduler source. This is a negative Gate 1 outcome, not permission to add a fallback executor.

- [ ] **Step 4: Implement mapped-host correctness checks**

  For every shape and route pattern, run CPU, VRAM, and mapped-host once before timing. Require the same finite-value and error thresholds from Task 3 for CPU versus mapped and VRAM versus mapped. Also assert that the mapped-host record reports:

  ```json
  {
    "timed_expert_weight_h2d_bytes": 0,
    "registration_count": 1,
    "timed_registration_count": 0,
    "timed_graph_build_count": 0
  }
  ```

- [ ] **Step 5: Run the mapped-host green phase**

  Run the Task 3 smoke command with `--mode all`. Expected result: either all three modes pass the numerical checks and report `mapping_kind` as `direct_gguf` or `mapped_staging`, or mapped host reports `unsupported` with its original CUDA failure. Never silently substitute a VRAM copy for mapped-host mode.

### Task 5: Add timing, telemetry, and Gate 1 output

**Files:**
- Modify: `tests/test-moe-mapped-host-bench.cpp`
- Reference: `ggml/include/ggml-backend.h:124-129, 208-211`
- Reference: `tests/test-moe-oracle-bench.cpp:34-68, 192-215, 322-345`

**Interfaces:**
- Consumes: all three storage modes and correctness results.
- Produces: timed JSONL rows, `evaluate_gate_one()`, and a final `gate_one` JSON object.

- [ ] **Step 1: Write the failing result-schema assertion**

  Before the timed loop, define one JSON output row that contains exactly these required fields:

  ```text
  mode, mapping_kind, shape, n_tokens, n_selected_experts, unique_experts,
  samples, wall_min_us, wall_median_us, wall_p95_us, wall_mean_us,
  wall_stddev_us, cuda_median_us, cuda_p95_us, cpu_median_us,
  registration_us, direct_registered_bytes, staging_bytes,
  setup_expert_weight_h2d_bytes, timed_expert_weight_h2d_bytes,
  timed_expert_weight_d2h_bytes, explicit_sync_count,
  backend_wide_sync_count, max_abs_error, mean_relative_error, nmse,
  mapped_host_status
  ```

  Assert before writing a result that the mapped-host row has zero timed expert-weight H2D bytes and zero backend-wide synchronization count.

- [ ] **Step 2: Run the telemetry red phase**

  Run the Task 4 smoke command with `--json`. Expected before implementation: the output lacks one or more required fields or has no CUDA event timing, and the command exits nonzero with the missing field named.

- [ ] **Step 3: Implement timed collection**

  Use one persistent pair of `ggml_backend_event_t` CUDA events around each graph compute. Record the wall-clock sample with `ggml_time_us()`, record CUDA elapsed time from the events, and synchronize only the completion event required to read that sample. Do not call `ggml_backend_synchronize()` inside the benchmark loop.

  Reserve both sample vectors for `reps` before warmup. Increment explicit synchronization count only when the completion event is synchronized. Set backend-wide synchronization count only at teardown if the backend requires it; it must remain zero during timed samples.

  Perform all VRAM expert setup copies and mapped staging copies before a boolean `timing_started` becomes true. Any expert-weight copy attempted after that point must abort the run and set `timed_expert_weight_h2d_bytes` to the attempted byte count.

- [ ] **Step 4: Implement Gate 1 evaluation**

  Evaluate top-8 `tg1` P50 and P95 against CPU, then evaluate TG4, TG8, PP128, and PP512. Emit exactly one of:

  ```text
  strong_positive
  conditional_positive
  negative
  unsupported
  ```

  Use these predicates:

  ```cpp
  const bool strong = mapped_tg1.p50_us < cpu_tg1.p50_us
                   && mapped_tg1.p95_us < cpu_tg1.p95_us
                   && mapped_tg1.timed_expert_weight_h2d_bytes == 0;

  const bool tg1_equivalent = mapped_tg1.p50_us <= cpu_tg1.p50_us * 1.05
                           && mapped_tg1.p95_us <= cpu_tg1.p95_us * 1.05;

  const bool batched_win = any_case_has_mapped_latency_at_most_cpu_times(0.90);
  const bool conditional = tg1_equivalent && batched_win
                        && mapped_tg1.timed_expert_weight_h2d_bytes == 0;
  ```

  `unsupported` is selected only when no mapped-host graph ran. `negative` is selected when mapped-host runs but neither positive predicate holds. Include CPU, VRAM, and mapped P50/P95 figures in the final object.

- [ ] **Step 5: Run the telemetry green phase**

  Run the Task 4 smoke command. Expected result: JSON contains every required field, all timed expert-weight H2D byte fields are zero, and the final object contains one valid Gate 1 status.

### Task 6: Run the complete hardware crossover matrix

**Files:**
- Create: `tools/results/expert-cache/mapped-host-oracle/2026-08-31-metadata.json`
- Create: `tools/results/expert-cache/mapped-host-oracle/2026-08-31-samples.jsonl`
- Reference: `G:/qwen3.6-35b-a3b-presets-exc-latest.ini`

**Interfaces:**
- Consumes: completed benchmark target and the fixed model.
- Produces: raw reproducible samples and a single Gate 1 record. No production source changes.

- [ ] **Step 1: Record environment metadata before execution**

  Write `2026-08-31-metadata.json` with the model absolute path, model SHA-256, executable build revision, CUDA runtime version, GPU name and compute capability, Windows version, CPU thread count `14`, mmap enabled state, selected layer, exact CLI, 100 warmups, and 1,000 repetitions.

- [ ] **Step 2: Run the full matrix once per storage mode**

  Run:

  ```powershell
  build/bin/Release/test-moe-mapped-host-bench.exe `
    -m C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf `
    --layer 5 --shapes tg1,tg4,tg8,pp128,pp512 --pattern distinct `
    --warmup 100 --reps 1000 --mode all `
    --json tools/results/expert-cache/mapped-host-oracle/2026-08-31-samples.jsonl
  ```

  Repeat with `--pattern repeated` and append the result rows to the same JSONL file. Retain stderr from each invocation beside the JSONL file.

- [ ] **Step 3: Verify the raw result contract**

  Confirm every `(mode, shape, pattern)` result contains 1,000 samples, direct or staging registration metadata, error metrics, synchronization counts, and zero timed expert-weight H2D bytes for mapped-host rows. If direct mmap registration failed but mapped staging succeeded, record both statuses; do not hide the direct-registration failure.

- [ ] **Step 4: Apply Gate 1 exactly once**

  Read only the final Gate 1 object from the `distinct` top-k=8 results. Classify it using Task 5 predicates. The repeated pattern is correctness and sensitivity evidence; it cannot replace the distinct result for the gate.

  Expected result: a source-backed `strong_positive`, `conditional_positive`, `negative`, or `unsupported` decision. A negative or unsupported result is complete and valid evidence.

### Task 7: Publish the result and close or re-plan

**Files:**
- Modify: `EXPERT_CACHE.md`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`
- Modify: `docs/superpowers/analysis/moe-low-vram-execution-options.md`
- Reference: `tools/results/expert-cache/mapped-host-oracle/2026-08-31-metadata.json`
- Reference: `tools/results/expert-cache/mapped-host-oracle/2026-08-31-samples.jsonl`

**Interfaces:**
- Consumes: raw benchmark matrix and Gate 1 record.
- Produces: the durable architecture decision and, only on a positive gate, the input to a new separate production-integration specification.

- [ ] **Step 1: Write the numerical result table**

  Add CPU, VRAM, direct-mapped, and staging-mapped P50/P95 wall and CUDA timing for TG1, TG4, TG8, PP128, and PP512. Include direct registered bytes, staging bytes, setup expert-weight H2D bytes, timed expert-weight H2D bytes, explicit synchronization count, backend-wide synchronization count, and correctness errors.

- [ ] **Step 2: Record the source and command evidence**

  In `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`, add the build revision, exact command, metadata file, JSONL file, stderr files, model hash, CUDA version, GPU, and gate classification. State whether Windows mmap registration succeeded. Do not report PCIe device-read bandwidth unless it was actually measured.

- [ ] **Step 3: Apply the stop rule**

  For `negative` or `unsupported`, state that direct mapped-host GPU expert execution is rejected for the target configuration. Retain native CPU misses plus full-hit VRAM sidecar, do not change the 8/8 threshold, and close this execution path.

  For `strong_positive` or `conditional_positive`, state only that Gate 1 permits a new design and plan for model-lifetime registration, storage-aware `MUL_MAT_ID`, bundle-atomic VRAM promotion, telemetry, fit/manual placement, and speculative validation. Do not implement any of those components in this branch under this plan.

- [ ] **Step 4: Verify documentation completeness**

  Re-read every changed documentation section. Confirm it distinguishes direct mapped-host reads from `moe-l2` copy-on-route caching, names the exact gate result, and does not imply a production-path change that did not occur.

## Plan Self-Review

### Spec coverage

- Architecture comparison and source study: Task 1.
- Real Qwen dimensions, types, layout, and APEX scales: Task 3.
- CPU, VRAM, and direct mapped-host full FFN modes: Tasks 3 and 4.
- TG1, TG4, TG8, PP128, PP512, selected and repeated experts: Tasks 2, 3, and 6.
- 100 warmups, 1,000 timed iterations, P50/P95, CPU/GPU timing, bytes, and synchronization: Task 5 and Task 6.
- mmap registration, persistent staging fallback, and lifetime safety: Task 4.
- no timed expert-weight H2D and no hot-path allocation or graph rebuild: Tasks 4 and 5.
- hard decision gate and negative stop condition: Tasks 5 and 7.
- current fallback, PP scheduler, fit, and speculative isolation: Global Constraints and Task 7.

### Placeholder scan

The plan contains no unresolved implementation placeholder. Unsupported direct mapping is an explicit measurable result with specified output and stop behavior.

### Type consistency

`moe_shape_case`, `moe_expert_selection`, `moe_sample_stats`, `qwen_apex_expert_bundle`, `moe_ffn_oracle`, and `moe_mapped_host_range` are defined once in Shared Interfaces and consumed only by subsequent tasks.
