# Cacheless Upstream Parity Design

**Goal:** Restore cacheless automatic-fit TG512 throughput parity with the upstream baseline on the GTX 1080 without changing expert-cache semantics, then remeasure whether the repaired runtime improves cache-on throughput.

## Evidence

A fresh process was used for every row in ten alternating branch/baseline pairs. Five pairs ran branch first and five ran baseline first.

```text
-m C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf
-p 0 -n 512 -r 1
-t 14 -b 4096 -ub 2048
-ctk q8_0 -ctv q8_0 -fa on
-lm mlock -fitt 256 -o jsonl
```

No expert-cache parameter was passed.

| Binary | Mean TG tok/s | Mean paired delta | Median paired delta | Positive pairs |
|---|---:|---:|---:|---:|
| Current branch `1b6898fc3` | 18.227795 | -28.909% | -29.365% | 0/10 |
| Baseline `ca3d5a3e1` | 25.647510 | reference | reference | 10/10 |

The branch rows report `expert_cache_size=0`, zero requests, and zero zero-copy hits. They also report 166 CPU-host MoE route nodes, which confirms that automatic fitting leaves host expert weights on the decode graph.

Two branch-only hot paths explain the regression mechanism:

1. Upstream `ggml_backend_sched_compute_splits()` copies only the routed expert slices when a host WEIGHTS tensor feeds `MUL_MAT_ID`. The branch removed that path in `681e8e404`; its input loop performs a generic full-tensor copy before the later cache state machine.
2. Commit `c9d0d8c4a` added `cudaStreamSynchronize()` after every computed CUDA node. CUDA graphs are disabled on the GTX 1080 because Pascal compute capability 6.1 is below the Volta 7.0 gate, so every direct graph execution pays this synchronization.

The comparison proves the end-to-end regression but does not isolate the percentage caused by each hot path. Each repair must therefore be benchmarked independently.

## Scope

1. Remove the branch-only per-node CUDA synchronization and its diagnostic print.
2. Restore upstream cacheless routed-expert subset copying.
3. Bypass expert-cache bookkeeping when no cache exists.
4. Preserve all current cache-enabled slot remapping, route-ready sidecar, CPU fallback, and telemetry behavior.
5. Benchmark after each source change before combining conclusions.
6. Rerun retained cache-on operating points only after cacheless parity passes.

## Approaches considered

### Staged upstream hot-path restoration

Restore only the divergent hot paths and keep the cache-enabled state machine intact. This is the selected approach. It isolates performance contributions, limits integration risk, and gives cacheless execution the upstream behavior without duplicating the scheduler.

### Separate cacheless scheduler implementation

Copy the complete upstream `compute_splits()` implementation beside the cache-aware implementation. This gives a strong semantic boundary but duplicates a large, fast-changing scheduler and creates permanent drift risk. Rejected.

### Rebase and replay the expert-cache changes

Rebase the fork onto the baseline and reapply the cache implementation. This has the broadest parity surface but turns a two-path regression repair into a large merge and reimplementation. Rejected.

## Stage A: remove CUDA debug serialization

Delete the branch-only block immediately after `ggml_cuda_compute_forward()` in `ggml_cuda_graph_evaluate_and_capture()`:

- `cudaStreamSynchronize(cuda_ctx->stream())`
- the `[CUDA FAIL]` diagnostic
- the manual error branch around `CUDA_CHECK`

Do not replace it. Retain upstream's intentional backend synchronization and the two `MUL_MAT_ID` fallback synchronizations. CUDA stream ordering remains the dependency mechanism between nodes.

### Stage A validation

- Build the affected CUDA and benchmark targets in Release.
- Run CUDA backend operation correctness, including `MUL_MAT_ID`.
- Run the expert-cache test executable.
- Run one model smoke with `CUDA_LAUNCH_BLOCKING=1` to surface delayed CUDA failures.
- Run one normal coherent-output smoke.
- Run the same ten-pair cacheless branch/baseline matrix and record the isolated throughput change.

The benchmark is the failing behavioral reproduction for this performance-only debug serialization. A source-text test or production test hook is prohibited.

## Stage B: restore upstream cacheless MoE subset copying

Use one `ggml_backend_sched_has_expert_cache()` predicate in graph splitting and split computation. When no backend owns an expert cache, the graph and input-copy paths must match upstream behavior:

1. `ggml_backend_sched_split_graph()` skips route-plan discovery, route census, and route-ready dispatch construction.
2. User inputs retain the immediate synchronized full copy.
3. A non-user input that is a host WEIGHTS tensor and feeds the split's `MUL_MAT_ID` consumer reads the route IDs, builds the used-expert bitset, groups consecutive expert IDs, and calls `ggml_backend_tensor_set_async()` for only those byte ranges, including upstream's final padding rule.
4. Other inputs retain the generic async-copy fallback.
5. Cache route-step, prefetch carry-forward, cache scratch reset, remapped-ID preparation, and cache-node processing do not run.
6. Standard split graph submission, event handling, callbacks, and error propagation remain shared with the cache-enabled path.

The cache-enabled path must not be rewritten during this stage. In particular, do not change slot-tensor ownership, route-ready bundle dispatch, hetero execution, CPU miss fallback, or node restoration.

### Stage B test contract

Add a deterministic scheduler test using the existing backend-spy pattern:

- no expert cache is configured;
- host WEIGHTS contain several experts;
- route IDs select non-consecutive experts;
- observed destination writes equal the selected contiguous expert runs, not the full tensor size;
- output is numerically equal to a CPU reference;
- cache requests, actions, zero-copy hits, cache RAM-to-GPU bytes, route-plan counters, and route-census counters remain zero.

The test must fail on the current generic full-copy implementation before production code changes. Existing route-ready tests remain the complementary assertion that cache-enabled ownership is unchanged.

### Stage B validation

- Run the new targeted scheduler test red, then green.
- Run the complete expert-cache test executable.
- Run CUDA backend operation correctness.
- Run the same normal and `CUDA_LAUNCH_BLOCKING=1` model smokes.
- Run the same ten-pair cacheless branch/baseline matrix.

## Parity gate

Cacheless parity passes only when all conditions hold:

- mean paired TG512 delta is within +/-3% of baseline;
- the paired 95% interval includes zero;
- no correctness test fails;
- normal model output is coherent;
- `CUDA_LAUNCH_BLOCKING=1` reports no CUDA failure;
- cacheless rows report no expert-cache requests, actions, route plans, or route-census activity.

If Stage A alone passes the gate, Stage B is still required because strict upstream behavior was selected and the full expert-tensor copy remains a known cacheless divergence.

If Stage B misses the gate, investigate remaining cacheless-only scheduler differences before changing cache-enabled behavior.

## Cache-on remeasurement

After parity passes, rerun two retained regimes against the repaired `-exc 0` control:

1. Explicit placement: `-ngl 99 -ncmoe 40` with the 3 GiB static layers 0-3 sidecar profile.
2. Automatic fit target 256: the closest active small-cache candidates, dynamic 128 MiB with periods 32 and 256.

The CUDA synchronization removal should improve both cache-off and cache-on execution on Pascal. The strict cacheless subset-copy restoration improves cache-off only. It can therefore reduce the relative cache-on delta even if absolute cache-on throughput rises.

If cache-on still performs a generic full expert-weight copy before slot or sidecar interception, defer that optimization to a separate design. The follow-up may defer cache-owned weight copies until after the cache decision and use subset copying only for genuine bypass or miss paths. It must not be bundled with parity restoration because that would prevent attribution.

## Documentation and raw evidence

Append every build, test, smoke, and benchmark attempt to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`, including rejected attempts. Update `EXPERT_CACHE.md` only after a behavior or operating point is retained. Keep every raw alternating benchmark row under `tools/results/expert-cache/`.

## Non-goals

- Do not redesign the expert-cache scheduler during parity restoration.
- Do not add a CUDA kernel or change quantized matrix multiplication.
- Do not enable CUDA graphs on Pascal.
- Do not retain a runtime environment switch for the removed debug synchronization.
- Do not claim cache improvement from absolute throughput alone; compare cache-on with the repaired cache-off control.
- Do not rebase the branch or update unrelated upstream code.
