# Cacheless Upstream Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore cacheless automatic-fit TG512 parity with upstream baseline on GTX 1080, preserve expert-cache behavior, and remeasure cache-on throughput against the repaired control.

**Architecture:** Apply two independently measured repairs. Stage A removes branch-only CUDA debug serialization. Stage B adds one scheduler-wide cache-presence predicate, restores upstream selective routed-expert copies only when no cache exists, and skips cache-only planning/bookkeeping on that path. Cache-enabled slot remapping and route-ready execution remain unchanged until parity is proven.

**Tech Stack:** C++17, CUDA, CMake/CTest, `test-expert-cache`, `test-backend-ops`, `llama-cli`, `llama-bench`, Python 3 benchmark drivers.

## Global Constraints

- Target hardware is NVIDIA GeForce GTX 1080, compute capability 6.1.
- Target model is `C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf`.
- Cacheless parity command uses `-p 0 -n 512 -r 1 -t 14 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -fa on -lm mlock -fitt 256 -o jsonl` and no expert-cache option.
- Parity requires mean paired TG512 delta within +/-3%, a paired 95% interval containing zero, correct tests, coherent output, and no CUDA failure.
- Keep cache-enabled slot remapping, route-ready sidecar, heterogeneous execution, CPU miss fallback, and node restoration unchanged during parity repair.
- Use ASCII in source, comments, documentation, and result metadata.
- Append every attempt, including rejected attempts, to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` and retain every raw JSONL row.
- Do not commit or push without explicit user approval for that exact operation. Commit commands below are prepared checkpoints, not standing authorization.

---

### Task 1: Remove Per-Node CUDA Debug Serialization

**Files:**
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu:4168-4184`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`
- Create results: `tools/results/expert-cache/cacheless-parity-stage-a/*.jsonl`

**Interfaces:**
- Consumes: upstream CUDA stream-ordering behavior in `ggml_cuda_graph_evaluate_and_capture()`.
- Produces: a CUDA node loop with no branch-only per-node synchronization; isolated Stage A throughput evidence.

- [ ] **Step 1: Record the existing failing behavioral reproduction**

Use the retained cacheless comparison as the red performance test:

```text
branch mean:              18.227795 tok/s
baseline mean:            25.647510 tok/s
mean paired delta:        -28.909%
median paired delta:      -29.365%
branch-positive pairs:    0/10
raw rows: tools/results/expert-cache/cacheless-baseline-comparison/
```

Do not rerun it before the source change. It already proves the current behavior.

- [ ] **Step 2: Delete only the debug synchronization block**

In `ggml_cuda_graph_evaluate_and_capture()`, retain the upstream compute and assertion:

```cpp
bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
if (!ok) {
    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
}
GGML_ASSERT(ok);

if (!is_concurrent_event_active) {
    try_launch_concurrent_event(node);
}
```

Delete this complete branch-only block and add no replacement:

```cpp
cudaError_t err = cudaStreamSynchronize(cuda_ctx->stream());
if (err != cudaSuccess) {
    fprintf(stderr, "[CUDA FAIL] Op failed: %s (%s, node_idx=%d)\n", node->name ? node->name : "unnamed", ggml_op_name(node->op), i);
    fflush(stderr);
    CUDA_CHECK(err);
}
```

- [ ] **Step 3: Build Stage A targets**

Run:

```bash
cmake --build build --config Release --target test-backend-ops test-expert-cache llama-cli llama-bench
cmake --build G:/code/AI/llama.cpp/build --config Release --target llama-bench
```

Expected: the four branch targets and the baseline benchmark target build successfully.

- [ ] **Step 4: Run targeted CUDA and expert-cache correctness**

Run:

```bash
build/bin/Release/test-backend-ops.exe test -b CUDA0 -o MUL_MAT_ID
build/bin/Release/test-expert-cache.exe
```

Expected: all selected `MUL_MAT_ID` cases pass; `test-expert-cache` ends with `all test-expert-cache tests passed successfully!`.

- [ ] **Step 5: Exercise delayed CUDA errors and normal output**

Run the same 32-token prompt twice, once with `CUDA_LAUNCH_BLOCKING=1` and once without it:

```bash
CUDA_LAUNCH_BLOCKING=1 build/bin/Release/llama-cli.exe -m "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" -p "Write one concise sentence about scheduler parity." -n 32 -t 14 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -fa on -lm mlock -fitt 256
build/bin/Release/llama-cli.exe -m "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" -p "Write one concise sentence about scheduler parity." -n 32 -t 14 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -fa on -lm mlock -fitt 256
```

Expected: neither run reports a CUDA failure; both outputs are coherent. Do not require token equality because sampling options are not fixed.

- [ ] **Step 6: Run the isolated Stage A ten-pair comparison**

Use a session-local Python driver, not a repository source file, with:

```python
import os
from pathlib import Path

OUTPUT_DIR = Path("tools/results/expert-cache/cacheless-parity-stage-a")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BENCHES = {
    "branch": Path(r"G:/code/AI/llamacpptuned/llama.cpp/build/bin/Release/llama-bench.exe"),
    "baseline": Path(r"G:/code/AI/llama.cpp/build/bin/Release/llama-bench.exe"),
}
COMMON = [
    "-m", r"C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf",
    "-p", "0", "-n", "512", "-r", "1", "-t", "14",
    "-b", "4096", "-ub", "2048", "-ctk", "q8_0", "-ctv", "q8_0",
    "-fa", "on", "-lm", "mlock", "-fitt", "256", "-o", "jsonl",
]
ENV = {key: value for key, value in os.environ.items()
       if not key.startswith("GGML_EXPERT_CACHE_")}
ENV["GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL"] = "0"
orders = [("branch", "baseline")] * 5 + [("baseline", "branch")] * 5
```

For pair `run` 1 through 10, execute both labels in `orders[run - 1]` with `env=ENV`, require process exit code zero, parse the single JSONL row, and write it to:

```text
tools/results/expert-cache/cacheless-parity-stage-a/2026-08-29-cacheless-parity-stage-a-{branch,baseline}-{run}.jsonl
```

Expected: 20 valid JSONL files. Do not pass `-exc`, `-excp`, `-excm`, or `-pe` to either binary.

- [ ] **Step 7: Analyze and log Stage A**

For each pair calculate:

```python
delta_pct = 100.0 * (branch_tok_s / baseline_tok_s - 1.0)
```

Record branch and baseline means, mean and median paired delta, range, positive pairs, sample standard deviation, 95% Student-t interval, commands, revision/build provenance, and raw paths in a new append-only section of `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`.

State only the isolated Stage A result. Do not attribute any remaining regression to Stage B until Stage B is measured.

- [ ] **Step 8: Prepare the Stage A commit checkpoint**

Run:

```bash
git diff --check -- ggml/src/ggml-cuda/ggml-cuda.cu EXPERT_CACHE_OPTIMIZATIONS_LOG.md tools/results/expert-cache/cacheless-parity-stage-a
```

Request explicit user approval. Only after approval:

```bash
git add -- ggml/src/ggml-cuda/ggml-cuda.cu EXPERT_CACHE_OPTIMIZATIONS_LOG.md tools/results/expert-cache/cacheless-parity-stage-a
git commit -m "ggml-cuda: remove per-node debug synchronization"
```

---

### Task 2: Specify Cacheless Subset-Copy Behavior With a Failing Test

**Files:**
- Modify: `tests/test-expert-cache.cpp:212-249`
- Modify: `tests/test-expert-cache.cpp:2021-2058`

**Interfaces:**
- Consumes: backend interface callbacks `set_tensor_async` and `cpy_tensor_async` from `ggml-backend-impl.h`.
- Produces: `test_cacheless_moe_subset_copy()` and test-only tracked copy ranges used to prove upstream selective-copy behavior.

- [ ] **Step 1: Extend the existing backend spy state**

Add test-only records beside `test_set_tensor_async_calls`:

```cpp
struct test_tensor_write {
    size_t offset;
    size_t size;
};

static const char * test_tracked_tensor_name = nullptr;
static int test_tracked_cpy_tensor_async_calls = 0;
static std::vector<test_tensor_write> test_tracked_set_tensor_async_writes;
static bool (*test_original_cpy_tensor_async)(
    ggml_backend_t backend_src,
    ggml_backend_t backend_dst,
    const ggml_tensor * src,
    ggml_tensor * dst) = nullptr;
```

Replace `test_count_set_tensor_async()` with the complete body below. Preserve its existing all-call counter for older tests; add range records only for the tracked expert tensor. Scheduler copies prefix the original name with the destination backend, so exact name equality would miss the writes.

```cpp
static void test_count_set_tensor_async(
        ggml_backend_t backend,
        struct ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size) {
    test_set_tensor_async_calls++;
    if (test_tracked_tensor_name != nullptr &&
        strstr(tensor->name, test_tracked_tensor_name) != nullptr) {
        test_tracked_set_tensor_async_writes.push_back({ offset, size });
    }
    test_original_set_tensor_async(backend, tensor, data, offset, size);
}
```

Add this callback:

```cpp
static bool test_count_cpy_tensor_async(
        ggml_backend_t backend_src,
        ggml_backend_t backend_dst,
        const ggml_tensor * src,
        ggml_tensor * dst) {
    if (test_tracked_tensor_name != nullptr && strstr(src->name, test_tracked_tensor_name) != nullptr) {
        test_tracked_cpy_tensor_async_calls++;
    }
    return test_original_cpy_tensor_async(backend_src, backend_dst, src, dst);
}
```

- [ ] **Step 2: Add the cacheless scheduler test fixture**

Add `test_cacheless_moe_subset_copy()` before `main()`. Use separate CPU-reference, host-weight, and scheduler graph contexts so scheduler allocation cannot overwrite the reference:

```cpp
static void test_cacheless_moe_subset_copy() {
    printf("testing cacheless MoE subset copy...\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_device == nullptr) {
        printf("  no GPU backend available; skipped\n");
        return;
    }

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_device, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    require(gpu_backend != nullptr);
    require(cpu_backend != nullptr);

    constexpr int64_t n_experts = 8;
    constexpr int64_t n_embd = 4;
    constexpr size_t expert_bytes = n_embd * n_embd * sizeof(float);
    const float input_data[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const int32_t route_data[] = { 0, 4 };
    std::vector<float> weights((size_t) n_experts * n_embd * n_embd, 0.0f);
    for (int64_t expert = 0; expert < n_experts; ++expert) {
        for (int64_t diagonal = 0; diagonal < n_embd; ++diagonal) {
            weights[(size_t) expert * n_embd * n_embd + diagonal * n_embd + diagonal] =
                (float) expert + 1.0f;
        }
    }

    struct ggml_init_params reference_params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * reference_ctx = ggml_init(reference_params);
    require(reference_ctx != nullptr);
    ggml_tensor * reference_weights =
        ggml_new_tensor_3d(reference_ctx, GGML_TYPE_F32, n_embd, n_embd, n_experts);
    ggml_tensor * reference_input =
        ggml_new_tensor_3d(reference_ctx, GGML_TYPE_F32, n_embd, 1, 1);
    ggml_tensor * reference_ids =
        ggml_new_tensor_2d(reference_ctx, GGML_TYPE_I32, 2, 1);
    ggml_tensor * reference_output =
        ggml_mul_mat_id(reference_ctx, reference_weights, reference_input, reference_ids);
    ggml_cgraph * reference_graph = ggml_new_graph(reference_ctx);
    ggml_build_forward_expand(reference_graph, reference_output);
    ggml_backend_buffer_t reference_buffer =
        ggml_backend_alloc_ctx_tensors(reference_ctx, cpu_backend);
    require(reference_buffer != nullptr);
    ggml_backend_tensor_set(reference_weights, weights.data(), 0, ggml_nbytes(reference_weights));
    ggml_backend_tensor_set(reference_input, input_data, 0, sizeof(input_data));
    ggml_backend_tensor_set(reference_ids, route_data, 0, sizeof(route_data));
    require(ggml_backend_graph_compute(cpu_backend, reference_graph) == GGML_STATUS_SUCCESS);
    std::vector<float> expected(ggml_nelements(reference_output));
    ggml_backend_tensor_get(reference_output, expected.data(), 0, ggml_nbytes(reference_output));

    struct ggml_init_params weights_params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * weights_ctx = ggml_init(weights_params);
    require(weights_ctx != nullptr);
    ggml_tensor * expert_weights =
        ggml_new_tensor_3d(weights_ctx, GGML_TYPE_F32, n_embd, n_embd, n_experts);
    ggml_set_name(expert_weights, "blk.0.ffn_gate_exps.weight");
    ggml_backend_buffer_t weights_buffer =
        ggml_backend_alloc_ctx_tensors(weights_ctx, cpu_backend);
    require(weights_buffer != nullptr);
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(expert_weights, weights.data(), 0, ggml_nbytes(expert_weights));

    struct ggml_init_params graph_params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(graph_params);
    require(ctx != nullptr);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, 1);
    ggml_tensor * route_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    ggml_set_input(input);
    ggml_set_input(route_ids);
    ggml_tensor * output = ggml_mul_mat_id(ctx, expert_weights, input, route_ids);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_t backends[] = { gpu_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    require(sched != nullptr);
    ggml_backend_sched_set_tensor_backend(sched, input, gpu_backend);
    ggml_backend_sched_set_tensor_backend(sched, route_ids, gpu_backend);
    require(ggml_backend_sched_alloc_graph(sched, graph));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    ggml_backend_tensor_set(route_ids, route_data, 0, sizeof(route_data));

    test_tracked_tensor_name = "blk.0.ffn_gate_exps.weight";
    test_tracked_cpy_tensor_async_calls = 0;
    test_tracked_set_tensor_async_writes.clear();
    test_original_cpy_tensor_async = gpu_backend->iface.cpy_tensor_async;
    test_original_set_tensor_async = gpu_backend->iface.set_tensor_async;
    require(test_original_cpy_tensor_async != nullptr);
    require(test_original_set_tensor_async != nullptr);
    gpu_backend->iface.cpy_tensor_async = test_count_cpy_tensor_async;
    gpu_backend->iface.set_tensor_async = test_count_set_tensor_async;

    const enum ggml_status status = ggml_backend_sched_graph_compute(sched, graph);

    gpu_backend->iface.set_tensor_async = test_original_set_tensor_async;
    gpu_backend->iface.cpy_tensor_async = test_original_cpy_tensor_async;
    test_tracked_tensor_name = nullptr;
    require(status == GGML_STATUS_SUCCESS);
    require(test_tracked_cpy_tensor_async_calls == 0);
    require(test_tracked_set_tensor_async_writes.size() == 2);
    require(test_tracked_set_tensor_async_writes[0].offset == 0);
    require(test_tracked_set_tensor_async_writes[0].size == 2 * expert_bytes);
    require(test_tracked_set_tensor_async_writes[1].offset == 4 * expert_bytes);
    require(test_tracked_set_tensor_async_writes[1].size == 2 * expert_bytes);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        require(fabsf(actual[i] - expected[i]) < 1e-5f);
    }

    ggml_backend_expert_cache_stats stats = {};
    require(!ggml_backend_sched_get_expert_cache_stats(sched, -1, &stats));
    require(stats.n_requests == 0);
    require(stats.n_route_census_nodes == 0);
    require(stats.n_route_census_plans == 0);
    require(stats.n_route_census_split_inputs == 0);
    require(stats.n_route_ready_dispatches == 0);
    require(stats.n_route_ready_classifications == 0);
    require(stats.n_route_ready_actions == 0);
    require(stats.n_zero_copy_hits == 0);
    require(stats.bytes_ram_to_gpu == 0);

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_buffer_free(weights_buffer);
    ggml_free(weights_ctx);
    ggml_backend_buffer_free(reference_buffer);
    ggml_free(reference_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("  cacheless MoE subset copy tests passed\n");
}
```

Do not configure or register an expert cache. Do not weaken the two exact range assertions to a total byte count.

- [ ] **Step 3: Register and run the red test**

Add to `main()` before cache-enabled tests:

```cpp
test_cacheless_moe_subset_copy();
```

Build and run:

```bash
cmake --build build --config Release --target test-expert-cache
build/bin/Release/test-expert-cache.exe
```

Expected on current Stage A production code: the first new failing assertion is `test_tracked_cpy_tensor_async_calls == 0`, because the tracked host expert tensor uses `cpy_tensor_async` once and emits no selective `set_tensor_async` ranges. The later zero-telemetry assertions are part of the green contract and are not the accepted red signal.

---

### Task 3: Restore the Upstream Cacheless Scheduler Path

**Files:**
- Modify: `ggml/src/ggml-backend.cpp:1190-1976`
- Modify: `ggml/src/ggml-backend.cpp:2275-2998`
- Test: `tests/test-expert-cache.cpp`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`
- Create results: `tools/results/expert-cache/cacheless-parity-stage-b/*.jsonl`

**Interfaces:**
- Consumes: `test_cacheless_moe_subset_copy()` from Task 2 and upstream split synchronization, selective-copy, fallback, and event logic from baseline `ggml-backend.cpp:1604-1788`.
- Produces: `static bool ggml_backend_sched_has_expert_cache(ggml_backend_sched_t sched)` and a cacheless scheduler path with zero cache planning/bookkeeping.

- [ ] **Step 1: Add one scheduler cache-presence predicate**

Near the scheduler route-planning helpers, add:

```cpp
static bool ggml_backend_sched_has_expert_cache(ggml_backend_sched_t sched) {
    for (int b = 0; b < sched->n_backends; ++b) {
        if (sched->expert_caches[b] != nullptr) {
            return true;
        }
    }
    return false;
}
```

Do not expose it in a public header.

- [ ] **Step 2: Gate graph-time cache placement and planning**

In `ggml_backend_sched_split_graph()`, compute the predicate once after the scheduler context is recreated and before backend-assignment pass 1:

```cpp
const bool has_expert_cache = ggml_backend_sched_has_expert_cache(sched);
```

The registered route-ready bundle map exists even when no cache was requested. Gate every route-ready placement override with `has_expert_cache` so cacheless backend assignment matches upstream:

```cpp
if (has_expert_cache) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        struct ggml_tensor * node = graph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] != nullptr &&
            ggml_backend_sched_has_registered_host_expert_weight(sched, node->src[0])) {
            tensor_backend_id(node) = sched->n_backends - 1;
            SET_CAUSE(node, "1.route-ready-cpu");
        }
    }
}
```

At each of the four backend-expansion checks that call `ggml_backend_sched_is_registered_host_expert_weight()`, make registration block offload only when a cache exists:

```cpp
if ((has_expert_cache &&
     ggml_backend_sched_is_registered_host_expert_weight(sched, backend_id, node->src[0])) ||
    !(sched->op_offload && ggml_backend_offload_op(sched->backends[backend_id], node))) {
    continue;
}
```

Use the local backend variable already present at each site (`cur_backend_id` or `b`); do not otherwise restructure the assignment passes.
Wrap route-plan discovery and original-graph census:

```cpp
if (has_expert_cache) {
    ggml_backend_sched_discover_route_plans(sched, graph);
    ggml_backend_sched_record_route_census(sched, graph);
}
```

After split construction:

```cpp
if (has_expert_cache) {
    ggml_backend_sched_build_route_ready_dispatches(sched);
} else {
    sched->route_plans.clear();
    sched->bundle_plans.clear();
    for (auto & dispatch : sched->route_ready_dispatches) {
        ggml_moe_route_ready_sidecar_free(dispatch.sidecar);
    }
    sched->route_ready_dispatches.clear();
}
```

This explicit cleanup covers a scheduler whose cache was disabled after an earlier graph. Keep all three plan containers empty as an internal no-cache invariant; the public test observes the corresponding zero route-plan and route-ready counters.

Migrate `test_route_census_classifies_original_graph()` and `test_route_plan_groups_shared_ids()` so their existing positive assertions run with an actual CUDA cache:

- load the CUDA backend and select its first device;
- return a reported skip when CUDA is unavailable;
- create a scheduler with `{gpu_backend, cpu_backend}`;
- call `ggml_backend_sched_set_expert_cache(sched, 1 * 1024 * 1024)` before `ggml_backend_sched_split_graph()`;
- keep the expert weight tensors in CPU host buffers and keep their existing assertions;
- free the scheduler, tensor buffer, graph context, GPU backend, and CPU backend on every non-aborting path.

The new `test_cacheless_moe_subset_copy()` remains the negative coverage for a scheduler without a cache. Do not weaken the positive route-plan assertions to make them pass cacheless.

- [ ] **Step 3: Gate compute-time cache setup**

At the start of `ggml_backend_sched_compute_splits()`:

```cpp
const bool has_expert_cache = ggml_backend_sched_has_expert_cache(sched);
```

Only when `has_expert_cache` is true:

- increment `expert_cache_route_step`;
- call `ggml_backend_sched_prefetch_carry_forward()`;
- read `GGML_EXPERT_CACHE_DEBUG_EPOCH`;
- begin cache steps and read residency epochs;
- clear cache scratch vectors and slot uses;
- scan route-census split inputs;
- reset remapped-ID buffers.

Keep the cache-enabled statements in their existing order. The no-cache path must not allocate cache scratch or touch route counters.

- [ ] **Step 4: Port the upstream selective-copy branch exactly**

Inside the non-user input branch, after waiting for the destination event and before generic `cpy_tensor_async`, execute upstream's block only when `has_expert_cache` is false:

Declare these cacheless-local values before the split loop:

```cpp
ggml_tensor * cacheless_prev_ids_tensor = nullptr;
std::vector<int32_t> cacheless_ids;
std::vector<ggml_bitset_t> cacheless_used_ids;
int cacheless_prev_backend_id = -1;
```

At the start of each split, restore upstream's cross-backend allocator-reuse guard only for the cacheless path:

```cpp
if (!has_expert_cache &&
    split->n_inputs == 0 &&
    cacheless_prev_backend_id >= 0 &&
    cacheless_prev_backend_id != split_backend_id) {
    if (sched->events[cacheless_prev_backend_id][sched->cur_copy] != nullptr) {
        ggml_backend_event_synchronize(
            sched->events[cacheless_prev_backend_id][sched->cur_copy]);
    } else {
        ggml_backend_synchronize(sched->backends[cacheless_prev_backend_id]);
    }
}
```

Inside the non-user input branch, after waiting for the destination event and before generic `cpy_tensor_async`, port this upstream block:

```cpp
ggml_tensor * node = split->graph.nodes[0];
if (!has_expert_cache &&
    split->graph.n_nodes > 0 &&
    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
    ggml_backend_buffer_is_host(input->buffer) &&
    node->src[0] == input_cpy &&
    node->op == GGML_OP_MUL_MAT_ID) {
    const int64_t n_expert = input->ne[2];
    const size_t expert_size = input->nb[2];

    ggml_backend_synchronize(input_backend);

    ggml_tensor * ids_tensor = node->src[2];
    ggml_backend_t ids_backend = split_backend;
    for (int i = input_id + 1; i < split->n_inputs; ++i) {
        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
            ids_tensor = split->inputs[i];
            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
            break;
        }
    }

    if (ids_tensor != cacheless_prev_ids_tensor) {
        cacheless_ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
        ggml_backend_tensor_get_async(
            ids_backend, ids_tensor, cacheless_ids.data(), 0, ggml_nbytes(ids_tensor));
        ggml_backend_synchronize(ids_backend);

        cacheless_used_ids.clear();
        cacheless_used_ids.resize(ggml_bitset_size(n_expert));
        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; ++i1) {
            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; ++i0) {
                const int32_t id = cacheless_ids[
                    i1 * ids_tensor->nb[1] / sizeof(int32_t) +
                    i0 * ids_tensor->nb[0] / sizeof(int32_t)];
                GGML_ASSERT(id >= 0 && id < n_expert);
                ggml_bitset_set(cacheless_used_ids.data(), id);
            }
        }
        cacheless_prev_ids_tensor = ids_tensor;
    }

    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
        const size_t expert_offset = first_id * expert_size;
        const size_t expert_size_copy = (last_id - first_id + 1) * expert_size;
        const size_t padding = std::min<size_t>(expert_size, 512);
        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;
        ggml_backend_tensor_set_async(
            split_backend,
            input_cpy,
            (const uint8_t *) input->data + expert_offset,
            expert_offset,
            expert_size_copy + padding_end);
    };

    int id = 0;
    while (!ggml_bitset_get(cacheless_used_ids.data(), id)) {
        ++id;
    }
    int32_t first_id = id;
    int32_t last_id = first_id;
    for (++id; id < n_expert; ++id) {
        if (!ggml_bitset_get(cacheless_used_ids.data(), id)) {
            continue;
        }
        if (id == last_id + 1) {
            last_id = id;
            continue;
        }
        copy_experts(first_id, last_id);
        first_id = id;
        last_id = id;
    }
    copy_experts(first_id, last_id);
} else {
    if (!split_backend->iface.cpy_tensor_async ||
        !split_backend->iface.cpy_tensor_async(
            input_backend, split_backend, input, input_cpy)) {
        if (!has_expert_cache) {
            ggml_backend_synchronize(input_backend);
            if (sched->events[split_backend_id][sched->cur_copy] != nullptr) {
                ggml_backend_event_synchronize(
                    sched->events[split_backend_id][sched->cur_copy]);
            } else {
                ggml_backend_synchronize(split_backend);
            }
        }
        ggml_backend_tensor_copy(input, input_cpy);
    }
}
```

The extra source and destination synchronization in the generic fallback is upstream behavior and applies only when `has_expert_cache` is false. Keep the current cache-enabled async-failure fallback unchanged.

Keep the route-ID copy search at `input_id + 1`. Upstream searches only later split inputs because the route-ID copy may not have been submitted yet; this deliberately differs from the cache-enabled scan.

Do not use or clear cache scratch vectors on this path. Do not generalize the predicate beyond upstream's first-node `MUL_MAT_ID` contract.

When `has_expert_cache` is true, retain the current generic input-copy and cache-node processing behavior byte-for-byte.

- [ ] **Step 5: Skip post-copy cache processing when cacheless**

Run the following only when `has_expert_cache` is true:

- the current cache tensor/node loop and every node mutation;
- remapped-ID upload and slot-use tracking;
- construction, sorting, and execution of route-ready and heterogeneous dispatches;
- `ggml_backend_sched_record_host_route_snapshots()`;
- slot-use attribution;
- node restoration;
- the end-of-call residency-epoch assertion.

When `sched->callback_eval == nullptr` and `has_expert_cache` is false, submit the complete split graph directly with `ggml_backend_graph_compute_async()`. Preserve the existing callback-driven path for both modes.

Restore upstream's cacheless split event invariant at the end of each successful split:

```cpp
if (sched->events[split_backend_id][sched->cur_copy] != nullptr &&
    (!has_expert_cache || split->n_inputs > 0)) {
    ggml_backend_event_record(
        sched->events[split_backend_id][sched->cur_copy],
        split_backend);
}
if (!has_expert_cache) {
    cacheless_prev_backend_id = split_backend_id;
}
```

Cacheless records every available split event and updates `cacheless_prev_backend_id`; cache-enabled retains the current `split->n_inputs > 0` condition. Keep graph status propagation shared. Do not alter the order or content of the cache-enabled processing body.

- [ ] **Step 6: Run the green test and focused regressions**

Run:

```bash
cmake --build build --config Release --target test-expert-cache test-backend-ops llama-cli llama-bench
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-backend-ops.exe test -b CUDA0 -o MUL_MAT_ID
```

Expected: the new range assertions pass; all existing route-ready and expert-cache tests pass; all selected CUDA operations pass.

- [ ] **Step 7: Repeat model error and coherence smokes**

Run both commands:

```bash
CUDA_LAUNCH_BLOCKING=1 build/bin/Release/llama-cli.exe -m "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" -p "Write one concise sentence about scheduler parity." -n 32 -t 14 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -fa on -lm mlock -fitt 256
build/bin/Release/llama-cli.exe -m "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" -p "Write one concise sentence about scheduler parity." -n 32 -t 14 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -fa on -lm mlock -fitt 256
```

Expected: no CUDA failure and coherent output.

- [ ] **Step 8: Run the Stage B ten-pair parity matrix**

Use a session-local Python driver with:

```python
import os
from pathlib import Path

OUTPUT_DIR = Path("tools/results/expert-cache/cacheless-parity-stage-b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BENCHES = {
    "branch": Path(r"G:/code/AI/llamacpptuned/llama.cpp/build/bin/Release/llama-bench.exe"),
    "baseline": Path(r"G:/code/AI/llama.cpp/build/bin/Release/llama-bench.exe"),
}
COMMON = [
    "-m", r"C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf",
    "-p", "0", "-n", "512", "-r", "1", "-t", "14",
    "-b", "4096", "-ub", "2048", "-ctk", "q8_0", "-ctv", "q8_0",
    "-fa", "on", "-lm", "mlock", "-fitt", "256", "-o", "jsonl",
]
ENV = {key: value for key, value in os.environ.items()
       if not key.startswith("GGML_EXPERT_CACHE_")}
ENV["GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL"] = "0"
orders = [("branch", "baseline")] * 5 + [("baseline", "branch")] * 5
```

For pair `run` 1 through 10, execute both labels in `orders[run - 1]` with `env=ENV`, require process exit code zero, parse the single JSONL row, and write it to:

```text
tools/results/expert-cache/cacheless-parity-stage-b/2026-08-29-cacheless-parity-stage-b-{branch,baseline}-{run}.jsonl
```

Expected gate: mean paired branch delta within +/-3%, paired 95% interval includes zero, 0 cache requests/actions, and 0 route-plan/census counters.

- [ ] **Step 9: Analyze and log Stage B**

Append a new section to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md` recording branch and baseline means, mean and median paired delta, range, positive pairs, sample standard deviation, 95% Student-t interval, commands, revision/build provenance, and raw paths. State whether strict parity passed. If it did not pass, stop and investigate remaining cacheless scheduler differences before changing cache-on behavior.

- [ ] **Step 10: Prepare the Stage B commit checkpoint**

Run:

```bash
git diff --check -- ggml/src/ggml-backend.cpp tests/test-expert-cache.cpp EXPERT_CACHE_OPTIMIZATIONS_LOG.md tools/results/expert-cache/cacheless-parity-stage-b
```

Request explicit user approval. Only after approval:

```bash
git add -- ggml/src/ggml-backend.cpp tests/test-expert-cache.cpp EXPERT_CACHE_OPTIMIZATIONS_LOG.md tools/results/expert-cache/cacheless-parity-stage-b
git commit -m "ggml: restore cacheless MoE copy parity"
```

---

### Task 4: Remeasure Expert Cache Against the Repaired Control

**Files:**
- Modify: `EXPERT_CACHE.md`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`
- Create results: `tools/results/expert-cache/post-parity-cache-matrix/*.jsonl`

**Interfaces:**
- Consumes: parity-passing Stage B binary and unchanged cache-enabled scheduler behavior.
- Produces: new cache-on absolute throughput and paired deltas against the repaired cache-off control.

- [ ] **Step 1: Rerun the explicit-placement retained winner**

Use `tools/results/expert-cache/run-tg-matrix.py` for ten control/cache pairs in each order, preserving fresh processes and raw rows:

```bash
python tools/results/expert-cache/run-tg-matrix.py --model "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" --output-dir tools/results/expert-cache/post-parity-cache-matrix --prefix 2026-08-29-post-parity-explicit-control-first-n512 --runs 10 --n-gen 512 --load-mode mmap --fit-target 0 --gpu-layers 99 --cpu-moe-layers 40 --cache-mib 3072 --cache-period 65536 --max-swaps 0 --pinned-experts tools/results/expert-cache/active-sidecar/pinned_layer03_all_256_3g.json
python tools/results/expert-cache/run-tg-matrix.py --model "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" --output-dir tools/results/expert-cache/post-parity-cache-matrix --prefix 2026-08-29-post-parity-explicit-cache-first-n512 --runs 10 --n-gen 512 --load-mode mmap --fit-target 0 --gpu-layers 99 --cpu-moe-layers 40 --cache-mib 3072 --cache-period 65536 --max-swaps 0 --pinned-experts tools/results/expert-cache/active-sidecar/pinned_layer03_all_256_3g.json --cache-first
```

Expected telemetry on cache rows: route-ready actions and zero-copy hits are nonzero; expert RAM-to-GPU bytes remain zero.

- [ ] **Step 2: Rerun automatic-fit dynamic 128 MiB candidates**

For periods 32 and 256, run five pairs in each order:

```bash
python tools/results/expert-cache/run-tg-matrix.py --model "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" --output-dir tools/results/expert-cache/post-parity-cache-matrix --prefix 2026-08-29-post-parity-fit256-period32-control-first-n512 --runs 5 --n-gen 512 --load-mode mmap --fit-target 256 --cache-mib 128 --cache-period 32 --max-swaps -1
python tools/results/expert-cache/run-tg-matrix.py --model "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" --output-dir tools/results/expert-cache/post-parity-cache-matrix --prefix 2026-08-29-post-parity-fit256-period32-cache-first-n512 --runs 5 --n-gen 512 --load-mode mmap --fit-target 256 --cache-mib 128 --cache-period 32 --max-swaps -1 --cache-first
python tools/results/expert-cache/run-tg-matrix.py --model "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" --output-dir tools/results/expert-cache/post-parity-cache-matrix --prefix 2026-08-29-post-parity-fit256-period256-control-first-n512 --runs 5 --n-gen 512 --load-mode mmap --fit-target 256 --cache-mib 128 --cache-period 256 --max-swaps -1
python tools/results/expert-cache/run-tg-matrix.py --model "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" --output-dir tools/results/expert-cache/post-parity-cache-matrix --prefix 2026-08-29-post-parity-fit256-period256-cache-first-n512 --runs 5 --n-gen 512 --load-mode mmap --fit-target 256 --cache-mib 128 --cache-period 256 --max-swaps -1 --cache-first
```

- [ ] **Step 3: Analyze absolute and relative effects**

For each configuration report:

- repaired cache-off mean;
- cache-on mean;
- mean and median paired delta;
- positive pairs;
- 95% Student-t interval;
- cache requests, zero-copy hits, route-ready actions, dispatches, classifications, and RAM-to-GPU bytes;
- comparison with the historical pre-parity absolute cache-on throughput.

Conclude separately:

1. whether removing CUDA serialization improved absolute cache-on throughput;
2. whether cache-on beats the repaired cache-off control;
3. whether any remaining generic full-weight copy belongs in a separate cache-owned-copy design.

- [ ] **Step 4: Update retained documentation**

Append all rows and decisions to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`. Update `EXPERT_CACHE.md` only with the retained operating point and its scope. If no cache setting beats the repaired control, state that cache remains disabled for automatic fit and preserve the explicit-placement result only if its new paired interval remains positive.

- [ ] **Step 5: Prepare the benchmark evidence commit checkpoint**

Normalize generated JSONL files to LF with one final newline, then run:

```bash
git diff --check -- EXPERT_CACHE.md EXPERT_CACHE_OPTIMIZATIONS_LOG.md tools/results/expert-cache/post-parity-cache-matrix
```

Request explicit user approval. Only after approval:

```bash
git add -- EXPERT_CACHE.md EXPERT_CACHE_OPTIMIZATIONS_LOG.md tools/results/expert-cache/post-parity-cache-matrix
git commit -m "docs: record post-parity expert cache results"
```

---

### Task 5: Full Verification and Review

**Files:**
- Verify: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Verify: `ggml/src/ggml-backend.cpp`
- Verify: `tests/test-expert-cache.cpp`
- Verify: `EXPERT_CACHE.md`
- Verify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: completed Stage A, Stage B, and post-parity benchmark evidence.
- Produces: final correctness and repository-quality evidence; no new behavior.

- [ ] **Step 1: Build all affected Release targets**

Run:

```bash
cmake --build build --config Release --target test-expert-cache test-backend-ops llama-cli llama-bench
```

Expected: successful build.

- [ ] **Step 2: Run the targeted behavioral suite**

Run:

```bash
build/bin/Release/test-expert-cache.exe
build/bin/Release/test-backend-ops.exe test -b CUDA0 -o MUL_MAT_ID
python tools/results/expert-cache/test_run_tg_matrix.py
```

Expected: all pass.

- [ ] **Step 3: Run the complete test suite**

Run:

```bash
ctest --test-dir build -C Release --output-on-failure
```

Expected: 100% pass.

- [ ] **Step 4: Review the final behavior against the specification**

Confirm from current source and raw output:

- no per-node CUDA synchronization remains;
- cacheless graph splitting performs no cache plan/census work;
- cacheless host MoE copies use routed expert ranges;
- cache-enabled route-ready tests and telemetry remain unchanged;
- cacheless parity gate passes;
- cache-on claims compare against the repaired cache-off control.

- [ ] **Step 5: Check repository hygiene**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended source, test, documentation, and retained result files are modified or untracked.

- [ ] **Step 6: Request final commit or push approval**

Do not commit or push automatically. Present exact verification output, commit list, and remaining working-tree paths. Request explicit approval for any final commit and separately for any push.
