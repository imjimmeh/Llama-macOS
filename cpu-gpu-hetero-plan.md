## Agent Implementation Instructions — Concurrent CPU/GPU Heterogeneous MoE Execution

Your next task is to implement the **event-driven concurrent heterogeneous route path** and use measured latency to determine which partial-hit masks should be admitted in production.

Do **not** change cache sizing, manifest generation, fit policy, promotion strategy, or full-hit sidecar behaviour during this work.

The current production policy is intentionally:

```text
0–6 / 8 hits -> CPU-base
7 / 8 hits   -> serial heterogeneous
8 / 8 hits   -> full-hit GPU sidecar
```

because the current serial partial-hit implementation made 1–6 hits slower than CPU-base execution. The objective is to determine whether concurrency moves that break-even point lower. The existing docs explicitly say event-driven concurrency is still staged.

---

# 1. Scope

Implement concurrent execution for TG1 partial-hit bundles only.

For:

```text
0 < n_hits < top_k
```

the desired execution is:

```text
                         ┌──────── GPU hit routes ────────┐
route IDs ready ─────────┤                                ├── merge
                         └─ CPU miss routes + result H2D ─┘
```

The critical path should approach:

```text
max(
    GPU_hit_execution,
    CPU_miss_execution + necessary transfers
)
+ merge overhead
```

rather than the current:

```text
GPU_hit_execution
+ wait
+ CPU_miss_execution
+ H2D
+ wait
+ merge
```

Do not implement PP or multi-token support as part of this change.

---

# 2. Preserve the current three-way production model

Do not delete the current policy before measurements exist.

Keep:

```cpp
if (n_hits == 0) {
    execute_cpu_base();
}
else if (n_hits == top_k) {
    execute_full_hit_sidecar();
}
else {
    // partial hit
}
```

Initially the partial branch should still only admit:

```text
7/8
```

to the new concurrent path.

Cases 1–6/8 remain CPU-base until the latency matrix proves otherwise.

That gives you a controlled comparison:

```text
existing serial 7/8
vs
new concurrent 7/8
```

before changing admission policy.

---

# 3. Do not modify the full-hit sidecar

The 8/8 sidecar is already useful and has produced a credible end-to-end speedup.

Leave its:

- persistent graph;
- slot descriptors;
- slot reservation;
- async uploads;
- graph execution;
- output handling

unchanged unless an explicit correctness bug is discovered.

Do not try to generalise the full-hit sidecar into the partial implementation during this task.

Treat 8/8 as the control for optimal cached execution.

---

# 4. Create a dedicated persistent partial-hit executor

Do not continue growing `ggml_backend_moe_hetero_execute_serial()` into the final implementation.

Introduce a distinct execution object, for example:

```cpp
struct ggml_moe_partial_executor;
```

and APIs along the lines of:

```cpp
ggml_moe_partial_executor_t
ggml_moe_partial_executor_new(
    ggml_backend_t gpu_backend,
    ggml_backend_t cpu_backend,
    int64_t d_model,
    int64_t d_ff,
    int32_t top_k,
    bool is_fused);

ggml_status
ggml_moe_partial_executor_execute(
    ggml_moe_partial_executor_t executor,
    const ggml_moe_bundle_plan * bundle,
    ggml_backend_expert_cache_t cache,
    const int32_t * route_ids,
    int32_t n_hits,
    int32_t n_misses,
    ...);
```

The existing serial implementation should remain available temporarily as a correctness/reference path.

---

# 5. Persistent allocation is mandatory

The current old heterogeneous code repeatedly created:

```text
GGML contexts
graphs
backend buffers
std::vectors
```

inside decode execution.

Do not repeat that architecture.

At executor creation time allocate everything needed for the maximum TG1 route count:

```text
top_k = 8
```

Persistent GPU-side objects should include:

```text
input tensor

hit gate IDs[8]
hit up IDs[8]
hit down IDs[8]

GPU hit output[8][d_model]

CPU-uploaded miss output[8][d_model]

hit route indices[8]
miss route indices[8]

merged output if required
```

Persistent host-side state should include:

```text
host activation x[d_model]

CPU miss output[8][d_model]

hit descriptors[8]
miss descriptors[8]

slot indices
route indices
expert IDs
```

No allocation should occur during steady-state TG execution.

Use fixed arrays where top-k is bounded:

```cpp
std::array<..., 8>
```

rather than constructing vectors each token.

---

# 6. Prebuild GPU hit graphs

You need support for:

```text
K = 1..7 GPU routes
```

Do not dynamically mutate a graph built for eight routes.

Choose one of these approaches:

### Preferred

Prebuild seven graph variants:

```text
gpu_graph[1]
gpu_graph[2]
...
gpu_graph[7]
```

Each graph has fixed dimensions matching its route count.

This is simple and safe.

### Alternative

Use a custom backend helper/kernel accepting dynamic K.

Only choose this if it is materially cleaner than seven small persistent graphs.

Do not mutate `ne[]` in already allocated graph objects per token.

---

# 7. Prebuild/reuse CPU miss execution state

Do the same for:

```text
M = 1..7 CPU misses
```

The CPU path should not perform:

```cpp
ggml_init(...)
ggml_new_graph(...)
ggml_free(...)
```

every token.

Either:

```text
prebuild CPU graph[M]
```

for M=1..7, or use a persistent CPU execution helper that consumes explicit expert slices.

The goal is:

```text
zero graph construction in decode
zero backend buffer allocation in decode
```

---

# 8. Use the existing bundle partition result once

At route-ready classification, produce one snapshot:

```cpp
struct partial_route_snapshot {
    int32_t n_hits;
    int32_t n_misses;

    route_desc hits[8];
    route_desc misses[8];
};
```

Each hit descriptor must already contain:

```text
route index
expert ID

gate slot
up slot
down slot
```

or fused equivalent.

Do not call `find_slot()` again later independently.

The partition result is the authoritative residency snapshot for this invocation.

---

# 9. Reserve hit slots until GPU completion

Immediately after partitioning:

```cpp
reserve_bundle_slots(...)
```

for every GPU route.

Those slots must not be:

```text
evicted
rebalanced
reassigned
```

until the GPU hit graph has finished consuming them.

Release them only after a completion event associated with GPU hit execution.

Do not release them merely because CPU work has finished.

---

# 10. Execution order for the concurrent path

Once route IDs and residency are known, the partial executor should perform approximately:

### Step A — prepare metadata

Populate:

```text
hit slot IDs
hit route indices

miss expert IDs
miss route indices
```

No device synchronization should be required here beyond whatever route-ID readiness guarantee already exists.

### Step B — launch GPU hit work asynchronously

Queue:

```text
input upload/copy if needed
slot ID uploads
GPU hit FFN graph
```

Do **not** synchronize.

Record:

```text
gpu_hits_done_event
```

after the GPU hit output is ready.

### Step C — CPU miss work proceeds concurrently

While GPU hit computation is running:

1. Obtain the host activation.
2. Execute exactly `n_misses` CPU expert FFNs.
3. Produce unweighted down outputs.
4. Queue H2D upload of those CPU results.

Record:

```text
cpu_results_uploaded_event
```

Do not wait for GPU hit completion before beginning CPU miss computation.

That overlap is the entire point of this task.

---

# 11. Reuse host activation when possible

Before issuing a device-to-host transfer, determine whether the bundle input already has a valid host representation.

Use:

```text
existing host copy
```

if available.

Only perform:

```text
D2H x
```

when necessary.

For TG1 the transfer is small, but the synchronization associated with obtaining it is more important than its byte count.

The API should therefore distinguish:

```cpp
host_x_available
```

from:

```cpp
host_x_needs_copy
```

rather than always forcing a tensor get.

---

# 12. If a D2H activation copy is necessary, make it asynchronous

Do not do:

```cpp
ggml_backend_tensor_get(...)
```

followed by a backend-wide synchronization.

Use:

```text
async D2H
event
```

and wait only for that specific transfer before starting CPU miss computation.

The desired timeline is:

```text
GPU hit graph ────────────────────────────────>

D2H x ──>
          CPU miss compute ───────────>
                                    H2D result ──>
```

The CPU may need to wait for `x`, but the GPU hit graph should continue independently.

---

# 13. Use real pinned host memory

The previous code labelled ordinary `malloc()` memory as pinned.

Do not repeat that.

Use the backend/CUDA-supported pinned-host allocation mechanism already present in the codebase.

Verify explicitly that the exchange buffer is page-locked.

Use it for:

```text
D2H activation
CPU miss output
H2D miss result
```

where applicable.

Add a debug assertion/log that prints whether the exchange buffer is genuinely pinned.

---

# 14. CPU miss execution must operate only on misses

For:

```text
6 hits / 2 misses
```

CPU execution must compute exactly two complete FFNs.

Not:

```text
8 CPU routes
```

and not:

```text
all gate/up plus only partial down
```

Each miss route is:

```text
gate/up or gate_up
SwiGLU
down
```

entirely on CPU.

The GPU and CPU output both represent the same semantic object:

```text
unweighted down output per route
```

---

# 15. GPU hit execution must operate only on hits

Likewise, for:

```text
6/8
```

the GPU graph must execute exactly six routes.

Telemetry must independently confirm:

```text
classified hits = 6
GPU routes executed = 6
CPU routes executed = 2
```

Do not use route classification as a proxy for actual execution.

---

# 16. Join using events, not backend-wide synchronization

After both paths have been launched:

```text
merge/scatter waits on:
    GPU hits complete
    CPU result upload complete
```

Use targeted backend events.

Do not insert:

```cpp
ggml_backend_synchronize(gpu_backend);
```

between hit execution and CPU work.

Avoid global synchronization before merge unless the backend API gives you no narrower mechanism and you have documented/proven it.

The final partial path should contain **zero unnecessary full-device synchronization calls**.

---

# 17. Scatter/merge only after both sides are ready

Inputs:

```text
gpu_hit_output[K][d_model]
cpu_miss_output[M][d_model]

hit_route_indices[K]
miss_route_indices[M]
```

Output:

```text
canonical down-route tensor[8][d_model]
```

Each original route position 0..7 must be written exactly once.

Validation in debug/tests:

```cpp
bool written[8] = {};
```

After constructing descriptors:

```text
for each hit route -> mark written
for each miss route -> mark written

assert all 8 true
assert none duplicated
```

Do this on CPU before launch, not inside the production CUDA kernel.

---

# 18. Preserve canonical route ordering

Packed hit order is not semantic route order.

Example:

```text
route positions:
0 1 2 3 4 5 6 7

hits:
0 2 5 7

misses:
1 3 4 6
```

The merged tensor must reconstruct:

```text
0 1 2 3 4 5 6 7
```

exactly.

Add permutation tests.

Do not only test:

```text
first K = hits
remaining = misses
```

---

# 19. Preserve existing router weighting

The partial executor must produce:

```text
unweighted down output per routed expert
```

and let the existing model graph perform router weighting/reduction.

Do not add route probability multiplication into the executor during this work.

That keeps the numerical surface area small.

---

# 20. Keep the current cache-off/CPU-base path untouched

This work must not alter the normal fallback path.

If the partial executor rejects a bundle for any reason:

```text
unsupported type
unexpected shape
missing slot
event failure
allocation/init failure
```

return:

```text
NOT_ADMITTED
```

and use the existing CPU-base/original execution path.

Do not improvise a third fallback path.

---

# 21. Add a runtime feature gate during development

Add an explicit toggle such as:

```text
GGML_EXPERT_CACHE_HETERO_CONCURRENT=1
```

or equivalent internal switch.

Default it off initially.

This allows clean A/B testing between:

```text
existing serial 7/8
new concurrent 7/8
```

without modifying cache placement.

After verification, concurrent can replace serial.

---

# 22. Add timing telemetry

For every admitted partial bundle, collect:

```text
partition_us

activation_d2h_us

gpu_hit_submit_us
gpu_hit_elapsed_us

cpu_miss_compute_us

cpu_result_h2d_us

join_wait_gpu_us
join_wait_cpu_us

scatter_us

total_partial_us
```

Also counters:

```text
partial_exec_1_hit
...
partial_exec_7_hit

GPU_routes_executed
CPU_routes_executed

activation_d2h_bytes
CPU_result_h2d_bytes

weight_h2d_bytes
```

Hard invariant:

```text
weight_h2d_bytes == 0
```

for every partial execution.

---

# 23. Measure serial and concurrent critical paths directly

For each hit mask:

```text
1/8
2/8
...
7/8
```

record:

```text
CPU-base latency
serial heterogeneous latency
concurrent heterogeneous latency
```

Use the same exact input/expert route set.

The important comparison is not just:

```text
concurrent < serial
```

but:

```text
concurrent < CPU-base
```

because that determines production admission.

---

# 24. Mandatory hit-mask latency experiment

Run at least:

```text
100 warmups
1000 timed iterations
```

per mask in the isolated partial-hit benchmark.

Produce:

| GPU hits | CPU misses | CPU-base median µs | Serial median µs | Concurrent median µs | Best |
| -------: | ---------: | -----------------: | ---------------: | -------------------: | ---- |
|        1 |          7 |                    |                  |                      |      |
|        2 |          6 |                    |                  |                      |      |
|        3 |          5 |                    |                  |                      |      |
|        4 |          4 |                    |                  |                      |      |
|        5 |          3 |                    |                  |                      |      |
|        6 |          2 |                    |                  |                      |      |
|        7 |          1 |                    |                  |                      |      |

Also record P95.

Do not choose the admission threshold from mean alone.

---

# 25. Determine the new admission threshold from evidence

Once the table exists, production admission should be derived from the lowest hit count for which concurrent execution is robustly faster.

For example, if measurements show:

```text
4/8 concurrent: slower than CPU
5/8 concurrent: 2% faster but noisy
6/8 concurrent: 12% faster
7/8 concurrent: 25% faster
```

then use:

```text
admit >= 6/8
```

not 5/8.

Require a meaningful margin, ideally at least ~5%, to avoid bouncing around measurement noise.

Do not automatically enable all 1–7 masks.

---

# 26. Add route-mask permutation coverage

For each useful K, test multiple hit positions.

Especially:

```text
7/8:
miss route 0
miss route 7

6/8:
miss [0,1]
miss [3,7]

4/8:
GGGGCCCC
GCGCGCGC
CCGGGGCC
```

Numerical output must remain equivalent regardless of packed route ordering.

---

# 27. Compare against the canonical backend result

Use the existing normal model/backend output as the reference.

Do not validate only against separately coded mathematical FFN logic.

For each mask compare:

```text
partial executor output
vs
ordinary bundle output
```

with backend/type-appropriate numerical tolerance.

Then compare final layer output.

---

# 28. Run deterministic real-model generation after every admission change

For each proposed admission threshold:

```text
cache off
cache on
temperature 0
top-k 1
same prompt
same seed
256+ generated tokens
```

Record token hashes.

At minimum ensure:

```text
coherent output
no NaN/Inf
no CUDA errors
```

Where the execution path should be numerically identical, require exact hash equality.

Where GPU/CPU arithmetic legitimately differs, inspect first-token divergence and logit margins rather than blindly accepting arbitrary sequence differences.

---

# 29. Real-model telemetry must prove concurrency is being reached

Run the actual Compact model and collect:

```text
mask histogram
admitted mask histogram

GPU route count
CPU route count

partial executor invocation count
```

You should be able to show something like:

```text
6/8 classified: 90
6/8 admitted:   90

GPU routes executed: +540
CPU routes executed: +180
```

Do not infer production benefit from synthetic tests alone.

---

# 30. Compare end-to-end TG after the microbenchmark passes

Use the existing fresh-process alternating matrix.

Recommended initial comparison:

```text
A: current production policy
   7/8 serial
   8/8 sidecar

B: same placement/cache
   7/8 concurrent
   8/8 sidecar
```

Then, if concurrent 6/8 is proven useful:

```text
C:
6–7/8 concurrent
8/8 sidecar
```

Do not change `-exc`, manifest, fit target or rebalance period between A/B/C.

---

# 31. Do not use the 1024 MiB auto-fit profile to judge this work

The latest 1024 MiB v2 automatic-fit result is dominated by the VRAM placement penalty:

```text
~ -7%
```

because cache reservation displaces complete GPU layers.

That would obscure the effect of concurrency.

Use either:

- the current 128 MiB dynamic configuration; or
- explicit placement with identical model-layer residency.

The experiment must isolate execution policy.

---

# 32. Recommended primary production benchmark

Use the known useful dynamic configuration:

```text
-fitt 256
-exc 128
-excp 32
```

with the exact same model placement between comparison runs.

This configuration already has a credible paired positive result and therefore gives the new executor a meaningful workload.

---

# 33. Do not change dynamic promotion while benchmarking concurrency

Freeze all unrelated behavior.

If necessary use:

```text
static seeded residency
```

for the isolated mask test.

For the production dynamic test, keep:

```text
same rebalance period
same max swaps
same fit configuration
same prompt/workload
```

Only the serial-vs-concurrent execution policy changes.

---

# 34. Handle failures conservatively

If:

```text
GPU graph submission fails
event creation fails
CPU execution fails
slot reservation becomes invalid
scatter fails
```

then:

1. synchronize only what is necessary to make pending work safe;
2. release reserved slots;
3. mark the partial attempt failed;
4. fall back to the ordinary bundle path on the next invocation.

Do not attempt partial recovery inside a half-executed bundle.

---

# 35. Do not regress the known fallback ordering fix

The project previously produced corrupt generation because `down` was executed before its activation existed.

Preserve this invariant:

> No node or sidecar computation may consume an activation before the producer has completed.

The stale-activation failure and its fix are documented in the current handover.

Add an assertion/test specifically protecting that ordering.

---

# 36. Investigate the intermittent cross-split numeric test before widening production admission

There is still a documented intermittent failure in the synthetic cross-split sidecar test.

Before enabling 5/8 or 6/8 production execution, reproduce and understand that failure.

At minimum determine whether it is:

```text
test construction issue
slot lifetime issue
stream/event ordering issue
tensor-layout issue
actual sidecar arithmetic discrepancy
```

Do not widen heterogeneous execution while an unresolved cross-split correctness fault may share the same infrastructure.

---

# 37. Suggested commit sequence

Keep this work decomposed:

```text
1. test: add serial-vs-concurrent partial-hit timing harness

2. expert-cache: add persistent partial executor state

3. expert-cache: prebuild K=1..7 GPU hit graphs

4. expert-cache: prebuild/reuse M=1..7 CPU miss execution

5. expert-cache: add async activation/result transfer events

6. expert-cache: overlap GPU hits with CPU misses

7. expert-cache: add event-driven scatter join

8. test: verify route permutations and numerical parity

9. bench: measure 1/8..7/8 break-even curve

10. expert-cache: update production admission threshold

11. docs: record measured threshold and results
```

Do not bundle all of this into one giant commit.

---

# 38. Acceptance criteria

Do not mark concurrent heterogeneous execution complete until all of these pass:

1. No per-token GGML context creation in the concurrent path.
2. No per-token backend buffer allocation/free.
3. No per-token heap-vector construction in the executor hot path.
4. 7/8 executes exactly:

    ```text
    7 GPU routes
    1 CPU route
    ```

5. GPU and CPU work overlap in the timeline/profiler.
6. There is no backend-wide synchronization between GPU hit launch and CPU miss execution.
7. Current-token expert-weight H2D remains exactly zero.
8. Route output matches canonical execution within justified tolerance.
9. Cross-split real-model generation remains coherent.
10. Partial mask latency table exists for 1/8 through 7/8.
11. Production threshold is chosen from measured break-even data.
12. Whole-model TG matrix shows no regression versus the current policy.
13. `EXPERT_CACHE.md` and optimisation log accurately describe which masks are admitted and why.

---

## Target result

The immediate success case is not “make every partial hit use the GPU”.

It is:

> **Make 7/8 materially faster by hiding the one CPU miss under GPU execution, then determine experimentally whether 6/8, 5/8, or lower masks cross the CPU-base break-even threshold.**

A successful implementation should turn the current admission rule from an arbitrary/static:

```text
>= 7 hits
```

into a measured rule such as:

```text
>= 5 hits
```

or:

```text
>= 6 hits
```

while preserving 8/8 full-hit sidecar execution and leaving lower-value masks on the CPU-base path.
