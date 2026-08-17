Yes. I went through current llama.cpp master and I think there is a reasonably clean way to build this without turning it into a Qwen-specific hack.

I based this on current master as of **17 August 2026**, around commit `805984d676d001b194b844e1573c245edb4dbb34`. The most important discovery is that llama.cpp already has several pieces we can exploit: dense models converge on the shared `llm_graph_context::build_ffn()` path; `--fit` already understands **partial layers at whole-tensor granularity** (`attention → up → gate → down`); tensor buffer overrides already exist; and the backend scheduler already supports graphs split across different backends and submits backend graph work asynchronously. ([GitHub][1])

What it **doesn't** currently have is a way for one dense FFN tensor to be partly persistent in CPU RAM and partly persistent in VRAM while evaluating the two pieces independently.

## What we're actually building

Call it, provisionally:

```text
Dense FFN heterogeneous partitioning
```

For a SwiGLU FFN:

```text
              gate
          ┌──────────┐
x ───────►│ W_gate   ├── SiLU ─┐
          └──────────┘         │
                               × ─────► W_down ─────► y
          ┌──────────┐         │
x ───────►│ W_up     ├─────────┘
          └──────────┘
```

we partition the **intermediate FFN dimension**, `n_ff`:

```text
channels [0 ........ G)        GPU
channels [G ........ n_ff)     CPU
```

Mathematically:

```text
a_gpu = SiLU(Wgate_gpu x) * (Wup_gpu x)
a_cpu = SiLU(Wgate_cpu x) * (Wup_cpu x)

y_gpu = Wdown_gpu a_gpu
y_cpu = Wdown_cpu a_cpu

y = y_gpu + y_cpu
```

This is **exact inference** apart from ordinary floating-point accumulation-order differences. We're not dropping neurons.

Qwen3, for example, creates dense FFN tensors as:

```cpp
ffn_gate = { n_embd, n_ff }
ffn_up   = { n_embd, n_ff }
ffn_down = { n_ff,   n_embd }
```

and feeds them through the common `build_ffn()` path, exactly matching the partition above. Qwen3.5 dense uses that same common FFN builder, including optional quantization/scale tensors. ([GitHub][2])

The target graph becomes:

```text
                        ┌──── CUDA ──────────────────┐
                        │ up_gpu                     │
                  ┌────►│ gate_gpu → activation     │
                  │     │          → down_gpu ──────┼──┐
                  │     └────────────────────────────┘  │
input ────────────┤                                     ├── ADD → output
                  │     ┌──── CPU ───────────────────┐  │
                  └────►│ up_cpu                     │  │
                        │ gate_cpu → activation      │  │
                        │          → down_cpu ───────┼──┘
                        └─────────────────────────────┘
```

The desired runtime is:

```text
CUDA:  ████████████
CPU:   ████████████
       ^ concurrently

merge:             █
```

rather than merely shifting some work from CPU to GPU and running it sequentially.

---

# Stage 0 — build a timing-only proof before touching storage

I would do this first because **CPU/GPU concurrency is the entire premise**.

The scheduler currently constructs graph splits by backend and calls `ggml_backend_graph_compute_async()` for each split. That gives us a plausible route to overlap GPU execution with CPU execution, but we need to verify actual behavior, particularly because the CPU backend may effectively execute its "async" call synchronously.

Add temporary instrumentation around:

```text
ggml/src/ggml-backend.cpp

ggml_backend_sched_compute_splits()
```

Specifically around:

```cpp
ggml_backend_graph_compute_async(split_backend, &split->graph);
```

Record:

```cpp
struct split_trace {
    int split_id;
    const char * backend_name;
    int64_t begin_us;
    int64_t return_us;
};
```

Then use Nsight Systems for CUDA timeline validation.

We need to establish:

```text
1. CUDA split is submitted.
2. function returns before CUDA completes.
3. CPU split executes while CUDA is still running.
4. eventual dependency causes correct synchronization before merge.
```

If this works, **do not modify the scheduler initially**.

If CPU execution blocks submission because CPU appears first in graph order, deliberately build the GPU branch first in `build_ffn_split()`. CUDA submission returns quickly; then CPU work runs while CUDA is executing.

This would be my first test.

---

# Stage 1 — introduce a generic FFN partition description

I would **not** make this a new `llama_split_mode`.

Existing:

```text
LLAMA_SPLIT_MODE_LAYER
LLAMA_SPLIT_MODE_ROW
LLAMA_SPLIT_MODE_TENSOR
```

describe model/multi-GPU distribution. Current CLI documents row/tensor splits primarily as multiple-GPU strategies. ([GitHub][3])

This feature is orthogonal.

Add something approximately like this to:

```text
src/llama-model.h
```

```cpp
struct llama_ffn_partition {
    // 0 = disabled
    int64_t n_ff_accel = 0;

    // persistent accelerator copies
    ggml_tensor * up_accel   = nullptr;
    ggml_tensor * gate_accel = nullptr;
    ggml_tensor * down_accel = nullptr;

    // optional associated scale tensors
    ggml_tensor * up_s_accel   = nullptr;
    ggml_tensor * gate_s_accel = nullptr;
    ggml_tensor * down_s_accel = nullptr;

    // device containing accelerator partition
    ggml_backend_dev_t dev = nullptr;
};
```

Then extend the generic layer representation, wherever the current `llama_layer` FFN tensor members live:

```cpp
struct llama_layer {
    ...

    ggml_tensor * ffn_up;
    ggml_tensor * ffn_gate;
    ggml_tensor * ffn_down;

    ...

    std::unique_ptr<llama_ffn_partition> ffn_partition;
};
```

### Why put this on `llama_layer`?

Because the existing architecture-specific files already populate:

```cpp
layer.ffn_up
layer.ffn_gate
layer.ffn_down
```

with normal tensors. For example, both current Qwen and several hybrid architectures use these common layer members. ([GitHub][4])

The partition becomes metadata attached to the layer, rather than something architectures need to understand.

---

# Stage 2 — add a central partition builder to model loading

This is the hardest part.

I would add:

```text
src/llama-model-partition.h
src/llama-model-partition.cpp
```

with an API roughly like:

```cpp
bool llama_build_ffn_partition(
    llama_model & model,
    llama_layer & layer,
    int il,
    ggml_backend_dev_t accel_dev,
    float fraction);
```

It receives:

```text
original full CPU tensors
        +
desired fraction
        +
destination backend
```

and generates persistent accelerator tensors.

### Channel count calculation

Don't allow arbitrary channel boundaries.

Quantized GGML tensors are block quantized, so select a partition aligned to the relevant quantization block size.

Conceptually:

```cpp
int64_t n_ff = layer.ffn_up->ne[1];

int64_t n_ff_gpu =
    align_down(
        int64_t(n_ff * fraction),
        required_quant_alignment(layer));
```

For MVP I'd use a conservative alignment:

```text
max block-size required by:
    ffn_up
    ffn_gate
    ffn_down
```

and reject pathological combinations.

---

# Stage 3 — create the actual split tensors

This is where an important subtlety arises.

Given:

```text
up   = [n_embd, n_ff]
gate = [n_embd, n_ff]
down = [n_ff, n_embd]
```

the GPU prefix behaves differently for `up/gate` versus `down`.

### `up` and `gate`

We're selecting rows:

```text
row 0
row 1
...
row G-1
```

which are contiguous in GGML's normal layout.

So:

```cpp
up_gpu = new_tensor_2d(
    type(up),
    n_embd,
    n_ff_gpu);

gate_gpu = new_tensor_2d(
    type(gate),
    n_embd,
    n_ff_gpu);
```

And the source range is basically:

```cpp
0 .. n_ff_gpu * up->nb[1]
```

Easy.

### `down`

Its shape is:

```text
[n_ff, n_embd]
```

We're selecting the **first `n_ff_gpu` elements of every row**.

So source layout is:

```text
row 0: [ GPU | CPU ]
row 1: [ GPU | CPU ]
row 2: [ GPU | CPU ]
...
```

That is not one contiguous block.

Therefore `down_gpu` must be **repacked once during loading**:

```cpp
down_gpu = new_tensor_2d(
    down->type,
    n_ff_gpu,
    n_embd);
```

Then:

```cpp
for (int64_t row = 0; row < n_embd; ++row) {
    copy_quantized_range(
        src = down + row * down->nb[1],
        dst = down_gpu + row * down_gpu->nb[1],
        n_elements = n_ff_gpu);
}
```

No dequantize/requantize should occur.

Copy complete quantization blocks verbatim.

That is why the split boundary must be block aligned.

---

# Stage 4 — don't duplicate the CPU portion initially

The original GGUF tensors can remain entirely in RAM:

```text
RAM:
    full ffn_up
    full ffn_gate
    full ffn_down

VRAM:
    GPU prefix ffn_up
    GPU prefix ffn_gate
    packed GPU prefix ffn_down
```

That means system RAM consumption does **not** improve.

That's fine for MVP.

CPU computation can use views representing the complement.

For up/gate:

```cpp
ggml_tensor * up_cpu =
    ggml_view_2d(
        ctx0,
        up,
        n_embd,
        n_ff - n_ff_gpu,
        up->nb[1],
        n_ff_gpu * up->nb[1]);
```

Same for gate.

`down` is slightly trickier:

```cpp
ggml_tensor * down_cpu =
    ggml_view_2d(
        ctx0,
        down,
        n_ff - n_ff_gpu,
        n_embd,
        down->nb[1],
        byte_offset_for_channel(n_ff_gpu));
```

The stride remains the original full row stride.

Before relying on this, add a backend capability test because some quantized matmul implementations may reject the resulting non-contiguous weight tensor.

If CPU supports it:

```text
great
```

If not:

```text
also create packed down_cpu
```

at load time.

For a first implementation, frankly, I would probably **pack both down partitions**. It wastes some extra host memory but eliminates an entire class of weird quantized-stride bugs.

---

# Stage 5 — add a new shared graph builder

This is the most important code change.

Current dense architectures call:

```cpp
build_ffn(
    cur,
    layer.ffn_up,   ...,
    layer.ffn_gate, ...,
    layer.ffn_down, ...,
    ...,
    LLM_FFN_SILU,
    LLM_FFN_PAR,
    il);
```

Qwen3, Qwen3.5 dense, dense DeepSeek lead layers, Laguna dense lead layers and others all converge through that mechanism. ([GitHub][5])

Current `build_ffn()` already handles parallel gate/up calculation, activation and final down projection centrally.

Don't duplicate all that logic.

Refactor first.

## 5a. Extract `build_ffn_impl`

Change:

```cpp
ggml_tensor * llm_graph_context::build_ffn(...)
```

to essentially:

```cpp
ggml_tensor * llm_graph_context::build_ffn_impl(
    ggml_tensor * cur,
    ggml_tensor * up,
    ggml_tensor * up_b,
    ggml_tensor * up_s,
    ggml_tensor * gate,
    ggml_tensor * gate_b,
    ggml_tensor * gate_s,
    ggml_tensor * down,
    ggml_tensor * down_b,
    ggml_tensor * down_s,
    ggml_tensor * act_scales,
    llm_ffn_op_type type_op,
    llm_ffn_gate_type type_gate,
    int il,
    const char * cb_suffix);
```

Existing `build_ffn()` simply calls this unchanged.

That guarantees baseline output remains identical.

## 5b. Add:

```cpp
ggml_tensor * llm_graph_context::build_ffn_partitioned(
    ggml_tensor * cur,
    const llama_ffn_weights & full,
    const llama_ffn_partition & part,
    ...);
```

Conceptually:

```cpp
auto * gpu = build_ffn_impl(
    cur,
    part.up_accel,
    ...,
    part.gate_accel,
    ...,
    part.down_accel,
    ...,
    type_op,
    type_gate,
    il,
    "_gpu");

auto * cpu = build_ffn_impl(
    cur,
    up_cpu_view,
    ...,
    gate_cpu_view,
    ...,
    down_cpu_view,
    ...,
    type_op,
    type_gate,
    il,
    "_cpu");

auto * out = ggml_add(ctx0, gpu, cpu);

cb(out, "ffn_split_out", il);

return out;
```

The graph itself then tells the scheduler:

```text
GPU weight → accelerator branch
CPU weight → CPU branch
```

instead of embedding backend-specific execution code in model architecture implementations.

---

# Stage 6 — make `build_ffn()` automatically use the partition

This is where architecture agnosticism matters.

I **wouldn't** add this to every:

```text
qwen.cpp
llama.cpp
gemma.cpp
...
```

Instead, give `llm_graph_context` access to the partition metadata.

Currently `llm_graph_params` already carries common graph information such as architecture, hyperparameters and context parameters, and `build_ffn()` is centralized on `llm_graph_context`.

Add something like:

```cpp
struct llm_graph_params {
    ...

    const llama_model * model = nullptr;
};
```

or, preferably narrower:

```cpp
const llama_ffn_partition_registry * ffn_partitions = nullptr;
```

Registry:

```cpp
using llama_ffn_partition_registry =
    std::unordered_map<
        const ggml_tensor *,
        const llama_ffn_partition *>;
```

Key it by the **original `ffn_up` tensor pointer**.

Then inside existing `build_ffn()`:

```cpp
const llama_ffn_partition * part =
    ffn_partitions
        ? ffn_partitions->find(up)
        : nullptr;

if (part && part->n_ff_accel > 0) {
    return build_ffn_partitioned(...);
}

return build_ffn_impl(...);
```

That's the key architectural choice.

Now every architecture already using `build_ffn()` gets the feature automatically.

No:

```cpp
if (arch == QWEN)
if (arch == LLAMA)
if (arch == GEMMA)
```

anywhere.

---

# Stage 7 — backend placement

The GPU split tensors need to live in a GPU weight buffer.

Current llama.cpp model parameters already support explicit tensor→buffer-type placement via `llama_model_tensor_buft_override`, and the loader builds device/buffer lists for CPU, host and accelerators. ([GitHub][6])

For derived tensors I wouldn't abuse regex tensor overrides.

Instead, when creating the partition:

```cpp
ggml_backend_buffer_type_t accel_buft =
    ggml_backend_dev_buffer_type(accel_dev);
```

allocate a dedicated model-owned buffer:

```cpp
ggml_backend_buffer_t ffn_split_buffer;
```

Potentially one buffer per accelerator, containing all derived slices:

```text
CUDA0 dense partition buffer

layer 0 up
layer 0 gate
layer 0 down
layer 1 up
layer 1 gate
layer 1 down
...
```

**Do not allocate one CUDA buffer per tensor.**

That would give dozens/hundreds of small backend allocations.

Create metadata tensors first, measure total required size, then allocate one buffer and assign offsets.

Something roughly:

```cpp
struct llama_ffn_partition_buffer {
    ggml_backend_dev_t dev;
    ggml_backend_buffer_t buffer;

    std::unique_ptr<ggml_context, ...> ctx;

    size_t size;
};
```

Model owns these for its lifetime.

---

# Stage 8 — the graph/backend callback needs to respect the derived tensors

llama.cpp uses graph callbacks to influence allocation/offloading, and the graph parameter callback is explicitly described as the place for custom tensor logic.

Derived split tensors already have backing buffers, so the scheduler should identify their owning backend correctly.

I'd still explicitly name them:

```text
blk.12.ffn_up.hsplit_gpu
blk.12.ffn_gate.hsplit_gpu
blk.12.ffn_down.hsplit_gpu

blk.12.ffn_up.hsplit_cpu
...
```

Debugging graph dumps will otherwise become miserable.

---

# Stage 9 — CLI/API configuration

For the initial experimental branch, use **one parameter**:

```bash
--ffn-split 0.35
```

meaning:

> Place 35% of every eligible dense FFN intermediate dimension on the accelerator; compute the remainder on CPU.

Add to:

```text
common/common.h
```

roughly:

```cpp
float ffn_split = 0.0f;
```

Current common model/offload parameters such as `n_gpu_layers`, tensor split and fit settings already live there. ([GitHub][7])

Add parsing in:

```text
common/arg.cpp
```

near:

```text
--n-gpu-layers
--split-mode
--tensor-split
```

Current command-line argument registration lives there. ([GitHub][8])

I'd use:

```text
--ffn-split P
LLAMA_ARG_FFN_SPLIT
```

with:

```text
0        disabled
0.0–1.0  explicit fraction
auto     later
```

For the first implementation, **do not add `auto` yet**.

---

# Stage 10 — model parameter plumbing

Eventually put it into `llama_model_params`, because it changes persistent model weight allocation:

```cpp
struct llama_model_params {
    ...

    float ffn_split;
    int32_t ffn_split_device;
};
```

Current `llama_model_params` is already where tensor buffer overrides, `n_gpu_layers`, split mode, main GPU and tensor split are represented. ([GitHub][6])

For an experimental branch, however, I'd initially keep it internal to common params rather than immediately altering the public ABI.

Once proven:

```cpp
params.ffn_split = ...
```

becomes public.

---

# Stage 11 — eligibility detection

Do **not** blindly split every call to `build_ffn()`.

Start with the safe/common case:

```cpp
bool can_partition_ffn(
    const ggml_tensor * up,
    const ggml_tensor * gate,
    const ggml_tensor * down,
    llm_ffn_op_type type_op,
    llm_ffn_gate_type type_gate) {

    if (!up || !down) {
        return false;
    }

    if (type_gate != LLM_FFN_PAR) {
        return false;
    }

    if (gate == nullptr) {
        return false;
    }

    if (up->ne[1] != gate->ne[1]) {
        return false;
    }

    if (down->ne[0] != up->ne[1]) {
        return false;
    }

    return true;
}
```

This catches the canonical:

```text
SwiGLU / GEGLU
gate + up → elementwise activation → down
```

I'd initially support:

```text
LLM_FFN_SILU
LLM_FFN_GELU / GEGLU where mathematically compatible
LLM_FFN_RELU variants
```

but develop and benchmark SILU first.

Current `build_ffn()` has multiple activation/gating variants and should remain the source of truth rather than copying their implementations.

---

# Stage 12 — deal with scale tensors properly

Qwen3.5 is one reason not to ignore this.

Its calls can include:

```text
ffn_up_s
ffn_gate_s
ffn_down_s
```

rather than passing null scales. ([GitHub][9])

So the partition builder needs a generic scale-partition function:

```cpp
partition_aux_tensor(
    original_scale,
    partition_axis,
    n_ff_gpu);
```

Exactly how that slice works depends on the quantization format represented by the scale tensor.

I'd make this capability-driven:

```cpp
bool llama_can_partition_tensor(
    const ggml_tensor * weight,
    const ggml_tensor * scale,
    llama_partition_axis axis);
```

Unsupported format:

```text
log warning
do not split this FFN
```

rather than risking incorrect output.

That lets support grow incrementally.

---

# Stage 13 — force CPU and GPU branches onto the correct backends

The scheduler generally infers execution backend from weight placement, which is useful here.

But for this experiment I would make it explicit through the graph callback:

```text
*_hsplit_gpu nodes → partition GPU backend
*_hsplit_cpu nodes → CPU backend
```

This avoids a situation where a tiny decode matmul gets migrated back to CPU because a backend heuristic thinks that's cheaper.

We've already seen why this matters in the MoE case: simply putting weights somewhere doesn't necessarily guarantee the operation executes where you intend.

So add a tensor flag or recognisable callback identity.

I prefer **a proper flag**, not parsing names.

For example in GGML:

```cpp
enum ggml_tensor_flag {
    ...
    GGML_TENSOR_FLAG_FORCE_CPU,
    GGML_TENSOR_FLAG_FORCE_ACCEL,
};
```

although upstream may understandably dislike that API.

For an experimental branch, the callback can simply keep a set:

```cpp
std::unordered_set<const ggml_tensor *> force_cpu;
std::unordered_set<const ggml_tensor *> force_accel;
```

and assign backend there.

Cleaner.

---

# Stage 14 — ensure the GPU branch is submitted first

This could make or break performance.

Current scheduler walks graph splits in order and invokes `ggml_backend_graph_compute_async()` on each.

Build:

```cpp
gpu = build_ffn_impl(...);
cpu = build_ffn_impl(...);
out = ggml_add(...);
```

rather than:

```cpp
cpu = ...
gpu = ...
```

because the desired timeline is:

```text
thread:
    submit CUDA
        ↓
    CUDA starts asynchronously

thread:
    run CPU

CUDA: █████████████
CPU:    ███████████
```

If CPU comes first and its backend call blocks:

```text
CPU:  ███████████
CUDA:            █████████████
```

and we've gained very little.

**This should be explicitly tested in Stage 0.**

---

# Stage 15 — choose where the final add happens

We end with:

```text
y_gpu [n_embd × tokens]
y_cpu [n_embd × tokens]
```

Those are tiny compared with the FFN weights.

For decode, we're talking on the order of `n_embd` floats per token, not hundreds of megabytes of weights.

I would place the final:

```cpp
ggml_add(y_gpu, y_cpu)
```

on whichever backend owns the next major operation.

Probably GPU for GPU-resident next-layer attention.

So:

```text
CPU FFN result
        │
        │ only n_embd activations
        ▼
GPU ── ADD ── next layer
```

That is vastly preferable to transferring weights.

---

# Stage 16 — first version should use a uniform prefix, NOT activation profiling

Earlier we discussed profiling "hot neurons".

I would **not do that initially**.

Use:

```text
GPU = channels [0, N)
CPU = channels [N, n_ff)
```

Why?

Because every channel gets computed anyway.

Unlike MoE:

```text
hot expert → caching avoids CPU work
```

dense:

```text
hot channel → CPU still has to compute all cold channels
```

So the first-order optimisation is **load balancing**, not semantic importance.

Prefix slicing also gives us:

```text
simple quant block boundaries
simple up/gate copies
simple deterministic layout
minimal lookup overhead
```

Once it works, arbitrary channel ordering can be V2.

---

# Stage 17 — auto balancing

This is where I think the feature becomes genuinely good.

After explicit fractions work:

```bash
--ffn-split 0.25
--ffn-split 0.35
--ffn-split 0.45
```

add:

```bash
--ffn-split auto
```

The objective is not:

```text
fill VRAM
```

It's:

```text
T_layer ≈ max(T_cpu_partition, T_gpu_partition)
```

subject to:

```text
VRAM capacity
```

So at startup benchmark representative FFN shapes:

```text
CPU Q4_K matmul throughput
CUDA Q4_K matmul throughput
CPU activation
CUDA activation
PCIe output-transfer time
```

Then solve approximately:

```text
fraction_gpu =
    cpu_rate / (cpu_rate + gpu_rate)
```

adjusted for transfer and kernel overhead.

For example, if effective decode throughput says:

```text
CPU can process 35% of FFN channels
in the time GPU processes 65%
```

we choose:

```text
GPU 65%
CPU 35%
```

rather than "whatever fits".

---

# Stage 18 — integrate with `--fit`

Current `--fit` is already more sophisticated than ordinary `n_gpu_layers`.

It explicitly models partial layers using:

```cpp
LAYER_FRACTION_ATTN
LAYER_FRACTION_UP
LAYER_FRACTION_GATE
LAYER_FRACTION_MOE
```

and uses tensor buffer overrides to spill selected components when a complete layer doesn't fit. ([GitHub][1])

That gives us a very natural extension.

Eventually add:

```cpp
LAYER_FRACTION_FFN_SPLIT
```

but unlike the existing enum entries, it carries a **continuous fraction** rather than just a discrete tensor group.

I'd introduce:

```cpp
struct common_layer_fit {
    common_layer_fraction_t type;

    float ffn_split = 0.0f;
};
```

Then fit can ask:

```text
VRAM remaining = 726 MiB

full next layer = 1.1 GiB
full FFN = 820 MiB

how much of its FFN could be persistent?

→ perhaps 31%
```

This is much better than today's boundary:

```text
whole gate fits
down doesn't
```

Current fit already uses regex tensor overrides to put whole FFN components of partially fitting layers onto different buffer types; our partitioning makes that final step continuous rather than discrete. ([GitHub][1])

---

# Stage 19 — but eventually I would change the philosophy of `--fit`

Once heterogeneous FFNs exist, the goal should arguably become:

```text
1. Put latency-critical / cheap tensors permanently on GPU.
2. Allocate remaining VRAM to FFN partitions across MANY CPU layers.
3. Balance CPU/GPU execution.
```

rather than:

```text
GPU:
    last N whole layers

CPU:
    first M whole layers
```

For an 8 GB card I'd explicitly test:

### Current

```text
CPU: layers 0-20
GPU: layers 21-31
```

against:

### Hybrid

```text
all layers:

attention    GPU where possible
FFN          40% GPU / 60% CPU
```

and:

### Hybrid mixed

```text
some complete GPU layers

remaining CPU layers:
    FFN 20–40% GPU
```

The third is probably the realistic winner.

---

# Stage 20 — testing correctness

Before benchmarking speed, compare logits.

For each supported quant type/model:

```text
baseline
vs
ffn-split=0.25
vs
ffn-split=0.50
vs
ffn-split=0.75
```

Record:

```text
max absolute logit error
mean absolute logit error
top-1 token differences
generation divergence point
```

For F32/F16:

```text
expect only normal accumulation differences
```

For quantized tensors, we're copying quantized blocks unchanged, so the main numerical difference is computation/reduction ordering, **not re-quantization**.

Do not accept a code path that:

```text
dequantizes → partitions → requantizes
```

during loading.

---

# Stage 21 — performance telemetry

Add counters to the partition subsystem:

```cpp
struct llama_ffn_partition_stats {
    uint64_t n_calls;

    uint64_t gpu_us;
    uint64_t cpu_us;
    uint64_t join_wait_us;

    uint64_t cpu_result_bytes_transferred;

    uint64_t gpu_channels;
    uint64_t cpu_channels;
};
```

At shutdown or with verbose logging:

```text
dense FFN split:
  layers partitioned:       24
  GPU FFN fraction:         38.2%
  partition VRAM:           1816 MiB

  decode:
    CPU branch avg:         3.92 ms
    GPU branch avg:         3.51 ms
    join wait avg:          0.46 ms

  overlap efficiency:       89.1%
```

The most useful metric is:

```text
join_wait
```

If:

```text
CPU 4ms
GPU 1ms
```

we're badly balanced.

If:

```text
CPU 4ms
GPU 3.8ms
```

we're close.

---

# I would implement this in these commits

If I were handing this directly to a coding agent, I'd make the work sequence:

| Commit                       | Files                                              | Purpose                                                |
| ---------------------------- | -------------------------------------------------- | ------------------------------------------------------ |
| **1. Trace backend overlap** | `ggml/src/ggml-backend.cpp`                        | Prove CUDA/CPU can overlap                             |
| **2. Refactor FFN builder**  | `src/llama-graph.{h,cpp}`                          | Extract `build_ffn_impl()` with zero behavioral change |
| **3. Partition data model**  | `src/llama-model.h`, new `llama-model-partition.*` | Represent persistent dense FFN slices                  |
| **4. Pack FFN slices**       | `llama-model-partition.cpp`, model loading path    | Create block-aligned GPU `up/gate/down` tensors        |
| **5. Build dual FFN graph**  | `src/llama-graph.cpp`                              | CPU + GPU FFN branches + sum                           |
| **6. Automatic registry**    | `llama-model.*`, `llama-graph.*`                   | Architecture-agnostic lookup by original `up` tensor   |
| **7. Configuration**         | `common/common.h`, `common/arg.cpp`                | `--ffn-split P`                                        |
| **8. Backend pinning**       | graph callback / backend assignment                | Guarantee CPU/GPU branch placement                     |
| **9. Correctness tests**     | tests                                              | Compare unsplit/split logits                           |
| **10. Bench support**        | `tools/llama-bench`                                | Sweep split fractions                                  |
| **11. Auto balance**         | partition subsystem                                | Measure CPU/GPU and select ratio                       |
| **12. Fit integration**      | `common/fit.cpp`                                   | Spend residual VRAM on dense FFN slices                |

Current `llama-bench` already exposes the regular GPU-layer/split/offload controls, so adding an FFN split dimension there is straightforward once common args exist. ([GitHub][10])

---

# The first usable MVP I'd actually build

I would aggressively limit scope:

```text
Backend:
    CPU + one CUDA GPU

FFN:
    parallel gate/up only
    SwiGLU/SILU

Weights:
    common GGML block quant formats
    start Q4_K / Q5_K / Q6_K / Q8_0 / F16

Partition:
    contiguous prefix of n_ff
    fixed fraction across all eligible layers

Placement:
    original full weights remain CPU
    GPU prefix is persistent VRAM copy

Execution:
    GPU branch first
    CPU branch second
    final sum on GPU

CLI:
    --ffn-split 0.00–1.00

No:
    arbitrary hot channels
    multi-GPU
    Vulkan
    dynamic partition
    auto-fit
    per-layer ratios
```

That MVP answers the fundamental research question:

> **Can a partially-offloaded dense model be faster if we parallelize each FFN across CPU and GPU instead of assigning whole transformer layers to either CPU or GPU?**

If the answer is no, we've avoided months of infrastructure work.

If the answer is yes, then we generalize it.

---

## The three code changes I'd start with

If you're going to start implementation immediately, these are the first three meaningful changes.

**1. Refactor `llm_graph_context::build_ffn()` in `src/llama-graph.cpp`.**

Take its current implementation and move the body to:

```cpp
build_ffn_impl(...)
```

Then make current:

```cpp
build_ffn(...)
```

a thin wrapper. Current model architectures already funnel dense FFNs through this common function, so this is our architecture-neutral hook.

**2. Add `llama_ffn_partition` + the persistent GPU packing code.**

Create:

```text
src/llama-model-partition.h
src/llama-model-partition.cpp
```

Initially hardcode:

```cpp
fraction = getenv("LLAMA_FFN_SPLIT")
    ? atof(...)
    : 0.0;
```

Don't waste time plumbing CLI/API yet.

Create:

```text
up_gpu
gate_gpu
down_gpu
```

for each eligible CPU layer.

**3. Modify `build_ffn()` to detect a partition and generate the two branches.**

Conceptually the final change is:

```cpp
ggml_tensor * llm_graph_context::build_ffn(...) {
    auto * part = find_ffn_partition(up);

    if (!part) {
        return build_ffn_impl(...);
    }

    auto * gpu = build_ffn_impl(
        cur,
        part->up_gpu,
        ...,
        part->gate_gpu,
        ...,
        part->down_gpu,
        ...);

    auto * cpu = build_ffn_impl(
        cur,
        make_cpu_up_view(up, part),
        ...,
        make_cpu_gate_view(gate, part),
        ...,
        make_cpu_down_view(down, part),
        ...);

    return ggml_add(ctx0, gpu, cpu);
}
```

That is the first version worth benchmarking.

---

### One thing I would **not** do

I would not start by inventing a new GGML operation such as:

```text
GGML_OP_SPLIT_MUL_MAT
```

or modifying CUDA matmul kernels.

We don't need it yet.

The existing graph abstraction can express:

```text
two matmuls
two FFN branches
two backends
one reduction
```

and the current scheduler is already designed around backend graph splits and asynchronous compute.

Only if profiling shows that scheduler boundaries/copies dominate should we create a fused heterogeneous operation.

The **biggest technical uncertainty is CPU/CUDA overlap**, not whether the FFN can be mathematically partitioned. The partition itself is straightforward. I would spend the first implementation session proving overlap, then build the packed `down` slice and dual-branch `build_ffn`; those three pieces are enough to get the first real tok/s result on the 1080.

[1]: https://github.com/ggml-org/llama.cpp/blob/master/common/fit.cpp "llama.cpp/common/fit.cpp at master · ggml-org/llama.cpp · GitHub"
[2]: https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen3.cpp?utm_source=chatgpt.com "llama.cpp/src/models/qwen3.cpp at master · ggml-org/llama.cpp · GitHub"
[3]: https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md?utm_source=chatgpt.com "llama.cpp/tools/completion/README.md at master · ggml-org/llama.cpp · GitHub"
[4]: https://github.com/ggml-org/llama.cpp/blob/master/src/models/laguna.cpp?utm_source=chatgpt.com "llama.cpp/src/models/laguna.cpp at master · ggml-org/llama.cpp · GitHub"
[5]: https://github.com/ggml-org/llama.cpp/blob/master/src/models/deepseek.cpp?utm_source=chatgpt.com "llama.cpp/src/models/deepseek.cpp at master · ggml-org/llama.cpp · GitHub"
[6]: https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h?utm_source=chatgpt.com "llama.cpp/include/llama.h at master · ggml-org/llama.cpp · GitHub"
[7]: https://github.com/ggml-org/llama.cpp/blob/master/common/common.h?utm_source=chatgpt.com "llama.cpp/common/common.h at master · ggml-org/llama.cpp · GitHub"
[8]: https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp?utm_source=chatgpt.com "llama.cpp/common/arg.cpp at master · ggml-org/llama.cpp · GitHub"
[9]: https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp?utm_source=chatgpt.com "llama.cpp/src/models/qwen35.cpp at master · ggml-org/llama.cpp · GitHub"
[10]: https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/llama-bench.cpp?utm_source=chatgpt.com "llama.cpp/tools/llama-bench/llama-bench.cpp at master · ggml-org/llama.cpp · GitHub"

Yes. In fact, the cleanest long-term architecture is to make both features share one lower-level mechanism: **sub-tensor placement and execution across CPU/GPU**.

For dense FFNs, the split is static and exhaustive: every channel is used, so we permanently place some FFN channels on GPU and the rest on CPU, then compute both branches and add the results.

For MoE FFNs, the split is dynamic and sparse: only selected experts are used for a token, so some expert slices are permanently or dynamically resident on GPU, while misses come from CPU RAM. The old expert-cache branch already operates at that “slice of a larger tensor” level. Current llama.cpp's MoE selective-transfer path also identifies expert slices via `MUL_MAT_ID`, so it gives us the right runtime hook.

The unified abstraction I'd aim for is something like:

```cpp
enum llama_weight_partition_kind {
    LLAMA_WEIGHT_PARTITION_DENSE_FFN,
    LLAMA_WEIGHT_PARTITION_EXPERT,
};

struct llama_weight_slice {
    const ggml_tensor * source;

    int64_t axis;
    int64_t begin;
    int64_t length;

    ggml_backend_t backend;
    ggml_tensor * resident;
};

struct llama_partitioned_weight {
    const ggml_tensor * source;
    std::vector<llama_weight_slice> slices;
};
```

Then the two cases become:

```text
Dense FFN
─────────
ffn_up:
  channels 0..4095       → CUDA
  channels 4096..11007   → CPU

ffn_gate:
  same channel partition

ffn_down:
  matching input-channel partition


MoE
───
ffn_up_exps:
  expert 17  → CUDA cache
  expert 53  → CUDA cache
  expert 91  → CUDA cache
  others     → CPU

ffn_down_exps:
  matching expert slices

...
```

So yes: **experts can participate in the same heterogeneous execution scheme**, not merely in the dynamic cache.

There are actually three useful MoE modes.

First is the existing dynamic expert cache concept:

```text
selected expert cached?
        │
   yes  │  no
        │
        ▼
GPU cache     RAM → GPU temporary
        \       /
         MUL_MAT_ID
```

That's ideal when VRAM is scarce and expert routing has temporal locality.

Second is **static hot-expert placement**:

```text
Most frequently used experts
        → permanently resident GPU

remaining experts
        → RAM
```

The profiler can determine which expert slices deserve VRAM. Unlike dense channels, this really matters semantically for performance because unused experts aren't evaluated at all.

Third, and most interesting, is combining the dense-style partitioning **inside individual experts**.

Suppose Qwen has:

```text
256 experts
8 active/token
```

and an individual expert itself is an FFN:

```text
expert:
    gate
    up
    down
```

Instead of choosing:

```text
expert 17 entirely GPU
expert 42 entirely CPU
```

we could do:

```text
expert 17:
    60% FFN channels GPU
    40% CPU

expert 42:
    30% GPU
    70% CPU
```

So the hierarchy becomes:

```text
MoE layer
│
├── router selects 8 experts
│
├── expert 17
│      ├── GPU FFN slice
│      └── CPU FFN slice
│
├── expert 42
│      ├── GPU FFN slice
│      └── CPU FFN slice
│
└── ...
```

That is mathematically valid for the same reason the dense FFN split is valid: an expert FFN is just another FFN.

The important point is that **expert selection happens first**, then heterogeneous execution happens only for the selected experts.

Conceptually:

```text
router
  ↓
selected experts [3, 19, 42, 87, ...]
  ↓

for each selected expert:

   GPU-resident channels ─┐
                          ├─ expert output
   CPU-resident channels ─┘
```

That gives us a continuum rather than today's binary decision:

```text
whole tensor on CPU
vs
whole tensor on GPU
```

You could spend VRAM something like:

```text
VRAM budget = 2 GB

500 MB:
    permanently resident hottest experts

1 GB:
    partial channel slices across frequently-used experts

500 MB:
    dynamic expert cache for cold/miss experts
```

That could be substantially better than any one technique alone.

## I'd architect it in layers

The lower layer should know nothing about LLM architectures:

```text
GGML / backend layer

subtensor resident store
    ↓
CPU/GPU buffers
    ↓
slice copies
    ↓
backend execution
```

Above that sits a generic llama.cpp partition manager:

```cpp
class llama_weight_partition_manager {
public:
    const llama_weight_slice *
    find_resident_slice(
        const ggml_tensor * tensor,
        llama_slice_id id,
        ggml_backend_t backend);

    ...
};
```

Then there are two policies.

### Dense policy

```text
tensor = ffn_up

slice ID:
    FFN channel range

placement:
    static
```

### MoE policy

```text
tensor = ffn_up_exps

slice ID:
    expert ID
or
    expert ID + channel range

placement:
    static and/or cache-managed
```

That distinction belongs at the llama/model-policy level, not GGML.

The backend should only understand:

> Here's a tensor slice and here's where it lives.

## For `MUL_MAT_ID`, I'd eventually change the API slightly

Today's `MUL_MAT_ID` conceptually expects:

```text
one expert-bank tensor
+
expert IDs
```

and llama.cpp arranges for the required expert data to appear in the destination buffer.

The expert cache prototype therefore still does:

```text
GPU cache
  ↓ copy
temporary expert-bank GPU tensor
  ↓
MUL_MAT_ID
```

The ideal unified implementation would allow a selected expert to point directly at its backing slice.

Something like:

```cpp
struct ggml_expert_source {
    ggml_tensor * tensor;
    size_t offset;
    ggml_backend_t backend;
};
```

Then:

```text
expert 5  → CPU tensor offset X
expert 7  → GPU cache offset Y
expert 11 → permanent GPU tensor offset Z
```

And `MUL_MAT_ID` consumes those directly.

That removes the GPU→GPU copy on cache hits.

But I would keep that as V2 because it requires backend kernel changes.

For MVP, use the same staging model as the old cache branch. It already demonstrated the mechanism: classify selected experts into cached and uncached sets, transfer misses CPU→GPU, use GPU→GPU copies for hits, then execute the expert matmul.

## The combined implementation order I'd use now

I would actually change the roadmap slightly from what I suggested before.

**Stage 1 — generic slice metadata.**

Introduce:

```cpp
struct llama_tensor_slice {
    const ggml_tensor * source;

    int64_t dim;
    int64_t start;
    int64_t length;

    ggml_tensor * resident;
    ggml_backend_dev_t device;
};
```

No dense/MoE assumptions.

**Stage 2 — expert cache using generic slices.**

Port Alderson's cache, but each cache entry becomes:

```cpp
struct expert_cache_entry {
    llama_tensor_slice slice;
    int32_t expert_id;

    uint64_t last_used;
};
```

That immediately gives us a working feature and validates the abstraction.

**Stage 3 — dense FFN static slicing.**

Use the exact same `llama_tensor_slice` representation for:

```text
up/gate channel ranges
down matching channel range
```

Then build the two FFN graph branches.

**Stage 4 — static expert residency.**

Allow:

```text
expert 17 → permanent CUDA slice
expert 23 → permanent CUDA slice
```

before consulting the dynamic cache.

Lookup order:

```text
selected expert
    │
    ├─ permanent GPU resident? → use it
    │
    ├─ dynamic cache hit?      → use cache
    │
    └─ miss                    → RAM transfer
```

**Stage 5 — partially split experts.**

Once dense splitting works, reuse `build_ffn_partitioned()` for selected experts.

This is the hardest variant because current MoE computation is batched through `MUL_MAT_ID` rather than constructing eight individual FFN subgraphs.

We would either need:

```text
A. teach MUL_MAT_ID about channel-partitioned experts
```

or:

```text
B. expand selected experts into explicit FFN branches
```

A is ultimately better.

## The really interesting automatic allocator

Once both systems exist, `--fit` no longer needs to think in terms of just:

```text
how many layers fit?
```

It could solve:

```text
Given 8 GB VRAM, what allocation produces minimum decode latency?
```

Possible allocations:

```text
dense attention tensors
whole layers
hot experts
dynamic expert-cache space
partial dense FFN channels
partial expert FFN channels
KV/cache/compute
MTP
```

The allocator can attach a benefit-per-byte estimate to each candidate.

For example:

```text
candidate                     VRAM       expected benefit
──────────────────────────────────────────────────────────
whole dense layer             430MB       1.9 ms/token
hot expert #17                 11MB        0.21 ms/token
hot expert #23                 11MB        0.18 ms/token
+5% FFN channels all layers   190MB        0.7 ms/token
expert cache +256MB           256MB        0.9 ms/token
```

Then approximately maximize:

```text
saved decode time / VRAM byte
```

subject to the GPU memory limit.

That's much more sophisticated than `n_gpu_layers`, but the generic slice system makes it possible.

For your Qwen MoE case, I suspect the best practical configuration may end up looking something like:

```text
GTX 1080 8 GB
────────────────────────────

dense/shared tensors            ~3.5 GB
KV + compute + MTP               ~1.5 GB

static hottest experts           ~1.0 GB
dynamic expert cache             ~1.0 GB
partial FFN slices / remainder   ~1.0 GB
```

rather than trying to fit whole sequential layers.

The exact proportions need measurement, but that's precisely what the profiler/cache statistics would let us learn.

So **yes, I would deliberately design the dense work and expert-cache work as one subsystem now**. The common primitive is:

> **A logical weight tensor can have independently resident slices on different backends.**

Dense FFNs use static channel slices. MoE uses expert slices. Later MoE can use `(expert, channel-range)` slices. Once that primitive exists, the policies above it become much simpler, and `--fit` can eventually allocate VRAM based on actual performance value rather than model-layer boundaries.
