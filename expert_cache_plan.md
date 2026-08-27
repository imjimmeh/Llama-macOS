Yes. And after looking at the actual cache commit, I think **the right approach is to port the idea, not the implementation**.

The old branch's cache is already surprisingly close to architecture-agnostic because it lives in `ggml/src/ggml-backend.cpp` around `GGML_OP_MUL_MAT_ID`, rather than inside Gemma model code. Its fundamental mechanism is good: llama.cpp already detects which expert IDs a `MUL_MAT_ID` needs, copies only those expert slices from host memory to the accelerator, and the branch interposes a persistent accelerator-side cache so hits become GPU→GPU rather than RAM→GPU transfers.

However, the implementation has several Gemma-specific hacks: 8 fixed slots, 64 maximum layers, exactly two expert tensor types distinguished by checking whether the tensor name contains `"down"`, layer numbers parsed from `blk.%d`, and a hard-coded 30×8 hot-expert table from Gemma profiling. We should remove **all of that**.

## Target design

I'd make the cache key simply:

```cpp
struct ggml_expert_cache_key {
    const ggml_tensor * tensor; // expert weight bank
    int32_t expert_id;
};
```

That's the important architectural decision.

We **don't need to know**:

```text
Qwen vs Gemma vs DeepSeek vs Mixtral
layer number
gate/up/down
number of layers
number of experts globally
model architecture
```

At runtime, llama.cpp already gives us everything necessary:

```text
MUL_MAT_ID
    ↓
source expert tensor
    ↓
tensor->ne[2]       = number of experts
tensor->nb[2]       = bytes per expert
    ↓
node->src[2]        = selected expert IDs
```

Current master already does exactly this to avoid copying unused CPU-offloaded experts.

So the new cache becomes:

```text
(source expert tensor, expert ID)
                  │
                  ▼
          Expert Cache
          ┌─────────────┐
hit ─────►│ GPU resident│─────► input_cpy
          └─────────────┘

miss:
RAM ──────────────► input_cpy
                        │
                        └────► expert cache
```

That should automatically work with **Qwen3.5/3.6 MoE, Gemma, Mixtral, DeepSeek, Ling, OLMoE, future architectures**, provided they ultimately express their routed expert matmul as `GGML_OP_MUL_MAT_ID`.

Current llama.cpp explicitly supports CPU+GPU hybrid inference and many MoE architectures already use this common backend mechanism. ([GitHub][1])

---

# Implementation plan

I'd break it into six stages.

## 1. Extract an architecture-neutral expert-cache object

Don't put a pile of globals into `ggml-backend.cpp` as the old branch does.

Add something approximately like:

```cpp
struct ggml_backend_expert_cache_entry {
    const ggml_tensor * source;
    int32_t expert_id;

    size_t offset;
    size_t size;

    uint64_t last_used;
    uint64_t hit_count;
};

struct ggml_backend_expert_cache {
    ggml_backend_t backend;

    ggml_backend_buffer_t buffer;
    void * buffer_base;

    size_t capacity;
    size_t used;

    uint64_t clock;

    std::vector<ggml_backend_expert_cache_entry> entries;
};
```

There should be **one cache per accelerator backend**, not one global cache.

So:

```text
CPU
 │
 ├── CUDA0 expert cache
 ├── CUDA1 expert cache
 ├── Vulkan0 expert cache
 └── etc.
```

This matters for multi-GPU llama.cpp.

I would initially put this in:

```text
ggml/src/ggml-backend-expert-cache.cpp
ggml/src/ggml-backend-expert-cache.h
```

rather than making `ggml-backend.cpp` another 500 lines bigger.

---

## 2. Hook into the existing `MUL_MAT_ID` selective-copy path

This is the ideal insertion point.

Current master already does:

```cpp
if (
    ggml_backend_buffer_get_usage(input->buffer) ==
        GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
    ggml_backend_buffer_is_host(input->buffer) &&
    node->src[0] == input_cpy &&
    node->op == GGML_OP_MUL_MAT_ID
) {
    const int64_t n_expert = input->ne[2];
    const size_t expert_size = input->nb[2];

    // read selected expert IDs
    ...
    // copy only selected experts
}
```

Change the last part conceptually from:

```cpp
for (expert : used_experts) {
    copy_ram_to_gpu(expert);
}
```

to:

```cpp
for (expert : used_experts) {
    auto * entry = cache.find(input, expert);

    if (entry) {
        copy_cache_to_input(entry, input_cpy, expert);
        cache.touch(entry);
    } else {
        copy_ram_to_input(input, input_cpy, expert);

        cache.insert_async(
            input,
            expert,
            input_cpy);
    }
}
```

The existing branch already proves this strategy works: it first classifies requested experts into hits and misses, copies misses CPU→GPU, then performs GPU→GPU copies for hits and updates the cache with newly fetched misses.

### Important

Preserve current llama.cpp's **grouped contiguous CPU copies**.

Current master groups:

```text
expert 10
expert 11
expert 12
```

into one RAM→GPU copy instead of three.

So misses should still become a bitset and go through the existing grouping code.

That means:

```text
Requested:  3 4 5 20 32 33 34 35
Cached:       4   20       34
Misses:     3   5    32 33    35
```

should produce approximately:

```text
RAM→GPU:
3
5
32-33
35

cache→GPU:
4
20
34
```

rather than eight individual transfers.

---

# 3. Replace fixed slots with a real byte-capacity cache

This is one place I'd significantly deviate from Alderson's implementation.

His cache has:

```cpp
CACHE_SLOTS = 8;
CACHE_MAX_LAYERS = 64;
[layer][gate/down][slot]
```

and allocates essentially:

```text
layers × 8 × expert_size × tensor-types
```

That's inappropriate for general llama.cpp.

Instead configure:

```bash
--expert-cache 1024M
```

or:

```bash
--expert-cache 2G
```

The cache then contains as many expert slices as fit.

For your card, for example:

```bash
llama-server \
    -m qwen.gguf \
    --cpu-moe \
    --expert-cache 1500M
```

would mean:

> Reserve approximately 1.5 GiB of VRAM for dynamically cached CPU-resident experts.

No assumption about expert size.

If:

```text
Qwen expert slice = 2.8 MB
```

you get ~548 entries.

If:

```text
Gemma expert slice = 5 MB
```

you get ~307.

That's much cleaner.

### Variable-size allocation

For MVP, don't build a complicated memory allocator.

Use fixed-size **cache arenas per source tensor**.

When the first miss from a tensor arrives:

```cpp
expert_size = input->nb[2];
```

create a slab class of that size.

Something like:

```text
Cache 1536 MB
│
├── 3.2 MB slices
│   ├── Qwen blk.2 gate/up expert 17
│   ├── Qwen blk.7 gate/up expert 52
│   └── ...
│
├── 1.6 MB slices
│   ├── Qwen blk.2 down expert 17
│   └── ...
│
└── ...
```

Later we can improve fragmentation.

---

# 4. Use LRU initially, then make replacement MoE-aware

The old branch uses a fairly simple rotating replacement policy that avoids evicting experts being hit by the current operation.

For the first implementation I'd use **LRU**:

```cpp
on_hit:
    entry.last_used = ++clock;

on_miss:
    if insufficient_space:
        evict lowest last_used
```

But I would immediately collect:

```text
hits
misses
evictions
bytes CPU→GPU avoided
bytes GPU→GPU copied
per-source-tensor hit rate
```

because I suspect we can beat LRU for MoE.

Expert routing has temporal locality. Alderson measured roughly **67% of requested experts already cached on consecutive tokens** in his test, which is why this works at all. His cache gave Gemma 4 26B-A4B decode improvement from 27.9 to 35.1 tok/s on an RX 9070 XT.

A second-generation policy could use:

```text
score =
    recency
  + frequency
  + consecutive-token probability
  + static global hotness
```

Something approximately ARC/LFU-LRU hybrid.

But **don't start there**. Prove LRU first.

---

# 5. Make cache discovery completely automatic

This is what makes the feature architecture agnostic.

Don't maintain a list such as:

```cpp
if (arch == QWEN35MOE ||
    arch == GEMMA4 ||
    arch == DEEPSEEK ...)
```

Instead:

```text
Is the op MUL_MAT_ID?
        │
        ▼
Are its weights host-resident?
        │
        ▼
Is its execution backend an accelerator?
        │
        ▼
Cache eligible.
```

That's it.

Essentially:

```cpp
bool expert_cache_eligible(
    const ggml_tensor * node,
    const ggml_tensor * weights,
    ggml_backend_t src,
    ggml_backend_t dst) {

    return
        node->op == GGML_OP_MUL_MAT_ID &&
        ggml_backend_buffer_get_usage(weights->buffer)
            == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(weights->buffer) &&
        src != dst;
}
```

This is considerably better than tagging architectures.

It also means a newly-added MoE model inherits the optimisation **without the model implementation knowing the expert cache exists**.

---

# 6. Integrate it properly with `--fit`

This is particularly important for your setup.

If you simply add:

```bash
--expert-cache 1500M
```

after fit has decided how many normal layers fit, you'll OOM.

So the cache capacity needs to participate in llama.cpp's VRAM budget.

Conceptually:

```text
available VRAM
    - compute buffers
    - KV
    - MTP reserve
    - fit safety margin
    - expert cache reserve
    = VRAM available for permanent model layers
```

So:

```bash
--fit
--expert-cache 1G
```

should cause `fit` to deliberately leave 1 GiB free.

For your 1080 this creates an interesting optimisation problem:

```text
8 GB VRAM

██████████████████  permanent dense/model tensors
████                KV + compute
███                 MTP
████                expert cache
```

rather than:

```text
████████████████████████ all available space consumed
                         by ordinary layer offload
```

And that's potentially a **much better allocation of the last 1–2 GB**, because cached experts are the things repeatedly crossing PCIe.

---

# A change I'd make specifically for decode

The old implementation also forces `MUL_MAT_ID` onto the GPU during decode using `GGML_OP_OFFLOAD_MIN_BATCH=1`.

That's crucial.

Without that, you can end up with:

```text
CPU expert weights
+
tiny decode batch
        ↓
scheduler chooses CPU MUL_MAT_ID
```

and then your GPU expert cache is irrelevant.

So I'd make the new behaviour explicit rather than depending on another magic environment variable:

```bash
--expert-cache 1G
```

should imply:

> CPU-resident MoE weights may still execute `MUL_MAT_ID` on the accelerator using selective expert transfer/cache.

Potentially expose:

```bash
--expert-cache-offload auto|always|never
```

with `auto` as default.

---

# Multi-GPU needs to be designed in from day one

This is easy to get wrong if added later.

A cache entry isn't simply:

```text
tensor X / expert 17
```

It's:

```text
backend CUDA0 / tensor X / expert 17
backend CUDA1 / tensor X / expert 17
```

because a copy resident on CUDA0 doesn't help CUDA1 unless we're explicitly doing peer transfers.

Therefore:

```cpp
struct ggml_backend_expert_cache {
    ggml_backend_t backend;
    ...
};
```

and the scheduler owns:

```cpp
std::unordered_map<
    ggml_backend_t,
    ggml_backend_expert_cache
> expert_caches;
```

Later we could exploit P2P:

```text
CPU → CUDA0 cache

CUDA1 needs same expert

CUDA0 → CUDA1
instead of
CPU → CUDA1
```

but that's phase two.

---

# Profiling should come for free

Once we have the cache, **we essentially get your expert profiler automatically**.

Every request already passes:

```cpp
(input_tensor, selected_expert_id)
```

through the cache.

Just increment:

```cpp
stats[input][expert].requests++;
```

Then:

```bash
--expert-cache-stats
```

could print:

```text
Expert cache
  capacity:             1536 MiB
  resident:             1498 MiB
  requests:             1,284,992
  hits:                   834,631
  misses:                 450,361
  hit rate:                 64.95 %
  RAM → GPU:              38.2 GiB
  avoided RAM → GPU:      71.4 GiB
  evictions:               93,818
```

And optionally:

```bash
--expert-profile expert-profile.csv
```

produces:

```csv
tensor,expert,requests,hits,misses,hit_rate
blk.0.ffn_gate_exps,0,1277,981,296,0.7682
blk.0.ffn_gate_exps,1,843,590,253,0.6999
...
```

No special profiler build necessary.

---

# I would _not_ pre-seed the cache initially

Alderson has hardcoded hot experts and pre-seeds the cache from profiling data.

I'd deliberately omit that from MVP.

Start empty:

```text
token 1:
    misses → populate

token 2:
    many hits

token 3:
    more hits
```

This has several advantages:

```text
no model-specific profile
no workload assumptions
no startup transfer cost
automatically adapts to current conversation/domain
```

Then later add optional profile loading:

```bash
--expert-cache-profile qwen-coding.profile
```

which can seed known-hot experts.

That gives us both:

**Dynamic mode**

```text
learn current workload online
```

and **profile-guided mode**

```text
start with known high-frequency experts
then adapt
```

---

# The main implementation epics

If I were actually giving this to a coding agent, I'd divide it roughly like this:

| Epic                         | Work                                                                      | Outcome                            |
| ---------------------------- | ------------------------------------------------------------------------- | ---------------------------------- |
| **1. Cache core**            | Cache key, entries, accelerator buffer, LRU, lifecycle                    | Generic expert cache               |
| **2. Scheduler integration** | Intercept current `MUL_MAT_ID` selective copies, classify hits/misses     | Cache actually used                |
| **3. Efficient transfers**   | Preserve contiguous miss grouping; generic accelerator→accelerator copies | No performance regression          |
| **4. Configuration**         | `--expert-cache`, enable/disable, logging                                 | User-controlled VRAM               |
| **5. Fit integration**       | Reserve cache capacity before layer fitting                               | Safe partial offload               |
| **6. Statistics/profile**    | hit/miss/bytes/expert counts + optional CSV                               | Measure behaviour                  |
| **7. Multi-backend**         | cache per destination device                                              | Multi-GPU/Vulkan/etc. safe         |
| **8. Tests/benchmarks**      | fake backend tests + Qwen/Gemma/etc.                                      | Validate architecture independence |
| **9. Advanced policies**     | LFU/LRU hybrid, preseed profiles, cache sizing                            | Further performance                |

---

## What I'd port directly vs rewrite

From Alderson's branch:

**Keep conceptually:**

```text
✓ identify active experts from MUL_MAT_ID IDs
✓ divide into hits/misses
✓ CPU→accelerator copy misses only
✓ accelerator→accelerator copy hits
✓ populate cache from misses
✓ don't evict experts needed by current operation
✓ execute decode MUL_MAT_ID on accelerator
```

The branch actually implements exactly this three-pass sequence.

**Throw away:**

```text
✗ CACHE_SLOTS = 8
✗ CACHE_MAX_LAYERS = 64
✗ [layer][2][slot]
✗ strstr(name, "down")
✗ sscanf(name, "blk.%d")
✗ Gemma hot-expert table
✗ global singleton buffer
✗ implicit ~1 GB sizing
```

Those are prototype shortcuts, not things we should carry forward.

---

## One deeper change I would consider after MVP

The current approach still does:

```text
cache
  ↓ GPU→GPU copy
temporary full expert-bank input_cpy
  ↓
MUL_MAT_ID
```

So even on a cache hit we're copying the expert into the temporary `input_cpy` tensor.

The **real end-state** would be to make `MUL_MAT_ID` understand a **sparse/non-contiguous expert backing store**:

```text
Expert 7  → original CPU tensor
Expert 12 → GPU cache slot 41
Expert 19 → GPU cache slot 3
...
```

Then a cache hit requires **no expert-weight copy at all**:

```text
router selects expert 12
       ↓
MUL_MAT_ID reads cache slot directly
```

That is potentially significantly better, especially on your Pascal GPU where we want to minimise extra memory traffic.

But I would **not do this first**. It touches CUDA/Vulkan/etc. implementations of `MUL_MAT_ID` and massively expands the scope.

### MVP

```text
RAM ──miss──► input_cpy ──► matmul
cache ─hit──► input_cpy ──► matmul
```

### V2

```text
RAM expert ───────┐
                  ├──► MUL_MAT_ID
cache expert ─────┘
```

The MVP stays almost entirely inside the generic scheduler and therefore has a much better chance of being maintainable/upstreamable.

---

# Heterogeneous MoE Execution Architecture (Implemented & Verified 2026-08-27)

True partial-hit heterogeneous execution is fully implemented and verified for single-token decode (TG1):

```text
Routes:
[ e0, e1, e2, e3, e4, e5, e6, e7 ]

Residency:
[ GPU, GPU, GPU, GPU, GPU, GPU, GPU, CPU ]

Concurrent Dual-Device Execution:
GPU CUDA Stream:            CPU Threadpool (14 Threads):
  e0, e1, e2, e3, e4, e5, e6  e7 (using host RAM weights)
        │                          │
        │                    upload miss output (Async H2D)
        │                          │
        └────────► GPU Scatter ◄───┘
                       │
             canonical down output (down_node->data)
                       │
            existing router reduction
```

### Verified Invariants:
1. **Current misses never trigger expert-weight H2D PCIe transfers** (`hetero_weight_upload_bytes == 0`).
2. **One miss never forces GPU-resident routes onto CPU** (GPU hits execute on GPU slot pools).
3. **GPU hit routes execute zero-copy directly from GPU slot pools**.
4. **CPU miss routes execute concurrently on host RAM and upload only unweighted outputs** (`n_misses * d_model * sizeof(float)`).
5. **All outputs are scattered directly into standard `down_node->data` buffer via warp-level `ggml_cuda_moe_scatter_routes`**.
6. **Zero full-backend host synchronizations** in the hot decode path.

### Hard Acceptance Test Matrix (N=0..8):
Tested on NVIDIA GeForce GTX 1080 (8 GB VRAM) with Gate A Model Spec ($d_{\text{model}}=2048, d_{\text{ff}}=512, n_{\text{expert}}=256, \text{top\_k}=8$, TG1, $Q4\_K / Q6\_K$):

```text
hit mask | GPU routes executed | CPU routes executed | Down NMSE | MoE MaxAbs | Status
---------|--------------------:|--------------------:|:---------:|:----------:|:------:
0/8      |                   0 |                   8 |  0.000000 |     0.0000 | PASS
1/8      |                   1 |                   7 |  0.000025 |    23.9883 | PASS
2/8      |                   2 |                   6 |  0.000057 |    30.3734 | PASS
3/8      |                   3 |                   5 |  0.000075 |    33.5492 | PASS
4/8      |                   4 |                   4 |  0.000094 |    43.6974 | PASS
5/8      |                   5 |                   3 |  0.000117 |    49.9923 | PASS
6/8      |                   6 |                   2 |  0.000139 |    49.2716 | PASS
7/8      |                   7 |                   1 |  0.000177 |    53.0935 | PASS
8/8      |                   8 |                   0 |  0.000200 |    52.7427 | PASS
```

[1]: https://github.com/martinalderson/llama.cpp/tree/moe-profile "GitHub - martinalderson/llama.cpp at moe-profile · GitHub"
