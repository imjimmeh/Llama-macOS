Yes. Ignoring the variable-expert experiment and targeting simply:

**GTX 1080 (SM61) + Qwen 3.x + llama.cpp + fastest possible generation**, I'd implement these five in this order.

### 1. Add a dedicated SM61 Qwen decode GEMV path

This is the biggest change and the one most directly inspired by NInfer.

During batch-1 generation, don't send Qwen's linear layers through the generic MMQ machinery. Detect the common case:

```cpp
cc == 610
&& batch_size == 1
&& model_arch == QWEN
&& quant_supported
```

and dispatch to a dedicated DP4A GEMV kernel.

Something structurally like:

```cpp
template <ggml_type QTYPE, int BLOCK_SIZE>
__global__ void qwen_sm61_gemv(
    const void * __restrict__ weight,
    const float * __restrict__ x,
    float * __restrict__ y,
    int ncols,
    int nrows) {

    const int row = blockIdx.x;

    int acc0 = 0;
    int acc1 = 0;
    int acc2 = 0;
    int acc3 = 0;

    // quantize/cache activation fragments
    // then four independent DP4A chains

#pragma unroll
    for (int k = threadIdx.x; k < ncols_q; k += blockDim.x) {
        const int4 w = load_q4k_as_int4(weight, row, k);
        const int4 a = load_q8_activation(x, k);

        acc0 = ggml_cuda_dp4a(w.x, a.x, acc0);
        acc1 = ggml_cuda_dp4a(w.y, a.y, acc1);
        acc2 = ggml_cuda_dp4a(w.z, a.z, acc2);
        acc3 = ggml_cuda_dp4a(w.w, a.w, acc3);
    }

    float sum = dequantize(acc0 + acc1 + acc2 + acc3);

    // warp/block reduction
    ...
}
```

The key differences from generic llama.cpp would be:

```text
GEMV, not GEMM/MMQ
compile-time quant type
SM61-only
fixed Qwen shapes where useful
multiple independent DP4A accumulators
minimal branching
no generic ID/broadcast/layout handling
```

I'd initially implement only the **exact quant you're using**, probably Q4_K or Q5_K.

Don't try to make this a new universal llama.cpp kernel. Hard-specialize it.

This is essentially applying NInfer's philosophy to Pascal.

---

### 2. Fuse Qwen's gate + up + SwiGLU into one SM61 kernel

This is probably the clearest Qwen-specific optimisation.

Instead of:

```text
gate = GEMV(W_gate, x)
up   = GEMV(W_up, x)

tmp = SiLU(gate)
out = tmp * up
```

make:

```cpp
qwen_sm61_swiglu_gemv(...)
```

that processes both matrices simultaneously:

```cpp
template <ggml_type QTYPE>
__global__ void qwen_sm61_swiglu_gemv(
    const void * __restrict__ gate_w,
    const void * __restrict__ up_w,
    const float * __restrict__ x,
    float * __restrict__ out,
    int ncols,
    int nrows) {

    const int row = blockIdx.x;

    int gate0 = 0, gate1 = 0;
    int gate2 = 0, gate3 = 0;

    int up0 = 0, up1 = 0;
    int up2 = 0, up3 = 0;

#pragma unroll
    for (...) {
        const int4 a = load_activation(...);

        const int4 wg = load_gate(...);
        const int4 wu = load_up(...);

        gate0 = ggml_cuda_dp4a(wg.x, a.x, gate0);
        gate1 = ggml_cuda_dp4a(wg.y, a.y, gate1);
        gate2 = ggml_cuda_dp4a(wg.z, a.z, gate2);
        gate3 = ggml_cuda_dp4a(wg.w, a.w, gate3);

        up0 = ggml_cuda_dp4a(wu.x, a.x, up0);
        up1 = ggml_cuda_dp4a(wu.y, a.y, up1);
        up2 = ggml_cuda_dp4a(wu.z, a.z, up2);
        up3 = ggml_cuda_dp4a(wu.w, a.w, up3);
    }

    float gate = reduce_and_dequant(...);
    float up   = reduce_and_dequant(...);

    if (threadIdx.x == 0) {
        out[row] = silu(gate) * up;
    }
}
```

The major win isn't the SiLU arithmetic.

It's that `x` is shared between both operations:

```text
                       generic

read X → gate GEMV → write gate
read X → up GEMV   → write up
read gate
read up
SiLU + multiply
write result


                       fused

read X
  ├── DP4A gate
  └── DP4A up
        ↓
    SiLU × up
        ↓
       write
```

On the GTX 1080, where inference is strongly bandwidth-sensitive, that's exactly what we want.

This is one of the strongest lessons I'd directly copy from NInfer.

---

### 3. Add a Qwen-specific fused MoE decode kernel

Even without changing the number of active experts, Qwen MoE deserves its own decode path.

Instead of:

```text
router
↓
top-k
↓
MUL_MAT_ID down/up/gate
↓
activation
↓
weighted sum
```

make the decoder know this is Qwen.

Something like:

```cpp
template <
    ggml_type QTYPE,
    int TOP_K,
    int HIDDEN,
    int EXPERT_DIM
>
__global__ void qwen_sm61_moe_decode(...);
```

The important change is mapping expert execution directly onto warps/blocks rather than treating it as generic indexed matrix multiplication.

For example:

```cpp
const int expert_slot = blockIdx.y;

if (expert_slot >= TOP_K) {
    return;
}

const int expert_id = expert_ids[expert_slot];
const float weight  = expert_weights[expert_slot];

const void * gate =
    experts_gate + expert_id * gate_stride;

const void * up =
    experts_up + expert_id * up_stride;

// fused gate/up GEMV:
float activated =
    expert_swiglu_dp4a(gate, up, x);

...
```

Then either:

- perform down projection immediately, or
- write a compact expert intermediate and launch one fused down/reduction kernel.

The end goal is replacing generic:

```cpp
GGML_OP_MUL_MAT_ID
```

for Qwen decode.

NInfer's treatment of MoE as a **special operator rather than generic indexed GEMMs** is probably one of its most important architectural lessons.

For a Qwen MoE model, I'd rank this above generic MMQ tuning.

---

### 4. Specialize the existing Pascal MMQ config for GTX 1080 small-batch work

Keep our earlier change, because the specialized kernels won't handle every shape.

Current Pascal configs are heavily biased toward:

```cpp
256 threads
I = 64
K = 256
```

Add GTX-1080-oriented low-J cases in:

```text
ggml/src/ggml-cuda/mmq-config-pascal.cuh
```

For example:

```cpp
// Existing:
CASE(
    GGML_TYPE_Q4_K,
    256, 2,
    64, 8,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1,
    MMQ_ITER_K,
    false, false);

// SM61 small-batch:
CASE(
    GGML_TYPE_Q4_K,
    128, 2,
    32, 8,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1,
    MMQ_ITER_K,
    false, false);
```

Likewise `J=16`.

I'd test:

```text
J=8:
    128 / I32
    128 / I64
    256 / I32
    256 / I64 baseline

J=16:
    same
```

Don't assume `128/I32` wins.

GTX 1080 has only 20 SMs, and optimal scheduling can be surprisingly sensitive to block count and register usage.

Make these selectable at compile time initially:

```cpp
#ifdef GGML_CUDA_SM61_SMALL_TILES
...
#else
...
#endif
```

so benchmarking A/B is trivial.

---

### 5. Add SM-wave-aware tile selection specifically for GP104

This complements #4.

llama.cpp shouldn't select the "smallest number of blocks" if doing so produces horrible scheduling across 20 SMs.

Suppose two possibilities produce:

```text
21 blocks
vs
38 blocks
```

The first gives:

```text
wave 1: 20/20
wave 2:  1/20
```

The second:

```text
wave 1: 20/20
wave 2: 18/20
```

Despite having more blocks, the second can be substantially better.

So modify Pascal MMQ configuration selection to score wave occupancy.

For the GTX 1080:

```cpp
static inline int sm61_wave_score(
    int blocks,
    int nsm) {

    const int waves =
        (blocks + nsm - 1) / nsm;

    const int available_slots =
        waves * nsm;

    return (blocks * 1000) /
           available_slots;
}
```

Then when choosing `J`:

```cpp
if (cc == 610) {
    const int nx =
        (ncols + config.J - 1) / config.J;

    const int ny =
        (nrows + config.I - 1) / config.I;

    const int blocks =
        nx * ny * nchannels * nsamples;

    const int wave_eff =
        sm61_wave_score(blocks, device.nsm);

    // Efficiency is dominant, but avoid making
    // pathological numbers of tiny blocks.
    const int score =
        wave_eff * 10000 -
        blocks;

    if (score > best_score) {
        best_score = score;
        J_best = J;
    }
}
```

For your GTX 1080:

```cpp
device.nsm == 20
```

so this automatically tunes around its actual geometry rather than assuming "Pascal" means one homogeneous GPU.

---

## I'd implement them in this order

| Priority | Change                      | Expected target                 |
| -------- | --------------------------- | ------------------------------- |
| **1**    | SM61 Qwen decode GEMV       | Most dense linear layers        |
| **2**    | Fused gate/up/SwiGLU GEMV   | Every Qwen FFN                  |
| **3**    | Qwen MoE decode fast path   | MoE models specifically         |
| **4**    | 128-thread/I32 MMQ variants | Remaining small GEMMs           |
| **5**    | 20-SM-aware MMQ selection   | Better scheduling across shapes |

And there's an important distinction between **#1–3** and **#4–5**.

#4–5 are "make llama.cpp's generic CUDA backend somewhat better."

#1–3 are the NInfer approach:

> **Stop paying for generality on the path that accounts for most of your inference time.**

For a dedicated local setup, that's where I'd expect the larger gains to come from.

I would also scope the first implementation extremely narrowly:

```text
GPU:
    sm_61 only

architecture:
    Qwen only

mode:
    batch-1 generation first

quant:
    your actual GGUF quant only

fallback:
    existing llama.cpp
```

If you're using **Q4_K_M**, for example, I wouldn't initially write Q5/Q6 kernels at all. I'd optimize precisely `Q4_K × Q8 activation → FP32 accumulator` with DP4A, benchmark it, and only generalize once we know the approach wins. That's the part of NInfer's strategy I'd copy most aggressively.
