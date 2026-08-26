# Heterogeneous Inference: Dynamic MoE Expert Caching & Dense FFN Partitioning

This document describes the design, architecture, and usage of **Dynamic MoE Expert Caching** and **Dense FFN Heterogeneous Partitioning** in `llama.cpp`.

---

## 1. Overview & Architecture

When running large modern LLMs on memory-constrained consumer accelerators (e.g. 8 GB - 16 GB VRAM), standard layer-level offloading faces two distinct computational bottlenecks:

1. **Mixture-of-Experts (MoE) Models**: Only a small subset of experts (e.g. 8 out of 64 or 128) are activated per token. Transferring all CPU-resident expert weights across the PCIe bus on every token introduces massive memory bandwidth bottlenecks.
2. **Dense Exhaustive FFNs / Shared Experts**: Full dense projections (in dense models or shared-expert branches of MoE models) require evaluating the entire layer. When a layer does not fit in VRAM, running it entirely on CPU underutilizes available GPU compute and VRAM.

To address both bottlenecks, `llama.cpp` implements two complementary, decoupled optimization domains united under a single memory allocator:

```text
                                Transformer Layer
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             │                                                     │
      Dense FFN Branch                                     Routed MoE Branch
 (e.g. ffn_*, ffn_*_shexp)                               (e.g. ffn_*_exps, ffn_gate_up_exps)
             │                                                     │
        build_ffn()                                          build_moe_ffn()
             │                                                     │
      FFN Partitioning                                        Expert Cache
   (GPU Prefix + CPU Rest)                                 (RAM <-> VRAM Dynamic)
             │                                                     │
             └──────────────────────────┬──────────────────────────┘
                                        │
                                   Layer Output
```

---

## 2. Dynamic MoE Expert Cache

### 2.1 Current Operation and Scope

The expert cache is a scheduler-level optimization for a routed `GGML_OP_MUL_MAT_ID` operation that already has a host-weight input copied into a non-CPU backend split. It is not a general CPU/GPU MoE dispatcher.

The active slot-pool path can execute an eligible route union directly from a persistent accelerator buffer:

```text
eligible host-weight MUL_MAT_ID in non-CPU split
                  |
                  v
read selected IDs and form unique union
                  |
        +---------+----------+
        |                    |
union fits ready slots   union cannot use slots
        |                    |
        v                    v
explicit ID remap      legacy copied-tensor fallback
slot_tensor source     D2D hit copies plus host-to-device misses
        |                    |
        +---------+----------+
                  |
                  v
            Execute MUL_MAT_ID
```

The slot-pool full-hit path avoids in-band D2D reconstruction. It is available for any eligible batch whose unique expert union fits slots. It does not make a CPU-routed decode operation cache-eligible, and it does not provide CPU-on-miss or mixed CPU/GPU expert execution.

Current miss handling can fill slots immediately for a cache-eligible operation. The cache has full-pool ghost admission and eviction cooldown, but does not have a bounded fill-job queue or an atomic complete-bundle dispatch contract.

The proposed general design moves the CPU/GPU decision until after current route IDs are ready:

```text
current route ready
  -> complete resident bundle: GPU slot-pool execution
  -> any missing or loading member: unchanged CPU execution
                                  + optional bounded fill for later work
```

This design is not implemented. See `superpowers/specs/2026-08-26-general-decode-moe-dispatch-design.md` and `superpowers/plans/2026-08-26-general-decode-moe-dispatch.md`.
---

## 3. Dense FFN Heterogeneous Channel Partitioning

### 3.1 Mathematical Formulation

For SwiGLU / parallel gate-up FFN architectures:

$$\begin{aligned}
x &\in \mathbb{R}^{B \times n_{\text{embd}}} \\
W_{\text{gate}}, W_{\text{up}} &\in \mathbb{R}^{n_{\text{embd}} \times n_{\text{ff}}} \\
W_{\text{down}} &\in \mathbb{R}^{n_{\text{ff}} \times n_{\text{embd}}}
\end{aligned}$$

The intermediate hidden dimension $n_{\text{ff}}$ is partitioned into an accelerator slice $G \in [0, n_{\text{ff\_accel}})$ and a CPU slice $C \in [n_{\text{ff\_accel}}, n_{\text{ff}})$:

$$\begin{aligned}
a_{\text{gpu}} &= \text{SiLU}(x W_{\text{gate,gpu}}) \odot (x W_{\text{up,gpu}}) \\
a_{\text{cpu}} &= \text{SiLU}(x W_{\text{gate,cpu}}) \odot (x W_{\text{up,cpu}}) \\
y_{\text{gpu}} &= a_{\text{gpu}} W_{\text{down,gpu}} \\
y_{\text{cpu}} &= a_{\text{cpu}} W_{\text{down,cpu}} \\
y &= y_{\text{gpu}} + y_{\text{cpu}}
\end{aligned}$$

This transformation produces mathematically identical outputs (modulo floating-point summation order) with no parameter pruning or approximation.

### 3.2 Memory Layout & Repacking

```text
up / gate tensor: [n_embd, n_ff] (Row-major)
┌────────────────────────────────────────────────────────┐
│ GPU Slice: rows 0 .. G-1 (Contiguous in memory)        │ -> Copied directly to VRAM
├────────────────────────────────────────────────────────┤
│ CPU Slice: rows G .. n_ff-1                            │ -> Zero-copy view in host RAM
└────────────────────────────────────────────────────────┘

down tensor: [n_ff, n_embd] (Row-major)
Row 0: [ GPU (cols 0..G-1) | CPU (cols G..n_ff-1) ]
Row 1: [ GPU (cols 0..G-1) | CPU (cols G..n_ff-1) ]
...
Row N: [ GPU (cols 0..G-1) | CPU (cols G..n_ff-1) ]
```

Because column slicing across rows is non-contiguous in row-major memory, `down` is repacked once at load time:
- `down_accel` ($G \times n_{\text{embd}}$): Packed row-by-row into the accelerator buffer.
- `down_cpu` ($(n_{\text{ff}} - G) \times n_{\text{embd}}$): Packed row-by-row into host memory to avoid non-contiguous stride overhead in quantized kernels.

### 3.3 Asynchronous Execution Overlap

To achieve true physical concurrency between CPU and GPU:
1. The GPU branch graph is constructed and submitted first via `ggml_backend_graph_compute_async()`, which launches the kernels onto the GPU stream and returns immediately.
2. The scheduler then computes the CPU branch while the GPU is executing.
3. The merge node `ggml_add(ctx0, gpu, cpu)` acts as the synchronization point.

---

## 4. Hybrid MoE Layers (Simultaneous Dual-Optimization)

Architectures such as **Qwen3.5-MoE**, **DeepSeek-V2**, and **DeepSeek-V3** contain both routed expert banks and dense shared experts within the same transformer layer.

`llama.cpp` evaluates both optimizations concurrently:

```text
                                   MoE Layer
                                       │
                ┌──────────────────────┴──────────────────────┐
                │                                             │
         Routed Experts                                 Shared Expert
          ffn_*_exps                                     ffn_*_shexp
                │                                             │
           Expert Cache                                FFN Partitioner
      (Dynamic VRAM Cache)                        (GPU Prefix + CPU Remainder)
                │                                             │
                └──────────────────── ADD ────────────────────┘
                                       │
                                  Layer Output
```

- **Routed Branch**: Slices loaded dynamically into the Expert Cache based on router selection.
- **Shared Expert Branch**: Partitioned across GPU and CPU according to `--ffn-split`.

---

## 5. Multi-Token Prediction (MTP) Support

Models utilizing Multi-Token Prediction (e.g. DeepSeek-V3/R1, Qwen3.5-MoE next-n draft heads) construct secondary computation graphs containing routed and shared experts.

- **Registry Resolution**: The tensor-keyed registry (`std::unordered_map<const ggml_tensor *, std::unique_ptr<llama_ffn_partition>>`) automatically registers MTP shared-expert tensors (`mtp.*.ffn_up_shexp`).
- **Memory Tracking**: MTP context memory and cache requirements are included in the global VRAM budget calculation.

---

## 6. Unified VRAM Budgeting (`--fit`)

To prevent Out-Of-Memory (OOM) errors during automatic fitting, the VRAM budget is calculated holistically:

$$\text{VRAM}_{\text{model+partitions}} = \text{VRAM}_{\text{total}} - (\text{KV Cache} + \text{Compute Buffers} + \text{MTP Reserve} + \text{Expert Cache Reserve} + \text{Safety Margin})$$

- In `llama_model::memory_breakdown()`, FFN partition accelerator buffers (`buf_accel`) and CPU buffers (`buf_cpu`) are reported under their respective backend types.
- In `llama_context::memory_breakdown()`, `cparams.expert_cache_size` is reserved on the accelerator compute buffer.
- `--fit` uses these measurements to allocate whole layers and FFN partitions without exceeding device limits.

---

## 7. CLI Reference & Usage

### 7.1 Options

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-exc`, `--expert-cache` | `SIZE` | `0` (disabled) | VRAM capacity for caching CPU-offloaded MoE experts (e.g. `1024M`, `1.5G`, `2G`). |
| `-excp`, `--expert-cache-period` | `INT` | `64` | Token interval for periodic JIT expert rebalancing (`0` = on-demand LRU). |
| `-excs`, `--expert-cache-stats` | `FLAG` | `false` | Print detailed hit rate, bandwidth savings, and partition stats on exit. |
| `--ffn-split` | `FLOAT` | `0.0` (disabled) | Fraction of dense FFN intermediate dimension placed on GPU (`0.0` to `1.0`). |

### 7.2 Example Commands

#### Running a Hybrid MoE Model (e.g. Qwen3.5-MoE) with Cache + Shared Expert Slicing
```sh
llama-cli \
    -m models/qwen3.5-moe-q4_k.gguf \
    -p "Explain the difference between TCP and UDP in detail." \
    -ngl 16 \
    --expert-cache 1500M \
    --ffn-split 0.35 \
    --expert-cache-stats
```

#### Running a Dense Model with Heterogeneous FFN Partitioning
```sh
llama-cli \
    -m models/llama-3-8b-q4_k.gguf \
    -p "Write a fast matrix multiplication kernel in C++." \
    -ngl 20 \
    --ffn-split 0.40 \
    --fit on
```

#### Benchmarking Token Generation Throughput
```sh
llama-bench \
    -m models/qwen3.5-moe-q4_k.gguf \
    -n 128 \
    -ngl 12 \
    --expert-cache 2G \
    --ffn-split 0.30 \
    --expert-cache-stats
```

---

## 8. Diagnostic Telemetry Output

When `--expert-cache-stats` is enabled, `llama.cpp` prints detailed metrics upon context destruction:

```text
Expert Cache (CUDA0):
  capacity:              1500.00 MiB
  resident:              1482.50 MiB (530 entries)
  period:                64 tokens
  rebalances:            4
  JIT swaps:             28
  requests:              40960
  hits:                  29184
  misses:                11776
  hit rate:                71.25 %
  RAM -> GPU:               5.88 GiB
  avoided RAM -> GPU:      14.59 GiB
  evictions:                1248

FFN Heterogeneous Partitioning:
  partitioned branches: 28
  accelerator VRAM:      432.18 MiB
```
