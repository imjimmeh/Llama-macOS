# Heterogeneous Inference: MoE Expert Residency & Dense FFN Partitioning

This document describes the design, architecture, and usage of **MoE Heterogeneous Expert Residency** and **Dense FFN Channel Partitioning** in `llama.cpp`.

---

## 1. Overview & Architecture

When running large modern LLMs on memory-constrained consumer accelerators (e.g. 8 GB - 16 GB VRAM), standard layer-level offloading faces two distinct computational bottlenecks:

1. **Mixture-of-Experts (MoE) Models**: Only a small subset of experts (e.g. 8 out of 64 or 128) are activated per token. Streaming cold expert weights over PCIe on every miss stalls the GPU decode pipeline.
2. **Dense Exhaustive FFNs / Shared Experts**: Full dense projections (in dense models or shared-expert branches of MoE models) require evaluating the entire layer. When a layer does not fit entirely in VRAM, running it on CPU underutilizes GPU compute and VRAM.

To address both bottlenecks, `llama.cpp` implements two complementary heterogeneous execution domains:

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
     FFN Partitioning                                    MoE Expert Residency
  (GPU Prefix + CPU Rest)                             (Concurrent GPU Hot + CPU Cold)
             │                                                     │
             └──────────────────────────┬──────────────────────────┘
                                        │
                                   Layer Output
```

---

## 2. MoE Heterogeneous Expert Residency (`--moe-hot-vram`, `--moe-resident-fraction`)

### 2.1 The Inverted Execution Paradigm

Traditional offloading treats host RAM as cold storage and attempts to stream missed weights to the GPU on demand. At decode batch size 1, matrix-vector multiplications on missed experts are memory-bandwidth-bound rather than compute-bound.

Heterogeneous Expert Residency inverts this model:
- **Host RAM is a first-class compute store**: Cold experts execute directly on CPU threads from host memory bandwidth (40-60 GB/s).
- **GPU VRAM is an accelerator for resident hot experts**: The most frequently routed experts reside permanently in VRAM slots and execute on CUDA at full GPU memory bandwidth (300-1000+ GB/s).
- **Misses are ALWAYS CPU compute**: Non-resident experts **never** trigger synchronous PCIe weight transfers during decode.
- **Vectors cross PCIe, not weights**: Only tiny token activation vectors ($[1 \times n_{\text{embd}}]$, ~4-8 KB) cross the bus to merge branch outputs.

```text
                                  Input Activation (x)
                                           │
                                  Host / Device Router
                                           │
                          ┌────────────────┴────────────────┐
                          │                                 │
                 Resident Expert IDs              Non-Resident Expert IDs
                          │                                 │
                    Async Launch                      Immediate Launch
                   CUDA MUL_MAT_ID                     CPU MUL_MAT_ID
               (Fast VRAM Bandwidth)             (Fast Host RAM Bandwidth)
                          │                                 │
                          │   concurrent parallel execute   │
                          │                                 │
                          └────────────────┬────────────────┘
                                           │
                                      Merge (ADD)
                                           │
                                      Layer Output
```

### 2.2 Dynamic Slot Lookup & Graph Construction

Each MoE layer maintains an accelerator-resident lookup table `expert_to_slot_table` (`[1, n_expert_total]`):
- If expert $e$ is resident in VRAM slot $s \in [0, N_{\text{accel}}-1]$, the table entry is $s$.
- If expert $e$ resides in host RAM, the entry is $-1.0$.

During graph construction:
1. `looked_up_slots = ggml_get_rows(ctx0, slot_table, selected_experts_1d)` queries slot assignments.
2. Step masks compute branch weights:
   - $is\_gpu = \text{step}(\text{looked\_up\_slots} + 0.5)$
   - $is\_cpu = 1.0 - is\_gpu$
   - $gpu\_weights = weights \times is\_gpu$
   - $cpu\_weights = weights \times is\_cpu$
3. GPU branch evaluates resident experts with clamped slot indices against packed accelerator tensors (`gate_exps_accel`, `up_exps_accel`, `down_exps_accel`).
4. CPU branch evaluates non-resident experts against host tensors with `cpu_weights`.
5. Outputs merge via `out = ggml_add(ctx0, gpu_branch, cpu_branch)`.

### 2.3 Periodic EMA Frequency Tracking & Hysteresis

Expert routing patterns shift dynamically across topics. The `llama_moe_tracker` maintains runtime telemetry and balances the residency set:

1. **Exponential Moving Average (EMA) Scoring**:
   $$S_e = S_e \cdot \alpha + \text{accesses}_{\text{epoch}}$$
2. **Periodic Rebalancing**: Every $N$ tokens (configured by `--moe-rebalance-period`, default `64`), the tracker identifies:
   - The hottest non-resident candidate: $C = \operatorname{argmax}_{e \notin \text{resident}}(S_e)$
   - The coldest resident expert: $R = \operatorname{argmin}_{e \in \text{resident}}(S_e)$
3. **Hysteresis Promotion Threshold**: To prevent thrashing from minor frequency fluctuations, candidate $C$ is promoted to slot $S_R$ only if:
   $$S_C > S_R + \Delta_{\text{hysteresis}}$$
4. **Decoupled Background Promotion**:
   `promote_expert(candidate_id, slot_id)` updates tensor slices on the device memory buffer and updates `expert_to_slot_table` between tokens without interrupting active compute streams.

---

## 3. Dense FFN Heterogeneous Channel Partitioning (`--ffn-split`)

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

This transformation produces mathematically identical outputs with no parameter pruning or approximation.

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
- `down_accel` ($G \times n_{\text{embd}}$): Packed row-by-row into accelerator VRAM.
- `down_cpu` ($(n_{\text{ff}} - G) \times n_{\text{embd}}$): Packed row-by-row into host RAM to maintain continuous stride in quantized CPU kernels.

---

## 4. Comparison: MoE Heterogeneous Residency vs. Legacy Expert Cache

| Feature | Heterogeneous Residency (`--moe-hot-vram`) | Legacy Expert Cache (`-exc`, `--expert-cache`) |
| :--- | :--- | :--- |
| **Execution Paradigm** | Inverted dual-branch compute (Concurrent GPU + CPU) | Streaming cache (CPU weights transferred to GPU) |
| **Handling of Misses** | **Executed on CPU directly from host RAM** | **Transferred over PCIe to GPU on critical path** |
| **PCIe Traffic on Miss** | **0 bytes weight data** (only ~4-8 KB activation vector) | Entire expert weight matrix (~10-50+ MB per miss) |
| **GPU Decode Bubbles** | **Zero stalls** | GPU waits for PCIe DMA completion on every miss |
| **Dynamic Rebalance** | Periodic background EMA promotion with hysteresis | On-demand LRU / JIT block replacement |
| **Recommended For** | **All MoE models on PCIe-limited systems (e.g. PCIe 3.0/4.0)** | High-bandwidth PCIe 5.0 systems with excess bandwidth |

> [!NOTE]
> The `-exc` / `--expert-cache` flags remain functional in the CLI parser for backwards compatibility, but `--moe-hot-vram` and `--moe-resident-fraction` are strongly recommended for maximum throughput.

---

## 5. CLI Reference & Usage

### 5.1 Options

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--moe-hot-vram` | `SIZE` | `0` (disabled) | Total VRAM capacity dedicated to caching hot resident MoE experts (e.g. `1024M`, `2G`, `2048MB`). |
| `--moe-resident-fraction` | `FLOAT` | `0.0` (disabled) | Fraction of MoE experts to place in VRAM per layer (`0.0` to `1.0`, e.g. `0.35`). |
| `--moe-rebalance-period` | `INT` | `64` | Token interval between dynamic EMA expert rebalancing promotions. |
| `--moe-stats` | `FLAG` | `false` | Print detailed MoE heterogeneous residency telemetry on exit. |
| `--ffn-split` | `FLOAT` | `0.0` (disabled) | Fraction of dense FFN intermediate dimension placed in VRAM (`0.0` to `1.0`). |
| `-exc`, `--expert-cache` | `SIZE` | `0` (disabled) | *(Legacy)* VRAM cache capacity for streaming scheduler-level expert copies. |
| `-excp`, `--expert-cache-period` | `INT` | `64` | *(Legacy)* Token interval for legacy expert cache swapping. |
| `-excs`, `--expert-cache-stats` | `FLAG` | `false` | *(Legacy)* Print legacy expert cache statistics on exit. |

---

### 5.2 Example Commands

#### 1. Running a ~35B MoE Model on an 8GB GPU (GTX 1080 / RTX 3070 / RTX 4060)
```powershell
llama-cli `
    -m models/qwen3.6-35b-moe.Q4_K_M.gguf `
    -ngl 99 `
    --moe-hot-vram 2G `
    --moe-rebalance-period 64 `
    --moe-stats `
    -t 8 `
    -p "Explain quantum computing in simple terms."
```

#### 2. Running a Hybrid MoE Model (Routed MoE + Dense Shared Experts)
```powershell
llama-cli `
    -m models/qwen3.5-moe-27b.Q4_K_M.gguf `
    -ngl 99 `
    --moe-resident-fraction 0.35 `
    --ffn-split 0.40 `
    --moe-stats `
    -t 8
```

#### 3. Running a Dense Model with Heterogeneous FFN Partitioning
```powershell
llama-cli `
    -m models/llama-3-8b.Q4_K_M.gguf `
    -ngl 22 `
    --ffn-split 0.40 `
    -t 8
```

---

## 6. Diagnostic Telemetry Output

When `--moe-stats` is enabled, `llama.cpp` prints detailed telemetry on exit:

```text
--- MoE Heterogeneous Expert Residency Telemetry ---
  Total tokens processed:       512
  Resident VRAM hits:           1428 (69.7%)
  Non-resident CPU misses:      620 (30.3%)
  Background expert promotions: 14
  Zero PCIe weight transfers on decode critical path.
----------------------------------------------------
```
