# Expert Cache: Phase-Aware Optimization & Next Ideas

This document captures the adversarial analysis and evaluation of phase-aware expert caching (Prompt Processing vs. Token Generation) for heterogeneous CPU/GPU Mixture-of-Experts (MoE) inference in llama.cpp.

---

## 1. Adversarial Analysis: Prompt Processing (PP) vs. Token Generation (TG)

### Why Swapping Experts Specifically for PP is an Anti-Pattern

#### 1. The Expert Dispersion Problem: PP Touches Almost *All* Experts
* **In TG ($N=1$)**: Each step routes to only 8 experts out of 64. Over a 64-token generation window, high semantic locality means the same 12-18 specialized experts are hit repeatedly ($>70\%$ hit rate).
* **In PP ($N=512$ or $N=2048$)**: In a single 512-token prompt, the router makes $512 \times 8 = 4,096$ routing decisions per layer. Across 512 diverse vocabulary tokens, **virtually all 64/64 experts in every layer are activated**.
* **The Math**: A 256 MiB cache can only hold ~15% of the total experts across 64 layers. 
  - If PP needs 100% of the experts, pre-loading 15% into cache still results in an **85% miss rate**.
  - PP would still have to stream the other 85% over PCIe anyway.

#### 2. Compute-Bound vs. Memory-Bound Dynamics
* **TG is Memory-Bandwidth Bound**: The GPU matrix engines sit idle waiting for weights. Zero-Copy D2D slot execution is transformative here because eliminating PCIe transfers directly increases tok/s.
* **PP is Compute-Bound**: With large batch sizes ($M=512$), matrix multiplication arithmetic intensity is high ($512 \times \text{FLOPs/byte}$). The CUDA compute engine spends significant time on GEMM math, allowing bulk PCIe transfers to be overlapped asynchronously in the background via compute splits.
* **The Swapping Penalty**: Swapping 256 MB before PP starts costs ~15-25 ms of pipeline stall. For a 512-token prompt running at 480 tok/s (~1,060 ms total), pre-swapping adds pure latency without increasing throughput.

#### 3. The "Cache Poisoning" Problem on Token 1 of TG
* If PP were allowed to populate or swap the cache based on the broad, generic token distribution of the whole prompt, it would **evict the conversational context's hot experts**.
* When TG starts, tokens 1-32 would suffer a massive **cold-start storm of cache misses** while re-learning what experts the generation topic needs, degrading early generation speed.
* *(This is why Vector 6--freezing slot pool mutations during PP--boosted prompt processing to 481+ tok/s while preserving 27.25 tok/s in generation).*

---

## 2. Refined Phase-Aware Opportunities

While full PP swapping is counter-productive, three refined mechanisms offer tangible speedups:

### Vector A: "Prompt-Tail Seeding" (Warming Token 1 Before Generation Starts)
**Status: DEFERRED**

* **The Concept**: Don't track all 512 prompt tokens. Instead, observe only the **last 32-64 tokens** of the prompt (the user's immediate question or code block).
* **Why it works**: The vocabulary and semantic domain of the *last few tokens* of the prompt have near-100% correlation with the *first few tokens* of generation.
* **Mechanism**:
  1. During PP, ignore tokens $0 \dots (N - 64)$.
  2. For the final 64 prompt tokens, record expert routing hits into a dedicated staging buffer.
  3. On the final prompt forward pass, promote the top-8 most frequent prompt-tail experts directly into the TG slot pool.
  4. **Result**: Token 1 of TG starts with an immediate **100% hit rate** instead of 0% cold-start latency.

### Vector B: Universal "Core Anchor" Pinning (System Prompt & Grammar Experts)
**Status: DEFERRED**

* **The Concept**: Certain experts are responsible for syntax (punctuation, code indentation, JSON formatting, basic English grammar). These experts are activated by *every single prompt and every single generation*.
* **Mechanism**:
  - Keep a tiny static protected partition in SLRU (e.g., top 4-6 experts across all layers, ~32 MB).
  - Pin them permanently so neither PP nor diverse TG topics can ever evict them.
  - **Result**: Guaranteed zero-overhead baseline across all queries.

### Vector C: Asynchronous Pre-H2D Pipeline Overlap at PP $\to$ TG Transition
**Status: DEFERRED**

* **The Concept**: While the final layer of PP is executing its logits/sampling kernel on the GPU, trigger the host CPU to asynchronously kick off the first PCIe DMA transfers for Token 1's suspected Layer 0/1 experts.
* **Result**: Hides the initial PCIe transfer of Token 1 completely inside the GPU's final logit reduction kernel.

---

## 3. Summary Matrix

| Strategy | Feasibility | Expected Performance Impact | Recommendation | Status |
|---|---|---|---|---|
| **Full Cache Swap at Start/End of PP** | Low | **Negative** (stalls pipeline, destroys TG locality) | **Avoid** | IMPLEMENTED: PP bypass is default |
| **Separate Frequency Trackers (PP vs TG)** | High | **Neutral** (adds tracking overhead with little actionable gain) | **Skip** | IMPLEMENTED: `tg_access_freq` + `pp_access_freq` exist but unused for admission |
| **Prompt-Tail Seeding (Last 32-64 tokens)** | High | **Positive** (eliminates TG cold-start miss storm on Token 1) | **Recommended Next Vector** | DEFERRED |
| **Static Core Anchor Pinning** | Medium | **Positive** (guaranteed hits for syntax/grammar experts) | **Recommended Next Vector** | DEFERRED |

---

## 4. Adversarial Review of 10 Architectural Proposals

### 1. Pipeline expert transfers one or more layers ahead ($L+1$ / $L+2$)
**Status: REJECTED**

* **Concept**: Transfer missing experts for $L+1$ in a background copy stream while computing layer $L$.
* **Critique**: In a causal autoregressive Transformer, **layer $L+1$'s routing IDs cannot be known until layer $L$ finishes computing**. The input to router $L+1$ is the hidden state output of layer $L$. True multi-layer lookahead is mathematically impossible without speculative approximation (which saturated PCIe and regressed performance to 23.48 tok/s).
* **Viable Derivative**: **Intra-layer overlap** -- initiating miss transfers immediately after layer $L$'s router kernel finishes, overlapping with attention normalization and residual math.

### 2 & 3. GPU-Side Cache Lookup & ID Remapping (`d_expert_to_slot` in CUDA kernel)
**Status: REMOVED (2026-08-20)**

* **Concept**: Maintain an 8 KB GPU-resident lookup table `int16_t expert_to_slot[64][64]` and resolve slot indices directly in the CUDA `MUL_MAT_ID` kernel.
* **Critique**: **Extremely High Merit**. Currently, the CPU scheduler intercepts `MUL_MAT_ID`, transfers `ids_tensor` (32 bytes), computes slot assignments, and rewrites the graph on every layer and every token. In steady-state generation, 70-80% of tokens are 100% hits. Moving lookup to GPU L1/constant memory eliminates all CPU scheduler interception and host synchronization on hits.
* **Resolution**: The device slot map was removed on 2026-08-20. The graph now consumes explicitly uploaded `ids_tensor` remap. The GPU-side lookup concept remains a valid future optimization but requires re-implementation if pursued.

### 4. Projection-Aware Residency (Don't treat `{gate, up, down}` equally)
**Status: REJECTED**

* **Concept**: `down_proj` is 2x larger than `gate` or `up`. Keep `down` on CPU host and cache only `gate` and `up` on GPU.
* **Critique**: **Fatal Flaw**. If `gate/up` run on GPU and `down` runs on CPU, the intermediate activation tensor must be copied GPU $\to$ CPU (D2H transfer), the CPU computes `down`, and the output is copied CPU $\to$ GPU (H2D transfer). This adds **two synchronous PCIe activation stalls per layer**, destroying throughput. `{gate, up, down}` must stay co-resident on the same compute device.

### 5. Layer-Weighted Slot Budgeting ($V_i = \text{stall\_ns\_saved} / \text{VRAM\_bytes}$)
**Status: IMPLEMENTED (as value-per-byte rebalancing)**

* **Concept**: Middle layers (layers 12-48 in 64-layer models) have much higher expert routing concentration and stability than early or late layers. Allocate 24 slots to middle layers and 6 slots to outer layers instead of uniform 14 slots.
* **Critique**: **Strong Merit**. Increases global cache hit rate by 10-15% within the exact same 256 MiB VRAM budget with zero runtime copy cost.
* **Resolution**: Implemented as V2 Component 3: Empirical Global Value-per-Byte Rebalancing. The cache computes $\text{value} = \frac{\text{hits} \times \text{size}}{\text{alloc\_size}}$ per slot pool and rebalances dynamically.

### 6 & 7. Direct Page DMA via `cudaHostRegister`
**Status: IMPLEMENTED**

* **Concept**: Page-lock un-offloaded host MoE weights with `cudaHostRegister`, enabling direct `cudaMemcpyAsync` without CPU `memcpy` into the 16-slot staging buffer.
* **Critique**: **High Merit**. Eliminates the CPU memory bus copy completely on misses. Must be scoped to CPU-offloaded MoE weight tensors to avoid OS non-paged pool exhaustion on Windows.
* **Resolution**: Implemented as `registered_host_ranges` with 1 GiB safety cap in `ggml_backend_expert_cache`.

### 8. Batch Misses into Contiguous Transfers
**Status: REJECTED (Neutral/Marginal)**

* **Concept**: Combine non-contiguous expert misses into a single large DMA transfer.
* **Critique**: **Neutral / Marginal**. Packing non-contiguous host weights requires a CPU `memcpy` pack step that takes more CPU time than PCIe command launch overhead (~5 us) for 2.5 MB blocks.

### 9 & 10. Victim Cache & TinyLFU Admission
**Status: IMPLEMENTED (as SLRU)**

* **Concept**: 2-level victim buffer and Count-Min sketch frequency filter.
* **Critique**: **Diminishing Returns**. SLRU (probationary/protected segments) already captures >90% of admission filtering with $O(1)$ overhead in MoE decode without reducing main cache capacity.
* **Resolution**: Implemented as `probationary_cap` / `protected_cap` split in slot pools with ghost sightings + eviction cooldown.

---

## 5. Current Implementation Status (2026-08-26)

### Implemented
- All 5 original vectors (Zero-Copy Slot Pool, Universal Batch, Pinned DMA, Admission Hysteresis, Atomic Bundling)
- V2: Direct Page DMA (`registered_host_ranges`)
- V2: Value-per-Byte Rebalancing
- V2: Partial Rebalance (`-excm`)
- MTP dynamic offload correctness fixes (`380f9af17`)

### Removed
- Device slot map (`d_expert_to_slot`) -- removed 2026-08-20

### Deferred (Not Yet Implemented)
- Prompt-Tail Seeding (Vector A)
- Core Anchor Pinning (Vector B)
- Async Pre-H2D Pipeline Overlap (Vector C)
- GPU-side routing table (re-implementation)

### Rejected
- Full PP cache swapping (anti-pattern)
- Multi-layer lookahead (mathematically impossible)
- Projection-aware residency (fatal flaw)
- Batched miss transfers (neutral/marginal)

---

## 6. Performance Verdict for GTX 1080 / Compact Profile

Benchmark sweep shows `-exc 0` (cache disabled) is the throughput control for Qwen3.6-35B-A3B-APEX-Compact on GTX 1080. No cache configuration beats the control beyond measurement noise. The cache remains useful for VRAM-pressure fitting via `--fit auto` but is not a throughput accelerator on this hardware profile.

For models with larger expert weights or higher PCIe bandwidth targets, the implemented vectors (zero-copy slot pools, pinned DMA, direct page DMA, value-per-byte rebalancing) provide a solid foundation. The deferred vectors (prompt-tail seeding, core anchor pinning) represent the highest-impact remaining opportunities.
