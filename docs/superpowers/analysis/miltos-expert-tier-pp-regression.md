# miltos22 WackMall Expert-Tier PP Regression Analysis

Status: Phase-zero evidence record (Task 1 of the mapped-host oracle epic).
Scope: Analysis only. This document examines the external WackMall references
to attribute the reported prompt-processing regressions and to fix the reuse
boundary for the mapped-host candidate. It does not change executable code,
CMake, cache behavior, scheduler behavior, fit behavior, MTP, or tests.

## 1. Sources

- WackMall architecture record:
  https://raw.githubusercontent.com/miltos22/llama-wackMall/master/ARCHITECTURE.md
- Research repository: https://github.com/miltos22/llama-wackMall
- Merged-request PR #26563: https://github.com/ggml-org/llama.cpp/pull/26563
  (opened 2026-08-04, closed 2026-08-21)
- Successor merged-request PR #26824: https://github.com/ggml-org/llama.cpp/pull/26824
  (opened 2026-08-10, closed 2026-08-23)

The approved design (`docs/superpowers/specs/2026-08-31-low-vram-moe-mapped-host-design.md`,
lines 39-47) summarizes: "The WackMall tier path places hot experts on GPU and
computes cold experts on CPU. It adds hot-store lookup and cold operators
rather than mapped-host GPU computation. Its merged-request reports identify
severe PP regressions; the research fork limits tiering to small token counts
and leaves larger prompt graphs on the stock path. Reuse its bundle metadata
and admission analysis only. Do not reuse its split hot/cold execution model."

## 2. What WackMall is

PR #26563 overview: "I made a CUDA only for now feature where it tracks a heat
map of all expert usage and the hottest experts are cached on GPU and computed
there, while cold experts are computed on the CPU." ARCHITECTURE.md sec 1:
"This engine tiers experts individually: a small set of hot experts lives in
VRAM, the rest stay RAM-resident and are computed on CPU only when routed."

Mechanism relevant to PP:

- Hot store sentinel trick (ARCHITECTURE.md sec 2): per MoE layer and per
  weight matrix, a `.hot` tensor [ne0, ne1, S+1] holds S hot expert slices plus
  one zero-filled sentinel slot at index S; `lut` int32[N] maps expert to slot
  (or S). The graph is rewritten so `ids_hot = get_rows(lut, ids)`, then the
  stock `mul_mat_id` runs on the `.hot` tensor; cold selections hit the zeroed
  sentinel and contribute exact zero (SWIGLU/GELU zero property), so placement
  changes speed, never logits.
- Cold execution paths (sec 3): `MUL_MAT_ID_COLD` (generic per-matrix CPU op
  with the same I/O layout as stock `mul_mat_id`; computes only experts that
  are both selected and cold, dedups inside the op, writes zeros for hot rows;
  a `ggml_add` merges with the GPU result); `MOE_COLD` (fused 3-phase gated);
  `MOE_COUNT` (count-only).
- TMAX gate (sec 3c): "The tiered kernels engage only for graphs of at most
  TMAX tokens (default 16, the decode regime; larger prompt graphs keep the
  stock path). MOE_COUNT harvests router decisions from large batches into the
  same per-layer count buffer."
- Graph rewrite and merge work: the rewrite adds a LUT indirection
  (`get_rows(lut, ids)`), the `mul_mat_id` on the `.hot` tensor, per-layer cold
  ops, and a `ggml_add` merge; dispatch follows the `src0` buffer (sec 9):
  "hot tensors in CUDA buffers execute on GPU; cold host-pinned tensors stay on
  CPU."
- PR #26563 known limitation: "Tier only engages at n_tokens == 1 (single-token
  decode); multi-slotbatches fall back to stock mul_mat_id."

## 3. PP-regression attribution

The merged-request threads report PP regressions on several hardware
configurations. This document attributes them to four mechanisms present in the
cited sources.

### 3.1 CPU cold path

Cold experts are computed on the CPU (PR #26563 overview; ARCHITECTURE.md sec
1, 3). Prompt batches activate many experts per layer, so the cold CPU workload
grows with batch size; a PR #26563 reviewer states "Putting more MoE layers in
VRAM always wins for PP." PR #26824 comments report PP collapse on multiple
systems: Green-Sky "Ok, seems like prompt processing is ALWAYS done on cpu.
(300 -> 2 tps)" (early version, RTX 2070 8 GB); troed "PP is very low, around
a third of usual"; kabhinara "the prompt processing got divided by 3
essentially (almost 4 actually :/)" (RTX 4070 Laptop 8 GB); Tha14 "That's the
expected outcome of this. You are trading PP for TG speed."

### 3.2 Hot/cold route fragmentation

The sentinel mechanism splits each layer's routes: hot selections execute on
the GPU, cold selections execute on the CPU and are merged back with a
`ggml_add` (ARCHITECTURE.md sec 2-3); dispatch follows the `src0` buffer, so
hot tensors in CUDA buffers execute on GPU while cold host-pinned tensors stay
on CPU (sec 9). A single MoE layer's result therefore requires two backends and
a merge step, fragmenting what stock execution does in one op.

### 3.3 Graph rewrite and merge work

The tier adds a LUT indirection (`get_rows(lut, ids)`), the `.hot`
`mul_mat_id`, per-layer cold ops, and a `ggml_add` merge (ARCHITECTURE.md sec
2-3). A PR #26563 reviewer (AMD 7950X, Intel Arc Pro B70, 32 GB, DeepSeek-V4-
Flash-0731 IQ2XXS) measured the cost of the accompanying heatmap accounting
during PP: "Updating the cache heatmap during PP is a waste of resources and
slows down PP... it also slows down prefill", with pp2048 at 86.60 t/s while
updating during PP versus 101.75 t/s with updates disabled; the same reviewer
notes "Caching experts vs caching MoE layers is always going to be a tradeoff
between PP and TG." Those numbers were measured on different hardware and are
reported here only as evidence of mechanism, not as GTX 1080 predictions.

### 3.4 TMAX decode gate

The tiered kernels engage only for graphs of at most TMAX tokens (default 16);
larger prompt graphs keep the stock path while `MOE_COUNT` still harvests
router decisions from large batches (ARCHITECTURE.md sec 3c). PR #26563 states
the tier engages only at `n_tokens == 1`. The design spec (line 41) records
that the research fork limits tiering to small token counts and leaves larger
prompt graphs on the stock path. The consequence for PP: prompt graphs never
execute on the hot tier, yet still incur the counting work, and (in the
merged-request version) the heatmap accounting described in 3.3.

## 4. Post-closure status: explicitly unknown

PR #26563 was closed on 2026-08-21; the author's closing note states "This has
been closed with plans to organize and re-open... I have redesigned about half
of the entire system in ways that fix all major issues exposed by this pr."
PR #26824 was closed on 2026-08-23. No cited source records the final state of
the redesigned system. Specifically unknown, and not claimed here: whether the
redesign eliminated the PP regressions attributed in section 3; the final TMAX
behavior and batch policy; multi-GPU behavior; and any performance figures of
the post-closure fork. All mechanism statements above are anchored to
ARCHITECTURE.md (updated 2026-07-29) and the two PR threads at their closure
dates.

## 5. Benchmark applicability

The reported speedups and PP regressions were measured on hardware other than
the GTX 1080 target:

- PR #26563: the author's tables cover Qwen3.6-35B-A3B (Q2_M 11 GB, Q5_K_P
  28 GB) and other models; a commenter measured on an RTX 2000-series laptop
  ("Nvidia rtx 2000 in a laptop", 31.51 t/s stock vs 26.46 t/s with -ehs -1).
- PR #26824 methodology: "Hardware: 8gb laptop 3070 + 32 GB of ram + laptop
  ssd."
- ARCHITECTURE.md sec 10: "RTX 3070 8 GB VRAM, 31 GB RAM, SSD model."
- PR #26563 reviewer benchmark: AMD 7950X + Intel Arc Pro B70 (SYCL),
  DeepSeek-V4-Flash-0731.

None of these is a GTX 1080 (Pascal, sm_61). This document does not claim that
any WackMall benchmark value applies to the GTX 1080; the only experiment that
may inform the target hardware is the mapped-host oracle benchmark (section 7).

## 6. Reuse boundary for the mapped-host candidate

Reuse: bundle metadata and admission analysis only (design spec line 41). Do
not reuse: the split hot/cold execution model, the sentinel graph rewrite, the
cold CPU operators, or TMAX gating. The candidate measures direct mapped-host
GPU reads with zero timed expert-weight H2D (design spec lines 55-59, 150);
WackMall's model is explicitly the opposite (hot-store lookup and cold
operators rather than mapped-host GPU computation, design spec line 40). PP
remains on the stock batched path until mapped-host `MUL_MAT_ID` has
demonstrated preserved batch semantics (design spec line 154).

## 7. First executable experiment

The first executable experiment of this epic is `tests/test-moe-mapped-host-bench.cpp`
(it does not exist in the tree yet; the design mandates adding it beside the
current MoE oracle targets, design spec line 72). It records CPU, VRAM, and
mapped-host storage modes at TG1, TG4, TG8, PP128, and PP512 with 100 warmups
and 1,000 timed iterations, requires real Qwen3.6 APEX Compact GGUF tensors
(no synthetic weights), and enforces a zero timed expert-weight H2D invariant
(design spec lines 53-57, 74-85, 150). The current sidecar and native fallback
remain untouched even if the benchmark succeeds (design spec lines 61-68, 155).
No copy-on-route cache and no CPU/GPU partial executor is proposed as this
experiment (design spec lines 59, 134).

## 8. Conclusion

This document and `docs/superpowers/analysis/moe-low-vram-execution-options.md`
both identify the same isolated mapped-host benchmark as the sole next
executable step.
