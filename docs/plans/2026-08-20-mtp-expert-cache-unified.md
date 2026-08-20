# Unified MTP Dynamic Offload and Expert-Cache Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans task-by-task. Tasks use checkbox syntax for tracking.

**Goal:** Isolate the source of client-visible gibberish, repair confirmed Qwen3.6 dynamic-MTP and expert-cache correctness defects, then improve only measured single-request performance bottlenecks without changing target sampling semantics.

**Architecture:** The target model samples and verifies MTP candidates. The draft MTP context shares the target model, initially keeps its MTP block in host memory, and dynamically relocates it to a contiguous GPU allocation before its first draft decode. The expert cache rewrites selected MoE expert IDs after CPU cache lookup. Therefore the work proceeds in this order: deterministic target/cache/MTP isolation; dynamic relocation and promotion correctness; server batch-order validation; single-request placement/cache tuning; low-risk cache overhead removal; trace-gated backend-ordering and tensor-lifetime changes.

**Tech Stack:** C++17, ggml backend scheduler, CUDA backend, llama-server, Nsight Systems, Qwen3.6-35B-A3B MTP GGUF.

## Global Constraints

- Keep code comments concise and ASCII-only.
- Use a fresh server process, a fixed text-only prompt, fixed seed, `temperature = 0`, `top-k = 1`, and explicit token-ID capture for all correctness comparisons.
- A first differing emitted target token ID is a correctness failure. Do not use rendered text, draft text, acceptance rate, or first-token latency as a substitute.
- Do not alter target sampling, verification, or acceptance unless target-only output proves that boundary faulty.
- Preserve ordinary host-MTP fallback whenever dynamic promotion is unsupported or fails.
- Do not add release-path tensor readbacks, per-tensor synchronizations, or diagnostic allocations.
- Do not change public C APIs or remove `llama_model_mtp_demote_to_host` in this series.
- Keep `ggml_expert_cache_key.tensor` identity and `expert_offset = expert_id * expert_size` unchanged.
- Do not free `cache->tensor`; slot-pool tensors are allocated from that backing buffer.
- Do not remove CUDA event waits or `ggml_backend_synchronize(ids_backend)` without an explicit replacement event protocol and deterministic regression coverage.
- Do not raise or remove the bounded host-registration cap.
- Do not use `--fit` to establish numerical correctness. Fit sizing is verified only after promotion correctness is complete.
- The supplied model has one MTP block, `blk.40`, and lacks `blk.40.nextn.shared_head_head`; its MTP graph falls back to `model.output` at `src/models/qwen35moe.cpp:722-737`.

---

## Phase 1: Establish the fault domain and baseline metrics

### Task 1: Build one reproducible correctness and performance record

**Files:**
- Verify: `build/bin/Release/llama-server.exe`
- Verify: existing expert-cache statistics and server timing output

**Interfaces:**
- Consumes: the supplied MTP GGUF, one fixed prompt, and the user configuration.
- Produces: a record for every matrix row containing the first 32 emitted target token IDs, `draft_n`, `draft_n_accepted`, promotion logs, expert-cache counters, prompt tok/s, generation tok/s, and peak/reserved GPU memory.

- [ ] **Step 1: Freeze the request and baseline environment**

Use one short text-only prompt and a fresh server for every row. Keep model file, CUDA backend, GPU-layer placement, KV types, flash attention, context size, template, threads, and batch values fixed. Set `parallel = 1` for all one-request rows.

Use `temperature = 0`, `top-k = 1`, a fixed seed, and at least 256 generated tokens for performance rows. Capture emitted token IDs before rendering.

- [ ] **Step 2: Run the correctness isolation matrix**

Run these rows in order:

| Row | Speculation | Dynamic MTP | Expert cache | Required result |
|---|---|---:|---:|---|
| A | none | off | 0 | coherent, repeatable target tokens |
| B | none | off | requested cache/profile | token IDs equal A |
| C | draft-mtp | off | 0 | token IDs equal A |
| D | draft-mtp | on | 0 | promotion log present; token IDs equal A |
| E | draft-mtp | on | requested cache/profile | token IDs equal A |

For row D, require both:

```text
MTP dynamic offload enabled: ... staged in host memory
MTP weights promoted to GPU in ... ms (... MiB)
```

An absent promotion line means host-MTP fallback, not a valid dynamic-offload comparison.

- [ ] **Step 3: Apply the stop rules**

```text
A fails -> stop MTP work; debug target model loading, target CUDA, KV, or prompt/template.
B fails -> stop MTP work; debug expert-cache rewriting or host-to-GPU expert transfer.
C fails -> debug static MTP/model integration before dynamic relocation.
D fails -> execute Tasks 2, 4, and 5 before performance work.
E fails while D passes -> execute Task 3 and expert-cache regression work before performance work.
```

Draft acceptance may differ between rows. It is not proof of output corruption because the server emits target-verified tokens.

- [ ] **Step 4: Record the single-request reserve comparison**

After the relevant correctness rows pass, compare one submitted request at `parallel = 1` and `parallel = 2`. Keep every other setting identical.

Require identical token IDs. Retain `parallel = 1` for a one-request deployment if it lowers reserve memory or permits better output-head/MTP placement. `parallel = 2` remains required only for the two-slot regression in Task 6.

- [ ] **Step 5: Capture the initial CUDA trace**

Capture a 256-token one-request trace for row E after it is correct. Record:

```text
MTP promotion duration
selected-expert ID D2H/synchronization time
slot-map H2D bytes and count
target hidden-state D2H bytes
draft input H2D bytes
periodic rebalance duration
```

This trace is the baseline for Tasks 9-12. Do not implement trace-gated work before it exists.

---

## Phase 2: Repair dynamic-MTP correctness and promotion resilience

### Task 2: Initialize quantized allocation padding during MTP promotion

**Files:**
- Modify: `src/llama-model.cpp:2163-2224`
- Test: `tests/test-mtp-dynamic-offload.cpp`
- Verify: `build/bin/Release/llama-server.exe`

**Interfaces:**
- Consumes: `mtp_residency_state::tensors`, with host source, logical size, and aligned GPU offset.
- Produces: GPU-resident MTP tensors with logical host bytes copied and backend-required allocation padding initialized.

- [ ] **Step 1: Add a test-only red diagnostic**

In a CUDA diagnostic build, inspect promoted quantized allocations once before the fix. At least one tensor with `alloc_size > ggml_nbytes(tensor)` must expose nonzero padding. Keep the inspection out of release code.

- [ ] **Step 2: Initialize before copying logical bytes**

In `llama_model::mtp_promote_to_gpu()`, after assigning `data` and `buffer`, call:

```cpp
if (ggml_backend_buffer_init_tensor(pimpl->mtp_state.buf_gpu.get(), info.tensor) != GGML_STATUS_SUCCESS) {
    LLAMA_LOG_ERROR("%s: failed to initialize MTP tensor '%s'\n", __func__, info.tensor->name);
    return false;
}

ggml_backend_tensor_set_async(backend, info.tensor, info.host_data, 0, info.nbytes);
```

Retain one synchronization after the copy loop. Do not synchronize per tensor.

- [ ] **Step 3: Keep residency transactional**

Do not set `is_gpu_resident` until initialization and all copies complete. On initialization failure, restore every touched tensor's captured host `data` and `buffer`, synchronize/free temporary backend state, and return `false`.

- [ ] **Step 4: Verify**

Build and run `test-mtp-dynamic-offload`. Re-run rows C and D. Require no padding diagnostic hit, the dynamic promotion log, and target token IDs equal to row A.

---

### Task 3: Exclude promoted MTP experts from host expert-cache registration

**Files:**
- Modify: `src/llama-context.cpp:613-633`
- Test: `tests/test-expert-cache.cpp`
- Verify: `build/bin/Release/llama-server.exe`

**Interfaces:**
- Consumes: dynamic-MTP residency state and `model.hparams.n_layer()`.
- Produces: host-cache registrations only for trunk host experts.

- [ ] **Step 1: Add the red coverage**

Exercise context construction with dynamic MTP and an enabled expert cache. Record host registration bytes/ranges. Before the fix, MTP gate/up/down expert ranges are registered while still host-resident.

- [ ] **Step 2: Exclude only dynamic MTP layers**

Before accessing `model.layers[il]` in the registration loop:

```cpp
if (model.has_mtp() && il >= (int) model.hparams.n_layer()) {
    continue;
}
```

This must represent current dynamic-MTP state, not merely model NextN metadata. Static MTP remains eligible for its normal registration behavior.

- [ ] **Step 3: Verify**

Re-run rows B and E. Require trunk cache traffic to remain active, MTP-layer host registrations to be absent, promotion to succeed, and emitted IDs to equal row A. Do not add an unregister-after-promotion path.

---

### Task 4: Reject unsupported dynamic-relocation topology as a whole

**Files:**
- Modify: `src/llama-model.cpp:1747-1786`
- Test: `tests/test-mtp-dynamic-offload.cpp`
- Verify: `build/bin/Release/llama-server.exe`

**Interfaces:**
- Consumes: collected MTP relocation candidates.
- Produces: either a complete owned-host relocation set or disabled dynamic promotion with ordinary host-MTP available.

- [ ] **Step 1: Define the supported set**

Every collected tensor must satisfy:

```cpp
t->view_src == NULL
ggml_backend_buffer_is_host(t->buffer)
```

The decision covers all collected MTP tensors; do not relocate a partial set.

- [ ] **Step 2: Implement all-or-nothing fallback**

On the first view or non-host tensor, clear collected relocation entries, disable dynamic MTP, log one concise warning naming the tensor, and retain the original host tensors.

- [ ] **Step 3: Verify both paths**

The supplied Qwen model has no MTP view tensors, so its dynamic path remains enabled. A tensor-buffer override that places one MTP tensor on a non-host backend must disable dynamic promotion and generate correct host-MTP output.

---

### Task 5: Account for deferred MTP bytes and make promotion failure one-shot

**Files:**
- Modify: `src/llama-model.cpp:1057-1074,1710-1786,2163-2224`
- Modify: `src/llama-model.h`
- Modify: `src/llama-ext.h`
- Modify: `common/fit.cpp:55-97,134-142`
- Test: `tests/test-mtp-dynamic-offload.cpp`
- Verify: `build/bin/Release/llama-fit-params.exe`, `build/bin/Release/llama-server.exe`

**Interfaces:**
- Produces: internal `llama_model_mtp_dynamic_gpu_size(const llama_model *)` and a one-shot `promotion_failed` state.
- Consumes: precomputed supported dynamic-MTP allocation size during fit estimation.

- [ ] **Step 1: Precompute supported deferred allocation size**

Before the model loader's `no_alloc` early return, select the same first GPU used by promotion and sum:

```cpp
GGML_PAD(ggml_backend_buft_get_alloc_size(gpu_buft, t), alignment)
```

for every supported MTP tensor. Store only size/device metadata at this stage; add host data pointers only after loaded tensor data exists.

- [ ] **Step 2: Expose a narrow internal accessor**

Add:

```cpp
size_t llama_model::mtp_dynamic_gpu_size() const;
size_t llama_model_mtp_dynamic_gpu_size(const struct llama_model * model);
```

Declare the wrapper in `src/llama-ext.h`, never `include/llama.h`. Return zero unless dynamic collection is enabled and supported.

- [ ] **Step 3: Charge the selected promotion GPU**

After `llama_get_memory_breakdown(ctx)`, add the deferred size to the promotion device's model requirement and print it separately:

```text
deferred MTP promotion buffer = ... MiB
```

Do not change static GPU layer counts.

- [ ] **Step 4: Prevent repeated failures**

Add `bool promotion_failed = false` to `mtp_residency_state`. Set it when GPU buffer allocation, backend initialization, or tensor initialization fails. Restore touched tensor state. Future promotion calls return false without a second allocation attempt or repeated error log.

Do not claim async copy errors are detectable from `ggml_backend_tensor_set_async()`.

- [ ] **Step 5: Verify fit and fallback**

Test deliberately insufficient and sufficient GPU headroom. Require exactly one failure log plus correct host-MTP fallback in the first case; exactly one promotion success in the second; and deferred MTP bytes in fit output.

---

### Task 6: Validate the `parallel = 2` MTP batch-contiguity invariant

**Files:**
- Modify only if reproduced: `tools/server/server-context.cpp`
- Test: existing server integration infrastructure under `tests/unit/`
- Verify: `build/bin/Release/llama-server.exe`

**Interfaces:**
- Consumes: target batches passed to `common_speculative_impl_draft_mtp::process()`.
- Produces: contiguous row ranges per sequence, or a process implementation that groups rows while preserving original output-row indexing.

- [ ] **Step 1: Add a two-slot red test**

Start two concurrent slots with unequal prompt lengths, `parallel = 2`, and draft MTP. At `temperature = 0`, assert each response equals its own `parallel = 1` target-only baseline.

- [ ] **Step 2: Reproduce before editing server code**

In test-only logging, capture `(batch row, seq_id, position)` delivered to MTP processing. Fail if a sequence appears in two disjoint row ranges.

- [ ] **Step 3: Fix only an observed violation**

If interleaving exists, group input rows by sequence before copying shifted hidden states. Preserve one output row per original target token and restore `verify_h` indexing to original batch rows. Otherwise leave server batching unchanged.

- [ ] **Step 4: Verify isolation**

Run the two-slot integration test and re-run one-slot rows C-E. Neither static nor dynamic one-slot streams may change.

---

## Phase 3: Select correct runtime placement and cache policy

### Task 7: Tune only after rows A-E are numerically identical

**Files:**
- Verify: `build/bin/Release/llama-server.exe`
- Verify: expert-cache statistics and CUDA trace from Task 1

**Interfaces:**
- Consumes: a correct row-E baseline.
- Produces: deployment settings for `parallel`, expert-cache capacity, rebalance period, and GPU placement.

- [ ] **Step 1: Verify dynamic MTP and output-head residency**

For every dynamic performance row, require both dynamic-MTP logs from Task 1. Capture layer placement and confirm GPU residency for `output.weight`, which supplies the MTP head for this GGUF. If it remains host-resident, reduce reserve consumers, add GPU headroom, or use static MTP before changing cache code.

- [ ] **Step 2: Find the expert-cache capacity/period knee**

With selected `parallel`, compare:

```text
cram = 1024 MiB, period = 0
cram = 1024 MiB, period = 64
cram = 1024 MiB, period = 256
```

Record `n_hits`, `n_zero_copy_hits`, `n_misses`, `n_evictions`, `n_rebalances`, `bytes_ram_to_gpu`, and generation tok/s.

```text
frequent evictions and low zero-copy hits with spare VRAM -> increase cram
rebalance spikes around multiples of 64 -> prefer period 256
low evictions and no throughput gain -> retain the smaller cache
```

- [ ] **Step 3: Select dynamic or static MTP from measured steady state**

Compare corrected row C and corrected/promoted row D with identical placement and one request. Dynamic offload may save initial VRAM but pays promotion work. Keep the mode with correct token IDs and better measured prompt/generation throughput for the actual prompt distribution.

---

## Phase 4: Implement low-risk expert-cache improvements

### Task 8: Reuse expert-cache scheduler scratch storage

**Files:**
- Modify: `ggml/src/ggml-backend.cpp:776-835,1599-1801`
- Test: `tests/test-expert-cache.cpp`
- Verify: `build/bin/Release/test-expert-cache.exe`, `build/bin/Release/llama-server.exe`

**Interfaces:**
- Consumes: selected IDs, counts, bitsets, cache keys, requested experts, and remapped IDs for one MoE split.
- Produces: scheduler-owned containers valid only during one `ggml_backend_sched_compute_splits()` call.

- [ ] **Step 1: Add cold/warm scheduler coverage**

Add deterministic multi-expert coverage that compares cold and warm output tensor bytes and asserts at least one cold miss plus at least one warm hit or zero-copy hit. It must exercise the scheduler MoE path, not only cache helpers.

- [ ] **Step 2: Add reusable scheduler containers**

Add to `ggml_backend_sched`:

```cpp
std::vector<int32_t> expert_ids_scratch;
std::vector<ggml_bitset_t> expert_bitset_scratch;
std::vector<uint32_t> expert_counts_scratch;
std::vector<int32_t> requested_experts_scratch;
std::vector<ggml_expert_cache_key> pinned_keys_scratch;
std::vector<int32_t> remapped_ids_scratch;
```

Reserve from current tensor dimensions, clear/reuse inside the MoE branch, and do not store asynchronous backend references into this storage. Keep `restored_nodes` split-local.

- [ ] **Step 3: Verify**

Require identical deterministic tokens and expected cache hit/miss order over 256 tokens. With a debug allocation counter around the MoE branch, require no capacity-growth allocation after warmup.

---

### Task 9: Validate and batch expert-cache profile seeding

**Files:**
- Modify: `common/expert-cache-profile.cpp:42-126`
- Modify: profile loading caller in `common/common.cpp:1435-1542`
- Test: `tests/test-expert-cache-profile.cpp`
- Verify: `build/bin/Release/llama-server.exe`

**Interfaces:**
- Consumes: `(tensor_name, expert_id, frequency)` profile rows and loaded model tensors.
- Produces: validated, deduplicated, tensor-grouped seed submissions and one synchronization per backend.

- [ ] **Step 1: Add profile compatibility tests**

Cover valid tensor/ID, unknown tensor, out-of-range expert ID, and duplicate tensor/expert entries. Unknown/out-of-range entries are skipped and counted. Retain the largest frequency for duplicates.

- [ ] **Step 2: Resolve and group once**

Use a local record:

```cpp
struct expert_cache_seed_entry {
    const ggml_tensor * tensor;
    int32_t expert_id;
    uint32_t frequency;
};
```

Resolve names once, validate tensor rank and bounds, deduplicate by `(tensor, expert_id)`, and sort by tensor then expert ID. Do not retain JSON-owned character pointers after profile parsing.

- [ ] **Step 3: Submit and synchronize by backend**

Use existing seed APIs for sorted entries grouped by backend. Synchronize once after each backend's final seed, not per entry.

- [ ] **Step 4: Verify**

With and without the `coder` profile, require identical deterministic token IDs, expected seeded-entry counters, and no regression to first-token latency versus one-entry-at-a-time seeding.

---

### Task 10: Remove device slot-map traffic only if the graph proves it dead

**Files:**
- Modify conditionally: `ggml/src/ggml-backend-expert-cache.cpp:299-308,947-963`
- Modify conditionally: `ggml/src/ggml-backend.cpp:1619-1621`
- Test: `tests/test-expert-cache.cpp`
- Verify: CUDA trace and 256-token server run

**Interfaces:**
- Consumes: host expert-to-slot mapping and direct remapped-ID upload.
- Produces: either existing device-map traffic or a removed device map with equivalent selected-ID rewriting.

- [ ] **Step 1: Prove graph non-consumption**

Build a representative MoE graph and assert no graph leaf or node source uses `pool.d_expert_to_slot_tensor`. Also prove cache hits continue to rewrite and upload `ids_tensor` through `ggml/src/ggml-backend.cpp:1793-1797`.

- [ ] **Step 2: Measure warm-cache traffic**

Record `n_map_updates`, `map_update_bytes`, and CUDA H2D copies to the slot-map allocation. If updates are zero after warmup, do not prioritize removal.

- [ ] **Step 3: Remove only proven-dead state**

If Step 1 passes and traffic is material, remove device slot-map allocation, dirty state, flush function, and callers. Keep host slot lookup and direct remapped-ID upload.

- [ ] **Step 4: Verify**

Require cold/warm equality and 256 deterministic server tokens across four period-64 boundaries, unchanged cache hit/miss behavior, and zero slot-map update bytes.

---

## Phase 5: Trace-gated throughput work

### Task 11: Overlap selected-expert ID transfer only with explicit events

**Files:**
- Modify only after gate: `ggml/src/ggml-backend.cpp:1673-1677`
- Test: `tests/test-expert-cache.cpp`
- Verify: Nsight Systems and deterministic server run

**Interfaces:**
- Consumes: GPU-selected expert IDs.
- Produces: host-visible double-buffered IDs that are read only after their matching transfer event completes.

- [ ] **Step 1: Enforce the trace gate**

Proceed only when the Task 1 trace shows `ggml_backend_tensor_get_async()` plus `ggml_backend_synchronize(ids_backend)` consumes at least 5 percent of one-token decode wall time after Tasks 8-10.

- [ ] **Step 2: Design exact ownership**

Allocate two pinned host ID buffers and two events. For step `n`, GPU writes buffer `n % 2`, records that buffer's event after D2H transfer, CPU waits for that event before lookup, and cannot reuse the buffer until dependent remapped-ID upload is queued.

- [ ] **Step 3: Implement with safe fallback**

Replace unconditional synchronization only when event and pinned-memory support exist. Retain the current synchronous path for unsupported backends.

- [ ] **Step 4: Verify**

Require 256 identical token IDs, equality across four period-64 boundaries, debug detection of no stale ID, reduced synchronization time, and improved generation tok/s.

---

### Task 12: Hand target hidden states to MTP on-device only when transfer-bound

**Files:**
- Modify only after gate: `src/llama-context.cpp:1579-1586`, `common/speculative.cpp:1459-1547`
- Test: MTP server integration test under `tests/unit/`
- Verify: Nsight Systems, CUDA memcheck or AddressSanitizer, and `llama-server.exe`

**Interfaces:**
- Consumes: ordered target `t_h_nextn` rows.
- Produces: same-device MTP handoff with the present host-copy route for CPU and mixed-backend cases.

- [ ] **Step 1: Enforce the transfer trace gate**

With `ubatch-size = 2048`, measure target hidden-state D2H bytes, draft input H2D bytes, and host memcpy time in MTP `process()`. Proceed only if their combined path consumes at least 5 percent of prompt-processing wall time.

- [ ] **Step 2: Add ordering coverage before code**

Compare host and candidate device handoff for one/multiple prompt microbatches, one accepted draft token, and zero accepted draft tokens. Assert equal target token IDs and equal `pending_h` selection after every verification step.

- [ ] **Step 3: Add event-guarded same-device handoff**

Expose target output only after its producing scheduler split completes. Make MTP wait on that event before consuming the tensor. Retain host extraction/batch-input fallback when target and MTP backends differ. Do not expose a temporary tensor that either context can reset before MTP compute completes.

- [ ] **Step 4: Verify performance and teardown**

Require lower traced D2H+H2D traffic, unchanged or improved generation tok/s, and improved prompt tok/s. Exercise context destruction after an in-flight MTP request under CUDA memcheck or AddressSanitizer before enabling by default.

---

## Final Verification and Decision Record

- [ ] Re-run rows A-E after every correctness-affecting change; all emitted token IDs must equal row A.
- [ ] Re-run the `parallel = 2` two-slot regression after Task 6 and after Task 12.
- [ ] Re-run the 256-token cache-period benchmark after each cache-path change; record cache counters and throughput next to the Task 1 baseline.
- [ ] Report promotion outcome, selected runtime mode, cache capacity/period, output-head placement, and each trace gate result.
- [ ] Do not report MTP dynamic offload as the client-visible gibberish root cause unless Phase 1 isolates row D or E as the first failing boundary.

## Explicit Non-Goals

- No sampler or target-verification rewrite without a target-only failure.
- No generic relocation support for all MTP architectures; unsupported topology must use host-MTP fallback.
- No changes justified only by a template mismatch, Q8 KV cache, context shifting, or `spec-draft-n-max = 2` without a reproducer.
- No cache backing-store consolidation, D2D fallback rewrite, or public API cleanup in this plan.

## Plan Self-Review

- Dependency order: Phase 1 isolates the fault before relocation changes; Phase 2 establishes correctness/fallback; Phase 3 chooses correct deployment settings; Phases 4-5 optimize only measured paths.
- Coverage: includes deterministic diagnostics, padding, stale registrations, unsafe topology, deferred fit, promotion failure, parallel batching, placement/cache tuning, scratch reuse, profile seeding, slot maps, ID synchronization, and hidden-state handoff.
- Safety: Tasks 10-12 are evidence-gated; all tensor-lifetime and event-ordering changes have deterministic token-ID checks.
- Existing symbols: `mtp_residency_state`, `has_mtp()`, `hparams.n_layer()`, `ggml_backend_sched`, `ggml_expert_cache_key`, `ggml_tensor`, `llama_context`, and `t_h_nextn` exist. New `promotion_failed` and `expert_cache_seed_entry` are defined in their tasks.
