# MTP and Expert-Cache Performance Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans task-by-task. Tasks use checkbox syntax for tracking.

**Goal:** Improve single-request Qwen3.6 MTP throughput and latency by removing measured expert-cache overhead, improving profile warmup, and pursuing GPU handoff work only when traces prove it dominates.

**Architecture:** This plan does not modify MTP numerical behavior, model weights, cache keys, or expert byte offsets. It first establishes a deterministic single-request baseline with MTP promotion confirmed. The low-risk implementation work reuses scheduler-side host scratch storage and batches profile uploads. Higher-payoff changes to selected-expert synchronization and target-to-MTP hidden-state transfer remain trace-gated because they cross CPU/GPU ordering and tensor-lifetime boundaries.

**Tech Stack:** C++17, ggml scheduler, CUDA backend, Nsight Systems, llama-server, Qwen3.6-35B-A3B MTP GGUF.

## Global Constraints

- Complete the dynamic-MTP correctness plan before this plan. In particular, promoted quantized padding must be initialized and MTP host expert registrations must be excluded.
- Use fixed prompt, seed, `temperature = 0`, and at least 256 generated tokens for every before/after comparison.
- Compare emitted token IDs, not rendered text. The first differing ID is a correctness failure.
- Keep `ggml_expert_cache_key.tensor` identity and `expert_offset = expert_id * expert_size` unchanged.
- Do not remove CUDA event waits or `ggml_backend_synchronize(ids_backend)` without a replacement event protocol proved by a deterministic regression test.
- Do not raise or remove the bounded host-registration cap.
- Do not free `cache->tensor` as a "legacy" allocation. Slot-pool tensors are carved from it.
- Do not use `--fit` for baseline correctness comparisons. Confirm the dynamic MTP promotion log before measuring performance.

---

### Task 1: Establish a single-request performance baseline and choose runtime settings

**Files:**
- Verify: `build/bin/Release/llama-server.exe`
- Verify: existing expert-cache statistics and server timing output

**Interfaces:**
- Consumes: one fixed coding prompt and the supplied Qwen3.6 MTP model.
- Produces: JSON or text records containing emitted token IDs, prompt tok/s, generation tok/s, promotion latency, draft acceptance, and expert-cache counters for each configuration.

- [ ] **Step 1: Prove promoted dynamic MTP is active**

Start a fresh server with one active request and require:

```text
MTP dynamic offload enabled: ... staged in host memory
MTP weights promoted to GPU in ... ms (... MiB)
```

If promotion is absent or fails, stop. Measure/fix deferred GPU headroom first; host-MTP fallback is not a dynamic-offload performance baseline.

- [ ] **Step 2: Compare one active sequence against reserved two-sequence capacity**

Run the same request twice:

```text
A: parallel = 1
B: parallel = 2, only one request submitted
```

Keep model placement, context size, cache settings, seed, and prompt identical.

Expected:

```text
identical emitted token IDs
parallel = 1 has lower reserve memory
```

Keep `parallel = 1` for a single-request deployment unless concurrent slots are required. This changes reserved state, not the active request's sampling semantics.

- [ ] **Step 3: Find the cache period and capacity knee**

For the selected parallel value, run:

```text
cram = 1024 MiB, period = 0
cram = 1024 MiB, period = 64
cram = 1024 MiB, period = 256
```

Record `n_hits`, `n_zero_copy_hits`, `n_misses`, `n_evictions`, `n_rebalances`, `bytes_ram_to_gpu`, and generation tok/s.

Selection rule:

```text
high rebalances or latency spikes near token multiples of 64 -> prefer 256
low zero-copy hits plus frequent evictions and spare VRAM -> increase cram
low evictions and no tok/s gain from more cram -> keep 1024 MiB
```

- [ ] **Step 4: Confirm MTP output-head placement**

This GGUF lacks `blk.40.nextn.shared_head_head`, so the MTP graph uses `model.output` at `src/models/qwen35moe.cpp:722-737`. Capture model-load placement logs and confirm `output.weight` is GPU-resident. If it is host-resident, lower reserve consumers or increase GPU layer placement before changing expert-cache code.

---

### Task 2: Reuse expert-cache scheduler scratch storage

**Files:**
- Modify: `ggml/src/ggml-backend.cpp:776-835,1599-1801`
- Test: `tests/test-expert-cache.cpp`
- Verify: `build/bin/Release/llama-server.exe`

**Interfaces:**
- Consumes: selected expert IDs, per-expert counts, cache keys, bitsets, and remapped IDs created while processing one MoE split.
- Produces: scheduler-owned reusable containers that are valid only during one `ggml_backend_sched_compute_splits()` invocation.

- [ ] **Step 1: Add deterministic expert-cache coverage before changing allocation behavior**

Extend `tests/test-expert-cache.cpp` with a cold-cache then warm-cache decode fixture that uses multiple expert IDs and asserts:

```cpp
assert(cold_output == warm_output);
assert(cold_cache_stats.n_misses > 0);
assert(warm_cache_stats.n_hits + warm_cache_stats.n_zero_copy_hits > 0);
```

Use deterministic input activations and compare output tensor bytes. The test must exercise the scheduler MoE path, not only direct cache helper APIs.

- [ ] **Step 2: Add scheduler-owned reusable containers**

Add the following C++ containers to `struct ggml_backend_sched`:

```cpp
std::vector<int32_t> expert_ids_scratch;
std::vector<ggml_bitset_t> expert_bitset_scratch;
std::vector<uint32_t> expert_counts_scratch;
std::vector<int32_t> requested_experts_scratch;
std::vector<ggml_expert_cache_key> pinned_keys_scratch;
std::vector<int32_t> remapped_ids_scratch;
```

Reserve capacity from the current tensor dimensions before filling each container. Clear and reuse storage rather than constructing local vectors in the MoE branch of `ggml_backend_sched_compute_splits()`.

- [ ] **Step 3: Preserve split-local lifetime rules**

`restored_nodes` remains split-local because it restores graph source pointers after compute. Do not store pointers or references into scratch containers in backend work queued after the current split. Upload expert IDs before mutating the next split's scratch state.

- [ ] **Step 4: Verify behavior and allocator reduction**

Run:

```text
cmake --build build --config Release --target test-expert-cache llama-server
build/bin/Release/test-expert-cache.exe
```

Then generate 256 deterministic tokens with cache period 64. Require identical token IDs and cache counters within expected hit/miss ordering. Add a debug-only allocation counter around the MoE branch and require zero capacity-growth allocations after warmup.

---

### Task 3: Batch and validate expert-cache profile seeding

**Files:**
- Modify: `common/expert-cache-profile.cpp:42-126`
- Modify: the profile load caller in `common/common.cpp:1435-1542`
- Test: `tests/test-expert-cache-profile.cpp`
- Verify: `build/bin/Release/llama-server.exe`

**Interfaces:**
- Consumes: profile entries of `(tensor_name, expert_id, frequency)` and loaded model tensors.
- Produces: validated, tensor-grouped seed batches submitted once per backend, with one synchronization after all submits for that backend.

- [ ] **Step 1: Add profile compatibility tests**

Extend `tests/test-expert-cache-profile.cpp` with entries for:

```text
existing tensor and valid expert ID       -> accepted
unknown tensor name                       -> skipped and counted
expert ID >= tensor->ne[2]                -> skipped and counted
same tensor/expert twice                  -> highest frequency retained
```

Assert that accepted entries preserve the existing profile scoring semantics and rejected entries do not seed cache storage.

- [ ] **Step 2: Resolve, deduplicate, and group entries before upload**

Create a local load record:

```cpp
struct expert_cache_seed_entry {
    const ggml_tensor * tensor;
    int32_t expert_id;
    uint32_t frequency;
};
```

Resolve model tensor names once. Validate tensor rank and expert bounds before adding a record. Deduplicate by `(tensor, expert_id)`, retaining the largest frequency. Sort accepted records by tensor pointer then expert ID.

- [ ] **Step 3: Submit grouped seeds and synchronize once**

Submit each sorted entry through the existing seed API, grouped by backend. Do not retain raw `const char *` pointers from JSON strings after the profile object is released. Synchronize once after the final submitted seed for each backend, not after each entry.

- [ ] **Step 4: Verify first-request and warmed behavior**

Run profile tests, then compare a fixed request with and without the `coder` profile:

```text
same deterministic token IDs
profiled run has no more first-token latency than one-entry-at-a-time seeding
profiled cache counters show the expected seeded entries
```

---

### Task 4: Measure slot-map traffic and remove it only if the current graph never consumes it

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.cpp:299-308,947-963`
- Modify: `ggml/src/ggml-backend.cpp:1619-1621`
- Test: `tests/test-expert-cache.cpp`
- Verify: CUDA trace from `llama-server.exe`

**Interfaces:**
- Consumes: host expert-to-slot state and remapped expert IDs.
- Produces: either the existing device slot-map path, or no device slot-map allocation/upload when the current scheduler always uploads remapped IDs directly.

- [ ] **Step 1: Add a graph-consumption assertion**

Before removing slot maps, add a test that builds a representative MoE graph and asserts no graph leaf or node source equals `pool.d_expert_to_slot_tensor`. The test must also verify that cache hits still rewrite `ids_tensor` through the existing upload at `ggml/src/ggml-backend.cpp:1793-1797`.

- [ ] **Step 2: Capture warm-cache map metrics**

Generate 256 tokens and record:

```text
n_map_updates
map_update_bytes
CUDA H2D copies with the slot-map allocation as destination
```

If `n_map_updates == 0` after warmup, do not prioritize this task; it has no steady-state payoff.

- [ ] **Step 3: Remove dead device-map state only after Step 1 passes**

If no graph consumes the device map, remove `d_expert_to_slot_tensor`, `map_dirty`, the allocation branch, and `ggml_backend_expert_cache_flush_slot_maps()` calls. Retain host slot lookup and explicit remapped-ID upload.

- [ ] **Step 4: Verify cache correctness**

Run the cold/warm scheduler test and a 256-token deterministic server request over at least four period-64 boundaries. Require identical token IDs, unchanged cache hit/miss behavior, and zero map update bytes.

---

### Task 5: Trace-gate selected-expert synchronization overlap

**Files:**
- Modify only after the trace gate passes: `ggml/src/ggml-backend.cpp:1673-1677`
- Verify: Nsight Systems trace and deterministic server run

**Interfaces:**
- Consumes: device-generated selected expert IDs.
- Produces: a host-visible ID buffer guarded by an explicit completion event before CPU cache lookup.

- [ ] **Step 1: Capture the current critical path**

Trace 256 single-token decode steps. Count `ggml_backend_tensor_get_async()` calls and the elapsed time under `ggml_backend_synchronize(ids_backend)` at `ggml/src/ggml-backend.cpp:1673-1677`.

Proceed only if this synchronization consumes at least 5 percent of decode wall time after Tasks 2-4.

- [ ] **Step 2: Specify event ownership before implementation**

The design must allocate two pinned host ID buffers and two backend events. For step `n`:

```text
GPU writes IDs to buffer[n % 2]
backend records event[n % 2] after the D2H copy
CPU waits for that exact event before reading buffer[n % 2]
CPU does not reuse the buffer until its dependent expert-ID upload is queued
```

No implementation may reuse an ID buffer based only on a decode counter.

- [ ] **Step 3: Implement only after event design review**

Replace the unconditional synchronize with the event protocol from Step 2. Keep the current synchronous path available when the backend lacks events or pinned host memory.

- [ ] **Step 4: Verify ordering and payoff**

Require all of:

```text
identical deterministic token IDs for 256 tokens
identical results across at least four period-64 boundaries
no stale expert ID reported by a debug assertion
Nsight shows lower synchronization time and higher generation tok/s
```

---

### Task 6: Trace-gate GPU-resident target-to-MTP hidden-state handoff

**Files:**
- Modify only after the trace gate passes: `src/llama-context.cpp:1579-1586`, `common/speculative.cpp:1459-1547`
- Test: MTP server integration test added under `tests/unit/`
- Verify: Nsight Systems trace and `llama-server.exe`

**Interfaces:**
- Consumes: target `t_h_nextn` rows and their output order.
- Produces: a same-device handoff to the MTP context, with host-copy fallback for CPU and mixed-backend execution.

- [ ] **Step 1: Measure the current transfer volume**

Trace prompt processing with `ubatch-size = 2048` and generation with one-token decode. Measure:

```text
target hidden-state D2H bytes
draft embedding H2D bytes
host memcpy time in MTP process()
```

Proceed only if the combined transfer path consumes at least 5 percent of prompt-processing wall time.

- [ ] **Step 2: Write the ordering regression test**

Add a fixed-prompt MTP integration test that compares static host handoff with the new handoff for:

```text
one prompt microbatch
multiple prompt microbatches
one accepted draft token
zero accepted draft tokens
```

Assert identical target token IDs and identical `pending_h` selection after every verification step.

- [ ] **Step 3: Implement a device handoff with host fallback**

Expose the target graph result only after its scheduler has completed the producing split. Make the MTP graph consume that same-device tensor through an event dependency. If target and draft do not use the same backend device, retain the current host extraction and host batch input path.

Do not share a temporary tensor whose allocator may be reset by either context before MTP compute completes.

- [ ] **Step 4: Verify throughput and lifetime safety**

Run the integration test, then compare prompt tok/s and output token IDs on the real model. Require lower traced D2H plus H2D traffic and no regression in generation tok/s. Test context destruction after an in-flight MTP request under AddressSanitizer or CUDA memcheck before enabling the path by default.

---

## Explicit Non-Goals

- Do not duplicate dynamic-MTP padding initialization, view fallback, expert-registration cleanup, deferred fit sizing, or promotion-failure behavior from `docs/plans/2026-08-20-mtp-dynamic-offload-fixes.md`.
- Do not remove `cache->entries`, collapse cache backing storage, or rewrite D2D fallback behavior in this plan.
- Do not make `parallel = 2` a correctness hypothesis for a one-request reproduction.
- Do not optimize template rendering, KV quantization, or context shifting without a profile proving they dominate.

## Plan Self-Review

- Scope coverage: Tasks 1-6 cover the runtime recommendations, scheduler allocation churn, profile startup cost, slot-map traffic, selected-expert synchronization, and target-to-MTP transfer overhead.
- Safety gates: Tasks 5 and 6 require trace evidence before implementation because they change backend ordering or tensor lifetime.
- Overlap: correctness and deferred-fit work remain in the revised dynamic-offload plan; this plan consumes their verified result.
- Type consistency: `ggml_backend_sched`, `ggml_backend_expert_cache_key`, `ggml_tensor`, and `llama_context` are existing types. `expert_cache_seed_entry` is defined in Task 3.
