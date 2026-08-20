# MTP Dynamic Offload Investigation and Correctness Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans task-by-task. Tasks use checkbox syntax for tracking.

**Goal:** Prove the source of client-visible gibberish, then fix confirmed dynamic-MTP correctness defects without conflating draft-quality failures with target-model corruption.

**Architecture:** Qwen3.6 draft MTP uses a separate MTP context sharing the target model. Dynamic offload loads the MTP block on host, then mutates its tensor residency to a contiguous GPU buffer before the first draft decode. The server samples and verifies candidates using target logits, so MTP draft corruption normally reduces acceptance rather than changing emitted target tokens. The plan first separates target, expert-cache, static-MTP, dynamic-MTP, and parallel-batching behavior. It then applies the proven relocation fixes with explicit fallback behavior.

**Tech Stack:** C++17, ggml backend scheduler, CUDA backend, llama-server, Qwen3.6-35B-A3B MTP GGUF.

## Global Constraints

- Keep code comments concise and ASCII-only.
- Do not modify the target sampling or acceptance algorithm until the diagnostic matrix proves it is the faulty boundary.
- Do not remove or change public C APIs as part of this correctness series.
- Preserve a correct host-resident MTP fallback if promotion cannot run.
- Do not add permanent GPU readback, per-tensor synchronization, or logging allocations solely for diagnosis.
- Use fixed prompt, seed, `temperature = 0`, and a fresh server process for comparisons.
- Do not use `--fit` while isolating correctness. It omits the deferred MTP GPU allocation from its estimate.

---

## Evidence and Scope

### Observed source facts

1. `llama_model::mtp_promote_to_gpu()` assigns each collected tensor a GPU address and copies `ggml_nbytes()`, but does not call `ggml_backend_buffer_init_tensor()`:
   - `src/llama-model.cpp:2163-2224`
   - CUDA uses that initializer to zero quantized allocation padding: `ggml/src/ggml-cuda/ggml-cuda.cu:761-765`.

2. Dynamic collection is name-based (`blk.<MTP-layer>.*`) and currently relocates view tensors independently:
   - `src/llama-model.cpp:1747-1786`.

3. The MTP draft context registers host-resident MTP experts with the expert cache before promotion:
   - `src/llama-context.cpp:613-633`.
   - Host registrations persist in the cache until destruction: `ggml/src/ggml-backend-expert-cache.cpp:985-1005`.

4. The server emits the target-sampled accepted sequence, not raw MTP draft candidates:
   - target sampling and exact-match comparison: `common/sampling.cpp:678-705`;
   - server emission after verification: `tools/server/server-context.cpp:3845-3953`.

5. MTP `process()` requires every sequence's target-batch tokens to be contiguous. The code records only a first and last row:
   - `common/speculative.cpp:1437-1454, 1476-1480`.

6. The inspected model is `qwen35moe` with one MTP block (`blk.40`). It has no dedicated `blk.40.nextn.shared_head_head`; the MTP graph falls back to `model.output`:
   - MTP graph fallback: `src/models/qwen35moe.cpp:722-737`.

7. `--fit-target 256` reserves only 256 MiB per device, while dynamic MTP intentionally excludes the later promotion buffer from fit:
   - argument meaning: `common/arg.cpp:2939-2961`;
   - dynamic fit behavior: `common/fit.cpp:139-142`.

### Consequence

Unzeroed dynamic MTP weights are a confirmed correctness defect and must be fixed. They do not alone prove the source of client-visible gibberish, because target verification should reject bad draft tokens. The first task is therefore a decision gate, not an optional final validation.

---

### Task 1: Isolate the failing boundary with deterministic server runs

**Files:**
- Modify: `docs/plans/2026-08-20-mtp-dynamic-offload-fixes.md`
- Verify: `build/bin/Release/llama-server.exe`

**Interfaces:**
- Consumes: the supplied Qwen3.6 MTP GGUF and the existing server configuration.
- Produces: one recorded result per matrix row containing first 32 token IDs, `draft_n`, `draft_n_accepted`, promotion log lines, and expert-cache statistics.

- [ ] **Step 1: Create a fixed reproduction input**

Use one short text-only prompt, fixed seed, `temperature = 0`, `top-k = 1`, one response, and `parallel = 1`. Keep all model, KV, CUDA, context, thread, and template settings fixed. Start a fresh server for every matrix row.

- [ ] **Step 2: Run the target-only baseline**

Run with `spec-type = none` and expert cache size `0`.

Expected:

```text
draft_n = 0
draft_n_accepted = 0
first 32 token IDs are coherent and repeat on a second fresh run
```

If this row emits gibberish, stop. The fault is outside MTP dynamic offload. Investigate target model loading, target CUDA execution, KV cache, or template/input handling before modifying MTP code.

- [ ] **Step 3: Isolate expert-cache numerical behavior**

Repeat Step 2 with the user's expert cache size, period, and `coder` profile enabled.

Expected:

```text
target-only token IDs match Step 2
expert-cache statistics show requests only after MoE work
```

If this row differs from Step 2, stop MTP work. The active fault domain is expert-cache graph rewriting or host-to-GPU expert transfers.

- [ ] **Step 4: Compare static and dynamic MTP without expert cache**

Run these two rows with expert cache size `0`:

```text
C: spec-type = draft-mtp, mtp-dynamic-offload = off
D: spec-type = draft-mtp, mtp-dynamic-offload = on
```

For D, require both log lines before considering it a dynamic run:

```text
MTP dynamic offload enabled: ... staged in host memory
MTP weights promoted to GPU in ... ms
```

If the promotion line is absent or reports allocation failure, the run is a host-MTP fallback and cannot diagnose promoted weights. Increase GPU headroom first; `--fit-target 256` is not sufficient evidence that the deferred MTP buffer fits.

- [ ] **Step 5: Test the full requested feature combination**

Repeat D with expert cache enabled. Then repeat it with `parallel = 2` and two concurrent requests with unequal prompt lengths.

Expected:

```text
parallel = 1 and parallel = 2 emit coherent target tokens
draft acceptance may differ, but target-only output must not become gibberish
```

If only the `parallel = 2` row fails, inspect the MTP batch contiguity assumption before changing CUDA relocation.

- [ ] **Step 6: Record the decision**

Use this table:

| Result | Decision |
|---|---|
| Target-only, cache-off fails | Leave MTP unchanged; debug target execution first. |
| Target-only cache-on fails | Debug expert cache before MTP promotion. |
| Static MTP works, dynamic MTP cache-off fails | Implement Tasks 2-4. |
| Dynamic MTP cache-off works, cache-on fails | Implement Task 3 and expert-cache-specific regression coverage. |
| Only parallel 2 fails | Implement Task 6 before changing MTP weights. |

---

### Task 2: Zero initialized padding before copying promoted weights

**Files:**
- Modify: `src/llama-model.cpp:2163-2224`
- Verify: `build/bin/Release/llama-server.exe` and `build/bin/Release/test-mtp-dynamic-offload.exe`

**Interfaces:**
- Consumes: `mtp_residency_state::tensors`, where each entry has a host data pointer, logical byte count, and aligned GPU offset.
- Produces: GPU-resident MTP tensors whose logical bytes equal their host source and whose backend-required quantized padding is initialized.

- [ ] **Step 1: Define the red check**

With Task 1 row D, run a CUDA diagnostic build that inspects the promoted padding once outside the production path. Before the fix, at least one quantized promoted tensor with `alloc_size > ggml_nbytes(tensor)` must contain nonzero bytes. Do not add this readback loop to release behavior.

- [ ] **Step 2: Initialize each tensor allocation**

In `llama_model::mtp_promote_to_gpu()`:

```cpp
info.tensor->data   = gpu_ptr;
info.tensor->buffer = pimpl->mtp_state.buf_gpu.get();

if (ggml_backend_buffer_init_tensor(pimpl->mtp_state.buf_gpu.get(), info.tensor) != GGML_STATUS_SUCCESS) {
    LLAMA_LOG_ERROR("%s: failed to initialize MTP tensor '%s'\n", __func__, info.tensor->name);
    return false;
}

ggml_backend_tensor_set_async(backend, info.tensor, info.host_data, 0, info.nbytes);
```

Keep the existing single synchronization after the loop. Do not synchronize or allocate a host vector per tensor.

- [ ] **Step 3: Preserve transactional residency**

Do not set `is_gpu_resident = true` until initialization and all copies have completed. If initialization fails, restore every tensor touched in this call to its captured `host_data` and `host_buffer`, synchronize/free the temporary backend, and return `false`.

- [ ] **Step 4: Verify**

Run:

```text
cmake --build build --config Release --target llama-server test-mtp-dynamic-offload
build/bin/Release/test-mtp-dynamic-offload.exe
```

Then re-run Task 1 rows C and D. The dynamic row must show promotion, coherent emitted tokens, and no padding diagnostic hit.

---

### Task 3: Exclude dynamically promoted MTP experts from host cache registration

**Files:**
- Modify: `src/llama-context.cpp:613-633`
- Verify: `build/bin/Release/llama-server.exe`

**Interfaces:**
- Consumes: model dynamic-MTP residency state and `model.hparams.n_layer()`.
- Produces: expert-cache registrations for trunk host experts only; dynamically promoted MTP experts are not registered as host memory.

- [ ] **Step 1: Define the red check**

Run Task 1 row E with debug expert-cache statistics. Record the host-registration count/bytes before first MTP promotion. The current behavior includes MTP-layer gate/up/down expert ranges because the MTP layer starts on host.

- [ ] **Step 2: Gate registration on dynamic-MTP state**

Before accessing `model.layers[il]` in the expert-cache registration loop:

```cpp
if (model.has_mtp() && il >= (int) model.hparams.n_layer()) {
    continue;
}
```

This condition applies only to the current dynamic-MTP state. It must not exclude static MTP models solely because they have NextN metadata.

- [ ] **Step 3: Verify cache behavior**

Re-run Task 1 rows B and E. Confirm:

```text
target trunk expert-cache statistics remain active
MTP promotion succeeds
MTP-layer host registrations are absent
emitted target tokens remain coherent
```

Do not add an unregister-after-promotion path. New contexts must avoid the stale registration from the start.

---

### Task 4: Make dynamic collection safe for views and non-host overrides

**Files:**
- Modify: `src/llama-model.cpp:1747-1786`
- Verify: `build/bin/Release/llama-server.exe`

**Interfaces:**
- Consumes: collected MTP tensor candidates.
- Produces: either a complete host-resident relocation set with no views, or dynamic promotion disabled while ordinary host MTP remains available.

- [ ] **Step 1: Define the supported topology**

Dynamic relocation is supported only when every collected MTP tensor satisfies both conditions:

```cpp
t->view_src == NULL
ggml_backend_buffer_is_host(t->buffer)
```

Do not add a synthetic unit test that only tests booleans. The relevant behavior depends on real ggml view and buffer ownership. Use the real-model and GPU-override integration checks below.

- [ ] **Step 2: Reject unsafe collection as a whole**

While collecting MTP tensors:

```cpp
if (t->view_src != NULL || !ggml_backend_buffer_is_host(t->buffer)) {
    pimpl->mtp_state.tensors.clear();
    pimpl->mtp_state.enabled = false;
    LLAMA_LOG_WARN("%s: dynamic MTP offload disabled: tensor '%s' is not an owned host tensor\n", __func__, t->name);
    break;
}
```

Do not skip only the unsafe tensor. Partial relocation creates mixed aliasing and is not a defined dynamic-MTP mode.

- [ ] **Step 3: Verify supported and fallback behavior**

Run the real Qwen3.6 model. It has no MTP view tensors, so promotion must remain enabled. Run a second configuration that places an MTP tensor on a non-host backend through a tensor-buffer override. It must log the fallback and execute correct host-MTP behavior rather than relocate a partial tensor set.

---

### Task 5: Protect promotion failure and include deferred MTP bytes in fit

**Files:**
- Modify: `src/llama-model.cpp:1057-1074,1710-1786,2163-2224`
- Modify: `src/llama-model.h`
- Modify: `src/llama-ext.h`
- Modify: `common/fit.cpp:55-97,134-142`
- Verify: `build/bin/Release/llama-fit-params.exe` and `build/bin/Release/llama-server.exe`

**Interfaces:**
- Produces: `size_t llama_model_mtp_dynamic_gpu_size(const llama_model * model)` in `src/llama-ext.h`.
- Consumes: the model's precomputed deferred MTP allocation size during `common_get_device_memory_data_impl`.
- Produces: one failed promotion attempt per model instance and a fit estimate that includes the deferred MTP allocation on the device selected for promotion.

- [ ] **Step 1: Compute deferred size before the no-allocation return**

`common_get_device_memory_data_impl()` loads the model with `no_alloc = true`, while current MTP collection runs after the `ml.no_alloc` early return. Split dynamic-MTP sizing from data-bearing collection:

```cpp
// Before `if (ml.no_alloc) return true;`
// Select the same first GPU device used by promotion.
// Sum GGML_PAD(ggml_backend_buft_get_alloc_size(gpu_buft, t), alignment)
// for every supported MTP tensor.
```

Store this size and selected GPU device in `mtp_residency_state`. Only create relocation entries with `host_data` after model data has been loaded.

- [ ] **Step 2: Add a narrow internal fit accessor**

Add these declarations and implementations:

```cpp
size_t llama_model::mtp_dynamic_gpu_size() const;
size_t llama_model_mtp_dynamic_gpu_size(const struct llama_model * model);
```

The accessor returns `0` unless dynamic MTP collection is enabled and supported. Keep it in `src/llama-ext.h`; do not add a public C API declaration in `include/llama.h`.

- [ ] **Step 3: Charge deferred bytes to the promotion device**

After reading `llama_get_memory_breakdown(ctx)`, locate the first GPU in `devs` selected by dynamic promotion and add `llama_model_mtp_dynamic_gpu_size(model)` to that device's `mb.model`. Print it separately:

```text
deferred MTP promotion buffer = ... MiB
```

Do not inflate the static layer count. The allocation is deferred in time but required before MTP generation.

- [ ] **Step 4: Prevent repeated failed promotion**

Add a boolean to `mtp_residency_state`:

```cpp
bool promotion_failed = false;
```

Set it only when GPU-buffer allocation, backend initialization, or tensor initialization fails. Restore every tensor touched in the failed initialization path to `host_data` and `host_buffer`. Future promotion calls return `false` without retrying allocation or repeating the error log. `ggml_backend_tensor_set_async()` has no error return; do not claim that asynchronous copy failure can be detected here.

- [ ] **Step 5: Verify failure and sizing**

Run once with deliberately insufficient GPU headroom and once with sufficient headroom.

Expected:

```text
insufficient: one promotion failure log, continued correct host-MTP fallback
sufficient: one promotion success log, no repeated allocation attempts
fit output: deferred MTP bytes are included in the selected GPU requirement
```

---

### Task 6: Test the `parallel = 2` MTP batching invariant

**Files:**
- Modify: `tools/server/server-context.cpp` only if a batch-order defect is reproduced
- Test: existing server integration test infrastructure under `tests/unit/`
- Verify: `build/bin/Release/llama-server.exe`

**Interfaces:**
- Consumes: target batches provided to `common_speculative_impl_draft_mtp::process()`.
- Produces: batches where each sequence's rows are contiguous, or a MTP process implementation that groups rows by sequence before constructing its draft batch.

- [ ] **Step 1: Write the server-level red test**

Start two slots with unequal prompt lengths and issue generation requests concurrently with `parallel = 2` and `draft-mtp`. The test must capture token IDs and assert that both responses match the corresponding `parallel = 1` target-only baselines at `temperature = 0`.

- [ ] **Step 2: Reproduce before changing code**

Log the `(batch row, seq_id, position)` triples sent to MTP `process()` only in the test build. Fail if a sequence appears in more than one disjoint row range.

- [ ] **Step 3: Fix only if the invariant fails**

If server batching interleaves a sequence, group rows by sequence before the shifted hidden-state copy. Preserve one output row per original target token and update `verify_h` indexing to the original batch row. Do not modify this code if the server already guarantees contiguous runs.

- [ ] **Step 4: Verify**

Run the new two-slot integration test and Task 1 row F. Confirm no change to the one-slot static or dynamic MTP token stream.

---

## Explicit Non-Goals

- Do not remove `llama_model_mtp_demote_to_host` in this series. It is public API cleanup, not evidence-driven gibberish remediation.
- Do not alter sampler acceptance logic without a target-only or target-verification failure.
- Do not treat a template mismatch, Q8 KV cache, context shifting, or `spec-draft-n-max = 2` as the primary cause without a reproducer.
- Do not add generic relocation support for every MTP architecture. Keep this repair correct for the current Qwen3.6 MTP path and fall back safely for unsupported tensor topology.

## Plan Self-Review

- Scope coverage: Tasks 1-6 cover the measured failure boundary, padding initialization, expert-cache lifecycle, unsafe relocation topology, fit/promotion failure, and two-slot batching.
- Known-plan changes: removes unrelated public API deletion; replaces runtime padding readback with test-only diagnostics; replaces partial view skipping with whole-mode fallback; adds mandatory target-only and cache-only controls.
- Type consistency: `mtp_residency_state`, `promotion_failed`, `has_mtp()`, `hparams.n_layer()`, and `total_gpu_size` are existing names except the explicitly defined new boolean.
- No unresolved implementation may be reported as fixing client-visible gibberish until Task 1 identifies the failing row.