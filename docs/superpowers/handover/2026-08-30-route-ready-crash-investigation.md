# Route-Ready CPU-Base Fallback Crash Investigation - Handover

**Date**: 2026-08-30
**Branch**: feat/expert-cache-only
**Hardware**: GTX 1080 + Ryzen 7 5700X + Qwen3.6-35B-A3B-APEX-Compact
**Repro command**: `llama-server -m ...gguf -c 8192 -t 14 -b 1024 -ub 512 -fa on -ctk q8_0 -ctv q8_0 -ngl 99 -ncmoe 1 -exc 128M -excp 32 --no-expert-cache-persist --port 8081`
Then: `curl http://127.0.0.1:8081/completion -H 'Content-Type: application/json' --data-raw '{"prompt":"Explain how transformer attention works, step by step.","n_predict":64,"temperature":0,"top_k":1,"seed":42,"cache_prompt":false}'`

---

## 1. Bugs Found and FIXED (3)

### Bug 1: Route-ID Producer Stream Not Synchronized
**File**: `ggml/src/ggml-backend.cpp` (route_ready dispatch, ~line 3030)
**Symptom**: `memcpy` AV in `ggml_backend_expert_cache_rebalance` reading GPU-resident IDs via stale CPU pointer.
**Root cause**: Cross-split route-ready dispatch read IDs from a tensor produced on a different GPU backend without synchronizing that backend first. The env gate `GGML_ROUTE_READY_SYNC_PRODUCER` existed but was not unconditional.
**Fix**: Made `ggml_backend_synchronize(sched->backends[producer_backend_id])` unconditional for cross-split ID reads. Removed the `GGML_ROUTE_READY_SYNC_PRODUCER` env gate.
**Test**: `test_route_ready_cross_split_sidecar` now asserts `test_synchronize_calls >= 1` on the producer backend.

### Bug 2: GPU-Resident Expert Cache Seeding
**File**: `ggml/src/ggml-backend-expert-cache.cpp` (`ggml_backend_expert_cache_seed()`, ~line 2007)
**Symptom**: `memcpy` AV in rebalance when persistent profile loaded GPU-resident MoE weight tensors into the host-frequency map.
**Root cause**: `ggml_backend_expert_cache_seed()` accepted any tensor with `data != NULL` regardless of buffer type. GPU-resident tensors have device pointers that cannot be host-memcpy'd.
**Fix**: Added `ggml_expert_cache_tensor_is_host()` helper (checks `buffer_is_host` or `buffer == NULL`). Applied to both `record_access_count()` and `seed()`. GPU-resident tensors are now rejected at admission.
**Test**: `test_rebalance_ignores_gpu_resident_weights` now asserts `!ggml_backend_expert_cache_seed(cache, weights, 0, 1)` for GPU-allocated weights.

### Bug 3: Non-Host Weight Admission
**File**: `ggml/src/ggml-backend-expert-cache.cpp` (`ggml_backend_expert_cache_record_access_count()`, ~line 763)
**Root cause**: Same invariant as Bug 2 but for the access-recording path. A GPU-resident weight tensor could enter `tg_access_freq` and later trigger a host memcpy during rebalance.
**Fix**: Uses the same `ggml_expert_cache_tensor_is_host()` guard.

---

## 2. Remaining Bug: CPU-Base Fallback Produces Wrong Results

### Symptom
With cache enabled (even without persistent profile), the server returns garbled output. Example with `n_predict=64`:
```
"Here's a thinking process:\n\n1.  **Understand User User User User User User User User User User User User**\n   - The user**\n   - The user**\n   ..."
```
Compared to cache-off which returns coherent text:
```
"Transformer attention, specifically **Scaled Dot-Product Attention**, is the core mechanism..."
```

### Diagnostic Evidence
A comparison probe (`GGML_ROUTE_READY_FALLBACK_VERIFY=1`) was added to the CPU-base fallback path in `ggml_backend_sched_compute_splits()`. It:
1. Runs the CPU-down computation (the fallback path)
2. Runs the canonical down node via `ggml_graph_view` on the split backend
3. Computes `max_abs` difference between the two

**Result**: `[ROUTE_READY_FALLBACK_VERIFY] layer=0 max_abs=255.0176`

This is a massive divergence at the very first MoE layer, meaning the CPU-computed down projection is completely wrong.

### Telemetry Confirms Route-Ready is Active
```
expert cache = no requests (840 MUL_MAT_ID inputs, 0 eligible ops, 0 capacity bypasses, 21 CPU backend bypasses, 819 non-host bypasses)
expert route census = 1800 nodes, 600 plans, 45 CPU-host, 0 non-CPU-host, 1755 non-host
```
- 0 cache requests = no cache engagement (all weights are GPU-resident due to `-ngl 99`)
- 600 plans = 600 MoE layers with bundle plans
- 45 CPU-host plans = the route-ready fallback path handles these
- The fallback path IS executing, and producing garbage

### Hypothesized Root Cause (UNFIXED)
The CPU-base fallback at `ggml/src/ggml-backend.cpp:3195-3296` creates three clone tensors:

```c
struct ggml_tensor cpu_src1 = *down->src[1];  // activation clone
cpu_src1.data = sched->cpu_sched_act_x.data();
cpu_src1.buffer = NULL;

struct ggml_tensor cpu_ids = *down->src[2];   // IDs clone
cpu_ids.data = cpu_id_vals.data();
cpu_ids.buffer = NULL;

struct ggml_tensor cpu_out = *down;            // output clone
cpu_out.data = sched->cpu_sched_down_out.data();
cpu_out.buffer = NULL;
```

The `*down->src[1]` copy preserves all metadata including `nb[]` strides. If the original tensor is a view with non-contiguous strides (e.g., `nb[1]` is larger than `ne[0] * sizeof(float)`), the CPU MUL_MAT_ID kernel (`ggml_compute_forward_mul_mat_id` in `ggml-cpu.c`) will index into the flat buffer using those stale strides, producing wrong memory offsets.

The CPU kernel indexes activation data as:
```c
const char * src1_col = (const char *) wdata +
    (src1_cont || src1->type != vec_dot_type
    ? (i11 + i12*ne11)*row_size      // contiguous path
    : (i11*nb11 + i12*nb12));         // non-contiguous path
```

If `ggml_is_contiguous(cpu_src1)` returns false (because original `nb[]` != contiguous layout), the kernel uses `nb11*nb12` strides on a flat buffer -> wrong data -> wrong computation -> `max_abs=255`.

### The Fix (NOT YET APPLIED)
Reset `nb[]` strides on all three clone tensors to contiguous layout:

```c
// After: struct ggml_tensor cpu_src1 = *down->src[1];
cpu_src1.data = sched->cpu_sched_act_x.data();
cpu_src1.buffer = NULL;
cpu_src1.view_src = NULL;
// ADD: make contiguous
cpu_src1.nb[0] = ggml_type_size(cpu_src1.type);
for (int d = 1; d < GGML_MAX_DIMS; d++) {
    cpu_src1.nb[d] = cpu_src1.nb[d-1] * cpu_src1.ne[d-1];
}

// After: struct ggml_tensor cpu_ids = *down->src[2];
cpu_ids.data = cpu_id_vals.data();
cpu_ids.buffer = NULL;
cpu_ids.view_src = NULL;
// ADD: make contiguous
cpu_ids.nb[0] = sizeof(int32_t);
for (int d = 1; d < GGML_MAX_DIMS; d++) {
    cpu_ids.nb[d] = cpu_ids.nb[d-1] * cpu_ids.ne[d-1];
}

// After: struct ggml_tensor cpu_out = *down;
cpu_out.data = sched->cpu_sched_down_out.data();
cpu_out.buffer = NULL;
cpu_out.view_src = NULL;
// ADD: make contiguous
cpu_out.nb[0] = sizeof(float);
for (int d = 1; d < GGML_MAX_DIMS; d++) {
    cpu_out.nb[d] = cpu_out.nb[d-1] * cpu_out.ne[d-1];
}
```

**Verify**: After this fix, `max_abs` should drop to ~0 (within float rounding). Then verify with the full 64-token request that output matches the cache-off baseline.

---

## 3. Files Changed in This Session

### `ggml/src/ggml-backend-expert-cache.cpp`
- Added `ggml_expert_cache_tensor_is_host()` static helper (host buffer admission check)
- Applied to `record_access_count()` and `seed()` entry guards
- **Status**: PERMANENT, TESTED, GREEN

### `ggml/src/ggml-backend.cpp`
- Made route-ID producer sync unconditional (removed `GGML_ROUTE_READY_SYNC_PRODUCER` env gate)
- Removed temporary diagnostic envs (`GGML_ROUTE_READY_SKIP_ID_REWRITE`, `GGML_ROUTE_READY_VERIFY`)
- **Status**: PERMANENT, TESTED, GREEN
- **ACTIVE DIAGNOSTIC**: `GGML_ROUTE_READY_FALLBACK_VERIFY` env still in place at line ~3269 (temporary, should be removed after fix)

### `tests/test-expert-cache.cpp`
- Added `require(!ggml_backend_expert_cache_seed(...))` assertion for GPU-resident tensor
- Added `test_synchronize_calls >= 1` assertion in cross-split test
- **Status**: PERMANENT, TESTED, GREEN

---

## 4. What Was Tried and Did NOT Work

| Attempt | What | Why it failed |
|---------|------|---------------|
| Split sync barrier for cache-enabled scheduling (`GGML_EXPERT_CACHE_SYNC_SPLITS`) | Suspected missing CPU/GPU barrier between splits | No change - corruption is within the fallback path itself, not between splits |
| Bundle node deferral (skip gate/up/down in generic remap) | Suspected generic remap overwrote route-ready data | No change - telemetry showed 0 eligible ops, generic remap never engages |
| Full-hit sidecar comparison (`GGML_ROUTE_READY_FULL_HIT_VERIFY`) | Suspected full-hit GPU sidecar produced wrong results | Never triggered - all plans route to CPU fallback, no full-hit cache engagement |
| Clone IDs from `plan.route_ids` instead of `down->src[2]` | Suspected ID tensor mismatch | No change - the IDs data was correct, the issue is in tensor strides |

---

## 5. All Temporary Diagnostic Code Still in Place

| Env var | Location | Purpose | Remove after fix |
|---------|----------|---------|------------------|
| `GGML_ROUTE_READY_FALLBACK_VERIFY` | `ggml-backend.cpp:3269` | Compares CPU fallback output vs canonical node output | YES |

---

## 6. Test Suite Status

```
test-expert-cache: ALL 35 TESTS PASS
  - GPU-resident rebalance: PASS (with seed rejection)
  - Cross-split sidecar: PASS (with sync assertion)
  - All other tests: PASS

Build: Release (cmake --build build --config Release)
```

---

## 7. Key Code Locations

| What | File | Lines |
|------|------|-------|
| CPU-base fallback path (THE BUG) | `ggml/src/ggml-backend.cpp` | 3195-3296 |
| Route-ready dispatch loop | `ggml/src/ggml-backend.cpp` | 3005-3310 |
| CPU MUL_MAT_ID kernel | `ggml/src/ggml-cpu/ggml-cpu.c` | 1542-1717 |
| Full-hit sidecar executor | `ggml/src/ggml-backend-moe-hetero.cpp` | 532-631 |
| Expert cache host guard | `ggml/src/ggml-backend-expert-cache.cpp` | 199-206 |
| Cache seed function | `ggml/src/ggml-backend-expert-cache.cpp` | 2002-2044 |

---

## 8. Next Steps (in order)

1. **Apply the stride fix** described in section 2 to all three clone tensors in the CPU-base fallback
2. **Build and verify** `max_abs` drops to ~0 with `GGML_ROUTE_READY_FALLBACK_VERIFY=1`
3. **Run full 64-token request** and confirm output matches cache-off baseline
4. **Run full test suite** (`test-expert-cache`, `test-backend-ops`)
5. **Remove temporary diagnostic** (`GGML_ROUTE_READY_FALLBACK_VERIFY` code block)
6. **Benchmark** against the 20 t/s target with `tools/results/expert-cache/run-tg-matrix.py`
7. **Update** EXPERT_CACHE.md and EXPERT_CACHE_OPTIMIZATIONS_LOG.md
