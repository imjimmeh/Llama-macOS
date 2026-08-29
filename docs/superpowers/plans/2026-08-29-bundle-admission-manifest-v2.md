# Bundle-Admission Static Manifest v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the obsolete individual-route static pinned-expert manifests with a placement-aware, bundle-admission-aware generator, and make manifest loading and route admission fully observable so engagement can be verified from benchmark telemetry alone.

**Architecture:** Three coordinated changes. (1) Add route-admission outcome counters (full-hit / fallback / per-mask histogram) to the scheduler route-census stats and expose them as `llama-bench` columns. (2) Track per-entry manifest loader outcomes (parsed / seeded / unregistered layer / seed failure / pinned / residency) in the expert cache and print them at teardown. (3) Rewrite the static ranking in `test-moe-tg-profiler.cpp` as a greedy bundle-coverage selection that only targets layers whose MoE weights are host-resident under the deployment placement, ranks by expected 8/8 and 7/8 route admissions, and emits a v2 manifest. Validate with the standard alternating TG512 matrix.

**Tech Stack:** C++17 (MSVC), ggml scheduler internals, nlohmann-free manifest I/O (manual string scan in loader), existing `tools/results/expert-cache/run-tg-matrix.py` benchmark harness.

## Global Constraints

- ASCII only in all code, comments, and new doc paragraphs: no emdash, no unicode arrows or symbols (AGENTS.md).
- Code comments concise (1-2 lines), explain why not what, ASD-STE100 wording.
- Build: `cmake --build build --config Release --target test-expert-cache test-moe-tg-profiler llama-bench` (MSVC + CUDA, GTX 1080).
- Every new `llama-bench` stat column MUST be appended to ALL of: `get_fields()`, `get_values()`, the INT type list, and any subtract/normalize helper, in `tools/llama-bench/llama-bench.cpp`. Then run `python tools/results/expert-cache/test_run_tg_matrix.py`.
- Run single test binaries while debugging; full `ctest` only at the end (known separate CTest-only failure in `test_route_ready_cross_split_sidecar` is pre-existing and out of scope).
- Benchmarks use `tools/results/expert-cache/run-tg-matrix.py` with `GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL=0` semantics (the script enforces this), model `C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf`.
- Every commit requires explicit user approval before `git commit` (repo convention).
- Branch: `feat/expert-cache-without-prediction`.

## Background (why this work exists)

The 2026-08-29 analysis of the 1,024 MiB static profile run (recorded in `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`, section "1,024 MiB Zero-Action Root-Cause Analysis") established:

1. Production dispatch (`ggml_backend_sched_compute_splits`, `ggml/src/ggml-backend.cpp:2967-3001`) admits only two GPU cases: `n_hits == top_k` (full-hit sidecar) and `n_hits == top_k - 1` (one-miss heterogeneous). All other masks stay on the CPU-base path.
2. `pinned_experts_1024mb.json` spreads 537 entries across all 40 layers (8-18 experts per layer, mean 13.4 of 256). No classified bundle reached 7/8 residency: 17,408 classifications, 68 planned dispatches, 0 actions, 0 requests.
3. `expert_cache_requests` counts executed cache tensors, not route lookups; `n_route_ready_actions` counts admissions only. Neither exposes how close rejected bundles were.
4. The loader (`ggml_backend_expert_cache_load_pinned_manifest`, `ggml/src/ggml-backend-expert-cache.cpp:2118-2213`) prints parsed-entry count as "loaded", silently skips unregistered layers and failed seeds, and does not account for unused capacity (666.77 MiB resident of 1,024 MiB requested in the measured run).
5. The legacy manifest was generated before placement, so entries target layers whose MoE weights are not host-resident under deployment placement (9,216 non-host weight bypasses in the measured row).

---

### Task 1: Route-Admission Outcome Telemetry

**Files:**
- Modify: `ggml/include/ggml-backend.h` (struct `ggml_backend_expert_cache_stats`, after `n_route_ready_classifications` around line 421)
- Modify: `ggml/src/ggml-backend.cpp:2941-3001` (classification site)
- Modify: `ggml/src/ggml-backend.cpp:3444-3470` (`ggml_backend_sched_print_expert_cache_stats`)
- Modify: `tools/llama-bench/llama-bench.cpp` (get_fields / get_values / INT list / helper)
- Test: `tests/test-expert-cache.cpp`

**Interfaces:**
- Consumes: existing `route_census_stats` (zero-initialized at `ggml/src/ggml-backend.cpp:877`).
- Produces: `uint64_t n_route_ready_full_hits`, `uint64_t n_route_ready_fallbacks`, `uint64_t n_route_ready_mask_counts[9]` in `ggml_backend_expert_cache_stats`. Scheduler-side only: these counters are incremented at the classification site and are NOT summed from per-backend cache stats in `ggml_backend_sched_get_expert_cache_stats` (they live in `route_census_stats` which that function copies at line 3486). Bench columns `expert_cache_route_ready_full_hits`, `expert_cache_route_ready_fallbacks`.

- [ ] **Step 1: Write the failing test**

In `tests/test-expert-cache.cpp`, extend `test_route_ready_cross_split_sidecar()` (the function ending at line 2074) after the existing `second_stats` assertions (lines 2059-2064):

```cpp
    // admission outcome counters: 1 full-hit admission per compute, no fallbacks
    require(second_stats.n_route_ready_full_hits == 2);
    require(second_stats.n_route_ready_fallbacks == 0);
    require(second_stats.n_route_ready_mask_counts[8] == 2);
    require(second_stats.n_route_ready_mask_counts[7] == 0);
```

Also add a fallback-path check to `test_registered_bundle_cpu_base_placement` (or the nearest existing test that classifies a bundle with low residency): after its compute call, assert `stats.n_route_ready_fallbacks >= 1` and the corresponding mask bucket incremented. If no existing test classifies an incomplete bundle end-to-end, extend `test_hit_mask_matrix_partitioning` style coverage by asserting on the partition helper only and keep the scheduler-side assertions in the cross-split test.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build --config Release --target test-expert-cache && build/bin/Release/test-expert-cache.exe`
Expected: compile error (`n_route_ready_full_hits` not a member) or assertion failure. Fix build errors only if they block the run.

- [ ] **Step 3: Implement the counters**

a) `ggml/include/ggml-backend.h`, inside `ggml_backend_expert_cache_stats` right after `n_route_ready_classifications`:

```cpp
        uint64_t n_route_ready_full_hits;
        uint64_t n_route_ready_fallbacks;
        uint64_t n_route_ready_mask_counts[9];
```

b) `ggml/src/ggml-backend.cpp`, at the classification site. After the `partition_bundle_routes` call and `record_route_ready_accesses` (lines 2946-2951), insert before the `n_route_ready_classifications++` line:

```cpp
                    // admission outcome: count how close each bundle was to the 7/8 and 8/8 gates
                    if (n_hits >= 0 && n_hits < 9) {
                        sched->route_census_stats.n_route_ready_mask_counts[n_hits]++;
                    }
                    if (n_hits == top_k && n_misses == 0) {
                        sched->route_census_stats.n_route_ready_full_hits++;
                    } else if (!(n_hits > 0 && n_hits == top_k - 1 && n_hits + n_misses == top_k)) {
                        sched->route_census_stats.n_route_ready_fallbacks++;
                    }
```

Keep the existing branch structure below unchanged (the `if (n_hits == top_k ...)` execution branches stay as they are; the new counters are observation-only).

c) Extend `ggml_backend_sched_print_expert_cache_stats` (`ggml/src/ggml-backend.cpp:3444`): add an ungated block (outside the `hetero_layers > 0` guard) printed when `n_route_ready_classifications > 0`:

```cpp
    if (sched->route_census_stats.n_route_ready_classifications > 0) {
        printf("\n=== Route-Ready Admission Telemetry ===\n");
        printf("  Classifications:  %" PRIu64 "\n", sched->route_census_stats.n_route_ready_classifications);
        printf("  Full-Hit (8/8):   %" PRIu64 "\n", sched->route_census_stats.n_route_ready_full_hits);
        printf("  Fallbacks:        %" PRIu64 "\n", sched->route_census_stats.n_route_ready_fallbacks);
        printf("  Mask Histogram    :");
        for (int k = 0; k < 9; k++) {
            printf(" %d:%" PRIu64, k, sched->route_census_stats.n_route_ready_mask_counts[k]);
        }
        printf("\n=====================================\n\n");
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `build/bin/Release/test-expert-cache.exe`
Expected: `all test-expert-cache tests passed successfully!` (same pre-existing CTest caveat does not apply to the direct run).

- [ ] **Step 5: Add the two bench columns**

In `tools/llama-bench/llama-bench.cpp`, mirror `expert_cache_route_ready_classifications` exactly in all four places:
1. `get_fields()` list (near line 1795): add `"expert_cache_route_ready_full_hits"`, `"expert_cache_route_ready_fallbacks"`.
2. `get_values()` list: add `std::to_string(expert_cache_stats.n_route_ready_full_hits)`, `std::to_string(expert_cache_stats.n_route_ready_fallbacks)` in the SAME position relative to neighbors.
3. INT type list (near line 1865): add both names.
4. Any subtract/normalize helper that lists these fields.

Then run: `python tools/results/expert-cache/test_run_tg_matrix.py`
Expected: pass (column alignment validated).

- [ ] **Step 6: Smoke check on the real model**

Run:
```bash
build/bin/Release/llama-bench.exe -m "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" -p 0 -n 32 -r 1 -t 14 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -fa on -lm mmap -fitt 256 -exc 1024 -excp 65536 -pe pinned_experts_1024mb.json -o jsonl
```
Expected: new columns present; with the legacy manifest `expert_cache_route_ready_fallbacks` equals `expert_cache_route_ready_classifications` and `expert_cache_route_ready_full_hits` is 0, confirming the diagnosis end to end.

- [ ] **Step 7: Commit checkpoint**

Run: `git diff --check -- ggml/include/ggml-backend.h ggml/src/ggml-backend.cpp tools/llama-bench/llama-bench.cpp tests/test-expert-cache.cpp`
Request explicit user approval. Only after approval:
```bash
git add -- ggml/include/ggml-backend.h ggml/src/ggml-backend.cpp tools/llama-bench/llama-bench.cpp tests/test-expert-cache.cpp
git commit -m "ggml: add route-ready admission outcome telemetry"
```

---

### Task 2: Manifest Loader Outcome Accounting

**Files:**
- Modify: `ggml/src/ggml-backend-expert-cache.h` (public struct + accessor)
- Modify: `ggml/src/ggml-backend-expert-cache.cpp:2118-2213` (loader), `:1900-1935` area (teardown stats print)
- Test: `tests/test-expert-cache.cpp`

**Interfaces:**
- Produces:

```cpp
struct ggml_backend_expert_cache_manifest_stats {
    uint64_t n_parsed;               // (layer, expert) pairs found in the file
    uint64_t n_unregistered_layer;   // no bundle registration for the layer at load time
    uint64_t n_seed_failed;          // ggml_backend_expert_cache_seed returned false
    uint64_t n_seeded;               // seed succeeded (slot claimed or already present)
    uint64_t n_pinned_marked;        // slot found and is_pinned set
    uint64_t n_slot_lookup_failed;   // find_slot returned -1 after successful seed
};

GGML_API void ggml_backend_expert_cache_get_manifest_stats(
    ggml_backend_expert_cache_t cache,
    struct ggml_backend_expert_cache_manifest_stats * out_stats);
```

- The existing print at teardown (`GGML_LOG_INFO("  resident: ...")` around `ggml/src/ggml-backend-expert-cache.cpp:1911`) gains a manifest block when `n_parsed > 0`.

- [ ] **Step 1: Write the failing test**

Add `test_manifest_loader_outcome_stats()` to `tests/test-expert-cache.cpp` (register it in `main`). Model it on the existing scheduler tests: build a CPU backend + scheduler, register one expert bundle for layer 0 via `ggml_backend_sched_register_expert_bundle` with small gate/up/down weight tensors (copy the tensor construction from `test_route_ready_cross_split_sidecar`, lines 1958-2033), create a cache with a tiny capacity (e.g. one expert stride) via the existing cache-creation path used by other tests, then write a temp manifest file:

```cpp
    std::string manifest_path = "test-manifest-outcomes.json";
    {
        std::ofstream ofs(manifest_path);
        ofs << "{\"pinned_experts\": [\n"
            << "  {\"layer\": 0, \"expert_id\": 0},\n"   // registered layer, fits
            << "  {\"layer\": 0, \"expert_id\": 1},\n"   // registered layer, capacity exhausted
            << "  {\"layer\": 9, \"expert_id\": 5}\n"    // no registration for layer 9
            << "]}\n";
    }
```

Load it with `ggml_backend_sched_load_pinned_manifest`, then:

```cpp
    struct ggml_backend_expert_cache_manifest_stats ms = {};
    ggml_backend_expert_cache_get_manifest_stats(cache, &ms);
    require(ms.n_parsed == 3);
    require(ms.n_unregistered_layer == 1);
    require(ms.n_seeded + ms.n_seed_failed == 2);
    require(ms.n_seeded >= 1);
    std::remove(manifest_path.c_str());
```

Adjust the exact cache-construction calls to match whichever existing test creates an `expert_caches[b]` entry (grep `expert_cache_init` or the wrapper the suite already uses); the scheduler-level loader requires `sched->expert_caches[b]` to be non-NULL, which the existing route-ready tests already arrange.

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build --config Release --target test-expert-cache && build/bin/Release/test-expert-cache.exe`
Expected: compile error (struct/accessor missing).

- [ ] **Step 3: Implement**

a) Add the struct and accessor declaration to `ggml/src/ggml-backend-expert-cache.h` near the other stats accessors.

b) Add `ggml_backend_expert_cache_manifest_stats manifest_stats = {};` to the cache object (find the cache struct definition at the top of `ggml-backend-expert-cache.cpp`).

c) In `ggml_backend_expert_cache_load_pinned_manifest`:
- Count `n_parsed` where `pinned_bundles.insert` happens today (line 2149). Note `pinned_bundles` is a set of pairs: parse into a `std::vector<std::pair<int,int>>` first so duplicate entries still count as parsed, then insert into the set.
- Replace the `if (reg_it != cache->bundle_registrations.end())` silent skip (line 2163) with an else-branch incrementing `n_unregistered_layer`.
- For each tensor branch (gate_up/down and gate/up/down), count seed success/failure and `find_slot` outcome:

```cpp
                if (reg.down && reg.down->data != NULL) {
                    if (ggml_backend_expert_cache_seed(cache, reg.down, expert_id, 1000)) {
                        cache->manifest_stats.n_seeded++;
                        int32_t s = ggml_backend_expert_cache_find_slot(cache, reg.down, expert_id);
                        if (s >= 0) {
                            auto * pool = ggml_backend_expert_cache_get_or_create_pool(cache, reg.down);
                            if (pool && s < pool->max_slots) { pool->slots[s].is_pinned = true; cache->manifest_stats.n_pinned_marked++; }
                        } else {
                            cache->manifest_stats.n_slot_lookup_failed++;
                        }
                    } else {
                        cache->manifest_stats.n_seed_failed++;
                    }
                }
```

Apply the same pattern to all four tensor variants (fused: gate_up, down; separate: gate, up, down). Keep the existing stderr summary line but make it truthful:

```cpp
    fprintf(stderr, "%s: parsed %zu entries (%zu seeded, %zu unregistered layer, %zu seed failed) from '%s'\n",
        __func__, cache->manifest_stats.n_parsed, cache->manifest_stats.n_seeded,
        cache->manifest_stats.n_unregistered_layer, cache->manifest_stats.n_seed_failed,
        manifest_json_path);
```

d) Implement the accessor (copy field-by-field under the cache mutex if one exists; otherwise direct copy).

e) Extend the teardown print block (the `GGML_LOG_INFO` sequence containing `resident:` at line 1911): after the existing lines add, still inside the same function:

```cpp
    if (cache->manifest_stats.n_parsed > 0) {
        GGML_LOG_INFO("  manifest parsed:      %" PRIu64 "\n", cache->manifest_stats.n_parsed);
        GGML_LOG_INFO("  manifest seeded:      %" PRIu64 "\n", cache->manifest_stats.n_seeded);
        GGML_LOG_INFO("  manifest unreg layer: %" PRIu64 "\n", cache->manifest_stats.n_unregistered_layer);
        GGML_LOG_INFO("  manifest seed failed: %" PRIu64 "\n", cache->manifest_stats.n_seed_failed);
        GGML_LOG_INFO("  manifest pinned:      %" PRIu64 "\n", cache->manifest_stats.n_pinned_marked);
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `build/bin/Release/test-expert-cache.exe`
Expected: pass.

- [ ] **Step 5: Real-model smoke**

Repeat the Task 1 Step 6 command and read the teardown stderr block.
Expected: `manifest parsed: 537`, `manifest seeded` well below 537 or `seed failed`/`unreg layer` accounting for the gap; `resident` plus these numbers now explain the 666.77 MiB observation instead of hiding it.

- [ ] **Step 6: Commit checkpoint**

Run: `git diff --check -- ggml/src/ggml-backend-expert-cache.h ggml/src/ggml-backend-expert-cache.cpp tests/test-expert-cache.cpp`
Request explicit user approval. Only after approval:
```bash
git add -- ggml/src/ggml-backend-expert-cache.h ggml/src/ggml-backend-expert-cache.cpp tests/test-expert-cache.cpp
git commit -m "ggml: account pinned manifest loader outcomes"
```

---

### Task 3: Placement-Aware Bundle-Admission Generator (Manifest v2)

**Files:**
- Modify: `tests/test-moe-tg-profiler.cpp` (args, eligibility filter, greedy selection, manifest writer)

**Interfaces:**
- Consumes: `pctx.layer_routes[l][t]` (already captured), `init->model()` tensor buffers, existing CLI parsing at lines 178-192.
- Produces: manifest files `<prefix>_<tier>mb.json` with format v2:

```json
{
  "format": 2,
  "admission": "7of8",
  "top_k": 8,
  "w_full": 1.0,
  "w_hetero": 0.4,
  "placement_cache_mib": 1024,
  "eligible_layers": [0, 3, 5],
  "pinned_experts": [
    {"layer": 0, "expert_id": 59, "activations": 63, "bundle_full_hits": 12, "bundle_seven_hits": 4}
  ]
}
```

The loader (Task 2 version) parses `layer` / `expert_id` by string scan and ignores the added keys, so v2 files load unchanged.

- [ ] **Step 1: Add CLI arguments**

Extend the arg loop (lines 178-192):

```cpp
        } else if (strcmp(argv[i], "--cache-mib") == 0 && i + 1 < argc) {
            cache_mib = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--w-full") == 0 && i + 1 < argc) {
            w_full = atof(argv[++i]);
        } else if (strcmp(argv[i], "--w-hetero") == 0 && i + 1 < argc) {
            w_hetero = atof(argv[++i]);
        }
```

with `size_t cache_mib = 0; double w_full = 1.0; double w_hetero = 0.4;` declared beside `fit_target_bytes`. Set `params.expert_cache_size = cache_mib * 1024 * 1024;` (replacing the hardcoded 0 at line 217) so `common/fit.cpp` deducts the same cache size as deployment and reproduces the deployment layer placement. Keep `expert_cache_period` at its inert default and load no manifest, so the profiling cache stays empty; routes are captured by `cb_eval` regardless.

- [ ] **Step 2: Placement eligibility filter**

After model load, before ranking:

```cpp
    std::vector<bool> layer_eligible(pctx.n_layers, false);
    int n_eligible_layers = 0;
    for (int l = 0; l < pctx.n_layers; l++) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", l);
        const ggml_tensor * t = llama_model_get_tensor(model, name);
        layer_eligible[l] = t != nullptr && ggml_backend_buffer_is_host(t->buffer);
        if (layer_eligible[l]) n_eligible_layers++;
    }
    printf("Placement-eligible host MoE layers: %d of %d (cache-mib %zu)\n",
        n_eligible_layers, pctx.n_layers, cache_mib);
```

- [ ] **Step 3: Greedy bundle-coverage selection**

Replace the tier emission block (lines 487-521). Data structures:

```cpp
    const int top_k_model = llama_model_n_expert_used(model);   // 8 for this model
    const int n_layers_model = pctx.n_layers;
    // inverted index: occurrences[l][e] = decode token indices routing to e
    std::vector<std::vector<std::vector<int>>> occurrences(n_layers_model,
        std::vector<std::vector<int>>(n_experts));
    std::vector<std::vector<int>> hit_count(n_layers_model);
    for (int l = 0; l < n_layers_model; l++) {
        hit_count[l].assign(pctx.layer_routes[l].size(), 0);
        for (size_t t = 0; t < pctx.layer_routes[l].size(); t++) {
            for (int exp : pctx.layer_routes[l][t]) {
                if (exp >= 0 && exp < n_experts) occurrences[l][exp].push_back((int)t);
            }
        }
    }
```

Selection loop per tier (tiers stay 64/128/256/512/1024 MiB, `bundle_bytes` stays the 1.95 MiB triple):

```cpp
    std::vector<std::set<int>> chosen(n_layers_model);
    for (size_t tier_mb : tiers_mb) {
        const size_t max_bundles = (tier_mb * 1024 * 1024) / expert_bundle_bytes;
        std::vector<std::set<int>> sel = chosen;   // tiers are cumulative like today
        size_t used = 0;
        std::vector<std::vector<int>> hc = hit_count;   // scratch copy of |routes_t intersect sel_l|
        while (used < max_bundles) {
            int best_l = -1, best_e = -1;
            double best_gain = 0.0;
            for (int l = 0; l < n_layers_model; l++) {
                if (!layer_eligible[l]) continue;
                for (int e = 0; e < n_experts; e++) {
                    if (sel[l].count(e)) continue;
                    uint64_t d_full = 0, d_seven = 0;
                    for (int t : occurrences[l][e]) {
                        if (hc[l][t] == top_k_model - 1) d_full++;
                        else if (hc[l][t] == top_k_model - 2) d_seven++;
                    }
                    double gain = w_full * (double)d_full + w_hetero * (double)d_seven;
                    if (gain > best_gain) { best_gain = gain; best_l = l; best_e = e; }
                }
            }
            if (best_l < 0 || best_gain <= 0.0) break;
            for (int t : occurrences[best_l][best_e]) hc[best_l][t]++;
            sel[best_l].insert(best_e);
            used++;
        }
        chosen = sel;
        /* emit manifest for this tier (Step 4) */
    }
```

Complexity note for the implementer: rounds x total occurrences is roughly `537 x (n_gen x top_k x 40)` which is about 22M cheap operations at `n_gen = 512`; no memoization needed.

Per-entry emitted stats: after selection, for each chosen `(l, e)` compute `bundle_full_hits` = tokens with all `top_k` routes in `sel[l]`, and `bundle_seven_hits` = tokens with exactly `top_k - 1` in `sel[l]` where `e` is one of them. Also compute tier-level projections:

```cpp
    uint64_t proj_full = 0, proj_seven = 0;
    for (int l = 0; l < n_layers_model; l++) {
        for (size_t t = 0; t < pctx.layer_routes[l].size(); t++) {
            int c = 0;
            for (int exp : pctx.layer_routes[l][t]) if (sel[l].count(exp)) c++;
            if (c == top_k_model) proj_full++;
            else if (c == top_k_model - 1) proj_seven++;
        }
    }
```

Print per tier: `Tier 1024 MiB: N pinned | projected 8/8 admissions X (Y% of decode steps) | projected 7/8 admissions Z`.

- [ ] **Step 4: Emit the v2 manifest**

Write the JSON exactly as the interface block above shows (manual `ofs <<` like today, ASCII only). The `pinned_experts` entries come from `sel` sorted by `(layer, expert_id)`.

- [ ] **Step 5: Build and run the generator**

```bash
cmake --build build --config Release --target test-moe-tg-profiler
mkdir -p tools/results/expert-cache/bundle-v2
build/bin/Release/test-moe-tg-profiler.exe -m "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" -t 14 -p 16 -n 512 -fitt 256 --cache-mib 1024 --out-manifest tools/results/expert-cache/bundle-v2/pinned_bundle_v2
```

Expected:
- `Placement-eligible host MoE layers:` prints a nonzero count consistent with the 204 cpu-host census nodes seen on 2026-08-29 (roughly 34 of 40 layers under this placement).
- The 1,024 MiB tier projects a nonzero `8/8 admissions` count; concentration differs sharply from the legacy per-layer spread (expect few layers with most entries).
- Six manifest files written, v2 keys present.

Sanity gate: if projected 8/8 admissions for the 1,024 MiB tier is 0, stop and investigate the eligibility filter or weights before proceeding (the 3 GiB full-layer manifest demonstrated real 8/8 admissions exist for layers 0-3, so a correct greedy must find some).

- [ ] **Step 6: Commit checkpoint**

Run: `git diff --check -- tests/test-moe-tg-profiler.cpp`
Request explicit user approval. Only after approval:
```bash
git add -- tests/test-moe-tg-profiler.cpp
git commit -m "tools: rank pinned experts by bundle admission coverage"
```

---

### Task 4: Validation Matrix (Engagement Gate, Then Performance)

**Files:**
- Create results: `tools/results/expert-cache/bundle-v2/*.jsonl`

**Interfaces:**
- Consumes: the v2 1,024 MiB manifest from Task 3, the telemetry columns from Task 1, loader stats from Task 2.

- [ ] **Step 1: Single-run engagement smoke**

```bash
build/bin/Release/llama-bench.exe -m "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" -p 0 -n 32 -r 1 -t 14 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -fa on -lm mmap -fitt 256 -exc 1024 -excp 65536 -pe tools/results/expert-cache/bundle-v2/pinned_bundle_v2_1024mb.json -o jsonl
```

Gates (hard, from the JSONL row):
- `expert_cache_route_ready_actions > 0`
- `expert_cache_route_ready_full_hits + expert_cache_route_ready_fallbacks` equals `expert_cache_route_ready_classifications`
- `expert_cache_zero_copy_hits > 0`
- teardown stderr: `manifest seeded` close to `manifest parsed`, `resident` close to the tier size
If any gate fails, stop and diagnose with the mask histogram and loader stats before any matrix run.

- [ ] **Step 2: Five-pair alternating matrix, both orders**

```bash
python tools/results/expert-cache/run-tg-matrix.py --model "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" --output-dir tools/results/expert-cache/bundle-v2 --prefix 2026-08-30-bundle-v2-1024m-control-first-n512 --runs 5 --n-gen 512 --load-mode mmap --fit-target 256 --cache-mib 1024 --cache-period 65536 --max-swaps 0 --pinned-experts tools/results/expert-cache/bundle-v2/pinned_bundle_v2_1024mb.json
python tools/results/expert-cache/run-tg-matrix.py --model "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" --output-dir tools/results/expert-cache/bundle-v2 --prefix 2026-08-30-bundle-v2-1024m-cache-first-n512 --runs 5 --n-gen 512 --load-mode mmap --fit-target 256 --cache-mib 1024 --cache-period 65536 --max-swaps 0 --pinned-experts tools/results/expert-cache/bundle-v2/pinned_bundle_v2_1024mb.json --cache-first
```

- [ ] **Step 3: Analyze**

For the ten pairs report: control mean, cache mean, mean and median paired delta, positive pairs, 95% Student-t interval, plus means of `route_ready_actions`, `route_ready_full_hits`, `route_ready_fallbacks`, `zero_copy_hits`, `bytes_ram_to_gpu`. Compare absolute cache-on throughput against the repaired cache-off automatic-fit mean (22.0-23.3 tok/s range from 2026-08-29) and against the retained dynamic 128 MiB period-32 result (22.765 tok/s, +3.305%).

Engagement is the acceptance gate; throughput is a measured outcome recorded either way. A negative or flat delta with full engagement is a valid result: it means static 7/8-8/8 residency at 1,024 MiB does not beat this placement, and the log records it as such.

- [ ] **Step 4: Commit checkpoint**

Normalize JSONL to LF with one final newline, then `git diff --check -- tools/results/expert-cache/bundle-v2`.
Request explicit user approval. Only after approval:
```bash
git add -- tools/results/expert-cache/bundle-v2
git commit -m "bench: validate bundle-admission pinned manifest"
```

---

### Task 5: Documentation

**Files:**
- Modify: `EXPERT_CACHE.md`
- Modify: `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`

**Interfaces:**
- Consumes: Task 1-4 results.

- [ ] **Step 1: Update `EXPERT_CACHE.md`**

- Section 2.3: replace the legacy tier table and its coverage claims with the v2 generator description (placement-aware, bundle-admission objective, weight parameters) and point at `tools/results/expert-cache/bundle-v2/`. Keep one sentence noting the legacy root-level manifests are obsolete for the current dispatcher.
- Regime section: add the v2 1,024 MiB result row (numbers from Task 4 Step 3).
- CLI table: no changes (loader interface unchanged).

- [ ] **Step 2: Append to `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`**

Record: telemetry added (Task 1), loader accounting (Task 2), generator design with weights and eligibility count (Task 3), the full Task 4 table, and the explicit conclusion on whether static bundle-admission residency beats the dynamic 128 MiB period-32 operating point.

- [ ] **Step 3: Final verification**

```bash
cmake --build build --config Release --target test-expert-cache test-moe-tg-profiler llama-bench
build/bin/Release/test-expert-cache.exe
python tools/results/expert-cache/test_run_tg_matrix.py
git diff --check
```

Expected: build succeeds; direct test run passes; alignment test passes; no whitespace errors. Full `ctest` remains blocked by the separate pre-existing CTest-only failure; state that explicitly in the final report.

- [ ] **Step 4: Commit checkpoint**

Request explicit user approval. Only after approval:
```bash
git add -- EXPERT_CACHE.md EXPERT_CACHE_OPTIMIZATIONS_LOG.md
git commit -m "docs: record bundle-admission manifest v2 results"
```

---

## Self-Review

1. **Spec coverage:** diagnosis items (1) obsolete ranking objective -> Task 3; (2) pre-placement generation -> Task 3 Steps 1-2; (3) loader telemetry/accounting -> Task 2; observable admission outcomes -> Task 1; validation and docs -> Tasks 4-5. The "regenerate manifests before using" guidance in `EXPERT_CACHE.md` is realized by Task 3. No gap.
2. **Placeholder scan:** all steps carry concrete code or exact commands; no TBD/TODO.
3. **Type consistency:** `ggml_backend_expert_cache_manifest_stats` field names are identical in Task 2 interface, test, and implementation steps; bench column names identical in Task 1 and Task 4 gates; manifest keys identical in Task 3 interface and writer.

## Risks

- Runtime pool shapes (fused vs separate registrations) may still leave residency below the tier byte budget even with correct accounting; Task 4 Step 1's loader-stats gate catches it, and Task 2 makes the shortfall visible rather than silent.
- Greedy 7/8-8/8 coverage is a heuristic, not optimal submodular coverage; acceptable because the gate is engagement, and weights are CLI-tunable for later calibration against `test-moe-partial-hit-bench` (out of scope here).
- Route streams captured at `-exc 1024` placement can drift slightly from the deployed run's routes due to floating-point backend differences; the alternating paired matrix absorbs this.
