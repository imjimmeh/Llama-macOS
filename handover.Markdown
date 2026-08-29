# Route-Ready Bundle Dispatch Handover

## Status

Implemented the route-ready bundle dispatch plan in the current worktree.

No commit or push was performed.

## Changes

- Added scheduler dispatch records linking shared route-ID producers to Gate/Up/Down consumers across splits.
- Added producer-prefix execution and backend synchronization before route-ID reads.
- Added complete-bundle admission: every requested expert must be resident before remapping.
- Preserved normal graph execution for incomplete bundles.
- Removed the route-ready CPU fallback path.
- Added non-allocating slot-pool lookup APIs:
  - `ggml_backend_expert_cache_find_slot_tensor`
  - non-allocating `ggml_backend_expert_cache_find_slot`
- Preserved asynchronous remapped-ID host storage until backend consumption.
- Tightened route-ready regression assertions:
  - 6 zero-copy hits
  - 6 requests
  - 1 route-ready action
  - 0 RAM-to-GPU weight bytes
  - CPU-reference-equivalent output
  - successful graph reuse
- Updated `EXPERT_CACHE.md`.
- Updated `EXPERT_CACHE_OPTIMIZATIONS_LOG.md`.

## Verification

- `test-expert-cache.exe`: all tests passed.
- Built `test-expert-cache`, `llama-bench`, and `llama-server`.
- `test-fit-debug.exe`: passed with the Qwen3.6-35B model.
- Cache-enabled `llama-bench` smoke run reported `Weight Upload Bytes: 0`.
- `git diff --check`: clean.

## Known Limitation

The real-model TG baseline still reports zero eligible cache operations and zero requests. Therefore, no new throughput or deterministic-server performance claim is made for this recovery. The topology trace attempt also produced no scheduler node-assignment trace; the optimization log records this as an unsatisfied trace gate rather than inferred evidence.

## Worktree Notes

The worktree contains existing user changes and untracked benchmark artifacts. They were preserved. The implementation was made directly in the existing checkout without creating a branch or worktree.
