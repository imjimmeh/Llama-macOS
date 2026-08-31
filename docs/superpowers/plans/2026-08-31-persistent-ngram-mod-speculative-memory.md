# Epic: Persistent, Scalable and Adaptive N-Gram Speculative Memory for llama.cpp

## 1. Epic Summary

Extend llama.cpp's `ngram-mod` speculative decoding implementation from its current small, fixed-size, process-local hash pool into a **persistent, configurable, scalable and observable speculative-memory subsystem**.

The primary goals are to:

1. Allow `ngram-mod` knowledge to survive `llama-server` restarts.
2. Allow the memory pool to be substantially larger than the current ~16 MB implementation.
3. Support memory sizes expressed explicitly or as a percentage of model size.
4. Experiment with memory budgets up to approximately **20% of loaded model weight size**, inspired by the conditional-memory/storage-vs-compute trade-offs explored by DeepSeek Engram.
5. Keep the active lookup path in **system RAM**, not GPU VRAM.
6. Use SSD/disk primarily for persistence rather than latency-sensitive speculation.
7. Allow optional runtime growth when additional capacity can reasonably improve speculative performance.
8. Add sufficient telemetry to determine whether larger memories actually improve:

   * draft hit rate;
   * acceptance rate;
   * accepted sequence length;
   * target-model evaluations avoided;
   * total generation throughput.
9. Preserve the defining characteristics of `ngram-mod`:

   * very low computational overhead;
   * O(1)-style lookup;
   * no additional neural model;
   * no GPU requirement;
   * shared learning between server slots.
10. Establish an architecture that can later support more sophisticated Engram-inspired speculative memory without requiring the initial implementation to become a general-purpose database.

The first implementation SHOULD remain recognisably `ngram-mod`.

It SHOULD NOT attempt to reproduce DeepSeek Engram itself.

---

# 2. Background

Current llama.cpp `ngram-mod` speculative decoding operates as a lightweight shared n-gram hash pool.

For an observed token sequence it approximately performs:

```text
n-gram tokens
     │
     ▼
rolling / LCG hash
     │
     ▼
hash pool slot
     │
     ▼
next token
```

During speculative decoding:

```text
current token history
        │
        ▼
compute hash of trailing n-gram
        │
        ▼
lookup predicted next token
        │
        ▼
append predicted token
        │
        ▼
repeat to create speculative draft
```

The current design has several useful properties:

* approximately 16 MB memory requirement;
* constant bounded memory;
* cheap lookups;
* cheap insertions;
* shared pool across server slots;
* variable-length speculative drafts;
* no separate draft model;
* learning from normal server activity.

However, the current design is effectively ephemeral.

Conceptually:

```text
server start
    │
    ▼
empty ngram-mod pool
    │
    ▼
requests gradually populate pool
    │
    ▼
useful speculative knowledge develops
    │
    ▼
server exits
    │
    ▼
knowledge lost
```

After restart:

```text
empty pool again
```

This prevents `ngram-mod` from becoming a long-lived speculative model of the user's workload.

It also means the existing ~16 MB capacity has largely been treated as an implementation constant rather than as a tunable memory/performance parameter.

---

# 3. Motivation

There are several reasons to explore significantly larger persistent n-gram memory.

## 3.1 Server workloads contain persistent repetition

Real inference servers repeatedly encounter:

* programming syntax;
* common libraries;
* coding patterns;
* repository-specific structures;
* boilerplate;
* command-line snippets;
* common reasoning phrases;
* standard answer structures;
* recurring user prompts;
* documentation;
* templated content;
* repeated transformations;
* repeated portions of agent workflows.

A server that processes thousands or millions of tokens can therefore gradually construct useful workload-specific speculative knowledge.

Today this knowledge is discarded when the process exits.

---

## 3.2 Memory is significantly cheaper than model computation

A speculative predictor can be useful even if each prediction has only modest intelligence, provided that:

```text
cost(prediction) << cost(target model token)
```

An n-gram lookup requiring a small number of RAM accesses is extremely inexpensive compared with evaluating a multi-billion-parameter transformer.

This becomes particularly interesting for:

* large models;
* CPU/GPU hybrid inference;
* partially offloaded MoE models;
* systems where token generation is memory-bandwidth constrained;
* models with expensive expert routing/expert execution;
* models where speculative verification batches are substantially cheaper than sequential target decoding.

---

# 4. Relationship to DeepSeek Engram

The design is inspired by, but MUST NOT be conflated with, DeepSeek Engram.

Engram explores **conditional memory through deterministic lookup**, demonstrating that large lookup-based memories can complement neural computation.

The relevant conceptual observation is:

```text
not every useful dependency needs expensive neural computation
```

and therefore some capacity may be more efficiently represented as directly addressable memory.

This epic explores a related systems-level question:

> Can a substantially larger persistent n-gram memory cheaply predict enough target-model tokens to improve end-to-end llama.cpp inference?

The proposed ~20% target is therefore an **experimental memory budget**, not a theoretical requirement.

For example:

```text
Loaded model weight size: 20 GB
20% maximum ngram memory: 4 GB
```

or:

```text
Loaded model weight size: 35 GB
20% maximum ngram memory: 7 GB
```

The implementation MUST allow benchmarking to determine whether performance saturates far below this level.

Possible outcomes include:

```text
Best capacity = 64 MB
```

or:

```text
Best capacity = 512 MB
```

or:

```text
Best capacity = 10% model size
```

or:

```text
Larger capacity continues helping through 20%
```

No particular result should be assumed.

---

# 5. Core Hypothesis

The principal hypothesis is:

> Increasing `ngram-mod` capacity and retaining learned n-grams across sessions will increase useful speculative matches and accepted speculative sequences sufficiently to improve end-to-end token-generation throughput, while RAM lookup overhead remains negligible compared with target-model inference.

Secondary hypotheses are:

1. A persisted warm cache will outperform a cold cache on representative recurring workloads.
2. Larger hash pools will reduce destructive collisions.
3. Larger pools will retain useful long-tail n-grams that would otherwise be overwritten.
4. Benefits will eventually plateau because capacity ceases to be the principal limiting factor.
5. Very large pools may eventually hurt performance through CPU cache/TLB/DRAM behaviour.
6. The optimum capacity will vary by workload.
7. Coding and agentic workloads are likely to benefit more than unconstrained one-off prose.
8. A dynamically growing cache may achieve most of the benefits of a large cache without always allocating the maximum memory.
9. Persistence may provide a larger real-world benefit than raw capacity because the memory can accumulate useful recurring patterns over days or weeks.

---

# 6. Non-Goals

The initial epic SHOULD NOT:

* implement DeepSeek Engram;
* modify transformer architecture;
* retrain target models;
* store embeddings;
* require GPU memory;
* introduce vector search;
* introduce semantic retrieval;
* implement a general-purpose language model database;
* replace `ngram-cache`;
* fundamentally change llama.cpp sampling;
* guarantee that 20% of model size is optimal;
* dynamically access SSD for every speculative prediction;
* make `ngram-mod` dependent on external services;
* require a database server;
* make persistence mandatory;
* change default `ngram-mod` behaviour unless the new functionality is explicitly enabled.

Backwards compatibility is important.

A user who runs:

```bash
--spec-type ngram-mod
```

without any new options SHOULD obtain behaviour equivalent to current upstream behaviour unless benchmark evidence strongly justifies changing the default later.

---

# 7. Proposed End-State Architecture

The target architecture should conceptually support:

```text
                         llama-server
                              │
                              ▼
                    token stream / requests
                              │
                              ▼
                  ┌──────────────────────┐
                  │ ngram-mod predictor  │
                  └──────────┬───────────┘
                             │
                    RAM-resident lookup
                             │
             ┌───────────────┴────────────────┐
             │                                │
             ▼                                ▼
       prediction found                  no prediction
             │                                │
             ▼                                ▼
       speculative draft                 target model
             │
             ▼
       target verifies
             │
       ┌─────┴─────┐
       ▼           ▼
    accepted    rejected
       │
       ▼
learning / statistics
       │
       ▼
persistent checkpoint
       │
       ▼
SSD / filesystem
```

Disk is primarily responsible for:

```text
durability
```

RAM is primarily responsible for:

```text
latency-sensitive lookup
```

---

# 8. Phase 1 Target Architecture

The initial implementation SHOULD use the simplest reliable architecture:

```text
             Startup
                │
                ▼
       determine desired capacity
                │
                ▼
         allocate RAM pool
                │
                ▼
       persistent cache exists?
          ┌─────┴─────┐
          │           │
         yes          no
          │           │
          ▼           ▼
         load       initialise
          │           │
          └─────┬─────┘
                ▼
          normal operation
                │
       ┌────────┴─────────┐
       │                  │
     lookup             learn
       │                  │
       └────────┬─────────┘
                ▼
         mark pool dirty
                │
                ▼
       periodic checkpoint
                │
                ▼
             disk
```

This avoids introducing a complicated RAM/SSD hierarchy before the value of increased n-gram capacity has been demonstrated.

---

# 9. Functional Requirement: Configurable Pool Size

The hard-coded or implicitly fixed `ngram-mod` pool size MUST become configurable.

The user SHOULD be able to specify an absolute capacity.

Proposed CLI:

```bash
--spec-ngram-mod-size 256M
```

Examples:

```bash
--spec-ngram-mod-size 16M
--spec-ngram-mod-size 64M
--spec-ngram-mod-size 256M
--spec-ngram-mod-size 1G
--spec-ngram-mod-size 4G
```

Standard llama.cpp size parsing conventions SHOULD be reused where possible.

The implementation SHOULD support at minimum:

```text
K
M
G
```

suffixes if existing CLI parsing supports them.

---

# 10. Functional Requirement: Model-Relative Sizing

Users SHOULD also be able to express capacity relative to the loaded model.

Example:

```bash
--spec-ngram-mod-size 20%
```

Semantics MUST be explicitly defined.

Recommended definition:

> Percentage of loaded model weight bytes, not parameter count.

This is intentionally an operational definition.

For example:

```text
GGUF/model weight bytes = 24 GB

--spec-ngram-mod-size 20%

ngram memory target ≈ 4.8 GB
```

The code SHOULD use the best reliable model-size value available within llama.cpp rather than relying on filesystem size if the model API already exposes actual loaded weight size.

The implementation MUST document exactly what value is used.

---

# 11. Functional Requirement: Persistent Cache

Add optional persistence specifically for `ngram-mod`.

Proposed interface:

```bash
--spec-ngram-mod-cache ./ngram-mod.cache
```

When provided:

### Startup

```text
cache exists
    │
    ├── yes → validate → load
    │
    └── no  → create empty memory
```

### Runtime

The memory continues learning normally.

### Shutdown

A final checkpoint SHOULD occur during graceful shutdown.

### Restart

The cache SHOULD restore the learned state.

---

# 12. Persistence Behaviour

Persistence MUST NOT require serialising after every token or request.

The cache SHOULD be considered dirty whenever mutations occur.

Recommended behaviour:

```text
mutation
   │
   ▼
dirty = true
   │
   ▼
periodic checkpoint interval reached
   │
   ├── dirty=false → skip
   │
   └── dirty=true  → checkpoint
```

Proposed option:

```bash
--spec-ngram-mod-save-interval 60
```

where the value represents seconds.

A value such as:

```bash
--spec-ngram-mod-save-interval 0
```

could disable periodic saving while retaining graceful-shutdown persistence.

Exact semantics should align with llama.cpp CLI conventions.

---

# 13. Crash-Safe Persistence

The implementation MUST avoid corrupting the existing valid cache during interrupted writes.

Recommended write path:

```text
ngram-mod.cache
      │
existing valid cache

save begins
      │
      ▼
ngram-mod.cache.tmp
      │
      ▼
write full contents
      │
      ▼
flush
      │
      ▼
close
      │
      ▼
atomic rename/replace
      │
      ▼
ngram-mod.cache
```

A failed checkpoint SHOULD leave the previous valid cache available.

Where platform semantics differ between Linux, macOS and Windows, existing llama.cpp atomic-file helpers or established project patterns SHOULD be preferred.

---

# 14. Cache File Format

The persistence format MUST be versioned.

Do NOT simply dump unversioned raw process memory.

Recommended conceptual header:

```cpp
struct ngram_mod_cache_header {
    uint64_t magic;
    uint32_t version;
    uint32_t header_size;

    uint64_t capacity;
    uint64_t entry_size;
    uint64_t entry_count;

    uint32_t n_match;

    uint64_t tokenizer_fingerprint;
    uint64_t vocabulary_fingerprint;

    uint64_t created_timestamp;
    uint64_t last_saved_timestamp;

    uint64_t inserts;
    uint64_t lookups;
    uint64_t predictions;
    uint64_t accepted_tokens;

    uint64_t flags;
};
```

The exact structure MAY differ.

Important properties are:

* identifiable format;
* explicit version;
* capacity encoded;
* entry representation encoded;
* tokenizer/vocabulary compatibility represented;
* validation possible before allocation/loading.

---

# 15. Tokenizer / Vocabulary Compatibility

Persisted token IDs are not universally meaningful.

A cache generated using one vocabulary MUST NOT silently be loaded into a model with an incompatible vocabulary.

For example:

```text
token ID 12345
```

can represent completely different tokens under different tokenizers.

Therefore cache loading MUST validate compatibility.

The implementation SHOULD derive a stable fingerprint from sufficient vocabulary/tokenizer information.

Possible inputs include:

* vocabulary size;
* token strings;
* token types;
* special-token mappings;
* tokenizer metadata.

A full GGUF/model-file hash is probably unnecessarily restrictive because two differently quantised versions of the same model family may have compatible token IDs.

Ideal behaviour:

```text
same tokenizer/vocab + different quant
           │
           ▼
cache can be reused
```

while:

```text
incompatible vocabulary
           │
           ▼
cache rejected
```

Failure MUST be explicit in logs.

The server SHOULD preferably fall back to a clean cache rather than fail model startup, unless a strict mode is later added.

---

# 16. Entry Representation

The current `ngram-mod` representation MUST be inspected before implementation.

Do not assume its exact struct layout.

However, scaling and persistence create new requirements.

The planning agent MUST evaluate whether the existing entry representation contains sufficient information to support:

* collision detection;
* persistence;
* future rehashing;
* runtime resizing;
* statistics.

If the current slot effectively stores only:

```text
next_token
```

then true dynamic resizing is difficult because the original hash cannot necessarily be reconstructed from the stored slot.

For scalable `ngram-mod`, consider storing a compact fingerprint.

Conceptual representation:

```cpp
struct ngram_mod_entry {
    uint32_t fingerprint;
    llama_token next_token;
};
```

Potentially packed into:

```cpp
uint64_t
```

where practical.

Advantages:

* distinguish many accidental hash collisions from genuine matches;
* measure collisions;
* support safer resizing;
* support rehashing if sufficient hash bits are retained;
* improve persistence validation;
* permit more useful telemetry.

The memory overhead MUST be measured before adopting this change.

---

# 17. Hash-Table Requirements

The implementation SHOULD preserve:

* fast deterministic addressing;
* power-of-two capacities where beneficial;
* cheap indexing;
* no heap allocation on each insert;
* no pointer-heavy node structures;
* cache-friendly entry representation;
* concurrency behaviour compatible with shared server slots.

Avoid converting the pool into:

```cpp
std::unordered_map<...>
```

unless benchmarks prove that doing so is acceptable.

The point of `ngram-mod` is to remain extremely cheap.

---

# 18. Large-Memory Considerations

Increasing the pool from:

```text
16 MB
```

to:

```text
4 GB
```

changes hardware behaviour even if algorithmic complexity remains O(1).

The implementation and benchmark plan MUST account for:

* CPU cache miss rate;
* DRAM latency;
* DRAM bandwidth;
* TLB pressure;
* page faults;
* NUMA placement;
* memory allocation time;
* server startup time;
* checkpoint time.

A larger table may increase acceptance while simultaneously increasing lookup cost.

End-to-end throughput is therefore the deciding metric.

---

# 19. RAM vs SSD Policy

The default architecture MUST use:

```text
RAM → active speculation
SSD → persistence
```

Do NOT make normal speculative lookup depend on random SSD access.

Direct SSD lookup is expected to have substantially higher latency than RAM and could destroy the low-overhead property of `ngram-mod`.

A future mmap-based cold tier MAY be explored, but it is not required for the first successful implementation.

---

# 20. Optional mmap Persistence Experiment

The design SHOULD leave room for an mmap mode.

Possible future interface:

```bash
--spec-ngram-mod-storage mmap
```

Architecture:

```text
persistent file
      │
      ▼
memory mapping
      │
      ▼
OS page cache
      │
      ▼
ngram-mod virtual address space
```

Potential advantages:

* persistence naturally backed by a file;
* reduced explicit load/copy;
* very fast warm restart;
* potentially larger virtual cache than physical RAM.

Risks:

* random hashing spreads accesses across pages;
* page faults can be much more expensive than ordinary lookups;
* the OS may evict useful pages;
* very large mappings may introduce TLB/page-cache effects.

This SHOULD be considered an experimental follow-up rather than mandatory MVP functionality.

---

# 21. Runtime Growth

The system SHOULD eventually support optional runtime growth.

Proposed CLI:

```bash
--spec-ngram-mod-grow
```

with:

```bash
--spec-ngram-mod-size 256M
--spec-ngram-mod-max-size 20%
```

Meaning:

```text
initial = 256 MB
maximum = 20% model weight size
```

The pool may increase but MUST never exceed the configured maximum.

---

# 22. Growth Strategy

Do NOT initially grow continuously in small increments.

Prefer geometric growth:

```text
256 MB
512 MB
1 GB
2 GB
4 GB
...
```

or:

```text
1%
2%
4%
8%
16%
20%
```

This:

* reduces resize frequency;
* simplifies benchmarking;
* makes behaviour predictable;
* avoids repeated large memory copies.

---

# 23. Growth Trigger

Growth MUST NOT simply occur because some number of tokens has passed.

Capacity should grow because telemetry indicates capacity pressure.

Candidate signals:

```text
slot overwrites
hash collisions
useful-entry eviction
pool utilisation
prediction hit rate
accepted prediction rate
```

The implementation agent MUST first determine which of these are observable with the chosen entry representation.

A possible trigger:

```text
measurement window reached
        │
        ▼
overwrite/collision pressure > threshold
        │
        AND
        ▼
accepted speculative predictions demonstrate usefulness
        │
        AND
        ▼
current size < maximum
        │
        ▼
grow
```

Exact thresholds MUST be benchmark-driven.

Do not bake arbitrary thresholds permanently without telemetry.

---

# 24. Rehashing Strategy

If pool capacity changes and slot addressing depends on:

```text
hash & (capacity - 1)
```

then existing entries cannot simply remain at their previous indexes.

A growth implementation therefore requires one of:

1. enough stored hash information to rehash entries;
2. a segmented table;
3. rebuilding from another retained representation.

The implementation agent MUST explicitly solve this problem.

---

# 25. Preferred Initial Dynamic-Growth Design

If full hash/fingerprint information can cheaply be retained, prefer:

```text
allocate new larger pool
        │
        ▼
rehash existing valid entries
        │
        ▼
atomically switch active pool
        │
        ▼
release old pool
```

Growth is expected to be rare, so occasional O(n) rebuild cost may be acceptable.

Measure it.

---

# 26. Alternative: Segmented Growth

If rehashing would significantly increase entry size, investigate segmented growth.

Example:

```text
Segment 0: 256 MB
Segment 1: 256 MB
Segment 2: 512 MB
Segment 3: 1 GB
```

New insertions could target the newest segment.

However, this potentially changes lookup from:

```text
one table lookup
```

to:

```text
multiple table lookups
```

which directly increases speculative overhead.

Therefore segmented growth MUST NOT be assumed superior.

Benchmark it against rehashing.

---

# 27. Shrinking

Automatic runtime shrinking is NOT required.

Preferred semantics:

```text
grow if useful
never shrink during normal runtime
```

Reasons:

* shrinking requires eviction decisions;
* shrink events could invalidate useful learned information;
* memory pressure policy is better handled through an explicit maximum initially;
* implementation complexity is not justified until growth is proven useful.

A future explicit compaction tool may be considered.

---

# 28. Persisted Dynamic Size

If the cache has grown to:

```text
3 GB
```

the next server start SHOULD NOT automatically reset it to the original:

```text
256 MB
```

when automatic sizing is enabled.

The cache file SHOULD contain its capacity.

Potential mode:

```bash
--spec-ngram-mod-size auto
```

Meaning:

```text
if valid persisted cache exists:
    use persisted capacity
else:
    use default/initial capacity
```

Alternative CLI naming MAY be chosen to better match llama.cpp conventions.

---

# 29. Model-Relative Maximum

Desired example:

```bash
llama-server \
    -m model.gguf \
    --spec-type ngram-mod \
    --spec-ngram-mod-size 256M \
    --spec-ngram-mod-max-size 20% \
    --spec-ngram-mod-grow \
    --spec-ngram-mod-cache ./model.ngrammod
```

Semantics:

```text
start at 256 MB

learn normally

if capacity pressure warrants:
    grow

maximum:
    20% of model weight bytes

persist both:
    contents
    current capacity
```

---

# 30. Multi-Slot Concurrency

Current `ngram-mod` has the useful property that its pool is shared between server slots.

This MUST be retained unless there is compelling evidence otherwise.

Example:

```text
Slot A ───┐
Slot B ───┤
Slot C ───┼──► shared ngram memory
Slot D ───┘
```

This allows:

```text
request A learns pattern
        │
        ▼
request B benefits later
```

Persistence extends that concept from:

```text
cross-request learning
```

to:

```text
cross-request + cross-process-lifetime learning
```

---

# 31. Thread Safety

The implementation MUST audit current `ngram-mod` concurrency semantics.

Specifically determine:

* whether inserts are atomic;
* whether reads race with writes;
* whether current races are deliberately benign;
* whether resizing introduces new unsafe pointer lifetime issues;
* whether checkpointing can safely read while inference writes;
* whether a global lock would harm throughput.

Do NOT introduce a lock around every lookup.

The fast lookup path should ideally remain lock-free or near lock-free.

---

# 32. Safe Runtime Resize

A runtime resize introduces a potentially difficult concurrency problem.

While:

```text
Thread A → lookup old pool
Thread B → insert old pool
Thread C → resize pool
```

the old allocation MUST NOT be freed while another worker is using it.

Candidate designs include:

* short global synchronisation at resize time;
* shared ownership / epoch-based swap;
* server-level safe point;
* stop speculative operations very briefly during resize.

Because growth should be rare, a short controlled pause may be substantially better than adding overhead to every lookup.

Optimise the common path, not the rare resize.

---

# 33. Persistence Concurrency

Checkpointing a multi-GB cache must not pause inference for seconds if it can be avoided.

Possible strategies:

### Option A: Stop-the-world copy

Simple but potentially expensive.

### Option B: Snapshot into a secondary buffer

Consistent but doubles temporary memory.

### Option C: Save table concurrently while accepting benign mutations

May be sufficient if individual entries are atomically readable.

A checkpoint need not necessarily represent a perfectly transactional snapshot of every recent insertion.

For this use case:

```text
slightly stale but valid
```

is preferable to:

```text
perfectly current but causes major inference pauses
```

The implementation agent should define the consistency guarantee explicitly.

---

# 34. Telemetry Requirements

This epic MUST add substantially better `ngram-mod` telemetry.

At minimum track:

```text
lookups
hits
misses

insertions

drafts generated
draft tokens generated

drafts accepted
accepted speculative tokens

rejected speculative tokens
```

Existing speculative statistics SHOULD be reused/extended rather than duplicated.

---

# 35. Capacity Telemetry

Where technically possible also track:

```text
configured capacity
actual allocated bytes

valid entries
occupancy

slot overwrites
fingerprint collisions

growth events
resize duration

cache loads
cache saves
load duration
save duration
cache file bytes
```

---

# 36. Quality Telemetry

Important derived metrics:

```text
lookup hit rate =
hits / lookups
```

```text
draft acceptance rate =
accepted draft tokens / generated draft tokens
```

```text
average accepted run length
```

```text
accepted speculative tokens per target verification
```

```text
predictions per lookup
```

Where possible:

```text
useful hit rate =
hits that produced >=1 accepted token / hits
```

This is more meaningful than simply measuring hash hits.

---

# 37. Performance Telemetry

Measure `ngram-mod` work separately from target-model work.

Useful metrics:

```text
ngram lookup ns/token or us/token
ngram insertion time
draft construction time
target verification time
```

Existing speculative duration metrics SHOULD be extended where appropriate.

---

# 38. Persistence Telemetry

On startup log:

```text
ngram-mod cache: loaded
path: ...
format version: ...
capacity: ...
entries: ...
size: ...
load time: ...
tokenizer fingerprint: matched
```

On rejection:

```text
ngram-mod cache incompatible:
reason = vocabulary fingerprint mismatch
```

On checkpoint:

```text
ngram-mod cache saved:
size = ...
duration = ...
```

Logs MUST avoid becoming noisy during ordinary inference.

---

# 39. Cache Warming

Persistence naturally enables warm startup.

Benchmark:

```text
cold
```

versus:

```text
warm
```

where both receive identical test workloads.

Expected experimental sequence:

```text
Run workload A
      │
      ▼
persist cache
      │
      ▼
restart server
      │
      ▼
Run workload A again
      │
      ▼
compare
```

---

# 40. Optional Corpus Pre-Seeding

The architecture SHOULD avoid making later corpus pre-seeding difficult.

A future utility could:

```text
input corpus
    │
    ▼
model tokenizer
    │
    ▼
token stream
    │
    ▼
same ngram insertion algorithm
    │
    ▼
persistent cache
```

Possible interface:

```bash
llama-ngram-mod-build \
    -m model.gguf \
    --input corpus.txt \
    --output corpus.ngrammod
```

This is NOT required for initial persistence but the file format SHOULD allow it later.

There is already upstream interest in pre-filling n-gram speculative data from external text, so interoperability with that general direction should be considered.

---

# 41. Potential Future Two-Tier Memory

If testing demonstrates that multi-GB caches improve hit coverage but harm lookup latency, the next architecture SHOULD investigate:

```text
               ngram lookup
                    │
                    ▼
           ┌────────────────┐
           │ HOT RAM TABLE  │
           │ 64-512 MB      │
           └───────┬────────┘
                   │ miss
                   ▼
           ┌────────────────┐
           │ LONG-TERM RAM  │
           │ or mmap TABLE  │
           │ multiple GB    │
           └────────────────┘
```

This is a later optimisation.

Do NOT build it unless benchmarking establishes a need.

---

# 42. Potential Future Admission Policy

With very large memories, retaining every one-off n-gram may become undesirable.

The current tiny pool naturally forgets data through collisions/overwrites.

A large pool may accidentally preserve low-value entries such as:

* UUID fragments;
* timestamps;
* random numbers;
* generated hashes;
* one-off identifiers;
* highly unique prose.

A later system could use an admission mechanism:

```text
new ngram
   │
   ▼
small probationary table
   │
observed repeatedly?
   │
   ├── no → naturally expire
   │
   └── yes
        │
        ▼
long-term memory
```

This SHOULD NOT block the first capacity experiments.

First determine whether raw additional capacity helps.

---

# 43. Potential Future Confidence Information

A future richer entry might track:

```text
next token
observation count
prediction count
acceptance count
```

For example:

```cpp
struct entry {
    uint32_t fingerprint;
    uint32_t token;
    uint16_t observations;
    uint16_t accepted;
};
```

This could support:

```text
confidence = accepted / predictions
```

or similar policies.

However, every metadata byte reduces the number of n-grams that fit within a given memory budget.

Any metadata expansion MUST therefore be justified experimentally.

---

# 44. Potential Future Multiple Continuations

Current-style mapping:

```text
ngram → one next token
```

loses multimodal continuations.

Example:

```text
"return "
    ├── nullptr
    ├── true
    ├── false
    └── result
```

A future representation might retain top-K candidate continuations.

This could resemble:

```text
ngram
   │
   ├── token A : count
   ├── token B : count
   └── token C : count
```

This is outside the initial epic unless required by measurements.

Do not increase complexity before proving that raw scale/persistence is useful.

---

# 45. Potential Future Multi-Head Hashing

Engram-inspired experimentation could later evaluate multiple independent hash functions/tables.

Conceptually:

```text
                 ngram
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
      hash A     hash B     hash C
        │          │          │
        ▼          ▼          ▼
      table A    table B    table C
```

Agreement could provide confidence.

Example:

```text
A → token 42
B → token 42
C → token 42

high confidence
```

versus:

```text
A → 42
B → 317
C → 91

low confidence
```

Again, this is a future research extension rather than initial scope.

---

# 46. CLI Design

The planning agent SHOULD review existing llama.cpp naming patterns before finalising option names.

Proposed options:

```text
--spec-ngram-mod-size SIZE
--spec-ngram-mod-max-size SIZE
--spec-ngram-mod-grow
--spec-ngram-mod-cache PATH
--spec-ngram-mod-save-interval N
```

Potential future:

```text
--spec-ngram-mod-storage ram|mmap
```

Expected examples:

### Fixed 1 GB persistent cache

```bash
llama-server \
    -m model.gguf \
    --spec-type ngram-mod \
    --spec-ngram-mod-size 1G \
    --spec-ngram-mod-cache ./cache.ngrammod
```

### 20% fixed cache

```bash
llama-server \
    -m model.gguf \
    --spec-type ngram-mod \
    --spec-ngram-mod-size 20% \
    --spec-ngram-mod-cache ./cache.ngrammod
```

### Adaptive cache

```bash
llama-server \
    -m model.gguf \
    --spec-type ngram-mod \
    --spec-ngram-mod-size 256M \
    --spec-ngram-mod-max-size 20% \
    --spec-ngram-mod-grow \
    --spec-ngram-mod-cache ./cache.ngrammod
```

---

# 47. Configuration Precedence

Define behaviour for combinations such as:

```text
persisted cache = 2 GB
CLI size = 1 GB
```

Recommended rules:

### Explicit fixed size

If user specifies:

```bash
--spec-ngram-mod-size 1G
```

then requested runtime capacity is 1 GB.

Loading a differently sized persistent cache should either:

* rehash/migrate it into 1 GB; or
* reject/reset it with a clear message.

Do not silently allocate a different amount than requested.

### Automatic mode

If user requests auto/resume semantics:

```text
persisted size wins subject to configured max
```

### Maximum

A persisted cache MUST NOT cause an allocation beyond the explicit configured maximum.

---

# 48. Corrupt Cache Handling

Test:

* truncated header;
* invalid magic;
* unsupported version;
* impossible capacity;
* entry-size mismatch;
* truncated body;
* extra data;
* checksum failure if checksums are implemented;
* incompatible tokenizer;
* incompatible n-match configuration.

Default behaviour should usually be:

```text
warn
ignore cache
start clean
```

rather than crashing the server.

Security-sensitive size fields MUST be bounds checked before allocating memory.

---

# 49. Cache Integrity

A checksum SHOULD be considered.

At minimum:

```text
header integrity
```

and ideally:

```text
payload integrity
```

However, hashing several GB at startup can add meaningful startup latency.

Possible approaches:

1. no full payload checksum, rely on atomic checkpoint writes and expected file size;
2. per-block checksums;
3. optional checksum;
4. fast non-cryptographic checksum.

The planning agent should choose the simplest mechanism that catches likely corruption without imposing disproportionate cost.

---

# 50. Backwards Compatibility

Default `ngram-mod` behaviour SHOULD remain close to upstream:

```text
small fixed memory
non-persistent
shared between slots
```

unless explicitly configured otherwise.

No user should unexpectedly receive:

```text
+4 GB RAM allocation
```

merely because they enabled `ngram-mod`.

The 20% behaviour must be opt-in.

---

# 51. Memory Allocation Failure

Large allocations may fail.

Example:

```bash
--spec-ngram-mod-size 20%
```

on a low-memory system.

The system MUST:

* detect allocation failure;
* emit a clear message;
* avoid undefined behaviour.

Possible policy:

```text
explicit fixed size requested
→ fail ngram-mod initialisation clearly
```

or:

```text
adaptive/auto allocation
→ fall back to smaller pool
```

The exact behaviour should follow llama.cpp conventions.

Silent memory reduction is undesirable unless clearly logged.

---

# 52. Operating-System Considerations

Testing should cover:

* Linux;
* Windows;
* macOS where feasible.

Important differences include:

* atomic file replacement;
* mmap semantics;
* filesystem flush behaviour;
* large allocation behaviour;
* page allocation;
* virtual memory;
* file locking.

Avoid Linux-only assumptions in common code.

---

# 53. NUMA Considerations

For multi-socket / NUMA systems, a multi-gigabyte random-access table may be sensitive to placement.

Do not necessarily implement explicit NUMA placement initially.

But benchmarks SHOULD record topology when testing very large pools.

If large-memory performance behaves unexpectedly, investigate:

* first-touch allocation;
* NUMA interleaving;
* remote-memory access.

---

# 54. Benchmarking Principle

Do NOT judge this work based only on:

```text
ngram hit rate
```

The primary metric is:

```text
end-to-end target generation throughput
```

because an improvement that produces more speculative predictions but slows total generation is a regression.

---

# 55. Required Benchmark Matrix

At minimum compare:

```text
No speculative decoding
Current/default ngram-mod
16 MB
64 MB
256 MB
1 GB
5% model size
10% model size
20% model size
```

Skip capacities that are impossible on the test system but document why.

---

# 56. Cold vs Warm Matrix

Each relevant capacity SHOULD have:

```text
cold cache
```

and:

```text
persisted warm cache
```

where possible.

Example:

| Capacity | Cold | Warm |
| -------- | ---- | ---- |
| 16 MB    | yes  | yes  |
| 64 MB    | yes  | yes  |
| 256 MB   | yes  | yes  |
| 1 GB     | yes  | yes  |
| 5%       | yes  | yes  |
| 10%      | yes  | yes  |
| 20%      | yes  | yes  |

---

# 57. Workload Categories

Do not benchmark only one repetitive prompt.

Include several workload classes.

## Coding / editing

* rewriting existing source files;
* generating similar functions;
* repository-style coding;
* repeated syntax;
* diffs;
* refactoring.

## Agentic

* tool-use traces;
* repeated command structures;
* iterative development;
* recurring prompts.

## Reasoning

* reasoning models that repeat/refine content.

## Summarisation

* summarising material containing repeated source phrases.

## General chat/prose

Control workload where persistent repetition may be weaker.

## Synthetic repetition

Useful for confirming upper-bound behaviour.

---

# 58. Benchmark Metrics

For every test capture:

```text
prompt tokens
generated tokens

prompt-processing tok/s
target-generation tok/s
end-to-end elapsed time

speculative calls
drafts generated

draft tokens generated
accepted draft tokens
rejected draft tokens

acceptance rate
mean accepted run length

ngram lookup time
ngram draft construction time

RAM usage
cache capacity
cache occupancy

cache load time
cache save time
```

Where available also capture:

```text
CPU cache misses
LLC misses
memory bandwidth
page faults
TLB misses
```

for selected deep-dive runs.

---

# 59. Success Criteria

The feature is successful if all core engineering goals are achieved even if the 20% performance hypothesis is disproven.

## Functional success

A persisted cache:

1. survives server restart;
2. reloads correctly;
3. produces identical or compatible n-gram behaviour after reload;
4. rejects incompatible vocabulary safely;
5. supports configurable capacities beyond the current default;
6. does not regress default behaviour.

## Performance success

At least one representative workload SHOULD demonstrate a measurable end-to-end gain from either:

```text
persistence
```

or:

```text
increased capacity
```

relative to current `ngram-mod`.

If no throughput gain exists, the benchmark evidence must clearly establish where the hypothesis fails.

That is still a valid experimental outcome.

---

# 60. Performance Guardrails

The new implementation MUST NOT materially regress default-size `ngram-mod`.

Suggested target:

```text
default 16 MB fast-path overhead:
within approximately 1-2% of upstream
```

unless a larger regression is explicitly justified by significantly improved accuracy.

The exact threshold may be adjusted based on benchmark noise.

Persistence disabled MUST impose effectively zero I/O overhead.

---

# 61. Persistence Acceptance Tests

Create tests covering:

### Round trip

```text
populate
save
destroy
load
verify entries
```

### Empty cache

```text
save empty
load empty
```

### Incompatible vocab

```text
save under vocab A
load under vocab B
→ rejected
```

### Truncated cache

```text
load
→ safely rejected
```

### Invalid version

```text
load
→ safely rejected
```

### Different configured capacity

Validate chosen migration/rejection semantics.

---

# 62. Hash / Entry Tests

Test:

* deterministic hash output;
* insertion;
* retrieval;
* collision handling;
* fingerprint matching;
* invalid-entry representation;
* full pool;
* overwrite behaviour;
* boundary capacities.

If capacity requires power-of-two sizing, test rounding/validation explicitly.

---

# 63. Resize Tests

If dynamic growth is implemented:

```text
populate small pool
grow
verify previously retrievable entries
verify new entries
```

Test multiple growth steps.

Test growth under active multi-slot use.

Test growth at maximum.

Test failed allocation.

Test checkpoint after growth.

Test restart after growth.

---

# 64. Concurrency Tests

Stress:

```text
many readers
many writers
checkpointing
resize
```

where relevant.

Use sanitizers where supported:

* ASan;
* UBSan;
* TSan where practical.

The test should deliberately attempt to expose:

* use-after-free;
* partially initialised pool;
* torn entry writes;
* races during swapping;
* save/load state corruption.

---

# 65. Long-Running Soak Test

Run a server for a substantial token volume.

Example:

```text
10M+
```

tokens if practical.

Measure over time:

```text
pool occupancy
overwrites
hit rate
acceptance
TG
memory consumption
```

Look for:

* degradation;
* cache pollution;
* runaway growth;
* save latency growth;
* statistical saturation.

---

# 66. Restart Soak Test

Repeatedly:

```text
start
load
process workload
save
stop
```

for many cycles.

Ensure:

* cache remains valid;
* file size remains sensible;
* startup behaviour remains stable;
* no cumulative corruption occurs.

---

# 67. Benchmark Against Existing ngram-cache

Since llama.cpp also contains `ngram-cache`, compare against it where appropriate.

The purpose is not necessarily to replace it.

Determine which design performs better for:

* persistent workloads;
* memory usage;
* long-running servers;
* coding;
* general prompts.

This is particularly important because existing `ngram-cache` already has concepts for dynamic/static statistics and persistence that may contain reusable implementation patterns.

---

# 68. Inspect Existing llama.cpp Infrastructure Before Coding

The implementation agent MUST audit current upstream code before creating new abstractions.

Specifically inspect:

```text
common/ngram-mod.*
common/ngram-cache.*
common/ngram-map.*
common/speculative.*
common/common.*
tools/server/*
tools/server/bench/speed-bench/*
```

Also inspect:

* existing binary persistence formats;
* model/vocab fingerprint helpers;
* size parsing helpers;
* mmap wrappers;
* atomic-file utilities;
* server shutdown hooks;
* logging conventions;
* speculative statistics infrastructure.

Prefer reuse over parallel infrastructure.

---

# 69. Upstream Drift

`ngram-mod` and speculative decoding are actively changing.

Before implementation:

1. record the target llama.cpp commit;
2. inspect current `master`;
3. confirm CLI names;
4. confirm current `ngram-mod` memory representation;
5. identify any recent speculative decoding changes;
6. design patches against current code rather than assumptions from older versions.

The epic requirements describe desired behaviour, not exact file/line locations.

---

# 70. Recent Upstream Behaviour Worth Considering

Recent llama.cpp development has highlighted how persistent/cross-request n-gram state can create incorrect or low-quality predictions if request-specific and global state are not clearly separated.

This reinforces an important requirement:

> Persistent global `ngram-mod` memory must contain only information intended to be shared globally. Per-request state must remain correctly reset between requests.

The implementation agent should audit `begin()`, slot lifecycle and accumulated history semantics rather than assuming existing lifecycle behaviour is automatically suitable for persistence.

---

# 71. User-Facing Documentation

Update speculative decoding documentation with:

* explanation of persistent `ngram-mod`;
* memory sizing;
* percentage sizing;
* disk vs RAM behaviour;
* compatibility rules;
* dynamic growth if implemented;
* examples;
* expected memory consumption;
* performance caveats.

Explicitly state:

> Larger is not necessarily faster.

and:

> `20%` is an experimental upper memory allocation, not a recommended universal default.

---

# 72. Example Documentation

Example:

```bash
# Existing behaviour
llama-server \
    -m model.gguf \
    --spec-type ngram-mod
```

```bash
# Persistent 512 MB speculative memory
llama-server \
    -m model.gguf \
    --spec-type ngram-mod \
    --spec-ngram-mod-size 512M \
    --spec-ngram-mod-cache ./model.ngram
```

```bash
# Allocate 10% of model weight size
llama-server \
    -m model.gguf \
    --spec-type ngram-mod \
    --spec-ngram-mod-size 10% \
    --spec-ngram-mod-cache ./model.ngram
```

```bash
# Start smaller and grow up to 20%
llama-server \
    -m model.gguf \
    --spec-type ngram-mod \
    --spec-ngram-mod-size 256M \
    --spec-ngram-mod-max-size 20% \
    --spec-ngram-mod-grow \
    --spec-ngram-mod-cache ./model.ngram
```

---

# 73. Recommended Delivery Stages

The work SHOULD be decomposed into incremental stages.

## Stage 0 -- Baseline Investigation

Before changing implementation:

* inspect current `ngram-mod`;
* document exact entry layout;
* document capacity calculation;
* document hashing;
* document concurrency;
* document lifecycle;
* benchmark current implementation;
* identify current test coverage.

Deliverable:

```text
baseline technical note + reproducible benchmarks
```

---

## Stage 1 -- Runtime-Configurable Fixed Capacity

Implement:

```text
--spec-ngram-mod-size
```

Support absolute sizes first.

Do NOT implement persistence yet if doing so complicates isolation.

Benchmark:

```text
16 MB
64 MB
256 MB
1 GB
```

This answers the first critical question:

> Does additional raw capacity provide any benefit?

---

## Stage 2 -- Percentage-Based Capacity

Add:

```text
5%
10%
20%
```

sizing.

Validate model-size calculation.

Add memory-limit handling.

Benchmark larger capacities.

---

## Stage 3 -- Persistence Format

Implement:

* versioned header;
* vocab compatibility;
* cache save;
* cache load;
* atomic replacement;
* graceful-shutdown save;
* round-trip tests.

Still use fixed capacity.

This isolates persistence correctness from dynamic resizing complexity.

---

## Stage 4 -- Periodic Checkpointing

Implement:

* dirty tracking;
* save interval;
* checkpoint telemetry;
* crash-safe replacement.

Benchmark checkpoint impact for:

```text
256 MB
1 GB
20%
```

---

## Stage 5 -- Extended Telemetry

Implement any missing:

* collisions;
* overwrites;
* occupancy;
* lookup hit rate;
* accepted-hit rate;
* lookup latency.

This instrumentation is required before designing automatic growth.

---

## Stage 6 -- Determine Whether Dynamic Growth Is Justified

Analyse evidence.

If:

```text
larger capacity improves TG
AND
different workloads prefer different sizes
```

then implement growth.

If:

```text
performance saturates early
```

dynamic growth may be unnecessary.

Do NOT implement complexity simply because it was initially proposed.

---

## Stage 7 -- Runtime Growth

If justified:

* add initial size;
* add maximum size;
* implement growth trigger;
* implement safe rehash/swap;
* persist grown size;
* test concurrency;
* benchmark resize pauses.

---

## Stage 8 -- Long-Term Memory Experiments

Only after core functionality is stable, consider:

* mmap-backed persistence;
* hot/cold tiers;
* corpus pre-seeding;
* admission policy;
* richer confidence metadata;
* multiple continuation candidates;
* multi-head hashing.

Each should be independently benchmarked.

---

# 74. Critical Decision Gates

## Gate A: Does larger fixed capacity improve speculation?

If NO:

Stop capacity work.

Investigate:

* prediction representation;
* confidence;
* n-gram length;
* continuation representation.

Persistence may still be valuable.

---

## Gate B: Does persistence improve warm workloads?

If NO:

Determine whether:

* workloads lack repetition;
* cache is polluted;
* cross-request information is harmful;
* entry retention policy needs improvement.

---

## Gate C: Does 1 GB+ remain fast enough?

If NO:

Investigate:

```text
hot/cold architecture
```

rather than blindly increasing memory.

---

## Gate D: Does performance continue increasing toward 20%?

If YES:

Large conditional speculative memory is worth pursuing.

If NO:

Set sensible optimum below 20%.

---

# 75. Research Questions the Agent Must Answer

The implementation/planning agent should explicitly investigate and answer:

1. What is the exact current `ngram-mod` entry representation?
2. Why is its current capacity approximately 16 MB?
3. Is the current size compile-time or runtime-derived?
4. Is a slot a full hash or only predicted-token storage?
5. Can current entries be rehashed?
6. What collision behaviour occurs today?
7. Can collision rate currently be measured?
8. What synchronisation exists for shared server-slot access?
9. Can entries be safely copied concurrently?
10. What existing binary persistence utilities exist?
11. How does `ngram-cache` persist data, and can anything be reused?
12. What existing model/vocabulary identity mechanism is suitable?
13. What is the most appropriate measure of loaded model bytes?
14. Can existing CLI size parsing support percentages?
15. Can SPEED-Bench directly produce all required metrics?
16. What benchmark corpus best represents repeat-heavy coding workloads?
17. At what table size does RAM/cache behaviour begin hurting lookup latency?
18. Does a persisted global cache improve or hurt unrelated requests?
19. Is one global cache sufficient, or would workload/domain segmentation eventually help?
20. Does scaling capacity improve acceptance because collisions fall, or simply because more history survives?

---

# 76. Architecture Principle: Measure Before Enriching Entries

A major risk is turning this project into a sophisticated n-gram database before proving the underlying hypothesis.

Follow this order:

```text
current representation
        │
        ▼
larger current representation
        │
        ▼
measure
        │
        ▼
persist
        │
        ▼
measure
        │
        ▼
only then improve representation
```

Do not immediately add:

```text
counts
LRU
LFU
timestamps
top-k tokens
multi-head hashes
probabilities
SSD hierarchy
```

unless a measurement demonstrates that the missing feature addresses an observed limitation.

---

# 77. Architecture Principle: Keep the Fast Path Stupid

Ideal lookup remains approximately:

```cpp
hash = rolling_hash(...);
index = hash & mask;
entry = pool[index];

if (entry_matches(entry, hash)) {
    return entry.token;
}

return no_prediction;
```

Avoid:

* heap allocation;
* mutex acquisition;
* filesystem access;
* complex trees;
* multiple pointer indirections;
* expensive statistics updates.

Anything occurring once per speculative token must be treated as performance-sensitive.

---

# 78. Architecture Principle: Spend Complexity Off the Fast Path

Expensive operations are acceptable when rare:

* startup;
* shutdown;
* checkpoint;
* resize;
* offline corpus build;
* compaction.

Therefore prefer:

```text
rare 100 ms resize
```

over:

```text
additional 100 ns every lookup forever
```

if the trade-off is otherwise equivalent.

---

# 79. Architecture Principle: Optimise End-to-End Generation

A speculative decoder exists to make:

```text
target model output arrive faster
```

not to maximise:

```text
its own prediction accuracy
```

Therefore a change that raises acceptance from:

```text
70% -> 75%
```

but lowers TG from:

```text
35 t/s -> 32 t/s
```

is a regression.

---

# 80. Expected Experimental Outcomes

Several results are plausible.

## Outcome A -- Capacity strongly helps

Example:

```text
16 MB   -> 30 t/s
256 MB  -> 34 t/s
1 GB    -> 38 t/s
4 GB    -> 42 t/s
```

Then large persistent memory is highly promising.

---

## Outcome B -- Rapid plateau

```text
16 MB  -> 30
64 MB  -> 34
256 MB -> 35
1 GB   -> 35
4 GB   -> 34
```

Then default/optimal memory should likely remain hundreds of MB.

The 20% budget would be unnecessary.

---

## Outcome C -- Persistence dominates size

```text
cold 1 GB -> 31
warm 1 GB -> 42
```

Then long-lived learning is the main feature.

---

## Outcome D -- Large memory harms latency

```text
acceptance rises
but TG falls
```

Then investigate hot/cold memory.

---

## Outcome E -- Raw hash memory becomes polluted

Long-run performance decreases despite increased capacity.

Then admission/retention policy becomes the next research problem.

---

# 81. Definition of Done

The core epic is complete when:

1. `ngram-mod` capacity can be configured at runtime.
2. Absolute memory sizes are supported.
3. Model-relative sizing is supported.
4. At least up to the requested ~20% model-size experiment can be attempted where sufficient system RAM exists.
5. `ngram-mod` can optionally save its learned memory.
6. Persisted memory can be loaded after restart.
7. Cache compatibility is validated against vocabulary/tokenizer identity.
8. Corrupt/incompatible cache files fail safely.
9. Saving is crash-safe.
10. Periodic checkpointing does not cause unacceptable inference stalls.
11. Existing shared-across-slots behaviour remains functional.
12. Default non-persistent behaviour remains backwards compatible.
13. Useful capacity/collision/acceptance/performance telemetry exists.
14. SPEED-Bench or an equivalent reproducible benchmark compares capacities.
15. Cold-vs-warm persistence benchmarks exist.
16. Performance has been measured from 16 MB through the practical upper range.
17. The results clearly establish whether 20%-scale memory is useful.
18. Dynamic growth has either:

    * been implemented based on evidence; or
    * explicitly rejected/deferred because measurements do not justify it.
19. Documentation explains all new options and trade-offs.
20. Automated tests cover persistence, compatibility, corruption and resizing where applicable.

---

# 82. Desired Final User Experience

A simple persistent deployment should eventually look approximately like:

```bash
llama-server \
    -m Qwen3.8-27B.gguf \
    --spec-type ngram-mod \
    --spec-ngram-mod-size 10% \
    --spec-ngram-mod-cache ./qwen38.ngram
```

The user should be able to stop the server and later restart it with:

```bash
llama-server \
    -m Qwen3.8-27B.gguf \
    --spec-type ngram-mod \
    --spec-ngram-mod-size 10% \
    --spec-ngram-mod-cache ./qwen38.ngram
```

and receive:

```text
ngram-mod: loading persistent memory
ngram-mod: vocabulary compatible
ngram-mod: capacity = ...
ngram-mod: loaded entries = ...
ngram-mod: ready
```

The speculative predictor should immediately begin from its previously learned state.

An adaptive deployment might eventually be:

```bash
llama-server \
    -m Qwen3.8-27B.gguf \
    --spec-type ngram-mod \
    --spec-ngram-mod-size 256M \
    --spec-ngram-mod-max-size 20% \
    --spec-ngram-mod-grow \
    --spec-ngram-mod-cache ./qwen38.ngram
```

with behaviour:

```text
Start small
   │
   ▼
learn
   │
   ▼
measure capacity pressure
   │
   ├── sufficient capacity -> stay
   │
   └── constrained
          │
          ▼
        grow
          │
          ▼
      persist new size
```

---

# 83. Longer-Term Vision

If the experiments prove successful, `ngram-mod` could evolve from:

```text
small ephemeral speculative hash cache
```

into:

```text
persistent workload-trained non-neural speculative memory
```

with a hierarchy such as:

```text
                  TARGET LLM
                      ^
                      | verifies
                      |
             speculative sequence
                      ^
                      |
            +---------------------+
            | Hot predictive RAM  |
            | hundreds of MB      |
            +----------+----------+
                       |
                  fallback lookup
                       |
            +----------v----------+
            | Long-term memory    |
            | multiple GB         |
            | RAM / mapped store  |
            +----------+----------+
                       |
                    persists
                       |
            +----------v----------+
            | SSD cache file      |
            | survives restart    |
            +---------------------+
```

Potential later enhancements could include:

```text
frequency-based admission
acceptance-derived confidence
multiple candidate continuations
multi-hash agreement
domain-specific memories
corpus-pretrained memories
cache import/export
cache merge
cache inspection tools
cache compaction
hot/cold tiering
mmap backing
```

But these should follow empirical evidence.

The immediate goal is deliberately narrower:

> Determine whether making `ngram-mod` persistent and substantially larger can convert spare system RAM into measurably faster target-model inference without sacrificing the extremely cheap lookup behaviour that makes `ngram-mod` attractive in the first place.

That result should determine the architecture of every subsequent optimisation.
