#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>

//
// common_ngram_mod
// ref: https://github.com/ggml-org/llama.cpp/pull/19164
//

static constexpr int32_t COMMON_NGRAM_MOD_EMPTY = -1;

// hash table slot: fingerprint + predicted token
struct ngram_mod_slot {
    uint32_t fingerprint; // upper bits of hash for collision detection
    int32_t  next_token;  // predicted token, COMMON_NGRAM_MOD_EMPTY if unused

    bool is_empty() const { return next_token == COMMON_NGRAM_MOD_EMPTY; }
    bool matches(uint32_t fp) const { return !is_empty() && fingerprint == fp; }
};

// basic n-gram hasher
struct common_ngram_mod {
    using entry_t = int32_t;

    static constexpr entry_t EMPTY = COMMON_NGRAM_MOD_EMPTY;

    common_ngram_mod(uint16_t n, size_t size);

    // hash n-gram tokens and return slot index
    size_t idx(const entry_t * tokens) const;

    // compute fingerprint for an n-gram
    uint32_t fp(const entry_t * tokens) const;

    // insert n-gram -> next token
    void add(const entry_t * tokens);

    // lookup: returns next token or EMPTY
    entry_t get(const entry_t * tokens) const;

    void reset();

    size_t get_n()    const;
    size_t get_used() const;

    size_t size()       const; // number of slots
    size_t size_bytes() const; // size in bytes

    const ngram_mod_slot * slots() const { return entries.data(); }
    ngram_mod_slot       * slots()       { return entries.data(); }

    // telemetry counters
    struct stats {
        size_t n_lookups      = 0;
        size_t n_hits         = 0; // fingerprint matched, non-empty slot
        size_t n_miss_fp      = 0; // slot occupied but fingerprint mismatched
        size_t n_inserts      = 0;
        size_t n_overwrites   = 0; // add() replaced a non-empty slot
    };

    const stats & get_stats() const { return m_stats; }
    void clear_stats() { m_stats = {}; }

private:
    size_t n;   // ngram size to hash
    size_t used;

    std::vector<ngram_mod_slot> entries;

    mutable stats m_stats;
};
