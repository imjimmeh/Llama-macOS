#pragma once

#include "ngram-mod-cache.h"

// two-tier n-gram table: hot pool in RAM, cold store in a writable mapping.
// the cold store is keyed by fingerprint (cold idx = fp % cold size), so an
// evicted hot entry can be demoted from its stored {fingerprint, token} pair
// without rehashing the original n-gram
struct common_ngram_mod_tier {
    common_ngram_mod     hot;
    ngram_mod_cold_store cold;

    // consult the cold store on a hot miss (tiered mode only)
    bool cold_fallback = true;

    struct stats {
        size_t n_cold_lookups = 0; // hot misses that consulted the cold store
        size_t n_cold_hits    = 0; // cold lookups that matched
        size_t n_promotions   = 0; // cold hits promoted into the hot pool
        size_t n_demotions    = 0; // evicted hot entries written to cold
        size_t n_flushed      = 0; // hot entries written to cold on reset
    };

    stats t_stats;

    common_ngram_mod_tier(uint16_t n, size_t hot_slots) : hot(n, hot_slots) {}

    bool tiered() const { return cold.is_open(); }

    // insert: hot add; an evicted entry with a different fingerprint is
    // demoted to the cold store
    void add(const int32_t * tokens);

    // lookup: hot first, then the cold store when enabled; a cold hit is
    // promoted into the hot pool
    int32_t get(const int32_t * tokens);

    // write all hot entries to cold (preserving cold hit counters), then
    // reset the hot pool
    void flush_reset();
};
