#include "ngram-mod.h"

#include <algorithm>

//
// common_ngram_mod
//

common_ngram_mod::common_ngram_mod(uint16_t n, size_t size) : n(n), used(0) {
    entries.resize(size);

    reset();
}

size_t common_ngram_mod::hash(const entry_t * tokens) const {
    size_t res = 0;

    for (size_t i = 0; i < n; ++i) {
        res = res*6364136223846793005ULL + tokens[i];
    }

    return res;
}

size_t common_ngram_mod::idx(const entry_t * tokens) const {
    // fingerprint index: cold-store entries carry only {fp, token}, so the
    // hot pool must place and find entries by fp for load-time selection
    return fp(tokens) % entries.size();
}

uint32_t common_ngram_mod::fp(const entry_t * tokens) const {
    // use upper 32 bits as fingerprint
    return (uint32_t)(hash(tokens) >> 32);
}

void common_ngram_mod::add(const entry_t * tokens) {
    m_stats.n_inserts++;

    const size_t i = idx(tokens);
    const uint32_t f = fp(tokens);

    if (!entries[i].is_empty()) {
        m_stats.n_overwrites++;
    } else {
        used++;
    }

    entries[i].fingerprint = f;
    entries[i].next_token  = tokens[n];
}

common_ngram_mod::entry_t common_ngram_mod::get(const entry_t * tokens) const {
    m_stats.n_lookups++;

    const size_t i = idx(tokens);
    const uint32_t f = fp(tokens);

    const auto & slot = entries[i];

    if (slot.is_empty()) {
        return EMPTY;
    }

    if (!slot.matches(f)) {
        m_stats.n_miss_fp++;
        return EMPTY;
    }

    m_stats.n_hits++;
    return slot.next_token;
}

void common_ngram_mod::reset() {
    ngram_mod_slot empty;
    empty.fingerprint = 0;
    empty.next_token  = EMPTY;
    std::fill(entries.begin(), entries.end(), empty);
    used = 0;
}

size_t common_ngram_mod::get_n() const {
    return n;
}

size_t common_ngram_mod::get_used() const {
    return used;
}

size_t common_ngram_mod::size() const {
    return entries.size();
}

size_t common_ngram_mod::size_bytes() const {
    return entries.size() * sizeof(ngram_mod_slot);
}
