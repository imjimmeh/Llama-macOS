#include "ngram-mod-tier.h"

#include <vector>

void common_ngram_mod_tier::add(const int32_t * tokens) {
    if (tiered()) {
        const size_t   h = hot.hash(tokens);
        const uint32_t f = (uint32_t)(h >> 32);
        const size_t   i = f % hot.size(); // slot hot.add() will overwrite

        const auto & slot = hot.slots()[i];
        if (!slot.is_empty() && !slot.matches(f)) {
            // keep the evicted entry in the cold store, preserving its hits
            const size_t j = slot.fingerprint % cold.size();
            const auto * cs = cold.get_slot(j);
            const uint32_t hits = (cs && cs->matches(slot.fingerprint)) ? cold.get_hits(j) : 0;
            cold.set_slot(j, slot, hits);
            t_stats.n_demotions++;
        }
    }

    hot.add(tokens);
}

int32_t common_ngram_mod_tier::get(const int32_t * tokens) {
    const int32_t tok = hot.get(tokens);
    if (tok != COMMON_NGRAM_MOD_EMPTY || !tiered() || !cold_fallback) {
        return tok;
    }

    t_stats.n_cold_lookups++;

    const uint32_t f = (uint32_t)(hot.hash(tokens) >> 32);
    const size_t   j = f % cold.size();

    const auto * cs = cold.get_slot(j);
    if (cs == nullptr || !cs->matches(f)) {
        return COMMON_NGRAM_MOD_EMPTY;
    }

    t_stats.n_cold_hits++;

    // promote into the hot pool: tokens[n] = predicted token
    const size_t n = hot.get_n();
    std::vector<int32_t> promo(tokens, tokens + n);
    promo.push_back(cs->next_token);
    add(promo.data());
    t_stats.n_promotions++;

    return cs->next_token;
}

void common_ngram_mod_tier::flush_reset() {
    if (!tiered()) {
        hot.reset();
        return;
    }

    const size_t n_slots = hot.size();
    auto * slots = hot.slots();

    for (size_t i = 0; i < n_slots; i++) {
        if (slots[i].is_empty()) {
            continue;
        }
        const size_t j = slots[i].fingerprint % cold.size();
        const auto * cs = cold.get_slot(j);
        const uint32_t hits = (cs && cs->matches(slots[i].fingerprint)) ? cold.get_hits(j) : 0;
        cold.set_slot(j, slots[i], hits);
        t_stats.n_flushed++;
    }

    hot.reset();
}

size_t common_ngram_mod_tier::load_hot_from_cold() {
    if (!tiered()) {
        return 0;
    }

    const size_t hot_cap = hot.size();
    auto * hslots = hot.slots();
    std::vector<uint32_t> hot_hits(hot_cap, 0); // parallel hotness, hot slots have no counter

    for (size_t j = 0; j < cold.size(); j++) {
        const auto * cs = cold.get_slot(j);
        if (cs == nullptr) {
            continue;
        }
        const uint32_t hits = cold.get_hits(j);
        const size_t i = cs->fingerprint % hot_cap;
        auto & hs = hslots[i];
        if (hs.is_empty() || hot_hits[i] < hits) {
            hs = *cs;
            hot_hits[i] = hits;
        }
    }

    hot.recount_used();
    t_stats.n_hot_loaded = hot.get_used();
    return t_stats.n_hot_loaded;
}
