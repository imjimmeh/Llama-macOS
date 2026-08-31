// ngram-mod pool: occupancy/collision behaviour micro-benchmark
// plus format v1/v2 persistence round-trips
//
// Measures how lookup hit rate and overwrite rate degrade as a direct-mapped
// hash pool fills up. The curve is scale-free (determined by occupancy, not
// absolute slot count), so small pools stand in for the real 256M/1G/4G sizes.

#include "ngram-mod-cache.h"
#include "ngram-mod-tier.h"
#include "ngram-mod.h"
#include <map>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static void require_impl(bool condition, const char * expression, const char * file, int line) {
    if (!condition) {
        fprintf(stderr, "test requirement failed: %s:%d: %s\n", file, line, expression);
        fflush(stderr);
        abort();
    }
}

#define require(condition) require_impl((condition), #condition, __FILE__, __LINE__)

// insert K n-grams into a pool, then measure lookup hit rate on a sparse
// sample of the inserted sequences plus the overwrite rate
static void test_occupancy(size_t size, double target, uint16_t n) {
    const size_t K = (size_t)(size * target);
    const size_t SAMPLE_STEP = K > 100000 ? K / 100000 : 1; // sample ~100k lookups

    common_ngram_mod mod(n, size);

    std::mt19937 rng(42);
    std::vector<int32_t> seq(n + 1);
    std::vector<int32_t> sampled; // every SAMPLE_STEP-th inserted sequence

    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j <= n; j++) {
            seq[j] = (int32_t)(rng() % 100000);
        }
        if (i % SAMPLE_STEP == 0) {
            sampled.insert(sampled.end(), seq.begin(), seq.end());
        }
        mod.add(seq.data());
    }

    const auto & stats = mod.get_stats();
    const double overwrite_rate = stats.n_inserts > 0 ? (double) stats.n_overwrites / stats.n_inserts : 0.0;

    size_t hits = 0;
    const size_t n_sample = sampled.size() / (n + 1);
    for (size_t i = 0; i < n_sample; i++) {
        const int32_t * s = sampled.data() + i * (n + 1);
        if (mod.get(s) == s[n]) {
            hits++;
        }
    }
    const double hit_rate = (double) hits / n_sample;

    printf("  %9" PRIu64 " slots  occupancy %5.2f  inserts %9" PRIu64
           "  overwrite %7.4f  hit_rate %7.4f\n",
           (uint64_t) size, target, (uint64_t) K, overwrite_rate, hit_rate);
}

// under saturation, lookups of never-inserted n-grams must not produce hits
// (fingerprint mismatch), except for ~1/2^32 accidental collisions
static void test_false_hits(uint16_t n) {
    const size_t size = 1U << 20;
    const size_t K = size; // 100% occupancy

    common_ngram_mod mod(n, size);

    std::mt19937 rng(7);
    std::vector<int32_t> seq(n + 1);
    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j <= n; j++) {
            seq[j] = (int32_t)(rng() % 100000);
        }
        mod.add(seq.data());
    }

    size_t false_hits = 0;
    const size_t N = 100000;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j <= n; j++) {
            seq[j] = (int32_t)(rng() % 100000);
        }
        if (mod.get(seq.data()) != common_ngram_mod::EMPTY) {
            false_hits++;
        }
    }
    printf("  false hits at 100%% occupancy: %zu / %zu\n", false_hits, N);
    require(false_hits < 10); // expects ~0 (1/2^32 per lookup)
}

// ---- format v1/v2 persistence ----

static const char * COLD_PATH = "test-ngram-mod-cold.bin";
static const char * V1_PATH   = "test-ngram-mod-v1.bin";

// v1 save/load round-trip (tiering off path)
static void test_v1_round_trip() {
    const size_t cap = 1U << 16;
    common_ngram_mod mod(24, cap);

    std::mt19937 rng(1);
    std::vector<int32_t> seq(25);
    for (int i = 0; i < 500; i++) {
        for (auto & t : seq) {
            t = (int32_t)(rng() % 1000);
        }
        mod.add(seq.data());
    }

    ngram_mod_cache_header header;
    ngram_mod_cache_fill_header(header, mod, 24, 0, "", "");
    require(ngram_mod_cache_save(V1_PATH, mod, header));

    common_ngram_mod loaded(24, cap);
    ngram_mod_cache_header lh;
    require(ngram_mod_cache_load(V1_PATH, loaded, lh, 24, 0, "", ""));
    require(loaded.get_used() == mod.get_used());
    require(memcmp(loaded.slots(), mod.slots(), cap * sizeof(ngram_mod_slot)) == 0);

    remove(V1_PATH);
}

// write a genuine v1 file (version 1 header + slots + footer)
static void write_v1_file(const char * path, const common_ngram_mod & mod, uint32_t n_match) {
    FILE * f = fopen(path, "wb");
    require(f != nullptr);

    ngram_mod_cache_header h;
    memset(&h, 0, sizeof(h));
    h.magic       = NGRAM_MOD_CACHE_MAGIC;
    h.version     = 1;
    h.header_size = NGRAM_MOD_CACHE_HEADER_V1_SIZE;
    h.slot_size   = sizeof(ngram_mod_slot);
    h.capacity    = mod.size();
    h.entry_count = mod.get_used();
    h.n_match     = n_match;

    const ngram_mod_cache_footer footer = { 0 };

    require(fwrite(&h, 1, NGRAM_MOD_CACHE_HEADER_V1_SIZE, f) == NGRAM_MOD_CACHE_HEADER_V1_SIZE);
    require(fwrite(mod.slots(), 1, mod.size() * sizeof(ngram_mod_slot), f) == mod.size() * sizeof(ngram_mod_slot));
    require(fwrite(&footer, 1, sizeof(footer), f) == sizeof(footer));
    fclose(f);
}

// v2 cold store: create, write, flush, reopen, crash recovery, validation
static void test_cold_store_round_trip() {
    const size_t cap = 1U << 20; // 1M cold slots = 12 MB file
    remove(COLD_PATH);

    ngram_mod_cold_store store;
    require(store.open(COLD_PATH, cap, 24, 0, "", ""));
    require(store.is_open());
    require(store.size() == cap);
    require(store.used_count() == 0);

    ngram_mod_slot s;
    s.fingerprint = 0xDEADBEEF;
    s.next_token  = 42;
    store.set_slot(10, s, 7);
    require(store.used_count() == 1);

    s.fingerprint = 0x12345678;
    s.next_token  = -3;
    store.set_slot(10, s, 9); // overwrite in place
    require(store.used_count() == 1);

    s.fingerprint = 0xCAFEBABE;
    s.next_token  = 1000;
    store.set_slot(123456, s, 1);
    require(store.used_count() == 2);

    require(store.flush());

    // header round-trips: tiered v2 header, expected file size
    {
        FILE * f = fopen(COLD_PATH, "rb");
        require(f != nullptr);
        ngram_mod_cache_header h;
        memset(&h, 0, sizeof(h));
        require(fread(&h, 1, sizeof(h), f) == sizeof(h));
        require(h.magic == NGRAM_MOD_CACHE_MAGIC);
        require(h.version == 2);
        require(h.header_size == sizeof(ngram_mod_cache_header));
        require(h.hits_size == sizeof(uint32_t));
        require(h.capacity_cold == cap);
        require(h.flags & NGRAM_MOD_CACHE_FLAG_TIERED);
        require(h.flags & NGRAM_MOD_CACHE_FLAG_CLEAN_CLOSE);
        require(h.cold_entry_count == 2);

        const size_t expect = sizeof(ngram_mod_cache_header)
            + cap * (sizeof(ngram_mod_slot) + sizeof(uint32_t))
            + sizeof(ngram_mod_cache_footer);
        require(fseek(f, 0, SEEK_END) == 0);
        require(ftell(f) == (long) expect);
        fclose(f);
    }

    // reopen: slots + hits persist
    {
        ngram_mod_cold_store s2;
        require(s2.open(COLD_PATH, cap, 24, 0, "", ""));
        const ngram_mod_slot * got = s2.get_slot(10);
        require(got != nullptr);
        require(got->fingerprint == 0x12345678);
        require(got->next_token == -3);
        require(s2.get_hits(10) == 9);
        got = s2.get_slot(123456);
        require(got != nullptr);
        require(got->fingerprint == 0xCAFEBABE);
        require(s2.get_hits(123456) == 1);
        require(s2.get_slot(0) == nullptr); // empty slot
        require(s2.used_count() == 2);
    }

    // crash: clear the clean-close flag; reopen must still work (checksum skipped)
    {
        FILE * f = fopen(COLD_PATH, "rb+");
        require(f != nullptr);
        const long off = (long) offsetof(ngram_mod_cache_header, flags);
        require(fseek(f, off, SEEK_SET) == 0);
        uint32_t flags = 0;
        require(fread(&flags, 1, sizeof(flags), f) == sizeof(flags));
        flags &= ~NGRAM_MOD_CACHE_FLAG_CLEAN_CLOSE;
        require(fseek(f, off, SEEK_SET) == 0);
        require(fwrite(&flags, 1, sizeof(flags), f) == sizeof(flags));
        fclose(f);

        ngram_mod_cold_store s3;
        require(s3.open(COLD_PATH, cap, 24, 0, "", ""));
        const ngram_mod_slot * got = s3.get_slot(10);
        require(got != nullptr);
        require(got->fingerprint == 0x12345678);
        require(s3.is_dirty()); // crashed store needs a footer rewrite
    }

    // header validation: wrong n_match must be rejected
    {
        ngram_mod_cold_store s4;
        require(!s4.open(COLD_PATH, cap, 32, 0, "", ""));
    }

    store.close(); // release the mapping so remove() can delete the file
    remove(COLD_PATH);
}

// v1 cache file upgraded in place to a v2 cold store (hits = 0)
static void test_v1_upgrade() {
    const size_t v1_cap = 1U << 16;
    common_ngram_mod mod(24, v1_cap);

    std::mt19937 rng(2);
    std::vector<int32_t> seq(25);
    for (int i = 0; i < 1000; i++) {
        for (auto & t : seq) {
            t = (int32_t)(rng() % 5000);
        }
        mod.add(seq.data());
    }
    const size_t v1_used = mod.get_used();
    require(v1_used > 0);

    write_v1_file(V1_PATH, mod, 24);

    const size_t cold_cap = 1U << 18; // 4x the v1 capacity
    ngram_mod_cold_store store;
    require(store.open(V1_PATH, cold_cap, 24, 0, "", ""));
    require(store.size() == cold_cap);
    // entries land at their fingerprint index; later entries win collisions
    const ngram_mod_slot * src = mod.slots();
    std::map<size_t, ngram_mod_slot> expected;
    for (size_t i = 0; i < v1_cap; i++) {
        if (!src[i].is_empty()) {
            expected[src[i].fingerprint % cold_cap] = src[i];
        }
    }
    require(store.used_count() == expected.size());
    for (const auto & kv : expected) {
        const ngram_mod_slot * got = store.get_slot(kv.first);
        require(got != nullptr);
        require(got->fingerprint == kv.second.fingerprint);
        require(got->next_token == kv.second.next_token);
        require(store.get_hits(kv.first) == 0);
    }
    require(store.used_count() == expected.size());
    require(store.flush());

    // upgraded file reopens as a valid tiered v2 store
    {
        ngram_mod_cold_store s2;
        require(s2.open(V1_PATH, cold_cap, 24, 0, "", ""));
        require(s2.used_count() == expected.size());
    }

    store.close(); // release the mapping so remove() can delete the file
    remove(V1_PATH);
}

// tier: hot add/get with demotion, cold fallback, promotion, fallback gate
static void test_tier_hot_cold() {
    const char * path = "test-ngram-mod-tier.bin";
    remove(path);

    common_ngram_mod_tier tier(3, 64);
    require(tier.cold.open(path, 1024, 3, 0, "", ""));
    require(tier.tiered());

    int32_t a[4] = {1, 2, 3, 10};
    // find a sequence that collides with a on the hot index but differs in
    // fingerprint, so adding it evicts a; vary a middle token so the
    // fingerprint changes freely while the index repeats
    int32_t b[4] = {1, 0, 0, 20};
    bool found = false;
    for (int32_t x = 4; x < 100000; x++) {
        b[1] = x;
        if (tier.hot.idx(a) == tier.hot.idx(b) && tier.hot.fp(a) != tier.hot.fp(b)) {
            found = true;
            break;
        }
    }
    require(found);

    tier.add(a);
    require(tier.get(a) == 10); // hot hit

    tier.add(b); // evicts a -> demoted to cold
    require(tier.t_stats.n_demotions == 1);
    require(tier.get(b) == 20); // hot hit

    // a now lives in cold; the lookup promotes it back into the hot pool
    require(tier.get(a) == 10);
    require(tier.t_stats.n_cold_lookups == 1);
    require(tier.t_stats.n_cold_hits == 1);
    require(tier.t_stats.n_promotions == 1);
    require(tier.get(a) == 10); // hot hit again, no extra cold lookup
    require(tier.t_stats.n_cold_lookups == 1);

    // fallback off: hot miss returns EMPTY without consulting cold
    tier.cold_fallback = false;
    int32_t c[4] = {7, 8, 9, 30};
    require(tier.get(c) == COMMON_NGRAM_MOD_EMPTY);
    require(tier.t_stats.n_cold_lookups == 1); // unchanged

    tier.cold.close(); // release the mapping so remove() can delete the file
    remove(path);
}

// tier: flush_reset moves hot entries to cold and they survive a reopen
static void test_tier_flush_reset() {
    const char * path = "test-ngram-mod-tier.bin";
    remove(path);

    const size_t cold_cap = 1U << 16;
    common_ngram_mod_tier tier(3, 64);
    require(tier.cold.open(path, cold_cap, 3, 0, "", ""));

    // vary a middle token: consecutive last tokens give consecutive hashes,
    // which mostly share a fingerprint and pile onto one cold slot
    for (int32_t x = 0; x < 24; x++) {
        int32_t t[4] = {100, x, 300, 1000 + x};
        tier.add(t);
    }
    require(tier.hot.get_used() == 24);

    tier.flush_reset();
    require(tier.hot.get_used() == 0);
    require(tier.t_stats.n_flushed == 24);
    require(tier.cold.used_count() == 24); // distinct fp indices for this data

    // entries are found via the cold store and promoted on hit
    int32_t q[3] = {100, 5, 300};
    require(tier.get(q) == 1005);
    require(tier.t_stats.n_cold_hits == 1);
    require(tier.hot.get_used() == 1);
    require(tier.cold.flush());

    // a fresh tier over the same file must find the flushed entries
    {
        common_ngram_mod_tier tier2(3, 64);
        require(tier2.cold.open(path, cold_cap, 3, 0, "", ""));
        int32_t q2[3] = {100, 10, 300};
        require(tier2.get(q2) == 1010);
        int32_t q3[3] = {100, 999, 300};
        require(tier2.get(q3) == COMMON_NGRAM_MOD_EMPTY);
        tier2.cold.close();
    }

    tier.cold.close(); // release the mapping so remove() can delete the file
    remove(path);
}

int main() {
    const uint16_t n = 24; // matches --spec-ngram-mod-n-match default

    printf("test-ngram-mod: occupancy/collision sweep (n=%u)\n", n);
    printf("  scale-free curve; 1M/4M/16M slots stand in for 256M/1G/4G pools\n");
    printf("  (slot count scales capacity; occupancy determines collision damage)\n");

    const size_t sizes[] = { 1U << 20, 4U << 20, 16U << 20 };
    const double targets[] = { 0.05, 0.10, 0.25, 0.50, 0.75, 1.00 };

    for (size_t size : sizes) {
        for (double target : targets) {
            test_occupancy(size, target, n);
        }
    }

    test_false_hits(n);

    printf("test-ngram-mod: persistence round-trips\n");
    test_v1_round_trip();
    test_tier_hot_cold();
    test_tier_flush_reset();
    test_v1_upgrade();

    printf("test-ngram-mod: OK\n");
    return 0;
}
