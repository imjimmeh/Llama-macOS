// ngram-mod pool: occupancy/collision behaviour micro-benchmark
//
// Measures how lookup hit rate and overwrite rate degrade as a direct-mapped
// hash pool fills up. The curve is scale-free (determined by occupancy, not
// absolute slot count), so small pools stand in for the real 256M/1G/4G sizes.

#include "ngram-mod.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
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

    printf("test-ngram-mod: OK\n");
    return 0;
}
