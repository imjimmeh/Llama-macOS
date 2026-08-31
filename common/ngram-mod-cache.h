#pragma once

#include "ngram-mod.h"

#include <cstdint>
#include <string>

// cache file format version
static constexpr uint32_t NGRAM_MOD_CACHE_VERSION = 1;

// magic: "NGMD"
static constexpr uint32_t NGRAM_MOD_CACHE_MAGIC = 0x4E474D44;

// max allowed slot count to prevent absurd allocations from corrupt files
static constexpr uint64_t NGRAM_MOD_CACHE_MAX_SLOTS = 256ULL * 1024 * 1024; // 256M slots = 2 GB

struct ngram_mod_cache_header {
    uint32_t magic;
    uint32_t version;
    uint32_t header_size;   // sizeof(ngram_mod_cache_header)
    uint32_t slot_size;     // sizeof(ngram_mod_slot)

    uint64_t capacity;      // number of slots in the pool
    uint64_t entry_count;   // non-empty slots at save time

    uint32_t n_match;       // n-gram lookup length

    // tokenizer/vocab compatibility
    uint32_t vocab_size;
    char     tokenizer_model[64]; // tokenizer.ggml.model
    char     tokenizer_pre[64];   // tokenizer.ggml.pre

    // timestamps
    uint64_t created_ts;
    uint64_t saved_ts;

    // aggregate stats at save time
    uint64_t total_inserts;
    uint64_t total_lookups;
    uint64_t total_hits;
};

// payload checksum (appended after slot array)
struct ngram_mod_cache_footer {
    uint64_t payload_checksum; // xxh64 of slot array
};

// populate header fields from current state
void ngram_mod_cache_fill_header(
    ngram_mod_cache_header & header,
    const common_ngram_mod & mod,
    uint32_t n_match,
    uint32_t vocab_size,
    const char * tokenizer_model,
    const char * tokenizer_pre);

// save cache to disk (crash-safe: writes to .tmp then renames)
// returns true on success
bool ngram_mod_cache_save(
    const std::string & path,
    const common_ngram_mod & mod,
    const ngram_mod_cache_header & header);

// load cache from disk
// on success: populates mod with saved slots, returns true
// on failure: returns false, mod is unchanged
bool ngram_mod_cache_load(
    const std::string & path,
    common_ngram_mod & mod,
    ngram_mod_cache_header & header,
    uint32_t expected_n_match,
    uint32_t expected_vocab_size,
    const char * expected_tokenizer_model,
    const char * expected_tokenizer_pre);
