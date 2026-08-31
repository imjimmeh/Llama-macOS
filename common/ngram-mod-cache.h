#pragma once

#include "ngram-mod.h"

#include <cstdint>
#include <string>

// cache file format version
// v1: [header][slots][footer], slots = full hot pool dump, 2 GB load cap
// v2: [header][cold slots][cold hits][footer], cold store mapped read-write
static constexpr uint32_t NGRAM_MOD_CACHE_VERSION = 2;

// magic: "NGMD"
static constexpr uint32_t NGRAM_MOD_CACHE_MAGIC = 0x4E474D44;

// max allowed slot count to prevent absurd allocations from corrupt files
// (v1 hot pool: slots are read into RAM in one fread)
static constexpr uint64_t NGRAM_MOD_CACHE_MAX_SLOTS = 256ULL * 1024 * 1024; // 256M slots = 2 GB

// max cold store slot count (mapped, never read into RAM; 8G slots = 96 GB)
static constexpr uint64_t NGRAM_MOD_CACHE_MAX_COLD_SLOTS = 8ULL * 1024 * 1024 * 1024;

// header flags
static constexpr uint32_t NGRAM_MOD_CACHE_FLAG_TIERED      = 1u << 0;
static constexpr uint32_t NGRAM_MOD_CACHE_FLAG_CLEAN_CLOSE = 1u << 1;

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

    // v2 additions; v1 files end at hits_size (header_size = offsetof(..., hits_size))
    uint32_t hits_size;         // 0 in v1 and non-tiered v2; sizeof(uint32_t) when tiered
    uint64_t capacity_cold;     // cold store slots; 0 = non-tiered
    uint64_t cold_entry_count;  // non-empty cold slots at save time
    uint32_t flags;             // NGRAM_MOD_CACHE_FLAG_*
};

// size of the v1 header (fields up to and including total_hits)
static constexpr size_t NGRAM_MOD_CACHE_HEADER_V1_SIZE = offsetof(ngram_mod_cache_header, hits_size);

// payload checksum (appended after the slot/hits arrays)
struct ngram_mod_cache_footer {
    uint64_t payload_checksum; // xxh64 of the slot array (and hits array when tiered)
};

// writable memory mapping (POSIX mmap / Windows CreateFileMapping)
struct ngram_mod_mmap {
    void * data = nullptr;
    size_t size = 0;

#ifdef _WIN32
    void * h_file = nullptr; // HANDLE
    void * h_map  = nullptr; // HANDLE
#else
    int fd = -1;
#endif

    ngram_mod_mmap() = default;
    ~ngram_mod_mmap();

    ngram_mod_mmap(const ngram_mod_mmap &) = delete;
    ngram_mod_mmap & operator=(const ngram_mod_mmap &) = delete;

    // create/truncate the file to size and map it read-write
    bool open(const std::string & path, size_t size);
    bool flush();
    void close();
};

// format v2 cold store: [header][cold slots: cap*8 B][cold hits: cap*4 B][footer]
// one cold slot maps 1:1 to a hot pool slot by index
struct ngram_mod_cold_store {
    ngram_mod_mmap mmap;

    ngram_mod_slot * slots = nullptr;
    uint32_t       * hits  = nullptr;

    size_t capacity = 0; // cold slots
    size_t used     = 0; // non-empty cold slots (best-effort bookkeeping)

    bool dirty = false; // footer/header need writing back to the mapping

    ngram_mod_cache_header header; // in-memory copy of the file header

    bool open_ok = false;

    ~ngram_mod_cold_store();

    ngram_mod_cold_store() = default;
    ngram_mod_cold_store(const ngram_mod_cold_store &) = delete;
    ngram_mod_cold_store & operator=(const ngram_mod_cold_store &) = delete;

    // open/create the cold store at the configured capacity.
    // an existing v1 file is upgraded in place to v2 (hits = 0).
    // returns true when tiering is active
    bool open(
            const std::string & path,
            size_t capacity_cold,
            uint32_t n_match,
            uint32_t vocab_size,
            const char * tokenizer_model,
            const char * tokenizer_pre);

    // write footer checksum + clean-close flag, then flush to disk
    bool flush();
    void close();

    bool is_open() const { return open_ok; }
    size_t size() const { return capacity; }
    size_t used_count() const { return used; }
    bool is_dirty() const { return dirty; }

    const ngram_mod_slot * get_slot(size_t idx) const;
    uint32_t get_hits(size_t idx) const;
    void set_slot(size_t idx, const ngram_mod_slot & s, uint32_t h);
    void bump_hits(size_t idx);
    void recount_used();
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

// load cache from disk (non-tiered: reads the whole pool into RAM)
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
