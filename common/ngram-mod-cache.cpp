#include "ngram-mod-cache.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

// simple xxh64 - matches ggml's xxhash for consistency
static uint64_t xxh64(const void * data, size_t len) {
    const uint8_t * p = (const uint8_t *)data;
    uint64_t h = 0x1656679197377081ULL;

    for (size_t i = 0; i + 8 <= len; i += 8) {
        uint64_t v;
        memcpy(&v, p + i, 8);
        h ^= v;
        h = h * 0x9E3779B97F4A7C15ULL;
        h = (h << 31) | (h >> 33);
    }

    // process remaining bytes
    for (size_t i = len & ~7ULL; i < len; i++) {
        h ^= (uint64_t)p[i] << ((i & 7) * 8);
    }

    // final mix
    h ^= len;
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 33;

    return h;
}

void ngram_mod_cache_fill_header(
        ngram_mod_cache_header & header,
        const common_ngram_mod & mod,
        uint32_t n_match,
        uint32_t vocab_size,
        const char * tokenizer_model,
        const char * tokenizer_pre) {
    memset(&header, 0, sizeof(header));

    header.magic         = NGRAM_MOD_CACHE_MAGIC;
    header.version       = NGRAM_MOD_CACHE_VERSION;
    header.header_size   = sizeof(ngram_mod_cache_header);
    header.slot_size     = sizeof(ngram_mod_slot);
    header.capacity      = mod.size();
    header.entry_count   = mod.get_used();
    header.n_match       = n_match;
    header.vocab_size    = vocab_size;

    if (tokenizer_model) {
        strncpy(header.tokenizer_model, tokenizer_model, sizeof(header.tokenizer_model) - 1);
    }
    if (tokenizer_pre) {
        strncpy(header.tokenizer_pre, tokenizer_pre, sizeof(header.tokenizer_pre) - 1);
    }

    header.created_ts    = 0; // caller can set if desired
    header.saved_ts      = 0;

    const auto & s = mod.get_stats();
    header.total_inserts = s.n_inserts;
    header.total_lookups = s.n_lookups;
    header.total_hits    = s.n_hits;
}

static bool atomic_write(const std::string & path, const void * data, size_t len) {
    std::string tmp = path + ".tmp";

#ifdef _WIN32
    FILE * f = nullptr;
    fopen_s(&f, tmp.c_str(), "wb");
#else
    FILE * f = fopen(tmp.c_str(), "wb");
#endif

    if (!f) {
        return false;
    }

    size_t written = fwrite(data, 1, len, f);
    if (written != len) {
        fclose(f);
        return false;
    }

    fflush(f);
#ifdef _WIN32
    _fileno(f);
    _commit(_fileno(f));
#else
    fsync(fileno(f));
#endif
    fclose(f);

    // atomic replace
#ifdef _WIN32
    // Windows: MoveFileEx with MOVEFILE_REPLACE_EXISTING
    // Convert to wide string for MoveFileExW
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath(wlen, 0);
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &wpath[0], wlen);

    int wlen2 = MultiByteToWideChar(CP_UTF8, 0, tmp.c_str(), -1, nullptr, 0);
    std::wstring wtmp(wlen2, 0);
    MultiByteToWideChar(CP_UTF8, 0, tmp.c_str(), -1, &wtmp[0], wlen2);

    if (!MoveFileExW(wtmp.c_str(), wpath.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        // fallback: remove then rename
        DeleteFileW(wpath.c_str());
        if (!MoveFileExW(wtmp.c_str(), wpath.c_str(), MOVEFILE_REPLACE_EXISTING)) {
            return false;
        }
    }
#else
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        return false;
    }
#endif

    return true;
}

bool ngram_mod_cache_save(
        const std::string & path,
        const common_ngram_mod & mod,
        const ngram_mod_cache_header & header) {
    // compute payload checksum
    const ngram_mod_slot * slots = mod.slots();
    const size_t capacity = mod.size();
    const size_t payload_bytes = capacity * sizeof(ngram_mod_slot);

    ngram_mod_cache_footer footer;
    footer.payload_checksum = xxh64(slots, payload_bytes);

    // write: header + slots + footer
    const size_t total = sizeof(header) + payload_bytes + sizeof(footer);

    std::string buf(total, '\0');
    size_t off = 0;

    memcpy(&buf[off], &header, sizeof(header));
    off += sizeof(header);

    memcpy(&buf[off], slots, payload_bytes);
    off += payload_bytes;

    memcpy(&buf[off], &footer, sizeof(footer));
    off += sizeof(footer);

    return atomic_write(path, buf.data(), off);
}

// validation helper
static bool validate_header(
        const ngram_mod_cache_header & h,
        uint32_t expected_n_match,
        uint32_t expected_vocab_size,
        const char * expected_tokenizer_model,
        const char * expected_tokenizer_pre) {
    if (h.magic != NGRAM_MOD_CACHE_MAGIC) {
        return false;
    }
    if (h.version != NGRAM_MOD_CACHE_VERSION) {
        return false;
    }
    if (h.header_size != sizeof(ngram_mod_cache_header)) {
        return false;
    }
    if (h.slot_size != sizeof(ngram_mod_slot)) {
        return false;
    }
    if (h.capacity == 0 || h.capacity > NGRAM_MOD_CACHE_MAX_SLOTS) {
        return false;
    }
    if (h.n_match != expected_n_match) {
        return false;
    }
    if (h.vocab_size != expected_vocab_size) {
        return false;
    }
    if (strncmp(h.tokenizer_model, expected_tokenizer_model, sizeof(h.tokenizer_model)) != 0) {
        return false;
    }
    if (strncmp(h.tokenizer_pre, expected_tokenizer_pre, sizeof(h.tokenizer_pre)) != 0) {
        return false;
    }
    return true;
}

bool ngram_mod_cache_load(
        const std::string & path,
        common_ngram_mod & mod,
        ngram_mod_cache_header & header,
        uint32_t expected_n_match,
        uint32_t expected_vocab_size,
        const char * expected_tokenizer_model,
        const char * expected_tokenizer_pre) {
#ifdef _WIN32
    FILE * f = nullptr;
    fopen_s(&f, path.c_str(), "rb");
#else
    FILE * f = fopen(path.c_str(), "rb");
#endif

    if (!f) {
        return false;
    }

    // read header
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fclose(f);
        return false;
    }

    // validate header
    if (!validate_header(header, expected_n_match, expected_vocab_size,
                         expected_tokenizer_model, expected_tokenizer_pre)) {
        fclose(f);
        return false;
    }

    const size_t capacity = header.capacity;
    const size_t payload_bytes = capacity * sizeof(ngram_mod_slot);

    // allocate and read slots
    // mod must already be the right size; if not, recreate
    if (mod.size() != capacity) {
        mod = common_ngram_mod(header.n_match, capacity);
    } else {
        mod.reset();
    }

    ngram_mod_slot * slots = mod.slots();
    if (fread(slots, sizeof(ngram_mod_slot), capacity, f) != capacity) {
        fclose(f);
        mod.reset();
        return false;
    }

    // read and verify footer
    ngram_mod_cache_footer footer;
    if (fread(&footer, sizeof(footer), 1, f) != 1) {
        fclose(f);
        mod.reset();
        return false;
    }

    fclose(f);

    // verify checksum
    uint64_t actual = xxh64(slots, payload_bytes);
    if (actual != footer.payload_checksum) {
        mod.reset();
        return false;
    }

    return true;
}
