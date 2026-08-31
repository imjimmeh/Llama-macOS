#include "ngram-mod-cache.h"

#include <algorithm>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// simple xxh64 - matches ggml's xxhash for consistency
static uint64_t xxh64(const void * data, size_t len) {
    const uint64_t seed = 0;
    const uint64_t prime1 = 0x9E3779B185EBCA87ULL;
    const uint64_t prime2 = 0xC2B2AE3D27D4EB4FULL;
    const uint64_t prime3 = 0x165667B19E3779F9ULL;
    const uint64_t prime4 = 0x85EBCA77C2B2AE63ULL;
    const uint64_t prime5 = 0x27D4EB2F165667C5ULL;

    const uint8_t * p = (const uint8_t *) data;
    const uint8_t * const end = p + len;

    uint64_t h64 = seed + prime5 + len;

    // process 32-byte blocks
    while (p + 32 <= end) {
        uint64_t k1, k2, k3, k4;
        memcpy(&k1, p + 0, 8);
        memcpy(&k2, p + 8, 8);
        memcpy(&k3, p + 16, 8);
        memcpy(&k4, p + 24, 8);
        p += 32;

        k1 *= prime2; k1 = (k1 << 31) | (k1 >> 33); k1 *= prime1; h64 ^= k1;
        h64 = (h64 << 27) | (h64 >> 37); h64 += prime4 + prime1;
        k2 *= prime2; k2 = (k2 << 31) | (k2 >> 33); k2 *= prime1; h64 ^= k2;
        h64 = (h64 << 27) | (h64 >> 37); h64 += prime4 + prime1;
        k3 *= prime2; k3 = (k3 << 31) | (k3 >> 33); k3 *= prime1; h64 ^= k3;
        h64 = (h64 << 27) | (h64 >> 37); h64 += prime4 + prime1;
        k4 *= prime2; k4 = (k4 << 31) | (k4 >> 33); k4 *= prime1; h64 ^= k4;
        h64 = (h64 << 27) | (h64 >> 37); h64 += prime4 + prime1;
    }

    // process remaining 8-byte blocks
    while (p + 8 <= end) {
        uint64_t k;
        memcpy(&k, p, 8);
        p += 8;
        k *= prime2; k = (k << 31) | (k >> 33); k *= prime1; h64 ^= k;
        h64 = (h64 << 27) | (h64 >> 37); h64 += prime4 + prime1;
    }

    // process remaining 4-byte blocks
    while (p + 4 <= end) {
        uint32_t k;
        memcpy(&k, p, 4);
        p += 4;
        h64 ^= (uint64_t) k * prime1;
        h64 = (h64 << 23) | (h64 >> 41); h64 *= prime2;
    }

    // process remaining 1-byte blocks
    while (p < end) {
        h64 ^= (uint64_t) *p * prime5;
        h64 = (h64 << 11) | (h64 >> 53); h64 *= prime1;
    }

    // avalanche
    h64 ^= h64 >> 33; h64 *= prime2;
    h64 ^= h64 >> 29; h64 *= prime3;
    h64 ^= h64 >> 32;
    return h64;
}

#ifdef _WIN32
static std::wstring utf8_to_wide(const std::string & s) {
    int wlen = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring wstr(wlen, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &wstr[0], wlen);
    return wstr;
}
#endif

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
    header.flags         = NGRAM_MOD_CACHE_FLAG_FP_INDEX;
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
    std::wstring wpath = utf8_to_wide(path);
    std::wstring wtmp = utf8_to_wide(tmp);

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
    if (h.version != 1 && h.version != 2) {
        return false;
    }
    if (h.header_size != NGRAM_MOD_CACHE_HEADER_V1_SIZE &&
            h.header_size != sizeof(ngram_mod_cache_header)) {
        return false;
    }
    if (h.slot_size != sizeof(ngram_mod_slot)) {
        return false;
    }
    const bool tiered = (h.flags & NGRAM_MOD_CACHE_FLAG_TIERED) != 0;
    if (h.hits_size != 0 && h.hits_size != sizeof(uint32_t)) {
        return false;
    }
    if (!tiered && (h.capacity == 0 || h.capacity > NGRAM_MOD_CACHE_MAX_SLOTS)) {
        return false;
    }
    if (h.capacity_cold > NGRAM_MOD_CACHE_MAX_COLD_SLOTS) {
        return false;
    }
    if (tiered != (h.capacity_cold > 0)) {
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

// read header with v1/v2 handling; v1 headers zero-fill the v2 tail fields
static bool read_cache_header(FILE * f, ngram_mod_cache_header & header) {
    memset(&header, 0, sizeof(header));

    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t header_size = 0;

    if (fread(&magic, sizeof(magic), 1, f) != 1) {
        return false;
    }
    if (fread(&version, sizeof(version), 1, f) != 1) {
        return false;
    }
    if (fread(&header_size, sizeof(header_size), 1, f) != 1) {
        return false;
    }
    if (magic != NGRAM_MOD_CACHE_MAGIC) {
        return false;
    }
    if (version != 1 && version != 2) {
        return false;
    }
    if (header_size != NGRAM_MOD_CACHE_HEADER_V1_SIZE &&
            header_size != sizeof(ngram_mod_cache_header)) {
        return false;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        return false;
    }
    if (fread(&header, 1, header_size, f) != header_size) {
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
    if (!read_cache_header(f, header)) {
        fclose(f);
        return false;
    }

    // validate header
    if (!validate_header(header, expected_n_match, expected_vocab_size,
                         expected_tokenizer_model, expected_tokenizer_pre)) {
        fclose(f);
        return false;
    }

    // tiered files are the cold store, not a hot pool dump
    if (header.flags & NGRAM_MOD_CACHE_FLAG_TIERED) {
        fclose(f);
        return false;
    }

    // older dumps index the pool by full hash; the pool now keys by
    // fingerprint, so those entries would be unreachable
    if (!(header.flags & NGRAM_MOD_CACHE_FLAG_FP_INDEX)) {
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

    // recount occupied slots after read (reset() cleared used counter)
    mod.recount_used();

    return true;
}

// ---- writable mmap wrapper ----

ngram_mod_mmap::~ngram_mod_mmap() {
    close();
}

bool ngram_mod_mmap::open(const std::string & path, size_t size) {
    close();

#ifdef _WIN32
    HANDLE h = CreateFileW(utf8_to_wide(path).c_str(),
                           GENERIC_READ | GENERIC_WRITE,
                           FILE_SHARE_READ | FILE_SHARE_WRITE,
                           nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return false;
    }

    LARGE_INTEGER li;
    li.QuadPart = (LONGLONG) size;
    if (!SetFilePointerEx(h, li, nullptr, FILE_BEGIN) || !SetEndOfFile(h)) {
        CloseHandle(h);
        return false;
    }

    HANDLE hm = CreateFileMappingW(h, nullptr, PAGE_READWRITE, 0, 0, nullptr);
    if (!hm) {
        CloseHandle(h);
        return false;
    }

    void * p = MapViewOfFile(hm, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!p) {
        CloseHandle(hm);
        CloseHandle(h);
        return false;
    }

    h_file = h;
    h_map  = hm;
    data   = p;
    this->size = size;
    return true;
#else
    int fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        return false;
    }
    if (ftruncate(fd, (off_t) size) != 0) {
        ::close(fd);
        return false;
    }
    void * p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        ::close(fd);
        return false;
    }
    this->fd = fd;
    data = p;
    this->size = size;
    return true;
#endif
}

bool ngram_mod_mmap::flush() {
    if (!data) {
        return false;
    }
#ifdef _WIN32
    if (FlushViewOfFile(data, size) == 0) {
        return false;
    }
    return FlushFileBuffers((HANDLE) h_file) != 0;
#else
    return msync(data, size, MS_SYNC) == 0;
#endif
}

void ngram_mod_mmap::close() {
    if (!data) {
        return;
    }
#ifdef _WIN32
    UnmapViewOfFile(data);
    if (h_map) {
        CloseHandle((HANDLE) h_map);
    }
    if (h_file) {
        CloseHandle((HANDLE) h_file);
    }
    data = nullptr;
    h_map = nullptr;
    h_file = nullptr;
    size = 0;
#else
    munmap(data, size);
    if (fd >= 0) {
        ::close(fd);
    }
    data = nullptr;
    fd = -1;
    size = 0;
#endif
}

// ---- format v2 cold store ----

// file layout: [header][cold slots][cold hits][footer]
static size_t cold_payload_bytes(size_t capacity) {
    return capacity * (sizeof(ngram_mod_slot) + sizeof(uint32_t));
}

static size_t cold_file_size(size_t capacity) {
    return sizeof(ngram_mod_cache_header) + cold_payload_bytes(capacity) + sizeof(ngram_mod_cache_footer);
}

// read a v1 file's slots into a heap buffer (used for the in-place upgrade)
static bool read_v1_slots(
        const std::string & path,
        ngram_mod_cache_header & header,
        std::vector<ngram_mod_slot> & out) {
#ifdef _WIN32
    FILE * f = nullptr;
    fopen_s(&f, path.c_str(), "rb");
#else
    FILE * f = fopen(path.c_str(), "rb");
#endif
    if (!f) {
        return false;
    }

    if (!read_cache_header(f, header)) {
        fclose(f);
        return false;
    }
    if (header.version != 1 || header.header_size != NGRAM_MOD_CACHE_HEADER_V1_SIZE) {
        fclose(f);
        return false;
    }

    out.resize(header.capacity);
    if (fread(out.data(), sizeof(ngram_mod_slot), header.capacity, f) != header.capacity) {
        fclose(f);
        out.clear();
        return false;
    }
    fclose(f);
    return true;
}

ngram_mod_cold_store::~ngram_mod_cold_store() {
    close();
}

bool ngram_mod_cold_store::open(
        const std::string & path,
        size_t capacity_cold,
        uint32_t n_match,
        uint32_t vocab_size,
        const char * tokenizer_model,
        const char * tokenizer_pre) {
    close();

    if (capacity_cold == 0 || capacity_cold > NGRAM_MOD_CACHE_MAX_COLD_SLOTS) {
        return false;
    }

    // v1 file present: upgrade in place to v2 (hits = 0)
    {
        ngram_mod_cache_header v1_header;
        std::vector<ngram_mod_slot> v1_slots;
        if (read_v1_slots(path, v1_header, v1_slots)) {
            if (v1_header.n_match != n_match) {
                return false;
            }
            // rebuild the file at the configured cold capacity, copying as many
            // v1 slots as fit
            if (!mmap.open(path, cold_file_size(capacity_cold))) {
                return false;
            }
            slots = (ngram_mod_slot *) ((uint8_t *) mmap.data + sizeof(ngram_mod_cache_header));
            hits  = (uint32_t *) ((uint8_t *) slots + capacity_cold * sizeof(ngram_mod_slot));

            memset(&header, 0, sizeof(header));
            header.magic         = NGRAM_MOD_CACHE_MAGIC;
            header.version       = NGRAM_MOD_CACHE_VERSION;
            header.header_size   = sizeof(ngram_mod_cache_header);
            header.slot_size     = sizeof(ngram_mod_slot);
            header.hits_size     = sizeof(uint32_t);
            header.capacity      = v1_header.capacity;
            header.entry_count   = 0;
            header.n_match       = n_match;
            header.vocab_size    = vocab_size;
            if (tokenizer_model) {
                strncpy(header.tokenizer_model, tokenizer_model, sizeof(header.tokenizer_model) - 1);
            }
            if (tokenizer_pre) {
                strncpy(header.tokenizer_pre, tokenizer_pre, sizeof(header.tokenizer_pre) - 1);
            }
            header.created_ts    = v1_header.created_ts;
            header.saved_ts      = v1_header.saved_ts;
            header.total_inserts = v1_header.total_inserts;
            header.total_lookups = v1_header.total_lookups;
            header.total_hits    = v1_header.total_hits;
            header.capacity_cold = capacity_cold;
            header.cold_entry_count = 0;
            header.flags         = NGRAM_MOD_CACHE_FLAG_TIERED;

            // place v1 entries at their fingerprint index so runtime cold
            // lookups (fp % capacity) find them; later entries win collisions
            memset(slots, 0xFF, capacity_cold * sizeof(ngram_mod_slot));
            memset(hits, 0, capacity_cold * sizeof(uint32_t));
            used = 0;
            this->capacity = capacity_cold; // set_slot/get_slot guard on these
            open_ok = true;
            for (size_t i = 0; i < v1_slots.size(); i++) {
                if (v1_slots[i].is_empty()) {
                    continue;
                }
                const size_t j = v1_slots[i].fingerprint % capacity_cold;
                const ngram_mod_slot * cs = get_slot(j);
                const uint32_t h = (cs && cs->matches(v1_slots[i].fingerprint)) ? get_hits(j) : 0;
                set_slot(j, v1_slots[i], h);
            }
            header.cold_entry_count = used;
            dirty = true; // header + footer must be written back
            open_ok = true;
            return true;
        }
    }

    // create or reopen a v2 file
    if (!mmap.open(path, cold_file_size(capacity_cold))) {
        return false;
    }
    slots = (ngram_mod_slot *) ((uint8_t *) mmap.data + sizeof(ngram_mod_cache_header));
    hits  = (uint32_t *) ((uint8_t *) slots + capacity_cold * sizeof(ngram_mod_slot));

    bool have_header = false;

    // try to parse an existing v2 header at the start of the mapping
    {
        const uint8_t * base = (const uint8_t *) mmap.data;
        memset(&header, 0, sizeof(header));
        uint32_t magic, version, header_size;
        memcpy(&magic, base + 0, 4);
        memcpy(&version, base + 4, 4);
        memcpy(&header_size, base + 8, 4);
        if (magic == NGRAM_MOD_CACHE_MAGIC && (version == 1 || version == 2) &&
                (header_size == NGRAM_MOD_CACHE_HEADER_V1_SIZE || header_size == sizeof(ngram_mod_cache_header))) {
            memcpy(&header, base, header_size);
            have_header = true;
        }
    }

    if (have_header) {
        if (!validate_header(header, n_match, vocab_size, tokenizer_model, tokenizer_pre)) {
            mmap.close();
            return false;
        }
        if (header.capacity_cold != capacity_cold) {
            // store was created with a different size; start fresh at the
            // configured size (mapping already truncated/extended the file)
            memset(slots, 0xFF, capacity_cold * sizeof(ngram_mod_slot));
            memset(hits, 0, capacity_cold * sizeof(uint32_t));
            header.capacity_cold = capacity_cold;
            header.cold_entry_count = 0;
            header.flags = NGRAM_MOD_CACHE_FLAG_TIERED;
            header.version = NGRAM_MOD_CACHE_VERSION;
            header.header_size = sizeof(ngram_mod_cache_header);
            used = 0;
            dirty = true;
        } else {
            used = header.cold_entry_count;
            if (used > capacity_cold) {
                used = capacity_cold;
            }
            dirty = (header.flags & NGRAM_MOD_CACHE_FLAG_CLEAN_CLOSE) == 0;
            if (header.flags & NGRAM_MOD_CACHE_FLAG_CLEAN_CLOSE) {
                // graceful close: verify checksum; mismatch is self-healing
                // (fingerprints catch per-slot corruption), so only warn
                ngram_mod_cache_footer footer;
                memcpy(&footer, (const uint8_t *) mmap.data + cold_file_size(capacity_cold) - sizeof(footer), sizeof(footer));
                const uint64_t actual = xxh64((const uint8_t *) mmap.data + sizeof(header), cold_payload_bytes(capacity_cold));
                if (actual != footer.payload_checksum) {
                    dirty = true; // rewrite a correct footer on next flush
                }
            }
        }
    } else {
        // fresh store
        memset(&header, 0, sizeof(header));
        header.magic         = NGRAM_MOD_CACHE_MAGIC;
        header.version       = NGRAM_MOD_CACHE_VERSION;
        header.header_size   = sizeof(ngram_mod_cache_header);
        header.slot_size     = sizeof(ngram_mod_slot);
        header.hits_size     = sizeof(uint32_t);
        header.capacity      = 0;
        header.entry_count   = 0;
        header.n_match       = n_match;
        header.vocab_size    = vocab_size;
        if (tokenizer_model) {
            strncpy(header.tokenizer_model, tokenizer_model, sizeof(header.tokenizer_model) - 1);
        }
        if (tokenizer_pre) {
            strncpy(header.tokenizer_pre, tokenizer_pre, sizeof(header.tokenizer_pre) - 1);
        }
        header.created_ts    = (uint64_t) time(nullptr);
        header.saved_ts      = 0;
        header.capacity_cold = capacity_cold;
        header.cold_entry_count = 0;
        header.flags         = NGRAM_MOD_CACHE_FLAG_TIERED;
        // fresh mapped pages are zero-filled; mark slots empty (next_token = -1)
        memset(slots, 0xFF, capacity_cold * sizeof(ngram_mod_slot));
        used = 0;
        dirty = true;
    }

    this->capacity = capacity_cold;
    open_ok = true;
    return true;
}

bool ngram_mod_cold_store::flush() {
    if (!open_ok || !mmap.data) {
        return false;
    }

    // footer checksum over cold slots + cold hits
    header.cold_entry_count = used;
    header.saved_ts = (uint64_t) time(nullptr);
    header.flags |= NGRAM_MOD_CACHE_FLAG_CLEAN_CLOSE | NGRAM_MOD_CACHE_FLAG_FP_INDEX;

    ngram_mod_cache_footer footer;
    footer.payload_checksum = xxh64((const uint8_t *) mmap.data + sizeof(header), cold_payload_bytes(capacity));

    memcpy(mmap.data, &header, sizeof(header));
    memcpy((uint8_t *) mmap.data + cold_file_size(capacity) - sizeof(footer), &footer, sizeof(footer));

    if (!mmap.flush()) {
        return false;
    }
    dirty = false;
    return true;
}

void ngram_mod_cold_store::close() {
    if (open_ok) {
        flush();
    }
    mmap.close();
    slots = nullptr;
    hits = nullptr;
    capacity = 0;
    used = 0;
    open_ok = false;
}

const ngram_mod_slot * ngram_mod_cold_store::get_slot(size_t idx) const {
    if (!open_ok || idx >= capacity) {
        return nullptr;
    }
    return slots[idx].is_empty() ? nullptr : &slots[idx];
}

uint32_t ngram_mod_cold_store::get_hits(size_t idx) const {
    if (!open_ok || idx >= capacity) {
        return 0;
    }
    return hits[idx];
}

void ngram_mod_cold_store::set_slot(size_t idx, const ngram_mod_slot & s, uint32_t h) {
    if (!open_ok || idx >= capacity) {
        return;
    }
    const bool was_empty = slots[idx].is_empty();
    slots[idx] = s;
    hits[idx] = h;
    if (was_empty && !s.is_empty()) {
        used++;
    } else if (!was_empty && s.is_empty() && used > 0) {
        used--;
    }
    dirty = true;
}

void ngram_mod_cold_store::bump_hits(size_t idx) {
    if (!open_ok || idx >= capacity) {
        return;
    }
    if (hits[idx] != UINT32_MAX) {
        hits[idx]++;
    }
    dirty = true;
}

void ngram_mod_cold_store::recount_used() {
    used = 0;
    for (size_t i = 0; i < capacity; i++) {
        if (!slots[i].is_empty()) {
            used++;
        }
    }
    dirty = true;
}
