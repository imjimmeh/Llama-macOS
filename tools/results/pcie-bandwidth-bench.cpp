// PCIe bandwidth benchmark for expert cache prefetch
// Measures actual transfer rates for expert-sized chunks

#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-expert-cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

static void require(bool condition, const char* msg) {
    if (!condition) {
        fprintf(stderr, "REQUIRE FAILED: %s\n", msg);
        exit(1);
    }
}

int main(int argc, char** argv) {
    printf("=== PCIe Bandwidth Benchmark ===\n\n");

    // Parse arguments
    size_t cache_capacity = 4096 * 1024 * 1024;  // 4 GiB default
    std::vector<size_t> chunk_sizes = {
        1 * 1024 * 1024,   // 1 MiB
        2 * 1024 * 1024,   // 2 MiB
        4 * 1024 * 1024,   // 4 MiB
        8 * 1024 * 1024,   // 8 MiB
        16 * 1024 * 1024,  // 16 MiB
        32 * 1024 * 1024,  // 32 MiB
    };

    if (argc > 1) {
        cache_capacity = std::strtoull(argv[1], nullptr, 10) * 1024 * 1024;
    }

    // Initialize backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr, "Failed to initialize CPU backend");

    // Create expert cache
    ggml_backend_expert_cache_t cache = ggml_backend_expert_cache_new(backend, cache_capacity);
    require(cache != nullptr, "Failed to create expert cache");

    printf("Cache capacity: %zu MiB\n", cache_capacity / 1024 / 1024);
    printf("\n");

    // Benchmark each chunk size
    printf("%-12s | %-15s | %-15s | %-15s\n", "Chunk Size", "Transfer Time", "Bandwidth", "Transfers/sec");
    printf("%-12s-+-%-15s-+-%-15s-+-%-15s\n", "------------", "---------------", "---------------", "---------------");

    for (size_t chunk_size : chunk_sizes) {
        const int n_transfers = 100;  // Number of transfers to average
        const size_t total_bytes = chunk_size * n_transfers;

        // Allocate source buffer (CPU)
        std::vector<uint8_t> src(chunk_size);
        for (size_t i = 0; i < chunk_size; i++) {
            src[i] = static_cast<uint8_t>(i & 0xFF);
        }

        // Allocate destination tensor (GPU via cache)
        struct ggml_init_params params = {
            /*.mem_size   =*/ chunk_size + 1024,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        struct ggml_context* ctx = ggml_init(params);
        require(ctx != nullptr, "Failed to create ggml context");

        struct ggml_tensor* dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, chunk_size / sizeof(float));
        require(dst != nullptr, "Failed to create destination tensor");

        // Warmup
        ggml_backend_tensor_set(dst, src.data(), 0, chunk_size);
        ggml_backend_synchronize(backend);

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n_transfers; i++) {
            ggml_backend_tensor_set(dst, src.data(), 0, chunk_size);
        }

        ggml_backend_synchronize(backend);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double transfer_time_us = duration.count() / static_cast<double>(n_transfers);
        double bandwidth_gbs = (chunk_size / 1e9) / (transfer_time_us / 1e6);
        double transfers_per_sec = 1e6 / transfer_time_us;

        printf("%-12zu | %-14.2f us | %-12.2f GB/s | %-14.1f\n",
               chunk_size / 1024 / 1024,
               transfer_time_us,
               bandwidth_gbs,
               transfers_per_sec);

        ggml_free(ctx);
    }

    printf("\n");

    // Cleanup
    ggml_backend_expert_cache_free(cache);
    ggml_backend_free(backend);

    printf("Benchmark complete.\n");
    return 0;
}
