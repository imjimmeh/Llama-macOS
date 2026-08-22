#include "ggml-routing-predictor.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

// Test 1: Initialize predictor (Variant A - stale future router)
static void test_predictor_init() {
    printf("Test 1: Initialize predictor (Variant A)\n");

    ggml_routing_predictor_config config = {};
    config.type = GGML_ROUTING_PREDICTOR_STALE_FUTURE;
    config.input_dim = 128;
    config.num_experts = 64;
    config.horizon = 8;

    ggml_routing_predictor_t predictor = ggml_routing_predictor_init(&config);
    assert(predictor != nullptr);
    printf("  OK: Predictor initialized successfully\n");

    ggml_routing_predictor_free(predictor);
    printf("  OK: Predictor freed successfully\n");
}

// Test 2: Extract features from router logits
static void test_extract_features() {
    printf("\nTest 2: Extract features from router logits\n");

    ggml_routing_predictor_config config = {};
    config.type = GGML_ROUTING_PREDICTOR_STALE_FUTURE;
    config.input_dim = 128;
    config.num_experts = 64;
    config.horizon = 8;

    ggml_routing_predictor_t predictor = ggml_routing_predictor_init(&config);

    // Create dummy router logits (64 experts)
    float router_logits[64];
    for (int i = 0; i < 64; i++) {
        router_logits[i] = (float)i * 0.1f;
    }

    // Extract features (should copy and pad to 128 dims)
    float features[128];
    ggml_routing_predictor_extract_features(predictor, router_logits, 64, features);

    // Verify first 64 dims match router logits
    for (int i = 0; i < 64; i++) {
        assert(features[i] == router_logits[i]);
    }

    // Verify remaining 64 dims are zero-padded
    for (int i = 64; i < 128; i++) {
        assert(features[i] == 0.0f);
    }

    printf("  OK: Features extracted and padded correctly\n");

    ggml_routing_predictor_free(predictor);
}

// Test 3: Run prediction (Variant A)
static void test_predict() {
    printf("\nTest 3: Run prediction (Variant A)\n");

    ggml_routing_predictor_config config = {};
    config.type = GGML_ROUTING_PREDICTOR_STALE_FUTURE;
    config.input_dim = 128;
    config.num_experts = 64;
    config.horizon = 8;

    ggml_routing_predictor_t predictor = ggml_routing_predictor_init(&config);

    // Create features with some high values for specific experts
    float features[128];
    memset(features, 0, sizeof(features));
    features[5] = 10.0f;   // Expert 5 should be top
    features[12] = 8.0f;   // Expert 12 should be second
    features[23] = 6.0f;   // Expert 23 should be third

    // Run prediction
    int32_t predicted_experts[16];
    float confidences[16];
    int32_t n_predicted = ggml_routing_predictor_predict(
        predictor,
        features,
        predicted_experts,
        confidences,
        16
    );

    printf("  Predicted %d experts:\n", n_predicted);
    for (int i = 0; i < n_predicted && i < 5; i++) {
        printf("    Expert %d: confidence %.3f\n", predicted_experts[i], confidences[i]);
    }

    // Verify we got predictions
    assert(n_predicted > 0);
    assert(n_predicted <= 16);

    // Verify top experts are in correct order (5, 12, 23 should be first three)
    assert(predicted_experts[0] == 5);
    assert(predicted_experts[1] == 12);
    assert(predicted_experts[2] == 23);

    printf("  OK: Prediction returned correct top experts\n");

    ggml_routing_predictor_free(predictor);
}

// Test 4: Null pointer handling
static void test_null_handling() {
    printf("\nTest 4: Null pointer handling\n");

    // Test init with null config
    ggml_routing_predictor_t predictor = ggml_routing_predictor_init(nullptr);
    assert(predictor == nullptr);
    printf("  OK: init(nullptr) returns nullptr\n");

    // Test free with null
    ggml_routing_predictor_free(nullptr);  // Should not crash
    printf("  OK: free(nullptr) does not crash\n");

    // Test extract_features with null
    ggml_routing_predictor_config config = {};
    config.type = GGML_ROUTING_PREDICTOR_STALE_FUTURE;
    config.input_dim = 128;
    config.num_experts = 64;
    config.horizon = 8;

    predictor = ggml_routing_predictor_init(&config);
    
    ggml_routing_predictor_extract_features(nullptr, nullptr, 0, nullptr);  // Should not crash
    printf("  OK: extract_features with null args does not crash\n");

    // Test predict with null
    int32_t n = ggml_routing_predictor_predict(nullptr, nullptr, nullptr, nullptr, 0);
    assert(n == 0);
    printf("  OK: predict(nullptr) returns 0\n");

    ggml_routing_predictor_free(predictor);
}

// Test 5: Different input dimensions
static void test_different_dims() {
    printf("\nTest 5: Different input dimensions\n");

    // Test with 256-dim input
    ggml_routing_predictor_config config = {};
    config.type = GGML_ROUTING_PREDICTOR_STALE_FUTURE;
    config.input_dim = 256;
    config.num_experts = 128;
    config.horizon = 8;

    ggml_routing_predictor_t predictor = ggml_routing_predictor_init(&config);
    assert(predictor != nullptr);

    // Create 128 router logits, should pad to 256
    float router_logits[128];
    for (int i = 0; i < 128; i++) {
        router_logits[i] = (float)i * 0.05f;
    }

    float features[256];
    ggml_routing_predictor_extract_features(predictor, router_logits, 128, features);

    // Verify first 128 dims match
    for (int i = 0; i < 128; i++) {
        assert(features[i] == router_logits[i]);
    }

    // Verify remaining 128 dims are zero-padded
    for (int i = 128; i < 256; i++) {
        assert(features[i] == 0.0f);
    }

    printf("  OK: 256-dim features extracted correctly\n");

    ggml_routing_predictor_free(predictor);
}

// Test 6: Load LRPD model and predict (variant B round-trip)
static void test_load_lprd_and_predict() {
    printf("\nTest 6: Load LRPD model and predict (Variant B)\n");

    // Write a minimal LRPD v2 file with known weights.
    // Layout: magic, version(2), input_dim, rank, num_experts,
    //         down_weight[rank*input_dim], down_bias[rank],
    //         output_weight[num_experts*rank], output_bias[num_experts]
    const int32_t input_dim = 4;
    const int32_t rank = 2;
    const int32_t num_experts = 4;
    const char * path = "test_lprd_model.bin";

    FILE * f = fopen(path, "wb");
    assert(f);
    uint32_t magic = 0x4C525044; // "LRPD"
    uint32_t version = 2;
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&input_dim, sizeof(int32_t), 1, f);
    fwrite(&rank, sizeof(int32_t), 1, f);
    fwrite(&num_experts, sizeof(int32_t), 1, f);

    // down_weight [rank=2, input_dim=4]: row0 picks x[0], row1 picks x[1]
    float down_weight[8] = { 0 };
    down_weight[0*4 + 0] = 1.0f; // row0 <- x0
    down_weight[1*4 + 1] = 1.0f; // row1 <- x1
    float down_bias[2] = { 0, 0 };
    // output_weight [num_experts=4, rank=2]: expert 2 reads hidden[0]
    float output_weight[8] = { 0 };
    output_weight[2*2 + 0] = 1.0f; // expert2 <- hidden0
    float output_bias[4] = { 0, 0, 0, 0 };

    fwrite(down_weight, sizeof(float), 8, f);
    fwrite(down_bias, sizeof(float), 2, f);
    fwrite(output_weight, sizeof(float), 8, f);
    fwrite(output_bias, sizeof(float), 4, f);
    fclose(f);

    ggml_routing_predictor_config config = {};
    config.type = GGML_ROUTING_PREDICTOR_LOW_RANK_MLP;
    config.input_dim = input_dim;
    config.num_experts = num_experts;
    config.horizon = 8;
    config.rank = rank;
    config.model_path = path;

    ggml_routing_predictor_t predictor = ggml_routing_predictor_init(&config);
    assert(predictor != nullptr);

    // features with x0=2.0 -> hidden0=gelu(2.0)>0 -> logits[2] largest
    float features[4] = { 2.0f, 0.0f, 0.0f, 0.0f };
    int32_t out_ids[4] = {};
    float conf[4] = {};
    int32_t n = ggml_routing_predictor_predict(predictor, features, out_ids, conf, 4);
    assert(n > 0);
    assert(out_ids[0] == 2);

    printf("  OK: top predicted expert = %d (expected 2)\n", out_ids[0]);

    ggml_routing_predictor_free(predictor);
    remove(path);
}

int main() {
    printf("=== Routing Predictor Tests ===\n\n");

    test_predictor_init();
    test_extract_features();
    test_predict();
    test_null_handling();
    test_different_dims();
    test_load_lprd_and_predict();

    printf("\n=== All tests passed! ===\n");
    return 0;
}
