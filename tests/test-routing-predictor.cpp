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

int main() {
    printf("=== Routing Predictor Tests ===\n\n");

    test_predictor_init();
    test_extract_features();
    test_predict();
    test_null_handling();
    test_different_dims();

    printf("\n=== All tests passed! ===\n");
    return 0;
}
