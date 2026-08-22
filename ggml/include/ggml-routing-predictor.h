#ifndef GGML_ROUTING_PREDICTOR_H
#define GGML_ROUTING_PREDICTOR_H

#include "ggml.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Predictor variant types
enum ggml_routing_predictor_type {
    GGML_ROUTING_PREDICTOR_STALE_FUTURE = 0,   // Variant A: no training
    GGML_ROUTING_PREDICTOR_LOW_RANK_MLP = 1,   // Variant B: learned MLP
    GGML_ROUTING_PREDICTOR_FUTURE_RESIDUAL = 2 // Variant C: future router + residual
};

// Predictor configuration
struct ggml_routing_predictor_config {
    enum ggml_routing_predictor_type type;
    int32_t input_dim;        // Feature dimension (128 for router logits, 256 for projected hidden)
    int32_t num_experts;      // Total number of experts
    int32_t horizon;          // Prediction horizon H (e.g., 8 for L+8)
    int32_t rank;             // Low-rank dimension (for variants B and C)
    const char * model_path;  // Path to trained model (NULL for variant A)
};

// Opaque predictor context
typedef struct ggml_routing_predictor * ggml_routing_predictor_t;

// Lifecycle
GGML_API ggml_routing_predictor_t ggml_routing_predictor_init(
    const struct ggml_routing_predictor_config * config);

GGML_API void ggml_routing_predictor_free(ggml_routing_predictor_t predictor);

// Feature extraction
GGML_API void ggml_routing_predictor_extract_features(
    ggml_routing_predictor_t predictor,
    const float * router_logits,  // [num_experts] router output at layer L
    int32_t num_experts,
    float * out_features);        // [input_dim] output features

// Prediction
// Load trained model for Variant B/C
GGML_API bool ggml_routing_predictor_load_model(
    ggml_routing_predictor_t predictor,
    const char * model_path);

GGML_API int32_t ggml_routing_predictor_predict(
    ggml_routing_predictor_t predictor,
    const float * features,       // [input_dim] input features
    int32_t * out_expert_ids,     // [max_predictions] output expert IDs
    float * out_confidences,      // [max_predictions] output confidences (optional, can be NULL)
    int32_t max_predictions);

#ifdef __cplusplus
}
#endif

#endif // GGML_ROUTING_PREDICTOR_H
