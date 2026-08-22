#include "ggml-routing-predictor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Internal predictor structure
struct ggml_routing_predictor {
    enum ggml_routing_predictor_type type;
    int32_t input_dim;
    int32_t num_experts;
    int32_t horizon;
    int32_t rank;

    // Model weights for variants B and C
    std::vector<float> down_weight;   // [rank * input_dim]
    std::vector<float> down_bias;     // [rank]
    std::vector<float> output_weight; // [num_experts * rank]
    std::vector<float> output_bias;   // [num_experts]

    // For variant C: residual correction weights
    std::vector<float> residual_weight; // [rank * input_dim]
    std::vector<float> residual_bias;   // [rank]
    std::vector<float> residual_output; // [num_experts * rank]
};

// GELU activation approximation
static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Matrix-vector multiplication: y = W * x + b
static void mat_vec(
        const float * W,
        const float * x,
        const float * b,
        float * y,
        int32_t rows,
        int32_t cols) {
    for (int32_t i = 0; i < rows; i++) {
        float sum = b ? b[i] : 0.0f;
        for (int32_t j = 0; j < cols; j++) {
            sum += W[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}

// Matrix-vector multiplication with GELU activation
static void mat_vec_gelu(
        const float * W,
        const float * x,
        const float * b,
        float * y,
        int32_t rows,
        int32_t cols) {
    for (int32_t i = 0; i < rows; i++) {
        float sum = b ? b[i] : 0.0f;
        for (int32_t j = 0; j < cols; j++) {
            sum += W[i * cols + j] * x[j];
        }
        y[i] = gelu(sum);
    }
}

// Load model from binary file
static bool load_model(ggml_routing_predictor_t pred, const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open model file: %s\n", path);
        return false;
    }

    // Read header
    uint32_t magic, version;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 ||
        fread(&version, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return false;
    }

    if (magic != 0x4C525044) { // "LRPD"
        fprintf(stderr, "Invalid model magic: 0x%08X\n", magic);
        fclose(f);
        return false;
    }

    if (version != 2) {
        fprintf(stderr, "Unsupported model version: %u\n", version);
        fclose(f);
        return false;
    }

    // Read dimensions
    int32_t input_dim, rank, num_experts;
    if (fread(&input_dim, sizeof(int32_t), 1, f) != 1 ||
        fread(&rank, sizeof(int32_t), 1, f) != 1 ||
        fread(&num_experts, sizeof(int32_t), 1, f) != 1) {
        fclose(f);
        return false;
    }

    // Verify dimensions match
    if (input_dim != pred->input_dim || rank != pred->rank || num_experts != pred->num_experts) {
        fprintf(stderr, "Model dimensions mismatch: expected (%d, %d, %d), got (%d, %d, %d)\n",
                pred->input_dim, pred->rank, pred->num_experts,
                input_dim, rank, num_experts);
        fclose(f);
        return false;
    }

    // Allocate and read weights
    pred->down_weight.resize(rank * input_dim);
    pred->down_bias.resize(rank);
    pred->output_weight.resize(num_experts * rank);
    pred->output_bias.resize(num_experts);

    if (fread(pred->down_weight.data(), sizeof(float), rank * input_dim, f) != (size_t)(rank * input_dim) ||
        fread(pred->down_bias.data(), sizeof(float), rank, f) != (size_t)rank ||
        fread(pred->output_weight.data(), sizeof(float), num_experts * rank, f) != (size_t)(num_experts * rank) ||
        fread(pred->output_bias.data(), sizeof(float), num_experts, f) != (size_t)num_experts) {
        fprintf(stderr, "Failed to read model weights\n");
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

ggml_routing_predictor_t ggml_routing_predictor_init(
        const struct ggml_routing_predictor_config * config) {
    if (!config) {
        return nullptr;
    }

    auto * pred = new ggml_routing_predictor();
    pred->type = config->type;
    pred->input_dim = config->input_dim;
    pred->num_experts = config->num_experts;
    pred->horizon = config->horizon;
    pred->rank = config->rank;

    // Variant A: no model loading needed
    if (config->type == GGML_ROUTING_PREDICTOR_STALE_FUTURE) {
        fprintf(stderr, "Routing predictor: Variant A (stale future router) initialized\n");
        fprintf(stderr, "  input_dim=%d, num_experts=%d, horizon=%d\n",
                pred->input_dim, pred->num_experts, pred->horizon);
        return pred;
    }

    // Variants B and C: load model
    if (!config->model_path) {
        fprintf(stderr, "Variant %c requires model_path\n",
                config->type == GGML_ROUTING_PREDICTOR_LOW_RANK_MLP ? 'B' : 'C');
        delete pred;
        return nullptr;
    }

    if (!load_model(pred, config->model_path)) {
        fprintf(stderr, "Failed to load model from %s\n", config->model_path);
        delete pred;
        return nullptr;
    }

    fprintf(stderr, "Routing predictor: Variant %c initialized\n",
            config->type == GGML_ROUTING_PREDICTOR_LOW_RANK_MLP ? 'B' : 'C');
    fprintf(stderr, "  input_dim=%d, rank=%d, num_experts=%d, horizon=%d\n",
            pred->input_dim, pred->rank, pred->num_experts, pred->horizon);

    return pred;
}

void ggml_routing_predictor_free(ggml_routing_predictor_t predictor) {
    if (!predictor) {
        return;
    }
    delete predictor;
}

bool ggml_routing_predictor_load_model(
    ggml_routing_predictor_t predictor,
    const char * model_path) {
    
    if (!predictor || !model_path) return false;
    
    FILE * file = fopen(model_path, "rb");
    if (!file) return false;
    
    // Read magic number
    uint32_t magic = 0;
    if (fread(&magic, sizeof(magic), 1, file) != 1) {
        fclose(file);
        return false;
    }
    
    // Magic: 0x4C525044 = "LRPD" (Learned Routing Predictor Data)
    if (magic != 0x4C525044) {
        fclose(file);
        return false;
    }
    
    // Read version
    uint32_t version = 0;
    if (fread(&version, sizeof(version), 1, file) != 1) {
        fclose(file);
        return false;
    }
    
    // Accept trainer versions (1 and 2). Magic and dimension order are the
    // authoritative contract; version differences only reflect tool updates.
    if (version != 1 && version != 2) {
        fclose(file);
        return false;
    }
    
    // Read dimensions. Header order matches trainer output:
    // LRPD magic, version(2), int32 input_dim, int32 rank, int32 num_experts.
    int32_t input_dim = 0, num_experts = 0, rank = 0;
    if (fread(&input_dim, sizeof(input_dim), 1, file) != 1 ||
        fread(&rank, sizeof(rank), 1, file) != 1 ||
        fread(&num_experts, sizeof(num_experts), 1, file) != 1) {
        fclose(file);
        return false;
    }
    
    // Validate dimensions match predictor
    if (input_dim != predictor->input_dim || num_experts != predictor->num_experts || rank != predictor->rank) {
        fclose(file);
        return false;
    }
    
    // Allocate weight buffers
    predictor->down_weight.resize(rank * input_dim);
    predictor->down_bias.resize(rank);
    predictor->output_weight.resize(num_experts * rank);
    predictor->output_bias.resize(num_experts);
    
    // Read weights
    if (fread(predictor->down_weight.data(), sizeof(float), rank * input_dim, file) != (size_t)(rank * input_dim) ||
        fread(predictor->down_bias.data(), sizeof(float), rank, file) != (size_t)rank ||
        fread(predictor->output_weight.data(), sizeof(float), num_experts * rank, file) != (size_t)(num_experts * rank) ||
        fread(predictor->output_bias.data(), sizeof(float), num_experts, file) != (size_t)num_experts) {
        fclose(file);
        return false;
    }
    
    // For Variant C, read residual correction weights
    if (predictor->type == GGML_ROUTING_PREDICTOR_FUTURE_RESIDUAL) {
        predictor->residual_weight.resize(rank * input_dim);
        predictor->residual_bias.resize(rank);
        predictor->residual_output.resize(num_experts * rank);
        
        if (fread(predictor->residual_weight.data(), sizeof(float), rank * input_dim, file) != (size_t)(rank * input_dim) ||
            fread(predictor->residual_bias.data(), sizeof(float), rank, file) != (size_t)rank ||
            fread(predictor->residual_output.data(), sizeof(float), num_experts * rank, file) != (size_t)(num_experts * rank)) {
            fclose(file);
            return false;
        }
    }
    
    fclose(file);
    return true;
}

void ggml_routing_predictor_extract_features(
        ggml_routing_predictor_t predictor,
        const float * router_logits,
        int32_t num_experts,
        float * out_features) {
    if (!predictor || !router_logits || !out_features) {
        return;
    }

    // Features are the router logits (truncated or padded to input_dim)
    int32_t copy_dim = std::min(num_experts, predictor->input_dim);
    memcpy(out_features, router_logits, copy_dim * sizeof(float));

    // Pad with zeros if needed
    if (copy_dim < predictor->input_dim) {
        memset(out_features + copy_dim, 0, (predictor->input_dim - copy_dim) * sizeof(float));
    }
}

// Variant B/C prediction with learned weights
static int32_t predict_learned(
    ggml_routing_predictor_t predictor,
    const float * features,
    int32_t * out_expert_ids,
    float * out_confidences,
    int32_t max_experts) {
    
    if (!predictor || !features || !out_expert_ids) return 0;
    
    int32_t rank = predictor->rank;
    int32_t num_experts = predictor->num_experts;
    
    // Check if weights are loaded
    if (predictor->down_weight.empty() || predictor->output_weight.empty()) {
        return 0;
    }
    
    // Hidden layer: h = GELU(W_down * x + b_down)
    std::vector<float> hidden(rank);
    mat_vec_gelu(
        predictor->down_weight.data(),
        features,
        predictor->down_bias.data(),
        hidden.data(),
        rank,
        predictor->input_dim
    );
    
    // Output layer: logits = W_out * h + b_out
    std::vector<float> logits(num_experts);
    mat_vec(
        predictor->output_weight.data(),
        hidden.data(),
        predictor->output_bias.data(),
        logits.data(),
        num_experts,
        rank
    );
    
    // For Variant C: add residual correction
    if (predictor->type == GGML_ROUTING_PREDICTOR_FUTURE_RESIDUAL && !predictor->residual_weight.empty()) {
        std::vector<float> residual_hidden(rank);
        mat_vec_gelu(
            predictor->residual_weight.data(),
            features,
            predictor->residual_bias.data(),
            residual_hidden.data(),
            rank,
            predictor->input_dim
        );
        
        std::vector<float> residual_logits(num_experts);
        mat_vec(
            predictor->residual_output.data(),
            residual_hidden.data(),
            nullptr,
            residual_logits.data(),
            num_experts,
            rank
        );
        
        // Add residual to main logits
        for (int32_t i = 0; i < num_experts; i++) {
            logits[i] += residual_logits[i];
        }
    }
    
    // Apply softmax to get probabilities
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    for (int32_t i = 0; i < num_experts; i++) {
        logits[i] = expf(logits[i] - max_logit);
        sum_exp += logits[i];
    }
    for (int32_t i = 0; i < num_experts; i++) {
        logits[i] /= sum_exp;
    }
    
    // Get top-K experts
    int32_t n_return = std::min(max_experts, num_experts);
    std::vector<std::pair<float, int32_t>> expert_probs;
    for (int32_t i = 0; i < num_experts; i++) {
        expert_probs.push_back({logits[i], i});
    }
    std::sort(expert_probs.begin(), expert_probs.end(), std::greater<std::pair<float, int32_t>>());
    
    for (int32_t i = 0; i < n_return; i++) {
        out_expert_ids[i] = expert_probs[i].second;
        if (out_confidences) {
            out_confidences[i] = expert_probs[i].first;
        }
    }
    
    return n_return;
}

int32_t ggml_routing_predictor_predict(
        ggml_routing_predictor_t predictor,
        const float * features,
        int32_t * out_expert_ids,
        float * out_confidences,
        int32_t max_predictions) {
    if (!predictor || !features || !out_expert_ids || max_predictions <= 0) {
        return 0;
    }

    std::vector<float> logits(predictor->num_experts);

    if (predictor->type == GGML_ROUTING_PREDICTOR_STALE_FUTURE) {
        // Variant A: features are already the logits (or router output)
        // Just copy and apply softmax-like scaling
        for (int32_t i = 0; i < predictor->num_experts; i++) {
            logits[i] = (i < predictor->input_dim) ? features[i] : 0.0f;
        }
    } else if (predictor->type == GGML_ROUTING_PREDICTOR_LOW_RANK_MLP) {
        // Variant B: low-rank MLP
        // features -> down_proj (with GELU) -> output_proj -> logits
        std::vector<float> hidden(predictor->rank);
        mat_vec_gelu(
            predictor->down_weight.data(),
            features,
            predictor->down_bias.data(),
            hidden.data(),
            predictor->rank,
            predictor->input_dim);

        mat_vec(
            predictor->output_weight.data(),
            hidden.data(),
            predictor->output_bias.data(),
            logits.data(),
            predictor->num_experts,
            predictor->rank);
    } else if (predictor->type == GGML_ROUTING_PREDICTOR_FUTURE_RESIDUAL) {
        // Variant C: future router + learned residual
        // For now, treat features as stale logits and add residual correction
        // In a full implementation, stale_logits would come from W_router[L+H] * x_L
        std::vector<float> stale_logits(predictor->num_experts);
        for (int32_t i = 0; i < predictor->num_experts; i++) {
            stale_logits[i] = (i < predictor->input_dim) ? features[i] : 0.0f;
        }

        // Compute residual correction
        std::vector<float> hidden(predictor->rank);
        mat_vec_gelu(
            predictor->residual_weight.data(),
            features,
            predictor->residual_bias.data(),
            hidden.data(),
            predictor->rank,
            predictor->input_dim);

        std::vector<float> residual(predictor->num_experts);
        mat_vec(
            predictor->residual_output.data(),
            hidden.data(),
            nullptr,
            residual.data(),
            predictor->num_experts,
            predictor->rank);

        // Add residual to stale logits
        for (int32_t i = 0; i < predictor->num_experts; i++) {
            logits[i] = stale_logits[i] + residual[i];
        }
    } else {
        return 0;
    }

    // Convert logits to probabilities (softmax)
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    std::vector<float> probs(predictor->num_experts);
    for (int32_t i = 0; i < predictor->num_experts; i++) {
        probs[i] = expf(logits[i] - max_logit);
        sum_exp += probs[i];
    }
    for (int32_t i = 0; i < predictor->num_experts; i++) {
        probs[i] /= sum_exp;
    }

    // Top-k selection
    std::vector<std::pair<float, int32_t>> scored;
    scored.reserve(predictor->num_experts);
    for (int32_t i = 0; i < predictor->num_experts; i++) {
        scored.push_back({probs[i], i});
    }

    int32_t n_output = std::min(max_predictions, predictor->num_experts);
    std::partial_sort(
        scored.begin(),
        scored.begin() + n_output,
        scored.end(),
        [](const auto & a, const auto & b) { return a.first > b.first; });

    for (int32_t i = 0; i < n_output; i++) {
        out_expert_ids[i] = scored[i].second;
        if (out_confidences) {
            out_confidences[i] = scored[i].first;
        }
    }

    return n_output;
}
