#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void randomize_tensor(struct ggml_tensor * tensor) {
    float * p = (float *) tensor->data;
    const size_t ne = ggml_nelements(tensor);
    for (size_t i = 0; i < ne; ++i) {
        p[i] = ((float) rand() / (float) RAND_MAX) * 2.0f - 1.0f;
    }
}

int main() {
    printf("running test-moe-expert-partition (numerical equivalence)...\n");

    const int64_t n_embd        = 32;
    const int64_t n_ff          = 64;
    const int64_t n_tok         = 4;
    const int64_t n_expert      = 8;
    const int64_t n_expert_used = 2;
    const int64_t n_accel       = 4; // first 4 experts resident in accelerator partition

    size_t mem_size = 32 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    // input activations [n_embd, 1, n_tok]
    ggml_tensor * cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tok);
    randomize_tensor(cur);

    // 3D expert weights [n_embd, n_ff, n_expert]
    ggml_tensor * gate_exps = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
    ggml_tensor * up_exps   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
    ggml_tensor * down_exps = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_embd, n_expert);
    randomize_tensor(gate_exps);
    randomize_tensor(up_exps);
    randomize_tensor(down_exps);

    // router selection [n_expert_used, n_tok]
    ggml_tensor * selected_experts = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, n_tok);
    int32_t * exp_data = (int32_t *) selected_experts->data;
    // ensure test covers GPU-only, CPU-only, and mixed selections
    // tok 0: [1, 2] (both GPU)
    exp_data[0] = 1; exp_data[1] = 2;
    // tok 1: [5, 6] (both CPU)
    exp_data[2] = 5; exp_data[3] = 6;
    // tok 2: [0, 7] (mixed)
    exp_data[4] = 0; exp_data[5] = 7;
    // tok 3: [3, 4] (border: 3 is GPU, 4 is CPU)
    exp_data[6] = 3; exp_data[7] = 4;

    // routing weights [1, n_expert_used, n_tok]
    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_expert_used, n_tok);
    float * w_data = (float *) weights->data;
    for (int i = 0; i < n_expert_used * n_tok; ++i) {
        w_data[i] = 0.5f;
    }

    // 1. Reference unpartitioned MoE computation
    ggml_tensor * up_ref = ggml_mul_mat_id(ctx, up_exps, cur, selected_experts);
    ggml_tensor * gate_ref = ggml_mul_mat_id(ctx, gate_exps, cur, selected_experts);
    ggml_tensor * act_ref = ggml_silu(ctx, gate_ref);
    ggml_tensor * ffn_mid_ref = ggml_mul(ctx, act_ref, up_ref);
    ggml_tensor * down_ref = ggml_mul_mat_id(ctx, down_exps, ffn_mid_ref, selected_experts);
    ggml_tensor * exp_weighted_ref = ggml_mul(ctx, down_ref, weights);

    ggml_tensor * cur_exp_ref[2];
    for (uint32_t i = 0; i < n_expert_used; ++i) {
        cur_exp_ref[i] = ggml_view_2d(ctx, exp_weighted_ref, n_embd, n_tok, exp_weighted_ref->nb[2], i * exp_weighted_ref->nb[1]);
    }
    ggml_tensor * out_ref = ggml_add(ctx, cur_exp_ref[0], cur_exp_ref[1]);

    // 2. Partitioned Heterogeneous MoE computation with arbitrary slot table mapping
    // Slot mapping: slot 0 -> expert 1, slot 1 -> expert 5, slot 2 -> expert 6, slot 3 -> expert 7
    // Non-resident experts in RAM: 0, 2, 3, 4
    std::vector<int32_t> slot_to_expert = { 1, 5, 6, 7 };
    std::vector<float> expert_to_slot(n_expert, -1.0f);
    for (int32_t s = 0; s < n_accel; ++s) {
        expert_to_slot[slot_to_expert[s]] = (float) s;
    }

    // Lookup table tensor [1, n_expert]
    ggml_tensor * slot_table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_expert);
    memcpy(slot_table->data, expert_to_slot.data(), n_expert * sizeof(float));

    // Accelerator packed tensors [n_embd, n_ff, n_accel]
    ggml_tensor * gate_exps_accel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_accel);
    ggml_tensor * up_exps_accel   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_accel);
    ggml_tensor * down_exps_accel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_embd, n_accel);

    // Copy selected resident slices into accelerator slots
    for (int32_t s = 0; s < n_accel; ++s) {
        int32_t exp = slot_to_expert[s];
        memcpy((uint8_t *) gate_exps_accel->data + s * gate_exps_accel->nb[2],
               (uint8_t *) gate_exps->data       + exp * gate_exps->nb[2],
               gate_exps_accel->nb[2]);
        memcpy((uint8_t *) up_exps_accel->data   + s * up_exps_accel->nb[2],
               (uint8_t *) up_exps->data         + exp * up_exps->nb[2],
               up_exps_accel->nb[2]);
        memcpy((uint8_t *) down_exps_accel->data + s * down_exps_accel->nb[2],
               (uint8_t *) down_exps->data       + exp * down_exps->nb[2],
               down_exps_accel->nb[2]);
    }

    // Dynamic slot lookup
    ggml_tensor * sel_flat = ggml_reshape_1d(ctx, selected_experts, n_expert_used * n_tok);
    ggml_tensor * looked_up_slots = ggml_get_rows(ctx, slot_table, sel_flat);
    looked_up_slots = ggml_reshape_3d(ctx, looked_up_slots, 1, n_expert_used, n_tok);

    ggml_tensor * diff_gpu = ggml_add(ctx, looked_up_slots, ggml_new_f32(ctx, 0.5f));
    ggml_tensor * is_gpu_expert = ggml_step(ctx, diff_gpu);
    ggml_tensor * is_cpu_expert = ggml_add(ctx, ggml_scale(ctx, is_gpu_expert, -1.0f), ggml_new_f32(ctx, 1.0f));

    ggml_tensor * gpu_weights = ggml_mul(ctx, weights, is_gpu_expert);
    ggml_tensor * cpu_weights = ggml_mul(ctx, weights, is_cpu_expert);

    ggml_tensor * slots_clamped = ggml_clamp(ctx, ggml_dup(ctx, looked_up_slots), 0.0f, (float) (n_accel - 1));
    ggml_tensor * gpu_experts   = ggml_cast(ctx, slots_clamped, GGML_TYPE_I32);
    gpu_experts = ggml_reshape_2d(ctx, gpu_experts, n_expert_used, n_tok);

    // GPU branch
    ggml_tensor * up_gpu = ggml_mul_mat_id(ctx, up_exps_accel, cur, gpu_experts);
    ggml_tensor * gate_gpu = ggml_mul_mat_id(ctx, gate_exps_accel, cur, gpu_experts);
    ggml_tensor * act_gpu = ggml_silu(ctx, gate_gpu);
    ggml_tensor * ffn_mid_gpu = ggml_mul(ctx, act_gpu, up_gpu);
    ggml_tensor * down_gpu = ggml_mul_mat_id(ctx, down_exps_accel, ffn_mid_gpu, gpu_experts);
    ggml_tensor * exp_weighted_gpu = ggml_mul(ctx, down_gpu, gpu_weights);

    ggml_tensor * cur_exp_gpu[2];
    for (uint32_t i = 0; i < n_expert_used; ++i) {
        cur_exp_gpu[i] = ggml_view_2d(ctx, exp_weighted_gpu, n_embd, n_tok, exp_weighted_gpu->nb[2], i * exp_weighted_gpu->nb[1]);
    }
    ggml_tensor * out_gpu = ggml_add(ctx, cur_exp_gpu[0], cur_exp_gpu[1]);

    // CPU branch
    ggml_tensor * up_cpu = ggml_mul_mat_id(ctx, up_exps, cur, selected_experts);
    ggml_tensor * gate_cpu = ggml_mul_mat_id(ctx, gate_exps, cur, selected_experts);
    ggml_tensor * act_cpu = ggml_silu(ctx, gate_cpu);
    ggml_tensor * ffn_mid_cpu = ggml_mul(ctx, act_cpu, up_cpu);
    ggml_tensor * down_cpu = ggml_mul_mat_id(ctx, down_exps, ffn_mid_cpu, selected_experts);
    ggml_tensor * exp_weighted_cpu = ggml_mul(ctx, down_cpu, cpu_weights);

    ggml_tensor * cur_exp_cpu[2];
    for (uint32_t i = 0; i < n_expert_used; ++i) {
        cur_exp_cpu[i] = ggml_view_2d(ctx, exp_weighted_cpu, n_embd, n_tok, exp_weighted_cpu->nb[2], i * exp_weighted_cpu->nb[1]);
    }
    ggml_tensor * out_cpu = ggml_add(ctx, cur_exp_cpu[0], cur_exp_cpu[1]);

    // Sum branches
    ggml_tensor * out_partitioned = ggml_add(ctx, out_gpu, out_cpu);

    // Compute both graphs
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_ref);
    ggml_build_forward_expand(gf, out_partitioned);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_status status = ggml_backend_graph_compute(backend, gf);
    assert(status == GGML_STATUS_SUCCESS);
    ggml_backend_free(backend);

    // Verify numerical equivalence
    const float * ref_data  = (const float *) out_ref->data;
    const float * part_data = (const float *) out_partitioned->data;
    const size_t n_out = ggml_nelements(out_ref);

    float max_diff = 0.0f;
    for (size_t i = 0; i < n_out; ++i) {
        float diff = std::abs(ref_data[i] - part_data[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    printf("Stage 1 (arbitrary slot mapping): max absolute diff: %e\n", max_diff);
    if (max_diff > 1e-4f) {
        printf("FAIL: Stage 1 max_diff exceeds tolerance!\n");
        return 1;
    }

    // Stage 2: Dynamic Expert Promotion (swap expert 2 into slot 1, replacing expert 5)
    printf("running Stage 2 (dynamic expert promotion and slot swapping)...\n");
    const int32_t promoted_exp = 2;
    const int32_t target_slot  = 1;
    const int32_t evicted_exp  = slot_to_expert[target_slot]; // 5

    // Copy promoted expert slices into target slot
    memcpy((uint8_t *) gate_exps_accel->data + target_slot * gate_exps_accel->nb[2],
           (uint8_t *) gate_exps->data       + promoted_exp * gate_exps->nb[2],
           gate_exps_accel->nb[2]);
    memcpy((uint8_t *) up_exps_accel->data   + target_slot * up_exps_accel->nb[2],
           (uint8_t *) up_exps->data         + promoted_exp * up_exps->nb[2],
           up_exps_accel->nb[2]);
    memcpy((uint8_t *) down_exps_accel->data + target_slot * down_exps_accel->nb[2],
           (uint8_t *) down_exps->data       + promoted_exp * down_exps->nb[2],
           down_exps_accel->nb[2]);

    // Update slot tables
    slot_to_expert[target_slot] = promoted_exp;
    expert_to_slot[evicted_exp] = -1.0f;
    expert_to_slot[promoted_exp] = (float) target_slot;
    memcpy(slot_table->data, expert_to_slot.data(), n_expert * sizeof(float));

    // Recompute graph
    backend = ggml_backend_cpu_init();
    status = ggml_backend_graph_compute(backend, gf);
    assert(status == GGML_STATUS_SUCCESS);
    ggml_backend_free(backend);

    max_diff = 0.0f;
    for (size_t i = 0; i < n_out; ++i) {
        float diff = std::abs(ref_data[i] - part_data[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    printf("Stage 2 (post-promotion): max absolute diff: %e\n", max_diff);
    if (max_diff > 1e-4f) {
        printf("FAIL: Stage 2 max_diff exceeds tolerance!\n");
        return 1;
    }

    ggml_free(ctx);
    printf("test-moe-expert-partition passed all stages successfully.\n");
    return 0;
}
