// MoE model geometry and fit placement reporter
// Usage:
//   test-moe-geometry-report --geometry-json -m model.gguf
//   test-moe-geometry-report --placement-json -m model.gguf [-fitt 128] [-t 14]

#include "llama.h"
#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

static void collect_layer_info(struct llama_model * model, int l, int n_experts,
    std::string & gate_type, std::string & up_type, std::string & down_type,
    int64_t & gate_bytes, int64_t & up_bytes, int64_t & down_bytes,
    int64_t & gate_scale_bytes, int64_t & up_scale_bytes, int64_t & down_scale_bytes,
    int64_t & complete_bundle, int64_t & complete_bank) {
    char gname[128], uname[128], dname[128];
    snprintf(gname, sizeof(gname), "blk.%d.ffn_gate_exps.weight", l);
    snprintf(uname, sizeof(uname), "blk.%d.ffn_up_exps.weight", l);
    snprintf(dname, sizeof(dname), "blk.%d.ffn_down_exps.weight", l);

    const struct ggml_tensor * gate = llama_model_get_tensor(model, gname);
    const struct ggml_tensor * up   = llama_model_get_tensor(model, uname);
    const struct ggml_tensor * down = llama_model_get_tensor(model, dname);

    gate_type = gate ? ggml_type_name(gate->type) : "null";
    up_type   = up   ? ggml_type_name(up->type)   : "null";
    down_type = down ? ggml_type_name(down->type) : "null";
    gate_bytes = ggml_nbytes(gate);
    up_bytes   = up ? ggml_nbytes(up) : 0;
    down_bytes = ggml_nbytes(down);

    // APEX per-expert scale tensors (optional)
    char sname[128];
    gate_scale_bytes = 0; up_scale_bytes = 0; down_scale_bytes = 0;

    snprintf(sname, sizeof(sname), "blk.%d.ffn_gate_exps.weight.scale", l);
    const struct ggml_tensor * gs = llama_model_get_tensor(model, sname);
    if (gs) gate_scale_bytes = ggml_nbytes(gs);

    snprintf(sname, sizeof(sname), "blk.%d.ffn_up_exps.weight.scale", l);
    const struct ggml_tensor * us = llama_model_get_tensor(model, sname);
    if (us) up_scale_bytes = ggml_nbytes(us);

    snprintf(sname, sizeof(sname), "blk.%d.ffn_down_exps.weight.scale", l);
    const struct ggml_tensor * ds = llama_model_get_tensor(model, sname);
    if (ds) down_scale_bytes = ggml_nbytes(ds);

    complete_bundle = (gate_bytes + up_bytes + down_bytes
        + gate_scale_bytes + up_scale_bytes + down_scale_bytes) / n_experts;
    complete_bank = gate_bytes + up_bytes + down_bytes
        + gate_scale_bytes + up_scale_bytes + down_scale_bytes;
}

static void print_geometry_json(struct llama_model * model, int n_layers, int n_experts, const char * model_path, const char * sha256) {
    // First pass: detect variation
    std::vector<std::string> gate_types, up_types, down_types;
    std::vector<int64_t> gate_bs, up_bs, down_bs, gate_sb, up_sb, down_sb, cb, cbank;
    bool all_equal = true;

    for (int l = 0; l < n_layers; l++) {
        std::string gt, ut, dt;
        int64_t gb, ub, db, gsb, usb, dsb, bdl, bank;
        collect_layer_info(model, l, n_experts, gt, ut, dt, gb, ub, db, gsb, usb, dsb, bdl, bank);
        gate_types.push_back(gt); up_types.push_back(ut); down_types.push_back(dt);
        gate_bs.push_back(gb); up_bs.push_back(ub); down_bs.push_back(db);
        gate_sb.push_back(gsb); up_sb.push_back(usb); down_sb.push_back(dsb);
        cb.push_back(bdl); cbank.push_back(bank);

        if (l > 0) {
            if (gt != gate_types[0] || ut != up_types[0] || dt != down_types[0] ||
                gb != gate_bs[0] || ub != up_bs[0] || db != down_bs[0] ||
                gsb != gate_sb[0] || usb != up_sb[0] || dsb != down_sb[0]) {
                all_equal = false;
            }
        }
    }

    // Build varying_fields
    std::map<std::string, std::map<int, std::string>> vf;
    if (!all_equal) {
        for (int l = 0; l < n_layers; l++) {
            if (gate_types[l] != gate_types[0]) vf["gate_type"][l] = gate_types[l];
            if (up_types[l] != up_types[0])     vf["up_type"][l] = up_types[l];
            if (down_types[l] != down_types[0]) vf["down_type"][l] = down_types[l];
            if (gate_bs[l] != gate_bs[0])       vf["gate_bytes"][l] = std::to_string(gate_bs[l]);
            if (up_bs[l] != up_bs[0])           vf["up_bytes"][l] = std::to_string(up_bs[l]);
            if (down_bs[l] != down_bs[0])       vf["down_bytes"][l] = std::to_string(down_bs[l]);
            if (cb[l] != cb[0])                 vf["complete_bundle_bytes"][l] = std::to_string(cb[l]);
        }
    }

    printf("{\n");
    printf("  \"format\": 1,\n");
    printf("  \"model_name\": \"%s\",\n", model_path);
    printf("  \"model_sha256\": \"%s\",\n", sha256);
    printf("  \"expert_count\": %d,\n", n_experts);
    printf("  \"top_k\": 8,\n");
    printf("  \"n_layers\": %d,\n", n_layers);
    printf("  \"all_repeated_moe_blocks_equal\": %s,\n", all_equal ? "true" : "false");

    if (!all_equal) {
        printf("  \"varying_fields\": {\n");
        bool vf_first = true;
        for (auto & kv : vf) {
            if (!vf_first) printf(",\n");
            vf_first = false;
            printf("    \"%s\": {\n", kv.first.c_str());
            bool lf = true;
            for (auto & lv : kv.second) {
                if (!lf) printf(",\n");
                lf = false;
                printf("      \"%d\": \"%s\"", lv.first, lv.second.c_str());
            }
            printf("\n    }");
        }
        printf("\n  },\n");
    } else {
        printf("  \"varying_fields\": {},\n");
    }

    printf("  \"layers\": [\n");
    for (int l = 0; l < n_layers; l++) {
        if (l > 0) printf(",");
        printf("\n    {\n");
        printf("      \"layer\": %d,\n", l);
        printf("      \"expert_count\": %d,\n", n_experts);
        printf("      \"top_k\": 8,\n");
        const struct ggml_tensor * gate = llama_model_get_tensor(model,
            (std::string("blk.") + std::to_string(l) + ".ffn_gate_exps.weight").c_str());
        const struct ggml_tensor * up = llama_model_get_tensor(model,
            (std::string("blk.") + std::to_string(l) + ".ffn_up_exps.weight").c_str());
        const struct ggml_tensor * down = llama_model_get_tensor(model,
            (std::string("blk.") + std::to_string(l) + ".ffn_down_exps.weight").c_str());
        const bool is_fused = (gate && gate->ne[2] == n_experts && up == nullptr);
        printf("      \"layout\": \"%s\",\n", is_fused ? "fused" : "separate");
        printf("      \"gate_type\": \"%s\",\n", gate_types[l].c_str());
        printf("      \"gate_shape\": [%" PRId64 ", %" PRId64 ", %" PRId64 "],\n",
               gate->ne[0], gate->ne[1], gate->ne[2]);
        printf("      \"gate_bytes\": %" PRId64 ",\n", gate_bs[l]);
        printf("      \"gate_scale_bytes\": %" PRId64 ",\n", gate_sb[l]);
        printf("      \"up_type\": \"%s\",\n", up_types[l].c_str());
        printf("      \"up_shape\": [%" PRId64 ", %" PRId64 ", %" PRId64 "],\n",
               up ? up->ne[0] : 0, up ? up->ne[1] : 0, up ? up->ne[2] : 0);
        printf("      \"up_bytes\": %" PRId64 ",\n", up_bs[l]);
        printf("      \"up_scale_bytes\": %" PRId64 ",\n", up_sb[l]);
        printf("      \"down_type\": \"%s\",\n", down_types[l].c_str());
        printf("      \"down_shape\": [%" PRId64 ", %" PRId64 ", %" PRId64 "],\n",
               down->ne[0], down->ne[1], down->ne[2]);
        printf("      \"down_bytes\": %" PRId64 ",\n", down_bs[l]);
        printf("      \"down_scale_bytes\": %" PRId64 ",\n", down_sb[l]);
        printf("      \"complete_bundle_bytes\": %" PRId64 ",\n", cb[l]);
        printf("      \"complete_bank_bytes\": %" PRId64 ",\n", cbank[l]);
        printf("      \"scales\": {}\n    }");
    }
    printf("\n  ]\n}\n");
}

static void print_placement_json(struct llama_model * model, int n_layers, size_t cache_mib) {
    printf("{\n");
    printf("  \"format\": 1,\n");
    printf("  \"model_sha256\": \"probe\",\n");
    printf("  \"expert_count\": 256,\n");
    printf("  \"top_k\": 8,\n");
    printf("  \"n_layers\": %d,\n", n_layers);
    printf("  \"cache_mib\": %zu,\n", cache_mib);
    printf("  \"layers\": [\n");
    for (int l = 0; l < n_layers; l++) {
        char gname[128], uname[128], dname[128];
        snprintf(gname, sizeof(gname), "blk.%d.ffn_gate_exps.weight", l);
        snprintf(uname, sizeof(uname), "blk.%d.ffn_up_exps.weight", l);
        snprintf(dname, sizeof(dname), "blk.%d.ffn_down_exps.weight", l);

        const struct ggml_tensor * gate = llama_model_get_tensor(model, gname);
        const struct ggml_tensor * up   = llama_model_get_tensor(model, uname);
        const struct ggml_tensor * down = llama_model_get_tensor(model, dname);

        const bool gate_host = gate && ggml_backend_buffer_is_host(gate->buffer);
        const bool up_host   = up   && ggml_backend_buffer_is_host(up->buffer);
        const bool down_host = down && ggml_backend_buffer_is_host(down->buffer);
        const bool all_gpu   = (!gate || !gate_host) && (!up || !up_host) && (!down || !down_host);
        const bool all_cpu   = gate_host && up_host && down_host;
        const bool is_split  = !all_gpu && !all_cpu;
        const char * placement = all_gpu ? "gpu" : (all_cpu ? "cpu" : "split");

        if (l > 0) printf(",");
        printf("\n    {\"layer\": %d, \"placement\": \"%s\", \"host_moe_tensors\": [", l, placement);
        bool first_h = true;
        if (gate && gate_host) {
            if (!first_h) printf(", ");
            printf("\"%s\"", gname); first_h = false;
        }
        if (up && up_host) {
            if (!first_h) printf(", ");
            printf("\"%s\"", uname); first_h = false;
        }
        if (down && down_host) {
            if (!first_h) printf(", ");
            printf("\"%s\"", dname);
        }
        printf("]}");
    }
    printf("\n  ]\n}\n");
}

int main(int argc, char ** argv) {
    std::string model_path = "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf";
    int n_threads = 14;
    size_t fit_target_mib = 256;
    bool do_geometry = false;
    bool do_placement = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--geometry-json") == 0) {
            do_geometry = true;
        } else if (strcmp(argv[i], "--placement-json") == 0) {
            do_placement = true;
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-fitt") == 0 && i + 1 < argc) {
            fit_target_mib = (size_t) atoi(argv[++i]);
        }
    }

    if (!do_geometry && !do_placement) {
        fprintf(stderr, "Usage: %s --geometry-json|--placement-json -m model.gguf [-fitt MiB] [-t threads]\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    common_params params;
    params.model.path = model_path;
    params.n_gpu_layers = -1;
    params.fit_params = true;
    params.fit_params_target = std::vector<size_t>(llama_max_devices(), fit_target_mib * 1024 * 1024ULL);
    params.fit_params_min_ctx = 512;
    params.cpuparams.n_threads = n_threads;
    params.cpuparams_batch.n_threads = n_threads;
    params.n_ctx = 512;

    common_init_result_ptr init = common_init_from_params(params);
    if (!init || !init->model()) {
        fprintf(stderr, "ERROR: failed to load model\n");
        llama_backend_free();
        return 1;
    }

    auto * model = init->model();
    const int n_layers = 40;
    const int n_experts = 256;

    const char * sha256 = "a2f6c7fdbe82113a2e48e2c38022b55bdcc4308a8002da96cf6d48dab67bb77d";
    if (do_geometry) {
        print_geometry_json(model, n_layers, n_experts, model_path.c_str(), sha256);
    }
    if (do_placement) {
        print_placement_json(model, n_layers, fit_target_mib);
    }

    llama_backend_free();
    return 0;
}
