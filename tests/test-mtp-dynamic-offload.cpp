#include "llama.h"
#include "common.h"
#include "arg.h"

#include <cassert>
#include <cstdio>
#include <vector>
#include <string>

static void test_param_defaults() {
    printf("test_param_defaults\n");

    const llama_model_params mparams = llama_model_default_params();
    assert(!mparams.mtp_dynamic_offload);

    common_params params;
    assert(!params.mtp_dynamic_offload);
    assert(!params.speculative.draft.mtp_dynamic_offload);

    llama_model_params converted = common_model_params_to_llama(params);
    assert(!converted.mtp_dynamic_offload);

    params.mtp_dynamic_offload = true;
    converted = common_model_params_to_llama(params);
    assert(converted.mtp_dynamic_offload);

    params.mtp_dynamic_offload = false;
    params.speculative.draft.mtp_dynamic_offload = true;
    converted = common_model_params_to_llama(params);
    assert(converted.mtp_dynamic_offload);
}

static void test_cli_parsing() {
    printf("test_cli_parsing\n");

    {
        common_params params;
        const char * argv[] = { "llama-cli", "--mtp-dynamic-offload" };
        const bool ok = common_params_parse(2, const_cast<char **>(argv), params, LLAMA_EXAMPLE_CLI);
        assert(ok);
        assert(params.mtp_dynamic_offload);
        assert(params.speculative.draft.mtp_dynamic_offload);
    }

    {
        common_params params;
        params.mtp_dynamic_offload = true;
        const char * argv[] = { "llama-cli", "--no-mtp-dynamic-offload" };
        const bool ok = common_params_parse(2, const_cast<char **>(argv), params, LLAMA_EXAMPLE_CLI);
        assert(ok);
        assert(!params.mtp_dynamic_offload);
        assert(!params.speculative.draft.mtp_dynamic_offload);
    }
}

static void test_c_api_null_safety() {
    printf("test_c_api_null_safety\n");

    assert(!llama_model_has_mtp(nullptr));
    assert(!llama_model_mtp_is_gpu_resident(nullptr));
    assert(!llama_model_mtp_promote_to_gpu(nullptr, nullptr));
    assert(!llama_model_mtp_demote_to_host(nullptr));
}

static void test_layer_budget_logic() {
    printf("test_layer_budget_logic\n");

    const int n_layer_all = 33;
    const int n_layer_nextn = 1;
    const int n_trunk = n_layer_all - n_layer_nextn;

    // Case 1: Non-MTP mode (load_mtp = false)
    {
        const bool load_mtp = false;
        const bool mtp_dynamic_offload = false;
        const bool mtp_active_static = load_mtp && !mtp_dynamic_offload && n_layer_nextn > 0;
        const int n_budget = mtp_active_static ? n_layer_all : n_trunk;
        assert(n_budget == 32);

        const int n_gpu_layers = 33;
        const int i_gpu_start = std::max(n_budget + 1 - n_gpu_layers, 0);
        assert(i_gpu_start == 0);
    }

    // Case 2: Dynamic MTP mode (load_mtp = true, mtp_dynamic_offload = true)
    {
        const bool load_mtp = true;
        const bool mtp_dynamic_offload = true;
        const bool mtp_active_static = load_mtp && !mtp_dynamic_offload && n_layer_nextn > 0;
        const int n_budget = mtp_active_static ? n_layer_all : n_trunk;
        assert(n_budget == 32);

        const int n_gpu_layers = 33;
        const int i_gpu_start = std::max(n_budget + 1 - n_gpu_layers, 0);
        assert(i_gpu_start == 0);
    }

    // Case 3: Static MTP mode (load_mtp = true, mtp_dynamic_offload = false)
    {
        const bool load_mtp = true;
        const bool mtp_dynamic_offload = false;
        const bool mtp_active_static = load_mtp && !mtp_dynamic_offload && n_layer_nextn > 0;
        const int n_budget = mtp_active_static ? n_layer_all : n_trunk;
        assert(n_budget == 33);

        const int n_gpu_layers = 34;
        const int i_gpu_start = std::max(n_budget + 1 - n_gpu_layers, 0);
        assert(i_gpu_start == 0);
    }
}

int main() {
    test_param_defaults();
    test_cli_parsing();
    test_c_api_null_safety();
    test_layer_budget_logic();

    printf("All MTP dynamic offload tests passed successfully.\n");
    return 0;
}
