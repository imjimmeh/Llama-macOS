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

int main() {
    test_param_defaults();
    test_cli_parsing();
    test_c_api_null_safety();

    printf("All MTP dynamic offload tests passed successfully.\n");
    return 0;
}
