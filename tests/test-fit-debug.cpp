#include "llama.h"
#include "common.h"
#include <windows.h>
#include <dbghelp.h>
#include <cstdio>
#include <vector>

#pragma comment(lib, "dbghelp.lib")

LONG WINAPI CrashFilter(EXCEPTION_POINTERS * ExceptionInfo) {
    if (ExceptionInfo->ExceptionRecord->ExceptionCode == 0xE06D7363) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    HANDLE process = GetCurrentProcess();
    HANDLE thread = GetCurrentThread();
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    SymInitialize(process, "G:\\code\\AI\\llamacpptuned\\llama.cpp\\build\\bin\\Release", TRUE);
    SymRefreshModuleList(process);

    CONTEXT ctx = *ExceptionInfo->ContextRecord;
    void * addr = (void *)ctx.Rip;
    HMODULE hMod = NULL;
    char modName[MAX_PATH] = "Unknown";
    uintptr_t offset = 0;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)addr, &hMod)) {
        GetModuleFileNameA(hMod, modName, sizeof(modName));
        offset = (uintptr_t)addr - (uintptr_t)hMod;
        SymLoadModuleEx(process, NULL, modName, NULL, (DWORD64)hMod, 0, NULL, 0);
    }

    char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
    PSYMBOL_INFO symbol = (PSYMBOL_INFO)buffer;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = MAX_SYM_NAME;

    DWORD64 displacement = 0;
    IMAGEHLP_LINE64 line;
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
    DWORD displacementLine = 0;

    fprintf(stderr, "\n\n*** CRASH DETECTED: ExceptionCode = 0x%08X, Address = %p (%s + 0x%llX) ***\n",
        ExceptionInfo->ExceptionRecord->ExceptionCode,
        addr, modName, (unsigned long long)offset);

    if (SymFromAddr(process, (DWORD64)addr, &displacement, symbol)) {
        if (SymGetLineFromAddr64(process, (DWORD64)addr, &displacementLine, &line)) {
            fprintf(stderr, "Faulting function: %s (%s:%d)\n", symbol->Name, line.FileName, line.LineNumber);
        } else {
            fprintf(stderr, "Faulting symbol: %s + 0x%llX\n", symbol->Name, (unsigned long long)displacement);
        }
    }

    STACKFRAME64 stackFrame;
    memset(&stackFrame, 0, sizeof(stackFrame));
    stackFrame.AddrPC.Offset = ctx.Rip;
    stackFrame.AddrPC.Mode = AddrModeFlat;
    stackFrame.AddrFrame.Offset = ctx.Rbp;
    stackFrame.AddrFrame.Mode = AddrModeFlat;
    stackFrame.AddrStack.Offset = ctx.Rsp;
    stackFrame.AddrStack.Mode = AddrModeFlat;

    fprintf(stderr, "\nFault Context Callstack:\n");
    int frameNum = 0;
    while (StackWalk64(IMAGE_FILE_MACHINE_AMD64, process, thread, &stackFrame, &ctx, NULL, SymFunctionTableAccess64, SymGetModuleBase64, NULL) && frameNum < 32) {
        DWORD64 frameDisp = 0;
        char frameMod[MAX_PATH] = "Unknown";
        uintptr_t frameOff = 0;
        HMODULE hFMod = NULL;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)stackFrame.AddrPC.Offset, &hFMod)) {
            GetModuleFileNameA(hFMod, frameMod, sizeof(frameMod));
            frameOff = (uintptr_t)stackFrame.AddrPC.Offset - (uintptr_t)hFMod;
            SymLoadModuleEx(process, NULL, frameMod, NULL, (DWORD64)hFMod, 0, NULL, 0);
        }
        if (SymFromAddr(process, stackFrame.AddrPC.Offset, &frameDisp, symbol)) {
            if (SymGetLineFromAddr64(process, stackFrame.AddrPC.Offset, &displacementLine, &line)) {
                fprintf(stderr, "  [%02d] %s!%s (%s:%d)\n", frameNum, frameMod, symbol->Name, line.FileName, line.LineNumber);
            } else {
                fprintf(stderr, "  [%02d] %s!%s + 0x%llX (offset 0x%llX)\n", frameNum, frameMod, symbol->Name, (unsigned long long)frameDisp, (unsigned long long)frameOff);
            }
        } else {
            fprintf(stderr, "  [%02d] %s + 0x%llX (0x%llX)\n", frameNum, frameMod, (unsigned long long)frameOff, (unsigned long long)stackFrame.AddrPC.Offset);
        }
        frameNum++;
    }

    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}

int main(int argc, char ** argv) {
    AddVectoredExceptionHandler(1, CrashFilter);

    printf("=== Starting test-fit-debug with exact server flags ===\n");
    fflush(stdout);

    std::string model_path = "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf";
    if (argc > 1) {
        model_path = argv[1];
    }

    common_params params;
    params.model.path = model_path;
    params.n_gpu_layers = -1;
    params.fit_params = true;
    params.fit_params_target = std::vector<size_t>(llama_max_devices(), 256 * 1024 * 1024);
    params.fit_params_min_ctx = 128000;
    params.cpuparams.n_threads = 14;
    params.cpuparams_batch.n_threads = 14;
    params.n_ctx = 128000;
    params.n_batch = 4096;
    params.n_ubatch = 2048;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.cache_type_k = GGML_TYPE_Q8_0;
    params.cache_type_v = GGML_TYPE_Q8_0;
    params.kv_unified = true;
    params.expert_cache_size = 1024ULL * 1024 * 1024;
    params.expert_cache_period = 256;
    params.expert_cache_profile = "coder";
    params.expert_cache_persist = false;
    params.expert_cache_stats = true;
    params.pinned_experts_manifest = "G:/code/AI/llamacpptuned/llama.cpp/pinned_experts_1024mb.json";
    params.warmup = true;

    printf("Initializing model and context via common_init_from_params...\n");
    fflush(stdout);

    common_init_result_ptr init = common_init_from_params(params);
    if (!init || !init->model() || !init->context()) {
        fprintf(stderr, "ERROR: Failed to initialize model\n");
        return 1;
    }

    printf("Model and context initialized and warmed up successfully!\n");
    fflush(stdout);

    printf("Testing prompt batch decode with 20 tokens...\n");
    fflush(stdout);

    const auto * vocab = llama_model_get_vocab(init->model());
    std::vector<llama_token> tokens = common_tokenize(vocab, "Hello, this is a test prompt to verify multi-token prefill decoding with MoE expert cache enabled.", true, true);
    printf("Tokenized prompt into %zu tokens\n", tokens.size());
    fflush(stdout);

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    int decode_res = llama_decode(init->context(), batch);
    if (decode_res != 0) {
        fprintf(stderr, "ERROR: llama_decode failed with code %d\n", decode_res);
        return 1;
    }

    printf("Prompt batch decoded successfully!\n");
    fflush(stdout);

    printf("=== SUCCESS ===\n");
    return 0;
}
