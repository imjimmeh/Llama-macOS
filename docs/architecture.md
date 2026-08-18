# llama.cpp Architecture Documentation

## Overview

llama.cpp is an LLM/VLM inference engine written in C/C++. It provides efficient inference on CPU and GPU hardware with minimal dependencies. The project sits on top of the ggml tensor library.

## High-Level Architecture

```
+-------------------------------------------------------------------+
|                        Tools / CLI                                 |
|  llama (main)  server  quantize  perplexity  batched-bench  ...   |
+------------------------+---------------------------+----------------+
                         |                           |
                         v                           v
+-------------------------------------------------------------------+
|                     Library Layer (libllama)                       |
|  llama.h (public C API)                                           |
|                                                                    |
|  +------------------+  +------------------+  +-----------------+  |
|  |  Model Loading   |  |  Inference Core  |  |  Sampling & KV  |  |
|  |  - llama-model   |  |  - llama-context |  |  - llama-kv-cache| |
|  |  - llama-vocab   |  |  - llama-batch   |  |  - llama-memory  | |
|  |  - llama-io      |  |  - llama-graph   |  |  - llama-sampler | |
|  +------------------+  +------------------+  +-----------------+  |
+------------------------+---------------------------+----------------+
                         |                           |
                         v                           v
+-------------------------------------------------------------------+
|                      ggml Backend Layer                            |
|  ggml-backend.h (backend registry, buffer management)             |
|                                                                    |
|  +----------+  +--------+  +-------+  +----------+  +---------+   |
|  |   CPU    |  | CUDA   |  | Metal |  | Vulkan   |  | SYCL    |   |
|  | scheduler|  | kernels|  | kernels|  | backend  |  | backend |   |
|  +----------+  +--------+  +-------+  +----------+  +---------+   |
|  +----------+  +--------+  +-------+  +----------+  +---------+   |
|  |   HIP    |  | MUSA   |  | OpenCL|  | OpenVINO |  | CANN  |   |
|  | kernels  |  | kernels|  | kernels|  | backend  |  | backend |   |
|  +----------+  +--------+  +-------+  +----------+  +---------+   |
+-------------------------------------------------------------------+
                         |
                         v
+-------------------------------------------------------------------+
|                     Hardware / OS                                  |
|  x86 (AVX/AVX2/AVX512/AMX) | ARM (NEON) | RISC-V (RVV)          |
|  NVIDIA GPU | AMD GPU | Apple Silicon | Ascend NPU                |
+-------------------------------------------------------------------+
```

## Core Components

### 1. ggml - Tensor Library

ggml is the foundational tensor library. It provides:

- Tensor definitions and operations graph
- Backend abstraction (CPU, CUDA, Metal, Vulkan, SYCL, etc.)
- Buffer management (host/device memory)
- Memory allocation and scheduling
- Quantization support (Q4_0, Q8_0, Q5_K, Q2_K, etc.)

```
+-------------------------+
|      ggml_tensor        |
|  (dimensions, data,     |
|   backend, usage)       |
+-------------------------+
            |
            v
+-------------------------+
|   ggml_cgraph (compute  |
|   graph / operations)   |
+-------------------------+
            |
            v
+-------------------------+
|   ggml_backend_sched    |
|  (schedules tensors     |
|   to backends)          |
+-------------------------+
            |
            v
+-------------------------+
|   ggml_backend_reg      |
|  (backend registry:     |
|   CPU, CUDA, Metal...)  |
+-------------------------+
```

### 2. llama - Inference Engine

The llama library wraps ggml and provides LLM-specific functionality:

```
+---------------------------+
|       llama_model         |
|  - architecture (llm_type)|
|  - hparams (layers, dim,  |
|    n_heads, etc.)         |
|  - vocab                  |
|  - weights (loaded from   |
|    GGUF)                  |
+---------------------------+

+---------------------------+
|     llama_context         |
|  - compute graph          |
|  - KV cache               |
|  - batch processing       |
|  - backend scheduler      |
|  - memory (hybrid, MSA,   |
|    etc.)                  |
+---------------------------+

+---------------------------+
|     llama_sampler_chain   |
|  - temperature            |
|  - top-k / top-p          |
|  - grammar                |
|  - dry / repetition       |
+---------------------------+
```

### 3. Model Loader

The model loader handles GGUF file parsing:

```
GGUF File
    |
    v
+------------------+
|  gguf_context    |  (GGUF parser)
|  - magic, version|
|  - tensor meta   |
|  - metadata      |
+------------------+
    |
    v
+------------------+
|  llama_model     |
|  loader          |
|  - load tensors  |
|  - validate arch |
|  - create model  |
+------------------+
    |
    v
+------------------+
|  ggml_backend    |
|  buffer          |  (allocate weight buffers on device)
+------------------+
```

### 4. Memory System

llama.cpp uses an abstract memory interface for KV storage:

```
+--------------------+
|   llama_memory_i   |  (abstract interface)
|  - init/freeze     |
|  - update          |
|  - get_capacity    |
+--------------------+
        ^
        |
  +-----+-----+-----+
  |     |     |     |
  v     v     v     v
+-----+ +-----+ +-----+ +-----+
|kv_  | |hyb  | |rec   | |msa  |
|cache| |rid  | |urrent| |cascaded|
+-----+ +-----+ +-----+ +-----+
```

### 5. Server (HTTP API)

The built-in server provides an OpenAI-compatible API:

```
+--------------------------------------------------+
|                 llama-server                      |
+--------------------------------------------------+
|  main.cpp (entry point)                          |
|       |                                          |
|       v                                          |
|  +------------------+   +---------------------+  |
|  | server.cpp       |   | server-http.cpp     |  |
|  | (HTTP server,    |   | (HTTP request       |  |
|  |  routing)        |   |  parsing, response) |  |
|  +------------------+   +---------------------+  |
|       |                                          |
|       v                                          |
|  +------------------+   +---------------------+  |
|  | server-task.cpp  |   | server-context.cpp  |  |
|  | (task lifecycle, |   | (per-request        |  |
|  |  concurrency)    |   |  llama context)     |  |
|  +------------------+   +---------------------+  |
|       |                                          |
|       v                                          |
|  +------------------+   +---------------------+  |
|  | server-queue.cpp |   | server-stream.cpp   |  |
|  | (request queue,  |   | (SSE streaming)     |  |
|  |  scheduling)     |   |                     |  |
|  +------------------+   +---------------------+  |
|       |                                          |
|       v                                          |
|  +------------------+   +---------------------+  |
|  | server-models.cpp|   | server-tools.cpp    |  |
|  | (model loading,  |   | (function calling,  |  |
|  |  model switching) |   |  grammar)           |  |
|  +------------------+   +---------------------+  |
|       |                                          |
|       v                                          |
|  +------------------+   +---------------------+  |
|  | server-chat.cpp  |   | server-mcp.cpp      |  |
|  | (chat templates, |   | (Model Context       |  |
|  |  prompt format)  |   |  Protocol)           |  |
|  +------------------+   +---------------------+  |
+--------------------------------------------------+
```

## Data Flow

### Inference Pipeline

```
+----------+     +-----------+     +----------+     +------------+
|  GGUF    | --> | llama_model| --> | llama_   | --> | llama_     |
|  Load    |     | (weights, |     | context  |     | decode     |
|          |     |  hparams) |     | (graph,  |     | (compute)  |
|          |     +-----------+     | KV)      |     +------------+
+----------+                      +----------+            |
                                               |          v
                                               |     +------------+
                                               |     | sample     |
                                               |     | (sampler)  |
                                               |     +------------+
                                               |          |
                                               v          v
                                        +-------------------+
                                        |  token output     |
                                        +-------------------+
```

### Batch Processing Flow

```
+-------------------+
| llama_batch       |
|  - tokens         |
|  - embeddings     |
|  - positions      |
|  - seq_ids        |
|  - logits mask    |
+-------------------+
        |
        v
+-------------------+
| llama_encode()    |  (build compute graph)
+-------------------+
        |
        v
+-------------------+
| llama_decode()    |  (execute graph)
+-------------------+
        |
        v
+-------------------+
| ggml_backend_sched|
|  - assign tensors |
|  - to backends    |
|  - execute        |
+-------------------+
```

### Server Request Flow

```
Client Request
      |
      v
+------------------+
| server-http.cpp  |  (parse HTTP, route)
+------------------+
      |
      v
+------------------+
| server-task.cpp  |  (create task, assign context)
+------------------+
      |
      v
+------------------+
| server-context   |  (wrap llama_context)
+------------------+
      |
      v
+------------------+
| llama_encode()   |  (encode prompt)
+------------------+
      |
      v
+------------------+
| llama_decode()   |  (autoregressive decode)
+------------------+
      |
      v
+------------------+
| server-stream.cpp|  (SSE response)
+------------------+
      |
      v
Client Response (JSON/SSE)
```

## Backend System

### Backend Registry

```
+---------------------------+
|   ggml_backend_reg_t      |  (backend registration)
|  (CPU, CUDA, Metal, ...)  |
+---------------------------+
            |
            v
+---------------------------+
|   ggml_backend_dev_t      |  (device info, capabilities)
+---------------------------+
            |
            v
+---------------------------+
|   ggml_backend_buft_t     |  (buffer type per device)
+---------------------------+
            |
            v
+---------------------------+
|   ggml_backend_buffer_t   |  (allocated memory buffer)
+---------------------------+
```

### Backend Plugin Architecture

Each backend (CUDA, Metal, Vulkan, etc.) implements:
- `ggml_backend_dev_register()` - register the device
- Buffer type allocator
- Device capabilities query
- Backend operations (forward, backward)

```
+------------------------+
| ggml-backend.cpp       |  (core backend management)
| - registry             |
| - device discovery     |
| - plugin loading       |
+------------------------+
         |
   +-----+-----+-----+-----+-----+
   |     |     |     |     |
   v     v     v     v     v
+---+ +---+ +---+ +---+ +---+
|CPU| |CUD| |Met| |Vul| |SYC|
|  a|  a| l | an| CL|  L|
+---+ +---+ +---+ +---+ +---+
```

## Common Utilities

```
+--------------------------+
| common/                  |
|                          |
|  arg.cpp      (CLI args) |
|  chat.cpp     (chat)     |
|  jinja/       (template) |
|  sampling.cpp (samplers) |
|  speculative.cpp (spec)  |
|  ngram-cache.cpp         |
|  log.cpp        (logging)|
|  download.cpp (models)   |
|  preset.cpp   (presets)  |
+--------------------------+
```

## Quantization System

```
+---------------------------+
|  llama_ftype enum         |
|  - F32, F16, BF16        |
|  - Q4_0, Q4_1, Q8_0      |
|  - Q5_K, Q3_K, Q2_K      |
|  - IQ2, IQ3, IQ4 (extreme)|
|  - MXFP4_MOE, NVFP4      |
+---------------------------+
            |
            v
+---------------------------+
| ggml_quants.c             |  (quantization kernels)
| - dequantize             |
| - quantize               |
| - quantize per-channel   |
+---------------------------+
            |
            v
+---------------------------+
| ggml-backend (dequant)    |  (runtime dequant on device)
+---------------------------+
```

## Model Architecture Support

The `llm_type` enum in `llama-model.h` covers 100+ model architectures:

```
+-----------------------------------+
|  llm_type (in src/llama-model.h)  |
+-----------------------------------+
|  LLAMA_TYPE_*                     |
|  QWEN2_TYPE_*                     |
|  MISTRAL_TYPE_*                   |
|  GPT2_TYPE_*                      |
|  LLAMA_TYPE_*                     |
|  PHI2_TYPE_*                      |
|  STABLELM2_TYPE_*                 |
|  DEEPSEEK2_TYPE_*                 |
|  GEMMA_TYPE_*                     |
|  ... (100+ types total)           |
+-----------------------------------+
```

Each architecture has a corresponding adapter in `llama-adapter.cpp` that handles:
- Architecture-specific weight loading
- RoPE configuration
- Attention parameters
- Model-specific quirks

## Key Design Decisions

1. **ggml as dependency**: llama.cpp builds on ggml for tensor operations, not the other way around
2. **C API first**: `include/llama.h` is a C-style API for maximum language interoperability
3. **Backend abstraction**: ggml-backend allows plugging in any compute backend
4. **GGUF format**: Custom binary format for model storage with built-in metadata
5. **Quantization-first**: All models load as quantized weights, dequant at runtime
6. **CPU+GPU hybrid**: Supports partial model offloading across devices
7. **Single binary**: All tools bundle into a single `llama` executable

## Directory Structure Reference

```
llama.cpp/
├── include/              # Public headers (llama.h, ggml.h, gguf.h)
├── src/                  # llama library source
│   ├── llama-*.cpp/h     # Core inference components
│   └── models/           # Model adapter implementations
├── ggml/                 # ggml tensor library (submodule)
│   ├── include/          # ggml public headers
│   └── src/              # ggml source + backend kernels
├── common/               # Shared utilities
│   ├── jinja/            # Jinja template engine
│   └── *.cpp             # CLI helpers, sampling, etc.
├── tools/                # CLI tools
│   ├── server/           # OpenAI-compatible HTTP server
│   ├── quantize/         # Quantization tool
│   ├── cli/              # Main llama binary
│   └── *.                # Other tools
├── examples/             # Example programs
├── tests/                # Test suite
├── cmake/                # CMake build configuration
└── docs/                 # Documentation
```
