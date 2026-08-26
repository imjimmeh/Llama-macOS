#!/bin/bash
export GGML_EXPERT_ROUTE_TRACE="G:/code/AI/llamacpptuned/llama.cpp/tools/training_data/route_trace.bin"
cd "G:/code/AI/llamacpptuned/llama.cpp"
exec ./build/bin/Release/llama-server.exe \
  -m "C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf" \
  --ctx-size 4096 \
  --batch-size 4096 \
  --ubatch-size 2048 \
  --threads 14 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --no-context-shift \
  --cache-ram 1024 \
  -ngl 20 \
  -exc 256M \
  -excp 512 \
  -excs \
  --routing-predictor-horizon 8 \
  --routing-predictor-stats \
  --port 8137 \
  --temp 0.8 \
  --top-k 20 \
  --top-p 0.95 \
  --repeat-penalty 1.0 \
  --no-mmproj \
  --no-mmap
