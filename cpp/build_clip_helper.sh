#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
TORCH_PATH="/data/venv/lib/python3.12/site-packages/torch"
TORCH_INCLUDE="$TORCH_PATH/include"
TORCH_LIB="$TORCH_PATH/lib"
CUDA_INC="/data/cuda/include"
CUDA_LIB="/data/cuda/lib64"

echo "=== Building clip_helper.so (CLIP from safetensors) ==="
g++ -O2 -shared -fPIC -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 \
    clip_helper.cpp \
    -I"$TORCH_INCLUDE" \
    -I"$TORCH_INCLUDE/torch/csrc/api/include" \
    -I"$CUDA_INC" \
    -L"$TORCH_LIB" \
    -L/opt/ReScheme -ltorch -ltorch_cpu -lc10 -l:libtorch_std_helper.so \
    -Wl,-rpath,"$TORCH_LIB" \
    -o clip_helper.so 2>&1
echo "=== Build complete ==="
ls -lh clip_helper.so
