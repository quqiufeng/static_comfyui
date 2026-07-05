#!/bin/bash
# build_torch_std_helper.sh — 编译 libtorch_std_helper.so
set -euo pipefail

cd "$(dirname "$0")"

TORCH_PATH="/data/venv/lib/python3.12/site-packages/torch"
if [ ! -d "$TORCH_PATH" ]; then
    echo "Error: torch not found at $TORCH_PATH"
    exit 1
fi

TORCH_INCLUDE="$TORCH_PATH/include"
TORCH_LIB="$TORCH_PATH/lib"

echo "=== Building libtorch_std_helper.so ==="

GLIBC_SYSROOT="${GLIBC_SYSROOT:-}"
LDFLAGS=()
if [ -n "$GLIBC_SYSROOT" ]; then
  LDFLAGS+=("-L$GLIBC_SYSROOT/lib")
  LDFLAGS+=("-L$GLIBC_SYSROOT/lib/x86_64-linux-gnu")
  LDFLAGS+=("-L$GLIBC_SYSROOT/usr/lib/x86_64-linux-gnu")
  LDFLAGS+=("-Wl,-rpath-link,$GLIBC_SYSROOT/lib")
  LDFLAGS+=("-Wl,-rpath-link,$GLIBC_SYSROOT/lib/x86_64-linux-gnu")
  LDFLAGS+=("-Wl,-rpath-link,$GLIBC_SYSROOT/usr/lib/x86_64-linux-gnu")
fi

CUDA_INC="/data/cuda/include"
CUDA_LIB="/data/cuda/lib64"

g++ -O3 -shared -fPIC -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 \
    libtorch_std_helper.cpp \
    -I"$TORCH_INCLUDE" \
    -I"$TORCH_INCLUDE/torch/csrc/api/include" \
    -I"$CUDA_INC" \
    -I/usr/local/cuda/include \
    -I"$TORCH_INCLUDE/c10/cuda/.." \
    "${LDFLAGS[@]}" \
    -L"$TORCH_LIB" \
    -L"$CUDA_LIB" \
    -ltorch -ltorch_cpu -lc10 -lcuda -lcudart -lpng \
    -Wl,-rpath,"$TORCH_LIB" \
    -o libtorch_std_helper.so

echo "=== Build complete ==="
echo "Output: libtorch_std_helper.so"
ls -lh libtorch_std_helper.so
