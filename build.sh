#!/bin/bash
# build.sh — ComfyCLI 编译脚本
#
# 只编译二进制，不做打包部署。部署请用 deploy.sh。
# 用法:
#   ./build.sh                           # 编译全部
#   GLIBC_TARGET=2.35 ./build.sh        # 指定目标 GLIBC 版本（用于 GLIBC 兼容编译）
#
# 设计原则：
#   - 职责单一：只编译，不打包
#   - 编译产物 = comfycli-bin + comfycli-bin.so + libsdcpp_adapter.so
#   - GLIBC 兼容：编译时用 /opt/deb/<version>/ 做 sysroot 链接，
#     确保二进制版本标签 ≤ 目标 GLIBC（配合 deploy.sh 的 lib/ 兼容层使用）

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATICPY_DIR="$PROJECT_DIR/staticpy"
CPP_DIR="$PROJECT_DIR/cpp"

GLIBC_TARGET="${GLIBC_TARGET:-}"
if [ -n "$GLIBC_TARGET" ]; then
  GLIBC_SYSROOT="/opt/deb/$GLIBC_TARGET"
else
  GLIBC_SYSROOT=""
fi

echo "============================================"
echo " ComfyCLI 编译"
echo "============================================"

# ── Step 1: 编译 C++ 推理后端 ──
echo ""
echo ">>> Step 1: 编译 libsdcpp_adapter.so"
GLIBC_SYSROOT="$GLIBC_SYSROOT" bash "$CPP_DIR/sd/scripts/build.sh"
echo "  OK"

# ── Step 2: 编译 ELF 二进制 ──
echo ""
echo ">>> Step 2: 编译 comfycli-bin"
GLIBC_SYSROOT="$GLIBC_SYSROOT" bash "$STATICPY_DIR/static_build.sh" \
  "$PROJECT_DIR/comfycli/_bundle.static.py" comfycli-bin
echo "  OK"

echo ""
echo "  产物:"
echo "    $PROJECT_DIR/comfycli-bin       (ELF binary)"
echo "    $PROJECT_DIR/comfycli-bin.so    (AOT compiled Scheme)"
echo "    $CPP_DIR/sd/build/libsdcpp_adapter.so  (C++ SD backend)"
echo ""
echo "============================================"
echo " 编译完成"
echo ""
echo "  本地运行（动态后端模式，需指定 CUDA 后端路径）:"
echo "   LD_LIBRARY_PATH=cpp/sd/build:/opt/sd/build-dl/bin \\"
echo "     GGML_BACKEND_PATH=/opt/sd/build-dl/bin/libggml-cuda.so \\"
echo "     ./comfycli-bin workflow.json --output-dir ./output"
echo ""
echo " 部署:"
echo "   # GPU 远程（远程已有 CUDA Runtime）:"
echo "   ./deploy.sh"
echo ""
echo "   # GPU 远程（无 CUDA Runtime，一起打包）:"
echo "   WITH_CUDA=1 ./deploy.sh"
echo ""
echo "   # CPU-only 远程:"
echo "   WITH_CUDA_BACKEND=0 ./deploy.sh"
echo ""
echo "   # 指定 GLIBC 版本:"
echo "   GLIBC_TARGET=2.35 ./deploy.sh"
echo "============================================"
