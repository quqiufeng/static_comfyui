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
#   - 编译产物 = comfycli + comfycli.so + libtorch_std_helper.so
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
echo ">>> Step 1: 编译 libtorch_std_helper.so"
GLIBC_SYSROOT="$GLIBC_SYSROOT" bash "$CPP_DIR/build_torch_std_helper.sh"
echo "  OK"

# ── Step 2: 编译 ELF 二进制 ──
echo ""
echo ">>> Step 2: 编译 comfycli"
GLIBC_SYSROOT="$GLIBC_SYSROOT" bash "$STATICPY_DIR/static_build.sh" \
  "$PROJECT_DIR/main.static.py" comfycli
echo "  OK"

echo ""
echo "  产物:"
echo "    $PROJECT_DIR/comfycli       (ELF binary)"
echo "    $PROJECT_DIR/comfycli.so    (AOT compiled Scheme)"
echo "    $CPP_DIR/libtorch_std_helper.so  (C++ bridge)"
echo ""
echo "============================================"
echo " 编译完成"
echo ""
echo " 本地运行:"
echo "   LD_LIBRARY_PATH=cpp/:/data/venv/lib/python3.12/site-packages/torch/lib \\"
echo "     ./comfycli workflow.json --output-dir ./output"
echo ""
echo " 部署:"
echo "   GLIBC_TARGET=2.35 ./deploy.sh --scp user@host"
echo "============================================"
