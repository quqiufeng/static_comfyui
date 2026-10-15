#!/bin/bash
# build.sh — 构建独立图像处理工具（C++ 底层库 + StaticPy 程序）
#
#   bash imgproc/build.sh
#
# 产物：
#   cpp/imgproc/build/libimgproc.so   （图像编解码底层库，运行时依赖 libgif.so.7）
#   imgproc/mirror_gif + .so          （StaticPy AOT 编译的独立 ELF）
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"

# 1) C++ 底层库
echo ">>> [1/2] libimgproc.so"
( cd "$ROOT/cpp/imgproc" && \
  g++ -O2 -fPIC -shared -Ithirdparty/giflib imgproc.cpp \
      -o build/libimgproc.so -L/usr/lib/x86_64-linux-gnu -l:libgif.so.7 )

# 2) StaticPy 程序 -> 独立 ELF
echo ">>> [2/2] mirror_gif (StaticPy)"
mkdir -p /tmp/staticpy-cache/imgproc
( cd "$ROOT" && bash staticpy/static_build_comfycli.sh \
      imgproc/mirror_gif.static.py imgproc/mirror_gif imgproc/imgproc_ffi.scm )

echo "=== build done ==="
