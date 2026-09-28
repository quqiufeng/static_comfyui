#!/bin/bash
# =============================================================================
# FreeU 数学自测（对应 patch patches/sdcpp-freeu-sag-v2.patch 的 FreeU 部分）
#   test_fourier   ggml_ext_fourier_filter_lowfreq vs 朴素 DFT 参考
#                  （复刻 ComfyUI Fourier_filter，覆盖 f32/f16、奇偶尺寸）
#   test_freeu_v2  ggml_ext_freeu_v2_backbone vs torch 参考
#                  （hidden_mean 归一化 + 前半通道动态缩放，覆盖 batch=2、f16）
# 依赖：/opt/sd/build-dl（先跑 cpp/sd/build_sd_dl.sh）
# 用法：./run.sh
# =============================================================================
set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SD_DIR="${SD_DIR:-/opt/sd}"
BIN="${BIN:-${SD_DIR}/build-dl/bin}"

export LD_LIBRARY_PATH="$BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CXXFLAGS=(-std=c++17 -O2 -DGGML_MAX_NAME=160 -I"${SD_DIR}/ggml/include" -I"${SD_DIR}/src")
LDFLAGS=(-L"$BIN" -Wl,-rpath,"$BIN" -lggml -lggml-base -lggml-cpu)

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail=0
for t in test_fourier test_freeu_v2; do
    g++ "${CXXFLAGS[@]}" "${TEST_DIR}/${t}.cpp" -o "${WORK}/${t}" "${LDFLAGS[@]}"
    "${WORK}/${t}" || fail=1
done
exit "$fail"
