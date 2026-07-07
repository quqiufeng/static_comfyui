#!/bin/bash
# =============================================================================
# scripts/build.sh - 一键编译 cpp/sd 适配层 + 示例
# =============================================================================
# 用法：
#   ./scripts/build.sh              # Release + CUDA
#   BUILD_TYPE=Debug ./scripts/build.sh
#   JOBS=4 ./scripts/build.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================"
echo "  Building cpp/sd adapter + examples"
echo "========================================"
echo "PROJ_DIR:     ${PROJ_DIR}"
echo "BUILD_TYPE:   ${BUILD_TYPE}"
echo "JOBS:         ${JOBS}"
echo ""

# 1. 确保 /opt/sd 已编译
if [ ! -f "/opt/sd/build/libstable-diffusion.a" ]; then
    echo -e "${YELLOW}/opt/sd build not found, running build_sd.sh first...${NC}"
    "${PROJ_DIR}/build_sd.sh"
fi

# 2. 配置并编译
cd "${PROJ_DIR}"
rm -rf build
mkdir build

echo -e "${BLUE}Configuring CMake...${NC}"
cmake -S . -B build -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

echo ""
echo -e "${BLUE}Building...${NC}"
cmake --build build -j"${JOBS}"

echo ""
echo "========================================"
echo "  Build artifacts"
echo "========================================"
if [ -f "${PROJ_DIR}/build/libsdcpp_adapter.a" ]; then
    size=$(du -h "${PROJ_DIR}/build/libsdcpp_adapter.a" | cut -f1)
    echo -e "${GREEN}✓${NC} ${PROJ_DIR}/build/libsdcpp_adapter.a (${size})"
fi
if [ -f "${PROJ_DIR}/build/sdxl_txt2img" ]; then
    size=$(du -h "${PROJ_DIR}/build/sdxl_txt2img" | cut -f1)
    echo -e "${GREEN}✓${NC} ${PROJ_DIR}/build/sdxl_txt2img (${size})"
fi

echo ""
echo -e "${GREEN}✓ cpp/sd build complete${NC}"
echo "Run example:"
echo "  ${PROJ_DIR}/build/sdxl_txt2img -W 1024 -H 1024 -s 42"

exit 0
