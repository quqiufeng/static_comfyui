#!/bin/bash
# =============================================================================
# build_sd_dl.sh - 编译 /opt/sd 为动态后端共享库
# =============================================================================
# 用途：将 /opt/sd 编译为 shared library + 动态 backend plugin，
#      使 CUDA 后端 (libggml-cuda.so) 与主库分离，便于 CPU-only 部署。
#
# 产物路径：/opt/sd/build-dl
# 关键选项：
#   - SD_BUILD_SHARED_LIBS=ON
#   - SD_BUILD_SHARED_GGML_LIB=ON
#   - GGML_BACKEND_DL=ON
#   - CMAKE_CUDA_ARCHITECTURES=native
#
# 用法：
#   ./build_sd_dl.sh                  # 默认 Release + CUDA + 动态后端
#   BUILD_TYPE=Debug ./build_sd_dl.sh
#   JOBS=4 ./build_sd_dl.sh
# =============================================================================

set -euo pipefail

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# /opt/sd 路径（稳定扩散 C++ 后端）
SD_DIR="${SD_DIR:-/opt/sd}"
SD_BUILD_DIR="${SD_BUILD_DIR:-${SD_DIR}/build-dl}"

# 构建类型
BUILD_TYPE="${BUILD_TYPE:-Release}"

# 并行任务数，默认使用所有 CPU 核心
JOBS="${JOBS:-$(nproc)}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================"
echo "  Building stable-diffusion.cpp (DL)"
echo "========================================"
echo "SD_DIR:       ${SD_DIR}"
echo "SD_BUILD_DIR: ${SD_BUILD_DIR}"
echo "BUILD_TYPE:   ${BUILD_TYPE}"
echo "JOBS:         ${JOBS}"
echo ""

# 1. 检查 /opt/sd 是否存在
if [ ! -d "${SD_DIR}" ]; then
    echo -e "${RED}Error: ${SD_DIR} not found${NC}"
    echo "Please clone first:"
    echo "  git clone --depth 1 https://github.com/leejet/stable-diffusion.cpp ${SD_DIR}"
    exit 1
fi

# 2. 检查 git 状态
cd "${SD_DIR}"
if [ ! -d ".git" ]; then
    echo -e "${RED}Error: ${SD_DIR} is not a git repository${NC}"
    exit 1
fi

CURRENT_COMMIT=$(git rev-parse --short HEAD)
echo -e "${BLUE}Current sd.cpp commit: ${CURRENT_COMMIT}${NC}"

# 3. 检查子模块是否已初始化
if [ ! -f "ggml/CMakeLists.txt" ]; then
    echo -e "${YELLOW}Submodules not initialized, initializing...${NC}"
    git submodule update --init --recursive
fi

# 4. 检测 GPU
CUDA_AVAILABLE=0
if command -v nvidia-smi > /dev/null 2>>1; then
    echo -e "${GREEN}✓ CUDA device detected:${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
    CUDA_AVAILABLE=1
else
    echo -e "${YELLOW}⚠ nvidia-smi not found, building CPU-only${NC}"
fi

# 5. 选择编译器
if command -v gcc-12 > /dev/null 2>>1 && command -v g++-12 > /dev/null 2>>1; then
    CMAKE_C_COMPILER="-DCMAKE_C_COMPILER=/usr/bin/gcc-12"
    CMAKE_CXX_COMPILER="-DCMAKE_CXX_COMPILER=/usr/bin/g++-12"
    echo -e "${BLUE}Using GCC 12${NC}"
elif command -v gcc > /dev/null 2>>1; then
    CMAKE_C_COMPILER="-DCMAKE_C_COMPILER=/usr/bin/gcc"
    CMAKE_CXX_COMPILER="-DCMAKE_CXX_COMPILER=/usr/bin/g++"
    GCC_VERSION=$(gcc --version | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    echo -e "${BLUE}Using GCC ${GCC_VERSION}${NC}"
else
    echo -e "${RED}Error: No GCC found${NC}"
    exit 1
fi

# 6. 配置 CMake 参数
CMAKE_ARGS=(
    "-S" "${SD_DIR}"
    "-B" "${SD_BUILD_DIR}"
    "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
    "${CMAKE_C_COMPILER}"
    "${CMAKE_CXX_COMPILER}"
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
    "-DSD_BUILD_SHARED_LIBS=ON"
    "-DSD_BUILD_SHARED_GGML_LIB=ON"
    "-DGGML_BACKEND_DL=ON"
    "-DGGML_NATIVE=OFF"
    "-DCMAKE_CUDA_ARCHITECTURES=native"
    "-DGGML_LTO=OFF"
)

if [ "${CUDA_AVAILABLE}" -eq 1 ]; then
    CMAKE_ARGS+=(
        "-DSD_CUDA=ON"
        "-DGGML_CUDA=ON"
    )
    echo -e "${GREEN}✓ CUDA build enabled${NC}"
else
    CMAKE_ARGS+=(
        "-DSD_CUDA=OFF"
        "-DGGML_CUDA=OFF"
    )
    echo -e "${YELLOW}⚠ CPU build (no CUDA)${NC}"
fi

# 7. 清理并创建 build 目录
if [ -d "${SD_BUILD_DIR}" ]; then
    echo -e "${YELLOW}Cleaning existing build directory...${NC}"
    rm -rf "${SD_BUILD_DIR}"
fi
mkdir -p "${SD_BUILD_DIR}"

# 8. 运行 CMake
echo ""
echo -e "${BLUE}Configuring CMake...${NC}"
cmake "${CMAKE_ARGS[@]}"

# 9. 编译
echo ""
echo -e "${BLUE}Building stable-diffusion.cpp (shared + dynamic backends)...${NC}"
make -C "${SD_BUILD_DIR}" -j"${JOBS}" stable-diffusion
# 显式编译 CUDA 后端插件（在 shared 目标中可能不触发）
if [ "${CUDA_AVAILABLE}" -eq 1 ]; then
    make -C "${SD_BUILD_DIR}" -j"${JOBS}" ggml-cuda
fi
make -C "${SD_BUILD_DIR}" -j"${JOBS}" ggml-cpu

# 10. 检查产物
echo ""
echo "========================================"
echo "  Build artifacts check"
echo "========================================"

REQUIRED_LIBS=(
    "${SD_BUILD_DIR}/bin/libstable-diffusion.so"
    "${SD_BUILD_DIR}/bin/libggml.so"
    "${SD_BUILD_DIR}/bin/libggml-base.so"
    "${SD_BUILD_DIR}/bin/libggml-cpu.so"
)

if [ "${CUDA_AVAILABLE}" -eq 1 ]; then
    REQUIRED_LIBS+=("${SD_BUILD_DIR}/bin/libggml-cuda.so")
fi

ALL_OK=1
for lib in "${REQUIRED_LIBS[@]}"; do
    if [ -f "${lib}" ]; then
        size=$(du -h "${lib}" | cut -f1)
        echo -e "${GREEN}✓${NC} ${lib} (${size})"
    else
        echo -e "${RED}✗ ${lib} not found${NC}"
        ALL_OK=0
    fi
done

if [ "${ALL_OK}" -eq 0 ]; then
    echo -e "${RED}Build failed: some libraries are missing${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ stable-diffusion.cpp (DL) built successfully${NC}"
echo "Build dir: ${SD_BUILD_DIR}"
echo ""

# 11. 记录版本信息
LOCK_FILE="${SCRIPT_DIR}/SD_VERSION.lock"
if [ -d "${SCRIPT_DIR}/.git" ] || [ -f "${SCRIPT_DIR}/CMakeLists.txt" ]; then
    echo "${CURRENT_COMMIT}" > "${LOCK_FILE}"
    echo -e "${BLUE}Updated SD_VERSION.lock: ${CURRENT_COMMIT}${NC}"
fi

exit 0
