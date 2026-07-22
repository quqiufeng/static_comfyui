#!/bin/bash
# =============================================================================
# build_sd.sh - 编译 /opt/sd (stable-diffusion.cpp)
# =============================================================================
# 用途：将 /opt/sd 编译为静态库，供 cpp/sd 兼容层使用
# 默认：启用 GPU (CUDA) 支持
#
# 用法：
#   ./build_sd.sh                  # 默认 Release + CUDA
#   BUILD_TYPE=Debug ./build_sd.sh # Debug 模式
#   JOBS=4 ./build_sd.sh            # 限制并行数
# =============================================================================

set -euo pipefail

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# /opt/sd 路径（稳定扩散 C++ 后端）
SD_DIR="${SD_DIR:-/opt/sd}"
SD_BUILD_DIR="${SD_BUILD_DIR:-${SD_DIR}/build}"

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
echo "  Building stable-diffusion.cpp"
echo "========================================"
echo "SD_DIR:       ${SD_DIR}"
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
    echo -e "${YELLOW}⚠ nvidia-smi not found, falling back to CPU build${NC}"
fi

# 5. 选择编译器（避免 GCC 版本混用导致的 LTO 问题）
# 优先使用 gcc-12 / g++-12，如果存在
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

# 禁用 LTO 以避免 GCC 版本混用问题
CMAKE_ARGS+=("-DGGML_LTO=OFF")

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
echo -e "${BLUE}Building stable-diffusion.cpp...${NC}"
make -C "${SD_BUILD_DIR}" -j"${JOBS}" stable-diffusion

# 10. 检查产物
echo ""
echo "========================================"
echo "  Build artifacts check"
echo "========================================"

REQUIRED_LIBS=(
    "${SD_BUILD_DIR}/libstable-diffusion.a"
    "${SD_BUILD_DIR}/ggml/src/libggml.a"
    "${SD_BUILD_DIR}/ggml/src/libggml-cpu.a"
    "${SD_BUILD_DIR}/ggml/src/libggml-base.a"
)

if [ "${CUDA_AVAILABLE}" -eq 1 ]; then
    REQUIRED_LIBS+=("${SD_BUILD_DIR}/ggml/src/ggml-cuda/libggml-cuda.a")
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
echo -e "${GREEN}✓ stable-diffusion.cpp built successfully${NC}"
echo "Build dir: ${SD_BUILD_DIR}"
echo ""

# 11. 记录版本信息（用于上层 cpp/sd 构建）
LOCK_FILE="${SCRIPT_DIR}/SD_VERSION.lock"
if [ -d "${SCRIPT_DIR}/.git" ] || [ -f "${SCRIPT_DIR}/CMakeLists.txt" ]; then
    echo "${CURRENT_COMMIT}" > "${LOCK_FILE}"
    echo -e "${BLUE}Updated SD_VERSION.lock: ${CURRENT_COMMIT}${NC}"
fi

exit 0
