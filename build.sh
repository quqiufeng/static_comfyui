#!/usr/bin/env bash
# build.sh — StaticPy → Scheme → Chez AOT .so 编译管线
#
# 用法:
#   ./build.sh input.static.py             编译单个文件
#   ./build.sh input1.static.py input2.py  编译多个文件拼接
#
# 环境变量:
#   CHEZ_SCHEME=scheme        Chez Scheme 可执行文件路径 (默认: scheme)
#   BUILD_DIR=build_out       输出目录 (默认: build_out)
#   TORCH_STD_HELPER=cpp_runtime      libtorch_std_helper 目录 (默认: cpp_runtime)
#   CLEAN=1                   先清理 build 目录

set -euo pipefail

# ====== 配置 ======
CHEZ="${CHEZ_SCHEME:-scheme}"
BUILD_DIR="${BUILD_DIR:-build_out}"
LIBS_DIR="${BUILD_DIR}/libs"
TORCH_STD="${TORCH_STD_HELPER:-$(cd "$(dirname "$0")" && pwd)/cpp_runtime}"

COMPILER_DIR="$(cd "$(dirname "$0")/compiler" && pwd)"
PRELUDE="${COMPILER_DIR}/prelude.scm"
STDLIB="${COMPILER_DIR}/stdlib.scm"
TRANSLATE="${COMPILER_DIR}/translate.py"

mkdir -p "${BUILD_DIR}" "${LIBS_DIR}"

# ====== 配置 libtorch_std_helper.so 路径 (供 translate.py 生成 load-shared-object) ======
export STATICPY_TORCH_STD_SO="${STATICPY_TORCH_STD_SO:-${TORCH_STD}/build/libtorch_std_helper.so}"

# ====== 编译 StaticPy → .ss ======
echo "=== Step 1: StaticPy → Scheme ==="

# 拼接所有输入 .static.py → 一次 translate.py 调用 (共享 extern fn 声明)
INPUT_SS="${BUILD_DIR}/input.ss"
TMPPY="${BUILD_DIR}/combined_input.py"
{
    for f in "$@"; do
        if [ -f "$f" ]; then
            cat "$f"
            echo ""
        else
            echo "Error: file not found: $f" >&2
            exit 1
        fi
    done
} > "${TMPPY}"
python3 "${TRANSLATE}" < "${TMPPY}" > "${INPUT_SS}"

echo "  → ${INPUT_SS}"

# ====== 合并 prelude + stdlib + 用户代码 ======
echo "=== Step 2: Merge prelude + stdlib + user code ==="

MERGED_SS="${BUILD_DIR}/merged.ss"
{
    cat "${PRELUDE}"
    echo ""
    echo ";; === stdlib ==="
    cat "${STDLIB}"
    echo ""
    echo ";; === user code ==="
    cat "${INPUT_SS}"
} > "${MERGED_SS}"

echo "  → ${MERGED_SS} ($(wc -l < "${MERGED_SS}") lines)"

# ====== Chez Scheme AOT 编译 ======
echo "=== Step 3: Chez Scheme compile-file ==="

OUTPUT_SO="${BUILD_DIR}/runtime.so"
# 清理旧的 .so 避免重名冲突
rm -f "${OUTPUT_SO}" "${BUILD_DIR}/merged.so"

# compile-file 生成 merged.so (文件名基于源文件)
(cd "${BUILD_DIR}" && "${CHEZ}" --compile-file "../${MERGED_SS}")

# 检查输出
if [ -f "${BUILD_DIR}/merged.so" ]; then
    mv "${BUILD_DIR}/merged.so" "${OUTPUT_SO}"
    echo "  → ${OUTPUT_SO} ($(stat -c%s "${OUTPUT_SO}") bytes)"
else
    echo "Error: compile-file failed" >&2
    ls -la "${BUILD_DIR}/" >&2
    exit 1
fi

# ====== 编译 libtorch_std_helper.so ======
echo "=== Step 4: Build libtorch_std_helper.so ==="

TORCH_SO="${LIBS_DIR}/libtorch_std_helper.so"

# 检测 PyTorch 路径
PYTHON="${PYTHON:-python3}"
TORCH_HOME=$(${PYTHON} -c "import torch; import os; print(os.path.join(torch.__path__[0], 'lib'))" 2>/dev/null || echo "/data/venv/lib/python3.12/site-packages/torch/lib")
TORCH_INCLUDE=$(${PYTHON} -c "import torch; print(torch.__path__[0])" 2>/dev/null || echo "/data/venv/lib/python3.12/site-packages/torch")

if [ -f "${TORCH_STD}/libtorch_std_helper.cpp" ]; then
    echo "  Compiling libtorch_std_helper.so..."
    g++ -O2 -shared -fPIC -o "${TORCH_SO}" \
        "${TORCH_STD}/libtorch_std_helper.cpp" \
        -I"${TORCH_INCLUDE}/include" \
        -I"${TORCH_INCLUDE}/include/torch/csrc/api/include" \
        -L"${TORCH_HOME}" \
        -ltorch -lc10 -ltorch_cpu -ltorch_cuda -lc10_cuda \
        -Wl,-rpath,"${TORCH_HOME}" \
        2>&1 | sed 's/^/  /'
    echo "  → ${TORCH_SO} ($(stat -c%s "${TORCH_SO}") bytes)"
else
    echo "  Warning: libtorch_std_helper.cpp not found at ${TORCH_STD}/"
    echo "  Skip building. Will use pre-built .so if available."
fi

echo ""
echo "=== Build complete ==="
echo "  Runtime SO:  ${OUTPUT_SO}"
echo "  Torch SO:    ${TORCH_SO}"
echo ""
echo "Next: ./deliver.sh to package as ELF"
