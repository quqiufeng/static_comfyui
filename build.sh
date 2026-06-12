#!/bin/bash
# build.sh — static_comfyui 自包含编译器
# Python 静态子集 → Scheme → .so 原生机器码
#
# 用法: bash build.sh input.static.py [output_name]
#
# 不依赖外部项目（ReScheme），所有工具链在 compiler/ 和 runtime/ 下

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCHEME="${SCHEME:-/opt/ChezScheme/ta6le/bin/ta6le/scheme}"
BUILD_DIR="${SCRIPT_DIR}/build_tmp"
OUT_DIR="${SCRIPT_DIR}/build_out"

INPUT="${1:-}"
OUTPUT_NAME="${2:-$(basename "$INPUT" .static.py)}"

if [ -z "$INPUT" ] || [ ! -f "$INPUT" ]; then
    echo "Usage: bash build.sh input.static.py [output_name]"
    echo "  input.static.py  — StaticPy 源文件"
    echo "  output_name      — 输出文件名（不含扩展名，默认与输入同名）"
    echo ""
    echo "输出: build_out/output_name.so"
    exit 1
fi

mkdir -p "$BUILD_DIR" "$OUT_DIR"

echo "=== static_comfyui — Build ==="
echo "Source: $INPUT"

# Step 1: 翻译 StaticPy → Scheme
echo "  [1/3] Translating StaticPy → Scheme..."
/data/venv/bin/python3 "${SCRIPT_DIR}/compiler/translate.py" < "$INPUT" > "$BUILD_DIR/${OUTPUT_NAME}_code.ss"

# Step 2: 合并 prelude + stdlib + 用户代码
echo "  [2/3] Merging runtime libraries..."
MERGED_SS="$BUILD_DIR/${OUTPUT_NAME}.ss"
cat "${SCRIPT_DIR}/compiler/prelude.scm" > "$MERGED_SS"
echo "" >> "$MERGED_SS"
cat "${SCRIPT_DIR}/compiler/stdlib.scm" >> "$MERGED_SS"
echo "" >> "$MERGED_SS"
cat "$BUILD_DIR/${OUTPUT_NAME}_code.ss" >> "$MERGED_SS"

# Step 3: 编译为 .so
echo "  [3/3] Compiling to native code..."
COMPILE_SS="$BUILD_DIR/compile_${OUTPUT_NAME}.ss"
cat > "$COMPILE_SS" << 'EOF'
(import (chezscheme))
(compile-file "MERGED_SS")
EOF
sed -i "s|MERGED_SS|${MERGED_SS}|g" "$COMPILE_SS"
$SCHEME --quiet "$COMPILE_SS" 2>&1

# 移动产出
cp "${MERGED_SS}o" "$OUT_DIR/${OUTPUT_NAME}.so" 2>/dev/null || \
    find "$BUILD_DIR" -name "${OUTPUT_NAME}.so" -exec cp {} "$OUT_DIR/" \; 2>/dev/null || true

echo "       -> $(ls -lh "$OUT_DIR/${OUTPUT_NAME}.so" 2>/dev/null | awk '{print $5}')"
echo "Output: $OUT_DIR/${OUTPUT_NAME}.so"
echo "Run: $SCHEME --quiet $OUT_DIR/${OUTPUT_NAME}.so"

# 清理大文件
rm -f "$BUILD_DIR/${OUTPUT_NAME}_code.ss" "$COMPILE_SS"
