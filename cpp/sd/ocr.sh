#!/bin/bash
# =============================================================================
# ocr.sh — 图片文字识别（PaddleOCR PP-OCRv4，ONNX Runtime + CUDA）
#
# 用法: ./ocr.sh <图片> [--json]
#       ./ocr.sh screenshot.png          # 按行输出识别文字
#       ./ocr.sh screenshot.png --json   # 输出 [{"x","y","w","h","text"},...]
#
# 环境变量:
#   OCR_MODEL_DIR  模型目录 (默认 /data/models/ocr)
#   OUT            结果写到该文件 (默认仅打印)
# =============================================================================
set -euo pipefail

CYAN="\033[0;36m"; GREEN="\033[0;32m"; RED="\033[0;31m"; NC="\033[0m"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMGSOCR="${IMGSOCR:-$SCRIPT_DIR/../ocr/imgsocr}"
OCR_MODEL_DIR="${OCR_MODEL_DIR:-/data/models/ocr}"

die() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

[ "$#" -ge 1 ] || die "用法: ./ocr.sh <图片> [--json]"
IN="$1"
[ -f "$IN" ] || die "图片不存在: $IN"
[ -x "$IMGSOCR" ] || die "imgsocr 未编译: $IMGSOCR (先跑 cpp/ocr/build.sh)"

RESULT="$("$IMGSOCR" --model-dir "$OCR_MODEL_DIR" "$@")"
if [ -n "${OUT:-}" ]; then
    printf '%s\n' "$RESULT" > "$OUT"
    echo -e "${GREEN}✓ written ${OUT}${NC}"
else
    echo -e "${CYAN}==== OCR: $IN ====${NC}"
    printf '%s\n' "$RESULT"
fi
