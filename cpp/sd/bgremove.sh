#!/bin/bash
# =============================================================================
# bgremove.sh — 去除图片背景（BiRefNet，ONNX Runtime + CUDA）
#
# 用法: ./bgremove.sh <输入图> [output.png] [mode]
#       ./bgremove.sh photo.jpg                    # 透明背景, 存到 $HOME
#       ./bgremove.sh photo.jpg out.png            # 指定输出
#       ./bgremove.sh photo.jpg out.png white      # 白底
#       ./bgremove.sh photo.jpg out.png green      # 绿幕
#       ./bgremove.sh photo.jpg out.png color      # 纯色底 (COLOR=#rrggbb)
#       ./bgremove.sh photo.jpg out.png mask       # 只输出灰度遮罩
#
# mode: transparent(默认) | white | black | green | color | mask
#
# 环境变量:
#   COLOR    mode=color 时的底色 (#rrggbb, 默认 #ffffff)
#   MASK     额外输出灰度遮罩到该路径 (非 mask 模式)
#   MODEL    BiRefNet ONNX 路径 (默认 /data/models/birefnet/onnx/model.onnx)
#   CPU=1    强制 CPU
# =============================================================================
set -euo pipefail

RED="\033[0;31m"; GREEN="\033[0;32m"; YELLOW="\033[1;33m"
BLUE="\033[0;34m"; CYAN="\033[0;36m"; NC="\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BGREMOVE="${BGREMOVE:-$SCRIPT_DIR/../birefnet/bgremove}"
MODEL="${MODEL:-${BIREFNET_MODEL:-/data/models/birefnet/onnx/model.onnx}}"
MODE="${MODE:-transparent}"
COLOR="${COLOR:-#ffffff}"
MASK="${MASK:-}"
CPU="${CPU:-0}"

die() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }
check_file() { [ -f "$1" ] || die "not found: $1"; }

if [ "$#" -lt 1 ]; then
    die "用法: ./bgremove.sh <输入图> [output.png] [mode]"
fi

IN="$1"
OUT="${2:-}"
[ -n "${3:-}" ] && MODE="$3"

check_file "$IN"
[ -x "$BGREMOVE" ] || die "bgremove 未编译: $BGREMOVE (先跑 cpp/birefnet/build.sh)"
check_file "$MODEL"

if [[ "$OUT" == ~* ]]; then
    OUT="${HOME}${OUT:1}"
fi

echo -e "${GREEN}✓ All checks passed${NC}"
echo -e "${BLUE}[INFO] BiRefNet mode=${MODE}$([ "$MODE" = "color" ] && echo " color=${COLOR}")${NC}"
[ -n "$OUT" ] && echo -e "Output: ${GREEN}${OUT}${NC}" || echo -e "Output: ${GREEN}\$HOME/<timestamp>_nobg.png${NC}"

ARGS=(--model "$MODEL" --mode "$MODE")
[ "$MODE" = "color" ] && ARGS+=(--color "$COLOR")
[ -n "$MASK" ] && ARGS+=(--mask "$MASK")
[ "$CPU" = "1" ] && ARGS+=(--cpu)
ARGS+=("$IN")
[ -n "$OUT" ] && ARGS+=("$OUT")

START_TIME=$(date +%s)
RESULT="$("$BGREMOVE" "${ARGS[@]}")"
END_TIME=$(date +%s)

if [ -f "$RESULT" ]; then
    echo ""
    echo "========================================"
    echo -e "${GREEN}✓ Done!${NC}"
    echo -e "File:   ${GREEN}$RESULT${NC}"
    echo -e "Size:   ${BLUE}$(du -h "$RESULT" | cut -f1)${NC}"
    echo -e "Time:   ${YELLOW}$((END_TIME - START_TIME))s${NC}"
    [ -n "$MASK" ] && echo -e "Mask:   ${CYAN}$MASK${NC}"
    echo "========================================"
else
    die "输出文件未生成"
fi
