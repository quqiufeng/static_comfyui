#!/bin/bash
# =============================================================================
# img2img.sh — 给一张参考图，自动出提示词并生成新图（backup.sh 的参考图版本）
#
# 流程: 参考图 --(Florence-2 图像→提示词)--> PROMPT --(backup.sh)--> 新图
#
# 用法: ./img2img.sh <参考图> [output.png] [width] [height] [backup.sh 的 flag...]
#       ./img2img.sh ref.jpg                        # 默认 2560x1440, 存到 $HOME
#       ./img2img.sh ref.jpg out.png                # 默认 2560x1440
#       ./img2img.sh ref.jpg out.png 1280 768       # 指定输出与尺寸
#       ./img2img.sh ref.jpg 1920 1080              # 默认存 $HOME, 指定尺寸
#       ./img2img.sh ref.jpg out.png 2560 1440 --lora style.safetensors:0.8
#
# 出图完全委托 backup.sh（配方/后处理/加速单处维护），参考图之外的位置参数与
# flag 原样透传；CFG_SCALE/STEPS/HIRES_STRENGTH/SEED/... 等环境变量同样透传。
#
# 本脚本专有环境变量:
#   TASK    Florence-2 任务 (默认 <MORE_DETAILED_CAPTION>)
#           <CAPTION> <DETAILED_CAPTION> <MORE_DETAILED_CAPTION> <OCR>
#   BEAMS   beam search 宽度 (默认 3)
#   GEN=0   只输出提示词，不出图
#
# 说明: 参考图只用于生成提示词，不保留构图/人物身份（非真·img2img）。
#       要保留构图/身份请用 Qwen 参考重绘（cpp/sd/edit.sh，路线 C）。
# =============================================================================
set -euo pipefail

RED="\033[0;31m"; GREEN="\033[0;32m"; YELLOW="\033[1;33m"
BLUE="\033[0;34m"; CYAN="\033[0;36m"; NC="\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG2PROMPT="${IMG2PROMPT:-$SCRIPT_DIR/../florence2/img2prompt}"
MODEL_DIR="${MODEL_DIR:-/data/models/florence2}"
BACKUP="${BACKUP:-$SCRIPT_DIR/backup.sh}"
TASK="${TASK:-<MORE_DETAILED_CAPTION>}"
BEAMS="${BEAMS:-3}"
GEN="${GEN:-1}"

die() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }
check_file() { [ -f "$1" ] || die "not found: $1"; }

if [ "$#" -lt 1 ]; then
    die "用法: ./img2img.sh <参考图> [output.png] [width] [height] [flags...]"
fi

REF="$1"; shift   # 参考图固定为第一个参数，其余 (output/宽高/flags) 透传 backup.sh

check_file "$REF"
[ -x "$IMG2PROMPT" ] || die "img2prompt 未编译: $IMG2PROMPT (先跑 cpp/florence2/build.sh)"
[ -x "$BACKUP" ] || die "backup.sh 不存在: $BACKUP"

echo -e "${GREEN}✓ All checks passed${NC}"

# ---- 图 → 词 ----
echo -e "${BLUE}[INFO] Florence-2: ${TASK} (beams=$BEAMS)${NC}"
PROMPT="$("$IMG2PROMPT" --model-dir "$MODEL_DIR" --task "$TASK" --beams "$BEAMS" "$REF")"

echo ""
echo "========================================"
echo "  Reference -> Prompt"
echo "========================================"
echo -e "Reference: ${CYAN}$REF${NC}"
echo -e "Prompt:    ${YELLOW}$PROMPT${NC}"
echo "========================================"
echo ""

if [ "$GEN" = "0" ]; then
    echo -e "${YELLOW}(GEN=0，仅输出提示词)${NC}"
    exit 0
fi

# ---- 出图：委托 backup.sh；无输出名时 backup.sh 默认存 $HOME 带时间戳 ----
bash "$BACKUP" "$PROMPT" "$@"
