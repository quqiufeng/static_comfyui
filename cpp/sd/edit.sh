#!/bin/bash
# =============================================================================
# edit.sh - Qwen-Image-2.1 指令图片编辑专用脚本 (sd-cli -r 参考图路线)
# =============================================================================
# 用法:
#   bash edit.sh <输入图> <编辑指令> <输出图> [宽] [高]
#
#   bash edit.sh in.jpg "change the background to a sunset beach" out.png
#   bash edit.sh in.jpg "keep everything unchanged" small.png        # 自动缩小
#   bash edit.sh in.jpg "change dress to red gown" out.png 768 1024  # 指定分辨率
#
# 宽高省略时: 按输入图宽高比, 最长边 <= MAX_SIDE(默认1024), 宽高 32 整除
#             (输入比 MAX_SIDE 小时按原比例输出, 不放大)
#
# 加速 (默认全开):
#   FA=0       关 Flash Attention (--diffusion-fa)
#   CACHE=0    关 EasyCache 步缓存 (--cache-mode easycache)
#   SAGE=1     额外启用 SageAttention (--sage-attn)
# 可调参数:
#   STEPS=20 CFG=6.0 METHOD=euler SEED=<int> MAX_SIDE=1024
#
# 模型 (与 backup_qwen.sh 同一套, 路径可用环境变量覆盖):
#   qwen-image-2.1-Q6_K.gguf + qwen_image_2.1_vae_bf16.safetensors
#   Qwen3VL-8B-Instruct-Q4_K_M.gguf + mmproj-Qwen3VL-8B-Instruct-F16.gguf (视觉, 编辑必需)
#
# 远程布局 (同路线 B): 脚本与 sd-cli/依赖同放, 模型在 $MODEL_DIR
#   SD_CLI 未指定时依次找: $SCRIPT_DIR/sd-cli, $HOME/sdcli/sd-cli, $HOME/build/sd-cli
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[edit]${NC} $*"; }
warn()  { echo -e "${YELLOW}[edit]${NC} $*"; }
die()   { echo -e "${RED}[edit] error: $*${NC}" >&2; exit 1; }

[ $# -ge 3 ] || die "用法: bash edit.sh <输入图> <编辑指令> <输出图> [宽] [高]"

INPUT="$1"; PROMPT="$2"; OUTPUT="$3"
W="${4:-}"; H="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-/data/models/image}"
DIFFUSION_MODEL="${DIFFUSION_MODEL:-$MODEL_DIR/qwen-image-2.1-Q6_K.gguf}"
VAE_MODEL="${VAE_MODEL:-$MODEL_DIR/qwen_image_2.1_vae_bf16.safetensors}"
LLM_MODEL="${LLM_MODEL:-$MODEL_DIR/Qwen3VL-8B-Instruct-Q4_K_M.gguf}"
LLM_VISION="${LLM_VISION:-$MODEL_DIR/mmproj-Qwen3VL-8B-Instruct-F16.gguf}"

STEPS="${STEPS:-20}"
CFG="${CFG:-6.0}"
METHOD="${METHOD:-euler}"
MAX_SIDE="${MAX_SIDE:-1024}"
FA="${FA:-1}"
CACHE="${CACHE:-1}"
SAGE="${SAGE:-0}"

# ---- sd-cli 定位 ----
SD_CLI="${SD_CLI:-}"
if [ -z "$SD_CLI" ]; then
    for c in "$SCRIPT_DIR/sd-cli" "$HOME/sdcli/sd-cli" "$HOME/build/sd-cli" "$SCRIPT_DIR/build/sd-cli"; do
        [ -x "$c" ] && SD_CLI="$c" && break
    done
fi
[ -n "$SD_CLI" ] && [ -x "$SD_CLI" ] || die "找不到 sd-cli, 用 SD_CLI=/path/to/sd-cli 指定"
SD_BIN_DIR="$(cd "$(dirname "$SD_CLI")" && pwd)"
export LD_LIBRARY_PATH="$SD_BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ -f "$SD_BIN_DIR/libggml-cuda.so" ] && export GGML_BACKEND_PATH="$SD_BIN_DIR/libggml-cuda.so"

# ---- 输入检查 ----
[ -f "$INPUT" ] || die "输入图不存在: $INPUT"
for f in "$DIFFUSION_MODEL" "$VAE_MODEL" "$LLM_MODEL" "$LLM_VISION"; do
    [ -f "$f" ] || die "模型缺失: $f"
done
mkdir -p "$(dirname "$OUTPUT")"

# ---- 宽高: 显式优先, 否则按输入比例自动 (32 整除) ----
if [ -z "$W" ] || [ -z "$H" ]; then
    AUTO_WH="$(python3 -c "
from PIL import Image
w, h = Image.open('$INPUT').size
s = min($MAX_SIDE / max(w, h), 1.0)
print(max(round(w * s / 32) * 32, 32), max(round(h * s / 32) * 32, 32))
" 2>/dev/null || true)"
    if [ -z "$AUTO_WH" ]; then
        warn "PIL 解析尺寸失败, 回退 768x1024"
        AUTO_WH="768 1024"
    fi
    set -- $AUTO_WH; W="$1"; H="$2"
    info "输出尺寸 ${W}x${H} (按输入比例, 最长边 <=${MAX_SIDE})"
fi

# ---- 加速选项 ----
ACCEL=()
[ "$FA" = 1 ]    && ACCEL+=(--diffusion-fa)
[ "$CACHE" = 1 ] && ACCEL+=(--cache-mode easycache)
[ "$SAGE" = 1 ]  && ACCEL+=(--sage-attn)

info "编辑: $INPUT -> $OUTPUT  ${W}x${H} steps=$STEPS cfg=$CFG method=$METHOD"
info "指令: $PROMPT"

cd "$SD_BIN_DIR"
START=$(date +%s)
"$SD_CLI" \
    --diffusion-model "$DIFFUSION_MODEL" \
    --vae "$VAE_MODEL" \
    --llm "$LLM_MODEL" \
    --llm_vision "$LLM_VISION" \
    -r "$INPUT" \
    -p "$PROMPT" \
    -W "$W" -H "$H" \
    --steps "$STEPS" \
    --cfg-scale "$CFG" \
    --sampling-method "$METHOD" \
    ${SEED:+-s "$SEED"} \
    "${ACCEL[@]}" \
    -o "$OUTPUT" -v
END=$(date +%s)
info "完成: $OUTPUT  用时 $((END - START))s"
