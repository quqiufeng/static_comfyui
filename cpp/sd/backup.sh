#!/bin/bash
# =============================================================================
# img_hires 封装脚本 — HiRes Fix 两阶段出图，VAE Tiling 显存自适应
# 用法: ./backup.sh "prompt" [output.png] [width] [height] [--flags...]
# 环境变量: VAE_TILE_SIZE, VAE_TILE_OVERLAP, CFG_SCALE, SAMPLING_METHOD 等
# =============================================================================
#
# 【调优参数（人像, 2026-07-12 验证）】
#   CFG=2.5  Sampler=euler  Scheduler=discrete  Steps=20→45
#   HiRes strength=0.35  FreeU b1=1.3 b2=1.4  SAG=关
#   规律: 人像 CFG 勿超 2.5; discrete 比 karras 稳; FreeU 降强度用
#
# 【VAE Tiling 峰值参考】
#   Tile    | VAE Buffer | 峰值估算  | 适用显卡
#   128×128 |  6.7 GB    | ~15.6 GB  | 20G+ (RTX 3080 Ti / 4060 Ti)
#   256×256 | 18.7 GB    | ~20.7 GB  | 24G  (RTX 4090 / 3090)
#   512×512 | 23.4 GB    | ~27.0 GB  | 32G+ (A100, 不推荐)
#   默认 128x128, 设 VAE_TILE_SIZE=256x256 切高性能模式
#
# 【示例】
#   20G 卡:  ./backup.sh "portrait" ~/out.png 2560 1440
#   24G 卡:  VAE_TILE_SIZE=256x256 ./backup.sh "portrait" ~/out.png 2560 1440
#   LoRA:    ./backup.sh "prompt" ~/out.png 2560 1440 --lora style.safetensors:0.8
# =============================================================================
set -euo pipefail

RED="\033[0;31m"; GREEN="\033[0;32m"; YELLOW="\033[1;33m"
BLUE="\033[0;34m"; CYAN="\033[0;36m"; NC="\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-/data/models/image}"
SD_CLI="${SD_CLI:-$SCRIPT_DIR/build/img_hires}"
DIFFUSION_MODEL="${DIFFUSION_MODEL:-$MODEL_DIR/z_image_turbo-Q5_K_M.gguf}"
VAE_MODEL="${VAE_MODEL:-$MODEL_DIR/ae.safetensors}"
LLM_MODEL="${LLM_MODEL:-$MODEL_DIR/Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
UPSCALE_MODEL="${UPSCALE_MODEL:-$MODEL_DIR/2x_ESRGAN.gguf}"
TARGET_RATIO=""

VAE_TILE_SIZE="${VAE_TILE_SIZE:-128x128}"
VAE_TILE_OVERLAP="${VAE_TILE_OVERLAP:-0.5}"

UPSCALE_FLAG=0; LORA_CONFIG=""; PROMPT_SCHEDULE=""; REGIONAL_PROMPTS=""
FACE_RESTORE_FLAG=0; FACE_RESTORE_MODEL=""
FACE_SWAP_FLAG=0; FACE_SWAP_SOURCE=""
IPADAPTER_FLAG=0; IPADAPTER_MODEL=""; IPADAPTER_IMAGE=""
T2I_ADAPTER_FLAG=0; T2I_ADAPTER_MODEL=""; T2I_ADAPTER_IMAGE=""
PHOTOMAKER_FLAG=0; PHOTOMAKER_MODEL=""; PHOTOMAKER_ID_IMAGES=""
ARGS=()

next_val() { i=$((i+1)); echo "${@:$((i+1)):1}"; }

i=0
while [ $i -lt $# ]; do
    arg="${@:$((i+1)):1}"
    case "$arg" in
        --upscale)          UPSCALE_FLAG=1 ;;
        --lora)             LORA_CONFIG=$(next_val "$@") ;;
        --prompt-schedule)  PROMPT_SCHEDULE=$(next_val "$@") ;;
        --regional-prompts) REGIONAL_PROMPTS=$(next_val "$@") ;;
        --face-restore)     FACE_RESTORE_FLAG=1 ;;
        --face-restore-model) FACE_RESTORE_MODEL=$(next_val "$@") ;;
        --face-swap)        FACE_SWAP_FLAG=1 ;;
        --face-swap-source) FACE_SWAP_SOURCE=$(next_val "$@") ;;
        --ipadapter)        IPADAPTER_FLAG=1 ;;
        --ipadapter-model)  IPADAPTER_MODEL=$(next_val "$@") ;;
        --ipadapter-image)  IPADAPTER_IMAGE=$(next_val "$@") ;;
        --t2i-adapter)      T2I_ADAPTER_FLAG=1 ;;
        --t2i-adapter-model) T2I_ADAPTER_MODEL=$(next_val "$@") ;;
        --t2i-adapter-image) T2I_ADAPTER_IMAGE=$(next_val "$@") ;;
        --photomaker)       PHOTOMAKER_FLAG=1 ;;
        --photomaker-model) PHOTOMAKER_MODEL=$(next_val "$@") ;;
        --photomaker-id-images) PHOTOMAKER_ID_IMAGES=$(next_val "$@") ;;
        *)                  ARGS+=("$arg") ;;
    esac
    i=$((i+1))
done

PROMPT="${ARGS[0]:-A beautiful landscape}"
OUTPUT_FILE="${ARGS[1]:-}"
WIDTH="${ARGS[2]:-1280}"
HEIGHT="${ARGS[3]:-720}"

if [[ "$OUTPUT_FILE" == ~* ]]; then
    OUTPUT_FILE="${HOME}${OUTPUT_FILE:1}"
fi

die() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }
check_file() { [ -f "$1" ] || die "not found: $1"; }

[ -f "$SD_CLI" ] || die "img_hires not found: $SD_CLI"
[ -x "$SD_CLI" ] || die "img_hires not executable: $SD_CLI"
check_file "$DIFFUSION_MODEL"
check_file "$VAE_MODEL"
check_file "$LLM_MODEL"

[[ "$WIDTH" =~ ^[0-9]+$ ]] && [ "$WIDTH" -gt 0 ] || die "width must be positive integer"
[[ "$HEIGHT" =~ ^[0-9]+$ ]] && [ "$HEIGHT" -gt 0 ] || die "height must be positive integer"

if [ "$UPSCALE_FLAG" -eq 1 ]; then
    check_file "$UPSCALE_MODEL"
    echo -e "${CYAN}✓ Upscale mode enabled (2x ESRGAN)${NC}"
fi

echo -e "${GREEN}✓ All checks passed${NC}"

SAMPLING_METHOD="${SAMPLING_METHOD:-euler}"
SCHEDULER="${SCHEDULER:-discrete}"
CFG_SCALE="${CFG_SCALE:-2.5}"
STEPS="${STEPS:-20}"
HIRES_STEPS="${HIRES_STEPS:-45}"
HIRES_STRENGTH="${HIRES_STRENGTH:-0.35}"

echo -e "${BLUE}[INFO] $([ "$WIDTH" -ge 1920 ] && echo "Ultra HD" || echo "HD") Mode: steps=$STEPS, cfg=$CFG_SCALE, sampler=$SAMPLING_METHOD${NC}"

QUALITY_PREFIX="masterpiece, best quality, ultra-detailed, sharp focus, 8k uhd, photorealistic, highly detailed, crisp, clear, centered composition, professional portrait, medium shot, realistic skin texture, soft lighting"
if [[ "$PROMPT" != *"masterpiece"* ]]; then
    PROMPT="$QUALITY_PREFIX, $PROMPT"
fi

NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, worst quality, jpeg artifacts, noise, grain, soft focus, out of focus, hazy, unclear, bad anatomy, deformed, border artifacts, edge distortion, tiling artifacts, edge artifacts, frame distortion, warped edges, stretched proportions, asymmetrical face, off-center, cropped, out of frame, partial face, cut off, incomplete head, cropped head, watermark, text, logo, signature, cropped shoulders, embedding:EasyNegative, embedding:bad-hands-5}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "$OUTPUT_FILE" ]; then
    if [[ "$OUTPUT_FILE" == *"/"* ]]; then
        OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
        BASE=$(basename "$OUTPUT_FILE")
    else
        OUTPUT_DIR="$HOME"
        BASE="$OUTPUT_FILE"
    fi
    OUTPUT="${BASE%.png}_${TIMESTAMP}.png"
else
    OUTPUT_DIR="$HOME"
    MD5=$(echo "$PROMPT" | md5sum | cut -c1-8)
    OUTPUT="${TIMESTAMP}_${MD5}.png"
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT"

TARGET_LATENT_W=$((WIDTH / 8))
TARGET_LATENT_H=$((HEIGHT / 8))

# HiRes Fix 两阶段: 先低分辨率构图 → latent 放大 refine
# 原则: 基础分辨率越高越好（放大倍数小 → 画质好）
# 已知分辨率对: 3840x2160→2560x1440, 2560x1440→1920x1080, 1920x1080→1536x864

if [ "$WIDTH" -eq 3840 ] && [ "$HEIGHT" -eq 2160 ]; then
    # 4K: 2560x1440 基础 → 1.5x 放大
    LOW_W=2560
    LOW_H=1440
elif [ "$WIDTH" -eq 2560 ] && [ "$HEIGHT" -eq 1440 ]; then
    # 2K: 1920x1080 基础 → 1.33x 放大（20G显存安全方案）
    LOW_W=1920
    LOW_H=1080
elif [ "$WIDTH" -eq 1920 ] && [ "$HEIGHT" -eq 1080 ]; then
    # 1080p: 1536x864 基础 → 1.25x 放大
    LOW_W=1536
    LOW_H=864
elif [ "$WIDTH" -eq 1280 ] && [ "$HEIGHT" -eq 720 ]; then
    # 720p: 1024x576 基础 → 1.25x 放大
    LOW_W=1024
    LOW_H=576
else
    # 通用计算：使用目标分辨率的 80% 作为基础（4090D优化，更小放大倍数）
    LOW_LATENT_W=$((TARGET_LATENT_W * 4 / 5))
    LOW_LATENT_H=$((TARGET_LATENT_H * 4 / 5))
    
    # 对齐到 8 的倍数
    LOW_LATENT_W=$(((LOW_LATENT_W + 7) / 8 * 8))
    LOW_LATENT_H=$(((LOW_LATENT_H + 7) / 8 * 8))
    
    LOW_W=$((LOW_LATENT_W * 8))
    LOW_H=$((LOW_LATENT_H * 8))
fi

# 保持比例的最小限制：只在单边小于512时按比例放大
if [ "$LOW_W" -lt 512 ] || [ "$LOW_H" -lt 512 ]; then
    TARGET_RATIO=$(echo "scale=6; $WIDTH / $HEIGHT" | bc)
    if [ "$LOW_W" -lt "$LOW_H" ]; then
        LOW_W=512
        LOW_H=$(echo "scale=0; $LOW_W / $TARGET_RATIO / 8 * 8" | bc)
        if [ "$LOW_H" -lt 512 ]; then LOW_H=512; fi
    else
        LOW_H=512
        LOW_W=$(echo "scale=0; $LOW_H * $TARGET_RATIO / 8 * 8" | bc)
        if [ "$LOW_W" -lt 512 ]; then LOW_W=512; fi
    fi
fi

echo ""
echo "========================================"
echo "  HD Image Generation"
echo "========================================"
echo -e "Target Size: ${GREEN}${WIDTH}x${HEIGHT}${NC}"
echo -e "Low-res Pass: ${GREEN}${LOW_W}x${LOW_H} -> ${WIDTH}x${HEIGHT}${NC}"
echo -e "Steps: $STEPS -> $HIRES_STEPS (HiRes)"
echo -e "CFG Scale: ${CYAN}$CFG_SCALE${NC}"
echo -e "HiRes Strength: $HIRES_STRENGTH"
echo -e "Sampler: ${CYAN}$SAMPLING_METHOD${NC} + ${CYAN}$SCHEDULER${NC}"
if [ "$UPSCALE_FLAG" -eq 1 ]; then
    UPSCALED_W=$((WIDTH * 2))
    UPSCALED_H=$((HEIGHT * 2))
    echo -e "Upscale: ${CYAN}2x ESRGAN -> ${UPSCALED_W}x${UPSCALED_H}${NC}"
fi
echo "----------------------------------------"
echo -e "Prompt: ${YELLOW}$PROMPT${NC}"
echo -e "Output: ${GREEN}$OUTPUT_PATH${NC}"
echo "========================================"
echo ""

SEED="${SEED:-$RANDOM}"
echo "Generating...  $(date '+%H:%M:%S')"

# Convert VAE_TILE_SIZE "128x128" -> single int for img_hires
VAE_TILE_INT="${VAE_TILE_SIZE%x*}"
if ! [[ "$VAE_TILE_INT" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: VAE_TILE_SIZE must be like 128x128 or 128${NC}"
    exit 1
fi

SD_CMD=("$SD_CLI"
  --diffusion-model "$DIFFUSION_MODEL"
  --vae "$VAE_MODEL"
  --llm "$LLM_MODEL"
  --negative "$NEGATIVE_PROMPT"
  --cfg "$CFG_SCALE"
  --method "$SAMPLING_METHOD"
  --scheduler "$SCHEDULER"
  --diffusion-fa
  --vae-tiling
  --vae-tile-size "$VAE_TILE_INT"
  --vae-tile-overlap "$VAE_TILE_OVERLAP"
  --freeu
  --freeu-b1 1.3
  --freeu-b2 1.4
  --clarity 0.2
  --sharpen 0.3
  --sharpen-radius 1
  --smart-sharpen 0.5
  --smart-sharpen-radius 2
  --edge-sharpen 1.5
  --edge-sharpen-radius 2
  --edge-sharpen-threshold 0.3
  -W "$LOW_W" -H "$LOW_H"
  --steps "$STEPS"
  --hires
  --hires-width "$WIDTH"
  --hires-height "$HEIGHT"
  --hires-strength "$HIRES_STRENGTH"
  --hires-steps "$HIRES_STEPS"
  -s "$SEED"
  "$PROMPT"
  "$OUTPUT_PATH"
)

if [ -n "$LORA_CONFIG" ]; then
    SD_CMD+=(--lora "$LORA_CONFIG")
fi

if [ -n "$PROMPT_SCHEDULE" ]; then
    SD_CMD+=(--prompt-schedule "$PROMPT_SCHEDULE")
fi

if [ -n "$REGIONAL_PROMPTS" ]; then
    SD_CMD+=(--regional-prompts "$REGIONAL_PROMPTS")
fi

if [ "$FACE_RESTORE_FLAG" -eq 1 ]; then
    SD_CMD+=(--face-restore)
    if [ -n "$FACE_RESTORE_MODEL" ]; then
        SD_CMD+=(--face-restore-model "$FACE_RESTORE_MODEL")
    fi
fi

if [ "$FACE_SWAP_FLAG" -eq 1 ] && [ -n "$FACE_SWAP_SOURCE" ]; then
    SD_CMD+=(--face-swap --face-swap-source "$FACE_SWAP_SOURCE")
    SD_CMD+=(--face-swap-detection-model "$MODEL_DIR/yunet_320_320.onnx")
    SD_CMD+=(--face-swap-model "$MODEL_DIR/inswapper_128.onnx")
fi

if [ "$IPADAPTER_FLAG" -eq 1 ] && [ -n "$IPADAPTER_MODEL" ] && [ -n "$IPADAPTER_IMAGE" ]; then
    SD_CMD+=(--ipadapter --ipadapter-model "$IPADAPTER_MODEL")
    SD_CMD+=(--ipadapter-clip-vision "$MODEL_DIR/clip_vision_sd15.safetensors")
    SD_CMD+=(--ipadapter-image "$IPADAPTER_IMAGE")
fi

if [ "$T2I_ADAPTER_FLAG" -eq 1 ] && [ -n "$T2I_ADAPTER_MODEL" ] && [ -n "$T2I_ADAPTER_IMAGE" ]; then
    SD_CMD+=(--t2i-adapter --t2i-adapter-model "$T2I_ADAPTER_MODEL")
    SD_CMD+=(--t2i-adapter-image "$T2I_ADAPTER_IMAGE")
fi

if [ "$PHOTOMAKER_FLAG" -eq 1 ] && [ -n "$PHOTOMAKER_MODEL" ] && [ -n "$PHOTOMAKER_ID_IMAGES" ]; then
    SD_CMD+=(--photomaker --photomaker-model "$PHOTOMAKER_MODEL")
    SD_CMD+=(--photomaker-id-images "$PHOTOMAKER_ID_IMAGES")
fi

if [ "$UPSCALE_FLAG" -eq 1 ]; then
    SD_CMD+=(--upscale-model "$UPSCALE_MODEL")
    SD_CMD+=(--upscale-repeats 1)
    SD_CMD+=(--upscale-tile-size 1440)
fi

START_TIME=$(date +%s)
"${SD_CMD[@]}"
END_TIME=$(date +%s)
GEN_DURATION=$((END_TIME - START_TIME))

fmt_duration() {
    local s=$1
    [ $s -ge 60 ] && echo "$((s/60))m $((s%60))s" || echo "${s}s"
}

if [ -f "$OUTPUT_PATH" ]; then
    echo ""
    echo "========================================"
    echo -e "${GREEN}✓ Generation successful!${NC}"
    echo -e "File:   ${GREEN}$OUTPUT_PATH${NC}"
    echo -e "Size:   ${BLUE}$(du -h "$OUTPUT_PATH" | cut -f1)${NC}"
    echo -e "Time:   ${YELLOW}$(fmt_duration $GEN_DURATION)${NC}"
    echo -e "Seed:   ${YELLOW}$SEED${NC}"
    echo -e "CFG:    ${CYAN}$CFG_SCALE${NC}"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo -e "${RED}✗ Generation failed! Output file not found${NC}"
    echo "========================================"
    exit 1
fi
