#!/bin/bash
# =============================================================================
# backup_qwen.sh — Qwen-Image-2.1 HiRes Fix 两阶段出图（GPU 计算 + 权重 offload 到内存）
# 用法: ./backup_qwen.sh "prompt" [output.png] [width] [height] [--flags...]
# 环境变量: CFG, STEPS, HIRES_STEPS, HIRES_STRENGTH, SAMPLING_METHOD, SCHEDULER,
#           VAE_TILE_SIZE, VAE_TILE_OVERLAP, OFFLOAD, MODEL_DIR 等
# =============================================================================
#
# 【为什么需要 --offload-to-cpu】
#   Qwen-Image-2.1 比 z_image 重得多：文本编码器 Qwen3-VL-8B(~4.3GB) +
#   diffusion(~5GB) + 3D VAE(wan_vae)。20GB 卡上权重常驻后，VAE 解码
#   （1024 约需 ~10GB，随面积增长）放不下 → OOM。
#   --offload-to-cpu 让权重常驻内存、按需上卡；**采样与 VAE 解码仍在 GPU**，
#   速度几乎无损（sd.cpp 官方 Qwen-Image 文档即如此推荐）。
#
# 【分辨率】Qwen-Image 要求宽高为 32 的倍数；脚本已按 /32 计算 base。
#   已知分辨率对（target → base，均为 32 的倍数）：
#     3840x2160 → 2560x1440 | 2560x1440 → 2048x1152
#     1920x1080 → 1536x864  | 1280x720  → 1024x576
#
# 【参数】CFG=6.0  Sampler=euler  Scheduler=discrete  Steps=20→20
#   HiRes strength=0.35  VAE tiling 32/0.5  后处理 clarity0.2/sharpen0.3/smart0.5/edge1.5
#
# 【示例】
#   ./backup_qwen.sh "a lovely cat holding a sign that says hello" ~/cat.png 1024 1024
#   ./backup_qwen.sh "portrait" ~/p.png 2560 1440
# =============================================================================
set -euo pipefail

RED="\033[0;31m"; GREEN="\033[0;32m"; YELLOW="\033[1;33m"
BLUE="\033[0;34m"; CYAN="\033[0;36m"; NC="\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-/data/models/image}"
SD_CLI="${SD_CLI:-$SCRIPT_DIR/build/img_hires}"
SD_BACKEND_DIR="${SD_BACKEND_DIR:-/opt/sd/build-dl/bin}"

DIFFUSION_MODEL="${DIFFUSION_MODEL:-$MODEL_DIR/qwen-image-2.1-Q5_K_M.gguf}"
LLM_MODEL="${LLM_MODEL:-$MODEL_DIR/Qwen3VL-8B-Instruct-Q4_K_M.gguf}"
VAE_MODEL="${VAE_MODEL:-$MODEL_DIR/qwen_image_2.1_vae_bf16.safetensors}"

# 共享库搜索路径（libsdcpp_adapter.so + sd.cpp/ggml）
export LD_LIBRARY_PATH="$SCRIPT_DIR/build:$SD_BACKEND_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ARGS=()
for arg in "$@"; do
    ARGS+=("$arg")
done

PROMPT="${ARGS[0]:-a lovely cat holding a sign that says hello}"
OUTPUT_FILE="${ARGS[1]:-}"
WIDTH="${ARGS[2]:-1024}"
HEIGHT="${ARGS[3]:-1024}"

if [[ "$OUTPUT_FILE" == ~* ]]; then
    OUTPUT_FILE="${HOME}${OUTPUT_FILE:1}"
fi

die() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }
check_file() { [ -f "$1" ] || die "not found: $1"; }

[ -x "$SD_CLI" ] || die "img_hires not found: $SD_CLI"
check_file "$DIFFUSION_MODEL"
check_file "$LLM_MODEL"
check_file "$VAE_MODEL"

[[ "$WIDTH" =~ ^[0-9]+$ ]] && [ "$WIDTH" -gt 0 ] || die "width must be positive integer"
[[ "$HEIGHT" =~ ^[0-9]+$ ]] && [ "$HEIGHT" -gt 0 ] || die "height must be positive integer"

echo -e "${GREEN}✓ All checks passed${NC}"

CFG_SCALE="${CFG:-6.0}"
STEPS="${STEPS:-20}"
HIRES_STEPS="${HIRES_STEPS:-20}"
HIRES_STRENGTH="${HIRES_STRENGTH:-0.35}"
SAMPLING_METHOD="${SAMPLING_METHOD:-euler}"
SCHEDULER="${SCHEDULER:-discrete}"
VAE_TILE_SIZE="${VAE_TILE_SIZE:-32}"
VAE_TILE_OVERLAP="${VAE_TILE_OVERLAP:-0.5}"
OFFLOAD="${OFFLOAD:-1}"

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
# 转绝对路径：下面会在后端目录下执行（以便自动发现 CPU/CUDA 后端）
OUTPUT_PATH="$(cd "$OUTPUT_DIR" && pwd)/$OUTPUT"

# ---- base 分辨率：Qwen 要求 /32 的倍数 ----
round32() { echo $(( ( ($1) / 32 ) * 32 )); }
if [ "$WIDTH" -eq 3840 ] && [ "$HEIGHT" -eq 2160 ]; then
    LOW_W=2560; LOW_H=1440
elif [ "$WIDTH" -eq 2560 ] && [ "$HEIGHT" -eq 1440 ]; then
    LOW_W=2048; LOW_H=1152
elif [ "$WIDTH" -eq 1920 ] && [ "$HEIGHT" -eq 1080 ]; then
    LOW_W=1536; LOW_H=864
elif [ "$WIDTH" -eq 1280 ] && [ "$HEIGHT" -eq 720 ]; then
    LOW_W=1024; LOW_H=576
else
    # 目标 80%，向下取整到 /32
    LOW_W=$(round32 $(( WIDTH * 4 / 5 )))
    LOW_H=$(round32 $(( HEIGHT * 4 / 5 )))
    [ "$LOW_W" -lt 512 ] && LOW_W=512
    [ "$LOW_H" -lt 512 ] && LOW_H=512
fi

echo ""
echo "========================================"
echo "  Qwen-Image-2.1 HiRes Fix"
echo "========================================"
echo -e "Target Size: ${GREEN}${WIDTH}x${HEIGHT}${NC}"
echo -e "Low-res Pass: ${GREEN}${LOW_W}x${LOW_H} -> ${WIDTH}x${HEIGHT}${NC}"
echo -e "Steps: $STEPS -> $HIRES_STEPS (HiRes)"
echo -e "CFG Scale: ${CYAN}$CFG_SCALE${NC}"
echo -e "HiRes Strength: $HIRES_STRENGTH"
echo -e "Sampler: ${CYAN}$SAMPLING_METHOD${NC} + ${CYAN}$SCHEDULER${NC}"
echo -e "VAE Tiling: ${VAE_TILE_SIZE} overlap ${VAE_TILE_OVERLAP}"
echo -e "Offload to CPU (weights in RAM, compute on GPU): ${OFFLOAD}"
echo "----------------------------------------"
echo -e "Prompt: ${YELLOW}$PROMPT${NC}"
echo -e "Output: ${GREEN}$OUTPUT_PATH${NC}"
echo "========================================"
echo ""

SEED="${SEED:-$RANDOM}"
echo "Generating...  $(date '+%H:%M:%S')"

SD_CMD=("$SD_CLI"
  --diffusion-model "$DIFFUSION_MODEL"
  --llm "$LLM_MODEL"
  --vae "$VAE_MODEL"
  --negative "$NEGATIVE_PROMPT"
  --cfg "$CFG_SCALE"
  --method "$SAMPLING_METHOD"
  --scheduler "$SCHEDULER"
  --diffusion-fa
  --vae-tiling
  --vae-tile-size "$VAE_TILE_SIZE"
  --vae-tile-overlap "$VAE_TILE_OVERLAP"
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
)

if [ "$OFFLOAD" -eq 1 ]; then
    SD_CMD+=(--offload-to-cpu)
fi

SD_CMD+=("$PROMPT" "$OUTPUT_PATH")

START_TIME=$(date +%s)
# 在后端目录执行：ggml 从「可执行文件目录/当前目录」发现后端，
# 这样 libggml-cpu.so（--offload-to-cpu 需要）与 libggml-cuda.so 都能加载。
( cd "$SD_BACKEND_DIR" && "${SD_CMD[@]}" )
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
    echo -e "${RED}✗ Generation failed! Output file not found${NC}"
    exit 1
fi
