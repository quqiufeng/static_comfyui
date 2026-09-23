#!/bin/bash
# =============================================================================
# backup_qwen2.sh — Qwen-Image-2.1 HiRes 两阶段出图（Qwen 原生配方 v2）
# 用法: ./backup_qwen2.sh "prompt" [output.png] [width] [height]
# 环境变量: CFG, STEPS, HIRES_STEPS, HIRES_STRENGTH, HIRES_UPSCALER,
#           SAMPLING_METHOD, SCHEDULER, VAE_TILE_SIZE, VAE_TILE_OVERLAP,
#           OFFLOAD, POSTPROC, CLARITY, SHARPEN, SMART_SHARPEN, EDGE_SHARPEN,
#           FREEU, REALISM, MODEL_DIR
# =============================================================================
#
# 【v2 相对 backup_qwen.sh 的修正】
#   1) scheduler: discrete → flux（sd.cpp 对 VERSION_QWEN_IMAGE_2_1 的默认调度器
#      就是 FLUX_SCHEDULER；官方示例不传 --scheduler。强制 discrete 会明显掉画质）
#   2) 脚本层去掉 SD1.5 味的 quality prefix（"masterpiece, best quality, 8k uhd,
#      professional portrait, medium shot"）。注意: img_hires 二进制内置同文前缀
#      默认仍会自动加（历次 Qwen 验证图均带此前提）; NO_QUALITY_PREFIX=1 透传
#      --no-quality-prefix 才能真正关闭
#   3) 去掉 negative 里的 SD1.5 textual inversion（embedding:EasyNegative /
#      embedding:bad-hands-5）——Qwen3-VL 文本编码器没有这些 embedding
#   4) 默认关闭重后处理（clarity/sharpen/smart/edge）——避免过锐、塑料感
#   5) steps 提到 25→20，cfg 6.0 / euler（与官方 qwen_image_2.1.md 一致）
#
# 【v3 人像写实固化（2026-09-22 combo B）】
#   - cfg 6.0 / euler / scheduler flux / steps 25→50 / hires strength 0.5
#   - HiRes 放大器 latent-bislerp（更锐），base 2048x1152 → 2560x1440
#   - 后处理默认开启：clarity 0.3 / sharpen 0.3 / smart 0.5 / edge 2.0
#   - 正向自动追加写实词（REALISM=0 关）；负向加 anime/cartoon/illustration/
#     3d render 等反动漫词 + 皮肤油腻词
#   - FreeU 默认关（Qwen 为 DiT，FreeU 空操作；FREEU=1 可开）
#   - POSTPROC=0 关闭后处理；OFFLOAD=1 权重常驻内存（20G 卡必需）
#
# 【v4 甜点同步（2026-09-23, 与 backup.sh 同步 z_image 14 组扫描结论）】
#   用法与 backup.sh 一致: 只提供提示词（+ 输出路径、分辨率）, 其余全默认裸跑复现。
#   甜点值: steps 20→40 / hires strength 0.4 / clarity 0.15 / edge-sharpen 0.0
#           upscaler latent-bislerp（不变）/ SEED 默认时间戳随机（复现用 SEED=25630）
#           / 无质量前缀 / 皮肤词负面已固化
#   cfg 保持 6.0 — z_image 甜点 3.0 不适用 Qwen（两模型引导尺度不同,
#      Qwen 官方/已验证值即 6.0; 想试 CFG=3.0 需单独扫）
#   参数范围与原理、14 张对比调试经验见 backup.sh 头部注释（同步维护）:
#     - edge-sharpen 必须 0: 白底剪影高通锐化出白边/振铃 halo
#     - clarity 0.15~0.3: 0 皮肤死平, 0.8 塑料过锐
#     - strength 0.35~0.5: 0.25 无纹理, 0.75 跑构图
#     - 蒸馏 turbo 步数收益低: 40→80 更慢更差, 甜点 20→40
#     - FreeU 对 Qwen(DiT) 空操作; 质量前缀污染自然语言语义（v2 已去）
#
# 【分辨率】Qwen 要求宽高为 32 的倍数；脚本按 /32 计算 base。
# 【显存】20GB 卡必须 --offload-to-cpu（权重常驻内存、采样/VAE 仍在 GPU）。
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

# 运行环境依赖内聚到脚本内, 外部无需再 export
export LD_LIBRARY_PATH="$SCRIPT_DIR/build:$SD_BACKEND_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GGML_BACKEND_PATH="${GGML_BACKEND_PATH:-$SD_BACKEND_DIR/libggml-cuda.so}"

ARGS=()
for arg in "$@"; do ARGS+=("$arg"); done

PROMPT="${ARGS[0]:-solo,single woman,half body portrait of a young woman, soft natural lighting, elegant pose, studio lighting, sharp eyes, pure white background, fair skin, pale skin, smooth skin, matte skin, porcelain skin, flawless skin, medium close up}"
OUTPUT_FILE="${ARGS[1]:-}"
WIDTH="${ARGS[2]:-1024}"
HEIGHT="${ARGS[3]:-1024}"

if [[ "$OUTPUT_FILE" == ~* ]]; then OUTPUT_FILE="${HOME}${OUTPUT_FILE:1}"; fi

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
HIRES_STEPS="${HIRES_STEPS:-40}"
HIRES_STRENGTH="${HIRES_STRENGTH:-0.4}"
SAMPLING_METHOD="${SAMPLING_METHOD:-euler}"
SCHEDULER="${SCHEDULER:-flux}"
HIRES_UPSCALER="${HIRES_UPSCALER:-latent-bislerp}"
VAE_TILE_SIZE="${VAE_TILE_SIZE:-32}"
VAE_TILE_OVERLAP="${VAE_TILE_OVERLAP:-0.5}"
OFFLOAD="${OFFLOAD:-1}"
POSTPROC="${POSTPROC:-1}"
CLARITY="${CLARITY:-0.15}"
SHARPEN="${SHARPEN:-0.3}"
SMART_SHARPEN="${SMART_SHARPEN:-0.5}"
EDGE_SHARPEN="${EDGE_SHARPEN:-0.0}"
FREEU="${FREEU:-0}"
REALISM="${REALISM:-1}"
NO_QUALITY_PREFIX="${NO_QUALITY_PREFIX:-0}"
REALISM_SUFFIX="photorealistic, realistic photograph, raw photo, natural skin texture"

# v2：不再自动加 booru quality prefix（需要时可用 QUALITY_PREFIX 显式指定）
if [ -n "${QUALITY_PREFIX:-}" ] && [[ "$PROMPT" != *"masterpiece"* ]]; then
    PROMPT="$QUALITY_PREFIX, $PROMPT"
fi

# v3：写实约束（Qwen 默认偏动漫，追加写实关键词；REALISM=0 关闭）
if [ "$REALISM" = "1" ] && [[ "$PROMPT" != *"photorealistic"* ]]; then
    PROMPT="$PROMPT, $REALISM_SUFFIX"
fi

NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, worst quality, jpeg artifacts, noise, bad anatomy, deformed, watermark, text, logo, signature, oily skin, shiny skin, greasy skin, glossy skin, plastic skin, skin blemishes, anime, cartoon, illustration, painting, drawing, 3d render, cgi, anime face, cel shading}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "$OUTPUT_FILE" ]; then
    if [[ "$OUTPUT_FILE" == *"/"* ]]; then
        OUTPUT_DIR=$(dirname "$OUTPUT_FILE"); BASE=$(basename "$OUTPUT_FILE")
    else
        OUTPUT_DIR="$HOME"; BASE="$OUTPUT_FILE"
    fi
    OUTPUT="${BASE%.png}_${TIMESTAMP}.png"
else
    OUTPUT_DIR="$HOME"
    MD5=$(echo "$PROMPT" | md5sum | cut -c1-8)
    OUTPUT="${TIMESTAMP}_${MD5}.png"
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_PATH="$(cd "$OUTPUT_DIR" && pwd)/$OUTPUT"

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
    LOW_W=$(round32 $(( WIDTH * 4 / 5 )))
    LOW_H=$(round32 $(( HEIGHT * 4 / 5 )))
    [ "$LOW_W" -lt 512 ] && LOW_W=512
    [ "$LOW_H" -lt 512 ] && LOW_H=512
fi

echo ""
echo "========================================"
echo "  Qwen-Image-2.1 HiRes Fix (native v2)"
echo "========================================"
echo -e "Target Size: ${GREEN}${WIDTH}x${HEIGHT}${NC}"
echo -e "Low-res Pass: ${GREEN}${LOW_W}x${LOW_H} -> ${WIDTH}x${HEIGHT}${NC}"
echo -e "Steps: $STEPS -> $HIRES_STEPS (HiRes)"
echo -e "CFG Scale: ${CYAN}$CFG_SCALE${NC}"
echo -e "HiRes Strength: $HIRES_STRENGTH"
echo -e "HiRes Upscaler: ${CYAN}$HIRES_UPSCALER${NC}"
echo -e "Sampler: ${CYAN}$SAMPLING_METHOD${NC} + ${CYAN}$SCHEDULER${NC}"
echo -e "VAE Tiling: ${VAE_TILE_SIZE} overlap ${VAE_TILE_OVERLAP}"
echo -e "Post-processing: ${POSTPROC} (0=off), realism=${REALISM}"
echo -e "FreeU: ${FREEU} (DiT 空操作)"
echo -e "Offload to CPU: ${OFFLOAD}"
echo "----------------------------------------"
echo -e "Prompt: ${YELLOW}$PROMPT${NC}"
echo -e "Output: ${GREEN}$OUTPUT_PATH${NC}"
echo "========================================"
echo ""

SEED="${SEED:-$(date +%s)}"
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
  -W "$LOW_W" -H "$LOW_H"
  --steps "$STEPS"
  --hires
  --hires-width "$WIDTH"
  --hires-height "$HEIGHT"
  --hires-strength "$HIRES_STRENGTH"
  --hires-steps "$HIRES_STEPS"
  --hires-upscaler "$HIRES_UPSCALER"
  -s "$SEED"
)

if [ "$POSTPROC" -eq 1 ]; then
    SD_CMD+=(--clarity "$CLARITY" --sharpen "$SHARPEN" --sharpen-radius 1
             --smart-sharpen "$SMART_SHARPEN" --smart-sharpen-radius 2
             --edge-sharpen "$EDGE_SHARPEN" --edge-sharpen-radius 2
             --edge-sharpen-threshold 0.3)
else
    SD_CMD+=(--clarity 0 --sharpen 0 --smart-sharpen 0 --edge-sharpen 0)
fi
if [ "$FREEU" -eq 1 ]; then
    SD_CMD+=(--freeu --freeu-b1 1.3 --freeu-b2 1.4)
fi
if [ "$OFFLOAD" -eq 1 ]; then
  SD_CMD+=(--offload-to-cpu)
fi
if [ "$NO_QUALITY_PREFIX" -eq 1 ]; then
  SD_CMD+=(--no-quality-prefix)
fi

SD_CMD+=("$PROMPT" "$OUTPUT_PATH")

START_TIME=$(date +%s)
( cd "$SD_BACKEND_DIR" && "${SD_CMD[@]}" )
END_TIME=$(date +%s)
GEN_DURATION=$((END_TIME - START_TIME))

fmt_duration() { local s=$1; [ $s -ge 60 ] && echo "$((s/60))m $((s%60))s" || echo "${s}s"; }

if [ -f "$OUTPUT_PATH" ]; then
    echo ""
    echo "========================================"
    echo -e "${GREEN}✓ Generation successful!${NC}"
    echo -e "File:   ${GREEN}$OUTPUT_PATH${NC}"
    echo -e "Size:   ${BLUE}$(du -h "$OUTPUT_PATH" | cut -f1)${NC}"
    echo -e "Time:   ${YELLOW}$(fmt_duration $GEN_DURATION)${NC}"
    echo -e "Seed:   ${YELLOW}$SEED${NC}"
    echo "========================================"
else
    echo ""
    echo -e "${RED}✗ Generation failed! Output file not found${NC}"
    exit 1
fi
