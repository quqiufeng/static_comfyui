#!/bin/bash
# =============================================================================
# img3.sh - Checkpoint model + HiRes Fix wrapper
# =============================================================================
# Usage: ./img3.sh "prompt" ~/output.png 2560 1440
#
# Environment variables:
#   SD_CLI            - path to img_hires binary (default: ./build/img_hires)
#   MODEL_DIR         - model directory (default: /data/models/image)
#   MODEL             - full checkpoint path (default: DreamShaperXL_Turbo_v2_1.safetensors)
#   VAE_MODEL         - external VAE model (default: ae_sdcpp.safetensors, requires USE_EXTERNAL_VAE=1)
#   CLIP_L_MODEL      - external CLIP-L model (default: clip_l_sdcpp.safetensors)
#   CLIP_G_MODEL      - external CLIP-G model (default: clip_g_sdcpp.safetensors)
#   USE_EXTERNAL_CLIP - default 1, set 0 to use checkpoint built-in CLIP
#   USE_EXTERNAL_VAE  - default 0, set 1 to use external VAE (often fails with checkpoints)
#   VAE_TILE_SIZE     - tile size as NxN or single int (default: 32x32)
#   VAE_TILE_OVERLAP  - overlap ratio (default: 0.5)
#   SAMPLING_METHOD   - sampler name (default: euler)
#   SCHEDULER         - scheduler name (default: karras)
#   CFG_SCALE         - CFG scale (default: 3.0)
#   STEPS             - base steps (default: 20)
#   HIRES_STEPS       - HiRes steps (default: 45)
#   HIRES_STRENGTH    - HiRes denoising strength (default: 0.35)
#   SEED              - random seed (default: random)
#
# Examples:
#   ./img3.sh "a photo of a cat" ~/cat.png 1024 1024
#   MODEL=/path/to/another.safetensors ./img3.sh "prompt" ~/out.png 1280 720
#   CFG_SCALE=7.0 STEPS=30 HIRES_STEPS=45 ./img3.sh "prompt" ~/out.png 1280 720
# =============================================================================
set -euo pipefail

RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
NC="\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-/data/models/image}"
SD_CLI="${SD_CLI:-$SCRIPT_DIR/build/img_hires}"
MODEL="${MODEL:-$MODEL_DIR/RealVisXL_V5.0_fp16.safetensors}"
VAE_MODEL="${VAE_MODEL:-$MODEL_DIR/ae_sdcpp.safetensors}"
CLIP_L_MODEL="${CLIP_L_MODEL:-$MODEL_DIR/clip_l_sdcpp.safetensors}"
CLIP_G_MODEL="${CLIP_G_MODEL:-$MODEL_DIR/clip_g_sdcpp.safetensors}"

# Use external VAE/CLIP with checkpoint model? External CLIP is enabled by default
# because it improves prompt following; external VAE is off by default because it
# fails to load with many checkpoints. Override with USE_EXTERNAL_VAE=1 / USE_EXTERNAL_CLIP=0.
USE_EXTERNAL_VAE="${USE_EXTERNAL_VAE:-0}"
USE_EXTERNAL_CLIP="${USE_EXTERNAL_CLIP:-1}"

# FreeU / SAG (RealVisXL usually does not need them)
FREEU="${FREEU:-0}"
FREEU_B1="${FREEU_B1:-1.4}"
FREEU_B2="${FREEU_B2:-1.5}"
SAG="${SAG:-0}"
SAG_SCALE="${SAG_SCALE:-1.0}"

VAE_TILE_SIZE="${VAE_TILE_SIZE:-32x32}"
VAE_TILE_OVERLAP="${VAE_TILE_OVERLAP:-0.5}"

# Defaults tuned for RealVisXL V5.0 (native SDXL, photorealism).
# Override via environment variables for other models.
SAMPLING_METHOD="${SAMPLING_METHOD:-dpm++2m_sde}"
SCHEDULER="${SCHEDULER:-karras}"
CFG_SCALE="${CFG_SCALE:-4.0}"
STEPS="${STEPS:-30}"
HIRES_STEPS="${HIRES_STEPS:-25}"
HIRES_STRENGTH="${HIRES_STRENGTH:-0.25}"

LORA_CONFIG=""

ARGS=()
i=0
while [ $i -lt $# ]; do
    arg="${@:$((i+1)):1}"
    if [ "$arg" = "--lora" ]; then
        i=$((i+1))
        LORA_CONFIG="${@:$((i+1)):1}"
    else
        ARGS+=("$arg")
    fi
    i=$((i+1))
done

PROMPT="${ARGS[0]:-A beautiful landscape}"
OUTPUT_FILE="${ARGS[1]:-}"
WIDTH="${ARGS[2]:-1280}"
HEIGHT="${ARGS[3]:-720}"

if [[ "$OUTPUT_FILE" == ~* ]]; then
    OUTPUT_FILE="${HOME}${OUTPUT_FILE:1}"
fi

echo "========================================"
echo "  Pre-check"
echo "========================================"

if [ ! -f "$SD_CLI" ]; then echo -e "${RED}Error: img_hires binary not found: $SD_CLI${NC}"; exit 1; fi
if [ ! -x "$SD_CLI" ]; then echo -e "${RED}Error: img_hires binary not executable: $SD_CLI${NC}"; exit 1; fi

for model in "$MODEL" "$VAE_MODEL" "$CLIP_L_MODEL" "$CLIP_G_MODEL"; do
    if [ ! -f "$model" ]; then echo -e "${RED}Error: model not found: $model${NC}"; exit 1; fi
done

echo -e "${GREEN}✓ All checks passed${NC}"

if ! [[ "$WIDTH" =~ ^[0-9]+$ ]] || [ "$WIDTH" -le 0 ]; then echo -e "${RED}Error: width must be positive integer${NC}"; exit 1; fi
if ! [[ "$HEIGHT" =~ ^[0-9]+$ ]] || [ "$HEIGHT" -le 0 ]; then echo -e "${RED}Error: height must be positive integer${NC}"; exit 1; fi

VAE_TILE_INT="${VAE_TILE_SIZE%x*}"
if ! [[ "$VAE_TILE_INT" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: VAE_TILE_SIZE must be like 128x128 or 128${NC}"
    exit 1
fi

if [ -n "$OUTPUT_FILE" ]; then
    if [[ "$OUTPUT_FILE" == *"/"* ]]; then
        OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"
        OUTPUT="$(basename "$OUTPUT_FILE")"
    else
        OUTPUT_DIR="$HOME"
        OUTPUT="$OUTPUT_FILE"
    fi
    BASE="${OUTPUT%.png}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT="${BASE}_${TIMESTAMP}.png"
else
    OUTPUT_DIR="$HOME"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    MD5=$(echo "$PROMPT" | md5sum | cut -c1-8)
    OUTPUT="${TIMESTAMP}_${MD5}.png"
fi

# Add quality keywords - same as img2.sh
QUALITY_PREFIX="masterpiece, best quality, ultra-detailed, sharp focus, 8k uhd, photorealistic, highly detailed, crisp, clear, centered composition, complete face, full head, professional portrait"
if [[ "$PROMPT" != *"masterpiece"* ]]; then
    PROMPT="$QUALITY_PREFIX, $PROMPT"
fi

NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, worst quality, jpeg artifacts, noise, grain, soft focus, out of focus, hazy, unclear, bad anatomy, deformed, disfigured, mutated, mutation, malformed, missing limbs, floating limbs, disconnected limbs, extra limbs, duplicate, morbid, mutilated, bad face, cloned face, extra face, double head, extra head, ugly, gross proportions, dehydrated, long neck, cross-eyed, asymmetrical eyes, bad eyes, bad mouth, bad nose, bad teeth, extra fingers, fused fingers, too many fingers, mutated hands, poorly drawn hands, poorly drawn face, malformed hands, missing fingers, border artifacts, edge distortion, tiling artifacts, edge artifacts, frame distortion, warped edges, stretched proportions, asymmetrical face, off-center, cropped, out of frame, partial face, cut off, incomplete head, cropped head, watermark, text, logo, signature, username, artist name, cropped shoulders, embedding:EasyNegative, embedding:bad-hands-5}"

mkdir -p "$OUTPUT_DIR"
OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT"

# Low-res pass resolution (same aspect ratio, 80% of target latent)
TARGET_LATENT_W=$((WIDTH / 8))
TARGET_LATENT_H=$((HEIGHT / 8))

if [ "$WIDTH" -eq 3840 ] && [ "$HEIGHT" -eq 2160 ]; then
    LOW_W=2560; LOW_H=1440
elif [ "$WIDTH" -eq 2560 ] && [ "$HEIGHT" -eq 1440 ]; then
    LOW_W=1920; LOW_H=1080
elif [ "$WIDTH" -eq 1920 ] && [ "$HEIGHT" -eq 1080 ]; then
    LOW_W=1536; LOW_H=864
elif [ "$WIDTH" -eq 1280 ] && [ "$HEIGHT" -eq 720 ]; then
    LOW_W=1024; LOW_H=576
else
    LOW_LATENT_W=$((TARGET_LATENT_W * 4 / 5))
    LOW_LATENT_H=$((TARGET_LATENT_H * 4 / 5))
    LOW_LATENT_W=$(((LOW_LATENT_W + 7) / 8 * 8))
    LOW_LATENT_H=$(((LOW_LATENT_H + 7) / 8 * 8))
    LOW_W=$((LOW_LATENT_W * 8))
    LOW_H=$((LOW_LATENT_H * 8))
fi

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
echo "  SDXL HiRes Generation"
echo "========================================"
echo -e "Target Size: ${GREEN}${WIDTH}x${HEIGHT}${NC}"
echo -e "Low-res Pass: ${GREEN}${LOW_W}x${LOW_H} -> ${WIDTH}x${HEIGHT}${NC}"
echo -e "Steps: $STEPS -> $HIRES_STEPS (HiRes)"
echo -e "CFG Scale: ${CYAN}$CFG_SCALE${NC}"
echo -e "HiRes Strength: $HIRES_STRENGTH"
echo -e "Model:  ${CYAN}$MODEL${NC}"
echo -e "VAE:    ${CYAN}$VAE_MODEL${NC}"
echo -e "CLIP-L: ${CYAN}$CLIP_L_MODEL${NC}"
echo -e "CLIP-G: ${CYAN}$CLIP_G_MODEL${NC}"
echo -e "Sampler: ${CYAN}$SAMPLING_METHOD${NC} + ${CYAN}$SCHEDULER${NC}"
echo -e "VAE Tile: ${CYAN}${VAE_TILE_INT}x${VAE_TILE_INT}${NC} (overlap $VAE_TILE_OVERLAP)"
echo "----------------------------------------"
echo -e "Prompt: ${YELLOW}$PROMPT${NC}"
echo -e "Output: ${GREEN}$OUTPUT_PATH${NC}"
echo "========================================"
echo ""

SEED="${SEED:-$RANDOM}"
echo "Generating...  $(date '+%H:%M:%S')"

SD_CMD=("$SD_CLI"
  --model "$MODEL"
  --negative "$NEGATIVE_PROMPT"
  --cfg "$CFG_SCALE"
  --method "$SAMPLING_METHOD"
  --scheduler "$SCHEDULER"
  --diffusion-fa
  --vae-tiling
  --vae-tile-size "$VAE_TILE_INT"
  --vae-tile-overlap "$VAE_TILE_OVERLAP"
  --clarity 0.4
  --sharpen 0.8
  --sharpen-radius 1
  --smart-sharpen 0.5
  --smart-sharpen-radius 2
  --edge-sharpen 1.5
  --edge-sharpen-radius 2
  --edge-sharpen-threshold 0.3
  -W "$LOW_W"
  -H "$LOW_H"
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

if [ "$USE_EXTERNAL_VAE" -eq 1 ]; then
    SD_CMD+=(--vae "$VAE_MODEL")
fi

if [ "$USE_EXTERNAL_CLIP" -eq 1 ]; then
    SD_CMD+=(--clip-l "$CLIP_L_MODEL" --clip-g "$CLIP_G_MODEL")
fi

if [ "$FREEU" -eq 1 ]; then
    SD_CMD+=(--freeu --freeu-b1 "$FREEU_B1" --freeu-b2 "$FREEU_B2")
fi

if [ "$SAG" -eq 1 ]; then
    SD_CMD+=(--sag --sag-scale "$SAG_SCALE")
fi

if [ -n "$LORA_CONFIG" ]; then
    SD_CMD+=(--lora "$LORA_CONFIG")
fi

START_TIME=$(date +%s)
"${SD_CMD[@]}"
END_TIME=$(date +%s)
GEN_DURATION=$((END_TIME - START_TIME))

if [ -f "$OUTPUT_PATH" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
    if [ $GEN_DURATION -ge 60 ]; then
        DURATION_MIN=$((GEN_DURATION / 60))
        DURATION_SEC=$((GEN_DURATION % 60))
        DURATION_STR="${DURATION_MIN}m ${DURATION_SEC}s"
    else
        DURATION_STR="${GEN_DURATION}s"
    fi
    echo ""
    echo "========================================"
    echo -e "${GREEN}✓ Generation successful!${NC}"
    echo -e "File:   ${GREEN}$OUTPUT_PATH${NC}"
    echo -e "Size:   ${BLUE}$FILE_SIZE${NC}"
    echo -e "Time:   ${YELLOW}$DURATION_STR${NC}"
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
