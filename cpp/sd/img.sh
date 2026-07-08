#!/bin/bash
# =============================================================================
# img.sh - SDXL direct image generation using our own img_hires CLI
# =============================================================================
# Usage: ./img.sh "prompt" ~/output.png 1024 1024
#
# Environment variables:
#   SD_CLI            - path to img_hires binary (default: ./build/img_hires)
#   MODEL_DIR         - model directory (default: /data/models/image)
#   MODEL             - full SDXL checkpoint (default: sd_xl_base_1.0.safetensors)
#   VAE_MODEL         - VAE model (default: ae.safetensors)
#   VAE_TILE_SIZE     - tile size as NxN or single int (default: 128x128)
#   VAE_TILE_OVERLAP  - overlap ratio (default: 0.5)
#   SAMPLING_METHOD   - sampler name (default: euler)
#   SCHEDULER         - scheduler name (default: discrete)
#   CFG_SCALE         - CFG scale (default: 7.0)
#   STEPS             - steps (default: 20)
#   SEED              - random seed (default: random)
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
MODEL="${MODEL:-$MODEL_DIR/sd_xl_base_1.0.safetensors}"
VAE_MODEL="${VAE_MODEL:-$MODEL_DIR/ae.safetensors}"

VAE_TILE_SIZE="${VAE_TILE_SIZE:-128x128}"
VAE_TILE_OVERLAP="${VAE_TILE_OVERLAP:-0.5}"

SAMPLING_METHOD="${SAMPLING_METHOD:-euler}"
SCHEDULER="${SCHEDULER:-discrete}"
CFG_SCALE="${CFG_SCALE:-7.0}"
STEPS="${STEPS:-20}"

PROMPT="${1:-A beautiful landscape}"
OUTPUT_FILE="${2:-}"
WIDTH="${3:-1024}"
HEIGHT="${4:-1024}"

if [[ "$OUTPUT_FILE" == ~* ]]; then
    OUTPUT_FILE="${HOME}${OUTPUT_FILE:1}"
fi

echo "========================================"
echo "  Pre-check"
echo "========================================"

if [ ! -f "$SD_CLI" ]; then echo -e "${RED}Error: img_hires binary not found: $SD_CLI${NC}"; exit 1; fi
if [ ! -x "$SD_CLI" ]; then echo -e "${RED}Error: img_hires binary not executable: $SD_CLI${NC}"; exit 1; fi

for model in "$MODEL" "$VAE_MODEL"; do
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

NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, worst quality, jpeg artifacts, noise, grain, soft focus, out of focus, hazy, unclear, bad anatomy, deformed, watermark, text, logo, signature}"

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

mkdir -p "$OUTPUT_DIR"
OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT"

echo ""
echo "========================================"
echo "  SDXL Direct Generation"
echo "========================================"
echo -e "Target Size: ${GREEN}${WIDTH}x${HEIGHT}${NC}"
echo -e "Steps: $STEPS"
echo -e "CFG Scale: ${CYAN}$CFG_SCALE${NC}"
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
  --vae "$VAE_MODEL"
  --negative "$NEGATIVE_PROMPT"
  --cfg "$CFG_SCALE"
  --method "$SAMPLING_METHOD"
  --scheduler "$SCHEDULER"
  --diffusion-fa
  --vae-tiling
  --vae-tile-size "$VAE_TILE_INT"
  --vae-tile-overlap "$VAE_TILE_OVERLAP"
  --freeu
  --freeu-b1 1.4
  --freeu-b2 1.5
  --sag
  --sag-scale 1.0
  --clarity 0.4
  --sharpen 0.8
  --sharpen-radius 1
  --smart-sharpen 0.5
  --smart-sharpen-radius 2
  --edge-sharpen 1.5
  --edge-sharpen-radius 2
  --edge-sharpen-threshold 0.3
  -W "$WIDTH"
  -H "$HEIGHT"
  --steps "$STEPS"
  -s "$SEED"
  "$PROMPT"
  "$OUTPUT_PATH"
)

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
