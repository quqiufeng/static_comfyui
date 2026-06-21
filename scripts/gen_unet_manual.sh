#!/bin/bash
# scripts/gen_unet_manual.sh — 生成 manual attention 版本的 unet_forward.static.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
INDEX="${1:-/tmp/sdxl_unet_merged_f32/index.json}"

if [ ! -f "$INDEX" ]; then
    echo "Error: index.json not found at $INDEX" >&2
    echo "Generate it first with:" >&2
    echo "  python3 export_sd_weights.py /data/models/image/sd_xl_base_1.0.safetensors /tmp/sdxl_all_f32" >&2
    echo "  python3 merge_weights.py /tmp/sdxl_all_f32 /tmp/sdxl_unet_merged_f32 model.diffusion_model." >&2
    exit 1
fi

cd "$ROOT"
python3 gen_unet.py "$INDEX" manual > sd_runtime/unet_forward.static.py
echo "Generated sd_runtime/unet_forward.static.py (manual attention)"
