#!/bin/bash
# deliver.sh — static_comfyui 单文件 ELF 打包
set -euo pipefail
SCHEME="/opt/ChezScheme/ta6le/bin/ta6le/scheme"
RESCHEME_DIR="/opt/ReScheme"
OUTPUT="${1:-sd_generate}"
echo "=== static_comfyui — ELF Build ==="
echo "  [1] Merging StaticPy sources..."
MERGED_PY="/tmp/sd_merged.py"
> "$MERGED_PY"
for f in sd_runtime/array_ops.static.py sd_runtime/nn_ops.static.py sd_runtime/unet_blocks.static.py sd_runtime/samplers.static.py sd_runtime/clip.static.py sd_runtime/vae.static.py sd_runtime/model_loader.static.py sd_runtime/unet.static.py sd_runtime/main.static.py; do
    if [ -f "$f" ]; then cat "$f" >> "$MERGED_PY"; echo "" >> "$MERGED_PY"; fi
done
echo "       $(wc -l < "$MERGED_PY") lines merged"
echo "  [2] Translating..."
/data/venv/bin/python3 "$RESCHEME_DIR/static_translate.py" < "$MERGED_PY" > /tmp/sd_translated.ss
echo "  [3] Compiling..."
CACHE_DIR="/tmp/staticpy-cache"; mkdir -p "$CACHE_DIR"
MERGED_SS="/tmp/sd_merged.ss"
cat "$RESCHEME_DIR/static_prelude.scm" > "$MERGED_SS"
echo "" >> "$MERGED_SS"
cat "$RESCHEME_DIR/static_stdlib.scm" >> "$MERGED_SS"
echo "" >> "$MERGED_SS"
cat /tmp/sd_translated.ss >> "$MERGED_SS"
cat > /tmp/sd_compile.ss << EOF
(import (chezscheme))
(compile-file "$MERGED_SS")
EOF
$SCHEME --quiet /tmp/sd_compile.ss 2>&1
mv "${MERGED_SS}o" /tmp/sd_runtime.so 2>/dev/null || true
echo "       -> /tmp/sd_runtime.so"
echo ""
echo "Run: $SCHEME --quiet /tmp/sd_runtime.so"
