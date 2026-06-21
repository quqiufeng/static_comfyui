#!/bin/bash
# scripts/run_attention_compare.sh — 编译运行 manual 和 SDPA 两个版本的 UNet，比较输出
# 注意：需要在 GPU 空闲时运行
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."

if [ ! -f /tmp/sdxl_unet_merged_f32/weights.bin ]; then
    echo "Error: /tmp/sdxl_unet_merged_f32/weights.bin not found" >&2
    echo "Generate it first with:" >&2
    echo "  python3 export_sd_weights.py /data/models/image/sd_xl_base_1.0.safetensors /tmp/sdxl_all_f32" >&2
    echo "  python3 merge_weights.py /tmp/sdxl_all_f32 /tmp/sdxl_unet_merged_f32 model.diffusion_model." >&2
    exit 1
fi

cd "$ROOT"

assemble() {
    local mode="$1"
    local out="$2"
    cat sd_runtime/array_ops.static.py \
        sd_runtime/torch_ops.static.py \
        sd_runtime/nn_ops.static.py \
        sd_runtime/transformer_ops.static.py \
        sd_runtime/unet_forward.static.py \
        test/test_attention_compare.static.py \
        > "$out"
}

# 1. manual attention
bash scripts/gen_unet_manual.sh
assemble manual /tmp/test_attention_manual.static.py
bash build.sh /tmp/test_attention_manual.static.py test_attention_manual
/opt/ChezScheme/ta6le/bin/ta6le/scheme --quiet build_out/test_attention_manual.so
mv /tmp/unet_output.bin /tmp/unet_output_manual.bin

# 2. SDPA attention
bash scripts/gen_unet_sdpa.sh
assemble sdpa /tmp/test_attention_sdpa.static.py
bash build.sh /tmp/test_attention_sdpa.static.py test_attention_sdpa
/opt/ChezScheme/ta6le/bin/ta6le/scheme --quiet build_out/test_attention_sdpa.so
mv /tmp/unet_output.bin /tmp/unet_output_sdpa.bin

# 3. compare with numpy
/data/venv/bin/python3 - <<'PY'
import numpy as np
a = np.fromfile('/tmp/unet_output_manual.bin', dtype=np.float32)
b = np.fromfile('/tmp/unet_output_sdpa.bin', dtype=np.float32)
print('shape:', a.shape, b.shape)
print('manual sum:', a.sum())
print('sdpa sum:', b.sum())
print('max abs diff:', np.abs(a - b).max())
print('mean abs diff:', np.abs(a - b).mean())
PY

# restore default manual attention
bash scripts/gen_unet_manual.sh
