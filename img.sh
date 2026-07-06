#!/bin/bash
# SDXL txt2img using myimg-cli (GGML-based C++ inference)
MODEL="/data/models/image/sd_xl_base_1.0.safetensors"
CLIP_L="/data/models/image/clip_l.safetensors"
CLIP_G="/data/models/image/clip_g.safetensors"
OUTPUT="/tmp/output.png"

/opt/my-img/build/myimg-cli \
    -m "$MODEL" \
    --clip-l "$CLIP_L" \
    --clip-g "$CLIP_G" \
    -p "solo,single woman,half body portrait of a young woman, soft natural lighting, elegant pose, studio lighting, sharp eyes, clean white background, medium close up" \
    -n "blurry, low quality, ugly" \
    -W 1024 -H 1024 \
    --steps 20 \
    --cfg-scale 7.0 \
    --sampling-method euler \
    --scheduler karras \
    -s 42 \
    --output "$OUTPUT"

echo "Saved to $OUTPUT"
