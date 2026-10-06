#!/bin/bash
# =============================================================================
# make_face_dataset.sh — 从"人物图片目录"生成对齐人脸 LoRA 数据集
#   （用 cpp/face 的 facecli 检测+对齐裁剪，再 Florence 打标）
#
# 用法:
#   bash make_face_dataset.sh <原始图片目录> <触发词> [输出数据集目录]
#   bash make_face_dataset.sh /path/to/celeb_photos celebA
#
# 说明:
#   - 图片是**同一个人** → 人物/身份 LoRA；是不同人 → "审美/风格脸" LoRA
#   - 扣脸成功才入选；无脸的图自动跳过
#   - 产出 <数据集>/images/*.png（对齐脸 512）+ *.txt + dataset.toml
#   - 之后按 train_lora.md §13 预缓存 + 训练
# =============================================================================
set -euo pipefail
shopt -s nullglob

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FACECLI="${FACECLI:-$DIR/../cpp/face/facecli}"
PY="${STATICPY_PYTHON:-/data/venv/bin/python}"

SRC="${1:?用法: bash make_face_dataset.sh <原始图片目录> <触发词> [输出数据集目录]}"
TRIGGER="${2:?缺少触发词}"
DS="${3:-/data/datasets/$(basename "$SRC")_face}"

[ -x "$FACECLI" ] || { echo "facecli 未编译: $FACECLI (先跑 cpp/face/build.sh)"; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
n=0; skip=0
for f in "$SRC"/*.jpg "$SRC"/*.jpeg "$SRC"/*.png "$SRC"/*.webp "$SRC"/*.JPG "$SRC"/*.PNG; do
    b="$(basename "$f")"; stem="${b%.*}"
    if "$FACECLI" crop "$f" "$TMP/${stem}.png" 512 >/dev/null 2>&1; then
        n=$((n+1))
    else
        skip=$((skip+1)); echo "  跳过(无脸): $b"
    fi
done
echo "扣脸成功 $n 张，跳过 $skip 张"
[ "$n" -gt 0 ] || { echo "没有可用人脸"; exit 1; }

echo ">>> 打标 + 整理数据集 -> $DS"
"$PY" "$DIR/prepare_dataset.py" "$TMP" "$DS" --trigger "$TRIGGER" --caption \
    --task '<CAPTION>' --beams 1 --max-side 768

cat > "$DS/dataset.toml" <<EOF
[general]
resolution = [512, 512]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "$DS/images"
cache_directory = "$DS/cache"
num_repeats = 1
EOF

echo ""
echo "数据集就绪: $DS"
echo "训练示例（人物 LoRA 建议 rank32）："
echo "  # 预缓存"
echo "  D=$DS PY=/data/venv-musubi/bin; M=/opt/musubi-tuner/src/musubi_tuner"
echo "  \$PY/python \$M/zimage_cache_latents.py --dataset_config \$D/dataset.toml --vae /data/models/z-image-turbo/vae/diffusion_pytorch_model.safetensors --device cuda"
echo "  \$PY/python \$M/zimage_cache_text_encoder_outputs.py --dataset_config \$D/dataset.toml --text_encoder /data/models/z-image-te-qwen3.safetensors --batch_size 8 --device cuda"
echo "  # 训练"
echo "  DIT=/data/models/z-image-base/transformer/diffusion_pytorch_model-00001-of-00002.safetensors \\"
echo "    FP8=0 BLOCKS=12 DIM=32 EPOCHS=10 DATA=$DS OUT=/data/lora/$TRIGGER bash $DIR/musubi_train.sh"
