#!/bin/bash
# =============================================================================
# train_lora.sh — 训练 z_image (Z-Image-Turbo) 风格 LoRA（QLoRA 4bit, 20G 可跑）
#
# 用法:
#   DATA=/data/datasets/mystyle RANK=16 STEPS=800 bash train_lora.sh
#
# 环境变量:
#   BASE   训练基座 (默认 /data/models/z-image-turbo)
#   DATA   数据集目录（含 images/ 与 instance_prompt.txt；由 prepare_dataset.py 生成）
#   OUT    输出目录 (默认 /data/lora/<data 名>)
#   RANK   LoRA rank (默认 16)
#   STEPS  训练步数 (默认 800)
#   RES    分辨率 (默认 1024)
# =============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/data/venv/bin

BASE="${BASE:-/data/models/z-image-turbo}"
DATA="${DATA:?请设置 DATA=<数据集目录>}"
OUT="${OUT:-/data/lora/$(basename "$DATA")}"
RANK="${RANK:-16}"
STEPS="${STEPS:-800}"
RES="${RES:-1024}"

[ -d "$BASE/transformer" ] || { echo "基座未就绪: $BASE (先下 Tongyi-MAI/Z-Image-Turbo)"; exit 1; }
[ -d "$DATA/images" ] || { echo "数据集缺 $DATA/images"; exit 1; }
PROMPT="$(cat "$DATA/instance_prompt.txt")"
mkdir -p "$OUT"

echo "==================== 训练 z_image LoRA ===================="
echo "  base   : $BASE"
echo "  data   : $DATA/images ($(ls "$DATA/images" | wc -l) imgs)"
echo "  out    : $OUT"
echo "  prompt : $PROMPT"
echo "  rank=$RANK steps=$STEPS res=$RES  (QLoRA 4bit)"
echo "==========================================================="

"$PY/accelerate" launch \
  --num_processes=1 --num_machines=1 --mixed_precision=bf16 \
  "$DIR/train_dreambooth_lora_z_image.py" \
  --pretrained_model_name_or_path "$BASE" \
  --instance_data_dir "$DATA/images" \
  --output_dir "$OUT" \
  --instance_prompt "$PROMPT" \
  --resolution "$RES" \
  --center_crop \
  --random_flip \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --rank "$RANK" \
  --lora_alpha "$RANK" \
  --lora_dropout 0.0 \
  --learning_rate 1e-4 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --max_train_steps "$STEPS" \
  --checkpointing_steps 200 \
  --checkpoints_total_limit 2 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --offload \
  --bnb_quantization_config_path "$DIR/qnbit4.json" \
  --mixed_precision bf16 \
  --seed 42 \
  --report_to none \
  --skip_final_inference

echo ""
echo "=== 训练完成 ==="
RAW="$OUT/pytorch_lora_weights.safetensors"
if [ -f "$RAW" ]; then
    CONV="$OUT/$(basename "$OUT")_sdcpp.safetensors"
    "$PY/python" "$DIR/convert_lora.py" "$RAW" "$CONV"
    echo ""
    echo "sd.cpp 可直接加载的 LoRA: $CONV"
    echo "出图: cd /opt/static_comfyui/cpp/sd && ./backup.sh \"$PROMPT\" ~/lora_test.png 2560 1440 --lora $CONV:0.8"
else
    echo "未找到 $RAW，检查训练日志"
    ls -la "$OUT" 2>/dev/null || true
fi
