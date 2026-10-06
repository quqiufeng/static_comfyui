#!/bin/bash
# =============================================================================
# musubi_train.sh — musubi-tuner 训练 z_image LoRA（预缓存，快）
#
# 环境变量:
#   DIT       DiT 权重（默认 Z-Image **Base**；turbo 用 z-image-turbo/transformer/...）
#   ADAPTER   turbo training adapter（默认空=不用；用 turbo 训练时才需要）
#   DATA      数据集目录（含 dataset.toml + images + cache）
#   OUT       输出目录
#   DIM       network_dim (默认 16)   LR (默认 1e-4)
#   EPOCHS    训练轮数 (默认 8)   STEPS  若设置则用 max_train_steps 覆盖
#   FP8       1=开 --fp8_base --fp8_scaled（默认 0，保质量）
#   BLOCKS    --blocks_to_swap N（默认 12，省显存）
#   RES/SEED  可选
# =============================================================================
set -euo pipefail
PY=/data/venv-musubi/bin
MUSUBI=/opt/musubi-tuner/src/musubi_tuner

DIT="${DIT:-/data/models/z-image-base/transformer/diffusion_pytorch_model-00001-of-00002.safetensors}"
ADAPTER="${ADAPTER:-}"
VAE="/data/models/z-image-turbo/vae/diffusion_pytorch_model.safetensors"
TE="/data/models/z-image-te-qwen3.safetensors"
DATA="${DATA:-/data/datasets/mystyle1536}"
OUT="${OUT:-/data/lora/mystyle_base}"
OUT_NAME="${OUT_NAME:-mystyle}"
DIM="${DIM:-16}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-8}"
FP8="${FP8:-0}"
BLOCKS="${BLOCKS:-12}"

mkdir -p "$OUT"

EXTRA=()
[ -n "$ADAPTER" ] && EXTRA+=(--base_weights "$ADAPTER")
[ "$FP8" = "1" ] && EXTRA+=(--fp8_base --fp8_scaled)
[ -n "$BLOCKS" ] && [ "$BLOCKS" != "0" ] && EXTRA+=(--blocks_to_swap "$BLOCKS")

EPOCH_ARGS=(--max_train_epochs "$EPOCHS" --save_every_n_epochs 1)
[ -n "${STEPS:-}" ] && EPOCH_ARGS=(--max_train_steps "$STEPS" --save_every_n_steps "${SAVE_EVERY:-200}")

echo "DIT=$DIT"; echo "ADAPTER=${ADAPTER:-<none>} DATA=$DATA OUT=$OUT DIM=$DIM EPOCHS=$EPOCHS FP8=$FP8 BLOCKS=$BLOCKS"

"$PY/accelerate" launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  "$MUSUBI/zimage_train_network.py" \
  --dit "$DIT" --vae "$VAE" --text_encoder "$TE" \
  --dataset_config "$DATA/dataset.toml" \
  --sdpa --mixed_precision bf16 \
  --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 2.0 \
  --optimizer_type adamw8bit --learning_rate "$LR" \
  --gradient_checkpointing \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --network_module networks.lora_zimage --network_dim "$DIM" --network_alpha "$DIM" \
  --seed "${SEED:-42}" \
  --output_dir "$OUT" --output_name "$OUT_NAME" \
  "${EXTRA[@]}" "${EPOCH_ARGS[@]}"

# 转成 sd.cpp 可加载的 diffusers/PEFT 命名
RAW="$OUT/$OUT_NAME.safetensors"
if [ -f "$RAW" ]; then
    "$PY/python" "$MUSUBI/convert_lora.py" --input "$RAW" \
        --output "$OUT/${OUT_NAME}_sdcpp.safetensors" --target other
    echo "sd.cpp LoRA: $OUT/${OUT_NAME}_sdcpp.safetensors"
fi
