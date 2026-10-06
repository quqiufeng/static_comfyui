#!/bin/bash
# =============================================================================
# musubi_train.sh — 用 musubi-tuner 训练 z_image LoRA（含预缓存 + fp8，快）
#
# 前置（一次性）:
#   dataset.toml 已就绪；latent/TE 已缓存（见 train_lora/README_musubi.md）
#
# 用法:
#   bash musubi_train.sh [MAX_STEPS]
# 环境变量: DIM(默认32) LR(1e-4) STEPS EPOCHS OUT
# =============================================================================
set -euo pipefail
DEPLOY=/data/models
PY=/data/venv-musubi/bin
MUSUBI=/opt/musubi-tuner/src/musubi_tuner

DIT="$DEPLOY/z-image-turbo/transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
VAE="$DEPLOY/z-image-turbo/vae/diffusion_pytorch_model.safetensors"
TE="$DEPLOY/z-image-te-qwen3.safetensors"
ADAPTER="$DEPLOY/zimage_turbo_training_adapter_v2.safetensors"
DATA=/data/datasets/mystyle
OUT="${OUT:-/data/lora/mystyle_musubi}"
DIM="${DIM:-32}"
LR="${LR:-1e-4}"
STEPS="${1:-${STEPS:-}}"

mkdir -p "$OUT"

EPOCH_ARGS=(--max_train_epochs "${EPOCHS:-16}" --save_every_n_epochs 1)
if [ -n "$STEPS" ]; then EPOCH_ARGS=(--max_train_steps "$STEPS" --save_every_n_steps "${SAVE_EVERY:-200}"); fi

"$PY/accelerate" launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  "$MUSUBI/zimage_train_network.py" \
  --dit "$DIT" \
  --vae "$VAE" \
  --text_encoder "$TE" \
  --base_weights "$ADAPTER" \
  --dataset_config "$DATA/dataset.toml" \
  --sdpa --mixed_precision bf16 \
  --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 2.0 \
  --optimizer_type adamw8bit --learning_rate "$LR" \
  --gradient_checkpointing --fp8_base --fp8_scaled \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --network_module networks.lora_zimage --network_dim "$DIM" --network_alpha "$DIM" \
  --seed 42 \
  --output_dir "$OUT" --output_name mystyle \
  "${EPOCH_ARGS[@]}"

# 转成 sd.cpp 可加载的 diffusers/PEFT 命名
RAW="$OUT/mystyle.safetensors"
if [ -f "$RAW" ]; then
    "$PY/python" "$MUSUBI/convert_lora.py" --input "$RAW" \
        --output "$OUT/mystyle_sdcpp.safetensors" --target other
    echo ""
    echo "sd.cpp LoRA: $OUT/mystyle_sdcpp.safetensors"
    echo "出图: cd /opt/static_comfyui/cpp/sd && ./backup.sh \"mystyle style, ...\" ~/out.png 2560 1440 --lora $OUT/mystyle_sdcpp.safetensors:0.8"
fi
