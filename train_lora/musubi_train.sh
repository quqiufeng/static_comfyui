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

# ── 显存自检 & 自动 blocks_to_swap（未显式设置 BLOCKS 时）──────────────
detect_vram_gb() {
    local v=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        v="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1 || true)"
        [ -n "$v" ] && { awk -v m="$v" 'BEGIN{printf "%d", m/1024}'; return; }
    fi
    if [ -x "$PY/python" ]; then
        "$PY/python" -c "import torch;print(int(torch.cuda.get_device_properties(0).total_memory/1024**3))" 2>/dev/null || true
    fi
}
auto_blocks() {  # $1=vram_gb $2=resolution(max side)
    local vram="$1" res="$2" act
    [ -z "$vram" ] && { echo 12; return; }   # 探测失败→保守
    case "$res" in
        512)  act=2.0 ;;
        768)  act=2.5 ;;
        1024) act=3.5 ;;
        1280) act=4.5 ;;
        *)    act=6.0 ;;   # 1536+
    esac
    # 基座 DiT ~12G(bf16)/30 块；框架+LoRA+8bit 优化器 ~3G；激活按分辨率估
    awk -v v="$vram" -v a="$act" 'BEGIN{
        d=12.0; o=3.0; per=d/30.0; avail=v*0.90; need=d+o+a;
        b=(need<=avail)?0:int((need-avail)/per+0.999);
        if(b<0)b=0; if(b>28)b=28; printf "%d", b }'
}
VRAM_GB="$(detect_vram_gb)"
RES=1024
[ -f "$DATA/dataset.toml" ] && RES="$(grep -oE 'resolution *= *\[[0-9]+' "$DATA/dataset.toml" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1)"
RES="${RES:-1024}"
if [ -z "${BLOCKS:-}" ]; then
    BLOCKS="$(auto_blocks "$VRAM_GB" "$RES")"
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
    echo "[auto] GPU=${GPU_NAME:-?} VRAM=${VRAM_GB:-?}G res=${RES} -> BLOCKS=${BLOCKS}"
fi
BLOCKS="${BLOCKS:-12}"

mkdir -p "$OUT"

EXTRA=()
[ -n "$ADAPTER" ] && EXTRA+=(--base_weights "$ADAPTER")
[ "$FP8" = "1" ] && EXTRA+=(--fp8_base --fp8_scaled)
if [ -n "$BLOCKS" ] && [ "$BLOCKS" != "0" ]; then
    EXTRA+=(--blocks_to_swap "$BLOCKS")
    # 加速 CPU<->GPU 换出：LoRA 冻基座只需 H2D 流式 + pinned 内存（SWAP_MODE=plain 可关）
    case "${SWAP_MODE:-h2d}" in
        plain)  : ;;
        pinned) EXTRA+=(--use_pinned_memory_for_block_swap) ;;
        *)      EXTRA+=(--use_pinned_memory_for_block_swap --block_swap_h2d_only --block_swap_ring_size 2) ;;
    esac
fi

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
