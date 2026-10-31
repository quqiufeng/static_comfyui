#!/bin/bash
# =============================================================================
# run_style_pipeline.sh — 一条龙：抽样/复制 → 打标 → 预缓存 → 训练
#
# 用法:
#   bash run_style_pipeline.sh [数量=2000] [名字=style2000] [触发词=mystyle]
#
# 环境变量:
#   RUN=<dir>       本次输出目录（默认 /dataset/_runs/<name>_<时间>）
#   EPOCHS=2       训练轮数（快速验证；完整跑用 EPOCHS=4）
#   DIM=16         LoRA rank/alpha
#   FP8=0          1=开 fp8（默认关）
#   ATTN=flash     注意力实现；不可用自动回退 sdpa
#   SEED=42        抽样种子
#   OUT=/data/lora/<name>
#
# 由启动器用 setsid 拉起，stdout/stderr 重定向到 $RUN/pipeline.log；
# tmux 里只跑 tail -f 该日志，方便随时 attach 监控。
# =============================================================================
set -euo pipefail

TARGET="${1:-2000}"
NAME="${2:-style2000}"
TRIG="${3:-mystyle}"
EPOCHS="${EPOCHS:-2}"
DIM="${DIM:-16}"
FP8="${FP8:-0}"
ATTN="${ATTN:-flash}"
SEED="${SEED:-42}"
SKIP_TAG="${SKIP_TAG:-0}"
SKIP_LATENT="${SKIP_LATENT:-0}"
ZIMAGE_TOKENIZER_DIR="${ZIMAGE_TOKENIZER_DIR:-/data/models/z-image-turbo/tokenizer}"
BASE="${BASE:-/dataset}"
OUT_ROOT="${OUT_ROOT:-$BASE/_runs}"
REPO="${REPO:-/opt/static_comfyui}"
RUN="${RUN:-$OUT_ROOT/${NAME}_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-/data/lora/$NAME}"

PY=/data/venv-musubi/bin
M=/opt/musubi-tuner/src/musubi_tuner
VAE=/data/models/z-image-turbo/vae/diffusion_pytorch_model.safetensors
TE=/data/models/z-image-te-qwen3.safetensors

mkdir -p "$RUN" "$OUT"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
status() { printf '%s %s\n' "$(date '+%F %T')" "$*" >"$RUN/STATUS"; }
trap 'status "FAILED line $LINENO"; log "FAILED line $LINENO"' ERR

{
    echo "RUN=$RUN"
    echo "TARGET=$TARGET"
    echo "NAME=$NAME"
    echo "TRIG=$TRIG"
    echo "EPOCHS=$EPOCHS"
    echo "DIM=$DIM"
    echo "FP8=$FP8"
    echo "ATTN=$ATTN"
    echo "SEED=$SEED"
    echo "OUT=$OUT"
} >"$RUN/run.env"

if [ "$SKIP_TAG" = "1" ]; then
    log "== 1/3 跳过抽样/打标（复用 $RUN）=="
else
    status "tagging (select + copy + tag)"
    log "== 1/3 抽样 + 复制 + 打标（$TARGET 张）=="
    RUN="$RUN" SEED="$SEED" PROCESS=1 SKIP_MANUAL=1 \
        bash "$REPO/train_lora/make_style_dataset.sh" "$TARGET" "$NAME" "$TRIG"
fi

D="$RUN/style/$NAME"
[ -f "$D/dataset.toml" ] || { log "缺少 $D/dataset.toml"; exit 1; }

status "caching (latent + text encoder)"
log "== 2/3 预缓存 latent + TE =="
if [ "$SKIP_LATENT" = "1" ]; then
    log "跳过 latent 缓存（复用已有）"
else
    "$PY/python" "$M/zimage_cache_latents.py" --dataset_config "$D/dataset.toml" \
        --vae "$VAE" --device cuda --skip_existing
fi
# tokenizer 用本地副本，避免 HuggingFace 联网超时
ZIMAGE_TOKENIZER_DIR="$ZIMAGE_TOKENIZER_DIR" "$PY/python" "$M/zimage_cache_text_encoder_outputs.py" \
    --dataset_config "$D/dataset.toml" --text_encoder "$TE" --batch_size 8 --device cuda --skip_existing

status "training"
log "== 3/3 训练（EPOCHS=$EPOCHS DIM=$DIM ATTN=$ATTN）=="
DATA="$D" OUT="$OUT" OUT_NAME="$TRIG" EPOCHS="$EPOCHS" DIM="$DIM" FP8="$FP8" ATTN="$ATTN" \
    bash "$REPO/train_lora/musubi_train.sh"

status "done"
log "全部完成: $OUT/$TRIG.safetensors"
