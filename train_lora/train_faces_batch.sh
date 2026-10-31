#!/bin/bash
# 按 /dataset/<明星>/ 目录逐个训练身份 LoRA（>=200 张的 10 人）。
# 产出: /dataset/<明星>/lora/<trigger>.safetensors + .gguf
set -u

REPO=/opt/static_comfyui
RUNROOT=/dataset/_face
DATASET_ROOT=/dataset
PY=/data/venv-musubi/bin
M=/opt/musubi-tuner/src/musubi_tuner
VAE=/data/models/z-image-turbo/vae/diffusion_pytorch_model.safetensors
TE=/data/models/z-image-te-qwen3.safetensors
DIT=/data/models/z-image-base/transformer/diffusion_pytorch_model-00001-of-00002.safetensors
CONF=/data/lora/lora_triggers.conf
EPOCHS="${EPOCHS:-4}"
DIM="${DIM:-32}"
CLEAN="${CLEAN:-1}"

mkdir -p "$RUNROOT"
STATUS="$RUNROOT/status.tsv"
[ -f "$STATUS" ] || printf 'time\tstar\ttrigger\timages\tstatus\n' > "$STATUS"

log() { echo "[$(date '+%F %T')] $*"; }

STARS=(
  "田曦薇|tianxiwei"
  "杨幂|yangmi"
  "娜扎|naza"
  "陈都灵|chenduling"
  "迪丽热巴|dilireba"
  "白鹿|bailu"
  "王楚然|wangchuran"
  "程潇|chengxiao"
  "杨超越|yangchaoyue"
  "鞠婧祎|jujingyi"
)

for spec in "${STARS[@]}"; do
  IFS='|' read -r STAR TRIG <<< "$spec"
  SRC="$DATASET_ROOT/$STAR"
  DS="$RUNROOT/${TRIG}_face"
  OUT="$SRC/lora"
  FINAL="$OUT/$TRIG.safetensors"
  N=$(find "$SRC" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l)

  if [ -s "$FINAL" ]; then
    log "SKIP $STAR ($TRIG): LoRA 已存在"
    continue
  fi

  mkdir -p "$OUT"
  log "=== START $STAR ($TRIG) images=$N epochs=$EPOCHS ==="

  if [ ! -s "$DS/dataset.toml" ]; then
    log "--- crop+tag $STAR -> $DS"
    if ! bash "$REPO/train_lora/make_face_dataset.sh" "$SRC" "$TRIG" "$DS"; then
      log "FAIL $STAR: make_face_dataset"
      printf '%s\t%s\t%s\t%s\tfail_dataset\n' "$(date '+%F %T')" "$STAR" "$TRIG" "$N" >> "$STATUS"
      continue
    fi
  fi

  log "--- cache latents $TRIG"
  if ! "$PY/python" "$M/zimage_cache_latents.py" --dataset_config "$DS/dataset.toml" \
      --vae "$VAE" --device cuda; then
    log "FAIL $STAR: cache_latents"
    printf '%s\t%s\t%s\t%s\tfail_latents\n' "$(date '+%F %T')" "$STAR" "$TRIG" "$N" >> "$STATUS"
    continue
  fi

  log "--- cache TE $TRIG"
  if ! "$PY/python" "$M/zimage_cache_text_encoder_outputs.py" --dataset_config "$DS/dataset.toml" \
      --text_encoder "$TE" --batch_size 8 --device cuda; then
    log "FAIL $STAR: cache_te"
    printf '%s\t%s\t%s\t%s\tfail_te\n' "$(date '+%F %T')" "$STAR" "$TRIG" "$N" >> "$STATUS"
    continue
  fi

  log "--- train $TRIG"
  if ! DIT="$DIT" FP8=0 BLOCKS=0 DIM="$DIM" EPOCHS="$EPOCHS" \
      OUT_NAME="$TRIG" DATA="$DS" OUT="$OUT" ATTN=flash \
      bash "$REPO/train_lora/musubi_train.sh"; then
    log "FAIL $STAR: train"
    printf '%s\t%s\t%s\t%s\tfail_train\n' "$(date '+%F %T')" "$STAR" "$TRIG" "$N" >> "$STATUS"
    continue
  fi

  if [ ! -s "$FINAL" ]; then
    log "FAIL $STAR: missing $FINAL"
    printf '%s\t%s\t%s\t%s\tfail_missing\n' "$(date '+%F %T')" "$STAR" "$TRIG" "$N" >> "$STATUS"
    continue
  fi

  rm -f "$OUT/$TRIG"-[0-9]*.safetensors
  grep -qxF "$TRIG.gguf = $TRIG" "$CONF" 2>/dev/null || echo "$TRIG.gguf = $TRIG" >> "$CONF"
  if [ -s "$OUT/$TRIG.gguf" ]; then
    rm -f "$FINAL"
  fi

  if [ "$CLEAN" = "1" ]; then
    rm -rf "$DS"
    log "cleaned $DS"
  fi

  log "DONE $STAR -> $FINAL"
  printf '%s\t%s\t%s\t%s\tdone\n' "$(date '+%F %T')" "$STAR" "$TRIG" "$N" >> "$STATUS"
done

log "ALL DONE"
