#!/bin/bash
# 身份 LoRA（可选叠加风格 LoRA）批量出图，一人一张。默认提示词。
# 可选: STYLE_LORA=/data/lora/style2000_v5/mystyle.gguf:1.2
set -u

REPO=/opt/static_comfyui
OUTDIR="${OUTDIR:-$HOME/face_2560}"
WEIGHT="${WEIGHT:-0.7}"
GEN_W="${GEN_W:-2560}"
GEN_H="${GEN_H:-1440}"
GEN_TAG="${GEN_TAG:-2560x1440}"
STYLE_LORA="${STYLE_LORA:-}"
LOGDIR="$OUTDIR/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

NEG=$(sed -n 's/^NEGATIVE_PROMPT="\${NEGATIVE_PROMPT:-\(.*\)}"$/\1/p' "$REPO/cpp/sd/backup.sh")
EXTRA="two people, multiple women, duplicate, cloned face, extra head, disembodied head, second face, twins, mirror image, reflection"

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
  LORA="/dataset/$STAR/lora/$TRIG.gguf"
  LOG="$LOGDIR/$TRIG.log"
  echo "[$(date '+%F %T')] START $STAR ($TRIG) ${GEN_W}x${GEN_H} w=$WEIGHT"
  [ -s "$LORA" ] || { echo "missing $LORA"; continue; }

  LORA_ARGS=(--lora "$LORA:$WEIGHT")
  [ -n "$STYLE_LORA" ] && LORA_ARGS=(--lora "$STYLE_LORA" "${LORA_ARGS[@]}")

  OK=0
  for attempt in 1 2 3; do
    SEED=$(shuf -i 1-2147483646 -n1)
    echo "[$(date '+%F %T')] attempt=$attempt seed=$SEED" >> "$LOG"
    if [ "$attempt" -eq 1 ]; then
      ( cd "$REPO" && SEED="$SEED" ./cpp/sd/backup.sh "$OUTDIR/${TRIG}_${GEN_TAG}.png" "$GEN_W" "$GEN_H" \
          "${LORA_ARGS[@]}" ) >> "$LOG" 2>&1
    else
      ( cd "$REPO" && SEED="$SEED" NEGATIVE_PROMPT="$NEG, $EXTRA" \
          ./cpp/sd/backup.sh "$OUTDIR/${TRIG}_${GEN_TAG}.png" "$GEN_W" "$GEN_H" \
          "${LORA_ARGS[@]}" ) >> "$LOG" 2>&1
    fi
    F=$(ls -t "$OUTDIR/${TRIG}_${GEN_TAG}_"*.png 2>/dev/null | head -1)
    if [ -n "$F" ]; then
      N=$("$REPO/cpp/face/facecli" detect "$F" 2>/dev/null | grep -o '"score"' | wc -l)
      echo "[$(date '+%F %T')] file=$F faces=$N" >> "$LOG"
      if [ "$N" -eq 1 ]; then
        echo "[$(date '+%F %T')] DONE $STAR -> $F"
        printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$STAR" "$TRIG" "$F" "ok" >> "$OUTDIR/status.tsv"
        OK=1
        break
      fi
    fi
  done

  if [ "$OK" -ne 1 ]; then
    echo "[$(date '+%F %T')] FAIL $STAR after 3 attempts"
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$STAR" "$TRIG" "-" "fail" >> "$OUTDIR/status.tsv"
  fi
done

echo "[$(date '+%F %T')] ALL DONE"
