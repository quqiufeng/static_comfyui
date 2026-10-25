#!/bin/bash
# =============================================================================
# check_labels.sh — 校验图片与 .txt 一一对应，并修复
#
#   有图无 txt  -> 用 Florence 重新打标
#   有 txt 无图 -> 删除该 txt（图片已被手工筛掉）
#
# 用法: check_labels.sh <images_dir> <trigger> [task]
#   task 默认 <CAPTION>
# =============================================================================
set -euo pipefail

DIR="${1:?用法: check_labels.sh <images_dir> <trigger> [task]}"
TRIG="${2:?缺少 trigger}"
TASK="${3:-<CAPTION>}"
IMG2PROMPT="${IMG2PROMPT:-/opt/static_comfyui/cpp/florence2/img2prompt}"
FLORENCE_DIR="${FLORENCE_DIR:-/data/models/florence2}"

[ -d "$DIR" ] || { echo "目录不存在: $DIR" >&2; exit 1; }
[ -x "$IMG2PROMPT" ] || { echo "缺少 img2prompt: $IMG2PROMPT" >&2; exit 1; }

declare -A IMGPATH
while IFS= read -r -d '' f; do
    IMGPATH["${f%.*}"]="$f"
done < <(find "$DIR" -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.bmp' \) -print0)

shopt -s nullglob
DEL=0
for t in "$DIR"/*.txt; do
    stem="${t%.txt}"
    if [ -z "${IMGPATH[$stem]:-}" ]; then
        rm -f "$t"; echo "  [del] $(basename "$t")"; DEL=$((DEL+1))
    fi
done

MISS=()
for stem in "${!IMGPATH[@]}"; do
    [ -s "$stem.txt" ] || MISS+=("${IMGPATH[$stem]}")
done

FIX=0
if [ "${#MISS[@]}" -gt 0 ]; then
    echo "  补打标 ${#MISS[@]} 张..."
    printf '%s\n' "${MISS[@]}" | "$IMG2PROMPT" --model-dir "$FLORENCE_DIR" \
        --task "$TASK" --beams 1 --batch > /tmp/check_labels_caps.txt 2>/dev/null || true
    mapfile -t CAPS < /tmp/check_labels_caps.txt
    for i in "${!MISS[@]}"; do
        img="${MISS[$i]}"; cap="${CAPS[$i]:-}"
        if [ -n "$cap" ]; then printf '%s, %s\n' "$TRIG" "$cap" > "${img%.*}.txt"
        else printf '%s\n' "$TRIG" > "${img%.*}.txt"; fi
        FIX=$((FIX+1))
    done
fi

NIMG=$(find "$DIR" -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.bmp' \) | wc -l)
NTXT=$(find "$DIR" -maxdepth 1 -type f -name '*.txt' | wc -l)
echo "  图片=$NIMG txt=$NTXT | 删orphan=$DEL 补打标=$FIX"
[ "$NIMG" -eq "$NTXT" ] || { echo "  ERROR: 数量仍不一致"; exit 2; }
