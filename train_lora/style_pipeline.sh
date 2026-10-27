#!/bin/bash
# =============================================================================
# style_pipeline.sh — 纯风格图源 → style LoRA 数据集（不要求人脸、不扣脸、不去背景）
#
# 用法:
#   bash style_pipeline.sh <源图片目录> <名字> [触发词]
#   bash style_pipeline.sh /dataset/_src/weibostyle weibostyle weibostyle
#
# 环境变量:
#   BASE=/dataset  MAX=1536  MIN_SIDE=512  LIMIT=0（0=全部）
#   TASK=<CAPTION>（可改 <DETAILED_CAPTION>）  SKIP_MANUAL=0
#
# 流程: 缩放(只缩) → 手工筛 → 打标 → dataset.toml → 重建 style 总表
# =============================================================================
set -euo pipefail

SRC="${1:?用法: style_pipeline.sh <源目录> <名字> [触发词]}"
NAME="${2:?缺少 name}"
TRIG="${3:-$NAME}"

BASE="${BASE:-/dataset}"
MAX="${MAX:-1536}"
MIN_SIDE="${MIN_SIDE:-512}"
LIMIT="${LIMIT:-0}"
TASK="${TASK:-<CAPTION>}"
SKIP_MANUAL="${SKIP_MANUAL:-0}"
REPO="${REPO:-/opt/static_comfyui}"
STATICPY="${STATICPY:-/opt/StaticPy}"

IMG2PROMPT="$REPO/cpp/florence2/img2prompt"
PIL="$STATICPY/imgproc/pil"
FLORENCE_DIR="/data/models/florence2"
OUT_BASE="$BASE/style/$NAME"
OUT="$OUT_BASE/images"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { log "ERROR: $*"; exit 1; }

[ -d "$SRC" ] || die "源目录不存在: $SRC"
if [ ! -d "$BASE" ]; then
    mkdir -p "$BASE" 2>/dev/null || die "缺少 $BASE；先执行: sudo mkdir -p $BASE && sudo chown \$USER:\$USER $BASE"
fi
for x in "$IMG2PROMPT" "$PIL"; do [ -x "$x" ] || die "工具不存在: $x"; done
command -v identify >/dev/null || die "需要 ImageMagick identify"

mkdir -p "$OUT"
rm -f "$OUT"/* 2>/dev/null || true

mapfile -t FILES < <(find "$SRC" -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.bmp' \) | sort)
[ "${#FILES[@]}" -gt 0 ] || die "源目录没有图片"
if [ "$LIMIT" -gt 0 ] && [ "${#FILES[@]}" -gt "$LIMIT" ]; then
    FILES=("${FILES[@]:0:$LIMIT}")
fi
log "处理 ${#FILES[@]} 张（MAX=$MAX，LIMIT=$LIMIT）..."

N=0
for f in "${FILES[@]}"; do
    read -r w h < <(identify -format '%w %h\n' "$f" 2>/dev/null || echo "0 0")
    [ "$w" -gt 0 ] && [ "$h" -gt 0 ] || { log "跳过无法读尺寸: $(basename "$f")"; continue; }
    if [ "$w" -ge "$h" ]; then m=$w; else m=$h; fi
    [ "$m" -ge "$MIN_SIDE" ] || { log "跳过过小: $(basename "$f")"; continue; }
    ext="${f##*.}"; ext="${ext,,}"; [ "$ext" = "jpeg" ] && ext=jpg
    out="$OUT/$(basename "${f%.*}").$ext"
    if [ "$m" -gt "$MAX" ]; then
        read -r nw nh < <(awk -v w="$w" -v h="$h" -v m="$MAX" \
            'BEGIN{s=m/(w>h?w:h); printf "%d %d\n", (w*s+0.5), (h*s+0.5)}')
        LD_LIBRARY_PATH="$STATICPY/cpp/imgproc/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$PIL" resize "$f" "$out" "$nw" "$nh" lanczos >/dev/null || { log "缩放失败: $(basename "$f")"; continue; }
    else
        cp -f "$f" "$out"
    fi
    N=$((N+1))
done
log "缩放完成: $N 张"

if [ "$SKIP_MANUAL" != "1" ]; then
    echo
    echo "==== 手工筛选 ===="
    echo "目录: $OUT"
    echo "删除不要的图后回车继续（Ctrl-C 中止）。"
    read -r -p "> " _
fi

mapfile -t IMGS < <(find "$OUT" -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.bmp' \) | sort)
[ "${#IMGS[@]}" -gt 0 ] || die "筛选后为空"

# 一次 --batch 把所有图片喂给同一个 img2prompt 进程：Florence 只加载一次，
# 后续逐张推理复用同一模型，不会每张重启。
log "打标 ${#IMGS[@]} 张（task=$TASK，一次 --batch，模型只加载一次）..."
printf '%s\n' "${IMGS[@]}" | "$IMG2PROMPT" --model-dir "$FLORENCE_DIR" \
    --task "$TASK" --beams 1 --max-new-tokens 32 --batch > /tmp/style_caps.txt 2>/dev/null || true
mapfile -t CAPS < /tmp/style_caps.txt
for i in "${!IMGS[@]}"; do
    img="${IMGS[$i]}"; cap="${CAPS[$i]:-}"
    if [ -n "$cap" ]; then printf '%s, %s\n' "$TRIG" "$cap" > "${img%.*}.txt"
    else printf '%s\n' "$TRIG" > "${img%.*}.txt"; fi
done

cat > "$OUT_BASE/dataset.toml" <<EOF
[general]
resolution = [$MAX, $MAX]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "$OUT"
cache_directory = "$OUT_BASE/cache"
num_repeats = 1
EOF

# style 总表：汇总所有 style 子目录
{
    cat <<EOF
[general]
resolution = [$MAX, $MAX]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true
EOF
    for d in "$BASE"/style/*/; do
        [ -d "$d/images" ] || continue
        cat <<EOF

[[datasets]]
image_directory = "$d/images"
cache_directory = "$d/cache"
num_repeats = 1
EOF
    done
} > "$BASE/style/dataset.toml"

log "完成"
echo "  style: $OUT ($(find "$OUT" -maxdepth 1 -name '*.txt' | wc -l) 张)"
echo "  toml : $OUT_BASE/dataset.toml"
echo "  master: $BASE/style/dataset.toml"
