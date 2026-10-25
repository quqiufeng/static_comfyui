#!/bin/bash
# =============================================================================
# image_pipeline.sh — 明星图源 → style/face 训练集（shell + 现成二进制，零 Python）
#
# 用法:
#   bash image_pipeline.sh <src> <name> [style_trigger] [face_trigger]
#   bash image_pipeline.sh /run/user/.../weibo/佟丽娅 佟丽娅
#
# 流程（尺寸/筛选都在拷贝目录内完成，最后只 cp 合格图）:
#   1) cp/scp 子目录 -> /data/dataset/<name>
#   2) 人物门禁：无人脸/非人物先删除（不进入后续处理）
#   3) 在 <name>/ 内逐张等比缩放到长边 <= MAX
#   4) facecli detect + Florence 反推自动筛（多人脸/低分/低清/非写真/无人），不合格删除
#   5) 手工筛选（暂停，直接在 <name>/ 里删）
#   6) 去背景（BiRefNet，只处理筛选后的合格图）
#   7) 合格图 cp 到 /data/dataset/style/images；facecli crop -> /data/dataset/face/<name>/images
#   8) img2prompt 打标 -> 同名 .txt
#   9) 校验图片↔txt
#   10) 写 dataset.toml
#
# 环境变量:
#   BASE=/data/dataset  MAX=1536  MIN_SIDE=512  MIN_FACE_SCORE=0.60
#   FACE_SIZE=512  STYLE_TRIGGER=mystyle  FACE_TRIGGER=<name>
#   SKIP_MANUAL=1  跳过手工筛选
#   REPO=/opt/static_comfyui  STATICPY=/opt/StaticPy
# =============================================================================
set -euo pipefail

SRC="${1:?用法: image_pipeline.sh <src> <name> [style_trigger] [face_trigger]}"
NAME="${2:?缺少 name（明星/触发词）}"
STYLE_TRIGGER="${3:-mystyle}"
FACE_TRIGGER="${4:-$NAME}"

BASE="${BASE:-/data/dataset}"
MAX="${MAX:-1536}"
MIN_SIDE="${MIN_SIDE:-512}"
MIN_FACE_SCORE="${MIN_FACE_SCORE:-0.60}"
FACE_SIZE="${FACE_SIZE:-512}"
REPO="${REPO:-/opt/static_comfyui}"
STATICPY="${STATICPY:-/opt/StaticPy}"
SKIP_MANUAL="${SKIP_MANUAL:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FACECLI="$REPO/cpp/face/facecli"
IMG2PROMPT="$REPO/cpp/florence2/img2prompt"
PIL="$STATICPY/imgproc/pil"
FLORENCE_DIR="/data/models/florence2"

SRC_DIR="$BASE/$NAME"
STYLE_DIR="$BASE/style/images"
FACE_DIR="$BASE/face/$NAME/images"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { log "ERROR: $*"; exit 1; }
is_img() { case "${1,,}" in *.jpg|*.jpeg|*.png|*.webp|*.bmp) return 0;; *) return 1;; esac; }

for x in "$FACECLI" "$IMG2PROMPT" "$PIL"; do [ -x "$x" ] || die "工具不存在: $x"; done
command -v jq >/dev/null || die "需要 jq"
command -v identify >/dev/null || die "需要 ImageMagick identify"
[ "$MAX" -ge 64 ] || die "MAX 太小: $MAX"

mkdir -p "$BASE" "$STYLE_DIR" "$FACE_DIR"
rm -f "$FACE_DIR"/* 2>/dev/null || true   # 重跑时清掉该明星的旧 face

# ── 1) 拷贝图源 ─────────────────────────────────────────────
if [ -d "$SRC_DIR" ]; then
    log "图源已存在，增量拷贝: $SRC -> $SRC_DIR"
    cp -r "$SRC/." "$SRC_DIR/"
elif [[ "$SRC" != /* && "$SRC" == *:* ]]; then
    log "scp 图源: $SRC -> $SRC_DIR"
    scp -r "$SRC" "$SRC_DIR"
else
    log "拷贝图源: $SRC -> $SRC_DIR"
    cp -r "$SRC" "$SRC_DIR"
fi

# 记录本次图源全部 stem（用于清掉历史 style/face 残留）
declare -A ALL
while IFS= read -r -d '' f; do ALL["${f%.*}"]=1; done < <(find "$SRC_DIR" -maxdepth 1 -type f -print0)

# ── 2) 人物门禁：先确认有人，再缩放 ─────────────────────────
export LD_LIBRARY_PATH="$REPO/cpp/face${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
log "人物检测（无人脸直接删除）..."
mapfile -t GATE < <(find "$SRC_DIR" -maxdepth 1 -type f | sort)
printf '%s\n' "${GATE[@]}" | "$FACECLI" detect --batch > /tmp/ip_gate.jsonl 2>/dev/null || true
mapfile -t GJSON < /tmp/ip_gate.jsonl
NG=0
for i in "${!GATE[@]}"; do
    n="$(jq 'length' <<<"${GJSON[$i]:-[]}" 2>/dev/null || echo 0)"
    if [ "$n" -lt 1 ]; then
        log "非人物/无人脸，删除: $(basename "${GATE[$i]}")"
        rm -f "${GATE[$i]}"; NG=$((NG+1))
    fi
done
log "人物门禁: 删除 $NG"

# ── 3) 在拷贝目录内等比缩放 ─────────────────────────────────
log "在 $SRC_DIR 内缩放（长边上限 $MAX）..."
find "$SRC_DIR" -maxdepth 1 -type f -print0 | sort -z | while IFS= read -r -d '' f; do
    is_img "$f" || continue
    read -r w h < <(identify -format '%w %h\n' "$f" 2>/dev/null || echo "0 0")
    [ "$w" -gt 0 ] && [ "$h" -gt 0 ] || { log "跳过无法读尺寸: $f"; continue; }
    if [ "$w" -ge "$h" ]; then m=$w; else m=$h; fi
    [ "$m" -gt "$MAX" ] || continue
    ext="${f##*.}"; ext="${ext,,}"; [ "$ext" = "jpeg" ] && ext=jpg
    tmp="${f%.*}.resize.$$.$ext"
    read -r nw nh < <(awk -v w="$w" -v h="$h" -v m="$MAX" \
        'BEGIN{s=m/(w>h?w:h); printf "%d %d\n", (w*s+0.5), (h*s+0.5)}')
    if LD_LIBRARY_PATH="$STATICPY/cpp/imgproc/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$PIL" resize "$f" "$tmp" "$nw" "$nh" lanczos >/dev/null; then
        mv -f "$tmp" "$f"
    else
        rm -f "$tmp"; log "缩放失败，保留原图: $(basename "$f")"
    fi
done

# ── 4) 自动筛选（人脸 + 内容）────────────────────────────────
mapfile -t IMGS < <(find "$SRC_DIR" -maxdepth 1 -type f | sort)
[ "${#IMGS[@]}" -gt 0 ] || die "图源为空"

log "人脸检测 ${#IMGS[@]} 张..."
export LD_LIBRARY_PATH="$REPO/cpp/face${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
printf '%s\n' "${IMGS[@]}" | "$FACECLI" detect --batch > /tmp/ip_faces.jsonl 2>/dev/null || true
mapfile -t FACES < /tmp/ip_faces.jsonl

log "反推描述（内容筛选 + 后续 style 打标）..."
printf '%s\n' "${IMGS[@]}" | "$IMG2PROMPT" --model-dir "$FLORENCE_DIR" \
    --task '<CAPTION>' --beams 1 --batch > /tmp/ip_caps.txt 2>/dev/null || true
mapfile -t CAPS < /tmp/ip_caps.txt

NONPHOTO='anime|manga|cartoon|illustration|illustrated|drawing|painting|sketch|logo|watermark|poster|advertisement|screenshot|comic|graphic|vector|clipart|emoji|sticker|album cover'
PERSON='woman|women|man|men|girl|boy|person|people|portrait|face|model|female|male|lady|guy|child|kid'
declare -A CAP
KEPT=0; DROP=0
for i in "${!IMGS[@]}"; do
    img="${IMGS[$i]}"; js="${FACES[$i]:-[]}"; cap="${CAPS[$i]:-}"
    n="$(jq 'length' <<<"$js" 2>/dev/null || echo 0)"
    score="$(jq '[.[].score] | max // 0' <<<"$js" 2>/dev/null || echo 0)"
    read -r w h < <(identify -format '%w %h\n' "$img" 2>/dev/null || echo "0 0")
    if [ "$w" -ge "$h" ]; then m=$w; else m=$h; fi
    reason=""
    [ "$n" -eq 0 ] && reason="no_face"
    [ "$n" -gt 1 ] && reason="multi_face"
    [ "$n" -eq 1 ] && awk -v s="$score" -v t="$MIN_FACE_SCORE" 'BEGIN{exit !(s<t)}' && reason="${reason:-low_score}"
    [ "$m" -lt "$MIN_SIDE" ] && reason="${reason:-small}"
    if [ -n "$cap" ]; then
        if grep -Eiq "$NONPHOTO" <<<"$cap"; then reason="${reason:-not_photo}"
        elif ! grep -Eiq "$PERSON" <<<"$cap"; then reason="${reason:-no_person}"
        fi
    fi
    if [ -n "$reason" ]; then
        log "剔除[$reason] $(basename "$img")"
        rm -f "$img"; DROP=$((DROP+1))
    else
        CAP["${img%.*}"]="$cap"; KEPT=$((KEPT+1))
    fi
done
log "自动筛选: 保留 $KEPT，剔除 $DROP"

# ── 5) 手工筛选（在拷贝目录里删）────────────────────────────
if [ "$SKIP_MANUAL" != "1" ]; then
    echo
    echo "==== 手工筛选 ===="
    echo "目录: $SRC_DIR"
    echo "请直接删除不要的图片，完成后回车继续（Ctrl-C 中止）。"
    read -r -p "> " _
fi
mapfile -t IMGS < <(find "$SRC_DIR" -maxdepth 1 -type f | sort)
[ "${#IMGS[@]}" -gt 0 ] || die "手工筛选后目录为空"

# ── 6) 去背景（只处理合格图；batch 一次加载模型）──────────
log "去背景（BiRefNet batch，仅 ${#IMGS[@]} 张合格图）..."
BGREMOVE="$STATICPY/cpp/birefnet/bgremove"
if [ -x "$BGREMOVE" ]; then
    : > /tmp/ip_bg.tsv
    for f in "${IMGS[@]}"; do
        [ -f "$f" ] || continue
        stem="$(basename "${f%.*}")"
        printf '%s\t%s\n' "$f" "$SRC_DIR/$stem.bg.$$.png" >> /tmp/ip_bg.tsv
    done
    if LD_LIBRARY_PATH="$STATICPY/cpp/birefnet${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$BGREMOVE" --mode transparent --list /tmp/ip_bg.tsv >/dev/null 2>&1; then
        for f in "${IMGS[@]}"; do
            [ -f "$f" ] || continue
            stem="$(basename "${f%.*}")"
            tmp="$SRC_DIR/$stem.bg.$$.png"; out="$SRC_DIR/$stem.png"
            if [ -s "$tmp" ]; then
                mv -f "$tmp" "$out"
                if [ "$f" != "$out" ]; then rm -f "$f"; fi
            else
                log "去背景失败，保留原图: $(basename "$f")"
            fi
        done
    else
        log "bgremove batch 失败，保留原图"
        rm -f "$SRC_DIR"/*.bg.$$.png
    fi
else
    log "缺少 $BGREMOVE，跳过去背景"
fi
mapfile -t IMGS < <(find "$SRC_DIR" -maxdepth 1 -type f | sort)

# ── 7) 合格图 cp 到 style + 扣脸到 face ─────────────────────
: > /tmp/ip_crop.tsv
for img in "${IMGS[@]}"; do
    stem="$(basename "${img%.*}")"
    # 清掉该 stem 的旧 style 副本/标注，避免 jpg→png 后同 stem 两份
    for e in jpg jpeg png webp bmp txt; do rm -f "$STYLE_DIR/$stem.$e"; done
    cp -f "$img" "$STYLE_DIR/"
    printf '%s\t%s %s\n' "$img" "$FACE_DIR/$stem.png" "$FACE_SIZE" >> /tmp/ip_crop.tsv
done
log "批量扣脸 ${#IMGS[@]} 张..."
"$FACECLI" crop --batch < /tmp/ip_crop.tsv >/tmp/ip_crop.log 2>/dev/null || true

# 扣脸失败 → 两张都不保留
for img in "${IMGS[@]}"; do
    stem="$(basename "${img%.*}")"
    if [ ! -s "$FACE_DIR/$stem.png" ]; then
        log "扣脸失败，移除: $(basename "$img")"
        rm -f "$img" "$STYLE_DIR/$(basename "$img")"
    fi
done

# 用当前目录实际保留的 stem 清理 style/face 残留
declare -A KEEP
for img in "${IMGS[@]}"; do [ -e "$img" ] && KEEP["$(basename "${img%.*}")"]=1; done
for f in "$STYLE_DIR"/*; do
    [ -f "$f" ] || continue
    is_img "$f" || continue
    stem="$(basename "${f%.*}")"
    if [ -n "${ALL[$stem]:-}" ] && [ -z "${KEEP[$stem]:-}" ]; then rm -f "$f"; CUT=$((CUT+1)); fi
done
for f in "$FACE_DIR"/*.png; do
    [ -f "$f" ] || continue
    stem="$(basename "${f%.*}")"
    if [ -n "${ALL[$stem]:-}" ] && [ -z "${KEEP[$stem]:-}" ]; then rm -f "$f"; CUT=$((CUT+1)); fi
done
log "清理历史副本: $CUT 个"

# ── 8) 打标（就地写 .txt）──────────────────────────────────
mapfile -t IMGS < <(find "$SRC_DIR" -maxdepth 1 -type f | sort)
log "style 打标..."
for img in "${IMGS[@]}"; do
    stem="$(basename "${img%.*}")"
    out="$STYLE_DIR/$(basename "$img")"
    cap="${CAP[$stem]:-}"
    if [ -z "$cap" ]; then
        cap="$("$IMG2PROMPT" --model-dir "$FLORENCE_DIR" --task '<CAPTION>' --beams 1 "$out" 2>/dev/null | head -1)"
    fi
    if [ -n "$cap" ]; then printf '%s, %s\n' "$STYLE_TRIGGER" "$cap" > "${out%.*}.txt"
    else printf '%s\n' "$STYLE_TRIGGER" > "${out%.*}.txt"; fi
done

mapfile -t FACES_IMG < <(find "$FACE_DIR" -maxdepth 1 -type f -iname '*.png' | sort)
log "face 打标..."
if [ "${#FACES_IMG[@]}" -gt 0 ]; then
    printf '%s\n' "${FACES_IMG[@]}" | "$IMG2PROMPT" --model-dir "$FLORENCE_DIR" \
        --task '<CAPTION>' --beams 1 --batch > /tmp/ip_facecaps.txt 2>/dev/null || true
    mapfile -t FCAPS < /tmp/ip_facecaps.txt
    for i in "${!FACES_IMG[@]}"; do
        img="${FACES_IMG[$i]}"; cap="${FCAPS[$i]:-}"
        if [ -n "$cap" ]; then printf '%s, %s\n' "$FACE_TRIGGER" "$cap" > "${img%.*}.txt"
        else printf '%s\n' "$FACE_TRIGGER" > "${img%.*}.txt"; fi
    done
fi

# ── 9) 校验：图片 ↔ txt ────────────────────────────────────
if [ -x "$SCRIPT_DIR/check_labels.sh" ]; then
    log "校验 style 标注..."
    bash "$SCRIPT_DIR/check_labels.sh" "$STYLE_DIR" "$STYLE_TRIGGER" '<CAPTION>'
    log "校验 face 标注..."
    bash "$SCRIPT_DIR/check_labels.sh" "$FACE_DIR" "$FACE_TRIGGER" '<CAPTION>'
fi

# ── 10) dataset.toml ────────────────────────────────────────
write_toml() { # $1=dir $2=res
    cat > "$1/dataset.toml" <<EOF
[general]
resolution = [$2, $2]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "$1/images"
cache_directory = "$1/cache"
num_repeats = 1
EOF
}
write_toml "$BASE/style" "$MAX"
write_toml "$BASE/face/$NAME" "$FACE_SIZE"

log "完成"
echo "  source: $SRC_DIR ($(find "$SRC_DIR" -maxdepth 1 -type f | wc -l) 张)"
echo "  style : $STYLE_DIR ($(find "$STYLE_DIR" -maxdepth 1 -type f -name '*.txt' | wc -l) 张)"
echo "  face  : $FACE_DIR ($(find "$FACE_DIR" -maxdepth 1 -type f -name '*.txt' | wc -l) 张)"
