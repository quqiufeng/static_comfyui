#!/bin/bash
# =============================================================================
# make_style_dataset.sh — 每次训练：从 /dataset/<明星>/ 分层抽 N 张 → 记录 JSON
#                         → 复制到新目录 → 调用 style_pipeline 处理/打标
#
# 特点:
#   - 源目录只读：只 cp，绝不 mv / resize 源图
#   - 按明星轮转抽样（round-robin），避免田曦薇/unknown 等占比过高
#   - 选了哪些图、每个明星多少张、随机种子，全部写入 selection.json
#   - 处理/打标复用现有 style_pipeline.sh，产物落到本次 run 的新目录
#
# 用法:
#   bash make_style_dataset.sh [数量=2000] [名字=style2000] [触发词=名字]
#   bash make_style_dataset.sh 2000 style2000 mystyle
#
# 环境变量:
#   SRC_BASE=/dataset       源目录（只读）
#   OUT_ROOT=/dataset/_runs 每次 run 的输出根目录
#   SEED=42                 抽样种子（同种子=同一批图；换种子=换一批）
#   INCLUDE_UNKNOWN=0       1=把 unknown 目录也纳入抽样
#   MAX=1536                训练长边（传给 style_pipeline）
#   TASK=<CAPTION>          打标任务（可改 <DETAILED_CAPTION>）
#   SKIP_MANUAL=1           跳过手工筛选（默认 1，全自动）
#   PROCESS=1               0=只抽样/复制/写 JSON，不跑打标
#   RUN=<dir>               自定义本次输出目录
# =============================================================================
set -euo pipefail
export LC_ALL=C

TARGET="${1:-2000}"
NAME="${2:-style${TARGET}}"
TRIG="${3:-$NAME}"

SRC_BASE="${SRC_BASE:-/dataset}"
OUT_ROOT="${OUT_ROOT:-$SRC_BASE/_runs}"
SEED="${SEED:-42}"
INCLUDE_UNKNOWN="${INCLUDE_UNKNOWN:-0}"
MAX="${MAX:-1536}"
TASK="${TASK:-<CAPTION>}"
SKIP_MANUAL="${SKIP_MANUAL:-1}"
PROCESS="${PROCESS:-1}"
REPO="${REPO:-/opt/static_comfyui}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$SCRIPT_DIR/style_pipeline.sh"

log() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

[ -d "$SRC_BASE" ] || die "源目录不存在: $SRC_BASE"
[ -x "$PIPELINE" ] || die "缺少 style_pipeline.sh: $PIPELINE"
command -v jq >/dev/null || die "需要 jq"
command -v awk >/dev/null || die "需要 awk"
[[ "$TARGET" =~ ^[1-9][0-9]*$ ]] || die "数量必须是正整数: $TARGET"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "SEED 必须是整数: $SEED"

RUN="${RUN:-$OUT_ROOT/${NAME}_$(date +%Y%m%d_%H%M%S)}"
SEL="$RUN/selected"
mkdir -p "$SEL"

# ── 1) 汇总 + 分层抽样 ───────────────────────────────────────
# find 只取明星目录下一层图片；排除 others / _runs / 隐藏目录；默认排除 unknown。
# 先按 star+path 排序保证输入顺序稳定 → 每个明星内用 seed 打乱 → 编号 rank
# → 按 rank 轮转（每个明星第 1 张、第 2 张…）取前 TARGET 张。
log "扫描 $SRC_BASE（排除 others/_runs/隐藏；INCLUDE_UNKNOWN=$INCLUDE_UNKNOWN）..."
find "$SRC_BASE" -mindepth 2 -maxdepth 2 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
       -o -iname '*.webp' -o -iname '*.bmp' \) \
    -not -path "$SRC_BASE/others/*" \
    -not -path "$SRC_BASE/_runs/*" \
    -not -path "$SRC_BASE/.*" \
    -printf '%h\t%p\n' |
awk -F'\t' -v inc_unknown="$INCLUDE_UNKNOWN" '{
    star=$1; sub(/^.*\//, "", star);
    if (star == "unknown" && inc_unknown != "1") next;
    print star "\t" $2;
}' |
sort -t$'\t' -k1,1 -k2,2 |
awk -F'\t' -v seed="$SEED" 'BEGIN{srand(seed)} {print $1 "\t" rand() "\t" $2}' |
sort -t$'\t' -k1,1 -k2,2g |
awk -F'\t' '{n[$1]++; print n[$1] "\t" $1 "\t" $3}' |
sort -t$'\t' -k1,1n -k2,2 >"$RUN/.ranked.tsv"
head -n "$TARGET" "$RUN/.ranked.tsv" >"$RUN/selected.tsv"
rm -f "$RUN/.ranked.tsv"

PICKED="$(wc -l <"$RUN/selected.tsv")"
[ "$PICKED" -gt 0 ] || die "没有抽到图片"
if [ "$PICKED" -lt "$TARGET" ]; then
    log "警告: 可用图片不足，只抽到 $PICKED/$TARGET 张"
fi

# ── 2) 复制到新目录（扁平），源图不动 ─────────────────────────
# 同名冲突时加明星前缀；JSON 里记录 staged 文件名。
: >"$RUN/selection.tsv"
while IFS=$'\t' read -r rank star src; do
    [ -f "$src" ] || continue
    base="$(basename "$src")"
    dst="$SEL/$base"
    [ -e "$dst" ] && dst="$SEL/${star}__$base"
    if [ -e "$dst" ]; then
        dst="$SEL/${base%.*}.$(date +%s%N).${base##*.}"
    fi
    if ! cp -p --reflink=auto -- "$src" "$dst" 2>/dev/null; then
        cp -p -- "$src" "$dst" || die "复制失败: $src"
    fi
    printf '%s\t%s\t%s\n' "$star" "$src" "$(basename "$dst")" >>"$RUN/selection.tsv"
done <"$RUN/selected.tsv"

COPIED="$(wc -l <"$RUN/selection.tsv")"
log "已复制 $COPIED 张到 $SEL"

# ── 3) selection.json（选了谁、每明星多少、seed、输出路径）────
jq -R -s \
    --arg created "$(date -Iseconds)" \
    --argjson seed "$SEED" \
    --argjson target "$TARGET" \
    --argjson selected "$COPIED" \
    --argjson include_unknown "$INCLUDE_UNKNOWN" \
    --arg src_base "$SRC_BASE" \
    --arg run "$RUN" \
    --arg name "$NAME" \
    --arg trigger "$TRIG" \
    --argjson max_side "$MAX" \
    --arg task "$TASK" \
    '
    split("\n")
    | map(select(length > 0) | split("\t") | {star: .[0], source: .[1], staged: .[2]})
    | . as $imgs
    | {
        created_at: $created,
        seed: $seed,
        target: $target,
        selected: $selected,
        include_unknown: ($include_unknown == 1),
        source_root: $src_base,
        run_dir: $run,
        name: $name,
        trigger: $trigger,
        max_side: $max_side,
        task: $task,
        per_star: ($imgs | group_by(.star) | map({key: .[0].star, value: length}) | from_entries),
        images: $imgs,
        output: {
          selected: ($run + "/selected"),
          images: ($run + "/style/" + $name + "/images"),
          dataset_toml: ($run + "/style/" + $name + "/dataset.toml")
        }
      }
    ' "$RUN/selection.tsv" >"$RUN/selection.json"

log "selection.json: $RUN/selection.json"

# ── 4) 处理 + 打标（读 $SEL，写 $RUN/style/$NAME，源图仍不动）──
if [ "$PROCESS" = "1" ]; then
    log "调用 style_pipeline（MAX=$MAX TASK=$TASK）..."
    BASE="$RUN" MAX="$MAX" LIMIT=0 TASK="$TASK" SKIP_MANUAL="$SKIP_MANUAL" \
        bash "$PIPELINE" "$SEL" "$NAME" "$TRIG"
    log "完成"
    echo "  selection : $RUN/selection.json"
    echo "  staged    : $SEL ($COPIED 张，源图未动)"
    echo "  images    : $RUN/style/$NAME/images"
    echo "  dataset   : $RUN/style/$NAME/dataset.toml"
else
    log "PROCESS=0：只抽样/复制/写 JSON，未打标"
    echo "  selection : $RUN/selection.json"
    echo "  staged    : $SEL ($COPIED 张)"
fi
