#!/bin/bash
# =============================================================================
# download_weibo_album.sh — 下载微博相册大图（调用 my-agent 的 weibo_album）
#
# 用法:
#   bash download_weibo_album.sh <uid|昵称|微博URL> <输出目录> [页数|all] [limit|0]
#
#   bash download_weibo_album.sh 7619798979 /dataset/_src/weibostyle 100
#   bash download_weibo_album.sh 'https://weibo.com/u/7619798979?tabtype=album' \
#       /dataset/_src/weibostyle all 0
#
# 说明:
#   - 直接复用 /opt/my-agent/operator/tools/weibo_album.sh（cookie/CDP/相册接口）
#   - 默认 100 页；all = 翻到上限（WEIBO_ALBUM_MAX_PAGE，默认 500）
#   - limit=0 表示不限制下载张数
# =============================================================================
set -euo pipefail

TARGET="${1:?用法: download_weibo_album.sh <uid|昵称|URL> <输出目录> [页数|all] [limit]}"
OUT="${2:?缺少输出目录}"
PAGES="${3:-100}"
LIMIT="${4:-0}"

TOOL="${WEIBO_ALBUM_TOOL:-/opt/my-agent/operator/tools/weibo_album.sh}"
JOBS="${WEIBO_JOBS_TOOL:-/opt/my-agent/operator/tools/jobs.sh}"
[ -x "$TOOL" ] || { echo "缺少 $TOOL" >&2; exit 1; }
[ -x "$JOBS" ] || { echo "缺少 $JOBS" >&2; exit 1; }

# URL 里提取 uid
if [[ "$TARGET" =~ weibo\.com/u/([0-9]+) ]]; then
    TARGET="${BASH_REMATCH[1]}"
fi

mkdir -p "$OUT" || { echo "无法创建 $OUT" >&2; exit 1; }

LOG="/tmp/weibo_album_$$.log"
MAXPAGE="${WEIBO_ALBUM_MAX_PAGE:-500}"
echo "下载微博相册: target=$TARGET pages=$PAGES limit=$LIMIT out=$OUT"
WEIBO_ALBUM_MAX_PAGE="$MAXPAGE" "$TOOL" "$TARGET" "$PAGES" --save "$OUT" --limit "$LIMIT" 2>&1 | tee "$LOG"

JOB="$(sed -n 's/.*job=\([^ ]*\).*/\1/p' "$LOG" | tail -1)"
if [ -n "$JOB" ]; then
    echo "等待下载任务 job=$JOB ..."
    for _ in $(seq 1 7200); do
        ST="$("$JOBS" status "$JOB" 2>/dev/null | tr -d '\r' || true)"
        [ "$ST" = "done" ] && break
        [ "$ST" = "failed" ] && { echo "下载失败 job=$JOB" >&2; exit 1; }
        sleep 2
    done
fi

MAX="${MAX:-2048}"
if command -v mogrify >/dev/null 2>&1; then
    find "$OUT" -maxdepth 1 -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) -print0 |
        xargs -0 -r mogrify -resize "${MAX}x${MAX}>" -quality 95
    echo "resize 完成（长边 ≤$MAX）"
fi

N=$(find "$OUT" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
SIZE=$(du -sh "$OUT" | cut -f1)
echo "完成: $N 张 / $SIZE -> $OUT"
[ "$N" -gt 0 ] || exit 1
