#!/bin/bash
# =============================================================================
# download_weibo_accounts.sh — 微博相册批量采集（CDP，逐张 →，/large 原图）
#
# 用法:
#   bash download_weibo_accounts.sh [每号张数=100]
#   FORCE=1 bash download_weibo_accounts.sh 100   # 忽略已完成检查
#
# 注意：参数是「每个号」的目标张数，不是所有号的总数。
#
# 依赖:
#   - 已启动专用 Chrome: /opt/my-agent/operator/tools/chrome_debug.sh
#   - node + train_lora/weibo_album_cdp.js + celebrities.txt
#
# 输出:
#   /dataset/_src/<name>/<明星>/*.jpg + metadata.tsv
#   状态: /tmp/weibo_accounts.status
#   日志: /tmp/weibo_accounts_<uid>.log
# =============================================================================
set -uo pipefail

COUNT="${1:-100}"
OUTBASE="${OUTBASE:-/dataset/_src}"
CDP_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/weibo_album_cdp.js"
CHROME_DEBUG="/opt/my-agent/operator/tools/chrome_debug.sh"
STATUS=/tmp/weibo_accounts.status
PROGRESS="$OUTBASE/weibo_progress.tsv"
: > "$STATUS"
[ -s "$PROGRESS" ] || printf 'uid\tname\tcount\ttarget\tlast_url\tupdated\n' > "$PROGRESS"

record_progress() { # uid name count outdir
  local u="$1" n="$2" c="$3" o="$4" last
  last=$(tail -1 "$o/metadata.tsv" 2>/dev/null | cut -f1)
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$u" "$n" "$c" "$COUNT" "$last" "$(date '+%F %T')" >> "$PROGRESS"
}

# uid|输出目录名
ACCOUNTS=(
  "7619798979|weibo_album_7619798979_large"
  "7954334448|weibo_album_7954334448"
  "5880302175|weibo_album_5880302175"
  "6872799860|weibo_album_6872799860"
  "3018707285|weibo_album_3018707285"
  "6657546529|weibo_album_6657546529"
  "7972556454|weibo_album_7972556454"
  "7918574208|weibo_album_7918574208"
)

[ -f "$CDP_SCRIPT" ] || { echo "缺少 $CDP_SCRIPT" >&2; exit 1; }
mkdir -p "$OUTBASE" 2>/dev/null || { echo "无法创建 $OUTBASE" >&2; exit 1; }

if ! curl -s --max-time 2 http://127.0.0.1:9222/json/version >/dev/null; then
  echo "启动专用 Chrome（9222）..."
  bash "$CHROME_DEBUG"
  sleep 2
fi

for entry in "${ACCOUNTS[@]}"; do
  uid="${entry%%|*}"
  name="${entry##*|}"
  out="$OUTBASE/$name"
  log="/tmp/weibo_accounts_$uid.log"

  n=0
  [ -f "$out/metadata.tsv" ] && n=$(wc -l < "$out/metadata.tsv")
  if [ "${FORCE:-0}" != "1" ] && [ "$n" -ge "$COUNT" ]; then
    echo "SKIP $uid ($n)" >> "$STATUS"
    record_progress "$uid" "$name" "$n" "$out"
    continue
  fi

  free=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ -n "$free" ] && [ "$free" -lt 15 ]; then
    echo "STOP disk<$free G" >> "$STATUS"; break
  fi

  echo "START $uid $name target=$COUNT $(date '+%F %T')" >> "$STATUS"
  timeout 1800 node "$CDP_SCRIPT" \
    "https://weibo.com/u/$uid?tabtype=album&uid=$uid&index=0" "$COUNT" "$out" \
    > "$log" 2>&1
  rc=$?
  cnt=$(wc -l < "$out/metadata.tsv" 2>/dev/null || echo 0)
  echo "END $uid rc=$rc n=$cnt $(date '+%F %T')" >> "$STATUS"
  record_progress "$uid" "$name" "$cnt" "$out"
done
echo "ALL DONE $(date '+%F %T')" >> "$STATUS"
