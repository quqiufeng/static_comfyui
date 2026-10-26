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
  "3018707285|weibo_album_3018707285"
  "6657546529|weibo_album_6657546529"
  "7972556454|weibo_album_7972556454"
  "7918574208|weibo_album_7918574208"
  "7864668908|weibo_album_7864668908"
  "6898913719|weibo_album_6898913719"
  "7311284104|weibo_album_7311284104"
)

[ -f "$CDP_SCRIPT" ] || { echo "缺少 $CDP_SCRIPT" >&2; exit 1; }
mkdir -p "$OUTBASE" 2>/dev/null || { echo "无法创建 $OUTBASE" >&2; exit 1; }

if ! curl -s --max-time 2 http://127.0.0.1:9222/json/version >/dev/null; then
  echo "ERROR: Chrome 9222 未运行；请先手动启动专用实例（本脚本不会杀/重启浏览器）" >&2
  exit 1
fi

for entry in "${ACCOUNTS[@]}"; do
  uid="${entry%%|*}"
  name="${entry##*|}"
  out="$OUTBASE/$name"
  log="/tmp/weibo_accounts_$uid.log"

  n=0
  [ -f "$out/metadata.tsv" ] && n=$(wc -l < "$out/metadata.tsv")
  ended=false
  [ -f "$out/progress.json" ] && ended=$(jq -r '.ended // false' "$out/progress.json" 2>/dev/null || echo false)

  if [ "${FORCE:-0}" != "1" ] && { [ "$n" -ge "$COUNT" ] || { [ "$ended" = "true" ] && [ "$n" -gt 0 ]; }; }; then
    echo "SKIP $uid ($n, ended=$ended)" >> "$STATUS"
    record_progress "$uid" "$name" "$n" "$out"
    continue
  fi

  free=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ -n "$free" ] && [ "$free" -lt 15 ]; then
    echo "STOP disk<$free G" >> "$STATUS"; break
  fi

  # 不自动启动/重启 Chrome：9222 不在就停下，绝不杀现有浏览器
  if ! curl -s --max-time 2 http://127.0.0.1:9222/json/version >/dev/null; then
    echo "NO_CHROME $uid (请手动启动 9222，脚本不杀浏览器)" >> "$STATUS"
    break
  fi

  echo "START $uid $name target=$COUNT $(date '+%F %T')" >> "$STATUS"
  : > "$log"
  timeout 1800 node "$CDP_SCRIPT" \
    "https://weibo.com/u/$uid?tabtype=album&uid=$uid&index=0" "$COUNT" "$out" \
    >> "$log" 2>&1 &
  np=$!
  lost=0
  while kill -0 "$np" 2>/dev/null; do
    sleep 5
    if ! curl -s --max-time 2 http://127.0.0.1:9222/json/version >/dev/null; then
      echo "CHROME_LOST $uid (不再自动重启；已采部分保留)" >> "$STATUS"
      kill "$np" 2>/dev/null
      wait "$np" 2>/dev/null
      lost=1
      break
    fi
  done
  if [ "$lost" -eq 0 ]; then
    wait "$np"; rc=$?
  else
    rc=137
  fi

  cnt=$(wc -l < "$out/metadata.tsv" 2>/dev/null || echo 0)
  ended=false
  [ -f "$out/progress.json" ] && ended=$(jq -r '.ended // false' "$out/progress.json" 2>/dev/null || echo false)
  echo "END $uid rc=$rc n=$cnt ended=$ended $(date '+%F %T')" >> "$STATUS"
  record_progress "$uid" "$name" "$cnt" "$out"
done
echo "ALL DONE $(date '+%F %T')" >> "$STATUS"
