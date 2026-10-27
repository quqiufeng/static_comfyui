#!/bin/bash
# =============================================================================
# move_nonperson.sh — 汇总所有图片路径 → 并发 Florence-2 反推 → 非人物 mv 到 others
#
# 思路：先把 /dataset/<明星>/ 下所有图片路径汇总成一个清单，切成小块放进队列，
#       再由 N 个 worker 并发领取（模型每块只加载一次），按反推描述判断
#       是否含人物；不含人物的图片 mv 到 /dataset/others/。
#
# 用法:
#   bash move_nonperson.sh start [workers]   # 启动（默认 4 并发，后台）
#   bash move_nonperson.sh status            # 查看进度
#   bash move_nonperson.sh stop              # 停止 worker（可再次 start 续跑）
#   bash move_nonperson.sh finalize          # 手动收尾（正常会自动执行）
#   bash move_nonperson.sh reeval            # 用缓存短描述重判 + 无人脸图补详描（不重跑短描述）
#
# 环境变量:
#   BASE=/dataset  OTHERS=$BASE/others  WORKERS=4  CHUNK=200
#   MODEL_DIR=/data/models/florence2  TASK='<CAPTION>'
#   MAX_TOKENS=24  IMG2PROMPT=<path>
#   DETAIL=1                # 1=对短描述判为"整人"且无人脸的图，再用 DETAIL_TASK 补判
#   DETAIL_TASK='<DETAILED_CAPTION>'  DETAIL_MAX_TOKENS=96
#   FACECLI=<path>           # 默认 /opt/static_comfyui/cpp/face/facecli
#   REPO=/opt/static_comfyui
# =============================================================================
set -uo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
BASE="${BASE:-/dataset}"
OTHERS="${OTHERS:-$BASE/others}"
WORKERS="${WORKERS:-4}"
CHUNK="${CHUNK:-200}"
MODEL_DIR="${MODEL_DIR:-/data/models/florence2}"
TASK="${TASK:-<CAPTION>}"
MAX_TOKENS="${MAX_TOKENS:-24}"
IMG2PROMPT="${IMG2PROMPT:-/opt/static_comfyui/cpp/florence2/img2prompt}"
REPO="${REPO:-/opt/static_comfyui}"
FACECLI="${FACECLI:-$REPO/cpp/face/facecli}"
DETAIL="${DETAIL:-1}"
DETAIL_TASK="${DETAIL_TASK:-<DETAILED_CAPTION>}"
DETAIL_MAX_TOKENS="${DETAIL_MAX_TOKENS:-96}"
STATE="$BASE/.nonperson"
C_ID=""
C_PATHS=()
C_CAPS=()

# 人物相关词。命中任意一个 → 认为有人物，保留。
# 宁可多保留，不要误移真人照片；反推常用的主语/身体部位/角色词都放进来。
PERSON_RE='\b(woman|women|man|men|girl|girls|boy|boys|person|persons|people|portrait|model|female|male|lady|ladies|guy|guys|child|children|kid|kids|baby|babies|toddler|teen|teenager|adult|human|couple|couples|family|bride|groom|actress|actor|singer|dancer|athlete|soldier|officer|police|doctor|nurse|student|teacher|mother|father|mom|dad|son|daughter|brother|sister|grandma|grandpa|grandmother|grandfather|husband|wife|boyfriend|girlfriend|chef|waiter|waitress|driver|rider|passenger|crowd|audience|spectator|tourist|shopper|customer|worker|farmer|fisherman|painter|artist|musician|rapper|host|hostess|presenter|reporter|journalist|anchor|speaker|leader|president|captain|sailor|pilot|astronaut|scientist|engineer|programmer|professor|patient|neighbor|friend|selfie|smile|smiling|she|he|her|his|him)\b'

# 局部特写：只有脚/腿/手/脸/鞋等，不是"整人" → 移走。
# 只匹配"以身体部位/鞋开头"、明确的 close-up、或"某人的脚/鞋…"，
# 避免误伤 "A woman ... with her hands on her shoulders" 这种整身描述。
BODY='feet|foot|legs?|hands?|arms?|fingers?|toes?|eyes?|lips?|mouth|face|head|hair|shoulders?|torso|chest|neck|waist|thighs?|knees?'
FOOT='sandals?|shoes?|heels?|high[- ]?heels?|boots?|sneakers?|slippers?|flats?|loafers?|pumps?|stilettos?|wedges?|mules?|clogs?|footwear'
PARTIAL_RE="\\b(close[- ]?up|closeup|macro|zoomed[- ]in)\\b|^[[:space:]]*(a|an|the)?[[:space:]]*($BODY|$FOOT)\\b|^[[:space:]]*(a|an|the)?[[:space:]]*pair of[[:space:]]+([[:alpha:]'-]+[[:space:]]+){0,3}($BODY|$FOOT)\\b|\\b(woman|women|man|men|girl|boy|person|model|lady|ladies|guy)['’]s[[:space:]]+([[:alpha:]'-]+[[:space:]]+){0,3}($BODY|$FOOT)\\b|^[[:space:]]*(her|his)[[:space:]]+([[:alpha:]'-]+[[:space:]]+){0,3}($BODY|$FOOT)\\b|\\b($BODY|$FOOT)[[:space:]]+of[[:space:]]+(a|an|the|her|his|this|that)\\b|\\b(only|just)[[:space:]]+([[:alpha:]]+[[:space:]]+){0,4}($BODY|$FOOT)\\b"

log() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

# 去掉 Florence 详描常见的开头（"In this image we can see …"），
# 让"以身体部位/鞋开头"的规则对详描同样生效。
norm_cap() {
    local c="${1,,}" p
    for p in \
        "in this image we can see " "in this image, we can see " \
        "in this image we see " "in this image, " \
        "in the image we can see " "in the image, we can see " \
        "the image shows " "this image shows " "the image features " \
        "there is "; do
        c="${c#"$p"}"
    done
    printf '%s' "$c"
}
is_person() { grep -Eiq "$PERSON_RE" <<<"$(norm_cap "$1")"; }
is_partial() { grep -Eiq "$PARTIAL_RE" <<<"$(norm_cap "$1")"; }

# 移动到 others，文件名冲突时加明星前缀/纳秒后缀；成功写 report
move_to_others() {
    local src="$1" reason="$2" cap="$3" report="$4"
    [ -f "$src" ] || return 0
    local base star dst
    base="$(basename "$src")"
    star="$(basename "$(dirname "$src")")"
    dst="$OTHERS/$base"
    if [ -e "$dst" ]; then dst="$OTHERS/${star}__$base"; fi
    if [ -e "$dst" ]; then
        dst="$OTHERS/${base%.*}.$(date +%s%N).${base##*.}"
    fi
    if mv -n -- "$src" "$dst"; then
        printf '%s\t%s\t%s\t%s\n' "$src" "$cap" "$reason" "$dst" >>"$report"
    else
        printf '%s\tmove_failed\n' "$src" >>"$STATE/logs/move_fail.log"
    fi
}

# 第二道：短描述判为"整人"的图，先用/facecli 查人脸。
# 无人脸 → 用 DETAIL_TASK 再反推一次，避免短描述把"只有腿/脚"说成"女人站着"。
# 使用全局 C_ID / C_PATHS[] / C_CAPS[]（与短描述一一对应）。
detail_pass() {
    local report="$1" skip="$2"
    [ "$DETAIL" = "1" ] || return 0
    [ -x "$FACECLI" ] || return 0

    local -a keep_idx=() keep_paths=()
    local i p cap
    for i in "${!C_PATHS[@]}"; do
        p="${C_PATHS[$i]}"; cap="${C_CAPS[$i]:-}"
        [ -f "$p" ] || continue
        [ -z "$cap" ] && continue
        if is_person "$cap" && ! is_partial "$cap"; then
            keep_idx+=("$i"); keep_paths+=("$p")
        fi
    done
    [ "${#keep_paths[@]}" -gt 0 ] || return 0

    local faces="$STATE/caps/$C_ID.faces.jsonl"
    printf '%s\n' "${keep_paths[@]}" | \
        LD_LIBRARY_PATH="$REPO/cpp/face${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$FACECLI" detect --batch >"$faces" 2>/dev/null || true
    mapfile -t FACEJSON <"$faces"

    local -a review_idx=() review_paths=()
    for i in "${!keep_paths[@]}"; do
        local js="${FACEJSON[$i]:-[]}" n=0
        if command -v jq >/dev/null 2>&1; then
            n="$(jq 'length' <<<"$js" 2>/dev/null || echo 0)"
        else
            n="$(grep -o '"score"' <<<"$js" | wc -l)"
        fi
        [ "${n:-0}" -eq 0 ] && { review_idx+=("${keep_idx[$i]}"); review_paths+=("${keep_paths[$i]}"); }
    done
    [ "${#review_paths[@]}" -gt 0 ] || return 0

    local dcaps="$STATE/caps/$C_ID.detail.caps"
    printf '%s\n' "${review_paths[@]}" | "$IMG2PROMPT" --model-dir "$MODEL_DIR" \
        --task "$DETAIL_TASK" --beams 1 --max-new-tokens "$DETAIL_MAX_TOKENS" --batch \
        >"$dcaps" 2>"$STATE/logs/$C_ID.detail.err" || true
    mapfile -t DCAPS <"$dcaps"

    for i in "${!review_paths[@]}"; do
        p="${review_paths[$i]}"
        [ -f "$p" ] || continue
        cap="${DCAPS[$i]:-}"
        if [ -z "$cap" ]; then
            printf '%s\tno_caption_detail\n' "$p" >>"$skip"
            continue
        fi
        if ! is_person "$cap"; then
            move_to_others "$p" "no_person_detail" "$cap" "$report"
        elif is_partial "$cap"; then
            move_to_others "$p" "partial_body_detail" "$cap" "$report"
        fi
    done
}

process_chunk() {
    local chunk="$1"
    local id; id="$(basename "$chunk" .txt)"
    local caps="$STATE/caps/$id.caps"
    local report="$STATE/moved/$id.tsv"
    local skip="$STATE/skipped/$id.tsv"
    local err="$STATE/logs/$id.err"
    : >"$report"
    : >"$skip"

    mapfile -t paths <"$chunk"
    local -a imgs=()
    local p
    for p in "${paths[@]}"; do
        [ -f "$p" ] && imgs+=("$p")
    done
    [ "${#imgs[@]}" -eq 0 ] && return 0

    printf '%s\n' "${imgs[@]}" | "$IMG2PROMPT" --model-dir "$MODEL_DIR" \
        --task "$TASK" --beams 1 --max-new-tokens "$MAX_TOKENS" --batch \
        >"$caps" 2>"$err" || true

    mapfile -t capsarr <"$caps"
    local i=""
    for i in "${!imgs[@]}"; do
        p="${imgs[$i]}"
        [ -f "$p" ] || continue
        local cap="${capsarr[$i]:-}"
        if [ -z "$cap" ]; then
            printf '%s\tno_caption\n' "$p" >>"$skip"
            continue
        fi
        if ! is_person "$cap"; then
            move_to_others "$p" "no_person" "$cap" "$report"
        elif is_partial "$cap"; then
            move_to_others "$p" "partial_body" "$cap" "$report"
        fi
    done

    C_ID="$id"
    C_PATHS=("${imgs[@]}")
    C_CAPS=("${capsarr[@]}")
    detail_pass "$report" "$skip"
}

finalize() {
    local lock="$STATE/lock.finalize"
    exec 201>"$lock"
    flock 201
    [ -f "$STATE/finalized" ] && { flock -u 201; return 0; }
    if [ "${FORCE:-0}" != "1" ] && { [ -n "$(find "$STATE/queue" -maxdepth 1 -type f 2>/dev/null)" ] || \
                                     [ -n "$(find "$STATE/running" -maxdepth 1 -type f 2>/dev/null)" ]; }; then
        log "还有任务块未完成，拒绝收尾（确需强制用 FORCE=1 finalize）"
        flock -u 201
        return 1
    fi

    log "收尾：合并报告、清理空目录..."
    local merged="$STATE/moved_all.tsv"
    : >"$merged"
    local f
    for f in "$STATE"/moved/*.tsv; do
        [ -f "$f" ] && cat "$f" >>"$merged"
    done

    if [ -s "$merged" ]; then
        if [ ! -s "$OTHERS/_moved.tsv" ]; then
            printf '#src\tcaption\treason\tdst\n' >"$OTHERS/_moved.tsv"
        fi
        cat "$merged" >>"$OTHERS/_moved.tsv"
    fi

    # 汇总无法反推的图片，便于人工抽查（不移动）
    local sk="$STATE/skipped_all.tsv"
    : >"$sk"
    for f in "$STATE"/skipped/*.tsv; do
        [ -f "$f" ] && cat "$f" >>"$sk"
    done
    if [ -s "$sk" ]; then
        { printf '#src\treason\n'; cat "$sk"; } >"$OTHERS/_skipped.tsv"
    fi

    # 只删空的明星目录（others / 隐藏目录除外）
    find "$BASE" -mindepth 1 -maxdepth 1 -type d \
        ! -name others ! -name '.*' -empty -print -delete \
        >"$STATE/logs/removed_empty_dirs.log" 2>&1 || true

    {
        printf 'moved: %d\n' "$(wc -l <"$merged")"
        printf 'skipped(no_caption): %d\n' "$(wc -l <"$sk")"
        printf 'remaining_dirs: %d\n' "$(find "$BASE" -mindepth 1 -maxdepth 1 -type d ! -name '.*' ! -name others | wc -l)"
        printf 'remaining_files: %d\n' "$(find "$BASE" -mindepth 2 -maxdepth 2 -type f -not -path "$OTHERS/*" -not -path "$BASE/.*" | wc -l)"
        printf 'finished_at: %s\n' "$(date '+%F %T')"
    } >"$STATE/summary.txt"
    touch "$STATE/finalized"
    log "完成：$(tr '\n' ' ' <"$STATE/summary.txt")"
    flock -u 201
}

worker() {
    local wid="$1"
    log "worker $wid 启动"
    local lock="$STATE/lock.queue"
    while :; do
        exec 200>"$lock"
        flock 200
        local chunk
        chunk="$(find "$STATE/queue" -maxdepth 1 -type f -name 'chunk.*.txt' 2>/dev/null | sort | head -1)"
        if [ -z "$chunk" ]; then
            flock -u 200
            break
        fi
        local id; id="$(basename "$chunk" .txt)"
        mv -f -- "$chunk" "$STATE/running/$id.$$"
        flock -u 200
        local t0; t0=$(date +%s)
        log "worker $wid 领取 $id（$(wc -l <"$STATE/running/$id.$$") 张）"
        process_chunk "$STATE/running/$id.$$"
        rm -f "$STATE/running/$id.$$"
        touch "$STATE/done/$id"
        log "worker $wid 完成 $id（$(( $(date +%s) - t0 ))s）"
    done
    touch "$STATE/done/worker.$wid"
    local nworkers; nworkers="$(cat "$STATE/workers" 2>/dev/null || echo "$WORKERS")"
    if [ "$(find "$STATE/done" -maxdepth 1 -type f -name 'worker.*' | wc -l)" -ge "$nworkers" ]; then
        finalize
    fi
    log "worker $wid 退出"
}

start_all() {
    command -v flock >/dev/null || die "需要 flock"
    [ -x "$IMG2PROMPT" ] || die "img2prompt 不存在: $IMG2PROMPT"
    [ -d "$MODEL_DIR" ] || die "Florence-2 模型目录不存在: $MODEL_DIR"
    if [ "$DETAIL" = "1" ] && [ ! -x "$FACECLI" ]; then
        log "提示: facecli 不存在（$FACECLI），无人脸详描补判会自动跳过"
    fi
    mkdir -p "$OTHERS"

    if [ -s "$STATE/pids" ]; then
        local p alive=0
        while read -r p; do kill -0 "$p" 2>/dev/null && alive=1; done <"$STATE/pids"
        [ "$alive" -eq 1 ] && die "已有 worker 在运行，先 status/stop"
    fi

    # 全新一轮：保留 others/_moved.tsv，清空运行状态
    rm -rf "$STATE"
    mkdir -p "$STATE"/{queue,running,done,caps,moved,skipped,logs}
    echo "$WORKERS" >"$STATE/workers"
    : >"$STATE/pids"
    rm -f "$STATE/finalized"

    # 1) 汇总所有图片路径（排除 others 与隐藏目录）
    local list="$STATE/images.list"
    find "$BASE" -mindepth 2 -maxdepth 2 -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
           -o -iname '*.webp' -o -iname '*.bmp' -o -iname '*.gif' \) \
        -not -path "$OTHERS/*" -not -path "$BASE/.*" \
        | sort >"$list"
    local total; total="$(wc -l <"$list")"
    [ "$total" -gt 0 ] || die "没有找到图片"
    log "汇总图片: $total 张；切块 CHUNK=$CHUNK，WORKERS=$WORKERS"

    # 2) 切块进队列
    split -l "$CHUNK" -d -a 5 --additional-suffix=.txt "$list" "$STATE/queue/chunk."
    log "任务块: $(find "$STATE/queue" -type f | wc -l)"

    # 3) 拉起并发 worker（setsid 脱离当前会话）
    local i
    for ((i = 1; i <= WORKERS; i++)); do
        setsid bash "$SCRIPT" --worker "$i" >>"$STATE/logs/worker.$i.log" 2>&1 </dev/null &
        echo $! >>"$STATE/pids"
    done
    log "已启动 $WORKERS 个 worker，PID: $(tr '\n' ' ' <"$STATE/pids")"
    log "查看进度: bash $0 status    日志: $STATE/logs/"
}

show_status() {
    [ -d "$STATE" ] || { echo "没有运行状态（尚未 start）"; return 0; }
    local q r d mov
    q=$(find "$STATE/queue" -maxdepth 1 -type f 2>/dev/null | wc -l)
    r=$(find "$STATE/running" -maxdepth 1 -type f 2>/dev/null | wc -l)
    d=$(find "$STATE/done" -maxdepth 1 -type f -name 'chunk.*' 2>/dev/null | wc -l)
    mov=$(cat "$STATE"/moved/*.tsv 2>/dev/null | wc -l)
    echo "完成块: $d  运行中: $r  待处理: $q  已移动: $mov"
    if [ -s "$STATE/pids" ]; then
        local p alive=0
        while read -r p; do kill -0 "$p" 2>/dev/null && alive=$((alive + 1)); done <"$STATE/pids"
        echo "存活 worker: $alive"
    fi
    [ -f "$STATE/finalized" ] && echo "状态: 已完成" || echo "状态: 进行中"
    [ -f "$STATE/summary.txt" ] && sed 's/^/  /' "$STATE/summary.txt"
    [ -f "$STATE/logs/worker.1.log" ] && echo "--- worker.1 最近日志 ---" && tail -3 "$STATE/logs/worker.1.log"
}

stop_all() {
    [ -s "$STATE/pids" ] || { echo "没有记录 PID"; return 0; }
    local p
    while read -r p; do
        kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null
        echo "已停止进程组 $p"
    done <"$STATE/pids"
}

# 用上次缓存的 反推结果 + 最新规则重新判定（不重跑 Florence，秒级/分钟级）
reeval() {
    [ -d "$STATE/caps" ] || die "缺少 $STATE/caps（先用 start 跑过一轮）"
    [ -f "$STATE/images.list" ] || die "缺少 $STATE/images.list"
    mkdir -p "$OTHERS"
    local report="$STATE/reeval_moved.tsv"
    : >"$report"
    mapfile -t ALL <"$STATE/images.list"
    local total="${#ALL[@]}"
    local nchunks=$(( (total + CHUNK - 1) / CHUNK ))
    local c i idx capfile
    local moved=0 short_moved=0 kept=0 skipped=0
    for ((c = 0; c < nchunks; c++)); do
        capfile="$(ls "$STATE/caps/chunk.$(printf '%05d' "$c")."*.caps 2>/dev/null | head -1)"
        if [ -z "$capfile" ]; then
            log "reeval: 缺少 chunk $c 的缓存，跳过"
            continue
        fi
        mapfile -t CAPS <"$capfile"
        local -a cp=() cc=()
        local skipfile="$STATE/skipped/reeval.$(printf '%05d' "$c").tsv"
        : >"$skipfile"
        for ((i = 0; i < CHUNK; i++)); do
            idx=$(( c * CHUNK + i ))
            [ "$idx" -lt "$total" ] || break
            local p="${ALL[$idx]}"
            [ -f "$p" ] || { skipped=$((skipped + 1)); continue; }
            local cap="${CAPS[$i]:-}"
            if [ -z "$cap" ]; then skipped=$((skipped + 1)); continue; fi
            cp+=("$p"); cc+=("$cap")
            if ! is_person "$cap"; then
                move_to_others "$p" "no_person" "$cap" "$report"
                moved=$((moved + 1)); short_moved=$((short_moved + 1))
            elif is_partial "$cap"; then
                move_to_others "$p" "partial_body" "$cap" "$report"
                moved=$((moved + 1)); short_moved=$((short_moved + 1))
            else
                kept=$((kept + 1))
            fi
        done

        # 第二道：短描述判为"整人"但无人脸的图，用详描再判
        C_ID="refine.$(printf '%05d' "$c")"
        C_PATHS=("${cp[@]}")
        C_CAPS=("${cc[@]}")
        detail_pass "$report" "$skipfile"
        moved="$(wc -l <"$report")"
        kept=$(( kept - (moved - short_moved) ))
        short_moved="$moved"

        log "reeval: chunk $((c + 1))/$nchunks（累计新增 $moved）"
    done

    if [ -s "$report" ]; then
        if [ ! -s "$OTHERS/_moved.tsv" ]; then
            printf '#src\tcaption\treason\tdst\n' >"$OTHERS/_moved.tsv"
        fi
        cat "$report" >>"$OTHERS/_moved.tsv"
    fi
    find "$BASE" -mindepth 1 -maxdepth 1 -type d \
        ! -name others ! -name '.*' -empty -print -delete \
        >"$STATE/logs/reeval_empty.log" 2>&1 || true

    {
        printf 'reeval_added: %d\n' "$moved"
        printf 'kept: %d\n' "$kept"
        printf 'skipped_missing_or_nocap: %d\n' "$skipped"
        printf 'remaining_dirs: %d\n' "$(find "$BASE" -mindepth 1 -maxdepth 1 -type d ! -name '.*' ! -name others | wc -l)"
        printf 'remaining_files: %d\n' "$(find "$BASE" -mindepth 2 -maxdepth 2 -type f -not -path "$OTHERS/*" -not -path "$BASE/.*" | wc -l)"
        printf 'finished_at: %s\n' "$(date '+%F %T')"
    } >"$STATE/reeval_summary.txt"
    log "reeval 完成：新增移动 $moved，保留 $kept，跳过 $skipped"
}

case "${1:-}" in
    start)       start_all ;;
    status)      show_status ;;
    stop)        stop_all ;;
    finalize)    finalize ;;
    reeval)      reeval ;;
    --worker)    worker "${2:?缺少 worker id}" ;;
    *)           sed -n '2,27p' "$0"; exit 1 ;;
esac
