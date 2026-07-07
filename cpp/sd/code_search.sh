#!/bin/bash
# =============================================================================
# code_search.sh - /opt/sd 代码语义搜索工具
# =============================================================================
# 用途：
#   1. 索引 /opt/sd (stable-diffusion.cpp) 源码
#   2. 将索引导入 KV Cache (/memory)
#   3. 使用自然语言搜索 /opt/sd 代码
#
# 依赖：
#   - /opt/my_db/ai_code_search.sh
#   - /opt/my_db/tools/cache_import
#   - /opt/my_db/tools/cache_query
# =============================================================================

set -euo pipefail

# 配置
SD_DIR="${SD_DIR:-/opt/sd}"
CACHE_DIR="${CACHE_DIR:-/opt/code_caches/sd_cache}"
NAMESPACE="${NAMESPACE:-/code/sd}"
MY_DB="${MY_DB:-/opt/my_db}"
JOBS="${JOBS:-$(nproc)}"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

usage() {
    cat <<EOF
用法: $0 <command> [args]

命令:
  index              索引 /opt/sd 源码（生成 chunks + 向量）
  import             将索引导入 KV Cache (/memory)
  full               一键执行 index + import
  search "<query>"   语义搜索 /opt/sd 代码
  context <symbol>   查看某个符号的调用上下文
  demo               运行几个示例搜索
  status             查看索引和缓存状态

示例:
  $0 index
  $0 import
  $0 full
  $0 search "UNet forward cross attention"
  $0 search "VAE decode tiling" --max-results 5
  $0 context CrossAttention::forward
  $0 context generate_image
EOF
}

# 检查依赖
check_deps() {
    local missing=0
    for cmd in "${MY_DB}/ai_code_search.sh" "${MY_DB}/tools/cache_import" "${MY_DB}/tools/cache_query"; do
        if [ ! -f "$cmd" ]; then
            echo -e "${RED}Error: missing $cmd${NC}"
            missing=1
        fi
    done
    if [ ! -d "${SD_DIR}" ]; then
        echo -e "${RED}Error: ${SD_DIR} not found${NC}"
        missing=1
    fi
    if [ "$missing" -eq 1 ]; then
        exit 1
    fi
}

# 索引源码
cmd_index() {
    echo "========================================"
    echo -e "${CYAN}Step 1: 索引 /opt/sd 源码${NC}"
    echo "========================================"
    echo "命令:"
    echo "  ${MY_DB}/ai_code_search.sh analyze ${SD_DIR} ${CACHE_DIR}"
    echo ""

    if [ -d "${CACHE_DIR}/vectors" ] && [ -f "${CACHE_DIR}/chunks_meta.jsonl" ]; then
        echo -e "${YELLOW}⚠ 缓存目录已存在: ${CACHE_DIR}${NC}"
        read -r -p "是否重新索引? (y/N) " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo -e "${GREEN}跳过索引${NC}"
            return 0
        fi
        rm -rf "${CACHE_DIR}"
    fi

    mkdir -p "${CACHE_DIR}"
    "${MY_DB}/ai_code_search.sh" analyze "${SD_DIR}" "${CACHE_DIR}"

    echo ""
    echo -e "${GREEN}✓ 索引完成${NC}"
    echo "输出文件:"
    echo "  ${CACHE_DIR}/chunks_text.txt"
    echo "  ${CACHE_DIR}/chunks_meta.jsonl"
    echo "  ${CACHE_DIR}/call_graph.json"
    echo "  ${CACHE_DIR}/dataflow.json"
    echo "  ${CACHE_DIR}/vectors/code_local_sd.jina.bin"
    echo "  ${CACHE_DIR}/vectors/code_local_sd.jina.bin.hnsw"
    echo ""
}

# 导入 KV Cache
cmd_import() {
    echo "========================================"
    echo -e "${CYAN}Step 2: 导入 KV Cache${NC}"
    echo "========================================"
    echo "命令:"
    echo "  ${MY_DB}/tools/cache_import ${CACHE_DIR} ${NAMESPACE}"
    echo ""

    if [ ! -f "${CACHE_DIR}/chunks_meta.jsonl" ]; then
        echo -e "${RED}Error: 请先运行 $0 index${NC}"
        exit 1
    fi

    "${MY_DB}/tools/cache_import" "${CACHE_DIR}" "${NAMESPACE}"

    echo ""
    echo -e "${GREEN}✓ 导入完成${NC}"
    echo "命名空间: ${NAMESPACE}"
    echo ""
}

# 一键执行
cmd_full() {
    cmd_index
    cmd_import
    echo "========================================"
    echo -e "${GREEN}✓ /opt/sd 已可搜索${NC}"
    echo "========================================"
    echo ""
    echo "试试："
    echo "  $0 search \"UNet forward cross attention\""
    echo "  $0 search \"VAE decode tiling\""
    echo "  $0 context generate_image"
}

# 语义搜索
cmd_search() {
    local query="${1:-}"
    local max_results="${2:-10}"

    if [ -z "$query" ]; then
        echo -e "${RED}Error: 需要提供搜索 query${NC}"
        echo "  $0 search \"你的搜索词\""
        exit 1
    fi

    echo "========================================"
    echo -e "${CYAN}语义搜索: ${query}${NC}"
    echo "========================================"
    echo "命令:"
    echo "  ${MY_DB}/tools/cache_query \"${query}\" \\"
    echo "    --repo ${NAMESPACE} --type search --analysis-dir ${CACHE_DIR}"
    echo ""

    "${MY_DB}/tools/cache_query" "$query" \
        --repo "${NAMESPACE}" \
        --type search \
        --analysis-dir "${CACHE_DIR}" \
        --max-results "$max_results"
}

# 符号上下文
cmd_context() {
    local symbol="${1:-}"

    if [ -z "$symbol" ]; then
        echo -e "${RED}Error: 需要提供符号名${NC}"
        echo "  $0 context CrossAttention::forward"
        exit 1
    fi

    echo "========================================"
    echo -e "${CYAN}符号上下文: ${symbol}${NC}"
    echo "========================================"
    echo "命令:"
    echo "  ${MY_DB}/tools/cache_query /${NAMESPACE}/symbols/${symbol} --type context --depth 2"
    echo ""

    # 精确符号查询（推荐）
    "${MY_DB}/tools/cache_query" "${NAMESPACE}/symbols/${symbol}" \
        --type context --depth 2 2>/dev/null || \
    # 回退到语义搜索
    "${MY_DB}/tools/cache_query" "$symbol" \
        --repo "${NAMESPACE}" \
        --type search \
        --analysis-dir "${CACHE_DIR}" \
        --max-results 5
}

# 运行示例
cmd_demo() {
    echo "========================================"
    echo -e "${CYAN}示例搜索${NC}"
    echo "========================================"

    local queries=(
        "UNet forward cross attention"
        "VAE decode tiling latent"
        "text encoder embedding CLIP"
        "sample method euler scheduler"
        "model loader safetensors gguf"
        "FreeU unet backbone skip"
    )

    for q in "${queries[@]}"; do
        echo ""
        echo -e "${BLUE}Query: $q${NC}"
        echo "----------------------------------------"
        "${MY_DB}/tools/cache_query" "$q" \
            --repo "${NAMESPACE}" \
            --type search \
            --analysis-dir "${CACHE_DIR}" \
            --max-results 3
    done
}

# 状态检查
cmd_status() {
    echo "========================================"
    echo -e "${CYAN}状态检查${NC}"
    echo "========================================"
    echo "SD_DIR:      ${SD_DIR}"
    echo "CACHE_DIR:   ${CACHE_DIR}"
    echo "NAMESPACE:   ${NAMESPACE}"
    echo ""

    if [ -d "${SD_DIR}" ]; then
        echo -e "${GREEN}✓${NC} 源码目录存在"
        cd "${SD_DIR}"
        echo "  commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    else
        echo -e "${RED}✗${NC} 源码目录不存在"
    fi

    if [ -f "${CACHE_DIR}/chunks_meta.jsonl" ]; then
        local chunks=$(wc -l < "${CACHE_DIR}/chunks_meta.jsonl")
        echo -e "${GREEN}✓${NC} 已索引: ${chunks} chunks"
    else
        echo -e "${RED}✗${NC} 未索引"
    fi

    if [ -f "${CACHE_DIR}/vectors/code_local_sd.jina.bin.hnsw" ]; then
        echo -e "${GREEN}✓${NC} HNSW 向量索引已生成"
    else
        echo -e "${RED}✗${NC} 向量索引未生成"
    fi

    echo ""
    echo "KV Cache 检查（/memory 中是否有 ${NAMESPACE} 数据）："
    # 检查 /memory/index.bin 是否包含该 namespace
    if "${MY_DB}/tools/cache_query" "${NAMESPACE}/_meta/info" --type exact >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} KV Cache 已导入"
    else
        echo -e "${YELLOW}⚠${NC} 无法确认 KV Cache 状态（或未导入）"
    fi
}

# 主入口
main() {
    local cmd="${1:-help}"
    shift || true

    case "$cmd" in
        index)
            check_deps
            cmd_index
            ;;
        import)
            check_deps
            cmd_import
            ;;
        full)
            check_deps
            cmd_full
            ;;
        search)
            check_deps
            cmd_search "$@"
            ;;
        context)
            check_deps
            cmd_context "$@"
            ;;
        demo)
            check_deps
            cmd_demo
            ;;
        status)
            cmd_status
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            echo -e "${RED}Error: unknown command: $cmd${NC}"
            usage
            exit 1
            ;;
    esac
}

main "$@"
