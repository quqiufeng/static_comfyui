#!/bin/bash
# =============================================================================
# img_hires 封装脚本 — HiRes Fix 两阶段出图，VAE Tiling 显存自适应
# 用法: ./backup.sh ["prompt"] [output.png] [width] [height] [--flags...]
#       ./backup.sh 1440 1920                    # 全默认, 图落到 $HOME
#       ./backup.sh xhs.png 1440 1920            # 默认提示词 + 指定文件名
# 环境变量: VAE_TILE_SIZE, VAE_TILE_OVERLAP, CFG_SCALE, SAMPLING_METHOD 等
# =============================================================================
#
# 【默认配方（2026-09-27 更新）】
#   默认已切到「MIN 全低配」人像档（Q5 模型 / seed 时间戳随机 / 皮肤词负面已固化）:
#   CFG=2.0  Steps=16→30  HiRes strength=0.25  upscaler=latent-bicubic
#   clarity=0.0  edge-sharpen=0.0  Sampler=euler  Scheduler=discrete
#   效果: 皮肤白净柔和、无锐化痕迹（胜出图 ~/scan_MIN_q5_20260927_171626.png）。
#   默认提示词（2026-09-27 更新）: 白底 → 浅草绿纯色背景
#     （solid soft light green / sage green / clean seamless plain background, no props）。
#   注: 该档低于下方「甜点范围」下限, 属刻意柔化取向; 想要毛孔纹理可回
#       E1xMIN 中点档（CFG=3.0/20→40/0.4/latent-bislerp/clarity0.15,
#       2026-09-23 扫描, 图 zimage_p_E1xMIN.png）。
#   质量前缀: 带 img_hires 内置前缀（同本脚本文本, 单份）。
#     SKIP_QUALITY_PREFIX=1(默认) 只关脚本层添加, img_hires 仍会加同一份;
#     彻底不加前缀需给 img_hires 传 --no-quality-prefix（当前脚本未暴露）。
#
# 【甜点参数范围（每项单独扫描过的离散区间, 两括号内为安全值, 甜点在其中）】
#   cfg           3.0~3.5   (3.0~5.0)
#     原理: 提示词约束强度。z_image 蒸馏 turbo 模型, <2.5 皮肤纹理词失效发散;
#           >5.0 肖像易过饱和僵硬。
#   base steps    20        (16~25)
#   hires steps   40        (30~50)
#     原理: 二次采样细节量。蒸馏 turbo 不吃步数, 40→80 被否(更慢更差);
#           16→30 可用但细节不足, 甜点 20→40。
#   strength      0.4       (0.35~0.5)
#     原理: HiRes 二次改写幅度。0.25 皮肤无纹理(磨皮假感); 0.75 跑构图出噪点。
#   upscaler      bislerp   (latent-bislerp)
#     原理: latent 放大插值。bicubic 软到丢毛孔, 模型级(ESRGAN)高分辨率崩溃。
#   clarity       0.15      (0.15~0.3)
#     原理: 局部对比度。0 皮肤死平; 0.8 塑料过锐; 0.15~0.3 微纹理真实感。
#   edge-sharpen  0.0       (必须 0)
#     原理: 轮廓高通锐化。纯白背景剪影强, >0 必出白边/振铃 halo, 数码处理感最伤写实。
#
# 【本轮调试经验（14 张对比, seed 25630, 2560×1440, z_image_turbo-Q8）】
#   1. 决定性败笔是 edge-sharpen: 基准(3.5/25→50/0.5/bislerp/clarity0.3/edge2.0)被否,
#      与最优 E1 唯一差异就是 edge 2.0→0 — 白底人像轮廓锐化必出白边。
#   2. MIN(全低配: 2.0/16→30/0.25/bicubic/clarity0/edge0) 好看但皮肤过腻 —
#      真实皮肤纹理必须由 strength+clarity+bislerp+足量步数四项同时供给。
#   3. 综合 = E1 与 MIN 参数取中点(即本配方), 二者互补: E1 纹理真, MIN 柔自然。
#   4. FreeU 只对 UNet 生效（现已 1:1 对齐 ComfyUI FreeU_V2，见 cpp/sd/design.md）；
#      z_image/Qwen 是 DiT 依旧空操作，故本脚本不再传 --freeu。
#      DiT 的频域细节增强走 FreSca（模型无关, nodes_fresca.py 同款），脚本默认开启，FRESCA=0 关闭。
#   5. 蒸馏 turbo 模型步数收益低: 40→80 反而更差; z_image 走 discrete, Qwen 必须 flux。
#   6. 质量前缀(QUALITY_PREFIX)在宽画幅+close-up 会诱导主体复制, 需 SKIP_QUALITY_PREFIX=1。
#   7. ESRGAN(model) hires 内部路径高分辨率必崩(weight preparation), 只用 latent 路径。
#
# 【人像参数扫描 + 最终选定（2026-09-27, Q5, 2560×1440）】
#   10 组扫描(seed=timestamp, 图 ~/scan_c01..c10_20260927_*.png):
#     c01 默认(3.0/20→40/0.40/0.15/bislerp) c02 cfg3.5 c03 cfg4.0
#     c04 16→30  c05 25→50  c06 str0.35  c07 str0.50  c08 clarity0.30
#     c09 3.5/20→45/0.45/0.20  c10 latent-bicubic
#   4 组 MIN+质感微调(图 ~/texture_u1..u4_20260927_*.png):
#     u1 2.0/0.35/bicubic  u2 2.0/0.40/bislerp  u3 2.0/0.50/bislerp
#     u4 = c07 配方(3.0/20→40/0.50/0.15/bislerp, 质感参考)
#   ★最终选定: MIN 全低配 + Q5 模型（皮肤白净柔和, 无锐化痕迹; Q8 同参数不可辨, Q5 省 1.7G）
#     CFG=2.0  steps=16→30  strength=0.25  upscaler=latent-bicubic
#     clarity=0  edge-sharpen=0  sampler=euler  scheduler=discrete
#     胜出图 ~/scan_MIN_q5_20260927_171626.png（Q8 对照 ~/scan_MIN_repro_20260927_171118.png）
#     复现: CFG_SCALE=2.0 STEPS=16 HIRES_STEPS=30 HIRES_STRENGTH=0.25 \
#           HIRES_UPSCALER=latent-bicubic CLARITY=0 EDGE_SHARPEN=0 SEED=25630 \
#           cpp/sd/backup.sh "<默认肖像提示词>" ~/out.png 2560 1440
#   注: 下方默认值已按最终选定更新为 MIN 档; 想要纹理可显式回 E1xMIN 中点档。
#
# 【VAE Tiling 峰值参考】
#   Tile    | VAE Buffer | 峰值估算  | 适用显卡
#   128×128 |  6.7 GB    | ~15.6 GB  | 20G+ (RTX 3080 Ti / 4060 Ti)
#   256×256 | 18.7 GB    | ~20.7 GB  | 24G  (RTX 4090 / 3090)
#   512×512 | 23.4 GB    | ~27.0 GB  | 32G+ (A100, 不推荐)
#   默认 128x128。注意: adapter (sdcpp_adapter.cpp:599) 把 tile 硬夹到 128 latent
#   （= scale-8 VAE 输出 1024px），传 256 会被静默降回 128，"高性能模式"当前不生效。
#
# 【示例】
#   20G 卡:  ./backup.sh "portrait" ~/out.png 2560 1440
#   24G 卡:  VAE_TILE_SIZE=256x256 ./backup.sh "portrait" ~/out.png 2560 1440   # 现被夹到 128
#   LoRA:    ./backup.sh "prompt" ~/out.png 2560 1440 --lora style.safetensors:0.8
#
# 【采样加速（2026-09-24 实测，默认已开）】
#   RTX 3080 20G / 2560×1440 / E1xMIN 配方 seed=25630：
#     优化前 ~665s（11min）→ 优化后 ~210s（3.5min），约 3.2×。
#   1) EasyCache（主因，省 ~80% hires 采样）
#      原理：相邻步去噪结果变化小于阈值时，复用上一步 latent、跳过本步 UNet/DiT forward。
#      蒸馏 turbo（z_image）后期步变化极小 → 更易命中；base 跳 9/20，hires 跳 28–30/41。
#      接线：ImageGenerationParams.cache_* → sd_img_gen_params_t.cache → SampleCacheRuntime。
#      调参：CACHE_MODE=disabled|easycache|cache-dit|spectrum
#            CACHE_THRESHOLD 越低跳得越多（默认 0.2；0.15 更激进，0.3 更保守，过低画质漂）。
#   2) GGML_CUDA_GRAPHS=ON（build_sd_dl.sh）
#      原理：把一步采样的 CUDA kernel 序列录成 graph 一次提交，砍 launch 开销。
#      本例约再省数秒～十数秒；需重编 /opt/sd/build-dl。
#   3) 未做/评估中：batch CFG（z_image 断言 N==1，改模型层收益待测）、降 hires steps（画质换速度）。
#   分段计时看日志：Model loaded / generate wall / Post-processing / TOTAL wall / EasyCache skipped。
#   对照关缓存：CACHE_MODE=disabled ./backup.sh ...
# =============================================================================
set -euo pipefail

RED="\033[0;31m"; GREEN="\033[0;32m"; YELLOW="\033[1;33m"
BLUE="\033[0;34m"; CYAN="\033[0;36m"; NC="\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-/data/models/image}"
SD_CLI="${SD_CLI:-$SCRIPT_DIR/build/img_hires}"
SD_BACKEND_DIR="${SD_BACKEND_DIR:-/opt/sd/build-dl/bin}"

# 运行环境依赖内聚到脚本内, 外部无需再 export
export LD_LIBRARY_PATH="$SCRIPT_DIR/build:$SD_BACKEND_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GGML_BACKEND_PATH="${GGML_BACKEND_PATH:-$SD_BACKEND_DIR/libggml-cuda.so}"
DIFFUSION_MODEL="${DIFFUSION_MODEL:-$MODEL_DIR/z_image_turbo-Q5_K_M.gguf}"
VAE_MODEL="${VAE_MODEL:-$MODEL_DIR/ae.safetensors}"
LLM_MODEL="${LLM_MODEL:-$MODEL_DIR/Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
UPSCALE_MODEL="${UPSCALE_MODEL:-$MODEL_DIR/2x_ESRGAN.gguf}"
TARGET_RATIO=""

VAE_TILE_SIZE="${VAE_TILE_SIZE:-128x128}"
VAE_TILE_OVERLAP="${VAE_TILE_OVERLAP:-0.5}"

UPSCALE_FLAG=0; LORA_CONFIG=""; PROMPT_SCHEDULE=""; REGIONAL_PROMPTS=""
FACE_RESTORE_FLAG=0; FACE_RESTORE_MODEL=""
FACE_SWAP_FLAG=0; FACE_SWAP_SOURCE=""
IPADAPTER_FLAG=0; IPADAPTER_MODEL=""; IPADAPTER_IMAGE=""
T2I_ADAPTER_FLAG=0; T2I_ADAPTER_MODEL=""; T2I_ADAPTER_IMAGE=""
PHOTOMAKER_FLAG=0; PHOTOMAKER_MODEL=""; PHOTOMAKER_ID_IMAGES=""
ARGS=()

next_val() { i=$((i+1)); echo "${@:$((i+1)):1}"; }

i=0
while [ $i -lt $# ]; do
    arg="${@:$((i+1)):1}"
    case "$arg" in
        --upscale)          UPSCALE_FLAG=1 ;;
        --lora)             LORA_CONFIG=$(next_val "$@") ;;
        --prompt-schedule)  PROMPT_SCHEDULE=$(next_val "$@") ;;
        --regional-prompts) REGIONAL_PROMPTS=$(next_val "$@") ;;
        --face-restore)     FACE_RESTORE_FLAG=1 ;;
        --face-restore-model) FACE_RESTORE_MODEL=$(next_val "$@") ;;
        --face-swap)        FACE_SWAP_FLAG=1 ;;
        --face-swap-source) FACE_SWAP_SOURCE=$(next_val "$@") ;;
        --ipadapter)        IPADAPTER_FLAG=1 ;;
        --ipadapter-model)  IPADAPTER_MODEL=$(next_val "$@") ;;
        --ipadapter-image)  IPADAPTER_IMAGE=$(next_val "$@") ;;
        --t2i-adapter)      T2I_ADAPTER_FLAG=1 ;;
        --t2i-adapter-model) T2I_ADAPTER_MODEL=$(next_val "$@") ;;
        --t2i-adapter-image) T2I_ADAPTER_IMAGE=$(next_val "$@") ;;
        --photomaker)       PHOTOMAKER_FLAG=1 ;;
        --photomaker-model) PHOTOMAKER_MODEL=$(next_val "$@") ;;
        --photomaker-id-images) PHOTOMAKER_ID_IMAGES=$(next_val "$@") ;;
        *)                  ARGS+=("$arg") ;;
    esac
    i=$((i+1))
done

# 位置参数宽松解析: 纯数字 → width/height; 以 .png 结尾 → 输出文件名; 其余第一个 → prompt。
# 三项均可省略 —— 省略输出文件名时图自动写到 $HOME (见下方 OUTPUT_DIR 分支)。
_str=(); _num=()
for a in "${ARGS[@]}"; do
    if [[ "$a" =~ ^[0-9]+$ ]]; then _num+=("$a"); else _str+=("$a"); fi
done
PROMPT_ARG=""; OUTPUT_FILE=""
for a in "${_str[@]}"; do
    if [[ "$a" == *.png ]]; then [ -n "$OUTPUT_FILE" ] || OUTPUT_FILE="$a"
    elif [ -z "$PROMPT_ARG" ]; then PROMPT_ARG="$a"; fi
done

PROMPT="${PROMPT_ARG:-solo,single woman,half body portrait of a young woman, soft natural lighting, elegant pose, studio lighting, sharp eyes, solid soft light green background, sage green, clean seamless plain background, no props, flat solid color backdrop, fair skin, pale skin, smooth skin, matte skin, porcelain skin, flawless skin, medium close up}"
WIDTH="${_num[0]:-1280}"
HEIGHT="${_num[1]:-720}"
unset _str _num PROMPT_ARG

if [[ "$OUTPUT_FILE" == ~* ]]; then
    OUTPUT_FILE="${HOME}${OUTPUT_FILE:1}"
fi

die() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }
check_file() { [ -f "$1" ] || die "not found: $1"; }

[ -f "$SD_CLI" ] || die "img_hires not found: $SD_CLI"
[ -x "$SD_CLI" ] || die "img_hires not executable: $SD_CLI"
check_file "$DIFFUSION_MODEL"
check_file "$VAE_MODEL"
check_file "$LLM_MODEL"

[[ "$WIDTH" =~ ^[0-9]+$ ]] && [ "$WIDTH" -gt 0 ] || die "width must be positive integer"
[[ "$HEIGHT" =~ ^[0-9]+$ ]] && [ "$HEIGHT" -gt 0 ] || die "height must be positive integer"

if [ "$UPSCALE_FLAG" -eq 1 ]; then
    check_file "$UPSCALE_MODEL"
    echo -e "${CYAN}✓ Upscale mode enabled (2x ESRGAN)${NC}"
fi

echo -e "${GREEN}✓ All checks passed${NC}"

SAMPLING_METHOD="${SAMPLING_METHOD:-euler}"
SCHEDULER="${SCHEDULER:-discrete}"
CFG_SCALE="${CFG_SCALE:-2.0}"
STEPS="${STEPS:-16}"
HIRES_STEPS="${HIRES_STEPS:-30}"
HIRES_STRENGTH="${HIRES_STRENGTH:-0.25}"
# HiRes 上采样方式: latent-bicubic（默认，MIN 柔化档）| latent-bislerp（保细节）| model（ESRGAN 高分辨率会崩）
HIRES_UPSCALER="${HIRES_UPSCALER:-latent-bicubic}"
# 后处理（甜点见头部注释）: MIN 默认 clarity 0（需纹理可调 0.15）; edge-sharpen 必须 0（白底轮廓锐化出白边）
# sharpen / smart-sharpen 默认非 0（USM 0.3 + Sobel 加权 0.5），但 sharpen-threshold 无 CLI 入口
# → 阈值恒 0 = 全图锐化（平坦皮肤/纯色背景也加强）。想"真·无锐化痕迹"设 SHARPEN=0 SMART_SHARPEN=0。
CLARITY="${CLARITY:-0.0}"
EDGE_SHARPEN="${EDGE_SHARPEN:-0.0}"
SHARPEN="${SHARPEN:-0.3}"
SHARPEN_RADIUS="${SHARPEN_RADIUS:-1}"
SMART_SHARPEN="${SMART_SHARPEN:-0.5}"
SMART_SHARPEN_RADIUS="${SMART_SHARPEN_RADIUS:-2}"
# 彻底关质量前缀（img_hires 内置那份）。默认 0 = 维持现状（前缀仍生效）
NO_QUALITY_PREFIX="${NO_QUALITY_PREFIX:-0}"
# CPU 线程数（LLM 文本编码等）, 默认跟 img_hires 的 8
THREADS="${THREADS:-8}"
# 采样步缓存（EasyCache/DiT 步跳过）: 默认开启; CACHE_MODE=disabled 关闭
# CACHE_THRESHOLD 越低跳步越多（默认 0.2; 0.15 更激进, 0.3 更保守）
CACHE_MODE="${CACHE_MODE:-easycache}"
CACHE_THRESHOLD="${CACHE_THRESHOLD:-0.2}"
CACHE_START="${CACHE_START:-0.15}"
CACHE_END="${CACHE_END:-0.95}"
# FreSca 频域 guidance 增强（ComfyUI nodes_fresca.py 同款, DiT/UNet 模型无关）: 默认开启, FRESCA=0 关闭
# 低频 ×FRESCA_LOW / 高频 ×FRESCA_HIGH, latent 频率盒半宽 FRESCA_CUTOFF（默认 1.0/1.25/20）
FRESCA="${FRESCA:-1}"
FRESCA_LOW="${FRESCA_LOW:-1.0}"
FRESCA_HIGH="${FRESCA_HIGH:-1.25}"
FRESCA_CUTOFF="${FRESCA_CUTOFF:-20}"

echo -e "${BLUE}[INFO] $([ "$WIDTH" -ge 1920 ] && echo "Ultra HD" || echo "HD") Mode: steps=$STEPS, cfg=$CFG_SCALE, sampler=$SAMPLING_METHOD${NC}"

QUALITY_PREFIX="masterpiece, best quality, ultra-detailed, sharp focus, 8k uhd, photorealistic, highly detailed, crisp, clear, centered composition, professional portrait, medium shot, realistic skin texture, soft lighting"
# 脚本层前缀默认关（SKIP_QUALITY_PREFIX=1）；但 img_hires 仍会自加同一份（quality_prefix 默认开，
# 见 img_hires.cpp:361），所以最终 prompt 里前缀其实只有一份、且是 C++ 那份。
# 宽画幅 + close-up 加前缀会诱导主体复制 → 彻底对照需 NO_QUALITY_PREFIX=1（脚本据此传 --no-quality-prefix）。
if [ "${SKIP_QUALITY_PREFIX:-1}" != "1" ] && [[ "$PROMPT" != *"masterpiece"* ]]; then
    PROMPT="$QUALITY_PREFIX, $PROMPT"
fi

# E1xMIN 复现负面词: 基础负面 + 皮肤油腻词（防磨皮/油光）
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, worst quality, jpeg artifacts, noise, grain, soft focus, out of focus, hazy, unclear, bad anatomy, deformed, border artifacts, edge distortion, tiling artifacts, edge artifacts, frame distortion, warped edges, stretched proportions, asymmetrical face, off-center, cropped, out of frame, partial face, cut off, incomplete head, cropped head, watermark, text, logo, signature, cropped shoulders, oily skin, shiny skin, greasy skin, glossy skin, plastic skin, skin blemishes, embedding:EasyNegative, embedding:bad-hands-5}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "$OUTPUT_FILE" ]; then
    if [[ "$OUTPUT_FILE" == *"/"* ]]; then
        OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
        BASE=$(basename "$OUTPUT_FILE")
    else
        OUTPUT_DIR="$HOME"
        BASE="$OUTPUT_FILE"
    fi
    OUTPUT="${BASE%.png}_${TIMESTAMP}.png"
else
    OUTPUT_DIR="$HOME"
    MD5=$(echo "$PROMPT" | md5sum | cut -c1-8)
    OUTPUT="${TIMESTAMP}_${MD5}.png"
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_PATH="$(cd "$OUTPUT_DIR" && pwd)/$OUTPUT"

TARGET_LATENT_W=$((WIDTH / 8))
TARGET_LATENT_H=$((HEIGHT / 8))

# HiRes Fix 两阶段: 先低分辨率构图 → latent 放大 refine
# 原则: 基础分辨率越高越好（放大倍数小 → 画质好）
# 已知分辨率对: 3840x2160→2560x1440, 2560x1440→1920x1080, 1920x1080→1536x864

if [ "$WIDTH" -eq 3840 ] && [ "$HEIGHT" -eq 2160 ]; then
    # 4K: 2560x1440 基础 → 1.5x 放大
    LOW_W=2560
    LOW_H=1440
elif [ "$WIDTH" -eq 2560 ] && [ "$HEIGHT" -eq 1440 ]; then
    # 2K: 1920x1080 基础 → 1.33x 放大（20G显存安全方案）
    LOW_W=1920
    LOW_H=1080
elif [ "$WIDTH" -eq 1920 ] && [ "$HEIGHT" -eq 1080 ]; then
    # 1080p: 1536x864 基础 → 1.25x 放大
    LOW_W=1536
    LOW_H=864
elif [ "$WIDTH" -eq 1280 ] && [ "$HEIGHT" -eq 720 ]; then
    # 720p: 1024x576 基础 → 1.25x 放大
    LOW_W=1024
    LOW_H=576
else
    # 通用计算：使用目标分辨率的 80% 作为基础（4090D优化，更小放大倍数）
    LOW_LATENT_W=$((TARGET_LATENT_W * 4 / 5))
    LOW_LATENT_H=$((TARGET_LATENT_H * 4 / 5))
    
    # 对齐到 8 的倍数
    LOW_LATENT_W=$(((LOW_LATENT_W + 7) / 8 * 8))
    LOW_LATENT_H=$(((LOW_LATENT_H + 7) / 8 * 8))
    
    LOW_W=$((LOW_LATENT_W * 8))
    LOW_H=$((LOW_LATENT_H * 8))
fi

# 允许显式覆盖 base 分辨率（ESRGAN hires 2x 上采样时，base×2 需放得下显存）
if [ -n "${LOW_W_OVERRIDE:-}" ] && [ -n "${LOW_H_OVERRIDE:-}" ]; then
    LOW_W="$LOW_W_OVERRIDE"
    LOW_H="$LOW_H_OVERRIDE"
fi

# 保持比例的最小限制：只在单边小于512时按比例放大
if [ "$LOW_W" -lt 512 ] || [ "$LOW_H" -lt 512 ]; then
    TARGET_RATIO=$(echo "scale=6; $WIDTH / $HEIGHT" | bc)
    if [ "$LOW_W" -lt "$LOW_H" ]; then
        LOW_W=512
        LOW_H=$(echo "scale=0; $LOW_W / $TARGET_RATIO / 8 * 8" | bc)
        if [ "$LOW_H" -lt 512 ]; then LOW_H=512; fi
    else
        LOW_H=512
        LOW_W=$(echo "scale=0; $LOW_H * $TARGET_RATIO / 8 * 8" | bc)
        if [ "$LOW_W" -lt 512 ]; then LOW_W=512; fi
    fi
fi

echo ""
echo "========================================"
echo "  HD Image Generation"
echo "========================================"
echo -e "Target Size: ${GREEN}${WIDTH}x${HEIGHT}${NC}"
echo -e "Low-res Pass: ${GREEN}${LOW_W}x${LOW_H} -> ${WIDTH}x${HEIGHT}${NC}"
echo -e "Steps: $STEPS -> $HIRES_STEPS (HiRes)"
echo -e "CFG Scale: ${CYAN}$CFG_SCALE${NC}"
echo -e "HiRes Strength: $HIRES_STRENGTH"
echo -e "HiRes Upscaler: ${CYAN}$HIRES_UPSCALER${NC}"
echo -e "Sampler: ${CYAN}$SAMPLING_METHOD${NC} + ${CYAN}$SCHEDULER${NC}"
if [ "$CACHE_MODE" != "disabled" ]; then
    echo -e "Cache: ${CYAN}$CACHE_MODE${NC} threshold=$CACHE_THRESHOLD range=[$CACHE_START,$CACHE_END]"
fi
echo -e "Post: clarity=$CLARITY sharpen=$SHARPEN smart=$SMART_SHARPEN edge=$EDGE_SHARPEN | prefix: $([ "$NO_QUALITY_PREFIX" = "1" ] && echo off || echo on)"
if [ "$FRESCA" -eq 1 ]; then
    echo "FreSca: low=$FRESCA_LOW high=$FRESCA_HIGH cutoff=$FRESCA_CUTOFF"
fi
if [ "$UPSCALE_FLAG" -eq 1 ]; then
    UPSCALED_W=$((WIDTH * 2))
    UPSCALED_H=$((HEIGHT * 2))
    echo -e "Upscale: ${CYAN}2x ESRGAN -> ${UPSCALED_W}x${UPSCALED_H}${NC}"
fi
echo "----------------------------------------"
echo -e "Prompt: ${YELLOW}$PROMPT${NC}"
echo -e "Output: ${GREEN}$OUTPUT_PATH${NC}"
echo "========================================"
echo ""

SEED="${SEED:-$(date +%s)}"
echo "Generating...  $(date '+%H:%M:%S')"

# Convert VAE_TILE_SIZE "128x128" -> single int for img_hires
VAE_TILE_INT="${VAE_TILE_SIZE%x*}"
if ! [[ "$VAE_TILE_INT" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: VAE_TILE_SIZE must be like 128x128 or 128${NC}"
    exit 1
fi

# FreeU 只对 UNet 生效，z_image 是 DiT → 不传 --freeu（传了也是空操作）
SD_CMD=("$SD_CLI"
  --diffusion-model "$DIFFUSION_MODEL"
  --vae "$VAE_MODEL"
  --llm "$LLM_MODEL"
  --negative "$NEGATIVE_PROMPT"
  --cfg "$CFG_SCALE"
  --method "$SAMPLING_METHOD"
  --scheduler "$SCHEDULER"
  --diffusion-fa
  --vae-tiling
  --vae-tile-size "$VAE_TILE_INT"
  --vae-tile-overlap "$VAE_TILE_OVERLAP"
  --clarity "$CLARITY"
  --sharpen "$SHARPEN"
  --sharpen-radius "$SHARPEN_RADIUS"
  --smart-sharpen "$SMART_SHARPEN"
  --smart-sharpen-radius "$SMART_SHARPEN_RADIUS"
  --edge-sharpen "$EDGE_SHARPEN"
  --edge-sharpen-radius 2
  --edge-sharpen-threshold 0.3
  -t "$THREADS"
  -W "$LOW_W" -H "$LOW_H"
  --steps "$STEPS"
  --hires
  --hires-width "$WIDTH"
  --hires-height "$HEIGHT"
  --hires-strength "$HIRES_STRENGTH"
  --hires-steps "$HIRES_STEPS"
  --hires-upscaler "$HIRES_UPSCALER"
  --cache-mode "$CACHE_MODE"
  --cache-threshold "$CACHE_THRESHOLD"
  --cache-start "$CACHE_START"
  --cache-end "$CACHE_END"
  -s "$SEED"
  "$PROMPT"
  "$OUTPUT_PATH"
)

# 彻底关 img_hires 内置质量前缀（默认关, 维持既有配方）
if [ "$NO_QUALITY_PREFIX" = "1" ]; then
    SD_CMD+=(--no-quality-prefix)
fi

if [ "$FRESCA" -eq 1 ]; then
    SD_CMD+=(--fresca --fresca-low "$FRESCA_LOW" --fresca-high "$FRESCA_HIGH" --fresca-cutoff "$FRESCA_CUTOFF")
fi

if [ "$HIRES_UPSCALER" = "model" ]; then
    check_file "$UPSCALE_MODEL"
    SD_CMD+=(--hires-upscaler-model "$UPSCALE_MODEL")
fi

if [ -n "$LORA_CONFIG" ]; then
    SD_CMD+=(--lora "$LORA_CONFIG")
fi

if [ -n "$PROMPT_SCHEDULE" ]; then
    SD_CMD+=(--prompt-schedule "$PROMPT_SCHEDULE")
fi

if [ -n "$REGIONAL_PROMPTS" ]; then
    SD_CMD+=(--regional-prompts "$REGIONAL_PROMPTS")
fi

if [ "$FACE_RESTORE_FLAG" -eq 1 ]; then
    SD_CMD+=(--face-restore)
    if [ -n "$FACE_RESTORE_MODEL" ]; then
        SD_CMD+=(--face-restore-model "$FACE_RESTORE_MODEL")
    fi
fi

if [ "$FACE_SWAP_FLAG" -eq 1 ] && [ -n "$FACE_SWAP_SOURCE" ]; then
    SD_CMD+=(--face-swap --face-swap-source "$FACE_SWAP_SOURCE")
    SD_CMD+=(--face-swap-detection-model "$MODEL_DIR/yunet_320_320.onnx")
    SD_CMD+=(--face-swap-model "$MODEL_DIR/inswapper_128.onnx")
fi

if [ "$IPADAPTER_FLAG" -eq 1 ] && [ -n "$IPADAPTER_MODEL" ] && [ -n "$IPADAPTER_IMAGE" ]; then
    SD_CMD+=(--ipadapter --ipadapter-model "$IPADAPTER_MODEL")
    SD_CMD+=(--ipadapter-clip-vision "$MODEL_DIR/clip_vision_sd15.safetensors")
    SD_CMD+=(--ipadapter-image "$IPADAPTER_IMAGE")
fi

if [ "$T2I_ADAPTER_FLAG" -eq 1 ] && [ -n "$T2I_ADAPTER_MODEL" ] && [ -n "$T2I_ADAPTER_IMAGE" ]; then
    SD_CMD+=(--t2i-adapter --t2i-adapter-model "$T2I_ADAPTER_MODEL")
    SD_CMD+=(--t2i-adapter-image "$T2I_ADAPTER_IMAGE")
fi

if [ "$PHOTOMAKER_FLAG" -eq 1 ] && [ -n "$PHOTOMAKER_MODEL" ] && [ -n "$PHOTOMAKER_ID_IMAGES" ]; then
    SD_CMD+=(--photomaker --photomaker-model "$PHOTOMAKER_MODEL")
    SD_CMD+=(--photomaker-id-images "$PHOTOMAKER_ID_IMAGES")
fi

if [ "$UPSCALE_FLAG" -eq 1 ]; then
    SD_CMD+=(--upscale-model "$UPSCALE_MODEL")
    SD_CMD+=(--upscale-repeats 1)
    SD_CMD+=(--upscale-tile-size 1440)
fi

START_TIME=$(date +%s)
# 在后端目录运行: ggml 按 exe 目录/当前目录搜索 cpu 插件; 低显存时 auto-fit 会把 te/vae params 放 cpu
( cd "$SD_BACKEND_DIR" && "${SD_CMD[@]}" )
END_TIME=$(date +%s)
GEN_DURATION=$((END_TIME - START_TIME))

fmt_duration() {
    local s=$1
    [ $s -ge 60 ] && echo "$((s/60))m $((s%60))s" || echo "${s}s"
}

if [ -f "$OUTPUT_PATH" ]; then
    echo ""
    echo "========================================"
    echo -e "${GREEN}✓ Generation successful!${NC}"
    echo -e "File:   ${GREEN}$OUTPUT_PATH${NC}"
    echo -e "Size:   ${BLUE}$(du -h "$OUTPUT_PATH" | cut -f1)${NC}"
    echo -e "Time:   ${YELLOW}$(fmt_duration $GEN_DURATION)${NC}"
    echo -e "Seed:   ${YELLOW}$SEED${NC}"
    echo -e "CFG:    ${CYAN}$CFG_SCALE${NC}"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo -e "${RED}✗ Generation failed! Output file not found${NC}"
    echo "========================================"
    exit 1
fi
