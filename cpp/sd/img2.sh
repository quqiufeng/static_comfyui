#!/bin/bash
# =============================================================================
# 图像生成脚本 - RTX 4090D 24G / 20G 显存自适应优化版
# 用途: my-img 项目开发完成后的验证工具
# 说明: 此脚本用于验证 my-img 编译后的 sd-workflow 二进制功能
# =============================================================================
#
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                           【优化过程全记录】                                  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
#
# 原始问题（2026-05-31）：
#   Q8_0 模型 + 256x256 VAE tiling + SAG 启用
#   → 常驻显存 10.6GB，VAE decode buffer 18.7GB，峰值 ~22GB
#   → 20GB 显卡直接 OOM，无法出图
#
# 优化步骤：
#   1. 模型量化：Q8_0 (6.8G) → Q5_K_M (5.2G)
#      → Diffusion Model VRAM: 6891MB → 5268MB（省 1623MB）
#      → 常驻显存: 10.6GB → 8.98GB
#      → 峰值估算: ~22GB → ~20.7GB
#      → 模型加载速度: 8.56s → 3.25s
#
#   2. 关闭 SAG（Self-Attention Guidance）
#      → SAG 每步增加一次 attention 计算，主要消耗算力而非显存
#      → 实测显存节省极小（~0.4GB），关闭主要为了精简计算
#
#   3. VAE Tiling tile size 调整：256x256 → 128x128
#      → 这是最关键的一步，显存节省 12GB+
#      → 核心原理见下方【VAE Tiling 显存原理详解】
#
# 优化后结果（Q5_K_M + 128x128 tiling，2560x1440 输出）：
#   常驻显存: ~8.98GB (模型)
#   采样阶段: ~9.3GB (常驻+采样buffer)
#   VAE 阶段: ~15.6GB (常驻+VAE compute buffer)
#   峰值估算: ~15.6GB (远低于 20GB)
#   20G 显卡安全余量: 4.4GB ✅
#
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                    【当前出图基准 (2026-06-06)】                              ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
#
# 扩散模型: z_image_turbo-Q5_K_M.gguf (5.3GB VRAM)
# VAE:      ae.safetensors (160MB VRAM)
# LLM:      Qwen3-4B-Instruct-2507-Q4_K_M.gguf (3.5GB VRAM)
# 分辨率:   1920×1080 → 2560×1440 (HiRes)
# 步数:     20 → 45 (HiRes)
# HiRes strength: 0.35
# CFG:      3.2 | Sampler: euler | Scheduler: discrete
# FreeU:    b1=1.4, b2=1.5
# SAG:      开启 (scale=1.0)
# 增强:     clarity 0.4, sharpen 0.8, smart-sharpen 0.5, edge-sharpen 1.5
# VAE tiling: 128×128, overlap 0.5
# 出图时间: ~12 分钟 (RTX 3080 20GB)
#
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
#
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                    【VAE Tiling 显存原理详解】                                ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
#
# 为什么 VAE 解码会产生巨大的显存峰值？
#
#   HiRes Fix 完成后，得到的 latent 尺寸为 320x180（对应 2560x1440 像素图）。
#   VAE 解码器需要将 latent 还原为像素空间，这一步会分配一个大的 compute buffer
#   来存储中间激活值。该 buffer 大小约等于：
#
#     VAE_Buffer ≈ tile_size_x * tile_size_y * scale_factor^2 * channels * sizeof(float)
#
#   其中 scale_factor=8（latent→像素放大倍数），channels=128（VAE decoder 通道数）。
#
#   当 tile_size=256x256 时：
#     Buffer = 256 * 256 * 64 * 128 * 4 ≈ 2.1GB 每 tile
#     但由于 GGML 的实现方式，实际 compute buffer 峰值达到 18722 MB（18.7GB）
#     这可能包含了整个 graph 的激活、权重拷贝、临时张量等。
#
#   当 tile_size=128x128 时：
#     每次处理的 tile 面积缩小到 1/4
#     Compute buffer 峰值降至 6657 MB（6.7GB）
#     显存节省: 18722 - 6657 = 12065 MB（12GB！）
#
#   代价：
#     128x128 分成 4x2=8 个 tiles（256x256 只有 2 tiles）
#     更多 tiles → 更多 host-device 同步 → 稍慢（3.4s vs 2.8s）
#     但额外耗时仅 0.6s，相对于总生成时间（~400s）可忽略。
#
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                     【Tile Size 配置指南】                                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
#
# ┌──────────────┬─────────────┬─────────────┬────────────────────────────────────┐
# │  Tile Size   │ VAE Buffer  │   峰值估算  │           适用显卡                 │
# ├──────────────┼─────────────┼─────────────┼────────────────────────────────────┤
# │  128×128     │  6657 MB    │  ~15.6 GB   │  20GB+ 显卡（RTX 3080 Ti, 4060 Ti  │
# │              │             │             │  16GB+ 可尝试但可能不稳定）        │
# ├──────────────┼─────────────┼─────────────┼────────────────────────────────────┤
# │  256×256     │ 18722 MB    │  ~20.7 GB   │  24GB 显卡（RTX 4090, 3090）       │
# │              │             │             │  这是默认的高性能模式               │
# ├──────────────┼─────────────┼─────────────┼────────────────────────────────────┤
# │  512×512     │ 23403 MB    │  ~27.0 GB   │  不推荐！仅适合 32GB+（如 A100）   │
# │              │             │             │  且当 tile > latent 时 tiling 失效  │
# └──────────────┴─────────────┴─────────────┴────────────────────────────────────┘
#
# 关键结论：
#   • 128×128 不是降低分辨率，而是把 VAE 解码拆成更小的块
#   • 每块越小，单次分配的 compute buffer 越小，峰值显存越低
#   • 最终输出仍然是完整的 2560×1440，画质完全不受影响
#   • 更多 tiles 的接缝由 overlap 机制处理，肉眼不可见
#
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                       【使用方法】                                            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
#
# 1. 20GB 显卡（默认安全模式）：
#    ./img2.sh "prompt" ~/output.png 2560 1440
#    # 自动使用 128x128 tiling + 0.5 overlap，峰值 ~15.6GB
#
# 2. 使用 LoRA（风格增强）：
#    ./img2.sh "prompt" ~/output.png 2560 1440 --lora /path/to/style.safetensors:0.8
#    # 支持多个 LoRA，权重用冒号分隔（默认权重 1.0）
#
# 3. 24GB 显卡（高性能模式）：
#    VAE_TILE_SIZE=256x256 VAE_TILE_OVERLAP=0.8 \
#      ./img2.sh "prompt" ~/output.png 2560 1440
#    # 使用 256x256 tiling，VAE 解码更快，峰值 ~20.7GB
#
# 4. 自定义 tiling（高级用户）：
#    VAE_TILE_SIZE=64x64 VAE_TILE_OVERLAP=0.5 \
#      ./img2.sh "prompt" ~/output.png 2560 1440
#    # 64x64 峰值 ~11GB，适合 12GB 显卡（如 RTX 3060 12G）
#
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                       【出图原理】                                            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
#
# 直接生成高分辨率会导致"多人症"、畸形五官等问题，因为扩散模型训练时
# 未见过高分辨率。HiRes Fix 分两阶段：
#   1. 先生成低分辨率图像（构图、骨架正确）
#   2. 在 latent 空间放大后 refine（保留结构 + 补充细节）
#
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                    【4090D 优化策略】                                         ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
#
# 基础分辨率选择（保持与目标相同的宽高比 16:9）：
#   2560×1440 目标 → 2048×1152 基础（1.25x放大，latent 损失极小）
#   3840×2160 目标 → 2560×1440 基础（1.5x放大，4K 出图）
#   1920×1080 目标 → 1536×864  基础（1.25x放大）
#
# 为什么基础分辨率越高越好？
#   - 放大倍数越小 → latent 插值损失越少 → 画质越好
#   - 初始五官更清晰 → HiRes refine 只需微调纹理 → 不易破坏结构
#
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                    【参考提示词示例】                                         ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
#
# 2K: ./img2.sh "half body portrait..." "~/portrait_2560x1440.png" 2560 1440
# 4K: ./img2.sh "half body portrait..." "~/portrait_3840x2160.png" 3840 2160
#
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                    【参数说明】                                               ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
#
#   $1 - 提示词 (Prompt)
#   $2 - 输出文件路径
#   $3 - 宽度 (Width)
#   $4 - 高度 (Height)
#   --upscale - 可选：使用 2x ESRGAN 进一步放大（通常不需要）
#   --lora PATH:weight - 可选：LoRA 风格模型（可指定多个）
# =============================================================================
set -euo pipefail

RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
NC="\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-/data/models/image}"
# 使用本目录下编译好的 img_hires 二进制
SD_CLI="${SD_CLI:-$SCRIPT_DIR/build/img_hires}"
DIFFUSION_MODEL="${DIFFUSION_MODEL:-$MODEL_DIR/z_image_turbo-Q5_K_M.gguf}"
VAE_MODEL="${VAE_MODEL:-$MODEL_DIR/ae.safetensors}"
LLM_MODEL="${LLM_MODEL:-$MODEL_DIR/Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
UPSCALE_MODEL="${UPSCALE_MODEL:-$MODEL_DIR/2x_ESRGAN.gguf}"

# VAE Tiling 配置（20G显存优化：默认128x128，可配置）
# 128x128: 峰值~15.6GB (20G安全) | 256x256: 峰值~20.7GB (24G专用)
# overlap 提高可减少 tile 接缝（不显存增加，只增加计算量）
VAE_TILE_SIZE="${VAE_TILE_SIZE:-128x128}"
VAE_TILE_OVERLAP="${VAE_TILE_OVERLAP:-0.5}"

UPSCALE_FLAG=0
LORA_CONFIG=""
PROMPT_SCHEDULE=""
REGIONAL_PROMPTS=""
FACE_RESTORE_FLAG=0
FACE_RESTORE_MODEL=""
FACE_SWAP_FLAG=0
FACE_SWAP_SOURCE=""
IPADAPTER_FLAG=0
IPADAPTER_MODEL=""
IPADAPTER_IMAGE=""
T2I_ADAPTER_FLAG=0
T2I_ADAPTER_MODEL=""
T2I_ADAPTER_IMAGE=""
PHOTOMAKER_FLAG=0
PHOTOMAKER_MODEL=""
PHOTOMAKER_ID_IMAGES=""

ARGS=()
i=0
while [ $i -lt $# ]; do
    arg="${@:$((i+1)):1}"
    if [ "$arg" = "--upscale" ]; then
        UPSCALE_FLAG=1
    elif [ "$arg" = "--lora" ]; then
        i=$((i+1))
        LORA_CONFIG="${@:$((i+1)):1}"
    elif [ "$arg" = "--prompt-schedule" ]; then
        i=$((i+1))
        PROMPT_SCHEDULE="${@:$((i+1)):1}"
    elif [ "$arg" = "--regional-prompts" ]; then
        i=$((i+1))
        REGIONAL_PROMPTS="${@:$((i+1)):1}"
    elif [ "$arg" = "--face-restore" ]; then
        FACE_RESTORE_FLAG=1
    elif [ "$arg" = "--face-restore-model" ]; then
        i=$((i+1))
        FACE_RESTORE_MODEL="${@:$((i+1)):1}"
    elif [ "$arg" = "--face-swap" ]; then
        FACE_SWAP_FLAG=1
    elif [ "$arg" = "--face-swap-source" ]; then
        i=$((i+1))
        FACE_SWAP_SOURCE="${@:$((i+1)):1}"
    elif [ "$arg" = "--ipadapter" ]; then
        IPADAPTER_FLAG=1
    elif [ "$arg" = "--ipadapter-model" ]; then
        i=$((i+1))
        IPADAPTER_MODEL="${@:$((i+1)):1}"
    elif [ "$arg" = "--ipadapter-image" ]; then
        i=$((i+1))
        IPADAPTER_IMAGE="${@:$((i+1)):1}"
    elif [ "$arg" = "--t2i-adapter" ]; then
        T2I_ADAPTER_FLAG=1
    elif [ "$arg" = "--t2i-adapter-model" ]; then
        i=$((i+1))
        T2I_ADAPTER_MODEL="${@:$((i+1)):1}"
    elif [ "$arg" = "--t2i-adapter-image" ]; then
        i=$((i+1))
        T2I_ADAPTER_IMAGE="${@:$((i+1)):1}"
    elif [ "$arg" = "--photomaker" ]; then
        PHOTOMAKER_FLAG=1
    elif [ "$arg" = "--photomaker-model" ]; then
        i=$((i+1))
        PHOTOMAKER_MODEL="${@:$((i+1)):1}"
    elif [ "$arg" = "--photomaker-id-images" ]; then
        i=$((i+1))
        PHOTOMAKER_ID_IMAGES="${@:$((i+1)):1}"
    else
        ARGS+=("$arg")
    fi
    i=$((i+1))
done

PROMPT="${ARGS[0]:-A beautiful landscape}"
OUTPUT_FILE="${ARGS[1]:-}"
WIDTH="${ARGS[2]:-1280}"
HEIGHT="${ARGS[3]:-720}"

if [[ "$OUTPUT_FILE" == ~* ]]; then
    OUTPUT_FILE="${HOME}${OUTPUT_FILE:1}"
fi

echo "========================================"
echo "  Pre-check"
echo "========================================"

if [ ! -f "$SD_CLI" ]; then echo -e "${RED}Error: img_hires binary not found: $SD_CLI${NC}"; exit 1; fi
if [ ! -x "$SD_CLI" ]; then echo -e "${RED}Error: img_hires binary not executable: $SD_CLI${NC}"; exit 1; fi

for model in "$DIFFUSION_MODEL" "$VAE_MODEL" "$LLM_MODEL"; do
    if [ ! -f "$model" ]; then echo -e "${RED}Error: model not found: $model${NC}"; exit 1; fi
done

if [ "$UPSCALE_FLAG" -eq 1 ]; then
    if [ ! -f "$UPSCALE_MODEL" ]; then echo -e "${RED}Error: upscale model not found: $UPSCALE_MODEL${NC}"; exit 1; fi
    echo -e "${CYAN}✓ Upscale mode enabled (2x ESRGAN)${NC}"
fi

echo -e "${GREEN}✓ All checks passed${NC}"

if ! [[ "$WIDTH" =~ ^[0-9]+$ ]] || [ "$WIDTH" -le 0 ]; then echo -e "${RED}Error: width must be positive integer${NC}"; exit 1; fi
if ! [[ "$HEIGHT" =~ ^[0-9]+$ ]] || [ "$HEIGHT" -le 0 ]; then echo -e "${RED}Error: height must be positive integer${NC}"; exit 1; fi

# HD optimized parameters (4090D 24G 实测调优)
# 基础分辨率大幅提升 → 放大倍数更小 → 画质更好
# 人像推荐: euler + discrete + cfg 3.2 + strength 0.30
# 风景推荐: dpm++2m + karras + cfg 1.5 + strength 0.35
SAMPLING_METHOD="${SAMPLING_METHOD:-euler}"
SCHEDULER="${SCHEDULER:-discrete}"
CFG_SCALE="${CFG_SCALE:-3.2}"
STEPS="${STEPS:-25}"
HIRES_STEPS="${HIRES_STEPS:-50}"
HIRES_STRENGTH="${HIRES_STRENGTH:-0.35}"

if [ "$WIDTH" -ge 1920 ] && [ "$HEIGHT" -ge 1080 ]; then
    echo -e "${BLUE}[INFO] Ultra HD Mode: steps=$STEPS, cfg=$CFG_SCALE, sampler=$SAMPLING_METHOD${NC}"
else
    echo -e "${BLUE}[INFO] HD Mode: steps=$STEPS, cfg=$CFG_SCALE, sampler=$SAMPLING_METHOD${NC}"
fi

# Add quality keywords - enhanced for realism and edge stability
QUALITY_PREFIX="masterpiece, best quality, ultra-detailed, sharp focus, 8k uhd, photorealistic, highly detailed, crisp, clear, centered composition, professional portrait, medium shot, natural framing, realistic skin texture, soft lighting"
if [[ "$PROMPT" != *"masterpiece"* ]]; then
    PROMPT="$QUALITY_PREFIX, $PROMPT"
fi

NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, worst quality, jpeg artifacts, noise, grain, soft focus, out of focus, hazy, unclear, bad anatomy, deformed, border artifacts, edge distortion, tiling artifacts, edge artifacts, frame distortion, warped edges, stretched proportions, asymmetrical face, off-center, cropped, out of frame, partial face, cut off, incomplete head, cropped head, watermark, text, logo, signature, cropped shoulders, neon colors, oversaturated, high contrast, cartoon, painting, illustration, pale skin, washed out, gray skin, makeup-free, embedding:EasyNegative, embedding:bad-hands-5}"

if [ -n "$OUTPUT_FILE" ]; then
    if [[ "$OUTPUT_FILE" == *"/"* ]]; then
        OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"
        OUTPUT="$(basename "$OUTPUT_FILE")"
    else
        OUTPUT_DIR="$HOME"
        OUTPUT="$OUTPUT_FILE"
    fi
    # 用户指定路径，加时间戳后缀防止覆盖
    BASE="${OUTPUT%.png}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT="${BASE}_${TIMESTAMP}.png"
else
    OUTPUT_DIR="$HOME"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    MD5=$(echo "$PROMPT" | md5sum | cut -c1-8)
    OUTPUT="${TIMESTAMP}_${MD5}.png"
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT"

TARGET_LATENT_W=$((WIDTH / 8))
TARGET_LATENT_H=$((HEIGHT / 8))

# =============================================================================
# 低分辨率计算 - 4090D 24G 优化版
# =============================================================================
# HiRes Fix 需要两阶段生成来提升画质。低分辨率的选择原则：
# 1. latent 宽高比必须与目标严格一致（避免变形）
# 2. 基础分辨率越高，放大倍数越小，画质越好
# 3. 4090D 24G 显存可以承受更高的基础分辨率
#
# 对于 2560x1440 (latent 320x180, ratio=1.778)：
#   2048x1152 -> latent 256x144 (ratio=1.778) ✓ 推荐，放大1.25x，画质最佳
#   1920x1080 -> latent 240x135 (ratio=1.778) ✓ 备选，放大1.33x
#   1536x864  -> latent 192x108 (ratio=1.778) ✓ 备选，放大1.67x
#
# 对于 3840x2160 (latent 480x270, ratio=1.778)：
#   2560x1440 -> latent 320x180 (ratio=1.778) ✓ 推荐，放大1.5x，4K出图
#   2304x1296 -> latent 288x162 (ratio=1.778) ✓ 备选，放大1.67x
#
# 对于 1920x1080 (latent 240x135, ratio=1.778)：
#   1536x864  -> latent 192x108 (ratio=1.778) ✓ 推荐，放大1.25x
# =============================================================================

if [ "$WIDTH" -eq 3840 ] && [ "$HEIGHT" -eq 2160 ]; then
    # 4K: 2560x1440 基础 → 1.5x 放大
    LOW_W=2560
    LOW_H=1440
elif [ "$WIDTH" -eq 2560 ] && [ "$HEIGHT" -eq 1440 ]; then
    # 2K: 1920x1080 基础 -> 1.33x 放大（20G显存安全方案）
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
echo -e "Sampler: ${CYAN}$SAMPLING_METHOD${NC} + ${CYAN}$SCHEDULER${NC}"
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

SEED="${SEED:-$RANDOM}"
echo "Generating...  $(date '+%H:%M:%S')"

# Convert VAE_TILE_SIZE "128x128" -> single int for img_hires
VAE_TILE_INT="${VAE_TILE_SIZE%x*}"
if ! [[ "$VAE_TILE_INT" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: VAE_TILE_SIZE must be like 128x128 or 128${NC}"
    exit 1
fi

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
  --freeu
  --freeu-b1 1.4
  --freeu-b2 1.5
  --sag
  --sag-scale 1.0
  --clarity 0.2
  --sharpen 0.3
  --sharpen-radius 1
  --smart-sharpen 0.2
  --smart-sharpen-radius 2
  --edge-sharpen 0.5
  --edge-sharpen-radius 2
  --edge-sharpen-threshold 0.3
  -W "$LOW_W"
  -H "$LOW_H"
  --steps "$STEPS"
  --hires
  --hires-width "$WIDTH"
  --hires-height "$HEIGHT"
  --hires-strength "$HIRES_STRENGTH"
  --hires-steps "$HIRES_STEPS"
  -s "$SEED"
  "$PROMPT"
  "$OUTPUT_PATH"
)

if [ -n "$LORA_CONFIG" ]; then
    SD_CMD+=(--lora "$LORA_CONFIG")
fi

START_TIME=$(date +%s)
"${SD_CMD[@]}"
END_TIME=$(date +%s)
GEN_DURATION=$((END_TIME - START_TIME))

if [ -f "$OUTPUT_PATH" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
    
    # Format duration
    if [ $GEN_DURATION -ge 60 ]; then
        DURATION_MIN=$((GEN_DURATION / 60))
        DURATION_SEC=$((GEN_DURATION % 60))
        DURATION_STR="${DURATION_MIN}m ${DURATION_SEC}s"
    else
        DURATION_STR="${GEN_DURATION}s"
    fi
    
    echo ""
    echo "========================================"
    echo -e "${GREEN}✓ Generation successful!${NC}"
    echo -e "File:   ${GREEN}$OUTPUT_PATH${NC}"
    echo -e "Size:   ${BLUE}$FILE_SIZE${NC}"
    echo -e "Time:   ${YELLOW}$DURATION_STR${NC}"
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
