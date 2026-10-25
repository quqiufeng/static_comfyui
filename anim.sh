#!/bin/bash
# anim.sh —— 用 static-ComfyUI(comfycli-bin) 跑 AnimateDiff 动画（走 sd.cpp，零 torch）
# 用法: ./anim.sh ["<prompt>"] [outdir] [W H] [frames fps] [steps cfg seed]
# 例:   ./anim.sh "a cute cartoon knight walking right, side view, white background" ~/anim 512 512 16 8 20 8.0 42
# 环境: MODEL / VAE / MOTION / NEG / NODE_SET 可覆盖
#   NODE_SET=evolved(默认, ADE_* 节点) | basic(AnimateDiffLoader/Video)
#   运动 LoRA:  LORA=<name|abs path>  LORA_STRENGTH=1.0
#   逐帧控肢体: CONTROL_NET=<controlnet name|abs path>  CONTROL_DIR=<openpose 帧目录>  CONTROL_STRENGTH=1.0
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL="${MODEL:-/data/models/image/video/sd15.safetensors}"
VAE="${VAE:-/data/models/image/video/vae-ft-mse.safetensors}"
MOTION="${MOTION:-/data/models/image/video/mm_v2_prefixed.safetensors}"
NODE_SET="${NODE_SET:-evolved}"
NEG="${NEG:-blurry, low quality, deformed limbs, extra limbs, static}"
LORA="${LORA:-}"
LORA_STRENGTH="${LORA_STRENGTH:-1.0}"
CONTROL_NET="${CONTROL_NET:-}"
CONTROL_DIR="${CONTROL_DIR:-}"
CONTROL_STRENGTH="${CONTROL_STRENGTH:-1.0}"
DTYPE="${DTYPE:-}"
IPADAPTER="${IPADAPTER:-}"
CLIP_VISION="${CLIP_VISION:-}"
IPADAPTER_IMAGE="${IPADAPTER_IMAGE:-}"
IPADAPTER_STRENGTH="${IPADAPTER_STRENGTH:-0.8}"

PROMPT="${1:-a cute cartoon knight walking to the right, side view, plain white background, full body}"
OUTD="${2:-$HOME/iso_gen/anim_out}"
W="${3:-512}"; H="${4:-512}"; FRAMES="${5:-16}"; FPS="${6:-8}"
STEPS="${7:-20}"; CFG="${8:-8.0}"; SEED="${9:-42}"

[ -x "$DIR/comfycli-bin" ] || { echo "找不到 comfycli-bin"; exit 1; }
mkdir -p "$OUTD"

# 采样器附加输入（运动 LoRA / ControlNet / 逐帧 control）
EXTRA=""
[ -n "$LORA" ] && EXTRA="$EXTRA,\"lora\":\"$LORA\",\"lora_strength\":$LORA_STRENGTH"
[ -n "$CONTROL_NET" ] && EXTRA="$EXTRA,\"control_net\":\"$CONTROL_NET\""
if [ -n "$CONTROL_DIR" ]; then
  CF="$(ls "$CONTROL_DIR"/*.png 2>/dev/null | sort | paste -sd, -)"
  [ -n "$CF" ] && EXTRA="$EXTRA,\"control_frames\":\"$CF\",\"control_strength\":$CONTROL_STRENGTH"
fi

if [ "$NODE_SET" = "basic" ]; then
  LOADER="AnimateDiffLoader"; SAMPLER="AnimateDiffVideo"; CTXOPT=""
else
  LOADER="ADE_AnimateDiffLoaderWithContext"; SAMPLER="ADE_AnimateDiffSampler"
  CTXOPT='  "9": {"class_type":"ADE_AnimateDiffUniformContextOptions","inputs":{"context_length":16,"context_stride":1,"context_overlap":4,"fuse_method":"flat"}},'
fi

WF="$(mktemp)"
{
  echo '{'
  echo "  \"1\": {\"class_type\":\"CheckpointLoaderSimple\",\"inputs\":{\"ckpt_name\":\"$MODEL\",\"vae_name\":\"$VAE\"}},"
  [ -n "$CTXOPT" ] && echo "$CTXOPT"
  if [ -n "$DTYPE" ]; then
    echo "  \"8\": {\"class_type\":\"ModelComputeDtype\",\"inputs\":{\"model\":[\"1\",0],\"dtype\":\"$DTYPE\"}},"
    BASE_REF='["8",0]'
  else
    BASE_REF='["1",0]'
  fi
  if [ -n "$CTXOPT" ]; then
    echo "  \"2\": {\"class_type\":\"$LOADER\",\"inputs\":{\"model\":$BASE_REF,\"motion_model\":\"$MOTION\",\"context_options\":[\"9\",0]}},"
    MODELREF='["2",0]'
  else
    echo "  \"2\": {\"class_type\":\"$LOADER\",\"inputs\":{\"model\":$BASE_REF,\"motion_module\":\"$MOTION\"}},"
    MODELREF='["2",0]'
  fi
  if [ -n "$IPADAPTER" ]; then
    echo "  \"10\": {\"class_type\":\"IPAdapterApply\",\"inputs\":{\"model\":$MODELREF,\"ipadapter_model\":\"$IPADAPTER\",\"clip_vision_model\":\"$CLIP_VISION\",\"image_path\":\"$IPADAPTER_IMAGE\",\"weight\":$IPADAPTER_STRENGTH}},"
    MODELREF='["10",0]'
  fi
  echo "  \"3\": {\"class_type\":\"$SAMPLER\",\"inputs\":{\"model\":$MODELREF,\"positive\":\"$PROMPT\",\"negative\":\"$NEG\",\"width\":$W,\"height\":$H,\"steps\":$STEPS,\"cfg\":$CFG,\"seed\":$SEED,\"frames\":$FRAMES,\"fps\":$FPS,\"sampler_name\":\"euler\",\"scheduler\":\"discrete\",\"output_dir\":\"$OUTD\"$EXTRA}}"
  echo '}'
} > "$WF"

echo "[anim] $SAMPLER · ${W}x${H} · ${FRAMES}帧@${FPS}fps · steps=$STEPS cfg=$CFG seed=$SEED"
[ -n "$LORA" ] && echo "[anim] motion LoRA: $LORA (strength $LORA_STRENGTH)"
[ -n "$CONTROL_DIR" ] && echo "[anim] control: $CONTROL_NET dir=$CONTROL_DIR strength=$CONTROL_STRENGTH"
cd "$DIR"
LD_LIBRARY_PATH="$DIR/cpp/sd/build:/opt/sd/build-dl/bin" \
GGML_BACKEND_PATH=/opt/sd/build-dl/bin/libggml-cuda.so \
./comfycli-bin "$WF" --output-dir "$OUTD"
rm -f "$WF"

if command -v ffmpeg >/dev/null 2>&1 && ls "$OUTD"/frame_*.png >/dev/null 2>&1; then
  ffmpeg -y -framerate "$FPS" -i "$OUTD/frame_%03d.png" \
    -vf "split[a][b];[a]palettegen[p];[b][p]paletteuse" -loop 0 "$OUTD/anim.gif" >/dev/null 2>&1 || true
  echo "[anim] GIF: $OUTD/anim.gif"
fi
echo "[anim] frames: $OUTD"
