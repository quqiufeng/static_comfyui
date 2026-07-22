#!/bin/bash
# comfycli_remote.sh — 把本地编译好的 comfycli 部署到 Xiangongyun 远程 GPU 实例并运行
#
# 用法:
#   # 工作流模式
#   bash comfycli_remote.sh --workflow test_workflow.json --output-dir ./output --name comfy-4090
#
#   # prompt-only 模式
#   bash comfycli_remote.sh --checkpoint /data/models/image/sd_xl_base_1.0.safetensors \
#                           --prompt "a red apple" --output /tmp/apple.png \
#                           --name comfy-4090 --with-cuda
#
# 选项:
#   --name <name>           实例名称 (default: comfycli)
#   --workflow <json>       本地 workflow JSON 文件
#   --output-dir <dir>      远程/本地输出目录 (default: ./output)
#   --with-cuda             打包 CUDA Runtime 一起上传
#   --sync-models           自动 scp 模型文件到远程（默认只打印命令）
#   --run                   运行后立即下载输出图片
#   --checkpoint <path>     prompt-only 模式：模型路径
#   --prompt <text>         prompt-only 模式：文本
#   --negative <text>       prompt-only 模式：负向提示
#   --output <path>         prompt-only 模式：输出图片路径
#   --width, -W <int>
#   --height, -H <int>
#   --steps, -s <int>
#   --seed, -S <int>
#   --cfg, -C <float>
#   --sampler <name>
#   --scheduler <name>

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

debug() {
  echo "[comfycli_remote] $*"
}

# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------
NAME="comfycli"
WORKFLOW=""
OUTPUT_DIR="./output"
WITH_CUDA=0
SYNC_MODELS=0
RUN=0

CHECKPOINT=""
PROMPT=""
NEGATIVE=""
OUTPUT=""
WIDTH=1024
HEIGHT=1024
STEPS=20
SEED=42
CFG=7.0
SAMPLER="euler_a"
SCHEDULER="discrete"

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
INSTANCE_ID=""

while [ $# -gt 0 ]; do
  case "$1" in
    --name)
      NAME="$2"; shift 2 ;;
    --instance-id)
      INSTANCE_ID="$2"; shift 2 ;;
    --workflow)
      WORKFLOW="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --with-cuda)
      WITH_CUDA=1; shift ;;
    --sync-models)
      SYNC_MODELS=1; shift ;;
    --run)
      RUN=1; shift ;;
    --checkpoint)
      CHECKPOINT="$2"; shift 2 ;;
    --prompt)
      PROMPT="$2"; shift 2 ;;
    --negative)
      NEGATIVE="$2"; shift 2 ;;
    --output)
      OUTPUT="$2"; shift 2 ;;
    --width|-W)
      WIDTH="$2"; shift 2 ;;
    --height|-H)
      HEIGHT="$2"; shift 2 ;;
    --steps|-s)
      STEPS="$2"; shift 2 ;;
    --seed|-S)
      SEED="$2"; shift 2 ;;
    --cfg|-C)
      CFG="$2"; shift 2 ;;
    --sampler)
      SAMPLER="$2"; shift 2 ;;
    --scheduler)
      SCHEDULER="$2"; shift 2 ;;
    -h|--help)
      echo "see header comments"; exit 0 ;;
    *)
      echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------
if [ -n "$WORKFLOW" ] && [ -n "$CHECKPOINT" ]; then
  echo "错误: 不能同时指定 --workflow 和 --checkpoint"
  exit 1
fi
if [ -z "$WORKFLOW" ] && [ -z "$CHECKPOINT" ]; then
  echo "错误: 必须指定 --workflow 或 --checkpoint"
  exit 1
fi

for bin in ssh sshpass python3; do
  if ! command -v "$bin" &>/dev/null; then
    echo "错误: 需要 $bin，请先安装"
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# 1. 本地编译并打包
# ---------------------------------------------------------------------------
debug ">>> 本地编译 + 打包"
cd "$PROJECT_DIR"
bash build.sh
if [ "$WITH_CUDA" = "1" ]; then
  WITH_CUDA=1 bash deploy.sh
else
  bash deploy.sh
fi

# ---------------------------------------------------------------------------
# 2. 部署远程实例（如果没有提供 instance_id）
# ---------------------------------------------------------------------------
if [ -n "$INSTANCE_ID" ]; then
  debug ">>> 使用已有实例: $INSTANCE_ID"
else
  debug ">>> 部署 Xiangongyun 实例: $NAME"
  DEPLOY_OUT=$(python3 xgc_ctl.py deploy --name "$NAME")
  echo "$DEPLOY_OUT"
  INSTANCE_ID=$(echo "$DEPLOY_OUT" | grep '^instance_id=' | cut -d= -f2)
  if [ -z "$INSTANCE_ID" ]; then
    echo "错误: 部署失败，未获取 instance_id"
    exit 1
  fi
  debug "实例 ID: $INSTANCE_ID"
fi

# ---------------------------------------------------------------------------
# 3. 等待实例 running
# ---------------------------------------------------------------------------
debug ">>> 等待实例 $INSTANCE_ID 启动完成"
python3 xgc_ctl.py wait "$INSTANCE_ID" --status running

# ---------------------------------------------------------------------------
# 4. 获取 SSH 信息
# ---------------------------------------------------------------------------
debug ">>> 获取 SSH 信息"
SSH_INFO=$(python3 xgc_ctl.py info "$INSTANCE_ID")
SSH_HOST=$(echo "$SSH_INFO" | awk '/ssh_domain:/{print $2}')
SSH_PORT=$(echo "$SSH_INFO" | awk '/ssh_port:/{print $2}')
SSH_PASS=$(echo "$SSH_INFO" | awk '/password:/{print $2}')

if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ] || [ -z "$SSH_PASS" ]; then
  echo "错误: 无法获取 SSH 信息"
  exit 1
fi

SSH_CMD="sshpass -p '$SSH_PASS' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT root@$SSH_HOST"
SCP_CMD="sshpass -p '$SSH_PASS' scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $SSH_PORT"

# ---------------------------------------------------------------------------
# 5. 提取并同步模型文件
# ---------------------------------------------------------------------------
MODELS=()

if [ -n "$WORKFLOW" ]; then
  if [ ! -f "$WORKFLOW" ]; then
    echo "错误: workflow 文件不存在: $WORKFLOW"
    exit 1
  fi
  # 提取 workflow 里的模型路径
  while IFS= read -r m; do
    [ -n "$m" ] && MODELS+=("$m")
  done < <(python3 - "$WORKFLOW" <<'PY'
import json, sys
wf_path = sys.argv[1]
with open(wf_path) as f:
    wf = json.load(f)
keys = ("ckpt_name", "clip_l_name", "clip_g_name", "vae_name",
        "control_net_name", "clip_name1", "clip_name2")
models = set()
for node in wf.values():
    inputs = node.get("inputs", {})
    for k, v in inputs.items():
        if k in keys and isinstance(v, str) and v:
            if v.startswith("/"):
                models.add(v)
            else:
                models.add("/data/models/image/" + v)
for m in sorted(models):
    print(m)
PY
)
else
  # prompt-only 模式
  if [ -n "$CHECKPOINT" ] && [ -f "$CHECKPOINT" ]; then
    MODELS=("$CHECKPOINT")
  elif [ -n "$CHECKPOINT" ]; then
    echo "错误: checkpoint 文件不存在: $CHECKPOINT"
    exit 1
  fi
fi

if [ ${#MODELS[@]} -gt 0 ]; then
  debug ">>> 需要同步以下模型文件到远程 /data/models/image/"
  for m in "${MODELS[@]}"; do
    echo "    $m"
  done
  for m in "${MODELS[@]}"; do
    REMOTE_PATH="/data/models/image/$(basename "$m")"
    CMD="$SCP_CMD \"$m\" \"root@$SSH_HOST:$REMOTE_PATH\""
    if [ "$SYNC_MODELS" = "1" ]; then
      debug "执行: $CMD"
      eval "$CMD"
    else
      echo "[请手动执行] $CMD"
    fi
  done
  if [ "$SYNC_MODELS" = "0" ]; then
    echo "提示: 模型较大时建议手动确认；加 --sync-models 可自动执行上述 scp"
  fi
else
  debug ">>> 未检测到需要同步的模型文件"
fi

# ---------------------------------------------------------------------------
# 6. 上传 comfycli 部署包
# ---------------------------------------------------------------------------
debug ">>> 上传部署包到远程 /opt/comfycli/"
TARBALL=$(ls -t "$PROJECT_DIR/dist/comfycli_deploy"*.tar.gz | head -1)
if [ -z "$TARBALL" ]; then
  echo "错误: 找不到部署包"
  exit 1
fi
REMOTE_TARBALL="/opt/comfycli.tar.gz"
eval "$SCP_CMD \"$TARBALL\" \"root@$SSH_HOST:$REMOTE_TARBALL\""

# ---------------------------------------------------------------------------
# 7. 上传 workflow（工作流模式）
# ---------------------------------------------------------------------------
REMOTE_WORKFLOW=""
if [ -n "$WORKFLOW" ]; then
  REMOTE_WORKFLOW="/opt/comfycli/$(basename "$WORKFLOW")"
  eval "$SCP_CMD \"$WORKFLOW\" \"root@$SSH_HOST:$REMOTE_WORKFLOW\""
fi

# ---------------------------------------------------------------------------
# 8. 解压并在远程运行
# ---------------------------------------------------------------------------
debug ">>> 在远程解压并运行"
if [ -n "$WORKFLOW" ]; then
  REMOTE_CMD="rm -rf /opt/comfycli && mkdir -p /opt/comfycli && tar xzf $REMOTE_TARBALL -C /opt/comfycli && cd /opt/comfycli && bash run.sh $REMOTE_WORKFLOW --output-dir /opt/comfycli/$OUTPUT_DIR"
else
  REMOTE_CMD="rm -rf /opt/comfycli && mkdir -p /opt/comfycli && tar xzf $REMOTE_TARBALL -C /opt/comfycli && cd /opt/comfycli && bash run.sh --checkpoint '$CHECKPOINT' --prompt '$PROMPT' --negative '$NEGATIVE' --output '$OUTPUT' --width $WIDTH --height $HEIGHT --steps $STEPS --seed $SEED --cfg $CFG --sampler '$SAMPLER' --scheduler '$SCHEDULER'"
fi

eval "$SSH_CMD \"$REMOTE_CMD\""

# ---------------------------------------------------------------------------
# 9. 下载输出图片
# ---------------------------------------------------------------------------
if [ "$RUN" = "1" ]; then
  debug ">>> 下载远程输出到本地 $OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
  eval "$SCP_CMD \"root@$SSH_HOST:/opt/comfycli/$OUTPUT_DIR/*.png\" \"$OUTPUT_DIR/\"" || true
fi

debug ">>> 完成。实例 ID: $INSTANCE_ID"
debug "如需销毁止损: python3 xgc_ctl.py shutdown_destroy $INSTANCE_ID"
