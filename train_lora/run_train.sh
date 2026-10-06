#!/bin/bash
# =============================================================================
# run_train.sh — 一键：风格图目录 → 自动打标 → 训练 z_image LoRA
#
# 用法:
#   bash run_train.sh <风格图目录> [触发词] [数据集目录]
#   bash run_train.sh /path/to/style_imgs mystyle
#
# 环境变量（透传 train_lora.sh）: RANK / STEPS / RES / OUT
# =============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/data/venv/bin

SRC="${1:?用法: bash run_train.sh <风格图目录> [触发词] [数据集目录]}"
TRIGGER="${2:-mystyle}"
DATASET="${3:-/data/datasets/$(basename "$SRC")_lora}"

echo ">>> [1/2] 打标 + 整理数据集: $SRC -> $DATASET"
"$PY/python" "$DIR/prepare_dataset.py" "$SRC" "$DATASET" --trigger "$TRIGGER" --caption

echo ""
echo ">>> [2/2] 训练"
DATA="$DATASET" bash "$DIR/train_lora.sh"
