#!/bin/bash
# build.sh — ComfyCLI 完整编译流水线
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATICPY_DIR="$PROJECT_DIR/staticpy"
CPP_DIR="$PROJECT_DIR/cpp"
OUT_DIR="$PROJECT_DIR/out"

# 编译 libtorch_std_helper.so
echo "=== Step 1: Build libtorch_std_helper.so ==="
bash "$CPP_DIR/build_torch_std_helper.sh"

# 编译 ComfyCLI
echo "=== Step 2: Translate + AOT + ELF ==="
bash "$STATICPY_DIR/static_build.sh" "$PROJECT_DIR/main.static.py" comfycli

echo ""
echo "=== Build Complete ==="
echo "  $PROJECT_DIR/comfycli       (ELF binary)"
echo "  $PROJECT_DIR/comfycli.so    (AOT compiled Scheme)"
echo "  $CPP_DIR/libtorch_std_helper.so  (C++ bridge)"
echo ""
echo "Run:"
echo "  LD_LIBRARY_PATH=$CPP_DIR:/data/venv/lib/python3.12/site-packages/torch/lib \\
echo "    ./comfycli workflow.json --output-dir ./output"
