#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="cpp/sd/build:/opt/sd/build-dl/bin:/data/venv/onnxruntime-linux-x64-gpu-1.20.1/lib:${LD_LIBRARY_PATH:-}"
export GGML_BACKEND_PATH="/opt/sd/build-dl/bin/libggml-cuda.so"
exec "$DIR/comfycli-bin" "$@"
