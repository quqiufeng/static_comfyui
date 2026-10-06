#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="cpp/sd/build:$DIR/cpp/florence2:$DIR/cpp/birefnet:/data/venv/onnxruntime-linux-x64-gpu-1.26.0/lib:/opt/sd/build-dl/bin:${LD_LIBRARY_PATH:-}"
export GGML_BACKEND_PATH="/opt/sd/build-dl/bin/libggml-cuda.so"
exec "$DIR/comfycli-bin" "$@"
