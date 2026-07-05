#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="/data/venv/lib/python3.12/site-packages/torch/lib:/opt/ReScheme:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="libpng16.so.16:${LD_PRELOAD:-}"
exec "$DIR/build/comfycli" "$@"
