#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="/data/venv/lib/python3.12/site-packages/torch/lib:/opt/ReScheme:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
exec "$DIR/build/comfycli" "$@"
