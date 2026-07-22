#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="cpp/sd/build:${LD_LIBRARY_PATH:-}"
exec "$DIR/comfycli-bin" "$@"
