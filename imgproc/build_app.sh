#!/bin/bash
# build_app.sh USER.static.py OUT — 把用户程序与 pil_lib 拼成一个独立 ELF
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
USER="$1"; OUT="$2"
BUNDLE="/tmp/staticpy-cache/$(basename "$OUT").bundle.static.py"
mkdir -p /tmp/staticpy-cache
python3 "$HERE/concat.py" "$HERE/pil_lib.static.py" "$USER" > "$BUNDLE"
( cd "$ROOT" && bash staticpy/static_build_comfycli.sh "$BUNDLE" "$OUT" "$HERE/imgproc_ffi.scm" )
echo "built: $OUT  (run: LD_LIBRARY_PATH=$ROOT/cpp/imgproc/build ./$OUT ...)"
