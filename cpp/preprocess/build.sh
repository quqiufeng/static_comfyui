#!/bin/bash
# build.sh — compile the preprocess CLI + libpreprocess.so (OpenCV + ONNX Runtime).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORT_DIR="${ORT_DIR:-/data/venv/onnxruntime-linux-x64-gpu-1.26.0}"
OPENCV_CFLAGS="$(pkg-config --cflags opencv4 2>/dev/null || echo -I/usr/include/opencv4)"
OPENCV_LIBS="$(pkg-config --libs opencv4 2>/dev/null || echo -lopencv_core -lopencv_imgproc -lopencv_imgcodecs)"

CXXFLAGS=(-O2 -std=c++17 -fPIC)
INCS=(-I"$ORT_DIR/include" -I"$DIR" $OPENCV_CFLAGS)
LIBS=(-L"$ORT_DIR/lib" -Wl,-rpath,"$ORT_DIR/lib" -lonnxruntime $OPENCV_LIBS)

echo ">>> compiling preprocesscli"
g++ "${CXXFLAGS[@]}" "$DIR/preprocesscli.cpp" "$DIR/preprocess.cpp" "${INCS[@]}" "${LIBS[@]}" -o "$DIR/preprocesscli"

if [ -f "$DIR/preprocess_capi.cpp" ]; then
    echo ">>> compiling libpreprocess.so"
    g++ "${CXXFLAGS[@]}" -shared "$DIR/preprocess_capi.cpp" "$DIR/preprocess.cpp" "${INCS[@]}" "${LIBS[@]}" -o "$DIR/libpreprocess.so"
    echo "OK: $DIR/preprocesscli  $DIR/libpreprocess.so"
else
    echo "OK: $DIR/preprocesscli"
fi
