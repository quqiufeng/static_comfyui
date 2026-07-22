#!/bin/bash
# deploy.sh — ComfyCLI 纯二进制部署打包脚本
#
# 把编译好的二进制 + 所有依赖 .so 打包，远程无需 Python/pip/venv。
# 用法:
#   ./deploy.sh                         # 打包部署包到 dist/
#   ./deploy.sh --scp user@host         # 打包 + SCP 到远程
#   GLIBC_TARGET=2.35 ./deploy.sh --scp user@host  # 指定目标 GLIBC 版本
#   WITH_CUDA=1 ./deploy.sh --scp user@host         # 同时打包 CUDA Runtime
#
# 前提：先用 build.sh 编译好 comfycli-bin + comfycli-bin.so + libsdcpp_adapter.so
#
# 设计原则：
#   - 不编译，只打包部署（编译用 build.sh）
#   - GLIBC 兼容：从 /opt/deb/<version>/ 取目标版本 GLIBC + libstdc++
#     打包到 dist/lib/，远程通过 run.sh 设 LD_LIBRARY_PATH 优先加载
#   - 可选打包 CUDA Runtime（libcudart/cublas/cublasLt）到 dist/lib/
#   - run.sh 自动设置 LD_LIBRARY_PATH，远程一条命令启动
#
# 准备 deb（一次性的）：
#   ssh root@remote_host "ldd --version | head -1"  # 看远程 GLIBC 版本
#   mkdir -p /opt/deb/2.35 && cd /opt/deb/2.35
#   wget http://archive.ubuntu.com/ubuntu/pool/main/g/glibc/libc6_2.35-0ubuntu3_amd64.deb
#   wget http://archive.ubuntu.com/ubuntu/pool/main/g/glibc/libc6-dev_2.35-0ubuntu3_amd64.deb
#   wget http://archive.ubuntu.com/ubuntu/pool/main/g/gcc-11/libstdc++-11-dev_11.4.0-1ubuntu1~22.04_amd64.deb
#   for d in *.deb; do dpkg-deb -x "$d" . ; done

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="$PROJECT_DIR/dist"

SD_BACKEND_DL="${SD_BACKEND_DL:-1}"
SD_BUILD_DIR="${SD_BUILD_DIR:-/opt/sd/build-dl}"
WITH_CUDA_BACKEND="${WITH_CUDA_BACKEND:-1}"
WITH_CUDA="${WITH_CUDA:-0}"
WITH_ONNX_CUDA="${WITH_ONNX_CUDA:-0}"
CUDA_DIR="${CUDA_DIR:-/data/cuda/targets/x86_64-linux/lib}"
ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR:-/data/venv/onnxruntime-linux-x64-gpu-1.20.1/lib}"

GLIBC_TARGET="${GLIBC_TARGET:-}"
if [ -n "$GLIBC_TARGET" ]; then
  GLIBC_SYSROOT="/opt/deb/$GLIBC_TARGET"
  TARBALL_SUFFIX="_glibc${GLIBC_TARGET}"
else
  GLIBC_SYSROOT=""
  TARBALL_SUFFIX=""
fi

if [ "$WITH_CUDA" = "1" ]; then
  TARBALL_SUFFIX="${TARBALL_SUFFIX}_cuda"
  WITH_CUDA_BACKEND=1
fi

if [ "$WITH_ONNX_CUDA" = "1" ]; then
  TARBALL_SUFFIX="${TARBALL_SUFFIX}_onnx_cuda"
fi

if [ "$WITH_CUDA_BACKEND" = "1" ] && [ ! -f "$SD_BUILD_DIR/bin/libggml-cuda.so" ]; then
  echo "警告: WITH_CUDA_BACKEND=1 但找不到 $SD_BUILD_DIR/bin/libggml-cuda.so，已回退为 CPU-only"
  WITH_CUDA_BACKEND=0
fi

echo "============================================"
echo " ComfyCLI 纯二进制部署包"
if [ "$SD_BACKEND_DL" = "1" ]; then
  echo " 后端模式: 动态加载 (libggml-cuda.so 可独立分发)"
else
  echo " 后端模式: 静态链接"
fi
if [ "$WITH_CUDA_BACKEND" = "1" ]; then
  echo " CUDA 后端插件: 包含"
else
  echo " CUDA 后端插件: 不包含（CPU-only）"
fi
if [ "$WITH_CUDA" = "1" ]; then
  echo " CUDA Runtime: 打包"
else
  echo " CUDA Runtime: 不打包（远程需自带 CUDA Runtime）"
fi
if [ "$WITH_ONNX_CUDA" = "1" ]; then
  echo " ONNX Runtime CUDA provider: 打包（IPAdapter 会占用 ONNX Runtime CUDA，体积 +663MB）"
else
  echo " ONNX Runtime CPU-only: 打包（IPAdapter 默认 CPU 推理）"
fi
echo "============================================"

# ── 检查二进制是否存在 ──
if [ ! -f "$PROJECT_DIR/comfycli-bin" ] || [ ! -f "$PROJECT_DIR/comfycli-bin.so" ]; then
  echo "错误: 找不到 comfycli-bin 或 comfycli-bin.so"
  echo "请先运行 build.sh 编译"
  exit 1
fi

# ── 清空部署目录 ──
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR/lib"

# ── 收集依赖 .so 到 dist/lib/ ──
echo ""
echo ">>> 收集依赖 .so 到 $DIST_DIR/lib"

GLIBC_SRC="/lib/x86_64-linux-gnu"
if [ -n "$GLIBC_TARGET" ]; then
  GLIBC_SRC2="$GLIBC_SYSROOT/lib/x86_64-linux-gnu"
  [ -d "$GLIBC_SRC2" ] && GLIBC_SRC="$GLIBC_SRC2"
fi
echo "  GLIBC 源: $GLIBC_SRC"

LIBMISSING=0
for lib in libm.so.6 libc.so.6 libpthread.so.0 librt.so.1 libdl.so.2 ld-linux-x86-64.so.2 libstdc++.so.6 libgomp.so.1; do
  f=$(find "$GLIBC_SRC" -name "$lib" 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    cp -L "$f" "$DIST_DIR/lib/"
    echo "    ✓ $lib"
  else
    echo "    ✗ $lib  未找到"
    LIBMISSING=$((LIBMISSING + 1))
  fi
done

echo ""
echo "  共 $(ls "$DIST_DIR/lib" | wc -l) 个 .so 文件"
if [ "$LIBMISSING" -gt 0 ]; then
  echo "  警告: $LIBMISSING 个 GLIBC 兼容库未找到"
fi

# ── 打包 ONNX Runtime（IPAdapter 依赖）──
# IPAdapter 的 CLIP Vision 默认 CPU 推理，只需主库即可；
# 如需 ONNX Runtime 内部跑在 CUDA 上，设置 WITH_ONNX_CUDA=1。
echo ""
echo ">>> 收集 ONNX Runtime .so 到 $DIST_DIR/lib"
if [ ! -d "$ONNXRUNTIME_DIR" ]; then
  echo "  错误: ONNX Runtime 目录不存在: $ONNXRUNTIME_DIR"
  exit 1
fi
ONNX_MISSING=0
ONNX_LIBS=(libonnxruntime.so.1 libonnxruntime.so.1.20.1 libonnxruntime.so)
if [ "$WITH_ONNX_CUDA" = "1" ]; then
  ONNX_LIBS+=(libonnxruntime_providers_shared.so libonnxruntime_providers_cuda.so)
fi
for lib in "${ONNX_LIBS[@]}"; do
  f=$(find "$ONNXRUNTIME_DIR" -maxdepth 1 -name "$lib" 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    cp -L "$f" "$DIST_DIR/lib/"
    echo "    ✓ $lib"
  else
    echo "    ✗ $lib 未找到"
    ONNX_MISSING=$((ONNX_MISSING + 1))
  fi
done
if [ "$ONNX_MISSING" -gt 0 ]; then
  echo "  警告: $ONNX_MISSING 个 ONNX Runtime 库未找到"
fi

# ── 可选：打包 CUDA Runtime ──
if [ "$WITH_CUDA" = "1" ]; then
  echo ""
  echo ">>> 收集 CUDA Runtime .so 到 $DIST_DIR/lib"
  if [ ! -d "$CUDA_DIR" ]; then
    echo "  错误: CUDA 目录不存在: $CUDA_DIR"
    exit 1
  fi
  CUDA_MISSING=0
  for lib in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do
    f=$(find "$CUDA_DIR" -name "$lib" 2>/dev/null | head -1)
    if [ -n "$f" ]; then
      cp -L "$f" "$DIST_DIR/lib/"
      echo "    ✓ $lib"
    else
      echo "    ✗ $lib 未找到"
      CUDA_MISSING=$((CUDA_MISSING + 1))
    fi
  done
  if [ "$CUDA_MISSING" -gt 0 ]; then
    echo "  警告: $CUDA_MISSING 个 CUDA Runtime 库未找到"
    exit 1
  fi
fi

# ── 复制二进制到 dist/ ──
echo ""
echo ">>> 复制二进制"
cp "$PROJECT_DIR/comfycli-bin" "$DIST_DIR/"
cp "$PROJECT_DIR/comfycli-bin.so" "$DIST_DIR/"

# 复制 C++ 推理后端
if [ -f "$PROJECT_DIR/cpp/sd/build/libsdcpp_adapter.so" ]; then
  cp "$PROJECT_DIR/cpp/sd/build/libsdcpp_adapter.so" "$DIST_DIR/"
elif [ -f "$PROJECT_DIR/libsdcpp_adapter.so" ]; then
  cp "$PROJECT_DIR/libsdcpp_adapter.so" "$DIST_DIR/"
else
  echo "警告: libsdcpp_adapter.so 未找到"
fi

# ── 动态后端模式：复制 sd.cpp / ggml 共享库与后端插件 ──
if [ "$SD_BACKEND_DL" = "1" ]; then
  echo ""
  echo ">>> 复制 sd.cpp / ggml 共享库到 $DIST_DIR/lib"
  SD_LIBS=(
    "$SD_BUILD_DIR/bin/libstable-diffusion.so"
    "$SD_BUILD_DIR/bin/libggml.so"
    "$SD_BUILD_DIR/bin/libggml.so.0"
    "$SD_BUILD_DIR/bin/libggml.so.0.15.3"
    "$SD_BUILD_DIR/bin/libggml-base.so"
    "$SD_BUILD_DIR/bin/libggml-base.so.0"
    "$SD_BUILD_DIR/bin/libggml-base.so.0.15.3"
  )
  for f in "${SD_LIBS[@]}"; do
    if [ -f "$f" ]; then
      cp -L "$f" "$DIST_DIR/lib/"
      echo "    ✓ $(basename "$f")"
    else
      echo "    ✗ $(basename "$f") 未找到"
    fi
  done

  echo ""
  echo ">>> 复制 CPU 后端插件到 $DIST_DIR"
  for f in "$SD_BUILD_DIR/bin"/libggml-cpu-*.so; do
    if [ -f "$f" ]; then
      cp -L "$f" "$DIST_DIR/"
      echo "    ✓ $(basename "$f")"
    fi
  done

  if [ "$WITH_CUDA_BACKEND" = "1" ]; then
    echo ""
    echo ">>> 复制 CUDA 后端插件到 $DIST_DIR"
    if [ -f "$SD_BUILD_DIR/bin/libggml-cuda.so" ]; then
      cp -L "$SD_BUILD_DIR/bin/libggml-cuda.so" "$DIST_DIR/"
      echo "    ✓ libggml-cuda.so"
    else
      echo "    ✗ libggml-cuda.so 未找到"
    fi
  fi
fi

# 复制 workflow 示例（如果有）
for f in "$PROJECT_DIR"/workflow*.json; do
  [ -f "$f" ] && cp "$f" "$DIST_DIR/"
done

# ── 创建远程启动脚本 ──
cat > "$DIST_DIR/run.sh" << 'RUNEOF'
#!/bin/bash
# run.sh — ComfyCLI 远程启动脚本
# 远程只需要 NVIDIA 驱动，无需 pip install torch
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:$SCRIPT_DIR:$LD_LIBRARY_PATH"
export GGML_BACKEND_PATH="$SCRIPT_DIR/libggml-cuda.so"
cd "$SCRIPT_DIR"
exec ./comfycli-bin "$@"
RUNEOF
chmod +x "$DIST_DIR/run.sh"

# ── 启动前检查脚本 ──
cat > "$DIST_DIR/check_env.sh" << 'CHECKEOF'
#!/bin/bash
# check_env.sh — 远程环境检查
echo "=== 环境检查 ==="
echo "GLIBC: $(ldd --version 2>&1 | head -1)"
echo "NVIDIA 驱动: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo '未检测到')"
echo "CUDA 可用: $(nvidia-smi 2>/dev/null && echo '是' || echo '否')"
echo ""
echo "=== 依赖 .so 检查 ==="
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
for lib in libsdcpp_adapter.so libstable-diffusion.so libggml.so.0 libggml-base.so.0; do
  f=$(find "$SCRIPT_DIR" -maxdepth 2 -name "$lib*" 2>/dev/null | head -1)
  if [ -n "$f" ]; then echo "  ✓ $lib"; else echo "  ✗ $lib (未找到)"; fi
done
echo "  ONNX Runtime: $(find "$SCRIPT_DIR/lib" -maxdepth 1 -name 'libonnxruntime.so*' 2>/dev/null | head -1 | xargs -r basename 2>/dev/null || echo '未找到')"
echo "  CUDA 后端: $(find "$SCRIPT_DIR" -maxdepth 1 -name 'libggml-cuda.so*' 2>/dev/null | head -1 | xargs -r basename 2>/dev/null || echo '未找到')"
echo "  LD_LIBRARY_PATH=$SCRIPT_DIR/lib:$SCRIPT_DIR"
CHECKEOF
chmod +x "$DIST_DIR/check_env.sh"

# ── 打包 tarball ──
echo ""
echo ">>> 打包部署包"
TARBALL="$DIST_DIR/comfycli_deploy${TARBALL_SUFFIX}.tar.gz"
TMP_TARBALL=$(mktemp -t comfycli_deploy.XXXXXX.tar.gz)
trap 'rm -f "$TMP_TARBALL"' EXIT
cd "$DIST_DIR"
tar czf "$TMP_TARBALL" --exclude="*.tar.gz" .
mv "$TMP_TARBALL" "$TARBALL"
trap - EXIT
echo "  -> $TARBALL ($(du -h "$TARBALL" | cut -f1))"

echo ""
echo "  部署包目录: $DIST_DIR/"
echo "  文件:"
du -sh "$DIST_DIR"/* 2>/dev/null | grep -v '.tar.gz' | sed 's/^/    /'

# ── 可选: SCP 到远程 ──
if [ $# -ge 2 ] && [ "$1" = "--scp" ]; then
  REMOTE="$2"
  echo ""
  echo ">>> SCP 到 $REMOTE:/opt/comfycli/"
  ssh "$REMOTE" "mkdir -p /opt/comfycli" 2>/dev/null || true
  scp "$TARBALL" "$REMOTE:/opt/comfycli/" 2>&1
  ssh "$REMOTE" "cd /opt/comfycli && tar xzf $(basename "$TARBALL") && rm -f $(basename "$TARBALL")" 2>&1
  echo "  OK"
  echo ""
  echo "=== 远程命令 ==="
  echo "  环境检查: ssh $REMOTE \"bash /opt/comfycli/check_env.sh\""
  echo "  运行:     ssh $REMOTE \"bash /opt/comfycli/run.sh workflow.json --output-dir ./output\""
fi

echo ""
echo "============================================"
echo " 完成"
echo "============================================"
