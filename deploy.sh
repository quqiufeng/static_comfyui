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
CUDA_DIR="${CUDA_DIR:-/data/cuda/targets/x86_64-linux/lib}"
# Florence-2 图像→提示词节点（libflorence2.so + ONNX Runtime + OpenCV 依赖闭包）。
# 默认打包（远程用 CPU 跑 Florence，避免 344MB 的 CUDA provider）；
# WITH_FLORENCE2_MODELS=1 时附带 ~1.1GB 的 ONNX 权重目录。
WITH_FLORENCE2="${WITH_FLORENCE2:-1}"
WITH_FLORENCE2_MODELS="${WITH_FLORENCE2_MODELS:-0}"
# WITH_FLORENCE2_CUDA：打包 ONNX Runtime CUDA provider（Florence 走 GPU，+329MB）。
#   默认 1 —— 常规远程环境都自带 CUDA runtime 与 cuDNN 9，直接用 GPU；
#   完全全新的环境用 WITH_FLORENCE2_CUDA=0 退回 CPU（或 =1 时缺依赖会自动回落 CPU）。
#   WITH_FLORENCE2_CUDA_FULL=1 连 CUDA runtime（cudart/cublas/cublasLt/curand/cufft）
#   一起打包（+~0.9GB；cuDNN 仍不打包，太大）。
WITH_FLORENCE2_CUDA="${WITH_FLORENCE2_CUDA:-1}"
WITH_FLORENCE2_CUDA_FULL="${WITH_FLORENCE2_CUDA_FULL:-0}"
FLORENCE2_MODEL_DIR="${FLORENCE2_MODEL_DIR:-/data/models/florence2}"
# BiRefNet 抠图节点（libbirefnet.so，复用同一份 ONNX Runtime）。
WITH_BIREFNET="${WITH_BIREFNET:-1}"
WITH_BIREFNET_MODELS="${WITH_BIREFNET_MODELS:-0}"
BIREFNET_MODEL_DIR="${BIREFNET_MODEL_DIR:-/data/models/birefnet}"
# OCR 节点（libocr.so，PP-OCRv4；模型仅 ~15MB）。
WITH_OCR="${WITH_OCR:-1}"
WITH_OCR_MODELS="${WITH_OCR_MODELS:-1}"
OCR_MODEL_DIR="${OCR_MODEL_DIR:-/data/models/ocr}"

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
if [ "$WITH_FLORENCE2" = "1" ] && [ -f "$PROJECT_DIR/cpp/florence2/libflorence2.so" ]; then
  if [ "$WITH_FLORENCE2_CUDA" = "1" ]; then
    echo " Florence-2 节点: 打包 (GPU，含 CUDA provider)"
  else
    echo " Florence-2 节点: 打包 (CPU 推理)"
  fi
  [ "$WITH_FLORENCE2_MODELS" = "1" ] && echo " Florence-2 权重: 打包 (~1.1GB)"
else
  echo " Florence-2 节点: 不打包"
fi
if [ "$WITH_BIREFNET" = "1" ] && [ -f "$PROJECT_DIR/cpp/birefnet/libbirefnet.so" ]; then
  echo " BiRefNet 抠图节点: 打包"
  [ "$WITH_BIREFNET_MODELS" = "1" ] && echo " BiRefNet 权重: 打包 (~973MB)"
fi
if [ "$WITH_OCR" = "1" ] && [ -f "$PROJECT_DIR/cpp/ocr/libocr.so" ]; then
  echo " OCR 节点: 打包 (PP-OCRv4)"
  [ "$WITH_OCR_MODELS" = "1" ] && echo " OCR 模型: 打包 (~15MB)"
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

# ── 可选：Florence-2 图像→提示词（libflorence2.so + ONNX Runtime + OpenCV）──
if [ "$WITH_FLORENCE2" = "1" ] && [ -f "$PROJECT_DIR/cpp/florence2/libflorence2.so" ]; then
  echo ""
  echo ">>> 打包 Florence-2 节点"
  cp "$PROJECT_DIR/cpp/florence2/libflorence2.so" "$DIST_DIR/"
  echo "    ✓ libflorence2.so"

  # 依赖闭包：libflorence2.so + 其 RUNPATH 解析出的 libonnxruntime/opencv + 依赖。
  # 排除 GLIBC 核心（lib/ 兼容层已提供）、libcuda（远程驱动提供）、libstdc++/gomp（已复制）。
  FLOR2="$PROJECT_DIR/cpp/florence2/libflorence2.so"
  LDD_UNION="$(mktemp)"
  ldd "$FLOR2" 2>/dev/null | grep -oP '=> \K[^ ]+' | grep '^/' | sort -u >> "$LDD_UNION" || true
  # ONNX Runtime 的 provider 共享库由 libonnxruntime dlopen，不在 ldd 里，显式补上
  ORT_REAL=$(ldd "$FLOR2" 2>/dev/null | awk '/libonnxruntime\.so\.1 /{print $3}')
  if [ -n "$ORT_REAL" ]; then
    ORT_DIR="$(dirname "$ORT_REAL")"
    for extra in "$ORT_DIR"/libonnxruntime_providers_shared.so; do
      [ -f "$extra" ] && echo "$extra" >> "$LDD_UNION"
    done
  fi
  F2_COPIED=0
  while read -r f; do
    case "$f" in
      */libc.so.6|*/libm.so.6|*/libpthread.so.0|*/libdl.so.2|*/librt.so.1|*/ld-linux-x86-64.so.2|*/libresolv.so.2|*/libgcc_s.so.1|*/libcuda.so.1|*/libstdc++.so.6|*/libgomp.so.1) continue ;;
    esac
    cp -L "$f" "$DIST_DIR/lib/" && F2_COPIED=$((F2_COPIED + 1))
  done < "$LDD_UNION"
  rm -f "$LDD_UNION"
  echo "    ✓ 依赖闭包 $F2_COPIED 个 .so -> lib/"

  if [ "$WITH_FLORENCE2_CUDA" = "1" ] && [ -n "$ORT_REAL" ]; then
    ORT_DIR="$(dirname "$ORT_REAL")"
    if [ -f "$ORT_DIR/libonnxruntime_providers_cuda.so" ]; then
      cp -L "$ORT_DIR/libonnxruntime_providers_cuda.so" "$DIST_DIR/lib/"
      echo "    ✓ libonnxruntime_providers_cuda.so (Florence 走 GPU)"
      if [ "$WITH_FLORENCE2_CUDA_FULL" = "1" ]; then
        ldd "$ORT_DIR/libonnxruntime_providers_cuda.so" 2>/dev/null | grep -oP '=> \K[^ ]+' | grep '^/' | sort -u | while read -r f; do
          case "$f" in
            */libc.so.6|*/libm.so.6|*/libpthread.so.0|*/libdl.so.2|*/librt.so.1|*/ld-linux-x86-64.so.2|*/libresolv.so.2|*/libgcc_s.so.1|*/libcuda.so.1|*/libstdc++.so.6|*/libgomp.so.1|*/libcudnn*) continue ;;
          esac
          cp -L "$f" "$DIST_DIR/lib/" 2>/dev/null || true
        done
        echo "    ✓ CUDA runtime 依赖已打包 (FULL)"
      else
        echo "    · CUDA runtime/cuDNN 未打包；远程需自带（缺失则自动回落 CPU）"
      fi
    else
      echo "    ⚠ 未找到 libonnxruntime_providers_cuda.so，Florence 将走 CPU"
    fi
  fi

  if [ "$WITH_FLORENCE2_MODELS" = "1" ]; then
    if [ -d "$FLORENCE2_MODEL_DIR" ]; then
      mkdir -p "$DIST_DIR/florence2"
      cp -r "$FLORENCE2_MODEL_DIR/onnx" "$DIST_DIR/florence2/" 2>/dev/null || true
      [ -f "$FLORENCE2_MODEL_DIR/vocab.bin" ] && cp "$FLORENCE2_MODEL_DIR/vocab.bin" "$DIST_DIR/florence2/"
      echo "    ✓ Florence-2 权重 -> florence2/"
    else
      echo "    ⚠ 找不到模型目录 $FLORENCE2_MODEL_DIR，跳过权重"
    fi
  else
    echo "    · 权重未打包；远程需把模型放到 $FLORENCE2_MODEL_DIR（或用 FLORENCE2_MODEL_DIR 指向）"
  fi
fi

# ── 可选：BiRefNet 抠图（libbirefnet.so，复用同一份 ONNX Runtime）──
if [ "$WITH_BIREFNET" = "1" ] && [ -f "$PROJECT_DIR/cpp/birefnet/libbirefnet.so" ]; then
  echo ""
  echo ">>> 打包 BiRefNet 抠图节点"
  cp "$PROJECT_DIR/cpp/birefnet/libbirefnet.so" "$DIST_DIR/"
  echo "    ✓ libbirefnet.so"
  if [ "$WITH_BIREFNET_MODELS" = "1" ] && [ -d "$BIREFNET_MODEL_DIR" ]; then
    mkdir -p "$DIST_DIR/birefnet"
    cp -r "$BIREFNET_MODEL_DIR/onnx" "$DIST_DIR/birefnet/" 2>/dev/null || true
    [ -f "$BIREFNET_MODEL_DIR/preprocessor_config.json" ] && cp "$BIREFNET_MODEL_DIR/preprocessor_config.json" "$DIST_DIR/birefnet/"
    echo "    ✓ BiRefNet 权重 -> birefnet/"
  else
    echo "    · 权重未打包；远程需放 $BIREFNET_MODEL_DIR（或用 BIREFNET_MODEL 指向）"
  fi
fi

# ── 可选：OCR（libocr.so，PP-OCRv4，复用同一份 ONNX Runtime）──
if [ "$WITH_OCR" = "1" ] && [ -f "$PROJECT_DIR/cpp/ocr/libocr.so" ]; then
  echo ""
  echo ">>> 打包 OCR 节点"
  cp "$PROJECT_DIR/cpp/ocr/libocr.so" "$DIST_DIR/"
  echo "    ✓ libocr.so"
  if [ "$WITH_OCR_MODELS" = "1" ] && [ -d "$OCR_MODEL_DIR" ]; then
    mkdir -p "$DIST_DIR/ocr"
    cp "$OCR_MODEL_DIR"/ch_PP-OCRv4_det_infer.onnx "$OCR_MODEL_DIR"/ch_PP-OCRv4_rec_infer.onnx \
       "$OCR_MODEL_DIR"/ppocr_keys_v1.txt "$DIST_DIR/ocr/" 2>/dev/null || true
    echo "    ✓ PP-OCRv4 模型 -> ocr/"
  fi
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
[ -d "$SCRIPT_DIR/florence2" ] && export FLORENCE2_MODEL_DIR="$SCRIPT_DIR/florence2"
[ -f "$SCRIPT_DIR/birefnet/onnx/model.onnx" ] && export BIREFNET_MODEL="$SCRIPT_DIR/birefnet/onnx/model.onnx"
[ -d "$SCRIPT_DIR/ocr" ] && export OCR_MODEL_DIR="$SCRIPT_DIR/ocr"
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
