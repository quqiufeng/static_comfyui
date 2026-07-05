#!/bin/bash
# deploy.sh — ComfyCLI 纯二进制部署打包脚本
#
# 把编译好的二进制 + 所有依赖 .so 打包，远程无需 Python/pip/venv。
# 用法:
#   ./deploy.sh                         # 打包部署包到 dist/
#   ./deploy.sh --scp user@host         # 打包 + SCP 到远程
#   GLIBC_TARGET=2.35 ./deploy.sh --scp user@host  # 指定目标 GLIBC 版本
#
# 前提：先用 build.sh 编译好 comfycli + comfycli.so + libtorch_std_helper.so
#
# 设计原则：
#   - 不编译，只打包部署（编译用 build.sh）
#   - torch 运行时 .so 打包到 dist/lib/，远程无需 pip install torch
#   - GLIBC 兼容：从 /opt/deb/<version>/ 取目标版本 GLIBC + libstdc++
#     打包到 dist/lib/，远程通过 run.sh 设 LD_LIBRARY_PATH 优先加载
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

GLIBC_TARGET="${GLIBC_TARGET:-}"
if [ -n "$GLIBC_TARGET" ]; then
  GLIBC_SYSROOT="/opt/deb/$GLIBC_TARGET"
  TARBALL_SUFFIX="_glibc${GLIBC_TARGET}"
else
  GLIBC_SYSROOT=""
  TARBALL_SUFFIX=""
fi

echo "============================================"
echo " ComfyCLI 纯二进制部署包"
echo "============================================"

# ── 检查二进制是否存在 ──
if [ ! -f "$PROJECT_DIR/comfycli" ] || [ ! -f "$PROJECT_DIR/comfycli.so" ]; then
  echo "错误: 找不到 comfycli 或 comfycli.so"
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
for lib in libm.so.6 libc.so.6 libpthread.so.0 librt.so.1 libdl.so.2 ld-linux-x86-64.so.2 libstdc++.so.6; do
  f=$(find "$GLIBC_SRC" -name "$lib" 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    cp -L "$f" "$DIST_DIR/lib/"
    echo "    ✓ $lib"
  else
    echo "    ✗ $lib  未找到"
    LIBMISSING=$((LIBMISSING + 1))
  fi
done

# torch 运行时 .so
TORCH_LIB="/data/venv/lib/python3.12/site-packages/torch/lib"
if [ ! -d "$TORCH_LIB" ]; then
  echo "错误: torch lib 目录不存在: $TORCH_LIB"
  echo "请确保 Python 虚拟环境可用或设置正确的路径"
  exit 1
fi
for lib in libtorch.so libtorch_cpu.so libtorch_cuda.so libc10.so libc10_cuda.so libgomp*.so* libshm.so; do
  f=$(find "$TORCH_LIB" -maxdepth 1 -name "$lib" 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    cp -L "$f" "$DIST_DIR/lib/"
    echo "    ✓ $lib"
  fi
done

echo ""
echo "  共 $(ls "$DIST_DIR/lib" | wc -l) 个 .so 文件"
if [ "$LIBMISSING" -gt 0 ]; then
  echo "  警告: $LIBMISSING 个 GLIBC 兼容库未找到"
fi

# ── 复制二进制到 dist/ ──
echo ""
echo ">>> 复制二进制"
cp "$PROJECT_DIR/comfycli" "$DIST_DIR/"
cp "$PROJECT_DIR/comfycli.so" "$DIST_DIR/"

# 复制 C++ 推理后端
if [ -f "$PROJECT_DIR/cpp/libtorch_std_helper.so" ]; then
  cp "$PROJECT_DIR/cpp/libtorch_std_helper.so" "$DIST_DIR/"
elif [ -f "$PROJECT_DIR/libtorch_std_helper.so" ]; then
  cp "$PROJECT_DIR/libtorch_std_helper.so" "$DIST_DIR/"
else
  echo "警告: libtorch_std_helper.so 未找到"
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
cd "$SCRIPT_DIR"
exec ./comfycli "$@"
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
for lib in libtorch.so libc10.so libcudart.so; do
  f=$(find "$SCRIPT_DIR/lib" -name "$lib*" 2>/dev/null | head -1)
  if [ -n "$f" ]; then echo "  ✓ $lib"; else echo "  ✗ $lib (未找到)"; fi
done
echo "  LD_LIBRARY_PATH=$SCRIPT_DIR/lib:$SCRIPT_DIR"
CHECKEOF
chmod +x "$DIST_DIR/check_env.sh"

# ── 打包 tarball ──
echo ""
echo ">>> 打包部署包"
TARBALL="$DIST_DIR/comfycli_deploy${TARBALL_SUFFIX}.tar.gz"
cd "$DIST_DIR"
tar czf "$TARBALL" --exclude="*.tar.gz" .
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
