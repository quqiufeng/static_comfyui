# ComfyCLI 纯二进制部署

## 原理

把编译好的二进制 + 所有依赖 `.so` 打包到一个目录，远程无需 pip install torch、无需 Python 解释器、无需 venv。

```
本地 (Ubuntu 24.04, GLIBC 2.39)          远程服务器 (Ubuntu 22.04, GLIBC 2.35)
                                           NVIDIA 驱动 ✅
┌────────────────────┐                     Python ❌
│  comfycli-bin (ELF)│ ──scp──→           pip ❌
│  comfycli-bin.so   │                     venv ❌
│  libsdcpp_adapter. │                     CUDA toolkit ⚠️（可选，见下）
│    so              │                     ┌──────────────────────────┐
│  libstable-        │                     │ comfycli-bin           │
│    diffusion.so    │                     │ comfycli-bin.so        │
│  libggml.so /      │                     │ libsdcpp_adapter.so    │
│    libggml-base.so │                     │ libstable-diffusion.so │
│  libggml-cuda.so   │                     │ libggml*.so            │
│  lib/              │                     │ libggml-cuda.so        │
│    libc.so.6       │                     │ lib/                   │
│    libstdc++.so.6  │                     │ run.sh                 │
│    ...             │                     └──────────────────────────┘
└────────────────────┘
```

**远程只要求**：
- NVIDIA 驱动（兼容本地 CUDA 版本）——若用 `WITH_CUDA_BACKEND=0` 打 CPU-only 包则不需要
- GLIBC ≥ 目标版本（通过 `lib/` 兼容层解决）
- CUDA Runtime（`libcudart.so.12` / `libcublas.so.12` / `libcublasLt.so.12`），通常 CUDA toolkit 自带；
  远程没有时用 `WITH_CUDA=1 ./deploy.sh` 把它们一并打包

---

## 编译

```bash
# 本地编译
./build.sh

# 产物
#   comfycli-bin                    — 独立 ELF 二进制 (~4.2MB)
#   comfycli-bin.so                 — Chez AOT 编译产物 (~2.0MB)
#   cpp/sd/build/libsdcpp_adapter.so — 推理适配层 (~134KB，动态链接 sd.cpp)
#
# 推理后端的 .so 来自 sd.cpp 动态构建（SD_BACKEND_DL=1，默认）：
#   /opt/sd/build-dl/bin/libstable-diffusion.so  (~39MB)
#   /opt/sd/build-dl/bin/libggml.so + libggml-base.so + libggml-cpu.so  (~2MB)
#   /opt/sd/build-dl/bin/libggml-cuda.so         (~129MB，CUDA 后端插件)
```

编译详情见 [BUILD.md](./BUILD.md)。

---

## 打包

```bash
# 打包到 dist/（包含二进制 + 所有依赖 .so + run.sh）
./deploy.sh

# 按远程 GLIBC 版本打包（推荐）
GLIBC_TARGET=2.35 ./deploy.sh
```

产物 `dist/`：

```
dist/
├── comfycli-bin                 # ELF 二进制
├── comfycli-bin.so              # Chez AOT 编译产物
├── libsdcpp_adapter.so          # 推理适配层（C API 封装）
├── libstable-diffusion.so       # sd.cpp 推理实现
├── libggml.so / libggml-base.so / libggml-cpu*.so
├── libggml-cuda.so              # CUDA 后端插件（WITH_CUDA_BACKEND=0 时不打包）
├── libcudart.so.12 / libcublas*.so.12   # 仅 WITH_CUDA=1
├── lib/                          # GLIBC 兼容层 + 基础运行时
│   ├── ld-linux-x86-64.so.2     # 动态链接器 (GLIBC 兼容)
│   ├── libc.so.6                 # GLIBC 兼容
│   ├── libm.so.6                 # GLIBC 兼容
│   ├── libpthread.so.0           # GLIBC 兼容
│   ├── librt.so.1                # GLIBC 兼容
│   ├── libdl.so.2                # GLIBC 兼容
│   ├── libstdc++.so.6            # libstdc++ 兼容
│   └── libgomp.so.1              # OpenMP 运行时
├── run.sh                        # 启动脚本 (设 LD_LIBRARY_PATH + GGML_BACKEND_PATH)
├── check_env.sh                  # 环境检查脚本
└── comfycli_deploy[_glibc2.35][_cuda].tar.gz  # 部署 tarball
```

### 三种打包规格

| 命令 | 体积 | 远程要求 |
|------|------|---------|
| `./deploy.sh` | ~57MB | 已装 CUDA 12.x Runtime（含 `libggml-cuda.so`） |
| `WITH_CUDA=1 ./deploy.sh` | ~439MB | 仅需 NVIDIA 驱动（自带 cudart/cublas） |
| `WITH_CUDA_BACKEND=0 ./deploy.sh` | ~35MB | 无需 GPU（CPU-only，不带 `libggml-cuda.so`） |

详见 [remote_server.md](./remote_server.md) 的「打包体积」表。

### 打包内容说明

| 内容 | 来源 | 远程作用 |
|------|------|---------|
| `comfycli-bin` | 本地编译 | 主程序（编排逻辑） |
| `comfycli-bin.so` | 本地编译 | Chez AOT Scheme 机器码 |
| `libsdcpp_adapter.so` | 本地编译 | sd.cpp 的 C API 适配层（`extern fn` 目标） |
| `libstable-diffusion.so` / `libggml*.so` | `/opt/sd/build-dl/bin/` | 推理实现（sd.cpp + GGML） |
| `libggml-cuda.so` | `/opt/sd/build-dl/bin/` | CUDA 后端插件，由 `GGML_BACKEND_PATH` 加载 |
| `libcudart.so.12` / `libcublas*.so.12` | `$CUDA_DIR` | 仅 `WITH_CUDA=1` 时打包的 CUDA Runtime |
| `libgomp.so.1` | `/lib/x86_64-linux-gnu/` | OpenMP 并行 |
| `libc.so.6` / `libm.so.6` / `libpthread.so.0` / `librt.so.1` / `libdl.so.2` / `ld-linux-x86-64.so.2` | `/lib/x86_64-linux-gnu/` (或 `/opt/deb/<version>/`) | **GLIBC 兼容层**，远程系统 GLIBC 过旧时用这里打包的版本 |
| `libstdc++.so.6` | `/lib/x86_64-linux-gnu/` (或 `/opt/deb/<version>/`) | C++ ABI 兼容 |
| `run.sh` | deploy.sh 生成 | 自动设 `LD_LIBRARY_PATH=lib/` + `GGML_BACKEND_PATH` 后启动 |
| `check_env.sh` | deploy.sh 生成 | 远程环境诊断 |

---

## 部署

### 前提：先编译

```bash
./build.sh
```

### 一键打包 + SCP

```bash
# 打包 + 发送到远程（须先编译）
GLIBC_TARGET=2.35 ./deploy.sh --scp user@remote_host

# 等效分步操作：
./deploy.sh                                    # 打包到 dist/
scp dist/comfycli_deploy_glibc2.35.tar.gz user@remote_host:/opt/comfycli/
ssh user@remote_host "cd /opt/comfycli && tar xzf comfycli_deploy_glibc2.35.tar.gz && rm comfycli_deploy_glibc2.35.tar.gz"
```

### 远程环境检查

```bash
ssh user@remote_host "bash /opt/comfycli/check_env.sh"
```

输出示例：
```
=== 环境检查 ===
GLIBC: GNU C Library (Ubuntu GLIBC 2.35-0ubuntu3) stable release version 2.35.
NVIDIA 驱动: 550.120
CUDA 可用: 是

=== 依赖 .so 检查 ===
  ✓ libsdcpp_adapter.so
  ✓ libstable-diffusion.so
  ✓ libggml.so.0
  ✓ libggml-base.so.0
  CUDA 后端: libggml-cuda.so
  LD_LIBRARY_PATH=/opt/comfycli/lib:/opt/comfycli
```

### 远程运行

```bash
ssh user@remote_host "bash /opt/comfycli/run.sh /path/to/workflow.json --output-dir ./output"
```

也可以 SSH 登录后手动运行：

```bash
ssh user@remote_host
export LD_LIBRARY_PATH=/opt/comfycli/lib:/opt/comfycli
cd /opt/comfycli
./comfycli-bin workflow.json --output-dir ./output
```

### 更新部署

```bash
# 只重新编译
./build.sh

# 重新打包 + SCP（增量：只传变化的文件）
GLIBC_TARGET=2.35 ./deploy.sh --scp user@remote_host
```

---

## GLIBC 兼容

### 为什么需要

| 环境 | GLIBC 版本 | 说明 |
|------|-----------|------|
| 本地 (Ubuntu 24.04) | 2.39 | 编译环境 |
| 远程 (Ubuntu 22.04) | 2.35 | 常见服务器 |
| 远程 (Ubuntu 20.04) | 2.31 | 旧服务器 |
| 远程 (CentOS 7) | 2.17 | 极旧服务器 |

直接用本地 GLIBC 2.39 编译的二进制在 GLIBC 2.35 上运行会报：

```
./comfycli-bin: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found
```

### 方案：sysroot 编译 + lib/ 兼容层

分两步解决：

**1. 编译时指定 GLIBC 版本标签**

设置 `GLIBC_TARGET` 后，编译脚本会用 `/opt/deb/<version>/` 下的 GLIBC sysroot 做链接。
`-L` + `-rpath-link` 指向旧版本 GLIBC，保证生成的 ELF 不引用比目标版本新的符号。

```
# 编译时版本标签变化
GLIBC_TARGET=    → 引用 GLIBC_2.39 (只能在 2.39+ 系统运行)
GLIBC_TARGET=2.35 → 引用 GLIBC_2.35 (可在 2.35+ 系统运行)
```

**2. 运行时加载兼容层**

部署包的 `lib/` 包含目标版本的 `libc.so.6`、`libstdc++.so.6` 等。
`run.sh` 设置 `LD_LIBRARY_PATH=lib/` 优先加载这些旧版本 `.so`，覆盖系统自带。

### 配置步骤

```bash
# 1. 确定远程 GLIBC 版本
ssh root@remote_host "ldd --version | head -1"
# 输出: GNU C Library (Ubuntu GLIBC 2.35-0ubuntu3) stable release version 2.35.

# 2. 本地下载对应 deb 到 /opt/deb/<version>/（一次性的）
mkdir -p /opt/deb/2.35 && cd /opt/deb/2.35

# Ubuntu 22.04 (GLIBC 2.35)
wget http://archive.ubuntu.com/ubuntu/pool/main/g/glibc/libc6_2.35-0ubuntu3_amd64.deb
wget http://archive.ubuntu.com/ubuntu/pool/main/g/glibc/libc6-dev_2.35-0ubuntu3_amd64.deb
wget http://archive.ubuntu.com/ubuntu/pool/main/gcc-11/libstdc++-11-dev_11.4.0-1ubuntu1~22.04_amd64.deb

for d in *.deb; do dpkg-deb -x "$d" .; done

# 3. 打包部署时指定目标版本
GLIBC_TARGET=2.35 ./deploy.sh --scp root@remote_host
```

### 验证兼容性

```bash
# 检查编译后的二进制引用了哪些 GLIBC 符号
objdump -T comfycli-bin | grep -oP 'GLIBC_\S+' | sort -t. -k1,1n -k2,2n -k3,3n | uniq

# 如果最大版本 ≤ 2.35，则兼容性 OK
```

### 常见问题

| 症状 | 原因 | 解决 |
|------|------|------|
| `GLIBC_X.XX not found` | 二进制引用了比远程更新的 GLIBC 符号 | 设置 `GLIBC_TARGET` 重编 |
| `undefined symbol: _Z...` | libstdc++ ABI 不匹配 | 检查 libstdc++.so.6 版本 |
| `cannot open shared object file: No such file or directory` | 缺少依赖 .so | 检查 lib/ 目录是否完整 |
| `libcuda.so: cannot open shared object file` | 远程没有 NVIDIA 驱动 | 确认驱动已安装 |
| Segmentation fault on startup | Chez Scheme boot 文件与架构不匹配 | 确认远程为 x86_64 |

---

## 远程服务器要求

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| 架构 | x86_64 | x86_64 |
| GLIBC | 2.17 | ≥ 2.35 |
| NVIDIA 驱动 | 与本地 CUDA 版本兼容 | ≥ 550 |
| 磁盘 | 1GB | 2GB+ |
| 内存 | 4GB | 16GB |
| 显存 | 4GB | 8GB+ |

远程需要：
- NVIDIA 驱动（兼容本地 CUDA 版本）——CPU-only 包（`WITH_CUDA_BACKEND=0`）不需要
- CUDA Runtime（`libcudart.so.12`）和 cuBLAS（`libcublas.so.12` / `libcublasLt.so.12`），通常安装 CUDA toolkit 后自带；
  远程没有则用 `WITH_CUDA=1 ./deploy.sh` 打进包里
- GLIBC ≥ 目标版本（通过 `lib/` 兼容层解决）

远程不需要：
- ❌ Python 解释器
- ❌ pip
- ❌ venv / conda
- ❌ PyTorch pip 包

---

## 完整部署流程

```bash
# 1. 本地编译
cd /opt/static_comfyui
./build.sh

# 2. 查看远程 GLIBC
ssh root@remote_host "ldd --version | head -1"
# → GLIBC 2.35

# 3. 准备 /opt/deb/2.35/（如果没有的话）
# 4. 打包 + 部署
GLIBC_TARGET=2.35 ./deploy.sh --scp root@remote_host

# 5. 检查远程环境
ssh root@remote_host "bash /opt/comfycli/check_env.sh"

# 6. 运行推理
ssh root@remote_host "bash /opt/comfycli/run.sh workflow.json --output-dir ./output"

# 7. 查看输出
ssh root@remote_host "ls -la /opt/comfycli/output/"
```

---

## 与 `build.sh` 的关系

| 脚本 | 职责 | 是否编译 | 是否部署 |
|------|------|---------|---------|
| `build.sh` | 编译二进制 + C++ `.so` | ✅ | ❌ |
| `deploy.sh` | 打包二进制 + 依赖 `.so` + GLIBC 兼容层 + SCP | ❌ | ✅ |

推荐用法：`build.sh` 只用于本地开发编译，`deploy.sh` 用于正式部署。

---

## 体积优化（已完成 + 后续方向）

瘦身是分两步完成的：

1. **移除 PyTorch 运行时**（~700MB → ~200MB）：后端从 libtorch 切到 stable-diffusion.cpp，
   不再打包 `libtorch.so` / `libc10.so` 等；IPAdapter 也改用 sd.cpp 原生实现，去掉 ONNX Runtime（79MB → 57MB）。
2. **拆分 CUDA 后端为动态插件**（~200MB → 57MB）：`SD_BACKEND_DL=1` 下
   `libsdcpp_adapter.so` 只有 ~134KB，sd.cpp 拆成 `libstable-diffusion.so`（~39MB），
   CUDA 单独成 `libggml-cuda.so`（~129MB）按需分发——于是有了 57MB / 35MB 两档。

当前体积主要来源：

| 内容 | 大小 |
|------|------|
| `libggml-cuda.so`（CUDA 后端插件） | ~129MB（CPU-only 包不含） |
| `libstable-diffusion.so` | ~39MB |
| GLIBC / libstdc++ 兼容层 | ~20MB |
| `libggml*.so` + `libsdcpp_adapter.so` | ~2MB |
| `comfycli-bin` / `comfycli-bin.so` | ~6MB |

后续进一步瘦身方向：
1. 只打包目标架构需要的 CUDA 计算能力（`CMAKE_CUDA_ARCHITECTURES` 收窄）
2. 用 `strip` + `--as-needed` 精简 `libstable-diffusion.so`
3. 按需裁剪 sd.cpp 未使用的模型族代码（Wan/LTX/Z-Image 等）
