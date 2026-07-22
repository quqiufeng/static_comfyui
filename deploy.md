# ComfyCLI 纯二进制部署

## 原理

把编译好的二进制 + 所有依赖 `.so` 打包到一个目录，远程无需 pip install torch、无需 Python 解释器、无需 venv。

```
本地 (Ubuntu 24.04, GLIBC 2.39)          远程服务器 (Ubuntu 22.04, GLIBC 2.35)
                                           NVIDIA 驱动 ✅
┌────────────────────┐                     Python ❌
│  comfycli-bin (ELF)│ ──scp──→           pip ❌
│  comfycli-bin.so   │                     venv ❌
│  libsdcpp_adapter. │                     CUDA toolkit ❌
│    so              │                     ┌──────────────────────┐
│  lib/              │                     │ comfycli-bin       │
│    libc.so.6       │                     │ comfycli-bin.so    │
│    libstdc++.so.6  │                     │ libsdcpp_adapter.so│
│    libtorch.so     │                     │ lib/               │
│    ...             │                     │ run.sh             │
└────────────────────┘                     └──────────────────────┘
```

**远程只要求**：
- NVIDIA 驱动（兼容本地 CUDA 版本）
- GLIBC ≥ 目标版本（通过 `lib/` 兼容层解决）

---

## 编译

```bash
# 本地编译
./build.sh

# 产物
#   comfycli-bin                   — 独立 ELF 二进制 (~4MB)
#   comfycli-bin.so                — Chez AOT 编译产物 (~1.6MB)
#   cpp/sd/build/libsdcpp_adapter.so — stable-diffusion.cpp 推理后端 (~185MB)
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
├── libsdcpp_adapter.so          # stable-diffusion.cpp 推理后端
├── lib/                          # GLIBC 兼容层 + 基础运行时
│   ├── ld-linux-x86-64.so.2     # 动态链接器 (GLIBC 兼容)
│   ├── libc.so.6                 # GLIBC 兼容
│   ├── libm.so.6                 # GLIBC 兼容
│   ├── libpthread.so.0           # GLIBC 兼容
│   ├── librt.so.1                # GLIBC 兼容
│   ├── libdl.so.2                # GLIBC 兼容
│   ├── libstdc++.so.6            # libstdc++ 兼容
│   └── libgomp.so.1              # OpenMP 运行时
├── run.sh                        # 启动脚本 (自动设 LD_LIBRARY_PATH)
├── check_env.sh                  # 环境检查脚本
└── comfycli_deploy[_glibc2.35].tar.gz  # 部署 tarball
```

### 打包内容说明

| 内容 | 来源 | 远程作用 |
|------|------|---------|
| `comfycli-bin` | 本地编译 | 主程序（编排逻辑） |
| `comfycli-bin.so` | 本地编译 | Chez AOT Scheme 机器码 |
| `libsdcpp_adapter.so` | 本地编译 | stable-diffusion.cpp 推理 API |
| `libgomp.so.1` | `/lib/x86_64-linux-gnu/` | OpenMP 并行 |
| `libc.so.6` / `libm.so.6` / `libpthread.so.0` / `librt.so.1` / `libdl.so.2` / `ld-linux-x86-64.so.2` | `/lib/x86_64-linux-gnu/` (或 `/opt/deb/<version>/`) | **GLIBC 兼容层**，远程系统 GLIBC 过旧时用这里打包的版本 |
| `libstdc++.so.6` | `/lib/x86_64-linux-gnu/` (或 `/opt/deb/<version>/`) | C++ ABI 兼容 |
| `run.sh` | deploy.sh 生成 | 自动设 `LD_LIBRARY_PATH=lib/` 后启动 |
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
  ✓ libtorch.so
  ✓ libc10.so
  ✗ libcudart.so (未找到)
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
- NVIDIA 驱动（兼容本地 CUDA 版本）
- CUDA Runtime（`libcudart.so.12`）和 cuBLAS（`libcublas.so.12` / `libcublasLt.so.12`），通常安装 CUDA toolkit 后自带
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

## 体积优化

本次已移除 `libtorch.so` 等 PyTorch 运行时依赖，部署包从 ~700MB 降到约 ~200MB。

当前体积主要来源：

- `libsdcpp_adapter.so` 静态嵌入了 sd.cpp + ggml + CUDA 后端 (~185MB)
- GLIBC / libstdc++ 兼容层 (~20MB)
- `comfycli-bin` / `comfycli-bin.so` (~6MB)

后续进一步瘦身方向：
1. 对 `libsdcpp_adapter.so` 做动态依赖拆分（CUDA 后端单独 .so）
2. 只打包目标架构需要的 CUDA 计算能力
3. 可选：将 CUDA Runtime / cuBLAS 也打包到 `lib/`，实现真正的零外部依赖（但会显著增大包体积）
