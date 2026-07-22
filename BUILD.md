# ComfyCLI 编译流水线

## 总览

```
comfycli/*.static.py  ──→  concat_src.py  ──→  comfycli/_bundle.static.py
                                                │
                                                ▼
                         staticpy/static_translate.py  ──→  .ss (Scheme)
                                                │
                                                ▼
                         staticpy/static_build.sh  ──→  Chez AOT compile-file
                                                │
                                                ▼
                         C launcher + objcopy + gcc  ──→  comfycli-bin (ELF)
                                                                 │
                                                                 ▼
                                                    cpp/sd/build/libsdcpp_adapter.so
                                                    (stable-diffusion.cpp 推理后端)
```

**产物不依赖 Python 解释器、pip 或 venv**，只需系统 libc + `libsdcpp_adapter.so` + CUDA 运行时。

---

## 项目结构

```
comfycli/
├── main.static.py              # 主入口 (StaticPy 源码)
├── execution.static.py         # 节点 DAG 调度
├── nodes.static.py             # ComfyUI 节点定义（当前为最小 MVP）
├── cli_args.static.py          # 命令行参数解析
├── folder_paths.static.py      # 路径管理
├── model_base.static.py        # 模型基类（保留，但当前由 sd.cpp 内置）
├── model_management.static.py  # GPU 显存管理（保留，但当前由 sd.cpp 内置）
├── sd_backend.static.py        # stable-diffusion.cpp FFI 封装
├── sample.static.py            # 采样入口（保留，但当前由 sd.cpp 内置）
├── ...

staticpy/                       # StaticPy 编译器 + 运行时
├── static_translate.py         # Python → Scheme 翻译器
├── static_prelude.scm          # 值类型运行时
├── static_stdlib.scm           # 公共标准库 (含 FFI 封装)
└── static_build.sh             # 适配为本项目本地路径的构建脚本

cpp/sd/                         # C++ 推理后端
├── src/adapters/sdcpp_adapter.h   # C API 头文件
├── src/adapters/sdcpp_adapter.cpp # stable-diffusion.cpp 桥接实现
├── CMakeLists.txt               # 构建配置
├── build_sd.sh                  # 编译 /opt/sd 为 PIC 静态库
└── build/                       # 产物
    └── libsdcpp_adapter.so      # 推理后端共享库

build.sh                        # 编译入口
deploy.sh                       # 部署入口
README.md                       # 项目说明
BUILD.md                        # 本文件
deploy.md                       # 部署文档
```

---

## 编译

### 一键构建

```bash
./build.sh
```

等价于：编译 sd.cpp 适配器 → 合并 StaticPy 源码 → 翻译 → AOT 编译 → 链接 ELF。

### 分步编译

```bash
# Step 1: 编译 /opt/sd 为 PIC 静态库（通常只需一次）
bash cpp/sd/build_sd.sh

# Step 2: 编译 sd.cpp 适配器
mkdir -p cpp/sd/build && cd cpp/sd/build
cmake .. && make -j$(nproc)

# Step 3: StaticPy 编排层（每次改源码后）
/data/venv/bin/python3 concat_src.py
bash staticpy/static_build.sh comfycli/_bundle.static.py comfycli-bin
```

### C++ 推理后端

编译 `cpp/sd/src/adapters/sdcpp_adapter.cpp` → `cpp/sd/build/libsdcpp_adapter.so`，
封装 stable-diffusion.cpp 的 C API（`sd_pipeline_create/load/generate/free` 等）。

除非修改 C++ 适配器或升级 sd.cpp 版本，否则不需要重新编译。

### StaticPy 编排层

`static_build.sh` 内部三步：

```
 1. static_translate.py     _bundle.static.py → comfycli-bin_code.ss
 2. 合并 prelude + stdlib + user code
    Chez compile-file AOT → .so
 3. objcopy 嵌入 Chez boot 文件 → .o
    C launcher + gcc 链接 → 独立 ELF 二进制
```

---

## 编译产物

| 文件 | 大小 | 说明 |
|------|------|------|
| `comfycli-bin` | ~4MB | 独立 ELF 二进制 (主程序，含编排逻辑) |
| `comfycli-bin.so` | ~1.6MB | Chez AOT 编译的 Scheme 机器码 (运行时加载) |
| `cpp/sd/build/libsdcpp_adapter.so` | ~185MB | stable-diffusion.cpp 推理后端（含 GGML/CUDA 静态链接） |

---

## 增量编译

`static_build.sh` 对 `static_prelude.scm` + `static_stdlib.scm` + `static_translate.py` + 用户代码计算 md5 缓存 key。输入不变时跳过 AOT 编译和 ELF 链接，直接复用 `/tmp/staticpy-cache/` 中的缓存产物。

---

## 运行

### 本地开发

```bash
LD_LIBRARY_PATH=cpp/sd/build:/data/venv/lib/python3.12/site-packages/torch/lib \
  ./comfycli-bin workflow.json --output-dir ./output
```

注意：`/data/venv/lib/python3.12/site-packages/torch/lib` 目前仍用于链接 `libtorch.so` 等符号（ELF 仍带 torch rpath），实际运行时推理不再经过 torch。

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CHEZ_SCHEME_DIR` | `/opt/ChezScheme/ta6le` | Chez Scheme 安装目录 |
| `CHEZ_SCHEME` | `$CHEZ_SCHEME_DIR/bin/ta6le/scheme` | Chez 可执行文件 |
| `CHEZ_BOOT_DIR` | `/opt/ChezScheme/boot/ta6le` | Chez boot 文件目录 |
| `GLIBC_TARGET` | (未设置) | 目标 GLIBC 版本，如 `2.35` |

---

## 与官方 StaticPy 的关系

本项目从 `/opt/ReScheme` (StaticPy 官方仓库) 拷贝核心文件到 `staticpy/`：

| 文件 | 来源 | 说明 |
|------|------|------|
| `staticpy/static_translate.py` | StaticPy | Python→Scheme 翻译器 |
| `staticpy/static_prelude.scm` | StaticPy | 值类型运行时 |
| `staticpy/static_stdlib.scm` | StaticPy | 公共标准库 |
| `staticpy/static_build.sh` | StaticPy | 构建脚本 (适配本地路径) |

`cpp/sd/` 是独立维护的 stable-diffusion.cpp 适配器，不来自 StaticPy。

**为什么拷贝？** StaticPy 核心小 (< 6000 行)，每个项目自包含，避免运行时路径依赖。

---

## 依赖的开源项目

| 项目 | 协议 | 说明 |
|------|------|------|
| [Chez Scheme](https://github.com/cisco/ChezScheme) | Apache 2.0 | AOT 编译后端，将 Scheme 编译为机器码 |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | MIT | C++ 推理后端，封装 UNet/VAE/CLIP/Sampler/LoRA/ControlNet |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD | 目前 ELF 仍链接 torch 运行时符号（过渡保留） |

---

## 部署

部署（打包依赖 .so、GLIBC 兼容层、SCP 到远程）请见 [deploy.md](./deploy.md)。

---

## 调试

| 问题 | 方法 |
|------|------|
| 类型错误 | `python3 staticpy/static_translate.py comfycli/_bundle.static.py` 看 stderr |
| Scheme 编译错误 | 查看 `/tmp/staticpy-cache/${STEM}_${HASH}.ss` |
| 运行时找不到 .so | `LD_LIBRARY_PATH=cpp/sd/build:/data/venv/lib/python3.12/site-packages/torch/lib ./comfycli-bin` |
| FFI 符号未找到 | `nm -D cpp/sd/build/libsdcpp_adapter.so \| grep symbol` |
| 生成阶段 OOM | sd.cpp 已对大图自动启用 VAE tiling；仍 OOM 时降分辨率或显式调小 tile |
