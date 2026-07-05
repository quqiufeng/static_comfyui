# ComfyCLI 编译流水线

## 总览

```
main.static.py  ──→  staticpy/static_translate.py  ──→  .ss (Scheme)
                                                              │
cpp/libtorch_std_helper.so  (C++ 推理后端, 提前编译)           │
                                                              ▼
                                                    staticpy/static_build.sh
                                                              │
                                                    Chez AOT compile-file
                                                              │
                                                    .so (机器码)
                                                              │
                                                    C launcher + objcopy + gcc
                                                              │
                              ┌────────────────────────────────┘
                              ▼
                    comfycli  (独立 ELF 二进制)
                    comfycli.so  (Chez AOT 产物, 运行时加载)
```

**产物不依赖 Python 解释器、pip 或 venv**，只需系统 libc + libtorch + CUDA 运行时。

---

## 项目结构

```
comfycli/
├── main.static.py              # 主入口 (StaticPy 源码)
├── execution.static.py         # 节点 DAG 调度
├── nodes.static.py             # 200+ 节点定义
├── model_config.static.py      # 模型架构检测
├── model_management.static.py  # GPU 显存管理
├── validate.static.py          # prompt 校验
├── cache.static.py             # 输出缓存
├── sampler.static.py           # 采样配置
├── folder_paths.static.py      # 路径管理
├── latent_formats.static.py    # 潜空间格式
│
├── staticpy/                   # StaticPy 编译器 + 运行时 (项目本地拷贝)
│   ├── static_translate.py       # Python → Scheme 翻译器
│   ├── static_prelude.scm        # 值类型运行时
│   ├── static_stdlib.scm         # 公共标准库 (含 libtorch FFI)
│   └── static_build.sh           # 适配为本项目本地路径的构建脚本
│
├── cpp/                        # C++ 推理后端
│   ├── libtorch_std_helper.h     # C API 头文件
│   ├── libtorch_std_helper.cpp   # libtorch 桥接实现
│   └── build_torch_std_helper.sh # 编译脚本
│
├── build.sh                    # 编译入口
├── deploy.sh                   # 部署入口
├── BUILD.md                    # 本文件
├── deploy.md                   # 部署文档
└── README.md                   # 项目说明
```

---

## 编译

### 一键构建

```bash
./build.sh
```

等价于：编译 C++ `.so` → 翻译 StaticPy → AOT 编译 → 链接 ELF。

### 分步编译

```bash
# Step 1: C++ 推理后端 (只编译一次)
bash cpp/build_torch_std_helper.sh

# Step 2: StaticPy 编排层 (每次改源码后)
bash staticpy/static_build.sh main.static.py comfycli
```

### C++ 推理后端

编译 `libtorch_std_helper.cpp` → `libtorch_std_helper.so`，封装 UNet forward、VAE、CLIP、sampler、ControlNet、LoRA 等所有推理操作。

除非修改了 C++ 代码或升级 libtorch 版本，否则不需要重新编译。

### StaticPy 编排层

`static_build.sh` 内部三步：

```
 1. static_translate.py     main.static.py → comfycli_code.ss
 2. 合并 prelude + stdlib + user code
    Chez compile-file AOT → .so
 3. objcopy 嵌入 Chez boot 文件 → .o
    C launcher + gcc 链接 → 独立 ELF 二进制
```

---

## 编译产物

| 文件 | 大小 | 说明 |
|------|------|------|
| `comfycli` | ~12MB | 独立 ELF 二进制 (主程序) |
| `comfycli.so` | ~6MB | Chez AOT 编译的 Scheme 机器码 (运行时加载) |
| `cpp/libtorch_std_helper.so` | ~1MB | C++ 推理后端 (运行时加载) |

---

## 增量编译

`static_build.sh` 对 `static_prelude.scm` + `static_stdlib.scm` + `static_translate.py` + 用户代码计算 md5 缓存 key。输入不变时跳过 AOT 编译和 ELF 链接，直接复用 `/tmp/staticpy-cache/` 中的缓存产物。

---

## 运行

### 本地开发

```bash
LD_LIBRARY_PATH=cpp/:/data/venv/lib/python3.12/site-packages/torch/lib \
  ./comfycli workflow.json --output-dir ./output
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CHEZ_SCHEME_DIR` | `/opt/ChezScheme/ta6le` | Chez Scheme 安装目录 |
| `CHEZ_SCHEME` | `$CHEZ_SCHEME_DIR/bin/ta6le/scheme` | Chez 可执行文件 |
| `CHEZ_BOOT_DIR` | `/opt/ChezScheme/boot/ta6le` | Chez boot 文件目录 |
| `GLIBC_TARGET` | (未设置) | 目标 GLIBC 版本，如 `2.35` |

---

## 与官方 StaticPy 的关系

本项目从 `/opt/ReScheme` (StaticPy 官方仓库) 拷贝核心文件到 `staticpy/` 和 `cpp/`：

| 文件 | 来源 | 说明 |
|------|------|------|
| `staticpy/static_translate.py` | StaticPy | Python→Scheme 翻译器 |
| `staticpy/static_prelude.scm` | StaticPy | 值类型运行时 |
| `staticpy/static_stdlib.scm` | StaticPy | 公共标准库 |
| `staticpy/static_build.sh` | StaticPy | 构建脚本 (适配本地路径) |
| `cpp/libtorch_std_helper.*` | StaticPy | C++ 推理桥接 |
| `cpp/build_torch_std_helper.sh` | StaticPy | 编译脚本 |

**为什么拷贝？** StaticPy 核心小 (< 6000 行)，每个项目自包含，避免运行时路径依赖。

---

## 依赖的开源项目

| 项目 | 协议 | 说明 |
|------|------|------|
| [Chez Scheme](https://github.com/cisco/ChezScheme) | Apache 2.0 | AOT 编译后端，将 Scheme 编译为机器码 |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD | C++ 推理运行时 (libtorch) |

---

## 部署

部署（打包依赖 .so、GLIBC 兼容层、SCP 到远程）请见 [deploy.md](./deploy.md)。

---

## 调试

| 问题 | 方法 |
|------|------|
| 类型错误 | `python3 staticpy/static_translate.py main.static.py` 看 stderr |
| Scheme 编译错误 | 查看 `/tmp/staticpy-cache/${STEM}_${HASH}.ss` |
| 运行时找不到 .so | `LD_LIBRARY_PATH=cpp/:/data/venv/lib/python3.12/site-packages/torch/lib ./comfycli` |
| FFI 符号未找到 | `nm -D cpp/libtorch_std_helper.so \| grep symbol` |
