# ComfyCLI 编译流水线

## 总览

```
main.static.py  ──→  staticpy/static_translate.py  ──→  .ss (Scheme)
                                                              │
cpp/libtorch_std_helper.so  (C++ 推理后端, 提前编译)            │
                                                              ▼
                                                    staticpy/static_build.sh
                                                              │
                                                    Chez AOT compile-file
                                                              │
                                                    .so (机器码)
                                                              │
                                                    C launcher + objcopy + gcc
                                                              │
                              ┌─────────────────────────────────┘
                              ▼
                    comfycli  (独立 ELF 二进制)
                    comfycli.so  (Chez AOT 产物, 运行时加载)
```

**产物不依赖 Python 解释器、pip 或 venv**，只需系统 libc/libm + libtorch + cuDNN。

---

## 项目结构

```
comfycli/
├── main.static.py          # 主入口 (StaticPy 源码)
├── execution.static.py     # 节点 DAG 调度
├── nodes.static.py         # 200+ 节点定义
├── model_config.static.py  # 模型架构检测
├── model_management.static.py  # GPU 显存管理
├── validate.static.py      # prompt 校验
├── cache.static.py         # 输出缓存
├── sampler.static.py       # 采样配置
├── folder_paths.static.py  # 路径管理
├── latent_formats.static.py # 潜空间格式
│
├── staticpy/               # StaticPy 编译器 + 运行时 (项目本地拷贝)
│   ├── static_translate.py    # Python → Scheme 翻译器
│   ├── static_prelude.scm     # 值类型运行时 (fixnum/flonum/数组/字典/GC)
│   ├── static_stdlib.scm      # 公共标准库 (含 libtorch FFI)
│   └── static_build.sh        # Scheme → .so → ELF 构建脚本
│
├── cpp/                    # C++ 推理后端
│   ├── libtorch_std_helper.h    # C API 头文件
│   ├── libtorch_std_helper.cpp  # libtorch 桥接实现
│   └── build_torch_std_helper.sh  # 编译脚本
│
├── build.sh                # 一键编译入口
├── BUILD.md                # 本文件
└── README.md               # 项目远景说明
```

---

## 两阶段编译

### Phase 1: C++ 推理后端 (只编译一次)

```bash
bash cpp/build_torch_std_helper.sh
```

编译 `cpp/libtorch_std_helper.cpp` → `libtorch_std_helper.so`。

这个 `.so` 封装了所有 libtorch 推理操作：UNet forward、VAE encode/decode、CLIP encode、sampler 步进、ControlNet apply、LoRA merge 等。

除非修改了 C++ 代码或升级 libtorch 版本，否则不需要重新编译。

### Phase 2: StaticPy 编排层 (每次修改源码后)

```bash
bash staticpy/static_build.sh main.static.py comfycli
```

内部三步：

```
 1. static_translate.py     main.static.py → comfycli_code.ss
 2. static_build.sh 合并 prelude + stdlib + user code
                     Chez compile-file AOT → .so
 3. objcopy 嵌入 Chez boot 文件 → .o
    C launcher + gcc 链接 → 独立 ELF 二进制
```

---

## 一键构建

```bash
./build.sh
```

等价于按顺序执行 Phase 1 + Phase 2。

---

## 运行

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

---

## 增量编译

`static_build.sh` 用 md5 对 `static_prelude.scm` + `static_stdlib.scm` + `static_translate.py` + 用户代码计算缓存 key。若输入未变则跳过 AOT 编译和 ELF 链接，直接复用 `/tmp/staticpy-cache/` 中的缓存产物。

---

## 与官方 StaticPy 的关系

本项目从 [/opt/ReScheme](file:///opt/ReScheme) (StaticPy 官方仓库) 拷贝了四份核心文件到 `staticpy/`：

| 文件 | 来源 | 说明 |
|------|------|------|
| `static_translate.py` | StaticPy 编译器 | Python→Scheme 翻译器 |
| `static_prelude.scm` | StaticPy 运行时 | 值类型、数组、字典、GC |
| `static_stdlib.scm` | StaticPy 运行时 | 公共标准库 (libtorch FFI 等) |
| `static_build.sh` | StaticPy 构建脚本 | 适配为本项目本地路径 |

以及 `cpp/` 中的 C++ 推理后端：

| 文件 | 来源 | 说明 |
|------|------|------|
| `libtorch_std_helper.h` | StaticPy torch 桥接 | C API 头文件 |
| `libtorch_std_helper.cpp` | StaticPy torch 桥接 | libtorch 调用实现 |
| `build_torch_std_helper.sh` | StaticPy 构建脚本 | 编译 `.so` |

**为什么要拷贝而不是引用？**

- StaticPy 核心很小 (< 6000 行)，每个项目应当自包含
- 项目间的 prelude/stdlib 定制需求不同 (例如本项目的 `static_stdlib.scm` 专注于 torch，不需要 OpenBLAS/XGBoost 等)
- 避免对 `/opt/ReScheme` 的运行时路径依赖
- 未来可按需裁剪或扩展 `static_prelude.scm` / `static_stdlib.scm`

---

## 编译产物

| 文件 | 说明 |
|------|------|
| `comfycli` | 独立 ELF 二进制 (主程序) |
| `comfycli.so` | Chez AOT 编译的 Scheme 机器码 (运行时加载) |
| `cpp/libtorch_std_helper.so` | C++ 推理后端 (运行时加载) |

---

## 调试

| 问题 | 方法 |
|------|------|
| 类型错误 | `python3 staticpy/static_translate.py main.static.py` 看 stderr |
| Scheme 编译错误 | 查看 `/tmp/staticpy-cache/${STEM}_${HASH}.ss` |
| 运行时找不到 .so | `LD_LIBRARY_PATH=cpp/:/data/venv/lib/python3.12/site-packages/torch/lib ./comfycli` |
| FFI 符号未找到 | `nm -D cpp/libtorch_std_helper.so \| grep symbol` |
