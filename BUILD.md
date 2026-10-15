# ComfyCLI 编译流水线

## 总览

```
comfycli/*.static.py  ──→  concat_src.py  ──→  comfycli/_bundle.static.py
                                                │
                                                ▼
                    staticpy/static_translate.py  ──→  .ss (Scheme)
                                                │
                                                ▼
        staticpy/static_build_comfycli.sh  ──→  Chez AOT compile-file
        （拼接 prelude + stdlib + ffi.scm + code）      │
                                                ▼
                         C launcher + objcopy + gcc  ──→  comfycli-bin (ELF)
                                                                │
                                                                ▼
                                                   cpp/sd/build/libsdcpp_adapter.so
                                                   (stable-diffusion.cpp 推理后端)
```

**产物不依赖 Python 解释器、pip、venv，也不链接 torch**，只需系统 libc + `libsdcpp_adapter.so` + CUDA 运行时。

---

## StaticPy 工具链：上游拷贝 + 少量本地补丁 + 项目胶水

`staticpy/` 下的编译器核心从 `/opt/ReScheme` 拷贝而来，**不是完全原样**：

| 文件 | 与上游关系 | 说明 |
|------|-----------|------|
| `staticpy/static_translate.py` | **有本地补丁**（见下表） | Python → Scheme 翻译器 |
| `staticpy/static_prelude.scm` | 原样 | 值类型运行时 |
| `staticpy/static_stdlib.scm` | 原样 | 公共标准库 |
| `staticpy/static_build.sh` | 原样 | 上游构建脚本（**本项目的 `build.sh` 不使用它**） |

`static_translate.py` 相对 `/opt/ReScheme/static_translate.py` 的本地改动（`diff` 约 107 行）：

| 补丁 | 原因 |
|------|------|
| `is` / `is not`（None 判定）翻译 | 上游未处理，会生成非法的 `(None x y)` |
| `break` / `continue`（`call/cc` 实现） | 上游 while/for 只生成 `(let loop ...)`，无法提前退出 |
| 模块级变量进入 `module_env` 类型环境 | 上游只收 enum 常量，导致函数内引用模块变量被判 `undefined name` |
| 顶层 `AnnAssign` / `Assign` / `Expr` 语句翻译 | 上游丢弃顶层语句，`concat_src.py` 生成的 bundle 头部会失效 |
| `float` 映射为 `float`（非上游默认的 `double`） | C API（`sdcpp_adapter`）ABI 为 32 位 float，用 `double` 会传垃圾值 |
| 类型收集递归进 `while`/`if`/`for`/`with` 体 | 上游只扫函数直层，控制流内的类型标注丢失 |

项目侧另维护两处"胶水"（不属于编译器补丁）：

| 文件 | 说明 |
|------|------|
| `staticpy/static_build_comfycli.sh` | 本地构建脚本：项目相对路径、不链接 torch、支持 `GLIBC_SYSROOT`、产物输出到项目根目录 |
| `comfycli/comfycli_ffi.scm` | FFI 声明 + 上游缺失的内置（见下） |

> **升级上游时**：`prelude.scm` / `stdlib.scm` 可直接 `cp` 覆盖；
> `static_translate.py` 覆盖后需重放上表 6 处补丁（`diff /opt/ReScheme/static_translate.py staticpy/static_translate.py` 列出差异），
> 然后重跑 `./build.sh` 验证。

---

## FFI 约定（`comfycli/comfycli_ffi.scm`）

上游翻译器对 `extern fn ... from "lib"` 只生成 `foreign-procedure` 绑定，**不生成 `load-shared-object`**（库名映射表里没有 `sdcpp_adapter`）。共享库的加载由 `ffi.scm` 负责，作为 `static_build_comfycli.sh` 的第 3 个参数拼接到 stdlib 之后、用户代码之前：

```scheme
(load-shared-object "libsdcpp_adapter.so")   ;; 运行时经 LD_LIBRARY_PATH 解析
```

`sd_backend.static.py` 仍用 `extern fn ... from "sdcpp_adapter"` 声明；翻译器生成对应的 `foreign-procedure`，运行前由上面的 `load-shared-object` 完成加载。

`ffi.scm` 同时补齐上游 prelude 缺失、但 comfycli 需要的内置：

| 名称 | 说明 |
|------|------|
| `dict_keys` | `(hashtable-keys d)`，返回 vector（StaticPy 的 list = Scheme vector） |
| `is_none` / `is_some` | 判空谓词（`dict_get` 缺失返回 `#f`） |
| `is_link` | workflow 链接 `[node_id, index]` 判定 |
| `path_dirname` / `path_split` | 路径处理 |

这些名字通过 bundle 头部的一条 `from comfycli_builtins import ...` 让类型检查器认识（import 本身不生成代码，由 `concat_src.py` 注入）。

---

## 新版 StaticPy 的语言约束

上游编译器比旧 fork 更严格，写 `comfycli/*.static.py` 时须遵守：

1. **模块级变量可读**（本地补丁已把模块级 `Assign`/`AnnAssign` 收进 `module_env`，故 `NODE_GROUP`、`QUALITY_PREFIX` 这类常量可在函数内引用）；**函数内重新赋值**模块级变量不支持，仍须改为参数传递。
2. **`break` / `continue` 可用**（仅限直接位于循环体内；本地补丁用 `call/cc` 实现，见上文补丁表）。嵌套循环/函数返回仍须改写循环条件。
3. **`is None` / `is not None` 可用**（本地补丁，翻译为 `(eq? x #f)`）；其他 `is` 比较仍不支持。
4. **无 lambda 闭包 / `eval` / `exec` / 一等函数值**：`f = handler`（裸函数引用）会报 `undefined name`，需分发时按类分组为多个函数。
5. **无类继承 / try-except**。
6. **`list` 即 Scheme vector**：`len()` → `list_length`，索引 → `vector-ref`。

---

## 项目结构

```
comfycli/                       # StaticPy 编排层源码（存活模块）
├── cli_args.static.py          # CLI 参数解析
├── sd_backend.static.py        # stable-diffusion.cpp FFI extern 声明 + wrapper
├── nodes.static.py             # ComfyUI 节点定义
├── execution.static.py         # 节点 DAG 调度
├── main.static.py              # 入口
├── comfycli_ffi.scm            # FFI 加载 + 缺失内置（见上）
└── _bundle.static.py           # concat_src.py 生成的合并源码（构建产物）

staticpy/                       # 上游 StaticPy 工具链（translate.py 有本地补丁）+ 本地构建胶水
├── static_translate.py         # 翻译器（上游）
├── static_prelude.scm          # 运行时（上游）
├── static_stdlib.scm           # 标准库（上游）
├── static_build.sh             # 上游构建脚本（未使用）
└── static_build_comfycli.sh    # 本项目构建脚本

cpp/sd/                         # C++ 推理后端
├── src/adapters/sdcpp_adapter.h/.cpp
├── src/postproc/ src/adapters/ipadapter.*
└── build/libsdcpp_adapter.so   # 推理后端共享库

concat_src.py                   # 合并 comfycli/*.static.py → _bundle.static.py
build.sh                        # 编译入口
deploy.sh                       # 部署入口
```

> `concat_src.py` 只合并 sd.cpp 后端实际使用的模块（`cli_args / sd_backend / nodes / execution / main`）。torch 时代的模型栈（`sd / model_base / supported_models / model_detection / latent_formats / clip_model / controlnet / sample / lora / k_diffusion` 等）已废弃，不再进 bundle。

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
python3 concat_src.py
bash staticpy/static_build_comfycli.sh \
  comfycli/_bundle.static.py comfycli-bin comfycli/comfycli_ffi.scm
```

### StaticPy 编排层

`static_build_comfycli.sh` 内部三步：

```
 1. static_translate.py   _bundle.static.py → <stem>_code.ss
 2. 拼接 static_prelude.scm + static_stdlib.scm + comfycli_ffi.scm + 用户代码
    Chez compile-file AOT → .so
 3. objcopy 嵌入 Chez boot 文件 → .o
    C launcher + gcc 链接 → 独立 ELF 二进制（不链接 torch）
```

---

## 编译产物

| 文件 | 大小 | 说明 |
|------|------|------|
| `comfycli-bin` | ~4.2MB | 独立 ELF 二进制（主程序，含编排逻辑） |
| `comfycli-bin.so` | ~2.0MB | Chez AOT 编译的 Scheme 机器码（运行时加载） |
| `cpp/sd/build/libsdcpp_adapter.so` | ~134KB | sd.cpp 推理适配层（动态链接 `/opt/sd/build-dl` 的 `.so`） |

> 推理后端本体（`libstable-diffusion.so` ~39MB、`libggml*.so`、`libggml-cuda.so` ~129MB）来自
> `SD_BACKEND_DL=1` 的 sd.cpp 构建，由 `deploy.sh` 打包；见 [deploy.md](./deploy.md)。

---

## 增量编译

`static_build_comfycli.sh` 对 `static_prelude.scm` + `static_stdlib.scm` + `static_translate.py` + `comfycli_ffi.scm` + 用户代码计算 md5 缓存 key。输入不变时跳过 AOT 编译和 ELF 链接，直接复用 `/tmp/staticpy-cache/` 中的缓存产物。

`build.sh` 另外检测 `comfycli/*.static.py` 是否有更新，决定是否重新运行 `concat_src.py` 生成 bundle。

---

## 运行

### 本地开发

```bash
LD_LIBRARY_PATH=cpp/sd/build:/opt/sd/build-dl/bin \
  GGML_BACKEND_PATH=/opt/sd/build-dl/bin/libggml-cuda.so \
  ./comfycli-bin workflow.json --output-dir ./output
```

`libsdcpp_adapter.so` 由 `comfycli_ffi.scm` 在程序启动时经 `LD_LIBRARY_PATH` 加载。

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CHEZ_SCHEME_DIR` | `/opt/ChezScheme/ta6le` | Chez Scheme 安装目录 |
| `CHEZ_SCHEME` | `$CHEZ_SCHEME_DIR/bin/ta6le/scheme` | Chez 可执行文件 |
| `CHEZ_BOOT_DIR` | `/opt/ChezScheme/boot/ta6le` | Chez boot 文件目录 |
| `STATICPY_PYTHON` | `python3` | 翻译器所用 Python |
| `GLIBC_SYSROOT` | (未设置) | 目标 GLIBC sysroot，如 `/opt/deb/2.35` |
| `STATICPY_WARN` | `0` | 设为 `1` 时翻译器用 `--warn` 模式 |

---

## 扩展集成最佳实践（新增节点 / FFI / sd.cpp 补丁）

以移植 **AnimateDiff（逐帧 ControlNet + 运动 LoRA）** 为例：新增一个能力通常要自底向上改 3~4 层，每层各改各的，最后按序重编。

### Step 1 — sd.cpp 后端（`/opt/sd`，以补丁维护）

sd.cpp 已有的能力直接调；**没有的行为才改源码**（例如给 `sd_img_gen_params_t` 加 `control_frames`、在 `image.cpp` 里按帧堆 batch 走 ControlNet）。
`/opt/sd` 是上游仓（无推送权），改动以补丁纳入本仓库：

```bash
cd /opt/sd
git add -N <新增文件>                      # 新增文件必须 -N 才会进 diff
git diff > /opt/static_comfyui/cpp/sd/patches/sdcpp-freeu-sag-v2.patch
git reset -q
```

> 别把无关的临时文件（如误建的 `1`）带进补丁；生成后 `grep -c "^diff --git"` 核对文件数。

### Step 2 — 适配层 C API（`cpp/sd/src/adapters/sdcpp_adapter.{h,cpp}`）

- 在 `SDPipeline` 加方法（如 `set_control_frames`），`Impl` 里**自持所有权**（拷贝像素 buffer，别留悬空指针）。
- 加 C ABI 函数 `sd_pipeline_xxx`；C 侧不能过数组/复杂类型，故再提供 **CSV/换行字符串**变体（`..._csv`），方便 FFI。
- 在 `generate_video` / `generate_image` 里把参数填进 `sd_*_gen_params_t`。

### Step 3 — FFI 声明 + 包装（`comfycli/sd_backend.static.py`）

```python
extern fn sd_pipeline_xxx(pipeline: ptr, a: str, b: float) -> int from "sdcpp_adapter"

def sd_xxx(pipeline: ptr, a: str, b: float) -> int:
    return sd_pipeline_xxx(pipeline, a, b)
```

### Step 4 — 节点（`comfycli/nodes.static.py`）

- 取值用 `get_str/get_int/get_float`；**取上游句柄必须带注解**：`m: SDPipelineHandle = dict_get(inputs, "model")`。
- `register_node("ClassName", "显示名", "handler", (输出类型...), is_output)`。
- 加入 `dispatch_*` 的 `elif class_type == "..."`（**不能**用字符串→函数查表）；多输出节点要在 `register_group(..., 2)` 注册。
- 复用逻辑抽 helper，参数同样要带 `SDPipelineHandle` 注解（如 `setup_animatediff_extras(model, inputs)`）。

### Step 5 — 脚本 / 文档

把新参数用环境变量暴露给使用脚本（如 `anim.sh` 的 `LORA` / `CONTROL_NET` / `CONTROL_DIR`），并在本文 + `AGENTS.md` 记一笔。

### 踩过的坑（先看这里）

| 坑 | 规避 |
|----|------|
| 取句柄不写注解 → 译成裸变量，运行报 `variable pipeline is not bound` | 所有 `dict_get(inputs, ...)` 取的句柄都写 `x: SDPipelineHandle = ...` |
| 模块级 `for` 循环不生效 | 节点注册写显式 `register_node(...)`，不要循环注册 |
| FFI 传不了字符串数组 | C 侧收 CSV/换行串并 split；StaticPy 侧传单串 |
| 适配层帧 buffer 指针失效 | `std::vector<std::vector<uint8_t>>` 先 `reserve()` 再 `push_back`，否则 realloc 使已存的 `sd_image_t::data` 悬空 |
| sd.cpp 结构体插字段后偏移变化 | 结构体中间插字段会改后续偏移 → 必须连同适配层重编；`*params = {}` 会清零新字段 |
| 运动 LoRA 不生效 | sd.cpp lora 键约定是 `lora.<模型张量名>.lora_down`；motion 块在模型里嵌于 `model.diffusion_model.motion_module.*`，而 LoRA 键在 UNet 顶层 → 在 `name_conversion.cpp` 识别并重嵌；`convert_sep_to_dot` 保护词需含 `motion_module(s)/attention_blocks/temporal_transformer`，否则 `_` 被拆坏 |
| 改了 `/opt/sd` 却没重编库 | 见下方重编顺序第 1 步 |

### 重编顺序（改哪层编哪层）

```bash
bash cpp/sd/build_sd_dl.sh        # 1. 改过 /opt/sd 源码/补丁
bash cpp/sd/scripts/build.sh      # 2. 改过适配层（会先确保 libstable-diffusion.so）
./build.sh                        # 3. 改过 comfycli/*.static.py
```

---

## 与官方 StaticPy 的关系

本项目从 `/opt/ReScheme`（StaticPy 官方仓库）拷贝编译器核心到 `staticpy/`：

| 文件 | 来源 | 本地改动 |
|------|------|---------|
| `staticpy/static_translate.py` | StaticPy | **6 处补丁**（`is None` / `break`-`continue` / 模块级变量 / 顶层语句 / `float` ABI / 类型收集递归），见本文「工具链」章节 |
| `staticpy/static_prelude.scm` | StaticPy | 无（原样） |
| `staticpy/static_stdlib.scm` | StaticPy | 无（原样） |

编译器之外的 comfycli 特有内容全放在**项目侧**（`comfycli/comfycli_ffi.scm` + `staticpy/static_build_comfycli.sh`），因此升级上游只需重放 `static_translate.py` 的补丁。

> 核对差异：`diff /opt/ReScheme/static_translate.py staticpy/static_translate.py`

`cpp/sd/` 是独立维护的 stable-diffusion.cpp 适配器，不来自 StaticPy。

---

## 依赖的开源项目

| 项目 | 协议 | 说明 |
|------|------|------|
| [Chez Scheme](https://github.com/cisco/ChezScheme) | Apache 2.0 | AOT 编译后端，将 Scheme 编译为机器码 |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | MIT | C++ 推理后端（UNet/VAE/CLIP/Sampler/LoRA/ControlNet） |
| [StaticPy / ReScheme](https://github.com/quqiufeng/ReScheme) | — | 编排层编译工具链 |

---

## 部署

部署（打包依赖 `.so`、GLIBC 兼容层、SCP 到远程）请见 [deploy.md](./deploy.md)。

---

## 调试

| 问题 | 方法 |
|------|------|
| 类型错误 | `python3 staticpy/static_translate.py comfycli/_bundle.static.py` 看 stderr |
| Scheme 编译错误 | 查看 `/tmp/staticpy-cache/<stem>_<hash>.ss`（拼接后的完整 Scheme） |
| 运行时找不到 `.so` | `LD_LIBRARY_PATH` 需同时含 `cpp/sd/build` 与 `/opt/sd/build-dl/bin`；CUDA 后端另需 `GGML_BACKEND_PATH=/opt/sd/build-dl/bin/libggml-cuda.so`（本地可直接 `bash run.sh <args>`） |
| FFI 符号未找到 | `nm -D cpp/sd/build/libsdcpp_adapter.so \| grep symbol` |
| 生成阶段 OOM | sd.cpp 已对大图自动启用 VAE tiling；仍 OOM 时降分辨率或显式调小 tile |
