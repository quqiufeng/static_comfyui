# ComfyCLI — ComfyUI 的 StaticPy 编译重写

将 ComfyUI 1:1 重写为 StaticPy 编译二进制，零 Python 运行时依赖，单 ELF 文件部署。

> **给 AI 阅读者：如何理解 StaticPy 语言**
>
> StaticPy 是面向 AI Agent 的 ML 编译语言，语法是 Python 子集 + FFI 扩展。
> 阅读以下三个源文件即可完整掌握这门语言（比任何文档都快）：
>
> 1. **翻译器** — [`staticpy/static_translate.py`](./staticpy/static_translate.py)
>    - `typecheck_module()`（行~220）定义支持/不支持的 Python AST 节点
>    - `generate_scheme()`（行~1500）定义 Python 语法到 Scheme 的翻译规则
>    - `BUILTIN_MODULES`（行~60）列出内置模块映射
>    - `extern fn` 提取逻辑在文件开头的正则匹配中
> 2. **运行时** — [`staticpy/static_prelude.scm`](./staticpy/static_prelude.scm)
>    - 值类型映射：int→fixnum、float→flonum、bool→boolean
>    - 数组/dict/文件 I/O/JSON/字符串工具的实现
> 3. **标准库 FFI** — [`staticpy/static_stdlib.scm`](./staticpy/static_stdlib.scm)
>    - `foreign-procedure` 如何映射 `extern fn`
>    - libtorch 等外部库的 Scheme 封装
>
> **学习路径：**翻译器定义"能写什么"→ 运行时定义"编译成什么"→ 标准库定义"能调什么"。
> 三者对照阅读，无需单独的语言规范文档。

## 远景

ComfyUI 是优秀的 Stable Diffusion 工作流引擎，但 Python 解释器带来可移植性痛点：

| 问题 | 影响 |
|------|------|
| `pip install` 依赖地狱 | 多项目共用 venv 冲突，复现难 |
| Python 运行时开销 | 特别是模型加载、节点调度阶段的 GIL/解释开销 |
| 部署体积大 | 需要完整 Python 环境 + torch + 数十个 pip 包 |
| 打包困难 | PyInstaller/Nuitka 打包 torch 动辄 2GB+，且兼容性差 |

**ComfyCLI 方案：静态编译，无痛部署。**

- 用 **StaticPy**（Python 子集编译器）重写编排逻辑
- 用 **C++ FFI (`libtorch_std_helper.so`)** 封装所有推理计算
- 编译产物：**独立 ELF 二进制**（+ libtorch.so + libcudnn.so），零 Python
- 用 **code search（my_db）** 语义索引辅助定位源码，加速 1:1 翻译

## 快速开始

```bash
# 1. 编译
./build.sh

# 2. 打包部署包（含依赖 .so + GLIBC 兼容层）
GLIBC_TARGET=2.35 ./deploy.sh

# 3. 发送到远程服务器
GLIBC_TARGET=2.35 ./deploy.sh --scp user@remote_host

# 4. 远程运行（零 Python）
ssh user@remote_host "bash /opt/comfycli/run.sh workflow.json --output-dir ./output"
```

## 技术架构

```
  workflow.json                    用户输入
       │
  ┌────▼─────────────────────────────────────┐
  │  编排层（StaticPy 编译为机器码）          │
  │                                          │
  │  execution   节点 DAG 调度                │
  │  nodes       200+ 节点定义                │
  │  validate    prompt 校验                  │
  │  cache       输出缓存 (is_changed)        │
  └────┬─────────────────────────────────────┘
       │ extern fn FFI 调用
  ┌────▼─────────────────────────────────────┐
  │  推理后端（C++）                          │
  │                                          │
  │  libtorch_std_helper.so                  │
  │  ├── UNet forward (SD1.5/SDXL/FLUX/…)   │
  │  ├── VAE encode/decode (tiled)           │
  │  ├── CLIP/T5 tokenizer + encoder         │
  │  ├── Sampler (DDIM/Euler/DPM++/FM)       │
  │  ├── ControlNet forward + apply          │
  │  ├── LoRA load + merge + apply           │
  │  └── Image I/O + safetensors load        │
  └────┬─────────────────────────────────────┘
       │
  ┌────▼─────────────────────────────────────┐
  │  libtorch / cuDNN / cuBLAS / CUDA        │
  └──────────────────────────────────────────┘
```

### 编译流水线

```
.static.py  ──→  static_translate.py  ──→  .ss (Scheme)  ──→  Chez AOT  ──→  .so  ──→  ld  ──→  ELF
                                                                                             │
                                                                                    libtorch_std_helper.so
```

### 技术栈

| 层 | 组件 | 职责 |
|---|------|------|
| 源码语言 | StaticPy（Python 子集） | 无类继承、无异常、无 lambda 闭包，可直接 AOT 编译 |
| 编译器 | `static_translate.py` + Chez Scheme AOT | Python → Scheme → 机器码 |
| 推理后端 | `libtorch_std_helper.so`（C++） | UNet/VAE/CLIP/sampler/ControlNet/LoRA |
| 代码定位 | code search（my_db） | 语义搜索 + 调用链分析，辅助 1:1 翻译 |

## 核心优势

### 零 Python 依赖

```
# Python 版 ComfyUI — 需要
pip install torch torchvision ...  # ~2GB, 20+ 包
apt-get install ...                 # 系统依赖

# ComfyCLI — 只需要
LD_LIBRARY_PATH=./lib ./comfycli workflow.json  # 单文件
```

- 不需要 Python 解释器
- 不需要 `pip install` 任何包
- 不需要虚拟环境
- 不需要 conda

### 二进制部署

```
# 产物列表
comfycli            # 主二进制（ELF，包含所有编排逻辑）
libtorch_std_helper.so  # 推理后端
libtorch.so             # PyTorch C++ 运行时
libcudnn.so             # cuDNN
```

- 单目录部署，`scp` 即用
- 不依赖系统 Python 版本
- Docker 镜像极小（alpine 兼容）
- 嵌入式 / 离线环境友好

### 编译期优化

StaticPy 编译到机器码而非 CPython 字节码：

- 节点调度循环无 GIL 开销
- 函数调用为直接跳转而非 Python 属性查找
- 数据类型静态化，无装箱拆箱
- 启动时间≈进程启动时间，无 Python 模块导入开销

### code search 辅助翻译

用语义搜索定位 ComfyUI 对应源码，保证 1:1 翻译精度：

```bash
# 搜索 PromptExecutor 实现
cache_query "PromptExecutor execution loop" --repo /code/comfyui --type search

# 查看调用链上下文
cache_query PromptExecutor --repo /code/comfyui --type context --depth 2

# 搜索模型检测逻辑
cache_query "model config detect unet architecture" --repo /code/comfyui --type search
```

- 20k+ 代码块向量索引
- 2.6k+ 函数调用关系图
- 语义搜索而非关键词匹配

## 可行性

### 架构

分层架构（StaticPy 编排 → C++ FFI 推理 → libtorch/CUDA）在生产项目中已有验证。
`libtorch_std_helper.so` 已覆盖 UNet、VAE、CLIP、T5、Sampler、ControlNet、LoRA 等核心推理操作，
编排层翻译是机械工作，不存在根本性技术障碍。

### 工作量

重写 ComfyUI 的 200+ 节点是数量问题而非难度问题。每节点：
- 用 code search 定位对应源码（秒级）
- 翻译为 StaticPy 的 `@register` + `extern fn` 模式（分钟级）

复杂节点（model_config 多架构检测、sampler 步进逻辑）需更多关注，但核心路径已有 C++ helper 覆盖。

### code search 的作用

传统翻译的最大成本是手工在源码树中导航定位。本项目的 code search 系统（my_db）
已对 ComfyUI 做了全量语义索引（656 文件、20,141 chunks、2,633 函数、864 调用边）：

```
cache_query "model_config_from_unet" --type context --depth 3
  → 一次展示所有架构检测分支
  → 每个分支的 key pattern + 对应 config 结构
  → 调用链完整可见，无需手动翻文件
```

**源码定位成本接近于零，翻译变为纯粹的机械工作。**

### 对 StaticPy 的意义

ComfyUI 是 Python ML 生态中最复杂的纯推理项目之一：

- 200+ 节点、数十种模型架构（SD1.5 → SDXL → SD3 → FLUX）
- 动态 DAG 调度 + 显存管理 + 模型检测
- 依赖 torch、torchvision、transformers、scipy、Pillow 等数十个 pip 包

这个项目如果跑通，对 StaticPy 的推广有直接价值：**连 ComfyUI 这种体量的项目都能编译成单 ELF 零 Python 部署，其他 ML 项目只会更简单。** 它证明 StaticPy 不是玩具语言，而是能承载生产级推理引擎的编译工具。同时验证"Python 写编排 + C++ 写计算"的混合编译模式在大型项目中的可行性。

### 风险点

| 风险 | 等级 | 应对 |
|------|------|------|
| 冷门节点缺 C++ helper | 低 | libtorch_std_helper 覆盖 90% 场景，剩余用 StaticPy 纯逻辑实现 |
| model_config 多架构兼容 | 中 | code search 可快速定位各架构检测逻辑，C++ 侧需对应新增接口 |
| 异常处理缺失 | 低 | StaticPy 无 try/except，用返回值检查 + Result 类型替代 |

## 开发进度

```
Phase 0: 基础设施 (无 GPU 需求)
  [ ] folder_paths.static.py     路径管理
  [ ] cli_args.static.py         CLI 参数解析
  [ ] comfy_types.static.py      Node 类型定义

Phase 1: 模型检测 (纯逻辑, 可直接翻译)
  [ ] supported_models_base.static.py   模型基类
  [ ] supported_models.static.py        模型注册表
  [ ] model_detection.static.py         state_dict → 架构识别
  [ ] model_sampling.static.py          sigma 调度
  [ ] latent_formats.static.py          潜空间缩放

Phase 2: 模型加载 + 显存管理
  [ ] sd.static.py              safetensors 加载 → 模型对象
  [ ] model_base.static.py      模型基类
  [ ] model_management.static.py GPU 调度 (load/offload)
  [ ] lora.static.py            LoRA 合并

Phase 3: 组件级推理
  [ ] clip_model.static.py      CLIP encode
  [ ] k_diffusion/sampling.static.py  采样器
  [ ] sample.static.py          采样入口
  [ ] controlnet.static.py      ControlNet

Phase 4: 节点 → DAG → 入口
  [ ] nodes.static.py           200+ 节点定义
  [ ] execution.static.py       PromptExecutor
  [ ] main.static.py            CLI 入口

Phase 5: 端到端验证
  [ ] workflow SDXL → 图片输出
  [ ] --prompt 命令行模式验证
```

各阶段产出可独立编译、单独测试。Phase 0–1 甚至不需要 GPU。

## 局限

- 不支持 Python 动态特性（类继承、异常、eval、生成器）—— ComfyUI 核心逻辑均不需要
- 无自定义节点动态加载——自定义节点需编译期注册
- CLI 先行，无 WebSocket/HTTP UI
- 同步执行，无 asyncio

## 项目文件

### 文档

| 文件 | 说明 |
|------|------|
| [设计文档](./design.md) | 技术架构、模块映射、翻译策略、工程顺序 |
| [编译流水线](./BUILD.md) | 本地编译、增量编译、编译产物说明 |
| [部署文档](./deploy.md) | 纯二进制部署、GLIBC 兼容方案、远程要求 |
| [ComfyUI 分析报告](./comfyui_analysis.md) | code search 语义索引结果 (656 文件, 20k+ chunks) |
| [code search 使用文档](https://github.com/quqiufeng/my_db/blob/main/coding.md) | 语义搜索 + 向量查询工具用法 |

### 脚本

| 脚本 | 说明 | 常用命令 |
|------|------|---------|
| [`./build.sh`](./build.sh) | 编译 ELF 二进制 + C++ `.so` | `./build.sh` |
| [`./deploy.sh`](./deploy.sh) | 打包依赖 `.so` + GLIBC 兼容层 + SCP | `GLIBC_TARGET=2.35 ./deploy.sh --scp user@host` |

### 目录

| 目录 | 说明 |
|------|------|
| [`staticpy/`](./staticpy/) | StaticPy 编译器 (static_translate.py) + 运行时 (prelude / stdlib) + 构建脚本 |
| [`cpp/`](./cpp/) | C++ 推理后端 (libtorch_std_helper.h / .cpp + build_torch_std_helper.sh) |

### 参考

- [ComfyUI 源码](https://github.com/comfyanonymous/ComfyUI)
- [Chez Scheme (Cisco)](https://github.com/cisco/ChezScheme) — Apache 2.0 开源，AOT 编译后端
