# ComfyCLI — StaticPy 前端 + stable-diffusion.cpp 后端的 ComfyUI 编译重写

用 **StaticPy** 1:1 复刻 ComfyUI 的节点编排与 DAG 调度，底层推理全部交给 **stable-diffusion.cpp**（GGML/CUDA），最终编译成**独立 ELF 二进制**，零 Python 运行时依赖。

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
>    - 外部库（sd.cpp 等）的 Scheme 封装
>
> **学习路径：**翻译器定义"能写什么"→ 运行时定义"编译成什么"→ 标准库定义"能调什么"。
> 三者对照阅读，无需单独的语言规范文档。

## 远景

ComfyUI 是优秀的 Stable Diffusion 工作流引擎，但 Python 解释器带来可移植性痛点：

| 问题 | 影响 |
|------|------|
| `pip install` 依赖地狱 | 多项目共用 venv 冲突，复现难 |
| Python 运行时开销 | 节点调度、prompt 解析的解释器开销 |
| 部署体积大 | 需要完整 Python 环境 + torch + 数十个 pip 包 |
| 打包困难 | PyInstaller/Nuitka 打包 torch 动辄 2GB+，且兼容性差 |

**ComfyCLI 方案：静态编译编排层，复用成熟 C++ 推理后端。**

- 用 **StaticPy** 重写 ComfyUI 的编排逻辑（节点、DAG、调度、链接解析）
- 用 **stable-diffusion.cpp** 承担所有推理计算（UNet、VAE、CLIP、采样、ControlNet、LoRA 等）
- 通过 **FFI** 调用 `libsdcpp_adapter.so` 的 C API
- 编译产物：**独立 ELF 二进制**（+ `libsdcpp_adapter.so` + GLIBC 兼容层），零 Python
- 用 **code search（my_db）** 语义索引辅助定位源码，加速 1:1 翻译

## 快速开始

```bash
# 1. 编译
./build.sh

# 2. 运行（workflow 模式，动态后端）
LD_LIBRARY_PATH=cpp/sd/build:/opt/sd/build-dl/bin \
  GGML_BACKEND_PATH=/opt/sd/build-dl/bin/libggml-cuda.so \
  ./comfycli-bin workflow.json --output-dir ./output

# 3. 或 prompt 模式
LD_LIBRARY_PATH=cpp/sd/build:/opt/sd/build-dl/bin \
  GGML_BACKEND_PATH=/opt/sd/build-dl/bin/libggml-cuda.so \
  ./comfycli-bin --checkpoint /data/models/image/sd_xl_base_1.0.safetensors \
  --prompt "a photo of a cat" --output ./out.png

# 4. 打包部署包（含依赖 .so + GLIBC 兼容层）
GLIBC_TARGET=2.35 ./deploy.sh

# 5. 发送到远程服务器
GLIBC_TARGET=2.35 ./deploy.sh --scp user@remote_host

# 6. 远程运行（零 Python）
ssh user@remote_host "bash /opt/comfycli/run.sh workflow.json --output-dir ./output"
```

### workflow JSON 示例

`test_remote_2560.json` 是一个完整的 HiResFix 工作流，包含 DiffusionModelLoader → CLIPTextEncode → HiResFix → VAEDecode → SaveImage 节点链，支持 FreeU、SAG、VAE Tiling 及后处理（clarity/sharpen/smart_sharpen/edge_sharpen）参数：

```json
{
  "1": {
    "class_type": "DiffusionModelLoader",
    "inputs": {
      "diffusion_model_name": "z_image_turbo-Q5_K_M.gguf",
      "llm_name": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
      "vae_name": "ae.safetensors"
    }
  },
  "2": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "solo,single woman,half body portrait of a young woman, soft natural lighting, elegant pose, studio lighting, sharp eyes",
      "clip": ["1", 0]
    }
  },
  "3": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "blurry, low quality, worst quality, bad anatomy, deformed, watermark, text, logo",
      "clip": ["1", 0]
    }
  },
  "4": {
    "class_type": "HiResFix",
    "inputs": {
      "model": ["1", 0],
      "positive": ["2", 0],
      "negative": ["3", 0],
      "width": 2560, "height": 1440,
      "steps": 20, "cfg": 2.5,
      "sampler_name": "euler", "scheduler": "discrete", "seed": 42,
      "hires_steps": 45, "hires_strength": 0.35,
      "freeu": 1, "freeu_b1": 1.3, "freeu_b2": 1.4,
      "clarity": 0.2, "sharpen": 0.3, "sharpen_radius": 1,
      "smart_sharpen": 0.5, "smart_sharpen_radius": 2,
      "edge_sharpen": 1.5, "edge_sharpen_radius": 2, "edge_sharpen_threshold": 0.3
    }
  },
  "5": { "class_type": "VAEDecode", "inputs": { "samples": ["4", 0], "vae": ["1", 2] } },
  "6": { "class_type": "SaveImage", "inputs": { "images": ["5", 0], "filename_prefix": "output" } }
}
```

使用该 JSON 生成图片：
```bash
LD_LIBRARY_PATH=cpp/sd/build:/opt/sd/build-dl/bin \
  GGML_BACKEND_PATH=/opt/sd/build-dl/bin/libggml-cuda.so \
  ./comfycli-bin test_remote_2560.json --output-dir ./output
```

## 已实现节点

以下 ComfyUI 节点已在 `comfycli/nodes.static.py` 中实现，可直接在工作流 JSON 中使用。节点数量持续按需求扩展。

### 模型加载

| 节点 | 输出 | 说明 |
|------|------|------|
| `CheckpointLoaderSimple` | `MODEL`, `CLIP`, `VAE` | 加载 SDXL/SD1.5 等 checkpoint（safetensors），sd.cpp 后端自动识别 |
| `DiffusionModelLoader` | `MODEL`, `CLIP`, `VAE` | 加载 GGUF 格式的 diffusion 模型 + LLM 文本编码器（Flux 风格） |
| `DualCLIPLoader` | `CLIP` | 兼容 ComfyUI 拓扑的占位节点（后端 pipeline 已内含 CLIP） |

### 条件 / 文本编码

| 节点 | 输出 | 说明 |
|------|------|------|
| `CLIPTextEncode` | `CONDITIONING` | 文本编码，后端 CLIP 内部处理 |
| `CLIPSetLastLayer` | `CLIP` | 设置 CLIP 输出层（当前为兼容占位，后端使用默认层） |
| `ConditioningCombine` | `CONDITIONING` | 拼接两个条件文本，用逗号连接 |
| `ConditioningConcat` | `CONDITIONING` | 同 `ConditioningCombine`，兼容标准 ComfyUI 命名 |
| `ConditioningAverage` | `CONDITIONING` | 按 `conditioning_to_strength` 决定文本拼接顺序 |

### 图像 / 潜空间

| 节点 | 输出 | 说明 |
|------|------|------|
| `EmptyLatentImage` | `LATENT` | 创建指定尺寸的空白潜空间 |
| `LatentUpscale` | `LATENT` | 直接修改 `LATENT` 的 width/height（后端采样时使用目标尺寸） |
| `LatentCrop` | `LATENT` | 同 `LatentUpscale`，裁剪到目标尺寸 |
| `LoadImage` | `IMAGE`, `MASK` | 从文件路径加载图片，返回图片路径（当前 MASK 为占位） |
| `PreviewImage` | `IMAGE` | 透传图片路径，用于工作流可视化 |
| `VAEDecode` | `IMAGE` | 兼容节点，后端采样已完成 VAE decode |
| `SaveImage` | - | 输出节点，保存图片到指定目录 |

### 采样与优化

| 节点 | 输出 | 说明 |
|------|------|------|
| `KSampler` | `LATENT` | 核心采样节点，支持 sampler/scheduler/seed/cfg/steps |
| `KSamplerAdvanced` | `LATENT`, `IMAGE` | 扩展参数（start_at_step / end_at_step / return_noise 等占位） |
| `HiResFix` | `LATENT` | 原生高清修复节点，支持 hires_steps/hires_strength、FreeU、SAG、VAE Tiling 及后处理 |
| `ADetailer` | `IMAGE` | 对生成结果进行局部重绘修复 |

### 模型增强 / 风格注入

| 节点 | 输出 | 说明 |
|------|------|------|
| `LORALoader` | `MODEL` | 加载 LoRA 并合并到当前 pipeline |
| `IPAdapterApply` | `MODEL` | 应用 IPAdapter 风格/人脸参考 |
| `CLIPVisionLoader` | `CLIP_VISION` | 加载 CLIP Vision ONNX 模型 |
| `IPAdapterModelLoader` | `IPADAPTER` | 加载 IPAdapter ONNX 模型 |

### 工具

| 节点 | 输出 | 说明 |
|------|------|------|
| `Reroute` | `*` | 透传任意输入，仅用于整理连线 |

> **注意**：StaticPy 无运行期自定义节点加载能力。新增节点需在 `comfycli/nodes.static.py` 中注册并重新编译。

## 技术架构

```
  workflow.json / --prompt --checkpoint
       │
  ┌────▼─────────────────────────────────────┐
  │  编排层（StaticPy 编译为机器码）          │
  │                                          │
  │  execution   节点 DAG 拓扑排序 + 输入链接解析│
  │  nodes       ComfyUI 节点定义（逐步补齐）  │
  │  cli_args    命令行参数解析               │
  │  main        CLI 入口                     │
  └────┬─────────────────────────────────────┘
       │ extern fn FFI 调用
  ┌────▼─────────────────────────────────────┐
  │  推理后端（stable-diffusion.cpp）          │
  │                                          │
  │  libsdcpp_adapter.so                     │
  │  ├── sd_pipeline_create / load / free      │
  │  ├── sd_pipeline_generate (txt2img)       │
  │  └── 内部封装：UNet/VAE/CLIP/Sampler/ControlNet/LoRA
  └────┬─────────────────────────────────────┘
       │
  ┌────▼─────────────────────────────────────┐
  │  GGML / CUDA / cuBLAS / cuDNN             │
  └──────────────────────────────────────────┘
```

StaticPy 只负责**编排**：解析 workflow JSON、拓扑排序、把节点输入解析为正确的 C API 参数、调用 sd.cpp 生成图片。所有张量计算都在 sd.cpp 内部完成。

### 编译流水线

```
comfycli/*.static.py  ──→  concat_src.py  ──→  _bundle.static.py
                                                │
                                                ▼
                         staticpy/static_translate.py  ──→  .ss (Scheme)
                                                │
                                                ▼
                         staticpy/static_build.sh  ──→  Chez AOT compile-file
                                                │
                                                ▼
                         C launcher + objcopy + gcc  ──→  comfycli-bin (ELF)
```

### 技术栈

| 层 | 组件 | 职责 |
|---|------|------|
| 源码语言 | StaticPy（Python 子集） | 无类继承、无异常、无 lambda 闭包，可直接 AOT 编译 |
| 编译器 | `static_translate.py` + Chez Scheme AOT | Python → Scheme → 机器码 |
| 推理后端 | `libsdcpp_adapter.so`（stable-diffusion.cpp） | UNet/VAE/CLIP/sampler/ControlNet/LoRA |
| 代码定位 | code search（my_db） | 语义搜索 + 调用链分析，辅助 1:1 翻译 |

## 核心优势

### 零 Python 依赖

```bash
# Python 版 ComfyUI — 需要
pip install torch torchvision ...  # ~2GB, 20+ 包
apt-get install ...                 # 系统依赖

# ComfyCLI — 只需要
LD_LIBRARY_PATH=./lib ./comfycli-bin workflow.json  # 单文件
```

- 不需要 Python 解释器
- 不需要 `pip install` 任何包
- 不需要虚拟环境
- 不需要 conda

### 二进制部署

```bash
# 产物列表
comfycli-bin            # 主二进制（ELF，包含所有编排逻辑）
comfycli-bin.so         # Chez AOT 编译产物
libsdcpp_adapter.so     # stable-diffusion.cpp 推理后端
lib/                    # GLIBC 兼容层 + 依赖运行时
```

- 单目录部署，`scp` 即用
- 不依赖系统 Python 版本
- Docker 镜像极小（alpine 兼容）
- 嵌入式 / 离线环境友好

### 编排层编译期优化

StaticPy 编译到机器码而非 CPython 字节码：

- 节点调度循环无 GIL 开销
- 函数调用为直接跳转而非 Python 属性查找
- 数据类型静态化，无装箱拆箱
- 启动时间≈进程启动时间，无 Python 模块导入开销

### 复用成熟推理后端

stable-diffusion.cpp 已经实现并验证：

- SD1.5 / SDXL / SD3 / FLUX 等多架构
- UNet 采样、VAE decode、CLIP encode
- LoRA、ControlNet、HiRes Fix、VAE tiling 等
- GGML 跨后端 + CUDA 高性能

StaticPy 不需要重新实现这些，只需要把它们包装成 ComfyUI 节点。

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

分层架构（StaticPy 编排 → sd.cpp 推理 → GGML/CUDA）在生产项目中已有验证。
stable-diffusion.cpp 已经覆盖 UNet、VAE、CLIP、Sampler、ControlNet、LoRA 等核心推理操作，
编排层翻译是机械工作，不存在根本性技术障碍。

### 工作量

重写 ComfyUI 的 200+ 节点是数量问题而非难度问题。每节点：
- 用 code search 定位对应源码（秒级）
- 翻译为 StaticPy 的 `register_node` + 参数解析 + `sd_pipeline_*` 调用（分钟级）

复杂节点（多模型架构、采样器步进逻辑）需更多关注，但核心路径已由 sd.cpp 覆盖。

### code search 的作用

传统翻译的最大成本是手工在源码树中导航定位。本项目的 code search 系统（my_db）
已对 ComfyUI 做了全量语义索引（656 文件、20,141 chunks、2,633 函数、864 调用边）：

```bash
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

这个项目如果跑通，对 StaticPy 的推广有直接价值：**连 ComfyUI 这种体量的项目都能把编排层编译成单 ELF 零 Python 部署，其他 ML 项目只会更简单。** 它证明 StaticPy 不是玩具语言，而是能承载生产级推理引擎编排的编译工具。同时验证"Python 写编排 + C++ 写计算"的混合编译模式在大型项目中的可行性。

## 开发进度

```
Phase 0: 基础设施 (无 GPU 需求)
  [x] folder_paths.static.py     路径管理
  [x] cli_args.static.py         CLI 参数解析
  [x] comfy_types.static.py      Node 类型定义

Phase 1: 模型检测 (纯逻辑, 可直接翻译)
  [x] supported_models_base.static.py   模型基类
  [x] supported_models.static.py        模型注册表
  [x] model_detection.static.py         state_dict → 架构识别
  [x] model_sampling.static.py          sigma 调度
  [x] latent_formats.static.py          潜空间缩放

Phase 2: 模型加载 + 显存管理（已切换为 sd.cpp 后端）
  [x] sd_backend.static.py       stable-diffusion.cpp C API FFI 封装
  [~] sd.static.py / model_base.static.py / model_management.static.py / lora.static.py
      — 传统 PyTorch 模型栈已由 sd.cpp 内置替代，当前通过 C API 直接加载/推理

Phase 3: 组件级推理
  [x] 由 libsdcpp_adapter.so 统一提供：CLIP encode、UNet 采样、VAE decode、ControlNet 等
  [ ] 如需在 StaticPy 层暴露更多采样参数/ControlNet 节点，可继续扩展

Phase 4: 节点 → DAG → 入口（MVP 已通）
  [x] nodes.static.py           已支持节点：CheckpointLoaderSimple / DualCLIPLoader / CLIPTextEncode / CLIPSetLastLayer / ConditioningCombine / ConditioningConcat / ConditioningAverage / EmptyLatentImage / LatentUpscale / LatentCrop / KSampler / KSamplerAdvanced / LORALoader / DiffusionModelLoader / HiResFix / ADetailer / IPAdapterApply / CLIPVisionLoader / IPAdapterModelLoader / LoadImage / PreviewImage / Reroute / VAEDecode / SaveImage
  [x] execution.static.py       PromptExecutor（拓扑排序 + 输入链接解析）
  [x] main.static.py            CLI 入口（workflow JSON / --checkpoint --prompt）
  [ ] 200+ 完整节点集（按需逐步补充）

Phase 5: 端到端验证（workflow + prompt 均已通）
  [x] workflow SDXL → 图片输出（1024×1024，sd_xl_base_1.0 + clip_l/clip_g）
  [x] --prompt 命令行模式验证（自动生成 CheckpointLoaderSimple / KSampler / SaveImage 节点）
```

各阶段产出可独立编译、单独测试。Phase 0–1 甚至不需要 GPU。

## 局限

- 不支持 Python 动态特性（类继承、异常、eval、生成器）—— ComfyUI 核心编排逻辑均不需要
- 无自定义节点动态加载——自定义节点需编译期注册
- CLI 先行，无 WebSocket/HTTP UI
- 同步执行，无 asyncio
- 已实现 24 个核心节点，完整 ComfyUI 节点集仍在按需扩展中

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
| [`./deploy.sh`](./deploy.sh) | 打包依赖 `.so` + GLIBC 兼容层 + SCP | `GLIBC_TARGET=2.35 ./deploy.sh --scp user@remote_host` |
| [`./concat_src.py`](./concat_src.py) | 按依赖顺序合并 `comfycli/*.static.py` 为 `_bundle.static.py` | `/data/venv/bin/python3 concat_src.py` |

### 目录

| 目录 | 说明 |
|------|------|
| [`staticpy/`](./staticpy/) | StaticPy 编译器 (static_translate.py) + 运行时 (prelude / stdlib) + 构建脚本 |
| [`comfycli/`](./comfycli/) | StaticPy 编排层源码（节点、DAG、CLI） |
| [`cpp/sd/`](./cpp/sd/) | stable-diffusion.cpp 适配器 (`sdcpp_adapter.h/.cpp` + build 脚本) |

### 参考

- [ComfyUI 源码](https://github.com/comfyanonymous/ComfyUI)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [Chez Scheme (Cisco)](https://github.com/cisco/ChezScheme) — Apache 2.0 开源，AOT 编译后端
