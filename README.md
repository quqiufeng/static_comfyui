# ComfyCLI — ComfyUI 的 StaticPy 编译重写

将 ComfyUI 1:1 重写为 StaticPy 编译二进制，零 Python 运行时依赖，单 ELF 文件部署。

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

## 当前状态

| 模块 | 状态 |
|------|------|
| 架构设计 | ✅ design.md 完成 |
| 源码索引 | ✅ ComfyUI 全量索引 (656 文件, 20,141 chunks) |
| execution 引擎 | 🔄 翻译中 |
| nodes 定义 | ⏳ 待开始 |
| model_config | ⏳ 待开始 |
| model_management | ⏳ 待开始 |
| CLI 入口 | ⏳ 待开始 |

## 局限

- 不支持 Python 动态特性（类继承、异常、eval、生成器）—— ComfyUI 核心逻辑均不需要
- 无自定义节点动态加载——自定义节点需编译期注册
- CLI 先行，无 WebSocket/HTTP UI
- 同步执行，无 asyncio

## 参考

- [ComfyUI 源码](https://github.com/comfyanonymous/ComfyUI)
- [StaticPy 编译器](/opt/ReScheme)
- [libtorch_std_helper](//opt/ReScheme/libtorch_std_helper.h)
- [code search 分析报告](/opt/my_db/reports/comfyui_analysis.md)
