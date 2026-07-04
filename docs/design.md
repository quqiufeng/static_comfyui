# static_comfyui — 基于 StaticPy 的 ComfyUI 二进制运行时

## 核心理念

**把 ComfyUI 的全部推理管线编译成单文件 ELF 二进制**，消除 Python 运行时和 diffusers 依赖。

```
ComfyUI 源码 (comfyui_ref/)
    ↓ 对齐参考
StaticPy 代码 (.static.py)    ←  高阶管线编排
    ↓ compiler/translate.py
Scheme S-表达式
    ↓ foreign-procedure FFI
cpp_runtime/libtorch_std_helper  (extern "C" API)    ←  张量基元 + 模型前向
    ↓
PyTorch C++ API (libtorch)  / cuBLAS / cuDNN
    ↓
Chez Scheme AOT compile-file → runtime.so
    ↓
C launcher (嵌入 scheme+boot+runtime.so)
    + lib/ (PyTorch + CUDA .so)
    ↓
部署：单文件 ELF + lib/ 目录
    部署机只需：NVIDIA 驱动 + libc
    不需要：Python / PyTorch
```

## 架构分层

```
┌──────────────────────────────────────────────────┐
│             StaticPy 用户代码                     │
│  (pipeline orchestration, sampling loop,          │
│   image post-processing, CLI)                    │
│  extern fn 调用 C++ 运行时                        │
├──────────────────────────────────────────────────┤
│  cpp_runtime/libtorch_std_helper  (C++ extern)   │
│  ┌──────────────┐  ┌────────────────────────┐    │
│  │  张量基元层    │  │    模型前向层           │    │
│  │ conv2d        │  │ torch_std_sd1_unet_   │    │
│  │ matmul        │  │ forward()              │    │
│  │ group_norm    │  │ torch_std_sdxl_unet_   │    │
│  │ silu/relu/    │  │ forward()              │    │
│  │ gelu/softmax  │  │ torch_std_flux_        │    │
│  │ add/sub/mul/  │  │ forward()              │    │
│  │ div           │  │ torch_std_clip_text_   │    │
│  │ cat/reshape/  │  │ forward()              │    │
│  │ transpose     │  │ torch_std_vae_encode_  │    │
│  │ slice/clone   │  │ tiled()                │    │
│  │ attention     │  │ torch_std_controlnet_  │    │
│  │ pool/upsample │  │ forward()              │    │
│  │ ...           │  │ torch_std_sample_*()   │    │
│  └──────┬───────┘  │ torch_std_lora_apply()  │    │
│         │          │ torch_std_safetensors_  │    │
│         │          │ load()                  │    │
│         │          │ torch_std_gguf_load()   │    │
│         │          │ torch_std_clip_         │    │
│         │          │ tokenizer_*()           │    │
│         │          │ torch_std_t5_           │    │
│         │          │ tokenizer_*()           │    │
│         │          │ torch_std_jit_load()    │    │
│         │          │ torch_std_image_*()     │    │
│         │          └────────────────────────┘    │
│              ↓ 内部调用 ↓                         │
│         PyTorch C++ API / cuBLAS / cuDNN          │
├──────────────────────────────────────────────────┤
│           Chez Scheme AOT 编译器                  │
│           .ss → .so (native machine code)         │
├──────────────────────────────────────────────────┤
│           ELF 打包器 (launcher/)                   │
│           嵌入 .so + scheme + boot → 单文件      │
└──────────────────────────────────────────────────┘
```

### Layer 1: StaticPy — 管线编排层

StaticPy 是 Python 的**值类型静态子集**，编译为 Chez Scheme fixnum/flonum，无 Python 对象开销：

```python
extern fn torch_std_conv2d(input: ptr, weight: ptr, bias: ptr,
    stride_h: int, stride_w: int, pad_h: int, pad_w: int,
    dilation_h: int, dilation_w: int, groups: int) -> ptr from "torch_std_helper"

def unet_resblock_forward(x: ptr, emb: ptr,
                          w_conv: ptr, b_conv: ptr,
                          n: int, c: int, h: int, w: int) -> ptr:
    h1: ptr = torch_std_group_norm(x, 32, w_gn, b_gn, 1e-5)
    h1 = torch_std_silu(h1)
    h1 = torch_std_conv2d(h1, w_conv, b_conv, n, c, c, h, w, 3, 1, 1)
    return h1
```

**规则**：
- `int` / `float` / `bool` — 值类型（fixnum/flonum），零开销
- `ptr` — 不透明 GPU 张量指针（由 C++ runtime 管理）
- `list[float]` — C float 数组指针（`foreign-alloc`），由 guardian GC 自动回收
- `extern fn ... from "torch_std_helper"` — 声明 C++ 运行时函数，编译为 `foreign-procedure`
- 无 class、无闭包、无异常
- 循环只有 `while` / `for range`

**翻译器已知问题**（`compiler/translate.py`）：
- `CLASS_INSTANCES` 追踪遗漏通过包装函数创建的实例（如 `static_KSampler`），导致方法分发翻译为裸 `(obj method args...)` 而非 `(ClassName_method obj args...)`
- `dict.get(key, default)` 翻译为 `(obj get key default)`，需手动替换为 `(hashtable-ref obj key default)`
- 局部变量名被加上 `static_` 前缀（如 `fn` → `static_fn`），导致函数调用 `(fn ...)` 变成 `(static_fn ...)`

### Layer 2: cpp_runtime — C++ 运行时层

**该层是项目的核心资产**（位于 `cpp_runtime/libtorch_std_helper.cpp`），提供涵盖张量基元和完整模型前向的 `extern "C"` API（~5200 行）。

关键模块：

| 模块 | 函数 | 说明 |
|------|------|------|
| 张量创建 | `torch_std_zeros`, `ones`, `empty`, `full`, `randn`, `clone`, `detach` | |
| 数学运算 | `add`, `sub`, `mul`, `div`, `pow`, `exp`, `log` | |
| 激活函数 | `silu`, `relu`, `gelu`, `softmax` | |
| 形状操作 | `reshape`, `transpose`, `permute`, `cat`, `narrow`, `slice` | |
| 线性代数 | `matmul`, `linear`, `conv2d` | |
| 归一化 | `group_norm`, `layer_norm` | |
| 采样器 | `torch_std_sample_euler`, `dpmpp_2m`, `ddim`, `fm_sigmas` | |
| 模型前向 | `torch_std_sd15_unet_forward_dict`, `torch_std_sdxl_unet_forward` | 重量级前向 |
| CLIP | `torch_std_clip_text_forward`, `sdxl_dual_clip`, tokenizer | |
| safetensors | `torch_std_safetensors_load/count/name/tensor/free` | 直接加载 .safetensors |
| VAE | `torch_std_jit_load/forward` | JIT traced VAE |
| 图像 I/O | `torch_std_load/save_image` | PNG 编解码 |

**SDXL UNet 前向** (`torch_std_sdxl_unet_forward`):

当前实现支持标准 SDXL 格式。模型 `sd_xl_base_1.0.safetensors` 是标准 SDXL 1.0 Base，但 SDXL 本身存在多种内部命名惯例。本仓库的模型使用 **DDPM/LDM 命名体系**（`input_blocks.*`/`output_blocks.*`/`label_emb.*`），而非部分参考实现使用的简化命名（`down_blocks.*`/`up_blocks.*`/`add_embed.*`）。两者都是标准 SDXL，只是命名不同。

`st_to_map()` 使用前缀匹配自动归一化权重名称：
- `model.diffusion_model.` → 空（去掉前缀后按原始 key 名查找）
- `model.` → 空
- `first_stage_model.` → 空
- `conditioner.embedders.0/1.` → 空

**设计原则**：
1. **所有函数 extern "C"**，直接可被 Chez Scheme 的 `foreign-procedure` 调用
2. **模型前向**在 C++ 层实现（而非 StaticPy），因为 PyTorch C++ API 直接调用 cuDNN/cuBLAS，无跨语言开销
3. **StaticPy 负责管线编排**——采样循环、多模型串联、参数控制、CLI 交互
4. **重量级模型（UNet/VAE/CLIP）的前向在 C++**，轻量级逻辑在 StaticPy

### Layer 3: 编译器 (compiler/)

| 文件 | 职责 |
|------|------|
| `translate.py` | StaticPy → Scheme S-表达式（883 行） |
| `prelude.scm` | 值类型运行时：C float/int/ptr 数组、GC guardian、文件 I/O、JSON、HTTP、libm |
| `stdlib.scm` | libtorch_std_helper 的 FFI 声明 + 辅助函数 |

**编译流程**：

```
StaticPy (.static.py)
    │ translate.py 解析 extern fn 声明、转换 AST
    ▼
Scheme S-表达式 + foreign-procedure 绑定
    │ cat prelude.scm + stdlib.scm + output.ss → merged.ss
    ▼
merged.ss
    │ Chez Scheme compile-file
    ▼
runtime.so  (原生机器码)
    │ 链接 libtorch_std_helper.so (PyTorch C++)
    ▼
单文件 ELF  (scheme + boot + .so 全部嵌入)
```

**编译已知限制**：
- `compile-file` 批量编译此代码库时触发 Chez 10.5 编译器 bug（segfault）；改用 `read`/`eval` 循环源文件加载作为 workaround

## 编译管线 (build.sh→launcher/)

```
Input:  build_out/input.ss  (translated Scheme)
        │
        ▼
[1] cat compiler/prelude.scm compiler/stdlib.scm input.ss > merged.ss
        │
        ▼
[2] scheme --compile-file merged.ss → runtime.so
        │
        ▼
[3] g++ -shared -fPIC cpp_runtime/libtorch_std_helper.cpp → libtorch_std_helper.so
        │
        ▼
[4] 收集 PyTorch/CUDA .so → build_out/deploy/lib/
        │
        ▼
[5] C launcher (launcher.c): 嵌入 scheme+boot+runtime.so
        │
        ▼
Output: build_out/deploy/sd_generate.elf
```

## 部署包结构

```
build_out/deploy/
├── sd_generate.elf          ← 单文件 ELF（嵌入 scheme 二进制 + boot + runtime.so）
└── lib/
    ├── libtorch_std_helper.so   ← 我们的 C++ 运行时包装
    ├── libtorch.so              ← PyTorch 核心库
    ├── libc10.so                ← PyTorch 基础库
    ├── libtorch_cpu.so          ← PyTorch CPU 后端
    ├── libtorch_cuda.so         ← PyTorch CUDA 后端
    ├── libc10_cuda.so           ← PyTorch CUDA 基础库
    ├── libcudart.so.12          ← CUDA 运行时
    ├── libcublas.so.12          ← CUDA BLAS
    ├── libcublasLt.so.12        ← CUDA BLAS (轻量)
    ├── libcudnn.so.9            ← cuDNN
    └── ...                      ← 其他 DT_NEEDED 依赖
```

### 部署要求（最低）

| 依赖 | 要求 |
|------|------|
| OS | Linux x86_64（glibc） |
| GPU | NVIDIA GPU + 驱动（≥CUDA 12.0 兼容） |
| 磁盘 | ~3GB（ELF + PyTorch/CUDA lib/ 目录） |
| Python | **不需要** |
| PyTorch | **不需要安装** |

## 权重管理

### 模型权重来源

所有模型权重直接从 HuggingFace 或 CivitAI 下载的 `.safetensors` 文件加载：

| 模型 | 路径 | 大小 | 格式 |
|------|------|------|------|
| SDXL UNet | `sd_xl_base_1.0.safetensors` | 6.5 GB | safetensors (F16) |
| CLIP-L | 从 JIT 导出 `clip_l.pt` | 470 MB | TorchScript |
| CLIP-G | 从 JIT 导出 `clip_g.pt` | ~2 GB | TorchScript |
| VAE | 从 JIT 导出 `vae_decode.pt` | 320 MB | TorchScript |

### safetensors 加载器

`torch_std_safetensors_load` 直接从 `.safetensors` 文件中解析 JSON header + 读取张量数据。

**已知修复**（2026-07-04）：
1. **`skip_ws` 顺序 bug**：`parse_safetensors_header` 在设置 `obj_start` 后才调用 `skip_ws()`，导致 JSON 值字符串开头多了一个空格（`" {"...`），使后续 `json_get_string` 查找 key 失败。修复：将 `skip_ws()` 移到 `obj_start = i` 之前。
2. **括号嵌套 bug**：`json_get_string` 遇到数组内逗号（如 `[1280, 320]`）就停止读取，导致 `json_get_ints` 只解析到 1 个维度，张量变为 0 维。修复：添加 `brackets` 计数器，在 `[...]` 内时忽略逗号。
3. **`if (n < 3)` debug 残留**：调试代码残留导致只有前 3 个 key 的 value 被存储。修复：移除残留条件。

### JIT 模型加载

CLIP 和 VAE 使用 PyTorch JIT traced 模型（`.pt` 文件），通过 `torch_std_jit_load` 加载到 GPU。

**已知修复**：
- CLIP JIT 模型导出在 GPU 上，但输入 token tensor 创建在 CPU。修复：`torch_std_clip_text_forward` 和 `torch_std_sdxl_dual_clip` 中添加 `input.to(torch::kCUDA)`。

## 管线状态（2026-07-04）

### 已通过

| 阶段 | 状态 | 说明 |
|------|------|------|
| safetensors 加载 | ✅ | 2516 个 key 全部正确加载 |
| 模型类型检测 | ✅ | `cond` 代替 `if` 链，正确返回 SDXL |
| SDXL pipeline 分支 | ✅ | `mt` 变量控制调用 `SD15Pipeline_txt2img` 或 `SDXLPipeline_txt2img` |
| CLIP 编码 | ✅ | 双 CLIP（L + G）正确编码，`pooled_emb` 提取通过 `torch_std_sdxl_get_pooled` |
| 采样器初始化 | ✅ | `KSampler`, `static_get_sigmas`, `static_get_sampler` |
| wrapped_fn 调用 | ✅ | 移除多余的 `self` 参数 |
| UNet 前向 `time_embed` | ✅ | 时间步嵌入 320 → 1280 |
| UNet 前向 `label_emb` | ✅ | 文本池化 2048 + CLIP-L 池化 768 = 2816 → 1280 |
| UNet 前向 encoder | ✅ | `input_blocks.0` 到 `.8`，含下采样 + attention |
| UNet 前向 mid | ✅ | `middle_block.0/1/2` |

### 进行中/阻塞

| 阶段 | 状态 | 说明 |
|------|------|------|
| UNet 前向 decoder | ❌ | `output_blocks` 处理中维度不匹配（32 vs 3），skip connection 空间尺寸对应关系需调整 |
| VAE 解码 | ❌ | 依赖 UNet forward 完成后测试 |
| SD15 管线 | ❌ | 使用相同的 `input.ss`，但 SD15 函数尚未独立测试 |
| ELF 打包 | ❌ | `launcher.c` 需要更新为 `read`/`eval` 启动方式 |

### 关键修复记录

| 日期 | 修复 | 文件 |
|------|------|------|
| 2026-07-04 | safetensors parser `skip_ws` 顺序 | `libtorch_std_helper.cpp` |
| 2026-07-04 | safetensors parser 数组内逗号处理 | `libtorch_std_helper.cpp` |
| 2026-07-04 | `if (n < 3)` debug 残留移除 | `libtorch_std_helper.cpp` |
| 2026-07-04 | `STATE_TOKENIZER` 常量冲突（SDXL 覆盖 SD15） | `build_out/input.ss` |
| 2026-07-04 | CLIP token → CUDA 设备转移 | `libtorch_std_helper.cpp` |
| 2026-07-04 | `static_get_sigmas`/`static_get_sampler` 字典 `get` → `hashtable-ref` | `build_out/input.ss` |
| 2026-07-04 | `static_sd_generate` 条件分支 SDXL/SD15 | `build_out/input.ss` |
| 2026-07-04 | SDXL/SD15 wrapped_fn 移除多余 `self` | `build_out/merged.ss` |
| 2026-07-04 | `SDXLPipeline_txt2img` 提取 `pooled_emb` | `build_out/merged.ss` |
| 2026-07-04 | `sdxl_dual_clip` 提取 CLIP-L pooled | `libtorch_std_helper.cpp` |

### UNet 前向命名对齐（进行中）

`libtorch_std_helper.cpp` 中的 `torch_std_sdxl_unet_forward` 最初按简化的命名惯例编写，与标准 SDXL GGUF 模型的 DDPM/LDM 命名不匹配。两种命名都是标准 SDXL，只是内部结构命名不同：

| C++ 原命名 | 模型实际 DDPM 命名 | 状态 |
|------------|-------------------|------|
| `conv_in` | `input_blocks.0.0` | ✅ 已改 |
| `time_embed.0`/`.2` | `time_embed.0`/`.2` | ✅ 匹配 |
| `add_embed.0`/`.2` | `label_emb.0.0`/`.0.2` | ✅ 已改 |
| `down_blocks.*`（分组） | `input_blocks.*`（平铺，每块一个 resblock） | ✅ 已改 |
| `mid_block.*` | `middle_block.*` | ✅ 已改 |
| `conv_norm_out` | `out.0` | ✅ 已改 |
| `conv_out` | `out.2` | ✅ 已改 |
| `up_blocks.*` | `output_blocks.*` | ⏳ 进行中 |
| resblock `norm1`/`conv1`/`time_emb_proj` 等 | `in_layers`/`emb_layers`/`out_layers`/`skip_connection` | ✅ 已加 `sdxl_resblock_ld` |

## 测试方式

当前测试使用 `read`/`eval` 循环加载 `merged.ss`（绕过 `compile-file` bug）：

```bash
cd /tmp
export MODEL=/data/models/image/sd_xl_base_1.0.safetensors
export CLIP_L=/tmp/sd_jit/clip_l.pt
export CLIP_G=/tmp/sd_jit/clip_g.pt
export VAE=/tmp/sd_jit/vae_decode.pt
export STEPS=2 CFG=1.0 SEED=42 PROMPT="a cute cat"

/opt/ChezScheme/ta6le/bin/ta6le/scheme \
  -e '(load-shared-object "libstatic_prelude.so")' \
  --script /tmp/test_script.scm
```

核心测试文件：
- `build_out/input.ss` — 翻译后的 Scheme 代码（已手动补丁）
- `build_out/merged.ss` — prelude + stdlib + input.ss 拼接
- `build_out/runtime.so` — 编译产物
- `cpp_runtime/libtorch_std_helper.so` — C++ 运行时

## 项目文件结构

```
/opt/static_comfyui/
├── compiler/
│   ├── translate.py          ← StaticPy → Scheme 翻译器（883 行）
│   ├── prelude.scm           ← 值类型运行时（420 行）
│   └── stdlib.scm            ← FFI 声明 + 辅助函数
├── cpp_runtime/
│   └── libtorch_std_helper.cpp ← C++ 运行时（~5200 行）
├── build_out/
│   ├── input.ss              ← 翻译后的 Scheme 管线（已手动补丁）
│   ├── merged.ss             ← prelude + stdlib + input.ss
│   ├── runtime.so            ← Chez AOT 编译产物
│   ├── deploy/
│   │   ├── sd_generate.elf   ← 最终 ELF
│   │   └── lib/              ← 运行时 .so
│   ├── embed/                ← 嵌入的 boot 文件
│   └── launcher/             ← C launcher 源码
├── docs/
│   └── design.md             ← 本文档
├── comfyui_ref/              ← ComfyUI 参考源码
├── AGENTS.md                 ← AI 助手指南
└── STATUS.md                 ← 状态追踪
```


