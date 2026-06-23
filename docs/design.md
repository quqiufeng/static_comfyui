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
libtorch_std_helper  (extern "C" API)    ←  张量基元 + 模型前向
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
│  libtorch_std_helper  (C++ extern "C" API)       │
│  ┌──────────────┐  ┌────────────────────────┐    │
│  │  张量基元层    │  │    模型前向层           │    │
│  │ conv2d        │  │ torch_std_sd_unet_     │    │
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
│           ELF 打包器 (deliver.sh)                  │
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

### Layer 2: libtorch_std_helper — C++ 运行时层

**该层是项目的核心资产**（位于 `/opt/ReScheme/libtorch_std_helper.h/.cpp`），提供涵盖张量基元和完整模型前向的 364 行 extern "C" API。

| 类别 | 函数 | 行数 |
|------|------|------|
| 张量创建 | `torch_std_tensor_from_blob`, `zeros`, `ones`, `empty`, `full`, `randn`, `randint`, `clone`, `detach`, `to_dtype` | ~30 |
| 数学 | `add`, `sub`, `mul`, `div`, `pow`, `exp`, `log`, `sqrt`, `neg`, `abs` | ~20 |
| 激活 | `relu`, `leaky_relu`, `sigmoid`, `tanh`, `softmax`, `log_softmax` | ~12 |
| 归约 | `sum`, `sum_dim`, `mean`, `mean_dim`, `max`, `max_dim`, `min`, `min_dim` | ~16 |
| 形状 | `reshape`, `transpose`, `permute`, `squeeze`, `unsqueeze`, `cat`, `stack` | ~14 |
| 线性代数 | `matmul`, `linear`, `conv1d`, `conv2d`, `max_pool2d`, `avg_pool2d`, `batch_norm1d/2d` | ~14 |
| 损失/梯度 | `mse_loss`, `cross_entropy_loss`, `backward`, `grad`, `sgd_create`, `adam_create`, `adamw_create` | ~20 |
| 工具 | `narrow`, `slice`, `masked_select`, `where`, `eq/gt/lt/ge/le`, `clamp`, `clip_grad_norm` | ~16 |
| 设备 | `cuda_is_available`, `to_cuda`, `to_cpu`, `is_cuda` | ~8 |
| 序列化 | `save_state_dict`, `jit_load`, `jit_forward` | ~8 |
| **SD UNet** | `torch_std_sd_unet_forward` (SD1.5, 418 weights) | 1 |
| **SDXL UNet** | `torch_std_sdxl_unet_forward` (dict-based, 6 dims) | 1 |
| **FLUX** | `torch_std_flux_forward` (MMDiT, guidance) | 1 |
| **CLIP** | `torch_std_clip_text_forward`, `torch_std_clip_tokenizer_create/encode/free` | 3 |
| **T5** | `torch_std_t5_tokenizer_create/encode/free` | 3 |
| **Samplers** | `torch_std_sample_ddim/euler/euler_ancestral/dpmpp_2m`, `torch_std_sampler_sigmas`, `torch_std_fm_sigmas/step` | ~8 |
| **ControlNet** | `torch_std_controlnet_forward`, `torch_std_controlnet_apply` | 2 |
| **LoRA** | `torch_std_lora_apply`, `torch_std_lora_merge_into` | 2 |
| **VAE tiling** | `torch_std_vae_encode_tiled`, `torch_std_vae_decode_tiled` | 2 |
| **safetensors** | `torch_std_safetensors_load/count/name/tensor/free/get_tensor_by_name` | ~6 |
| **GGUF** | `torch_std_gguf_load/count/name/load_tensor/free` | ~5 |
| **Image I/O** | `torch_std_load_image/save_image`, `resize/crop/composite/color_convert`, `load/save_png` | ~8 |
| 其他 | `manual_seed`, `numel`, `ndim`, `shape`, `to_double/float/int64_array`, `copy_probs` | ~10 |

**设计原则**：
1. **所有函数 extern "C"**，直接可被 Chez Scheme 的 `foreign-procedure` 调用
2. **模型前向**在 C++ 层实现（而非 StaticPy），因为 PyTorch C++ API 直接调用 cuDNN/cuBLAS，无跨语言开销
3. **StaticPy 负责管线编排**——采样循环、多模型串联、参数控制、CLI 交互
4. **重量级模型（UNet/VAE/CLIP）的前向在 C++**，轻量级逻辑（image post-processing、格式转换）在 StaticPy

### Layer 3: 编译器 (compiler/)

| 文件 | 职责 |
|------|------|
| `translate.py` | StaticPy → Scheme S-表达式；解析 `extern fn` 生成 `foreign-procedure` 绑定 |
| `prelude.scm` | 值类型运行时：C float/int/ptr 数组、GC guardian、文件 I/O、JSON、HTTP、libm 数学函数 |
| `stdlib.scm` | libtorch_std_helper 的 FFI 声明 + 简单 CPU fallback 函数 |

**编译流程**：

```
StaticPy (.static.py)
    │ translate.py 解析 extern fn 声明、转换 AST
    ▼
Scheme S-表达式 + foreign-procedure 绑定
    │ cat prelude.scm + stdlib.scm + user.ss → merged.ss
    ▼
merged.ss
    │ Chez Scheme compile-file
    ▼
merged.so  (原生机器码)
    │ 链接 libtorch_std_helper.so (PyTorch C++)
    ▼
单文件 ELF  (scheme + boot + .so 全部嵌入)
```

## 编译管线 (build.sh)

```
Input:  user_module.static.py
        │
        ▼
[1] python3 compiler/translate.py  →  user_module.ss
        │
        ▼
[2] cat compiler/prelude.scm compiler/stdlib.scm user_module.ss > merged.ss
        │
        ▼
[3] scheme --compile-file merged.ss → merged.so
        │
        ▼
[4] arm-none-eabi-objcopy (or objcopy) → 嵌入 .so
        │
        ▼
Output: build_out/user_module.so  (~400KB-1.5MB)
```

## 部署包 (deliver.sh)

`deliver.sh` 产出**完整的自包含部署包**，部署机只需 NVIDIA 驱动 + libc，**无需安装 PyTorch，无需 Python**。

### 部署包结构

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
    ├── libnvrtc.so.12           ← CUDA JIT 编译
    └── ...                      ← 其他 DT_NEEDED 依赖自动追踪
```

### 构建流程

```
sd_runtime/*.static.py   (多个 StaticPy 源文件)
        │
        ▼
python3 compiler/translate.py  →  sd_runtime.ss
        │
        ▼
cat prelude.scm stdlib.scm sd_runtime.ss > merged.ss
        │
        ▼
scheme --compile-file merged.ss → build_out/runtime.so
        │
        ▼
g++ -shared -fPIC -o build_out/libs/libtorch_std_helper.so \
    /opt/ReScheme/libtorch_std_helper.cpp \
    -I/data/venv/lib/python3.12/site-packages/torch/include \
    -I/data/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include \
    -L/data/venv/lib/python3.12/site-packages/torch/lib \
    -ltorch -lc10 -ltorch_cpu -ltorch_cuda -lc10_cuda
        │
        ▼
C launcher:
  │ 嵌入 scheme 二进制
  │ 嵌入 petite.boot + scheme.boot
  │ 嵌入 runtime.so (StaticPy 编译产物)
  │
  ▼
gcc -o sd_generate.elf -Wl,-rpath,'$ORIGIN/lib'
  │
  ▼
收集 PyTorch/CUDA .so → lib/
  │
  ▼
build_out/deploy/
  ├── sd_generate.elf   (≈3MB, 仅嵌入 scheme+boot+runtime)
  └── lib/              (≈2-3GB, 全部 PyTorch+CUDA .so)
```

### ELF 运行时行为

1. ELF 启动后通过 `/proc/self/exe` 获取自身路径
2. 提取嵌入的 `scheme` 二进制、`petite.boot`、`scheme.boot`、`runtime.so` 到 `/stock/.tmp/`
3. 设置 `LD_LIBRARY_PATH` 指向：
   - `{self_dir}/lib/`（与 ELF 同目录的部署库）
   - 标准 CUDA/系统路径
4. `execv(scheme, ["--quiet", "--boot", "petite.boot", "runtime.so"])`
5. Chez Scheme 加载 `runtime.so`，执行 `(static_main)`
6. `runtime.so` 内部的 `foreign-procedure` 从 `LD_LIBRARY_PATH` 找到 `libtorch_std_helper.so` → `libtorch.so` → GPU

## 模型权重管理

### 导出流程

```
safetensors 文件 (sd_xl_base_1.0.safetensors, 6.5GB)
    │
    ▼
export_weights.py → 有两种模式：
    │
    ├── 模式 A: 直接调用 torch_std_safetensors_load()
    │   StaticPy 通过 extern fn 直接读取 safetensors
    │   运行时加载，无需预转换
    │
    └── 模式 B: 预导出为 weights.bin + index.json
        float32 平面数组 + 偏移索引
        适用于嵌入 ELF 的静态权重
```

### 加载流程 (StaticPy)

```python
# 模式 A: safetensors 直接加载
def load_model(path: str) -> ptr:
    sd: ptr = torch_std_safetensors_load(path)
    return sd  # dict of tensor ptrs

# 模式 B: 预导出权重
def load_weights(path: str) -> ptr:
    data: list[float] = file_read_floats(path + "/weights.bin", total_n)
    index: ptr = parse_json(file_read_all(path + "/index.json"))
    return (data, index)
```

## 与 ComfyUI 源码的关系

```
comfyui_ref/comfy/
├── clip_model.py            ← CLIP text encoder 参考
├── sd1_clip.py
├── ldm/modules/
│   ├── diffusionmodules/
│   │   ├── openaimodel.py   ← UNet 架构参考
│   │   └── model.py         ← VAE 参考
│   └── attention.py         ← Attention 实现参考
├── k_diffusion/
│   └── sampling.py          ← 采样器参考
├── controlnet.py            ← ControlNet 参考
└── lora.py                  ← LoRA 参考
```

**对齐规则**：
1. C++ 模型前向代码（libtorch_std_helper）的数值必须与 ComfyUI 源码一致（`max_diff < 1e-3`）
2. StaticPy 管线代码调用 C++ 前向函数时，参数传递顺序与 ComfyUI 源码一致
3. 新模型实现前，必须在 `comfyui_ref/comfy/` 找到对应源码

## 与 ReScheme/libtorch_std_helper 的关系

```
/opt/ReScheme/
├── libtorch_std_helper.h     ← 364 行 extern "C" API 声明（核心接口）
├── libtorch_std_helper.cpp   ← C++ 实现（PyTorch C++ API 包装）
└── ...                        ← ReScheme 自有代码

/opt/static_comfyui/           ← 本项目的入口
├── compiler/                  ← StaticPy 编译器
├── sd_runtime/                ← StaticPy 管线代码
├── build.sh                   ← 编译管线
├── deliver.sh                 ← ELF 打包
└── comfyui_ref/               ← ComfyUI 参考源码
```

**关键路径**：
- `libtorch_std_helper` 在 ReScheme 仓库中维护，但它的 364 行 extern "C" API 是**跨项目共享资产**
- `static_comfyui` **不复制** libtorch_std_helper 的 C++ 代码，编译时直接引用 `/opt/ReScheme/libtorch_std_helper.cpp`
- 对 libtorch_std_helper 的修改需要保证 ReScheme 和 static_comfyui 都通过编译

## 添加新模型的工作流

1. **在 `comfyui_ref/comfy/` 中找到源码**
2. **在 `libtorch_std_helper.h/.cpp` 中添加模型前向函数**（extern "C"）
3. **在 `compiler/stdlib.scm` 中添加对应 FFI 声明**
4. **在 `sd_runtime/` 中写 StaticPy 管线代码**调用新函数
5. **单模块测试**：编译为 `.so`，用 PyTorch reference 验证数值
6. **端到端**：运行 `deliver.sh` → 单文件 ELF → 验证输出

## 性能特征

| 操作 | 路径 | 性能 |
|------|------|------|
| conv2d / matmul / attention | PyTorch C++ → cuDNN/cuBLAS | GPU 峰值 |
| group_norm / layer_norm | PyTorch C++ | GPU 峰值 |
| CLIP tokenizer (BPE) | C++ 直接实现 | 微秒级 |
| 采样循环控制流 | Chez Scheme AOT | 接近 C |
| 管线编排 | StaticPy → Scheme | 极低开销 |

**关键洞察**：所有计算密集型操作（conv2d, matmul, attention, norm）都运行在 PyTorch C++/GPU 上。StaticPy 只负责高层管线编排——控制流、参数组合、模型间数据传递。性能与原生 PyTorch 基本等价。

## 部署方式

### 部署要求（最低）

| 依赖 | 要求 |
|------|------|
| OS | Linux x86_64（glibc） |
| GPU | NVIDIA GPU + 驱动（≥CUDA 12.0 兼容） |
| 磁盘 | ~3GB（ELF + PyTorch/CUDA lib/ 目录） |
| Python | **不需要** |
| PyTorch | **不需要安装**（lib/ 目录自带） |

### 部署步骤

```bash
# 在构建机上
./build.sh sd_runtime/*.static.py   # 编译 StaticPy → runtime.so
./deliver.sh                         # 打包 → build_out/deploy/

# 把 deploy/ 整个目录复制到目标机器
scp -r build_out/deploy/ target:/app/

# 在目标机器上直接运行
/app/deploy/sd_generate.elf
```

### 工作原理

1. `sd_generate.elf` 是 C launcher，内嵌了 Chez Scheme 二进制 + boot 文件 + StaticPy 编译的 `runtime.so`
2. 同目录的 `lib/` 包含全部 PyTorch/CUDA 运行时 .so（`libtorch.so`, `libc10.so`, `libcudart.so.12`, `libcublas.so.12`, `libcudnn.so.9` 等）
3. ELF 的 RPATH 设置为 `$ORIGIN/lib`，运行时自动找到这些 .so
4. 目标机器只需 NVIDIA 驱动（供 CUDA 运行时调用），无需安装 Python 或 PyTorch

---

## 为什么这比 pip/conda 部署好太多

### 传统 Python 方案

```bash
# 部署一台新机器，第一步就疯了：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# ↑ 3GB 下载，30 分钟，还得祈祷 CUDA 版本不冲突

pip install diffusers transformers accelerate pillow
# ↑ 又 10 分钟，中间可能 glibc 版本不够新编译失败

# 建个虚拟环境，装错了还得重来
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# ↑ 光解决 torch+cuda 版本组合就能折磨半天

# 更别提：
#   - CUDA 驱动版本 vs CUDA toolkit 版本 mismatch
#   - glibc 版本不够 → "undefined symbol: __cxa_thread_atexit_impl"
#   - 两个项目需要不同 PyTorch 版本 → 两个 venv
#   - pip 跑一半网络断了 → 重来
#   - sudo pip install 污染系统 Python
```

### 本方案

```bash
# 部署（唯一命令）：
tar xzf deploy.tar.gz
./deploy/sd_generate.elf
```

**有显卡驱动就能跑。没有 Python，没有 pip，没有 venv，没有 conda，没有 glibc 版本战争，没有 CUDA toolkit 安装，没有依赖地狱。**

### 对比表

| 维度 | Python (pip/conda) | 本方案 (StaticPy → ELF) |
|------|-------------------|-------------------------|
| 部署步骤 | 装 Python → 建 venv → pip install → 试错 → 修冲突 → 终于能跑 | 解压 → 跑 |
| 环境隔离 | venv/conda（额外学习成本） | 不需要，OS 级隔离 |
| CUDA 版本 | 必须匹配 PyTorch 版本 | 构建时固定，部署时只要驱动兼容 |
| glibc 版本 | 容易冲突导致编译/运行失败 | 构建时固定，同 OS 架构直接跑 |
| 多项目共存 | 每个项目一个 venv（几 GB 重复） | 每个项目一个 deploy/ 目录 |
| 启动时间 | `import torch` 就要 ~1 秒 | Chez Scheme AOT 原生码，毫秒级 |
| 跨机器部署 | 每台都要装一模一样的 Python 环境 | 解压即用 |
| 网络依赖 | 部署时需要装 PyTorch（3GB 下载） | 不需要（部署包自包含） |
| 升级代价 | `pip install --upgrade` 可能打破兼容性 | 构建机上重新 `./deliver.sh` |
| 依赖组成 | pip 自动拉取，但可能拉错版本 | DT_NEEDED 自动解析，固定版本 |
| "它在我机器上能跑" | 经典问题（环境不一样） | **永远不会出现**——环境就在包里 |

### 一句话

> **pip 解决的是"怎么装"的问题，但引出了"装对了没有"的问题。**
> StaticPy 编译到 ELF 的方案根本不走这条路——编译期把所有依赖都固定好，部署只是文件拷贝。
