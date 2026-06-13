# ComfyUI → StaticPy：重写 Stable Diffusion 推理引擎

## 一、重写动机

### 1.1 ComfyUI 的问题

```
部署:
  pip install torch torchvision xformers diffusers transformers ...
  装 ComfyUI + 自定义节点 + ControlNet 扩展
  Python main.py
  → 2-3GB 依赖，版本冲突家常便饭

启动:
  import torch → 加载 CUDA 库 → 注册算子 → 解析节点图
  → 冷启动 5-15 秒

运行时:
  Python GIL 限制多线程
  PyTorch 每次调用都有 Python → C++ 上下文切换
```

### 1.2 重写后

```
部署:
  scp sd_generate.elf root@server:/opt/
  /opt/sd_generate --prompt "cat" --output out.png
  → 5.6MB 二进制（方案）+ 权重文件（模型 GGUF/Safetensors）
  → 零 Python、零 pip、零版本冲突

启动:
  exec scheme → 加载 .so → dlopen cuBLAS
  → 亚秒级

运行时:
  原生机器码，无解释器开销
  dgemm_row_auto 直接调 cuBLAS 或 CPU 回退
```

---

## 二、Stable Diffusion 推理管线的数学拆解

### 2.1 整体流程

```
Text Prompt
    │
    ▼
CLIP / T5 / Qwen2 文本编码器
    │  ┌──────────────────────────────┐
    │  │  Token Embedding             │
    │  │  + Transformer × N 层        │
    │  │    = DGEMM(Q,K,V) + softmax  │
    │  │    + LayerNorm + MLP         │
    │  └──────────────────────────────┘
    ▼
Text Embeddings (77×768 或 77×4096)
    │
Noise Latent (64×64×4 随机噪声)
    │
    ▼
UNet × N 步去噪
    │  ┌──────────────────────────────┐
    │  │  DownBlock × N:              │
    │  │    Conv2D → GroupNorm → SiLU │
    │  │    → CrossAttention → 残差加 │
    │  │    → DownSample              │
    │  │                              │
    │  │  MidBlock:                   │
    │  │    Conv2D → Attention × N    │
    │  │                              │
    │  │  UpBlock × N:                │
    │  │    UpSample → Conv2D → GN    │
    │  │    → CrossAttention → 残差加 │
    │  └──────────────────────────────┘
    │         每个算子 = DGEMM + 逐元素
    ▼
Denoised Latent
    │
    ▼
VAE Decoder
    │  Conv2D × N → GroupNorm → SiLU → UpSample
    │  → 最终 Conv2D → 像素值
    ▼
Image

全部可用 DGEMM + 逐元素运算表达。
```

### 2.2 算子 → 数学运算 映射表

| SD 算子 | 数学本质 | StaticPy 实现 | 行数 |
|---------|---------|--------------|------|
| **Conv2D** | `im2col + DGEMM` | 展成矩阵 → dgemm_row_auto | 20 |
| **Conv2D 转置** | `col2im + DGEMM^T` | DGEMM → 折叠回张量 | 20 |
| **CrossAttention** | `DGEMM(Q,K^T) → softmax → DGEMM(attn,V)` | 三个 DGEMM + 逐元素 softmax | 30 |
| **SelfAttention** | 同上，Q=K=V 来自同源 | 同上 | 30 |
| **GroupNorm** | `mean → var → (x-mean)/sqrt(var+eps) * γ + β` | 逐元素：求和 → 平方和 → 归一化 | 15 |
| **LayerNorm** | 同上，对特征维做 | 逐元素 | 10 |
| **SiLU (Swish)** | `x * sigmoid(x)` | 逐元素计算 | 5 |
| **GELU** | `0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))` | 逐元素分段多项式 | 8 |
| **ReLU** | `max(0, x)` | 逐元素比较 | 3 |
| **Sigmoid** | `1/(1+exp(-x))` | 已有 `sigm()` 函数 | 1 |
| **Tanh** | `(exp(2x)-1)/(exp(2x)+1)` | 已有 `tanh_f()` 函数 | 1 |
| **Softmax** | `exp(x_i) / Σexp(x_j)` | 逐元素：求 exp → 求和 → 归一化 | 10 |
| **残差连接** | `out = x + skip(x)` | 逐元素加 | 3 |
| **下采样** | Conv2D stride=2 | Conv2D + 跳步读取 | 20 |
| **上采样** | 插值 + Conv2D | 最近邻/双线性 + Conv2D | 25 |
| **Additive Noise** | `noise = noise * sqrt(α) + ε * sqrt(1-α)` | 逐元素 | 3 |
| **DDIM Step** | `x_{t-1} = sqrt(α_{t-1}) * pred_x0 + sqrt(1-α_{t-1}) * ε_t` | 浮点运算 + 逐元素 | 15 |
| **VAE Encoder** | Conv2D + GN + SiLU + DownSample | 重复 Conv2D + 逐元素 | 50 |
| **VAE Decoder** | Conv2D + GN + SiLU + UpSample | 同上，方向相反 | 50 |
| **CLIP Text Encoder** | Token Embed + Transformer × 12 层 | DGEMM + Attention + MLP | 80 |

**合计：所有算子 ≈ 400 行 StaticPy**

### 2.3 核心：Conv2D 的 im2col + DGEMM

Conv2D 是所有视觉模型的计算瓶颈（UNet 中占 60%+ 算力）。

数学本质：

```
输入:  x[N, C, H, W]
卷积核: w[K, C, 3, 3]  (3×3 为例)
输出:  y[N, K, H', W']

im2col:  将输入展成矩阵 col[N*H'*W', C*3*3]
DGEMM:   y_col[N*H'*W', K] = col[N*H'*W', C*3*3] @ w[C*3*3, K]
col2im:  把 y_col 折叠回 y 的形状

等价于:  y = dgemm_row_auto(N*H'*W', K, C*3*3, 1.0, col, w, 0.0, y_col)
```

StaticPy 实现：

```python
def conv2d(x, w, n, c_in, c_out, h, w_in, k_size, stride, padding):
    """Conv2D: x[N,C,H,W] → y[N,K,H',W']"""
    h_out = (h + 2*padding - k_size) // stride + 1
    w_out = (w_in + 2*padding - k_size) // stride + 1
    col: list[float] = make_float_array(n * h_out * w_out * c_in * k_size * k_size)
    # im2col: 把输入展开为矩阵（嵌套循环遍历所有位置）
    i: int = 0; j: int = 0; ...
    # DGEMM
    y: list[float] = make_float_array(n * h_out * w_out * c_out)
    dgemm_row_auto(n*h_out*w_out, c_out, c_in*k_size*k_size, 1.0, col, w, 0.0, y)
    return y
```

关键直觉：**整个 UNet 就是几十个 DGEMM 调用 + 几百个逐元素运算。没有别的东西。**

---

## 三、架构设计

### 3.1 整体架构

```
┌─ 输入层 ───────────────────────────────────┐
│                                             │
│  sd_generate.elf（单文件二进制）              │
│    --prompt "..."                           │
│    --cfg 7.5                                │
│    --steps 20                               │
│    --model /path/to/model.gguf              │
│    --output out.png                         │
│                                             │
│  嵌入: scheme + boot + sd_runtime.so         │
│  运行时 dlopen: cuBLAS / cuDNN / OpenBLAS   │
│                                             │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│  sd_runtime.static.py                        │
│  (StaticPy 编译产物，~800 行)               │
│                                             │
│  主要模块:                                  │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐  │
│  │  CLIP      │ │ UNet     │ │ VAE      │  │
│  │  文本编码   │ │  去噪     │ │  编解码  │  │
│  └────────────┘ └──────────┘ └──────────┘  │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐  │
│  │  Sampler   │ │ Control  │ │ LoRA     │  │
│  │  采样器    │ │  Control │ │  权重注入│  │
│  └────────────┘ └──────────┘ └──────────┘  │
│                                             │
└─────────────────────────────────────────────┘
```

### 3.2 模块划分

#### 模块 1：张量运算基础设施

```python
# 已有的（直接复用 CNN-GRU 的代码）:
dgemm_row_auto(m, n, k, alpha, A, B, beta, C)   # GPU/CPU 自动切换
arr_add(a, b, n)          # 逐元素加
arr_mul(a, b, r, n)       # 逐元素乘
arr_sigmoid(a, r, n)      # sigmoid
arr_tanh(a, r, n)         # tanh
arr_relu(a, r, n)         # ReLU
arr_copy(dst, src, n)     # 数组复制
arr_fill(a, v, n)         # 填充

# 需要新增的:
arr_exp(a, r, n)          # 逐元素 exp（softmax 需要）
arr_pow(a, p, r, n)       # 逐元素幂（GELU 需要）
arr_sqrt(a, r, n)         # 逐元素平方根（GroupNorm 需要）
arr_rsqrt(a, r, n)        # 逐元素 1/sqrt（GroupNorm 需要）
arr_maximum(a, b, r, n)   # 逐元素 max（ReLU/clip）
arr_clip(a, lo, hi, n)    # 截断
softmax(a, dim, shape)    # 沿维度做 softmax
layer_norm(x, gamma, beta, n)          # LayerNorm
group_norm(x, gamma, beta, n_groups)   # GroupNorm
```

#### 模块 2：Conv2D 系列

```python
def im2col(x, n, c, h, w, k_size, stride, pad) -> list[float]
def conv2d(x, w, ...) -> list[float]
def conv2d_transposed(x, w, ...) -> list[float]
def downsample(x, scale) -> list[float]
def upsample_nearest(x, scale) -> list[float]
```

#### 模块 3：Attention 系列

```python
def cross_attention(q, k, v, n_head) -> list[float]
def self_attention(x, n_head) -> list[float]
def optimized_attention(q, k, v, n_head) -> list[float]  # Flash Attention 模拟
```

#### 模块 4：UNet 块

```python
def resnet_block(x, w_conv1, w_conv2, gamma_gn1, beta_gn1, ...) -> list[float]
def down_block(x, resnets, attentions, downsample) -> list[float]
def up_block(x, skip, resnets, attentions, upsample) -> list[float]
def mid_block(x, resnet, attentions) -> list[float]
def unet_forward(latent, t, context, ...) -> list[float]
```

#### 模块 5：采样器

```python
def ddim_step(x, pred_noise, t, alpha_bar, alpha_bar_prev) -> list[float]
def dpmpp_step(x, pred_noise, t, ...) -> list[float]
def sample(unet_fn, latent, steps, cfg_scale, ...) -> list[float]
```

#### 模块 6：文本编码

```python
def clip_tokenize(text: str) -> list[int]
def clip_encode(tokens, weight_matrix, ...) -> list[float]
def t5_encode(text, ...) -> list[float]  # SD3/Flux 需要
```

#### 模块 7：VAE

```python
def vae_encode(image) -> list[float]    # 图 → latent
def vae_decode(latent) -> list[float]   # latent → 图
def vae_tiling_decode(latent, tile_size, overlap)  # 大图分块解码
```

### 3.3 权重格式

```
ComfyUI 用的 Safetensors / GGUF

方案 1: 用 dlopen 加载 libggml 来读 GGUF
  extern fn gguf_init_from_file(path: str) -> ptr from "ggml"
  extern fn gguf_get_tensor(ctx: ptr, name: str) -> ptr from "ggml"
  → 直接加载现存模型格式，不需要转换

方案 2: 用 export_weights.py 导出为 .bin
  → 跟股票预测一样的流程
  → PyTorch 读取 Safetensors → 逐一写为 flat .bin
  → StaticPy load_bin 直接加载
  → 简单可靠，但需要转换脚本

方案 3: 自解析 Safetensors
  Safetensors = JSON header + 二进制数据
  纯 C 解析 JSON header → fseek → fread 数据
  extern fn fopen/fread/fseek 即可
```

推荐方案 3——静态解析 Safetensors 头部（JSON）后直接 `fread` 权重数据，不需任何第三方库。

---

## 四、与 my-img (C++) 的对比

| 维度 | my-img (C++) | StaticPy 版 |
|------|-------------|-------------|
| 代码量 | ~5000 行 C++ | ~800 行 StaticPy |
| 开发周期 | 247 commits，数月 | 预估 1-2 周 |
| 构建系统 | CMake + 第三方库编译 | static_build.sh 一键 |
| 编译时间 | 5-10 分钟 | ~5 秒 |
| 部署产物 | 动态链接二进制 + libtorch | 单文件 ELF（5.6MB） |
| GPU 加速 | 手动 cudaMalloc/cudaFree | dgemm_row_auto 自动 |
| CPU 回退 | sd.cpp GGML | 纯 C 矩阵乘 |
| 内存管理 | 手动 RAII | float array → GC |
| 适配层 | sdcpp_adapter 封装 C API | extern fn 直接贴 C 函数 |
| 内核升级 | 改适配层 | 改 extern 声明 |
| ControlNet | sd.cpp 内置支持 | 同路线：额外 Conv2D |
| LoRA | sd.cpp 内置 | 权重合并：W' = W + α * ΔW |
| Flash Attention | sd.cpp 内置 | 手写优化的 attention |

**结论：StaticPy 版本以 Python 级别的开发效率，产出 C++ 级别的交付产物。**

---

## 五、实现路线

### Phase 1：张量基础设施（1 天）

```
arr_exp, arr_pow, arr_sqrt, arr_rsqrt
softmax, layer_norm, group_norm
im2col, conv2d, conv2d_transposed
→ 已有 dgemm_row_auto 和大部分 arr_*，新增约 50 行
```

### Phase 2：UNet 核心（3 天）

```
resnet_block → ResBlock × N
down_block   → DownBlock × N  
up_block     → UpBlock × N
mid_block    → MidBlock
cross_attention → CrossAttention

→ 每个 block 约 30-50 行 StaticPy
→ 验证：用随机权重跑 forward，输出形状与 PyTorch 一致
```

### Phase 3：采样器（1 天）

```
DDIM / DPM++ / Euler
CFG 缩放: pred = unconditional + cfg * (conditional - unconditional)
SAG (Self-Attention Guidance)
FreeU

→ 纯浮点运算 + 逐元素，约 80 行
```

### Phase 4：文本编码 + VAE（2 天）

```
CLIP tokenizer（BPE 编码，纯查表）
CLIP transformer（DGEMM + Attention）
VAE encoder / decoder（Conv2D × 15 层）
VAE tiling（大图分块）

→ 约 200 行
```

### Phase 5：模型加载 + 控制（1 天）

```
Safetensors 解析（JSON header → fread）
LoRA 权重注入（W += alpha * ΔW）
ControlNet 适配（额外 Conv2D）
CLI + JSON workflow 解析

→ 约 150 行
```

### Phase 6：打包测试（0.5 天）

```
sd_runtime.static.py → deliver.sh → sd_generate.elf
验证与 ComfyUI 输出一致性（数值误差 < 1e-5）
```

**总计：约 8 个工作日，800 行 StaticPy。**

---

## 六、与工业界的对比

```
PyTorch + ComfyUI 路线:
  Python 解释器（~10MB）
  + PyTorch 框架（~2GB）
  + CUDA Toolkit（~3GB）
  + cuDNN（~1GB）
  + diffusers / transformers / xformers（~500MB）
  + 自定义节点（~200MB）
  = 约 7GB 部署体积
  = 启动 5-15 秒
  = pip install 30 分钟

StaticPy 路线:
  Chez Scheme 运行时（1.1MB）
  + boot 文件（3.3MB）
  + sd_runtime.so（~50KB）
  + 权重（可选，单独存放）
  = 5.6MB 部署体积（不含权重）
  = 启动 < 1 秒
  = scp → chmod +x → 跑
```

**核心洞察**：7GB → 5.6MB 的差距，差的不是功能——差的是 Python。

---

## 六、staticpyTorch：运行时封装 libtorch

### 6.1 为什么需要 staticpyTorch

最初用纯 StaticPy 手写所有算子：

```python
# 手写 conv2d：im2col + dgemm_row_auto + bias_add
# 手写 group_norm：mean → var → normalize → affine
# 手写 attention：Q@K^T → softmax → @V
```

每个算子 20-50 行 StaticPy，整个 UNet forward 需要 ~4000 行。
难写、难调、难维护。

### 6.2 解决方案

PyTorch 本身就是 C++。`libtorch.so` 就在 `/data/venv/lib/.../torch/lib/` 下。
直接封装它的 C API，StaticPy 通过 `extern fn` 调用：

```
StaticPy 用户代码
    │  torch.conv2d(x, w, b, stride, padding)  ← Python 语法
    ▼
torch.static.py（高级 API 层）
    │  def torch_conv2d(...): return st_conv2d(...)
    ▼
torch_ops.static.py（extern fn 声明层）
    │  extern fn st_conv2d(...) -> ptr from "staticpy_torch"
    ▼
runtime/staticpy_torch.cpp（C 包装层）
    │  extern "C" void* st_conv2d(...) {
    │      return new torch::Tensor(torch::conv2d(inp, w, b, ...));
    │  }
    ▼
libtorch.so（PyTorch 原生 C++ 库）
    │  at::conv2d(input, weight, bias, stride, padding, dilation, groups)
    ▼
cuDNN / cuBLAS（GPU 加速）
```

**四层架构，每层只做一件事：**

| 层 | 文件 | 职责 | 用户是否接触 |
|----|------|------|------------|
| 用户 API | `sd_runtime/torch.static.py` | `torch.conv2d(...)` 风格接口 | ✅ 是 |
| FFI 声明 | `sd_runtime/torch_ops.static.py` | `extern fn` 到 C 函数 | ❌ 否（内部） |
| C 包装 | `runtime/staticpy_torch.cpp` | C++ API → C 函数 | ❌ 否（内部） |
| 原生库 | `libtorch.so` | 真正的计算（cuDNN/cuBLAS） | ❌ 否（运行时加载） |

### 6.3 运行时加载原理

所有 ML 库都在运行时通过 `dlopen` 按需加载，**编译时不需要任何 ML 头文件或库**：

```
编译时（build.sh）:
  g++ staticpy_torch.cpp → libstaticpy_torch.so
    ↑ 只需要 libtorch 头文件（编译时检测类型）
    ↑ 不链接 libtorch（运行时才 dlopen）

运行时（scheme --quiet app.so）:
  load-shared-object "libc10.so"          ← dlopen C10 基础库
  load-shared-object "libtorch_cpu.so"    ← dlopen PyTorch CPU
  load-shared-object "libtorch_cuda.so"   ← dlopen PyTorch CUDA
  load-shared-object "libtorch.so"        ← dlopen PyTorch 主库
  load-shared-object "libstaticpy_torch.so" ← dlopen 我们的包装层
    ↑ 全部在 .so 加载时通过 dlopen 完成
    ↑ 有 GPU 用 libtorch_cuda，没有就跳过
```

关键设计：`staticpy_torch.cpp` 编译时**不链接 libtorch**（`-l` 参数只在链接时提供符号引用，真正加载在运行时）。编译产出仅 95KB。运行时如果机器上装了 PyTorch（libtorch.so 在标准路径），自动用 GPU 加速；没有就报错（可回退到手写算子）。

### 6.4 与传统 PyTorch 对比

```
传统 PyTorch:
  Python 代码
    ↓ CPython 解释执行
    ↓ pybind11 类型转换
    ↓ C++ libtorch API
    ↓ cuDNN / cuBLAS
  → 每层都有开销

staticpyTorch:
  Python 语法代码（StaticPy）
    ↓ 编译为原生机器码
    ↓ extern fn 直接调 C 函数
    ↓ C++ libtorch API
    ↓ cuDNN / cuBLAS
  → 编译后零解释器开销
```

**用户写的还是 Python，但运行时不经过 Python 解释器。** StaticPy 编译器把 Python 语法翻译成 Scheme，`compile-file` 编译为原生机器码。`torch.conv2d(...)` 在最终二进制里就是一次 C 函数调用，没有任何 Python 层的存在。

### 6.5 已封装的算子

| torch API | C 函数 | libtorch 后端 | 行数 |
|-----------|--------|-------------|------|
| `torch_conv2d` | `st_conv2d` | `torch::conv2d` → cuDNN | 10 |
| `torch_group_norm` | `st_group_norm` | `torch::group_norm` → cuDNN | 8 |
| `torch_layer_norm` | `st_layer_norm` | `torch::layer_norm` | 8 |
| `torch_linear` | `st_linear` | `torch::linear` → cuBLAS | 6 |
| `torch_silu` | `st_silu` | `torch::silu` | 4 |
| `torch_relu` | `st_relu` | `torch::relu` | 4 |
| `torch_gelu` | `st_gelu` | `torch::gelu` | 4 |
| `torch_softmax` | `st_softmax` | `torch::softmax` | 4 |
| `torch_mm` | `st_mm` | `a.mm(b)` → cuBLAS | 4 |
| `torch_upsample_nearest` | `st_upsample_nearest` | `torch::upsample_nearest2d` | 5 |
| `torch_add/sub/mul` | `st_add_tensor` | `a + b` | 3 |
| `torch_sum/mean` | `st_sum/mean` | `t.sum()` | 3 |

---

## 七、限制

| 功能 | 能改？ | 说明 |
|------|--------|------|
| txt2img 推理 | ✅ | Conv2D + Attention = DGEMM ✅ |
| img2img | ✅ | 多一步 VAE encode + 加噪 |
| HiRes Fix | ✅ | latent 放大 + 第二轮去噪 |
| ControlNet | ✅ | 额外 Conv2D 注入 |
| LoRA | ✅ | W += alpha * ΔW 权重合并 |
| SAG / FreeU | ✅ | attention 中间结果操作 |
| 节点图编排 | ❌ | JS 前端工作，不是数学问题 |
| 自定义节点 | ❌ | 动态 Python 代码，无法静态编译 |
| 训练/LoRA 训练 | ⚠️ | 前向可，反向要手写 |
| IPAdapter | ⚠️ | CLIP Vision 编码 + Attention |
| 模型下载 | ❌ | 文件管理，跟推理无关 |

---

## 八、文件结构

```
stock/
├── sd_runtime.static.py      ← 主文件：所有 SD 推理代码（~800 行）
├── deliver.sh                 ← 打包脚本（复用现有）
├── export_sd_weights.py      ← PyTorch → .bin 权重导出
├── comfyui.md                 ← 本文档
└── test_sd_generate.sh       ← 验证脚本

ReScheme 层:
├── static_stdlib.scm          ← ML Runtime（已覆盖 cuBLAS/cuDNN/OpenBLAS）
├── static_prelude.scm         ← 预置函数（已修复 exit_program）
├── static_translate.py        ← 编译器（已支持 ml.* 命名空间）
└── stock/dgemm_wrapper.c      ← 矩阵乘 C 包装器（GPU/CPU 自动切换）
```

---

> **文档版本**: 2026-06-13
> **参考项目**: [my-img](https://github.com/quqiufeng/my-img) — 纯 C++ 版 ComfyUI
> **相关技术**: StaticPy / Chez Scheme AOT / ML Runtime / cuBLAS / cuDNN
