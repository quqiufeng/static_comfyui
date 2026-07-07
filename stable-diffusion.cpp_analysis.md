# stable-diffusion.cpp 代码分析报告

> 基于 code search 生成。用于指导从零实现 SDXL 推理引擎。

## 1. 整体架构

```
stable-diffusion.h (C API)
    ↓
stable-diffusion.cpp (主逻辑: new_sd_ctx, generate_image, sample)
    ↓
model.h/cpp           ← 模型加载 (safetensors, gguf, torch legacy)
unet.hpp              ← UNet 模型定义
clip.hpp              ← CLIP 文本编码
auto_encoder_kl.hpp   ← VAE 编解码
denoiser.hpp          ← 噪声调度 + 采样器
conditioner.hpp       ← 文本条件处理
ggml_extend.hpp       ← GGMLRunner (graph 构建 + 调度器 + 执行)
common_block.hpp      ← 通用网络块 (ResBlock, SpatialTransformer, etc.)
```

## 2. 核心类关系

```
ModelLoader
  ├── init_from_safetensors_file()  ← safetensors 加载
  ├── init_from_gguf_file()         ← GGUF 加载
  └── get_tensor(name)              ← 按名取张量

GGMLRunner
  ├── compute(get_graph_cb, ...)    ← 构建 + 计算图
  ├── offload_all_params()          ← 权重 CPU→GPU 搬运
  ├── alloc_compute_buffer()        ← gallocr 分配
  └── execute_graph()               ← 执行计算

DiffusionModelRunner : GGMLRunner
  └── compute(params)               ← 虚函数, 由 UNetModelRunner 实现

UNetModelRunner : DiffusionModelRunner
  ├── build_graph(x, timesteps, context, y)  ← 构建 UNet 计算图
  └── compute(params)                        ← 执行

CompVisDenoiser
  ├── sigmas[], log_sigmas[]        ← 1000 个预计算 sigma
  ├── get_scalings(sigma)           → [c_skip, c_out, c_in]
  ├── sigma_to_t(sigma)            → timestep [0,999]
  ├── t_to_sigma(t)                → sigma
  ├── noise_scaling(sigma, noise, latent) → 初始噪声
  └── timestep_embedding(timesteps, dim)  → 正弦编码
```

## 3. SDXL UNet 结构

### 参数
- `model_channels`: 320
- `channel_mult`: [1, 2, 4]
- `num_res_blocks`: [2, 2, 2]
- `attention_levels`: [2, 4] (即 depth=2 和 4 处有 attention)
- `context_dim`: 2048 (CLIP-L 768 + CLIP-G 1280)
- `adm_in_channels`: 2816 (pooled 1280 + time_ids 6×256)

### 编码器
```
input_blocks:
  0: Conv2d(4→320, k=3, p=1)         → [320, 128, 128]
  1: ResBlock(320→320)               → [320, 128, 128]
  2: ResBlock(320→320)               → [320, 128, 128]
  3: DownSample(320→320)             → [320, 64, 64]
  4: ResBlock(320→640) + Attn×2      → [640, 64, 64]
  5: ResBlock(640→640) + Attn×2      → [640, 64, 64]
  6: DownSample(640→640)             → [640, 32, 32]
  7: ResBlock(640→1280) + Attn×10    → [1280, 32, 32]
  8: ResBlock(1280→1280) + Attn×10   → [1280, 32, 32]
```

### 中间
```
middle_block:
  0: ResBlock(1280→1280)
  1: SpatialTransformer ×10 (self + cross + ff)
  2: ResBlock(1280→1280)
```

### 解码器 (skip connections 匹配 spatial dims)
```
output_blocks:
  0: ResBlock(2560→1280) + Attn×10   ← cat skip 8
  1: ResBlock(2560→1280) + Attn×10   ← cat skip 7
  2: ResBlock(1920→1280) + Attn×10   ← cat skip 6 → Upsample → 64×64
  3: ResBlock(1280+640→640) + Attn×2  ← cat skip 5
  4: ResBlock(1280→640) + Attn×2     ← cat skip 4
  5: ResBlock(1280→640) + Attn×2     ← cat skip 3 → Upsample → 128×128
  6: ResBlock(640→320)               ← cat skip 2
  7: ResBlock(640→320)               ← cat skip 1
  8: ResBlock(640→320)               ← cat skip 0/conv_in
```

### 输出
```
out:
  0: GroupNorm(32, 320) + silu
  2: Conv2d(320→4, k=3, p=1)
```

## 4. ResBlock 结构

```
ResBlock(in_channels, emb_channels, out_channels):
  in_layers.0: GroupNorm(32, in_channels)
  (silu - 内置在 forward 中)
  in_layers.2: Conv2d(in_channels, out_channels, k=3, p=1)
  emb_layers.1: Linear(emb_channels, out_channels)  # time embedding projection
  out_layers.0: GroupNorm(32, out_channels)
  (silu)
  out_layers.3: Conv2d(out_channels, out_channels, k=3, p=1)
  skip_connection: Conv2d(in→out, k=1) [if in != out]
```

Forward:
```
h = silu(group_norm(x))
h = conv2d(h)
h = h + linear(silu(time_embed))  # broadcast [out_ch, 1, 1]
h = silu(group_norm(h))
h = conv2d(h)
return h + skip_connection(x)
```

## 5. SpatialTransformer 结构

```
SpatialTransformer(in_channels, n_head, d_head, depth, context_dim):
  norm: GroupNorm(32, in_channels)
  proj_in: Linear(in_channels, inner_dim)  # inner_dim = n_head * d_head
  transformer_blocks × depth:
    BasicTransformerBlock(inner_dim, n_head, d_head, context_dim):
      norm1: LayerNorm(inner_dim)
      attn1: SelfAttention(inner_dim, n_head, d_head)  # Q=K=V from x
      norm2: LayerNorm(inner_dim)
      attn2: CrossAttention(inner_dim, context_dim, n_head, d_head)  # Q from x, KV from context
      norm3: LayerNorm(inner_dim)
      ff: GEGLU(inner_dim, inner_dim×4)  # Linear(inner_dim, inner_dim×8) → gelu(gate)*value
  proj_out: Linear(inner_dim, in_channels)
```

Forward:
```
x_in = x
x = norm(x)
x = permute → reshape → [C, H*W, N]  # for linear mode
x = proj_in(x)
for each transformer_block:
  r = x
  x = attn1(norm1(x)) + r       # self-attention
  r = x
  x = attn2(norm2(x), context) + r  # cross-attention
  r = x
  x = ff(norm3(x)) + r          # GEGLU
x = proj_out(x)
x = reshape → permute → [W, H, C, N]  # back to spatial
x = x + x_in                    # residual
```

## 6. 注意力实现 (CrossAttention)

```
CrossAttention(query_dim, context_dim, heads, dim_head):
  to_q: Linear(query_dim, inner_dim)        # inner_dim = heads * dim_head
  to_k: Linear(context_dim, inner_dim)
  to_v: Linear(context_dim, inner_dim)
  to_out.0: Linear(inner_dim, query_dim)
```

Forward (stable-diffusion.cpp 的 ggml_ext_attention_ext):
```
q = to_q(x)    # [N, L_q, inner_dim]
k = to_k(ctx)  # [N, L_kv, inner_dim]
v = to_v(ctx)  # [N, L_kv, inner_dim]

# reshape for multi-head:
q = permute(q, 1, 2, 0, 3)  # [N, n_head, L_q, d_head]
k = permute(k, 1, 2, 0, 3)  # [N, n_head, L_kv, d_head]
v = permute(v, 1, 2, 0, 3)  # [N, n_head, L_kv, d_head]

# attention:
kq = matmul(k, q)  # [N, n_head, L_q, L_kv]
kq = softmax(kq / sqrt(d_head))
kqv = matmul(v, kq)  # [N, n_head, L_q, d_head]

# reshape back:
kqv = permute(kqv, 0, 2, 1, 3)  # [N, L_q, n_head, d_head]
out = to_out.0(kqv)
```

## 7. 采样流程

### generate_image 伪代码
```
generate_image(ctx, params):
  1. 生成 noise: randn([1, 4, H/8, W/8])
  2. 计算 sigmas (Karras/Discrete/etc)
  3. cond = get_learned_condition(prompt)        # CLIP encode
  4. uncond = get_learned_condition(neg_prompt)    # 或无条件
  5. x = noise_scaling(sigmas[0], noise, latent)   # x = noise * sigma_0
  6. for step in steps:
       c_in = get_scalings(sigma)[2]               # 1/sqrt(sigma²+sigma_data²)
       noised = x * c_in
       cond_out = UNet(noised, timestep, cond)
       uncond_out = UNet(noised, timestep, uncond)
       guided = CFG(cond_out, uncond_out, cfg_scale)
       denoised = guided * c_out + x * c_skip       # c_out=-sigma, c_skip=1
       d = (x - denoised) / sigma                   # = eps
       x = x + d * dt                               # Euler step
  7. decode_first_stage(x) → image                 # VAE decode
```

### Denoiser 缩放公式
```
get_scalings(sigma) → [c_skip, c_out, c_in]:
  c_skip = 1.0
  c_out  = -sigma
  c_in   = 1/sqrt(sigma² + sigma_data²)    # sigma_data = 1.0
```

### Euler 采样器
```
sample_euler(model, x, sigmas):
  for i in range(steps):
    sigma = sigmas[i]
    denoised = model(x, sigma)
    d = (x - denoised) / sigma
    x = x + d * (sigmas[i+1] - sigma)
  return x
```

## 8. 模型加载

```
ModelLoader::load_from_file(path, ...):
  1. 检测文件格式 (safetensors / gguf / torch / diffusers)
  2. 读取 tensor_storage 列表 (名称, 形状, 数据类型, 偏移量)
  3. 过滤无用 tensor (is_unused_tensor)
  4. 存入 ModelLoader.tensors map
```

权重 tensor 存储格式:
```cpp
struct TensorStorage {
    std::string name;        // 完整名称 (含前缀)
    std::vector<int64_t> shape;
    ggml_type type;          // F16/F32/etc
    size_t file_index;       // 对应文件
    size_t offset;           // 在文件中的偏移
    size_t size;             // 数据大小
};
```

## 9. Conditioning (文本编码)

### SDXL 双 CLIP 编码
```
get_learned_condition:
  1. tokenize(text) → tokens (77 tokens, padding/EOS)
  2. CLIP-L (768-dim, 12 layers):
     text_model.compute(tokens) → [1, 77, 768]
  3. CLIP-G (1280-dim, 32 layers):
     text_model2.compute(tokens) → [1, 77, 1280]
  4. hidden_states = concat(chunk_l, chunk_g, dim=2)  # [1, 77, 2048]
  5. pooled = text_model2.compute(tokens, return_pooled=true)  # [1280]
  6. vector = concat(pooled, time_ids_sinusoidal, ...)  # [2816]
```

### Vector Conditioning (y / adm_in_channels)
```
y = [pooled_CLIP_G(1280), sine(H,256), sine(W,256), sine(0,256), sine(0,256), sine(H,256), sine(W,256)]
  = 1280 + 6×256 = 2816
```

## 10. GGML Runner 执行流程

```
GGMLRunner::compute(get_graph, n_threads):
  1. alloc_compute_ctx()          # ggml_init(no_alloc=true)
  2. gf = get_graph()             # 构建计算图 (tensor metadata only)
  3. alloc_compute_buffer(gf)     # gallocr reserve + measure
  4. offload_all_params()         # 权重 CPU→GPU 搬运
  5. execute_graph(gf):           # 执行
     a. ggml_gallocr_alloc_graph  # 分配内存
     b. copy_data_to_backend      # 复制输入数据到 backend
     c. backend_graph_compute     # 计算
     d. read_graph_tensor(result) # 读取结果
```

### 权重搬运 (offload_all_params)
```cpp
// 1. 创建 offload_ctx (duplicate tensors)
for each tensor in params_ctx:
    ggml_dup_tensor(offload_ctx, t)

// 2. 分配 GPU 内存
runtime_params_buffer = ggml_backend_alloc_ctx_tensors(offload_ctx, runtime_backend)

// 3. 复制数据 + 交换指针
for each (src in params_ctx, dst in offload_ctx):
    ggml_backend_tensor_copy(src, dst)
    swap(src->data, dst->data)     // now params_ctx tensors point to GPU
    swap(src->buffer, dst->buffer)
```

## 11. 关键文件清单

| 文件 | 用途 | 头文件/源文件 |
|------|------|-------------|
| `model.h/cpp` | ModelLoader, TensorStorage | `.h` + `.cpp` |
| `unet.hpp` | UNetModel, UNetModelRunner | 纯头文件 |
| `clip.hpp` | CLIPTextModel | 纯头文件 |
| `common_block.hpp` | ResBlock, SpatialTransformer, CrossAttention, GEGLU | 纯头文件 |
| `denoiser.hpp` | Denoiser 基类 + CompVisDenoiser, 采样器 | 纯头文件 |
| `diffusion_model.hpp` | DiffusionModelRunner | 纯头文件 |
| `ggml_extend.hpp` | GGMLRunner (核心执行引擎) | 纯头文件 |
| `auto_encoder_kl.hpp` | AutoencoderKL (VAE) | 纯头文件 |
| `conditioner.hpp` | FrozenCLIPEmbedder | 纯头文件 |
| `guidance.h/cpp` | CFG 实现 | `.h` + `.cpp` |
| `stable-diffusion.cpp` | 主逻辑 | `.cpp` 7500+ 行 |
| `model_io/safetensors_io.h/cpp` | safetensors 读取 | `.h` + `.cpp` |
| `tokenizers/clip_tokenizer.h/cpp` | CLIP BPE tokenizer | `.h` + `.cpp` |
| `ggml_extend_backend.h/cpp` | Backend 管理 | `.h` + `.cpp` |
