# stable-diffusion.cpp → GGML 代码调用路径 & API

> 对应文件: `cpp/stable-diffusion-cli.cpp` → `libstable-diffusion.a` → `ggml-*`

## 1. 初始化

```
main()
├── sd_ctx_params_init(&ctx_params)                          // stable-diffusion.h
├── new_sd_ctx(&ctx_params)                                  // stable-diffusion.h
│   └── new StableDiffusionGGML()
│       └── StableDiffusionGGML::init(ctx_params)
│           │
│           ├── init_backend(ctx_params)
│           │   └── SDBackendManager::init()
│           │       ├── ggml_backend_cuda_init(0)            // ggml-cuda.h
│           │       └── ggml_backend_cpu_init()              // ggml-cpu.h
│           │
│           ├── ModelLoader::init_from_file(model_path)
│           │   └── nlohmann::json::parse()                  // JSON header
│           │   └── → tensor_storage_map  (String2TensorStorage)
│           │
│           ├── new UNetModelRunner(backend, params_backend, storage, prefix)
│           │   └── DiffusionModelRunner::DiffusionModelRunner()
│           │       ├── alloc_params_ctx()
│           │       │   └── ggml_init({mem, NULL, true})    // ggml.h — no_alloc=true
│           │       ├── alloc_params_buffer()
│           │       │   ├── unet.init(params_ctx, storage, prefix)
│           │       │   │   └── ggml_new_tensor_4d(ctx, F32, ...)  × N
│           │       │   │   └── ggml_set_name(t, name)
│           │       │   └── ggml_backend_alloc_ctx_tensors(params_ctx, params_backend)
│           │       │       └── ggml_backend_buft_alloc_buffer()  → cudaMalloc
│           │       └── alloc_compute_ctx()
│           │           └── ggml_init({mem, NULL, true})    // compute_ctx
│           │
│           ├── new CLIPTextModel(backend, ...) ×2           // CLIP-L + CLIP-G
│           │   └── (同上流程: alloc_params_ctx + alloc_params_buffer)
│           │
│           ├── new AutoencoderKL(backend, ...)              // VAE
│           │   └── (同上流程)
│           │
│           └── ModelLoader::load_tensors(tensors)
│               └── 多线程读 safetensors F16 → 转 F32
│                   └── ggml_backend_tensor_set(t, data, 0, nbytes)  // CPU → GPU
│                       └── cudaMemcpyAsync()                // CUDA
│
├── sd_img_gen_params_init(&img_params)                      // stable-diffusion.h
```

## 2. 推理

```
generate_image(ctx, &img_params)                             // stable-diffusion.h
├── prepare latents:
│   └── RNG::randn() → noise
│
├── prepare embeds:
│   └── cond_stage_model->get_learned_condition(prompt)
│       ├── tokenize(prompt) → tokens[77]
│       ├── CLIP-L.compute(tokens)
│       │   └── GGMLRunner::compute<float>()
│       │       ├── alloc_compute_buffer(gf)
│       │       │   └── ggml_gallocr_new(buft)              // ggml-alloc.h
│       │       │   └── ggml_gallocr_reserve(allocr, gf)
│       │       ├── ggml_gallocr_alloc_graph(allocr, gf)
│       │       ├── copy_data_to_backend_tensor(gf)
│       │       │   └── ggml_backend_tensor_set(t, data, 0, nbytes)
│       │       └── ggml_backend_graph_compute(backend, gf) // ggml-backend.h
│       │           └── ggml_backend_cuda_graph_compute()
│       │
│       ├── CLIP-G.compute(tokens) → [1, 77, 1280]
│       ├── CLIP-G.compute(pooled) → [1280]
│       ├── concat(L, G, dim=2) → [1, 77, 2048] = context
│       └── concat(pooled, time_ids) → y
│
└── sample()
    ├── Karras sigmas
    └── for each step:
        ├── UNetModelRunner::compute() [cond]
        │   └── GGMLRunner::compute<float>()
        │       ├── alloc_compute_buffer(gf)
        │       │   └── ggml_gallocr_new(buft)
        │       │   └── ggml_gallocr_reserve(allocr, gf)
        │       ├── offload_all_params()
        │       │   ├── ggml_dup_tensor(offload_ctx, src)   // ggml.h
        │       │   ├── ggml_backend_alloc_ctx_tensors(offload_ctx, runtime_backend)
        │       │   └── ggml_backend_tensor_copy(src, dst)  // ggml-backend.h
        │       ├── ggml_gallocr_alloc_graph(allocr, gf)
        │       ├── copy_data_to_backend_tensor(gf)
        │       │   └── ggml_backend_tensor_set(t, data, 0, nbytes)
        │       └── ggml_backend_graph_compute(backend, gf)
        │
        ├── UNetModelRunner::compute() [uncond]             // 同上
        ├── CFG: uncond + scale × (cond − uncond)
        └── Euler step
    │
    └── VAE decode
        └── AutoencoderKL::decode(latent)
            └── GGMLRunner::compute<float>()                // 同上
```

## 3. stable-diffusion.h API

| API | 作用 |
|-----|------|
| `sd_ctx_params_t` | 上下文参数结构体 |
| `sd_ctx_params_init()` | 初始化上下文参数 |
| `new_sd_ctx()` | 创建 SD 上下文（加载模型） |
| `free_sd_ctx()` | 释放上下文 |
| `sd_img_gen_params_t` | 图像生成参数结构体 |
| `sd_img_gen_params_init()` | 初始化生成参数 |
| `generate_image()` | 执行 txt2img 管线 |
| `sd_image_t` | 图像数据结构体 |

## 4. GGML API

| API | 头文件 | 用途 |
|-----|--------|------|
| `ggml_init()` | `ggml.h` | 创建 context (metadata pool) |
| `ggml_init_params` | `ggml.h` | init 参数: `{mem_size, buffer, no_alloc}` |
| `ggml_new_tensor_1d()` | `ggml.h` | 创建 1D tensor (仅 metadata 若 no_alloc=true) |
| `ggml_new_tensor_2d()` | `ggml.h` | 创建 2D tensor |
| `ggml_new_tensor_3d()` | `ggml.h` | 创建 3D tensor |
| `ggml_new_tensor_4d()` | `ggml.h` | 创建 4D tensor |
| `ggml_dup_tensor()` | `ggml.h` | 复制 tensor metadata 到新 ctx |
| `ggml_set_name()` | `ggml.h` | 命名 tensor |
| `ggml_nbytes()` | `ggml.h` | tensor 数据字节数 |
| `ggml_mul_mat()` | `ggml.h` | 矩阵乘法 (A × B) |
| `ggml_conv_2d()` | `ggml.h` | 2D 卷积 |
| `ggml_add()` | `ggml.h` | 逐元素加法 |
| `ggml_silu()` | `ggml.h` | SiLU 激活函数 |
| `ggml_gelu()` | `ggml.h` | GELU 激活函数 |
| `ggml_group_norm()` | `ggml.h` | Group Normalization |
| `ggml_concat()` | `ggml.h` | 张量拼接 |
| `ggml_upscale()` | `ggml.h` | 上采样 (nearest) |
| `ggml_permute()` | `ggml.h` | 维度重排 (view) |
| `ggml_cont()` | `ggml.h` | 确保内存连续 (copy if needed) |
| `ggml_reshape_2d/3d/4d()` | `ggml.h` | 改变维度 (view) |
| `ggml_timestep_embedding()` | `ggml.h` | 时间嵌入 |
| `ggml_soft_max()` | `ggml.h` | Softmax |
| `ggml_scale()` | `ggml.h` | 缩放 |
| `ggml_flash_attn_ext()` | `ggml.h` | Flash Attention |
| `ggml_new_graph()` | `ggml.h` | 创建计算图 |
| `ggml_build_forward_expand()` | `ggml.h` | 展开计算图 |
| `ggml_backend_init_by_name()` | `ggml-backend.h` | 按名称初始化 backend |
| `ggml_backend_alloc_ctx_tensors()` | `ggml-backend.h` | 为 ctx 所有 tensor 分配 buffer |
| `ggml_backend_alloc_ctx_tensors_from_buft()` | `ggml-backend.h` | 同上，指定 buffer type |
| `ggml_backend_tensor_set()` | `ggml-backend.h` | CPU→GPU 拷数据 (同步) |
| `ggml_backend_tensor_get()` | `ggml-backend.h` | GPU→CPU 取数据 (同步) |
| `ggml_backend_tensor_set_async()` | `ggml-backend.h` | CPU→GPU 拷数据 (异步) |
| `ggml_backend_tensor_copy()` | `ggml-backend.h` | 跨 backend 拷 tensor |
| `ggml_backend_graph_compute()` | `ggml-backend.h` | 执行计算图 |
| `ggml_backend_synchronize()` | `ggml-backend.h` | 同步 backend |
| `ggml_backend_get_default_buffer_type()` | `ggml-backend.h` | 获取 backend 默认 buffer type |
| `ggml_backend_cpu_buffer_type()` | `ggml-backend.h` | CPU buffer type |
| `ggml_gallocr_new()` | `ggml-alloc.h` | 创建图内存分配器 |
| `ggml_gallocr_reserve()` | `ggml-alloc.h` | 预计算所需内存 |
| `ggml_gallocr_alloc_graph()` | `ggml-alloc.h` | 分配图中所有 tensor 内存 |
| `ggml_gallocr_free()` | `ggml-alloc.h` | 释放分配器 |
| `ggml_backend_cuda_init()` | `ggml-cuda.h` | 初始化 CUDA backend |
| `ggml_backend_cuda_get_device_count()` | `ggml-cuda.h` | CUDA 设备数 |
| `ggml_backend_cuda_buffer_type()` | `ggml-cuda.h` | CUDA buffer type |
| `ggml_backend_cpu_init()` | `ggml-cpu.h` | 初始化 CPU backend |

## 5. 关键设计

### 权重加载三步
```
1. params_ctx = ggml_init({mem, NULL, true})     // no_alloc=true, 仅 metadata
2. ggml_new_tensor_4d(ctx, F32, ...) ×N           // 创建 F32 tensor 对象
3. ggml_backend_alloc_ctx_tensors(ctx, backend)    // 分配 GPU buffer
4. ggml_backend_tensor_set(t, data, 0, nbytes)    // F16→F32 转换 + 填入 GPU
```

### 推理执行
```
1. ggml_gallocr_new(buft)         → 创建分配器
2. ggml_gallocr_reserve(allocr, gf) → 预计算 GPU 显存
3. ggml_gallocr_alloc_graph(allocr, gf) → 分配
4. ggml_backend_tensor_set/async(inputs) → 拷输入
5. ggml_backend_graph_compute(backend, gf) → CUDA kernel
6. ggml_backend_tensor_get(output) → GPU→CPU
```
