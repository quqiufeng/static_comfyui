# stable-diffusion.cpp 代码调用路径 & API 文档

## 代码调用路径

```
main()
├── sd_ctx_params_init(&ctx_params)            // stable-diffusion.h:stable-diffusion.cpp:2924
├── new_sd_ctx(&ctx_params)                    // stable-diffusion.h:stable-diffusion.cpp:3225
│   └── StableDiffusionGGML::init()            // src/stable-diffusion.cpp:438
│       ├── init_backend()                     // src/stable-diffusion.cpp:413
│       │   ├── ggml_backend_cuda_init(0)      // include/ggml-cuda.h:23 → ggml-cuda.cu:5662
│       │   └── ggml_backend_cpu_init()        // include/ggml-cpu.h:131 → ggml-cpu.cpp
│       ├── ModelLoader::init_from_file()      // src/model.cpp:211
│       │   └── init_from_safetensors_file()   // src/model.cpp:293
│       │       └── read_safetensors_file()    // src/model_io/safetensors_io.cpp:87
│       ├── new UNetModelRunner(backend)       // src/unet.hpp:651
│       │   └── DiffusionModelRunner()
│       │       ├── alloc_params_ctx()         // src/ggml_extend.hpp:1749
│       │       │   └── ggml_init({n,NULL,true})  // ggml.h:ggml.c:1736
│       │       ├── alloc_params_buffer()      // src/ggml_extend.hpp:2650
│       │       │   ├── unet.init(params_ctx)  // src/unet.hpp:203
│       │       │   │   └── ggml_new_tensor_4d(ctx,F32,...)  // ggml.c:1839
│       │       │   └── ggml_backend_alloc_ctx_tensors(ctx,be) // ggml-backend.h
│       │       └── alloc_compute_ctx()        // src/ggml_extend.hpp:1797
│       ├── CLIPTextModel ×2                   // src/clip.hpp
│       ├── AutoencoderKL                      // src/auto_encoder_kl.hpp
│       └── ModelLoader::load_tensors(map)     // src/model.cpp:1173
│           └── ggml_backend_tensor_set(t,data)   // ggml-backend.h:ggml-backend.cpp
│
├── sd_img_gen_params_init(&img)               // stable-diffusion.h:stable-diffusion.cpp:3100
├── generate_image(ctx, &img)                  // stable-diffusion.h:stable-diffusion.cpp:4626
│   ├── prepare latents
│   │   └── RNG::randn()
│   ├── prepare embeds
│   │   └── cond_stage_model->get_learned_condition(prompt)
│   │       ├── tokenize(prompt) → tokens[77]
│   │       ├── CLIP-L (GGMLRunner::compute)   // ggml_extend.hpp:2778
│   │       │   ├── alloc_compute_buffer(gf)   // ggml_extend.hpp:1884
│   │       │   │   └── ggml_gallocr_new(buft) // ggml-alloc.c:533
│   │       │   │   └── ggml_gallocr_reserve() // ggml-alloc.c:824
│   │       │   ├── ggml_gallocr_alloc_graph() // ggml-alloc.c:1051
│   │       │   ├── copy_data_to_backend_tensor() // ggml_extend.hpp:2036
│   │       │   │   └── ggml_backend_tensor_set(t,data)
│   │       │   └── ggml_backend_graph_compute(be,gf) // ggml-backend.h
│   │       │       └── ggml_backend_cuda_graph_compute() // ggml-cuda.cu:4454
│   │       ├── CLIP-G → [1,77,1280]
│   │       ├── CLIP-G pool → [1280]
│   │       ├── concat(L,G,dim=2) → [1,77,2048]
│   │       └── concat(pooled,time_ids) → y
│   └── StableDiffusionGGML::sample()          // stable-diffusion.cpp
│       ├── Karras sigmas
│       └── for each step:
│           ├── UNet forward [cond]
│           │   └── UNetModelRunner::compute()  // unet.hpp:675
│           │       └── GGMLRunner::compute()   // ggml_extend.hpp:2778
│           │           ├── alloc_compute_buffer(gf)
│           │           ├── offload_all_params() // ggml_extend.hpp:2091
│           │           ├── ggml_gallocr_alloc_graph()
│           │           ├── copy_data_to_backend_tensor()
│           │           └── ggml_backend_graph_compute(be,gf)
│           ├── UNet forward [uncond]
│           ├── CFG: uncond + scale×(cond−uncond)
│           └── Euler step
│   └── VAE decode → RGB image
└── free_sd_ctx(ctx)                           // stable-diffusion.h:stable-diffusion.cpp:3246
```

## stable-diffusion.h API（声明: include/stable-diffusion.h）

| API | 声明位置 | 实现位置 | 作用 |
|-----|----------|----------|------|
| `sd_ctx_params_t` | `:179` | — | 上下文参数结构体 |
| `sd_ctx_params_init()` | `:469` | `stable-diffusion.cpp:2924` | 初始化上下文参数 |
| `new_sd_ctx()` | `:472` | `stable-diffusion.cpp:3225` | 创建上下文 + 加载模型 |
| `free_sd_ctx()` | `:473` | `stable-diffusion.cpp:3246` | 释放上下文 |
| `sd_sample_params_t` | `:261` | — | 采样参数结构体 |
| `sd_sample_params_init()` | `:476` | `stable-diffusion.cpp:3040` | 初始化采样参数 |
| `sd_img_gen_params_t` | `:372` | — | 图像生成参数结构体 |
| `sd_img_gen_params_init()` | `:482` | `stable-diffusion.cpp:3100` | 初始化生成参数 |
| `generate_image()` | `:484` | `stable-diffusion.cpp:4626` | 完整 txt2img |
| `sd_image_t` | `:239` | — | {width, height, channel, *data} |

## 内部函数位置

| 函数 | 文件:行 |
|------|---------|
| `StableDiffusionGGML::init()` | `src/stable-diffusion.cpp:438` |
| `StableDiffusionGGML::init_backend()` | `src/stable-diffusion.cpp:413` |
| `StableDiffusionGGML::sample()` | `src/stable-diffusion.cpp` |
| `ModelLoader::init_from_file()` | `src/model.cpp:211` |
| `ModelLoader::init_from_safetensors_file()` | `src/model.cpp:293` |
| `ModelLoader::load_tensors(map)` | `src/model.cpp:1173` |
| `read_safetensors_file()` | `src/model_io/safetensors_io.cpp:87` |
| `GGMLRunner::alloc_params_ctx()` | `src/ggml_extend.hpp:1749` |
| `GGMLRunner::alloc_params_buffer()` | `src/ggml_extend.hpp:2650` |
| `GGMLRunner::alloc_compute_ctx()` | `src/ggml_extend.hpp:1797` |
| `GGMLRunner::alloc_compute_buffer()` | `src/ggml_extend.hpp:1884` |
| `GGMLRunner::offload_all_params()` | `src/ggml_extend.hpp:2091` |
| `GGMLRunner::execute_graph()` | `src/ggml_extend.hpp:2393` |
| `GGMLRunner::compute()` | `src/ggml_extend.hpp:2778` |
| `copy_data_to_backend_tensor()` | `src/ggml_extend.hpp:2036` |
| `UNetModelRunner::UNetModelRunner()` | `src/unet.hpp:651` |
| `UNetModelRunner::build_graph()` | `src/unet.hpp:675` |
| `unet.init()` | `src/unet.hpp:203` |
| `SpatialTransformer` 构造 | `src/common_block.hpp:457` |
| `CrossAttention::forward()` | `src/common_block.hpp:329` |
| `BasicTransformerBlock` 构造 | `src/common_block.hpp:384` |
| `ResBlock` 构造 | `src/common_block.hpp` |
| `FeedForward` 构造 | `src/common_block.hpp:230` |
| `Decoder::forward()` (VAE) | `src/auto_encoder_kl.hpp:442` |

## GGML API（权重加载）

| API | 实现位置 | 用途 |
|-----|----------|------|
| `ggml_init({n,NULL,true})` | `ggml.c:1736` | 建 ctx, no_alloc=true |
| `ggml_new_tensor_4d(ctx,F32,...)` | `ggml.c:1839` | 建 tensor (metadata only) |
| `ggml_set_name(t,name)` | `ggml.c:1907` | 命名 tensor |
| `ggml_backend_alloc_ctx_tensors(ctx,be)` | `ggml-backend.cpp` | 分配 GPU buffer |
| `ggml_backend_tensor_set(t,data,0,n)` | `ggml-backend.cpp` | 填数据 (CPU→GPU) |
| `ggml_backend_tensor_get(t,data,0,n)` | `ggml-backend.cpp` | 取数据 (GPU→CPU) |
| `ggml_backend_tensor_copy(src,dst)` | `ggml-backend.cpp:477` | 跨 backend 拷 tensor |

```
权重加载三步:
1. ggml_init({n, NULL, true})      // no_alloc=true
2. ggml_new_tensor_4d(ctx, F32, ...)  // 创建 tensor (仅 metadata)
3. ggml_backend_alloc_ctx_tensors(ctx, backend)  // 分配 GPU buffer
4. ggml_backend_tensor_set(t, data, 0, nbytes)  // 填入 F32 数据
```

## GGML API（推理）

| API | 实现位置 | 用途 |
|-----|----------|------|
| `ggml_gallocr_new(buft)` | `ggml-alloc.c:533` | 新建分配器 |
| `ggml_gallocr_reserve(allocr,gf)` | `ggml-alloc.c:824` | 预计算显存 |
| `ggml_gallocr_alloc_graph(allocr,gf)` | `ggml-alloc.c:1051` | 分配 |
| `ggml_backend_tensor_set_async(be,t,data)` | `ggml-backend.h` | 拷输入到 GPU |
| `ggml_backend_graph_compute(be,gf)` | `ggml-backend.h` | CUDA kernel 计算 |
| `ggml_backend_cuda_graph_compute()` | `ggml-cuda.cu:4454` | CUDA 后端实现 |

```
推理流程:
1. ggml_gallocr_new(buft)
2. ggml_gallocr_reserve(allocr, gf)
3. ggml_gallocr_alloc_graph(allocr, gf)
4. ggml_backend_tensor_set_async(inputs)
5. ggml_backend_graph_compute(backend, gf)
6. ggml_backend_tensor_get(output)
```

## GGML 算子（CUDA 后端支持）

| API | 实现 (ggml.c) | CUDA (ggml-cuda.cu) |
|-----|---------------|---------------------|
| `ggml_mul_mat()` | 矩阵乘法 | `ggml_cuda_op_mul_mat` → mmf.cu/mmq.cu |
| `ggml_conv_2d()` | 卷积 | → im2col + mul_mat (CUDA) |
| `ggml_add()` | 逐元素加 | `ggml_cuda_op_add` → binbcast.cu |
| `ggml_mul()` | 逐元素乘 | `ggml_cuda_op_mul` |
| `ggml_silu()` | SiLU | `ggml_cuda_op_silu` |
| `ggml_gelu()` | GELU | `ggml_cuda_op_gelu` |
| `ggml_group_norm()` | GroupNorm | CUDA kernel |
| `ggml_concat()` | 拼接 | CUDA kernel |
| `ggml_upscale()` | 上采样 | CUDA kernel |
| `ggml_permute()` | 维度重排 (view) | — |
| `ggml_cont()` | 连续化 | → `ggml_cont` / CUDA CPY |
| `ggml_reshape_*()` | 改维度 (view) | — |
| `ggml_timestep_embedding()` | 时间嵌入 | CUDA kernel |
| `ggml_soft_max()` | Softmax | CUDA kernel |
| `ggml_scale()` | 缩放 | `ggml_cuda_op_scale` |
| `ggml_flash_attn_ext()` | Flash Attention | `ggml_cuda_flash_attn_ext` |
| `ggml_new_graph()` | 建计算图 | — |
| `ggml_build_forward_expand()` | 展开图 | — |
| `ggml_dup_tensor()` | 复制 metadata | — |
| `ggml_nbytes()` | tensor 字节数 | — |
| `ggml_nelements()` | tensor 元素数 | — |

## 关键设计模式

1. **params_ctx**: `ggml_init({MAX_TENSORS×overhead, NULL, true})` — no_alloc=true, 只为 metadata
2. **weights 是 F32 类型**, 文件加载时做 F16→F32 转换
3. **计算图 graph**: 每次 `compute()` 重建, `ggml_build_forward_expand` 从 output 回溯展开
4. **所有 tensor 创建在前, buffer 分配在后**: 先建完所有 tensor 对象, 再一次性 `ggml_backend_alloc_ctx_tensors` 分配 GPU buffer
5. **推理 per-step**: gallocr 新建 → reserve → alloc_graph → set_input → compute → get_output → free
