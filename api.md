# stable-diffusion.cpp 代码调用路径 & API 文档

## 代码调用路径

```
main()                                        // examples/cli/main.cpp:?
├── sd_ctx_params_init(&ctx_params)            // stable-diffusion.cpp:2924
├── new_sd_ctx(&ctx_params)                    // stable-diffusion.cpp:3225
│   └── StableDiffusionGGML::init()            // stable-diffusion.cpp:438
│       ├── init_backend()                     // stable-diffusion.cpp:413
│       │   ├── ggml_backend_cuda_init(0)      // ggml-cuda.h
│       │   └── ggml_backend_cpu_init()        // ggml-cpu.h
│       ├── ModelLoader::init_from_file()      // model.cpp:211
│       │   └── init_from_safetensors_file()   // model.cpp:293
│       │       └── read_safetensors_file()    // model_io/safetensors_io.cpp:87
│       │           └── nlohmann::json::parse() // thirdparty/json.hpp
│       ├── new UNetModelRunner(backend)       // unet.hpp:651
│       │   └── DiffusionModelRunner()
│       │       ├── alloc_params_ctx()         // ggml_extend.hpp:1749
│       │       │   └── ggml_init({n,NULL,true})
│       │       ├── alloc_params_buffer()      // ggml_extend.hpp:2650
│       │       │   ├── unet.init(params_ctx)   // unet.hpp:203
│       │       │   │   ├── ggml_new_tensor_4d(F32) ×N
│       │       │   │   └── ggml_set_name()
│       │       │   └── ggml_backend_alloc_ctx_tensors(params_ctx, backend)
│       │       └── alloc_compute_ctx()         // ggml_extend.hpp:1797
│       │           └── ggml_init({n,NULL,true})
│       ├── new CLIPTextModel(backend) ×2      // clip.hpp
│       ├── new AutoencoderKL(backend)         // auto_encoder_kl.hpp
│       └── ModelLoader::load_tensors()         // model.cpp:1173
│           └── ggml_backend_tensor_set(t,data) // ggml-backend.cpp:?
│
├── sd_img_gen_params_init(&img)                // stable-diffusion.cpp:3100
├── generate_image(ctx, &img)                   // stable-diffusion.cpp:4626
│   ├── prepare latents (randn)
│   ├── prepare embeds
│   │   └── CLIP-L + CLIP-G encode prompt → [77,2048]
│   └── sample() ×20 steps
│       ├── Karras sigmas
│       ├── UNet forward [cond]
│       │   └── GGMLRunner::compute()           // ggml_extend.hpp:2778
│       │       ├── alloc_compute_buffer(gf)    // ggml_extend.hpp:1884
│       │       │   ├── ggml_gallocr_new(buft)
│       │       │   └── ggml_gallocr_reserve()
│       │       ├── offload_all_params()        // ggml_extend.hpp:2091
│       │       │   ├── ggml_dup_tensor()
│       │       │   ├── ggml_backend_alloc_ctx_tensors()
│       │       │   └── ggml_backend_tensor_copy()
│       │       ├── ggml_gallocr_alloc_graph()
│       │       ├── copy_data_to_backend_tensor() // ggml_extend.hpp:2036
│       │       │   └── ggml_backend_tensor_set()
│       │       └── ggml_backend_graph_compute(be,gf)
│       │           └── ggml_backend_cuda_graph_compute() // ggml-cuda.cu:4454
│       ├── UNet forward [uncond]
│       ├── CFG: uncond + scale×(cond−uncond)
│       └── Euler step
│   └── VAE decode → RGB image
└── free_sd_ctx(ctx)                            // stable-diffusion.cpp:3246
```

## stable-diffusion.h API

| API | 位置 | 作用 |
|-----|------|------|
| `sd_ctx_params_t` | `include/stable-diffusion.h:179` | 上下文参数结构体 |
| `sd_ctx_params_init()` | `src/stable-diffusion.cpp:2924` | 初始化上下文参数 |
| `new_sd_ctx()` | `src/stable-diffusion.cpp:3225` | 创建 SD 上下文（加载模型） |
| `free_sd_ctx()` | `src/stable-diffusion.cpp:3246` | 释放上下文 |
| `sd_sample_params_t` | `include/stable-diffusion.h:261` | 采样参数结构体 |
| `sd_sample_params_init()` | `src/stable-diffusion.cpp:3040` | 初始化采样参数 |
| `sd_img_gen_params_t` | `include/stable-diffusion.h:372` | 图像生成参数结构体 |
| `sd_img_gen_params_init()` | `src/stable-diffusion.cpp:3100` | 初始化生成参数 |
| `generate_image()` | `src/stable-diffusion.cpp:4626` | 完整 txt2img：CLIP→UNet→VAE→PNG |
| `sd_image_t` | `include/stable-diffusion.h:239` | {width, height, channel, *data} |

## 内部关键函数

| 函数 | 位置 | 作用 |
|------|------|------|
| `StableDiffusionGGML::init()` | `src/stable-diffusion.cpp:438` | 模型加载主入口 |
| `StableDiffusionGGML::init_backend()` | `src/stable-diffusion.cpp:413` | 初始化计算后端 |
| `ModelLoader::init_from_file()` | `src/model.cpp:211` | 打开 safetensors 文件 |
| `ModelLoader::init_from_safetensors_file()` | `src/model.cpp:293` | 解析 safetensors JSON |
| `ModelLoader::load_tensors(map)` | `src/model.cpp:1173` | 填充 tensor 数据 |
| `GGMLRunner::alloc_params_ctx()` | `src/ggml_extend.hpp:1749` | 创建 params_ctx (no_alloc=true) |
| `GGMLRunner::alloc_params_buffer()` | `src/ggml_extend.hpp:2650` | 分配 GPU buffer + 创建 tensor |
| `GGMLRunner::alloc_compute_ctx()` | `src/ggml_extend.hpp:1797` | 创建 compute_ctx |
| `GGMLRunner::alloc_compute_buffer()` | `src/ggml_extend.hpp:1884` | 创建 gallocr + reserve |
| `GGMLRunner::offload_all_params()` | `src/ggml_extend.hpp:2091` | 权重 CPU→GPU |
| `GGMLRunner::execute_graph()` | `src/ggml_extend.hpp:2393` | 分配+拷入+计算+取结果 |
| `GGMLRunner::compute()` | `src/ggml_extend.hpp:2778` | 推理入口 |
| `copy_data_to_backend_tensor()` | `src/ggml_extend.hpp:2036` | 拷输入数据到 backend |
| `read_safetensors_file()` | `src/model_io/safetensors_io.cpp:87` | 读取 safetensors 数据 |
| `UNetModelRunner::UNetModelRunner()` | `src/unet.hpp:651` | UNet 构造 |
| `unet.init()` | `src/unet.hpp:203` | 创建 UNet 所有权重 tensor |
| `TensorStorage` | `src/model_io/tensor_storage.h` | tensor 元数据 {name, type, ne, file_offset} |

## GGML 权重加载 API

| API | 声明位置 | 用途 |
|-----|----------|------|
| `ggml_init({n, NULL, true})` | `include/ggml.h` | no_alloc=true, 仅metadata |
| `ggml_new_tensor_4d(ctx, F32, ...)` | `include/ggml.h` | 创建 F32 tensor 对象 |
| `ggml_set_name()` | `include/ggml.h` | 命名 tensor |
| `ggml_backend_alloc_ctx_tensors(ctx, be)` | `include/ggml-backend.h` | 为 ctx 下所有 tensor 分配 GPU buffer |
| `ggml_backend_alloc_ctx_tensors_from_buft()` | `include/ggml-backend.h` | 同上，指定 buffer type |
| `ggml_backend_tensor_set(t, data, 0, nbytes)` | `include/ggml-backend.h` | CPU→GPU 拷数据 |

```
权重加载三步:
1. params_ctx = ggml_init({n, NULL, true})     // no_alloc=true
2. ggml_new_tensor_4d(ctx, F32, ...) ×N        // 创建 F32 tensor (仅metadata)
3. ggml_backend_alloc_ctx_tensors(ctx, backend) // 分配 GPU buffer
4. ggml_backend_tensor_set(t, data, 0, nbytes) // 填入 F32 数据
```

## GGML 推理 API

| API | 声明位置 | 用途 |
|-----|----------|------|
| `ggml_gallocr_new(buft)` | `include/ggml-alloc.h` | 创建图内存分配器 |
| `ggml_gallocr_reserve(allocr, gf)` | `include/ggml-alloc.h` | 预计算所需 GPU 显存 |
| `ggml_gallocr_alloc_graph(allocr, gf)` | `include/ggml-alloc.h` | 分配图中所有 tensor 内存 |
| `ggml_gallocr_free(allocr)` | `include/ggml-alloc.h` | 释放分配器 |
| `ggml_backend_tensor_set_async(be,t,data)` | `include/ggml-backend.h` | 拷输入数据到 GPU |
| `ggml_backend_graph_compute(be, gf)` | `include/ggml-backend.h` | 执行计算图 (→ CUDA kernel) |
| `ggml_backend_cuda_graph_compute()` | `src/ggml-cuda/ggml-cuda.cu:4454` | CUDA 后端实现 |
| `ggml_backend_tensor_get(t, data, 0, nbytes)` | `include/ggml-backend.h` | GPU→CPU 取结果 |
| `ggml_backend_tensor_copy(src, dst)` | `include/ggml-backend.h` | 跨 backend 拷 tensor |
| `ggml_backend_synchronize(be)` | `include/ggml-backend.h` | 同步 backend |
| `ggml_backend_get_default_buffer_type(be)` | `include/ggml-backend.h` | 获取 backend 默认 buffer type |
| `ggml_backend_cpu_buffer_type()` | `include/ggml-backend.h` | CPU buffer type |

```
推理流程:
1. ggml_gallocr_new(buft)
2. ggml_gallocr_reserve(allocr, gf)
3. ggml_gallocr_alloc_graph(allocr, gf)
4. ggml_backend_tensor_set_async(inputs)
5. ggml_backend_graph_compute(backend, gf)
6. ggml_backend_tensor_get(output)
```

## GGML 算子 API

| API | 声明位置 | 用途 | CUDA 实现 |
|-----|----------|------|-----------|
| `ggml_mul_mat()` | `include/ggml.h` | 矩阵乘法 | `ggml-cuda.cu:ggml_cuda_op_mul_mat` |
| `ggml_conv_2d()` | `include/ggml.h` | 2D 卷积 | `ggml-cuda.cu:ggml_cuda_compute_forward` |
| `ggml_add()` | `include/ggml.h` | 逐元素加法 | `ggml-cuda.cu:ggml_cuda_op_add` |
| `ggml_mul()` | `include/ggml.h` | 逐元素乘法 | `ggml-cuda.cu:ggml_cuda_op_mul` |
| `ggml_silu()` | `include/ggml.h` | SiLU 激活 | `ggml-cuda.cu:ggml_cuda_op_silu` |
| `ggml_gelu()` | `include/ggml.h` | GELU 激活 | `ggml-cuda.cu` |
| `ggml_group_norm()` | `include/ggml.h` | GroupNorm | `ggml-cuda.cu:GGML_OP_GROUP_NORM` |
| `ggml_concat()` | `include/ggml.h` | 拼接 | `ggml-cuda.cu:GGML_OP_CONCAT` |
| `ggml_upscale()` | `include/ggml.h` | 上采样 | `ggml-cuda.cu:GGML_OP_UPSCALE` |
| `ggml_permute()` | `include/ggml.h` | 维度重排(view) | — |
| `ggml_cont()` | `include/ggml.h` | 确保连续 | `ggml-cuda.cu:GGML_OP_CPY` |
| `ggml_reshape_2d/3d/4d()` | `include/ggml.h` | 改维度(view) | — |
| `ggml_timestep_embedding()` | `include/ggml.h` | 时间嵌入 | `ggml-cuda.cu:GGML_OP_TIMESTEP_EMBEDDING` |
| `ggml_soft_max()` | `include/ggml.h` | Softmax | `ggml-cuda.cu` |
| `ggml_scale()` | `include/ggml.h` | 缩放 | `ggml-cuda.cu:ggml_cuda_op_scale` |
| `ggml_flash_attn_ext()` | `include/ggml.h` | Flash Attention | `ggml-cuda.cu:ggml_cuda_flash_attn_ext` |
| `ggml_new_graph()` | `include/ggml.h` | 建计算图 | — |
| `ggml_build_forward_expand()` | `include/ggml.h` | 展开计算图 | — |
| `ggml_dup_tensor()` | `include/ggml.h` | 复制 tensor metadata | — |
| `ggml_nbytes()` | `include/ggml.h` | tensor 字节数 | — |
| `ggml_nelements()` | `include/ggml.h` | tensor 元素数 | — |

## Backend API

| API | 声明位置 | 用途 |
|-----|----------|------|
| `ggml_backend_cuda_init(device)` | `include/ggml-cuda.h` | 初始化 CUDA backend |
| `ggml_backend_cuda_get_device_count()` | `include/ggml-cuda.h` | CUDA 设备数 |
| `ggml_backend_cuda_buffer_type(device)` | `include/ggml-cuda.h` | CUDA buffer type |
| `ggml_backend_cuda_host_buffer_type()` | `include/ggml-cuda.h` | CUDA pinned host buffer type |
| `ggml_backend_cpu_init()` | `include/ggml-cpu.h` | 初始化 CPU backend |
| `ggml_backend_cpu_buffer_type()` | `include/ggml-backend.h` | CPU buffer type |
| `ggml_backend_is_cpu(be)` | `include/ggml-backend.h` | 判断是否 CPU backend |
| `ggml_backend_init_by_name()` | `include/ggml-backend.h` | 按名称初始化 backend |
