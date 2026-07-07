# stable-diffusion.cpp 代码调用路径 & API 文档

## 代码调用路径

```
main()
├── sd_ctx_params_init()                    // stable-diffusion.cpp:2924
├── new_sd_ctx()                            // stable-diffusion.cpp:3225
│   └── StableDiffusionGGML::init()         // stable-diffusion.cpp:438
│       ├── init_backend()                  // stable-diffusion.cpp:413
│       │   ├── ggml_backend_cuda_init()
│       │   └── ggml_backend_cpu_init()
│       ├── ModelLoader::init_from_file()   // model.cpp:211
│       │   └── read_safetensors_file()     // safetensors_io.cpp:87
│       ├── new UNetModelRunner()           // unet.hpp:651
│       │   ├── alloc_params_ctx()          // ggml_extend.hpp:1749
│       │   │   └── ggml_init(no_alloc=true)
│       │   ├── alloc_params_buffer()       // ggml_extend.hpp:2650
│       │   │   ├── unet.init() → ggml_new_tensor_4d(F32) ×N
│       │   │   └── ggml_backend_alloc_ctx_tensors()
│       │   │       └── cudaMalloc (GPU buffer)
│       │   └── alloc_compute_ctx()         // ggml_extend.hpp:1797
│       ├── CLIP(×2) + VAE (同上流程)
│       └── ModelLoader::load_tensors()     // model.cpp:1173
│           └── ggml_backend_tensor_set() → cudaMemcpyAsync
│
├── sd_img_gen_params_init()                // stable-diffusion.cpp:3100
├── generate_image()                        // stable-diffusion.cpp:4626
│   ├── prepare latents (randn)
│   ├── prepare embeds → CLIP-L + CLIP-G → [77,2048]
│   └── sample() ×20 steps
│       ├── Karras sigmas
│       ├── UNet [cond] → GGMLRunner::compute()  // ggml_extend.hpp:2778
│       │   ├── alloc_compute_buffer(gf)    // ggml_extend.hpp:1884
│       │   │   └── ggml_gallocr_new + reserve
│       │   ├── offload_all_params()        // ggml_extend.hpp:2091
│       │   ├── ggml_gallocr_alloc_graph()
│       │   ├── copy_data_to_backend_tensor()
│       │   └── ggml_backend_graph_compute(be,gf)
│       │       └── ggml_backend_cuda_graph_compute()  // cuda.cu:4454
│       ├── UNet [uncond]
│       ├── CFG
│       └── Euler step
│   └── VAE decode
└── free_sd_ctx()                           // stable-diffusion.cpp:3246
```

## stable-diffusion.h API

| API | 位置 |
|-----|------|
| `sd_ctx_params_t` | include/stable-diffusion.h:179 |
| `sd_ctx_params_init()` | src/stable-diffusion.cpp:2924 |
| `new_sd_ctx()` | src/stable-diffusion.cpp:3225 |
| `free_sd_ctx()` | src/stable-diffusion.cpp:3246 |
| `sd_sample_params_t` | include/stable-diffusion.h:261 |
| `sd_sample_params_init()` | src/stable-diffusion.cpp:3040 |
| `sd_img_gen_params_t` | include/stable-diffusion.h:372 |
| `sd_img_gen_params_init()` | src/stable-diffusion.cpp:3100 |
| `generate_image()` | src/stable-diffusion.cpp:4626 |
| `sd_image_t` | include/stable-diffusion.h:239 |

## 权重加载三步

| API | 用途 |
|-----|------|
| `ggml_init({n,NULL,true})` | 建 ctx, no_alloc=true |
| `ggml_new_tensor_4d(ctx,F32,...)` | 建 tensor (metadata only) |
| `ggml_backend_alloc_ctx_tensors(ctx,be)` | 分配 GPU buffer |
| `ggml_backend_tensor_set(t,data,0,n)` | 填数据 (CPU→GPU) |

## 推理流程

| API | 用途 |
|-----|------|
| `ggml_gallocr_new(buft)` | 新建分配器 |
| `ggml_gallocr_reserve(allocr,gf)` | 预计算显存 |
| `ggml_gallocr_alloc_graph(allocr,gf)` | 分配 |
| `ggml_backend_tensor_set_async(be,t,data)` | 拷输入 |
| `ggml_backend_graph_compute(be,gf)` | 计算 |
| `ggml_backend_tensor_get(t,data,0,n)` | 取结果 |

## 内部函数位置

| 函数 | 文件:行 |
|------|---------|
| StableDiffusionGGML::init | stable-diffusion.cpp:438 |
| StableDiffusionGGML::init_backend | stable-diffusion.cpp:413 |
| ModelLoader::init_from_file | model.cpp:211 |
| ModelLoader::init_from_safetensors_file | model.cpp:293 |
| ModelLoader::load_tensors(map) | model.cpp:1173 |
| GGMLRunner::alloc_params_ctx | ggml_extend.hpp:1749 |
| GGMLRunner::alloc_params_buffer | ggml_extend.hpp:2650 |
| GGMLRunner::alloc_compute_ctx | ggml_extend.hpp:1797 |
| GGMLRunner::alloc_compute_buffer | ggml_extend.hpp:1884 |
| GGMLRunner::offload_all_params | ggml_extend.hpp:2091 |
| GGMLRunner::execute_graph | ggml_extend.hpp:2393 |
| GGMLRunner::compute | ggml_extend.hpp:2778 |
| read_safetensors_file | safetensors_io.cpp:87 |
| UNetModelRunner::UNetModelRunner | unet.hpp:651 |
