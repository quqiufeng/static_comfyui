# SDXL Pipeline 参考实现

## sdxl_pipeline.py

基于 ComfyUI 原生 `sample()` 的 SDXL txt2img 参考实现。

### 路径
```
/opt/static_comfyui/sdxl_pipeline.py
```

### 运行
```bash
/data/venv/bin/python /opt/static_comfyui/sdxl_pipeline.py
```

### 功能
- 加载 `/data/models/image/sd_xl_base_1.0.safetensors` 完整 checkpoint
- 使用 ComfyUI 的 `load_checkpoint_guess_config` 加载模型（UNet + CLIP + VAE）
- 20 步 Euler 采样，CFG=7.0，seed=42
- 输出：`/home/quqiufeng/python_reference.png`（1024×1024 PNG）

### 输出验证
```
$ file /home/quqiufeng/python_reference.png
PNG image data, 1024 x 1024, 8-bit/color RGB, non-interlaced
$ ls -lh /home/quqiufeng/python_reference.png
1.3M
```

### 依赖
- Python 3.12（`/data/venv/bin/python`）
- ComfyUI（`/opt/static_comfyui/ComfyUI/`）
- PyTorch + CUDA
- 模型文件（safetensors）

## img.sh（stable-diffusion.cpp）

基于 myimg-cli（stable-diffusion.cpp GGML 后端）的 SDXL 出图。

### 路径
```
/opt/static_comfyui/img.sh
```

### 运行
```bash
bash /opt/static_comfyui/img.sh
```

### 调用路径追踪

```
img.sh
  └── myimg-cli (main.cpp)
        └── sdcpp_adapter.cpp
              ├── new_sd_ctx()
              │     └── StableDiffusionGGML::init()
              │           ├── ModelLoader::load_from_file()  ← safetensors 加载
              │           │     ├── read_safetensors_file()
              │           │     ├── ModelLoader::init_storage() → tensor map
              │           │     └── check_unused_tensors()  ← "unknown tensor" 警告
              │           ├── UNetModelRunner::load_from_file()  ← UNet weights
              │           ├── CLIPTextModel::load_from_file()  ×2  ← CLIP-L + CLIP-G
              │           ├── AutoencoderKL::load_from_file()  ← VAE weights
              │           └── CompVisDenoiser()  ← sigma 调度初始化
              │
              └── generate_image()
                    ├── prepare_image_generation_latents()
                    │     └── rng → randn → noise_scaling(sigmas[0], noise)
                    │
                    ├── prepare_image_generation_embeds()
                    │     └── cond_stage_model->get_learned_condition()
                    │           ├── clip.tokenize(prompt)  → tokens[77]
                    │           ├── CLIP-L.compute(tokens)  → [1, 77, 768]
                    │           ├── CLIP-G.compute(tokens)  → [1, 77, 1280]
                    │           ├── CLIP-G.compute(pooled)  → [1280]
                    │           ├── concat(L, G, dim=2)     → [1, 77, 2048]
                    │           └── concat(pooled, time_ids)→ [2816] (y)
                    │
                    └── StableDiffusionGGML::sample()
                          ├── Karras sigmas (get_sigmas)
                          │
                          ├── denoise loop (for each step):
                          │   ├── get_scalings(sigma) → [c_skip=1, c_out=-σ, c_in=1/√(σ²+1)]
                          │   ├── noised_input = x * c_in
                          │   ├── prepare_sample_timesteps(sigma) → timestep [0-999]
                          │   │
                          │   ├── run_condition(cond)  ← cond UNet forward
                          │   │     └── UNetModelRunner::compute()
                          │   │           ├── build_graph() → GGML cgraph
                          │   │           ├── GGMLRunner::compute()
                          │   │           │     ├── alloc_compute_buffer()  ← gallocr
                          │   │           │     ├── offload_all_params()    ← weights CPU→GPU
                          │   │           │     └── execute_graph()
                          │   │           │           ├── gallocr_alloc_graph
                          │   │           │           ├── copy_data_to_backend
                          │   │           │           ├── backend_graph_compute (CUDA)
                          │   │           │           └── read_graph_tensor → eps
                          │   │
                          │   ├── run_condition(uncond)  ← uncond UNet forward
                          │   │
                          │   ├── CFG: guided = uncond + cfg_scale × (cond - uncond)
                          │   │     (ClassifierFreeGuidance::forward)
                          │   │
                          │   ├── denoised = guided.pred × c_out + x × c_skip
                          │   │
                          │   └── Euler step:
                          │         d = (x - denoised) / sigma  (= eps)
                          │         x = x + d × (sigma_next - sigma)
                          │
                          └── decode_first_stage()  ← VAE decode
                                └── AutoencoderKL::decode()
                                      ├── post_quant_conv
                                      ├── decoder up blocks ×4
                                      └── conv_out → RGB image
```

### CFG 公式（stable-diffusion.cpp guidance.cpp:72）
```
ClassifierFreeGuidance::forward:
  guided = pred_uncond + guidance_scale × (pred_cond - pred_uncond)
```
然后: `denoised = guided × c_out + x × c_skip`，其中 `c_out=-sigma, c_skip=1`

### 关键类
| 类 | 文件 | 职责 |
|------|------|------|
| `StableDiffusionGGML` | `stable-diffusion.cpp` | 主控制器（init, sample, decode） |
| `ModelLoader` | `model.h/cpp` | 模型加载（safetensors） |
| `UNetModelRunner` | `unet.hpp` | UNet 前向计算 |
| `CLIPTextModel` | `clip.hpp` | CLIP 文本编码 |
| `CompVisDenoiser` | `denoiser.hpp` | sigma 调度 + get_scalings |
| `AutoencoderKL` | `auto_encoder_kl.hpp` | VAE 编解码 |
| `GGMLRunner` | `ggml_extend.hpp` | GGML 图执行引擎 |
| `ClassifierFreeGuidance` | `guidance.cpp` | CFG 公式实现 |

### 验证
```bash
$ bash img.sh  # 约12秒
$ file /tmp/output.png
PNG image data, 1024 x 1024, 8-bit/color RGB, non-interlaced
```

## stable-diffusion-cli.cpp

基于 stable-diffusion.cpp C API 的 SDXL txt2img，单 C++ 文件。

### 路径
```
/opt/static_comfyui/cpp/stable-diffusion-cli_v1.cpp
```

### 编译
```bash
cd cpp
SD=/opt/stable-diffusion.cpp
GGML=$SD/build/ggml/src
CUDA=/data/cuda/targets/x86_64-linux/lib
g++ -O3 -std=gnu++17 stable-diffusion-cli.cpp \
    -I$SD/include \
    $SD/build/libstable-diffusion.a \
    $GGML/libggml.a $GGML/libggml-base.a $GGML/libggml-cpu.a \
    -Wl,--whole-archive $GGML/ggml-cuda/libggml-cuda.a -Wl,--no-whole-archive \
    -ldl -lpthread -lm -lgomp \
    $CUDA/libcudart.so $CUDA/libcublas.so.12.6.4.1 \
    $CUDA/libcublasLt.so.12.6.4.1 $CUDA/stubs/libcuda.so \
    $CUDA/libculibos.a \
    -L/usr/lib/x86_64-linux-gnu -lz -ljpeg -lwebp -lpng \
    -o stable-diffusion-cli
```

### 运行的 API 调用路径
```
main()
  └── sd_ctx_params_init()           ← 初始化上下文参数
  └── new_sd_ctx()                   ← 创建 SD 上下文
        └── StableDiffusionGGML::init()  ← 加载模型 + CLIP + VAE
  └── sd_img_gen_params_init()       ← 初始化生成参数
  └── generate_image()               ← 生成图像
        ├── prepare_image_generation_latents()   ← 创建噪声
        ├── prepare_image_generation_embeds()     ← CLIP 编码
        └── StableDiffusionGGML::sample()        ← 采样循环
              ├── Karras sigmas
              ├── Cond UNet forward
              ├── Uncond UNet forward
              ├── CFG: uncond + scale*(cond - uncond)
              └── Euler step
  └── free_sd_ctx()                  ← 释放上下文
```

### 用到的 stable-diffusion.cpp API
| API | 头文件 | 作用 |
|-----|--------|------|
| `sd_ctx_params_init()` | `stable-diffusion.h` | 初始化上下文参数结构体 |
| `new_sd_ctx()` | `stable-diffusion.h` | 创建推理上下文（加载模型） |
| `sd_img_gen_params_init()` | `stable-diffusion.h` | 初始化生成参数结构体 |
| `generate_image()` | `stable-diffusion.h` | 执行 txt2img 管线 |
| `free_sd_ctx()` | `stable-diffusion.h` | 释放上下文 |
| `sd_image_t` | `stable-diffusion.h` | 图像数据结构体 |

### 第三方库依赖
| 库 | 作用 | 来源 |
|----|------|------|
| libstable-diffusion.a | stable-diffusion.cpp 静态库 | /opt/stable-diffusion.cpp/build |
| libggml.a / libggml-base.a / libggml-cpu.a | GGML 张量计算库 | stable-diffusion.cpp 自带 |
| libggml-cuda.a | GGML CUDA 后端 | stable-diffusion.cpp 自带 |
| libcudart.so / libcublas.so / libcublasLt.so | CUDA 运行时 + cuBLAS | CUDA 12 |
| libcuda.so | CUDA 驱动 API | CUDA 12 |
| libculibos.a | CUDA 设备库 | CUDA 12 |
| libz / libjpeg / libwebp / libpng | 图像 I/O | 系统 |

### 运行
```bash
LD_LIBRARY_PATH=/data/cuda/targets/x86_64-linux/lib \
  ./stable-diffusion-cli_v1 -p "prompt" -n "negative" --steps 20 --cfg 7 -s 42
```

### 输出
```
/tmp/sd_cli_output.ppm (PPM 格式, 1024×1024, ~12秒)
```

### 与 Python 参考的对比
| 指标 | sdxl_pipeline.py (ComfyUI) | img.sh (stable-diffusion.cpp) | stable-diffusion-cli.cpp |
|------|---------------------------|-------------------------------|--------------------------|
| 后端 | PyTorch + ComfyUI | GGML + stable-diffusion.cpp | stable-diffusion.cpp C API |
| 速度 | ~6秒/20步 | ~12秒/20步 | ~12秒/20步 |
| 输出 | ~1.3MB PNG | ~2MB PNG | ~3MB PPM |
| 依赖 | Python, PyTorch | 独立 ELF | 独立 ELF |
