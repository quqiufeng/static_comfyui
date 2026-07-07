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

### 与 Python 参考的对比
| 指标 | sdxl_pipeline.py (ComfyUI) | img.sh (stable-diffusion.cpp) |
|------|---------------------------|-------------------------------|
| 后端 | PyTorch + ComfyUI | GGML + stable-diffusion.cpp |
| 速度 | ~6秒/20步 | ~12秒/20步 |
| 输出 | ~1.3MB PNG | ~2MB PNG |
| 依赖 | Python, PyTorch, ComfyUI | 独立 ELF 二进制 |
