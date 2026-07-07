# stable-diffusion.cpp 代码架构报告

> 本报告基于 `/opt/sd`（`stable-diffusion.cpp`）仓库，commit `bb84971`（`refactor: move model-specific args into model parsers (#1757)`），通过源码、`include/stable-diffusion.h` 公开 C API 以及代码缓存索引 `/opt/code_caches/sd_cache/` 进行整理。

---

## 1. 项目概览

`stable-diffusion.cpp`（又称 `sd.cpp`）是 leejet 等人发起的纯 C/C++ 扩散模型推理项目，底层依赖 [ggml](https://github.com/ggml-org/ggml)，目标与 `llama.cpp` 类似：在本地 CPU/GPU 上零 Python 依赖地运行 Stable Diffusion、SDXL、SD3、FLUX、FLUX.2、Wan、LTX 等文生图/图生图/视频模型。

| 项目信息 | 说明 |
|----------|------|
| 仓库路径 | `/opt/sd` |
| 当前 commit | `bb84971` / `bb84971129d2a094ab8051c6feed5406d3b4409d` |
| 主要语言 | C++17 / C11 |
| 公开 API | `include/stable-diffusion.h`（纯 C） |
| 依赖 | `ggml` 子模块、`thirdparty`（zip 等）、可选 `libwebp`/`libwebm` |
| 跟踪文件数 | 约 344 个文件（其中 `.c/.cpp/.h/.hpp` 约 161 个，`.md` 约 48 个） |
| `src` 目录大小 | 约 236 MB |

支持的模型族（参见 `src/model.h:13` 的 `SDVersion` 枚举）：

- SD 1.x / 2.x / Inpaint / Instruct-Pix2Pix / Tiny UNet
- SDXL / SDXL Inpaint / SDXL Instruct-Pix2Pix / SDXL-Turbo
- SD3 / SD3.5
- FLUX.1-dev/schnell / FLUX.2-dev/klein / FLUX Fill / FLUX Control
- 图像模型：Lens、Chroma、Qwen Image、PiD、LongCat、Z-Image、MiniT2I、Ovis、Anima、Ernie、Boogu、Krea2、SeFi、Ideogram4、HiDream-O1 等
- 视频模型：Wan2.1/Wan2.2、LTX-2.3
- 辅助：PhotoMaker、ControlNet、LoRA、ESRGAN、TAESD

支持的后端：

- CPU（AVX / AVX2 / AVX512）
- CUDA（NVIDIA）
- HIPBLAS（AMD ROCm）
- Metal（Apple）
- Vulkan
- OpenCL（Adreno 为主）
- SYCL（Intel）
- MUSA（摩尔线程）

支持的权重格式：PyTorch checkpoint（`.ckpt` / `.pth` / `.pt`）、Safetensors（`.safetensors`）、GGUF（`.gguf`），以及 Diffusers 目录结构。

---

## 2. 目录结构与模块划分

`/opt/sd` 的顶层目录如下：

```
/opt/sd
├── .github/              # CI / issue 模板
├── assets/               # Logo 等
├── cmake/                # CMake 配置模板
├── docker/               # Docker 构建文件
├── docs/                 # 文档（build.md、sd.md、flux.md 等）
├── examples/             # 可执行示例
│   ├── cli/              # 命令行工具 sd-cli
│   ├── server/           # HTTP 服务 sd-server
│   └── common/           # CLI / Server 共用代码（参数解析、媒体 IO）
├── ggml/                 # ggml 子模块
├── include/              # 公开 C API 头文件
│   └── stable-diffusion.h
├── scripts/              # 辅助脚本
├── src/                  # 核心实现
│   ├── conditioning/     # 文本条件 / Embedding 注入
│   ├── core/             # GGML 扩展、backend 封装、张量工具、RNG
│   ├── extensions/       # PhotoMaker、PuLID 等扩展
│   ├── model/            # 模型结构定义
│   │   ├── diffusion/    # UNet / MMDiT / DiT / Flux 等扩散模型
│   │   ├── te/           # CLIP / T5 / LLM 文本编码器
│   │   ├── vae/          # AutoEncoderKL / TAESD / Wan / LTX VAE
│   │   └── upscaler/     # ESRGAN / LTX 潜在上采样
│   ├── model_io/         # safetensors / gguf / pickle / torch_zip 读取
│   ├── runtime/          # 采样器、去噪器、scheduler、缓存、预处理
│   ├── tokenizers/       # BPE / CLIP / T5 / Gemma / Qwen 等分词器
│   ├── model_loader.h/.cpp
│   ├── model_manager.h/.cpp
│   ├── stable-diffusion.cpp
│   └── upscaler.cpp
├── thirdparty/           # zip 等第三方代码
├── CMakeLists.txt
└── README.md
```

核心模块对应关系：

| 模块 | 主要文件 | 职责 |
|------|----------|------|
| 公开 C API | `include/stable-diffusion.h` | 对外暴露的 C 接口与结构体 |
| 主入口 | `src/stable-diffusion.cpp` | `new_sd_ctx`、`generate_image`、`generate_video` 等 API 实现；`StableDiffusionGGML` 主类 |
| 模型加载 | `src/model_loader.h/.cpp` | 多格式文件解析、张量元信息整理 |
| 权重管理 | `src/model_manager.h/.cpp` | 张量驻留模式、params backend / compute backend、LoRA 应用 |
| 文本条件 | `src/conditioning/conditioner.hpp` | CLIP 编码器、Embedding 注入、权重文本解析 |
| 分词器 | `src/tokenizers/` | BPE、CLIP、T5、Gemma、Qwen 等 |
| 扩散模型 | `src/model/diffusion/` | UNet、MMDiT、DiT、Flux、Wan、LTX 等 |
| VAE | `src/model/vae/` | 解码潜在向量到图像 |
| 采样器 | `src/runtime/denoiser.hpp` | sigma 调度、K-diffusion 采样 |
| GGML 扩展 | `src/core/` | `ggml_extend.hpp`、backend 初始化、graph cut |

---

## 3. 核心架构与数据流

一个典型的 SDXL txt2img 流程可以概括为：

```
[用户 prompt / negative prompt]
        |
        v
[CLIP 分词器] --BPE + 权重--> [token ids + weights]
        |
        v
[CLIP 文本编码器] (CLIP-L + OpenCLIP-G)
        |
        +---> [cond embeddings]  (正向文本条件)
        +---> [uncond embeddings]  (空负向条件)
        |
        v
[采样 scheduler] 生成 sigma 序列
        |
        v
[初始潜在噪声] randn_like(latent_shape)
        |
        v
[UNet / DiffusionModel] 多步采样去噪
        |   输入：x_t + timestep + context + y(pooled)
        |   输出：预测噪声 / 速度 v / flow
        |
        v
[最终潜在 latent]
        |
        v
[VAE 解码器] 将 latent 解码为 RGB 图像
        |
        v
[后处理/量化] -> sd_image_t {width, height, channel, data}
        |
        v
[free_sd_images / 保存文件]
```

关键类层次：

- `StableDiffusionGGML`（`src/stable-diffusion.cpp:187`）：主控制器，持有 `ModelManager`、`Conditioner`、`DiffusionModelRunner`、`VAE`、采样器、RNG 等。
- `ModelManager`（`src/model_manager.h:15`）：负责张量注册、加载到 params backend、stage 到 compute backend、LoRA 合并。
- `ModelLoader`（`src/model_loader.h:32`）：从文件解析出 `TensorStorage` 列表，并识别 `SDVersion`。
- `DiffusionModelRunner`（`src/model/diffusion/model.hpp:94`）：抽象基类，定义 `compute(...)` 接口。
- `UNetModelRunner`（`src/model/diffusion/unet.hpp:692`）：传统 UNet 实现，用于 SD1.x/2.x/SDXL。
- `MMDiTRunner`（`src/model/diffusion/mmdit.hpp`）：用于 SD3 / Flux 等 DiT 架构。
- `VAE` / `AutoEncoderKL`（`src/model/vae/vae.hpp`、`src/model/vae/auto_encoder_kl.hpp`）：VAE 编码/解码包装。
- `Conditioner`（`src/conditioning/conditioner.hpp`）：文本到条件的完整链路。

---

## 4. 关键组件

### 4.1 模型加载

`ModelLoader` 统一支持以下格式（`src/model_loader.h:32`，`src/model_loader.cpp`）：

| 格式 | 入口函数 | 说明 |
|------|----------|------|
| GGUF | `init_from_gguf_file` | 读取 `gguf_context` 与 `ggml_context` 元数据 |
| Safetensors | `init_from_safetensors_file` | 通过 `safe_read` 解析 header 与张量偏移 |
| Torch zip | `init_from_torch_zip_file` | 基于 `thirdparty/zip` 解包 `.pth/.ckpt` |
| Torch legacy | `init_from_torch_legacy_file` | 旧版 pickle 协议 |
| Diffusers | `init_from_diffusers_file` | 读取 `unet/`、`vae/`、`text_encoder/` 子目录 |

`TensorStorage`（`src/model_io/tensor_storage.h`）保存每个张量的：名称、GGML 类型、形状、数据文件偏移、所属文件索引。加载后按统一前缀映射到 `String2TensorStorage`（`src/model.h:287`），供后续模型构造使用。

### 4.2 CLIP 分词器与文本编码

- `BPETokenizer`（`src/tokenizers/bpe_tokenizer.h/.cpp`）：基础 BPE 实现，支持 webui 风格的 token 权重（`()`、`[]`、`:`）。
- `CLIPTokenizer`（`src/tokenizers/clip_tokenizer.h/.cpp`）：继承 `BPETokenizer`，加载 CLIP merges，添加 `<|startoftext|>` / `<|endoftext|>` 特殊 token。
- `CLIPTextModel`（`src/model/te/clip.hpp:250`）：实现 CLIP 文本编码器，支持 `OPENAI_CLIP_VIT_L_14` 与 `OPEN_CLIP_VIT_H_14` / `OPEN_CLIP_VIT_BIGG_14`（SDXL 的 CLIP-G）。
- `CLIPTextModelRunner`（`src/model/te/clip.hpp:466`）：包装为 `GGMLRunner`，可返回最后一层或倒数 `clip_skip` 层隐藏态，以及 pooled 投影。
- `FrozenCLIPEmbedderWithCustomWords`（`src/conditioning/conditioner.hpp:141`）：支持自定义 Embedding（ textual inversion ）注入和权重文本。

### 4.3 UNet / Diffusion Model

- `DiffusionModelRunner`（`src/model/diffusion/model.hpp:94`）：所有扩散模型的基类，核心方法：
  ```cpp
  virtual sd::Tensor<float> compute(int n_threads, const DiffusionParams& diffusion_params) = 0;
  ```
- `UNetModelRunner`（`src/model/diffusion/unet.hpp:692`）：构建 `UNetModel` 图，输入 `x`、`timesteps`、`context`、`c_concat`、`y`，输出预测值。用于 SD1.x/2.x/SDXL/SVD 等 UNet 架构。
- `MMDiTRunner`（`src/model/diffusion/mmdit.hpp`）：用于 SD3 / Flux 的 MMDiT（Multimodal Diffusion Transformer）架构。
- Flux / Flux2 / Wan / LTX / Qwen / MiniT2I 等各自有独立的 `*Runner` 子类，位于 `src/model/diffusion/`。

`DiffusionParams` 结构（`src/model/diffusion/model.hpp:70`）包含 `x`、`timesteps`、`context`、`c_concat`、`y`、`ref_latents` 及 `extra`（如 `UNetDiffusionExtra` 的 ControlNet 控制）。

### 4.4 VAE

- `VAE`（`src/model/vae/vae.hpp:8`）：VAE 基类，提供 `encode()` / `decode()` 与 tiling 参数设置。
- `AutoEncoderKL`（`src/model/vae/auto_encoder_kl.hpp`）：标准 SD 1.x/2.x/SDXL/SD3 VAE，包含 `Encoder` 和 `Decoder`。
- `TAESD`（`src/model/vae/tae.hpp`）：Tiny AutoEncoder，用于快速预览和低内存解码。
- `WanVAE`（`src/model/vae/wan_vae.hpp`）、`LTXVideoVAE`（`src/model/vae/ltx_vae.hpp`）：视频 VAE。
- `VAE` tiling 支持空间切片（`tile_size_x`、`tile_size_y`、`target_overlap`），用于大分辨率图解码时降低显存占用。

### 4.5 采样器与调度器

采样逻辑集中在 `src/runtime/denoiser.hpp`：

- `Denoiser` / `CompVisDenoiser` / `EDMVDenoiser` / `EDMDenoiser` / `FlowDenoiser` 等：将模型的原始输出转换为去噪信号。
- `SigmaScheduler` 及子类：
  - `DiscreteScheduler`（离散线性）
  - `KarrasScheduler`（Karras 边界）
  - `ExponentialScheduler`、`SGMUniformScheduler`、`LCMScheduler`、`BetaScheduler`、`AYSScheduler`、`GITSScheduler`、`LTX2Scheduler`、`FLUXScheduler`、`FLUX2Scheduler`、`LogitNormalScheduler` 等
- `sample_k_diffusion`（`src/runtime/denoiser.hpp:2663`）：根据 `sample_method_t` 分发 Euler、Euler-A、Heun、DPM2、DPM++2M、DPM++2S a、LCM、TCD、ER-SDE 等采样器。
- `StableDiffusionGGML::sample`（`src/stable-diffusion.cpp:2184`）：将 UNet 调用、CFG、ControlNet、ref_latents、cache 等封装为单次采样。

### 4.6 随机数与图像预处理

- `RNG` / `PhiloxRNG` / `MT19937RNG`（`src/core/rng.hpp`、`rng_philox.hpp`、`rng_mt19937.hpp`）：支持 `CUDA_RNG`（默认 webui 风格）和 `CPU_RNG`（ComfyUI 风格）。
- `preprocessing_tensor_to_sd_image` / `tensor_to_sd_image`（`src/core/util.cpp:678`）：将 `[H, W, C]` 的 float 张量量化为 `uint8_t` 的 `sd_image_t`。
- `examples/common/media_io.cpp`：PNG / JPEG / WebP / PPM 等文件读写，以及 PNG 元数据嵌入（webui 参数格式）。

---

## 5. SDXL 推理路径

以 `comfycli` / `sd-cli` 常见的 SDXL txt2img 命令为例，其内部代码路径如下：

1. **上下文创建**
   - 用户通过 `sd_ctx_params_t` 设置：`model_path`（SDXL base）、`clip_l_path`、`clip_g_path`、`vae_path`（可选）、`wtype`、`n_threads`、`rng_type`、`flash_attn` 等。
   - 调用 `new_sd_ctx(&ctx_params)`（`src/stable-diffusion.cpp:3337`）构造 `StableDiffusionGGML` 并执行 `init()`。

2. **模型加载**
   - `init()` 中 `ModelLoader` 依次读取 base、clip_l、clip_g、vae，随后 `convert_tensors_name()` 统一 key 名。
   - 通过 `model_loader.get_sd_version()` 识别为 `VERSION_SDXL`（`src/model.h:23`）。
   - 根据 `SDVersion` 创建 `FrozenCLIPEmbedderWithCustomWords` 和 `UNetModelRunner`（`src/stable-diffusion.cpp:652` 附近）。

3. **文本条件**
   - `prepare_image_generation_embeds`（`src/stable-diffusion.cpp:4582`）调用 `cond_stage_model->encode(...)`。
   - SDXL 使用两套 CLIP：OpenAI CLIP-L（`cond_stage_model.transformer.`）和 OpenCLIP-G（`cond_stage_model.1.transformer.`）。
   - 正向 prompt 生成 `cond`，空字符串生成 `uncond`；两者均包含 `hidden_states` 和 `pooled_projection`（`y`）。

4. **潜在初始化**
   - `prepare_image_generation_latents`（`src/stable-diffusion.cpp:4311`）按 `width/8`、`height/8` 生成随机噪声 latent，并分配 `control_image`（若启用 ControlNet）。

5. **采样去噪**
   - `StableDiffusionGGML::sample`（`src/stable-diffusion.cpp:2184`）按 `sample_method` 和 `scheduler` 构造 sigma 序列。
   - 默认 SDXL 常用 `EULER_A_SAMPLE_METHOD` + `KARRAS_SCHEDULER`。
   - 每步调用 `diffusion_model->compute(...)` 得到噪声预测；CFG 通过 `guidance.txt_cfg` 对 `cond` 与 `uncond` 做线性外推。

6. **VAE 解码**
   - 去噪完成后得到 `final_latents`，`decode_image_outputs`（`src/stable-diffusion.cpp:4674`）调用 `first_stage_model->decode()`。
   - 输出被裁切到 `[-1, 1]` 后映射到 `[0, 255]`，生成 `sd_image_t` 数组。

7. **输出**
   - `generate_image` 返回 `sd_image_t** images_out` 与 `num_images_out`（`include/stable-diffusion.h:466`）。
   - 调用者使用 `free_sd_images(...)` 释放内存，避免跨 CRT 问题（`include/stable-diffusion.h:549`）。

关键参数：

| 参数 | 典型值 | 作用 |
|------|--------|------|
| `width` / `height` | 1024×1024 | 输出尺寸，SDXL latent 为 128×128×4 |
| `sample_steps` | 20~30 | 去噪迭代次数 |
| `txt_cfg` | 7.0 | 无分类器引导强度 |
| `sample_method` | `EULER_A_SAMPLE_METHOD` | 采样算法 |
| `scheduler` | `KARRAS_SCHEDULER` | sigma 调度方式 |
| `seed` | 任意 | 噪声随机种子 |
| `clip_skip` | -1（默认） | 使用 CLIP 第几层隐藏态 |

---

## 6. 公开 C API 详细说明

`include/stable-diffusion.h` 是唯一的对外接口，所有类型、枚举、函数均使用 `SD_API` 宏导出。主要结构体：

| 结构体 | 关键字段 | 说明 |
|--------|----------|------|
| `sd_ctx_params_t` | `model_path`, `clip_l_path`, `clip_g_path`, `t5xxl_path`, `vae_path`, `control_net_path`, `wtype`, `n_threads`, `rng_type`, `flash_attn`, `max_vram`, `backend`, `params_backend`, `split_mode`, `enable_mmap`, `lora_apply_mode` 等 | 创建上下文所需的全局参数（`include/stable-diffusion.h:186`） |
| `sd_image_t` | `width`, `height`, `channel`, `uint8_t* data` | 图像数据块，由库内 `malloc` 分配（`include/stable-diffusion.h:239`） |
| `sd_sample_params_t` | `guidance`, `scheduler`, `sample_method`, `sample_steps`, `eta`, `custom_sigmas`, `flow_shift` 等 | 采样参数（`include/stable-diffusion.h:261`） |
| `sd_img_gen_params_t` | `prompt`, `negative_prompt`, `width`, `height`, `seed`, `batch_count`, `init_image`, `mask_image`, `control_image`, `control_strength`, `loras`, `hires`, `vae_tiling_params`, `cache` 等 | 图像生成参数（`include/stable-diffusion.h:357`） |
| `sd_vid_gen_params_t` | 类似图像参数，额外包含 `video_frames`, `fps`, `control_frames` 等 | 视频生成参数（`include/stable-diffusion.h:387`） |
| `sd_lora_t` | `is_high_noise`, `multiplier`, `path` | 单个 LoRA（`include/stable-diffusion.h:323`） |
| `sd_hires_params_t` | `enabled`, `upscaler`, `scale`, `target_width`, `target_height`, `steps`, `denoising_strength` | 高清修复（`include/stable-diffusion.h:343`） |
| `sd_tiling_params_t` | `enabled`, `tile_size_x`, `tile_size_y`, `target_overlap`, `rel_size_x`, `rel_size_y` | VAE tiling（`include/stable-diffusion.h:162`） |
| `sd_cache_params_t` | `mode`（`SD_CACHE_EASYCACHE` 等）及阈值参数 | 采样加速缓存（`include/stable-diffusion.h:296`） |

核心 API：

```c
// 初始化参数结构体
void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params);

// 创建/释放上下文
sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params);
void free_sd_ctx(sd_ctx_t* sd_ctx);

// 图像生成（当前签名）
bool generate_image(sd_ctx_t* sd_ctx,
                    const sd_img_gen_params_t* sd_img_gen_params,
                    sd_image_t** images_out,
                    int* num_images_out);

// 视频生成
bool generate_video(sd_ctx_t* sd_ctx,
                    const sd_vid_gen_params_t* sd_vid_gen_params,
                    sd_image_t** frames_out,
                    int* num_frames_out,
                    sd_audio_t** audio_out);

// 上采样
upscaler_ctx_t* new_upscaler_ctx(...);
bool upscale(upscaler_ctx_t* upscaler_ctx, sd_image_t input_image,
             uint32_t upscale_factor, sd_image_t** images_out, int* num_images_out);

// 模型转换
bool convert(const char* input_path, const char* vae_path, const char* output_path,
             enum sd_type_t output_type, const char* tensor_type_rules, bool convert_name);
bool convert_with_components(...);

// 回调与工具
void sd_set_log_callback(sd_log_cb_t cb, void* data);
void sd_set_progress_callback(sd_progress_cb_t cb, void* data);
void sd_set_preview_callback(sd_preview_cb_t cb, enum preview_t mode, ...);
void sd_cancel_generation(sd_ctx_t* sd_ctx, enum sd_cancel_mode_t mode);
void free_sd_images(sd_image_t* result_images, int num_images);
size_t sd_list_devices(char* buffer, size_t buffer_size);
const char* sd_get_system_info();
const char* sd_commit();
const char* sd_version();
```

**内存所有权**：`generate_image` 返回的 `images_out` 必须由调用者通过 `free_sd_images` 释放。`free_sd_images` 会释放每个 `sd_image_t.data` 和数组本身，从而避免 Windows 上跨 CRT 释放问题（`include/stable-diffusion.h:548`）。

---

## 7. 模型格式与加载

`ModelLoader` 通过文件魔数识别格式，并在 `init_from_file` 中分派（`src/model_loader.h:53`）：

| 文件扩展 | 实际解析 | 说明 |
|----------|----------|------|
| `.gguf` | `read_gguf_file`（`src/model_io/gguf_io.cpp:42`） | 使用 `gguf_init_from_file` 或直接 `GGUFReader` |
| `.safetensors` | `read_safetensors_file`（`src/model_io/safetensors_io.cpp`） | JSON header + 原始张量数据 |
| `.ckpt` / `.pth` / `.pt` | `torch_zip_io.cpp` / `torch_legacy_io.cpp` / `pickle_io.cpp` | zip 包或 pickle 存储 |
| Diffusers 目录 | `init_from_diffusers_file`（`src/model_loader.cpp:404`） | 读取 `unet/diffusion_pytorch_model.safetensors` 等 |

张量名转换：

- `model_loader.convert_tensors_name()` 调用 `name_conversion.cpp` 中的规则，将 PyTorch / Diffusers 命名映射到项目内部统一命名（如 `model.diffusion_model.*`、`cond_stage_model.transformer.*`、`vae.*`）。
- 不同 `SDVersion` 对命名前缀有特定要求（如 SDXL 的 CLIP-G 前缀为 `cond_stage_model.1.transformer.`，见 `src/stable-diffusion.cpp:662`）。

量化与张量类型：

- `sd_type_t` 枚举与 `ggml_type` 一一对应（`include/stable-diffusion.h:93`）。
- `model_loader.set_wtype_override(...)` 可将权重覆盖到指定量化类型（如 `SD_TYPE_Q4_0`）。
- `tensor_type_rules` 支持按正则匹配指定某些层保持 FP16/FP32，其余层量化。

`ModelManager` 加载流程：

1. `register_param_tensors` / `register_runner_params`：模型将自身 `params` 字典注册到 `ModelManager`。
2. `validate_registered_tensors`：检查每个张量是否存在、形状是否匹配。
3. 根据 `ResidencyMode`（`Disk` 或 `ParamBackend`）与 backend 分配，将张量加载到 `params_backend` 的 buffer。
4. 在计算前 `prepare_params` 将张量 stage 到 `compute_backend`。
5. 若启用 `enable_mmap`，则通过 `mmap` 将文件直接映射到 backend buffer。

---

## 8. 后端与硬件加速

`stable-diffusion.cpp` 通过 ggml 的 backend 系统支持异构计算。后端管理位于 `src/core/ggml_extend_backend.cpp` 和 `src/core/ggml_extend.hpp`。

### 8.1 后端初始化

- `sd_get_default_backend()`（`src/core/ggml_extend_backend.cpp:533`）优先使用用户指定的 `--backend`，否则按编译选项选择 CUDA/HIP/Metal/Vulkan/CPU。
- `StableDiffusionGGML::init_backend()` 为不同模块（`TE`、`DIFFUSION`、`VAE`、`CONTROL_NET`、`UPSCALER`）分别建立 runtime backend 和 params backend。

### 8.2 后端类型

| CMake 选项 | 说明 |
|------------|------|
| `-DSD_CUDA=ON` | NVIDIA CUDA，推荐 4GB+ VRAM |
| `-DSD_HIPBLAS=ON` | AMD ROCm |
| `-DSD_METAL=ON` | Apple Metal |
| `-DSD_VULKAN=ON` | Vulkan SDK |
| `-DSD_OPENCL=ON` | OpenCL（Adreno 为主） |
| `-DSD_SYCL=ON` | Intel SYCL |
| `-DSD_MUSA=ON` | 摩尔线程 MUSA |
| `-DGGML_OPENBLAS=ON` | CPU 端 OpenBLAS |

### 8.3 显存与内存管理

- `max_vram`（`sd_ctx_params_t` 的字符串）：指定以 GiB 为单位的 VRAM 预算或后端分配规则。启用后 `sd::backend_fit::derive_backend_specs` 会决定哪些模块放到 CPU/其他设备。
- `auto_fit`：自动根据预算切分计算图。
- `ggml_graph_cut`（`src/core/ggml_graph_cut.h/.cpp`）：在图执行中插入切分点，将中间结果驻留到 CPU 或磁盘，实现参数卸载/流水线。
- `stream_layers`：在 `--max-vram` 基础上启用参数预取流。
- `eager_load`：模型加载时就把所有参数加载到 params backend，而非首次使用时懒加载。
- `params_backend` / `backend`：可分别指定参数存储后端和计算后端，例如参数放 CPU、计算在 CUDA。
- `split_mode`：支持 `layer` 或 `row` 切分，以及模块级分配如 `diffusion=row`。

### 8.4 Flash Attention

`ggml_ext_attention_ext`（`src/core/ggml_extend.hpp:1315`）根据 `flash_attn` 标志与 mask 维度判断是否可调用 Flash Attention 内核；若不可行则回退到常规 reshape + matmul 实现。在 CUDA/Vulkan 等后端开启 `flash_attn` 可显著降低大上下文显存占用。

---

## 9. 扩展机制

### 9.1 LoRA

`LoraModel`（`src/model/adapter/lora.hpp`）与 `MultiLoraAdapter`（`src/model/adapter/lora.hpp`）在运行时或加载时修改 Linear/Conv2d 的权重。

- `LORA_APPLY_AUTO`：`StableDiffusionGGML::init()` 中根据是否存在量化权重、params 是否卸载、是否启用 row split 自动决定立即合并还是运行时合并（`src/stable-diffusion.cpp:815`）。
- `LORA_APPLY_IMMEDIATELY`：在模型加载时通过 `ModelManager::apply_loras_to_params` 将 LoRA 合并到参数 buffer。
- `LORA_APPLY_AT_RUNTIME`：在每次 Linear/Conv2d 前向传播时通过 `MultiLoraAdapter::forward_with_lora` 动态计算 `out_diff` 并相加（`src/model/adapter/lora.hpp:925`）。

### 9.2 ControlNet

`ControlNet`（`src/model/diffusion/control.hpp:311`）读取 control_net 权重，在 UNet 各层下采样处注入控制特征。`ControlNet::compute` 接受 `hint`（如 canny 边缘图）并返回多尺度 `controls`，最终由 `UNetModelRunner::compute` 中的 `control_strength` 加权。

### 9.3 HiRes Fix

`generate_image` 中若 `sd_hires_params_t.enabled` 为真，则先完成一次低分辨率采样，然后：

1. 按 `upscaler` 选择上采样方式（latent 最近邻/bicubic/Lanczos/ESRGAN 模型）。
2. 将 latent 放大到目标尺寸。
3. 根据 `denoising_strength` 裁剪新的 sigma 序列。
4. 再次调用 `StableDiffusionGGML::sample` 做二次去噪（`src/stable-diffusion.cpp:5065`）。

### 9.4 VAE Tiling

`VAE::set_tiling_params`（`src/model/vae/vae.hpp:229`）将用户参数转换为实际 latent tile 大小，在 `encode`/`decode` 中对大图像分块处理，降低单步显存峰值。`LTXVideoVAE` 额外支持时序切片（`temporal_tiling`）。

### 9.5 其他扩展

- **PhotoMaker**：`src/extensions/photomaker_extension.cpp` 通过 ID 图像修改 CLIP 条件。
- **PuLID**：`src/extensions/pulid_extension.cpp` 提供人脸 ID 保持能力。
- **TAESD**：低延迟预览和快速解码（`src/model/vae/tae.hpp`）。
- **ESRGAN**：独立上采样模型（`src/model/upscaler/esrgan.hpp`）。
- **采样缓存**：`sd_cache_params_t` 支持 `EASYCACHE`、`UCACHE`、`DBCache`、`TAYLORSEER`、`SPECTRUM` 等策略，跳过相邻步骤间相似度高的计算。

> 注：当前 `/opt/sd` 主干中并未直接内置 FreeU、Self-Attention Guidance（SAG）或 IP-Adapter 的完整实现；这些功能如要在 ComfyUI 等价工作流中使用，通常需要在上层适配器（如本项目的 `sdcpp_adapter`）或扩展模块中自行实现。

---

## 10. CLI 与 Server

### 10.1 CLI（`examples/cli`）

`examples/cli/main.cpp` 实现 `sd-cli` 可执行文件，支持以下运行模式：

```
img_gen, vid_gen, convert, upscale, metadata
```

典型命令：

```bash
# txt2img
./bin/sd-cli -m sd_xl_base_1.0.safetensors \
             --clip-l clip_l.safetensors \
             --clip-g clip_g.safetensors \
             -p "a lovely cat" -n "blurry" \
             -o output.png -W 1024 -H 1024 --steps 20 --cfg 7.0 -s 42

# 模型格式转换
./bin/sd-cli -M convert -m model.safetensors -o model.gguf --type q4_0

# 元数据查看
./bin/sd-cli -M metadata --image output.png
```

参数解析：

- `SDCliParams` 定义 CLI 通用选项（输出、模式、预览、metadata 等）。
- `SDContextParams`（`examples/common/common.h:116`）映射到 `sd_ctx_params_t`。
- `SDGenerationParams`（`examples/common/common.h:185`）映射到 `sd_img_gen_params_t`。
- 三者的 `get_options()` 通过 `parse_options` 统一解析命令行（`examples/common/common.h:109`）。

### 10.2 Server（`examples/server`）

`examples/server/main.cpp` 基于 `httplib.h` 构建 HTTP 服务，注册四类路由：

| 路由文件 | 说明 |
|----------|------|
| `routes_index.cpp` | 根页面 `/` 与静态资源 |
| `routes_openai.cpp` | OpenAI 兼容接口（`/v1/images/generations` 等） |
| `routes_sdapi.cpp` | Stable Diffusion WebUI API 风格接口 |
| `routes_sdcpp.cpp` | 本项目自定义接口 |

运行时结构：

- `ServerRuntime`（`examples/server/runtime.h`）持有 `sd_ctx_t*`、LoRA 缓存、Upscaler 缓存、`AsyncJobManager`。
- `AsyncJobManager` 维护后台任务队列，避免长请求阻塞 HTTP 线程。
- 通过 `sd_set_log_callback` 将库日志重定向到服务端日志输出。

典型启动：

```bash
./bin/sd-server -m model.safetensors --listen 0.0.0.0 --port 8080
```

---

## 11. 构建与部署

### 11.1 基本构建

```bash
cd /opt/sd
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=ON
cmake --build . -j$(nproc)
```

产物：

- `build/bin/sd-cli`
- `build/bin/sd-server`
- `build/bin/libstable-diffusion.a`（或 `.so` 若开启 `SD_BUILD_SHARED_LIBS`）

### 11.2 CUDA 构建

```bash
mkdir build-cuda && cd build-cuda
cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DSD_BUILD_EXAMPLES=ON
cmake --build . -j$(nproc)
```

### 11.3 常用 CMake 选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `SD_BUILD_EXAMPLES` | `ON`（standalone） | 构建 `sd-cli` 和 `sd-server` |
| `SD_WEBP` / `SD_WEBM` | 依赖子模块 | 启用 WebP/WebM 图像/视频 IO |
| `SD_USE_SYSTEM_WEBP` / `SD_USE_SYSTEM_WEBM` | `OFF` | 使用系统 libwebp/libwebm |
| `SD_CUDA` | `OFF` | CUDA 后端 |
| `SD_HIPBLAS` | `OFF` | AMD ROCm 后端 |
| `SD_METAL` | `OFF` | Apple Metal 后端 |
| `SD_VULKAN` | `OFF` | Vulkan 后端 |
| `SD_OPENCL` | `OFF` | OpenCL 后端 |
| `SD_SYCL` | `OFF` | Intel SYCL 后端 |
| `SD_MUSA` | `OFF` | MUSA 后端 |
| `SD_BUILD_SHARED_LIBS` | `OFF` | 构建动态库 |
| `SD_BUILD_SHARED_GGML_LIB` | `OFF` | 将 ggml 也构建为动态库 |
| `SD_USE_SYSTEM_GGML` | `OFF` | 使用系统已安装的 ggml |
| `SD_RPC` | `OFF` | 启用 RPC 后端 |

### 11.4 部署注意事项

- 运行依赖：CUDA 构建产物需要 `libcudart.so` / `libcublas.so` / `libcublasLt.so` / `libcuda.so`（stub 或真实驱动）。
- 若使用 `--max-vram` 或 `--params-backend CPU`，需要确保参数后端 buffer 可用内存。
- 发布二进制时通常需保留 `ggml` 后端共享库（如果 `SD_BUILD_SHARED_GGML_LIB=ON`）和 `libwebp`/`libwebm` 依赖。

---

## 12. v1 C++ CLI 代码调用流程与 API 定位

### 12.1 v1 CLI 源码结构

`/opt/static_comfyui/cpp/stable-diffusion-cli_v1.cpp` 是一个最小化的 SDXL txt2img 示例，其主流程如下：

1. 命令行解析，收集 `model`、`clip_l`、`clip_g`、`prompt`、`negative_prompt`、`width`、`height`、`steps`、`cfg`、`seed`、`output`。
2. 填充 `sd_ctx_params_t`：
   ```cpp
   sd_ctx_params_t ctx_params;
   sd_ctx_params_init(&ctx_params);
   ctx_params.model_path     = model;
   ctx_params.clip_l_path    = clip_l;
   ctx_params.clip_g_path    = clip_g;
   ctx_params.n_threads      = 8;
   ctx_params.wtype          = SD_TYPE_F16;
   ctx_params.rng_type       = STD_DEFAULT_RNG;
   ctx_params.keep_vae_on_cpu = false;
   ```
3. 创建上下文：
   ```cpp
   sd_ctx_t* ctx = new_sd_ctx(&ctx_params);
   ```
4. 填充 `sd_img_gen_params_t`：
   ```cpp
   sd_img_gen_params_t img_params;
   sd_img_gen_params_init(&img_params);
   img_params.prompt          = prompt;
   img_params.negative_prompt = neg;
   img_params.width           = W;
   img_params.height          = H;
   img_params.seed            = seed;
   img_params.batch_count     = 1;
   img_params.sample_params.sample_steps       = steps;
   img_params.sample_params.guidance.txt_cfg    = cfg;
   img_params.sample_params.sample_method       = EULER_SAMPLE_METHOD;
   img_params.sample_params.scheduler           = KARRAS_SCHEDULER;
   ```
5. 生成图像：
   ```cpp
   sd_image_t* images = generate_image(ctx, &img_params);
   ```
6. 保存并释放：
   ```cpp
   save_ppm(output, images[0].data, images[0].width, images[0].height, 3);
   free(images[0].data);
   free(images);
   free_sd_ctx(ctx);
   ```

### 12.2 v1 API 签名与当前 `/opt/sd` 不兼容

v1 源码编写时使用的 `generate_image` 签名是：

```cpp
sd_image_t* generate_image(sd_ctx_t* ctx, const sd_img_gen_params_t* params);
```

该签名在较新的 `/opt/sd` 中已被废弃并替换为：

```cpp
bool generate_image(sd_ctx_t* sd_ctx,
                    const sd_img_gen_params_t* sd_img_gen_params,
                    sd_image_t** images_out,
                    int* num_images_out);
```

（定义在 `include/stable-diffusion.h:466`，实现在 `src/stable-diffusion.cpp:4947`。）

主要差异：

| 旧签名 | 新签名 |
|--------|--------|
| 直接返回 `sd_image_t*` | 返回 `bool` 表示成功/失败 |
| 默认单张图 | 通过 `images_out` / `num_images_out` 输出批量结果 |
| 调用者直接 `free(images[0].data)` | 必须调用 `free_sd_images(images, num_images)` 释放 |
| 无法区分失败与空指针 | 函数返回 `false` 时 `images_out` 为 `nullptr` |

这种不兼容导致 v1 CLI 直接链接 `/opt/sd` 新库时会编译失败（或如果按旧 ABI 链接则运行时崩溃）。此外，v1 中 `rng_type = STD_DEFAULT_RNG` 在 `include/stable-diffusion.h:31` 仅作为占位符，实际有效枚举为 `CUDA_RNG` / `CPU_RNG` 等；`new_sd_ctx` 内部会根据 `rng_type` 选择 `PhiloxRNG` 或 `MT19937RNG`（`src/stable-diffusion.cpp:608`）。

### 12.3 与公开 API 的对应关系

v1 CLI 中使用的 API 在 `include/stable-diffusion.h` 中的定位如下：

| v1 调用 | 当前 API 定位 |
|--------|---------------|
| `sd_ctx_params_init` | `include/stable-diffusion.h:451` |
| `new_sd_ctx` | `include/stable-diffusion.h:454` / `src/stable-diffusion.cpp:3337` |
| `sd_img_gen_params_init` | `include/stable-diffusion.h:464` / `src/stable-diffusion.cpp:3140` 附近 |
| `generate_image` | `include/stable-diffusion.h:466` / `src/stable-diffusion.cpp:4947` |
| `free_sd_ctx` | `include/stable-diffusion.h:455` / `src/stable-diffusion.cpp:3358` |

正确迁移 v1 代码时，需要：

1. 将 `generate_image` 调用改为：
   ```cpp
   sd_image_t* images = nullptr;
   int num_images     = 0;
   bool ok = generate_image(ctx, &img_params, &images, &num_images);
   if (!ok || images == nullptr || num_images == 0) { ... }
   ```
2. 使用 `free_sd_images(images, num_images)` 替代 `free(images[0].data); free(images);`。
3. 将 `STD_DEFAULT_RNG` 替换为 `CUDA_RNG`（默认）或 `CPU_RNG`。

这正是第 13 节 v2 适配层所做的事情。

---

## 13. v2 适配层实现与验证

### 13.1 新增文件

| 文件 | 作用 |
|------|------|
| `/opt/static_comfyui/cpp/sd/src/adapters/sdcpp_adapter.h` | C++ 适配层头文件：暴露 `sd::Image`, `sd::ModelConfig`, `sd::ImageGenerationParams`, `sd::SDPipeline` |
| `/opt/static_comfyui/cpp/sd/src/adapters/sdcpp_adapter.cpp` | 唯一 `#include <stable-diffusion.h>` 的源文件，封装 `sd_ctx_t` / `generate_image` |
| `/opt/static_comfyui/cpp/sd/examples/sdxl_txt2img.cpp` | v2 CLI 示例，功能对齐 v1 的 `stable-diffusion-cli_v1.cpp`，但使用新的 API 签名 |
| `/opt/static_comfyui/cpp/sd/CMakeLists.txt` | 构建配置：编译 `sdcpp_adapter` 静态库 + `sdxl_txt2img` 可执行文件 |

### 13.2 适配层设计

- **隔离原则**：只有 `sdcpp_adapter.cpp` 接触 `/opt/sd` 的 C API；上层代码只依赖 `sdcpp_adapter.h` 中的 C++ 类型。
- **内存管理**：`generate_image` 返回的 `sd_image_t*` 在适配层内部复制到 `sd::Image::data`（`std::vector<uint8_t>`），然后立即调用 `free_sd_images` 释放原始内存，避免跨 CRT 释放问题。
- **API 签名兼容**：使用 `/opt/sd` 最新的 `bool generate_image(..., sd_image_t**, int*)` 签名，不使用 v1 中已废弃的 `sd_image_t* generate_image(...)` 签名。
- **默认参数对齐 sd-cli**：`rng_type = CUDA_RNG`、`wtype = SD_TYPE_COUNT`（auto）、`sample_method = euler_a`、`scheduler = discrete`、`steps = 20`，确保与官方 `sd-cli` 输出一致。
- **链接顺序**：参考 `/opt/sd/build/examples/cli/CMakeFiles/sd-cli.dir/link.txt` 中的顺序，确保静态库依赖正确解析：
  ```
  sdxl_txt2img.o -> sdcpp_adapter.a -> libstable-diffusion.a -> webp, webpmux -> libggml.a -> dl ->
  libggml-cpu.a -> libggml-cuda.a -> libggml-base.a -> gomp, pthread ->
  libcudart.so -> libcublas.so -> libcublasLt.so -> libculibos.a -> libcuda.so (stub) ->
  webp, sharpyuv -> libm
  ```

### 13.3 构建命令

```bash
cd /opt/static_comfyui/cpp/sd

# 1. 先编译 /opt/sd 后端（CUDA 版）
./build_sd.sh

# 2. 再编译适配层 + 示例
rm -rf build && mkdir build
cmake -S . -B build
cmake --build build -j$(nproc)

# 产物
#   build/libsdcpp_adapter.a
#   build/sdxl_txt2img
```

### 13.4 端到端验证结果

- 参考命令（sd-cli 官方 CLI，用于对比验证）：
  ```bash
  /opt/sd/build/bin/sd-cli -m /data/models/image/sd_xl_base_1.0.safetensors \
    --clip_l /data/models/image/clip_l.safetensors \
    --clip_g /data/models/image/clip_g.safetensors \
    -p "solo,single woman,half body portrait of a young woman, soft natural lighting, elegant pose, studio lighting, sharp eyes, clean white background, medium close up" \
    -n "blurry, low quality, ugly" \
    -W 1024 -H 1024 --steps 20 -s 42 \
    -o /home/quqiufeng/sdcli_1024_20.png
  ```
- v2 adapter 命令（默认 20 steps、Euler A、离散 scheduler）：
  ```bash
  ./build/sdxl_txt2img -W 1024 -H 1024 -s 42
  ```
  输出：`~/sdxl_txt2img.png`（1024×1024，3 channels，约 1.8 MB）
- 像素级对比：
  - 使用 20 steps 时，`sdxl_txt2img` 生成的 PNG 与 `sd-cli` 生成的 PNG 像素完全一致（mean diff = 0，max diff = 0）。
  - 哈希值不同仅因为 PNG 压缩参数/元数据差异，不影响图像内容。
- 说明：
  - 少于 20 steps（如 5 steps）的图像质量会明显下降，看起来"破碎/失真"，这是 SDXL 正常行为，不是 bug。
  - 建议 SDXL 推理至少使用 20 steps。

### 13.5 已知问题与后续方向

- 当前示例只支持 txt2img，尚未接入 img2img、ControlNet、LoRA、HiRes 等 `/opt/sd` 已支持的功能。
- 适配层目前仅做最小封装，尚未实现 `design.md` 中规划的 `sd::Tensor` / `sd::nn::Module` / `sd::ops` 抽象。
- 下一步可按 `design.md` Phase 1 推进：先扩展 `SDPipeline` 接口（LoRA、VAE tiling、HiRes），再逐步引入 `api/` 层。

---

**报告更新时间**：2026-07-07（补充 v1 API 分析）
**报告更新时间**：2026-07-08（补充 v2 适配层实现与端到端验证）
