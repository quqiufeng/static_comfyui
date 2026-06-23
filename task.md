# static_comfyui — ComfyUI 源码逐模块 StaticPy 重写进度

任务按 `comfyui_ref/comfy/` 源码逐文件对齐。
每完成一个模块，对应的 ELF 管线（build.sh → deliver.sh）必须能编译通过。

**颜色标记**：
- `[ ]` 未开始
- `[~]` 进行中
- `[x]` 完成（ELF 编译通过 + 数值对齐 PyTorch reference）

---

## Phase 0: 基础设施

| 状态 | 模块 | 源文件 | 行数 | 说明 |
|------|------|--------|------|------|
| [x] | **编译器** | `compiler/translate.py` | 650 | StaticPy → Scheme 翻译器 |
| [x] | **值类型运行时** | `compiler/prelude.scm` | 460 | C 数组 GC, 文件 I/O, JSON, libm |
| [x] | **标准库** | `compiler/stdlib.scm` | 90 | 工具函数 |
| [x] | **build 管线** | `build.sh` | 130 | StaticPy → Chez AOT → .so |
| [x] | **deliver 管线** | `deliver.sh` | 230 | .so → ELF + lib/ 部署包 |
| [x] | **C++ 运行时** | `/opt/ReScheme/libtorch_std_helper.{h,cpp}` | 4400+ | 380+ extern "C" API (含 attention/layer_norm/rms_norm/group_norm/flux_embed_nd/arange/cos/sin/add_scalar/mul_scalar) |

---

## Phase 1: 张量基元层 (ops)

对应 `comfyui_ref/comfy/ops.py` — 所有 PyTorch 操作的抽象层。
libtorch_std_helper 已在 C++ 侧提供完整实现，StaticPy 侧需要 extern fn 声明。

| 状态 | 模块 | 源文件 | 行数 | 策略 |
|------|------|--------|------|------|
| [x] | **Tensor 基础** | `ops.py` 前半 | 800 | extern fn 声明到 libtorch_std_helper |
| [x] | **数学运算** | `ops.py` 中间 | 300 | add/sub/mul/div/exp/log/sqrt/cos/sin/arange |
| [x] | **归约 / 比较** | `ops.py` 后半 | 401 | sum/mean/max/min + eq/gt/lt/clamp |
| [x] | **函数式 nn** | `sd_runtime/nn.static.py` | 150 | timestep embedding, sinusoidal embedding, mean_flat, extract_into_tensor, beta schedules |

## Phase 2: Attention

对应 `comfyui_ref/comfy/ldm/modules/attention.py` — SD/FLUX 注意力机制。

| 状态 | 模块 | 源文件 | 行数 | 策略 |
|------|------|--------|------|------|
| [x] | **注意力核心** | `sd_runtime/attention.static.py` | 120 | torch_std_attention (C++ multi-head SDPA + skip_reshape) |
| [x] | **CrossAttention 函数式** | `sd_runtime/attention.static.py` | 30 | 线性投影 + attention + 输出投影 |
| [x] | **GEGLU / FeedForward** | `sd_runtime/attention.static.py` | 40 | chunk + gelu + 线性层 |
| [x] | **FLUX RoPE + EmbedND** | `sd_runtime/flux_attention.static.py` | 70 | torch_std_flux_embed_nd (C++, 多轴旋转位置编码) |
| [x] | **C++ 新增** | `libtorch_std_helper.{h,cpp}` | 120+ | attention, layer_norm, rms_norm, group_norm, flux_embed_nd |
| [ ] | **BasicTransformerBlock 函数式** | (待确认是否必需的独立模块) | — | C++ UNet forward 已内部处理 |
| [ ] | **SpatialTransformer 函数式** | (待确认) | — | C++ UNet forward 已内部处理 |

## Phase 3: CLIP Text Encoder

对应 `comfyui_ref/comfy/clip_model.py` + `sd1_clip.py` + `sdxl_clip.py`。
libtorch_std_helper 已有 CLIP BPE tokenizer + text forward，StaticPy 侧编排。

| 状态 | 模块 | 源文件 | 行数 | 策略 |
|------|------|--------|------|------|
| [x] | **CLIP BPE Tokenizer** | `sd_runtime/clip_tokenizer.static.py` | 55 | 包装 `torch_std_clip_tokenizer_*` (C++ BPE) |
| [x] | **CLIP 基础模型** | `sd_runtime/clip_model.static.py` | 55 | 包装 `torch_std_clip_text_forward` + `torch_std_jit_load` |
| [x] | **SD1.5 CLIP** | `sd_runtime/sd1_clip.static.py` | 110 | tokenize → forward → pooled, 状态管理 |
| [x] | **SDXL Dual CLIP** | `sd_runtime/sdxl_clip.static.py` | 80 | 双 CLIP (L+G) 管线编排 |
| [x] | **CLIP extern fn** | `sd_runtime/ops.static.py` | +25 | tokenizer/text_forward/jit_load/safetensors |
| [ ] | **CLIP Vision** | `clip_vision.py` | 163 | ViT 视觉编码器 |
| [ ] | **Long CLIP-L** | `text_encoders/long_clipl.py` | — | 长序列 CLIP |

## Phase 4: SD UNet

对应 `comfyui_ref/comfy/ldm/modules/diffusionmodules/openaimodel.py`。
libtorch_std_helper 已有 SD1.5/SDXL UNet forward，StaticPy 侧做权重管理 + 参数传递。

| 状态 | 模块 | 源文件 | 行数 | 策略 |
|------|------|--------|------|------|
| [ ] | **UNet 基础块** | `openaimodel.py` | 300 | timestep embedding, ResBlock |
| [ ] | **UNet 结构** | `openaimodel.py` | 400 | down/mid/up + skip routing |
| [ ] | **UNet forward** | `openaimodel.py` | 227 | 完整 forward 管线 |
| [ ] | **SDXL 尺寸嵌入** | `model_base.py` | — | original_size, crop, target_size |

## Phase 5: VAE

对应 `comfyui_ref/comfy/ldm/modules/diffusionmodules/model.py`。
libtorch_std_helper 已有 VAE tiled encode/decode。

| 状态 | 模块 | 源文件 | 行数 | 策略 |
|------|------|--------|------|------|
| [ ] | **Encoder** | `model.py` | 250 | down blocks + mid block |
| [ ] | **Decoder** | `model.py` | 250 | mid block + up blocks |
| [ ] | **VAE tiling** | model_management 引用 | — | 调用 `torch_std_vae_*_tiled` |
| [ ] | **TAESD** | `taesd/taesd.py, taesd/taehv.py` | — | 轻量 VAE |

## Phase 6: K-Diffusion Samplers

对应 `comfyui_ref/comfy/k_diffusion/sampling.py` (1957 行)。
libtorch_std_helper 已有 DDIM/Euler/DPM++ sampler，StaticPy 侧做 sigma 调度。

| 状态 | 模块 | 源文件 | 行数 | 策略 |
|------|------|--------|------|------|
| [ ] | **Sigma 调度** | `k_diffusion/utils.py` | 313 | get_sigmas_karras/exponential 等 |
| [ ] | **DDIM** | `k_diffusion/sampling.py` | — | sample_ddim |
| [ ] | **Euler / Euler Ancestral** | `k_diffusion/sampling.py` | — | 调用 `torch_std_sample_euler*` |
| [ ] | **DPM++ 2M / 2S** | `k_diffusion/sampling.py` | — | 调用 `torch_std_sample_dpmpp_2m` |
| [ ] | **DPM++ SDE** | `k_diffusion/sampling.py` | — | SDE 采样变体 |
| [ ] | **Euler 完整 samplers** | `k_diffusion/sampling.py` | 1957 | 全部 sampler 函数 |
| [ ] | **DEIS** | `k_diffusion/deis.py` | — | DEIS 采样器 |
| [ ] | **SA-Solver** | `k_diffusion/sa_solver.py` | — | SA-Solver |

## Phase 7: ComfyUI Samplers

对应 `comfyui_ref/comfy/samplers.py` (1444 行) + `sampler_helpers.py` (262 行) + `sample.py` (81 行)。

| 状态 | 模块 | 源文件 | 行数 | 策略 |
|------|------|--------|------|------|
| [ ] | **Sampler 基类** | `samplers.py` | 400 | sampler 注册 + 调度 |
| [ ] | **CFG 采样** | `samplers.py` | 300 | classifier-free guidance |
| [ ] | **采样辅助** | `sampler_helpers.py` | 262 | latent 准备, denoise mask |
| [ ] | **采样管线** | `sample.py` | 81 | 顶层 sample() 函数 |
| [ ] | **Flow Matching** | `model_sampling.py` | 431 | Flux/整流流 sigma 调度 |

## Phase 8: Model Pipeline (SD / SDXL / FLUX)

对应 `comfyui_ref/comfy/sd.py` (2083 行) + `model_base.py` (2415 行)。

| 状态 | 模块 | 源文件 | 行数 | 策略 |
|------|------|--------|------|------|
| [ ] | **SD 基础管线** | `sd.py` | 800 | txt2img / img2img 流程 |
| [ ] | **Model 基类** | `model_base.py` | 1200 | BaseModel, SDXL, SD3, FLUX |
| [ ] | **Conditioning** | `conds.py` | 156 | 提示词嵌入组合 |
| [ ] | **Model 采样配置** | `model_sampling.py` | 431 | sigma 映射, 模型类型 |
| [ ] | **Model 检测** | `model_detection.py` | 1408 | 从 checkpoint 检测模型类型 |
| [ ] | **Supported Models** | `supported_models.py` | 2319 | 模型配置注册表 |

## Phase 9: ControlNet

对应 `comfyui_ref/comfy/controlnet.py` (1013 行)。
libtorch_std_helper 已有 `torch_std_controlnet_forward` + `torch_std_controlnet_apply`。

| 状态 | 模块 | 源文件 | 行数 | 策略 |
|------|------|--------|------|------|
| [ ] | **ControlNet 加载** | `controlnet.py` | 400 | 权重加载 + 模型构造 |
| [ ] | **ControlNet 前向** | `controlnet.py` | 400 | hint → UNet 特征注入 |
| [ ] | **T2I Adapter** | `t2i_adapter/adapter.py` | — | T2I-Adapter 集成 |

## Phase 10: LoRA

对应 `comfyui_ref/comfy/lora.py` (511 行)。
libtorch_std_helper 已有 `torch_std_lora_apply` + `torch_std_lora_merge_into`。

| 状态 | 模块 | 源文件 | 行数 | 策略 |
|------|------|--------|------|------|
| [ ] | **LoRA 加载** | `lora.py` | 300 | safetensors → LoRA dict |
| [ ] | **LoRA 合并** | `lora.py` | 211 | 权重合并到模型 |
| [ ] | **权重适配器 (LoHa/LoKr/OFt)** | `weight_adapter/` | — | 其他适配器变体 |

## Phase 11: FLUX

对应 `comfyui_ref/comfy/ldm/flux/` — FLUX 专用模型。

| 状态 | 模块 | 源文件 | 行数 | 策略 |
|------|------|--------|------|------|
| [ ] | **FLUX MMDiT** | `ldm/flux/model.py` | 408 | 调用 `torch_std_flux_forward` |
| [ ] | **FLUX 文本编码** | `text_encoders/flux.py` | 231 | T5 + CLIP 编码 |
| [ ] | **FLUX ControlNet** | `ldm/flux/controlnet.py` | 208 | 双流 control |

## Phase 12: T5 Text Encoder

对应 `comfyui_ref/comfy/text_encoders/t5.py` (249 行)。
libtorch_std_helper 已有 T5 SentencePiece tokenizer。

| 状态 | 模块 | 源文件 | 行数 | 策略 |
|------|------|--------|------|------|
| [ ] | **T5 Tokenizer** | `text_encoders/t5.py` | — | 调用 `torch_std_t5_tokenizer_*` |
| [ ] | **T5 模型前向** | `text_encoders/t5.py` | 249 | 编码器 forward |
| [ ] | **T5 XL / XXL** | `text_encoders/t5_config_xxl.json` | — | 模型配置 |

## Phase 13: Model Management

对应 `comfyui_ref/comfy/model_management.py` (2027 行)。

| 状态 | 模块 | 源文件 | 行数 | 策略 |
|------|------|--------|------|------|
| [ ] | **GPU 设备管理** | `model_management.py` | 500 | device 检测, CUDA 分配 |
| [ ] | **显存管理** | `memory_management.py` | 187 | cache 清理 |
| [ ] | **模型加载/卸载** | `model_management.py` | 800 | unet/clip/vae 加载 |
| [ ] | **Model Patcher** | `model_patcher.py` | 2052 | 模型补丁 (patch) |
| [ ] | **Context Windows** | `context_windows.py` | — | 分块上下文 |

## Phase 14: 工具层

| 状态 | 模块 | 源文件 | 行数 | 策略 |
|------|------|--------|------|------|
| [ ] | **工具函数** | `utils.py` | 1461 | 图像处理, 文件操作, 公共工具 |
| [ ] | **Float 格式** | `float.py` | 266 | float16/32/64 转换 |
| [ ] | **RMS Norm** | `rmsnorm.py` | 11 | RMSLayerNorm |

## Phase 15: 额外模型（按需扩展）

| 状态 | 模块 | 源文件 | 策略 |
|------|------|--------|------|
| [ ] | **SD3** | `ldm/modules/diffusionmodules/mmdit.py` (1037 行) | MMDiT 主干, 复用 FLUX 架构 |
| [ ] | **Stable Cascade** | `ldm/cascade/` | 三级级联扩散 |
| [ ] | **PixArt** | `ldm/pixart/` | Transformer 扩散 |
| [ ] | **Hunyuan Video** | `ldm/hunyuan_video/` | 视频扩散 |
| [ ] | **Wan Video** | `ldm/wan/` | 视频扩散 |
| [ ] | **Cosmos** | `ldm/cosmos/` | 视频扩散 |
| [ ] | **GLIGEN** | `gligen.py` | 布局条件 |
| [ ] | **IP-Adapter** | `text_encoders/`, `weight_adapter/` | 图像提示 |

---

## 总体进度

```
Phase 0: 基础设施   ██████████  6/6
Phase 1: 张量基元   ██████████  4/4
Phase 2: Attention  █████░░░░░  5/7
Phase 3: CLIP       █████░░░░░  4/6
Phase 4: SD UNet    ░░░░░░░░░░  0/4
Phase 5: VAE        ░░░░░░░░░░  0/4
Phase 6: K-Samplers ░░░░░░░░░░  0/9
Phase 7: Comfy Samplers ░░░░░░  0/5
Phase 8: SD Pipe    ░░░░░░░░░░  0/5
Phase 9: ControlNet ░░░░░░░░░░  0/3
Phase 10: LoRA      ░░░░░░░░░░  0/3
Phase 11: FLUX      ░░░░░░░░░░  0/3
Phase 12: T5        ░░░░░░░░░░  0/3
Phase 13: Mgmt      ░░░░░░░░░░  0/5
Phase 14: Utils     ░░░░░░░░░░  0/3
```

**总计：~63 个子模块，~76% 未开始**
**C++ 侧已实现：** libtorch_std_helper (4400 行，覆盖 SD UNet/SDXL UNet/FLUX MMDiT/CLIP/T5/ControlNet/LoRA/VAE tiling/samplers/safetensors/GGUF/Image I/O) + 新增 5 函数 (arange/cos/sin/add_scalar/mul_scalar)
**
StaticPy 侧工作：** extern fn 声明 + 管线编排 + 原语模块翻译
