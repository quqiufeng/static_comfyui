# ComfyUI StaticPy 重写方案

## 模型文件位置（调试用）

| 模型 | 路径 | 大小 | 格式 |
|------|------|------|------|
| SDXL 1.0 base | `/data/models/image/sd_xl_base_1.0.safetensors` | 6.5G | safetensors |
| CLIP-G (SDXL) | `/data/models/image/clip_g.safetensors` | 2.6G | safetensors |
| CLIP-L (SDXL) | `/data/models/image/clip_l.safetensors` | 1.6G | safetensors |
| VAE (SDXL) | `/data/models/image/ae.safetensors` | 320M | safetensors |
| CLIP Vision (SD1.5) | `/data/models/image/clip_vision_sd15.safetensors` | 2.4G | safetensors |
| IP-Adapter SDXL | `/data/models/image/ip-adapter-plus_sdxl_vit-h.safetensors` | 809M | safetensors |
| IP-Adapter SD1.5 | `/data/models/image/ip-adapter-plus_sd15.safetensors` | 94M | safetensors |
| 2x ESRGAN upscaler | `/data/models/image/2x_ESRGAN.gguf` | — | GGUF |
| z_image_turbo | `/data/models/image/z_image_turbo-Q5_K_M.gguf` | — | GGUF |

**注意**：`libtorch_std_helper.so` 已内置 `torch_std_safetensors_load` 和 `torch_std_gguf_load`，可直接加载上述任意格式。

## CLI 接口

```
# 模式 1：直接命令行出图
comfycli \
  --checkpoint /data/models/image/sd_xl_base_1.0.safetensors \
  --prompt "cat wearing hat" \
  --output ./output.png

# 模式 2：执行 ComfyUI workflow JSON
comfycli workflow.json \
  --output-dir ./output
```

两种模式共享同一推理引擎，区别仅在输入解析层：
- **模式 1**：内部构造单节点 Workflow（CheckpointLoader → CLIPTextEncode → KSampler → VAEDecode → SaveImage）
- **模式 2**：解析 ComfyUI 标准 workflow JSON，遍历 DAG 执行

## 目标

用 StaticPy（Python 子集编译器）+ `libtorch_std_helper.so`（C++ libtorch 封装）1:1 重写 ComfyUI，编译为独立 ELF 二进制，消除 Python 解释器和 pip 依赖。

---

## 技术栈

| 层 | 技术 | 职责 |
|---|------|------|
| 编排语言 | StaticPy | 节点 DAG 调度、模型管理、prompt 校验、缓存 |
| 推理后端 | `libtorch_std_helper.so` | UNet forward、VAE、CLIP、sampler、ControlNet、LoRA |
| 编译器 | `static_translate.py` + Chez Scheme AOT | Python 子集 → Scheme → `.so` → ELF |
| 辅助工具 | code search（my_db） | 语义搜索定位源码，辅助 1:1 翻译 |

---

## 项目结构

```
comfycli/
├── main.static.py             # CLI 入口：读取 workflow JSON → 执行 → 输出
├── execution.static.py        # PromptExecutor DAG 调度
├── nodes.static.py            # 200+ 节点定义（INPUT_TYPES + extern fn）
├── model_config.static.py     # 模型架构检测（state_dict → SD1.5/SDXL/FLUX…）
├── model_management.static.py # GPU 显存管理（load/offload/unload）
├── validate.static.py         # prompt 校验
├── cache.static.py            # 节点输出缓存（is_changed 检测）
├── sampler.static.py          # 采样配置（sigma schedule、步数）
├── folder_paths.static.py     # 路径管理
├── latent_formats.static.py   # 潜空间缩放因子
│
├── lib/                       # C++ helper（链接 libtorch_std_helper.so）
│   └── comfy_helpers.cpp     # 补充 helper（如有需要）
│
├── build.sh                   # 编译脚本
└── README.md
```

---

## 分层架构

```
┌─────────────────────────────────────────────────────┐
│                  CLI 层 (main.static.py)             │
│  read workflow.json → validate → execute → save     │
├─────────────────────────────────────────────────────┤
│              编排层 (StaticPy 编译)                  │
│                                                      │
│  execution.static.py    节点 DAG 调度                │
│  nodes.static.py        节点注册表 + 200+ 节点定义   │
│  validate.static.py     prompt 结构校验              │
│  cache.static.py        输出缓存                     │
│  sampler.static.py      采样参数配置                 │
├─────────────────────────────────────────────────────┤
│              模型管理层 (StaticPy)                   │
│                                                      │
│  model_config.static.py     架构检测与配置           │
│  model_management.static.py GPU 显存调度             │
│  folder_paths.static.py     路径管理                 │
│  latent_formats.static.py   潜空间格式               │
├─────────────────────────────────────────────────────┤
│         C++ FFI 层 (extern fn → .so)                │
│                                                      │
│  libtorch_std_helper.so                             │
│  └── UNet forward (SD1.5/SDXL/SD3/FLUX/…)          │
│  └── VAE encode/decode (tiled)                      │
│  └── CLIP tokenizer + encoder                       │
│  └── T5 tokenizer                                   │
│  └── Sampler (DDIM/Euler/DPM++/FM)                  │
│  └── ControlNet forward + apply                     │
│  └── LoRA load + merge + apply                      │
│  └── safetensors / GGUF / JIT load                  │
│  └── Image I/O (PNG/JPEG)                           │
│  └── Attention (multi-head)                         │
├─────────────────────────────────────────────────────┤
│            libtorch / cuDNN / cuBLAS                 │
└─────────────────────────────────────────────────────┘
```

---

## 文件级 1:1 映射

ComfyUI 每个 Python 文件直接对应一个 `.static.py` 文件。
C++ 依赖类型决定翻译策略：**torch** 走 `extern fn` 调 `libtorch_std_helper.so`，**sys** 直接翻译为纯 StaticPy。

### 核心目录 (`comfy/` → `comfycli/`)

| ComfyUI 源文件 | StaticPy 目标 | C++ 依赖 | 翻译策略 |
|---------------|--------------|---------|---------|
| `comfy/sd.py` | `sd.static.py` | torch | 模型加载/组合, 调 extern fn |
| `comfy/clip_model.py` | `clip_model.static.py` | torch | CLIP 推理逻辑 |
| `comfy/clip_vision.py` | `clip_vision.static.py` | sys | 纯编排, 直接翻译 |
| `comfy/conds.py` | `conds.static.py` | torch | 条件 embedding 拼接 |
| `comfy/controlnet.py` | `controlnet.static.py` | torch | ControlNet 加载+apply |
| `comfy/gligen.py` | `gligen.static.py` | torch | GLIGEN 逻辑 |
| `comfy/hooks.py` | `hooks.static.py` | torch | 模型 patch hook |
| `comfy/latent_formats.py` | `latent_formats.static.py` | torch | 潜空间缩放因子 |
| `comfy/lora.py` | `lora.static.py` | torch | LoRA 加载+合并 |
| `comfy/model_base.py` | `model_base.static.py` | torch | 模型基类 (影响大量子模块) |
| `comfy/model_detection.py` | `model_detection.static.py` | torch | state_dict → 架构检测 |
| `comfy/model_management.py` | `model_management.static.py` | torch + xformers + comfy_aimdo | 显存调度, CUDA API 替代 comfy_aimdo |
| `comfy/model_sampling.py` | `model_sampling.static.py` | torch | sigma 调度 |
| `comfy/model_patcher.py` | `model_patcher.static.py` | torch + comfy_aimdo | 模型 patching |
| `comfy/ops.py` | `ops.static.py` | torch + comfy_aimdo | 算子注册 (extern fn 替代) |
| `comfy/sample.py` | `sample.static.py` | torch + numpy | 采样入口 |
| `comfy/samplers.py` | `samplers.static.py` | torch + scipy | 采样器配置 |
| `comfy/sampler_helpers.py` | `sampler_helpers.static.py` | torch | 采样辅助 |
| `comfy/sd1_clip.py` | `sd1_clip.static.py` | torch + transformers | SD1.5 CLIP |
| `comfy/sdxl_clip.py` | `sdxl_clip.static.py` | torch | SDXL CLIP |
| `comfy/float.py` | `float.static.py` | torch + comfy_kitchen | FP8 量化 |
| `comfy/quant_ops.py` | `quant_ops.static.py` | torch + comfy_kitchen + triton | 量化算子 |
| `comfy/rmsnorm.py` | `rmsnorm.static.py` | torch | RMSNorm |
| `comfy/memory_management.py` | `memory_management.static.py` | torch + ctypes + comfy_aimdo | 内存分配跟踪 |
| `comfy/pinned_memory.py` | `pinned_memory.static.py` | torch + comfy_aimdo | 固定内存 CUDA API |
| `comfy/nested_tensor.py` | `nested_tensor.static.py` | torch | 嵌套 tensor 工具 |
| `comfy/utils.py` | `utils.static.py` | torch + ctypes + numpy + PIL + comfy_aimdo | 工具函数 (mmap, 图片) |

### 纯编排 (无 torch, 直接翻译)

| ComfyUI 源文件 | StaticPy 目标 | C++ 依赖 | 翻译策略 |
|---------------|--------------|---------|---------|
| `comfy/cli_args.py` | `cli_args.static.py` | sys | 直接翻译 |
| `comfy/options.py` | `options.static.py` | sys | 常量定义 |
| `comfy/patcher_extension.py` | `patcher_extension.static.py` | sys | 接口定义 |
| `comfy/deploy_environment.py` | `deploy_environment.static.py` | sys | 环境检测 |
| `comfy/diffusers_load.py` | `diffusers_load.static.py` | sys | diffusers 加载逻辑 |
| `comfy/supported_models.py` | `supported_models.static.py` | torch | 模型注册表 |
| `comfy/supported_models_base.py` | `supported_models_base.static.py` | torch | 基类定义 |
| `comfy/context_windows.py` | `context_windows.static.py` | torch | 上下文窗口 |
| `comfy/comfy_types/node_typing.py` | `node_typing.static.py` | sys | 类型定义 |

### 根目录 (`ComfyUI/` → `comfycli/`)

| ComfyUI 源文件 | StaticPy 目标 | C++ 依赖 | 翻译策略 |
|---------------|--------------|---------|---------|
| `execution.py` | `execution.static.py` | torch + comfy_aimdo | DAG 执行引擎 |
| `nodes.py` | `nodes.static.py` | torch + PIL | 200+ 节点注册 |
| `main.py` | `main.static.py` | comfy_aimdo | CLI 入口 |
| `folder_paths.py` | `folder_paths.static.py` | sys | 路径管理 |
| `node_helpers.py` | `node_helpers.static.py` | torch + PIL | 节点辅助 |
| `cuda_malloc.py` | `cuda_malloc.static.py` | ctypes | CUDA 内存分配 |

### 模型架构目录 (`comfy/ldm/` → `comfycli/ldm/`)

所有模型架构文件均依赖 **torch**，统一走 `extern fn` 调用 `libtorch_std_helper.so`：

| ComfyUI 子目录 | StaticPy 目标 | 说明 |
|---------------|--------------|------|
| `ldm/flux/` (5 文件) | `ldm/flux/` | FLUX 双流 DiT |
| `ldm/cascade/` (6 文件) | `ldm/cascade/` | Stable Cascade |
| `ldm/cosmos/` (7 文件) | `ldm/cosmos/` | Cosmos 视频模型 |
| `ldm/hunyuan_video/` (4 文件) | `ldm/hunyuan_video/` | Hunyuan 视频 |
| `ldm/wan/` (7 文件) | `ldm/wan/` | Wan 视频 |
| `ldm/genmo/` (6 文件) | `ldm/genmo/` | Genmo 视频 |
| `ldm/lightricks/` (14 文件) | `ldm/lightricks/` | Lightricks 视频 |
| `ldm/ace/` (7 文件) | `ldm/ace/` | ACE 音频 |
| `ldm/sam3/` (3 文件) | `ldm/sam3/` | SAM3 分割 |
| `ldm/pixart/` (2 文件) | `ldm/pixart/` | PixArt |
| `ldm/audio/` (4 文件) | `ldm/audio/` | 音频生成 |
| `ldm/models/autoencoder.py` | `ldm/models/autoencoder.static.py` | VAE |

### 文本编码器 (`comfy/text_encoders/` → `comfycli/text_encoders/`)

| ComfyUI 源文件 | StaticPy 目标 | C++ 依赖 | 翻译策略 |
|---------------|--------------|---------|---------|
| `text_encoders/t5.py` | `text_encoders/t5.static.py` | torch | T5 编码器 |
| `text_encoders/flux.py` | `text_encoders/flux.static.py` | torch | FLUX 双编码器 |
| `text_encoders/bert.py` | `text_encoders/bert.static.py` | torch | BERT |
| `text_encoders/sd3_clip.py` | `text_encoders/sd3_clip.static.py` | torch | SD3 CLIP |
| `text_encoders/gemma4.py` | `text_encoders/gemma4.static.py` | torch + numpy | Gemma |
| `text_encoders/llama.py` | `text_encoders/llama.static.py` | torch | LLM |
| `text_encoders/gpt_oss.py` | `text_encoders/gpt_oss.static.py` | torch | OSS GPT |
| `text_encoders/spiece_tokenizer.py` | `text_encoders/spiece_tokenizer.static.py` | sentencepiece | T5 tokenizer (C++ FFI) |
| **其余 30+ 文件** | `text_encoders/*.static.py` | sys/torch | 各架构 tokenizer 封装 |

### k_diffusion (`comfy/k_diffusion/` → `comfycli/k_diffusion/`)

| ComfyUI 源文件 | StaticPy 目标 | C++ 依赖 | 翻译策略 |
|---------------|--------------|---------|---------|
| `k_diffusion/sampling.py` | `k_diffusion/sampling.static.py` | torch + scipy | 采样器 (调 extern fn) |
| `k_diffusion/utils.py` | `k_diffusion/utils.static.py` | torch | 采样工具 |
| `k_diffusion/deis.py` | `k_diffusion/deis.static.py` | torch + numpy | DEIS 采样 |
| `k_diffusion/sa_solver.py` | `k_diffusion/sa_solver.static.py` | torch | SA 求解器 |

### comfy_extras (非核心，首次跳过)

约 90 个文件，均为面向特定模型的节点包装层。首次只翻译核心路径，冷门节点后续补充。

---

## code search 辅助翻译流程

每个 ComfyUI 模块对应一次 code search 定位 + 1:1 翻译：

```
ComfyUI Python 源码                    StaticPy 编译产物
────────────────────                  ──────────────────

[execution.py]
PromptExecutor         ──语义搜索──→  execution.static.py
ExecutionList                           class PromptExecutor:
stage_node_execution                       def execute(self, prompt):
validate_prompt                                while not list.is_empty():

[nodes.py]
CheckpointLoaderSimple  ──语义搜索──→  nodes.static.py
CLIPTextEncode                             @node
VAEDecode                                  class CheckpointLoaderSimple:
KSampler                                       inputs: {...}
UNETLoader                                     outputs: ("MODEL",)
...200+ nodes                                  fn = extern load_checkpoint

[comfy/model_detection.py]
detect_unet_config     ──语义搜索──→  model_config.static.py
model_config_from_unet                      def detect_unet_config(sd):
model_config_from_unet_config                   # key pattern → architecture

[comfy/model_management.py]
load_models_gpu         ──语义搜索──→  model_management.static.py
free_memory                                     def load_models_gpu(models):
soft_empty_cache                                    for m in models:
unload_all_models                                       extern_load_model_gpu(m)
cleanup_models_gc

[comfy/sd.py]
load_diffusion_model   ──语义搜索──→  (直接调 extern fn)
load_clip                                        extern fn load_diffusion_model
load_vae                                              → libtorch_std_helper
load_lora_for_models
```

搜索命令示例：

```bash
# 定位 PromptExecutor 源码
cache_query "PromptExecutor execution loop" --repo /code/comfyui --type search \
  --analysis-dir /opt/code_caches/comfyui_cache

# 查看调用链
cache_query PromptExecutor --repo /code/comfyui --type context --depth 2

# 定位模型检测逻辑
cache_query "model config detect unet architecture" --repo /code/comfyui --type search \
  --analysis-dir /opt/code_caches/comfyui_cache

# 定位节点定义模式
cache_query "node registry class type mapping" --repo /code/comfyui --type search \
  --analysis-dir /opt/code_caches/comfyui_cache
```

---

## 模块映射（1:1）

### execution.static.py — DAG 执行引擎

```python
extern fn load_model(model_type: str, model_path: str) -> ptr from "torch_std"
extern fn unet_forward(model: ptr, x: ptr, t: ptr, cond: ptr) -> ptr from "torch_std"
extern fn vae_decode(vae: ptr, latent: ptr) -> ptr from "torch_std"
extern fn vae_decode_tiled(vae: ptr, latent: ptr, tile: int, overlap: int) -> ptr from "torch_std"

class PromptExecutor:
    def __init__(self, cache_type: str):
        self.cache = CacheSet(cache_type)
        self.models = {}

    def execute(self, prompt: dict) -> dict:
        dynamic_prompt = self.resolve_dynamic(prompt)
        models = self.prefetch_models(dynamic_prompt)
        execution_list = ExecutionList(dynamic_prompt, self.cache)
        outputs = {}
        while not execution_list.is_empty():
            node_id = execution_list.stage_node_execution()
            node_def = NODE_REGISTRY[node_id.class_type]
            inputs = self.resolve_inputs(node_def, node_id, outputs)
            result = node_def.function(**inputs)
            outputs[node_id] = result
        return outputs
```

### nodes.static.py — 节点注册表

```python
@dataclass
class NodeDef:
    class_type: str
    inputs: dict
    outputs: tuple
    category: str
    function: callable

NODE_REGISTRY: dict = {}

def register(node_class):
    NODE_REGISTRY[node_class.__name__] = NodeDef(
        class_type = node_class.__name__,
        inputs = node_class.INPUT_TYPES(),
        outputs = node_class.RETURN_TYPES,
        category = node_class.CATEGORY,
        function = node_class.FUNCTION,
    )

@register
class UNETLoader:
    INPUT_TYPES = {"required": {"unet_name": str, "weight_dtype": str}}
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "model/loaders"

    def load_unet(self, unet_name: str, weight_dtype: str) -> tuple:
        path = get_full_path("diffusion_models", unet_name)
        model = extern_load_diffusion_model(path, weight_dtype)
        return (model,)
```

### model_config.static.py — 模型架构检测

```python
def detect_unet_config(sd: ptr) -> dict:
    # 检查 state_dict key 模式 → 确定模型架构
    if has_key(sd, "model.diffusion_model.input_blocks.0.0.weight"):
        return {"model_type": "sd15", "unet_config": {...}}
    if has_key(sd, "conditioner.embedders.0.model"):
        return {"model_type": "sdxl", "unet_config": {...}}
    if has_key(sd, "double_blocks.0.img_block"):
        return {"model_type": "flux", "unet_config": {...}}
    ...

def model_config_from_unet(unet_config: dict) -> BASE:
    # 返回对应架构的模型配置类
    ...
```

### model_management.static.py — 显存管理

```python
extern fn cuda_get_free_memory() -> int from "torch_std"
extern fn cuda_load_model(device: int, model: ptr) -> int from "torch_std"
extern fn cuda_unload_model(model: ptr) -> int from "torch_std"
extern fn cuda_soft_empty_cache() from "torch_std"

loaded_models: list = []

def load_models_gpu(models: list):
    for m in models:
        if not m.loaded:
            needed = estimate_memory(m)
            free_memory(needed)
            cuda_load_model(0, m.ptr)
            m.loaded = True

def free_memory(needed: int):
    while cuda_get_free_memory() < needed and loaded_models:
        m = loaded_models.pop()
        cuda_unload_model(m.ptr)
        m.loaded = False
```

---

## C++ 依赖清单

ComfyUI 自身不含 C/C++ 源文件，所有 C++ 功能来自外部 pip 包。以下按类别列出，
标注覆盖状态，方便翻译时对照。

### 核心 ML 计算（`libtorch_std_helper.so` 已覆盖）

| Python import | ComfyUI 使用文件 | C++ 底层 | 覆盖状态 |
|--------------|-----------------|---------|---------|
| `torch` (UNet forward) | `comfy/ldm/modules/diffusionmodules/model.py` | libtorch/cuDNN | ✅ `unet_forward` |
| `torch` (VAE) | `comfy/sd.py` | libtorch/cuDNN | ✅ `vae_encode` / `vae_decode` |
| `torch` (CLIP) | `comfy/text_encoders/` | libtorch | ✅ `clip_encode` |
| `torch` (sampler) | `comfy/k_diffusion/sampling.py` | libtorch | ✅ sampler step |
| `xformers.ops` | `comfy/ldm/modules/attention.py:20` | xformers (CUDA C++) | ✅ attention (libtorch 内置) |
| `flash_attn` | `comfy/ldm/modules/attention.py:44` | flash-attn (CUDA C++) | ✅ attention (libtorch 内置) |
| `sageattention` | `comfy/ldm/modules/attention.py:25` | sageattention (CUDA C++) | ✅ attention (libtorch 内置) |
| `torchsde` | `comfy/k_diffusion/sampling.py:7` | torchsde (C++ ext) | ⏳ SDE sampler (后续) |

### 量化（`libtorch_std_helper.so` 部分覆盖）

| Python import | ComfyUI 使用文件 | C++ 底层 | 覆盖状态 |
|--------------|-----------------|---------|---------|
| `comfy_kitchen` | `comfy/quant_ops.py:7`, `comfy/float.py:7` | FP8/FP4 CUDA C++ | ⏳ 需要时补 extern fn |
| `triton` | `comfy/quant_ops.py:29` | triton (LLVM) | ❌ 跳过，非核心路径 |

### 显存管理（StaticPy 侧用 CUDA API 替代）

| Python import | ComfyUI 使用文件 | C++ 底层 | 覆盖状态 |
|--------------|-----------------|---------|---------|
| `comfy_aimdo.host_buffer` | `comfy/model_management.py:35` | comfy-aimdo (C++) | 🔄 用 `cudaHostRegister` FFI 替代 |
| `comfy_aimdo.vram_buffer` | `comfy/model_management.py:36` | comfy-aimdo (C++) | 🔄 用 `cudaMalloc` FFI 替代 |
| `comfy_aimdo.model_mmap` | `comfy/utils.py:86` | comfy-aimdo (C++) | 🔄 用 `mmap` FFI 替代 |
| `comfy_aimdo.model_vbar` | `comfy/ops.py:30`, `execution.py:20` | comfy-aimdo (C++) | 🔄 用 CUDA API 替代 |
| `comfy_aimdo.torch` | `comfy/ops.py:31`, `pinned_memory.py:7` | comfy-aimdo (C++) | 🔄 用 `torch.cuda` FFI 替代 |
| `comfy_aimdo.control` | `main.py:55` | comfy-aimdo (C++) | 🔄 跳过，CLI 版不需要 |
| `torch.cuda.cudart()` | `comfy/pinned_memory.py:56` | CUDA Runtime API | ✅ `extern fn cudaHostRegister` |

### Tokenizer（需新增 extern fn 或纯 StaticPy 实现）

| Python import | ComfyUI 使用文件 | C++ 底层 | 覆盖状态 |
|--------------|-----------------|---------|---------|
| `tokenizers.Tokenizer` | `comfy/text_encoders/gpt_oss.py:432` | HuggingFace tokenizers (Rust) | ⏳ 需封装 extern fn |
| `sentencepiece` | `comfy/text_encoders/spiece_tokenizer.py:13` | sentencepiece (C++) | ⏳ 需封装 extern fn |

### 图像 / 视频 I/O（prelude + libtorch_std_helper 覆盖）

| Python import | ComfyUI 使用文件 | C++ 底层 | 覆盖状态 |
|--------------|-----------------|---------|---------|
| `PIL` (Pillow) | `comfy_extras/nodes_wan.py` 等 | libjpeg/libpng (C) | ✅ `extern fn save_image` / `load_image` |
| `av` (PyAV) | `comfy_extras/nodes_video.py:2` | FFmpeg (C) | ✅ 已有，非核心路径可后补 |
| `torchaudio` | `comfy_extras/nodes_audio.py:2` | SoX/FFmpeg (C++) | ⏳ 非核心路径，后续 |
| `cv2` (OpenCV) | `comfy_extras/nodes_sdpose.py:82` | OpenCV (C++) | ⏳ 非核心路径，后续 |

### 图像处理 / 超分（非核心路径，后续）

| Python import | ComfyUI 使用文件 | C++ 底层 | 覆盖状态 |
|--------------|-----------------|---------|---------|
| `torchvision.ops` | `comfy/background_removal/birefnet.py:7` | torchvision (C++/CUDA) | ⏳ 非核心，后续 |
| `kornia` | 依赖项 (requirements.txt) | kornia (C++/CUDA) | ⏳ 非核心，后续 |
| `spandrel` | 依赖项 (requirements.txt) | spandrel (C++/CUDA) | ⏳ 非核心，后续 |

### 3D / GLSL 渲染（跳过，CLI 版不需要）

| Python import | ComfyUI 使用文件 | C++ 底层 | 覆盖状态 |
|--------------|-----------------|---------|---------|
| `comfy_angle` | `comfy_extras/nodes_glsl.py:12` | ANGLE (C++) | ❌ 跳过 |
| `OpenGL.EGL` | `comfy_extras/nodes_glsl.py:63` | libEGL (C) | ❌ 跳过 |
| `OpenGL.GLES3` | `comfy_extras/nodes_glsl.py:64` | libGLESv2 (C) | ❌ 跳过 |

### 工具库（prelude / 系统 lib 覆盖）

| Python import | ComfyUI 使用文件 | C++ 底层 | 覆盖状态 |
|--------------|-----------------|---------|---------|
| `safetensors` | 模型加载 (Rust) | safetensors (Rust) | ✅ `extern fn load_safetensors` |
| `blake3` | `app/assets/services/hashing.py:9` | blake3 (Rust) | ⏳ 缓存哈希，可简化 |
| `numpy` | 多处 | BLAS/LAPACK (C/Fortran) | ⏳ 少量使用，可用 StaticPy 替代 |
| `scipy` | 少量使用 | BLAS/LAPACK (C/Fortran) | ⏳ 极少使用，可用 StaticPy 替代 |

### 状态标记

| 标记 | 含义 |
|------|------|
| ✅ 已覆盖 | `libtorch_std_helper.so` 已有对应 extern fn |
| 🔄 替代方案 | 用 CUDA API 或系统 lib 的 extern fn 替代 |
| ⏳ 后续 | 非核心路径，首次发布后补 |
| ❌ 跳过 | CLI 版不需要 |

---

## 编译流水线

```bash
# 1. 编译 StaticPy 源码 → Scheme → AOT .so → ELF
./build.sh

# 内部流程：
# static_translate.py main.static.py → main_code.ss
# static_build.sh main_code.ss \
#   --prelude static_prelude.scm \
#   --stdlib static_stdlib.scm \
#   --output ./comfycli

# 2. 运行时
LD_LIBRARY_PATH=/opt/ReScheme:/data/venv/lib/python3.12/site-packages/torch/lib \
  ./comfycli workflow.json --output-dir ./output
```

`build.sh` 封装：

```bash
#!/bin/bash
set -e
SCRIPT_DIR="/opt/ReScheme"

# 翻译
python3 $SCRIPT_DIR/static_translate.py main.static.py > main_code.ss

# 编译
$SCRIPT_DIR/static_build.sh main_code.ss \
  --prelude $SCRIPT_DIR/static_prelude.scm \
  --stdlib $SCRIPT_DIR/static_stdlib.scm \
  --output ./comfycli

# 产物
# ./comfycli          — 独立 ELF 二进制
# ./comfycli.so       — Chez AOT 编译产物
echo "Build complete: ./comfycli"
```

---

## 依赖

运行时依赖（无 pip）：

| 库 | 来源 |
|----|------|
| `libtorch_std_helper.so` | `build_torch_std_helper.sh` |
| `libtorch.so` | PyTorch C++ 分发包 |
| `libcudnn.so` | CUDA 12 |
| `libcudart.so` | CUDA 12 |
| `libc.so.6` | 系统 |
| `libm.so.6` | 系统 |

---

## 翻译策略

| ComfyUI 特性 | StaticPy 方案 | 备注 |
|-------------|--------------|------|
| `@classmethod` | `@dataclass` + 直接函数 | StaticPy 不支持 classmethod，节点定义扁平化 |
| `import torch` | `extern fn ... from "torch_std"` | 所有 torch 操作走 FFI |
| `try/except` | if 守卫 + error code | StaticPy 无异常，用返回值表示错误 |
| `dynamic import` | 静态注册表 | `custom_nodes/` 在编译期注册 |
| `__init__` 继承 | dataclass + 组合 | StaticPy 无类继承 |
| `asyncio / await` | 同步执行 | CLI 版不需要异步，`asyncio.run()` 替换为同步循环 |
| `eval/exec` | 编译期展开 | ComfyUI 内部无动态代码执行 |
| `WebSocket` | 跳过（CLI 版） | 后续用嵌入式 C WS server |

---

## 工程顺序

```
Phase 1: 执行引擎          execution.static.py      ← PromptExecutor 纯逻辑，最简单
Phase 2: 节点定义          nodes.static.py           ← 200 节点，机械翻译
Phase 3: 模型检测          model_config.static.py    ← key pattern 匹配
Phase 4: 显存管理          model_management.static.py ← 调度策略
Phase 5: Prompt 校验       validate.static.py        ← 结构检查
Phase 6: 缓存系统          cache.static.py           ← is_changed 检测
Phase 7: 采样配置          sampler.static.py         ← sigma 参数
Phase 8: CLI 入口          main.static.py            ← 胶水集成
Phase 9: 验证              workflow.json → 输出比对  ← 与原始 ComfyUI 输出对比
```

---

## 局限与取舍

| 取舍 | 说明 |
|------|------|
| 无自定义节点 | `custom_nodes/` 需转为 StaticPy 编译期注册，运行时无动态加载 |
| 仅推理，无训练 | 重写范围限 inference pipeline，ComfyUI 本身也不是训练工具 |
| CLI 先行，无 UI | 后续通过 `extern fn` 嵌入 C HTTP/WS server |
| 同步执行 | 去掉 asyncio，单线程 DAG 调度（CLI 场景不需要并发） |
| 模型精度 | 对齐 `libtorch_std_helper` 支持的精度（fp32/fp16/bf16/fp8） |
