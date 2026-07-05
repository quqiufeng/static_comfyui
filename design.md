# ComfyUI StaticPy 重写方案

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
