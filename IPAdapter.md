# IPAdapter：使用 sd.cpp 原生实现

## 设计目标

- 直接使用 **stable-diffusion.cpp 原生 IP-Adapter**（SD1.5 / SDXL，含 Plus / Resampler）
- 不再自研 ONNX 推理与 `c_crossattn` 注入
- 移除 ONNX Runtime 依赖（部署包更小）

## 当前状态

- ✅ `sd_ctx_params_t.ip_adapter_path` + `clip_vision_path` 加载（在 ctx 创建时）
- ✅ `sd_img_gen_params_t.ip_adapter_image` + `ip_adapter_strength` 每代传入
- ✅ 端到端验证：SDXL Base + IP-Adapter Plus（16 tokens，weight=0.8）出图
- ✅ 适配层零 ONNX 依赖（`ldd` 无 `libonnxruntime`）

## 数据流

```
参考图（文件路径）
  ↓ OpenCV 读取 → RGB → sd_image_t
SDPipeline::set_ipadapter(model_path, clip_vision_path, image_path, weight)
  ├─ config.ip_adapter_path = model_path          ┐
  ├─ config.clip_vision_path = clip_vision_path   ├─ 写入 ModelConfig 并重载 ctx
  └─ load(config)（重载 new_sd_ctx）              ┘
  ↓
KSampler → generate()
  └─ img_params.ip_adapter_image    = 参考图
     img_params.ip_adapter_strength = weight
  ↓
sd.cpp 原生：CLIP-Vision(ViT-H/14) → 投影/Resampler → 注入 UNet attn2
```

## 架构层级

```
StaticPy nodes.static.py
  IPAdapterApply / IPAdapterModelLoader / CLIPVisionLoader
       ↓ extern fn
C API（sdcpp_adapter.cpp）
  sd_pipeline_set_ipadapter(pipeline, ipadapter_path, clip_vision_path, image, weight)
       ↓
SDPipeline（写 ModelConfig → 重载 ctx；加载参考图）
       ↓
sd.cpp 原生 IP-Adapter（sd_ctx_params_t.ip_adapter_path）
```

## 所需模型（sd.cpp 格式，非 ONNX）

| 文件 | 说明 | 来源 |
|------|------|------|
| base SD1.5 / SDXL | 主模型 | — |
| `clip_vision_h.safetensors` | CLIP-Vision ViT-H/14 编码器 | h94/IP-Adapter 或 Comfy-Org 重打包 |
| `ip-adapter-plus_sdxl_vit-h.safetensors` | SDXL Plus IP-Adapter | h94/IP-Adapter |

本机可用：
- `/data/models/image/clip_vision_sd15.safetensors`（ViT-H，键 `vision_model.*`）
- `/data/models/image/ip-adapter-plus_sdxl_vit-h.safetensors`（SDXL Plus）

## 节点用法

```json
{
  "1": { "class_type": "CheckpointLoaderSimple", "inputs": { "ckpt_name": "sd_xl_base_1.0.safetensors" } },
  "2": { "class_type": "IPAdapterModelLoader", "inputs": { "ipadapter_file": "ip-adapter-plus_sdxl_vit-h.safetensors" } },
  "3": { "class_type": "CLIPVisionLoader", "inputs": { "clip_name": "clip_vision_sd15.safetensors" } },
  "4": { "class_type": "IPAdapterApply", "inputs": {
           "model": ["1", 0], "ipadapter": ["2", 0], "clip_vision": ["3", 0],
           "image_path": "/path/to/ref.png", "weight": 0.8 } },
  "5": { "class_type": "CLIPTextEncode", "inputs": { "text": "a woman", "clip": ["1", 0] } },
  "6": { "class_type": "EmptyLatentImage", "inputs": { "width": 512, "height": 512 } },
  "7": { "class_type": "KSampler", "inputs": {
           "model": ["4", 0], "positive": ["5", 0], "negative": ["5", 0],
           "latent_image": ["6", 0], "steps": 20, "cfg": 6.0, "seed": 42,
           "sampler_name": "dpm++2m", "scheduler": "karras" } }
}
```

## 运行

```bash
LD_LIBRARY_PATH=cpp/sd/build:/opt/sd/build-dl/bin \
  GGML_BACKEND_PATH=/opt/sd/build-dl/bin/libggml-cuda.so \
  ./comfycli-bin workflow.json --output-dir ./output
```

启动日志会显示 `IP-Adapter: 16 image tokens`（Plus）或 `4`（经典）。

## 依赖的上游修复

sd.cpp `7f410a3` 有两个回归会破坏原生 IP-Adapter，已在本项目 patch 修复（见 `cpp/sd/design.md` §4.6/§4.7）：

1. `model_loader.cpp`：`unused_tensors` 含 `"vision_model."` → 过滤掉独立 CLIP vision 文件（#1935 引入）
2. `diffusion_engine.cpp`：clip vision 加载前缀被误改为 `"clip_vision."`（#1957），应为 `"cond_stage_model.transformer."`

## 验证方法

```bash
# 端到端：参考图 + weight=0.8 出图
# 与无 IPAdapter 的 baseline 对比像素差异（应显著不同）
```
