# IPAdapter：Static ComfyUI 实现方案

## 设计目标

- **SDXL IPAdapter Plus**（从 my-img 复用 ONNX 模型）：参考图 → CLIP Vision → Perceiver Resampler → image tokens → 追加到 `c_crossattn`
- **不修改 CrossAttention**：在 `prepare_image_generation_embeds()` 里拼接 context，最小 patch
- **无 70 层 per-layer 权重**：不依赖 `ipadapter_unet_weights.bin`

## 当前状态

- ✅ 从 my-img 复用 `IPAdapter` 类（ONNX Runtime CLIP Vision + IPAdapter Resampler）
- ✅ 接入 `sdcpp_adapter` 构建系统
- ✅ 对 sd.cpp 增加最小 patch：
  - `include/stable-diffusion.h`: 新增 `sd_ipadapter_params_t` 并加入 `sd_img_gen_params_t`
  - `src/stable-diffusion.cpp`: 在 `prepare_image_generation_embeds()` 中拼接 IPA tokens 到 `c_crossattn`
- ✅ sd.cpp + adapter + ELF 编译通过
- ✅ 端到端测试通过：SDXL Base + IPAdapter Plus（16 tokens，weight=0.8）生成 1024×1024 图片
- ✅ 与无 IPAdapter 的 baseline 对比：92% 像素不同，确认 IPAdapter 确实影响生成

## 测试记录

```bash
# IPAdapter 测试
LD_LIBRARY_PATH=cpp/sd/build:/opt/sd/build-dl/bin:/data/venv/onnxruntime-linux-x64-gpu-1.20.1/lib:/data/cuda/targets/x86_64-linux/lib \
  GGML_BACKEND_PATH=/opt/sd/build-dl/bin/libggml-cuda.so \
  ./comfycli-bin test_ipadapter.json --output-dir /tmp/comfy_output

# 输出
/tmp/comfy_output/ipadapter_test.png (1024x1024, 3 ch)

# Baseline（无 IPAdapter）
./comfycli-bin test_baseline.json --output-dir /tmp/comfy_output
/tmp/comfy_output/baseline_test.png

# 对比
md5sum:
  ipadapter_test.png  14cb34881ecdec7dbdba0ee188be1bd7
  baseline_test.png   aec50aba196ade88004d1174f6fc5a6e
  92% pixels differ, MSE=28.27
```

## 运行依赖

本地运行需要把 ONNX Runtime lib 加入 `LD_LIBRARY_PATH`：

```bash
LD_LIBRARY_PATH=cpp/sd/build:/opt/sd/build-dl/bin:/data/venv/onnxruntime-linux-x64-gpu-1.20.1/lib:/data/cuda/targets/x86_64-linux/lib \
  GGML_BACKEND_PATH=/opt/sd/build-dl/bin/libggml-cuda.so \
  ./comfycli-bin test_ipadapter.json --output-dir ./output
```

## 技术方案

### 核心数据流

```
参考图
  ↓
CLIP Vision ONNX (ViT-H/14 hidden states) → [1, 257, 1280]
  ↓
IPAdapter Plus ONNX (Perceiver Resampler)   → [1, 16, 2048]
  ↓
C++ IPAdapter 类提取 tokens [16, 2048]
  ↓
SDPipeline::generate() 将 tokens 写入 sd_img_gen_params_t.ipadapter
  ↓
sd.cpp prepare_image_generation_embeds():
    cond.c_crossattn  shape: [2048, 77] → [2048, 77+16]
    uncond.c_crossattn  同上
```

### 数据格式转换

IPAdapter ONNX 输出是 `[num_tokens, token_dim]`（row-major，token 维度优先）。
sd.cpp 的 `c_crossattn` 形状是 `[context_dim, num_tokens]`（feature 维度优先）。

转换代码（在 `stable-diffusion.cpp` 中）：

```cpp
for (int t = 0; t < n_ipa; t++) {
    for (int d = 0; d < copy_dim; d++) {
        ipa_data[d * n_ipa + t] = ipa.tokens[t * ipa_dim + d] * weight;
    }
}
```

### 架构层级

```
StaticPy nodes.static.py
  IPAdapterApply node
       ↓
C API (sdcpp_adapter.cpp)
  sd_pipeline_set_ipadapter()
  sd_pipeline_set_ipadapter_enabled()
       ↓
SDPipeline::Impl
  std::unique_ptr<IPAdapter> ipadapter
       ↓
IPAdapter (adapters/ipadapter.cpp)
  ONNX Runtime CLIP Vision + IPAdapter Resampler
       ↓
sd.cpp generate_image()
  sd_img_gen_params_t.ipadapter.*
       ↓
prepare_image_generation_embeds()
  拼接 image tokens 到 c_crossattn
```

## 与 my-img 方案的关键差异

| 维度 | my-img（已放弃） | 我们 |
|------|-----------------|------|
| CLIP Vision | ONNX Runtime 2.4GB | ONNX Runtime（从 my-img 复用）✅ |
| IPAdapter 模型 | ONNX Runtime | ONNX Runtime（从 my-img 复用）✅ |
| UNet 注入 | 70 层 per-layer k/v，改 `common_block.hpp` | c_crossattn 拼接，只改 `stable-diffusion.cpp` ✅ |
| 代码侵入 | 改 6 个 sd.cpp 文件 | 2 个 sd.cpp 文件（header + cpp）✅ |
| 依赖 | ONNX Runtime + 70 层 `.bin` 权重（1.3GB） | ONNX Runtime + 原始 onnx 模型 ✅ |
| DiT 支持 | 需要额外 2048→2560 投影 | 不优先支持 |

## 关键依赖

- ONNX Runtime：`/data/venv/onnxruntime-linux-x64-gpu-1.20.1/`
- OpenCV：用于图像预处理（resize / BGR→RGB / normalize）
- sd.cpp 需重新编译以包含 IPAdapter 字段

## 模型文件

从 my-img / ComfyUI 的 IPAdapter 生态获取：

```
/data/models/image/
├── clip_vision_vit_h_hidden.onnx      # ViT-H/14 hidden states [1, 257, 1280]
├── ipadapter_sdxl_plus_v3.onnx        # Perceiver Resampler [257, 1280] → [16, 2048]
├── ip-adapter_sdxl_vit-h.safetensors  # 源 PyTorch 权重
└── ...
```

## 验证方法

```bash
# 1. CLIP Vision + IPAdapter ONNX 推理（C++ 侧验证）
# 2. IPAdapter 字段成功传递到 sd.cpp
# 3. c_crossattn shape 变大（77 → 77+16）
# 4. 端到端：参考图 + weight=0.8 生成效果有明显构图参考
# 5. CLIP 相似度量化：生成图 vs 参考图 > 无 IPA 的 baseline
```

## 节点用法

```json
{
  "1": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {
      "ckpt_name": "sd_xl_base_1.0.safetensors"
    }
  },
  "2": {
    "class_type": "IPAdapterApply",
    "inputs": {
      "model": ["1", 0],
      "ipadapter_model": "ipadapter_sdxl_plus_v3.onnx",
      "clip_vision_model": "clip_vision_vit_h_hidden.onnx",
      "image_path": "/path/to/ref.png",
      "weight": 0.8
    }
  },
  "3": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "a red apple"
    }
  },
  "4": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["2", 0],
      "positive": ["3", 0]
    }
  }
}
```
