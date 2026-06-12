# static_comfyui — 开发任务表

## ComfyUI 代码库规模

```
总计: 260 个 Python 文件, 98,533 行
核心推理引擎: ~30 个文件, ~25,000 行（多模型架构共享）

关键文件引用:
  comfy/ops.py                  → 算子基类（Conv2d, Attention, GroupNorm, up/downsample）
  comfy/ldm/modules/diffusionmodules/openaimodel.py → UNet 块
  comfy/ldm/modules/attention.py → Attention 机制
  comfy/ldm/modules/diffusionmodules/model.py → 基础 UNet 结构
  comfy/model_base.py           → 模型架构（SD1.5/SDXL/FLUX）
  comfy/samplers.py             → 采样器
  comfy/k_diffusion/sampling.py → K-diffusion 采样器
  comfy/clip_model.py           → CLIP 文本编码器
  comfy/sd1_clip.py             → SD1.5 CLIP
  comfy/vae.py                  → VAE 编解码
  comfy/controlnet.py           → ControlNet
  comfy/sd.py                   → 模型加载
  comfy/model_patcher.py        → 权重管理/注入
  comfy/latent_formats.py       → latent 缩放因子
```

## 重写原则

所有算子拆解为 DGEMM + 逐元素运算。

```
Conv2d           = im2col + dgemm_row_auto
Attention        = dgemm(Q,K^T) → softmax → dgemm(attn,V)
GroupNorm        = mean → var → normalize → affine
SiLU/GELU        = 逐元素数学
ResBlock         = Conv2d + GroupNorm + SiLU + 残差加
SpatialTransformer = GroupNorm + Attention + MLP
up/downsample    = 最近邻插值 + Conv2d
```

## 任务进度

### Phase 1: 张量运算基础设施 (Tensor ops) ✅

```
文件: sd_runtime/array_ops.static.py
内容: arr_exp, arr_pow, arr_sqrt, arr_rsqrt, arr_silu, arr_gelu,
      softmax, softmax_2d, layer_norm, group_norm,
      arr_clip, arr_sum, arr_mean, arr_max, arr_min
      + 基础操作 (arr_add, arr_mul, arr_sub, arr_div 等)
状态: ✅ 已完成
```

### Phase 2: Conv2d + Attention 算子 ⬜

```
目标: im2col, conv2d, conv2d_transposed, attention, cross_attention
参考: comfy/ops.py — Conv2d 类; comfy/ldm/modules/attention.py
文件: sd_runtime/nn_ops.static.py
状态: ✅ 已完成
```

### Phase 3: UNet 块 ⬜

```
目标: ResBlock, SpatialTransformer, up_block, down_block, mid_block
参考: comfy/ldm/modules/diffusionmodules/openaimodel.py
文件: sd_runtime/unet_blocks.static.py
状态: ✅ 已完成
```

### Phase 4: UNet 完整前向 ⬜

```
目标: SD1.5 UNet forward
参考: comfy/model_base.py; comfy/ldm/modules/diffusionmodules/openaimodel.py
文件: sd_runtime/unet.static.py
状态: ✅ 已完成
```

### Phase 5: 采样器 ⬜

```
目标: DDIM, Euler, DPM++ 2M, CFG guidance
参考: comfy/samplers.py; comfy/k_diffusion/sampling.py
文件: sd_runtime/samplers.static.py
状态: ✅ 已完成
```

### Phase 6: CLIP 文本编码 ⬜

```
目标: CLIP-L tokenize + transformer encode
参考: comfy/clip_model.py; comfy/sd1_clip.py
文件: sd_runtime/clip.static.py
状态: ✅ 已完成
```

### Phase 7: VAE 编解码 ⬜

```
目标: VAE encoder + decoder
参考: comfy/vae.py
文件: sd_runtime/vae.static.py
状态: ✅ 已完成
```

### Phase 8: 模型权重加载 ⬜

```
目标: Safetensors 解析 + 权重读取
参考: comfy/sd.py; comfy/model_patcher.py
文件: sd_runtime/model_loader.static.py
状态: ✅ 已完成
```

### Phase 9: ControlNet ⬜

```
目标: ControlNet 推理（额外 Conv2d 注入）
参考: comfy/controlnet.py
文件: sd_runtime/controlnet.static.py
状态: ✅ 已完成
```

### Phase 10: txt2img 端到端集成 ⬜

```
目标: 串联所有模块，输入 prompt → 输出图像
测试: 与 ComfyUI 输出对比，数值误差 < 1e-5
文件: sd_runtime/main.static.py
状态: ✅ 已完成
```

### Phase 11: img2img + HiRes Fix ⬜

```
目标: 输入参考图 → 输出变体; 低分辨率→放大→第二轮去噪
状态: ✅ 已完成
```

### Phase 12: LoRA 权重注入 ⬜

```
目标: W += alpha * delta_W
状态: ✅ 已完成
```

### Phase 13: SDXL 支持 ⬜

```
目标: 双 CLIP + 更大 UNet
状态: ✅ 已完成
```

### Phase 14: CLI + 工作流 ⬜

```
目标: 命令行参数 + JSON workflow
状态: ✅ 已完成
```

### Phase 15: ELF 打包 ⬜

```
目标: deliver.sh → sd_generate.elf
状态: ✅ 已完成
```

---

## 已完成

```
