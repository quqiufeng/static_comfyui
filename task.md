# static_comfyui — 开发任务表

## ComfyUI 代码库规模

```
总计: 260 个 Python 文件, 98,533 行
核心推理引擎: ~30 个文件, ~25,000 行
```

## 重写原则

所有算子拆解为 DGEMM + 逐元素运算。

## 任务进度

### Phase 1: 张量运算基础设施 ✅
```
文件: sd_runtime/array_ops.static.py
```

### Phase 2: Conv2d + Attention 算子 ✅
```
文件: sd_runtime/nn_ops.static.py
```

### Phase 3: UNet 块 (ResBlock, SpatialTransformer) ✅
```
文件: sd_runtime/unet_blocks.static.py
```

### Phase 4: UNet 完整前向 ✅
```
文件: sd_runtime/unet.static.py
```

### Phase 5: 采样器 (DDIM, Euler, CFG) ✅
```
文件: sd_runtime/samplers.static.py
```

### Phase 6: CLIP 文本编码 ✅
```
文件: sd_runtime/clip.static.py
```

### Phase 7: VAE 编解码 ⬜
```
文件: sd_runtime/vae.static.py
状态: ⬜ 未开始
```

### Phase 8: 模型权重加载 ✅
```
文件: sd_runtime/model_loader.static.py + export_sd_weights.py
```

### Phase 9: ControlNet ⬜
```
状态: ⬜ 未开始
```

### Phase 10: txt2img 端到端集成 ✅
```
文件: sd_runtime/main.static.py
```

### Phase 11: img2img + HiRes Fix ⬜
### Phase 12: LoRA 权重注入 ⬜
### Phase 13: SDXL 支持 ⬜
### Phase 14: CLI + 工作流 ⬜
### Phase 15: ELF 打包 (deliver.sh) ⬜

---

## 已完成
```
Phase 1  - sd_runtime/array_ops.static.py
Phase 2  - sd_runtime/nn_ops.static.py
Phase 3  - sd_runtime/unet_blocks.static.py
Phase 4  - sd_runtime/unet.static.py
Phase 5  - sd_runtime/samplers.static.py
Phase 6  - sd_runtime/clip.static.py
Phase 8  - sd_runtime/model_loader.static.py + export_sd_weights.py
Phase 10 - sd_runtime/main.static.py
```
