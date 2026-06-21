# static_comfyui — 项目完成状态

## 已完成

### 工具链 ✅
- [x] compiler/ — StaticPy → Scheme 翻译器 + 运行时
- [x] runtime/dgemm_wrapper.c — GPU/CPU 矩阵乘
- [x] build.sh — 一键编译（5 秒）
- [x] deliver.sh — → 单文件 ELF（5.2MB）
- [x] gen_unet.py — UNet 代码生成器

### 算子库 ✅
- [x] conv2d_inline: im2col + dgemm + bias（真实权重验证）
- [x] group_norm / layer_norm / softmax
- [x] arr_silu / arr_gelu / arr_tanh
- [x] upsample_nearest / downsample_conv
- [x] attention_sd: QKV → DGEMM → softmax → DGEMM
- [x] DDIM / Euler / DPM++ 采样

### SDXL UNet ✅
- [x] 所有权重加载（1680 个，从 Safetensors 导出）
- [x] 全部 9+3+9 块结构 + skip routing
- [x] ResBlock(temb) + SpatialTransformer 函数
- [x] 编译通过，结构正确
- [x] 权重合并为单文件 `weights.bin` + `index.json`
- [x] 端到端数值与 PyTorch 参考一致（max diff ~9e-4）

### VAE Decoder ⚠️
- [x] conv_in(16→512) + mid_blocks(512×2) 验证通过
- ❗ up blocks 需要 encoder skip connections（autoencoder 架构限制）

## 待完成

| 任务 | 难度 | 工作量 | 说明 |
|------|------|--------|------|
| ~~合并权重文件~~ | 低 | 1h | 已完成：单 `weights.bin` + `index.json` |
| CLIP tokenizer | 中 | 4h | BPE 词表 + transformer encode |
| VAE encoder | 中 | 4h | encoder + skip connections for decoder |
| txt2img 端到端 | 中 | 4h | CLIP→UNet→VAE 串联 |
| ControlNet/LoRA | 高 | 8h | 额外 conv 注入 + 权重合并 |
| img2img/HiResFix | 低 | 2h | 加噪声→去噪 / latent 放大→refine |

## 当前阻塞

- GPU 正在跑训练，SDXL UNet 5-step 采样 OOM，需等 GPU 空闲后再验证。
- 详见 `docs/testing.md`。

## 模型路径

```
/data/models/image/
├── sd_xl_base_1.0.safetensors    ← SDXL UNet（6.5GB）
├── ae.safetensors                 ← VAE（320MB）
├── clip_l.safetensors             ← CLIP-L（1.6GB）
├── clip_g.safetensors             ← CLIP-G（2.6GB）
└── z_image_turbo-Q5_K_M.gguf     ← 量化模型（my-img 用）
```
