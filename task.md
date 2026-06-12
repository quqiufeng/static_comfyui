# static_comfyui — ML → ELF 编译运行时

## 架构

```
用户 StaticPy 代码 (.static.py)
    │
    ├── compiler/translate.py    ← Python AST → Scheme
    ├── compiler/prelude.scm     ← 运行时基础库
    └── compiler/stdlib.scm      ← ML 库（按需 dlopen）
    │
    ▼  Chez compile-file
    │
    .so 原生机器码
    │
    ├── runtime/dgemm_wrapper.c  ← GPU/CPU 矩阵乘
    └── deliver.sh               ← 嵌入 scheme+boot+.so → ELF
    │
    ▼
单文件 ELF（5.2MB，只依赖 libc）
```

## 已覆盖的 ML C 库

cuBLAS / cuDNN / cuRAND / cuSOLVER / OpenBLAS / LAPACK
libcurl / libcjson / XGBoost / LightGBM / ONNX Runtime

全部运行时 dlopen，有就用，没有就跳过。

## 完成进度

### 工具链 ✅
- [x] compiler/translate.py — StaticPy → Scheme 翻译器
- [x] compiler/prelude.scm — 运行时基础库
- [x] compiler/stdlib.scm — ML 运行时（11 个 C 库按需加载）
- [x] runtime/dgemm_wrapper.c — 矩阵乘（GPU cuBLAS / CPU 纯 C）
- [x] build.sh — 一键编译 StaticPy → .so
- [x] deliver.sh — 一键打包 → 单文件 ELF

### SD 推理库 (sd_runtime/) ⬜
- [x] array_ops.static.py — 逐元素运算 + softmax + norm
- [x] nn_ops.static.py — Conv2d(im2col+dgemm) + Attention + up/downsample
- [x] unet_blocks.static.py — ResBlock + SpatialTransformer
- [x] samplers.static.py — DDIM / Euler / DPM++ / CFG
- [x] clip.static.py — CLIP tokenizer + transformer
- [x] model_loader.static.py — 权重加载
- [x] unet.static.py — UNet scaffold
- [x] vae.static.py — VAE decoder（conv_in + mid 已验证）
- [x] main.static.py — 主入口

### 已验证
- [x] ELF 编译打包管线完整（5.2MB，ldd=libc only）
- [x] conv2d(im2col+dgemm+bias) 正确
- [x] group_norm / silu / upsample_nearest 正确
- [x] VAE conv_in(16→512) + mid_blocks(512→512×2) 数值正确

### 待完成
- [ ] VAE decoder up blocks（需 encoder skip connections）
- [ ] UNet 真实权重接入
- [ ] CLIP 真实权重接入
- [ ] txt2img 端到端管线
- [ ] ControlNet / LoRA
- [ ] img2img / HiRes Fix
