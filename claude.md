# SDXL Pipeline 参考实现

## sdxl_pipeline.py

基于 ComfyUI 原生 `sample()` 的 SDXL txt2img 参考实现。

### 路径
```
/opt/static_comfyui/sdxl_pipeline.py
```

### 运行
```bash
/data/venv/bin/python /opt/static_comfyui/sdxl_pipeline.py
```

### 功能
- 加载 `/data/models/image/sd_xl_base_1.0.safetensors` 完整 checkpoint
- 使用 ComfyUI 的 `load_checkpoint_guess_config` 加载模型（UNet + CLIP + VAE）
- 20 步 Euler 采样，CFG=7.0，seed=42
- 输出：`/home/quqiufeng/python_reference.png`（1024×1024 PNG）

### 输出验证
```
$ file /home/quqiufeng/python_reference.png
PNG image data, 1024 x 1024, 8-bit/color RGB, non-interlaced
$ ls -lh /home/quqiufeng/python_reference.png
1.3M
```

### 依赖
- Python 3.12（`/data/venv/bin/python`）
- ComfyUI（`/opt/static_comfyui/ComfyUI/`）
- PyTorch + CUDA
- 模型文件（safetensors）

## img.sh（stable-diffusion.cpp）

基于 myimg-cli（stable-diffusion.cpp GGML 后端）的 SDXL 出图。

### 路径
```
/opt/static_comfyui/img.sh
```

### 运行
```bash
bash /opt/static_comfyui/img.sh
```

### 功能
- 加载 SDXL checkpoint（UNet + CLIP-G + CLIP-L 合并 safetensors）
- 使用 stable-diffusion.cpp GGML 后端推理
- 20 步 Euler + Karras 调度，CFG=7.0，seed=42
- 输出：`/tmp/output.png`

### 验证
```bash
$ bash img.sh  # 约12秒
$ file /tmp/output.png
PNG image data, 1024 x 1024, 8-bit/color RGB, non-interlaced
```

### 与 Python 参考的对比
| 指标 | sdxl_pipeline.py (ComfyUI) | img.sh (stable-diffusion.cpp) |
|------|---------------------------|-------------------------------|
| 后端 | PyTorch + ComfyUI | GGML + stable-diffusion.cpp |
| 速度 | ~6秒/20步 | ~12秒/20步 |
| 输出 | ~1.3MB PNG | ~2MB PNG |
| 依赖 | Python, PyTorch, ComfyUI | 独立 ELF 二进制 |

### 备注
- ComfyUI 输出是参考标准，C++ 引擎输出应与之一致
- 速度差异来自 GGML 后端 vs PyTorch cuDNN 优化
