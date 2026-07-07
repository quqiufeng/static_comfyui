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

### 备注
- 这是验证 C++ GGML 引擎输出的参考标准
- 采样速度：~3.3 it/s（RTX 3080, 20步约6秒）
- 不依赖 custom_nodes，纯 ComfyUI 核心 API
