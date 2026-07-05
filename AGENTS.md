# Static ComfyUI — AI 开发指引

## 项目目标
用 StaticPy（Python 子集编译器）+ `libtorch_std_helper.so`（C++ libtorch 封装）
1:1 重写 ComfyUI，编译为独立 ELF 二进制，零 pip 依赖。

## CLI 接口（main.static.py）
```
# 模式 1：直接命令行出图
comfycli --checkpoint /data/models/image/sd_xl_base_1.0.safetensors --prompt "cat" --output ./out.png

# 模式 2：执行 ComfyUI workflow JSON
comfycli workflow.json --output-dir ./output
```

## 模型文件位置
所有模型在 `/data/models/image/`：
- `sd_xl_base_1.0.safetensors` (6.5G), `clip_g.safetensors` (2.6G), `clip_l.safetensors` (1.6G), `ae.safetensors` (320M)
- `libtorch_std_helper.so` 已内置 `torch_std_safetensors_load` + `torch_std_sdxl_unet_forward`

## 已搭建的基础设施
- `staticpy/` — StaticPy 编译器 (`static_translate.py`)、运行时 (`prelude`/`stdlib`)、构建脚本
- `cpp/` — C++ 推理后端 (libtorch_std_helper.h/.cpp + build_torch_std_helper.sh)
- `build.sh` — 编译 ELF + C++ .so
- `deploy.sh` — 打包依赖 .so + GLIBC 兼容层 + SCP
- `design.md` — 架构设计、文件映射、C++ 依赖清单
- code search: ComfyUI 全量索引 (656 文件, 20,141 chunks, 2,633 函数)

## CUDA 显存管理（已就绪）
4 个 extern fn 已加到 `libtorch_std_helper` + `static_stdlib.scm` + `static_translate.py`：
- `torch.cuda_get_free_memory() → int`
- `torch.cuda_load_model(device, tensor) → tensor`
- `torch.cuda_unload_model(tensor)`
- `torch.cuda_soft_empty_cache()`

## StaticPy 关键约束
- 无类继承/多态 → dataclass + 组合
- 无 try/except → if 守卫 + Result 类型
- 无 lambda 闭包 / eval / exec
- `import torch` 自动映射为 `torch-*` Scheme 函数
- `extern fn foo(x: int, y: float) -> int from "torch_std"` 声明 C FFI

## 开发顺序（bottom-up）

### Phase 0: 基础设施（无 GPU 需求）
- [ ] folder_paths.static.py    路径管理
- [ ] cli_args.static.py        CLI 参数解析
- [ ] comfy_types.static.py     Node 类型定义

### Phase 1: 模型检测（纯逻辑，可直接翻译）
- [ ] supported_models_base.static.py  模型基类
- [ ] supported_models.static.py       模型注册表
- [ ] model_detection.static.py        state_dict → 架构识别
- [ ] model_sampling.static.py         sigma 调度
- [ ] latent_formats.static.py         潜空间缩放

### Phase 2: 模型加载 + 显存管理
- [ ] sd.static.py             safetensors 加载 → 模型对象
- [ ] model_base.static.py     模型基类
- [ ] model_management.static.py  GPU 调度
- [ ] lora.static.py           LoRA 合并

### Phase 3: 组件级推理
- [ ] clip_model.static.py     CLIP encode
- [ ] k_diffusion/sampling.static.py  采样器
- [ ] sample.static.py         采样入口
- [ ] controlnet.static.py     ControlNet

### Phase 4: 节点 → DAG → 入口
- [ ] nodes.static.py          200+ 节点定义
- [ ] execution.static.py      PromptExecutor
- [ ] main.static.py           CLI 入口

### Phase 5: 端到端验证
- [ ] workflow SDXL → 图片输出
- [ ] --prompt 命令行模式验证

## 命名约定
- ComfyUI 的 `a.py` → `a.static.py`
- 放在 `comfycli/` 目录下（尚未创建，从 Phase 0 起逐文件创建）
- `extern fn` 走 `from "torch_std"` 库

## 关键设计决策
- 命令版先不做 HTTP/WS，后续再补 UI 层
- C++ 层基本不需要新增 helper，libtorch_std_helper.so 已覆盖 90%
- 编译（build.sh）和部署（deploy.sh）职责分离
- ComfyUI 无自有 C/C++ 源文件，所有 C++ 来自外部 pip 包
