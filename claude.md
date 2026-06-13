# static_comfyui — 工程原则与对接规范

## 1. 项目定位

`static_comfyui` 不是重新发明一个扩散框架，而是 **ComfyUI 推理管线的二进制编译版**。所有模型结构、算子语义、采样逻辑、 Conditioning 拼接方式必须与 `comfyui_ref/` 中的 ComfyUI 源码 **1:1 对齐**。

> 目标：ComfyUI 的 workflow JSON 能直接映射到 StaticPy 生成管线上；Python 节点能改成 StaticPy 函数后得到逐张量一致的输出。

## 2. 核心原则

### 2.1 源码优先，禁止脑补
- 任何新模块（CLIP、VAE、UNet、Sampler、ControlNet、IPAdapter 等）实现前，必须先在 `comfyui_ref/comfy/` 找到对应源码。
- 结构、超参、epsilon、激活函数、reshape/transpose 顺序、head split 方式都要按源码来。
- 参考实现脚本也要用同样的源码逻辑，不能另行写一套。

### 2.2 数值对齐是第一验收标准
- 每个子模块实现后，必须跟一个 **ComfyUI 源码逻辑一致的 PyTorch reference** 对比到 `max_diff < 1e-3`（默认 float32）。
- 先对齐子模块，再串管线；不准“大概对”就往下走。

### 2.3 静态化不是重写
- StaticPy 代码是 ComfyUI 节点/模块的直接翻译：同模块名、同 forward 流程、同张量名（在合理范围内）。
- 权重文件的组织方式沿用 ComfyUI 的命名空间（如 `clip_l.text_model...`、`first_stage_model.encoder...`），方便直接用 ComfyUI 导出的 state_dict。

### 2.4 兼容 ComfyUI 节点生态
- 后续 ControlNet、IPAdapter、FreeU、SAG、T2IAdapter、LoRA 等都要能找到对应 ComfyUI 节点/模块源码并对齐实现。
- Conditioning 格式、latent 格式、pooled 输出格式与 ComfyUI 一致。

## 3. 当前进度与待对齐项

| 模块 | 状态 | 说明 |
|------|------|------|
| UNet | ✅ 已对齐 | 按 `comfy/ldm/modules/diffusionmodules/openaimodel.py` 结构实现；数值与参考一致 |
| CLIP tokenizer/encoder | 待做 | 参考 `comfy/clip_model.py`、`comfy/sd1_clip.py` |
| VAE encoder/decoder | 待做 | 参考 `comfy/ldm/modules/diffusionmodules/model.py` |
| Sampler | 待做 | 参考 `comfy/k_diffusion/sampling.py` |
| txt2img 端到端 | 待做 | 按 ComfyUI Conditioning/采样/VAE decode 流程串联 |

## 4. 参考模型路径

```
/data/models/image/
├── sd_xl_base_1.0.safetensors    ← SDXL UNet + 内置 VAE
├── ae.safetensors                 ← SDXL VAE（可选外部）
├── clip_l.safetensors             ← CLIP-L
├── clip_g.safetensors             ← CLIP-G
```

## 5. 禁止事项
- ❌ 不查源码就写模块
- ❌ 用“简化版” tokenizer / attention / VAE 充数
- ❌ 自己定义 Conditioning 格式
- ❌ 以“能跑”代替“数值对”

## 6. 权重管理

- 优先使用 `comfyui_ref` 源码加载 Safetensors，提取 state_dict。
- 导出工具 `export_sd_weights.py` / `merge_weights.py` 只负责把 ComfyUI 的 state_dict 转成 float32 单 `weights.bin` + offset 索引。
- 索引里的 `name` 保持 ComfyUI 原始 key（可带 `model.diffusion_model.`、`clip_l.`、`first_stage_model.` 等前缀）。

---

> 一句话：我们要的是一个**能编译成单文件 ELF 的 ComfyUI**，而不是一个跑通 demo 的自研推理引擎。
