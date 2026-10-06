# 训练自己的 z_image LoRA（风格锁定）

> 目标：用几十张同风格图片训一个 **Z-Image-Turbo LoRA**。以后出图直接
> `backup.sh ... --lora my_style.safetensors:0.8` 就带这个风格，**不用再手写风格提示词**。

---

## 1. 关键事实：出图用什么模型

`cpp/sd/backup.sh` 默认出图链路（决定 LoRA 必须挂在哪个基座上）：

| 组件 | 默认文件 | 说明 |
|------|----------|------|
| 扩散模型 | `/data/models/image/z_image_turbo-Q5_K_M.gguf` | **Z-Image-Turbo**（~6B DiT，GGUF 量化） |
| VAE | `/data/models/image/ae.safetensors` | Z-Image VAE |
| 文本编码器 | `/data/models/image/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | **Qwen3-4B**（LLM 文本编码） |

→ 因此 LoRA **只能训 Z-Image-Turbo** 这一支，训练基座与出图基座必须同源。
（本地另有 `z_image_turbo-Q8_K_M.gguf`，出图可替换，LoRA 通用。）

---

## 2. 可行性 / 现状

| 项 | 状态 |
|----|------|
| GPU | RTX 3080 **20GB** ✅ |
| torch 2.6+cu126 / diffusers 0.38 / peft / accelerate / bitsandbytes | ✅ 已装 |
| **diffusers 原生支持 z_image** | ✅ `pipelines/z_image/*`（txt2img/img2img/inpaint/controlnet）+ `ZImageLoraLoaderMixin` |
| **官方训练脚本** | ✅ `diffusers/examples/dreambooth/train_dreambooth_lora_z_image.py`（支持 `--bnb_quantization_config_path` 量化、`--do_fp8_training`、gradient checkpointing） |
| 训练基座（diffusers 格式） | ⬇️ `Tongyi-MAI/Z-Image-Turbo`（transformer ~24.6G fp32≈6B / text_encoder Qwen3-4B ~8G / vae 168M），也可以用 `T5B/Z-Image-Turbo-FP8` |
| torchvision | ⚠️ 当前 `0.19.0+cu121` 与 torch `2.6.0+cu126` **不匹配（已损坏）**，训练脚本 import 了 torchvision，需先修 |
| 网络 / 磁盘 | ✅（根分区 372G 空闲） |

20G 显存策略：**QLoRA（transformer 4bit）+ 文本编码器量化/offload + 梯度检查点**，batch 1 + 梯度累积。

---

## 3. 流程

```
风格图 ~30 张
   │  Florence-2 自动打标（cpp/florence2/img2prompt）
   ▼
metadata.jsonl  (file_name + text)
   │  train_dreambooth_lora_z_image.py（QLoRA）
   ▼
my_style.safetensors
   │  backup.sh --lora my_style.safetensors:0.8
   ▼
带风格的成图（配方 HiRes/FreSca/EasyCache 原样）
```

### 3.1 数据
- ~30 张同风格图放一个目录，统一长边（建议 ≥1024），去重、去水印。
- 打标：用我们已集成的 Florence-2 生成描述，或（动漫）用 WD14 tagger。
  - 生成 `metadata.jsonl`：每行 `{"file_name": "001.png", "text": "..."}`
  - 触发词（trigger token）在 caption 里统一加一个词（如 `mystyle`），出图时 prompt 带上它。

### 3.2 训练基座
```bash
# 下 diffusers 版基座（~33G fp32；或选 FP8 变体）
huggingface-cli download Tongyi-MAI/Z-Image-Turbo --local-dir /data/models/z-image-turbo
```

### 3.3 训练（示意）
```bash
# 依赖：diffusers examples 的训练脚本 + 其 requirements（含 prodigyopt 等）
accelerate launch train_dreambooth_lora_z_image.py \
  --pretrained_model_name_or_path /data/models/z-image-turbo \
  --instance_data_dir /data/datasets/mystyle \
  --output_dir /data/lora/mystyle \
  --instance_prompt "mystyle, <打标描述>" \
  --resolution 1024 \
  --train_batch_size 1 --gradient_accumulation_steps 4 \
  --rank 16 --learning_rate 1e-4 --max_train_steps 800 \
  --gradient_checkpointing --use_8bit_adam --mixed_precision bf16 \
  --bnb_quantization_config_path 8bit.json   # 或 --do_fp8_training
```

### 3.4 出图验证
```bash
cd cpp/sd
./backup.sh "mystyle, a woman portrait, soft light" ~/lora_test.png 2560 1440 \
  --lora /data/lora/mystyle/pytorch_lora_weights.safetensors:0.8
```

---

## 4. 待定 / 风险

1. **sd.cpp 能否加载 diffusers/PEFT 命名的 z_image LoRA** —— 待现场验证；
   不兼容则该步需要 key 转换（kohya 命名），或改用 diffusers 出图。
2. **torchvision 版本错配** —— 训练前修（装匹配 cu126 的 torchvision）或用不依赖它的路径。
3. **基座体积**：bf16 ~33G vs `T5B/Z-Image-Turbo-FP8`（更小）——按本地磁盘/速度选。
4. **20G 显存调参**：分辨率/rank/累积步数需实测；OOM 则降分辨率或提高量化等级。
5. **打标质量**：Florence 描述偏长，风格 LoRA 常用更短的触发词 + 少量描述；可人工微调。

---

## 5. 开放问题（待你确认）

- 用 **bf16 33G** 还是 **FP8** 基座？
- 风格样本目录路径？（给我图我就自动打标开训）
- 触发词用什么（如 `mystyle`）？
