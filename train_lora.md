# 训练自己的 z_image LoRA（风格锁定）

> 目标：用几十张同风格图片训一个 **Z-Image-Turbo LoRA**。以后出图直接
> `backup.sh ... --lora my_style.safetensors:0.8` 就带该风格，**不用再手写风格提示词**。
>
> **当前训练架构：musubi-tuner（方案 B）**，比 diffusers 官方脚本快 ~3.4×（6.6s/步 vs 23s/步），
> 已端到端跑通（训练 → 转换 → sd.cpp `--lora` 420/420 生效）。diffusers 路线保留为备选。

---

## 1. 出图用什么模型（决定 LoRA 挂哪个基座）

`cpp/sd/backup.sh`：

| 组件 | 默认文件 | 说明 |
|------|----------|------|
| 扩散模型 | `/data/models/image/z_image_turbo-Q5_K_M.gguf` | **Z-Image-Turbo**（~6B DiT，GGUF 量化） |
| VAE | `/data/models/image/ae.safetensors` | Z-Image VAE |
| 文本编码器 | `/data/models/image/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | Qwen3-4B |

→ LoRA 必须训 **Z-Image-Turbo** 这一支，才能挂到 backup.sh。本地另有 `z_image_turbo-Q8_K_M.gguf`。

---

## 2. 环境（已搭好）

| 项 | 值 |
|----|----|
| GPU | RTX 3080 **20GB** |
| 训练 venv | **`/data/venv-musubi`**（`python3 -m venv --system-site-packages`） |
| torch | 2.6.0+cu126（软链自 `/data/venv`，musubi venv 自带被 `~/.local` CPU 版抢先，故软链 CUDA 版） |
| 库版本 | transformers **4.57.6** / diffusers **0.38.0** / huggingface_hub **0.36.2**（降到与 torch 2.6 匹配；musubi 默认装的 5.17/0.40/1.32 需要更新 torch，故降级） |
| musubi-tuner | `/opt/musubi-tuner`（源码 `-e` 安装） |

---

## 3. 模型文件（复用 20G diffusers 基座 + adatper）

musubi 从各组件 safetensors 加载，**可复用** `dimitribarbot/Z-Image-Turbo-BF16`（`/data/models/z-image-turbo`）：

| 角色 | 路径 | 说明 |
|------|------|------|
| DiT | `/data/models/z-image-turbo/transformer/diffusion_pytorch_model-00001-of-00002.safetensors` | 原始命名（`all_x_embedder`/`layers.*`），musubi 直接加载 ✅ |
| VAE | `/data/models/z-image-turbo/vae/diffusion_pytorch_model.safetensors` | `All keys matched successfully` ✅ |
| 文本编码器 | `/data/models/z-image-te-qwen3.safetensors` | diffusers 导出缺 `model.` 前缀，已加前缀转出（398 keys） |
| 训练 adapter | `/data/models/zimage_turbo_training_adapter_v2.safetensors` | `ostris/zimage_turbo_training_adapter`，**turbo 直训不稳，靠它稳定**；训出的 LoRA 直接挂 turbo |

> turbo 是蒸馏模型，musubi 文档明确直训不稳，推荐用 base / De-Turbo / 训练 adapter。选 adapter 以匹配 backup.sh 的 turbo。

---

## 4. 数据集

`train_lora/prepare_dataset.py`：把风格图目录 → 预缩（长边 ≤1536）+ Florence 自动打标。

产物 `/data/datasets/mystyle/`：
```
images/            56 张（≤1536）
images/*.txt       per-image caption（"mystyle, ..."）
metadata.jsonl
dataset.toml       musubi 数据集配置（image_directory 直挂 [[datasets]]）
cache/             latent (*_zi.safetensors) + TE (*_zi_te.safetensors)
```

`dataset.toml`：
```toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "/data/datasets/mystyle/images"
cache_directory = "/data/datasets/mystyle/cache"
num_repeats = 1
```

---

## 5. 预缓存（musubi 的关键提速）

```bash
PY=/data/venv-musubi/bin
M=/opt/musubi-tuner/src/musubi_tuner

# latent（VAE 编码，一次 ~25s/56 张）
$PY/python $M/zimage_cache_latents.py \
  --dataset_config /data/datasets/mystyle/dataset.toml \
  --vae /data/models/z-image-turbo/vae/diffusion_pytorch_model.safetensors --device cuda

# 文本编码器输出（~8s/56 张）
$PY/python $M/zimage_cache_text_encoder_outputs.py \
  --dataset_config /data/datasets/mystyle/dataset.toml \
  --text_encoder /data/models/z-image-te-qwen3.safetensors --batch_size 8 --device cuda
```

---

## 6. 训练

`train_lora/musubi_train.sh`（一键：训练 + 自动转 sd.cpp 命名）：

```bash
OUT=/data/lora/mystyle_musubi DIM=32 EPOCHS=16 bash train_lora/musubi_train.sh
```

关键参数（脚本内）：
```
--dit <DiT 分片> --vae <VAE> --text_encoder <TE> --base_weights <adapter>
--sdpa --mixed_precision bf16
--timestep_sampling shift --weighting_scheme none --discrete_flow_shift 2.0
--optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing
--fp8_base --fp8_scaled            # ⚠️ 见 §8，会掉质量，建议去掉
--network_module networks.lora_zimage --network_dim 32 --network_alpha 32
--max_train_epochs 16 --save_every_n_epochs 1 --seed 42
```

实测：56 图 / batch1 / 896 步 ≈ **6.6s/步，~1h40m**。

---

## 7. 转换 + 出图

musubi 输出是 sd-scripts 命名（`lora_unet_...`），sd.cpp 要 **diffusers/PEFT 命名**（`diffusion_model.*.lora_A/B`）：

```bash
$PY/python /opt/musubi-tuner/src/musubi_tuner/convert_lora.py \
  --input /data/lora/mystyle_musubi/mystyle.safetensors \
  --output /data/lora/mystyle_musubi/mystyle_sdcpp.safetensors --target other
# → diffusion_model.layers.0.attention.to_k.lora_A/B.weight ...
```

出图：
```bash
./cpp/sd/backup.sh "mystyle style, <主体>" ~/out.png 2560 1440 \
  --lora /data/lora/mystyle_musubi/mystyle_sdcpp.safetensors:0.8
```
sd.cpp 日志：`(420 / 420) LoRA tensors have been applied` ✅

---

## 8. 已知问题：LoRA 出图偏"磨皮/模糊"

现象：带 LoRA 的皮肤/发丝比底模（不带 LoRA）**明显更平、细节更少**。LoRA 权重本身正常（abs max ~0.03）。

原因（按主次）：
1. **数据本身是精修/磨皮人像** → LoRA 忠实学到"平滑皮肤"，盖掉底模真实纹理（风格使然）。
2. **fp8 训练**（`--fp8_base --fp8_scaled`）→ musubi 明确会掉质量。
3. **rank32 + 16 epochs** 过拟合，强度 0.6~0.8 偏强。

改进方向：
- **去 fp8 重训**：去掉 `--fp8_base --fp8_scaled`，rank 16、epochs 8~10、lr 5e-5；显存不够用 `--blocks_to_swap`。
- **换数据**：用未磨皮/高细节样本（最有效）。
- **推理强度** 降到 0.5~0.6，或用 Q8 底模。

---

## 9. 工具清单（`train_lora/`）

| 文件 | 作用 |
|------|------|
| `musubi_train.sh` | **musubi 训练封装**（当前主用）+ 训练后自动转名 |
| `prepare_dataset.py` | 风格图 → 预缩 + Florence 打标 + `metadata.jsonl` + `.txt` |
| `run_train.sh` | diffusers 路线一键（备选） |
| `train_dreambooth_lora_z_image.py` | diffusers 官方脚本（备选，已打补丁适配 0.38） |
| `qnbit4.json` | bitsandbytes 4bit 配置（diffusers 用） |
| `convert_lora.py` | 通用转换（musubi 官方 `--target other` 已够用，此脚本备选） |

---

## 10. 快速复现（换一套风格图）

```bash
# 1) 数据
/data/venv/bin/python train_lora/prepare_dataset.py <风格图目录> /data/datasets/<name> \
  --trigger <trigger> --caption --max-side 1536

# 2) 写 dataset.toml（改 image_directory/cache_directory）

# 3) 预缓存（latent + TE，见 §5）

# 4) 训练 + 转换
OUT=/data/lora/<name> DIM=16 EPOCHS=8 bash train_lora/musubi_train.sh

# 5) 出图
./cpp/sd/backup.sh "<trigger> style, ..." ~/out.png 2560 1440 \
  --lora /data/lora/<name>/mystyle_sdcpp.safetensors:0.6
```
