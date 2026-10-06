# 训练自己的 z_image LoRA（风格锁定）

> 目标：用几十张同风格图片训一个 **Z-Image LoRA**。以后出图直接
> `backup.sh ... --lora my_style.safetensors:0.7` 就带该风格，**不用再手写风格提示词**。

## 当前推荐配方（2026-10 实测定稿）

| 项 | 值 |
|----|----|
| 训练框架 | **musubi-tuner**（kohya-ss）|
| DiT 基座 | **Z-Image Base**（非蒸馏）`Tongyi-MAI/Z-Image`，bf16 |
| 文本编码器 / VAE | 与 Turbo **共享**（复用已下载的） |
| fp8 | **关**（`--fp8_base --fp8_scaled` 会掉质量） |
| 分辨率 | **1536**（bucketed） |
| 网络 | lora_zimage，`rank=alpha=16` |
| epochs/步数 | **8 epochs ≈ 448 步** |
| 显存 | 20G：`--blocks_to_swap 12` |
| 实测 | 11.4s/步，~1h25m；LoRA ~140MB |
| 出图 | `backup.sh ... --lora <lora_sdcpp.safetensors>:0.7` |

**关键结论**：Base 模型训练 → 保细节（发丝/皮肤纹理）；turbo+adapter+fp8 会磨皮。
详见 §9。

---

## 1. 出图用什么模型（决定 LoRA 挂哪个基座）

`cpp/sd/backup.sh`：

| 组件 | 默认文件 | 说明 |
|------|----------|------|
| 扩散模型 | `/data/models/image/z_image_turbo-Q5_K_M.gguf` | **Z-Image-Turbo**（~6B DiT，GGUF 量化） |
| VAE | `/data/models/image/ae.safetensors` | Z-Image VAE |
| 文本编码器 | `/data/models/image/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | Qwen3-4B |

→ 用 **Base 训 LoRA，挂到 Turbo 推理**（业界标准：非蒸馏 Base 的 latent dynamics 完整，风格向量纯净，挂 Turbo 后保留底模高频细节）。另有 `z_image_turbo-Q8_K_M.gguf`。

---

## 2. 环境（已搭好）

| 项 | 值 |
|----|----|
| GPU | RTX 3080 **20GB** |
| 训练 venv | **`/data/venv-musubi`**（`python3 -m venv --system-site-packages`） |
| torch | 2.6.0+cu126（软链自 `/data/venv`；musubi venv 自带被 `~/.local` CPU 版抢先，故软链 CUDA 版） |
| 库版本 | transformers **4.57.6** / diffusers **0.38.0** / huggingface_hub **0.36.2**（降到与 torch 2.6 匹配） |
| musubi-tuner | `/opt/musubi-tuner`（源码 `-e` 安装） |

---

## 3. 模型文件

| 角色 | 路径 | 说明 |
|------|------|------|
| **DiT (Base)** | `/data/models/z-image-base/transformer/diffusion_pytorch_model-00001-of-00002.safetensors` | `Tongyi-MAI/Z-Image`，12.3G，原始命名，musubi 直接加载 ✅ |
| VAE | `/data/models/z-image-turbo/vae/diffusion_pytorch_model.safetensors` | 与 Turbo 共享 ✅ |
| 文本编码器 | `/data/models/z-image-te-qwen3.safetensors` | diffusers 导出缺 `model.` 前缀，已加前缀转出（398 keys） |
| （旧）Turbo DiT | `/data/models/z-image-turbo/transformer/...` | 20G diffusers 基座，turbo+adapter 路线用（已淘汰） |
| （旧）Adapter | `/data/models/zimage_turbo_training_adapter_v2.safetensors` | turbo 直训用的训练 adapter（已淘汰） |

---

## 4. 数据集

`train_lora/prepare_dataset.py`：风格图目录 → 预缩（长边 ≤1536）+ Florence 自动打标。

产物 `/data/datasets/mystyle1536/`：
```
images/            56 张（≤1536）+ *.txt per-image caption（"mystyle, ..."）
dataset.toml
cache/             latent (*_zi.safetensors，按 1536 重算) + TE (*_zi_te.safetensors，跨分辨率复用)
```

`dataset.toml`：
```toml
[general]
resolution = [1536, 1536]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "/data/datasets/mystyle1536/images"
cache_directory = "/data/datasets/mystyle1536/cache"
num_repeats = 1
```

> caption 解耦（进阶）：把"磨皮/柔光"等特征**写进 caption**（如 `smooth skin, studio soft lighting`），
> 推理时只给触发词，风格就不会强行附带磨皮。见 §9。

---

## 5. 预缓存

```bash
PY=/data/venv-musubi/bin
M=/opt/musubi-tuner/src/musubi_tuner
D=/data/datasets/mystyle1536

# latent（VAE 编码；分辨率相关：换分辨率必须重算）
$PY/python $M/zimage_cache_latents.py --dataset_config $D/dataset.toml \
  --vae /data/models/z-image-turbo/vae/diffusion_pytorch_model.safetensors --device cuda

# 文本编码器输出（与分辨率无关，可跨数据集复用）
$PY/python $M/zimage_cache_text_encoder_outputs.py --dataset_config $D/dataset.toml \
  --text_encoder /data/models/z-image-te-qwen3.safetensors --batch_size 8 --device cuda
```

---

## 6. 训练（`train_lora/musubi_train.sh`）

```bash
DIT=/data/models/z-image-base/transformer/diffusion_pytorch_model-00001-of-00002.safetensors \
ADAPTER= FP8=0 BLOCKS=12 DIM=16 EPOCHS=8 \
DATA=/data/datasets/mystyle1536 OUT=/data/lora/mystyle_base \
bash train_lora/musubi_train.sh
```

环境变量：

| 变量 | 默认 | 说明 |
|------|------|------|
| `DIT` | Base DiT | DiT 权重（第一个分片） |
| `ADAPTER` | 空 | turbo 训练时才需要；Base 留空 |
| `DATA` | `/data/datasets/mystyle1536` | 数据集目录 |
| `OUT` | `/data/lora/mystyle_base` | 输出目录 |
| `DIM` / `LR` | 16 / 1e-4 | network_dim=alpha / 学习率 |
| `EPOCHS` / `STEPS` | 8 / 空 | 轮数；设 `STEPS` 则用 max_train_steps |
| `FP8` | 0 | 1=开 fp8（掉质量，默认关） |
| `BLOCKS` | 12 | `--blocks_to_swap`（省显存） |

脚本内固定：`--sdpa --mixed_precision bf16 --timestep_sampling shift --weighting_scheme none
--discrete_flow_shift 2.0 --optimizer_type adamw8bit --gradient_checkpointing
--network_module networks.lora_zimage`，训练后自动 `convert_lora.py --target other`。

---

## 7. 转换 + 出图

musubi 输出是 sd-scripts 命名（`lora_unet_...`），sd.cpp 要 **diffusers/PEFT**（`diffusion_model.*.lora_A/B`），脚本已自动转换：

```bash
$PY/python $M/convert_lora.py --input OUT/mystyle.safetensors \
  --output OUT/mystyle_sdcpp.safetensors --target other
```

出图：
```bash
./cpp/sd/backup.sh "mystyle style, a single woman portrait, elegant, soft light" \
  ~/out.png 2560 1440 --lora /data/lora/mystyle_base/mystyle_sdcpp.safetensors:0.7
# sd.cpp: (420 / 420) LoRA tensors have been applied
```

---

## 8. 工具清单（`train_lora/`）

| 文件 | 作用 |
|------|------|
| `musubi_train.sh` | **musubi 训练封装**（当前主用，支持 Base/turbo、fp8 开关、blocks_to_swap）+ 自动转名 |
| `prepare_dataset.py` | 风格图 → 预缩 + Florence 批量打标（`--batch`）+ `metadata.jsonl`/`.txt` |
| `run_train.sh` / `train_dreambooth_lora_z_image.py` / `qnbit4.json` | diffusers 路线（备选，已适配 0.38） |
| `convert_lora.py` | 通用 LoRA 命名转换（musubi 官方 `--target format other` 已够用） |

---

## 9. 质量调优记录（为什么 Base + 无 fp8）

| 配置 | 结果 |
|------|------|
| turbo + training_adapter + fp8 + rank32 + 16epochs + 1024 | 风格生效，但**皮肤/发丝明显磨皮/CG、丢细节** |
| **Base + 无 fp8 + rank16 + 8epochs + 1536** | **细节/质感显著恢复**，发丝皮肤纹理保留，风格仍生效 |

原因：
1. **数据本身是精修磨皮人像** → LoRA 学到"平滑皮肤"，盖掉底模真实纹理（用正则图/caption 解耦可缓解）。
2. **fp8** 截断权重尾数 → 高频细节梯度被量化噪声掩盖。
3. **rank32 + 16epochs** 过拟合。
4. **turbo+adapter** 强行把少步数轨迹拽回连续流 → 偏低频。
5. **分辨率 1024** 高频信息不足 → 2560 放大后更糊。

改进（已采纳）：换 Base、关 fp8、rank16/alpha16、8 epochs、1536 分辨率、blocks_to_swap 4。

仍可做：
- **caption 解耦**：把 `smooth skin / studio soft lighting` 写进 caption，推理只给触发词。
- **混入正则图**：15-20 张未修图高清人像（不带触发词，只写物理描述）防过拟合。
- **强度**：0.6~0.8 试；过高会漂构图（如 2560 偶发双人，prompt 加 `single woman` 规避）。

---

## 10. 快速复现（换一套风格图）

```bash
# 1) 数据（预缩 1536 + Florence 打标）
/data/venv/bin/python train_lora/prepare_dataset.py <风格图目录> /data/datasets/<name> \
  --trigger <trigger> --caption --max-side 1536
cp -r /data/datasets/<name> /data/datasets/<name>1536  # 或直接改 dataset.toml

# 2) 写 dataset.toml（resolution [1536,1536], image/cache_directory）

# 3) 预缓存 latent + TE（§5；TE 可复用）

# 4) 训练 + 自动转换（§6）
DIT=/data/models/z-image-base/transformer/diffusion_pytorch_model-00001-of-00002.safetensors \
FP8=0 BLOCKS=12 DIM=16 EPOCHS=8 DATA=/data/datasets/<name>1536 OUT=/data/lora/<name> \
bash train_lora/musubi_train.sh

# 5) 出图（§7）
./cpp/sd/backup.sh "<trigger> style, a single woman portrait, ..." ~/out.png 2560 1440 \
  --lora /data/lora/<name>/mystyle_sdcpp.safetensors:0.7
```
