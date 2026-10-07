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
| 扩散模型 | `/data/models/image/z_image_turbo-Q8_K_M.gguf` | **Z-Image-Turbo Q8**（~6B DiT，GGUF 量化；2026-10 起为默认） |
| VAE | `/data/models/image/ae.safetensors` | Z-Image VAE |
| 文本编码器 | `/data/models/image/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | Qwen3-4B |

→ 用 **Base 训 LoRA，挂到 Turbo 推理**（业界标准：非蒸馏 Base 的 latent dynamics 完整，风格向量纯净，挂 Turbo 后保留底模高频细节）。推理默认 **Q8**（细节/通透更好），另有 `z_image_turbo-Q5_K_M.gguf`（省 ~2G，细节略差）。

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
| `BLOCKS` | 12 | `--blocks_to_swap`（省显存）。**512 人脸数据集设 `0` 可提速约 4 倍**（见 §16） |

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

出图（**触发词自动注入**，无需手写）：
```bash
# backup.sh 会按 /data/lora/lora_triggers.conf 自动把触发词前置到默认提示词
./cpp/sd/backup.sh 2560 1440 \
  --lora /data/lora/mystyle_base/mystyle_sdcpp.safetensors:0.7
# 输出: ✓ LoRA 触发词注入: mystyle
#       Prompt: mystyle, solo,single woman,half body portrait...
# sd.cpp: (420 / 420) LoRA tensors have been applied
```

触发词映射（`/data/lora/lora_triggers.conf`，可用 `LORA_TRIGGERS_FILE` 覆盖；模板见
`train_lora/lora_triggers.conf.example`）：
```
<lora 文件名> = <触发词>
mystyle_sdcpp.safetensors = mystyle
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
./cpp/sd/backup.sh 2560 1440 --lora /data/lora/<name>/mystyle_sdcpp.safetensors:0.7
```

---

## 11. 指定人脸 + 风格：双 LoRA 方案（需求与实现思路）

### 11.1 需求
想要**高清大图 + 指定某个人的脸 + 指定风格**。

### 11.2 约束
- 出图基座 z_image 是 **DiT**；IPAdapter-Face / InstantID / PuLID / PhotoMaker 等"人脸注入"方案基本是 **UNet 时代**的，DiT 上没有。
- 因此 DiT 上"指定人脸"的可行路线 = **人物 LoRA**（社区已验证，z_image 生态有大量人物 LoRA）。

### 11.3 方案：训两个 LoRA，出图叠加
**风格 LoRA + 人脸 LoRA 用两套图分开训**，出图时同时挂载，互不干扰、可自由组合（"这张脸 × 任意风格"）。

| | 风格 LoRA（已有 `mystyle`） | 人脸 LoRA（新增） |
|---|---|---|
| 数据 | 同风格、**不同人/不同内容** | **同一个人**，不同服装/背景/角度/表情 |
| 数量 | 30~60 | 20~50（多正面/半侧、清晰少遮挡） |
| caption | 描述内容，触发词绑风格 | `celebA, <服装/背景/表情/光照>`，**身份只归触发词**，不写五官 |
| rank/alpha | 16 | **32~64**（脸更吃容量） |
| epochs/步数 | 8（~400-500） | 12~20（~800-1500） |
| 正则 | 可选 | 强烈建议（普通人/同类别图，不带触发词） |

解耦要点：**风格集人脸多样**（否则风格带上某张脸）；**人脸集画风多样**（否则脸带上某画风）。

### 11.4 出图叠加 + 触发词注入
```bash
./cpp/sd/backup.sh 2560 1440 \
  --lora /data/lora/celeba_base/mystyle_sdcpp.safetensors:0.7 \
  --lora /data/lora/mystyle_base/mystyle_sdcpp.safetensors:0.6
```
- 两个 LoRA 权重**别都拉满**（0.5~0.7 起调），过高会在同区块互相打架出伪影。
- 触发词映射扩展为多行注入（`celebA` + 风格词），`backup.sh` 逐个 `--lora` 查找并前置。

### 11.5 待实现清单
1. **`backup.sh` 支持多个 `--lora`**：`--lora A:w --lora B:w`（逗号分隔或重复），多触发词按序前置到 prompt。
2. **`prepare_dataset.py` 人物模式**：`--mode face` → caption 模板 `celebA, <可变描述>`（身份中性）；可选 `--face-crop`（裁到脸）与 `--reg-dir`（掺正则图）。
3. **可选：人脸区域精修**：HiRes 后对脸局部再重绘一次（sd.cpp ADetailer / face-restore），让 2560 下的脸更锐；身份仍来自 LoRA。
4. **多 LoRA 触发词映射**：`/data/lora/lora_triggers.conf` 支持两条（人脸 + 风格）。

### 11.6 风险 / 备选
- 两 LoRA 冲突（同区块抵消/伪影）→ 降权重；仍不行则用"人脸图 + 风格图混合"训**联合 LoRA**（可复用性差）。
- 身份相似度不足 → 提高 rank、增加脸部特写占比、加正则、多步早停对比。
- 合规：真实人物肖像权/用途由使用者自负。

---

## 12. 一句话总结

- **高清大图** = `backup.sh` HiRes 管线（Base + bf16 + 1536 + 不过拟合的 LoRA）。
- **指定人脸** = 该人的**人物 LoRA**（DiT 无人脸 adapter，LoRA 即正路）。
- **风格 + 人脸** = 两套图训两个 LoRA，出图叠加（多 `--lora`）。

---

## 13. 人脸图片集合制作（cpp/face 扣脸）→ 人物/审美 LoRA

> 下次只要**找到某个人的图片集合**，跑下面流程即可出"对齐人脸"训练集，再训练人脸 LoRA。

### 13.1 工具（`cpp/face/`）
```
build.sh 编译 → facecli / libface.so
facecli detect <img>                       # 人脸框+分数 (JSON)
facecli crop   <img> <out.png> [size=512]  # 检测+5点对齐裁剪（缩到 size）
facecli parse  <img> <out.png> [color|face|skin|hair]   # 人脸解析（在裁剪图上跑最准）
```
模型：SCRFD `det_10g.onnx`（本地）+ `faceparser.onnx`（BiSeNet，`/data/models/face/`）。

### 13.2 一键制作数据集
```bash
# 传入"某个人的图片目录"（jpg/png 皆可），自动扣脸 + 打标 + 写 dataset.toml
bash train_lora/make_face_dataset.sh <人物图片目录> <触发词>
# 例：
bash train_lora/make_face_dataset.sh /data/celebA_photos celebA
# → /data/datasets/celebA_face/{images/*.png(512对齐脸), *.txt, dataset.toml}
```
- **同一个人**的图 → 训出**身份 LoRA**（脸越来越像这个人）。
- **不同人**的美图集合 → 训出**"审美/风格脸" LoRA**（生成的脸偏这个风格，不绑定具体人）。
- 无脸的图自动跳过；建议 20~50 张、角度/表情/光照多样、换装换背景。

### 13.3 预缓存 + 训练（人物建议 rank32）
```bash
D=/data/datasets/celebA_face
PY=/data/venv-musubi/bin; M=/opt/musubi-tuner/src/musubi_tuner
$PY/python $M/zimage_cache_latents.py --dataset_config $D/dataset.toml \
  --vae /data/models/z-image-turbo/vae/diffusion_pytorch_model.safetensors --device cuda
$PY/python $M/zimage_cache_text_encoder_outputs.py --dataset_config $D/dataset.toml \
  --text_encoder /data/models/z-image-te-qwen3.safetensors --batch_size 8 --device cuda

DIT=/data/models/z-image-base/transformer/diffusion_pytorch_model-00001-of-00002.safetensors \
  FP8=0 BLOCKS=0 DIM=32 EPOCHS=10 DATA=$D OUT=/data/lora/celebA OUT_NAME=celebA \
  bash train_lora/musubi_train.sh
# → /data/lora/celebA/celebA_sdcpp.safetensors
```

### 13.4 出图（触发词自动注入）
在 `/data/lora/lora_triggers.conf` 加一行 `mystyle_sdcpp.safetensors = celebA`（或把训练输出的
LoRA 重命名为 `<触发词>_sdcpp.safetensors`），然后：
```bash
./cpp/sd/backup.sh 2560 1440 --lora /data/lora/celebA/mystyle_sdcpp.safetensors:0.7
```

> 说明：`musubi_train.sh` 的 LoRA 名由 `OUT_NAME`（默认 `mystyle`）决定；建议设成触发词，
> 如 `OUT_NAME=celebA`，输出 `celebA_sdcpp.safetensors`，映射一目了然。

---

## 14. 单明星身份 LoRA 专项方案

> 目标：让生成的脸**稳定像某个具体明星**。DiT 无人脸 adapter（IPAdapter-Face/InstantID/PuLID 都是 UNet 时代的），
> 所以正路就是**用这个人的照片训一个身份 LoRA**。与风格 LoRA 分开训、出图叠加。

### 14.1 核心原则
- **只能用同一个人的照片**（是谁 → 像谁）；混入其他人会"脸糊成一团"，相似度下降。
- **身份归触发词**（如 `celebA`），caption 只描述可变因素（服饰/背景/表情/光照）→ 出图只给触发词即锁定这张脸。

### 14.2 收集数据（人工）
| 项 | 要求 |
|----|------|
| 数量 | 20~50 张（质量优先，宁精勿滥） |
| 内容 | **同一个人**；角度（正/半侧/侧）、表情、光照、远近、换装换背景多样 |
| 排除 | 他人、重滤镜/重磨皮、遮脸（口罩/手/大墨镜）、糊图、低清、多脸同框（会自动取最大脸，但尽量单人） |
| 分辨率 | 越高越好（≥1024 更好，脚本会裁到 512 对齐脸） |

### 14.3 制作对齐脸数据集（cpp/face）
```bash
# 一键：检测 → 5点对齐裁剪(512) → Florence 打标（身份中性的内容描述）
bash train_lora/make_face_dataset.sh /path/to/star_photos celebA
# → /data/datasets/celebA_face/{images/*.png(512对齐脸), *.txt, dataset.toml}
```
> caption 形如 `celebA, A woman with long black hair wearing a red dress.`——描述服装/背景，身份留给触发词，
> 正是身份 LoRA 需要的"解耦"。

### 14.4 预缓存 + 训练（身份建议 rank32~64、步数偏多）
```bash
D=/data/datasets/celebA_face
PY=/data/venv-musubi/bin; M=/opt/musubi-tuner/src/musubi_tuner
$PY/python $M/zimage_cache_latents.py --dataset_config $D/dataset.toml \
  --vae /data/models/z-image-turbo/vae/diffusion_pytorch_model.safetensors --device cuda
$PY/python $M/zimage_cache_text_encoder_outputs.py --dataset_config $D/dataset.toml \
  --text_encoder /data/models/z-image-te-qwen3.safetensors --batch_size 8 --device cuda

DIT=/data/models/z-image-base/transformer/diffusion_pytorch_model-00001-of-00002.safetensors \
  FP8=0 BLOCKS=0 DIM=32 EPOCHS=12 OUT_NAME=celebA DATA=$D OUT=/data/lora/celebA \
  bash train_lora/musubi_train.sh
# → /data/lora/celebA/celebA_sdcpp.safetensors（512 + BLOCKS=0，约 29 分钟，见 §16）
```
| 超参 | 身份 LoRA 建议 |
|------|----------------|
| rank/alpha | 32（不够像 → 64） |
| epochs | 10~16（每 epoch 存 checkpoint，横向对比选最优） |
| 分辨率 | 512（对齐脸原生） |
| lr | 1e-4 |
| fp8 | 关 |
| blocks_to_swap | **0**（512 显存充裕，比 12 快约 4 倍；见 §16） |

### 14.5 触发词映射
`/data/lora/lora_triggers.conf` 加一行：
```
celebA_sdcpp.safetensors = celebA
```

### 14.6 出图
```bash
# 单独：只出这张脸
./cpp/sd/backup.sh 2560 1440 --lora /data/lora/celebA/celebA_sdcpp.safetensors:0.7

# 叠加风格 LoRA（脸 × 风格）
./cpp/sd/backup.sh 2560 1440 \
  --lora /data/lora/celebA/celebA_sdcpp.safetensors:0.7 \
  --lora /data/lora/mystyle_base/mystyle_sdcpp.safetensors:0.6
```

### 14.7 相似度不够时的调法
1. **rank 32→64**、epochs 12→16；多存 checkpoint 挑最像但不崩的。
2. 数据里**脸部特写占比提高**（或 `--max-side` 更小、脸更满）。
3. **出图权重** 0.7~0.9 上调。
4. **剔除重滤镜样本**（磨皮会让身份特征丢失）。
5. 加**人脸区域精修**（ADetailer 式局部重绘，待做）让 2560 下的脸更锐、更像。
6. 数据不足（<15）时先补图，别硬训。

### 14.8 合规
真实人物肖像权与用途由使用者自负。

---

## 15. 相关文件速查
| 路径 | 说明 |
|------|------|
| `cpp/face/facecli` | 检测/对齐裁剪/解析 CLI |
| `train_lora/make_face_dataset.sh` | 人物图片目录 → 对齐脸数据集 |
| `train_lora/musubi_train.sh` | LoRA 训练（`OUT_NAME` 命名，自动转 sd.cpp 命名） |
| `/data/lora/lora_triggers.conf` | LoRA→触发词映射（backup.sh 自动注入） |
| `/data/lora/<name>/<OUT_NAME>_sdcpp.safetensors` | 训练产出（sd.cpp 可直接加载） |

---

## 16. 速度调优（2026-10 实测，512 人脸数据集）

> **结论：512 人脸/身份 LoRA 用 `BLOCKS=0`（不换出），约 29 分钟训完，
> 比旧配方 `BLOCKS=12` 快约 4 倍。** 旧参数一直要 ~50 分钟。

测试环境：RTX 3080 20G；98 张 512 对齐脸；Z-Image Base bf16；rank32；
12 epochs（1176 步）；`--gradient_checkpointing` 开；fp8 关。

| `BLOCKS` | 显存 | 步速 | 12 epochs 预计 |
|----------|------|------|----------------|
| 12（旧默认） | ~10.3G | 6.15 s/步 | ~2h |
| 4 | ~13.1G | 2.47 s/步 | ~47min |
| **0（新推荐）** | **~14.5G** | **1.57 s/步** | **~29min** |

原理：
1. **block swap 是主要瓶颈**：每次前向/反向都要把换出的 block 在 CPU↔GPU 间搬运。
   实测每多换出 1 个 block ≈ +0.47 s/步（12→6.15、4→2.47 线性外推）。
2. **512 数据集显存充裕**：Base DiT bf16 全驻留仅 ~12.3G，加激活/优化器共 ~14.5G，
   20G 卡完全放得下，无需换出。
3. **不能关 `--gradient_checkpointing`**：实测关掉后 PyTorch 占用 18.6G 仍 OOM
   （`torch.OutOfMemoryError ... 16.00 MiB`），必须保留 GC。

何时仍用 `BLOCKS=12`：1536 风格数据集（§9）显存不够，必须换出。

命令（身份 LoRA，见 §14.4）：
```bash
DIT=/data/models/z-image-base/transformer/diffusion_pytorch_model-00001-of-00002.safetensors \
  FP8=0 BLOCKS=0 DIM=32 EPOCHS=12 OUT_NAME=<trigger> \
  DATA=/data/datasets/<name>_face OUT=/data/lora/<trigger> \
  bash train_lora/musubi_train.sh
```

> 注意：`setsid bash ... &` 让训练脱离终端会话运行，避免被父 shell/工具超时杀掉；
> 日志重定向到文件后轮询进度即可。

---

## 17. 训练配方对比结论（2026-10 实测：训练无显著增益，Q8 才有效）

为提升 liuhaocun 身份 LoRA 的"像"程度，做了 A/B：

| 配方 | caption | rank | epochs | 数据 |
|------|---------|------|--------|------|
| BASE | Florence 描述式（`liuhaocun, A woman with long black hair...`） | 32 | 12 | `/data/datasets/liuhaocun_face`（98） |
| V1 | **极简，只写 `liuhaocun`** | 32 | 10 | `/data/datasets/liuhaocun_min`（98） |
| V2 | 极简，只写 `liuhaocun` | **64** | 10 | `/data/datasets/liuhaocun_min`（98） |

评估：`backup.sh` **2560×1440**，**Q8** 模型，同一 seed，双 LoRA
（`liuhaocun:X` + `mystyle:0.6`）。LoRA：`/data/lora/exp/liuhaocun_min_r{32,64}_sdcpp.safetensors`；
对比图 `~/cmpq8_{BASE,V1,V2}_*.png`、竖版 `~/vq8_{BASE,V1,V2}_*.png`。

**结论**：
1. **2560×1440 下 BASE / V1 / V2 几乎无差别** → caption 解耦、rank 32→64
   都没有可感知收益（数据与身份已足够）。
2. **Q8 明显优于 Q5**（细节/通透），是真正有效的杠杆 —— 优先用
   `DIFFUSION_MODEL=/data/models/image/z_image_turbo-Q8_K_M.gguf`。
3. 想更"像"应走**推理侧**：LoRA 权重 0.7→0.9、CFG 2.0→3.0、或人脸局部重绘
   （ADetailer 式），继续调训练配方收益很低。

---

## 18. 出图尺寸预设（小红书图 / 朋友圈图）

`backup.sh` 内置 `--preset` 社媒尺寸（Q8 默认，双 LoRA 照常）：

| preset | 别名 | 尺寸 | 比例 | 用途 |
|--------|------|------|------|------|
| `xhs` | `小红书` / `xiaohongshu` | **1920×2560** | 3:4 | 小红书竖图 |
| `pyq` | `朋友圈` / `moments` | **2048×2048** | 1:1 | 朋友圈方图（约 2.5min/张） |

```bash
./cpp/sd/backup.sh --preset xhs ~/xhs.png \
  --lora /data/lora/liuhaocun/liuhaocun_sdcpp.safetensors:0.7 \
  --lora /data/lora/mystyle_base/mystyle_sdcpp.safetensors:0.6

./cpp/sd/backup.sh --preset pyq ~/pyq.png \
  --lora /data/lora/liuhaocun/liuhaocun_sdcpp.safetensors:0.7 \
  --lora /data/lora/mystyle_base/mystyle_sdcpp.safetensors:0.6
```

说明：不给 preset 时默认 2560×1440 横版；也可直接传像素，如 `backup.sh 1920 2560 ...`。
朋友圈统一用 2048×2048（1440×1440 偏糊，2560² 太慢）。

---

## 19. 出图参数内嵌 + 参考图复刻（ComfyUI 风格）

每张出图都会把生成参数写入 PNG `tEXt` 块 `parameters`（`img_hires` 实现），
含：prompt / negative / steps / hires_steps / cfg / seed / method / scheduler /
尺寸（最终 + base）/ hires 参数 / 模型路径 / **lora 列表** / fresca / cache / 后处理。

```bash
# 查看某张图的参数
./cpp/sd/build/img_hires --dump-meta ~/out.png

# 从参考图复刻（读回 prompt/seed/steps/cfg/sampler/lora/尺寸，重出一张）
./cpp/sd/backup.sh --from-image ~/out.png ~/new.png
# 说明：CLI/env 显式给的参数优先，未给的用参考图里的
```

用途：忘了用过的 prompt / 参数，直接拿参考图即可复刻。旧图（本功能之前生成的）没有该元数据。

