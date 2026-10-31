# 训练自己的 z_image LoRA（风格锁定）

> 目标：用几十张同风格图片训一个 **Z-Image LoRA**。以后出图直接
> `backup.sh ... --lora my_style.safetensors:1.2` 就带该风格，**不用再手写风格提示词**。
> 注意：权重不是固定 0.7；本机风格 LoRA 实测 **0.7 几乎无感，v5 定稿 1.2**（见 §7.1）。
> **训练规模已定稿：300 张高质量精选图 / 7 epochs（2100 步）即可**，继续加图没有明显收益。
> **人脸 LoRA 已定稿：4 epochs 足够**（不必 8–12），512 对齐脸 / rank32 / `BLOCKS=0`；
> 批量脚本 `train_lora/train_faces_batch.sh`（见 §14.9）。

## 训练任务（只有两类，2026-10-25 明确）

| 任务 | 数据 | 图片尺寸 | 产出 |
|---|---|---|---|
| **风格 LoRA（1 个）** | **所有图片合并**（各明星子目录全部汇总） | **1536**（长边，保持原比例，不裁脸） | 一个 style LoRA，学整体画风/质感 |
| **人脸 LoRA（每个明星 1 个）** | **各自的明星子目录**，扣脸 | **512×512**（5 点对齐脸） | 每个明星一个 identity LoRA |

- **两套尺寸不要混**：风格集用 1536，人脸集用 512；分别生成各自的 `images/` 与 `dataset.toml`。
- 风格集要混入所有人 → 风格向量与人脸解耦；出图按需叠加：
  `--lora style:0.6 --lora face:0.7`。
- 不做“风格+人脸联合训练”，也不按明星重复训风格。

## 工具策略（2026-10-25：告别 Python 脚本）

图像 / 数据准备一律用 **StaticPy 二进制工具**（`/opt/StaticPy`），不再新增或沿用 Python 脚本：

1. **缺工具 → 先用 StaticPy 封装**：C ABI（`.so`）+ FFI + 薄 CLI → 独立 ELF；
   优先复用已有的 `imgproc` / `face` / `birefnet` / `florence2` / `hashlib` 等库。
2. **管线用 shell 编排**（检测 → 筛 → 裁 → 打标 → 写 `dataset.toml`），产物零 Python/pip。
3. **允许 Python 的仅两处**：musubi 训练本体（`musubi_train.sh` 内部调用）、
   以及 `difftest/coverage` 这类“测量对照”（不进产物）。
4. 现存 Python 脚本逐步迁到 StaticPy：`curate_dataset.py`、`prepare_dataset.py`
   （迁移清单见 §4.2）；`convert_lora.py` 正常流程不再需要。

## 当前推荐配方（2026-10 实测定稿）

| 项 | 值 |
|----|----|
| 训练框架 | **musubi-tuner**（kohya-ss）|
| DiT 基座 | **Z-Image Base**（非蒸馏）`Tongyi-MAI/Z-Image`，bf16 |
| 文本编码器 / VAE | 与 Turbo **共享**（复用已下载的） |
| fp8 | **关**（`--fp8_base --fp8_scaled` 会掉质量） |
| 分辨率 | **1536**（bucketed） |
| 网络 | lora_zimage，`rank=alpha=16` |
| 数据量 | **300 张高质量精选**（定稿；510 / 866 均未优于 300，2000 张只会更慢；关键在质量与统一度） |
| 步数 | **总步数 = 图片数 × repeats × epochs ÷ batch_size**。定稿：300 张 × 7 epochs = **2100 步**。**每 epoch 存 checkpoint，训完出图盲测选最优**——最优往往**不是**最后一步 |
| 显存 | 20G：`--blocks_to_swap 12`（见 §16；显存被占时需更大，或先释放占显存的服务） |
| 实测 | **5.98s/步**（1536 / 20G 3080 / `ATTN=flash` / `BLOCKS=8`）；v5 300 张 / 7 epochs = 2100 步 / **3h31m** / loss 0.346；LoRA ~140MB |
| 出图 | `backup.sh ... --lora <OUT_NAME>.safetensors:1.2`；**v5 定稿 1.2**（0.7 通常无感，见 §7.1） |

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
| GPU | RTX 3080 **20GB**（已解锁，训练约 +30%） |
| 训练 venv | **`/data/venv-musubi`**（`python3 -m venv --system-site-packages`） |
| torch | 2.6.0+cu126（软链自 `/data/venv`；musubi venv 自带被 `~/.local` CPU 版抢先，故软链 CUDA 版） |
| 库版本 | transformers **4.57.6** / diffusers **0.38.0** / huggingface_hub **0.36.2**（降到与 torch 2.6 匹配） |
| flash-attn | **2.7.4**（已链接进 `/data/venv-musubi`，训练默认 `ATTN=flash`；GPU smoke test 通过） |
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

> **原始图片位置（2026-10-25 记录）**：`dav://192.168.1.53:5005/sata11-18072858035/weibo`
> WebDAV 共享；Dolphin/文件管理器可直接打开，命令行挂载可改用
> `http://192.168.1.53:5005/sata11-18072858035/weibo`（davfs2 / rclone / curl 均可）。
> 训练前先**拷到本地目录**（如 `/dataset/<name>_src`），再用 `curate_dataset.py` 处理；
> 不要直接对网络目录跑批量检测/打标，避免 IO 抖动和半途断连。
>
> **测试目录（2026-10-25 指定）**：`dav://192.168.1.53:5005/sata11-18072858035/weibo/佟丽娅`
> （145 张，约 937M），用于流水线 / 小规模训练测试；正式训练仍用全量图源。
>
> ```bash
> # 示例：挂载后拷贝（凭据按实际填写）
> sudo mount -t davfs http://192.168.1.53:5005/sata11-18072858035/weibo /mnt/weibo
> cp -r /mnt/weibo/<子目录> /dataset/<name>_src
> ```

`train_lora/prepare_dataset.py`：风格图目录 → 预缩（长边 ≤1536）+ Florence 自动打标。

> **两套数据集分开产出**：风格集 `<name>1536/`（1536，全部图合并）；
> 人脸集 `<name>_face/`（512×512 对齐脸，按明星分目录分别训练）。

产物 `/dataset/mystyle1536/`：
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
image_directory = "/dataset/mystyle1536/images"
cache_directory = "/dataset/mystyle1536/cache"
num_repeats = 1
```

> caption 解耦（进阶）：把"磨皮/柔光"等特征**写进 caption**（如 `smooth skin, studio soft lighting`），
> 推理时只给触发词，风格就不会强行附带磨皮。见 §9。

### 4.1 图像预处理工具（全部来自 `/opt/StaticPy` 的 StaticPy 二进制）

**下面所有工具都是 `/opt/StaticPy` 用 StaticPy 编译出的独立 ELF**（Python 子集 → Scheme → AOT），
不是 Python 脚本/外部命令。统一模式：C ABI（`.so`）+ StaticPy FFI，运行时按需挂 C 库或
ONNX Runtime；产物零 Python/pip。完整能力清单见 `/opt/StaticPy/CAPABILITIES.md`。

```bash
# StaticPy + ONNX Runtime 工具：用自带包装（内部已设好各 C 库路径）
bash /opt/StaticPy/tools/run.sh caption in.jpg '<CAPTION>' 1

# 根目录 / imgproc / upscale 工具：需带 .so 路径
export LD_LIBRARY_PATH=/opt/StaticPy/cpp/imgproc/build:/opt/StaticPy/cpp/sr/build
/opt/StaticPy/imgproc/pil resize in.png out.png 1024 1536 lanczos
```

| 命令 | 能力 | 数据集场景 |
|---|---|---|
| `tools detect IN` | 人脸检测（JSON） | 筛单脸 / 脸占比（curate 内用 facecli） |
| `tools crop IN OUT [size]` | 最大脸对齐裁剪 | 人脸 LoRA 512 数据集 |
| `tools parse IN OUT [color\|face\|skin\|hair]` | 人脸解析分割 | 换背景 / 局部处理 |
| `tools remove IN OUT [mode] [color]` | BiRefNet 抠图 | 去背景 / 合成 |
| `tools caption IN [task] [beams]` | Florence-2 反推描述 | 自动打标（curate 内用 img2prompt） |
| `tools ocr IN` | PP-OCRv4 | 剔除截图 / 带字图 |
| `tools sr IN OUT` | Real-ESRGAN 4x | 低清图放大后再训 |
| `tools restore IN OUT [w]` | CodeFormer 人脸修复 | 糊脸修复（会改变身份，慎用） |
| `tools pose / depth / edge / yolo` | 骨架 / 深度 / 软边缘 / 检测 | ControlNet 训练与验证素材 |
| `imgproc/pil <op> IN OUT ...` | Pillow 对标 CLI（convert/resize/crop/transpose/filter/quantize…） | 批量缩放 / 转码 / 裁剪 |
| `upscale/up 2x\|4x\|esrgan IN OUT` | 像素 / 模型放大 | 低清素材增强 |
| `flip IN OUT`、`hstack OUT GAP ...`、`concat OUT IN1 IN2` | 翻转 / 横拼 / 纵向拼接 | 对照图、拼图 |
| `split_grid IN OUTDIR COLS ROWS`、`anim/split_sheet OUTDIR COLS SHEET.png` | 网格 / 精灵表切分 | 视频帧、精灵图转单图 |
| `align_head OUTDIR CW CH PIVOT_X PIVOT_Y ...` | 按头部枢轴对齐一批图 | 大头像 / 表情数据对齐 |
| `flatten IN OUT BG_RGBA`、`cutout IN OUTDIR [TH]` | 压平 alpha / 批量抠图 | 统一背景 / 透明素材 |
| `anim/frames_to_gif OUT.gif FPS ...`、`anim/pack_sprites ...` | 帧转 GIF / 打包精灵图 | 动画数据集预览 |
| `cvops/cv erode/dilate/open/close/otsu/sobel`、`atlas`、`font/font`、`warp3d/w3d` | 形态学 / 图集 / 字形 / 伪 3D | 素材增强、图集制作 |

`imgproc/pil` 对标 Pillow：覆盖面与差分测试均 100/100，可替代 Python PIL 做确定性预处理。

注意：
- `seg`（MobileSAM）当前缺 `libmobilesam.so`，不可用；
- 根目录工具（`flip` / `hstack` / `align_head` 等）需先设 `LD_LIBRARY_PATH`（见上），`tools/run.sh` 自带；
- 这一层只做**数据准备**，训练仍走 `musubi_train.sh`。

### 4.2 Python → StaticPy 迁移清单

新增需求先查 `/opt/StaticPy/CAPABILITIES.md`；没有的能力**先封装 StaticPy 工具**，不要新写 Python。

| 现有脚本 | 当前依赖 | StaticPy 替代 | 状态 |
|---|---|---|---|
| `prepare_dataset.py` | PIL + Florence | `imgproc/pil` + `tools caption` + shell 写 `.txt` | 可直接迁 |
| `curate_dataset.py` | PIL/numpy + facecli/img2prompt | `tools detect/crop/caption` + `imgproc/pil` | 缺批量筛图/打分/写 `report.csv` 的 CLI，需封装 `curate` |
| `convert_lora.py` | safetensors + torch | **不需要替代**：sd.cpp 原生解析 `lora_unet_*` / `lora_A/B` / `lora_down/up`，safetensors/GGUF 直接加载 | 先直接加载原始 LoRA；仅遇到未识别前缀（如 `base_model.model.`）再处理 |
| `make_face_dataset.sh` | shell + facecli | 已是 StaticPy 工具链 | 保留 |
| `musubi_train.sh` | Python（musubi-tuner） | — | 允许例外（训练本体） |
| `train_lora.sh` / `run_train.sh` | diffusers/torch | — | 旧路线，删除/不再维护 |

封装约定：C ABI（`.so`）+ `extern fn` + `main()` 薄 CLI + `build.sh`；多值参数走 CSV 字符串；
产物独立 ELF，放 `/opt/StaticPy/<tool>/`，并在 `CAPABILITIES.md` 登记。

### 4.3 图片处理流水线（按步骤）

> **已封装为脚本**：`train_lora/image_pipeline.sh <src> <name> [style_trigger] [face_trigger]`
> 流程：cp → **人物门禁（非人物先删）** → 缩放 → 自动筛 → 手工筛 → 扣脸/`cp` style → 打标 → 校验 → `dataset.toml`。
> 无人脸/非人物在门禁阶段直接删除，**不进入 resize、扣脸和打标**。
> `SKIP_MANUAL=1` 跳过手工暂停；`BASE=/dataset`、`MAX=1536`、`FACE_SIZE=512` 可覆盖。

1. **拷贝子目录到 `/dataset/<子目录>`**：

   ```bash
   cp -r <图源>/<子目录> /dataset/<子目录>
   ```

   `<子目录>` 为单个明星目录（如 `佟丽娅`）。

2. **遍历图片，超过尺寸上限的按比例缩放**（只缩不放、不裁切）：

   ```bash
   MAX=1536   # 风格集长边上限；人脸集在后续步骤按 512 对齐裁剪
   # 逐张读取 w/h；若 max(w,h) > MAX：
   #   scale = MAX / max(w,h)
   #   nw = round(w*scale)，nh = round(h*scale)
   /opt/StaticPy/imgproc/pil resize IN.png OUT.png "$nw" "$nh" lanczos
   ```

   要求：长边 ≤ `MAX`，宽高比不变，不放大，不裁切。

3. **识别图片内容，剔除不合格图**（复用现有工具，不新开发）：

   - **内容识别 / 反推**：`tools caption IN '<CAPTION>' 1`（Florence-2）
     → 命中 `anime/illustration/screenshot/poster/logo/...` 或无人 → 剔除
   - **人脸检测**：`tools detect IN`（JSON：人脸框 + 分数），或 `facecli detect --batch`
   - **剔除规则**：
     - 人脸数 = 0 → 无人脸
     - 人脸数 > 1 → 多人脸
     - 最高人脸分数 < `0.60` → 脸不可靠
   - **质量不高**：结合反推描述（blurry / low quality / 模糊等）+ 人脸分数剔除，不新增质量工具

4. **分流保存（直接 cp 到 dataset，不移动）**：

   - **过滤后的整图 → `/dataset/style/<子目录>/images/`**（第 2 步已缩到长边 ≤1536）
   - **扣人脸 → `/dataset/face/<子目录>/images/`**（512×512 五点对齐脸）：
     ```bash
     facecli crop IN.png /dataset/face/<子目录>/images/OUT.png 512
     # 批量：facecli crop --batch  （每行 "IN\tOUT 512"）
     # 或单张：tools crop IN.png /dataset/face/<子目录>/images/OUT.png 512
     ```
   - **face 必须从原始大图扣**：脚本用硬链接保留原图（resize 不影响），扣完 512 再单独去背景；
     不要从 1536 的 style 图扣，否则脸小会被放大发虚
   - 两套 `images/` 一一对应；第 3 步已剔除无人脸，正常不会缺脸

5. **不合格的删除**：

   - 自动筛选时，不合格的 style 副本直接删除；扣脸失败也删
   - 处理完成后，清理 `/dataset/<子目录>` **原始拷贝**：只保留进入 `style/images` 的合格图，其余删除
   - `image_pipeline.sh` 第 8 步会自动做原始拷贝清理

6. **手工筛选**（自动规则之后，补一刀）：

   - 直接浏览 `/dataset/style/<子目录>/images/`、`/dataset/face/<子目录>/images/`
   - 不要的**在当前目录手动删掉**（`rm`），不建 rejected/、不移到别处
   - 删除：糊 / 重复 / 脸不对 / 非目标人物 / 构图不可用 / 自动反推漏检的
   - **先筛完再打标**，避免打完再删、浪费语料

7. **分别反推打标，生成训练语料**（输入目录固定）：

   - **style**：读 `/dataset/style/<子目录>/images/` → `img2prompt --task '<DETAILED_CAPTION>'`（或 `<CAPTION>` 更快）
   - **face**：读 `/dataset/face/<子目录>/images/` → `img2prompt --task '<CAPTION>'`
   - 批量：`img2prompt --batch`，stdin 为该固定目录下的图片列表，stdout 每行一条描述；**模型只加载一次**，后续图片复用
   - 图片在第 4 步已 **cp** 到 `dataset/*/images/`；**就地打标**，不移动
   - 每张图写同名 `.txt`：`<触发词>, <反推描述>`
   - 分别生成 `dataset.toml`：style `resolution=[1536,1536]`，face `resolution=[512,512]`
   - 最终结构：
     ```
     /dataset/style/<子目录>/{images/*, *.txt, dataset.toml}
     /dataset/face/<子目录>/{images/*, *.txt, dataset.toml}
     /dataset/style/dataset.toml   # 汇总所有 style 子目录，供 style LoRA 训练
     ```
   - 训练直接以这两个目录的 `dataset.toml` 为输入

8. **训练前校验（图片 ↔ `.txt`）**：

   - **有图无 `.txt`** → 用 `img2prompt` 重新给这张图打标
   - **有 `.txt` 无图**（手工删过图）→ 删掉对应 `.txt`
   - 脚本：`bash train_lora/check_labels.sh <images_dir> <trigger>`
   - `image_pipeline.sh` 结束会**自动**对 style/face 各跑一次

### 4.4 Style-only 数据源（不要求人脸）

适用于「整本相册都是同一风格、但不按明星」的图源，不扣脸、不去背景、不做人物门禁。

**下载**（复用 my-agent 的微博相册接口）：
```bash
bash train_lora/download_weibo_album.sh <uid|昵称|微博URL> <输出目录> [页数|all] [limit]
# 例：
bash train_lora/download_weibo_album.sh \
  'https://weibo.com/u/7619798979?tabtype=album' /dataset/_src/weibostyle 100
```
**CDP 采集（推荐，caption 识别明星）**
```bash
# 批量账号（含 8 个已记录号，自动跳过已完成）
bash train_lora/download_weibo_accounts.sh 100   # 100 = 每号 100 张（不是总数）

# 单号：逐张 →，取 /large/ 原图，按 caption 明星分目录
node train_lora/weibo_album_cdp.js \
  'https://weibo.com/u/7619798979?tabtype=album&uid=7619798979&index=0' \
  200 /dataset/_src/weibo_album_7619798979
```

- 默认 100 页；`all` 翻到 `WEIBO_ALBUM_MAX_PAGE`（默认 500）
- **下载后立即 resize 到长边 ≤2048**（源图留余量；`MAX` 可调，`RESIZE=0` 保留原图）
- 原图实测可达 7285×5464 / 8–9MB；resize 后按 2048 存，兼顾细节与空间
- 该账号相册至少 7000+ 张，按需用页数/`limit` 控制，不必全量

> **CDP 续采注意（2026-10-26 实测）**
> - 先用 `curl -s http://127.0.0.1:9222/json/list` 查已有 tab：
>   `weibo.com/u/<uid>?tabtype=album&index=N` 就是活着的 viewer
> - **复用已有 tab**（按 uid 匹配后 attach），不要再 `Target.createTarget` 从 0 新建
> - 按 `→` 后 viewer **index 先变、图片后加载**：必须等到「index 增加 **且** img `src` 变化」再取图
> - 固定 sleep 1.3s 会读到上一张的 `src`，连续重复后被误判成“相册到底”
> - index 没有 ~1330 上限：实测 `1344 → 1350` 正常，并在 1350 拿到新图
> - 单号 `timeout 1800`（30 分钟）会在大相册约 1300 张处中断；大号改用更长超时/续采

**处理**：
```bash
BASE=/dataset SKIP_MANUAL=1 LIMIT=2000 \
  bash train_lora/style_pipeline.sh /dataset/_src/weibostyle weibostyle weibostyle
```
- 只做：resize（长边 ≤1536，从 2048 源图再缩）→ 手工筛（可跳过）→ Florence 打标 → `style/<name>/dataset.toml` + 汇总表
- **不扣脸、不去背景、不要求有人**；背景和场景本身就是风格的一部分

### 4.5 每次训练抽样 300 张高质量图（新目录 + JSON）

> **定稿：300 张高质量精选即可**（2026-10-31）。实测 300 张 / 7 epochs（2100 步）
> 已能达到目标风格；510、866 张都没有更好，2000 张只会更慢，继续加量主要是浪费时间。
> 图源已按明星分目录（`/dataset/<明星>/`）；**源图只读**（只 `cp`，不 `mv`/不 resize）；
> 处理/打标复用 `style_pipeline.sh`。命令行里的 `style2000` 是历史目录名，实际数量以参数为准。

`train_lora/make_style_dataset.sh` 做四件事：

1. **分层抽样**：按明星轮转（round-robin）轮流取，避免大目录占比过高；`unknown` 默认排除。
2. **写 `selection.json`**：seed、目标数、每明星张数、源路径、复制后文件名、输出路径。
3. **复制到本次 run 的 `selected/`**：`cp -p`（尽量 reflink），源目录完全不动。
4. **调 `style_pipeline.sh`**：resize（≤1536）→ 打标 → `style/<name>/dataset.toml`。

> **打标只启动一次模型**：`style_pipeline.sh` 用一次 `img2prompt --batch` 把全部图片
> 喂给同一个进程，Florence-2 加载一次后连续复用，不会逐张重启。

```bash
# 抽 300 张高质量图 + 处理打标（定稿规模）
bash train_lora/make_style_dataset.sh 300 style2000 mystyle

# 只抽样/复制/写 JSON，暂不打标
PROCESS=0 bash train_lora/make_style_dataset.sh 300 style2000 mystyle

# 换一批（换 seed；同 seed = 同一批图，便于复现）
SEED=123 bash train_lora/make_style_dataset.sh 300 style2000 mystyle
```

环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `SRC_BASE` | `/dataset` | 源目录（只读） |
| `OUT_ROOT` | `/dataset/_runs` | 每次 run 的输出根目录 |
| `SEED` | `42` | 抽样种子；同 seed 抽到同一批 |
| `INCLUDE_UNKNOWN` | `0` | 1=把 `unknown/` 也纳入抽样 |
| `MAX` | `1536` | 训练长边（传给 `style_pipeline.sh`） |
| `TASK` | `<CAPTION>` | 打标任务；可改 `<DETAILED_CAPTION>` |
| `SKIP_MANUAL` | `1` | 跳过手工筛选（全自动） |
| `PROCESS` | `1` | 0=只抽样/复制/写 JSON，不跑打标 |
| `RUN` | 自动时间戳 | 自定义本次输出目录 |

产物：

```
/dataset/_runs/style2000_<时间>/
  selection.json          # 选了哪些图（可复现）
  selected/               # 复制出的源图（源图不动）
  style/style2000/images/ # 处理后图片 + 同名 .txt
  style/style2000/dataset.toml
```

`selection.json` 关键字段：

```json
{
  "created_at": "2026-10-27T13:30:00+08:00",
  "seed": 42,
  "target": 300,
  "selected": 300,
  "source_root": "/dataset",
  "per_star": {"田曦薇": 26, "娜扎": 24, "...": 18},
  "images": [
    {"star": "田曦薇", "source": "/dataset/田曦薇/x.jpg", "staged": "x.jpg"}
  ],
  "output": {
    "selected": "/dataset/_runs/style2000_<时间>/selected",
    "images": "/dataset/_runs/style2000_<时间>/style/style2000/images",
    "dataset_toml": "/dataset/_runs/style2000_<时间>/style/style2000/dataset.toml"
  }
}
```

训练：

```bash
DATA=/dataset/_runs/style2000_<时间>/style/style2000 \
  EPOCHS=7 BLOCKS=8 DIM=16 OUT=/data/lora/style2000 \
  bash train_lora/musubi_train.sh
```

> **定稿：300 张 × 7 epochs = 2100 步 / 3h31m / loss 0.346**（2026-10-31 实测，5.98s/步）。
> 2000 张 × 4 epochs ≈ 8000 步 / 约 13 小时，且效果并不更好；每轮自动存 checkpoint，训完出图盲测选最优。

---

## 5. 预缓存

```bash
PY=/data/venv-musubi/bin
M=/opt/musubi-tuner/src/musubi_tuner
D=/dataset/mystyle1536   # 或 §4.5 的 /dataset/_runs/<run>/style/<name>

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
ADAPTER= FP8=0 DIM=16 EPOCHS=7 \
DATA=/dataset/mystyle1536 OUT=/data/lora/mystyle_base \
bash train_lora/musubi_train.sh
# BLOCKS 不设时按显存/分辨率自动（20G/1536 约 8）；ATTN 默认 flash
```

环境变量：

| 变量 | 默认 | 说明 |
|------|------|------|
| `DIT` | Base DiT | DiT 权重（第一个分片） |
| `ADAPTER` | 空 | turbo 训练时才需要；Base 留空 |
| `DATA` | `/dataset/mystyle1536` | 数据集目录 |
| `OUT` | `/data/lora/mystyle_base` | 输出目录 |
| `DIM` / `LR` | 16 / 1e-4 | network_dim=alpha / 学习率 |
| `EPOCHS` / `STEPS` | 7 / 空 | 轮数；定稿 300 张 × 7 = 2100 步；设 `STEPS` 则用 max_train_steps |
| `FP8` | 0 | 1=开 fp8（掉质量，默认关；3080 无原生 fp8 加速） |
| `BLOCKS` | 自动 | `--blocks_to_swap`（省显存）；不设时按显存/分辨率自动（20G/1536 约 8）。**512 人脸数据集设 `0` 可提速约 4 倍**（见 §16） |
| `ATTN` | `flash` | `flash` / `sdpa` / `sage` / `xformers`；`flash_attn` 不可用时自动回退 `sdpa` |

脚本内默认：`--flash_attn`（`ATTN=flash`）、`--mixed_precision bf16 --timestep_sampling shift --weighting_scheme none
--discrete_flow_shift 2.0 --optimizer_type adamw8bit --gradient_checkpointing
--network_module networks.lora_zimage`；训练后保留原始 `out.safetensors`，并自动转出 `.gguf`（f16）。

> **脚本已内置优化，无需再手调**：VRAM 自检 → 按分辨率自动 `--blocks_to_swap`；
> block swap 默认 `pinned + H2D-only + ring_size 2`（`SWAP_MODE=plain|pinned` 可切）；
> `--gradient_checkpointing`、`adamw8bit`、`--flash_attn`（不可用回退 `--sdpa`）、bf16、persistent data loader；
> 每 epoch 自动存 checkpoint。1536 档在解锁 3080 上约 **8.8s/步**，直接按此配方跑即可。

---

## 7. 转换 + 出图

musubi 输出是 sd-scripts 命名（`lora_unet_*`）。**sd.cpp 已原生兼容**（`src/name_conversion.cpp`：
`lora_unet_*` / `lora_A/B` / `lora_down/up` 都会映射到 `model.diffusion_model.*`），
`safetensors` 与 `GGUF` 都能按扩展名自动加载，**不需要先转命名**：

```bash
# 0.7 起调；风格 LoRA 若 0.7 看不出效果，按 §7.1 上调到 1.0~1.2
./cpp/sd/backup.sh 2560 1440 \
  --lora /data/lora/<name>/<OUT_NAME>.safetensors:0.7
```

`musubi_train.sh` 产出**原始权重** `<OUT_NAME>.safetensors`，并自动转出 `<OUT_NAME>.gguf`（f16）；
两者都能直接加载。把 LoRA 转 GGUF 只是换容器/量化，**不会改 key**，sd.cpp 同样能读。
只有遇到 sd.cpp 不认识的额外前缀（如 diffusers `base_model.model.`）时才需转换。

出图（**触发词自动注入**，无需手写）：
```bash
# backup.sh 会按 /data/lora/lora_triggers.conf 自动把触发词前置到默认提示词
# 原始 musubi 输出、*.safetensors、转出的 .gguf 三种都可直接加载
./cpp/sd/backup.sh 2560 1440 \
  --lora /data/lora/style2000_v5/mystyle.gguf:1.2
# 输出: ✓ LoRA 触发词注入: mystyle
#       Prompt: ... soft lighting, mystyle, solo,single woman,half body portrait...
# sd.cpp: (630 / 630) LoRA tensors have been applied
```

触发词映射（`/data/lora/lora_triggers.conf`，可用 `LORA_TRIGGERS_FILE` 覆盖；模板见
`train_lora/lora_triggers.conf.example`）：
```
<lora 文件名> = <触发词>
mystyle.safetensors = mystyle
```

### 7.1 2026-10-31 定稿：v5 风格 LoRA 出图参数（实测）

训练规模已定稿：**300 张高质量精选 / 7 epochs = 2100 步 / 3h31m / loss 0.346**。
`style2000_v5` 定稿 **1.2**；0.7 基本无感（v4 实测），1.0 偶发双人，1.2 稳定，效果已获用户确认。
训练用 **Base**、推理挂 **Turbo**；sd.cpp 显示 **630/630** tensors 全应用。
（v4 为 510 张 / 4 epochs，需 1.2 是因为权重幅度偏小；v5 已取代 v4。）

| 项 | 值 |
|---|---|
| 扩散模型 | `/data/models/image/z_image_turbo-Q8_K_M.gguf` |
| LoRA | `/data/lora/style2000_v5/mystyle.gguf`（或同目录 `.safetensors`） |
| 权重 | **1.2**（定稿；0.7 无感；更高权重更易脸变/双人，未采用） |
| 触发词 | `mystyle`（由 `/data/lora/lora_triggers.conf` 自动注入） |
| steps / hires_steps | **20 / 40** |
| cfg / sampler / scheduler | **2.0 / euler / discrete** |
| hires | 开，strength **0.25**，upscaler **latent-bicubic** |
| 尺寸 | 2560×1440（base 1920×1080）；pyq 2048×2048；xhs 1920×2560；竖屏 1440×2560 |
| VAE tiling | tile 128 / overlap 0.5 |
| FreSca | low 1.0 / high 1.25 / cutoff 20 |
| EasyCache | `easycache,1,0.15,0.95` |
| 后处理 | clarity 0；sharpen 0.3,1；smart_sharpen 0.5,2；edge_sharpen 0,2,0.3 |

**最终 prompt**（质量前缀 + 默认 prompt + 自动触发词）：

```
masterpiece, best quality, ultra-detailed, sharp focus, 8k uhd, photorealistic, highly detailed, crisp, clear, centered composition, professional portrait, medium shot, realistic skin texture, soft lighting, mystyle, solo,single woman,half body standard portrait of a young woman, subject perfectly centered, symmetrical composition, front view, soft natural lighting, elegant pose, studio lighting, sharp eyes, solid soft light green background, sage green, clean seamless plain background, no props, flat solid color backdrop, fair skin, pale skin, smooth skin, matte skin, porcelain skin, flawless skin, medium close up
```

**negative（backup.sh 默认）**：

```
blurry, low quality, worst quality, jpeg artifacts, noise, grain, soft focus, out of focus, hazy, unclear, bad anatomy, deformed, border artifacts, edge distortion, tiling artifacts, edge artifacts, frame distortion, warped edges, stretched proportions, asymmetrical face, off-center, cropped, out of frame, partial face, cut off, incomplete head, cropped head, watermark, text, logo, signature, cropped shoulders, oily skin, shiny skin, greasy skin, glossy skin, plastic skin, skin blemishes, embedding:EasyNegative, embedding:bad-hands-5
```

```bash
# 定稿效果（用户确认）：2560×1440 + w1.2 + seed 102986709
REF=/home/quqiufeng/v4gguf_ref_2560x1440_20261030_231734.png
SEED=102986709 ./cpp/sd/backup.sh --from-image "$REF" ~/v5_2560x1440.png 2560 1440 \
  --lora /data/lora/style2000_v5/mystyle.gguf:1.2

# 同参数、随机 seed
SEED=$(shuf -i 1-2147483646 -n1) ./cpp/sd/backup.sh --from-image "$REF" ~/v5_2560x1440.png 2560 1440 \
  --lora /data/lora/style2000_v5/mystyle.gguf:1.2

# 朋友圈 2048×2048（同一 seed 可保持同一人）
SEED=102986709 ./cpp/sd/backup.sh --from-image "$REF" ~/pyq.png 2048 2048 \
  --lora /data/lora/style2000_v5/mystyle.gguf:1.2

# 小红书 1920×2560
SEED=102986709 ./cpp/sd/backup.sh --from-image "$REF" ~/xhs.png 1920 2560 \
  --lora /data/lora/style2000_v5/mystyle.gguf:1.2
```

**已确认样张**（三张均通过）：

| 用途 | 文件 | seed |
|---|---|---|
| 2560×1440 | `/home/quqiufeng/v5_2560x1440_w12_seed102986709_20261031_064703.png` | 102986709 |
| 朋友圈 2048² | `/home/quqiufeng/v5_pyq_seed102986709_20261031_065635.png` | 102986709 |
| 小红书 1920×2560 | `/home/quqiufeng/v5_xhs_seed_reroll2_20261031_071430.png` | 1340838768 |

实测耗时（v5 GGUF 1.2，RTX 3080 20G）：

| 尺寸 | 耗时 |
|---|---|
| 2560×1440 | 3m52s–3m55s |
| pyq 2048×2048 | 4m48s |
| xhs 1920×2560 | 5m55s–5m59s |
| 1440×2560 | 约 4m10s（v4 同参实测） |

注意：

- `--from-image` **不还原 negative**（仍用 `backup.sh` 默认负面词）；参考图里内嵌的是 v4 LoRA，
  命令中必须显式 `--lora .../style2000_v5/mystyle.gguf:1.2` 覆盖。尺寸可被位置参数覆盖；
  seed 用环境变量 `SEED=...` 覆盖。**同 seed 可逐像素复现**（实测 diff=0）。
- **xhs 1920×2560 容易在底部再生成一个头/半身**；在默认 negative 末尾追加
  `two people, multiple women, duplicate, cloned face, extra head, disembodied head, second face, twins, mirror image, reflection`
  后设为 `NEGATIVE_PROMPT`，实测恢复单人。
- v5 训练：`/dataset/_runs/style2000_20261027_133634/`，300 张 / 7 epochs / 2100 步 /
  3h31m / loss 0.346；输出 `/data/lora/style2000_v5/`
  （`mystyle-000001~000006`、`mystyle.safetensors`、`mystyle.gguf`）。

### 7.2 定稿样张参数记录（2026-10-31）

三张用户确认样张的**完整出图参数**已单独落盘（PNG 内也嵌有 `tEXt:parameters`）：

- 记录文件：`train_lora/records/2026-10-31_v5_final.json`
  （prompt / negative / seed / 尺寸 / base 尺寸 / steps / cfg / sampler / lora / 耗时 / sha256 / 复现命令）
- xhs 防双人负面词：`train_lora/records/2026-10-31_v5_xhs_negative.txt`

| 用途 | 文件 | seed | 尺寸（base） | negative |
|---|---|---|---|---|
| 2560×1440 | `~/v5_2560x1440_w12_seed102986709_20261031_064703.png` | 102986709 | 2560×1440（1920×1080） | 默认 |
| 朋友圈 | `~/v5_pyq_seed102986709_20261031_065635.png` | 102986709 | 2048×2048（1664×1664） | 默认 |
| 小红书 | `~/v5_xhs_seed_reroll2_20261031_071430.png` | 1340838768 | 1920×2560（1536×2048） | 默认 + 防双人 |

复现 xhs 定稿样张：

```bash
REF=/home/quqiufeng/v4gguf_ref_2560x1440_20261030_231734.png
NEGATIVE_PROMPT="$(cat train_lora/records/2026-10-31_v5_xhs_negative.txt)" \
SEED=1340838768 ./cpp/sd/backup.sh --from-image "$REF" ~/v5_xhs.png 1920 2560 \
  --lora /data/lora/style2000_v5/mystyle.gguf:1.2
```

查看任意成图的记录参数：

```bash
./cpp/sd/build/img_hires --dump-meta ~/v5_xhs_seed_reroll2_20261031_071430.png
```

---

## 8. 工具清单（`train_lora/`）

| 文件 | 作用 |
|------|------|
| `image_pipeline.sh` | **图片流水线**：cp → 缩放 → 自动筛 → 扣脸 → 手工筛 → 打标 → 校验 → `dataset.toml`（shell + 现有二进制） |
| `check_labels.sh` | **图片↔标注校验**：有图无 txt 补打标；有 txt 无图删 txt |
| `download_weibo_album.sh` | **微博相册下载**：uid/昵称/URL → 大图目录（复用 my-agent 接口） |
| `style_pipeline.sh` | **纯 style 流水线**：resize → 手工筛 → 打标 → dataset.toml（不扣脸/不去背景） |
| `make_style_dataset.sh` | **每次训练抽样**：从明星库分层抽 N 张 → `selection.json` → 复制到新 run → 调 `style_pipeline.sh`；源图不动（§4.5） |
| `move_nonperson.sh` | **非人物/局部特写清理**：Florence 短描述 → 无人脸图再用 `<DETAILED_CAPTION>` 复判 → mv 到 `others/`；支持断点/并发/缓存重判 |
| `musubi_train.sh` | **musubi 训练封装**（当前主用，支持 Base/turbo、fp8、blocks_to_swap、`ATTN=flash`）+ 保留原始权重 + 转 GGUF |
| `prepare_dataset.py` | 风格图 → 预缩 + Florence 批量打标（`--batch`）+ `metadata.jsonl`/`.txt` |
| `run_train.sh` / `train_dreambooth_lora_z_image.py` / `qnbit4.json` | diffusers 路线（备选，已适配 0.38） |
| `convert_lora.py` | 可选：仅处理 sd.cpp 未识别的额外前缀（正常训练/出图不再需要） |

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

改进（已采纳）：换 Base、关 fp8、rank16/alpha16、1536 分辨率、300 张精选 / 7 epochs（定稿）、blocks_to_swap 4。

仍可做：
- **caption 解耦**：把 `smooth skin / studio soft lighting` 写进 caption，推理只给触发词。
- **混入正则图**：15-20 张未修图高清人像（不带触发词，只写物理描述）防过拟合。
- **强度**：0.6~0.8 试；过高会漂构图（如 2560 偶发双人，prompt 加 `single woman` 规避）。

---

## 10. 快速复现（换一套风格图）

```bash
# 0) 图源已按明星分目录：抽 300 张高质量图 + 生成 style 数据集（源图不动，详见 §4.5）
bash train_lora/make_style_dataset.sh 300 style2000 mystyle
# → DATA=/dataset/_runs/style2000_<时间>/style/style2000
# 用第 0 步时，跳过下面第 1–2 步，直接预缓存/训练

# 1) 新风格图源（预缩 1536 + Florence 打标）
/data/venv/bin/python train_lora/prepare_dataset.py <风格图目录> /dataset/<name> \
  --trigger <trigger> --caption --max-side 1536
cp -r /dataset/<name> /dataset/<name>1536  # 或直接改 dataset.toml

# 2) 写 dataset.toml（resolution [1536,1536], image/cache_directory）

# 3) 预缓存 latent + TE（§5；TE 可复用）

# 4) 训练 + 转 GGUF（§6）
DIT=/data/models/z-image-base/transformer/diffusion_pytorch_model-00001-of-00002.safetensors \
FP8=0 BLOCKS=12 DIM=16 EPOCHS=7 DATA=/dataset/<name>1536 OUT=/data/lora/<name> \
bash train_lora/musubi_train.sh

# 5) 出图（§7）
./cpp/sd/backup.sh 2560 1440 --lora /data/lora/<name>/mystyle.safetensors:1.2
```

---

## 11. 指定人脸 + 风格：双 LoRA 方案（需求与实现思路）

> 以下 §11/§13/§14 是早期展开的细节。**当前只按文首“训练任务”两档执行**：
> 风格 LoRA 用全部图；人脸 LoRA 按明星子目录分别训练；不做联合模型。

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
  --lora /data/lora/celeba_base/mystyle.safetensors:0.7 \
  --lora /data/lora/mystyle_base/mystyle.safetensors:0.6
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
- **风格 + 人脸** = 1 个风格 LoRA（全部图）+ 每个明星各自的 face LoRA，出图叠加（多 `--lora`）。

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
# → /dataset/celebA_face/{images/*.png(512对齐脸), *.txt, dataset.toml}
```
- **同一个人**的图 → 训出**身份 LoRA**（脸越来越像这个人）。
- **不同人**的美图集合 → 训出**"审美/风格脸" LoRA**（生成的脸偏这个风格，不绑定具体人）。
- 无脸的图自动跳过；建议 20~50 张、角度/表情/光照多样、换装换背景。

### 13.3 预缓存 + 训练（人物建议 rank32）
```bash
D=/dataset/celebA_face
PY=/data/venv-musubi/bin; M=/opt/musubi-tuner/src/musubi_tuner
$PY/python $M/zimage_cache_latents.py --dataset_config $D/dataset.toml \
  --vae /data/models/z-image-turbo/vae/diffusion_pytorch_model.safetensors --device cuda
$PY/python $M/zimage_cache_text_encoder_outputs.py --dataset_config $D/dataset.toml \
  --text_encoder /data/models/z-image-te-qwen3.safetensors --batch_size 8 --device cuda

DIT=/data/models/z-image-base/transformer/diffusion_pytorch_model-00001-of-00002.safetensors \
  FP8=0 BLOCKS=0 DIM=32 EPOCHS=4 DATA=$D OUT=/data/lora/celebA OUT_NAME=celebA \
  bash train_lora/musubi_train.sh
# → /data/lora/celebA/celebA.safetensors（4 epochs 足够；见 §14.9）
```

### 13.4 出图（触发词自动注入）
在 `/data/lora/lora_triggers.conf` 加一行 `celebA.safetensors = celebA`（或把训练输出的
LoRA 重命名为 `<触发词>.safetensors`），然后：
```bash
./cpp/sd/backup.sh 2560 1440 --lora /data/lora/celebA/celebA.safetensors:0.7
```

> 说明：`musubi_train.sh` 的 LoRA 名由 `OUT_NAME`（默认 `mystyle`）决定；建议设成触发词，
> 如 `OUT_NAME=celebA`，输出 `celebA.safetensors`，映射一目了然。

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
| 数量 | **≥20 张可训；200+ 张更稳**。2026-10-31 实测 214–1685 张均用 **4 epochs** 足够，不必 8–12 |
| 内容 | **同一个人**；角度（正/半侧/侧）、表情、光照、远近、换装换背景多样 |
| 排除 | 他人、重滤镜/重磨皮、遮脸（口罩/手/大墨镜）、糊图、低清、多脸同框（会自动取最大脸，但尽量单人） |
| 分辨率 | 越高越好（≥1024 更好，脚本会裁到 512 对齐脸） |

> 数量不是越多越好：大目录里混入的他人/糊图/重滤镜会稀释身份；宁精勿滥。
> 本机已按 `/dataset/<明星>/` 直接批量训练（≥200 张的 10 位，见 §14.8）。

### 14.3 制作对齐脸数据集（cpp/face）
```bash
# 一键：检测 → 5点对齐裁剪(512) → Florence 打标（身份中性的内容描述）
bash train_lora/make_face_dataset.sh /path/to/star_photos celebA
# → /dataset/celebA_face/{images/*.png(512对齐脸), *.txt, dataset.toml}
```
> caption 形如 `celebA, A woman with long black hair wearing a red dress.`——描述服装/背景，表情/光照，
> 身份留给触发词，正是身份 LoRA 需要的"解耦"。

### 14.4 预缓存 + 训练（4 epochs 定稿）
```bash
D=/dataset/celebA_face
PY=/data/venv-musubi/bin; M=/opt/musubi-tuner/src/musubi_tuner
$PY/python $M/zimage_cache_latents.py --dataset_config $D/dataset.toml \
  --vae /data/models/z-image-turbo/vae/diffusion_pytorch_model.safetensors --device cuda
$PY/python $M/zimage_cache_text_encoder_outputs.py --dataset_config $D/dataset.toml \
  --text_encoder /data/models/z-image-te-qwen3.safetensors --batch_size 8 --device cuda

DIT=/data/models/z-image-base/transformer/diffusion_pytorch_model-00001-of-00002.safetensors \
  FP8=0 BLOCKS=0 DIM=32 EPOCHS=4 OUT_NAME=celebA DATA=$D OUT=/data/lora/celebA \
  bash train_lora/musubi_train.sh
# → celebA.safetensors + celebA.gguf；确认 gguf 后可删 safetensors
```
| 超参 | 身份 LoRA 定稿 |
|------|----------------|
| rank/alpha | **32**（不够像再 64） |
| epochs | **4**（不必 8–12；数据很少/相似度不足先 6） |
| 分辨率 | **512**（对齐脸原生） |
| lr | 1e-4 |
| fp8 | 关 |
| blocks_to_swap | **0**（512 显存充裕，比 12 快约 4 倍；见 §16） |

实测速度 **1.1–1.25s/步**（512 / BLOCKS=0）：
100 张 / 4 epochs ≈ **8–10 分钟**；200 张 ≈ **18–22 分钟**；500 张 ≈ **45–50 分钟**；1000 张 ≈ **1.5 小时**。
（含预缓存的整星耗时见 §14.8。）

### 14.5 触发词映射与存放
产出建议放 `/dataset/<明星>/lora/`，并在 `/data/lora/lora_triggers.conf` 加一行（`backup.sh` 只看文件名）：
```
celebA.gguf = celebA
```
`train_faces_batch.sh` 会自动写入映射；同名 `.safetensors` 不需要保留。

### 14.6 出图
```bash
# 只出这张脸：2560×1440
./cpp/sd/backup.sh 2560 1440 \
  --lora /dataset/田曦薇/lora/tianxiwei.gguf:0.7

# 朋友圈 2048×2048
./cpp/sd/backup.sh 2048 2048 \
  --lora /dataset/田曦薇/lora/tianxiwei.gguf:0.7
```
- 默认提示词即可；触发词由 `lora_triggers.conf` 自动前置。
- **只用身份 LoRA** = 不加风格 LoRA；需要风格时再叠加 `--lora /data/lora/style2000_v5/mystyle.gguf:1.2`。
- 权重 **0.7 起**；不够像优先提到 0.8–0.9，而不是继续加 epoch。

### 14.7 相似度不够时的调法
1. **出图权重 0.8–0.9**（先动推理侧）。
2. **提高数据质量**：去重、去他人/糊图/重磨皮；正脸/半侧占比提高。
3. epochs **4→6**（仅数据少或身份弱时）；不要回到 10–16，收益低且易过拟合。
4. rank **32→64**；多存 checkpoint 对比。
5. 加**人脸区域精修**（ADetailer 式局部重绘，待做）让 2560 下的脸更锐、更像。
6. 数据不足（<15）先补图，别硬训。

### 14.8 批量训练：≥200 张的 10 位（2026-10-31 实测）

`train_lora/train_faces_batch.sh` 顺序处理 `/dataset/<明星>/`：
扣脸 512 → Florence 打标 → 缓存 latent/TE → **4 epochs / rank32 / BLOCKS=0** → 转 GGUF →
删 checkpoint 和 safetensors（只留 `.gguf`）→ 清理临时数据集。

```bash
# 默认就是这 10 位；EPOCHS/DIM/CLEAN 可覆盖
nohup bash train_lora/train_faces_batch.sh > /dataset/_face/batch.log 2>&1 &
```

| 明星 | 源图 | 触发词 | 整星耗时（含扣脸/打标/缓存） | 产出 |
|---|---:|---|---:|---|
| 田曦薇 | 1685 | `tianxiwei` | 2h57m | `/dataset/田曦薇/lora/tianxiwei.gguf` |
| 杨幂 | 458 | `yangmi` | 50m | `/dataset/杨幂/lora/yangmi.gguf` |
| 娜扎 | 429 | `naza` | 46m | `/dataset/娜扎/lora/naza.gguf` |
| 陈都灵 | 351 | `chenduling` | 36m | `/dataset/陈都灵/lora/chenduling.gguf` |
| 迪丽热巴 | 289 | `dilireba` | 30m | `/dataset/迪丽热巴/lora/dilireba.gguf` |
| 白鹿 | 271 | `bailu` | 28m | `/dataset/白鹿/lora/bailu.gguf` |
| 王楚然 | 235 | `wangchuran` | 24m | `/dataset/王楚然/lora/wangchuran.gguf` |
| 程潇 | 231 | `chengxiao` | 24m | `/dataset/程潇/lora/chengxiao.gguf` |
| 杨超越 | 215 | `yangchaoyue` | 22m | `/dataset/杨超越/lora/yangchaoyue.gguf` |
| 鞠婧祎 | 214 | `jujingyi` | 22m | `/dataset/鞠婧祎/lora/jujingyi.gguf` |

合计 **4,378 张 / 7h38m**。状态：`/dataset/_face/status.tsv`；日志：`/dataset/_face/latest.log`。
每星产出 268M safetensors + 140M gguf；本机最终只保留 **gguf**（约 140M/人）。

### 14.9 高清出图验证（2560 / pyq）

`cpp/sd/gen_faces_2560.sh` 按人只挂身份 LoRA，默认提示词，**一人一张**：
自动检测人脸数；双人/多人会用防重复负面词重试（最多 3 次）。

```bash
# 2560×1440
OUTDIR=$HOME/face_2560 GEN_W=2560 GEN_H=1440 GEN_TAG=2560x1440 WEIGHT=0.7 \
  bash cpp/sd/gen_faces_2560.sh

# pyq 2048×2048
OUTDIR=$HOME/face_pyq GEN_W=2048 GEN_H=2048 GEN_TAG=pyq2048 WEIGHT=0.7 \
  bash cpp/sd/gen_faces_2560.sh
```

- 2026-10-31 两套各 10 张全部单人通过；2560 每张约 **4m10s–4m20s**，pyq 约 **5m–5m10s**。
- 输出目录：`~/face_2560/`、`~/face_pyq/`；状态 `status.tsv`。
- pyq 的田曦薇首张曾出双人，自动重试后单人。
- **结论**：高清大图的人像迁移用身份 LoRA 训练解决；参考图/IPAdapter 路线在 2560 下容易漂。

### 14.10 合规
真实人物肖像权与用途由使用者自负。

---

## 15. 相关文件速查
| 路径 | 说明 |
|------|------|
| `dav://192.168.1.53:5005/sata11-18072858035/weibo` | **原始图片（WebDAV 图源）**，训练前先拷到本地 |
| `dav://192.168.1.53:5005/sata11-18072858035/weibo/佟丽娅` | **测试目录**（145 张 / ~937M），用于流水线与小规模训练测试 |
| `/opt/StaticPy` | **StaticPy 编译的图像处理二进制工具集**（`tools/`、`imgproc/pil`、`upscale/up`、`align_head` 等，见 §4.1） |
| `cpp/face/facecli` | 检测/对齐裁剪/解析 CLI |
| `train_lora/make_face_dataset.sh` | 人物图片目录 → 对齐脸数据集 |
| `train_lora/train_faces_batch.sh` | **批量身份 LoRA**：按明星目录扣脸→打标→缓存→4 epochs 训练→只留 gguf（§14.8） |
| `cpp/sd/gen_faces_2560.sh` | **批量出图**：一人一张身份 LoRA，自动重试双人（§14.9） |
| `train_lora/musubi_train.sh` | LoRA 训练（`OUT_NAME` 命名，自动转 sd.cpp 命名） |
| `/data/lora/lora_triggers.conf` | LoRA→触发词映射（backup.sh 自动注入；只保留 `.gguf` 行） |
| `/dataset/<明星>/lora/<trigger>.gguf` | 身份 LoRA 产出（sd.cpp 直接加载） |
| `/data/lora/<name>/<OUT_NAME>.gguf` | 风格/通用 LoRA 产出 |

---

## 16. 速度调优（2026-10 实测，512 人脸数据集）

> **结论：512 人脸/身份 LoRA 用 `BLOCKS=0`（不换出），实测 **1.1–1.25s/步**，
> 比旧配方 `BLOCKS=12` 快约 4 倍。**
> **epochs 定稿 4**：100 张≈8–10min，200 张≈18–22min，500 张≈45–50min，1000 张≈1.5h。
> 12 epochs（98 张≈29min）只作历史对照，不再默认。

测试环境：RTX 3080 20G；512 对齐脸；Z-Image Base bf16；rank32；
`--gradient_checkpointing` 开；fp8 关。下表是早期 **12 epochs / 98 张** 的对比；
定稿为 **4 epochs**，速度按 1.1–1.25s/步 × 图片数 × 4 估算。

| `BLOCKS` | 显存 | 步速 | 12 epochs 预计 |
|----------|------|------|----------------|
| 12（旧默认） | ~10.3G | 6.15 s/步 | ~2h |
| 4 | ~13.1G | 2.47 s/步 | ~47min |
| **0（推荐）** | **~14.5G** | **1.1–1.25 s/步** | **~29min** |

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
  FP8=0 BLOCKS=0 DIM=32 EPOCHS=4 OUT_NAME=<trigger> \
  DATA=/dataset/<name>_face OUT=/dataset/<明星>/lora \
  bash train_lora/musubi_train.sh
```

> 注意：`setsid bash ... &` 让训练脱离终端会话运行，避免被父 shell/工具超时杀掉；
> 日志重定向到文件后轮询进度即可。

---

## 17. 训练配方对比结论（2026-10 实测：训练无显著增益，Q8 才有效）

为提升 liuhaocun 身份 LoRA 的"像"程度，做了 A/B：

| 配方 | caption | rank | epochs | 数据 |
|------|---------|------|--------|------|
| BASE | Florence 描述式（`liuhaocun, A woman with long black hair...`） | 32 | 12 | `/dataset/liuhaocun_face`（98） |
| V1 | **极简，只写 `liuhaocun`** | 32 | 10 | `/dataset/liuhaocun_min`（98） |
| V2 | 极简，只写 `liuhaocun` | **64** | 10 | `/dataset/liuhaocun_min`（98） |

评估：`backup.sh` **2560×1440**，**Q8** 模型，同一 seed，双 LoRA
（`liuhaocun:X` + `mystyle:0.6`）。LoRA：`/data/lora/exp/liuhaocun_min_r{32,64}.safetensors`；
对比图 `~/cmpq8_{BASE,V1,V2}_*.png`、竖版 `~/vq8_{BASE,V1,V2}_*.png`。

**结论**：
1. **2560×1440 下 BASE / V1 / V2 几乎无差别** → caption 解耦、rank 32→64
   都没有可感知收益（数据与身份已足够）。
2. **Q8 明显优于 Q5**（细节/通透），是真正有效的杠杆 —— 优先用
   `DIFFUSION_MODEL=/data/models/image/z_image_turbo-Q8_K_M.gguf`。
3. 想更"像"应走**推理侧**：LoRA 权重 0.7→0.9（风格 LoRA 实测要到 **1.0–1.2**，见 §7.1）、
   CFG 2.0→3.0、或人脸局部重绘（ADetailer 式），继续调训练配方收益很低。

---

## 18. 出图尺寸预设（小红书图 / 朋友圈图）

`backup.sh` 内置 `--preset` 社媒尺寸（Q8 默认；耗时按 §7.1 的 v5 GGUF 1.2 实测）：

| preset | 别名 | 尺寸 | 比例 | 用途 / 实测耗时 |
|--------|------|------|------|------|
| `xhs` | `小红书` / `xiaohongshu` | **1920×2560** | 3:4 | 小红书竖图，约 **5m55s–5m59s** |
| `pyq` | `朋友圈` / `moments` | **2048×2048** | 1:1 | 朋友圈方图，约 **4m48s** |
| — | 直接传像素 | **1440×2560** | 9:16 | 竖屏，约 **4m10s**（v4 同参） |

```bash
# 风格 LoRA v5（0.7 无感，用 1.2）
./cpp/sd/backup.sh --preset xhs ~/xhs.png \
  --lora /data/lora/style2000_v5/mystyle.gguf:1.2

./cpp/sd/backup.sh --preset pyq ~/pyq.png \
  --lora /data/lora/style2000_v5/mystyle.gguf:1.2

# 竖屏 1440×2560
./cpp/sd/backup.sh 1440 2560 \
  --lora /data/lora/style2000_v5/mystyle.gguf:1.2

# 定稿 seed（2560/pyq 已验证）；xhs 同 seed 易出双人，见 §7.1 防双人负面词
SEED=102986709 ./cpp/sd/backup.sh --preset pyq ~/pyq.png \
  --lora /data/lora/style2000_v5/mystyle.gguf:1.2
SEED=102986709 ./cpp/sd/backup.sh --preset xhs ~/xhs.png \
  --lora /data/lora/style2000_v5/mystyle.gguf:1.2
```

说明：不给 preset 时默认 2560×1440 横版；也可直接传像素，如 `backup.sh 1920 2560 ...`。
朋友圈统一用 2048×2048（1440×1440 偏糊，2560² 太慢）。
身份 LoRA 权重仍从 0.7 起；风格 LoRA 若 0.7 看不出，按 §7.1 上调到 1.0–1.2。

---

## 19. 出图参数内嵌 + 参考图复刻（ComfyUI 风格）

每张出图都会把生成参数写入 PNG `tEXt` 块 `parameters`（`img_hires` 实现），
含：prompt / negative / steps / hires_steps / cfg / seed / method / scheduler /
尺寸（最终 + base）/ hires 参数 / 模型路径 / **lora 列表** / fresca / cache / 后处理。

```bash
# 查看某张图的参数
./cpp/sd/build/img_hires --dump-meta ~/out.png

# 从参考图复刻（读回 prompt/steps/cfg/sampler/hires/lora/尺寸）
./cpp/sd/backup.sh --from-image ~/out.png ~/new.png

# 只换随机 seed（复刻 §7.1 v5 定稿参数；参考图内已含 v5 LoRA 1.2）
REF=/home/quqiufeng/v5_2560x1440_w12_seed102986709_20261031_064703.png
SEED=$(shuf -i 1-2147483646 -n1) ./cpp/sd/backup.sh --from-image "$REF" ~/out.png

# 复刻参数但改尺寸（位置参数优先于参考图尺寸）
SEED=$(shuf -i 1-2147483646 -n1) ./cpp/sd/backup.sh --from-image "$REF" ~/pyq.png 2048 2048
```

说明：

- CLI/env 显式给的参数优先；未给的用参考图里的。`lora` 连同权重（如 `mystyle.gguf:1.2`）一起恢复。
- **`negative` 不还原**，仍用 `backup.sh` 默认负面词；§7.1 的 xhs 定稿图另加了防双人负面词，复刻时需自行设 `NEGATIVE_PROMPT`。
- `clarity/sharpen/fresca/cache` 也走当前默认值；需要不同就显式设环境变量。
- `--preset` 会被参考图尺寸覆盖；要改尺寸请用位置参数（`out.png 2048 2048`）。

用途：忘了用过的 prompt / 参数，直接拿参考图即可复刻。旧图（本功能之前生成的）没有该元数据。

---

## 20. 数据集清洗 + 批量训练（2026-10 新增，替代手动 prepare_dataset）

### 20.1 一键数据集：`train_lora/curate_dataset.py`
「图片目录」→ LoRA 数据集（复用 `cpp/face/facecli` + `cpp/florence2/img2prompt`，都走**批量**=模型只加载一次）：

1. **检测筛**：有人脸 + **单脸**（多脸/无脸剔）+ **是写真人像**（Florence 反推描述，排除 动漫/插画/截图/海报/logo）+ 画质(Laplacian 方差) + 去重(dHash)
2. **统一尺寸**：**宽 = 训练宽**（style 1536 / face 512，`--train-width`），**高按原比例**；**横图(宽>高)以人脸为锚裁竖构图**（左右留余量、人物大，`--land-aspect`/`--face-scale`）
3. **自然语言打标**：Florence 批量（style `<DETAILED_CAPTION>`，face `<CAPTION>`）
4. 产出 `<out>/images/` + 同名 `.txt` + `dataset.toml` + `report.csv`（每张判据/原因）

关键参数：`--mode style|face`、`--train-width`、`--land-aspect`(默认0.75)、`--face-scale`(默认6)、`--max-faces`(默认1)、`--rm-rejected`(删不合格原图)、`--rm-source`(处理完删源图)、`--copy-to`。

### 20.2 步数怎么定（通用最佳实践）
- **总步数 = 图片数 × repeats × epochs ÷ batch_size**。例：200 张×1×8 = 1600 步；几十张的小集 8 epochs ≈ 400 步也够。
- **epochs 8 是常见起点**（要更充分/复杂风格 10~16），**不是固定值**——最终按**出图盲测**定。
- **黄金法则**：`--save_every_n_epochs 1`（每轮存 checkpoint），训完**出图盲测挑最优**（最优往往不是最后一步；本项目 `musubi_train.sh` 已每轮存）。
- **rank/alpha**：人像/简单风格 **16~32**（复杂/DiT 可 64+），`alpha=rank`；**lr**：DiT/SDXL `5e-5~2e-4`（本项目实测 **1e-4**）；优化器 `adamw8bit`；bf16；调度 cosine + warmup 0.05。
- `musubi_train.sh` 支持 `STEPS`(max_train_steps) 或 `EPOCHS`；`lora.sh` 用 `LORA_STYLE_STEPS`/`LORA_FACE_STEPS` 覆盖（不设则走 epochs）。**没有捷径**：正规训练就是每人上千步、数小时。

### 20.3 my-agent 集成：`tools/lora.sh train <图片目录> <触发词> --mode style|face|both`
- **both = 图片处理两遍、两个 LoRA 先后训练**：① 写真/整体（缩放→写真人像检测→打标→训练）；② 人脸（基于 ① 的图 → 扣脸→打标→训练）
- 合格图存 `<源目录>/style`、`<源目录>/face`（**已存在即直接复用、跳过处理**）；两个 LoRA 也各存回 `<源目录>`
- 后台跑、每个完成自动发微信；`status` / `list` / `cancel`
- 批量（一批人）：**每个 LoRA 串行**；务必用步数封顶（20.2）

### 20.4 facecli 批量模式（新增）
`facecli detect --batch` / `facecli crop --batch`：从 stdin 每行一个路径，**模型只加载一次**——单张 2.9s → ~0.1–0.7s。

