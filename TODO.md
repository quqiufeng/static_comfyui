# TODO

## 待验证（代码已实现，缺模型文件）

以下节点代码已实现，但本机 `/data/models` 无对应模型，**尚未运行验证**。

### GLIGEN（`GLIGENLoader` / `GLIGENTextBoxApply`）— 部分实现
- 已完成：`src/model/diffusion/gligen.hpp`（`GatedSelfAttentionDense`/`PositionNet`/`GligenModules`）；UNet transformer 块注入（`BasicTransformerBlock` self-attn 后，`transformer_index` 递增）；`GGMLRunnerContext`/`UNetDiffusionExtra`/`StableDiffusionGGML` 传递链；`sd_ctx_params_t.gligen_path`
- **待完成**：GLIGEN 权重加载 runner（含 `alpha_attn`/`alpha_dense` 标量）、PositionNet objs 计算（Fourier + MLP）、节点接线
- 需要模型：GLIGEN checkpoint（含 `position_net.*` 与 `*.fuser.*`）
- 验证：CheckpointLoader → GLIGENLoader → GLIGENTextBoxApply → KSampler，确认区域物体按 box 出现

### StyleModel（`StyleModelLoader` / `StyleModelApply`）
- 需要模型：含 `style_embedding` 的 StyleAdapter，或含 `redux_down.weight` 的 Flux Redux
- 备注：现有 `t2iadapter_*.pth` 是 ControlNet 型，走 `ControlNetLoader`（已支持），不是 style model
- 验证：style model → StyleModelApply → KSampler

### unCLIP（`unCLIPConditioning`）
- 需要模型：unCLIP prior（`model.diffusion_model.*` 为 prior transformer）

### SVD_img2vid（`SVD_img2vid_Conditioning`）
- 需要模型：Stable Video Diffusion（svd / svd_xt）；sd.cpp 支持 SVD，需走视频管线

### CLIPVisionEncode
- 架构障碍：sd.cpp 的 clip_vision 从主模型权重表构建，节点无 `model` 输入拿不到 pipeline
- 已备 C API：`sd_clip_vision_encode` / `sd_pipeline_clip_vision_encode`
- 待办：给 `CLIPVisionLoader` 增加 standalone clip_vision ctx，或让节点携带 pipeline

## 待验证（有模型，可直接测）
- `ModelMerge*`：用 Juggernaut-XI / DreamShaperXL / RealVisXL 做不同模型合并验证
- `ModelMergeSimple/Blocks/Add/Subtract`：ratio / 逐块 ratio 行为
- `CLIPMerge*`：不同 CLIP 合并

## 采样加速（已完成，2026-09-24）
**目标**：E1xMIN 2560×1440 出图 ~11min → ≤5min。

| 项 | 原理 | 结果 |
|----|------|------|
| **EasyCache**（默认开） | 相邻步 latent 变化 < 阈值则复用、跳过本步 forward；turbo 后期更易命中 | base 跳 9/20，hires 跳 ~30/41；**主因** |
| **GGML_CUDA_GRAPHS=ON** | 一步采样 kernel 序列录成 CUDA graph 一次提交，减 launch 开销 | 再省数秒～十数秒（`build_sd_dl.sh`） |
| 分段计时 | `img_hires` 打 load/generate/post/save + EasyCache summary | 便于回归对比 |

**实测**（RTX 3080 20G，seed=25630，backup.sh 默认提示词）：
- 优化前 ~665s → **210s（3m30s，~3.2×）**；hires 采样 531s→142s。
- 开关：`CACHE_MODE=disabled|easycache`（默认 easycache）；`CACHE_THRESHOLD` 越低跳越多（默认 0.2）。
- 接线：`ImageGenerationParams.cache_*` → `sd_img_gen_params_t.cache` → `SampleCacheRuntime`；CLI `--cache-mode/--cache-threshold/--cache-start/--cache-end`。
- 未做：batch CFG（`z_image` 断言 `x->ne[3]==1`）、降 hires steps（画质换速度）。

## HiRes Fix 出图质量优化
**原理**：latent 放大 + 二次采样（denoise<1），即 ComfyUI 的 `LatentUpscale`（bicubic/bislerp）+ `KSampler(denoise)`；`backup.sh` 同原理（低分构图 → latent 放大 refine，基础分辨率越高、放大倍数越小越好）。

已修（对齐 `cpp/sd/backup.sh` 配方）：
- `img_hires.cpp` 默认值：quality prefix 原样对齐 backup.sh；默认负面词对齐 backup.sh；
  `--hires-steps` 默认 20→45；后处理默认开启并取 backup.sh 值（clarity 0.2 / sharpen 0.3 /
  smart 0.5 / edge 1.5）
- `HiResFix` 节点：prompt 未含 masterpiece 时前置 backup.sh 质量关键词（`quality_prefix=0` 可关）；
  负面词留空时补 backup.sh 默认负面词（`default_negative=0` 可关）
- 模型相关项（cfg / sampler / scheduler / FreeU / VAE tiling）仍由调用方显式传入，
  不设为默认，避免影响 SDXL 等不同模型的正常区间

历史修复（`23d9050` + `c950631`）：
- 修复 `hires.model_path` 从未设置 / 枚举串大小写不匹配（"Model" vs "model"）
- 默认改回 **latent-bicubic**（sd.cpp 的 `LATENT` 实为 Bilinear 偏软，ComfyUI 常用 bicubic）
- 后处理默认关闭（旧 backup.sh 不做后处理）
- base 分辨率计算与 backup.sh 一致（2560×1440→1920×1080 等）
- ESRGAN 保留为可选（`upscaler=model` + `upscaler_model`）

仍待做：
- **二次采样参数**：`hires_strength`/`hires_steps`/`scheduler`/`cfg` 已对齐 backup.sh（0.35/45/…），如需更好需跑实验标定
- **FreeU 默认值**：已对齐 backup.sh（b1=1.3/b2=1.4）；旧 sdxl_pipeline 曾记录过拟合，待复标
- **VAE tiling 接缝**：参数已对齐（128/0.5），高分辨率 tile 边界若有接缝需查 sd.cpp tiling
- **bislerp**：已实现（`upscaler=latent-bislerp`，对齐 ComfyUI `bislerp`）；默认仍 bicubic
- **多步渐进放大** / **与 ComfyUI 同工作流质量对照**

## 已完成并验证
- `ModelNoiseScale`：euler_a 下 noise_scale=8 与基线不同
- `ModelSamplingDiscrete`：eps 与基线一致、v_prediction 不同
- 区域条件（`ConditioningSetArea` 等）：左上区域统计与其余不同，峰值显存 ~2.7GB
- `ModelMerge*` 自合并 ratio=0.5 与基线逐字节一致
- `CLIPMergeSimple` 自合并 ratio=0.5 与基线逐字节一致
- `CheckpointSave` 导出 6.9GB 权重
