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

## 已完成并验证
- `ModelNoiseScale`：euler_a 下 noise_scale=8 与基线不同
- `ModelSamplingDiscrete`：eps 与基线一致、v_prediction 不同
- 区域条件（`ConditioningSetArea` 等）：左上区域统计与其余不同，峰值显存 ~2.7GB
- `ModelMerge*` 自合并 ratio=0.5 与基线逐字节一致
- `CLIPMergeSimple` 自合并 ratio=0.5 与基线逐字节一致
- `CheckpointSave` 导出 6.9GB 权重
