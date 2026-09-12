# TODO — 待验证项

## 需要模型文件才能验证的节点

以下节点代码已实现，但本机 `/data/models` 缺少对应模型文件，**尚未运行验证**。
取得模型后按「验证方法」逐项测试。

### GLIGEN（`GLIGENLoader` / `GLIGENTextBoxApply`）
- 需要的模型：GLIGEN checkpoint（含 `position_net.*` 与 `*.fuser.*` 权重，SD1.5/SDXL 版）
- 验证方法：workflow = CheckpointLoaderSimple → GLIGENLoader → GLIGENTextBoxApply → KSampler，确认区域物体按 box 出现
- 期望：带 GLIGEN 与不带时输出明显不同；box 位置物体语义正确

### StyleModel（`StyleModelLoader` / `StyleModelApply`）
- 需要的模型：含 `style_embedding` 的 StyleAdapter，或含 `redux_down.weight` 的 Flux Redux 模型
- 验证方法：workflow = 加载 style model → StyleModelApply 到 positive conditioning → KSampler
- 备注：现有 `t2iadapter_*.pth` 是 ControlNet 型，走 `ControlNetLoader`（已支持），不是 style model

### unCLIP（`unCLIPConditioning`）
- 需要的模型：unCLIP prior（`model.diffusion_model.*` 为 prior transformer 的 checkpoint）
- 验证方法：workflow = unCLIPCheckpointLoader → CLIPVisionLoader → unCLIPConditioning → KSampler

### SVD_img2vid（`SVD_img2vid_Conditioning`）
- 需要的模型：Stable Video Diffusion（svd / svd_xt）
- 验证方法：SVD 视频管线，image → conditioning → 视频采样

### CLIPVisionEncode
- 架构障碍：sd.cpp 的 clip_vision 从主模型权重表构建，节点无 `model` 输入拿不到 pipeline
- 已备 C API：`sd_clip_vision_encode` / `sd_pipeline_clip_vision_encode`
- 待办：给 `CLIPVisionLoader` 增加 standalone clip_vision ctx，或让节点携带 pipeline

## 可用现有模型验证的
- `ModelMerge*`：可用 Juggernaut-XI / DreamShaperXL / RealVisXL 做不同模型合并验证
- `ModelNoiseScale` / `ModelSamplingDiscrete`：已用 sd_xl_base 验证
