# sd.cpp 集成 z_image (DiT) ControlNet — 方案与实施计划

> 目标：让 sd.cpp（`backup.sh` 出图链路）支持 **Z-Image 的 ControlNet**（Alibaba **Fun-ControlNet Union**），
> 从而在现有 C++ 链路里做结构控制（Canny/Depth/Pose/Lineart/Tile…）。
>
> 参考实现（优先级）：**ComfyUI 自带** `comfy/ldm/lumina/controlnet.py` + `comfy_extras/nodes_model_patch.py`
> （Z-Image 在 ComfyUI 即 Lumina 实现，见 §2.1）；辅以 diffusers
> `models/controlnets/controlnet_z_image.py` + `pipelines/z_image/pipeline_z_image_controlnet.py`。
> 权重 `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`。

---

## 1. 现状

| 项 | 状态 |
|----|------|
| z_image ControlNet 权重 | ✅ 已有（Fun-ControlNet Union 2.1：full 6.7G / lite 2G / 8steps 变体） |
| sd.cpp ControlNet | ❌ 仅 **UNet**（`src/model/diffusion/control.hpp`：`ControlNetBlock(version=SD1/SD2/SDXL/SVD)`） |
| sd.cpp z_image | ✅ DiT（`src/model/diffusion/z_image.hpp`），但无 control 分支 |
| diffusers | ✅ `ZImageControlNetModel` + `ZImageControlNetPipeline`（可作移植参考/数值基准） |

`src/pipeline/model_builders.cpp:build_control_net_runner` 用**主模型 version** 构造 `ControlNet`，
z_image 走到这里会当成 UNet 处理 → 权重对不上，无法加载 Fun-ControlNet。

---

## 2. Fun-ControlNet Union 架构（diffusers 参考）

- **control_layers**：在 `control_layers_places`（层号列表）放若干 `ZImageControlTransformerBlock`
  - `block_id==0` 有 `before_proj`（zero-init），每块有 `after_proj`（zero-init）
  - 结构 = 单流注意力 + FFN + RMSNorm +（可选 adaLN modulation）
- **control_all_x_embedder**：控制图 latent → tokens
- **control_noise_refiner**（可选，`add_control_noise_refiner` 三种模式）
- **共享**主 transformer 的：`t_embedder / all_x_embedder / rope_embedder / noise_refiner /
  context_refiner / x_pad_token / cap_pad_token`（`from_transformer()` 直接引用）
- `forward(x, t, cap_feats, control_context, conditioning_scale)` →
  `{layer_idx: residual*scale}`（per-layer 残差）
- 采样：把残差加到主 DiT 对应层的 hidden states（`control_layers_places`）

---

## 2.1 权威参考：ComfyUI 自带 z_image ControlNet（Lumina/NextDiT 同源）

Z-Image 在 ComfyUI 走的是 **Lumina** 实现（`comfy/ldm/lumina/`），因此已有可直接对照的完整实现：

| 文件 | 内容 |
|------|------|
| `ComfyUI/comfy/ldm/lumina/controlnet.py` | `ZImageControlTransformerBlock`（继承 `JointTransformerBlock`）+ `ZImage_Control` |
| `ComfyUI/comfy_extras/nodes_model_patch.py` | 权重检测/键名转换 `z_image_convert()`、`ZImageControlPatch`（注入 hook）、节点 `ZImageFunControlnet` |

### 网络结构（controlnet.py）

- `ZImage_Control(dim=3840, n_heads=30, n_kv_heads=30, multiple_of=256, ffn_dim_multiplier=8/3,
  norm_eps=1e-5, qk_norm=True, n_control_layers=6, control_in_dim=16, additional_in_dim=0,
  refiner_control=False, broken=False)`
- `control_layers`：`n_control_layers` 个 `ZImageControlTransformerBlock`
  - `block_id==0` 额外有 `before_proj`（`c = before_proj(c) + x`），每块有 `after_proj`
  - forward 返回 `(c_skip = after_proj(c), c)`
- `control_all_x_embedder["2-1"]`：`Linear(2*2*(control_in_dim+additional_in_dim), dim, bias=True)`
  （默认 `16*4=64 → 3840`）
- `control_noise_refiner`：2 个块；`refiner_control=False` 时是普通 `JointTransformerBlock`，
  为 `True` 时是带 `before_proj/after_proj` 的 `ZImageControlTransformerBlock`

### 前向与注入（精确算法）

1. `control_context` = **控制图经 VAE 编码的 latent**（16ch, `[B,C,H,W]`）。
2. `ZImage_Control.forward(cap_feats, control_context, x_freqs_cis, adaln_input)`：
   - 先 patchify：`control_all_x_embedder["2-1"](...)` → tokens
   - `refiner_control=False` 时再跑 2 个 `control_noise_refiner` → 得运行态 `control_context`
   - `pe`（rope/`x_freqs_cis`）与 `vec`（`t_embedder`+`adaLN`）由**主模型算好传入**，control 侧不重算。
3. 主模型每个 block 上挂 hook（`nodes_model_patch.py:551-585`），主 DiT 共 30 层：
   - `div = round(30 / n_control_layers)`；`cnet_index = block_index // div`
   - `noise_refiner` 块：`forward_noise_refiner_block(...)` 推进，再把 `c_skip` 乘 `strength`
     加到 `img` 残差：`img[:, :L] += c_skip * strength`
   - 主层：`forward_control_block(...)` 依次推进到 `cnet_index`，命中时 `img += c_skip * strength`；
     当 `cnet_index_float > n_control_layers-1` 后停止（`temp_data=None`）

### 权重变体与检测（`nodes_model_patch.py:263-280`）

以 `control_all_x_embedder.2-1.weight` 存在为标志；不同 checkpoint 的 config：

| 条件 | n_control_layers | additional_in_dim | refiner_control |
|------|------------------|-------------------|-----------------|
| 默认（Union-2.1 lite/full） | 6 | 0 | False |
| 无 `control_layers.4.adaLN_modulation.0.weight` | 3 | 17 | True |
| 有 `control_layers.14.adaLN_modulation.0.weight` | 15 | 17 | True（`broken`=refiner.0.after_proj 全 0） |

### diffusers → comfy 键名转换（`z_image_convert`，`nodes_model_patch.py:203`）

```
.attention.to_out.0.weight/bias  → .attention.out.weight/bias
.attention.norm_k.weight         → .attention.k_norm.weight
.attention.norm_q.weight         → .attention.q_norm.weight
to_q + to_k + to_v (cat dim0)    → .attention.qkv.weight
```

→ 与 sd.cpp `z_image.hpp` 的 `JointTransformerBlock` 命名一致，移植时用同一套映射。

---

## 3. sd.cpp 集成点

| # | 文件 | 改动 |
|---|------|------|
| 1 | `src/model/diffusion/z_image.hpp`（或新增 `z_image_control.hpp`） | 新增 `ZImageControlModel` + `ZImageControlRunner`；control block = 复用 `JointTransformerBlock` 结构 + `before_proj/after_proj`；`control_all_x_embedder`、`control_noise_refiner` |
| 2 | `src/pipeline/model_builders.cpp` | `build_control_net_runner` 按 `version` 分派：z_image → `ZImageControlRunner`；其余 → 原 `ControlNet` |
| 3 | `src/pipeline/diffusion_engine.cpp` | z_image 采样分支：每步调 control runner 得 per-layer residual，注入 `z_image.forward` 对应层（现 UNet 的 block-sample 注入可作模板，见 `control_net->compute` 附近 ~L2200） |
| 4 | 权重加载 | Fun-ControlNet safetensors 键名（`control_layers.*`、`control_all_x_embedder.*`、`control_noise_refiner.*`）→ `ModelManager` 注册/前缀映射；共享模块复用主 z_image 的 tensor |
| 5 | C API | **无需改签名**：`control_net_path` / `control_image` / `control_strength` 已在 `sd_ctx_params_t`/`sd_img_gen_params_t` |
| 6 | 共享权重 | control runner 需引用主 z_image runner 的 `t_embedder/noise_refiner/...`；sd.cpp 里跨 runner 共享 tensor 需设计（可能把 control 做成 z_image runner 的一个附加分支） |

---

## 3.5 Phase A 勘察结果（Fun-ControlNet Union **lite** safetensors）

共 **91** 个张量，三类前缀，子模块命名与 z_image 块同构（仅前缀不同）：

| 前缀 | 数量 | 键名示例 |
|------|------|----------|
| `control_all_x_embedder.*` | 2 | `control_all_x_embedder.2-1.weight/.bias` |
| `control_layers.{0..N}.*` | 53 | `.before_proj/.after_proj.{weight,bias}`、`.adaLN_modulation.0.*`、`.attention.{to_q,to_k,to_v,to_out.0,norm_q,norm_k}`、`.attention_norm1/2`、`.feed_forward.{w1,w2,w3}`、`.ffn_norm1/2` |
| `control_noise_refiner.{0,1}.*` | 36 | 同上块结构 |

→ 子模块名（`attention.to_q` / `feed_forward.w1` / `attention_norm1` …）与 `z_image.hpp` 的
`JointTransformerBlock` **一致**，control 特有的只有 `before_proj`/`after_proj` + `control_all_x_embedder`
+ `control_noise_refiner`。移植时前缀映射 + 复用块结构即可，Phase A 无阻塞。

## 4. 实施结果（Phase B–D 已完成并验证）

### 权威配置（`hlky/Z-Image-Turbo-Fun-Controlnet-Union-2.1/config.json`）

```json
{ "add_control_noise_refiner": "control_noise_refiner", "control_in_dim": 33,
  "control_layers_places": [0,2,4,...,28], "control_refiner_layers_places": [0,1] }
```

lite-2601-8steps 实测：`n_control_layers=3`、`additional_in_dim=17`（`control_in_dim=33`）、
`refiner_control=True`；用 `div=round(30/3)=10` → 主层注入点 `[0,10,20]`。

### 代码落点（`/opt/sd`）

- `src/model/diffusion/z_image.hpp`：`ZImageControlConfig/ZImageControlBlock/ZImageControlModel`；
  `ZImageModel::forward_core` 内联控制分支；`ZImageRunner` 持有并加载 control（空前缀）。
- `src/model/diffusion/model.hpp`：新增 `ZImageDiffusionExtra{control_context, control_strength}`。
- `src/pipeline/image.cpp`：z_image 时把控制图 VAE 编码 + 补零到 `control_in_dim`，作为 control context。
- `src/pipeline/diffusion_engine.cpp`：z_image 分支填 `ZImageDiffusionExtra`；control net 加载/卸载走 Core 重建。
- `src/pipeline/model_builders.cpp`：z_image 跳过 UNet ControlNet runner。
- 适配层 `cpp/sd/examples/img_hires.cpp` + `backup.sh`：`--control-net/--control-image/--control-strength`
  （env: `CONTROL_NET/CONTROL_IMAGE/CONTROL_STRENGTH`，默认 0.75）。

### 关键正确性要点（易错）

1. **每层只注入一次**：第 k 个 control layer 的 residual 只在 `block_index % div == 0` 的主层注入
   （不要在该 range 内每层都加，否则强约 10× → 0.1 才正常的假象）。
2. **control layers 跑在 unified `[cap, img]` 上**（含文本 token），与 diffusers 一致；
   `control_noise_refiner` 仍只看图像 token，跑在主 noise_refiner 处。
3. **额外 17 通道补零**（diffusers 行为）；`additional_in_dim=17` 由 `control_all_x_embedder` 输入维度
   `132 = 4×33` 推出。
4. 键名转换：`attention.norm_q/k → q_norm/k_norm`（见 `cpp/sd/scripts/convert_controlnet.py`）。
5. `control_context` 与采样 latent 同空间（Flux 缩放，实测 rms≈1.3 与 diffusers 一致）。

### 验证

canny（`asset/canny.jpg`）→ 512×512、8 步、cfg 1.0：strength 0.5/0.75/1.0 均得到结构正确、
干净的猫；`backup.sh --control-net ... --control-image ...` 出图正常。

### lite vs full 15 层（实测）

| 变体 | 权重 | control layers | 表现 |
|------|------|----------------|------|
| lite-2601-8steps | 2GB | 3（注入点 0/10/20）| canny 0.75 干净；**pose 弱**（骨架→服装形）|
| full-2.1-8steps | 6.7GB | 15（注入点 0/2/4…/28）| **pose 0.75 出正确人体**；canny 偏强，用 ~0.4 |

→ 需要 pose/强结构控制用 **full**；追求速度/体积用 lite（canny/depth 足够）。
权重：`/data/models/image/z_image_turbo_fun_controlnet_union_{2.1_lite_2601,2.1}_8steps_sdcpp.safetensors`。

---

## 5. 工作量 / 风险

- **大**：`z_image.hpp` 的 patchify / rope / noise_refiner / context_refiner / unified 结构较复杂，
  control 分支要 1:1 复刻并正确共享权重 → 非小改。
- 依赖 **权重键名逐一对齐** + **数值对齐测试**（否则静默出图错乱）。
- 建议先做 **Phase A/B** 打通加载与单步数值，再决定是否继续 C/D。

## 6. 替代方案（若移植成本过高）
- **diffusers 推理**跑 Fun-ControlNet（Python，脱离 sd.cpp 链路）。
- z_image 用 **img2img（init_image+denoise）/ Qwen 参考重绘** 做结构控制（现成，零改动）。
- 等 sd.cpp 上游支持 z_image ControlNet 后再接入。

---

**状态**：✅ 已实现并端到端验证（见 §4）。sd.cpp 支持 Z-Image Fun-ControlNet（lite-2601-8steps 实测），
单阶段与 hires 两阶段（`backup.sh --hires`）均可用；`backup.sh` 通过
`--control-net/--control-image/--control-strength` 做 canny/depth/pose 等结构控制。

hires 说明：控制图在第二阶段按目标分辨率**重新 VAE 编码**（`build_z_image_control_context`），
使 control token 数与主图 token 数对齐（同 ComfyUI 在 latent 尺寸变化时重编码的行为）。
