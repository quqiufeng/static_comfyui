# sd.cpp 集成 z_image (DiT) ControlNet — 方案与实施计划

> 目标：让 sd.cpp（`backup.sh` 出图链路）支持 **Z-Image 的 ControlNet**（Alibaba **Fun-ControlNet Union**），
> 从而在现有 C++ 链路里做结构控制（Canny/Depth/Pose/Lineart/Tile…）。
>
> 参考实现：diffusers `models/controlnets/controlnet_z_image.py` +
> `pipelines/z_image/pipeline_z_image_controlnet.py`；权重 `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`。

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

## 4. 分阶段实施

1. **Phase A — 加载**（✅ 勘察完成，见 §3.5）：键名已确认，与 z_image 块同构。下一步把 control 文件交给
   `ModelLoader` 并用调试打印确认张量被读到。
2. **Phase B — 构建 control runner**：实现 `ZImageControlModel`（control_layers/embedder/refiner），
   数值对齐：与 diffusers 同输入单步输出对比。
3. **Phase C — 采样注入**：在主 z_image `forward` 的对应层叠加 residual × conditioning_scale。
4. **Phase D — 端到端**：`sd-cli`/`backup.sh` 传 control 图（canny/depth）出图，与 diffusers 参考对比。

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

**状态**：方案落档，未开始编码。下一步：Phase A（权重键名勘察 + 加载验证）。
