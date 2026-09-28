# cpp/sd — stable-diffusion.cpp 集成与升级指南（AI 阅读版）

本文档给 AI 阅读者理解本项目与 sd.cpp 的集成边界，以及升级 sd.cpp 版本时需要检查的所有文件。

---

## 1. 目录结构（实际存在的）

```
/opt/static_comfyui/cpp/sd/
├── design.md                       # 本文档
├── CMakeLists.txt                  # 适配层 cmake 构建
├── SD_VERSION.lock                 # 锁定 /opt/sd commit hash
├── build_sd.sh                     # 编译 sd.cpp（静态链接，旧，已废弃）
├── build_sd_dl.sh                  # 编译 sd.cpp（动态后端，当前默认）
├── patches/
│   └── sdcpp-freeu-sag-v2.patch    # 唯一 patch：FreeU/SAG/DynCFG/RescaleCFG/video CFG/FreSca/sigma 区间/区域条件/GLIGEN 等（15 文件 1632 行）
├── src/
│   ├── adapters/
│   │   ├── sdcpp_adapter.h         # C++ SDPipeline 类 + C API 声明
│   │   └── sdcpp_adapter.cpp       # 唯一 #include <stable-diffusion.h> 的文件
│   └── postproc/
│       ├── postproc.h              # 后处理参数结构体
│       └── postproc.cpp            # 锐化/清晰度实现（不依赖 sd.cpp）
├── examples/
│   ├── sdxl_txt2img.cpp            # 独立测试程序
│   └── img_hires.cpp               # HiRes Fix 测试程序
├── tests/
│   ├── run.sh                      # FreeU/FreSca 数学自测（编译 + 运行，依赖 build_sd_dl.sh）
│   ├── test_fourier.cpp            # Fourier_filter: ggml 闭式投影 vs 朴素 DFT
│   ├── test_freeu_v2.cpp           # FreeU_V2 backbone vs torch 参考（含 batch=2）
│   └── test_fresca.cpp             # FreSca 频带滤波 vs 朴素 DFT（奇偶尺寸/截断/空盒）
└── scripts/
    └── build.sh                    # 编译适配层（被根目录 build.sh 调用）
```

其他目录（`api/`, `native/sampler/`, `native/attention/`, `scripts/upgrade_sdcpp.sh` 等）是蓝图，尚未创建。

---

## 2. 依赖关系

```
StaticPy extern fn ← sdcpp_adapter.h (C API) ← sdcpp_adapter.cpp ← stable-diffusion.h
```

- **只有 `sdcpp_adapter.cpp`** 可以 `#include <stable-diffusion.h>`
- **只有 `sdcpp_adapter.h`** 定义被 StaticPy 调用的 C API 函数签名
- **只有 `sd_backend.static.py`** 声明对应的 `extern fn` 和 wrapper 函数
- **上层 StaticPy 代码**（`nodes.static.py`、`execution.static.py`、`main.static.py`）**完全不接触 sd.cpp API**

---

## 3. 本项目的文件 vs sd.cpp 的边界

| 项目内文件 | 职责 | 和 sd.cpp 的关系 |
|-----------|------|-----------------|
| `sdcpp_adapter.h` | 定义 `sd::ModelConfig`、`sd::ImageGenerationParams`、`sd::SDPipeline` 类；声明 C API | 手动映射 `sd_ctx_params_t` / `sd_img_gen_params_t` 的字段 |
| `sdcpp_adapter.cpp` | `SDPipeline::load()` 构建 `sd_ctx_params_t` 并调用 `new_sd_ctx`；`SDPipeline::generate()` 构建 `sd_img_gen_params_t` 并调用 `generate_image`；VAE tile cap 在传入前就地裁剪 | **唯一接触 sd.cpp C API 的代码** |
| `sd_backend.static.py` | 声明 `extern fn` 和 Python 风格 wrapper | 签名必须和 `sdcpp_adapter.h` C API 一致 |
| `nodes.static.py` | ComfyUI 节点实现 | 只调用 `sd_backend.static.py` 的 wrapper |

---

## 4. 我们对 sd.cpp 的改动

所有改动集中在 **一个 patch 文件** `patches/sdcpp-freeu-sag-v2.patch`（1632 行），修改 sd.cpp 的 **15 个文件**（含新增 `gligen.hpp`、`ggml_fourier.h`、`freq_filter.h`）。当前基准 commit：**`6dcb5bb`**（见 `SD_VERSION.lock`）。

| patch 中的文件 | 新增行数 | § |
|---|---|---|
| `include/stable-diffusion.h` | 71 | 4.1 |
| `src/core/ggml_fourier.h`（新增） | 185 | 4.2 |
| `src/model/diffusion/unet.hpp` | 81 | 4.2 |
| `src/pipeline/diffusion_engine.h` | 59 | 4.3 |
| `src/pipeline/diffusion_engine.cpp` | 219 | 4.4 / 4.7 |
| `src/pipeline/image.cpp` | 198 | 4.5 |
| `src/model_loader.cpp` | （已删除） | 4.6 |
| `src/model/diffusion/model.hpp` | 3 | 4.8 |
| `src/model/diffusion/gligen.hpp`（新增） | 136 | 4.9 |
| `src/core/ggml_runner.h` | 12 | 4.10 |
| `src/model/common/block.hpp` | 4 | 4.11 |
| `src/runtime/denoiser.hpp` | 21 | 4.12 |
| `src/stable-diffusion.cpp` | 19 | 4.13 |
| `src/runtime/freq_filter.h`（新增） | 115 | 4.17 |
| `src/runtime/guidance.h` | 15 | 4.17 |
| `src/runtime/guidance.cpp` | 32 | 4.17 |

> **注意**：sd.cpp 在 `7f410a3` 做了大重构（#1956/#1957），生成管线从 `src/stable-diffusion.cpp` 拆到 `src/pipeline/`。patch 已随之重定位。`6dcb5bb` 起上游又移除了 `unused_tensors`/`vision_model.` 过滤（#1984），旧的 `model_loader.cpp` hunk 已废弃、从 patch 中删除。

### 4.1 `include/stable-diffusion.h`
新增 4 个结构体 + 在 `sd_img_gen_params_t` 末尾追加对应字段：
- `sd_freeu_params_t`：`{enabled, b1, b2, s1, s2}`
- `sd_sag_params_t`：`{enabled, scale}`
- `sd_dynamic_cfg_params_t`：`{enabled, percentile, mimic_scale, threshold_percentile}`
- `sd_fresca_params_t`：`{enabled, scale_low, scale_high, freq_cutoff}`（FreSca 频域 guidance，§4.17）

> IP-Adapter 现用 **sd.cpp 原生**（`sd_ctx_params_t.ip_adapter_path` + `sd_img_gen_params_t.ip_adapter_image/ip_adapter_strength`），不再自定义结构体/注入。

### 4.2 `src/model/diffusion/unet.hpp` + `src/core/ggml_fourier.h`
两处独立改动：

**A. `UnetModelBlock` 类**（FreeU 核心计算）
- 新增 5 个字段 + `set_freeu()` 方法
- 在 `forward()` 的 output block 中加入 FreeU，**1:1 复刻 ComfyUI `FreeU_V2`**
  （`comfy_extras/nodes_freelunch.py`）：
  - **backbone**：`h[:, :ch//2] *= (b-1)*hidden_mean + 1`，其中 `hidden_mean` = 通道均值
    在空间维按 min/max 归一化到 `[0,1]`（`ggml_ext_freeu_v2_backbone`）
  - **skip**：`Fourier_filter(hsp, threshold=1, scale=s)`——频谱 fftshift 后中心 2×2 bin
    乘 `s`（`ggml_ext_fourier_filter_lowfreq`）。不引 FFT，用等价闭式投影
    `x' = x + (s-1)·Re{P(x)}`，O(H·W·C)（见 `ggml_fourier.h` 头注释）
  - `channel == model_channels*4` → 用 `{b1,s1}`；`channel == model_channels*2` → 用 `{b2,s2}`
  - **仅 UNet 生效**：`sd_version_is_unet()`（SD1/2/SDXL）才 `set_freeu_params`；z_image/Qwen/Flux/SD3
    是 DiT，ComfyUI 同样只挂 UNet → 恒为空操作（`backup.sh` 等 DiT 脚本因此不传 `--freeu`）
  - 数学自测：`tests/run.sh`（`test_fourier` ggml vs 朴素 DFT，f32/f16/奇偶尺寸；
    `test_freeu_v2` vs torch 参考，含 batch=2）均 `ALL OK`
  - **关键设计**：`UnetModelBlock` 只存储 FreeU 参数，不关心来源

**B. `UNetModelRunner` 类**（FreeU 参数传递层）
- 新增 5 个字段 + `set_freeu_params()` 方法
- `compute(DiffusionParams&)` 中用 `this->freeu_*` 调用 `unet.set_freeu()`
- **不修改 `DiffusionParams`**——FreeU 字段直接挂在 runner 上

### 4.3 `src/pipeline/diffusion_engine.h`
`StableDiffusionGGML` 类新增字段（各功能默认 disabled）：
- FreeU: `freeu_enabled`, `freeu_b1/b2/s1/s2`
- SAG: `sag_enabled`, `sag_scale`
- Dynamic CFG: `dynamic_cfg_enabled`, `dynamic_cfg_percentile/mimic_scale/threshold_percentile`
- FreSca: `fresca_enabled`, `fresca_scale_low/scale_high/freq_cutoff`（§4.17）

### 4.4 `src/pipeline/diffusion_engine.cpp`
四处改动：

**A. 顶部 include**：`#include "model/diffusion/unet.hpp"`（FreeU 的 `dynamic_cast<UNetModelRunner*>` 需要完整定义）

**B. `run_condition` lambda 中**——通过 `dynamic_cast<UNetModelRunner*>` + `sd_version_is_unet()` 设置 FreeU：
```cpp
if (sd_version_is_unet(version)) {
    auto* unet_runner = dynamic_cast<UNetModelRunner*>(work_diffusion_model.get());
    if (unet_runner) {
        unet_runner->set_freeu_params(freeu_enabled, ...);
    }
}
```

**C. 采样循环的 post-compute 阶段**——插入 SAG + Dynamic CFG：
- SAG：`guided.pred = pred * scale + uncond * (1-scale)`
- Dynamic CFG：找到 `pred` 最大绝对值，若 >1 则全张量除以该值

**D. `ClassifierFreeGuidance` 构造处**（FreSca，§4.17）——把 `fresca_*` 字段打包成 `FreqEnhanceParams` 传入 CFG；`forward()` 内对 guidance 差值做频带滤波后再乘 `guidance_scale`。

### 4.5 `src/pipeline/image.cpp`
一处改动：`generate_image()` 中从 `sd_img_gen_params_t` 读取 freeu/sag/dynamic_cfg/rescale_cfg/video_cfg/**fresca** 存入 `sd->*` 字段。

### 4.6 `src/model_loader.cpp` ⚠️ 已从 patch 移除（`6dcb5bb`）
历史修复（从 `unused_tensors` 移除 `"vision_model."`）在 `6dcb5bb` 上游已通过 #1984 删除整个 `unused_tensors[]` 过滤逻辑，hunk 自动作废，不再需要 patch。若将来 patch 再出现 `model_loader.cpp` 冲突，优先确认上游是否已删除该过滤。

### 4.7 `src/pipeline/diffusion_engine.cpp`（clip vision 前缀，修复上游回归）
clip vision 加载前缀从 `"clip_vision."` 改回 `"cond_stage_model.transformer."`。`#1957` 重构把旧版的 `cond_stage_model.transformer.` 误改成 `clip_vision.`，而 `FrozenCLIPVisionEmbedder` 仍按 `cond_stage_model.transformer.` 查找。

### 4.8 `src/model/diffusion/model.hpp`
`UNetDiffusionExtra` 追加 2 个 GLIGEN 字段：`gligen_applier`（`void*`，不引入头文件依赖）、`gligen_objs`。
> FreeU 参数**不经过** `DiffusionParams`——`UNetModelRunner` 通过自己的 `set_freeu_params()` 直接接收。

### 4.9 `src/model/diffusion/gligen.hpp`（新增文件）
GLIGEN 的 ggml 模块：PositionNet（Fourier 坐标编码 + MLP）与 gated affine 的前向实现。
> 状态：模块与 UNet 注入已完成，**尚未接线/未验证**（缺权重 loader 与节点接线），见 `TODO.md`。

### 4.10 `src/core/ggml_runner.h`
- 定义抽象接口 `struct GligenApplier { virtual ggml_tensor* apply(GGMLRunnerContext*, ggml_tensor*, int) = 0; }`
- `GGMLRunnerContext` 追加 `gligen_applier` / `gligen_objs` / `gligen_index`

### 4.11 `src/model/common/block.hpp`
Transformer 块中唯一的调用点：在 self-attn 之后、cross-attn 之前插入
`if (ctx->gligen_applier) x = ctx->gligen_applier->apply(ctx, x, ctx->gligen_index);`

### 4.12 `src/runtime/denoiser.hpp`
- `Denoiser` 增加 `sigma_min_override` / `sigma_max_override` + `set_sigma_range()`——
  支持 ComfyUI `ModelSamplingContinuousEDM` / `ModelSamplingContinuousV`（负数=用模型自带区间）
- `sampler_noise_scale()` 全局量——支持 `ModelNoiseScale`（ancestral 噪声 `sigma_up * noise_scale`）

### 4.13 `src/stable-diffusion.cpp`
C API 兜底与注册：
- `sd_hires_upscaler` 枚举追加 `"Latent (bislerp)"`
- `sd_img_gen_params_init` 补默认值 `area_conds` / `area_cond_count` / `noise_scale` / `fresca = {false, 1.0f, 1.25f, 20}`
- 新增导出 `sd_clip_vision_encode()`（CLIPVisionEncode 节点）与 `sd_set_prediction()`（ModelSamplingDiscrete）

### 4.14 `src/model/vae/vae.hpp` — 不移除 ⚠️ 未修改
VAE tile 大小上限已从 patch 中移除，改由 adapter 层在调用 `generate_image` 前自行 cap。详见 §4.16。

### 4.15 架构设计原则：参数流向
```
StaticPy → adapter (sdcpp_adapter.cpp)
  → sd_img_gen_params_t.freeu  (C API struct)
  → StableDiffusionGGML::freeu_enabled  (class fields)
  → UNetModelRunner::freeu_enabled  (via dynamic_cast + set_freeu_params)
  → UnetModelBlock::freeu_enabled  (via unet.set_freeu)
  → ggml 计算图
```
FreeU 参数从右上到左下垂直传递，**不污染**水平方向的现有数据结构（如 `DiffusionParams`）。

### 4.16 不需要改 sd.cpp 的功能（通过原生 C API 调用）
LoRA、ControlNet、HiRes Fix、**VAE tiling（含 tile cap）**、sampler/scheduler 枚举、PhotoMaker、ESRGAN upscale、TAESD——全部通过 `sd_ctx_params_t` / `sd_img_gen_params_t` 的标准字段控制，不需要 patch。

其中 VAE tile cap（防止 OOM）在 `sdcpp_adapter.cpp` 的 `SDPipeline::generate()` 中实现：tile 尺寸上限 128 个 latent 像素（对应 scale=8 的 VAE 输出 1024px）。

### 4.17 `src/runtime/freq_filter.h` + `src/runtime/guidance.{h,cpp}`（FreSca：模型无关的频域 guidance 增强）
ComfyUI `comfy_extras/nodes_fresca.py` 的 1:1 复刻。与 FreeU（U-Net 结构级，靠 skip/backbone）不同，FreSca 挂在 **CFG 层**——对 `cond − uncond` 的 guidance 差值做频带缩放后再乘 `guidance_scale`——与网络结构无关，因此 **UNet 与 DiT（z_image/Qwen/Flux）均生效**。

- **`freq_filter.h`（新增，纯头文件、零依赖）**：
  `out = scale_high·x + (scale_low − scale_high)·Re{IDFT(box·DFT(x))}`，
  box = 频率 `k ∈ [−T, T−1]`（对应 ComfyUI fftshift 后窗口 `[cc−f_c, cc+f_c)`，`T = min(freq_cutoff, n//2)`）。
  掩码可分离（rect⊗rect）→ 拆成沿 W/H 各一次 1D 圆卷积（复 Dirichlet 核
  `g[j] = (1/N)·Σ_{k=−T}^{T−1} e^{i2πkj/N}`，取实部组合），**免 FFT**，
  `O(planes·(H·W² + W·H²))`，实张量上直接 host 计算。
- **`guidance.{h,cpp}`**：新增 `FreqEnhanceParams {enabled, scale_low, scale_high, freq_cutoff}`；
  `ClassifierFreeGuidance` 持有该参数（构造函数第三参，默认关），`forward()` 先算
  `delta = pred_cond − pred_uncond`，启用时对 `delta` 做 `freq_band_filter` 再乘
  `guidance_scale`——与 ComfyUI pre_cfg 语义一致（`uncond + g·F(delta)`）。
  APG 路径未接（APG 默认关，需 `extra_sample_args` 显式开启）。
- **配线**：`sd_fresca_params_t`（§4.1）→ `image.cpp`（§4.5）→ `StableDiffusionGGML::fresca_*`（§4.3）→ CFG 构造（§4.4-D）。只在 plain CFG 生效，hires 第二遍同样生效。
- **暴露面**（与 rescale_cfg 同款 setter，**不改** 5 个 `sd_pipeline_generate*` 签名）：
  - adapter：`SDPipeline::set_fresca()` + C 导出 `sd_pipeline_set_fresca`
  - `img_hires`：`--fresca` / `--fresca-low` / `--fresca-high` / `--fresca-cutoff`
  - comfycli：`FreSca` 节点（`nodes.static.py`，`MODEL` 进 `MODEL` 出；`sd_backend.static.py` extern + wrapper）
  - 脚本：`backup.sh` / `backup_qwen.sh` 默认 `FRESCA=1` 开启（`FRESCA=0` 关，附 `FRESCA_LOW/HIGH/CUTOFF`）
- **测试**：`tests/test_fresca.cpp` vs 朴素 DFT 参考（严格复刻 nodes_fresca.py 的 fftshift/mask/.real），
  覆盖奇偶尺寸、cutoff 超长截断、空盒（cutoff=0）、`low==high`、轴长 1，8 例 max_abs ≤ 1.2e-7。
- **默认**：sd.cpp API / `FreSca` 节点默认关（不传即关闭，零回归）；`backup.sh` / `backup_qwen.sh`
  默认 `FRESCA=1` 开启。开启时默认参数 `1.0/1.25/20`（= ComfyUI 节点默认）。注意：极低步数
  （如 6 步 raw SDXL base）+ 强高频增益会有振铃/条纹，属参数敏感（20 步即正常），不是实现偏差。

---

## 5. 升级 sd.cpp 版本：AI 检查清单

升级目标是升级 `/opt/sd` 到新的 commit，保持本项目能编译并通过验证。

> **为什么不能全自动化**：fetch/编译/回归/更新 lock 是机械步骤，但 **patch rebase、C API 映射、枚举/行为语义判断** 需要理解上游改动意图后人工（AI）介入。本节是给执行者的 playbook。

### 5.0 依赖面（升级前必看）

适配层对 sd.cpp 的依赖集中在 `sdcpp_adapter.cpp`，升级时按以下清单逐一核对。

**用到的函数 / 类型**

| 类别 | 符号 |
|------|------|
| 生命周期 | `new_sd_ctx` / `free_sd_ctx` / `generate_image` / `free_sd_images` |
| 初始化 | `sd_ctx_params_init` / `sd_img_gen_params_init` |
| 字符串→枚举 | `str_to_sample_method` / `str_to_scheduler` / `str_to_sd_hires_upscaler` |
| ControlNet 热插拔 | `sd_ctx_load_control_net` / `sd_ctx_unload_control_net` |
| ADetailer | `new_adetailer_ctx` / `adetail_image` / `free_adetailer_ctx` |
| 日志 | `sd_set_log_callback` |
| 结构体 | `sd_ctx_params_t` / `sd_img_gen_params_t` / `sd_image_t` / `sd_lora_t` / `sd_adetailer_params_t` / `sd_log_level_t` / `sd_type_t` |

**写入 `sd_ctx_params_t` 的字段**（`SDPipeline::load`）

`model_path, clip_l_path, clip_g_path, clip_vision_path, vae_path, diffusion_model_path, llm_path, n_threads, wtype, rng_type, sampler_rng_type, prediction, flash_attn, diffusion_flash_attn, enable_mmap, lora_apply_mode, backend, params_backend`

**写入 `sd_img_gen_params_t` 的字段**（`SDPipeline::generate`）

`prompt, negative_prompt, width, height, clip_skip, seed, batch_count, sample_params.{sample_steps, guidance.{txt_cfg, img_cfg, distilled_guidance}, sample_method, scheduler, eta}, loras/lora_count, vae_tiling_params.{enabled, tile_size_x, tile_size_y, target_overlap}, hires.{enabled, upscaler, target_width, target_height, scale, steps, denoising_strength, upscale_tile_size}, freeu.{enabled, b1, b2, s1, s2}, sag.{enabled, scale}, ipadapter.{tokens, num_tokens, token_dim, weight}, init_image, strength, control_image, control_strength, mask_image`

**常见破坏 → 修法**

| 上游变化 | 症状 | 修法 |
|----------|------|------|
| `sd_ctx_params_t` 新增必填字段 | 加载失败 / 行为异常 | 在 `ModelConfig` 加字段 → `SDPipeline::load` 映射 |
| `sd_img_gen_params_t` 新增字段 | 通常无碍（init 后默认值安全） | 需要时才在 `ImageGenerationParams` 暴露 |
| 结构体字段**改名** | 编译错误（点出字段） | 改适配层对应赋值 |
| 枚举值新增/删除 | 编译错误或落默认 | 检查 `str_to_*` 映射与 fallback |
| `new_sd_ctx` / `generate_image` 签名变化 | 编译错误 | 改 `load()` / `generate()` 调用 |
| FreeU/SAG 被官方合入 | patch 冲突 | 删 patch 对应部分，改用官方字段 |

> **关键抗性**：适配层总是 `sd_*_params_init()` 先零初始化再按字段名赋值，所以"新增字段"通常自动安全；只有"删 / 改名 / 改签名 / 改枚举"才会编译报错，且报错点集中在本文件。

### 5.1 准备

```bash
cd /opt/sd
git fetch origin
git log --oneline origin/master -30     # 查看最近的提交
git diff "$(cat /opt/static_comfyui/cpp/sd/SD_VERSION.lock)"..origin/master --stat   # 查看变更概览（以 lock 为基线）
```

### 5.2 第一步：checkout + **强制**更新 submodule

```bash
cd /opt/sd
git checkout <new-commit>
# ⚠️ 必须加 --force：submodule 常停在旧 commit（状态带 +），
#    不加 --force 会导致 ggml 头文件与 sd.cpp 不匹配（编译报函数未声明）
git submodule update --init --recursive --force
git submodule status          # 确认 ggml 前无 '+'，HEAD 与 git ls-tree HEAD ggml 一致
git apply --check /opt/static_comfyui/cpp/sd/patches/sdcpp-freeu-sag-v2.patch
```

- 如果成功：跳到 5.4
- 如果失败（`error: patch failed`）：进入 5.3

### 5.3 patch 冲突时：AI 手动 rebase

patch 修改了 **3 个文件**，需要逐一检查每个文件在新版中的对应位置：

**对比文件（新版 vs 旧版）：**

| patch 涉及的文件 | 对比命令 | 需要检查什么 |
|-----------------|----------|-------------|
| `include/stable-diffusion.h` | `git diff <old>..<new> -- include/stable-diffusion.h` | `sd_img_gen_params_t` 末尾是否新增字段；FreeU/SAG/DynCFG/FreSca/IPAdapter 是否已被官方合入 |
| `src/model/diffusion/unet.hpp` | `git diff <old>..<new> -- src/model/diffusion/unet.hpp` | `UnetModelBlock::forward()` 签名/基类；`UNetModelRunner` 位置 |
| `src/pipeline/diffusion_engine.h` | `git diff <old>..<new> -- src/pipeline/diffusion_engine.h` | `StableDiffusionGGML` class fields 位置（在 `is_using_edm_v_parameterization` 之后） |
| `src/pipeline/diffusion_engine.cpp` | `git diff <old>..<new> -- src/pipeline/diffusion_engine.cpp` | `run_condition` lambda 的 `diffusion_params.extra` 链末尾；采样循环 `guided.pred` post-compute；include 块 |
| `src/pipeline/image.cpp` | `git diff <old>..<new> -- src/pipeline/image.cpp` | `generate_image` 中 `apply_circular_axes` 之后；`prepare_image_generation_embeds` 中 `ImageGenerationEmbeds embeds;` 之前 |

**不需要对比** `model.hpp`（不再修改）和 `vae.hpp`（tile cap 移到 adapter，不涉及 sd.cpp）。

**经典 rebase 步骤：**
```bash
cd /opt/sd
# 保存当前 sd.cpp 的 diff（含已应用的 patch）
git diff > /tmp/sd_our_changes.diff

# 切到新版本
git fetch origin
git checkout <new-commit>
git submodule update --init --recursive

# 尝试应用 patch
git apply /opt/static_comfyui/cpp/sd/patches/sdcpp-freeu-sag-v2.patch 2>&1

# 如果 offset 失败，用 --reject 看具体冲突
git apply --reject /opt/static_comfyui/cpp/sd/patches/sdcpp-freeu-sag-v2.patch 2>&1
# 检查 *.rej 文件，手动修复后删除
```

**常见冲突处理：**
1. **stable-diffusion.h**: 如果 upstream 在末尾新增了字段，patch 的 4 个结构体/字段拼在它们后面即可。
2. **unet.hpp**: `UnetModelBlock::forward()` 中的 FreeU block 跟在 `control_offset--` 之后、output block 的 concat 之前。
3. **pipeline/diffusion_engine.h**: 类字段插在 `is_using_edm_v_parameterization` 之后。
4. **pipeline/diffusion_engine.cpp**:
   - include：`#include "model/diffusion/unet.hpp"`
   - `run_condition` 的 `dynamic_cast`：插在 `diffusion_params.extra` if/else 链之后、`cached_output` 之前
   - SAG/DynCFG：插在 `if (guided.pred.empty()) return {};` 之后、`denoised = guided.pred * c_out...` 之前
5. **pipeline/image.cpp**:
   - IPAdapter 注入：`prepare_image_generation_embeds` 中 `ImageGenerationEmbeds embeds;` 之前
   - 参数配线：`generate_image` 中 `apply_circular_axes` 之后、`resolve_ref_image_params` 之前

> **重构历史**：`7f410a3` 起生成管线从 `stable-diffusion.cpp` 拆到 `src/pipeline/`；`6dcb5bb` 起上游删除了 `model_loader.cpp` 的 `unused_tensors` 过滤。若上游再次移动这些函数，用 `grep -rn 'run_condition\|prepare_image_generation_embeds\|apply_circular_axes' src/` 重新定位。

**如果官方已经合入了 FreeU/SAG：** 删除 patch 中对应部分，只保留未合入的部分。
**如果官方 API 大变：** 对照 patch 的修改意图，在新代码的对应位置重新实现。

### 5.4 第二步：检查 sd.cpp C API 变化

```bash
git diff "$(cat /opt/static_comfyui/cpp/sd/SD_VERSION.lock)"..<new> -- include/stable-diffusion.h
```

逐项检查以下内容：

**A. `sd_ctx_params_t` 字段变化**
- 新增字段：需要在 `sdcpp_adapter.h` 的 `ModelConfig` 中添加对应字段
- 在 `sdcpp_adapter.cpp` 的 `SDPipeline::load()` 中添加映射代码
- 如果新增字段需要 StaticPy 暴露，更新 `sd_backend.static.py` 的 extern fn

**B. `sd_img_gen_params_t` 字段变化**
- 新增字段：在 `sdcpp_adapter.h` 的 `ImageGenerationParams` 中添加
- 在 `sdcpp_adapter.cpp` 的 `SDPipeline::generate()` 中添加映射

**C. 函数签名变化**
- `new_sd_ctx`、`generate_image`、`free_sd_ctx`、`free_sd_images` 等：检查参数列表是否变化
- `sd_set_log_callback`、`sd_set_progress_callback` 等 callback 类型是否变化

**D. 枚举值变化**
- `sample_method_t`、`scheduler_t`、`prediction_t`、`sd_type_t` 是否新增/删除枚举值
- `sd_hires_upscaler_t` 等是否变化

### 5.5 第三步：检查 adapter C API 是否需要更新

`sd_pipeline_generate_hires` 是 StaticPy 直接调用的主入口。如果有新增功能需要暴露：

1. 在 `sdcpp_adapter.h` 的 C API 区添加新参数
2. 在 `sdcpp_adapter.cpp` 实现中映射到 `ImageGenerationParams`
3. 在 `sd_backend.static.py` 添加对应的 `extern fn` 参数 + wrapper
4. 在 `nodes.static.py` 的对应节点中读取新参数

注意 C API 是**扁平的（所有参数平铺）**——新增参数只能在末尾追加，不能插在中间或删除旧参数（否则 StaticPy extern fn 签名不匹配）。

### 5.6 第四步：编译验证

```bash
# 1. 重新编译 sd.cpp
cd /opt/static_comfyui/cpp/sd
./build_sd_dl.sh

# 2. 重新编译适配层 + ELF
cd /opt/static_comfyui
./build.sh

# 如果编译失败，常见原因：
#   A. ggml 头文件不匹配 → 见 5.2 的 submodule --force
#   B. patch 引入的类型未声明（如 UNetModelRunner）→ 补 #include
#      （例：pipeline/diffusion_engine.cpp 加 #include "model/diffusion/unet.hpp"）
#   C. sdcpp_adapter.cpp 引用的 sd.cpp 类型/函数已改名 → 改适配层
#   D. 枚举值被重命名 → 检查 str_to_* 映射
```

### 5.7 第五步：回归验证

```bash
# 1. 基础出图
LD_LIBRARY_PATH=cpp/sd/build:/opt/sd/build-dl/bin \
  GGML_BACKEND_PATH=/opt/sd/build-dl/bin/libggml-cuda.so \
  ./comfycli-bin test_sdxl.json --output-dir /tmp/regression

# 2. FreeU/SAG/FreSca 必须验证「开/关输出不同」，否则说明参数没生效
#    （固定 seed 跑同一 workflow，仅切换 freeu/sag/fresca，比较 md5）
#    历史教训：nodes.parse_sampler_opts 曾硬编码 freeu=0，导致 KSampler 的 FreeU 静默失效
```

### 5.8 第六步：锁定版本 + 更新 patch 与文档

```bash
# build_sd_dl.sh 会自动更新 lock；手工核对：
cd /opt/sd && git rev-parse --short HEAD > /opt/static_comfyui/cpp/sd/SD_VERSION.lock

# 重新生成 patch（排除 submodule 指针；gligen.hpp / ggml_fourier.h / freq_filter.h 是 untracked，需先 intent-to-add）
cd /opt/sd
git add -N src/model/diffusion/gligen.hpp src/core/ggml_fourier.h src/runtime/freq_filter.h
git diff -- include src > /opt/static_comfyui/cpp/sd/patches/sdcpp-freeu-sag-v2.patch
git reset src/model/diffusion/gligen.hpp src/core/ggml_fourier.h src/runtime/freq_filter.h   # 保持 untracked，避免误提交
# 生成后在干净 worktree 验证：git worktree add /tmp/x 6dcb5bb --detach && cd /tmp/x &&
#   git apply --check /opt/static_comfyui/cpp/sd/patches/sdcpp-freeu-sag-v2.patch
```

> 更新 `design.md` §4（patch 目标文件/位置）与本文档的基准 commit。  
> 当前基准：**`6dcb5bb`**（`SD_VERSION.lock`）。

### 5.9 实战踩坑记录

**`6dcb5bb` 升级（`7f410a3` → `6dcb5bb`，35 commits）实际遇到的问题：**

| # | 问题 | 现象 | 根因 | 修法 |
|---|------|------|------|------|
| 1 | `gligen_path` hunk 冲突 | `include/stable-diffusion.h.rej` | 上游新增 `audio_vae_path`/`audio_encoder_path` 改变了上下文锚点 | 锚点改为 `control_net_path` 后紧跟插入 `gligen_path` |
| 2 | `model_loader.cpp` hunk 失效 | `.rej` | #1984 删除了整个 `unused_tensors[]`/`vision_model.` 过滤 | 该 hunk 作废，从 patch 中删除（§4.6） |
| 3 | `clip_vision_encoded` / `encode_clip_vision_image` 冲突 | `diffusion_engine.h/.cpp.rej` | 上游新增 `get_audio_embedding` / `compute_ip_adapter_tokens` 改变锚点 | 声明/实现插入到 `compute_ip_adapter_tokens` 之前 |
| 4 | `sampler_noise_scale()` 配线缺失 | 编译/行为回归 | `RunnerEndOnExit sample_control_runner_end` 后锚点上下文变化 | 在 `sample_control_runner_end` 与 `apply_denoise_mask` 之间插入 `sampler_noise_scale() = this->noise_scale;` |
| 5 | 适配层 | **零改动**（编译一次通过） | `sd_*_params_init` + 按字段名赋值的抗性；`generate_video` 签名变化未被适配层使用 | — |
| 6 | 回归 | FreeU/SAG 开关 MD5 不同；SDXL 1024×1024 出图 OK | — | `test_sdxl.json` + freeu on/off 双跑验证 |

> **教训**：`git apply --reject` 后务必**读 `.rej` 真实内容再改锚点**——上游在目标函数附近插新行（如 `audio_encoder_path`）会让「按记忆中的旧上下文」编写的 fix 脚本断言失败。`model_loader.cpp` 这类修上游回归的 hunk，升级时先 `git log -S`/看上游 commit 确认是否已被上游自己修掉，避免盲目重放。

**`7f410a3` 升级（74 commits，含生成管线大重构）历史问题（仍值得参考）：**

| # | 问题 | 现象 | 根因 | 修法 |
|---|------|------|------|------|
| 1 | ggml submodule 未更新 | 编译报 `ggml_mul_mat_i8_tensorwise` 未声明 | submodule 停在旧 commit（状态带 `+`） | `git submodule update --init --recursive --force` |
| 2 | patch 冲突 | `stable-diffusion.cpp:239` patch failed | #1956/#1957 把生成管线从 `stable-diffusion.cpp` 拆到 `src/pipeline/` | 5 个 hunk 重定位（见下表） |
| 3 | 缺 include | `UNetModelRunner does not name a type` | 新 `diffusion_engine.cpp` 未包含 `unet.hpp` | 加 `#include "model/diffusion/unet.hpp"` |
| 4 | FreeU 静默失效 | 开关输出 hash 相同 | `nodes.parse_sampler_opts` 硬编码 `freeu=0`/`sag=0` | 改为 `get_int(inputs, "freeu", 0)` |
| 5 | 适配层 | **零改动**（未报错） | `sd_*_params_init` + 按字段名赋值的抗性 | — |
| 6 | 原生 IP-Adapter 的 CLIP Vision 加载为 0 张量 | `vision_model.* not in model metadata` | `#1935` 把 `"vision_model."` 加进 `unused_tensors`，而 `is_unused_tensor` 在加前缀前检查 | 从 `unused_tensors` 移除 `"vision_model."`（**`6dcb5bb` 起上游 #1984 已删除整个过滤，本修复作废**） |
| 7 | 原生 IP-Adapter 前缀不匹配 | 同上 | `#1957` 把 clip vision 加载前缀从 `cond_stage_model.transformer.` 误改为 `clip_vision.` | 改回 `cond_stage_model.transformer.`（**`6dcb5bb` 仍需保留**） |

> **教训**：升级后即使适配层零改动，**原生功能仍可能因上游重构/新特性引入回归**（#1935、#1957）。启用原生功能前务必实机验证，并对照旧版代码确认前缀/过滤逻辑是否被改。定位方法：`git log -S '<可疑字符串>' -- <file>` 找引入点。

**patch hunk 重定位对照（旧 → 新）：**

| 改动 | 旧位置 | 新位置 |
|------|--------|--------|
| 类字段 | `stable-diffusion.cpp` `StableDiffusionGGML` | `pipeline/diffusion_engine.h`（`is_using_edm_v_parameterization` 后） |
| run_condition FreeU | `stable-diffusion.cpp` `run_condition` | `pipeline/diffusion_engine.cpp`（`diffusion_params.extra` if/else 链后） |
| SAG/DynCFG | `stable-diffusion.cpp` 采样循环 | `pipeline/diffusion_engine.cpp`（`guided.pred.empty()` 后） |
| IPAdapter 注入 | `stable-diffusion.cpp` `prepare_image_generation_embeds` | `pipeline/image.cpp` 同名函数 |
| 参数配线 | `stable-diffusion.cpp` `generate_image` | `pipeline/image.cpp` `generate_image`（`apply_circular_axes` 后） |

**经验**：上游重构会**移动函数到新文件**——不要硬按文件名找，用 `grep -rn '<函数名>' src/` 重新定位。`run_condition` / `prepare_image_generation_embeds` / `generate_image` / `apply_circular_axes` 是 patch 的锚点。

---

## 6. 功能支持矩阵

| 功能 | sd.cpp 原生 | 需 patch | C API 已暴露 | StaticPy 已暴露 |
|------|-----------|---------|-------------|----------------|
| SDXL txt2img | ✅ | ❌ | ✅ `sd_pipeline_load` + `sd_pipeline_generate` | ✅ `CheckpointLoaderSimple` + `KSampler` |
| SD1.5 txt2img | ✅ | ❌ | ✅ 同上 | ✅ 同上 |
| FLUX / Z-Image | ✅ | ❌ | ✅ `sd_pipeline_load_ex` | ✅ `DiffusionModelLoader` |
| HiRes Fix | ✅ | ❌ | ✅ `sd_pipeline_generate_hires` | ✅ `HiResFix` |
| LoRA | ✅ | ❌ | ✅ `sd_pipeline_load_lora` | ✅ `LORALoader` |
| VAE Tiling | ✅ (tile cap 在 adapter) | ❌ | ✅ `vae_tiling` / `vae_tile_size` 参数 | ✅ `HiResFix` / `KSampler` |
| FreeU | ❌ | ✅ | ✅ `freeu` / `freeu_b1` / `freeu_b2` 参数 | ✅ `HiResFix` / `KSampler`（仅 UNet） |
| SAG | ❌ | ✅ | ✅ `sag` / `sag_scale` 参数 | ✅ `HiResFix` / `KSampler` |
| FreSca（频域 guidance） | ❌ | ✅ | ✅ `sd_pipeline_set_fresca` setter | ✅ `FreSca` 节点（UNet/DiT 均生效） |
| ADetailer | ✅ | ❌ | ✅ `sd_pipeline_generate_adetailer` | ✅ `ADetailer` 节点 |
| ControlNet | ✅ | ❌ | ✅ `sd_pipeline_load_control_net` + `control_strength` | ✅ `ControlNetLoader` / `ControlNetApply(-Advanced)` |
| IPAdapter | ✅（原生） | ✅（clip vision 前缀回归修复，§4.7） | ✅ `sd_pipeline_set_ipadapter(_enabled)` | ✅ `IPAdapterModelLoader` / `CLIPVisionLoader` / `IPAdapterApply` |
| PhotoMaker | ✅ | ❌ | ❌ | ❌ |
| ESRGAN Upscale | ✅ | ❌ | ❌ | ❌ |

> C API 未暴露 ≠ 不可用：`backup.sh` 走 `img_hires` 二进制可直接用 PhotoMaker / ESRGAN；
> 要在 `comfycli-bin` 里用，需在 `sdcpp_adapter.h` 加 C API → `sd_backend.static.py` 加 `extern fn` → `nodes.static.py` 接线。

---

## 7. 构建配置

### SD_BACKEND_DL=ON（当前默认，动态后端加载）

```
libsdcpp_adapter.so → dlopen → libstable-diffusion.so → libggml.so → dlopen → libggml-cuda.so
```

- `libsdcpp_adapter.so` 只含适配层代码（很小）
- CUDA 是 ggml 插件，通过 `GGML_BACKEND_PATH` 环境变量在运行时加载
- 支持 CPU-only 部署（35MB 部署包，不带 CUDA 后端的 `.so`）

### SD_BACKEND_DL=OFF（旧模式，静态链接）

```
libsdcpp_adapter.so -> links -> libstable-diffusion.a + libggml-cuda.a + CUDA Runtime
```

- 所有东西链接进一个 `.so`
- 不能再 CPU-only 部署
