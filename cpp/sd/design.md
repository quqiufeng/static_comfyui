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
│   └── sdcpp-freeu-sag-v2.patch    # 唯一 patch：FreeU + SAG + DynCFG（3 文件 159 行）
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
└── scripts/
    └── build.sh                    # 编译适配层（被根目录 build.sh 调用）
```

其他目录（`api/`, `native/sampler/`, `native/attention/`, `tests/`, `scripts/upgrade_sdcpp.sh` 等）是蓝图，尚未创建。

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

所有改动集中在 **一个 patch 文件** `patches/sdcpp-freeu-sag-v2.patch`（约 240 行），修改 sd.cpp 的 **3 个文件**：

### 4.1 `include/stable-diffusion.h`
新增 3 个结构体 + 在 `sd_img_gen_params_t` 末尾追加对应字段：
- `sd_freeu_params_t`：`{enabled, b1, b2, s1, s2}`
- `sd_sag_params_t`：`{enabled, scale}`
- `sd_dynamic_cfg_params_t`：`{enabled, percentile, mimic_scale, threshold_percentile}`

### 4.2 `src/model/diffusion/unet.hpp`
两处独立改动：

**A. `UnetModelBlock` 类**（FreeU 核心计算）
- 新增 5 个字段 + `set_freeu()` 方法
- 在 `forward()` 的 output block 中加入 FreeU 通道缩放逻辑：
  - 匹配 ComfyUI `nodes_freelunch.py`：backbone 前半通道 × `b`，skip connection × `s`
  - `channel == model_channels*4` → 用 `{b1,s1}`；`channel == model_channels*2` → 用 `{b2,s2}`
  - **关键设计**：`UnetModelBlock` 只存储 FreeU 参数，不关心来源

**B. `UNetModelRunner` 类**（FreeU 参数传递层）
- 新增 5 个字段 + `set_freeu_params()` 方法
- `compute(DiffusionParams&)` 中用 `this->freeu_*` 调用 `unet.set_freeu()`
- **不修改 `DiffusionParams`**——FreeU 字段直接挂在 runner 上

### 4.3 `src/stable-diffusion.cpp`
三处独立改动：

**A. `StableDiffusionGGML` 类**——新增类字段（各功能默认 disabled）：
- FreeU: `freeu_enabled`, `freeu_b1/b2/s1/s2`
- SAG: `sag_enabled`, `sag_scale`
- Dynamic CFG: `dynamic_cfg_enabled`, `dynamic_cfg_percentile/mimic_scale/threshold_percentile`

**B. `run_condition` lambda 中**——通过 `dynamic_cast<UNetModelRunner*>` + `sd_version_is_unet()` 设置 FreeU：
```cpp
if (sd_version_is_unet(version)) {
    auto* unet_runner = dynamic_cast<UNetModelRunner*>(work_diffusion_model.get());
    if (unet_runner) {
        unet_runner->set_freeu_params(this->freeu_enabled, ...);
    }
}
```

**C. 采样循环的 post-compute 阶段**——插入 SAG + Dynamic CFG：
- SAG：`guided.pred = pred * scale + uncond * (1-scale)`（100% 擦除的纯数学插值）
- Dynamic CFG：找到 `pred` 最大绝对值，若 >1 则全张量除以该值

**D. `generate_image()` 中**——从 `sd_img_gen_params_t` 读取参数存入类字段

### 4.4 `src/model/vae/vae.hpp` — 不移除 ⚠️ 未修改
VAE tile 大小上限已从 patch 中移除，改由 adapter 层在调用 `generate_image` 前自行 cap。详见 §4.7。

### 4.5 `src/model/diffusion/model.hpp` ⚠️ 未修改
FreeU 参数不再经过 `DiffusionParams`。`UNetModelRunner` 通过自己的 `set_freeu_params()` 直接接收。

### 4.6 架构设计原则：参数流向
```
StaticPy → adapter (sdcpp_adapter.cpp)
  → sd_img_gen_params_t.freeu  (C API struct)
  → StableDiffusionGGML::freeu_enabled  (class fields)
  → UNetModelRunner::freeu_enabled  (via dynamic_cast + set_freeu_params)
  → UnetModelBlock::freeu_enabled  (via unet.set_freeu)
  → ggml 计算图
```
FreeU 参数从右上到左下垂直传递，**不污染**水平方向的现有数据结构（如 `DiffusionParams`）。

### 4.7 不需要改 sd.cpp 的功能（通过原生 C API 调用）
LoRA、ControlNet、HiRes Fix、**VAE tiling（含 tile cap）**、sampler/scheduler 枚举、PhotoMaker、ESRGAN upscale、TAESD——全部通过 `sd_ctx_params_t` / `sd_img_gen_params_t` 的标准字段控制，不需要 patch。

其中 VAE tile cap（防止 OOM）在 `sdcpp_adapter.cpp` 的 `SDPipeline::generate()` 中实现：tile 尺寸上限 128 个 latent 像素（对应 scale=8 的 VAE 输出 1024px）。

---

## 5. 升级 sd.cpp 版本：AI 检查清单

升级目标是升级 `/opt/sd` 到新的 commit，保持本项目能编译并通过验证。

### 5.1 准备

```bash
cd /opt/sd
git fetch origin
git log --oneline origin/master -30     # 查看最近的提交
git diff bb84971..origin/master --stat   # 查看变更概览
```

### 5.2 第一步：检查 patch 能否干净应用

```bash
cd /opt/sd
git checkout <new-commit>
git submodule update --init --recursive
git apply --check /opt/static_comfyui/cpp/sd/patches/sdcpp-freeu-sag-v2.patch
```

- 如果成功：跳到 5.4
- 如果失败（`error: patch failed`）：进入 5.3

### 5.3 patch 冲突时：AI 手动 rebase

patch 修改了 **3 个文件**，需要逐一检查每个文件在新版中的对应位置：

**对比文件（新版 vs 旧版）：**

| patch 涉及的文件 | 对比命令 | 需要检查什么 |
|-----------------|----------|-------------|
| `include/stable-diffusion.h` | `git diff <old>..<new> -- include/stable-diffusion.h` | `sd_img_gen_params_t` 末尾是否新增了字段（拼在 `hires` 后面）；FreeU/SAG/DynCFG 是否已被官方合入 |
| `src/model/diffusion/unet.hpp` | `git diff <old>..<new> -- src/model/diffusion/unet.hpp` | `UnetModelBlock::forward()` signature/base class 是否变化；`UNetModelRunner` 起始位置是否偏移 |
| `src/stable-diffusion.cpp` | `git diff <old>..<new> -- src/stable-diffusion.cpp` | `StableDiffusionGGML` class fields 位置；`run_condition` lambda 位置；`generate_image()` 内部逻辑 |

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
1. **stable-diffusion.h**: 如果 upstream 在 `hires` 后新增了字段，patch 的 3 个字段拼在它们后面即可。用 `git diff` 确认 `sd_img_gen_params_t` 末尾内容，调整 offset（或直接在末尾追加）。
2. **unet.hpp**: `UnetModelBlock::forward()` 中的 FreeU block 需要跟在 `control_offset--` 之后、`ggml_concat(h, h_skip)` 之前。找到这个位置重新插入即可。
3. **stable-diffusion.cpp**: 三个改动独立互不影响：
   - Class fields：插在 `is_using_edm_v_parameterization` 后面
   - `run_condition` 中的 `dynamic_cast`：插在 `diffusion_params.extra` 赋值之后、`cached_output` 之前
   - SAG/DynCFG：插在 `if (guided.pred.empty()) return {};` 之后、`denoised = guided.pred * c_out...` 之前
   - `generate_image` 配线：插在 `apply_circular_axes` 之后、`resolve_ref_image_params` 之前

**如果官方已经合入了 FreeU/SAG：** 删除 patch 中对应部分，只保留未合入的部分。
**如果官方 API 大变：** 对照 patch 的修改意图，在新代码的对应位置重新实现。

### 5.4 第二步：检查 sd.cpp C API 变化

```bash
git diff bb84971..<new> -- include/stable-diffusion.h
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

# 如果编译失败，检查：
#   - sdcpp_adapter.cpp 中引用的 sd.cpp 类型/函数是否已改名
#   - 枚举值是否已被官方重命名
```

### 5.7 第五步：回归验证

```bash
# 在本地（有 GPU）运行已知 workflow
LD_LIBRARY_PATH=cpp/sd/build:/opt/sd/build-dl/bin \
  GGML_BACKEND_PATH=/opt/sd/build-dl/bin/libggml-cuda.so \
  ./comfycli-bin test_remote_2560.json --output-dir /tmp/regression

# 对比输出 PNG 的像素 hash 和旧版本是否一致（固定 seed 下应一致）
sha256sum /tmp/regression/*.png
```

### 5.8 第六步：锁定版本

```bash
cd /opt/sd
git rev-parse --short HEAD > /opt/static_comfyui/cpp/sd/SD_VERSION.lock
```

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
| FreeU | ❌ | ✅ | ✅ `freeu` / `freeu_b1` / `freeu_b2` 参数 | ✅ `HiResFix` / `KSampler` |
| SAG | ❌ | ✅ | ✅ `sag` / `sag_scale` 参数 | ✅ `HiResFix` / `KSampler` |
| ADetailer | ✅ | ❌ | ✅ `sd_pipeline_generate_adetailer` | ✅ `ADetailer` 节点 |
| ControlNet | ✅ | ❌ | ❌ | ❌ |
| PhotoMaker | ✅ | ❌ | ❌ | ❌ |
| ESRGAN Upscale | ✅ | ❌ | ❌ | ❌ |
| IPAdapter | ❌ | ❌ | ❌ | ❌ |

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
