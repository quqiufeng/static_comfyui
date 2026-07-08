# cpp/sd 兼容层封装方案

> 基于 `/opt/sd`（stable-diffusion.cpp 最新 shallow clone）的 C++ 后端封装设计。
> 目标：对上层代码暴露 PyTorch 风格的 API，同时把 sd.cpp 源码改动压到最小。

---

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| 零补丁优先 | 尽量不改 `/opt/sd` 源码，只通过 `stable-diffusion.h` 公开 API 调用 |
| 最小侵入 | 必须改 sd.cpp 的功能，控制在 3 个以内小 patch（FreeU/SAG/IPAdapter） |
| PyTorch 风格 API | 上层代码用 `at::Tensor`、`nn::Module`、`Module::forward` 等风格编写 |
| 可升级 | 通过 `SD_VERSION.lock` 锁定版本，patch 文件 + 回归测试支撑升级 |
| 无 Python | 最终产物是 C++ 静态库/可执行文件，不依赖 Python 运行时 |

---

## 2. 目录结构

```
/opt/static_comfyui/cpp/sd/
├── design.md                  # 本文档
├── CMakeLists.txt             # 顶层构建配置
├── SD_VERSION.lock            # 锁定 /opt/sd 版本
├── patches/                   # 对 /opt/sd 的必要 patch（尽量少）
│   ├── README.md
│   ├── 001-unet-freeu.patch   # 可选：FreeU 半通道缩放
│   ├── 002-unet-sag.patch     # 可选：Self-Attention Guidance
│   └── 003-unet-ipadapter.patch # 可选：IPAdapter UNet 注入
│
├── src/
│   ├── api/                   # 公开 API：PyTorch 风格接口
│   │   ├── tensor.h/.cpp      # 张量抽象
│   │   ├── module.h/.cpp      # Module / Parameter 抽象
│   │   ├── ops.h/.cpp         # 基础算子（线性、卷积、注意力等）
│   │   └── pipeline.h/.cpp    # 高层 SDXL 推理接口
│   │
│   ├── adapters/              # 唯一接触 /opt/sd 的目录
│   │   └── sdcpp_adapter.h/.cpp   # 封装 stable-diffusion.h C API
│   │
│   ├── native/                # 不依赖 sd.cpp 的原生实现
│   │   ├── postproc/          # 图像后处理（锐化、放大、人脸）
│   │   ├── sampler/           # 未来可能自定义的采样器
│   │   └── attention/         # 未来可能自定义的 attention 实现
│   │
│   └── workflow/              # 可选：ComfyUI 风格的节点执行引擎
│       ├── node.h/.cpp
│       ├── executor.h/.cpp
│       └── nodes/             # 核心节点实现
│
├── tests/                     # 测试
│   ├── unit/                  # 单元测试
│   ├── integration/           # 集成测试
│   └── regression/            # 图像回归测试（固定 seed 对比 hash）
│
└── scripts/
    ├── build.sh               # 一键编译
    ├── upgrade_sdcpp.sh       # 升级 /opt/sd 并重新 patch
    └── run_regression.sh      # 跑回归测试
```

---

## 3. 核心架构

```
┌────────────────────────────────────────────┐
│ 上层代码 / 节点引擎 / 新模型实现              │
│ 使用 sd::Tensor / sd::nn::Module / sd::ops  │
├────────────────────────────────────────────┤
│ libsd_backend.a                              │
│ ├── api/      : 抽象张量、模块、算子接口      │
│ ├── adapters/ : sdcpp_adapter（唯一接触 sd.cpp）│
│ └── native/   : 不依赖 sd.cpp 的扩展实现      │
├────────────────────────────────────────────┤
│ /opt/sd (stable-diffusion.cpp)              │
│ 尽量不改源码，只通过 stable-diffusion.h 调用 │
└────────────────────────────────────────────┘
```

### 3.1 隔离原则

- **只有 `src/adapters/sdcpp_adapter.cpp` 可以 `#include <stable-diffusion.h>`。**
- 上层代码只 `#include "sd/api/xxx.h"`。
- `api/` 层不暴露任何 sd.cpp 内部类型（如 `ggml_tensor`、`sd_ctx_t` 等）。

### 3.2 为什么不是完整 PyTorch 兼容

完整 PyTorch 兼容意味着重新实现 dispatcher、autograd、动态 shape 等，不现实。本方案只提供 **SDXL 推理需要的子集**：

- `Tensor`：形状、dtype、device、view、permute、contiguous
- `nn::Module`：参数注册、子模块注册、`forward` 虚函数
- `ops`：`add`, `mul`, `linear`, `conv2d`, `group_norm`, `silu`, `attention`
- `pipeline`：`load_model`, `encode_prompt`, `sample`, `decode_vae`, `generate_image`

---

## 4. API 层设计（示例）

### 4.1 张量

```cpp
// src/api/tensor.h
namespace sd {

enum class Device { CPU, CUDA };
enum class Dtype { F32, F16 };

class Tensor {
public:
    Tensor() = default;
    Tensor(std::vector<int64_t> shape, Dtype dtype, Device device);
    
    std::vector<int64_t> sizes() const;
    Dtype dtype() const;
    Device device() const;
    
    Tensor view(std::vector<int64_t> shape) const;
    Tensor permute(std::vector<int64_t> dims) const;
    Tensor to(Device device) const;
    
    // 内部实现由后端决定，上层不直接访问
    class Impl;
    std::shared_ptr<Impl> impl() const;
};

} // namespace sd
```

### 4.2 模块

```cpp
// src/api/module.h
namespace sd::nn {

class Module {
public:
    virtual ~Module() = default;
    virtual Tensor forward(Tensor input) = 0;
    
    void register_parameter(const std::string& name, Tensor param);
    void register_module(const std::string& name, std::shared_ptr<Module> module);
    
    std::vector<std::pair<std::string, Tensor>> named_parameters() const;
    void to(Device device);
};

} // namespace sd::nn
```

### 4.3 高层 Pipeline

```cpp
// src/api/pipeline.h
namespace sd {

struct ModelConfig {
    std::string diffusion_model_path;
    std::string vae_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string llm_path;
    int n_threads = 4;
    bool keep_vae_on_cpu = false;
    bool keep_clip_on_cpu = false;
};

struct ImageGenerationParams {
    std::string prompt;
    std::string negative_prompt;
    int width = 512;
    int height = 512;
    int steps = 20;
    float cfg_scale = 7.5f;
    int64_t seed = -1;
};

class SDPipeline {
public:
    bool load(const ModelConfig& config);
    std::vector<uint8_t> generate(const ImageGenerationParams& params);
    
private:
    std::unique_ptr<class SDPipelineImpl> impl_;
};

} // namespace sd
```

---

## 5. 功能拆分：什么在 sd.cpp 里，什么移出来

| 功能 | 位置 | 说明 |
|------|------|------|
| 模型加载 | sd.cpp | 通过 `new_sd_ctx` / `sd_ctx_params_t` |
| 文本编码 | sd.cpp | 通过 `get_learned_condition`（封装后） |
| UNet 采样 | sd.cpp | 核心推理，不改动 |
| VAE 编解码 | sd.cpp | 通过 `decode_first_stage` / `encode_first_stage` |
| LoRA | sd.cpp | 原生支持 |
| ControlNet | sd.cpp | 原生支持 |
| HiRes Fix | sd.cpp | 原生支持 |
| VAE tiling | sd.cpp | 通过参数控制，不改代码 |
| FreeU | sd.cpp patch | 必须在 UNet 内部 |
| SAG | sd.cpp patch | 必须在 UNet 内部 |
| IPAdapter UNet | sd.cpp patch | 必须在 CrossAttention 内部 |
| 图像锐化/清晰度 | native/postproc | 像素后处理 |
| 人脸修复/替换 | native/postproc | ONNX / OpenCV |
| ESRGAN 放大 | native/postproc | 外部 upscale |
| 自定义采样器 | native/sampler | 未来扩展 |
| 自定义 attention | native/attention | 未来扩展 |

---

## 6. Patch 策略

### 6.1 零补丁优先

默认情况下 `/opt/sd` 不改源码。当前为支持 FreeU/SAG，已应用最小 patch：`patches/sdcpp-freeu-sag-v2.patch`。

### 6.2 必要 Patch 的最小化

如果必须加 patch，满足以下条件：

1. **每个 patch 只做一件事**（当前 FreeU/SAG patch 合并为一组，因为二者都依赖 `sd_img_gen_params_t` 的新增字段，拆分反而增加维护成本）。
2. **每个 patch 配一个 README 说明**：见 `README.md` 13.6 节。
3. **patch 尽量小**：当前 patch 约 220 行，分布在 5 个文件。
4. **升级时自动重放**：`scripts/upgrade_sdcpp.sh` 自动 `git apply patches/*.patch`。

### 6.3 当前 Patch 清单

| Patch | 文件 | 大约行数 | 说明 |
|-------|------|---------|------|
| `patches/sdcpp-freeu-sag-v2.patch` | `include/stable-diffusion.h`<br>`src/model/diffusion/model.hpp`<br>`src/model/diffusion/unet.hpp`<br>`src/stable-diffusion.cpp`<br>`src/model/vae/vae.hpp` | ~220 | 新增 `sd_freeu_params_t`、`sd_sag_params_t`、`sd_dynamic_cfg_params_t`；在 UNet 输出块加入 FreeU 缩放；在采样循环中加入 SAG 与 Dynamic CFG；VAE decode tiling 输出 tile 上限保护 |

**Phase 1 已完成**：基础 SDXL txt2img + HiRes + VAE tiling + LoRA + FreeU + SAG 已跑通。

---

## 7. 升级流程

### 7.1 版本锁定

```bash
# SD_VERSION.lock
bb84971
```

### 7.2 升级脚本

```bash
#!/bin/bash
# scripts/upgrade_sdcpp.sh
set -euo pipefail

TARGET=$(cat /opt/static_comfyui/cpp/sd/SD_VERSION.lock)

cd /opt/sd
git fetch origin
git checkout "$TARGET"
git submodule update --init --recursive

# 重放必要 patch
for p in /opt/static_comfyui/cpp/sd/patches/*.patch; do
    git apply "$p" || { echo "patch failed: $p"; exit 1; }
done

# 编译 sd.cpp
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DGGML_CUDA=ON
make -j$(nproc) stable-diffusion

# 编译 cpp/sd
cd /opt/static_comfyui/cpp/sd
rm -rf build && mkdir build && cd build
cmake ..
make -j$(nproc)

# 回归测试
../scripts/run_regression.sh
```

### 7.3 升级检查清单

- [ ] `git diff HEAD..origin/master -- include/stable-diffusion.h` 查看 API 变化
- [ ] 检查 `sd_ctx_params_t` / `sd_img_gen_params_t` 新增字段
- [ ] 检查函数签名变更（如 `new_upscaler_ctx`）
- [ ] 检查 patch 是否冲突
- [ ] 跑编译测试 + 图像回归测试

---

## 8. 测试策略

| 测试 | 内容 | 作用 |
|------|------|------|
| 编译测试 | `build.sh` 全量通过 | 发现 API 签名变更 |
| 单元测试 | adapter 映射、tensor 操作 | 验证接口正确 |
| 集成测试 | 加载模型、encode、sample、decode | 验证 pipeline 连通 |
| 图像回归 | 固定 seed/prompt，对比输出 PNG hash | 发现 sd.cpp 内部行为变化 |
| 显存回归 | 监控 VAE decode 峰值 | 发现 OOM 回退 |

### 8.1 图像回归示例

```bash
# tests/regression/generate_baseline.sh
./build/sd_cli \
  --diffusion-model /data/models/image/z_image_turbo-Q5_K_M.gguf \
  --vae /data/models/image/ae.safetensors \
  -p "a red apple" -s 42 -W 512 -H 512 --steps 5 \
  -o /tmp/test_baseline.png

# 对比 hash
sha256sum /tmp/test_baseline.png > tests/regression/baseline.sha256
```

---

## 9. 构建配置

### 9.1 /opt/sd 静态库

```cmake
set(SD_DIR /opt/sd)
set(SD_BUILD_DIR ${SD_DIR}/build)

find_library(SD_LIB stable-diffusion PATHS ${SD_BUILD_DIR})
find_library(GGML_LIB ggml PATHS ${SD_BUILD_DIR}/ggml/src)
find_library(GGML_CPU_LIB ggml-cpu PATHS ${SD_BUILD_DIR}/ggml/src)
find_library(GGML_BASE_LIB ggml-base PATHS ${SD_BUILD_DIR}/ggml/src)
find_library(GGML_CUDA_LIB ggml-cuda PATHS ${SD_BUILD_DIR}/ggml/src/ggml-cuda)
```

### 9.2 链接顺序

```cmake
target_link_libraries(sd_backend
    ${SD_LIB}
    "-Wl,--whole-archive" ${GGML_CUDA_LIB} "-Wl,--no-whole-archive"
    ${GGML_CPU_LIB}
    ${GGML_BASE_LIB}
    ${GGML_LIB}
    CUDA::cudart
)
```

---

## 10. 开发阶段

### Phase 1：最小 patch 基础版（已完成）

- [x] 实现 `sdcpp_adapter`，封装 `stable-diffusion.h`
- [x] 实现 `SDPipeline::generate` 基础 SDXL txt2img
- [x] 扩展 `SDPipeline` 支持 LoRA、VAE tiling、HiRes Fix
- [x] 按需加入 FreeU / SAG patch（`patches/sdcpp-freeu-sag-v2.patch`）
- [x] 端到端验证（1024×1024 txt2img、1280×720 HiRes）
- [ ] 实现 `sd::Tensor` / `sd::nn::Module` 抽象（暂缓，保持最小封装）
- [ ] 写 5 个基础回归测试（待补）

### Phase 2：Workflow 引擎

- [ ] 实现节点基类 `Node`
- [ ] 实现 DAG executor
- [ ] 实现核心节点：LoadCheckpoint, CLIPTextEncode, EmptyLatentImage, KSampler, VAEDecode, SaveImage
- [ ] 能解析 ComfyUI 导出的 JSON workflow

### Phase 3：可选扩展（按需）

- [x] FreeU / SAG patch（已完成）
- [ ] IPAdapter patch 或 native 实现
- [ ] img2img / ControlNet / 图像后处理（锐化、放大）
- [ ] 扩展模块体系，允许替换 sd.cpp 子模块

---

## 11. 关键决策

1. **是否允许改 sd.cpp？**
   - 默认：不允许。
   - 例外：FreeU/SAG/IPAdapter 这类必须进 UNet 的功能，以最小 patch 形式保留。

2. **是否支持所有 ComfyUI 节点？**
   - 不支持 200+ 节点。
   - 先支持 SDXL txt2img 核心节点（~10 个），后续按需扩展。

3. **是否支持 ComfyUI 原生 Python 自定义节点？**
   - 不支持。本方案是无 Python 的 native 方案。

4. **未来是否替换 sd.cpp？**
   - 不排除。适配层设计允许未来用 native 实现替换 sd.cpp 的某些模块，上层代码不变。

---

## 12. 参考

- `/opt/my-img/SD_GUIDE.md`：sd.cpp 升级实战经验
- `/opt/stable-diffusion.cpp/include/stable-diffusion.h`：sd.cpp 公开 C API
- `/opt/static_comfyui/AGENTS.md`：StaticPy 项目背景
