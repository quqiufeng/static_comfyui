# ComfyUI StaticPy 重写方案

## 架构

```
GGML (tensor library, cloned to ggml/)
    ↓
cpp/sd/  (stable-diffusion.cpp 源码副本)
    ↓
cpp/ggml_engine/engine.h   (我们的 API，类似 stable-diffusion.h)
cpp/ggml_engine/engine.cpp (实现，调用 sd/ 的类)
    ↓
cpp/ggml_engine/sdxl_standalone.cpp (单文件 CLI，用 engine.h)
```

### 关键设计

- **GGML**: 底层张量库（`/opt/static_comfyui/ggml/`）
- **cpp/sd/**: stable-diffusion.cpp 源码副本（只复制需要的文件）
- **engine.h**: 我们的 C API（`sdxl_new`, `sdxl_unet_forward`, `sdxl_txt2img`）
- **engine.cpp**: 调用 sd/ 下的 stable-diffusion.cpp 类实现 API
- **sdxl_standalone.cpp**: 单 C++ 文件，`#include "engine.h"`，出图

### 与 stable-diffusion.cpp 的关系

| 组件 | stable-diffusion.cpp | 我们的项目 |
|------|---------------------|-----------|
| 张量库 | `ggml/` | `ggml/`（cloned） |
| 模型加载 | `model.cpp`, `safetensors_io.cpp` | `cpp/sd/model.cpp`（copy） |
| UNet | `unet.hpp` | `cpp/sd/unet.hpp`（copy） |
| CLIP | `clip.hpp` | `cpp/sd/clip.hpp`（copy） |
| VAE | `auto_encoder_kl.hpp` | `cpp/sd/auto_encoder_kl.hpp`（copy） |
| Denoiser | `denoiser.hpp` | `cpp/sd/denoiser.hpp`（copy） |
| API | `stable-diffusion.h` | `cpp/ggml_engine/engine.h` |
| CLI | `main.cpp` | `cpp/ggml_engine/sdxl_standalone.cpp` |

### 编译（静态链接 GGML + sd/ 源码）

```bash
cd cpp/ggml_engine
SD=/opt/static_comfyui/ggml    # GGML 库
CUDALIB=/data/cuda/targets/x86_64-linux/lib
g++ -O1 -std=gnu++17 -DGGML_MAX_NAME=128 \
    engine.cpp sdxl_standalone.cpp \
    sd/model.cpp sd/safetensors_io.cpp sd/name_conversion.cpp \
    sd/util.cpp sd/version.cpp sd/ggml_extend_backend.cpp \
    sd/ggml_graph_cut.cpp sd/guidance.cpp \
    sd/tokenizers/clip_tokenizer.cpp sd/tokenizers/bpe_tokenizer.cpp \
    sd/tokenizers/tokenizer.cpp sd/tokenizers/tokenize_util.cpp \
    -I. -Isd -Isd/model_io -Isd/tokenizers -I$SD/include \
    $SD/build/src/libggml.a $SD/build/src/libggml-base.a $SD/build/src/libggml-cpu.a \
    -Wl,--whole-archive $SD/build/src/ggml-cuda/libggml-cuda.a -Wl,--no-whole-archive \
    -ldl -lpthread -lm -lgomp \
    $CUDALIB/libcudart.so $CUDALIB/libcublas.so.12.6.4.1 \
    $CUDALIB/libcublasLt.so.12.6.4.1 $CUDALIB/stubs/libcuda.so \
    $CUDALIB/libculibos.a \
    -o sdxl_ggml
```

**注意**: GGML 必须用 `cmake -DGGML_CUDA=ON` 编译。

### 运行

```bash
LD_LIBRARY_PATH=/data/cuda/targets/x86_64-linux/lib ./sdxl_ggml -p "prompt" --steps 20
```

## code search 使用

需要查阅 stable-diffusion.cpp 的实现时：

```bash
# 语义搜索（推荐）
/opt/my_db/tools/cache_query "UNet forward resblock group_norm conv2d" \
  --repo stable-diffusion.cpp --type search \
  --analysis-dir /opt/code_caches/stable-diffusion.cpp_cache

上下文查询（函数调用关系）
/opt/my_db/tools/cache_query "CompVisDenoiser::get_scalings" \
  --repo stable-diffusion.cpp --type context --depth 1
```

## 模型文件位置

| 模型 | 路径 | 大小 |
|------|------|------|
| SDXL 1.0 base | `/data/models/image/sd_xl_base_1.0.safetensors` | 6.5G |
| CLIP-G | `/data/models/image/clip_g.safetensors` | 2.6G |
| CLIP-L | `/data/models/image/clip_l.safetensors` | 1.6G |

## 参考实现

```bash
/data/venv/bin/python /opt/static_comfyui/sdxl_pipeline.py
# 输出: /home/quqiufeng/python_reference.png (1024×1024, ~6秒)
```

C++ 引擎的输出应与 Python 参考一致。
