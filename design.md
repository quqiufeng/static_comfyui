# 项目方案

## 核心思路

**StaticPy 做编排前端，1:1 复刻 ComfyUI 的节点与 DAG 调度；stable-diffusion.cpp 做推理后端，封装 C API 供 StaticPy FFI 调用；最终编译成独立 ELF 二进制。**

缺什么代码或执行流程不清楚，用 code search 搜索 ComfyUI 源码定位，再用 code search 搜索 stable-diffusion.cpp 定位对应 C API 用法。

## 架构

```
workflow.json / --prompt
       │
       ▼
┌──────────────────────────────┐
│  comfycli/                    │
│  StaticPy 编排层                │
│  ├── nodes.static.py          │  ComfyUI 节点定义
│  ├── execution.static.py     │  DAG 拓扑排序 + 链接解析
│  ├── cli_args.static.py       │  CLI 参数
│  ├── main.static.py           │  入口
│  └── sd_backend.static.py     │  sd.cpp C API FFI 声明
└──────────┬───────────────────┘
           │ extern fn FFI
           ▼
┌──────────────────────────────┐
│  cpp/sd/                      │
│  stable-diffusion.cpp 适配器  │
│  ├── src/adapters/sdcpp_adapter.h   │  C API 头文件
│  ├── src/adapters/sdcpp_adapter.cpp │  C API 实现（封装 sd.cpp）
│  └── build/libsdcpp_adapter.so     │  推理后端共享库
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  /opt/sd                      │
│  stable-diffusion.cpp 源码    │
│  （编译为静态库，链接到适配器）│
└──────────┬───────────────────┘
           ▼
        GGML / CUDA
```

## 工作方式

1. 需要某个 ComfyUI 节点 → `code search` 查 ComfyUI 源码
2. 将节点逻辑翻译为 StaticPy 的 `register_node` + 参数解析
3. 需要底层推理能力 → `code search` 查 stable-diffusion.cpp 对应 C API
4. 在 `sd_backend.static.py` 声明 `extern fn`，在 `sdcpp_adapter.cpp` 实现 C API 封装
5. 节点函数通过 FFI 调用 C API，完成出图

## 职责边界

| 层 | 负责 | 不负责 |
|---|------|--------|
| StaticPy 编排层 | 节点 DAG 调度、输入链接解析、参数类型转换、CLI | 张量计算、模型加载、采样算法 |
| stable-diffusion.cpp | 模型加载、UNet/VAE/CLIP/采样器/LoRA/ControlNet | 节点语义、工作流编排 |
| 适配器 (`sdcpp_adapter`) | 把 sd.cpp 的 C++ 类封装为 C API，让 StaticPy 可以 FFI 调用 | 新增推理算法 |

## 参考实现

```bash
# 本地 workflow 模式
LD_LIBRARY_PATH=cpp/sd/build:/data/venv/lib/python3.12/site-packages/torch/lib \
  ./comfycli-bin /tmp/test_workflow.json --output-dir /tmp/comfy_output
# → /tmp/comfy_output/test_cat.png (1024×1024)

# 本地 prompt 模式
LD_LIBRARY_PATH=cpp/sd/build:/data/venv/lib/python3.12/site-packages/torch/lib \
  ./comfycli-bin --checkpoint /data/models/image/sd_xl_base_1.0.safetensors \
  --prompt "a photo of a cat" --output /tmp/comfy_output/prompt_cat.png
# → /tmp/comfy_output/prompt_cat.png (1024×1024)
```

## code search

```bash
# 搜索 ComfyUI 节点实现
/opt/my_db/tools/cache_query "KSampler node" \
  --repo /code/comfyui --type search \
  --analysis-dir /opt/code_caches/comfyui_cache

# 搜索 stable-diffusion.cpp C API 用法
/opt/my_db/tools/cache_query "generate_image" \
  --repo /opt/sd --type search \
  --analysis-dir /opt/code_caches/stable-diffusion.cpp_cache
```
