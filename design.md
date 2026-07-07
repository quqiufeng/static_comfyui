# 项目方案

## 核心思路

**GGML 做底层，封装类似 stable-diffusion.cpp 的 API，用在本项目。然后用单个 C++ 代码基于这个 API 实现 SDXL 出图。**

缺什么代码或执行流程不清楚，用 code search 搜索 stable-diffusion.cpp 定位。C++ helper 层采用此方案。

## 架构

```
GGML (tensor library)
    ↓
cpp/sd/              ← 从 stable-diffusion.cpp 复制的代码（需要什么复制什么）
cpp/ggml_engine/
├── engine.h         ← 我们的 API（类似 stable-diffusion.h）
├── engine.cpp       ← 实现（用 GGML + sd/）
└── sdxl_standalone.cpp ← 单 C++ 文件，基于 engine.h 出图
```

## 工作方式

1. 需要某个功能 → `code search` 查 stable-diffution.cpp 实现
2. 复制具体代码/文件到 `cpp/sd/`（只复制需要的，不是全部）
3. engine.cpp 中使用 sd/ 的代码
4. 编译：GGML + sd/ 源码，不依赖 stable-diffusion.cpp

## 参考实现

```bash
/data/venv/bin/python /opt/static_comfyui/sdxl_pipeline.py
# → /home/quqiufeng/python_reference.png (6秒, 1024×1024)
```

## code search

```bash
/opt/my_db/tools/cache_query "搜索内容" \
  --repo stable-diffusion.cpp --type search \
  --analysis-dir /opt/code_caches/stable-diffusion.cpp_cache

