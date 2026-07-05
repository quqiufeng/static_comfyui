# Static ComfyUI — AI 开发指引

## 项目目标
用 StaticPy（Python 子集编译器）+ `libtorch_std_helper.so`（C++ libtorch 封装）
1:1 重写 ComfyUI，编译为独立 ELF 二进制，零 pip 依赖。

## 理解 StaticPy 语言

三文件对照阅读（无需单独文档）：

| 文件 | 读什么 |
|------|--------|
| `staticpy/static_translate.py` | 支持/不支持哪些 Python 语法，`import torch` 如何映射为 `torch-*` |
| `staticpy/static_prelude.scm` | int/float/bool 怎么编译为 fixnum/flonum，文件 I/O/dict/JSON 等内置函数 |
| `staticpy/static_stdlib.scm` | `extern fn` 如何映射到 C++，`torch.add` 等 80+ 函数签名 |

快速理解：
- **值类型**：int→fixnum（机器整数）、float→flonum（64位浮点）、bool→boolean
- **列表**：`list[int]` → Scheme vector，非 Python list
- **字典**：`make_dict()` / `dict_get()` → Scheme hashtable
- **异常**：不支持 try/except，用 `if` + 返回值检查替代
- **类**：不支持继承，用 `@dataclass` + 组合替代
- **FFI**：`torch.zeros([1,4,64,64])` → `(torch-zeros #(1 4 64 64))` → C++ `torch::zeros`

## 编译与部署

```bash
# 编译 ELF + C++ .so（本地开发）
./build.sh

# 产物
# ./comfycli          — 独立 ELF 二进制
# ./comfycli.so       — Chez AOT 编译产物
# ./cpp/libtorch_std_helper.so  — C++ 推理后端

# 部署打包（依赖 .so + GLIBC 兼容层）
GLIBC_TARGET=2.35 ./deploy.sh

# 打包 + SCP 到远程
GLIBC_TARGET=2.35 ./deploy.sh --scp user@remote_host

# 远程运行（零 pip）
ssh user@remote_host "bash /opt/comfycli/run.sh workflow.json --output-dir ./output"
```

注意：`build.sh` 只编译不部署，`deploy.sh` 负责打包 + 传输，职责分离。

## 了解最近开发日志

进入项目后先运行 `git log --oneline -20` 查看近期提交记录，
了解最新进展和正在开发的功能。提交信息使用中文书写。

## 使用 code search 辅助翻译

**原则**：翻译每个模块前，先语义搜索 ComfyUI 对应源码，理解实现细节后再 1:1 复刻。

code search 工具位于 `/opt/my_db/tools/`，ComfyUI 索引在 `/opt/code_caches/comfyui_cache/`。

```bash
# 语义搜索：用自然语言找对应实现（推荐）
/opt/my_db/tools/cache_query "model_base BaseModel load_model_weights" \
  --repo /code/comfyui \
  --type search \
  --analysis-dir /opt/code_caches/comfyui_cache

# 查看函数上下文（调用者/被调用者）
/opt/my_db/tools/cache_query "load_checkpoint_guess_config" \
  --repo /code/comfyui \
  --type context --depth 2

# 查看类定义
/opt/my_db/tools/cache_query "class BaseModel" \
  --repo /code/comfyui \
  --type search \
  --analysis-dir /opt/code_caches/comfyui_cache

# 精确符号查找
/opt/my_db/tools/cache_query "/code/comfyui/symbols/load_checkpoint_guess_config" \
  --type exact
```

> **注意**：`cache_query --type search` 必须加 `--analysis-dir` 参数；`--type context` 无需 `--analysis-dir` 但需要 KV 缓存已导入。`/opt/my_db/tools/vector_search` 是底层 C 搜索引擎，但需要独立 embedder 配置，推荐直接用 `cache_query`。

搜索到对应源码后，按 `a.py → a.static.py` 模式 1:1 翻译。

## CLI 接口（main.static.py）
```
# 模式 1：直接命令行出图
comfycli --checkpoint /data/models/image/sd_xl_base_1.0.safetensors --prompt "cat" --output ./out.png

# 模式 2：执行 ComfyUI workflow JSON
comfycli workflow.json --output-dir ./output
```

## 模型文件位置
所有模型在 `/data/models/image/`：
- `sd_xl_base_1.0.safetensors` (6.5G), `clip_g.safetensors` (2.6G), `clip_l.safetensors` (1.6G), `ae.safetensors` (320M)
- `libtorch_std_helper.so` 已内置 `torch_std_safetensors_load` + `torch_std_sdxl_unet_forward`

## 已搭建的基础设施
- `staticpy/` — StaticPy 编译器 (`static_translate.py`)、运行时 (`prelude`/`stdlib`)、构建脚本
- `cpp/` — C++ 推理后端 (libtorch_std_helper.h/.cpp + build_torch_std_helper.sh)
- `build.sh` — 编译 ELF + C++ .so
- `deploy.sh` — 打包依赖 .so + GLIBC 兼容层 + SCP
- `design.md` — 架构设计、文件映射、C++ 依赖清单
- code search: ComfyUI 全量索引 (656 文件, 20,141 chunks, 2,633 函数)

## CUDA 显存管理（已就绪）
4 个 extern fn 已加到 `libtorch_std_helper` + `static_stdlib.scm` + `static_translate.py`：
- `torch.cuda_get_free_memory() → int`
- `torch.cuda_load_model(device, tensor) → tensor`
- `torch.cuda_unload_model(tensor)`
- `torch.cuda_soft_empty_cache()`

## StaticPy 关键约束
- 无类继承/多态 → dataclass + 组合
- 无 try/except → if 守卫 + Result 类型
- 无 lambda 闭包 / eval / exec
- `import torch` 自动映射为 `torch-*` Scheme 函数
- `extern fn foo(x: int, y: float) -> int from "torch_std"` 声明 C FFI

## 开发顺序（bottom-up）

### Phase 0: 基础设施（已完成）
- [x] folder_paths.static.py    路径管理（294 行，编译通过）
- [x] cli_args.static.py        CLI 参数解析（95 行，编译通过）
- [x] comfy_types.static.py     Node 类型定义（70+ 常量，编译通过）

### Phase 1: 模型检测（已完成）
- [x] supported_models_base.static.py  模型基类（BAC + matches 匹配）
- [x] supported_models.static.py       模型注册表（SDXL/SD1.5/SD3/Flux 等）
- [x] model_detection.static.py        state_dict → 架构识别
- [x] model_sampling.static.py         sigma 调度（采样类型枚举 + 映射）
- [x] latent_formats.static.py         潜空间缩放（SD15/SDXL/Flux/Flux2 等）

### Phase 2: 模型加载 + 显存管理（开发中）
- [ ] sd.static.py             safetensors 加载 → 模型对象
- [ ] model_base.static.py     模型基类
- [ ] model_management.static.py  GPU 调调度
- [ ] lora.static.py           LoRA 合并

### Phase 3: 组件级推理
- [ ] clip_model.static.py     CLIP encode
- [ ] k_diffusion/sampling.static.py  采样器
- [ ] sample.static.py         采样入口
- [ ] controlnet.static.py     ControlNet

### Phase 4: 节点 → DAG → 入口
- [ ] nodes.static.py          200+ 节点定义
- [ ] execution.static.py      PromptExecutor
- [ ] main.static.py           CLI 入口

### Phase 5: 端到端验证
- [ ] workflow SDXL → 图片输出
- [ ] --prompt 命令行模式验证

## 命名约定
- ComfyUI 的 `a.py` → `a.static.py`
- 放在 `comfycli/` 目录下（尚未创建，从 Phase 0 起逐文件创建）
- `extern fn` 走 `from "torch_std"` 库

## 关键设计决策
- 命令版先不做 HTTP/WS，后续再补 UI 层
- C++ 层基本不需要新增 helper，libtorch_std_helper.so 已覆盖 90%
- 编译（build.sh）和部署（deploy.sh）职责分离
- ComfyUI 无自有 C/C++ 源文件，所有 C++ 来自外部 pip 包
