# Static ComfyUI — AI 开发指引

## 项目目标
用 StaticPy（Python 子集编译器）+ `libsdcpp_adapter.so`（stable-diffusion.cpp C API 封装）
1:1 重写 ComfyUI，编译为独立 ELF 二进制，零 pip 依赖。

## 理解 StaticPy 语言

编译器核心（`staticpy/static_{translate.py,prelude.scm,stdlib.scm}`）是 `/opt/ReScheme` 上游的**原样拷贝，项目不做修改**。三文件对照阅读：

| 文件 | 读什么 |
|------|--------|
| `staticpy/static_translate.py` | 支持/不支持哪些 Python 语法 |
| `staticpy/static_prelude.scm` | int/float/bool 编译为 fixnum/flonum，文件 I/O/dict/JSON 等内置 |
| `staticpy/static_stdlib.scm` | `extern fn` / 张量函数签名 |

项目侧只维护两处**胶水**（不 fork 翻译器）：
- `comfycli/comfycli_ffi.scm` — `load-shared-object "libsdcpp_adapter.so"` + 上游缺失的内置
- `staticpy/static_build_comfycli.sh` — 本地构建脚本（不链 torch、支持 GLIBC sysroot）

快速理解：
- **值类型**：int→fixnum（机器整数）、float→flonum（64位浮点）、bool→boolean
- **列表**：`list[int]` → Scheme vector，非 Python list（`len`→`list_length`，索引→`vector-ref`）
- **字典**：`make_dict()` / `dict_get()` / `dict_set()` → Scheme hashtable（缺失返回 `#f`）
- **异常**：不支持 try/except，用 `if` + 返回值检查替代
- **类**：不支持继承，用 `@dataclass` + 组合替代
- **FFI**：`extern fn ... from "sdcpp_adapter"` 声明 → 翻译器生成 `foreign-procedure`，`comfycli_ffi.scm` 负责 `load-shared-object`

## 编译与部署

```bash
# 编译 ELF + C++ .so（本地开发）
./build.sh

# 产物
# ./comfycli-bin      — 独立 ELF 二进制
# ./comfycli-bin.so   — Chez AOT 编译产物
# ./cpp/sd/build/libsdcpp_adapter.so  — C++ 适配层（很小，依赖 sd.cpp/ggml 共享库）
# /opt/sd/build-dl/bin/libstable-diffusion.so / libggml.so / libggml-base.so / libggml-cuda.so — sd.cpp 共享库

# 本地运行（动态后端模式，需让 ggml 找到 CUDA 后端插件）
LD_LIBRARY_PATH=cpp/sd/build:/opt/sd/build-dl/bin \
  GGML_BACKEND_PATH=/opt/sd/build-dl/bin/libggml-cuda.so \
  ./comfycli-bin workflow.json --output-dir ./output

# 部署打包（默认 GPU，假设远程已有 CUDA Runtime）
./deploy.sh

# 部署打包 + 包含 CUDA Runtime（远程无需 CUDA Toolkit）
WITH_CUDA=1 ./deploy.sh

# CPU-only 部署（不含 libggml-cuda.so，35MB 左右）
WITH_CUDA_BACKEND=0 ./deploy.sh

# 打包 + SCP 到远程（依赖 SSH 密钥配置）
GLIBC_TARGET=2.35 ./deploy.sh --scp user@remote_host

# 远程运行（零 pip）
ssh user@remote_host "bash /opt/comfycli/run.sh workflow.json --output-dir ./output"
```

注意：`build.sh` 只编译不部署，`deploy.sh` 负责打包 + 传输，职责分离。
后端已切换为动态加载模式：`libsdcpp_adapter.so` 只含适配层，运行时在可执行文件目录或当前目录查找 `libggml-cuda.so` / `libggml-cpu-*.so`。
`WITH_CUDA=1` 会把 `libcudart.so.12` / `libcublas.so.12` / `libcublasLt.so.12` 一起打包；默认只打包 CUDA 后端插件，不打包 CUDA Runtime。
IPAdapter 已改用 sd.cpp 原生实现，不再依赖 ONNX Runtime；部署包约 57MB。

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
comfycli-bin --checkpoint /data/models/image/sd_xl_base_1.0.safetensors --prompt "cat" --output ./out.png

# 模式 1 扩展：覆盖分辨率/采样/seed 等参数
comfycli-bin --checkpoint ... --prompt "cat" --output ./out.png \
  --width 512 --height 512 --steps 5 --seed 1 --cfg 7.0 --sampler euler_a --scheduler discrete

# 模式 2：执行 ComfyUI workflow JSON
comfycli-bin workflow.json --output-dir ./output
```

## 模型文件位置
所有模型在 `/data/models/image/`：
- `sd_xl_base_1.0.safetensors` (6.5G), `clip_g.safetensors` (2.6G), `clip_l.safetensors` (1.6G), `ae.safetensors` (320M)
- `libsdcpp_adapter.so` 已封装 `sd_pipeline_create/load/generate/free` 等 C API，供 StaticPy FFI 调用

## 已搭建的基础设施
- `staticpy/` — StaticPy 工具链（上游 `/opt/ReScheme` 原样拷贝）+ `static_build_comfycli.sh` 本地构建胶水
- `comfycli/comfycli_ffi.scm` — sd.cpp 共享库加载 + 上游缺失的内置（`dict_keys`/`is_none`/`is_link`/`path_dirname` 等）；可选库 `libcomfycli_torch.so` 按是否加载定义 torch 绑定（否则报错桩，避免 AOT 载入期符号缺失）
- `cpp/sd/` — stable-diffusion.cpp 推理后端封装 (`sdcpp_adapter.h/.cpp` + `build.sh` / `build_sd_dl.sh`)；`patches/sdcpp-freeu-sag-v2.patch` 为上游补丁（FreeU/SAG/DynCFG/RescaleCFG/video CFG/sigma 区间/区域条件/GLIGEN 注入等）
- `cpp/libtorch_std_helper.cpp` + `cpp/build_torch_std_helper.sh` — 可选 torch helper（`libcomfycli_torch.so`），仅用于 ggml 没有的**权重级**操作（模型/CLIP 合并、权重导出）与备用独立管线
- `TODO.md` — 待验证节点与所需模型清单
- `build.sh` — 编译 ELF + `libsdcpp_adapter.so`
- `deploy.sh` — 打包依赖 .so + GLIBC 兼容层 + 动态后端插件 + 可选 CUDA Runtime
- `comfycli_remote.sh` + `xgc_ctl.py` + `.env` + `remote_server.md` — Xiangongyun 远程 GPU 部署
- `design.md` — 架构设计、文件映射、C++ 依赖清单
- code search: ComfyUI 全量索引 (797 文件, 25,586 chunks, 4,017 函数) — 见 `comfyui_analysis.md`

## CUDA 显存管理
后端为 stable-diffusion.cpp（`libsdcpp_adapter.so`），显存由 sd.cpp 内部通过 CUDA backend 自动管理；1024×1024 等较大分辨率在 VAE decode 阶段会自动启用 tiling 以避免 OOM。

## StaticPy 关键约束
- 无类继承/多态 → dataclass + 组合
- 无 try/except → if 守卫 + 返回值检查
- 无 lambda 闭包 / eval / exec
- **无一等函数值**：`f = handler`（裸函数引用）会报 `undefined name`（翻译器给用户函数加 `static_` 前缀、引用不加）。因此**不能**用「字符串→函数」查表分发；需要分发时按类分组为多个函数（见 `nodes.static.py` 的 `NODE_GROUP` + `dispatch_*`）。
- `continue` **可用**（见 `cli_args.static.py`）；`break` 未验证，循环退出仍建议改写条件。
- `is None` / `is not None` **可用**（见 `execution.static.py`）。
- 模块级注解字典**可用**（如 `NODE_CLASS_MAPPINGS: dict = make_dict()`，函数内 `dict_set/dict_get` 正常）；早期文档所述「硬错误」已不成立。
- `list` 即 Scheme vector（`len`→`list_length`，索引→`vector-ref`）
- `extern fn foo(x: int) -> int from "sdcpp_adapter"` 声明 C FFI；共享库由 `comfycli_ffi.scm` 加载

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

### Phase 2: 模型加载 + 显存管理（已切换为 sd.cpp 后端）
- [x] `sd_backend.static.py`   stable-diffusion.cpp C API FFI 封装
- [~] `sd.static.py` / `model_base.static.py` / `model_management.static.py` / `lora.static.py` — 传统 PyTorch 模型栈已由 sd.cpp 内置替代，当前通过 C API 直接加载/推理

### Phase 3: 组件级推理
- [x] 由 `libsdcpp_adapter.so` 统一提供：CLIP encode、UNet 采样、VAE decode、ControlNet 等
- [ ] 如需在 StaticPy 层暴露更多采样参数/ControlNet 节点，可继续扩展

### Phase 4: 节点 → DAG → 入口（MVP 已通）
- [x] `nodes.static.py`          132 个节点，覆盖 ComfyUI 全部 120 个内置节点名（100%）；**115 真实实现 / 17 透传**（详见 README「对齐度说明」与 `TODO.md`）
- [x] `execution.static.py`      PromptExecutor（拓扑排序 + 输入链接解析）
- [x] `main.static.py`           CLI 入口（workflow JSON + prompt-only 模式已通）
- [x] `cli_args.static.py`       扩展 CLI 参数：--width/--height/--steps/--seed/--cfg/--sampler/--scheduler
- [x] `deploy.sh`                支持动态后端拆分：CPU-only 35MB，GPU 57MB（远程有 CUDA Runtime，IPAdapter 用 sd.cpp 原生），`WITH_CUDA=1` 含 CUDA Runtime
- [x] `cpp/sd/CMakeLists.txt`      支持 `SD_BACKEND_DL=ON`：CUDA 后端以 `libggml-cuda.so` 插件形式独立加载
- [x] `comfycli_remote.sh`       远程 GPU 实例一键部署脚本（Xiangongyun + scp + run）
- [ ] 200+ 完整节点集（按需逐步补充）

### Phase 5: 端到端验证（workflow + prompt 均已通）
- [x] workflow SDXL → 图片输出（1024×1024，sd_xl_base_1.0 + clip_l/clip_g）
- [x] `--prompt` 命令行模式验证（自动生成 CheckpointLoaderSimple / KSampler / SaveImage 节点）
- [x] `WITH_CUDA=1 ./deploy.sh` 打包部署验证（含 CUDA Runtime）
- [x] 默认 `./deploy.sh` 打包部署验证（含 CUDA 后端插件，不含 CUDA Runtime）
- [x] `WITH_CUDA_BACKEND=0 ./deploy.sh` CPU-only 打包验证（35MB）
- [x] 远程 Xiangongyun 部署脚本 `comfycli_remote.sh` 就绪

## 远程 GPU 部署

针对 Xiangongyun（仙宫云）远程 GPU 实例，已提供一键部署脚本：

- `xgc_ctl.py`：复用 ReScheme 的 Xiangongyun API 控制器，负责实例生命周期（deploy / wait / ssh / shutdown_destroy）。
- `.env`：Xiangongyun API Token + 镜像 ID，脚本自动读取。
- `comfycli_remote.sh`：端到端脚本，自动完成本地编译打包、创建远程实例、同步模型文件、上传二进制、远程运行、下载输出。
- `remote_server.md`：完整远程部署说明。两条路线——**A** `comfycli-bin` workflow（`comfycli_remote.sh` 一键）；**B** `img_hires` 出图管线手动部署（编译命令、scp 文件清单、系统 so 闭包配方、裸机清理）。只跑 `backup*.sh` 出图用 B。

典型用法：

```bash
# 工作流模式
bash comfycli_remote.sh \
  --workflow test_workflow.json \
  --output-dir ./output \
  --name comfy-4090 \
  --with-cuda \
  --run

# prompt-only 模式
bash comfycli_remote.sh \
  --checkpoint /data/models/image/sd_xl_base_1.0.safetensors \
  --prompt "a red apple" \
  --output /tmp/apple.png \
  --width 512 --height 512 --steps 5 \
  --name comfy-4090 --with-cuda

# 销毁止损
python3 xgc_ctl.py shutdown_destroy <instance_id>
```

注意：模型文件（如 sd_xl_base、clip_l、clip_g）较大，脚本默认只打印 scp 命令；
加 `--sync-models` 会自动执行模型同步。远程实例必须有模型文件才能运行推理。

## 命名约定
- ComfyUI 的 `a.py` → `a.static.py`
- 放在 `comfycli/` 目录下
- `extern fn` 走 `from "sdcpp_adapter"` 库；共享库加载与缺失内置统一放 `comfycli/comfycli_ffi.scm`

## 关键设计决策
- 命令版先不做 HTTP/WS，后续再补 UI 层
- C++ 推理后端已切换为 stable-diffusion.cpp（`libsdcpp_adapter.so`），不再依赖 libtorch helper
- StaticPy 编译器核心用 `/opt/ReScheme` **上游原样拷贝，不打补丁**；comfycli 特有内容全放项目侧（`comfycli_ffi.scm` + `static_build_comfycli.sh`），便于跟随上游升级
- 编译（build.sh）和部署（deploy.sh）职责分离
- 产物 ELF 命名为 `comfycli-bin`，避免与源码目录 `comfycli/` 冲突
