# ComfyUI 源码级分析报告

> 分析日期：2026-09-11
> 分析目标：`/opt/static_comfyui/ComfyUI`
> 当前 Commit：`1d48d9cf`（`v0.35.0-11-g1d48d9cf`）
> 版本：`0.35.0`
> 命名空间：`/code/comfyui` · 分析目录：`/opt/code_caches/comfyui_cache` · KV Cache：`/memory`
> 分析工具：my_db code search system（code_indexer + batch_embedder + call_graph + dataflow + cache_import）

---

## 一、项目概览

| 指标 | 本次（2026-09-11） | 上次（2026-07-05） | 变化 |
|------|------|------|------|
| 源码文件 | 797 | 656 | +21% |
| 索引 Chunks | 25,586 | 20,141 | +27% |
| 唯一函数（调用图） | 4,017 | 2,633 | +53% |
| 函数体 | 5,378 | — | — |
| 调用边 | 1,357 | 864 | +57% |
| 唯一变量 / 跟踪字段 | 2,000 / 1,213 | 2,000 / — | — |
| 向量条数 / 维度 | 25,586 / 768 | 20,141 / 768 | — |
| 注册节点（nodes.py） | 65 | — | — |
| 节点文件（comfy_extras） | 136 | — | — |
| 合作伙伴节点文件 | 41 | — | — |
| 模型架构类（supported_models） | 102 | — | — |
| LatentFormat 类 | ~50 | — | — |

**Chunk 类型分布**：class 8,516 / function 5,378 / module 5,287 / unknown 5,247 / file 797 / namespace 361，语言 100% Python。

ComfyUI 是节点流程图驱动的扩散模型工作流引擎：用户连线构建 DAG，后端按拓扑序执行节点，推理计算交给 PyTorch。核心设计是把生成流程拆解为可组合节点，并以「模型补丁（ModelPatcher）+ 钩子（Hook）」实现 LoRA/ControlNet/风格注入等非侵入式改造。

---

## 二、源码目录结构

```
ComfyUI/
├── main.py / execution.py / server.py / nodes.py / node_helpers.py
├── folder_paths.py / protocol.py / latent_preview.py / cuda_malloc.py
├── comfy/                    # 核心库：模型定义 + 推理（44 顶层 .py + 子目录）
│   ├── sd.py                 # 模型加载（diffusion/CLIP/VAE/LoRA）
│   ├── model_management.py   # 显存/设备调度
│   ├── memory_management.py  # 内存管理（新增）
│   ├── system_memory.py      # 系统内存探测（新增）
│   ├── model_patcher.py      # 模型补丁（LoRA/ControlNet/动态加载）
│   ├── patcher_extension.py  # 补丁扩展（新增）
│   ├── hooks.py              # 钩子系统（新增，786 行）
│   ├── model_base.py / model_detection.py / supported_models*.py
│   ├── clip_model.py / clip_vision.py / sd1_clip.py / sdxl_clip.py
│   ├── controlnet.py / lora.py / lora_convert.py / taesd.py
│   ├── latent_formats.py / model_sampling.py
│   ├── samplers.py / sample.py / sampler_helpers.py / k_diffusion/
│   ├── ops.py / quant_ops.py / rmsnorm.py / nested_tensor.py
│   ├── pinned_memory.py / model_prefetch.py / multigpu.py
│   ├── context_windows.py / pixel_space_convert.py
│   ├── ldm/                  # 各模型族网络定义（sdxl/flux/wan/…）
│   ├── text_encoders/ clip_vision/ image_encoders/ audio_encoders/
│   ├── cldm/ t2i_adapter/ extra_samplers/ weight_adapter/ taesd/
│   └── ...
├── comfy_execution/          # 执行引擎扩展
│   ├── graph.py              # DynamicPrompt + TopologicalSort + ExecutionList
│   ├── caching.py            # CacheKeySet / BasicCache / 输入签名缓存
│   ├── cache_provider.py     # 缓存提供者（Hierarchical/LRU/RAM/Null）
│   ├── jobs.py               # 任务模型（新增，550 行）
│   ├── validation.py / asset_enrichment.py / progress.py / utils.py
├── comfy_extras/             # 额外节点（136 个 nodes_*.py）
├── comfy_api/                # ComfyAPI（latest/ NodeReplace 注册）
├── comfy_api_nodes/          # 合作伙伴节点（41 个）
├── comfy_config/ app/ api_server/ middleware/ blueprints/
├── alembic_db/ tests/ tests-unit/ script_examples/
└── models/ input/ output/ custom_nodes/ utils/
```

---

## 三、分层架构

```
┌─────────────────────────────────────────────────────────────┐
│ 1 入口层   main.py: prompt_worker(351) → startup_server(568) │
├─────────────────────────────────────────────────────────────┤
│ 2 服务层   server.py: PromptServer(215) + comfy_api/         │
│            WebSocket 推送 / HTTP /prompt /queue /history     │
├─────────────────────────────────────────────────────────────┤
│ 3 执行引擎 execution.py + comfy_execution/                   │
│            PromptExecutor(664) → ExecutionList(193) 拓扑执行 │
│            get_input_data(159) / validate_inputs(846)        │
│            缓存 caching.py + IsChangedCache(60)              │
├─────────────────────────────────────────────────────────────┤
│ 4 节点系统 nodes.py(65) + comfy_extras/(136) + api_nodes(41) │
│            NODE_CLASS_MAPPINGS + NodeReplaceManager          │
├─────────────────────────────────────────────────────────────┤
│ 5 模型推理 comfy/                                            │
│   sd.py 加载 → model_patcher 补丁 → model_base.apply_model   │
│   samplers/sample 采样 · clip 文本 · vae 编解码 · hooks 钩子 │
│   model_management 显存/设备调度                             │
├─────────────────────────────────────────────────────────────┤
│ 6 基础设施 comfy_config/ folder_paths.py utils/ alembic_db/  │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、核心执行引擎（`execution.py` + `comfy_execution/`）

### 4.1 关键函数（精确行号）

| 符号 | 位置 | 职责 |
|------|------|------|
| `IsChangedCache` | `execution.py:60` | `IS_CHANGED` 判定，决定节点是否重算 |
| `get_input_data` | `execution.py:159` | 解析节点输入：链接 `[node_id, idx]`、hidden inputs（`unique_id/prompt/extra_pnginfo`） |
| `_async_map_node_over_list` | `execution.py:243` | **list 输入广播**：把 list 输入逐元素映射到节点函数 |
| `get_output_data` | `execution.py:343` | 调用节点 `FUNCTION` 并收集输出 |
| `execute` | `execution.py:438` | 单节点执行（含 subgraph/async/ExecutionBlocker 处理） |
| `PromptExecutor` | `execution.py:664` | 执行器主体（`execute_async`） |
| `validate_inputs` | `execution.py:846` | 递归校验节点与链接，去重 |
| `PromptQueue` | `execution.py:1251` | 任务队列（入队/出队/中断/历史） |

### 4.2 图与调度（`comfy_execution/graph.py`）

| 符号 | 位置 | 职责 |
|------|------|------|
| `DynamicPrompt` | `graph.py:21` | 动态 prompt 解析（含 subgraph 展开） |
| `get_input_info` | `graph.py:64` | 取节点输入信息 |
| `TopologicalSort` | `graph.py:106` | DAG 拓扑排序（`DependencyCycleError` 检测环） |
| `ExecutionList` | `graph.py:193` | 可执行节点列表 + 阻塞/就绪判定 |

### 4.3 缓存（`comfy_execution/caching.py`）

- `CacheKeySet`（`:26`）→ `CacheKeySetID`（`:67`）/ `CacheKeySetInputSignature`（`:82`，按输入签名做 key）
- `BasicCache`（`:150`）管理 local/subcache，`clean_unused()` 回收
- `cache_provider.py` 提供 Hierarchical / LRU / RAMPressure / Null 四种缓存

### 4.4 执行流程（函数级）

```
PromptExecutor.execute_async
  ├─ validate_inputs(846)              递归校验
  ├─ IsChangedCache(60) 判缓存
  ├─ ExecutionList(193) 拓扑排序
  └─ while 未空:
       ├─ execution_list.stage_node_execution()
       ├─ execute(438)
       │    ├─ get_input_data(159)      解析输入/链接
       │    ├─ _async_map_node_over_list(243)   list 广播
       │    ├─ get_output_data(343)     调用节点 FUNCTION
       │    └─ 写缓存
       └─ WebSocket 推送进度
  └─ cleanup_models_gc(1041)
```

---

## 五、节点系统

### 5.1 节点契约（以 `KSampler` 为例，`nodes.py:1597`）

```python
class KSampler:
    @classmethod
    def INPUT_TYPES(s):   # required/optional/hidden，含类型 + 默认值 + tooltip
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"    # 实际调用的方法名
    CATEGORY = "model/sampling"
    DESCRIPTION = "..."
    SEARCH_ALIASES = [...]  # 前端搜索别名（v0.35）
    def sample(self, ...): return common_ksampler(...)
```

关键数据类型：`MODEL`=ModelPatcher，`CONDITIONING`=`[[tensor, {pooled_output, control, area, ...}]]`，`LATENT`=`{"samples": tensor, ...}`，`IMAGE`=torch tensor `[B,H,W,C]`。

### 5.2 代表节点（`nodes.py`）

| 节点 | 行号 | 节点 | 行号 |
|------|------|------|------|
| `CLIPTextEncode` | 56 | `common_ksampler` | 1572 |
| `VAEDecode` | 316 | `KSampler` | 1597 |
| `VAEDecodeTiled` | 343 | `KSamplerAdvanced` | 1626 |
| `CheckpointLoaderSimple` | 616 | `SaveImage` | 1660 |
| `EmptyLatentImage` | 1246 | `LoadImage` | 1738 |

### 5.3 自定义节点加载链（`nodes.py`）

```
init_extra_nodes(2571)
  ├─ init_builtin_extra_nodes(2394)      # comfy_extras/
  ├─ init_builtin_api_nodes(2552)        # comfy_api_nodes/
  └─ init_external_custom_nodes(2344)    # custom_nodes/ 目录发现
        └─ load_custom_node(2246)        # 动态 import 模块
```

---

## 六、模型层（`comfy/`）

### 6.1 加载与检测（`comfy/sd.py`）

| 符号 | 行号 | 说明 |
|------|------|------|
| `load_checkpoint_guess_config` | 2092 | 从 checkpoint 猜 UNet/CLIP/VAE 配置 |
| `load_checkpoint_guess_config_model_only` | 2143 | 仅模型 |
| `load_diffusion_model_state_dict` | 2262 | state_dict → 模型 |
| `load_diffusion_model` | 2359 | 按路径加载 diffusion |
| `load_vae_patcher` | 2369 | VAE 加载 |
| `load_clip` | 1562 | CLIP 加载 |
| `class CLIP` / `class VAE` | 237 / 487 | 文本编码器 / 变分自编码器 |

- 架构注册表 `supported_models.py`：**102 个模型类**（SD15/SDXL/SD3/Flux/Wan/Hunyuan/…）
- `model_detection.py`：`detect_unet_config` 按 state_dict key 模式识别架构
- `latent_formats.py`：**~50 个 LatentFormat**（SD15/SDXL/Flux/Flux2/Wan/LTXV/MiniMax/Trellis2/…），定义潜空间缩放与 sigma

### 6.2 模型补丁（`comfy/model_patcher.py`）

| 符号 | 行号 | 说明 |
|------|------|------|
| `ModelPatcher` | 340 | 包装 torch 模型，管理 patches/lowvram/设备 |
| `clone` | 430 | 克隆（共享权重，独立补丁） |
| `set_model_unet_function_wrapper` | 655 | 注入 UNet forward 包装（FreeU/自定义） |
| `load` | 982 | 按显存策略加载权重 |
| `patch_model` | 1113 | 应用补丁到权重 |
| `ModelPatcherDynamic` | 1749 | 动态权重加载变体 |

`model_base.py`：`ModelType`(97)、`BaseModel`(166)、`apply_model`(207) —— 所有架构的推理入口。

### 6.3 钩子系统（`comfy/hooks.py`，v0.35 新增，786 行）

`Hook`(82) 为基类，派生 `WeightHook`(131)、`ObjectPatchHook`(191)、`AdditionalModelsHook`(206)、`TransformerOptionsHook`(229)、`InjectionsHook`(270)；`HookGroup`(287) 管理分组，`HookKeyframe`(420)/`HookKeyframeGroup`(446) 支持关键帧插值。LoRA 也可通过 `create_hook_lora`(602) 以 Hook 形式注入。

---

## 七、采样栈（`comfy/samplers.py` + `sample.py` + `k_diffusion/`）

```
nodes.py: common_ksampler(1572)
  └─ comfy/samplers.py: sample(1349)
       ├─ sampler_object(1388)            选采样器
       ├─ KSampler(1399) 采样循环
       │    ├─ CFGGuider(1188)            CFG 引导
       │    │    └─ calc_cond_batch(208)  正/负条件批量推理
       │    │         └─ get_area_and_mult(33)  区域/强度条件
       │    └─ comfy/k_diffusion/sampling.py  各采样算法（如 DDPMSampler_step:1052）
       └─ model_sampling.py: ModelSamplingDiscrete(141)   sigma/时间步调度
```

| 符号 | 位置 |
|------|------|
| `get_area_and_mult` | `samplers.py:33` |
| `calc_cond_batch` | `samplers.py:208` |
| `CFGGuider` | `samplers.py:1188` |
| `sample` | `samplers.py:1349` |
| `KSampler`（采样器基类） | `samplers.py:1399` |
| `DDPMSampler_step` | `k_diffusion/sampling.py:1052` |
| `ModelSamplingDiscrete` | `model_sampling.py:141` |
| `_calc_cond_batch_multigpu` | `samplers.py:358`（v0.35 多卡） |

---

## 八、文本 / VAE / LoRA

- **CLIP**：`comfy/clip_model.py:148` `CLIPTextModel_`；分词 `text_encoders/bpe_tokenizer.py:117` `BPETokenizer`；SDXL 双编码器 `sdxl_clip.py`
- **VAE**：`comfy/sd.py:1190` `VAE.encode_tiled_1d::encode_fn`（tiling 编码）、`:1172` `encode_tiled_`；节点 `nodes.py:343` `VAEDecodeTiled`
- **LoRA**：`comfy/lora.py:451` `calculate_weight`、`:516` `prefetch_prepared_value`；`comfy/sd.py:103` `load_lora_for_models`、`:138` `load_bypass_lora_for_models`

---

## 九、显存 / 内存管理（`comfy/model_management.py`）

| 函数 | 行号 | 作用 |
|------|------|------|
| `get_torch_device` | 195 | 选择设备 |
| `free_memory` | 879 | 按需求量释放 |
| `load_models_gpu` | 925 | 批量加载模型到 GPU（带 offload 调度） |
| `cleanup_models_gc` | 1041 | GC + 清理 |
| `pin_memory` | 1624 | 锁页内存 |
| `get_free_memory` | 1765 | 查询可用显存 |
| `soft_empty_cache` | 2080 | CUDA cache 软清空 |
| `unload_all_models` | 2098 | 卸载全部模型 |

**调用关系**（code search）：`load_models_gpu ← calc_lora_model / load_model_gpu / _prepare_sampling`。

v0.35 进一步拆出 `memory_management.py`（187 行）、`system_memory.py`（128 行）、`pinned_memory.py`（127 行）、`multigpu.py`（254 行）、`model_prefetch.py`（267 行）。

---

## 十、v0.35 新增子系统（本次索引才可见）

| 文件 | 行数 | 说明 |
|------|------|------|
| `comfy/hooks.py` | 786 | 统一钩子/关键帧系统 |
| `comfy_execution/jobs.py` | 550 | 任务模型 |
| `comfy/quant_ops.py` | 276 | 量化算子（fp8 等） |
| `comfy/model_prefetch.py` | 267 | 权重预取 |
| `comfy/multigpu.py` | 254 | 多 GPU |
| `comfy/memory_management.py` | 187 | 内存管理 |
| `comfy/patcher_extension.py` | 159 | 补丁扩展 |
| `comfy/system_memory.py` | 128 | 系统内存探测 |
| `comfy/pinned_memory.py` | 127 | 锁页内存 |
| `comfy/rmsnorm.py` | 11 | RMSNorm |
| `comfy_execution/validation.py` | — | 校验拆分 |

模型族新增：`ldm/sam3d_body/`、`ldm/seedvr/`、`ldm/sensenova/`、`ldm/minimax/`、`ldm/minimax_music/`、`ldm/mage_flow/`、`ldm/joyimage/`、`ldm/anima/`、`ldm/trellis2/`、`ldm/lightricks/` 等。

---

## 十一、完整执行流程

```
[Web UI] ── POST /prompt ──→ PromptServer(215).post_prompt
                                 ├─ validate_inputs(846)
                                 └─ queue.push → 200 OK + prompt_id
                                        │
                        prompt_worker(351) 取出任务
                                 └─ PromptExecutor(664).execute_async
                                      ├─ 判缓存 IsChangedCache(60)
                                      ├─ ExecutionList(193) 拓扑排序
                                      └─ 逐节点:
                                           execute(438)
                                             ├─ get_input_data(159)
                                             ├─ _async_map_node_over_list(243)
                                             ├─ get_output_data(343) → 节点 FUNCTION
                                             └─ 缓存 + WebSocket 推送
                                      └─ cleanup_models_gc(1041)
[Web UI] ←── WebSocket 进度/结果 ──┘
```

典型 txt2img 节点链：`CheckpointLoaderSimple(616)` → `CLIPTextEncode(56)` ×2 → `EmptyLatentImage(1246)` → `KSampler(1597)` → `VAEDecode(316)` → `SaveImage(1660)`。

---

## 十二、关键设计模式

| 模式 | 位置 | 说明 |
|------|------|------|
| 工厂 | `sd.py:2092 load_checkpoint_guess_config` | 按文件推断架构并建模型 |
| 策略 | `supported_models.py`（102 类） | 各架构统一接口 |
| 观察者 | `server.py` WebSocket | 执行进度推送前端 |
| 装饰器/补丁 | `model_patcher.py` + `hooks.py` | LoRA/ControlNet 非侵入注入 |
| 命令队列 | `execution.py:1251 PromptQueue` | Prompt 入队顺序执行 |
| 缓存 | `comfy_execution/caching.py` | 输入签名缓存 + `IS_CHANGED` |
| DAG 调度 | `comfy_execution/graph.py:193` | 拓扑排序 + 就绪/阻塞 |
| 插件 | `nodes.py:2571 init_extra_nodes` | 三层节点初始化 |

---

## 十三、与主流项目对比

| 维度 | ComfyUI | A1111 | Diffusers |
|------|---------|-------|-----------|
| 架构 | 节点图 DAG | 单页表单 | Python API |
| 执行模型 | 懒加载 + 拓扑排序 | 顺序执行 | 程序控制 |
| 显存管理 | 动态 offload + 多卡 | 手动切换 | 手动管理 |
| 扩展性 | 自定义节点 + Hook | 脚本扩展 | Python 直接 |
| 并发 | asyncio + 多客户端 | 单用户 | N/A |

---

## 十四、code search 查询入口

```bash
# 语义搜索（必须带 --analysis-dir）
/opt/my_db/tools/cache_query "sampling denoise scheduler sigma" \
  --repo /code/comfyui --type search \
  --analysis-dir /opt/code_caches/comfyui_cache --max-results 5

# 调用关系（caller/callee/调用链/数据流）
/opt/my_db/tools/cache_query load_models_gpu \
  --repo /code/comfyui --type context --depth 2

/opt/my_db/tools/cache_query get_input_data \
  --repo /code/comfyui --type context --depth 1
```

> KV Cache 默认目录 `/memory`（coding.md 约定），无需显式 `--cache-dir`。
> 注意：调用图由文本匹配构建，方法级调用（`model.apply_model` 等）覆盖有限，关键流程建议结合源码行号确认。
