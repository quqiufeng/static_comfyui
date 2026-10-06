# TODO

## 参考图 / IPAdapter 说明

`--ipadapter`（sd.cpp 原生）只支持 **SD1.5 / SDXL**；**z_image(DiT) 不支持 IPAdapter**。
z_image 侧“参考图 / 同人 / 同风格”改用：

- **LoRA**：`train_lora/`（musubi 训练身份/风格 LoRA）+ `backup.sh --lora`（多 LoRA 叠加、触发词自动注入）——即 IPAdapter 的等价物；
- **原生参考图**：`img_hires --ref-image`（z_image Omni/编辑，无需训练）；
- **结构控制**：`preprocesscli` + `backup.sh --control-image`（canny/depth/pose/hed/mlsd/lineart/gray，见 `cpp/sd/z_image_controlnet.md`）。

→ 不要再为 z_image 找 `--ipadapter`；SD1.5/SDXL 仍可用原生 `--ipadapter`。

## 待验证（代码已实现，缺模型文件）

以下节点代码已实现，但本机 `/data/models` 无对应模型，**尚未运行验证**。

### GLIGEN（`GLIGENLoader` / `GLIGENTextBoxApply`）— 部分实现
- 已完成：`src/model/diffusion/gligen.hpp`（`GatedSelfAttentionDense`/`PositionNet`/`GligenModules`）；UNet transformer 块注入（`BasicTransformerBlock` self-attn 后，`transformer_index` 递增）；`GGMLRunnerContext`/`UNetDiffusionExtra`/`StableDiffusionGGML` 传递链；`sd_ctx_params_t.gligen_path`
- **待完成**：GLIGEN 权重加载 runner（含 `alpha_attn`/`alpha_dense` 标量）、PositionNet objs 计算（Fourier + MLP）、节点接线
- 需要模型：GLIGEN checkpoint（含 `position_net.*` 与 `*.fuser.*`）
- 验证：CheckpointLoader → GLIGENLoader → GLIGENTextBoxApply → KSampler，确认区域物体按 box 出现

### StyleModel（`StyleModelLoader` / `StyleModelApply`）
- 需要模型：含 `style_embedding` 的 StyleAdapter，或含 `redux_down.weight` 的 Flux Redux
- 备注：现有 `t2iadapter_*.pth` 是 ControlNet 型，走 `ControlNetLoader`（已支持），不是 style model
- 验证：style model → StyleModelApply → KSampler

### unCLIP（`unCLIPConditioning`）
- 需要模型：unCLIP prior（`model.diffusion_model.*` 为 prior transformer）

### SVD_img2vid（`SVD_img2vid_Conditioning`）
- 需要模型：Stable Video Diffusion（svd / svd_xt）；sd.cpp 支持 SVD，需走视频管线

### CLIPVisionEncode
- 架构障碍：sd.cpp 的 clip_vision 从主模型权重表构建，节点无 `model` 输入拿不到 pipeline
- 已备 C API：`sd_clip_vision_encode` / `sd_pipeline_clip_vision_encode`
- 待办：给 `CLIPVisionLoader` 增加 standalone clip_vision ctx，或让节点携带 pipeline

## 待验证（有模型，可直接测）
- `ModelMerge*`：用 Juggernaut-XI / DreamShaperXL / RealVisXL 做不同模型合并验证
- `ModelMergeSimple/Blocks/Add/Subtract`：ratio / 逐块 ratio 行为
- `CLIPMerge*`：不同 CLIP 合并

## 采样加速（已完成，2026-09-24）
**目标**：E1xMIN 2560×1440 出图 ~11min → ≤5min。

| 项 | 原理 | 结果 |
|----|------|------|
| **EasyCache**（默认开） | 相邻步 latent 变化 < 阈值则复用、跳过本步 forward；turbo 后期更易命中 | base 跳 9/20，hires 跳 ~30/41；**主因** |
| **GGML_CUDA_GRAPHS=ON** | 一步采样 kernel 序列录成 CUDA graph 一次提交，减 launch 开销 | 再省数秒～十数秒（`build_sd_dl.sh`） |
| 分段计时 | `img_hires` 打 load/generate/post/save + EasyCache summary | 便于回归对比 |

**实测**（RTX 3080 20G，seed=25630，backup.sh 默认提示词）：
- 优化前 ~665s → **210s（3m30s，~3.2×）**；hires 采样 531s→142s。
- 开关：`CACHE_MODE=disabled|easycache`（默认 easycache）；`CACHE_THRESHOLD` 越低跳越多（默认 0.2）。已同步到 `backup.sh` / `backup_qwen.sh` / `backup_scene.sh`。
- 接线：`ImageGenerationParams.cache_*` → `sd_img_gen_params_t.cache` → `SampleCacheRuntime`；CLI `--cache-mode/--cache-threshold/--cache-start/--cache-end`。
- 未做：降 hires steps（画质换速度）。
- ~~batch CFG~~ **已证伪，勿再做**：z_image / Qwen / SDXL 文本变长（TE 无 min_length），分支合批
  要求 `L_cond == L_uncond` 恒成立才与串行 1:1 等价（图像 rope 索引含 padded_context_len）；
  nihui ncnn 两项目均为串行两趟，ComfyUI / sd.cpp 亦无 batch CFG。SDXL 方案被否（过时）。

## VAE tile 显存自适应（已完成，2026-10-01）
**目标**：VAE decode/encode 按运行时空闲显存选 tile，替代静态启发式（ncnn qwenimage-ncnn-vulkan/src/vae.cpp → sd.cpp 移植）。

- 实现：`sd::backend_fit::fit_vae_tiling_to_memory`（`src/core/backend_fit.cpp`），在
  `decode_first_stage`（`diffusion_engine.cpp`）权重已加载后调用；候选/目标函数复刻 ncnn
  （min tile count → min area → min pixels），count 用 `sd_tiling_calc_tiles` 非 circular 分支
  精确复刻（`target_overlap=0.5` 时 180/60 实测 5 tiles 而非 3，不算 overlap 会选错）。
- 显存模型：`budget = free − 160MiB(reserve) − 64MiB(overhead)`，`bytes/px` 默认 dec 8704 /
  enc 2304（ncnn bf16 实测 ×2 覆盖 f32），env `SD_VAE_MEM_PER_PX_DEC/_ENC` 可校准，
  `SD_VAE_TILING=manual` 完全关闭（保留静态/OOM-retry 兜底，CPU 后端跳过）。
- 校准法：跑一次看 `vae decode fit:` 日志与 `ggml_runner.cpp:994 compute buffer size` →
  bpp = buffer_MiB×1048576/pixels；实测 z_image VAE decode ≈6600 B/px（1024 全图 6657 MiB、
  2560×1440 tile 107×180 7825 MiB，线性一致），默认 8704 留 1.3× 余量，无需改。
- 实测（RTX 3080 20G，2560×1440 z_image）：auto 4 tiles **decode 4.58s** vs manual(128) 8 tiles
  7.55s（-40%）；1024×1024 自动全图单 tile 1.15s；均 exit 0 出图正常。
- 未做：encode 接线（`get_tile_sizes` 对 encode 有 encoding_factor×2 歧义，先跑通 decode）、
  video/circular 分支（fit 已跳过）、comfycli 端 SDXL 验证（本机无 sd_xl_base 模型）。
- 构建注意：`cpp/sd/build_sd_dl.sh` 每次 `rm -rf` 全清重建；改 `/opt/sd` 后增量编译用
  `cmake --build /opt/sd/build-dl --target stable-diffusion -j$(nproc)`（约 1 分钟），无需重跑
  `build.sh`（adapter/ELF 未变，动态加载新 .so）。

## prefix KV caching（#1）：benchmark 完成，判定不迁移（2026-10-01）
**方法**：prompt 长度 A/B（img_hires 直跑，1024²、6 步、`--cache-mode disabled`、seed42、cfg2.0
双分支，各 2 次）：短 prompt（~45 tok 含质量前缀）vs 长 prompt（1307 tok，无截断）。
`sampling completed`（不含 TE）：19.81s → 23.02s，**Δ=0.535s/步**（2 次重复差 <0.3s）。

**结论**（z_image，1024²，上界 = text 全缓存含 prelude + joint text 行）：
- 长 prompt 极端档：sampling 的 **~16%**；典型 prompt（ΔT≈55~150 tok）：**0.35~1%**；
  主力 2560×1440（img tokens 14400）占比更小，EasyCache 跳步后再打折。
- ncnn 项目该优化的主因是**显存**（README low-vram：prefix KV 可落 host RAM、
  decode 激活 ∝ image-only q 更小），非速度。
- 成本：z_image joint 层为全双向 concat 注意力（ComfyUI `lumina/model.py` mask=None，
  sd.cpp mask=nullptr 同步），text 行依赖 image → 复刻 ncnn 冻结 text KV 即**近似**、
  破坏与 ComfyUI 1:1；且需拆 JointAttention + 每层持久 KV 张量，改动大。
- 判定：**典型收益 <1%，不迁移**。若未来要做低显存模式（小卡部署），再评估
  ncnn 式 host-KV（届时以显存收益为指标）。

## HiRes Fix 出图质量优化
**原理**：latent 放大 + 二次采样（denoise<1），即 ComfyUI 的 `LatentUpscale`（bicubic/bislerp）+ `KSampler(denoise)`；`backup.sh` 同原理（低分构图 → latent 放大 refine，基础分辨率越高、放大倍数越小越好）。

已修（对齐 `cpp/sd/backup.sh` 配方）：
- `img_hires.cpp` 默认值：quality prefix 原样对齐 backup.sh；默认负面词对齐 backup.sh；
  `--hires-steps` 默认 20→45；后处理默认开启并取 backup.sh 值（clarity 0.2 / sharpen 0.3 /
  smart 0.5 / edge 1.5）
- `HiResFix` 节点：prompt 未含 masterpiece 时前置 backup.sh 质量关键词（`quality_prefix=0` 可关）；
  负面词留空时补 backup.sh 默认负面词（`default_negative=0` 可关）
- 模型相关项（cfg / sampler / scheduler / FreeU / VAE tiling）仍由调用方显式传入，
  不设为默认，避免影响 SDXL 等不同模型的正常区间

历史修复（`23d9050` + `c950631`）：
- 修复 `hires.model_path` 从未设置 / 枚举串大小写不匹配（"Model" vs "model"）
- 默认改回 **latent-bicubic**（sd.cpp 的 `LATENT` 实为 Bilinear 偏软，ComfyUI 常用 bicubic）
- 后处理默认关闭（旧 backup.sh 不做后处理）
- base 分辨率计算与 backup.sh 一致（2560×1440→1920×1080 等）
- ESRGAN 保留为可选（`upscaler=model` + `upscaler_model`）

仍待做：
- **二次采样参数**：`hires_strength`/`hires_steps`/`scheduler`/`cfg` 已对齐 backup.sh（0.35/45/…），如需更好需跑实验标定
- **FreeU 默认值**：已对齐 backup.sh（b1=1.3/b2=1.4）；旧 sdxl_pipeline 曾记录过拟合，待复标
- **VAE tiling 接缝**：参数已对齐（128/0.5），高分辨率 tile 边界若有接缝需查 sd.cpp tiling
- **bislerp**：已实现（`upscaler=latent-bislerp`，对齐 ComfyUI `bislerp`）；默认仍 bicubic
- **多步渐进放大** / **与 ComfyUI 同工作流质量对照**

## 已完成并验证
- `ModelNoiseScale`：euler_a 下 noise_scale=8 与基线不同
- `ModelSamplingDiscrete`：eps 与基线一致、v_prediction 不同
- 区域条件（`ConditioningSetArea` 等）：左上区域统计与其余不同，峰值显存 ~2.7GB
- `ModelMerge*` 自合并 ratio=0.5 与基线逐字节一致
- `CLIPMergeSimple` 自合并 ratio=0.5 与基线逐字节一致
- `CheckpointSave` 导出 6.9GB 权重

## 代码复盘 P0 修复（2026-10-01：双 agent 审计 23+25 条 + 逐条人工复核）

已修（编译通过 + e2e 验证）：
- prompt 模式丢 prompt：`ksampler` 改 `resolve_prompt_text` 回退（`main.static.py` 写 `prompt` 键，
  ksampler 此前只读 `positive` → 空提示词出图）
- 不带 `--output-dir` 必然 rc=-1：`main.static.py` 空串默认 `./output`（cli_args 初值 `""` 判不住 None）
- 退出码恒 0：`execute_prompt` 成功写 `_ok` 标记，校验失败/中止/环 → `exit 1`
- 声明输出数>返回数越界崩：`execution.static.py` resolve/upstream_missing 加边界；
  `LoadImage`/`ImagePadForOutpaint` 补 mask=None；`KSamplerAdvanced` 删多余 IMAGE 输出；
  `unCLIP`/`ImageOnly` 专用包装（CLIP_VISION→None，不再错传文本 CLIP 句柄）；
  `LoraLoader` 返回 `(model, clip)` 透传 + 读 `strength_model` 键
- `conditioning_passthrough` 补实现（7 节点此前 unbound 崩）
- FFI seed 32 位：`sd_backend.clamp_seed` 折回 31 位（e2e：seed=4294967295 → seed=1，确定性）
- `KSamplerAdvanced` 读 `noise_seed`（ComfyUI 标准键，此前恒默认 42）
- HiResFix `seed=0` 不再翻译成随机（ComfyUI 0 = 确定性）
- `LatentRotate` "90 degrees" 按前缀解析；`LatentFlip` 按 `y-axis` 前缀分支（此前恒 0/恒竖翻）
- `LoraLoaderBypass` 改真透传（此前误走 lora_loader 会真加载）
- `next_val` 子 shell bug（backup.sh / backup_scene.sh）：`--lora x:0.8 "prompt"` 曾把 `x:0.8` 当 prompt
- `HIRES=0` 关断两阶段：单阶段直出目标分辨率（`HIRES_STEPS=0` 不是关断哨兵）
- EasyCache 阈值文档方向反转：代码 `cumulative < threshold → skip` 即**阈值越高跳越多**
 （backup.sh ×2、img_hires usage 已改；调参方向按"高=更快易花"重新理解）

代码复盘 P0 修复后续 — A 轮（2026-10-01，comfycli 接 EasyCache setter）：
- adapter：`Impl` 加 cache 四字段 + `SDPipeline::set_cache` + generate 中 impl_ 优先
 （per-call params 回退，img_hires 路线不受影响）+ `sd_pipeline_set_cache(mode 字符串→sd_cache_mode_t,
 含 ucache/dbcache/taylorseer 全枚举，未知名告警回落 disabled)`
- StaticPy：`sd_backend` extern + `sd_set_cache`；`nodes.parse_sampler_opts` 加
 cache_mode/cache_threshold/cache_start/cache_end（默认 easycache/0.2/0.15/0.95 对齐 backup.sh）；
 `run_sampler` 单漏斗（4 采样点全过它）调 `sd_set_cache`
- 构建：只编 `sdcpp_adapter_shared`（img_hires 二进制未动，保 strings 规格，B 轮回源用）
- e2e：默认→`EasyCache enabled threshold 0.200` + rc=0 + png；`cache_threshold=0.5`→
 `skipped 3/15 steps (1.25x)`（阈值钮丝全链路）；`cache_mode=bogus`→adapter 告警回落+rc=0
- 遗留 #7 关闭

代码复盘修复 — B 轮（2026-10-01，img_hires 源码回源 + sampler 映射告警）：
- `img_hires.cpp` 回源 4 组 flag（与二进制 strings 规格逐字对账 13/13）：
  `--llm-vision`（mmproj）、`--ref-image`（可重复）/`--ref-image-args`、`--ipadapter` 5 件套
 （缺参报错文案同源）、config 打印 `ipadapter:` / `ref-image:` 行
- adapter：`ModelConfig.llm_vision_path` → `sd_ctx_params_t.llm_vision_path`；
  `SDPipeline::set_ref_image`（cv::imread→sd_image_t 持久缓冲，日志同二进制文案）→
  generate 写 `img_params.ref_images/ref_images_count/ref_image_args`；C API `sd_pipeline_set_ref_image`
- sampler/scheduler 别名（ComfyUI→sd.cpp）：`euler_ancestral→euler_a`、`ddim→ddim_trailing`、
  `dpm_2→dpm2`、`dpmpp_2m(_sde/_gpu)→dpm++2m(_sde)`；`normal/ddim_uniform/uniform/linear→discrete`
 （comfycli 默认 scheduler "normal" 终于命中）；未知名 `[C++ gen] unknown sampler/... fallback` 告警
 （不再静默回落）
- 统一重编（增量不 wipe）；usage 中 P0 的 "higher = more skips" 生效
- e2e：euler_ancestral+normal 零告警 rc=0；foo_sampler 告警回落 rc=0；`--ipadapter` 缺参 rc=1；
  基础出图 8.79s 无回归；**Qwen-Image-2.1 真测 --llm-vision + --ref-image**
 （mmproj 加载 + `enable llm vision` + `ref-image: 1 image(s)` + 22.5s 出图 rc=0）
- 遗留 #2 关闭。新遗留：
  (a) ~~mmproj 需 `libggml-cpu.so` 在搜索路径，deploy.sh GPU 模式需补打包~~
      **决定不做（2026-10-01）：CPU 后端太慢，远程不支持 --llm-vision/--ref-image 路线，
      deploy.sh 保持只打包 cuda 插件**；本地 `cpp/sd/build/` 的 cpu 软链保留供开发机测试
  (b) `--ipadapter` 运行时未测（无 SD1.x/SDXL 模型），仅验证参数校验错误路径

代码复盘修复 — C 轮（2026-10-01，KSampler LatentImage 化 + latent 链架构）：
- **StaticPy 关键坑（记档）**：无注解参数的 dataclass 属性访问会被翻成裸调用
 （`v.image_path` → `(image_path v)` 未绑定崩溃）→ helper 参数必须写 `v: LatentImage`；
 但 `-> str` 返回注解会被类型检查器拒（`return v` 是 LatentImage 分支）→ 返回不写注解。
 `is_string` 已内置（comfycli_ffi.scm:72），无需新 builtin
- KSampler/KSamplerAdvanced/HiResFix 改返回 `LatentImage(w, h, batch, out[1], "")`
 （batch 取自输入 latent；ADetailer 属 IMAGE 保持字符串）→ latent 链类型统一
- 全部 IMAGE 消费点经 `image_path_of` 解包：SaveImage/PreviewImage/VAEEncode×2/
  LoadImageMask/ControlNetApply×2/InpaintModelConditioning/9 个 Image 操作/
  ImageToMask/CLIPVisionEncode
- SaveImage 真落盘：新 C API `sd_copy_file`（fs::copy_file，二进制安全；
  prelude 仅有文本 I/O 不可拷 PNG）→ `output_dir/filename_prefix.png` 终于生效
 （此前恒被 KSampler 的 comfy.png 顶替，SaveImage 的 prefix 从未用上）
- LatentUpscale/LatentUpscaleBy → `sd_resize_image` 真缩放；LatentCrop → `sd_crop_image`
 真裁剪（此前全部只改元数据 = no-op）
- e2e（全 rc=0）：T1 p0test.png 512²（prefix 生效）；T2 采样→LatentUpscale→VAEDecode→SaveImage
 出 cchain.png 768²；T3 pcrop 256²（真裁）+ pscale 300×200（LoadImage→ImageScale 混合链，双 SaveImage）；
 回归 wf_alias（euler_ancestral+normal 零告警 + EasyCache enabled）
- 遗留 #1 关闭

未修（遗留，按优先级）：
1. KSampler 返回路径字符串当 LATENT → latent 链（采样→LatentUpscale）崩溃；需 LatentImage 化统一类型（架构级）
2. **`build/img_hires` 二进制含源码/git 都没有的 `--ipadapter/--ref-image/--llm-vision`**（未提交的工作区产物，源码已丢）
   → 重编 examples 前必须先回源（strings 已取回完整 usage 规格），本轮未重编、无破坏
3. backup.sh 14 个未接线 flag（--face-restore*/--photomaker*/...）：开对应 env 即 exit 1
4. pipeline 状态跨节点不清：init image/mask/controlnet 泄漏到后续 KSampler（apply_latent_extras 只 set 不 reset）
5. 14 处 `/tmp` 固定中间文件名互覆盖；LoadImage 不解析 input 目录（folder_paths 零引用）
6. ModelMergeBlocks 等 14 个不在 NODE_GROUP → 静默中止；collect_block_ratios 会吃 output_dir
7. **速度最大项**：comfycli/workflow 路线无 EasyCache setter（cache_mode 恒 0，慢 ~2.2×）
8. cache-start 对 hires 恒"全开"（前几步结构步无保护）；cfg=1 静默丢负向词+FreSca；cfg=0 被改 7.0
9. sampler/scheduler 名无映射+静默回落（ComfyUI 名 → sd.cpp 名表见 stable-diffusion.cpp:80-102/123-141）
10. SDXL clip_skip 默认偏差（conditioner.hpp:458 默认 skip1→2 vs ComfyUI 最后层，无模型待验）；backup_qwen.sh 脚本不一致
