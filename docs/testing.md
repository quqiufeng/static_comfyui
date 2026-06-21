# 测试记录与待验证清单

> 记录日期：2026-06-21  
> 当前状态：GPU 正在跑训练，暂无条件运行完整验证。下次 GPU 空闲时按本清单继续。

## 测试文件位置

仓库 `test/` 目录下：

| 文件 | 说明 | 当前状态 |
|------|------|----------|
| `test_twofwd_main.static.py` | UNet 双 forward 烟雾测试（dummy ctx/y） | 已通过 |
| `test_sampler_main.static.py` | 5-step Euler sampler 测试（dummy ctx/y，cfg=7.5） | 2-step 通过，5-step OOM |
| `test_attention_compare.static.py` | UNet 单 forward，保存输出用于 attention 对比 | 未运行 |

便捷脚本：

```bash
# 切换并生成 manual / SDPA 版本的 UNet
bash scripts/gen_unet_manual.sh
bash scripts/gen_unet_sdpa.sh

# 对比 manual vs SDPA 输出（需要 GPU）
bash scripts/run_attention_compare.sh
```

### 1. 构建 libstaticpy_torch.so

```bash
cd /opt/static_comfyui
bash deliver.sh
```

`deliver.sh` 会自动编译并打包出 `sd_generate.elf`，同时生成 `/tmp/libstaticpy_torch.so`。

### 2. 临时完整测试（快速编译）

```bash
# 双 forward UNet
cat sd_runtime/array_ops.static.py sd_runtime/torch_ops.static.py \
    sd_runtime/nn_ops.static.py sd_runtime/transformer_ops.static.py \
    sd_runtime/unet_forward.static.py test/test_twofwd_main.static.py \
    > /tmp/test_twofwd.static.py
bash build.sh /tmp/test_twofwd.static.py test_twofwd
/opt/ChezScheme/ta6le/bin/ta6le/scheme --quiet build_out/test_twofwd.so

# 5-step sampler
cat sd_runtime/array_ops.static.py sd_runtime/torch_ops.static.py \
    sd_runtime/nn_ops.static.py sd_runtime/transformer_ops.static.py \
    sd_runtime/unet_forward.static.py sd_runtime/sampler.static.py \
    test/test_sampler_main.static.py \
    > /tmp/test_sampler.static.py
bash build.sh /tmp/test_sampler.static.py test_sampler
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /opt/ChezScheme/ta6le/bin/ta6le/scheme --quiet build_out/test_sampler.so
```

## 依赖的模型/数据文件

### 原始模型（放在 `/data/models/image/`）

| 文件 | 来源 | 用途 |
|------|------|------|
| `sd_xl_base_1.0.safetensors` | SDXL Base 官方 | UNet 权重 |
| `clip_l.safetensors` | OpenAI CLIP-L | text encoder L |
| `clip_g.safetensors` | OpenAI CLIP-G | text encoder G |
| `ae.safetensors` | SDXL VAE | 解码 latent |

### 导出/合并后的中间文件

测试目前只需要 UNet 权重。导出命令：

```bash
cd /opt/static_comfyui
/data/venv/bin/python3 export_sd_weights.py \
    /data/models/image/sd_xl_base_1.0.safetensors /tmp/sdxl_all_f32
/data/venv/bin/python3 merge_weights.py /tmp/sdxl_all_f32 /tmp/sdxl_unet_merged_f32 model.diffusion_model.
```

产物：

- `/tmp/sdxl_unet_merged_f32/weights.bin`（10.27 GB，2567463684 个 float32）
- `/tmp/sdxl_unet_merged_f32/index.json`

> 注：这些文件太大，不放入仓库。下次验证前按上面命令重新生成即可。

### CLIP/VAE 相关（端到端验证需要）

`sd_runtime/main.static.py` 还会依赖：

- `/tmp/sd_weights/clip_lg_merged_f32/weights.bin`（CLIP-L/G 合并权重）
- `/tmp/tokens_l.bin`（77 个 token id）
- `/tmp/tokens_g.bin`（77 个 token id）

这些当前还没生成，等 CLIP tokenizer 接入后再处理。

## 已完成的改动

- `gen_unet.py` / `sd_runtime/unet_forward.static.py`：UNet 权重只加载一次，`load_unet_weights()` + `weights: ptr` 参数。
- skip connection 保存前 `st_clone()`，避免悬垂指针。
- output block cat 的 `cur_ch/skip_ch` 顺序修复。
- 单块 fp16 GPU 权重 buffer + view，减少 allocator 碎片。
- attention 切换到 PyTorch SDPA，降低激活峰值。
- `file_read_floats()` 直接读取 `.bin` 权重文件。

## 已知问题

1. **5-step Euler sampler 在 64×64 下仍 OOM**
   - 环境空闲显存约 7 GB，但其他训练进程已占 12 GB。
   - 2-step sampler（cfg=1.0）可通过。
   - 5-step 卡在第二步 forward 的 group_norm/linear 分配。

2. **SDPA 改变数值路径**
   - 已不再是手动 `q@k^T / softmax / attn@v`，因此与 ComfyUI reference 的 1:1 对齐需要重新验证。

3. **main.static.py 中的 `y` 是 zeros 占位**
   - 需要实现 SDXL 的 image-dim 正弦位置编码（height/width/crop/target）。

## 下次验证 todo

1. [ ] GPU 空闲时重新跑 `test_twofwd_main.static.py`，确认双 forward 仍通过。
2. [ ] 尝试 5-step sampler 是否能在更大空闲显存下通过；若仍 OOM，考虑 weight streaming 或 CPU offload。
3. [ ] 跑通 `main.static.py`：CLIP encode → UNet forward → 保存输出。
4. [ ] 用真实 prompt 生成 token，替换 dummy tokens。
5. [ ] 实现正确的 `y` 编码。
6. [ ] 数值对齐：将 ELF 输出与 ComfyUI reference 对比 `max_diff`。
7. [ ] VAE decode：将 sampler 输出送入 VAE，得到最终图像。

## Attention 对齐

- 详见 `docs/attention_alignment.md`。
- 当前 `sd_runtime/unet_forward.static.py` 默认是 **manual attention** 版本（源码对齐）。
- 需要 GPU 空闲后运行 `bash scripts/run_attention_compare.sh` 验证 manual vs SDPA 输出差异。

## 显存调参记录

已尝试但未解决 5-step OOM：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:64'
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'
```

结论：当前环境下其他训练进程占用 12 GB，剩余空间不足以支撑 SDXL UNet 多步采样。
