# Attention 对齐方案

> 目标：让 UNet 的 attention 与 ComfyUI 源码 `openaimodel.py` 中的 manual attention（q @ k^T / scale / softmax / attn @ v）逐张量对齐。  
> 当前状态：代码已写好，等待 GPU 空闲后运行验证。

## 两种 attention 实现

### 1. SDPA（显存友好，数值可能不对齐）

文件：`sd_runtime/transformer_ops.static.py` 中的 `attention_torch`

调用 PyTorch `torch.nn.functional.scaled_dot_product_attention`，可能使用 FlashAttention / Memory-Efficient Attention 等后端。  
优点：显存峰值低。  
缺点：数值路径与 ComfyUI 源码不完全一致，且不同 PyTorch 版本/CUDA 版本可能有差异。

### 2. Manual Attention（源码对齐）

文件：`sd_runtime/transformer_ops.static.py` 中的 `attention_torch_manual`

完全按照 ComfyUI 源码逻辑：

```
q,k,v = linear(x)
q = q.reshape(batch, tokens, heads, dim_head).permute(0,2,1,3)
k = k.reshape(batch, tokens, heads, dim_head).permute(0,2,1,3)
v = v.reshape(batch, tokens, heads, dim_head).permute(0,2,1,3)
sim = q @ k.transpose(-1,-2) / sqrt(dim_head)
attn = softmax(sim, dim=-1)
out = attn @ v
out = out.permute(0,2,1,3).reshape(batch*tokens, dim)
```

优点：与 ComfyUI 源码 1:1，容易对齐。  
缺点：激活峰值高（`sim` 和 `attn` 矩阵额外占用显存）。

## 生成器切换

`gen_unet.py` 支持 `--attention {manual,sdpa}` 参数：

```bash
# 默认生成 manual 版本（用于对齐）
python3 gen_unet.py /tmp/sdxl_unet_merged_f32/index.json manual > sd_runtime/unet_forward.static.py

# 生成 SDPA 版本（用于低显存推理）
python3 gen_unet.py /tmp/sdxl_unet_merged_f32/index.json sdpa > sd_runtime/unet_forward.static.py
```

便捷脚本：

```bash
bash scripts/gen_unet_manual.sh
bash scripts/gen_unet_sdpa.sh
```

`gen_unet.py` 在 manual 模式下会在生成的 `unet_forward.static.py` 里加入一个别名：

```python
# attention mode: manual (ComfyUI source-aligned q@k^T / softmax / attn@v)
def attention_torch(q: ptr, k: ptr, v: ptr, batch: int, tokens_q: int, tokens_k: int, dim: int, heads: int) -> ptr:
    return attention_torch_manual(q, k, v, batch, tokens_q, tokens_k, dim, heads)
```

这样 `spatial_transformer_block` 内部无需修改。

## 验证方法

### 1. manual vs SDPA 输出差异

运行：

```bash
bash scripts/run_attention_compare.sh
```

该脚本会：
1. 生成 manual 版本 UNet，编译运行，保存输出到 `/tmp/unet_output_manual.bin`
2. 生成 SDPA 版本 UNet，编译运行，保存输出到 `/tmp/unet_output_sdpa.bin`
3. 用 numpy 计算 `max_abs_diff` 和 `mean_abs_diff`

预期输出示例：

```
shape: (16384,) (16384,)
manual sum: 983.5
sdpa sum: 983.5
max abs diff: 1.23e-04
mean abs diff: 3.45e-06
```

### 2. manual vs ComfyUI reference

下一步需要写一个 Python reference 脚本，直接调用 `comfyui_ref` 里的 `UNetModel.forward`，和 manual 版本的 ELF 输出对比。

```bash
/data/venv/bin/python3 scripts/reference_unet.py \
    --weights /tmp/sdxl_unet_merged_f32/weights.bin \
    --output /tmp/unet_output_ref.bin

# 对比
/data/venv/bin/python3 scripts/compare_bin.py \
    /tmp/unet_output_ref.bin /tmp/unet_output_manual.bin
```

这些脚本还没写，等 attention 差异验证完后再补充。

## 待验证清单

- [ ] `scripts/run_attention_compare.sh` 能完整跑完 manual + SDPA 两遍
- [ ] manual 与 SDPA 的 `max_abs_diff` 在合理范围（fp16 下 < 1e-3 可接受）
- [ ] manual 版本 UNet 单 forward 在真实/ dummy ctx,y 下不崩溃
- [ ] manual 版本 UNet 输出与 ComfyUI reference 的 `max_diff < 1e-3`
- [ ] 若 manual 版本 OOM，记录具体显存峰值并考虑 activation checkpointing

## 阻塞

当前 GPU 正在跑训练，无法运行需要 CUDA 的对比测试。等 GPU 空闲后再执行 `scripts/run_attention_compare.sh`。

## 备注

- `sd_runtime/transformer_ops.static.py` 里的 `attention_torch_masked` 是 CLIP 用的 masked attention， unaffected，保持 manual。
- 当前仓库默认 `sd_runtime/unet_forward.static.py` 已经是 manual 版本（由 `scripts/gen_unet_manual.sh` 生成）。
