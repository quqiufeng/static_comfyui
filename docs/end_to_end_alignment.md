# 端到端 txt2img 对齐方案

> 目标：让 ELF 版本从 prompt + 尺寸 开始，经过 CLIP encode → UNet forward，输出与 ComfyUI reference 逐张量对齐。  
> 当前状态：代码已写好，等待 GPU 空闲后验证。

## 设计选择

### Tokenizer 不在二进制里实现（第一阶段）

完整 BPE tokenizer 需要 vocab/merges/dict/regex，在 StaticPy 里实现工作量大。  
当前策略：用 Python 的 `transformers.CLIPTokenizer` 预处理 prompt，导出 `tokens_l.bin` 和 `tokens_g.bin`，StaticPy 只负责读取并传入 `clip_encode_lg`。

这和权重文件一样属于"外部输入数据"，不破坏"二进制推理"的核心目标。

### SDXL y conditioning 在 main.static.py 里拼接

ComfyUI 的 `SDXL.encode_adm`：

```python
y = [pooled_g (1280),
     timestep_embedding(height, 256),
     timestep_embedding(width, 256),
     timestep_embedding(crop_h, 256),
     timestep_embedding(crop_w, 256),
     timestep_embedding(target_h, 256),
     timestep_embedding(target_w, 256)]
# total 2816
```

`pooled_g` 来自 CLIP-G encode 的输出；image dim 的正弦编码在 Python 脚本里预计算并保存为 `image_dim.bin`；StaticPy 的 `main.static.py` 把两者 cat 起来再转 fp16。

## 运行步骤

### 1. 预处理 prompt

```bash
python3 scripts/preprocess_prompt.py "a photo of a cat" \
    --width 1024 --height 1024 \
    --crop-w 0 --crop-h 0 \
    --target-width 1024 --target-height 1024
```

产物：
- `/tmp/tokens_l.bin` (77 int32)
- `/tmp/tokens_g.bin` (77 int32)
- `/tmp/image_dim.bin` (1536 float32)

### 2. 编译运行 ELF

```bash
bash deliver.sh
./sd_generate.elf
```

`main.static.py` 会：
1. 加载 CLIP-L/G 权重
2. 读取 tokens，CLIP encode 得到 `ctx` 和 `pooled_g`
3. 读取 image_dim，与 pooled_g 拼接成 `y`
4. 加载 UNet 权重，forward 得到输出
5. 保存 `/tmp/elf_ctx.bin`, `/tmp/elf_pooled_g.bin`, `/tmp/elf_y.bin`, `/tmp/elf_unet_output.bin`

### 3. 生成 ComfyUI reference

```bash
python3 scripts/reference_unet.py \
    --model /data/models/image/sd_xl_base_1.0.safetensors \
    --tokens-l /tmp/tokens_l.bin \
    --tokens-g /tmp/tokens_g.bin \
    --image-dim /tmp/image_dim.bin \
    --out-dir /tmp/ref_out
```

产物：
- `/tmp/ref_out/ctx.bin`
- `/tmp/ref_out/pooled_g.bin`
- `/tmp/ref_out/y.bin`
- `/tmp/ref_out/unet_output.bin`

### 4. 对比

```bash
python3 scripts/compare_bin.py /tmp/ref_out/ctx.bin /tmp/elf_ctx.bin --shape 1,77,2048
python3 scripts/compare_bin.py /tmp/ref_out/pooled_g.bin /tmp/elf_pooled_g.bin --shape 1,1280
python3 scripts/compare_bin.py /tmp/ref_out/y.bin /tmp/elf_y.bin --shape 1,2816
python3 scripts/compare_bin.py /tmp/ref_out/unet_output.bin /tmp/elf_unet_output.bin --shape 1,4,64,64
```

验收标准：`max abs diff < 1e-3`（fp16 推理可放宽到 `5e-3`）。

## 待验证清单

- [ ] `scripts/preprocess_prompt.py` 能正确生成 tokens（可用 `transformers` 单独验证）
- [ ] `image_dim.bin` 的值与 ComfyUI `Timestep(256).forward()` 输出一致
- [ ] `main.static.py` 编译通过，能读取新文件
- [ ] `reference_unet.py` 能加载 ComfyUI checkpoint 并运行
- [ ] ELF 输出的 `ctx` 与 reference 对齐
- [ ] ELF 输出的 `pooled_g` 与 reference 对齐
- [ ] ELF 输出的 `y` 与 reference 对齐
- [ ] ELF 输出的 `unet_output` 与 reference 对齐

## 阻塞

当前 GPU 正在跑训练，无法运行需要 CUDA 的 `deliver.sh`、`reference_unet.py` 和 ELF。

等 GPU 空闲后，按上面步骤一键执行即可。

## 后续：把 tokenizer 搬进二进制

第一阶段先用 Python 预处理。后续如果要完全自包含，可以在 StaticPy 里实现一个简化 BPE tokenizer：

1. 预处理阶段把 `vocab.json` 和 `merges.txt` 转成静态数组或 trie
2. `sd_runtime/tokenizer.static.py` 实现 `tokenize(prompt)`
3. 不再依赖 Python 预处理

这个优先级低于端到端对齐。
