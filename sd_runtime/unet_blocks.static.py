# unet_blocks.static.py — SD Runtime UNet 块
# Phase 3: ResBlock, SpatialTransformer, 上/下采样块
#
# 依赖 nn_ops (conv2d, attention_sd) + array_ops

# ─── 时间步嵌入 ─────────────────────────────────────

def timestep_embedding(t: float, dim: int, max_period: float) -> list[float]:
    """Sinusoidal 时间步嵌入，与 DDPM 一致"""
    emb: list[float] = make_float_array(dim)
    half: int = dim // 2
    i: int = 0
    while i < half:
        freq: float = exp(i * (-log(max_period) / half))
        float_array_set(emb, i, sin(t * freq))
        float_array_set(emb, half + i, cos(t * freq))
        i = i + 1
    if dim % 2 == 1:
        float_array_set(emb, dim - 1, 0.0)
    return emb

def timestep_embedding_batch(ts: list[float], dim: int, n: int, max_period: float) -> list[float]:
    """批量时间步嵌入: [n, dim]"""
    emb: list[float] = make_float_array(n * dim)
    i: int = 0
    while i < n:
        single: list[float] = timestep_embedding(float_array_ref(ts, i), dim, max_period)
        j: int = 0
        while j < dim:
            float_array_set(emb, i * dim + j, float_array_ref(single, j))
            j = j + 1
        i = i + 1
    return emb

# ─── ResBlock ─────────────────────────────────────────

def resblock_forward(x: list[float], emb: list[float],
                     w_in_gn1: list[float], b_in_gn1: list[float],
                     w_in_conv1: list[float], b_in_conv1: list[float],
                     w_emb_linear: list[float], b_emb_linear: list[float],
                     w_out_gn: list[float], b_out_gn: list[float],
                     w_out_conv: list[float], b_out_conv: list[float],
                     w_skip: list[float], b_skip: list[float],
                     n: int, c_in: int, c_out: int, h: int, w: int,
                     emb_dim: int, use_skip: int) -> list[float]:
    """ResBlock 前向:
    x → GN → SiLU → Conv2d → + emb_proj → GN → SiLU → Conv2d → + skip(x)
    emb → SiLU → Linear → (scale, shift) → 注入
    """
    # 输入: x[n, c_in, h, w]
    # 1. GN → SiLU → Conv2d (c_in → c_out)
    h1: list[float] = make_float_array(n * c_in * h * w)
    arr_copy(h1, x, n * c_in * h * w)
    group_norm(h1, w_in_gn1, b_in_gn1, 32, c_in, h * w)
    arr_silu(h1, h1, n * c_in * h * w)

    # 如果 c_in != c_out 或 kernel > 1, conv2d 会改变通道数
    h1 = conv2d(h1, w_in_conv1, b_in_conv1, n, c_in, c_out, h, w, 3, 1, 1)

    # 2. 时间步嵌入投影: SiLU → Linear → (scale, shift)
    emb_proj: list[float] = make_float_array(emb_dim)
    arr_copy(emb_proj, emb, emb_dim)
    arr_silu(emb_proj, emb_proj, emb_dim)

    # Linear: emb_proj[1, emb_dim] @ w_emb_linear[emb_dim, 2*c_out]
    emb_out: list[float] = make_float_array(2 * c_out)
    dgemm_row_auto(1, 2 * c_out, emb_dim, 1.0, emb_proj, w_emb_linear, 0.0, emb_out)
    i: int = 0
    while i < 2 * c_out:
        float_array_set(emb_out, i, float_array_ref(emb_out, i) + float_array_ref(b_emb_linear, i))
        i = i + 1

    scale: list[float] = make_float_array(c_out)
    shift: list[float] = make_float_array(c_out)
    i = 0
    while i < c_out:
        float_array_set(scale, i, float_array_ref(emb_out, i))
        float_array_set(shift, i, float_array_ref(emb_out, c_out + i))
        i = i + 1

    # 3. GN → SiLU → scale/shift → Conv2d
    total: int = n * c_out * h * w
    h2: list[float] = make_float_array(total)
    arr_copy(h2, h1, total)
    group_norm(h2, w_out_gn, b_out_gn, 32, c_out, h * w)

    # 应用 scale 和 shift（per-channel）
    batch: int = 0
    while batch < n:
        ch: int = 0
        while ch < c_out:
            s: float = float_array_ref(scale, ch)
            sh: float = float_array_ref(shift, ch)
            i = 0
            while i < h * w:
                idx: int = ((batch * c_out + ch) * h * w) + i
                v: float = float_array_ref(h2, idx)
                float_array_set(h2, idx, v * s + sh)
                i = i + 1
            ch = ch + 1
        batch = batch + 1

    arr_silu(h2, h2, total)
    h2 = conv2d(h2, w_out_conv, b_out_conv, n, c_out, c_out, h, w, 3, 1, 1)

    # 4. Skip connection
    if use_skip:
        skip: list[float] = conv2d(x, w_skip, b_skip, n, c_in, c_out, h, w, 1, 1, 0)
        i = 0
        while i < total:
            float_array_set(h2, i, float_array_ref(h2, i) + float_array_ref(skip, i))
            i = i + 1

    return h2

# ─── SpatialTransformer ──────────────────────────────

def spatial_transformer_forward(x: list[float], context: list[float],
                                 w_gn: list[float], b_gn: list[float],
                                 w_proj_in: list[float], b_proj_in: list[float],
                                 w_proj_out: list[float], b_proj_out: list[float],
                                 attn_params,
                                 n: int, c: int, h: int, w_in: int,
                                 heads: int, dim_head: int, has_context: int) -> list[float]:
    """SpatialTransformer:
    x → GN → Conv2d(1x1, c → inner_dim) → reshape [b, t, d] → Attention × depth → reshape back → Conv2d(1x1) + skip
    """
    inner_dim: int = heads * dim_head
    total: int = n * c * h * w_in

    # GN
    h_norm: list[float] = make_float_array(total)
    arr_copy(h_norm, x, total)
    group_norm(h_norm, w_gn, b_gn, 32, c, h * w_in)

    # proj_in: Conv2d 1x1, c → inner_dim
    h_proj: list[float] = conv2d(h_norm, w_proj_in, b_proj_in, n, c, inner_dim, h, w_in, 1, 1, 0)

    # reshape [n, inner_dim, h, w] → [n, h*w, inner_dim] (通道后移)
    tokens: int = h * w_in
    h_flat: list[float] = make_float_array(n * tokens * inner_dim)
    batch: int = 0
    while batch < n:
        ch: int = 0
        while ch < inner_dim:
            i: int = 0
            while i < tokens:
                src: int = ((batch * inner_dim + ch) * h) * w_in + i
                dst: int = (batch * tokens + i) * inner_dim + ch
                float_array_set(h_flat, dst, float_array_ref(h_proj, src))
                i = i + 1
            ch = ch + 1
        batch = batch + 1

    # Self-attention + Cross-attention
    if has_context and context is not None:
        # Cross-attention: Q 来自 h_flat, K/V 来自 context
        out: list[float] = cross_attention(h_flat, context, context,
                                           n, tokens, tokens, inner_dim, heads)
    else:
        # Self-attention: Q=K=V 来自 h_flat
        out = attention_sd(h_flat, h_flat, h_flat,
                          n, tokens, tokens, inner_dim, heads)

    # reshape back [n, tokens, inner_dim] → [n, inner_dim, h, w]
    h_back: list[float] = make_float_array(n * inner_dim * h * w_in)
    batch = 0
    while batch < n:
        ch = 0
        while ch < inner_dim:
            i = 0
            while i < tokens:
                src: int = (batch * tokens + i) * inner_dim + ch
                dst: int = ((batch * inner_dim + ch) * h) * w_in + i
                float_array_set(h_back, dst, float_array_ref(out, src))
                i = i + 1
            ch = ch + 1
        batch = batch + 1

    # proj_out: Conv2d 1x1, inner_dim → c
    out_final: list[float] = conv2d(h_back, w_proj_out, b_proj_out, n, inner_dim, c, h, w_in, 1, 1, 0)

    # + skip
    i = 0
    while i < total:
        float_array_set(out_final, i, float_array_ref(out_final, i) + float_array_ref(x, i))
        i = i + 1

    return out_final

# ─── 下采样块 ────────────────────────────────────────

def down_block_forward(x: list[float], emb: list[float], context: list[float],
                       resnet_params, transformer_params,
                       downsample_params,
                       n: int, c_in: int, c_out: int, h: int, w: int,
                       emb_dim: int, has_attn: int, has_downsample: int) -> list[float]:
    """下采样块: ResBlock × N → [Attention] → Downsample"""
    h_current: list[float] = make_float_array(n * c_in * h * w)
    arr_copy(h_current, x, n * c_in * h * w)

    # ResBlock
    h_current = resblock_forward(h_current, emb,
                                 resnet_params[0], resnet_params[1],
                                 resnet_params[2], resnet_params[3],
                                 resnet_params[4], resnet_params[5],
                                 resnet_params[6], resnet_params[7],
                                 resnet_params[8], resnet_params[9],
                                 resnet_params[10], resnet_params[11],
                                 n, c_in, c_out, h, w, emb_dim, 1 if c_in != c_out else 0)

    # Attention
    if has_attn:
        h_current = spatial_transformer_forward(h_current, context,
                                               transformer_params[0], transformer_params[1],
                                               transformer_params[2], transformer_params[3],
                                               transformer_params[4], transformer_params[5],
                                               None, n, c_out, h, w, 8, 64, 1)

    # Downsample
    if has_downsample:
        h_out: int = h // 2
        w_out: int = w // 2
        h_current = conv2d(h_current, downsample_params[0], downsample_params[1],
                          n, c_out, c_out, h, w, 3, 2, 1)

    return h_current

def up_block_forward(x: list[float], skip: list[float], emb: list[float], context: list[float],
                     resnet_params, transformer_params,
                     upsample_params,
                     n: int, c_in: int, c_out: int, h: int, w: int,
                     emb_dim: int, has_attn: int, has_upsample: int) -> list[float]:
    """上采样块: Upsample → ResBlock × N → [Attention]"""
    h_current: list[float] = make_float_array(n * c_in * h * w)
    arr_copy(h_current, x, n * c_in * h * w)

    # Upsample
    if has_upsample:
        h_current = upsample_nearest(h_current, n, c_in, h, w, 2)
        h = h * 2
        w = w * 2
        h_current = conv2d(h_current, upsample_params[0], upsample_params[1],
                          n, c_in, c_in, h, w, 3, 1, 1)

    # 拼接 skip 连接
    total_skip: int = n * c_out * h * w
    total_concat: int = total_skip * 2
    concat: list[float] = make_float_array(total_concat)
    i: int = 0
    while i < total_skip:
        float_array_set(concat, i, float_array_ref(h_current, i))
        float_array_set(concat, total_skip + i, float_array_ref(skip, i))
        i = i + 1

    # ResBlock（这里简化：只用一个 ResBlock）
    h_current = resblock_forward(concat, emb,
                                resnet_params[0], resnet_params[1],
                                resnet_params[2], resnet_params[3],
                                resnet_params[4], resnet_params[5],
                                resnet_params[6], resnet_params[7],
                                resnet_params[8], resnet_params[9],
                                resnet_params[10], resnet_params[11],
                                n, c_out * 2, c_out, h, w, emb_dim, 1)

    # Attention
    if has_attn:
        h_current = spatial_transformer_forward(h_current, context,
                                               transformer_params[0], transformer_params[1],
                                               transformer_params[2], transformer_params[3],
                                               transformer_params[4], transformer_params[5],
                                               None, n, c_out, h, w, 8, 64, 1)

    return h_current
