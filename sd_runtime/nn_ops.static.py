# nn_ops.static.py — SD Runtime 神经网络算子
# Phase 2: Conv2d + Attention + 上/下采样
#
# 所有算子基于 DGEMM + 逐元素运算
# 依赖 dgemm_row_auto (GPU/CPU 自动切换) + array_ops.static.py

# ─── 辅助：张量操作 ───────────────────────────────────

def reshape_2d(src: list[float], dst: list[float], n: int):
    """flatten copy"""
    i: int = 0
    while i < n:
        float_array_set(dst, i, float_array_ref(src, i))
        i = i + 1

def transpose_2d(src: list[float], dst: list[float], rows: int, cols: int):
    """矩阵转置: src[rows][cols] → dst[cols][rows]"""
    i: int = 0
    while i < rows:
        j: int = 0
        while j < cols:
            float_array_set(dst, j * rows + i, float_array_ref(src, i * cols + j))
            j = j + 1
        i = i + 1

# ─── Conv2d ────────────────────────────────────────────

def im2col(x: list[float], n: int, c: int, h: int, w: int,
           k_size: int, stride: int, pad: int,
           col: list[float]):
    """图像→列矩阵: col[n * h_out * w_out, c * k_size * k_size]
    x: [n, c, h, w] 展平为 [n*c*h*w]
    """
    h_out: int = (h + 2 * pad - k_size) // stride + 1
    w_out: int = (w + 2 * pad - k_size) // stride + 1
    col_idx: int = 0
    # 对每个 batch
    batch: int = 0
    while batch < n:
        # 对每个输出位置
        hi: int = 0
        while hi < h_out:
            wi: int = 0
            while wi < w_out:
                # 输入左上角
                h_start: int = hi * stride - pad
                w_start: int = wi * stride - pad
                # 对每个通道
                ch: int = 0
                while ch < c:
                    # 对每个卷积核位置
                    kh: int = 0
                    while kh < k_size:
                        kw: int = 0
                        while kw < k_size:
                            ih: int = h_start + kh
                            iw: int = w_start + kw
                            val: float = 0.0
                            if ih >= 0 and ih < h and iw >= 0 and iw < w:
                                idx: int = ((batch * c + ch) * h + ih) * w + iw
                                val = float_array_ref(x, idx)
                            float_array_set(col, col_idx, val)
                            col_idx = col_idx + 1
                            kw = kw + 1
                        kh = kh + 1
                    ch = ch + 1
                wi = wi + 1
            hi = hi + 1
        batch = batch + 1

def conv2d(x, w, b, n, c_in, c_out, h, w_in, k_size, stride, pad):
    _h_out: int = (h + 2*pad - k_size)//stride + 1
    _w_out: int = (w_in + 2*pad - k_size)//stride + 1
    _ncol: int = n * _h_out * _w_out
    _kdim: int = c_in * k_size * k_size
    _col: list[float] = make_float_array(_ncol * _kdim)
    im2col(x, n, c_in, h, w_in, k_size, stride, pad, _col)
    _y: list[float] = make_float_array(_ncol * c_out)
    dgemm_row_auto(_ncol, c_out, _kdim, 1.0, _col, w, 0.0, _y)
    _i: int = 0
    while _i < _ncol:
        _j: int = 0
        while _j < c_out:
            float_array_set(_y, _i*c_out+_j, float_array_ref(_y, _i*c_out+_j) + float_array_ref(b, _j))
            _j = _j + 1
        _i = _i + 1
    return _y

def conv2d_transposed(x: list[float], w: list[float], b: list[float],
                      n: int, c_in: int, c_out: int, h: int, w_in: int,
                      k_size: int, stride: int, pad: int,
                      out_h: int, out_w: int) -> list[float]:
    """转置卷积: x[n,c_in,h,w] → y[n,c_out,out_h,out_w]
    通过 DGEMM + col2im 实现
    """
    n_cols: int = n * h * w
    k_dim: int = c_in * k_size * k_size

    # y = x @ w^T (转置卷积的权重转置)
    w_t: list[float] = make_float_array(k_dim * c_out)
    transpose_2d(w, w_t, k_dim, c_out)

    y_col: list[float] = make_float_array(n_cols * k_dim)
    dgemm_row_auto(n_cols, k_dim, c_out, 1.0, x, w_t, 0.0, y_col)

    # col2im: 将列矩阵折叠回输出张量
    y: list[float] = make_float_array(n * c_out * out_h * out_w)
    arr_fill(y, 0.0, n * c_out * out_h * out_w)

    col_idx: int = 0
    batch: int = 0
    while batch < n:
        hi: int = 0
        while hi < h:
            wi: int = 0
            while wi < w_in:
                h_start: int = hi * stride - pad
                w_start: int = wi * stride - pad
                ch: int = 0
                while ch < c_in:
                    kh: int = 0
                    while kh < k_size:
                        kw: int = 0
                        while kw < k_size:
                            oh: int = h_start + kh
                            ow: int = w_start + kw
                            if oh >= 0 and oh < out_h and ow >= 0 and ow < out_w:
                                # y[batch, ch, oh, ow] += y_col[col_idx, ch * k_size * k_size + kh * k_size + kw]
                                src_val: float = float_array_ref(y_col, col_idx)
                                dst_idx: int = ((batch * c_in + ch) * out_h + oh) * out_w + ow
                                float_array_set(y, dst_idx, float_array_ref(y, dst_idx) + src_val)
                            col_idx = col_idx + 1
                            kw = kw + 1
                        kh = kh + 1
                    ch = ch + 1
                wi = wi + 1
            hi = hi + 1
        batch = batch + 1

    # 加偏置
    total: int = n * c_out * out_h * out_w
    i = 0
    while i < total:
        co: int = (i // (out_h * out_w)) % c_out
        float_array_set(y, i, float_array_ref(y, i) + float_array_ref(b, co))
        i = i + 1
    return y

# ─── 采样/插值 ──────────────────────────────────────

def upsample_nearest(x: list[float], n: int, c: int, h: int, w: int, scale: int) -> list[float]:
    """最近邻上采样: scale 倍"""
    oh: int = h * scale
    ow: int = w * scale
    y: list[float] = make_float_array(n * c * oh * ow)
    batch: int = 0
    while batch < n:
        ch: int = 0
        while ch < c:
            hi: int = 0
            while hi < oh:
                src_h: int = hi // scale
                wi: int = 0
                while wi < ow:
                    src_w: int = wi // scale
                    src_idx: int = ((batch * c + ch) * h + src_h) * w + src_w
                    dst_idx: int = ((batch * c + ch) * oh + hi) * ow + wi
                    float_array_set(y, dst_idx, float_array_ref(x, src_idx))
                    wi = wi + 1
                hi = hi + 1
            ch = ch + 1
        batch = batch + 1
    return y

def downsample_maxpool(x: list[float], n: int, c: int, h: int, w: int, k_size: int, stride: int) -> list[float]:
    """MaxPool 下采样"""
    oh: int = (h - k_size) // stride + 1
    ow: int = (w - k_size) // stride + 1
    y: list[float] = make_float_array(n * c * oh * ow)
    batch: int = 0
    while batch < n:
        ch: int = 0
        while ch < c:
            hi: int = 0
            while hi < oh:
                wi: int = 0
                while wi < ow:
                    max_v: float = -1e10
                    kh: int = 0
                    while kh < k_size:
                        kw: int = 0
                        while kw < k_size:
                            ih: int = hi * stride + kh
                            iw: int = wi * stride + kw
                            idx: int = ((batch * c + ch) * h + ih) * w + iw
                            v: float = float_array_ref(x, idx)
                            if v > max_v: max_v = v
                            kw = kw + 1
                        kh = kh + 1
                    dst_idx: int = ((batch * c + ch) * oh + hi) * ow + wi
                    float_array_set(y, dst_idx, max_v)
                    wi = wi + 1
                hi = hi + 1
            ch = ch + 1
        batch = batch + 1
    return y

def downsample_conv(x, w, b, n, c, h, w_in, c_out, k_size, stride, pad):
    """Conv2d stride=2 下采样"""
    return conv2d(x, w, b, n, c, c_out, h, w_in, k_size, stride, pad)

# ─── Attention ────────────────────────────────────────

def attention_sd(q: list[float], k: list[float], v: list[float],
                 batch: int, tokens_q: int, tokens_k: int, dim: int, heads: int):
    """Scaled Dot-Product Attention
    Q[batch, tokens_q, dim], K[batch, tokens_k, dim], V[batch, tokens_k, dim]
    输出: [batch, tokens_q, dim]

    实现: reshape → Q @ K^T * scale → softmax → @ V → reshape back
    """
    dim_head: int = dim // heads
    scale: float = 1.0 / sqrt(dim_head)
    total_q: int = batch * tokens_q
    total_k: int = batch * tokens_k

    # 1. Reshape Q, K, V from [batch, tokens, dim] to [batch*heads, tokens, dim_head]
    #    由于是展平存储，需要重新排列数据
    q_h: list[float] = make_float_array(batch * heads * tokens_q * dim_head)
    k_h: list[float] = make_float_array(batch * heads * tokens_k * dim_head)
    v_h: list[float] = make_float_array(batch * heads * tokens_k * dim_head)

    b: int = 0
    while b < batch:
        h: int = 0
        while h < heads:
            t: int = 0
            while t < tokens_q:
                d: int = 0
                while d < dim_head:
                    src: int = ((b * tokens_q + t) * dim) + h * dim_head + d
                    dst: int = ((b * heads + h) * tokens_q + t) * dim_head + d
                    float_array_set(q_h, dst, float_array_ref(q, src))
                    d = d + 1
                t = t + 1
            t = 0
            while t < tokens_k:
                d = 0
                while d < dim_head:
                    src_k: int = ((b * tokens_k + t) * dim) + h * dim_head + d
                    src_v: int = src_k
                    dst_k: int = ((b * heads + h) * tokens_k + t) * dim_head + d
                    dst_v: int = ((b * heads + h) * tokens_k + t) * dim_head + d
                    float_array_set(k_h, dst_k, float_array_ref(k, src_k))
                    float_array_set(v_h, dst_v, float_array_ref(v, src_v))
                    d = d + 1
                t = t + 1
            h = h + 1
        b = b + 1

    # 2. Q @ K^T: sim[b*heads*tokens_q, tokens_k] = Q[b*heads*tokens_q, dim_head] @ K^T[dim_head, tokens_k]
    n_sim_rows: int = batch * heads * tokens_q
    sim: list[float] = make_float_array(n_sim_rows * tokens_k)
    dgemm_row_auto(n_sim_rows, tokens_k, dim_head, scale, q_h, k_h, 0.0, sim)

    # 3. softmax 沿 tokens_k 维
    r: int = 0
    while r < n_sim_rows:
        offset: int = r * tokens_k
        # max
        max_v: float = float_array_ref(sim, offset)
        c: int = 1
        while c < tokens_k:
            vv: float = float_array_ref(sim, offset + c)
            if vv > max_v: max_v = vv
            c = c + 1
        # exp + sum
        s: float = 0.0
        c = 0
        while c < tokens_k:
            ev: float = exp(float_array_ref(sim, offset + c) - max_v)
            float_array_set(sim, offset + c, ev)
            s = s + ev
            c = c + 1
        # normalize
        if s > 0.0:
            c = 0
            while c < tokens_k:
                float_array_set(sim, offset + c, float_array_ref(sim, offset + c) / s)
                c = c + 1
        r = r + 1

    # 4. sim @ V: out[n_sim_rows, dim_head] = sim[n_sim_rows, tokens_k] @ V[..., dim_head]
    out_h: list[float] = make_float_array(n_sim_rows * dim_head)
    dgemm_row_auto(n_sim_rows, dim_head, tokens_k, 1.0, sim, v_h, 0.0, out_h)

    # 5. Reshape back: [batch*heads, tokens, dim_head] → [batch, tokens, dim]
    out: list[float] = make_float_array(batch * tokens_q * dim)
    b = 0
    while b < batch:
        h = 0
        while h < heads:
            t = 0
            while t < tokens_q:
                d = 0
                while d < dim_head:
                    src: int = ((b * heads + h) * tokens_q + t) * dim_head + d
                    dst: int = ((b * tokens_q + t) * dim) + h * dim_head + d
                    float_array_set(out, dst, float_array_ref(out_h, src))
                    d = d + 1
                t = t + 1
            h = h + 1
        b = b + 1

    return out

def cross_attention(q: list[float], k: list[float], v: list[float],
                    batch: int, tokens_q: int, tokens_k: int, dim: int, heads: int) -> list[float]:
    """Cross-attention: Q 来自 x, K/V 来自 context"""
    return attention_sd(q, k, v, batch, tokens_q, tokens_k, dim, heads)

def conv2d_inline(src, w, b, n, c_in, c_out, hh, ww):
    "Conv2d wrapper: im2col + dgemm + bias (uses _i,_j locals)"
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = c_in * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(src, n, c_in, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * c_out)
    dgemm_row_auto(_nc, c_out, _kd, 1.0, _col, w, 0.0, _y)
    _i = 0
    while _i < _nc:
        _j = 0
        while _j < c_out:
            float_array_set(_y, _i*c_out+_j, float_array_ref(_y, _i*c_out+_j) + float_array_ref(b, _j))
            _j = _j + 1
        _i = _i + 1
    return _y

def add_bias(arr, bias, n_rows, n_cols):
    _i = 0
    while _i < n_rows:
        _j = 0
        while _j < n_cols:
            float_array_set(arr, _i*n_cols+_j, float_array_ref(arr, _i*n_cols+_j) + float_array_ref(bias, _j))
            _j = _j + 1
        _i = _i + 1

def add_arr(a, b, n):
    _i = 0
    while _i < n:
        float_array_set(a, _i, float_array_ref(a, _i) + float_array_ref(b, _i))
        _i = _i + 1
