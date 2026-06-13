# unet_forward.static.py — SD UNet forward 通用实现
# 不生成代码，用函数抽象所有 block 类型
# 模型结构由 block_params 定义

# ─── 通用 ResBlock ─────────────────────────────────

def resblock_forward(x, emb, w_norm1, b_norm1, w_conv1, b_conv1,
                      w_norm2, b_norm2, w_conv2, b_conv2,
                      w_emb, b_emb, w_skip, b_skip,
                      n, c_in, c_out, hh, ww):
    """ResBlock: GN→SiLU→Conv→+temb→GN→SiLU→Conv→+skip"""
    group_norm(x, w_norm1, b_norm1, 32, c_in, hh * ww)
    arr_silu(x, x, n * c_in * hh * ww)
    x = conv2d_inline(x, w_conv1, b_conv1, n, c_in, c_out, hh, ww)
    
    # time embedding → scale + shift
    _y: list[float] = make_float_array(n * c_out * 2)
    dgemm_row_auto(n, c_out * 2, 1280, 1.0, emb, w_emb, 0.0, _y)
    add_bias(_y, b_emb, n, c_out * 2)
    apply_scale_shift(x, _y, n, c_out, hh * ww)
    
    group_norm(x, w_norm2, b_norm2, 32, c_out, hh * ww)
    arr_silu(x, x, n * c_out * hh * ww)
    x = conv2d_inline(x, w_conv2, b_conv2, n, c_out, c_out, hh, ww)
    
    if c_in != c_out:
        _skip: list[float] = conv2d_inline(x_orig, w_skip, b_skip, n, c_in, c_out, hh, ww)
        add_arr(x, _skip, n * c_out * hh * ww)
    return x

# ─── 通用 SpatialTransformer ──────────────────────

def transformer_forward(x, context, w_norm, b_norm,
                         w_proj_in, b_proj_in,
                         w_proj_out, b_proj_out,
                         attn_k_weight, attn_k_bias,
                         attn_q_weight, attn_q_bias,
                         attn_v_weight, attn_v_bias,
                         attn_out_weight, attn_out_bias,
                         n, c, hh, ww, d_head):
    """SpatialTransformer: GN → 1x1Conv → reshape → Attention → reshape → 1x1Conv → +x"""
    _inner: int = c  # proj_in keeps channels same normally
    _total: int = n * c * hh * ww
    _src: list[float] = make_float_array(_total)
    arr_copy(_src, x, _total)
    
    group_norm(x, w_norm, b_norm, 32, c, hh * ww)
    x = conv2d_inline(x, w_proj_in, b_proj_in, n, c, _inner, hh, ww)
    
    # reshape [n, c, h, w] → [n, h*w, c]
    _tokens: int = hh * ww
    _h_flat: list[float] = make_float_array(n * _tokens * _inner)
    _b: int = 0
    while _b < n:
        _ch: int = 0
        while _ch < _inner:
            _t: int = 0
            while _t < _tokens:
                float_array_set(_h_flat, (_b * _tokens + _t) * _inner + _ch, float_array_ref(x, ((_b * _inner + _ch) * hh) * ww + _t))
                _t = _t + 1
            _ch = _ch + 1
        _b = _b + 1
    
    # Q = h_flat @ w_q + b_q  (same for K, V)
    _dim: int = _inner
    _q: list[float] = make_float_array(n * _tokens * _dim)
    dgemm_row_auto(n * _tokens, _dim, _dim, 1.0, _h_flat, attn_q_weight, 0.0, _q)
    add_bias(_q, attn_q_bias, n * _tokens, _dim)
    
    _k: list[float] = make_float_array(n * _tokens * _dim)
    dgemm_row_auto(n * _tokens, _dim, _dim, 1.0, _h_flat, attn_k_weight, 0.0, _k)
    add_bias(_k, attn_k_bias, n * _tokens, _dim)
    
    _v: list[float] = make_float_array(n * _tokens * _dim)
    dgemm_row_auto(n * _tokens, _dim, _dim, 1.0, _h_flat, attn_v_weight, 0.0, _v)
    add_bias(_v, attn_v_bias, n * _tokens, _dim)
    
    # Attention
    _out = attention_sd(_q, _k, _v, n, _tokens, _tokens, _dim, _dim // d_head)
    
    # proj_out
    _out2: list[float] = make_float_array(n * _tokens * _dim)
    dgemm_row_auto(n * _tokens, _dim, _dim, 1.0, _out, attn_out_weight, 0.0, _out2)
    add_bias(_out2, attn_out_bias, n * _tokens, _dim)
    
    # reshape back
    _h_back: list[float] = make_float_array(_total)
    _b = 0
    while _b < n:
        _ch = 0
        while _ch < _dim:
            _t = 0
            while _t < _tokens:
                float_array_set(_h_back, ((_b * _dim + _ch) * hh) * ww + _t, float_array_ref(_out2, (_b * _tokens + _t) * _dim + _ch))
                _t = _t + 1
            _ch = _ch + 1
        _b = _b + 1
    
    add_arr(_h_back, _src, _total)
    return _h_back

# ─── 辅助函数 ─────────────────────────────────────

def apply_scale_shift(x, ss, n, c, spatial):
    """ss 的前半是 scale, 后半是 shift: x = x * scale + shift"""
    _b: int = 0
    while _b < n:
        _ch: int = 0
        while _ch < c:
            _s: float = float_array_ref(ss, _ch)
            _sh: float = float_array_ref(ss, c + _ch)
            _i: int = 0
            while _i < spatial:
                _idx: int = (_b * c + _ch) * spatial + _i
                float_array_set(x, _idx, float_array_ref(x, _idx) * _s + _sh)
                _i = _i + 1
            _ch = _ch + 1
        _b = _b + 1

def downsample_conv(x, w, b, n, c, hh, ww):
    """直接 conv2d stride=2"""
    return conv2d_inline(x, w, b, n, c, c, hh, ww)
