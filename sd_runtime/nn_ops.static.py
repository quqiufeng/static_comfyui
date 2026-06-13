# nn_ops.static.py — 神经网络算子（torch 加速版 + 手写回退）
# 优先使用 torch.*（libtorch 加速），手写 im2col+dgemm 作为回退

# ═════════════════════════════════════════════════════
# Torch 版算子（优先使用）
# ═════════════════════════════════════════════════════

def conv2d_torch(x, w, b, n, c_in, c_out, hh, ww, ks, sd, pd):
    """Conv2d via libtorch"""
    _x_t: ptr = st_from_blob_4d(x, n, c_in, hh, ww)
    _w_t: ptr = st_from_blob_4d(w, c_out, c_in, ks, ks)
    _b_t: ptr = 0
    if b: _b_t = st_from_blob_1d(b, c_out)
    _y_t: ptr = st_conv2d(_x_t, _w_t, _b_t, sd, sd, pd, pd, 1, 1, 1)
    _n: int = n * c_out * ((hh+2*pd-ks)//sd+1) * ((ww+2*pd-ks)//sd+1)
    _r: list[float] = make_float_array(_n)
    _d: ptr = st_tensor_data(_y_t)
    _i: int = 0
    while _i < _n:
        float_array_set(_r, _i, float_array_ref(_d, _i))
        _i = _i + 1
    st_tensor_free(_x_t); st_tensor_free(_w_t)
    if _b_t: st_tensor_free(_b_t)
    st_tensor_free(_y_t)
    return _r

def group_norm_torch(x, w, b, n_groups, c, spatial):
    """GroupNorm via libtorch"""
    _t: ptr = st_from_blob_3d(x, 1, c, spatial)
    _w_t: ptr = st_from_blob_1d(w, c)
    _b_t: ptr = st_from_blob_1d(b, c)
    _r_t: ptr = st_group_norm(_t, n_groups, _w_t, _b_t, 1e-6)
    _d: ptr = st_tensor_data(_r_t)
    _i: int = 0; _n: int = c * spatial
    while _i < _n:
        float_array_set(x, _i, float_array_ref(_d, _i))
        _i = _i + 1
    st_tensor_free(_t); st_tensor_free(_w_t); st_tensor_free(_b_t); st_tensor_free(_r_t)

def linear_torch(x, w, b, n, in_d, out_d):
    """Linear via libtorch: x[n,in_d] → y[n,out_d] = x@w^T + b"""
    _x_t: ptr = st_from_blob_2d(x, n, in_d)
    _w_t: ptr = st_from_blob_2d(w, out_d, in_d)
    _b_t: ptr = st_from_blob_1d(b, out_d)
    _y_t: ptr = st_linear(_x_t, _w_t, _b_t)
    _d: ptr = st_tensor_data(_y_t)
    _r: list[float] = make_float_array(n * out_d)
    _i: int = 0; _n: int = n * out_d
    while _i < _n:
        float_array_set(_r, _i, float_array_ref(_d, _i))
        _i = _i + 1
    st_tensor_free(_x_t); st_tensor_free(_w_t); st_tensor_free(_b_t); st_tensor_free(_y_t)
    return _r
    """SiLU via libtorch"""
    _t: ptr = st_from_blob(x, _mkdims(1, 1, n), 3)
    _r_t: ptr = st_silu(_t)
    _d: ptr = st_tensor_data(_r_t)
    _i: int = 0
    while _i < n:
        float_array_set(x, _i, float_array_ref(_d, _i))
        _i = _i + 1
    st_tensor_free(_t); st_tensor_free(_r_t)

def softmax_torch(x, n, dim):
    """Softmax via libtorch"""
    _t: ptr = st_from_blob(x, _mkdims(1, 1, n), 3)
    _r_t: ptr = st_softmax(_t, dim)
    _d: ptr = st_tensor_data(_r_t)
    _i: int = 0
    while _i < n:
        float_array_set(x, _i, float_array_ref(_d, _i))
        _i = _i + 1
    st_tensor_free(_t); st_tensor_free(_r_t)

def conv2d_inline(src, w, b, n, c_in, c_out, hh, ww):
    """Conv2d 默认 k=3,s=1,p=1，使用 torch 加速"""
    return conv2d_torch(src, w, b, n, c_in, c_out, hh, ww, 3, 1, 1)

# ═════════════════════════════════════════════════════
# 以下为手写实现（保留作为回退/参考）
# ═════════════════════════════════════════════════════
