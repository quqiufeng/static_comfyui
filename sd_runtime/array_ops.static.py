# array_ops.static.py — SD Runtime 张量运算基础设施
# Phase 1: 逐元素运算 + softmax + 归一化 + 规约
#
# 依赖:
#   extern fn dgemm_row_auto from "dgemm_row" (已在 stdlib)
#   extern fn make_float_array, float_array_set, float_array_ref from prelude
#   extern fn exp, sqrt from libm (已在 prelude)

# ─── 逐元素数学 ─────────────────────────────────────

def arr_fill(a: list[float], v: float, n: int):
    i: int = 0
    while i < n: float_array_set(a, i, v); i = i + 1

def arr_copy(dst: list[float], src: list[float], n: int):
    i: int = 0
    while i < n: float_array_set(dst, i, float_array_ref(src, i)); i = i + 1

def arr_add(a: list[float], b: list[float], r: list[float], n: int):
    i: int = 0
    while i < n:
        float_array_set(r, i, float_array_ref(a, i) + float_array_ref(b, i))
        i = i + 1

def arr_sub(a: list[float], b: list[float], r: list[float], n: int):
    i: int = 0
    while i < n:
        float_array_set(r, i, float_array_ref(a, i) - float_array_ref(b, i))
        i = i + 1

def arr_mul(a: list[float], b: list[float], r: list[float], n: int):
    i: int = 0
    while i < n:
        float_array_set(r, i, float_array_ref(a, i) * float_array_ref(b, i))
        i = i + 1

def arr_mul_scalar(a: list[float], s: float, n: int):
    i: int = 0
    while i < n:
        float_array_set(a, i, float_array_ref(a, i) * s)
        i = i + 1

def arr_add_scalar(a: list[float], s: float, n: int):
    i: int = 0
    while i < n:
        float_array_set(a, i, float_array_ref(a, i) + s)
        i = i + 1

def arr_div(a: list[float], b: list[float], r: list[float], n: int):
    i: int = 0
    while i < n:
        float_array_set(r, i, float_array_ref(a, i) / float_array_ref(b, i))
        i = i + 1

def arr_div_scalar(a: list[float], s: float, n: int):
    i: int = 0
    while i < n:
        float_array_set(a, i, float_array_ref(a, i) / s)
        i = i + 1

def arr_pow(a: list[float], e: float, r: list[float], n: int):
    i: int = 0
    while i < n:
        float_array_set(r, i, pow(float_array_ref(a, i), e))
        i = i + 1

def arr_exp(a: list[float], r: list[float], n: int):
    i: int = 0
    while i < n:
        float_array_set(r, i, exp(float_array_ref(a, i)))
        i = i + 1

def arr_sqrt(a: list[float], r: list[float], n: int):
    i: int = 0
    while i < n:
        float_array_set(r, i, sqrt(float_array_ref(a, i)))
        i = i + 1

def arr_rsqrt(a: list[float], r: list[float], n: int):
    i: int = 0
    while i < n:
        float_array_set(r, i, 1.0 / sqrt(float_array_ref(a, i)))
        i = i + 1

def arr_clip(a: list[float], lo: float, hi: float, n: int):
    i: int = 0
    while i < n:
        v: float = float_array_ref(a, i)
        if v < lo: float_array_set(a, i, lo)
        elif v > hi: float_array_set(a, i, hi)
        i = i + 1

def arr_sigm(a: list[float], r: list[float], n: int):
    i: int = 0
    while i < n:
        x: float = float_array_ref(a, i)
        float_array_set(r, i, 1.0 / (1.0 + exp(-x)))
        i = i + 1

def arr_tanh(a: list[float], r: list[float], n: int):
    i: int = 0
    while i < n:
        x: float = float_array_ref(a, i)
        if x > 20.0: float_array_set(r, i, 1.0)
        elif x < -20.0: float_array_set(r, i, -1.0)
        else:
            e2x: float = exp(2.0 * x)
            float_array_set(r, i, (e2x - 1.0) / (e2x + 1.0))
        i = i + 1

def arr_relu(a: list[float], r: list[float], n: int):
    i: int = 0
    while i < n:
        v: float = float_array_ref(a, i)
        float_array_set(r, i, v if v > 0.0 else 0.0)
        i = i + 1

def arr_silu(a: list[float], r: list[float], n: int):
    """SiLU = x * sigmoid(x)"""
    i: int = 0
    while i < n:
        x: float = float_array_ref(a, i)
        float_array_set(r, i, x / (1.0 + exp(-x)))
        i = i + 1

def arr_gelu(a: list[float], r: list[float], n: int):
    """GELU = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    coeff: float = 0.7978845608028654  # sqrt(2/pi)
    i: int = 0
    while i < n:
        x: float = float_array_ref(a, i)
        x3: float = x * x * x
        inner: float = coeff * (x + 0.044715 * x3)
        if inner > 20.0:
            tanh_v: float = 1.0
        elif inner < -20.0:
            tanh_v: float = -1.0
        else:
            e2x: float = exp(2.0 * inner)
            tanh_v = (e2x - 1.0) / (e2x + 1.0)
        float_array_set(r, i, 0.5 * x * (1.0 + tanh_v))
        i = i + 1

def arr_neg(a: list[float], r: list[float], n: int):
    i: int = 0
    while i < n:
        float_array_set(r, i, -float_array_ref(a, i))
        i = i + 1

# ─── 规约操作 ───────────────────────────────────────

def arr_sum(a: list[float], n: int) -> float:
    s: float = 0.0; i: int = 0
    while i < n: s = s + float_array_ref(a, i); i = i + 1
    return s

def arr_mean(a: list[float], n: int) -> float:
    return arr_sum(a, n) / n

def arr_max(a: list[float], n: int) -> float:
    m: float = float_array_ref(a, 0); i: int = 1
    while i < n:
        v: float = float_array_ref(a, i)
        if v > m: m = v
        i = i + 1
    return m

def arr_min(a: list[float], n: int) -> float:
    m: float = float_array_ref(a, 0); i: int = 1
    while i < n:
        v: float = float_array_ref(a, i)
        if v < m: m = v
        i = i + 1
    return m

# ─── Softmax ─────────────────────────────────────────

def softmax(x: list[float], n: int):
    """就地 softmax 沿最后一维"""
    max_val: float = arr_max(x, n)
    arr_add_scalar(x, -max_val, n)
    arr_exp(x, x, n)
    s: float = arr_sum(x, n)
    if s > 0.0: arr_div_scalar(x, s, n)

def softmax_2d(x: list[float], rows: int, cols: int):
    """softmax 对每行分别做"""
    r: int = 0
    while r < rows:
        offset: int = r * cols
        max_val: float = float_array_ref(x, offset)
        c: int = 1
        while c < cols:
            v: float = float_array_ref(x, offset + c)
            if v > max_val: max_val = v
            c = c + 1
        s: float = 0.0
        c = 0
        while c < cols:
            ev: float = exp(float_array_ref(x, offset + c) - max_val)
            float_array_set(x, offset + c, ev)
            s = s + ev
            c = c + 1
        if s > 0.0:
            c = 0
            while c < cols:
                float_array_set(x, offset + c, float_array_ref(x, offset + c) / s)
                c = c + 1
        r = r + 1

# ─── 归一化 ─────────────────────────────────────────

def layer_norm(x: list[float], gamma: list[float], beta: list[float],
               n_features: int, n_elements: int):
    """LayerNorm: 对每个样本的特征维做归一化
    x: [n_elements] 展平的
    n_features: 特征维度大小
    n_elements / n_features = batch_size * spatial_dims
    """
    step: int = n_features
    n_samples: int = n_elements // n_features
    s: int = 0
    while s < n_samples:
        offset: int = s * step
        # 计算 mean
        mean: float = 0.0
        i: int = 0
        while i < step:
            mean = mean + float_array_ref(x, offset + i)
            i = i + 1
        mean = mean / step
        # 计算 var
        var: float = 0.0
        i = 0
        while i < step:
            diff: float = float_array_ref(x, offset + i) - mean
            var = var + diff * diff
            i = i + 1
        var = var / step
        # 归一化 + affine
        inv_std: float = 1.0 / sqrt(var + 1e-5)
        i = 0
        while i < step:
            norm: float = (float_array_ref(x, offset + i) - mean) * inv_std
            float_array_set(x, offset + i, norm * float_array_ref(gamma, i) + float_array_ref(beta, i))
            i = i + 1
        s = s + 1

def group_norm(x: list[float], gamma: list[float], beta: list[float],
               n_groups: int, n_channels: int, spatial: int):
    """GroupNorm: 对每组 channels 做归一化
    x: [n_channels * spatial] 展平的
    每组大小 = n_channels // n_groups * spatial
    """
    group_size: int = (n_channels // n_groups) * spatial
    g: int = 0
    while g < n_groups:
        offset: int = g * group_size
        # mean
        mean: float = 0.0
        i: int = 0
        while i < group_size:
            mean = mean + float_array_ref(x, offset + i)
            i = i + 1
        mean = mean / group_size
        # var
        var: float = 0.0
        i = 0
        while i < group_size:
            diff: float = float_array_ref(x, offset + i) - mean
            var = var + diff * diff
            i = i + 1
        var = var / group_size
        # normalize
        inv_std: float = 1.0 / sqrt(var + 1e-5)
        i = 0
        while i < group_size:
            ch: int = (offset + i) // spatial
            norm: float = (float_array_ref(x, offset + i) - mean) * inv_std
            float_array_set(x, offset + i, norm * float_array_ref(gamma, ch) + float_array_ref(beta, ch))
            i = i + 1
        g = g + 1
