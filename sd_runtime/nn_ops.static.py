# nn_ops.static.py — 神经网络算子（GPU ptr 版本）
# 所有张量都是 st_* 返回的 GPU torch Tensor 指针（ptr）

def conv2d_torch(x: ptr, w: ptr, b: ptr, n: int, c_in: int, c_out: int, hh: int, ww: int, ks: int, sd: int, pd: int) -> ptr:
    return st_conv2d(x, w, b, sd, sd, pd, pd, 1, 1, 1)

def linear_torch(x: ptr, w: ptr, b: ptr, n: int, in_d: int, out_d: int) -> ptr:
    return st_linear(x, w, b)

def group_norm_torch(x: ptr, w: ptr, b: ptr, n_groups: int, c: int, hh: int, ww: int) -> ptr:
    return st_group_norm(x, n_groups, w, b, 1e-5)

def silu_torch(x: ptr) -> ptr:
    return st_silu(x)

def gelu_torch(x: ptr) -> ptr:
    return st_gelu(x)

def softmax_torch(x: ptr, dim: int) -> ptr:
    return st_softmax(x, dim)

def upsample_nearest_torch(x: ptr, scale: int) -> ptr:
    return st_upsample_nearest(x, scale, scale)

def add_tensor(a: ptr, b: ptr) -> ptr:
    return st_add_tensor(a, b)

# 保持旧接口的 in-place 语义：返回新指针，调用方覆盖
# 实际 GPU 上无法真正 in-place，只能产生新 tensor
