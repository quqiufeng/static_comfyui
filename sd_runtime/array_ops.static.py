# array_ops.static.py — 张量运算基础设施（GPU ptr 版本保留）
# 现在主要计算在 GPU 上，此文件保留辅助函数

# 时间 emb broadcast add: h_cur [N,C,H,W], emb [N,C]
def add_time_emb_tensor(h: ptr, emb: ptr, n: int, c: int) -> ptr:
    # emb 是 [N,C]，reshape 为 [N,C,1,1]
    _dims: list[int] = make_int_array(4)
    int_array_set(_dims, 0, n)
    int_array_set(_dims, 1, c)
    int_array_set(_dims, 2, 1)
    int_array_set(_dims, 3, 1)
    emb4: ptr = st_reshape(emb, _dims, 4)
    r: ptr = st_add_tensor(h, emb4)
    st_tensor_free(emb4)
    return r

# 保留 CPU 数组辅助函数
def arr_fill(a: list[float], v: float, n: int):
    i: int = 0
    while i < n:
        float_array_set(a, i, v)
        i = i + 1

# 通道维度拼接: cur [N,C1,H,W], skip [N,C2,H,W] -> [N,C1+C2,H,W]
def cat_channel_tensors(cur: ptr, skip: ptr, n: int, c1: int, c2: int, h: int, w: int) -> ptr:
    return st_cat_channel(cur, skip)

extern fn st_reshape(t: ptr, dims: ptr, ndim: int) -> ptr from "staticpy_torch"
extern fn st_new(dims: ptr, ndim: int) -> ptr from "staticpy_torch"
extern fn st_cat_channel(a: ptr, b: ptr) -> ptr from "staticpy_torch"
extern fn st_add_tensor(a: ptr, b: ptr) -> ptr from "staticpy_torch"
extern fn make_int_array(n: int) -> list[int] from "prelude"
extern fn int_array_set(a: list[int], i: int, v: int) -> void from "prelude"
extern fn make_ptr_array(n: int) -> ptr from "prelude"
extern fn ptr_array_set(a: ptr, i: int, v: ptr) -> void from "prelude"
extern fn ptr_array_ref(a: ptr, i: int) -> ptr from "prelude"
