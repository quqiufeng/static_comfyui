# unet_forward.static.py — SDXL UNet forward
# 权重从合并文件 weights.bin 加载，按偏移索引访问
# 所有算子通过 torch.*（libtorch C API）

extern fn fopen(path: str, mode: str) -> ptr from "libc"
extern fn fseek(fp: ptr, offset: int, whence: int) -> int from "libc"
extern fn fread(buf: ptr, size: int, count: int, fp: ptr) -> int from "libc"
extern fn fclose(fp: ptr) -> int from "libc"

# ─── 合并权重加载 ──────────────────────────────────

def load_weights(weights_dir: str):
    """加载合并权重文件和偏移索引，返回 (data, index)"""
    # 先读 index.json
    _fp = fopen(weights_dir + "/index.json", "rb")
    if _fp == 0: return (0, 0)
    fclose(_fp)
    # 加载整个权重文件
    _data: list[float] = make_float_array(1283731842)  # 9.6GB / 8 bytes
    _fp2 = fopen(weights_dir + "/weights.bin", "rb")
    if _fp2 != 0:
        fread(_data, 8, 1283731842, _fp2)
        fclose(_fp2)
    return _data

def w_slice(data, offset, n):
    """从权重数据中切出一段"""
    _r: list[float] = make_float_array(n)
    _i: int = 0
    while _i < n:
        float_array_set(_r, _i, float_array_ref(data, offset + _i))
        _i = _i + 1
    return _r
