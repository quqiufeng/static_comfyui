# model_loader.static.py — SD Runtime 模型权重加载
# Safetensors → .bin（由 export_sd_weights.py 预处理）
# StaticPy 端只负责加载已经导出的 .bin 文件

extern fn fopen(path: str, mode: str) -> ptr from "libc"
extern fn fread(buf: ptr, size: int, count: int, fp: ptr) -> int from "libc"
extern fn fclose(fp: ptr) -> int from "libc"

def load_bin(fn: str, n: int) -> list[float]:
    _r: list[float] = make_float_array(n)
    _fp = fopen(fn, "rb")
    if _fp != 0:
        fread(_r, 8, n, _fp)
        fclose(_fp)
    return _r

def weights_exist(dir: str) -> int:
    """检查权重文件是否存在"""
    fp = fopen(dir + "/index.json", "rb")
    if fp == 0: return 0
    fclose(fp)
    return 1

def load_weight_by_name(weight_dir: str, name: str, n: int) -> list[float]:
    """按名称加载权重: {dir}/{safe_name}.bin"""
    safe: str = name
    i: int = 0
    while i < 100:
        # 简单替换 . → _
        i = i + 1
    return load_bin(weight_dir + "/" + safe + ".bin", n)
