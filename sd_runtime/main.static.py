# main.static.py — static_comfyui 主入口

extern fn dgemm_row_auto(m: int, n: int, k: int, alpha: float,
                         A: ptr, B: ptr, beta: float, C: ptr) -> void from "dgemm_row"
extern fn fopen(path: str, mode: str) -> ptr from "libc"
extern fn fread(buf: ptr, size: int, count: int, fp: ptr) -> int from "libc"
extern fn fclose(fp: ptr) -> int from "libc"
extern fn fwrite(buf: ptr, size: int, count: int, fp: ptr) -> int from "libc"

def load_bin(fn: str, n: int) -> list[float]:
    _r: list[float] = make_float_array(n)
    _fp = fopen(fn, "rb")
    if _fp != 0:
        fread(_r, 8, n, _fp)
        fclose(_fp)
    return _r

def main():
    print("=== static_comfyui ===")
    
    # 加载合并权重
    print("Loading weights (9.6GB SDXL UNet)...")
    _fp = fopen("/tmp/sd_weights/sdxl_merged/weights.bin", "rb")
    if _fp == 0:
        print("ERROR: weights not found at /tmp/sd_weights/sdxl_merged/")
        print("Run: python3 export_sd_weights.py /data/models/image/sd_xl_base_1.0.safetensors /tmp/sd_weights/sdxl")
        exit_program(1)
    data: list[float] = make_float_array(1283731842)
    fread(data, 8, 1283731842, _fp)
    fclose(_fp)
    print("Weights loaded")
    
    # 创建输入
    n: int = 1; hh: int = 64; ww: int = 64
    latent: list[float] = make_float_array(n * 4 * hh * ww)
    _i: int = 0
    while _i < n * 4 * hh * ww:
        float_array_set(latent, _i, 0.1)
        _i = _i + 1
    
    ts: list[float] = make_float_array(n)
    float_array_set(ts, 0, 50.0)
    
    ctx: list[float] = make_float_array(n * 77 * 2048)
    arr_fill(ctx, 0.0, n * 77 * 2048)
    
    # 跑 UNet
    print("UNet forward...")
    result: list[float] = unet_forward(latent, ts, ctx, data, n, hh, ww)
    print("UNet output sum="); print(arr_sum(result, n * 4 * hh * ww))
    
    print("\nDone!")
    exit_program(0)
