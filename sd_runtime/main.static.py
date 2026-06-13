# main.static.py — static_comfyui 主入口

extern fn dgemm_row_auto(m: int, n: int, k: int, alpha: float,
                         A: ptr, B: ptr, beta: float, C: ptr) -> void from "dgemm_row"
extern fn fopen(path: str, mode: str) -> ptr from "libc"
extern fn fread(buf: ptr, size: int, count: int, fp: ptr) -> int from "libc"
extern fn fwrite(buf: ptr, size: int, count: int, fp: ptr) -> int from "libc"
extern fn fclose(fp: ptr) -> int from "libc"

def load_bin(fn: str, n: int) -> list[float]:
    _r: list[float] = make_float_array(n)
    _fp = fopen(fn, "rb")
    if _fp != 0:
        fread(_r, 8, n, _fp)
        fclose(_fp)
    return _r

def main():
    print("=== static_comfyui ===")
    wdir: str = "/tmp/sd_weights/sdxl"
    
    # 创建随机 latent [1,4,64,64] 和 timestep
    n: int = 1; hh: int = 64; ww: int = 64
    latent: list[float] = make_float_array(n * 4 * hh * ww)
    arr_fill(latent, 0.1, n * 4 * hh * ww)
    
    ts: list[float] = make_float_array(n)
    float_array_set(ts, 0, 50.0)
    
    ctx: list[float] = make_float_array(n * 77 * 2048)
    arr_fill(ctx, 0.0, n * 77 * 2048)
    
    print("Running UNet forward...")
    result: list[float] = unet_forward(latent, ts, ctx, wdir, n, hh, ww)
    print("UNet output sum="); print(arr_sum(result, n * 4 * hh * ww))
    
    print("\nDone!")
    exit_program(0)
