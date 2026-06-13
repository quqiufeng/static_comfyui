# main.static.py — static_comfyui 主入口（GPU ptr 版本）

extern fn fopen(path: str, mode: str) -> ptr from "libc"
extern fn fread(buf: ptr, size: int, count: int, fp: ptr) -> int from "libc"
extern fn fclose(fp: ptr) -> int from "libc"

def main():
    print("=== static_comfyui ===")
    
    print("Loading weights (10.3GB SDXL UNet float32)...")
    _fp = fopen("/tmp/sd_weights/sdxl_merged_f32/weights.bin", "rb")
    if _fp == 0:
        print("ERROR: weights not found")
        exit_program(1)
    data: list[float] = make_float_array(2567463684)
    fread(data, 4, 2567463684, _fp)
    fclose(_fp)
    print("Weights loaded")
    
    n: int = 1; hh: int = 64; ww: int = 64
    latent: list[float] = make_float_array(n * 4 * hh * ww)
    _i: int = 0
    while _i < n * 4 * hh * ww:
        float_array_set(latent, _i, 0.1)
        _i = _i + 1
    latent_t: ptr = st_from_blob_4d(latent, n, 4, hh, ww)
    
    ts: list[float] = make_float_array(n)
    float_array_set(ts, 0, 50.0)
    ts_t: ptr = st_from_blob_1d(ts, n)
    
    ctx: list[float] = make_float_array(n * 77 * 2048)
    arr_fill(ctx, 0.0, n * 77 * 2048)
    ctx_t: ptr = st_from_blob_2d(ctx, n * 77, 2048)
    
    print("UNet forward...")
    result_t: ptr = unet_forward(latent_t, ts, ctx_t, data, n, hh, ww)
    
    print("UNet output sum="); print(st_sum(result_t))
    print("UNet output max="); print(st_max(result_t))
    print("UNet output min="); print(st_min(result_t))
    st_tensor_save(result_t, "/tmp/elf_unet_output.bin")
    
    print("\nDone!")
    exit_program(0)
