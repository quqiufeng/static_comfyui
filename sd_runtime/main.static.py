# main.static.py — static_comfyui 主入口（GPU ptr 版本）

extern fn fopen(path: str, mode: str) -> ptr from "libc"
extern fn fread(buf: ptr, size: int, count: int, fp: ptr) -> int from "libc"
extern fn fclose(fp: ptr) -> int from "libc"

extern fn make_float_array(n: int) -> list[float] from "prelude"
extern fn float_array_set(a: list[float], i: int, v: float) -> void from "prelude"
extern fn float_array_ref(a: list[float], i: int) -> float from "prelude"

extern fn st_from_blob_1d(data: ptr, d0: int) -> ptr from "staticpy_torch"
extern fn st_from_blob_2d(data: ptr, d0: int, d1: int) -> ptr from "staticpy_torch"
extern fn st_from_blob_4d(data: ptr, d0: int, d1: int, d2: int, d3: int) -> ptr from "staticpy_torch"
extern fn st_sum(t: ptr) -> float from "staticpy_torch"
extern fn st_max(t: ptr) -> float from "staticpy_torch"
extern fn st_min(t: ptr) -> float from "staticpy_torch"
extern fn st_tensor_save(t: ptr, path: str) -> void from "staticpy_torch"

def load_bin(fn: str, n: int) -> list[float]:
    r: list[float] = make_float_array(n)
    fp = fopen(fn, "rb")
    if fp != 0:
        fread(r, 4, n, fp)
        fclose(fp)
    return r

def main():
    print("=== static_comfyui ===")
    
    print("Loading CLIP weights...")
    clip_data: list[float] = load_bin("/tmp/sd_weights/clip_lg_merged_f32/weights.bin", 817720398)
    print("CLIP weights loaded")
    
    print("Loading tokens...")
    tokens_l: list[float] = load_bin("/tmp/tokens_l.bin", 77)
    tokens_g: list[float] = load_bin("/tmp/tokens_g.bin", 77)
    print("Tokens loaded")
    
    print("CLIP encode...")
    ctx: ptr = clip_encode_lg(tokens_l, tokens_g, clip_data)
    print("Context sum="); print(st_sum(ctx))
    st_tensor_save(ctx, "/tmp/elf_ctx.bin")
    
    print("Loading UNet weights...")
    unet_data: list[float] = load_bin("/tmp/sd_weights/sdxl_merged_f32/weights.bin", 2567463684)
    print("UNet weights loaded")
    
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
    
    print("UNet forward...")
    result_t: ptr = unet_forward(latent_t, ts, ctx, unet_data, n, hh, ww)
    
    print("UNet output sum="); print(st_sum(result_t))
    print("UNet output max="); print(st_max(result_t))
    print("UNet output min="); print(st_min(result_t))
    st_tensor_save(result_t, "/tmp/elf_unet_output.bin")
    
    print("\nDone!")
    exit_program(0)
