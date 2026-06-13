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
    weights_dir: str = "/tmp/sd_weights/sdxl"
    
    print("\nTest 1: conv2d_inline with real weights...")
    _w = load_bin("/tmp/sd_weights/vae/decoder_conv_in_weight.bin", 512*16*3*3)
    _b = load_bin("/tmp/sd_weights/vae/decoder_conv_in_bias.bin", 512)
    _x: list[float] = make_float_array(1*16*8*8)
    arr_fill(_x, 0.5, 1*16*8*8)
    _y: list[float] = conv2d_inline(_x, _w, _b, 1, 16, 512, 8, 8)
    print("conv2d_inline sum="); print(arr_sum(_y, 64*512))
    
    print("\nTest 2: Loading SDXL UNet weights...")
    _fp = fopen(weights_dir + "/index.json", "rb")
    if _fp != 0:
        fclose(_fp)
        _a = load_bin(weights_dir + "/model_diffusion_model_input_blocks_0_0_weight.bin", 4*320*3*3)
        _b2 = load_bin(weights_dir + "/model_diffusion_model_input_blocks_0_0_bias.bin", 320)
        print("First weight loaded, val[0]="); print(float_array_ref(_a, 0))
    else:
        print("No SDXL weights at "); print(weights_dir)
    
    print("\nTest 3: group_norm + silu 512ch...")
    _z: list[float] = make_float_array(1*512*8*8)
    arr_fill(_z, 0.5, 1*512*8*8)
    _gw: list[float] = make_float_array(512)
    _gb: list[float] = make_float_array(512)
    arr_fill(_gw, 1.0, 512)
    arr_fill(_gb, 0.0, 512)
    group_norm(_z, _gw, _gb, 32, 512, 64)
    arr_silu(_z, _z, 1*512*64)
    print("group_norm+silu OK, sum="); print(arr_sum(_z, 1*512*64))
    
    print("\nAll tests passed!")
    exit_program(0)
