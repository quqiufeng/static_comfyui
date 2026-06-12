# main.static.py — 最小测试

extern fn dgemm_row_auto(m: int, n: int, k: int, alpha: float,
                         A: ptr, B: ptr, beta: float, C: ptr) -> void from "dgemm_row"
extern fn fopen(path: str, mode: str) -> ptr from "libc"
extern fn fread(buf: ptr, size: int, count: int, fp: ptr) -> int from "libc"
extern fn fclose(fp: ptr) -> int from "libc"

def load_bin(fn: str, n: int) -> list[float]:
    r: list[float] = make_float_array(n)
    fp = fopen(fn, "rb")
    if fp != 0:
        fread(r, 8, n, fp)
        fclose(fp)
    return r

def main():
    print("=== minimal test ===")
    wgt = load_bin("/tmp/sd_weights/vae/decoder_conv_in_weight.bin", 512*16*3*3)
    bias: list[float] = make_float_array(512)
    arr_fill(bias, 0.0, 512)
    x: list[float] = make_float_array(1*16*8*8)
    arr_fill(x, 0.5, 1*16*8*8)
    
    # inline conv2d
    _h_out: int = 8; _w_out: int = 8
    _ncol: int = 64; _kdim: int = 144
    _col: list[float] = make_float_array(64 * 144)
    im2col(x, 1, 16, 8, 8, 3, 1, 1, _col)
    _y: list[float] = make_float_array(64 * 512)
    dgemm_row_auto(64, 512, 144, 1.0, _col, wgt, 0.0, _y)
    _i: int = 0
    while _i < 64:
        _j: int = 0
        while _j < 512:
            float_array_set(_y, _i*512+_j, float_array_ref(_y, _i*512+_j) + float_array_ref(bias, _j))
            _j = _j + 1
        _i = _i + 1
    print("sum="); print(arr_sum(_y, 64*512))
    exit_program(0)
