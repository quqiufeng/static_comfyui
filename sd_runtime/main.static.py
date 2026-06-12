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
    wdir: str = "/tmp/sd_weights/vae"
    n: int = 1; h: int = 8; w: int = 8
    latent: list[float] = make_float_array(n * 16 * h * w)
    i: int = 0
    while i < n * 16 * h * w:
        float_array_set(latent, i, (i * 7 + 3) % 100 / 100.0 - 0.5)
        i = i + 1
    
    print("Running VAE decoder...")
    image: list[float] = vae_decoder_forward(latent, wdir, n, h, w)
    oh: int = h * 8; ow: int = w * 8
    print("Output: 1x3x"); print(oh); print("x"); print(ow)
    mp: float = arr_sum(image, 1*3*oh*ow) / (1*3*oh*ow)
    print("Mean pixel: "); print(mp)
    
    _fp = fopen("/tmp/vae_output.bin", "wb")
    if _fp != 0:
        fwrite(image, 8, 1*3*oh*ow, _fp)
        fclose(_fp)
        print("Saved to /tmp/vae_output.bin")
    print("Done!")
    exit_program(0)
