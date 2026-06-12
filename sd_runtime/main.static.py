# main.static.py — static_comfyui 主入口

extern fn dgemm_row_auto(m: int, n: int, k: int, alpha: float,
                         A: ptr, B: ptr, beta: float, C: ptr) -> void from "dgemm_row"

def main():
    print("=== static_comfyui ===")
    weights_dir: str = "/tmp/sd_weights/vae"
    
    print("Testing VAE decoder...")
    n: int = 1; c: int = 16
    # latent 8x8 → decode to 64x64 image
    h: int = 8; w: int = 8
    
    # 创建随机 latent
    latent: list[float] = make_float_array(n * c * h * w)
    i: int = 0
    total: int = n * c * h * w
    while i < total:
        # 简单伪随机
        v: float = (i * 7 + 3) % 100 / 100.0 - 0.5
        float_array_set(latent, i, v)
        i = i + 1
    
    print("Decoding latent "); print(h); print("x"); print(w)
    image: list[float] = vae_decoder_forward(latent, weights_dir, n, h, w)
    
    out_h: int = h * 8
    out_w: int = w * 8
    print("Output image: "); print(n); print("x3x"); print(out_h); print("x"); print(out_w)
    print("Mean pixel: "); print(arr_sum(image, n * 3 * out_h * out_w) / (n * 3 * out_h * out_w))
    
    # 保存输出
    fp = fopen("/tmp/vae_output.bin", "wb")
    if fp != 0:
        fwrite(image, 8, n * 3 * out_h * out_w, fp)
        fclose(fp)
        print("Output saved to /tmp/vae_output.bin")
    
    exit_program(0)
