# test_attention_compare.static.py — 对比 UNet manual vs SDPA attention 输出
# 用法：先生成 manual 版本编译运行保存输出，再生成 sdpa 版本编译运行保存输出，最后用脚本比较。

extern fn make_float_array(n: int) -> list[float] from "prelude"
extern fn float_array_set(a: list[float], i: int, v: float) -> void from "prelude"
extern fn file_read_floats(path: str, n: int) -> list[float] from "prelude"

extern fn st_from_blob_1d(data: ptr, d0: int) -> ptr from "staticpy_torch"
extern fn st_from_blob_2d(data: ptr, d0: int, d1: int) -> ptr from "staticpy_torch"
extern fn st_from_blob_3d(data: ptr, d0: int, d1: int, d2: int) -> ptr from "staticpy_torch"
extern fn st_from_blob_4d(data: ptr, d0: int, d1: int, d2: int, d3: int) -> ptr from "staticpy_torch"
extern fn st_tensor_to_half(t: ptr) -> ptr from "staticpy_torch"
extern fn st_sum(t: ptr) -> float from "staticpy_torch"
extern fn st_tensor_save(t: ptr, path: str) -> void from "staticpy_torch"

def main():
    print("Loading UNet weights...")
    unet_data: list[float] = file_read_floats("/tmp/sdxl_unet_merged_f32/weights.bin", 2567463684)
    print("Loaded")
    unet_weights: ptr = load_unet_weights(unet_data)
    print("Weights ptr ready")

    n: int = 1; hh: int = 64; ww: int = 64
    latent: list[float] = make_float_array(n * 4 * hh * ww)
    i: int = 0
    while i < n * 4 * hh * ww:
        float_array_set(latent, i, 0.1)
        i = i + 1
    latent_t: ptr = st_tensor_to_half(st_from_blob_4d(latent, n, 4, hh, ww))

    # dummy context [1,77,2048]
    ctx_cpu: list[float] = make_float_array(1 * 77 * 2048)
    i = 0
    while i < 1 * 77 * 2048:
        float_array_set(ctx_cpu, i, 0.02)
        i = i + 1
    ctx: ptr = st_tensor_to_half(st_from_blob_3d(ctx_cpu, 1, 77, 2048))

    # dummy y [1,2816]
    y_cpu: list[float] = make_float_array(2816)
    i = 0
    while i < 2816:
        float_array_set(y_cpu, i, 0.03)
        i = i + 1
    y: ptr = st_tensor_to_half(st_from_blob_2d(y_cpu, 1, 2816))

    ts: list[float] = make_float_array(1)
    float_array_set(ts, 0, 50.0)

    print("UNet forward...")
    out: ptr = unet_forward(latent_t, ts, ctx, y, unet_weights, n, hh, ww)
    print("output sum="); print(st_sum(out))

    # Save output for later comparison
    st_tensor_save(out, "/tmp/unet_output.bin")
    print("Saved /tmp/unet_output.bin")
    print("Done")
    exit_program(0)
