# test_sampler_main.static.py — 5-step Euler sampler smoke test

extern fn make_float_array(n: int) -> list[float] from "prelude"
extern fn float_array_set(a: list[float], i: int, v: float) -> void from "prelude"
extern fn file_read_floats(path: str, n: int) -> list[float] from "prelude"

extern fn st_from_blob_1d(data: ptr, d0: int) -> ptr from "staticpy_torch"
extern fn st_from_blob_2d(data: ptr, d0: int, d1: int) -> ptr from "staticpy_torch"
extern fn st_from_blob_3d(data: ptr, d0: int, d1: int, d2: int) -> ptr from "staticpy_torch"
extern fn st_from_blob_4d(data: ptr, d0: int, d1: int, d2: int, d3: int) -> ptr from "staticpy_torch"
extern fn st_tensor_to_half(t: ptr) -> ptr from "staticpy_torch"
extern fn st_sum(t: ptr) -> float from "staticpy_torch"
extern fn st_cuda_torch_memory_allocated() -> int from "staticpy_torch"
extern fn st_cuda_empty_cache() -> void from "staticpy_torch"

def main():
    print("Loading UNet weights...")
    unet_data: list[float] = file_read_floats("/tmp/sdxl_unet_merged_f32/weights.bin", 2567463684)
    unet_weights: ptr = load_unet_weights(unet_data)
    print("Weights ready")

    n: int = 1; c: int = 4; h: int = 64; w: int = 64
    latent: list[float] = make_float_array(n * c * h * w)
    i: int = 0
    while i < n * c * h * w:
        float_array_set(latent, i, 0.1)
        i = i + 1
    x: ptr = st_tensor_to_half(st_from_blob_4d(latent, n, c, h, w))

    ctx_cpu: list[float] = make_float_array(1 * 77 * 2048)
    i = 0
    while i < 1 * 77 * 2048:
        float_array_set(ctx_cpu, i, 0.02)
        i = i + 1
    ctx: ptr = st_tensor_to_half(st_from_blob_3d(ctx_cpu, 1, 77, 2048))
    uncond_ctx: ptr = st_tensor_to_half(st_from_blob_3d(ctx_cpu, 1, 77, 2048))

    y_cpu: list[float] = make_float_array(2816)
    i = 0
    while i < 2816:
        float_array_set(y_cpu, i, 0.03)
        i = i + 1
    y: ptr = st_tensor_to_half(st_from_blob_2d(y_cpu, 1, 2816))
    uncond_y: ptr = st_tensor_to_half(st_from_blob_2d(y_cpu, 1, 2816))

    steps: int = 5
    sigmas: list[float] = karras_sigmas(steps, 0.0292, 14.6146, 7.0)
    timesteps: list[float] = make_float_array(steps)
    i = 0
    while i < steps:
        float_array_set(timesteps, i, float(steps - 1 - i) * (999.0 / float(steps - 1)))
        i = i + 1

    print("Sampling...")
    out: ptr = sample_euler(unet_weights, x, sigmas, timesteps, ctx, uncond_ctx, y, uncond_y, n, c, h, w, steps, 1.0)
    print("sample sum="); print(st_sum(out))
    print("mem="); print(st_cuda_torch_memory_allocated())
    print("Done")
    exit_program(0)
