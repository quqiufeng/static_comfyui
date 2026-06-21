# main.static.py — static_comfyui 主入口（GPU ptr 版本）

extern fn make_float_array(n: int) -> list[float] from "prelude"
extern fn float_array_set(a: list[float], i: int, v: float) -> void from "prelude"
extern fn float_array_ref(a: list[float], i: int) -> float from "prelude"
extern fn file_read_floats(path: str, n: int) -> list[float] from "prelude"
extern fn make_ptr_array(n: int) -> ptr from "prelude"
extern fn ptr_array_ref(a: ptr, i: int) -> ptr from "prelude"

extern fn st_from_blob_1d(data: ptr, d0: int) -> ptr from "staticpy_torch"
extern fn st_from_blob_2d(data: ptr, d0: int, d1: int) -> ptr from "staticpy_torch"
extern fn st_from_blob_4d(data: ptr, d0: int, d1: int, d2: int, d3: int) -> ptr from "staticpy_torch"
extern fn st_tensor_to_half(t: ptr) -> ptr from "staticpy_torch"
extern fn st_cat_dim(a: ptr, b: ptr, dim: int) -> ptr from "staticpy_torch"
extern fn st_sum(t: ptr) -> float from "staticpy_torch"
extern fn st_max(t: ptr) -> float from "staticpy_torch"
extern fn st_min(t: ptr) -> float from "staticpy_torch"
extern fn st_tensor_save(t: ptr, path: str) -> void from "staticpy_torch"

def main():
    print("=== static_comfyui ===")

    print("Loading CLIP weights...")
    clip_data: list[float] = file_read_floats("/tmp/sd_weights/clip_lg_merged_f32/weights.bin", 817720398)
    print("CLIP weights loaded")

    print("Loading tokens...")
    tokens_l: list[float] = file_read_floats("/tmp/tokens_l.bin", 77)
    tokens_g: list[float] = file_read_floats("/tmp/tokens_g.bin", 77)
    print("Tokens loaded")

    print("CLIP encode...")
    clip_out: ptr = clip_encode_lg(tokens_l, tokens_g, clip_data)
    ctx: ptr = ptr_array_ref(clip_out, 0)
    pooled_g: ptr = ptr_array_ref(clip_out, 1)
    print("Context sum="); print(st_sum(ctx))
    print("Pooled sum="); print(st_sum(pooled_g))
    st_tensor_save(ctx, "/tmp/elf_ctx.bin")
    st_tensor_save(pooled_g, "/tmp/elf_pooled_g.bin")

    print("Loading UNet weights...")
    unet_data: list[float] = file_read_floats("/tmp/sdxl_unet_merged_f32/weights.bin", 2567463684)
    print("UNet weights loaded")
    unet_weights: ptr = load_unet_weights(unet_data)

    n: int = 1; hh: int = 64; ww: int = 64
    latent: list[float] = make_float_array(n * 4 * hh * ww)
    _i: int = 0
    while _i < n * 4 * hh * ww:
        float_array_set(latent, _i, 0.1)
        _i = _i + 1
    latent_t: ptr = st_tensor_to_half(st_from_blob_4d(latent, n, 4, hh, ww))

    ts: list[float] = make_float_array(n)
    float_array_set(ts, 0, 50.0)

    # Build SDXL y conditioning: [pooled_g, image_dim_embed]
    # TODO: use proper sinusoidal embedding for height/width/crop/target dims
    img_dim: list[float] = make_float_array(1536)
    _i = 0
    while _i < 1536:
        float_array_set(img_dim, _i, 0.0)
        _i = _i + 1
    img_dim_t: ptr = st_from_blob_2d(img_dim, 1, 1536)
    y_f: ptr = st_cat_dim(pooled_g, img_dim_t, 1)
    y: ptr = st_tensor_to_half(y_f)

    print("UNet forward...")
    result_t: ptr = unet_forward(latent_t, ts, ctx, y, unet_weights, n, hh, ww)

    print("UNet output sum="); print(st_sum(result_t))
    print("UNet output max="); print(st_max(result_t))
    print("UNet output min="); print(st_min(result_t))
    st_tensor_save(result_t, "/tmp/elf_unet_output.bin")

    print("\nDone!")
    exit_program(0)
