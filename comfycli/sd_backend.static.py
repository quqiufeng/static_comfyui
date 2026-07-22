# SD.cpp (stable-diffusion.cpp) backend FFI wrapper.
# This replaces the old libtorch_std_helper backend.

# Opaque pipeline handle
extern fn sd_pipeline_create() -> ptr from "sdcpp_adapter"
extern fn sd_pipeline_free(pipeline: ptr) -> int from "sdcpp_adapter"
extern fn sd_pipeline_load(pipeline: ptr, model_path: str, clip_l_path: str, clip_g_path: str, vae_path: str, wtype: int, n_threads: int, diffusion_fa: int) -> int from "sdcpp_adapter"
extern fn sd_pipeline_generate(pipeline: ptr, prompt: str, negative_prompt: str, width: int, height: int, steps: int, cfg: float, sample_method: str, scheduler: str, seed: int, vae_tiling: int, vae_tile_size: int, vae_tile_overlap: float, hires: int, hires_width: int, hires_height: int, hires_steps: int, hires_strength: float, freeu: int, freeu_b1: float, freeu_b2: float, sag: int, sag_scale: float, output_path: str) -> int from "sdcpp_adapter"
extern fn sd_ensure_dir(path: str) -> int from "sdcpp_adapter"

# SD weight type constants (matching stable-diffusion.h sd_type_t)
SD_WTYPE_F32: int = 0
SD_WTYPE_F16: int = 1
SD_WTYPE_AUTO: int = 42  # SD_TYPE_COUNT


def sd_create() -> ptr:
    return sd_pipeline_create()


def sd_free(pipeline: ptr) -> int:
    return sd_pipeline_free(pipeline)


def sd_load(pipeline: ptr, model_path: str, clip_l_path: str, clip_g_path: str,
            vae_path: str, wtype: int, n_threads: int, diffusion_fa: int) -> int:
    return sd_pipeline_load(pipeline, model_path, clip_l_path, clip_g_path,
                            vae_path, wtype, n_threads, diffusion_fa)


def sd_generate(pipeline: ptr, prompt: str, negative_prompt: str,
                width: int, height: int, steps: int, cfg: float,
                sample_method: str, scheduler: str, seed: int,
                output_path: str) -> int:
    return sd_pipeline_generate(pipeline, prompt, negative_prompt,
                                width, height, steps, cfg,
                                sample_method, scheduler, seed,
                                0, 0, 0.0,
                                0, 0, 0, 0, 0.0,
                                0, 0.0, 0.0,
                                0, 0.0,
                                output_path)


def sd_generate_with_options(pipeline: ptr, prompt: str, negative_prompt: str,
                              width: int, height: int, steps: int, cfg: float,
                              sample_method: str, scheduler: str, seed: int,
                              vae_tiling: int, vae_tile_size: int, vae_tile_overlap: float,
                              hires: int, hires_width: int, hires_height: int,
                              hires_steps: int, hires_strength: float,
                              freeu: int, freeu_b1: float, freeu_b2: float,
                              sag: int, sag_scale: float,
                              output_path: str) -> int:
    return sd_pipeline_generate(pipeline, prompt, negative_prompt,
                                width, height, steps, cfg,
                                sample_method, scheduler, seed,
                                vae_tiling, vae_tile_size, vae_tile_overlap,
                                hires, hires_width, hires_height, hires_steps, hires_strength,
                                freeu, freeu_b1, freeu_b2,
                                sag, sag_scale,
                                output_path)


def sd_ensure_directory(path: str) -> int:
    return sd_ensure_dir(path)


def main():
    pass
