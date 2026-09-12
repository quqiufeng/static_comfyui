# SD.cpp (stable-diffusion.cpp) backend FFI wrapper.
# This replaces the old libtorch_std_helper backend.

# Opaque pipeline handle
extern fn sd_pipeline_create() -> ptr from "sdcpp_adapter"
extern fn sd_pipeline_free(pipeline: ptr) -> int from "sdcpp_adapter"
extern fn sd_pipeline_load(pipeline: ptr, model_path: str, clip_l_path: str, clip_g_path: str, vae_path: str, wtype: int, n_threads: int, diffusion_fa: int) -> int from "sdcpp_adapter"
extern fn sd_pipeline_load_ex(pipeline: ptr, model_path: str, clip_l_path: str, clip_g_path: str, vae_path: str, wtype: int, n_threads: int, diffusion_fa: int, diffusion_model_path: str, llm_path: str) -> int from "sdcpp_adapter"
extern fn sd_pipeline_generate(pipeline: ptr, prompt: str, negative_prompt: str, width: int, height: int, steps: int, cfg: float, sample_method: str, scheduler: str, seed: int, vae_tiling: int, vae_tile_size: int, vae_tile_overlap: float, hires: int, hires_width: int, hires_height: int, hires_steps: int, hires_strength: float, freeu: int, freeu_b1: float, freeu_b2: float, sag: int, sag_scale: float, output_path: str) -> int from "sdcpp_adapter"
extern fn sd_pipeline_generate_hires(pipeline: ptr, prompt: str, negative_prompt: str, target_width: int, target_height: int, steps: int, cfg: float, sample_method: str, scheduler: str, seed: int, vae_tiling: int, vae_tile_size: int, vae_tile_overlap: float, hires_steps: int, hires_strength: float, freeu: int, freeu_b1: float, freeu_b2: float, sag: int, sag_scale: float, clarity: float, sharpen_amount: float, sharpen_radius: int, smart_sharpen_strength: float, smart_sharpen_radius: int, edge_sharpen_amount: float, edge_sharpen_radius: int, edge_sharpen_threshold: float, output_path: str) -> int from "sdcpp_adapter"

extern fn sd_pipeline_load_lora(pipeline: ptr, lora_path: str, multiplier: float) -> int from "sdcpp_adapter"
extern fn sd_pipeline_set_ipadapter(pipeline: ptr, model_path: str, clip_vision_path: str, image_path: str, weight: float) -> int from "sdcpp_adapter"
extern fn sd_pipeline_set_ipadapter_enabled(pipeline: ptr, enabled: int, weight: float) -> int from "sdcpp_adapter"
extern fn sd_pipeline_set_init_image(pipeline: ptr, image_path: str, strength: float) -> int from "sdcpp_adapter"
extern fn sd_pipeline_load_control_net(pipeline: ptr, path: str) -> int from "sdcpp_adapter"
extern fn sd_pipeline_set_control_image(pipeline: ptr, image_path: str, strength: float) -> int from "sdcpp_adapter"
extern fn sd_pipeline_set_mask(pipeline: ptr, mask_path: str) -> int from "sdcpp_adapter"
extern fn sd_pipeline_set_batch_count(pipeline: ptr, n: int) -> int from "sdcpp_adapter"
extern fn sd_resize_image(input_path: str, output_path: str, width: int, height: int) -> int from "sdcpp_adapter"
extern fn sd_scale_image(input_path: str, output_path: str, scale_by: float) -> int from "sdcpp_adapter"
extern fn sd_invert_image(input_path: str, output_path: str) -> int from "sdcpp_adapter"
extern fn sd_pipeline_get_model_version_name(pipeline: ptr) -> str from "sdcpp_adapter"
extern fn sd_pipeline_set_clip_skip(pipeline: ptr, n: int) -> int from "sdcpp_adapter"
extern fn sd_pipeline_set_flow_shift(pipeline: ptr, shift: float) -> int from "sdcpp_adapter"
extern fn sd_pipeline_set_wtype(pipeline: ptr, wtype: int) -> int from "sdcpp_adapter"
extern fn sd_pipeline_set_flash_attn(pipeline: ptr, enabled: int) -> int from "sdcpp_adapter"
extern fn sd_pipeline_set_rescale_cfg(pipeline: ptr, enabled: int, multiplier: float) -> int from "sdcpp_adapter"
extern fn sd_pipeline_set_video_cfg(pipeline: ptr, enabled: int, mode: int, min_cfg: float) -> int from "sdcpp_adapter"
extern fn sd_pipeline_set_sigma_range(pipeline: ptr, enabled: int, sigma_min: float, sigma_max: float) -> int from "sdcpp_adapter"
extern fn sd_rotate_image(input_path: str, output_path: str, degrees: int) -> int from "sdcpp_adapter"
extern fn sd_flip_image(input_path: str, output_path: str, method: int) -> int from "sdcpp_adapter"
extern fn sd_blend_images(path1: str, path2: str, output_path: str, factor: float) -> int from "sdcpp_adapter"
extern fn sd_make_solid_image(output_path: str, width: int, height: int, r: int, g: int, b: int) -> int from "sdcpp_adapter"
extern fn sd_pad_image(input_path: str, output_path: str, left: int, top: int, right: int, bottom: int, r: int, g: int, b: int) -> int from "sdcpp_adapter"
extern fn sd_blur_image(input_path: str, output_path: str, sigma: float) -> int from "sdcpp_adapter"
extern fn sd_batch_images(path1: str, path2: str, output_path: str) -> int from "sdcpp_adapter"
extern fn sd_composite_masked(dest_path: str, src_path: str, mask_path: str, output_path: str, x: int, y: int) -> int from "sdcpp_adapter"
extern fn sd_crop_image(input_path: str, output_path: str, x: int, y: int, width: int, height: int) -> int from "sdcpp_adapter"
extern fn sd_pipeline_generate_adetailer(pipeline: ptr, prompt: str, negative_prompt: str, width: int, height: int, steps: int, cfg: float, sample_method: str, scheduler: str, seed: int, vae_tiling: int, vae_tile_size: int, vae_tile_overlap: float, hires: int, hires_width: int, hires_height: int, hires_steps: int, hires_strength: float, freeu: int, freeu_b1: float, freeu_b2: float, sag: int, sag_scale: float, ad_model_path: str, ad_prompt: str, ad_negative_prompt: str, output_path: str) -> int from "sdcpp_adapter"
extern fn sd_pipeline_generate_full(pipeline: ptr, prompt: str, negative_prompt: str, width: int, height: int, hires_width: int, hires_height: int, steps: int, cfg: float, sample_method: str, scheduler: str, seed: int, vae_tiling: int, vae_tile_size: int, vae_tile_overlap: float, hires_steps: int, hires_strength: float, freeu: int, freeu_b1: float, freeu_b2: float, sag: int, sag_scale: float, clarity: float, sharpen_amount: float, sharpen_radius: int, smart_sharpen_strength: float, smart_sharpen_radius: int, edge_sharpen_amount: float, edge_sharpen_radius: int, edge_sharpen_threshold: float, ad_model_path: str, ad_prompt: str, ad_negative_prompt: str, output_path: str) -> int from "sdcpp_adapter"
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


def sd_generate_adetailer(pipeline: ptr, prompt: str, negative_prompt: str,
                           width: int, height: int, steps: int, cfg: float,
                           sample_method: str, scheduler: str, seed: int,
                           vae_tiling: int, vae_tile_size: int, vae_tile_overlap: float,
                           hires: int, hires_width: int, hires_height: int,
                           hires_steps: int, hires_strength: float,
                           freeu: int, freeu_b1: float, freeu_b2: float,
                           sag: int, sag_scale: float,
                           ad_model_path: str, ad_prompt: str, ad_negative_prompt: str,
                           output_path: str) -> int:
    return sd_pipeline_generate_adetailer(pipeline, prompt, negative_prompt,
                                           width, height, steps, cfg,
                                           sample_method, scheduler, seed,
                                           vae_tiling, vae_tile_size, vae_tile_overlap,
                                           hires, hires_width, hires_height,
                                           hires_steps, hires_strength,
                                           freeu, freeu_b1, freeu_b2,
                                           sag, sag_scale,
                                           ad_model_path, ad_prompt, ad_negative_prompt,
                                           output_path)


def sd_load_lora(pipeline: ptr, lora_path: str, multiplier: float) -> int:
    return sd_pipeline_load_lora(pipeline, lora_path, multiplier)


def sd_set_ipadapter(pipeline: ptr, model_path: str, clip_vision_path: str, image_path: str, weight: float) -> int:
    return sd_pipeline_set_ipadapter(pipeline, model_path, clip_vision_path, image_path, weight)


def sd_set_ipadapter_enabled(pipeline: ptr, enabled: int, weight: float) -> int:
    return sd_pipeline_set_ipadapter_enabled(pipeline, enabled, weight)


def sd_set_init_image(pipeline: ptr, image_path: str, strength: float) -> int:
    return sd_pipeline_set_init_image(pipeline, image_path, strength)


def sd_load_control_net(pipeline: ptr, path: str) -> int:
    return sd_pipeline_load_control_net(pipeline, path)


def sd_set_control_image(pipeline: ptr, image_path: str, strength: float) -> int:
    return sd_pipeline_set_control_image(pipeline, image_path, strength)


def sd_set_mask(pipeline: ptr, mask_path: str) -> int:
    return sd_pipeline_set_mask(pipeline, mask_path)


def sd_set_batch_count(pipeline: ptr, n: int) -> int:
    return sd_pipeline_set_batch_count(pipeline, n)


def sd_get_model_version_name(pipeline: ptr) -> str:
    return sd_pipeline_get_model_version_name(pipeline)


def sd_set_clip_skip(pipeline: ptr, n: int) -> int:
    return sd_pipeline_set_clip_skip(pipeline, n)


def sd_set_flow_shift(pipeline: ptr, shift: float) -> int:
    return sd_pipeline_set_flow_shift(pipeline, shift)


def sd_set_wtype(pipeline: ptr, wtype: int) -> int:
    return sd_pipeline_set_wtype(pipeline, wtype)


def sd_set_flash_attn(pipeline: ptr, enabled: bool) -> int:
    if enabled:
        return sd_pipeline_set_flash_attn(pipeline, 1)
    return sd_pipeline_set_flash_attn(pipeline, 0)


def sd_set_rescale_cfg(pipeline: ptr, multiplier: float) -> int:
    return sd_pipeline_set_rescale_cfg(pipeline, 1, multiplier)


def sd_set_video_cfg(pipeline: ptr, mode: int, min_cfg: float) -> int:
    return sd_pipeline_set_video_cfg(pipeline, 1, mode, min_cfg)


def sd_set_sigma_range(pipeline: ptr, sigma_min: float, sigma_max: float) -> int:
    return sd_pipeline_set_sigma_range(pipeline, 1, sigma_min, sigma_max)


def sd_ensure_directory(path: str) -> int:
    return sd_ensure_dir(path)


def sd_load_ex(pipeline: ptr, model_path: str, clip_l_path: str, clip_g_path: str,
               vae_path: str, wtype: int, n_threads: int, diffusion_fa: int,
               diffusion_model_path: str, llm_path: str) -> int:
    return sd_pipeline_load_ex(pipeline, model_path, clip_l_path, clip_g_path,
                               vae_path, wtype, n_threads, diffusion_fa,
                               diffusion_model_path, llm_path)


def sd_generate_hires(pipeline: ptr, prompt: str, negative_prompt: str,
                       target_width: int, target_height: int,
                       steps: int, cfg: float,
                       sample_method: str, scheduler: str, seed: int,
                       vae_tiling: int, vae_tile_size: int, vae_tile_overlap: float,
                       hires_steps: int, hires_strength: float,
                       freeu: int, freeu_b1: float, freeu_b2: float,
                       sag: int, sag_scale: float,
                       clarity: float, sharpen_amount: float, sharpen_radius: int,
                       smart_sharpen_strength: float, smart_sharpen_radius: int,
                       edge_sharpen_amount: float, edge_sharpen_radius: int,
                       edge_sharpen_threshold: float,
                       output_path: str) -> int:
    return sd_pipeline_generate_hires(pipeline, prompt, negative_prompt,
                                       target_width, target_height,
                                       steps, cfg,
                                       sample_method, scheduler, seed,
                                       vae_tiling, vae_tile_size, vae_tile_overlap,
                                       hires_steps, hires_strength,
                                       freeu, freeu_b1, freeu_b2,
                                       sag, sag_scale,
                                       clarity, sharpen_amount, sharpen_radius,
                                       smart_sharpen_strength, smart_sharpen_radius,
                                       edge_sharpen_amount, edge_sharpen_radius,
                                       edge_sharpen_threshold,
                                       output_path)


def sd_generate_full(pipeline: ptr, prompt: str, negative_prompt: str,
                     width: int, height: int, opts, output_path: str) -> int:
    return sd_pipeline_generate_full(
        pipeline, prompt, negative_prompt,
        width, height,
        dict_get(opts, "hires_width"), dict_get(opts, "hires_height"),
        dict_get(opts, "steps"), dict_get(opts, "cfg"),
        dict_get(opts, "sampler_name"), dict_get(opts, "scheduler"),
        dict_get(opts, "seed"),
        dict_get(opts, "vae_tiling"), dict_get(opts, "vae_tile_size"),
        dict_get(opts, "vae_tile_overlap"),
        dict_get(opts, "hires_steps"), dict_get(opts, "hires_strength"),
        dict_get(opts, "freeu"), dict_get(opts, "freeu_b1"), dict_get(opts, "freeu_b2"),
        dict_get(opts, "sag"), dict_get(opts, "sag_scale"),
        dict_get(opts, "clarity"), dict_get(opts, "sharpen"), dict_get(opts, "sharpen_radius"),
        dict_get(opts, "smart_sharpen"), dict_get(opts, "smart_sharpen_radius"),
        dict_get(opts, "edge_sharpen"), dict_get(opts, "edge_sharpen_radius"),
        dict_get(opts, "edge_sharpen_threshold"),
        dict_get(opts, "ad_model_path"), dict_get(opts, "ad_prompt"),
        dict_get(opts, "ad_negative_prompt"),
        output_path)


def main():
    pass
