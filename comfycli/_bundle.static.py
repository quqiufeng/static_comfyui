from comfycli_builtins import dict_keys, is_link, path_dirname, path_split

# === cli_args.static.py ===
def parse_cli_args() -> dict:
    args_list: list[str] = list_to_py_list(argv())
    argc: int = py_list_length(args_list)

    checkpoint: str = ""
    prompt: str = ""
    output: str = ""
    output_dir: str = ""
    workflow: str = ""
    show_help: bool = False
    list_nodes: bool = False
    cpu: bool = False
    cuda_device: str = "0"
    highvram: bool = False
    lowvram: bool = False
    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg: float = 7.0
    seed: int = 42
    sampler: str = "euler_a"
    scheduler: str = "discrete"

    i: int = 1
    while i < argc:
        arg: str = py_list_ref(args_list, i)
        if arg == "--help" or arg == "-h":
            show_help = True
            i = i + 1
            continue
        elif arg == "--list-nodes":
            list_nodes = True
        elif arg == "--checkpoint" or arg == "--ckpt":
            i = i + 1
            if i < argc:
                checkpoint = py_list_ref(args_list, i)
        elif arg == "--prompt" or arg == "-p":
            i = i + 1
            if i < argc:
                prompt = py_list_ref(args_list, i)
        elif arg == "--output" or arg == "-o":
            i = i + 1
            if i < argc:
                output = py_list_ref(args_list, i)
        elif arg == "--output-dir" or arg == "--output_directory":
            i = i + 1
            if i < argc:
                output_dir = py_list_ref(args_list, i)
        elif arg == "--cpu":
            cpu = True
        elif arg == "--width" or arg == "-W":
            i = i + 1
            if i < argc:
                width = string_to_int(py_list_ref(args_list, i))
        elif arg == "--height" or arg == "-H":
            i = i + 1
            if i < argc:
                height = string_to_int(py_list_ref(args_list, i))
        elif arg == "--steps" or arg == "-s":
            i = i + 1
            if i < argc:
                steps = string_to_int(py_list_ref(args_list, i))
        elif arg == "--seed" or arg == "-S":
            i = i + 1
            if i < argc:
                seed = string_to_int(py_list_ref(args_list, i))
        elif arg == "--cfg" or arg == "-C":
            i = i + 1
            if i < argc:
                cfg = string_to_float(py_list_ref(args_list, i))
        elif arg == "--sampler":
            i = i + 1
            if i < argc:
                sampler = py_list_ref(args_list, i)
        elif arg == "--scheduler":
            i = i + 1
            if i < argc:
                scheduler = py_list_ref(args_list, i)
        elif arg == "--cuda-device" or arg == "--cuda_device":
            i = i + 1
            if i < argc:
                cuda_device = py_list_ref(args_list, i)
        elif arg == "--highvram" or arg == "--gpu-only":
            highvram = True
        elif arg == "--lowvram":
            lowvram = True
        else:
            if not str_starts_with(arg, "-"):
                workflow = arg
        i = i + 1

    result = make_dict()
    dict_set(result, "checkpoint", checkpoint)
    dict_set(result, "prompt", prompt)
    dict_set(result, "output", output)
    dict_set(result, "output_dir", output_dir)
    dict_set(result, "workflow", workflow)
    dict_set(result, "show_help", show_help)
    dict_set(result, "list_nodes", list_nodes)
    dict_set(result, "cpu", cpu)
    dict_set(result, "cuda_device", cuda_device)
    dict_set(result, "highvram", highvram)
    dict_set(result, "lowvram", lowvram)
    dict_set(result, "width", width)
    dict_set(result, "height", height)
    dict_set(result, "steps", steps)
    dict_set(result, "cfg", cfg)
    dict_set(result, "seed", seed)
    dict_set(result, "sampler", sampler)
    dict_set(result, "scheduler", scheduler)
    return result


def print_help():
    print("ComfyCLI - Stable Diffusion workflow executor")
    print("")
    print("Usage:")
    print("  comfycli-bin [--checkpoint <path>] [--prompt <text>] [--output <path>]")
    print("  comfycli-bin <workflow.json> [--output-dir <path>]")
    print("  comfycli-bin --help")
    print("")
    print("Options:")
    print("  --checkpoint, --ckpt <path>     Model checkpoint path")
    print("  --prompt, -p <text>             Text prompt")
    print("  --output, -o <path>             Output image path")
    print("  --output-dir <path>             Output directory")
    print("  --width, -W <int>               Image width (default: 1024)")
    print("  --height, -H <int>              Image height (default: 1024)")
    print("  --steps, -s <int>                 Sampling steps (default: 20)")
    print("  --seed, -S <int>                  Random seed (default: 42)")
    print("  --cfg, -C <float>               CFG scale (default: 7.0)")
    print("  --sampler <name>                Sampler name (default: euler_a)")
    print("  --scheduler <name>              Scheduler name (default: discrete)")
    print("  --cpu                           CPU mode (no GPU)")
    print("  --cuda-device <id>              CUDA device ID (default: 0)")
    print("  --highvram, --gpu-only          Keep all models on GPU")
    print("  --lowvram                       Offload models to CPU")
    print("  --list-nodes                    List all registered node types")
    print("  --help, -h                      Show this help")
# === sd_backend.static.py ===
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
# === torch_helper.static.py ===
# torch_helper.static.py — libtorch_std_helper.so FFI
#
# 用于 sd.cpp C API 无法覆盖的权重级操作（模型合并 / CLIP 合并 / 权重导出）。
# 共享库由 comfycli_ffi.scm 以 guard 方式加载；缺失时相关节点不可用。

extern fn torch_std_safetensors_load(path: str) -> ptr from "torch_helper"
extern fn torch_std_safetensors_count(d: ptr) -> int from "torch_helper"
extern fn torch_std_safetensors_save(d: ptr, path: str) -> int from "torch_helper"
extern fn torch_std_safetensors_merge(a: ptr, b: ptr, mode: int, prefixes_csv: str, ratios_csv: str, default_ratio: float, strip_prefix: str) -> ptr from "torch_helper"
extern fn torch_std_safetensors_free(d: ptr) -> None from "torch_helper"
extern fn torch_std_copy_file(src: str, dst: str) -> int from "torch_helper"

# === nodes.static.py ===

NODE_CLASS_MAPPINGS: dict = make_dict()
NODE_DISPLAY_NAMES: dict = make_dict()


@dataclass
class SDPipelineHandle:
    pipeline: ptr
    model_path: str
    clip_l_path: str
    clip_g_path: str
    vae_path: str


@dataclass
class Conditioning:
    text: str
    control_net_path: str
    control_image_path: str
    control_strength: float


@dataclass
class ControlNetModel:
    name: str


@dataclass
class LatentImage:
    width: int
    height: int
    batch_size: int
    image_path: str
    mask_path: str


@dataclass
class CLIPVisionModel:
    name: str


@dataclass
class IPAdapterModel:
    name: str


def make_sd_pipeline_handle(pipeline: ptr, model_path: str, clip_l_path: str, clip_g_path: str, vae_path: str) -> SDPipelineHandle:
    return SDPipelineHandle(pipeline, model_path, clip_l_path, clip_g_path, vae_path)


def model_root() -> str:
    root = os_getenv("COMFYCLI_MODEL_DIR")
    if str_length(root) == 0:
        return "/data/models/image"
    return root


def resolve_model_path(name: str) -> str:
    if str_starts_with(name, "/"):
        return name
    return model_root() + "/" + name


def resolve_prompt_text(inputs, key: str, fallback_key: str) -> str:
    c: Conditioning = dict_get(inputs, key)
    if c is not None:
        return c.text
    return get_str(inputs, fallback_key, "")


def conditioning_text(c: Conditioning) -> str:
    if c is None:
        return ""
    return c.text


def merge_conditioning_text(a: str, b: str) -> str:
    if a == "" and b == "":
        return ""
    if a == "":
        return b
    if b == "":
        return a
    return a + ", " + b


def parse_sampler_opts(inputs) -> dict:
    opts = make_dict()
    dict_set(opts, "hires_width", 0)
    dict_set(opts, "hires_height", 0)
    dict_set(opts, "steps", get_int(inputs, "steps", 20))
    dict_set(opts, "cfg", get_float(inputs, "cfg", 7.0))
    dict_set(opts, "sampler_name", get_str(inputs, "sampler_name", "euler"))
    dict_set(opts, "scheduler", get_str(inputs, "scheduler", "normal"))
    dict_set(opts, "seed", get_int(inputs, "seed", 42))
    dict_set(opts, "vae_tiling", get_int(inputs, "vae_tiling", 0))
    dict_set(opts, "vae_tile_size", get_int(inputs, "vae_tile_size", 0))
    dict_set(opts, "vae_tile_overlap", get_float(inputs, "vae_tile_overlap", 0.5))
    dict_set(opts, "hires_steps", 0)
    dict_set(opts, "hires_strength", 0.0)
    dict_set(opts, "freeu", get_int(inputs, "freeu", 0))
    dict_set(opts, "freeu_b1", get_float(inputs, "freeu_b1", 0.0))
    dict_set(opts, "freeu_b2", get_float(inputs, "freeu_b2", 0.0))
    dict_set(opts, "sag", get_int(inputs, "sag", 0))
    dict_set(opts, "sag_scale", get_float(inputs, "sag_scale", 0.0))
    dict_set(opts, "clarity", 0.0)
    dict_set(opts, "sharpen", 0.0)
    dict_set(opts, "sharpen_radius", 0)
    dict_set(opts, "smart_sharpen", 0.0)
    dict_set(opts, "smart_sharpen_radius", 0)
    dict_set(opts, "edge_sharpen", 0.0)
    dict_set(opts, "edge_sharpen_radius", 0)
    dict_set(opts, "edge_sharpen_threshold", 0.0)
    dict_set(opts, "ad_model_path", "")
    dict_set(opts, "ad_prompt", "")
    dict_set(opts, "ad_negative_prompt", "")
    return opts


def sampler_output(inputs):
    output_dir = get_str(inputs, "output_dir", "/tmp/comfy_output")
    filename_prefix = get_str(inputs, "filename_prefix", "comfy")
    return [output_dir, output_dir + "/" + filename_prefix + ".png"]


def run_sampler(model: SDPipelineHandle, prompt: str, negative_prompt: str,
                width: int, height: int, opts, output_dir: str, output_path: str) -> int:
    rc = sd_ensure_directory(output_dir)
    if rc != 0:
        print("Failed to create output dir: " + output_dir)
        return -1
    return sd_generate_full(model.pipeline, prompt, negative_prompt,
                            width, height, opts, output_path)


def register_node(class_type: str, display: str, func_name: str, ret_types: list,
                  is_output: bool):
    meta = make_dict()
    dict_set(meta, "display", display)
    dict_set(meta, "function", func_name)
    dict_set(meta, "return_types", ret_types)
    dict_set(meta, "output_node", is_output)
    dict_set(NODE_CLASS_MAPPINGS, class_type, meta)
    dict_set(NODE_DISPLAY_NAMES, class_type, display)


def node_exists(class_type: str) -> bool:
    return dict_get(NODE_CLASS_MAPPINGS, class_type) is not None


def node_return_count(class_type: str) -> int:
    meta = dict_get(NODE_CLASS_MAPPINGS, class_type)
    if meta is None:
        return 0
    rt = dict_get(meta, "return_types")
    if rt is None:
        return 0
    return len(rt)


def get_int(inputs, key: str, default: int) -> int:
    v = dict_get(inputs, key)
    if v is None:
        return default
    return v


def get_float(inputs, key: str, default: float) -> float:
    v = dict_get(inputs, key)
    if v is None:
        return default
    return v


def get_str(inputs, key: str, default: str) -> str:
    v = dict_get(inputs, key)
    if v is None:
        return default
    return v


def checkpoint_loader_simple(inputs):
    ckpt_name = dict_get(inputs, "ckpt_name")
    if ckpt_name is None:
        ckpt_name = dict_get(inputs, "model_path")
    clip_l_name = dict_get(inputs, "clip_l_name")
    clip_g_name = dict_get(inputs, "clip_g_name")

    ckpt_path = resolve_model_path(ckpt_name)
    if clip_l_name is None:
        clip_l_path = ""
    else:
        clip_l_path = resolve_model_path(clip_l_name)
    if clip_g_name is None:
        clip_g_path = ""
    else:
        clip_g_path = resolve_model_path(clip_g_name)

    pipeline = sd_create()
    rc = sd_load(pipeline, ckpt_path, clip_l_path, clip_g_path, "", SD_WTYPE_AUTO, 8, 0)
    if rc != 0:
        print("SD checkpoint load failed, rc=" + string_of_int(rc))
        return (None, None, None)

    print("Checkpoint loaded, model version: " + sd_get_model_version_name(pipeline))
    handle = make_sd_pipeline_handle(pipeline, ckpt_path, clip_l_path, clip_g_path, "")
    return (handle, handle, handle)


register_node("CheckpointLoaderSimple", "Load Checkpoint",
              "checkpoint_loader_simple", ("MODEL", "CLIP", "VAE"), False)


def unet_loader(inputs):
    # 兼容 ComfyUI 的 UNETLoader：加载单个 diffusion/unet 文件为 pipeline。
    # 对一体化 checkpoint（含 CLIP/VAE）有效；分离组件请配合 DiffusionModelLoader。
    unet_name = dict_get(inputs, "unet_name")
    if unet_name is None:
        print("UNETLoader: unet_name is required")
        return (None,)
    path = resolve_model_path(unet_name)
    pipeline = sd_create()
    rc = sd_load(pipeline, path, "", "", "", SD_WTYPE_AUTO, 8, 0)
    if rc != 0:
        print("UNETLoader: load failed, rc=" + string_of_int(rc))
        return (None,)
    handle = make_sd_pipeline_handle(pipeline, path, "", "", "")
    return (handle,)


register_node("UNETLoader", "Load UNET",
              "unet_loader", ("MODEL",), False)


def vae_loader(inputs):
    name = get_str(inputs, "vae_name", "")
    if name == "":
        print("VAELoader: no vae_name provided")
        return (None,)
    return (resolve_model_path(name),)


register_node("VAELoader", "Load VAE",
              "vae_loader", ("VAE",), False)


def clip_loader(inputs):
    name = get_str(inputs, "clip_name", "")
    if name == "":
        print("CLIPLoader: no clip_name provided")
        return (None,)
    return (resolve_model_path(name),)


register_node("CLIPLoader", "Load CLIP",
              "clip_loader", ("CLIP",), False)


def dual_clip_loader(inputs):
    clip_name1 = get_str(inputs, "clip_name1", "")
    clip_name2 = get_str(inputs, "clip_name2", "")
    type_name = get_str(inputs, "type", "sdxl")
    # In this backend CLIP is bundled with the pipeline; the loader is a no-op
    # placeholder that satisfies the standard ComfyUI link topology.
    return (None,)


register_node("DualCLIPLoader", "Dual CLIP Loader",
              "dual_clip_loader", ("CLIP",), False)


def clip_text_encode(inputs):
    text = get_str(inputs, "text", "")
    clip = dict_get(inputs, "clip")
    # clip is ignored here because sd.cpp handles CLIP encode internally.
    return (Conditioning(text, "", "", 0.0),)


register_node("CLIPTextEncode", "CLIP Text Encode",
              "clip_text_encode", ("CONDITIONING",), False)


def empty_latent_image(inputs):
    width = get_int(inputs, "width", 1024)
    height = get_int(inputs, "height", 1024)
    batch_size = get_int(inputs, "batch_size", 1)
    return (LatentImage(width, height, batch_size, "", ""),)


register_node("EmptyLatentImage", "Empty Latent Image",
              "empty_latent_image", ("LATENT",), False)


def apply_latent_extras(model: SDPipelineHandle, inputs, latent: LatentImage):
    # 采样前的通用设置：batch / img2img / inpainting / ControlNet
    if latent is not None and latent.batch_size > 1:
        sd_set_batch_count(model.pipeline, latent.batch_size)
    else:
        sd_set_batch_count(model.pipeline, 1)
    if latent is not None and latent.image_path != "":
        sd_set_init_image(model.pipeline, latent.image_path, get_float(inputs, "denoise", 1.0))
    if latent is not None and latent.mask_path != "":
        sd_set_mask(model.pipeline, latent.mask_path)
    pos_c: Conditioning = dict_get(inputs, "positive")
    neg_c: Conditioning = dict_get(inputs, "negative")
    cn_image = ""
    cn_path = ""
    cn_strength = 0.0
    if pos_c is not None and pos_c.control_image_path != "":
        cn_image = pos_c.control_image_path
        cn_path = pos_c.control_net_path
        cn_strength = pos_c.control_strength
    elif neg_c is not None and neg_c.control_image_path != "":
        cn_image = neg_c.control_image_path
        cn_path = neg_c.control_net_path
        cn_strength = neg_c.control_strength
    if cn_image != "":
        sd_load_control_net(model.pipeline, cn_path)
        sd_set_control_image(model.pipeline, cn_image, cn_strength)


def ksampler(inputs):
    model: SDPipelineHandle = dict_get(inputs, "model")
    if model is None:
        print("KSampler: model is missing")
        return (None,)

    prompt = resolve_prompt_text(inputs, "positive", "prompt")
    negative_prompt = resolve_prompt_text(inputs, "negative", "negative_prompt")

    latent: LatentImage = dict_get(inputs, "latent_image")
    if latent is None:
        width = get_int(inputs, "width", 1024)
        height = get_int(inputs, "height", 1024)
    else:
        width = latent.width
        height = latent.height

    apply_latent_extras(model, inputs, latent)

    opts = parse_sampler_opts(inputs)
    out = sampler_output(inputs)
    rc = run_sampler(model, prompt, negative_prompt, width, height, opts, out[0], out[1])
    if rc != 0:
        print("SD generate failed, rc=" + string_of_int(rc))
        return (None,)

    return (out[1],)


register_node("KSampler", "KSampler",
              "ksampler", ("LATENT",), False)


def vae_decode(inputs):
    samples = dict_get(inputs, "samples")
    vae = dict_get(inputs, "vae")
    # In this backend VAE decode is already performed inside KSampler, so
    # this node just passes the already-decoded image path through.
    if samples is None:
        print("VAEDecode: no samples received")
        return (None,)
    return (samples,)


register_node("VAEDecode", "VAE Decode",
              "vae_decode", ("IMAGE",), False)


def vae_encode(inputs):
    # img2img：把参考图路径编码为 LATENT（sd.cpp 在采样时做 VAE encode）
    image_path = dict_get(inputs, "pixels")
    if image_path is None:
        image_path = get_str(inputs, "image", "")
    if image_path == "":
        print("VAEEncode: no image received")
        return (None,)
    return (LatentImage(0, 0, 1, image_path, ""),)


register_node("VAEEncode", "VAE Encode",
              "vae_encode", ("LATENT",), False)
register_node("VAEEncodeTiled", "VAE Encode (Tiled)",
              "vae_encode", ("LATENT",), False)


def load_image_mask(inputs):
    # ComfyUI 的 LoadImageMask：从图像提取通道作为 mask。
    # 本后端简化为返回图像路径，C++ 侧按灰度读取。
    image_path = get_str(inputs, "image", "")
    if image_path == "":
        print("LoadImageMask: no image provided")
        return (None,)
    return (image_path,)


register_node("LoadImageMask", "Load Image (as Mask)",
              "load_image_mask", ("MASK",), False)


def vae_encode_for_inpaint(inputs):
    image_path = dict_get(inputs, "pixels")
    if image_path is None:
        image_path = get_str(inputs, "image", "")
    mask_path = dict_get(inputs, "mask")
    if mask_path is None:
        mask_path = ""
    if image_path == "":
        print("VAEEncodeForInpaint: no image received")
        return (None,)
    return (LatentImage(0, 0, 1, image_path, mask_path),)


register_node("VAEEncodeForInpaint", "VAE Encode (for Inpainting)",
              "vae_encode_for_inpaint", ("LATENT",), False)


def diffusion_model_loader(inputs):
    diffusion_model_name = dict_get(inputs, "diffusion_model_name")
    llm_name = dict_get(inputs, "llm_name")
    vae_name = dict_get(inputs, "vae_name")

    if diffusion_model_name is None:
        print("DiffusionModelLoader: diffusion_model_name is required")
        return (None,)
    if llm_name is None:
        print("DiffusionModelLoader: llm_name is required")
        return (None,)

    diffusion_model_path = resolve_model_path(diffusion_model_name)
    llm_path = resolve_model_path(llm_name)
    vae_path = ""
    if vae_name is not None:
        vae_path = resolve_model_path(vae_name)

    pipeline = sd_create()
    rc = sd_load_ex(pipeline, "", "", "", vae_path, SD_WTYPE_AUTO, 8, 1,
                    diffusion_model_path, llm_path)
    if rc != 0:
        print("DiffusionModelLoader: load failed, rc=" + string_of_int(rc))
        return (None,)

    handle = make_sd_pipeline_handle(pipeline, diffusion_model_path, llm_path, "", vae_path)
    return (handle, handle, handle)


register_node("DiffusionModelLoader", "Load Diffusion Model (GGUF)",
              "diffusion_model_loader", ("MODEL", "CLIP", "VAE"), False)


def lora_loader(inputs):
    model: SDPipelineHandle = dict_get(inputs, "model")
    if model is None:
        print("LORALoader: model is missing")
        return (None,)
    lora_name = get_str(inputs, "lora_name", "")
    lora_scale = get_float(inputs, "lora_scale", 1.0)
    if lora_name == "":
        print("LORALoader: no lora_name provided, skipping")
        return (model,)
    lora_path = resolve_model_path(lora_name)
    pipeline = model.pipeline
    rc = sd_load_lora(pipeline, lora_path, lora_scale)
    if rc != 0:
        print("LORALoader: load failed for " + lora_path + ", rc=" + string_of_int(rc))
        return (None,)
    print("LORALoader: loaded " + lora_path + " scale=" + format_float(lora_scale, 2))
    return (model,)


register_node("LORALoader", "Load LoRA",
              "lora_loader", ("MODEL",), False)


def hires_fix(inputs):
    model: SDPipelineHandle = dict_get(inputs, "model")
    if model is None:
        print("HiResFix: model is missing")
        return (None,)

    prompt = resolve_prompt_text(inputs, "positive", "prompt")
    negative_prompt = resolve_prompt_text(inputs, "negative", "negative_prompt")

    target_width = get_int(inputs, "width", 1024)
    target_height = get_int(inputs, "height", 1024)

    seed = get_int(inputs, "seed", 0)
    if seed == 0:
        seed = -1

    opts = parse_sampler_opts(inputs)
    dict_set(opts, "cfg", get_float(inputs, "cfg", 2.5))
    dict_set(opts, "scheduler", get_str(inputs, "scheduler", "discrete"))
    dict_set(opts, "seed", seed)
    dict_set(opts, "vae_tiling", get_int(inputs, "vae_tiling", 1))
    dict_set(opts, "vae_tile_size", get_int(inputs, "vae_tile_size", 128))
    dict_set(opts, "hires_width", target_width)
    dict_set(opts, "hires_height", target_height)
    dict_set(opts, "hires_steps", get_int(inputs, "hires_steps", 45))
    dict_set(opts, "hires_strength", get_float(inputs, "hires_strength", 0.35))
    dict_set(opts, "freeu", get_int(inputs, "freeu", 1))
    dict_set(opts, "freeu_b1", get_float(inputs, "freeu_b1", 1.3))
    dict_set(opts, "freeu_b2", get_float(inputs, "freeu_b2", 1.4))
    dict_set(opts, "sag", get_int(inputs, "sag", 0))
    dict_set(opts, "sag_scale", get_float(inputs, "sag_scale", 1.0))
    dict_set(opts, "clarity", get_float(inputs, "clarity", 0.2))
    dict_set(opts, "sharpen", get_float(inputs, "sharpen", 0.3))
    dict_set(opts, "sharpen_radius", get_int(inputs, "sharpen_radius", 1))
    dict_set(opts, "smart_sharpen", get_float(inputs, "smart_sharpen", 0.5))
    dict_set(opts, "smart_sharpen_radius", get_int(inputs, "smart_sharpen_radius", 2))
    dict_set(opts, "edge_sharpen", get_float(inputs, "edge_sharpen", 1.5))
    dict_set(opts, "edge_sharpen_radius", get_int(inputs, "edge_sharpen_radius", 2))
    dict_set(opts, "edge_sharpen_threshold", get_float(inputs, "edge_sharpen_threshold", 0.3))

    out = sampler_output(inputs)
    rc = run_sampler(model, prompt, negative_prompt, 0, 0, opts, out[0], out[1])
    if rc != 0:
        print("HiResFix generate failed, rc=" + string_of_int(rc))
        return (None,)

    print("HiResFix: saved " + out[1])
    return (out[1],)


register_node("HiResFix", "HiRes Fix",
              "hires_fix", ("LATENT",), False)


def save_image(inputs):
    image_path = dict_get(inputs, "images")
    if image_path is None:
        print("SaveImage: no image path received")
        return (None,)
    print("Image saved to: " + image_path)
    return (image_path,)


register_node("SaveImage", "Save Image",
              "save_image", ("IMAGE",), True)


def adetailer(inputs):
    model: SDPipelineHandle = dict_get(inputs, "model")
    if model is None:
        print("ADetailer: model is missing")
        return (None,)

    prompt = resolve_prompt_text(inputs, "positive", "prompt")
    negative_prompt = resolve_prompt_text(inputs, "negative", "negative_prompt")

    width = get_int(inputs, "width", 1024)
    height = get_int(inputs, "height", 1024)

    opts = parse_sampler_opts(inputs)
    dict_set(opts, "ad_model_path", get_str(inputs, "ad_model_path", ""))
    dict_set(opts, "ad_prompt", get_str(inputs, "ad_prompt", prompt))
    dict_set(opts, "ad_negative_prompt", get_str(inputs, "ad_negative_prompt", negative_prompt))

    out = sampler_output(inputs)
    rc = run_sampler(model, prompt, negative_prompt, width, height, opts, out[0], out[1])
    if rc != 0:
        print("ADetailer generate failed, rc=" + string_of_int(rc))
        return (None,)

    return (out[1],)


register_node("ADetailer", "ADetailer",
              "adetailer", ("IMAGE",), False)


def ipadapter_apply(inputs):
    model: SDPipelineHandle = dict_get(inputs, "model")
    if model is None:
        print("IPAdapterApply: model is missing")
        return (None,)
    pipeline = model.pipeline

    ipadapter_obj: IPAdapterModel = dict_get(inputs, "ipadapter")
    clip_vision_obj: CLIPVisionModel = dict_get(inputs, "clip_vision")
    if ipadapter_obj is not None:
        ipadapter_model = ipadapter_obj.name
    else:
        ipadapter_model = get_str(inputs, "ipadapter_model", "")
    if clip_vision_obj is not None:
        clip_vision_model = clip_vision_obj.name
    else:
        clip_vision_model = get_str(inputs, "clip_vision_model", "")

    image_path = get_str(inputs, "image_path", "")
    weight = get_float(inputs, "weight", 1.0)

    if ipadapter_model == "" or clip_vision_model == "" or image_path == "":
        print("IPAdapterApply: empty model/image path, skipping")
        return (model,)

    ipadapter_path = resolve_model_path(ipadapter_model)
    clip_vision_path = resolve_model_path(clip_vision_model)

    rc = sd_set_ipadapter(pipeline, ipadapter_path, clip_vision_path, image_path, weight)
    if rc != 0:
        print("IPAdapterApply: set_ipadapter failed, rc=" + string_of_int(rc))
        return (None,)

    print("IPAdapterApply: model=" + ipadapter_model + " clip=" + clip_vision_model + " image=" + image_path + " weight=" + format_float(weight, 2))
    return (model,)


register_node("IPAdapterApply", "IPAdapter Apply",
              "ipadapter_apply", ("MODEL",), False)


def clip_vision_loader(inputs):
    name = get_str(inputs, "clip_name", "")
    if name == "":
        print("CLIPVisionLoader: no clip_name provided")
        return (None,)
    return (CLIPVisionModel(name),)


register_node("CLIPVisionLoader", "CLIP Vision Loader",
              "clip_vision_loader", ("CLIP_VISION",), False)


def ipadapter_model_loader(inputs):
    name = get_str(inputs, "ipadapter_file", "")
    if name == "":
        print("IPAdapterModelLoader: no ipadapter_file provided")
        return (None,)
    return (IPAdapterModel(name),)


register_node("IPAdapterModelLoader", "IPAdapter Model Loader",
              "ipadapter_model_loader", ("IPADAPTER",), False)


def load_image(inputs):
    image_path = get_str(inputs, "image", "")
    if image_path == "":
        print("LoadImage: no image path provided")
        return (None,)
    return (image_path,)


register_node("LoadImage", "Load Image",
              "load_image", ("IMAGE", "MASK"), False)
register_node("LoadImageOutput", "Load Image (from Outputs)",
              "load_image", ("IMAGE", "MASK"), False)


def image_scale(inputs):
    image_path = dict_get(inputs, "image")
    if image_path is None:
        print("ImageScale: no image received")
        return (None,)
    width = get_int(inputs, "width", 1024)
    height = get_int(inputs, "height", 1024)
    out = "/tmp/comfycli_scaled_" + string_of_int(width) + "x" + string_of_int(height) + ".png"
    rc = sd_resize_image(image_path, out, width, height)
    if rc != 0:
        print("ImageScale: resize failed, rc=" + string_of_int(rc))
        return (None,)
    return (out,)


register_node("ImageScale", "Image Scale",
              "image_scale", ("IMAGE",), False)


def image_scale_by(inputs):
    image_path = dict_get(inputs, "image")
    if image_path is None:
        print("ImageScaleBy: no image received")
        return (None,)
    scale_by = get_float(inputs, "scale_by", 1.0)
    out = "/tmp/comfycli_scaled_by.png"
    rc = sd_scale_image(image_path, out, scale_by)
    if rc != 0:
        print("ImageScaleBy: scale failed, rc=" + string_of_int(rc))
        return (None,)
    return (out,)


register_node("ImageScaleBy", "Image Scale By",
              "image_scale_by", ("IMAGE",), False)


def image_invert(inputs):
    image_path = dict_get(inputs, "image")
    if image_path is None:
        print("ImageInvert: no image received")
        return (None,)
    out = "/tmp/comfycli_inverted.png"
    rc = sd_invert_image(image_path, out)
    if rc != 0:
        print("ImageInvert: invert failed, rc=" + string_of_int(rc))
        return (None,)
    return (out,)


register_node("ImageInvert", "Invert Image",
              "image_invert", ("IMAGE",), False)


def empty_image(inputs):
    width = get_int(inputs, "width", 512)
    height = get_int(inputs, "height", 512)
    color = get_int(inputs, "color", 0)
    r = color // 65536
    g = (color // 256) % 256
    b = color % 256
    out = "/tmp/comfycli_empty_" + string_of_int(width) + "x" + string_of_int(height) + ".png"
    rc = sd_make_solid_image(out, width, height, r, g, b)
    if rc != 0:
        print("EmptyImage: failed, rc=" + string_of_int(rc))
        return (None,)
    return (out,)


register_node("EmptyImage", "Empty Image",
              "empty_image", ("IMAGE",), False)


def image_pad_for_outpaint(inputs):
    image_path = dict_get(inputs, "image")
    if image_path is None:
        print("ImagePadForOutpaint: no image received")
        return (None,)
    left = get_int(inputs, "left", 0)
    top = get_int(inputs, "top", 0)
    right = get_int(inputs, "right", 0)
    bottom = get_int(inputs, "bottom", 0)
    out = "/tmp/comfycli_padded.png"
    rc = sd_pad_image(image_path, out, left, top, right, bottom, 0, 0, 0)
    if rc != 0:
        print("ImagePadForOutpaint: failed, rc=" + string_of_int(rc))
        return (None,)
    return (out,)


register_node("ImagePadForOutpaint", "Pad Image for Outpainting",
              "image_pad_for_outpaint", ("IMAGE", "MASK"), False)


def image_blur(inputs):
    image = dict_get(inputs, "image")
    if image is None:
        print("ImageBlur: no image received")
        return (None,)
    sigma = get_float(inputs, "sigma", 1.0)
    out = "/tmp/comfycli_blur.png"
    rc = sd_blur_image(image, out, sigma)
    if rc != 0:
        print("ImageBlur: failed, rc=" + string_of_int(rc))
        return (None,)
    return (out,)


register_node("ImageBlur", "Blur Image",
              "image_blur", ("IMAGE",), False)


def image_batch(inputs):
    i1 = dict_get(inputs, "image1")
    i2 = dict_get(inputs, "image2")
    if i1 is None or i2 is None:
        print("ImageBatch: need image1 and image2")
        return (None,)
    out = "/tmp/comfycli_batch.png"
    rc = sd_batch_images(i1, i2, out)
    if rc != 0:
        print("ImageBatch: failed, rc=" + string_of_int(rc))
        return (None,)
    return (out,)


register_node("ImageBatch", "Batch Images",
              "image_batch", ("IMAGE",), False)


def image_composite_masked(inputs):
    dest = dict_get(inputs, "destination")
    src = dict_get(inputs, "source")
    if dest is None or src is None:
        print("ImageCompositeMasked: need destination and source")
        return (None,)
    mask = dict_get(inputs, "mask")
    if mask is None:
        mask = ""
    x = get_int(inputs, "x", 0)
    y = get_int(inputs, "y", 0)
    out = "/tmp/comfycli_composite.png"
    rc = sd_composite_masked(dest, src, mask, out, x, y)
    if rc != 0:
        print("ImageCompositeMasked: failed, rc=" + string_of_int(rc))
        return (None,)
    return (out,)


register_node("ImageCompositeMasked", "Image Composite Masked",
              "image_composite_masked", ("IMAGE",), False)


def image_crop(inputs):
    image = dict_get(inputs, "image")
    if image is None:
        print("ImageCrop: no image received")
        return (None,)
    x = get_int(inputs, "x", 0)
    y = get_int(inputs, "y", 0)
    width = get_int(inputs, "width", 512)
    height = get_int(inputs, "height", 512)
    out = "/tmp/comfycli_crop.png"
    rc = sd_crop_image(image, out, x, y, width, height)
    if rc != 0:
        print("ImageCrop: failed, rc=" + string_of_int(rc))
        return (None,)
    return (out,)


register_node("ImageCrop", "Crop Image",
              "image_crop", ("IMAGE",), False)


def image_to_mask(inputs):
    image = dict_get(inputs, "image")
    if image is None:
        print("ImageToMask: no image received")
        return (None,)
    return (image,)


register_node("ImageToMask", "Convert Image to Mask",
              "image_to_mask", ("MASK",), False)


def mask_to_image(inputs):
    mask = dict_get(inputs, "mask")
    if mask is None:
        print("MaskToImage: no mask received")
        return (None,)
    return (mask,)


register_node("MaskToImage", "Convert Mask to Image",
              "mask_to_image", ("IMAGE",), False)


def clip_vision_encode(inputs):
    # IPAdapter 后端直接吃图片路径，此节点透传 image
    image = dict_get(inputs, "image")
    if image is None:
        print("CLIPVisionEncode: no image received")
        return (None,)
    return (image,)


register_node("CLIPVisionEncode", "CLIP Vision Encode",
              "clip_vision_encode", ("CLIP_VISION_OUTPUT",), False)


def preview_image(inputs):
    image_path = dict_get(inputs, "images")
    if image_path is None:
        print("PreviewImage: no image received")
        return (None,)
    return (image_path,)


register_node("PreviewImage", "Preview Image",
              "preview_image", ("IMAGE",), True)


def clip_set_last_layer(inputs):
    clip: SDPipelineHandle = dict_get(inputs, "clip")
    if clip is None:
        return (None,)
    layer = get_int(inputs, "stop_at_clip_layer", -1)
    # ComfyUI 的 stop_at_clip_layer 为负数（如 -2）；sd.cpp 的 clip_skip 为正数
    skip = -layer
    if skip < 0:
        skip = 1
    sd_set_clip_skip(clip.pipeline, skip)
    return (clip,)


register_node("CLIPSetLastLayer", "CLIP Set Last Layer",
              "clip_set_last_layer", ("CLIP",), False)


def conditioning_combine(inputs):
    text = merge_conditioning_text(
        conditioning_text(dict_get(inputs, "conditioning_1")),
        conditioning_text(dict_get(inputs, "conditioning_2")))
    return (Conditioning(text, "", "", 0.0),)


register_node("ConditioningCombine", "Conditioning Combine",
              "conditioning_combine", ("CONDITIONING",), False)


def conditioning_concat(inputs):
    text = merge_conditioning_text(
        conditioning_text(dict_get(inputs, "conditioning_to")),
        conditioning_text(dict_get(inputs, "conditioning_from")))
    return (Conditioning(text, "", "", 0.0),)


register_node("ConditioningConcat", "Conditioning Concat",
              "conditioning_concat", ("CONDITIONING",), False)


def conditioning_average(inputs):
    c_to = conditioning_text(dict_get(inputs, "conditioning_to"))
    c_from = conditioning_text(dict_get(inputs, "conditioning_from"))
    strength = get_float(inputs, "conditioning_to_strength", 0.5)
    # Simple strength-aware combination: stronger text goes first.
    if strength >= 0.5:
        return (Conditioning(merge_conditioning_text(c_to, c_from), "", "", 0.0),)
    return (Conditioning(merge_conditioning_text(c_from, c_to), "", "", 0.0),)


register_node("ConditioningAverage", "Conditioning Average",
              "conditioning_average", ("CONDITIONING",), False)


def latent_upscale(inputs):
    latent: LatentImage = dict_get(inputs, "samples")
    if latent is None:
        print("LatentUpscale: no samples received")
        return (None,)
    # In this simplified backend, width/height directly replace latent dimensions.
    width = get_int(inputs, "width", latent.width)
    height = get_int(inputs, "height", latent.height)
    batch_size = latent.batch_size
    return (LatentImage(width, height, batch_size, latent.image_path, latent.mask_path),)


register_node("LatentUpscale", "Latent Upscale",
              "latent_upscale", ("LATENT",), False)


def latent_crop(inputs):
    latent: LatentImage = dict_get(inputs, "samples")
    if latent is None:
        print("LatentCrop: no samples received")
        return (None,)
    width = get_int(inputs, "width", latent.width)
    height = get_int(inputs, "height", latent.height)
    batch_size = latent.batch_size
    return (LatentImage(width, height, batch_size, latent.image_path, latent.mask_path),)


register_node("LatentCrop", "Latent Crop",
              "latent_crop", ("LATENT",), False)


def reroute(inputs):
    val = dict_get(inputs, "anything")
    return (val,)


register_node("Reroute", "Reroute",
              "reroute", ("*",), False)


def ksampler_advanced(inputs):
    model: SDPipelineHandle = dict_get(inputs, "model")
    if model is None:
        print("KSamplerAdvanced: model is missing")
        return (None,)

    prompt = resolve_prompt_text(inputs, "positive", "prompt")
    negative_prompt = resolve_prompt_text(inputs, "negative", "negative_prompt")

    latent: LatentImage = dict_get(inputs, "latent_image")
    if latent is None:
        width = get_int(inputs, "width", 1024)
        height = get_int(inputs, "height", 1024)
    else:
        width = latent.width
        height = latent.height

    apply_latent_extras(model, inputs, latent)

    opts = parse_sampler_opts(inputs)
    out = sampler_output(inputs)
    rc = run_sampler(model, prompt, negative_prompt, width, height, opts, out[0], out[1])
    if rc != 0:
        print("KSamplerAdvanced generate failed, rc=" + string_of_int(rc))
        return (None,)

    return (out[1],)


register_node("KSamplerAdvanced", "KSampler Advanced",
              "ksampler_advanced", ("LATENT", "IMAGE"), False)


def conditioning_zero_out(inputs):
    # 空条件（ComfyUI ConditioningZeroOut）
    return (Conditioning("", "", "", 0.0),)


register_node("ConditioningZeroOut", "Conditioning Zero Out",
              "conditioning_zero_out", ("CONDITIONING",), False)


def controlnet_loader(inputs):
    name = get_str(inputs, "control_net_name", "")
    if name == "":
        print("ControlNetLoader: no control_net_name provided")
        return (None,)
    return (ControlNetModel(name),)


register_node("ControlNetLoader", "Load ControlNet",
              "controlnet_loader", ("CONTROL_NET",), False)
register_node("DiffControlNetLoader", "Load Diff ControlNet",
              "controlnet_loader", ("CONTROL_NET",), False)


def controlnet_apply(inputs):
    c: Conditioning = dict_get(inputs, "conditioning")
    if c is None:
        print("ControlNetApply: no conditioning received")
        return (None,)
    cn: ControlNetModel = dict_get(inputs, "control_net")
    cn_path = ""
    if cn is not None:
        cn_path = cn.name
    image_path = get_str(inputs, "image", "")
    strength = get_float(inputs, "strength", 1.0)
    return (Conditioning(c.text, cn_path, image_path, strength),)


register_node("ControlNetApply", "Apply ControlNet",
              "controlnet_apply", ("CONDITIONING",), False)
register_node("CheckpointLoader", "Load Checkpoint",
              "checkpoint_loader_simple", ("MODEL", "CLIP", "VAE"), False)
register_node("LoraLoader", "Load LoRA",
              "lora_loader", ("MODEL", "CLIP"), False)
register_node("LoraLoaderModelOnly", "Load LoRA (Model Only)",
              "lora_loader", ("MODEL",), False)
register_node("LoraLoaderBypass", "LoraLoaderBypass",
              "lora_loader", ("MODEL", "CLIP"), False)
register_node("LoraLoaderBypassModelOnly", "LoraLoaderBypassModelOnly",
              "lora_loader", ("MODEL",), False)
register_node("VAEDecodeTiled", "VAE Decode (Tiled)",
              "vae_decode", ("IMAGE",), False)


def conditioning_passthrough(inputs):
    # area/mask/timestep 等条件修饰在 sd.cpp 后端无对应能力，透传原条件
    c: Conditioning = dict_get(inputs, "conditioning")
    if c is None:
        return (None,)
    return (c,)


register_node("ConditioningSetArea", "ConditioningSetArea",
              "conditioning_passthrough", ("CONDITIONING",), False)
register_node("ConditioningSetAreaPercentage", "ConditioningSetAreaPercentage",
              "conditioning_passthrough", ("CONDITIONING",), False)
register_node("ConditioningSetAreaStrength", "ConditioningSetAreaStrength",
              "conditioning_passthrough", ("CONDITIONING",), False)
register_node("ConditioningSetMask", "ConditioningSetMask",
              "conditioning_passthrough", ("CONDITIONING",), False)
register_node("ConditioningMultiply", "ConditioningMultiply",
              "conditioning_passthrough", ("CONDITIONING",), False)
register_node("ConditioningSetTimestepRange", "ConditioningSetTimestepRange",
              "conditioning_passthrough", ("CONDITIONING",), False)


def latent_rotate(inputs):
    latent: LatentImage = dict_get(inputs, "samples")
    if latent is None:
        return (None,)
    rotation = string_to_int(get_str(inputs, "rotation", "0"))
    w = latent.width
    h = latent.height
    if rotation == 90 or rotation == 270:
        w = latent.height
        h = latent.width
    image_path = latent.image_path
    if image_path != "":
        out = "/tmp/comfycli_rotate.png"
        if sd_rotate_image(image_path, out, rotation) == 0:
            image_path = out
    return (LatentImage(w, h, latent.batch_size, image_path, latent.mask_path),)


register_node("LatentRotate", "LatentRotate", "latent_rotate", ("LATENT",), False)


def latent_flip(inputs):
    latent: LatentImage = dict_get(inputs, "samples")
    if latent is None:
        return (None,)
    method = 0
    if get_str(inputs, "flip_method", "x") == "y":
        method = 1
    image_path = latent.image_path
    if image_path != "":
        out = "/tmp/comfycli_flip.png"
        if sd_flip_image(image_path, out, method) == 0:
            image_path = out
    return (LatentImage(latent.width, latent.height, latent.batch_size, image_path, latent.mask_path),)


register_node("LatentFlip", "LatentFlip", "latent_flip", ("LATENT",), False)


def latent_composite(inputs):
    to_lat: LatentImage = dict_get(inputs, "samples_to")
    from_lat: LatentImage = dict_get(inputs, "samples_from")
    if to_lat is None or from_lat is None:
        return (None,)
    if to_lat.image_path != "" and from_lat.image_path != "":
        out = "/tmp/comfycli_latcomposite.png"
        if sd_composite_masked(to_lat.image_path, from_lat.image_path, "", out,
                               get_int(inputs, "x", 0), get_int(inputs, "y", 0)) == 0:
            return (LatentImage(to_lat.width, to_lat.height, to_lat.batch_size, out, to_lat.mask_path),)
    return (to_lat,)


register_node("LatentComposite", "LatentComposite", "latent_composite", ("LATENT",), False)


def latent_blend(inputs):
    l1: LatentImage = dict_get(inputs, "samples1")
    l2: LatentImage = dict_get(inputs, "samples2")
    if l1 is None or l2 is None:
        return (None,)
    factor = get_float(inputs, "blend_factor", 0.5)
    if l1.image_path != "" and l2.image_path != "":
        out = "/tmp/comfycli_latblend.png"
        if sd_blend_images(l1.image_path, l2.image_path, out, factor) == 0:
            return (LatentImage(l1.width, l1.height, l1.batch_size, out, l1.mask_path),)
    return (l1,)


register_node("LatentBlend", "LatentBlend", "latent_blend", ("LATENT",), False)


def repeat_latent_batch(inputs):
    latent: LatentImage = dict_get(inputs, "samples")
    if latent is None:
        return (None,)
    amount = get_int(inputs, "amount", 1)
    return (LatentImage(latent.width, latent.height, latent.batch_size * amount,
                        latent.image_path, latent.mask_path),)


register_node("RepeatLatentBatch", "RepeatLatentBatch",
              "repeat_latent_batch", ("LATENT",), False)


def latent_from_batch(inputs):
    latent: LatentImage = dict_get(inputs, "samples")
    if latent is None:
        return (None,)
    length = get_int(inputs, "length", 1)
    return (LatentImage(latent.width, latent.height, length,
                        latent.image_path, latent.mask_path),)


register_node("LatentFromBatch", "LatentFromBatch",
              "latent_from_batch", ("LATENT",), False)


def set_latent_noise_mask(inputs):
    latent: LatentImage = dict_get(inputs, "samples")
    if latent is None:
        return (None,)
    mask = dict_get(inputs, "mask")
    if mask is None:
        mask = ""
    return (LatentImage(latent.width, latent.height, latent.batch_size,
                        latent.image_path, mask),)


register_node("SetLatentNoiseMask", "SetLatentNoiseMask",
              "set_latent_noise_mask", ("LATENT",), False)


def style_model_loader(inputs):
    name = get_str(inputs, "style_model_name", "")
    return (name,)


register_node("StyleModelLoader", "Load Style Model",
              "style_model_loader", ("STYLE_MODEL",), False)


def style_model_apply(inputs):
    # 风格模型在 sd.cpp 后端无对应能力，透传条件
    c: Conditioning = dict_get(inputs, "conditioning")
    if c is None:
        return (None,)
    return (c,)


register_node("StyleModelApply", "Apply Style Model",
              "style_model_apply", ("CONDITIONING",), False)


def unclip_conditioning(inputs):
    c: Conditioning = dict_get(inputs, "conditioning")
    if c is None:
        return (None,)
    return (c,)


register_node("unCLIPConditioning", "Apply unCLIP Conditioning",
              "unclip_conditioning", ("CONDITIONING",), False)


def gligen_loader(inputs):
    name = get_str(inputs, "gligen_name", "")
    return (name,)


register_node("GLIGENLoader", "Load GLIGEN",
              "gligen_loader", ("GLIGEN",), False)


def gligen_textbox_apply(inputs):
    c: Conditioning = dict_get(inputs, "conditioning_to")
    if c is None:
        return (None,)
    return (c,)


register_node("GLIGENTextBoxApply", "GLIGEN Textbox Apply",
              "gligen_textbox_apply", ("CONDITIONING",), False)


def latent_upscale_by(inputs):
    latent: LatentImage = dict_get(inputs, "samples")
    if latent is None:
        print("LatentUpscaleBy: no samples received")
        return (None,)
    scale_by = get_float(inputs, "scale_by", 1.0)
    width = int(latent.width * scale_by)
    height = int(latent.height * scale_by)
    return (LatentImage(width, height, latent.batch_size, latent.image_path, latent.mask_path),)


register_node("LatentUpscaleBy", "Latent Upscale By",
              "latent_upscale_by", ("LATENT",), False)


def controlnet_apply_advanced(inputs):
    pos: Conditioning = dict_get(inputs, "positive")
    neg: Conditioning = dict_get(inputs, "negative")
    cn: ControlNetModel = dict_get(inputs, "control_net")
    cn_path = ""
    if cn is not None:
        cn_path = cn.name
    image_path = get_str(inputs, "image", "")
    strength = get_float(inputs, "strength", 1.0)
    if pos is None:
        return (None, None)
    new_pos = Conditioning(pos.text, cn_path, image_path, strength)
    return (new_pos, neg)


register_node("ControlNetApplyAdvanced", "Apply ControlNet (Advanced)",
              "controlnet_apply_advanced", ("CONDITIONING", "CONDITIONING"), False)


def inpaint_model_conditioning(inputs):
    pos = dict_get(inputs, "positive")
    neg = dict_get(inputs, "negative")
    image_path = dict_get(inputs, "pixels")
    if image_path is None:
        image_path = ""
    mask_path = dict_get(inputs, "mask")
    if mask_path is None:
        mask_path = ""
    latent = LatentImage(0, 0, 1, image_path, mask_path)
    return (pos, neg, latent)


register_node("InpaintModelConditioning", "InpaintModelConditioning",
              "inpaint_model_conditioning", ("CONDITIONING", "CONDITIONING", "LATENT"), False)


def preview_any(inputs):
    return (dict_get(inputs, "source"),)


register_node("PreviewAny", "Preview Any",
              "preview_any", ("*",), True)


register_node("CLIPMergeSimple", "Merge CLIP (Simple)",
              "clip_merge_simple", ("CLIP",), False)
register_node("CLIPMergeAdd", "Merge CLIP (Add)",
              "clip_merge_add", ("CLIP",), False)
register_node("CLIPMergeSubtract", "Merge CLIP (Subtract)",
              "clip_merge_subtract", ("CLIP",), False)


def model_passthrough(inputs):
    # 采样/模型配置类节点：sd.cpp 内部按模型自动选择调度，这里透传 model
    m = dict_get(inputs, "model")
    if m is None:
        return (None,)
    return (m,)


def model_sampling_flow(inputs):
    # Flux/SD3/AuraFlow 的 shift 参数 → sd.cpp flow_shift
    m: SDPipelineHandle = dict_get(inputs, "model")
    if m is None:
        return (None,)
    shift = get_float(inputs, "shift", 0.0)
    if shift <= 0.0:
        shift = get_float(inputs, "max_shift", 0.0)
    if shift > 0.0:
        sd_set_flow_shift(m.pipeline, shift)
    return (m,)


register_node("ModelSamplingDiscrete", "ModelSamplingDiscrete",
              "model_passthrough", ("MODEL",), False)
register_node("ModelSamplingFlux", "ModelSamplingFlux",
              "model_sampling_flow", ("MODEL",), False)
register_node("ModelSamplingSD3", "ModelSamplingSD3",
              "model_sampling_flow", ("MODEL",), False)
def model_sampling_sigma_range(inputs):
    m: SDPipelineHandle = dict_get(inputs, "model")
    if m is None:
        return (None,)
    sigma_min = get_float(inputs, "sigma_min", 0.002)
    sigma_max = get_float(inputs, "sigma_max", 120.0)
    sd_set_sigma_range(m.pipeline, sigma_min, sigma_max)
    return (m,)


register_node("ModelSamplingContinuousEDM", "ModelSamplingContinuousEDM",
              "model_sampling_sigma_range", ("MODEL",), False)
register_node("ModelSamplingContinuousV", "ModelSamplingContinuousV",
              "model_sampling_sigma_range", ("MODEL",), False)
register_node("ModelSamplingAuraFlow", "ModelSamplingAuraFlow",
              "model_sampling_flow", ("MODEL",), False)
register_node("ModelSamplingStableCascade", "ModelSamplingStableCascade",
              "model_passthrough", ("MODEL",), False)
def rescale_cfg(inputs):
    m: SDPipelineHandle = dict_get(inputs, "model")
    if m is None:
        return (None,)
    multiplier = get_float(inputs, "multiplier", 0.7)
    sd_set_rescale_cfg(m.pipeline, multiplier)
    return (m,)


register_node("RescaleCFG", "RescaleCFG",
              "rescale_cfg", ("MODEL",), False)


def model_compute_dtype(inputs):
    m: SDPipelineHandle = dict_get(inputs, "model")
    if m is None:
        return (None,)
    dtype = get_str(inputs, "dtype", "default")
    wtype = -1
    if dtype == "fp32":
        wtype = 0
    elif dtype == "fp16":
        wtype = 1
    elif dtype == "bf16":
        wtype = 30
    if wtype >= 0:
        sd_set_wtype(m.pipeline, wtype)
    return (m,)


register_node("ModelComputeDtype", "ModelComputeDtype",
              "model_compute_dtype", ("MODEL",), False)
def model_attention_backend(inputs):
    m: SDPipelineHandle = dict_get(inputs, "model")
    if m is None:
        return (None,)
    backend = get_str(inputs, "backend", "default")
    # sd.cpp 仅暴露 diffusion flash attention 开关
    sd_set_flash_attn(m.pipeline, backend == "flash_attn")
    return (m,)


register_node("ModelAttentionBackend", "ModelAttentionBackend",
              "model_attention_backend", ("MODEL",), False)
register_node("ModelNoiseScale", "ModelNoiseScale",
              "model_passthrough", ("MODEL",), False)


def save_latent(inputs):
    latent: LatentImage = dict_get(inputs, "samples")
    if latent is None:
        print("SaveLatent: no samples received")
        return (None,)
    output_dir = get_str(inputs, "output_dir", "/tmp/comfy_output")
    prefix = get_str(inputs, "filename_prefix", "latent")
    sd_ensure_directory(output_dir)
    path = output_dir + "/" + prefix + ".latent"
    d = make_dict()
    dict_set(d, "width", latent.width)
    dict_set(d, "height", latent.height)
    dict_set(d, "batch_size", latent.batch_size)
    dict_set(d, "image_path", latent.image_path)
    dict_set(d, "mask_path", latent.mask_path)
    fp = file_open(path, "w")
    file_write(fp, json_dumps(d))
    file_close(fp)
    print("Latent saved to: " + path)
    return (path,)


register_node("SaveLatent", "Save Latent", "save_latent", ("LATENT",), True)


def load_latent(inputs):
    latent_name = get_str(inputs, "latent", "")
    if latent_name == "":
        print("LoadLatent: no latent provided")
        return (None,)
    path = resolve_model_path(latent_name)
    fp = file_open(path, "r")
    content = file_read_all(fp)
    file_close(fp)
    d = parse_json(content)
    return (LatentImage(get_int(d, "width", 1024), get_int(d, "height", 1024),
                        get_int(d, "batch_size", 1),
                        get_str(d, "image_path", ""), get_str(d, "mask_path", "")),)


register_node("LoadLatent", "Load Latent", "load_latent", ("LATENT",), False)


def save_component(inputs, key: str, default_prefix: str):
    handle: SDPipelineHandle = dict_get(inputs, key)
    if handle is None:
        print("Save: input '" + key + "' missing")
        return ("",)
    src = handle.model_path
    if key == "vae" and handle.vae_path != "":
        src = handle.vae_path
    if key == "clip" and handle.clip_l_path != "":
        src = handle.clip_l_path
    if src == "":
        print("Save: no source file for '" + key + "'")
        return ("",)
    output_dir = get_str(inputs, "output_dir", "/tmp/comfy_output")
    prefix = get_str(inputs, "filename_prefix", default_prefix)
    sd_ensure_directory(output_dir)
    dst = output_dir + "/" + prefix + ".safetensors"
    rc = torch_std_copy_file(src, dst)
    if rc != 0:
        print("Save: copy failed, rc=" + string_of_int(rc) + " (libtorch_std_helper.so missing?)")
        return ("",)
    print("Saved to: " + dst)
    return (dst,)


def checkpoint_save(inputs):
    return save_component(inputs, "model", "checkpoint")


def vae_save(inputs):
    return save_component(inputs, "vae", "vae")


def clip_save(inputs):
    return save_component(inputs, "clip", "clip")


def model_save(inputs):
    return save_component(inputs, "model", "model")


def webcam_capture(inputs):
    # 命令行环境无摄像头设备
    print("WebcamCapture: not available in CLI environment")
    return (None,)


register_node("CheckpointSave", "CheckpointSave", "checkpoint_save", ("*",), True)
register_node("VAESave", "VAESave", "vae_save", ("*",), True)
register_node("CLIPSave", "CLIPSave", "clip_save", ("*",), True)
register_node("ModelSave", "ModelSave", "model_save", ("*",), True)
register_node("ImageOnlyCheckpointSave", "ImageOnlyCheckpointSave", "checkpoint_save", ("*",), True)
register_node("WebcamCapture", "WebcamCapture", "webcam_capture", ("IMAGE",), False)


@dataclass
class MergeRatios:
    prefixes: str
    ratios: str
    default_ratio: float


def merge_dir() -> str:
    d = os_getenv("COMFYCLI_MERGE_DIR")
    if d is None or d == "":
        d = "/tmp/comfycli_merged"
    sd_ensure_directory(d)
    return d


def merge_output_path() -> str:
    return merge_dir() + "/merged_" + string_of_int(random_int()) + ".safetensors"


def collect_block_ratios(inputs) -> MergeRatios:
    keys = dict_keys(inputs)
    prefixes: str = ""
    ratios: str = ""
    default_ratio: float = 1.0
    first: bool = True
    i: int = 0
    n: int = list_length(keys)
    while i < n:
        k: str = keys[i]
        if k != "model1" and k != "model2":
            v = dict_get(inputs, k)
            if v is None:
                v = 0.0
            if first:
                default_ratio = v
                first = False
            else:
                prefixes = prefixes + ","
                ratios = ratios + ","
            prefixes = prefixes + k
            ratios = ratios + format_float(v, 6)
        i = i + 1
    return MergeRatios(prefixes, ratios, default_ratio)


def merge_models(p1: str, p2: str, mode: int, prefixes: str, ratios: str, default_ratio: float, strip_prefix: str):
    if p1 == "" or p2 == "":
        print("Merge: source checkpoint path missing")
        return (None,)
    d1 = torch_std_safetensors_load(p1)
    d2 = torch_std_safetensors_load(p2)
    if d1 is None or d2 is None:
        print("Merge: failed to load safetensors (libtorch_std_helper.so missing?)")
        return (None,)
    merged = torch_std_safetensors_merge(d1, d2, mode, prefixes, ratios, default_ratio, strip_prefix)
    torch_std_safetensors_free(d1)
    torch_std_safetensors_free(d2)
    if merged is None:
        print("Merge: merge failed")
        return (None,)
    out_path = merge_output_path()
    rc = torch_std_safetensors_save(merged, out_path)
    torch_std_safetensors_free(merged)
    if rc != 0:
        print("Merge: save failed, rc=" + string_of_int(rc))
        return (None,)
    pipeline = sd_create()
    rc2 = sd_load(pipeline, out_path, "", "", "", SD_WTYPE_AUTO, 8, 0)
    if rc2 != 0:
        print("Merge: failed to load merged checkpoint, rc=" + string_of_int(rc2))
        return (None,)
    print("Merged checkpoint saved to: " + out_path)
    return (make_sd_pipeline_handle(pipeline, out_path, "", "", ""),)


def model_merge_simple(inputs):
    m1: SDPipelineHandle = dict_get(inputs, "model1")
    m2: SDPipelineHandle = dict_get(inputs, "model2")
    if m1 is None or m2 is None:
        return (None,)
    ratio = get_float(inputs, "ratio", 1.0)
    return merge_models(m1.model_path, m2.model_path, 0, "", "", ratio, "diffusion_model.")


def model_merge_add(inputs):
    m1: SDPipelineHandle = dict_get(inputs, "model1")
    m2: SDPipelineHandle = dict_get(inputs, "model2")
    if m1 is None or m2 is None:
        return (None,)
    return merge_models(m1.model_path, m2.model_path, 1, "", "", 0.0, "diffusion_model.")


def model_merge_subtract(inputs):
    m1: SDPipelineHandle = dict_get(inputs, "model1")
    m2: SDPipelineHandle = dict_get(inputs, "model2")
    if m1 is None or m2 is None:
        return (None,)
    multiplier = get_float(inputs, "multiplier", 1.0)
    return merge_models(m1.model_path, m2.model_path, 2, "", "", multiplier, "diffusion_model.")


def model_merge_blocks(inputs):
    m1: SDPipelineHandle = dict_get(inputs, "model1")
    m2: SDPipelineHandle = dict_get(inputs, "model2")
    if m1 is None or m2 is None:
        return (None,)
    br = collect_block_ratios(inputs)
    return merge_models(m1.model_path, m2.model_path, 0, br.prefixes, br.ratios, br.default_ratio, "diffusion_model.")


def clip_merge_simple(inputs):
    c1: SDPipelineHandle = dict_get(inputs, "clip1")
    c2: SDPipelineHandle = dict_get(inputs, "clip2")
    if c1 is None or c2 is None:
        return (None,)
    ratio = get_float(inputs, "ratio", 1.0)
    return merge_models(c1.model_path, c2.model_path, 0, "", "", ratio,
                        "conditioner.embedders.,cond_stage_model.,text_encoders.")


def clip_merge_add(inputs):
    c1: SDPipelineHandle = dict_get(inputs, "clip1")
    c2: SDPipelineHandle = dict_get(inputs, "clip2")
    if c1 is None or c2 is None:
        return (None,)
    return merge_models(c1.model_path, c2.model_path, 1, "", "", 0.0,
                        "conditioner.embedders.,cond_stage_model.,text_encoders.")


def clip_merge_subtract(inputs):
    c1: SDPipelineHandle = dict_get(inputs, "clip1")
    c2: SDPipelineHandle = dict_get(inputs, "clip2")
    if c1 is None or c2 is None:
        return (None,)
    multiplier = get_float(inputs, "multiplier", 1.0)
    return merge_models(c1.model_path, c2.model_path, 2, "", "", multiplier,
                        "conditioner.embedders.,cond_stage_model.,text_encoders.")


register_node("ModelMergeSimple", "ModelMergeSimple", "model_merge_simple", ("MODEL",), False)
register_node("ModelMergeBlocks", "ModelMergeBlocks", "model_merge_blocks", ("MODEL",), False)
register_node("ModelMergeAdd", "ModelMergeAdd", "model_merge_add", ("MODEL",), False)
register_node("ModelMergeSubtract", "ModelMergeSubtract", "model_merge_subtract", ("MODEL",), False)
register_node("ModelMergeSD1", "ModelMergeSD1", "model_merge_blocks", ("MODEL",), False)
register_node("ModelMergeSD2", "ModelMergeSD2", "model_merge_blocks", ("MODEL",), False)
register_node("ModelMergeSDXL", "ModelMergeSDXL", "model_merge_blocks", ("MODEL",), False)
register_node("ModelMergeSD3_2B", "ModelMergeSD3_2B", "model_merge_blocks", ("MODEL",), False)
register_node("ModelMergeSD35_Large", "ModelMergeSD35_Large", "model_merge_blocks", ("MODEL",), False)
register_node("ModelMergeFlux1", "ModelMergeFlux1", "model_merge_blocks", ("MODEL",), False)
register_node("ModelMergeMochiPreview", "ModelMergeMochiPreview", "model_merge_blocks", ("MODEL",), False)
register_node("ModelMergeLTXV", "ModelMergeLTXV", "model_merge_blocks", ("MODEL",), False)
register_node("ModelMergeCosmos7B", "ModelMergeCosmos7B", "model_merge_blocks", ("MODEL",), False)
register_node("ModelMergeCosmos14B", "ModelMergeCosmos14B", "model_merge_blocks", ("MODEL",), False)
register_node("ModelMergeCosmosPredict2_2B", "ModelMergeCosmosPredict2_2B", "model_merge_blocks", ("MODEL",), False)
register_node("ModelMergeCosmosPredict2_14B", "ModelMergeCosmosPredict2_14B", "model_merge_blocks", ("MODEL",), False)
register_node("ModelMergeAuraflow", "ModelMergeAuraflow", "model_merge_blocks", ("MODEL",), False)
register_node("ModelMergeQwenImage", "ModelMergeQwenImage", "model_merge_blocks", ("MODEL",), False)
register_node("ModelMergeKrea2", "ModelMergeKrea2", "model_merge_blocks", ("MODEL",), False)
register_node("ModelMergeWAN2_1", "ModelMergeWAN2_1", "model_merge_blocks", ("MODEL",), False)


def svd_img2vid_conditioning(inputs):
    return (dict_get(inputs, "positive"), dict_get(inputs, "negative"),
            LatentImage(0, 0, 1, "", ""))


register_node("SVD_img2vid_Conditioning", "SVD_img2vid_Conditioning",
              "svd_img2vid_conditioning", ("CONDITIONING", "CONDITIONING", "LATENT"), False)


def model_patch_loader(inputs):
    n = dict_get(inputs, "model_name")
    if n is None:
        n = get_str(inputs, "model_name", "")
    return (n,)


register_node("ModelPatchLoader", "Load Model Patch",
              "model_patch_loader", ("MODEL_PATCH",), False)
register_node("DiffusersLoader", "Load Diffusers",
              "checkpoint_loader_simple", ("MODEL", "CLIP", "VAE"), False)
register_node("unCLIPCheckpointLoader", "Load unCLIP Checkpoint",
              "checkpoint_loader_simple", ("MODEL", "CLIP", "VAE", "CLIP_VISION"), False)
register_node("ImageOnlyCheckpointLoader", "Load Image-Only Checkpoint",
              "checkpoint_loader_simple", ("MODEL", "CLIP_VISION", "VAE"), False)
register_node("ConditioningSetAreaPercentageVideo", "ConditioningSetAreaPercentageVideo",
              "conditioning_passthrough", ("CONDITIONING",), False)
register_node("AnimaLLLiteApply", "AnimaLLLiteApply",
              "conditioning_passthrough", ("CONDITIONING",), False)
register_node("QwenImageDiffsynthControlnet", "QwenImageDiffsynthControlnet",
              "conditioning_passthrough", ("CONDITIONING",), False)
register_node("ZImageFunControlnet", "ZImageFunControlnet",
              "conditioning_passthrough", ("CONDITIONING",), False)
register_node("WanUni3CControlnetApply", "WanUni3CControlnetApply",
              "conditioning_passthrough", ("CONDITIONING",), False)
register_node("SUPIRApply", "SUPIRApply",
              "conditioning_passthrough", ("CONDITIONING",), False)
register_node("USOStyleReference", "USOStyleReference",
              "conditioning_passthrough", ("CONDITIONING",), False)
def video_linear_cfg_guidance(inputs):
    m: SDPipelineHandle = dict_get(inputs, "model")
    if m is None:
        return (None,)
    min_cfg = get_float(inputs, "min_cfg", 1.0)
    sd_set_video_cfg(m.pipeline, 0, min_cfg)
    return (m,)


def video_triangle_cfg_guidance(inputs):
    m: SDPipelineHandle = dict_get(inputs, "model")
    if m is None:
        return (None,)
    min_cfg = get_float(inputs, "min_cfg", 1.0)
    sd_set_video_cfg(m.pipeline, 1, min_cfg)
    return (m,)


register_node("VideoLinearCFGGuidance", "VideoLinearCFGGuidance",
              "video_linear_cfg_guidance", ("MODEL",), False)
register_node("VideoTriangleCFGGuidance", "VideoTriangleCFGGuidance",
              "video_triangle_cfg_guidance", ("MODEL",), False)


def print_node_list():
    keys = dict_keys(NODE_CLASS_MAPPINGS)
    i = 0
    n = len(keys)
    while i < n:
        print(keys[i])
        i = i + 1


def call_node(class_type: str, inputs):
    if class_type == "CheckpointLoaderSimple":
        return checkpoint_loader_simple(inputs)
    elif class_type == "DualCLIPLoader":
        return dual_clip_loader(inputs)
    elif class_type == "CLIPTextEncode":
        return clip_text_encode(inputs)
    elif class_type == "CLIPSetLastLayer":
        return clip_set_last_layer(inputs)
    elif class_type == "ConditioningCombine":
        return conditioning_combine(inputs)
    elif class_type == "ConditioningConcat":
        return conditioning_concat(inputs)
    elif class_type == "ConditioningAverage":
        return conditioning_average(inputs)
    elif class_type == "EmptyLatentImage":
        return empty_latent_image(inputs)
    elif class_type == "LatentUpscale":
        return latent_upscale(inputs)
    elif class_type == "LatentCrop":
        return latent_crop(inputs)
    elif class_type == "KSampler":
        return ksampler(inputs)
    elif class_type == "KSamplerAdvanced":
        return ksampler_advanced(inputs)
    elif class_type == "LORALoader":
        return lora_loader(inputs)
    elif class_type == "DiffusionModelLoader":
        return diffusion_model_loader(inputs)
    elif class_type == "HiResFix":
        return hires_fix(inputs)
    elif class_type == "ADetailer":
        return adetailer(inputs)
    elif class_type == "IPAdapterApply":
        return ipadapter_apply(inputs)
    elif class_type == "CLIPVisionLoader":
        return clip_vision_loader(inputs)
    elif class_type == "IPAdapterModelLoader":
        return ipadapter_model_loader(inputs)
    elif class_type == "VAEDecode":
        return vae_decode(inputs)
    elif class_type == "VAEEncode" or class_type == "VAEEncodeTiled":
        return vae_encode(inputs)
    elif class_type == "LoadImageMask":
        return load_image_mask(inputs)
    elif class_type == "VAEEncodeForInpaint":
        return vae_encode_for_inpaint(inputs)
    elif class_type == "LoadImage":
        return load_image(inputs)
    elif class_type == "ImageScale":
        return image_scale(inputs)
    elif class_type == "ImageScaleBy":
        return image_scale_by(inputs)
    elif class_type == "ImageInvert":
        return image_invert(inputs)
    elif class_type == "EmptyImage":
        return empty_image(inputs)
    elif class_type == "ImagePadForOutpaint":
        return image_pad_for_outpaint(inputs)
    elif class_type == "ImageBlur":
        return image_blur(inputs)
    elif class_type == "ImageBatch":
        return image_batch(inputs)
    elif class_type == "ImageCompositeMasked":
        return image_composite_masked(inputs)
    elif class_type == "ImageCrop":
        return image_crop(inputs)
    elif class_type == "ImageToMask":
        return image_to_mask(inputs)
    elif class_type == "MaskToImage":
        return mask_to_image(inputs)
    elif class_type == "CLIPVisionEncode":
        return clip_vision_encode(inputs)
    elif class_type == "LoadImageOutput":
        return load_image(inputs)
    elif class_type == "StyleModelLoader":
        return style_model_loader(inputs)
    elif class_type == "StyleModelApply":
        return style_model_apply(inputs)
    elif class_type == "unCLIPConditioning":
        return unclip_conditioning(inputs)
    elif class_type == "GLIGENLoader":
        return gligen_loader(inputs)
    elif class_type == "GLIGENTextBoxApply":
        return gligen_textbox_apply(inputs)
    elif class_type == "PreviewImage":
        return preview_image(inputs)
    elif class_type == "Reroute":
        return reroute(inputs)
    elif class_type == "SaveImage":
        return save_image(inputs)
    elif class_type == "CheckpointLoader":
        return checkpoint_loader_simple(inputs)
    elif class_type == "UNETLoader":
        return unet_loader(inputs)
    elif class_type == "VAELoader":
        return vae_loader(inputs)
    elif class_type == "CLIPLoader":
        return clip_loader(inputs)
    elif class_type == "LoraLoader" or class_type == "LoraLoaderModelOnly" or class_type == "LoraLoaderBypass" or class_type == "LoraLoaderBypassModelOnly":
        return lora_loader(inputs)
    elif class_type == "CLIPMergeSimple":
        return clip_merge_simple(inputs)
    elif class_type == "CLIPMergeAdd":
        return clip_merge_add(inputs)
    elif class_type == "CLIPMergeSubtract":
        return clip_merge_subtract(inputs)
    elif class_type == "ModelSamplingFlux" or class_type == "ModelSamplingSD3" or class_type == "ModelSamplingAuraFlow":
        return model_sampling_flow(inputs)
    elif class_type == "ModelComputeDtype":
        return model_compute_dtype(inputs)
    elif class_type == "ModelAttentionBackend":
        return model_attention_backend(inputs)
    elif class_type == "RescaleCFG":
        return rescale_cfg(inputs)
    elif class_type == "ModelSamplingContinuousEDM" or class_type == "ModelSamplingContinuousV":
        return model_sampling_sigma_range(inputs)
    elif class_type == "ModelSamplingDiscrete" or class_type == "ModelSamplingStableCascade" or class_type == "ModelNoiseScale":
        return model_passthrough(inputs)
    elif class_type == "SaveLatent":
        return save_latent(inputs)
    elif class_type == "LoadLatent":
        return load_latent(inputs)
    elif class_type == "CheckpointSave":
        return checkpoint_save(inputs)
    elif class_type == "VAESave":
        return vae_save(inputs)
    elif class_type == "CLIPSave":
        return clip_save(inputs)
    elif class_type == "ModelSave":
        return model_save(inputs)
    elif class_type == "ModelMergeSimple":
        return model_merge_simple(inputs)
    elif class_type == "ModelMergeAdd":
        return model_merge_add(inputs)
    elif class_type == "ModelMergeSubtract":
        return model_merge_subtract(inputs)
    elif str_starts_with(class_type, "ModelMerge"):
        return model_merge_blocks(inputs)
    elif class_type == "DiffusersLoader" or class_type == "unCLIPCheckpointLoader" or class_type == "ImageOnlyCheckpointLoader":
        return checkpoint_loader_simple(inputs)
    elif class_type == "ImageOnlyCheckpointSave":
        return checkpoint_save(inputs)
    elif class_type == "WebcamCapture":
        return webcam_capture(inputs)
    elif class_type == "ModelPatchLoader":
        return model_patch_loader(inputs)
    elif class_type == "SVD_img2vid_Conditioning":
        return svd_img2vid_conditioning(inputs)
    elif class_type == "ConditioningSetAreaPercentageVideo" or class_type == "AnimaLLLiteApply" or class_type == "QwenImageDiffsynthControlnet" or class_type == "ZImageFunControlnet" or class_type == "WanUni3CControlnetApply" or class_type == "SUPIRApply" or class_type == "USOStyleReference":
        return conditioning_passthrough(inputs)
    elif class_type == "VideoLinearCFGGuidance":
        return video_linear_cfg_guidance(inputs)
    elif class_type == "VideoTriangleCFGGuidance":
        return video_triangle_cfg_guidance(inputs)
    elif class_type == "VAEDecodeTiled":
        return vae_decode(inputs)
    elif class_type == "ConditioningZeroOut":
        return conditioning_zero_out(inputs)
    elif class_type == "ControlNetLoader" or class_type == "DiffControlNetLoader":
        return controlnet_loader(inputs)
    elif class_type == "ControlNetApply":
        return controlnet_apply(inputs)
    elif class_type == "ControlNetApplyAdvanced":
        return controlnet_apply_advanced(inputs)
    elif class_type == "InpaintModelConditioning":
        return inpaint_model_conditioning(inputs)
    elif class_type == "LatentUpscaleBy":
        return latent_upscale_by(inputs)
    elif class_type == "PreviewAny":
        return preview_any(inputs)
    elif class_type == "ConditioningSetArea" or class_type == "ConditioningSetAreaPercentage" or class_type == "ConditioningSetAreaStrength" or class_type == "ConditioningSetMask" or class_type == "ConditioningMultiply" or class_type == "ConditioningSetTimestepRange":
        return conditioning_passthrough(inputs)
    elif class_type == "LatentRotate":
        return latent_rotate(inputs)
    elif class_type == "LatentFlip":
        return latent_flip(inputs)
    elif class_type == "LatentComposite":
        return latent_composite(inputs)
    elif class_type == "LatentBlend":
        return latent_blend(inputs)
    elif class_type == "RepeatLatentBatch":
        return repeat_latent_batch(inputs)
    elif class_type == "LatentFromBatch":
        return latent_from_batch(inputs)
    elif class_type == "SetLatentNoiseMask":
        return set_latent_noise_mask(inputs)
    else:
        return (None,)
# === execution.static.py ===

def build_deps(prompt):
    node_ids = dict_keys(prompt)
    deps = make_dict()
    inputs_cache = make_dict()
    n = len(node_ids)
    i = 0
    while i < n:
        nid = node_ids[i]
        node = dict_get(prompt, nid)
        raw_inputs = dict_get(node, "inputs")
        resolved = make_dict()
        dep_list = py_list()
        input_keys = dict_keys(raw_inputs)
        k = 0
        m = len(input_keys)
        while k < m:
            key = input_keys[k]
            val = dict_get(raw_inputs, key)
            if is_link(val):
                src_id = val[0]
                src_idx = val[1]
                dict_set(resolved, key, val)  # keep original [src_id, src_idx] array
                dep_list = py_list_append(dep_list, src_id)
            else:
                dict_set(resolved, key, val)
            k = k + 1
        dict_set(deps, nid, dep_list)
        dict_set(inputs_cache, nid, resolved)
        i = i + 1
    return deps, inputs_cache


def resolve_all(inputs, node_outputs, default_output_dir: str):
    resolved = make_dict()
    keys = dict_keys(inputs)
    k = 0
    n = len(keys)
    while k < n:
        key = keys[k]
        val = dict_get(inputs, key)
        if is_link(val):
            src_id = val[0]
            src_idx = val[1]
            src_outputs = dict_get(node_outputs, src_id)
            resolved_val = src_outputs[src_idx]
        else:
            resolved_val = val
        dict_set(resolved, key, resolved_val)
        k = k + 1
    # 节点未显式指定 output_dir 时，用 CLI --output-dir 作为默认
    if dict_get(resolved, "output_dir") is None:
        dict_set(resolved, "output_dir", default_output_dir)
    return resolved


def validate_prompt(prompt) -> int:
    # 对照 ComfyUI validate_inputs：节点类型存在 + 链接指向合法节点/输出
    node_ids = dict_keys(prompt)
    i = 0
    ni = len(node_ids)
    while i < ni:
        nid = node_ids[i]
        node = dict_get(prompt, nid)
        class_type = dict_get(node, "class_type")
        if not node_exists(class_type):
            print("validate: unknown node type '" + class_type + "' at node " + nid)
            return 1
        raw_inputs = dict_get(node, "inputs")
        input_keys = dict_keys(raw_inputs)
        k = 0
        nk = len(input_keys)
        while k < nk:
            key = input_keys[k]
            val = dict_get(raw_inputs, key)
            if is_link(val):
                src_id = val[0]
                src_idx = val[1]
                src_node = dict_get(prompt, src_id)
                if src_node is None:
                    print("validate: node " + nid + " input '" + key + "' links to missing node " + src_id)
                    return 2
                src_class = dict_get(src_node, "class_type")
                rc = node_return_count(src_class)
                if src_idx < 0 or src_idx >= rc:
                    print("validate: node " + nid + " input '" + key + "' links to invalid output " + string_of_int(src_idx) + " of " + src_class)
                    return 3
            k = k + 1
        i = i + 1
    return 0


def input_signature(class_type: str, raw) -> str:
    # 简化版输入签名：class_type + 原始输入（链接以 [src_id, idx] 参与）。
    # 静态 DAG 下，签名相同的节点输出相同，可复用缓存。
    return class_type + "|" + json_dumps(raw)


def execute_prompt(prompt_json: str, output_dir: str):
    prompt = parse_json(prompt_json)
    vrc = validate_prompt(prompt)
    if vrc != 0:
        print("Prompt validation failed, rc=" + string_of_int(vrc))
        return make_dict()
    node_ids = dict_keys(prompt)
    deps, inputs_cache = build_deps(prompt)
    node_outputs = make_dict()
    executed = make_dict()
    cache = make_dict()
    n = len(node_ids)
    remaining = n
    while remaining > 0:
        progress = 0
        i = 0
        while i < n:
            nid = node_ids[i]
            if dict_get(executed, nid) is None:
                ready = 1
                dep_list = dict_get(deps, nid)
                m = len(dep_list)
                j = 0
                while j < m:
                    dep_id = dep_list[j]
                    if dict_get(executed, dep_id) is None:
                        ready = 0
                    j = j + 1
                if ready == 1:
                    node = dict_get(prompt, nid)
                    class_type = dict_get(node, "class_type")
                    inputs = dict_get(inputs_cache, nid)
                    sig = input_signature(class_type, inputs)
                    cached = dict_get(cache, sig)
                    if cached is not None:
                        outputs = cached
                    else:
                        resolved = resolve_all(inputs, node_outputs, output_dir)
                        outputs = call_node(class_type, resolved)
                        dict_set(cache, sig, outputs)
                    dict_set(node_outputs, nid, outputs)
                    dict_set(executed, nid, 1)
                    remaining = remaining - 1
                    progress = 1
            i = i + 1
        if progress == 0:
            break
    if remaining > 0:
        print("Prompt has a cycle or missing dependency: " + string_of_int(remaining) + " node(s) not executed")
    return node_outputs
# === main.static.py ===


def guard_main():
    main()


def make_workflow_node(class_type: str, inputs) -> dict:
    node = make_dict()
    dict_set(node, "class_type", class_type)
    dict_set(node, "inputs", inputs)
    return node


def build_prompt_workflow(checkpoint: str, prompt: str, output_path: str, output_dir: str,
                          width: int, height: int, steps: int, cfg: float,
                          seed: int, sampler: str, scheduler: str) -> str:
    # Determine output directory and filename prefix.
    if output_path is not None and str_length(output_path) > 0:
        out_dir = path_dirname(output_path)
        if str_length(out_dir) == 0:
            out_dir = "."
        parts = path_split(output_path)
        filename = parts[1]
        # Strip .png extension if present.
        if str_ends_with(filename, ".png"):
            filename = str_slice(filename, 0, str_length(filename) - 4)
        filename_prefix = filename
    else:
        out_dir = output_dir
        filename_prefix = "comfy_cli"

    # Default external CLIP encoders for SDXL.
    clip_l = "clip_l.safetensors"
    clip_g = "clip_g.safetensors"

    workflow = make_dict()

    ckpt_inputs = make_dict()
    dict_set(ckpt_inputs, "ckpt_name", checkpoint)
    dict_set(ckpt_inputs, "clip_l_name", clip_l)
    dict_set(ckpt_inputs, "clip_g_name", clip_g)
    dict_set(workflow, "1", make_workflow_node("CheckpointLoaderSimple", ckpt_inputs))

    sampler_inputs = make_dict()
    dict_set(sampler_inputs, "model", py_list("1", 0))
    dict_set(sampler_inputs, "prompt", prompt)
    dict_set(sampler_inputs, "negative_prompt", "")
    dict_set(sampler_inputs, "width", width)
    dict_set(sampler_inputs, "height", height)
    dict_set(sampler_inputs, "steps", steps)
    dict_set(sampler_inputs, "cfg", cfg)
    dict_set(sampler_inputs, "sampler_name", sampler)
    dict_set(sampler_inputs, "scheduler", scheduler)
    dict_set(sampler_inputs, "seed", seed)
    dict_set(sampler_inputs, "output_dir", out_dir)
    dict_set(sampler_inputs, "filename_prefix", filename_prefix)
    dict_set(workflow, "2", make_workflow_node("KSampler", sampler_inputs))

    save_inputs = make_dict()
    dict_set(save_inputs, "images", py_list("2", 0))
    dict_set(workflow, "3", make_workflow_node("SaveImage", save_inputs))

    return json_dumps(workflow)


def main():
    args = parse_cli_args()
    show_help: bool = dict_get(args, "show_help")
    if show_help:
        print_help()
        exit_program(0)
    list_nodes: bool = dict_get(args, "list_nodes")
    if list_nodes:
        print_node_list()
        exit_program(0)
    output_dir = dict_get(args, "output_dir")
    if output_dir is None:
        output_dir = "./output"
    workflow_path = dict_get(args, "workflow")
    if workflow_path is not None and str_length(workflow_path) > 0:
        fp = file_open(workflow_path, "r")
        content = file_read_all(fp)
        file_close(fp)
        result = execute_prompt(content, output_dir)
    else:
        prompt = dict_get(args, "prompt")
        checkpoint = dict_get(args, "checkpoint")
        output_path = dict_get(args, "output")
        width = get_int(args, "width", 1024)
        height = get_int(args, "height", 1024)
        steps = get_int(args, "steps", 20)
        cfg = get_float(args, "cfg", 7.0)
        seed = get_int(args, "seed", 42)
        sampler = get_str(args, "sampler", "euler_a")
        scheduler = get_str(args, "scheduler", "discrete")
        if checkpoint is not None and prompt is not None:
            content = build_prompt_workflow(checkpoint, prompt, output_path, output_dir,
                                              width, height, steps, cfg, seed,
                                              sampler, scheduler)
            result = execute_prompt(content, output_dir)
        else:
            print("Usage: comfycli-bin workflow.json --output-dir ./output")
            print("   or: comfycli-bin --checkpoint model.safetensors --prompt 'cat' --output ./out.png")
            exit_program(1)

