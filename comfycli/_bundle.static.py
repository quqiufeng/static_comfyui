from comfycli_builtins import dict_keys, is_none, is_some, is_link, path_dirname, path_split

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
            if arg[0] != "-":
                workflow = arg
        i = i + 1

    result = make_dict()
    dict_set(result, "checkpoint", checkpoint)
    dict_set(result, "prompt", prompt)
    dict_set(result, "output", output)
    dict_set(result, "output_dir", output_dir)
    dict_set(result, "workflow", workflow)
    dict_set(result, "show_help", show_help)
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
extern fn sd_pipeline_generate_adetailer(pipeline: ptr, prompt: str, negative_prompt: str, width: int, height: int, steps: int, cfg: float, sample_method: str, scheduler: str, seed: int, vae_tiling: int, vae_tile_size: int, vae_tile_overlap: float, hires: int, hires_width: int, hires_height: int, hires_steps: int, hires_strength: float, freeu: int, freeu_b1: float, freeu_b2: float, sag: int, sag_scale: float, ad_model_path: str, ad_prompt: str, ad_negative_prompt: str, output_path: str) -> int from "sdcpp_adapter"
extern fn sd_pipeline_generate_full(pipeline: ptr, prompt: str, negative_prompt: str, width: int, height: int, hires_width: int, hires_height: int, steps: int, cfg: float, sample_method: str, scheduler: str, seed: int, vae_tiling: int, vae_tile_size: int, vae_tile_overlap: float, hires_steps: int, hires_strength: float, freeu: int, freeu_b1: float, freeu_b2: float, sag: int, sag_scale: float, clarity: float, sharpen_amount: float, sharpen_radius: int, smart_sharpen_strength: float, smart_sharpen_radius: int, edge_sharpen_amount: float, edge_sharpen_radius: int, edge_sharpen_threshold: float, ad_model_path: str, ad_prompt: str, ad_negative_prompt: str, output_path: str) -> int from "sdcpp_adapter"

extern fn sd_ensure_dir(path: str) -> int from "sdcpp_adapter"

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
# === nodes.static.py ===

@dataclass
class SDPipelineHandle:
    pipeline: ptr


@dataclass
class Conditioning:
    text: str


@dataclass
class LatentImage:
    width: int
    height: int
    batch_size: int


@dataclass
class CLIPVisionModel:
    name: str


@dataclass
class IPAdapterModel:
    name: str


def make_sd_pipeline_handle(pipeline: ptr) -> SDPipelineHandle:
    return SDPipelineHandle(pipeline)


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
    if is_some(c):
        return c.text
    return get_str(inputs, fallback_key, "")


def conditioning_text(c: Conditioning) -> str:
    if is_none(c):
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
    dict_set(opts, "freeu", 0)
    dict_set(opts, "freeu_b1", 0.0)
    dict_set(opts, "freeu_b2", 0.0)
    dict_set(opts, "sag", 0)
    dict_set(opts, "sag_scale", 0.0)
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
    pass


def get_int(inputs, key: str, default: int) -> int:
    v = dict_get(inputs, key)
    if is_none(v):
        return default
    return v


def get_float(inputs, key: str, default: float) -> float:
    v = dict_get(inputs, key)
    if is_none(v):
        return default
    return v


def get_str(inputs, key: str, default: str) -> str:
    v = dict_get(inputs, key)
    if is_none(v):
        return default
    return v


def checkpoint_loader_simple(inputs):
    ckpt_name = dict_get(inputs, "ckpt_name")
    clip_l_name = dict_get(inputs, "clip_l_name")
    clip_g_name = dict_get(inputs, "clip_g_name")

    ckpt_path = resolve_model_path(ckpt_name)
    if is_none(clip_l_name):
        clip_l_path = ""
    else:
        clip_l_path = resolve_model_path(clip_l_name)
    if is_none(clip_g_name):
        clip_g_path = ""
    else:
        clip_g_path = resolve_model_path(clip_g_name)

    pipeline = sd_create()
    rc = sd_load(pipeline, ckpt_path, clip_l_path, clip_g_path, "", 42, 8, 0)
    if rc != 0:
        print("SD checkpoint load failed, rc=" + string_of_int(rc))
        return (None, None, None)

    handle = make_sd_pipeline_handle(pipeline)
    return (handle, handle, handle)


register_node("CheckpointLoaderSimple", "Load Checkpoint",
              "checkpoint_loader_simple", ("MODEL", "CLIP", "VAE"), False)


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
    return (Conditioning(text),)


register_node("CLIPTextEncode", "CLIP Text Encode",
              "clip_text_encode", ("CONDITIONING",), False)


def empty_latent_image(inputs):
    width = get_int(inputs, "width", 1024)
    height = get_int(inputs, "height", 1024)
    batch_size = get_int(inputs, "batch_size", 1)
    return (LatentImage(width, height, batch_size),)


register_node("EmptyLatentImage", "Empty Latent Image",
              "empty_latent_image", ("LATENT",), False)


def ksampler(inputs):
    model: SDPipelineHandle = dict_get(inputs, "model")
    if is_none(model):
        print("KSampler: model is missing")
        return (None,)

    prompt = resolve_prompt_text(inputs, "positive", "prompt")
    negative_prompt = resolve_prompt_text(inputs, "negative", "negative_prompt")

    latent: LatentImage = dict_get(inputs, "latent_image")
    if is_none(latent):
        width = get_int(inputs, "width", 1024)
        height = get_int(inputs, "height", 1024)
    else:
        width = latent.width
        height = latent.height

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
    if is_none(samples):
        print("VAEDecode: no samples received")
        return (None,)
    return (samples,)


register_node("VAEDecode", "VAE Decode",
              "vae_decode", ("IMAGE",), False)


def diffusion_model_loader(inputs):
    diffusion_model_name = dict_get(inputs, "diffusion_model_name")
    llm_name = dict_get(inputs, "llm_name")
    vae_name = dict_get(inputs, "vae_name")

    if is_none(diffusion_model_name):
        print("DiffusionModelLoader: diffusion_model_name is required")
        return (None,)
    if is_none(llm_name):
        print("DiffusionModelLoader: llm_name is required")
        return (None,)

    diffusion_model_path = resolve_model_path(diffusion_model_name)
    llm_path = resolve_model_path(llm_name)
    vae_path = ""
    if is_some(vae_name):
        vae_path = resolve_model_path(vae_name)

    pipeline = sd_create()
    rc = sd_load_ex(pipeline, "", "", "", vae_path, 42, 8, 1,
                    diffusion_model_path, llm_path)
    if rc != 0:
        print("DiffusionModelLoader: load failed, rc=" + string_of_int(rc))
        return (None,)

    handle = make_sd_pipeline_handle(pipeline)
    return (handle, handle, handle)


register_node("DiffusionModelLoader", "Load Diffusion Model (GGUF)",
              "diffusion_model_loader", ("MODEL", "CLIP", "VAE"), False)


def lora_loader(inputs):
    model: SDPipelineHandle = dict_get(inputs, "model")
    if is_none(model):
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
    if is_none(model):
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
    if is_none(image_path):
        print("SaveImage: no image path received")
        return (None,)
    print("Image saved to: " + image_path)
    return (image_path,)


register_node("SaveImage", "Save Image",
              "save_image", ("IMAGE",), True)


def adetailer(inputs):
    model: SDPipelineHandle = dict_get(inputs, "model")
    if is_none(model):
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
    if is_none(model):
        print("IPAdapterApply: model is missing")
        return (None,)
    pipeline = model.pipeline

    ipadapter_obj: IPAdapterModel = dict_get(inputs, "ipadapter")
    clip_vision_obj: CLIPVisionModel = dict_get(inputs, "clip_vision")
    if is_some(ipadapter_obj):
        ipadapter_model = ipadapter_obj.name
    else:
        ipadapter_model = get_str(inputs, "ipadapter_model", "")
    if is_some(clip_vision_obj):
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


def preview_image(inputs):
    image_path = dict_get(inputs, "images")
    if is_none(image_path):
        print("PreviewImage: no image received")
        return (None,)
    return (image_path,)


register_node("PreviewImage", "Preview Image",
              "preview_image", ("IMAGE",), True)


def clip_set_last_layer(inputs):
    clip = dict_get(inputs, "clip")
    layer = get_int(inputs, "stop_at_clip_layer", -1)
    # The backend currently always uses the default CLIP layer;
    # this node is provided for workflow compatibility.
    if is_none(clip):
        return (None,)
    return (clip,)


register_node("CLIPSetLastLayer", "CLIP Set Last Layer",
              "clip_set_last_layer", ("CLIP",), False)


def conditioning_combine(inputs):
    text = merge_conditioning_text(
        conditioning_text(dict_get(inputs, "conditioning_1")),
        conditioning_text(dict_get(inputs, "conditioning_2")))
    return (Conditioning(text),)


register_node("ConditioningCombine", "Conditioning Combine",
              "conditioning_combine", ("CONDITIONING",), False)


def conditioning_concat(inputs):
    text = merge_conditioning_text(
        conditioning_text(dict_get(inputs, "conditioning_to")),
        conditioning_text(dict_get(inputs, "conditioning_from")))
    return (Conditioning(text),)


register_node("ConditioningConcat", "Conditioning Concat",
              "conditioning_concat", ("CONDITIONING",), False)


def conditioning_average(inputs):
    c_to = conditioning_text(dict_get(inputs, "conditioning_to"))
    c_from = conditioning_text(dict_get(inputs, "conditioning_from"))
    strength = get_float(inputs, "conditioning_to_strength", 0.5)
    # Simple strength-aware combination: stronger text goes first.
    if strength >= 0.5:
        return (Conditioning(merge_conditioning_text(c_to, c_from)),)
    return (Conditioning(merge_conditioning_text(c_from, c_to)),)


register_node("ConditioningAverage", "Conditioning Average",
              "conditioning_average", ("CONDITIONING",), False)


def latent_upscale(inputs):
    latent = dict_get(inputs, "samples")
    if is_none(latent):
        print("LatentUpscale: no samples received")
        return (None,)
    # In this simplified backend, width/height directly replace latent dimensions.
    width = get_int(inputs, "width", latent.width)
    height = get_int(inputs, "height", latent.height)
    batch_size = latent.batch_size
    return (LatentImage(width, height, batch_size),)


register_node("LatentUpscale", "Latent Upscale",
              "latent_upscale", ("LATENT",), False)


def latent_crop(inputs):
    latent = dict_get(inputs, "samples")
    if is_none(latent):
        print("LatentCrop: no samples received")
        return (None,)
    width = get_int(inputs, "width", latent.width)
    height = get_int(inputs, "height", latent.height)
    batch_size = latent.batch_size
    return (LatentImage(width, height, batch_size),)


register_node("LatentCrop", "Latent Crop",
              "latent_crop", ("LATENT",), False)


def reroute(inputs):
    val = dict_get(inputs, "anything")
    return (val,)


register_node("Reroute", "Reroute",
              "reroute", ("*",), False)


def ksampler_advanced(inputs):
    model: SDPipelineHandle = dict_get(inputs, "model")
    if is_none(model):
        print("KSamplerAdvanced: model is missing")
        return (None,)

    prompt = resolve_prompt_text(inputs, "positive", "prompt")
    negative_prompt = resolve_prompt_text(inputs, "negative", "negative_prompt")

    latent: LatentImage = dict_get(inputs, "latent_image")
    if is_none(latent):
        width = get_int(inputs, "width", 1024)
        height = get_int(inputs, "height", 1024)
    else:
        width = latent.width
        height = latent.height

    opts = parse_sampler_opts(inputs)
    out = sampler_output(inputs)
    rc = run_sampler(model, prompt, negative_prompt, width, height, opts, out[0], out[1])
    if rc != 0:
        print("KSamplerAdvanced generate failed, rc=" + string_of_int(rc))
        return (None,)

    return (out[1],)


register_node("KSamplerAdvanced", "KSampler Advanced",
              "ksampler_advanced", ("LATENT", "IMAGE"), False)


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
    elif class_type == "LoadImage":
        return load_image(inputs)
    elif class_type == "PreviewImage":
        return preview_image(inputs)
    elif class_type == "Reroute":
        return reroute(inputs)
    elif class_type == "SaveImage":
        return save_image(inputs)
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


def resolve_all(inputs, node_outputs):
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
    return resolved


def execute_prompt(prompt_json: str, output_dir: str):
    prompt = parse_json(prompt_json)
    node_ids = dict_keys(prompt)
    deps, inputs_cache = build_deps(prompt)
    node_outputs = make_dict()
    executed = make_dict()
    n = len(node_ids)
    remaining = n
    while remaining > 0:
        progress = 0
        i = 0
        while i < n:
            nid = node_ids[i]
            if is_none(dict_get(executed, nid)):
                ready = 1
                dep_list = dict_get(deps, nid)
                m = len(dep_list)
                j = 0
                while j < m:
                    dep_id = dep_list[j]
                    if is_none(dict_get(executed, dep_id)):
                        ready = 0
                    j = j + 1
                if ready == 1:
                    node = dict_get(prompt, nid)
                    class_type = dict_get(node, "class_type")
                    inputs = dict_get(inputs_cache, nid)
                    resolved = resolve_all(inputs, node_outputs)
                    outputs = call_node(class_type, resolved)
                    dict_set(node_outputs, nid, outputs)
                    dict_set(executed, nid, 1)
                    remaining = remaining - 1
                    progress = 1
            i = i + 1
        if progress == 0:
            remaining = 0
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
    if is_some(output_path) and str_length(output_path) > 0:
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
    output_dir = dict_get(args, "output_dir")
    if is_none(output_dir):
        output_dir = "./output"
    workflow_path = dict_get(args, "workflow")
    if is_some(workflow_path) and str_length(workflow_path) > 0:
        content = file_read_all(workflow_path)
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
        if is_some(checkpoint) and is_some(prompt):
            content = build_prompt_workflow(checkpoint, prompt, output_path, output_dir,
                                              width, height, steps, cfg, seed,
                                              sampler, scheduler)
            result = execute_prompt(content, output_dir)
        else:
            print("Usage: comfycli-bin workflow.json --output-dir ./output")
            print("   or: comfycli-bin --checkpoint model.safetensors --prompt 'cat' --output ./out.png")
            exit_program(1)

