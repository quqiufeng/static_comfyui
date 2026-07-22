from sd_backend import sd_create, sd_free, sd_load, sd_load_ex, sd_generate_with_options, sd_generate_hires, sd_ensure_directory, SD_WTYPE_AUTO


NODE_CLASS_MAPPINGS: dict = make_dict()
NODE_DISPLAY_NAMES: dict = make_dict()


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


def make_sd_pipeline_handle(pipeline: ptr) -> SDPipelineHandle:
    return SDPipelineHandle(pipeline)


def resolve_model_path(name: str) -> str:
    if str_starts_with(name, "/"):
        return name
    return "/data/models/image/" + name


def register_node(class_type: str, display: str, func_name: str, ret_types: list,
                  is_output: bool):
    meta = make_dict()
    dict_set(meta, "display", display)
    dict_set(meta, "function", func_name)
    dict_set(meta, "return_types", ret_types)
    dict_set(meta, "output_node", is_output)
    dict_set(NODE_CLASS_MAPPINGS, class_type, meta)
    dict_set(NODE_DISPLAY_NAMES, class_type, display)


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
    if model is None:
        print("KSampler: model is missing")
        return (None,)
    pipeline = model.pipeline

    positive: Conditioning = dict_get(inputs, "positive")
    if positive is None:
        prompt = get_str(inputs, "prompt", "")
    else:
        prompt = positive.text

    negative: Conditioning = dict_get(inputs, "negative")
    if negative is None:
        negative_prompt = get_str(inputs, "negative_prompt", "")
    else:
        negative_prompt = negative.text

    latent: LatentImage = dict_get(inputs, "latent_image")
    if latent is None:
        width = get_int(inputs, "width", 1024)
        height = get_int(inputs, "height", 1024)
    else:
        width = latent.width
        height = latent.height

    steps = get_int(inputs, "steps", 20)
    cfg = get_float(inputs, "cfg", 7.0)
    sampler_name = get_str(inputs, "sampler_name", "euler")
    scheduler = get_str(inputs, "scheduler", "normal")
    seed = get_int(inputs, "seed", 42)
    denoise = get_float(inputs, "denoise", 1.0)

    vae_tiling = get_int(inputs, "vae_tiling", 0)
    vae_tile_size = get_int(inputs, "vae_tile_size", 0)
    vae_tile_overlap = get_float(inputs, "vae_tile_overlap", 0.5)

    output_dir = get_str(inputs, "output_dir", "/tmp/comfy_output")
    filename_prefix = get_str(inputs, "filename_prefix", "comfy")
    output_path = output_dir + "/" + filename_prefix + ".png"

    rc = sd_ensure_directory(output_dir)
    if rc != 0:
        print("Failed to create output dir: " + output_dir)
        return (None,)

    rc = sd_generate_with_options(pipeline, prompt, negative_prompt,
                                  width, height, steps, cfg,
                                  sampler_name, scheduler, seed,
                                  vae_tiling, vae_tile_size, vae_tile_overlap,
                                  0, 0, 0, 0, 0.0,
                                  0, 0.0, 0.0,
                                  0, 0.0,
                                  output_path)
    if rc != 0:
        print("SD generate failed, rc=" + string_of_int(rc))
        return (None,)

    return (output_path,)


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

    handle = make_sd_pipeline_handle(pipeline)
    return (handle, handle, handle)


register_node("DiffusionModelLoader", "Load Diffusion Model (GGUF)",
              "diffusion_model_loader", ("MODEL", "CLIP", "VAE"), False)


def hires_fix(inputs):
    model: SDPipelineHandle = dict_get(inputs, "model")
    if model is None:
        print("HiResFix: model is missing")
        return (None,)
    pipeline = model.pipeline

    positive: Conditioning = dict_get(inputs, "positive")
    if positive is None:
        prompt = get_str(inputs, "prompt", "")
    else:
        prompt = positive.text

    negative: Conditioning = dict_get(inputs, "negative")
    if negative is None:
        negative_prompt = get_str(inputs, "negative_prompt", "")
    else:
        negative_prompt = negative.text

    width = get_int(inputs, "width", 1024)
    height = get_int(inputs, "height", 1024)

    steps = get_int(inputs, "steps", 20)
    cfg = get_float(inputs, "cfg", 2.5)
    sampler_name = get_str(inputs, "sampler_name", "euler")
    scheduler = get_str(inputs, "scheduler", "discrete")
    seed = get_int(inputs, "seed", 0)
    if seed == 0:
        seed = -1

    vae_tiling = get_int(inputs, "vae_tiling", 1)
    vae_tile_size = get_int(inputs, "vae_tile_size", 128)
    vae_tile_overlap = get_float(inputs, "vae_tile_overlap", 0.5)

    hires_steps = get_int(inputs, "hires_steps", 45)
    hires_strength = get_float(inputs, "hires_strength", 0.35)

    freeu = get_int(inputs, "freeu", 1)
    freeu_b1 = get_float(inputs, "freeu_b1", 1.3)
    freeu_b2 = get_float(inputs, "freeu_b2", 1.4)

    sag = get_int(inputs, "sag", 0)
    sag_scale = get_float(inputs, "sag_scale", 1.0)

    clarity = get_float(inputs, "clarity", 0.2)
    sharpen = get_float(inputs, "sharpen", 0.3)
    sharpen_radius = get_int(inputs, "sharpen_radius", 1)

    output_dir = get_str(inputs, "output_dir", "/tmp/comfy_output")
    filename_prefix = get_str(inputs, "filename_prefix", "comfy")
    output_path = output_dir + "/" + filename_prefix + ".png"

    rc = sd_ensure_directory(output_dir)
    if rc != 0:
        print("Failed to create output dir: " + output_dir)
        return (None,)

    rc = sd_generate_hires(pipeline, prompt, negative_prompt,
                           width, height, steps, cfg,
                           sampler_name, scheduler, seed,
                           vae_tiling, vae_tile_size, vae_tile_overlap,
                           hires_steps, hires_strength,
                           freeu, freeu_b1, freeu_b2,
                           sag, sag_scale,
                           clarity, sharpen, sharpen_radius,
                           output_path)
    if rc != 0:
        print("HiResFix generate failed, rc=" + string_of_int(rc))
        return (None,)

    print("HiResFix: saved " + output_path)
    return (output_path,)


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


def call_node(class_type: str, inputs):
    if class_type == "CheckpointLoaderSimple":
        return checkpoint_loader_simple(inputs)
    elif class_type == "DualCLIPLoader":
        return dual_clip_loader(inputs)
    elif class_type == "CLIPTextEncode":
        return clip_text_encode(inputs)
    elif class_type == "EmptyLatentImage":
        return empty_latent_image(inputs)
    elif class_type == "KSampler":
        return ksampler(inputs)
    elif class_type == "DiffusionModelLoader":
        return diffusion_model_loader(inputs)
    elif class_type == "HiResFix":
        return hires_fix(inputs)
    elif class_type == "VAEDecode":
        return vae_decode(inputs)
    elif class_type == "SaveImage":
        return save_image(inputs)
    else:
        return (None,)


def main():
    pass
