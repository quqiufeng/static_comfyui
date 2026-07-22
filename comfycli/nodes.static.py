from sd_backend import sd_create, sd_free, sd_load, sd_load_ex, sd_load_lora, sd_generate_with_options, sd_generate_hires, sd_generate_adetailer, sd_ensure_directory, sd_set_ipadapter, sd_set_ipadapter_enabled, SD_WTYPE_AUTO


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


@dataclass
class CLIPVisionModel:
    name: str


@dataclass
class IPAdapterModel:
    name: str


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
    smart_sharpen = get_float(inputs, "smart_sharpen", 0.5)
    smart_sharpen_radius = get_int(inputs, "smart_sharpen_radius", 2)
    edge_sharpen = get_float(inputs, "edge_sharpen", 1.5)
    edge_sharpen_radius = get_int(inputs, "edge_sharpen_radius", 2)
    edge_sharpen_threshold = get_float(inputs, "edge_sharpen_threshold", 0.3)

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
                           smart_sharpen, smart_sharpen_radius,
                           edge_sharpen, edge_sharpen_radius,
                           edge_sharpen_threshold,
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


def adetailer(inputs):
    model: SDPipelineHandle = dict_get(inputs, "model")
    if model is None:
        print("ADetailer: model is missing")
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
    cfg = get_float(inputs, "cfg", 7.0)
    sampler_name = get_str(inputs, "sampler_name", "euler")
    scheduler = get_str(inputs, "scheduler", "normal")
    seed = get_int(inputs, "seed", 42)

    vae_tiling = get_int(inputs, "vae_tiling", 0)
    vae_tile_size = get_int(inputs, "vae_tile_size", 0)
    vae_tile_overlap = get_float(inputs, "vae_tile_overlap", 0.5)

    ad_model_path = get_str(inputs, "ad_model_path", "")
    ad_prompt = get_str(inputs, "ad_prompt", prompt)
    ad_negative_prompt = get_str(inputs, "ad_negative_prompt", negative_prompt)

    output_dir = get_str(inputs, "output_dir", "/tmp/comfy_output")
    filename_prefix = get_str(inputs, "filename_prefix", "comfy")
    output_path = output_dir + "/" + filename_prefix + ".png"

    rc = sd_ensure_directory(output_dir)
    if rc != 0:
        print("Failed to create output dir: " + output_dir)
        return (None,)

    rc = sd_generate_adetailer(pipeline, prompt, negative_prompt,
                                width, height, steps, cfg,
                                sampler_name, scheduler, seed,
                                vae_tiling, vae_tile_size, vae_tile_overlap,
                                0, 0, 0, 0, 0.0,
                                0, 0.0, 0.0,
                                0, 0.0,
                                ad_model_path, ad_prompt, ad_negative_prompt,
                                output_path)
    if rc != 0:
        print("ADetailer generate failed, rc=" + string_of_int(rc))
        return (None,)

    return (output_path,)


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


def preview_image(inputs):
    image_path = dict_get(inputs, "images")
    if image_path is None:
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
    if clip is None:
        return (None,)
    return (clip,)


register_node("CLIPSetLastLayer", "CLIP Set Last Layer",
              "clip_set_last_layer", ("CLIP",), False)


def conditioning_combine(inputs):
    c1: Conditioning = dict_get(inputs, "conditioning_1")
    c2: Conditioning = dict_get(inputs, "conditioning_2")
    t1 = ""
    t2 = ""
    if c1 is not None:
        t1 = c1.text
    if c2 is not None:
        t2 = c2.text
    if t1 == "" and t2 == "":
        return (Conditioning(""),)
    if t1 == "":
        return (Conditioning(t2),)
    if t2 == "":
        return (Conditioning(t1),)
    return (Conditioning(t1 + ", " + t2),)


register_node("ConditioningCombine", "Conditioning Combine",
              "conditioning_combine", ("CONDITIONING",), False)


def conditioning_concat(inputs):
    c_to: Conditioning = dict_get(inputs, "conditioning_to")
    c_from: Conditioning = dict_get(inputs, "conditioning_from")
    t1 = ""
    t2 = ""
    if c_to is not None:
        t1 = c_to.text
    if c_from is not None:
        t2 = c_from.text
    if t1 == "" and t2 == "":
        return (Conditioning(""),)
    if t1 == "":
        return (Conditioning(t2),)
    if t2 == "":
        return (Conditioning(t1),)
    return (Conditioning(t1 + ", " + t2),)


register_node("ConditioningConcat", "Conditioning Concat",
              "conditioning_concat", ("CONDITIONING",), False)


def conditioning_average(inputs):
    c_to: Conditioning = dict_get(inputs, "conditioning_to")
    c_from: Conditioning = dict_get(inputs, "conditioning_from")
    strength = get_float(inputs, "conditioning_to_strength", 0.5)
    t1 = ""
    t2 = ""
    if c_to is not None:
        t1 = c_to.text
    if c_from is not None:
        t2 = c_from.text
    if t1 == "" and t2 == "":
        return (Conditioning(""),)
    if t1 == "":
        return (Conditioning(t2),)
    if t2 == "":
        return (Conditioning(t1),)
    # Simple strength-aware combination: stronger text goes first.
    if strength >= 0.5:
        return (Conditioning(t1 + ", " + t2),)
    else:
        return (Conditioning(t2 + ", " + t1),)


register_node("ConditioningAverage", "Conditioning Average",
              "conditioning_average", ("CONDITIONING",), False)


def latent_upscale(inputs):
    latent = dict_get(inputs, "samples")
    if latent is None:
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
    if latent is None:
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
    if model is None:
        print("KSamplerAdvanced: model is missing")
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
    add_noise = get_str(inputs, "add_noise", "default")
    start_at_step = get_int(inputs, "start_at_step", 0)
    end_at_step = get_int(inputs, "end_at_step", steps)
    return_noise = get_str(inputs, "return_noise", "false")

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

    # For the simplified backend, start_at_step/end_at_step are passed as
    # additional parameters if the C API supports them; otherwise they are ignored.
    rc = sd_generate_with_options(pipeline, prompt, negative_prompt,
                                  width, height, steps, cfg,
                                  sampler_name, scheduler, seed,
                                  vae_tiling, vae_tile_size, vae_tile_overlap,
                                  0, 0, 0, 0, 0.0,
                                  0, 0.0, 0.0,
                                  0, 0.0,
                                  output_path)
    if rc != 0:
        print("KSamplerAdvanced generate failed, rc=" + string_of_int(rc))
        return (None,)

    return (output_path,)


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


def main():
    pass
