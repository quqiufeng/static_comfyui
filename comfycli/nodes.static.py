from sd_backend import sd_create, sd_free, sd_load, sd_load_ex, sd_load_lora, sd_generate_full, sd_ensure_directory, sd_set_ipadapter, sd_set_ipadapter_enabled, sd_set_init_image, sd_load_control_net, sd_set_control_image, SD_WTYPE_AUTO


NODE_CLASS_MAPPINGS: dict = make_dict()
NODE_DISPLAY_NAMES: dict = make_dict()


@dataclass
class SDPipelineHandle:
    pipeline: ptr


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
    handle = make_sd_pipeline_handle(pipeline)
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
    return (LatentImage(width, height, batch_size, ""),)


register_node("EmptyLatentImage", "Empty Latent Image",
              "empty_latent_image", ("LATENT",), False)


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

    denoise = get_float(inputs, "denoise", 1.0)
    if latent is not None and latent.image_path != "":
        sd_set_init_image(model.pipeline, latent.image_path, denoise)

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
    return (LatentImage(0, 0, 1, image_path),)


register_node("VAEEncode", "VAE Encode",
              "vae_encode", ("LATENT",), False)


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
    latent = dict_get(inputs, "samples")
    if latent is None:
        print("LatentUpscale: no samples received")
        return (None,)
    # In this simplified backend, width/height directly replace latent dimensions.
    width = get_int(inputs, "width", latent.width)
    height = get_int(inputs, "height", latent.height)
    batch_size = latent.batch_size
    return (LatentImage(width, height, batch_size, latent.image_path),)


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
    return (LatentImage(width, height, batch_size, latent.image_path),)


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
register_node("VAEDecodeTiled", "VAE Decode (Tiled)",
              "vae_decode", ("IMAGE",), False)


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
    elif class_type == "VAEEncode":
        return vae_encode(inputs)
    elif class_type == "LoadImage":
        return load_image(inputs)
    elif class_type == "ImageScale":
        return image_scale(inputs)
    elif class_type == "ImageScaleBy":
        return image_scale_by(inputs)
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
    elif class_type == "LoraLoader" or class_type == "LoraLoaderModelOnly":
        return lora_loader(inputs)
    elif class_type == "VAEDecodeTiled":
        return vae_decode(inputs)
    elif class_type == "ConditioningZeroOut":
        return conditioning_zero_out(inputs)
    elif class_type == "ControlNetLoader":
        return controlnet_loader(inputs)
    elif class_type == "ControlNetApply":
        return controlnet_apply(inputs)
    else:
        return (None,)


def main():
    pass
