from sd_backend import sd_create, sd_free, sd_load, sd_load_ex, sd_load_lora, sd_generate_full, sd_ensure_directory, sd_set_ipadapter, sd_set_ipadapter_enabled, sd_set_init_image, sd_load_control_net, sd_set_control_image, SD_WTYPE_AUTO


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
class CondEntry:
    text: str
    area_x: int
    area_y: int
    area_w: int
    area_h: int
    strength: float
    area_percent: bool


@dataclass
class Conditioning:
    entries: list
    control_net_path: str
    control_image_path: str
    control_strength: float


def make_cond_entry(text: str) -> CondEntry:
    return CondEntry(text, 0, 0, 0, 0, 1.0, False)


def make_conditioning(text: str) -> Conditioning:
    return Conditioning([make_cond_entry(text)], "", "", 0.0)


def cond_text(c: Conditioning) -> str:
    if c is None:
        return ""
    result = ""
    i = 0
    n = list_length(c.entries)
    while i < n:
        e: CondEntry = c.entries[i]
        if i > 0:
            result = result + ", "
        result = result + e.text
        i = i + 1
    return result


def concat_entries(a, b) -> list:
    result = []
    i = 0
    while i < list_length(a):
        result = result + [a[i]]
        i = i + 1
    i = 0
    while i < list_length(b):
        result = result + [b[i]]
        i = i + 1
    return result


def cond_has_area(c: Conditioning) -> bool:
    if c is None:
        return False
    i = 0
    n = list_length(c.entries)
    while i < n:
        e: CondEntry = c.entries[i]
        if e.area_w > 0 and e.area_h > 0:
            return True
        i = i + 1
    return False


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
        return cond_text(c)
    return get_str(inputs, fallback_key, "")


def conditioning_text(c: Conditioning) -> str:
    return cond_text(c)


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
    return (make_conditioning(text),)


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


def torch_jit_path(name: str) -> str:
    d = os_getenv("COMFYCLI_TORCH_DIR")
    if str_length(d) == 0:
        d = model_root()
    return d + "/" + name


def ksampler_torch(model: SDPipelineHandle, pos_c: Conditioning, neg_c: Conditioning,
                   width: int, height: int, opts, output_dir: str, output_path: str) -> int:
    pos_text = ""
    area_prompts = ""
    area_rects = ""
    area_strengths = ""
    first_area = True
    i = 0
    while i < list_length(pos_c.entries):
        e: CondEntry = pos_c.entries[i]
        if e.area_w > 0 and e.area_h > 0:
            if not first_area:
                area_prompts = area_prompts + "\n"
                area_rects = area_rects + ","
                area_strengths = area_strengths + ","
            first_area = False
            area_prompts = area_prompts + e.text
            ax = e.area_x
            ay = e.area_y
            aw = e.area_w
            ah = e.area_h
            if e.area_percent:
                ax = e.area_x * (width // 8) // 1000
                ay = e.area_y * (height // 8) // 1000
                aw = e.area_w * (width // 8) // 1000
                ah = e.area_h * (height // 8) // 1000
            area_rects = area_rects + string_of_int(ax) + "," + string_of_int(ay) + "," + string_of_int(aw) + "," + string_of_int(ah)
            area_strengths = area_strengths + format_float(e.strength, 4)
        else:
            pos_text = merge_conditioning_text(pos_text, e.text)
        i = i + 1

    rc = sd_ensure_directory(output_dir)
    if rc != 0:
        print("torch: failed to create output dir: " + output_dir)
        return -1

    return torch_std_sdxl_generate_areas_paths(
        model.model_path,
        torch_jit_path("clip_l_jit.pt"), torch_jit_path("clip_g_jit.pt"), torch_jit_path("vae_jit.pt"),
        torch_jit_path("clip_l_vocab.json"), torch_jit_path("clip_l_merges.txt"),
        pos_text, cond_text(neg_c), width, height,
        dict_get(opts, "steps"), dict_get(opts, "cfg"), dict_get(opts, "sampler_name"),
        dict_get(opts, "seed"), area_prompts, area_rects, area_strengths, output_path)


def ksampler(inputs):
    model: SDPipelineHandle = dict_get(inputs, "model")
    if model is None:
        print("KSampler: model is missing")
        return (None,)

    pos_c: Conditioning = dict_get(inputs, "positive")
    neg_c: Conditioning = dict_get(inputs, "negative")

    latent: LatentImage = dict_get(inputs, "latent_image")
    if latent is None:
        width = get_int(inputs, "width", 1024)
        height = get_int(inputs, "height", 1024)
    else:
        width = latent.width
        height = latent.height

    opts = parse_sampler_opts(inputs)
    out = sampler_output(inputs)

    # 带区域条件的正向 → 走 torch 管线（sd.cpp C API 无区域条件能力）
    if cond_has_area(pos_c):
        rc = ksampler_torch(model, pos_c, neg_c, width, height, opts, out[0], out[1])
        if rc != 0:
            print("torch generate failed, rc=" + string_of_int(rc))
            return (None,)
        return (out[1],)

    prompt = cond_text(pos_c)
    negative_prompt = cond_text(neg_c)

    apply_latent_extras(model, inputs, latent)

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
    c1: Conditioning = dict_get(inputs, "conditioning_1")
    c2: Conditioning = dict_get(inputs, "conditioning_2")
    if c1 is None:
        return (c2,)
    if c2 is None:
        return (c1,)
    # 拼接条目，保留各自的 area/strength（torch 管线按区域合成）
    return (Conditioning(concat_entries(c1.entries, c2.entries), c1.control_net_path,
                         c1.control_image_path, c1.control_strength),)


register_node("ConditioningCombine", "Conditioning Combine",
              "conditioning_combine", ("CONDITIONING",), False)


def conditioning_concat(inputs):
    text = merge_conditioning_text(
        conditioning_text(dict_get(inputs, "conditioning_to")),
        conditioning_text(dict_get(inputs, "conditioning_from")))
    return (make_conditioning(text),)


register_node("ConditioningConcat", "Conditioning Concat",
              "conditioning_concat", ("CONDITIONING",), False)


def conditioning_average(inputs):
    c_to = conditioning_text(dict_get(inputs, "conditioning_to"))
    c_from = conditioning_text(dict_get(inputs, "conditioning_from"))
    strength = get_float(inputs, "conditioning_to_strength", 0.5)
    # Simple strength-aware combination: stronger text goes first.
    if strength >= 0.5:
        return (make_conditioning(merge_conditioning_text(c_to, c_from)),)
    return (make_conditioning(merge_conditioning_text(c_from, c_to)),)


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
    return (make_conditioning(""),)


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
    return (Conditioning(c.entries, cn_path, image_path, strength),)


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


def to_milli(f: float) -> int:
    return string_to_int(format_float(f * 1000.0, 0))


def cond_with_area(c: Conditioning, x: int, y: int, w: int, h: int, strength: float, percent: bool) -> Conditioning:
    new_entries = []
    i = 0
    n = list_length(c.entries)
    while i < n:
        e: CondEntry = c.entries[i]
        new_entries = new_entries + [CondEntry(e.text, x, y, w, h, strength, percent)]
        i = i + 1
    return Conditioning(new_entries, c.control_net_path, c.control_image_path, c.control_strength)


def conditioning_set_area(inputs):
    c: Conditioning = dict_get(inputs, "conditioning")
    if c is None:
        return (None,)
    width = get_int(inputs, "width", 0)
    height = get_int(inputs, "height", 0)
    x = get_int(inputs, "x", 0)
    y = get_int(inputs, "y", 0)
    strength = get_float(inputs, "strength", 1.0)
    return (cond_with_area(c, x // 8, y // 8, width // 8, height // 8, strength, False),)


def conditioning_set_area_percentage(inputs):
    c: Conditioning = dict_get(inputs, "conditioning")
    if c is None:
        return (None,)
    width = get_float(inputs, "width", 1.0)
    height = get_float(inputs, "height", 1.0)
    x = get_float(inputs, "x", 0.0)
    y = get_float(inputs, "y", 0.0)
    strength = get_float(inputs, "strength", 1.0)
    return (cond_with_area(c, to_milli(x), to_milli(y), to_milli(width), to_milli(height), strength, True),)


def cond_map_strength(c: Conditioning, mode: int, value: float) -> Conditioning:
    new_entries = []
    i = 0
    n = list_length(c.entries)
    while i < n:
        e: CondEntry = c.entries[i]
        s = value
        if mode == 1:
            s = e.strength * value
        new_entries = new_entries + [CondEntry(e.text, e.area_x, e.area_y, e.area_w, e.area_h, s, e.area_percent)]
        i = i + 1
    return Conditioning(new_entries, c.control_net_path, c.control_image_path, c.control_strength)


def conditioning_set_area_strength(inputs):
    c: Conditioning = dict_get(inputs, "conditioning")
    if c is None:
        return (None,)
    strength = get_float(inputs, "strength", 1.0)
    return (cond_map_strength(c, 0, strength),)


def conditioning_multiply(inputs):
    c: Conditioning = dict_get(inputs, "conditioning")
    if c is None:
        return (None,)
    multiplier = get_float(inputs, "multiplier", 1.0)
    return (cond_map_strength(c, 1, multiplier),)


def conditioning_set_timestep_range(inputs):
    # 时间步范围在 torch 管线中暂未分段处理，透传
    c: Conditioning = dict_get(inputs, "conditioning")
    if c is None:
        return (None,)
    return (c,)


def conditioning_set_mask(inputs):
    # 掩码条件暂按区域矩形处理（set_cond_area=mask bounds 未实现），透传
    c: Conditioning = dict_get(inputs, "conditioning")
    if c is None:
        return (None,)
    return (c,)


register_node("ConditioningSetArea", "ConditioningSetArea",
              "conditioning_set_area", ("CONDITIONING",), False)
register_node("ConditioningSetAreaPercentage", "ConditioningSetAreaPercentage",
              "conditioning_set_area_percentage", ("CONDITIONING",), False)
register_node("ConditioningSetAreaStrength", "ConditioningSetAreaStrength",
              "conditioning_set_area_strength", ("CONDITIONING",), False)
register_node("ConditioningSetMask", "ConditioningSetMask",
              "conditioning_set_mask", ("CONDITIONING",), False)
register_node("ConditioningMultiply", "ConditioningMultiply",
              "conditioning_multiply", ("CONDITIONING",), False)
register_node("ConditioningSetTimestepRange", "ConditioningSetTimestepRange",
              "conditioning_set_timestep_range", ("CONDITIONING",), False)


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
    new_pos = Conditioning(pos.entries, cn_path, image_path, strength)
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
    v = dict_get(inputs, "source")
    text = "None"
    if v is None:
        text = "None"
    else:
        text = json_dumps(v)
    print("PreviewAny: " + text)
    return (text,)


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
    elif class_type == "ConditioningSetArea":
        return conditioning_set_area(inputs)
    elif class_type == "ConditioningSetAreaPercentage":
        return conditioning_set_area_percentage(inputs)
    elif class_type == "ConditioningSetAreaStrength":
        return conditioning_set_area_strength(inputs)
    elif class_type == "ConditioningSetMask":
        return conditioning_set_mask(inputs)
    elif class_type == "ConditioningMultiply":
        return conditioning_multiply(inputs)
    elif class_type == "ConditioningSetTimestepRange":
        return conditioning_set_timestep_range(inputs)
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


def main():
    pass
