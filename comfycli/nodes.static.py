from sd_backend import sd_create, sd_free, sd_load, sd_generate_with_options, sd_ensure_directory, SD_WTYPE_AUTO


NODE_CLASS_MAPPINGS: dict = make_dict()
NODE_DISPLAY_NAMES: dict = make_dict()


def register_node(class_type: str, display: str, func_name: str, ret_types: list,
                  is_output: bool):
    meta = make_dict()
    dict_set(meta, "display", display)
    dict_set(meta, "function", func_name)
    dict_set(meta, "return_types", ret_types)
    dict_set(meta, "output_node", is_output)
    dict_set(NODE_CLASS_MAPPINGS, class_type, meta)
    dict_set(NODE_DISPLAY_NAMES, class_type, display)


@dataclass
class SDPipelineHandle:
    pipeline: ptr


def make_sd_pipeline_handle(pipeline: ptr) -> SDPipelineHandle:
    return SDPipelineHandle(pipeline)


def resolve_model_path(name: str) -> str:
    if str_starts_with(name, "/"):
        return name
    return "/data/models/image/" + name


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
        return (None,)

    handle = make_sd_pipeline_handle(pipeline)
    return (handle,)


register_node("CheckpointLoaderSimple", "Load Checkpoint",
              "checkpoint_loader_simple", ("MODEL",), False)


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


def sd_txt2img(inputs):
    model: SDPipelineHandle = dict_get(inputs, "model")
    pipeline = model.pipeline

    prompt = get_str(inputs, "prompt", "")
    negative_prompt = get_str(inputs, "negative_prompt", "")

    width = get_int(inputs, "width", 1024)
    height = get_int(inputs, "height", 1024)
    steps = get_int(inputs, "steps", 20)
    cfg = get_float(inputs, "cfg", 7.0)
    sample_method = get_str(inputs, "sample_method", "euler_a")
    scheduler = get_str(inputs, "scheduler", "discrete")
    seed = get_int(inputs, "seed", 42)

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
                                  sample_method, scheduler, seed,
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
              "sd_txt2img", ("IMAGE",), False)


def save_image(inputs):
    # Image is already saved by KSampler; this node just passes through.
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
    elif class_type == "KSampler":
        return sd_txt2img(inputs)
    elif class_type == "SaveImage":
        return save_image(inputs)
    else:
        return (None,)


def main():
    pass
