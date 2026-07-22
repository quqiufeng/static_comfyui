from sd_backend import sd_create, sd_free, sd_load, sd_generate_with_options, SD_WTYPE_AUTO


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


def checkpoint_loader_simple(inputs):
    print("DEBUG inputs dict keys:")
    keys = dict_keys(inputs)
    i = 0
    while i < len(keys):
        print("  key=" + keys[i])
        i = i + 1

    ckpt_name = dict_get(inputs, "ckpt_name")
    clip_l_name = dict_get(inputs, "clip_l_name")
    clip_g_name = dict_get(inputs, "clip_g_name")

    print("DEBUG ckpt_name=[" + ckpt_name + "]")
    print("DEBUG clip_l_name=[" + clip_l_name + "]")
    print("DEBUG clip_g_name=[" + clip_g_name + "]")

    base_path = "/data/models/image/"
    if str_starts_with(ckpt_name, "/"):
        ckpt_path = ckpt_name
    else:
        ckpt_path = base_path + ckpt_name

    if clip_l_name is None:
        clip_l_path = ""
    elif str_starts_with(clip_l_name, "/"):
        clip_l_path = clip_l_name
    else:
        clip_l_path = base_path + clip_l_name

    if clip_g_name is None:
        clip_g_path = ""
    elif str_starts_with(clip_g_name, "/"):
        clip_g_path = clip_g_name
    else:
        clip_g_path = base_path + clip_g_name

    pipeline = sd_create()
    rc = sd_load(pipeline, ckpt_path, clip_l_path, clip_g_path, "", SD_WTYPE_AUTO, 8, 0)
    if rc != 0:
        print("SD checkpoint load failed, rc=" + string_of_int(rc))
        return (None,)

    handle = make_sd_pipeline_handle(pipeline)
    return (handle,)


register_node("CheckpointLoaderSimple", "Load Checkpoint",
              "checkpoint_loader_simple", ("MODEL",), False)


def sd_txt2img(inputs):
    model: SDPipelineHandle = dict_get(inputs, "model")
    pipeline = model.pipeline

    prompt = dict_get(inputs, "prompt")
    negative_prompt = dict_get(inputs, "negative_prompt")
    if negative_prompt is None:
        negative_prompt = ""

    width = dict_get(inputs, "width")
    if width is None:
        width = 1024
    height = dict_get(inputs, "height")
    if height is None:
        height = 1024
    steps = dict_get(inputs, "steps")
    if steps is None:
        steps = 20
    cfg = dict_get(inputs, "cfg")
    if cfg is None:
        cfg = 7.0
    sample_method = dict_get(inputs, "sample_method")
    if sample_method is None:
        sample_method = "euler_a"
    scheduler = dict_get(inputs, "scheduler")
    if scheduler is None:
        scheduler = "discrete"
    seed = dict_get(inputs, "seed")
    if seed is None:
        seed = 42

    output_dir = dict_get(inputs, "output_dir")
    if output_dir is None:
        output_dir = "/tmp/comfy_output"
    filename_prefix = dict_get(inputs, "filename_prefix")
    if filename_prefix is None:
        filename_prefix = "comfy"
    output_path = output_dir + "/" + filename_prefix + ".png"

    rc = sd_generate_with_options(pipeline, prompt, negative_prompt,
                                  width, height, steps, cfg,
                                  sample_method, scheduler, seed,
                                  0, 0, 0.0,
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
