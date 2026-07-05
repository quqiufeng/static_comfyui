from sd import load_checkpoint, LoadResult, CLIP, VAE, ModelPatcher
from clip_model import encode_sdxl, clip_tokenizer_create, clip_tokenizer_free


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


def checkpoint_loader_simple(inputs):
    ckpt_name = dict_get(inputs, "ckpt_name")
    if str_starts_with(ckpt_name, "/"):
        ckpt_path = ckpt_name
    else:
        ckpt_path = "/data/models/image/" + ckpt_name
    result = load_checkpoint(ckpt_path)
    return (result.model, result.clip, result.vae)


register_node("CheckpointLoaderSimple", "Load Checkpoint",
              "checkpoint_loader_simple", ("MODEL", "CLIP", "VAE"), False)


def dual_clip_loader(inputs):
    clip_name1 = dict_get(inputs, "clip_name1")
    clip_name2 = dict_get(inputs, "clip_name2")
    base_path = "/data/models/image/"
    clip_g_sd = torch.safetensors_load(base_path + clip_name1)
    clip_l_sd = torch.safetensors_load(base_path + clip_name2)
    result = make_dict()
    dict_set(result, "clip_g", clip_g_sd)
    dict_set(result, "clip_l", clip_l_sd)
    return (result,)
    return (result,)


register_node("DualCLIPLoader", "Dual CLIP Loader",
              "dual_clip_loader", ("CLIP",), False)


def clip_text_encode(inputs):
    text = dict_get(inputs, "text")
    clip_obj = dict_get(inputs, "clip")
    clip_l = dict_get(clip_obj, "clip_l")
    clip_g = dict_get(clip_obj, "clip_g")
    tokenizer_l = clip_tokenizer_create("/data/models/image/clip_l_vocab.json",
                                         "/data/models/image/clip_l_merges.txt")
    tokenizer_g = clip_tokenizer_create("/data/models/image/clip_g_vocab.json",
                                         "/data/models/image/clip_g_merges.txt")
    text_emb, pooled = encode_sdxl(clip_l, clip_g, tokenizer_l, tokenizer_g, text)
    clip_tokenizer_free(tokenizer_l)
    clip_tokenizer_free(tokenizer_g)
    cond = make_dict()
    dict_set(cond, "crossattn", text_emb)
    dict_set(cond, "pooled_output", pooled)
    return (cond,)


register_node("CLIPTextEncode", "CLIP Text Encode (Prompt)",
              "clip_text_encode", ("CONDITIONING",), False)


def empty_latent_image(inputs):
    width = dict_get(inputs, "width")
    height = dict_get(inputs, "height")
    batch_size = 1
    latent = torch.zeros([batch_size, 4, height // 8, width // 8])
    result = make_dict()
    dict_set(result, "samples", latent)
    return (result,)


register_node("EmptyLatentImage", "Empty Latent Image",
              "empty_latent_image", ("LATENT",), False)


def vae_decode(inputs):
    vae = dict_get(inputs, "vae")
    samples = dict_get(inputs, "samples")
    tile_size = 512
    overlap = 64
    latent_tensor = dict_get(samples, "samples")
    vae_ptr = vae.vae_ptr
    image = torch.vae_decode_tiled(vae_ptr, latent_tensor, tile_size, overlap)
    return (image,)


register_node("VAEDecode", "VAE Decode",
              "vae_decode", ("IMAGE",), False)


def vae_encode(inputs):
    vae = dict_get(inputs, "vae")
    pixels = dict_get(inputs, "pixels")
    tile_size = 512
    overlap = 64
    vae_ptr = vae.vae_ptr
    latent = torch.vae_encode_tiled(vae_ptr, pixels, tile_size, overlap)
    result = make_dict()
    dict_set(result, "samples", latent)
    return (result,)


register_node("VAEEncode", "VAE Encode",
              "vae_encode", ("LATENT",), False)


def k_sampler_inner(inputs):
    model: ModelPatcher = dict_get(inputs, "model")
    seed = dict_get(inputs, "seed")
    steps = dict_get(inputs, "steps")
    cfg = dict_get(inputs, "cfg")
    sampler_name = dict_get(inputs, "sampler_name")
    scheduler = dict_get(inputs, "scheduler")
    positive = dict_get(inputs, "positive")
    negative = dict_get(inputs, "negative")
    latent_image = dict_get(inputs, "latent_image")
    denoise = dict_get(inputs, "denoise")
    cond = dict_get(positive, "crossattn")
    uncond = dict_get(negative, "crossattn")
    pooled_pos = dict_get(positive, "pooled_output")
    pooled_neg = dict_get(negative, "pooled_output")
    latent_tensor = dict_get(latent_image, "samples")
    h = tensor_shape_dim(latent_tensor, 2)
    w = tensor_shape_dim(latent_tensor, 3)
    torch.manual_seed(seed)
    noise = torch.randn([1, 4, h, w])
    x = noise
    sigma_min = 0.029
    sigma_max = 14.615
    sigmas = torch.sampler_sigmas(steps, sigma_min, sigma_max, scheduler)
    sd_handle = model.sd_handle
    n = 0
    while n < steps:
        sigma_t = torch.narrow(sigmas, 0, n, 1)
        sigma_prev = torch.narrow(sigmas, 0, n + 1, 1)
        s_in = sigma_t
        cond_out = model_fn(sd_handle, x, s_in, cond, pooled_pos)
        uncond_out = model_fn(sd_handle, x, s_in, uncond, pooled_neg)
        # CFG: denoised = uncond + cfg * (cond - uncond)
        # Use scale tensor for cfg value
        scale = torch.ones([1])
        cfg_tensor = torch.mul(scale, cfg)
        diff = torch.sub(cond_out, uncond_out)
        scaled = torch.mul(diff, cfg_tensor)
        denoised = torch.add(uncond_out, scaled)
        x = torch.sample_euler(denoised, x, sigma_t, sigma_prev)
        n = n + 1
    result = make_dict()
    dict_set(result, "samples", x)
    return (result,)


def model_fn(sd_handle, x, sigma, text_emb, pooled_emb):
    os_h = 1024.0
    os_w = 1024.0
    crop_t = 0.0
    crop_l = 0.0
    ts_h = 1024.0
    ts_w = 1024.0
    return torch.sdxl_unet_forward(sd_handle, x, sigma, text_emb, pooled_emb,
                                   os_h, os_w, crop_t, crop_l, ts_h, ts_w)


register_node("KSampler", "KSampler",
              "k_sampler_inner", ("LATENT",), False)


def save_image(inputs):
    images = dict_get(inputs, "images")
    filename_prefix = dict_get(inputs, "filename_prefix")
    output_dir = dict_get(inputs, "output_dir")
    if output_dir is None:
        output_dir = "/tmp/comfy_output"
    torch.save_image(images, output_dir + "/" + filename_prefix + ".png", 0)
    return (images,)


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
    elif class_type == "VAEDecode":
        return vae_decode(inputs)
    elif class_type == "VAEEncode":
        return vae_encode(inputs)
    elif class_type == "KSampler":
        return k_sampler_inner(inputs)
    elif class_type == "SaveImage":
        return save_image(inputs)
    else:
        return (None,)


def main():
    pass
