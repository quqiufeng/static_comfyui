@dataclass
class ModelPatcher:
    sd_handle: ptr
    model_type: int
    load_device: int
    offload_device: int


def make_model_patcher(sd_handle: ptr, model_type: int, load_device: int, offload_device: int) -> ModelPatcher:
    return ModelPatcher(sd_handle, model_type, load_device, offload_device)


@dataclass
class CLIP:
    clip_ptr: ptr
    tokenizer_ptr: ptr


def make_clip(clip_ptr: ptr, tokenizer_ptr: ptr) -> CLIP:
    return CLIP(clip_ptr, tokenizer_ptr)


@dataclass
class VAE:
    vae_ptr: ptr


def make_vae(vae_ptr: ptr) -> VAE:
    return VAE(vae_ptr)


@dataclass
class LoadResult:
    model: ModelPatcher
    clip: CLIP
    vae: VAE


def make_load_result(model: ModelPatcher, clip: CLIP, vae: VAE) -> LoadResult:
    return LoadResult(model, clip, vae)


def build_key_dict(sd_ptr: ptr) -> dict:
    n: int = torch.safetensors_count(sd_ptr)
    d: dict = make_dict()
    i: int = 0
    while i < n:
        name = torch.safetensors_name(sd_ptr, i)
        dict_set(d, name, 1)
        i = i + 1
    return d


def load_checkpoint(ckpt_path: str) -> LoadResult:
    sd_ptr: ptr = torch.safetensors_load(ckpt_path)
    state_dict: dict = build_key_dict(sd_ptr)

    unet_prefix: str = "model.diffusion_model."
    has_unet: bool = dict_contains(state_dict, unet_prefix + "input_blocks.0.0.weight")
    if not has_unet:
        has_unet = dict_contains(state_dict, unet_prefix + "double_blocks.0.img_attn.norm.key_norm.weight")
    if not has_unet:
        has_unet = dict_contains(state_dict, unet_prefix + "joint_blocks.0.context_block.attn.qkv.weight")

    has_clip_l: bool = dict_contains(state_dict, "text_model.encoder.layers.0.layer_norm.weight")
    has_clip_g: bool = dict_contains(state_dict, "text_model.encoder.layers.30.mlp.fc1.weight")
    if not has_clip_l:
        has_clip_l = dict_contains(state_dict, "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.weight")
    if not has_clip_g:
        has_clip_g = dict_contains(state_dict, "conditioner.embedders.1.transformer.text_model.encoder.layers.30.mlp.fc1.weight")
    has_vae: bool = dict_contains(state_dict, "first_stage_model.decoder.conv_in.weight")
    if not has_vae:
        has_vae = dict_contains(state_dict, "decoder.conv_in.weight")
    if not has_vae:
        has_vae = dict_contains(state_dict, "conditioner.embedders.3.decoder.conv_in.weight")

    model_type_val: int = 0
    if has_unet:
        model_type_val = 1

    model_mp: ModelPatcher = make_model_patcher(sd_ptr, model_type_val, 0, -1)
    clip_obj = None
    vae_obj = None

    if has_clip_l or has_clip_g:
        clip_obj = make_clip(sd_ptr, sd_ptr)
    if has_vae:
        vae_obj = make_vae(sd_ptr)

    return make_load_result(model_mp, clip_obj, vae_obj)


def main():
    pass
