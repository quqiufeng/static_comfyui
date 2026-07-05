# model_detection: state_dict → model architecture identification
# Uses key-PRESENCE patterns (no tensor shape inspection needed for basic detection)


def detect_unet_config(state_dict: dict, prefix: str) -> dict:
    result: dict = make_dict()

    has_double_blocks: bool = dict_contains(state_dict, prefix + "double_blocks.0.img_attn.norm.key_norm.weight")
    has_joint_blocks: bool = dict_contains(state_dict, prefix + "joint_blocks.0.context_block.attn.qkv.weight")
    has_input_blocks: bool = dict_contains(state_dict, prefix + "input_blocks.0.0.weight")

    if has_double_blocks:
        dict_set(result, "image_model", "flux")
        has_guidance: bool = dict_contains(state_dict, prefix + "guidance_in.lin.weight")
        dict_set(result, "guidance_embed", has_guidance)
    elif has_joint_blocks:
        has_pos_embed: bool = dict_contains(state_dict, prefix + "pos_embed")
        dict_set(result, "in_channels", 16)
        if has_pos_embed:
            dict_set(result, "pos_embed_scaling_factor", 0)
    elif has_input_blocks:
        has_label_emb: bool = dict_contains(state_dict, prefix + "label_emb.0.0.weight")
        if has_label_emb:
            dict_set(result, "model_channels", 320)
            dict_set(result, "use_linear_in_transformer", True)
            dict_set(result, "context_dim", 2048)
            dict_set(result, "adm_in_channels", 2816)
            dict_set(result, "use_temporal_attention", False)
        else:
            has_context_768: bool = dict_contains(state_dict, prefix + "input_blocks.0.1.transformer_blocks.0.attn2.to_k.weight")
            if has_context_768:
                dict_set(result, "context_dim", 768)
            else:
                dict_set(result, "context_dim", 1024)
            dict_set(result, "model_channels", 320)
            dict_set(result, "use_linear_in_transformer", False)
            dict_set(result, "adm_in_channels", 0)
            dict_set(result, "use_temporal_attention", False)

    return result


def model_config_from_unet(state_dict: dict, unet_config: dict, models_list: list) -> list:
    i: int = 0
    while i < py_list_length(models_list):
        m: list = py_list_ref(models_list, i)
        if bac_matches(m, unet_config, state_dict):
            return m
        i = i + 1
    return py_list()


def main():
    pass
