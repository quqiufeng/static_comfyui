from supported_models_base import *

# SDXL
SDXL = make_bac(
    make_dict_from("model_channels", 320, "use_linear_in_transformer", True, "transformer_depth", py_list(0, 0, 2, 2, 10, 10), "context_dim", 2048, "adm_in_channels", 2816, "use_temporal_attention", False),
    py_list(),
    "SDXL",
    1
)

# SDXL Refiner
SDXL_REFINER = make_bac(
    make_dict_from("model_channels", 384, "use_linear_in_transformer", True, "transformer_depth", py_list(0, 0, 4, 4), "context_dim", 2560, "adm_in_channels", 2560, "use_temporal_attention", False),
    py_list(),
    "SDXL",
    1
)

# SD1.5
SD15 = make_bac(
    make_dict_from("context_dim", 768, "model_channels", 320, "use_linear_in_transformer", False, "adm_in_channels", 0, "use_temporal_attention", False),
    py_list(),
    "SD15",
    1
)

# SD2.0
SD20 = make_bac(
    make_dict_from("context_dim", 1024, "model_channels", 320, "use_linear_in_transformer", False, "adm_in_channels", 0, "use_temporal_attention", False),
    py_list(),
    "SD15",
    1
)

# SD3
SD3 = make_bac(
    make_dict_from("in_channels", 16, "pos_embed_scaling_factor", 0),
    py_list(),
    "SD3",
    6
)

# Flux
FLUX = make_bac(
    make_dict_from("image_model", "flux", "guidance_embed", True),
    py_list(),
    "Flux",
    8
)

# Flux Schnell (no guidance)
FLUX_SCHNELL = make_bac(
    make_dict_from("image_model", "flux", "guidance_embed", False),
    py_list(),
    "Flux",
    8
)

# All registered models
ALL_MODELS: list = py_list(SDXL, SDXL_REFINER, SD15, SD20, SD3, FLUX, FLUX_SCHNELL)


def find_model(unet_config: dict, state_dict: dict) -> list:
    i: int = 0
    while i < py_list_length(ALL_MODELS):
        m: list = py_list_ref(ALL_MODELS, i)
        if bac_matches(m, unet_config, state_dict):
            return m
        i = i + 1
    return py_list()


def main():
    pass
