# BAC = model config base
# Stored as list: [unet_cfg_keys, unet_cfg_vals, required_keys, latent_name, model_type_val]

def make_bac(unet_cfg_keys: list[str], unet_cfg_vals: list, required_keys: list[str], latent_name: str, model_type_val: int) -> list:
    return py_list(unet_cfg_keys, unet_cfg_vals, required_keys, latent_name, model_type_val)

def bac_unet_cfg_keys(bac: list) -> list[str]:
    return py_list_ref(bac, 0)

def bac_unet_cfg_vals(bac: list) -> list:
    return py_list_ref(bac, 1)

def bac_required_keys(bac: list) -> list[str]:
    return py_list_ref(bac, 2)

def bac_latent_format_name(bac: list) -> str:
    return py_list_ref(bac, 3)

def bac_model_type_value(bac: list) -> int:
    return py_list_ref(bac, 4)


def bac_matches(bac: list, unet_config: dict, state_dict: dict) -> bool:
    cfg_keys: list[str] = bac_unet_cfg_keys(bac)
    cfg_vals: list = bac_unet_cfg_vals(bac)
    i: int = 0
    while i < py_list_length(cfg_keys):
        k: str = py_list_ref(cfg_keys, i)
        if not dict_contains(unet_config, k):
            return False
        v = py_list_ref(cfg_vals, i)
        if dict_get(unet_config, k) != v:
            return False
        i = i + 1
    req: list[str] = bac_required_keys(bac)
    j: int = 0
    while j < py_list_length(req):
        rk: str = py_list_ref(req, j)
        if not dict_contains(state_dict, rk):
            return False
        j = j + 1
    return True


def main():
    pass
