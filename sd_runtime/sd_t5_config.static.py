# sd_runtime/sd_t5_config.static.py — T5 XL / XXL 配置 (Phase 12 gap)
# 对位 comfyui_ref/comfy/text_encoders/t5_config_xxl.json
#
# T5 模型配置常量 + 创建辅助函数。

from ops import *


# ==============================================================================
# T5 Small / Base / Large / XL / XXL 配置
# ==============================================================================

T5_CONFIGS: ptr  # dict of name -> config dict


def t5_config_init() -> void:
    """Initialize T5 config registry."""
    global T5_CONFIGS
    
    d = make_dict()
    
    # T5 Small
    small = make_dict()
    dict_set(small, "d_model", 512)
    dict_set(small, "d_ff", 2048)
    dict_set(small, "num_layers", 6)
    dict_set(small, "num_heads", 8)
    dict_set(small, "vocab_size", 32128)
    dict_set(small, "max_len", 512)
    dict_set(d, "t5-small", small)
    
    # T5 Base
    base = make_dict()
    dict_set(base, "d_model", 768)
    dict_set(base, "d_ff", 3072)
    dict_set(base, "num_layers", 12)
    dict_set(base, "num_heads", 12)
    dict_set(base, "vocab_size", 32128)
    dict_set(base, "max_len", 512)
    dict_set(d, "t5-base", base)
    
    # T5 Large
    large = make_dict()
    dict_set(large, "d_model", 1024)
    dict_set(large, "d_ff", 4096)
    dict_set(large, "num_layers", 24)
    dict_set(large, "num_heads", 16)
    dict_set(large, "vocab_size", 32128)
    dict_set(large, "max_len", 512)
    dict_set(d, "t5-large", large)
    
    # T5 XL
    xl = make_dict()
    dict_set(xl, "d_model", 2048)
    dict_set(xl, "d_ff", 8192)
    dict_set(xl, "num_layers", 24)
    dict_set(xl, "num_heads", 32)
    dict_set(xl, "vocab_size", 32128)
    dict_set(xl, "max_len", 512)
    dict_set(d, "t5-xl", xl)
    
    # T5 XXL
    xxl = make_dict()
    dict_set(xxl, "d_model", 4096)
    dict_set(xxl, "d_ff", 16384)
    dict_set(xxl, "num_layers", 24)
    dict_set(xxl, "num_heads", 64)
    dict_set(xxl, "vocab_size", 32128)
    dict_set(xxl, "max_len", 512)
    dict_set(d, "t5-xxl", xxl)
    
    # FLUX T5 (custom)
    flux = make_dict()
    dict_set(flux, "d_model", 4096)
    dict_set(flux, "d_ff", 16384)
    dict_set(flux, "num_layers", 24)
    dict_set(flux, "num_heads", 64)
    dict_set(flux, "vocab_size", 32128)
    dict_set(flux, "max_len", 512)
    dict_set(d, "flux-t5", flux)
    
    T5_CONFIGS = d


def t5_config_get(name: str) -> ptr:
    """Get T5 config dict by name."""
    global T5_CONFIGS
    return dict_get(T5_CONFIGS, name)


def t5_config_get_int(config: ptr, key: str) -> int:
    """Get int value from config dict."""
    return int(dict_get(config, key))


def t5_config_get_max_len(name: str) -> int:
    """Get max sequence length for a T5 model."""
    cfg = t5_config_get(name)
    if cfg:
        return t5_config_get_int(cfg, "max_len")
    return 512
