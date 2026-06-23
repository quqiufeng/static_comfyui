# sd_runtime/sd_lora.static.py — LoRA 加载 + 匹配 (Phase 10)
#
# Flat functions pattern. 状态以全局变量存储。

from ops import *


# LoRA 状态
_lora_dict: ptr
_n_weights: int
_max_lora: int
_lora_indices: ptr
_lora_A: ptr
_lora_B: ptr
_lora_count: int


def lora_init(n_unet_weights: int, max_lora: int) -> void:
    """Initialize LoRA state."""
    global _lora_indices, _lora_A, _lora_B
    global _n_weights, _max_lora, _lora_count
    _n_weights = n_unet_weights
    _max_lora = max_lora
    _lora_indices = make_int_array(max_lora)
    _lora_A = make_ptr_array(max_lora)
    _lora_B = make_ptr_array(max_lora)
    _lora_count = 0


def lora_load(lora_path: str) -> void:
    """Load LoRA safetensors file."""
    global _lora_dict
    _lora_dict = torch_std_safetensors_load(lora_path)


def lora_load_from_dict(lora_dict: ptr) -> void:
    """Load from existing dict pointer."""
    global _lora_dict
    _lora_dict = lora_dict


def lora_match() -> int:
    """Match LoRA keys to UNet weight indices.
    
    Returns: number of matched pairs.
    """
    global _lora_dict, _n_weights, _max_lora
    global _lora_indices, _lora_A, _lora_B, _lora_count
    _lora_count = torch_std_lora_match_to_unet(
        _lora_dict, _n_weights,
        _lora_indices, _lora_A, _lora_B, _max_lora)
    return _lora_count


def lora_get_indices() -> ptr:
    return _lora_indices

def lora_get_A() -> ptr:
    return _lora_A

def lora_get_B() -> ptr:
    return _lora_B

def lora_get_count() -> int:
    return _lora_count


def lora_free() -> void:
    """Free LoRA resources."""
    global _lora_dict
    torch_std_safetensors_free(_lora_dict)
