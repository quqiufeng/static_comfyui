# sd_runtime/sd_utils.static.py — 工具函数 (Phase 14)
#
# Flat functions: 图像 I/O, 设备管理, 模型检测, 张量工具。

from ops import *


# ==============================================================================
# 模型类型常量
# ==============================================================================

MODEL_TYPE_UNKNOWN: int = 0
MODEL_TYPE_SD15: int = 1
MODEL_TYPE_SDXL: int = 2
MODEL_TYPE_SD3: int = 3
MODEL_TYPE_FLUX: int = 4


# ==============================================================================
# 图像 I/O
# ==============================================================================

def load_image(path: str) -> ptr:
    return torch_std_load_image(path)


def save_image(tensor: ptr, path: str) -> void:
    torch_std_save_image(tensor, path, 0)


def load_image_png(path: str) -> ptr:
    return torch_std_load_image_png(path)


def save_image_png(tensor: ptr, path: str) -> void:
    torch_std_save_image_png(tensor, path)


# ==============================================================================
# 设备管理
# ==============================================================================

def is_cuda_available() -> int:
    return torch_std_cuda_is_available()


def to_cuda(t: ptr) -> ptr:
    return torch_std_to_cuda(t)


def to_cpu(t: ptr) -> ptr:
    return torch_std_to_cpu(t)


def is_cuda(t: ptr) -> int:
    return torch_std_is_cuda(t)


# ==============================================================================
# 浮点格式
# ==============================================================================

def to_float32(t: ptr) -> ptr:
    return torch_std_to_dtype(t, 0)


def to_float64(t: ptr) -> ptr:
    return torch_std_to_dtype(t, 1)


# ==============================================================================
# 模型类型检测
# ==============================================================================

def detect_model_type(sd_dict: ptr) -> int:
    """Detect SD model type from safetensors key names."""
    n = torch_std_safetensors_count(sd_dict)
    has_sd3 = 0
    has_flux = 0
    has_xl = 0
    has_double = 0
    has_single = 0
    
    for i in range(n):
        name = torch_std_safetensors_name(sd_dict, i)
        if name == null:
            continue
        if str_contains(name, "double_blocks"):
            has_double = 1
        if str_contains(name, "single_blocks"):
            has_single = 1
        if str_contains(name, "flux") or str_contains(name, "txt_in"):
            has_flux = 1
        if str_contains(name, "label_emb"):
            has_xl = 1
        if str_contains(name, "x_embedder"):
            has_sd3 = 1
    
    if has_flux or (has_double and has_single):
        return MODEL_TYPE_FLUX
    if has_sd3 or has_double:
        return MODEL_TYPE_SD3
    if has_xl:
        return MODEL_TYPE_SDXL
    return MODEL_TYPE_SD15


# ==============================================================================
# 随机种子
# ==============================================================================

def manual_seed(seed: int) -> void:
    torch_std_manual_seed(seed)


# ==============================================================================
# 张量工具
# ==============================================================================

def numel(t: ptr) -> int:
    return torch_std_numel(t)


def ndim(t: ptr) -> int:
    return torch_std_ndim(t)


def tensor_shape(t: ptr) -> ptr:
    """Return shape as int array."""
    n = ndim(t)
    shape = make_int_array(n)
    torch_std_shape(t, shape)
    return shape


def tensor_size(t: ptr, dim: int) -> int:
    return torch_std_size(t, dim)


def tensor_scalar(t: ptr) -> float:
    """Extract float value from 1-element tensor."""
    out = make_float_array(1)
    torch_std_to_float_array(t, out, 1)
    return float_array_ref(out, 0)


# ==============================================================================
# LoRA 合并工具
# ==============================================================================

def lora_merge_into_model(model_dict: ptr, lora_dict: ptr,
                           prefix: str, scale: float) -> int:
    """Merge LoRA into model dict. Returns merged count."""
    return torch_std_lora_merge_into(model_dict, lora_dict, prefix, scale)


def make_scalar_tensor(value: float) -> ptr:
    """Create a 1-element float32 tensor."""
    shape = make_int_array(1)
    int_array_set(shape, 0, 1)
    return torch_std_full(shape, 1, value, 0)


# ==============================================================================
# 采样器工具
# ==============================================================================

def to_alpha_bar(sigma: float) -> float:
    """Convert sigma to alpha_cumprod for DDIM."""
    return 1.0 / (1.0 + sigma * sigma)
