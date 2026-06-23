# sd_runtime/sd_core.static.py — Model management + device mgmt (Phase 13)
# Flat functions. 单例全局状态管理。

from ops import *


# ==============================================================================
# 全局状态
# ==============================================================================

_device_cuda: int     # 0=cpu, 1=cuda
_current_device: int  # GPU index


# ==============================================================================
# GPU 设备管理
# ==============================================================================

def core_init() -> void:
    """Initialize device management. Detects CUDA."""
    global _device_cuda, _current_device
    _device_cuda = torch_std_cuda_is_available()
    _current_device = 0


def is_cuda_available() -> int:
    return torch_std_cuda_is_available()


def set_device(cuda: int) -> void:
    """Set default device: 0=cpu, 1=cuda."""
    global _device_cuda
    _device_cuda = cuda


def get_device() -> int:
    return _device_cuda


def to_device(t: ptr) -> ptr:
    """Move tensor to current device."""
    global _device_cuda
    if _device_cuda:
        return torch_std_to_cuda(t)
    return t


def to_cpu(t: ptr) -> ptr:
    return torch_std_to_cpu(t)


def to_cuda(t: ptr) -> ptr:
    return torch_std_to_cuda(t)


# ==============================================================================
# 模型加载
# ==============================================================================

# Model type constants
MODEL_UNKNOWN: int = 0
MODEL_SD15: int = 1
MODEL_SDXL: int = 2
MODEL_FLUX: int = 3
MODEL_T5: int = 4
MODEL_CLIP: int = 5
MODEL_VAE: int = 6
MODEL_CONTROLNET: int = 7
MODEL_LORA: int = 8

# Loaded model registry (simple ptr arrays for handles)
_loaded_models: ptr    # ptr array of model handles
_loaded_types: ptr     # int array of model types
_n_loaded: int
_MAX_LOADED: int = 32


def model_registry_init() -> void:
    """Initialize model registry."""
    global _loaded_models, _loaded_types, _n_loaded
    _loaded_models = make_ptr_array(_MAX_LOADED)
    _loaded_types = make_int_array(_MAX_LOADED)
    _n_loaded = 0


def model_load_jit(path: str, model_type: int) -> int:
    """Load TorchScript model. Returns handle index."""
    global _loaded_models, _loaded_types, _n_loaded
    handle = torch_std_jit_load(path)
    idx = _n_loaded
    ptr_array_set(_loaded_models, idx, handle)
    int_array_set(_loaded_types, idx, model_type)
    _n_loaded = _n_loaded + 1
    return idx


def model_load_safetensors(path: str, model_type: int) -> int:
    """Load safetensors model. Returns handle index."""
    global _loaded_models, _loaded_types, _n_loaded
    handle = torch_std_safetensors_load(path)
    idx = _n_loaded
    ptr_array_set(_loaded_models, idx, handle)
    int_array_set(_loaded_types, idx, model_type)
    _n_loaded = _n_loaded + 1
    return idx


def model_get(idx: int) -> ptr:
    """Get model handle by index."""
    return ptr_array_ref(_loaded_models, idx)


def model_get_type(idx: int) -> int:
    return int_array_ref(_loaded_types, idx)


def model_unload(idx: int) -> void:
    """Unload model (free resources)."""
    global _loaded_models, _loaded_types, _n_loaded
    t = int_array_ref(_loaded_types, idx)
    h = ptr_array_ref(_loaded_models, idx)
    if t == MODEL_SD15 or t == MODEL_SDXL or t == MODEL_FLUX:
        torch_std_delete_tensor(h)
    elif t == MODEL_CLIP:
        torch_std_jit_module_delete(h)
    elif t == MODEL_VAE:
        torch_std_jit_module_delete(h)
    elif t == MODEL_LORA:
        torch_std_safetensors_free(h)
    ptr_array_set(_loaded_models, idx, null)
    int_array_set(_loaded_types, idx, 0)


def model_unload_all() -> void:
    """Unload all models."""
    global _n_loaded
    for i in range(_n_loaded):
        model_unload(i)
    _n_loaded = 0


# ==============================================================================
# 显存管理 (simplified)
# ==============================================================================

def memory_empty_cache() -> void:
    """Clear PyTorch CUDA cache (no-op if not available)."""
    pass


def memory_get_free() -> int:
    """Get free CUDA memory in bytes. Returns 0 if unavailable."""
    return 0


# ==============================================================================
# 模型 Patcher (simplified — for LoRA merging)
# ==============================================================================

_patch_targets: ptr   # ptr array of weight ptrs
_patch_scales: ptr    # float array of scales
_n_patches: int
_MAX_PATCHES: int = 64


def patcher_init() -> void:
    """Initialize model patcher state."""
    global _patch_targets, _patch_scales, _n_patches
    _patch_targets = make_ptr_array(_MAX_PATCHES)
    _patch_scales = make_float_array(_MAX_PATCHES)
    _n_patches = 0


def patcher_add(target: ptr, scale: float) -> void:
    """Register a weight patch."""
    global _patch_targets, _patch_scales, _n_patches
    ptr_array_set(_patch_targets, _n_patches, target)
    float_array_set(_patch_scales, _n_patches, scale)
    _n_patches = _n_patches + 1


def patcher_apply() -> void:
    """Apply all patches (simple scaling of target weights)."""
    global _patch_targets, _patch_scales, _n_patches
    for i in range(_n_patches):
        t = ptr_array_ref(_patch_targets, i)
        s = float_array_ref(_patch_scales, i)
        torch_std_mul_scalar(t, s)
    _n_patches = 0


def patcher_clear() -> void:
    """Clear all patches without applying."""
    global _n_patches
    _n_patches = 0


# ==============================================================================
# Context Windows (simplified — placeholder for future)
# ==============================================================================

_ctx_window_size: int
_ctx_stride: int


def context_set(window_size: int, stride: int) -> void:
    """Set context window parameters for tiled processing."""
    global _ctx_window_size, _ctx_stride
    _ctx_window_size = window_size
    _ctx_stride = stride


def context_get_window() -> int:
    return _ctx_window_size


def context_get_stride() -> int:
    return _ctx_stride
