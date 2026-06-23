# sd_runtime/sd_clip_vision.static.py — CLIP Vision encoder (Phase 3 gap)
# 对位 comfyui_ref/comfy/clip_vision.py (163 行)
#
# C++: JIT model forward (复用 torch_std_jit_*)
# StaticPy: 图像预处理 → ViT 编码 → pooled 输出

from ops import *


# CLIP Vision 常量
VISION_INPUT_SIZE: int = 224
VISION_MEAN: float = 0.48145466
VISION_STD: float = 0.26862954


# 全局状态
_cv_jit_module: ptr   # JIT TorchScript module


def clip_vision_init(jit_model_path: str) -> void:
    """Load CLIP Vision TorchScript JIT model."""
    global _cv_jit_module
    _cv_jit_module = torch_std_jit_load(jit_model_path)


def clip_vision_preprocess(image: ptr) -> ptr:
    """Preprocess image for CLIP Vision.
    
    image: (1, 3, H, W) float32 [0,1]
    Returns: (1, 3, 224, 224) normalized float32
    """
    # Resize to 224x224
    resized = torch_std_image_resize(image, VISION_INPUT_SIZE, VISION_INPUT_SIZE, "bilinear")
    
    # Normalize: (x - mean) / std
    mean_shape = make_int_array(4)
    int_array_set(mean_shape, 0, 1)
    int_array_set(mean_shape, 1, 3)
    int_array_set(mean_shape, 2, 1)
    int_array_set(mean_shape, 3, 1)
    
    mean_t = torch_std_full(mean_shape, 4, VISION_MEAN, 0)
    std_t = torch_std_full(mean_shape, 4, VISION_STD, 0)
    
    normalized = torch_std_div(torch_std_sub(resized, mean_t), std_t)
    return normalized


def clip_vision_encode(image: ptr) -> ptr:
    """Encode image through CLIP Vision.
    
    image: (1, 3, H, W) float32 [0,1]
    Returns: ptr array of [pooled_emb, seq_emb]
      pooled_emb: (1, D)  — CLS token embedding
      seq_emb: (1, 257, D) — all patch tokens + CLS
    """
    global _cv_jit_module
    processed = clip_vision_preprocess(image)
    output = torch_std_jit_forward(_cv_jit_module, processed)
    # JIT module returns [seq_emb, pooled_emb] or just seq_emb
    # Assume seq_emb (1, 257, D) — first token is CLS
    result = make_ptr_array(2)
    
    # Try getting shape to determine what JIT returned
    nd = torch_std_ndim(output)
    if nd == 3:
        # (1, L, D) — sequence output, extract CLS
        pooled = torch_std_narrow(output, 1, 0, 1)  # CLS token
        ptr_array_set(result, 0, pooled)
        ptr_array_set(result, 1, output)
    else:
        # Unknown format, return as-is
        ptr_array_set(result, 0, output)
        ptr_array_set(result, 1, null)
    
    return result


def clip_vision_free() -> void:
    """Free CLIP Vision resources."""
    global _cv_jit_module
    torch_std_jit_module_delete(_cv_jit_module)
