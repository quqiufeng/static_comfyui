# sd_runtime/clip_model.static.py — CLIP Text Encoder
# 1:1 对齐 comfyui_ref/comfy/clip_model.py + comfy/sd1_clip.py SDClipModel
#
# C++ 运行时提供 JIT 模块 forward：torch_std_clip_text_forward
# 本模块提供加载和调用的包装。
#
# CLIP 文本模型架构（以 SD1.5 ViT-L/14 为例）:
#   12 层 transformer, embed_dim=768, heads=12, intermediate_size=3072
#   position_embedding: learned (77, 768)
#   token_embedding: vocabulary (49408, 768)
#
# 用法:
#   clip = load_clip_model("/path/to/clip_jit.pt")
#   emb = clip_encode(clip, token_ids)  # returns (1, 77, 768)
#   free_model(clip)

from ops import *


def load_clip_model(jit_path: str) -> ptr:
    """Load CLIP text encoder JIT module from .pt file.
    
    Returns opaque pointer to torch::jit::Module.
    """
    return torch_std_jit_load(jit_path)


def clip_encode(clip_module: ptr, token_ids: ptr, use_f16: int) -> ptr:
    """Encode token IDs to text embeddings.
    
    token_ids: (1, 77) int64 tensor (from tokenizer)
    use_f16: 1 to cast output to float16, 0 for float32
    
    Returns: (1, 77, embed_dim) float tensor
    """
    return torch_std_clip_text_forward(clip_module, token_ids, use_f16)


def sdxl_dual_clip(clip1: ptr, clip2: ptr, token_ids: ptr) -> ptr:
    """Run both CLIP models for SDXL and return combined dict.
    
    clip1: ViT-L/14 -> (1, 77, 768)
    clip2: ViT-bigG -> (1, 77, 1280)
    token_ids: (1, 77) int64
    
    Returns opaque struct with pooled and text_emb tensors.
    """
    return torch_std_sdxl_dual_clip(clip1, clip2, token_ids)


def free_model(module: ptr) -> void:
    """Free JIT module (not directly supported yet - GC handles it)."""
    pass  # JIT modules are managed by C++ GC / RAII
