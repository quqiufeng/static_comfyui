# sd_runtime/sdxl_clip.static.py — SDXL Dual CLIP Pipeline
# 1:1 对齐 comfyui_ref/comfy/sdxl_clip.py
#
# SDXL uses two CLIP models:
#   clip_l: ViT-L/14 -> (1, 77, 768)
#   clip_g: ViT-bigG -> (1, 77, 1280)
#
# Output: concatenated (1, 77, 2048) + pooled (1, 1280)
#
# C++ torch_std_sdxl_dual_clip handles both CLIP forwards internally.
#
# 用法:
#   state = sdxl_clip_init(clip_l_path, clip_g_path, vocab_path, merges_path)
#   seq_emb, pooled = sdxl_clip_encode(state, "a cat")
#   sdxl_clip_free(state)

from ops import *
from clip_tokenizer import *
from clip_model import *

# State struct (pointer array):
#   0: clip_l module (JIT)
#   1: clip_g module (JIT)
#   2: tokenizer ptr
#   3: use_f16 (int)
#   4: embed_dim_l (int, 768)
#   5: embed_dim_g (int, 1280)

STATE_CLIP_L: int = 0
STATE_CLIP_G: int = 1
STATE_TOKENIZER: int = 2
STATE_USE_F16: int = 3
STATE_EMBED_DIM_L: int = 4
STATE_EMBED_DIM_G: int = 5
STATE_N_FIELDS: int = 6


def sdxl_clip_init(clip_l_path: str, clip_g_path: str,
                   vocab_path: str, merges_path: str,
                   use_f16: int) -> ptr:
    """Initialize SDXL Dual CLIP pipeline."""
    state: ptr = make_int_array(STATE_N_FIELDS)
    
    clip_l: ptr = load_clip_model(clip_l_path)
    clip_g: ptr = load_clip_model(clip_g_path)
    tok: ptr = create_tokenizer(vocab_path, merges_path)
    
    ptr_array_set(state, STATE_CLIP_L, clip_l)
    ptr_array_set(state, STATE_CLIP_G, clip_g)
    ptr_array_set(state, STATE_TOKENIZER, tok)
    int_array_set(state, STATE_USE_F16, use_f16)
    int_array_set(state, STATE_EMBED_DIM_L, 768)
    int_array_set(state, STATE_EMBED_DIM_G, 1280)
    
    return state


def sdxl_clip_encode(state: ptr, text: str) -> ptr:
    """Encode text with both CLIP models.
    
    Returns a 2-element ptr array:
      [0]: (1, 77, 2048) concatenated sequence embedding
      [1]: (1, 1280) pooled embedding (from clip_g)
    """
    clip_l: ptr = ptr_array_ref(state, STATE_CLIP_L)
    clip_g: ptr = ptr_array_ref(state, STATE_CLIP_G)
    tok: ptr = ptr_array_ref(state, STATE_TOKENIZER)
    
    # Tokenize
    token_ids: ptr = tokenize(tok, text)  # (1, 77)
    
    # Run dual CLIP (C++ handles both forwards + concat)
    result_struct: ptr = sdxl_dual_clip(clip_l, clip_g, token_ids)
    
    # result_struct is opaque — the C++ function packs (text_emb, pooled)
    # For now, return as-is (caller unpacks).
    return result_struct


def sdxl_clip_free(state: ptr) -> void:
    """Free SDXL CLIP resources."""
    tok: ptr = ptr_array_ref(state, STATE_TOKENIZER)
    free_tokenizer(tok)
