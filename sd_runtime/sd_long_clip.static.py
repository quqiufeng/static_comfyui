# sd_runtime/sd_long_clip.static.py — Long CLIP-L 配置 (Phase 3 gap)
# 对位 comfyui_ref/comfy/text_encoders/long_clipl.py
#
# Long CLIP-L: extended sequence length (up to 248 tokens)
# Reuses CLIP tokenizer + model infrastructure with different config.

from ops import *
from clip_tokenizer import *
from clip_model import *


# Long CLIP-L 配置
LONG_CLIP_MAX_TOKENS: int = 248
LONG_CLIP_EMBED_DIM: int = 768
LONG_CLIP_N_LAYERS: int = 24
LONG_CLIP_N_HEADS: int = 12


# 全局状态
_lc_jit_module: ptr
_lc_tokenizer: ptr


def long_clip_init(jit_model_path: str, vocab_path: str,
                   merges_path: str) -> void:
    """Initialize Long CLIP-L model."""
    global _lc_jit_module, _lc_tokenizer
    _lc_jit_module = torch_std_jit_load(jit_model_path)
    _lc_tokenizer = torch_std_clip_tokenizer_create(vocab_path, merges_path)


def long_clip_encode(text: str) -> ptr:
    """Encode text with Long CLIP-L.
    
    Returns: (1, 248, 768) embeddings
    """
    global _lc_jit_module, _lc_tokenizer
    tokens = torch_std_clip_tokenizer_encode(_lc_tokenizer, text)
    # Pad or truncate to LONG_CLIP_MAX_TOKENS
    cur_len = int(torch_std_size(tokens, 1))
    if cur_len < LONG_CLIP_MAX_TOKENS:
        pad_len = LONG_CLIP_MAX_TOKENS - cur_len
        pad_shape = make_int_array(2)
        int_array_set(pad_shape, 0, 1)
        int_array_set(pad_shape, 1, pad_len)
        pad = torch_std_full(pad_shape, 2, 0.0, 2)  # pad with 0 token IDs
        tensors = make_ptr_array(2)
        ptr_array_set(tensors, 0, tokens)
        ptr_array_set(tensors, 1, pad)
        tokens = torch_std_cat(tensors, 2, 1)
    else:
        tokens = torch_std_narrow(tokens, 1, 0, LONG_CLIP_MAX_TOKENS)
    
    return torch_std_clip_text_forward(_lc_jit_module, tokens, 0)


def long_clip_free() -> void:
    global _lc_jit_module, _lc_tokenizer
    torch_std_clip_tokenizer_free(_lc_tokenizer)
    torch_std_jit_module_delete(_lc_jit_module)
