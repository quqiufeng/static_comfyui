# sd_runtime/sd_embed.static.py — CLIP embedding 工具 (Phase 2/3 gap)
# Flat functions.

from ops import *


def encode_token_embedding(clip_module: ptr, tokens: ptr) -> ptr:
    """CLIP text forward → (1, 77, 768) float32 embeddings."""
    return torch_std_clip_text_forward(clip_module, tokens, 0)


def pad_tokens(tokens: ptr, target_len: int, pad_token_id: int) -> ptr:
    """Pad token sequence to target_len."""
    cur_len = int(tensor_size(tokens, 1))
    if cur_len >= target_len:
        return torch_std_narrow(tokens, 1, 0, target_len)
    pad_len = target_len - cur_len
    shape = make_int_array(2)
    int_array_set(shape, 0, 1)
    int_array_set(shape, 1, pad_len)
    pad = torch_std_full(shape, 2, float(pad_token_id), 2)  # int64
    tensors = make_ptr_array(2)
    ptr_array_set(tensors, 0, tokens)
    ptr_array_set(tensors, 1, pad)
    return torch_std_cat(tensors, 2, 1)


def extract_pooled(emb: ptr) -> ptr:
    """Extract EOS token pooling from CLIP output.
    
    emb: (1, 77, 768) → (1, 1, 768) from last token.
    """
    return torch_std_narrow(emb, 1, 76, 1)


def mean_pool(emb: ptr, mask: ptr) -> ptr:
    """Mean-pool embeddings along sequence dim with mask."""
    if mask != null:
        m = torch_std_unsqueeze(mask, 2)
        masked = torch_std_mul(emb, m)
        summed = torch_std_sum_dim(masked, 1, 0)
        msum = torch_std_sum_dim(mask, 1, 0)
        msum = torch_std_clamp(msum, 1.0, 1e10)
        return torch_std_div(summed, msum)
    return torch_std_mean_dim(emb, 1, 0)
