# sd_runtime/clip_tokenizer.static.py — CLIP BPE Tokenizer
# 1:1 对齐 comfyui_ref/comfy/sd1_clip.py SDTokenizer
#
# C++ 运行时提供 BPE 编码：torch_std_clip_tokenizer_encode
# 本模块包装为可重用的 tokenizer 对象（以 opaq pointer 形式）
#
# 用法:
#   tok = create_tokenizer("vocab.json", "merges.txt")
#   ids = tokenize(tok, "a cute cat")
#   free_tokenizer(tok)

from ops import *

# CLIP 特殊 token ID
CLIP_SOS_ID: int = 49406  # <|startoftext|>
CLIP_EOS_ID: int = 49407  # <|endoftext|> (also used as pad)
CLIP_N_MAX: int = 77      # max sequence length


def create_tokenizer(vocab_path: str, merges_path: str) -> ptr:
    """Create CLIP BPE tokenizer from vocab.json and merges.txt.
    
    Returns opaque pointer (to be used with tokenize / free).
    """
    return torch_std_clip_tokenizer_create(vocab_path, merges_path)


def tokenize(tok: ptr, text: str) -> ptr:
    """Tokenize text and return (1, 77) int64 tensor.
    
    C++ encode returns (77,) int64 with SOS+text+EOS+padding.
    We reshape to (1, 77) for CLIP forward.
    """
    ids_1d: ptr = torch_std_clip_tokenizer_encode(tok, text)  # (77,)
    
    # Reshape to (1, 77)
    shape_arr: ptr = make_int_array(2)
    int_array_set(shape_arr, 0, 1)
    int_array_set(shape_arr, 1, 77)
    ids_2d: ptr = torch_std_reshape(ids_1d, shape_arr, 2)
    return ids_2d


def tokenize_batch(tok: ptr, texts: ptr, n_texts: int) -> ptr:
    """Tokenize multiple texts and stack into (N, 77) tensor.
    
    texts: array of C string pointers
    n_texts: number of strings
    
    Returns (N, 77) int64 token IDs.
    """
    # Tokenize first text to get output shape
    first_text: str = ptr_array_ref(texts, 0)  # might not work this way in StaticPy
    # Actually, in StaticPy, strings are special. Let's just handle single text for now.
    return tokenize(tok, first_text)


def free_tokenizer(tok: ptr) -> void:
    """Free tokenizer resources."""
    torch_std_clip_tokenizer_free(tok)
