# sd_runtime/sd_t5.static.py — T5 Text Encoder 编排 (Phase 12)
#
# Flat functions pattern (translate.py 函数级编译)
# 所有状态用全局 ptr 变量存储（单例兼容）
# 不使用方法链调用

from ops import *


# T5Encoder 全局状态
_t5_tok: ptr
_t5_jit: ptr


def t5_init(tokenizer_path: str, jit_model_path: str) -> void:
    """Initialize T5 tokenizer + model."""
    global _t5_tok, _t5_jit
    _t5_tok = torch_std_t5_tokenizer_create(tokenizer_path)
    _t5_jit = torch_std_jit_load(jit_model_path)


def t5_encode(text: str, max_len: int) -> ptr:
    """Tokenize and encode → (1, L, D) embeddings."""
    global _t5_tok, _t5_jit
    tokens = torch_std_t5_tokenizer_encode(_t5_tok, text, max_len)
    return torch_std_jit_forward(_t5_jit, tokens)


def t5_tokenize(text: str, max_len: int) -> ptr:
    """Tokenize only → token IDs."""
    global _t5_tok
    return torch_std_t5_tokenizer_encode(_t5_tok, text, max_len)


def t5_free() -> void:
    """Free T5 resources."""
    global _t5_tok, _t5_jit
    torch_std_t5_tokenizer_free(_t5_tok)
    torch_std_jit_module_delete(_t5_jit)
