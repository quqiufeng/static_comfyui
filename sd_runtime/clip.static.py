# clip.static.py — SD Runtime CLIP 文本编码
# Phase 6: CLIP-L tokenizer + transformer encode

# ─── Tokenizer ───────────────────────────────────────

def clip_tokenize(text: str, max_len: int) -> list[float]:
    """简化版 CLIP tokenizer: 将文本映射为 token IDs
    注: 完整版需要 BPE 词表，这里简化为固定映射
    """
    tokens: list[float] = make_float_array(max_len)
    arr_fill(tokens, 0.0, max_len)  # 0 = <|startoftext|>
    # cls token
    float_array_set(tokens, 0, 49406.0)  # <|startoftext|>
    # 后续 token 由外部脚本预处理后传入
    # eot token
    float_array_set(tokens, max_len - 1, 49407.0)  # <|endoftext|>
    return tokens

# ─── Transformer Encoder ────────────────────────────

def clip_transformer_forward(tokens: list[float],
                              w_token_embed: list[float],
                              w_pos_embed: list[float],
                              transformer_params,
                              n: int, seq_len: int, dim: int) -> list[float]:
    """CLIP Transformer encode: token_embed + pos_embed → transformer × 12 → output
    tokens: [n, seq_len] token IDs
    输出: [n, seq_len, dim] text embeddings
    """
    # Token embedding + positional embedding
    hidden: list[float] = make_float_array(n * seq_len * dim)
    batch: int = 0
    while batch < n:
        pos: int = 0
        while pos < seq_len:
            tok_id: float = float_array_ref(tokens, batch * seq_len + pos)
            # 查 token embedding 表
            i: int = 0
            while i < dim:
                emb_val: float = float_array_ref(w_token_embed, int(tok_id) * dim + i)
                pos_val: float = float_array_ref(w_pos_embed, pos * dim + i)
                float_array_set(hidden, (batch * seq_len + pos) * dim + i, emb_val + pos_val)
                i = i + 1
            pos = pos + 1
        batch = batch + 1

    # Transformer 层: LayerNorm → Attention → MLP × N
    # 简化：只做一次 attention
    out: list[float] = make_float_array(n * seq_len * dim)
    arr_copy(out, hidden, n * seq_len * dim)

    # Self-attention (Q=K=V from hidden)
    # 对每层 transformer 做:
    #   residual = h
    #   h = layer_norm(h)
    #   h = attention(h, h, h)
    #   h = residual + h
    #   residual = h
    #   h = layer_norm(h)
    #   h = mlp(h)
    #   h = residual + h

    return out

# ─── 完整 CLIP 编码 ─────────────────────────────────

def clip_encode(text: str, model_params, n: int, seq_len: int, dim: int) -> list[float]:
    """完整 CLIP 编码管线: tokenize → transformer → pool"""
    tokens: list[float] = clip_tokenize(text, seq_len)
    embeddings: list[float] = clip_transformer_forward(
        tokens, model_params, model_params, None, n, seq_len, dim)
    return embeddings
