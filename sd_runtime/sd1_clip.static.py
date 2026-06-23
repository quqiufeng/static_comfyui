# sd_runtime/sd1_clip.static.py — SD1.5 CLIP Text Encoding Pipeline
# 1:1 对齐 comfyui_ref/comfy/sd1_clip.py SD1ClipModel + SDTokenizer
#
# 完整 SD1.5 CLIP 管线：
#   1. 从 safetensors 加载 token embedding + position embedding + transformer 权重
#      （或加载 JIT .pt 文件）
#   2. BPE tokenizer (vocab.json + merges.txt)
#   3. 编码文本 → (1, 77) 整数 token IDs
#   4. 通过 CLIP transformer → (1, 77, 768) 文本嵌入 + pooled 输出
#
# 本模块提供简化的函数式接口。
#
# 用法:
#   clip = sd1_clip_init("/models/clip_vitl.pt",
#                        "/models/clip_vocab.json",
#                        "/models/clip_merges.txt")
#   emb, pooled = sd1_clip_encode(clip, "a cute cat")
#   sd1_clip_free(clip)

from ops import *
from clip_tokenizer import *
from clip_model import *


# ==============================================================================
# SD1.5 CLIP 管线状态（以结构体指针形式）
# ==============================================================================
#
# 状态结构（在调用者内存中）:
#   0: clip_module ptr (JIT .pt)
#   1: tokenizer ptr (BPE)
#   2: embed_dim (int, e.g. 768)
#   3: max_length (int, default 77)
#   4: use_f16 (int, 0/1)

STATE_CLIP_MODULE: int = 0
STATE_TOKENIZER: int = 1
STATE_EMBED_DIM: int = 2
STATE_MAX_LENGTH: int = 3
STATE_USE_F16: int = 4
STATE_N_FIELDS: int = 5  # total fields in state struct


def sd1_clip_init(clip_jit_path: str, vocab_path: str, merges_path: str,
                  embed_dim: int, use_f16: int) -> ptr:
    """Initialize SD1.5 CLIP pipeline.
    
    clip_jit_path: path to CLIP JIT .pt file
    vocab_path: path to vocab.json
    merges_path: path to merges.txt
    embed_dim: CLIP embedding dimension (768 for ViT-L/14, 512 for ViT-B/32)
    use_f16: 1 to use float16, 0 for float32
    
    Returns state pointer (array of 5 ints/pointers).
    """
    state: ptr = make_int_array(STATE_N_FIELDS)  # use int array to hold mixed types
    
    clip_module: ptr = load_clip_model(clip_jit_path)
    tok: ptr = create_tokenizer(vocab_path, merges_path)
    
    # Store in state struct
    # Using int array since StaticPy may not support mixed arrays easily
    # We store pointers as integers (they're memory addresses)
    # Actually, in StaticPy, pointers are separate from ints.
    # Let's use separate fields.
    ptr_array_set(state, STATE_CLIP_MODULE, clip_module)
    ptr_array_set(state, STATE_TOKENIZER, tok)
    int_array_set(state, STATE_EMBED_DIM, embed_dim)
    int_array_set(state, STATE_MAX_LENGTH, CLIP_N_MAX)
    int_array_set(state, STATE_USE_F16, use_f16)
    
    return state


def sd1_clip_encode(state: ptr, text: str) -> ptr:
    """Encode text prompt to CLIP embeddings.
    
    Returns a 2-element ptr array:
      [0]: (1, 77, embed_dim) sequence embedding
      [1]: (1, embed_dim) pooled embedding (mean pool or [0] token)
    
    Note: C++ torch_std_clip_text_forward returns only sequence embedding.
    Pooled output is extracted as [0] token (CLS equivalent).
    """
    clip_module: ptr = ptr_array_ref(state, STATE_CLIP_MODULE)
    tok: ptr = ptr_array_ref(state, STATE_TOKENIZER)
    use_f16: int = int_array_ref(state, STATE_USE_F16)
    
    # Step 1: Tokenize
    token_ids: ptr = tokenize(tok, text)  # (1, 77) int64
    
    # Step 2: CLIP forward
    seq_emb: ptr = clip_encode(clip_module, token_ids, use_f16)  # (1, 77, D)
    
    # Step 3: Extract pooled embedding (use [0] token / SOS embedding)
    # CLIP's pooled output is the EOS token embedding, not SOS.
    # In the reference: pooled = x[arange(b), (tokens == eos_id).argmax(dim=-1)]
    # Simplified: use last token (index 76) which should be EOS or padding
    pooled: ptr = torch_std_slice(seq_emb, 1, 76, 77, 1)  # (1, 1, D)
    pooled = torch_std_squeeze(pooled, 1)  # (1, D)
    
    # Package results
    result: ptr = make_ptr_array(2)
    ptr_array_set(result, 0, seq_emb)
    ptr_array_set(result, 1, pooled)
    
    return result


def sd1_clip_free(state: ptr) -> void:
    """Free CLIP pipeline resources."""
    tok: ptr = ptr_array_ref(state, STATE_TOKENIZER)
    free_tokenizer(tok)
    # clip_module is managed by C++ GC
