from comfy_types import CLIPType


CLIP_MAX_TOKEN_LENGTH = 77

CLIP_VOCAB_SIZE = 49408
CLIP_EOS_TOKEN = 49407
CLIP_SOS_TOKEN = 49406
CLIP_PAD_TOKEN_L = 49407
CLIP_PAD_TOKEN_G = 0

CLIP_L_EMBED_DIM = 768
CLIP_G_EMBED_DIM = 1280
CLIP_L_NUM_LAYERS = 12
CLIP_G_NUM_LAYERS = 32

T5_MAX_TOKEN_LENGTH = 512


def clip_tokenizer_create(vocab_path: str, merges_path: str):
    return torch.clip_tokenizer_create(vocab_path, merges_path)


def clip_tokenizer_encode(tokenizer, text: str):
    return torch.clip_tokenizer_encode(tokenizer, text)


def clip_tokenizer_free(tokenizer):
    torch.clip_tokenizer_free(tokenizer)


def clip_text_forward(clip_module, token_ids, cast_to_float16: bool):
    return torch.clip_text_forward(clip_module, token_ids, cast_to_float16)


def sdxl_dual_clip(clip_l, clip_g, token_ids):
    return torch.sdxl_dual_clip(clip_l, clip_g, token_ids)


def sdxl_get_pooled():
    return torch.sdxl_get_pooled()


def sdxl_get_pooled_l():
    return torch.sdxl_get_pooled_l()


def t5_tokenizer_create(model_path: str):
    return torch.t5_tokenizer_create(model_path)


def t5_tokenizer_encode(tokenizer, text: str, max_len: int):
    return torch.t5_tokenizer_encode(tokenizer, text, max_len)


def t5_tokenizer_free(tokenizer):
    torch.t5_tokenizer_free(tokenizer)


def encode_sd15(clip_module, tokenizer, text: str, cast_fp16: bool):
    token_ids = clip_tokenizer_encode(tokenizer, text)
    return clip_text_forward(clip_module, token_ids, cast_fp16)


def encode_sdxl(clip_l_dict, clip_g_dict, tokenizer_l, tokenizer_g, text: str):
    tokens_l = clip_tokenizer_encode(tokenizer_l, text)
    # CLIP-L: dim=768, layers=12, heads=12, ffn=3072
    emb_l = torch.clip_text_forward_from_dict(clip_l_dict, tokens_l, 768, 12, 12, 3072)
    # CLIP-G: dim=1280, layers=32, heads=20, ffn=5120
    emb_g = torch.clip_text_forward_from_dict(clip_g_dict, tokens_l, 1280, 32, 20, 5120)
    result_list = py_list(emb_l, emb_g)
    text_emb = torch.cat(result_list, 2)
    # Pooled: approximate EOS is at position 60 (after text, before padding)
    pooled = torch.narrow(emb_g, 1, 60, 1)
    pooled = torch.squeeze(pooled, 1)
    return text_emb, pooled


def encode_flux(t5_tokenizer, t5_model, clip_l_module, clip_l_tok, text: str, max_len: int):
    t5_ids = t5_tokenizer_encode(t5_tokenizer, text, max_len)
    clip_ids = clip_tokenizer_encode(clip_l_tok, text)
    t5_emb = torch.jit_forward(t5_model, t5_ids)
    clip_emb = clip_text_forward(clip_l_module, clip_ids, True)
    return t5_emb, clip_emb


def encode_t5(t5_tokenizer, t5_model, text: str, max_len: int):
    token_ids = t5_tokenizer_encode(t5_tokenizer, text, max_len)
    return torch.jit_forward(t5_model, token_ids)


def main():
    pass
