#!/usr/bin/env python3
"""preprocess_prompt.py — 用 transformers 的 CLIPTokenizer 生成 tokens + SDXL y conditioning

输出:
  /tmp/tokens_l.bin   : 77 int32 token ids for CLIP-L
  /tmp/tokens_g.bin   : 77 int32 token ids for CLIP-G
  /tmp/y.bin          : 2816 float32 for SDXL adm_in_channels

用法:
  python3 scripts/preprocess_prompt.py "a photo of a cat" --width 1024 --height 1024
"""
import argparse, json, struct, math, os
import numpy as np
from transformers import CLIPTokenizer

def timestep_embedding(t: float, dim: int = 256, max_period: float = 10000.0):
    """ComfyUI util.timestep_embedding, for a scalar t."""
    half = dim // 2
    freqs = np.exp(-math.log(max_period) * np.arange(half, dtype=np.float32) / half)
    args = np.array([t], dtype=np.float32)[:, None] * freqs[None]
    emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    if dim % 2:
        emb = np.concatenate([emb, np.zeros_like(emb[:, :1])], axis=-1)
    return emb.flatten()

def encode_adm_sdxl(pooled_g: np.ndarray, width: int, height: int,
                    crop_w: int = 0, crop_h: int = 0,
                    target_width: int = None, target_height: int = None) -> np.ndarray:
    """ComfyUI SDXL.encode_adm 的 numpy 实现."""
    if target_width is None: target_width = width
    if target_height is None: target_height = height
    parts = [pooled_g.flatten()]
    parts.append(timestep_embedding(height, 256))
    parts.append(timestep_embedding(width, 256))
    parts.append(timestep_embedding(crop_h, 256))
    parts.append(timestep_embedding(crop_w, 256))
    parts.append(timestep_embedding(target_height, 256))
    parts.append(timestep_embedding(target_width, 256))
    return np.concatenate(parts).astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str, default='a photo of a cat')
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--crop-w', type=int, default=0)
    parser.add_argument('--crop-h', type=int, default=0)
    parser.add_argument('--target-width', type=int, default=None)
    parser.add_argument('--target-height', type=int, default=None)
    parser.add_argument('--tokenizer-path', type=str,
                        default='/opt/static_comfyui/comfyui_ref/comfy/sd1_tokenizer')
    parser.add_argument('--out-dir', type=str, default='/tmp')
    args = parser.parse_args()

    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_path)

    # CLIP-L and CLIP-G use the same tokenizer (openai/clip-vit-large-patch14)
    # but different max_length / special tokens padding behavior in ComfyUI.
    # For SDXL both are 77, start=49406, end=49407, pad=0.
    ids = tokenizer(args.prompt, max_length=77, padding='max_length',
                    truncation=True, return_tensors='np')['input_ids'][0]

    # CLIP-L padding uses pad=0 (same as ids above)
    tokens_l = ids.astype(np.int32)
    # CLIP-G: ComfyUI SDXLClipGTokenizer sets pad_with_end=False, so padding is 0 too
    tokens_g = ids.astype(np.int32)

    # dummy pooled_g for y construction; in real pipeline it comes from clip_g text_projection
    # Here we export y as all zeros + image dim embeds, because pooled_g must be computed by
    # clip_g encoder. The StaticPy main.static.py will concatenate real pooled_g with image dim.
    # Wait: easier approach — compute pooled_g here using CLIP model? No, we don't have CLIP-G
    # text_projection weights in this script. So we export only image_dim part and let main.static.py
    # concatenate after clip_encode_lg.
    # But y is input to UNet, so main.static.py must build it. We'll save image_dim separately.
    image_dim = encode_adm_sdxl(np.zeros(1280, dtype=np.float32),
                                args.width, args.height,
                                args.crop_w, args.crop_h,
                                args.target_width, args.target_height)[1280:]

    out_dir = args.out_dir
    tokens_l.tofile(os.path.join(out_dir, 'tokens_l.bin'))
    tokens_g.tofile(os.path.join(out_dir, 'tokens_g.bin'))
    image_dim.astype(np.float32).tofile(os.path.join(out_dir, 'image_dim.bin'))
    print(f"Wrote tokens_l.bin, tokens_g.bin, image_dim.bin to {out_dir}")
    print(f"tokens: {tokens_l[:5]}...")
    print(f"image_dim shape: {image_dim.shape}")

if __name__ == '__main__':
    main()
