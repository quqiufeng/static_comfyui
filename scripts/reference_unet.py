#!/usr/bin/env python3
"""reference_unet.py — 用 comfyui_ref 源码跑 UNet forward，输出 reference bin

用法:
  python3 scripts/reference_unet.py \
      --model /data/models/image/sd_xl_base_1.0.safetensors \
      --tokens-l /tmp/tokens_l.bin --tokens-g /tmp/tokens_g.bin \
      --image-dim /tmp/image_dim.bin \
      --out-dir /tmp/ref_out
"""
import argparse, os, sys, json, struct
import numpy as np
import torch

sys.path.insert(0, '/opt/static_comfyui/comfyui_ref')
import comfy.model_management as mm
from comfy import model_detection, supported_models
from comfy.sd import load_checkpoint_guess_config
from comfy.sdxl_clip import SDXLTokenizer, SDXLClipModel
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from comfy.ldm.modules.diffusionmodules.util import timestep_embedding
from safetensors import safe_open

def load_sdxl_unet_state_dict(safetensors_path):
    with safe_open(safetensors_path, framework='pt', device='cpu') as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
    # keep only diffusion_model keys
    return {k: v for k, v in sd.items() if k.startswith('model.diffusion_model.')}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/data/models/image/sd_xl_base_1.0.safetensors')
    parser.add_argument('--tokens-l', type=str, default='/tmp/tokens_l.bin')
    parser.add_argument('--tokens-g', type=str, default='/tmp/tokens_g.bin')
    parser.add_argument('--image-dim', type=str, default='/tmp/image_dim.bin')
    parser.add_argument('--latent-value', type=float, default=0.1)
    parser.add_argument('--timestep', type=float, default=50.0)
    parser.add_argument('--out-dir', type=str, default='/tmp/ref_out')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Tokenizer + CLIP encode
    tokenizer = SDXLTokenizer()
    token_weights = tokenizer.tokenize_with_weights("dummy")  # placeholder; we use provided tokens
    clip_model = SDXLClipModel(device='cuda', dtype=torch.float16)
    # Actually we need load_sd. For CLIP we directly use provided tokens and compute embeddings
    # But without loading weights we can't. So this script expects tokens already encode context/pooled
    # from main pipeline? No — reference needs its own CLIP weights.
    # Better: load full checkpoint via ComfyUI loader.

    print("Loading full checkpoint via ComfyUI...")
    # Load checkpoint and get model, clip, vae
    # This is the simplest way to get aligned CLIP + UNet.
    from comfy.sd import load_checkpoint_guess_config
    out = load_checkpoint_guess_config(args.model, output_vae=True, output_clip=True, output_model=True)
    model, clip, vae, _ = out

    # tokens from files
    tokens_l = np.fromfile(args.tokens_l, dtype=np.int32).tolist()
    tokens_g = np.fromfile(args.tokens_g, dtype=np.int32).tolist()

    # CLIP encode
    token_weights = {"l": [[(t, 1.0) for t in tokens_l]], "g": [[(t, 1.0) for t in tokens_g]]}
    ctx, pooled_g = clip.encode_token_weights(token_weights)
    ctx = ctx.half().cuda()
    pooled_g = pooled_g.half().cuda()

    # y conditioning
    image_dim = np.fromfile(args.image_dim, dtype=np.float32)
    y = torch.cat([pooled_g, torch.from_numpy(image_dim).half().cuda().unsqueeze(0)], dim=1)

    # latent
    latent = torch.full((1, 4, 64, 64), args.latent_value, dtype=torch.float16, device='cuda')
    ts = torch.Tensor([args.timestep]).half().cuda()

    # UNet forward
    print("UNet forward...")
    with torch.no_grad():
        result = model.model.diffusion_model(latent, ts, context=ctx, y=y)

    # save
    ctx.float().cpu().numpy().tofile(os.path.join(args.out_dir, 'ctx.bin'))
    pooled_g.float().cpu().numpy().tofile(os.path.join(args.out_dir, 'pooled_g.bin'))
    y.float().cpu().numpy().tofile(os.path.join(args.out_dir, 'y.bin'))
    result.float().cpu().numpy().tofile(os.path.join(args.out_dir, 'unet_output.bin'))

    print(f"Saved reference outputs to {args.out_dir}")
    print(f"ctx sum={ctx.float().sum().item():.2f}")
    print(f"unet_output sum={result.float().sum().item():.2f}")

if __name__ == '__main__':
    main()
