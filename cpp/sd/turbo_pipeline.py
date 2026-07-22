#!/data/venv/bin/python
"""
z_image_turbo (Lumina2) PyTorch CUDA inference
Reads GGUF weights, builds matching DiT architecture, runs on CUDA.
"""
import os, sys, time, math, hashlib, json
from dataclasses import dataclass
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, '/data/venv/lib/python3.12/site-packages')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MODEL_DIR = '/data/models/image'

# ─── GGUF weight reader ─────────────────────────────────────────────────────

from gguf import GGUFReader, dequantize, GGMLQuantizationType

def load_gguf_tensors(path):
    """Read GGUF, return dict of torch tensors in PyTorch layout"""
    r = GGUFReader(path)
    state = {}
    for t in r.tensors:
        name = t.name
        dtype = t.tensor_type
        raw = np.asarray(t.data, dtype=np.uint8)

        if dtype == GGMLQuantizationType.F32:
            arr = raw.view(np.float32)
        elif dtype == GGMLQuantizationType.BF16:
            as_u16 = raw.view(np.uint16)
            f32_u32 = as_u16.astype(np.uint32) << 16
            arr = f32_u32.view(np.float32)
        elif dtype in (GGMLQuantizationType.Q5_K, GGMLQuantizationType.Q6_K,
                       GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q8_0):
            flat = dequantize(raw.ravel(), dtype)
            arr = flat
        else:
            flat = dequantize(raw.ravel(), dtype)
            arr = flat

        # Reshape to match tensor dimensions, then transpose for PyTorch
        shape = [int(s) for s in t.shape]
        if arr.ndim == 1 and len(shape) > 1:
            n_expected = int(np.prod(shape))
            # For quantized tensors, dequantize may return more elements due to padding
            arr = arr[:n_expected]
            arr = arr.reshape(shape[0], shape[1]).T
        elif len(shape) == 1:
            pass  # 1D tensors keep as-is

        state[name] = torch.from_numpy(np.ascontiguousarray(arr)).float()
    return state

# ─── DiT components (Lumina2) ────────────────────────────────────────────────

class DiTBlock(nn.Module):
    def __init__(self, hidden=3840, ffn_hidden=10240):
        super().__init__()
        self.attn_norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn_norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ffn_norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ffn_norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        # Self-attention
        self.to_q = nn.Linear(hidden, hidden, bias=False)
        self.to_k = nn.Linear(hidden, hidden, bias=False)
        self.to_v = nn.Linear(hidden, hidden, bias=False)
        self.to_out = nn.Linear(hidden, hidden, bias=False)
        self.norm_q = nn.LayerNorm(hidden // 30, elementwise_affine=True)  # QK-norm
        self.norm_k = nn.LayerNorm(hidden // 30, elementwise_affine=True)
        # Gated FFN (SwiGLU)
        self.ff_gate = nn.Linear(hidden, ffn_hidden, bias=False)
        self.ff_up = nn.Linear(hidden, ffn_hidden, bias=False)
        self.ff_down = nn.Linear(ffn_hidden, hidden, bias=False)

    def forward(self, x, cond):
        B, N, C = x.shape
        H = 30
        Dh = C // H

        # Pre-norm
        xn = self.attn_norm1(x)
        q = self.to_q(xn).view(B, N, H, Dh).transpose(1, 2)
        k = self.to_k(xn).view(B, N, H, Dh).transpose(1, 2)
        v = self.to_v(xn).view(B, N, H, Dh).transpose(1, 2)

        q = self.norm_q(q)
        k = self.norm_k(k)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        x = x + self.to_out(out)

        # FFN
        xn = self.ffn_norm1(x)
        gate = F.silu(self.ff_gate(xn))
        up = self.ff_up(xn)
        x = x + self.ff_down(gate * up)
        return x

class Lumina2(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embedder = nn.Linear(64, 3840, bias=True)
        self.t_embedder = nn.Sequential(
            nn.Linear(256, 1024, bias=True),
            nn.SiLU(),
            nn.Linear(1024, 256, bias=True),
        )
        self.cap_embedder = nn.Sequential(
            nn.LayerNorm(2560),
            nn.Linear(2560, 3840, bias=True),
        )
        self.cap_pad_token = nn.Parameter(torch.zeros(3840))
        self.context_refiner = nn.ModuleList([
            DiTBlock() for _ in range(2)
        ])
        self.noise_refiner = nn.ModuleList([
            DiTBlock() for _ in range(2)
        ])
        self.layers = nn.ModuleList([
            DiTBlock() for _ in range(30)
        ])
        self.final_norm = nn.LayerNorm(3840, elementwise_affine=True)
        self.final_linear = nn.Linear(3840, 64, bias=True)

    def timestep_embed(self, t, dim=256):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x, t, context):
        B, C, H, W = x.shape
        p = 4
        x = F.unfold(x, kernel_size=p, stride=p)
        x = x.permute(0, 2, 1)
        x = self.x_embedder(x)

        t_emb = self.timestep_embed(t)
        t_emb = self.t_embedder(t_emb)

        pad = self.cap_pad_token[None, None, :].expand(B, -1, -1)
        ctx = torch.cat([context, pad], dim=1)
        for ref in self.context_refiner:
            ctx = ref(ctx, t_emb)

        tokens = torch.cat([ctx, x], dim=1)
        for layer in self.layers:
            tokens = layer(tokens, t_emb)

        x_out = tokens[:, -x.shape[1]:]
        x_out = self.final_norm(x_out)
        x_out = self.final_linear(x_out)

        x_out = x_out.permute(0, 2, 1)
        x_out = F.fold(x_out, output_size=(H, W), kernel_size=p, stride=p)
        return x_out

# ─── Text encoder ───────────────────────────────────────────────────────────

from llama_cpp import Llama

class QwenEncoder:
    def __init__(self):
        self.llm = Llama(
            model_path=f'{MODEL_DIR}/Qwen3-4B-Instruct-2507-Q4_K_M.gguf',
            n_ctx=4096, n_gpu_layers=0, embedding=True, verbose=False,
        )
    def encode(self, text):
        emb = self.llm.create_embedding(text)
        vecs = [e['embedding'] for e in emb['data']]
        return torch.tensor(vecs, dtype=torch.float32)

# ─── VAE ────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ComfyUI'))
from comfy.sd import load_checkpoint_guess_config

def load_vae():
    _, _, vae, _ = load_checkpoint_guess_config(
        f'{MODEL_DIR}/ae.safetensors')
    return vae

# ─── k-diffusion sampling ───────────────────────────────────────────────────

@torch.no_grad()
def sample(model, noise, sigmas, cfg, cond, uncond):
    x = noise * sigmas[0]
    for i in range(len(sigmas) - 1):
        t = torch.log(sigmas[i]).expand(noise.shape[0])
        pred_cond = model(x, t, cond)
        pred_uncond = model(x, t, uncond)
        pred = pred_uncond + cfg * (pred_cond - pred_uncond)
        d = (x - pred) / sigmas[i]
        x = x + d * (sigmas[i + 1] - sigmas[i])
    return x

def make_sigmas(n, sigma_min=0.0292, sigma_max=14.6146, device='cpu'):
    ramp = torch.linspace(0, 1, n, device=device)
    return (sigma_max ** (1/7) + ramp * (sigma_min ** (1/7) - sigma_max ** (1/7))) ** 7

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "portrait"
    out_path = sys.argv[2] if len(sys.argv) > 2 else ''
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 2560
    height = int(sys.argv[4]) if len(sys.argv) > 4 else 1440
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42
    cfg = float(os.environ.get('CFG', '3.0'))
    steps = int(os.environ.get('STEPS', '20'))
    hires_steps = int(os.environ.get('HIRES_STEPS', '30'))
    hires_strength = float(os.environ.get('HIRES_STRENGTH', '0.35'))

    torch.manual_seed(seed)
    device = 'cuda'

    quality_prefix = ("masterpiece, best quality, ultra-detailed, photorealistic, "
                      "highly detailed, sharp focus, realistic skin texture")
    full_prompt = f"{quality_prefix}, {prompt}" if "masterpiece" not in prompt else prompt
    neg_prompt = ("blurry, low quality, worst quality, bad anatomy, deformed, "
                  "oily skin, plastic skin, jpeg artifacts")

    # Load model
    print("Loading z_image_turbo GGUF...")
    state = load_gguf_tensors(f'{MODEL_DIR}/z_image_turbo-Q5_K_M.gguf')
    model = Lumina2().to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # VAE
    print("Loading VAE...")
    vae = load_vae()
    vae.to(device)

    # Text
    print("Encoding text...")
    enc = QwenEncoder()
    cond = enc.encode(full_prompt).unsqueeze(0).to(device)
    uncond = enc.encode(neg_prompt).unsqueeze(0).to(device)

    # Cap embedder
    with torch.no_grad():
        cond = model.cap_embedder(cond.to(device))
        uncond = model.cap_embedder(uncond.to(device))

    # Pad latent dimensions
    def pad8(h, w):
        return ((h + 7) // 8 * 8, (w + 7) // 8 * 8)

    target_h, target_w = pad8(height // 8, width // 8)
    low_h = ((target_h * 4 // 5) + 3) // 4 * 4
    low_w = ((target_w * 4 // 5) + 3) // 4 * 4

    print(f"Low-res: {low_w}x{low_h} → Target: {target_w}x{target_h}")

    # Low-res
    noise = torch.randn(1, 4, low_h, low_w, device=device)
    sigmas = make_sigmas(steps, device=device)
    samples = sample(model, noise, sigmas, cfg, cond, uncond)

    # HiRes
    samples = F.interpolate(samples, (target_h, target_w), mode='bilinear')
    noise_h = torch.randn(1, 4, target_h, target_w, device=device)
    hs = make_sigmas(hires_steps, device=device)
    samples = sample(model, noise_h, hs, cfg, cond, uncond)

    # Trim & VAE
    samples = samples[:, :, :height//8, :width//8]
    decoded = vae.decode(samples.float())
    img = decoded[0].cpu().float()
    if img.dim() == 3 and img.size(0) <= 4:
        img = img.permute(1, 2, 0)
    img = (img * 255).clamp(0, 255).byte()

    from PIL import Image
    if not out_path:
        out_path = os.path.expanduser(f'~/{datetime.now():%Y%m%d_%H%M%S}_turbo.png')
    os.makedirs(os.path.dirname(os.path.expanduser(out_path)) or '.', exist_ok=True)
    Image.fromarray(img.numpy()).save(os.path.expanduser(out_path))
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
