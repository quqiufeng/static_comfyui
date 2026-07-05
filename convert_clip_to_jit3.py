#!/usr/bin/env python3
"""Convert CLIP-L/CLIP-G safetensors to TorchScript JIT modules."""

import torch
from safetensors.torch import load_file

def convert(name, in_path, out_path, d_model=768, n_layers=12, n_heads=12, d_ffn=3072):
    sd = load_file(in_path)
    print(f"\n=== {name} ===")
    print(f"Loaded {len(sd)} keys from {in_path}")

    # CLIP text model matching ComfyUI's clip_model.py architecture
    class QuickGELU(torch.nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(1.702 * x)

    class CLIPAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.k_proj = torch.nn.Linear(d_model, d_model, bias=True)
            self.v_proj = torch.nn.Linear(d_model, d_model, bias=True)
            self.q_proj = torch.nn.Linear(d_model, d_model, bias=True)
            self.out_proj = torch.nn.Linear(d_model, d_model, bias=True)
            self.head_dim = d_model // n_heads
            self.num_heads = n_heads
        def forward(self, x):
            b, n, c = x.shape
            q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1,2)
            k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1,2)
            v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1,2)
            attn = torch.matmul(q, k.transpose(-2,-1)) / (self.head_dim ** 0.5)
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v).transpose(1,2).contiguous().view(b, n, c)
            return self.out_proj(out)

    class CLIPMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(d_model, d_ffn)
            self.fc2 = torch.nn.Linear(d_ffn, d_model)
            self.activation = QuickGELU()
        def forward(self, x):
            return self.fc2(self.activation(self.fc1(x)))

    class CLIPLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = CLIPAttention()
            self.layer_norm1 = torch.nn.LayerNorm(d_model)
            self.mlp = CLIPMLP()
            self.layer_norm2 = torch.nn.LayerNorm(d_model)
        def forward(self, x):
            x = x + self.self_attn(self.layer_norm1(x))
            x = x + self.mlp(self.layer_norm2(x))
            return x

    class CLIPEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([CLIPLayer() for _ in range(n_layers)])
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class CLIPTextModel_(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.ModuleDict({
                'token_embedding': torch.nn.Embedding(49408, d_model),
                'position_embedding': torch.nn.Embedding(77, d_model),
            })
            self.encoder = CLIPEncoder()
            self.final_layer_norm = torch.nn.LayerNorm(d_model)
        def forward(self, input_ids):
            x = self.embeddings['token_embedding'](input_ids)
            pos_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            x = x + self.embeddings['position_embedding'](pos_ids)
            x = self.encoder(x)
            x = self.final_layer_norm(x)
            return x

    model = CLIPTextModel_()
    model.eval()

    # Remap weights from CLIP format
    remap = {}
    for k, v in sd.items():
        if k.startswith('text_model.'):
            new_k = k[len('text_model.'):]
            remap[new_k] = v
        # Skip vision_model.* and logit_scale

    missing, unexpected = model.load_state_dict(remap, strict=False)
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print(f"  Sample missing: {missing[:5]}")
    if unexpected:
        print(f"  Sample unexpected: {unexpected[:5]}")

    if len(missing) == 0:
        # TorchScript trace
        example = torch.randint(0, 49408, (1, 77))
        traced = torch.jit.trace(model, example)
        torch.jit.save(traced, out_path)
        print(f"Saved to {out_path}")
        return True
    else:
        print(f"FAILED: keys mismatch")
        return False

if __name__ == "__main__":
    convert("CLIP-L", "/data/models/image/clip_l.safetensors", "/data/models/image/clip_l_jit.pt",
            d_model=768, n_layers=12, n_heads=12, d_ffn=3072)
    convert("CLIP-G", "/data/models/image/clip_g.safetensors", "/data/models/image/clip_g_jit.pt",
            d_model=1280, n_layers=32, n_heads=20, d_ffn=5120)
