#!/usr/bin/env python3
"""Convert CLIP safetensors to TorchScript using ScriptModule (device-agnostic)."""
import torch
from safetensors.torch import load_file

def convert(name, in_path, out_path, d_model=768, n_layers=12, n_heads=12, d_ffn=3072):
    sd = load_file(in_path)
    print(f"\n=== {name} === Loaded {len(sd)} keys")

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
            self.num_heads = n_heads
            self.head_dim = d_model // n_heads
        def forward(self, x):
            b, n, c = x.shape
            q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1,2)
            k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1,2)
            v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1,2)
            attn = (q @ k.transpose(-2,-1)) / (self.head_dim ** 0.5)
            attn = torch.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1,2).contiguous().view(b, n, c)
            return self.out_proj(out)

    class CLIPMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(d_model, d_ffn)
            self.fc2 = torch.nn.Linear(d_ffn, d_model)
        def forward(self, x):
            return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

    class CLIPEncoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = CLIPAttention()
            self.layer_norm1 = torch.nn.LayerNorm(d_model)
            self.mlp = CLIPMLP()
            self.layer_norm2 = torch.nn.LayerNorm(d_model)
        def forward(self, x):
            x = x + self.self_attn(self.layer_norm1(x))
            return x + self.mlp(self.layer_norm2(x))

    class CLIPEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([CLIPEncoderLayer() for _ in range(n_layers)])
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class CLIPTextModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding = torch.nn.Embedding(49408, d_model)
            self.position_embedding = torch.nn.Embedding(77, d_model)
            self.encoder = CLIPEncoder()
            self.final_layer_norm = torch.nn.LayerNorm(d_model)
        @torch.jit.export
        def forward(self, input_ids):
            x = self.token_embedding(input_ids)
            pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            x = x + self.position_embedding(pos)
            x = self.encoder(x)
            return self.final_layer_norm(x)

    model = CLIPTextModel()
    model.eval()

    remap = {}
    for k, v in sd.items():
        if k.startswith('text_model.'):
            new_k = k[len('text_model.'):]
            # embeddings.token_embedding.weight -> token_embedding.weight
            if new_k.startswith('embeddings.'):
                new_k = new_k[len('embeddings.'):]
            remap[new_k] = v

    missing, unexpected = model.load_state_dict(remap, strict=False)
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print(f"  Missing: {missing[:5]}")
        return False

    # Use script (not trace) for device agnosticism
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, out_path)
    print(f"Saved to {out_path}")
    return True

if __name__ == "__main__":
    convert("CLIP-L", "/data/models/image/clip_l.safetensors", "/data/models/image/clip_l_jit.pt",
            d_model=768, n_layers=12, n_heads=12, d_ffn=3072)
    convert("CLIP-G", "/data/models/image/clip_g.safetensors", "/data/models/image/clip_g_jit.pt",
            d_model=1280, n_layers=32, n_heads=20, d_ffn=5120)
