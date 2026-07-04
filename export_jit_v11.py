"""Export JIT models - fixed VAE decode output."""
import sys, os, gc, torch, torch.nn as nn

device = torch.device("cuda")
print(f"Device: {device}")

output_dir = "/tmp/sd_jit"
os.makedirs(output_dir, exist_ok=True)

# === VAE ===
print("\n=== VAE ===")
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32)
vae.to(device).eval()

class VAEDecodeWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, z):
        return self.m.decode(z).sample  # extract tensor from DecoderOutput

class VAEEncodeWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        return self.m.encode(x).latent_dist.sample()

dec = VAEDecodeWrapper(vae).to(device).eval()
dummy_latent = torch.randn(1, 4, 64, 64, device=device)
with torch.no_grad():
    traced_dec = torch.jit.trace(dec, (dummy_latent,))
traced_dec.save(os.path.join(output_dir, "vae_decode.pt"))
print("  VAE Decode done")

enc = VAEEncodeWrapper(vae).to(device).eval()
dummy_img = torch.randn(1, 3, 512, 512, device=device)
with torch.no_grad():
    traced_enc = torch.jit.trace(enc, (dummy_img,))
traced_enc.save(os.path.join(output_dir, "vae_encode.pt"))
print("  VAE Encode done")

del vae, dec, enc, traced_dec, traced_enc
gc.collect(); torch.cuda.empty_cache()

# === CLIP ===
from transformers import CLIPTextModel
for name, model_id in [("clip_l", "openai/clip-vit-large-patch14"),
                        ("clip_g", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")]:
    print(f"\n=== {name} ===")
    model = CLIPTextModel.from_pretrained(model_id).to(device).eval()
    class CLIPWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, t):
            return self.m(t)[0]
    wrapper = CLIPWrapper(model).to(device).eval()
    dummy = torch.randint(0, 49407, (1, 77), device=device)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy,))
    traced.save(os.path.join(output_dir, f"{name}.pt"))
    sz = os.path.getsize(os.path.join(output_dir, f"{name}.pt")) / 1024 / 1024
    print(f"  {name} done ({sz:.1f} MB)")
    del model, wrapper, traced
    gc.collect(); torch.cuda.empty_cache()

print(f"\nAll exported!")
for f in os.listdir(output_dir):
    print(f"  {f}: {os.path.getsize(os.path.join(output_dir,f))/1024/1024:.1f} MB")
