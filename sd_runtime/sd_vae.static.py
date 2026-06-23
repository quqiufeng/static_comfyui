# sd_runtime/sd_vae.static.py — VAE encode/decode 编排
# 1:1 对齐 comfyui_ref/comfy/sd.py (class VAE)
#
# 策略：使用 JIT 导出的 TorchScript VAE 模块（包含 encoder+quant_conv+regularizer
# 或 post_quant_conv+decoder），通过 C++ torch_std_vae_encode/decode 调用。
#
# VAE 模块导出方式（构建时预处理）：
#   import torch, comfy.sd, comfy.model_management
#   vae = comfy.sd.VAE(sd=state_dict)
#   vae.first_stage_model.eval()
#   vae_encode = torch.jit.trace(lambda x: vae.first_stage_model.encode(x)[0], example_input)
#   vae_decode = torch.jit.trace(lambda z: vae.first_stage_model.decode(z), example_latent)
#   vae_encode.save("vae_encode.pt")
#   vae_decode.save("vae_decode.pt")
#
# 用法:
#   vae = VAE("models/vae_encode.pt", "models/vae_decode.pt")
#   latent = vae.encode(image)    # (B, 3, H, W) → (B, 4, H/8, W/8)
#   image  = vae.decode(latent)   # (B, 4, H, W) → (B, 3, H*8, W*8)

from ops import *


class VAE:
    """Non-tiled VAE wrapper (standard resolution)."""
    
    encode_module: ptr
    decode_module: ptr
    
    def __init__(self, encode_pt_path: str, decode_pt_path: str):
        """Load JIT-compiled VAE encoder and decoder TorchScript modules."""
        self.encode_module = torch_std_jit_load(encode_pt_path)
        self.decode_module = torch_std_jit_load(decode_pt_path)
    
    def encode(self, image: ptr) -> ptr:
        """Encode image tensor to latent.
        
        image: (B, 3, H, W) float32, RGB, values in [-1, 1]
        Returns: (B, 4, H/8, W/8) float32 latent
        """
        return torch_std_vae_encode(self.encode_module, image)
    
    def decode(self, latent: ptr) -> ptr:
        """Decode latent tensor to image.
        
        latent: (B, 4, H, W) float32
        Returns: (B, 3, H*8, W*8) float32 image, values in [-1, 1]
        """
        return torch_std_vae_decode(self.decode_module, latent)
    
    def free(self) -> void:
        """Free JIT modules."""
        torch_std_jit_module_delete(self.encode_module)
        torch_std_jit_module_delete(self.decode_module)


class TiledVAE:
    """Tiled VAE for high-resolution images (>1024px)."""
    
    encode_module: ptr
    decode_module: ptr
    
    def __init__(self, encode_pt_path: str, decode_pt_path: str):
        """Load JIT-compiled VAE encoder and decoder modules."""
        self.encode_module = torch_std_jit_load(encode_pt_path)
        self.decode_module = torch_std_jit_load(decode_pt_path)
    
    def encode(self, image: ptr, tile_size: int = 512, overlap: int = 64) -> ptr:
        """Encode with tiling for memory efficiency.
        
        image: (B, 3, H, W) float32
        tile_size: pixel tile size (e.g. 512)
        overlap: pixel overlap between tiles (e.g. 64)
        Returns: (B, 4, H/8, W/8) float32 latent
        """
        return torch_std_vae_encode_tiled(self.encode_module, image,
                                           tile_size, overlap)
    
    def decode(self, latent: ptr, tile_size: int = 512, overlap: int = 64) -> ptr:
        """Decode with tiling for memory efficiency.
        
        latent: (B, 4, H, W) float32
        tile_size: pixel tile size (e.g. 512)
        overlap: pixel overlap between tiles (e.g. 64)
        Returns: (B, 3, H*8, W*8) float32 image
        """
        return torch_std_vae_decode_tiled(self.decode_module, latent,
                                           tile_size, overlap)
    
    def free(self) -> void:
        """Free JIT modules."""
        torch_std_jit_module_delete(self.encode_module)
        torch_std_jit_module_delete(self.decode_module)
