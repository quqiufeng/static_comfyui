#!/data/venv/bin/python
"""PyTorch SDXL HiRes Fix — 替代 GGML img_hires，FreeU + 完整后处理"""
import os, sys, time, argparse, hashlib, math
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ComfyUI'))

import torch
import torch.nn.functional as F
from comfy.sd import load_checkpoint_guess_config
import comfy.sample

MODEL_DIR = os.environ.get('MODEL_DIR', '/data/models/image')

def parse_args():
    p = argparse.ArgumentParser(description='PyTorch SDXL HiRes Fix')
    p.add_argument('prompt', nargs='?', default='A beautiful landscape')
    p.add_argument('output', nargs='?', default='')
    p.add_argument('width', nargs='?', type=int, default=2560)
    p.add_argument('height', nargs='?', type=int, default=1440)
    p.add_argument('--checkpoint', default=os.environ.get('CHECKPOINT',
        f'{MODEL_DIR}/RealVisXL_V5.0_fp16.safetensors'))
    p.add_argument('--negative', default=os.environ.get('NEGATIVE_PROMPT',
        'blurry, low quality, worst quality, jpeg artifacts, noise, grain, '
        'soft focus, out of focus, hazy, unclear, bad anatomy, deformed, '
        'border artifacts, edge distortion, tiling artifacts, edge artifacts, '
        'frame distortion, warped edges, stretched proportions, asymmetrical face, '
        'off-center, cropped, out of frame, partial face, cut off, incomplete head, '
        'cropped head, watermark, text, logo, signature, cropped shoulders'))
    p.add_argument('--cfg', type=float, default=float(os.environ.get('CFG_SCALE', '5.0')))
    p.add_argument('--steps', type=int, default=int(os.environ.get('STEPS', '25')))
    p.add_argument('--hires-steps', type=int, default=int(os.environ.get('HIRES_STEPS', '45')))
    p.add_argument('--hires-strength', type=float, default=float(os.environ.get('HIRES_STRENGTH', '0.35')))
    p.add_argument('--sampler', default=os.environ.get('SAMPLING_METHOD', 'euler'))
    p.add_argument('--scheduler', default=os.environ.get('SCHEDULER', 'karras'))
    p.add_argument('--seed', type=int, default=int(os.environ.get('SEED', '0')) or None)
    return p.parse_args()

def compute_low_res(w, h):
    known = {(3840, 2160): (2560, 1440), (2560, 1440): (1920, 1080),
             (1920, 1080): (1536, 864), (1280, 720): (1024, 576)}
    if (w, h) in known:
        return known[(w, h)]
    lw, lh = w * 4 // 5 // 8 * 8, h * 4 // 5 // 8 * 8
    lw = max(lw, 512); lh = max(lh, 512)
    return lw, lh

def encode_prompt(clip, text):
    tokens = clip.tokenize(text)
    out, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return out.cuda().half(), pooled.cuda().float()

def apply_freeu_v2(model_patcher, b1, b2, s1, s2):
    model_channels = model_patcher.model.model_config.unet_config["model_channels"]
    scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}
    on_cpu_devices = {}

    def output_block_patch(h, hsp, transformer_options):
        scale = scale_dict.get(int(h.shape[1]), None)
        if scale is not None:
            hidden_mean = h.mean(1).unsqueeze(1)
            B = hidden_mean.shape[0]
            hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) \
                        / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
            h[:,:h.shape[1] // 2] = h[:,:h.shape[1] // 2] * ((scale[0] - 1) * hidden_mean + 1)
            if hsp.device not in on_cpu_devices:
                try:
                    x_freq = torch.fft.fftn(hsp.float(), dim=(-2, -1))
                    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
                    B, C, H, W = x_freq.shape
                    mask = torch.ones((B, C, H, W), device=hsp.device)
                    crow, ccol = H // 2, W // 2
                    t = 1
                    mask[..., crow - t:crow + t, ccol - t:ccol + t] = scale[1]
                    x_freq = x_freq * mask
                    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
                    hsp = torch.fft.ifftn(x_freq, dim=(-2, -1)).real.to(hsp.dtype)
                except:
                    on_cpu_devices[hsp.device] = True
                    hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)
            else:
                hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)
        return h, hsp

    m = model_patcher.clone()
    m.set_model_output_block_patch(output_block_patch)
    return m

def apply_clarity(img_t, amount):
    if amount <= 0: return img_t
    radius = max(1, int(amount * 50))
    k = radius * 2 + 1
    blurred = F.avg_pool2d(img_t, kernel_size=k, stride=1, padding=radius)
    return img_t + (img_t - blurred) * 0.5

def apply_sharpen(img_t, amount, radius):
    if amount <= 0: return img_t
    k = radius * 2 + 1
    blurred = F.avg_pool2d(img_t, k, 1, padding=radius)
    return img_t + (img_t - blurred) * amount

def apply_smart_sharpen(img_t, amount, radius):
    if amount <= 0: return img_t
    gray = img_t.mean(dim=1, keepdim=True)
    sobel_x = F.conv2d(gray, torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]],
        device=img_t.device, dtype=img_t.dtype), padding=1)
    sobel_y = F.conv2d(gray, torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]],
        device=img_t.device, dtype=img_t.dtype), padding=1)
    edge_mag = (sobel_x ** 2 + sobel_y ** 2).sqrt()
    eps = 1e-8
    weight = (edge_mag - edge_mag.min()) / (edge_mag.max() - edge_mag.min() + eps)
    k = radius * 2 + 1
    blurred = F.avg_pool2d(img_t, k, 1, padding=radius)
    return img_t + (img_t - blurred) * weight * amount

def apply_edge_sharpen(img_t, amount, radius, threshold):
    if amount <= 0: return img_t
    gray = img_t.mean(dim=1, keepdim=True)
    sobel_x = F.conv2d(gray, torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]],
        device=img_t.device, dtype=img_t.dtype), padding=1)
    sobel_y = F.conv2d(gray, torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]],
        device=img_t.device, dtype=img_t.dtype), padding=1)
    edge = (sobel_x ** 2 + sobel_y ** 2).sqrt()
    mask = (edge > threshold).float()
    k = radius * 2 + 1
    blurred = F.avg_pool2d(img_t, k, 1, padding=radius)
    return img_t + (img_t - blurred) * mask * amount

def postprocess(img_t):
    img_t = img_t.float() / 255.0
    img_t = img_t.unsqueeze(0).permute(0, 3, 1, 2)
    img_t = apply_clarity(img_t, 0.2)
    img_t = apply_sharpen(img_t, 0.3, 1)
    img_t = apply_smart_sharpen(img_t, 0.5, 2)
    img_t = apply_edge_sharpen(img_t, 1.5, 2, 0.3)
    img_t = img_t.permute(0, 2, 3, 1).squeeze(0)
    return (img_t.clamp(0, 1) * 255).byte()

def main():
    args = parse_args()
    device = 'cuda'
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), 'big')
    torch.manual_seed(seed)

    target_latent_w = args.width // 8
    target_latent_h = args.height // 8
    low_w, low_h = compute_low_res(args.width, args.height)
    low_latent_w = low_w // 8
    low_latent_h = low_h // 8

    quality_prefix = ("masterpiece, best quality, ultra-detailed, sharp focus, 8k uhd, "
        "photorealistic, highly detailed, crisp, clear, centered composition, "
        "professional portrait, medium shot, realistic skin texture, soft lighting")
    prompt = f"{quality_prefix}, {args.prompt}" if "masterpiece" not in args.prompt else args.prompt

    print(f"Loading checkpoint: {args.checkpoint}")
    model_patcher, clip, vae, _ = load_checkpoint_guess_config(args.checkpoint)
    model_patcher.model.to(device)
    print(f"Model loaded on {device}")

    freeu_b1 = float(os.environ.get('FREEU_B1', '1.1'))
    freeu_b2 = float(os.environ.get('FREEU_B2', '1.4'))
    freeu_s1 = float(os.environ.get('FREEU_S1', '0.9'))
    freeu_s2 = float(os.environ.get('FREEU_S2', '0.2'))
    model_patcher = apply_freeu_v2(model_patcher, b1=freeu_b1, b2=freeu_b2, s1=freeu_s1, s2=freeu_s2)
    print(f"FreeU_V2 applied (b1={freeu_b1}, b2={freeu_b2}, s1={freeu_s1}, s2={freeu_s2})")

    pos_out, pos_pooled = encode_prompt(clip, prompt)
    neg_out, neg_pooled = encode_prompt(clip, args.negative)
    positive = [(pos_out, {'pooled_output': pos_pooled})]
    negative = [(neg_out, {'pooled_output': neg_pooled})]

    print(f"\n{'='*40}")
    print(f"Target: {args.width}x{args.height}")
    print(f"Low-res: {low_w}x{low_h} -> {args.width}x{args.height}")
    print(f"Steps: {args.steps} -> {args.hires_steps} (HiRes)")
    print(f"CFG: {args.cfg} | Sampler: {args.sampler}+{args.scheduler}")
    print(f"Seed: {seed}")
    print(f"{'='*40}\n")

    latent = torch.zeros(1, 4, low_latent_h, low_latent_w, dtype=torch.float16, device=device)
    noise = torch.randn(1, 4, low_latent_h, low_latent_w, dtype=torch.float16, device=device)

    t0 = time.time()
    samples = comfy.sample.sample(model_patcher, noise, args.steps, args.cfg,
                                   args.sampler, args.scheduler,
                                   positive, negative, latent,
                                   denoise=1.0, seed=seed)
    t1 = time.time()
    print(f"Low-res pass: {t1-t0:.1f}s")

    samples = F.interpolate(samples, size=(target_latent_h, target_latent_w),
                            mode='bilinear', align_corners=False)
    noise_hires = torch.randn(1, 4, target_latent_h, target_latent_w,
                              dtype=torch.float16, device=device)

    samples = comfy.sample.sample(model_patcher, noise_hires, args.hires_steps, args.cfg,
                                   args.sampler, args.scheduler,
                                   positive, negative, samples,
                                   denoise=args.hires_strength, seed=seed)
    t2 = time.time()
    print(f"HiRes pass: {t2-t1:.1f}s")

    decoded = vae.decode(samples)
    t3 = time.time()
    print(f"VAE decode: {t3-t2:.1f}s")

    img = decoded[0].cpu().float()
    if img.dim() == 3 and img.size(0) <= 4:
        img = img.permute(1, 2, 0)
    img = (img * 255).clamp(0, 255).byte()

    print("Post-processing...")
    img = postprocess(img)

    out_path = args.output
    if not out_path:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        md5 = hashlib.md5(prompt.encode()).hexdigest()[:8]
        out_path = os.path.expanduser(f'~/{ts}_{md5}.png')
    out_path = os.path.expanduser(out_path)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    from PIL import Image
    Image.fromarray(img.numpy()).save(out_path)
    t4 = time.time()
    total = t4 - t0

    print(f"\n{'='*40}")
    print(f"Saved: {out_path}")
    print(f"Size: {args.width}x{args.height}")
    print(f"Time: {total:.0f}s ({total/60:.1f}m)")
    print(f"Seed: {seed}")
    print(f"{'='*40}")

if __name__ == '__main__':
    main()
