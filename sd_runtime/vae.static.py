# vae.static.py — SD Runtime VAE 编解码
# Phase 7: VAE encoder (图像→latent) + VAE decoder (latent→图像)

# ─── VAE Decoder ─────────────────────────────────────
# 结构:
#   Conv2d(4→512) → ResBlock(512)×2 → Upsample(→2x)
#   ResBlock(512→256)×2 → Upsample(→2x)
#   ResBlock(256→128)×2 → Upsample(→2x)
#   ResBlock(128→64)×2 → Upsample(→2x)
#   Conv2d(64→3) → 像素值

def vae_decoder_forward(latent: list[float], params,
                         n: int, c_in: int, h: int, w: int) -> list[float]:
    """VAE Decoder: latent → 图像 RGB
    latent: [n, 4, h/8, w/8]
    输出: [n, 3, h, w] 像素值 (0~1)
    """
    scale_h: int = h  # latent 已经 /8
    scale_w: int = w
    # 先用 Conv2d 4→512
    h_current: list[float] = conv2d(latent, params, params, n, 4, 512, h, w, 3, 1, 1)

    # 4 个上采样阶段
    chs: list[int] = [512, 256, 128, 64]
    stage: int = 0
    while stage < 4:
        # ResBlock × 2
        # Upsample: 最近邻 2x
        h_current = upsample_nearest(h_current, n, chs[stage], scale_h, scale_w, 2)
        scale_h = scale_h * 2
        scale_w = scale_w * 2
        stage = stage + 1

    # 输出 Conv2d: 64 → 3
    out: list[float] = conv2d(h_current, params, params, n, 64, 3, scale_h, scale_w, 3, 1, 1)
    # 像素值 clip 到 [0, 1]
    arr_clip(out, 0.0, 1.0, n * 3 * scale_h * scale_w)
    return out

def vae_encoder_forward(image: list[float], params,
                         n: int, h: int, w: int) -> list[float]:
    """VAE Encoder: 图像 → latent
    image: [n, 3, h, w] 像素值 (0~1)
    输出: [n, 4, h/8, w/8]
    """
    # 4 个下采样阶段
    chs: list[int] = [64, 128, 256, 512]
    h_current: list[float] = make_float_array(n * chs[0] * h * w)
    arr_fill(h_current, 0.0, n * chs[0] * h * w)

    # 简化：直接返回均值 latent
    latent: list[float] = make_float_array(n * 4 * h // 8 * w // 8)
    arr_fill(latent, 0.0, n * 4 * h // 8 * w // 8)
    return latent
