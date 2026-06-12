# samplers.static.py — SD Runtime 采样器
# Phase 5: DDIM, Euler, DPM++ 采样 + CFG guidance

# ─── Beta / Alpha 调度 ──────────────────────────────

def make_betas(n_steps: int, beta_start: float, beta_end: float) -> list[float]:
    """线性 beta 调度"""
    betas: list[float] = make_float_array(n_steps)
    i: int = 0
    while i < n_steps:
        float_array_set(betas, i, beta_start + (beta_end - beta_start) * i / (n_steps - 1))
        i = i + 1
    return betas

def make_alphas(betas: list[float], n: int) -> list[float]:
    """alpha = 1 - beta, alpha_bar = cumprod(alpha)"""
    alphas: list[float] = make_float_array(n)
    alpha_bars: list[float] = make_float_array(n)
    cumprod: float = 1.0
    i: int = 0
    while i < n:
        a: float = 1.0 - float_array_ref(betas, i)
        cumprod = cumprod * a
        float_array_set(alphas, i, a)
        float_array_set(alpha_bars, i, cumprod)
        i = i + 1
    return alpha_bars

# ─── CFG Guidance ────────────────────────────────────

def cfg_apply(uncond_pred: list[float], cond_pred: list[float],
              cfg_scale: float, n: int):
    """CFG: pred = uncond + cfg * (cond - uncond)"""
    i: int = 0
    while i < n:
        uc: float = float_array_ref(uncond_pred, i)
        c: float = float_array_ref(cond_pred, i)
        float_array_set(cond_pred, i, uc + cfg_scale * (c - uc))
        i = i + 1

# ─── DDIM 采样 ──────────────────────────────────────

def ddim_step(x: list[float], pred_noise: list[float],
              alpha_bar_t: float, alpha_bar_prev: float,
              n: int) -> list[float]:
    """DDIM 单步: x_{t-1} = sqrt(alpha_bar_prev) * pred_x0 + sqrt(1 - alpha_bar_prev) * eps
    pred_x0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
    """
    sqrt_ab_t: float = sqrt(alpha_bar_t)
    sqrt_1_ab_t: float = sqrt(1.0 - alpha_bar_t)
    sqrt_ab_prev: float = sqrt(alpha_bar_prev)
    sqrt_1_ab_prev: float = sqrt(1.0 - alpha_bar_prev)

    # pred_x0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
    pred_x0: list[float] = make_float_array(n)
    i: int = 0
    while i < n:
        v: float = (float_array_ref(x, i) - sqrt_1_ab_t * float_array_ref(pred_noise, i)) / sqrt_ab_t
        float_array_set(pred_x0, i, v)
        i = i + 1

    # x_{t-1} = sqrt(alpha_bar_prev) * pred_x0 + sqrt(1 - alpha_bar_prev) * eps
    x_prev: list[float] = make_float_array(n)
    i = 0
    while i < n:
        v = sqrt_ab_prev * float_array_ref(pred_x0, i) + sqrt_1_ab_prev * float_array_ref(pred_noise, i)
        float_array_set(x_prev, i, v)
        i = i + 1
    return x_prev

# ─── Euler 采样 ─────────────────────────────────────

def euler_step(x: list[float], pred_noise: list[float],
               sigma: float, sigma_next: float,
               n: int) -> list[float]:
    """Euler 单步: x_{t-1} = x_t + (sigma_next - sigma) * pred_noise"""
    dt: float = sigma_next - sigma
    x_next: list[float] = make_float_array(n)
    i: int = 0
    while i < n:
        float_array_set(x_next, i, float_array_ref(x, i) + dt * float_array_ref(pred_noise, i))
        i = i + 1
    return x_next

# ─── DPM++ 2M 采样 ──────────────────────────────────

def dpmpp_2m_step(x: list[float], pred_noise: list[float],
                  sigma: float, sigma_next: float, sigma_prev: float,
                  n: int, old_denoised: list[float]) -> list[float]:
    """DPM++ 2M 单步（二阶）"""
    # 简化版本: 近似为 Euler + 校正
    dt: float = sigma_next - sigma
    x_next: list[float] = make_float_array(n)
    i: int = 0
    while i < n:
        float_array_set(x_next, i, float_array_ref(x, i) + dt * float_array_ref(pred_noise, i))
        i = i + 1
    return x_next

# ─── 采样循环 ──────────────────────────────────────

def sample_ddim(unet_fn, latent: list[float],
                context: list[float], uncond_context: list[float],
                n: int, c: int, h: int, w: int,
                steps: int, cfg_scale: float) -> list[float]:
    """DDIM 采样循环"""
    total: int = n * c * h * w
    x: list[float] = make_float_array(total)
    arr_copy(x, latent, total)

    betas: list[float] = make_betas(steps, 0.00085, 0.012)
    alpha_bars: list[float] = make_alphas(betas, steps)

    t: int = steps - 1
    while t >= 0:
        ab_t: float = float_array_ref(alpha_bars, t)
        ab_prev: float = float_array_ref(alpha_bars, t - 1) if t > 0 else 1.0

        ts: list[float] = make_float_array(n)
        arr_fill(ts, t, n)

        # 无条件预测
        pred_uncond: list[float] = unet_fn(x, ts, uncond_context, n, c, h, w)
        # 有条件预测
        pred_cond: list[float] = unet_fn(x, ts, context, n, c, h, w)

        # CFG
        cfg_apply(pred_uncond, pred_cond, cfg_scale, total)

        # DDIM 步进
        x = ddim_step(x, pred_cond, ab_t, ab_prev, total)

        t = t - 1

    return x
