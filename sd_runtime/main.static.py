# main.static.py — SD Runtime 主入口

def load_all_weights(weight_dir: str):
    print("Weights dir: ")
    print(weight_dir)
    return 0

def dummy_unet(x, t, context, n, c, h, w):
    out: list[float] = make_float_array(n * c * h * w)
    i: int = 0
    while i < n * c * h * w:
        float_array_set(out, i, 0.0)
        i = i + 1
    return out

def generate(prompt: str, steps: int, cfg: float, width: int, height: int):
    print("Prompt: "); print(prompt)
    weights = load_all_weights("/stock/sd_weights")
    n: int = 1; c: int = 4; h: int = height // 8; w: int = width // 8
    total: int = n * c * h * w
    latent: list[float] = make_float_array(total)
    i: int = 0
    while i < total:
        float_array_set(latent, i, 0.1)
        i = i + 1
    context: list[float] = make_float_array(n * 77 * 768)
    arr_fill(context, 0.0, n * 77 * 768)
    print("Sampling...")

    # 占位采样：直接用 dummy 预测
    print("Sampling...")
    print("total="); print(total)
    x: list[float] = make_float_array(total)
    arr_copy(x, latent, total)
    print("making betas...")
    betas: list[float] = make_betas(steps, 0.00085, 0.012)
    print("making alphas...")
    alpha_bars: list[float] = make_alphas(betas, steps)
    print("alpha_bars[0]="); print(float_array_ref(alpha_bars, 0))
    t_step: int = steps - 1
    while t_step >= 0:
        ab_t: float = float_array_ref(alpha_bars, t_step)
        ab_prev: float = float_array_ref(alpha_bars, t_step - 1) if t_step > 0 else 1.0
        ts: list[float] = make_float_array(n)
        arr_fill(ts, t_step + 0.0, n)
        # dummy UNet: 返回全零
        pred: list[float] = make_float_array(total)
        arr_fill(pred, 0.0, total)
        print("ddim_step t="); print(t_step)
        print("ab_t="); print(ab_t)
        print("ab_prev="); print(ab_prev)
        x = ddim_step(x, pred, ab_t, ab_prev, total)
        print("done t="); print(t_step)
        t_step = t_step - 1

    print("Done")
    return x

def main():
    print("=== static_comfyui ===")
    result = generate("a cat", 20, 7.5, 512, 512)
    print("Latent sum: "); print(arr_sum(result, 4 * 64 * 64))
    exit_program(0)
