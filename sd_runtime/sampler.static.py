# sampler.static.py — ComfyUI Euler sampler (Karras) for SDXL, GPU ptr version

extern fn st_add_tensor(a: ptr, b: ptr) -> ptr from "staticpy_torch"
extern fn st_sub_tensor(a: ptr, b: ptr) -> ptr from "staticpy_torch"
extern fn st_mul_scalar_tensor(t: ptr, s: float) -> ptr from "staticpy_torch"
extern fn st_clone(t: ptr) -> ptr from "staticpy_torch"
extern fn st_tensor_free(t: ptr) -> void from "staticpy_torch"

extern fn st_cuda_torch_memory_allocated() -> int from "staticpy_torch"
extern fn st_cuda_empty_cache() -> void from "staticpy_torch"
extern fn st_cuda_empty_cache() -> void from "staticpy_torch"

def karras_sigmas(n: int, sigma_min: float, sigma_max: float, rho: float) -> list[float]:
    """Karras noise schedule, length n+1 with trailing zero."""
    sigmas: list[float] = make_float_array(n + 1)
    i: int = 0
    min_inv: float = exp(log(sigma_min) / rho)
    max_inv: float = exp(log(sigma_max) / rho)
    while i < n:
        ramp: float = float(i) / float(n - 1)
        s: float = exp(log(max_inv) + ramp * (log(min_inv) - log(max_inv)))
        s = exp(log(s) * rho)
        float_array_set(sigmas, i, s)
        i = i + 1
    float_array_set(sigmas, n, 0.0)
    return sigmas

def sigma_to_timestep(sigma: float, sigmas: list[float], n: int) -> float:
    """Find discrete timestep index whose sigma is closest in log space."""
    best: int = 0
    best_diff: float = 1000000.0
    log_s: float = log(sigma)
    i: int = 0
    while i < n:
        s: float = float_array_ref(sigmas, i)
        d: float = log_s - log(s)
        if d < 0.0:
            d = -d
        if d < best_diff:
            best_diff = d
            best = i
        i = i + 1
    return float(best)

def make_timestep_array(sigmas: list[float], n: int, discrete_sigmas: list[float], nd: int) -> list[float]:
    """Convert each Karras sigma to a discrete timestep (0..999)."""
    ts: list[float] = make_float_array(n)
    i: int = 0
    while i < n:
        t: float = sigma_to_timestep(float_array_ref(sigmas, i), discrete_sigmas, nd)
        float_array_set(ts, i, t)
        i = i + 1
    return ts

# UNet wrapper: returns eps prediction, then compute denoised = x - sigma*eps,
# derivative d = (x - denoised)/sigma = eps.
def unet_eps(weights: ptr, x: ptr, timestep: float,
             context: ptr, y: ptr, n: int, c: int, h: int, w: int) -> ptr:
    ts: list[float] = make_float_array(1)
    float_array_set(ts, 0, timestep)
    return unet_forward(x, ts, context, y, weights, n, h, w)

def sample_euler(weights: ptr,
                 x: ptr,
                 sigmas: list[float],
                 timesteps: list[float],
                 context: ptr,
                 uncond_context: ptr,
                 y: ptr,
                 uncond_y: ptr,
                 n: int, c: int, h: int, w: int,
                 steps: int, cfg_scale: float) -> ptr:
    """Euler sampler with Karras sigmas and CFG."""
    i: int = 0
    while i < steps:
        sigma: float = float_array_ref(sigmas, i)
        sigma_next: float = float_array_ref(sigmas, i + 1)
        t: float = float_array_ref(timesteps, i)

        # conditional eps
        eps_c: ptr = unet_eps(weights, x, t, context, y, n, c, h, w)
        if cfg_scale != 1.0:
            # unconditional eps
            eps_u: ptr = unet_eps(weights, x, t, uncond_context, uncond_y, n, c, h, w)
            # eps = eps_u + cfg * (eps_c - eps_u)
            diff: ptr = st_sub_tensor(eps_c, eps_u)
            st_tensor_free(eps_c)
            scaled: ptr = st_mul_scalar_tensor(diff, cfg_scale)
            st_tensor_free(diff)
            eps_c = st_add_tensor(eps_u, scaled)
            st_tensor_free(eps_u); st_tensor_free(scaled)

        dt: float = sigma_next - sigma
        step: ptr = st_mul_scalar_tensor(eps_c, dt)
        st_tensor_free(eps_c)
        x_next: ptr = st_add_tensor(x, step)
        st_tensor_free(step)
        if i > 0:
            st_tensor_free(x)
        x = x_next
        print("step "); print(i); print(" torch_mem="); print(st_cuda_torch_memory_allocated())
        st_cuda_empty_cache()
        i = i + 1
    return x
