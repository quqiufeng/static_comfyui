# Latent format: [scale_factor, latent_channels, name]

def make_latent_format(scale: float, channels: int, name: str) -> list:
    return py_list(scale, channels, name)

LF_SD15 = make_latent_format(0.18215, 4, "SD15")
LF_SDXL = make_latent_format(0.13025, 4, "SDXL")
LF_SD3 = make_latent_format(1.0, 16, "SD3")
LF_FLUX = make_latent_format(0.3611, 16, "Flux")
LF_FLUX2 = make_latent_format(0.3611, 16, "Flux2")
LF_SC_C = make_latent_format(1.0, 16, "StableCascade")
LF_SVD = make_latent_format(1.0, 4, "SVD")
LF_HUNYUAN_VIDEO = make_latent_format(1.0, 16, "HunyuanVideo")

def latent_format_scale(lf: list) -> float:
    return py_list_ref(lf, 0)

def latent_format_channels(lf: list) -> int:
    return py_list_ref(lf, 1)

def latent_format_name(lf: list) -> str:
    return py_list_ref(lf, 2)

def latent_process_in(latent: float, lf: list) -> float:
    return latent * latent_format_scale(lf)

def latent_process_out(latent: float, lf: list) -> float:
    return latent / latent_format_scale(lf)


def main():
    pass
