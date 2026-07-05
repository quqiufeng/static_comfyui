# Model type constants (match comfy_types ModelType)
MODEL_TYPE_EPS: int = 1
MODEL_TYPE_V_PREDICTION: int = 2
MODEL_TYPE_FLOW: int = 6
MODEL_TYPE_FLUX: int = 8


@dataclass
class BaseModel:
    model_type: int
    latent_format_name: str
    sampling_type: int
    unet_prefix: str


def make_base_model(model_type: int, latent_format_name: str, sampling_type: int, unet_prefix: str) -> BaseModel:
    return BaseModel(model_type, latent_format_name, sampling_type, unet_prefix)


def base_model_model_type(m: BaseModel) -> int:
    return m.model_type


def base_model_latent_format_name(m: BaseModel) -> str:
    return m.latent_format_name


def base_model_sampling_type(m: BaseModel) -> int:
    return m.sampling_type


def base_model_unet_prefix(m: BaseModel) -> str:
    return m.unet_prefix


def main():
    pass
