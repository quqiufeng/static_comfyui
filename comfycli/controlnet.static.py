def load_controlnet(controlnet_path: str):
    return torch.safetensors_load(controlnet_path)


def controlnet_forward(weights, input, timestep, text_emb, hint, hint_channels: int):
    return torch.controlnet_forward(weights, input, timestep, text_emb, hint, hint_channels)


def controlnet_apply(unet_features, control_features, strength: float):
    return torch.controlnet_apply(unet_features, control_features, strength)


def main():
    pass
