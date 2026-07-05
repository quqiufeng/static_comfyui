def load_lora(lora_path: str):
    return torch.safetensors_load(lora_path)


def apply_lora(unet_handle, lora_handle, strength: float):
    return torch.lora_apply(unet_handle, lora_handle, strength)


def merge_lora_into(model_handle, lora_handle) -> int:
    return torch.lora_merge_into(model_handle, lora_handle)


def main():
    pass
