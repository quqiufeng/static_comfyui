DEVICE_CPU: int = -1
DEVICE_CUDA_0: int = 0


def get_torch_device() -> int:
    return DEVICE_CUDA_0


def get_free_memory() -> int:
    return torch.cuda_get_free_memory()


def soft_empty_cache():
    torch.cuda_soft_empty_cache()


def load_model_to_device(sd_handle: ptr, device: int):
    return torch.cuda_load_model(device, sd_handle)


def unload_model_from_device(t):
    torch.cuda_unload_model(t)


def unet_offload_device() -> int:
    return DEVICE_CPU


def main():
    pass
