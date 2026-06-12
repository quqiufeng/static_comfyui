# main.static.py — SD Runtime 主入口
# Phase 10: txt2img 端到端集成

# 包含所有子模块
# 编译时会将 array_ops, nn_ops, unet_blocks, unet, samplers, clip, model_loader
# 与这个文件合并后编译

def load_all_weights(weight_dir: str) -> list[list[float]]:
    """加载所有权重到内存"""
    print("Loading SD model weights from "); print(weight_dir)
    # 简单验证：检查文件是否存在
    fp = fopen(weight_dir + "/index.json", "rb")
    if fp == 0:
        print("No weights found. Run export_sd_weights.py first.")
        return make_list()
    fclose(fp)
    print("Weights OK")
    return make_list()

def dummy_unet(x, t, context, n, c, h, w):
    """占位 UNet: 返回随机噪声"""
    out: list[float] = make_float_array(n * c * h * w)
    i: int = 0
    while i < n * c * h * w:
        float_array_set(out, i, 0.0)
        i = i + 1
    return out

def generate(prompt: str, steps: int, cfg: float, width: int, height: int):
    """txt2img 主函数"""
    print("Prompt: "); print(prompt)
    print("Steps: "); print(steps)
    print("CFG: "); print(cfg)

    # 1. 加载权重
    weights = load_all_weights("/stock/sd_weights")

    # 2. 编码文本
    n: int = 1
    context: list[float] = make_float_array(n * 77 * 768)
    arr_fill(context, 0.0, n * 77 * 768)

    # 3. 生成初始噪声 latent
    c: int = 4
    h: int = height // 8
    w: int = width // 8
    latent: list[float] = make_float_array(n * c * h * w)
    i: int = 0
    while i < n * c * h * w:
        float_array_set(latent, i, 0.1)  # 简化：固定噪声
        i = i + 1

    # 4. 采样
    print("Sampling...")
    result: list[float] = sample_ddim(
        dummy_unet, latent, context, context,
        n, c, h, w, steps, cfg)

    print("Generation complete")
    return result

def main():
    print("=== static_comfyui — SD Runtime ===")
    print()

    # 测试生成
    result = generate("a cat", 20, 7.5, 512, 512)
    print("Output latent: "); print(arr_sum(result, 4 * 64 * 64))

    print()
    print("=== Done ===")
    exit_program(0)
