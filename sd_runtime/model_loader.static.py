# model_loader.static.py — SD Runtime 模型权重加载
# Phase 8: Safetensors + GGUF 权重解析
#
# Safetensors 格式:
#   8 bytes: header_size (uint64, little-endian)
#   header_size bytes: JSON header
#   rest: binary tensor data

extern fn fopen(path: str, mode: str) -> ptr from "libc"
extern fn fread(buf: ptr, size: int, count: int, fp: ptr) -> int from "libc"
extern fn fclose(fp: ptr) -> int from "libc"
extern fn str_split(s: str, sep: str) -> list[str] from "prelude"
extern fn str_trim(s: str) -> str from "prelude"

def read_file_bytes(path: str) -> list[float]:
    """读取文件全部字节到 float 数组（按 8 字节双精度对齐）"""
    fp = fopen(path, "rb")
    if fp == 0: return make_float_array(0)
    # 获取文件大小
    fseek_res: int = 0
    # 简单方法：分块读取
    buf: list[float] = make_float_array(1024 * 1024)  # 8MB 缓冲区
    total: int = 0
    while True:
        n: int = fread(buf, 1, 8 * 1024 * 1024, fp)
        if n <= 0: break
        total = total + n
    fclose(fp)
    return buf

def parse_safetensors_header(path: str) -> list[float]:
    """解析 Safetensors 文件头，返回所有权重的字节偏移和大小"""
    fp = fopen(path, "rb")
    if fp == 0: return make_float_array(0)

    # 读 8 字节 header size
    header_len_bytes: list[float] = make_float_array(1)
    fread(header_len_bytes, 8, 1, fp)

    # Safetensors: header_len 是 uint64，但我们用 float 读
    # 实际需要按字节读 -> 简化：直接用 python 导出的 JSON 配置
    fclose(fp)
    return make_float_array(0)

def load_safetensors_tensor(path: str, offset: int, n: int) -> list[float]:
    """从 Safetensors 文件指定偏移读取 n 个 float"""
    fp = fopen(path, "rb")
    if fp == 0: return make_float_array(0)
    # fseek
    buf: list[float] = make_float_array(n)
    fread(buf, 8, n, fp)
    fclose(fp)
    return buf

def parse_sd15_weight_names() -> list[str]:
    """返回 SD1.5 所有权重名称列表（按 forward 顺序）"""
    # 由 export script 生成，这里只做文档占位
    return make_list()

# 权重字典：名称 → (文件路径, 偏移, 大小)
# 由 export_sd_weights.py 从 Safetensors 提取后生成

def load_sd_model(weight_dir: str, weights_json: str):
    """加载 SD 模型权重
    weights_json: 权重索引文件 [{name, offset, size}, ...]
    """
    # 先读索引
    fp = fopen(weights_json, "rb")
    if fp == 0: return
    fclose(fp)
    # 解析 JSON 后逐个加载
