# torch_helper.static.py — libtorch_std_helper.so FFI
#
# 用于 sd.cpp C API 无法覆盖的权重级操作（模型合并 / CLIP 合并 / 权重导出）。
# 共享库由 comfycli_ffi.scm 以 guard 方式加载；缺失时相关节点不可用。

extern fn torch_std_safetensors_load(path: str) -> ptr from "torch_helper"
extern fn torch_std_safetensors_count(d: ptr) -> int from "torch_helper"
extern fn torch_std_safetensors_save(d: ptr, path: str) -> int from "torch_helper"
extern fn torch_std_safetensors_merge(a: ptr, b: ptr, mode: int, prefixes_csv: str, ratios_csv: str, default_ratio: float, strip_prefix: str) -> ptr from "torch_helper"
extern fn torch_std_safetensors_free(d: ptr) -> None from "torch_helper"
extern fn torch_std_copy_file(src: str, dst: str) -> int from "torch_helper"
extern fn torch_std_sdxl_generate_areas_paths(model_path: str, clip_l_jit: str, clip_g_jit: str, vae_jit: str, vocab_path: str, merges_path: str, prompt: str, negative_prompt: str, width: int, height: int, steps: int, cfg: float, scheduler: str, seed: int, area_prompts: str, area_rects_csv: str, area_strengths_csv: str, output_path: str) -> int from "torch_helper"
