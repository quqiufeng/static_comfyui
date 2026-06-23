# sd_runtime/ops.static.py — 张量基元层
# 1:1 对齐 comfyui_ref/comfy/ops.py
# 所有 extern fn 映射到 /opt/ReScheme/libtorch_std_helper.h 的 extern "C" API
#
# 使用方式:
#   from ops import torch_std_linear, torch_std_conv2d, ...

# ==============================================================================
# Tensor 创建
# ==============================================================================

extern fn torch_std_tensor_from_blob(data: ptr, shape: ptr, ndim: int, dtype: int) -> ptr from "torch_std_helper"
extern fn torch_std_zeros(shape: ptr, ndim: int, dtype: int) -> ptr from "torch_std_helper"
extern fn torch_std_ones(shape: ptr, ndim: int, dtype: int) -> ptr from "torch_std_helper"
extern fn torch_std_empty(shape: ptr, ndim: int, dtype: int) -> ptr from "torch_std_helper"
extern fn torch_std_full(shape: ptr, ndim: int, value: float, dtype: int) -> ptr from "torch_std_helper"
extern fn torch_std_clone(t: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_to_dtype(t: ptr, dtype: int) -> ptr from "torch_std_helper"
extern fn torch_std_delete_tensor(t: ptr) -> void from "torch_std_helper"
extern fn torch_std_arange(start: int, end: int, step: int, dtype: int) -> ptr from "torch_std_helper"

# Tensor 切片与元数据
extern fn torch_std_narrow(a: ptr, dim: int, start: int, length: int) -> ptr from "torch_std_helper"
extern fn torch_std_size(a: ptr, dim: int) -> int from "torch_std_helper"
 
# 数据类型常量
DTYPE_FLOAT32: int = 0
DTYPE_FLOAT64: int = 1
DTYPE_INT32:   int = 2
DTYPE_INT64:   int = 3

# ==============================================================================
# Tensor 元数据
# ==============================================================================

extern fn torch_std_numel(t: ptr) -> int from "torch_std_helper"
extern fn torch_std_ndim(t: ptr) -> int from "torch_std_helper"
extern fn torch_std_shape(t: ptr, out: ptr) -> void from "torch_std_helper"

# ==============================================================================
# Tensor 设备
# ==============================================================================

extern fn torch_std_cuda_is_available() -> int from "torch_std_helper"
extern fn torch_std_to_cuda(t: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_to_cpu(t: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_is_cuda(t: ptr) -> int from "torch_std_helper"
extern fn torch_std_manual_seed(seed: int) -> void from "torch_std_helper"

# ==============================================================================
# 数学运算 （逐元素）
# ==============================================================================

extern fn torch_std_add(a: ptr, b: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_sub(a: ptr, b: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_mul(a: ptr, b: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_div(a: ptr, b: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_pow(a: ptr, exp: float) -> ptr from "torch_std_helper"
extern fn torch_std_exp(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_log(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_sqrt(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_neg(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_abs(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_cos(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_sin(a: ptr) -> ptr from "torch_std_helper"
 
# ==============================================================================
# 激活函数
# ==============================================================================

extern fn torch_std_relu(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_leaky_relu(a: ptr, negative_slope: float) -> ptr from "torch_std_helper"
extern fn torch_std_sigmoid(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_tanh(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_softmax(a: ptr, dim: int) -> ptr from "torch_std_helper"
extern fn torch_std_log_softmax(a: ptr, dim: int) -> ptr from "torch_std_helper"

# SiLU / GELU 需要手动实现或通过已有 ops 组合
# libtorch_std_helper 没有直接提供 silu/gelu
# 用 sigmoid 组合实现: silu(x) = x * sigmoid(x)

def torch_std_silu(a: ptr) -> ptr:
    """SiLU / Swish: x * sigmoid(x)"""
    s: ptr = torch_std_sigmoid(a)
    return torch_std_mul(a, s)

def torch_std_gelu(a: ptr) -> ptr:
    """GELU 近似: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    sqrt_2_over_pi: float = 0.7978845608028654  # sqrt(2/pi)
    a_cubed: ptr = torch_std_pow(a, 3.0)
    inner: ptr = torch_std_add(a, torch_std_mul_scalar(a_cubed, 0.044715))
    inner = torch_std_mul_scalar(inner, sqrt_2_over_pi)
    tanh_inner: ptr = torch_std_tanh(inner)
    one_plus_tanh: ptr = torch_std_add_scalar(tanh_inner, 1.0)
    result: ptr = torch_std_mul(torch_std_mul_scalar(a, 0.5), one_plus_tanh)
    return result

# ==============================================================================
# 归约
# ==============================================================================

extern fn torch_std_sum(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_sum_dim(a: ptr, dim: int, keepdim: int) -> ptr from "torch_std_helper"
extern fn torch_std_mean(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_mean_dim(a: ptr, dim: int, keepdim: int) -> ptr from "torch_std_helper"
extern fn torch_std_max(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_max_dim(a: ptr, dim: int, keepdim: int) -> ptr from "torch_std_helper"
extern fn torch_std_min(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_min_dim(a: ptr, dim: int, keepdim: int) -> ptr from "torch_std_helper"

# ==============================================================================
# 索引与采样
# ==============================================================================

extern fn torch_std_argmax(a: ptr) -> int from "torch_std_helper"
extern fn torch_std_argmax_dim1(a: ptr, dim: int) -> int from "torch_std_helper"
extern fn torch_std_multinomial(probs: ptr, num_samples: int, replacement: int) -> ptr from "torch_std_helper"
extern fn torch_std_gather(input: ptr, dim: int, index: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_index_select(input: ptr, dim: int, index: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_index_tensor(input: ptr, index: ptr) -> ptr from "torch_std_helper"

# ==============================================================================
# 形状操作
# ==============================================================================

extern fn torch_std_reshape(a: ptr, shape: ptr, ndim: int) -> ptr from "torch_std_helper"
extern fn torch_std_transpose(a: ptr, dim0: int, dim1: int) -> ptr from "torch_std_helper"
extern fn torch_std_permute(a: ptr, dims: ptr, ndim: int) -> ptr from "torch_std_helper"
extern fn torch_std_squeeze(a: ptr, dim: int) -> ptr from "torch_std_helper"
extern fn torch_std_unsqueeze(a: ptr, dim: int) -> ptr from "torch_std_helper"
extern fn torch_std_cat(tensors: ptr, n: int, dim: int) -> ptr from "torch_std_helper"
extern fn torch_std_stack(tensors: ptr, n: int, dim: int) -> ptr from "torch_std_helper"
extern fn torch_std_slice(a: ptr, dim: int, start: int, end: int, step: int) -> ptr from "torch_std_helper"

# ==============================================================================
# 神经网络层
# ==============================================================================

extern fn torch_std_matmul(a: ptr, b: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_linear(input: ptr, weight: ptr, bias: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_attention(q: ptr, k: ptr, v: ptr, heads: int, mask: ptr, skip_reshape: int) -> ptr from "torch_std_helper"
extern fn torch_std_conv1d(input: ptr, weight: ptr, bias: ptr,
    stride: int, padding: int, dilation: int, groups: int) -> ptr from "torch_std_helper"
extern fn torch_std_conv2d(input: ptr, weight: ptr, bias: ptr,
    stride_h: int, stride_w: int,
    pad_h: int, pad_w: int,
    dilation_h: int, dilation_w: int,
    groups: int) -> ptr from "torch_std_helper"
extern fn torch_std_max_pool2d(input: ptr, kernel_h: int, kernel_w: int,
    stride_h: int, stride_w: int,
    pad_h: int, pad_w: int,
    dilation_h: int, dilation_w: int) -> ptr from "torch_std_helper"
extern fn torch_std_avg_pool2d(input: ptr, kernel_h: int, kernel_w: int,
    stride_h: int, stride_w: int,
    pad_h: int, pad_w: int) -> ptr from "torch_std_helper"
extern fn torch_std_batch_norm2d(input: ptr, weight: ptr, bias: ptr,
    running_mean: ptr, running_var: ptr,
    training: int, momentum: float, eps: float) -> ptr from "torch_std_helper"
extern fn torch_std_layer_norm(input: ptr, weight: ptr, bias: ptr, eps: float) -> ptr from "torch_std_helper"
extern fn torch_std_rms_norm(input: ptr, weight: ptr, eps: float) -> ptr from "torch_std_helper"
extern fn torch_std_group_norm(input: ptr, weight: ptr, bias: ptr, num_groups: int, eps: float) -> ptr from "torch_std_helper"
 
# ==============================================================================
# 标量运算辅助
# ==============================================================================

extern fn torch_std_mul_scalar(t: ptr, s: float) -> ptr from "torch_std_helper"
extern fn torch_std_add_scalar(t: ptr, s: float) -> ptr from "torch_std_helper"

# ==============================================================================
# CLIP Tokenizer (libtorch_std_helper C++ BPE)
# ==============================================================================

extern fn torch_std_clip_tokenizer_create(vocab_path: str, merges_path: str) -> ptr from "torch_std_helper"
extern fn torch_std_clip_tokenizer_encode(tokenizer: ptr, text: str) -> ptr from "torch_std_helper"
extern fn torch_std_clip_tokenizer_free(tokenizer: ptr) -> void from "torch_std_helper"

# ==============================================================================
# CLIP Text Encoder (JIT module forward)
# ==============================================================================

extern fn torch_std_clip_text_forward(clip_module: ptr, token_ids: ptr, cast_to_f16: int) -> ptr from "torch_std_helper"
extern fn torch_std_sdxl_dual_clip(clip1: ptr, clip2: ptr, token_ids: ptr) -> ptr from "torch_std_helper"

# ==============================================================================
# JIT Module Loader
# ==============================================================================

extern fn torch_std_jit_load(path: str) -> ptr from "torch_std_helper"

# ==============================================================================
# Safetensors loader
# ==============================================================================

extern fn torch_std_safetensors_load(path: str) -> ptr from "torch_std_helper"
extern fn torch_std_safetensors_count(sd: ptr) -> int from "torch_std_helper"
extern fn torch_std_safetensors_name(sd: ptr, idx: int) -> str from "torch_std_helper"
extern fn torch_std_safetensors_tensor(sd: ptr, idx: int) -> ptr from "torch_std_helper"
extern fn torch_std_safetensors_get_tensor_by_name(sd: ptr, name: str) -> ptr from "torch_std_helper"
extern fn torch_std_safetensors_free(sd: ptr) -> void from "torch_std_helper"

# ==============================================================================
# SD UNet forward (C++ 完整 UNet 实现)
# ==============================================================================

extern fn torch_std_sd15_unet_forward_dict(
    sd_dict: ptr, input: ptr, timestep: ptr,
    text_emb: ptr,
    lora_A: ptr, lora_B: ptr,
    lora_indices: ptr, n_lora: int,
    lora_scale: float) -> ptr from "torch_std_helper"

extern fn torch_std_sdxl_unet_forward(
    weight_dict: ptr, input: ptr, timestep: ptr,
    text_emb: ptr, pooled_emb: ptr,
    os_h: float, os_w: float,
    crop_t: float, crop_l: float,
    ts_h: float, ts_w: float) -> ptr from "torch_std_helper"

# ==============================================================================
# VAE JIT encode/decode
# ==============================================================================

extern fn torch_std_vae_encode(vae: ptr, image: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_vae_decode(vae: ptr, latent: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_vae_encode_tiled(vae: ptr, image: ptr,
                                      tile_size: int, overlap: int) -> ptr from "torch_std_helper"
extern fn torch_std_vae_decode_tiled(vae: ptr, latent: ptr,
                                      tile_size: int, overlap: int) -> ptr from "torch_std_helper"

# ==============================================================================
# Sampler step functions
# ==============================================================================

extern fn torch_std_sample_ddim(noise_pred: ptr, x_t: ptr,
                                 alpha_bar_t: ptr, alpha_bar_prev: ptr,
                                 eta: float) -> ptr from "torch_std_helper"

extern fn torch_std_sample_ddim_from_sigma(noise_pred: ptr, x_t: ptr,
                                            sigma_t: ptr, sigma_prev: ptr,
                                            eta: float) -> ptr from "torch_std_helper"

extern fn torch_std_sample_euler(noise_pred: ptr, x_t: ptr,
                                  sigma_t: ptr, sigma_prev: ptr) -> ptr from "torch_std_helper"

extern fn torch_std_sample_euler_ancestral(noise_pred: ptr, x_t: ptr,
                                            sigma_t: ptr, sigma_prev: ptr) -> ptr from "torch_std_helper"

extern fn torch_std_sample_dpmpp_2m(noise_pred: ptr, x_t: ptr,
                                     sigma_t: ptr, sigma_prev: ptr,
                                     old_denoised: ptr, is_first_step: int) -> ptr from "torch_std_helper"

extern fn torch_std_sampler_sigmas(steps: int, sigma_min: float,
                                    sigma_max: float,
                                    schedule: str) -> ptr from "torch_std_helper"

# ==============================================================================
# LoRA matching helper
# ==============================================================================

extern fn torch_std_lora_match_to_unet(lora_dict: ptr, n_weights: int,
                                        out_indices: ptr,
                                        out_A: ptr, out_B: ptr,
                                        max_lora: int) -> int from "torch_std_helper"

# ==============================================================================
# T5 SentencePiece tokenizer
# ==============================================================================

extern fn torch_std_t5_tokenizer_create(path: str) -> ptr from "torch_std_helper"
extern fn torch_std_t5_tokenizer_encode(tok: ptr, text: str,
                                         max_len: int) -> ptr from "torch_std_helper"
extern fn torch_std_t5_tokenizer_free(tok: ptr) -> void from "torch_std_helper"

# ==============================================================================
# FLUX MMDiT forward
# ==============================================================================

extern fn torch_std_flux_forward(
    wdict: ptr, img: ptr, txt: ptr,
    timestep: ptr, img_pos: ptr,
    guidance: float, n_blocks: int,
    n_heads_img: int, n_heads_txt: int,
    head_dim: int) -> ptr from "torch_std_helper"

extern fn torch_std_flux_embed_nd(ids: ptr, dim: int, theta: float,
                                   axes_dim: ptr, n_axes: int) -> ptr from "torch_std_helper"

# ==============================================================================
# Flow matching step (for FLUX)
# ==============================================================================

extern fn torch_std_fm_step(velocity: ptr, x_t: ptr, dt: float) -> ptr from "torch_std_helper"

# ==============================================================================
# GGUF model loader
# ==============================================================================

extern fn torch_std_gguf_load(path: str) -> ptr from "torch_std_helper"
extern fn torch_std_gguf_tensor_count(model: ptr) -> int from "torch_std_helper"
extern fn torch_std_gguf_tensor_name(model: ptr, idx: int) -> str from "torch_std_helper"
extern fn torch_std_gguf_load_tensor(model: ptr, idx: int) -> ptr from "torch_std_helper"
extern fn torch_std_gguf_load_tensor_by_name(model: ptr, name: str) -> ptr from "torch_std_helper"
extern fn torch_std_gguf_free(model: ptr) -> void from "torch_std_helper"

# ==============================================================================
# Image processing
# ==============================================================================

extern fn torch_std_image_resize(img: ptr, new_h: int, new_w: int,
                                  mode: str) -> ptr from "torch_std_helper"
extern fn torch_std_image_crop(img: ptr, x: int, y: int, w: int, h: int) -> ptr from "torch_std_helper"
extern fn torch_std_load_image(path: str) -> ptr from "torch_std_helper"

# ==============================================================================
# DDPM noise utilities
# ==============================================================================

extern fn torch_std_ddpm_betas(T: int, beta_start: float,
                                beta_end: float) -> ptr from "torch_std_helper"
extern fn torch_std_ddpm_add_noise(latent: ptr, noise: ptr, timestep: ptr,
                                    alpha_bar: ptr) -> ptr from "torch_std_helper"

# ==============================================================================
# Scalar extraction from 1-element tensor
# ==============================================================================

extern fn torch_std_to_double_array(t: ptr, out: ptr, n: int) -> void from "torch_std_helper"
extern fn torch_std_to_float_array(t: ptr, out: ptr, n: int) -> void from "torch_std_helper"
extern fn torch_std_to_int64_array(t: ptr, out: ptr, n: int) -> void from "torch_std_helper"

# ==============================================================================
# Missing extern fn — align libtorch_std_helper.h (172 total, +58 added here)
# ==============================================================================

# Tensor creation extras
extern fn torch_std_randint(low: int, high: int, shape: ptr, ndim: int, dtype: int) -> ptr from "torch_std_helper"
extern fn torch_std_tensor_from_blob_3d(data: ptr, d0: int, d1: int, d2: int, dtype: int) -> ptr from "torch_std_helper"
extern fn torch_std_detach(t: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_randn(shape: ptr, ndim: int, dtype: int) -> ptr from "torch_std_helper"

# Array utilities
extern fn torch_std_float_array_ptr(arr: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_int_array_ptr(arr: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_float_array_ptr_offset(arr: ptr, offset: int) -> ptr from "torch_std_helper"
extern fn torch_std_int_array_ptr_offset(arr: ptr, offset: int) -> ptr from "torch_std_helper"

# Compare ops
extern fn torch_std_eq(a: ptr, b: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_gt(a: ptr, b: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_lt(a: ptr, b: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_ge(a: ptr, b: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_le(a: ptr, b: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_clamp(a: ptr, min_val: float, max_val: float) -> ptr from "torch_std_helper"
extern fn torch_std_where(condition: ptr, x: ptr, y: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_masked_select(a: ptr, mask: ptr) -> ptr from "torch_std_helper"

# JIT module management
extern fn torch_std_jit_forward(module: ptr, input_tensor: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_jit_module_delete(module: ptr) -> void from "torch_std_helper"
extern fn torch_std_jit_load_module(path: str) -> ptr from "torch_std_helper"
extern fn torch_std_jit_named_parameters(module: ptr, out_ptrs: ptr, max_n: int) -> int from "torch_std_helper"
extern fn torch_std_jit_parameters(module: ptr, out_ptrs: ptr, max_n: int) -> int from "torch_std_helper"

# Image I/O extras
extern fn torch_std_load_image_png(path: str) -> ptr from "torch_std_helper"
extern fn torch_std_save_image(tensor: ptr, path: str, as_pgm: int) -> void from "torch_std_helper"
extern fn torch_std_save_image_png(tensor: ptr, path: str) -> void from "torch_std_helper"
extern fn torch_std_image_composite(bg: ptr, fg: ptr, x: int, y: int) -> ptr from "torch_std_helper"
extern fn torch_std_color_convert(img: ptr, src_fmt: int, dst_fmt: int) -> ptr from "torch_std_helper"

# Loss functions
extern fn torch_std_mse_loss(pred: ptr, target: ptr, reduction: str) -> ptr from "torch_std_helper"
extern fn torch_std_l1_loss(pred: ptr, target: str, reduction: str) -> ptr from "torch_std_helper"
extern fn torch_std_cross_entropy_loss(logits: ptr, target: ptr, reduction: str) -> ptr from "torch_std_helper"
extern fn torch_std_nll_loss(log_probs: ptr, target: ptr, reduction: str) -> ptr from "torch_std_helper"
extern fn torch_std_bce_loss(pred: ptr, target: ptr, reduction: str) -> ptr from "torch_std_helper"
extern fn torch_std_bce_with_logits_loss(pred: ptr, target: ptr, reduction: str) -> ptr from "torch_std_helper"
extern fn torch_std_copy_probs(logits: ptr, temperature: float) -> ptr from "torch_std_helper"

# Gradient / autograd
extern fn torch_std_requires_grad(t: ptr) -> int from "torch_std_helper"
extern fn torch_std_set_requires_grad(t: ptr, requires_grad: int) -> ptr from "torch_std_helper"
extern fn torch_std_has_grad(t: ptr) -> int from "torch_std_helper"
extern fn torch_std_grad(t: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_backward(loss: ptr) -> void from "torch_std_helper"
extern fn torch_std_backward_retain_graph(loss: ptr) -> void from "torch_std_helper"
extern fn torch_std_zero_grad(params: ptr, n: int) -> void from "torch_std_helper"
extern fn torch_std_clip_grad_norm(params: ptr, n: int, max_norm: float) -> void from "torch_std_helper"

# Optimizers (SGD / Adam / AdamW)
extern fn torch_std_sgd_create(params: ptr, n: int, lr: float, momentum: float,
                                weight_decay: float) -> ptr from "torch_std_helper"
extern fn torch_std_adam_create(params: ptr, n: int, lr: float, beta1: float,
                                 beta2: float, eps: float, weight_decay: float) -> ptr from "torch_std_helper"
extern fn torch_std_adamw_create(params: ptr, n: int, lr: float, beta1: float,
                                  beta2: float, eps: float, weight_decay: float) -> ptr from "torch_std_helper"
extern fn torch_std_optimizer_step(optim: ptr) -> void from "torch_std_helper"
extern fn torch_std_optimizer_zero_grad(optim: ptr) -> void from "torch_std_helper"
extern fn torch_std_optimizer_destroy(optim: ptr) -> void from "torch_std_helper"

# ControlNet
extern fn torch_std_controlnet_forward(weight_ptrs: ptr, n_weights: int, x: ptr,
                                        timestep: ptr, text_emb: ptr,
                                        hint: ptr, num_hint_channels: int) -> ptr from "torch_std_helper"
extern fn torch_std_controlnet_apply(unet_output: ptr, control_features: ptr,
                                      strength: float) -> ptr from "torch_std_helper"

# LoRA
extern fn torch_std_lora_apply(weight: ptr, lora_down: ptr, lora_up: ptr,
                                scale: float) -> ptr from "torch_std_helper"
extern fn torch_std_lora_merge_into(model_dict: ptr, lora_dict: ptr,
                                     prefix: str, scale: float) -> int from "torch_std_helper"

# Normalization layers (batch norm)
extern fn torch_std_batch_norm1d(input: ptr, weight: ptr, bias: ptr,
                                  running_mean: ptr, running_var: ptr,
                                  training: int, momentum: float,
                                  eps: float) -> ptr from "torch_std_helper"

# SD UNet forward
extern fn torch_std_sd_unet_forward(wdict: ptr, x: ptr, timestep: ptr,
                                     context: ptr, y: ptr, guidance: ptr,
                                     num_tokens: int) -> ptr from "torch_std_helper"

# Flow matching
extern fn torch_std_fm_sigmas(steps: int, sigma_min: float, sigma_max: float) -> ptr from "torch_std_helper"

# Save/load state dict
extern fn torch_std_save_state_dict(module: ptr, path: str) -> void from "torch_std_helper"

# ==============================================================================
# Phase 15 新模型 (纯 libTorch C++ forward)
# ==============================================================================

# GLIGEN/IP-Adapter — UNet forward v2 with per-block attention hooks
extern fn torch_std_sd_unet_forward_v2(
    weight_ptrs: ptr, n_weights: int,
    input_ptr: ptr, timestep_ptr: ptr, text_emb_ptr: ptr,
    lora_A_ptrs: ptr, lora_B_ptrs: ptr, lora_target_indices: ptr, n_lora: int, lora_scale: float,
    gligen_objs: ptr, gligen_alphas: ptr, gligen_block_indices: ptr, n_gligen_blocks: int,
    ip_adapt_img: ptr, ip_adapt_scale: float) -> ptr from "torch_std_helper"

# Stable Cascade Stage C
extern fn torch_std_stable_cascade_stage_c(
    weight_ptrs: ptr, n_weights: int,
    x: ptr, r: ptr, timestep: ptr,
    clip_text: ptr, clip_text_pooled: ptr, clip_img: ptr) -> ptr from "torch_std_helper"

# PixArt DiT
extern fn torch_std_pixart_forward(
    weight_ptrs: ptr, n_weights: int,
    x: ptr, timestep: ptr, y: ptr,
    height: int, width: int, patch_size: int) -> ptr from "torch_std_helper"

# Hunyuan Video 3D UNet
extern fn torch_std_hunyuan_video_forward(
    sd_dict: ptr,
    x: ptr, timestep: ptr, text_emb: ptr,
    n_frames: int, height: int, width: int) -> ptr from "torch_std_helper"

# Wan Video 3D UNet
extern fn torch_std_wan_video_forward(
    sd_dict: ptr,
    x: ptr, timestep: ptr, text_emb: ptr,
    n_frames: int, height: int, width: int) -> ptr from "torch_std_helper"

# Cosmos Video
extern fn torch_std_cosmos_forward(
    sd_dict: ptr,
    x: ptr, timestep: ptr, text_emb: ptr,
    n_frames: int, height: int, width: int) -> ptr from "torch_std_helper"

# SD3 MMDiT
extern fn torch_std_sd3_mmdit_forward(
    sd_dict: ptr,
    x: ptr, timestep: ptr, y: ptr,
    cfg_scale: float) -> ptr from "torch_std_helper"
