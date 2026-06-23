#ifndef TORCH_STD_HELPER_H
#define TORCH_STD_HELPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// ============================================================
// 张量创建与转换
// ============================================================
void* torch_std_tensor_from_blob(void* data, int64_t* shape, int ndim, int dtype);
void* torch_std_tensor_from_blob_3d(void* data, int d0, int d1, int d2, int dtype);
void* torch_std_zeros(int64_t* shape, int ndim, int dtype);
void* torch_std_ones(int64_t* shape, int ndim, int dtype);
void* torch_std_empty(int64_t* shape, int ndim, int dtype);
void* torch_std_full(int64_t* shape, int ndim, double value, int dtype);
void* torch_std_randn(int64_t* shape, int ndim, int dtype);
void* torch_std_randint(int64_t low, int64_t high, int64_t* shape, int ndim, int dtype);
void* torch_std_clone(void* t);
void* torch_std_detach(void* t);
void* torch_std_to_dtype(void* t, int dtype);
void  torch_std_delete_tensor(void* t);
void* torch_std_arange(int64_t start, int64_t end, int64_t step, int dtype);
 
// 数据类型常量
#define TORCH_DTYPE_FLOAT32 0
#define TORCH_DTYPE_FLOAT64 1
#define TORCH_DTYPE_INT32   2
#define TORCH_DTYPE_INT64   3

// 数据导出
int64_t torch_std_numel(void* t);
int     torch_std_ndim(void* t);
void    torch_std_shape(void* t, int64_t* out);
void    torch_std_to_double_array(void* t, double* out, int64_t n);
void    torch_std_to_float_array(void* t, float* out, int64_t n);
void    torch_std_to_int64_array(void* t, int64_t* out, int64_t n);
void    torch_std_copy_probs(void* t, double* out, int n);

// ============================================================
// StaticPy array -> C pointer helpers
// ============================================================
void* torch_std_float_array_ptr(void* arr);
void* torch_std_int_array_ptr(void* arr);
void* torch_std_float_array_ptr_offset(void* arr, int offset);
void* torch_std_int_array_ptr_offset(void* arr, int offset);

// ============================================================
// 数学运算
// ============================================================
void* torch_std_add(void* a, void* b);
void* torch_std_sub(void* a, void* b);
void* torch_std_mul(void* a, void* b);
void* torch_std_div(void* a, void* b);
void* torch_std_pow(void* a, double exp);
void* torch_std_exp(void* a);
void* torch_std_log(void* a);
void* torch_std_sqrt(void* a);
void* torch_std_neg(void* a);
void* torch_std_abs(void* a);
void* torch_std_cos(void* a);
void* torch_std_sin(void* a);
void* torch_std_mul_scalar(void* a, double s);
void* torch_std_add_scalar(void* a, double s);
 
// ============================================================
// 激活函数
// ============================================================
void* torch_std_relu(void* a);
void* torch_std_leaky_relu(void* a, double negative_slope);
void* torch_std_sigmoid(void* a);
void* torch_std_tanh(void* a);
void* torch_std_softmax(void* a, int64_t dim);
void* torch_std_log_softmax(void* a, int64_t dim);

// ============================================================
// 归约
// ============================================================
void* torch_std_sum(void* a);
void* torch_std_sum_dim(void* a, int64_t dim, int keepdim);
void* torch_std_mean(void* a);
void* torch_std_mean_dim(void* a, int64_t dim, int keepdim);
void* torch_std_max(void* a);
void* torch_std_max_dim(void* a, int64_t dim, int keepdim);
void* torch_std_min(void* a);
void* torch_std_min_dim(void* a, int64_t dim, int keepdim);

// ============================================================
// 索引与采样
// ============================================================
int64_t torch_std_argmax(void* a);
int64_t torch_std_argmax_dim1(void* a, int64_t dim);
void*   torch_std_multinomial(void* probs, int64_t num_samples, int replacement);
void*   torch_std_gather(void* input, int64_t dim, void* index);
void*   torch_std_index_select(void* input, int64_t dim, void* index);
void*   torch_std_index_tensor(void* input, void* index);

// ============================================================
// 形状操作
// ============================================================
void* torch_std_reshape(void* a, int64_t* shape, int ndim);
void* torch_std_transpose(void* a, int64_t dim0, int64_t dim1);
void* torch_std_permute(void* a, int64_t* dims, int ndim);
void* torch_std_squeeze(void* a, int64_t dim);
void* torch_std_unsqueeze(void* a, int64_t dim);
void* torch_std_cat(void** tensors, int n, int64_t dim);
void* torch_std_stack(void** tensors, int n, int64_t dim);

// ============================================================
// 矩阵与线性层
// ============================================================
void* torch_std_matmul(void* a, void* b);
void* torch_std_linear(void* input, void* weight, void* bias);
void* torch_std_conv1d(void* input, void* weight, void* bias,
                       int64_t stride, int64_t padding, int64_t dilation, int64_t groups);
void* torch_std_conv2d(void* input, void* weight, void* bias,
                       int64_t stride_h, int64_t stride_w,
                       int64_t padding_h, int64_t padding_w,
                       int64_t dilation_h, int64_t dilation_w,
                       int64_t groups);
void* torch_std_max_pool2d(void* input, int64_t kernel_h, int64_t kernel_w,
                           int64_t stride_h, int64_t stride_w,
                           int64_t padding_h, int64_t padding_w,
                           int64_t dilation_h, int64_t dilation_w);
void* torch_std_avg_pool2d(void* input, int64_t kernel_h, int64_t kernel_w,
                           int64_t stride_h, int64_t stride_w,
                           int64_t padding_h, int64_t padding_w);
void* torch_std_batch_norm1d(void* input, void* weight, void* bias,
                             void* running_mean, void* running_var,
                             int training, double momentum, double eps);
void* torch_std_batch_norm2d(void* input, void* weight, void* bias,
                             void* running_mean, void* running_var,
                             int training, double momentum, double eps);

// ============================================================
// 损失函数
// ============================================================
void* torch_std_mse_loss(void* pred, void* target, const char* reduction);
void* torch_std_l1_loss(void* pred, void* target, const char* reduction);
void* torch_std_cross_entropy_loss(void* logits, void* target, const char* reduction);
void* torch_std_nll_loss(void* log_probs, void* target, const char* reduction);
void* torch_std_bce_loss(void* pred, void* target, const char* reduction);
void* torch_std_bce_with_logits_loss(void* logits, void* target, const char* reduction);

// ============================================================
// 自动微分
// ============================================================
void* torch_std_requires_grad(void* t);
void* torch_std_set_requires_grad(void* t, int requires_grad);
void  torch_std_backward(void* loss);
void  torch_std_backward_retain_graph(void* loss);
void* torch_std_grad(void* t);
void  torch_std_zero_grad(void* t);
int   torch_std_has_grad(void* t);

// ============================================================
// 优化器
// ============================================================
void* torch_std_sgd_create(void** params, int n, double lr, double momentum,
                           double dampening, double weight_decay, int nesterov);
void* torch_std_adam_create(void** params, int n, double lr,
                            double beta1, double beta2, double eps,
                            double weight_decay, int amsgrad);
void* torch_std_adamw_create(void** params, int n, double lr,
                             double beta1, double beta2, double eps,
                             double weight_decay, int amsgrad);
void  torch_std_optimizer_step(void* opt);
void  torch_std_optimizer_zero_grad(void* opt);
void  torch_std_optimizer_destroy(void* opt);

// ============================================================
// 工具
// ============================================================
void* torch_std_narrow(void* a, int64_t dim, int64_t start, int64_t length);
void* torch_std_slice(void* a, int64_t dim, int64_t start, int64_t end, int64_t step);
int64_t torch_std_size(void* a, int64_t dim);
void* torch_std_masked_select(void* a, void* mask);
void* torch_std_where(void* condition, void* x, void* y);
void* torch_std_eq(void* a, void* b);
void* torch_std_gt(void* a, void* b);
void* torch_std_lt(void* a, void* b);
void* torch_std_ge(void* a, void* b);
void* torch_std_le(void* a, void* b);
void* torch_std_clamp(void* a, double min_val, double max_val);
void  torch_std_clip_grad_norm(void** params, int n, double max_norm);

// 注意力 (multi-head scaled dot-product, handles reshape/permute internally)
void* torch_std_attention(void* q, void* k, void* v, int64_t heads, void* mask, int skip_reshape);

// 随机种子
void torch_std_manual_seed(int64_t seed);

// 设备
int torch_std_cuda_is_available(void);
void* torch_std_to_cuda(void* t);
void* torch_std_to_cpu(void* t);
int torch_std_is_cuda(void* t);

// ============================================================
// TorchScript JIT 模型加载与推理
// ============================================================
void* torch_std_jit_load(const char* path);
void* torch_std_jit_forward(void* module, void* input_tensor);
void torch_std_jit_module_delete(void* module);
int   torch_std_jit_parameters(void* module, void** out_params, char** out_names, int max_n);

// ============================================================
// 模型参数序列化（支持 StaticPy 续训）
// ============================================================
int   torch_std_save_state_dict(void* module, const char* path);
void* torch_std_jit_load_module(const char* path);

// ============================================================
// SD UNet Forward with LoRA (training-gradable)
// ============================================================
void* torch_std_sd_unet_forward(
    void** weight_ptrs, int n_weights,
    void* input, void* timestep, void* text_emb,
    void** lora_A, void** lora_B,
    int* lora_target_indices, int n_lora,
    double lora_scale);

// SD 1.5 UNet forward from safetensors dict (convenience wrapper)
// Extracts all needed weights by name internally.
void* torch_std_sd15_unet_forward_dict(
    void* safetensors_dict,
    void* input_ptr, void* timestep_ptr, void* text_emb_ptr,
    void** lora_A, void** lora_B,
    int* lora_target_indices, int n_lora,
    double lora_scale);

// ============================================================
// Image I/O
// ============================================================
void* torch_std_load_image(const char* path);
void  torch_std_save_image(void* tensor, const char* path, int as_pgm);

// ============================================================
// DDPM Scheduler
// ============================================================
void* torch_std_ddpm_betas(int T, double beta_start, double beta_end);
void* torch_std_ddpm_add_noise(void* latent, void* noise, void* timestep,
                                void* alphas_cumprod);

// ============================================================
// JIT Module parameter enumeration
// ============================================================
int torch_std_jit_named_parameters(void* module, void** out_ptrs,
                                    const char** out_names, int max_n);

// ============================================================
// safetensors loader
// ============================================================
void* torch_std_safetensors_load(const char* path);
int   torch_std_safetensors_count(void* dict);
const char* torch_std_safetensors_name(void* dict, int idx);
void* torch_std_safetensors_tensor(void* dict, int idx);
void  torch_std_safetensors_free(void* dict);
void* torch_std_safetensors_get_tensor_by_name(void* dict, const char* name);
void* torch_std_lora_apply(void* weight, void* lora_A, void* lora_B, double scale);

// ============================================================
// Samplers (DDIM / Euler / Euler Ancestral / DPM++ 2M)
// ============================================================
void* torch_std_sample_ddim_from_sigma(void* noise_pred, void* x_t,
                                        void* sigma_t, void* sigma_prev,
                                        double eta);
void* torch_std_sample_ddim(void* noise_pred, void* x_t,
                             void* alpha_bar_t, void* alpha_bar_prev, double eta);
void* torch_std_sample_euler(void* noise_pred, void* x_t,
                              void* sigma_t, void* sigma_prev);
void* torch_std_sample_euler_ancestral(void* noise_pred, void* x_t,
                                        void* sigma_t, void* sigma_prev);
void* torch_std_sample_dpmpp_2m(void* noise_pred, void* x_t,
                                 void* sigma_t, void* sigma_prev,
                                 void* old_denoised, int is_first_step);
void* torch_std_sampler_sigmas(int steps, double sigma_min, double sigma_max,
                                const char* schedule);

// ============================================================
// Image processing
// ============================================================
void* torch_std_image_resize(void* img, int new_h, int new_w, const char* mode);
void* torch_std_image_crop(void* img, int x, int y, int w, int h);
void* torch_std_image_composite(void* base, void* overlay, int x, int y);
void* torch_std_color_convert(void* img, const char* from, const char* to);

// ============================================================
// ControlNet
// ============================================================
void* torch_std_controlnet_forward(void** weight_ptrs, int n_weights,
                                    void* input, void* timestep,
                                    void* text_emb, void* hint,
                                    int num_hint_channels);
void* torch_std_controlnet_apply(void* unet_features, void* control_features,
                                  double strength);

// ============================================================
// LoRA key-to-index matching (for SD1.5 UNet)
// ============================================================
int torch_std_lora_match_to_unet(void* lora_dict, int n_weights,
                                  int64_t* out_indices,
                                  void** out_A, void** out_B,
                                  int max_lora);

// ============================================================
// VAE encode/decode (non-tiled and tiled)
// ============================================================
void* torch_std_vae_encode(void* vae_module, void* image);
void* torch_std_vae_decode(void* vae_module, void* latent);
void* torch_std_vae_encode_tiled(void* vae_module, void* image,
                                  int tile_size, int overlap);
void* torch_std_vae_decode_tiled(void* vae_module, void* latent,
                                  int tile_size, int overlap);

// ============================================================
// LoRA merge into model dict
// ============================================================
int torch_std_lora_merge_into(void* model_dict, void* lora_dict,
                               const char* prefix, double scale);

// ============================================================
// PNG/JPEG (requires -DWITH_LIBPNG / -DWITH_LIBJPEG at build time)
// ============================================================
void* torch_std_load_image_png(const char* path);
void  torch_std_save_image_png(void* tensor, const char* path);

// ============================================================
// CLIP BPE Tokenizer
// ============================================================
void* torch_std_clip_tokenizer_create(const char* vocab_path, const char* merges_path);
void* torch_std_clip_tokenizer_encode(void* tokenizer, const char* text);
void  torch_std_clip_tokenizer_free(void* tokenizer);

// ============================================================
// CLIP Text Encoder forward
// ============================================================
void* torch_std_clip_text_forward(void* clip_module, void* token_ids_tensor,
                                   int cast_to_float16);

// ============================================================
// SDXL UNet forward (named-weight dict-based)
// ============================================================
void* torch_std_sdxl_unet_forward(
    void* weight_dict_ptr,
    void* input_ptr, void* timestep_ptr,
    void* text_emb_ptr, void* pooled_emb_ptr,
    double original_size_h, double original_size_w,
    double crop_top, double crop_left,
    double target_size_h, double target_size_w);

// ============================================================
// SDXL Dual CLIP (run two, concat outputs)
// ============================================================
void* torch_std_sdxl_dual_clip(void* clip1, void* clip2, void* token_ids);

// ============================================================
// Flow Matching scheduler
// ============================================================
void* torch_std_fm_sigmas(int steps, double sigma_min, double sigma_max);
void* torch_std_fm_step(void* velocity, void* x_t, double dt);

// ============================================================
// 规范化层
// ============================================================
void* torch_std_layer_norm(void* input, void* weight, void* bias, double eps);
void* torch_std_rms_norm(void* input, void* weight, double eps);
void* torch_std_group_norm(void* input, void* weight, void* bias, int64_t num_groups, double eps);

// ============================================================
// FLUX MMDiT forward
// ============================================================
void* torch_std_flux_forward(
    void* wdict_ptr, void* img_ptr, void* txt_ptr,
    void* timestep_ptr, void* img_pos_ptr,
    double guidance_scale, int n_blocks,
    int n_heads_img, int n_heads_txt, int head_dim);
void* torch_std_flux_embed_nd(void* ids, int dim, double theta, int64_t* axes_dim, int n_axes);

// ============================================================
// T5 SentencePiece tokenizer
// ============================================================
void* torch_std_t5_tokenizer_create(const char* model_path);
void* torch_std_t5_tokenizer_encode(void* tokenizer, const char* text, int max_len);
void  torch_std_t5_tokenizer_free(void* tokenizer);

// ============================================================
// GGUF model loader + dequantize
// ============================================================
void* torch_std_gguf_load(const char* path);
int   torch_std_gguf_tensor_count(void* model);
const char* torch_std_gguf_tensor_name(void* model, int idx);
void* torch_std_gguf_load_tensor(void* model, int idx);
void* torch_std_gguf_load_tensor_by_name(void* model, const char* name);
void  torch_std_gguf_free(void* model);

// ============================================================
// GLIGEN — gated cross-attention injection into SD UNet
// ============================================================
// Modified UNet forward with per-block GLIGEN gated attention.
// Same as torch_std_sd_unet_forward but with extra gligen params.
// gligen_objs: (B, N_objs, D) object embeddings from PositionNet
// gligen_alphas: (B, n_blocks, 2) [alpha_attn, alpha_dense] per transformer block
// gligen_block_indices: int array of block indices to apply gligen
// n_gligen_blocks: number of gligen blocks (0 = no gligen)
// ip_adapt_img: (B, N_img, D) IP-Adapter image embeddings (or NULL)
// ip_adapt_scale: IP-Adapter injection scale
void* torch_std_sd_unet_forward_v2(
    void** weight_ptrs, int n_weights,
    void* input_ptr, void* timestep_ptr, void* text_emb_ptr,
    void** lora_A_ptrs, void** lora_B_ptrs, int* lora_target_indices, int n_lora, double lora_scale,
    void* gligen_objs, void* gligen_alphas, int* gligen_block_indices, int n_gligen_blocks,
    void* ip_adapt_img, double ip_adapt_scale);

// ============================================================
// Stable Cascade Stage C — 主扩散模型 (transformer backbone)
// ============================================================
void* torch_std_stable_cascade_stage_c(
    void** weight_ptrs, int n_weights,
    void* x, void* r, void* timestep,
    void* clip_text, void* clip_text_pooled, void* clip_img);

// ============================================================
// PixArt — DiT (Diffusion Transformer)
// ============================================================
void* torch_std_pixart_forward(
    void** weight_ptrs, int n_weights,
    void* x, void* timestep, void* y,  // y = T5 text embeddings
    int height, int width, int patch_size);

// ============================================================
// Hunyuan Video — 3D UNet (FLUX-based with 3D input)
// ============================================================
void* torch_std_hunyuan_video_forward(
    void* sd_dict,
    void* x, void* timestep, void* text_emb,
    int n_frames, int height, int width);

// ============================================================
// Wan Video — 3D UNet with RoPE
// ============================================================
void* torch_std_wan_video_forward(
    void* sd_dict,
    void* x, void* timestep, void* text_emb,
    int n_frames, int height, int width);

// ============================================================
// Cosmos — Video diffusion model
// ============================================================
void* torch_std_cosmos_forward(
    void* sd_dict,
    void* x, void* timestep, void* text_emb,
    int n_frames, int height, int width);

// ============================================================
// SD3 MMDiT — Joint attention with text conditioning
// ============================================================
void* torch_std_sd3_mmdit_forward(
    void* sd_dict,          // safetensors dict (unordered_map<string, Tensor>)
    void* x,                // (B, C, H, W) latent
    void* timestep,         // (B,) timesteps
    void* y,                // (B, L, D) text/T5 embeddings
    double cfg_scale);      // guidance scale (CFG)

#ifdef __cplusplus
}
#endif

#endif
