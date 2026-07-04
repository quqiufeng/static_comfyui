;; extern FFI functions
(load-shared-object "libtorch_std_helper.so")
(define torch_std_tensor_from_blob
  (foreign-procedure "torch_std_tensor_from_blob" (void* void* int int) void*))
(define torch_std_zeros
  (foreign-procedure "torch_std_zeros" (void* int int) void*))
(define torch_std_ones
  (foreign-procedure "torch_std_ones" (void* int int) void*))
(define torch_std_empty
  (foreign-procedure "torch_std_empty" (void* int int) void*))
(define torch_std_full
  (foreign-procedure "torch_std_full" (void* int double int) void*))
(define torch_std_clone
  (foreign-procedure "torch_std_clone" (void*) void*))
(define torch_std_to_dtype
  (foreign-procedure "torch_std_to_dtype" (void* int) void*))
(define torch_std_delete_tensor
  (foreign-procedure "torch_std_delete_tensor" (void*) void))
(define torch_std_arange
  (foreign-procedure "torch_std_arange" (int int int int) void*))
(define torch_std_narrow
  (foreign-procedure "torch_std_narrow" (void* int int int) void*))
(define torch_std_size
  (foreign-procedure "torch_std_size" (void* int) int))
(define torch_std_numel
  (foreign-procedure "torch_std_numel" (void*) int))
(define torch_std_ndim
  (foreign-procedure "torch_std_ndim" (void*) int))
(define torch_std_shape
  (foreign-procedure "torch_std_shape" (void* void*) void))
(define torch_std_cuda_is_available
  (foreign-procedure "torch_std_cuda_is_available" () int))
(define torch_std_to_cuda
  (foreign-procedure "torch_std_to_cuda" (void*) void*))
(define torch_std_to_cpu
  (foreign-procedure "torch_std_to_cpu" (void*) void*))
(define torch_std_is_cuda
  (foreign-procedure "torch_std_is_cuda" (void*) int))
(define torch_std_manual_seed
  (foreign-procedure "torch_std_manual_seed" (int) void))
(define torch_std_add
  (foreign-procedure "torch_std_add" (void* void*) void*))
(define torch_std_sub
  (foreign-procedure "torch_std_sub" (void* void*) void*))
(define torch_std_mul
  (foreign-procedure "torch_std_mul" (void* void*) void*))
(define torch_std_div
  (foreign-procedure "torch_std_div" (void* void*) void*))
(define torch_std_pow
  (foreign-procedure "torch_std_pow" (void* double) void*))
(define torch_std_exp
  (foreign-procedure "torch_std_exp" (void*) void*))
(define torch_std_log
  (foreign-procedure "torch_std_log" (void*) void*))
(define torch_std_sqrt
  (foreign-procedure "torch_std_sqrt" (void*) void*))
(define torch_std_neg
  (foreign-procedure "torch_std_neg" (void*) void*))
(define torch_std_abs
  (foreign-procedure "torch_std_abs" (void*) void*))
(define torch_std_cos
  (foreign-procedure "torch_std_cos" (void*) void*))
(define torch_std_sin
  (foreign-procedure "torch_std_sin" (void*) void*))
(define torch_std_relu
  (foreign-procedure "torch_std_relu" (void*) void*))
(define torch_std_leaky_relu
  (foreign-procedure "torch_std_leaky_relu" (void* double) void*))
(define torch_std_sigmoid
  (foreign-procedure "torch_std_sigmoid" (void*) void*))
(define torch_std_tanh
  (foreign-procedure "torch_std_tanh" (void*) void*))
(define torch_std_softmax
  (foreign-procedure "torch_std_softmax" (void* int) void*))
(define torch_std_log_softmax
  (foreign-procedure "torch_std_log_softmax" (void* int) void*))
(define torch_std_sum
  (foreign-procedure "torch_std_sum" (void*) void*))
(define torch_std_sum_dim
  (foreign-procedure "torch_std_sum_dim" (void* int int) void*))
(define torch_std_mean
  (foreign-procedure "torch_std_mean" (void*) void*))
(define torch_std_mean_dim
  (foreign-procedure "torch_std_mean_dim" (void* int int) void*))
(define torch_std_max
  (foreign-procedure "torch_std_max" (void*) void*))
(define torch_std_max_dim
  (foreign-procedure "torch_std_max_dim" (void* int int) void*))
(define torch_std_min
  (foreign-procedure "torch_std_min" (void*) void*))
(define torch_std_min_dim
  (foreign-procedure "torch_std_min_dim" (void* int int) void*))
(define torch_std_argmax
  (foreign-procedure "torch_std_argmax" (void*) int))
(define torch_std_argmax_dim1
  (foreign-procedure "torch_std_argmax_dim1" (void* int) int))
(define torch_std_multinomial
  (foreign-procedure "torch_std_multinomial" (void* int int) void*))
(define torch_std_gather
  (foreign-procedure "torch_std_gather" (void* int void*) void*))
(define torch_std_index_select
  (foreign-procedure "torch_std_index_select" (void* int void*) void*))
(define torch_std_index_tensor
  (foreign-procedure "torch_std_index_tensor" (void* void*) void*))
(define torch_std_reshape
  (foreign-procedure "torch_std_reshape" (void* void* int) void*))
(define torch_std_transpose
  (foreign-procedure "torch_std_transpose" (void* int int) void*))
(define torch_std_permute
  (foreign-procedure "torch_std_permute" (void* void* int) void*))
(define torch_std_squeeze
  (foreign-procedure "torch_std_squeeze" (void* int) void*))
(define torch_std_unsqueeze
  (foreign-procedure "torch_std_unsqueeze" (void* int) void*))
(define torch_std_cat
  (foreign-procedure "torch_std_cat" (void* int int) void*))
(define torch_std_stack
  (foreign-procedure "torch_std_stack" (void* int int) void*))
(define torch_std_slice
  (foreign-procedure "torch_std_slice" (void* int int int int) void*))
(define torch_std_matmul
  (foreign-procedure "torch_std_matmul" (void* void*) void*))
(define torch_std_linear
  (foreign-procedure "torch_std_linear" (void* void* void*) void*))
(define torch_std_attention
  (foreign-procedure "torch_std_attention" (void* void* void* int void* int) void*))
(define torch_std_conv1d
  (foreign-procedure "torch_std_conv1d" (void* void* void* int int int int) void*))
(define torch_std_conv2d
  (foreign-procedure "torch_std_conv2d" (void* void* void* int int int int int int int) void*))
(define torch_std_max_pool2d
  (foreign-procedure "torch_std_max_pool2d" (void* int int int int int int int int) void*))
(define torch_std_avg_pool2d
  (foreign-procedure "torch_std_avg_pool2d" (void* int int int int int int) void*))
(define torch_std_batch_norm2d
  (foreign-procedure "torch_std_batch_norm2d" (void* void* void* void* void* int double double) void*))
(define torch_std_layer_norm
  (foreign-procedure "torch_std_layer_norm" (void* void* void* double) void*))
(define torch_std_rms_norm
  (foreign-procedure "torch_std_rms_norm" (void* void* double) void*))
(define torch_std_group_norm
  (foreign-procedure "torch_std_group_norm" (void* void* void* int double) void*))
(define torch_std_mul_scalar
  (foreign-procedure "torch_std_mul_scalar" (void* double) void*))
(define torch_std_add_scalar
  (foreign-procedure "torch_std_add_scalar" (void* double) void*))
(define torch_std_clip_tokenizer_create
  (foreign-procedure "torch_std_clip_tokenizer_create" (string string) void*))
(define torch_std_clip_tokenizer_encode
  (foreign-procedure "torch_std_clip_tokenizer_encode" (void* string) void*))
(define torch_std_clip_tokenizer_free
  (foreign-procedure "torch_std_clip_tokenizer_free" (void*) void))
(define torch_std_clip_text_forward
  (foreign-procedure "torch_std_clip_text_forward" (void* void* int) void*))
(define torch_std_sdxl_dual_clip
  (foreign-procedure "torch_std_sdxl_dual_clip" (void* void* void*) void*))
(define torch_std_jit_load
  (foreign-procedure "torch_std_jit_load" (string) void*))
(define torch_std_safetensors_load
  (foreign-procedure "torch_std_safetensors_load" (string) void*))
(define torch_std_safetensors_count
  (foreign-procedure "torch_std_safetensors_count" (void*) int))
(define torch_std_safetensors_name
  (foreign-procedure "torch_std_safetensors_name" (void* int) string))
(define torch_std_safetensors_tensor
  (foreign-procedure "torch_std_safetensors_tensor" (void* int) void*))
(define torch_std_safetensors_get_tensor_by_name
  (foreign-procedure "torch_std_safetensors_get_tensor_by_name" (void* string) void*))
(define torch_std_safetensors_free
  (foreign-procedure "torch_std_safetensors_free" (void*) void))
(define torch_std_sd15_unet_forward_dict
  (foreign-procedure "torch_std_sd15_unet_forward_dict" (void* void* void* void* void* void* void* int double) void*))
(define torch_std_sdxl_unet_forward
  (foreign-procedure "torch_std_sdxl_unet_forward" (void* void* void* void* void* double double double double double double) void*))
(define torch_std_vae_encode
  (foreign-procedure "torch_std_vae_encode" (void* void*) void*))
(define torch_std_vae_decode
  (foreign-procedure "torch_std_vae_decode" (void* void*) void*))
(define torch_std_vae_encode_tiled
  (foreign-procedure "torch_std_vae_encode_tiled" (void* void* int int) void*))
(define torch_std_vae_decode_tiled
  (foreign-procedure "torch_std_vae_decode_tiled" (void* void* int int) void*))
(define torch_std_sample_ddim
  (foreign-procedure "torch_std_sample_ddim" (void* void* void* void* double) void*))
(define torch_std_sample_ddim_from_sigma
  (foreign-procedure "torch_std_sample_ddim_from_sigma" (void* void* void* void* double) void*))
(define torch_std_sample_euler
  (foreign-procedure "torch_std_sample_euler" (void* void* void* void*) void*))
(define torch_std_sample_euler_ancestral
  (foreign-procedure "torch_std_sample_euler_ancestral" (void* void* void* void*) void*))
(define torch_std_sample_dpmpp_2m
  (foreign-procedure "torch_std_sample_dpmpp_2m" (void* void* void* void* void* int) void*))
(define torch_std_sampler_sigmas
  (foreign-procedure "torch_std_sampler_sigmas" (int double double string) void*))
(define torch_std_lora_match_to_unet
  (foreign-procedure "torch_std_lora_match_to_unet" (void* int void* void* void* int) int))
(define torch_std_t5_tokenizer_create
  (foreign-procedure "torch_std_t5_tokenizer_create" (string) void*))
(define torch_std_t5_tokenizer_encode
  (foreign-procedure "torch_std_t5_tokenizer_encode" (void* string int) void*))
(define torch_std_t5_tokenizer_free
  (foreign-procedure "torch_std_t5_tokenizer_free" (void*) void))
(define torch_std_flux_forward
  (foreign-procedure "torch_std_flux_forward" (void* void* void* void* void* double int int int int) void*))
(define torch_std_flux_embed_nd
  (foreign-procedure "torch_std_flux_embed_nd" (void* int double void* int) void*))
(define torch_std_fm_step
  (foreign-procedure "torch_std_fm_step" (void* void* double) void*))
(define torch_std_gguf_load
  (foreign-procedure "torch_std_gguf_load" (string) void*))
(define torch_std_gguf_tensor_count
  (foreign-procedure "torch_std_gguf_tensor_count" (void*) int))
(define torch_std_gguf_tensor_name
  (foreign-procedure "torch_std_gguf_tensor_name" (void* int) string))
(define torch_std_gguf_load_tensor
  (foreign-procedure "torch_std_gguf_load_tensor" (void* int) void*))
(define torch_std_gguf_load_tensor_by_name
  (foreign-procedure "torch_std_gguf_load_tensor_by_name" (void* string) void*))
(define torch_std_gguf_free
  (foreign-procedure "torch_std_gguf_free" (void*) void))
(define torch_std_image_resize
  (foreign-procedure "torch_std_image_resize" (void* int int string) void*))
(define torch_std_image_crop
  (foreign-procedure "torch_std_image_crop" (void* int int int int) void*))
(define torch_std_load_image
  (foreign-procedure "torch_std_load_image" (string) void*))
(define torch_std_ddpm_betas
  (foreign-procedure "torch_std_ddpm_betas" (int double double) void*))
(define torch_std_ddpm_add_noise
  (foreign-procedure "torch_std_ddpm_add_noise" (void* void* void* void*) void*))
(define torch_std_to_double_array
  (foreign-procedure "torch_std_to_double_array" (void* void* int) void))
(define torch_std_to_float_array
  (foreign-procedure "torch_std_to_float_array" (void* void* int) void))
(define torch_std_to_int64_array
  (foreign-procedure "torch_std_to_int64_array" (void* void* int) void))
(define torch_std_randint
  (foreign-procedure "torch_std_randint" (int int void* int int) void*))
(define torch_std_tensor_from_blob_3d
  (foreign-procedure "torch_std_tensor_from_blob_3d" (void* int int int int) void*))
(define torch_std_detach
  (foreign-procedure "torch_std_detach" (void*) void*))
(define torch_std_randn
  (foreign-procedure "torch_std_randn" (void* int int) void*))
(define torch_std_float_array_ptr
  (foreign-procedure "torch_std_float_array_ptr" (void*) void*))
(define torch_std_int_array_ptr
  (foreign-procedure "torch_std_int_array_ptr" (void*) void*))
(define torch_std_float_array_ptr_offset
  (foreign-procedure "torch_std_float_array_ptr_offset" (void* int) void*))
(define torch_std_int_array_ptr_offset
  (foreign-procedure "torch_std_int_array_ptr_offset" (void* int) void*))
(define torch_std_eq
  (foreign-procedure "torch_std_eq" (void* void*) void*))
(define torch_std_gt
  (foreign-procedure "torch_std_gt" (void* void*) void*))
(define torch_std_lt
  (foreign-procedure "torch_std_lt" (void* void*) void*))
(define torch_std_ge
  (foreign-procedure "torch_std_ge" (void* void*) void*))
(define torch_std_le
  (foreign-procedure "torch_std_le" (void* void*) void*))
(define torch_std_clamp
  (foreign-procedure "torch_std_clamp" (void* double double) void*))
(define torch_std_where
  (foreign-procedure "torch_std_where" (void* void* void*) void*))
(define torch_std_masked_select
  (foreign-procedure "torch_std_masked_select" (void* void*) void*))
(define torch_std_jit_forward
  (foreign-procedure "torch_std_jit_forward" (void* void*) void*))
(define torch_std_jit_module_delete
  (foreign-procedure "torch_std_jit_module_delete" (void*) void))
(define torch_std_jit_load_module
  (foreign-procedure "torch_std_jit_load_module" (string) void*))
(define torch_std_jit_named_parameters
  (foreign-procedure "torch_std_jit_named_parameters" (void* void* int) int))
(define torch_std_jit_parameters
  (foreign-procedure "torch_std_jit_parameters" (void* void* int) int))
(define torch_std_load_image_png
  (foreign-procedure "torch_std_load_image_png" (string) void*))
(define torch_std_save_image
  (foreign-procedure "torch_std_save_image" (void* string int) void))
(define torch_std_save_image_png
  (foreign-procedure "torch_std_save_image_png" (void* string) void))
(define torch_std_image_composite
  (foreign-procedure "torch_std_image_composite" (void* void* int int) void*))
(define torch_std_color_convert
  (foreign-procedure "torch_std_color_convert" (void* int int) void*))
(define torch_std_mse_loss
  (foreign-procedure "torch_std_mse_loss" (void* void* string) void*))
(define torch_std_l1_loss
  (foreign-procedure "torch_std_l1_loss" (void* string string) void*))
(define torch_std_cross_entropy_loss
  (foreign-procedure "torch_std_cross_entropy_loss" (void* void* string) void*))
(define torch_std_nll_loss
  (foreign-procedure "torch_std_nll_loss" (void* void* string) void*))
(define torch_std_bce_loss
  (foreign-procedure "torch_std_bce_loss" (void* void* string) void*))
(define torch_std_bce_with_logits_loss
  (foreign-procedure "torch_std_bce_with_logits_loss" (void* void* string) void*))
(define torch_std_copy_probs
  (foreign-procedure "torch_std_copy_probs" (void* double) void*))
(define torch_std_requires_grad
  (foreign-procedure "torch_std_requires_grad" (void*) int))
(define torch_std_set_requires_grad
  (foreign-procedure "torch_std_set_requires_grad" (void* int) void*))
(define torch_std_has_grad
  (foreign-procedure "torch_std_has_grad" (void*) int))
(define torch_std_grad
  (foreign-procedure "torch_std_grad" (void*) void*))
(define torch_std_backward
  (foreign-procedure "torch_std_backward" (void*) void))
(define torch_std_backward_retain_graph
  (foreign-procedure "torch_std_backward_retain_graph" (void*) void))
(define torch_std_zero_grad
  (foreign-procedure "torch_std_zero_grad" (void* int) void))
(define torch_std_clip_grad_norm
  (foreign-procedure "torch_std_clip_grad_norm" (void* int double) void))
(define torch_std_sgd_create
  (foreign-procedure "torch_std_sgd_create" (void* int double double double) void*))
(define torch_std_adam_create
  (foreign-procedure "torch_std_adam_create" (void* int double double double double double) void*))
(define torch_std_adamw_create
  (foreign-procedure "torch_std_adamw_create" (void* int double double double double double) void*))
(define torch_std_optimizer_step
  (foreign-procedure "torch_std_optimizer_step" (void*) void))
(define torch_std_optimizer_zero_grad
  (foreign-procedure "torch_std_optimizer_zero_grad" (void*) void))
(define torch_std_optimizer_destroy
  (foreign-procedure "torch_std_optimizer_destroy" (void*) void))
(define torch_std_controlnet_forward
  (foreign-procedure "torch_std_controlnet_forward" (void* int void* void* void* void* int) void*))
(define torch_std_controlnet_apply
  (foreign-procedure "torch_std_controlnet_apply" (void* void* double) void*))
(define torch_std_lora_apply
  (foreign-procedure "torch_std_lora_apply" (void* void* void* double) void*))
(define torch_std_lora_merge_into
  (foreign-procedure "torch_std_lora_merge_into" (void* void* string double) int))
(define torch_std_batch_norm1d
  (foreign-procedure "torch_std_batch_norm1d" (void* void* void* void* void* int double double) void*))
(define torch_std_sd_unet_forward
  (foreign-procedure "torch_std_sd_unet_forward" (void* void* void* void* void* void* int) void*))
(define torch_std_fm_sigmas
  (foreign-procedure "torch_std_fm_sigmas" (int double double) void*))
(define torch_std_save_state_dict
  (foreign-procedure "torch_std_save_state_dict" (void* string) void))
(define torch_std_sd_unet_forward_v2
  (foreign-procedure "torch_std_sd_unet_forward_v2" (void* int void* void* void* void* void* void* int double void* void* void* int void* double) void*))
(define torch_std_stable_cascade_stage_c
  (foreign-procedure "torch_std_stable_cascade_stage_c" (void* int void* void* void* void* void* void*) void*))
(define torch_std_pixart_forward
  (foreign-procedure "torch_std_pixart_forward" (void* int void* void* void* int int int) void*))
(define torch_std_hunyuan_video_forward
  (foreign-procedure "torch_std_hunyuan_video_forward" (void* void* void* void* int int int) void*))
(define torch_std_wan_video_forward
  (foreign-procedure "torch_std_wan_video_forward" (void* void* void* void* int int int) void*))
(define torch_std_cosmos_forward
  (foreign-procedure "torch_std_cosmos_forward" (void* void* void* void* int int int) void*))
(define torch_std_sd3_mmdit_forward
  (foreign-procedure "torch_std_sd3_mmdit_forward" (void* void* void* void* double) void*))

;; Source: <input>

;; StaticPy functions
(define DTYPE_FLOAT32 0)
(define DTYPE_FLOAT64 1)
(define DTYPE_INT32 2)
(define DTYPE_INT64 3)
(define (static_torch_std_silu a)
  (begin
    (set! s (torch_std_sigmoid a))
    (torch_std_mul a s)
  ))
(define (static_torch_std_gelu a)
  (begin
    (set! sqrt_2_over_pi 0.7978845608028654)
    (set! a_cubed (torch_std_pow a 3.0))
    (set! inner (torch_std_add a (torch_std_mul_scalar a_cubed 0.044715)))
    (set! inner (torch_std_mul_scalar inner sqrt_2_over_pi))
    (set! tanh_inner (torch_std_tanh inner))
    (set! one_plus_tanh (torch_std_add_scalar tanh_inner 1.0))
    (set! result (torch_std_mul (torch_std_mul_scalar a 0.5) one_plus_tanh))
    result
  ))
(define (static_timestep_embedding timesteps dim max_period)
  (begin
    (set! half (quotient dim 2))
    (set! idx (torch_std_arange 0 half 1 DTYPE_FLOAT32))
    (if (torch_std_is_cuda timesteps)
        (begin (set! idx (torch_std_to_cuda idx))))
    (set! neg_log (- (libm-log max_period)))
    (set! freqs (torch_std_mul_scalar idx (/ neg_log half)))
    (set! freqs (torch_std_exp freqs))
    (set! ts_2d (torch_std_unsqueeze timesteps 1))
    (set! freqs_2d (torch_std_unsqueeze freqs 0))
    (set! args (torch_std_mul ts_2d freqs_2d))
    (set! cos_args (torch_std_cos args))
    (set! sin_args (torch_std_sin args))
    (set! tensors (make_ptr_array 2))
    (ptr_array_set tensors 0 cos_args)
    (ptr_array_set tensors 1 sin_args)
    (set! embedding (torch_std_cat tensors 2 (- 1)))
    (if (equal? (mod dim 2) 1)
        (begin (set! first_col (torch_std_slice embedding 1 0 1 1)) (set! zeros (torch_std_mul_scalar first_col 0.0)) (set! tensors2 (make_ptr_array 2)) (ptr_array_set tensors2 0 embedding) (ptr_array_set tensors2 1 zeros) (set! embedding (torch_std_cat tensors2 2 (- 1)))))
    embedding
  ))
(define (static_mean_flat tensor)
  (begin
    (set! n (torch_std_numel tensor))
    (set! nd (torch_std_ndim tensor))
    (if (<= nd 1)
        (begin tensor))
    (set! shape_buf (make_int_array nd))
    (torch_std_shape tensor shape_buf)
    (set! batch (int_array_ref shape_buf 0))
    (set! new_shape (make_int_array 2))
    (int_array_set new_shape 0 batch)
    (int_array_set new_shape 1 (quotient n batch))
    (set! flat (torch_std_reshape tensor new_shape 2))
    (set! result (torch_std_mean_dim flat 1 0))
    result
  ))
(define (static_extract_into_tensor a t x_shape)
  (begin
    (set! result (torch_std_index_select a 0 t))
    (set! result (torch_std_unsqueeze result 1))
    (set! result (torch_std_unsqueeze result 1))
    (set! batch (torch_std_numel result))
    (set! shape_arr (make_int_array 4))
    (int_array_set shape_arr 0 batch)
    (int_array_set shape_arr 1 1)
    (int_array_set shape_arr 2 1)
    (int_array_set shape_arr 3 1)
    (set! result (torch_std_reshape result shape_arr 4))
    result
  ))
(define (static_sinusoidal_embedding dim max_period length)
  (begin
    (set! positions (torch_std_arange 0 length 1 DTYPE_FLOAT32))
    (set! positions (torch_std_unsqueeze positions 1))
    (set! half (quotient dim 2))
    (set! idx (torch_std_arange 0 half 1 DTYPE_FLOAT32))
    (set! freqs (torch_std_mul_scalar idx (/ (- (libm-log max_period)) half)))
    (set! freqs (torch_std_exp freqs))
    (set! freqs (torch_std_unsqueeze freqs 0))
    (set! args (torch_std_mul positions freqs))
    (set! cos_args (torch_std_cos args))
    (set! sin_args (torch_std_sin args))
    (set! tensors (make_ptr_array 2))
    (ptr_array_set tensors 0 cos_args)
    (ptr_array_set tensors 1 sin_args)
    (set! emb (torch_std_cat tensors 2 (- 1)))
    emb
  ))
(define (static_make_beta_schedule_linear n_timestep)
  (begin
    (set! linear_start 0.0001)
    (set! linear_end 0.02)
    (set! start_sqrt (libm-sqrt linear_start))
    (set! end_sqrt (libm-sqrt linear_end))
    (set! ar (torch_std_arange 0 n_timestep 1 DTYPE_FLOAT32))
    (set! step (/ (- end_sqrt start_sqrt) (static_max (- n_timestep 1) 1)))
    (set! betas (torch_std_mul_scalar ar step))
    (set! betas (torch_std_add_scalar betas start_sqrt))
    (set! betas (torch_std_pow betas 2.0))
    betas
  ))
(define (static_optimized_attention q k v heads mask skip_reshape)
  (torch_std_attention q k v heads mask skip_reshape))
(define (static_cross_attention q_w k_w v_w o_w o_b x context heads mask)
  (begin
    (set! q (torch_std_linear x q_w null))
    (set! k (torch_std_linear context k_w null))
    (set! v (torch_std_linear context v_w null))
    (set! attn_out (torch_std_attention q k v heads mask 0))
    (torch_std_linear attn_out o_w o_b)
  ))
(define (static_geglu_forward proj_w proj_b x)
  (begin
    (set! proj (torch_std_linear x proj_w proj_b))
    (set! ndim (torch_std_ndim proj))
    (set! shape_buf (make_int_array static_ndim))
    (torch_std_shape proj shape_buf)
    (set! last_dim (int_array_ref shape_buf (- static_ndim 1)))
    (set! half (quotient last_dim 2))
    (set! x_part (torch_std_slice proj (- static_ndim 1) 0 half 1))
    (set! gate (torch_std_slice proj (- static_ndim 1) half last_dim 1))
    (torch_std_mul x_part (static_torch_std_gelu gate))
  ))
(define (static_feed_forward in_w in_b out_w out_b x glu glu_w glu_b)
  (begin
    (if glu
        (begin (set! hidden (static_geglu_forward glu_w glu_b x)))
        (begin (set! hidden (static_torch_std_gelu (torch_std_linear x in_w in_b)))))
    (torch_std_linear hidden out_w out_b)
  ))
(define (static_embed_nd dim theta axes_dim n_axes ids)
  (torch_std_flux_embed_nd ids dim theta axes_dim n_axes))
(define (static_flux_timestep_embedding t dim max_period)
  (begin
    (set! t_scaled (torch_std_mul_scalar t 1000.0))
    (static_timestep_embedding t_scaled dim max_period)
  ))
(define CLIP_SOS_ID 49406)
(define CLIP_EOS_ID 49407)
(define CLIP_N_MAX 77)
(define (static_create_tokenizer vocab_path merges_path)
  (torch_std_clip_tokenizer_create vocab_path merges_path))
(define (static_tokenize tok text)
  (begin
    (set! ids_1d (torch_std_clip_tokenizer_encode tok text))
    (set! shape_arr (make_int_array 2))
    (int_array_set shape_arr 0 1)
    (int_array_set shape_arr 1 77)
    (set! ids_2d (torch_std_reshape ids_1d shape_arr 2))
    ids_2d
  ))
(define (static_tokenize_batch tok texts n_texts)
  (begin
    (set! first_text (ptr_array_ref texts 0))
    (static_tokenize tok first_text)
  ))
(define (static_free_tokenizer tok)
  (torch_std_clip_tokenizer_free tok))
(define (static_load_clip_model jit_path)
  (torch_std_jit_load jit_path))
(define (static_clip_encode clip_module token_ids use_f16)
  (torch_std_clip_text_forward clip_module token_ids use_f16))
(define (static_sdxl_dual_clip clip1 clip2 token_ids)
  (torch_std_sdxl_dual_clip clip1 clip2 token_ids))
(define (static_free_model module)
  (void))
(define STATE_CLIP_MODULE 0)
(define STATE_TOKENIZER 1)
(define STATE_EMBED_DIM 2)
(define STATE_MAX_LENGTH 3)
(define STATE_USE_F16 4)
(define STATE_N_FIELDS 5)
(define (static_sd1_clip_init clip_jit_path vocab_path merges_path embed_dim use_f16)
  (begin
    (set! state (make_int_array STATE_N_FIELDS))
    (set! clip_module (static_load_clip_model clip_jit_path))
    (set! tok (static_create_tokenizer vocab_path merges_path))
    (ptr_array_set state STATE_CLIP_MODULE clip_module)
    (ptr_array_set state STATE_TOKENIZER tok)
    (int_array_set state STATE_EMBED_DIM embed_dim)
    (int_array_set state STATE_MAX_LENGTH CLIP_N_MAX)
    (int_array_set state STATE_USE_F16 use_f16)
    state
  ))
(define (static_sd1_clip_encode state text)
  (begin
    (set! clip_module (ptr_array_ref state STATE_CLIP_MODULE))
    (set! tok (ptr_array_ref state STATE_TOKENIZER))
    (set! use_f16 (int_array_ref state STATE_USE_F16))
    (set! token_ids (static_tokenize tok text))
    (set! seq_emb (static_clip_encode clip_module token_ids use_f16))
    (set! pooled (torch_std_slice seq_emb 1 76 77 1))
    (set! pooled (torch_std_squeeze pooled 1))
    (set! result (make_ptr_array 2))
    (ptr_array_set result 0 seq_emb)
    (ptr_array_set result 1 pooled)
    result
  ))
(define (static_sd1_clip_free state)
  (begin
    (set! tok (ptr_array_ref state STATE_TOKENIZER))
    (static_free_tokenizer tok)
  ))
(define SDXL_STATE_CLIP_L 0)
(define SDXL_STATE_CLIP_G 1)
(define SDXL_STATE_TOKENIZER 2)
(define SDXL_STATE_USE_F16 3)
(define SDXL_STATE_EMBED_DIM_L 4)
(define SDXL_STATE_EMBED_DIM_G 5)
(define SDXL_STATE_N_FIELDS 6)
(define (static_sdxl_clip_init clip_l_path clip_g_path vocab_path merges_path use_f16)
  (begin
    (set! state (make_int_array SDXL_STATE_N_FIELDS))
    (set! clip_l (static_load_clip_model clip_l_path))
    (set! clip_g (static_load_clip_model clip_g_path))
    (set! tok (static_create_tokenizer vocab_path merges_path))
    (ptr_array_set state SDXL_STATE_CLIP_L clip_l)
    (ptr_array_set state SDXL_STATE_CLIP_G clip_g)
    (ptr_array_set state SDXL_STATE_TOKENIZER tok)
    (int_array_set state SDXL_STATE_USE_F16 use_f16)
    (int_array_set state SDXL_STATE_EMBED_DIM_L 768)
    (int_array_set state SDXL_STATE_EMBED_DIM_G 1280)
    state
  ))
(define (static_sdxl_clip_encode state text)
  (begin
    (set! clip_l (ptr_array_ref state SDXL_STATE_CLIP_L))
    (set! clip_g (ptr_array_ref state SDXL_STATE_CLIP_G))
    (set! tok (ptr_array_ref state SDXL_STATE_TOKENIZER))
    (set! token_ids (static_tokenize tok text))
    (set! result_struct (static_sdxl_dual_clip clip_l clip_g token_ids))
    result_struct
  ))
(define (static_sdxl_clip_free state)
  (begin
    (set! tok (ptr_array_ref state SDXL_STATE_TOKENIZER))
    (static_free_tokenizer tok)
  ))
(define SD15_IN_CHANNELS 4)
(define SD15_MODEL_CHANNELS 320)
(define SD15_OUT_CHANNELS 4)
(define SDXL_IN_CHANNELS 4)
(define SDXL_MODEL_CHANNELS 320)
(define SDXL_OUT_CHANNELS 4)
(define (static_sd15_unet_forward sd_dict latent timestep text_emb)
  (torch_std_sd15_unet_forward_dict sd_dict latent timestep text_emb null null null 0 0.0))
(define (static_sdxl_unet_forward sd_dict latent timestep text_emb pooled_emb original_size_h original_size_w crop_top crop_left target_size_h target_size_w)
  (torch_std_sdxl_unet_forward sd_dict latent timestep text_emb pooled_emb original_size_h original_size_w crop_top crop_left target_size_h target_size_w))
;; VAE init from __init__
(define (VAE_init self encode_pt_path decode_pt_path)
  (begin
    (set! VAE_encode_module (torch_std_jit_load encode_pt_path))
    (set! VAE_decode_module (torch_std_jit_load decode_pt_path))
  ))
(define (VAE_encode self image)
  (torch_std_vae_encode VAE_encode_module image))
(define (VAE_decode self latent)
  (torch_std_vae_decode VAE_decode_module latent))
(define (VAE_free self)
  (begin
    (torch_std_jit_module_delete VAE_encode_module)
    (torch_std_jit_module_delete VAE_decode_module)
  ))
(define (static_VAE encode_pt_path decode_pt_path)
  (VAE_init 'VAE encode_pt_path decode_pt_path)
  'VAE)
;; TiledVAE init from __init__
(define (TiledVAE_init self encode_pt_path decode_pt_path)
  (begin
    (set! TiledVAE_encode_module (torch_std_jit_load encode_pt_path))
    (set! TiledVAE_decode_module (torch_std_jit_load decode_pt_path))
  ))
(define (TiledVAE_encode self image tile_size overlap)
  (torch_std_vae_encode_tiled TiledVAE_encode_module image tile_size overlap))
(define (TiledVAE_decode self latent tile_size overlap)
  (torch_std_vae_decode_tiled TiledVAE_decode_module latent tile_size overlap))
(define (TiledVAE_free self)
  (begin
    (torch_std_jit_module_delete TiledVAE_encode_module)
    (torch_std_jit_module_delete TiledVAE_decode_module)
  ))
(define (static_TiledVAE encode_pt_path decode_pt_path)
  (TiledVAE_init 'TiledVAE encode_pt_path decode_pt_path)
  'TiledVAE)
(define (static_get_sigmas_karras steps sigma_min sigma_max)
  (torch_std_sampler_sigmas steps sigma_min sigma_max "karras"))
(define (static_get_sigmas_exponential steps sigma_min sigma_max)
  (torch_std_sampler_sigmas steps sigma_min sigma_max "exponential"))
(define (static_get_sigmas_linear steps sigma_min sigma_max)
  (torch_std_sampler_sigmas steps sigma_min sigma_max "linear"))
(define (static_get_sigmas_ddim steps sigma_min sigma_max)
  (begin
    (set! sigmas (torch_std_sampler_sigmas steps sigma_min sigma_max "linear"))
    sigmas
  ))
(define (static_tensor_scalar t)
  (begin
    (set! out (make_float_array 1))
    (torch_std_to_float_array t out 1)
    (float_array_ref out 0)
  ))
(define (static_to_alpha_bar sigma)
  (/ 1.0 (+ 1.0 (* sigma sigma))))
(define (static_cfg_predict unet_fn x sigma text_emb_cond text_emb_uncond cfg_scale)
  (begin
    (set! cond_eps (unet_fn x sigma text_emb_cond))
    (set! uncond_eps (unet_fn x sigma text_emb_uncond))
    (set! eps_diff (torch_std_sub cond_eps uncond_eps))
    (set! eps_scaled (torch_std_mul_scalar eps_diff cfg_scale))
    (set! eps_cfg (torch_std_add uncond_eps eps_scaled))
    (set! sigma_eps (torch_std_mul sigma eps_cfg))
    (set! denoised (torch_std_sub x sigma_eps))
    denoised
  ))
(define (static_sample_euler unet_fn x sigmas text_emb_cond text_emb_uncond cfg_scale)
  (begin
    (set! n_steps (- (torch_std_size sigmas 0) 1))
    (do ((i 0 (+ i 1))) ((>= i n_steps)) (set! sigma_t (torch_std_narrow sigmas 0 i 1)) (set! sigma_next (torch_std_narrow sigmas 0 (+ i 1) 1)) (set! noise_pred (static_cfg_predict unet_fn x sigma_t text_emb_cond text_emb_uncond cfg_scale)) (set! x (torch_std_sample_euler noise_pred x sigma_t sigma_next)))
    x
  ))
(define (static_sample_euler_ancestral unet_fn x sigmas text_emb_cond text_emb_uncond cfg_scale)
  (begin
    (set! n_steps (- (torch_std_size sigmas 0) 1))
    (do ((i 0 (+ i 1))) ((>= i n_steps)) (set! sigma_t (torch_std_narrow sigmas 0 i 1)) (set! sigma_next (torch_std_narrow sigmas 0 (+ i 1) 1)) (set! noise_pred (static_cfg_predict unet_fn x sigma_t text_emb_cond text_emb_uncond cfg_scale)) (set! x (torch_std_sample_euler_ancestral noise_pred x sigma_t sigma_next)))
    x
  ))
(define (static_sample_dpmpp_2m unet_fn x sigmas text_emb_cond text_emb_uncond cfg_scale)
  (begin
    (set! n_steps (- (torch_std_size sigmas 0) 1))
    (set! old_denoised null)
    (do ((i 0 (+ i 1))) ((>= i n_steps)) (set! sigma_t (torch_std_narrow sigmas 0 i 1)) (set! sigma_next (torch_std_narrow sigmas 0 (+ i 1) 1)) (set! noise_pred (static_cfg_predict unet_fn x sigma_t text_emb_cond text_emb_uncond cfg_scale)) (set! result (torch_std_sample_dpmpp_2m noise_pred x sigma_t sigma_next old_denoised (if (equal? old_denoised null) 1 0))) (set! x (torch_std_narrow result 0 0 1)) (set! old_denoised (torch_std_narrow result 0 1 1)))
    x
  ))
(define (static_sample_ddim unet_fn x sigmas text_emb_cond text_emb_uncond cfg_scale eta)
  (begin
    (set! n_steps (- (torch_std_size sigmas 0) 1))
    (do ((i 0 (+ i 1))) ((>= i n_steps)) (set! sigma_t (torch_std_narrow sigmas 0 i 1)) (set! sigma_next (torch_std_narrow sigmas 0 (+ i 1) 1)) (set! noise_pred (static_cfg_predict unet_fn x sigma_t text_emb_cond text_emb_uncond cfg_scale)) (set! x (torch_std_sample_ddim_from_sigma noise_pred x sigma_t sigma_next eta)))
    x
  ))
(define (static_sample_dpmpp_sde unet_fn x sigmas text_emb_cond text_emb_uncond cfg_scale eta)
  (begin
    (set! n_steps (- (torch_std_size sigmas 0) 1))
    (set! shape (make_int_array 4))
    (int_array_set shape 0 (torch_std_size x 0))
    (int_array_set shape 1 (torch_std_size x 1))
    (int_array_set shape 2 (torch_std_size x 2))
    (int_array_set shape 3 (torch_std_size x 3))
    (do ((i 0 (+ i 1))) ((>= i n_steps)) (set! sigma_t (torch_std_narrow sigmas 0 i 1)) (set! sigma_next (torch_std_narrow sigmas 0 (+ i 1) 1)) (set! noise_pred (static_cfg_predict unet_fn x sigma_t text_emb_cond text_emb_uncond cfg_scale)) (set! x_euler (torch_std_sample_euler noise_pred x sigma_t sigma_next)) (set! noise (torch_std_randn shape 4 0)) (set! sigma_t_val (static_tensor_scalar sigma_t)) (set! sigma_n_val (static_tensor_scalar sigma_next)) (set! noise_scale (* (libm-sqrt (- (* sigma_t_val sigma_t_val) (* sigma_n_val sigma_n_val))) eta)) (set! x (torch_std_add x_euler (torch_std_mul_scalar noise noise_scale))))
    x
  ))
(define (static_sigma_to_t sigma sigma_data)
  (* 0.5 (libm-log (* 0.5 (+ (* sigma sigma) (* sigma_data sigma_data))))))
(define (static_deis_coeffs t_vals order)
  (begin
    (set! coeffs (vector ))
    (do ((j 0 (+ j 1))) ((>= j order)) (if (equal? order 1)
        (begin (coeffs append 1.0))
        (begin (if (equal? order 2)
        (begin (set! dt (- (float_array_ref t_vals 0) (float_array_ref t_vals 1))) (if (equal? j 0)
        (begin (coeffs append (if (not (equal? dt 0.0)) 2.0 1.0)))
        (begin (coeffs append (if (not (equal? dt 0.0)) (- 1.0) 0.0)))))
        (begin (coeffs append (if (equal? j 0) 1.0 0.0)))))))
    coeffs
  ))
(define (static_sample_deis unet_fn x sigmas text_emb_cond text_emb_uncond cfg_scale order)
  (begin
    (set! n_steps (- (torch_std_size sigmas 0) 1))
    (set! denoised_history null)
    (set! n_history (static_min (- order 1) n_steps))
    (do ((i 0 (+ i 1))) ((>= i n_steps)) (set! sigma_t (torch_std_narrow sigmas 0 i 1)) (set! sigma_next (torch_std_narrow sigmas 0 (+ i 1) 1)) (set! noise_pred (static_cfg_predict unet_fn x sigma_t text_emb_cond text_emb_uncond cfg_scale)) (if (or (equal? i 0) (equal? order 1))
        (begin (set! x (torch_std_sample_euler noise_pred x sigma_t sigma_next)))
        (begin (set! x_euler (torch_std_sample_euler noise_pred x sigma_t sigma_next)) (if (not (equal? denoised_history null))
        (begin (set! h (- (static_tensor_scalar sigma_t) (static_tensor_scalar sigma_next))) (set! x_correction (torch_std_mul_scalar (torch_std_sub x_euler denoised_history) (* 0.5 h))) (set! x (torch_std_add x_euler x_correction)))
        (begin (set! x x_euler))))) (set! denoised_history noise_pred))
    x
  ))
(define (static_sample_sa_solver unet_fn x sigmas text_emb_cond text_emb_uncond cfg_scale corrector_steps)
  (begin
    (set! n_steps (- (torch_std_size sigmas 0) 1))
    (do ((i 0 (+ i 1))) ((>= i n_steps)) (set! sigma_t (torch_std_narrow sigmas 0 i 1)) (set! sigma_next (torch_std_narrow sigmas 0 (+ i 1) 1)) (set! noise_pred (static_cfg_predict unet_fn x sigma_t text_emb_cond text_emb_uncond cfg_scale)) (set! x_pred (torch_std_sample_euler noise_pred x sigma_t sigma_next)) (if (> corrector_steps 0)
        (begin (set! x_c x_pred) (do ((c 0 (+ c 1))) ((>= c corrector_steps)) (set! noise_pred_c (static_cfg_predict unet_fn x_c sigma_next text_emb_cond text_emb_uncond cfg_scale)) (set! x_c_fwd (torch_std_sample_euler noise_pred_c x_c sigma_next sigma_t)) (set! x_c x_pred)) (set! x x_c))
        (begin (set! x x_pred))))
    x
  ))
(define (static_sample_flow_match unet_fn x sigmas text_emb_cond text_emb_uncond cfg_scale)
  (begin
    (set! n_steps (- (torch_std_size sigmas 0) 1))
    (set! dt (/ 1.0 (inexact n_steps)))
    (do ((i 0 (+ i 1))) ((>= i n_steps)) (set! sigma_t (torch_std_narrow sigmas 0 i 1)) (if (or (<= cfg_scale 1.0) (equal? text_emb_uncond null))
        (begin (set! velocity (unet_fn x sigma_t text_emb_cond)))
        (begin (set! v_cond (unet_fn x sigma_t text_emb_cond)) (set! v_uncond (unet_fn x sigma_t text_emb_uncond)) (set! v_diff (torch_std_sub v_cond v_uncond)) (set! v_scaled (torch_std_mul_scalar v_diff cfg_scale)) (set! velocity (torch_std_add v_uncond v_scaled)))) (set! x (torch_std_fm_step velocity x dt)))
    x
  ))
(define (static_t5_init tokenizer_path jit_model_path)
  (begin
    (set! _t5_tok (torch_std_t5_tokenizer_create tokenizer_path))
    (set! _t5_jit (torch_std_jit_load jit_model_path))
  ))
(define (static_t5_encode text max_len)
  (begin
    (set! tokens (torch_std_t5_tokenizer_encode _t5_tok text max_len))
    (torch_std_jit_forward _t5_jit tokens)
  ))
(define (static_t5_tokenize text max_len)
  (torch_std_t5_tokenizer_encode _t5_tok text max_len))
(define (static_t5_free )
  (begin
    (torch_std_t5_tokenizer_free _t5_tok)
    (torch_std_jit_module_delete _t5_jit)
  ))
(define (static_lora_init n_unet_weights max_lora)
  (begin
    (set! _n_weights n_unet_weights)
    (set! _max_lora max_lora)
    (set! _lora_indices (make_int_array max_lora))
    (set! _lora_A (make_ptr_array max_lora))
    (set! _lora_B (make_ptr_array max_lora))
    (set! _lora_count 0)
  ))
(define (static_lora_load lora_path)
  (set! _lora_dict (torch_std_safetensors_load lora_path)))
(define (static_lora_load_from_dict lora_dict)
  (set! _lora_dict lora_dict))
(define (static_lora_match )
  (begin
    (set! _lora_count (torch_std_lora_match_to_unet _lora_dict _n_weights _lora_indices _lora_A _lora_B _max_lora))
    _lora_count
  ))
(define (static_lora_get_indices )
  _lora_indices)
(define (static_lora_get_A )
  _lora_A)
(define (static_lora_get_B )
  _lora_B)
(define (static_lora_get_count )
  _lora_count)
(define (static_lora_free )
  (torch_std_safetensors_free _lora_dict))
(define FLUX_LATENT_CHANNELS 16)
(define FLUX_N_BLOCKS 24)
(define FLUX_N_HEADS_IMG 24)
(define FLUX_N_HEADS_TXT 16)
(define FLUX_HEAD_DIM 128)
(define (static_flux_init wdict)
  (set! _flux_dict wdict))
(define (static_flux_txt2img t5_encode_fn prompt steps guidance height width seed vae_decode_fn)
  (begin
    (set! txt_emb (t5_encode_fn prompt 512))
    (torch_std_manual_seed seed)
    (set! latent_h (quotient height 16))
    (set! latent_w (quotient width 16))
    (set! shape (make_int_array 4))
    (int_array_set shape 0 1)
    (int_array_set shape 1 FLUX_LATENT_CHANNELS)
    (int_array_set shape 2 latent_h)
    (int_array_set shape 3 latent_w)
    (set! x (torch_std_randn shape 4 0))
    (set! dt (/ 1.0 (inexact steps)))
    (do ((i 0 (+ i 1))) ((>= i steps)) (set! t (* (inexact i) dt)) (set! scalar_shape (make_int_array 1)) (int_array_set scalar_shape 0 1) (set! t_tensor (torch_std_full scalar_shape 1 t 0)) (set! velocity (torch_std_flux_forward _flux_dict x txt_emb t_tensor null guidance FLUX_N_BLOCKS FLUX_N_HEADS_IMG FLUX_N_HEADS_TXT FLUX_HEAD_DIM)) (set! x (torch_std_fm_step velocity x dt)))
    (vae_decode_fn x)
  ))
(define (static_flux_free )
  (void))
(define (static_controlnet_load safetensors_path)
  (begin
    (set! dict_ptr (torch_std_safetensors_load safetensors_path))
    (set! _cn_n_weights (torch_std_safetensors_count dict_ptr))
    (set! _cn_weights (make_ptr_array _cn_n_weights))
    (do ((i 0 (+ i 1))) ((>= i _cn_n_weights)) (ptr_array_set _cn_weights i (torch_std_safetensors_tensor dict_ptr i)))
  ))
(define (static_controlnet_forward x timestep text_emb hint num_hint_channels)
  (torch_std_controlnet_forward _cn_weights _cn_n_weights x timestep text_emb hint num_hint_channels))
(define (static_controlnet_apply unet_output control_features strength)
  (torch_std_controlnet_apply unet_output control_features strength))
(define (static_encode_token_embedding clip_module tokens)
  (torch_std_clip_text_forward clip_module tokens 0))
(define (static_pad_tokens tokens target_len pad_token_id)
  (begin
    (set! cur_len (exact (truncate (static_tensor_size tokens 1))))
    (if (>= cur_len target_len)
        (begin (torch_std_narrow tokens 1 0 target_len)))
    (set! pad_len (- target_len cur_len))
    (set! shape (make_int_array 2))
    (int_array_set shape 0 1)
    (int_array_set shape 1 pad_len)
    (set! pad (torch_std_full shape 2 (inexact pad_token_id) 2))
    (set! tensors (make_ptr_array 2))
    (ptr_array_set tensors 0 tokens)
    (ptr_array_set tensors 1 pad)
    (torch_std_cat tensors 2 1)
  ))
(define (static_extract_pooled emb)
  (torch_std_narrow emb 1 76 1))
(define (static_mean_pool emb mask)
  (begin
    (if (not (equal? mask null))
        (begin (set! m (torch_std_unsqueeze mask 2)) (set! masked (torch_std_mul emb m)) (set! summed (torch_std_sum_dim masked 1 0)) (set! msum (torch_std_sum_dim mask 1 0)) (set! msum (torch_std_clamp msum 1.0 10000000000.0)) (torch_std_div summed msum)))
    (torch_std_mean_dim emb 1 0)
  ))
(define MODEL_TYPE_UNKNOWN 0)
(define MODEL_TYPE_SD15 1)
(define MODEL_TYPE_SDXL 2)
(define MODEL_TYPE_SD3 3)
(define MODEL_TYPE_FLUX 4)
(define (static_load_image path)
  (torch_std_load_image path))
(define (static_save_image tensor path)
  (torch_std_save_image tensor path 0))
(define (static_load_image_png path)
  (torch_std_load_image_png path))
(define (static_save_image_png tensor path)
  (torch_std_save_image_png tensor path))
(define (static_is_cuda_available )
  (torch_std_cuda_is_available ))
(define (static_to_cuda t)
  (torch_std_to_cuda t))
(define (static_to_cpu t)
  (torch_std_to_cpu t))
(define (static_is_cuda t)
  (torch_std_is_cuda t))
(define (static_to_float32 t)
  (torch_std_to_dtype t 0))
(define (static_to_float64 t)
  (torch_std_to_dtype t 1))
(define (static_detect_model_type sd_dict)
  (begin
    (set! n (torch_std_safetensors_count sd_dict))
    (set! has_sd3 0)
    (set! has_flux 0)
    (set! has_xl 0)
    (set! has_double 0)
    (set! has_single 0)
    (do ((i 0 (+ i 1))) ((>= i n)) (call/cc (lambda (continue) (set! name (torch_std_safetensors_name sd_dict i)) (if (equal? name null)
        (begin (continue))) (if (str_contains name "double_blocks")
        (begin (set! has_double 1))) (if (str_contains name "single_blocks")
        (begin (set! has_single 1))) (if (or (str_contains name "flux") (str_contains name "txt_in"))
        (begin (set! has_flux 1))) (if (str_contains name "label_emb")
        (begin (set! has_xl 1))) (if (str_contains name "x_embedder")
        (begin (set! has_sd3 1))))))
    (cond
      ((or (not (zero? has_flux)) (and (not (zero? has_double)) (not (zero? has_single)))) MODEL_TYPE_FLUX)
      ((or (not (zero? has_sd3)) (not (zero? has_double))) MODEL_TYPE_SD3)
      ((not (zero? has_xl)) MODEL_TYPE_SDXL)
      (else MODEL_TYPE_SD15))
  ))
(define (static_manual_seed seed)
  (torch_std_manual_seed seed))
(define (static_numel t)
  (torch_std_numel t))
(define (static_ndim t)
  (torch_std_ndim t))
(define (static_tensor_shape t)
  (begin
    (set! n (static_ndim t))
    (set! shape (make_int_array n))
    (torch_std_shape t shape)
    shape
  ))
(define (static_tensor_size t dim)
  (torch_std_size t dim))
(define (static_tensor_scalar t)
  (begin
    (set! out (make_float_array 1))
    (torch_std_to_float_array t out 1)
    (float_array_ref out 0)
  ))
(define (static_lora_merge_into_model model_dict lora_dict prefix scale)
  (torch_std_lora_merge_into model_dict lora_dict prefix scale))
(define (static_make_scalar_tensor value)
  (begin
    (set! shape (make_int_array 1))
    (int_array_set shape 0 1)
    (torch_std_full shape 1 value 0)
  ))
(define (static_to_alpha_bar sigma)
  (/ 1.0 (+ 1.0 (* sigma sigma))))
(define VISION_INPUT_SIZE 224)
(define VISION_MEAN 0.48145466)
(define VISION_STD 0.26862954)
(define (static_clip_vision_init jit_model_path)
  (set! _cv_jit_module (torch_std_jit_load jit_model_path)))
(define (static_clip_vision_preprocess image)
  (begin
    (set! resized (torch_std_image_resize image VISION_INPUT_SIZE VISION_INPUT_SIZE "bilinear"))
    (set! mean_shape (make_int_array 4))
    (int_array_set mean_shape 0 1)
    (int_array_set mean_shape 1 3)
    (int_array_set mean_shape 2 1)
    (int_array_set mean_shape 3 1)
    (set! mean_t (torch_std_full mean_shape 4 VISION_MEAN 0))
    (set! std_t (torch_std_full mean_shape 4 VISION_STD 0))
    (set! normalized (torch_std_div (torch_std_sub resized mean_t) std_t))
    normalized
  ))
(define (static_clip_vision_encode image)
  (begin
    (set! processed (static_clip_vision_preprocess image))
    (set! output (torch_std_jit_forward _cv_jit_module processed))
    (set! result (make_ptr_array 2))
    (set! nd (torch_std_ndim output))
    (if (equal? nd 3)
        (begin (set! pooled (torch_std_narrow output 1 0 1)) (ptr_array_set result 0 pooled) (ptr_array_set result 1 output))
        (begin (ptr_array_set result 0 output) (ptr_array_set result 1 null)))
    result
  ))
(define (static_clip_vision_free )
  (torch_std_jit_module_delete _cv_jit_module))
(define (static_core_init )
  (begin
    (set! _device_cuda (torch_std_cuda_is_available ))
    (set! _current_device 0)
  ))
(define (static_is_cuda_available )
  (torch_std_cuda_is_available ))
(define (static_set_device cuda)
  (set! _device_cuda cuda))
(define (static_get_device )
  _device_cuda)
(define (static_to_device t)
  (begin
    (if _device_cuda
        (begin (torch_std_to_cuda t)))
    t
  ))
(define (static_to_cpu t)
  (torch_std_to_cpu t))
(define (static_to_cuda t)
  (torch_std_to_cuda t))
(define MODEL_UNKNOWN 0)
(define MODEL_SD15 1)
(define MODEL_SDXL 2)
(define MODEL_FLUX 3)
(define MODEL_T5 4)
(define MODEL_CLIP 5)
(define MODEL_VAE 6)
(define MODEL_CONTROLNET 7)
(define MODEL_LORA 8)
(define _MAX_LOADED 32)
(define (static_model_registry_init )
  (begin
    (set! _loaded_models (make_ptr_array _MAX_LOADED))
    (set! _loaded_types (make_int_array _MAX_LOADED))
    (set! _n_loaded 0)
  ))
(define (static_model_load_jit path model_type)
  (begin
    (set! handle (torch_std_jit_load path))
    (set! idx _n_loaded)
    (ptr_array_set _loaded_models idx handle)
    (int_array_set _loaded_types idx model_type)
    (set! _n_loaded (+ _n_loaded 1))
    idx
  ))
(define (static_model_load_safetensors path model_type)
  (begin
    (set! handle (torch_std_safetensors_load path))
    (set! idx _n_loaded)
    (ptr_array_set _loaded_models idx handle)
    (int_array_set _loaded_types idx model_type)
    (set! _n_loaded (+ _n_loaded 1))
    idx
  ))
(define (static_model_get idx)
  (ptr_array_ref _loaded_models idx))
(define (static_model_get_type idx)
  (int_array_ref _loaded_types idx))
(define (static_model_unload idx)
  (begin
    (set! t (int_array_ref _loaded_types idx))
    (set! h (ptr_array_ref _loaded_models idx))
    (if (or (equal? t MODEL_SD15) (equal? t MODEL_SDXL) (equal? t MODEL_FLUX))
        (begin (torch_std_delete_tensor h))
        (begin (if (equal? t MODEL_CLIP)
        (begin (torch_std_jit_module_delete h))
        (begin (if (equal? t MODEL_VAE)
        (begin (torch_std_jit_module_delete h))
        (begin (if (equal? t MODEL_LORA)
        (begin (torch_std_safetensors_free h)))))))))
    (ptr_array_set _loaded_models idx null)
    (int_array_set _loaded_types idx 0)
  ))
(define (static_model_unload_all )
  (begin
    (do ((i 0 (+ i 1))) ((>= i _n_loaded)) (static_model_unload i))
    (set! _n_loaded 0)
  ))
(define (static_memory_empty_cache )
  (void))
(define (static_memory_get_free )
  0)
(define _MAX_PATCHES 64)
(define (static_patcher_init )
  (begin
    (set! _patch_targets (make_ptr_array _MAX_PATCHES))
    (set! _patch_scales (make_float_array _MAX_PATCHES))
    (set! _n_patches 0)
  ))
(define (static_patcher_add target scale)
  (begin
    (ptr_array_set _patch_targets _n_patches target)
    (float_array_set _patch_scales _n_patches scale)
    (set! _n_patches (+ _n_patches 1))
  ))
(define (static_patcher_apply )
  (begin
    (do ((i 0 (+ i 1))) ((>= i _n_patches)) (set! t (ptr_array_ref _patch_targets i)) (set! s (float_array_ref _patch_scales i)) (torch_std_mul_scalar t s))
    (set! _n_patches 0)
  ))
(define (static_patcher_clear )
  (set! _n_patches 0))
(define (static_context_set window_size stride)
  (begin
    (set! _ctx_window_size window_size)
    (set! _ctx_stride stride)
  ))
(define (static_context_get_window )
  _ctx_window_size)
(define (static_context_get_stride )
  _ctx_stride)
(define (static_models_init )
  (begin
    (set! sd15 (make_dict ))
    (dict_set sd15 "sd_v1.5" "sd_v1.5")
    (dict_set sd15 "sd_v1.4" "sd_v1.4")
    (set! SD15_CONFIGS sd15)
    (set! sdxl (make_dict ))
    (dict_set sdxl "sdxl_v1.0" "sdxl_v1.0")
    (set! SDXL_CONFIGS sdxl)
    (set! flux_cfg (make_dict ))
    (dict_set flux_cfg "flux_schnell" "flux_schnell")
    (dict_set flux_cfg "flux_dev" "flux_dev")
    (set! FLUX_CONFIGS flux_cfg)
  ))
(define (static_detect_model_config dict_ptr)
  (begin
    (set! mt (static_detect_model_type dict_ptr))
    (if (equal? mt MODEL_TYPE_FLUX)
        (begin "flux")
        (begin (if (equal? mt MODEL_TYPE_SDXL)
        (begin "sdxl")
        (begin (if (equal? mt MODEL_TYPE_SD15)
        (begin "sd15"))))))
    "unknown"
  ))
(define (static_sd_generate prompt model_path clip_path vae_path steps cfg height width seed sampler scheduler lora_path vocab_path merges_path clip_l_path clip_g_path)
  (begin
    (static_core_init )
    (static_models_init )
    (set! sd_dict (torch_std_safetensors_load model_path))
    (set! mt (static_detect_model_type sd_dict))
    (set! vae_module (torch_std_jit_load vae_path))
    (if (equal? mt MODEL_TYPE_SDXL)
        (begin (if (equal? clip_l_path "")
        (begin (set! clip_l_path clip_path))) (set! clip_state (static_sdxl_clip_init clip_l_path clip_g_path vocab_path merges_path 0)) (set! unet_fn (static_make_sdxl_unet_fn sd_dict height width 0 0 height width)) (set! pipeline (static_SDXLPipeline sd_dict clip_state vae_module unet_fn height width 0 0 height width)))
        (begin (set! clip_state (static_sd1_clip_init clip_path vocab_path merges_path 768 0)) (set! unet_fn (static_make_sd15_unet_fn sd_dict)) (set! pipeline (static_SD15Pipeline sd_dict clip_state vae_module unet_fn))))
    (if (and (not (equal? lora_path "")) (equal? mt MODEL_TYPE_SD15))
        (begin (set! lora_dict (torch_std_safetensors_load lora_path)) (set! lora_num (torch_std_lora_match_to_unet lora_dict 276 null null null 64)) (SD15Pipeline_load_lora pipeline lora_dict lora_num 64)))
    (if (equal? mt MODEL_TYPE_SDXL)
        (set! image (SDXLPipeline_txt2img pipeline prompt steps cfg sampler scheduler height width seed 1.0 0.03 14.6))
        (set! image (SD15Pipeline_txt2img pipeline prompt steps cfg sampler scheduler height width seed 1.0 0.03 14.6)))
    image
  ))
(define (static_main )
  (begin
    (set! prompt (os_getenv "PROMPT"))
    (if (equal? prompt null)
        (begin (set! prompt "a cute cat")))
    (set! model_path (os_getenv "MODEL"))
    (set! clip_path (os_getenv "CLIP"))
    (set! vae_path (os_getenv "VAE"))
    (if (equal? model_path null)
        (begin (exit_program 1)))
    (set! steps_str (os_getenv "STEPS"))
    (set! steps 20)
    (if (not (equal? steps_str null))
        (begin (set! steps (string_to_int steps_str))))
    (set! cfg_str (os_getenv "CFG"))
    (set! cfg 7.0)
    (if (not (equal? cfg_str null))
        (begin (set! cfg (string_to_float cfg_str))))
    (set! seed_str (os_getenv "SEED"))
    (set! seed 42)
    (if (not (equal? seed_str null))
        (begin (set! seed (string_to_int seed_str))))
    (set! vocab_path (os_getenv "VOCAB"))
    (if (equal? vocab_path null)
        (begin (set! vocab_path "")))
    (set! merges_path (os_getenv "MERGES"))
    (if (equal? merges_path null)
        (begin (set! merges_path "")))
    (set! clip_l_path (os_getenv "CLIP_L"))
    (if (equal? clip_l_path null)
        (begin (set! clip_l_path clip_path)))
    (set! clip_g_path (os_getenv "CLIP_G"))
    (if (equal? clip_g_path null)
        (begin (set! clip_g_path "")))
    (set! image (static_sd_generate prompt model_path clip_path vae_path steps cfg 512 512 seed "euler" "karras" "" vocab_path merges_path clip_l_path clip_g_path))
    (set! output_path (os_getenv "OUTPUT"))
    (if (equal? output_path null)
        (begin (set! output_path "output.png")))
    (static_save_image image output_path)
  ))
(define (static_sigmas_karras steps sigma_min sigma_max)
  (static_get_sigmas_karras steps sigma_min sigma_max))
(define (static_sigmas_exponential steps sigma_min sigma_max)
  (static_get_sigmas_exponential steps sigma_min sigma_max))
(define (static_sigmas_linear steps sigma_min sigma_max)
  (static_get_sigmas_linear steps sigma_min sigma_max))
(define SIGMA_SCHEDULERS (let ((d (make-dict))) (dict-set! d "karras" static_sigmas_karras) (dict-set! d "exponential" static_sigmas_exponential) (dict-set! d "linear" static_sigmas_linear) (dict-set! d "simple" static_sigmas_karras) d))
(define (static_get_sigmas scheduler_name steps sigma_min sigma_max)
  (begin
    (set! fn (hashtable-ref SIGMA_SCHEDULERS scheduler_name (dict-get SIGMA_SCHEDULERS "karras")))
    (fn steps sigma_min sigma_max)
  ))
(define (static_predict_noise unet_fn x sigma text_emb_cond text_emb_uncond cfg_scale)
  (begin
    (if (or (<= cfg_scale 1.0) (equal? text_emb_uncond null))
        (begin (set! eps (unet_fn x sigma text_emb_cond)))
        (begin (set! cond_eps (unet_fn x sigma text_emb_cond)) (set! uncond_eps (unet_fn x sigma text_emb_uncond)) (set! eps_diff (torch_std_sub cond_eps uncond_eps)) (set! eps_scaled (torch_std_mul_scalar eps_diff cfg_scale)) (set! eps (torch_std_add uncond_eps eps_scaled))))
    (set! sigma_eps (torch_std_mul sigma eps))
    (torch_std_sub x sigma_eps)
  ))
(define (static_sample_euler_loop unet_fn x sigmas text_emb_cond text_emb_uncond cfg)
  (static_sample_euler unet_fn x sigmas text_emb_cond text_emb_uncond cfg))
(define (static_sample_euler_ancestral_loop unet_fn x sigmas text_emb_cond text_emb_uncond cfg)
  (static_sample_euler_ancestral unet_fn x sigmas text_emb_cond text_emb_uncond cfg))
(define (static_sample_dpmpp_2m_loop unet_fn x sigmas text_emb_cond text_emb_uncond cfg)
  (static_sample_dpmpp_2m unet_fn x sigmas text_emb_cond text_emb_uncond cfg))
(define (static_sample_ddim_loop unet_fn x sigmas text_emb_cond text_emb_uncond cfg)
  (static_sample_ddim unet_fn x sigmas text_emb_cond text_emb_uncond cfg))
(define (static_sample_dpmpp_sde_loop unet_fn x sigmas text_emb_cond text_emb_uncond cfg)
  (static_sample_dpmpp_sde unet_fn x sigmas text_emb_cond text_emb_uncond cfg))
(define SAMPLER_FUNCTIONS (let ((d (make-dict))) (dict-set! d "euler" static_sample_euler_loop) (dict-set! d "euler_ancestral" static_sample_euler_ancestral_loop) (dict-set! d "dpmpp_2m" static_sample_dpmpp_2m_loop) (dict-set! d "dpmpp_sde" static_sample_dpmpp_sde_loop) (dict-set! d "ddim" static_sample_ddim_loop) d))
(define (static_get_sampler sampler_name)
  (hashtable-ref SAMPLER_FUNCTIONS sampler_name (dict-get SAMPLER_FUNCTIONS "euler")))
;; KSampler init from __init__
(define (KSampler_init self steps sampler_name scheduler_name sigma_min sigma_max)
  (begin
    (set! KSampler_steps steps)
    (set! KSampler_sampler_name sampler_name)
    (set! KSampler_scheduler_name scheduler_name)
    (set! KSampler_sigma_min sigma_min)
    (set! KSampler_sigma_max sigma_max)
  ))
(define (KSampler_sample self noise cfg unet_fn text_emb_cond text_emb_uncond)
  (begin
    (set! sigmas (static_get_sigmas KSampler_scheduler_name KSampler_steps KSampler_sigma_min KSampler_sigma_max))
    (set! sampler_fn (static_get_sampler KSampler_sampler_name))
    (sampler_fn unet_fn noise sigmas text_emb_cond text_emb_uncond cfg)
  ))
(define (static_KSampler steps sampler_name scheduler_name sigma_min sigma_max)
  (KSampler_init 'KSampler steps sampler_name scheduler_name sigma_min sigma_max)
  'KSampler)
(define (static_sample noise steps cfg sampler_name scheduler_name unet_fn text_emb_cond text_emb_uncond sigma_min sigma_max denoise)
  (begin
    (if (< denoise 1.0)
        (begin (set! full_steps (exact (truncate (/ steps denoise)))) (set! sigmas (static_get_sigmas scheduler_name full_steps sigma_min sigma_max)) (set! n_total (torch_std_size sigmas 0)) (set! start_idx (- (- n_total steps) 1)) (set! sigmas (torch_std_narrow sigmas 0 start_idx (+ steps 1))))
        (begin (set! sigmas (static_get_sigmas scheduler_name steps sigma_min sigma_max))))
    (set! sampler_fn (static_get_sampler sampler_name))
    (sampler_fn unet_fn noise sigmas text_emb_cond text_emb_uncond cfg)
  ))
(define SD15_LATENT_CHANNELS 4)
(define SD15_IMAGE_SIZE 512)
(define SD15_LATENT_SIZE 64)
(define SDXL_LATENT_CHANNELS 4)
(define SDXL_IMAGE_SIZE 1024)
(define SDXL_LATENT_SIZE 128)
(define (static_make_sd15_unet_fn sd_dict)
  (letrec ((unet_fn (lambda (x sigma text_emb)
    (torch_std_sd15_unet_forward_dict sd_dict x sigma text_emb null null null 0 0.0))))
    unet_fn))
(define (static_make_sdxl_unet_fn sd_dict original_size_h original_size_w crop_top crop_left target_size_h target_size_w)
  (letrec ((unet_fn (lambda (x sigma text_emb pooled_emb)
    (torch_std_sdxl_unet_forward sd_dict x sigma text_emb pooled_emb original_size_h original_size_w crop_top crop_left target_size_h target_size_w))))
    unet_fn))
;; SD15Pipeline init from __init__
(define (SD15Pipeline_init self sd_dict clip vae unet_fn)
  (begin
    (set! SD15Pipeline_sd_dict sd_dict)
    (set! SD15Pipeline_clip clip)
    (set! SD15Pipeline_vae vae)
    (if unet_fn
    (begin (set! SD15Pipeline_unet_fn unet_fn))
    (begin (set! SD15Pipeline_unet_fn (static_make_sd15_unet_fn sd_dict))))
    (set! SD15Pipeline_lora_A null)
    (set! SD15Pipeline_lora_B null)
    (set! SD15Pipeline_lora_indices null)
    (set! SD15Pipeline_n_lora 0)
  ))
(define (SD15Pipeline_load_lora self lora_dict n_weights max_lora)
  (begin
    (static_lora_init n_weights max_lora)
    (static_lora_load_from_dict lora_dict)
    (set! n (static_lora_match ))
    (if (> n 0)
    (begin (set! SD15Pipeline_lora_A (static_lora_get_A )) (set! SD15Pipeline_lora_B (static_lora_get_B )) (set! SD15Pipeline_lora_indices (static_lora_get_indices )) (set! SD15Pipeline_n_lora n)))
    n
  ))
(define (SD15Pipeline_encode_prompt self text)
  (static_sd1_clip_encode SD15Pipeline_clip text))
(define (SD15Pipeline_prepare_noise self batch_size height width seed)
  (begin
    (torch_std_manual_seed seed)
    (set! latent_h (quotient height 8))
    (set! latent_w (quotient width 8))
    (set! shape (make_int_array 4))
    (int_array_set shape 0 batch_size)
    (int_array_set shape 1 SD15_LATENT_CHANNELS)
    (int_array_set shape 2 latent_h)
    (int_array_set shape 3 latent_w)
    (torch_std_randn shape 4 0)
  ))
(define (SD15Pipeline_txt2img self prompt steps cfg sampler_name scheduler_name height width seed denoise sigma_min sigma_max)
  (letrec ((wrapped_fn (lambda (x sigma te)
    (SD15Pipeline_unet_fn self x sigma te))))
    (begin
      (set! text_emb (SD15Pipeline_encode_prompt self prompt))
    (set! text_emb_uncond (SD15Pipeline_encode_prompt self ""))
    (set! noise (SD15Pipeline_prepare_noise self 1 height width seed))
    (set! sampler (static_KSampler steps sampler_name scheduler_name sigma_min sigma_max))
    (set! latent (KSampler_sample sampler noise cfg wrapped_fn text_emb text_emb_uncond))
    (set! image (torch_std_vae_decode SD15Pipeline_vae latent))
    image
    )))
(define (static_SD15Pipeline sd_dict clip vae unet_fn)
  (SD15Pipeline_init 'SD15Pipeline sd_dict clip vae unet_fn)
  'SD15Pipeline)
;; SDXLPipeline init from __init__
(define (SDXLPipeline_init self sd_dict clip vae unet_fn original_size_h original_size_w crop_top crop_left target_size_h target_size_w)
  (begin
    (set! SDXLPipeline_sd_dict sd_dict)
    (set! SDXLPipeline_clip clip)
    (set! SDXLPipeline_vae vae)
    (if unet_fn
    (begin (set! SDXLPipeline_unet_fn unet_fn))
    (begin (set! SDXLPipeline_unet_fn (static_make_sdxl_unet_fn sd_dict original_size_h original_size_w crop_top crop_left target_size_h target_size_w))))
  ))
(define (SDXLPipeline_encode_prompt self text)
  (static_sdxl_clip_encode SDXLPipeline_clip text))
(define (SDXLPipeline_prepare_noise self batch_size height width seed)
  (begin
    (torch_std_manual_seed seed)
    (set! latent_h (quotient height 8))
    (set! latent_w (quotient width 8))
    (set! shape (make_int_array 4))
    (int_array_set shape 0 batch_size)
    (int_array_set shape 1 SDXL_LATENT_CHANNELS)
    (int_array_set shape 2 latent_h)
    (int_array_set shape 3 latent_w)
    (torch_std_randn shape 4 0)
  ))
(define (SDXLPipeline_txt2img self prompt steps cfg sampler_name scheduler_name height width seed denoise sigma_min sigma_max)
  (letrec ((wrapped_fn (lambda (x sigma te)
    (SDXLPipeline_unet_fn self x sigma te pooled_emb))))
    (begin
      (set! text_emb (SDXLPipeline_encode_prompt self prompt))
    (set! text_emb_uncond (SDXLPipeline_encode_prompt self ""))
    (set! noise (SDXLPipeline_prepare_noise self 1 height width seed))
    (set! sampler (static_KSampler steps sampler_name scheduler_name sigma_min sigma_max))
    (set! latent (KSampler_sample sampler noise cfg wrapped_fn text_emb text_emb_uncond))
    (set! image (torch_std_vae_decode SDXLPipeline_vae latent))
    image
    )))
(define (static_SDXLPipeline sd_dict clip vae unet_fn original_size_h original_size_w crop_top crop_left target_size_h target_size_w)
  (SDXLPipeline_init 'SDXLPipeline sd_dict clip vae unet_fn original_size_h original_size_w crop_top crop_left target_size_h target_size_w)
  'SDXLPipeline)
(define (static_loha_apply weight lora_A1 lora_A2 lora_B1 lora_B2 scale)
  (begin
    (set! delta1 (torch_std_matmul lora_A1 lora_B1))
    (set! delta2 (torch_std_matmul lora_A2 lora_B2))
    (set! delta (torch_std_mul delta1 delta2))
    (set! delta (torch_std_mul_scalar delta scale))
    (torch_std_add weight delta)
  ))
(define (static_loha_merge_into_dict model_dict loha_dict prefix scale)
  (begin
    (set! n (torch_std_safetensors_count loha_dict))
    (set! merged 0)
    (do ((i 0 (+ i 1))) ((>= i n)) (call/cc (lambda (continue) (set! name (torch_std_safetensors_name loha_dict i)) (if (equal? name null)
        (begin (continue))) (if (not (str_contains name "loha_down1"))
        (begin (continue))) (set! base name) (set! base (str_replace base "loha_down1" "")) (set! target_name base) (set! a1_name (str_replace name "loha_down1" "loha_down1")) (set! a2_name (str_replace name "loha_down1" "loha_down2")) (set! b1_name (str_replace name "loha_down1" "loha_up1")) (set! b2_name (str_replace name "loha_down1" "loha_up2")) (set! a1 (torch_std_safetensors_get_tensor_by_name loha_dict a1_name)) (set! a2 (torch_std_safetensors_get_tensor_by_name loha_dict a2_name)) (set! b1 (torch_std_safetensors_get_tensor_by_name loha_dict b1_name)) (set! b2 (torch_std_safetensors_get_tensor_by_name loha_dict b2_name)) (if (or (equal? a1 null) (equal? a2 null) (equal? b1 null) (equal? b2 null))
        (begin (continue))) (set! target (torch_std_safetensors_get_tensor_by_name model_dict target_name)) (if (equal? target null)
        (begin (continue))) (set! merged_w (static_loha_apply target a1 a2 b1 b2 scale)) (set! merged (+ merged 1)))))
    merged
  ))
(define (static_lokr_apply weight lora_A lora_B scale factor)
  (begin
    (set! delta (torch_std_matmul lora_A lora_B))
    (set! w_m (torch_std_size weight 0))
    (set! w_n (torch_std_size weight 1))
    (set! d_m (torch_std_size delta 0))
    (set! d_n (torch_std_size delta 1))
    (if (or (< d_m w_m) (< d_n w_n))
        (begin (set! repeats (make_int_array 2)) (int_array_set repeats 0 (quotient w_m d_m)) (int_array_set repeats 1 (quotient w_n d_n)) (set! reshape_shape (make_int_array 4)) (int_array_set reshape_shape 0 1) (int_array_set reshape_shape 1 d_m) (int_array_set reshape_shape 2 1) (int_array_set reshape_shape 3 d_n) (set! delta_r (torch_std_reshape delta reshape_shape 4)) (set! expand_shape (make_int_array 4)) (int_array_set expand_shape 0 (quotient w_m d_m)) (int_array_set expand_shape 1 d_m) (int_array_set expand_shape 2 (quotient w_n d_n)) (int_array_set expand_shape 3 d_n)))
    (set! delta (torch_std_mul_scalar delta (* scale factor)))
    (torch_std_add weight delta)
  ))
(define (static_lokr_merge_into_dict model_dict lokr_dict prefix scale)
  (begin
    (set! n (torch_std_safetensors_count lokr_dict))
    (set! merged 0)
    (do ((i 0 (+ i 1))) ((>= i n)) (call/cc (lambda (continue) (set! name (torch_std_safetensors_name lokr_dict i)) (if (equal? name null)
        (begin (continue))) (if (not (str_contains name "lokr"))
        (begin (continue))) (if (or (str_contains name "up") (str_contains name "down"))
        (begin (continue))) (set! base name) (set! a_name (string-append base ".up")) (set! b_name (string-append base ".down")) (set! a (torch_std_safetensors_get_tensor_by_name lokr_dict a_name)) (set! b (torch_std_safetensors_get_tensor_by_name lokr_dict b_name)) (if (or (equal? a null) (equal? b null))
        (begin (continue))) (set! target_name (str_replace name "lokr_" "")) (set! target (torch_std_safetensors_get_tensor_by_name model_dict target_name)) (if (equal? target null)
        (begin (continue))) (set! merged_w (static_lokr_apply target a b scale 1.0)) (set! merged (+ merged 1)))))
    merged
  ))
(define (static_oft_apply weight oft_R scale)
  (begin
    (set! n (torch_std_size oft_R 0))
    (set! delta (torch_std_matmul oft_R weight))
    (set! delta (torch_std_sub delta weight))
    (set! delta (torch_std_mul_scalar delta scale))
    (torch_std_add weight delta)
  ))
(define (static_t2i_adapter_init jit_model_path)
  (set! _adapter_jit (torch_std_jit_load jit_model_path)))
(define (static_t2i_adapter_preprocess hint target_h target_w)
  (begin
    (set! c (torch_std_size hint 1))
    (if (> c 3)
        (begin (set! hint (torch_std_narrow hint 1 0 3)))
        (begin (if (equal? c 1)
        (begin (set! tensors (make_ptr_array 3)) (ptr_array_set tensors 0 hint) (ptr_array_set tensors 1 hint) (ptr_array_set tensors 2 hint) (set! hint (torch_std_cat tensors 3 1))))))
    (set! resized (torch_std_image_resize hint target_h target_w "bilinear"))
    resized
  ))
(define (static_t2i_adapter_forward hint)
  (torch_std_jit_forward _adapter_jit hint))
(define (static_t2i_adapter_apply unet_features adapter_features strength start_percent end_percent)
  (begin
    (if (<= strength 0.0)
        (begin unet_features))
    (set! scaled (torch_std_mul_scalar adapter_features strength))
    (torch_std_add unet_features scaled)
  ))
(define (static_t2i_adapter_free )
  (torch_std_jit_module_delete _adapter_jit))
(define (static_flux_controlnet_init safetensors_path)
  (begin
    (set! dict_ptr (torch_std_safetensors_load safetensors_path))
    (set! _fc_n_weights (torch_std_safetensors_count dict_ptr))
    (set! _fc_weights (make_ptr_array _fc_n_weights))
    (do ((i 0 (+ i 1))) ((>= i _fc_n_weights)) (ptr_array_set _fc_weights i (torch_std_safetensors_tensor dict_ptr i)))
    (set! _fc_n_blocks 24)
    (set! _fc_n_heads_img 24)
    (set! _fc_n_heads_txt 16)
    (set! _fc_head_dim 128)
  ))
(define (static_flux_controlnet_forward img txt timestep hint guidance)
  (torch_std_controlnet_forward _fc_weights _fc_n_weights img timestep txt hint 3))
(define (static_flux_controlnet_inject flux_features control_features strength)
  (begin
    (set! scaled (torch_std_mul_scalar control_features strength))
    (torch_std_add flux_features scaled)
  ))
(define (static_flux_controlnet_free )
  (void))
(define LONG_CLIP_MAX_TOKENS 248)
(define LONG_CLIP_EMBED_DIM 768)
(define LONG_CLIP_N_LAYERS 24)
(define LONG_CLIP_N_HEADS 12)
(define (static_long_clip_init jit_model_path vocab_path merges_path)
  (begin
    (set! _lc_jit_module (torch_std_jit_load jit_model_path))
    (set! _lc_tokenizer (torch_std_clip_tokenizer_create vocab_path merges_path))
  ))
(define (static_long_clip_encode text)
  (begin
    (set! tokens (torch_std_clip_tokenizer_encode _lc_tokenizer text))
    (set! cur_len (torch_std_size tokens 1))
    (if (< cur_len LONG_CLIP_MAX_TOKENS)
        (begin (set! pad_len (- LONG_CLIP_MAX_TOKENS cur_len)) (set! pad_shape (make_int_array 2)) (int_array_set pad_shape 0 1) (int_array_set pad_shape 1 pad_len) (set! pad (torch_std_full pad_shape 2 0.0 2)) (set! tensors (make_ptr_array 2)) (ptr_array_set tensors 0 tokens) (ptr_array_set tensors 1 pad) (set! tokens (torch_std_cat tensors 2 1)))
        (begin (set! tokens (torch_std_narrow tokens 1 0 LONG_CLIP_MAX_TOKENS))))
    (torch_std_clip_text_forward _lc_jit_module tokens 0)
  ))
(define (static_long_clip_free )
  (begin
    (torch_std_clip_tokenizer_free _lc_tokenizer)
    (torch_std_jit_module_delete _lc_jit_module)
  ))
(define (static_t5_config_init )
  (begin
    (set! d (make_dict ))
    (set! small (make_dict ))
    (dict_set small "d_model" 512)
    (dict_set small "d_ff" 2048)
    (dict_set small "num_layers" 6)
    (dict_set small "num_heads" 8)
    (dict_set small "vocab_size" 32128)
    (dict_set small "max_len" 512)
    (dict_set d "t5-small" small)
    (set! base (make_dict ))
    (dict_set base "d_model" 768)
    (dict_set base "d_ff" 3072)
    (dict_set base "num_layers" 12)
    (dict_set base "num_heads" 12)
    (dict_set base "vocab_size" 32128)
    (dict_set base "max_len" 512)
    (dict_set d "t5-base" base)
    (set! large (make_dict ))
    (dict_set large "d_model" 1024)
    (dict_set large "d_ff" 4096)
    (dict_set large "num_layers" 24)
    (dict_set large "num_heads" 16)
    (dict_set large "vocab_size" 32128)
    (dict_set large "max_len" 512)
    (dict_set d "t5-large" large)
    (set! xl (make_dict ))
    (dict_set xl "d_model" 2048)
    (dict_set xl "d_ff" 8192)
    (dict_set xl "num_layers" 24)
    (dict_set xl "num_heads" 32)
    (dict_set xl "vocab_size" 32128)
    (dict_set xl "max_len" 512)
    (dict_set d "t5-xl" xl)
    (set! xxl (make_dict ))
    (dict_set xxl "d_model" 4096)
    (dict_set xxl "d_ff" 16384)
    (dict_set xxl "num_layers" 24)
    (dict_set xxl "num_heads" 64)
    (dict_set xxl "vocab_size" 32128)
    (dict_set xxl "max_len" 512)
    (dict_set d "t5-xxl" xxl)
    (set! flux (make_dict ))
    (dict_set flux "d_model" 4096)
    (dict_set flux "d_ff" 16384)
    (dict_set flux "num_layers" 24)
    (dict_set flux "num_heads" 64)
    (dict_set flux "vocab_size" 32128)
    (dict_set flux "max_len" 512)
    (dict_set d "flux-t5" flux)
    (set! T5_CONFIGS d)
  ))
(define (static_t5_config_get name)
  (dict_get T5_CONFIGS name))
(define (static_t5_config_get_int config key)
  (exact (truncate (dict_get config key))))
(define (static_t5_config_get_max_len name)
  (begin
    (set! cfg (static_t5_config_get name))
    (if cfg
        (begin (static_t5_config_get_int cfg "max_len")))
    512
  ))
(define (static_sd3_init jit_model_path)
  (set! _sd3_dict (torch_std_safetensors_load jit_model_path)))
(define (static_sd3_encode_prompt text)
  (begin
    (set! t5_out (static_t5_encode text))
    (set! clip_l_out (static_sd1_clip_encode _sd3_clip_l text 77))
    (set! clip_gokens (static_sdxl_clip_tokenize _sd3_clip_g text))
    (set! clip_g_out (static_sdxl_clip_encode_tokens _sd3_clip_g clip_gokens))
    (set! tensors (make_ptr_array 3))
    (ptr_array_set tensors 0 clip_l_out)
    (ptr_array_set tensors 1 clip_g_out)
    (ptr_array_set tensors 2 t5_out)
    (set! concat (torch_std_cat tensors 3 1))
    concat
  ))
(define (static_sd3_forward img txt timestep guidance)
  (torch_std_sd3_mmdit_forward _sd3_dict img timestep txt guidance))
(define (static_sd3_generate prompt steps cfg seed width height)
  (begin
    (set! txt (static_sd3_encode_prompt prompt))
    (set! sigmas (torch_std_fm_sigmas steps 0.002 30.0))
    (set! latent_h (quotient height 8))
    (set! latent_w (quotient width 8))
    (set! x (torch_std_randn (* latent_h 16) latent_w seed))
    (set! x (torch_std_unsqueeze x 0))
    (set! dt (/ 1.0 (inexact steps)))
    (do ((i 0 (+ i 1))) ((>= i steps)) (set! t_scalar (* (inexact i) dt)) (set! t_shape (make_int_array 1)) (int_array_set t_shape 0 1) (set! t (torch_std_full t_shape 1 t_scalar 0)) (set! guidance_shape (make_int_array 1)) (int_array_set guidance_shape 0 1) (set! guidance_tensor (torch_std_full guidance_shape 1 cfg 0)) (set! v (static_sd3_forward x txt t guidance_tensor)) (set! x (torch_std_fm_step v x dt)))
    (set! image (torch_std_vae_decode x))
    image
  ))
(define (static_sd3_free )
  (void))
(define (static_gligen_init weight_ptrs n_weights)
  (set! _gligen_n_blocks 0))
(define (static_gligen_set_position boxes masks pos_emb n_objs)
  (set! _gligen_n_blocks n_objs))
(define (static_gligen_apply unet_latent timestep text_emb weight_ptrs n_weights guidance)
  (begin
    (if (> _gligen_n_blocks 0)
        (begin (torch_std_sd_unet_forward_v2 weight_ptrs n_weights unet_latent timestep text_emb null 0 null null 0 0.0 _gligen_objs _gligen_alphas _gligen_block_indices _gligen_n_blocks null 0.0)))
    (torch_std_sd_unet_forward weight_ptrs n_weights unet_latent timestep text_emb null null null 0 0.0)
  ))
(define (static_gligen_free )
  (void))
(define (static_ip_adapter_init )
  (begin
    (set! _ip_adapter_img_emb null)
    (set! _ip_adapter_scale 1.0)
  ))
(define (static_ip_adapter_set_image image_emb scale)
  (begin
    (set! _ip_adapter_img_emb image_emb)
    (set! _ip_adapter_scale scale)
  ))
(define (static_ip_adapter_apply unet_latent timestep text_emb weight_ptrs n_weights)
  (torch_std_sd_unet_forward_v2 weight_ptrs n_weights unet_latent timestep text_emb null 0 null null 0 0.0 null null null 0 _ip_adapter_img_emb _ip_adapter_scale))
(define (static_ip_adapter_free )
  (void))
(define (static_stable_cascade_init stage_a_path stage_b_path stage_c_path)
  (begin
    (set! _sc_stage_a (torch_std_jit_load stage_a_path))
    (set! _sc_stage_b (torch_std_jit_load stage_b_path))
    (set! sd_dict (torch_std_safetensors_load stage_c_path))
    (set! n (torch_std_safetensors_count sd_dict))
    (set! w (make_ptr_array n))
    (do ((i 0 (+ i 1))) ((>= i n)) (ptr_array_set w i (torch_std_safetensors_tensor sd_dict i)))
    (set! _sc_weight_ptrs w)
    (set! _sc_n_weights n)
    (set! _sc_has_init 1)
  ))
(define (static_sc_stage_a_encode image)
  (torch_std_jit_forward _sc_stage_a image))
(define (static_sc_stage_a_decode latent)
  (torch_std_jit_forward _sc_stage_a latent))
(define (static_sc_stage_b_forward latent timestep r)
  (torch_std_jit_forward _sc_stage_b latent timestep r))
(define (static_sc_stage_b_sample noise steps r_val seed)
  (begin
    (set! dt (/ 1.0 (inexact steps)))
    (set! x noise)
    (do ((i 0 (+ i 1))) ((>= i steps)) (set! t_scalar (* (inexact i) dt)) (set! t_shape (make_int_array 1)) (int_array_set t_shape 0 1) (set! t (torch_std_full t_shape 1 t_scalar 0)) (set! r_shape (make_int_array 1)) (int_array_set r_shape 0 1) (set! r (torch_std_full r_shape 1 r_val 0)) (set! x (static_sc_stage_b_forward x t r)))
    x
  ))
(define (static_sc_stage_c_forward latent timestep r clip_text clip_text_pooled clip_img)
  (torch_std_stable_cascade_stage_c _sc_weight_ptrs _sc_n_weights latent r timestep clip_text clip_text_pooled clip_img))
(define (static_sc_stage_c_sample noise steps r_val clip_text clip_text_pooled seed)
  (begin
    (set! dt (/ 1.0 (inexact steps)))
    (set! x noise)
    (do ((i 0 (+ i 1))) ((>= i steps)) (set! t_scalar (* (inexact i) dt)) (set! t_shape (make_int_array 1)) (int_array_set t_shape 0 1) (set! t (torch_std_full t_shape 1 t_scalar 0)) (set! r_shape (make_int_array 1)) (int_array_set r_shape 0 1) (set! r (torch_std_full r_shape 1 r_val 0)) (set! x (static_sc_stage_c_forward x t r clip_text clip_text_pooled null)))
    x
  ))
(define (static_stable_cascade_generate prompt steps_b steps_c height width seed)
  (begin
    (set! latent_h_c (quotient height 32))
    (set! latent_w_c (quotient width 32))
    (set! noise (torch_std_randn 16 (* latent_h_c latent_w_c) seed))
    (set! noise (torch_std_unsqueeze noise 0))
    (set! clip_text null)
    (set! clip_text_pooled null)
    (set! c_out (static_sc_stage_c_sample noise steps_c 1.0 clip_text clip_text_pooled seed))
    (set! b_out c_out)
    (set! image (static_sc_stage_a_decode b_out))
    image
  ))
(define (static_stable_cascade_free )
  (begin
    (torch_std_jit_module_delete _sc_stage_a)
    (torch_std_jit_module_delete _sc_stage_b)
    (torch_std_jit_module_delete _sc_stage_c)
  ))
(define (static_pixart_init safetensors_path)
  (begin
    (set! sd_dict (torch_std_safetensors_load safetensors_path))
    (set! n (torch_std_safetensors_count sd_dict))
    (set! w (make_ptr_array n))
    (do ((i 0 (+ i 1))) ((>= i n)) (ptr_array_set w i (torch_std_safetensors_tensor sd_dict i)))
    (set! _pixart_weight_ptrs w)
    (set! _pixart_n_weights n)
  ))
(define (static_pixart_forward latent timestep text_emb height width)
  (torch_std_pixart_forward _pixart_weight_ptrs _pixart_n_weights latent timestep text_emb height width 2))
(define (static_pixart_generate prompt steps cfg height width seed)
  (begin
    (set! txt (static_t5_encode prompt 120))
    (set! latent_h (quotient height 8))
    (set! latent_w (quotient width 8))
    (set! noise (torch_std_randn 4 (* latent_h latent_w) seed))
    (set! noise (torch_std_unsqueeze noise 0))
    (set! x noise)
    (set! dt (/ 1.0 (inexact steps)))
    (do ((i 0 (+ i 1))) ((>= i steps)) (set! t_scalar (* (inexact i) dt)) (set! t_shape (make_int_array 1)) (int_array_set t_shape 0 1) (set! t (torch_std_full t_shape 1 t_scalar 0)) (set! noise_pred (static_pixart_forward x t txt height width)) (set! x (torch_std_sub x (torch_std_mul_scalar noise_pred dt))))
    (torch_std_vae_decode x)
  ))
(define (static_pixart_free )
  (void))
(define (static_hunyuan_video_init safetensors_path vae_path)
  (begin
    (set! _hunyuan_dict (torch_std_safetensors_load safetensors_path))
    (set! _hunyuan_vae (torch_std_jit_load vae_path))
  ))
(define (static_hunyuan_video_encode frames)
  (torch_std_jit_forward _hunyuan_vae frames))
(define (static_hunyuan_video_decode latent)
  (torch_std_jit_forward _hunyuan_vae latent))
(define (static_hunyuan_video_forward latent timestep text_emb n_frames height width)
  (torch_std_hunyuan_video_forward _hunyuan_dict latent timestep text_emb n_frames height width))
(define (static_hunyuan_video_generate prompt steps cfg n_frames height width seed)
  (begin
    (set! latent_h (quotient height 8))
    (set! latent_w (quotient width 8))
    (set! noise (torch_std_randn (* (* (* n_frames 4) latent_h) latent_w) 1 seed))
    (set! shape (make_int_array 4))
    (int_array_set shape 0 n_frames)
    (int_array_set shape 1 4)
    (int_array_set shape 2 latent_h)
    (int_array_set shape 3 latent_w)
    (set! x (torch_std_reshape noise shape 4))
    (set! dt (/ 1.0 (inexact steps)))
    (do ((i 0 (+ i 1))) ((>= i steps)) (set! t_scalar (* (inexact i) dt)) (set! t_shape (make_int_array 1)) (int_array_set t_shape 0 1) (set! t (torch_std_full t_shape 1 t_scalar 0)) (set! noise_pred (static_hunyuan_video_forward x t null n_frames height width)) (set! x (torch_std_sub x (torch_std_mul_scalar noise_pred dt))))
    (static_hunyuan_video_decode x)
  ))
(define (static_hunyuan_video_free )
  (void))
(define (static_wan_video_init safetensors_path vae_path)
  (begin
    (set! _wan_dict (torch_std_safetensors_load safetensors_path))
    (set! _wan_vae (torch_std_jit_load vae_path))
  ))
(define (static_wan_video_encode frames)
  (torch_std_jit_forward _wan_vae frames))
(define (static_wan_video_decode latent)
  (torch_std_jit_forward _wan_vae latent))
(define (static_wan_video_forward latent timestep text_emb n_frames height width)
  (torch_std_wan_video_forward _wan_dict latent timestep text_emb n_frames height width))
(define (static_wan_video_generate prompt steps n_frames height width seed)
  (begin
    (set! latent_h (quotient height 8))
    (set! latent_w (quotient width 8))
    (set! noise (torch_std_randn (* (* (* n_frames 16) latent_h) latent_w) 1 seed))
    (set! shape (make_int_array 4))
    (int_array_set shape 0 n_frames)
    (int_array_set shape 1 16)
    (int_array_set shape 2 latent_h)
    (int_array_set shape 3 latent_w)
    (set! x (torch_std_reshape noise shape 4))
    (set! dt (/ 1.0 (inexact steps)))
    (do ((i 0 (+ i 1))) ((>= i steps)) (set! t_scalar (* (inexact i) dt)) (set! t_shape (make_int_array 1)) (int_array_set t_shape 0 1) (set! t (torch_std_full t_shape 1 t_scalar 0)) (set! noise_pred (static_wan_video_forward x t null n_frames height width)) (set! x (torch_std_sub x (torch_std_mul_scalar noise_pred dt))))
    (static_wan_video_decode x)
  ))
(define (static_wan_video_free )
  (void))
(define (static_cosmos_init safetensors_path vae_path)
  (begin
    (set! _cosmos_dict (torch_std_safetensors_load safetensors_path))
    (set! _cosmos_vae (torch_std_jit_load vae_path))
  ))
(define (static_cosmos_encode frames)
  (torch_std_jit_forward _cosmos_vae frames))
(define (static_cosmos_decode latent)
  (torch_std_jit_forward _cosmos_vae latent))
(define (static_cosmos_forward latent timestep text_emb n_frames height width)
  (torch_std_cosmos_forward _cosmos_dict latent timestep text_emb n_frames height width))
(define (static_cosmos_generate prompt steps n_frames height width seed)
  (begin
    (set! latent_h (quotient height 8))
    (set! latent_w (quotient width 8))
    (set! noise (torch_std_randn (* (* (* n_frames 16) latent_h) latent_w) 1 seed))
    (set! shape (make_int_array 4))
    (int_array_set shape 0 n_frames)
    (int_array_set shape 1 16)
    (int_array_set shape 2 latent_h)
    (int_array_set shape 3 latent_w)
    (set! x (torch_std_reshape noise shape 4))
    (set! dt (/ 1.0 (inexact steps)))
    (do ((i 0 (+ i 1))) ((>= i steps)) (set! t_scalar (* (inexact i) dt)) (set! t_shape (make_int_array 1)) (int_array_set t_shape 0 1) (set! t (torch_std_full t_shape 1 t_scalar 0)) (set! noise_pred (static_cosmos_forward x t null n_frames height width)) (set! x (torch_std_sub x (torch_std_mul_scalar noise_pred dt))))
    (static_cosmos_decode x)
  ))
(define (static_cosmos_free )
  (void))

;; Entry point
(static_main)

;;=== WARNINGS ===
;;  ! Parameter 'sigma_min' has default value 0.03 — caller must provide all args at line 1228
;;  ! Parameter 'sigma_max' has default value 14.6 — caller must provide all args at line 1228
;;  ! Parameter 'sigma_min' has default value 0.03 — caller must provide all args at line 1237
;;  ! Parameter 'sigma_max' has default value 14.6 — caller must provide all args at line 1237
;;  ! Parameter 'sigma_min' has default value 0.03 — caller must provide all args at line 1243
;;  ! Parameter 'sigma_max' has default value 14.6 — caller must provide all args at line 1243
;;  ! Parameter 'sigma_min' has default value 0.03 — caller must provide all args at line 1249
;;  ! Parameter 'sigma_max' has default value 14.6 — caller must provide all args at line 1249
;;  ! Parameter 'cfg_scale' has default value 7.0 — caller must provide all args at line 1313
;;  ! Parameter 'cfg_scale' has default value 7.0 — caller must provide all args at line 1342
;;  ! Parameter 'cfg_scale' has default value 7.0 — caller must provide all args at line 1364
;;  ! Parameter 'cfg_scale' has default value 7.0 — caller must provide all args at line 1393
;;  ! Parameter 'eta' has default value 0.0 — caller must provide all args at line 1393
;;  ! Parameter 'cfg_scale' has default value 7.0 — caller must provide all args at line 1420
;;  ! Parameter 'eta' has default value 1.0 — caller must provide all args at line 1420
;;  ! Parameter 'sigma_min' has default value 0.03 — caller must provide all args at line 2571
;;  ! Parameter 'sigma_max' has default value 14.6 — caller must provide all args at line 2571
;;  ! Parameter 'crop_top' has default value 0.0 — caller must provide all args at line 2699
;;  ! Parameter 'crop_left' has default value 0.0 — caller must provide all args at line 2699
;;  ! Parameter 'target_size_h' has default value 1024.0 — caller must provide all args at line 2699
;;  ! Parameter 'target_size_w' has default value 1024.0 — caller must provide all args at line 2699
