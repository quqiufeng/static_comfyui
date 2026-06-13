
# unet_forward.static.py — SD UNet forward (auto-generated)
# 所有 Conv2d 使用内联 im2col + dgemm 模式

def unet_forward(latent, timestep, context, weights_dir, n, h, w):
    hh: int = h
    ww: int = w
    h_cur: list[float]

    input_blocks_0_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_0_0_bias.bin", 320)
    input_blocks_0_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_0_0_weight.bin", 11520)
    input_blocks_1_0_emb_layers_1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_1_0_emb_layers_1_bias.bin", 320)
    input_blocks_1_0_emb_layers_1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_1_0_emb_layers_1_weight.bin", 409600)
    input_blocks_1_0_in_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_1_0_in_layers_0_bias.bin", 320)
    input_blocks_1_0_in_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_1_0_in_layers_0_weight.bin", 320)
    input_blocks_1_0_in_layers_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_1_0_in_layers_2_bias.bin", 320)
    input_blocks_1_0_in_layers_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_1_0_in_layers_2_weight.bin", 921600)
    input_blocks_1_0_out_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_1_0_out_layers_0_bias.bin", 320)
    input_blocks_1_0_out_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_1_0_out_layers_0_weight.bin", 320)
    input_blocks_1_0_out_layers_3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_1_0_out_layers_3_bias.bin", 320)
    input_blocks_1_0_out_layers_3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_1_0_out_layers_3_weight.bin", 921600)
    input_blocks_2_0_emb_layers_1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_2_0_emb_layers_1_bias.bin", 320)
    input_blocks_2_0_emb_layers_1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_2_0_emb_layers_1_weight.bin", 409600)
    input_blocks_2_0_in_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_2_0_in_layers_0_bias.bin", 320)
    input_blocks_2_0_in_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_2_0_in_layers_0_weight.bin", 320)
    input_blocks_2_0_in_layers_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_2_0_in_layers_2_bias.bin", 320)
    input_blocks_2_0_in_layers_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_2_0_in_layers_2_weight.bin", 921600)
    input_blocks_2_0_out_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_2_0_out_layers_0_bias.bin", 320)
    input_blocks_2_0_out_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_2_0_out_layers_0_weight.bin", 320)
    input_blocks_2_0_out_layers_3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_2_0_out_layers_3_bias.bin", 320)
    input_blocks_2_0_out_layers_3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_2_0_out_layers_3_weight.bin", 921600)
    input_blocks_3_0_op_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_3_0_op_bias.bin", 320)
    input_blocks_3_0_op_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_3_0_op_weight.bin", 921600)
    input_blocks_4_0_emb_layers_1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_0_emb_layers_1_bias.bin", 640)
    input_blocks_4_0_emb_layers_1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_0_emb_layers_1_weight.bin", 819200)
    input_blocks_4_0_in_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_0_in_layers_0_bias.bin", 320)
    input_blocks_4_0_in_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_0_in_layers_0_weight.bin", 320)
    input_blocks_4_0_in_layers_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_0_in_layers_2_bias.bin", 640)
    input_blocks_4_0_in_layers_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_0_in_layers_2_weight.bin", 1843200)
    input_blocks_4_0_out_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_0_out_layers_0_bias.bin", 640)
    input_blocks_4_0_out_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_0_out_layers_0_weight.bin", 640)
    input_blocks_4_0_out_layers_3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_0_out_layers_3_bias.bin", 640)
    input_blocks_4_0_out_layers_3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_0_out_layers_3_weight.bin", 3686400)
    input_blocks_4_0_skip_connection_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_0_skip_connection_bias.bin", 640)
    input_blocks_4_0_skip_connection_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_0_skip_connection_weight.bin", 204800)
    input_blocks_4_1_norm_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_norm_bias.bin", 640)
    input_blocks_4_1_norm_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_norm_weight.bin", 640)
    input_blocks_4_1_proj_in_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_proj_in_bias.bin", 640)
    input_blocks_4_1_proj_in_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_proj_in_weight.bin", 409600)
    input_blocks_4_1_proj_out_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_proj_out_bias.bin", 640)
    input_blocks_4_1_proj_out_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_proj_out_weight.bin", 409600)
    input_blocks_4_1_transformer_blocks_0_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_attn1_to_k_weight.bin", 409600)
    input_blocks_4_1_transformer_blocks_0_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_attn1_to_out_0_bias.bin", 640)
    input_blocks_4_1_transformer_blocks_0_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_attn1_to_out_0_weight.bin", 409600)
    input_blocks_4_1_transformer_blocks_0_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_attn1_to_q_weight.bin", 409600)
    input_blocks_4_1_transformer_blocks_0_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_attn1_to_v_weight.bin", 409600)
    input_blocks_4_1_transformer_blocks_0_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_attn2_to_k_weight.bin", 1310720)
    input_blocks_4_1_transformer_blocks_0_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_attn2_to_out_0_bias.bin", 640)
    input_blocks_4_1_transformer_blocks_0_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_attn2_to_out_0_weight.bin", 409600)
    input_blocks_4_1_transformer_blocks_0_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_attn2_to_q_weight.bin", 409600)
    input_blocks_4_1_transformer_blocks_0_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_attn2_to_v_weight.bin", 1310720)
    input_blocks_4_1_transformer_blocks_0_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_ff_net_0_proj_bias.bin", 5120)
    input_blocks_4_1_transformer_blocks_0_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_ff_net_0_proj_weight.bin", 3276800)
    input_blocks_4_1_transformer_blocks_0_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_ff_net_2_bias.bin", 640)
    input_blocks_4_1_transformer_blocks_0_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_ff_net_2_weight.bin", 1638400)
    input_blocks_4_1_transformer_blocks_0_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_norm1_bias.bin", 640)
    input_blocks_4_1_transformer_blocks_0_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_norm1_weight.bin", 640)
    input_blocks_4_1_transformer_blocks_0_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_norm2_bias.bin", 640)
    input_blocks_4_1_transformer_blocks_0_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_norm2_weight.bin", 640)
    input_blocks_4_1_transformer_blocks_0_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_norm3_bias.bin", 640)
    input_blocks_4_1_transformer_blocks_0_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_0_norm3_weight.bin", 640)
    input_blocks_4_1_transformer_blocks_1_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_attn1_to_k_weight.bin", 409600)
    input_blocks_4_1_transformer_blocks_1_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_attn1_to_out_0_bias.bin", 640)
    input_blocks_4_1_transformer_blocks_1_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_attn1_to_out_0_weight.bin", 409600)
    input_blocks_4_1_transformer_blocks_1_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_attn1_to_q_weight.bin", 409600)
    input_blocks_4_1_transformer_blocks_1_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_attn1_to_v_weight.bin", 409600)
    input_blocks_4_1_transformer_blocks_1_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_attn2_to_k_weight.bin", 1310720)
    input_blocks_4_1_transformer_blocks_1_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_attn2_to_out_0_bias.bin", 640)
    input_blocks_4_1_transformer_blocks_1_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_attn2_to_out_0_weight.bin", 409600)
    input_blocks_4_1_transformer_blocks_1_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_attn2_to_q_weight.bin", 409600)
    input_blocks_4_1_transformer_blocks_1_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_attn2_to_v_weight.bin", 1310720)
    input_blocks_4_1_transformer_blocks_1_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_ff_net_0_proj_bias.bin", 5120)
    input_blocks_4_1_transformer_blocks_1_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_ff_net_0_proj_weight.bin", 3276800)
    input_blocks_4_1_transformer_blocks_1_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_ff_net_2_bias.bin", 640)
    input_blocks_4_1_transformer_blocks_1_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_ff_net_2_weight.bin", 1638400)
    input_blocks_4_1_transformer_blocks_1_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_norm1_bias.bin", 640)
    input_blocks_4_1_transformer_blocks_1_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_norm1_weight.bin", 640)
    input_blocks_4_1_transformer_blocks_1_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_norm2_bias.bin", 640)
    input_blocks_4_1_transformer_blocks_1_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_norm2_weight.bin", 640)
    input_blocks_4_1_transformer_blocks_1_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_norm3_bias.bin", 640)
    input_blocks_4_1_transformer_blocks_1_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_4_1_transformer_blocks_1_norm3_weight.bin", 640)
    input_blocks_5_0_emb_layers_1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_0_emb_layers_1_bias.bin", 640)
    input_blocks_5_0_emb_layers_1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_0_emb_layers_1_weight.bin", 819200)
    input_blocks_5_0_in_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_0_in_layers_0_bias.bin", 640)
    input_blocks_5_0_in_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_0_in_layers_0_weight.bin", 640)
    input_blocks_5_0_in_layers_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_0_in_layers_2_bias.bin", 640)
    input_blocks_5_0_in_layers_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_0_in_layers_2_weight.bin", 3686400)
    input_blocks_5_0_out_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_0_out_layers_0_bias.bin", 640)
    input_blocks_5_0_out_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_0_out_layers_0_weight.bin", 640)
    input_blocks_5_0_out_layers_3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_0_out_layers_3_bias.bin", 640)
    input_blocks_5_0_out_layers_3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_0_out_layers_3_weight.bin", 3686400)
    input_blocks_5_1_norm_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_norm_bias.bin", 640)
    input_blocks_5_1_norm_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_norm_weight.bin", 640)
    input_blocks_5_1_proj_in_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_proj_in_bias.bin", 640)
    input_blocks_5_1_proj_in_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_proj_in_weight.bin", 409600)
    input_blocks_5_1_proj_out_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_proj_out_bias.bin", 640)
    input_blocks_5_1_proj_out_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_proj_out_weight.bin", 409600)
    input_blocks_5_1_transformer_blocks_0_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_attn1_to_k_weight.bin", 409600)
    input_blocks_5_1_transformer_blocks_0_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_attn1_to_out_0_bias.bin", 640)
    input_blocks_5_1_transformer_blocks_0_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_attn1_to_out_0_weight.bin", 409600)
    input_blocks_5_1_transformer_blocks_0_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_attn1_to_q_weight.bin", 409600)
    input_blocks_5_1_transformer_blocks_0_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_attn1_to_v_weight.bin", 409600)
    input_blocks_5_1_transformer_blocks_0_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_attn2_to_k_weight.bin", 1310720)
    input_blocks_5_1_transformer_blocks_0_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_attn2_to_out_0_bias.bin", 640)
    input_blocks_5_1_transformer_blocks_0_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_attn2_to_out_0_weight.bin", 409600)
    input_blocks_5_1_transformer_blocks_0_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_attn2_to_q_weight.bin", 409600)
    input_blocks_5_1_transformer_blocks_0_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_attn2_to_v_weight.bin", 1310720)
    input_blocks_5_1_transformer_blocks_0_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_ff_net_0_proj_bias.bin", 5120)
    input_blocks_5_1_transformer_blocks_0_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_ff_net_0_proj_weight.bin", 3276800)
    input_blocks_5_1_transformer_blocks_0_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_ff_net_2_bias.bin", 640)
    input_blocks_5_1_transformer_blocks_0_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_ff_net_2_weight.bin", 1638400)
    input_blocks_5_1_transformer_blocks_0_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_norm1_bias.bin", 640)
    input_blocks_5_1_transformer_blocks_0_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_norm1_weight.bin", 640)
    input_blocks_5_1_transformer_blocks_0_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_norm2_bias.bin", 640)
    input_blocks_5_1_transformer_blocks_0_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_norm2_weight.bin", 640)
    input_blocks_5_1_transformer_blocks_0_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_norm3_bias.bin", 640)
    input_blocks_5_1_transformer_blocks_0_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_0_norm3_weight.bin", 640)
    input_blocks_5_1_transformer_blocks_1_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_attn1_to_k_weight.bin", 409600)
    input_blocks_5_1_transformer_blocks_1_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_attn1_to_out_0_bias.bin", 640)
    input_blocks_5_1_transformer_blocks_1_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_attn1_to_out_0_weight.bin", 409600)
    input_blocks_5_1_transformer_blocks_1_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_attn1_to_q_weight.bin", 409600)
    input_blocks_5_1_transformer_blocks_1_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_attn1_to_v_weight.bin", 409600)
    input_blocks_5_1_transformer_blocks_1_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_attn2_to_k_weight.bin", 1310720)
    input_blocks_5_1_transformer_blocks_1_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_attn2_to_out_0_bias.bin", 640)
    input_blocks_5_1_transformer_blocks_1_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_attn2_to_out_0_weight.bin", 409600)
    input_blocks_5_1_transformer_blocks_1_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_attn2_to_q_weight.bin", 409600)
    input_blocks_5_1_transformer_blocks_1_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_attn2_to_v_weight.bin", 1310720)
    input_blocks_5_1_transformer_blocks_1_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_ff_net_0_proj_bias.bin", 5120)
    input_blocks_5_1_transformer_blocks_1_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_ff_net_0_proj_weight.bin", 3276800)
    input_blocks_5_1_transformer_blocks_1_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_ff_net_2_bias.bin", 640)
    input_blocks_5_1_transformer_blocks_1_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_ff_net_2_weight.bin", 1638400)
    input_blocks_5_1_transformer_blocks_1_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_norm1_bias.bin", 640)
    input_blocks_5_1_transformer_blocks_1_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_norm1_weight.bin", 640)
    input_blocks_5_1_transformer_blocks_1_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_norm2_bias.bin", 640)
    input_blocks_5_1_transformer_blocks_1_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_norm2_weight.bin", 640)
    input_blocks_5_1_transformer_blocks_1_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_norm3_bias.bin", 640)
    input_blocks_5_1_transformer_blocks_1_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_5_1_transformer_blocks_1_norm3_weight.bin", 640)
    input_blocks_6_0_op_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_6_0_op_bias.bin", 640)
    input_blocks_6_0_op_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_6_0_op_weight.bin", 3686400)
    input_blocks_7_0_emb_layers_1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_0_emb_layers_1_bias.bin", 1280)
    input_blocks_7_0_emb_layers_1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_0_emb_layers_1_weight.bin", 1638400)
    input_blocks_7_0_in_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_0_in_layers_0_bias.bin", 640)
    input_blocks_7_0_in_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_0_in_layers_0_weight.bin", 640)
    input_blocks_7_0_in_layers_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_0_in_layers_2_bias.bin", 1280)
    input_blocks_7_0_in_layers_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_0_in_layers_2_weight.bin", 7372800)
    input_blocks_7_0_out_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_0_out_layers_0_bias.bin", 1280)
    input_blocks_7_0_out_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_0_out_layers_0_weight.bin", 1280)
    input_blocks_7_0_out_layers_3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_0_out_layers_3_bias.bin", 1280)
    input_blocks_7_0_out_layers_3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_0_out_layers_3_weight.bin", 14745600)
    input_blocks_7_0_skip_connection_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_0_skip_connection_bias.bin", 1280)
    input_blocks_7_0_skip_connection_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_0_skip_connection_weight.bin", 819200)
    input_blocks_7_1_norm_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_norm_bias.bin", 1280)
    input_blocks_7_1_norm_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_norm_weight.bin", 1280)
    input_blocks_7_1_proj_in_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_proj_in_bias.bin", 1280)
    input_blocks_7_1_proj_in_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_proj_in_weight.bin", 1638400)
    input_blocks_7_1_proj_out_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_proj_out_bias.bin", 1280)
    input_blocks_7_1_proj_out_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_proj_out_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_0_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_attn1_to_k_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_0_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_attn1_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_0_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_0_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_attn1_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_0_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_attn1_to_v_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_0_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_attn2_to_k_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_0_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_attn2_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_0_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_0_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_attn2_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_0_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_attn2_to_v_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_0_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_ff_net_0_proj_bias.bin", 10240)
    input_blocks_7_1_transformer_blocks_0_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_7_1_transformer_blocks_0_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_ff_net_2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_0_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_ff_net_2_weight.bin", 6553600)
    input_blocks_7_1_transformer_blocks_0_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_norm1_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_0_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_norm1_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_0_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_norm2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_0_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_norm2_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_0_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_norm3_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_0_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_0_norm3_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_1_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_attn1_to_k_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_1_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_attn1_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_1_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_1_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_attn1_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_1_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_attn1_to_v_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_1_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_attn2_to_k_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_1_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_attn2_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_1_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_1_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_attn2_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_1_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_attn2_to_v_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_1_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_ff_net_0_proj_bias.bin", 10240)
    input_blocks_7_1_transformer_blocks_1_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_7_1_transformer_blocks_1_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_ff_net_2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_1_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_ff_net_2_weight.bin", 6553600)
    input_blocks_7_1_transformer_blocks_1_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_norm1_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_1_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_norm1_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_1_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_norm2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_1_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_norm2_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_1_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_norm3_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_1_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_1_norm3_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_2_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_attn1_to_k_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_2_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_attn1_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_2_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_2_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_attn1_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_2_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_attn1_to_v_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_2_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_attn2_to_k_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_2_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_attn2_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_2_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_2_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_attn2_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_2_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_attn2_to_v_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_2_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_ff_net_0_proj_bias.bin", 10240)
    input_blocks_7_1_transformer_blocks_2_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_7_1_transformer_blocks_2_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_ff_net_2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_2_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_ff_net_2_weight.bin", 6553600)
    input_blocks_7_1_transformer_blocks_2_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_norm1_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_2_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_norm1_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_2_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_norm2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_2_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_norm2_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_2_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_norm3_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_2_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_2_norm3_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_3_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_attn1_to_k_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_3_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_attn1_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_3_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_3_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_attn1_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_3_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_attn1_to_v_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_3_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_attn2_to_k_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_3_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_attn2_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_3_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_3_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_attn2_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_3_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_attn2_to_v_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_3_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_ff_net_0_proj_bias.bin", 10240)
    input_blocks_7_1_transformer_blocks_3_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_7_1_transformer_blocks_3_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_ff_net_2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_3_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_ff_net_2_weight.bin", 6553600)
    input_blocks_7_1_transformer_blocks_3_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_norm1_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_3_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_norm1_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_3_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_norm2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_3_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_norm2_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_3_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_norm3_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_3_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_3_norm3_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_4_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_attn1_to_k_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_4_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_attn1_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_4_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_4_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_attn1_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_4_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_attn1_to_v_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_4_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_attn2_to_k_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_4_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_attn2_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_4_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_4_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_attn2_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_4_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_attn2_to_v_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_4_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_ff_net_0_proj_bias.bin", 10240)
    input_blocks_7_1_transformer_blocks_4_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_7_1_transformer_blocks_4_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_ff_net_2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_4_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_ff_net_2_weight.bin", 6553600)
    input_blocks_7_1_transformer_blocks_4_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_norm1_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_4_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_norm1_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_4_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_norm2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_4_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_norm2_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_4_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_norm3_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_4_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_4_norm3_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_5_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_attn1_to_k_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_5_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_attn1_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_5_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_5_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_attn1_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_5_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_attn1_to_v_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_5_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_attn2_to_k_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_5_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_attn2_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_5_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_5_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_attn2_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_5_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_attn2_to_v_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_5_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_ff_net_0_proj_bias.bin", 10240)
    input_blocks_7_1_transformer_blocks_5_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_7_1_transformer_blocks_5_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_ff_net_2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_5_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_ff_net_2_weight.bin", 6553600)
    input_blocks_7_1_transformer_blocks_5_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_norm1_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_5_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_norm1_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_5_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_norm2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_5_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_norm2_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_5_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_norm3_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_5_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_5_norm3_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_6_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_attn1_to_k_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_6_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_attn1_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_6_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_6_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_attn1_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_6_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_attn1_to_v_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_6_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_attn2_to_k_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_6_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_attn2_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_6_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_6_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_attn2_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_6_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_attn2_to_v_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_6_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_ff_net_0_proj_bias.bin", 10240)
    input_blocks_7_1_transformer_blocks_6_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_7_1_transformer_blocks_6_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_ff_net_2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_6_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_ff_net_2_weight.bin", 6553600)
    input_blocks_7_1_transformer_blocks_6_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_norm1_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_6_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_norm1_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_6_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_norm2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_6_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_norm2_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_6_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_norm3_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_6_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_6_norm3_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_7_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_attn1_to_k_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_7_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_attn1_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_7_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_7_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_attn1_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_7_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_attn1_to_v_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_7_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_attn2_to_k_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_7_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_attn2_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_7_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_7_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_attn2_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_7_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_attn2_to_v_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_7_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_ff_net_0_proj_bias.bin", 10240)
    input_blocks_7_1_transformer_blocks_7_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_7_1_transformer_blocks_7_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_ff_net_2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_7_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_ff_net_2_weight.bin", 6553600)
    input_blocks_7_1_transformer_blocks_7_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_norm1_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_7_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_norm1_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_7_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_norm2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_7_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_norm2_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_7_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_norm3_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_7_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_7_norm3_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_8_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_attn1_to_k_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_8_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_attn1_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_8_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_8_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_attn1_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_8_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_attn1_to_v_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_8_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_attn2_to_k_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_8_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_attn2_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_8_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_8_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_attn2_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_8_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_attn2_to_v_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_8_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_ff_net_0_proj_bias.bin", 10240)
    input_blocks_7_1_transformer_blocks_8_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_7_1_transformer_blocks_8_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_ff_net_2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_8_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_ff_net_2_weight.bin", 6553600)
    input_blocks_7_1_transformer_blocks_8_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_norm1_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_8_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_norm1_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_8_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_norm2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_8_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_norm2_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_8_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_norm3_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_8_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_8_norm3_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_9_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_attn1_to_k_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_9_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_attn1_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_9_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_9_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_attn1_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_9_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_attn1_to_v_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_9_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_attn2_to_k_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_9_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_attn2_to_out_0_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_9_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_9_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_attn2_to_q_weight.bin", 1638400)
    input_blocks_7_1_transformer_blocks_9_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_attn2_to_v_weight.bin", 2621440)
    input_blocks_7_1_transformer_blocks_9_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_ff_net_0_proj_bias.bin", 10240)
    input_blocks_7_1_transformer_blocks_9_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_7_1_transformer_blocks_9_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_ff_net_2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_9_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_ff_net_2_weight.bin", 6553600)
    input_blocks_7_1_transformer_blocks_9_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_norm1_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_9_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_norm1_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_9_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_norm2_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_9_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_norm2_weight.bin", 1280)
    input_blocks_7_1_transformer_blocks_9_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_norm3_bias.bin", 1280)
    input_blocks_7_1_transformer_blocks_9_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_7_1_transformer_blocks_9_norm3_weight.bin", 1280)
    input_blocks_8_0_emb_layers_1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_0_emb_layers_1_bias.bin", 1280)
    input_blocks_8_0_emb_layers_1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_0_emb_layers_1_weight.bin", 1638400)
    input_blocks_8_0_in_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_0_in_layers_0_bias.bin", 1280)
    input_blocks_8_0_in_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_0_in_layers_0_weight.bin", 1280)
    input_blocks_8_0_in_layers_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_0_in_layers_2_bias.bin", 1280)
    input_blocks_8_0_in_layers_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_0_in_layers_2_weight.bin", 14745600)
    input_blocks_8_0_out_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_0_out_layers_0_bias.bin", 1280)
    input_blocks_8_0_out_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_0_out_layers_0_weight.bin", 1280)
    input_blocks_8_0_out_layers_3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_0_out_layers_3_bias.bin", 1280)
    input_blocks_8_0_out_layers_3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_0_out_layers_3_weight.bin", 14745600)
    input_blocks_8_1_norm_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_norm_bias.bin", 1280)
    input_blocks_8_1_norm_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_norm_weight.bin", 1280)
    input_blocks_8_1_proj_in_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_proj_in_bias.bin", 1280)
    input_blocks_8_1_proj_in_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_proj_in_weight.bin", 1638400)
    input_blocks_8_1_proj_out_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_proj_out_bias.bin", 1280)
    input_blocks_8_1_proj_out_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_proj_out_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_0_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_attn1_to_k_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_0_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_attn1_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_0_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_0_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_attn1_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_0_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_attn1_to_v_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_0_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_attn2_to_k_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_0_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_attn2_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_0_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_0_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_attn2_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_0_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_attn2_to_v_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_0_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_ff_net_0_proj_bias.bin", 10240)
    input_blocks_8_1_transformer_blocks_0_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_8_1_transformer_blocks_0_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_ff_net_2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_0_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_ff_net_2_weight.bin", 6553600)
    input_blocks_8_1_transformer_blocks_0_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_norm1_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_0_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_norm1_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_0_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_norm2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_0_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_norm2_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_0_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_norm3_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_0_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_0_norm3_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_1_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_attn1_to_k_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_1_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_attn1_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_1_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_1_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_attn1_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_1_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_attn1_to_v_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_1_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_attn2_to_k_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_1_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_attn2_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_1_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_1_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_attn2_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_1_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_attn2_to_v_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_1_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_ff_net_0_proj_bias.bin", 10240)
    input_blocks_8_1_transformer_blocks_1_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_8_1_transformer_blocks_1_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_ff_net_2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_1_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_ff_net_2_weight.bin", 6553600)
    input_blocks_8_1_transformer_blocks_1_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_norm1_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_1_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_norm1_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_1_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_norm2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_1_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_norm2_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_1_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_norm3_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_1_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_1_norm3_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_2_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_attn1_to_k_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_2_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_attn1_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_2_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_2_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_attn1_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_2_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_attn1_to_v_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_2_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_attn2_to_k_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_2_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_attn2_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_2_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_2_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_attn2_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_2_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_attn2_to_v_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_2_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_ff_net_0_proj_bias.bin", 10240)
    input_blocks_8_1_transformer_blocks_2_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_8_1_transformer_blocks_2_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_ff_net_2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_2_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_ff_net_2_weight.bin", 6553600)
    input_blocks_8_1_transformer_blocks_2_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_norm1_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_2_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_norm1_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_2_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_norm2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_2_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_norm2_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_2_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_norm3_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_2_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_2_norm3_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_3_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_attn1_to_k_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_3_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_attn1_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_3_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_3_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_attn1_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_3_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_attn1_to_v_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_3_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_attn2_to_k_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_3_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_attn2_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_3_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_3_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_attn2_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_3_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_attn2_to_v_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_3_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_ff_net_0_proj_bias.bin", 10240)
    input_blocks_8_1_transformer_blocks_3_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_8_1_transformer_blocks_3_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_ff_net_2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_3_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_ff_net_2_weight.bin", 6553600)
    input_blocks_8_1_transformer_blocks_3_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_norm1_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_3_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_norm1_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_3_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_norm2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_3_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_norm2_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_3_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_norm3_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_3_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_3_norm3_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_4_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_attn1_to_k_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_4_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_attn1_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_4_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_4_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_attn1_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_4_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_attn1_to_v_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_4_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_attn2_to_k_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_4_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_attn2_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_4_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_4_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_attn2_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_4_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_attn2_to_v_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_4_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_ff_net_0_proj_bias.bin", 10240)
    input_blocks_8_1_transformer_blocks_4_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_8_1_transformer_blocks_4_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_ff_net_2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_4_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_ff_net_2_weight.bin", 6553600)
    input_blocks_8_1_transformer_blocks_4_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_norm1_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_4_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_norm1_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_4_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_norm2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_4_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_norm2_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_4_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_norm3_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_4_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_4_norm3_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_5_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_attn1_to_k_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_5_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_attn1_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_5_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_5_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_attn1_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_5_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_attn1_to_v_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_5_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_attn2_to_k_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_5_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_attn2_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_5_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_5_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_attn2_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_5_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_attn2_to_v_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_5_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_ff_net_0_proj_bias.bin", 10240)
    input_blocks_8_1_transformer_blocks_5_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_8_1_transformer_blocks_5_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_ff_net_2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_5_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_ff_net_2_weight.bin", 6553600)
    input_blocks_8_1_transformer_blocks_5_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_norm1_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_5_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_norm1_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_5_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_norm2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_5_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_norm2_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_5_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_norm3_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_5_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_5_norm3_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_6_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_attn1_to_k_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_6_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_attn1_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_6_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_6_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_attn1_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_6_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_attn1_to_v_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_6_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_attn2_to_k_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_6_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_attn2_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_6_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_6_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_attn2_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_6_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_attn2_to_v_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_6_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_ff_net_0_proj_bias.bin", 10240)
    input_blocks_8_1_transformer_blocks_6_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_8_1_transformer_blocks_6_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_ff_net_2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_6_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_ff_net_2_weight.bin", 6553600)
    input_blocks_8_1_transformer_blocks_6_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_norm1_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_6_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_norm1_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_6_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_norm2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_6_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_norm2_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_6_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_norm3_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_6_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_6_norm3_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_7_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_attn1_to_k_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_7_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_attn1_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_7_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_7_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_attn1_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_7_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_attn1_to_v_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_7_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_attn2_to_k_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_7_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_attn2_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_7_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_7_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_attn2_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_7_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_attn2_to_v_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_7_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_ff_net_0_proj_bias.bin", 10240)
    input_blocks_8_1_transformer_blocks_7_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_8_1_transformer_blocks_7_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_ff_net_2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_7_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_ff_net_2_weight.bin", 6553600)
    input_blocks_8_1_transformer_blocks_7_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_norm1_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_7_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_norm1_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_7_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_norm2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_7_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_norm2_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_7_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_norm3_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_7_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_7_norm3_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_8_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_attn1_to_k_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_8_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_attn1_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_8_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_8_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_attn1_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_8_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_attn1_to_v_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_8_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_attn2_to_k_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_8_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_attn2_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_8_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_8_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_attn2_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_8_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_attn2_to_v_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_8_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_ff_net_0_proj_bias.bin", 10240)
    input_blocks_8_1_transformer_blocks_8_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_8_1_transformer_blocks_8_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_ff_net_2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_8_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_ff_net_2_weight.bin", 6553600)
    input_blocks_8_1_transformer_blocks_8_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_norm1_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_8_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_norm1_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_8_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_norm2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_8_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_norm2_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_8_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_norm3_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_8_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_8_norm3_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_9_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_attn1_to_k_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_9_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_attn1_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_9_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_attn1_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_9_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_attn1_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_9_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_attn1_to_v_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_9_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_attn2_to_k_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_9_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_attn2_to_out_0_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_9_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_attn2_to_out_0_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_9_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_attn2_to_q_weight.bin", 1638400)
    input_blocks_8_1_transformer_blocks_9_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_attn2_to_v_weight.bin", 2621440)
    input_blocks_8_1_transformer_blocks_9_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_ff_net_0_proj_bias.bin", 10240)
    input_blocks_8_1_transformer_blocks_9_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_ff_net_0_proj_weight.bin", 13107200)
    input_blocks_8_1_transformer_blocks_9_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_ff_net_2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_9_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_ff_net_2_weight.bin", 6553600)
    input_blocks_8_1_transformer_blocks_9_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_norm1_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_9_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_norm1_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_9_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_norm2_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_9_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_norm2_weight.bin", 1280)
    input_blocks_8_1_transformer_blocks_9_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_norm3_bias.bin", 1280)
    input_blocks_8_1_transformer_blocks_9_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_input_blocks_8_1_transformer_blocks_9_norm3_weight.bin", 1280)
    label_emb_0_0_bias = load_bin(weights_dir + "/model_diffusion_model_label_emb_0_0_bias.bin", 1280)
    label_emb_0_0_weight = load_bin(weights_dir + "/model_diffusion_model_label_emb_0_0_weight.bin", 3604480)
    label_emb_0_2_bias = load_bin(weights_dir + "/model_diffusion_model_label_emb_0_2_bias.bin", 1280)
    label_emb_0_2_weight = load_bin(weights_dir + "/model_diffusion_model_label_emb_0_2_weight.bin", 1638400)
    middle_block_0_emb_layers_1_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_0_emb_layers_1_bias.bin", 1280)
    middle_block_0_emb_layers_1_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_0_emb_layers_1_weight.bin", 1638400)
    middle_block_0_in_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_0_in_layers_0_bias.bin", 1280)
    middle_block_0_in_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_0_in_layers_0_weight.bin", 1280)
    middle_block_0_in_layers_2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_0_in_layers_2_bias.bin", 1280)
    middle_block_0_in_layers_2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_0_in_layers_2_weight.bin", 14745600)
    middle_block_0_out_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_0_out_layers_0_bias.bin", 1280)
    middle_block_0_out_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_0_out_layers_0_weight.bin", 1280)
    middle_block_0_out_layers_3_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_0_out_layers_3_bias.bin", 1280)
    middle_block_0_out_layers_3_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_0_out_layers_3_weight.bin", 14745600)
    middle_block_1_norm_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_norm_bias.bin", 1280)
    middle_block_1_norm_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_norm_weight.bin", 1280)
    middle_block_1_proj_in_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_proj_in_bias.bin", 1280)
    middle_block_1_proj_in_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_proj_in_weight.bin", 1638400)
    middle_block_1_proj_out_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_proj_out_bias.bin", 1280)
    middle_block_1_proj_out_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_proj_out_weight.bin", 1638400)
    middle_block_1_transformer_blocks_0_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_attn1_to_k_weight.bin", 1638400)
    middle_block_1_transformer_blocks_0_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_attn1_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_0_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_attn1_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_0_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_attn1_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_0_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_attn1_to_v_weight.bin", 1638400)
    middle_block_1_transformer_blocks_0_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_attn2_to_k_weight.bin", 2621440)
    middle_block_1_transformer_blocks_0_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_attn2_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_0_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_attn2_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_0_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_attn2_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_0_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_attn2_to_v_weight.bin", 2621440)
    middle_block_1_transformer_blocks_0_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_ff_net_0_proj_bias.bin", 10240)
    middle_block_1_transformer_blocks_0_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_ff_net_0_proj_weight.bin", 13107200)
    middle_block_1_transformer_blocks_0_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_ff_net_2_bias.bin", 1280)
    middle_block_1_transformer_blocks_0_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_ff_net_2_weight.bin", 6553600)
    middle_block_1_transformer_blocks_0_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_norm1_bias.bin", 1280)
    middle_block_1_transformer_blocks_0_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_norm1_weight.bin", 1280)
    middle_block_1_transformer_blocks_0_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_norm2_bias.bin", 1280)
    middle_block_1_transformer_blocks_0_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_norm2_weight.bin", 1280)
    middle_block_1_transformer_blocks_0_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_norm3_bias.bin", 1280)
    middle_block_1_transformer_blocks_0_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_0_norm3_weight.bin", 1280)
    middle_block_1_transformer_blocks_1_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_attn1_to_k_weight.bin", 1638400)
    middle_block_1_transformer_blocks_1_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_attn1_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_1_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_attn1_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_1_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_attn1_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_1_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_attn1_to_v_weight.bin", 1638400)
    middle_block_1_transformer_blocks_1_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_attn2_to_k_weight.bin", 2621440)
    middle_block_1_transformer_blocks_1_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_attn2_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_1_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_attn2_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_1_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_attn2_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_1_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_attn2_to_v_weight.bin", 2621440)
    middle_block_1_transformer_blocks_1_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_ff_net_0_proj_bias.bin", 10240)
    middle_block_1_transformer_blocks_1_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_ff_net_0_proj_weight.bin", 13107200)
    middle_block_1_transformer_blocks_1_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_ff_net_2_bias.bin", 1280)
    middle_block_1_transformer_blocks_1_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_ff_net_2_weight.bin", 6553600)
    middle_block_1_transformer_blocks_1_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_norm1_bias.bin", 1280)
    middle_block_1_transformer_blocks_1_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_norm1_weight.bin", 1280)
    middle_block_1_transformer_blocks_1_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_norm2_bias.bin", 1280)
    middle_block_1_transformer_blocks_1_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_norm2_weight.bin", 1280)
    middle_block_1_transformer_blocks_1_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_norm3_bias.bin", 1280)
    middle_block_1_transformer_blocks_1_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_1_norm3_weight.bin", 1280)
    middle_block_1_transformer_blocks_2_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_attn1_to_k_weight.bin", 1638400)
    middle_block_1_transformer_blocks_2_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_attn1_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_2_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_attn1_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_2_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_attn1_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_2_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_attn1_to_v_weight.bin", 1638400)
    middle_block_1_transformer_blocks_2_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_attn2_to_k_weight.bin", 2621440)
    middle_block_1_transformer_blocks_2_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_attn2_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_2_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_attn2_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_2_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_attn2_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_2_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_attn2_to_v_weight.bin", 2621440)
    middle_block_1_transformer_blocks_2_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_ff_net_0_proj_bias.bin", 10240)
    middle_block_1_transformer_blocks_2_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_ff_net_0_proj_weight.bin", 13107200)
    middle_block_1_transformer_blocks_2_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_ff_net_2_bias.bin", 1280)
    middle_block_1_transformer_blocks_2_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_ff_net_2_weight.bin", 6553600)
    middle_block_1_transformer_blocks_2_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_norm1_bias.bin", 1280)
    middle_block_1_transformer_blocks_2_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_norm1_weight.bin", 1280)
    middle_block_1_transformer_blocks_2_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_norm2_bias.bin", 1280)
    middle_block_1_transformer_blocks_2_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_norm2_weight.bin", 1280)
    middle_block_1_transformer_blocks_2_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_norm3_bias.bin", 1280)
    middle_block_1_transformer_blocks_2_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_2_norm3_weight.bin", 1280)
    middle_block_1_transformer_blocks_3_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_attn1_to_k_weight.bin", 1638400)
    middle_block_1_transformer_blocks_3_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_attn1_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_3_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_attn1_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_3_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_attn1_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_3_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_attn1_to_v_weight.bin", 1638400)
    middle_block_1_transformer_blocks_3_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_attn2_to_k_weight.bin", 2621440)
    middle_block_1_transformer_blocks_3_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_attn2_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_3_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_attn2_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_3_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_attn2_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_3_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_attn2_to_v_weight.bin", 2621440)
    middle_block_1_transformer_blocks_3_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_ff_net_0_proj_bias.bin", 10240)
    middle_block_1_transformer_blocks_3_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_ff_net_0_proj_weight.bin", 13107200)
    middle_block_1_transformer_blocks_3_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_ff_net_2_bias.bin", 1280)
    middle_block_1_transformer_blocks_3_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_ff_net_2_weight.bin", 6553600)
    middle_block_1_transformer_blocks_3_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_norm1_bias.bin", 1280)
    middle_block_1_transformer_blocks_3_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_norm1_weight.bin", 1280)
    middle_block_1_transformer_blocks_3_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_norm2_bias.bin", 1280)
    middle_block_1_transformer_blocks_3_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_norm2_weight.bin", 1280)
    middle_block_1_transformer_blocks_3_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_norm3_bias.bin", 1280)
    middle_block_1_transformer_blocks_3_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_3_norm3_weight.bin", 1280)
    middle_block_1_transformer_blocks_4_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_attn1_to_k_weight.bin", 1638400)
    middle_block_1_transformer_blocks_4_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_attn1_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_4_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_attn1_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_4_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_attn1_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_4_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_attn1_to_v_weight.bin", 1638400)
    middle_block_1_transformer_blocks_4_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_attn2_to_k_weight.bin", 2621440)
    middle_block_1_transformer_blocks_4_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_attn2_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_4_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_attn2_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_4_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_attn2_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_4_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_attn2_to_v_weight.bin", 2621440)
    middle_block_1_transformer_blocks_4_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_ff_net_0_proj_bias.bin", 10240)
    middle_block_1_transformer_blocks_4_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_ff_net_0_proj_weight.bin", 13107200)
    middle_block_1_transformer_blocks_4_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_ff_net_2_bias.bin", 1280)
    middle_block_1_transformer_blocks_4_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_ff_net_2_weight.bin", 6553600)
    middle_block_1_transformer_blocks_4_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_norm1_bias.bin", 1280)
    middle_block_1_transformer_blocks_4_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_norm1_weight.bin", 1280)
    middle_block_1_transformer_blocks_4_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_norm2_bias.bin", 1280)
    middle_block_1_transformer_blocks_4_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_norm2_weight.bin", 1280)
    middle_block_1_transformer_blocks_4_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_norm3_bias.bin", 1280)
    middle_block_1_transformer_blocks_4_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_4_norm3_weight.bin", 1280)
    middle_block_1_transformer_blocks_5_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_attn1_to_k_weight.bin", 1638400)
    middle_block_1_transformer_blocks_5_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_attn1_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_5_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_attn1_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_5_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_attn1_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_5_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_attn1_to_v_weight.bin", 1638400)
    middle_block_1_transformer_blocks_5_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_attn2_to_k_weight.bin", 2621440)
    middle_block_1_transformer_blocks_5_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_attn2_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_5_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_attn2_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_5_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_attn2_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_5_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_attn2_to_v_weight.bin", 2621440)
    middle_block_1_transformer_blocks_5_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_ff_net_0_proj_bias.bin", 10240)
    middle_block_1_transformer_blocks_5_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_ff_net_0_proj_weight.bin", 13107200)
    middle_block_1_transformer_blocks_5_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_ff_net_2_bias.bin", 1280)
    middle_block_1_transformer_blocks_5_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_ff_net_2_weight.bin", 6553600)
    middle_block_1_transformer_blocks_5_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_norm1_bias.bin", 1280)
    middle_block_1_transformer_blocks_5_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_norm1_weight.bin", 1280)
    middle_block_1_transformer_blocks_5_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_norm2_bias.bin", 1280)
    middle_block_1_transformer_blocks_5_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_norm2_weight.bin", 1280)
    middle_block_1_transformer_blocks_5_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_norm3_bias.bin", 1280)
    middle_block_1_transformer_blocks_5_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_5_norm3_weight.bin", 1280)
    middle_block_1_transformer_blocks_6_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_attn1_to_k_weight.bin", 1638400)
    middle_block_1_transformer_blocks_6_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_attn1_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_6_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_attn1_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_6_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_attn1_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_6_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_attn1_to_v_weight.bin", 1638400)
    middle_block_1_transformer_blocks_6_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_attn2_to_k_weight.bin", 2621440)
    middle_block_1_transformer_blocks_6_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_attn2_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_6_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_attn2_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_6_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_attn2_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_6_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_attn2_to_v_weight.bin", 2621440)
    middle_block_1_transformer_blocks_6_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_ff_net_0_proj_bias.bin", 10240)
    middle_block_1_transformer_blocks_6_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_ff_net_0_proj_weight.bin", 13107200)
    middle_block_1_transformer_blocks_6_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_ff_net_2_bias.bin", 1280)
    middle_block_1_transformer_blocks_6_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_ff_net_2_weight.bin", 6553600)
    middle_block_1_transformer_blocks_6_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_norm1_bias.bin", 1280)
    middle_block_1_transformer_blocks_6_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_norm1_weight.bin", 1280)
    middle_block_1_transformer_blocks_6_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_norm2_bias.bin", 1280)
    middle_block_1_transformer_blocks_6_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_norm2_weight.bin", 1280)
    middle_block_1_transformer_blocks_6_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_norm3_bias.bin", 1280)
    middle_block_1_transformer_blocks_6_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_6_norm3_weight.bin", 1280)
    middle_block_1_transformer_blocks_7_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_attn1_to_k_weight.bin", 1638400)
    middle_block_1_transformer_blocks_7_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_attn1_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_7_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_attn1_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_7_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_attn1_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_7_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_attn1_to_v_weight.bin", 1638400)
    middle_block_1_transformer_blocks_7_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_attn2_to_k_weight.bin", 2621440)
    middle_block_1_transformer_blocks_7_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_attn2_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_7_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_attn2_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_7_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_attn2_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_7_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_attn2_to_v_weight.bin", 2621440)
    middle_block_1_transformer_blocks_7_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_ff_net_0_proj_bias.bin", 10240)
    middle_block_1_transformer_blocks_7_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_ff_net_0_proj_weight.bin", 13107200)
    middle_block_1_transformer_blocks_7_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_ff_net_2_bias.bin", 1280)
    middle_block_1_transformer_blocks_7_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_ff_net_2_weight.bin", 6553600)
    middle_block_1_transformer_blocks_7_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_norm1_bias.bin", 1280)
    middle_block_1_transformer_blocks_7_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_norm1_weight.bin", 1280)
    middle_block_1_transformer_blocks_7_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_norm2_bias.bin", 1280)
    middle_block_1_transformer_blocks_7_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_norm2_weight.bin", 1280)
    middle_block_1_transformer_blocks_7_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_norm3_bias.bin", 1280)
    middle_block_1_transformer_blocks_7_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_7_norm3_weight.bin", 1280)
    middle_block_1_transformer_blocks_8_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_attn1_to_k_weight.bin", 1638400)
    middle_block_1_transformer_blocks_8_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_attn1_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_8_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_attn1_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_8_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_attn1_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_8_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_attn1_to_v_weight.bin", 1638400)
    middle_block_1_transformer_blocks_8_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_attn2_to_k_weight.bin", 2621440)
    middle_block_1_transformer_blocks_8_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_attn2_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_8_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_attn2_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_8_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_attn2_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_8_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_attn2_to_v_weight.bin", 2621440)
    middle_block_1_transformer_blocks_8_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_ff_net_0_proj_bias.bin", 10240)
    middle_block_1_transformer_blocks_8_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_ff_net_0_proj_weight.bin", 13107200)
    middle_block_1_transformer_blocks_8_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_ff_net_2_bias.bin", 1280)
    middle_block_1_transformer_blocks_8_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_ff_net_2_weight.bin", 6553600)
    middle_block_1_transformer_blocks_8_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_norm1_bias.bin", 1280)
    middle_block_1_transformer_blocks_8_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_norm1_weight.bin", 1280)
    middle_block_1_transformer_blocks_8_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_norm2_bias.bin", 1280)
    middle_block_1_transformer_blocks_8_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_norm2_weight.bin", 1280)
    middle_block_1_transformer_blocks_8_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_norm3_bias.bin", 1280)
    middle_block_1_transformer_blocks_8_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_8_norm3_weight.bin", 1280)
    middle_block_1_transformer_blocks_9_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_attn1_to_k_weight.bin", 1638400)
    middle_block_1_transformer_blocks_9_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_attn1_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_9_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_attn1_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_9_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_attn1_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_9_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_attn1_to_v_weight.bin", 1638400)
    middle_block_1_transformer_blocks_9_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_attn2_to_k_weight.bin", 2621440)
    middle_block_1_transformer_blocks_9_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_attn2_to_out_0_bias.bin", 1280)
    middle_block_1_transformer_blocks_9_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_attn2_to_out_0_weight.bin", 1638400)
    middle_block_1_transformer_blocks_9_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_attn2_to_q_weight.bin", 1638400)
    middle_block_1_transformer_blocks_9_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_attn2_to_v_weight.bin", 2621440)
    middle_block_1_transformer_blocks_9_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_ff_net_0_proj_bias.bin", 10240)
    middle_block_1_transformer_blocks_9_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_ff_net_0_proj_weight.bin", 13107200)
    middle_block_1_transformer_blocks_9_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_ff_net_2_bias.bin", 1280)
    middle_block_1_transformer_blocks_9_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_ff_net_2_weight.bin", 6553600)
    middle_block_1_transformer_blocks_9_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_norm1_bias.bin", 1280)
    middle_block_1_transformer_blocks_9_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_norm1_weight.bin", 1280)
    middle_block_1_transformer_blocks_9_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_norm2_bias.bin", 1280)
    middle_block_1_transformer_blocks_9_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_norm2_weight.bin", 1280)
    middle_block_1_transformer_blocks_9_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_norm3_bias.bin", 1280)
    middle_block_1_transformer_blocks_9_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_1_transformer_blocks_9_norm3_weight.bin", 1280)
    middle_block_2_emb_layers_1_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_2_emb_layers_1_bias.bin", 1280)
    middle_block_2_emb_layers_1_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_2_emb_layers_1_weight.bin", 1638400)
    middle_block_2_in_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_2_in_layers_0_bias.bin", 1280)
    middle_block_2_in_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_2_in_layers_0_weight.bin", 1280)
    middle_block_2_in_layers_2_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_2_in_layers_2_bias.bin", 1280)
    middle_block_2_in_layers_2_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_2_in_layers_2_weight.bin", 14745600)
    middle_block_2_out_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_2_out_layers_0_bias.bin", 1280)
    middle_block_2_out_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_2_out_layers_0_weight.bin", 1280)
    middle_block_2_out_layers_3_bias = load_bin(weights_dir + "/model_diffusion_model_middle_block_2_out_layers_3_bias.bin", 1280)
    middle_block_2_out_layers_3_weight = load_bin(weights_dir + "/model_diffusion_model_middle_block_2_out_layers_3_weight.bin", 14745600)
    out_0_bias = load_bin(weights_dir + "/model_diffusion_model_out_0_bias.bin", 320)
    out_0_weight = load_bin(weights_dir + "/model_diffusion_model_out_0_weight.bin", 320)
    out_2_bias = load_bin(weights_dir + "/model_diffusion_model_out_2_bias.bin", 4)
    out_2_weight = load_bin(weights_dir + "/model_diffusion_model_out_2_weight.bin", 11520)
    output_blocks_0_0_emb_layers_1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_0_emb_layers_1_bias.bin", 1280)
    output_blocks_0_0_emb_layers_1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_0_emb_layers_1_weight.bin", 1638400)
    output_blocks_0_0_in_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_0_in_layers_0_bias.bin", 2560)
    output_blocks_0_0_in_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_0_in_layers_0_weight.bin", 2560)
    output_blocks_0_0_in_layers_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_0_in_layers_2_bias.bin", 1280)
    output_blocks_0_0_in_layers_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_0_in_layers_2_weight.bin", 29491200)
    output_blocks_0_0_out_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_0_out_layers_0_bias.bin", 1280)
    output_blocks_0_0_out_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_0_out_layers_0_weight.bin", 1280)
    output_blocks_0_0_out_layers_3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_0_out_layers_3_bias.bin", 1280)
    output_blocks_0_0_out_layers_3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_0_out_layers_3_weight.bin", 14745600)
    output_blocks_0_0_skip_connection_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_0_skip_connection_bias.bin", 1280)
    output_blocks_0_0_skip_connection_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_0_skip_connection_weight.bin", 3276800)
    output_blocks_0_1_norm_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_norm_bias.bin", 1280)
    output_blocks_0_1_norm_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_norm_weight.bin", 1280)
    output_blocks_0_1_proj_in_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_proj_in_bias.bin", 1280)
    output_blocks_0_1_proj_in_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_proj_in_weight.bin", 1638400)
    output_blocks_0_1_proj_out_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_proj_out_bias.bin", 1280)
    output_blocks_0_1_proj_out_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_proj_out_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_0_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_attn1_to_k_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_0_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_attn1_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_0_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_0_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_attn1_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_0_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_attn1_to_v_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_0_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_attn2_to_k_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_0_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_attn2_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_0_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_0_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_attn2_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_0_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_attn2_to_v_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_0_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_ff_net_0_proj_bias.bin", 10240)
    output_blocks_0_1_transformer_blocks_0_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_0_1_transformer_blocks_0_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_ff_net_2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_0_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_ff_net_2_weight.bin", 6553600)
    output_blocks_0_1_transformer_blocks_0_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_norm1_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_0_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_norm1_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_0_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_norm2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_0_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_norm2_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_0_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_norm3_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_0_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_0_norm3_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_1_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_attn1_to_k_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_1_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_attn1_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_1_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_1_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_attn1_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_1_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_attn1_to_v_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_1_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_attn2_to_k_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_1_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_attn2_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_1_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_1_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_attn2_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_1_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_attn2_to_v_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_1_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_ff_net_0_proj_bias.bin", 10240)
    output_blocks_0_1_transformer_blocks_1_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_0_1_transformer_blocks_1_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_ff_net_2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_1_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_ff_net_2_weight.bin", 6553600)
    output_blocks_0_1_transformer_blocks_1_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_norm1_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_1_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_norm1_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_1_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_norm2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_1_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_norm2_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_1_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_norm3_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_1_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_1_norm3_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_2_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_attn1_to_k_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_2_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_attn1_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_2_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_2_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_attn1_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_2_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_attn1_to_v_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_2_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_attn2_to_k_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_2_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_attn2_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_2_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_2_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_attn2_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_2_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_attn2_to_v_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_2_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_ff_net_0_proj_bias.bin", 10240)
    output_blocks_0_1_transformer_blocks_2_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_0_1_transformer_blocks_2_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_ff_net_2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_2_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_ff_net_2_weight.bin", 6553600)
    output_blocks_0_1_transformer_blocks_2_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_norm1_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_2_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_norm1_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_2_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_norm2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_2_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_norm2_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_2_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_norm3_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_2_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_2_norm3_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_3_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_attn1_to_k_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_3_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_attn1_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_3_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_3_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_attn1_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_3_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_attn1_to_v_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_3_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_attn2_to_k_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_3_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_attn2_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_3_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_3_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_attn2_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_3_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_attn2_to_v_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_3_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_ff_net_0_proj_bias.bin", 10240)
    output_blocks_0_1_transformer_blocks_3_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_0_1_transformer_blocks_3_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_ff_net_2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_3_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_ff_net_2_weight.bin", 6553600)
    output_blocks_0_1_transformer_blocks_3_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_norm1_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_3_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_norm1_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_3_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_norm2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_3_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_norm2_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_3_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_norm3_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_3_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_3_norm3_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_4_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_attn1_to_k_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_4_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_attn1_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_4_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_4_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_attn1_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_4_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_attn1_to_v_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_4_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_attn2_to_k_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_4_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_attn2_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_4_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_4_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_attn2_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_4_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_attn2_to_v_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_4_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_ff_net_0_proj_bias.bin", 10240)
    output_blocks_0_1_transformer_blocks_4_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_0_1_transformer_blocks_4_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_ff_net_2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_4_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_ff_net_2_weight.bin", 6553600)
    output_blocks_0_1_transformer_blocks_4_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_norm1_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_4_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_norm1_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_4_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_norm2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_4_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_norm2_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_4_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_norm3_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_4_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_4_norm3_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_5_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_attn1_to_k_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_5_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_attn1_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_5_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_5_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_attn1_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_5_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_attn1_to_v_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_5_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_attn2_to_k_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_5_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_attn2_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_5_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_5_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_attn2_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_5_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_attn2_to_v_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_5_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_ff_net_0_proj_bias.bin", 10240)
    output_blocks_0_1_transformer_blocks_5_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_0_1_transformer_blocks_5_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_ff_net_2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_5_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_ff_net_2_weight.bin", 6553600)
    output_blocks_0_1_transformer_blocks_5_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_norm1_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_5_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_norm1_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_5_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_norm2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_5_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_norm2_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_5_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_norm3_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_5_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_5_norm3_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_6_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_attn1_to_k_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_6_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_attn1_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_6_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_6_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_attn1_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_6_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_attn1_to_v_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_6_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_attn2_to_k_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_6_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_attn2_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_6_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_6_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_attn2_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_6_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_attn2_to_v_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_6_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_ff_net_0_proj_bias.bin", 10240)
    output_blocks_0_1_transformer_blocks_6_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_0_1_transformer_blocks_6_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_ff_net_2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_6_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_ff_net_2_weight.bin", 6553600)
    output_blocks_0_1_transformer_blocks_6_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_norm1_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_6_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_norm1_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_6_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_norm2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_6_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_norm2_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_6_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_norm3_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_6_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_6_norm3_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_7_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_attn1_to_k_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_7_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_attn1_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_7_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_7_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_attn1_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_7_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_attn1_to_v_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_7_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_attn2_to_k_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_7_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_attn2_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_7_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_7_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_attn2_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_7_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_attn2_to_v_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_7_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_ff_net_0_proj_bias.bin", 10240)
    output_blocks_0_1_transformer_blocks_7_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_0_1_transformer_blocks_7_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_ff_net_2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_7_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_ff_net_2_weight.bin", 6553600)
    output_blocks_0_1_transformer_blocks_7_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_norm1_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_7_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_norm1_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_7_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_norm2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_7_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_norm2_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_7_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_norm3_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_7_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_7_norm3_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_8_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_attn1_to_k_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_8_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_attn1_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_8_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_8_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_attn1_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_8_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_attn1_to_v_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_8_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_attn2_to_k_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_8_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_attn2_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_8_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_8_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_attn2_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_8_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_attn2_to_v_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_8_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_ff_net_0_proj_bias.bin", 10240)
    output_blocks_0_1_transformer_blocks_8_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_0_1_transformer_blocks_8_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_ff_net_2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_8_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_ff_net_2_weight.bin", 6553600)
    output_blocks_0_1_transformer_blocks_8_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_norm1_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_8_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_norm1_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_8_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_norm2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_8_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_norm2_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_8_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_norm3_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_8_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_8_norm3_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_9_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_attn1_to_k_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_9_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_attn1_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_9_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_9_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_attn1_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_9_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_attn1_to_v_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_9_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_attn2_to_k_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_9_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_attn2_to_out_0_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_9_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_9_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_attn2_to_q_weight.bin", 1638400)
    output_blocks_0_1_transformer_blocks_9_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_attn2_to_v_weight.bin", 2621440)
    output_blocks_0_1_transformer_blocks_9_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_ff_net_0_proj_bias.bin", 10240)
    output_blocks_0_1_transformer_blocks_9_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_0_1_transformer_blocks_9_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_ff_net_2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_9_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_ff_net_2_weight.bin", 6553600)
    output_blocks_0_1_transformer_blocks_9_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_norm1_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_9_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_norm1_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_9_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_norm2_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_9_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_norm2_weight.bin", 1280)
    output_blocks_0_1_transformer_blocks_9_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_norm3_bias.bin", 1280)
    output_blocks_0_1_transformer_blocks_9_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_0_1_transformer_blocks_9_norm3_weight.bin", 1280)
    output_blocks_1_0_emb_layers_1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_0_emb_layers_1_bias.bin", 1280)
    output_blocks_1_0_emb_layers_1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_0_emb_layers_1_weight.bin", 1638400)
    output_blocks_1_0_in_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_0_in_layers_0_bias.bin", 2560)
    output_blocks_1_0_in_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_0_in_layers_0_weight.bin", 2560)
    output_blocks_1_0_in_layers_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_0_in_layers_2_bias.bin", 1280)
    output_blocks_1_0_in_layers_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_0_in_layers_2_weight.bin", 29491200)
    output_blocks_1_0_out_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_0_out_layers_0_bias.bin", 1280)
    output_blocks_1_0_out_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_0_out_layers_0_weight.bin", 1280)
    output_blocks_1_0_out_layers_3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_0_out_layers_3_bias.bin", 1280)
    output_blocks_1_0_out_layers_3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_0_out_layers_3_weight.bin", 14745600)
    output_blocks_1_0_skip_connection_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_0_skip_connection_bias.bin", 1280)
    output_blocks_1_0_skip_connection_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_0_skip_connection_weight.bin", 3276800)
    output_blocks_1_1_norm_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_norm_bias.bin", 1280)
    output_blocks_1_1_norm_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_norm_weight.bin", 1280)
    output_blocks_1_1_proj_in_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_proj_in_bias.bin", 1280)
    output_blocks_1_1_proj_in_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_proj_in_weight.bin", 1638400)
    output_blocks_1_1_proj_out_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_proj_out_bias.bin", 1280)
    output_blocks_1_1_proj_out_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_proj_out_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_0_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_attn1_to_k_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_0_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_attn1_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_0_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_0_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_attn1_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_0_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_attn1_to_v_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_0_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_attn2_to_k_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_0_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_attn2_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_0_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_0_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_attn2_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_0_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_attn2_to_v_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_0_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_ff_net_0_proj_bias.bin", 10240)
    output_blocks_1_1_transformer_blocks_0_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_1_1_transformer_blocks_0_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_ff_net_2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_0_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_ff_net_2_weight.bin", 6553600)
    output_blocks_1_1_transformer_blocks_0_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_norm1_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_0_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_norm1_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_0_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_norm2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_0_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_norm2_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_0_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_norm3_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_0_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_0_norm3_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_1_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_attn1_to_k_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_1_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_attn1_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_1_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_1_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_attn1_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_1_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_attn1_to_v_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_1_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_attn2_to_k_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_1_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_attn2_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_1_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_1_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_attn2_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_1_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_attn2_to_v_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_1_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_ff_net_0_proj_bias.bin", 10240)
    output_blocks_1_1_transformer_blocks_1_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_1_1_transformer_blocks_1_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_ff_net_2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_1_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_ff_net_2_weight.bin", 6553600)
    output_blocks_1_1_transformer_blocks_1_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_norm1_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_1_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_norm1_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_1_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_norm2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_1_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_norm2_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_1_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_norm3_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_1_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_1_norm3_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_2_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_attn1_to_k_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_2_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_attn1_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_2_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_2_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_attn1_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_2_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_attn1_to_v_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_2_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_attn2_to_k_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_2_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_attn2_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_2_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_2_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_attn2_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_2_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_attn2_to_v_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_2_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_ff_net_0_proj_bias.bin", 10240)
    output_blocks_1_1_transformer_blocks_2_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_1_1_transformer_blocks_2_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_ff_net_2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_2_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_ff_net_2_weight.bin", 6553600)
    output_blocks_1_1_transformer_blocks_2_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_norm1_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_2_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_norm1_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_2_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_norm2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_2_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_norm2_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_2_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_norm3_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_2_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_2_norm3_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_3_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_attn1_to_k_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_3_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_attn1_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_3_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_3_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_attn1_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_3_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_attn1_to_v_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_3_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_attn2_to_k_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_3_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_attn2_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_3_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_3_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_attn2_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_3_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_attn2_to_v_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_3_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_ff_net_0_proj_bias.bin", 10240)
    output_blocks_1_1_transformer_blocks_3_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_1_1_transformer_blocks_3_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_ff_net_2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_3_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_ff_net_2_weight.bin", 6553600)
    output_blocks_1_1_transformer_blocks_3_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_norm1_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_3_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_norm1_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_3_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_norm2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_3_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_norm2_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_3_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_norm3_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_3_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_3_norm3_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_4_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_attn1_to_k_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_4_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_attn1_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_4_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_4_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_attn1_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_4_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_attn1_to_v_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_4_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_attn2_to_k_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_4_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_attn2_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_4_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_4_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_attn2_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_4_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_attn2_to_v_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_4_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_ff_net_0_proj_bias.bin", 10240)
    output_blocks_1_1_transformer_blocks_4_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_1_1_transformer_blocks_4_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_ff_net_2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_4_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_ff_net_2_weight.bin", 6553600)
    output_blocks_1_1_transformer_blocks_4_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_norm1_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_4_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_norm1_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_4_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_norm2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_4_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_norm2_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_4_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_norm3_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_4_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_4_norm3_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_5_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_attn1_to_k_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_5_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_attn1_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_5_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_5_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_attn1_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_5_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_attn1_to_v_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_5_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_attn2_to_k_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_5_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_attn2_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_5_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_5_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_attn2_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_5_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_attn2_to_v_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_5_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_ff_net_0_proj_bias.bin", 10240)
    output_blocks_1_1_transformer_blocks_5_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_1_1_transformer_blocks_5_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_ff_net_2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_5_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_ff_net_2_weight.bin", 6553600)
    output_blocks_1_1_transformer_blocks_5_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_norm1_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_5_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_norm1_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_5_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_norm2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_5_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_norm2_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_5_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_norm3_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_5_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_5_norm3_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_6_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_attn1_to_k_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_6_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_attn1_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_6_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_6_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_attn1_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_6_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_attn1_to_v_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_6_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_attn2_to_k_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_6_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_attn2_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_6_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_6_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_attn2_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_6_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_attn2_to_v_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_6_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_ff_net_0_proj_bias.bin", 10240)
    output_blocks_1_1_transformer_blocks_6_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_1_1_transformer_blocks_6_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_ff_net_2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_6_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_ff_net_2_weight.bin", 6553600)
    output_blocks_1_1_transformer_blocks_6_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_norm1_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_6_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_norm1_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_6_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_norm2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_6_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_norm2_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_6_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_norm3_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_6_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_6_norm3_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_7_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_attn1_to_k_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_7_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_attn1_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_7_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_7_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_attn1_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_7_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_attn1_to_v_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_7_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_attn2_to_k_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_7_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_attn2_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_7_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_7_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_attn2_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_7_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_attn2_to_v_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_7_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_ff_net_0_proj_bias.bin", 10240)
    output_blocks_1_1_transformer_blocks_7_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_1_1_transformer_blocks_7_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_ff_net_2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_7_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_ff_net_2_weight.bin", 6553600)
    output_blocks_1_1_transformer_blocks_7_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_norm1_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_7_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_norm1_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_7_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_norm2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_7_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_norm2_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_7_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_norm3_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_7_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_7_norm3_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_8_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_attn1_to_k_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_8_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_attn1_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_8_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_8_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_attn1_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_8_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_attn1_to_v_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_8_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_attn2_to_k_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_8_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_attn2_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_8_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_8_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_attn2_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_8_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_attn2_to_v_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_8_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_ff_net_0_proj_bias.bin", 10240)
    output_blocks_1_1_transformer_blocks_8_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_1_1_transformer_blocks_8_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_ff_net_2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_8_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_ff_net_2_weight.bin", 6553600)
    output_blocks_1_1_transformer_blocks_8_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_norm1_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_8_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_norm1_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_8_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_norm2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_8_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_norm2_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_8_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_norm3_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_8_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_8_norm3_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_9_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_attn1_to_k_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_9_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_attn1_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_9_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_9_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_attn1_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_9_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_attn1_to_v_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_9_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_attn2_to_k_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_9_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_attn2_to_out_0_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_9_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_9_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_attn2_to_q_weight.bin", 1638400)
    output_blocks_1_1_transformer_blocks_9_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_attn2_to_v_weight.bin", 2621440)
    output_blocks_1_1_transformer_blocks_9_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_ff_net_0_proj_bias.bin", 10240)
    output_blocks_1_1_transformer_blocks_9_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_1_1_transformer_blocks_9_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_ff_net_2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_9_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_ff_net_2_weight.bin", 6553600)
    output_blocks_1_1_transformer_blocks_9_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_norm1_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_9_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_norm1_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_9_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_norm2_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_9_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_norm2_weight.bin", 1280)
    output_blocks_1_1_transformer_blocks_9_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_norm3_bias.bin", 1280)
    output_blocks_1_1_transformer_blocks_9_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_1_1_transformer_blocks_9_norm3_weight.bin", 1280)
    output_blocks_2_0_emb_layers_1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_0_emb_layers_1_bias.bin", 1280)
    output_blocks_2_0_emb_layers_1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_0_emb_layers_1_weight.bin", 1638400)
    output_blocks_2_0_in_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_0_in_layers_0_bias.bin", 1920)
    output_blocks_2_0_in_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_0_in_layers_0_weight.bin", 1920)
    output_blocks_2_0_in_layers_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_0_in_layers_2_bias.bin", 1280)
    output_blocks_2_0_in_layers_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_0_in_layers_2_weight.bin", 22118400)
    output_blocks_2_0_out_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_0_out_layers_0_bias.bin", 1280)
    output_blocks_2_0_out_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_0_out_layers_0_weight.bin", 1280)
    output_blocks_2_0_out_layers_3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_0_out_layers_3_bias.bin", 1280)
    output_blocks_2_0_out_layers_3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_0_out_layers_3_weight.bin", 14745600)
    output_blocks_2_0_skip_connection_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_0_skip_connection_bias.bin", 1280)
    output_blocks_2_0_skip_connection_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_0_skip_connection_weight.bin", 2457600)
    output_blocks_2_1_norm_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_norm_bias.bin", 1280)
    output_blocks_2_1_norm_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_norm_weight.bin", 1280)
    output_blocks_2_1_proj_in_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_proj_in_bias.bin", 1280)
    output_blocks_2_1_proj_in_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_proj_in_weight.bin", 1638400)
    output_blocks_2_1_proj_out_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_proj_out_bias.bin", 1280)
    output_blocks_2_1_proj_out_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_proj_out_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_0_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_attn1_to_k_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_0_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_attn1_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_0_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_0_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_attn1_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_0_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_attn1_to_v_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_0_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_attn2_to_k_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_0_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_attn2_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_0_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_0_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_attn2_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_0_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_attn2_to_v_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_0_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_ff_net_0_proj_bias.bin", 10240)
    output_blocks_2_1_transformer_blocks_0_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_2_1_transformer_blocks_0_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_ff_net_2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_0_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_ff_net_2_weight.bin", 6553600)
    output_blocks_2_1_transformer_blocks_0_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_norm1_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_0_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_norm1_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_0_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_norm2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_0_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_norm2_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_0_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_norm3_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_0_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_0_norm3_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_1_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_attn1_to_k_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_1_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_attn1_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_1_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_1_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_attn1_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_1_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_attn1_to_v_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_1_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_attn2_to_k_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_1_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_attn2_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_1_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_1_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_attn2_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_1_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_attn2_to_v_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_1_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_ff_net_0_proj_bias.bin", 10240)
    output_blocks_2_1_transformer_blocks_1_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_2_1_transformer_blocks_1_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_ff_net_2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_1_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_ff_net_2_weight.bin", 6553600)
    output_blocks_2_1_transformer_blocks_1_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_norm1_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_1_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_norm1_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_1_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_norm2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_1_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_norm2_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_1_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_norm3_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_1_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_1_norm3_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_2_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_attn1_to_k_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_2_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_attn1_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_2_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_2_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_attn1_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_2_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_attn1_to_v_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_2_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_attn2_to_k_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_2_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_attn2_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_2_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_2_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_attn2_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_2_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_attn2_to_v_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_2_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_ff_net_0_proj_bias.bin", 10240)
    output_blocks_2_1_transformer_blocks_2_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_2_1_transformer_blocks_2_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_ff_net_2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_2_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_ff_net_2_weight.bin", 6553600)
    output_blocks_2_1_transformer_blocks_2_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_norm1_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_2_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_norm1_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_2_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_norm2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_2_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_norm2_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_2_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_norm3_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_2_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_2_norm3_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_3_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_attn1_to_k_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_3_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_attn1_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_3_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_3_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_attn1_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_3_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_attn1_to_v_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_3_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_attn2_to_k_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_3_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_attn2_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_3_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_3_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_attn2_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_3_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_attn2_to_v_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_3_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_ff_net_0_proj_bias.bin", 10240)
    output_blocks_2_1_transformer_blocks_3_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_2_1_transformer_blocks_3_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_ff_net_2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_3_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_ff_net_2_weight.bin", 6553600)
    output_blocks_2_1_transformer_blocks_3_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_norm1_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_3_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_norm1_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_3_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_norm2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_3_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_norm2_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_3_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_norm3_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_3_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_3_norm3_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_4_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_attn1_to_k_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_4_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_attn1_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_4_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_4_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_attn1_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_4_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_attn1_to_v_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_4_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_attn2_to_k_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_4_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_attn2_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_4_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_4_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_attn2_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_4_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_attn2_to_v_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_4_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_ff_net_0_proj_bias.bin", 10240)
    output_blocks_2_1_transformer_blocks_4_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_2_1_transformer_blocks_4_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_ff_net_2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_4_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_ff_net_2_weight.bin", 6553600)
    output_blocks_2_1_transformer_blocks_4_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_norm1_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_4_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_norm1_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_4_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_norm2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_4_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_norm2_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_4_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_norm3_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_4_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_4_norm3_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_5_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_attn1_to_k_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_5_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_attn1_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_5_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_5_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_attn1_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_5_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_attn1_to_v_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_5_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_attn2_to_k_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_5_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_attn2_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_5_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_5_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_attn2_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_5_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_attn2_to_v_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_5_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_ff_net_0_proj_bias.bin", 10240)
    output_blocks_2_1_transformer_blocks_5_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_2_1_transformer_blocks_5_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_ff_net_2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_5_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_ff_net_2_weight.bin", 6553600)
    output_blocks_2_1_transformer_blocks_5_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_norm1_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_5_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_norm1_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_5_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_norm2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_5_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_norm2_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_5_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_norm3_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_5_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_5_norm3_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_6_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_attn1_to_k_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_6_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_attn1_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_6_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_6_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_attn1_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_6_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_attn1_to_v_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_6_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_attn2_to_k_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_6_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_attn2_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_6_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_6_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_attn2_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_6_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_attn2_to_v_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_6_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_ff_net_0_proj_bias.bin", 10240)
    output_blocks_2_1_transformer_blocks_6_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_2_1_transformer_blocks_6_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_ff_net_2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_6_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_ff_net_2_weight.bin", 6553600)
    output_blocks_2_1_transformer_blocks_6_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_norm1_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_6_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_norm1_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_6_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_norm2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_6_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_norm2_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_6_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_norm3_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_6_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_6_norm3_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_7_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_attn1_to_k_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_7_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_attn1_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_7_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_7_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_attn1_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_7_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_attn1_to_v_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_7_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_attn2_to_k_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_7_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_attn2_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_7_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_7_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_attn2_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_7_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_attn2_to_v_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_7_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_ff_net_0_proj_bias.bin", 10240)
    output_blocks_2_1_transformer_blocks_7_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_2_1_transformer_blocks_7_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_ff_net_2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_7_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_ff_net_2_weight.bin", 6553600)
    output_blocks_2_1_transformer_blocks_7_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_norm1_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_7_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_norm1_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_7_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_norm2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_7_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_norm2_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_7_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_norm3_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_7_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_7_norm3_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_8_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_attn1_to_k_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_8_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_attn1_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_8_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_8_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_attn1_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_8_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_attn1_to_v_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_8_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_attn2_to_k_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_8_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_attn2_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_8_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_8_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_attn2_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_8_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_attn2_to_v_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_8_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_ff_net_0_proj_bias.bin", 10240)
    output_blocks_2_1_transformer_blocks_8_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_2_1_transformer_blocks_8_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_ff_net_2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_8_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_ff_net_2_weight.bin", 6553600)
    output_blocks_2_1_transformer_blocks_8_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_norm1_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_8_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_norm1_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_8_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_norm2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_8_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_norm2_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_8_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_norm3_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_8_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_8_norm3_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_9_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_attn1_to_k_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_9_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_attn1_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_9_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_attn1_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_9_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_attn1_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_9_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_attn1_to_v_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_9_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_attn2_to_k_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_9_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_attn2_to_out_0_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_9_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_attn2_to_out_0_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_9_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_attn2_to_q_weight.bin", 1638400)
    output_blocks_2_1_transformer_blocks_9_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_attn2_to_v_weight.bin", 2621440)
    output_blocks_2_1_transformer_blocks_9_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_ff_net_0_proj_bias.bin", 10240)
    output_blocks_2_1_transformer_blocks_9_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_ff_net_0_proj_weight.bin", 13107200)
    output_blocks_2_1_transformer_blocks_9_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_ff_net_2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_9_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_ff_net_2_weight.bin", 6553600)
    output_blocks_2_1_transformer_blocks_9_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_norm1_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_9_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_norm1_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_9_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_norm2_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_9_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_norm2_weight.bin", 1280)
    output_blocks_2_1_transformer_blocks_9_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_norm3_bias.bin", 1280)
    output_blocks_2_1_transformer_blocks_9_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_1_transformer_blocks_9_norm3_weight.bin", 1280)
    output_blocks_2_2_conv_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_2_conv_bias.bin", 1280)
    output_blocks_2_2_conv_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_2_2_conv_weight.bin", 14745600)
    output_blocks_3_0_emb_layers_1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_0_emb_layers_1_bias.bin", 640)
    output_blocks_3_0_emb_layers_1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_0_emb_layers_1_weight.bin", 819200)
    output_blocks_3_0_in_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_0_in_layers_0_bias.bin", 1920)
    output_blocks_3_0_in_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_0_in_layers_0_weight.bin", 1920)
    output_blocks_3_0_in_layers_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_0_in_layers_2_bias.bin", 640)
    output_blocks_3_0_in_layers_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_0_in_layers_2_weight.bin", 11059200)
    output_blocks_3_0_out_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_0_out_layers_0_bias.bin", 640)
    output_blocks_3_0_out_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_0_out_layers_0_weight.bin", 640)
    output_blocks_3_0_out_layers_3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_0_out_layers_3_bias.bin", 640)
    output_blocks_3_0_out_layers_3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_0_out_layers_3_weight.bin", 3686400)
    output_blocks_3_0_skip_connection_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_0_skip_connection_bias.bin", 640)
    output_blocks_3_0_skip_connection_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_0_skip_connection_weight.bin", 1228800)
    output_blocks_3_1_norm_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_norm_bias.bin", 640)
    output_blocks_3_1_norm_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_norm_weight.bin", 640)
    output_blocks_3_1_proj_in_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_proj_in_bias.bin", 640)
    output_blocks_3_1_proj_in_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_proj_in_weight.bin", 409600)
    output_blocks_3_1_proj_out_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_proj_out_bias.bin", 640)
    output_blocks_3_1_proj_out_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_proj_out_weight.bin", 409600)
    output_blocks_3_1_transformer_blocks_0_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_attn1_to_k_weight.bin", 409600)
    output_blocks_3_1_transformer_blocks_0_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_attn1_to_out_0_bias.bin", 640)
    output_blocks_3_1_transformer_blocks_0_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_attn1_to_out_0_weight.bin", 409600)
    output_blocks_3_1_transformer_blocks_0_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_attn1_to_q_weight.bin", 409600)
    output_blocks_3_1_transformer_blocks_0_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_attn1_to_v_weight.bin", 409600)
    output_blocks_3_1_transformer_blocks_0_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_attn2_to_k_weight.bin", 1310720)
    output_blocks_3_1_transformer_blocks_0_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_attn2_to_out_0_bias.bin", 640)
    output_blocks_3_1_transformer_blocks_0_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_attn2_to_out_0_weight.bin", 409600)
    output_blocks_3_1_transformer_blocks_0_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_attn2_to_q_weight.bin", 409600)
    output_blocks_3_1_transformer_blocks_0_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_attn2_to_v_weight.bin", 1310720)
    output_blocks_3_1_transformer_blocks_0_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_ff_net_0_proj_bias.bin", 5120)
    output_blocks_3_1_transformer_blocks_0_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_ff_net_0_proj_weight.bin", 3276800)
    output_blocks_3_1_transformer_blocks_0_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_ff_net_2_bias.bin", 640)
    output_blocks_3_1_transformer_blocks_0_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_ff_net_2_weight.bin", 1638400)
    output_blocks_3_1_transformer_blocks_0_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_norm1_bias.bin", 640)
    output_blocks_3_1_transformer_blocks_0_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_norm1_weight.bin", 640)
    output_blocks_3_1_transformer_blocks_0_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_norm2_bias.bin", 640)
    output_blocks_3_1_transformer_blocks_0_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_norm2_weight.bin", 640)
    output_blocks_3_1_transformer_blocks_0_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_norm3_bias.bin", 640)
    output_blocks_3_1_transformer_blocks_0_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_0_norm3_weight.bin", 640)
    output_blocks_3_1_transformer_blocks_1_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_attn1_to_k_weight.bin", 409600)
    output_blocks_3_1_transformer_blocks_1_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_attn1_to_out_0_bias.bin", 640)
    output_blocks_3_1_transformer_blocks_1_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_attn1_to_out_0_weight.bin", 409600)
    output_blocks_3_1_transformer_blocks_1_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_attn1_to_q_weight.bin", 409600)
    output_blocks_3_1_transformer_blocks_1_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_attn1_to_v_weight.bin", 409600)
    output_blocks_3_1_transformer_blocks_1_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_attn2_to_k_weight.bin", 1310720)
    output_blocks_3_1_transformer_blocks_1_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_attn2_to_out_0_bias.bin", 640)
    output_blocks_3_1_transformer_blocks_1_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_attn2_to_out_0_weight.bin", 409600)
    output_blocks_3_1_transformer_blocks_1_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_attn2_to_q_weight.bin", 409600)
    output_blocks_3_1_transformer_blocks_1_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_attn2_to_v_weight.bin", 1310720)
    output_blocks_3_1_transformer_blocks_1_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_ff_net_0_proj_bias.bin", 5120)
    output_blocks_3_1_transformer_blocks_1_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_ff_net_0_proj_weight.bin", 3276800)
    output_blocks_3_1_transformer_blocks_1_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_ff_net_2_bias.bin", 640)
    output_blocks_3_1_transformer_blocks_1_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_ff_net_2_weight.bin", 1638400)
    output_blocks_3_1_transformer_blocks_1_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_norm1_bias.bin", 640)
    output_blocks_3_1_transformer_blocks_1_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_norm1_weight.bin", 640)
    output_blocks_3_1_transformer_blocks_1_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_norm2_bias.bin", 640)
    output_blocks_3_1_transformer_blocks_1_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_norm2_weight.bin", 640)
    output_blocks_3_1_transformer_blocks_1_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_norm3_bias.bin", 640)
    output_blocks_3_1_transformer_blocks_1_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_3_1_transformer_blocks_1_norm3_weight.bin", 640)
    output_blocks_4_0_emb_layers_1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_0_emb_layers_1_bias.bin", 640)
    output_blocks_4_0_emb_layers_1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_0_emb_layers_1_weight.bin", 819200)
    output_blocks_4_0_in_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_0_in_layers_0_bias.bin", 1280)
    output_blocks_4_0_in_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_0_in_layers_0_weight.bin", 1280)
    output_blocks_4_0_in_layers_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_0_in_layers_2_bias.bin", 640)
    output_blocks_4_0_in_layers_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_0_in_layers_2_weight.bin", 7372800)
    output_blocks_4_0_out_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_0_out_layers_0_bias.bin", 640)
    output_blocks_4_0_out_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_0_out_layers_0_weight.bin", 640)
    output_blocks_4_0_out_layers_3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_0_out_layers_3_bias.bin", 640)
    output_blocks_4_0_out_layers_3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_0_out_layers_3_weight.bin", 3686400)
    output_blocks_4_0_skip_connection_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_0_skip_connection_bias.bin", 640)
    output_blocks_4_0_skip_connection_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_0_skip_connection_weight.bin", 819200)
    output_blocks_4_1_norm_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_norm_bias.bin", 640)
    output_blocks_4_1_norm_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_norm_weight.bin", 640)
    output_blocks_4_1_proj_in_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_proj_in_bias.bin", 640)
    output_blocks_4_1_proj_in_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_proj_in_weight.bin", 409600)
    output_blocks_4_1_proj_out_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_proj_out_bias.bin", 640)
    output_blocks_4_1_proj_out_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_proj_out_weight.bin", 409600)
    output_blocks_4_1_transformer_blocks_0_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_attn1_to_k_weight.bin", 409600)
    output_blocks_4_1_transformer_blocks_0_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_attn1_to_out_0_bias.bin", 640)
    output_blocks_4_1_transformer_blocks_0_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_attn1_to_out_0_weight.bin", 409600)
    output_blocks_4_1_transformer_blocks_0_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_attn1_to_q_weight.bin", 409600)
    output_blocks_4_1_transformer_blocks_0_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_attn1_to_v_weight.bin", 409600)
    output_blocks_4_1_transformer_blocks_0_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_attn2_to_k_weight.bin", 1310720)
    output_blocks_4_1_transformer_blocks_0_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_attn2_to_out_0_bias.bin", 640)
    output_blocks_4_1_transformer_blocks_0_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_attn2_to_out_0_weight.bin", 409600)
    output_blocks_4_1_transformer_blocks_0_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_attn2_to_q_weight.bin", 409600)
    output_blocks_4_1_transformer_blocks_0_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_attn2_to_v_weight.bin", 1310720)
    output_blocks_4_1_transformer_blocks_0_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_ff_net_0_proj_bias.bin", 5120)
    output_blocks_4_1_transformer_blocks_0_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_ff_net_0_proj_weight.bin", 3276800)
    output_blocks_4_1_transformer_blocks_0_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_ff_net_2_bias.bin", 640)
    output_blocks_4_1_transformer_blocks_0_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_ff_net_2_weight.bin", 1638400)
    output_blocks_4_1_transformer_blocks_0_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_norm1_bias.bin", 640)
    output_blocks_4_1_transformer_blocks_0_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_norm1_weight.bin", 640)
    output_blocks_4_1_transformer_blocks_0_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_norm2_bias.bin", 640)
    output_blocks_4_1_transformer_blocks_0_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_norm2_weight.bin", 640)
    output_blocks_4_1_transformer_blocks_0_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_norm3_bias.bin", 640)
    output_blocks_4_1_transformer_blocks_0_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_0_norm3_weight.bin", 640)
    output_blocks_4_1_transformer_blocks_1_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_attn1_to_k_weight.bin", 409600)
    output_blocks_4_1_transformer_blocks_1_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_attn1_to_out_0_bias.bin", 640)
    output_blocks_4_1_transformer_blocks_1_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_attn1_to_out_0_weight.bin", 409600)
    output_blocks_4_1_transformer_blocks_1_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_attn1_to_q_weight.bin", 409600)
    output_blocks_4_1_transformer_blocks_1_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_attn1_to_v_weight.bin", 409600)
    output_blocks_4_1_transformer_blocks_1_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_attn2_to_k_weight.bin", 1310720)
    output_blocks_4_1_transformer_blocks_1_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_attn2_to_out_0_bias.bin", 640)
    output_blocks_4_1_transformer_blocks_1_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_attn2_to_out_0_weight.bin", 409600)
    output_blocks_4_1_transformer_blocks_1_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_attn2_to_q_weight.bin", 409600)
    output_blocks_4_1_transformer_blocks_1_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_attn2_to_v_weight.bin", 1310720)
    output_blocks_4_1_transformer_blocks_1_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_ff_net_0_proj_bias.bin", 5120)
    output_blocks_4_1_transformer_blocks_1_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_ff_net_0_proj_weight.bin", 3276800)
    output_blocks_4_1_transformer_blocks_1_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_ff_net_2_bias.bin", 640)
    output_blocks_4_1_transformer_blocks_1_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_ff_net_2_weight.bin", 1638400)
    output_blocks_4_1_transformer_blocks_1_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_norm1_bias.bin", 640)
    output_blocks_4_1_transformer_blocks_1_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_norm1_weight.bin", 640)
    output_blocks_4_1_transformer_blocks_1_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_norm2_bias.bin", 640)
    output_blocks_4_1_transformer_blocks_1_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_norm2_weight.bin", 640)
    output_blocks_4_1_transformer_blocks_1_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_norm3_bias.bin", 640)
    output_blocks_4_1_transformer_blocks_1_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_4_1_transformer_blocks_1_norm3_weight.bin", 640)
    output_blocks_5_0_emb_layers_1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_0_emb_layers_1_bias.bin", 640)
    output_blocks_5_0_emb_layers_1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_0_emb_layers_1_weight.bin", 819200)
    output_blocks_5_0_in_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_0_in_layers_0_bias.bin", 960)
    output_blocks_5_0_in_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_0_in_layers_0_weight.bin", 960)
    output_blocks_5_0_in_layers_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_0_in_layers_2_bias.bin", 640)
    output_blocks_5_0_in_layers_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_0_in_layers_2_weight.bin", 5529600)
    output_blocks_5_0_out_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_0_out_layers_0_bias.bin", 640)
    output_blocks_5_0_out_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_0_out_layers_0_weight.bin", 640)
    output_blocks_5_0_out_layers_3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_0_out_layers_3_bias.bin", 640)
    output_blocks_5_0_out_layers_3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_0_out_layers_3_weight.bin", 3686400)
    output_blocks_5_0_skip_connection_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_0_skip_connection_bias.bin", 640)
    output_blocks_5_0_skip_connection_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_0_skip_connection_weight.bin", 614400)
    output_blocks_5_1_norm_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_norm_bias.bin", 640)
    output_blocks_5_1_norm_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_norm_weight.bin", 640)
    output_blocks_5_1_proj_in_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_proj_in_bias.bin", 640)
    output_blocks_5_1_proj_in_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_proj_in_weight.bin", 409600)
    output_blocks_5_1_proj_out_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_proj_out_bias.bin", 640)
    output_blocks_5_1_proj_out_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_proj_out_weight.bin", 409600)
    output_blocks_5_1_transformer_blocks_0_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_attn1_to_k_weight.bin", 409600)
    output_blocks_5_1_transformer_blocks_0_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_attn1_to_out_0_bias.bin", 640)
    output_blocks_5_1_transformer_blocks_0_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_attn1_to_out_0_weight.bin", 409600)
    output_blocks_5_1_transformer_blocks_0_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_attn1_to_q_weight.bin", 409600)
    output_blocks_5_1_transformer_blocks_0_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_attn1_to_v_weight.bin", 409600)
    output_blocks_5_1_transformer_blocks_0_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_attn2_to_k_weight.bin", 1310720)
    output_blocks_5_1_transformer_blocks_0_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_attn2_to_out_0_bias.bin", 640)
    output_blocks_5_1_transformer_blocks_0_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_attn2_to_out_0_weight.bin", 409600)
    output_blocks_5_1_transformer_blocks_0_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_attn2_to_q_weight.bin", 409600)
    output_blocks_5_1_transformer_blocks_0_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_attn2_to_v_weight.bin", 1310720)
    output_blocks_5_1_transformer_blocks_0_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_ff_net_0_proj_bias.bin", 5120)
    output_blocks_5_1_transformer_blocks_0_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_ff_net_0_proj_weight.bin", 3276800)
    output_blocks_5_1_transformer_blocks_0_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_ff_net_2_bias.bin", 640)
    output_blocks_5_1_transformer_blocks_0_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_ff_net_2_weight.bin", 1638400)
    output_blocks_5_1_transformer_blocks_0_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_norm1_bias.bin", 640)
    output_blocks_5_1_transformer_blocks_0_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_norm1_weight.bin", 640)
    output_blocks_5_1_transformer_blocks_0_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_norm2_bias.bin", 640)
    output_blocks_5_1_transformer_blocks_0_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_norm2_weight.bin", 640)
    output_blocks_5_1_transformer_blocks_0_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_norm3_bias.bin", 640)
    output_blocks_5_1_transformer_blocks_0_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_0_norm3_weight.bin", 640)
    output_blocks_5_1_transformer_blocks_1_attn1_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_attn1_to_k_weight.bin", 409600)
    output_blocks_5_1_transformer_blocks_1_attn1_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_attn1_to_out_0_bias.bin", 640)
    output_blocks_5_1_transformer_blocks_1_attn1_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_attn1_to_out_0_weight.bin", 409600)
    output_blocks_5_1_transformer_blocks_1_attn1_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_attn1_to_q_weight.bin", 409600)
    output_blocks_5_1_transformer_blocks_1_attn1_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_attn1_to_v_weight.bin", 409600)
    output_blocks_5_1_transformer_blocks_1_attn2_to_k_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_attn2_to_k_weight.bin", 1310720)
    output_blocks_5_1_transformer_blocks_1_attn2_to_out_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_attn2_to_out_0_bias.bin", 640)
    output_blocks_5_1_transformer_blocks_1_attn2_to_out_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_attn2_to_out_0_weight.bin", 409600)
    output_blocks_5_1_transformer_blocks_1_attn2_to_q_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_attn2_to_q_weight.bin", 409600)
    output_blocks_5_1_transformer_blocks_1_attn2_to_v_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_attn2_to_v_weight.bin", 1310720)
    output_blocks_5_1_transformer_blocks_1_ff_net_0_proj_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_ff_net_0_proj_bias.bin", 5120)
    output_blocks_5_1_transformer_blocks_1_ff_net_0_proj_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_ff_net_0_proj_weight.bin", 3276800)
    output_blocks_5_1_transformer_blocks_1_ff_net_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_ff_net_2_bias.bin", 640)
    output_blocks_5_1_transformer_blocks_1_ff_net_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_ff_net_2_weight.bin", 1638400)
    output_blocks_5_1_transformer_blocks_1_norm1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_norm1_bias.bin", 640)
    output_blocks_5_1_transformer_blocks_1_norm1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_norm1_weight.bin", 640)
    output_blocks_5_1_transformer_blocks_1_norm2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_norm2_bias.bin", 640)
    output_blocks_5_1_transformer_blocks_1_norm2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_norm2_weight.bin", 640)
    output_blocks_5_1_transformer_blocks_1_norm3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_norm3_bias.bin", 640)
    output_blocks_5_1_transformer_blocks_1_norm3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_1_transformer_blocks_1_norm3_weight.bin", 640)
    output_blocks_5_2_conv_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_2_conv_bias.bin", 640)
    output_blocks_5_2_conv_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_5_2_conv_weight.bin", 3686400)
    output_blocks_6_0_emb_layers_1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_6_0_emb_layers_1_bias.bin", 320)
    output_blocks_6_0_emb_layers_1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_6_0_emb_layers_1_weight.bin", 409600)
    output_blocks_6_0_in_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_6_0_in_layers_0_bias.bin", 960)
    output_blocks_6_0_in_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_6_0_in_layers_0_weight.bin", 960)
    output_blocks_6_0_in_layers_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_6_0_in_layers_2_bias.bin", 320)
    output_blocks_6_0_in_layers_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_6_0_in_layers_2_weight.bin", 2764800)
    output_blocks_6_0_out_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_6_0_out_layers_0_bias.bin", 320)
    output_blocks_6_0_out_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_6_0_out_layers_0_weight.bin", 320)
    output_blocks_6_0_out_layers_3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_6_0_out_layers_3_bias.bin", 320)
    output_blocks_6_0_out_layers_3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_6_0_out_layers_3_weight.bin", 921600)
    output_blocks_6_0_skip_connection_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_6_0_skip_connection_bias.bin", 320)
    output_blocks_6_0_skip_connection_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_6_0_skip_connection_weight.bin", 307200)
    output_blocks_7_0_emb_layers_1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_7_0_emb_layers_1_bias.bin", 320)
    output_blocks_7_0_emb_layers_1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_7_0_emb_layers_1_weight.bin", 409600)
    output_blocks_7_0_in_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_7_0_in_layers_0_bias.bin", 640)
    output_blocks_7_0_in_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_7_0_in_layers_0_weight.bin", 640)
    output_blocks_7_0_in_layers_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_7_0_in_layers_2_bias.bin", 320)
    output_blocks_7_0_in_layers_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_7_0_in_layers_2_weight.bin", 1843200)
    output_blocks_7_0_out_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_7_0_out_layers_0_bias.bin", 320)
    output_blocks_7_0_out_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_7_0_out_layers_0_weight.bin", 320)
    output_blocks_7_0_out_layers_3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_7_0_out_layers_3_bias.bin", 320)
    output_blocks_7_0_out_layers_3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_7_0_out_layers_3_weight.bin", 921600)
    output_blocks_7_0_skip_connection_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_7_0_skip_connection_bias.bin", 320)
    output_blocks_7_0_skip_connection_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_7_0_skip_connection_weight.bin", 204800)
    output_blocks_8_0_emb_layers_1_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_8_0_emb_layers_1_bias.bin", 320)
    output_blocks_8_0_emb_layers_1_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_8_0_emb_layers_1_weight.bin", 409600)
    output_blocks_8_0_in_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_8_0_in_layers_0_bias.bin", 640)
    output_blocks_8_0_in_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_8_0_in_layers_0_weight.bin", 640)
    output_blocks_8_0_in_layers_2_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_8_0_in_layers_2_bias.bin", 320)
    output_blocks_8_0_in_layers_2_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_8_0_in_layers_2_weight.bin", 1843200)
    output_blocks_8_0_out_layers_0_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_8_0_out_layers_0_bias.bin", 320)
    output_blocks_8_0_out_layers_0_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_8_0_out_layers_0_weight.bin", 320)
    output_blocks_8_0_out_layers_3_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_8_0_out_layers_3_bias.bin", 320)
    output_blocks_8_0_out_layers_3_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_8_0_out_layers_3_weight.bin", 921600)
    output_blocks_8_0_skip_connection_bias = load_bin(weights_dir + "/model_diffusion_model_output_blocks_8_0_skip_connection_bias.bin", 320)
    output_blocks_8_0_skip_connection_weight = load_bin(weights_dir + "/model_diffusion_model_output_blocks_8_0_skip_connection_weight.bin", 204800)
    time_embed_0_bias = load_bin(weights_dir + "/model_diffusion_model_time_embed_0_bias.bin", 1280)
    time_embed_0_weight = load_bin(weights_dir + "/model_diffusion_model_time_embed_0_weight.bin", 409600)
    time_embed_2_bias = load_bin(weights_dir + "/model_diffusion_model_time_embed_2_bias.bin", 1280)
    time_embed_2_weight = load_bin(weights_dir + "/model_diffusion_model_time_embed_2_weight.bin", 1638400)

    # Input Conv: 4→320
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 4 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(latent, n, 4, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_0_0_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_0_0_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y

    # input_blocks.1
    #   0 emb_layers.1.bias [320]
    #   0 emb_layers.1.weight [320, 1280]
    #   0 in_layers.0.bias [320]
    #   0 in_layers.0.weight [320]
    #   0 in_layers.2.bias [320]
    #   0 in_layers.2.weight [320, 320, 3, 3]
    #   0 out_layers.0.bias [320]
    #   0 out_layers.0.weight [320]
    #   0 out_layers.3.bias [320]
    #   0 out_layers.3.weight [320, 320, 3, 3]
    group_norm(h_cur, input_blocks_1_0_in_layers_0_weight, input_blocks_1_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_1_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_1_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # input_blocks.2
    #   0 emb_layers.1.bias [320]
    #   0 emb_layers.1.weight [320, 1280]
    #   0 in_layers.0.bias [320]
    #   0 in_layers.0.weight [320]
    #   0 in_layers.2.bias [320]
    #   0 in_layers.2.weight [320, 320, 3, 3]
    #   0 out_layers.0.bias [320]
    #   0 out_layers.0.weight [320]
    #   0 out_layers.3.bias [320]
    #   0 out_layers.3.weight [320, 320, 3, 3]
    group_norm(h_cur, input_blocks_2_0_in_layers_0_weight, input_blocks_2_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_2_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_2_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # input_blocks.3
    #   0 op.bias [320]
    #   0 op.weight [320, 320, 3, 3]
    group_norm(h_cur, input_blocks_3_0_in_layers_0_weight, input_blocks_3_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_3_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_3_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # input_blocks.4
    #   0 emb_layers.1.bias [640]
    #   0 emb_layers.1.weight [640, 1280]
    #   0 in_layers.0.bias [320]
    #   0 in_layers.0.weight [320]
    #   0 in_layers.2.bias [640]
    #   0 in_layers.2.weight [640, 320, 3, 3]
    #   0 out_layers.0.bias [640]
    #   0 out_layers.0.weight [640]
    #   0 out_layers.3.bias [640]
    #   0 out_layers.3.weight [640, 640, 3, 3]
    #   0 skip_connection.bias [640]
    #   0 skip_connection.weight [640, 320, 1, 1]
    #   1 norm.bias [640]
    #   1 norm.weight [640]
    #   1 proj_in.bias [640]
    #   1 proj_in.weight [640, 640]
    #   1 proj_out.bias [640]
    #   1 proj_out.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_k.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_out.0.bias [640]
    #   1 transformer_blocks.0.attn1.to_out.0.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_q.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_v.weight [640, 640]
    #   1 transformer_blocks.0.attn2.to_k.weight [640, 2048]
    #   1 transformer_blocks.0.attn2.to_out.0.bias [640]
    #   1 transformer_blocks.0.attn2.to_out.0.weight [640, 640]
    #   1 transformer_blocks.0.attn2.to_q.weight [640, 640]
    #   1 transformer_blocks.0.attn2.to_v.weight [640, 2048]
    #   1 transformer_blocks.0.ff.net.0.proj.bias [5120]
    #   1 transformer_blocks.0.ff.net.0.proj.weight [5120, 640]
    #   1 transformer_blocks.0.ff.net.2.bias [640]
    #   1 transformer_blocks.0.ff.net.2.weight [640, 2560]
    #   1 transformer_blocks.0.norm1.bias [640]
    #   1 transformer_blocks.0.norm1.weight [640]
    #   1 transformer_blocks.0.norm2.bias [640]
    #   1 transformer_blocks.0.norm2.weight [640]
    #   1 transformer_blocks.0.norm3.bias [640]
    #   1 transformer_blocks.0.norm3.weight [640]
    #   1 transformer_blocks.1.attn1.to_k.weight [640, 640]
    #   1 transformer_blocks.1.attn1.to_out.0.bias [640]
    #   1 transformer_blocks.1.attn1.to_out.0.weight [640, 640]
    #   1 transformer_blocks.1.attn1.to_q.weight [640, 640]
    #   1 transformer_blocks.1.attn1.to_v.weight [640, 640]
    #   1 transformer_blocks.1.attn2.to_k.weight [640, 2048]
    #   1 transformer_blocks.1.attn2.to_out.0.bias [640]
    #   1 transformer_blocks.1.attn2.to_out.0.weight [640, 640]
    #   1 transformer_blocks.1.attn2.to_q.weight [640, 640]
    #   1 transformer_blocks.1.attn2.to_v.weight [640, 2048]
    #   1 transformer_blocks.1.ff.net.0.proj.bias [5120]
    #   1 transformer_blocks.1.ff.net.0.proj.weight [5120, 640]
    #   1 transformer_blocks.1.ff.net.2.bias [640]
    #   1 transformer_blocks.1.ff.net.2.weight [640, 2560]
    #   1 transformer_blocks.1.norm1.bias [640]
    #   1 transformer_blocks.1.norm1.weight [640]
    #   1 transformer_blocks.1.norm2.bias [640]
    #   1 transformer_blocks.1.norm2.weight [640]
    #   1 transformer_blocks.1.norm3.bias [640]
    #   1 transformer_blocks.1.norm3.weight [640]
    group_norm(h_cur, input_blocks_4_0_in_layers_0_weight, input_blocks_4_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_4_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_4_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # input_blocks.5
    #   0 emb_layers.1.bias [640]
    #   0 emb_layers.1.weight [640, 1280]
    #   0 in_layers.0.bias [640]
    #   0 in_layers.0.weight [640]
    #   0 in_layers.2.bias [640]
    #   0 in_layers.2.weight [640, 640, 3, 3]
    #   0 out_layers.0.bias [640]
    #   0 out_layers.0.weight [640]
    #   0 out_layers.3.bias [640]
    #   0 out_layers.3.weight [640, 640, 3, 3]
    #   1 norm.bias [640]
    #   1 norm.weight [640]
    #   1 proj_in.bias [640]
    #   1 proj_in.weight [640, 640]
    #   1 proj_out.bias [640]
    #   1 proj_out.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_k.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_out.0.bias [640]
    #   1 transformer_blocks.0.attn1.to_out.0.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_q.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_v.weight [640, 640]
    #   1 transformer_blocks.0.attn2.to_k.weight [640, 2048]
    #   1 transformer_blocks.0.attn2.to_out.0.bias [640]
    #   1 transformer_blocks.0.attn2.to_out.0.weight [640, 640]
    #   1 transformer_blocks.0.attn2.to_q.weight [640, 640]
    #   1 transformer_blocks.0.attn2.to_v.weight [640, 2048]
    #   1 transformer_blocks.0.ff.net.0.proj.bias [5120]
    #   1 transformer_blocks.0.ff.net.0.proj.weight [5120, 640]
    #   1 transformer_blocks.0.ff.net.2.bias [640]
    #   1 transformer_blocks.0.ff.net.2.weight [640, 2560]
    #   1 transformer_blocks.0.norm1.bias [640]
    #   1 transformer_blocks.0.norm1.weight [640]
    #   1 transformer_blocks.0.norm2.bias [640]
    #   1 transformer_blocks.0.norm2.weight [640]
    #   1 transformer_blocks.0.norm3.bias [640]
    #   1 transformer_blocks.0.norm3.weight [640]
    #   1 transformer_blocks.1.attn1.to_k.weight [640, 640]
    #   1 transformer_blocks.1.attn1.to_out.0.bias [640]
    #   1 transformer_blocks.1.attn1.to_out.0.weight [640, 640]
    #   1 transformer_blocks.1.attn1.to_q.weight [640, 640]
    #   1 transformer_blocks.1.attn1.to_v.weight [640, 640]
    #   1 transformer_blocks.1.attn2.to_k.weight [640, 2048]
    #   1 transformer_blocks.1.attn2.to_out.0.bias [640]
    #   1 transformer_blocks.1.attn2.to_out.0.weight [640, 640]
    #   1 transformer_blocks.1.attn2.to_q.weight [640, 640]
    #   1 transformer_blocks.1.attn2.to_v.weight [640, 2048]
    #   1 transformer_blocks.1.ff.net.0.proj.bias [5120]
    #   1 transformer_blocks.1.ff.net.0.proj.weight [5120, 640]
    #   1 transformer_blocks.1.ff.net.2.bias [640]
    #   1 transformer_blocks.1.ff.net.2.weight [640, 2560]
    #   1 transformer_blocks.1.norm1.bias [640]
    #   1 transformer_blocks.1.norm1.weight [640]
    #   1 transformer_blocks.1.norm2.bias [640]
    #   1 transformer_blocks.1.norm2.weight [640]
    #   1 transformer_blocks.1.norm3.bias [640]
    #   1 transformer_blocks.1.norm3.weight [640]
    group_norm(h_cur, input_blocks_5_0_in_layers_0_weight, input_blocks_5_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_5_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_5_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # input_blocks.6
    #   0 op.bias [640]
    #   0 op.weight [640, 640, 3, 3]
    group_norm(h_cur, input_blocks_6_0_in_layers_0_weight, input_blocks_6_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_6_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_6_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # input_blocks.7
    #   0 emb_layers.1.bias [1280]
    #   0 emb_layers.1.weight [1280, 1280]
    #   0 in_layers.0.bias [640]
    #   0 in_layers.0.weight [640]
    #   0 in_layers.2.bias [1280]
    #   0 in_layers.2.weight [1280, 640, 3, 3]
    #   0 out_layers.0.bias [1280]
    #   0 out_layers.0.weight [1280]
    #   0 out_layers.3.bias [1280]
    #   0 out_layers.3.weight [1280, 1280, 3, 3]
    #   0 skip_connection.bias [1280]
    #   0 skip_connection.weight [1280, 640, 1, 1]
    #   1 norm.bias [1280]
    #   1 norm.weight [1280]
    #   1 proj_in.bias [1280]
    #   1 proj_in.weight [1280, 1280]
    #   1 proj_out.bias [1280]
    #   1 proj_out.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.0.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.0.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.0.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.0.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.0.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.0.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.0.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.0.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.0.ff.net.2.bias [1280]
    #   1 transformer_blocks.0.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.0.norm1.bias [1280]
    #   1 transformer_blocks.0.norm1.weight [1280]
    #   1 transformer_blocks.0.norm2.bias [1280]
    #   1 transformer_blocks.0.norm2.weight [1280]
    #   1 transformer_blocks.0.norm3.bias [1280]
    #   1 transformer_blocks.0.norm3.weight [1280]
    #   1 transformer_blocks.1.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.1.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.1.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.1.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.1.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.1.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.1.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.1.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.1.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.1.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.1.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.1.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.1.ff.net.2.bias [1280]
    #   1 transformer_blocks.1.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.1.norm1.bias [1280]
    #   1 transformer_blocks.1.norm1.weight [1280]
    #   1 transformer_blocks.1.norm2.bias [1280]
    #   1 transformer_blocks.1.norm2.weight [1280]
    #   1 transformer_blocks.1.norm3.bias [1280]
    #   1 transformer_blocks.1.norm3.weight [1280]
    #   1 transformer_blocks.2.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.2.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.2.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.2.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.2.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.2.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.2.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.2.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.2.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.2.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.2.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.2.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.2.ff.net.2.bias [1280]
    #   1 transformer_blocks.2.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.2.norm1.bias [1280]
    #   1 transformer_blocks.2.norm1.weight [1280]
    #   1 transformer_blocks.2.norm2.bias [1280]
    #   1 transformer_blocks.2.norm2.weight [1280]
    #   1 transformer_blocks.2.norm3.bias [1280]
    #   1 transformer_blocks.2.norm3.weight [1280]
    #   1 transformer_blocks.3.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.3.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.3.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.3.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.3.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.3.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.3.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.3.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.3.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.3.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.3.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.3.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.3.ff.net.2.bias [1280]
    #   1 transformer_blocks.3.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.3.norm1.bias [1280]
    #   1 transformer_blocks.3.norm1.weight [1280]
    #   1 transformer_blocks.3.norm2.bias [1280]
    #   1 transformer_blocks.3.norm2.weight [1280]
    #   1 transformer_blocks.3.norm3.bias [1280]
    #   1 transformer_blocks.3.norm3.weight [1280]
    #   1 transformer_blocks.4.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.4.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.4.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.4.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.4.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.4.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.4.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.4.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.4.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.4.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.4.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.4.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.4.ff.net.2.bias [1280]
    #   1 transformer_blocks.4.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.4.norm1.bias [1280]
    #   1 transformer_blocks.4.norm1.weight [1280]
    #   1 transformer_blocks.4.norm2.bias [1280]
    #   1 transformer_blocks.4.norm2.weight [1280]
    #   1 transformer_blocks.4.norm3.bias [1280]
    #   1 transformer_blocks.4.norm3.weight [1280]
    #   1 transformer_blocks.5.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.5.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.5.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.5.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.5.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.5.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.5.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.5.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.5.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.5.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.5.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.5.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.5.ff.net.2.bias [1280]
    #   1 transformer_blocks.5.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.5.norm1.bias [1280]
    #   1 transformer_blocks.5.norm1.weight [1280]
    #   1 transformer_blocks.5.norm2.bias [1280]
    #   1 transformer_blocks.5.norm2.weight [1280]
    #   1 transformer_blocks.5.norm3.bias [1280]
    #   1 transformer_blocks.5.norm3.weight [1280]
    #   1 transformer_blocks.6.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.6.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.6.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.6.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.6.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.6.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.6.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.6.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.6.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.6.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.6.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.6.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.6.ff.net.2.bias [1280]
    #   1 transformer_blocks.6.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.6.norm1.bias [1280]
    #   1 transformer_blocks.6.norm1.weight [1280]
    #   1 transformer_blocks.6.norm2.bias [1280]
    #   1 transformer_blocks.6.norm2.weight [1280]
    #   1 transformer_blocks.6.norm3.bias [1280]
    #   1 transformer_blocks.6.norm3.weight [1280]
    #   1 transformer_blocks.7.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.7.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.7.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.7.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.7.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.7.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.7.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.7.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.7.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.7.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.7.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.7.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.7.ff.net.2.bias [1280]
    #   1 transformer_blocks.7.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.7.norm1.bias [1280]
    #   1 transformer_blocks.7.norm1.weight [1280]
    #   1 transformer_blocks.7.norm2.bias [1280]
    #   1 transformer_blocks.7.norm2.weight [1280]
    #   1 transformer_blocks.7.norm3.bias [1280]
    #   1 transformer_blocks.7.norm3.weight [1280]
    #   1 transformer_blocks.8.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.8.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.8.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.8.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.8.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.8.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.8.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.8.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.8.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.8.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.8.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.8.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.8.ff.net.2.bias [1280]
    #   1 transformer_blocks.8.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.8.norm1.bias [1280]
    #   1 transformer_blocks.8.norm1.weight [1280]
    #   1 transformer_blocks.8.norm2.bias [1280]
    #   1 transformer_blocks.8.norm2.weight [1280]
    #   1 transformer_blocks.8.norm3.bias [1280]
    #   1 transformer_blocks.8.norm3.weight [1280]
    #   1 transformer_blocks.9.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.9.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.9.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.9.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.9.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.9.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.9.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.9.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.9.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.9.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.9.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.9.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.9.ff.net.2.bias [1280]
    #   1 transformer_blocks.9.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.9.norm1.bias [1280]
    #   1 transformer_blocks.9.norm1.weight [1280]
    #   1 transformer_blocks.9.norm2.bias [1280]
    #   1 transformer_blocks.9.norm2.weight [1280]
    #   1 transformer_blocks.9.norm3.bias [1280]
    #   1 transformer_blocks.9.norm3.weight [1280]
    group_norm(h_cur, input_blocks_7_0_in_layers_0_weight, input_blocks_7_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_7_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_7_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # input_blocks.8
    #   0 emb_layers.1.bias [1280]
    #   0 emb_layers.1.weight [1280, 1280]
    #   0 in_layers.0.bias [1280]
    #   0 in_layers.0.weight [1280]
    #   0 in_layers.2.bias [1280]
    #   0 in_layers.2.weight [1280, 1280, 3, 3]
    #   0 out_layers.0.bias [1280]
    #   0 out_layers.0.weight [1280]
    #   0 out_layers.3.bias [1280]
    #   0 out_layers.3.weight [1280, 1280, 3, 3]
    #   1 norm.bias [1280]
    #   1 norm.weight [1280]
    #   1 proj_in.bias [1280]
    #   1 proj_in.weight [1280, 1280]
    #   1 proj_out.bias [1280]
    #   1 proj_out.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.0.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.0.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.0.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.0.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.0.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.0.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.0.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.0.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.0.ff.net.2.bias [1280]
    #   1 transformer_blocks.0.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.0.norm1.bias [1280]
    #   1 transformer_blocks.0.norm1.weight [1280]
    #   1 transformer_blocks.0.norm2.bias [1280]
    #   1 transformer_blocks.0.norm2.weight [1280]
    #   1 transformer_blocks.0.norm3.bias [1280]
    #   1 transformer_blocks.0.norm3.weight [1280]
    #   1 transformer_blocks.1.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.1.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.1.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.1.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.1.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.1.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.1.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.1.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.1.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.1.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.1.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.1.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.1.ff.net.2.bias [1280]
    #   1 transformer_blocks.1.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.1.norm1.bias [1280]
    #   1 transformer_blocks.1.norm1.weight [1280]
    #   1 transformer_blocks.1.norm2.bias [1280]
    #   1 transformer_blocks.1.norm2.weight [1280]
    #   1 transformer_blocks.1.norm3.bias [1280]
    #   1 transformer_blocks.1.norm3.weight [1280]
    #   1 transformer_blocks.2.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.2.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.2.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.2.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.2.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.2.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.2.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.2.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.2.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.2.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.2.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.2.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.2.ff.net.2.bias [1280]
    #   1 transformer_blocks.2.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.2.norm1.bias [1280]
    #   1 transformer_blocks.2.norm1.weight [1280]
    #   1 transformer_blocks.2.norm2.bias [1280]
    #   1 transformer_blocks.2.norm2.weight [1280]
    #   1 transformer_blocks.2.norm3.bias [1280]
    #   1 transformer_blocks.2.norm3.weight [1280]
    #   1 transformer_blocks.3.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.3.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.3.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.3.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.3.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.3.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.3.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.3.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.3.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.3.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.3.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.3.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.3.ff.net.2.bias [1280]
    #   1 transformer_blocks.3.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.3.norm1.bias [1280]
    #   1 transformer_blocks.3.norm1.weight [1280]
    #   1 transformer_blocks.3.norm2.bias [1280]
    #   1 transformer_blocks.3.norm2.weight [1280]
    #   1 transformer_blocks.3.norm3.bias [1280]
    #   1 transformer_blocks.3.norm3.weight [1280]
    #   1 transformer_blocks.4.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.4.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.4.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.4.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.4.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.4.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.4.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.4.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.4.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.4.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.4.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.4.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.4.ff.net.2.bias [1280]
    #   1 transformer_blocks.4.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.4.norm1.bias [1280]
    #   1 transformer_blocks.4.norm1.weight [1280]
    #   1 transformer_blocks.4.norm2.bias [1280]
    #   1 transformer_blocks.4.norm2.weight [1280]
    #   1 transformer_blocks.4.norm3.bias [1280]
    #   1 transformer_blocks.4.norm3.weight [1280]
    #   1 transformer_blocks.5.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.5.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.5.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.5.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.5.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.5.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.5.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.5.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.5.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.5.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.5.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.5.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.5.ff.net.2.bias [1280]
    #   1 transformer_blocks.5.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.5.norm1.bias [1280]
    #   1 transformer_blocks.5.norm1.weight [1280]
    #   1 transformer_blocks.5.norm2.bias [1280]
    #   1 transformer_blocks.5.norm2.weight [1280]
    #   1 transformer_blocks.5.norm3.bias [1280]
    #   1 transformer_blocks.5.norm3.weight [1280]
    #   1 transformer_blocks.6.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.6.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.6.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.6.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.6.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.6.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.6.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.6.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.6.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.6.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.6.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.6.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.6.ff.net.2.bias [1280]
    #   1 transformer_blocks.6.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.6.norm1.bias [1280]
    #   1 transformer_blocks.6.norm1.weight [1280]
    #   1 transformer_blocks.6.norm2.bias [1280]
    #   1 transformer_blocks.6.norm2.weight [1280]
    #   1 transformer_blocks.6.norm3.bias [1280]
    #   1 transformer_blocks.6.norm3.weight [1280]
    #   1 transformer_blocks.7.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.7.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.7.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.7.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.7.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.7.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.7.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.7.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.7.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.7.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.7.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.7.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.7.ff.net.2.bias [1280]
    #   1 transformer_blocks.7.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.7.norm1.bias [1280]
    #   1 transformer_blocks.7.norm1.weight [1280]
    #   1 transformer_blocks.7.norm2.bias [1280]
    #   1 transformer_blocks.7.norm2.weight [1280]
    #   1 transformer_blocks.7.norm3.bias [1280]
    #   1 transformer_blocks.7.norm3.weight [1280]
    #   1 transformer_blocks.8.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.8.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.8.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.8.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.8.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.8.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.8.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.8.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.8.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.8.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.8.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.8.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.8.ff.net.2.bias [1280]
    #   1 transformer_blocks.8.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.8.norm1.bias [1280]
    #   1 transformer_blocks.8.norm1.weight [1280]
    #   1 transformer_blocks.8.norm2.bias [1280]
    #   1 transformer_blocks.8.norm2.weight [1280]
    #   1 transformer_blocks.8.norm3.bias [1280]
    #   1 transformer_blocks.8.norm3.weight [1280]
    #   1 transformer_blocks.9.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.9.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.9.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.9.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.9.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.9.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.9.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.9.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.9.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.9.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.9.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.9.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.9.ff.net.2.bias [1280]
    #   1 transformer_blocks.9.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.9.norm1.bias [1280]
    #   1 transformer_blocks.9.norm1.weight [1280]
    #   1 transformer_blocks.9.norm2.bias [1280]
    #   1 transformer_blocks.9.norm2.weight [1280]
    #   1 transformer_blocks.9.norm3.bias [1280]
    #   1 transformer_blocks.9.norm3.weight [1280]
    group_norm(h_cur, input_blocks_8_0_in_layers_0_weight, input_blocks_8_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_8_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_8_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # label_emb.0
    #   0 bias [1280]
    #   0 weight [1280, 2816]
    #   2 bias [1280]
    #   2 weight [1280, 1280]
    group_norm(h_cur, input_blocks_0_0_in_layers_0_weight, input_blocks_0_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_0_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_0_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # middle_block.0
    #   emb_layers 1.bias [1280]
    #   emb_layers 1.weight [1280, 1280]
    #   in_layers 0.bias [1280]
    #   in_layers 0.weight [1280]
    #   in_layers 2.bias [1280]
    #   in_layers 2.weight [1280, 1280, 3, 3]
    #   out_layers 0.bias [1280]
    #   out_layers 0.weight [1280]
    #   out_layers 3.bias [1280]
    #   out_layers 3.weight [1280, 1280, 3, 3]
    group_norm(h_cur, input_blocks_0_0_in_layers_0_weight, input_blocks_0_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_0_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_0_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # middle_block.1
    #   norm bias [1280]
    #   norm weight [1280]
    #   proj_in bias [1280]
    #   proj_in weight [1280, 1280]
    #   proj_out bias [1280]
    #   proj_out weight [1280, 1280]
    #   transformer_blocks 0.attn1.to_k.weight [1280, 1280]
    #   transformer_blocks 0.attn1.to_out.0.bias [1280]
    #   transformer_blocks 0.attn1.to_out.0.weight [1280, 1280]
    #   transformer_blocks 0.attn1.to_q.weight [1280, 1280]
    #   transformer_blocks 0.attn1.to_v.weight [1280, 1280]
    #   transformer_blocks 0.attn2.to_k.weight [1280, 2048]
    #   transformer_blocks 0.attn2.to_out.0.bias [1280]
    #   transformer_blocks 0.attn2.to_out.0.weight [1280, 1280]
    #   transformer_blocks 0.attn2.to_q.weight [1280, 1280]
    #   transformer_blocks 0.attn2.to_v.weight [1280, 2048]
    #   transformer_blocks 0.ff.net.0.proj.bias [10240]
    #   transformer_blocks 0.ff.net.0.proj.weight [10240, 1280]
    #   transformer_blocks 0.ff.net.2.bias [1280]
    #   transformer_blocks 0.ff.net.2.weight [1280, 5120]
    #   transformer_blocks 0.norm1.bias [1280]
    #   transformer_blocks 0.norm1.weight [1280]
    #   transformer_blocks 0.norm2.bias [1280]
    #   transformer_blocks 0.norm2.weight [1280]
    #   transformer_blocks 0.norm3.bias [1280]
    #   transformer_blocks 0.norm3.weight [1280]
    #   transformer_blocks 1.attn1.to_k.weight [1280, 1280]
    #   transformer_blocks 1.attn1.to_out.0.bias [1280]
    #   transformer_blocks 1.attn1.to_out.0.weight [1280, 1280]
    #   transformer_blocks 1.attn1.to_q.weight [1280, 1280]
    #   transformer_blocks 1.attn1.to_v.weight [1280, 1280]
    #   transformer_blocks 1.attn2.to_k.weight [1280, 2048]
    #   transformer_blocks 1.attn2.to_out.0.bias [1280]
    #   transformer_blocks 1.attn2.to_out.0.weight [1280, 1280]
    #   transformer_blocks 1.attn2.to_q.weight [1280, 1280]
    #   transformer_blocks 1.attn2.to_v.weight [1280, 2048]
    #   transformer_blocks 1.ff.net.0.proj.bias [10240]
    #   transformer_blocks 1.ff.net.0.proj.weight [10240, 1280]
    #   transformer_blocks 1.ff.net.2.bias [1280]
    #   transformer_blocks 1.ff.net.2.weight [1280, 5120]
    #   transformer_blocks 1.norm1.bias [1280]
    #   transformer_blocks 1.norm1.weight [1280]
    #   transformer_blocks 1.norm2.bias [1280]
    #   transformer_blocks 1.norm2.weight [1280]
    #   transformer_blocks 1.norm3.bias [1280]
    #   transformer_blocks 1.norm3.weight [1280]
    #   transformer_blocks 2.attn1.to_k.weight [1280, 1280]
    #   transformer_blocks 2.attn1.to_out.0.bias [1280]
    #   transformer_blocks 2.attn1.to_out.0.weight [1280, 1280]
    #   transformer_blocks 2.attn1.to_q.weight [1280, 1280]
    #   transformer_blocks 2.attn1.to_v.weight [1280, 1280]
    #   transformer_blocks 2.attn2.to_k.weight [1280, 2048]
    #   transformer_blocks 2.attn2.to_out.0.bias [1280]
    #   transformer_blocks 2.attn2.to_out.0.weight [1280, 1280]
    #   transformer_blocks 2.attn2.to_q.weight [1280, 1280]
    #   transformer_blocks 2.attn2.to_v.weight [1280, 2048]
    #   transformer_blocks 2.ff.net.0.proj.bias [10240]
    #   transformer_blocks 2.ff.net.0.proj.weight [10240, 1280]
    #   transformer_blocks 2.ff.net.2.bias [1280]
    #   transformer_blocks 2.ff.net.2.weight [1280, 5120]
    #   transformer_blocks 2.norm1.bias [1280]
    #   transformer_blocks 2.norm1.weight [1280]
    #   transformer_blocks 2.norm2.bias [1280]
    #   transformer_blocks 2.norm2.weight [1280]
    #   transformer_blocks 2.norm3.bias [1280]
    #   transformer_blocks 2.norm3.weight [1280]
    #   transformer_blocks 3.attn1.to_k.weight [1280, 1280]
    #   transformer_blocks 3.attn1.to_out.0.bias [1280]
    #   transformer_blocks 3.attn1.to_out.0.weight [1280, 1280]
    #   transformer_blocks 3.attn1.to_q.weight [1280, 1280]
    #   transformer_blocks 3.attn1.to_v.weight [1280, 1280]
    #   transformer_blocks 3.attn2.to_k.weight [1280, 2048]
    #   transformer_blocks 3.attn2.to_out.0.bias [1280]
    #   transformer_blocks 3.attn2.to_out.0.weight [1280, 1280]
    #   transformer_blocks 3.attn2.to_q.weight [1280, 1280]
    #   transformer_blocks 3.attn2.to_v.weight [1280, 2048]
    #   transformer_blocks 3.ff.net.0.proj.bias [10240]
    #   transformer_blocks 3.ff.net.0.proj.weight [10240, 1280]
    #   transformer_blocks 3.ff.net.2.bias [1280]
    #   transformer_blocks 3.ff.net.2.weight [1280, 5120]
    #   transformer_blocks 3.norm1.bias [1280]
    #   transformer_blocks 3.norm1.weight [1280]
    #   transformer_blocks 3.norm2.bias [1280]
    #   transformer_blocks 3.norm2.weight [1280]
    #   transformer_blocks 3.norm3.bias [1280]
    #   transformer_blocks 3.norm3.weight [1280]
    #   transformer_blocks 4.attn1.to_k.weight [1280, 1280]
    #   transformer_blocks 4.attn1.to_out.0.bias [1280]
    #   transformer_blocks 4.attn1.to_out.0.weight [1280, 1280]
    #   transformer_blocks 4.attn1.to_q.weight [1280, 1280]
    #   transformer_blocks 4.attn1.to_v.weight [1280, 1280]
    #   transformer_blocks 4.attn2.to_k.weight [1280, 2048]
    #   transformer_blocks 4.attn2.to_out.0.bias [1280]
    #   transformer_blocks 4.attn2.to_out.0.weight [1280, 1280]
    #   transformer_blocks 4.attn2.to_q.weight [1280, 1280]
    #   transformer_blocks 4.attn2.to_v.weight [1280, 2048]
    #   transformer_blocks 4.ff.net.0.proj.bias [10240]
    #   transformer_blocks 4.ff.net.0.proj.weight [10240, 1280]
    #   transformer_blocks 4.ff.net.2.bias [1280]
    #   transformer_blocks 4.ff.net.2.weight [1280, 5120]
    #   transformer_blocks 4.norm1.bias [1280]
    #   transformer_blocks 4.norm1.weight [1280]
    #   transformer_blocks 4.norm2.bias [1280]
    #   transformer_blocks 4.norm2.weight [1280]
    #   transformer_blocks 4.norm3.bias [1280]
    #   transformer_blocks 4.norm3.weight [1280]
    #   transformer_blocks 5.attn1.to_k.weight [1280, 1280]
    #   transformer_blocks 5.attn1.to_out.0.bias [1280]
    #   transformer_blocks 5.attn1.to_out.0.weight [1280, 1280]
    #   transformer_blocks 5.attn1.to_q.weight [1280, 1280]
    #   transformer_blocks 5.attn1.to_v.weight [1280, 1280]
    #   transformer_blocks 5.attn2.to_k.weight [1280, 2048]
    #   transformer_blocks 5.attn2.to_out.0.bias [1280]
    #   transformer_blocks 5.attn2.to_out.0.weight [1280, 1280]
    #   transformer_blocks 5.attn2.to_q.weight [1280, 1280]
    #   transformer_blocks 5.attn2.to_v.weight [1280, 2048]
    #   transformer_blocks 5.ff.net.0.proj.bias [10240]
    #   transformer_blocks 5.ff.net.0.proj.weight [10240, 1280]
    #   transformer_blocks 5.ff.net.2.bias [1280]
    #   transformer_blocks 5.ff.net.2.weight [1280, 5120]
    #   transformer_blocks 5.norm1.bias [1280]
    #   transformer_blocks 5.norm1.weight [1280]
    #   transformer_blocks 5.norm2.bias [1280]
    #   transformer_blocks 5.norm2.weight [1280]
    #   transformer_blocks 5.norm3.bias [1280]
    #   transformer_blocks 5.norm3.weight [1280]
    #   transformer_blocks 6.attn1.to_k.weight [1280, 1280]
    #   transformer_blocks 6.attn1.to_out.0.bias [1280]
    #   transformer_blocks 6.attn1.to_out.0.weight [1280, 1280]
    #   transformer_blocks 6.attn1.to_q.weight [1280, 1280]
    #   transformer_blocks 6.attn1.to_v.weight [1280, 1280]
    #   transformer_blocks 6.attn2.to_k.weight [1280, 2048]
    #   transformer_blocks 6.attn2.to_out.0.bias [1280]
    #   transformer_blocks 6.attn2.to_out.0.weight [1280, 1280]
    #   transformer_blocks 6.attn2.to_q.weight [1280, 1280]
    #   transformer_blocks 6.attn2.to_v.weight [1280, 2048]
    #   transformer_blocks 6.ff.net.0.proj.bias [10240]
    #   transformer_blocks 6.ff.net.0.proj.weight [10240, 1280]
    #   transformer_blocks 6.ff.net.2.bias [1280]
    #   transformer_blocks 6.ff.net.2.weight [1280, 5120]
    #   transformer_blocks 6.norm1.bias [1280]
    #   transformer_blocks 6.norm1.weight [1280]
    #   transformer_blocks 6.norm2.bias [1280]
    #   transformer_blocks 6.norm2.weight [1280]
    #   transformer_blocks 6.norm3.bias [1280]
    #   transformer_blocks 6.norm3.weight [1280]
    #   transformer_blocks 7.attn1.to_k.weight [1280, 1280]
    #   transformer_blocks 7.attn1.to_out.0.bias [1280]
    #   transformer_blocks 7.attn1.to_out.0.weight [1280, 1280]
    #   transformer_blocks 7.attn1.to_q.weight [1280, 1280]
    #   transformer_blocks 7.attn1.to_v.weight [1280, 1280]
    #   transformer_blocks 7.attn2.to_k.weight [1280, 2048]
    #   transformer_blocks 7.attn2.to_out.0.bias [1280]
    #   transformer_blocks 7.attn2.to_out.0.weight [1280, 1280]
    #   transformer_blocks 7.attn2.to_q.weight [1280, 1280]
    #   transformer_blocks 7.attn2.to_v.weight [1280, 2048]
    #   transformer_blocks 7.ff.net.0.proj.bias [10240]
    #   transformer_blocks 7.ff.net.0.proj.weight [10240, 1280]
    #   transformer_blocks 7.ff.net.2.bias [1280]
    #   transformer_blocks 7.ff.net.2.weight [1280, 5120]
    #   transformer_blocks 7.norm1.bias [1280]
    #   transformer_blocks 7.norm1.weight [1280]
    #   transformer_blocks 7.norm2.bias [1280]
    #   transformer_blocks 7.norm2.weight [1280]
    #   transformer_blocks 7.norm3.bias [1280]
    #   transformer_blocks 7.norm3.weight [1280]
    #   transformer_blocks 8.attn1.to_k.weight [1280, 1280]
    #   transformer_blocks 8.attn1.to_out.0.bias [1280]
    #   transformer_blocks 8.attn1.to_out.0.weight [1280, 1280]
    #   transformer_blocks 8.attn1.to_q.weight [1280, 1280]
    #   transformer_blocks 8.attn1.to_v.weight [1280, 1280]
    #   transformer_blocks 8.attn2.to_k.weight [1280, 2048]
    #   transformer_blocks 8.attn2.to_out.0.bias [1280]
    #   transformer_blocks 8.attn2.to_out.0.weight [1280, 1280]
    #   transformer_blocks 8.attn2.to_q.weight [1280, 1280]
    #   transformer_blocks 8.attn2.to_v.weight [1280, 2048]
    #   transformer_blocks 8.ff.net.0.proj.bias [10240]
    #   transformer_blocks 8.ff.net.0.proj.weight [10240, 1280]
    #   transformer_blocks 8.ff.net.2.bias [1280]
    #   transformer_blocks 8.ff.net.2.weight [1280, 5120]
    #   transformer_blocks 8.norm1.bias [1280]
    #   transformer_blocks 8.norm1.weight [1280]
    #   transformer_blocks 8.norm2.bias [1280]
    #   transformer_blocks 8.norm2.weight [1280]
    #   transformer_blocks 8.norm3.bias [1280]
    #   transformer_blocks 8.norm3.weight [1280]
    #   transformer_blocks 9.attn1.to_k.weight [1280, 1280]
    #   transformer_blocks 9.attn1.to_out.0.bias [1280]
    #   transformer_blocks 9.attn1.to_out.0.weight [1280, 1280]
    #   transformer_blocks 9.attn1.to_q.weight [1280, 1280]
    #   transformer_blocks 9.attn1.to_v.weight [1280, 1280]
    #   transformer_blocks 9.attn2.to_k.weight [1280, 2048]
    #   transformer_blocks 9.attn2.to_out.0.bias [1280]
    #   transformer_blocks 9.attn2.to_out.0.weight [1280, 1280]
    #   transformer_blocks 9.attn2.to_q.weight [1280, 1280]
    #   transformer_blocks 9.attn2.to_v.weight [1280, 2048]
    #   transformer_blocks 9.ff.net.0.proj.bias [10240]
    #   transformer_blocks 9.ff.net.0.proj.weight [10240, 1280]
    #   transformer_blocks 9.ff.net.2.bias [1280]
    #   transformer_blocks 9.ff.net.2.weight [1280, 5120]
    #   transformer_blocks 9.norm1.bias [1280]
    #   transformer_blocks 9.norm1.weight [1280]
    #   transformer_blocks 9.norm2.bias [1280]
    #   transformer_blocks 9.norm2.weight [1280]
    #   transformer_blocks 9.norm3.bias [1280]
    #   transformer_blocks 9.norm3.weight [1280]
    group_norm(h_cur, input_blocks_1_0_in_layers_0_weight, input_blocks_1_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_1_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_1_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # middle_block.2
    #   emb_layers 1.bias [1280]
    #   emb_layers 1.weight [1280, 1280]
    #   in_layers 0.bias [1280]
    #   in_layers 0.weight [1280]
    #   in_layers 2.bias [1280]
    #   in_layers 2.weight [1280, 1280, 3, 3]
    #   out_layers 0.bias [1280]
    #   out_layers 0.weight [1280]
    #   out_layers 3.bias [1280]
    #   out_layers 3.weight [1280, 1280, 3, 3]
    group_norm(h_cur, input_blocks_2_0_in_layers_0_weight, input_blocks_2_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_2_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_2_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # out.0
    #   bias  [320]
    #   weight  [320]
    group_norm(h_cur, input_blocks_0_0_in_layers_0_weight, input_blocks_0_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_0_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_0_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # out.2
    #   bias  [4]
    #   weight  [4, 320, 3, 3]
    group_norm(h_cur, input_blocks_2_0_in_layers_0_weight, input_blocks_2_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_2_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_2_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # output_blocks.0
    #   0 emb_layers.1.bias [1280]
    #   0 emb_layers.1.weight [1280, 1280]
    #   0 in_layers.0.bias [2560]
    #   0 in_layers.0.weight [2560]
    #   0 in_layers.2.bias [1280]
    #   0 in_layers.2.weight [1280, 2560, 3, 3]
    #   0 out_layers.0.bias [1280]
    #   0 out_layers.0.weight [1280]
    #   0 out_layers.3.bias [1280]
    #   0 out_layers.3.weight [1280, 1280, 3, 3]
    #   0 skip_connection.bias [1280]
    #   0 skip_connection.weight [1280, 2560, 1, 1]
    #   1 norm.bias [1280]
    #   1 norm.weight [1280]
    #   1 proj_in.bias [1280]
    #   1 proj_in.weight [1280, 1280]
    #   1 proj_out.bias [1280]
    #   1 proj_out.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.0.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.0.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.0.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.0.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.0.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.0.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.0.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.0.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.0.ff.net.2.bias [1280]
    #   1 transformer_blocks.0.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.0.norm1.bias [1280]
    #   1 transformer_blocks.0.norm1.weight [1280]
    #   1 transformer_blocks.0.norm2.bias [1280]
    #   1 transformer_blocks.0.norm2.weight [1280]
    #   1 transformer_blocks.0.norm3.bias [1280]
    #   1 transformer_blocks.0.norm3.weight [1280]
    #   1 transformer_blocks.1.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.1.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.1.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.1.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.1.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.1.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.1.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.1.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.1.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.1.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.1.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.1.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.1.ff.net.2.bias [1280]
    #   1 transformer_blocks.1.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.1.norm1.bias [1280]
    #   1 transformer_blocks.1.norm1.weight [1280]
    #   1 transformer_blocks.1.norm2.bias [1280]
    #   1 transformer_blocks.1.norm2.weight [1280]
    #   1 transformer_blocks.1.norm3.bias [1280]
    #   1 transformer_blocks.1.norm3.weight [1280]
    #   1 transformer_blocks.2.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.2.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.2.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.2.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.2.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.2.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.2.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.2.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.2.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.2.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.2.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.2.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.2.ff.net.2.bias [1280]
    #   1 transformer_blocks.2.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.2.norm1.bias [1280]
    #   1 transformer_blocks.2.norm1.weight [1280]
    #   1 transformer_blocks.2.norm2.bias [1280]
    #   1 transformer_blocks.2.norm2.weight [1280]
    #   1 transformer_blocks.2.norm3.bias [1280]
    #   1 transformer_blocks.2.norm3.weight [1280]
    #   1 transformer_blocks.3.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.3.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.3.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.3.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.3.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.3.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.3.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.3.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.3.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.3.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.3.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.3.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.3.ff.net.2.bias [1280]
    #   1 transformer_blocks.3.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.3.norm1.bias [1280]
    #   1 transformer_blocks.3.norm1.weight [1280]
    #   1 transformer_blocks.3.norm2.bias [1280]
    #   1 transformer_blocks.3.norm2.weight [1280]
    #   1 transformer_blocks.3.norm3.bias [1280]
    #   1 transformer_blocks.3.norm3.weight [1280]
    #   1 transformer_blocks.4.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.4.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.4.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.4.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.4.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.4.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.4.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.4.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.4.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.4.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.4.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.4.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.4.ff.net.2.bias [1280]
    #   1 transformer_blocks.4.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.4.norm1.bias [1280]
    #   1 transformer_blocks.4.norm1.weight [1280]
    #   1 transformer_blocks.4.norm2.bias [1280]
    #   1 transformer_blocks.4.norm2.weight [1280]
    #   1 transformer_blocks.4.norm3.bias [1280]
    #   1 transformer_blocks.4.norm3.weight [1280]
    #   1 transformer_blocks.5.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.5.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.5.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.5.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.5.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.5.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.5.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.5.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.5.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.5.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.5.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.5.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.5.ff.net.2.bias [1280]
    #   1 transformer_blocks.5.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.5.norm1.bias [1280]
    #   1 transformer_blocks.5.norm1.weight [1280]
    #   1 transformer_blocks.5.norm2.bias [1280]
    #   1 transformer_blocks.5.norm2.weight [1280]
    #   1 transformer_blocks.5.norm3.bias [1280]
    #   1 transformer_blocks.5.norm3.weight [1280]
    #   1 transformer_blocks.6.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.6.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.6.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.6.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.6.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.6.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.6.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.6.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.6.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.6.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.6.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.6.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.6.ff.net.2.bias [1280]
    #   1 transformer_blocks.6.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.6.norm1.bias [1280]
    #   1 transformer_blocks.6.norm1.weight [1280]
    #   1 transformer_blocks.6.norm2.bias [1280]
    #   1 transformer_blocks.6.norm2.weight [1280]
    #   1 transformer_blocks.6.norm3.bias [1280]
    #   1 transformer_blocks.6.norm3.weight [1280]
    #   1 transformer_blocks.7.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.7.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.7.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.7.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.7.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.7.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.7.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.7.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.7.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.7.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.7.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.7.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.7.ff.net.2.bias [1280]
    #   1 transformer_blocks.7.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.7.norm1.bias [1280]
    #   1 transformer_blocks.7.norm1.weight [1280]
    #   1 transformer_blocks.7.norm2.bias [1280]
    #   1 transformer_blocks.7.norm2.weight [1280]
    #   1 transformer_blocks.7.norm3.bias [1280]
    #   1 transformer_blocks.7.norm3.weight [1280]
    #   1 transformer_blocks.8.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.8.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.8.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.8.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.8.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.8.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.8.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.8.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.8.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.8.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.8.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.8.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.8.ff.net.2.bias [1280]
    #   1 transformer_blocks.8.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.8.norm1.bias [1280]
    #   1 transformer_blocks.8.norm1.weight [1280]
    #   1 transformer_blocks.8.norm2.bias [1280]
    #   1 transformer_blocks.8.norm2.weight [1280]
    #   1 transformer_blocks.8.norm3.bias [1280]
    #   1 transformer_blocks.8.norm3.weight [1280]
    #   1 transformer_blocks.9.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.9.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.9.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.9.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.9.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.9.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.9.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.9.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.9.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.9.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.9.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.9.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.9.ff.net.2.bias [1280]
    #   1 transformer_blocks.9.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.9.norm1.bias [1280]
    #   1 transformer_blocks.9.norm1.weight [1280]
    #   1 transformer_blocks.9.norm2.bias [1280]
    #   1 transformer_blocks.9.norm2.weight [1280]
    #   1 transformer_blocks.9.norm3.bias [1280]
    #   1 transformer_blocks.9.norm3.weight [1280]
    group_norm(h_cur, input_blocks_0_0_in_layers_0_weight, input_blocks_0_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_0_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_0_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # output_blocks.1
    #   0 emb_layers.1.bias [1280]
    #   0 emb_layers.1.weight [1280, 1280]
    #   0 in_layers.0.bias [2560]
    #   0 in_layers.0.weight [2560]
    #   0 in_layers.2.bias [1280]
    #   0 in_layers.2.weight [1280, 2560, 3, 3]
    #   0 out_layers.0.bias [1280]
    #   0 out_layers.0.weight [1280]
    #   0 out_layers.3.bias [1280]
    #   0 out_layers.3.weight [1280, 1280, 3, 3]
    #   0 skip_connection.bias [1280]
    #   0 skip_connection.weight [1280, 2560, 1, 1]
    #   1 norm.bias [1280]
    #   1 norm.weight [1280]
    #   1 proj_in.bias [1280]
    #   1 proj_in.weight [1280, 1280]
    #   1 proj_out.bias [1280]
    #   1 proj_out.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.0.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.0.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.0.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.0.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.0.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.0.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.0.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.0.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.0.ff.net.2.bias [1280]
    #   1 transformer_blocks.0.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.0.norm1.bias [1280]
    #   1 transformer_blocks.0.norm1.weight [1280]
    #   1 transformer_blocks.0.norm2.bias [1280]
    #   1 transformer_blocks.0.norm2.weight [1280]
    #   1 transformer_blocks.0.norm3.bias [1280]
    #   1 transformer_blocks.0.norm3.weight [1280]
    #   1 transformer_blocks.1.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.1.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.1.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.1.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.1.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.1.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.1.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.1.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.1.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.1.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.1.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.1.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.1.ff.net.2.bias [1280]
    #   1 transformer_blocks.1.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.1.norm1.bias [1280]
    #   1 transformer_blocks.1.norm1.weight [1280]
    #   1 transformer_blocks.1.norm2.bias [1280]
    #   1 transformer_blocks.1.norm2.weight [1280]
    #   1 transformer_blocks.1.norm3.bias [1280]
    #   1 transformer_blocks.1.norm3.weight [1280]
    #   1 transformer_blocks.2.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.2.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.2.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.2.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.2.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.2.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.2.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.2.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.2.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.2.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.2.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.2.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.2.ff.net.2.bias [1280]
    #   1 transformer_blocks.2.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.2.norm1.bias [1280]
    #   1 transformer_blocks.2.norm1.weight [1280]
    #   1 transformer_blocks.2.norm2.bias [1280]
    #   1 transformer_blocks.2.norm2.weight [1280]
    #   1 transformer_blocks.2.norm3.bias [1280]
    #   1 transformer_blocks.2.norm3.weight [1280]
    #   1 transformer_blocks.3.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.3.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.3.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.3.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.3.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.3.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.3.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.3.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.3.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.3.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.3.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.3.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.3.ff.net.2.bias [1280]
    #   1 transformer_blocks.3.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.3.norm1.bias [1280]
    #   1 transformer_blocks.3.norm1.weight [1280]
    #   1 transformer_blocks.3.norm2.bias [1280]
    #   1 transformer_blocks.3.norm2.weight [1280]
    #   1 transformer_blocks.3.norm3.bias [1280]
    #   1 transformer_blocks.3.norm3.weight [1280]
    #   1 transformer_blocks.4.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.4.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.4.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.4.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.4.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.4.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.4.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.4.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.4.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.4.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.4.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.4.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.4.ff.net.2.bias [1280]
    #   1 transformer_blocks.4.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.4.norm1.bias [1280]
    #   1 transformer_blocks.4.norm1.weight [1280]
    #   1 transformer_blocks.4.norm2.bias [1280]
    #   1 transformer_blocks.4.norm2.weight [1280]
    #   1 transformer_blocks.4.norm3.bias [1280]
    #   1 transformer_blocks.4.norm3.weight [1280]
    #   1 transformer_blocks.5.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.5.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.5.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.5.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.5.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.5.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.5.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.5.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.5.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.5.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.5.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.5.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.5.ff.net.2.bias [1280]
    #   1 transformer_blocks.5.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.5.norm1.bias [1280]
    #   1 transformer_blocks.5.norm1.weight [1280]
    #   1 transformer_blocks.5.norm2.bias [1280]
    #   1 transformer_blocks.5.norm2.weight [1280]
    #   1 transformer_blocks.5.norm3.bias [1280]
    #   1 transformer_blocks.5.norm3.weight [1280]
    #   1 transformer_blocks.6.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.6.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.6.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.6.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.6.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.6.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.6.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.6.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.6.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.6.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.6.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.6.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.6.ff.net.2.bias [1280]
    #   1 transformer_blocks.6.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.6.norm1.bias [1280]
    #   1 transformer_blocks.6.norm1.weight [1280]
    #   1 transformer_blocks.6.norm2.bias [1280]
    #   1 transformer_blocks.6.norm2.weight [1280]
    #   1 transformer_blocks.6.norm3.bias [1280]
    #   1 transformer_blocks.6.norm3.weight [1280]
    #   1 transformer_blocks.7.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.7.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.7.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.7.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.7.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.7.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.7.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.7.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.7.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.7.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.7.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.7.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.7.ff.net.2.bias [1280]
    #   1 transformer_blocks.7.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.7.norm1.bias [1280]
    #   1 transformer_blocks.7.norm1.weight [1280]
    #   1 transformer_blocks.7.norm2.bias [1280]
    #   1 transformer_blocks.7.norm2.weight [1280]
    #   1 transformer_blocks.7.norm3.bias [1280]
    #   1 transformer_blocks.7.norm3.weight [1280]
    #   1 transformer_blocks.8.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.8.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.8.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.8.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.8.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.8.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.8.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.8.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.8.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.8.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.8.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.8.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.8.ff.net.2.bias [1280]
    #   1 transformer_blocks.8.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.8.norm1.bias [1280]
    #   1 transformer_blocks.8.norm1.weight [1280]
    #   1 transformer_blocks.8.norm2.bias [1280]
    #   1 transformer_blocks.8.norm2.weight [1280]
    #   1 transformer_blocks.8.norm3.bias [1280]
    #   1 transformer_blocks.8.norm3.weight [1280]
    #   1 transformer_blocks.9.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.9.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.9.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.9.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.9.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.9.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.9.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.9.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.9.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.9.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.9.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.9.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.9.ff.net.2.bias [1280]
    #   1 transformer_blocks.9.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.9.norm1.bias [1280]
    #   1 transformer_blocks.9.norm1.weight [1280]
    #   1 transformer_blocks.9.norm2.bias [1280]
    #   1 transformer_blocks.9.norm2.weight [1280]
    #   1 transformer_blocks.9.norm3.bias [1280]
    #   1 transformer_blocks.9.norm3.weight [1280]
    group_norm(h_cur, input_blocks_1_0_in_layers_0_weight, input_blocks_1_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_1_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_1_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # output_blocks.2
    #   0 emb_layers.1.bias [1280]
    #   0 emb_layers.1.weight [1280, 1280]
    #   0 in_layers.0.bias [1920]
    #   0 in_layers.0.weight [1920]
    #   0 in_layers.2.bias [1280]
    #   0 in_layers.2.weight [1280, 1920, 3, 3]
    #   0 out_layers.0.bias [1280]
    #   0 out_layers.0.weight [1280]
    #   0 out_layers.3.bias [1280]
    #   0 out_layers.3.weight [1280, 1280, 3, 3]
    #   0 skip_connection.bias [1280]
    #   0 skip_connection.weight [1280, 1920, 1, 1]
    #   1 norm.bias [1280]
    #   1 norm.weight [1280]
    #   1 proj_in.bias [1280]
    #   1 proj_in.weight [1280, 1280]
    #   1 proj_out.bias [1280]
    #   1 proj_out.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.0.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.0.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.0.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.0.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.0.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.0.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.0.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.0.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.0.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.0.ff.net.2.bias [1280]
    #   1 transformer_blocks.0.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.0.norm1.bias [1280]
    #   1 transformer_blocks.0.norm1.weight [1280]
    #   1 transformer_blocks.0.norm2.bias [1280]
    #   1 transformer_blocks.0.norm2.weight [1280]
    #   1 transformer_blocks.0.norm3.bias [1280]
    #   1 transformer_blocks.0.norm3.weight [1280]
    #   1 transformer_blocks.1.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.1.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.1.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.1.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.1.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.1.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.1.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.1.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.1.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.1.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.1.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.1.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.1.ff.net.2.bias [1280]
    #   1 transformer_blocks.1.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.1.norm1.bias [1280]
    #   1 transformer_blocks.1.norm1.weight [1280]
    #   1 transformer_blocks.1.norm2.bias [1280]
    #   1 transformer_blocks.1.norm2.weight [1280]
    #   1 transformer_blocks.1.norm3.bias [1280]
    #   1 transformer_blocks.1.norm3.weight [1280]
    #   1 transformer_blocks.2.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.2.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.2.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.2.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.2.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.2.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.2.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.2.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.2.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.2.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.2.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.2.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.2.ff.net.2.bias [1280]
    #   1 transformer_blocks.2.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.2.norm1.bias [1280]
    #   1 transformer_blocks.2.norm1.weight [1280]
    #   1 transformer_blocks.2.norm2.bias [1280]
    #   1 transformer_blocks.2.norm2.weight [1280]
    #   1 transformer_blocks.2.norm3.bias [1280]
    #   1 transformer_blocks.2.norm3.weight [1280]
    #   1 transformer_blocks.3.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.3.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.3.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.3.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.3.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.3.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.3.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.3.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.3.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.3.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.3.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.3.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.3.ff.net.2.bias [1280]
    #   1 transformer_blocks.3.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.3.norm1.bias [1280]
    #   1 transformer_blocks.3.norm1.weight [1280]
    #   1 transformer_blocks.3.norm2.bias [1280]
    #   1 transformer_blocks.3.norm2.weight [1280]
    #   1 transformer_blocks.3.norm3.bias [1280]
    #   1 transformer_blocks.3.norm3.weight [1280]
    #   1 transformer_blocks.4.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.4.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.4.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.4.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.4.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.4.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.4.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.4.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.4.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.4.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.4.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.4.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.4.ff.net.2.bias [1280]
    #   1 transformer_blocks.4.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.4.norm1.bias [1280]
    #   1 transformer_blocks.4.norm1.weight [1280]
    #   1 transformer_blocks.4.norm2.bias [1280]
    #   1 transformer_blocks.4.norm2.weight [1280]
    #   1 transformer_blocks.4.norm3.bias [1280]
    #   1 transformer_blocks.4.norm3.weight [1280]
    #   1 transformer_blocks.5.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.5.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.5.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.5.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.5.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.5.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.5.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.5.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.5.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.5.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.5.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.5.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.5.ff.net.2.bias [1280]
    #   1 transformer_blocks.5.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.5.norm1.bias [1280]
    #   1 transformer_blocks.5.norm1.weight [1280]
    #   1 transformer_blocks.5.norm2.bias [1280]
    #   1 transformer_blocks.5.norm2.weight [1280]
    #   1 transformer_blocks.5.norm3.bias [1280]
    #   1 transformer_blocks.5.norm3.weight [1280]
    #   1 transformer_blocks.6.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.6.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.6.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.6.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.6.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.6.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.6.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.6.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.6.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.6.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.6.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.6.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.6.ff.net.2.bias [1280]
    #   1 transformer_blocks.6.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.6.norm1.bias [1280]
    #   1 transformer_blocks.6.norm1.weight [1280]
    #   1 transformer_blocks.6.norm2.bias [1280]
    #   1 transformer_blocks.6.norm2.weight [1280]
    #   1 transformer_blocks.6.norm3.bias [1280]
    #   1 transformer_blocks.6.norm3.weight [1280]
    #   1 transformer_blocks.7.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.7.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.7.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.7.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.7.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.7.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.7.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.7.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.7.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.7.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.7.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.7.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.7.ff.net.2.bias [1280]
    #   1 transformer_blocks.7.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.7.norm1.bias [1280]
    #   1 transformer_blocks.7.norm1.weight [1280]
    #   1 transformer_blocks.7.norm2.bias [1280]
    #   1 transformer_blocks.7.norm2.weight [1280]
    #   1 transformer_blocks.7.norm3.bias [1280]
    #   1 transformer_blocks.7.norm3.weight [1280]
    #   1 transformer_blocks.8.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.8.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.8.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.8.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.8.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.8.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.8.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.8.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.8.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.8.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.8.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.8.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.8.ff.net.2.bias [1280]
    #   1 transformer_blocks.8.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.8.norm1.bias [1280]
    #   1 transformer_blocks.8.norm1.weight [1280]
    #   1 transformer_blocks.8.norm2.bias [1280]
    #   1 transformer_blocks.8.norm2.weight [1280]
    #   1 transformer_blocks.8.norm3.bias [1280]
    #   1 transformer_blocks.8.norm3.weight [1280]
    #   1 transformer_blocks.9.attn1.to_k.weight [1280, 1280]
    #   1 transformer_blocks.9.attn1.to_out.0.bias [1280]
    #   1 transformer_blocks.9.attn1.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.9.attn1.to_q.weight [1280, 1280]
    #   1 transformer_blocks.9.attn1.to_v.weight [1280, 1280]
    #   1 transformer_blocks.9.attn2.to_k.weight [1280, 2048]
    #   1 transformer_blocks.9.attn2.to_out.0.bias [1280]
    #   1 transformer_blocks.9.attn2.to_out.0.weight [1280, 1280]
    #   1 transformer_blocks.9.attn2.to_q.weight [1280, 1280]
    #   1 transformer_blocks.9.attn2.to_v.weight [1280, 2048]
    #   1 transformer_blocks.9.ff.net.0.proj.bias [10240]
    #   1 transformer_blocks.9.ff.net.0.proj.weight [10240, 1280]
    #   1 transformer_blocks.9.ff.net.2.bias [1280]
    #   1 transformer_blocks.9.ff.net.2.weight [1280, 5120]
    #   1 transformer_blocks.9.norm1.bias [1280]
    #   1 transformer_blocks.9.norm1.weight [1280]
    #   1 transformer_blocks.9.norm2.bias [1280]
    #   1 transformer_blocks.9.norm2.weight [1280]
    #   1 transformer_blocks.9.norm3.bias [1280]
    #   1 transformer_blocks.9.norm3.weight [1280]
    #   2 conv.bias [1280]
    #   2 conv.weight [1280, 1280, 3, 3]
    group_norm(h_cur, input_blocks_2_0_in_layers_0_weight, input_blocks_2_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_2_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_2_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # output_blocks.3
    #   0 emb_layers.1.bias [640]
    #   0 emb_layers.1.weight [640, 1280]
    #   0 in_layers.0.bias [1920]
    #   0 in_layers.0.weight [1920]
    #   0 in_layers.2.bias [640]
    #   0 in_layers.2.weight [640, 1920, 3, 3]
    #   0 out_layers.0.bias [640]
    #   0 out_layers.0.weight [640]
    #   0 out_layers.3.bias [640]
    #   0 out_layers.3.weight [640, 640, 3, 3]
    #   0 skip_connection.bias [640]
    #   0 skip_connection.weight [640, 1920, 1, 1]
    #   1 norm.bias [640]
    #   1 norm.weight [640]
    #   1 proj_in.bias [640]
    #   1 proj_in.weight [640, 640]
    #   1 proj_out.bias [640]
    #   1 proj_out.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_k.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_out.0.bias [640]
    #   1 transformer_blocks.0.attn1.to_out.0.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_q.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_v.weight [640, 640]
    #   1 transformer_blocks.0.attn2.to_k.weight [640, 2048]
    #   1 transformer_blocks.0.attn2.to_out.0.bias [640]
    #   1 transformer_blocks.0.attn2.to_out.0.weight [640, 640]
    #   1 transformer_blocks.0.attn2.to_q.weight [640, 640]
    #   1 transformer_blocks.0.attn2.to_v.weight [640, 2048]
    #   1 transformer_blocks.0.ff.net.0.proj.bias [5120]
    #   1 transformer_blocks.0.ff.net.0.proj.weight [5120, 640]
    #   1 transformer_blocks.0.ff.net.2.bias [640]
    #   1 transformer_blocks.0.ff.net.2.weight [640, 2560]
    #   1 transformer_blocks.0.norm1.bias [640]
    #   1 transformer_blocks.0.norm1.weight [640]
    #   1 transformer_blocks.0.norm2.bias [640]
    #   1 transformer_blocks.0.norm2.weight [640]
    #   1 transformer_blocks.0.norm3.bias [640]
    #   1 transformer_blocks.0.norm3.weight [640]
    #   1 transformer_blocks.1.attn1.to_k.weight [640, 640]
    #   1 transformer_blocks.1.attn1.to_out.0.bias [640]
    #   1 transformer_blocks.1.attn1.to_out.0.weight [640, 640]
    #   1 transformer_blocks.1.attn1.to_q.weight [640, 640]
    #   1 transformer_blocks.1.attn1.to_v.weight [640, 640]
    #   1 transformer_blocks.1.attn2.to_k.weight [640, 2048]
    #   1 transformer_blocks.1.attn2.to_out.0.bias [640]
    #   1 transformer_blocks.1.attn2.to_out.0.weight [640, 640]
    #   1 transformer_blocks.1.attn2.to_q.weight [640, 640]
    #   1 transformer_blocks.1.attn2.to_v.weight [640, 2048]
    #   1 transformer_blocks.1.ff.net.0.proj.bias [5120]
    #   1 transformer_blocks.1.ff.net.0.proj.weight [5120, 640]
    #   1 transformer_blocks.1.ff.net.2.bias [640]
    #   1 transformer_blocks.1.ff.net.2.weight [640, 2560]
    #   1 transformer_blocks.1.norm1.bias [640]
    #   1 transformer_blocks.1.norm1.weight [640]
    #   1 transformer_blocks.1.norm2.bias [640]
    #   1 transformer_blocks.1.norm2.weight [640]
    #   1 transformer_blocks.1.norm3.bias [640]
    #   1 transformer_blocks.1.norm3.weight [640]
    group_norm(h_cur, input_blocks_3_0_in_layers_0_weight, input_blocks_3_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_3_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_3_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # output_blocks.4
    #   0 emb_layers.1.bias [640]
    #   0 emb_layers.1.weight [640, 1280]
    #   0 in_layers.0.bias [1280]
    #   0 in_layers.0.weight [1280]
    #   0 in_layers.2.bias [640]
    #   0 in_layers.2.weight [640, 1280, 3, 3]
    #   0 out_layers.0.bias [640]
    #   0 out_layers.0.weight [640]
    #   0 out_layers.3.bias [640]
    #   0 out_layers.3.weight [640, 640, 3, 3]
    #   0 skip_connection.bias [640]
    #   0 skip_connection.weight [640, 1280, 1, 1]
    #   1 norm.bias [640]
    #   1 norm.weight [640]
    #   1 proj_in.bias [640]
    #   1 proj_in.weight [640, 640]
    #   1 proj_out.bias [640]
    #   1 proj_out.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_k.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_out.0.bias [640]
    #   1 transformer_blocks.0.attn1.to_out.0.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_q.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_v.weight [640, 640]
    #   1 transformer_blocks.0.attn2.to_k.weight [640, 2048]
    #   1 transformer_blocks.0.attn2.to_out.0.bias [640]
    #   1 transformer_blocks.0.attn2.to_out.0.weight [640, 640]
    #   1 transformer_blocks.0.attn2.to_q.weight [640, 640]
    #   1 transformer_blocks.0.attn2.to_v.weight [640, 2048]
    #   1 transformer_blocks.0.ff.net.0.proj.bias [5120]
    #   1 transformer_blocks.0.ff.net.0.proj.weight [5120, 640]
    #   1 transformer_blocks.0.ff.net.2.bias [640]
    #   1 transformer_blocks.0.ff.net.2.weight [640, 2560]
    #   1 transformer_blocks.0.norm1.bias [640]
    #   1 transformer_blocks.0.norm1.weight [640]
    #   1 transformer_blocks.0.norm2.bias [640]
    #   1 transformer_blocks.0.norm2.weight [640]
    #   1 transformer_blocks.0.norm3.bias [640]
    #   1 transformer_blocks.0.norm3.weight [640]
    #   1 transformer_blocks.1.attn1.to_k.weight [640, 640]
    #   1 transformer_blocks.1.attn1.to_out.0.bias [640]
    #   1 transformer_blocks.1.attn1.to_out.0.weight [640, 640]
    #   1 transformer_blocks.1.attn1.to_q.weight [640, 640]
    #   1 transformer_blocks.1.attn1.to_v.weight [640, 640]
    #   1 transformer_blocks.1.attn2.to_k.weight [640, 2048]
    #   1 transformer_blocks.1.attn2.to_out.0.bias [640]
    #   1 transformer_blocks.1.attn2.to_out.0.weight [640, 640]
    #   1 transformer_blocks.1.attn2.to_q.weight [640, 640]
    #   1 transformer_blocks.1.attn2.to_v.weight [640, 2048]
    #   1 transformer_blocks.1.ff.net.0.proj.bias [5120]
    #   1 transformer_blocks.1.ff.net.0.proj.weight [5120, 640]
    #   1 transformer_blocks.1.ff.net.2.bias [640]
    #   1 transformer_blocks.1.ff.net.2.weight [640, 2560]
    #   1 transformer_blocks.1.norm1.bias [640]
    #   1 transformer_blocks.1.norm1.weight [640]
    #   1 transformer_blocks.1.norm2.bias [640]
    #   1 transformer_blocks.1.norm2.weight [640]
    #   1 transformer_blocks.1.norm3.bias [640]
    #   1 transformer_blocks.1.norm3.weight [640]
    group_norm(h_cur, input_blocks_4_0_in_layers_0_weight, input_blocks_4_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_4_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_4_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # output_blocks.5
    #   0 emb_layers.1.bias [640]
    #   0 emb_layers.1.weight [640, 1280]
    #   0 in_layers.0.bias [960]
    #   0 in_layers.0.weight [960]
    #   0 in_layers.2.bias [640]
    #   0 in_layers.2.weight [640, 960, 3, 3]
    #   0 out_layers.0.bias [640]
    #   0 out_layers.0.weight [640]
    #   0 out_layers.3.bias [640]
    #   0 out_layers.3.weight [640, 640, 3, 3]
    #   0 skip_connection.bias [640]
    #   0 skip_connection.weight [640, 960, 1, 1]
    #   1 norm.bias [640]
    #   1 norm.weight [640]
    #   1 proj_in.bias [640]
    #   1 proj_in.weight [640, 640]
    #   1 proj_out.bias [640]
    #   1 proj_out.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_k.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_out.0.bias [640]
    #   1 transformer_blocks.0.attn1.to_out.0.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_q.weight [640, 640]
    #   1 transformer_blocks.0.attn1.to_v.weight [640, 640]
    #   1 transformer_blocks.0.attn2.to_k.weight [640, 2048]
    #   1 transformer_blocks.0.attn2.to_out.0.bias [640]
    #   1 transformer_blocks.0.attn2.to_out.0.weight [640, 640]
    #   1 transformer_blocks.0.attn2.to_q.weight [640, 640]
    #   1 transformer_blocks.0.attn2.to_v.weight [640, 2048]
    #   1 transformer_blocks.0.ff.net.0.proj.bias [5120]
    #   1 transformer_blocks.0.ff.net.0.proj.weight [5120, 640]
    #   1 transformer_blocks.0.ff.net.2.bias [640]
    #   1 transformer_blocks.0.ff.net.2.weight [640, 2560]
    #   1 transformer_blocks.0.norm1.bias [640]
    #   1 transformer_blocks.0.norm1.weight [640]
    #   1 transformer_blocks.0.norm2.bias [640]
    #   1 transformer_blocks.0.norm2.weight [640]
    #   1 transformer_blocks.0.norm3.bias [640]
    #   1 transformer_blocks.0.norm3.weight [640]
    #   1 transformer_blocks.1.attn1.to_k.weight [640, 640]
    #   1 transformer_blocks.1.attn1.to_out.0.bias [640]
    #   1 transformer_blocks.1.attn1.to_out.0.weight [640, 640]
    #   1 transformer_blocks.1.attn1.to_q.weight [640, 640]
    #   1 transformer_blocks.1.attn1.to_v.weight [640, 640]
    #   1 transformer_blocks.1.attn2.to_k.weight [640, 2048]
    #   1 transformer_blocks.1.attn2.to_out.0.bias [640]
    #   1 transformer_blocks.1.attn2.to_out.0.weight [640, 640]
    #   1 transformer_blocks.1.attn2.to_q.weight [640, 640]
    #   1 transformer_blocks.1.attn2.to_v.weight [640, 2048]
    #   1 transformer_blocks.1.ff.net.0.proj.bias [5120]
    #   1 transformer_blocks.1.ff.net.0.proj.weight [5120, 640]
    #   1 transformer_blocks.1.ff.net.2.bias [640]
    #   1 transformer_blocks.1.ff.net.2.weight [640, 2560]
    #   1 transformer_blocks.1.norm1.bias [640]
    #   1 transformer_blocks.1.norm1.weight [640]
    #   1 transformer_blocks.1.norm2.bias [640]
    #   1 transformer_blocks.1.norm2.weight [640]
    #   1 transformer_blocks.1.norm3.bias [640]
    #   1 transformer_blocks.1.norm3.weight [640]
    #   2 conv.bias [640]
    #   2 conv.weight [640, 640, 3, 3]
    group_norm(h_cur, input_blocks_5_0_in_layers_0_weight, input_blocks_5_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_5_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_5_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip
# Total: 1680 tensors, 26 blocks


    # output_blocks.6
    #   0 emb_layers.1.bias [320]
    #   0 emb_layers.1.weight [320, 1280]
    #   0 in_layers.0.bias [960]
    #   0 in_layers.0.weight [960]
    #   0 in_layers.2.bias [320]
    #   0 in_layers.2.weight [320, 960, 3, 3]
    #   0 out_layers.0.bias [320]
    #   0 out_layers.0.weight [320]
    #   0 out_layers.3.bias [320]
    #   0 out_layers.3.weight [320, 320, 3, 3]
    #   0 skip_connection.bias [320]
    #   0 skip_connection.weight [320, 960, 1, 1]
    group_norm(h_cur, input_blocks_6_0_in_layers_0_weight, input_blocks_6_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_6_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_6_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # output_blocks.7
    #   0 emb_layers.1.bias [320]
    #   0 emb_layers.1.weight [320, 1280]
    #   0 in_layers.0.bias [640]
    #   0 in_layers.0.weight [640]
    #   0 in_layers.2.bias [320]
    #   0 in_layers.2.weight [320, 640, 3, 3]
    #   0 out_layers.0.bias [320]
    #   0 out_layers.0.weight [320]
    #   0 out_layers.3.bias [320]
    #   0 out_layers.3.weight [320, 320, 3, 3]
    #   0 skip_connection.bias [320]
    #   0 skip_connection.weight [320, 640, 1, 1]
    group_norm(h_cur, input_blocks_7_0_in_layers_0_weight, input_blocks_7_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_7_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_7_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # output_blocks.8
    #   0 emb_layers.1.bias [320]
    #   0 emb_layers.1.weight [320, 1280]
    #   0 in_layers.0.bias [640]
    #   0 in_layers.0.weight [640]
    #   0 in_layers.2.bias [320]
    #   0 in_layers.2.weight [320, 640, 3, 3]
    #   0 out_layers.0.bias [320]
    #   0 out_layers.0.weight [320]
    #   0 out_layers.3.bias [320]
    #   0 out_layers.3.weight [320, 320, 3, 3]
    #   0 skip_connection.bias [320]
    #   0 skip_connection.weight [320, 640, 1, 1]
    group_norm(h_cur, input_blocks_8_0_in_layers_0_weight, input_blocks_8_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_8_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_8_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # time_embed.0
    #   bias  [1280]
    #   weight  [1280, 320]
    group_norm(h_cur, input_blocks_0_0_in_layers_0_weight, input_blocks_0_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_0_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_0_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    # time_embed.2
    #   bias  [1280]
    #   weight  [1280, 1280]
    group_norm(h_cur, input_blocks_2_0_in_layers_0_weight, input_blocks_2_0_in_layers_0_bias, 32, 320, hh*ww)
    arr_silu(h_cur, h_cur, n*320*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1
    _nc = n * _ho * _wo; _kd = 320 * 3 * 3
    _col = make_float_array(_nc * _kd)
    im2col(h_cur, n, 320, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 320)
    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_2_0_in_layers_2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 320:
            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_2_0_in_layers_2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    h_cur = h_cur  # placeholder for skip

    return h_cur
