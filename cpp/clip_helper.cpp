#include <torch/torch.h>
#include <cstring>
#include <cstdio>
#include <cmath>

// Forward declarations for types used by CLIP functions
struct STTensor {
    char name[256];
    void* tensor;
};
struct STDict {
    int count;
    STTensor entries[4096];
};

static at::Tensor& unwrap(void* t) {
    return *static_cast<at::Tensor*>(t);
}
static at::Tensor* wrap(at::Tensor t) {
    return new at::Tensor(std::move(t));
}

// ============================================================
// CLIP 文本模型：从 safetensors 权重构建前向计算
// ============================================================

// 从 STDict 中按名称获取 tensor 引用（不 clone，避免额外分配）
static at::Tensor clip_get_tensor(STDict* d, const char* name) {
    for (int i = 0; i < d->count; i++) {
        if (strcmp(d->entries[i].name, name) == 0) {
            return *static_cast<torch::Tensor*>(d->entries[i].tensor);
        }
    }
    std::cerr << "clip: tensor not found: " << name << std::endl;
    return at::Tensor();
}

// CLIP 快速 GELU：x * sigmoid(1.702 * x)
static at::Tensor clip_gelu(const at::Tensor& x) {
    return x * at::sigmoid(1.702 * x);
}

// 单层 Transformer 块
static at::Tensor clip_transformer_layer(
    const at::Tensor& x,
    const at::Tensor& ln1_w, const at::Tensor& ln1_b,
    const at::Tensor& q_w, const at::Tensor& q_b,
    const at::Tensor& k_w, const at::Tensor& k_b,
    const at::Tensor& v_w, const at::Tensor& v_b,
    const at::Tensor& out_w, const at::Tensor& out_b,
    const at::Tensor& ln2_w, const at::Tensor& ln2_b,
    const at::Tensor& fc1_w, const at::Tensor& fc1_b,
    const at::Tensor& fc2_w, const at::Tensor& fc2_b,
    int n_heads) {
    int d_model = x.size(2);
    int head_dim = d_model / n_heads;

    // Self-attention with pre-LN
    auto residual = x;
    auto h = at::layer_norm(x, {d_model}, ln1_w, ln1_b, 1e-5);
    int B = h.size(0), N = h.size(1);
    auto q = at::matmul(h.reshape({B, N, d_model}), q_w.t()) + q_b;
    auto k = at::matmul(h.reshape({B, N, d_model}), k_w.t()) + k_b;
    auto v = at::matmul(h.reshape({B, N, d_model}), v_w.t()) + v_b;
    q = q.reshape({B, N, n_heads, head_dim}).transpose(1, 2);
    k = k.reshape({B, N, n_heads, head_dim}).transpose(1, 2);
    v = v.reshape({B, N, n_heads, head_dim}).transpose(1, 2);
    auto attn = at::matmul(q, k.transpose(-2, -1)) / std::sqrt((double)head_dim);
    attn = at::softmax(attn, -1);
    auto attn_out = at::matmul(attn, v).transpose(1, 2).reshape({B, N, d_model});
    attn_out = at::matmul(attn_out, out_w.t()) + out_b;
    h = residual + attn_out;

    // MLP with pre-LN
    residual = h;
    h = at::layer_norm(h, {d_model}, ln2_w, ln2_b, 1e-5);
    h = at::matmul(h, fc1_w.t()) + fc1_b;
    h = clip_gelu(h);
    h = at::matmul(h, fc2_w.t()) + fc2_b;
    h = residual + h;
    return h;
}

// 完整的 CLIP 文本模型 forward（从 safetensors 权重读取）
// 返回 (1, 77, d_model) text embeddings
static at::Tensor clip_text_forward_from_dict(
    STDict* dict, const at::Tensor& token_ids,
    int d_model, int n_layers, int n_heads, int d_ffn) {
    using namespace at;

    // 权重名称前缀映射
    auto load = [&](const char* name) -> at::Tensor {
        return clip_get_tensor(dict, name);
    };

    // Embeddings
    auto tok_emb_w = load("text_model.embeddings.token_embedding.weight");
    auto pos_emb_w = load("text_model.embeddings.position_embedding.weight");
    auto final_ln_w = load("text_model.final_layer_norm.weight");
    auto final_ln_b = load("text_model.final_layer_norm.bias");

    // Token embedding
    auto input_ids = token_ids;
    if (input_ids.dim() == 1) input_ids = input_ids.unsqueeze(0);
    if (input_ids.scalar_type() != torch::kInt64) input_ids = input_ids.to(torch::kInt64);

    auto x = at::embedding(tok_emb_w, input_ids);
    auto pos = at::arange(input_ids.size(1), input_ids.device()).unsqueeze(0);
    x = x + at::embedding(pos_emb_w, pos);

    // Transformer layers
    for (int i = 0; i < n_layers; i++) {
        char buf[256];
        auto ln1_w = load((snprintf(buf, sizeof(buf), "text_model.encoder.layers.%d.layer_norm1.weight", i), buf));
        auto ln1_b = load((snprintf(buf, sizeof(buf), "text_model.encoder.layers.%d.layer_norm1.bias", i), buf));
        auto q_w  = load((snprintf(buf, sizeof(buf), "text_model.encoder.layers.%d.self_attn.q_proj.weight", i), buf));
        auto q_b  = load((snprintf(buf, sizeof(buf), "text_model.encoder.layers.%d.self_attn.q_proj.bias", i), buf));
        auto k_w  = load((snprintf(buf, sizeof(buf), "text_model.encoder.layers.%d.self_attn.k_proj.weight", i), buf));
        auto k_b  = load((snprintf(buf, sizeof(buf), "text_model.encoder.layers.%d.self_attn.k_proj.bias", i), buf));
        auto v_w  = load((snprintf(buf, sizeof(buf), "text_model.encoder.layers.%d.self_attn.v_proj.weight", i), buf));
        auto v_b  = load((snprintf(buf, sizeof(buf), "text_model.encoder.layers.%d.self_attn.v_proj.bias", i), buf));
        auto o_w  = load((snprintf(buf, sizeof(buf), "text_model.encoder.layers.%d.self_attn.out_proj.weight", i), buf));
        auto o_b  = load((snprintf(buf, sizeof(buf), "text_model.encoder.layers.%d.self_attn.out_proj.bias", i), buf));
        auto ln2_w = load((snprintf(buf, sizeof(buf), "text_model.encoder.layers.%d.layer_norm2.weight", i), buf));
        auto ln2_b = load((snprintf(buf, sizeof(buf), "text_model.encoder.layers.%d.layer_norm2.bias", i), buf));
        auto fc1_w = load((snprintf(buf, sizeof(buf), "text_model.encoder.layers.%d.mlp.fc1.weight", i), buf));
        auto fc1_b = load((snprintf(buf, sizeof(buf), "text_model.encoder.layers.%d.mlp.fc1.bias", i), buf));
        auto fc2_w = load((snprintf(buf, sizeof(buf), "text_model.encoder.layers.%d.mlp.fc2.weight", i), buf));
        auto fc2_b = load((snprintf(buf, sizeof(buf), "text_model.encoder.layers.%d.mlp.fc2.bias", i), buf));
        x = clip_transformer_layer(x, ln1_w, ln1_b, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b,
                                   ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b, n_heads);
    }

    // Final layer norm
    x = at::layer_norm(x, {d_model}, final_ln_w, final_ln_b, 1e-5);
    return x.clone();  // ensure owning tensor, no dangling views
}

// 公开 API：从 safetensors 权重执行 CLIP 文本编码
// 输入：clip_dict - safetensors handle (STDict*)
//       token_ids - wrapped tensor (1,77) int64
//       d_model / n_layers / n_heads / d_ffn - 架构参数
// 返回：wrapped tensor (1,77,d_model)
extern "C" void* torch_std_clip_text_forward_from_dict(void* clip_dict, void* token_ids,
                                            int d_model, int n_layers,
                                            int n_heads, int d_ffn) {
    try {
        auto* dict = static_cast<STDict*>(clip_dict);
        auto& tokens = unwrap(token_ids);
        auto result = clip_text_forward_from_dict(dict, tokens, d_model, n_layers, n_heads, d_ffn);
        return wrap(result);
    } catch (const std::exception& e) {
        std::cerr << "clip_text_forward_from_dict error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================
// VAE Decoder: 从 safetensors 权重构建前向计算
// ============================================================

// 从 STDict 按名称获取 tensor
static at::Tensor vae_get_tensor(STDict* d, const char* name) {
    // Try exact name first
    for (int i = 0; i < d->count; i++) {
        if (strcmp(d->entries[i].name, name) == 0)
            return *static_cast<torch::Tensor*>(d->entries[i].tensor);
    }
    // Try with first_stage_model. prefix (checkpoint format)
    char buf[256];
    snprintf(buf, sizeof(buf), "first_stage_model.%s", name);
    for (int i = 0; i < d->count; i++) {
        if (strcmp(d->entries[i].name, buf) == 0)
            return *static_cast<torch::Tensor*>(d->entries[i].tensor);
    }
    std::cerr << "vae: tensor not found: " << name << std::endl;
    return at::Tensor();
}

// Load VAE weight and convert to fp32
static at::Tensor vae_get_tensor_f32(STDict* d, const char* name) {
    auto t = vae_get_tensor(d, name);
    if (!t.defined()) throw std::runtime_error(std::string("VAE tensor not found: ") + name);
    return t.to(torch::kFloat32);
}

// GroupNorm + silu
static at::Tensor vae_gn_silu(const at::Tensor& x, const at::Tensor& w,
                               const at::Tensor& b, int groups) {
    int C = x.size(1);
    return at::silu(at::group_norm(x, groups, w, b, 1e-6));
}

// ResnetBlock: norm → silu → conv → norm → silu → conv + skip
static at::Tensor vae_resblock(STDict* d, const at::Tensor& x,
                                const std::string& prefix, int ch) {
    char buf[256];
    auto norm1_w = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.norm1.weight", prefix.c_str()), buf));
    auto norm1_b = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.norm1.bias", prefix.c_str()), buf));
    auto conv1_w = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.conv1.weight", prefix.c_str()), buf));
    auto conv1_b = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.conv1.bias", prefix.c_str()), buf));
    auto norm2_w = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.norm2.weight", prefix.c_str()), buf));
    auto norm2_b = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.norm2.bias", prefix.c_str()), buf));
    auto conv2_w = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.conv2.weight", prefix.c_str()), buf));
    auto conv2_b = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.conv2.bias", prefix.c_str()), buf));
    if (!conv1_w.defined()) throw std::runtime_error(std::string("VAE missing: ") + prefix + ".conv1.weight");

    // nin_shortcut if channel changes (optional, use non-throwing lookup)
    bool has_shortcut = false;
    at::Tensor shortcut_w, shortcut_b;
    auto t = vae_get_tensor(d, (snprintf(buf, sizeof(buf), "%s.nin_shortcut.weight", prefix.c_str()), buf));
    if (t.defined()) {
        has_shortcut = true;
        shortcut_w = t.to(torch::kFloat32);
        shortcut_b = vae_get_tensor(d, (snprintf(buf, sizeof(buf), "%s.nin_shortcut.bias", prefix.c_str()), buf)).to(torch::kFloat32);
    }

    auto h = vae_gn_silu(x, norm1_w, norm1_b, 32);
    h = at::conv2d(h, conv1_w, conv1_b, at::IntArrayRef{1,1}, at::IntArrayRef{1,1});
    h = vae_gn_silu(h, norm2_w, norm2_b, 32);
    h = at::conv2d(h, conv2_w, conv2_b, at::IntArrayRef{1,1}, at::IntArrayRef{1,1});
    if (has_shortcut)
        h = h + at::conv2d(x, shortcut_w, shortcut_b, at::IntArrayRef{1,1}, at::IntArrayRef{0,0});
    else
        h = h + x;
    return h;
}

// Spatial attention block
static at::Tensor vae_attn(STDict* d, const at::Tensor& x,
                             const std::string& prefix) {
    char buf[256];
    auto norm_w = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.norm.weight", prefix.c_str()), buf));
    auto norm_b = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.norm.bias", prefix.c_str()), buf));
    auto q_w = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.q.weight", prefix.c_str()), buf));
    auto q_b = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.q.bias", prefix.c_str()), buf));
    auto k_w = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.k.weight", prefix.c_str()), buf));
    auto k_b = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.k.bias", prefix.c_str()), buf));
    auto v_w = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.v.weight", prefix.c_str()), buf));
    auto v_b = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.v.bias", prefix.c_str()), buf));
    auto proj_w = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.proj_out.weight", prefix.c_str()), buf));
    auto proj_b = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.proj_out.bias", prefix.c_str()), buf));

    int C = x.size(1);
    auto h = at::group_norm(x, 32, norm_w, norm_b, 1e-6);
    int B = h.size(0), H = h.size(2), W = h.size(3);
    auto q = at::conv2d(h, q_w, q_b, at::IntArrayRef{1,1}, at::IntArrayRef{0,0}).view({B, C, H*W}).transpose(1,2);
    auto k = at::conv2d(h, k_w, k_b, at::IntArrayRef{1,1}, at::IntArrayRef{0,0}).view({B, C, H*W}).transpose(1,2);
    auto v = at::conv2d(h, v_w, v_b, at::IntArrayRef{1,1}, at::IntArrayRef{0,0}).view({B, C, H*W}).transpose(1,2);
    auto attn = at::softmax(q.matmul(k.transpose(-2,-1)) / std::sqrt((double)C), -1);
    h = attn.matmul(v).transpose(1,2).view({B, C, H, W});
    h = at::conv2d(h, proj_w, proj_b, at::IntArrayRef{1,1}, at::IntArrayRef{0,0});
    return x + h;
}

// Downsample block: conv with stride=2 or average pool
static at::Tensor vae_downsample(STDict* d, const at::Tensor& x, const std::string& prefix) {
    char buf[256];
    auto w = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.op.weight", prefix.c_str()), buf));
    auto b = vae_get_tensor_f32(d, (snprintf(buf, sizeof(buf), "%s.op.bias", prefix.c_str()), buf));
    return at::conv2d(x, w, b, at::IntArrayRef{2,2}, at::IntArrayRef{0,0});
}

// 纯 Decoder forward（对齐 ComfyUI Decoder 类）
static at::Tensor vae_decoder_forward(STDict* dict, const at::Tensor& z) {
    auto load = [&](const char* name) {
        char buf[256];
        snprintf(buf, sizeof(buf), "first_stage_model.%s", name);
        auto t = vae_get_tensor_f32(dict, buf);
        if (t.defined()) return t;
        return vae_get_tensor_f32(dict, name);
    };

    auto h = z.to(torch::kCPU).to(torch::kFloat32) / 0.13025;
    h = at::conv2d(h, load("decoder.conv_in.weight"), load("decoder.conv_in.bias"),
                   at::IntArrayRef{1,1}, at::IntArrayRef{1,1});

    h = vae_resblock(dict, h, "first_stage_model.decoder.mid.block_1", 0);
    h = vae_attn(dict, h, "first_stage_model.decoder.mid.attn_1");
    h = vae_resblock(dict, h, "first_stage_model.decoder.mid.block_2", 0);

    // Up blocks: 4 levels, 3 resblocks each, upsample for level>0
    for (int level = 3; level >= 0; level--) {
        for (int b = 0; b < 3; b++) {
            char key[128];
            snprintf(key, sizeof(key), "first_stage_model.decoder.up.%d.block.%d", level, b);
            h = vae_resblock(dict, h, key, 0);
        }
        if (level > 0) {
            namespace F = torch::nn::functional;
            h = F::interpolate(h, F::InterpolateFuncOptions()
                .scale_factor(std::vector<double>{2.0, 2.0}).mode(torch::kNearest));
            char wk[128], bk[128];
            snprintf(wk, sizeof(wk), "first_stage_model.decoder.up.%d.upsample.conv.weight", level);
            snprintf(bk, sizeof(bk), "first_stage_model.decoder.up.%d.upsample.conv.bias", level);
            h = at::conv2d(h, vae_get_tensor_f32(dict, wk), vae_get_tensor_f32(dict, bk),
                           at::IntArrayRef{1,1}, at::IntArrayRef{1,1});
        }
    }

    // norm_out + conv_out
    h = at::silu(at::group_norm(h, 32, load("decoder.norm_out.weight"), load("decoder.norm_out.bias"), 1e-6));
    h = at::conv2d(h, load("decoder.conv_out.weight"), load("decoder.conv_out.bias"),
                   at::IntArrayRef{1,1}, at::IntArrayRef{1,1});
    return h;
}

extern "C" void* torch_std_euler_step(void* x_ptr, void* sigma_t_ptr, void* sigma_next_ptr,
                                       void* cond_ptr, void* uncond_ptr, double cfg) {
    try {
        auto& x = unwrap(x_ptr);
        auto& sig_t = unwrap(sigma_t_ptr);
        auto& sig_next = unwrap(sigma_next_ptr);
        auto& cond = unwrap(cond_ptr);
        auto& uncond = unwrap(uncond_ptr);
        auto dev = x.device();
        torch::Tensor eps = uncond + (cond - uncond) * cfg;
        auto sig_t_d = sig_t.to(dev);
        auto sig_next_d = sig_next.to(dev);
        auto result = x + (sig_next_d - sig_t_d) * eps;
        return wrap(result);
    } catch (const std::exception& e) {
        std::cerr << "euler_step error: " << e.what() << std::endl;
        return nullptr;
    }
}

extern "C" void* torch_std_vae_decode_from_dict(void* vae_dict, void* latent) {
    try {
        auto* dict = static_cast<STDict*>(vae_dict);
        auto& lat = unwrap(latent);
        auto result = vae_decoder_forward(dict, lat);
        return wrap(result.to(torch::kFloat16).to(torch::kCUDA));
    } catch (const std::exception& e) {
        std::cerr << "vae_decode_from_dict error: " << e.what() << std::endl;
        return nullptr;
    }
}

