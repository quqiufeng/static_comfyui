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

