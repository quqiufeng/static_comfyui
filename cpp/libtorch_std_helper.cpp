#include "libtorch_std_helper.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <unistd.h>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <c10/util/Exception.h>

using namespace at;

// 过滤已知的 cuDNN RNN contiguous weights 警告，避免训练输出被刷屏
class QuietRNNWarningHandler : public c10::WarningHandler {
public:
    void process(const c10::Warning& warning) override {
        std::string s(warning.msg());
        if (s.find("RNN module weights are not part of single contiguous chunk") != std::string::npos) {
            return;
        }
        c10::WarningHandler::process(warning);
    }
};

static QuietRNNWarningHandler g_quiet_rnn_warning;

static bool g_warning_handler_installed = false;
static void install_quiet_rnn_warning() {
    if (!g_warning_handler_installed) {
        c10::WarningUtils::set_warning_handler(&g_quiet_rnn_warning);
        g_warning_handler_installed = true;
    }
}

// ============================================================
// 工具函数
// ============================================================
static void log_tensor(const at::Tensor& t, const char* name) {
    char buf[256];
    int n = snprintf(buf, sizeof(buf), "CMP %-30s shape=%d,%d,%d,%d mean=%.4f std=%.4f\n",
        name, (int)t.size(0), (int)t.size(1), (int)t.size(2), (int)t.size(3),
        t.mean().item<float>(), t.std().item<float>());
    write(STDERR_FILENO, buf, n);
}
static torch::TensorOptions make_options(int dtype) {
    switch (dtype) {
        case TORCH_DTYPE_FLOAT32: return torch::TensorOptions().dtype(torch::kFloat32);
        case TORCH_DTYPE_FLOAT64: return torch::TensorOptions().dtype(torch::kFloat64);
        case TORCH_DTYPE_INT32:   return torch::TensorOptions().dtype(torch::kInt32);
        case TORCH_DTYPE_INT64:   return torch::TensorOptions().dtype(torch::kInt64);
        default:                  return torch::TensorOptions().dtype(torch::kFloat32);
    }
}

static c10::ScalarType dtype_to_scalar(int dtype) {
    switch (dtype) {
        case TORCH_DTYPE_FLOAT32: return torch::kFloat32;
        case TORCH_DTYPE_FLOAT64: return torch::kFloat64;
        case TORCH_DTYPE_INT32:   return torch::kInt32;
        case TORCH_DTYPE_INT64:   return torch::kInt64;
        case 5:                   return torch::kHalf;
        case 6:                   return torch::kBFloat16;
        default:                  return torch::kFloat32;
    }
}

static torch::Tensor* wrap(at::Tensor t) {
    return new torch::Tensor(std::move(t));
}

static torch::Tensor& unwrap(void* t) {
    return *static_cast<torch::Tensor*>(t);
}

static std::vector<int64_t> to_shape(int64_t* shape, int ndim) {
    return std::vector<int64_t>(shape, shape + ndim);
}

static std::string to_str(const char* s) {
    return s ? std::string(s) : "mean";
}

static torch::Reduction::Reduction to_reduction(const char* s) {
    std::string r = to_str(s);
    if (r == "none" || r == "None") return torch::Reduction::None;
    if (r == "sum" || r == "Sum") return torch::Reduction::Sum;
    return torch::Reduction::Mean;
}

// ============================================================
// 张量创建与转换
// ============================================================
void* torch_std_tensor_from_blob(void* data, int64_t* shape, int ndim, int dtype) {
    try {
        auto s = to_shape(shape, ndim);
        auto opts = make_options(dtype).requires_grad(false);
        auto t = torch::from_blob(data, s, opts).clone();
        return wrap(t);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_tensor_from_blob error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_tensor_from_blob_3d(void* data, int d0, int d1, int d2, int dtype) {
    int64_t shape[3] = {d0, d1, d2};
    return torch_std_tensor_from_blob(data, shape, 3, dtype);
}

void* torch_std_zeros(int64_t* shape, int ndim, int dtype) {
    try {
        return wrap(torch::zeros(to_shape(shape, ndim), torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA)));
    } catch (const std::exception& e) {
        std::cerr << "torch_std_zeros error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_ones(int64_t* shape, int ndim, int dtype) {
    try {
        return wrap(torch::ones(to_shape(shape, ndim), make_options(dtype).device(torch::kCUDA)));
    } catch (const std::exception& e) {
        std::cerr << "torch_std_ones error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_empty(int64_t* shape, int ndim, int dtype) {
    try {
        return wrap(torch::empty(to_shape(shape, ndim), make_options(dtype)));
    } catch (const std::exception& e) {
        std::cerr << "torch_std_empty error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_full(int64_t* shape, int ndim, double value, int dtype) {
    try {
        return wrap(torch::full(to_shape(shape, ndim), value, torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA)));
    } catch (const std::exception& e) {
        write(2, "FULL_ERROR ", 10);
        write(2, e.what(), strlen(e.what()));
        write(2, "\n", 1);
        return nullptr;
    }
}

void* torch_std_randn(int64_t* shape, int ndim, int dtype) {
    try {
        return wrap(torch::randn(to_shape(shape, ndim), torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA)));
    } catch (const std::exception& e) {
        std::cerr << "torch_std_randn error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_randint(int64_t low, int64_t high, int64_t* shape, int ndim, int dtype) {
    try {
        return wrap(torch::randint(low, high, to_shape(shape, ndim), make_options(dtype).device(torch::kCUDA)));
    } catch (const std::exception& e) {
		std::cerr << "torch_std_randint error: " << e.what() << std::endl;
		return nullptr;
	}
}

void* torch_std_arange(int64_t start, int64_t end, int64_t step, int dtype) {
	try {
		return wrap(torch::arange(start, end, step, make_options(dtype)));
	} catch (const std::exception& e) {
		std::cerr << "torch_std_arange error: " << e.what() << std::endl;
		return nullptr;
	}
}

void* torch_std_clone(void* t) {
    try {
        return wrap(unwrap(t).clone());
    } catch (const std::exception& e) {
        std::cerr << "torch_std_clone error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_detach(void* t) {
    try {
        return wrap(unwrap(t).detach().clone());
    } catch (const std::exception& e) {
        std::cerr << "torch_std_detach error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_to_dtype(void* t, int dtype) {
    try {
        return wrap(unwrap(t).to(dtype_to_scalar(dtype)));
    } catch (const std::exception& e) {
        std::cerr << "torch_std_to_dtype error: " << e.what() << std::endl;
        return nullptr;
    }
}

void torch_std_delete_tensor(void* t) {
    delete static_cast<torch::Tensor*>(t);
}

int64_t torch_std_numel(void* t) {
    try {
        return unwrap(t).numel();
    } catch (...) { return 0; }
}

int torch_std_ndim(void* t) {
    try {
        return unwrap(t).dim();
    } catch (...) { return 0; }
}

void torch_std_shape(void* t, int64_t* out) {
    try {
        auto& tensor = unwrap(t);
        for (int i = 0; i < tensor.dim(); i++) {
            out[i] = tensor.size(i);
        }
    } catch (...) {}
}

void torch_std_to_double_array(void* t, double* out, int64_t n) {
    try {
        auto tensor = unwrap(t).to(torch::kCPU).to(torch::kFloat64).contiguous();
        int64_t copy_n = std::min(n, tensor.numel());
        std::memcpy(out, tensor.data_ptr<double>(), copy_n * sizeof(double));
    } catch (const std::exception& e) {
        std::cerr << "torch_std_to_double_array error: " << e.what() << std::endl;
    }
}

void torch_std_to_float_array(void* t, float* out, int64_t n) {
    try {
        auto tensor = unwrap(t).to(torch::kCPU).to(torch::kFloat32).contiguous();
        int64_t copy_n = std::min(n, tensor.numel());
        std::memcpy(out, tensor.data_ptr<float>(), copy_n * sizeof(float));
    } catch (const std::exception& e) {
        std::cerr << "torch_std_to_float_array error: " << e.what() << std::endl;
    }
}

void torch_std_to_int64_array(void* t, int64_t* out, int64_t n) {
    try {
        auto tensor = unwrap(t).to(torch::kCPU).to(torch::kInt64).contiguous();
        int64_t copy_n = std::min(n, tensor.numel());
        std::memcpy(out, tensor.data_ptr<int64_t>(), copy_n * sizeof(int64_t));
    } catch (const std::exception& e) {
        std::cerr << "torch_std_to_int64_array error: " << e.what() << std::endl;
    }
}

void torch_std_copy_probs(void* t, double* out, int n) {
    torch_std_to_double_array(t, out, n);
}

// ============================================================
// StaticPy array -> C pointer helpers
// ============================================================
void* torch_std_float_array_ptr(void* arr) { return arr; }
void* torch_std_int_array_ptr(void* arr) { return arr; }
void* torch_std_float_array_ptr_offset(void* arr, int offset) {
    return static_cast<char*>(arr) + offset * sizeof(double);
}
void* torch_std_int_array_ptr_offset(void* arr, int offset) {
    return static_cast<char*>(arr) + offset * sizeof(int64_t);
}

// ============================================================
// 数学运算
// ============================================================
void* torch_std_add(void* a, void* b) { try { return wrap(unwrap(a) + unwrap(b)); } catch (...) { return nullptr; } }
void* torch_std_sub(void* a, void* b) { try { return wrap(unwrap(a) - unwrap(b)); } catch (...) { return nullptr; } }
void* torch_std_mul(void* a, void* b) { try { return wrap(unwrap(a) * unwrap(b)); } catch (...) { return nullptr; } }
void* torch_std_div(void* a, void* b) { try { return wrap(unwrap(a) / unwrap(b)); } catch (...) { return nullptr; } }
void* torch_std_pow(void* a, double exp) { try { return wrap(unwrap(a).pow(exp)); } catch (...) { return nullptr; } }
void* torch_std_exp(void* a) { try { return wrap(unwrap(a).exp()); } catch (...) { return nullptr; } }
void* torch_std_log(void* a) { try { return wrap(unwrap(a).log()); } catch (...) { return nullptr; } }
void* torch_std_sqrt(void* a) { try { return wrap(unwrap(a).sqrt()); } catch (...) { return nullptr; } }
void* torch_std_neg(void* a) { try { return wrap(-unwrap(a)); } catch (...) { return nullptr; } }
void* torch_std_abs(void* a) { try { return wrap(unwrap(a).abs()); } catch (...) { return nullptr; } }
void* torch_std_cos(void* a) { try { return wrap(unwrap(a).cos()); } catch (...) { return nullptr; } }
void* torch_std_sin(void* a) { try { return wrap(unwrap(a).sin()); } catch (...) { return nullptr; } }
void* torch_std_mul_scalar(void* a, double s) { try { return wrap(unwrap(a) * s); } catch (...) { return nullptr; } }
void* torch_std_add_scalar(void* a, double s) { try { return wrap(unwrap(a) + s); } catch (...) { return nullptr; } }

// ============================================================
// 激活函数
// ============================================================
void* torch_std_relu(void* a) { try { return wrap(unwrap(a).relu()); } catch (...) { return nullptr; } }
void* torch_std_leaky_relu(void* a, double negative_slope) { try { return wrap(torch::nn::functional::leaky_relu(unwrap(a), torch::nn::functional::LeakyReLUFuncOptions().negative_slope(negative_slope))); } catch (...) { return nullptr; } }
void* torch_std_sigmoid(void* a) { try { return wrap(unwrap(a).sigmoid()); } catch (...) { return nullptr; } }
void* torch_std_tanh(void* a) { try { return wrap(unwrap(a).tanh()); } catch (...) { return nullptr; } }
void* torch_std_softmax(void* a, int64_t dim) { try { return wrap(torch::softmax(unwrap(a), dim)); } catch (...) { return nullptr; } }
void* torch_std_log_softmax(void* a, int64_t dim) { try { return wrap(torch::log_softmax(unwrap(a), dim)); } catch (...) { return nullptr; } }

// ============================================================
// 归约
// ============================================================
void* torch_std_sum(void* a) { try { return wrap(unwrap(a).sum()); } catch (...) { return nullptr; } }
void* torch_std_sum_dim(void* a, int64_t dim, int keepdim) { try { return wrap(unwrap(a).sum(dim, keepdim != 0)); } catch (...) { return nullptr; } }
void* torch_std_mean(void* a) { try { return wrap(unwrap(a).mean()); } catch (...) { return nullptr; } }
void* torch_std_mean_dim(void* a, int64_t dim, int keepdim) { try { return wrap(unwrap(a).mean(dim, keepdim != 0)); } catch (...) { return nullptr; } }
void* torch_std_max(void* a) { try { return wrap(unwrap(a).max()); } catch (...) { return nullptr; } }
void* torch_std_max_dim(void* a, int64_t dim, int keepdim) { try { return wrap(std::get<0>(unwrap(a).max(dim, keepdim != 0))); } catch (...) { return nullptr; } }
void* torch_std_min(void* a) { try { return wrap(unwrap(a).min()); } catch (...) { return nullptr; } }
void* torch_std_min_dim(void* a, int64_t dim, int keepdim) { try { return wrap(std::get<0>(unwrap(a).min(dim, keepdim != 0))); } catch (...) { return nullptr; } }

// ============================================================
// 索引与采样
// ============================================================
int64_t torch_std_argmax(void* a) {
    try { return unwrap(a).argmax().item<int64_t>(); } catch (...) { return -1; }
}

int64_t torch_std_argmax_dim1(void* a, int64_t dim) {
    try { return unwrap(a).argmax(dim).item<int64_t>(); } catch (...) { return -1; }
}

void* torch_std_multinomial(void* probs, int64_t num_samples, int replacement) {
    try { return wrap(torch::multinomial(unwrap(probs), num_samples, replacement != 0)); } catch (...) { return nullptr; }
}

void* torch_std_gather(void* input, int64_t dim, void* index) {
    try { return wrap(unwrap(input).gather(dim, unwrap(index))); } catch (...) { return nullptr; }
}

void* torch_std_index_select(void* input, int64_t dim, void* index) {
    try { return wrap(unwrap(input).index_select(dim, unwrap(index))); } catch (...) { return nullptr; }
}

void* torch_std_index_tensor(void* input, void* index) {
    try { return wrap(unwrap(input).index({unwrap(index)})); } catch (...) { return nullptr; }
}

// ============================================================
// 形状操作
// ============================================================
void* torch_std_reshape(void* a, int64_t* shape, int ndim) { try { return wrap(unwrap(a).reshape(to_shape(shape, ndim))); } catch (...) { return nullptr; } }
void* torch_std_transpose(void* a, int64_t dim0, int64_t dim1) { try { return wrap(unwrap(a).transpose(dim0, dim1)); } catch (...) { return nullptr; } }
void* torch_std_permute(void* a, int64_t* dims, int ndim) { try { return wrap(unwrap(a).permute(at::IntArrayRef(dims, ndim))); } catch (...) { return nullptr; } }
void* torch_std_squeeze(void* a, int64_t dim) { try { return wrap(unwrap(a).squeeze(dim)); } catch (...) { return nullptr; } }
void* torch_std_unsqueeze(void* a, int64_t dim) { try { return wrap(unwrap(a).unsqueeze(dim)); } catch (...) { return nullptr; } }

void* torch_std_cat(void** tensors, int n, int64_t dim) {
    try {
        std::vector<at::Tensor> v;
        for (int i = 0; i < n; i++) v.push_back(unwrap(tensors[i]));
        return wrap(torch::cat(v, dim));
    } catch (...) { return nullptr; }
}

void* torch_std_stack(void** tensors, int n, int64_t dim) {
    try {
        std::vector<at::Tensor> v;
        for (int i = 0; i < n; i++) v.push_back(unwrap(tensors[i]));
        return wrap(torch::stack(v, dim));
    } catch (...) { return nullptr; }
}

// ============================================================
// 矩阵与线性层
// ============================================================
void* torch_std_matmul(void* a, void* b) { try { return wrap(unwrap(a).matmul(unwrap(b))); } catch (...) { return nullptr; } }

void* torch_std_linear(void* input, void* weight, void* bias) {
    try {
        auto& x = unwrap(input);
        auto& w = unwrap(weight);
        auto out = torch::matmul(x, w.t());
        if (bias) out = out + unwrap(bias);
        return wrap(out);
    } catch (...) { return nullptr; }
}

void* torch_std_attention(void* q, void* k, void* v, int64_t heads, void* mask, int skip_reshape) {
    try {
        at::Tensor q_t, k_t, v_t;
        int64_t B, N, M, dim_head;

        if (skip_reshape) {
            // Input format: (B, heads, seq, dim_head) — pre-reshaped by FLUX
            q_t = unwrap(q);
            k_t = unwrap(k);
            v_t = unwrap(v);
            B = q_t.size(0);
            heads = q_t.size(1);
            N = q_t.size(2);
            M = k_t.size(2);
            dim_head = q_t.size(3);

            // Flatten heads into batch: (B, heads, N, dim_head) -> (B*heads, N, dim_head)
            q_t = q_t.reshape({B * heads, N, dim_head});
            k_t = k_t.reshape({B * heads, M, dim_head});
            v_t = v_t.reshape({B * heads, M, dim_head});
        } else {
            // Input format: (B, N, D) — standard
            q_t = unwrap(q);
            k_t = unwrap(k);
            v_t = unwrap(v);
            B = q_t.size(0);
            N = q_t.size(1);
            M = k_t.size(1);
            auto D = q_t.size(2);
            dim_head = D / heads;

            // Reshape to multi-head format
            q_t = q_t.view({B, N, heads, dim_head})
                .permute({0, 2, 1, 3})
                .reshape({B * heads, N, dim_head})
                .contiguous();
            k_t = k_t.view({B, M, heads, dim_head})
                .permute({0, 2, 1, 3})
                .reshape({B * heads, M, dim_head})
                .contiguous();
            v_t = v_t.view({B, M, heads, dim_head})
                .permute({0, 2, 1, 3})
                .reshape({B * heads, M, dim_head})
                .contiguous();
        }

        // Scaled dot-product: scores = Q @ K^T / sqrt(dim_head)
        auto scale = 1.0 / std::sqrt((double)dim_head);
        auto scores = torch::matmul(q_t, k_t.transpose(1, 2)) * scale;

        // Apply mask if provided
        if (mask) {
            auto& mask_t = unwrap(mask);
            scores = scores + mask_t;
        }

        // Softmax over last dim
        auto attn = torch::softmax(scores, -1);

        // out = attn @ V
        auto out = torch::matmul(attn, v_t);

        // Reshape back
        if (skip_reshape) {
            // (B*heads, N, dim_head) -> (B, heads, N, dim_head)
            out = out.view({B, heads, N, dim_head});
        } else {
            // (B*heads, N, dim_head) -> (B, heads, N, dim_head) -> (B, N, D)
            out = out.view({B, heads, N, dim_head})
                .permute({0, 2, 1, 3})
                .reshape({B, N, heads * dim_head})
                .contiguous();
        }

        return wrap(out);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_attention error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_conv1d(void* input, void* weight, void* bias,
                       int64_t stride, int64_t padding, int64_t dilation, int64_t groups) {
    try {
        std::optional<at::Tensor> b = bias ? std::optional<at::Tensor>(unwrap(bias)) : std::nullopt;
        auto out = torch::conv1d(unwrap(input), unwrap(weight), b,
                                 {stride}, {padding}, {dilation}, groups);
        return wrap(out);
    } catch (...) { return nullptr; }
}

void* torch_std_conv2d(void* input, void* weight, void* bias,
                       int64_t stride_h, int64_t stride_w,
                       int64_t padding_h, int64_t padding_w,
                       int64_t dilation_h, int64_t dilation_w,
                       int64_t groups) {
    try {
        std::optional<at::Tensor> b = bias ? std::optional<at::Tensor>(unwrap(bias)) : std::nullopt;
        auto out = torch::conv2d(unwrap(input), unwrap(weight), b,
                                 {stride_h, stride_w}, {padding_h, padding_w},
                                 {dilation_h, dilation_w}, groups);
        return wrap(out);
    } catch (...) { return nullptr; }
}

void* torch_std_max_pool2d(void* input, int64_t kernel_h, int64_t kernel_w,
                           int64_t stride_h, int64_t stride_w,
                           int64_t padding_h, int64_t padding_w,
                           int64_t dilation_h, int64_t dilation_w) {
    try {
        auto out = torch::max_pool2d(unwrap(input), {kernel_h, kernel_w},
                                     {stride_h, stride_w}, {padding_h, padding_w},
                                     {dilation_h, dilation_w});
        return wrap(out);
    } catch (...) { return nullptr; }
}

void* torch_std_avg_pool2d(void* input, int64_t kernel_h, int64_t kernel_w,
                           int64_t stride_h, int64_t stride_w,
                           int64_t padding_h, int64_t padding_w) {
    try {
        auto out = torch::avg_pool2d(unwrap(input), {kernel_h, kernel_w},
                                     {stride_h, stride_w}, {padding_h, padding_w});
        return wrap(out);
    } catch (...) { return nullptr; }
}

void* torch_std_batch_norm1d(void* input, void* weight, void* bias,
                             void* running_mean, void* running_var,
                             int training, double momentum, double eps) {
    try {
        std::optional<at::Tensor> w = weight ? std::optional<at::Tensor>(unwrap(weight)) : std::nullopt;
        std::optional<at::Tensor> b = bias ? std::optional<at::Tensor>(unwrap(bias)) : std::nullopt;
        std::optional<at::Tensor> rm = running_mean ? std::optional<at::Tensor>(unwrap(running_mean)) : std::nullopt;
        std::optional<at::Tensor> rv = running_var ? std::optional<at::Tensor>(unwrap(running_var)) : std::nullopt;
        auto out = torch::batch_norm(unwrap(input), w, b, rm, rv,
                                     training != 0, momentum, eps, true);
        return wrap(out);
    } catch (...) { return nullptr; }
}

void* torch_std_batch_norm2d(void* input, void* weight, void* bias,
                             void* running_mean, void* running_var,
                             int training, double momentum, double eps) {
    try {
        std::optional<at::Tensor> w = weight ? std::optional<at::Tensor>(unwrap(weight)) : std::nullopt;
        std::optional<at::Tensor> b = bias ? std::optional<at::Tensor>(unwrap(bias)) : std::nullopt;
        std::optional<at::Tensor> rm = running_mean ? std::optional<at::Tensor>(unwrap(running_mean)) : std::nullopt;
        std::optional<at::Tensor> rv = running_var ? std::optional<at::Tensor>(unwrap(running_var)) : std::nullopt;
        auto out = torch::batch_norm(unwrap(input), w, b, rm, rv,
                                     training != 0, momentum, eps, true);
        return wrap(out);
    } catch (...) { return nullptr; }
}

// ============================================================
// 损失函数
// ============================================================
void* torch_std_mse_loss(void* pred, void* target, const char* reduction) {
    try { return wrap(torch::mse_loss(unwrap(pred), unwrap(target), to_reduction(reduction))); } catch (...) { return nullptr; }
}

void* torch_std_l1_loss(void* pred, void* target, const char* reduction) {
    try { return wrap(torch::l1_loss(unwrap(pred), unwrap(target), to_reduction(reduction))); } catch (...) { return nullptr; }
}

void* torch_std_cross_entropy_loss(void* logits, void* target, const char* reduction) {
    try { return wrap(torch::cross_entropy_loss(unwrap(logits), unwrap(target), {}, to_reduction(reduction))); } catch (...) { return nullptr; }
}

void* torch_std_nll_loss(void* log_probs, void* target, const char* reduction) {
    try { return wrap(torch::nll_loss(unwrap(log_probs), unwrap(target), {}, to_reduction(reduction))); } catch (...) { return nullptr; }
}

void* torch_std_bce_loss(void* pred, void* target, const char* reduction) {
    try { return wrap(torch::binary_cross_entropy(unwrap(pred), unwrap(target), {}, to_reduction(reduction))); } catch (...) { return nullptr; }
}

void* torch_std_bce_with_logits_loss(void* logits, void* target, const char* reduction) {
    try { return wrap(torch::binary_cross_entropy_with_logits(unwrap(logits), unwrap(target), {}, {}, to_reduction(reduction))); } catch (...) { return nullptr; }
}

// ============================================================
// 自动微分
// ============================================================
void* torch_std_requires_grad(void* t) {
    try {
        return wrap(unwrap(t).detach().clone().set_requires_grad(true));
    } catch (...) { return nullptr; }
}

void* torch_std_set_requires_grad(void* t, int requires_grad) {
    try {
        unwrap(t).set_requires_grad(requires_grad != 0);
        return t;
    } catch (...) { return nullptr; }
}

void torch_std_backward(void* loss) {
    try {
        unwrap(loss).backward();
    } catch (const std::exception& e) {
        std::cerr << "torch_std_backward error: " << e.what() << std::endl;
    }
}

void torch_std_backward_retain_graph(void* loss) {
    try {
        unwrap(loss).backward({}, true);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_backward_retain_graph error: " << e.what() << std::endl;
    }
}

void* torch_std_grad(void* t) {
    try {
        auto& tensor = unwrap(t);
        if (!tensor.grad().defined()) return nullptr;
        return wrap(tensor.grad().clone());
    } catch (...) { return nullptr; }
}

void torch_std_zero_grad(void* t) {
    try {
        auto& tensor = unwrap(t);
        if (tensor.grad().defined()) {
            tensor.grad().detach().zero_();
        }
    } catch (...) {}
}

int torch_std_has_grad(void* t) {
    try {
        return unwrap(t).grad().defined() ? 1 : 0;
    } catch (...) { return 0; }
}

// ============================================================
// 优化器
// ============================================================
void* torch_std_sgd_create(void** params, int n, double lr, double momentum,
                           double dampening, double weight_decay, int nesterov) {
    try {
        std::vector<at::Tensor> p;
        for (int i = 0; i < n; i++) p.push_back(unwrap(params[i]));
        auto opt = new torch::optim::SGD(p, torch::optim::SGDOptions(lr)
            .momentum(momentum)
            .dampening(dampening)
            .weight_decay(weight_decay)
            .nesterov(nesterov != 0));
        return opt;
    } catch (...) { return nullptr; }
}

void* torch_std_adam_create(void** params, int n, double lr,
                            double beta1, double beta2, double eps,
                            double weight_decay, int amsgrad) {
    try {
        std::vector<at::Tensor> p;
        for (int i = 0; i < n; i++) p.push_back(unwrap(params[i]));
        auto opt = new torch::optim::Adam(p, torch::optim::AdamOptions(lr)
            .betas({beta1, beta2})
            .eps(eps)
            .weight_decay(weight_decay)
            .amsgrad(amsgrad != 0));
        return opt;
    } catch (...) { return nullptr; }
}

void* torch_std_adamw_create(void** params, int n, double lr,
                             double beta1, double beta2, double eps,
                             double weight_decay, int amsgrad) {
    try {
        std::vector<at::Tensor> p;
        for (int i = 0; i < n; i++) p.push_back(unwrap(params[i]));
        auto opt = new torch::optim::AdamW(p, torch::optim::AdamWOptions(lr)
            .betas({beta1, beta2})
            .eps(eps)
            .weight_decay(weight_decay)
            .amsgrad(amsgrad != 0));
        return opt;
    } catch (...) { return nullptr; }
}

void torch_std_optimizer_step(void* opt) {
    try {
        static_cast<torch::optim::Optimizer*>(opt)->step();
    } catch (const std::exception& e) {
        std::cerr << "torch_std_optimizer_step error: " << e.what() << std::endl;
    }
}

void torch_std_optimizer_zero_grad(void* opt) {
    try {
        static_cast<torch::optim::Optimizer*>(opt)->zero_grad();
    } catch (const std::exception& e) {
        std::cerr << "torch_std_optimizer_zero_grad error: " << e.what() << std::endl;
    }
}

void torch_std_optimizer_destroy(void* opt) {
    delete static_cast<torch::optim::Optimizer*>(opt);
}

// ============================================================
// 工具
// ============================================================
void* torch_std_narrow(void* a, int64_t dim, int64_t start, int64_t length) {
    try { return wrap(unwrap(a).narrow(dim, start, length)); } catch (...) { return nullptr; }
}

void* torch_std_slice(void* a, int64_t dim, int64_t start, int64_t end, int64_t step) {
    try { return wrap(unwrap(a).slice(dim, start, end, step)); } catch (...) { return nullptr; }
}

int64_t torch_std_size(void* a, int64_t dim) {
    try { return unwrap(a).size(dim); } catch (...) { return -1; }
}

void* torch_std_masked_select(void* a, void* mask) {
    try { return wrap(unwrap(a).masked_select(unwrap(mask))); } catch (...) { return nullptr; }
}

void* torch_std_where(void* condition, void* x, void* y) {
    try { return wrap(torch::where(unwrap(condition), unwrap(x), unwrap(y))); } catch (...) { return nullptr; }
}

void* torch_std_eq(void* a, void* b) { try { return wrap(unwrap(a) == unwrap(b)); } catch (...) { return nullptr; } }
void* torch_std_gt(void* a, void* b) { try { return wrap(unwrap(a) > unwrap(b)); } catch (...) { return nullptr; } }
void* torch_std_lt(void* a, void* b) { try { return wrap(unwrap(a) < unwrap(b)); } catch (...) { return nullptr; } }
void* torch_std_ge(void* a, void* b) { try { return wrap(unwrap(a) >= unwrap(b)); } catch (...) { return nullptr; } }
void* torch_std_le(void* a, void* b) { try { return wrap(unwrap(a) <= unwrap(b)); } catch (...) { return nullptr; } }
void* torch_std_clamp(void* a, double min_val, double max_val) { try { return wrap(unwrap(a).clamp(min_val, max_val)); } catch (...) { return nullptr; } }

void torch_std_clip_grad_norm(void** params, int n, double max_norm) {
    try {
        std::vector<at::Tensor> p;
        for (int i = 0; i < n; i++) p.push_back(unwrap(params[i]));
        torch::nn::utils::clip_grad_norm_(p, max_norm);
    } catch (...) {}
}

void torch_std_manual_seed(int64_t seed) {
    try { torch::manual_seed(seed); torch::cuda::manual_seed(seed); } catch (...) {}
}

int torch_std_cuda_is_available(void) {
    try { return torch::cuda::is_available() ? 1 : 0; } catch (...) { return 0; }
}

extern "C" void torch_std_cuda_synchronize(void) {
    try { torch::cuda::synchronize(); } catch (...) {}
}

void* torch_std_to_cuda(void* t) {
    try { return wrap(unwrap(t).to(torch::kCUDA)); } catch (...) { return nullptr; }
}

void* torch_std_to_cpu(void* t) {
    try { return wrap(unwrap(t).to(torch::kCPU)); } catch (...) { return nullptr; }
}

int torch_std_is_cuda(void* t) {
    try { return unwrap(t).device().is_cuda() ? 1 : 0; } catch (...) { return 0; }
}

// ============================================================
// CUDA 显存管理
// ============================================================
int64_t torch_std_cuda_get_free_memory(void) {
    try {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        return static_cast<int64_t>(free);
    } catch (...) { return 0; }
}

void* torch_std_cuda_load_model(int device, void* tensor) {
    try {
        auto t = unwrap(tensor).to(torch::Device(torch::kCUDA, device));
        return wrap(t);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_cuda_load_model error: " << e.what() << std::endl;
        return nullptr;
    }
}

void torch_std_cuda_unload_model(void* tensor) {
    try {
        delete static_cast<torch::Tensor*>(tensor);
    } catch (...) {}
}

void torch_std_cuda_soft_empty_cache(void) {
    try {
        c10::cuda::CUDACachingAllocator::emptyCache();
    } catch (...) {}
}

// ============================================================
// TorchScript JIT 模型加载与推理
// ============================================================
void* torch_std_jit_load(const char* path) {
    try {
        install_quiet_rnn_warning();
        torch::Device device = torch::kCUDA;
        auto module = new torch::jit::Module(torch::jit::load(path, device));
        return module;
    } catch (const std::exception& e) {
        std::cerr << "torch_std_jit_load error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_jit_forward(void* module, void* input_tensor) {
    try {
        auto* mod = static_cast<torch::jit::Module*>(module);
        auto input = unwrap(input_tensor).to(torch::kCUDA);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        auto output = mod->forward(inputs).toTensor();
        return wrap(output);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_jit_forward error: " << e.what() << std::endl;
        return nullptr;
    }
}

void torch_std_jit_module_delete(void* module) {
    delete static_cast<torch::jit::Module*>(module);
}

int torch_std_jit_parameters(void* module, void** out_params, char** out_names, int max_n) {
    try {
        auto* mod = static_cast<torch::jit::Module*>(module);
        int count = 0;
        for (const auto& p : mod->parameters()) {
            if (count >= max_n) break;
            if (out_params) out_params[count] = new torch::Tensor(p);
            if (out_names) {
                std::string name = "param_" + std::to_string(count);
                out_names[count] = strdup(name.c_str());
            }
            count++;
        }
        return count;
    } catch (const std::exception& e) {
        std::cerr << "torch_std_jit_parameters error: " << e.what() << std::endl;
        return -1;
    }
}

// ============================================================
// 模型参数序列化（支持 StaticPy 续训）
// ============================================================
int torch_std_save_state_dict(void* module, const char* path) {
    try {
        auto* mod = static_cast<torch::jit::Module*>(module);
        mod->save(path);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "torch_std_save_state_dict error: " << e.what() << std::endl;
        return -1;
    }
}

void* torch_std_jit_load_module(const char* path) {
    return torch_std_jit_load(path);
}

// ============================================================
// SD UNet 权重布局 — 用于 torch_std_sd_unet_forward
// ============================================================
// 权重编号约定（n_weights=186，适用于 SD 1.5 UNet）：
//   0-1:    time_embed.0.weight/.bias         [320,320]
//   2-3:    time_embed.2.weight/.bias         [320,320]
//   4-5:    input_blocks.0.0.weight/.bias     [320,4,3,3]
//   6-7:    input_blocks.1.0.in_layers.0.weight/.bias   [320,320,3,3]
//   8-9:    input_blocks.1.0.in_layers.2.weight/.bias   [320,320,3,3]
//   10-11:  input_blocks.1.0.skip_connection.weight/.bias  [320,320,1,1]
//   12-13:  input_blocks.1.0.emb_layers.1.weight/.bias    [320,320]
//   ...
// 完整映射见 libtorch_std_helper.md

// ============================================================
// SD 1.5 UNet forward with LoRA (autograd-aware)
// ============================================================
// 实现 SD 1.5 完整 UNet 架构，支持 LoRA 权重注入和梯度回传。
// 所有操作使用 torch:: 原生 API，自动构建 autograd 图。

// ---------- 辅助函数 ----------

static at::Tensor timestep_embedding(const at::Tensor& timesteps, int dim, int max_period=10000) {
    // Sin/cos timestep embedding
    int half = dim / 2;
    auto log_period = at::log(at::full(1, static_cast<double>(max_period), 
                               at::TensorOptions().dtype(torch::kFloat32)));
    auto freqs = at::exp(-log_period * 
                          at::arange(0, half, at::TensorOptions().dtype(torch::kFloat32)) / half);
    freqs = freqs.to(timesteps.device());
    // timesteps: (B,) → (B, half)
    auto emb = timesteps.unsqueeze(1) * freqs.unsqueeze(0);
    std::vector<at::Tensor> cats = {at::cos(emb), at::sin(emb)};
    return at::cat(cats, 1);
}

// GroupNorm32 (Silu → conv) is the SD convention
static at::Tensor groupnorm_32(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b, int num_channels) {
    return at::group_norm(x, 32, w, b, 1e-6);
}

// ResBlock forward with timestep embedding injection
struct ResBlockWeights {
    at::Tensor norm1_w, norm1_b;    // groupnorm
    at::Tensor conv1_w, conv1_b;    // conv3x3
    at::Tensor norm2_w, norm2_b;    // groupnorm
    at::Tensor conv2_w, conv2_b;    // conv3x3
    at::Tensor skip_w, skip_b;      // skip conv (may be empty)
    at::Tensor time_mlp_w, time_mlp_b; // time embedding MLP layer
};

static at::Tensor resblock_forward(
    const at::Tensor& x, const at::Tensor& temb,
    const ResBlockWeights& w) {
    auto h = groupnorm_32(x, w.norm1_w, w.norm1_b, x.size(1));
    h = at::silu(h);
    h = at::conv2d(h, w.conv1_w, w.conv1_b, at::IntArrayRef{1,1}, at::IntArrayRef{1,1});

    // Time embedding injection
    auto temb_linear = at::linear(at::silu(temb), w.time_mlp_w, w.time_mlp_b);
    // Reshape to (B, C, 1, 1) for broadcasting
    auto temb_reshaped = temb_linear.view({temb_linear.size(0), temb_linear.size(1), 1, 1});
    h = h + temb_reshaped;

    h = groupnorm_32(h, w.norm2_w, w.norm2_b, h.size(1));
    h = at::silu(h);
    h = at::conv2d(h, w.conv2_w, w.conv2_b, at::IntArrayRef{1,1}, at::IntArrayRef{1,1});

    // Skip connection
    if (w.skip_w.defined()) {
        auto skip = at::conv2d(x, w.skip_w, w.skip_b, at::IntArrayRef{1,1});
        h = h + skip;
    } else {
        h = h + x;
    }
    return h;
}

// Transformer (self-attn + cross-attn + FFN)
struct TransformerWeights {
    at::Tensor norm_w, norm_b;      // groupnorm
    // Self-attention
    at::Tensor self_q_w, self_q_b, self_k_w, self_k_b, self_v_w, self_v_b, self_out_w, self_out_b;
    // Cross-attention
    at::Tensor cross_q_w, cross_q_b, cross_k_w, cross_k_b, cross_v_w, cross_v_b, cross_out_w, cross_out_b;
    // FFN
    at::Tensor ff1_w, ff1_b, ff2_w, ff2_b;
};

static at::Tensor transformer_forward(
    const at::Tensor& x, const at::Tensor& text_emb,
    const TransformerWeights& w, int n_heads=8) {
    int ch = x.size(1);
    int head_dim = ch / n_heads;

    // Pre-norm
    auto h = groupnorm_32(x, w.norm_w, w.norm_b, ch);
    
    // Self-attention
    auto self_q = at::linear(h, w.self_q_w, w.self_q_b);
    auto self_k = at::linear(h, w.self_k_w, w.self_k_b);
    auto self_v = at::linear(h, w.self_v_w, w.self_v_b);
    
    // Reshape to (B*H, N, head_dim)
    auto B = h.size(0), N = h.size(2) * h.size(3);
    self_q = self_q.view({B, N, n_heads, head_dim}).permute({0, 2, 1, 3}).reshape({B*n_heads, N, head_dim});
    self_k = self_k.view({B, N, n_heads, head_dim}).permute({0, 2, 1, 3}).reshape({B*n_heads, N, head_dim});
    self_v = self_v.view({B, N, n_heads, head_dim}).permute({0, 2, 1, 3}).reshape({B*n_heads, N, head_dim});
    
    auto self_attn = at::softmax(at::bmm(self_q, self_k.transpose(1, 2)) / std::sqrt(double(head_dim)), -1);
    auto self_out = at::bmm(self_attn, self_v);
    self_out = self_out.view({B, n_heads, N, head_dim}).permute({0, 2, 1, 3}).reshape({B, N, ch});
    self_out = at::linear(self_out, w.self_out_w, w.self_out_b);
    h = h + self_out;  // residual

    // Cross-attention
    auto cross_q = at::linear(h, w.cross_q_w, w.cross_q_b);
    auto cross_k = at::linear(text_emb, w.cross_k_w, w.cross_k_b);
    auto cross_v = at::linear(text_emb, w.cross_v_w, w.cross_v_b);
    
    cross_q = cross_q.view({B, N, n_heads, head_dim}).permute({0, 2, 1, 3}).reshape({B*n_heads, N, head_dim});
    cross_k = cross_k.view({B, -1, n_heads, head_dim}).permute({0, 2, 1, 3}).reshape({B*n_heads, -1, head_dim});
    cross_v = cross_v.view({B, -1, n_heads, head_dim}).permute({0, 2, 1, 3}).reshape({B*n_heads, -1, head_dim});
    
    auto cross_attn = at::softmax(at::bmm(cross_q, cross_k.transpose(1, 2)) / std::sqrt(double(head_dim)), -1);
    auto cross_out = at::bmm(cross_attn, cross_v);
    cross_out = cross_out.view({B, n_heads, N, head_dim}).permute({0, 2, 1, 3}).reshape({B, N, ch});
    cross_out = at::linear(cross_out, w.cross_out_w, w.cross_out_b);
    h = h + cross_out;  // residual

    // FFN (GeGLU style)
    auto ff_h = at::linear(h, w.ff1_w, w.ff1_b);
    auto gate = at::silu(ff_h.slice(/*dim=*/2, /*start=*/0, /*end=*/ch*4));
    ff_h = ff_h * gate;
    h = h + at::linear(ff_h, w.ff2_w, w.ff2_b);

    return h;
}

// ============================================================
// Main UNet forward (public C API)
// ============================================================
// Takes a flat array of weight tensors (organized as per SDUNetWeightIndex)
// plus LoRA tensors, input latent, timesteps, text embedding.
// Returns output latent tensor. Autograd graph is built.

// ---- GGUF file helpers (used inside extern C below) ----
template<typename T>
static bool gguf_read(FILE* f, T* val) {
    return fread(val, sizeof(T), 1, f) == 1;
}

extern "C" {

void* torch_std_sd_unet_forward(
    void** weight_ptrs,     // flat array of UNet weight tensors
    int n_weights,          // 186 for SD 1.5 UNet
    void* input_ptr,        // (B, 4, H, W) latent
    void* timestep_ptr,     // (B,) float timesteps
    void* text_emb_ptr,     // (B, 77, 768) text embeddings
    void** lora_A_ptrs,     // LoRA A matrices per target (or NULL)
    void** lora_B_ptrs,     // LoRA B matrices per target (or NULL)
    int* lora_target_indices, // which weight index each LoRA modifies
    int n_lora,             // number of LoRA targets
    double lora_scale       // LoRA scaling factor
) {
    try {
        // Convert to tensors
        // We'll cast weight_ptrs[i] to at::Tensor* and access
        auto w = [&](int i) -> at::Tensor& {
            return *static_cast<at::Tensor*>(weight_ptrs[i]);
        };
        
        // Apply LoRA: for each target, create modified weights
        // W_eff = W + scale * B @ A (builds autograd graph to A, B if they require grad)
        std::vector<at::Tensor> modified_w;
        std::vector<int> modified_indices;
        
        for (int i = 0; i < n_lora; i++) {
            int idx = lora_target_indices[i];
            auto& W = w(idx);
            auto BA = at::matmul(*static_cast<at::Tensor*>(lora_B_ptrs[i]),
                                 *static_cast<at::Tensor*>(lora_A_ptrs[i])) * lora_scale;
            auto W_eff = W + BA;
            modified_w.push_back(W_eff);
            modified_indices.push_back(idx);
        }
        
        // Helper: get weight with LoRA applied
        auto w_lora = [&](int i) -> at::Tensor {
            for (int j = 0; j < (int)modified_indices.size(); j++) {
                if (modified_indices[j] == i) return modified_w[j];
            }
            return w(i);
        };
        
        auto x = unwrap(input_ptr);
        auto t = unwrap(timestep_ptr);
        auto text = unwrap(text_emb_ptr);
        
        // ---- Time embedding ----
        auto temb = timestep_embedding(t, 320);
        temb = at::linear(temb, w_lora(0), w_lora(1));   // time_embed.0
        temb = at::silu(temb);
        temb = at::linear(temb, w_lora(2), w_lora(3));   // time_embed.2
        
        // ---- Store skip connections ----
        std::vector<at::Tensor> skips;
        
        // Input conv
        auto h = at::conv2d(x, w_lora(4), w_lora(5), at::IntArrayRef{1,1}, at::IntArrayRef{1,1});  // input_blocks.0.0
        
        // ---- Down blocks ----
        // Stage 1 (ch=320): 2 ResBlocks
        // input_blocks.1.0, input_blocks.1.1
        skips.push_back(h);
        // input_blocks.1.0
        {
            ResBlockWeights rbw = {
                w_lora(6), w_lora(7), w_lora(8), w_lora(9),
                w_lora(10), w_lora(11), w_lora(12), w_lora(13),
                w_lora(14), w_lora(15),  // skip weight/bias (might be dummy)
                w_lora(16), w_lora(17)   // time_mlp
            };
            if (!rbw.skip_w.defined() || rbw.skip_w.numel() == 0) {
                rbw.skip_w = at::Tensor();  // mark undefined
            }
            h = resblock_forward(h, temb, rbw);
        }
        skips.push_back(h);
        // input_blocks.1.1
        {
            ResBlockWeights rbw = {
                w_lora(18), w_lora(19), w_lora(20), w_lora(21),
                w_lora(22), w_lora(23), w_lora(24), w_lora(25),
                at::Tensor(), at::Tensor(),  // no skip needed
                w_lora(26), w_lora(27)
            };
            h = resblock_forward(h, temb, rbw);
        }
        skips.push_back(h);
        
        // Downsample + Stage 2 (ch=320→640): 2 ResBlocks + Transformer
        // input_blocks.2.0 = Downsample
        h = at::conv2d(h, w_lora(28), w_lora(29), at::IntArrayRef{2,2}, at::IntArrayRef{1,1});
        skips.push_back(h);
        // input_blocks.2.1 → ResBlock 320→640
        {
            ResBlockWeights rbw = {
                w_lora(30), w_lora(31), w_lora(32), w_lora(33),
                w_lora(34), w_lora(35), w_lora(36), w_lora(37),
                w_lora(38), w_lora(39),  // skip: 320→640
                w_lora(40), w_lora(41)
            };
            h = resblock_forward(h, temb, rbw);
        }
        skips.push_back(h);
        // input_blocks.2.2 → ResBlock 640→640
        {
            ResBlockWeights rbw = {
                w_lora(42), w_lora(43), w_lora(44), w_lora(45),
                w_lora(46), w_lora(47), w_lora(48), w_lora(49),
                at::Tensor(), at::Tensor(),
                w_lora(50), w_lora(51)
            };
            h = resblock_forward(h, temb, rbw);
        }
        skips.push_back(h);
        // input_blocks.2.3 → Transformer (ch=640)
        {
            TransformerWeights tfw = {
                w_lora(52), w_lora(53),
                w_lora(54), w_lora(55), w_lora(56), w_lora(57),
                w_lora(58), w_lora(59), w_lora(60), w_lora(61),
                w_lora(62), w_lora(63), w_lora(64), w_lora(65),
                w_lora(66), w_lora(67), w_lora(68), w_lora(69),
                w_lora(70), w_lora(71), w_lora(72), w_lora(73)
            };
            h = transformer_forward(h, text, tfw, 8);
        }
        skips.push_back(h);
        
        // Downsample + Stage 3 (ch=640→1280): 2 ResBlocks + Transformer
        h = at::conv2d(h, w_lora(78), w_lora(79), at::IntArrayRef{2,2}, at::IntArrayRef{1,1});
        skips.push_back(h);
        // input_blocks.3.1 → ResBlock 640→1280
        {
            ResBlockWeights rbw = {
                w_lora(80), w_lora(81), w_lora(82), w_lora(83),
                w_lora(84), w_lora(85), w_lora(86), w_lora(87),
                w_lora(88), w_lora(89),  // skip: 640→1280
                w_lora(90), w_lora(91)
            };
            h = resblock_forward(h, temb, rbw);
        }
        skips.push_back(h);
        // input_blocks.3.2 → ResBlock 1280→1280
        {
            ResBlockWeights rbw = {
                w_lora(92), w_lora(93), w_lora(94), w_lora(95),
                w_lora(96), w_lora(97), w_lora(98), w_lora(99),
                at::Tensor(), at::Tensor(),
                w_lora(100), w_lora(101)
            };
            h = resblock_forward(h, temb, rbw);
        }
        skips.push_back(h);
        // input_blocks.3.3 → Transformer (ch=1280)
        {
            TransformerWeights tfw = {
                w_lora(102), w_lora(103),
                w_lora(104), w_lora(105), w_lora(106), w_lora(107),
                w_lora(108), w_lora(109), w_lora(110), w_lora(111),
                w_lora(112), w_lora(113), w_lora(114), w_lora(115),
                w_lora(116), w_lora(117), w_lora(118), w_lora(119),
                w_lora(120), w_lora(121), w_lora(122), w_lora(123)
            };
            h = transformer_forward(h, text, tfw, 8);
        }
        skips.push_back(h);
        
        // Downsample + Stage 4 (ch=1280→1280): 2 ResBlocks + Transformer
        h = at::conv2d(h, w_lora(128), w_lora(129), at::IntArrayRef{2,2}, at::IntArrayRef{1,1});
        skips.push_back(h);
        // input_blocks.4.1 → ResBlock 1280→1280
        {
            ResBlockWeights rbw = {
                w_lora(130), w_lora(131), w_lora(132), w_lora(133),
                w_lora(134), w_lora(135), w_lora(136), w_lora(137),
                at::Tensor(), at::Tensor(),
                w_lora(138), w_lora(139)
            };
            h = resblock_forward(h, temb, rbw);
        }
        skips.push_back(h);
        // input_blocks.4.2 → ResBlock 1280→1280
        {
            ResBlockWeights rbw = {
                w_lora(140), w_lora(141), w_lora(142), w_lora(143),
                w_lora(144), w_lora(145), w_lora(146), w_lora(147),
                at::Tensor(), at::Tensor(),
                w_lora(148), w_lora(149)
            };
            h = resblock_forward(h, temb, rbw);
        }
        skips.push_back(h);
        // input_blocks.4.3 → Transformer (ch=1280)
        {
            TransformerWeights tfw = {
                w_lora(150), w_lora(151),
                w_lora(152), w_lora(153), w_lora(154), w_lora(155),
                w_lora(156), w_lora(157), w_lora(158), w_lora(159),
                w_lora(160), w_lora(161), w_lora(162), w_lora(163),
                w_lora(164), w_lora(165), w_lora(166), w_lora(167),
                w_lora(168), w_lora(169), w_lora(170), w_lora(171)
            };
            h = transformer_forward(h, text, tfw, 8);
        }
        skips.push_back(h);
        
        // ---- Middle block ----
        // middle_block.0 → ResBlock
        {
            ResBlockWeights rbw = {
                w_lora(176), w_lora(177), w_lora(178), w_lora(179),
                w_lora(180), w_lora(181), w_lora(182), w_lora(183),
                at::Tensor(), at::Tensor(),
                w_lora(184), w_lora(185)
            };
            h = resblock_forward(h, temb, rbw);
        }
        // middle_block.1 → Transformer
        {
            TransformerWeights tfw = {
                w_lora(186), w_lora(187),
                w_lora(188), w_lora(189), w_lora(190), w_lora(191),
                w_lora(192), w_lora(193), w_lora(194), w_lora(195),
                w_lora(196), w_lora(197), w_lora(198), w_lora(199),
                w_lora(200), w_lora(201), w_lora(202), w_lora(203),
                w_lora(204), w_lora(205), w_lora(206), w_lora(207)
            };
            h = transformer_forward(h, text, tfw, 8);
        }
        // middle_block.2 → ResBlock
        {
            ResBlockWeights rbw = {
                w_lora(212), w_lora(213), w_lora(214), w_lora(215),
                w_lora(216), w_lora(217), w_lora(218), w_lora(219),
                at::Tensor(), at::Tensor(),
                w_lora(220), w_lora(221)
            };
            h = resblock_forward(h, temb, rbw);
        }
        skips.push_back(h);  // middle output for skip
        
        // ---- Up blocks (reverse order) ----
        // Helper to pop last skip
        auto pop_skip = [&]() -> at::Tensor {
            auto s = skips.back();
            skips.pop_back();
            return s;
        };
        
        // Up stage 4 (skip from stage 4 down + middle)
        // output_blocks.0.0
        {
            auto skip = pop_skip();  // middle output
            h = at::cat({h, skip}, 1);
            ResBlockWeights rbw = {
                w_lora(222), w_lora(223), w_lora(224), w_lora(225),
                w_lora(226), w_lora(227), w_lora(228), w_lora(229),
                w_lora(230), w_lora(231),
                w_lora(232), w_lora(233)
            };
            h = resblock_forward(h, temb, rbw);
        }
        // output_blocks.0.1
        {
            auto skip = pop_skip();  // last down skip
            h = at::cat({h, skip}, 1);
            ResBlockWeights rbw = {
                w_lora(234), w_lora(235), w_lora(236), w_lora(237),
                w_lora(238), w_lora(239), w_lora(240), w_lora(241),
                w_lora(242), w_lora(243),
                w_lora(244), w_lora(245)
            };
            h = resblock_forward(h, temb, rbw);
        }
        // output_blocks.0.2 → Transformer (ch=2560→1280 skip+cat)
        {
            TransformerWeights tfw = {
                w_lora(246), w_lora(247),
                w_lora(248), w_lora(249), w_lora(250), w_lora(251),
                w_lora(252), w_lora(253), w_lora(254), w_lora(255),
                w_lora(256), w_lora(257), w_lora(258), w_lora(259),
                w_lora(260), w_lora(261), w_lora(262), w_lora(263),
                w_lora(264), w_lora(265), w_lora(266), w_lora(267)
            };
            h = transformer_forward(h, text, tfw, 8);
        }
        // Upsample: nearest 2x + conv
        h = at::upsample_nearest2d(h, {h.size(2)*2, h.size(3)*2});
        { at::IntArrayRef stride = {1,1}; at::IntArrayRef pad = {0,0};
        h = at::conv2d(h, w_lora(272), w_lora(273), stride, pad); }
        
        // Up stage 3 (skip from stage 3 down)
        // output_blocks.1.0
        {
            auto skip = pop_skip();
            h = at::cat({h, skip}, 1);
            ResBlockWeights rbw = {
                w_lora(274), w_lora(275), w_lora(276), w_lora(277),
                w_lora(278), w_lora(279), w_lora(280), w_lora(281),
                w_lora(282), w_lora(283),
                w_lora(284), w_lora(285)
            };
            h = resblock_forward(h, temb, rbw);
        }
        // output_blocks.1.1
        {
            auto skip = pop_skip();
            h = at::cat({h, skip}, 1);
            ResBlockWeights rbw = {
                w_lora(286), w_lora(287), w_lora(288), w_lora(289),
                w_lora(290), w_lora(291), w_lora(292), w_lora(293),
                w_lora(294), w_lora(295),
                w_lora(296), w_lora(297)
            };
            h = resblock_forward(h, temb, rbw);
        }
        // output_blocks.1.2 → Transformer
        {
            TransformerWeights tfw = {
                w_lora(298), w_lora(299),
                w_lora(300), w_lora(301), w_lora(302), w_lora(303),
                w_lora(304), w_lora(305), w_lora(306), w_lora(307),
                w_lora(308), w_lora(309), w_lora(310), w_lora(311),
                w_lora(312), w_lora(313), w_lora(314), w_lora(315),
                w_lora(316), w_lora(317), w_lora(318), w_lora(319)
            };
            h = transformer_forward(h, text, tfw, 8);
        }
        h = at::upsample_nearest2d(h, {h.size(2)*2, h.size(3)*2});
        { at::IntArrayRef stride = {1,1}; at::IntArrayRef pad = {0,0};
        h = at::conv2d(h, w_lora(324), w_lora(325), stride, pad); }
        
        // Up stage 2 (skip from stage 2 down)
        // output_blocks.2.0
        {
            auto skip = pop_skip();
            h = at::cat({h, skip}, 1);
            ResBlockWeights rbw = {
                w_lora(326), w_lora(327), w_lora(328), w_lora(329),
                w_lora(330), w_lora(331), w_lora(332), w_lora(333),
                w_lora(334), w_lora(335),
                w_lora(336), w_lora(337)
            };
            h = resblock_forward(h, temb, rbw);
        }
        // output_blocks.2.1
        {
            auto skip = pop_skip();
            h = at::cat({h, skip}, 1);
            ResBlockWeights rbw = {
                w_lora(338), w_lora(339), w_lora(340), w_lora(341),
                w_lora(342), w_lora(343), w_lora(344), w_lora(345),
                w_lora(346), w_lora(347),
                w_lora(348), w_lora(349)
            };
            h = resblock_forward(h, temb, rbw);
        }
        // output_blocks.2.2 → Transformer
        {
            TransformerWeights tfw = {
                w_lora(350), w_lora(351),
                w_lora(352), w_lora(353), w_lora(354), w_lora(355),
                w_lora(356), w_lora(357), w_lora(358), w_lora(359),
                w_lora(360), w_lora(361), w_lora(362), w_lora(363),
                w_lora(364), w_lora(365), w_lora(366), w_lora(367),
                w_lora(368), w_lora(369), w_lora(370), w_lora(371)
            };
            h = transformer_forward(h, text, tfw, 8);
        }
        h = at::upsample_nearest2d(h, {h.size(2)*2, h.size(3)*2});
        { at::IntArrayRef stride = {1,1}; at::IntArrayRef pad = {0,0};
        h = at::conv2d(h, w_lora(376), w_lora(377), stride, pad); }
        
        // Up stage 1 (skip from stage 1 down, no transformer)
        // output_blocks.3.0
        {
            auto skip = pop_skip();
            h = at::cat({h, skip}, 1);
            ResBlockWeights rbw = {
                w_lora(378), w_lora(379), w_lora(380), w_lora(381),
                w_lora(382), w_lora(383), w_lora(384), w_lora(385),
                w_lora(386), w_lora(387),
                w_lora(388), w_lora(389)
            };
            h = resblock_forward(h, temb, rbw);
        }
        // output_blocks.3.1
        {
            auto skip = pop_skip();
            h = at::cat({h, skip}, 1);
            ResBlockWeights rbw = {
                w_lora(390), w_lora(391), w_lora(392), w_lora(393),
                w_lora(394), w_lora(395), w_lora(396), w_lora(397),
                w_lora(398), w_lora(399),
                w_lora(400), w_lora(401)
            };
            h = resblock_forward(h, temb, rbw);
        }
        // output_blocks.3.2 → ResBlock (no transformer at end)
        {
            auto skip = pop_skip();
            h = at::cat({h, skip}, 1);
            ResBlockWeights rbw = {
                w_lora(402), w_lora(403), w_lora(404), w_lora(405),
                w_lora(406), w_lora(407), w_lora(408), w_lora(409),
                w_lora(410), w_lora(411),
                w_lora(412), w_lora(413)
            };
            h = resblock_forward(h, temb, rbw);
        }
        
        // ---- Output conv ----
        h = groupnorm_32(h, w_lora(414), w_lora(415), 320);
        h = at::silu(h);
        h = at::conv2d(h, w_lora(416), w_lora(417), at::IntArrayRef{1,1}, at::IntArrayRef{1,1});
        
        return wrap(h);
        
    } catch (const std::exception& e) {
        std::cerr << "torch_std_sd_unet_forward error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================
// SD 1.5 UNet forward from safetensors dict (inference convenience)
// ============================================================
// Builds the weight ptrs array from named safetensors dict,
// then calls torch_std_sd_unet_forward.

// Forward declarations for functions defined later in safetensors section
int   torch_std_safetensors_count(void* dict);
const char* torch_std_safetensors_name(void* dict, int idx);
void* torch_std_safetensors_tensor(void* dict, int idx);
void* torch_std_safetensors_get_tensor_by_name(void* dict, const char* name);

static const char* sd15_weight_names[] = {
    // Time embed (0-3)
    "time_embed.0.weight", "time_embed.0.bias",
    "time_embed.2.weight", "time_embed.2.bias",
    // Input conv (4-5)
    "input_blocks.0.0.weight", "input_blocks.0.0.bias",
    // input_blocks.1.0 — ResBlock (6-17)
    "input_blocks.1.0.in_layers.0.weight", "input_blocks.1.0.in_layers.0.bias",
    "input_blocks.1.0.in_layers.2.weight", "input_blocks.1.0.in_layers.2.bias",
    "input_blocks.1.0.skip_connection.weight", "input_blocks.1.0.skip_connection.bias",
    "input_blocks.1.0.emb_layers.1.weight", "input_blocks.1.0.emb_layers.1.bias",
    "input_blocks.1.0.out_layers.0.weight", "input_blocks.1.0.out_layers.0.bias",
    "input_blocks.1.0.out_layers.2.weight", "input_blocks.1.0.out_layers.2.bias",
    // input_blocks.1.1 — ResBlock (18-27)
    "input_blocks.1.1.in_layers.0.weight", "input_blocks.1.1.in_layers.0.bias",
    "input_blocks.1.1.in_layers.2.weight", "input_blocks.1.1.in_layers.2.bias",
    "input_blocks.1.1.emb_layers.1.weight", "input_blocks.1.1.emb_layers.1.bias",
    "input_blocks.1.1.out_layers.0.weight", "input_blocks.1.1.out_layers.0.bias",
    "input_blocks.1.1.out_layers.2.weight", "input_blocks.1.1.out_layers.2.bias",
    // input_blocks.2.0 — Downsample (28-29)
    "input_blocks.2.0.op.weight", "input_blocks.2.0.op.bias",
    // input_blocks.2.1 — ResBlock (30-41)
    "input_blocks.2.1.in_layers.0.weight", "input_blocks.2.1.in_layers.0.bias",
    "input_blocks.2.1.in_layers.2.weight", "input_blocks.2.1.in_layers.2.bias",
    "input_blocks.2.1.skip_connection.weight", "input_blocks.2.1.skip_connection.bias",
    "input_blocks.2.1.emb_layers.1.weight", "input_blocks.2.1.emb_layers.1.bias",
    "input_blocks.2.1.out_layers.0.weight", "input_blocks.2.1.out_layers.0.bias",
    "input_blocks.2.1.out_layers.2.weight", "input_blocks.2.1.out_layers.2.bias",
    // input_blocks.2.2 — ResBlock (42-51)
    "input_blocks.2.2.in_layers.0.weight", "input_blocks.2.2.in_layers.0.bias",
    "input_blocks.2.2.in_layers.2.weight", "input_blocks.2.2.in_layers.2.bias",
    "input_blocks.2.2.emb_layers.1.weight", "input_blocks.2.2.emb_layers.1.bias",
    "input_blocks.2.2.out_layers.0.weight", "input_blocks.2.2.out_layers.0.bias",
    "input_blocks.2.2.out_layers.2.weight", "input_blocks.2.2.out_layers.2.bias",
    // input_blocks.2.3 — Transformer (52-73)
    "input_blocks.2.3.norm.weight", "input_blocks.2.3.norm.bias",
    "input_blocks.2.3.attn1.to_q.weight", "input_blocks.2.3.attn1.to_q.bias",
    "input_blocks.2.3.attn1.to_k.weight", "input_blocks.2.3.attn1.to_k.bias",
    "input_blocks.2.3.attn1.to_v.weight", "input_blocks.2.3.attn1.to_v.bias",
    "input_blocks.2.3.attn1.to_out.0.weight", "input_blocks.2.3.attn1.to_out.0.bias",
    "input_blocks.2.3.attn2.to_q.weight", "input_blocks.2.3.attn2.to_q.bias",
    "input_blocks.2.3.attn2.to_k.weight", "input_blocks.2.3.attn2.to_k.bias",
    "input_blocks.2.3.attn2.to_v.weight", "input_blocks.2.3.attn2.to_v.bias",
    "input_blocks.2.3.attn2.to_out.0.weight", "input_blocks.2.3.attn2.to_out.0.bias",
    "input_blocks.2.3.ff.net.0.proj.weight", "input_blocks.2.3.ff.net.0.proj.bias",
    "input_blocks.2.3.ff.net.2.weight", "input_blocks.2.3.ff.net.2.bias",
    // (gap: indices 74-77, not used)
    // input_blocks.3.0 — Downsample (78-79)
    "input_blocks.3.0.op.weight", "input_blocks.3.0.op.bias",
    // input_blocks.3.1 — ResBlock (80-91)
    "input_blocks.3.1.in_layers.0.weight", "input_blocks.3.1.in_layers.0.bias",
    "input_blocks.3.1.in_layers.2.weight", "input_blocks.3.1.in_layers.2.bias",
    "input_blocks.3.1.skip_connection.weight", "input_blocks.3.1.skip_connection.bias",
    "input_blocks.3.1.emb_layers.1.weight", "input_blocks.3.1.emb_layers.1.bias",
    "input_blocks.3.1.out_layers.0.weight", "input_blocks.3.1.out_layers.0.bias",
    "input_blocks.3.1.out_layers.2.weight", "input_blocks.3.1.out_layers.2.bias",
    // input_blocks.3.2 — ResBlock (92-101)
    "input_blocks.3.2.in_layers.0.weight", "input_blocks.3.2.in_layers.0.bias",
    "input_blocks.3.2.in_layers.2.weight", "input_blocks.3.2.in_layers.2.bias",
    "input_blocks.3.2.emb_layers.1.weight", "input_blocks.3.2.emb_layers.1.bias",
    "input_blocks.3.2.out_layers.0.weight", "input_blocks.3.2.out_layers.0.bias",
    "input_blocks.3.2.out_layers.2.weight", "input_blocks.3.2.out_layers.2.bias",
    // input_blocks.3.3 — Transformer (102-123)
    "input_blocks.3.3.norm.weight", "input_blocks.3.3.norm.bias",
    "input_blocks.3.3.attn1.to_q.weight", "input_blocks.3.3.attn1.to_q.bias",
    "input_blocks.3.3.attn1.to_k.weight", "input_blocks.3.3.attn1.to_k.bias",
    "input_blocks.3.3.attn1.to_v.weight", "input_blocks.3.3.attn1.to_v.bias",
    "input_blocks.3.3.attn1.to_out.0.weight", "input_blocks.3.3.attn1.to_out.0.bias",
    "input_blocks.3.3.attn2.to_q.weight", "input_blocks.3.3.attn2.to_q.bias",
    "input_blocks.3.3.attn2.to_k.weight", "input_blocks.3.3.attn2.to_k.bias",
    "input_blocks.3.3.attn2.to_v.weight", "input_blocks.3.3.attn2.to_v.bias",
    "input_blocks.3.3.attn2.to_out.0.weight", "input_blocks.3.3.attn2.to_out.0.bias",
    "input_blocks.3.3.ff.net.0.proj.weight", "input_blocks.3.3.ff.net.0.proj.bias",
    "input_blocks.3.3.ff.net.2.weight", "input_blocks.3.3.ff.net.2.bias",
    // (gap: indices 124-127)
    // input_blocks.4.0 — Downsample (128-129)
    "input_blocks.4.0.op.weight", "input_blocks.4.0.op.bias",
    // input_blocks.4.1 — ResBlock (130-141)
    "input_blocks.4.1.in_layers.0.weight", "input_blocks.4.1.in_layers.0.bias",
    "input_blocks.4.1.in_layers.2.weight", "input_blocks.4.1.in_layers.2.bias",
    "input_blocks.4.1.skip_connection.weight", "input_blocks.4.1.skip_connection.bias",
    "input_blocks.4.1.emb_layers.1.weight", "input_blocks.4.1.emb_layers.1.bias",
    "input_blocks.4.1.out_layers.0.weight", "input_blocks.4.1.out_layers.0.bias",
    "input_blocks.4.1.out_layers.2.weight", "input_blocks.4.1.out_layers.2.bias",
    // input_blocks.4.2 — ResBlock (142-151)
    "input_blocks.4.2.in_layers.0.weight", "input_blocks.4.2.in_layers.0.bias",
    "input_blocks.4.2.in_layers.2.weight", "input_blocks.4.2.in_layers.2.bias",
    "input_blocks.4.2.emb_layers.1.weight", "input_blocks.4.2.emb_layers.1.bias",
    "input_blocks.4.2.out_layers.0.weight", "input_blocks.4.2.out_layers.0.bias",
    "input_blocks.4.2.out_layers.2.weight", "input_blocks.4.2.out_layers.2.bias",
    // input_blocks.4.3 — Transformer (150-171)
    "input_blocks.4.3.norm.weight", "input_blocks.4.3.norm.bias",
    "input_blocks.4.3.attn1.to_q.weight", "input_blocks.4.3.attn1.to_q.bias",
    "input_blocks.4.3.attn1.to_k.weight", "input_blocks.4.3.attn1.to_k.bias",
    "input_blocks.4.3.attn1.to_v.weight", "input_blocks.4.3.attn1.to_v.bias",
    "input_blocks.4.3.attn1.to_out.0.weight", "input_blocks.4.3.attn1.to_out.0.bias",
    "input_blocks.4.3.attn2.to_q.weight", "input_blocks.4.3.attn2.to_q.bias",
    "input_blocks.4.3.attn2.to_k.weight", "input_blocks.4.3.attn2.to_k.bias",
    "input_blocks.4.3.attn2.to_v.weight", "input_blocks.4.3.attn2.to_v.bias",
    "input_blocks.4.3.attn2.to_out.0.weight", "input_blocks.4.3.attn2.to_out.0.bias",
    "input_blocks.4.3.ff.net.0.proj.weight", "input_blocks.4.3.ff.net.0.proj.bias",
    "input_blocks.4.3.ff.net.2.weight", "input_blocks.4.3.ff.net.2.bias",
    // (gap: indices 172-175)
    // middle_block.0 — ResBlock (176-185)
    "middle_block.0.in_layers.0.weight", "middle_block.0.in_layers.0.bias",
    "middle_block.0.in_layers.2.weight", "middle_block.0.in_layers.2.bias",
    "middle_block.0.emb_layers.1.weight", "middle_block.0.emb_layers.1.bias",
    "middle_block.0.out_layers.0.weight", "middle_block.0.out_layers.0.bias",
    "middle_block.0.out_layers.2.weight", "middle_block.0.out_layers.2.bias",
    // middle_block.1 — Transformer (186-207)
    "middle_block.1.norm.weight", "middle_block.1.norm.bias",
    "middle_block.1.attn1.to_q.weight", "middle_block.1.attn1.to_q.bias",
    "middle_block.1.attn1.to_k.weight", "middle_block.1.attn1.to_k.bias",
    "middle_block.1.attn1.to_v.weight", "middle_block.1.attn1.to_v.bias",
    "middle_block.1.attn1.to_out.0.weight", "middle_block.1.attn1.to_out.0.bias",
    "middle_block.1.attn2.to_q.weight", "middle_block.1.attn2.to_q.bias",
    "middle_block.1.attn2.to_k.weight", "middle_block.1.attn2.to_k.bias",
    "middle_block.1.attn2.to_v.weight", "middle_block.1.attn2.to_v.bias",
    "middle_block.1.attn2.to_out.0.weight", "middle_block.1.attn2.to_out.0.bias",
    "middle_block.1.ff.net.0.proj.weight", "middle_block.1.ff.net.0.proj.bias",
    "middle_block.1.ff.net.2.weight", "middle_block.1.ff.net.2.bias",
    // (gap: indices 208-211)
    // middle_block.2 — ResBlock (212-221)
    "middle_block.2.in_layers.0.weight", "middle_block.2.in_layers.0.bias",
    "middle_block.2.in_layers.2.weight", "middle_block.2.in_layers.2.bias",
    "middle_block.2.emb_layers.1.weight", "middle_block.2.emb_layers.1.bias",
    "middle_block.2.out_layers.0.weight", "middle_block.2.out_layers.0.bias",
    "middle_block.2.out_layers.2.weight", "middle_block.2.out_layers.2.bias",
    // End marker
    nullptr
};

// Helper: get tensor from safetensors STDict, return empty tensor if not found
// Tries exact name first, then with common prefixes.
static const char* sd15_prefixes[] = {"model.diffusion_model.", "model.", "", nullptr};

static at::Tensor sd15_get_weight(void* dict_ptr, const char* name) {
    // Use the public API torch_std_safetensors_get_tensor_by_name which
    // has access to the STDict definition (defined later in the file).
    char buf[512];
    for (int p = 0; sd15_prefixes[p] != nullptr; p++) {
        snprintf(buf, sizeof(buf), "%s%s", sd15_prefixes[p], name);
        void* t = torch_std_safetensors_get_tensor_by_name(dict_ptr, buf);
        if (t) return *static_cast<at::Tensor*>(t);
    }
    return at::Tensor();
}

void* torch_std_sd15_unet_forward_dict(
    void* safetensors_dict,
    void* input_ptr, void* timestep_ptr, void* text_emb_ptr,
    void** lora_A, void** lora_B,
    int* lora_target_indices, int n_lora,
    double lora_scale) {
    try {
        // Count total weights
        int total = 0;
        while (sd15_weight_names[total] != nullptr) total++;
        
        // Allocate weight ptrs array
        std::vector<at::Tensor> weight_tensors(total);
        std::vector<void*> weight_ptrs(total, nullptr);
        
        for (int i = 0; i < total; i++) {
            auto t = sd15_get_weight(safetensors_dict, sd15_weight_names[i]);
            if (t.defined()) {
                weight_tensors[i] = t;
                weight_ptrs[i] = &weight_tensors[i];
            }
            // If not found, keep nullptr (will cause error in forward if accessed)
        }
        
        // Call the existing forward
        return torch_std_sd_unet_forward(
            weight_ptrs.data(), total,
            input_ptr, timestep_ptr, text_emb_ptr,
            lora_A, lora_B,
            lora_target_indices, n_lora,
            lora_scale);
    } catch (const std::exception& e) {
        std::cerr << "sd15_unet_forward_dict error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================
// Image I/O for training data
// ============================================================

void* torch_std_load_image(const char* path) {
    try {
        std::string spath(path);
        // Try TorchScript tensor first (pre-converted .pt files)
        if (spath.size() > 3 && spath.substr(spath.size()-3) == ".pt") {
            auto mod = torch::jit::load(path);
            if (mod.hasattr("data")) {
                return wrap(mod.attr("data").toTensor());
            }
            // Extract parameters directly
            auto named_params = mod.named_parameters();
            for (const auto& p : named_params) {
                return wrap(p.value.clone());
            }
        }
        // PPM/PGM raw format (no external deps needed)
        FILE* f = fopen(path, "rb");
        if (!f) { std::cerr << "load_image: cannot open " << path << std::endl; return nullptr; }
        char magic[3] = {0};
        int W=0, H=0, maxv=0;
        if (fscanf(f, "%2s %d %d %d", magic, &W, &H, &maxv) < 4) {
            fclose(f); std::cerr << "load_image: bad header " << path << std::endl; return nullptr;
        }
        fgetc(f); // skip newline after header
        // P5=grayscale, P6=RGB
        int C = (magic[1] == '6') ? 3 : 1;
        auto img = torch::empty({H, W, C}, at::TensorOptions().dtype(torch::kUInt8));
        size_t bytes_read = fread(img.data_ptr<uint8_t>(), 1, H*W*C, f);
        fclose(f);
        if ((int)bytes_read != H*W*C) {
            std::cerr << "load_image: short read " << path << std::endl;
            return nullptr;
        }
        // Convert HWC to CHW for PyTorch convention
        auto img_chw = img.permute({2, 0, 1});  // (C, H, W)
        return wrap(img_chw.contiguous());
    } catch (const std::exception& e) {
        std::cerr << "torch_std_load_image error: " << e.what() << std::endl;
        return nullptr;
    }
}

void torch_std_save_image(void* tensor, const char* path, int as_pgm) {
    try {
        auto& t = unwrap(tensor);
        auto cpu_t = t.to(torch::kCPU).to(torch::kFloat32);
        // VAE DEBUG: print pixel at [100,100]
        if (cpu_t.numel() > 0) {
            float r, g, b;
            if (cpu_t.dim() == 4 && cpu_t.size(1) == 3) { r=cpu_t[0][0][100][100].item<float>(); g=cpu_t[0][1][100][100].item<float>(); b=cpu_t[0][2][100][100].item<float>(); }
            else if (cpu_t.dim() == 4 && cpu_t.size(3) == 3) { r=cpu_t[0][100][100][0].item<float>(); g=cpu_t[0][100][100][1].item<float>(); b=cpu_t[0][100][100][2].item<float>(); }
            else if (cpu_t.dim() == 3 && cpu_t.size(0) == 3) { r=cpu_t[0][100][100].item<float>(); g=cpu_t[1][100][100].item<float>(); b=cpu_t[2][100][100].item<float>(); }
            else if (cpu_t.dim() == 3 && cpu_t.size(2) == 3) { r=cpu_t[100][100][0].item<float>(); g=cpu_t[100][100][1].item<float>(); b=cpu_t[100][100][2].item<float>(); }
            char buf[128];
            int n = snprintf(buf, sizeof(buf), "VAE_PIXEL [100,100] R=%.4f G=%.4f B=%.4f\n", r, g, b);
            write(2, buf, n);
        }
        
        // Handle various shapes: (B,C,H,W), (C,H,W), (H,W,C), (H,W)
        at::Tensor img;
        if (cpu_t.dim() == 4) {
            if (cpu_t.size(0) == 1) cpu_t = cpu_t.squeeze(0);
            if (cpu_t.dim() == 3) {
                if (cpu_t.size(0) <= 4) img = cpu_t.permute({1,2,0}).contiguous();
                else img = cpu_t.contiguous();
            } else {
                img = cpu_t;
            }
        } else if (cpu_t.dim() == 3) {
            if (cpu_t.size(0) <= 4) {
                img = cpu_t.permute({1,2,0}).contiguous();
            } else {
                img = cpu_t.contiguous();
            }
        } else {
            img = cpu_t;
        }
        
        // Normalize VAE output (typical [-1,1] range) to [0,255] and clamp
        img = at::clamp((img + 1.0) * 127.5, 0, 255).to(torch::kUInt8);
        
        int H = img.size(0), W = img.size(1);
        int C = img.dim() == 3 ? img.size(2) : 1;
        
        if (as_pgm || C == 1) {
            // Grayscale PGM
            at::Tensor gray;
            if (C >= 3) {
                gray = (0.299 * img.select(2, 0) + 
                        0.587 * img.select(2, 1) + 
                        0.114 * img.select(2, 2));
            } else {
                gray = img.squeeze(-1);
            }
            gray = gray.to(torch::kUInt8);
            
            FILE* f = fopen(path, "wb");
            if (!f) return;
            fprintf(f, "P5\n%d %d\n255\n", W, H);
            fwrite(gray.data_ptr<uint8_t>(), 1, H*W, f);
            fclose(f);
        } else {
            // Color PPM
            FILE* f = fopen(path, "wb");
            if (!f) return;
            fprintf(f, "P6\n%d %d\n255\n", W, H);
            fwrite(img.data_ptr<uint8_t>(), 1, H*W*C, f);
            fclose(f);
        }
    } catch (const std::exception& e) {
        std::cerr << "torch_std_save_image error: " << e.what() << std::endl;
    }
}

// ============================================================
// DDPM helpers
// ============================================================

void* torch_std_ddpm_betas(int T, double beta_start, double beta_end) {
    try {
        // Linear beta schedule
        auto betas = at::linspace(beta_start, beta_end, T, 
                                   at::TensorOptions().dtype(torch::kFloat64));
        return wrap(betas);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_ddpm_betas error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_ddpm_add_noise(void* latent, void* noise, void* timestep,
                                void* alphas_cumprod) {
    try {
        auto& x = unwrap(latent);
        auto& n = unwrap(noise);
        auto& t = unwrap(timestep);
        auto& ac = unwrap(alphas_cumprod);
        
        // Gather alpha_cumprod for each timestep
        auto ac_t = at::index(ac, {t.to(torch::kLong)});  // (B,)
        // Reshape for broadcasting: (B,) → (B,1,1,1)
        for (int i = 0; i < x.dim() - 1; i++) ac_t = ac_t.unsqueeze(-1);
        
        auto sqrt_ac = at::sqrt(ac_t);
        auto sqrt_one_minus_ac = at::sqrt(1 - ac_t);
        
        auto noisy = sqrt_ac * x + sqrt_one_minus_ac * n;
        return wrap(noisy);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_ddpm_add_noise error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================
// JIT module named parameters extraction
// ============================================================

int torch_std_jit_named_parameters(void* module, void** out_ptrs, 
                                    const char** out_names, int max_n) {
    try {
        auto* mod = static_cast<torch::jit::Module*>(module);
        auto params = mod->named_parameters();
        int count = 0;
        for (const auto& p : params) {
            if (count >= max_n) break;
            auto tensor = p.value.detach().clone();
            if (out_ptrs) out_ptrs[count] = new torch::Tensor(tensor);
            if (out_names) out_names[count] = strdup(p.name.c_str());
            count++;
        }
        return count;
    } catch (const std::exception& e) {
        std::cerr << "torch_std_jit_named_parameters error: " << e.what() << std::endl;
        return -1;
    }
}

// ============================================================
// safetensors 解析器 — 无外部依赖
// ============================================================
// 格式：
//   8 bytes: uint64_t header_len (little-endian)
//   header_len bytes: JSON header
//   ... data (8-byte aligned)
//
// Header JSON: { "tensor_name": {"dtype":"F32|F16|BF16|I64",
//                                  "shape": [dims...],
//                                  "data_offsets": [start, end]} }

// 极简 JSON 解析 — 只取 safetensors header 需要的 key-value
// 返回平坦 key/value 数组供 C 调用者处理
struct STKey {
    char key[256];
    char val[1024];
};

static int parse_safetensors_header(const char* json, int json_len,
                                      STKey* out, int max_keys) {
    int n = 0;
    int i = 0;
    auto skip_ws = [&]() { while (i < json_len && (json[i]==' '||json[i]=='\n'||json[i]=='\t'||json[i]=='\r')) i++; };
    auto expect = [&](char c) { skip_ws(); if (i < json_len && json[i]==c) { i++; return true; } return false; };
    auto read_string = [&](char* buf, int maxlen) {
        skip_ws();
        if (i >= json_len || json[i] != '"') return false;
        i++;  // skip opening quote
        int pos = 0;
        while (i < json_len && json[i] != '"' && pos < maxlen-1) {
            if (json[i] == '\\' && i+1 < json_len) { i++; }  // skip escape
            buf[pos++] = json[i++];
        }
        buf[pos] = '\0';
        if (i < json_len && json[i] == '"') i++;
        return true;
    };

    if (!expect('{')) return 0;
    while (n < max_keys) {
        skip_ws();
        if (expect('}')) break;
        if (!read_string(out[n].key, 256)) break;
        skip_ws();
        if (!expect(':')) break;
        // For safetensors, value is an object {"dtype":...,"shape":...,"data_offsets":...}
        // Store entire object as string
        skip_ws();
        int obj_start = i;
        int brace = 0;
        if (i < json_len && json[i] == '{') { brace = 1; i++; }
        while (i < json_len && brace > 0) {
            if (json[i] == '{') brace++;
            if (json[i] == '}') brace--;
            i++;
        }
        int obj_len = i - obj_start;
        if (obj_len > 0 && obj_len < 1024) {
            strncpy(out[n].val, json + obj_start, obj_len);
            out[n].val[obj_len] = '\0';
        }
        n++;
        skip_ws();
        expect(',');
    }
    return n;
}

static int json_get_string(const char* obj, const char* key, char* out, int maxlen) {
    // Find key in JSON object string
    const char* p = strstr(obj, key);
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    p++; while (*p && (*p==' '||*p=='\"')) p++;
    int pos = 0;
    int brackets = 0;
    while (*p && *p != '"' && pos < maxlen-1) {
        if (*p == '[' || *p == '{') brackets++;
        else if (*p == ']' || *p == '}') { brackets--; if (brackets < 0) break; }
        else if (*p == ',' && brackets <= 0) break;
        out[pos++] = *p++;
    }
    out[pos] = '\0';
    return pos > 0;
}

// Temporary debug - print first 10 val strings

static int json_get_int(const char* obj, const char* key) {
    char buf[64] = {0};
    if (!json_get_string(obj, key, buf, 64)) return 0;
    return atoi(buf);
}

static void json_get_ints(const char* obj, const char* key, int64_t* out, int max_n) {
    char buf[512] = {0};
    json_get_string(obj, key, buf, 512);
    // Parse "[a,b,c]" list
    char* p = buf;
    int idx = 0;
    while (*p && idx < max_n) {
        if (*p >= '0' && *p <= '9') {
            out[idx++] = atoll(p);
            while (*p && *p >= '0' && *p <= '9') p++;
        } else p++;
    }
}

// safetensors dtype string → torch scalar type
static int safetensors_dtype_to_torch(const char* dtype_str) {
    if (strcmp(dtype_str, "F32") == 0) return 0;  // kFloat
    if (strcmp(dtype_str, "F16") == 0) return 5;  // kHalf
    if (strcmp(dtype_str, "BF16") == 0) return 15; // kBFloat16
    if (strcmp(dtype_str, "I64") == 0) return 3;   // kLong
    if (strcmp(dtype_str, "I32") == 0) return 2;   // kInt
    if (strcmp(dtype_str, "I8") == 0) return 6;    // kChar
    if (strcmp(dtype_str, "U8") == 0) return 7;    // kByte
    return 0; // default float32
}

static int dtype_size(int dtype) {
    switch (dtype) {
        case 0: return 4;  // Float32
        case 1: return 8;  // Float64
        case 2: return 4;  // Int32
        case 3: return 8;  // Int64
        case 5: return 2;  // Half
        case 6: return 1;  // Char
        case 7: return 1;  // Byte
        case 15: return 2; // BFloat16
        default: return 4;
    }
}

// Main safetensors load function
// Returns opaque dict handle (simple array of key/tensor pairs)
// Users iterate with torch_safetensors_get()

struct STTensor {
    char name[256];
    void* tensor;
};

struct STDict {
    int count;
    STTensor entries[4096];
};

void* torch_std_safetensors_load(const char* path) {
    try {
        FILE* f = fopen(path, "rb");
        if (!f) { std::cerr << "safetensors: cannot open " << path << std::endl; return nullptr; }

        // Read header length
        uint64_t header_len = 0;
        if (fread(&header_len, 8, 1, f) != 1) {
            fclose(f); return nullptr;
        }

        // Read header
        char* header = (char*)malloc(header_len + 1);
        if (fread(header, 1, header_len, f) != header_len) {
            free(header); fclose(f); return nullptr;
        }
        header[header_len] = '\0';

        // Parse header
        STKey keys[4096];
        int n_keys = parse_safetensors_header(header, header_len, keys, 4096);
        free(header);

        auto* dict = new STDict();
        dict->count = 0;

        for (int i = 0; i < n_keys; i++) {
            if (dict->count >= 4096) break;
            auto& entry = dict->entries[dict->count];

            strncpy(entry.name, keys[i].key, 255);
            entry.name[255] = '\0';

            // Parse value object
            char dtype_str[32] = {0};
            json_get_string(keys[i].val, "dtype", dtype_str, 32);
            int torch_dtype = safetensors_dtype_to_torch(dtype_str);

            int64_t shape[16] = {0};
            json_get_ints(keys[i].val, "shape", shape, 16);

            int64_t offsets[2] = {0};
            json_get_ints(keys[i].val, "data_offsets", offsets, 2);

            // Seek to data
            int elem_size = dtype_size(torch_dtype);
            int64_t numel = 1;
            int ndim = 0;
            while (ndim < 16 && shape[ndim] > 0) {
                numel *= shape[ndim];
                ndim++;
            }

            int64_t data_start = 8 + header_len + offsets[0];
            int64_t data_size = (offsets[1] - offsets[0]);

            // Create tensor
            auto tensor = torch::empty(at::IntArrayRef(shape, ndim),
                                       at::TensorOptions().dtype(dtype_to_scalar(torch_dtype)));
            if (data_size > 0) {
                fseek(f, data_start, SEEK_SET);
                (void)fread(tensor.data_ptr(), 1, data_size, f);
            }

            entry.tensor = new torch::Tensor(tensor);
            dict->count++;
        }

        fclose(f);
        return dict;

    } catch (const std::exception& e) {
        std::cerr << "torch_std_safetensors_load error: " << e.what() << std::endl;
        return nullptr;
    }
}

int torch_std_safetensors_count(void* dict) {
    return static_cast<STDict*>(dict)->count;
}

const char* torch_std_safetensors_name(void* dict, int idx) {
    if (!dict || idx < 0) return "";
    auto* d = static_cast<STDict*>(dict);
    if (idx >= d->count) return "";
    return d->entries[idx].name;
}

void* torch_std_safetensors_tensor(void* dict, int idx) {
    if (!dict || idx < 0) return nullptr;
    auto* d = static_cast<STDict*>(dict);
    if (idx >= d->count) return nullptr;
    auto t = static_cast<torch::Tensor*>(d->entries[idx].tensor);
    return wrap(t->clone());  // return as tagged tensor
}

void torch_std_safetensors_free(void* dict) {
    if (!dict) return;
    auto* d = static_cast<STDict*>(dict);
    for (int i = 0; i < d->count; i++) {
        delete static_cast<torch::Tensor*>(d->entries[i].tensor);
    }
    delete d;
}

void* torch_std_safetensors_get_tensor_by_name(void* dict, const char* name) {
    if (!dict) return nullptr;
    auto* d = static_cast<STDict*>(dict);
    for (int i = 0; i < d->count; i++) {
        if (strcmp(d->entries[i].name, name) == 0) {
            return wrap(static_cast<torch::Tensor*>(d->entries[i].tensor)->clone());
        }
    }
    return nullptr;
}

// LoRA/LyCORIS merge: given a tensor and LoRA weights, apply W' = W + scale * B @ A
void* torch_std_lora_apply(void* weight, void* lora_A, void* lora_B, double scale) {
    try {
        auto& W = unwrap(weight);
        auto& A = unwrap(lora_A);
        auto& B = unwrap(lora_B);
        // W' = W + scale * B @ A
        auto BA = at::matmul(B, A) * scale;
        auto W_new = W + BA;
        return wrap(W_new);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_lora_apply error: " << e.what() << std::endl;
        return nullptr;
    }
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
    return x;
}

// 公开 API：从 safetensors 权重执行 CLIP 文本编码
// 输入：clip_dict - safetensors handle (STDict*)
//       token_ids - wrapped tensor (1,77) int64
//       d_model / n_layers / n_heads / d_ffn - 架构参数
// 返回：wrapped tensor (1,77,d_model)
void* torch_std_clip_text_forward_from_dict(void* clip_dict, void* token_ids,
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
// 采样器族 (DDIM / Euler / Euler Ancestral / DPM++ 2M)
// ============================================================

// DDIM sampler from sigmas (converts sigma→alpha_bar internally)
void* torch_std_sample_ddim_from_sigma(void* noise_pred, void* x_t,
                                        void* sigma_t, void* sigma_prev,
                                        double eta) {
    try {
        auto& sig_t = unwrap(sigma_t);
        auto& sig_prev = unwrap(sigma_prev);
        // alpha_bar = 1 / (1 + sigma^2)
        auto ab_t = (1.0f / (1.0f + sig_t * sig_t)).item<float>();
        auto ab_prev = (1.0f / (1.0f + sig_prev * sig_prev)).item<float>();
        return torch_std_sample_ddim(noise_pred, x_t,
                                     wrap(at::tensor(ab_t)),
                                     wrap(at::tensor(ab_prev)), eta);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_sample_ddim_from_sigma error: " << e.what() << std::endl;
        return nullptr;
    }
}

// DDIM sampler — 一步去噪
void* torch_std_sample_ddim(void* noise_pred, void* x_t, void* alpha_bar_t,
                             void* alpha_bar_prev, double eta) {
    try {
        auto& eps = unwrap(noise_pred);      // predicted noise
        auto& xt = unwrap(x_t);              // current noisy latent
        auto& ab_t = unwrap(alpha_bar_t);    // alpha_cumprod at t
        auto& ab_prev = unwrap(alpha_bar_prev); // alpha_cumprod at t-1

        auto sqrt_ab_t = at::sqrt(ab_t);
        auto sqrt_ab_prev = at::sqrt(ab_prev);
        auto sqrt_one_minus_ab_t = at::sqrt(1 - ab_t);
        auto sqrt_one_minus_ab_prev = at::sqrt(1 - ab_prev);

        // Predict x0: (x_t - sqrt(1-ab_t) * eps) / sqrt(ab_t)
        auto x0_pred = (xt - sqrt_one_minus_ab_t * eps) / sqrt_ab_t;

        // Direction pointing to x_t
        auto sigma = eta * at::sqrt((1 - ab_prev) / (1 - ab_t)) * at::sqrt(1 - ab_t / ab_prev);
        // Clamp sigma to avoid numerical issues
        sigma = at::clamp(sigma, 0.0, 1.0);

        auto c1 = sqrt_ab_prev;
        auto c2 = sqrt_one_minus_ab_prev - sigma;

        auto x_prev = c1 * x0_pred + c2 * eps;

        // Add random noise scaled by sigma
        if (eta > 0) {
            auto noise = at::randn_like(eps);
            x_prev = x_prev + sigma * noise;
        }

        return wrap(x_prev);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_sample_ddim error: " << e.what() << std::endl;
        return nullptr;
    }
}

// Euler sampler — 一步去噪
void* torch_std_sample_euler(void* noise_pred, void* x_t, void* sigma_t, void* sigma_prev) {
    try {
        auto& eps = unwrap(noise_pred);
        auto& xt = unwrap(x_t);
        auto& sig_t = unwrap(sigma_t);
        auto& sig_prev = unwrap(sigma_prev);
        auto dev = xt.device();
        auto sig_t_d = sig_t.to(dev);
        auto sig_prev_d = sig_prev.to(dev);
        auto d = (xt - eps) / sig_t_d;
        auto step = sig_prev_d - sig_t_d;
        auto x_prev = xt + step * d;
        return wrap(x_prev);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_sample_euler error: " << e.what() << std::endl;
        return nullptr;
    }
}

// Fused CFG-predict + Euler step: all intermediate tensors stay in C++ RAII scope
// Takes cond/uncond UNet outputs directly, no Scheme-level arithmetic crossings
void* torch_std_euler_step(void* x_ptr, void* sigma_t_ptr, void* sigma_next_ptr,
                           void* cond_ptr, void* uncond_ptr, double cfg) {
    try {
        auto& x = unwrap(x_ptr);
        auto& sig_t = unwrap(sigma_t_ptr);
        auto& sig_next = unwrap(sigma_next_ptr);
        auto& cond = unwrap(cond_ptr);
        auto& uncond = unwrap(uncond_ptr);
        auto dev = x.device();

        torch::Tensor eps;
        if (cfg <= 1.0) {
            eps = cond;
        } else {
            eps = uncond + (cond - uncond) * cfg;
        }

        auto sig_t_d = sig_t.to(dev);
        auto sig_next_d = sig_next.to(dev);
        auto dt = sig_next_d - sig_t_d;
        auto result = x + dt * eps;
        {
            static int step = 0;
            char buf[256];
            int n = snprintf(buf, sizeof(buf), "STK_EULER step=%d x=%.4f cond=%.4f uncond=%.4f eps=%.4f sig_t=%.4f\n",
                step, x.abs().mean().item<float>(), cond.abs().mean().item<float>(),
                uncond.abs().mean().item<float>(), eps.abs().mean().item<float>(), sig_t_d.item<float>());
            write(2, buf, n); step++;
        }
        return wrap(result);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_euler_step error: " << e.what() << std::endl;
        return nullptr;
    }
}

// Euler Ancestral sampler
void* torch_std_sample_euler_ancestral(void* noise_pred, void* x_t,
                                        void* sigma_t, void* sigma_prev) {
    try {
        auto& eps = unwrap(noise_pred);
        auto& xt = unwrap(x_t);
        auto& sig_t = unwrap(sigma_t);
        auto& sig_prev = unwrap(sigma_prev);

        // sigma_noise = sqrt(sigma_prev^2 - sigma_t^2)  (ancestral noise)
        auto sigma_noise = at::sqrt(sig_prev * sig_prev - sig_t * sig_t);

        // d = (x_t - eps) / sigma_t
        auto sig_t_d = sig_t.to(xt.device());
        auto sig_prev_d = sig_prev.to(xt.device());
        auto d = (xt - eps) / sig_t_d;
        auto step = sig_prev_d - sig_t_d;

        auto x_prev = xt + (sig_prev - sig_t) * d;
        // Add ancestral noise
        auto noise = at::randn_like(eps);
        x_prev = x_prev + sigma_noise * noise;

        return wrap(x_prev);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_sample_euler_ancestral error: " << e.what() << std::endl;
        return nullptr;
    }
}

// DPM++ 2M sampler (二阶)
void* torch_std_sample_dpmpp_2m(void* noise_pred, void* x_t,
                                  void* sigma_t, void* sigma_prev,
                                  void* old_denoised, int is_first_step) {
    try {
        auto& eps = unwrap(noise_pred);
        auto& xt = unwrap(x_t);
        auto& sig_t = unwrap(sigma_t);
        auto& sig_prev = unwrap(sigma_prev);

        // Denoised prediction: denoised = x_t - sigma_t * eps
        auto denoised = xt - sig_t * eps;

        if (is_first_step) {
            // 一阶 Euler 步
            auto d = (xt - denoised) / sig_t;
            auto x_prev = xt + (sig_prev - sig_t) * d;
            auto* result = new torch::Tensor(x_prev);
            auto* old_d = new torch::Tensor(denoised);
            // Pack both into returned tensor pair — caller unpacks
            return wrap(at::stack({x_prev, denoised}));
        } else {
            // 二阶 step
            auto& denoised_old = unwrap(old_denoised);
            // h = sigma_prev - sigma_t, h_old = sigma_t_old - sigma_t
            // DPM++ 2M formula
            auto h = sig_prev - sig_t;
            // Simplified: use linear extrapolation
            auto d = (xt - denoised) / sig_t;
            auto d_old = (xt - denoised_old) / sig_t;
            // 二阶修正
            auto x_prev = xt + h * d + 0.5 * h * h * (d - d_old) / (sig_t * 0.01 + 1e-5);
            return wrap(at::stack({x_prev, denoised}));
        }
    } catch (const std::exception& e) {
        std::cerr << "torch_std_sample_dpmpp_2m error: " << e.what() << std::endl;
        return nullptr;
    }
}

// Sampler sigma schedule (Karras / exponential / linear)
void* torch_std_sampler_sigmas(int steps, double sigma_min, double sigma_max,
                                const char* schedule) {
    try {
        std::string sched(schedule ? schedule : "karras");
        at::Tensor sigmas;
        if (sched == "karras") {
            // Karras: sigma_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
            double rho = 7.0;
            auto inv_rho = 1.0 / rho;
            auto max_pow = std::pow(sigma_max, inv_rho);
            auto min_pow = std::pow(sigma_min, inv_rho);
            double step = (steps > 1) ? (max_pow - min_pow) / (steps - 1) : 0.0;
            sigmas = at::zeros(steps);
            for (int i = 0; i < steps; i++) {
                sigmas[i] = std::pow(max_pow - i * step, rho);
            }
        } else if (sched == "exponential") {
            auto log_max = std::log(sigma_max);
            auto log_min = std::log(sigma_min);
            sigmas = at::exp(at::linspace(log_max, log_min, steps));
        } else {
            // Linear
            sigmas = at::linspace(sigma_max, sigma_min, steps);
        }
        // Append trailing 0 so samplers can denoise sigma_1 -> 0
        auto zero_cat = at::zeros(1, sigmas.options());
        sigmas = at::cat({sigmas, zero_cat}, 0);
        return wrap(sigmas.to(torch::kCUDA));
    } catch (const std::exception& e) {
        std::cerr << "torch_std_sampler_sigmas error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================
// 图像处理原语
// ============================================================

void* torch_std_image_resize(void* img, int new_h, int new_w, const char* mode) {
    try {
        auto& input = unwrap(img);
        std::string interp(mode ? mode : "bilinear");

        // Input: (C, H, W) or (H, W, C) uint8 or float
        at::Tensor tensor;
        bool is_chw = (input.dim() == 3 && input.size(0) <= 4);
        if (!is_chw) {
            // (H,W,C) → (C,H,W)
            tensor = input.permute({2, 0, 1}).unsqueeze(0); // (1,C,H,W)
        } else {
            tensor = input.unsqueeze(0);  // (1,C,H,W)
        }

        if (tensor.dtype() == torch::kUInt8) {
            tensor = tensor.to(torch::kFloat32) / 255.0;
        }

        // Interpolate — use PyTorch 2.4 C++ API
        namespace F = torch::nn::functional;

        torch::nn::functional::InterpolateFuncOptions opts;
        opts.size(std::vector<int64_t>({new_h, new_w}));
        if (interp == "nearest") opts.mode(torch::kNearest);
        else if (interp == "bicubic") opts.mode(torch::kBicubic);
        // Lanczos not available in PyTorch C++ API; fallback to bicubic
        else if (interp == "lanczos") opts.mode(torch::kBicubic);
        else opts.mode(torch::kBilinear);  // default

        auto resized = F::interpolate(tensor, opts);
        resized = (resized * 255.0).clamp(0, 255).to(torch::kUInt8);
        resized = resized.squeeze(0);  // (C,H,W)

        return wrap(resized);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_image_resize error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_image_crop(void* img, int x, int y, int w, int h) {
    try {
        auto& input = unwrap(img);
        // Assume (C, H, W)
        auto cropped = input.slice(1, y, y + h).slice(2, x, x + w);
        return wrap(cropped.clone());
    } catch (const std::exception& e) {
        std::cerr << "torch_std_image_crop error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_image_composite(void* base, void* overlay, int x, int y) {
    try {
        auto& b = unwrap(base);
        auto& o = unwrap(overlay);
        // base: (C, H, W), overlay: (C, h, w) or (C, h, w) with alpha
        int C = b.size(0);
        int oh = o.size(1), ow = o.size(2);
        auto result = b.clone();

        if (C == 4) {
            // RGBA with alpha blending
            auto alpha = o.slice(0, 3, 4).to(torch::kFloat32) / 255.0;
            for (int c = 0; c < 3; c++) {
                auto base_region = result.slice(0, c, c+1).slice(1, y, y+oh).slice(2, x, x+ow);
                auto overlay_region = o.slice(0, c, c+1);
                // result = overlay * alpha + base * (1 - alpha)
                auto blended = overlay_region.to(torch::kFloat32) * alpha +
                               base_region.to(torch::kFloat32) * (1.0 - alpha);
                result.slice(0, c, c+1).slice(1, y, y+oh).slice(2, x, x+ow).copy_(blended.to(torch::kUInt8));
            }
        } else {
            // No alpha: direct copy
            result.slice(0, 0, C).slice(1, y, y+oh).slice(2, x, x+ow).copy_(o);
        }
        return wrap(result);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_image_composite error: " << e.what() << std::endl;
        return nullptr;
    }
}

// Color space conversion: RGB ↔ HSV, RGB ↔ LAB, RGB ↔ YCbCr
void* torch_std_color_convert(void* img, const char* from_space, const char* to_space) {
    try {
        auto& input = unwrap(img);
        auto f = input.to(torch::kFloat32);
        bool from_rgb = (strcmp(from_space, "rgb") == 0);
        bool to_rgb = (strcmp(to_space, "rgb") == 0);

        if (from_rgb && strcmp(to_space, "hsv") == 0) {
            // RGB → HSV
            auto r = f.slice(0, 0, 1) / 255.0;
            auto g = f.slice(0, 1, 2) / 255.0;
            auto b = f.slice(0, 2, 3) / 255.0;
            auto mx = at::max(at::max(r, g), b);
            auto mn = at::min(at::min(r, g), b);
            auto diff = mx - mn;
            auto h = at::zeros_like(r);
            auto s = at::zeros_like(r);
            auto v = mx;

            // Hue calculation (approximate)
            auto mask_r = (mx == r);
            auto mask_g = (mx == g);
            auto mask_b = (mx == b);
            // Where mx == r: h = (g - b) / diff
            auto h_r = at::where(mask_r, (g - b) / (diff + 1e-6), h);
            // Where mx == g: h = 2.0 + (b - r) / diff
            auto h_g = at::where(mask_g, 2.0 + (b - r) / (diff + 1e-6), h);
            // Where mx == b: h = 4.0 + (r - g) / diff
            auto h_b = at::where(mask_b, 4.0 + (r - g) / (diff + 1e-6), h);
            h = h_r + h_g + h_b;
            h = h / 6.0;  // normalize to [0,1]
            h = at::fmod(h + 1.0, 1.0);  // keep positive

            s = at::where(diff > 0.01, diff / (mx + 1e-6), s);

            auto result = at::cat({h, s, v}, 0);
            return wrap(result);
        }
        else if (from_rgb && strcmp(to_space, "ycrcb") == 0) {
            // RGB → YCrCb (BT.601)
            auto r = f.slice(0, 0, 1);
            auto g = f.slice(0, 1, 2);
            auto b = f.slice(0, 2, 3);
            auto y  = 0.299 * r + 0.587 * g + 0.114 * b;
            auto cr = 128.0 + (r - y) * 0.713;
            auto cb = 128.0 + (b - y) * 0.564;
            return wrap(at::cat({y, cr, cb}, 0));
        }
        else if (strcmp(from_space, "ycbcr") == 0 && to_rgb) {
            // YCrCb → RGB
            auto y  = f.slice(0, 0, 1);
            auto cr = f.slice(0, 1, 2) - 128.0;
            auto cb = f.slice(0, 2, 3) - 128.0;
            auto r = y + 1.403 * cr;
            auto g = y - 0.714 * cr - 0.344 * cb;
            auto b = y + 1.773 * cb;
            r = at::clamp(r, 0, 255);
            g = at::clamp(g, 0, 255);
            b = at::clamp(b, 0, 255);
            return wrap(at::cat({r, g, b}, 0));
        }

        // Default: return input unchanged
        return wrap(input.clone());

    } catch (const std::exception& e) {
        std::cerr << "torch_std_color_convert error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================
// ControlNet forward — 并行 UNet 分支 + 特征注入
// ============================================================
// 输入: latent, timestep, text_emb, hint (conditioning image)
// ControlNet 输出 13 个特征图，加 13 个 zero_conv 权重
// 这些特征图加到 main UNet 的 down/mid block 输出

void* torch_std_controlnet_forward(
    void** weight_ptrs, int n_weights,
    void* input, void* timestep, void* text_emb, void* hint,
    int num_hint_channels) {
    try {
        // ControlNet 架构与 UNet 前半部分相同 (encoder),
        // 但每个 block 输出经过 zero_conv 后作为特征注入
        auto x = unwrap(input);
        auto t = unwrap(timestep);
        auto text = unwrap(text_emb);
        auto h = unwrap(hint);

        auto w = [&](int i) -> at::Tensor& {
            return *static_cast<at::Tensor*>(weight_ptrs[i]);
        };

        // Time embedding
        auto temb = timestep_embedding(t, 320);
        temb = at::linear(temb, w(0), w(1));
        temb = at::silu(temb);
        temb = at::linear(temb, w(2), w(3));

        // Input hint conv
        auto hint_conv = at::conv2d(h, w(4), w(5), at::IntArrayRef{1,1}, at::IntArrayRef{1,1});

        // Input blocks (same as UNet encoder but with hint added)
        auto h_in = at::conv2d(x, w(6), w(7), at::IntArrayRef{1,1}, at::IntArrayRef{1,1}) + hint_conv;

        // Collect 13 control outputs
        std::vector<at::Tensor> controls;
        // Simplified: collect block outputs through zero_convs
        // In real ControlNet, extract after each ResBlock/Transformer

        // Return stacked control features (13 of them, each [B,C,H,W])
        // Simplified: return 13 feature maps (some may be empty)
        auto stacked = at::cat({h_in}, 0);  // placeholder
        return wrap(stacked);

    } catch (const std::exception& e) {
        std::cerr << "torch_std_controlnet_forward error: " << e.what() << std::endl;
        return nullptr;
    }
}

// Apply ControlNet to UNet: add control features to UNet block outputs
// Takes UNet module, ControlNet features, and returns modified UNet output
void* torch_std_controlnet_apply(void* unet_features, void* control_features,
                                  double strength) {
    try {
        auto& uf = unwrap(unet_features);
        auto& cf = unwrap(control_features);
        // uf: stacked UNet intermediate outputs
        // cf: stacked ControlNet outputs
        // Apply: uf_i = uf_i + strength * cf_i
        auto result = uf + strength * cf;
        return wrap(result);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_controlnet_apply error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================
// LoRA key→index matching for SD1.5 UNet
// ============================================================
// Given a LoRA safetensors dict, match keys against sd15_weight_names[]
// and populate output arrays for the UNet forward.
// Uses public API functions (torch_std_safetensors_*) to avoid needing
// STDict definition here.
// Returns number of matched LoRA pairs, or -1 on error.
int torch_std_lora_match_to_unet(void* lora_dict, int n_weights,
                                  int64_t* out_indices,
                                  void** out_A, void** out_B,
                                  int max_lora) {
    try {
        int count = torch_std_safetensors_count(lora_dict);
        int matched = 0;

        for (int li = 0; li < count && matched < max_lora; li++) {
            const char* name_c = torch_std_safetensors_name(lora_dict, li);
            if (!name_c) continue;
            std::string lname(name_c);

            // Look for "lora_down.weight" (end of key)
            auto down_pos = lname.find("lora_down.weight");
            if (down_pos == std::string::npos) continue;

            // Extract base key
            std::string base = lname.substr(0, down_pos);
            std::string clean = base;
            // Strip common prefixes
            if (base.substr(0, 10) == "lora_unet_") clean = base.substr(10);
            else if (base.substr(0, 5) == "lora_") clean = base.substr(5);

            // Simple dot conversion: replace underscores with dots
            // Pattern: "transformer_blocks_0_..." -> "transformer_blocks.0."
            std::string unet_name;
            for (size_t j = 0; j < clean.length(); j++) {
                if (clean[j] == '_' && j > 0 && isdigit(clean[j-1])) {
                    unet_name += '.';
                } else if (clean[j] == '_' && j+1 < clean.length() &&
                           isdigit(clean[j+1])) {
                    unet_name += '.';
                } else if (clean[j] != '_') {
                    unet_name += clean[j];
                }
            }
            // Ensure .weight suffix
            if (unet_name.length() < 7 ||
                unet_name.substr(unet_name.length()-7) != ".weight") {
                unet_name += ".weight";
            }

            // Find matching index in sd15_weight_names[]
            for (int wi = 0; wi < n_weights; wi++) {
                if (sd15_weight_names[wi] == nullptr) break;
                if (unet_name == sd15_weight_names[wi]) {
                    // Found match — get the corresponding up tensor
                    std::string up_name = lname.substr(0, down_pos) + "lora_up.weight";
                    for (int uj = 0; uj < count; uj++) {
                        const char* up_name_c = torch_std_safetensors_name(lora_dict, uj);
                        if (up_name_c && up_name == std::string(up_name_c)) {
                            out_indices[matched] = wi;
                            out_A[matched] = torch_std_safetensors_tensor(lora_dict, li);
                            out_B[matched] = torch_std_safetensors_tensor(lora_dict, uj);
                            matched++;
                            break;
                        }
                    }
                    break;
                }
            }
        }
        return matched;
    } catch (const std::exception& e) {
        std::cerr << "lora_match_to_unet error: " << e.what() << std::endl;
        return -1;
    }
}

// ============================================================
// VAE tiling — 大图分块编解码
// ============================================================

void* torch_std_vae_encode(void* vae_module, void* image) {
    try {
        auto* vae = static_cast<torch::jit::Module*>(vae_module);
        auto& img = unwrap(image);
        // img: (B, C, H, W) float [-1, 1]
        auto latent = vae->forward({img}).toTensor() * 0.18215f;
        return wrap(latent);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_vae_encode error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_vae_decode(void* vae_module, void* latent) {
    try {
        auto* vae = static_cast<torch::jit::Module*>(vae_module);
        auto& lat = unwrap(latent);
        // VAE was loaded on CUDA by torch_std_jit_load; do NOT call .to() on module
        auto lat_ready = lat.to(torch::kFloat32);
        auto decoded = vae->forward({lat_ready / 0.18215f}).toTensor();
        return wrap(decoded);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_vae_decode error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_vae_encode_tiled(void* vae_module, void* image,
                                  int tile_size, int overlap) {
    try {
        auto* vae = static_cast<torch::jit::Module*>(vae_module);
        auto& img = unwrap(image);
        // img: (B, C, H, W) float [-1, 1]

        int B = img.size(0), C = img.size(1), H = img.size(2), W = img.size(3);
        int stride = tile_size - overlap;

        int n_h = (H + stride - 1) / stride;
        int n_w = (W + stride - 1) / stride;

        auto out = torch::zeros({B, 4, H/8, W/8}, img.options().dtype(torch::kFloat16));

        // Blending weights for overlap regions
        auto weight = at::ones({tile_size, tile_size});
        // Linear ramp for overlap
        for (int i = 0; i < overlap; i++) {
            float alpha = (float)(i + 1) / (overlap + 1);
            weight.slice(0, i, i+1).fill_(alpha);
            weight.slice(0, tile_size-i-1, tile_size-i).fill_(alpha);
            weight.slice(1, i, i+1).fill_(alpha);
            weight.slice(1, tile_size-i-1, tile_size-i).fill_(alpha);
        }

        for (int iy = 0; iy < n_h; iy++) {
            for (int ix = 0; ix < n_w; ix++) {
                int y0 = iy * stride;
                int x0 = ix * stride;
                int y1 = std::min(y0 + tile_size, H);
                int x1 = std::min(x0 + tile_size, W);

                auto tile = img.slice(2, y0, y1).slice(3, x0, x1);
                // Pad if tile is smaller than tile_size
                if (tile.size(2) < tile_size || tile.size(3) < tile_size) {
                    tile = at::constant_pad_nd(tile, {0, tile_size - tile.size(3),
                                                      0, tile_size - tile.size(2)}, 0);
                }
                tile = tile.unsqueeze(0);  // (1, C, Ts, Ts)

                // VAE encode
                auto latent = vae->forward({tile}).toTensor() * 0.18215;

                // Blend into output
                // (1, 4, Ts/8, Ts/8)
                int ly0 = y0 / 8, lx0 = x0 / 8;
                int ly1 = y1 / 8, lx1 = x1 / 8;
                auto weight_8 = at::upsample_nearest2d(weight.unsqueeze(0).unsqueeze(0),
                                                       {tile_size/8, tile_size/8});
                auto tile_out = latent * weight_8;
                out.slice(2, ly0, ly1).slice(3, lx0, lx1) += tile_out.squeeze(0);
            }
        }

        return wrap(out);

    } catch (const std::exception& e) {
        std::cerr << "torch_std_vae_encode_tiled error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_vae_decode_tiled(void* vae_module, void* latent,
                                  int tile_size, int overlap) {
    try {
        auto* vae = static_cast<torch::jit::Module*>(vae_module);
        auto& lat = unwrap(latent);
        // lat: (B, 4, H, W) float

        int B = lat.size(0), H = lat.size(2), W = lat.size(3);
        int lt = tile_size / 8;  // latent tile size
        int stride = lt - overlap / 8;

        int n_h = (H + stride - 1) / stride;
        int n_w = (W + stride - 1) / stride;

        auto out = torch::zeros({B, 3, H*8, W*8}, lat.options().dtype(torch::kFloat16));

        // Blending weights for overlap
        auto weight = at::ones({tile_size, tile_size});
        for (int i = 0; i < overlap; i++) {
            float alpha = (float)(i + 1) / (overlap + 1);
            weight.slice(0, i, i+1).fill_(alpha);
            weight.slice(0, tile_size-i-1, tile_size-i).fill_(alpha);
            weight.slice(1, i, i+1).fill_(alpha);
            weight.slice(1, tile_size-i-1, tile_size-i).fill_(alpha);
        }

        for (int iy = 0; iy < n_h; iy++) {
            for (int ix = 0; ix < n_w; ix++) {
                int y0 = iy * stride;
                int x0 = ix * stride;
                int y1 = std::min(y0 + lt, H);
                int x1 = std::min(x0 + lt, W);

                auto tile = lat.slice(2, y0, y1).slice(3, x0, x1);
                if (tile.size(2) < lt || tile.size(3) < lt) {
                    tile = at::constant_pad_nd(tile, {0, lt - tile.size(3),
                                                      0, lt - tile.size(2)}, 0);
                }

                // VAE decode
                auto decoded = vae->forward({tile / 0.18215}).toTensor();

                // Blend
                int dy0 = y0 * 8, dx0 = x0 * 8;
                auto weight_b = weight.to(decoded.device());
                auto tile_dec = decoded * weight_b;
                out.slice(2, dy0, dy0+tile_size).slice(3, dx0, dx0+tile_size) += tile_dec;
            }
        }

        return wrap(out);

    } catch (const std::exception& e) {
        std::cerr << "torch_std_vae_decode_tiled error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================
// All-in-one LoRA/LyCORIS weight injection
// ============================================================
// Given a model name→tensor dict (safetensors) and a target module,
// apply LoRA weights to all matching parameters.

int torch_std_lora_merge_into(void* model_dict, void* lora_dict,
                               const char* prefix, double scale) {
    try {
        auto* model = static_cast<STDict*>(model_dict);
        auto* lora = static_cast<STDict*>(lora_dict);
        int count = 0;

        for (int mi = 0; mi < model->count; mi++) {
            std::string mname(model->entries[mi].name);
            auto* m_tensor = static_cast<torch::Tensor*>(model->entries[mi].tensor);

            // Look for matching LoRA: lora_name = prefix + model_name
            std::string lora_name = std::string(prefix) + mname;
            // Also try with ".weight" removed
            std::string alt_name = lora_name;
            auto dot_pos = lora_name.rfind(".weight");
            if (dot_pos != std::string::npos) {
                alt_name = lora_name.substr(0, dot_pos);
            }

            for (int li = 0; li < lora->count; li++) {
                std::string lname(lora->entries[li].name);
                // Check if lora entry is like "lora.unet.xxx.lora_down.weight"
                // and matches the model weight
                if (lname.find(lora_name) != std::string::npos ||
                    lname.find(alt_name) != std::string::npos) {
                    // Found a LoRA weight — apply
                    // LoRA is stored as pairs: lora_A = "lora.unet.xxx.lora_down.weight"
                    //                        lora_B = "lora.unet.xxx.lora_up.weight"
                    auto* l_tensor = static_cast<torch::Tensor*>(lora->entries[li].tensor);
                    *m_tensor = *m_tensor + *l_tensor * scale;
                    count++;
                    break;
                }
            }
        }
        return count;
    } catch (const std::exception& e) {
        std::cerr << "torch_std_lora_merge_into error: " << e.what() << std::endl;
        return -1;
    }
}

// ============================================================
// PNG/JPEG encoding — 通过链接 libpng/libjpeg
// 需要 -lpng -ljpeg 编译选项；若不可用则 fallback 到 PPM
// ============================================================

void* torch_std_load_image_png(const char* path) {
    // If libpng is not linked, fall back to PPM
    FILE* f = fopen(path, "rb");
    if (!f) return nullptr;
    unsigned char sig[8];
    if (fread(sig, 1, 8, f) != 8) { fclose(f); return nullptr; }
    fclose(f);

    // Check PNG signature
    const unsigned char png_sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (memcmp(sig, png_sig, 8) == 0) {
#ifdef WITH_LIBPNG
        // Full PNG loading with libpng
        // (implementation omitted — depends on linking libpng)
        std::cerr << "PNG support requires compiling with -lPNG" << std::endl;
        return nullptr;
#else
        std::cerr << "PNG not supported (build with -DWITH_LIBPNG -lpng). Loading as PPM/PGM." << std::endl;
        return nullptr;
#endif
    }
    return nullptr;
}

#include <png.h>

void torch_std_save_image_png(void* tensor, const char* path) {
    try {
        auto& t = unwrap(tensor);
        auto cpu_t = t.to(torch::kCPU).to(torch::kFloat32);
        at::Tensor img;
        if (cpu_t.dim() == 4) {
            if (cpu_t.size(0) == 1) cpu_t = cpu_t.squeeze(0);
            if (cpu_t.dim() == 3 && cpu_t.size(0) <= 4) img = cpu_t.permute({1,2,0});
            else img = cpu_t;
        } else if (cpu_t.dim() == 3) {
            if (cpu_t.size(0) <= 4) img = cpu_t.permute({1,2,0});
            else img = cpu_t;
        } else img = cpu_t;
        // Normalize VAE output (typical [-1,1] range) to [0,255] and clamp
        img = at::clamp((img + 1.0) * 127.5, 0, 255).to(torch::kUInt8);
        int H = img.size(0), W = img.size(1);
        int C = img.dim() == 3 ? img.size(2) : 1;
        FILE* f = fopen(path, "wb");
        if (!f) return;
        png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if (!png) { fclose(f); return; }
        png_infop info = png_create_info_struct(png);
        if (!info) { png_destroy_write_struct(&png, NULL); fclose(f); return; }
        if (setjmp(png_jmpbuf(png))) { png_destroy_write_struct(&png, &info); fclose(f); return; }
        png_init_io(png, f);
        int color_type = (C >= 3) ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_GRAY;
        png_set_IHDR(png, info, W, H, 8, color_type, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png, info);
        std::vector<uint8_t> row(W * C);
        for (int y = 0; y < H; y++) {
            uint8_t* src = img.data_ptr<uint8_t>() + y * W * C;
            memcpy(row.data(), src, W * C);
            png_write_row(png, row.data());
        }
        png_write_end(png, NULL);
        png_destroy_write_struct(&png, &info);
        fclose(f);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_save_image_png error: " << e.what() << std::endl;
    }
}

// ============================================================
// CLIP BPE Tokenizer
// ============================================================

// The tokenizer handles byte-level BPE as used by GPT-2/CLIP.
// Data structures are plain C-compatible for opaque handle API.

// Map a byte to a unicode string for BPE
static void clip_build_byte_table(std::string byte_encoder[256]) {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 161; b <= 172; b++) bs.push_back(b);
    for (int b = 174; b <= 255; b++) bs.push_back(b);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool found = false;
        for (int x : bs) { if (x == b) { found = true; break; } }
        if (!found) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }
    for (size_t i = 0; i < bs.size() && i < cs.size(); i++) {
        int b = bs[i];
        int c = cs[i];
        char utf8[5] = {0};
        if (c < 128) {
            utf8[0] = (char)c; utf8[1] = 0;
        } else if (c < 2048) {
            utf8[0] = (char)(0xC0 | (c >> 6));
            utf8[1] = (char)(0x80 | (c & 0x3F));
            utf8[2] = 0;
        } else {
            utf8[0] = (char)(0xE0 | (c >> 12));
            utf8[1] = (char)(0x80 | ((c >> 6) & 0x3F));
            utf8[2] = (char)(0x80 | (c & 0x3F));
            utf8[3] = 0;
        }
        byte_encoder[b] = utf8;
    }
}

// CLIP special tokens
static const int CLIP_SOS_ID = 49406;
static const int CLIP_EOS_ID = 49407;
static const int CLIP_N_MAX = 77;

struct CLIPTokenizer {
    std::unordered_map<std::string, int> encoder;
    std::unordered_map<std::string, int> bpe_ranks;
    std::string byte_encoder[256];
    bool initialized;
};

static bool clip_load_json_vocab(const char* path,
                                  std::unordered_map<std::string, int>& encoder) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    f.close();

    // Parse flat JSON object: {"tok1": id1, "tok2": id2, ...}
    size_t i = 0;
    auto skip_ws = [&]() { while (i < content.size() && (content[i]==' '||content[i]=='\n'||content[i]=='\t'||content[i]=='\r')) i++; };
    auto expect = [&](char c) { skip_ws(); if (i < content.size() && content[i]==c) { i++; return true; } return false; };

    auto read_json_string = [&](std::string& out) {
        skip_ws();
        if (i >= content.size() || content[i] != '"') return false;
        i++; // skip opening quote
        out.clear();
        while (i < content.size() && content[i] != '"') {
            if (content[i] == '\\' && i+1 < content.size()) {
                i++;
                if (content[i] == 'n') out += '\n';
                else if (content[i] == 't') out += '\t';
                else if (content[i] == 'r') out += '\r';
                else if (content[i] == '\\') out += '\\';
                else if (content[i] == '"') out += '"';
                else if (content[i] == 'u') {
                    // Skip unicode escapes for simplicity
                    out += '?';
                }
                else out += content[i];
            } else {
                out += content[i];
            }
            i++;
        }
        if (i < content.size() && content[i] == '"') i++;
        return true;
    };

    auto read_json_int = [&](int& val) {
        skip_ws();
        if (i >= content.size() || (content[i] < '0' || content[i] > '9')) return false;
        val = 0;
        while (i < content.size() && content[i] >= '0' && content[i] <= '9') {
            val = val * 10 + (content[i] - '0');
            i++;
        }
        return true;
    };

    if (!expect('{')) return false;

    while (true) {
        skip_ws();
        if (expect('}')) break;
        std::string key;
        if (!read_json_string(key)) break;
        if (!expect(':')) break;
        int val = 0;
        if (!read_json_int(val)) break;
        encoder[key] = val;
        skip_ws();
        if (!expect(',')) {
            // Try closing brace
            skip_ws();
            if (i < content.size() && content[i] == '}') break;
        }
    }
    return !encoder.empty();
}

static bool clip_load_merges(const char* path,
                              std::unordered_map<std::string, int>& bpe_ranks) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    int rank = 0;
    while (std::getline(f, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        // Trim trailing whitespace
        while (!line.empty() && (line.back() == ' ' || line.back() == '\t' || line.back() == '\r'))
            line.pop_back();
        if (line.empty()) continue;
        bpe_ranks[line] = rank;
        rank++;
    }
    return rank > 0;
}

// Get all adjacent pairs of symbols in a word (represented as list of tokens)
static void clip_get_pairs(const std::vector<std::string>& word,
                            std::vector<std::pair<std::string,std::string>>& pairs) {
    pairs.clear();
    for (size_t i = 0; i + 1 < word.size(); i++) {
        pairs.push_back({word[i], word[i+1]});
    }
}

// Apply BPE merges to a single word (already byte-encoded as unicode chars)
static std::vector<std::string> clip_bpe(const std::string& token,
                                          const std::unordered_map<std::string, int>& bpe_ranks) {
    // Split into characters
    std::vector<std::string> word;
    for (size_t i = 0; i < token.size(); ) {
        unsigned char c = (unsigned char)token[i];
        if (c < 128) {
            word.push_back(token.substr(i, 1));
            i++;
        } else if ((c & 0xE0) == 0xC0) {
            word.push_back(token.substr(i, 2));
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            word.push_back(token.substr(i, 3));
            i += 3;
        } else {
            word.push_back(token.substr(i, 1));
            i++;
        }
    }

    if (word.size() == 1) return word;

    while (true) {
        // Find pair with lowest rank
        std::vector<std::pair<std::string,std::string>> pairs;
        clip_get_pairs(word, pairs);

        if (pairs.empty()) break;

        int best_rank = INT_MAX;
        size_t best_idx = 0;
        std::string best_pair;

        for (size_t i = 0; i < pairs.size(); i++) {
            std::string key = pairs[i].first + " " + pairs[i].second;
            auto it = bpe_ranks.find(key);
            if (it != bpe_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
                best_pair = key;
            }
        }

        if (best_rank == INT_MAX) break;  // No more merges

        // Merge the best pair
        std::vector<std::string> new_word;
        std::string merged = pairs[best_idx].first + pairs[best_idx].second;
        for (size_t i = 0; i < word.size(); i++) {
            if (i < word.size() - 1 &&
                word[i] == pairs[best_idx].first &&
                word[i+1] == pairs[best_idx].second) {
                new_word.push_back(merged);
                i++;
            } else {
                new_word.push_back(word[i]);
            }
        }
        word = new_word;
        if (word.size() == 1) break;
    }
    return word;
}

// Simple pre-tokenizer splitting text into tokens (CLIP/GPT-2 style patterns)
static void clip_pretokenize(const std::string& text,
                              std::vector<std::string>& tokens) {
    tokens.clear();
    size_t i = 0;
    while (i < text.size()) {
        // Skip whitespace
        if (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r') {
            i++;
            continue;
        }
        // Try special tokens
        if (text.substr(i, 13) == "<|startoftext|>") {
            tokens.push_back("<|startoftext|>");
            i += 13;
            continue;
        }
        if (text.substr(i, 11) == "<|endoftext|>") {
            tokens.push_back("<|endoftext|>");
            i += 11;
            continue;
        }
        // Try contractions
        if (i + 1 < text.size() && text[i] == '\'' && text[i+1] == 's') {
            tokens.push_back(text.substr(i, 2)); i += 2; continue;
        }
        if (i + 1 < text.size() && text[i] == '\'' && text[i+1] == 't') {
            tokens.push_back(text.substr(i, 2)); i += 2; continue;
        }
        if (i + 2 < text.size() && text.substr(i, 3) == "'re") {
            tokens.push_back("'re"); i += 3; continue;
        }
        if (i + 2 < text.size() && text.substr(i, 3) == "'ve") {
            tokens.push_back("'ve"); i += 3; continue;
        }
        if (i + 1 < text.size() && text[i] == '\'' && text[i+1] == 'm') {
            tokens.push_back(text.substr(i, 2)); i += 2; continue;
        }
        if (i + 2 < text.size() && text.substr(i, 3) == "'ll") {
            tokens.push_back("'ll"); i += 3; continue;
        }
        if (i + 1 < text.size() && text[i] == '\'' && text[i+1] == 'd') {
            tokens.push_back(text.substr(i, 2)); i += 2; continue;
        }
        // Word characters
        if (isalnum(text[i]) || text[i] == '_') {
            size_t start = i;
            while (i < text.size() && (isalnum(text[i]) || text[i] == '_')) i++;
            tokens.push_back(text.substr(start, i - start));
            continue;
        }
        // Punctuation
        if (strchr(",:;.!?\"\"()[]{}<>/-+*=@#$%^&|~`", text[i])) {
            tokens.push_back(text.substr(i, 1));
            i++;
            continue;
        }
        // Any other non-whitespace char
        tokens.push_back(text.substr(i, 1));
        i++;
    }
}

// Main encode function: text → vector of token IDs
static std::vector<int> clip_encode_text(CLIPTokenizer* tok, const std::string& text) {
    std::vector<int> ids;
    ids.push_back(CLIP_SOS_ID);

    std::vector<std::string> pretokens;
    // GPT-2 BPE convention: add leading space so "test" → " test" → token "test</w>"
    clip_pretokenize(" " + text, pretokens);

    for (const auto& token : pretokens) {
        // Check if token is directly in vocab
        auto it = tok->encoder.find(token);
        if (it != tok->encoder.end()) {
            ids.push_back(it->second);
            if ((int)ids.size() >= CLIP_N_MAX - 1) break;
            continue;
        }

        // Byte-encode and BPE
        std::string byte_str;
        for (unsigned char c : token) {
            byte_str += tok->byte_encoder[c];
        }

        std::vector<std::string> bpe_tokens = clip_bpe(byte_str, tok->bpe_ranks);
        for (const auto& bpe_tok : bpe_tokens) {
            auto it2 = tok->encoder.find(bpe_tok);
            if (it2 != tok->encoder.end()) {
                ids.push_back(it2->second);
                if ((int)ids.size() >= CLIP_N_MAX - 1) break;
            }
        }
        if ((int)ids.size() >= CLIP_N_MAX - 1) break;
    }

    ids.push_back(CLIP_EOS_ID);

    // Pad to CLIP_N_MAX with zeros
    while ((int)ids.size() < CLIP_N_MAX) {
        ids.push_back(0);
    }
    return ids;
}

// ============================================================
// Public API
// ============================================================

void* torch_std_clip_tokenizer_create(const char* vocab_path, const char* merges_path) {
    try {
        auto* tok = new CLIPTokenizer();
        tok->initialized = false;
        clip_build_byte_table(tok->byte_encoder);

        if (!clip_load_json_vocab(vocab_path, tok->encoder)) {
            std::cerr << "clip_tokenizer: failed to load vocab from " << vocab_path << std::endl;
            delete tok;
            return nullptr;
        }

        if (!clip_load_merges(merges_path, tok->bpe_ranks)) {
            std::cerr << "clip_tokenizer: failed to load merges from " << merges_path << std::endl;
            delete tok;
            return nullptr;
        }

        // Add special tokens if missing
        if (tok->encoder.find("<|startoftext|>") == tok->encoder.end())
            tok->encoder["<|startoftext|>"] = CLIP_SOS_ID;
        if (tok->encoder.find("<|endoftext|>") == tok->encoder.end())
            tok->encoder["<|endoftext|>"] = CLIP_EOS_ID;

        tok->initialized = true;
        return tok;
    } catch (const std::exception& e) {
        std::cerr << "clip_tokenizer_create error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_clip_tokenizer_encode(void* tokenizer, const char* text) {
    try {
        auto* tok = static_cast<CLIPTokenizer*>(tokenizer);
        if (!tok || !tok->initialized) return nullptr;

        std::vector<int> ids = clip_encode_text(tok, text);

        // Create int64 tensor [77]
        auto tensor = torch::empty({CLIP_N_MAX}, torch::kInt64);
        auto* data = tensor.data_ptr<int64_t>();
        for (int i = 0; i < CLIP_N_MAX; i++) {
            data[i] = ids[i];
        }
        return wrap(tensor);
    } catch (const std::exception& e) {
        std::cerr << "clip_tokenizer_encode error: " << e.what() << std::endl;
        return nullptr;
    }
}

void torch_std_clip_tokenizer_free(void* tokenizer) {
    delete static_cast<CLIPTokenizer*>(tokenizer);
}

// CLIP Text Encoder forward: convenience wrapper
// Accepts a loaded JIT module and token_id tensor, returns (1,77,768) embeddings
// If cast_to_float16 is nonzero, output is cast to float16
void* torch_std_clip_text_forward(void* clip_module, void* token_ids_tensor,
                                   int cast_to_float16) {
    try {
        auto* module = static_cast<torch::jit::Module*>(clip_module);
        auto& tokens = unwrap(token_ids_tensor);

        // CLIP expects (1, 77) int64
        at::Tensor input_tokens = tokens;
        if (input_tokens.dim() == 1) {
            input_tokens = input_tokens.unsqueeze(0);
        }
        if (input_tokens.scalar_type() != torch::kInt64) {
            input_tokens = input_tokens.to(torch::kInt64);
        }

        // Move input to same device as module
        if (module->parameters().size() > 0) {
            input_tokens = input_tokens.to(torch::kCUDA);
        }

        // Run JIT forward
        std::vector<torch::jit::IValue> inputs = {input_tokens};
        auto output = module->forward(inputs);

        at::Tensor result;
        if (output.isTuple()) {
            auto tuple = output.toTuple();
            result = tuple->elements()[0].toTensor();
        } else {
            result = output.toTensor();
        }

        // CLIP returns (1, 77, 512) for ViT-B/32 or (1, 77, 768) for ViT-L/14
        if (cast_to_float16) {
            result = result.to(torch::kFloat16);
        }

        return wrap(result.clone());
    } catch (const std::exception& e) {
        std::cerr << "clip_text_forward error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================
// GGUF Parser + Dequantization
// ============================================================

// GGUF tensor types
enum GGMLType {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
};

struct GGUFTensor {
    std::string name;
    std::vector<int64_t> shape;
    int type;      // GGMLType
    uint64_t offset; // offset in file
};

struct GGUFModel {
    std::vector<GGUFTensor> tensors;
    std::unordered_map<std::string, int> name_to_idx;
    FILE* file;
    uint64_t data_start; // offset where tensor data begins
    int version;
    bool valid;
};

static bool gguf_read_string(FILE* f, std::string& out) {
    uint64_t len;
    if (!gguf_read(f, &len)) return false;
    out.resize((size_t)len);
    if (len > 0 && fread(&out[0], 1, len, f) != len) return false;
    return true;
}

// Build gguf model index from file (reads header + tensor info, does NOT load weights)
static GGUFModel* gguf_model_create(const char* path) {
    GGUFModel* m = new GGUFModel();
    m->file = nullptr;
    m->valid = false;

    FILE* f = fopen(path, "rb");
    if (!f) { std::cerr << "gguf: cannot open " << path << std::endl; delete m; return nullptr; }

    // Read magic: "GGUF"
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4) != 0) {
        std::cerr << "gguf: invalid magic" << std::endl;
        fclose(f); delete m; return nullptr;
    }

    // Read version
    if (!gguf_read(f, &m->version)) { fclose(f); delete m; return nullptr; }
    if (m->version < 1 || m->version > 3) {
        std::cerr << "gguf: unsupported version " << m->version << std::endl;
        fclose(f); delete m; return nullptr;
    }

    // Read tensor_count + metadata_kv_count
    uint64_t tensor_count = 0, metadata_kv_count = 0;
    if (!gguf_read(f, &tensor_count) || !gguf_read(f, &metadata_kv_count)) {
        fclose(f); delete m; return nullptr;
    }

    // Skip metadata KV pairs (just read and discard)
    for (uint64_t i = 0; i < metadata_kv_count; i++) {
        std::string key;
        if (!gguf_read_string(f, key)) { fclose(f); delete m; return nullptr; }
        int32_t type;
        if (!gguf_read(f, &type)) { fclose(f); delete m; return nullptr; }
        // Skip value based on type
        switch (type) {
            case 0: { uint8_t v; gguf_read(f, &v); break; } // bool
            case 1: { uint8_t v; gguf_read(f, &v); break; } // uint8
            case 2: { int8_t v; gguf_read(f, &v); break; }  // int8
            case 3: { uint16_t v; gguf_read(f, &v); break; } // uint16
            case 4: { int16_t v; gguf_read(f, &v); break; }  // int16
            case 5: { uint32_t v; gguf_read(f, &v); break; } // uint32
            case 6: { int32_t v; gguf_read(f, &v); break; }   // int32
            case 7: { float v; gguf_read(f, &v); break; }     // float32
            case 8: { int64_t v; gguf_read(f, &v); break; }   // float64 (actually int64 for GGUF)
            case 9: { uint64_t v; gguf_read(f, &v); break; }  // uint64
            case 10: { // string
                uint64_t slen;
                if (gguf_read(f, &slen)) fseek(f, slen, SEEK_CUR);
                break;
            }
            case 11: { // array
                int32_t arr_type;
                uint64_t arr_len;
                if (gguf_read(f, &arr_type) && gguf_read(f, &arr_len)) {
                    for (uint64_t j = 0; j < arr_len; j++) {
                        switch (arr_type) {
                            case 0: case 1: { uint8_t v; gguf_read(f, &v); break; }
                            case 2: { int8_t v; gguf_read(f, &v); break; }
                            case 3: { uint16_t v; gguf_read(f, &v); break; }
                            case 4: { int16_t v; gguf_read(f, &v); break; }
                            case 5: { uint32_t v; gguf_read(f, &v); break; }
                            case 6: { int32_t v; gguf_read(f, &v); break; }
                            case 7: { float v; gguf_read(f, &v); break; }
                            case 8: { int64_t v; gguf_read(f, &v); break; }
                            case 9: { uint64_t v; gguf_read(f, &v); break; }
                            case 10: { std::string s; gguf_read_string(f, s); break; }
                        }
                    }
                }
                break;
            }
            default:
                // Unknown type, skip 4 bytes of data (GGUF has size prefix for arrays)
                break;
        }
    }

    // Read tensor info
    m->tensors.resize((size_t)tensor_count);
    for (uint64_t i = 0; i < tensor_count; i++) {
        auto& t = m->tensors[(size_t)i];
        if (!gguf_read_string(f, t.name)) { fclose(f); delete m; return nullptr; }
        // Read n_dimensions
        int32_t n_dims;
        if (!gguf_read(f, &n_dims)) { fclose(f); delete m; return nullptr; }
        // Read shape (in reverse order for GGUF)
        t.shape.resize((size_t)n_dims);
        for (int d = 0; d < n_dims; d++) {
            // GGUF stores shape reversed (last dim first)
            int64_t dim_val;
            if (!gguf_read(f, &dim_val)) { fclose(f); delete m; return nullptr; }
            t.shape[(size_t)d] = dim_val;
        }
        if (!gguf_read(f, &t.type)) { fclose(f); delete m; return nullptr; }
        if (!gguf_read(f, &t.offset)) { fclose(f); delete m; return nullptr; }
        m->name_to_idx[t.name] = (int)i;
    }

    // Data starts here
    m->data_start = ftell(f);
    m->file = f;
    m->valid = true;
    return m;
}

static int gguf_tensor_count(GGUFModel* m) { return (int)m->tensors.size(); }

static const char* gguf_tensor_name(GGUFModel* m, int idx) {
    if (idx < 0 || idx >= (int)m->tensors.size()) return "";
    return m->tensors[(size_t)idx].name.c_str();
}

// Get block size and element size for a quantization type
static void ggml_block_info(int type, int& block_size, int& block_bytes) {
    switch (type) {
        case GGML_TYPE_F32:  block_size = 1;  block_bytes = 4;  break;
        case GGML_TYPE_F16:  block_size = 1;  block_bytes = 2;  break;
        case GGML_TYPE_Q4_0: block_size = 32; block_bytes = 18; break; // 2 (scale) + 16 (4-bit data)
        case GGML_TYPE_Q4_1: block_size = 32; block_bytes = 20; break; // 2*2 (scales) + 16 (4-bit data)
        case GGML_TYPE_Q5_0: block_size = 32; block_bytes = 22; break; // 2 (scale) + 20 (5-bit data)
        case GGML_TYPE_Q5_1: block_size = 32; block_bytes = 24; break; // 2*2 (scales) + 20 (5-bit data)
        case GGML_TYPE_Q8_0: block_size = 32; block_bytes = 34; break; // 2 (scale) + 32 (8-bit data)
        default:             block_size = 1;  block_bytes = 4;  break;
    }
}

// Dequantize a single block
static void dequantize_block(const uint8_t* data, float* out, int type) {
    switch (type) {
        case GGML_TYPE_F32: {
            memcpy(out, data, 4);
            break;
        }
        case GGML_TYPE_F16: {
            // Convert fp16 to fp32
            uint16_t h;
            memcpy(&h, data, 2);
            // FP16: 1 sign, 5 exp, 10 mantissa
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp  = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x03FF;
            uint32_t f32;
            if (exp == 0) {
                // Subnormal
                if (mant == 0) f32 = sign << 31;
                else {
                    exp = 127 - 14;
                    while (!(mant & 0x0400)) { mant <<= 1; exp--; }
                    mant &= 0x03FF;
                    f32 = (sign << 31) | (exp << 23) | (mant << 13);
                }
            } else if (exp == 31) {
                f32 = (sign << 31) | 0x7F800000 | (mant << 13);
            } else {
                f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            }
            memcpy(out, &f32, 4);
            break;
        }
        case GGML_TYPE_Q4_0: {
            // Block: 2 bytes fp16 scale + 16 bytes 4-bit values (32 * 4b)
            // scale is at data[0..1]
            uint16_t scale_h;
            memcpy(&scale_h, data, 2);
            uint32_t sign = (scale_h >> 15) & 1;
            uint32_t exp  = (scale_h >> 10) & 0x1F;
            uint32_t mant = scale_h & 0x03FF;
            uint32_t scale_f32;
            if (exp == 0) {
                if (mant == 0) scale_f32 = sign << 31;
                else {
                    exp = 127 - 14;
                    while (!(mant & 0x0400)) { mant <<= 1; exp--; }
                    mant &= 0x03FF;
                    scale_f32 = (sign << 31) | (exp << 23) | (mant << 13);
                }
            } else if (exp == 31) {
                scale_f32 = (sign << 31) | 0x7F800000 | (mant << 13);
            } else {
                scale_f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            }
            float scale;
            memcpy(&scale, &scale_f32, 4);

            const uint8_t* q = data + 2;
            for (int i = 0; i < 16; i++) {
                uint8_t qv = q[i];
                out[2*i]     = ((int8_t)(qv & 0x0F) - 8) * scale;
                out[2*i + 1] = ((int8_t)(qv >> 4) - 8) * scale;
            }
            break;
        }
        case GGML_TYPE_Q4_1: {
            // Block: 2+2 bytes fp16 min/scale + 16 bytes 4-bit
            uint16_t min_h, scale_h;
            memcpy(&min_h, data, 2);
            memcpy(&scale_h, data+2, 2);
            // Convert fp16 to fp32
            auto fp16_to_f32 = [](uint16_t h) -> float {
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp  = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x03FF;
                uint32_t f32;
                if (exp == 0) {
                    if (mant == 0) { f32 = sign << 31; }
                    else {
                        exp = 127 - 14;
                        while (!(mant & 0x0400)) { mant <<= 1; exp--; }
                        mant &= 0x03FF;
                        f32 = (sign << 31) | (exp << 23) | (mant << 13);
                    }
                } else if (exp == 31) { f32 = (sign << 31) | 0x7F800000 | (mant << 13); }
                else { f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13); }
                float result; memcpy(&result, &f32, 4); return result;
            };
            float d = fp16_to_f32(scale_h);
            float m = fp16_to_f32(min_h);
            const uint8_t* q = data + 4;
            for (int i = 0; i < 16; i++) {
                uint8_t qv = q[i];
                out[2*i]     = (qv & 0x0F) * d + m;
                out[2*i + 1] = (qv >> 4) * d + m;
            }
            break;
        }
        case GGML_TYPE_Q5_0: {
            // Block: 2 bytes fp16 scale + 4 bytes high bits + 20 bytes 4-bit data = 26 bytes
            // Actually: 2 (scale) + 20 (5-bit packed) + 0 high-byte for old format
            // New format: scale(2) + high_4_bytes(4) + low_16_bytes(16) = 22
            uint16_t scale_h;
            memcpy(&scale_h, data, 2);
            uint32_t sign = (scale_h >> 15) & 1;
            uint32_t exp  = (scale_h >> 10) & 0x1F;
            uint32_t mant = scale_h & 0x03FF;
            uint32_t scale_f32;
            if (exp == 0) {
                if (mant == 0) scale_f32 = sign << 31;
                else { exp = 127-14; while (!(mant & 0x0400)) { mant <<= 1; exp--; } mant &= 0x03FF; scale_f32 = (sign<<31) | (exp<<23) | (mant<<13); }
            } else if (exp == 31) { scale_f32 = (sign<<31) | 0x7F800000 | (mant<<13); }
            else { scale_f32 = (sign<<31) | ((exp+112)<<23) | (mant<<13); }
            float scale; memcpy(&scale, &scale_f32, 4);

            // Q5_0 layout: 2 bytes scale + 4 bytes high bits + 16 bytes low bits
            uint32_t qh;
            memcpy(&qh, data + 2, 4);
            const uint8_t* ql = data + 6;
            for (int i = 0; i < 16; i++) {
                uint8_t low0 = ql[i] & 0x0F;
                uint8_t low1 = ql[i] >> 4;
                uint8_t high0 = ((qh >> (2*i)) & 1) ? 16 : 0;
                uint8_t high1 = ((qh >> (2*i+1)) & 1) ? 16 : 0;
                out[2*i]     = ((int8_t)(low0 | high0) - 16) * scale;
                out[2*i + 1] = ((int8_t)(low1 | high1) - 16) * scale;
            }
            break;
        }
        case GGML_TYPE_Q5_1: {
            uint16_t min_h, scale_h;
            memcpy(&min_h, data, 2);
            memcpy(&scale_h, data+2, 2);
            auto fp16_to_f32 = [](uint16_t h) -> float {
                uint32_t sign = (h>>15)&1, exp = (h>>10)&0x1F, mant = h&0x03FF;
                uint32_t f32;
                if (exp == 0) { if (mant==0) f32=sign<<31; else { exp=127-14; while(!(mant&0x0400)){mant<<=1;exp--;} mant&=0x03FF; f32=(sign<<31)|(exp<<23)|(mant<<13); } }
                else if (exp == 31) f32 = (sign<<31) | 0x7F800000 | (mant<<13);
                else f32 = (sign<<31) | ((exp+112)<<23) | (mant<<13);
                float r; memcpy(&r, &f32, 4); return r;
            };
            float d = fp16_to_f32(scale_h);
            float m = fp16_to_f32(min_h);
            uint32_t qh;
            memcpy(&qh, data + 4, 4);
            const uint8_t* ql = data + 8;
            for (int i = 0; i < 16; i++) {
                uint8_t low0 = ql[i] & 0x0F;
                uint8_t low1 = ql[i] >> 4;
                uint8_t high0 = ((qh >> (2*i)) & 1) ? 16 : 0;
                uint8_t high1 = ((qh >> (2*i+1)) & 1) ? 16 : 0;
                out[2*i]     = (low0 | high0) * d + m;
                out[2*i + 1] = (low1 | high1) * d + m;
            }
            break;
        }
        case GGML_TYPE_Q8_0: {
            // Block: 2 bytes fp16 scale + 32 bytes 8-bit data
            uint16_t scale_h;
            memcpy(&scale_h, data, 2);
            uint32_t sign = (scale_h>>15)&1, exp=(scale_h>>10)&0x1F, mant=scale_h&0x03FF;
            uint32_t sf32;
            if (exp==0) { if(mant==0) sf32=sign<<31; else{exp=127-14;while(!(mant&0x0400)){mant<<=1;exp--;}mant&=0x03FF;sf32=(sign<<31)|(exp<<23)|(mant<<13);} }
            else if(exp==31) sf32=(sign<<31)|0x7F800000|(mant<<13);
            else sf32=(sign<<31)|((exp+112)<<23)|(mant<<13);
            float scale; memcpy(&scale, &sf32, 4);
            const int8_t* q = (const int8_t*)(data + 2);
            for (int i = 0; i < 32; i++) {
                out[i] = q[i] * scale;
            }
            break;
        }
    }
}

// Load a specific tensor from GGUF file and dequantize to float32 tensor
static void* gguf_model_load_tensor(GGUFModel* m, int idx) {
    if (!m || !m->valid || idx < 0 || idx >= (int)m->tensors.size()) return nullptr;
    auto& t = m->tensors[(size_t)idx];

    // Compute total elements
    int64_t numel = 1;
    for (int64_t d : t.shape) numel *= d;

    // Compute size in file
    int block_size, block_bytes;
    ggml_block_info(t.type, block_size, block_bytes);
    int64_t n_blocks = (numel + block_size - 1) / block_size;

    // Read the raw data
    fseek(m->file, m->data_start + t.offset, SEEK_SET);
    int64_t raw_size = n_blocks * block_bytes;
    std::vector<uint8_t> raw(raw_size);
    if (fread(raw.data(), 1, raw_size, m->file) != (size_t)raw_size) {
        return nullptr;
    }

    // Build shape in torch order (reverse GGUF shape)
    // GGUF stores last dim first, torch expects first dim first
    int ndims = (int)t.shape.size();
    std::vector<int64_t> torch_shape((size_t)ndims);
    for (int i = 0; i < ndims; i++) {
        torch_shape[(size_t)i] = t.shape[(size_t)(ndims - 1 - i)];
    }

    // Create output tensor
    auto tensor = torch::empty(at::IntArrayRef(torch_shape.data(), ndims),
                               torch::TensorOptions().dtype(torch::kFloat32));
    float* out_data = tensor.data_ptr<float>();

    // Dequantize block by block
    int64_t offset = 0;
    for (int64_t b = 0; b < n_blocks; b++) {
        int block_elems = std::min(block_size, (int)(numel - offset));
        float out_block[32];
        dequantize_block(raw.data() + b * block_bytes, out_block, t.type);
        for (int i = 0; i < block_elems; i++) {
            out_data[offset + i] = out_block[i];
        }
        offset += block_elems;
    }

    return wrap(tensor);
}

static void gguf_model_free(GGUFModel* m) {
    if (m) {
        if (m->file) fclose(m->file);
        delete m;
    }
}

// ============================================================
// Public GGUF API
// ============================================================

void* torch_std_gguf_load(const char* path) {
    return gguf_model_create(path);
}

int torch_std_gguf_tensor_count(void* model) {
    if (!model) return 0;
    return gguf_tensor_count((GGUFModel*)model);
}

const char* torch_std_gguf_tensor_name(void* model, int idx) {
    if (!model) return "";
    return gguf_tensor_name((GGUFModel*)model, idx);
}

void* torch_std_gguf_load_tensor(void* model, int idx) {
    if (!model) return nullptr;
    return gguf_model_load_tensor((GGUFModel*)model, idx);
}

void* torch_std_gguf_load_tensor_by_name(void* model, const char* name) {
    if (!model) return nullptr;
    auto* m = (GGUFModel*)model;
    auto it = m->name_to_idx.find(name);
    if (it == m->name_to_idx.end()) return nullptr;
    return gguf_model_load_tensor(m, it->second);
}

void torch_std_gguf_free(void* model) {
    gguf_model_free((GGUFModel*)model);
}

// ============================================================
// SDXL UNet forward (named-weight based, works with safetensors/GGUF dicts)
// ============================================================

// Internal helpers
static at::Tensor sdxl_get_weight(const std::unordered_map<std::string, at::Tensor>& d,
                                   const std::string& n) {
    auto it = d.find(n);
    if (it == d.end()) return at::Tensor();
    return it->second;
}

static std::unordered_map<std::string, at::Tensor> st_to_map(void* safetensors_dict) {
    std::unordered_map<std::string, at::Tensor> m;
    auto* sd = static_cast<STDict*>(safetensors_dict);
    if (!sd) return m;
    static const char* prefixes[] = {"model.diffusion_model.", "model.", "first_stage_model.", "conditioner.embedders.0.", "conditioner.embedders.1.", ""};
    for (int i = 0; i < sd->count; i++) {
        auto& e = sd->entries[i];
        auto* t = static_cast<torch::Tensor*>(e.tensor);
        if (!t) continue;
        // Copy tensor by value (shallow refcounted copy — safe because STDict owns memory)
        m[e.name] = *t;
        // Also store with stripped prefixes
        for (int p = 0; prefixes[p][0]; p++) {
            size_t plen = strlen(prefixes[p]);
            if (strncmp(e.name, prefixes[p], plen) == 0) {
                m[e.name + plen] = *t;
                break;
            }
        }
    }
    return m;
}

static at::Tensor sdxl_linear(const at::Tensor& inp,
                               const std::unordered_map<std::string, at::Tensor>& d,
                               const std::string& p) {
    auto w = sdxl_get_weight(d, p + ".weight");
    auto b = sdxl_get_weight(d, p + ".bias");
    if (!w.defined()) throw std::runtime_error("SDXL missing " + p + ".weight");
    return b.defined() ? at::linear(inp, w, b) : at::linear(inp, w);
}

static at::Tensor sdxl_conv2d(const at::Tensor& inp,
                               const std::unordered_map<std::string, at::Tensor>& d,
                               const std::string& p, int stride, int pad) {
    auto w = sdxl_get_weight(d, p + ".weight");
    auto b = sdxl_get_weight(d, p + ".bias");
    if (!w.defined()) throw std::runtime_error("SDXL missing " + p + ".weight");
    int ph = (w.size(3)==1) ? 0 : pad;
    int pw = ph;
    return b.defined() ? at::conv2d(inp, w, b, {stride,stride}, {ph,pw})
                       : at::conv2d(inp, w, at::Tensor(), {stride,stride}, {ph,pw});
}

__attribute__((optimize("O0"))) static at::Tensor sdxl_gn(const at::Tensor& x,
                           const std::unordered_map<std::string, at::Tensor>& d,
                           const std::string& p, int g=32) {
    auto w = sdxl_get_weight(d, p + ".weight");
    auto b = sdxl_get_weight(d, p + ".bias");
    if (!w.defined()) return x;
    return at::group_norm(x, g, w, b, 1e-6);
}

// SDXL resblock in DDPM (ld) format: in_layers/emb_layers/out_layers/skip_connection
static at::Tensor sdxl_resblock_ld(const at::Tensor& x, const at::Tensor& te,
                                    const std::unordered_map<std::string, at::Tensor>& d,
                                    const std::string& p) {
    auto h = at::silu(sdxl_gn(x, d, p+".in_layers.0"));
    h = sdxl_conv2d(h, d, p+".in_layers.2", 1, 1);
    auto teo = sdxl_linear(at::silu(te), d, p+".emb_layers.1");
    h = h + teo.view({teo.size(0), teo.size(1), 1, 1});
    h = at::silu(sdxl_gn(h, d, p+".out_layers.0"));
    h = sdxl_conv2d(h, d, p+".out_layers.3", 1, 1);
    auto sw = sdxl_get_weight(d, p+".skip_connection.weight");
    if (sw.defined()) {
        try {
            auto sb = sdxl_get_weight(d, p+".skip_connection.bias");
            int64_t st_[] = {1,1}, pd_[] = {0,0};
            auto sc = sb.defined() ? at::conv2d(x, sw, sb, at::IntArrayRef(st_,2), at::IntArrayRef(pd_,2)) 
                                   : at::conv2d(x, sw, at::Tensor(), at::IntArrayRef(st_,2), at::IntArrayRef(pd_,2));
            h = h + sc;
        } catch (...) {}
        return h;
    }
    auto r = h + x;
    return r;
}

// Legacy resblock for standard SDXL format (kept for reference but unused with this model)
static at::Tensor sdxl_resblock(const at::Tensor& x, const at::Tensor& te,
                                 const std::unordered_map<std::string, at::Tensor>& d,
                                 const std::string& p) {
    auto h = at::silu(sdxl_gn(x, d, p+".norm1"));
    h = sdxl_conv2d(h, d, p+".conv1", 1, 1);
    auto teo = sdxl_linear(at::silu(te), d, p+".time_emb_proj");
    h = h + teo.view({teo.size(0), teo.size(1), 1, 1});
    h = at::silu(sdxl_gn(h, d, p+".norm2"));
    h = sdxl_conv2d(h, d, p+".conv2", 1, 1);
    auto sw = sdxl_get_weight(d, p+".conv_shortcut.weight");
    if (sw.defined()) {
        auto sb = sdxl_get_weight(d, p+".conv_shortcut.bias");
        int64_t st__[] = {1,1}, pd__[] = {0,0};
        h = h + (sb.defined() ? at::conv2d(x, sw, sb, at::IntArrayRef(st__,2), at::IntArrayRef(pd__,2)) : at::conv2d(x, sw, at::Tensor(), at::IntArrayRef(st__,2), at::IntArrayRef(pd__,2)));
    } else {
        h = h + x;
    }
    return h;
}

static at::Tensor sdxl_attn_layer(const at::Tensor& x, const at::Tensor& txt,
                                   const std::unordered_map<std::string, at::Tensor>& d,
                                   const std::string& p, int nh) {
    int ch = x.size(1), hd = ch / nh, B = x.size(0), N = x.size(2)*x.size(3);
    // Pre-norm self-attn
    auto hn = sdxl_gn(x, d, p+".norm1");
    auto sq = sdxl_linear(hn, d, p+".attn1.to_q").view({B,N,nh,hd}).permute({0,2,1,3}).reshape({B*nh,N,hd});
    auto sk = sdxl_linear(hn, d, p+".attn1.to_k").view({B,N,nh,hd}).permute({0,2,1,3}).reshape({B*nh,N,hd});
    auto sv = sdxl_linear(hn, d, p+".attn1.to_v").view({B,N,nh,hd}).permute({0,2,1,3}).reshape({B*nh,N,hd});
    auto sa = at::softmax(at::bmm(sq, sk.transpose(1,2)) / std::sqrt((double)hd), -1);
    auto so = at::bmm(sa, sv).view({B,nh,N,hd}).permute({0,2,1,3}).reshape({B,N,ch});
    so = sdxl_linear(so, d, p+".attn1.to_out.0");
    auto h = x + so;

    // Cross-attn
    auto hn2 = sdxl_gn(h, d, p+".norm2");
    auto cq = sdxl_linear(hn2, d, p+".attn2.to_q").view({B,N,nh,hd}).permute({0,2,1,3}).reshape({B*nh,N,hd});
    auto ck = sdxl_linear(txt, d, p+".attn2.to_k").view({B,-1,nh,hd}).permute({0,2,1,3}).reshape({B*nh,-1,hd});
    auto cv = sdxl_linear(txt, d, p+".attn2.to_v").view({B,-1,nh,hd}).permute({0,2,1,3}).reshape({B*nh,-1,hd});
    auto ca = at::softmax(at::bmm(cq, ck.transpose(1,2)) / std::sqrt((double)hd), -1);
    auto co = at::bmm(ca, cv).view({B,nh,N,hd}).permute({0,2,1,3}).reshape({B,N,ch});
    co = sdxl_linear(co, d, p+".attn2.to_out.0");
    h = h + co;

    // FFN (GeGLU)
    auto hn3 = sdxl_gn(h, d, p+".norm3");
    auto ff = sdxl_linear(hn3, d, p+".ff.net.0.proj");
    int half = ff.size(2)/2;
    auto ffg = ff.slice(2,0,half), ffv = ff.slice(2,half,2*half);
    ff = at::silu(ffg) * ffv;
    ff = sdxl_linear(ff, d, p+".ff.net.2");
    h = h + ff;
    return h;
}

static at::Tensor sdxl_down0(const at::Tensor& x, const at::Tensor& te, const at::Tensor& txt,
                              const std::unordered_map<std::string, at::Tensor>& d,
                              const std::string& p, int nl, int nh,
                              std::vector<at::Tensor>& skips) {
    auto h = x;
    for (int i = 0; i < nl; i++) {
        h = sdxl_resblock(h, te, d, p+".resnets."+std::to_string(i));
        h = sdxl_attn_layer(h, txt, d, p+".attentions."+std::to_string(i), nh);
        skips.push_back(h);
    }
    return h;
}

static std::vector<at::Tensor> sdxl_up0(const at::Tensor& x, const at::Tensor& te,
                                          const at::Tensor& txt,
                                          const std::unordered_map<std::string, at::Tensor>& d,
                                          const std::string& p, int nl, int nh,
                                          const std::vector<at::Tensor>& skips, int ss) {
    auto h = x;
    std::vector<at::Tensor> outs;
    for (int i = 0; i < nl; i++) {
        int si = ss + (nl-1-i);
        if (si < (int)skips.size()) h = at::cat({h, skips[si]}, 1);
        h = sdxl_resblock(h, te, d, p+".resnets."+std::to_string(i));
        h = sdxl_attn_layer(h, txt, d, p+".attentions."+std::to_string(i), nh);
        outs.push_back(h);
    }
    return outs;
}

// Forward declaration
static at::Tensor sdxl_up0_ld(const at::Tensor& x, const at::Tensor& te,
                               const at::Tensor& txt,
                               const std::unordered_map<std::string, at::Tensor>& d,
                               const std::string& p,
                               std::vector<at::Tensor>& skips, int nh);

#include <algorithm>
// Attention block (encoder/mid): proj_in → transformer_blocks → proj_out
// Does NOT call resblock - caller already did that with .0
static at::Tensor sdxl_attn_block(const at::Tensor& x, const at::Tensor& te,
                                   const at::Tensor& txt,
                                   const std::unordered_map<std::string, at::Tensor>& d,
                                   const std::string& p, int n_blocks) {
    auto hn = sdxl_gn(x, d, p+".1.norm");
    int ch = hn.size(1);
    hn = hn.permute({0,2,3,1}).reshape({-1, ch});
    hn = sdxl_linear(hn, d, p+".1.proj_in");
    ch = hn.size(1);
    int B = x.size(0), H = x.size(2), W = x.size(3);
    hn = hn.view({B, H, W, ch}).permute({0,3,1,2});
    int n_heads = ch / 64;
    int head_dim = 64;
    hn = hn.clone();

    for (int bi = 0; bi < n_blocks; bi++) {
        std::string tp = p+".1.transformer_blocks."+std::to_string(bi);
        int N = hn.size(2)*hn.size(3);
        auto hn_2d = hn.permute({0,2,3,1}).reshape({-1, ch});
        
        auto sdxl_ln = [&](const at::Tensor& x, const std::string& k) -> at::Tensor {
            auto w = sdxl_get_weight(d, k + ".weight");
            auto b = sdxl_get_weight(d, k + ".bias");
            if (!w.defined()) return x;
            auto f = x.permute({0,2,3,1}).reshape({-1, ch});
            auto out = at::layer_norm(f, {ch}, w, b, 1e-5);
            return out.reshape({B, H, W, ch}).permute({0,3,1,2});
        };

        double attn_scale = 1.0 / std::sqrt((double)head_dim);
        // Self-attention
        auto sa_hn = sdxl_ln(hn, tp+".norm1");
        auto n_2d = sa_hn.permute({0,2,3,1}).reshape({-1, ch});
        auto sq = sdxl_linear(n_2d, d, tp+".attn1.to_q").reshape({B,N,n_heads,head_dim}).transpose(1,2).contiguous();
        auto sk = sdxl_linear(n_2d, d, tp+".attn1.to_k").reshape({B,N,n_heads,head_dim}).transpose(1,2).contiguous();
        auto sv = sdxl_linear(n_2d, d, tp+".attn1.to_v").reshape({B,N,n_heads,head_dim}).transpose(1,2).contiguous();
        auto so = at::scaled_dot_product_attention(sq, sk, sv, {}, 0.0, false, attn_scale);
        so = so.transpose(1,2).contiguous().reshape({B,N,ch});
        so = sdxl_linear(so.reshape({-1, ch}), d, tp+".attn1.to_out.0").contiguous();
        hn = hn.clone() + so.view({B,H,W,ch}).permute({0,3,1,2});
        if (p.find("input_blocks.4") != std::string::npos) {
            char name[64];
            snprintf(name, sizeof(name), "ib4_b%d_after_self", bi);
            log_tensor(hn, name);
        }
        
        // Cross-attention
        auto ca_hn = sdxl_ln(hn, tp+".norm2");
        auto n2_2d = ca_hn.permute({0,2,3,1}).reshape({-1, ch});
        auto cq = sdxl_linear(n2_2d, d, tp+".attn2.to_q").reshape({B,N,n_heads,head_dim}).transpose(1,2).contiguous();
        auto ck = sdxl_linear(txt, d, tp+".attn2.to_k").reshape({B,-1,n_heads,head_dim}).transpose(1,2).contiguous();
        auto cv = sdxl_linear(txt, d, tp+".attn2.to_v").reshape({B,-1,n_heads,head_dim}).transpose(1,2).contiguous();
        auto co = at::scaled_dot_product_attention(cq, ck, cv, {}, 0.0, false, attn_scale);
        co = co.transpose(1,2).contiguous().reshape({B,N,ch});
        co = sdxl_linear(co.reshape({-1, ch}), d, tp+".attn2.to_out.0").contiguous();
        if (bi == 0 && p.find("input_blocks.4") != std::string::npos) {
            float cq_m = cq.abs().mean().item<float>(), ck_m = ck.abs().mean().item<float>();
            float co_m = co.abs().mean().item<float>(), hn_m = hn.abs().mean().item<float>();
            char buf[256];
            int nn = snprintf(buf, sizeof(buf), "ATTN_IB4 cq=%.4f ck=%.4f co=%.4f hn=%.4f\n", cq_m, ck_m, co_m, hn_m);
            write(2, buf, nn);
        }
        hn = hn + co.view({B,H,W,ch}).permute({0,3,1,2});
        if (p.find("input_blocks.4") != std::string::npos) {
            char name[64];
            snprintf(name, sizeof(name), "ib4_b%d_after_cross", bi);
            log_tensor(hn, name);
        }
        
        // FFN (GEGLU)
        auto ff_hn = sdxl_ln(hn, tp+".norm3");
        auto n3_2d = ff_hn.permute({0,2,3,1}).reshape({-1, ch});
        auto ff = sdxl_linear(n3_2d, d, tp+".ff.net.0.proj");
        int half = ff.size(1)/2;
        ff = ff.slice(1,0,half) * at::gelu(ff.slice(1,half,2*half));
        ff = sdxl_linear(ff, d, tp+".ff.net.2");
        hn = hn + ff.view({B,H,W,ch}).permute({0,3,1,2});
        if (p.find("input_blocks.4") != std::string::npos) {
            char name[64];
            snprintf(name, sizeof(name), "ib4_b%d_after_ffn", bi);
            log_tensor(hn, name);
        }
    }
    hn = hn.permute({0,2,3,1}).reshape({-1, ch});
    hn = sdxl_linear(hn, d, p+".1.proj_out");
    hn = hn.view({B, H, W, -1}).permute({0,3,1,2});
    // residual connection: return hn + original input (matching ComfyUI SpatialTransformer)
    return hn + x;
}

// LD-style up block: output_blocks.N has resblock at .0, optional attention at .1 (proj_in→tx_blocks→proj_out)
static at::Tensor sdxl_up0_ld(const at::Tensor& x, const at::Tensor& te,
                               const at::Tensor& txt,
                               const std::unordered_map<std::string, at::Tensor>& d,
                               const std::string& p,
                               std::vector<at::Tensor>& skips, int nh) {
    auto h = x;
    for (int si = (int)skips.size() - 1; si >= 0; si--) {
        auto& s = skips[si];
        if (s.dim() == 4 && s.size(2) == h.size(2) && s.size(3) == h.size(3)) {
            h = at::cat({h, s}, 1);
            skips.erase(skips.begin() + si);
            break;
        }
    }
    if (p.find("output_blocks.2") != std::string::npos) log_tensor(h, "ob2_cat");
    h = sdxl_resblock_ld(h, te, d, p+".0");
    if (p.find("output_blocks.2") != std::string::npos) log_tensor(h, "ob2_res");
    if (nh > 0) {
        h = sdxl_attn_block(h, te, txt, d, p, nh);
        if (p.find("output_blocks.2") != std::string::npos) log_tensor(h, "ob2_attn");
    }
    return h;
}

// Thread-local storage for SDXL CLIP pooled embeddings
static thread_local at::Tensor _sdxl_last_pooled;
static thread_local at::Tensor _sdxl_last_pooled_l;

// JIT-based SDXL UNet forward — drop-in replacement for the hand-written version
// Uses a traced UNet JIT module (exported from ComfyUI) for 1:1 compatibility
void* torch_std_sdxl_unet_jit_forward(
    void* jit_module,
    void* inp_ptr,
    void* timestep_ptr,
    void* text_emb_ptr,
    void* pooled_emb_ptr,
    double os_h, double os_w,
    double crop_t, double crop_l,
    double ts_h, double ts_w) {
    try {
        torch::NoGradGuard no_grad;
        auto* mod = static_cast<torch::jit::Module*>(jit_module);
        auto& inp_ref = unwrap(inp_ptr);
        auto dev = inp_ref.device();

        // Input latent
        at::Tensor x = inp_ref.to(torch::kHalf);

        // Convert sigma to timestep index (matching ComfyUI)
        static thread_local at::Tensor _cpu_log_sigmas = []() {
            double ls = 0.00085, le = 0.012;
            double sqrt_start = std::sqrt(ls), sqrt_end = std::sqrt(le);
            auto betas = at::linspace(sqrt_start, sqrt_end, 1000, torch::kFloat64);
            betas = betas * betas;
            auto alpha_bar = at::cumprod(1.0 - betas, 0);
            auto sigmas = at::sqrt((1.0 - alpha_bar) / alpha_bar.clamp_min(1e-8)).clamp_min(1e-8);
            return sigmas.log().to(torch::kFloat32).contiguous();
        }();
        auto ts_sigma = unwrap(timestep_ptr).to(torch::kFloat32).contiguous();
        float* ls_p = _cpu_log_sigmas.data_ptr<float>();
        float* s_p = ts_sigma.data_ptr<float>();
        float log_s = s_p ? std::log(std::max(*s_p, 1e-8f)) : 0.0f;
        int best_idx = 0;
        float best_dist = std::abs(log_s - ls_p[0]);
        for (int i = 1; i < 1000; i++) {
            float d = std::abs(log_s - ls_p[i]);
            if (d < best_dist) { best_dist = d; best_idx = i; }
        }
        // ComfyUI _apply_model: calculate_input scales by 1/sqrt(sigma^2+1)
        float sigma_v = s_p ? std::abs(*s_p) : 14.6f;
        float input_scale = 1.0f / std::sqrt(sigma_v * sigma_v + 1.0f);
        at::Tensor x_scaled = x * input_scale;

        auto t = at::full({1}, (float)best_idx, at::TensorOptions().dtype(torch::kFloat16).device(dev));

        // Text embedding
        at::Tensor context = unwrap(text_emb_ptr).to(dev).to(torch::kHalf);

        // Build y = pooled + time_ids
        at::Tensor pool = unwrap(pooled_emb_ptr).to(dev).to(torch::kHalf);
        auto ts_dev = dev;
        auto embed_256 = [&](double val) -> at::Tensor {
            auto t_val = at::full({1}, val, at::TensorOptions().dtype(torch::kFloat32).device(ts_dev));
            return timestep_embedding(t_val, 256).to(torch::kHalf);
        };
        auto time_ids = at::cat({
            embed_256(os_h), embed_256(os_w),
            embed_256(crop_t), embed_256(crop_l),
            embed_256(ts_h), embed_256(ts_w)
        }, 1);
        at::Tensor y = at::cat({pool, time_ids}, 1);

        // JIT forward
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(x_scaled);
        inputs.push_back(t);
        inputs.push_back(context);
        inputs.push_back(y);
        auto out = mod->forward(inputs);
        at::Tensor result = out.toTensor().to(torch::kFloat32);
        return wrap(result);
    } catch (const std::exception& e) {
        std::cerr << "sdxl_unet_jit_forward error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_sdxl_unet_forward(
    void* wdict_ptr,
    void* inp_ptr,
    void* timestep_ptr,
    void* text_emb_ptr,
    void* pooled_emb_ptr,
    double os_h, double os_w,
    double crop_t, double crop_l,
    double ts_h, double ts_w) {
    try {
        torch::NoGradGuard no_grad;
        
        if (!inp_ptr || !text_emb_ptr || !pooled_emb_ptr || !timestep_ptr) {
            std::cerr << "sdxl_unet_forward: null pointer argument\n";
            return nullptr;
        }
        auto& inp_ref = unwrap(inp_ptr);
        auto dev = inp_ref.device();
        // All local tensors go in scope block for deterministic cleanup
        at::Tensor out_fp32;
        {
        if (!wdict_ptr) { return nullptr; }
        auto& inp_ref = unwrap(inp_ptr);
        auto dev = inp_ref.device();
        auto d = st_to_map(wdict_ptr);
        for (auto& kv : d) kv.second = kv.second.to(dev).detach();
        auto inp = inp_ref.to(torch::kHalf);
        at::Tensor txt = unwrap(text_emb_ptr).to(dev).to(torch::kHalf);
        //txt.zero_();  // ZERO-OUT TEST: disabled for now
        at::Tensor pool = unwrap(pooled_emb_ptr).to(dev).to(torch::kHalf);
        // Convert sigma to discrete timestep index matching ComfyUI
        // Precompute sigmas on CPU once, then argmin(|log(sigma) - log_sigmas|) on CPU
        static thread_local at::Tensor _cpu_log_sigmas = []() {
            double ls = 0.00085, le = 0.012;
            // ComfyUI make_beta_schedule: linspace(sqrt(start), sqrt(end), n)**2
            auto sqrt_start = std::sqrt(ls), sqrt_end = std::sqrt(le);
            auto betas = at::linspace(sqrt_start, sqrt_end, 1000, torch::kFloat64);
            betas = betas * betas;
            auto alpha_bar = at::cumprod(1.0 - betas, 0);
            auto sigmas = at::sqrt((1.0 - alpha_bar) / alpha_bar.clamp_min(1e-8)).clamp_min(1e-8);
            return sigmas.log().to(torch::kFloat32).contiguous();
        }();
        auto ts_sigma = unwrap(timestep_ptr).to(torch::kFloat32).contiguous();  // CPU scalar
        float* ls_p = _cpu_log_sigmas.data_ptr<float>();
        float* s_p = ts_sigma.data_ptr<float>();
        float sigma_val = s_p ? *s_p : 0.0f;
        float log_s = s_p ? std::log(std::max(sigma_val, 1e-8f)) : 0.0f;
        int best_idx = 0;
        float best_dist = std::abs(log_s - ls_p[0]);
        for (int i = 1; i < 1000; i++) {
            float d = std::abs(log_s - ls_p[i]);
            if (d < best_dist) { best_dist = d; best_idx = i; }
        }
        static int unet_call_count = 0;
        if (unet_call_count < 8) {
            float inp_mean = inp.abs().mean().item<float>();
            float txt_mean = txt.abs().mean().item<float>();
            char buf[256];
            int n = snprintf(buf, sizeof(buf), "UNET_CALL=%d sigma=%.4f best_idx=%d inp=%.4f txt=%.4f\n",
                unet_call_count, sigma_val, best_idx, inp_mean, txt_mean);
            write(2, buf, n);
        }
        unet_call_count++;
        auto ts = at::full({1}, (float)best_idx, at::TensorOptions().dtype(torch::kFloat16).device(dev));
        // Timestep embed
        auto te = timestep_embedding(ts, 320).to(torch::kFloat16);
        te = at::silu(sdxl_linear(te, d, "time_embed.0"));
        te = sdxl_linear(te, d, "time_embed.2");

        // Added conditioning embed: CLIP-G pooled (1280) + time_ids (6*256=1536) = 2816
        auto ts_dev = ts.device();
        auto embed_256 = [&](double val) -> at::Tensor {
            auto t = at::full({1}, val, at::TensorOptions().dtype(torch::kFloat32).device(ts_dev));
            return timestep_embedding(t, 256).to(torch::kHalf);
        };
        auto time_ids = at::cat({
            embed_256(os_h), embed_256(os_w),
            embed_256(crop_t), embed_256(crop_l),
            embed_256(ts_h), embed_256(ts_w)
        }, 1);
        auto ae = at::cat({pool, time_ids}, 1);
        ae = at::silu(sdxl_linear(ae, d, "label_emb.0.0"));
        ae = sdxl_linear(ae, d, "label_emb.0.2");
        te = te + ae;

        // Conv in (input_blocks.0.0)
        auto h = sdxl_conv2d(inp, d, "input_blocks.0.0", 1, 1);

        log_tensor(h, "conv_in");

        // Encoder - push skip after each input block (matching Python's hs.append)
        std::vector<at::Tensor> sk;
        // input_blocks.0 is conv_in, result pushed
        sk.push_back(h);
        log_tensor(h, "ib0");
        // input_blocks.1: just resblock
        h = sdxl_resblock_ld(h, te, d, "input_blocks.1.0");
        sk.push_back(h);  log_tensor(h, "ib1");
        h = sdxl_resblock_ld(h, te, d, "input_blocks.2.0");
        sk.push_back(h);  log_tensor(h, "ib2");
        h = sdxl_conv2d(h, d, "input_blocks.3.0.op", 2, 1);
        sk.push_back(h);  log_tensor(h, "ib3");
        h = sdxl_resblock_ld(h, te, d, "input_blocks.4.0");
        h = sdxl_attn_block(h, te, txt, d, "input_blocks.4", 2);
        sk.push_back(h);  log_tensor(h, "ib4");
        h = sdxl_resblock_ld(h, te, d, "input_blocks.5.0");
        h = sdxl_attn_block(h, te, txt, d, "input_blocks.5", 2);
        sk.push_back(h);  log_tensor(h, "ib5");
        h = sdxl_conv2d(h, d, "input_blocks.6.0.op", 2, 1);
        sk.push_back(h);  log_tensor(h, "ib6");
        h = sdxl_resblock_ld(h, te, d, "input_blocks.7.0");
        h = sdxl_attn_block(h, te, txt, d, "input_blocks.7", 10);
        sk.push_back(h);  log_tensor(h, "ib7");
        h = sdxl_resblock_ld(h, te, d, "input_blocks.8.0");
        h = sdxl_attn_block(h, te, txt, d, "input_blocks.8", 10);
        sk.push_back(h);  log_tensor(h, "ib8");

        // Mid (output is NOT pushed to skips - matching ComfyUI)
        h = sdxl_resblock_ld(h, te, d, "middle_block.0");
        h = sdxl_attn_block(h, te, txt, d, "middle_block", 10);
        h = sdxl_resblock_ld(h, te, d, "middle_block.2");

        // Decoder
        h = sdxl_up0_ld(h, te, txt, d, "output_blocks.0", sk, 10); log_tensor(h, "ob0");
        h = sdxl_up0_ld(h, te, txt, d, "output_blocks.1", sk, 10); log_tensor(h, "ob1");
        h = sdxl_up0_ld(h, te, txt, d, "output_blocks.2", sk, 10); log_tensor(h, "ob2_pre");
        if (h.dim() == 4) {
            h = at::upsample_nearest2d(h, {h.size(2)*2, h.size(3)*2});
            h = sdxl_conv2d(h, d, "output_blocks.2.2.conv", 1, 1);
            log_tensor(h, "ob2_up");
        }
        h = sdxl_up0_ld(h, te, txt, d, "output_blocks.3", sk, 2); log_tensor(h, "ob3");
        h = sdxl_up0_ld(h, te, txt, d, "output_blocks.4", sk, 2); log_tensor(h, "ob4");
        h = sdxl_up0_ld(h, te, txt, d, "output_blocks.5", sk, 2); log_tensor(h, "ob5_pre");
        if (h.dim() == 4) {
            h = at::upsample_nearest2d(h, {h.size(2)*2, h.size(3)*2});
            h = sdxl_conv2d(h, d, "output_blocks.5.2.conv", 1, 1);
            log_tensor(h, "ob5_up");
        }
        h = sdxl_up0_ld(h, te, txt, d, "output_blocks.6", sk, 0); log_tensor(h, "ob6");
        h = sdxl_up0_ld(h, te, txt, d, "output_blocks.7", sk, 0); log_tensor(h, "ob7");
        h = sdxl_up0_ld(h, te, txt, d, "output_blocks.8", sk, 0); log_tensor(h, "ob8");

        // Out
        log_tensor(h, "pre_out");
        h = at::silu(sdxl_gn(h, d, "out.0"));
        log_tensor(h, "post_gn");
        h = sdxl_conv2d(h, d, "out.2", 1, 1);
        log_tensor(h, "out");
        out_fp32 = h.to(torch::kFloat32);
        {
            static int eps_call = 0;
            if (eps_call < 8) {
                float m = out_fp32.mean().item<float>();
                float s = out_fp32.std().item<float>();
                char buf[128];
                int n = snprintf(buf, sizeof(buf), "EPS_OUT call=%d mean=%.4f std=%.4f abs_mean=%.4f\n", eps_call, m, s, out_fp32.abs().mean().item<float>());
                write(2, buf, n);
                eps_call++;
            }
        }
        }  // end scope: all intermediate tensors freed
        return wrap(out_fp32);
    } catch (const std::exception& e) {
        std::cerr << "sdxl_unet_forward error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================
// SDXL Dual CLIP: run two CLIP JIT modules, concat outputs
// ============================================================
// clip_module1: ViT-L/14 → (1,77,768)
// clip_module2: ViT-bigG → (1,77,1280)
// Returns a dict-like struct with "text_emb" and "pooled" tensors.
// For simplicity, we encode both with the SAME tokens (77-token prompt).
// (Pooled output extracted from second CLIP's pooler_output or [0] token)

void* torch_std_sdxl_dual_clip(void* clip1, void* clip2, void* token_ids) {
    try {
        auto* mod1 = static_cast<torch::jit::Module*>(clip1);
        auto* mod2 = static_cast<torch::jit::Module*>(clip2);
        auto& tokens = unwrap(token_ids);

        at::Tensor tokens_input = tokens;
        if (tokens_input.dim() == 1) tokens_input = tokens_input.unsqueeze(0);
        if (tokens_input.scalar_type() != torch::kInt64) tokens_input = tokens_input.to(torch::kInt64);
        // Move tokens to GPU (models are on cuda:0)
        tokens_input = tokens_input.to(torch::kCUDA);

        // Run first CLIP
        auto out1 = mod1->forward({tokens_input});
        at::Tensor emb1, pooled_l;
        if (out1.isTuple()) {
            auto tup = out1.toTuple()->elements();
            emb1 = tup[0].toTensor();
            if (tup.size() > 1 && tup[1].isTensor()) {
                pooled_l = tup[1].toTensor(); // (1,768)
            }
        } else {
            emb1 = out1.toTensor();
        }
        // Function to extract pooled embedding from EOS token position
        auto extract_pooled = [](const at::Tensor& hidden_states, const at::Tensor& token_ids) -> at::Tensor {
            // Find first EOS token (49407) position
            auto eos_mask = (token_ids == 49407).to(torch::kInt64);
            auto eos_pos = at::argmax(eos_mask, 1).unsqueeze(1).unsqueeze(2);
            eos_pos = eos_pos.expand({1, 1, hidden_states.size(2)});
            return at::gather(hidden_states, 1, eos_pos).squeeze(1);
        };

        if (!pooled_l.defined()) {
            pooled_l = extract_pooled(emb1, tokens_input);
        }
        // (1,77,768)

        // Run second CLIP
        auto out2 = mod2->forward({tokens_input});
        at::Tensor emb2, pooled;
        if (out2.isTuple()) {
            auto tup = out2.toTuple()->elements();
            emb2 = tup[0].toTensor();
            if (tup.size() > 1 && tup[1].isTensor()) {
                pooled = tup[1].toTensor(); // (1,1280)
            }
        } else {
            emb2 = out2.toTensor();
        }

        if (!pooled.defined()) {
            pooled = extract_pooled(emb2, tokens_input);
        }

        // Store pooled_emb globally for retrieval
        _sdxl_last_pooled = pooled;
        _sdxl_last_pooled_l = pooled_l;

        // Concatenate text embeddings: (1,77,2048)
        auto text_emb = at::cat({emb1, emb2}, 2);
        return wrap(text_emb);
    } catch (const std::exception& e) {
        std::cerr << "sdxl_dual_clip error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_sdxl_get_pooled() {
    return wrap(_sdxl_last_pooled);
}

void* torch_std_sdxl_get_pooled_l() {
    return wrap(_sdxl_last_pooled_l);
}

// ============================================================
// Flow Matching scheduler (for FLUX / SD3)
// ============================================================
// Simple flow matching: sigmoid schedule.
// At timestep t, noisy_x = (1 - t) * x_0 + t * noise

void* torch_std_fm_sigmas(int steps, double sigma_min, double sigma_max) {
    try {
        auto sigmas = torch::linspace(1.0, 0.0, steps + 1, torch::kFloat64);
        // fliplr: sigmas[0] = 1 (noise), sigmas[N] = 0 (clean)
        sigmas = sigmas.flip(0);
        return wrap(sigmas.to(torch::kFloat32));
    } catch (const std::exception& e) {
        std::cerr << "fm_sigmas error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================
// 规范化层
// ============================================================
void* torch_std_layer_norm(void* input, void* weight, void* bias, double eps) {
    try {
        auto& x = unwrap(input);
        std::optional<at::Tensor> w = weight ? std::optional<at::Tensor>(unwrap(weight)) : std::nullopt;
        std::optional<at::Tensor> b = bias ? std::optional<at::Tensor>(unwrap(bias)) : std::nullopt;
        return wrap(torch::layer_norm(x, {x.size(-1)}, w, b, eps, false));
    } catch (const std::exception& e) {
        std::cerr << "torch_std_layer_norm error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_rms_norm(void* input, void* weight, double eps) {
    try {
        // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        auto& x = unwrap(input);
        auto x_sq_mean = x.mul(x).mean(-1, true); // (..., 1)
        auto rms = x_sq_mean.add(eps).sqrt();      // (..., 1)
        auto out = x.div(rms);
        if (weight) {
            out = out.mul(unwrap(weight));
        }
        return wrap(out);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_rms_norm error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_group_norm(void* input, void* weight, void* bias, int64_t num_groups, double eps) {
    try {
        auto& x = unwrap(input);
        std::optional<at::Tensor> w = weight ? std::optional<at::Tensor>(unwrap(weight)) : std::nullopt;
        std::optional<at::Tensor> b = bias ? std::optional<at::Tensor>(unwrap(bias)) : std::nullopt;
        return wrap(torch::group_norm(x, num_groups, w, b, eps));
    } catch (const std::exception& e) {
        std::cerr << "torch_std_group_norm error: " << e.what() << std::endl;
        return nullptr;
    }
}

// Flow matching step: given noise_pred (velocity), compute denoised
// x_t[x] = x_t + sigma_step * velocity
void* torch_std_fm_step(void* velocity, void* x_t, double dt) {
    try {
        auto& v = unwrap(velocity);
        auto& xt = unwrap(x_t);
        return wrap(xt + dt * v);
    } catch (const std::exception& e) {
        std::cerr << "fm_step error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================
// FLUX / SD3 MMDiT (Joint text+image transformer)
// ============================================================
// FLUX architecture:
// - Input: image latent (B, C, H, W) + text tokens (B, L, D)
// - Embed to tokens: img → patch_embed, txt → text_embed
// - RoPE positions for both
// - N joint transformer blocks (each has img→img, txt→txt, cross)
// - Modulation (adaLN-like) per block
// - Output: img tokens → unpatch → latent

static at::Tensor flux_rope(int h, int w, int dim, const at::Tensor& device_tensor) {
    // Simple 2D RoPE: compute frequencies for h and w positions
    auto dev = device_tensor.device();
    auto dt = device_tensor.dtype();
    int half = dim / 4;
    auto freqs = at::exp(-at::arange(0, half, at::TensorOptions().dtype(torch::kFloat32).device(dev)) *
                         std::log(10000.0) / half);
    // h positions
    auto pos_h = at::arange(0, h, at::TensorOptions().dtype(torch::kFloat32).device(dev)).unsqueeze(1);
    auto ang_h = pos_h * freqs.unsqueeze(0); // (h, half)
    // w positions
    auto pos_w = at::arange(0, w, at::TensorOptions().dtype(torch::kFloat32).device(dev)).unsqueeze(1);
    auto ang_w = pos_w * freqs.unsqueeze(0); // (w, half)
    // Combine: (h*w, half*2) with sin/cos pairs
    std::vector<at::Tensor> ang_vec = {
        ang_h.repeat_interleave(w, 0),
        ang_w.repeat({h, 1})
    };
    auto angles = at::cat(ang_vec, 1); // (h*w, half*2)
    auto cos_emb = at::cos(angles);
    auto sin_emb = at::sin(angles);
    std::vector<at::Tensor> stack_vec = {cos_emb, sin_emb};
    return at::stack(stack_vec, 0); // (2, h*w, half*2)
}

static void flux_apply_rope(at::Tensor& q, at::Tensor& k, const at::Tensor& rope) {
    int B = q.size(0), nh = q.size(1), N = q.size(2), hd = q.size(3);
    int half = rope.size(2);
    // Split q into two halves
    auto q1 = q.slice(3, 0, half);
    auto q2 = q.slice(3, half, 2*half);
    auto cos_emb = rope[0].unsqueeze(0).unsqueeze(1); // (1,1,N,half)
    auto sin_emb = rope[1].unsqueeze(0).unsqueeze(1);
    // Apply rotation
    auto q_rot = at::cat({q1 * cos_emb - q2 * sin_emb, q1 * sin_emb + q2 * cos_emb}, 3);
    // Same for k
    auto k1 = k.slice(3, 0, half);
    auto k2 = k.slice(3, half, 2*half);
    auto k_rot = at::cat({k1 * cos_emb - k2 * sin_emb, k1 * sin_emb + k2 * cos_emb}, 3);
    q = q_rot;
    k = k_rot;
}

// Single MMDiT block: joint text+image attention + modulation + MLP
static std::pair<at::Tensor, at::Tensor> flux_block(
    const at::Tensor& img, const at::Tensor& txt,
    const at::Tensor& timestep_emb, // (B, D) for modulation
    const at::Tensor& rope,
    const std::unordered_map<std::string, at::Tensor>& d,
    const std::string& prefix, int n_heads_img, int n_heads_txt, int head_dim) {

    int B = img.size(0), N_img = img.size(1), D_img = img.size(2);
    int N_txt = txt.size(1), D_txt = txt.size(2);

    // Modulation (adaLN): compute scale/shift for img and txt
    auto mod_hidden = sdxl_linear(timestep_emb, d, prefix + ".modulation.0");
    mod_hidden = at::silu(mod_hidden);
    auto mod_params = sdxl_linear(mod_hidden, d, prefix + ".modulation.2"); // (B, D_img*6 + D_txt*6)

    // Split modulation params
    int p = 0;
    auto img_mod1 = mod_params.slice(1, p, p+D_img*2); p += D_img*2;
    auto img_mod2 = mod_params.slice(1, p, p+D_img*2); p += D_img*2;
    auto img_mod3 = mod_params.slice(1, p, p+D_img*2); p += D_img*2;
    auto txt_mod1 = mod_params.slice(1, p, p+D_txt*2); p += D_txt*2;
    auto txt_mod2 = mod_params.slice(1, p, p+D_txt*2); p += D_txt*2;
    auto txt_mod3 = mod_params.slice(1, p, p+D_txt*2);

    // Split into scale/shift pairs
    auto split_mod = [](const at::Tensor& m) {
        int half = m.size(1) / 2;
        return std::make_pair(m.slice(1, 0, half), m.slice(1, half, 2*half));
    };
    auto [img_scale1, img_shift1] = split_mod(img_mod1);
    auto [img_scale2, img_shift2] = split_mod(img_mod2);
    auto [img_scale3, img_shift3] = split_mod(img_mod3);
    auto [txt_scale1, txt_shift1] = split_mod(txt_mod1);
    auto [txt_scale2, txt_shift2] = split_mod(txt_mod2);
    auto [txt_scale3, txt_shift3] = split_mod(txt_mod3);

    // Apply modulation to input
    auto img_norm = at::layer_norm(img,
        {D_img},
        sdxl_get_weight(d, prefix+".norm1_query.weight"),
        sdxl_get_weight(d, prefix+".norm1_query.bias"),
        1e-6);
    img_norm = img_norm * (1 + img_scale1) + img_shift1;
    auto txt_norm = at::layer_norm(txt,
        {D_txt},
        sdxl_get_weight(d, prefix+".norm1_add.weight"),
        sdxl_get_weight(d, prefix+".norm1_add.bias"),
        1e-6);
    txt_norm = txt_norm * (1 + txt_scale1) + txt_shift1;

    // Joint QKV projection
    // FLUX uses single QKV for joint space: img Q, txt KV, and vice versa
    auto qkv_img = sdxl_linear(img_norm, d, prefix+".attn.qkv"); // (B,N_img,3*D_img)
    auto qkv_txt = sdxl_linear(txt_norm, d, prefix+".attn.qkv"); // (B,N_txt,3*D_txt)

    auto q_img = qkv_img.slice(2, 0, D_img).view({B,N_img,n_heads_img,head_dim}).permute({0,2,1,3});
    auto k_img = qkv_img.slice(2, D_img, 2*D_img).view({B,N_img,n_heads_img,head_dim}).permute({0,2,1,3});
    auto v_img = qkv_img.slice(2, 2*D_img, 3*D_img).view({B,N_img,n_heads_img,head_dim}).permute({0,2,1,3});
    auto q_txt = qkv_txt.slice(2, 0, D_txt).view({B,N_txt,n_heads_txt,head_dim}).permute({0,2,1,3});
    auto k_txt = qkv_txt.slice(2, D_txt, 2*D_txt).view({B,N_txt,n_heads_txt,head_dim}).permute({0,2,1,3});
    auto v_txt = qkv_txt.slice(2, 2*D_txt, 3*D_txt).view({B,N_txt,n_heads_txt,head_dim}).permute({0,2,1,3});

    // Apply RoPE
    flux_apply_rope(q_img, k_img, rope);
    // For FLUX, text doesn't get RoPE typically

    // Concatenate for joint attention
    auto q_joint = at::cat({q_img, q_txt}, 2);
    auto k_joint = at::cat({k_img, k_txt}, 2);
    auto v_joint = at::cat({v_img, v_txt}, 2);

    // Attention
    auto score = at::bmm(q_joint.reshape({B*(n_heads_img+n_heads_txt), N_img+N_txt, head_dim}),
                          k_joint.reshape({B*(n_heads_img+n_heads_txt), N_img+N_txt, head_dim})
                          .transpose(1,2)) / std::sqrt((double)head_dim);
    // Causal mask or full attention
    auto attn = at::softmax(score, -1);
    auto out = at::bmm(attn, v_joint.reshape({B*(n_heads_img+n_heads_txt), N_img+N_txt, head_dim}));
    out = out.reshape({B, n_heads_img+n_heads_txt, N_img+N_txt, head_dim});

    // Split back
    auto out_img = out.slice(1, 0, n_heads_img).permute({0,2,1,3}).reshape({B,N_img,D_img});
    auto out_txt = out.slice(1, n_heads_img, n_heads_img+n_heads_txt).permute({0,2,1,3}).reshape({B,N_txt,D_txt});

    // Output projection
    out_img = sdxl_linear(out_img, d, prefix+".attn.proj");
    out_txt = sdxl_linear(out_txt, d, prefix+".attn.proj");

    // Residual with modulation
    auto h_img = img + out_img * img_scale2.unsqueeze(1);
    auto h_txt = txt + out_txt * txt_scale2.unsqueeze(1);

    // MLP for img
    auto img_mlp_norm = at::layer_norm(h_img, {D_img},
        sdxl_get_weight(d, prefix+".norm2.weight"),
        sdxl_get_weight(d, prefix+".norm2.bias"), 1e-6);
    img_mlp_norm = img_mlp_norm * (1 + img_scale3) + img_shift3;
    auto img_mlp = sdxl_linear(img_mlp_norm, d, prefix+".mlp.fc1");
    // GeGLU
    int half_f = img_mlp.size(2) / 2;
    auto img_mlp_gate = at::silu(img_mlp.slice(2, 0, half_f));
    auto img_mlp_val = img_mlp.slice(2, half_f, 2*half_f);
    img_mlp = img_mlp_gate * img_mlp_val;
    img_mlp = sdxl_linear(img_mlp, d, prefix+".mlp.fc2");
    h_img = h_img + img_mlp * img_scale3.unsqueeze(1);

    // MLP for txt
    auto txt_mlp_norm = at::layer_norm(h_txt, {D_txt},
        sdxl_get_weight(d, prefix+".norm2_add.weight"),
        sdxl_get_weight(d, prefix+".norm2_add.bias"), 1e-6);
    txt_mlp_norm = txt_mlp_norm * (1 + txt_scale3) + txt_shift3;
    auto txt_mlp = sdxl_linear(txt_mlp_norm, d, prefix+".mlp.fc1");
    auto txt_mlp_gate = at::silu(txt_mlp.slice(2, 0, half_f));
    auto txt_mlp_val = txt_mlp.slice(2, half_f, 2*half_f);
    txt_mlp = txt_mlp_gate * txt_mlp_val;
    txt_mlp = sdxl_linear(txt_mlp, d, prefix+".mlp.fc2");
    h_txt = h_txt + txt_mlp * txt_scale3.unsqueeze(1);

    return {h_img, h_txt};
}

// FLUX EmbedND: multi-axis RoPE position embedding
// ids: (..., n_axes) positions (e.g., (B*H*W, 3) for (t, h, w))
// dim: head_dim for attention
// theta: base frequency (typically 10000)
// axes_dim: array of per-axis dims (e.g., {64, 64, 64})
// n_axes: number of axes
void* torch_std_flux_embed_nd(void* ids, int dim, double theta, int64_t* axes_dim, int n_axes) {
    try {
        auto& ids_t = unwrap(ids);
        auto N = ids_t.size(0);
        auto rank = ids_t.size(1);
        std::vector<at::Tensor> ropes;
        int total_half = 0;
        for (int i = 0; i < n_axes && i < (int)rank; i++) {
            int axis_half = axes_dim[i] / 2;
            total_half += axis_half;
            
            // Extract axis i: ids[:, i]
            auto pos = ids_t.select(1, i); // (N,)
            
            // RoPE frequencies: omega_k = 1.0 / (theta ^ (2k/dim))
            // Using the standard RoPE: freq = theta^(-2k/dim)
            auto k = torch::arange(0, axis_half, torch::kFloat32).to(pos.device());
            auto freqs = torch::pow(torch::tensor(theta, torch::kFloat32), -2.0 * k / dim);
            
            // outer product: pos[N] × freqs[half]
            auto args = pos.unsqueeze(1) * freqs.unsqueeze(0); // (N, half)
            
            // Build 2x2 rotation: [[cos, -sin], [sin, cos]]
            auto c = args.cos();
            auto s = args.sin();
            
            // Stack into (N, half, 2, 2)
            auto neg_s = -s;
            auto row0 = torch::stack({c, neg_s}, -1); // (N, half, 2)
            auto row1 = torch::stack({s, c}, -1);     // (N, half, 2)
            auto rot = torch::stack({row0, row1}, -2); // (N, half, 2, 2)
            ropes.push_back(rot);
        }
        
        // Cat along half dim: (N, sum_half, 2, 2) -> unsqueeze(1) -> (1, N, sum_half, 2, 2)
        auto result = torch::cat(ropes, 1); // (N, total_half, 2, 2)
        result = result.unsqueeze(0); // (1, N, total_half, 2, 2)
        return wrap(result);
    } catch (const std::exception& e) {
        std::cerr << "torch_std_flux_embed_nd error: " << e.what() << std::endl;
        return nullptr;
    }
}

// Main FLUX forward
// FLUX config: img=64x64 latents, txt=512 T5 tokens, 24-38 blocks
void* torch_std_flux_forward(
    void* wdict_ptr,        // dict of weights
    void* img_ptr,          // (B, C, H, W) latent (float16)
    void* txt_ptr,          // (B, L, D) T5 text embeddings (float16)
    void* timestep_ptr,     // (B,) float timesteps (0..1 for flow matching)
    void* img_pos_ptr,      // (H*W, D) optional img positional embedding
    double guidance_scale,  // CFG scale
    int n_blocks,           // number of transformer blocks (24 for FLUX.1-schnell)
    int n_heads_img,        // images heads per block
    int n_heads_txt,        // text heads per block
    int head_dim) {         // dim per head
    try {
        auto d = *static_cast<std::unordered_map<std::string, at::Tensor>*>(wdict_ptr);
        auto& img = unwrap(img_ptr);
        auto& txt = unwrap(txt_ptr);
        auto& ts = unwrap(timestep_ptr);

        auto dev = img.device();

        // Double the batch for CFG
        bool do_cfg = (guidance_scale > 1.0 && guidance_scale < 2.0);
        // FLUX handles CFG via conditioning

        // Patch embed: conv2d latent → sequence
        // FLUX uses 2x2 patch, so 64x64 → 32x32 patches → 1024 tokens
        int H = img.size(2), W = img.size(3);
        auto h = sdxl_conv2d(img, d, "img_in", 2, 0); // stride=2, no pad
        // (B, D_img_patch, H/2, W/2)
        int D = h.size(1);
        h = h.permute({0,2,3,1}).reshape({img.size(0), -1, D}); // (B, N_img, D)

        // Add positional embedding
        if (img_pos_ptr) {
            auto& pos = unwrap(img_pos_ptr);
            h = h + pos.unsqueeze(0);
        } else {
            // Simple learned pos embed
            auto pos_w = sdxl_get_weight(d, "pos_embed");
            if (pos_w.defined()) {
                h = h + pos_w.unsqueeze(0);
            }
        }

        // Text embedding projection
        auto txt_h = sdxl_linear(txt, d, "txt_in");

        // Time embedding
        auto t = ts.to(torch::kFloat32);
        auto te = timestep_embedding(t, D); // (B, D)
        te = sdxl_linear(te, d, "time_in.0");
        te = at::silu(te);
        te = sdxl_linear(te, d, "time_in.2");
        // Add guidance embedding
        auto g_emb = at::full({(int)t.size(0), 1}, guidance_scale,
                              at::TensorOptions().dtype(torch::kFloat32).device(dev));
        auto ge = timestep_embedding(g_emb, D);
        ge = sdxl_linear(ge, d, "guidance_in.0");
        ge = at::silu(ge);
        ge = sdxl_linear(ge, d, "guidance_in.2");
        te = te + ge;

        // Also a vector embedding (like SDXL's add_embed)
        auto vec_in = sdxl_get_weight(d, "vector_in.weight");
        at::Tensor vec;
        if (vec_in.defined()) {
            vec = sdxl_linear(te, d, "vector_in");
        }

        // RoPE for image positions
        int h_patches = H / 2, w_patches = W / 2;
        auto rope = flux_rope(h_patches, w_patches, head_dim, img);

        // Transformer blocks
        for (int i = 0; i < n_blocks; i++) {
            std::string bp = "transformer_blocks." + std::to_string(i);
            auto [new_img, new_txt] = flux_block(h, txt_h, te, rope, d, bp,
                                                   n_heads_img, n_heads_txt, head_dim);
            h = new_img;
            txt_h = new_txt;
        }

        // Output projection
        h = sdxl_linear(h, d, "proj_out");

        // Unpatch: (B, N_img, D) → (B, D, H/2, W/2)
        h = h.reshape({img.size(0), h_patches, w_patches, D}).permute({0,3,1,2});

        return wrap(h);

    } catch (const std::exception& e) {
        std::cerr << "flux_forward error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================
// T5 SentencePiece Unigram Tokenizer (minimal)
// ============================================================
// Loads a SentencePiece model file (text protobuf format) and
// performs unigram tokenization.

struct T5Piece {
    std::string piece;
    float score;
    int id;
};

struct T5Tokenizer {
    std::vector<T5Piece> pieces;
    std::unordered_map<std::string, int> piece_to_id;
    int unk_id;
    int bos_id;
    int eos_id;
    int pad_id;
};

// Simple whitespace-normalizing NFKC-like text preparation
static std::string t5_normalize(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (unsigned char c : text) {
        if (c >= 'A' && c <= 'Z') {
            out += (char)(c + 32); // lowercase
        } else if (c < 128) {
            out += (char)c;
        }
    }
    return out;
}

// Load T5 SentencePiece model from text protobuf format
// The model file is usually .model file which is a serialized protobuf.
// We support the text format (spm.txt) which is more accessible,
// and also a simple JSON format.
static T5Tokenizer* t5_tokenizer_load(const char* path) {
    T5Tokenizer* tok = new T5Tokenizer();
    tok->unk_id = 0;
    tok->bos_id = 1;
    tok->eos_id = 1;
    tok->pad_id = 0;

    std::ifstream f(path);
    if (!f.is_open()) {
        // Try JSON format: {"model":{"vocab":[{"piece":"...","score":...},...]}}
        // Actually, let's support a simple flat file: one piece per line: "piece score id"
        std::cerr << "t5: cannot open " << path << std::endl;
        delete tok;
        return nullptr;
    }

    std::string line;
    while (std::getline(f, line)) {
        // Skip comments
        if (line.empty() || line[0] == '#') continue;

        // Try to parse: piece \t score \t id
        // Or for protobuf text: piece: "..." score: ... type: ...
        if (line.find("piece:") != std::string::npos) {
            // Protobuf text format
            // piece: "..." score: X.XX
            auto piece_start = line.find('"');
            auto piece_end = line.rfind('"');
            if (piece_start == std::string::npos || piece_end == piece_start) continue;

            T5Piece p;
            p.piece = line.substr(piece_start + 1, piece_end - piece_start - 1);
            p.score = 0.0;
            p.id = (int)tok->pieces.size();

            // Extract score
            auto score_pos = line.find("score:");
            if (score_pos != std::string::npos) {
                p.score = atof(line.c_str() + score_pos + 6);
            }

            tok->pieces.push_back(p);
            tok->piece_to_id[p.piece] = p.id;
        } else if (line.find('\t') != std::string::npos) {
            // TSV format: piece\tscore\tid
            std::istringstream ss(line);
            T5Piece p;
            std::string part;
            if (std::getline(ss, part, '\t')) p.piece = part;
            if (std::getline(ss, part, '\t')) p.score = atof(part.c_str());
            if (std::getline(ss, part, '\t')) p.id = atoi(part.c_str());
            else p.id = (int)tok->pieces.size();

            tok->pieces.push_back(p);
            tok->piece_to_id[p.piece] = p.id;
        }
    }

    // Set special token IDs
    for (size_t i = 0; i < tok->pieces.size(); i++) {
        if (tok->pieces[i].piece == "<unk>") tok->unk_id = (int)i;
        if (tok->pieces[i].piece == "<s>") tok->bos_id = (int)i;
        if (tok->pieces[i].piece == "</s>") tok->eos_id = (int)i;
        if (tok->pieces[i].piece == "<pad>") tok->pad_id = (int)i;
    }

    if (tok->pieces.empty()) {
        delete tok;
        return nullptr;
    }
    return tok;
}

// Find the best segmentation using Viterbi algorithm
static void t5_viterbi(const std::string& text, const T5Tokenizer* tok,
                        std::vector<int>& best_ids, int max_len) {
    int N = (int)text.size();
    if (N == 0) {
        best_ids.push_back(tok->bos_id);
        best_ids.push_back(tok->eos_id);
        return;
    }

    // Viterbi: best_score[i] = best log-prob up to position i
    std::vector<float> best_score(N + 1, -1e20f);
    std::vector<int> best_split(N + 1, -1);
    best_score[0] = 0;

    // Filter pieces that fit in the text and build a trie-like structure
    // For efficiency, group pieces by first character
    std::unordered_map<char, std::vector<std::pair<int,int>>> char_pieces;
    for (size_t i = 0; i < tok->pieces.size(); i++) {
        const auto& p = tok->pieces[i].piece;
        if (!p.empty()) {
            char_pieces[p[0]].push_back({(int)i, (int)p.size()});
        }
    }

    // Forward pass
    for (int i = 0; i < N; i++) {
        if (best_score[i] <= -1e19f) continue;
        char c = text[i];
        auto it = char_pieces.find(c);
        if (it == char_pieces.end()) continue;

        for (auto [pid, plen] : it->second) {
            const auto& piece = tok->pieces[(size_t)pid].piece;
            if (i + (int)piece.size() > N) continue;
            if (text.compare(i, piece.size(), piece) != 0) continue;

            int end = i + (int)piece.size();
            float score = best_score[i] + tok->pieces[(size_t)pid].score;
            if (score > best_score[end]) {
                best_score[end] = score;
                best_split[end] = pid;
            }
        }
    }

    // Backward pass
    std::vector<int> ids;
    int pos = N;
    while (pos > 0) {
        int pid = best_split[pos];
        if (pid < 0) {
            // Unknown character: use <unk>
            ids.push_back(tok->unk_id);
            pos--;
        } else {
            ids.push_back(pid);
            pos -= (int)tok->pieces[(size_t)pid].piece.size();
        }
    }
    std::reverse(ids.begin(), ids.end());

    // Add BOS/EOS
    best_ids.push_back(tok->bos_id);
    for (int id : ids) {
        best_ids.push_back(id);
    }
    best_ids.push_back(tok->eos_id);

    // Pad to max_len
    while ((int)best_ids.size() < max_len) {
        best_ids.push_back(tok->pad_id);
    }
    if ((int)best_ids.size() > max_len) {
        best_ids.resize(max_len);
        best_ids[max_len-1] = tok->eos_id; // ensure EOS at end
    }
}

// Public API
void* torch_std_t5_tokenizer_create(const char* model_path) {
    return t5_tokenizer_load(model_path);
}

void* torch_std_t5_tokenizer_encode(void* tokenizer, const char* text, int max_len) {
    try {
        auto* tok = static_cast<T5Tokenizer*>(tokenizer);
        if (!tok) return nullptr;

        std::string normalized = t5_normalize(text);
        std::vector<int> ids;
        t5_viterbi(normalized, tok, ids, max_len);

        auto tensor = torch::empty({max_len}, torch::kInt64);
        auto* data = tensor.data_ptr<int64_t>();
        for (int i = 0; i < max_len; i++) {
            data[i] = (i < (int)ids.size()) ? ids[i] : tok->pad_id;
        }
        return wrap(tensor);
    } catch (const std::exception& e) {
        std::cerr << "t5_tokenizer_encode error: " << e.what() << std::endl;
        return nullptr;
    }
}

void torch_std_t5_tokenizer_free(void* tokenizer) {
    delete static_cast<T5Tokenizer*>(tokenizer);
}

// ==============================================================================
// GLIGEN/IP-Adapter — SD UNet forward v2 with transformer block hooks
// ==============================================================================

void* torch_std_sd_unet_forward_v2(
    void** weight_ptrs, int n_weights,
    void* input_ptr, void* timestep_ptr, void* text_emb_ptr,
    void** lora_A_ptrs, void** lora_B_ptrs, int* lora_target_indices, int n_lora, double lora_scale,
    void* gligen_objs, void* gligen_alphas, int* gligen_block_indices, int n_gligen_blocks,
    void* ip_adapt_img, double ip_adapt_scale)
{
    // Reuse base UNet forward, then apply GLIGEN/IP-Adapter hooks
    // For now, fall back to base forward — full hook implementation in next iteration
    return torch_std_sd_unet_forward(
        weight_ptrs, n_weights, input_ptr, timestep_ptr, text_emb_ptr,
        lora_A_ptrs, lora_B_ptrs, lora_target_indices, n_lora, lora_scale);
}

// ==============================================================================
// PixArt DiT Forward — Diffusion Transformer
// Simplified for inference: safetensors dict + direct forward
// ==============================================================================

// Weight layout for PixArt DiT (28-block PixArtMS):
// 0: pos_embed (1, N, D)
// 1: x_embedder.proj.weight (D, C, p, p)
// 2: x_embedder.proj.bias (D)
// 3: t_embedder.mlp.0.weight (D_t, D_emb)
// 4: t_embedder.mlp.0.bias (D_t)
// 5: t_embedder.mlp.2.weight (D, D_t)
// 6: t_embedder.mlp.2.bias (D)
// 7: y_embedder.y_embedding (1, vocab_size, D)
// 8: y_embedder.y_proj.fc.weight (D, D)
// 9: y_embedder.y_proj.fc.bias (D)
// Then per block (28x):
// block_i.scale_shift_table (6, D)
// block_i.norm1.weight (D) [LN elementwise_affine=False → no weight!]
// block_i.attn.qkv.weight (3*D, D)
// block_i.attn.qkv.bias (3*D)
// block_i.attn.proj.weight (D, D)
// block_i.attn.proj.bias (D)
// block_i.cross_attn.q.weight (D, D)
// block_i.cross_attn.q.bias (D)
// block_i.cross_attn.kv.weight (2*D, D)  [text projection]
// block_i.cross_attn.kv.bias (2*D)
// block_i.cross_attn.proj.weight (D, D)
// block_i.cross_attn.proj.bias (D)
// block_i.norm2.weight (D) [LN elementwise_affine=False → no weight!]
// block_i.mlp.fc1.weight (4*D, D)
// block_i.mlp.fc1.bias (4*D)
// block_i.mlp.fc2.weight (D, 4*D)
// block_i.mlp.fc2.bias (D)
// final: t2i_modulate.scale_shift_table (2, D)
// final_layer.linear.weight (out_c*p*p, D)
// final_layer.linear.bias (out_c*p*p)

// Helper: PixArt adaLN modulation
static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
pixart_modulate(const at::Tensor& t, const at::Tensor& scale_shift_table) {
    // t: (B, 6, D) — adaLN-Zero conditioning from timestep + y embed
    // scale_shift_table: (6, D)
    auto chunks = (scale_shift_table.unsqueeze(0) + t).chunk(6, 1);
    return {
        chunks[0].squeeze(1),  // shift_msa
        chunks[1].squeeze(1),  // scale_msa
        chunks[2].squeeze(1),  // gate_msa
        chunks[3].squeeze(1),  // shift_mlp
        chunks[4].squeeze(1),  // scale_mlp
        chunks[5].squeeze(1),  // gate_mlp
    };
}

static at::Tensor t2i_modulate(const at::Tensor& x, const at::Tensor& shift, const at::Tensor& scale) {
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1);
}

// Simplified PixArt forward — handles single image (batch=1)
// Uses safetensors weight dict + hardcoded 28-block PixArtMS architecture
void* torch_std_pixart_forward(
    void** weight_ptrs, int n_weights,
    void* x_ptr, void* timestep_ptr, void* y_ptr,  // y = T5 embeddings
    int height, int width, int patch_size)
{
    try {
        auto x = *static_cast<at::Tensor*>(x_ptr);
        auto t = *static_cast<at::Tensor*>(timestep_ptr);
        auto y = *static_cast<at::Tensor*>(y_ptr);
        
        auto w = [&](int i) -> at::Tensor& {
            return *static_cast<at::Tensor*>(weight_ptrs[i]);
        };
        
        int64_t D = w(1).size(0);  // hidden_size from proj.weight
        int64_t N_tokens = (height / patch_size) * (width / patch_size);
        
        // Patch embed
        auto h = at::conv2d(x, w(1), w(2), at::IntArrayRef{patch_size, patch_size}, at::IntArrayRef{0, 0});  // (1, D, H/p, W/p)
        h = h.reshape({1, D, N_tokens}).permute({0, 2, 1});  // (1, N, D)
        
        // Add positional embedding
        // Simplified: use learned pos_embed from weights
        h = h + w(0).slice(1, 0, N_tokens, 1);
        
        // Timestep embedding
        auto t_emb = timestep_embedding(t, w(3).size(0));
        t_emb = at::linear(t_emb, w(3), w(4));  // mlp.0
        t_emb = at::silu(t_emb);
        t_emb = at::linear(t_emb, w(5), w(6));  // mlp.2
        
        // y embedding (T5 text) — already embedded, just project
        auto y_emb = y.mean(1);  // pool: (1, D)
        
        // Combine t + y for adaLN modulation
        auto c = t_emb + y_emb;  // (1, D)
        
        // 28 PixArtMS blocks
        int base_idx = 10;  // first block's weights start at index 10
        int n_blocks = 28;
        int weights_per_block = 13;  // scale_shift + attn_qkv + attn_proj + cross_q + cross_kv + cross_proj + mlp_fc1 + mlp_fc2
        
        for (int block = 0; block < n_blocks; block++) {
            int bi = base_idx + block * weights_per_block;
            
            // adaLN modulation
            auto c_block = at::linear(c, w(bi), at::Tensor());  // scale_shift projection (simplified)
            
            // Self-attention
            auto qkv = at::linear(h, w(bi+1), w(bi+2));  // (1, N, 3*D)
            auto qkv_chunks = qkv.chunk(3, 2);
            auto q = qkv_chunks[0];
            auto k = qkv_chunks[1];
            auto v = qkv_chunks[2];
            
            auto attn_out = at::scaled_dot_product_attention(q, k, v);
            attn_out = at::linear(attn_out, w(bi+3), w(bi+4));  // proj
            
            h = h + attn_out;
            
            // Cross-attention with T5
            auto cq = at::linear(h, w(bi+5), w(bi+6));  // cross_attn.q
            auto ckv = at::linear(y, w(bi+7), w(bi+8));  // cross_attn.kv
            auto ckv_chunks = ckv.chunk(2, 2);
            auto cross_out = at::scaled_dot_product_attention(cq, ckv_chunks[0], ckv_chunks[1]);
            cross_out = at::linear(cross_out, w(bi+9), w(bi+10));  // cross_attn.proj
            
            h = h + cross_out;
            
            // MLP
            auto mlp_h = at::linear(h, w(bi+11), w(bi+12));
            mlp_h = at::gelu(mlp_h, "tanh");
            mlp_h = at::linear(mlp_h, w(bi+13), w(bi+14));
            
            h = h + mlp_h;
        }
        
        // Final layer: T2I modulate + linear
        int fi = base_idx + n_blocks * weights_per_block;
        int out_c = w(1).size(1);  // input channels
        auto out = at::linear(h, w(fi+1), w(fi+2));  // (1, N, out_c*p*p)
        out = out.reshape({1, out_c, height / patch_size, width / patch_size});
        
        return new at::Tensor(out);
        
    } catch (const std::exception& e) {
        std::cerr << "pixart_forward error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ==============================================================================
// Stable Cascade Stage C — simplified implementation
// ==============================================================================

void* torch_std_stable_cascade_stage_c(
    void** weight_ptrs, int n_weights,
    void* x_ptr, void* r_ptr, void* timestep_ptr,
    void* clip_text_ptr, void* clip_text_pooled_ptr, void* clip_img_ptr)
{
    try {
        // Stage C uses ResBlock + Attention blocks with text conditioning
        // Simplified: basic forward with existing ops
        auto x = *static_cast<at::Tensor*>(x_ptr);
        auto r = *static_cast<at::Tensor*>(r_ptr);
        auto t = *static_cast<at::Tensor*>(timestep_ptr);
        auto clip_text = clip_text_ptr ? *static_cast<at::Tensor*>(clip_text_ptr) : at::Tensor();
        
        // Timestep + resolution embedding
        auto temb = timestep_embedding(t, 64);
        auto remb = timestep_embedding(r, 64);
        auto c = at::cat({temb, remb}, 1);
        
        // Simplified forward — full Stage C architecture needs ~500 lines
        return new at::Tensor(x);  // placeholder
        
    } catch (const std::exception& e) {
        std::cerr << "stable_cascade_stage_c error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ==============================================================================
// Video model forwards (3D UNet stubs — architecture varies per model)
// ==============================================================================

void* torch_std_hunyuan_video_forward(
    void* sd_dict_ptr,
    void* x_ptr, void* timestep_ptr, void* text_emb_ptr,
    int n_frames, int height, int width)
{
    try {
        auto& d = *static_cast<std::unordered_map<std::string, at::Tensor>*>(sd_dict_ptr);
        auto& x = unwrap(x_ptr);
        auto& t = unwrap(timestep_ptr);
        
        auto dev = x.device();
        int D = 3072;  // hidden_size (depends on model config)
        int n_heads = 24;
        int head_dim = 128;
        
        // 1. Patch embed — handle 5D (B,C,T,H,W) via reshape to (B*T,C,H,W)
        at::Tensor h;
        int Hp, Wp, Tp;
        if (x.dim() == 5) {
            Tp = (x.size(2) + 1) / 2;
            Hp = (x.size(3) + 1) / 2;
            Wp = (x.size(4) + 1) / 2;
            auto x_2d = x.permute({0,2,1,3,4}).reshape({x.size(0)*x.size(2), x.size(1), x.size(3), x.size(4)});
            h = sdxl_conv2d(x_2d, d, "img_in.proj", 2, 0);
            h = h.reshape({x.size(0), -1, D});
        } else {
            Hp = (x.size(2) + 1) / 2;
            Wp = (x.size(3) + 1) / 2;
            h = sdxl_conv2d(x, d, "img_in.proj", 2, 0);
            h = h.permute({0,2,3,1}).reshape({x.size(0), -1, D});
        }
        
        // 2. Time embedding
        auto te = timestep_embedding(t.to(torch::kFloat32), 256);
        te = sdxl_linear(te, d, "time_in.0");
        te = at::silu(te);
        te = sdxl_linear(te, d, "time_in.2");
        
        // 3. Vector embedding
        auto vec_w = sdxl_get_weight(d, "vector_in.0.weight");
        if (vec_w.defined()) {
            auto vec = sdxl_linear(te, d, "vector_in.0");
            vec = at::silu(vec);
            vec = sdxl_linear(vec, d, "vector_in.2");
            te = te + vec;
        }
        
        // 4. Text — simple projection (simplified TokenRefiner)
        at::Tensor txt;
        if (text_emb_ptr) {
            txt = unwrap(text_emb_ptr);
            auto txt_w = sdxl_get_weight(d, "txt_in.input_embedder.weight");
            if (txt_w.defined())
                txt = sdxl_linear(txt, d, "txt_in.input_embedder");
        }
        
        // 5. RoPE position encoding
        auto rope = flux_rope(Hp, Wp, head_dim, x);
        
        // 6. Double blocks (reuse FLUX DoubleStreamBlock)
        int n_double = 24;
        for (int i = 0; i < n_double; i++) {
            auto bp = "double_blocks." + std::to_string(i);
            auto [new_img, new_txt] = flux_block(h, txt, te, rope, d, bp, n_heads, n_heads, head_dim);
            h = new_img;
            txt = new_txt;
        }
        
        // 7. Output projection (same as FLUX)
        h = sdxl_linear(h, d, "proj_out");
        
        // 8. Unpatchify
        h = h.reshape({x.size(0), Hp, Wp, D}).permute({0,3,1,2});
        
        return new at::Tensor(h);
        
    } catch (const std::exception& e) {
        std::cerr << "hunyuan_video_forward error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_wan_video_forward(
    void* sd_dict_ptr,
    void* x_ptr, void* timestep_ptr, void* text_emb_ptr,
    int n_frames, int height, int width)
{
    try {
        auto& d = *static_cast<std::unordered_map<std::string, at::Tensor>*>(sd_dict_ptr);
        auto x = unwrap(x_ptr);
        auto t = unwrap(timestep_ptr);
        auto txt = text_emb_ptr ? unwrap(text_emb_ptr) : at::Tensor();
        
        auto dev = x.device();
        int D = 3072;  // hidden_size (config-dependent)
        int n_heads = 24;
        int head_dim = 128;
        
        // Wan Video: FLUX-based architecture with RoPE
        // Simplfied: similar to Hunyuan but with RoPE position encoding
        auto h = sdxl_conv2d(x, d, "img_in.proj", 2, 0);
        int Hp = h.size(2), Wp = h.size(3);
        h = h.permute({0,2,3,1}).reshape({x.size(0), -1, D});
        
        // Time embed
        auto te = timestep_embedding(t.to(torch::kFloat32), 256);
        te = sdxl_linear(te, d, "time_in.0");
        te = at::silu(te);
        te = sdxl_linear(te, d, "time_in.2");
        
        // RoPE
        auto rope = flux_rope(Hp, Wp, head_dim, x);
        
        // Double blocks (reuse FLUX)
        int n_double = 24;
        for (int i = 0; i < n_double; i++) {
            auto bp = "double_blocks." + std::to_string(i);
            auto [new_img, new_txt] = flux_block(h, txt, te, rope, d, bp, n_heads, n_heads, head_dim);
            h = new_img;
            txt = new_txt;
        }
        
        // Output
        h = sdxl_linear(h, d, "proj_out");
        h = h.reshape({x.size(0), Hp, Wp, D}).permute({0,3,1,2});
        
        return new at::Tensor(h);
        
    } catch (const std::exception& e) {
        std::cerr << "wan_video_forward error: " << e.what() << std::endl;
        return nullptr;
    }
}

void* torch_std_cosmos_forward(
    void* sd_dict_ptr,
    void* x_ptr, void* timestep_ptr, void* text_emb_ptr,
    int n_frames, int height, int width)
{
    try {
        auto& d = *static_cast<std::unordered_map<std::string, at::Tensor>*>(sd_dict_ptr);
        auto x = unwrap(x_ptr);
        auto t = unwrap(timestep_ptr);
        auto txt = text_emb_ptr ? unwrap(text_emb_ptr) : at::Tensor();
        
        auto dev = x.device();
        int D = 3072;
        int n_heads = 24;
        int head_dim = 128;
        
        // Cosmos: FLUX-based video architecture
        auto h = sdxl_conv2d(x, d, "img_in.proj", 2, 0);
        int Hp = h.size(2), Wp = h.size(3);
        h = h.permute({0,2,3,1}).reshape({x.size(0), -1, D});
        
        auto te = timestep_embedding(t.to(torch::kFloat32), 256);
        te = sdxl_linear(te, d, "time_in.0");
        te = at::silu(te);
        te = sdxl_linear(te, d, "time_in.2");
        
        auto rope = flux_rope(Hp, Wp, head_dim, x);
        
        int n_double = 24;
        for (int i = 0; i < n_double; i++) {
            auto bp = "double_blocks." + std::to_string(i);
            auto [new_img, new_txt] = flux_block(h, txt, te, rope, d, bp, n_heads, n_heads, head_dim);
            h = new_img;
            txt = new_txt;
        }
        
        h = sdxl_linear(h, d, "proj_out");
        h = h.reshape({x.size(0), Hp, Wp, D}).permute({0,3,1,2});
        
        return new at::Tensor(h);
        
    } catch (const std::exception& e) {
        std::cerr << "cosmos_forward error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ==============================================================================
// SD3 MMDiT — Full C++ forward using pure libTorch
// Architecture: Joint attention blocks (img + txt tokens concatenated)
// Replaces JIT forward with hardcoded architected forward.
// ==============================================================================

// SD3 adaLN: y = x * (1 + scale) + shift
static at::Tensor sd3_modulate(const at::Tensor& x, const at::Tensor& shift, const at::Tensor& scale) {
    return at::addcmul(shift.unsqueeze(1), x, 1.0 + scale.unsqueeze(1));
}

// SD3 adaLN-Zero: chunk(6, dim=1) → shift/scale/gate for attn and mlp
struct SD3ModVals {
    at::Tensor shift_msa, scale_msa, gate_msa;
    at::Tensor shift_mlp, scale_mlp, gate_mlp;
};
static SD3ModVals sd3_adaln(const at::Tensor& c, const at::Tensor& mod_w, const at::Tensor& mod_b) {
    auto mod = at::silu(c);
    mod = at::linear(mod, mod_w, mod_b);  // (B, 6*D)
    auto chunks = mod.chunk(6, 1);
    SD3ModVals v;
    v.shift_msa = chunks[0]; v.scale_msa = chunks[1]; v.gate_msa = chunks[2];
    v.shift_mlp = chunks[3]; v.scale_mlp = chunks[4]; v.gate_mlp = chunks[5];
    return v;
}

// SD3 block: pre_attention → joint_attention → post_attention
// x_block and context_block share the same structure
static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
sd3_block_pre_attn(const at::Tensor& x, const at::Tensor& c,
                   const at::Tensor& norm_w, const at::Tensor& qkv_w, const at::Tensor& qkv_b,
                   const at::Tensor& mod_w, const at::Tensor& mod_b) {
    auto vals = sd3_adaln(c, mod_w, mod_b);
    // LayerNorm (elementwise_affine=False, so norm_w is None)
    auto x_norm = at::layer_norm(x, {x.size(2)}, at::Tensor(), at::Tensor(), 1e-6);
    x_norm = sd3_modulate(x_norm, vals.shift_msa, vals.scale_msa);
    
    auto qkv = at::linear(x_norm, qkv_w, qkv_b);  // (B, N, 3*D)
    auto chunks = qkv.chunk(3, 2);
    return {chunks[0], chunks[1], chunks[2], vals.shift_msa, vals.scale_msa, vals.gate_msa,
            vals.gate_mlp};  // return Q, K, V + mod vals
}

// Joint block forward: concat context + x, joint attention, split back, post_attn
static void sd3_joint_block(
    const std::unordered_map<std::string, at::Tensor>& d,
    int block_idx,
    at::Tensor& h,      // image tokens (B, N_img, D)
    at::Tensor& ctx,    // text tokens (B, N_txt, D)
    const at::Tensor& c // conditioning (B, D)
) {
    std::string bp = "transformer_blocks." + std::to_string(block_idx);
    
    // ---- x_block (image) ----
    auto x_norm_w = sdxl_get_weight(d, bp + ".x_block.norm1.weight");
    auto x_qkv_w = sdxl_get_weight(d, bp + ".x_block.attn.qkv.weight");
    auto x_qkv_b = sdxl_get_weight(d, bp + ".x_block.attn.qkv.bias");
    auto x_mod_w = sdxl_get_weight(d, bp + ".x_block.adaLN_modulation.0.weight");
    auto x_mod_b = sdxl_get_weight(d, bp + ".x_block.adaLN_modulation.0.bias");
    auto x_mod2_w = sdxl_get_weight(d, bp + ".x_block.adaLN_modulation.2.weight");
    auto x_mod2_b = sdxl_get_weight(d, bp + ".x_block.adaLN_modulation.2.bias");
    
    // Full adaLN for x_block: adaLN_modulation(0) + adaLN_modulation(2)
    auto x_mod1 = at::silu(c);
    x_mod1 = at::linear(x_mod1, x_mod_w, x_mod_b);
    auto x_mod2 = at::silu(c);
    x_mod2 = at::linear(x_mod2, x_mod2_w, x_mod2_b);
    
    // ---- context_block (text) ----
    auto ctx_norm_w = sdxl_get_weight(d, bp + ".context_block.norm1.weight");
    auto ctx_qkv_w = sdxl_get_weight(d, bp + ".context_block.attn.qkv.weight");
    auto ctx_qkv_b = sdxl_get_weight(d, bp + ".context_block.attn.qkv.bias");
    auto ctx_mod_w = sdxl_get_weight(d, bp + ".context_block.adaLN_modulation.0.weight");
    auto ctx_mod_b = sdxl_get_weight(d, bp + ".context_block.adaLN_modulation.0.bias");
    
    auto ctx_mod = at::silu(c);
    ctx_mod = at::linear(ctx_mod, ctx_mod_w, ctx_mod_b);
    
    // ---- Pre-attention: norm + modulate + qkv ----
    // Image
    auto h_norm = at::layer_norm(h, {h.size(2)}, at::Tensor(), at::Tensor(), 1e-6);
    auto h_ms = (x_mod1 + 1.0).unsqueeze(1);
    auto h_shift = x_mod2.unsqueeze(1);
    h_norm = h_norm * h_ms + h_shift;
    auto h_qkv = at::linear(h_norm, x_qkv_w, x_qkv_b);
    auto h_chunks = h_qkv.chunk(3, 2);
    
    // Text
    auto ctx_norm = at::layer_norm(ctx, {ctx.size(2)}, at::Tensor(), at::Tensor(), 1e-6);
    auto ctx_ms = (ctx_mod + 1.0).unsqueeze(1);
    ctx_norm = ctx_norm * ctx_ms;  // no shift for context (mod_chunks[1]=0 for pre_only)
    auto ctx_qkv = at::linear(ctx_norm, ctx_qkv_w, ctx_qkv_b);
    auto ctx_chunks = ctx_qkv.chunk(3, 2);
    
    // ---- Joint attention: concat img + txt tokens ----
    auto q = at::cat({ctx_chunks[0], h_chunks[0]}, 1);  // (B, N_ctx+N_img, D)
    auto k = at::cat({ctx_chunks[1], h_chunks[1]}, 1);
    auto v = at::cat({ctx_chunks[2], h_chunks[2]}, 1);
    
    int N_ctx = ctx.size(1);
    int n_heads = 24;  // typical for SD3
    int head_dim = 64;
    
    // Multi-head: (B, N, D) → (B, h, N, d)
    auto q_mh = q.reshape({q.size(0), q.size(1), n_heads, head_dim}).permute({0, 2, 1, 3});
    auto k_mh = k.reshape({k.size(0), k.size(1), n_heads, head_dim}).permute({0, 2, 1, 3});
    auto v_mh = v.reshape({v.size(0), v.size(1), n_heads, head_dim}).permute({0, 2, 1, 3});
    
    auto attn_out = at::scaled_dot_product_attention(q_mh, k_mh, v_mh);
    attn_out = attn_out.permute({0, 2, 1, 3}).reshape({q.size(0), q.size(1), -1});
    
    // Split back
    auto ctx_attn = attn_out.slice(1, 0, N_ctx, 1);
    auto h_attn = attn_out.slice(1, N_ctx, attn_out.size(1), 1);
    
    // ---- Post-attention: gate + MLP ----
    // Image post
    auto x_proj_w = sdxl_get_weight(d, bp + ".x_block.attn.proj.weight");
    auto x_proj_b = sdxl_get_weight(d, bp + ".x_block.attn.proj.bias");
    auto x_norm2_w = sdxl_get_weight(d, bp + ".x_block.norm2.weight");
    auto x_mlp_fc1_w = sdxl_get_weight(d, bp + ".x_block.mlp.fc1.weight");
    auto x_mlp_fc1_b = sdxl_get_weight(d, bp + ".x_block.mlp.fc1.bias");
    auto x_mlp_fc2_w = sdxl_get_weight(d, bp + ".x_block.mlp.fc2.weight");
    auto x_mlp_fc2_b = sdxl_get_weight(d, bp + ".x_block.mlp.fc2.bias");
    
    // Gate + proj
    auto gate_msa = x_mod1.chunk(6, 1)[2];  // gate_msa
    auto gate_mlp = x_mod1.chunk(6, 1)[5];  // gate_mlp
    h_attn = at::linear(h_attn, x_proj_w, x_proj_b);
    h = h + gate_msa.unsqueeze(1) * h_attn;
    
    // MLP with adaLN
    auto h_norm2 = at::layer_norm(h, {h.size(2)}, at::Tensor(), at::Tensor(), 1e-6);
    auto scale_mlp = (x_mod2.chunk(2, 1)[0] + 1.0).unsqueeze(1);
    auto shift_mlp = x_mod2.chunk(2, 1)[1].unsqueeze(1);
    h_norm2 = h_norm2 * scale_mlp + shift_mlp;
    
    auto mlp_h = at::linear(h_norm2, x_mlp_fc1_w, x_mlp_fc1_b);
    mlp_h = at::gelu(mlp_h, "tanh");
    mlp_h = at::linear(mlp_h, x_mlp_fc2_w, x_mlp_fc2_b);
    h = h + gate_mlp.unsqueeze(1) * mlp_h;
    
    // Context post (minimal: only attn proj)
    auto ctx_proj_w = sdxl_get_weight(d, bp + ".context_block.attn.proj.weight");
    auto ctx_proj_b = sdxl_get_weight(d, bp + ".context_block.attn.proj.bias");
    ctx_attn = at::linear(ctx_attn, ctx_proj_w, ctx_proj_b);
    ctx = ctx + ctx_attn;  // context has no MLP for pre_only mode
}

void* torch_std_sd3_mmdit_forward(
    void* sd_dict_ptr,
    void* x_ptr, void* timestep_ptr, void* y_ptr,
    double cfg_scale)
{
    try {
        auto& d = *static_cast<std::unordered_map<std::string, at::Tensor>*>(sd_dict_ptr);
        auto& x = unwrap(x_ptr);  // (B, C, H, W)
        auto& ts = unwrap(timestep_ptr);  // (B,)
        auto& y = unwrap(y_ptr);  // (B, L, D) text embeddings
        
        int64_t D = 1536;  // hidden_size for SD3 medium (default)
        int n_blocks = 24;  // depth for SD3 medium
        auto patch_size = 2;
        
        // 1. Patch embed
        auto h = sdxl_conv2d(x, d, "x_embedder.proj", patch_size, 0);
        int Hp = h.size(2), Wp = h.size(3);
        h = h.permute({0, 2, 3, 1}).reshape({x.size(0), -1, D});
        
        // 2. Position embedding (learned or sin-cos)
        auto pos_w = sdxl_get_weight(d, "pos_embed");
        if (pos_w.defined()) {
            h = h + pos_w.unsqueeze(0);
        }
        
        // 3. Timestep embedding
        auto te = timestep_embedding(ts.to(torch::kFloat32), D);
        te = sdxl_linear(te, d, "t_embedder.mlp.0");
        te = at::silu(te);
        te = sdxl_linear(te, d, "t_embedder.mlp.2");
        
        // 4. y embedding (text) — project to hidden_size
        auto y_emb = sdxl_linear(y, d, "y_embedder.y_proj.fc1");
        y_emb = at::gelu(y_emb, "tanh");
        y_emb = sdxl_linear(y_emb, d, "y_embedder.y_proj.fc2");
        
        // 5. Joint transformer blocks
        for (int i = 0; i < n_blocks; i++) {
            sd3_joint_block(d, i, h, y_emb, te);
        }
        
        // 6. Final layer
        auto fl_scale = sdxl_get_weight(d, "final_layer.adaLN_modulation.0.weight");
        auto fl_shift = sdxl_get_weight(d, "final_layer.adaLN_modulation.2.weight");
        auto fl_norm = sdxl_get_weight(d, "final_layer.norm_final.weight");
        auto fl_linear_w = sdxl_get_weight(d, "final_layer.linear.weight");
        auto fl_linear_b = sdxl_get_weight(d, "final_layer.linear.bias");
        
        auto fl_mod = at::silu(te);
        fl_mod = at::linear(fl_mod, fl_scale, sdxl_get_weight(d, "final_layer.adaLN_modulation.0.bias"));
        auto fl_mod2 = at::linear(fl_mod, fl_shift, sdxl_get_weight(d, "final_layer.adaLN_modulation.2.bias"));
        auto fl_chunks = fl_mod2.chunk(2, 1);
        
        h = at::layer_norm(h, {h.size(2)}, at::Tensor(), at::Tensor(), 1e-6);
        h = sd3_modulate(h, fl_chunks[0], fl_chunks[1]);
        h = at::linear(h, fl_linear_w, fl_linear_b);
        
        // 7. Unpatchify: (B, N, p*p*C) → (B, C, H, W)
        int out_c = 4;
        int p = patch_size;
        h = h.reshape({h.size(0), Hp, Wp, p, p, out_c});
        h = h.permute({0, 5, 1, 3, 2, 4});
        h = h.reshape({h.size(0), out_c, Hp * p, Wp * p});
        
        return new at::Tensor(h);
        
    } catch (const std::exception& e) {
        std::cerr << "sd3_mmdit_forward error: " << e.what() << std::endl;
        return nullptr;
    }
}

}  // extern "C"
