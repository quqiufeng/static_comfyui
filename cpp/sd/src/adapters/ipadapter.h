#pragma once

#include <string>
#include <vector>
#include <memory>

namespace sd {

// IPAdapter: 图像提示词
// 通过 CLIP Vision 提取图像特征，注入到扩散模型的注意力中
struct IPAdapterConfig {
    std::string model_path;          // IPAdapter 模型路径 (.onnx)
    std::string clip_vision_path;    // CLIP Vision 模型路径 (.onnx)
    std::string image_path;          // 参考图像路径
    float weight = 1.0f;             // 注入权重 (0.0-1.0)
};

class IPAdapter {
public:
    IPAdapter();
    explicit IPAdapter(const IPAdapterConfig& config);
    ~IPAdapter();

    // 禁止拷贝，允许移动
    IPAdapter(const IPAdapter&) = delete;
    IPAdapter& operator=(const IPAdapter&) = delete;
    IPAdapter(IPAdapter&&) noexcept;
    IPAdapter& operator=(IPAdapter&&) noexcept;

    // 加载 ONNX 模型（CLIP Vision + IPAdapter MLP）
    bool load_model(const std::string& model_path, const std::string& clip_vision_path);

    // 加载参考图像，提取特征
    // 内部会运行 CLIP Vision → IPAdapter MLP 完整管线
    bool load_reference_image(const std::string& image_path);

    // 获取计算好的 image tokens（扁平化 float 向量）
    // SD1.5: [1, 2560] — 单 token
    // SDXL Plus: [16, 2560] — 16 tokens (flattened as [40960])
    // 可直接拼接到 text context（2560-dim cap_feat_dim）
    const std::vector<float>& get_image_tokens() const { return image_tokens_; }

    // 获取 token 数量（SD1.5=1, SDXL Plus=16）
    int get_num_tokens() const { return num_tokens_; }

    // 获取每个 token 的维度
    int get_token_dim() const {
        if (num_tokens_ <= 0) return 0;
        return static_cast<int>(image_tokens_.size()) / num_tokens_;
    }

    // 是否已加载
    bool is_loaded() const { return model_loaded_; }

    const IPAdapterConfig& config() const { return config_; }

private:
    IPAdapterConfig config_;
    bool model_loaded_ = false;

    // 缓存 image tokens (投影后)
    // SD1.5: [1, 2560] = 2560 floats
    // SDXL Plus: [16, 2560] = 40960 floats (flattened)
    std::vector<float> image_tokens_;
    int num_tokens_ = 1;  // SD1.5=1, SDXL Plus=16

    // ONNX Runtime 实现细节 (PIMPL)
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sd
