#include "ipadapter.h"

#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdio>

namespace sd {

struct IPAdapter::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "IPAdapter"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> clip_session;
    std::unique_ptr<Ort::Session> sdxl_plus_session;

    std::string clip_input_name;
    std::string clip_output_name;

    std::string sdxl_plus_input_name;
    std::string sdxl_plus_output_name;

    Impl() {
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }

    ~Impl() = default;

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    bool load_clip_vision(const std::string& path) {
        try {
            clip_session = std::make_unique<Ort::Session>(env, path.c_str(), session_options);

            auto input_count = clip_session->GetInputCount();
            auto output_count = clip_session->GetOutputCount();

            if (input_count < 1 || output_count < 1) {
                fprintf(stderr, "IPAdapter: CLIP Vision has no inputs/outputs\n");
                return false;
            }

            auto input_name_ptr = clip_session->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
            clip_input_name = input_name_ptr.get();

            auto output_name_ptr = clip_session->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
            clip_output_name = output_name_ptr.get();

            printf("IPAdapter: CLIP Vision ONNX model loaded: %s\n", path.c_str());
            printf("  Input: %s\n", clip_input_name.c_str());
            printf("  Output: %s\n", clip_output_name.c_str());

            auto input_type_info = clip_session->GetInputTypeInfo(0);
            auto input_shape = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
            std::string shape_str = "[";
            for (size_t i = 0; i < input_shape.size(); i++) {
                if (i > 0) shape_str += ", ";
                if (input_shape[i] == -1) {
                    shape_str += "N";
                } else {
                    shape_str += std::to_string(input_shape[i]);
                }
            }
            shape_str += "]";
            printf("  Input shape: %s\n", shape_str.c_str());

            return true;
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "IPAdapter: Failed to load CLIP Vision: %s\n", e.what());
            return false;
        }
    }

    bool load_sdxl_plus(const std::string& path) {
        try {
            sdxl_plus_session = std::make_unique<Ort::Session>(env, path.c_str(), session_options);

            auto input_name_ptr = sdxl_plus_session->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
            sdxl_plus_input_name = input_name_ptr.get();

            auto output_name_ptr = sdxl_plus_session->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
            sdxl_plus_output_name = output_name_ptr.get();

            printf("IPAdapter: SDXL Plus ONNX model loaded: %s\n", path.c_str());
            printf("  Input: %s\n", sdxl_plus_input_name.c_str());
            printf("  Output: %s\n", sdxl_plus_output_name.c_str());

            return true;
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "IPAdapter: Failed to load SDXL Plus: %s\n", e.what());
            return false;
        }
    }

    std::vector<float> run_clip_vision(const std::vector<float>& input_image) {
        if (!clip_session) {
            fprintf(stderr, "IPAdapter: CLIP Vision session not loaded\n");
            return {};
        }

        try {
            std::vector<int64_t> input_shape = {1, 3, 224, 224};
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float*>(input_image.data()),
                input_image.size(), input_shape.data(), input_shape.size());

            const char* input_names[] = {clip_input_name.c_str()};
            const char* output_names[] = {clip_output_name.c_str()};

            auto output_tensors = clip_session->Run(Ort::RunOptions{nullptr},
                                                      input_names, &input_tensor, 1,
                                                      output_names, 1);

            if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
                fprintf(stderr, "IPAdapter: CLIP Vision inference returned no output\n");
                return {};
            }

            auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            auto output_shape = output_info.GetShape();
            size_t num_elements = output_info.GetElementCount();

            if (num_elements == 0) {
                fprintf(stderr, "IPAdapter: CLIP Vision output is empty\n");
                return {};
            }

            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            std::vector<float> result(output_data, output_data + num_elements);

            if (output_shape.size() == 3) {
                printf("IPAdapter: CLIP Vision output shape [%zu x %zu x %zu] (hidden states)\n",
                       (size_t)output_shape[0], (size_t)output_shape[1], (size_t)output_shape[2]);
            } else {
                printf("IPAdapter: CLIP Vision output shape [%zu x %zu] (pooled embedding)\n",
                       output_shape.size() >= 1 ? (size_t)output_shape[0] : 0,
                       output_shape.size() >= 2 ? (size_t)output_shape[1] : num_elements);
            }

            return result;
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "IPAdapter: CLIP Vision inference failed: %s\n", e.what());
            return {};
        }
    }

    std::vector<float> run_sdxl_plus(const std::vector<float>& clip_embedding) {
        if (!sdxl_plus_session) {
            fprintf(stderr, "IPAdapter: SDXL Plus session not loaded\n");
            return {};
        }

        try {
            std::vector<int64_t> input_shape = {1, 257, 1280};
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float*>(clip_embedding.data()),
                clip_embedding.size(), input_shape.data(), input_shape.size());

            const char* input_names[] = {sdxl_plus_input_name.c_str()};
            const char* output_names[] = {sdxl_plus_output_name.c_str()};

            auto output_tensors = sdxl_plus_session->Run(Ort::RunOptions{nullptr},
                                                           input_names, &input_tensor, 1,
                                                           output_names, 1);

            if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
                fprintf(stderr, "IPAdapter: SDXL Plus inference returned no output\n");
                return {};
            }

            auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            size_t num_elements = output_info.GetElementCount();

            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            std::vector<float> result(output_data, output_data + num_elements);

            printf("IPAdapter: SDXL Plus output: %zu floats (16x2048)\n", num_elements);
            return result;
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "IPAdapter: SDXL Plus inference failed: %s\n", e.what());
            return {};
        }
    }
};

IPAdapter::IPAdapter()
    : impl_(std::make_unique<Impl>()) {
}

IPAdapter::IPAdapter(const IPAdapterConfig& config)
    : config_(config)
    , impl_(std::make_unique<Impl>()) {
    if (!config_.model_path.empty() && !config_.clip_vision_path.empty()) {
        load_model(config_.model_path, config_.clip_vision_path);
    }
    if (!config_.image_path.empty()) {
        load_reference_image(config_.image_path);
    }
}

IPAdapter::~IPAdapter() = default;

IPAdapter::IPAdapter(IPAdapter&&) noexcept = default;
IPAdapter& IPAdapter::operator=(IPAdapter&&) noexcept = default;

bool IPAdapter::load_model(const std::string& model_path, const std::string& clip_vision_path) {
    printf("IPAdapter: loading models...\n");
    printf("  IPAdapter: %s\n", model_path.c_str());
    printf("  CLIP Vision: %s\n", clip_vision_path.c_str());

    bool clip_ok = impl_->load_clip_vision(clip_vision_path);
    if (!clip_ok) {
        fprintf(stderr, "IPAdapter: CLIP Vision loading failed\n");
        model_loaded_ = false;
        return false;
    }

    bool ipa_ok = impl_->load_sdxl_plus(model_path);
    if (!ipa_ok) {
        fprintf(stderr, "IPAdapter: SDXL Plus loading failed\n");
        model_loaded_ = false;
        return false;
    }

    model_loaded_ = true;
    printf("IPAdapter: all models loaded successfully\n");
    return true;
}

bool IPAdapter::load_reference_image(const std::string& image_path) {
    printf("IPAdapter: processing reference image: %s\n", image_path.c_str());

    if (!model_loaded_) {
        fprintf(stderr, "IPAdapter: models not loaded, cannot process image\n");
        return false;
    }

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        fprintf(stderr, "IPAdapter: failed to load image: %s\n", image_path.c_str());
        return false;
    }
    printf("IPAdapter: loaded image %dx%d\n", img.cols, img.rows);

    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_val[3] = {0.229f, 0.224f, 0.225f};

    std::vector<float> input_data(3 * 224 * 224);
    for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
            cv::Vec3b pixel = resized.at<cv::Vec3b>(y, x);
            input_data[0 * 224 * 224 + y * 224 + x] = (pixel[0] / 255.0f - mean[0]) / std_val[0];
            input_data[1 * 224 * 224 + y * 224 + x] = (pixel[1] / 255.0f - mean[1]) / std_val[1];
            input_data[2 * 224 * 224 + y * 224 + x] = (pixel[2] / 255.0f - mean[2]) / std_val[2];
        }
    }

    printf("IPAdapter: preprocessed image to [1, 3, 224, 224] float32 CHW\n");

    auto clip_embedding = impl_->run_clip_vision(input_data);
    if (clip_embedding.empty()) {
        fprintf(stderr, "IPAdapter: CLIP Vision inference failed\n");
        return false;
    }
    printf("IPAdapter: CLIP Vision embedding size: %zu\n", clip_embedding.size());

    if (clip_embedding.size() != 257 * 1280) {
        fprintf(stderr, "IPAdapter: CLIP Vision output size %zu, expected %d for hidden states\n",
                clip_embedding.size(), 257 * 1280);
    }

    auto sdxl_tokens = impl_->run_sdxl_plus(clip_embedding);
    if (sdxl_tokens.empty()) {
        fprintf(stderr, "IPAdapter: SDXL Plus inference failed\n");
        return false;
    }
    printf("IPAdapter: SDXL Plus output size: %zu (16x2048)\n", sdxl_tokens.size());

    image_tokens_ = std::move(sdxl_tokens);
    num_tokens_ = 16;

    float min_val = *std::min_element(image_tokens_.begin(), image_tokens_.end());
    float max_val = *std::max_element(image_tokens_.begin(), image_tokens_.end());
    float mean_val = std::accumulate(image_tokens_.begin(), image_tokens_.end(), 0.0f) / image_tokens_.size();
    printf("IPAdapter: tokens stats: min=%.4f, max=%.4f, mean=%.4f\n",
           min_val, max_val, mean_val);

    return true;
}

} // namespace sd
