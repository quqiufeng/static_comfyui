#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sd {

/**
 * @brief Simple image container (RGB/RGBA, uint8)
 */
struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> data;

    size_t size() const { return width * height * channels; }
    bool empty() const { return data.empty(); }
};

/**
 * @brief SD model configuration.
 *
 * Mirrors sd_ctx_params_t from stable-diffusion.h but only exposes
 * the fields we need for SDXL txt2img.
 */
struct ModelConfig {
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string vae_path;
    int n_threads = 8;
    bool keep_vae_on_cpu = false;
    bool keep_clip_on_cpu = false;
};

/**
 * @brief Image generation parameters.
 *
 * Mirrors sd_img_gen_params_t from stable-diffusion.h.
 */
struct ImageGenerationParams {
    std::string prompt;
    std::string negative_prompt;
    int width = 1024;
    int height = 1024;
    int steps = 20;
    float cfg_scale = 7.0f;
    int64_t seed = 42;
    int batch_count = 1;
    std::string sample_method = "euler_a";
    std::string scheduler = "discrete";
};

/**
 * @brief Thin C++ wrapper around stable-diffusion.h C API.
 *
 * This is the only class that touches sd_ctx_t / generate_image etc.
 * Upper-layer code only sees sd::Image, sd::ModelConfig, sd::ImageGenerationParams.
 */
class SDPipeline {
public:
    SDPipeline();
    ~SDPipeline();

    // Non-copyable
    SDPipeline(const SDPipeline&) = delete;
    SDPipeline& operator=(const SDPipeline&) = delete;

    // Movable
    SDPipeline(SDPipeline&& other) noexcept;
    SDPipeline& operator=(SDPipeline&& other) noexcept;

    bool load(const ModelConfig& config);
    bool is_loaded() const;

    Image generate(const ImageGenerationParams& params);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sd
