#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
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
 * @brief LoRA configuration entry.
 */
struct LoraConfig {
    std::string path;
    float multiplier = 1.0f;
};

/**
 * @brief SD model configuration.
 *
 * Mirrors sd_ctx_params_t from stable-diffusion.h.
 */
struct ModelConfig {
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string vae_path;
    std::string diffusion_model_path;  // standalone diffusion model (e.g. Z-Image GGUF)
    std::string llm_path;              // LLM text encoder for DiT models
    int n_threads = 8;
    bool keep_vae_on_cpu = false;
    bool keep_clip_on_cpu = false;
    bool flash_attn = false;
    bool diffusion_flash_attn = false;
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

    // LoRA
    std::vector<LoraConfig> loras;

    // VAE tiling
    bool vae_tiling = false;
    int vae_tile_size_x = 128;
    int vae_tile_size_y = 128;
    float vae_tile_overlap = 0.5f;

    // HiRes Fix
    bool hires_enabled = false;
    int hires_width = 0;
    int hires_height = 0;
    int hires_steps = 20;
    float hires_strength = 0.35f;

    // FreeU
    bool freeu_enabled = false;
    float freeu_b1 = 1.3f;
    float freeu_b2 = 1.4f;
    float freeu_s1 = 0.9f;
    float freeu_s2 = 0.2f;

    // SAG (Self-Attention Guidance)
    bool sag_enabled = false;
    float sag_scale = 1.0f;
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
