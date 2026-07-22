#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <stable-diffusion.h>
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
    std::string clip_vision_path;
    std::string vae_path;
    std::string diffusion_model_path;  // standalone diffusion model (e.g. Z-Image GGUF)
    std::string llm_path;              // LLM text encoder for DiT models
    int n_threads = 8;
    bool keep_vae_on_cpu = false;
    bool keep_clip_on_cpu = false;
    // Weight type: SD_TYPE_F16, SD_TYPE_F32, etc. SD_TYPE_COUNT means auto.
    int wtype = SD_TYPE_COUNT;
    // RNG: STD_DEFAULT_RNG, CUDA_RNG, CPU_RNG
    int rng_type = STD_DEFAULT_RNG;
    int sampler_rng_type = RNG_TYPE_COUNT; // RNG_TYPE_COUNT means default
    // Prediction type: EPS_PRED, V_PRED, EDM_V_PRED, FLOW_PRED, ...
    int prediction = PREDICTION_COUNT; // PREDICTION_COUNT means auto
    bool flash_attn = false;
    bool diffusion_flash_attn = false;
    bool enable_mmap = false;
    // Optional backend spec (e.g. "CUDA", "CPU")
    std::string backend;
    std::string params_backend;
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
    float img_cfg_scale = INFINITY; // INFINITY means same as txt_cfg (native default)
    float distilled_guidance = 3.5f;
    int clip_skip = -1; // -1 means default
    int64_t seed = 42;
    int batch_count = 1;
    std::string sample_method = "euler_a";
    std::string scheduler = "discrete";
    float eta = INFINITY; // INFINITY means default

    // LoRA
    std::vector<LoraConfig> loras;

    // VAE tiling
    bool vae_tiling = false;
    int vae_tile_size_x = 128;
    int vae_tile_size_y = 128;
    float vae_tile_overlap = 0.5f;

    // HiRes Fix
    bool hires_enabled = false;
    std::string hires_upscaler = "latent"; // latent, latent-bicubic, lanczos, nearest, ...
    int hires_width = 0;
    int hires_height = 0;
    float hires_scale = 2.0f;
    int hires_steps = 20;
    float hires_strength = 0.35f;
    int hires_upscale_tile_size = 128;

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

/* --------------------------------------------------------------------------
 * C API for StaticPy / other language bindings.
 *
 * This is intentionally thin: it wraps sd::SDPipeline with opaque handles
 * and plain C types so that StaticPy's `extern fn` can call it directly.
 * -------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C" {
#endif

typedef void* sd_pipeline_t;

sd_pipeline_t sd_pipeline_create(void);
int sd_pipeline_free(sd_pipeline_t pipeline);

int sd_pipeline_load(sd_pipeline_t pipeline,
                     const char* model_path,
                     const char* clip_l_path,
                     const char* clip_g_path,
                     const char* vae_path,
                     int wtype,
                     int n_threads,
                     int diffusion_fa);

int sd_pipeline_generate(sd_pipeline_t pipeline,
                         const char* prompt,
                         const char* negative_prompt,
                         int width,
                         int height,
                         int steps,
                         float cfg,
                         const char* sample_method,
                         const char* scheduler,
                         int64_t seed,
                         int vae_tiling,
                         int vae_tile_size,
                         float vae_tile_overlap,
                         int hires,
                         int hires_width,
                         int hires_height,
                         int hires_steps,
                         float hires_strength,
                         int freeu,
                         float freeu_b1,
                         float freeu_b2,
                         int sag,
                         float sag_scale,
                         const char* output_path);

/** Utility: create directory and all parents if missing. Returns 0 on success. */
int sd_ensure_dir(const char* path);

#ifdef __cplusplus
}
#endif
