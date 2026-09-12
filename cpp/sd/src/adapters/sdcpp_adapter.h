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
    std::string ip_adapter_path;       // native IP-Adapter weights (sd.cpp format)
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
    int lora_apply_mode = LORA_APPLY_AT_RUNTIME;
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

    // ADetailer
    bool adetailer_enabled = false;
    std::string ad_model_path;
    std::string ad_prompt;
    std::string ad_negative_prompt;
    int ad_inpaint_width = 512;
    int ad_inpaint_height = 512;
    float ad_denoising_strength = 0.4f;

    // img2img: init image path + denoising strength (ignored when empty)
    std::string init_image_path;
    float strength = 1.0f;
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

    std::vector<Image> generate(const ImageGenerationParams& params);

    // LoRA management (persistent, applied at runtime during generate)
    void set_lora(const std::string& path, float multiplier);
    void clear_loras();

    // IPAdapter: load model + reference image once, use across generations
    void set_ipadapter(const std::string& model_path,
                       const std::string& clip_vision_path,
                       const std::string& image_path,
                       float weight);
    void set_ipadapter_enabled(bool enabled, float weight);

    // Native model version name (e.g. "SDXL") exposed by sd.cpp
    std::string get_model_version_name() const;

    // img2img: load the init image once; used by subsequent generate() calls.
    // Pass an empty path to clear.
    void set_init_image(const std::string& image_path, float strength);

    // ControlNet: hot-swap the control net and set the per-generation control image.
    bool load_control_net(const std::string& path);
    void set_control_image(const std::string& image_path, float strength);

    // Inpainting: set the mask image (used together with set_init_image).
    void set_mask(const std::string& mask_path);

    // Batch size for subsequent generate() calls (>= 1).
    void set_batch_count(int n);

    // Per-generation sampling overrides
    void set_clip_skip(int n);
    void set_flow_shift(float shift);

    // Model weight type override (requires reload); wtype < 0 keeps current
    void set_wtype(int wtype);

    // Enable diffusion flash attention (requires reload)
    void set_flash_attn(bool enabled);

    // Area conditioning (sd.cpp patch); prompts separated by '\n', rects/strengths CSV
    void set_area_conds(const char* prompts_sep, const char* rects_csv, const char* strengths_csv);

    // RescaleCFG override (applied per sampling step)
    void set_rescale_cfg(bool enabled, float multiplier);

    // Video CFG guidance override (per-batch CFG scale; mode 0=linear, 1=triangle)
    void set_video_cfg(bool enabled, int mode, float min_cfg);

    // Sigma range override (ModelSamplingContinuousEDM / ModelSamplingContinuousV)
    void set_sigma_range(bool enabled, float sigma_min, float sigma_max);

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

int sd_pipeline_load_ex(sd_pipeline_t pipeline,
                        const char* model_path,
                        const char* clip_l_path,
                        const char* clip_g_path,
                        const char* vae_path,
                        int wtype,
                        int n_threads,
                        int diffusion_fa,
                        const char* diffusion_model_path,
                        const char* llm_path);

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

int64_t sd_compute_hires_resolution(int target_w, int target_h);

/**
 * Unified generation entry point. Superset of sd_pipeline_generate /
 * sd_pipeline_generate_hires / sd_pipeline_generate_adetailer.
 *
 *   - hires_width/hires_height > 0  enables HiRes Fix; if width/height <= 0
 *     the base resolution is computed via sd_compute_hires_resolution().
 *   - vae_tiling != 0 forces explicit tiling params; otherwise auto-tiling
 *     is decided inside SDPipeline::generate for large images.
 *   - clarity/sharpen_* > 0 enables post-processing.
 *   - ad_model_path non-empty enables ADetailer.
 * Returns 0 on success, non-zero on error.
 */
int sd_pipeline_generate_full(sd_pipeline_t pipeline,
                              const char* prompt,
                              const char* negative_prompt,
                              int width,
                              int height,
                              int hires_width,
                              int hires_height,
                              int steps,
                              float cfg,
                              const char* sample_method,
                              const char* scheduler,
                              int64_t seed,
                              int vae_tiling,
                              int vae_tile_size,
                              float vae_tile_overlap,
                              int hires_steps,
                              float hires_strength,
                              int freeu,
                              float freeu_b1,
                              float freeu_b2,
                              int sag,
                              float sag_scale,
                              float clarity,
                              float sharpen_amount,
                              int sharpen_radius,
                              float smart_sharpen_strength,
                              int smart_sharpen_radius,
                              float edge_sharpen_amount,
                              int edge_sharpen_radius,
                              float edge_sharpen_threshold,
                              const char* ad_model_path,
                              const char* ad_prompt,
                              const char* ad_negative_prompt,
                              const char* output_path);

/**
 * All-in-one HiRes Fix generation:
 *   Computes base resolution via sd_compute_hires_resolution,
 *   loads the model (via sd_pipeline_load_ex),
 *   generates with hires enabled,
 *   applies post-processing (clarity/sharpen),
 *   and saves to output_path.
 * Returns 0 on success, non-zero on error.
 */
int sd_pipeline_generate_hires(sd_pipeline_t pipeline,
                                const char* prompt,
                                const char* negative_prompt,
                                int target_width,
                                int target_height,
                                int steps,
                                float cfg,
                                const char* sample_method,
                                const char* scheduler,
                                int64_t seed,
                                int vae_tiling,
                                int vae_tile_size,
                                float vae_tile_overlap,
                                int hires_steps,
                                float hires_strength,
                                int freeu,
                                float freeu_b1,
                                float freeu_b2,
                                int sag,
                                float sag_scale,
                                float clarity,
                                float sharpen_amount,
                                int sharpen_radius,
                                float smart_sharpen_strength,
                                int smart_sharpen_radius,
                                float edge_sharpen_amount,
                                int edge_sharpen_radius,
                                float edge_sharpen_threshold,
                                const char* output_path);

/**
 * Load a LoRA model and associate it with the pipeline.
 * Multiple calls add multiple LoRAs. Call with empty path to clear all.
 */
int sd_pipeline_load_lora(sd_pipeline_t pipeline,
                           const char* lora_path,
                           float multiplier);

/**
 * Load IPAdapter model + CLIP Vision model + reference image.
 * Once loaded, IPAdapter will be applied to subsequent generate() calls.
 */
int sd_pipeline_set_ipadapter(sd_pipeline_t pipeline,
                               const char* model_path,
                               const char* clip_vision_path,
                               const char* image_path,
                               float weight);

/**
 * Enable/disable the loaded IPAdapter for subsequent generate() calls.
 * Set weight <= 0 to disable.
 */
int sd_pipeline_set_ipadapter_enabled(sd_pipeline_t pipeline,
                                       int enabled,
                                       float weight);

/**
 * Load an img2img init image and associate it with the pipeline.
 * Subsequent generate() calls run img2img with the given denoising strength.
 * Pass an empty path to clear. Returns 0 on success.
 */
int sd_pipeline_set_init_image(sd_pipeline_t pipeline,
                               const char* image_path,
                               float strength);

/**
 * Hot-swap the ControlNet model on a loaded pipeline.
 * Returns 0 on success, non-zero on failure.
 */
int sd_pipeline_load_control_net(sd_pipeline_t pipeline, const char* path);

/**
 * Set the per-generation ControlNet control image + strength.
 * Pass an empty path to clear. Returns 0 on success.
 */
int sd_pipeline_set_control_image(sd_pipeline_t pipeline,
                                  const char* image_path,
                                  float strength);

/**
 * Set the inpainting mask image (grayscale). Used with sd_pipeline_set_init_image.
 * Pass an empty path to clear. Returns 0 on success.
 */
int sd_pipeline_set_mask(sd_pipeline_t pipeline, const char* mask_path);

/** Set the batch size for subsequent generate() calls. Returns 0 on success. */
int sd_pipeline_set_batch_count(sd_pipeline_t pipeline, int n);

/** Return the model version name detected by sd.cpp (e.g. "SDXL"). NULL if unknown. */
const char* sd_pipeline_get_model_version_name(sd_pipeline_t pipeline);

/** Per-generation sampling overrides. */
int sd_pipeline_set_clip_skip(sd_pipeline_t pipeline, int n);
int sd_pipeline_set_flow_shift(sd_pipeline_t pipeline, float shift);

/** Override model weight type (reloads the context). wtype < 0 keeps current. */
int sd_pipeline_set_wtype(sd_pipeline_t pipeline, int wtype);

/** Enable/disable diffusion flash attention (reloads the context). */
int sd_pipeline_set_flash_attn(sd_pipeline_t pipeline, int enabled);

/** Configure area conditioning (prompts separated by '\n', rects/strengths CSV). */
int sd_pipeline_set_area_conds(sd_pipeline_t pipeline, const char* prompts_sep, const char* rects_csv, const char* strengths_csv);

/** Configure RescaleCFG (applied per sampling step). */
int sd_pipeline_set_rescale_cfg(sd_pipeline_t pipeline, int enabled, float multiplier);

/** Configure video CFG guidance (mode 0=linear, 1=triangle). */
int sd_pipeline_set_video_cfg(sd_pipeline_t pipeline, int enabled, int mode, float min_cfg);

/** Override sampling sigma range (ModelSamplingContinuousEDM/V). */
int sd_pipeline_set_sigma_range(sd_pipeline_t pipeline, int enabled, float sigma_min, float sigma_max);

/** Rotate an image by 90/180/270 degrees (counter-clockwise) and save as PNG. */
int sd_rotate_image(const char* input_path, const char* output_path, int degrees);

/** Flip an image: method 0 = vertical, 1 = horizontal. */
int sd_flip_image(const char* input_path, const char* output_path, int method);

/** Blend two images: out = a*(1-factor) + b*factor. */
int sd_blend_images(const char* path1, const char* path2, const char* output_path,
                    float factor);

/**
 * Generate image with ADetailer face restoration post-processing.
 * Takes all the same params as sd_pipeline_generate plus:
 *   - ad_model_path: path to YOLO detector model (e.g. face_yolov8m.pt)
 *   - ad_prompt / ad_negative_prompt: optional per-region prompt override
 * Returns 0 on success, non-zero on error.
 */
int sd_pipeline_generate_adetailer(sd_pipeline_t pipeline,
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
                                    const char* ad_model_path,
                                    const char* ad_prompt,
                                    const char* ad_negative_prompt,
                                    const char* output_path);

/** Utility: create directory and all parents if missing. Returns 0 on success. */
int sd_ensure_dir(const char* path);

/** Resize an image file to width x height and save as PNG. Returns 0 on success. */
int sd_resize_image(const char* input_path, const char* output_path,
                    int width, int height);

/** Scale an image file by a factor (dims * scale_by) and save as PNG. Returns 0 on success. */
int sd_scale_image(const char* input_path, const char* output_path, float scale_by);

/** Invert an image file and save as PNG. Returns 0 on success. */
int sd_invert_image(const char* input_path, const char* output_path);

/** Create a solid-color image and save as PNG. Returns 0 on success. */
int sd_make_solid_image(const char* output_path, int width, int height,
                        int r, int g, int b);

/** Pad an image (constant color) and save as PNG. Returns 0 on success. */
int sd_pad_image(const char* input_path, const char* output_path,
                 int left, int top, int right, int bottom,
                 int r, int g, int b);

/** Gaussian blur an image and save as PNG. Returns 0 on success. */
int sd_blur_image(const char* input_path, const char* output_path, float sigma);

/** Stack two images vertically (second resized to first's width) and save as PNG. */
int sd_batch_images(const char* path1, const char* path2, const char* output_path);

/** Composite src onto dest at (x,y) using an optional grayscale mask. Saves PNG. */
int sd_composite_masked(const char* dest_path, const char* src_path,
                        const char* mask_path, const char* output_path,
                        int x, int y);

/** Crop an image to (x,y,width,height) and save as PNG. Returns 0 on success. */
int sd_crop_image(const char* input_path, const char* output_path,
                  int x, int y, int width, int height);

#ifdef __cplusplus
}
#endif
