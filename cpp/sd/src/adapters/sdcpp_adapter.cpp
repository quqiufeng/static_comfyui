#include "sdcpp_adapter.h"
#include "postproc.h"

#include <stable-diffusion.h>

#include <cstddef>
#include <cstdint>
#include <utility>

// Forward declaration for log callback used in constructor.
static void sdcpp_log_cb(enum sd_log_level_t level, const char* log, void* data);

namespace sd {

class SDPipeline::Impl {
public:
    sd_ctx_t* ctx = nullptr;

    // LoRA strings must outlive generate_image call
    std::vector<std::string> lora_paths;
    std::vector<sd_lora_t> lora_entries;

    ~Impl() {
        if (ctx) {
            free_sd_ctx(ctx);
            ctx = nullptr;
        }
    }
};

SDPipeline::SDPipeline() : impl_(std::make_unique<Impl>()) {
    sd_set_log_callback(sdcpp_log_cb, nullptr);
}

SDPipeline::~SDPipeline() = default;

SDPipeline::SDPipeline(SDPipeline&& other) noexcept = default;

SDPipeline& SDPipeline::operator=(SDPipeline&& other) noexcept = default;

bool SDPipeline::load(const ModelConfig& config) {
    if (!impl_) {
        return false;
    }
    if (impl_->ctx) {
        free_sd_ctx(impl_->ctx);
        impl_->ctx = nullptr;
    }

    sd_ctx_params_t params;
    sd_ctx_params_init(&params);

    params.model_path   = config.model_path.c_str();
    if (!config.clip_l_path.empty()) {
        params.clip_l_path = config.clip_l_path.c_str();
    }
    if (!config.clip_g_path.empty()) {
        params.clip_g_path = config.clip_g_path.c_str();
    }
    if (!config.clip_vision_path.empty()) {
        params.clip_vision_path = config.clip_vision_path.c_str();
    }
    if (!config.vae_path.empty()) {
        params.vae_path = config.vae_path.c_str();
    }
    if (!config.diffusion_model_path.empty()) {
        params.diffusion_model_path = config.diffusion_model_path.c_str();
    }
    if (!config.llm_path.empty()) {
        params.llm_path = config.llm_path.c_str();
    }
    params.n_threads            = config.n_threads;
    params.wtype                = static_cast<sd_type_t>(config.wtype);
    params.rng_type             = static_cast<rng_type_t>(config.rng_type);
    params.sampler_rng_type     = static_cast<rng_type_t>(config.sampler_rng_type);
    params.prediction           = static_cast<prediction_t>(config.prediction);
    params.flash_attn           = config.flash_attn;
    params.diffusion_flash_attn = config.diffusion_flash_attn;
    params.enable_mmap          = config.enable_mmap;
    if (!config.backend.empty()) {
        params.backend = config.backend.c_str();
    }
    if (!config.params_backend.empty()) {
        params.params_backend = config.params_backend.c_str();
    }

    impl_->ctx = new_sd_ctx(&params);
    return impl_->ctx != nullptr;
}

bool SDPipeline::is_loaded() const {
    return impl_ && impl_->ctx != nullptr;
}

Image SDPipeline::generate(const ImageGenerationParams& params) {
    Image result;
    if (!impl_ || !impl_->ctx) {
        return result;
    }

    sd_img_gen_params_t img_params;
    sd_img_gen_params_init(&img_params);

    img_params.prompt          = params.prompt.c_str();
    img_params.negative_prompt = params.negative_prompt.c_str();
    img_params.width           = params.width;
    img_params.height          = params.height;
    img_params.clip_skip       = params.clip_skip;
    img_params.seed            = params.seed;
    img_params.batch_count     = params.batch_count;

    img_params.sample_params.sample_steps     = params.steps;
    img_params.sample_params.guidance.txt_cfg = params.cfg_scale;
    if (std::isfinite(params.img_cfg_scale)) {
        img_params.sample_params.guidance.img_cfg = params.img_cfg_scale;
    }
    if (std::isfinite(params.distilled_guidance)) {
        img_params.sample_params.guidance.distilled_guidance = params.distilled_guidance;
    }
    img_params.sample_params.sample_method    = str_to_sample_method(params.sample_method.c_str());
    img_params.sample_params.scheduler      = str_to_scheduler(params.scheduler.c_str());
    if (img_params.sample_params.sample_method == SAMPLE_METHOD_COUNT) {
        img_params.sample_params.sample_method = EULER_A_SAMPLE_METHOD;
    }
    if (img_params.sample_params.scheduler == SCHEDULER_COUNT) {
        img_params.sample_params.scheduler = DISCRETE_SCHEDULER;
    }
    if (std::isfinite(params.eta)) {
        img_params.sample_params.eta = params.eta;
    }

    // LoRA
    impl_->lora_paths.clear();
    impl_->lora_entries.clear();
    for (const auto& lora : params.loras) {
        impl_->lora_paths.push_back(lora.path);
        sd_lora_t entry;
        entry.is_high_noise = false;
        entry.multiplier    = lora.multiplier;
        entry.path          = impl_->lora_paths.back().c_str();
        impl_->lora_entries.push_back(entry);
    }
    if (!impl_->lora_entries.empty()) {
        img_params.loras      = impl_->lora_entries.data();
        img_params.lora_count = static_cast<uint32_t>(impl_->lora_entries.size());
    }

    // VAE tiling
    if (params.vae_tiling) {
        img_params.vae_tiling_params.enabled      = true;
        img_params.vae_tiling_params.tile_size_x = params.vae_tile_size_x;
        img_params.vae_tiling_params.tile_size_y = params.vae_tile_size_y;
        img_params.vae_tiling_params.target_overlap = params.vae_tile_overlap;
    } else if (params.width > 512 || params.height > 512) {
        // Auto-enable VAE tiling for large images to avoid OOM during decode.
        // Tile size is in latent space; 64x64 latent = 512x512 pixels for SDXL VAE (scale=8).
        img_params.vae_tiling_params.enabled      = true;
        img_params.vae_tiling_params.tile_size_x = 64;
        img_params.vae_tiling_params.tile_size_y = 64;
        img_params.vae_tiling_params.target_overlap = 0.5f;
    }

    // HiRes Fix
    if (params.hires_enabled) {
        img_params.hires.enabled             = true;
        img_params.hires.upscaler            = str_to_sd_hires_upscaler(params.hires_upscaler.c_str());
        if (img_params.hires.upscaler == SD_HIRES_UPSCALER_COUNT) {
            img_params.hires.upscaler = SD_HIRES_UPSCALER_LATENT;
        }
        img_params.hires.target_width        = params.hires_width;
        img_params.hires.target_height       = params.hires_height;
        img_params.hires.scale               = params.hires_scale;
        img_params.hires.steps               = params.hires_steps;
        img_params.hires.denoising_strength  = params.hires_strength;
        img_params.hires.upscale_tile_size   = params.hires_upscale_tile_size;
    }

    // FreeU
    img_params.freeu.enabled = params.freeu_enabled;
    if (params.freeu_enabled) {
        img_params.freeu.b1 = params.freeu_b1;
        img_params.freeu.b2 = params.freeu_b2;
        img_params.freeu.s1 = params.freeu_s1;
        img_params.freeu.s2 = params.freeu_s2;
    }

    // SAG
    img_params.sag.enabled = params.sag_enabled;
    if (params.sag_enabled) {
        img_params.sag.scale = params.sag_scale;
    }

    sd_image_t* images = nullptr;
    int num_images = 0;
    std::fprintf(stderr, "[C++ gen] calling generate_image...\n");
    bool ok = generate_image(impl_->ctx, &img_params, &images, &num_images);
    std::fprintf(stderr, "[C++ gen] generate_image returned ok=%d images=%p num=%d\n", ok, (void*)images, num_images);
    if (!ok || !images || num_images == 0) {
        return result;
    }

    result.width    = static_cast<int>(images[0].width);
    result.height   = static_cast<int>(images[0].height);
    result.channels = static_cast<int>(images[0].channel);
    const size_t bytes = result.width * result.height * result.channels;

    if (images[0].data != nullptr && bytes > 0) {
        result.data.assign(images[0].data, images[0].data + bytes);
    }

    free_sd_images(images, num_images);
    return result;
}

} // namespace sd

/* --------------------------------------------------------------------------
 * C API implementation
 * -------------------------------------------------------------------------- */

#include <png.h>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

static void sdcpp_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    (void)data;
    const char* prefix = "SD";
    switch (level) {
        case SD_LOG_DEBUG: prefix = "SD-DEBUG"; break;
        case SD_LOG_INFO:  prefix = "SD-INFO";  break;
        case SD_LOG_WARN:  prefix = "SD-WARN";  break;
        case SD_LOG_ERROR: prefix = "SD-ERROR"; break;
        default: break;
    }
    std::fprintf(stderr, "[%s] %s\n", prefix, log);
}

static bool save_png(const char* path, const uint8_t* data, int w, int h, int channels) {
    FILE* fp = std::fopen(path, "wb");
    if (!fp) return false;

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { std::fclose(fp); return false; }

    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_write_struct(&png, nullptr); std::fclose(fp); return false; }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        std::fclose(fp);
        return false;
    }

    int color_type = (channels == 4) ? PNG_COLOR_TYPE_RGBA : PNG_COLOR_TYPE_RGB;
    png_init_io(png, fp);
    png_set_IHDR(png, info, w, h, 8, color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    for (int y = 0; y < h; y++) {
        png_write_row(png, data + y * w * channels);
    }

    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    std::fclose(fp);
    return true;
}

extern "C" {

sd_pipeline_t sd_pipeline_create(void) {
    sd_set_log_callback(sdcpp_log_cb, nullptr);
    return new sd::SDPipeline();
}

int sd_pipeline_free(sd_pipeline_t pipeline) {
    if (pipeline) {
        delete static_cast<sd::SDPipeline*>(pipeline);
    }
    return 0;
}

int sd_pipeline_load(sd_pipeline_t pipeline,
                     const char* model_path,
                     const char* clip_l_path,
                     const char* clip_g_path,
                     const char* vae_path,
                     int wtype,
                     int n_threads,
                     int diffusion_fa) {
    if (!pipeline || !model_path) return -1;

    std::fprintf(stderr, "[C API] sd_pipeline_load: model=%s clip_l=%s clip_g=%s vae=%s wtype=%d threads=%d fa=%d\n",
                 model_path ? model_path : "(null)",
                 clip_l_path ? clip_l_path : "(null)",
                 clip_g_path ? clip_g_path : "(null)",
                 vae_path ? vae_path : "(null)",
                 wtype, n_threads, diffusion_fa);

    sd::SDPipeline* p = static_cast<sd::SDPipeline*>(pipeline);

    sd::ModelConfig config;
    config.model_path = model_path;
    if (clip_l_path) config.clip_l_path = clip_l_path;
    if (clip_g_path) config.clip_g_path = clip_g_path;
    if (vae_path)    config.vae_path    = vae_path;
    config.wtype                = wtype;
    config.n_threads            = n_threads > 0 ? n_threads : 8;
    config.diffusion_flash_attn = diffusion_fa != 0;

    std::fprintf(stderr, "[C API] calling p->load...\n");
    bool ok = p->load(config);
    std::fprintf(stderr, "[C API] p->load returned %d, is_loaded=%d\n", ok ? 1 : 0, p->is_loaded() ? 1 : 0);
    return ok ? 0 : -2;
}

int sd_pipeline_load_ex(sd_pipeline_t pipeline,
                         const char* model_path,
                         const char* clip_l_path,
                         const char* clip_g_path,
                         const char* vae_path,
                         int wtype,
                         int n_threads,
                         int diffusion_fa,
                         const char* diffusion_model_path,
                         const char* llm_path) {
    if (!pipeline) return -1;

    sd::SDPipeline* p = static_cast<sd::SDPipeline*>(pipeline);
    sd::ModelConfig config;

    if (model_path) config.model_path = model_path;
    if (diffusion_model_path) config.diffusion_model_path = diffusion_model_path;
    if (llm_path) config.llm_path = llm_path;
    if (clip_l_path) config.clip_l_path = clip_l_path;
    if (clip_g_path) config.clip_g_path = clip_g_path;
    if (vae_path) config.vae_path = vae_path;
    config.wtype                = wtype;
    config.n_threads            = n_threads > 0 ? n_threads : 8;
    config.diffusion_flash_attn = diffusion_fa != 0;

    std::fprintf(stderr, "[C API] sd_pipeline_load_ex: model=%s diff=%s llm=%s clip_l=%s clip_g=%s vae=%s\n",
                 model_path ? model_path : "(null)",
                 diffusion_model_path ? diffusion_model_path : "(null)",
                 llm_path ? llm_path : "(null)",
                 clip_l_path ? clip_l_path : "(null)",
                 clip_g_path ? clip_g_path : "(null)",
                 vae_path ? vae_path : "(null)");

    bool ok = p->load(config);
    std::fprintf(stderr, "[C API] p->load returned %d, is_loaded=%d\n", ok ? 1 : 0, p->is_loaded() ? 1 : 0);
    return ok ? 0 : -2;
}

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
                         const char* output_path) {
    if (!pipeline || !prompt || !output_path) return -1;

    sd::SDPipeline* p = static_cast<sd::SDPipeline*>(pipeline);
    if (!p->is_loaded()) return -3;

    sd::ImageGenerationParams params;
    params.prompt          = prompt;
    params.negative_prompt = negative_prompt ? negative_prompt : "";
    params.width           = width  > 0 ? width  : 1024;
    params.height          = height > 0 ? height : 1024;
    params.steps           = steps  > 0 ? steps  : 20;
    params.cfg_scale       = cfg > 0.0f ? cfg : 7.0f;
    params.sample_method   = sample_method ? sample_method : "euler_a";
    params.scheduler       = scheduler ? scheduler : "discrete";
    params.seed            = seed;

    if (vae_tiling != 0) {
        params.vae_tiling       = true;
        params.vae_tile_size_x  = vae_tile_size > 0 ? vae_tile_size : 64;
        params.vae_tile_size_y  = params.vae_tile_size_x;
        params.vae_tile_overlap = vae_tile_overlap >= 0.0f && vae_tile_overlap <= 1.0f
                                  ? vae_tile_overlap : 0.5f;
    } else if (width > 512 || height > 512) {
        // Auto-enable VAE tiling for large images to avoid OOM during decode.
        params.vae_tiling       = true;
        params.vae_tile_size_x  = 64;
        params.vae_tile_size_y  = 64;
        params.vae_tile_overlap = 0.5f;
    }

    if (hires != 0 && hires_width > 0 && hires_height > 0) {
        params.hires_enabled  = true;
        params.hires_width    = hires_width;
        params.hires_height   = hires_height;
        params.hires_steps    = hires_steps > 0 ? hires_steps : 20;
        params.hires_strength = hires_strength >= 0.0f && hires_strength <= 1.0f
                                ? hires_strength : 0.35f;
    }

    if (freeu != 0) {
        params.freeu_enabled = true;
        params.freeu_b1      = freeu_b1 > 0.0f ? freeu_b1 : 1.3f;
        params.freeu_b2      = freeu_b2 > 0.0f ? freeu_b2 : 1.4f;
    }

    if (sag != 0) {
        params.sag_enabled = true;
        params.sag_scale   = sag_scale >= 0.0f ? sag_scale : 1.0f;
    }

    sd::Image image = p->generate(params);
    std::fprintf(stderr, "[C API] generate returned empty=%d w=%d h=%d c=%d\n",
                 image.empty(), image.width, image.height, image.channels);
    if (image.empty()) return -4;

    if (!save_png(output_path, image.data.data(), image.width, image.height, image.channels)) {
        std::fprintf(stderr, "[C API] save_png failed for %s\n", output_path);
        return -5;
    }
    std::fprintf(stderr, "[C API] saved %s (%dx%d, %d ch)\n", output_path, image.width, image.height, image.channels);

    return 0;
}

int64_t sd_compute_hires_resolution(int target_w, int target_h) {
    int low_w, low_h;

    if (target_w == 3840 && target_h == 2160) {
        low_w = 2560; low_h = 1440;
    } else if (target_w == 2560 && target_h == 1440) {
        low_w = 1920; low_h = 1080;
    } else if (target_w == 1920 && target_h == 1080) {
        low_w = 1536; low_h = 864;
    } else if (target_w == 1280 && target_h == 720) {
        low_w = 1024; low_h = 576;
    } else {
        int tw = target_w / 8;
        int th = target_h / 8;
        int lw = (tw * 4 / 5 + 7) / 8 * 8;
        int lh = (th * 4 / 5 + 7) / 8 * 8;
        low_w = lw * 8;
        low_h = lh * 8;
    }

    if (low_w < 512 || low_h < 512) {
        float ratio = static_cast<float>(target_w) / target_h;
        if (low_w < low_h) {
            low_w = 512;
            low_h = static_cast<int>(low_w / ratio / 8) * 8;
            if (low_h < 512) low_h = 512;
        } else {
            low_h = 512;
            low_w = static_cast<int>(low_h * ratio / 8) * 8;
            if (low_w < 512) low_w = 512;
        }
    }

    return (static_cast<int64_t>(low_w) << 32) | static_cast<int64_t>(low_h);
}

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
                                const char* output_path) {
    if (!pipeline || !prompt || !output_path) return -1;

    sd::SDPipeline* p = static_cast<sd::SDPipeline*>(pipeline);
    if (!p->is_loaded()) return -3;

    int64_t packed = sd_compute_hires_resolution(target_width, target_height);
    int low_w = static_cast<int>(packed >> 32);
    int low_h = static_cast<int>(packed & 0xFFFFFFFF);

    std::fprintf(stderr, "[C API] sd_pipeline_generate_hires: target=%dx%d base=%dx%d\n",
                 target_width, target_height, low_w, low_h);

    sd::ImageGenerationParams params;
    params.prompt          = prompt;
    params.negative_prompt = negative_prompt ? negative_prompt : "";
    params.width           = low_w;
    params.height          = low_h;
    params.steps           = steps > 0 ? steps : 20;
    params.cfg_scale       = cfg > 0.0f ? cfg : 7.0f;
    params.sample_method   = sample_method ? sample_method : "euler";
    params.scheduler       = scheduler ? scheduler : "discrete";
    params.seed            = seed;

    params.hires_enabled  = true;
    params.hires_width    = target_width;
    params.hires_height   = target_height;
    params.hires_steps    = hires_steps > 0 ? hires_steps : 20;
    params.hires_strength = hires_strength >= 0.0f && hires_strength <= 1.0f ? hires_strength : 0.35f;

    if (vae_tiling != 0) {
        params.vae_tiling       = true;
        params.vae_tile_size_x  = vae_tile_size > 0 ? vae_tile_size : 128;
        params.vae_tile_size_y  = params.vae_tile_size_x;
        params.vae_tile_overlap = vae_tile_overlap >= 0.0f && vae_tile_overlap <= 1.0f
                                  ? vae_tile_overlap : 0.5f;
    } else if (target_width > 512 || target_height > 512) {
        params.vae_tiling       = true;
        params.vae_tile_size_x  = 64;
        params.vae_tile_size_y  = 64;
        params.vae_tile_overlap = 0.5f;
    }

    if (freeu != 0) {
        params.freeu_enabled = true;
        params.freeu_b1      = freeu_b1 > 0.0f ? freeu_b1 : 1.3f;
        params.freeu_b2      = freeu_b2 > 0.0f ? freeu_b2 : 1.4f;
    }

    if (sag != 0) {
        params.sag_enabled = true;
        params.sag_scale   = sag_scale >= 0.0f ? sag_scale : 1.0f;
    }

    sd::Image image = p->generate(params);
    std::fprintf(stderr, "[C API] generate_hires returned empty=%d w=%d h=%d c=%d\n",
                 image.empty(), image.width, image.height, image.channels);
    if (image.empty()) return -4;

    bool has_postproc = (clarity > 0.0f || sharpen_amount > 0.0f);
    if (has_postproc) {
        std::fprintf(stderr, "[C API] postproc: clarity=%.2f sharpen=%.2f(r=%d)\n",
                     clarity, sharpen_amount, sharpen_radius);
        postproc::Params pp;
        pp.clarity         = clarity;
        pp.sharpen_amount  = sharpen_amount;
        pp.sharpen_radius  = sharpen_radius > 0 ? sharpen_radius : 1;
        if (!postproc::apply(image.data.data(), image.width, image.height, image.channels, pp)) {
            std::fprintf(stderr, "[C API] postproc failed\n");
            return -5;
        }
    }

    if (!save_png(output_path, image.data.data(), image.width, image.height, image.channels)) {
        std::fprintf(stderr, "[C API] save_png failed for %s\n", output_path);
        return -6;
    }
    std::fprintf(stderr, "[C API] saved %s (%dx%d, %d ch)\n", output_path, image.width, image.height, image.channels);

    return 0;
}

int sd_ensure_dir(const char* path) {
    if (!path || path[0] == '\0') return -1;
    try {
        fs::create_directories(fs::path(path));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[C API] sd_ensure_dir failed for %s: %s\n", path, e.what());
        return -1;
    }
}

} // extern "C"
