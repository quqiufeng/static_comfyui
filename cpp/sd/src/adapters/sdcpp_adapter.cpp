#include "sdcpp_adapter.h"
#include "postproc.h"

#include <stable-diffusion.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>

// Forward declaration for log callback used in constructor.
static void sdcpp_log_cb(enum sd_log_level_t level, const char* log, void* data);

namespace sd {

class SDPipeline::Impl {
public:
    sd_ctx_t* ctx = nullptr;
    int n_threads = 8;

    // Last config used to create the context; kept so we can reload
    // (native IP-Adapter / scale overrides are sd_ctx_params_t fields).
    ModelConfig config;

    // LoRA strings must outlive generate_image call
    std::vector<std::string> lora_paths;
    std::vector<sd_lora_t> lora_entries;

    // Native IP-Adapter reference image (owns the RGB pixel buffer)
    std::vector<uint8_t> ip_adapter_image_data;
    sd_image_t ip_adapter_image{};
    bool has_ip_adapter_image = false;
    float ip_adapter_strength = 1.0f;

    // img2img init image (owns the RGB pixel buffer)
    std::vector<uint8_t> init_image_data;
    sd_image_t init_image{};
    bool has_init_image = false;
    float init_strength = 1.0f;

    // ControlNet control image (owns the RGB pixel buffer)
    std::vector<uint8_t> control_image_data;
    sd_image_t control_image{};
    bool has_control_image = false;
    float control_strength = 1.0f;

    // Inpainting mask (grayscale)
    std::vector<uint8_t> mask_image_data;
    sd_image_t mask_image{};
    bool has_mask_image = false;

    int batch_count = 1;

    // Per-generation sampling overrides
    int clip_skip = -1;
    float flow_shift = 0.0f;

    // RescaleCFG
    bool rescale_cfg_enabled   = false;
    float rescale_cfg_multiplier = 0.7f;

    // Video CFG guidance
    bool video_cfg_enabled = false;
    int video_cfg_mode     = 0;
    float video_cfg_min    = 1.0f;

    // Sigma range override (ModelSamplingContinuousEDM/V)
    bool sigma_range_enabled = false;
    float sigma_range_min    = 0.0f;
    float sigma_range_max    = 0.0f;

    // Area conditioning (sd.cpp patch)
    std::vector<std::string> area_prompt_storage;
    std::vector<sd_area_cond_t> area_conds;

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
    impl_->config = config;  // 记住配置，供后续 reload（原生 IP-Adapter / scale）
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
    if (!config.ip_adapter_path.empty()) {
        params.ip_adapter_path = config.ip_adapter_path.c_str();
    }
    params.n_threads            = config.n_threads;
    impl_->n_threads            = config.n_threads;
    params.wtype                = static_cast<sd_type_t>(config.wtype);
    params.rng_type             = static_cast<rng_type_t>(config.rng_type);
    params.sampler_rng_type     = static_cast<rng_type_t>(config.sampler_rng_type);
    params.prediction           = static_cast<prediction_t>(config.prediction);
    params.flash_attn           = config.flash_attn;
    params.diffusion_flash_attn = config.diffusion_flash_attn;
    params.enable_mmap          = config.enable_mmap;
    params.lora_apply_mode      = static_cast<lora_apply_mode_t>(config.lora_apply_mode);
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

void SDPipeline::set_lora(const std::string& path, float multiplier) {
    if (!impl_) return;
    // If path is empty, clear all loras
    if (path.empty()) {
        impl_->lora_paths.clear();
        impl_->lora_entries.clear();
        return;
    }
    impl_->lora_paths.push_back(path);
    sd_lora_t entry;
    entry.is_high_noise = false;
    entry.multiplier    = multiplier;
    entry.path          = impl_->lora_paths.back().c_str();
    impl_->lora_entries.push_back(entry);
}

void SDPipeline::clear_loras() {
    if (!impl_) return;
    impl_->lora_paths.clear();
    impl_->lora_entries.clear();
}

void SDPipeline::set_ipadapter(const std::string& model_path,
                                const std::string& clip_vision_path,
                                const std::string& image_path,
                                float weight) {
    if (!impl_) return;
    if (model_path.empty() || clip_vision_path.empty() || image_path.empty()) {
        impl_->has_ip_adapter_image = false;
        impl_->ip_adapter_image_data.clear();
        return;
    }
    // 1) 参考图（OpenCV → RGB）
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::fprintf(stderr, "[C++ gen] set_ipadapter: failed to read image %s\n", image_path.c_str());
        impl_->has_ip_adapter_image = false;
        return;
    }
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    impl_->ip_adapter_image_data.assign(rgb.data, rgb.data + rgb.total() * rgb.channels());
    impl_->ip_adapter_image.width   = rgb.cols;
    impl_->ip_adapter_image.height  = rgb.rows;
    impl_->ip_adapter_image.channel = rgb.channels();
    impl_->ip_adapter_image.data    = impl_->ip_adapter_image_data.data();
    impl_->has_ip_adapter_image     = true;
    impl_->ip_adapter_strength      = weight;

    // 2) IP-Adapter / CLIP-Vision 是 sd_ctx_params_t 字段 → 需要重载 context
    if (impl_->config.ip_adapter_path != model_path ||
        impl_->config.clip_vision_path != clip_vision_path) {
        impl_->config.ip_adapter_path  = model_path;
        impl_->config.clip_vision_path = clip_vision_path;
        std::fprintf(stderr, "[C++ gen] set_ipadapter: reloading ctx with ip_adapter=%s clip_vision=%s\n",
                     model_path.c_str(), clip_vision_path.c_str());
        load(impl_->config);
    }
    std::fprintf(stderr, "[C++ gen] set_ipadapter: native ip-adapter ready, strength=%.2f\n", weight);
}

void SDPipeline::set_ipadapter_enabled(bool enabled, float weight) {
    if (!impl_) return;
    if (!enabled) {
        impl_->has_ip_adapter_image = false;
        impl_->ip_adapter_image_data.clear();
    }
    if (enabled && weight > 0.0f) {
        impl_->ip_adapter_strength = weight;
    }
}

std::string SDPipeline::get_model_version_name() const {
    if (!impl_ || !impl_->ctx) return std::string();
    const char* name = sd_get_model_version_name(impl_->ctx);
    return name ? std::string(name) : std::string();
}

void SDPipeline::set_init_image(const std::string& image_path, float strength) {
    if (!impl_) return;
    if (image_path.empty()) {
        impl_->has_init_image = false;
        impl_->init_image_data.clear();
        impl_->init_image = sd_image_t{};
        return;
    }
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::fprintf(stderr, "[C++ gen] set_init_image: failed to read %s\n", image_path.c_str());
        impl_->has_init_image = false;
        return;
    }
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    impl_->init_image_data.assign(rgb.data, rgb.data + rgb.total() * rgb.channels());
    impl_->init_image.width   = rgb.cols;
    impl_->init_image.height  = rgb.rows;
    impl_->init_image.channel = rgb.channels();
    impl_->init_image.data    = impl_->init_image_data.data();
    impl_->has_init_image     = true;
    impl_->init_strength      = strength;
    std::fprintf(stderr, "[C++ gen] set_init_image: %s (%dx%d) strength=%.2f\n",
                 image_path.c_str(), rgb.cols, rgb.rows, strength);
}

bool SDPipeline::load_control_net(const std::string& path) {
    if (!impl_ || !impl_->ctx) return false;
    if (path.empty()) {
        return sd_ctx_unload_control_net(impl_->ctx);
    }
    bool ok = sd_ctx_load_control_net(impl_->ctx, path.c_str());
    std::fprintf(stderr, "[C++ gen] load_control_net: %s ok=%d\n", path.c_str(), ok ? 1 : 0);
    return ok;
}

void SDPipeline::set_control_image(const std::string& image_path, float strength) {
    if (!impl_) return;
    if (image_path.empty()) {
        impl_->has_control_image = false;
        impl_->control_image_data.clear();
        impl_->control_image = sd_image_t{};
        return;
    }
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::fprintf(stderr, "[C++ gen] set_control_image: failed to read %s\n", image_path.c_str());
        impl_->has_control_image = false;
        return;
    }
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    impl_->control_image_data.assign(rgb.data, rgb.data + rgb.total() * rgb.channels());
    impl_->control_image.width   = rgb.cols;
    impl_->control_image.height  = rgb.rows;
    impl_->control_image.channel = rgb.channels();
    impl_->control_image.data    = impl_->control_image_data.data();
    impl_->has_control_image     = true;
    impl_->control_strength      = strength;
    std::fprintf(stderr, "[C++ gen] set_control_image: %s (%dx%d) strength=%.2f\n",
                 image_path.c_str(), rgb.cols, rgb.rows, strength);
}

void SDPipeline::set_mask(const std::string& mask_path) {
    if (!impl_) return;
    if (mask_path.empty()) {
        impl_->has_mask_image = false;
        impl_->mask_image_data.clear();
        impl_->mask_image = sd_image_t{};
        return;
    }
    cv::Mat img = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::fprintf(stderr, "[C++ gen] set_mask: failed to read %s\n", mask_path.c_str());
        impl_->has_mask_image = false;
        return;
    }
    impl_->mask_image_data.assign(img.data, img.data + img.total() * img.channels());
    impl_->mask_image.width   = img.cols;
    impl_->mask_image.height  = img.rows;
    impl_->mask_image.channel = img.channels();
    impl_->mask_image.data    = impl_->mask_image_data.data();
    impl_->has_mask_image     = true;
    std::fprintf(stderr, "[C++ gen] set_mask: %s (%dx%d)\n", mask_path.c_str(), img.cols, img.rows);
}

void SDPipeline::set_batch_count(int n) {
    if (!impl_) return;
    impl_->batch_count = n > 0 ? n : 1;
}

void SDPipeline::set_clip_skip(int n) {
    if (!impl_) return;
    impl_->clip_skip = n;
}

void SDPipeline::set_flow_shift(float shift) {
    if (!impl_) return;
    impl_->flow_shift = shift;
}

int SDPipeline::clip_vision_encode(const std::string& image_path) {
    if (!impl_ || !impl_->ctx) return -1;
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::fprintf(stderr, "[C++ gen] clip_vision_encode: failed to read %s\n", image_path.c_str());
        return -2;
    }
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    sd_image_t image{};
    image.width   = rgb.cols;
    image.height  = rgb.rows;
    image.channel = rgb.channels();
    image.data    = rgb.data;
    bool ok       = sd_clip_vision_encode(impl_->ctx, &image);
    return ok ? 0 : -3;
}

void SDPipeline::set_rescale_cfg(bool enabled, float multiplier) {
    if (!impl_) return;
    impl_->rescale_cfg_enabled    = enabled;
    impl_->rescale_cfg_multiplier = multiplier;
}

void SDPipeline::set_video_cfg(bool enabled, int mode, float min_cfg) {
    if (!impl_) return;
    impl_->video_cfg_enabled = enabled;
    impl_->video_cfg_mode    = mode;
    impl_->video_cfg_min     = min_cfg;
}

void SDPipeline::set_sigma_range(bool enabled, float sigma_min, float sigma_max) {
    if (!impl_) return;
    impl_->sigma_range_enabled = enabled;
    impl_->sigma_range_min     = sigma_min;
    impl_->sigma_range_max     = sigma_max;
}

void SDPipeline::set_area_conds(const char* prompts_sep, const char* rects_csv, const char* strengths_csv) {
    if (!impl_) return;
    impl_->area_prompt_storage.clear();
    impl_->area_conds.clear();
    if (!prompts_sep || !*prompts_sep) return;

    std::vector<std::string> ps;
    {
        std::string cur;
        for (const char* p = prompts_sep; *p; p++) {
            if (*p == '\n') {
                ps.push_back(cur);
                cur.clear();
            } else {
                cur += *p;
            }
        }
        ps.push_back(cur);
    }
    auto split = [](const char* s) {
        std::vector<std::string> o;
        if (!s) return o;
        std::string cur;
        for (const char* p = s; *p; p++) {
            if (*p == ',') {
                o.push_back(cur);
                cur.clear();
            } else {
                cur += *p;
            }
        }
        o.push_back(cur);
        return o;
    };
    std::vector<std::string> rects = split(rects_csv);
    std::vector<std::string> strs  = split(strengths_csv);

    impl_->area_prompt_storage.reserve(ps.size());
    for (size_t i = 0; i < ps.size(); i++) {
        impl_->area_prompt_storage.push_back(ps[i]);
    }
    for (size_t i = 0; i < impl_->area_prompt_storage.size(); i++) {
        sd_area_cond_t ac{};
        ac.prompt = impl_->area_prompt_storage[i].c_str();
        if (i * 4 + 3 < rects.size()) {
            ac.x = atoi(rects[i * 4 + 0].c_str());
            ac.y = atoi(rects[i * 4 + 1].c_str());
            ac.w = atoi(rects[i * 4 + 2].c_str());
            ac.h = atoi(rects[i * 4 + 3].c_str());
        }
        ac.strength = (i < strs.size() && !strs[i].empty()) ? (float)atof(strs[i].c_str()) : 1.0f;
        impl_->area_conds.push_back(ac);
    }
}

void SDPipeline::set_wtype(int wtype) {
    if (!impl_ || wtype < 0) return;
    impl_->config.wtype = wtype;
    load(impl_->config);  // wtype 是加载期参数 → 重载
}

void SDPipeline::set_flash_attn(bool enabled) {
    if (!impl_) return;
    if (impl_->config.diffusion_flash_attn == enabled) return;
    impl_->config.diffusion_flash_attn = enabled;
    load(impl_->config);  // flash_attn 是加载期参数 → 重载
}

std::vector<Image> SDPipeline::generate(const ImageGenerationParams& params) {
    std::vector<Image> results;
    if (!impl_ || !impl_->ctx) {
        return results;
    }

    sd_img_gen_params_t img_params;
    sd_img_gen_params_init(&img_params);

    // img2img 时若未指定宽高，则用 init image 的尺寸
    int eff_w = params.width;
    int eff_h = params.height;
    if (impl_->has_init_image) {
        if (eff_w <= 0) eff_w = impl_->init_image.width;
        if (eff_h <= 0) eff_h = impl_->init_image.height;
    }
    if (eff_w <= 0) eff_w = 1024;
    if (eff_h <= 0) eff_h = 1024;

    img_params.prompt          = params.prompt.c_str();
    img_params.negative_prompt = params.negative_prompt.c_str();
    img_params.width           = eff_w;
    img_params.height          = eff_h;
    img_params.clip_skip       = impl_->clip_skip >= 0 ? impl_->clip_skip : params.clip_skip;
    if (impl_->flow_shift > 0.0f) {
        img_params.sample_params.flow_shift = impl_->flow_shift;
    }
    img_params.seed            = params.seed;
    img_params.batch_count     = impl_->batch_count;

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

    // LoRA: prefer per-call loras; fall back to persistent stored loras
    if (!params.loras.empty()) {
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
    }
    // else: keep persistent loras loaded via set_lora()
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
    // Cap VAE tile size: 128 latent pixels = 1024 output pixels (scale=8)
    // Prevents OOM on consumer GPUs when decoding large tiles.
    if (img_params.vae_tiling_params.tile_size_x > 128) {
        img_params.vae_tiling_params.tile_size_x = 128;
    }
    if (img_params.vae_tiling_params.tile_size_y > 128) {
        img_params.vae_tiling_params.tile_size_y = 128;
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

    // RescaleCFG
    img_params.rescale_cfg.enabled = impl_->rescale_cfg_enabled;
    if (impl_->rescale_cfg_enabled) {
        img_params.rescale_cfg.multiplier = impl_->rescale_cfg_multiplier;
    }

    // Video CFG guidance
    img_params.video_cfg.enabled = impl_->video_cfg_enabled;
    if (impl_->video_cfg_enabled) {
        img_params.video_cfg.mode    = impl_->video_cfg_mode;
        img_params.video_cfg.min_cfg = impl_->video_cfg_min;
    }

    // Sigma range override
    img_params.sigma_range.enabled = impl_->sigma_range_enabled;
    if (impl_->sigma_range_enabled) {
        img_params.sigma_range.sigma_min = impl_->sigma_range_min;
        img_params.sigma_range.sigma_max = impl_->sigma_range_max;
    }

    // Area conditioning
    img_params.area_conds      = impl_->area_conds.empty() ? nullptr : impl_->area_conds.data();
    img_params.area_cond_count = (int)impl_->area_conds.size();

    // Native IP-Adapter (sd.cpp 原生实现)
    if (impl_->has_ip_adapter_image) {
        img_params.ip_adapter_image    = impl_->ip_adapter_image;
        img_params.ip_adapter_strength = impl_->ip_adapter_strength;
    }

    // img2img init image
    if (impl_->has_init_image) {
        img_params.init_image = impl_->init_image;
        img_params.strength   = impl_->init_strength;
    }

    // ControlNet control image
    if (impl_->has_control_image) {
        img_params.control_image    = impl_->control_image;
        img_params.control_strength = impl_->control_strength;
    }

    // inpainting mask
    if (impl_->has_mask_image) {
        img_params.mask_image = impl_->mask_image;
    }

    sd_image_t* images = nullptr;
    int num_images = 0;
    std::fprintf(stderr, "[C++ gen] calling generate_image...\n");
    bool ok = generate_image(impl_->ctx, &img_params, &images, &num_images);
    std::fprintf(stderr, "[C++ gen] generate_image returned ok=%d images=%p num=%d\n", ok, (void*)images, num_images);
    if (!ok || !images || num_images == 0) {
        return results;
    }

    for (int i = 0; i < num_images; i++) {
        Image img;
        img.width    = static_cast<int>(images[i].width);
        img.height   = static_cast<int>(images[i].height);
        img.channels = static_cast<int>(images[i].channel);
        const size_t bytes = static_cast<size_t>(img.width) * img.height * img.channels;
        if (images[i].data != nullptr && bytes > 0) {
            img.data.assign(images[i].data, images[i].data + bytes);
        }
        results.push_back(std::move(img));
    }

    free_sd_images(images, num_images);

    // ADetailer post-processing（作用于第一张）
    if (results.empty() || results[0].data.empty()) {
        return results;
    }
    if (params.adetailer_enabled && !params.ad_model_path.empty()) {
        std::fprintf(stderr, "[C++ gen] applying ADetailer with model=%s\n", params.ad_model_path.c_str());
        sd_adetailer_params_t ad_params{};
        ad_params.prompt          = params.ad_prompt.empty() ? nullptr : params.ad_prompt.c_str();
        ad_params.negative_prompt = params.ad_negative_prompt.empty() ? nullptr : params.ad_negative_prompt.c_str();
        ad_params.extra_ad_args   = nullptr;

        sd_image_t input_img;
        input_img.width  = static_cast<int>(results[0].width);
        input_img.height = static_cast<int>(results[0].height);
        input_img.channel = static_cast<int>(results[0].channels);
        input_img.data   = results[0].data.data();

        sd_img_gen_params_t inpaint_params;
        sd_img_gen_params_init(&inpaint_params);
        inpaint_params.width = params.ad_inpaint_width;
        inpaint_params.height   = params.ad_inpaint_height;
        inpaint_params.strength = params.ad_denoising_strength;

        adetailer_ctx_t* ad_ctx = new_adetailer_ctx(params.ad_model_path.c_str(),
                                                     impl_->n_threads,
                                                     nullptr,
                                                     nullptr);
        if (ad_ctx) {
            sd_image_t* detailed_images = nullptr;
            int detailed_count = 0;
            bool ad_ok = adetail_image(ad_ctx, impl_->ctx, input_img,
                                        &ad_params, &inpaint_params,
                                        &detailed_images, &detailed_count);
            if (ad_ok && detailed_count > 0 && detailed_images && detailed_images[0].data) {
                results[0].width    = static_cast<int>(detailed_images[0].width);
                results[0].height   = static_cast<int>(detailed_images[0].height);
                results[0].channels = static_cast<int>(detailed_images[0].channel);
                size_t ad_bytes = static_cast<size_t>(results[0].width) * results[0].height * results[0].channels;
                results[0].data.assign(detailed_images[0].data, detailed_images[0].data + ad_bytes);
                std::fprintf(stderr, "[C++ gen] ADetailer applied: %dx%d\n", results[0].width, results[0].height);
            } else {
                std::fprintf(stderr, "[C++ gen] ADetailer failed or returned no images\n");
            }
            free_sd_images(detailed_images, detailed_count);
            free_adetailer_ctx(ad_ctx);
        } else {
            std::fprintf(stderr, "[C++ gen] new_adetailer_ctx failed\n");
        }
    }

    return results;
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

// Insert a numeric suffix before the file extension: a.png -> a_00001.png
static std::string indexed_path(const char* path, int idx) {
    std::string p(path);
    char buf[16];
    std::snprintf(buf, sizeof(buf), "_%05d", idx);
    size_t dot = p.find_last_of('.');
    if (dot == std::string::npos) return p + buf;
    return p.substr(0, dot) + buf + p.substr(dot);
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
    return sd_pipeline_generate_full(pipeline, prompt, negative_prompt,
                                     width, height,
                                     hires != 0 ? hires_width : 0,
                                     hires != 0 ? hires_height : 0,
                                     steps, cfg, sample_method, scheduler, seed,
                                     vae_tiling, vae_tile_size, vae_tile_overlap,
                                     hires_steps, hires_strength,
                                     freeu, freeu_b1, freeu_b2,
                                     sag, sag_scale,
                                     0.0f, 0.0f, 0,
                                     0.0f, 0,
                                     0.0f, 0, 0.0f,
                                     nullptr, nullptr, nullptr,
                                     output_path);
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
                              const char* output_path) {
    if (!pipeline || !prompt || !output_path) return -1;

    sd::SDPipeline* p = static_cast<sd::SDPipeline*>(pipeline);
    if (!p->is_loaded()) return -3;

    const bool hires_on = hires_width > 0 && hires_height > 0;
    int base_w = width;
    int base_h = height;
    if (hires_on && (base_w <= 0 || base_h <= 0)) {
        int64_t packed = sd_compute_hires_resolution(hires_width, hires_height);
        base_w = static_cast<int>(packed >> 32);
        base_h = static_cast<int>(packed & 0xFFFFFFFF);
        std::fprintf(stderr, "[C API] hires: target=%dx%d base=%dx%d\n",
                     hires_width, hires_height, base_w, base_h);
    }
    if (base_w <= 0) base_w = 1024;
    if (base_h <= 0) base_h = 1024;

    sd::ImageGenerationParams params;
    params.prompt          = prompt;
    params.negative_prompt = negative_prompt ? negative_prompt : "";
    params.width           = base_w;
    params.height          = base_h;
    params.steps           = steps > 0 ? steps : 20;
    params.cfg_scale       = cfg > 0.0f ? cfg : 7.0f;
    params.sample_method   = sample_method ? sample_method : "euler_a";
    params.scheduler       = scheduler ? scheduler : "discrete";
    params.seed            = seed;

    if (hires_on) {
        params.hires_enabled  = true;
        params.hires_width    = hires_width;
        params.hires_height   = hires_height;
        params.hires_steps    = hires_steps > 0 ? hires_steps : 20;
        params.hires_strength = hires_strength >= 0.0f && hires_strength <= 1.0f
                                ? hires_strength : 0.35f;
    }

    // Explicit tiling only; auto-tiling for large images is handled
    // inside SDPipeline::generate (single place).
    if (vae_tiling != 0) {
        params.vae_tiling       = true;
        params.vae_tile_size_x  = vae_tile_size > 0 ? vae_tile_size : 64;
        params.vae_tile_size_y  = params.vae_tile_size_x;
        params.vae_tile_overlap = vae_tile_overlap >= 0.0f && vae_tile_overlap <= 1.0f
                                  ? vae_tile_overlap : 0.5f;
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

    if (ad_model_path && ad_model_path[0] != '\0') {
        params.adetailer_enabled  = true;
        params.ad_model_path      = ad_model_path;
        params.ad_prompt          = ad_prompt ? ad_prompt : "";
        params.ad_negative_prompt = ad_negative_prompt ? ad_negative_prompt : "";
    }

    std::vector<sd::Image> images = p->generate(params);
    std::fprintf(stderr, "[C API] generate_full returned %zu image(s)\n", images.size());
    if (images.empty()) return -4;

    bool has_postproc = (clarity > 0.0f || sharpen_amount > 0.0f ||
                         smart_sharpen_strength > 0.0f || edge_sharpen_amount > 0.0f);

    for (size_t idx = 0; idx < images.size(); idx++) {
        sd::Image& image = images[idx];
        if (image.data.empty()) continue;
        if (has_postproc) {
            std::fprintf(stderr, "[C API] postproc: clarity=%.2f sharpen=%.2f(r=%d)"
                         " smart=%.2f(r=%d) edge=%.2f(r=%d,t=%.2f)\n",
                         clarity, sharpen_amount, sharpen_radius,
                         smart_sharpen_strength, smart_sharpen_radius,
                         edge_sharpen_amount, edge_sharpen_radius, edge_sharpen_threshold);
            postproc::Params pp;
            pp.clarity                = clarity;
            pp.sharpen_amount         = sharpen_amount;
            pp.sharpen_radius         = sharpen_radius > 0 ? sharpen_radius : 1;
            pp.smart_sharpen_strength = smart_sharpen_strength;
            pp.smart_sharpen_radius   = smart_sharpen_radius > 0 ? smart_sharpen_radius : 2;
            pp.edge_sharpen_amount    = edge_sharpen_amount;
            pp.edge_sharpen_radius    = edge_sharpen_radius > 0 ? edge_sharpen_radius : 2;
            pp.edge_sharpen_threshold = edge_sharpen_threshold >= 0.0f ? edge_sharpen_threshold : 0.3f;
            if (!postproc::apply(image.data.data(), image.width, image.height, image.channels, pp)) {
                std::fprintf(stderr, "[C API] postproc failed\n");
                return -5;
            }
        }

        std::string path = (images.size() > 1)
            ? indexed_path(output_path, static_cast<int>(idx))
            : std::string(output_path);
        if (!save_png(path.c_str(), image.data.data(), image.width, image.height, image.channels)) {
            std::fprintf(stderr, "[C API] save_png failed for %s\n", path.c_str());
            return -6;
        }
        std::fprintf(stderr, "[C API] saved %s (%dx%d, %d ch)\n", path.c_str(), image.width, image.height, image.channels);
    }

    return 0;
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
                                float smart_sharpen_strength,
                                int smart_sharpen_radius,
                                float edge_sharpen_amount,
                                int edge_sharpen_radius,
                                float edge_sharpen_threshold,
                                const char* output_path) {
    return sd_pipeline_generate_full(pipeline, prompt, negative_prompt,
                                     0, 0,
                                     target_width, target_height,
                                     steps, cfg, sample_method, scheduler, seed,
                                     vae_tiling, vae_tile_size, vae_tile_overlap,
                                     hires_steps, hires_strength,
                                     freeu, freeu_b1, freeu_b2,
                                     sag, sag_scale,
                                     clarity, sharpen_amount, sharpen_radius,
                                     smart_sharpen_strength, smart_sharpen_radius,
                                     edge_sharpen_amount, edge_sharpen_radius,
                                     edge_sharpen_threshold,
                                     nullptr, nullptr, nullptr,
                                     output_path);
}

int sd_pipeline_load_lora(sd_pipeline_t pipeline,
                           const char* lora_path,
                           float multiplier) {
    if (!pipeline) return -1;
    sd::SDPipeline* p = static_cast<sd::SDPipeline*>(pipeline);
    if (!p->is_loaded()) return -2;
    std::fprintf(stderr, "[C API] sd_pipeline_load_lora: path=%s mult=%.2f\n",
                 lora_path ? lora_path : "(null)", multiplier);
    if (!lora_path || lora_path[0] == '\0') {
        p->clear_loras();
    } else {
        p->set_lora(lora_path, multiplier);
    }
    return 0;
}

int sd_pipeline_set_ipadapter(sd_pipeline_t pipeline,
                               const char* model_path,
                               const char* clip_vision_path,
                               const char* image_path,
                               float weight) {
    if (!pipeline) return -1;
    sd::SDPipeline* p = static_cast<sd::SDPipeline*>(pipeline);
    if (!p->is_loaded()) return -2;
    std::fprintf(stderr, "[C API] sd_pipeline_set_ipadapter: model=%s clip=%s image=%s weight=%.2f\n",
                 model_path ? model_path : "(null)",
                 clip_vision_path ? clip_vision_path : "(null)",
                 image_path ? image_path : "(null)",
                 weight);
    p->set_ipadapter(model_path ? model_path : "",
                      clip_vision_path ? clip_vision_path : "",
                      image_path ? image_path : "",
                      weight);
    return 0;
}

int sd_pipeline_set_ipadapter_enabled(sd_pipeline_t pipeline,
                                       int enabled,
                                       float weight) {
    if (!pipeline) return -1;
    sd::SDPipeline* p = static_cast<sd::SDPipeline*>(pipeline);
    std::fprintf(stderr, "[C API] sd_pipeline_set_ipadapter_enabled: enabled=%d weight=%.2f\n",
                 enabled, weight);
    p->set_ipadapter_enabled(enabled != 0, weight);
    return 0;
}

int sd_pipeline_set_init_image(sd_pipeline_t pipeline,
                               const char* image_path,
                               float strength) {
    if (!pipeline) return -1;
    sd::SDPipeline* p = static_cast<sd::SDPipeline*>(pipeline);
    std::fprintf(stderr, "[C API] sd_pipeline_set_init_image: path=%s strength=%.2f\n",
                 image_path ? image_path : "(null)", strength);
    p->set_init_image(image_path ? image_path : "", strength);
    return 0;
}

int sd_pipeline_load_control_net(sd_pipeline_t pipeline, const char* path) {
    if (!pipeline) return -1;
    sd::SDPipeline* p = static_cast<sd::SDPipeline*>(pipeline);
    std::fprintf(stderr, "[C API] sd_pipeline_load_control_net: path=%s\n", path ? path : "(null)");
    return p->load_control_net(path ? path : "") ? 0 : -2;
}

int sd_pipeline_set_control_image(sd_pipeline_t pipeline,
                                  const char* image_path,
                                  float strength) {
    if (!pipeline) return -1;
    sd::SDPipeline* p = static_cast<sd::SDPipeline*>(pipeline);
    std::fprintf(stderr, "[C API] sd_pipeline_set_control_image: path=%s strength=%.2f\n",
                 image_path ? image_path : "(null)", strength);
    p->set_control_image(image_path ? image_path : "", strength);
    return 0;
}

int sd_pipeline_set_mask(sd_pipeline_t pipeline, const char* mask_path) {
    if (!pipeline) return -1;
    sd::SDPipeline* p = static_cast<sd::SDPipeline*>(pipeline);
    std::fprintf(stderr, "[C API] sd_pipeline_set_mask: path=%s\n", mask_path ? mask_path : "(null)");
    p->set_mask(mask_path ? mask_path : "");
    return 0;
}

int sd_pipeline_set_batch_count(sd_pipeline_t pipeline, int n) {
    if (!pipeline) return -1;
    sd::SDPipeline* p = static_cast<sd::SDPipeline*>(pipeline);
    std::fprintf(stderr, "[C API] sd_pipeline_set_batch_count: n=%d\n", n);
    p->set_batch_count(n);
    return 0;
}

int sd_pipeline_set_clip_skip(sd_pipeline_t pipeline, int n) {
    if (!pipeline) return -1;
    static_cast<sd::SDPipeline*>(pipeline)->set_clip_skip(n);
    return 0;
}

int sd_pipeline_set_flow_shift(sd_pipeline_t pipeline, float shift) {
    if (!pipeline) return -1;
    static_cast<sd::SDPipeline*>(pipeline)->set_flow_shift(shift);
    return 0;
}

int sd_pipeline_set_wtype(sd_pipeline_t pipeline, int wtype) {
    if (!pipeline) return -1;
    static_cast<sd::SDPipeline*>(pipeline)->set_wtype(wtype);
    return 0;
}

int sd_pipeline_set_flash_attn(sd_pipeline_t pipeline, int enabled) {
    if (!pipeline) return -1;
    static_cast<sd::SDPipeline*>(pipeline)->set_flash_attn(enabled != 0);
    return 0;
}

int sd_pipeline_set_area_conds(sd_pipeline_t pipeline, const char* prompts_sep, const char* rects_csv, const char* strengths_csv) {
    if (!pipeline) return -1;
    static_cast<sd::SDPipeline*>(pipeline)->set_area_conds(prompts_sep, rects_csv, strengths_csv);
    return 0;
}

int sd_pipeline_clip_vision_encode(sd_pipeline_t pipeline, const char* image_path) {
    if (!pipeline || !image_path) return -1;
    return static_cast<sd::SDPipeline*>(pipeline)->clip_vision_encode(image_path);
}

int sd_pipeline_set_rescale_cfg(sd_pipeline_t pipeline, int enabled, float multiplier) {
    if (!pipeline) return -1;
    static_cast<sd::SDPipeline*>(pipeline)->set_rescale_cfg(enabled != 0, multiplier);
    return 0;
}

int sd_pipeline_set_video_cfg(sd_pipeline_t pipeline, int enabled, int mode, float min_cfg) {
    if (!pipeline) return -1;
    static_cast<sd::SDPipeline*>(pipeline)->set_video_cfg(enabled != 0, mode, min_cfg);
    return 0;
}

int sd_pipeline_set_sigma_range(sd_pipeline_t pipeline, int enabled, float sigma_min, float sigma_max) {
    if (!pipeline) return -1;
    static_cast<sd::SDPipeline*>(pipeline)->set_sigma_range(enabled != 0, sigma_min, sigma_max);
    return 0;
}

const char* sd_pipeline_get_model_version_name(sd_pipeline_t pipeline) {
    static std::string name;  // CLI 单线程，缓存即可；返回空串而非 NULL
    name.clear();
    if (pipeline) {
        sd::SDPipeline* p = static_cast<sd::SDPipeline*>(pipeline);
        name = p->get_model_version_name();
    }
    return name.c_str();
}

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
                                     const char* output_path) {
    return sd_pipeline_generate_full(pipeline, prompt, negative_prompt,
                                     width, height,
                                     hires != 0 ? hires_width : 0,
                                     hires != 0 ? hires_height : 0,
                                     steps, cfg, sample_method, scheduler, seed,
                                     vae_tiling, vae_tile_size, vae_tile_overlap,
                                     hires_steps, hires_strength,
                                     freeu, freeu_b1, freeu_b2,
                                     sag, sag_scale,
                                     0.0f, 0.0f, 0,
                                     0.0f, 0,
                                     0.0f, 0, 0.0f,
                                     ad_model_path, ad_prompt, ad_negative_prompt,
                                     output_path);
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

int sd_resize_image(const char* input_path, const char* output_path,
                    int width, int height) {
    if (!input_path || !output_path || width <= 0 || height <= 0) return -1;
    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::fprintf(stderr, "[C API] sd_resize_image: failed to read %s\n", input_path);
        return -2;
    }
    cv::Mat resized, rgb;
    cv::resize(img, resized, cv::Size(width, height), 0, 0, cv::INTER_LANCZOS4);
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    if (!save_png(output_path, rgb.data, rgb.cols, rgb.rows, rgb.channels())) {
        std::fprintf(stderr, "[C API] sd_resize_image: failed to write %s\n", output_path);
        return -3;
    }
    std::fprintf(stderr, "[C API] sd_resize_image: %s -> %s (%dx%d)\n",
                 input_path, output_path, width, height);
    return 0;
}

int sd_scale_image(const char* input_path, const char* output_path, float scale_by) {
    if (!input_path || !output_path || scale_by <= 0.0f) return -1;
    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::fprintf(stderr, "[C API] sd_scale_image: failed to read %s\n", input_path);
        return -2;
    }
    int w = (int)(img.cols * scale_by + 0.5f);
    int h = (int)(img.rows * scale_by + 0.5f);
    return sd_resize_image(input_path, output_path, w, h);
}

int sd_make_solid_image(const char* output_path, int width, int height,
                        int r, int g, int b) {
    if (!output_path || width <= 0 || height <= 0) return -1;
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(b, g, r));  // OpenCV is BGR
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    if (!save_png(output_path, rgb.data, rgb.cols, rgb.rows, rgb.channels())) {
        std::fprintf(stderr, "[C API] sd_make_solid_image: failed to write %s\n", output_path);
        return -3;
    }
    return 0;
}

int sd_pad_image(const char* input_path, const char* output_path,
                 int left, int top, int right, int bottom,
                 int r, int g, int b) {
    if (!input_path || !output_path || left < 0 || top < 0 || right < 0 || bottom < 0) return -1;
    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::fprintf(stderr, "[C API] sd_pad_image: failed to read %s\n", input_path);
        return -2;
    }
    cv::Mat out, rgb;
    cv::copyMakeBorder(img, out, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(b, g, r));
    cv::cvtColor(out, rgb, cv::COLOR_BGR2RGB);
    if (!save_png(output_path, rgb.data, rgb.cols, rgb.rows, rgb.channels())) {
        std::fprintf(stderr, "[C API] sd_pad_image: failed to write %s\n", output_path);
        return -3;
    }
    return 0;
}

int sd_blur_image(const char* input_path, const char* output_path, float sigma) {
    if (!input_path || !output_path || sigma <= 0.0f) return -1;
    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::fprintf(stderr, "[C API] sd_blur_image: failed to read %s\n", input_path);
        return -2;
    }
    cv::Mat out, rgb;
    cv::GaussianBlur(img, out, cv::Size(0, 0), (double)sigma);
    cv::cvtColor(out, rgb, cv::COLOR_BGR2RGB);
    if (!save_png(output_path, rgb.data, rgb.cols, rgb.rows, rgb.channels())) {
        std::fprintf(stderr, "[C API] sd_blur_image: failed to write %s\n", output_path);
        return -3;
    }
    return 0;
}

int sd_batch_images(const char* path1, const char* path2, const char* output_path) {
    if (!path1 || !path2 || !output_path) return -1;
    cv::Mat a = cv::imread(path1, cv::IMREAD_COLOR);
    cv::Mat b = cv::imread(path2, cv::IMREAD_COLOR);
    if (a.empty() || b.empty()) {
        std::fprintf(stderr, "[C API] sd_batch_images: failed to read inputs\n");
        return -2;
    }
    cv::Mat b2, out, rgb;
    cv::resize(b, b2, a.size());
    cv::vconcat(a, b2, out);
    cv::cvtColor(out, rgb, cv::COLOR_BGR2RGB);
    if (!save_png(output_path, rgb.data, rgb.cols, rgb.rows, rgb.channels())) {
        std::fprintf(stderr, "[C API] sd_batch_images: failed to write %s\n", output_path);
        return -3;
    }
    return 0;
}

int sd_composite_masked(const char* dest_path, const char* src_path,
                        const char* mask_path, const char* output_path,
                        int x, int y) {
    if (!dest_path || !src_path || !output_path || x < 0 || y < 0) return -1;
    cv::Mat dst = cv::imread(dest_path, cv::IMREAD_COLOR);
    cv::Mat src = cv::imread(src_path, cv::IMREAD_COLOR);
    if (dst.empty() || src.empty()) {
        std::fprintf(stderr, "[C API] sd_composite_masked: failed to read inputs\n");
        return -2;
    }
    if (x + src.cols > dst.cols || y + src.rows > dst.rows) {
        std::fprintf(stderr, "[C API] sd_composite_masked: source out of bounds\n");
        return -4;
    }
    cv::Rect roi(x, y, src.cols, src.rows);
    if (mask_path && mask_path[0] != '\0') {
        cv::Mat m = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
        if (m.empty()) {
            std::fprintf(stderr, "[C API] sd_composite_masked: failed to read mask\n");
            return -3;
        }
        cv::resize(m, m, src.size());
        src.copyTo(dst(roi), m);
    } else {
        src.copyTo(dst(roi));
    }
    cv::Mat rgb;
    cv::cvtColor(dst, rgb, cv::COLOR_BGR2RGB);
    if (!save_png(output_path, rgb.data, rgb.cols, rgb.rows, rgb.channels())) return -5;
    return 0;
}

int sd_crop_image(const char* input_path, const char* output_path,
                  int x, int y, int width, int height) {
    if (!input_path || !output_path || x < 0 || y < 0 || width <= 0 || height <= 0) return -1;
    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::fprintf(stderr, "[C API] sd_crop_image: failed to read %s\n", input_path);
        return -2;
    }
    if (x + width > img.cols || y + height > img.rows) {
        std::fprintf(stderr, "[C API] sd_crop_image: crop out of bounds\n");
        return -4;
    }
    cv::Mat out = img(cv::Rect(x, y, width, height)).clone();
    cv::Mat rgb;
    cv::cvtColor(out, rgb, cv::COLOR_BGR2RGB);
    if (!save_png(output_path, rgb.data, rgb.cols, rgb.rows, rgb.channels())) return -3;
    return 0;
}

int sd_rotate_image(const char* input_path, const char* output_path, int degrees) {
    if (!input_path || !output_path) return -1;
    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img.empty()) return -2;
    int d = ((degrees % 360) + 360) % 360;
    cv::Mat out, rgb;
    if (d == 90)       cv::rotate(img, out, cv::ROTATE_90_COUNTERCLOCKWISE);
    else if (d == 180) cv::rotate(img, out, cv::ROTATE_180);
    else if (d == 270) cv::rotate(img, out, cv::ROTATE_90_CLOCKWISE);
    else               out = img;
    cv::cvtColor(out, rgb, cv::COLOR_BGR2RGB);
    if (!save_png(output_path, rgb.data, rgb.cols, rgb.rows, rgb.channels())) return -3;
    return 0;
}

int sd_flip_image(const char* input_path, const char* output_path, int method) {
    if (!input_path || !output_path) return -1;
    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img.empty()) return -2;
    cv::Mat out, rgb;
    cv::flip(img, out, method == 0 ? 0 : 1);
    cv::cvtColor(out, rgb, cv::COLOR_BGR2RGB);
    if (!save_png(output_path, rgb.data, rgb.cols, rgb.rows, rgb.channels())) return -3;
    return 0;
}

int sd_blend_images(const char* path1, const char* path2, const char* output_path, float factor) {
    if (!path1 || !path2 || !output_path) return -1;
    cv::Mat a = cv::imread(path1, cv::IMREAD_COLOR);
    cv::Mat b = cv::imread(path2, cv::IMREAD_COLOR);
    if (a.empty() || b.empty()) return -2;
    cv::Mat b2, out, rgb;
    cv::resize(b, b2, a.size());
    cv::addWeighted(a, 1.0 - factor, b2, factor, 0.0, out);
    cv::cvtColor(out, rgb, cv::COLOR_BGR2RGB);
    if (!save_png(output_path, rgb.data, rgb.cols, rgb.rows, rgb.channels())) return -3;
    return 0;
}

int sd_invert_image(const char* input_path, const char* output_path) {
    if (!input_path || !output_path) return -1;
    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::fprintf(stderr, "[C API] sd_invert_image: failed to read %s\n", input_path);
        return -2;
    }
    cv::Mat inv, rgb;
    cv::bitwise_not(img, inv);
    cv::cvtColor(inv, rgb, cv::COLOR_BGR2RGB);
    if (!save_png(output_path, rgb.data, rgb.cols, rgb.rows, rgb.channels())) {
        std::fprintf(stderr, "[C API] sd_invert_image: failed to write %s\n", output_path);
        return -3;
    }
    return 0;
}

} // extern "C"
