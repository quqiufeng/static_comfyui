#include "sdcpp_adapter.h"

#include <stable-diffusion.h>

#include <cstddef>
#include <utility>

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

SDPipeline::SDPipeline() : impl_(std::make_unique<Impl>()) {}

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
    params.clip_l_path  = config.clip_l_path.c_str();
    params.clip_g_path  = config.clip_g_path.c_str();
    if (!config.vae_path.empty()) {
        params.vae_path = config.vae_path.c_str();
    }
    if (!config.diffusion_model_path.empty()) {
        params.diffusion_model_path = config.diffusion_model_path.c_str();
    }
    if (!config.llm_path.empty()) {
        params.llm_path = config.llm_path.c_str();
    }
    params.n_threads = config.n_threads;
    params.wtype     = SD_TYPE_COUNT;  // auto, matches sd-cli default
    params.flash_attn = config.flash_attn;
    params.diffusion_flash_attn = config.diffusion_flash_attn;

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
    img_params.seed            = params.seed;
    img_params.batch_count     = params.batch_count;

    img_params.sample_params.sample_steps     = params.steps;
    img_params.sample_params.guidance.txt_cfg = params.cfg_scale;
    img_params.sample_params.sample_method    = str_to_sample_method(params.sample_method.c_str());
    img_params.sample_params.scheduler        = str_to_scheduler(params.scheduler.c_str());
    if (img_params.sample_params.sample_method == SAMPLE_METHOD_COUNT) {
        img_params.sample_params.sample_method = EULER_A_SAMPLE_METHOD;
    }
    if (img_params.sample_params.scheduler == SCHEDULER_COUNT) {
        img_params.sample_params.scheduler = DISCRETE_SCHEDULER;
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
    }

    // HiRes Fix
    if (params.hires_enabled) {
        img_params.hires.enabled            = true;
        img_params.hires.target_width       = params.hires_width;
        img_params.hires.target_height      = params.hires_height;
        img_params.hires.steps              = params.hires_steps;
        img_params.hires.denoising_strength = params.hires_strength;
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
    bool ok = generate_image(impl_->ctx, &img_params, &images, &num_images);
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
