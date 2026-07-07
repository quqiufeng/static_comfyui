#include "sdcpp_adapter.h"

#include <stable-diffusion.h>

#include <cstddef>
#include <utility>

namespace sd {

class SDPipeline::Impl {
public:
    sd_ctx_t* ctx = nullptr;

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

    params.model_path    = config.model_path.c_str();
    params.clip_l_path   = config.clip_l_path.c_str();
    params.clip_g_path   = config.clip_g_path.c_str();
    if (!config.vae_path.empty()) {
        params.vae_path = config.vae_path.c_str();
    }
    params.n_threads = config.n_threads;
    params.wtype     = SD_TYPE_COUNT;  // auto, matches sd-cli default
    // Keep CUDA_RNG default (matches sd-cli). Do not override to STD_DEFAULT_RNG.

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
