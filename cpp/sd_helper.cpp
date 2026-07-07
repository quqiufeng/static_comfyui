/* sd_helper.cpp — stable-diffusion.cpp 关键函数实现参考
 * 按 api.md 文件行号从 /opt/stable-diffusion.cpp 复制完整代码 */
#include "stable-diffusion.h"
#include <cmath>
/* ============================================================= */
/* 1. sd_ctx_params_init  (src/stable-diffusion.cpp:2924)         */
/* ============================================================= */
void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params) {
    *sd_ctx_params                         = {};
    sd_ctx_params->vae_decode_only         = true;
    sd_ctx_params->free_params_immediately = true;
    sd_ctx_params->n_threads               = sd_get_num_physical_cores();
    sd_ctx_params->wtype                   = SD_TYPE_COUNT;
    sd_ctx_params->rng_type                = CUDA_RNG;
    sd_ctx_params->sampler_rng_type        = RNG_TYPE_COUNT;
    sd_ctx_params->prediction              = PREDICTION_COUNT;
    sd_ctx_params->lora_apply_mode         = LORA_APPLY_AUTO;
    sd_ctx_params->offload_params_to_cpu   = false;
    sd_ctx_params->max_vram                = 0.f;
    sd_ctx_params->enable_mmap             = false;
    sd_ctx_params->keep_clip_on_cpu        = false;
    sd_ctx_params->keep_control_net_on_cpu = false;
    sd_ctx_params->keep_vae_on_cpu         = false;
    sd_ctx_params->circular_x              = false;
    sd_ctx_params->circular_y              = false;
    sd_ctx_params->vae_format              = SD_VAE_FORMAT_AUTO;
    sd_ctx_params->backend                 = nullptr;
    sd_ctx_params->params_backend          = nullptr;
    sd_ctx_params->ipadapter_unet_mode     = false;
    sd_ctx_params->ipadapter_unet_weights_path = nullptr;
}

/* ============================================================= */
/* 2. sd_sample_params_init  (src/stable-diffusion.cpp:3040)     */
/* ============================================================= */
void sd_sample_params_init(sd_sample_params_t* sample_params) {
    *sample_params = {};
    sample_params->guidance.txt_cfg = 7.0f;
    sample_params->guidance.slg    = {nullptr, 0, 0, 0, 1.0f};
    sample_params->scheduler       = KARRAS_SCHEDULER;
    sample_params->sample_method   = EULER_A_SAMPLE_METHOD;
    sample_params->sample_steps    = 20;
    sample_params->eta             = 0.f;
    sample_params->shifted_timestep = -1;
    sample_params->custom_sigmas   = nullptr;
    sample_params->custom_sigmas_count = 0;
    sample_params->flow_shift      = INFINITY;
    sample_params->extra_sample_args = nullptr;
}

/* ============================================================= */
/* 3. sd_img_gen_params_init  (src/stable-diffusion.cpp:3100)   */
/* ============================================================= */
void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params) {
    *sd_img_gen_params = {};
    sd_sample_params_init(&sd_img_gen_params->sample_params);
    sd_img_gen_params->clip_skip                = -1;
    sd_img_gen_params->width                    = 512;
    sd_img_gen_params->height                   = 512;
    sd_img_gen_params->strength                 = 0.75f;
    sd_img_gen_params->seed                     = -1;
    sd_img_gen_params->batch_count              = 1;
    sd_img_gen_params->control_strength         = 0.5f;
    sd_img_gen_params->vae_tiling_params        = {false, false, 0, 0, 0.5f, 0, 0, nullptr};
    sd_img_gen_params->pm_params.style_strength = 0.8f;
    sd_img_gen_params->hires.enabled            = false;
    sd_img_gen_params->hires.upscaler           = SD_HIRES_UPSCALER_LATENT;
    sd_img_gen_params->hires.scale              = 2.f;
    sd_img_gen_params->hires.steps              = 0;
    sd_img_gen_params->hires.denoising_strength = 0.7f;
    sd_img_gen_params->freeu.enabled            = false;
    sd_img_gen_params->sag.enabled              = false;
    sd_img_gen_params->dynamic_cfg.enabled      = false;
    sd_img_gen_params->ipadapter_tokens         = nullptr;
    sd_img_gen_params->ipadapter_num_tokens     = 0;
    sd_img_gen_params->ipadapter_weight         = 0.5f;
    sd_img_gen_params->ipadapter_unet_mode      = false;
    sd_img_gen_params->ipadapter_unet_weights_path = nullptr;
}

/* ============================================================= */
/* 4. sd_ctx_t struct    (src/stable-diffusion.cpp:3213)          */
/* ============================================================= */
struct sd_ctx_t {
    StableDiffusionGGML* sd = nullptr;
};
sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params) {
    sd_ctx_t* sd_ctx = (sd_ctx_t*)malloc(sizeof(sd_ctx_t));
    if (sd_ctx == nullptr) {
        return nullptr;
    }

    sd_ctx->sd = new StableDiffusionGGML();
    if (sd_ctx->sd == nullptr) {
        free(sd_ctx);
        return nullptr;
    }

    if (!sd_ctx->sd->init(sd_ctx_params)) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
        free(sd_ctx);
        return nullptr;
    }
    return sd_ctx;
}
// ───
void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx->sd != nullptr) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
    }
    free(sd_ctx);
}
// ─── free_sd_audio ───
void free_sd_audio(sd_audio_t* audio) {
    if (audio == nullptr) {
        return;
    }
    free(audio->data);
    audio->data = nullptr;
    free(audio);
}
// ─── sd_ctx_supports_image_generation ───
SD_API bool sd_ctx_supports_image_generation(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    return sd_version_supports_image_generation(sd_ctx->sd->version);
}
// ─── sd_get_default_sample_method ───
enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* sd_ctx) {
    if (sd_ctx != nullptr && sd_ctx->sd != nullptr) {
        if (sd_version_is_pid(sd_ctx->sd->version)) {
            return LCM_SAMPLE_METHOD;
        }
        if (sd_version_is_dit(sd_ctx->sd->version)) {
            return EULER_SAMPLE_METHOD;
        }
    }
    return EULER_A_SAMPLE_METHOD;
}
// ─── generate_image ───
SD_API sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params) {
    if (sd_ctx == nullptr || sd_img_gen_params == nullptr) {
        return nullptr;
    }

    int64_t t0                    = ggml_time_ms();
    sd_ctx->sd->vae_tiling_params = sd_img_gen_params->vae_tiling_params;
    GenerationRequest request(sd_ctx, sd_img_gen_params);
    LOG_INFO("generate_image %dx%d", request.width, request.height);

    sd_ctx->sd->rng->manual_seed(request.seed);
    sd_ctx->sd->sampler_rng->manual_seed(request.seed);
    sd_ctx->sd->set_flow_shift(sd_img_gen_params->sample_params.flow_shift);
    sd_ctx->sd->apply_loras(sd_img_gen_params->loras, sd_img_gen_params->lora_count);
    sd_ctx->sd->freeu_enabled = sd_img_gen_params->freeu.enabled;
    if (sd_img_gen_params->freeu.enabled) {
        sd_ctx->sd->freeu_b1 = sd_img_gen_params->freeu.b1;
        sd_ctx->sd->freeu_b2 = sd_img_gen_params->freeu.b2;
        sd_ctx->sd->freeu_s1 = sd_img_gen_params->freeu.s1;
        sd_ctx->sd->freeu_s2 = sd_img_gen_params->freeu.s2;
    }
    sd_ctx->sd->sag_enabled = sd_img_gen_params->sag.enabled;
    if (sd_img_gen_params->sag.enabled) {
        sd_ctx->sd->sag_scale = sd_img_gen_params->sag.scale;
    }
    sd_ctx->sd->dynamic_cfg_enabled = sd_img_gen_params->dynamic_cfg.enabled;
    if (sd_img_gen_params->dynamic_cfg.enabled) {
        sd_ctx->sd->dynamic_cfg_percentile = sd_img_gen_params->dynamic_cfg.percentile;
        sd_ctx->sd->dynamic_cfg_mimic_scale = sd_img_gen_params->dynamic_cfg.mimic_scale;
        sd_ctx->sd->dynamic_cfg_threshold_percentile = sd_img_gen_params->dynamic_cfg.threshold_percentile;
    }

    ImageVaeAxesGuard axes_guard(sd_ctx, sd_img_gen_params, request);

    SamplePlan plan(sd_ctx, sd_img_gen_params, request);
    auto latents_opt = prepare_image_generation_latents(sd_ctx,
                                                        sd_img_gen_params,
                                                        &request,
                                                        &plan);
    if (!latents_opt.has_value()) {
        return nullptr;
    }
    ImageGenerationLatents latents = std::move(*latents_opt);

    auto embeds_opt = prepare_image_generation_embeds(sd_ctx,
                                                      sd_img_gen_params,
                                                      &request,
                                                      &plan,
                                                      &latents);
    if (!embeds_opt.has_value()) {
        return nullptr;
    }
    ImageGenerationEmbeds embeds = std::move(*embeds_opt);

    // Set up IPAdapter UNet image_embeds tensor for cross-attention injection
    if (g_ipadapter_image_embeds_tensor != nullptr && embeds.ipadapter_tokens != nullptr && embeds.ipadapter_num_tokens > 0) {
        int n_tokens    = embeds.ipadapter_num_tokens;
        int ctx_dim     = (int)g_ipadapter_image_embeds_tensor->ne[0];
        size_t num_floats = (size_t)n_tokens * ctx_dim;
        std::vector<float> scaled(num_floats);
        const float* src = embeds.ipadapter_tokens;
        float weight     = embeds.ipadapter_weight;
        for (int i = 0; i < n_tokens; i++) {
            for (int j = 0; j < ctx_dim; j++) {
                scaled[i * ctx_dim + j] = src[i * ctx_dim + j] * weight;
            }
        }
        ggml_backend_tensor_set(g_ipadapter_image_embeds_tensor, scaled.data(), 0, num_floats * sizeof(float));
        g_ipadapter_image_embeds = g_ipadapter_image_embeds_tensor;
        LOG_INFO("IPAdapter UNet: set image_embeds tensor [1, %d, %d] with weight=%.2f", n_tokens, ctx_dim, weight);
    } else {
        g_ipadapter_image_embeds = nullptr;
    }

    std::vector<sd::Tensor<float>> final_latents;
    int64_t denoise_start = ggml_time_ms();
    for (int b = 0; b < request.batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = request.seed + b;
        LOG_INFO("generating image: %i/%i - seed %" PRId64, b + 1, request.batch_count, cur_seed);

        sd_ctx->sd->rng->manual_seed(cur_seed);
        sd_ctx->sd->sampler_rng->manual_seed(cur_seed);
        sd::Tensor<float> noise = sd::randn_like<float>(latents.init_latent, sd_ctx->sd->rng);

        sd::Tensor<float> x_0 = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                                   true,
                                                   latents.init_latent,
                                                   std::move(noise),
                                                   embeds.cond,
                                                   embeds.uncond,
                                                   embeds.img_cond,
                                                   embeds.id_cond,
                                                   latents.control_image,
                                                   request.control_strength,
                                                   request.guidance,
                                                   plan.eta,
                                                   request.shifted_timestep,
                                                   plan.sample_method,
                                                   sd_ctx->sd->is_flow_denoiser(),
                                                   plan.extra_sample_args,
                                                   plan.sigmas,
                                                   plan.start_merge_step,
                                                   latents.ref_latents,
                                                   request.increase_ref_index,
                                                   latents.denoise_mask,
                                                   sd::Tensor<float>(),
                                                   1.f,
                                                   0,
                                                     static_cast<float>(request.fps),
                                                     request.cache_params);
        int64_t sampling_end  = ggml_time_ms();
        if (!x_0.empty()) {
            LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
            final_latents.push_back(std::move(x_0));
            continue;
        }

        LOG_ERROR("sampling for image %d/%d failed after %.2fs",
                  b + 1,
                  request.batch_count,
                  (sampling_end - sampling_start) * 1.0f / 1000);
        if (sd_ctx->sd->free_params_immediately) {
            sd_ctx->sd->diffusion_model->free_params_buffer();
        }
        return nullptr;
    }
    if (sd_ctx->sd->free_params_immediately && !request.hires.enabled) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }
    int64_t denoise_end = ggml_time_ms();
    LOG_INFO("generating %zu latent images completed, taking %.2fs",
             final_latents.size(),
             (denoise_end - denoise_start) * 1.0f / 1000);

    if (request.hires.enabled && request.hires.target_width > 0) {
        LOG_INFO("hires fix: upscaling to %dx%d", request.hires.target_width, request.hires.target_height);

        std::unique_ptr<UpscalerGGML> hires_upscaler;
        if (request.hires.upscaler == SD_HIRES_UPSCALER_MODEL) {
            LOG_INFO("hires fix: loading model upscaler from '%s'", request.hires.model_path);
            hires_upscaler                    = std::make_unique<UpscalerGGML>(sd_ctx->sd->n_threads,
                                                            false,
                                                            request.hires.upscale_tile_size,
                                                            sd_ctx->sd->backend_spec,
                                                            sd_ctx->sd->params_backend_spec);
            const size_t max_graph_vram_bytes = sd::ggml_graph_cut::max_vram_gib_to_bytes(sd_ctx->sd->max_vram);
            hires_upscaler->set_max_graph_vram_bytes(max_graph_vram_bytes);
            if (!hires_upscaler->load_from_file(request.hires.model_path,
                                                sd_ctx->sd->offload_params_to_cpu,
                                                sd_ctx->sd->n_threads)) {
                LOG_ERROR("load hires model upscaler failed");
                if (sd_ctx->sd->free_params_immediately) {
                    sd_ctx->sd->diffusion_model->free_params_buffer();
                }
                return nullptr;
            }
        }

        int hires_scheduler_steps = 0;
        std::vector<float> hires_sigma_sched =
            make_hires_sigma_schedule(sd_ctx,
                                      request.hires,
                                      sd_img_gen_params->sample_params,
                                      plan.sample_method,
                                      plan.sample_steps,
                                      sd_ctx->sd->get_image_seq_len(request.hires.target_height, request.hires.target_width),
                                      &hires_scheduler_steps);
        LOG_INFO("hires fix: scheduler_steps=%d, denoising_strength=%.2f, sigma_sched_size=%zu%s",
                 hires_scheduler_steps,
                 request.hires.denoising_strength,
                 hires_sigma_sched.size(),
                 request.hires.custom_sigmas_count > 0 ? ", custom_sigmas=true" : "");

        std::vector<sd::Tensor<float>> hires_final_latents;
        int64_t hires_denoise_start = ggml_time_ms();
        for (int b = 0; b < (int)final_latents.size(); b++) {
            int64_t cur_seed = request.seed + b;
            sd_ctx->sd->rng->manual_seed(cur_seed);
            sd_ctx->sd->sampler_rng->manual_seed(cur_seed);

            sd::Tensor<float> upscaled = upscale_hires_latent(sd_ctx,
                                                              final_latents[b],
                                                              request,
                                                              hires_upscaler.get());
            if (upscaled.empty()) {
                if (sd_ctx->sd->free_params_immediately) {
                    sd_ctx->sd->diffusion_model->free_params_buffer();
                }
                return nullptr;
            }

            sd::Tensor<float> noise = sd::randn_like<float>(upscaled, sd_ctx->sd->rng);

            sd::Tensor<float> hires_denoise_mask;
            if (!latents.denoise_mask.empty()) {
                std::vector<int64_t> mask_shape = latents.denoise_mask.shape();
                mask_shape[0]                   = upscaled.shape()[0];
                mask_shape[1]                   = upscaled.shape()[1];
                hires_denoise_mask              = sd::ops::interpolate(latents.denoise_mask,
                                                                       mask_shape,
                                                                       sd::ops::InterpolateMode::NearestMax);
            }

            int64_t hires_sample_start = ggml_time_ms();
            sd::Tensor<float> x_0      = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                                            true,
                                                            upscaled,
                                                            std::move(noise),
                                                            embeds.cond,
                                                            embeds.uncond,
                                                            embeds.img_cond,
                                                            embeds.id_cond,
                                                            latents.control_image,
                                                            request.control_strength,
                                                            request.guidance,
                                                            plan.eta,
                                                            request.shifted_timestep,
                                                            plan.sample_method,
                                                            sd_ctx->sd->is_flow_denoiser(),
                                                            plan.extra_sample_args,
                                                            hires_sigma_sched,
                                                            plan.start_merge_step,
                                                            latents.ref_latents,
                                                            request.increase_ref_index,
                                                            hires_denoise_mask,
                                                            sd::Tensor<float>(),
                                                            1.f,
                                                            0,
                                                            static_cast<float>(request.fps),
                                                            request.cache_params);
            int64_t hires_sample_end   = ggml_time_ms();
            if (!x_0.empty()) {
                LOG_INFO("hires sampling %d/%d completed, taking %.2fs",
                         b + 1,
                         (int)final_latents.size(),
                         (hires_sample_end - hires_sample_start) * 1.0f / 1000);
                hires_final_latents.push_back(std::move(x_0));
                continue;
            }

            LOG_ERROR("hires sampling for image %d/%d failed after %.2fs",
                      b + 1,
                      (int)final_latents.size(),
                      (hires_sample_end - hires_sample_start) * 1.0f / 1000);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->diffusion_model->free_params_buffer();
            }
            return nullptr;
        }
        if (sd_ctx->sd->free_params_immediately) {
            sd_ctx->sd->diffusion_model->free_params_buffer();
        }
        int64_t hires_denoise_end = ggml_time_ms();
        LOG_INFO("hires fix completed, taking %.2fs", (hires_denoise_end - hires_denoise_start) * 1.0f / 1000);

        final_latents = std::move(hires_final_latents);
    }

    auto result = decode_image_outputs(sd_ctx, request, final_latents);
    if (result == nullptr) {
        return nullptr;
    }

    sd_ctx->sd->lora_stat();

    int64_t t1 = ggml_time_ms();
    LOG_INFO("generate_image completed in %.2fs", (t1 - t0) * 1.0f / 1000);
    return result;
}

/* ============================================================= */
/* 6. StableDiffusionGGML::init_backend  (stable-diffusion.cpp:413) */
/* ============================================================= */
bool StableDiffusionGGML::init_backend(const sd_ctx_params_t* sd_ctx_params) {
    std::string error;
    if (!backend_manager.init(sd_ctx_params->backend,
                              sd_ctx_params->params_backend,
                              sd_ctx_params->offload_params_to_cpu,
                              sd_ctx_params->keep_clip_on_cpu,
                              sd_ctx_params->keep_vae_on_cpu,
                              sd_ctx_params->keep_control_net_on_cpu,
                              &error)) {
        LOG_ERROR("backend config failed: %s", error.c_str());
        return false;
    }
    return ensure_backend_pair(SDBackendModule::DIFFUSION);
}

/* ============================================================= */
/* 7. StableDiffusionGGML::init  (stable-diffusion.cpp:438)     */
/*    核心：模型加载入口, 约 1048 行. 仅复制前 100 行示意       */
/* ============================================================= */
bool StableDiffusionGGML::init(const sd_ctx_params_t* sd_ctx_params) {
    n_threads               = sd_ctx_params->n_threads;
    vae_decode_only         = sd_ctx_params->vae_decode_only;
    free_params_immediately = sd_ctx_params->free_params_immediately;
    offload_params_to_cpu   = sd_ctx_params->offload_params_to_cpu;
    max_vram                = sd_ctx_params->max_vram;
    backend_spec            = SAFE_STR(sd_ctx_params->backend);
    params_backend_spec     = SAFE_STR(sd_ctx_params->params_backend);
    rng = get_rng(sd_ctx_params->rng_type);
    sampler_rng = (sd_ctx_params->sampler_rng_type != RNG_TYPE_COUNT &&
                   sd_ctx_params->sampler_rng_type != sd_ctx_params->rng_type)
                  ? get_rng(sd_ctx_params->sampler_rng_type) : rng;
    ggml_log_set(ggml_log_callback_default, nullptr);
    if (!init_backend(sd_ctx_params)) return false;
    max_vram = sd::ggml_graph_cut::resolve_max_vram_gib(max_vram, backend_for(SDBackendModule::DIFFUSION));
    ModelLoader model_loader;
    if (strlen(SAFE_STR(sd_ctx_params->model_path)) > 0) {
        LOG_INFO("loading model from '%s'", sd_ctx_params->model_path);
        if (!model_loader.init_from_file(sd_ctx_params->model_path)) { return false; }
    }
    // ... (完整实现见 /opt/stable-diffusion.cpp/src/stable-diffusion.cpp:438-1486)
    return true;
}

/* ============================================================= */
/* 8. KarrasScheduler    (src/denoiser.hpp:282)                  */
/* ============================================================= */
struct KarrasScheduler : SigmaScheduler {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t) override {
        float rho = 7.f;
        if (sigma_min <= 1e-6f) sigma_min = 1e-6f;
        std::vector<float> result(n + 1);
        float min_inv_rho = pow(sigma_min, (1.f / rho));
        float max_inv_rho = pow(sigma_max, (1.f / rho));
        for (uint32_t i = 0; i < n; i++)
            result[i] = pow(max_inv_rho + (float)i / ((float)n - 1.f) * (min_inv_rho - max_inv_rho), rho);
        result[n] = 0.; return result;
    }
};

/* ============================================================= */
/* 9. CompVisDenoiser    (src/denoiser.hpp:645)                  */
/* ============================================================= */
struct CompVisDenoiser : Denoiser {
    Discretization* discretization;
    CompVisDenoiser() : discretization(new LegacyDDPMDiscretization()) {}
    float sigma_max() override { return sigmas[TIMESTEPS - 1]; }
};

/* ============================================================= */
/* 10. calculate_alphas_cumprod  (stable-diffusion.cpp:288)      */
/* ============================================================= */
void calculate_alphas_cumprod(float* alphas_cumprod,
                              float linear_start, float linear_end, int timesteps) {
    float ls_sqrt = sqrtf(linear_start), le_sqrt = sqrtf(linear_end);
    float amount = le_sqrt - ls_sqrt, product = 1.0f;
    for (int i = 0; i < timesteps; i++) {
        float beta = ls_sqrt + amount * ((float)i / (timesteps - 1));
        product *= 1.0f - powf(beta, 2.0f);
        alphas_cumprod[i] = product;
    }
}
