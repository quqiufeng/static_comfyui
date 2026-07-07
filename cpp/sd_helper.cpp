// sd_helper.cpp — stable-diffusion.cpp API 参考实现
// 按 api.md 文档从 /opt/stable-diffusion.cpp 各指定文件行号复制
// ================================================================
// 1. 类型定义 & API 声明 (include/stable-diffusion.h)
// ================================================================
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <cstdio>
#include <cstdlib>

// ── 枚举 ──
enum rng_type_t {
    STD_DEFAULT_RNG, CUDA_RNG, CPU_RNG, RNG_TYPE_COUNT
};
enum sample_method_t {
    EULER_SAMPLE_METHOD, EULER_A_SAMPLE_METHOD, HEUN_SAMPLE_METHOD,
    DPM2_SAMPLE_METHOD, DPMPP2S_A_SAMPLE_METHOD, DPMPP2M_SAMPLE_METHOD,
    DPMPP2Mv2_SAMPLE_METHOD, IPNDM_SAMPLE_METHOD, IPNDM_V_SAMPLE_METHOD,
    LCM_SAMPLE_METHOD, DDIM_TRAILING_SAMPLE_METHOD, TCD_SAMPLE_METHOD,
    RES_MULTISTEP_SAMPLE_METHOD, RES_2S_SAMPLE_METHOD, ER_SDE_SAMPLE_METHOD,
    EULER_CFG_PP_SAMPLE_METHOD, EULER_A_CFG_PP_SAMPLE_METHOD,
    EULER_GE_SAMPLE_METHOD, SAMPLE_METHOD_COUNT
};
enum scheduler_t {
    DISCRETE_SCHEDULER, KARRAS_SCHEDULER, EXPONENTIAL_SCHEDULER, AYS_SCHEDULER,
    GITS_SCHEDULER, SGM_UNIFORM_SCHEDULER, SIMPLE_SCHEDULER,
    SMOOTHSTEP_SCHEDULER, KL_OPTIMAL_SCHEDULER, LCM_SCHEDULER,
    BONG_TANGENT_SCHEDULER, LTX2_SCHEDULER, SCHEDULER_COUNT
};
enum prediction_t {
    EPS_PRED, V_PRED, EDM_V_PRED, FLOW_PRED, FLUX_FLOW_PRED, FLUX2_FLOW_PRED,
    PREDICTION_COUNT
};
enum sd_type_t {
    SD_TYPE_F32=0, SD_TYPE_F16=1, SD_TYPE_Q4_0=2, SD_TYPE_Q4_1=3,
    SD_TYPE_Q5_0=6, SD_TYPE_Q5_1=7, SD_TYPE_Q8_0=8, SD_TYPE_Q8_1=9,
    SD_TYPE_Q2_K=10, SD_TYPE_Q3_K=11, SD_TYPE_Q4_K=12, SD_TYPE_Q5_K=13,
    SD_TYPE_Q6_K=14, SD_TYPE_Q8_K=15, SD_TYPE_IQ2_XXS=16, SD_TYPE_IQ2_XS=17,
    SD_TYPE_IQ3_XXS=18, SD_TYPE_IQ1_S=19, SD_TYPE_IQ4_NL=20, SD_TYPE_IQ3_S=21,
    SD_TYPE_IQ2_S=22, SD_TYPE_IQ4_XS=23, SD_TYPE_I8=24, SD_TYPE_I16=25,
    SD_TYPE_I32=26, SD_TYPE_I64=27, SD_TYPE_F64=28, SD_TYPE_IQ1_M=29,
    SD_TYPE_BF16=30, SD_TYPE_TQ1_0=34, SD_TYPE_TQ2_0=35,
    SD_TYPE_MXFP4=39, SD_TYPE_NVFP4=40, SD_TYPE_Q1_0=41, SD_TYPE_COUNT=42,
};

// ── 结构体 ──
typedef struct { uint32_t width; uint32_t height; uint32_t channel; uint8_t* data; } sd_image_t;
typedef struct { float txt_cfg; float img_cfg; /*sd_slg_params_t slg;*/ } sd_guidance_params_t;
typedef struct {
    sd_guidance_params_t guidance; enum scheduler_t scheduler;
    enum sample_method_t sample_method; int sample_steps; float eta;
    int shifted_timestep; float* custom_sigmas; int custom_sigmas_count;
    float flow_shift; const char* extra_sample_args;
} sd_sample_params_t;
typedef struct {
    const char* model_path; const char* clip_l_path; const char* clip_g_path;
    const char* clip_vision_path; const char* t5xxl_path; const char* llm_path;
    const char* llm_vision_path; const char* diffusion_model_path;
    const char* high_noise_diffusion_model_path; const char* vae_path;
    const char* taesd_path; const char* control_net_path;
    const sd_embedding_t* embeddings; uint32_t embedding_count;
    bool vae_decode_only; bool free_params_immediately;
    int n_threads; enum sd_type_t wtype; enum rng_type_t rng_type;
    enum prediction_t prediction; bool offload_params_to_cpu; bool enable_mmap;
    bool keep_clip_on_cpu; bool keep_control_net_on_cpu; bool keep_vae_on_cpu;
    bool flash_attn; bool diffusion_flash_attn;
    float max_vram; const char* backend; const char* params_backend;
} sd_ctx_params_t;
typedef struct {
    const sd_lora_t* loras; uint32_t lora_count;
    const char* prompt; const char* negative_prompt; int clip_skip;
    sd_image_t init_image; sd_image_t mask_image;
    int width; int height; sd_sample_params_t sample_params;
    float strength; int64_t seed; int batch_count;
    sd_image_t control_image; float control_strength;
} sd_img_gen_params_t;

struct sd_ctx_t;
typedef struct sd_ctx_t sd_ctx_t;

// ── API 函数声明 ──
void        sd_ctx_params_init(sd_ctx_params_t*);
sd_ctx_t*   new_sd_ctx(const sd_ctx_params_t*);
void        free_sd_ctx(sd_ctx_t*);
void        sd_sample_params_init(sd_sample_params_t*);
void        sd_img_gen_params_init(sd_img_gen_params_t*);
sd_image_t* generate_image(sd_ctx_t*, const sd_img_gen_params_t*);
int32_t     sd_get_num_physical_cores();
const char* sd_get_system_info();
bool        sd_ctx_supports_image_generation(const sd_ctx_t*);

    bool init(const sd_ctx_params_t* sd_ctx_params) {
        n_threads               = sd_ctx_params->n_threads;
        vae_decode_only         = sd_ctx_params->vae_decode_only;
        free_params_immediately = sd_ctx_params->free_params_immediately;
        offload_params_to_cpu   = sd_ctx_params->offload_params_to_cpu;
        max_vram                = sd_ctx_params->max_vram;
        backend_spec            = SAFE_STR(sd_ctx_params->backend);
        params_backend_spec     = SAFE_STR(sd_ctx_params->params_backend);

        bool use_tae       = false;
        bool use_audio_vae = false;

        rng = get_rng(sd_ctx_params->rng_type);
        if (sd_ctx_params->sampler_rng_type != RNG_TYPE_COUNT && sd_ctx_params->sampler_rng_type != sd_ctx_params->rng_type) {
            sampler_rng = get_rng(sd_ctx_params->sampler_rng_type);
        } else {
            sampler_rng = rng;
        }

        ggml_log_set(ggml_log_callback_default, nullptr);

        if (!init_backend(sd_ctx_params)) {
            return false;
        }
        max_vram = sd::ggml_graph_cut::resolve_max_vram_gib(max_vram, backend_for(SDBackendModule::DIFFUSION));

        ModelLoader model_loader;

        if (strlen(SAFE_STR(sd_ctx_params->model_path)) > 0) {
            LOG_INFO("loading model from '%s'", sd_ctx_params->model_path);
            if (!model_loader.init_from_file(sd_ctx_params->model_path)) {
                LOG_ERROR("init model loader from file failed: '%s'", sd_ctx_params->model_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->diffusion_model_path)) > 0) {
            LOG_INFO("loading diffusion model from '%s'", sd_ctx_params->diffusion_model_path);
            if (!model_loader.init_from_file(sd_ctx_params->diffusion_model_path, "model.diffusion_model.")) {
                LOG_WARN("loading diffusion model from '%s' failed", sd_ctx_params->diffusion_model_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path)) > 0) {
            LOG_INFO("loading high noise diffusion model from '%s'", sd_ctx_params->high_noise_diffusion_model_path);
            if (!model_loader.init_from_file(sd_ctx_params->high_noise_diffusion_model_path, "model.high_noise_diffusion_model.")) {
                LOG_WARN("loading diffusion model from '%s' failed", sd_ctx_params->high_noise_diffusion_model_path);
            }
        }

        bool is_unet = sd_version_is_unet(model_loader.get_sd_version());

        if (strlen(SAFE_STR(sd_ctx_params->clip_l_path)) > 0) {
            LOG_INFO("loading clip_l from '%s'", sd_ctx_params->clip_l_path);
            std::string prefix = is_unet ? "cond_stage_model.transformer." : "text_encoders.clip_l.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_l_path, prefix)) {
                LOG_WARN("loading clip_l from '%s' failed", sd_ctx_params->clip_l_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->clip_g_path)) > 0) {
            LOG_INFO("loading clip_g from '%s'", sd_ctx_params->clip_g_path);
            std::string prefix = is_unet ? "cond_stage_model.1.transformer." : "text_encoders.clip_g.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_g_path, prefix)) {
                LOG_WARN("loading clip_g from '%s' failed", sd_ctx_params->clip_g_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->clip_vision_path)) > 0) {
            LOG_INFO("loading clip_vision from '%s'", sd_ctx_params->clip_vision_path);
            std::string prefix = "cond_stage_model.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_vision_path, prefix)) {
                LOG_WARN("loading clip_vision from '%s' failed", sd_ctx_params->clip_vision_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->t5xxl_path)) > 0) {
            LOG_INFO("loading t5xxl from '%s'", sd_ctx_params->t5xxl_path);
            if (!model_loader.init_from_file(sd_ctx_params->t5xxl_path, "text_encoders.t5xxl.transformer.")) {
                LOG_WARN("loading t5xxl from '%s' failed", sd_ctx_params->t5xxl_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->llm_path)) > 0) {
            LOG_INFO("loading llm from '%s'", sd_ctx_params->llm_path);
            if (!model_loader.init_from_file(sd_ctx_params->llm_path, "text_encoders.llm.")) {
                LOG_WARN("loading llm from '%s' failed", sd_ctx_params->llm_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->llm_vision_path)) > 0) {
            LOG_INFO("loading llm vision from '%s'", sd_ctx_params->llm_vision_path);
            if (!model_loader.init_from_file(sd_ctx_params->llm_vision_path, "text_encoders.llm.visual.")) {
                LOG_WARN("loading llm vision from '%s' failed", sd_ctx_params->llm_vision_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->vae_path)) > 0) {
            LOG_INFO("loading vae from '%s'", sd_ctx_params->vae_path);
            if (!model_loader.init_from_file(sd_ctx_params->vae_path, "vae.")) {
                LOG_WARN("loading vae from '%s' failed", sd_ctx_params->vae_path);
                external_vae_is_invalid = true;
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->taesd_path)) > 0) {
            LOG_INFO("loading tae from '%s'", sd_ctx_params->taesd_path);
            if (!model_loader.init_from_file(sd_ctx_params->taesd_path, "tae.")) {
                LOG_WARN("loading tae from '%s' failed", sd_ctx_params->taesd_path);
            } else {
                use_tae = true;
            }
        }

// ── sd_ctx_params_init ──
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
    sd_ctx_params->diffusion_flash_attn    = false;
    sd_ctx_params->circular_x              = false;
    sd_ctx_params->circular_y              = false;
    sd_ctx_params->chroma_use_dit_mask     = true;
    sd_ctx_params->chroma_use_t5_mask      = false;
    sd_ctx_params->chroma_t5_mask_pad      = 1;
    sd_ctx_params->vae_format              = SD_VAE_FORMAT_AUTO;
    sd_ctx_params->backend                 = nullptr;
    sd_ctx_params->params_backend          = nullptr;
    sd_ctx_params->ipadapter_unet_mode     = false;
    sd_ctx_params->ipadapter_unet_weights_path = nullptr;
}
// ── sd_sample_params_init ──
void sd_sample_params_init(sd_sample_params_t* sample_params) {
    *sample_params                             = {};
    sample_params->guidance.txt_cfg            = 7.0f;
    sample_params->guidance.img_cfg            = INFINITY;
    sample_params->guidance.distilled_guidance = 3.5f;
    sample_params->guidance.slg.layer_count    = 0;
    sample_params->guidance.slg.layer_start    = 0.01f;
    sample_params->guidance.slg.layer_end      = 0.2f;
    sample_params->guidance.slg.scale          = 0.f;
    sample_params->scheduler                   = SCHEDULER_COUNT;
    sample_params->sample_method               = SAMPLE_METHOD_COUNT;
    sample_params->sample_steps                = 20;
    sample_params->eta                         = INFINITY;
    sample_params->custom_sigmas               = nullptr;
    sample_params->custom_sigmas_count         = 0;
    sample_params->flow_shift                  = INFINITY;
    sample_params->extra_sample_args           = nullptr;
}
// ── sd_img_gen_params_init ──
void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params) {
    *sd_img_gen_params = {};
    sd_sample_params_init(&sd_img_gen_params->sample_params);
    sd_img_gen_params->clip_skip         = -1;
    sd_img_gen_params->ref_images_count  = 0;
    sd_img_gen_params->width             = 512;
    sd_img_gen_params->height            = 512;
    sd_img_gen_params->strength          = 0.75f;
    sd_img_gen_params->seed              = -1;
    sd_img_gen_params->batch_count       = 1;
    sd_img_gen_params->control_strength  = 0.9f;
    sd_img_gen_params->pm_params         = {nullptr, 0, nullptr, 20.f};
    sd_img_gen_params->vae_tiling_params = {false, false, 0, 0, 0.5f, 0.0f, 0.0f, nullptr};
    sd_cache_params_init(&sd_img_gen_params->cache);
    sd_hires_params_init(&sd_img_gen_params->hires);
}
// ─── new_sd_ctx ───
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
// ─── free_sd_ctx ───
void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx->sd != nullptr) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
    }
    free(sd_ctx);
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

// ──────────────────────────────────────────────────────────────
// 2. ModelLoader (src/model.cpp, src/model.h)
// ──────────────────────────────────────────────────────────────
#include "model.h"

// model.h:249-299 class ModelLoader declaration
// model.cpp:211 init_from_file
// model.cpp:293 init_from_safetensors_file  
// model.cpp:1173 load_tensors(map)

// ──────────────────────────────────────────────────────────────
// 3. GGMLRunner (src/ggml_extend.hpp)
// ──────────────────────────────────────────────────────────────
#include "ggml_extend.hpp"

// ggml_extend.hpp:1749 alloc_params_ctx
// ggml_extend.hpp:1797 alloc_compute_ctx
// ggml_extend.hpp:1884 alloc_compute_buffer
// ggml_extend.hpp:2036 copy_data_to_backend_tensor
// ggml_extend.hpp:2091 offload_all_params
// ggml_extend.hpp:2393 execute_graph
// ggml_extend.hpp:2650 alloc_params_buffer
// ggml_extend.hpp:2719 set_backend_tensor_data
// ggml_extend.hpp:2778 compute

// ──────────────────────────────────────────────────────────────
// 4. UNet (src/unet.hpp, src/common_block.hpp)
// ──────────────────────────────────────────────────────────────
#include "unet.hpp"
#include "common_block.hpp"

// unet.hpp:203 UnetModelBlock::UnetModelBlock init
// unet.hpp:651 UNetModelRunner::UNetModelRunner
// unet.hpp:675 UNetModelRunner::build_graph
// common_block.hpp:230 FeedForward
// common_block.hpp:282 CrossAttention  
// common_block.hpp:329 CrossAttention::forward
// common_block.hpp:384 BasicTransformerBlock
// common_block.hpp:457 SpatialTransformer
// common_block.hpp:483 SpatialTransformer::SpatialTransformer

// ──────────────────────────────────────────────────────────────
// 5. VAE (src/auto_encoder_kl.hpp)
// ──────────────────────────────────────────────────────────────
#include "auto_encoder_kl.hpp"

// auto_encoder_kl.hpp:442 Decoder::forward

// ──────────────────────────────────────────────────────────────
// 6. Denoiser (src/denoiser.hpp)
// ──────────────────────────────────────────────────────────────
#include "denoiser.hpp"
// denoiser.hpp:282 KarrasScheduler
// denoiser.hpp:645 CompVisDenoiser::sigma_max

// ──────────────────────────────────────────────────────────────
// 7. Safetensors I/O (src/model_io/safetensors_io.cpp)
// ──────────────────────────────────────────────────────────────
#include "model_io/safetensors_io.h"
// safetensors_io.cpp:87 read_safetensors_file

// ──────────────────────────────────────────────────────────────
// 8. ModelLoader::init_from_file (model.cpp:211)
// ──────────────────────────────────────────────────────────────
bool ModelLoader::init_from_file(const std::string& file_path, const std::string& prefix) {
    if (is_directory(file_path)) {
        return init_from_diffusers_file(file_path, prefix);
    } else if (is_gguf_file(file_path)) {
        return init_from_gguf_file(file_path, prefix);
    } else if (is_safetensors_file(file_path)) {
        return init_from_safetensors_file(file_path, prefix);
    } else if (is_torch_zip_file(file_path)) {
        return init_from_torch_zip_file(file_path, prefix);
    } else if (init_from_torch_legacy_file(file_path, prefix)) {
        return true;
    } else { return false; }
}

// ──────────────────────────────────────────────────────────────
// 9. ModelLoader::init_from_safetensors_file (model.cpp:293)
// ──────────────────────────────────────────────────────────────
bool ModelLoader::init_from_safetensors_file(const std::string& file_path, const std::string& prefix) {
    std::vector<TensorStorage> tensor_storages; std::string error;
    if (!read_safetensors_file(file_path, tensor_storages, &error)) { return false; }
    file_paths_.push_back(file_path);
    size_t file_index = file_paths_.size() - 1;
    for (auto& tensor_storage : tensor_storages) {
        if (is_unused_tensor(tensor_storage.name)) continue;
        if (!starts_with(tensor_storage.name, prefix))
            tensor_storage.name = prefix + tensor_storage.name;
        tensor_storage.file_index = file_index;
        add_tensor_storage(tensor_storage);
    }
    return true;
}

// ──────────────────────────────────────────────────────────────
// 10. GGMLRunner::alloc_params_ctx (ggml_extend.hpp:1749)
// ──────────────────────────────────────────────────────────────
void GGMLRunner::alloc_params_ctx() {
    ggml_init_params params;
    params.mem_size   = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
    params.mem_buffer = nullptr;
    params.no_alloc   = true;
    params_ctx = ggml_init(params);
    if (params_backend != runtime_backend)
        offload_ctx = ggml_init(params);
}

// ──────────────────────────────────────────────────────────────
// 11. GGMLRunner::alloc_compute_ctx (ggml_extend.hpp:1797)
// ──────────────────────────────────────────────────────────────
void GGMLRunner::alloc_compute_ctx() {
    ggml_init_params params;
    params.mem_size   = static_cast<size_t>(ggml_tensor_overhead() * MAX_GRAPH_SIZE + ggml_graph_overhead());
    params.mem_buffer = nullptr;
    params.no_alloc   = true;
    compute_ctx = ggml_init(params);
}

// ──────────────────────────────────────────────────────────────
// 12. GGMLRunner::alloc_compute_buffer (ggml_extend.hpp:1884)
// ──────────────────────────────────────────────────────────────
bool GGMLRunner::alloc_compute_buffer(ggml_cgraph* gf) {
    if (compute_allocr != nullptr) return true;
    compute_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_backend));
    if (!ggml_gallocr_reserve(compute_allocr, gf)) {
        free_compute_buffer(); return false;
    }
    return true;
}

// ──────────────────────────────────────────────────────────────
// 13. GGMLRunner::copy_data_to_backend_tensor (ggml_extend.hpp:2036)
// ──────────────────────────────────────────────────────────────
void GGMLRunner::copy_data_to_backend_tensor(ggml_cgraph* gf, bool clear_after_copy) {
    std::unordered_set<const ggml_tensor*> graph_tensor_set;
    for (int i = 0; i < sd::ggml_graph_cut::leaf_count(gf); ++i)
        graph_tensor_set.insert(sd::ggml_graph_cut::leaf_tensor(gf, i));
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i)
        graph_tensor_set.insert(ggml_graph_node(gf, i));
    for (auto& kv : backend_tensor_data_map) {
        auto tensor = kv.first; auto data = kv.second;
        if (!tensor || !data) continue;
        if (graph_tensor_set.find(tensor) == graph_tensor_set.end()) continue;
        if (tensor->buffer == nullptr) continue;
        ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
    }
    if (clear_after_copy) backend_tensor_data_map.clear();
}

// ──────────────────────────────────────────────────────────────
// 14. GGMLRunner::offload_all_params (ggml_extend.hpp:2091)
// ──────────────────────────────────────────────────────────────
bool GGMLRunner::offload_all_params() {
    if (params_backend == runtime_backend) return true;
    if (params_on_runtime_backend) return true;
    size_t num_tensors = ggml_tensor_num(offload_ctx);
    if (num_tensors == 0) {
        for (ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != nullptr;
             t = ggml_get_next_tensor(params_ctx, t)) {
            ggml_dup_tensor(offload_ctx, t);
        }
    }
    num_tensors = ggml_tensor_num(offload_ctx);
    runtime_params_buffer = ggml_backend_alloc_ctx_tensors(offload_ctx, runtime_backend);
    if (runtime_params_buffer == nullptr) return false;
    int i = 0;
    for (ggml_tensor* src = ggml_get_first_tensor(params_ctx); src != nullptr;
         src = ggml_get_next_tensor(params_ctx, src), i++) {
        ggml_tensor* dst = ggml_get_tensor(offload_ctx, ggml_get_name(src));
        if (src->view_src) continue;
        ggml_backend_tensor_copy(src, dst);
    }
    params_on_runtime_backend = true; return true;
}

// ──────────────────────────────────────────────────────────────
// 15. GGMLRunner::set_backend_tensor_data (ggml_extend.hpp:2719)
// ──────────────────────────────────────────────────────────────
void GGMLRunner::set_backend_tensor_data(ggml_tensor* tensor, const void* data) {
    backend_tensor_data_map[tensor] = data;
}

// ──────────────────────────────────────────────────────────────
// 16. UNetModelRunner::UNetModelRunner (unet.hpp:651)
// ──────────────────────────────────────────────────────────────
UNetModelRunner::UNetModelRunner(ggml_backend_t backend, ggml_backend_t params_backend,
    const String2TensorStorage& tensor_storage_map, const std::string prefix, SDVersion version)
    : DiffusionModelRunner(backend, params_backend, prefix), unet(version, tensor_storage_map) {
    if (g_ipadapter_unet_enabled) {
        g_ipadapter_image_embeds_tensor = ggml_new_tensor_3d(params_ctx, GGML_TYPE_F32, 2048, 16, 1);
        if (g_ipadapter_image_embeds_tensor != nullptr)
            ggml_set_name(g_ipadapter_image_embeds_tensor, "ipadapter_image_embeds");
    }
    unet.init(params_ctx, tensor_storage_map, prefix);
}

// ──────────────────────────────────────────────────────────────
// 17. UNetModelRunner::build_graph (unet.hpp:675)
// ──────────────────────────────────────────────────────────────
ggml_cgraph* UNetModelRunner::build_graph(const sd::Tensor<float>& x_tensor,
    const sd::Tensor<float>& timesteps_tensor, const sd::Tensor<float>& context_tensor,
    const sd::Tensor<float>& c_concat_tensor, const sd::Tensor<float>& y_tensor,
    int num_video_frames, const std::vector<sd::Tensor<float>>& controls_tensor,
    float control_strength) {
    ggml_cgraph* gf = new_graph_custom(UNET_GRAPH_SIZE);
    ggml_tensor* x         = make_input(x_tensor);
    ggml_tensor* timesteps = make_input(timesteps_tensor);
    ggml_tensor* context   = make_optional_input(context_tensor);
    ggml_tensor* c_concat  = make_optional_input(c_concat_tensor);
    ggml_tensor* y         = make_optional_input(y_tensor);
    std::vector<ggml_tensor*> controls;
    for (const auto& control_tensor : controls_tensor)
        controls.push_back(make_input(control_tensor));
    if (num_video_frames == -1)
        num_video_frames = static_cast<int>(x->ne[3]);
    auto runner_ctx = get_context();
    ggml_tensor* out = unet.forward(&runner_ctx, x, timesteps, context,
                                     c_concat, y, num_video_frames, controls, control_strength);
    ggml_build_forward_expand(gf, out);
    return gf;
}

// ──────────────────────────────────────────────────────────────
// 18. SpatialTransformer (common_block.hpp:457)
// ──────────────────────────────────────────────────────────────
class SpatialTransformer : public GGMLBlock {
public:
    int64_t in_channels, n_head, d_head, depth, context_dim;
    bool use_linear;
    SpatialTransformer(int64_t in_channels, int64_t n_head, int64_t d_head,
                       int64_t depth, int64_t context_dim, bool use_linear)
        : in_channels(in_channels), n_head(n_head), d_head(d_head),
          depth(depth), context_dim(context_dim), use_linear(use_linear) {
        int64_t inner_dim = n_head * d_head;
        blocks["norm"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(in_channels));
        if (use_linear)
            blocks["proj_in"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, inner_dim));
        else
            blocks["proj_in"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, inner_dim, {1, 1}));
        for (int i = 0; i < depth; i++) {
            std::string name = "transformer_blocks." + std::to_string(i);
            blocks[name] = std::shared_ptr<GGMLBlock>(
                new BasicTransformerBlock(inner_dim, n_head, d_head, context_dim, false));
        }
        if (use_linear)
            blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, in_channels));
        else
            blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Conv2d(inner_dim, in_channels, {1, 1}));
    }
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* context);
};

// ──────────────────────────────────────────────────────────────
// 19. SpatialTransformer::forward (common_block.hpp)
// ──────────────────────────────────────────────────────────────
ggml_tensor* SpatialTransformer::forward(GGMLRunnerContext* ctx,
                                          ggml_tensor* x, ggml_tensor* context) {
    auto x_in = x;
    // GroupNorm
    auto norm = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm"]);
    x = norm->forward(ctx, x);
    // proj_in
    if (use_linear) {
        auto proj_in = std::dynamic_pointer_cast<Linear>(blocks["proj_in"]);
        x = proj_in->forward(ctx, x);
        x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 2, 0, 3));
        x = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0], x->ne[1] * x->ne[2], x->ne[3]);
    } else {
        auto proj_in = std::dynamic_pointer_cast<Conv2d>(blocks["proj_in"]);
        x = proj_in->forward(ctx, x);
    }
    // Transformer blocks
    for (int i = 0; i < depth; i++) {
        std::string name = "transformer_blocks." + std::to_string(i);
        auto tb = std::dynamic_pointer_cast<BasicTransformerBlock>(blocks[name]);
        x = tb->forward(ctx, x, context);
    }
    // proj_out
    if (use_linear) {
        auto proj_out = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);
        x = proj_out->forward(ctx, x);
        x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
        x = ggml_reshape_4d(ctx->ggml_ctx, x, x->ne[0], x->ne[1],
                            x->ne[2] / (x->ne[1] * x->ne[0]), x->ne[3]);
    } else {
        auto proj_out = std::dynamic_pointer_cast<Conv2d>(blocks["proj_out"]);
        x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
        x = ggml_reshape_4d(ctx->ggml_ctx, x, x->ne[0], x->ne[1],
                            x->ne[2] / (x->ne[1] * x->ne[0]), x->ne[3]);
        x = proj_out->forward(ctx, x);
    }
    return ggml_add(ctx->ggml_ctx, x, x_in);
}

// ──────────────────────────────────────────────────────────────
// 20. CrossAttention (common_block.hpp:282,329)
// ──────────────────────────────────────────────────────────────
class CrossAttention : public GGMLBlock {
    int64_t query_dim, context_dim, n_head, d_head;
public:
    CrossAttention(int64_t query_dim, int64_t context_dim, int64_t n_head, int64_t d_head)
        : n_head(n_head), d_head(d_head), query_dim(query_dim), context_dim(context_dim) {
        int64_t inner_dim = d_head * n_head;
        blocks["to_q"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, false));
        blocks["to_k"] = std::shared_ptr<GGMLBlock>(new Linear(context_dim, inner_dim, false));
        blocks["to_v"] = std::shared_ptr<GGMLBlock>(new Linear(context_dim, inner_dim, false));
        blocks["to_out.0"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, query_dim));
    }
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* context) {
        auto to_q     = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
        auto to_k     = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
        auto to_v     = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
        auto to_out_0 = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);
        int64_t N = x->ne[2], n_token = x->ne[1], n_context = context->ne[1];
        int64_t inner_dim = d_head * n_head;
        auto q = to_q->forward(ctx, x);
        auto k = to_k->forward(ctx, context);
        auto v = to_v->forward(ctx, context);
        x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, n_head, nullptr, false, ctx->flash_attn_enabled);
        x = to_out_0->forward(ctx, x);
        return x;
    }
};

// ──────────────────────────────────────────────────────────────
// 21. BasicTransformerBlock (common_block.hpp:384)
// ──────────────────────────────────────────────────────────────
class BasicTransformerBlock : public GGMLBlock {
public:
    BasicTransformerBlock(int64_t dim, int64_t n_head, int64_t d_head,
                          int64_t context_dim, bool use_checkpoint) {
        blocks["norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
        blocks["attn1"] = std::shared_ptr<GGMLBlock>(new CrossAttention(dim, dim, n_head, d_head));
        blocks["norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
        blocks["attn2"] = std::shared_ptr<GGMLBlock>(new CrossAttention(dim, context_dim, n_head, d_head));
        blocks["norm3"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
        blocks["ff"]    = std::shared_ptr<GGMLBlock>(new FeedForward(dim, dim, 4, FeedForward::Activation::GEGLU));
    }
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* context) {
        auto attn1 = std::dynamic_pointer_cast<CrossAttention>(blocks["attn1"]);
        auto attn2 = std::dynamic_pointer_cast<CrossAttention>(blocks["attn2"]);
        auto ff    = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);
        auto norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
        auto norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
        auto norm3 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm3"]);
        auto r = x;
        x = norm1->forward(ctx, x);
        x = attn1->forward(ctx, x, x);
        x = ggml_add(ctx->ggml_ctx, x, r);
        r = x;
        x = norm2->forward(ctx, x);
        x = attn2->forward(ctx, x, context);
        x = ggml_add(ctx->ggml_ctx, x, r);
        r = x;
        x = norm3->forward(ctx, x);
        x = ff->forward(ctx, x);
        x = ggml_add(ctx->ggml_ctx, x, r);
        return x;
    }
};

// ──────────────────────────────────────────────────────────────
// 22. FeedForward (common_block.hpp:230)
// ──────────────────────────────────────────────────────────────
class FeedForward : public GGMLBlock {
public:
    enum class Activation { GEGLU, GELU };
    FeedForward(int64_t dim, int64_t dim_out, int64_t mult = 4,
                Activation activation = Activation::GEGLU, bool precision_fix = false) {
        int64_t inner_dim = dim * mult;
        if (activation == Activation::GELU)
            blocks["net.0"] = std::shared_ptr<GGMLBlock>(new GELU(dim, inner_dim));
        else
            blocks["net.0"] = std::shared_ptr<GGMLBlock>(new GEGLU(dim, inner_dim));
        blocks["net.2"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, dim_out, true, false, false, 1.0f));
    }
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto net_0 = std::dynamic_pointer_cast<UnaryBlock>(blocks["net.0"]);
        auto net_2 = std::dynamic_pointer_cast<Linear>(blocks["net.2"]);
        x = net_0->forward(ctx, x);
        x = net_2->forward(ctx, x);
        return x;
    }
};

// ──────────────────────────────────────────────────────────────
// 23. ggml_ext_attention_ext (ggml_extend.hpp:1288)
// ──────────────────────────────────────────────────────────────
__STATIC_INLINE__ ggml_tensor* ggml_ext_attention_ext(ggml_context* ctx,
    ggml_backend_t backend, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
    int64_t n_head, ggml_tensor* mask, bool skip_reshape, bool flash_attn, float kv_scale) {
    int64_t L_q, L_k, C, N, d_head, n_kv_head;
    if (!skip_reshape) {
        L_q = q->ne[1]; L_k = k->ne[1]; C = q->ne[0];
        N = q->ne[2]; d_head = C / n_head; n_kv_head = k->ne[0] / d_head;
        q = ggml_reshape_4d(ctx, q, d_head, n_head, L_q, N);
        q = ggml_ext_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        q = ggml_reshape_3d(ctx, q, d_head, L_q, n_head * N);
        k = ggml_reshape_4d(ctx, k, d_head, n_kv_head, L_k, N);
        k = ggml_ext_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        k = ggml_reshape_3d(ctx, k, d_head, L_k, n_kv_head * N);
        v = ggml_reshape_4d(ctx, v, d_head, n_kv_head, L_k, N);
    }
    float scale = (1.0f / sqrt((float)d_head));
    ggml_tensor* kqv = nullptr;
    auto build_kqv = [&](ggml_tensor* q_in, ggml_tensor* k_in, ggml_tensor* v_in, ggml_tensor* mask_in) {
        k_in = ggml_cast(ctx, k_in, GGML_TYPE_F16);
        v_in = ggml_ext_cont(ctx, ggml_permute(ctx, v_in, 0, 2, 1, 3));
        v_in = ggml_reshape_3d(ctx, v_in, d_head, L_k, n_kv_head * N);
        v_in = ggml_cast(ctx, v_in, GGML_TYPE_F16);
        auto out = ggml_flash_attn_ext(ctx, q_in, k_in, v_in, mask_in, scale / kv_scale, 0, 0);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        return out;
    };
    if (flash_attn) {
        kqv = build_kqv(q, k, v, mask);
        if (!ggml_backend_supports_op(backend, kqv)) kqv = nullptr;
        else kqv = ggml_view_3d(ctx, kqv, d_head, n_head, L_q, kqv->nb[1], kqv->nb[2], 0);
    }
    if (kqv == nullptr) {
        v = ggml_ext_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        v = ggml_reshape_3d(ctx, v, L_k, d_head, n_kv_head * N);
        auto kq = ggml_mul_mat(ctx, k, q);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_scale_inplace(ctx, kq, scale);
        if (mask) kq = ggml_add_inplace(ctx, kq, mask);
        kq = ggml_soft_max_inplace(ctx, kq);
        kqv = ggml_mul_mat(ctx, v, kq);
        kqv = ggml_reshape_4d(ctx, kqv, d_head, L_q, n_head, N);
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    }
    kqv = ggml_ext_cont(ctx, kqv);
    kqv = ggml_reshape_3d(ctx, kqv, d_head * n_head, L_q, N);
    return kqv;
}

// ──────────────────────────────────────────────────────────────
// 24. VAE Decoder::forward (auto_encoder_kl.hpp:442)
// ──────────────────────────────────────────────────────────────
ggml_tensor* Decoder::forward(GGMLRunnerContext* ctx, ggml_tensor* z) {
    auto conv_in     = std::dynamic_pointer_cast<Conv2d>(blocks["conv_in"]);
    auto mid_block_1 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_1"]);
    auto mid_attn_1  = std::dynamic_pointer_cast<AttnBlock>(blocks["mid.attn_1"]);
    auto mid_block_2 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_2"]);
    auto norm_out    = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm_out"]);
    auto conv_out    = std::dynamic_pointer_cast<Conv2d>(blocks["conv_out"]);
    auto h = conv_in->forward(ctx, z);
    h = mid_block_1->forward(ctx, h);
    h = mid_attn_1->forward(ctx, h);
    h = mid_block_2->forward(ctx, h);
    int num_resolutions = static_cast<int>(ch_mult.size());
    for (int i = num_resolutions - 1; i >= 0; i--) {
        for (int j = 0; j < num_res_blocks + 1; j++) {
            std::string name = "up." + std::to_string(i) + ".block." + std::to_string(j);
            auto up_block = std::dynamic_pointer_cast<ResnetBlock>(blocks[name]);
            h = up_block->forward(ctx, h);
        }
        if (i != 0) {
            std::string name = "up." + std::to_string(i) + ".upsample";
            auto up_sample = std::dynamic_pointer_cast<UpSampleBlock>(blocks[name]);
            h = up_sample->forward(ctx, h);
        }
    }
    h = norm_out->forward(ctx, h);
    h = ggml_silu_inplace(ctx->ggml_ctx, h);
    h = conv_out->forward(ctx, h);
    return h;
}

// ──────────────────────────────────────────────────────────────
// 25. Denoiser: KarrasScheduler (denoiser.hpp:282)
// ──────────────────────────────────────────────────────────────
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

// ──────────────────────────────────────────────────────────────
// 26. CompVisDenoiser (denoiser.hpp:645)
// ──────────────────────────────────────────────────────────────
struct CompVisDenoiser : Denoiser {
    Discretization* discretization;
    CompVisDenoiser() : discretization(new LegacyDDPMDiscretization()) {}
    float sigma_max() override { return sigmas[TIMESTEPS - 1]; }
};

// ──────────────────────────────────────────────────────────────
// 27. calculate_alphas_cumprod (stable-diffusion.cpp:288)
// ──────────────────────────────────────────────────────────────
void calculate_alphas_cumprod(float* alphas_cumprod, float linear_start = 0.00085f,
                               float linear_end = 0.0120f, int timesteps = 1000) {
    float ls_sqrt = sqrtf(linear_start), le_sqrt = sqrtf(linear_end);
    float amount = le_sqrt - ls_sqrt, product = 1.0f;
    for (int i = 0; i < timesteps; i++) {
        float beta = ls_sqrt + amount * ((float)i / (timesteps - 1));
        product *= 1.0f - powf(beta, 2.0f);
        alphas_cumprod[i] = product;
    }
}

// ──────────────────────────────────────────────────────────────
// 28. read_safetensors_file (safetensors_io.cpp:87)
// ──────────────────────────────────────────────────────────────
bool read_safetensors_file(const std::string& file_path,
                            std::vector<TensorStorage>& tensor_storages, std::string* error) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) { return false; }
    file.seekg(0, file.end); size_t file_size_ = file.tellg(); file.seekg(0, file.beg);
    if (file_size_ <= 8) { return false; }
    uint8_t header_size_buf[8]; file.read((char*)header_size_buf, 8);
    size_t header_size_ = model_io::read_u64(header_size_buf);
    if (header_size_ >= file_size_) { return false; }
    std::vector<char> header_buf; header_buf.resize(header_size_ + 1); header_buf[header_size_] = '\0';
    file.read(header_buf.data(), header_size_);
    nlohmann::json header_;
    try { header_ = nlohmann::json::parse(header_buf.data()); } catch (...) { return false; }
    tensor_storages.clear();
    for (auto& item : header_.items()) {
        std::string name = item.key(); nlohmann::json tensor_info = item.value();
        if (name == "__metadata__") continue;
        std::string dtype = tensor_info["dtype"]; nlohmann::json shape = tensor_info["shape"];
        if (dtype == "U8") continue;
        size_t begin = tensor_info["data_offsets"][0].get<size_t>();
        size_t end   = tensor_info["data_offsets"][1].get<size_t>();
        // ... dtype conversion and TensorStorage creation
        // (full implementation in safetensors_io.cpp)
    }
    return true;
}

// ──────────────────────────────────────────────────────────────
// 29. sd_ctx_t struct (stable-diffusion.cpp:3213)
// ──────────────────────────────────────────────────────────────
struct sd_ctx_t {
    StableDiffusionGGML* sd = nullptr;
};

// ──────────────────────────────────────────────────────────────
// SDVersion enum (model.h:16)
// ──────────────────────────────────────────────────────────────
enum SDVersion {
    VERSION_SD1, VERSION_SD1_INPAINT, VERSION_SD1_PIX2PIX,
    VERSION_SD1_TINY_UNET, VERSION_SD2, VERSION_SD2_INPAINT,
    VERSION_SD2_TINY_UNET, VERSION_SDXS_512_DS, VERSION_SDXS_09,
    VERSION_SDXL, VERSION_SDXL_INPAINT, VERSION_SDXL_PIX2PIX,
    VERSION_SDXL_VEGA, VERSION_SDXL_SSD1B, VERSION_SVD,
    VERSION_SD3, VERSION_FLUX, VERSION_FLUX_FILL,
    VERSION_FLUX_CONTROLS, VERSION_FLEX_2, VERSION_CHROMA_RADIANCE,
    VERSION_WAN2, VERSION_WAN2_2_I2V, VERSION_WAN2_2_TI2V,
    VERSION_QWEN_IMAGE, VERSION_ANIMA, VERSION_FLUX2,
    VERSION_FLUX2_KLEIN, VERSION_LTXAV, VERSION_HIDREAM_O1,
    VERSION_Z_IMAGE, VERSION_OVIS_IMAGE, VERSION_ERNIE_IMAGE,
    VERSION_LENS, VERSION_LONGCAT, VERSION_PID, VERSION_COUNT,
};
