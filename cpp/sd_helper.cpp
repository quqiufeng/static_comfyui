/* sd_helper.cpp — 按 api.md 文件行号从 /opt/stable-diffusion.cpp 逐函数原文复制 */

/* ====================================================== */
/* sd_ctx_params_init  (src/stable-diffusion.cpp:2924)    */
/* ====================================================== */
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

/* ====================================================== */
/* sd_sample_params_init  (src/stable-diffusion.cpp:3040) */
/* ====================================================== */
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

/* ====================================================== */
/* sd_img_gen_params_init  (src/stable-diffusion.cpp:3100) */
/* ====================================================== */
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

/* ====================================================== */
/* sd_ctx_t struct  (src/stable-diffusion.cpp:3213)       */
/* ====================================================== */
struct sd_ctx_t {
    StableDiffusionGGML* sd = nullptr;
};

/* ====================================================== */
/* new_sd_ctx  (src/stable-diffusion.cpp:3225)            */
/* ====================================================== */
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

/* ====================================================== */
/* free_sd_ctx  (src/stable-diffusion.cpp:3246)           */
/* ====================================================== */
void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx->sd != nullptr) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
    }
    free(sd_ctx);
}

/* ====================================================== */
/* sd_ctx_supports_image_generation  (src/stable-diffusion.cpp:3296) */
/* ====================================================== */
SD_API bool sd_ctx_supports_image_generation(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    return sd_version_supports_image_generation(sd_ctx->sd->version);
}

/* ====================================================== */
/* sd_get_default_sample_method  (src/stable-diffusion.cpp:3310) */
/* ====================================================== */
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

/* ====================================================== */
/* calculate_alphas_cumprod  (src/stable-diffusion.cpp:288) */
/* ====================================================== */
void calculate_alphas_cumprod(float* alphas_cumprod,
                              float linear_start = 0.00085f,
                              float linear_end   = 0.0120f,
                              int timesteps      = TIMESTEPS) {
    float ls_sqrt = sqrtf(linear_start);
    float le_sqrt = sqrtf(linear_end);
    float amount  = le_sqrt - ls_sqrt;
    float product = 1.0f;
    for (int i = 0; i < timesteps; i++) {
        float beta = ls_sqrt + amount * ((float)i / (timesteps - 1));
        product *= 1.0f - powf(beta, 2.0f);
        alphas_cumprod[i] = product;
    }
}

/* ====================================================== */
/* StableDiffusionGGML::init_backend  (src/stable-diffusion.cpp:413) */
/* ====================================================== */
    bool init_backend(const sd_ctx_params_t* sd_ctx_params) {
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

/* ============================================================ */
/* StableDiffusionGGML::init  (src/stable-diffusion.cpp:438)   */
/* ============================================================ */
bool StableDiffusionGGML::init(const sd_ctx_params_t* sd_ctx_params) {
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

        if (strlen(SAFE_STR(sd_ctx_params->embeddings_connectors_path)) > 0) {
            LOG_INFO("loading embeddings connectors from '%s'", sd_ctx_params->embeddings_connectors_path);
            if (!model_loader.init_from_file(sd_ctx_params->embeddings_connectors_path)) {
                LOG_WARN("loading embeddings connectors from '%s' failed", sd_ctx_params->embeddings_connectors_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->audio_vae_path)) > 0) {
            LOG_INFO("loading LTX audio VAE from '%s'", sd_ctx_params->audio_vae_path);
            if (!model_loader.init_from_file(sd_ctx_params->audio_vae_path)) {
                LOG_WARN("loading LTX audio VAE weights from '%s' failed", sd_ctx_params->audio_vae_path);
            } else {
                use_audio_vae = true;
            }
        }

        model_loader.convert_tensors_name();

        version = model_loader.get_sd_version();
        if (version == VERSION_COUNT) {
            // Fallback: if both clip_l and clip_g are provided, assume SDXL
            if (strlen(SAFE_STR(sd_ctx_params->clip_l_path)) > 0 &&
                strlen(SAFE_STR(sd_ctx_params->clip_g_path)) > 0) {
                LOG_WARN("get sd version from file failed, but clip_l and clip_g provided, assuming SDXL");
                version = VERSION_SDXL;
            } else {
                LOG_ERROR("get sd version from file failed: '%s'", SAFE_STR(sd_ctx_params->model_path));
                return false;
            }
        }

        auto& tensor_storage_map = model_loader.get_tensor_storage_map();

        LOG_INFO("Version: %s ", model_version_to_str[version]);
        ggml_type wtype               = (int)sd_ctx_params->wtype < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT)
                                            ? (ggml_type)sd_ctx_params->wtype
                                            : GGML_TYPE_COUNT;
        std::string tensor_type_rules = SAFE_STR(sd_ctx_params->tensor_type_rules);
        if (wtype != GGML_TYPE_COUNT || tensor_type_rules.size() > 0) {
            model_loader.set_wtype_override(wtype, tensor_type_rules);
        }

        std::map<ggml_type, uint32_t> wtype_stat                 = model_loader.get_wtype_stat();
        std::map<ggml_type, uint32_t> conditioner_wtype_stat     = model_loader.get_conditioner_wtype_stat();
        std::map<ggml_type, uint32_t> diffusion_model_wtype_stat = model_loader.get_diffusion_model_wtype_stat();
        std::map<ggml_type, uint32_t> vae_wtype_stat             = model_loader.get_vae_wtype_stat();

        auto wtype_stat_to_str = [](const std::map<ggml_type, uint32_t>& m, int key_width = 8, int value_width = 5) -> std::string {
            std::ostringstream oss;
            bool first = true;
            for (const auto& [type, count] : m) {
                if (!first)
                    oss << "|";
                first = false;
                oss << std::right << std::setw(key_width) << ggml_type_name(type)
                    << ": "
                    << std::left << std::setw(value_width) << count;
            }
            return oss.str();
        };

        LOG_INFO("Weight type stat:                 %s", wtype_stat_to_str(wtype_stat).c_str());
        LOG_INFO("Conditioner weight type stat:     %s", wtype_stat_to_str(conditioner_wtype_stat).c_str());
        LOG_INFO("Diffusion model weight type stat: %s", wtype_stat_to_str(diffusion_model_wtype_stat).c_str());
        LOG_INFO("VAE weight type stat:             %s", wtype_stat_to_str(vae_wtype_stat).c_str());

        LOG_DEBUG("ggml tensor size = %d bytes", (int)sizeof(ggml_tensor));

        if (sd_ctx_params->lora_apply_mode == LORA_APPLY_AUTO) {
            bool have_quantized_weight = false;
            if (wtype != GGML_TYPE_COUNT && ggml_is_quantized(wtype)) {
                have_quantized_weight = true;
            } else {
                for (const auto& [type, _] : wtype_stat) {
                    if (ggml_is_quantized(type)) {
                        have_quantized_weight = true;
                        break;
                    }
                }
            }
            if (have_quantized_weight) {
                apply_lora_immediately = false;
            } else {
                apply_lora_immediately = true;
            }
        } else if (sd_ctx_params->lora_apply_mode == LORA_APPLY_IMMEDIATELY) {
            apply_lora_immediately = true;
        } else {
            apply_lora_immediately = false;
        }

        std::map<std::string, ggml_tensor*> mmap_able_tensors;
        bool enable_mmap_tensors = false;
        bool needs_writable_mmap = false;
        if (sd_ctx_params->enable_mmap) {
            if (apply_lora_immediately) {
                needs_writable_mmap = true;
                LOG_WARN("in mode 'immediately', LoRAs will cause extra memory usage with mmap");
            }
            enable_mmap_tensors = true;
        }

        // split definition to avoid msvc choking on the extra parameter handling
        auto module_can_mmap = [&](SDBackendModule module) {
            return enable_mmap_tensors &&
                   (backend_manager.runtime_backend_is_cpu(module) ||
                    backend_manager.params_backend_is_cpu(module) ||
                    backend_manager.runtime_backend_supports_host_buffer(module));
        };

        auto is_ipadapter_unet_param = [](const std::string& key) -> bool {
            return key.find("attn2.to_k_ip_layer_") != std::string::npos ||
                   key.find("attn2.to_v_ip_layer_") != std::string::npos;
        };

        auto get_param_tensors_p = [&](auto&& model, bool do_mmap, const char* prefix) {
            std::map<std::string, ggml_tensor*> temp;
            model->get_param_tensors(temp, prefix);
            for (const auto& [key, tensor] : temp) {
                if (is_ipadapter_unet_param(key)) continue;
                tensors[key] = tensor;
                if (do_mmap) {
                    mmap_able_tensors[key] = tensor;
                }
            }
        };

        auto get_param_tensors = [&](auto&& model, bool do_mmap) {
            std::map<std::string, ggml_tensor*> temp;
            model->get_param_tensors(temp);
            for (const auto& [key, tensor] : temp) {
                if (is_ipadapter_unet_param(key)) continue;
                tensors[key] = tensor;
                if (do_mmap) {
                    mmap_able_tensors[key] = tensor;
                }
            }
        };

        if (sd_version_is_control(version)) {
            // Might need vae encode for control cond
            vae_decode_only = false;
        }
        bool tae_preview_only = sd_ctx_params->tae_preview_only;
        if (version == VERSION_SDXS_512_DS || version == VERSION_SDXS_09) {
            tae_preview_only = false;
            use_tae          = true;
        }

        if (sd_ctx_params->circular_x || sd_ctx_params->circular_y) {
            LOG_INFO("Using circular padding for convolutions");
        }

        const size_t max_graph_vram_bytes = sd::ggml_graph_cut::max_vram_gib_to_bytes(max_vram);

        {
            if (!ensure_backend_pair(SDBackendModule::TE) ||
                !ensure_backend_pair(SDBackendModule::DIFFUSION)) {
                return false;
            }

            if (sd_version_is_sd3(version)) {
                cond_stage_model = std::make_shared<SD3CLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                     params_backend_for(SDBackendModule::TE),
                                                                     tensor_storage_map);
                diffusion_model  = std::make_shared<MMDiTRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                params_backend_for(SDBackendModule::DIFFUSION),
                                                                tensor_storage_map,
                                                                "model.diffusion_model");
            } else if (sd_version_is_pid(version)) {
                vae_decode_only  = false;
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 params_backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version);
                diffusion_model  = std::make_shared<Pid::PiDRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                   params_backend_for(SDBackendModule::DIFFUSION),
                                                                   tensor_storage_map,
                                                                   "model.diffusion_model.net");
            } else if (sd_version_is_flux(version)) {
                bool is_chroma = false;
                for (auto pair : tensor_storage_map) {
                    if (pair.first.find("distilled_guidance_layer.in_proj.weight") != std::string::npos) {
                        is_chroma = true;
                        break;
                    }
                }
                if (is_chroma) {
                    if ((sd_ctx_params->flash_attn || sd_ctx_params->diffusion_flash_attn) && sd_ctx_params->chroma_use_dit_mask) {
                        LOG_WARN(
                            "!!!It looks like you are using Chroma with flash attention. "
                            "This is currently unsupported. "
                            "If you find that the generated images are broken, "
                            "try either disabling flash attention or specifying "
                            "--chroma-disable-dit-mask as a workaround.");
                    }

                    cond_stage_model = std::make_shared<T5CLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                        params_backend_for(SDBackendModule::TE),
                                                                        tensor_storage_map,
                                                                        sd_ctx_params->chroma_use_t5_mask,
                                                                        sd_ctx_params->chroma_t5_mask_pad);
                } else if (version == VERSION_OVIS_IMAGE) {
                    cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                     params_backend_for(SDBackendModule::TE),
                                                                     tensor_storage_map,
                                                                     version,
                                                                     "",
                                                                     false);
                } else {
                    cond_stage_model = std::make_shared<FluxCLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                          params_backend_for(SDBackendModule::TE),
                                                                          tensor_storage_map);
                }
                diffusion_model = std::make_shared<Flux::FluxRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                     params_backend_for(SDBackendModule::DIFFUSION),
                                                                     tensor_storage_map,
                                                                     "model.diffusion_model",
                                                                     version,
                                                                     sd_ctx_params->chroma_use_dit_mask);
            } else if (sd_version_is_flux2(version)) {
                bool is_chroma   = false;
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 params_backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version);
                diffusion_model  = std::make_shared<Flux::FluxRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                     params_backend_for(SDBackendModule::DIFFUSION),
                                                                     tensor_storage_map,
                                                                     "model.diffusion_model",
                                                                     version,
                                                                     sd_ctx_params->chroma_use_dit_mask);
            } else if (sd_version_is_ltxav(version)) {
                cond_stage_model = std::make_shared<LTXAVEmbedder>(backend_for(SDBackendModule::TE),
                                                                   params_backend_for(SDBackendModule::TE),
                                                                   tensor_storage_map);
                diffusion_model  = std::make_shared<LTXV::LTXAVRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                      params_backend_for(SDBackendModule::DIFFUSION),
                                                                      tensor_storage_map,
                                                                      "model.diffusion_model");
            } else if (sd_version_is_wan(version)) {
                cond_stage_model = std::make_shared<T5CLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                    params_backend_for(SDBackendModule::TE),
                                                                    tensor_storage_map,
                                                                    true,
                                                                    0,
                                                                    true);
                diffusion_model  = std::make_shared<WAN::WanRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                   params_backend_for(SDBackendModule::DIFFUSION),
                                                                   tensor_storage_map,
// ...(cont) init 551-800
/* ============================================================ */
/* generate_image  (src/stable-diffusion.cpp:4626)              */
/* ============================================================ */
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
/* ============================================================ */
/* KarrasScheduler  (src/denoiser.hpp:282)                     */
/* ============================================================ */
struct KarrasScheduler : SigmaScheduler {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) override {
        // These *COULD* be function arguments here,
        // but does anybody ever bother to touch them?
        float rho = 7.f;

        if (sigma_min <= 1e-6f) {
            sigma_min = 1e-6f;
        }

        std::vector<float> result(n + 1);

        float min_inv_rho = pow(sigma_min, (1.f / rho));
        float max_inv_rho = pow(sigma_max, (1.f / rho));
        for (uint32_t i = 0; i < n; i++) {
            // Eq. (5) from Karras et al 2022
            result[i] = pow(max_inv_rho + (float)i / ((float)n - 1.f) * (min_inv_rho - max_inv_rho), rho);
        }
        result[n] = 0.;
        return result;
    }
};
/* ============================================================ */
/* CompVisDenoiser::sigma_max  (src/denoiser.hpp:645)          */
/* ============================================================ */
    float sigma_max() override {
        return sigmas[TIMESTEPS - 1];
    }

