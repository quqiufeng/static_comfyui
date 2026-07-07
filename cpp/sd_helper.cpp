#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include "json.hpp"
using json = nlohmann::json;

/* ─── 完整 stable-diffusion.h 类型 ─── */
enum rng_type_t { STD_DEFAULT_RNG, CUDA_RNG, CPU_RNG, RNG_TYPE_COUNT };
enum sample_method_t { EULER_SAMPLE_METHOD, EULER_A_SAMPLE_METHOD, HEUN_SAMPLE_METHOD, DPM2_SAMPLE_METHOD, DPMPP2S_A_SAMPLE_METHOD, DPMPP2M_SAMPLE_METHOD, DPMPP2Mv2_SAMPLE_METHOD, IPNDM_SAMPLE_METHOD, IPNDM_V_SAMPLE_METHOD, LCM_SAMPLE_METHOD, DDIM_TRAILING_SAMPLE_METHOD, TCD_SAMPLE_METHOD, RES_MULTISTEP_SAMPLE_METHOD, RES_2S_SAMPLE_METHOD, ER_SDE_SAMPLE_METHOD, EULER_CFG_PP_SAMPLE_METHOD, EULER_A_CFG_PP_SAMPLE_METHOD, EULER_GE_SAMPLE_METHOD, SAMPLE_METHOD_COUNT };
enum scheduler_t { DISCRETE_SCHEDULER, KARRAS_SCHEDULER, EXPONENTIAL_SCHEDULER, AYS_SCHEDULER, GITS_SCHEDULER, SGM_UNIFORM_SCHEDULER, SIMPLE_SCHEDULER, SMOOTHSTEP_SCHEDULER, KL_OPTIMAL_SCHEDULER, LCM_SCHEDULER, BONG_TANGENT_SCHEDULER, LTX2_SCHEDULER, SCHEDULER_COUNT };
enum prediction_t { EPS_PRED, V_PRED, EDM_V_PRED, FLOW_PRED, FLUX_FLOW_PRED, FLUX2_FLOW_PRED, PREDICTION_COUNT };
enum sd_type_t { SD_TYPE_F32=0, SD_TYPE_F16=1, SD_TYPE_Q4_0=2, SD_TYPE_Q4_1=3, SD_TYPE_Q5_0=6, SD_TYPE_Q5_1=7, SD_TYPE_Q8_0=8, SD_TYPE_Q8_1=9, SD_TYPE_Q2_K=10, SD_TYPE_Q3_K=11, SD_TYPE_Q4_K=12, SD_TYPE_Q5_K=13, SD_TYPE_Q6_K=14, SD_TYPE_Q8_K=15, SD_TYPE_IQ2_XXS=16, SD_TYPE_IQ2_XS=17, SD_TYPE_IQ3_XXS=18, SD_TYPE_IQ1_S=19, SD_TYPE_IQ4_NL=20, SD_TYPE_IQ3_S=21, SD_TYPE_IQ2_S=22, SD_TYPE_IQ4_XS=23, SD_TYPE_I8=24, SD_TYPE_I16=25, SD_TYPE_I32=26, SD_TYPE_I64=27, SD_TYPE_F64=28, SD_TYPE_IQ1_M=29, SD_TYPE_BF16=30, SD_TYPE_TQ1_0=34, SD_TYPE_TQ2_0=35, SD_TYPE_MXFP4=39, SD_TYPE_NVFP4=40, SD_TYPE_Q1_0=41, SD_TYPE_COUNT=42 };
enum sd_log_level_t { SD_LOG_DEBUG, SD_LOG_INFO, SD_LOG_WARN, SD_LOG_ERROR };
enum lora_apply_mode_t { LORA_APPLY_AUTO, LORA_APPLY_IMMEDIATELY, LORA_APPLY_AT_RUNTIME, LORA_APPLY_MODE_COUNT };
enum sd_vae_format_t { SD_VAE_FORMAT_AUTO=-1, SD_VAE_FORMAT_FLUX, SD_VAE_FORMAT_SD3, SD_VAE_FORMAT_FLUX2, SD_VAE_FORMAT_COUNT };
enum sd_cache_mode_t { SD_CACHE_DISABLED=0, SD_CACHE_EASYCACHE, SD_CACHE_UCACHE, SD_CACHE_DBCACHE, SD_CACHE_TAYLORSEER, SD_CACHE_CACHE_DIT, SD_CACHE_SPECTRUM };

typedef struct { int* layers; size_t layer_count; float layer_start; float layer_end; float scale; } sd_slg_params_t;
typedef struct { float txt_cfg; float img_cfg; float distilled_guidance; sd_slg_params_t slg; } sd_guidance_params_t;
typedef struct { sd_guidance_params_t guidance; enum scheduler_t scheduler; enum sample_method_t sample_method; int sample_steps; float eta; int shifted_timestep; float* custom_sigmas; int custom_sigmas_count; float flow_shift; const char* extra_sample_args; } sd_sample_params_t;
typedef struct { const char* name; const char* path; } sd_embedding_t;
typedef struct { uint32_t width; uint32_t height; uint32_t channel; uint8_t* data; } sd_image_t;
typedef struct { sd_image_t* id_images; int id_images_count; const char* id_embed_path; float style_strength; } sd_pm_params_t;
typedef struct { bool enabled; float scale; } sd_sag_params_t;
typedef struct { bool enabled; float percentile; float mimic_scale; float threshold_percentile; } sd_dynamic_cfg_params_t;
typedef struct { bool enabled; float b1; float b2; float s1; float s2; } sd_freeu_params_t;
typedef struct { bool enabled; int tile_size_x; int tile_size_y; float target_overlap; float rel_size_x; float rel_size_y; const char* extra_tiling_args; } sd_tiling_params_t;
typedef struct { bool is_high_noise; float multiplier; const char* path; } sd_lora_t;
typedef struct { enum sd_cache_mode_t mode; float reuse_threshold; float start_percent; float end_percent; float error_decay_rate; bool use_relative_threshold; bool reset_error_on_compute; } sd_cache_params_t;
typedef struct { bool enabled; int upscaler; const char* model_path; float scale; int target_width; int target_height; int steps; float denoising_strength; float* custom_sigmas; int custom_sigmas_count; } sd_hires_params_t;
typedef struct {
    const char* model_path; const char* clip_l_path; const char* clip_g_path;
    const char* clip_vision_path; const char* t5xxl_path; const char* llm_path;
    const char* llm_vision_path; const char* diffusion_model_path;
    const char* high_noise_diffusion_model_path; const char* vae_path;
    const char* taesd_path; const char* control_net_path;
    const sd_embedding_t* embeddings; uint32_t embedding_count;
    bool vae_decode_only; bool free_params_immediately;
    int n_threads; enum sd_type_t wtype; enum rng_type_t rng_type;
    enum rng_type_t sampler_rng_type; enum prediction_t prediction;
    enum lora_apply_mode_t lora_apply_mode; bool offload_params_to_cpu;
    bool keep_vae_on_cpu; bool enable_mmap; bool keep_clip_on_cpu; bool keep_control_net_on_cpu; bool diffusion_flash_attn; bool chroma_use_dit_mask; bool chroma_use_t5_mask; int chroma_t5_mask_pad; bool circular_x; bool circular_y;
    enum sd_vae_format_t vae_format; float max_vram;
    const char* backend; const char* params_backend;
    bool ipadapter_unet_mode; const char* ipadapter_unet_weights_path;
} sd_ctx_params_t;
typedef struct {
    const sd_lora_t* loras; uint32_t lora_count;
    const char* prompt; const char* negative_prompt; int clip_skip;
    sd_image_t init_image; sd_image_t* ref_images; int ref_images_count; sd_image_t mask_image;
    int width; int height; sd_sample_params_t sample_params;
    float strength; int64_t seed; int batch_count;
    float control_strength; sd_tiling_params_t vae_tiling_params;
    sd_pm_params_t pm_params; sd_cache_params_t cache;
    sd_hires_params_t hires; sd_freeu_params_t freeu;
    sd_sag_params_t sag; sd_dynamic_cfg_params_t dynamic_cfg;
    const float* ipadapter_tokens; int ipadapter_num_tokens;
    float ipadapter_weight; bool ipadapter_unet_mode;
} sd_img_gen_params_t;

void sd_cache_params_init(sd_cache_params_t* p) { *p = {}; p->reuse_threshold = INFINITY; }
void sd_hires_params_init(sd_hires_params_t* p) { *p = {}; p->scale = 2.f; }

/* sd_ctx_params_init (2924) */
void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params) {
    *sd_ctx_params                         = {};
    sd_ctx_params->vae_decode_only         = true;
    sd_ctx_params->free_params_immediately = true;
    sd_ctx_params->n_threads               = 8;
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
/* sd_sample_params_init (3040) */
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
/* sd_img_gen_params_init (3100) */
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
    memset(&sd_img_gen_params->vae_tiling_params, 0, sizeof(sd_tiling_params_t)); sd_img_gen_params->vae_tiling_params.target_overlap = 0.5f;
    sd_cache_params_init(&sd_img_gen_params->cache);
    sd_hires_params_init(&sd_img_gen_params->hires);
}

/* sd_ctx_t (3213) */

class StableDiffusionGGML { public: bool init(const sd_ctx_params_t*); };
struct sd_ctx_t { StableDiffusionGGML* sd = nullptr; };

/* new_sd_ctx (3225) */
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
/* free_sd_ctx (3246) */
void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx->sd != nullptr) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
    }
    free(sd_ctx);
}

/* StableDiffusionGGML::init (stub) */
bool StableDiffusionGGML::init(const sd_ctx_params_t* p) { (void)p; return true; }
