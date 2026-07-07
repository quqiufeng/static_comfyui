#include "ggml.h"
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

#define LOG_INFO(...)  do{fprintf(stderr,__VA_ARGS__);fputc('\n',stderr);}while(0)

// ─── stable-diffusion.h 类型（完整字段名） ───
enum rng_type_t { STD_DEFAULT_RNG, CUDA_RNG, CPU_RNG, RNG_TYPE_COUNT };
enum sample_method_t { EULER_SAMPLE_METHOD, EULER_A_SAMPLE_METHOD, HEUN_SAMPLE_METHOD, DPM2_SAMPLE_METHOD, DPMPP2S_A_SAMPLE_METHOD, DPMPP2M_SAMPLE_METHOD, DPMPP2Mv2_SAMPLE_METHOD, IPNDM_SAMPLE_METHOD, IPNDM_V_SAMPLE_METHOD, LCM_SAMPLE_METHOD, DDIM_TRAILING_SAMPLE_METHOD, TCD_SAMPLE_METHOD, RES_MULTISTEP_SAMPLE_METHOD, RES_2S_SAMPLE_METHOD, ER_SDE_SAMPLE_METHOD, EULER_CFG_PP_SAMPLE_METHOD, EULER_A_CFG_PP_SAMPLE_METHOD, EULER_GE_SAMPLE_METHOD, SAMPLE_METHOD_COUNT };
enum scheduler_t { DISCRETE_SCHEDULER, KARRAS_SCHEDULER, EXPONENTIAL_SCHEDULER, AYS_SCHEDULER, GITS_SCHEDULER, SGM_UNIFORM_SCHEDULER, SIMPLE_SCHEDULER, SMOOTHSTEP_SCHEDULER, KL_OPTIMAL_SCHEDULER, LCM_SCHEDULER, BONG_TANGENT_SCHEDULER, LTX2_SCHEDULER, SCHEDULER_COUNT };
enum prediction_t { EPS_PRED, V_PRED, EDM_V_PRED, FLOW_PRED, FLUX_FLOW_PRED, FLUX2_FLOW_PRED, PREDICTION_COUNT };
enum sd_type_t { SD_TYPE_F32=0, SD_TYPE_F16=1, SD_TYPE_Q4_0=2, SD_TYPE_Q4_1=3, SD_TYPE_Q5_0=6, SD_TYPE_Q5_1=7, SD_TYPE_Q8_0=8, SD_TYPE_Q8_1=9, SD_TYPE_Q2_K=10, SD_TYPE_Q3_K=11, SD_TYPE_Q4_K=12, SD_TYPE_Q5_K=13, SD_TYPE_Q6_K=14, SD_TYPE_Q8_K=15, SD_TYPE_IQ2_XXS=16, SD_TYPE_IQ2_XS=17, SD_TYPE_IQ3_XXS=18, SD_TYPE_IQ1_S=19, SD_TYPE_IQ4_NL=20, SD_TYPE_IQ3_S=21, SD_TYPE_IQ2_S=22, SD_TYPE_IQ4_XS=23, SD_TYPE_I8=24, SD_TYPE_I16=25, SD_TYPE_I32=26, SD_TYPE_I64=27, SD_TYPE_F64=28, SD_TYPE_IQ1_M=29, SD_TYPE_BF16=30, SD_TYPE_TQ1_0=34, SD_TYPE_TQ2_0=35, SD_TYPE_MXFP4=39, SD_TYPE_NVFP4=40, SD_TYPE_Q1_0=41, SD_TYPE_COUNT=42 };
enum sd_cache_mode_t { SD_CACHE_DISABLED=0, SD_CACHE_EASYCACHE, SD_CACHE_UCACHE, SD_CACHE_DBCACHE, SD_CACHE_TAYLORSEER, SD_CACHE_CACHE_DIT, SD_CACHE_SPECTRUM };

typedef struct { uint32_t width; uint32_t height; uint32_t channel; uint8_t* data; } sd_image_t;
typedef struct { float txt_cfg; } sd_guidance_params_t;
typedef struct { sd_guidance_params_t guidance; enum scheduler_t scheduler; enum sample_method_t sample_method; int sample_steps; float eta; int shifted_timestep; float* custom_sigmas; int custom_sigmas_count; float flow_shift; const char* extra_sample_args; } sd_sample_params_t;
typedef struct { const char* name; const char* path; } sd_embedding_t;
typedef struct { bool is_high_noise; float multiplier; const char* path; } sd_lora_t;
typedef struct { bool enabled; int tile_size_x; int tile_size_y; float target_overlap; float rel_size_x; float rel_size_y; const char* extra_tiling_args; } sd_tiling_params_t;
typedef struct { enum sd_cache_mode_t mode; float reuse_threshold; float start_percent; float end_percent; float error_decay_rate; bool use_relative_threshold; } sd_cache_params_t;
typedef struct { bool enabled; int upscaler; const char* model_path; float scale; int target_width; int target_height; int steps; float denoising_strength; float* custom_sigmas; int custom_sigmas_count; } sd_hires_params_t;

typedef struct { const char* model_path; const char* clip_l_path; const char* clip_g_path; const char* clip_vision_path; const char* t5xxl_path; const char* diffusion_model_path; const char* vae_path; const char* taesd_path; const char* control_net_path; bool vae_decode_only; bool free_params_immediately; int n_threads; enum sd_type_t wtype; enum rng_type_t rng_type; enum prediction_t prediction; bool offload_params_to_cpu; float max_vram; bool keep_vae_on_cpu; bool flash_attn; const char* backend; } sd_ctx_params_t;
typedef struct { const char* prompt; const char* negative_prompt; int clip_skip; sd_image_t init_image; sd_image_t mask_image; int width; int height; sd_sample_params_t sample_params; float strength; int64_t seed; int batch_count; sd_image_t control_image; float control_strength; sd_tiling_params_t vae_tiling_params; sd_hires_params_t hires; struct { bool enabled; float b1; float b2; float s1; float s2; } freeu; const float* ipadapter_tokens; int ipadapter_num_tokens; float ipadapter_weight; } sd_img_gen_params_t;

// ─── 内部类型前向声明 ───
struct SDBackendManager{bool init(const char*,const char*,bool,bool,bool,bool,std::string*e){if(e)*e="";return true;}};
struct RNG{void manual_seed(int64_t){}};
struct Denoiser{};

// ─── StableDiffusionGGML ───
class StableDiffusionGGML {
public:
    SDBackendManager backend_manager;
    bool vae_decode_only=false,free_params_immediately=false;
    std::shared_ptr<RNG> rng=std::make_shared<RNG>();
    int n_threads=-1;
    bool offload_params_to_cpu=false; float max_vram=0.f;
    std::string backend_spec;
    bool freeu_enabled=false,sag_enabled=false;
    float freeu_b1=1.3f,freeu_b2=1.4f,freeu_s1=0.9f,freeu_s2=0.2f,sag_scale=1.f;
    std::shared_ptr<Denoiser> denoiser=std::make_shared<Denoiser>();
    sd_tiling_params_t vae_tiling_params={};
    StableDiffusionGGML()=default; ~StableDiffusionGGML()=default;
    bool init(const sd_ctx_params_t*p);
    void set_flow_shift(float){} void apply_loras(const sd_lora_t*,uint32_t){}
    bool is_flow_denoiser(){return false;} void lora_stat(){}
};

struct sd_ctx_t{StableDiffusionGGML*sd=nullptr;};

// ─── API 实现 ───
void sd_ctx_params_init(sd_ctx_params_t*p){
    *p={};p->vae_decode_only=true;p->free_params_immediately=true;
    p->n_threads=8;p->wtype=SD_TYPE_COUNT;p->rng_type=CUDA_RNG;
    p->prediction=PREDICTION_COUNT;p->offload_params_to_cpu=false;
    p->max_vram=0.f;p->keep_vae_on_cpu=false;p->backend=nullptr;
}
void sd_sample_params_init(sd_sample_params_t*p){
    *p={};p->scheduler=KARRAS_SCHEDULER;p->sample_method=EULER_A_SAMPLE_METHOD;
    p->sample_steps=20;p->eta=0.f;p->shifted_timestep=-1;
    p->custom_sigmas=nullptr;p->custom_sigmas_count=0;p->flow_shift=INFINITY;
}
void sd_img_gen_params_init(sd_img_gen_params_t*p){
    *p={};sd_sample_params_init(&p->sample_params);
    p->width=512;p->height=512;p->strength=0.75f;p->seed=-1;
    p->batch_count=1;p->clip_skip=-1;p->ipadapter_tokens=nullptr;
}
sd_ctx_t*new_sd_ctx(const sd_ctx_params_t*p){
    auto c=(sd_ctx_t*)malloc(sizeof(sd_ctx_t));if(!c)return nullptr;
    c->sd=new StableDiffusionGGML();if(!c->sd){free(c);return nullptr;}
    if(!c->sd->init(p)){delete c->sd;free(c);return nullptr;}return c;
}
void free_sd_ctx(sd_ctx_t*c){if(c->sd)delete c->sd;free(c);}
bool StableDiffusionGGML::init(const sd_ctx_params_t*p){
    n_threads=p->n_threads;vae_decode_only=p->vae_decode_only;
    free_params_immediately=p->free_params_immediately;
    offload_params_to_cpu=p->offload_params_to_cpu;max_vram=p->max_vram;
    backend_spec=p->backend?p->backend:"";return true;
}

// ─── GenerationRequest (stable-diffusion.cpp:3388) ───
struct GenerationRequest {
    std::string prompt, negative_prompt;
    int width=-1,height=-1,clip_skip=-1,vae_scale_factor=-1;
    int64_t seed=-1; bool use_uncond=false;
    const sd_cache_params_t* cache_params=nullptr;
    int batch_count=1,shifted_timestep=0;
    float strength=1.f,control_strength=0.f,eta=0.f;
    sd_guidance_params_t guidance;
    enum sample_method_t sample_method=EULER_A_SAMPLE_METHOD;
    int sample_steps=20;
    sd_hires_params_t hires={};
    GenerationRequest(sd_ctx_t*,const sd_img_gen_params_t*p){
        prompt=p->prompt?p->prompt:"";negative_prompt=p->negative_prompt?p->negative_prompt:"";
        width=p->width;height=p->height;seed=p->seed;batch_count=p->batch_count;
        strength=p->strength;control_strength=p->control_strength;
        guidance=p->sample_params.guidance;sample_method=p->sample_params.sample_method;
        sample_steps=p->sample_params.sample_steps;eta=p->sample_params.eta;
        hires=p->hires;shifted_timestep=p->sample_params.shifted_timestep;
        clip_skip=p->clip_skip;vae_scale_factor=8;
        cache_params=nullptr;
    }
};

// ─── generate_image (stable-diffusion.cpp:4626) ───
sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params) {
    if(!sd_ctx||!sd_img_gen_params)return nullptr;
    fprintf(stderr,"generate_image: SDXL pipeline stub\n");
    (void)sd_ctx;(void)sd_img_gen_params;
    return nullptr;
}
