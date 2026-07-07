#ifndef ENGINE_H
#define ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

enum sd_type_t {
    SD_TYPE_F32  = 0,
    SD_TYPE_F16  = 1,
};

enum rng_type_t {
    STD_DEFAULT_RNG,
    CUDA_RNG,
};

enum sample_method_t {
    EULER_SAMPLE_METHOD,
};

enum scheduler_t {
    DISCRETE_SCHEDULER,
    KARRAS_SCHEDULER,
};

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint8_t* data;
} sd_image_t;

typedef struct {
    float txt_cfg;
} sd_guidance_params_t;

typedef struct {
    sd_guidance_params_t guidance;
    enum scheduler_t scheduler;
    enum sample_method_t sample_method;
    int sample_steps;
} sd_sample_params_t;

typedef struct {
    const char* model_path;
    const char* clip_l_path;
    const char* clip_g_path;
    int n_threads;
    enum sd_type_t wtype;
    enum rng_type_t rng_type;
    bool keep_vae_on_cpu;
    const char* backend;
    const char* params_backend;
} sd_ctx_params_t;

typedef struct {
    const char* prompt;
    const char* negative_prompt;
    int width;
    int height;
    int64_t seed;
    int batch_count;
    sd_sample_params_t sample_params;
} sd_img_gen_params_t;

typedef struct sdxl_ctx sd_ctx_t;

void sd_ctx_params_init(sd_ctx_params_t* p);
void sd_img_gen_params_init(sd_img_gen_params_t* p);
void sd_sample_params_init(sd_sample_params_t* p);

sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* p);
void      free_sd_ctx(sd_ctx_t* ctx);
sd_image_t* generate_image(sd_ctx_t* ctx, const sd_img_gen_params_t* p);

#ifdef __cplusplus
}
#endif

#endif
