// stable-diffusion-cli_v2.cpp — SDXL txt2img using stable-diffusion.h API
// Reference implementation following sd_helper.cpp patterns
#include "stable-diffusion.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

static void save_ppm(const char* path, const unsigned char* data, int w, int h, int c) {
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(data, 1, w * h * c, f);
    fclose(f);
}

int main(int argc, char** argv) {
    const char* model  = "/data/models/image/sd_xl_base_1.0.safetensors";
    const char* clip_l = "/data/models/image/clip_l.safetensors";
    const char* clip_g = "/data/models/image/clip_g.safetensors";
    const char* prompt = "solo,single woman,half body portrait of a young woman, "
                         "soft natural lighting, elegant pose, studio lighting, "
                         "sharp eyes, clean white background, medium close up";
    const char* neg    = "blurry, low quality, ugly";
    const char* output = "/tmp/sd_cli_v2";
    int W = 1024, H = 1024, steps = 20, seed = 42;
    float cfg = 7.0f;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i],"-m")&&i+1<argc) model  = argv[++i];
        else if (!strcmp(argv[i],"--clip-l")&&i+1<argc) clip_l = argv[++i];
        else if (!strcmp(argv[i],"--clip-g")&&i+1<argc) clip_g = argv[++i];
        else if (!strcmp(argv[i],"-p")&&i+1<argc) prompt = argv[++i];
        else if (!strcmp(argv[i],"-n")&&i+1<argc) neg    = argv[++i];
        else if (!strcmp(argv[i],"-o")&&i+1<argc) output = argv[++i];
        else if (!strcmp(argv[i],"-W")&&i+1<argc) W      = atoi(argv[++i]);
        else if (!strcmp(argv[i],"-H")&&i+1<argc) H      = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--steps")&&i+1<argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--cfg")&&i+1<argc) cfg   = atof(argv[++i]);
        else if (!strcmp(argv[i],"-s")&&i+1<argc) seed  = atoi(argv[++i]);
    }

    fprintf(stderr, "stable-diffusion-cli_v2\n");
    fprintf(stderr, "  model:  %s\n", model);
    fprintf(stderr, "  prompt: %s\n", prompt);
    fprintf(stderr, "  size:   %dx%d steps=%d cfg=%.1f seed=%d\n", W, H, steps, cfg, seed);

    // 1. Init context (sd_ctx_params_init → new_sd_ctx → init_backend → ModelLoader → GGMLRunner)
    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path    = model;
    ctx_params.clip_l_path   = clip_l;
    ctx_params.clip_g_path   = clip_g;
    ctx_params.n_threads     = 8;
    ctx_params.wtype         = SD_TYPE_F16;
    ctx_params.rng_type      = STD_DEFAULT_RNG;
    ctx_params.keep_vae_on_cpu = false;

    sd_ctx_t* ctx = new_sd_ctx(&ctx_params);
    if (!ctx) { fprintf(stderr, "new_sd_ctx failed\n"); return 1; }

    // 2. Generate image (sd_img_gen_params_init → generate_image)
    sd_img_gen_params_t img_params;
    sd_img_gen_params_init(&img_params);
    img_params.prompt           = prompt;
    img_params.negative_prompt  = neg;
    img_params.width            = W;
    img_params.height           = H;
    img_params.seed             = seed;
    img_params.batch_count      = 1;
    img_params.sample_params.sample_steps               = steps;
    img_params.sample_params.guidance.txt_cfg            = cfg;
    img_params.sample_params.sample_method               = EULER_SAMPLE_METHOD;
    img_params.sample_params.scheduler                   = KARRAS_SCHEDULER;

    sd_image_t* images = generate_image(ctx, &img_params);
    if (!images) { fprintf(stderr, "generate_image failed\n"); free_sd_ctx(ctx); return 1; }

    // 3. Save output (PPM)
    char ppm_path[1024];
    snprintf(ppm_path, sizeof(ppm_path), "%s.ppm", output);
    save_ppm(ppm_path, images[0].data, images[0].width, images[0].height, 3);
    fprintf(stderr, "Saved %s (%dx%d)\n", ppm_path, images[0].width, images[0].height);

    // 4. Cleanup (free_sd_ctx)
    free(images[0].data);
    free(images);
    free_sd_ctx(ctx);
    return 0;
}
