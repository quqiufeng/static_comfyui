// sdxl_pipeline.cpp - C++ equivalent of sdxl_pipeline.py using /opt/sd stable-diffusion.cpp C API
// Build:
//   g++ -O3 -std=gnu++17 sdxl_pipeline.cpp \
//     -I/opt/sd/include \
//     /opt/sd/build/libstable-diffusion.a \
//     /opt/sd/build/ggml/src/libggml.a /opt/sd/build/ggml/src/libggml-base.a /opt/sd/build/ggml/src/libggml-cpu.a \
//     -Wl,--whole-archive /opt/sd/build/ggml/src/ggml-cuda/libggml-cuda.a -Wl,--no-whole-archive \
//     -ldl -lpthread -lm -lgomp -lpng -lz \
//     /data/cuda/lib64/libcudart.so.12 /data/cuda/lib64/libcublas.so.12 /data/cuda/lib64/libcublasLt.so.12 /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
//     -Wl,-rpath,/data/cuda/lib64 -Wl,-rpath,/usr/lib/x86_64-linux-gnu \
//     -o sdxl_pipeline
#include "stable-diffusion.h"
#include <png.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

static bool save_png(const char* path, const uint8_t* data, int w, int h, int channels) {
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open %s for writing\n", path);
        return false;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        fclose(fp);
        return false;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        fclose(fp);
        return false;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
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
    fclose(fp);
    return true;
}

int main(int argc, char** argv) {
    const char* model  = "/data/models/image/Juggernaut-XI-byRunDiffusion.safetensors";
    const char* prompt = "solo,single woman,half body portrait of a young woman, "
                         "soft natural lighting, elegant pose, studio lighting, "
                         "sharp eyes, clean white background, medium close up";
    const char* neg    = "blurry, low quality, worst quality, jpeg artifacts, noise, grain, "
                         "soft focus, out of focus, hazy, unclear, bad anatomy, deformed, "
                         "border artifacts, edge distortion, tiling artifacts, edge artifacts, "
                         "frame distortion, warped edges, stretched proportions, asymmetrical face, "
                         "off-center, cropped, out of frame, partial face, cut off, incomplete head, "
                         "cropped head, watermark, text, logo, signature, cropped shoulders, "
                         "embedding:EasyNegative, embedding:bad-hands-5";
    const char* output = "/home/quqiufeng/sdxl_pipeline_cpp.png";

    // Base pass: 1920x1080, HiRes target: 2560x1440 (same as sdxl_pipeline.py)
    int W = 1280, H = 720;
    int target_W = 2560, target_H = 1440;
    int steps = 30;
    int hires_steps = 45;
    float cfg = 4.0f;
    float hires_strength = 0.35f;
    int seed = 42;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-m") && i + 1 < argc) model  = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) neg    = argv[++i];
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) output = argv[++i];
        else if (!strcmp(argv[i], "-W") && i + 1 < argc) W      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-H") && i + 1 < argc) H      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--target-W") && i + 1 < argc) target_W = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--target-H") && i + 1 < argc) target_H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hires-steps") && i + 1 < argc) hires_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cfg") && i + 1 < argc) cfg = atof(argv[++i]);
        else if (!strcmp(argv[i], "--hires-strength") && i + 1 < argc) hires_strength = atof(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = atoi(argv[++i]);
    }

    fprintf(stderr, "sdxl_pipeline.cpp\n");
    fprintf(stderr, "  model:  %s\n", model);
    fprintf(stderr, "  output: %s\n", output);
    fprintf(stderr, "  base:   %dx%d -> target: %dx%d\n", W, H, target_W, target_H);
    fprintf(stderr, "  steps:  %d -> %d (HiRes)\n", steps, hires_steps);
    fprintf(stderr, "  cfg:    %.1f, hires_strength: %.2f, seed: %d\n", cfg, hires_strength, seed);

    // 1. Create context. Load UNet/VAE from SDXL checkpoint, external CLIP-L + CLIP-G for text encoding.
    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path  = model;
    ctx_params.clip_l_path = "/data/models/image/clip_l_sdcpp.safetensors";
    ctx_params.clip_g_path = "/data/models/image/clip_g_sdcpp.safetensors";
    ctx_params.vae_path    = nullptr;
    ctx_params.n_threads   = 8;
    ctx_params.wtype       = SD_TYPE_F16;
    ctx_params.rng_type    = STD_DEFAULT_RNG;
    ctx_params.diffusion_flash_attn = true;

    sd_ctx_t* ctx = new_sd_ctx(&ctx_params);
    if (!ctx) {
        fprintf(stderr, "new_sd_ctx failed\n");
        return 1;
    }

    // 2. Setup image generation parameters aligned with sdxl_pipeline.py
    sd_img_gen_params_t img_params;
    sd_img_gen_params_init(&img_params);
    img_params.prompt          = prompt;
    img_params.negative_prompt = neg;
    img_params.width           = W;
    img_params.height          = H;
    img_params.seed            = seed;
    img_params.batch_count     = 1;

    img_params.sample_params.guidance.txt_cfg = cfg;
    img_params.sample_params.sample_method    = DPMPP2M_SDE_SAMPLE_METHOD;
    img_params.sample_params.scheduler        = KARRAS_SCHEDULER;
    img_params.sample_params.sample_steps     = steps;
    img_params.sample_params.eta              = 0.0f;

    // // FreeU (currently disabled - caused overfitting)
    // img_params.freeu.enabled = true;
    // img_params.freeu.b1      = 1.05f;
    // img_params.freeu.b2      = 1.10f;
    // img_params.freeu.s1      = 0.50f;
    // img_params.freeu.s2      = 0.20f;

    // SAG (Self-Attention Guidance)
    img_params.sag.enabled = true;
    img_params.sag.scale   = 1.0f;

    // VAE tiling: 2560x1440 needs tiled decode to avoid OOM
    img_params.vae_tiling_params.enabled        = true;
    img_params.vae_tiling_params.tile_size_x    = 32;
    img_params.vae_tiling_params.tile_size_y    = 32;
    img_params.vae_tiling_params.target_overlap = 0.5f;

    // HiRes fix: low-res -> target via model upscaler (sharper than latent)
    img_params.hires.enabled             = true;
    img_params.hires.upscaler          = SD_HIRES_UPSCALER_MODEL;
    img_params.hires.model_path        = "/data/models/image/2x_ESRGAN.gguf";
    img_params.hires.upscale_tile_size = 128;
    img_params.hires.target_width      = target_W;
    img_params.hires.target_height     = target_H;
    img_params.hires.steps             = hires_steps;
    img_params.hires.denoising_strength = hires_strength;

    // 3. Generate
    sd_image_t* images = nullptr;
    int num_images = 0;
    if (!generate_image(ctx, &img_params, &images, &num_images)) {
        fprintf(stderr, "generate_image failed\n");
        free_sd_ctx(ctx);
        return 1;
    }

    if (!images || num_images == 0) {
        fprintf(stderr, "No images generated\n");
        free_sd_ctx(ctx);
        return 1;
    }

    // 4. Save PNG
    fprintf(stderr, "Saving %dx%d (channels=%d) to %s\n",
            images[0].width, images[0].height, images[0].channel, output);
    if (!save_png(output, images[0].data, images[0].width, images[0].height, images[0].channel)) {
        fprintf(stderr, "save_png failed\n");
        free_sd_images(images, num_images);
        free_sd_ctx(ctx);
        return 1;
    }

    // 5. Cleanup
    free_sd_images(images, num_images);
    free_sd_ctx(ctx);
    fprintf(stderr, "Done\n");
    return 0;
}
