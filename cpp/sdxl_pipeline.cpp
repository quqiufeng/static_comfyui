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
#include <cmath>

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

// ComfyUI-style LatentUpscale approximation: bilinear resize of RGB(A) uint8 image.
// We cannot access the raw latent tensor through the stable-diffusion.cpp C API, so
// we decode the base image, upscale in pixel space, then feed it back as init_image
// for an img2img pass at the target resolution. This is equivalent to:
//   EmptyLatentImage -> KSampler(base) -> LatentUpscale -> KSampler(refine)
// except the upscale happens in pixel space before VAE re-encode.
static sd_image_t resize_image_bilinear(const sd_image_t& src, int dst_w, int dst_h) {
    sd_image_t dst;
    dst.width   = dst_w;
    dst.height  = dst_h;
    dst.channel = src.channel;
    dst.data    = (uint8_t*)malloc(dst_w * dst_h * src.channel);

    float sx = (src.width  > 1) ? (float)(src.width  - 1) / (dst_w - 1) : 0.f;
    float sy = (src.height > 1) ? (float)(src.height - 1) / (dst_h - 1) : 0.f;

    for (int y = 0; y < dst_h; y++) {
        float fy = y * sy;
        int y0 = (int)floorf(fy);
        int y1 = (int)ceilf(fy);
        if (y1 >= src.height) y1 = src.height - 1;
        float wy = fy - y0;
        for (int x = 0; x < dst_w; x++) {
            float fx = x * sx;
            int x0 = (int)floorf(fx);
            int x1 = (int)ceilf(fx);
            if (x1 >= src.width) x1 = src.width - 1;
            float wx = fx - x0;
            for (int c = 0; c < (int)src.channel; c++) {
                uint8_t v00 = src.data[(y0 * src.width + x0) * src.channel + c];
                uint8_t v10 = src.data[(y0 * src.width + x1) * src.channel + c];
                uint8_t v01 = src.data[(y1 * src.width + x0) * src.channel + c];
                uint8_t v11 = src.data[(y1 * src.width + x1) * src.channel + c];
                float v0 = v00 + (v10 - v00) * wx;
                float v1 = v01 + (v11 - v01) * wx;
                float v  = v0 + (v1 - v0) * wy;
                int iv = (int)roundf(v);
                if (iv < 0) iv = 0;
                if (iv > 255) iv = 255;
                dst.data[(y * dst_w + x) * src.channel + c] = (uint8_t)iv;
            }
        }
    }
    return dst;
}

int main(int argc, char** argv) {
    const char* model  = "/data/models/image/RealVisXL_V5.0_fp16.safetensors";
    const char* prompt = "solo,single woman,half body portrait of a young woman, "
                         "realistic skin texture, natural pores, candid photo, film grain, RAW photo, "
                         "soft natural lighting, elegant pose, studio lighting, "
                         "natural eyes, clean white background, medium close up";
    const char* neg    = "blurry, low quality, worst quality, jpeg artifacts, "
                         "soft focus, out of focus, hazy, unclear, bad anatomy, deformed, "
                         "border artifacts, edge distortion, tiling artifacts, edge artifacts, "
                         "frame distortion, warped edges, stretched proportions, asymmetrical face, "
                         "off-center, cropped, out of frame, partial face, cut off, incomplete head, "
                         "cropped head, watermark, text, logo, signature, cropped shoulders, "
                         "embedding:EasyNegative, embedding:bad-hands-5";
    const char* output = "/home/quqiufeng/sdxl_pipeline_cpp.png";

    // Base pass: 1920x1080, HiRes target: 2560x1440 (same as sdxl_pipeline.py)
    int W = 1920, H = 1080;
    int target_W = 2560, target_H = 1440;
    int steps = 30;
    int hires_steps = 90;
    float cfg = 4.0f;
    float hires_strength = 0.3f;
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

    // 1. Create context. Full SDXL checkpoint already contains CLIP-L, CLIP-G, UNet and VAE.
    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path  = model;
    ctx_params.clip_l_path = nullptr;
    ctx_params.clip_g_path = nullptr;
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

    // VAE tiling: 2560x1440 needs tiled decode to avoid OOM
    img_params.vae_tiling_params.enabled        = true;
    img_params.vae_tiling_params.tile_size_x    = 32;
    img_params.vae_tiling_params.tile_size_y    = 32;
    img_params.vae_tiling_params.target_overlap = 0.5f;

    // HiRes fix: low-res -> target via latent upscaling (currently disabled)
    // img_params.hires.enabled             = true;
    // img_params.hires.upscaler            = SD_HIRES_UPSCALER_LATENT;
    // img_params.hires.target_width        = target_W;
    // img_params.hires.target_height       = target_H;
    // img_params.hires.steps               = hires_steps;
    // img_params.hires.denoising_strength  = hires_strength;

    // FreeU disabled, SAG enabled
    // img_params.freeu.enabled = true;
    // img_params.freeu.b1      = 1.4f;
    // img_params.freeu.b2      = 1.5f;
    // img_params.freeu.s1      = 0.9f;
    // img_params.freeu.s2      = 0.2f;

    img_params.sag.enabled = true;
    img_params.sag.scale   = 0.5f;

    // 3. Base pass: generate at W x H (ComfyUI EmptyLatentImage -> KSampler)
    sd_image_t* base_images = nullptr;
    int base_num = 0;
    if (!generate_image(ctx, &img_params, &base_images, &base_num)) {
        fprintf(stderr, "generate_image (base) failed\n");
        free_sd_ctx(ctx);
        return 1;
    }
    if (!base_images || base_num == 0) {
        fprintf(stderr, "No base images generated\n");
        free_sd_ctx(ctx);
        return 1;
    }
    sd_image_t base_image = base_images[0];

    // 4. ComfyUI-style HiRes: pixel-space upscale -> img2img at target resolution.
    //    This mimics LatentUpscale + KSampler: the upscaled image is VAE-encoded
    //    by the second generate_image call, noise is injected, and the UNet refines
    //    it for hires_steps at the target resolution.
    sd_image_t* final_images = nullptr;
    int final_num = 0;
    int actual_target_w = target_W;
    int actual_target_h = target_H;
    if (target_W != base_image.width || target_H != base_image.height) {
        fprintf(stderr, "HiRes upscale: %dx%d -> %dx%d (bilinear pixel-space)\n",
                base_image.width, base_image.height, target_W, target_H);

        sd_image_t upscaled = resize_image_bilinear(base_image, target_W, target_H);
        free_sd_images(base_images, base_num);
        base_images = nullptr;

        sd_img_gen_params_t hires_params;
        sd_img_gen_params_init(&hires_params);
        hires_params.prompt          = prompt;
        hires_params.negative_prompt = neg;
        hires_params.width           = target_W;
        hires_params.height          = target_H;
        hires_params.seed            = seed + 1;
        hires_params.batch_count     = 1;
        hires_params.strength        = hires_strength;
        hires_params.init_image      = upscaled;

        hires_params.sample_params.guidance.txt_cfg = cfg;
        hires_params.sample_params.sample_method    = DPMPP2M_SDE_SAMPLE_METHOD;
        hires_params.sample_params.scheduler        = KARRAS_SCHEDULER;
        hires_params.sample_params.sample_steps     = hires_steps;
        hires_params.sample_params.eta              = 0.0f;

        // hires_params.freeu.enabled = true;
        // hires_params.freeu.b1      = 1.4f;
        // hires_params.freeu.b2      = 1.5f;
        // hires_params.freeu.s1      = 0.9f;
        // hires_params.freeu.s2      = 0.2f;

        hires_params.sag.enabled = true;
        hires_params.sag.scale   = 0.5f;

        hires_params.vae_tiling_params.enabled        = true;
        hires_params.vae_tiling_params.tile_size_x    = 32;
        hires_params.vae_tiling_params.tile_size_y    = 32;
        hires_params.vae_tiling_params.target_overlap = 0.5f;

        if (!generate_image(ctx, &hires_params, &final_images, &final_num)) {
            fprintf(stderr, "generate_image (hires) failed\n");
            free(upscaled.data);
            free_sd_ctx(ctx);
            return 1;
        }
        free(upscaled.data);
        if (!final_images || final_num == 0) {
            fprintf(stderr, "No hires images generated\n");
            free_sd_ctx(ctx);
            return 1;
        }
    } else {
        final_images = base_images;
        final_num    = base_num;
    }

    // 5. Save PNG
    fprintf(stderr, "Saving %dx%d (channels=%d) to %s\n",
            final_images[0].width, final_images[0].height, final_images[0].channel, output);
    if (!save_png(output, final_images[0].data, final_images[0].width, final_images[0].height, final_images[0].channel)) {
        fprintf(stderr, "save_png failed\n");
        free_sd_images(final_images, final_num);
        free_sd_ctx(ctx);
        return 1;
    }

    // 6. Cleanup
    free_sd_images(final_images, final_num);
    free_sd_ctx(ctx);
    fprintf(stderr, "Done\n");
    return 0;
}
