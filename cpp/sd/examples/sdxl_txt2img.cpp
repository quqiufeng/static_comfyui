// examples/sdxl_txt2img.cpp - SDXL txt2img using sd::SDPipeline (v2 adapter)
// Build with the cpp/sd CMakeLists.txt.
#include "src/adapters/sdcpp_adapter.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static int save_png(const char* path, const uint8_t* data, int w, int h, int c) {
    return stbi_write_png(path, w, h, c, data, 0);
}

static std::string expand_tilde(const std::string& path) {
    if (!path.empty() && path[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home) {
            return std::string(home) + path.substr(1);
        }
    }
    return path;
}

static void print_usage(const char* argv0) {
    std::fprintf(stderr, "Usage: %s [options]\n", argv0);
    std::fprintf(stderr, "Options:\n");
    std::fprintf(stderr, "  -m, --model <path>        Diffusion model path (default: sd_xl_base_1.0.safetensors)\n");
    std::fprintf(stderr, "  --clip-l <path>           CLIP-L path\n");
    std::fprintf(stderr, "  --clip-g <path>           CLIP-G path\n");
    std::fprintf(stderr, "  --vae <path>              VAE path (optional)\n");
    std::fprintf(stderr, "  -p, --prompt <text>       Prompt\n");
    std::fprintf(stderr, "  -n, --negative <text>     Negative prompt\n");
    std::fprintf(stderr, "  -o, --output <path>       Output PNG path (default: ~/sdxl_txt2img.png)\n");
    std::fprintf(stderr, "  -W, --width <int>         Image width (default: 1024)\n");
    std::fprintf(stderr, "  -H, --height <int>        Image height (default: 1024)\n");
    std::fprintf(stderr, "  --steps <int>             Sampling steps (default: 20)\n");
    std::fprintf(stderr, "  --method <name>           Sampling method: euler, euler_a, heun, dpm2, dpmpp2s_a, dpmpp2m, lcm, ... (default: euler_a)\n");
    std::fprintf(stderr, "  --scheduler <name>        Scheduler: discrete, karras, exponential, ... (default: karras)\n");
    std::fprintf(stderr, "  --cfg <float>             CFG scale (default: 7.0)\n");
    std::fprintf(stderr, "  -s, --seed <int>          RNG seed (default: 42)\n");
    std::fprintf(stderr, "  -t, --threads <int>       CPU threads (default: 8)\n");
}

int main(int argc, char** argv) {
    std::string model  = "/data/models/image/sd_xl_base_1.0.safetensors";
    std::string clip_l = "/data/models/image/clip_l.safetensors";
    std::string clip_g = "/data/models/image/clip_g.safetensors";
    std::string vae;
    std::string prompt = "solo, single woman, half body portrait of a young woman, "
                         "soft natural lighting, elegant pose, studio lighting, "
                         "sharp eyes, clean white background, medium close up";
    std::string neg    = "blurry, low quality, ugly";
    std::string output = expand_tilde("~/sdxl_txt2img.png");
    std::string method = "euler_a";
    std::string scheduler = "discrete";
    int W = 1024, H = 1024, steps = 20, seed = 42, threads = 8;
    float cfg = 7.0f;

    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-m") == 0 || std::strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model = argv[++i];
        } else if (std::strcmp(argv[i], "--clip-l") == 0 && i + 1 < argc) {
            clip_l = argv[++i];
        } else if (std::strcmp(argv[i], "--clip-g") == 0 && i + 1 < argc) {
            clip_g = argv[++i];
        } else if (std::strcmp(argv[i], "--vae") == 0 && i + 1 < argc) {
            vae = argv[++i];
        } else if ((std::strcmp(argv[i], "-p") == 0 || std::strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--negative") == 0) && i + 1 < argc) {
            neg = argv[++i];
        } else if ((std::strcmp(argv[i], "-o") == 0 || std::strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
            output = argv[++i];
        } else if ((std::strcmp(argv[i], "-W") == 0 || std::strcmp(argv[i], "--width") == 0) && i + 1 < argc) {
            W = std::atoi(argv[++i]);
        } else if ((std::strcmp(argv[i], "-H") == 0 || std::strcmp(argv[i], "--height") == 0) && i + 1 < argc) {
            H = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--method") == 0 && i + 1 < argc) {
            method = argv[++i];
        } else if (std::strcmp(argv[i], "--scheduler") == 0 && i + 1 < argc) {
            scheduler = argv[++i];
        } else if (std::strcmp(argv[i], "--cfg") == 0 && i + 1 < argc) {
            cfg = std::atof(argv[++i]);
        } else if ((std::strcmp(argv[i], "-s") == 0 || std::strcmp(argv[i], "--seed") == 0) && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
        } else if ((std::strcmp(argv[i], "-t") == 0 || std::strcmp(argv[i], "--threads") == 0) && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    std::fprintf(stderr, "sdxl_txt2img (v2 adapter)\n");
    std::fprintf(stderr, "  model:  %s\n", model.c_str());
    std::fprintf(stderr, "  clip_l: %s\n", clip_l.c_str());
    std::fprintf(stderr, "  clip_g: %s\n", clip_g.c_str());
    std::fprintf(stderr, "  prompt: %s\n", prompt.c_str());
    std::fprintf(stderr, "  size:   %dx%d steps=%d cfg=%.1f seed=%d\n", W, H, steps, cfg, seed);

    sd::ModelConfig cfg_model;
    cfg_model.model_path  = model;
    cfg_model.clip_l_path = clip_l;
    cfg_model.clip_g_path = clip_g;
    cfg_model.vae_path    = vae;
    cfg_model.n_threads   = threads;

    sd::SDPipeline pipeline;
    if (!pipeline.load(cfg_model)) {
        std::fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    std::fprintf(stderr, "Model loaded\n");

    sd::ImageGenerationParams gen_params;
    gen_params.prompt          = prompt;
    gen_params.negative_prompt = neg;
    gen_params.width           = W;
    gen_params.height          = H;
    gen_params.steps           = steps;
    gen_params.cfg_scale       = cfg;
    gen_params.seed            = seed;
    gen_params.batch_count     = 1;
    gen_params.sample_method   = method;
    gen_params.scheduler       = scheduler;

    sd::Image image = pipeline.generate(gen_params);
    if (image.empty()) {
        std::fprintf(stderr, "Image generation failed\n");
        return 1;
    }

    std::string final_output = expand_tilde(output);
    if (!save_png(final_output.c_str(), image.data.data(), image.width, image.height, image.channels)) {
        std::fprintf(stderr, "Failed to save %s\n", final_output.c_str());
        return 1;
    }
    std::fprintf(stderr, "Saved %s (%dx%d, %d channels)\n",
                 final_output.c_str(), image.width, image.height, image.channels);
    return 0;
}
