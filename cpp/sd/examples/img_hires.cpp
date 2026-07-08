// examples/img_hires.cpp - SDXL HiRes Fix + VAE tiling using sd::SDPipeline
#include "src/adapters/sdcpp_adapter.h"
#include "postproc.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

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
    std::fprintf(stderr, "Usage: %s [options] <prompt> <output> [width] [height]\n", argv0);
    std::fprintf(stderr, "Options:\n");
    std::fprintf(stderr, "  -m, --model <path>        Full checkpoint model path (safetensors)\n");
    std::fprintf(stderr, "  --diffusion-model <path>  Standalone diffusion model path (GGUF, e.g. Z-Image)\n");
    std::fprintf(stderr, "  --llm <path>              LLM text encoder path (required for --diffusion-model)\n");
    std::fprintf(stderr, "  --clip-l <path>           CLIP-L path\n");
    std::fprintf(stderr, "  --clip-g <path>           CLIP-G path\n");
    std::fprintf(stderr, "  --vae <path>              VAE path (optional, default uses model built-in VAE)\n");
    std::fprintf(stderr, "  -n, --negative <text>     Negative prompt\n");
    std::fprintf(stderr, "  -W, --width <int>         Target width (default: 1024)\n");
    std::fprintf(stderr, "  -H, --height <int>        Target height (default: 1024)\n");
    std::fprintf(stderr, "  --steps <int>             Sampling steps (default: 20)\n");
    std::fprintf(stderr, "  --method <name>           Sampling method: euler, euler_a, heun, dpm2, ... (default: euler_a)\n");
    std::fprintf(stderr, "  --scheduler <name>        Scheduler: discrete, karras, exponential, ... (default: discrete)\n");
    std::fprintf(stderr, "  --cfg <float>             CFG scale (default: 7.0)\n");
    std::fprintf(stderr, "  -s, --seed <int>          RNG seed (default: random)\n");
    std::fprintf(stderr, "  -t, --threads <int>       CPU threads (default: 8)\n");
    std::fprintf(stderr, "  --vae-tiling              Enable VAE tiling\n");
    std::fprintf(stderr, "  --vae-tile-size <int>     VAE tile size (default: 128)\n");
    std::fprintf(stderr, "  --vae-tile-overlap <float> VAE tile overlap (default: 0.5)\n");
    std::fprintf(stderr, "  --hires                   Enable HiRes Fix\n");
    std::fprintf(stderr, "  --hires-width <int>       HiRes target width (default: target width)\n");
    std::fprintf(stderr, "  --hires-height <int>      HiRes target height (default: target height)\n");
    std::fprintf(stderr, "  --hires-steps <int>       HiRes steps (default: 20)\n");
    std::fprintf(stderr, "  --hires-strength <float>  HiRes denoising strength (default: 0.35)\n");
    std::fprintf(stderr, "  --lora <path:weight>      LoRA, can be specified multiple times\n");
    std::fprintf(stderr, "  --freeu                   Enable FreeU\n");
    std::fprintf(stderr, "  --freeu-b1 <float>        FreeU backbone1 scale (default: 1.3)\n");
    std::fprintf(stderr, "  --freeu-b2 <float>        FreeU backbone2 scale (default: 1.4)\n");
    std::fprintf(stderr, "  --sag                     Enable Self-Attention Guidance\n");
    std::fprintf(stderr, "  --sag-scale <float>       SAG blend scale (default: 1.0)\n");
    std::fprintf(stderr, "  --diffusion-fa            Enable diffusion flash attention\n");
    std::fprintf(stderr, "  --no-quality-prefix       Do not prepend quality keywords to prompt\n");
    std::fprintf(stderr, "Post-processing:\n");
    std::fprintf(stderr, "  --clarity <float>         Clarity / local contrast, 0.0-1.0 (default: 0.0)\n");
    std::fprintf(stderr, "  --sharpen <float>         USM sharpen amount, 0.0-3.0 (default: 0.0)\n");
    std::fprintf(stderr, "  --sharpen-radius <int>    USM blur radius, 1-10 (default: 1)\n");
    std::fprintf(stderr, "  --smart-sharpen <float>   Edge-aware smart sharpen, 0.0-3.0 (default: 0.0)\n");
    std::fprintf(stderr, "  --smart-sharpen-radius <int>  Smart sharpen radius, 1-10 (default: 2)\n");
    std::fprintf(stderr, "  --edge-sharpen <float>    Edge-mask sharpen amount, 0.0-3.0 (default: 0.0)\n");
    std::fprintf(stderr, "  --edge-sharpen-radius <int>   Edge detection radius, 1-10 (default: 2)\n");
    std::fprintf(stderr, "  --edge-sharpen-threshold <float> Edge threshold, 0.0-1.0 (default: 0.3)\n");
}

static int64_t parse_seed(const char* s) {
    if (std::strcmp(s, "random") == 0) {
        return std::time(nullptr) % 2147483647;
    }
    return std::atoll(s);
}

static std::vector<std::pair<int, int>> compute_hires_resolution(int target_w, int target_h) {
    std::pair<int, int> low;
    if (target_w == 3840 && target_h == 2160) {
        low = {2560, 1440};
    } else if (target_w == 2560 && target_h == 1440) {
        low = {1920, 1080};
    } else if (target_w == 1920 && target_h == 1080) {
        low = {1536, 864};
    } else if (target_w == 1280 && target_h == 720) {
        low = {1024, 576};
    } else {
        int tw = target_w / 8;
        int th = target_h / 8;
        int lw = (tw * 4 / 5 + 7) / 8 * 8;
        int lh = (th * 4 / 5 + 7) / 8 * 8;
        low = {lw * 8, lh * 8};
    }

    // Ensure minimum 512 on both sides
    if (low.first < 512 || low.second < 512) {
        float ratio = static_cast<float>(target_w) / target_h;
        if (low.first < low.second) {
            low.first = 512;
            low.second = static_cast<int>(low.first / ratio / 8) * 8;
            if (low.second < 512) low.second = 512;
        } else {
            low.second = 512;
            low.first = static_cast<int>(low.second * ratio / 8) * 8;
            if (low.first < 512) low.first = 512;
        }
    }
    return {low, {target_w, target_h}};
}

int main(int argc, char** argv) {
    std::string model  = "";
    std::string diffusion_model = "/data/models/image/z_image_turbo-Q5_K_M.gguf";
    std::string llm    = "/data/models/image/Qwen3-4B-Instruct-2507-Q4_K_M.gguf";
    std::string clip_l = "/data/models/image/clip_l.safetensors";
    std::string clip_g = "/data/models/image/clip_g.safetensors";
    std::string vae    = "/data/models/image/ae.safetensors";
    std::string neg    = "blurry, low quality, worst quality, jpeg artifacts, noise, grain, soft focus, out of focus, hazy, unclear, bad anatomy, deformed";
    std::string output;
    std::string prompt;
    std::string method = "euler";
    std::string scheduler = "discrete";
    std::vector<sd::LoraConfig> loras;

    int W = 1024, H = 1024, steps = 20, threads = 8;
    int64_t seed = std::time(nullptr) % 2147483647;
    float cfg = 7.0f;

    bool vae_tiling = false;
    int vae_tile_size = 128;
    float vae_tile_overlap = 0.5f;

    bool hires = false;
    int hires_width = 0, hires_height = 0, hires_steps = 20;
    float hires_strength = 0.35f;

    bool freeu = false;
    float freeu_b1 = 1.3f;
    float freeu_b2 = 1.4f;

    bool sag = false;
    float sag_scale = 1.0f;

    bool diffusion_fa = false;
    bool quality_prefix = true;

    postproc::Params postproc;

    std::vector<char*> positional;
    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-m") == 0 || std::strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model = argv[++i];
        } else if (std::strcmp(argv[i], "--diffusion-model") == 0 && i + 1 < argc) {
            diffusion_model = argv[++i];
        } else if (std::strcmp(argv[i], "--llm") == 0 && i + 1 < argc) {
            llm = argv[++i];
        } else if (std::strcmp(argv[i], "--clip-l") == 0 && i + 1 < argc) {
            clip_l = argv[++i];
        } else if (std::strcmp(argv[i], "--clip-g") == 0 && i + 1 < argc) {
            clip_g = argv[++i];
        } else if (std::strcmp(argv[i], "--vae") == 0 && i + 1 < argc) {
            vae = argv[++i];
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
            seed = parse_seed(argv[++i]);
        } else if ((std::strcmp(argv[i], "-t") == 0 || std::strcmp(argv[i], "--threads") == 0) && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--vae-tiling") == 0) {
            vae_tiling = true;
        } else if (std::strcmp(argv[i], "--vae-tile-size") == 0 && i + 1 < argc) {
            vae_tile_size = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--vae-tile-overlap") == 0 && i + 1 < argc) {
            vae_tile_overlap = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--hires") == 0) {
            hires = true;
        } else if (std::strcmp(argv[i], "--hires-width") == 0 && i + 1 < argc) {
            hires_width = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--hires-height") == 0 && i + 1 < argc) {
            hires_height = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--hires-steps") == 0 && i + 1 < argc) {
            hires_steps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--hires-strength") == 0 && i + 1 < argc) {
            hires_strength = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--lora") == 0 && i + 1 < argc) {
            std::string lora_arg = argv[++i];
            size_t pos = lora_arg.find(':');
            sd::LoraConfig lora;
            if (pos == std::string::npos) {
                lora.path = lora_arg;
                lora.multiplier = 1.0f;
            } else {
                lora.path = lora_arg.substr(0, pos);
                lora.multiplier = std::atof(lora_arg.substr(pos + 1).c_str());
            }
            loras.push_back(lora);
        } else if (std::strcmp(argv[i], "--freeu") == 0) {
            freeu = true;
        } else if (std::strcmp(argv[i], "--freeu-b1") == 0 && i + 1 < argc) {
            freeu_b1 = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--freeu-b2") == 0 && i + 1 < argc) {
            freeu_b2 = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--sag") == 0) {
            sag = true;
        } else if (std::strcmp(argv[i], "--sag-scale") == 0 && i + 1 < argc) {
            sag_scale = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--diffusion-fa") == 0) {
            diffusion_fa = true;
        } else if (std::strcmp(argv[i], "--no-quality-prefix") == 0) {
            quality_prefix = false;
        } else if (std::strcmp(argv[i], "--clarity") == 0 && i + 1 < argc) {
            postproc.clarity = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--sharpen") == 0 && i + 1 < argc) {
            postproc.sharpen_amount = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--sharpen-radius") == 0 && i + 1 < argc) {
            postproc.sharpen_radius = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--smart-sharpen") == 0 && i + 1 < argc) {
            postproc.smart_sharpen_strength = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--smart-sharpen-radius") == 0 && i + 1 < argc) {
            postproc.smart_sharpen_radius = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--edge-sharpen") == 0 && i + 1 < argc) {
            postproc.edge_sharpen_amount = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--edge-sharpen-radius") == 0 && i + 1 < argc) {
            postproc.edge_sharpen_radius = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--edge-sharpen-threshold") == 0 && i + 1 < argc) {
            postproc.edge_sharpen_threshold = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (argv[i][0] != '-') {
            positional.push_back(argv[i]);
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (positional.size() >= 1) {
        prompt = positional[0];
    }
    if (positional.size() >= 2) {
        output = positional[1];
    }
    if (positional.size() >= 3) {
        W = std::atoi(positional[2]);
    }
    if (positional.size() >= 4) {
        H = std::atoi(positional[3]);
    }

    if (prompt.empty() || output.empty()) {
        std::fprintf(stderr, "Error: prompt and output are required\n");
        print_usage(argv[0]);
        return 1;
    }

    if (quality_prefix && prompt.find("masterpiece") == std::string::npos) {
        prompt = "masterpiece, best quality, ultra-detailed, sharp focus, photorealistic, highly detailed, " + prompt;
    }

    if (!hires) {
        // No HiRes: generate directly at target resolution
        hires_width = W;
        hires_height = H;
    } else {
        if (hires_width == 0) hires_width = W;
        if (hires_height == 0) hires_height = H;
    }

    auto resolutions = compute_hires_resolution(W, H);
    int low_w = resolutions[0].first;
    int low_h = resolutions[0].second;

    std::fprintf(stderr, "img_hires (v2 adapter)\n");
    std::fprintf(stderr, "  model:  %s\n", model.empty() ? diffusion_model.c_str() : model.c_str());
    std::fprintf(stderr, "  diffusion_model: %s\n", diffusion_model.c_str());
    std::fprintf(stderr, "  llm:    %s\n", llm.c_str());
    std::fprintf(stderr, "  prompt: %s\n", prompt.c_str());
    std::fprintf(stderr, "  low-res: %dx%d -> target: %dx%d\n", low_w, low_h, W, H);
    std::fprintf(stderr, "  steps: %d (HiRes: %d), cfg=%.1f, seed=%ld\n", steps, hires ? hires_steps : 0, cfg, seed);

    sd::ModelConfig cfg_model;
    cfg_model.model_path           = model;
    cfg_model.diffusion_model_path = diffusion_model;
    cfg_model.llm_path             = llm;
    cfg_model.clip_l_path          = clip_l;
    cfg_model.clip_g_path          = clip_g;
    cfg_model.vae_path             = vae;
    cfg_model.n_threads            = threads;
    cfg_model.diffusion_flash_attn = diffusion_fa;

    sd::SDPipeline pipeline;
    if (!pipeline.load(cfg_model)) {
        std::fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    std::fprintf(stderr, "Model loaded\n");

    sd::ImageGenerationParams gen_params;
    gen_params.prompt          = prompt;
    gen_params.negative_prompt = neg;
    gen_params.width           = low_w;
    gen_params.height          = low_h;
    gen_params.steps           = steps;
    gen_params.cfg_scale       = cfg;
    gen_params.seed            = seed;
    gen_params.batch_count     = 1;
    gen_params.sample_method   = method;
    gen_params.scheduler       = scheduler;
    gen_params.loras           = loras;
    gen_params.vae_tiling      = vae_tiling;
    gen_params.vae_tile_size_x = vae_tile_size;
    gen_params.vae_tile_size_y = vae_tile_size;
    gen_params.vae_tile_overlap = vae_tile_overlap;
    gen_params.hires_enabled   = hires;
    gen_params.hires_width     = hires_width;
    gen_params.hires_height    = hires_height;
    gen_params.hires_steps     = hires_steps;
    gen_params.hires_strength  = hires_strength;
    gen_params.freeu_enabled   = freeu;
    gen_params.freeu_b1        = freeu_b1;
    gen_params.freeu_b2        = freeu_b2;
    gen_params.sag_enabled     = sag;
    gen_params.sag_scale       = sag_scale;

    sd::Image image = pipeline.generate(gen_params);
    if (image.empty()) {
        std::fprintf(stderr, "Image generation failed\n");
        return 1;
    }

    bool has_postproc = (postproc.clarity > 0.0f ||
                         postproc.sharpen_amount > 0.0f ||
                         postproc.smart_sharpen_strength > 0.0f ||
                         postproc.edge_sharpen_amount > 0.0f);
    if (has_postproc) {
        std::fprintf(stderr, "Post-processing: clarity=%.2f, sharpen=%.2f(r=%d), "
                             "smart=%.2f(r=%d), edge=%.2f(r=%d,t=%.2f)\n",
                     postproc.clarity,
                     postproc.sharpen_amount, postproc.sharpen_radius,
                     postproc.smart_sharpen_strength, postproc.smart_sharpen_radius,
                     postproc.edge_sharpen_amount, postproc.edge_sharpen_radius, postproc.edge_sharpen_threshold);
        if (!postproc::apply(image.data.data(), image.width, image.height, image.channels, postproc)) {
            std::fprintf(stderr, "Post-processing failed\n");
            return 1;
        }
        std::fprintf(stderr, "Post-processing completed\n");
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
