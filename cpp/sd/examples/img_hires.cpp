// examples/img_hires.cpp - SDXL HiRes Fix + VAE tiling using sd::SDPipeline
#include "src/adapters/sdcpp_adapter.h"
#include "postproc.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

static double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static int save_png(const char* path, const uint8_t* data, int w, int h, int c) {
    return stbi_write_png(path, w, h, c, data, 0);
}

// ---- 生成参数元数据（A1111/ComfyUI 风格，写入 PNG tEXt 'parameters'）----
static std::string fmt_f(float x) {
    char b[32];
    std::snprintf(b, sizeof(b), "%g", x);
    return std::string(b);
}

static void meta_put(std::string& out, const std::string& k, const std::string& v) {
    out += k; out += ": "; out += v; out += "\n";
}

static uint32_t png_crc32(const unsigned char* d, size_t n) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = table[(c ^ d[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

static bool png_read_file(const std::string& path, std::vector<unsigned char>& buf) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0) { std::fclose(f); return false; }
    buf.resize(static_cast<size_t>(sz));
    size_t rd = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    return rd == buf.size();
}

// 在 IEND 前插入一个 tEXt 块（keyword\0value）
static bool png_add_text(const std::string& path, const std::string& key, const std::string& value) {
    std::vector<unsigned char> buf;
    if (!png_read_file(path, buf)) return false;
    static const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (buf.size() < 8 || std::memcmp(buf.data(), sig, 8) != 0) return false;
    size_t insert_pos = buf.size();
    size_t pos = 8;
    while (pos + 8 <= buf.size()) {
        uint32_t len = (static_cast<uint32_t>(buf[pos]) << 24) | (buf[pos + 1] << 16) |
                       (static_cast<uint32_t>(buf[pos + 2]) << 8) | buf[pos + 3];
        std::string type(reinterpret_cast<char*>(&buf[pos + 4]), 4);
        if (type == "IEND") { insert_pos = pos; break; }
        if (pos + 12 + len > buf.size()) break;
        pos += 12 + len;
    }
    std::vector<unsigned char> data;
    data.insert(data.end(), key.begin(), key.end());
    data.push_back(0);
    data.insert(data.end(), value.begin(), value.end());

    const char* t = "tEXt";
    std::vector<unsigned char> chunk;
    uint32_t dlen = static_cast<uint32_t>(data.size());
    chunk.push_back((dlen >> 24) & 0xFF);
    chunk.push_back((dlen >> 16) & 0xFF);
    chunk.push_back((dlen >> 8) & 0xFF);
    chunk.push_back(dlen & 0xFF);
    chunk.insert(chunk.end(), t, t + 4);
    chunk.insert(chunk.end(), data.begin(), data.end());
    std::vector<unsigned char> crcbuf;
    crcbuf.insert(crcbuf.end(), t, t + 4);
    crcbuf.insert(crcbuf.end(), data.begin(), data.end());
    uint32_t crc = png_crc32(crcbuf.data(), crcbuf.size());
    chunk.push_back((crc >> 24) & 0xFF);
    chunk.push_back((crc >> 16) & 0xFF);
    chunk.push_back((crc >> 8) & 0xFF);
    chunk.push_back(crc & 0xFF);

    std::vector<unsigned char> out;
    out.reserve(buf.size() + chunk.size());
    out.insert(out.end(), buf.begin(), buf.begin() + insert_pos);
    out.insert(out.end(), chunk.begin(), chunk.end());
    out.insert(out.end(), buf.begin() + insert_pos, buf.end());

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t wr = std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
    return wr == out.size();
}

// 读取 PNG 内首个 tEXt 的指定 key，并把 value 打印到 stdout
static int png_dump_text(const std::string& path, const std::string& key) {
    std::vector<unsigned char> buf;
    if (!png_read_file(path, buf)) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); return 1; }
    static const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (buf.size() < 8 || std::memcmp(buf.data(), sig, 8) != 0) {
        std::fprintf(stderr, "not a png: %s\n", path.c_str());
        return 1;
    }
    size_t pos = 8;
    while (pos + 8 <= buf.size()) {
        uint32_t len = (static_cast<uint32_t>(buf[pos]) << 24) | (buf[pos + 1] << 16) |
                       (static_cast<uint32_t>(buf[pos + 2]) << 8) | buf[pos + 3];
        std::string type(reinterpret_cast<char*>(&buf[pos + 4]), 4);
        if (pos + 12 + len > buf.size()) break;
        if (type == "tEXt") {
            const unsigned char* d = &buf[pos + 8];
            size_t klen = 0;
            while (klen < len && d[klen] != 0) klen++;
            std::string k(reinterpret_cast<const char*>(d), klen);
            if (k == key) {
                std::string v(reinterpret_cast<const char*>(d + klen + 1), len - klen - 1);
                std::fwrite(v.data(), 1, v.size(), stdout);
                return 0;
            }
        }
        if (type == "IEND") break;
        pos += 12 + len;
    }
    std::fprintf(stderr, "metadata key '%s' not found in %s\n", key.c_str(), path.c_str());
    return 1;
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
    std::fprintf(stderr, "  --llm-vision <path>       LLM vision projector (mmproj; required with --ref-image on Qwen)\n");
    std::fprintf(stderr, "  --ref-image <path>        Reference image (native ref conditioning: Qwen-Image / Z-Image-Omni)\n");
    std::fprintf(stderr, "  --ref-image-args <str>    Extra ref-image args, e.g. \"preset=qwen\" (default: model preset)\n");
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
    std::fprintf(stderr, "  --hires-steps <int>       HiRes steps (default: 45, aligned to backup.sh)\n");
    std::fprintf(stderr, "  --hires-strength <float>  HiRes denoising strength (default: 0.35)\n");
    std::fprintf(stderr, "  --hires-upscaler <name>   HiRes upscaler: latent-bicubic/bislerp/model (default: latent-bicubic)\n");
    std::fprintf(stderr, "  --hires-upscaler-model <path>  Upscaler model (for --hires-upscaler model)\n");
    std::fprintf(stderr, "  --upscale-model <path>    Post-upscale model (ESRGAN), e.g. 2x_ESRGAN.gguf\n");
    std::fprintf(stderr, "  --upscale-repeats <int>   Number of post-upscale passes (default: 0=off)\n");
    std::fprintf(stderr, "  --upscale-tile-size <int> Post-upscale tile size (default: 128)\n");
    std::fprintf(stderr, "  --lora <path:weight>      LoRA, can be specified multiple times\n");
    std::fprintf(stderr, "  --ipadapter               Enable native IP-Adapter (SD1.x / SDXL checkpoints)\n");
    std::fprintf(stderr, "  --ipadapter-model <path>  IP-Adapter weights (e.g. ip-adapter-plus_sdxl_vit-h.safetensors)\n");
    std::fprintf(stderr, "  --ipadapter-clip-vision <path>  CLIP Vision model (e.g. clip_vision_sd15.safetensors)\n");
    std::fprintf(stderr, "  --ipadapter-image <path>  Reference image for IP-Adapter\n");
    std::fprintf(stderr, "  --ipadapter-strength <f>  IP-Adapter conditioning strength (default: 1.0)\n");
    std::fprintf(stderr, "  --control-net <path>      ControlNet model (Z-Image Fun-ControlNet)\n");
    std::fprintf(stderr, "  --control-image <path>    ControlNet control image (canny/depth/pose...)\n");
    std::fprintf(stderr, "  --control-strength <f>    ControlNet strength (default: 1.0)\n");
    std::fprintf(stderr, "  --freeu                   Enable FreeU\n");
    std::fprintf(stderr, "  --freeu-b1 <float>        FreeU backbone1 scale (default: 1.3)\n");
    std::fprintf(stderr, "  --freeu-b2 <float>        FreeU backbone2 scale (default: 1.4)\n");
    std::fprintf(stderr, "  --sag                     Enable Self-Attention Guidance\n");
    std::fprintf(stderr, "  --sag-scale <float>       SAG blend scale (default: 1.0)\n");
    std::fprintf(stderr, "  --fresca                  Enable FreSca frequency guidance (model-agnostic)\n");
    std::fprintf(stderr, "  --fresca-low <float>      FreSca low-frequency scale (default: 1.0)\n");
    std::fprintf(stderr, "  --fresca-high <float>     FreSca high-frequency scale (default: 1.25)\n");
    std::fprintf(stderr, "  --fresca-cutoff <int>     FreSca frequency cutoff (default: 20)\n");
    std::fprintf(stderr, "  --diffusion-fa            Enable diffusion flash attention\n");
    std::fprintf(stderr, "  --cache-mode <name>       Sample-step cache: easycache (DiT) | disabled (default)\n");
    std::fprintf(stderr, "  --cache-threshold <float> EasyCache reuse threshold (default 0.2; higher = more skips)\n");
    std::fprintf(stderr, "  --cache-start <float>     Cache active from this progress (0-1, default 0.15)\n");
    std::fprintf(stderr, "  --cache-end <float>       Cache active until this progress (0-1, default 0.95)\n");
    std::fprintf(stderr, "  --offload-to-cpu          Keep weights in CPU RAM (params_backend \"*=cpu\")\n");
    std::fprintf(stderr, "  --backend <spec>          Backend spec (e.g. CUDA, CPU, \"te=cpu\")\n");
    std::fprintf(stderr, "  --params-backend <spec>   Params backend spec (e.g. \"*=cpu\")\n");
    std::fprintf(stderr, "  --no-quality-prefix       Do not prepend quality keywords to prompt\n");
    std::fprintf(stderr, "Post-processing (defaults aligned to backup.sh; pass 0 to disable a stage):\n");
    std::fprintf(stderr, "  --clarity <float>         Clarity / local contrast, 0.0-1.0 (default: 0.2)\n");
    std::fprintf(stderr, "  --sharpen <float>         USM sharpen amount, 0.0-3.0 (default: 0.3)\n");
    std::fprintf(stderr, "  --sharpen-radius <int>    USM blur radius, 1-10 (default: 1)\n");
    std::fprintf(stderr, "  --smart-sharpen <float>   Edge-aware smart sharpen, 0.0-3.0 (default: 0.5)\n");
    std::fprintf(stderr, "  --smart-sharpen-radius <int>  Smart sharpen radius, 1-10 (default: 2)\n");
    std::fprintf(stderr, "  --edge-sharpen <float>    Edge-mask sharpen amount, 0.0-3.0 (default: 1.5)\n");
    std::fprintf(stderr, "  --edge-sharpen-radius <int>   Edge detection radius, 1-10 (default: 2)\n");
    std::fprintf(stderr, "  --edge-sharpen-threshold <float> Edge threshold, 0.0-1.0 (default: 0.3)\n");
    std::fprintf(stderr, "  --dump-meta <png>         Print embedded generation metadata (tEXt 'parameters') and exit\n");
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
    bool clip_l_overridden = false;
    bool clip_g_overridden = false;
    bool vae_overridden    = false;
    // 默认负面词对齐 backup.sh（含反演 embedding，sd.cpp 按普通 token 处理，无副作用）
    std::string neg    = "blurry, low quality, worst quality, jpeg artifacts, noise, grain, soft focus, out of focus, hazy, unclear, bad anatomy, deformed, border artifacts, edge distortion, tiling artifacts, edge artifacts, frame distortion, warped edges, stretched proportions, asymmetrical face, off-center, cropped, out of frame, partial face, cut off, incomplete head, cropped head, watermark, text, logo, signature, cropped shoulders, embedding:EasyNegative, embedding:bad-hands-5";
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
    int low_w = 0, low_h = 0;
    int hires_width = 0, hires_height = 0, hires_steps = 45;
    float hires_strength = 0.35f;
    std::string hires_upscaler = "latent-bicubic";
    std::string hires_upscaler_model;

    std::string upscale_model;
    int upscale_repeats = 0;
    int upscale_tile_size = 128;

    bool freeu = false;
    float freeu_b1 = 1.3f;
    float freeu_b2 = 1.4f;

    bool sag = false;
    float sag_scale = 1.0f;

    bool fresca = false;
    float fresca_low = 1.0f;
    float fresca_high = 1.25f;
    int fresca_cutoff = 20;

    bool diffusion_fa = false;
    bool quality_prefix = true;
    bool offload_to_cpu = false;
    std::string backend;
    std::string params_backend;
    std::string cache_mode = "disabled";
    float cache_threshold = 0.2f;
    float cache_start = 0.15f;
    float cache_end = 0.95f;

    std::string llm_vision;
    std::vector<std::string> ref_image_paths;
    std::string ref_image_args;
    bool ipadapter = false;
    std::string ipadapter_model;
    std::string ipadapter_clip_vision;
    std::string ipadapter_image;
    float ipadapter_strength = 1.0f;

    std::string control_net;
    std::string control_image;
    float control_strength = 1.0f;

    // 后处理默认值对齐 backup.sh：清晰度 + 锐化 + 智能锐化 + 边缘锐化，提升清晰度/细节
    postproc::Params postproc;
    postproc.clarity                = 0.2f;
    postproc.sharpen_amount         = 0.3f;
    postproc.sharpen_radius         = 1;
    postproc.smart_sharpen_strength = 0.5f;
    postproc.smart_sharpen_radius   = 2;
    postproc.edge_sharpen_amount    = 1.5f;
    postproc.edge_sharpen_radius    = 2;
    postproc.edge_sharpen_threshold = 0.3f;

    std::vector<char*> positional;
    // --dump-meta <png>：读取内嵌参数并退出（不加载模型）
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump-meta") == 0 && i + 1 < argc) {
            return png_dump_text(expand_tilde(argv[i + 1]), "parameters");
        }
    }
    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-m") == 0 || std::strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model = argv[++i];
        } else if (std::strcmp(argv[i], "--diffusion-model") == 0 && i + 1 < argc) {
            diffusion_model = argv[++i];
        } else if (std::strcmp(argv[i], "--llm") == 0 && i + 1 < argc) {
            llm = argv[++i];
        } else if (std::strcmp(argv[i], "--llm-vision") == 0 && i + 1 < argc) {
            llm_vision = argv[++i];
        } else if (std::strcmp(argv[i], "--ref-image") == 0 && i + 1 < argc) {
            ref_image_paths.push_back(argv[++i]);
        } else if (std::strcmp(argv[i], "--ref-image-args") == 0 && i + 1 < argc) {
            ref_image_args = argv[++i];
        } else if (std::strcmp(argv[i], "--ipadapter") == 0) {
            ipadapter = true;
        } else if (std::strcmp(argv[i], "--ipadapter-model") == 0 && i + 1 < argc) {
            ipadapter_model = argv[++i];
        } else if (std::strcmp(argv[i], "--ipadapter-clip-vision") == 0 && i + 1 < argc) {
            ipadapter_clip_vision = argv[++i];
        } else if (std::strcmp(argv[i], "--ipadapter-image") == 0 && i + 1 < argc) {
            ipadapter_image = argv[++i];
        } else if (std::strcmp(argv[i], "--ipadapter-strength") == 0 && i + 1 < argc) {
            ipadapter_strength = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "--control-net") == 0 && i + 1 < argc) {
            control_net = argv[++i];
        } else if (std::strcmp(argv[i], "--control-image") == 0 && i + 1 < argc) {
            control_image = argv[++i];
        } else if (std::strcmp(argv[i], "--control-strength") == 0 && i + 1 < argc) {
            control_strength = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "--clip-l") == 0 && i + 1 < argc) {
            clip_l = argv[++i];
            clip_l_overridden = true;
        } else if (std::strcmp(argv[i], "--clip-g") == 0 && i + 1 < argc) {
            clip_g = argv[++i];
            clip_g_overridden = true;
        } else if (std::strcmp(argv[i], "--vae") == 0 && i + 1 < argc) {
            vae = argv[++i];
            vae_overridden = true;
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
        } else if (std::strcmp(argv[i], "--hires-upscaler") == 0 && i + 1 < argc) {
            hires_upscaler = argv[++i];
        } else if (std::strcmp(argv[i], "--hires-upscaler-model") == 0 && i + 1 < argc) {
            hires_upscaler_model = argv[++i];
        } else if (std::strcmp(argv[i], "--upscale-model") == 0 && i + 1 < argc) {
            upscale_model = argv[++i];
        } else if (std::strcmp(argv[i], "--upscale-repeats") == 0 && i + 1 < argc) {
            upscale_repeats = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--upscale-tile-size") == 0 && i + 1 < argc) {
            upscale_tile_size = std::atoi(argv[++i]);
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
        } else if (std::strcmp(argv[i], "--fresca") == 0) {
            fresca = true;
        } else if (std::strcmp(argv[i], "--fresca-low") == 0 && i + 1 < argc) {
            fresca_low = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--fresca-high") == 0 && i + 1 < argc) {
            fresca_high = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--fresca-cutoff") == 0 && i + 1 < argc) {
            fresca_cutoff = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--diffusion-fa") == 0) {
            diffusion_fa = true;
        } else if (std::strcmp(argv[i], "--cache-mode") == 0 && i + 1 < argc) {
            cache_mode = argv[++i];
        } else if (std::strcmp(argv[i], "--cache-threshold") == 0 && i + 1 < argc) {
            cache_threshold = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--cache-start") == 0 && i + 1 < argc) {
            cache_start = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--cache-end") == 0 && i + 1 < argc) {
            cache_end = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-quality-prefix") == 0) {
            quality_prefix = false;
        } else if (std::strcmp(argv[i], "--offload-to-cpu") == 0) {
            offload_to_cpu = true;
        } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend = argv[++i];
        } else if (std::strcmp(argv[i], "--params-backend") == 0 && i + 1 < argc) {
            params_backend = argv[++i];
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

    // Mutually exclusive model modes: SDXL checkpoint vs GGUF+LLM
    if (!model.empty()) {
        // SDXL checkpoint contains UNet/VAE/CLIP-L/CLIP-G; external diffusion/LLM are not used
        diffusion_model = "";
        llm = "";
        llm_vision = "";
        // Use external VAE/CLIP only when explicitly requested
        if (!clip_l_overridden) clip_l = "";
        if (!clip_g_overridden) clip_g = "";
        if (!vae_overridden)    vae = "";
    } else {
        // GGUF+LLM mode: no full checkpoint, VAE/diffusion/LLM required, external CLIP optional
        model = "";
    }

    if (ipadapter && (ipadapter_model.empty() || ipadapter_clip_vision.empty() || ipadapter_image.empty())) {
        std::fprintf(stderr, "Error: --ipadapter requires --ipadapter-model, --ipadapter-clip-vision and --ipadapter-image\n");
        return 1;
    }

    if (quality_prefix && prompt.find("masterpiece") == std::string::npos) {
        // 质量提示词原样对齐 backup.sh（画质 / 细节 / 人像取向）
        prompt = "masterpiece, best quality, ultra-detailed, sharp focus, 8k uhd, photorealistic, "
                 "highly detailed, crisp, clear, centered composition, professional portrait, "
                 "medium shot, realistic skin texture, soft lighting, " + prompt;
    }

    if (!hires) {
        // No HiRes: generate directly at target resolution
        hires_width = W;
        hires_height = H;
        low_w = W;
        low_h = H;
    } else {
        if (hires_width == 0) hires_width = W;
        if (hires_height == 0) hires_height = H;
        // If explicit hires target differs from W/H, treat W/H as the base resolution
        if (hires_width != W || hires_height != H) {
            low_w = W;
            low_h = H;
        } else {
            // W/H is the target; compute a low-res base
            auto resolutions = compute_hires_resolution(W, H);
            low_w = resolutions[0].first;
            low_h = resolutions[0].second;
        }
    }

    std::fprintf(stderr, "img_hires (v2 adapter)\n");
    std::fprintf(stderr, "  model:  %s\n", model.empty() ? diffusion_model.c_str() : model.c_str());
    std::fprintf(stderr, "  diffusion_model: %s\n", diffusion_model.c_str());
    std::fprintf(stderr, "  llm:    %s\n", llm.c_str());
    std::fprintf(stderr, "  clip_l: %s\n", clip_l.c_str());
    std::fprintf(stderr, "  clip_g: %s\n", clip_g.c_str());
    std::fprintf(stderr, "  vae:    %s\n", vae.c_str());
    if (ipadapter) {
        std::fprintf(stderr, "  ipadapter: %s (clip: %s, image: %s, strength: %.2f)\n",
                     ipadapter_model.c_str(), ipadapter_clip_vision.c_str(),
                     ipadapter_image.c_str(), ipadapter_strength);
    }
    std::fprintf(stderr, "  prompt: %s\n", prompt.c_str());
    std::fprintf(stderr, "  low-res: %dx%d -> target: %dx%d\n", low_w, low_h, hires_width, hires_height);
    std::fprintf(stderr, "  steps: %d (HiRes: %d), cfg=%.1f, seed=%ld\n", steps, hires ? hires_steps : 0, cfg, seed);
    for (const std::string& p : ref_image_paths) {
        std::fprintf(stderr, "  ref-image: %s (args: %s)\n", p.c_str(), ref_image_args.c_str());
    }

    sd::ModelConfig cfg_model;
    cfg_model.model_path           = model;
    cfg_model.diffusion_model_path = diffusion_model;
    cfg_model.llm_path             = llm;
    cfg_model.llm_vision_path      = llm_vision;
    cfg_model.clip_l_path          = clip_l;
    cfg_model.clip_g_path          = clip_g;
    cfg_model.vae_path             = vae;
    cfg_model.n_threads            = threads;
    cfg_model.diffusion_flash_attn = diffusion_fa;
    cfg_model.backend              = backend;
    // --offload-to-cpu 等价 sd.cpp 的 params_backend "*=cpu"（权重留 CPU RAM，按需上卡）
    if (offload_to_cpu) {
        cfg_model.params_backend = params_backend.empty() ? "*=cpu" : ("*=cpu," + params_backend);
    } else {
        cfg_model.params_backend = params_backend;
    }

    double t_load0 = now_sec();
    sd::SDPipeline pipeline;
    if (!pipeline.load(cfg_model)) {
        std::fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    std::fprintf(stderr, "Model loaded (%.2fs)\n", now_sec() - t_load0);

    pipeline.set_hires_upscaler(hires_upscaler, hires_upscaler_model);
    pipeline.set_fresca(fresca, fresca_low, fresca_high, fresca_cutoff);
    for (const std::string& p : ref_image_paths) {
        pipeline.set_ref_image(p, ref_image_args);
    }
    if (ipadapter) {
        pipeline.set_ipadapter(ipadapter_model, ipadapter_clip_vision, ipadapter_image, ipadapter_strength);
    }
    if (!control_net.empty()) {
        if (!pipeline.load_control_net(control_net)) {
            std::fprintf(stderr, "Failed to load control net: %s\n", control_net.c_str());
            return 1;
        }
        if (!control_image.empty()) {
            pipeline.set_control_image(control_image, control_strength);
        }
    }

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

    if (cache_mode == "easycache") {
        gen_params.cache_mode = 1; // SD_CACHE_EASYCACHE
    } else if (cache_mode == "cache-dit") {
        gen_params.cache_mode = 5; // SD_CACHE_CACHE_DIT
    } else if (cache_mode == "spectrum") {
        gen_params.cache_mode = 6; // SD_CACHE_SPECTRUM
    } else if (cache_mode != "disabled" && !cache_mode.empty()) {
        std::fprintf(stderr, "Unknown --cache-mode: %s (use easycache|cache-dit|spectrum|disabled)\n", cache_mode.c_str());
        return 1;
    }
    gen_params.cache_reuse_threshold = cache_threshold;
    gen_params.cache_start_percent   = cache_start;
    gen_params.cache_end_percent     = cache_end;
    if (gen_params.cache_mode != 0) {
        std::fprintf(stderr, "  cache:  %s threshold=%.3f range=[%.2f,%.2f]\n",
                     cache_mode.c_str(), cache_threshold, cache_start, cache_end);
    }

    double t_gen0 = now_sec();
    std::vector<sd::Image> images = pipeline.generate(gen_params);
    double t_gen1 = now_sec();
    std::fprintf(stderr, "  generate wall: %.2fs\n", t_gen1 - t_gen0);
    if (images.empty() || images[0].empty()) {
        std::fprintf(stderr, "Image generation failed\n");
        return 1;
    }
    sd::Image& image = images[0];

    bool has_postproc = (postproc.clarity > 0.0f ||
                         postproc.sharpen_amount > 0.0f ||
                         postproc.smart_sharpen_strength > 0.0f ||
                         postproc.edge_sharpen_amount > 0.0f);
    double t_pp0 = 0.0;
    if (has_postproc) {
        std::fprintf(stderr, "Post-processing: clarity=%.2f, sharpen=%.2f(r=%d), "
                             "smart=%.2f(r=%d), edge=%.2f(r=%d,t=%.2f)\n",
                     postproc.clarity,
                     postproc.sharpen_amount, postproc.sharpen_radius,
                     postproc.smart_sharpen_strength, postproc.smart_sharpen_radius,
                     postproc.edge_sharpen_amount, postproc.edge_sharpen_radius, postproc.edge_sharpen_threshold);
        t_pp0 = now_sec();
        if (!postproc::apply(image.data.data(), image.width, image.height, image.channels, postproc)) {
            std::fprintf(stderr, "Post-processing failed\n");
            return 1;
        }
        std::fprintf(stderr, "Post-processing completed (%.2fs)\n", now_sec() - t_pp0);
    }

    if (!upscale_model.empty() && upscale_repeats > 0) {
        const char* be  = backend.empty() ? nullptr : backend.c_str();
        const char* pbe = params_backend.empty() ? nullptr : params_backend.c_str();
        upscaler_ctx_t* upscaler = new_upscaler_ctx(upscale_model.c_str(),
                                                    false,
                                                    threads,
                                                    upscale_tile_size,
                                                    be,
                                                    pbe);
        if (upscaler == nullptr) {
            std::fprintf(stderr, "Failed to load upscaler: %s\n", upscale_model.c_str());
            return 1;
        }
        int factor = get_upscale_factor(upscaler);
        if (factor <= 0) {
            factor = 4;
        }
        for (int r = 0; r < upscale_repeats; ++r) {
            sd_image_t in;
            in.width   = static_cast<uint32_t>(image.width);
            in.height  = static_cast<uint32_t>(image.height);
            in.channel = static_cast<uint32_t>(image.channels);
            in.data    = image.data.data();
            sd_image_t* out = nullptr;
            int n           = 0;
            if (!upscale(upscaler, in, static_cast<uint32_t>(factor), &out, &n) ||
                n <= 0 || out == nullptr || out[0].data == nullptr) {
                free_sd_images(out, n);
                std::fprintf(stderr, "Upscale failed\n");
                free_upscaler_ctx(upscaler);
                return 1;
            }
            image.width    = static_cast<int>(out[0].width);
            image.height   = static_cast<int>(out[0].height);
            image.channels = static_cast<int>(out[0].channel);
            image.data.assign(out[0].data,
                              out[0].data + static_cast<size_t>(out[0].width) * out[0].height * out[0].channel);
            free_sd_images(out, n);
            std::fprintf(stderr, "Upscaled x%d -> %dx%d\n", factor, image.width, image.height);
        }
        free_upscaler_ctx(upscaler);
    }

    std::string final_output = expand_tilde(output);
    double t_save0 = now_sec();
    if (!save_png(final_output.c_str(), image.data.data(), image.width, image.height, image.channels)) {
        std::fprintf(stderr, "Failed to save %s\n", final_output.c_str());
        return 1;
    }
    // 写入生成参数（PNG tEXt 'parameters'），便于日后从图片恢复参数复刻
    {
        std::string meta;
        meta_put(meta, "application", "img_hires");
        meta_put(meta, "prompt", prompt);
        meta_put(meta, "negative", neg);
        meta_put(meta, "steps", std::to_string(steps));
        meta_put(meta, "hires_steps", std::to_string(hires_steps));
        meta_put(meta, "cfg", fmt_f(cfg));
        meta_put(meta, "seed", std::to_string(static_cast<long long>(seed)));
        meta_put(meta, "method", method);
        meta_put(meta, "scheduler", scheduler);
        meta_put(meta, "width", std::to_string(image.width));
        meta_put(meta, "height", std::to_string(image.height));
        meta_put(meta, "base_width", std::to_string(low_w));
        meta_put(meta, "base_height", std::to_string(low_h));
        meta_put(meta, "hires", hires ? "1" : "0");
        meta_put(meta, "hires_width", std::to_string(hires_width));
        meta_put(meta, "hires_height", std::to_string(hires_height));
        meta_put(meta, "hires_strength", fmt_f(hires_strength));
        meta_put(meta, "hires_upscaler", hires_upscaler);
        meta_put(meta, "vae_tiling", vae_tiling ? "1" : "0");
        meta_put(meta, "vae_tile_size", std::to_string(vae_tile_size));
        meta_put(meta, "vae_tile_overlap", fmt_f(vae_tile_overlap));
        if (!model.empty())           meta_put(meta, "model", model);
        if (!diffusion_model.empty()) meta_put(meta, "diffusion_model", diffusion_model);
        if (!llm.empty())             meta_put(meta, "llm", llm);
        if (!vae.empty())             meta_put(meta, "vae", vae);
        if (!clip_l.empty())          meta_put(meta, "clip_l", clip_l);
        if (!clip_g.empty())          meta_put(meta, "clip_g", clip_g);
        for (const sd::LoraConfig& l : loras) {
            meta_put(meta, "lora", l.path + ":" + fmt_f(l.multiplier));
        }
        meta_put(meta, "fresca", std::string(fresca ? "1" : "0") + "," + fmt_f(fresca_low) +
                                "," + fmt_f(fresca_high) + "," + std::to_string(fresca_cutoff));
        meta_put(meta, "cache", cache_mode + "," + fmt_f(cache_threshold) + "," +
                               fmt_f(cache_start) + "," + fmt_f(cache_end));
        meta_put(meta, "clarity", fmt_f(postproc.clarity));
        meta_put(meta, "sharpen", fmt_f(postproc.sharpen_amount) + "," + std::to_string(postproc.sharpen_radius));
        meta_put(meta, "smart_sharpen", fmt_f(postproc.smart_sharpen_strength) + "," + std::to_string(postproc.smart_sharpen_radius));
        meta_put(meta, "edge_sharpen", fmt_f(postproc.edge_sharpen_amount) + "," +
                                       std::to_string(postproc.edge_sharpen_radius) + "," +
                                       fmt_f(postproc.edge_sharpen_threshold));
        if (png_add_text(final_output, "parameters", meta)) {
            std::fprintf(stderr, "Metadata: embedded %zu bytes as PNG tEXt 'parameters'\n", meta.size());
        } else {
            std::fprintf(stderr, "Metadata: failed to embed\n");
        }
    }
    std::fprintf(stderr, "Saved %s (%dx%d, %d channels) in %.2fs\n",
                 final_output.c_str(), image.width, image.height, image.channels,
                 now_sec() - t_save0);
    std::fprintf(stderr, "TOTAL wall (load+gen+post+save): %.2fs\n",
                 (now_sec() - t_load0));
    return 0;
}
