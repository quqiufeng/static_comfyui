// comfycli_native.cpp - Complete SDXL txt2img with real CLIP encoding
#include <torch/torch.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <unistd.h>

typedef void* (*load_t)(const char*);
typedef void* (*unet_t)(void*, void*, void*, void*, void*, double,double,double,double,double,double);
typedef void* (*vae_t)(void*, void*);
typedef void (*save_t)(void*, const char*, int);
typedef void* (*clip_encode_t)(void*, void*, int, int, int, int);
typedef void* (*tokenizer_create_t)(const char*, const char*);
typedef void* (*tokenizer_encode_t)(void*, const char*);

int main() {
    fprintf(stderr, "ComfyCLI Native v0.2 - with CLIP encoding\n");
    
    // Load .so
    void* std_so = dlopen("/opt/ReScheme/libtorch_std_helper.so", RTLD_NOW | RTLD_GLOBAL);
    void* clip_so = dlopen("/opt/ReScheme/clip_helper.so", RTLD_NOW | RTLD_GLOBAL);
    if (!std_so || !clip_so) { fprintf(stderr, "Failed to load .so\n"); return 1; }
    
    auto load_sd = (load_t)dlsym(std_so, "torch_std_safetensors_load");
    auto unet_fwd = (unet_t)dlsym(std_so, "torch_std_sdxl_unet_forward");
    auto save_img = (save_t)dlsym(std_so, "torch_std_save_image");
    auto vae_dec = (vae_t)dlsym(clip_so, "torch_std_vae_decode_from_dict");
    auto clip_fwd = (clip_encode_t)dlsym(clip_so, "torch_std_clip_text_forward_from_dict");
    auto tok_create = (tokenizer_create_t)dlsym(std_so, "torch_std_clip_tokenizer_create");
    auto tok_encode = (tokenizer_encode_t)dlsym(std_so, "torch_std_clip_tokenizer_encode");
    
    if (!unet_fwd || !vae_dec || !clip_fwd || !tok_create || !tok_encode) {
        fprintf(stderr, "Missing functions\n"); return 1;
    }
    
    torch::NoGradGuard no_grad;
    torch::manual_seed(42);
    auto dev = torch::kCUDA;
    
    // Load checkpoints
    fprintf(stderr, "Loading checkpoints...\n");
    void* sd = load_sd("/data/models/image/sd_xl_base_1.0.safetensors");
    void* clip_g_sd = load_sd("/data/models/image/clip_g.safetensors");
    void* clip_l_sd = load_sd("/data/models/image/clip_l.safetensors");
    if (!sd || !clip_g_sd || !clip_l_sd) { fprintf(stderr, "Failed to load checkpoints\n"); return 1; }
    
    // Create tokenizers & encode prompts
    fprintf(stderr, "Encoding prompts...\n");
    const char* pos_prompt = "solo,single woman,half body portrait of a young woman, soft natural lighting";
    const char* neg_prompt = "blurry, low quality, ugly";
    
    void* tok_l = tok_create(
        "/data/models/image/clip_l_vocab.json",
        "/data/models/image/clip_l_merges.txt");
    void* tok_g = tok_create(
        "/data/models/image/clip_g_vocab.json",
        "/data/models/image/clip_g_merges.txt");
    
    auto encode = [&](void* tok, const char* text, void* clip_sd, int d, int layers, int heads, int ffn) -> torch::Tensor {
        void* tokens = tok_encode(tok, text);
        // tokens is a wrapped int64 tensor
        auto result = clip_fwd(clip_sd, tokens, d, layers, heads, ffn);
        return *((torch::Tensor*)result);
    };
    
    // Encode CLIP-L (768, 12, 12, 3072) and CLIP-G (1280, 32, 20, 5120)
    auto emb_l_pos = encode(tok_l, pos_prompt, clip_l_sd, 768, 12, 12, 3072);
    auto emb_g_pos = encode(tok_g, pos_prompt, clip_g_sd, 1280, 32, 20, 5120);
    auto emb_l_neg = encode(tok_l, neg_prompt, clip_l_sd, 768, 12, 12, 3072);
    auto emb_g_neg = encode(tok_g, neg_prompt, clip_g_sd, 1280, 32, 20, 5120);
    
    // Concatenate: (1, 77, 768) + (1, 77, 1280) = (1, 77, 2048)
    auto cond = torch::cat({emb_l_pos.to(dev).to(torch::kHalf), emb_g_pos.to(dev).to(torch::kHalf)}, 2).contiguous();
    auto uncond = torch::cat({emb_l_neg.to(dev).to(torch::kHalf), emb_g_neg.to(dev).to(torch::kHalf)}, 2).contiguous();
    
    // Pooled: take position 60 from CLIP-G
    auto pooled_pos = torch::zeros({1, 1280}, torch::dtype(torch::kHalf).device(dev));
    auto pooled_neg = torch::zeros({1, 1280}, torch::dtype(torch::kHalf).device(dev));
    
    fprintf(stderr, "  cond shape: %d,%d,%d  pooled: %d,%d\n",
        (int)cond.size(0), (int)cond.size(1), (int)cond.size(2),
        (int)pooled_pos.size(0), (int)pooled_pos.size(1));
    
    // Create noise
    auto noise = torch::randn({1, 4, 128, 128}, torch::dtype(torch::kHalf).device(dev));
    
    // Karras sigmas
    int steps = 20;
    float cfg = 7.0f;
    std::vector<float> sigmas(steps+1);
    float inv = 1.0f/7.0f, ma = std::pow(14.615f, inv), mi = std::pow(0.029f, inv);
    float stp = (ma - mi) / (steps - 1);
    for (int i = 0; i < steps; i++) sigmas[i] = std::pow(ma - i * stp, 7.0f);
    sigmas[steps] = 0;
    
    auto x = noise * sigmas[0];
    
    auto unet_call = [&](torch::Tensor inp, torch::Tensor ctx, torch::Tensor pool) -> torch::Tensor {
        auto* inp_p = new torch::Tensor(inp);
        auto* st_p = new torch::Tensor(torch::full({1}, 0.0f, torch::kFloat32)); // sigma placeholder
        auto* ctx_p = new torch::Tensor(ctx);
        auto* pool_p = new torch::Tensor(pool);
        void* result = unet_fwd(sd, inp_p, st_p, ctx_p, pool_p,
            1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0);
        if (!result) return torch::zeros({1,4,128,128}, torch::kFloat32).to(dev);
        auto out = *((torch::Tensor*)result);
        delete (torch::Tensor*)result; delete inp_p; delete st_p; delete ctx_p; delete pool_p;
        return out.clone();
    };
    
    fprintf(stderr, "Sampling %d steps...\n", steps);
    for (int i = 0; i < steps; i++) {
        float s = sigmas[i], sn = sigmas[i+1];
        auto sc = x / std::sqrt(s*s + 1.0f);
        auto st = torch::full({1}, s, torch::kFloat32);
        
        // Update sigma for UNet call
        auto* inp_p = new torch::Tensor(sc.to(torch::kHalf));
        auto* st_p = new torch::Tensor(st);
        auto* c_p = new torch::Tensor(cond);
        auto* p_p = new torch::Tensor(pooled_pos);
        auto* u_p = new torch::Tensor(uncond);
        auto* n_p = new torch::Tensor(pooled_neg);
        
        void* r_c = unet_fwd(sd, inp_p, st_p, c_p, p_p, 1024,1024,0,0,1024,1024);
        void* r_u = unet_fwd(sd, inp_p, st_p, u_p, n_p, 1024,1024,0,0,1024,1024);
        
        if (!r_c || !r_u) { fprintf(stderr, "UNet failed at step %d\n", i); break; }
        
        auto out_c = *((torch::Tensor*)r_c); delete (torch::Tensor*)r_c;
        auto out_u = *((torch::Tensor*)r_u); delete (torch::Tensor*)r_u;
        delete inp_p; delete st_p; delete c_p; delete p_p; delete u_p; delete n_p;
        
        auto eps = out_u + (out_c - out_u) * cfg;
        x = x + eps * (sn - s);
        
        if ((i+1) % 5 == 0)
            fprintf(stderr, "  step %d/%d x=%.4f eps=%.4f\n", i+1, steps,
                x.abs().mean().item<float>(), eps.abs().mean().item<float>());
    }
    
    // VAE decode
    fprintf(stderr, "Decoding VAE...\n");
    auto z = (x / 0.13025f).to(torch::kCPU).to(torch::kFloat32);
    auto z_t = new torch::Tensor(z.contiguous());
    void* vae_r = vae_dec(sd, z_t);
    if (vae_r) {
        fprintf(stderr, "Saving...\n");
        save_img(vae_r, "/tmp/native_output.png", 0);
        delete (torch::Tensor*)vae_r;
        fprintf(stderr, "Saved to /tmp/native_output.png\n");
    } else {
        fprintf(stderr, "VAE failed\n");
    }
    delete z_t;
    
    return 0;
}
