#include <torch/torch.h>
#include <cstdio>
#include <cstring>
#include <cstdint>

// Declare external functions from clip_helper.so
extern "C" void* torch_std_vae_decode_from_dict(void* vae_dict, void* latent);
extern "C" void* torch_std_safetensors_load(const char* path);

int main() {
    // Load safetensors
    void* sd = torch_std_safetensors_load("/data/models/image/sd_xl_base_1.0.safetensors");
    if (!sd) { fprintf(stderr, "Failed to load\n"); return 1; }
    
    // Load correct latent
    auto latent = torch::from_blob(malloc(131072), {1, 4, 128, 128}, torch::kFloat16);
    FILE* f = fopen("/tmp/correct_latent_raw.bin", "rb");
    fread(latent.data_ptr(), 1, 131072, f);
    fclose(f);
    
    printf("Latent loaded: mean=%.4f\n", latent.abs().mean().item<float>());
    
    // Decode with C++ VAE
    auto result_ptr = torch_std_vae_decode_from_dict(sd, &latent);
    if (!result_ptr) { fprintf(stderr, "VAE decode failed\n"); return 1; }
    
    auto& result = *static_cast<torch::Tensor*>(result_ptr);
    printf("VAE result: shape=%d,%d,%d,%d mean=%.4f\n",
           (int)result.size(0), (int)result.size(1), (int)result.size(2), (int)result.size(3),
           result.mean().item<float>());
    
    // Convert to image and save
    auto img = result.to(torch::kCPU).to(torch::kFloat32);
    img = img.squeeze(0).permute({1,2,0});
    img = (img + 1.0) * 127.5;
    img = img.clamp(0, 255).to(torch::kUInt8);
    
    // Save as PPM
    f = fopen("/tmp/vae_cpp_output.ppm", "wb");
    fprintf(f, "P6\n%d %d\n255\n", img.size(1), img.size(0));
    fwrite(img.data_ptr<uint8_t>(), 1, img.numel(), f);
    fclose(f);
    printf("Saved to /tmp/vae_cpp_output.ppm\n");
    
    return 0;
}
