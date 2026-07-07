// stable-diffusion-cli_v2.cpp — SDXL txt2img using sd_helper.cpp
#include "sd_helper.cpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>

int main(int argc, char** argv) {
    const char* model  = "/data/models/image/sd_xl_base_1.0.safetensors";
    const char* output = "/tmp/sd_cli_v2";
    int W=1024,H=1024,steps=5,seed=42; float cfg=7.0f;

    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-m")&&i+1<argc) model=argv[++i];
        else if(!strcmp(argv[i],"-o")&&i+1<argc) output=argv[++i];
        else if(!strcmp(argv[i],"--steps")&&i+1<argc) steps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--cfg")&&i+1<argc) cfg=atof(argv[++i]);
        else if(!strcmp(argv[i],"-s")&&i+1<argc) seed=atoi(argv[++i]);
    }

    sd_ctx_params_t cp; sd_ctx_params_init(&cp);
    cp.model_path=model; cp.n_threads=8; cp.wtype=SD_TYPE_F16;
    cp.rng_type=STD_DEFAULT_RNG;
    sd_ctx_t* ctx=new_sd_ctx(&cp);
    if(!ctx) return 1;

    sd_img_gen_params_t ip; sd_img_gen_params_init(&ip);
    ip.width=W; ip.height=H; ip.seed=seed;
    ip.sample_params.sample_steps=steps;
    ip.sample_params.guidance.txt_cfg=cfg;
    ip.sample_params.sample_method=EULER_SAMPLE_METHOD;
    ip.sample_params.scheduler=KARRAS_SCHEDULER;

    sd_image_t* imgs=generate_image(ctx,&ip);
    if(!imgs){free_sd_ctx(ctx);return 1;}

    char ppm[1024]; snprintf(ppm,sizeof(ppm),"%s.ppm",output);
    fprintf(stderr,"Saved %s (%dx%d)\n",ppm,imgs->width,imgs->height);
    free(imgs->data); free(imgs); free_sd_ctx(ctx);
    return 0;
}
