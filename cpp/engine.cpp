// engine.cpp - SDXL inference on GGML only
#include "engine.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include "/opt/stable-diffusion.cpp/thirdparty/json.hpp"
using json=nlohmann::json;

// Safetensors loader
struct ST{
    uint8_t*map=nullptr;size_t fsize=0;uint64_t hs=0;
    uint8_t*data(){return map+8+hs;}
    std::vector<std::string>names;std::vector<std::vector<int64_t>>shapes;std::vector<size_t>offs;
    bool load(const char*);~ST(){if(map)munmap(map,fsize);}
};
bool ST::load(const char*p){
    int fd=::open(p,O_RDONLY);if(fd<0)return false;
    fsize=(size_t)lseek(fd,0,SEEK_END);
    map=(uint8_t*)mmap(NULL,fsize,PROT_READ,MAP_PRIVATE,fd,0);
    ::close(fd);if(map==MAP_FAILED){map=nullptr;return false;}
    memcpy(&hs,map,8);
    std::string jstr((const char*)(map+8),hs);
    json j;try{j=json::parse(jstr);}catch(...){return false;}
    for(auto&[key,val]:j.items()){
        if(key=="__metadata__")continue;
        if(!val.contains("data_offsets")||!val.contains("shape"))continue;
        size_t stt=val["data_offsets"][0].get<size_t>();
        auto&sh=val["shape"];if(!sh.is_array())continue;
        std::vector<int64_t>shape;for(auto&d:sh)shape.push_back(d.get<int64_t>());
        names.push_back(key);shapes.push_back(shape);offs.push_back(stt);
    }
    return true;
}
static const ggml_fp16_t*f16p(ST&s,int i){return(const ggml_fp16_t*)(s.data()+s.offs[i]);}

// Denoiser
struct Den{float sigmas[1000],ls[1000];
    void init(const float*a){for(int i=0;i<1000;i++){float v=a[i];sigmas[i]=sqrtf((1-v)/std::max(v,1e-8f));ls[i]=logf(sigmas[i]);}}
    void sc(float s,float&cs,float&co,float&ci){cs=1.0f;co=-s;ci=1.0f/sqrtf(s*s+1);}
    float s2t(float s){float l=logf(s);int lo=0;for(int i=0;i<1000;i++)if(l-ls[i]>=0)lo++;
        lo=std::min(std::max(lo-1,0),998);int hi=lo+1;float w=std::max(0.f,std::min(1.f,(ls[lo]-l)/(ls[lo]-ls[hi])));return(1-w)*lo+w*hi;}
};

// Weight manager
struct WT{std::unordered_map<std::string,struct ggml_tensor*>t;struct ggml_context*ctx=nullptr;
    ggml_backend_buffer_t buf=nullptr;
    ~WT(){if(buf){ggml_backend_buffer_free(buf);buf=nullptr;}if(ctx)ggml_free(ctx);}
    struct ggml_tensor*get(const std::string&n){auto it=t.find(n);if(it!=t.end())return it->second;
        static const char*pf[]={"model.diffusion_model.","model.","first_stage_model.","conditioner.embedders.0.","conditioner.embedders.1.",""};
        for(int p=0;pf[p][0];p++){it=t.find(pf[p]+n);if(it!=t.end())return it->second;}return nullptr;}
};

// GGML helpers (no bias for F16 compat)
static struct ggml_tensor*lin(struct ggml_context*c,struct ggml_tensor*x,const std::string&n,WT&w){
    auto*wt=w.get(n+".weight");if(!wt)return x;return ggml_mul_mat(c,wt,x);}
static struct ggml_tensor*conv(struct ggml_context*c,struct ggml_tensor*x,const std::string&n,WT&w,int s,int p){
    auto*wt=w.get(n+".weight");if(!wt)return x;return ggml_conv_2d(c,wt,x,s,s,p,p,1,1);}
static struct ggml_tensor*gn_silu(struct ggml_context*c,struct ggml_tensor*x,int grp,const std::string&n,WT&w){
    auto*g=ggml_group_norm(c,x,grp,1e-6f);return ggml_silu(c,g);}

// Context
struct sdxl_ctx{ST st;WT w;Den den;int n_threads=4;bool use_gpu=false;ggml_backend_t compute_backend=nullptr;bool ok=false;
    ~sdxl_ctx(){if(compute_backend)ggml_backend_free(compute_backend);}};

// API init functions
void sd_ctx_params_init(sd_ctx_params_t*p){*p={};p->n_threads=8;p->wtype=SD_TYPE_F16;p->rng_type=CUDA_RNG;p->keep_vae_on_cpu=false;}
void sd_img_gen_params_init(sd_img_gen_params_t*p){*p={};p->width=1024;p->height=1024;p->seed=42;p->batch_count=1;
    p->sample_params.guidance.txt_cfg=7.0f;p->sample_params.scheduler=KARRAS_SCHEDULER;
    p->sample_params.sample_method=EULER_SAMPLE_METHOD;p->sample_params.sample_steps=20;}
void sd_sample_params_init(sd_sample_params_t*p){*p={};p->guidance.txt_cfg=7.0f;p->scheduler=KARRAS_SCHEDULER;
    p->sample_method=EULER_SAMPLE_METHOD;p->sample_steps=20;}

// new_sd_ctx
sd_ctx_t*new_sd_ctx(const sd_ctx_params_t*p){
    auto c=new sdxl_ctx();if(!c->st.load(p->model_path)){delete c;return nullptr;}
    c->n_threads=p->n_threads>0?p->n_threads:4;
    size_t total=0;for(int i=0;i<(int)c->st.names.size();i++){auto&n=c->st.names[i];
        if(n.find("model.diffusion_model.")!=0&&n.find("first_stage_model.")!=0&&n.find("alphas_cumprod")!=0)continue;
        size_t e=1;for(auto&s:c->st.shapes[i])e*=s;total+=e;}
    size_t ctx_size=total*2+256*1024*1024;
    fprintf(stderr,"engine: loading %zu elems F16 (%.1fGB)\n",total,total*2.0/(1024*1024*1024));
    ggml_init_params pp={ctx_size,NULL,false};
    c->w.ctx=ggml_init(pp);if(!c->w.ctx){delete c;return nullptr;}
    for(int i=0;i<(int)c->st.names.size();i++){auto&n=c->st.names[i];
        if(n.find("model.diffusion_model.")!=0&&n.find("first_stage_model.")!=0&&n.find("alphas_cumprod")!=0)continue;
        auto&sh=c->st.shapes[i];int nd=(int)sh.size();const ggml_fp16_t*src=f16p(c->st,i);
        struct ggml_tensor*ten=nullptr;size_t ne=1;for(auto&s:sh)ne*=s;
        switch(nd){case 1:ten=ggml_new_tensor_1d(c->w.ctx,GGML_TYPE_F16,sh[0]);memcpy(ten->data,src,ne*2);break;
            case 2:ten=ggml_new_tensor_2d(c->w.ctx,GGML_TYPE_F16,sh[1],sh[0]);memcpy(ten->data,src,ne*2);break;
            case 4:ten=ggml_new_tensor_4d(c->w.ctx,GGML_TYPE_F16,sh[3],sh[2],sh[1],sh[0]);memcpy(ten->data,src,ne*2);break;}
        if(ten)c->w.t[n]=ten;}
    // Denoiser
    auto*acp=c->w.get("alphas_cumprod");
    if(!acp)for(auto&[n,t]:c->w.t){if(n.find("alphas_cumprod")!=std::string::npos){float tmp[1000];
        auto*s=(ggml_fp16_t*)t->data;for(int j=0;j<1000;j++)tmp[j]=ggml_fp16_to_fp32(s[j]);c->den.init(tmp);acp=t;break;}}
    if(!acp){float def[1000];float ls=sqrtf(0.00085f),le=sqrtf(0.012f),amt=le-ls,prod=1;
        for(int i=0;i<1000;i++){float b=ls+amt*i/999;prod*=1-b*b;def[i]=prod;}c->den.init(def);}
    // Backend
    int nd_cuda=ggml_backend_cuda_get_device_count();
    if(nd_cuda>0){c->compute_backend=ggml_backend_cuda_init(0);c->use_gpu=(c->compute_backend!=nullptr);}
    if(!c->compute_backend)c->compute_backend=ggml_backend_cpu_init();
    fprintf(stderr,"engine: %s\n",c->use_gpu?"CUDA":"CPU");
    // GPU weight offload
    if(c->use_gpu){auto*olg=ggml_init({64*1024*1024,NULL,true});
        for(auto&[n,t]:c->w.t){auto*d=ggml_dup_tensor(olg,t);ggml_set_name(d,n.c_str());}
        c->w.buf=ggml_backend_alloc_ctx_tensors(olg,c->compute_backend);
        if(c->w.buf){for(auto&[n,t]:c->w.t){auto*dst=ggml_get_tensor(olg,n.c_str());
            if(dst&&!t->view_src)ggml_backend_tensor_set(dst,t->data,0,ggml_nbytes(t));}
            for(auto&[n,t]:c->w.t){auto*dst=ggml_get_tensor(olg,n.c_str());if(dst)c->w.t[n]=dst;}
            ggml_free(c->w.ctx);c->w.ctx=olg;fprintf(stderr,"  GPU offload OK\n");}
        else{ggml_free(olg);c->use_gpu=false;ggml_backend_free(c->compute_backend);c->compute_backend=ggml_backend_cpu_init();
            fprintf(stderr,"  offload failed, CPU\n");}}
    c->ok=true;return c;}
void free_sd_ctx(sd_ctx_t*ctx){delete ctx;}

// UNet builder (no attention yet)
static struct ggml_tensor*build_unet(struct ggml_context*c,WT&w,struct ggml_tensor*inp,struct ggml_tensor*te){
    auto rb=[&](struct ggml_tensor*x,const std::string&p,int oc){
        auto hh=gn_silu(c,x,32,p+".in_layers.0",w);hh=conv(c,hh,p+".in_layers.2",w,1,1);
        auto te2=lin(c,ggml_silu(c,te),p+".emb_layers.1",w);
        te2=ggml_reshape_4d(c,te2,1,1,te2->ne[0],1);hh=ggml_add(c,hh,te2);
        hh=gn_silu(c,hh,32,p+".out_layers.0",w);hh=conv(c,hh,p+".out_layers.3",w,1,1);
        auto*sk=w.get(p+".skip_connection.weight");if(sk){auto sc=conv(c,x,p+".skip_connection",w,1,0);hh=ggml_add(c,hh,sc);}
        else hh=ggml_add(c,hh,x);return hh;};
    auto ds=[&](struct ggml_tensor*x,std::string p){return conv(c,x,p+".op",w,2,1);};
    auto ms=[&](struct ggml_tensor*x,std::vector<struct ggml_tensor*>&sk){int tw=x->ne[0],th=x->ne[1];
        for(int i=(int)sk.size()-1;i>=0;i--)if(sk[i]->ne[0]==tw&&sk[i]->ne[1]==th){auto*r=ggml_concat(c,x,sk[i],2);sk.erase(sk.begin()+i);return r;}return x;};
    auto us=[&](struct ggml_tensor*x,std::string p){x=ggml_upscale(c,x,2,GGML_SCALE_MODE_NEAREST);return conv(c,x,p+".2.conv",w,1,1);};
    auto hh=conv(c,inp,"input_blocks.0.0",w,1,1);std::vector<struct ggml_tensor*>sk;sk.push_back(hh);
    hh=rb(hh,"input_blocks.1.0",320);sk.push_back(hh);hh=rb(hh,"input_blocks.2.0",320);sk.push_back(hh);
    hh=ds(hh,"input_blocks.3.0");sk.push_back(hh);
    hh=rb(hh,"input_blocks.4.0",640);sk.push_back(hh);hh=rb(hh,"input_blocks.5.0",640);sk.push_back(hh);
    hh=ds(hh,"input_blocks.6.0");sk.push_back(hh);
    hh=rb(hh,"input_blocks.7.0",1280);sk.push_back(hh);hh=rb(hh,"input_blocks.8.0",1280);sk.push_back(hh);
    hh=rb(hh,"middle_block.0",1280);hh=rb(hh,"middle_block.2",1280);
    int nob=0;auto ob=[&](int oc,bool up){hh=ms(hh,sk);
        hh=rb(hh,"output_blocks."+std::to_string(nob)+".0",oc);nob++;if(up)hh=us(hh,"output_blocks."+std::to_string(nob-1));};
    ob(1280,false);ob(1280,false);ob(1280,true);ob(640,false);ob(640,false);ob(640,true);
    ob(320,false);ob(320,false);ob(320,false);
    hh=gn_silu(c,hh,32,"out.0",w);return conv(c,hh,"out.2",w,1,1);}

// UNet compute per step
static bool compute_unet(sdxl_ctx*ctx,std::vector<float>&out,const std::vector<float>&inp,int lh,int lw,float timestep){
    ggml_init_params gp={32*1024*1024,NULL,true};auto*gctx=ggml_init(gp);if(!gctx)return false;
    auto*inp_t=ggml_new_tensor_4d(gctx,GGML_TYPE_F32,lw,lh,4,1);
    auto*ts=ggml_new_tensor_1d(gctx,GGML_TYPE_F32,1);auto*te=ggml_timestep_embedding(gctx,ts,320,10000);
    te=lin(gctx,te,"time_embed.0",ctx->w);te=ggml_silu(gctx,te);te=lin(gctx,te,"time_embed.2",ctx->w);
    auto ev=[&](float v){auto*t=ggml_new_tensor_1d(gctx,GGML_TYPE_F32,1);return ggml_timestep_embedding(gctx,t,256,10000);};
    auto*ae=ggml_concat(gctx,ev(128.f),ev(128.f),0);ae=ggml_concat(gctx,ae,ev(0),0);
    ae=ggml_concat(gctx,ae,ev(0),0);ae=ggml_concat(gctx,ae,ev(128.f),0);ae=ggml_concat(gctx,ae,ev(128.f),0);
    auto*pt=ggml_new_tensor_1d(gctx,GGML_TYPE_F32,1280);ae=ggml_concat(gctx,pt,ae,0);
    ae=lin(gctx,ae,"label_emb.0.0",ctx->w);ae=ggml_silu(gctx,ae);ae=lin(gctx,ae,"label_emb.0.2",ctx->w);te=ggml_add(gctx,te,ae);
    auto*result=build_unet(gctx,ctx->w,inp_t,te);auto*gf=ggml_new_graph(gctx);ggml_build_forward_expand(gf,result);
    if(ctx->use_gpu){auto*al=ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->compute_backend));
        if(!al||!ggml_gallocr_reserve(al,gf)){ggml_gallocr_free(al);ggml_free(gctx);return false;}
        if(!ggml_gallocr_alloc_graph(al,gf)){ggml_gallocr_free(al);ggml_free(gctx);return false;}
        ggml_backend_tensor_set_async(ctx->compute_backend,ts,&timestep,0,4);
        ggml_backend_tensor_set_async(ctx->compute_backend,inp_t,inp.data(),0,4*lh*lw*4);
        ggml_backend_synchronize(ctx->compute_backend);
        ggml_backend_graph_compute(ctx->compute_backend,gf);out.resize(4*lh*lw);
        ggml_backend_tensor_get(result,out.data(),0,4*lh*lw*4);ggml_gallocr_free(al);ggml_free(gctx);
    }else{auto*al=ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        if(!al||!ggml_gallocr_reserve(al,gf)){ggml_gallocr_free(al);ggml_free(gctx);return false;}
        if(!ggml_gallocr_alloc_graph(al,gf)){ggml_gallocr_free(al);ggml_free(gctx);return false;}
        ((float*)ts->data)[0]=timestep;memcpy(inp_t->data,inp.data(),4*lh*lw*4);
        ggml_backend_graph_compute(ctx->compute_backend,gf);out.assign((float*)result->data,((float*)result->data)+4*lh*lw);
        ggml_gallocr_free(al);ggml_free(gctx);}return true;}

// generate_image
sd_image_t*generate_image(sd_ctx_t*ctx,const sd_img_gen_params_t*p){
    if(!ctx||!ctx->ok||!p)return nullptr;auto c=ctx;int W=p->width,H=p->height,steps=p->sample_params.sample_steps;
    float cfg=p->sample_params.guidance.txt_cfg;int64_t seed=p->seed;if(steps<1)steps=1;
    float smin=0.029f,smax=14.615f,rho=7.f;if(smin<=1e-6f)smin=1e-6f;
    std::vector<float>sigs(steps+1);float mir=powf(smin,1.f/rho),mar=powf(smax,1.f/rho);
    for(int i=0;i<steps;i++)sigs[i]=powf(mar+(steps>1?(float)i/(steps-1):0.f)*(mir-mar),rho);sigs[steps]=0.f;
    int lh=H/8,lw=W/8,ls=4*lh*lw;std::vector<float>noise(ls),x(ls),eps(ls);
    srand((unsigned)seed);for(auto&v:noise)v=(float)rand()/RAND_MAX*2-1;for(int i=0;i<ls;i++)x[i]=noise[i]*sigs[0];
    for(int step=0;step<steps;step++){float s=sigs[step],sn=sigs[step+1],cs,co,ci;c->den.sc(s,cs,co,ci);
        std::vector<float>sc(ls);for(int j=0;j<ls;j++)sc[j]=x[j]*ci;float ts_v=c->den.s2t(s);
        if(!compute_unet(c,eps,sc,lh,lw,ts_v))return nullptr;
        for(int j=0;j<ls;j++)x[j]+=eps[j]*(sn-s);fprintf(stderr,"\rstep %d/%d",step+1,steps);fflush(stderr);}fprintf(stderr,"\n");
    // VAE decode
    ggml_init_params gp={32*1024*1024,NULL,true};auto*gctx=ggml_init(gp);if(!gctx)return nullptr;
    auto*z=ggml_new_tensor_4d(gctx,GGML_TYPE_F32,lw,lh,4,1);
    auto rb=[&](ggml_tensor*x,const std::string&pfx,int){auto r=gn_silu(gctx,x,32,pfx+".norm1",ctx->w);
        r=conv(gctx,r,pfx+".conv1",ctx->w,1,1);r=gn_silu(gctx,r,32,pfx+".norm2",ctx->w);
        r=conv(gctx,r,pfx+".conv2",ctx->w,1,1);
        auto*sk=ctx->w.get(pfx+".nin_shortcut.weight");if(sk)r=ggml_add(gctx,r,conv(gctx,x,pfx+".nin_shortcut",ctx->w,1,0));
        return r;};
    auto h=conv(gctx,z,"first_stage_model.decoder.conv_in",ctx->w,1,1);
    h=rb(h,"first_stage_model.decoder.mid.block_1",0);h=rb(h,"first_stage_model.decoder.mid.block_2",0);
    for(int lvl=3;lvl>=0;lvl--){auto dp="first_stage_model.decoder.up."+std::to_string(lvl);
        h=rb(h,dp+".block.0",0);h=rb(h,dp+".block.1",0);h=rb(h,dp+".block.2",0);
        if(lvl>0){h=ggml_upscale(gctx,h,2,GGML_SCALE_MODE_NEAREST);
            auto*uw=ctx->w.get(dp+".upsample.conv.weight");if(uw)h=conv(gctx,h,dp+".upsample.conv",ctx->w,1,1);}}
    h=ggml_group_norm(gctx,h,32,1e-6f);h=ggml_silu(gctx,h);h=conv(gctx,h,"first_stage_model.decoder.conv_out",ctx->w,1,1);
    auto*gf=ggml_new_graph(gctx);ggml_build_forward_expand(gf,h);
    auto*al=ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->compute_backend));
    if(!al||!ggml_gallocr_reserve(al,gf)){ggml_gallocr_free(al);ggml_free(gctx);return nullptr;}
    if(!ggml_gallocr_alloc_graph(al,gf)){ggml_gallocr_free(al);ggml_free(gctx);return nullptr;}
    if(ctx->use_gpu){ggml_backend_tensor_set_async(ctx->compute_backend,z,x.data(),0,ls*4);ggml_backend_synchronize(ctx->compute_backend);}
    else memcpy(z->data,x.data(),ls*4);
    ggml_backend_graph_compute(ctx->compute_backend,gf);
    int oh=h->ne[1],ow=h->ne[0],oc=h->ne[2];std::vector<float>buf(ow*oh*oc);
    if(ctx->use_gpu)ggml_backend_tensor_get(h,buf.data(),0,buf.size()*4);else memcpy(buf.data(),h->data,buf.size()*4);
    ggml_gallocr_free(al);ggml_free(gctx);
    sd_image_t*img=(sd_image_t*)malloc(sizeof(sd_image_t));if(!img)return nullptr;
    img->width=ow;img->height=oh;img->channel=3;img->data=(uint8_t*)malloc(ow*oh*3);if(!img->data){free(img);return nullptr;}
    for(int y=0;y<oh;y++)for(int x=0;x<ow;x++)for(int cc=0;cc<3;cc++){float v=buf[x+y*ow+cc*ow*oh]*0.5f+0.5f;
        if(v<0)v=0;if(v>1)v=1;img->data[(y*ow+x)*3+cc]=(uint8_t)(v*255);}return img;}
