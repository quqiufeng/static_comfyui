#include "engine.h"
#include <cstdio>
int main(){
    sd_ctx_params_t p;sd_ctx_params_init(&p);
    p.model_path="/data/models/image/sd_xl_base_1.0.safetensors";
    p.n_threads=1;p.backend="CPU";
    auto c=new_sd_ctx(&p);
    if(!c)return 1;
    // Check first 10 VAE weight names
    printf("VAE weights:\n");
    for(int i=0;i<20;i++){
        auto&n=c->st.names[i];
        if(n.find("first_stage")!=std::string::npos)
            printf("  [%d] %s\n",i,n.c_str());
    }
    free_sd_ctx(c);
}
