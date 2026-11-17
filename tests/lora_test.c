#include "lora.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static float output(const AMA10LoRA *l,const float *h,size_t k){float *z=(float*)calloc(l->vocabulary,sizeof(float));if(!z)return NAN;ama10_lora_apply(l,h,z);float x=z[k];free(z);return x;}
int main(void){AMA10Model m={0};AMA10LoRA l={0},r={0},wrong={0};float h[8]={0.2f,-0.4f,0.1f,0.8f,-0.5f,0.3f,0.6f,-0.2f};int ok=0;
    if(!ama10_model_init(&m,256,8,8,123)||!ama10_lora_init(&l,&m,2,0.25f,11,22,7))goto done;
    l.b[3*l.rank+0]=0.3f;l.b[3*l.rank+1]=-0.2f;float analytic=l.scale*l.b[3*l.rank+1];float original=l.a[1*l.width+2],eps=1e-3f;
    float yp; l.a[1*l.width+2]=original+eps;yp=output(&l,h,3);float ym;l.a[1*l.width+2]=original-eps;ym=output(&l,h,3);l.a[1*l.width+2]=original;
    analytic*=h[2];if(fabsf((yp-ym)/(2*eps)-analytic)>2e-4f)goto done;
    if(!ama10_lora_save(&l,"build/lora.test")||!ama10_lora_load(&r,"build/lora.test",&m,11,22)||ama10_lora_load(&wrong,"build/lora.test",&m,12,22))goto done;
    if(fabsf(output(&l,h,3)-output(&r,h,3))>1e-7f)goto done;
    ok=1;
done:ama10_lora_free(&l);ama10_lora_free(&r);ama10_lora_free(&wrong);ama10_model_free(&m);if(!ok){fprintf(stderr,"LoRA serialization or finite-difference test failed\n");return 1;}puts("LoRA adapter finite-difference, serialization, and base-binding checks: ok");return 0;}
