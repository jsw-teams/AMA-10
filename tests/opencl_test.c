#include "ama10.h"
#include "opencl_backend.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int close_values(const float *a,const float *b,size_t n,float tolerance){
    for(size_t i=0;i<n;i++){
        float scale=1.0f+fmaxf(fabsf(a[i]),fabsf(b[i]));
        if(fabsf(a[i]-b[i])>tolerance*scale)return 0;
    }
    return 1;
}
static int parity_case(const AMA10Model *m,AMA10OpenCL *gpu,float threshold){
    AMA10State cpu_state={0},gpu_state={0};
    float *cpu_logits=(float*)malloc(m->vocabulary*sizeof(float));
    float *gpu_logits=(float*)malloc(m->vocabulary*sizeof(float));
    if(!cpu_logits||!gpu_logits||!ama10_state_init(&cpu_state,m->width)||
       !ama10_state_init(&gpu_state,m->width)){free(cpu_logits);free(gpu_logits);return 0;}
    int ok=1;
    for(uint32_t t=0;t<12&&ok;t++){
        uint32_t token=(t*7u+3u)%(uint32_t)m->vocabulary;
        ok=ama10_step(m,&cpu_state,token,cpu_logits,4,threshold)&&
           ama10_opencl_step(gpu,&gpu_state,token,gpu_logits,4,threshold)&&
           close_values(cpu_logits,gpu_logits,m->vocabulary,3.0e-4f)&&
           close_values(cpu_state.fast,gpu_state.fast,m->width,3.0e-4f)&&
           close_values(cpu_state.slow,gpu_state.slow,m->width,3.0e-4f)&&
           cpu_state.steps==gpu_state.steps;
    }
    if(!ok)fprintf(stderr,"OpenCL/CPU parity failed at threshold %.4g\n",threshold);
    free(cpu_logits);free(gpu_logits);ama10_state_free(&cpu_state);ama10_state_free(&gpu_state);
    return ok;
}
static int stream_case(const AMA10Model *m,AMA10OpenCL *gpu,float threshold){
    AMA10State cpu={0};float *cpu_logits=(float*)malloc(m->vocabulary*sizeof(float));float *gpu_logits=(float*)malloc(m->vocabulary*sizeof(float));
    if(!cpu_logits||!gpu_logits||!ama10_state_init(&cpu,m->width)||!ama10_opencl_reset(gpu)){free(cpu_logits);free(gpu_logits);ama10_state_free(&cpu);return 0;}
    int ok=1;for(uint32_t t=0;t<12&&ok;t++){uint32_t token=(t*11u+5u)%(uint32_t)m->vocabulary;ok=ama10_step(m,&cpu,token,cpu_logits,4,threshold)&&ama10_opencl_step_stream(gpu,token,gpu_logits,4,threshold)&&close_values(cpu_logits,gpu_logits,m->vocabulary,3.0e-4f);}
    if(!ok)fprintf(stderr,"OpenCL stream logits parity failed at threshold %.4g\n",threshold);
    free(cpu_logits);free(gpu_logits);ama10_state_free(&cpu);return ok;
}
static void fill_gradient(AMA10Gradients *g,unsigned seed){
    size_t sizes[8]={g->width*g->input_width,g->width*g->width,g->width*g->width,g->width*g->width,
                     g->width,g->width,g->input_width*g->vocabulary,g->vocabulary*g->width};
    for(size_t k=0;k<8;k++)for(size_t i=0;i<sizes[k];i++)g->values[k][i]=(float)((int)((i+seed*(k+3))%53)-26)*0.002f;
    g->tokens=3;
}
static int optimizer_case(const AMA10Model *shape,AMA10OpenCL *gpu){
    AMA10Model cpu={0},device={0};AMA10Optimizer co={0},go={0};AMA10Gradients cg={0},gg={0};int ok=0;
    if(!ama10_model_init(&cpu,shape->vocabulary,shape->width,shape->input_width,919)||
       !ama10_model_init(&device,shape->vocabulary,shape->width,shape->input_width,919)||
       !ama10_optimizer_init(&co,&cpu)||!ama10_optimizer_init(&go,&device)||
       !ama10_gradients_init(&cg,&cpu)||!ama10_gradients_init(&gg,&device))goto done;
    for(unsigned update=0;update<2;update++){
        fill_gradient(&cg,update+1);fill_gradient(&gg,update+1);
        int cpu_ok=ama10_optimizer_apply(&cpu,&co,&cg,0.0013f,0.01f);
        int gpu_ok=ama10_opencl_adamw_apply(gpu,&device,&go,&gg,0.0013f,0.01f);
        if(!cpu_ok||!gpu_ok||co.step!=go.step||cg.tokens!=0||gg.tokens!=0)goto done;
    }
    if(!ama10_opencl_adamw_sync(gpu,&device,&go))goto done;
    {
        float *cp[]={cpu.wx,cpu.wh,cpu.wm,cpu.wu,cpu.bias,cpu.gate_bias,cpu.embed,cpu.head};
        float *dp[]={device.wx,device.wh,device.wm,device.wu,device.bias,device.gate_bias,device.embed,device.head};
        size_t sizes[]={cpu.width*cpu.input_width,cpu.width*cpu.width,cpu.width*cpu.width,cpu.width*cpu.width,cpu.width,cpu.width,cpu.input_width*cpu.vocabulary,cpu.vocabulary*cpu.width};
        for(size_t k=0;k<8;k++)if(!close_values(cp[k],dp[k],sizes[k],8.0e-5f)||
           !close_values(co.first_moment[k],go.first_moment[k],sizes[k],8.0e-5f)||
           !close_values(co.second_moment[k],go.second_moment[k],sizes[k],8.0e-5f))goto done;
    }
    ok=1;
done:if(!ok)fprintf(stderr,"OpenCL AdamW update parity failed\n");
    ama10_gradients_free(&cg);ama10_gradients_free(&gg);ama10_optimizer_free(&co);ama10_optimizer_free(&go);ama10_model_free(&cpu);ama10_model_free(&device);return ok;
}
static double head_loss(const float *head,const float *hidden,const float *errors,size_t length,size_t width,size_t vocabulary){
    double loss=0.0;for(size_t t=0;t<length;t++)for(size_t k=0;k<vocabulary;k++){
        double z=0.0;for(size_t j=0;j<width;j++)z+=(double)head[k*width+j]*hidden[t*width+j];
        loss+=z*errors[t*vocabulary+k];
    }return loss;
}
static int head_case(const AMA10Model *m,AMA10OpenCL *gpu){
    const size_t len=5,w=m->width,v=m->vocabulary;size_t hn=len*w,ln=len*v,headn=w*v;
    float *hidden=(float*)malloc(hn*sizeof(float)),*errors=(float*)malloc(ln*sizeof(float));
    float *logits=(float*)malloc(ln*sizeof(float)),*expected=(float*)malloc(ln*sizeof(float));
    float *head_grad=(float*)malloc(headn*sizeof(float)),*hidden_grad=(float*)malloc(hn*sizeof(float));
    float *expected_head=(float*)calloc(headn,sizeof(float)),*expected_hidden=(float*)calloc(hn,sizeof(float));
    if(!hidden||!errors||!logits||!expected||!head_grad||!hidden_grad||!expected_head||!expected_hidden)goto fail;
    for(size_t i=0;i<hn;i++)hidden[i]=(float)((int)(i%23)-11)*0.017f;
    for(size_t i=0;i<ln;i++)errors[i]=(float)((int)(i%17)-8)*0.013f;
    for(size_t t=0;t<len;t++)for(size_t k=0;k<v;k++){
        float z=0;for(size_t j=0;j<w;j++){z+=m->head[k*w+j]*hidden[t*w+j];expected_head[k*w+j]+=errors[t*v+k]*hidden[t*w+j];expected_hidden[t*w+j]+=m->head[k*w+j]*errors[t*v+k];}
        expected[t*v+k]=z;
    }
    if(!ama10_opencl_head_forward(gpu,m->head,hidden,len,logits)||
       !ama10_opencl_head_backward(gpu,m->head,hidden,errors,len,head_grad,hidden_grad)||
       !close_values(logits,expected,ln,5.0e-5f)||
       !close_values(head_grad,expected_head,headn,5.0e-5f)||
       !close_values(hidden_grad,expected_hidden,hn,5.0e-5f))goto fail;
    {
        size_t hi=7,xi=11;float old=m->head[hi],eps=1.0e-3f;
        double plus,minus;float *mutable_head=(float*)m->head;
        mutable_head[hi]=old+eps;plus=head_loss(mutable_head,hidden,errors,len,w,v);
        mutable_head[hi]=old-eps;minus=head_loss(mutable_head,hidden,errors,len,w,v);mutable_head[hi]=old;
        double numeric=(plus-minus)/(2.0*eps);
        if(fabs(numeric-head_grad[hi])>0.002*fmax(1.0,fabs(numeric)))goto fail;
        old=hidden[xi];hidden[xi]=old+eps;plus=head_loss(m->head,hidden,errors,len,w,v);
        hidden[xi]=old-eps;minus=head_loss(m->head,hidden,errors,len,w,v);hidden[xi]=old;
        numeric=(plus-minus)/(2.0*eps);
        if(fabs(numeric-hidden_grad[xi])>0.002*fmax(1.0,fabs(numeric)))goto fail;
    }
    free(hidden);free(errors);free(logits);free(expected);free(head_grad);free(hidden_grad);free(expected_head);free(expected_hidden);return 1;
fail:
    fprintf(stderr,"OpenCL output-head forward/backward or finite-difference check failed\n");
    free(hidden);free(errors);free(logits);free(expected);free(head_grad);free(hidden_grad);free(expected_head);free(expected_hidden);return 0;
}
static int training_gradient_case(const AMA10Model *shape,AMA10OpenCL *gpu){
    AMA10Model cpu={0},hybrid={0};AMA10Optimizer co={0},ho={0};AMA10Gradients cg={0},hg={0};
    AMA10State cs={0},hs={0};uint32_t in[7]={3,5,7,9,11,13,15},target[7]={5,7,9,11,13,15,17};
    float cl=0.0f,hl=0.0f;int ok=0;
    if(!ama10_model_init(&cpu,shape->vocabulary,shape->width,shape->input_width,771)||
       !ama10_model_init(&hybrid,shape->vocabulary,shape->width,shape->input_width,771)||
       !ama10_optimizer_init(&co,&cpu)||!ama10_optimizer_init(&ho,&hybrid)||
       !ama10_gradients_init(&cg,&cpu)||!ama10_gradients_init(&hg,&hybrid)||
       !ama10_state_init(&cs,cpu.width)||!ama10_state_init(&hs,hybrid.width))goto done;
    if(!ama10_train_sequence_accumulate(&cpu,&co,&cs,in,target,7,3,&cg,&cl)||
       !ama10_train_sequence_accumulate_opencl(&hybrid,&ho,&hs,in,target,7,3,&hg,gpu,&hl)||
       fabsf(cl-hl)>2.0e-5f)goto done;
    for(size_t k=0;k<8;k++){size_t n=0;
        switch(k){case 0:n=cpu.width*cpu.input_width;break;case 1:case 2:case 3:n=cpu.width*cpu.width;break;
        case 4:case 5:n=cpu.width;break;case 6:n=cpu.input_width*cpu.vocabulary;break;default:n=cpu.vocabulary*cpu.width;break;}
        if(!close_values(cg.values[k],hg.values[k],n,8.0e-5f))goto done;
    }
    if(cg.tokens!=hg.tokens||!close_values(cs.fast,hs.fast,cpu.width,1.0e-6f)||
       !close_values(cs.slow,hs.slow,cpu.width,1.0e-6f))goto done;
    ok=1;
done:
    if(!ok)fprintf(stderr,"OpenCL output-head training gradient parity failed\n");
    ama10_state_free(&cs);ama10_state_free(&hs);ama10_gradients_free(&cg);ama10_gradients_free(&hg);
    ama10_optimizer_free(&co);ama10_optimizer_free(&ho);ama10_model_free(&cpu);ama10_model_free(&hybrid);return ok;
}
int main(int argc,char **argv){
    if(argc!=2){fprintf(stderr,"usage: ama10_opencl_test <kernel-source>\n");return 2;}
    AMA10Model model={0};AMA10OpenCL *gpu=NULL;
    if(!ama10_model_init(&model,37,24,16,919)||!(gpu=ama10_opencl_create(&model,argv[1])))return 1;
    float probe[37];int ok=!ama10_opencl_step_stream(gpu,0,probe,2,2.0f)&&
        parity_case(&model,gpu,2.0f)&&parity_case(&model,gpu,0.0f)&&stream_case(&model,gpu,2.0f)&&stream_case(&model,gpu,0.0f)&&optimizer_case(&model,gpu)&&head_case(&model,gpu)&&training_gradient_case(&model,gpu);
    if(ok)printf("OpenCL inference, AdamW, output-head finite differences, and recurrent BPTT gradient parity on %s: ok\n",
                 ama10_opencl_device_name(gpu));
    ama10_opencl_destroy(gpu);ama10_model_free(&model);return ok?0:1;
}
