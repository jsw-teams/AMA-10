#include "ama10.h"
#include "opencl_backend.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int close_values(const float *a,const float *b,size_t n,float tolerance){
    for(size_t i=0;i<n;i++){
        float scale=1.0f+fmaxf(fabsf(a[i]),fabsf(b[i]));
        if(fabsf(a[i]-b[i])>tolerance*scale)return 0;
    }
    return 1;
}
static int report_close(const char *label,const float *a,const float *b,size_t n,float tolerance){
    size_t at=0;float worst=0.0f;
    for(size_t i=0;i<n;i++){float d=fabsf(a[i]-b[i])/(1.0f+fmaxf(fabsf(a[i]),fabsf(b[i])));if(d>worst){worst=d;at=i;}}
    if(worst>tolerance){fprintf(stderr,"%s mismatch at %zu: cpu=%.9g gpu=%.9g relative=%.9g limit=%.9g\n",label,at,a[at],b[at],worst,tolerance);return 0;}
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
static int feature_eval_case(const AMA10Model *m,AMA10OpenCL *gpu){
    const size_t length=6,w=m->width,v=m->vocabulary,e=m->input_width;
    uint32_t tokens[6]={1,4,7,10,13,16};unsigned char mask[6]={0,1,0,1,1,0};
    float features[6*16],initial_fast[24],initial_slow[24];
    float cpu_logits[37],gpu_logits[37],final_fast[24],final_slow[24];AMA10State state={0};
    for(size_t i=0;i<length*e;i++)features[i]=(float)((int)(i%29)-14)*0.021f;
    for(size_t i=0;i<w;i++){initial_fast[i]=(float)((int)(i%7)-3)*0.031f;initial_slow[i]=(float)((int)(i%5)-2)*0.027f;}
    if(!ama10_state_init(&state,w))return 0;
    memcpy(state.fast,initial_fast,w*sizeof(float));memcpy(state.slow,initial_slow,w*sizeof(float));
    int ok=1;
    for(size_t t=0;t<length&&ok;t++){
        memcpy(initial_fast,state.fast,w*sizeof(float));memcpy(initial_slow,state.slow,w*sizeof(float));
        if(mask[t])ok=ama10_step_features(m,&state,features+t*e,cpu_logits,2,2.0f);
        else ok=ama10_step(m,&state,tokens[t],cpu_logits,2,2.0f);
        ok=ok&&ama10_opencl_eval_logits(gpu,m,tokens+t,mask[t]?features+t*e:NULL,
             mask+t,1,2,initial_fast,initial_slow,gpu_logits,final_fast,final_slow)&&
             close_values(cpu_logits,gpu_logits,v,3.0e-4f)&&
             close_values(state.fast,final_fast,w,3.0e-4f)&&close_values(state.slow,final_slow,w,3.0e-4f);
        if(!ok)fprintf(stderr,"feature-step t=%zu mask=%u first_cpu=%.8g first_gpu=%.8g fast=%.8g/%.8g slow=%.8g/%.8g\n",t,(unsigned)mask[t],cpu_logits[0],gpu_logits[0],state.fast[0],final_fast[0],state.slow[0],final_slow[0]);
    }
    if(!ok)fprintf(stderr,"OpenCL multimodal feature-input inference parity failed\n");
    ama10_state_free(&state);return ok;
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
    {
        const size_t length=5;uint32_t tokens[5]={2,7,11,19,23};
        float initial_fast[24]={0},initial_slow[24]={0},final_slow[24];
        float gpu_hidden[5*24],cpu_logits[shape->vocabulary];AMA10State state={0};
        if(!ama10_state_init(&state,device.width)||
           !ama10_opencl_train_forward(gpu,&device,tokens,length,3,initial_fast,initial_slow,
                                       gpu_hidden,final_slow)){
            ama10_state_free(&state);goto done;
        }
        int parity=1;
        for(size_t t=0;t<length&&parity;t++){
            parity=ama10_step(&device,&state,tokens[t],cpu_logits,3,2.0f)&&
                   close_values(state.fast,gpu_hidden+t*device.width,device.width,3.0e-4f);
        }
        parity=parity&&close_values(state.slow,final_slow,device.width,3.0e-4f);
        ama10_state_free(&state);
        if(!parity)goto done;
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
    AMA10State cs={0},hs={0};uint32_t in[32],target[32];
    float cl=0.0f,hl=0.0f;int ok=0;
    for(size_t t=0;t<32;t++){in[t]=(uint32_t)((t*7+3)%shape->vocabulary);target[t]=(uint32_t)((t*11+5)%shape->vocabulary);}
    if(!ama10_model_init(&cpu,shape->vocabulary,shape->width,shape->input_width,771)||
       !ama10_model_init(&hybrid,shape->vocabulary,shape->width,shape->input_width,771)||
       !ama10_optimizer_init(&co,&cpu)||!ama10_optimizer_init(&ho,&hybrid)||
       !ama10_gradients_init(&cg,&cpu)||!ama10_gradients_init(&hg,&hybrid)||
       !ama10_state_init(&cs,cpu.width)||!ama10_state_init(&hs,hybrid.width))goto done;
    if(!ama10_train_sequence_accumulate(&cpu,&co,&cs,in,target,32,3,&cg,&cl)||
       !ama10_train_sequence_accumulate_opencl(&hybrid,&ho,&hs,in,target,32,3,&hg,gpu,&hl)||
       fabsf(cl-hl)>2.0e-5f)goto done;
    for(size_t k=0;k<8;k++){size_t n=0;
        switch(k){case 0:n=cpu.width*cpu.input_width;break;case 1:case 2:case 3:n=cpu.width*cpu.width;break;
        case 4:case 5:n=cpu.width;break;case 6:n=cpu.input_width*cpu.vocabulary;break;default:n=cpu.vocabulary*cpu.width;break;}
        if(!report_close("recurrent gradient",cg.values[k],hg.values[k],n,8.0e-5f))goto done;
    }
    if(cg.tokens!=hg.tokens||!report_close("fast state",cs.fast,hs.fast,cpu.width,1.0e-6f)||
       !report_close("slow state",cs.slow,hs.slow,cpu.width,1.0e-6f))goto done;
    ok=1;
done:
    if(!ok)fprintf(stderr,"OpenCL output-head training gradient parity failed\n");
    ama10_state_free(&cs);ama10_state_free(&hs);ama10_gradients_free(&cg);ama10_gradients_free(&hg);
    ama10_optimizer_free(&co);ama10_optimizer_free(&ho);ama10_model_free(&cpu);ama10_model_free(&hybrid);return ok;
}
static int multimodal_training_gradient_case(const AMA10Model *shape,AMA10OpenCL *gpu){
    AMA10Model cpu={0},hybrid={0};AMA10Optimizer co={0},ho={0};AMA10Gradients cg={0},hg={0};
    AMA10State cs={0},hs={0};uint32_t in[7]={3,5,7,9,11,13,15},target[7]={1,2,9,4,10,22,3};
    uint32_t starts[7]={0,0,5,0,2,18,0},counts[7]={37,37,9,37,12,7,37};
    unsigned char feature_mask[7]={0,0,1,0,0,1,0},loss_mask[7]={0,1,1,0,1,1,0};
    float features[7*16],cpu_features[7*16]={0},gpu_features[7*16]={0},cl=0.0f,hl=0.0f;int ok=0;
    for(size_t i=0;i<7*16;i++)features[i]=(float)((int)(i%31)-15)*0.013f;
    if(!ama10_model_init(&cpu,shape->vocabulary,shape->width,shape->input_width,1171)||
       !ama10_model_init(&hybrid,shape->vocabulary,shape->width,shape->input_width,1171)||
       !ama10_optimizer_init(&co,&cpu)||!ama10_optimizer_init(&ho,&hybrid)||
       !ama10_gradients_init(&cg,&cpu)||!ama10_gradients_init(&hg,&hybrid)||
       !ama10_state_init(&cs,cpu.width)||!ama10_state_init(&hs,hybrid.width))goto done;
    for(size_t i=0;i<cpu.width;i++){cs.fast[i]=hs.fast[i]=(float)((int)(i%9)-4)*0.02f;
        cs.slow[i]=hs.slow[i]=(float)((int)(i%7)-3)*0.015f;}
    if(!ama10_train_multimodal_accumulate_ranges(&cpu,&co,&cs,in,features,feature_mask,loss_mask,
           target,starts,counts,7,3,&cg,cpu_features,&cl)||
       !ama10_train_multimodal_accumulate_ranges_opencl(&hybrid,&ho,&hs,in,features,feature_mask,
           loss_mask,target,starts,counts,7,3,&hg,gpu_features,gpu,&hl)||fabsf(cl-hl)>2.0e-5f)goto done;
    for(size_t k=0;k<8;k++){size_t n=0;
        switch(k){case 0:n=cpu.width*cpu.input_width;break;case 1:case 2:case 3:n=cpu.width*cpu.width;break;
        case 4:case 5:n=cpu.width;break;case 6:n=cpu.input_width*cpu.vocabulary;break;default:n=cpu.vocabulary*cpu.width;break;}
        if(!close_values(cg.values[k],hg.values[k],n,1.5e-4f))goto done;
    }
    if(cg.tokens!=hg.tokens||!close_values(cpu_features,gpu_features,7*16,1.5e-4f)||
       !close_values(cs.fast,hs.fast,cpu.width,1.0e-6f)||!close_values(cs.slow,hs.slow,cpu.width,1.0e-6f))goto done;
    ok=1;
done:
    if(!ok)fprintf(stderr,"OpenCL multimodal/ranged-loss recurrent gradient parity failed\n");
    ama10_state_free(&cs);ama10_state_free(&hs);ama10_gradients_free(&cg);ama10_gradients_free(&hg);
    ama10_optimizer_free(&co);ama10_optimizer_free(&ho);ama10_model_free(&cpu);ama10_model_free(&hybrid);return ok;
}
static int gpu_task_min_norm_update_case(AMA10OpenCL *gpu){
    AMA10Model cpu={0},hybrid={0};AMA10Optimizer co={0},ho={0};
    AMA10Gradients task_a={0},task_b={0},merged={0};
    if(!ama10_model_init(&cpu,37,24,16,8221)||!ama10_model_init(&hybrid,37,24,16,8221)||
       !ama10_optimizer_init(&co,&cpu)||!ama10_optimizer_init(&ho,&hybrid)||
       !ama10_gradients_init(&task_a,&cpu)||!ama10_gradients_init(&task_b,&cpu)||
       !ama10_gradients_init(&merged,&cpu))goto fail;
    size_t sizes[8]={24u*16u,24u*24u,24u*24u,24u*24u,24u,24u,16u*37u,37u*24u};
    float *a[8]={task_a.values[0],task_a.values[1],task_a.values[2],task_a.values[3],
        task_a.values[4],task_a.values[5],task_a.values[6],task_a.values[7]};
    float *b[8]={task_b.values[0],task_b.values[1],task_b.values[2],task_b.values[3],
        task_b.values[4],task_b.values[5],task_b.values[6],task_b.values[7]};
    task_a.tokens=3;task_b.tokens=7;
    for(size_t k=0;k<8;k++)for(size_t i=0;i<sizes[k];i++){
        const float mean=(float)((int)(i%19)-9)*0.0017f;
        const float other=-0.63f*mean+(float)((int)(i%7)-3)*0.0004f;
        a[k][i]=mean*3.0f;b[k][i]=other*7.0f;
    }
    double cpu_cosine=0.0,cpu_weight=0.0,gpu_cosine=0.0,gpu_weight=0.0;
    int ok=ama10_gradients_combine_tasks_with_weights(&merged,&task_a,&task_b,2,&cpu_cosine,&cpu_weight)&&
        ama10_optimizer_apply(&cpu,&co,&merged,0.0001f,0.01f)&&
        ama10_opencl_adamw_apply_tasks(gpu,&hybrid,&ho,&task_a,&task_b,0.0001f,0.01f,&gpu_cosine,&gpu_weight)&&
        ama10_opencl_adamw_sync(gpu,&hybrid,&ho)&&fabs(cpu_cosine-gpu_cosine)<2.0e-5&&
        fabs(cpu_weight-gpu_weight)<2.0e-5&&co.step==ho.step;
    if(ok){
        float *cpu_params[8]={cpu.wx,cpu.wh,cpu.wm,cpu.wu,cpu.bias,cpu.gate_bias,cpu.embed,cpu.head};
        float *gpu_params[8]={hybrid.wx,hybrid.wh,hybrid.wm,hybrid.wu,hybrid.bias,hybrid.gate_bias,hybrid.embed,hybrid.head};
        for(size_t k=0;k<8&&ok;k++)ok=close_values(cpu_params[k],gpu_params[k],sizes[k],2.0e-4f)&&
            close_values(co.first_moment[k],ho.first_moment[k],sizes[k],2.0e-4f)&&
            close_values(co.second_moment[k],ho.second_moment[k],sizes[k],2.0e-4f);
    }
    if(!ok)fprintf(stderr,"GPU min-norm task-gradient merge/AdamW parity failed\n");
    ama10_gradients_free(&task_a);ama10_gradients_free(&task_b);ama10_gradients_free(&merged);
    ama10_optimizer_free(&co);ama10_optimizer_free(&ho);ama10_model_free(&cpu);ama10_model_free(&hybrid);return ok;
fail:
    ama10_gradients_free(&task_a);ama10_gradients_free(&task_b);ama10_gradients_free(&merged);
    ama10_optimizer_free(&co);ama10_optimizer_free(&ho);ama10_model_free(&cpu);ama10_model_free(&hybrid);return 0;
}
int main(int argc,char **argv){
    if(argc!=2){fprintf(stderr,"usage: ama10_opencl_test <kernel-source>\n");return 2;}
    AMA10Model model={0};AMA10OpenCL *gpu=NULL;
    if(!ama10_model_init(&model,37,24,16,919)||!(gpu=ama10_opencl_create(&model,argv[1])))return 1;
    float probe[37];int ok=!ama10_opencl_step_stream(gpu,0,probe,2,2.0f)&&
        parity_case(&model,gpu,2.0f)&&parity_case(&model,gpu,0.0f)&&stream_case(&model,gpu,2.0f)&&stream_case(&model,gpu,0.0f)&&feature_eval_case(&model,gpu)&&optimizer_case(&model,gpu)&&head_case(&model,gpu)&&training_gradient_case(&model,gpu)&&multimodal_training_gradient_case(&model,gpu)&&gpu_task_min_norm_update_case(gpu);
    if(ok)printf("OpenCL inference, AdamW, task min-norm coordination, output-head finite differences, and recurrent BPTT gradient parity on %s: ok\n",
                 ama10_opencl_device_name(gpu));
    ama10_opencl_destroy(gpu);ama10_model_free(&model);return ok?0:1;
}
