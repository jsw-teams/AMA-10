#include "ama10.h"
#include "opencl_backend.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int parse_size(const char *s,size_t *value){
    char *end=NULL;errno=0;unsigned long long n=strtoull(s,&end,10);
    if(errno||!end||*end||!n||n>SIZE_MAX)return 0;
    *value=(size_t)n;return 1;
}
static double seconds(void){
    struct timespec t;if(timespec_get(&t,TIME_UTC)!=TIME_UTC)return 0.0;
    return (double)t.tv_sec+(double)t.tv_nsec*1.0e-9;
}
int main(int argc,char **argv){
    size_t width=256,vocabulary=4096,steps=500;
    if((argc>1&&!parse_size(argv[1],&width))||(argc>2&&!parse_size(argv[2],&vocabulary))||
       (argc>3&&!parse_size(argv[3],&steps))||argc>4){
        fprintf(stderr,"usage: opencl_inference [width=256] [vocabulary=4096] [tokens=500]\n");return 2;
    }
    if(width>UINT32_MAX||vocabulary>UINT32_MAX||width>SIZE_MAX/width||
       vocabulary>SIZE_MAX/width||width>SIZE_MAX/vocabulary){
        fprintf(stderr,"benchmark dimensions exceed supported bounds\n");return 2;
    }
    AMA10Model model={0};AMA10State cpu={0},gpu={0};AMA10OpenCL *backend=NULL;
    float *logits=(float*)malloc(vocabulary*sizeof(float));
    if(!logits||!ama10_model_init(&model,vocabulary,width,width,UINT64_C(20261116))||
       !ama10_state_init(&cpu,width)||!ama10_state_init(&gpu,width)||
       !(backend=ama10_opencl_create(&model,"kernels/ama10_recurrent.cl"))){
        fprintf(stderr,"benchmark initialization failed\n");free(logits);
        ama10_state_free(&cpu);ama10_state_free(&gpu);ama10_model_free(&model);ama10_opencl_destroy(backend);return 1;
    }
    const size_t warmup=16;
    for(size_t i=0;i<warmup;i++){
        uint32_t token=(uint32_t)((i*97)%vocabulary);
        if(!ama10_step(&model,&cpu,token,logits,2,2.0f)||
           !ama10_opencl_step(backend,&gpu,token,logits,2,2.0f)){
            fprintf(stderr,"benchmark warmup failed\n");return 1;
        }
    }
    ama10_state_reset(&cpu);ama10_state_reset(&gpu);
    double start=seconds();
    for(size_t i=0;i<steps;i++)if(!ama10_step(&model,&cpu,(uint32_t)((i*97)%vocabulary),logits,2,2.0f))return 1;
    double cpu_time=seconds()-start;
    start=seconds();
    for(size_t i=0;i<steps;i++)if(!ama10_opencl_step(backend,&gpu,(uint32_t)((i*97)%vocabulary),logits,2,2.0f))return 1;
    double gpu_time=seconds()-start;
    if(!ama10_opencl_reset(backend))return 1;
    start=seconds();
    for(size_t i=0;i<steps;i++)if(!ama10_opencl_step_stream(backend,(uint32_t)((i*97)%vocabulary),logits,2,2.0f))return 1;
    double stream_time=seconds()-start;
    double cpu_rate=(double)steps/cpu_time,gpu_rate=(double)steps/gpu_time,stream_rate=(double)steps/stream_time;
    printf("device=%s width=%zu vocabulary=%zu tokens=%zu depth=2\n",
           ama10_opencl_device_name(backend),width,vocabulary,steps);
    printf("cpu_seconds=%.6f cpu_tokens_per_second=%.2f\n",cpu_time,cpu_rate);
    printf("opencl_seconds=%.6f opencl_tokens_per_second=%.2f speedup=%.3fx\n",
           gpu_time,gpu_rate,cpu_time/gpu_time);
    printf("opencl_stream_seconds=%.6f opencl_stream_tokens_per_second=%.2f speedup=%.3fx\n",
           stream_time,stream_rate,cpu_time/stream_time);
    free(logits);ama10_opencl_destroy(backend);ama10_state_free(&cpu);ama10_state_free(&gpu);
    ama10_model_free(&model);return 0;
}
