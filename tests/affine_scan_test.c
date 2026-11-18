#include "ama10_affine_scan.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t rng=UINT64_C(0x873af0195d42);
static uint32_t next_u32(void){rng^=rng>>12;rng^=rng<<25;rng^=rng>>27;return (uint32_t)((rng*UINT64_C(2685821657736338717))>>32);}
static float unit(void){return (float)(next_u32()>>8)*(1.0f/16777216.0f);}
static int close_arrays(const float *a,const float *b,size_t n,float tol){
    for(size_t i=0;i<n;i++){float scale=1.0f+fmaxf(fabsf(a[i]),fabsf(b[i]));if(!isfinite(a[i])||!isfinite(b[i])||fabsf(a[i]-b[i])>tol*scale)return 0;}return 1;
}
int main(int argc,char **argv){
    if(argc!=2){fprintf(stderr,"usage: ama10_affine_scan_test <kernel-source>\n");return 2;}
    const size_t batch=3,length=513,width=128,count=batch*length*width;
    float *a=(float*)malloc(count*sizeof(float)),*b=(float*)malloc(count*sizeof(float));
    float *cpu_a=(float*)malloc(count*sizeof(float)),*cpu_b=(float*)malloc(count*sizeof(float));
    float *gpu_a=(float*)malloc(count*sizeof(float)),*gpu_b=(float*)malloc(count*sizeof(float));
    if(!a||!b||!cpu_a||!cpu_b||!gpu_a||!gpu_b)return 1;
    for(size_t i=0;i<count;i++){a[i]=0.997f+unit()*0.002f;b[i]=(unit()-0.5f)*0.002f;}
    AMA10AffineScanOpenCL *gpu=ama10_affine_scan_opencl_create(argv[1]);
    size_t sequence_bytes=length*width*sizeof(float);
    int ok=gpu&&ama10_affine_scan_opencl_set_memory_budget(gpu,4*sequence_bytes)&&
        ama10_affine_scan_cpu(a,b,batch,length,width,cpu_a,cpu_b)&&
        ama10_affine_scan_opencl_run(gpu,a,b,batch,length,width,gpu_a,gpu_b)&&
        close_arrays(cpu_a,gpu_a,count,2.0e-5f)&&close_arrays(cpu_b,gpu_b,count,5.0e-4f);
    for(size_t i=0;i<count&&ok;i++){a[i]=0.2f+unit()*0.75f;b[i]=(unit()-0.5f)*0.05f;}
    if(ok)ok=ama10_affine_scan_cpu(a,b,batch,length,width,cpu_a,cpu_b)&&
        ama10_affine_scan_opencl_run(gpu,a,b,batch,length,width,gpu_a,gpu_b)&&
        close_arrays(cpu_a,gpu_a,count,2.0e-5f)&&close_arrays(cpu_b,gpu_b,count,5.0e-4f);
    if(ok)printf("affine prefix scan batch=%zu length=%zu width=%zu, two gate regimes, adaptive batch tiling, device=%s: CPU/OpenCL parity ok\n",
                 batch,length,width,ama10_affine_scan_opencl_device(gpu));
    else fprintf(stderr,"affine scan CPU/OpenCL parity failed\n");
    ama10_affine_scan_opencl_destroy(gpu);free(a);free(b);free(cpu_a);free(cpu_b);free(gpu_a);free(gpu_b);return ok?0:1;
}
