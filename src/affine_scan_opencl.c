#define CL_TARGET_OPENCL_VERSION 120
#include "ama10_affine_scan.h"
#include <CL/cl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct AMA10AffineScanOpenCL {
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel compose;
    cl_mem a[2],b[2];
    size_t capacity,max_alloc_bytes,memory_budget_bytes;
    char device_name[256];
};

int ama10_affine_scan_cpu(const float *a,const float *b,size_t batch,size_t length,
                          size_t width,float *prefix_a,float *prefix_b){
    if(!a||!b||!prefix_a||!prefix_b||!batch||!length||!width||
       batch>SIZE_MAX/length||batch*length>SIZE_MAX/width)return 0;
    const size_t total=batch*length*width;
    for(size_t i=0;i<total;i++)if(!isfinite(a[i])||!isfinite(b[i]))return 0;
    for(size_t n=0;n<batch;n++)for(size_t d=0;d<width;d++){
        const size_t first=(n*length)*width+d;
        prefix_a[first]=a[first];prefix_b[first]=b[first];
        for(size_t t=1;t<length;t++){
            const size_t i=(n*length+t)*width+d,previous=i-width;
            const float next_a=a[i],next_b=b[i];
            prefix_a[i]=next_a*prefix_a[previous];
            prefix_b[i]=next_a*prefix_b[previous]+next_b;
        }
    }
    return 1;
}

static int read_source(const char *path,char **out,size_t *length){
    FILE *f=fopen(path,"rb");if(!f)return 0;
    if(fseek(f,0,SEEK_END)!=0){fclose(f);return 0;}
    long n=ftell(f);if(n<=0||fseek(f,0,SEEK_SET)!=0){fclose(f);return 0;}
    char *p=(char*)malloc((size_t)n+1);if(!p){fclose(f);return 0;}
    int ok=fread(p,1,(size_t)n,f)==(size_t)n;
    if(fclose(f)!=0)ok=0;
    if(!ok){free(p);return 0;}p[n]=0;*out=p;*length=(size_t)n;return 1;
}

AMA10AffineScanOpenCL *ama10_affine_scan_opencl_create(const char *kernel_path){
    if(!kernel_path)return NULL;
    AMA10AffineScanOpenCL *s=(AMA10AffineScanOpenCL*)calloc(1,sizeof(*s));if(!s)return NULL;
    cl_int error=CL_SUCCESS;cl_uint platforms_count=0;cl_platform_id *platforms=NULL;
    if(clGetPlatformIDs(0,NULL,&platforms_count)!=CL_SUCCESS||!platforms_count)goto fail;
    platforms=(cl_platform_id*)malloc((size_t)platforms_count*sizeof(*platforms));if(!platforms)goto fail;
    if(clGetPlatformIDs(platforms_count,platforms,NULL)!=CL_SUCCESS)goto fail;
    for(cl_uint i=0;i<platforms_count&&!s->device;i++){
        cl_device_id device=NULL;
        if(clGetDeviceIDs(platforms[i],CL_DEVICE_TYPE_GPU,1,&device,NULL)==CL_SUCCESS)s->device=device;
    }
    free(platforms);platforms=NULL;if(!s->device)goto fail;
    cl_ulong max_alloc=0;
    if(clGetDeviceInfo(s->device,CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof(max_alloc),&max_alloc,NULL)!=CL_SUCCESS||
       max_alloc>SIZE_MAX)goto fail;
    s->max_alloc_bytes=(size_t)max_alloc;
    (void)clGetDeviceInfo(s->device,CL_DEVICE_NAME,sizeof(s->device_name),s->device_name,NULL);
    s->context=clCreateContext(NULL,1,&s->device,NULL,NULL,&error);if(error!=CL_SUCCESS||!s->context)goto fail;
    s->queue=clCreateCommandQueue(s->context,s->device,0,&error);if(error!=CL_SUCCESS||!s->queue)goto fail;
    char *source=NULL;size_t source_length=0;if(!read_source(kernel_path,&source,&source_length))goto fail;
    const char *source_ptr=source;s->program=clCreateProgramWithSource(s->context,1,&source_ptr,&source_length,&error);free(source);
    if(error!=CL_SUCCESS||!s->program)goto fail;
    error=clBuildProgram(s->program,1,&s->device,"-cl-std=CL1.2",NULL,NULL);
    if(error!=CL_SUCCESS){size_t n=0;(void)clGetProgramBuildInfo(s->program,s->device,CL_PROGRAM_BUILD_LOG,0,NULL,&n);
        if(n){char *log=(char*)malloc(n+1);if(log){(void)clGetProgramBuildInfo(s->program,s->device,CL_PROGRAM_BUILD_LOG,n,log,NULL);log[n]=0;fprintf(stderr,"Affine scan OpenCL build failed:\n%s\n",log);free(log);}}goto fail;}
    s->compose=clCreateKernel(s->program,"ama10_affine_scan_compose",&error);if(error!=CL_SUCCESS||!s->compose)goto fail;
    return s;
fail:
    free(platforms);ama10_affine_scan_opencl_destroy(s);return NULL;
}

static int ensure_buffers(AMA10AffineScanOpenCL *s,size_t bytes){
    if(bytes<=s->capacity)return 1;
    for(size_t i=0;i<2;i++){
        if(s->a[i])clReleaseMemObject(s->a[i]);
        if(s->b[i])clReleaseMemObject(s->b[i]);
        s->a[i]=NULL;s->b[i]=NULL;
    }
    s->capacity=0;
    cl_mem next[4]={0};cl_int error=CL_SUCCESS;
    for(size_t i=0;i<4;i++){
        next[i]=clCreateBuffer(s->context,CL_MEM_READ_WRITE,bytes,NULL,&error);
        if(error!=CL_SUCCESS||!next[i]){
            for(size_t j=0;j<4;j++)if(next[j])clReleaseMemObject(next[j]);
            return 0;
        }
    }
    for(size_t i=0;i<2;i++){if(s->a[i])clReleaseMemObject(s->a[i]);if(s->b[i])clReleaseMemObject(s->b[i]);s->a[i]=next[i];s->b[i]=next[i+2];}
    s->capacity=bytes;return 1;
}

int ama10_affine_scan_opencl_set_memory_budget(AMA10AffineScanOpenCL *s,size_t bytes){
    if(!s)return 0;
    s->memory_budget_bytes=bytes;
    return 1;
}

int ama10_affine_scan_opencl_run(AMA10AffineScanOpenCL *s,const float *a,const float *b,
                                 size_t batch,size_t length,size_t width,
                                 float *prefix_a,float *prefix_b){
    if(!s||!a||!b||!prefix_a||!prefix_b||!batch||!length||!width||
       batch>SIZE_MAX/length||
       batch*length>SIZE_MAX/width||batch*length*width>SIZE_MAX/sizeof(float))return 0;
    const size_t sequence_elements=length*width;
    if(sequence_elements>SIZE_MAX/sizeof(float))return 0;
    const size_t sequence_bytes=sequence_elements*sizeof(float);
    if(sequence_bytes>s->max_alloc_bytes)return 0;
    size_t tile_batch=batch;
    if(tile_batch>s->max_alloc_bytes/sequence_bytes)tile_batch=s->max_alloc_bytes/sequence_bytes;
    if(s->memory_budget_bytes){
        if(sequence_bytes>SIZE_MAX/4)return 0;
        size_t bytes_per_batch=4*sequence_bytes;
        size_t budget_batch=s->memory_budget_bytes/bytes_per_batch;
        if(!budget_batch)return 0;
        if(tile_batch>budget_batch)tile_batch=budget_batch;
    }
    size_t tile_bytes=0;
    while(tile_batch){
        if(tile_batch<=SIZE_MAX/sequence_bytes){tile_bytes=tile_batch*sequence_bytes;
            if(ensure_buffers(s,tile_bytes))break;}
        tile_batch/=2;
    }
    if(!tile_batch)return 0;
    for(size_t start=0;start<batch;start+=tile_batch){
        size_t active_batch=batch-start<tile_batch?batch-start:tile_batch;
        size_t count=active_batch*sequence_elements,bytes=count*sizeof(float),source_offset=start*sequence_elements;
        cl_int error=clEnqueueWriteBuffer(s->queue,s->a[0],CL_TRUE,0,bytes,(void*)(a+source_offset),0,NULL,NULL);
        if(error==CL_SUCCESS)error=clEnqueueWriteBuffer(s->queue,s->b[0],CL_TRUE,0,bytes,(void*)(b+source_offset),0,NULL,NULL);
        unsigned current=0;
        for(size_t offset=1;offset<length&&error==CL_SUCCESS;offset*=2){
            cl_ulong bat=(cl_ulong)active_batch,len=(cl_ulong)length,w=(cl_ulong)width,off=(cl_ulong)offset;
            cl_mem args[]={s->a[current],s->b[current],s->a[1-current],s->b[1-current]};
            for(cl_uint i=0;i<4&&error==CL_SUCCESS;i++)error=clSetKernelArg(s->compose,i,sizeof(cl_mem),&args[i]);
            if(error==CL_SUCCESS)error=clSetKernelArg(s->compose,4,sizeof(bat),&bat);
            if(error==CL_SUCCESS)error=clSetKernelArg(s->compose,5,sizeof(len),&len);
            if(error==CL_SUCCESS)error=clSetKernelArg(s->compose,6,sizeof(w),&w);
            if(error==CL_SUCCESS)error=clSetKernelArg(s->compose,7,sizeof(off),&off);
            size_t global=count;
            if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(s->queue,s->compose,1,NULL,&global,NULL,0,NULL,NULL);
            current=1-current;
            if(offset>SIZE_MAX/2)break;
        }
        if(error==CL_SUCCESS)error=clEnqueueReadBuffer(s->queue,s->a[current],CL_TRUE,0,bytes,prefix_a+source_offset,0,NULL,NULL);
        if(error==CL_SUCCESS)error=clEnqueueReadBuffer(s->queue,s->b[current],CL_TRUE,0,bytes,prefix_b+source_offset,0,NULL,NULL);
        if(error!=CL_SUCCESS)return 0;
    }
    return 1;
}

const char *ama10_affine_scan_opencl_device(const AMA10AffineScanOpenCL *s){return s?s->device_name:"";}
void ama10_affine_scan_opencl_destroy(AMA10AffineScanOpenCL *s){
    if(!s)return;
    for(size_t i=0;i<2;i++){
        if(s->a[i])clReleaseMemObject(s->a[i]);
        if(s->b[i])clReleaseMemObject(s->b[i]);
    }
    if(s->compose)clReleaseKernel(s->compose);
    if(s->program)clReleaseProgram(s->program);
    if(s->queue)clReleaseCommandQueue(s->queue);
    if(s->context)clReleaseContext(s->context);
    free(s);
}
