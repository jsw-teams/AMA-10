#define CL_TARGET_OPENCL_VERSION 120
#include "ama10_imggen.h"
#include <CL/cl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_kernel_source(const char *path,char **source,size_t *length){
    FILE *f=fopen(path,"rb");if(!f)return 0;
    if(fseek(f,0,SEEK_END)!=0){fclose(f);return 0;}long n=ftell(f);
    if(n<=0||fseek(f,0,SEEK_SET)!=0){fclose(f);return 0;}
    char *p=(char*)malloc((size_t)n+1);if(!p){fclose(f);return 0;}
    int ok=fread(p,1,(size_t)n,f)==(size_t)n;if(fclose(f)!=0)ok=0;
    if(!ok){free(p);return 0;}p[n]=0;*source=p;*length=(size_t)n;return 1;
}
static void release_mem(cl_mem *objects,size_t count){for(size_t i=0;i<count;i++)if(objects[i])clReleaseMemObject(objects[i]);}
int ama10_image_spatial_codec_train_opencl(AMA10ImageCodebook *book,const float *patches,
                                           size_t patch_count,size_t input_width,const char *kernel_path){
    if(!book||!patches||!patch_count||!input_width||!book->latent_width||!book->encoder||
       !book->decoder||!book->encoder_bias||!book->decoder_bias||!kernel_path||
       input_width!=book->feature_width)return 0;
    const size_t latent=book->latent_width;
    if(input_width>SIZE_MAX/(latent+1)||latent>SIZE_MAX/(input_width+1)||
       input_width>SIZE_MAX/sizeof(float)||latent>SIZE_MAX/sizeof(float))return 0;
    size_t samples=0;float *training=NULL;
    cl_platform_id platform=NULL;cl_device_id device=NULL;cl_context context=NULL;cl_command_queue queue=NULL;
    cl_program program=NULL;cl_kernel encode=NULL,decode=NULL,hidden_delta=NULL,update_decoder=NULL,update_encoder=NULL;
    cl_mem buffers[8]={0};char *source=NULL;size_t source_length=0;int ok=0;cl_int error=CL_SUCCESS;
    cl_uint platform_count=0;
    if(clGetPlatformIDs(0,NULL,&platform_count)!=CL_SUCCESS||!platform_count)goto done;
    cl_platform_id *platforms=(cl_platform_id*)malloc((size_t)platform_count*sizeof(*platforms));if(!platforms)goto done;
    if(clGetPlatformIDs(platform_count,platforms,NULL)!=CL_SUCCESS){free(platforms);goto done;}
    for(cl_uint i=0;i<platform_count&&!device;i++)
        (void)clGetDeviceIDs(platforms[i],CL_DEVICE_TYPE_GPU,1,&device,NULL);
    if(device)for(cl_uint i=0;i<platform_count;i++){cl_uint ndev=0;cl_device_id *devices=NULL;
        if(clGetDeviceIDs(platforms[i],CL_DEVICE_TYPE_GPU,0,NULL,&ndev)==CL_SUCCESS&&ndev){
            devices=(cl_device_id*)malloc((size_t)ndev*sizeof(*devices));
            if(devices&&clGetDeviceIDs(platforms[i],CL_DEVICE_TYPE_GPU,ndev,devices,NULL)==CL_SUCCESS)
                for(cl_uint j=0;j<ndev;j++)if(devices[j]==device){platform=platforms[i];break;}
            free(devices);if(platform)break;
        }}
    free(platforms);if(!device||!platform)goto done;
    cl_ulong global_memory=0,max_allocation=0;
    if(clGetDeviceInfo(device,CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(global_memory),&global_memory,NULL)!=CL_SUCCESS||
       clGetDeviceInfo(device,CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof(max_allocation),&max_allocation,NULL)!=CL_SUCCESS||
       !global_memory||!max_allocation)goto done;
    size_t input_bytes=input_width*sizeof(float);
    size_t max_samples_by_allocation=(size_t)(max_allocation/(cl_ulong)input_bytes);
    if(input_width>SIZE_MAX/2||latent>SIZE_MAX/2||2*input_width>SIZE_MAX-2*latent)goto done;
    size_t bytes_per_sample=(2*input_width+2*latent)*sizeof(float);
    if(bytes_per_sample/sizeof(float)!=2*input_width+2*latent)goto done;
    cl_ulong half_device_memory=global_memory/2;
    size_t device_budget=half_device_memory>(cl_ulong)SIZE_MAX?SIZE_MAX:(size_t)half_device_memory;
    size_t max_samples_by_budget=device_budget/bytes_per_sample;
    samples=patch_count;
    if(samples>max_samples_by_allocation)samples=max_samples_by_allocation;
    if(samples>max_samples_by_budget)samples=max_samples_by_budget;
    if(samples>UINT_MAX)samples=UINT_MAX;
    if(!samples||samples>SIZE_MAX/input_width||samples*input_width>SIZE_MAX/sizeof(float)||
       samples>SIZE_MAX/latent||samples*latent>SIZE_MAX/sizeof(float))goto done;
    fprintf(stderr,"spatial_codec_fit_samples=%zu/%zu device_budget_bytes=%zu device=%llu max_allocation=%llu\n",
        samples,patch_count,device_budget,(unsigned long long)global_memory,(unsigned long long)max_allocation);
    training=(float*)malloc(samples*input_width*sizeof(float));if(!training)goto done;
    for(size_t s=0;s<samples;s++){
        size_t index=s*patch_count/samples;memcpy(training+s*input_width,patches+index*input_width,input_width*sizeof(float));
        for(size_t j=0;j<input_width;j++)if(!isfinite(training[s*input_width+j]))goto done;
    }
    context=clCreateContext(NULL,1,&device,NULL,NULL,&error);if(error!=CL_SUCCESS||!context)goto done;
    queue=clCreateCommandQueue(context,device,0,&error);if(error!=CL_SUCCESS||!queue)goto done;
    if(!read_kernel_source(kernel_path,&source,&source_length))goto done;
    {const char *ptr=source;program=clCreateProgramWithSource(context,1,&ptr,&source_length,&error);}free(source);source=NULL;
    if(error!=CL_SUCCESS||!program)goto done;
    error=clBuildProgram(program,1,&device,"-cl-std=CL1.2",NULL,NULL);
    if(error!=CL_SUCCESS){size_t n=0;clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,0,NULL,&n);
        char *log=n?(char*)malloc(n+1):NULL;if(log){clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,n,log,NULL);log[n]=0;fprintf(stderr,"OpenCL spatial codec build failed: %s\n",log);free(log);}goto done;}
#define GET_KERNEL(field,name) do{field=clCreateKernel(program,name,&error);if(error!=CL_SUCCESS||!field)goto done;}while(0)
    GET_KERNEL(encode,"ama10_codec_encode");GET_KERNEL(decode,"ama10_codec_decode");
    GET_KERNEL(hidden_delta,"ama10_codec_hidden_delta");GET_KERNEL(update_decoder,"ama10_codec_update_decoder");
    GET_KERNEL(update_encoder,"ama10_codec_update_encoder");
#undef GET_KERNEL
    size_t xbytes=samples*input_width*sizeof(float),zbytes=samples*latent*sizeof(float);
    size_t ebytes=latent*input_width*sizeof(float),dbytes=input_width*latent*sizeof(float);
    buffers[0]=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,xbytes,training,&error);
    if(error==CL_SUCCESS)buffers[1]=clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,ebytes,book->encoder,&error);
    if(error==CL_SUCCESS)buffers[2]=clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,latent*sizeof(float),book->encoder_bias,&error);
    if(error==CL_SUCCESS)buffers[3]=clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,dbytes,book->decoder,&error);
    if(error==CL_SUCCESS)buffers[4]=clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,input_width*sizeof(float),book->decoder_bias,&error);
    if(error==CL_SUCCESS)buffers[5]=clCreateBuffer(context,CL_MEM_READ_WRITE,zbytes,NULL,&error);
    if(error==CL_SUCCESS)buffers[6]=clCreateBuffer(context,CL_MEM_READ_WRITE,xbytes,NULL,&error);
    if(error==CL_SUCCESS)buffers[7]=clCreateBuffer(context,CL_MEM_READ_WRITE,zbytes,NULL,&error);
    for(size_t i=0;i<8;i++)if(!buffers[i])goto done;
    cl_uint sn=(cl_uint)samples,d=(cl_uint)input_width,l=(cl_uint)latent;
    cl_mem encode_args[]={buffers[0],buffers[1],buffers[2],buffers[5]};
    for(cl_uint i=0;i<4;i++)if(clSetKernelArg(encode,i,sizeof(cl_mem),&encode_args[i])!=CL_SUCCESS)goto done;
    if(clSetKernelArg(encode,4,sizeof(sn),&sn)!=CL_SUCCESS||clSetKernelArg(encode,5,sizeof(d),&d)!=CL_SUCCESS||clSetKernelArg(encode,6,sizeof(l),&l)!=CL_SUCCESS)goto done;
    cl_mem decode_args[]={buffers[5],buffers[3],buffers[4],buffers[0],buffers[6]};
    for(cl_uint i=0;i<5;i++)if(clSetKernelArg(decode,i,sizeof(cl_mem),&decode_args[i])!=CL_SUCCESS)goto done;
    if(clSetKernelArg(decode,5,sizeof(sn),&sn)!=CL_SUCCESS||clSetKernelArg(decode,6,sizeof(d),&d)!=CL_SUCCESS||clSetKernelArg(decode,7,sizeof(l),&l)!=CL_SUCCESS)goto done;
    cl_mem hidden_args[]={buffers[5],buffers[3],buffers[6],buffers[7]};
    for(cl_uint i=0;i<4;i++)if(clSetKernelArg(hidden_delta,i,sizeof(cl_mem),&hidden_args[i])!=CL_SUCCESS)goto done;
    if(clSetKernelArg(hidden_delta,4,sizeof(sn),&sn)!=CL_SUCCESS||clSetKernelArg(hidden_delta,5,sizeof(d),&d)!=CL_SUCCESS||clSetKernelArg(hidden_delta,6,sizeof(l),&l)!=CL_SUCCESS)goto done;
    const float learning_rate=0.01f;cl_mem decoder_args[]={buffers[3],buffers[4],buffers[5],buffers[6]};
    for(cl_uint i=0;i<4;i++)if(clSetKernelArg(update_decoder,i,sizeof(cl_mem),&decoder_args[i])!=CL_SUCCESS)goto done;
    if(clSetKernelArg(update_decoder,4,sizeof(sn),&sn)!=CL_SUCCESS||clSetKernelArg(update_decoder,5,sizeof(d),&d)!=CL_SUCCESS||
       clSetKernelArg(update_decoder,6,sizeof(l),&l)!=CL_SUCCESS||clSetKernelArg(update_decoder,7,sizeof(learning_rate),&learning_rate)!=CL_SUCCESS)goto done;
    cl_mem encoder_args[]={buffers[1],buffers[2],buffers[0],buffers[7]};
    for(cl_uint i=0;i<4;i++)if(clSetKernelArg(update_encoder,i,sizeof(cl_mem),&encoder_args[i])!=CL_SUCCESS)goto done;
    if(clSetKernelArg(update_encoder,4,sizeof(sn),&sn)!=CL_SUCCESS||clSetKernelArg(update_encoder,5,sizeof(d),&d)!=CL_SUCCESS||
       clSetKernelArg(update_encoder,6,sizeof(l),&l)!=CL_SUCCESS||clSetKernelArg(update_encoder,7,sizeof(learning_rate),&learning_rate)!=CL_SUCCESS)goto done;
    size_t epochs=64,global_encode=samples*latent,global_decode=samples*input_width;
    size_t global_decoder=input_width*(latent+1),global_encoder=latent*(input_width+1);
    for(size_t epoch=0;epoch<epochs&&error==CL_SUCCESS;epoch++){
        error=clEnqueueNDRangeKernel(queue,encode,1,NULL,&global_encode,NULL,0,NULL,NULL);
        if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(queue,decode,1,NULL,&global_decode,NULL,0,NULL,NULL);
        if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(queue,hidden_delta,1,NULL,&global_encode,NULL,0,NULL,NULL);
        if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(queue,update_decoder,1,NULL,&global_decoder,NULL,0,NULL,NULL);
        if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(queue,update_encoder,1,NULL,&global_encoder,NULL,0,NULL,NULL);
    }
    if(error!=CL_SUCCESS)goto done;
    error=clEnqueueReadBuffer(queue,buffers[1],CL_TRUE,0,ebytes,book->encoder,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(queue,buffers[2],CL_TRUE,0,latent*sizeof(float),book->encoder_bias,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(queue,buffers[3],CL_TRUE,0,dbytes,book->decoder,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(queue,buffers[4],CL_TRUE,0,input_width*sizeof(float),book->decoder_bias,0,NULL,NULL);
    if(error==CL_SUCCESS)ok=1;
done:
    free(training);free(source);release_mem(buffers,8);
    if(encode)clReleaseKernel(encode);
    if(decode)clReleaseKernel(decode);
    if(hidden_delta)clReleaseKernel(hidden_delta);
    if(update_decoder)clReleaseKernel(update_decoder);
    if(update_encoder)clReleaseKernel(update_encoder);
    if(program)clReleaseProgram(program);
    if(queue)clReleaseCommandQueue(queue);
    if(context)clReleaseContext(context);
    return ok;
}
