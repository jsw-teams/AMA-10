#define CL_TARGET_OPENCL_VERSION 120
#include "opencl_backend.h"
#include <CL/cl.h>
#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct AMA10OpenCL {
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel cell,confidence,logits_kernel,adamw_update,sequence_logits,head_gradient,hidden_gradient;
    cl_kernel cell_backward_derivatives,cell_backward_accumulate,training_forward_cell;
    cl_mem parameters[8];
    cl_mem fast,slow,next_fast,next_slow,confidence_buffer,logits_buffer;
    size_t width,input_width,vocabulary;
    char device_name[256];
    int stream_ready;
    cl_mem adamw_p,adamw_a,adamw_v,adamw_next_p,adamw_next_a,adamw_next_v,adamw_gradient;
    size_t adamw_total;
    uint64_t adamw_step;
    const AMA10Model *adamw_model;
    const AMA10Optimizer *adamw_optimizer;
    float *adamw_model_shadow,*adamw_first_shadow,*adamw_second_shadow;
    int adamw_ready;
    cl_mem bptt_dh,bptt_dm,bptt_next_dh,bptt_next_dm,bptt_dpre,bptt_dgate;
    cl_mem bptt_direct_fast,bptt_direct_slow,bptt_cache,bptt_tokens,bptt_hidden_gradient;
    cl_mem bptt_gradients[8];
    size_t bptt_cache_capacity,bptt_sequence_capacity,bptt_token_capacity,bptt_depth;
    size_t bptt_forward_length,bptt_forward_depth;
    int bptt_forward_ready;
    int bptt_ready;
};

static int load_source(const char *path,char **source,size_t *length){
    FILE *f=fopen(path,"rb");if(!f){perror(path);return 0;}
    if(fseek(f,0,SEEK_END)!=0){fclose(f);return 0;}
    long n=ftell(f);if(n<=0||fseek(f,0,SEEK_SET)!=0){fclose(f);return 0;}
    char *p=(char*)malloc((size_t)n+1);if(!p){fclose(f);return 0;}
    if(fread(p,1,(size_t)n,f)!=(size_t)n){free(p);fclose(f);return 0;}
    if(fclose(f)!=0){free(p);return 0;}
    p[n]='\0';*source=p;*length=(size_t)n;return 1;
}
static cl_mem make_parameter_buffer(cl_context context,const float *values,size_t count){
    cl_int error=CL_SUCCESS;
    return clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                          count*sizeof(float),(void*)values,&error);
}
static cl_mem make_float_buffer(cl_context context,size_t count){
    cl_int error=CL_SUCCESS;
    return clCreateBuffer(context,CL_MEM_READ_WRITE,count*sizeof(float),NULL,&error);
}
static void print_build_log(cl_program program,cl_device_id device){
    size_t size=0;
    if(clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,0,NULL,&size)!=CL_SUCCESS||!size)return;
    char *log=(char*)malloc(size+1);if(!log)return;
    if(clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,size,log,NULL)==CL_SUCCESS){
        log[size]='\0';fprintf(stderr,"OpenCL build log:\n%s\n",log);
    }
    free(log);
}

AMA10OpenCL *ama10_opencl_create(const AMA10Model *m,const char *kernel_path){
    if(!m||!kernel_path||m->width>UINT32_MAX||m->input_width>UINT32_MAX||m->vocabulary>UINT32_MAX)return NULL;
    AMA10OpenCL *b=(AMA10OpenCL*)calloc(1,sizeof(*b));if(!b)return NULL;
    cl_uint platform_count=0;cl_platform_id *platforms=NULL;cl_int error=clGetPlatformIDs(0,NULL,&platform_count);
    if(error!=CL_SUCCESS||!platform_count){fprintf(stderr,"OpenCL platform discovery failed (%d)\n",error);goto fail;}
    platforms=(cl_platform_id*)malloc((size_t)platform_count*sizeof(*platforms));
    if(!platforms||clGetPlatformIDs(platform_count,platforms,NULL)!=CL_SUCCESS)goto fail;
    for(cl_uint i=0;i<platform_count&&!b->device;i++){
        cl_device_id device=NULL;
        if(clGetDeviceIDs(platforms[i],CL_DEVICE_TYPE_GPU,1,&device,NULL)==CL_SUCCESS)b->device=device;
    }
    free(platforms);platforms=NULL;
    if(!b->device){fprintf(stderr,"no OpenCL GPU device found\n");goto fail;}
    (void)clGetDeviceInfo(b->device,CL_DEVICE_NAME,sizeof(b->device_name),b->device_name,NULL);
    b->context=clCreateContext(NULL,1,&b->device,NULL,NULL,&error);
    if(error!=CL_SUCCESS||!b->context)goto fail;
    b->queue=clCreateCommandQueue(b->context,b->device,0,&error);
    if(error!=CL_SUCCESS||!b->queue)goto fail;
    char *source=NULL;size_t source_len=0;
    if(!load_source(kernel_path,&source,&source_len))goto fail;
    const char *source_ptr=source;
    b->program=clCreateProgramWithSource(b->context,1,&source_ptr,&source_len,&error);free(source);
    if(error!=CL_SUCCESS||!b->program)goto fail;
    error=clBuildProgram(b->program,1,&b->device,"-cl-std=CL1.2",NULL,NULL);
    if(error!=CL_SUCCESS){print_build_log(b->program,b->device);goto fail;}
    b->cell=clCreateKernel(b->program,"ama10_recurrent_cell",&error);
    if(error!=CL_SUCCESS||!b->cell)goto fail;
    b->confidence=clCreateKernel(b->program,"ama10_recurrent_confidence",&error);
    if(error!=CL_SUCCESS||!b->confidence)goto fail;
    b->logits_kernel=clCreateKernel(b->program,"ama10_recurrent_logits",&error);
    if(error!=CL_SUCCESS||!b->logits_kernel)goto fail;
    b->adamw_update=clCreateKernel(b->program,"ama10_adamw_update",&error);
    if(error!=CL_SUCCESS||!b->adamw_update)goto fail;
    b->sequence_logits=clCreateKernel(b->program,"ama10_sequence_logits",&error);
    if(error!=CL_SUCCESS||!b->sequence_logits)goto fail;
    b->head_gradient=clCreateKernel(b->program,"ama10_head_gradient",&error);
    if(error!=CL_SUCCESS||!b->head_gradient)goto fail;
    b->hidden_gradient=clCreateKernel(b->program,"ama10_hidden_gradient",&error);
    if(error!=CL_SUCCESS||!b->hidden_gradient)goto fail;
    b->cell_backward_derivatives=clCreateKernel(b->program,"ama10_cell_backward_derivatives",&error);
    if(error!=CL_SUCCESS||!b->cell_backward_derivatives)goto fail;
    b->cell_backward_accumulate=clCreateKernel(b->program,"ama10_cell_backward_accumulate",&error);
    if(error!=CL_SUCCESS||!b->cell_backward_accumulate)goto fail;
    b->training_forward_cell=clCreateKernel(b->program,"ama10_training_forward_cell",&error);
    if(error!=CL_SUCCESS||!b->training_forward_cell)goto fail;
    float *params[]={m->wx,m->wh,m->wm,m->wu,m->bias,m->gate_bias,m->embed,m->head};
    size_t sizes[]={m->width*m->input_width,m->width*m->width,m->width*m->width,
        m->width*m->width,m->width,m->width,m->input_width*m->vocabulary,m->vocabulary*m->width};
    for(size_t i=0;i<8;i++){
        b->parameters[i]=make_parameter_buffer(b->context,params[i],sizes[i]);
        if(!b->parameters[i])goto fail;
    }
    b->fast=clCreateBuffer(b->context,CL_MEM_READ_WRITE,m->width*sizeof(float),NULL,&error);
    if(error!=CL_SUCCESS||!b->fast)goto fail;
    b->slow=clCreateBuffer(b->context,CL_MEM_READ_WRITE,m->width*sizeof(float),NULL,&error);
    if(error!=CL_SUCCESS||!b->slow)goto fail;
    b->next_fast=clCreateBuffer(b->context,CL_MEM_READ_WRITE,m->width*sizeof(float),NULL,&error);
    if(error!=CL_SUCCESS||!b->next_fast)goto fail;
    b->next_slow=clCreateBuffer(b->context,CL_MEM_READ_WRITE,m->width*sizeof(float),NULL,&error);
    if(error!=CL_SUCCESS||!b->next_slow)goto fail;
    b->confidence_buffer=clCreateBuffer(b->context,CL_MEM_WRITE_ONLY,sizeof(float),NULL,&error);
    if(error!=CL_SUCCESS||!b->confidence_buffer)goto fail;
    b->logits_buffer=clCreateBuffer(b->context,CL_MEM_WRITE_ONLY,m->vocabulary*sizeof(float),NULL,&error);
    if(error!=CL_SUCCESS||!b->logits_buffer)goto fail;
    {
        size_t w2=m->width*m->width,we=m->width*m->input_width,ev=m->input_width*m->vocabulary;
        cl_mem *buffers[]={&b->bptt_dh,&b->bptt_dm,&b->bptt_next_dh,&b->bptt_next_dm,
            &b->bptt_dpre,&b->bptt_dgate,&b->bptt_direct_fast,&b->bptt_direct_slow};
        for(size_t i=0;i<8;i++){*buffers[i]=make_float_buffer(b->context,m->width);if(!*buffers[i])goto fail;}
        size_t parameter_sizes[8]={we,w2,w2,w2,m->width,m->width,ev,m->vocabulary*m->width};
        for(size_t i=0;i<8;i++){b->bptt_gradients[i]=make_float_buffer(b->context,parameter_sizes[i]);if(!b->bptt_gradients[i])goto fail;}
        b->bptt_ready=1;
    }
    b->width=m->width;b->input_width=m->input_width;b->vocabulary=m->vocabulary;
    return b;
fail:
    free(platforms);ama10_opencl_destroy(b);fprintf(stderr,"OpenCL recurrent backend initialization failed (%d)\n",error);return NULL;
}

static int opencl_run(AMA10OpenCL *b,AMA10State *s,uint32_t token,float *logits,
                      unsigned max_depth,float threshold){
    if(!b||!logits||token>=b->vocabulary||!max_depth||!isfinite(threshold)||threshold<0.0f||threshold>2.0f||(!s&&!b->stream_ready))return 0;
    if(s&&(!s->fast||!s->slow||s->width!=b->width||max_depth>UINT_MAX-s->steps))return 0;
    if(s&&(clEnqueueWriteBuffer(b->queue,b->fast,CL_TRUE,0,b->width*sizeof(float),s->fast,0,NULL,NULL)!=CL_SUCCESS||
       clEnqueueWriteBuffer(b->queue,b->slow,CL_TRUE,0,b->width*sizeof(float),s->slow,0,NULL,NULL)!=CL_SUCCESS))return 0;
    cl_uint width=(cl_uint)b->width,input_width=(cl_uint)b->input_width;
    cl_uint vocabulary=(cl_uint)b->vocabulary,token_id=(cl_uint)token;
    unsigned executed=0;
    for(unsigned d=0;d<max_depth;d++){
        cl_mem args[]={b->parameters[0],b->parameters[1],b->parameters[2],b->parameters[3],
            b->parameters[4],b->parameters[5],b->parameters[6],b->fast,b->slow,b->next_fast,b->next_slow};
        for(cl_uint i=0;i<11;i++)if(clSetKernelArg(b->cell,i,sizeof(cl_mem),&args[i])!=CL_SUCCESS)return 0;
        if(clSetKernelArg(b->cell,11,sizeof(width),&width)!=CL_SUCCESS||
           clSetKernelArg(b->cell,12,sizeof(input_width),&input_width)!=CL_SUCCESS||
           clSetKernelArg(b->cell,13,sizeof(vocabulary),&vocabulary)!=CL_SUCCESS||
           clSetKernelArg(b->cell,14,sizeof(token_id),&token_id)!=CL_SUCCESS)return 0;
        size_t global=b->width;
        if(clEnqueueNDRangeKernel(b->queue,b->cell,1,NULL,&global,NULL,0,NULL,NULL)!=CL_SUCCESS)return 0;
        float confidence=0.0f;
        if(d+1>=2&&threshold<=1.0f){
            cl_mem measure_args[]={b->fast,b->next_fast,b->confidence_buffer};
            for(cl_uint i=0;i<3;i++)if(clSetKernelArg(b->confidence,i,sizeof(cl_mem),&measure_args[i])!=CL_SUCCESS)return 0;
            if(clSetKernelArg(b->confidence,3,sizeof(width),&width)!=CL_SUCCESS)return 0;
            size_t one=1;
            if(clEnqueueNDRangeKernel(b->queue,b->confidence,1,NULL,&one,NULL,0,NULL,NULL)!=CL_SUCCESS||
               clEnqueueReadBuffer(b->queue,b->confidence_buffer,CL_TRUE,0,sizeof(confidence),&confidence,0,NULL,NULL)!=CL_SUCCESS)return 0;
        }
        cl_mem swap=b->fast;b->fast=b->next_fast;b->next_fast=swap;
        swap=b->slow;b->slow=b->next_slow;b->next_slow=swap;
        executed++;
        if(d+1>=2&&threshold<=1.0f&&confidence>=threshold)break;
    }
    cl_mem output_args[]={b->parameters[7],b->fast,b->logits_buffer};
    for(cl_uint i=0;i<3;i++)if(clSetKernelArg(b->logits_kernel,i,sizeof(cl_mem),&output_args[i])!=CL_SUCCESS)return 0;
    cl_uint width_arg=(cl_uint)b->width,vocab_arg=(cl_uint)b->vocabulary;
    if(clSetKernelArg(b->logits_kernel,3,sizeof(width_arg),&width_arg)!=CL_SUCCESS||
       clSetKernelArg(b->logits_kernel,4,sizeof(vocab_arg),&vocab_arg)!=CL_SUCCESS)return 0;
    size_t global=b->vocabulary;
    if(clEnqueueNDRangeKernel(b->queue,b->logits_kernel,1,NULL,&global,NULL,0,NULL,NULL)!=CL_SUCCESS)return 0;
    if(s&&(clEnqueueReadBuffer(b->queue,b->fast,CL_TRUE,0,b->width*sizeof(float),s->fast,0,NULL,NULL)!=CL_SUCCESS||
       clEnqueueReadBuffer(b->queue,b->slow,CL_TRUE,0,b->width*sizeof(float),s->slow,0,NULL,NULL)!=CL_SUCCESS))return 0;
    if(clEnqueueReadBuffer(b->queue,b->logits_buffer,CL_TRUE,0,b->vocabulary*sizeof(float),logits,0,NULL,NULL)!=CL_SUCCESS)return 0;
    if(s)s->steps+=executed;
    return 1;
}
int ama10_opencl_step(AMA10OpenCL *b,AMA10State *s,uint32_t token,float *logits,unsigned max_depth,float threshold){return opencl_run(b,s,token,logits,max_depth,threshold);}
int ama10_opencl_reset(AMA10OpenCL *b){if(!b)return 0;float *zero=(float*)calloc(b->width,sizeof(float));if(!zero)return 0;int ok=clEnqueueWriteBuffer(b->queue,b->fast,CL_TRUE,0,b->width*sizeof(float),zero,0,NULL,NULL)==CL_SUCCESS&&clEnqueueWriteBuffer(b->queue,b->slow,CL_TRUE,0,b->width*sizeof(float),zero,0,NULL,NULL)==CL_SUCCESS;free(zero);b->stream_ready=ok;return ok;}
int ama10_opencl_step_stream(AMA10OpenCL *b,uint32_t token,float *logits,unsigned max_depth,float threshold){return opencl_run(b,NULL,token,logits,max_depth,threshold);}

static void model_parameter_views(AMA10Model *m,float **params,size_t *sizes){
    size_t w=m->width,e=m->input_width,v=m->vocabulary;
    params[0]=m->wx;sizes[0]=w*e;params[1]=m->wh;sizes[1]=w*w;params[2]=m->wm;sizes[2]=w*w;
    params[3]=m->wu;sizes[3]=w*w;params[4]=m->bias;sizes[4]=w;params[5]=m->gate_bias;sizes[5]=w;
    params[6]=m->embed;sizes[6]=e*v;params[7]=m->head;sizes[7]=v*w;
}
static void optimizer_parameter_views(AMA10Optimizer *o,float **first,float **second){
    for(size_t k=0;k<8;k++){first[k]=o->first_moment[k];second[k]=o->second_moment[k];}
}
static void release_adamw_state(AMA10OpenCL *b){
    cl_mem *buffers[]={&b->adamw_p,&b->adamw_a,&b->adamw_v,&b->adamw_next_p,&b->adamw_next_a,&b->adamw_next_v,&b->adamw_gradient};
    for(size_t i=0;i<7;i++){if(*buffers[i])clReleaseMemObject(*buffers[i]);*buffers[i]=NULL;}
    free(b->adamw_model_shadow);free(b->adamw_first_shadow);free(b->adamw_second_shadow);
    b->adamw_model_shadow=b->adamw_first_shadow=b->adamw_second_shadow=NULL;
    b->adamw_total=0;b->adamw_step=0;b->adamw_model=NULL;b->adamw_optimizer=NULL;b->adamw_ready=0;
}
static int prepare_adamw_state(AMA10OpenCL *b,AMA10Model *m,AMA10Optimizer *o,
                               float **params,float **first,float **second,const size_t *sizes,
                               const size_t *offsets,size_t total){
    size_t bytes=total*sizeof(float);float *flat_p=(float*)malloc(bytes),*flat_a=(float*)malloc(bytes),*flat_v=(float*)malloc(bytes);
    if(!flat_p||!flat_a||!flat_v){free(flat_p);free(flat_a);free(flat_v);return 0;}
    for(size_t k=0;k<8;k++){memcpy(flat_p+offsets[k],params[k],sizes[k]*sizeof(float));memcpy(flat_a+offsets[k],first[k],sizes[k]*sizeof(float));memcpy(flat_v+offsets[k],second[k],sizes[k]*sizeof(float));}
    int reusable=b->adamw_ready&&b->adamw_model==m&&b->adamw_optimizer==o&&b->adamw_total==total&&b->adamw_step==o->step&&
        memcmp(flat_p,b->adamw_model_shadow,bytes)==0&&memcmp(flat_a,b->adamw_first_shadow,bytes)==0&&memcmp(flat_v,b->adamw_second_shadow,bytes)==0;
    if(reusable){free(flat_p);free(flat_a);free(flat_v);return 1;}
    release_adamw_state(b);b->adamw_model_shadow=(float*)malloc(bytes);b->adamw_first_shadow=(float*)malloc(bytes);b->adamw_second_shadow=(float*)malloc(bytes);
    if(!b->adamw_model_shadow||!b->adamw_first_shadow||!b->adamw_second_shadow)goto fail;
    cl_int error=CL_SUCCESS;
    b->adamw_p=clCreateBuffer(b->context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,bytes,flat_p,&error);
    if(error==CL_SUCCESS)b->adamw_a=clCreateBuffer(b->context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,bytes,flat_a,&error);
    if(error==CL_SUCCESS)b->adamw_v=clCreateBuffer(b->context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,bytes,flat_v,&error);
    if(error==CL_SUCCESS)b->adamw_gradient=clCreateBuffer(b->context,CL_MEM_READ_ONLY,bytes,NULL,&error);
    if(error==CL_SUCCESS)b->adamw_next_p=clCreateBuffer(b->context,CL_MEM_WRITE_ONLY,bytes,NULL,&error);
    if(error==CL_SUCCESS)b->adamw_next_a=clCreateBuffer(b->context,CL_MEM_WRITE_ONLY,bytes,NULL,&error);
    if(error==CL_SUCCESS)b->adamw_next_v=clCreateBuffer(b->context,CL_MEM_WRITE_ONLY,bytes,NULL,&error);
    if(error!=CL_SUCCESS)goto fail;
    memcpy(b->adamw_model_shadow,flat_p,bytes);memcpy(b->adamw_first_shadow,flat_a,bytes);memcpy(b->adamw_second_shadow,flat_v,bytes);
    b->adamw_total=total;b->adamw_step=o->step;b->adamw_model=m;b->adamw_optimizer=o;b->adamw_ready=1;
    free(flat_p);free(flat_a);free(flat_v);return 1;
fail:release_adamw_state(b);free(flat_p);free(flat_a);free(flat_v);return 0;
}
int ama10_opencl_adamw_apply(AMA10OpenCL *b,AMA10Model *m,AMA10Optimizer *o,AMA10Gradients *g,float lr,float wd){
    if(!b||!m||!o||!g||!g->values||!g->tokens||m->width!=b->width||m->input_width!=b->input_width||m->vocabulary!=b->vocabulary||
       g->width!=m->width||g->input_width!=m->input_width||g->vocabulary!=m->vocabulary||g->parameter_count!=8||
       o->algorithm!=AMA10_OPT_ADAMW||o->parameter_count!=8||!o->first_moment||!o->second_moment||
       !isfinite(lr)||lr<=0.0f||!isfinite(wd)||wd<0.0f||o->step==UINT64_MAX)return 0;
    float *params[8],*first[8],*second[8];size_t sizes[8];model_parameter_views(m,params,sizes);optimizer_parameter_views(o,first,second);size_t total=0,offsets[8];
    for(size_t k=0;k<8;k++){if(sizes[k]>SIZE_MAX-total)return 0;offsets[k]=total;total+=sizes[k];}
    if(!total||total>UINT32_MAX||total>SIZE_MAX/sizeof(float))return 0;
    double norm2=0.0;for(size_t k=0;k<8;k++)if(!g->values[k]||!params[k]||!o->first_moment[k]||!o->second_moment[k])return 0;
    for(size_t k=0;k<8;k++)for(size_t i=0;i<sizes[k];i++){double grad=(double)g->values[k][i]/(double)g->tokens;if(!isfinite(grad))return 0;norm2+=grad*grad;}
    if(!isfinite(norm2))return 0;
    float scale=norm2>1.0?(float)(1.0/sqrt(norm2)):1.0f;
    uint64_t next_step=o->step+1;float corr1=(float)(1.0-pow(0.9,(double)next_step)),corr2=(float)(1.0-pow(0.999,(double)next_step));
    if(!(corr1>0.0f&&corr2>0.0f))return 0;
    size_t bytes=total*sizeof(float);float *flat_g=(float*)malloc(bytes),*flat_p=(float*)malloc(bytes);
    if(!flat_g||!flat_p){free(flat_g);free(flat_p);return 0;}
    if(!prepare_adamw_state(b,m,o,params,first,second,sizes,offsets,total)){free(flat_g);free(flat_p);return 0;}
    for(size_t k=0;k<8;k++)for(size_t i=0;i<sizes[k];i++)flat_g[offsets[k]+i]=(float)((double)g->values[k][i]/(double)g->tokens)*scale;
    cl_int error=clEnqueueWriteBuffer(b->queue,b->adamw_gradient,CL_TRUE,0,bytes,flat_g,0,NULL,NULL);
    cl_mem args[]={b->adamw_p,b->adamw_a,b->adamw_v,b->adamw_gradient,b->adamw_next_p,b->adamw_next_a,b->adamw_next_v};
    for(cl_uint j=0;j<7&&error==CL_SUCCESS;j++)error=clSetKernelArg(b->adamw_update,j,sizeof(cl_mem),&args[j]);
    cl_uint count=(cl_uint)total;if(error==CL_SUCCESS)error=clSetKernelArg(b->adamw_update,7,sizeof(count),&count);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->adamw_update,8,sizeof(lr),&lr);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->adamw_update,9,sizeof(wd),&wd);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->adamw_update,10,sizeof(corr1),&corr1);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->adamw_update,11,sizeof(corr2),&corr2);
    size_t global=total;if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->adamw_update,1,NULL,&global,NULL,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,b->adamw_next_p,CL_TRUE,0,bytes,flat_p,0,NULL,NULL);
    int ok=error==CL_SUCCESS;
    if(ok)for(size_t i=0;i<total;i++)if(!isfinite(flat_p[i])){ok=0;break;}
    if(ok){cl_mem swap=b->adamw_p;b->adamw_p=b->adamw_next_p;b->adamw_next_p=swap;swap=b->adamw_a;b->adamw_a=b->adamw_next_a;b->adamw_next_a=swap;swap=b->adamw_v;b->adamw_v=b->adamw_next_v;b->adamw_next_v=swap;
        for(size_t k=0;k<8;k++)memcpy(params[k],flat_p+offsets[k],sizes[k]*sizeof(float));
        for(size_t k=0;k<8&&error==CL_SUCCESS;k++)error=clEnqueueWriteBuffer(b->queue,b->parameters[k],CL_TRUE,0,sizes[k]*sizeof(float),params[k],0,NULL,NULL);
        if(error!=CL_SUCCESS)ok=0;
        memcpy(b->adamw_model_shadow,flat_p,bytes);
        b->adamw_step=next_step;o->step=next_step;ama10_gradients_reset(g);}
    free(flat_g);free(flat_p);return ok;
}
int ama10_opencl_adamw_sync(AMA10OpenCL *b,AMA10Model *m,AMA10Optimizer *o){
    if(!b||!m||!o||!b->adamw_ready||b->adamw_model!=m||b->adamw_optimizer!=o||b->adamw_step!=o->step)return b&&m&&o&&!b->adamw_ready;
    float *params[8],*first[8],*second[8];size_t sizes[8],offsets[8],total=0;model_parameter_views(m,params,sizes);optimizer_parameter_views(o,first,second);
    for(size_t k=0;k<8;k++){if(sizes[k]>SIZE_MAX-total)return 0;offsets[k]=total;total+=sizes[k];}
    if(total!=b->adamw_total)return 0;
    size_t bytes=total*sizeof(float);float *a=(float*)malloc(bytes),*v=(float*)malloc(bytes);if(!a||!v){free(a);free(v);return 0;}
    cl_int e=clEnqueueReadBuffer(b->queue,b->adamw_a,CL_TRUE,0,bytes,a,0,NULL,NULL);
    if(e==CL_SUCCESS)e=clEnqueueReadBuffer(b->queue,b->adamw_v,CL_TRUE,0,bytes,v,0,NULL,NULL);
    int ok=e==CL_SUCCESS;if(ok){for(size_t k=0;k<8;k++){memcpy(first[k],a+offsets[k],sizes[k]*sizeof(float));memcpy(second[k],v+offsets[k],sizes[k]*sizeof(float));}memcpy(b->adamw_first_shadow,a,bytes);memcpy(b->adamw_second_shadow,v,bytes);}
    free(a);free(v);return ok;
}
static void release_buffers(cl_mem *buffers,size_t count){for(size_t i=0;i<count;i++)if(buffers[i])clReleaseMemObject(buffers[i]);}
int ama10_opencl_head_forward(AMA10OpenCL *b,const float *head,const float *hidden,size_t length,float *logits){
    if(!b||!head||!hidden||!logits||!length||length>SIZE_MAX/b->width||length*b->width>SIZE_MAX/sizeof(float)||
       length>SIZE_MAX/b->vocabulary||length*b->vocabulary>SIZE_MAX/sizeof(float)||b->vocabulary>SIZE_MAX/b->width)return 0;
    size_t hn=length*b->width,ln=length*b->vocabulary,headn=b->vocabulary*b->width;
    size_t hb=hn*sizeof(float),lb=ln*sizeof(float),wb=headn*sizeof(float);cl_int error=CL_SUCCESS;
    cl_mem bufs[3]={0};bufs[0]=clCreateBuffer(b->context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,wb,(void*)head,&error);
    if(error==CL_SUCCESS)bufs[1]=clCreateBuffer(b->context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,hb,(void*)hidden,&error);
    if(error==CL_SUCCESS)bufs[2]=clCreateBuffer(b->context,CL_MEM_WRITE_ONLY,lb,NULL,&error);
    if(error==CL_SUCCESS){for(cl_uint i=0;i<3&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->sequence_logits,i,sizeof(cl_mem),&bufs[i]);
        cl_uint len=(cl_uint)length,w=(cl_uint)b->width,v=(cl_uint)b->vocabulary;
        if(length>UINT32_MAX)error=CL_INVALID_VALUE;
        if(error==CL_SUCCESS)error=clSetKernelArg(b->sequence_logits,3,sizeof(len),&len);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->sequence_logits,4,sizeof(w),&w);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->sequence_logits,5,sizeof(v),&v);
        size_t global=ln;if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->sequence_logits,1,NULL,&global,NULL,0,NULL,NULL);
        if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,bufs[2],CL_TRUE,0,lb,logits,0,NULL,NULL);
    }
    release_buffers(bufs,3);return error==CL_SUCCESS;
}
int ama10_opencl_head_backward(AMA10OpenCL *b,const float *head,const float *hidden,const float *errors,size_t length,float *head_grad,float *hidden_grad){
    if(!b||!head||!hidden||!errors||!head_grad||!hidden_grad||!length||length>UINT32_MAX||
       length>SIZE_MAX/b->width||length*b->width>SIZE_MAX/sizeof(float)||length>SIZE_MAX/b->vocabulary||
       length*b->vocabulary>SIZE_MAX/sizeof(float)||b->vocabulary>SIZE_MAX/b->width)return 0;
    size_t hn=length*b->width,ln=length*b->vocabulary,headn=b->vocabulary*b->width;
    size_t hb=hn*sizeof(float),lb=ln*sizeof(float),wb=headn*sizeof(float);cl_int error=CL_SUCCESS;
    cl_mem bufs[5]={0};bufs[0]=clCreateBuffer(b->context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,wb,(void*)head,&error);
    if(error==CL_SUCCESS)bufs[1]=clCreateBuffer(b->context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,hb,(void*)hidden,&error);
    if(error==CL_SUCCESS)bufs[2]=clCreateBuffer(b->context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,lb,(void*)errors,&error);
    if(error==CL_SUCCESS)bufs[3]=clCreateBuffer(b->context,CL_MEM_WRITE_ONLY,wb,NULL,&error);
    if(error==CL_SUCCESS)bufs[4]=clCreateBuffer(b->context,CL_MEM_WRITE_ONLY,hb,NULL,&error);
    if(error==CL_SUCCESS){for(cl_uint i=0;i<3&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->head_gradient,i,sizeof(cl_mem),&bufs[i]);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->head_gradient,3,sizeof(cl_mem),&bufs[3]);
        cl_uint len=(cl_uint)length,w=(cl_uint)b->width,v=(cl_uint)b->vocabulary;
        if(error==CL_SUCCESS)error=clSetKernelArg(b->head_gradient,4,sizeof(len),&len);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->head_gradient,5,sizeof(w),&w);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->head_gradient,6,sizeof(v),&v);
        size_t global=headn;if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->head_gradient,1,NULL,&global,NULL,0,NULL,NULL);
        if(error==CL_SUCCESS){for(cl_uint i=0;i<3&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->hidden_gradient,i,sizeof(cl_mem),&bufs[i]);
            if(error==CL_SUCCESS)error=clSetKernelArg(b->hidden_gradient,3,sizeof(cl_mem),&bufs[4]);
            if(error==CL_SUCCESS)error=clSetKernelArg(b->hidden_gradient,4,sizeof(len),&len);
            if(error==CL_SUCCESS)error=clSetKernelArg(b->hidden_gradient,5,sizeof(w),&w);
            if(error==CL_SUCCESS)error=clSetKernelArg(b->hidden_gradient,6,sizeof(v),&v);
            global=hn;if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->hidden_gradient,1,NULL,&global,NULL,0,NULL,NULL);}
        if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,bufs[3],CL_TRUE,0,wb,head_grad,0,NULL,NULL);
        if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,bufs[4],CL_TRUE,0,hb,hidden_grad,0,NULL,NULL);
    }
    release_buffers(bufs,5);return error==CL_SUCCESS;
}

int ama10_opencl_train_forward(AMA10OpenCL *b,const AMA10Model *m,const uint32_t *tokens,
                               size_t length,unsigned depth,const float *initial_fast,
                               const float *initial_slow,float *sequence_hidden,float *final_slow){
    if(!b||!m||!tokens||!length||!depth||!initial_fast||!initial_slow||!sequence_hidden||!final_slow||
       length>UINT32_MAX||depth>UINT32_MAX||m->width!=b->width||m->input_width!=b->input_width||
       m->vocabulary!=b->vocabulary||length>SIZE_MAX/depth||length*depth>SIZE_MAX/8||
       length*depth*8>SIZE_MAX/b->width||length*depth*8*b->width>SIZE_MAX/sizeof(float)||
       length>SIZE_MAX/b->width||length*b->width>SIZE_MAX/sizeof(float))return 0;
    size_t cache_bytes=length*depth*8*b->width*sizeof(float),hidden_bytes=length*b->width*sizeof(float);
    size_t token_bytes=length*sizeof(uint32_t);cl_int error=CL_SUCCESS;
    if(cache_bytes>b->bptt_cache_capacity){if(b->bptt_cache)clReleaseMemObject(b->bptt_cache);
        b->bptt_cache=clCreateBuffer(b->context,CL_MEM_READ_WRITE,cache_bytes,NULL,&error);
        if(error!=CL_SUCCESS||!b->bptt_cache){b->bptt_cache_capacity=0;return 0;}b->bptt_cache_capacity=cache_bytes;}
    if(hidden_bytes>b->bptt_sequence_capacity){
        if(b->bptt_hidden_gradient)clReleaseMemObject(b->bptt_hidden_gradient);
        b->bptt_hidden_gradient=clCreateBuffer(b->context,CL_MEM_READ_WRITE,hidden_bytes,NULL,&error);
        if(error!=CL_SUCCESS||!b->bptt_hidden_gradient){b->bptt_sequence_capacity=0;return 0;}b->bptt_sequence_capacity=hidden_bytes;}
    if(token_bytes>b->bptt_token_capacity){if(b->bptt_tokens)clReleaseMemObject(b->bptt_tokens);
        b->bptt_tokens=clCreateBuffer(b->context,CL_MEM_READ_ONLY,token_bytes,NULL,&error);
        if(error!=CL_SUCCESS||!b->bptt_tokens){b->bptt_token_capacity=0;return 0;}b->bptt_token_capacity=token_bytes;}
    const float *params[]={m->wx,m->wh,m->wm,m->wu,m->bias,m->gate_bias,m->embed,m->head};
    size_t sizes[]={b->width*b->input_width,b->width*b->width,b->width*b->width,b->width*b->width,
        b->width,b->width,b->input_width*b->vocabulary,b->vocabulary*b->width};
    for(size_t i=0;i<8&&error==CL_SUCCESS;i++)error=clEnqueueWriteBuffer(b->queue,b->parameters[i],CL_TRUE,0,
        sizes[i]*sizeof(float),(void*)params[i],0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueWriteBuffer(b->queue,b->bptt_tokens,CL_TRUE,0,token_bytes,tokens,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueWriteBuffer(b->queue,b->bptt_direct_fast,CL_TRUE,0,
        b->width*sizeof(float),initial_fast,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueWriteBuffer(b->queue,b->bptt_direct_slow,CL_TRUE,0,
        b->width*sizeof(float),initial_slow,0,NULL,NULL);
    cl_uint width=(cl_uint)b->width,input_width=(cl_uint)b->input_width,vocabulary=(cl_uint)b->vocabulary;
    cl_uint depth_arg=(cl_uint)depth;float rho=0.995f;
    cl_mem args[]={b->parameters[0],b->parameters[1],b->parameters[2],b->parameters[3],
        b->parameters[4],b->parameters[5],b->parameters[6],b->bptt_tokens,b->bptt_direct_fast,
        b->bptt_direct_slow,b->bptt_cache,b->bptt_hidden_gradient};
    for(cl_uint i=0;i<12&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->training_forward_cell,i,sizeof(cl_mem),&args[i]);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->training_forward_cell,12,sizeof(width),&width);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->training_forward_cell,13,sizeof(input_width),&input_width);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->training_forward_cell,14,sizeof(vocabulary),&vocabulary);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->training_forward_cell,15,sizeof(depth_arg),&depth_arg);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->training_forward_cell,18,sizeof(rho),&rho);
    size_t global=b->width;
    for(size_t t=0;t<length&&error==CL_SUCCESS;t++)for(unsigned d=0;d<depth&&error==CL_SUCCESS;d++){
        cl_uint ti=(cl_uint)t,di=(cl_uint)d;
        error=clSetKernelArg(b->training_forward_cell,16,sizeof(ti),&ti);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->training_forward_cell,17,sizeof(di),&di);
        if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->training_forward_cell,1,NULL,&global,NULL,0,NULL,NULL);
    }
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,b->bptt_hidden_gradient,CL_TRUE,0,hidden_bytes,sequence_hidden,0,NULL,NULL);
    size_t final_slow_offset=((((length-1)*depth+(depth-1))*8+7)*b->width)*sizeof(float);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,b->bptt_cache,CL_TRUE,final_slow_offset,
        b->width*sizeof(float),final_slow,0,NULL,NULL);
    if(error!=CL_SUCCESS)return 0;
    b->bptt_depth=depth;b->bptt_forward_length=length;b->bptt_forward_depth=depth;b->bptt_forward_ready=1;
    return 1;
}

int ama10_opencl_bptt_begin(AMA10OpenCL *b,const AMA10Model *m,const float *trajectory,
                            const uint32_t *tokens,size_t length,unsigned depth,
                            const float *hidden_gradients){
    if(!b||!m||!tokens||!hidden_gradients||!b->bptt_ready||!length||!depth||
       length>UINT32_MAX||depth>UINT32_MAX||m->width!=b->width||m->input_width!=b->input_width||
       m->vocabulary!=b->vocabulary||length>SIZE_MAX/depth||length*depth>SIZE_MAX/8||
       length*depth*8>SIZE_MAX/b->width||length*depth*8*b->width>SIZE_MAX/sizeof(float)||
       length>SIZE_MAX/b->width||length*b->width>SIZE_MAX/sizeof(float))return 0;
    size_t cache_bytes=length*depth*8*b->width*sizeof(float),hidden_bytes=length*b->width*sizeof(float);
    int reuse_forward=b->bptt_forward_ready&&b->bptt_forward_length==length&&b->bptt_forward_depth==depth;
    if(!reuse_forward&&!trajectory)return 0;
    size_t token_bytes=length*sizeof(uint32_t);cl_int error=CL_SUCCESS;
    if(cache_bytes>b->bptt_cache_capacity){if(b->bptt_cache)clReleaseMemObject(b->bptt_cache);
        b->bptt_cache=clCreateBuffer(b->context,CL_MEM_READ_WRITE,cache_bytes,NULL,&error);
        if(error!=CL_SUCCESS||!b->bptt_cache){b->bptt_cache_capacity=0;return 0;}b->bptt_cache_capacity=cache_bytes;}
    if(hidden_bytes>b->bptt_sequence_capacity){
        if(b->bptt_hidden_gradient)clReleaseMemObject(b->bptt_hidden_gradient);
        b->bptt_hidden_gradient=clCreateBuffer(b->context,CL_MEM_READ_WRITE,hidden_bytes,NULL,&error);
        if(error!=CL_SUCCESS||!b->bptt_hidden_gradient){b->bptt_sequence_capacity=0;return 0;}
        b->bptt_sequence_capacity=hidden_bytes;}
    if(token_bytes>b->bptt_token_capacity){if(b->bptt_tokens)clReleaseMemObject(b->bptt_tokens);
        b->bptt_tokens=clCreateBuffer(b->context,CL_MEM_READ_ONLY,token_bytes,NULL,&error);
        if(error!=CL_SUCCESS||!b->bptt_tokens){b->bptt_token_capacity=0;return 0;}b->bptt_token_capacity=token_bytes;}
    const float *params[]={m->wx,m->wh,m->wm,m->wu,m->bias,m->gate_bias,m->embed,m->head};
    size_t sizes[]={b->width*b->input_width,b->width*b->width,b->width*b->width,b->width*b->width,
        b->width,b->width,b->input_width*b->vocabulary,b->vocabulary*b->width};
    if(!reuse_forward){
        for(size_t i=0;i<8&&error==CL_SUCCESS;i++)error=clEnqueueWriteBuffer(b->queue,b->parameters[i],CL_TRUE,0,
            sizes[i]*sizeof(float),(void*)params[i],0,NULL,NULL);
        if(error==CL_SUCCESS)error=clEnqueueWriteBuffer(b->queue,b->bptt_cache,CL_TRUE,0,cache_bytes,trajectory,0,NULL,NULL);
        if(error==CL_SUCCESS)error=clEnqueueWriteBuffer(b->queue,b->bptt_tokens,CL_TRUE,0,token_bytes,tokens,0,NULL,NULL);
    }
    if(error==CL_SUCCESS)error=clEnqueueWriteBuffer(b->queue,b->bptt_hidden_gradient,CL_TRUE,0,hidden_bytes,hidden_gradients,0,NULL,NULL);
    if(error!=CL_SUCCESS)return 0;
    b->bptt_depth=depth;
    const float zero=0.0f;error=CL_SUCCESS;
    cl_mem state[]={b->bptt_dh,b->bptt_dm};
    for(size_t i=0;i<2&&error==CL_SUCCESS;i++)error=clEnqueueFillBuffer(b->queue,state[i],&zero,sizeof(zero),0,b->width*sizeof(float),0,NULL,NULL);
    for(size_t i=0;i<8&&error==CL_SUCCESS;i++){
        size_t n=i==0?b->width*b->input_width:i<=3?b->width*b->width:
                  i<=5?b->width:i==6?b->input_width*b->vocabulary:b->vocabulary*b->width;
        error=clEnqueueFillBuffer(b->queue,b->bptt_gradients[i],&zero,sizeof(zero),0,n*sizeof(float),0,NULL,NULL);
    }
    if(error==CL_SUCCESS)b->bptt_forward_ready=0;
    return error==CL_SUCCESS;
}
int ama10_opencl_bptt_step(AMA10OpenCL *b,size_t time,unsigned layer,int add_hidden_gradient){
    if(!b||!b->bptt_ready||time>UINT32_MAX||layer>=b->bptt_depth)return 0;
    cl_int error=CL_SUCCESS;cl_uint width=(cl_uint)b->width,input_width=(cl_uint)b->input_width;
    cl_uint vocabulary=(cl_uint)b->vocabulary,depth=(cl_uint)b->bptt_depth,t=(cl_uint)time,d=(cl_uint)layer;
    cl_uint add_hidden=add_hidden_gradient?1u:0u;float rho=0.995f;
    cl_mem deriv_args[]={b->bptt_dh,b->bptt_dm,b->bptt_cache,b->bptt_hidden_gradient,
        b->bptt_dpre,b->bptt_dgate,b->bptt_direct_fast,b->bptt_direct_slow};
    for(cl_uint i=0;i<8&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->cell_backward_derivatives,i,sizeof(cl_mem),&deriv_args[i]);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->cell_backward_derivatives,8,sizeof(width),&width);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->cell_backward_derivatives,9,sizeof(depth),&depth);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->cell_backward_derivatives,10,sizeof(t),&t);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->cell_backward_derivatives,11,sizeof(d),&d);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->cell_backward_derivatives,12,sizeof(add_hidden),&add_hidden);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->cell_backward_derivatives,13,sizeof(rho),&rho);
    size_t global=b->width;if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->cell_backward_derivatives,1,NULL,&global,NULL,0,NULL,NULL);
    cl_mem accum_args[]={b->parameters[0],b->parameters[1],b->parameters[2],b->parameters[3],b->parameters[6],
        b->bptt_dpre,b->bptt_dgate,b->bptt_direct_fast,b->bptt_direct_slow,b->bptt_cache,b->bptt_tokens,
        b->bptt_next_dh,b->bptt_next_dm,b->bptt_gradients[0],b->bptt_gradients[1],b->bptt_gradients[2],
        b->bptt_gradients[3],b->bptt_gradients[4],b->bptt_gradients[5],b->bptt_gradients[6]};
    for(cl_uint i=0;i<20&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->cell_backward_accumulate,i,sizeof(cl_mem),&accum_args[i]);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->cell_backward_accumulate,20,sizeof(width),&width);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->cell_backward_accumulate,21,sizeof(input_width),&input_width);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->cell_backward_accumulate,22,sizeof(vocabulary),&vocabulary);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->cell_backward_accumulate,23,sizeof(depth),&depth);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->cell_backward_accumulate,24,sizeof(t),&t);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->cell_backward_accumulate,25,sizeof(d),&d);
    size_t ww=b->width*b->width,we=b->width*b->input_width;global=ww>we?ww:we;
    if(global<b->input_width)global=b->input_width;
    if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->cell_backward_accumulate,1,NULL,&global,NULL,0,NULL,NULL);
    cl_mem swap=b->bptt_dh;b->bptt_dh=b->bptt_next_dh;b->bptt_next_dh=swap;
    swap=b->bptt_dm;b->bptt_dm=b->bptt_next_dm;b->bptt_next_dm=swap;
    return error==CL_SUCCESS;
}
int ama10_opencl_bptt_finish(AMA10OpenCL *b,AMA10Gradients *g){
    if(!b||!g||!g->values||g->parameter_count!=8||g->width!=b->width||
       g->input_width!=b->input_width||g->vocabulary!=b->vocabulary)return 0;
    size_t sizes[7]={b->width*b->input_width,b->width*b->width,b->width*b->width,
        b->width*b->width,b->width,b->width,b->input_width*b->vocabulary};
    for(size_t i=0;i<7;i++)if(!g->values[i]||clEnqueueReadBuffer(b->queue,b->bptt_gradients[i],CL_TRUE,0,
        sizes[i]*sizeof(float),g->values[i],0,NULL,NULL)!=CL_SUCCESS)return 0;
    return 1;
}

const char *ama10_opencl_device_name(const AMA10OpenCL *b){return b?b->device_name:"";}
void ama10_opencl_destroy(AMA10OpenCL *b){
    if(!b)return;
    release_adamw_state(b);
    cl_mem bptt_buffers[]={b->bptt_dh,b->bptt_dm,b->bptt_next_dh,b->bptt_next_dm,b->bptt_dpre,b->bptt_dgate,
        b->bptt_direct_fast,b->bptt_direct_slow,b->bptt_cache,b->bptt_tokens,b->bptt_hidden_gradient};
    release_buffers(bptt_buffers,sizeof(bptt_buffers)/sizeof(bptt_buffers[0]));
    release_buffers(b->bptt_gradients,8);
    if(b->logits_buffer)clReleaseMemObject(b->logits_buffer);
    if(b->confidence_buffer)clReleaseMemObject(b->confidence_buffer);
    if(b->next_slow)clReleaseMemObject(b->next_slow);
    if(b->next_fast)clReleaseMemObject(b->next_fast);
    if(b->slow)clReleaseMemObject(b->slow);
    if(b->fast)clReleaseMemObject(b->fast);
    for(size_t i=0;i<8;i++)if(b->parameters[i])clReleaseMemObject(b->parameters[i]);
    if(b->logits_kernel)clReleaseKernel(b->logits_kernel);
    if(b->adamw_update)clReleaseKernel(b->adamw_update);
    if(b->hidden_gradient)clReleaseKernel(b->hidden_gradient);
    if(b->cell_backward_accumulate)clReleaseKernel(b->cell_backward_accumulate);
    if(b->cell_backward_derivatives)clReleaseKernel(b->cell_backward_derivatives);
    if(b->head_gradient)clReleaseKernel(b->head_gradient);
    if(b->sequence_logits)clReleaseKernel(b->sequence_logits);
    if(b->confidence)clReleaseKernel(b->confidence);
    if(b->cell)clReleaseKernel(b->cell);
    if(b->program)clReleaseProgram(b->program);
    if(b->queue)clReleaseCommandQueue(b->queue);
    if(b->context)clReleaseContext(b->context);
    free(b);
}
