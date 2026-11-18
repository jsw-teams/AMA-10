#define CL_TARGET_OPENCL_VERSION 120
#include "opencl_backend.h"
#include <CL/cl.h>
#include <float.h>
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
    cl_kernel cell,confidence,slow_update,logits_kernel,adamw_update,gradient_pair_stats,gradient_pair_finalize,gradient_pair_merge;
    cl_kernel sequence_logits,head_gradient,hidden_gradient;
    cl_kernel project_token,project_features,affine_update,state_context,reason_step;
    cl_kernel affine_project_sequence,affine_gate_sequence,affine_scan_compose;
    cl_kernel affine_reason_sequence;
    cl_kernel reason_backward_derivative,reason_backward_propagate,reason_backward_weight_gradient;
    cl_kernel affine_backward_deltas,affine_backward_z_gradient,affine_backward_recurrent_gradient;
    cl_kernel affine_backward_bias_gradient,affine_backward_input_gradient,affine_backward_input_weight_gradient;
    cl_kernel cell_backward_derivatives,cell_backward_accumulate,training_forward_cell,training_forward_fused;
    cl_mem parameters[8];
    cl_mem fast,slow,next_fast,next_slow,confidence_buffer,logits_buffer;
    cl_mem projected,state_context_buffer,reason_a,reason_b,input_buffer;
    cl_mem sequence_inputs,sequence_projected,sequence_gates,sequence_candidates;
    size_t sequence_capacity;
    cl_mem scan_a[2],scan_b[2];
    size_t scan_capacity;
    cl_mem reason_input,reason_states;
    size_t reason_input_capacity,reason_states_capacity;
    size_t width,input_width,vocabulary;
    size_t max_alloc_bytes;
    char device_name[256];
    const AMA10Model *parameter_model;
    int stream_ready;
    unsigned stream_steps;
    cl_mem adamw_p,adamw_a,adamw_v,adamw_next_p,adamw_next_a,adamw_next_v,adamw_gradient;
    cl_mem adamw_task_a,adamw_task_b,gradient_pair_partials,gradient_pair_stats_buffer;
    size_t gradient_pair_capacity;
    size_t adamw_total;
    uint64_t adamw_step;
    const AMA10Model *adamw_model;
    const AMA10Optimizer *adamw_optimizer;
    float *adamw_model_shadow,*adamw_first_shadow,*adamw_second_shadow;
    int adamw_ready;
    cl_mem bptt_dh,bptt_dm,bptt_next_dh,bptt_next_dm,bptt_dpre,bptt_dgate;
    cl_mem bptt_direct_fast,bptt_direct_slow,bptt_cache,bptt_tokens,bptt_hidden_gradient;
    cl_mem bptt_input_features,bptt_feature_mask,bptt_feature_gradients;
    cl_mem bptt_gradients[8],bptt_logits;
    size_t bptt_cache_capacity,bptt_sequence_capacity,bptt_token_capacity,bptt_feature_capacity,bptt_mask_capacity,bptt_logits_capacity,bptt_depth;
    size_t bptt_forward_length,bptt_forward_depth;
    int bptt_forward_ready;
    int bptt_ready;
};

static void release_buffers(cl_mem *buffers,size_t count);

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
    { cl_ulong max_alloc=0;
      if(clGetDeviceInfo(b->device,CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof(max_alloc),&max_alloc,NULL)!=CL_SUCCESS||max_alloc>SIZE_MAX)goto fail;
      b->max_alloc_bytes=(size_t)max_alloc;
    }
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
    b->slow_update=clCreateKernel(b->program,"ama10_slow_update",&error);
    if(error!=CL_SUCCESS||!b->slow_update)goto fail;
    b->logits_kernel=clCreateKernel(b->program,"ama10_recurrent_logits",&error);
    if(error!=CL_SUCCESS||!b->logits_kernel)goto fail;
    b->project_token=clCreateKernel(b->program,"ama10_project_token",&error);
    if(error!=CL_SUCCESS||!b->project_token)goto fail;
    b->project_features=clCreateKernel(b->program,"ama10_project_features",&error);
    if(error!=CL_SUCCESS||!b->project_features)goto fail;
    b->affine_update=clCreateKernel(b->program,"ama10_affine_state_update",&error);
    if(error!=CL_SUCCESS||!b->affine_update)goto fail;
    b->state_context=clCreateKernel(b->program,"ama10_state_context",&error);
    if(error!=CL_SUCCESS||!b->state_context)goto fail;
    b->reason_step=clCreateKernel(b->program,"ama10_reason_step",&error);
    if(error!=CL_SUCCESS||!b->reason_step)goto fail;
    b->affine_project_sequence=clCreateKernel(b->program,"ama10_affine_project_sequence",&error);
    if(error!=CL_SUCCESS||!b->affine_project_sequence)goto fail;
    b->affine_gate_sequence=clCreateKernel(b->program,"ama10_affine_gate_sequence",&error);
    if(error!=CL_SUCCESS||!b->affine_gate_sequence)goto fail;
    b->affine_scan_compose=clCreateKernel(b->program,"ama10_affine_scan_compose",&error);
    if(error!=CL_SUCCESS||!b->affine_scan_compose)goto fail;
    b->affine_reason_sequence=clCreateKernel(b->program,"ama10_affine_reason_sequence",&error);
    if(error!=CL_SUCCESS||!b->affine_reason_sequence)goto fail;
    b->reason_backward_derivative=clCreateKernel(b->program,"ama10_reason_backward_derivative",&error);
    if(error!=CL_SUCCESS||!b->reason_backward_derivative)goto fail;
    b->reason_backward_propagate=clCreateKernel(b->program,"ama10_reason_backward_propagate",&error);
    if(error!=CL_SUCCESS||!b->reason_backward_propagate)goto fail;
    b->reason_backward_weight_gradient=clCreateKernel(b->program,"ama10_reason_backward_weight_gradient",&error);
    if(error!=CL_SUCCESS||!b->reason_backward_weight_gradient)goto fail;
    b->affine_backward_deltas=clCreateKernel(b->program,"ama10_affine_backward_deltas",&error);
    if(error!=CL_SUCCESS||!b->affine_backward_deltas)goto fail;
    b->affine_backward_z_gradient=clCreateKernel(b->program,"ama10_affine_backward_z_gradient",&error);
    if(error!=CL_SUCCESS||!b->affine_backward_z_gradient)goto fail;
    b->affine_backward_recurrent_gradient=clCreateKernel(b->program,"ama10_affine_backward_recurrent_gradient",&error);
    if(error!=CL_SUCCESS||!b->affine_backward_recurrent_gradient)goto fail;
    b->affine_backward_bias_gradient=clCreateKernel(b->program,"ama10_affine_backward_bias_gradient",&error);
    if(error!=CL_SUCCESS||!b->affine_backward_bias_gradient)goto fail;
    b->affine_backward_input_gradient=clCreateKernel(b->program,"ama10_affine_backward_input_gradient",&error);
    if(error!=CL_SUCCESS||!b->affine_backward_input_gradient)goto fail;
    b->affine_backward_input_weight_gradient=clCreateKernel(b->program,"ama10_affine_backward_input_weight_gradient",&error);
    if(error!=CL_SUCCESS||!b->affine_backward_input_weight_gradient)goto fail;
    b->adamw_update=clCreateKernel(b->program,"ama10_adamw_update",&error);
    if(error!=CL_SUCCESS||!b->adamw_update)goto fail;
    b->gradient_pair_stats=clCreateKernel(b->program,"ama10_gradient_pair_stats",&error);
    if(error!=CL_SUCCESS||!b->gradient_pair_stats)goto fail;
    b->gradient_pair_finalize=clCreateKernel(b->program,"ama10_gradient_pair_finalize",&error);
    if(error!=CL_SUCCESS||!b->gradient_pair_finalize)goto fail;
    b->gradient_pair_merge=clCreateKernel(b->program,"ama10_gradient_pair_merge",&error);
    if(error!=CL_SUCCESS||!b->gradient_pair_merge)goto fail;
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
    b->training_forward_fused=clCreateKernel(b->program,"ama10_training_forward_fused",&error);
    if(error!=CL_SUCCESS||!b->training_forward_fused)goto fail;
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
    b->projected=clCreateBuffer(b->context,CL_MEM_READ_WRITE,m->width*sizeof(float),NULL,&error);
    if(error!=CL_SUCCESS||!b->projected)goto fail;
    b->state_context_buffer=clCreateBuffer(b->context,CL_MEM_READ_WRITE,m->width*sizeof(float),NULL,&error);
    if(error!=CL_SUCCESS||!b->state_context_buffer)goto fail;
    b->reason_a=clCreateBuffer(b->context,CL_MEM_READ_WRITE,m->width*sizeof(float),NULL,&error);
    if(error!=CL_SUCCESS||!b->reason_a)goto fail;
    b->reason_b=clCreateBuffer(b->context,CL_MEM_READ_WRITE,m->width*sizeof(float),NULL,&error);
    if(error!=CL_SUCCESS||!b->reason_b)goto fail;
    b->input_buffer=clCreateBuffer(b->context,CL_MEM_READ_ONLY,m->input_width*sizeof(float),NULL,&error);
    if(error!=CL_SUCCESS||!b->input_buffer)goto fail;
    {
        size_t w2=m->width*m->width,we=m->width*m->input_width,ev=m->input_width*m->vocabulary;
        cl_mem *buffers[]={&b->bptt_dh,&b->bptt_dm,&b->bptt_next_dh,&b->bptt_next_dm,
            &b->bptt_dpre,&b->bptt_dgate,&b->bptt_direct_fast,&b->bptt_direct_slow};
        for(size_t i=0;i<8;i++){*buffers[i]=make_float_buffer(b->context,m->width);if(!*buffers[i])goto fail;}
        size_t parameter_sizes[8]={we,w2,w2,w2,m->width,m->width,ev,m->vocabulary*m->width};
        for(size_t i=0;i<8;i++){b->bptt_gradients[i]=make_float_buffer(b->context,parameter_sizes[i]);if(!b->bptt_gradients[i])goto fail;}
        b->bptt_ready=1;
    }
    b->width=m->width;b->input_width=m->input_width;b->vocabulary=m->vocabulary;b->parameter_model=m;
    return b;
fail:
    free(platforms);ama10_opencl_destroy(b);fprintf(stderr,"OpenCL recurrent backend initialization failed (%d)\n",error);return NULL;
}

size_t ama10_opencl_recommend_width(size_t vocabulary,size_t sequence_length,
                                    unsigned depth,size_t memory_budget_bytes,
                                    size_t *selected_budget_bytes){
    if(!vocabulary||!sequence_length||!depth)return 0;
    cl_uint platform_count=0;if(clGetPlatformIDs(0,NULL,&platform_count)!=CL_SUCCESS||!platform_count)return 0;
    cl_platform_id *platforms=(cl_platform_id*)malloc((size_t)platform_count*sizeof(*platforms));
    if(!platforms)return 0;
    if(clGetPlatformIDs(platform_count,platforms,NULL)!=CL_SUCCESS){free(platforms);return 0;}
    cl_device_id device=NULL;
    for(cl_uint i=0;i<platform_count&&!device;i++)(void)clGetDeviceIDs(platforms[i],CL_DEVICE_TYPE_GPU,1,&device,NULL);
    free(platforms);if(!device)return 0;
    cl_ulong global_mem=0,max_alloc=0;
    if(clGetDeviceInfo(device,CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(global_mem),&global_mem,NULL)!=CL_SUCCESS||
       clGetDeviceInfo(device,CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof(max_alloc),&max_alloc,NULL)!=CL_SUCCESS||
       global_mem>SIZE_MAX||max_alloc>SIZE_MAX)return 0;
    size_t global=(size_t)global_mem,allocation=(size_t)max_alloc;
    if(!memory_budget_bytes)memory_budget_bytes=global-global/8;
    if(memory_budget_bytes>global)memory_budget_bytes=global;
    if(selected_budget_bytes)*selected_budget_bytes=memory_budget_bytes;
    const long double budget=(long double)memory_budget_bytes;
    const long double v=(long double)vocabulary,l=(long double)sequence_length,d=(long double)depth;
    size_t low=0,high=UINT32_MAX;
    while(low<high){
        size_t mid=low+(high-low)/2+(high-low)%2;
        long double w=(long double)mid;
        long double params=4.0L*w*w+2.0L*v*w+2.0L*w;
        long double param_bytes=params*4.0L;
        long double activation=(l*(d+28.0L)*w+2.0L*l*v+v*w)*4.0L;
        long double resident=params*36.0L+activation;
        int fits=mid>0&&w<=UINT32_MAX&&param_bytes<=(long double)allocation&&
            l*w*4.0L<=(long double)allocation&&l*v*4.0L<=(long double)allocation&&
            v*w*4.0L<=(long double)allocation&&resident<=budget;
        if(fits)low=mid;else high=mid-1;
    }
    return low;
}

int ama10_opencl_affine_scan(AMA10OpenCL *b,const float *a,const float *offset,
                             const float *initial,size_t length,float *states){
    if(!b||!a||!offset||!initial||!states||!length||length>UINT32_MAX||
       length>SIZE_MAX/b->width||length*b->width>SIZE_MAX/sizeof(float))return 0;
    const size_t n=length*b->width,bytes=n*sizeof(float);
    if(bytes>b->max_alloc_bytes)return 0;
    if(bytes>b->scan_capacity){
        cl_mem next_a[2]={0},next_b[2]={0};cl_int alloc_error=CL_SUCCESS;
        for(size_t i=0;i<2&&alloc_error==CL_SUCCESS;i++){
            next_a[i]=clCreateBuffer(b->context,CL_MEM_READ_WRITE,bytes,NULL,&alloc_error);
            if(alloc_error==CL_SUCCESS)next_b[i]=clCreateBuffer(b->context,CL_MEM_READ_WRITE,bytes,NULL,&alloc_error);
        }
        if(alloc_error!=CL_SUCCESS){release_buffers(next_a,2);release_buffers(next_b,2);return 0;}
        release_buffers(b->scan_a,2);release_buffers(b->scan_b,2);
        for(size_t i=0;i<2;i++){b->scan_a[i]=next_a[i];b->scan_b[i]=next_b[i];}
        b->scan_capacity=bytes;
    }
    cl_int error=clEnqueueWriteBuffer(b->queue,b->scan_a[0],CL_FALSE,0,bytes,(void*)a,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueWriteBuffer(b->queue,b->scan_b[0],CL_FALSE,0,bytes,(void*)offset,0,NULL,NULL);
    unsigned current=0;
    for(size_t step=1;step<length&&error==CL_SUCCESS;step*=2){
        cl_ulong batch=1,len=(cl_ulong)length,width=(cl_ulong)b->width,off=(cl_ulong)step;
        cl_mem args[]={b->scan_a[current],b->scan_b[current],b->scan_a[1-current],b->scan_b[1-current]};
        for(cl_uint i=0;i<4&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->affine_scan_compose,i,sizeof(cl_mem),&args[i]);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_scan_compose,4,sizeof(batch),&batch);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_scan_compose,5,sizeof(len),&len);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_scan_compose,6,sizeof(width),&width);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_scan_compose,7,sizeof(off),&off);
        size_t global=n;
        if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->affine_scan_compose,1,NULL,&global,NULL,0,NULL,NULL);
        current=1-current;
        if(step>SIZE_MAX/2)break;
    }
    float *prefix_a=(float*)malloc(bytes);
    float *prefix_b=(float*)malloc(bytes);
    if(!prefix_a||!prefix_b){free(prefix_a);free(prefix_b);return 0;}
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,b->scan_a[current],CL_FALSE,0,bytes,prefix_a,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,b->scan_b[current],CL_FALSE,0,bytes,prefix_b,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clFinish(b->queue);
    if(error==CL_SUCCESS)for(size_t t=0;t<length;t++)for(size_t i=0;i<b->width;i++){
        const size_t k=t*b->width+i;states[k]=prefix_a[k]*initial[i]+prefix_b[k];
    }
    free(prefix_a);free(prefix_b);return error==CL_SUCCESS;
}

static double opencl_distribution_entropy(const float *logits,size_t count){
    float maximum=logits[0];for(size_t i=1;i<count;i++)if(logits[i]>maximum)maximum=logits[i];
    double sum=0.0;for(size_t i=0;i<count;i++)sum+=exp((double)logits[i]-(double)maximum);
    if(!(sum>0.0)||!isfinite(sum))return INFINITY;
    double entropy=0.0;for(size_t i=0;i<count;i++){double p=exp((double)logits[i]-(double)maximum)/sum;if(p>0.0)entropy-=p*log(p);}
    return entropy;
}
static int opencl_run_input(AMA10OpenCL *b,AMA10State *s,uint32_t token,const float *features,float *logits,
                      unsigned max_depth,float threshold){
    if(!b||!logits||(!features&&token>=b->vocabulary)||!max_depth||!isfinite(threshold)||threshold<0.0f||threshold>2.0f||(!s&&!b->stream_ready))return 0;
    if(features)for(size_t j=0;j<b->input_width;j++)if(!isfinite(features[j]))return 0;
    if(s&&(!s->fast||!s->slow||s->width!=b->width||max_depth>UINT_MAX-s->steps))return 0;
    if(!s&&max_depth>UINT_MAX-b->stream_steps)return 0;
    cl_int error=CL_SUCCESS;cl_uint width=(cl_uint)b->width,input_width=(cl_uint)b->input_width;
    cl_uint vocabulary=(cl_uint)b->vocabulary,token_id=(cl_uint)token;
    if(s){
        error=clEnqueueWriteBuffer(b->queue,b->fast,CL_TRUE,0,b->width*sizeof(float),s->fast,0,NULL,NULL);
        if(error==CL_SUCCESS)error=clEnqueueWriteBuffer(b->queue,b->slow,CL_TRUE,0,b->width*sizeof(float),s->slow,0,NULL,NULL);
        if(error!=CL_SUCCESS)return 0;
    }
    cl_mem project_args[]={b->parameters[0],features?b->input_buffer:b->parameters[6],b->projected};
    cl_kernel project_kernel=features?b->project_features:b->project_token;
    cl_uint project_arg_count=features?3:3;
    for(cl_uint i=0;i<project_arg_count;i++)if(clSetKernelArg(project_kernel,i,sizeof(cl_mem),&project_args[i])!=CL_SUCCESS)return 0;
    if(features){
        if(clEnqueueWriteBuffer(b->queue,b->input_buffer,CL_TRUE,0,b->input_width*sizeof(float),(void*)features,0,NULL,NULL)!=CL_SUCCESS||
           clSetKernelArg(project_kernel,3,sizeof(input_width),&input_width)!=CL_SUCCESS||
           clSetKernelArg(project_kernel,4,sizeof(width),&width)!=CL_SUCCESS)return 0;
    }else if(clSetKernelArg(project_kernel,3,sizeof(input_width),&input_width)!=CL_SUCCESS||
       clSetKernelArg(project_kernel,4,sizeof(vocabulary),&vocabulary)!=CL_SUCCESS||
       clSetKernelArg(project_kernel,5,sizeof(token_id),&token_id)!=CL_SUCCESS||
       clSetKernelArg(project_kernel,6,sizeof(width),&width)!=CL_SUCCESS)return 0;
    size_t global=b->width;
    if(clEnqueueNDRangeKernel(b->queue,project_kernel,1,NULL,&global,NULL,0,NULL,NULL)!=CL_SUCCESS)return 0;
    cl_mem state_args[]={b->parameters[1],b->parameters[4],b->parameters[5],b->projected,
        b->fast,b->slow,b->next_fast,b->next_slow};
    for(cl_uint i=0;i<8;i++)if(clSetKernelArg(b->affine_update,i,sizeof(cl_mem),&state_args[i])!=CL_SUCCESS)return 0;
    if(clSetKernelArg(b->affine_update,8,sizeof(width),&width)!=CL_SUCCESS||
       clEnqueueNDRangeKernel(b->queue,b->affine_update,1,NULL,&global,NULL,0,NULL,NULL)!=CL_SUCCESS)return 0;
    cl_mem swap=b->fast;b->fast=b->next_fast;b->next_fast=swap;
    swap=b->slow;b->slow=b->next_slow;b->next_slow=swap;
    cl_mem context_args[]={b->fast,b->slow,b->state_context_buffer};
    for(cl_uint i=0;i<3;i++)if(clSetKernelArg(b->state_context,i,sizeof(cl_mem),&context_args[i])!=CL_SUCCESS)return 0;
    if(clSetKernelArg(b->state_context,3,sizeof(width),&width)!=CL_SUCCESS||
       clEnqueueNDRangeKernel(b->queue,b->state_context,1,NULL,&global,NULL,0,NULL,NULL)!=CL_SUCCESS||
       clEnqueueCopyBuffer(b->queue,b->state_context_buffer,b->reason_a,0,0,b->width*sizeof(float),0,NULL,NULL)!=CL_SUCCESS)return 0;
    cl_mem current=b->reason_a,following=b->reason_b;double previous_entropy=INFINITY;unsigned executed=0;
    for(unsigned d=0;d<max_depth;d++){
        cl_mem reason_args[]={b->parameters[3],b->parameters[2],current,b->state_context_buffer,following};
        for(cl_uint i=0;i<5;i++)if(clSetKernelArg(b->reason_step,i,sizeof(cl_mem),&reason_args[i])!=CL_SUCCESS)return 0;
        if(clSetKernelArg(b->reason_step,5,sizeof(width),&width)!=CL_SUCCESS||
           clEnqueueNDRangeKernel(b->queue,b->reason_step,1,NULL,&global,NULL,0,NULL,NULL)!=CL_SUCCESS)return 0;
        cl_mem output_args[]={b->parameters[7],following,b->logits_buffer};
        for(cl_uint i=0;i<3;i++)if(clSetKernelArg(b->logits_kernel,i,sizeof(cl_mem),&output_args[i])!=CL_SUCCESS)return 0;
        if(clSetKernelArg(b->logits_kernel,3,sizeof(width),&width)!=CL_SUCCESS||
           clSetKernelArg(b->logits_kernel,4,sizeof(vocabulary),&vocabulary)!=CL_SUCCESS)return 0;
        size_t output_global=b->vocabulary;
        if(clEnqueueNDRangeKernel(b->queue,b->logits_kernel,1,NULL,&output_global,NULL,0,NULL,NULL)!=CL_SUCCESS||
           clEnqueueReadBuffer(b->queue,b->logits_buffer,CL_TRUE,0,b->vocabulary*sizeof(float),logits,0,NULL,NULL)!=CL_SUCCESS)return 0;
        double entropy=opencl_distribution_entropy(logits,b->vocabulary);executed=d+1;
        if(d+1>=2&&previous_entropy-entropy<=threshold)break;
        previous_entropy=entropy;swap=current;current=following;following=swap;
    }
    if(s&&(clEnqueueReadBuffer(b->queue,b->fast,CL_TRUE,0,b->width*sizeof(float),s->fast,0,NULL,NULL)!=CL_SUCCESS||
       clEnqueueReadBuffer(b->queue,b->slow,CL_TRUE,0,b->width*sizeof(float),s->slow,0,NULL,NULL)!=CL_SUCCESS))return 0;
    if(s)s->steps+=executed;else b->stream_steps+=executed;
    return 1;
}
static int opencl_run(AMA10OpenCL *b,AMA10State *s,uint32_t token,float *logits,unsigned max_depth,float threshold){
    return opencl_run_input(b,s,token,NULL,logits,max_depth,threshold);
}
int ama10_opencl_step(AMA10OpenCL *b,AMA10State *s,uint32_t token,float *logits,unsigned max_depth,float threshold){
    return opencl_run(b,s,token,logits,max_depth,threshold);
}
int ama10_opencl_reset(AMA10OpenCL *b){if(!b)return 0;float *zero=(float*)calloc(b->width,sizeof(float));if(!zero)return 0;int ok=clEnqueueWriteBuffer(b->queue,b->fast,CL_TRUE,0,b->width*sizeof(float),zero,0,NULL,NULL)==CL_SUCCESS&&clEnqueueWriteBuffer(b->queue,b->slow,CL_TRUE,0,b->width*sizeof(float),zero,0,NULL,NULL)==CL_SUCCESS;free(zero);b->stream_ready=ok;b->stream_steps=0;return ok;}
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
    cl_mem *buffers[]={&b->adamw_p,&b->adamw_a,&b->adamw_v,&b->adamw_next_p,&b->adamw_next_a,&b->adamw_next_v,&b->adamw_gradient,
        &b->adamw_task_a,&b->adamw_task_b,&b->gradient_pair_partials,&b->gradient_pair_stats_buffer};
    for(size_t i=0;i<11;i++){if(*buffers[i])clReleaseMemObject(*buffers[i]);*buffers[i]=NULL;}
    b->gradient_pair_capacity=0;
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
    if(error==CL_SUCCESS)b->adamw_gradient=clCreateBuffer(b->context,CL_MEM_READ_WRITE,bytes,NULL,&error);
    if(error==CL_SUCCESS)b->adamw_task_a=clCreateBuffer(b->context,CL_MEM_READ_ONLY,bytes,NULL,&error);
    if(error==CL_SUCCESS)b->adamw_task_b=clCreateBuffer(b->context,CL_MEM_READ_ONLY,bytes,NULL,&error);
    if(error==CL_SUCCESS)b->gradient_pair_stats_buffer=clCreateBuffer(b->context,CL_MEM_READ_WRITE,5*sizeof(float),NULL,&error);
    if(error==CL_SUCCESS)b->adamw_next_p=clCreateBuffer(b->context,CL_MEM_WRITE_ONLY,bytes,NULL,&error);
    if(error==CL_SUCCESS)b->adamw_next_a=clCreateBuffer(b->context,CL_MEM_WRITE_ONLY,bytes,NULL,&error);
    if(error==CL_SUCCESS)b->adamw_next_v=clCreateBuffer(b->context,CL_MEM_WRITE_ONLY,bytes,NULL,&error);
    if(error!=CL_SUCCESS)goto fail;
    memcpy(b->adamw_model_shadow,flat_p,bytes);memcpy(b->adamw_first_shadow,flat_a,bytes);memcpy(b->adamw_second_shadow,flat_v,bytes);
    b->adamw_total=total;b->adamw_step=o->step;b->adamw_model=m;b->adamw_optimizer=o;b->adamw_ready=1;
    free(flat_p);free(flat_a);free(flat_v);return 1;
fail:release_adamw_state(b);free(flat_p);free(flat_a);free(flat_v);return 0;
}
static int ensure_gradient_pair_capacity(AMA10OpenCL *b,size_t groups){
    if(!b||!groups||groups>SIZE_MAX/(3u*sizeof(float)))return 0;
    if(groups<=b->gradient_pair_capacity)return 1;
    const size_t bytes=groups*3u*sizeof(float);
    if(bytes>b->max_alloc_bytes)return 0;
    cl_int error=CL_SUCCESS;cl_mem next=clCreateBuffer(b->context,CL_MEM_READ_WRITE,bytes,NULL,&error);
    if(error!=CL_SUCCESS||!next)return 0;
    if(b->gradient_pair_partials)clReleaseMemObject(b->gradient_pair_partials);
    b->gradient_pair_partials=next;b->gradient_pair_capacity=groups;return 1;
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
    /* flat_g is already clipped on the host in this single-task path. */
    const float device_scale=1.0f;
    if(error==CL_SUCCESS)error=clSetKernelArg(b->adamw_update,12,sizeof(device_scale),&device_scale);
    size_t global=total;if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->adamw_update,1,NULL,&global,NULL,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,b->adamw_next_p,CL_TRUE,0,bytes,flat_p,0,NULL,NULL);
    int ok=error==CL_SUCCESS;
    if(ok)for(size_t i=0;i<total;i++)if(!isfinite(flat_p[i])){ok=0;break;}
    if(ok){cl_mem swap=b->adamw_p;b->adamw_p=b->adamw_next_p;b->adamw_next_p=swap;swap=b->adamw_a;b->adamw_a=b->adamw_next_a;b->adamw_next_a=swap;swap=b->adamw_v;b->adamw_v=b->adamw_next_v;b->adamw_next_v=swap;
        for(size_t k=0;k<8;k++)memcpy(params[k],flat_p+offsets[k],sizes[k]*sizeof(float));
        for(size_t k=0;k<8&&error==CL_SUCCESS;k++)error=clEnqueueCopyBuffer(b->queue,b->adamw_p,b->parameters[k],
            offsets[k]*sizeof(float),0,sizes[k]*sizeof(float),0,NULL,NULL);
        if(error!=CL_SUCCESS)ok=0;
        memcpy(b->adamw_model_shadow,flat_p,bytes);
        b->adamw_step=next_step;o->step=next_step;b->parameter_model=m;ama10_gradients_reset(g);}
    free(flat_g);free(flat_p);return ok;
}
int ama10_opencl_adamw_apply_tasks(AMA10OpenCL *b,AMA10Model *m,AMA10Optimizer *o,
                                   AMA10Gradients *task_a,AMA10Gradients *task_b,
                                   float lr,float wd,double *cosine_out,double *weight_a_out){
    if(!b||!m||!o||!task_a||!task_b||!task_a->values||!task_b->values||
       !task_a->tokens||!task_b->tokens||task_a==task_b||
       m->width!=b->width||m->input_width!=b->input_width||m->vocabulary!=b->vocabulary||
       task_a->width!=m->width||task_b->width!=m->width||
       task_a->input_width!=m->input_width||task_b->input_width!=m->input_width||
       task_a->vocabulary!=m->vocabulary||task_b->vocabulary!=m->vocabulary||
       task_a->parameter_count!=8||task_b->parameter_count!=8||
       task_a->tokens>UINT64_MAX-task_b->tokens||o->algorithm!=AMA10_OPT_ADAMW||
       o->parameter_count!=8||!o->first_moment||!o->second_moment||
       !isfinite(lr)||lr<=0.0f||!isfinite(wd)||wd<0.0f||o->step==UINT64_MAX)return 0;
    float *params[8],*first[8],*second[8];size_t sizes[8],offsets[8],total=0;
    model_parameter_views(m,params,sizes);optimizer_parameter_views(o,first,second);
    for(size_t k=0;k<8;k++){
        if(!task_a->values[k]||!task_b->values[k]||!params[k]||!first[k]||!second[k]||sizes[k]>SIZE_MAX-total)return 0;
        offsets[k]=total;total+=sizes[k];
    }
    if(!total||total>UINT32_MAX||total>SIZE_MAX/sizeof(float)||total>SIZE_MAX-255u)return 0;
    const size_t groups=(total+255u)/256u;
    const size_t bytes=total*sizeof(float);
    float *flat_a=(float*)malloc(bytes),*flat_b=(float*)malloc(bytes),*flat_p=(float*)malloc(bytes);
    float *norm_partials=(float*)malloc(groups*sizeof(float));
    if(!flat_a||!flat_b||!flat_p||!norm_partials){free(flat_a);free(flat_b);free(flat_p);free(norm_partials);return 0;}
    int valid=1;
    for(size_t k=0;k<8&&valid;k++)for(size_t i=0;i<sizes[k];i++){
        const double a=(double)task_a->values[k][i]/(double)task_a->tokens;
        const double c=(double)task_b->values[k][i]/(double)task_b->tokens;
        if(!isfinite(a)||!isfinite(c)||fabs(a)>FLT_MAX||fabs(c)>FLT_MAX){valid=0;break;}
        flat_a[offsets[k]+i]=(float)a;flat_b[offsets[k]+i]=(float)c;
    }
    if(!valid||!prepare_adamw_state(b,m,o,params,first,second,sizes,offsets,total)){
        free(flat_a);free(flat_b);free(flat_p);free(norm_partials);return 0;
    }
    if(!ensure_gradient_pair_capacity(b,groups)){
        free(flat_a);free(flat_b);free(flat_p);free(norm_partials);return 0;
    }
    cl_int error=clEnqueueWriteBuffer(b->queue,b->adamw_task_a,CL_TRUE,0,bytes,flat_a,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueWriteBuffer(b->queue,b->adamw_task_b,CL_TRUE,0,bytes,flat_b,0,NULL,NULL);
    cl_uint count=(cl_uint)total;size_t global=((total+255u)/256u)*256u,local=256u;
    cl_mem stats_args[]={b->adamw_task_a,b->adamw_task_b,b->gradient_pair_partials};
    for(cl_uint i=0;i<3&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->gradient_pair_stats,i,sizeof(cl_mem),&stats_args[i]);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->gradient_pair_stats,3,sizeof(count),&count);
    if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->gradient_pair_stats,1,NULL,&global,&local,0,NULL,NULL);
    cl_uint group_count=(cl_uint)groups;
    cl_mem finalize_args[]={b->gradient_pair_partials,b->gradient_pair_stats_buffer};
    for(cl_uint i=0;i<2&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->gradient_pair_finalize,i,sizeof(cl_mem),&finalize_args[i]);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->gradient_pair_finalize,2,sizeof(group_count),&group_count);
    size_t reduce_global=256u;
    if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->gradient_pair_finalize,1,NULL,&reduce_global,&reduce_global,0,NULL,NULL);
    cl_mem merge_args[]={b->adamw_task_a,b->adamw_task_b,b->gradient_pair_stats_buffer,b->adamw_gradient,b->gradient_pair_partials};
    for(cl_uint i=0;i<5&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->gradient_pair_merge,i,sizeof(cl_mem),&merge_args[i]);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->gradient_pair_merge,5,sizeof(count),&count);
    if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->gradient_pair_merge,1,NULL,&global,&local,0,NULL,NULL);
    float stats[5]={0};
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,b->gradient_pair_stats_buffer,CL_TRUE,0,sizeof(stats),stats,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,b->gradient_pair_partials,CL_TRUE,0,groups*sizeof(float),norm_partials,0,NULL,NULL);
    double norm2=0.0;
    if(error==CL_SUCCESS){
        for(size_t i=0;i<groups;i++)norm2+=(double)norm_partials[i];
        if(!isfinite(norm2)||!isfinite(stats[3])||!isfinite(stats[4])||norm2<0.0)error=CL_INVALID_VALUE;
    }
    const float scale=error==CL_SUCCESS&&norm2>1.0?(float)(1.0/sqrt(norm2)):1.0f;
    const uint64_t next_step=o->step+1;
    const float corr1=(float)(1.0-pow(0.9,(double)next_step)),corr2=(float)(1.0-pow(0.999,(double)next_step));
    if(error==CL_SUCCESS&&(!(corr1>0.0f)||!(corr2>0.0f)))error=CL_INVALID_VALUE;
    cl_mem update_args[]={b->adamw_p,b->adamw_a,b->adamw_v,b->adamw_gradient,b->adamw_next_p,b->adamw_next_a,b->adamw_next_v};
    for(cl_uint i=0;i<7&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->adamw_update,i,sizeof(cl_mem),&update_args[i]);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->adamw_update,7,sizeof(count),&count);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->adamw_update,8,sizeof(lr),&lr);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->adamw_update,9,sizeof(wd),&wd);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->adamw_update,10,sizeof(corr1),&corr1);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->adamw_update,11,sizeof(corr2),&corr2);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->adamw_update,12,sizeof(scale),&scale);
    if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->adamw_update,1,NULL,&global,NULL,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,b->adamw_next_p,CL_TRUE,0,bytes,flat_p,0,NULL,NULL);
    int ok=error==CL_SUCCESS;
    if(ok)for(size_t i=0;i<total;i++)if(!isfinite(flat_p[i])){ok=0;break;}
    if(ok){
        cl_mem swap=b->adamw_p;b->adamw_p=b->adamw_next_p;b->adamw_next_p=swap;
        swap=b->adamw_a;b->adamw_a=b->adamw_next_a;b->adamw_next_a=swap;
        swap=b->adamw_v;b->adamw_v=b->adamw_next_v;b->adamw_next_v=swap;
        for(size_t k=0;k<8;k++)memcpy(params[k],flat_p+offsets[k],sizes[k]*sizeof(float));
        for(size_t k=0;k<8&&error==CL_SUCCESS;k++)error=clEnqueueCopyBuffer(b->queue,b->adamw_p,b->parameters[k],
            offsets[k]*sizeof(float),0,sizes[k]*sizeof(float),0,NULL,NULL);
        if(error==CL_SUCCESS){
            memcpy(b->adamw_model_shadow,flat_p,bytes);b->adamw_step=next_step;o->step=next_step;
            b->parameter_model=m;ama10_gradients_reset(task_a);ama10_gradients_reset(task_b);
            if(cosine_out)*cosine_out=(double)stats[4];
            if(weight_a_out)*weight_a_out=(double)stats[3];
        }else ok=0;
    }
    free(flat_a);free(flat_b);free(flat_p);free(norm_partials);return ok;
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
static int ensure_sequence_capacity(AMA10OpenCL *b,size_t length){
    if(length<=b->sequence_capacity)return 1;
    if(length>SIZE_MAX/b->input_width||length*b->input_width>SIZE_MAX/sizeof(float)||
       length>SIZE_MAX/b->width||length*b->width>SIZE_MAX/sizeof(float))return 0;
    cl_int error=CL_SUCCESS;cl_mem next[4]={0};
    next[0]=clCreateBuffer(b->context,CL_MEM_READ_ONLY,length*b->input_width*sizeof(float),NULL,&error);
    if(error==CL_SUCCESS)next[1]=clCreateBuffer(b->context,CL_MEM_READ_WRITE,length*b->width*sizeof(float),NULL,&error);
    if(error==CL_SUCCESS)next[2]=clCreateBuffer(b->context,CL_MEM_READ_WRITE,length*b->width*sizeof(float),NULL,&error);
    if(error==CL_SUCCESS)next[3]=clCreateBuffer(b->context,CL_MEM_READ_WRITE,length*b->width*sizeof(float),NULL,&error);
    for(size_t i=0;i<4;i++)if(!next[i]){release_buffers(next,4);return 0;}
    cl_mem old[]={b->sequence_inputs,b->sequence_projected,b->sequence_gates,b->sequence_candidates};
    release_buffers(old,4);b->sequence_inputs=next[0];b->sequence_projected=next[1];
    b->sequence_gates=next[2];b->sequence_candidates=next[3];b->sequence_capacity=length;return 1;
}
int ama10_opencl_affine_inputs(AMA10OpenCL *b,const AMA10Model *m,const float *inputs,
                               size_t length,float *projected,float *gates,float *candidates){
    if(!b||!m||!inputs||!length||!projected||!gates||!candidates||length>UINT32_MAX||
       m->width!=b->width||m->input_width!=b->input_width||m->vocabulary!=b->vocabulary||
       length>SIZE_MAX/b->input_width||length*b->input_width>SIZE_MAX/sizeof(float)||
       length>SIZE_MAX/b->width||length*b->width>SIZE_MAX/sizeof(float))return 0;
    for(size_t i=0;i<length*b->input_width;i++)if(!isfinite(inputs[i]))return 0;
    if(!ensure_sequence_capacity(b,length))return 0;
    const float *params[]={m->wx,m->wh,m->bias,m->gate_bias};
    const size_t bytes[]={b->width*b->input_width*sizeof(float),b->width*b->width*sizeof(float),
                          b->width*sizeof(float),b->width*sizeof(float)};
    const size_t indexes[]={0,1,4,5};cl_int error=CL_SUCCESS;
    if(b->parameter_model!=m)for(size_t i=0;i<4&&error==CL_SUCCESS;i++)error=clEnqueueWriteBuffer(b->queue,b->parameters[indexes[i]],CL_FALSE,
        0,bytes[i],params[i],0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueWriteBuffer(b->queue,b->sequence_inputs,CL_FALSE,0,
        length*b->input_width*sizeof(float),(void*)inputs,0,NULL,NULL);
    cl_uint len=(cl_uint)length,w=(cl_uint)b->width,e=(cl_uint)b->input_width;
    cl_mem p_args[]={b->parameters[0],b->sequence_inputs,b->sequence_projected};
    for(cl_uint i=0;i<3&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->affine_project_sequence,i,sizeof(cl_mem),&p_args[i]);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_project_sequence,3,sizeof(len),&len);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_project_sequence,4,sizeof(e),&e);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_project_sequence,5,sizeof(w),&w);
    size_t global[2]={length*16u,((b->width+15u)/16u)*16u},local[2]={16u,16u};
    if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->affine_project_sequence,2,NULL,global,local,0,NULL,NULL);
    cl_mem g_args[]={b->parameters[1],b->parameters[4],b->parameters[5],b->sequence_projected,
                     b->sequence_gates,b->sequence_candidates};
    for(cl_uint i=0;i<6&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->affine_gate_sequence,i,sizeof(cl_mem),&g_args[i]);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_gate_sequence,6,sizeof(len),&len);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_gate_sequence,7,sizeof(w),&w);
    if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->affine_gate_sequence,2,NULL,global,local,0,NULL,NULL);
    size_t bytes_out=length*b->width*sizeof(float);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,b->sequence_projected,CL_FALSE,0,bytes_out,projected,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,b->sequence_gates,CL_FALSE,0,bytes_out,gates,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,b->sequence_candidates,CL_FALSE,0,bytes_out,candidates,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clFinish(b->queue);
    return error==CL_SUCCESS;
}
int ama10_opencl_reason_sequence(AMA10OpenCL *b,const AMA10Model *m,const float *context,
                                 size_t length,unsigned depth,float *reason_states){
    if(!b||!m||!context||!reason_states||!length||!depth||length>UINT32_MAX||depth>UINT32_MAX||
       m->width!=b->width||m->input_width!=b->input_width||m->vocabulary!=b->vocabulary||
       length>SIZE_MAX/b->width||length*b->width>SIZE_MAX/sizeof(float)||
       length>SIZE_MAX/depth||length*depth>SIZE_MAX/b->width||
       length*depth*b->width>SIZE_MAX/sizeof(float))return 0;
    size_t input_count=length*b->width,state_count=length*depth*b->width;
    for(size_t i=0;i<input_count;i++)if(!isfinite(context[i]))return 0;
    cl_int error=CL_SUCCESS;
    if(input_count*sizeof(float)>b->reason_input_capacity){
        cl_mem next=clCreateBuffer(b->context,CL_MEM_READ_ONLY,input_count*sizeof(float),NULL,&error);
        if(error!=CL_SUCCESS||!next)return 0;
        if(b->reason_input)clReleaseMemObject(b->reason_input);
        b->reason_input=next;
        b->reason_input_capacity=input_count*sizeof(float);
    }
    if(state_count*sizeof(float)>b->reason_states_capacity){
        cl_mem next=clCreateBuffer(b->context,CL_MEM_READ_WRITE,state_count*sizeof(float),NULL,&error);
        if(error!=CL_SUCCESS||!next)return 0;
        if(b->reason_states)clReleaseMemObject(b->reason_states);
        b->reason_states=next;
        b->reason_states_capacity=state_count*sizeof(float);
    }
    if(b->parameter_model!=m)error=clEnqueueWriteBuffer(b->queue,b->parameters[3],CL_FALSE,0,b->width*b->width*sizeof(float),m->wu,0,NULL,NULL);
    if(error==CL_SUCCESS&&b->parameter_model!=m)error=clEnqueueWriteBuffer(b->queue,b->parameters[2],CL_FALSE,0,b->width*b->width*sizeof(float),m->wm,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueWriteBuffer(b->queue,b->reason_input,CL_FALSE,0,input_count*sizeof(float),(void*)context,0,NULL,NULL);
    cl_mem args[]={b->parameters[3],b->parameters[2],b->reason_input,b->reason_states};
    for(cl_uint i=0;i<4&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->affine_reason_sequence,i,sizeof(cl_mem),&args[i]);
    cl_uint len=(cl_uint)length,w=(cl_uint)b->width,d=(cl_uint)depth,layer=0;
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_reason_sequence,4,sizeof(len),&len);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_reason_sequence,5,sizeof(w),&w);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_reason_sequence,6,sizeof(d),&d);
    size_t global[2]={length*16u,((b->width+15u)/16u)*16u},local[2]={16u,16u};
    for(layer=0;layer<depth&&error==CL_SUCCESS;layer++){
        error=clSetKernelArg(b->affine_reason_sequence,7,sizeof(layer),&layer);
        if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->affine_reason_sequence,2,NULL,global,local,0,NULL,NULL);
    }
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,b->reason_states,CL_TRUE,0,state_count*sizeof(float),reason_states,0,NULL,NULL);
    return error==CL_SUCCESS;
}
int ama10_opencl_reason_backward(AMA10OpenCL *b,const float *reason_states,
                                 const float *final_state_gradient,size_t length,
                                 unsigned depth,float *wu_gradient,float *wm_gradient,
                                 float *context_gradient){
    if(!b||!reason_states||!final_state_gradient||!wu_gradient||!wm_gradient||!context_gradient||
       !length||!depth||length>UINT32_MAX||depth>UINT32_MAX||b->width>UINT32_MAX||
       length>SIZE_MAX/(depth+1u)||length*(depth+1u)>SIZE_MAX/b->width||
       length*(depth+1u)*b->width>SIZE_MAX/sizeof(float)||length>SIZE_MAX/b->width||
       length*b->width>UINT32_MAX||b->width>SIZE_MAX/b->width||b->width*b->width>UINT32_MAX||
       b->width*b->width>SIZE_MAX/sizeof(float))return 0;
    const size_t states_n=length*(depth+1u)*b->width,seq_n=length*b->width,mat_n=b->width*b->width;
    const size_t states_bytes=states_n*sizeof(float),seq_bytes=seq_n*sizeof(float),mat_bytes=mat_n*sizeof(float);
    for(size_t i=0;i<states_n;i++)if(!isfinite(reason_states[i]))return 0;
    for(size_t i=0;i<seq_n;i++)if(!isfinite(final_state_gradient[i]))return 0;
    cl_int error=CL_SUCCESS;cl_mem states=NULL,q_a=NULL,q_b=NULL,dpre=NULL,ctx=NULL,gu=NULL,gm=NULL;
    states=clCreateBuffer(b->context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,states_bytes,(void*)reason_states,&error);
    if(error==CL_SUCCESS)q_a=clCreateBuffer(b->context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,seq_bytes,(void*)final_state_gradient,&error);
    if(error==CL_SUCCESS)q_b=clCreateBuffer(b->context,CL_MEM_READ_WRITE,seq_bytes,NULL,&error);
    if(error==CL_SUCCESS)dpre=clCreateBuffer(b->context,CL_MEM_READ_WRITE,seq_bytes,NULL,&error);
    if(error==CL_SUCCESS)ctx=clCreateBuffer(b->context,CL_MEM_READ_WRITE,seq_bytes,NULL,&error);
    if(error==CL_SUCCESS)gu=clCreateBuffer(b->context,CL_MEM_READ_WRITE,mat_bytes,NULL,&error);
    if(error==CL_SUCCESS)gm=clCreateBuffer(b->context,CL_MEM_READ_WRITE,mat_bytes,NULL,&error);
    float zero=0.0f;
    if(error==CL_SUCCESS)error=clEnqueueFillBuffer(b->queue,ctx,&zero,sizeof(zero),0,seq_bytes,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueFillBuffer(b->queue,gu,&zero,sizeof(zero),0,mat_bytes,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueFillBuffer(b->queue,gm,&zero,sizeof(zero),0,mat_bytes,0,NULL,NULL);
    cl_uint len=(cl_uint)length,w=(cl_uint)b->width,d=(cl_uint)depth,layer=0;
    size_t seq_global=seq_n,mat_global=mat_n;cl_mem q_current=q_a,q_previous=q_b;
    for(size_t rev=depth;rev>0&&error==CL_SUCCESS;rev--){layer=(cl_uint)(rev-1);
        cl_mem deriv_args[]={states,q_current,dpre};
        for(cl_uint i=0;i<3&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->reason_backward_derivative,i,sizeof(cl_mem),&deriv_args[i]);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->reason_backward_derivative,3,sizeof(len),&len);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->reason_backward_derivative,4,sizeof(w),&w);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->reason_backward_derivative,5,sizeof(d),&d);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->reason_backward_derivative,6,sizeof(layer),&layer);
        if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->reason_backward_derivative,1,NULL,&seq_global,NULL,0,NULL,NULL);
        cl_mem propagate_args[]={b->parameters[3],b->parameters[2],dpre,q_previous,ctx};
        for(cl_uint i=0;i<5&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->reason_backward_propagate,i,sizeof(cl_mem),&propagate_args[i]);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->reason_backward_propagate,5,sizeof(len),&len);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->reason_backward_propagate,6,sizeof(w),&w);
        if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->reason_backward_propagate,1,NULL,&seq_global,NULL,0,NULL,NULL);
        cl_mem gradient_args[]={states,dpre,gu,gm};
        for(cl_uint i=0;i<4&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->reason_backward_weight_gradient,i,sizeof(cl_mem),&gradient_args[i]);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->reason_backward_weight_gradient,4,sizeof(len),&len);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->reason_backward_weight_gradient,5,sizeof(w),&w);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->reason_backward_weight_gradient,6,sizeof(d),&d);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->reason_backward_weight_gradient,7,sizeof(layer),&layer);
        if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->reason_backward_weight_gradient,1,NULL,&mat_global,NULL,0,NULL,NULL);
        cl_mem swap=q_current;q_current=q_previous;q_previous=swap;
    }
    float *direct_gradient=NULL;
    if(error==CL_SUCCESS){direct_gradient=(float*)malloc(seq_bytes);if(!direct_gradient)error=CL_OUT_OF_HOST_MEMORY;}
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,q_current,CL_TRUE,0,seq_bytes,direct_gradient,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,gu,CL_TRUE,0,mat_bytes,wu_gradient,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,gm,CL_TRUE,0,mat_bytes,wm_gradient,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,ctx,CL_TRUE,0,seq_bytes,context_gradient,0,NULL,NULL);
    if(error==CL_SUCCESS){for(size_t i=0;i<seq_n;i++)context_gradient[i]+=direct_gradient[i];
        for(size_t i=0;i<mat_n;i++)if(!isfinite(wu_gradient[i])||!isfinite(wm_gradient[i])){error=CL_INVALID_VALUE;break;}
        for(size_t i=0;i<seq_n&&error==CL_SUCCESS;i++)if(!isfinite(context_gradient[i]))error=CL_INVALID_VALUE;}
    free(direct_gradient);
    cl_mem buffers[]={states,q_a,q_b,dpre,ctx,gu,gm};release_buffers(buffers,7);
    return error==CL_SUCCESS;
}
int ama10_opencl_affine_backward(AMA10OpenCL *b,const AMA10Model *m,
                                 const float *inputs,const float *projected,
                                 const float *gates,const float *candidates,
                                 const float *fast,const float *slow,
                                 const float *memory_lambda,const float *fast_lambda,
                                 const float *initial_fast,const float *initial_slow,
                                 size_t length,float *wx_gradient,float *wh_gradient,
                                 float *bias_gradient,float *gate_bias_gradient,
                                 float *input_gradient){
    if(!b||!m||!inputs||!projected||!gates||!candidates||!fast||!slow||!memory_lambda||!fast_lambda||
       !wx_gradient||!wh_gradient||!bias_gradient||!gate_bias_gradient||!input_gradient||!length||
       length>UINT32_MAX||b->width>UINT32_MAX||b->input_width>UINT32_MAX||
       m->width!=b->width||m->input_width!=b->input_width||
       length>SIZE_MAX/b->width||length*b->width>SIZE_MAX/sizeof(float)||
       length>SIZE_MAX/b->input_width||length*b->input_width>SIZE_MAX/sizeof(float)||
       b->width>SIZE_MAX/b->width||b->width*b->width>SIZE_MAX/sizeof(float)||
       b->width>SIZE_MAX/b->input_width||b->width*b->input_width>SIZE_MAX/sizeof(float))return 0;
    const size_t w=b->width,e=b->input_width,nw=length*w,nx=length*e;
    const size_t wb=w*sizeof(float),nwb=nw*sizeof(float),nxb=nx*sizeof(float);
    const size_t whb=w*w*sizeof(float),wxb=w*e*sizeof(float);
    const float *sequence_values[]={inputs,projected,gates,candidates,fast,slow,memory_lambda,fast_lambda};
    const size_t sequence_sizes[]={nx,nw,nw,nw,nw,nw,nw,nw};
    for(size_t k=0;k<8;k++)for(size_t i=0;i<sequence_sizes[k];i++)if(!isfinite(sequence_values[k][i]))return 0;
    cl_int error=CL_SUCCESS;cl_mem q[18]={0};
    for(size_t k=0;k<8&&error==CL_SUCCESS;k++)q[k]=clCreateBuffer(b->context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        sequence_sizes[k]*sizeof(float),(void*)sequence_values[k],&error);
    if(error==CL_SUCCESS){q[8]=clCreateBuffer(b->context,CL_MEM_READ_ONLY,wb,NULL,&error);
        if(error==CL_SUCCESS&&initial_fast)error=clEnqueueWriteBuffer(b->queue,q[8],CL_FALSE,0,wb,(void*)initial_fast,0,NULL,NULL);
        if(error==CL_SUCCESS&&!initial_fast){float zero=0.0f;error=clEnqueueFillBuffer(b->queue,q[8],&zero,sizeof(zero),0,wb,0,NULL,NULL);}}
    if(error==CL_SUCCESS){q[9]=clCreateBuffer(b->context,CL_MEM_READ_ONLY,wb,NULL,&error);
        if(error==CL_SUCCESS&&initial_slow)error=clEnqueueWriteBuffer(b->queue,q[9],CL_FALSE,0,wb,(void*)initial_slow,0,NULL,NULL);
        if(error==CL_SUCCESS&&!initial_slow){float zero=0.0f;error=clEnqueueFillBuffer(b->queue,q[9],&zero,sizeof(zero),0,wb,0,NULL,NULL);}}
    for(size_t k=10;k<14&&error==CL_SUCCESS;k++)q[k]=clCreateBuffer(b->context,CL_MEM_READ_WRITE,k==13?nxb:nwb,NULL,&error);
    for(size_t k=14;k<18&&error==CL_SUCCESS;k++)q[k]=clCreateBuffer(b->context,CL_MEM_READ_WRITE,
        k==14?wxb:(k==15?whb:wb),NULL,&error);
    if(error==CL_SUCCESS&&b->parameter_model!=m)error=clEnqueueWriteBuffer(b->queue,b->parameters[0],CL_FALSE,0,wxb,m->wx,0,NULL,NULL);
    if(error==CL_SUCCESS&&b->parameter_model!=m)error=clEnqueueWriteBuffer(b->queue,b->parameters[1],CL_FALSE,0,whb,m->wh,0,NULL,NULL);
    cl_uint len=(cl_uint)length,width=(cl_uint)w,input_width=(cl_uint)e;size_t nw_global=nw,nx_global=nx,wh_global=w*w,wx_global=w*e;
    cl_mem delta_args[]={q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11]};
    for(cl_uint i=0;i<10&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->affine_backward_deltas,i,sizeof(cl_mem),&delta_args[i]);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_backward_deltas,10,sizeof(len),&len);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_backward_deltas,11,sizeof(width),&width);
    if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->affine_backward_deltas,1,NULL,&nw_global,NULL,0,NULL,NULL);
    cl_mem z_args[]={b->parameters[1],q[10],q[11],q[12]};
    for(cl_uint i=0;i<4&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->affine_backward_z_gradient,i,sizeof(cl_mem),&z_args[i]);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_backward_z_gradient,4,sizeof(len),&len);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_backward_z_gradient,5,sizeof(width),&width);
    if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->affine_backward_z_gradient,1,NULL,&nw_global,NULL,0,NULL,NULL);
    cl_mem rec_args[]={q[1],q[11],q[15]};
    for(cl_uint i=0;i<3&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->affine_backward_recurrent_gradient,i,sizeof(cl_mem),&rec_args[i]);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_backward_recurrent_gradient,3,sizeof(len),&len);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_backward_recurrent_gradient,4,sizeof(width),&width);
    if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->affine_backward_recurrent_gradient,1,NULL,&wh_global,NULL,0,NULL,NULL);
    cl_mem bias_args[]={q[10],q[11],q[16],q[17]};
    for(cl_uint i=0;i<4&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->affine_backward_bias_gradient,i,sizeof(cl_mem),&bias_args[i]);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_backward_bias_gradient,4,sizeof(len),&len);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_backward_bias_gradient,5,sizeof(width),&width);
    if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->affine_backward_bias_gradient,1,NULL,&(size_t){w},NULL,0,NULL,NULL);
    cl_mem in_args[]={b->parameters[0],q[12],q[13]};
    for(cl_uint i=0;i<3&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->affine_backward_input_gradient,i,sizeof(cl_mem),&in_args[i]);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_backward_input_gradient,3,sizeof(len),&len);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_backward_input_gradient,4,sizeof(width),&width);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_backward_input_gradient,5,sizeof(input_width),&input_width);
    if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->affine_backward_input_gradient,1,NULL,&nx_global,NULL,0,NULL,NULL);
    cl_mem wx_args[]={q[0],q[12],q[14]};
    for(cl_uint i=0;i<3&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->affine_backward_input_weight_gradient,i,sizeof(cl_mem),&wx_args[i]);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_backward_input_weight_gradient,3,sizeof(len),&len);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_backward_input_weight_gradient,4,sizeof(width),&width);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->affine_backward_input_weight_gradient,5,sizeof(input_width),&input_width);
    if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->affine_backward_input_weight_gradient,1,NULL,&wx_global,NULL,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,q[14],CL_TRUE,0,wxb,wx_gradient,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,q[15],CL_TRUE,0,whb,wh_gradient,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,q[16],CL_TRUE,0,wb,bias_gradient,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,q[17],CL_TRUE,0,wb,gate_bias_gradient,0,NULL,NULL);
    if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,q[13],CL_TRUE,0,nxb,input_gradient,0,NULL,NULL);
    if(error==CL_SUCCESS){for(size_t i=0;i<w*e;i++)if(!isfinite(wx_gradient[i])){error=CL_INVALID_VALUE;break;}
        for(size_t i=0;i<w*w&&error==CL_SUCCESS;i++)if(!isfinite(wh_gradient[i]))error=CL_INVALID_VALUE;
        for(size_t i=0;i<w&&error==CL_SUCCESS;i++)if(!isfinite(bias_gradient[i])||!isfinite(gate_bias_gradient[i]))error=CL_INVALID_VALUE;
        for(size_t i=0;i<nx&&error==CL_SUCCESS;i++)if(!isfinite(input_gradient[i]))error=CL_INVALID_VALUE;}
    release_buffers(q,18);return error==CL_SUCCESS;
}
int ama10_opencl_head_forward(AMA10OpenCL *b,const float *head,const float *hidden,size_t length,float *logits){
    if(!b||!head||!hidden||!logits||!length||length>SIZE_MAX/b->width||length*b->width>SIZE_MAX/sizeof(float)||
       length>SIZE_MAX/b->vocabulary||length*b->vocabulary>SIZE_MAX/sizeof(float)||b->vocabulary>SIZE_MAX/b->width)return 0;
    size_t hn=length*b->width,ln=length*b->vocabulary,headn=b->vocabulary*b->width;
    size_t hb=hn*sizeof(float),lb=ln*sizeof(float),wb=headn*sizeof(float);cl_int error=CL_SUCCESS;
    cl_mem bufs[2]={0};bufs[0]=clCreateBuffer(b->context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,hb,(void*)hidden,&error);
    if(error==CL_SUCCESS)bufs[1]=clCreateBuffer(b->context,CL_MEM_WRITE_ONLY,lb,NULL,&error);
    if(error==CL_SUCCESS&&(!b->parameter_model||head!=b->parameter_model->head))
        error=clEnqueueWriteBuffer(b->queue,b->parameters[7],CL_TRUE,0,wb,(void*)head,0,NULL,NULL);
    if(error==CL_SUCCESS){
        error=clSetKernelArg(b->sequence_logits,0,sizeof(cl_mem),&b->parameters[7]);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->sequence_logits,1,sizeof(cl_mem),&bufs[0]);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->sequence_logits,2,sizeof(cl_mem),&bufs[1]);
        cl_uint len=(cl_uint)length,w=(cl_uint)b->width,v=(cl_uint)b->vocabulary;
        if(length>UINT32_MAX)error=CL_INVALID_VALUE;
        if(error==CL_SUCCESS)error=clSetKernelArg(b->sequence_logits,3,sizeof(len),&len);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->sequence_logits,4,sizeof(w),&w);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->sequence_logits,5,sizeof(v),&v);
        size_t global=ln;if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->sequence_logits,1,NULL,&global,NULL,0,NULL,NULL);
        if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,bufs[1],CL_TRUE,0,lb,logits,0,NULL,NULL);
    }
    release_buffers(bufs,2);return error==CL_SUCCESS;
}
int ama10_opencl_head_backward(AMA10OpenCL *b,const float *head,const float *hidden,const float *errors,size_t length,float *head_grad,float *hidden_grad){
    if(!b||!head||!hidden||!errors||!head_grad||!hidden_grad||!length||length>UINT32_MAX||
       length>SIZE_MAX/b->width||length*b->width>SIZE_MAX/sizeof(float)||length>SIZE_MAX/b->vocabulary||
       length*b->vocabulary>SIZE_MAX/sizeof(float)||b->vocabulary>SIZE_MAX/b->width)return 0;
    size_t hn=length*b->width,ln=length*b->vocabulary,headn=b->vocabulary*b->width;
    size_t hb=hn*sizeof(float),lb=ln*sizeof(float),wb=headn*sizeof(float);cl_int error=CL_SUCCESS;
    cl_mem bufs[4]={0};bufs[0]=clCreateBuffer(b->context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,hb,(void*)hidden,&error);
    if(error==CL_SUCCESS)bufs[1]=clCreateBuffer(b->context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,lb,(void*)errors,&error);
    if(error==CL_SUCCESS)bufs[2]=clCreateBuffer(b->context,CL_MEM_WRITE_ONLY,wb,NULL,&error);
    if(error==CL_SUCCESS)bufs[3]=clCreateBuffer(b->context,CL_MEM_WRITE_ONLY,hb,NULL,&error);
    if(error==CL_SUCCESS&&(!b->parameter_model||head!=b->parameter_model->head))
        error=clEnqueueWriteBuffer(b->queue,b->parameters[7],CL_TRUE,0,wb,(void*)head,0,NULL,NULL);
    if(error==CL_SUCCESS){
        error=clSetKernelArg(b->head_gradient,0,sizeof(cl_mem),&b->parameters[7]);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->head_gradient,1,sizeof(cl_mem),&bufs[0]);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->head_gradient,2,sizeof(cl_mem),&bufs[1]);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->head_gradient,3,sizeof(cl_mem),&bufs[2]);
        cl_uint len=(cl_uint)length,w=(cl_uint)b->width,v=(cl_uint)b->vocabulary;
        if(error==CL_SUCCESS)error=clSetKernelArg(b->head_gradient,4,sizeof(len),&len);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->head_gradient,5,sizeof(w),&w);
        if(error==CL_SUCCESS)error=clSetKernelArg(b->head_gradient,6,sizeof(v),&v);
        size_t global=headn;if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->head_gradient,1,NULL,&global,NULL,0,NULL,NULL);
        if(error==CL_SUCCESS){
            error=clSetKernelArg(b->hidden_gradient,0,sizeof(cl_mem),&b->parameters[7]);
            if(error==CL_SUCCESS)error=clSetKernelArg(b->hidden_gradient,1,sizeof(cl_mem),&bufs[0]);
            if(error==CL_SUCCESS)error=clSetKernelArg(b->hidden_gradient,2,sizeof(cl_mem),&bufs[1]);
            if(error==CL_SUCCESS)error=clSetKernelArg(b->hidden_gradient,3,sizeof(cl_mem),&bufs[3]);
            if(error==CL_SUCCESS)error=clSetKernelArg(b->hidden_gradient,4,sizeof(len),&len);
            if(error==CL_SUCCESS)error=clSetKernelArg(b->hidden_gradient,5,sizeof(w),&w);
            if(error==CL_SUCCESS)error=clSetKernelArg(b->hidden_gradient,6,sizeof(v),&v);
            global=hn;if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->hidden_gradient,1,NULL,&global,NULL,0,NULL,NULL);}
        if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,bufs[2],CL_TRUE,0,wb,head_grad,0,NULL,NULL);
        if(error==CL_SUCCESS)error=clEnqueueReadBuffer(b->queue,bufs[3],CL_TRUE,0,hb,hidden_grad,0,NULL,NULL);
    }
    release_buffers(bufs,4);return error==CL_SUCCESS;
}

int ama10_opencl_train_forward(AMA10OpenCL *b,const AMA10Model *m,const uint32_t *tokens,
                               size_t length,unsigned depth,const float *initial_fast,
                               const float *initial_slow,float *sequence_hidden,float *final_slow){
    return ama10_opencl_train_forward_multimodal(b,m,tokens,NULL,NULL,length,depth,
        initial_fast,initial_slow,sequence_hidden,final_slow);
}
int ama10_opencl_train_forward_multimodal(AMA10OpenCL *b,const AMA10Model *m,const uint32_t *tokens,
                               const float *features,const unsigned char *feature_mask,
                               size_t length,unsigned depth,const float *initial_fast,
                               const float *initial_slow,float *sequence_hidden,float *final_slow){
    if(!b||!m||!tokens||!length||!depth||!initial_fast||!initial_slow||!sequence_hidden||!final_slow||
       m->width!=b->width||m->input_width!=b->input_width||m->vocabulary!=b->vocabulary||
       length>SIZE_MAX/b->width||length*b->width>SIZE_MAX/sizeof(float))return 0;
    size_t w=b->width,e=b->input_width,n=length*w;
    float *projected=(float*)malloc(n*sizeof(float)),*gate=(float*)malloc(n*sizeof(float));
    float *candidate=(float*)malloc(n*sizeof(float)),*slow_offset=(float*)malloc(n*sizeof(float));
    float *slow_states=(float*)malloc(n*sizeof(float));
    if(!projected||!gate||!candidate||!slow_offset||!slow_states){free(projected);free(gate);free(candidate);free(slow_offset);free(slow_states);return 0;}
    int ok=1;
    for(size_t t=0;t<length&&ok;t++){
        int use_feature=features&&feature_mask&&feature_mask[t];
        if(!use_feature&&tokens[t]>=m->vocabulary){ok=0;break;}
        for(size_t i=0;i<w;i++){
            double z=0.0;for(size_t j=0;j<e;j++){
                float x=use_feature?features[t*e+j]:m->embed[j*m->vocabulary+tokens[t]];
                if(!isfinite(x)){ok=0;break;}z+=(double)m->wx[i*e+j]*x;
            }
            projected[t*w+i]=(float)z;
        }
        for(size_t i=0;i<w&&ok;i++){
            double g=m->gate_bias[i];for(size_t j=0;j<w;j++)g+=(double)m->wh[i*w+j]*projected[t*w+j];
            gate[t*w+i]=g>=0.0?1.0f/(1.0f+expf(-(float)g)):expf((float)g)/(1.0f+expf((float)g));
            candidate[t*w+i]=tanhf(projected[t*w+i]+m->bias[i]);
        }
    }
    if(ok)ok=ama10_opencl_affine_scan(b,gate,candidate,initial_fast,length,sequence_hidden);
    if(ok){for(size_t k=0;k<n;k++)slow_offset[k]=(1.0f-gate[k])*sequence_hidden[k];
        ok=ama10_opencl_affine_scan(b,gate,slow_offset,initial_slow,length,slow_states);}
    if(ok)memcpy(final_slow,slow_states+(length-1)*w,w*sizeof(float));
    free(projected);free(gate);free(candidate);free(slow_offset);free(slow_states);return ok;
}int ama10_opencl_eval_logits(AMA10OpenCL *b,const AMA10Model *m,const uint32_t *tokens,
                             const float *features,const unsigned char *feature_mask,
                             size_t length,unsigned depth,const float *initial_fast,
                             const float *initial_slow,float *logits,float *final_fast,
                             float *final_slow){
    if(!b||!m||!tokens||!length||!depth||!initial_fast||!initial_slow||!logits||!final_fast||!final_slow||
       m->width!=b->width||m->input_width!=b->input_width||m->vocabulary!=b->vocabulary||
       length>SIZE_MAX/b->vocabulary||length*b->vocabulary>SIZE_MAX/sizeof(float))return 0;
    const size_t e=b->input_width,w=b->width,v=b->vocabulary;
    if(length>UINT32_MAX||length>SIZE_MAX/e||length*e>SIZE_MAX/sizeof(float)||
       length>SIZE_MAX/w||length*w>SIZE_MAX/sizeof(float)||length>SIZE_MAX/depth||
       length*depth>SIZE_MAX/w||length*depth*w>SIZE_MAX/sizeof(float)||
       length>SIZE_MAX/v||length*v>SIZE_MAX/sizeof(float))return 0;
    const size_t nx=length*e,nw=length*w,nr=length*depth*w,nl=length*v;
    float *inputs=(float*)malloc(nx*sizeof(float)),*projected=(float*)malloc(nw*sizeof(float));
    float *gates=(float*)malloc(nw*sizeof(float)),*candidates=(float*)malloc(nw*sizeof(float));
    float *fast=(float*)malloc(nw*sizeof(float)),*slow_offset=(float*)malloc(nw*sizeof(float));
    float *slow=(float*)malloc(nw*sizeof(float)),*context=(float*)malloc(nw*sizeof(float));
    float *reason=(float*)malloc(nr*sizeof(float)),*hidden=(float*)malloc(nw*sizeof(float));
    if(!inputs||!projected||!gates||!candidates||!fast||!slow_offset||!slow||!context||!reason||!hidden){
        free(inputs);free(projected);free(gates);free(candidates);free(fast);free(slow_offset);
        free(slow);free(context);free(reason);free(hidden);return 0;}
    int ok=1;
    for(size_t t=0;t<length&&ok;t++){
        int use_features=features&&feature_mask&&feature_mask[t];
        if(use_features)memcpy(inputs+t*e,features+t*e,e*sizeof(float));
        else if(tokens[t]<v)for(size_t j=0;j<e;j++)inputs[t*e+j]=m->embed[j*v+tokens[t]];
        else ok=0;
    }
    if(ok)ok=ama10_opencl_affine_inputs(b,m,inputs,length,projected,gates,candidates);
    if(ok)ok=ama10_opencl_affine_scan(b,gates,candidates,initial_fast,length,fast);
    if(ok){for(size_t i=0;i<nw;i++)slow_offset[i]=(1.0f-gates[i])*fast[i];
        ok=ama10_opencl_affine_scan(b,gates,slow_offset,initial_slow,length,slow);}
    if(ok){for(size_t i=0;i<nw;i++)context[i]=fast[i]+slow[i];
        ok=ama10_opencl_reason_sequence(b,m,context,length,depth,reason);}
    if(ok){for(size_t t=0;t<length;t++)memcpy(hidden+t*w,reason+(t*depth+depth-1)*w,w*sizeof(float));
        ok=ama10_opencl_head_forward(b,m->head,hidden,length,logits);}
    if(ok){memcpy(final_fast,fast+(length-1)*w,w*sizeof(float));memcpy(final_slow,slow+(length-1)*w,w*sizeof(float));
        for(size_t i=0;i<nx;i++)if(!isfinite(inputs[i])){ok=0;break;}
        for(size_t i=0;i<nl&&ok;i++)if(!isfinite(logits[i]))ok=0;}
    free(inputs);free(projected);free(gates);free(candidates);free(fast);free(slow_offset);
    free(slow);free(context);free(reason);free(hidden);return ok;
}int ama10_opencl_bptt_begin(AMA10OpenCL *b,const AMA10Model *m,const float *trajectory,
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
    if(error==CL_SUCCESS)error=clEnqueueFillBuffer(b->queue,b->bptt_feature_gradients,&zero,sizeof(zero),0,
        b->bptt_forward_length*b->input_width*sizeof(float),0,NULL,NULL);
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
        b->bptt_input_features,b->bptt_feature_mask,b->bptt_feature_gradients,b->bptt_next_dh,b->bptt_next_dm,
        b->bptt_gradients[0],b->bptt_gradients[1],b->bptt_gradients[2],b->bptt_gradients[3],
        b->bptt_gradients[4],b->bptt_gradients[5],b->bptt_gradients[6]};
    for(cl_uint i=0;i<23&&error==CL_SUCCESS;i++)error=clSetKernelArg(b->cell_backward_accumulate,i,sizeof(cl_mem),&accum_args[i]);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->cell_backward_accumulate,23,sizeof(width),&width);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->cell_backward_accumulate,24,sizeof(input_width),&input_width);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->cell_backward_accumulate,25,sizeof(vocabulary),&vocabulary);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->cell_backward_accumulate,26,sizeof(depth),&depth);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->cell_backward_accumulate,27,sizeof(t),&t);
    if(error==CL_SUCCESS)error=clSetKernelArg(b->cell_backward_accumulate,28,sizeof(d),&d);
    size_t ww=b->width*b->width,we=b->width*b->input_width;global=ww>we?ww:we;
    if(global<b->input_width)global=b->input_width;
    if(error==CL_SUCCESS)error=clEnqueueNDRangeKernel(b->queue,b->cell_backward_accumulate,1,NULL,&global,NULL,0,NULL,NULL);
    cl_mem swap=b->bptt_dh;b->bptt_dh=b->bptt_next_dh;b->bptt_next_dh=swap;
    swap=b->bptt_dm;b->bptt_dm=b->bptt_next_dm;b->bptt_next_dm=swap;
    return error==CL_SUCCESS;
}
int ama10_opencl_bptt_finish(AMA10OpenCL *b,AMA10Gradients *g,float *feature_gradients){
    if(!b||!g||!g->values||g->parameter_count!=8||g->width!=b->width||
       g->input_width!=b->input_width||g->vocabulary!=b->vocabulary)return 0;
    size_t sizes[7]={b->width*b->input_width,b->width*b->width,b->width*b->width,
        b->width*b->width,b->width,b->width,b->input_width*b->vocabulary};
    for(size_t i=0;i<7;i++)if(!g->values[i]||clEnqueueReadBuffer(b->queue,b->bptt_gradients[i],CL_TRUE,0,
        sizes[i]*sizeof(float),g->values[i],0,NULL,NULL)!=CL_SUCCESS)return 0;
    if(feature_gradients&&clEnqueueReadBuffer(b->queue,b->bptt_feature_gradients,CL_TRUE,0,
        b->bptt_forward_length*b->input_width*sizeof(float),feature_gradients,0,NULL,NULL)!=CL_SUCCESS)return 0;
    return 1;
}

const char *ama10_opencl_device_name(const AMA10OpenCL *b){return b?b->device_name:"";}
void ama10_opencl_destroy(AMA10OpenCL *b){
    if(!b)return;
    release_adamw_state(b);
    cl_mem bptt_buffers[]={b->bptt_dh,b->bptt_dm,b->bptt_next_dh,b->bptt_next_dm,b->bptt_dpre,b->bptt_dgate,
        b->bptt_direct_fast,b->bptt_direct_slow,b->bptt_cache,b->bptt_tokens,b->bptt_hidden_gradient,
        b->bptt_input_features,b->bptt_feature_mask,b->bptt_feature_gradients,b->bptt_logits};
    release_buffers(bptt_buffers,sizeof(bptt_buffers)/sizeof(bptt_buffers[0]));
    release_buffers(b->bptt_gradients,8);
    cl_mem sequence_buffers[]={b->sequence_inputs,b->sequence_projected,b->sequence_gates,b->sequence_candidates};
    release_buffers(sequence_buffers,4);
    release_buffers(b->scan_a,2);release_buffers(b->scan_b,2);
    if(b->reason_input)clReleaseMemObject(b->reason_input);
    if(b->reason_states)clReleaseMemObject(b->reason_states);
    if(b->logits_buffer)clReleaseMemObject(b->logits_buffer);
    if(b->input_buffer)clReleaseMemObject(b->input_buffer);
    if(b->projected)clReleaseMemObject(b->projected);
    if(b->state_context_buffer)clReleaseMemObject(b->state_context_buffer);
    if(b->reason_a)clReleaseMemObject(b->reason_a);
    if(b->reason_b)clReleaseMemObject(b->reason_b);
    if(b->confidence_buffer)clReleaseMemObject(b->confidence_buffer);
    if(b->next_slow)clReleaseMemObject(b->next_slow);
    if(b->next_fast)clReleaseMemObject(b->next_fast);
    if(b->slow)clReleaseMemObject(b->slow);
    if(b->fast)clReleaseMemObject(b->fast);
    for(size_t i=0;i<8;i++)if(b->parameters[i])clReleaseMemObject(b->parameters[i]);
    if(b->logits_kernel)clReleaseKernel(b->logits_kernel);
    if(b->project_features)clReleaseKernel(b->project_features);
    if(b->project_token)clReleaseKernel(b->project_token);
    if(b->affine_update)clReleaseKernel(b->affine_update);
    if(b->state_context)clReleaseKernel(b->state_context);
    if(b->reason_step)clReleaseKernel(b->reason_step);
    if(b->affine_project_sequence)clReleaseKernel(b->affine_project_sequence);
    if(b->affine_gate_sequence)clReleaseKernel(b->affine_gate_sequence);
    if(b->affine_scan_compose)clReleaseKernel(b->affine_scan_compose);
    if(b->affine_reason_sequence)clReleaseKernel(b->affine_reason_sequence);
    if(b->reason_backward_derivative)clReleaseKernel(b->reason_backward_derivative);
    if(b->reason_backward_propagate)clReleaseKernel(b->reason_backward_propagate);
    if(b->reason_backward_weight_gradient)clReleaseKernel(b->reason_backward_weight_gradient);
    if(b->affine_backward_deltas)clReleaseKernel(b->affine_backward_deltas);
    if(b->affine_backward_z_gradient)clReleaseKernel(b->affine_backward_z_gradient);
    if(b->affine_backward_recurrent_gradient)clReleaseKernel(b->affine_backward_recurrent_gradient);
    if(b->affine_backward_bias_gradient)clReleaseKernel(b->affine_backward_bias_gradient);
    if(b->affine_backward_input_gradient)clReleaseKernel(b->affine_backward_input_gradient);
    if(b->affine_backward_input_weight_gradient)clReleaseKernel(b->affine_backward_input_weight_gradient);
    if(b->adamw_update)clReleaseKernel(b->adamw_update);
    if(b->gradient_pair_stats)clReleaseKernel(b->gradient_pair_stats);
    if(b->gradient_pair_finalize)clReleaseKernel(b->gradient_pair_finalize);
    if(b->gradient_pair_merge)clReleaseKernel(b->gradient_pair_merge);
    if(b->hidden_gradient)clReleaseKernel(b->hidden_gradient);
    if(b->cell_backward_accumulate)clReleaseKernel(b->cell_backward_accumulate);
    if(b->cell_backward_derivatives)clReleaseKernel(b->cell_backward_derivatives);
    if(b->training_forward_fused)clReleaseKernel(b->training_forward_fused);
    if(b->head_gradient)clReleaseKernel(b->head_gradient);
    if(b->sequence_logits)clReleaseKernel(b->sequence_logits);
    if(b->confidence)clReleaseKernel(b->confidence);
    if(b->slow_update)clReleaseKernel(b->slow_update);
    if(b->cell)clReleaseKernel(b->cell);
    if(b->program)clReleaseProgram(b->program);
    if(b->queue)clReleaseCommandQueue(b->queue);
    if(b->context)clReleaseContext(b->context);
    free(b);
}
