#include "ama10.h"
#ifdef AMA10_ENABLE_OPENCL
#include "opencl_backend.h"
#endif
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define AFFINE_PARAM_COUNT 8u
static float affine_sigmoid(float x){
    if(x>=0.0f){float e=expf(-x);return 1.0f/(1.0f+e);}
    float e=expf(x);return e/(1.0f+e);
}
static double affine_entropy(const float *logits,size_t start,size_t count){
    float maximum=logits[start];
    for(size_t k=1;k<count;k++)if(logits[start+k]>maximum)maximum=logits[start+k];
    double sum=0.0;for(size_t k=0;k<count;k++)sum+=exp((double)logits[start+k]-(double)maximum);
    if(!(sum>0.0)||!isfinite(sum))return INFINITY;
    double h=0.0;for(size_t k=0;k<count;k++){double p=exp((double)logits[start+k]-(double)maximum)/sum;if(p>0.0)h-=p*log(p);}
    return h;
}
static int affine_scan_cpu(const float *a,const float *b,const float *initial,
                           size_t length,size_t width,float *states){
    if(!a||!b||!initial||!states||!length||!width||length>SIZE_MAX/width||
       length*width>SIZE_MAX/sizeof(float))return 0;
    for(size_t i=0;i<width;i++)if(!isfinite(initial[i]))return 0;
    for(size_t t=0;t<length;t++)for(size_t i=0;i<width;i++){
        size_t k=t*width+i;
        float previous=t?states[k-width]:initial[i];
        states[k]=a[k]*previous+b[k];
        if(!isfinite(states[k]))return 0;
    }
    return 1;
}
static int affine_alloc(float **p,size_t n){
    if(!n||n>SIZE_MAX/sizeof(float))return 0;
    *p=(float*)calloc(n,sizeof(float));return *p!=NULL;
}
static void affine_parameter_views(AMA10Model *m,float **p,size_t *n){
    size_t w=m->width,e=m->input_width,v=m->vocabulary;
    p[0]=m->wx;n[0]=w*e;p[1]=m->wh;n[1]=w*w;p[2]=m->wm;n[2]=w*w;
    p[3]=m->wu;n[3]=w*w;p[4]=m->bias;n[4]=w;p[5]=m->gate_bias;n[5]=w;
    p[6]=m->embed;n[6]=e*v;p[7]=m->head;n[7]=v*w;
}
static int affine_scan(const float *a,const float *b,const float *initial,
                       size_t length,size_t width,float *states,void *opencl){
#ifdef AMA10_ENABLE_OPENCL
    if(opencl)return ama10_opencl_affine_scan((AMA10OpenCL*)opencl,a,b,initial,length,states);
#else
    (void)opencl;
#endif
    return affine_scan_cpu(a,b,initial,length,width,states);
}

int ama10_affine_train_sequence_internal(AMA10Model *m,AMA10Optimizer *opt,AMA10State *state,
                         const uint32_t *input,const float *input_features,
                         const unsigned char *feature_mask,const unsigned char *loss_mask,
                         const uint32_t *class_starts,const uint32_t *class_counts,
                         const uint32_t *target,const float *soft_targets,size_t length,
                         unsigned depth,float halt_threshold,float learning_rate,float weight_decay,
                         AMA10Gradients *accumulator,float *feature_gradients,
                         float *mean_loss,float *mean_depth,void *opencl){
    if(!m||!opt||!m->width||!m->input_width||!m->vocabulary||(!input&&!input_features)||
       (!input&&feature_mask)||(input&&input_features&&!feature_mask)||(!target&&!soft_targets)||
       !length||!depth||(feature_gradients&&!input_features)||(input_features&&!feature_gradients)||
       (feature_mask&&!input_features)||(halt_threshold>=0.0f&&(!isfinite(halt_threshold)||halt_threshold>2.0f))||
       (!accumulator&&(!isfinite(learning_rate)||learning_rate<=0.0f||!isfinite(weight_decay)||weight_decay<0.0f))||
       opt->parameter_count!=AFFINE_PARAM_COUNT||!opt->first_moment||!opt->second_moment||
       (state&&(!state->fast||!state->slow||state->width!=m->width||length>UINT_MAX-state->steps))||
       (accumulator&&(!accumulator->values||accumulator->parameter_count!=AFFINE_PARAM_COUNT||
        accumulator->width!=m->width||accumulator->input_width!=m->input_width||
        accumulator->vocabulary!=m->vocabulary)))return 0;
    const size_t w=m->width,e=m->input_width,v=m->vocabulary;
    if(length>SIZE_MAX/e||length*e>SIZE_MAX/sizeof(float)||length>SIZE_MAX/w||
       length*w>SIZE_MAX/sizeof(float)||length>SIZE_MAX/v||length*v>SIZE_MAX/sizeof(float)||
       depth==UINT_MAX||length>SIZE_MAX/(depth+1u)||length*(depth+1u)>SIZE_MAX/w||
       length*(depth+1u)*w>SIZE_MAX/sizeof(float))return 0;
    size_t supervised=0;
    for(size_t t=0;t<length;t++)if(!loss_mask||loss_mask[t]){
        size_t start=class_starts?class_starts[t]:0;
        size_t count=class_counts?class_counts[t]:v-start;
        if(start>=v||!count||count>v-start)return 0;
        if(soft_targets){
            double mass=0.0;
            for(size_t k=0;k<v;k++){
                float p=soft_targets[t*v+k];
                if(!isfinite(p)||p<0.0f||((k<start||k>=start+count)&&p!=0.0f))return 0;
                mass+=p;
            }
            if(fabs(mass-1.0)>1.0e-4)return 0;
        }else if(target[t]<start||target[t]-start>=count)return 0;
        supervised++;
    }
    if(!supervised||(accumulator&&accumulator->tokens>UINT64_MAX-supervised))return 0;
    const size_t nw=length*w,nx=length*e,nq=length*(depth+1u)*w,nv=length*v;
    float *x=NULL,*z=NULL,*gate=NULL,*candidate=NULL,*fast=NULL,*slow=NULL,*context=NULL,*reason=NULL;
    float *logits=NULL,*errors=NULL,*context_gradient=NULL,*memory_lambda=NULL,*fast_lambda=NULL;
    float *reverse_a=NULL,*reverse_b=NULL,*reverse_states=NULL,*z_gradient=NULL;
    float *gpu_hidden=NULL,*gpu_hidden_gradient=NULL,*gpu_head_gradient=NULL,*gpu_reason_states=NULL;
    unsigned *used_depth=NULL;
    int gpu_batched_head=0;
#ifdef AMA10_ENABLE_OPENCL
    gpu_batched_head=opencl&&halt_threshold<0.0f;
#else
    (void)opencl;
#endif
    float *grads[AFFINE_PARAM_COUNT]={0},*params[AFFINE_PARAM_COUNT];size_t sizes[AFFINE_PARAM_COUNT];
    affine_parameter_views(m,params,sizes);
    if(!affine_alloc(&x,nx)||!affine_alloc(&z,nw)||!affine_alloc(&gate,nw)||
       !affine_alloc(&candidate,nw)||!affine_alloc(&fast,nw)||!affine_alloc(&slow,nw)||
       !affine_alloc(&context,nw)||!affine_alloc(&reason,nq)||!affine_alloc(&logits,nv)||
       !affine_alloc(&errors,nv)||!affine_alloc(&context_gradient,nw)||
       !affine_alloc(&memory_lambda,nw)||!affine_alloc(&fast_lambda,nw)||
       !affine_alloc(&reverse_a,nw)||!affine_alloc(&reverse_b,nw)||
       !affine_alloc(&reverse_states,nw)||!affine_alloc(&z_gradient,w)||
       !(used_depth=(unsigned*)calloc(length,sizeof(unsigned))))goto fail;
#ifdef AMA10_ENABLE_OPENCL
    if(gpu_batched_head&&(!affine_alloc(&gpu_hidden,nw)||!affine_alloc(&gpu_hidden_gradient,nw)||
       !affine_alloc(&gpu_head_gradient,sizes[7])))goto fail;
    if(gpu_batched_head&&(!affine_alloc(&gpu_reason_states,length*depth*w)))goto fail;
#endif
    if(feature_gradients)memset(feature_gradients,0,nx*sizeof(float));
    for(size_t k=0;k<AFFINE_PARAM_COUNT;k++)if(!affine_alloc(&grads[k],sizes[k]))goto fail;
    unsigned total_depth=0;
    for(size_t t=0;t<length;t++){
        int use_feature=input_features&&(!feature_mask||feature_mask[t]);
        if(!use_feature&&(!input||input[t]>=v))goto fail;
        for(size_t j=0;j<e;j++){
            float value=use_feature?input_features[t*e+j]:m->embed[j*v+input[t]];
            if(!isfinite(value))goto fail;
            x[t*e+j]=value;
        }
    }
#ifdef AMA10_ENABLE_OPENCL
    if(opencl){
        if(!ama10_opencl_affine_inputs((AMA10OpenCL*)opencl,m,x,length,z,gate,candidate))goto fail;
    }else
#endif
    {
      for(size_t t=0;t<length;t++){
        for(size_t i=0;i<w;i++){
            float projected=0.0f;for(size_t j=0;j<e;j++)projected+=m->wx[i*e+j]*x[t*e+j];
            z[t*w+i]=projected;
        }
        for(size_t i=0;i<w;i++){
            float gate_logit=m->gate_bias[i];
            for(size_t j=0;j<w;j++)gate_logit+=m->wh[i*w+j]*z[t*w+j];
            gate[t*w+i]=affine_sigmoid(gate_logit);
            candidate[t*w+i]=tanhf(z[t*w+i]+m->bias[i]);
        }
      }
    }
    {
        float *scan_b=(float*)malloc(nw*sizeof(float)),*slow_scan_b=(float*)malloc(nw*sizeof(float));
        float *zero=(float*)calloc(w,sizeof(float));
        if(!scan_b||!slow_scan_b||!zero){free(scan_b);free(slow_scan_b);free(zero);goto fail;}
        for(size_t k=0;k<nw;k++)scan_b[k]=candidate[k];
        const float *initial_fast=state?state->fast:zero;
        const float *initial_slow=state?state->slow:zero;
        int scanned=affine_scan(gate,scan_b,initial_fast,length,w,fast,opencl);
        if(scanned){for(size_t k=0;k<nw;k++)slow_scan_b[k]=(1.0f-gate[k])*fast[k];
            scanned=affine_scan(gate,slow_scan_b,initial_slow,length,w,slow,opencl);}
        free(scan_b);free(slow_scan_b);free(zero);if(!scanned)goto fail;
    }
    for(size_t t=0;t<length;t++)for(size_t i=0;i<w;i++){
        size_t k=t*w+i;context[k]=fast[k]+slow[k];reason[(t*(depth+1u))*w+i]=context[k];
    }
#ifdef AMA10_ENABLE_OPENCL
    if(gpu_batched_head){
        if(!ama10_opencl_reason_sequence((AMA10OpenCL*)opencl,m,context,length,depth,gpu_reason_states))goto fail;
        for(size_t t=0;t<length;t++){
            used_depth[t]=depth;total_depth+=depth;
            for(size_t d=0;d<depth;d++)memcpy(reason+(t*(depth+1u)+d+1u)*w,
                gpu_reason_states+(t*depth+d)*w,w*sizeof(float));
        }
    }else
#endif
    {
      for(size_t t=0;t<length;t++){
        double previous_entropy=INFINITY;
        size_t reason_start=class_starts?class_starts[t]:0;
        size_t reason_count=class_counts?class_counts[t]:v-reason_start;
        for(unsigned d=0;d<depth;d++){
            const float *previous=reason+(t*(depth+1u)+d)*w;
            float *next=reason+(t*(depth+1u)+d+1u)*w;
            for(size_t i=0;i<w;i++){
                float value=0.0f;for(size_t j=0;j<w;j++)value+=m->wu[i*w+j]*previous[j]+m->wm[i*w+j]*context[t*w+j];
                next[i]=tanhf(value);
            }
            used_depth[t]=d+1;
            if(halt_threshold>=0.0f)for(size_t k=reason_start;k<reason_start+reason_count;k++){
                float value=0.0f;for(size_t j=0;j<w;j++)value+=m->head[k*w+j]*next[j];
                logits[t*v+k]=value;
            }
            if(halt_threshold>=0.0f){
                double entropy=affine_entropy(logits+t*v,reason_start,reason_count);
                if(d+1>=2&&previous_entropy-entropy<=halt_threshold)break;
                previous_entropy=entropy;
            }
        }
        total_depth+=used_depth[t];
      }
    }
#ifdef AMA10_ENABLE_OPENCL
    if(gpu_batched_head){
        for(size_t t=0;t<length;t++)memcpy(gpu_hidden+t*w,reason+(t*(depth+1u)+used_depth[t])*w,w*sizeof(float));
        if(!ama10_opencl_head_forward((AMA10OpenCL*)opencl,m->head,gpu_hidden,length,logits))goto fail;
    }
#endif
    float loss=0.0f;
    for(size_t t=0;t<length;t++){
        const size_t start=class_starts?class_starts[t]:0;
        const size_t count=class_counts?class_counts[t]:v-start;
        const float *hidden=reason+(t*(depth+1u)+used_depth[t])*w;float maximum=-INFINITY;
        if(!gpu_batched_head&&halt_threshold<0.0f)for(size_t k=start;k<start+count;k++){
            float value=0.0f;for(size_t j=0;j<w;j++)value+=m->head[k*w+j]*hidden[j];logits[t*v+k]=value;
        }
        for(size_t k=start;k<start+count;k++)if(logits[t*v+k]>maximum)maximum=logits[t*v+k];
        double sum=0.0;for(size_t k=start;k<start+count;k++)sum+=exp((double)logits[t*v+k]-(double)maximum);
        if(!(sum>0.0)||!isfinite(sum))goto fail;
        double log_sum=log(sum);
        if(!loss_mask||loss_mask[t]){
            if(soft_targets){for(size_t k=start;k<start+count;k++)if(soft_targets[t*v+k]>0.0f)
                loss-=soft_targets[t*v+k]*(float)((double)logits[t*v+k]-(double)maximum-log_sum);}
            else loss-=logits[t*v+target[t]]-(float)maximum-(float)log_sum;
        }
        if(!loss_mask||loss_mask[t]){
            for(size_t k=start;k<start+count;k++)errors[t*v+k]=(float)(exp((double)logits[t*v+k]-(double)maximum)/sum)-
                (soft_targets?soft_targets[t*v+k]:(target&&target[t]==k?1.0f:0.0f));
        }
    }
#ifdef AMA10_ENABLE_OPENCL
    if(gpu_batched_head){
        if(!ama10_opencl_head_backward((AMA10OpenCL*)opencl,m->head,gpu_hidden,errors,length,
                                       gpu_head_gradient,gpu_hidden_gradient))goto fail;
        for(size_t i=0;i<sizes[7];i++)grads[7][i]+=gpu_head_gradient[i];
        if(!ama10_opencl_reason_backward((AMA10OpenCL*)opencl,reason,gpu_hidden_gradient,
            length,depth,grads[3],grads[2],context_gradient))goto fail;
    }
#endif
#ifdef AMA10_ENABLE_OPENCL
    if(!gpu_batched_head)
#endif
    for(size_t t=0;t<length;t++){
        const float *hidden=reason+(t*(depth+1u)+used_depth[t])*w;
        float *qgrad=(float*)calloc(w,sizeof(float));if(!qgrad)goto fail;
#ifdef AMA10_ENABLE_OPENCL
        if(gpu_batched_head)memcpy(qgrad,gpu_hidden_gradient+t*w,w*sizeof(float));
        else
#endif
        for(size_t k=0;k<v;k++)if(errors[t*v+k]!=0.0f){
            for(size_t j=0;j<w;j++){grads[7][k*w+j]+=errors[t*v+k]*hidden[j];qgrad[j]+=m->head[k*w+j]*errors[t*v+k];}
        }
        for(unsigned rev=used_depth[t];rev>0;rev--){
            const float *previous=reason+(t*(depth+1u)+rev-1u)*w;
            const float *current=reason+(t*(depth+1u)+rev)*w;
            float *previous_grad=(float*)calloc(w,sizeof(float));if(!previous_grad){free(qgrad);goto fail;}
            for(size_t i=0;i<w;i++){
                float dpre=qgrad[i]*(1.0f-current[i]*current[i]);
                for(size_t j=0;j<w;j++){
                    grads[3][i*w+j]+=dpre*previous[j];
                    grads[2][i*w+j]+=dpre*context[t*w+j];
                    previous_grad[j]+=m->wu[i*w+j]*dpre;
                    context_gradient[t*w+j]+=m->wm[i*w+j]*dpre;
                }
            }
            free(qgrad);qgrad=previous_grad;
        }
        for(size_t i=0;i<w;i++)context_gradient[t*w+i]+=qgrad[i];
        free(qgrad);
    }
    {
        float *lambda_m_direct=context_gradient;
        for(size_t reverse=0;reverse<length;reverse++){
            size_t t=length-1-reverse;
            for(size_t i=0;i<w;i++){
                size_t k=t*w+i;reverse_a[reverse*w+i]=t+1<length?gate[(t+1)*w+i]:0.0f;
                reverse_b[reverse*w+i]=lambda_m_direct[k];
            }
        }
        int scanned;
        { float *zero=(float*)calloc(w,sizeof(float));
#ifdef AMA10_ENABLE_OPENCL
          if(opencl)scanned=zero&&ama10_opencl_affine_scan((AMA10OpenCL*)opencl,reverse_a,reverse_b,zero,length,reverse_states);
          else
#endif
          scanned=zero&&affine_scan_cpu(reverse_a,reverse_b,zero,length,w,reverse_states);
          free(zero); }
        if(!scanned)goto fail;
        for(size_t t=0;t<length;t++)for(size_t i=0;i<w;i++)memory_lambda[t*w+i]=reverse_states[(length-1-t)*w+i];
        for(size_t t=0;t<length;t++)for(size_t i=0;i<w;i++){
            size_t k=t*w+i;reverse_a[(length-1-t)*w+i]=t+1<length?gate[(t+1)*w+i]:0.0f;
            reverse_b[(length-1-t)*w+i]=context_gradient[k]+(1.0f-gate[k])*memory_lambda[k];
        }
        { float *zero=(float*)calloc(w,sizeof(float));
#ifdef AMA10_ENABLE_OPENCL
          if(opencl)scanned=zero&&ama10_opencl_affine_scan((AMA10OpenCL*)opencl,reverse_a,reverse_b,zero,length,reverse_states);
          else
#endif
          scanned=zero&&affine_scan_cpu(reverse_a,reverse_b,zero,length,w,reverse_states);
          free(zero); }
        if(!scanned)goto fail;
        for(size_t t=0;t<length;t++)for(size_t i=0;i<w;i++)fast_lambda[t*w+i]=reverse_states[(length-1-t)*w+i];
    }
#ifdef AMA10_ENABLE_OPENCL
    if(gpu_batched_head){
        if(!ama10_opencl_affine_backward((AMA10OpenCL*)opencl,m,x,z,gate,candidate,fast,slow,
            memory_lambda,fast_lambda,state?state->fast:NULL,state?state->slow:NULL,length,
            grads[0],grads[1],grads[4],grads[5],x))goto fail;
        for(size_t time=0;time<length;time++){
            int use_feature=input_features&&(!feature_mask||feature_mask[time]);
            for(size_t col=0;col<e;col++){
                float dx=x[time*e+col];
                if(use_feature)feature_gradients[time*e+col]+=dx;
                else grads[6][col*v+input[time]]+=dx;
            }
        }
    }else
#endif
    for(size_t t=length;t>0;t--){
        size_t time=t-1;
        for(size_t j=0;j<w;j++)z_gradient[j]=0.0f;
        for(size_t i=0;i<w;i++){
            size_t k=time*w+i;
            float previous_fast=time?fast[(time-1)*w+i]:(state?state->fast[i]:0.0f);
            float previous_slow=time?slow[(time-1)*w+i]:(state?state->slow[i]:0.0f);
            float da=fast_lambda[k]*previous_fast+memory_lambda[k]*(previous_slow-fast[k]);
            float d_candidate_pre=fast_lambda[k]*(1.0f-candidate[k]*candidate[k]);
            float d_gate_pre=da*gate[k]*(1.0f-gate[k]);
            grads[4][i]+=d_candidate_pre;grads[5][i]+=d_gate_pre;
            for(size_t j=0;j<w;j++)grads[1][i*w+j]+=d_gate_pre*z[time*w+j];
            z_gradient[i]=d_candidate_pre;context_gradient[k]=d_gate_pre;
        }
        for(size_t i=0;i<w;i++)for(size_t j=0;j<w;j++)
            z_gradient[j]+=m->wh[i*w+j]*context_gradient[time*w+i];
        int use_feature=input_features&&(!feature_mask||feature_mask[time]);
        for(size_t j=0;j<w;j++)for(size_t col=0;col<e;col++)
            grads[0][j*e+col]+=z_gradient[j]*x[time*e+col];
        for(size_t col=0;col<e;col++){
            double dx=0.0;for(size_t row=0;row<w;row++)dx+=(double)m->wx[row*e+col]*z_gradient[row];
            if(use_feature)feature_gradients[time*e+col]+=(float)dx;
            else grads[6][col*v+input[time]]+=(float)dx;
        }
    }
    for(size_t i=0;i<nx&&feature_gradients;i++)feature_gradients[i]/=(float)supervised;
    loss/=(float)supervised;
    if(state){memcpy(state->fast,fast+(length-1)*w,w*sizeof(float));memcpy(state->slow,slow+(length-1)*w,w*sizeof(float));state->steps+=total_depth;}
    if(mean_depth)*mean_depth=(float)total_depth/(float)length;
    if(accumulator){
        for(size_t k=0;k<AFFINE_PARAM_COUNT;k++)for(size_t i=0;i<sizes[k];i++)accumulator->values[k][i]+=grads[k][i];
        accumulator->tokens+=(uint64_t)supervised;
    }else{
        AMA10Gradients update={0};update.values=grads;update.parameter_count=AFFINE_PARAM_COUNT;
        update.width=w;update.input_width=e;update.vocabulary=v;update.tokens=(uint64_t)supervised;
        if(!ama10_optimizer_apply(m,opt,&update,learning_rate,weight_decay))goto fail;
    }
    if(mean_loss)*mean_loss=loss;
    for(size_t k=0;k<AFFINE_PARAM_COUNT;k++)free(grads[k]);
    free(x);free(z);free(gate);free(candidate);free(fast);free(slow);free(context);free(reason);free(logits);free(errors);
    free(context_gradient);free(memory_lambda);free(fast_lambda);free(reverse_a);free(reverse_b);free(reverse_states);free(z_gradient);free(used_depth);
    free(gpu_hidden);free(gpu_hidden_gradient);free(gpu_head_gradient);
    free(gpu_reason_states);
    return 1;
fail:
    for(size_t k=0;k<AFFINE_PARAM_COUNT;k++)free(grads[k]);
    free(x);free(z);free(gate);free(candidate);free(fast);free(slow);free(context);free(reason);free(logits);free(errors);
    free(context_gradient);free(memory_lambda);free(fast_lambda);free(reverse_a);free(reverse_b);free(reverse_states);free(z_gradient);free(used_depth);
    free(gpu_hidden);free(gpu_hidden_gradient);free(gpu_head_gradient);
    free(gpu_reason_states);
    return 0;
}
