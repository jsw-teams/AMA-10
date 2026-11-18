__kernel void ama10_recurrent_cell(
    __global const float *wx,__global const float *wh,__global const float *wm,
    __global const float *wu,__global const float *bias,__global const float *gate_bias,
    __global const float *embed,__global const float *fast,__global const float *slow,
    __global float *next_fast,__global float *next_slow,
    const uint width,const uint input_width,const uint vocabulary,const uint token)
{
    const uint i=(uint)get_global_id(0);
    if(i>=width)return;
    float drive=bias[i],gate_logit=gate_bias[i];
    for(uint j=0;j<input_width;j++){
        float projected=wx[(size_t)i*input_width+j]*embed[(size_t)j*vocabulary+token];
        drive+=projected;gate_logit+=projected;
    }
    for(uint j=0;j<width;j++){
        drive+=wh[(size_t)i*width+j]*fast[j]+wm[(size_t)i*width+j]*slow[j];
        gate_logit+=wu[(size_t)i*width+j]*fast[j];
    }
    float proposal=tanh(drive);
    float gate=1.0f/(1.0f+exp(-gate_logit));
    float h=(1.0f-gate)*fast[i]+gate*proposal;
    next_fast[i]=h;
    next_slow[i]=slow[i];
}

__kernel void ama10_affine_scan_compose(
    __global const float *a_in,__global const float *b_in,
    __global float *a_out,__global float *b_out,
    const ulong batch,const ulong length,const ulong width,const ulong offset)
{
    const size_t index=get_global_id(0);
    const size_t total=(size_t)batch*(size_t)length*(size_t)width;
    if(index>=total)return;
    const ulong time=((ulong)index/width)%length;
    if(time<offset){a_out[index]=a_in[index];b_out[index]=b_in[index];return;}
    const size_t previous=index-(size_t)offset*(size_t)width;
    const float later_a=a_in[index],later_b=b_in[index];
    const float earlier_a=a_in[previous],earlier_b=b_in[previous];
    a_out[index]=later_a*earlier_a;
    b_out[index]=later_a*earlier_b+later_b;
}

__kernel void ama10_slow_update(
    __global const float *fast,__global const float *slow,__global float *next_slow,
    const uint width)
{
    const uint i=(uint)get_global_id(0);if(i>=width)return;
    next_slow[i]=0.995f*slow[i]+0.005f*fast[i];
}

__kernel void ama10_recurrent_confidence(
    __global const float *fast,__global const float *next_fast,
    __global float *confidence,const uint width)
{
    if(get_global_id(0)!=0)return;
    float activity=0.0f;
    for(uint i=0;i<width;i++)activity+=fabs(next_fast[i]-fast[i]);
    confidence[0]=1.0f/(1.0f+activity/(float)width);
}

__kernel void ama10_recurrent_logits(
    __global const float *head,__global const float *fast,
    __global float *logits,const uint width,const uint vocabulary)
{
    const uint k=(uint)get_global_id(0);
    if(k>=vocabulary)return;
    float value=0.0f;
    for(uint j=0;j<width;j++)value+=head[(size_t)k*width+j]*fast[j];
    logits[k]=value;
}

__kernel void ama10_project_token(__global const float *wx,__global const float *embed,
    __global float *projected,const uint input_width,const uint vocabulary,const uint token,
    const uint width){
    const uint i=(uint)get_global_id(0);if(i>=width)return;
    float value=0.0f;for(uint j=0;j<input_width;j++)value+=wx[(size_t)i*input_width+j]*embed[(size_t)j*vocabulary+token];
    projected[i]=value;
}
__kernel void ama10_project_features(__global const float *wx,__global const float *input,
    __global float *projected,const uint input_width,const uint width){
    const uint i=(uint)get_global_id(0);if(i>=width)return;
    float value=0.0f;for(uint j=0;j<input_width;j++)value+=wx[(size_t)i*input_width+j]*input[j];
    projected[i]=value;
}
__kernel void ama10_affine_state_update(__global const float *wh,__global const float *bias,
    __global const float *gate_bias,__global const float *projected,__global const float *fast,
    __global const float *slow,__global float *next_fast,__global float *next_slow,const uint width){
    const uint i=(uint)get_global_id(0);if(i>=width)return;
    float gate_logit=gate_bias[i];for(uint j=0;j<width;j++)gate_logit+=wh[(size_t)i*width+j]*projected[j];
    float gate=1.0f/(1.0f+exp(-gate_logit));float candidate=tanh(projected[i]+bias[i]);
    float updated_fast=gate*fast[i]+candidate;
    next_fast[i]=updated_fast;next_slow[i]=gate*slow[i]+(1.0f-gate)*updated_fast;
}

__kernel void ama10_affine_project_sequence(__global const float *wx,
    __global const float *inputs,__global float *projected,const uint length,
    const uint input_width,const uint width){
    const uint t=(uint)get_group_id(0),lane=(uint)get_local_id(0);
    const uint i=(uint)get_group_id(1)*16u+(uint)get_local_id(1);
    const uint slot=(uint)get_local_id(1)*16u+lane;
    __local float partial[256];
    float value=0.0f;
    if(t<length&&i<width)for(uint j=lane;j<input_width;j+=16u)
        value+=wx[(size_t)i*input_width+j]*inputs[(size_t)t*input_width+j];
    partial[slot]=value;barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=8u;stride;stride>>=1u){
        if(lane<stride)partial[slot]+=partial[slot+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lane==0u&&t<length&&i<width)projected[(size_t)t*width+i]=partial[slot];
}

__kernel void ama10_affine_gate_sequence(__global const float *wh,
    __global const float *bias,__global const float *gate_bias,
    __global const float *projected,__global float *gates,__global float *candidates,
    const uint length,const uint width){
    const uint t=(uint)get_group_id(0),lane=(uint)get_local_id(0);
    const uint i=(uint)get_group_id(1)*16u+(uint)get_local_id(1);
    const uint slot=(uint)get_local_id(1)*16u+lane;
    __local float partial[256];
    float value=0.0f;
    if(t<length&&i<width)for(uint j=lane;j<width;j+=16u)
        value+=wh[(size_t)i*width+j]*projected[(size_t)t*width+j];
    partial[slot]=value;barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=8u;stride;stride>>=1u){
        if(lane<stride)partial[slot]+=partial[slot+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lane==0u&&t<length&&i<width){
        const size_t index=(size_t)t*width+i;
        gates[index]=1.0f/(1.0f+exp(-(gate_bias[i]+partial[slot])));
        candidates[index]=tanh(projected[index]+bias[i]);
    }
}

__kernel void ama10_affine_reason_sequence(__global const float *wu,
    __global const float *wm,__global const float *context,__global float *states,
    const uint length,const uint width,const uint depth,const uint layer){
    const uint t=(uint)get_group_id(0),lane=(uint)get_local_id(0);
    const uint i=(uint)get_group_id(1)*16u+(uint)get_local_id(1);
    const uint slot=(uint)get_local_id(1)*16u+lane;
    __local float partial[256];
    float value=0.0f;
    if(t<length&&i<width){
        const size_t previous=((size_t)t*depth+layer-1u)*width;
        for(uint j=lane;j<width;j+=16u){
            const float q=layer==0u?context[(size_t)t*width+j]:states[previous+j];
            value+=wu[(size_t)i*width+j]*q+wm[(size_t)i*width+j]*context[(size_t)t*width+j];
        }
    }
    partial[slot]=value;barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=8u;stride;stride>>=1u){
        if(lane<stride)partial[slot]+=partial[slot+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lane==0u&&t<length&&i<width)
        states[((size_t)t*depth+layer)*width+i]=tanh(partial[slot]);
}
__kernel void ama10_state_context(__global const float *fast,__global const float *slow,
    __global float *context,const uint width){
    const uint i=(uint)get_global_id(0);if(i<width)context[i]=fast[i]+slow[i];
}
__kernel void ama10_reason_step(__global const float *wu,__global const float *wm,
    __global const float *previous,__global const float *context,__global float *next,const uint width){
    const uint i=(uint)get_global_id(0);if(i>=width)return;
    float value=0.0f;for(uint j=0;j<width;j++)value+=wu[(size_t)i*width+j]*previous[j]+wm[(size_t)i*width+j]*context[j];
    next[i]=tanh(value);
}

__kernel void ama10_codec_encode(__global const float *x,__global const float *encoder,
    __global const float *bias,__global float *latent,const uint samples,const uint input_width,const uint latent_width){
    const uint index=(uint)get_global_id(0);if(index>=samples*latent_width)return;
    const uint n=index/latent_width,l=index%latent_width;float value=bias[l];
    for(uint j=0;j<input_width;j++)value+=encoder[(size_t)l*input_width+j]*x[(size_t)n*input_width+j];
    latent[index]=tanh(value);
}
__kernel void ama10_codec_decode(__global const float *latent,__global const float *decoder,
    __global const float *bias,__global const float *x,__global float *delta,const uint samples,
    const uint input_width,const uint latent_width){
    const uint index=(uint)get_global_id(0);if(index>=samples*input_width)return;
    const uint n=index/input_width,j=index%input_width;float value=bias[j];
    for(uint l=0;l<latent_width;l++)value+=decoder[(size_t)j*latent_width+l]*latent[(size_t)n*latent_width+l];
    float y=1.0f/(1.0f+exp(-value));delta[index]=y-x[index];
}
__kernel void ama10_codec_hidden_delta(__global const float *latent,__global const float *decoder,
    __global const float *delta,__global float *hidden_delta,const uint samples,
    const uint input_width,const uint latent_width){
    const uint index=(uint)get_global_id(0);if(index>=samples*latent_width)return;
    const uint n=index/latent_width,l=index%latent_width;float value=0.0f;
    for(uint j=0;j<input_width;j++)value+=decoder[(size_t)j*latent_width+l]*delta[(size_t)n*input_width+j];
    float h=latent[index];hidden_delta[index]=value*(1.0f-h*h);
}
__kernel void ama10_codec_update_decoder(__global float *decoder,__global float *bias,
    __global const float *latent,__global const float *delta,const uint samples,
    const uint input_width,const uint latent_width,const float learning_rate){
    const uint index=(uint)get_global_id(0);if(index>=input_width*(latent_width+1))return;
    const uint j=index/(latent_width+1),l=index%(latent_width+1);float gradient=0.0f;
    for(uint n=0;n<samples;n++)gradient+=l==latent_width?delta[(size_t)n*input_width+j]:
        delta[(size_t)n*input_width+j]*latent[(size_t)n*latent_width+l];
    gradient/=samples;
    if(l==latent_width)bias[j]-=learning_rate*gradient;else decoder[(size_t)j*latent_width+l]-=learning_rate*gradient;
}
__kernel void ama10_codec_update_encoder(__global float *encoder,__global float *bias,
    __global const float *x,__global const float *hidden_delta,const uint samples,
    const uint input_width,const uint latent_width,const float learning_rate){
    const uint index=(uint)get_global_id(0);if(index>=latent_width*(input_width+1))return;
    const uint l=index/(input_width+1),j=index%(input_width+1);float gradient=0.0f;
    for(uint n=0;n<samples;n++)gradient+=j==input_width?hidden_delta[(size_t)n*latent_width+l]:
        hidden_delta[(size_t)n*latent_width+l]*x[(size_t)n*input_width+j];
    gradient/=samples;
    if(j==input_width)bias[l]-=learning_rate*gradient;else encoder[(size_t)l*input_width+j]-=learning_rate*gradient;
}

__kernel void ama10_adamw_update(
    __global const float *parameter,__global const float *first,__global const float *second,
    __global const float *gradient,__global float *next_parameter,__global float *next_first,
    __global float *next_second,const uint count,const float learning_rate,const float weight_decay,
    const float correction1,const float correction2,const float gradient_scale)
{
    const uint i=(uint)get_global_id(0);if(i>=count)return;
    const float g=gradient[i]*gradient_scale;const float m=0.9f*first[i]+0.1f*g;
    const float v=0.999f*second[i]+0.001f*g*g;
    const float update=(m/correction1)/(sqrt(v/correction2)+1.0e-8f);
    const float p=parameter[i];next_parameter[i]=p-learning_rate*(update+weight_decay*p);
    next_first[i]=m;next_second[i]=v;
}

__kernel void ama10_gradient_pair_stats(
    __global const float *task_a,__global const float *task_b,
    __global float *partials,const uint count)
{
    const uint i=(uint)get_global_id(0),lane=(uint)get_local_id(0),group=(uint)get_group_id(0);
    __local float dots[256],norm_a[256],norm_b[256];
    float a=0.0f,b=0.0f;
    if(i<count){a=task_a[i];b=task_b[i];}
    dots[lane]=a*b;norm_a[lane]=a*a;norm_b[lane]=b*b;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=128u;stride;stride>>=1u){
        if(lane<stride){dots[lane]+=dots[lane+stride];norm_a[lane]+=norm_a[lane+stride];norm_b[lane]+=norm_b[lane+stride];}
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lane==0u){partials[group*3u]=dots[0];partials[group*3u+1u]=norm_a[0];partials[group*3u+2u]=norm_b[0];}
}

__kernel void ama10_gradient_pair_finalize(__global const float *partials,
    __global float *stats,const uint groups)
{
    const uint lane=(uint)get_local_id(0);
    __local float dots[256],norm_a[256],norm_b[256];
    float dot=0.0f,a=0.0f,b=0.0f;
    for(uint g=lane;g<groups;g+=256u){dot+=partials[g*3u];a+=partials[g*3u+1u];b+=partials[g*3u+2u];}
    dots[lane]=dot;norm_a[lane]=a;norm_b[lane]=b;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=128u;stride;stride>>=1u){
        if(lane<stride){dots[lane]+=dots[lane+stride];norm_a[lane]+=norm_a[lane+stride];norm_b[lane]+=norm_b[lane+stride];}
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lane==0u){
        const float d=dots[0],na=norm_a[0],nb=norm_b[0];
        const float denom=na+nb-2.0f*d;
        float weight=denom>1.0e-30f?(nb-d)/denom:0.5f;
        weight=fmin(0.9f,fmax(0.1f,weight));
        stats[0]=d;stats[1]=na;stats[2]=nb;stats[3]=weight;
        stats[4]=na>0.0f&&nb>0.0f?d/(sqrt(na)*sqrt(nb)):0.0f;
    }
}

__kernel void ama10_gradient_pair_merge(__global const float *task_a,
    __global const float *task_b,__global const float *stats,
    __global float *merged,__global float *norm_partials,
    const uint count)
{
    const uint i=(uint)get_global_id(0),lane=(uint)get_local_id(0),group=(uint)get_group_id(0);
    __local float norm2[256];
    float value=0.0f;
    if(i<count){const float a=task_a[i],b=task_b[i];
        const float weight=stats[3];value=weight*a+(1.0f-weight)*b;merged[i]=value;}
    norm2[lane]=value*value;barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=128u;stride;stride>>=1u){
        if(lane<stride)norm2[lane]+=norm2[lane+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lane==0u)norm_partials[group]=norm2[0];
}

__kernel void ama10_sequence_logits(__global const float *head,__global const float *hidden,
    __global float *logits,const uint length,const uint width,const uint vocabulary){
    const uint index=(uint)get_global_id(0);if(index>=length*vocabulary)return;
    const uint t=index/vocabulary,k=index%vocabulary;float z=0.0f;
    for(uint j=0;j<width;j++)z+=head[(size_t)k*width+j]*hidden[(size_t)t*width+j];
    logits[index]=z;
}
__kernel void ama10_head_gradient(__global const float *head,__global const float *hidden,
    __global const float *errors,__global float *gradient,const uint length,
    const uint width,const uint vocabulary){
    const uint index=(uint)get_global_id(0);if(index>=vocabulary*width)return;
    const uint k=index/width,j=index%width;float sum=0.0f;
    for(uint t=0;t<length;t++)sum+=errors[(size_t)t*vocabulary+k]*hidden[(size_t)t*width+j];
    gradient[index]=sum;
}
__kernel void ama10_hidden_gradient(__global const float *head,__global const float *hidden,
    __global const float *errors,__global float *gradient,const uint length,
    const uint width,const uint vocabulary){
    const uint index=(uint)get_global_id(0);if(index>=length*width)return;
    const uint t=index/width,j=index%width;float sum=0.0f;
    for(uint k=0;k<vocabulary;k++)sum+=head[(size_t)k*width+j]*errors[(size_t)t*vocabulary+k];
    gradient[index]=sum;
}

/* Backpropagate through the fixed-depth shared reasoner over a whole sequence.
   Each token is independent here; sequence parallelism is exposed over tokens,
   while width-wise matrix reductions execute on the device. */
__kernel void ama10_reason_backward_derivative(__global const float *states,
    __global const float *q_gradient,__global float *pre_gradient,
    const uint length,const uint width,const uint depth,const uint layer){
    const uint index=(uint)get_global_id(0);if(index>=length*width)return;
    const uint t=index/width,i=index%width;
    const size_t base=((size_t)t*(depth+1)+layer)*width;
    const float current=states[base+width+i],q=q_gradient[index];
    pre_gradient[index]=q*(1.0f-current*current);
}
__kernel void ama10_reason_backward_propagate(__global const float *wu,
    __global const float *wm,__global const float *pre_gradient,
    __global float *previous_q,__global float *context_gradient,
    const uint length,const uint width){
    const uint index=(uint)get_global_id(0);if(index>=length*width)return;
    const uint t=index/width,j=index%width;float dq=0.0f,dc=0.0f;
    for(uint i=0;i<width;i++){
        const float d=pre_gradient[(size_t)t*width+i];
        dq+=wu[(size_t)i*width+j]*d;dc+=wm[(size_t)i*width+j]*d;
    }
    previous_q[index]=dq;context_gradient[index]+=dc;
}
__kernel void ama10_reason_backward_weight_gradient(__global const float *states,
    __global const float *pre_gradient,
    __global float *grad_wu,__global float *grad_wm,
    const uint length,const uint width,const uint depth,const uint layer){
    const uint index=(uint)get_global_id(0);if(index>=width*width)return;
    const uint i=index/width,j=index%width;float gu=0.0f,gm=0.0f;
    for(uint t=0;t<length;t++){
        const size_t base=((size_t)t*(depth+1)+layer)*width;
        const float d=pre_gradient[(size_t)t*width+i];
        gu+=d*states[base+j];
        gm+=d*states[((size_t)t*(depth+1))*width+j];
    }
    grad_wu[index]+=gu;grad_wm[index]+=gm;
}

__kernel void ama10_affine_backward_deltas(__global const float *gate,
    __global const float *candidate,__global const float *fast,__global const float *slow,
    __global const float *memory_lambda,__global const float *fast_lambda,
    __global const float *initial_fast,__global const float *initial_slow,
    __global float *candidate_pre,__global float *gate_pre,
    const uint length,const uint width){
    const uint index=(uint)get_global_id(0);if(index>=length*width)return;
    const uint t=index/width,i=index%width;
    const float previous_fast=t?fast[index-width]:initial_fast[i];
    const float previous_slow=t?slow[index-width]:initial_slow[i];
    const float a=fast_lambda[index]*previous_fast+
        memory_lambda[index]*(previous_slow-fast[index]);
    const float g=gate[index],c=candidate[index];
    candidate_pre[index]=fast_lambda[index]*(1.0f-c*c);
    gate_pre[index]=a*g*(1.0f-g);
}
__kernel void ama10_affine_backward_z_gradient(__global const float *wh,
    __global const float *candidate_pre,__global const float *gate_pre,
    __global float *z_gradient,const uint length,const uint width){
    const uint index=(uint)get_global_id(0);if(index>=length*width)return;
    const uint t=index/width,j=index%width;float value=candidate_pre[index];
    for(uint i=0;i<width;i++)value+=wh[(size_t)i*width+j]*gate_pre[(size_t)t*width+i];
    z_gradient[index]=value;
}
__kernel void ama10_affine_backward_recurrent_gradient(__global const float *z,
    __global const float *gate_pre,__global float *wh_gradient,
    const uint length,const uint width){
    const uint index=(uint)get_global_id(0);if(index>=width*width)return;
    const uint i=index/width,j=index%width;float value=0.0f;
    for(uint t=0;t<length;t++)value+=gate_pre[(size_t)t*width+i]*z[(size_t)t*width+j];
    wh_gradient[index]=value;
}
__kernel void ama10_affine_backward_bias_gradient(__global const float *candidate_pre,
    __global const float *gate_pre,__global float *bias_gradient,
    __global float *gate_bias_gradient,const uint length,const uint width){
    const uint i=(uint)get_global_id(0);if(i>=width)return;
    float db=0.0f,dg=0.0f;for(uint t=0;t<length;t++){
        db+=candidate_pre[(size_t)t*width+i];dg+=gate_pre[(size_t)t*width+i];}
    bias_gradient[i]=db;gate_bias_gradient[i]=dg;
}
__kernel void ama10_affine_backward_input_gradient(__global const float *wx,
    __global const float *z_gradient,__global float *input_gradient,
    const uint length,const uint width,const uint input_width){
    const uint index=(uint)get_global_id(0);if(index>=length*input_width)return;
    const uint t=index/input_width,j=index%input_width;float value=0.0f;
    for(uint i=0;i<width;i++)value+=wx[(size_t)i*input_width+j]*z_gradient[(size_t)t*width+i];
    input_gradient[index]=value;
}
__kernel void ama10_affine_backward_input_weight_gradient(__global const float *x,
    __global const float *z_gradient,__global float *wx_gradient,
    const uint length,const uint width,const uint input_width){
    const uint index=(uint)get_global_id(0);if(index>=width*input_width)return;
    const uint i=index/input_width,j=index%input_width;float value=0.0f;
    for(uint t=0;t<length;t++)value+=z_gradient[(size_t)t*width+i]*x[(size_t)t*input_width+j];
    wx_gradient[index]=value;
}

inline size_t ama10_training_cache_index(const uint t,const uint d,const uint field,
                                         const uint i,const uint depth,const uint width){
    return ((((size_t)t*depth+d)*8+field)*width)+i;
}

__kernel void ama10_training_forward_cell(
    __global const float *wx,__global const float *wh,__global const float *wm,
    __global const float *wu,__global const float *bias,__global const float *gate_bias,
    __global const float *embed,__global const uint *tokens,
    __global const float *initial_fast,__global const float *initial_slow,
    __global float *cache,__global float *sequence_hidden,
    const uint width,const uint input_width,const uint vocabulary,const uint depth,
    const uint time,const uint layer,const float rho){
    const uint i=(uint)get_global_id(0);if(i>=width)return;
    const uint token=tokens[time];float previous_fast,previous_slow;
    if(layer){
        previous_fast=cache[ama10_training_cache_index(time,layer-1,6,i,depth,width)];
        previous_slow=cache[ama10_training_cache_index(time,layer-1,1,i,depth,width)];
    }else if(time){
        previous_fast=cache[ama10_training_cache_index(time-1,depth-1,6,i,depth,width)];
        previous_slow=cache[ama10_training_cache_index(time-1,depth-1,7,i,depth,width)];
    }else{previous_fast=initial_fast[i];previous_slow=initial_slow[i];}
    float z=bias[i],q=gate_bias[i];
    for(uint j=0;j<input_width;j++){
        const float projected=wx[(size_t)i*input_width+j]*embed[(size_t)j*vocabulary+token];
        z+=projected;q+=projected;
    }
    for(uint j=0;j<width;j++){
        float hprev,mprev;
        if(layer){hprev=cache[ama10_training_cache_index(time,layer-1,6,j,depth,width)];
                  mprev=cache[ama10_training_cache_index(time,layer-1,1,j,depth,width)];}
        else if(time){hprev=cache[ama10_training_cache_index(time-1,depth-1,6,j,depth,width)];
                      mprev=cache[ama10_training_cache_index(time-1,depth-1,7,j,depth,width)];}
        else{hprev=initial_fast[j];mprev=initial_slow[j];}
        z+=wh[(size_t)i*width+j]*hprev+wm[(size_t)i*width+j]*mprev;
        q+=wu[(size_t)i*width+j]*hprev;
    }
    const float proposal=tanh(z),gate=1.0f/(1.0f+exp(-q));
    const float h=(1.0f-gate)*previous_fast+gate*proposal;
    const float slow=layer+1==depth?rho*previous_slow+(1.0f-rho)*h:previous_slow;
    cache[ama10_training_cache_index(time,layer,0,i,depth,width)]=previous_fast;
    cache[ama10_training_cache_index(time,layer,1,i,depth,width)]=previous_slow;
    cache[ama10_training_cache_index(time,layer,2,i,depth,width)]=z;
    cache[ama10_training_cache_index(time,layer,3,i,depth,width)]=proposal;
    cache[ama10_training_cache_index(time,layer,4,i,depth,width)]=q;
    cache[ama10_training_cache_index(time,layer,5,i,depth,width)]=gate;
    cache[ama10_training_cache_index(time,layer,6,i,depth,width)]=h;
    cache[ama10_training_cache_index(time,layer,7,i,depth,width)]=slow;
    if(layer+1==depth)sequence_hidden[(size_t)time*width+i]=h;
}

/* Fuse all recurrent microsteps for one token into one workgroup dispatch.
   A workgroup owns the token's width-sized state and uses local memory to
   exchange each microstep, avoiding one kernel launch per microstep. */
__kernel void ama10_training_forward_fused(
    __global const float *wx,__global const float *wh,__global const float *wm,
    __global const float *wu,__global const float *bias,__global const float *gate_bias,
    __global const float *embed,__global const float *input_features,
    __global const uchar *feature_mask,__global const uint *tokens,
    __global const float *initial_fast,__global const float *initial_slow,
    __global float *cache,__global float *sequence_hidden,
    const uint width,const uint input_width,const uint vocabulary,const uint depth,
    const uint time,const float rho,__local float *fast,__local float *slow,
    __local float *next_fast)
{
    const uint i=(uint)get_local_id(0);
    const uint token=tokens[time];
    if(i<width){
        if(time){
            fast[i]=cache[ama10_training_cache_index(time-1,depth-1,6,i,depth,width)];
            slow[i]=cache[ama10_training_cache_index(time-1,depth-1,7,i,depth,width)];
        }else{fast[i]=initial_fast[i];slow[i]=initial_slow[i];}
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __local float *current=fast,*following=next_fast;
    for(uint layer=0;layer<depth;layer++){
        if(i<width){
            float projected=0.0f;
            for(uint j=0;j<input_width;j++){
                float x=feature_mask[time]?input_features[(size_t)time*input_width+j]:embed[(size_t)j*vocabulary+token];
                projected+=wx[(size_t)i*input_width+j]*x;
            }
            float z=bias[i]+projected,q=gate_bias[i]+projected;
            for(uint j=0;j<width;j++){
                z+=wh[(size_t)i*width+j]*current[j]+wm[(size_t)i*width+j]*slow[j];
                q+=wu[(size_t)i*width+j]*current[j];
            }
            float proposal=tanh(z),gate=1.0f/(1.0f+exp(-q));
            float h=(1.0f-gate)*current[i]+gate*proposal;
            size_t base=ama10_training_cache_index(time,layer,0,i,depth,width);
            cache[base]=current[i];cache[base+width]=slow[i];
            cache[base+2*width]=z;cache[base+3*width]=proposal;
            cache[base+4*width]=q;cache[base+5*width]=gate;
            cache[base+6*width]=h;following[i]=h;
            cache[base+7*width]=layer+1==depth?rho*slow[i]+(1.0f-rho)*h:slow[i];
        }
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
        __local float *swap=current;current=following;following=swap;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(i<width)sequence_hidden[(size_t)time*width+i]=current[i];
}

__kernel void ama10_cell_backward_derivatives(
    __global const float *dh,__global const float *dm,__global const float *cache,
    __global const float *hidden_gradient,__global float *dpre,__global float *dgate,
    __global float *direct_fast,__global float *direct_slow,
    const uint width,const uint depth,const uint time,const uint layer,
    const uint apply_hidden,const float rho){
    const uint i=(uint)get_global_id(0);if(i>=width)return;
    const float h=cache[ama10_training_cache_index(time,layer,0,i,depth,width)];
    const float p=cache[ama10_training_cache_index(time,layer,3,i,depth,width)];
    const float g=cache[ama10_training_cache_index(time,layer,5,i,depth,width)];
    const float dnew=dh[i]+(apply_hidden?hidden_gradient[(size_t)time*width+i]:0.0f)+
        (apply_hidden?(1.0f-rho)*dm[i]:0.0f);
    dpre[i]=dnew*g*(1.0f-p*p);
    dgate[i]=dnew*(p-h)*g*(1.0f-g);
    direct_fast[i]=dnew*(1.0f-g);
    direct_slow[i]=apply_hidden?rho*dm[i]:dm[i];
}

__kernel void ama10_cell_backward_accumulate(
    __global const float *wx,__global const float *wh,__global const float *wm,
    __global const float *wu,__global const float *embed,__global const float *dpre,
    __global const float *dgate,__global const float *direct_fast,__global const float *direct_slow,
    __global const float *cache,__global const uint *tokens,
    __global const float *input_features,__global const uchar *feature_mask,
    __global float *feature_gradients,
    __global float *next_dh,__global float *next_dm,
    __global float *grad_wx,__global float *grad_wh,__global float *grad_wm,
    __global float *grad_wu,__global float *grad_bias,__global float *grad_gate_bias,
    __global float *grad_embed,const uint width,const uint input_width,
    const uint vocabulary,const uint depth,const uint time,const uint layer){
    const uint index=(uint)get_global_id(0);
    const uint token=tokens[time];
    if(index<width){
        float fast=direct_fast[index],slow=direct_slow[index];
        for(uint i=0;i<width;i++){
            fast+=wh[(size_t)i*width+index]*dpre[i]+wu[(size_t)i*width+index]*dgate[i];
            slow+=wm[(size_t)i*width+index]*dpre[i];
        }
        next_dh[index]=fast;next_dm[index]=slow;
        grad_bias[index]+=dpre[index];grad_gate_bias[index]+=dgate[index];
    }
    if(index<(size_t)width*width){
        const uint i=index/width,j=index%width;
        grad_wh[index]+=dpre[i]*cache[ama10_training_cache_index(time,layer,0,j,depth,width)];
        grad_wm[index]+=dpre[i]*cache[ama10_training_cache_index(time,layer,1,j,depth,width)];
        grad_wu[index]+=dgate[i]*cache[ama10_training_cache_index(time,layer,0,j,depth,width)];
    }
    if(index<(size_t)width*input_width){
        const uint i=index/input_width,j=index%input_width;
        float x=feature_mask[time]?input_features[(size_t)time*input_width+j]:embed[(size_t)j*vocabulary+token];
        grad_wx[index]+=(dpre[i]+dgate[i])*x;
    }
    if(index<input_width){
        float dx=0.0f;
        for(uint i=0;i<width;i++)dx+=wx[(size_t)i*input_width+index]*(dpre[i]+dgate[i]);
        if(feature_mask[time])feature_gradients[(size_t)time*input_width+index]+=dx;
        else grad_embed[(size_t)index*vocabulary+token]+=dx;
    }
}
