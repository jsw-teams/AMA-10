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
    next_slow[i]=0.995f*slow[i]+0.005f*h;
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

__kernel void ama10_adamw_update(
    __global const float *parameter,__global const float *first,__global const float *second,
    __global const float *gradient,__global float *next_parameter,__global float *next_first,
    __global float *next_second,const uint count,const float learning_rate,const float weight_decay,
    const float correction1,const float correction2)
{
    const uint i=(uint)get_global_id(0);if(i>=count)return;
    const float g=gradient[i];const float m=0.9f*first[i]+0.1f*g;
    const float v=0.999f*second[i]+0.001f*g*g;
    const float update=(m/correction1)/(sqrt(v/correction2)+1.0e-8f);
    const float p=parameter[i];next_parameter[i]=p-learning_rate*(update+weight_decay*p);
    next_first[i]=m;next_second[i]=v;
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
        previous_slow=cache[ama10_training_cache_index(time,layer-1,7,i,depth,width)];
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
                  mprev=cache[ama10_training_cache_index(time,layer-1,7,j,depth,width)];}
        else if(time){hprev=cache[ama10_training_cache_index(time-1,depth-1,6,j,depth,width)];
                      mprev=cache[ama10_training_cache_index(time-1,depth-1,7,j,depth,width)];}
        else{hprev=initial_fast[j];mprev=initial_slow[j];}
        z+=wh[(size_t)i*width+j]*hprev+wm[(size_t)i*width+j]*mprev;
        q+=wu[(size_t)i*width+j]*hprev;
    }
    const float proposal=tanh(z),gate=1.0f/(1.0f+exp(-q));
    const float h=(1.0f-gate)*previous_fast+gate*proposal;
    const float slow=rho*previous_slow+(1.0f-rho)*h;
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
    const float dnew=dh[i]+(apply_hidden?hidden_gradient[(size_t)time*width+i]:0.0f)+(1.0f-rho)*dm[i];
    dpre[i]=dnew*g*(1.0f-p*p);
    dgate[i]=dnew*(p-h)*g*(1.0f-g);
    direct_fast[i]=dnew*(1.0f-g);
    direct_slow[i]=rho*dm[i];
}

__kernel void ama10_cell_backward_accumulate(
    __global const float *wx,__global const float *wh,__global const float *wm,
    __global const float *wu,__global const float *embed,__global const float *dpre,
    __global const float *dgate,__global const float *direct_fast,__global const float *direct_slow,
    __global const float *cache,__global const uint *tokens,
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
        grad_wx[index]+=(dpre[i]+dgate[i])*embed[(size_t)j*vocabulary+token];
    }
    if(index<input_width){
        float dx=0.0f;
        for(uint i=0;i<width;i++)dx+=wx[(size_t)i*input_width+index]*(dpre[i]+dgate[i]);
        grad_embed[(size_t)index*vocabulary+token]+=dx;
    }
}
