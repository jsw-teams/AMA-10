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
    const float later_a=a_in[index];
    const float later_b=b_in[index];
    const float earlier_a=a_in[previous];
    const float earlier_b=b_in[previous];
    a_out[index]=later_a*earlier_a;
    b_out[index]=later_a*earlier_b+later_b;
}
