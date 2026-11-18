#include "ama10_imggen.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t hash_data(uint64_t h,const void *data,size_t n){
    const unsigned char *p=(const unsigned char*)data;
    for(size_t i=0;i<n;i++){h^=p[i];h*=UINT64_C(1099511628211);}return h;
}
void ama10_image_codebook_free(AMA10ImageCodebook *b){if(b){free(b->centers);free(b->residual_centers);free(b->context_centers);free(b->encoder);free(b->encoder_bias);free(b->decoder);free(b->decoder_bias);memset(b,0,sizeof(*b));}}
static int fit_centers(const float*x,size_t n,size_t d,uint32_t k,float*centers){
    if(!x||!centers||!n||!d||!k||k>n||d>SIZE_MAX/k||d*k>SIZE_MAX/sizeof(float))return 0;
    float*sums=(float*)calloc((size_t)k*d,sizeof(float));uint32_t*counts=(uint32_t*)calloc(k,sizeof(uint32_t));
    if(!sums||!counts){free(sums);free(counts);return 0;}
    for(uint32_t c=0;c<k;c++)memcpy(centers+(size_t)c*d,x+((size_t)c*n/k)*d,d*sizeof(float));
    for(int iter=0;iter<16;iter++){
        memset(sums,0,(size_t)k*d*sizeof(float));memset(counts,0,(size_t)k*sizeof(uint32_t));
        for(size_t i=0;i<n;i++){
            uint32_t best=0;double bd=INFINITY;
            for(uint32_t c=0;c<k;c++){double dist=0;const float*m=centers+(size_t)c*d;
                for(size_t j=0;j<d;j++){double q=(double)x[i*d+j]-m[j];dist+=q*q;}
                if(dist<bd){bd=dist;best=c;}}
            counts[best]++;for(size_t j=0;j<d;j++)sums[(size_t)best*d+j]+=x[i*d+j];
        }
        for(uint32_t c=0;c<k;c++)if(counts[c])for(size_t j=0;j<d;j++)centers[(size_t)c*d+j]=sums[(size_t)c*d+j]/(float)counts[c];
        else memcpy(centers+(size_t)c*d,x+((size_t)c*n/k)*d,d*sizeof(float));
    }
    free(sums);free(counts);return 1;
}
static uint32_t quantize_centers(const float*centers,size_t d,uint32_t k,const float*x){
    double best=INFINITY;uint32_t which=0;
    for(uint32_t c=0;c<k;c++){double dist=0;const float*m=centers+(size_t)c*d;
        for(size_t j=0;j<d;j++){double q=(double)x[j]-m[j];dist+=q*q;}
        if(dist<best){best=dist;which=c;}}
    return which;
}
static float codec_sigmoid(float x){if(x>=0){float e=expf(-x);return 1.0f/(1.0f+e);}float e=expf(x);return e/(1.0f+e);}
static size_t code_width(const AMA10ImageCodebook*b){return b->latent_width?b->latent_width:b->feature_width;}
static int codec_encode(const AMA10ImageCodebook*b,const float*x,float*z){
    if(!b||!x||!z)return 0;
    if(!b->latent_width){memcpy(z,x,b->feature_width*sizeof(float));return 1;}
    for(size_t l=0;l<b->latent_width;l++){double value=b->encoder_bias[l];for(size_t j=0;j<b->feature_width;j++)value+=(double)b->encoder[l*b->feature_width+j]*x[j];z[l]=tanhf((float)value);}
    return 1;
}
static int codec_decode(const AMA10ImageCodebook*b,const float*z,float*x){
    if(!b||!x||!z)return 0;
    if(!b->latent_width){memcpy(x,z,b->feature_width*sizeof(float));return 1;}
    for(size_t j=0;j<b->feature_width;j++){double value=b->decoder_bias[j];for(size_t l=0;l<b->latent_width;l++)value+=(double)b->decoder[j*b->latent_width+l]*z[l];x[j]=codec_sigmoid((float)value);}
    return 1;
}
#ifdef AMA10_ENABLE_OPENCL
static uint32_t codec_rng(uint32_t *state){*state^=*state<<13;*state^=*state>>17;*state^=*state<<5;return *state;}
static int codec_initialize(AMA10ImageCodebook*b,size_t d){
    if(!b||!d||d>UINT32_MAX)return 0;
    /* Scale the learned bottleneck with both patch dimensionality and the
       available codebook cardinality.  This avoids a global latent-width cap
       that silently discards spatial detail on larger patches. */
    size_t root=(size_t)sqrt((double)d*(double)b->code_count);
    size_t latent=root<d?root:d;if(!latent)latent=1;b->latent_width=(uint32_t)latent;
    if(d>SIZE_MAX/latent||d*latent>SIZE_MAX/sizeof(float))return 0;
    b->encoder=(float*)malloc(latent*d*sizeof(float));b->encoder_bias=(float*)calloc(latent,sizeof(float));
    b->decoder=(float*)malloc(d*latent*sizeof(float));b->decoder_bias=(float*)calloc(d,sizeof(float));
    if(!b->encoder||!b->encoder_bias||!b->decoder||!b->decoder_bias)return 0;
    uint32_t rng=UINT32_C(0x7a4a10c3);float scale=sqrtf(6.0f/(float)(d+latent));
    for(size_t i=0;i<latent*d;i++)b->encoder[i]=(((float)(codec_rng(&rng)&0xffffu)/32767.5f)-1.0f)*scale;
    for(size_t i=0;i<d*latent;i++)b->decoder[i]=(((float)(codec_rng(&rng)&0xffffu)/32767.5f)-1.0f)*scale;
    return 1;
}
static int codebook_finish_fit(AMA10ImageCodebook*b,const float*x,size_t n,size_t d,uint32_t k){
    if(b->latent_width>SIZE_MAX/n||n*b->latent_width>SIZE_MAX/sizeof(float))return 0;
    float*latents=(float*)malloc(n*b->latent_width*sizeof(float));b->centers=(float*)malloc((size_t)k*b->latent_width*sizeof(float));
    if(!latents||!b->centers){free(latents);return 0;}
    for(size_t i=0;i<n;i++)if(!codec_encode(b,x+i*d,latents+i*b->latent_width)){free(latents);return 0;}
    int ok=fit_centers(latents,n,b->latent_width,k,b->centers);free(latents);return ok;
}
#endif
static int codebook_fit_validate(const float*x,size_t n,size_t d,uint32_t k,uint32_t pw,uint32_t ph,
                                 const AMA10ImageCodebook*out){
    return x&&n&&d&&k&&k<=n&&pw&&ph&&d==(size_t)pw*ph*3&&d<=SIZE_MAX/k&&d*k<=SIZE_MAX/sizeof(float)&&out&&!out->centers;
}
int ama10_image_codebook_fit(const float *x,size_t n,size_t d,uint32_t k,
 uint32_t pw,uint32_t ph,uint64_t train_hash,AMA10ImageCodebook *out){
    if(!codebook_fit_validate(x,n,d,k,pw,ph,out))return 0;
    AMA10ImageCodebook b={0};b.patch_width=pw;b.patch_height=ph;b.stride_width=pw;b.stride_height=ph;b.code_count=k;b.feature_width=(uint32_t)d;b.levels=1;b.train_hash=train_hash;
    b.centers=(float*)malloc((size_t)k*d*sizeof(float));
    if(!b.centers||!fit_centers(x,n,d,k,b.centers)){ama10_image_codebook_free(&b);return 0;}
    *out=b;return 1;
}
#ifdef AMA10_ENABLE_OPENCL
int ama10_image_spatial_codec_train_opencl(AMA10ImageCodebook*,const float*,size_t,size_t,const char*);
#endif
int ama10_image_codebook_fit_opencl(const float*x,size_t n,size_t d,uint32_t k,uint32_t pw,uint32_t ph,
                                    uint64_t train_hash,const char*kernel_path,AMA10ImageCodebook*out){
#ifdef AMA10_ENABLE_OPENCL
    if(!codebook_fit_validate(x,n,d,k,pw,ph,out)||!kernel_path)return 0;
    AMA10ImageCodebook b={0};b.patch_width=pw;b.patch_height=ph;b.stride_width=pw;b.stride_height=ph;b.code_count=k;b.feature_width=(uint32_t)d;b.levels=1;b.train_hash=train_hash;
    if(!codec_initialize(&b,d)||!ama10_image_spatial_codec_train_opencl(&b,x,n,d,kernel_path)||!codebook_finish_fit(&b,x,n,d,k)){ama10_image_codebook_free(&b);return 0;}
    *out=b;return 1;
#else
    (void)x;(void)n;(void)d;(void)k;(void)pw;(void)ph;(void)train_hash;(void)kernel_path;(void)out;return 0;
#endif
}
int ama10_image_codebook_fit_residual(const float*x,size_t n,AMA10ImageCodebook*b){
    if(!x||!n||!b||!b->centers||b->residual_centers||b->levels!=1||b->code_count>n||
       !b->feature_width||n>SIZE_MAX/code_width(b)||n*code_width(b)>SIZE_MAX/sizeof(float))return 0;
    size_t d=code_width(b);float*r=(float*)malloc(n*d*sizeof(float)),*z=(float*)malloc(d*sizeof(float));
    float*c=(float*)malloc((size_t)b->code_count*d*sizeof(float));
    if(!r||!c||!z){free(r);free(c);free(z);return 0;}
    for(size_t i=0;i<n;i++){if(!codec_encode(b,x+i*b->feature_width,z)){free(r);free(c);free(z);return 0;}uint32_t q=quantize_centers(b->centers,d,b->code_count,z);const float*m=b->centers+(size_t)q*d;
        for(size_t j=0;j<d;j++)r[i*d+j]=z[j]-m[j];}
    int ok=fit_centers(r,n,d,b->code_count,c);free(r);free(z);
    if(!ok){free(c);return 0;}b->residual_centers=c;b->levels=2;return 1;
}
int ama10_image_codebook_fit_context(const float*x,size_t n,uint32_t k,AMA10ImageCodebook*b){
    if(!x||!n||!b||!b->centers||!k||k>n||b->context_centers||b->context_code_count||
       n>SIZE_MAX/code_width(b)||n*code_width(b)>SIZE_MAX/sizeof(float)||
       (size_t)k>SIZE_MAX/code_width(b)||(size_t)k*code_width(b)>SIZE_MAX/sizeof(float))return 0;
    size_t width=code_width(b);float*c=(float*)malloc((size_t)k*width*sizeof(float)),*latents=(float*)malloc(n*width*sizeof(float));if(!c||!latents){free(c);free(latents);return 0;}
    for(size_t i=0;i<n;i++)if(!codec_encode(b,x+i*b->feature_width,latents+i*width)){free(c);free(latents);return 0;}
    int fitted=fit_centers(latents,n,width,k,c);free(latents);if(!fitted){free(c);return 0;}
    b->context_centers=c;b->context_code_count=k;return 1;
}
int ama10_image_codebook_quantize(const AMA10ImageCodebook *b,const float *x,uint32_t *code){
    if(!b||!b->centers||!x||!code)return 0;
    size_t width=code_width(b);float*z=(float*)malloc(width*sizeof(float));if(!z)return 0;
    int ok=codec_encode(b,x,z);if(ok)*code=quantize_centers(b->centers,width,b->code_count,z)+1;free(z);return ok;
}
int ama10_image_context_quantize(const AMA10ImageCodebook*b,const float*x,uint32_t*code){
    if(!b||!b->centers||!x||!code)return 0;
    if(!b->context_code_count)return ama10_image_codebook_quantize(b,x,code);
    if(!b->context_centers)return 0;
    size_t width=code_width(b);float*z=(float*)malloc(width*sizeof(float));if(!z)return 0;
    int ok=codec_encode(b,x,z);if(ok)*code=quantize_centers(b->context_centers,width,b->context_code_count,z)+1;free(z);return ok;
}
static void append_spatial_neighbor(size_t position,size_t candidate,size_t neighbors[8],size_t*count){
    if(candidate>=position||*count>=8)return;
    for(size_t i=0;i<*count;i++)if(neighbors[i]==candidate)return;
    neighbors[(*count)++]=candidate;
}
size_t ama10_image_spatial_neighbors(size_t position,size_t grid_width,size_t grid_height,size_t neighbors[8]){
    if(!grid_width||!grid_height||!neighbors||grid_width>SIZE_MAX/grid_height||position>=grid_width*grid_height)return 0;
    size_t count=0,x=position%grid_width,y=position/grid_width;
    if(y>=2)append_spatial_neighbor(position,position-2*grid_width,neighbors,&count);
    if(y>=1){
        size_t row=y-1,begin=x>2?x-2:0,end=grid_width-1-x<2?grid_width-1:x+2;
        for(size_t nx=begin;nx<=end;nx++)append_spatial_neighbor(position,row*grid_width+nx,neighbors,&count);
    }
    if(x)append_spatial_neighbor(position,position-1,neighbors,&count);
    if(x>=2)append_spatial_neighbor(position,position-2,neighbors,&count);
    return count;
}
static int checksum(const AMA10ImageCodebook *b,uint64_t *out){
    uint64_t h=UINT64_C(14695981039346656037);h=hash_data(h,&b->patch_width,4);h=hash_data(h,&b->patch_height,4);
    h=hash_data(h,&b->stride_width,4);h=hash_data(h,&b->stride_height,4);
    h=hash_data(h,&b->code_count,4);h=hash_data(h,&b->feature_width,4);h=hash_data(h,&b->train_hash,8);
    size_t n=(size_t)b->code_count*b->feature_width;if(n>SIZE_MAX/sizeof(float))return 0;
    *out=hash_data(h,b->centers,n*sizeof(float));return 1;
}
static int checksum_v1(const AMA10ImageCodebook *b,uint64_t *out){
    uint64_t h=UINT64_C(14695981039346656037);h=hash_data(h,&b->patch_width,4);h=hash_data(h,&b->patch_height,4);
    h=hash_data(h,&b->code_count,4);h=hash_data(h,&b->feature_width,4);h=hash_data(h,&b->train_hash,8);
    size_t n=(size_t)b->code_count*b->feature_width;if(n>SIZE_MAX/sizeof(float))return 0;
    *out=hash_data(h,b->centers,n*sizeof(float));return 1;
}
static int checksum_v3(const AMA10ImageCodebook*b,uint64_t*out){
    if(b->levels<1||b->levels>2||(b->levels==2&&!b->residual_centers)||
       (b->levels==1&&b->residual_centers))return 0;
    uint64_t h=UINT64_C(14695981039346656037);h=hash_data(h,&b->patch_width,4);h=hash_data(h,&b->patch_height,4);
    h=hash_data(h,&b->stride_width,4);h=hash_data(h,&b->stride_height,4);h=hash_data(h,&b->levels,4);
    h=hash_data(h,&b->code_count,4);h=hash_data(h,&b->feature_width,4);h=hash_data(h,&b->train_hash,8);
    size_t n=(size_t)b->code_count*b->feature_width;if(n>SIZE_MAX/sizeof(float))return 0;
    h=hash_data(h,b->centers,n*sizeof(float));if(b->levels==2)h=hash_data(h,b->residual_centers,n*sizeof(float));*out=h;return 1;
}
static int checksum_v4(const AMA10ImageCodebook*b,uint64_t*out){
    if(!b->context_code_count||!b->context_centers||b->context_code_count>256||
       b->levels<1||b->levels>2||(b->levels==2&&!b->residual_centers)||
       (b->levels==1&&b->residual_centers))return 0;
    uint64_t h=UINT64_C(14695981039346656037);h=hash_data(h,&b->patch_width,4);h=hash_data(h,&b->patch_height,4);
    h=hash_data(h,&b->stride_width,4);h=hash_data(h,&b->stride_height,4);h=hash_data(h,&b->levels,4);
    h=hash_data(h,&b->code_count,4);h=hash_data(h,&b->feature_width,4);h=hash_data(h,&b->train_hash,8);
    h=hash_data(h,&b->context_code_count,4);size_t n=(size_t)b->code_count*b->feature_width;
    if(n>SIZE_MAX/sizeof(float)||(size_t)b->context_code_count>SIZE_MAX/b->feature_width||
       (size_t)b->context_code_count*b->feature_width>SIZE_MAX/sizeof(float))return 0;
    h=hash_data(h,b->centers,n*sizeof(float));if(b->levels==2)h=hash_data(h,b->residual_centers,n*sizeof(float));
    h=hash_data(h,b->context_centers,(size_t)b->context_code_count*b->feature_width*sizeof(float));*out=h;return 1;
}
static int checksum_v5(const AMA10ImageCodebook*b,uint64_t*out){
    if(!b->latent_width||!b->encoder||!b->encoder_bias||!b->decoder||!b->decoder_bias||
       (b->context_code_count&&!b->context_centers))return 0;
    if(b->levels<1||b->levels>2||(b->levels==2&&!b->residual_centers)||(b->levels==1&&b->residual_centers)||b->context_code_count>256)return 0;
    uint64_t h=UINT64_C(14695981039346656037);h=hash_data(h,&b->patch_width,4);h=hash_data(h,&b->patch_height,4);
    h=hash_data(h,&b->stride_width,4);h=hash_data(h,&b->stride_height,4);h=hash_data(h,&b->levels,4);
    h=hash_data(h,&b->code_count,4);h=hash_data(h,&b->feature_width,4);h=hash_data(h,&b->latent_width,4);
    h=hash_data(h,&b->train_hash,8);h=hash_data(h,&b->context_code_count,4);
    size_t d=b->feature_width,l=b->latent_width,k=b->code_count,c=b->context_code_count;
    if(d>SIZE_MAX/l||k>SIZE_MAX/l||c>SIZE_MAX/l)return 0;
    h=hash_data(h,b->encoder,l*d*sizeof(float));h=hash_data(h,b->encoder_bias,l*sizeof(float));
    h=hash_data(h,b->decoder,d*l*sizeof(float));h=hash_data(h,b->decoder_bias,d*sizeof(float));
    h=hash_data(h,b->centers,k*l*sizeof(float));if(b->levels==2)h=hash_data(h,b->residual_centers,k*l*sizeof(float));
    if(c)h=hash_data(h,b->context_centers,c*l*sizeof(float));
    *out=h;return 1;
}
int ama10_image_codebook_save(const AMA10ImageCodebook *b,const char *path){
    if(!b||!b->centers||!path||!b->stride_width||!b->stride_height||
       b->stride_width>b->patch_width||b->stride_height>b->patch_height)return 0;
    uint64_t sum;int v5=b->latent_width!=0,v4=b->context_code_count!=0;
    if(v5?!checksum_v5(b,&sum):v4?!checksum_v4(b,&sum):!checksum_v3(b,&sum))return 0;
    FILE *f=fopen(path,"wb");if(!f)return 0;
    const unsigned char magic3[8]={'A','1','0','I','M','G','0','3'},magic4[8]={'A','1','0','I','M','G','0','4'},magic5[8]={'A','1','0','I','M','G','0','5'};const unsigned char*magic=v5?magic5:v4?magic4:magic3;
    int ok=fwrite(magic,1,8,f)==8&&fwrite(&b->patch_width,4,1,f)==1&&fwrite(&b->patch_height,4,1,f)==1&&
      fwrite(&b->stride_width,4,1,f)==1&&fwrite(&b->stride_height,4,1,f)==1&&
      fwrite(&b->levels,4,1,f)==1&&fwrite(&b->code_count,4,1,f)==1&&fwrite(&b->feature_width,4,1,f)==1&&
      (!v5||fwrite(&b->latent_width,4,1,f)==1)&&fwrite(&b->train_hash,8,1,f)==1&&
      (!(v4||v5)||fwrite(&b->context_code_count,4,1,f)==1)&&fwrite(&sum,8,1,f)==1&&
      (!v5||(fwrite(b->encoder,sizeof(float),(size_t)b->latent_width*b->feature_width,f)==(size_t)b->latent_width*b->feature_width&&
       fwrite(b->encoder_bias,sizeof(float),b->latent_width,f)==b->latent_width&&
       fwrite(b->decoder,sizeof(float),(size_t)b->feature_width*b->latent_width,f)==(size_t)b->feature_width*b->latent_width&&
       fwrite(b->decoder_bias,sizeof(float),b->feature_width,f)==b->feature_width))&&
      fwrite(b->centers,sizeof(float),(size_t)b->code_count*(v5?b->latent_width:b->feature_width),f)==(size_t)b->code_count*(v5?b->latent_width:b->feature_width)&&
      (b->levels==1||fwrite(b->residual_centers,sizeof(float),(size_t)b->code_count*(v5?b->latent_width:b->feature_width),f)==(size_t)b->code_count*(v5?b->latent_width:b->feature_width))&&
      (!v4||fwrite(b->context_centers,sizeof(float),(size_t)b->context_code_count*(v5?b->latent_width:b->feature_width),f)==(size_t)b->context_code_count*(v5?b->latent_width:b->feature_width));
    if(fclose(f)!=0)ok=0;
    return ok;
}
int ama10_image_codebook_load(AMA10ImageCodebook *out,const char *path){
    if(!out||out->centers||!path)return 0;
    FILE *f=fopen(path,"rb");if(!f)return 0;
    AMA10ImageCodebook b={0};
    unsigned char magic[8],expected1[8]={'A','1','0','I','M','G','0','1'},expected2[8]={'A','1','0','I','M','G','0','2'},expected3[8]={'A','1','0','I','M','G','0','3'},expected4[8]={'A','1','0','I','M','G','0','4'},expected5[8]={'A','1','0','I','M','G','0','5'};uint64_t sum=0;int ok=fread(magic,1,8,f)==8&&
      (!memcmp(magic,expected1,8)||!memcmp(magic,expected2,8)||!memcmp(magic,expected3,8)||!memcmp(magic,expected4,8)||!memcmp(magic,expected5,8))&&
      fread(&b.patch_width,4,1,f)==1&&fread(&b.patch_height,4,1,f)==1;
    if(ok&&(!memcmp(magic,expected2,8)||!memcmp(magic,expected3,8)||!memcmp(magic,expected4,8)||!memcmp(magic,expected5,8)))ok=fread(&b.stride_width,4,1,f)==1&&fread(&b.stride_height,4,1,f)==1;
    else if(ok)b.stride_width=b.patch_width,b.stride_height=b.patch_height;
    if(ok&&(!memcmp(magic,expected3,8)||!memcmp(magic,expected4,8)||!memcmp(magic,expected5,8)))ok=fread(&b.levels,4,1,f)==1;
    else if(ok)b.levels=1;
    ok=ok&&fread(&b.code_count,4,1,f)==1&&fread(&b.feature_width,4,1,f)==1&&
      (!memcmp(magic,expected5,8)?fread(&b.latent_width,4,1,f)==1:1)&&fread(&b.train_hash,8,1,f)==1&&
      ((!memcmp(magic,expected4,8)||!memcmp(magic,expected5,8))?fread(&b.context_code_count,4,1,f)==1:1)&&fread(&sum,8,1,f)==1&&b.patch_width&&b.patch_height&&b.code_count&&b.feature_width==b.patch_width*b.patch_height*3&&
      b.code_count<=256&&b.feature_width<=4096&&b.levels>=1&&b.levels<=2&&b.stride_width&&b.stride_height&&
      b.stride_width<=b.patch_width&&b.stride_height<=b.patch_height&&b.context_code_count<=256&&
      (!memcmp(magic,expected4,8)?b.context_code_count>0:!memcmp(magic,expected5,8)||b.context_code_count==0)&&
      (!memcmp(magic,expected5,8)?(b.latent_width>0&&b.latent_width<=b.feature_width):b.latent_width==0);
    if(ok&&(b.patch_width>64||b.patch_height>64||b.feature_width!=(uint64_t)b.patch_width*b.patch_height*3))ok=0;
    if(ok&&!memcmp(magic,expected5,8)){
        size_t d=b.feature_width,l=b.latent_width;
        if(d>SIZE_MAX/l||d*l>SIZE_MAX/sizeof(float))ok=0;
        if(ok){b.encoder=(float*)malloc(d*l*sizeof(float));b.encoder_bias=(float*)malloc(l*sizeof(float));b.decoder=(float*)malloc(d*l*sizeof(float));b.decoder_bias=(float*)malloc(d*sizeof(float));
            if(!b.encoder||!b.encoder_bias||!b.decoder||!b.decoder_bias||fread(b.encoder,sizeof(float),d*l,f)!=d*l||
               fread(b.encoder_bias,sizeof(float),l,f)!=l||fread(b.decoder,sizeof(float),d*l,f)!=d*l||fread(b.decoder_bias,sizeof(float),d,f)!=d)ok=0;}
    }
    if(ok){size_t width=b.latent_width?b.latent_width:b.feature_width;size_t n=(size_t)b.code_count*width;b.centers=(float*)malloc(n*sizeof(float));if(!b.centers)ok=0;else if(fread(b.centers,sizeof(float),n,f)!=n)ok=0;
        if(ok&&b.levels==2){b.residual_centers=(float*)malloc(n*sizeof(float));if(!b.residual_centers||fread(b.residual_centers,sizeof(float),n,f)!=n)ok=0;}
        if(ok&&b.context_code_count){size_t cn=(size_t)b.context_code_count*width;b.context_centers=(float*)malloc(cn*sizeof(float));if(!b.context_centers||fread(b.context_centers,sizeof(float),cn,f)!=cn)ok=0;}}
    if(fgetc(f)!=EOF)ok=0;
    if(fclose(f)!=0)ok=0;
    uint64_t actual=0;if(ok){int valid=!memcmp(magic,expected1,8)?checksum_v1(&b,&actual):
        !memcmp(magic,expected2,8)?checksum(&b,&actual):!memcmp(magic,expected3,8)?checksum_v3(&b,&actual):!memcmp(magic,expected4,8)?checksum_v4(&b,&actual):checksum_v5(&b,&actual);ok=valid&&actual==sum;}
    if(ok)*out=b;else ama10_image_codebook_free(&b);return ok;
}
int ama10_image_encode_codes(const AMA10Image *im,const AMA10ImageCodebook *b,uint32_t **out,size_t *n){
    if(!im||!im->rgb||!b||!b->centers||!out||*out||!n||!im->width||!im->height||
       !b->patch_width||!b->patch_height||b->levels<1||b->levels>2||
       (b->levels==2&&!b->residual_centers)||b->feature_width!=(uint64_t)b->patch_width*b->patch_height*3)return 0;
    float *patches=NULL;size_t count=0,d=0;
    if(!ama10_image_patchify_overlap(im,b->patch_width,b->patch_height,b->stride_width,b->stride_height,&patches,&count,&d)||d!=b->feature_width)return 0;
    if(count>SIZE_MAX/b->levels||count*b->levels>SIZE_MAX/sizeof(uint32_t)){free(patches);return 0;}
    size_t width=code_width(b);
    uint32_t *codes=(uint32_t*)malloc(count*b->levels*sizeof(uint32_t));float*res=(b->levels==2)?(float*)malloc(width*sizeof(float)):NULL;
    float*z=(float*)malloc(width*sizeof(float));
    if(!codes||(b->levels==2&&!res)||!z){free(codes);free(res);free(z);free(patches);return 0;}
    int ok=1;for(size_t i=0;i<count&&ok;i++){
        const float*x=patches+i*d;ok=codec_encode(b,x,z);if(!ok)break;
        uint32_t q=quantize_centers(b->centers,width,b->code_count,z);codes[i*b->levels]=q+1;
        if(b->levels==2){const float*m=b->centers+(size_t)q*width;for(size_t j=0;j<width;j++)res[j]=z[j]-m[j];
            codes[i*b->levels+1]=quantize_centers(b->residual_centers,width,b->code_count,res)+1;}
    }
    free(res);free(z);free(patches);if(!ok){free(codes);return 0;}*out=codes;*n=count*b->levels;return 1;
}
int ama10_image_decode_codes(const AMA10ImageCodebook *b,const uint32_t *codes,size_t n,AMA10Image *im){
    return ama10_image_decode_codes_size(b,codes,n,64,64,im);
}
int ama10_image_decode_codes_size(const AMA10ImageCodebook *b,const uint32_t *codes,size_t n,
                                  size_t width,size_t height,AMA10Image *im){
    if(!b||!b->centers||!codes||!n||!width||!height||!im||im->rgb||
       !b->patch_width||!b->patch_height||width>SIZE_MAX/height||width*height>SIZE_MAX/4||
       b->feature_width!=(uint64_t)b->patch_width*b->patch_height*3)return 0;
    size_t dx=width>b->patch_width?width-b->patch_width:0;
    size_t dy=height>b->patch_height?height-b->patch_height:0;
    size_t grid_w=dx/b->stride_width+(dx%b->stride_width!=0)+1;
    size_t grid_h=dy/b->stride_height+(dy%b->stride_height!=0)+1;
    if(grid_w>SIZE_MAX/grid_h)return 0;
    size_t patch_count=grid_w*grid_h;if(b->levels<1||b->levels>2||patch_count>SIZE_MAX/b->levels||patch_count*b->levels!=n||
       (b->levels==2&&!b->residual_centers))return 0;
    size_t pixels=width*height;float *acc=(float*)calloc(pixels*4,sizeof(float));if(!acc)return 0;
    for(size_t i=0;i<patch_count;i++){
        uint32_t base_code=codes[i*b->levels];if(!base_code||base_code>b->code_count){free(acc);return 0;}
        uint32_t residual_code=0;if(b->levels==2){residual_code=codes[i*b->levels+1];if(!residual_code||residual_code>b->code_count){free(acc);return 0;}}
        size_t width_latent=code_width(b);float *latent=(float*)malloc(width_latent*sizeof(float)),*patch=(float*)malloc(b->feature_width*sizeof(float));
        if(!latent||!patch){free(latent);free(patch);free(acc);return 0;}
        const float *p=b->centers+(size_t)(base_code-1)*width_latent;
        const float *r=b->levels==2?b->residual_centers+(size_t)(residual_code-1)*width_latent:NULL;
        for(size_t j=0;j<width_latent;j++)latent[j]=p[j]+(r?r[j]:0.0f);
        if(!codec_decode(b,latent,patch)){free(latent);free(patch);free(acc);return 0;}
        free(latent);
        size_t py=i/grid_w,px=i%grid_w,x0=px*b->stride_width,y0=py*b->stride_height;
        size_t cw=width-x0<b->patch_width?width-x0:b->patch_width;
        size_t ch=height-y0<b->patch_height?height-y0:b->patch_height;
        for(size_t y=0;y<ch;y++)for(size_t x=0;x<cw;x++){
            size_t fy=(size_t)(((uint64_t)y*(uint64_t)b->patch_height)/(uint64_t)ch);
            size_t fx=(size_t)(((uint64_t)x*(uint64_t)b->patch_width)/(uint64_t)cw);
            size_t src=(fy*b->patch_width+fx)*3,dst=(y0+y)*width+x0+x;
            /* Triangular overlap weights blend neighboring decoded patches. */
            double wx=1.0/(1.0+fabs(((double)x+0.5)/(double)cw*2.0-1.0));
            double wy=1.0/(1.0+fabs(((double)y+0.5)/(double)ch*2.0-1.0));
            float weight=(float)(wx*wy);
            for(size_t c=0;c<3;c++){float v=patch[src+c];if(!isfinite(v)){free(patch);free(acc);return 0;}acc[dst*4+c]+=v*weight;}
            acc[dst*4+3]+=weight;
        }
        free(patch);
    }
    unsigned char *rgb=(unsigned char*)malloc(pixels*3);if(!rgb){free(acc);return 0;}
    for(size_t i=0;i<pixels;i++){float weight=acc[i*4+3];if(!(weight>0)||!isfinite(weight)){free(acc);free(rgb);return 0;}for(size_t c=0;c<3;c++){float v=acc[i*4+c]/weight;if(v<0)v=0;if(v>1)v=1;rgb[i*3+c]=(unsigned char)lrintf(v*255);}}
    free(acc);
    im->width=width;im->height=height;im->rgb=rgb;return 1;
}
int ama10_image_save_ppm(const AMA10Image *im,const char *path){if(!im||!im->rgb||!path||!im->width||!im->height||im->width>SIZE_MAX/im->height||im->width*im->height>SIZE_MAX/3)return 0;FILE *f=fopen(path,"wb");if(!f)return 0;
    int ok=fprintf(f,"P6\n%zu %zu\n255\n",im->width,im->height)>0&&fwrite(im->rgb,1,im->width*im->height*3,f)==im->width*im->height*3;
    if(fclose(f)!=0)ok=0;
    return ok;}
int ama10_image_metrics(const AMA10Image *ref,const AMA10Image *cand,double *psnr,double *ssim){
    if(!ref||!cand||!ref->rgb||!cand->rgb||!psnr||!ssim||!ref->width||!ref->height||ref->width!=cand->width||ref->height!=cand->height||
       ref->width>SIZE_MAX/ref->height||ref->width*ref->height>SIZE_MAX/3)return 0;
    size_t pixels=ref->width*ref->height;double mse=0;
    for(size_t i=0;i<pixels*3;i++){double d=(double)ref->rgb[i]-cand->rgb[i];mse+=d*d;}
    mse/=(double)(pixels*3);double p=mse==0.0?INFINITY:10.0*log10((255.0*255.0)/mse);
    const double c1=(0.01*255.0)*(0.01*255.0),c2=(0.03*255.0)*(0.03*255.0);double ssum=0;size_t windows=0;
    for(size_t y0=0;y0<ref->height;y0+=8)for(size_t x0=0;x0<ref->width;x0+=8){size_t x1=x0+8<ref->width?x0+8:ref->width,y1=y0+8<ref->height?y0+8:ref->height,n=(x1-x0)*(y1-y0);if(!n)return 0;
      for(size_t channel=0;channel<3;channel++){double mx=0,my=0;for(size_t y=y0;y<y1;y++)for(size_t x=x0;x<x1;x++){size_t i=(y*ref->width+x)*3+channel;mx+=ref->rgb[i];my+=cand->rgb[i];}mx/=(double)n;my/=(double)n;
        double vx=0,vy=0,cov=0;for(size_t y=y0;y<y1;y++)for(size_t x=x0;x<x1;x++){size_t i=(y*ref->width+x)*3+channel;double a=(double)ref->rgb[i]-mx,b=(double)cand->rgb[i]-my;vx+=a*a;vy+=b*b;cov+=a*b;}vx/=(double)n;vy/=(double)n;cov/=(double)n;
        double a=(2*mx*my+c1)*(2*cov+c2),b=(mx*mx+my*my+c1)*(vx+vy+c2);ssum+=a/b;windows++;}}
    if(!windows)return 0;
    *psnr=p;*ssim=ssum/(double)windows;return isfinite(*ssim)&&(*ssim>=-1.0&&*ssim<=1.0);
}
int ama10_image_score_ppm(const char *a,const char *b,double *psnr,double *ssim){AMA10Image ref={0},cand={0};int ok=ama10_image_load_ppm(&ref,a)&&ama10_image_load_ppm(&cand,b)&&ama10_image_metrics(&ref,&cand,psnr,ssim);ama10_image_free(&ref);ama10_image_free(&cand);return ok;}
