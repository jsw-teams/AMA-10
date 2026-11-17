#include "ama10_imggen.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t hash_data(uint64_t h,const void *data,size_t n){
    const unsigned char *p=(const unsigned char*)data;
    for(size_t i=0;i<n;i++){h^=p[i];h*=UINT64_C(1099511628211);}return h;
}
void ama10_image_codebook_free(AMA10ImageCodebook *b){if(b){free(b->centers);free(b->residual_centers);free(b->context_centers);memset(b,0,sizeof(*b));}}
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
int ama10_image_codebook_fit(const float *x,size_t n,size_t d,uint32_t k,
 uint32_t pw,uint32_t ph,uint64_t train_hash,AMA10ImageCodebook *out){
    if(!x||!n||!d||!k||k>n||pw==0||ph==0||
       d!=(size_t)pw*ph*3||d>SIZE_MAX/k||d*k>SIZE_MAX/sizeof(float)||!out||out->centers)return 0;
    AMA10ImageCodebook b={0};b.centers=(float*)malloc((size_t)k*d*sizeof(float));
    if(!b.centers||!fit_centers(x,n,d,k,b.centers)){ama10_image_codebook_free(&b);return 0;}
    b.patch_width=pw;b.patch_height=ph;b.stride_width=pw;b.stride_height=ph;b.code_count=k;b.feature_width=(uint32_t)d;b.levels=1;b.train_hash=train_hash;*out=b;return 1;
}
int ama10_image_codebook_fit_residual(const float*x,size_t n,AMA10ImageCodebook*b){
    if(!x||!n||!b||!b->centers||b->residual_centers||b->levels!=1||b->code_count>n||
       !b->feature_width||n>SIZE_MAX/b->feature_width||n*b->feature_width>SIZE_MAX/sizeof(float))return 0;
    size_t d=b->feature_width;float*r=(float*)malloc(n*d*sizeof(float));
    float*c=(float*)malloc((size_t)b->code_count*d*sizeof(float));
    if(!r||!c){free(r);free(c);return 0;}
    for(size_t i=0;i<n;i++){uint32_t q=quantize_centers(b->centers,d,b->code_count,x+i*d);const float*m=b->centers+(size_t)q*d;
        for(size_t j=0;j<d;j++)r[i*d+j]=x[i*d+j]-m[j];}
    int ok=fit_centers(r,n,d,b->code_count,c);free(r);
    if(!ok){free(c);return 0;}b->residual_centers=c;b->levels=2;return 1;
}
int ama10_image_codebook_fit_context(const float*x,size_t n,uint32_t k,AMA10ImageCodebook*b){
    if(!x||!n||!b||!b->centers||!k||k>n||b->context_centers||b->context_code_count||
       n>SIZE_MAX/b->feature_width||n*b->feature_width>SIZE_MAX/sizeof(float)||
       (size_t)k>SIZE_MAX/b->feature_width||
       (size_t)k*b->feature_width>SIZE_MAX/sizeof(float))return 0;
    float*c=(float*)malloc((size_t)k*b->feature_width*sizeof(float));if(!c)return 0;
    if(!fit_centers(x,n,b->feature_width,k,c)){free(c);return 0;}
    b->context_centers=c;b->context_code_count=k;return 1;
}
int ama10_image_codebook_quantize(const AMA10ImageCodebook *b,const float *x,uint32_t *code){
    if(!b||!b->centers||!x||!code)return 0;
    *code=quantize_centers(b->centers,b->feature_width,b->code_count,x)+1;return 1;
}
int ama10_image_context_quantize(const AMA10ImageCodebook*b,const float*x,uint32_t*code){
    if(!b||!b->centers||!x||!code)return 0;
    if(!b->context_code_count)return ama10_image_codebook_quantize(b,x,code);
    if(!b->context_centers)return 0;
    *code=quantize_centers(b->context_centers,b->feature_width,b->context_code_count,x)+1;return 1;
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
int ama10_image_codebook_save(const AMA10ImageCodebook *b,const char *path){
    if(!b||!b->centers||!path||!b->stride_width||!b->stride_height||
       b->stride_width>b->patch_width||b->stride_height>b->patch_height)return 0;
    uint64_t sum;int v4=b->context_code_count!=0;if(v4?!checksum_v4(b,&sum):!checksum_v3(b,&sum))return 0;
    FILE *f=fopen(path,"wb");if(!f)return 0;
    const unsigned char magic3[8]={'A','1','0','I','M','G','0','3'},magic4[8]={'A','1','0','I','M','G','0','4'};const unsigned char*magic=v4?magic4:magic3;
    int ok=fwrite(magic,1,8,f)==8&&fwrite(&b->patch_width,4,1,f)==1&&fwrite(&b->patch_height,4,1,f)==1&&
      fwrite(&b->stride_width,4,1,f)==1&&fwrite(&b->stride_height,4,1,f)==1&&
      fwrite(&b->levels,4,1,f)==1&&
      fwrite(&b->code_count,4,1,f)==1&&fwrite(&b->feature_width,4,1,f)==1&&fwrite(&b->train_hash,8,1,f)==1&&
      (!v4||fwrite(&b->context_code_count,4,1,f)==1)&&fwrite(&sum,8,1,f)==1&&
      fwrite(b->centers,sizeof(float),(size_t)b->code_count*b->feature_width,f)==(size_t)b->code_count*b->feature_width&&
      (b->levels==1||fwrite(b->residual_centers,sizeof(float),(size_t)b->code_count*b->feature_width,f)==(size_t)b->code_count*b->feature_width)&&
      (!v4||fwrite(b->context_centers,sizeof(float),(size_t)b->context_code_count*b->feature_width,f)==(size_t)b->context_code_count*b->feature_width);
    if(fclose(f)!=0)ok=0;
    return ok;
}
int ama10_image_codebook_load(AMA10ImageCodebook *out,const char *path){
    if(!out||out->centers||!path)return 0;
    FILE *f=fopen(path,"rb");if(!f)return 0;
    AMA10ImageCodebook b={0};
    unsigned char magic[8],expected1[8]={'A','1','0','I','M','G','0','1'},expected2[8]={'A','1','0','I','M','G','0','2'},expected3[8]={'A','1','0','I','M','G','0','3'},expected4[8]={'A','1','0','I','M','G','0','4'};uint64_t sum=0;int ok=fread(magic,1,8,f)==8&&
      (!memcmp(magic,expected1,8)||!memcmp(magic,expected2,8)||!memcmp(magic,expected3,8)||!memcmp(magic,expected4,8))&&
      fread(&b.patch_width,4,1,f)==1&&fread(&b.patch_height,4,1,f)==1;
    if(ok&&(!memcmp(magic,expected2,8)||!memcmp(magic,expected3,8)||!memcmp(magic,expected4,8)))ok=fread(&b.stride_width,4,1,f)==1&&fread(&b.stride_height,4,1,f)==1;
    else if(ok)b.stride_width=b.patch_width,b.stride_height=b.patch_height;
    if(ok&&(!memcmp(magic,expected3,8)||!memcmp(magic,expected4,8)))ok=fread(&b.levels,4,1,f)==1;
    else if(ok)b.levels=1;
    ok=ok&&fread(&b.code_count,4,1,f)==1&&fread(&b.feature_width,4,1,f)==1&&
      fread(&b.train_hash,8,1,f)==1&&(!memcmp(magic,expected4,8)?fread(&b.context_code_count,4,1,f)==1:1)&&fread(&sum,8,1,f)==1&&b.patch_width&&b.patch_height&&b.code_count&&b.feature_width==b.patch_width*b.patch_height*3&&
      b.code_count<=256&&b.feature_width<=4096&&b.levels>=1&&b.levels<=2&&b.stride_width&&b.stride_height&&
      b.stride_width<=b.patch_width&&b.stride_height<=b.patch_height&&b.context_code_count<=256&&
      (!memcmp(magic,expected4,8)?b.context_code_count>0:b.context_code_count==0);
    if(ok&&(b.patch_width>64||b.patch_height>64||b.feature_width!=(uint64_t)b.patch_width*b.patch_height*3))ok=0;
    if(ok){size_t n=(size_t)b.code_count*b.feature_width;b.centers=(float*)malloc(n*sizeof(float));if(!b.centers)ok=0;else if(fread(b.centers,sizeof(float),n,f)!=n)ok=0;
        if(ok&&b.levels==2){b.residual_centers=(float*)malloc(n*sizeof(float));if(!b.residual_centers||fread(b.residual_centers,sizeof(float),n,f)!=n)ok=0;}
        if(ok&&b.context_code_count){size_t cn=(size_t)b.context_code_count*b.feature_width;b.context_centers=(float*)malloc(cn*sizeof(float));if(!b.context_centers||fread(b.context_centers,sizeof(float),cn,f)!=cn)ok=0;}}
    if(fgetc(f)!=EOF)ok=0;
    if(fclose(f)!=0)ok=0;
    uint64_t actual=0;if(ok){int valid=!memcmp(magic,expected1,8)?checksum_v1(&b,&actual):
        !memcmp(magic,expected2,8)?checksum(&b,&actual):!memcmp(magic,expected3,8)?checksum_v3(&b,&actual):checksum_v4(&b,&actual);ok=valid&&actual==sum;}
    if(ok)*out=b;else ama10_image_codebook_free(&b);return ok;
}
int ama10_image_encode_codes(const AMA10Image *im,const AMA10ImageCodebook *b,uint32_t **out,size_t *n){
    if(!im||!im->rgb||!b||!b->centers||!out||*out||!n||!im->width||!im->height||
       !b->patch_width||!b->patch_height||b->levels<1||b->levels>2||
       (b->levels==2&&!b->residual_centers)||b->feature_width!=(uint64_t)b->patch_width*b->patch_height*3)return 0;
    float *patches=NULL;size_t count=0,d=0;
    if(!ama10_image_patchify_overlap(im,b->patch_width,b->patch_height,b->stride_width,b->stride_height,&patches,&count,&d)||d!=b->feature_width)return 0;
    if(count>SIZE_MAX/b->levels||count*b->levels>SIZE_MAX/sizeof(uint32_t)){free(patches);return 0;}
    uint32_t *codes=(uint32_t*)malloc(count*b->levels*sizeof(uint32_t));float*res=(b->levels==2)?(float*)malloc(d*sizeof(float)):NULL;
    if(!codes||(b->levels==2&&!res)){free(codes);free(res);free(patches);return 0;}
    int ok=1;for(size_t i=0;i<count&&ok;i++){
        const float*x=patches+i*d;uint32_t q=quantize_centers(b->centers,d,b->code_count,x);codes[i*b->levels]=q+1;
        if(b->levels==2){const float*m=b->centers+(size_t)q*d;for(size_t j=0;j<d;j++)res[j]=x[j]-m[j];
            codes[i*b->levels+1]=quantize_centers(b->residual_centers,d,b->code_count,res)+1;}
    }
    free(res);free(patches);if(!ok){free(codes);return 0;}*out=codes;*n=count*b->levels;return 1;
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
        const float *p=b->centers+(size_t)(base_code-1)*b->feature_width;
        const float *r=b->levels==2?b->residual_centers+(size_t)(residual_code-1)*b->feature_width:NULL;
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
            for(size_t c=0;c<3;c++){float v=p[src+c]+(r?r[src+c]:0.0f);if(!isfinite(v)){free(acc);return 0;}acc[dst*4+c]+=v*weight;}
            acc[dst*4+3]+=weight;
        }
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
