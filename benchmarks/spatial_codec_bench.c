#include "ama10_imggen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {char *relative,*split;AMA10Image image;uint64_t hash;} ImageRow;
static char *copy_string(const char*s){size_t n=strlen(s);char*p=(char*)malloc(n+1);if(p)memcpy(p,s,n+1);return p;}
static uint64_t image_hash(const AMA10Image*im){uint64_t h=UINT64_C(14695981039346656037);const unsigned char*p=im->rgb;size_t n=im->width*im->height*3;
    for(size_t i=0;i<n;i++){h^=p[i];h*=UINT64_C(1099511628211);}h^=im->width;h*=UINT64_C(1099511628211);h^=im->height;return h;}
static int join_path(const char*manifest,const char*relative,char**out){const char*a=strrchr(manifest,'/'),*b=strrchr(manifest,'\\'),*sep=a;if(b&&(!sep||b>sep))sep=b;size_t dir=sep?(size_t)(sep-manifest+1):0,n=strlen(relative);char*p=(char*)malloc(dir+n+1);if(!p)return 0;if(dir)memcpy(p,manifest,dir);memcpy(p+dir,relative,n+1);*out=p;return 1;}
static void free_rows(ImageRow*r,size_t n){for(size_t i=0;i<n;i++){free(r[i].relative);free(r[i].split);ama10_image_free(&r[i].image);}free(r);}
static int load_rows(const char*manifest,ImageRow**out,size_t*count){
    FILE*f=fopen(manifest,"rb");if(!f)return 0;char line[8192];ImageRow*r=NULL;size_t n=0,cap=0;int ok=1;
    while(fgets(line,sizeof(line),f)){size_t z=strlen(line);while(z&&(line[z-1]=='\n'||line[z-1]=='\r'))line[--z]=0;if(!z)continue;
        char*a=strchr(line,'\t');if(!a){ok=0;break;}*a++=0;char*b=strchr(a,'\t');if(!b){ok=0;break;}*b++=0;char*c=strchr(b,'\t');if(c)*c=0;
        if(!*line||(strcmp(b,"train")&&strcmp(b,"validation"))){ok=0;break;}
        if(n==cap){size_t nc=cap?cap*2:32;ImageRow*q=(ImageRow*)realloc(r,nc*sizeof(*q));if(!q){ok=0;break;}r=q;cap=nc;}
        ImageRow row={0};row.relative=copy_string(line);row.split=copy_string(b);char*path=NULL;
        if(!row.relative||!row.split||!join_path(manifest,line,&path)||!ama10_image_load(&row.image,path)){free(path);free(row.relative);free(row.split);ama10_image_free(&row.image);ok=0;break;}
        free(path);row.hash=image_hash(&row.image);r[n++]=row;
    }
    if(ferror(f))ok=0;
    fclose(f);
    if(!ok){free_rows(r,n);return 0;}
    *out=r;*count=n;return n>1;
}
static int is_first_train(const ImageRow*r,size_t index){for(size_t i=0;i<index;i++)if(!strcmp(r[i].split,"train")&&r[i].hash==r[index].hash&&r[i].image.width==r[index].image.width&&r[i].image.height==r[index].image.height)return 0;return 1;}
static int make_training_patches(const ImageRow*r,size_t n,float**out,size_t*count){
    size_t total=0;for(size_t i=0;i<n;i++)if(!strcmp(r[i].split,"train")&&is_first_train(r,i)){float*p=NULL;size_t np=0,d=0;
        if(!ama10_image_patchify_overlap(&r[i].image,16,16,12,12,&p,&np,&d)||d!=768){free(p);return 0;}
        size_t take=np<256?np:256;if(take>SIZE_MAX-total){free(p);return 0;}total+=take;free(p);}
    if(!total||total>SIZE_MAX/768||total*768>SIZE_MAX/sizeof(float))return 0;
    float*x=(float*)malloc(total*768*sizeof(float));if(!x)return 0;size_t at=0;
    for(size_t i=0;i<n;i++)if(!strcmp(r[i].split,"train")&&is_first_train(r,i)){float*p=NULL;size_t np=0,d=0;
        if(!ama10_image_patchify_overlap(&r[i].image,16,16,12,12,&p,&np,&d)||d!=768){free(p);free(x);return 0;}
        size_t take=np<256?np:256;for(size_t q=0;q<take;q++){size_t sample=q*np/take;memcpy(x+(at+q)*d,p+sample*d,d*sizeof(float));}at+=take;free(p);}
    if(at!=total){free(x);return 0;}*out=x;*count=total;return 1;
}
static int evaluate(const AMA10ImageCodebook*b,const ImageRow*r,size_t n,int validation,double*psnr,double*ssim,size_t*images){
    double p=0,s=0;size_t count=0;
    for(size_t i=0;i<n;i++)if((!strcmp(r[i].split,"validation"))==validation){uint32_t*codes=NULL;size_t code_count=0;AMA10Image decoded={0};double pi=0,si=0;
        if(!ama10_image_encode_codes(&r[i].image,b,&codes,&code_count)||!ama10_image_decode_codes_size(b,codes,code_count,r[i].image.width,r[i].image.height,&decoded)||
           !ama10_image_metrics(&r[i].image,&decoded,&pi,&si)){free(codes);ama10_image_free(&decoded);return 0;}
        p+=pi;s+=si;count++;free(codes);ama10_image_free(&decoded);}
    if(!count)return 0;
    *psnr=p/count;*ssim=s/count;*images=count;return 1;
}
int main(int argc,char**argv){
    if(argc!=5){fprintf(stderr,"usage: %s <manifest.tsv> <baseline.vq> <output.vq> <kernels.cl>\n",argv[0]);return 2;}
    ImageRow*rows=NULL;size_t n=0,patch_count=0;float*patches=NULL;AMA10ImageCodebook baseline={0},trained={0};
    if(!load_rows(argv[1],&rows,&n)||!make_training_patches(rows,n,&patches,&patch_count)||!ama10_image_codebook_load(&baseline,argv[2]))goto fail;
    if(!ama10_image_codebook_fit_opencl(patches,patch_count,768,baseline.code_count,16,16,
       UINT64_C(20261118),argv[4],&trained)){goto fail;}
    if(!ama10_image_codebook_fit_residual(patches,patch_count,&trained))goto fail;
    trained.stride_width=12;trained.stride_height=12;
    double base_p=0,base_s=0,new_p=0,new_s=0;size_t base_n=0,new_n=0;
    if(!evaluate(&baseline,rows,n,1,&base_p,&base_s,&base_n)||!evaluate(&trained,rows,n,1,&new_p,&new_s,&new_n)||
       base_n!=new_n||!ama10_image_codebook_save(&trained,argv[3]))goto fail;
    printf("spatial_codec=learned_linear_2d_patch_latent train_images=%zu training_patches=%zu validation_images=%zu baseline_psnr=%.4f baseline_ssim=%.6f learned_psnr=%.4f learned_ssim=%.6f checkpoint=%s\n",
        n,patch_count,new_n,base_p,base_s,new_p,new_s,argv[3]);
    free(patches);free_rows(rows,n);ama10_image_codebook_free(&baseline);ama10_image_codebook_free(&trained);return 0;
fail:
    fprintf(stderr,"spatial codec benchmark failed\n");free(patches);free_rows(rows,n);ama10_image_codebook_free(&baseline);ama10_image_codebook_free(&trained);return 1;
}
