#include "lora.h"
#include "checkpoint.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int put32(FILE *f,uint32_t x){unsigned char b[4]={(unsigned char)x,(unsigned char)(x>>8),(unsigned char)(x>>16),(unsigned char)(x>>24)};return fwrite(b,1,4,f)==4;}
static int put64(FILE *f,uint64_t x){return put32(f,(uint32_t)x)&&put32(f,(uint32_t)(x>>32));}
static int get32(FILE *f,uint32_t *x){unsigned char b[4];if(fread(b,1,4,f)!=4)return 0;*x=(uint32_t)b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24;return 1;}
static int get64(FILE *f,uint64_t *x){uint32_t a,b;if(!get32(f,&a)||!get32(f,&b))return 0;*x=(uint64_t)a|(uint64_t)b<<32;return 1;}
static uint32_t rng(uint64_t *s){*s^=*s>>12;*s^=*s<<25;*s^=*s>>27;return (uint32_t)((*s*UINT64_C(2685821657736338717))>>32);}
int ama10_lora_init(AMA10LoRA *l,const AMA10Model *m,size_t rank,float scale,uint64_t bh,uint64_t th,uint64_t seed){
    if(!l||!m||!m->width||!m->vocabulary||!rank||rank>m->width||!isfinite(scale)||scale<=0||!bh||!th||rank>SIZE_MAX/m->width||m->vocabulary>SIZE_MAX/rank)return 0;
    memset(l,0,sizeof(*l));l->rank=rank;l->width=m->width;l->vocabulary=m->vocabulary;l->scale=scale;l->base_hash=bh;l->tokenizer_hash=th;
    size_t na=rank*m->width,nb=m->vocabulary*rank;l->a=(float*)malloc(na*sizeof(float));l->b=(float*)calloc(nb,sizeof(float));if(!l->a||!l->b){ama10_lora_free(l);return 0;}
    if(!seed)seed=UINT64_C(0x4c6f5241414d4131);
    for(size_t i=0;i<na;i++)l->a[i]=((float)rng(&seed)/2147483648.0f-1.0f)*sqrtf(3.0f/(float)m->width);
    return 1;
}
void ama10_lora_free(AMA10LoRA *l){if(!l)return;free(l->a);free(l->b);memset(l,0,sizeof(*l));}
void ama10_lora_apply(const AMA10LoRA *l,const float *h,float *z){
    if(!l||!h||!z)return;
    for(size_t r=0;r<l->rank;r++){double c=0;for(size_t j=0;j<l->width;j++)c+=(double)l->a[r*l->width+j]*h[j];
        for(size_t k=0;k<l->vocabulary;k++)z[k]+=l->scale*l->b[k*l->rank+r]*(float)c;}
}
static int serialize(FILE *f,const AMA10LoRA *l){
    static const unsigned char magic[8]={'A','M','A','1','0','L','R','1'};
    if(fwrite(magic,1,8,f)!=8||!put32(f,1)||!put32(f,(uint32_t)l->rank)||!put32(f,(uint32_t)l->width)||!put32(f,(uint32_t)l->vocabulary)||!put64(f,l->base_hash)||!put64(f,l->tokenizer_hash))return 0;
    uint32_t bits;memcpy(&bits,&l->scale,4);if(!put32(f,bits))return 0;
    uint64_t count=(uint64_t)l->rank*l->width+(uint64_t)l->vocabulary*l->rank;uint64_t hash=UINT64_C(14695981039346656037);
    float *blocks[2]={l->a,l->b};size_t sizes[2]={l->rank*l->width,l->vocabulary*l->rank};
    for(int q=0;q<2;q++)for(size_t i=0;i<sizes[q];i++){memcpy(&bits,&blocks[q][i],4);unsigned char b[4]={(unsigned char)bits,(unsigned char)(bits>>8),(unsigned char)(bits>>16),(unsigned char)(bits>>24)};if(fwrite(b,1,4,f)!=4)return 0;hash=ama10_data_hash(b,4)^((hash<<7)|(hash>>57));}
    return put64(f,hash)&&count>0;
}
int ama10_lora_save(const AMA10LoRA *l,const char *path){
    if(!l||!l->a||!l->b||!path||l->rank>UINT32_MAX||l->width>UINT32_MAX||l->vocabulary>UINT32_MAX)return 0;
    size_t n=strlen(path);if(n>SIZE_MAX-5)return 0;char *tmp=(char*)malloc(n+5),*bak=(char*)malloc(n+5);if(!tmp||!bak){free(tmp);free(bak);return 0;}memcpy(tmp,path,n);memcpy(tmp+n,".tmp",5);memcpy(bak,path,n);memcpy(bak+n,".bak",5);
    FILE *f=fopen(tmp,"wb");int ok=f&&serialize(f,l);if(f){if(fflush(f)!=0)ok=0;if(fclose(f)!=0)ok=0;}int moved=0;
    if(ok){(void)remove(bak);if(rename(path,bak)==0)moved=1;else if(errno!=ENOENT)ok=0;}
    if(ok&&rename(tmp,path)!=0)ok=0;
    if(!ok){remove(tmp);if(moved)(void)rename(bak,path);}else if(moved)(void)remove(bak);
    free(tmp);free(bak);return ok;
}
int ama10_lora_load(AMA10LoRA *l,const char *path,const AMA10Model *m,uint64_t bh,uint64_t th){
    if(!l||!path||!m)return 0;
    memset(l,0,sizeof(*l));FILE *f=fopen(path,"rb");if(!f)return 0;
    unsigned char magic[8],want[8]={'A','M','A','1','0','L','R','1'};uint32_t ver=0,r=0,w=0,v=0,sbits=0;uint64_t base=0,tok=0;
    int ok=fread(magic,1,8,f)==8&&!memcmp(magic,want,8)&&get32(f,&ver)&&get32(f,&r)&&get32(f,&w)&&get32(f,&v)&&get64(f,&base)&&get64(f,&tok)&&get32(f,&sbits);
    float scale=0;memcpy(&scale,&sbits,4);
    ok=ok&&ver==1&&r>0&&r<=w&&w==m->width&&v==m->vocabulary&&base==bh&&tok==th&&isfinite(scale)&&scale>0&&r<=SIZE_MAX/w&&v<=SIZE_MAX/r;
    if(ok){l->rank=r;l->width=w;l->vocabulary=v;l->scale=scale;l->base_hash=base;l->tokenizer_hash=tok;l->a=(float*)malloc((size_t)r*w*sizeof(float));l->b=(float*)malloc((size_t)v*r*sizeof(float));if(!l->a||!l->b)ok=0;}
    uint64_t hash=UINT64_C(14695981039346656037);float *blocks[2]={l->a,l->b};size_t sizes[2]={(size_t)r*w,(size_t)v*r};
    for(int q=0;q<2&&ok;q++)for(size_t i=0;i<sizes[q];i++){unsigned char b[4];uint32_t bits;if(fread(b,1,4,f)!=4){ok=0;break;}bits=(uint32_t)b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24;memcpy(&blocks[q][i],&bits,4);if(!isfinite(blocks[q][i])){ok=0;break;}hash=ama10_data_hash(b,4)^((hash<<7)|(hash>>57));}
    uint64_t filehash=0;if(ok)ok=get64(f,&filehash)&&filehash==hash&&fgetc(f)==EOF;if(fclose(f)!=0)ok=0;if(!ok)ama10_lora_free(l);return ok;
}
