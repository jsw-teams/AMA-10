#include "ama10.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int same(const AMA10Model *a,const AMA10Model *b){
    const float *x[]={a->wx,a->wh,a->wm,a->wu,a->bias,a->gate_bias,a->embed,a->head};
    const float *y[]={b->wx,b->wh,b->wm,b->wu,b->bias,b->gate_bias,b->embed,b->head};
    size_t n[]={a->width*a->input_width,a->width*a->width,a->width*a->width,a->width*a->width,
        a->width,a->width,a->input_width*a->vocabulary,a->vocabulary*a->width};
    if(a->width!=b->width||a->input_width!=b->input_width||a->vocabulary!=b->vocabulary)return 0;
    for(size_t k=0;k<8;k++)if(memcmp(x[k],y[k],n[k]*sizeof(float))!=0)return 0;
    return 1;
}
static int unchanged(const AMA10Model *a,const AMA10Model *b){return same(a,b);}
static int contains(const unsigned char *p,size_t n,const char *needle){size_t k=strlen(needle);if(k>n)return 0;for(size_t i=0;i<=n-k;i++)if(!memcmp(p+i,needle,k))return 1;return 0;}
static int write_external_fixture(const char *path){
    const char *json="{\"head\":{\"data_offsets\":[0,32],\"shape\":[4,2],\"dtype\":\"F32\"},"
        "\"embed\":{\"dtype\":\"F32\",\"shape\":[3,4],\"data_offsets\":[32,80]},"
        "\"gate_bias\":{\"shape\":[2],\"data_offsets\":[80,88],\"dtype\":\"F32\"},"
        "\"bias\":{\"dtype\":\"F32\",\"data_offsets\":[88,96],\"shape\":[2]},"
        "\"wu\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[96,112]},"
        "\"wm\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[112,128]},"
        "\"wh\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[128,144]},"
        "\"wx\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[144,168]},"
        "\"__metadata__\":{\"note\":\"external format \\u4e2d\"}}";
    size_t raw=strlen(json),padded=(raw+7u)&~(size_t)7u;unsigned char size_bytes[8];uint64_t n=padded;
    for(unsigned i=0;i<8;i++)size_bytes[i]=(unsigned char)(n>>(i*8));
    FILE *f=fopen(path,"wb");if(!f)return 0;
    int ok=fwrite(size_bytes,1,8,f)==8&&fwrite(json,1,raw,f)==raw;
    for(size_t i=raw;i<padded&&ok;i++)ok=fputc(' ',f)!=EOF;
    for(uint32_t i=1;i<=42&&ok;i++){
        float value=(float)i;uint32_t bits;memcpy(&bits,&value,sizeof(bits));
        unsigned char b[4]={(unsigned char)bits,(unsigned char)(bits>>8),(unsigned char)(bits>>16),(unsigned char)(bits>>24)};
        ok=fwrite(b,1,4,f)==4;
    }
    if(fclose(f)!=0) ok=0;
    return ok;
}
int main(void){
    const char *path="build/safetensors-roundtrip.safetensors",*bad="build/safetensors-corrupt.safetensors";
    const char *external="build/safetensors-external-fixture.safetensors";
    AMA10Model source={0},loaded={0},sentinel={0},before={0},external_model={0};int ok=0;unsigned char *bytes=NULL;size_t size=0;
    if(!ama10_model_init(&source,37,24,16,918)||!ama10_model_save_safetensors(&source,path)||
       !ama10_model_load(&loaded,path)||!same(&source,&loaded)||
       !ama10_model_init(&sentinel,11,9,7,611)||!ama10_model_init(&before,11,9,7,611))goto done;
    {
        FILE *f=fopen(path,"rb");if(!f||fseek(f,0,SEEK_END)!=0){if(f)fclose(f);goto done;}
        long n=ftell(f);if(n<16||fseek(f,0,SEEK_SET)!=0){fclose(f);goto done;}
        bytes=(unsigned char*)malloc((size_t)n);if(!bytes){fclose(f);goto done;}
        size=(size_t)n;if(fread(bytes,1,size,f)!=size){fclose(f);goto done;}fclose(f);
        uint64_t h=0;for(unsigned i=0;i<8;i++)h|=(uint64_t)bytes[i]<<(i*8);
        if(h<8||h%8||h>size-8||bytes[8]!='{'||!contains(bytes+8,(size_t)h,"\"dtype\":\"F32\""))goto done;
        bytes[8]='[';f=fopen(bad,"wb");if(!f)goto done;
        int wrote=fwrite(bytes,1,size,f)==size;if(fclose(f)!=0)wrote=0;if(!wrote)goto done;
    }
    if(ama10_model_load(&sentinel,bad)||!unchanged(&sentinel,&before))goto done;
    if(!write_external_fixture(external)||!ama10_model_load_safetensors(&external_model,external)||
       external_model.width!=2||external_model.input_width!=3||external_model.vocabulary!=4||
       external_model.head[0]!=1.0f||external_model.head[7]!=8.0f||
       external_model.embed[0]!=9.0f||external_model.embed[11]!=20.0f||
       external_model.gate_bias[0]!=21.0f||external_model.bias[1]!=24.0f||
       external_model.wu[0]!=25.0f||external_model.wm[3]!=32.0f||
       external_model.wh[0]!=33.0f||external_model.wx[5]!=42.0f)goto done;
    ok=1;
done:
    if(!ok)fprintf(stderr,"SafeTensors model round-trip or malformed-header preservation failed\n");
    free(bytes);ama10_model_free(&source);ama10_model_free(&loaded);ama10_model_free(&sentinel);ama10_model_free(&before);ama10_model_free(&external_model);
    (void)remove(path);(void)remove(bad);(void)remove(external);
    if(ok)puts("SafeTensors F32 export/import, external header ordering, offsets, and malformed-header rejection: ok");
    return ok?0:1;
}
