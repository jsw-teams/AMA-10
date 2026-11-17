#include "ama10.h"
#ifdef AMA10_ENABLE_OPENCL
#include "opencl_backend.h"
#endif
#include <float.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static uint32_t rng_next(uint64_t *s) {
    *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
    return (uint32_t)((*s * UINT64_C(2685821657736338717)) >> 32);
}
static float rand_scale(uint64_t *s, size_t fan) {
    double u = (double)rng_next(s) / 4294967296.0;
    return (float)((u * 2.0 - 1.0) * sqrt(3.0 / (double)(fan ? fan : 1)));
}
static int alloc_f(float **p, size_t n) {
    if (n > SIZE_MAX / sizeof(float)) return 0;
    *p = (float *)calloc(n, sizeof(float)); return *p != NULL;
}
static void init_mat(float *p, size_t n, size_t fan, uint64_t *seed) {
    for (size_t i = 0; i < n; ++i) p[i] = rand_scale(seed, fan);
}

int ama10_model_init(AMA10Model *m, size_t v, size_t w, size_t e, uint64_t seed) {
    if (!m || !v || !w || !e || v > SIZE_MAX / e || w > SIZE_MAX / e ||
        w > SIZE_MAX / w || v > SIZE_MAX / w) return 0;
    memset(m, 0, sizeof(*m)); m->width=w; m->input_width=e; m->vocabulary=v;
    if (!alloc_f(&m->wx,w*e) || !alloc_f(&m->wh,w*w) || !alloc_f(&m->wm,w*w) ||
        !alloc_f(&m->wu,w*w) || !alloc_f(&m->bias,w) || !alloc_f(&m->gate_bias,w) ||
        !alloc_f(&m->embed,e*v) || !alloc_f(&m->head,v*w)) {
        ama10_model_free(m); return 0;
    }
    init_mat(m->wx,w*e,e,&seed); init_mat(m->wh,w*w,w,&seed);
    init_mat(m->wm,w*w,w,&seed); init_mat(m->wu,w*w,w,&seed);
    init_mat(m->embed,e*v,e,&seed);
    init_mat(m->head,v*w,w,&seed);
    return 1;
}
int ama10_model_resize_vocabulary(AMA10Model *m,size_t new_vocabulary,uint64_t seed){
    if(!m||!m->wx||!m->wh||!m->wm||!m->wu||!m->bias||!m->gate_bias||!m->embed||!m->head||
       new_vocabulary<=m->vocabulary)return 0;
    AMA10Model grown={0};
    if(!ama10_model_init(&grown,new_vocabulary,m->width,m->input_width,seed))return 0;
    size_t w=m->width,e=m->input_width,old_vocabulary=m->vocabulary;
    memcpy(grown.wx,m->wx,w*e*sizeof(float));
    memcpy(grown.wh,m->wh,w*w*sizeof(float));memcpy(grown.wm,m->wm,w*w*sizeof(float));
    memcpy(grown.wu,m->wu,w*w*sizeof(float));memcpy(grown.bias,m->bias,w*sizeof(float));
    memcpy(grown.gate_bias,m->gate_bias,w*sizeof(float));
    for(size_t row=0;row<e;row++)
        memcpy(grown.embed+row*new_vocabulary,m->embed+row*old_vocabulary,old_vocabulary*sizeof(float));
    memcpy(grown.head,m->head,old_vocabulary*w*sizeof(float));
    ama10_model_free(m);*m=grown;return 1;
}
void ama10_model_free(AMA10Model *m) {
    if (!m) return;
    free(m->wx); free(m->wh); free(m->wm); free(m->wu);
    free(m->bias); free(m->gate_bias); free(m->embed); free(m->head); memset(m,0,sizeof(*m));
}
static int model_write_u32(FILE *f,uint32_t x){
    unsigned char b[4]={(unsigned char)x,(unsigned char)(x>>8),(unsigned char)(x>>16),(unsigned char)(x>>24)};
    return fwrite(b,1,4,f)==4;
}
static int model_read_u32(FILE *f,uint32_t *x){
    unsigned char b[4];if(fread(b,1,4,f)!=4)return 0;
    *x=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);return 1;
}
static int model_write_floats(FILE *f,const float *p,size_t n){
    for(size_t i=0;i<n;i++){
        uint32_t bits;memcpy(&bits,&p[i],sizeof(bits));
        if(!model_write_u32(f,bits))return 0;
    }
    return 1;
}
static int model_read_floats(FILE *f,float *p,size_t n){
    for(size_t i=0;i<n;i++){
        uint32_t bits;if(!model_read_u32(f,&bits))return 0;
        memcpy(&p[i],&bits,sizeof(bits));if(!isfinite(p[i]))return 0;
    }
    return 1;
}
static void model_parameter_views(AMA10Model *m,float **p,size_t *n){
    size_t w=m->width,e=m->input_width,v=m->vocabulary;
    p[0]=m->wx;n[0]=w*e;p[1]=m->wh;n[1]=w*w;p[2]=m->wm;n[2]=w*w;
    p[3]=m->wu;n[3]=w*w;p[4]=m->bias;n[4]=w;p[5]=m->gate_bias;n[5]=w;
    p[6]=m->embed;n[6]=e*v;p[7]=m->head;n[7]=v*w;
}
typedef struct {const unsigned char *p,*end;} ModelJson;
typedef struct {uint64_t begin,end;size_t shape[2],rank;int seen;} ModelSafeTensor;
static void json_ws(ModelJson *j){while(j->p<j->end&&(*j->p==' '||*j->p=='\t'||*j->p=='\r'||*j->p=='\n'))j->p++;}
static int json_take(ModelJson *j,unsigned char c){json_ws(j);if(j->p>=j->end||*j->p!=c)return 0;j->p++;return 1;}
static int json_read_string(ModelJson *j,char *out,size_t cap){
    json_ws(j);if(j->p>=j->end||*j->p++!='"'||!cap)return 0;size_t n=0;
    while(j->p<j->end&&*j->p!='"'){
        unsigned char c=*j->p++;
        if(c=='\\'){
            if(j->p>=j->end)return 0;
            c=*j->p++;
            if(c=='"'||c=='\\'||c=='/'){}
            else if(c=='b')c='\b';else if(c=='f')c='\f';else if(c=='n')c='\n';else if(c=='r')c='\r';else if(c=='t')c='\t';
            else if(c=='u'){
                if((size_t)(j->end-j->p)<4)return 0;
                unsigned cp=0;
                for(unsigned i=0;i<4;i++){unsigned char h=*j->p++;unsigned x;
                    if(h>='0'&&h<='9')x=h-'0';else if(h>='a'&&h<='f')x=h-'a'+10;
                    else if(h>='A'&&h<='F')x=h-'A'+10;
                    else return 0;
                    cp=(cp<<4)|x;
                }
                if(cp>0x7f||cp<0x20)return 0;
                c=(unsigned char)cp;
            }else return 0;
        }else if(c<0x20)return 0;
        if(n+1>=cap)return 0;
        out[n++]=(char)c;
    }
    if(j->p>=j->end)return 0;
    j->p++;out[n]=0;return 1;
}
static int json_skip_string(ModelJson *j){
    json_ws(j);if(j->p>=j->end||*j->p++!='"')return 0;
    while(j->p<j->end){unsigned char c=*j->p++;if(c=='"')return 1;if(c<0x20)return 0;
        if(c=='\\'){if(j->p>=j->end)return 0;c=*j->p++;if(c=='u'){
            if((size_t)(j->end-j->p)<4)return 0;
            for(unsigned i=0;i<4;i++){unsigned char h=*j->p++;
                if(!((h>='0'&&h<='9')||(h>='a'&&h<='f')||(h>='A'&&h<='F')))return 0;}}
            else if(c!='"'&&c!='\\'&&c!='/'&&c!='b'&&c!='f'&&c!='n'&&c!='r'&&c!='t')return 0;}}
    return 0;
}
static int json_skip_value(ModelJson *j,unsigned depth){
    if(depth>32)return 0;
    json_ws(j);if(j->p>=j->end)return 0;
    if(*j->p=='"')return json_skip_string(j);
    if(*j->p=='{'){
        j->p++;json_ws(j);if(json_take(j,'}'))return 1;
        for(;;){if(!json_skip_string(j)||!json_take(j,':')||!json_skip_value(j,depth+1))return 0;
            json_ws(j);if(json_take(j,'}'))return 1;if(!json_take(j,','))return 0;}
    }
    if(*j->p=='['){j->p++;json_ws(j);if(json_take(j,']'))return 1;
        for(;;){if(!json_skip_value(j,depth+1))return 0;json_ws(j);if(json_take(j,']'))return 1;if(!json_take(j,','))return 0;}}
    const unsigned char *start=j->p;
    if(*j->p=='-'||(*j->p>='0'&&*j->p<='9')){
        if(*j->p=='-')j->p++;
        if(j->p>=j->end)return 0;
        if(*j->p=='0')j->p++;else{if(*j->p<'1'||*j->p>'9')return 0;while(j->p<j->end&&*j->p>='0'&&*j->p<='9')j->p++;}
        if(j->p<j->end&&*j->p=='.'){j->p++;const unsigned char *d=j->p;while(j->p<j->end&&*j->p>='0'&&*j->p<='9')j->p++;if(d==j->p)return 0;}
        if(j->p<j->end&&(*j->p=='e'||*j->p=='E')){j->p++;if(j->p<j->end&&(*j->p=='+'||*j->p=='-'))j->p++;const unsigned char *d=j->p;while(j->p<j->end&&*j->p>='0'&&*j->p<='9')j->p++;if(d==j->p)return 0;}
        return j->p>start;
    }
    static const char *literals[]={"true","false","null"};
    for(size_t i=0;i<3;i++){size_t n=strlen(literals[i]);if((size_t)(j->end-j->p)>=n&&!memcmp(j->p,literals[i],n)){j->p+=n;return 1;}}
    return 0;
}
static int json_uint(ModelJson *j,uint64_t *out){
    json_ws(j);if(j->p>=j->end||*j->p<'0'||*j->p>'9')return 0;
    uint64_t n=0;if(*j->p=='0'){j->p++;if(j->p<j->end&&*j->p>='0'&&*j->p<='9')return 0;*out=0;return 1;}
    while(j->p<j->end&&*j->p>='0'&&*j->p<='9'){unsigned d=*j->p++-'0';if(n>(UINT64_MAX-d)/10)return 0;n=n*10+d;}*out=n;return 1;
}
static int safe_dimension_string(const char *s,uint64_t *out){
    if(!s||!*s)return 0;
    uint64_t n=0;
    for(const unsigned char *p=(const unsigned char*)s;*p;p++){
        if(*p<'0'||*p>'9')return 0;
        unsigned d=*p-'0';if(n>(UINT64_MAX-d)/10)return 0;n=n*10+d;
    }
    if(!n)return 0;
    *out=n;return 1;
}
static int safetensors_group(const char *name){
    static const char *names[]={"wx","wh","wm","wu","bias","gate_bias","embed","head"};
    for(int i=0;i<8;i++)if(strcmp(name,names[i])==0)return i;
    return -1;
}
static int parse_safe_tensor(ModelJson *j,ModelSafeTensor *t){
    if(!json_take(j,'{'))return 0;
    unsigned flags=0;json_ws(j);if(json_take(j,'}'))return 0;
    for(;;){char key[64],value[32];if(!json_read_string(j,key,sizeof(key))||!json_take(j,':'))return 0;
        unsigned bit=0;
        if(strcmp(key,"dtype")==0){bit=1;if(!json_read_string(j,value,sizeof(value))||strcmp(value,"F32")!=0)return 0;}
        else if(strcmp(key,"shape")==0){bit=2;if(!json_take(j,'['))return 0;json_ws(j);t->rank=0;
            if(!json_take(j,']'))for(;;){uint64_t d;if(t->rank>=2||!json_uint(j,&d)||!d||d>SIZE_MAX)return 0;t->shape[t->rank++]=(size_t)d;
                json_ws(j);if(json_take(j,']'))break;if(!json_take(j,','))return 0;}}
        else if(strcmp(key,"data_offsets")==0){bit=4;if(!json_take(j,'[')||!json_uint(j,&t->begin)||!json_take(j,',')||!json_uint(j,&t->end)||!json_take(j,']'))return 0;}
        else if(!json_skip_value(j,0))return 0;
        if(bit&&(flags&bit))return 0;
        flags|=bit;
        json_ws(j);if(json_take(j,'}'))break;if(!json_take(j,','))return 0;
    }
    t->seen=1;return flags==7&&t->rank>0&&t->end>=t->begin;
}
static int parse_safe_metadata(ModelJson *j,uint64_t dims[3],unsigned *dimension_mask){
    if(!json_take(j,'{'))return 0;
    json_ws(j);if(json_take(j,'}'))return 1;
    for(;;){char key[80],value[128];if(!json_read_string(j,key,sizeof(key))||!json_take(j,':'))return 0;
        int known=!strcmp(key,"format")||!strcmp(key,"version")||!strcmp(key,"width")||
                  !strcmp(key,"input_width")||!strcmp(key,"vocabulary");
        if(known){if(!json_read_string(j,value,sizeof(value)))return 0;}
        else if(!json_skip_string(j))return 0;
        unsigned bit=0;uint64_t n=0;
        if(strcmp(key,"format")==0){if(strcmp(value,"ama10")!=0)return 0;bit=8;}
        else if(strcmp(key,"version")==0){if(strcmp(value,"1")!=0)return 0;bit=16;}
        else if(strcmp(key,"width")==0){bit=1;if(!safe_dimension_string(value,&n))return 0;dims[0]=n;}
        else if(strcmp(key,"input_width")==0){bit=2;if(!safe_dimension_string(value,&n))return 0;dims[1]=n;}
        else if(strcmp(key,"vocabulary")==0){bit=4;if(!safe_dimension_string(value,&n))return 0;dims[2]=n;}
        if(bit&&(*dimension_mask&bit))return 0;
        *dimension_mask|=bit;
        json_ws(j);if(json_take(j,'}'))return 1;if(!json_take(j,','))return 0;
    }
}
static int parse_safe_header(const unsigned char *header,size_t header_size,ModelSafeTensor tensors[8],
                             uint64_t dims[3],unsigned *dimension_mask){
    ModelJson j={header,header+header_size};if(!json_take(&j,'{'))return 0;unsigned seen_metadata=0;json_ws(&j);
    if(!json_take(&j,'}'))for(;;){char key[128];if(!json_read_string(&j,key,sizeof(key))||!json_take(&j,':'))return 0;
        if(strcmp(key,"__metadata__")==0){if(seen_metadata++||!parse_safe_metadata(&j,dims,dimension_mask))return 0;}
        else{int group=safetensors_group(key);if(group<0||tensors[group].seen||!parse_safe_tensor(&j,&tensors[group]))return 0;}
        json_ws(&j);if(json_take(&j,'}'))break;if(!json_take(&j,','))return 0;
    }
    json_ws(&j);if(j.p!=j.end)return 0;for(size_t i=0;i<8;i++)if(!tensors[i].seen)return 0;return 1;
}
static int model_safe_tensor_shape(const ModelSafeTensor *t,size_t a,size_t b){
    return b?(t->rank==2&&t->shape[0]==a&&t->shape[1]==b):(t->rank==1&&t->shape[0]==a);
}
static int safe_offset_order(const void *a,const void *b){
    const ModelSafeTensor *const *x=(const ModelSafeTensor *const*)a,*const *y=(const ModelSafeTensor *const*)b;
    return (*x)->begin<(*y)->begin?-1:(*x)->begin>(*y)->begin?1:0;
}
static int write_u64_le(FILE *f,uint64_t x){unsigned char b[8];for(unsigned i=0;i<8;i++)b[i]=(unsigned char)(x>>(8*i));return fwrite(b,1,8,f)==8;}
static int read_u64_le(FILE *f,uint64_t *x){unsigned char b[8];if(fread(b,1,8,f)!=8)return 0;*x=0;for(unsigned i=0;i<8;i++)*x|=(uint64_t)b[i]<<(8*i);return 1;}
int ama10_model_save(const AMA10Model *m,const char *path){
    if(!m||!path||m->vocabulary>UINT32_MAX||m->width>UINT32_MAX||m->input_width>UINT32_MAX||
       sizeof(float)!=4||FLT_RADIX!=2||FLT_MANT_DIG!=24||FLT_MAX_EXP!=128)return 0;
    FILE *f=fopen(path,"wb");if(!f)return 0;
    static const unsigned char magic[8]={'A','M','A','1','0','W','0','3'};
    int ok=fwrite(magic,1,8,f)==8&&model_write_u32(f,(uint32_t)m->vocabulary)&&
           model_write_u32(f,(uint32_t)m->width)&&model_write_u32(f,(uint32_t)m->input_width);
    float *params[8];size_t sizes[8];model_parameter_views((AMA10Model*)m,params,sizes);
    for(size_t i=0;i<8&&ok;i++)ok=model_write_floats(f,params[i],sizes[i]);
    if(fclose(f)!=0)ok=0;
    return ok;
}
int ama10_model_save_safetensors(const AMA10Model *m,const char *path){
    if(!m||!path||!m->vocabulary||!m->width||!m->input_width||m->vocabulary>UINT32_MAX||
       m->width>UINT32_MAX||m->input_width>UINT32_MAX||sizeof(float)!=4||FLT_RADIX!=2||
       FLT_MANT_DIG!=24||FLT_MAX_EXP!=128)return 0;
    float *params[8];size_t sizes[8];model_parameter_views((AMA10Model*)m,params,sizes);uint64_t offsets[8],total=0;
    for(size_t k=0;k<8;k++){
        if(!params[k]||sizes[k]>UINT64_MAX/4||total>UINT64_MAX-(uint64_t)sizes[k]*4)return 0;
        offsets[k]=total;total+=(uint64_t)sizes[k]*4;
        for(size_t i=0;i<sizes[k];i++)if(!isfinite(params[k][i]))return 0;
    }
    char header[2048];int n=snprintf(header,sizeof(header),
        "{\"wx\":{\"dtype\":\"F32\",\"shape\":[%zu,%zu],\"data_offsets\":[%llu,%llu]},"
        "\"wh\":{\"dtype\":\"F32\",\"shape\":[%zu,%zu],\"data_offsets\":[%llu,%llu]},"
        "\"wm\":{\"dtype\":\"F32\",\"shape\":[%zu,%zu],\"data_offsets\":[%llu,%llu]},"
        "\"wu\":{\"dtype\":\"F32\",\"shape\":[%zu,%zu],\"data_offsets\":[%llu,%llu]},"
        "\"bias\":{\"dtype\":\"F32\",\"shape\":[%zu],\"data_offsets\":[%llu,%llu]},"
        "\"gate_bias\":{\"dtype\":\"F32\",\"shape\":[%zu],\"data_offsets\":[%llu,%llu]},"
        "\"embed\":{\"dtype\":\"F32\",\"shape\":[%zu,%zu],\"data_offsets\":[%llu,%llu]},"
        "\"head\":{\"dtype\":\"F32\",\"shape\":[%zu,%zu],\"data_offsets\":[%llu,%llu]},"
        "\"__metadata__\":{\"format\":\"ama10\",\"version\":\"1\",\"width\":\"%zu\",\"input_width\":\"%zu\",\"vocabulary\":\"%zu\"}}",
        m->width,m->input_width,(unsigned long long)offsets[0],(unsigned long long)(offsets[0]+(uint64_t)sizes[0]*4),
        m->width,m->width,(unsigned long long)offsets[1],(unsigned long long)(offsets[1]+(uint64_t)sizes[1]*4),
        m->width,m->width,(unsigned long long)offsets[2],(unsigned long long)(offsets[2]+(uint64_t)sizes[2]*4),
        m->width,m->width,(unsigned long long)offsets[3],(unsigned long long)(offsets[3]+(uint64_t)sizes[3]*4),
        m->width,(unsigned long long)offsets[4],(unsigned long long)(offsets[4]+(uint64_t)sizes[4]*4),
        m->width,(unsigned long long)offsets[5],(unsigned long long)(offsets[5]+(uint64_t)sizes[5]*4),
        m->input_width,m->vocabulary,(unsigned long long)offsets[6],(unsigned long long)(offsets[6]+(uint64_t)sizes[6]*4),
        m->vocabulary,m->width,(unsigned long long)offsets[7],(unsigned long long)(offsets[7]+(uint64_t)sizes[7]*4),
        m->width,m->input_width,m->vocabulary);
    if(n<0||(size_t)n>=sizeof(header))return 0;
    size_t raw_size=(size_t)n,header_size=(raw_size+7u)&~(size_t)7u;
    if(header_size>UINT64_C(100)*1024*1024)return 0;
    memset(header+raw_size,' ',header_size-raw_size);
    size_t path_len=strlen(path);if(path_len>SIZE_MAX-5)return 0;
    char *tmp=(char*)malloc(path_len+5),*backup=(char*)malloc(path_len+5);if(!tmp||!backup){free(tmp);free(backup);return 0;}
    memcpy(tmp,path,path_len);memcpy(tmp+path_len,".tmp",5);memcpy(backup,path,path_len);memcpy(backup+path_len,".bak",5);
    FILE *f=fopen(tmp,"wb");if(!f){free(tmp);free(backup);return 0;}
    int ok=write_u64_le(f,(uint64_t)header_size)&&fwrite(header,1,header_size,f)==header_size;
    for(size_t k=0;k<8&&ok;k++)ok=model_write_floats(f,params[k],sizes[k]);
    if(fflush(f)!=0)ok=0;
    if(fclose(f)!=0)ok=0;
    int had_backup=0;
    if(ok){(void)remove(backup);if(rename(path,backup)==0)had_backup=1;else if(errno!=ENOENT)ok=0;}
    if(ok&&rename(tmp,path)!=0)ok=0;
    if(!ok){(void)remove(tmp);if(had_backup)(void)rename(backup,path);}else if(had_backup)(void)remove(backup);
    free(tmp);free(backup);return ok;
}
int ama10_model_load(AMA10Model *m,const char *path){
    if(!m||!path||sizeof(float)!=4||FLT_RADIX!=2||FLT_MANT_DIG!=24||FLT_MAX_EXP!=128)return 0;
    FILE *f=fopen(path,"rb");if(!f)return 0;
    if(fseek(f,0,SEEK_END)!=0){fclose(f);return 0;}
    long file_size=ftell(f);if(file_size<8||fseek(f,0,SEEK_SET)!=0){fclose(f);return 0;}
    unsigned char magic[8],expected[8]={'A','M','A','1','0','W','0','3'};
    if(fread(magic,1,8,f)!=8){fclose(f);return 0;}
    if(memcmp(magic,expected,8)!=0){fclose(f);return ama10_model_load_safetensors(m,path);}
    if(file_size<20||fseek(f,8,SEEK_SET)!=0){fclose(f);return 0;}
    uint32_t v=0,w=0,e=0;
    int ok=model_read_u32(f,&v)&&model_read_u32(f,&w)&&model_read_u32(f,&e)&&v&&w&&e;
    AMA10Model tmp={0};
    if(ok)ok=ama10_model_init(&tmp,v,w,e,UINT64_C(1));
    if(ok){
        float *params[8];size_t sizes[8];model_parameter_views(&tmp,params,sizes);
        uint64_t values=0;for(size_t i=0;i<8;i++)values+=(uint64_t)sizes[i];
        if(values>(uint64_t)(LONG_MAX-20)/4||(uint64_t)file_size!=20+values*4)ok=0;
        for(size_t i=0;i<8&&ok;i++)ok=model_read_floats(f,params[i],sizes[i]);
    }
    if(ok&&fgetc(f)!=EOF)ok=0;
    if(fclose(f)!=0)ok=0;
    if(ok){ama10_model_free(m);*m=tmp;}else ama10_model_free(&tmp);
    return ok;
}
int ama10_model_load_safetensors(AMA10Model *m,const char *path){
    if(!m||!path||sizeof(float)!=4||FLT_RADIX!=2||FLT_MANT_DIG!=24||FLT_MAX_EXP!=128)return 0;
    FILE *f=fopen(path,"rb");if(!f)return 0;if(fseek(f,0,SEEK_END)!=0){fclose(f);return 0;}
    long file_size=ftell(f);if(file_size<8||fseek(f,0,SEEK_SET)!=0){fclose(f);return 0;}
    uint64_t header_length=0;int ok=read_u64_le(f,&header_length)&&header_length>=2&&
        header_length<=UINT64_C(100)*1024*1024&&header_length%8==0&&header_length<=(uint64_t)file_size-8;
    unsigned char *header=NULL;if(ok){header=(unsigned char*)malloc((size_t)header_length);if(!header)ok=0;}
    if(ok&&fread(header,1,(size_t)header_length,f)!=(size_t)header_length)ok=0;
    ModelSafeTensor tensors[8]={0};uint64_t dims[3]={0};unsigned metadata_mask=0;
    if(ok)ok=parse_safe_header(header,(size_t)header_length,tensors,dims,&metadata_mask);
    free(header);if(!ok){fclose(f);return 0;}
    uint64_t width=tensors[4].shape[0],input_width=tensors[0].shape[1],vocabulary=tensors[6].shape[1];
    if(!model_safe_tensor_shape(&tensors[0],(size_t)tensors[4].shape[0],(size_t)input_width)||
       !model_safe_tensor_shape(&tensors[1],(size_t)width,(size_t)width)||
       !model_safe_tensor_shape(&tensors[2],(size_t)width,(size_t)width)||
       !model_safe_tensor_shape(&tensors[3],(size_t)width,(size_t)width)||
       !model_safe_tensor_shape(&tensors[4],(size_t)width,0)||!model_safe_tensor_shape(&tensors[5],(size_t)width,0)||
       !model_safe_tensor_shape(&tensors[6],(size_t)input_width,(size_t)vocabulary)||
       !model_safe_tensor_shape(&tensors[7],(size_t)vocabulary,(size_t)width)||
       width>UINT32_MAX||input_width>UINT32_MAX||vocabulary>UINT32_MAX||
       ((metadata_mask&7u)&&((metadata_mask&7u)!=7u||dims[0]!=width||dims[1]!=input_width||dims[2]!=vocabulary))){fclose(f);return 0;}
    size_t w=(size_t)width,e=(size_t)input_width,v=(size_t)vocabulary;
    AMA10Model tmp={0};if(!ama10_model_init(&tmp,v,w,e,UINT64_C(1))){fclose(f);return 0;}
    float *params[8];size_t sizes[8];model_parameter_views(&tmp,params,sizes);
    const ModelSafeTensor *ordered[8];for(size_t k=0;k<8;k++)ordered[k]=&tensors[k];qsort(ordered,8,sizeof(ordered[0]),safe_offset_order);
    uint64_t cursor=0;uint64_t data_start=8+header_length;uint64_t data_length=(uint64_t)file_size-data_start;
    for(size_t k=0;k<8&&ok;k++){
        const ModelSafeTensor *t=ordered[k];if(t->begin!=cursor||t->end<t->begin){ok=0;break;}
        uint64_t expected_bytes=0;int group=-1;for(int i=0;i<8;i++)if(&tensors[i]==t){group=i;break;}
        if(group<0||sizes[group]>UINT64_MAX/4){ok=0;break;}expected_bytes=(uint64_t)sizes[group]*4;
        if(t->end-t->begin!=expected_bytes||t->end>data_length){ok=0;break;}cursor=t->end;
    }
    if(cursor!=data_length)ok=0;
    for(size_t k=0;k<8&&ok;k++){
        if(data_start+tensors[k].begin>(uint64_t)LONG_MAX||fseek(f,(long)(data_start+tensors[k].begin),SEEK_SET)!=0||
           !model_read_floats(f,params[k],sizes[k]))ok=0;
    }
    if(fclose(f)!=0)ok=0;
    if(ok){ama10_model_free(m);*m=tmp;}else ama10_model_free(&tmp);return ok;
}
int ama10_state_init(AMA10State *s,size_t w) {
    if(!s||!w||w>SIZE_MAX/sizeof(float)) return 0;
    memset(s,0,sizeof(*s)); s->fast=(float*)calloc(w,sizeof(float)); s->slow=(float*)calloc(w,sizeof(float));
    if(!s->fast||!s->slow){ama10_state_free(s);return 0;} s->width=w; return 1;
}
void ama10_state_reset(AMA10State *s){if(s&&s->fast&&s->slow){memset(s->fast,0,s->width*sizeof(float));memset(s->slow,0,s->width*sizeof(float));s->steps=0;}}
void ama10_state_free(AMA10State *s){if(s){free(s->fast);free(s->slow);memset(s,0,sizeof(*s));}}

static float sigmoid(float x){if(x>=0){float z=expf(-x);return 1.0f/(1.0f+z);}float z=expf(x);return z/(1.0f+z);}
int ama10_step_features(const AMA10Model *m,AMA10State *s,const float *x,float *logits,
                        unsigned max_depth,float threshold){
    if(!m||!s||!x||!logits||!s->fast||!s->slow||s->width!=m->width||
       !max_depth||!isfinite(threshold)||threshold<0||threshold>2)return 0;
    size_t w=m->width,e=m->input_width;
    for(size_t j=0;j<e;j++)if(!isfinite(x[j]))return 0;
    float *next=(float*)malloc(w*sizeof(float)),*proposal=(float*)malloc(w*sizeof(float));
    if(!next||!proposal){free(next);free(proposal);return 0;}
    for(unsigned d=0;d<max_depth;d++){
        float activity=0;
        for(size_t i=0;i<w;i++){
            float drive=m->bias[i],input_drive=0.0f;
            for(size_t j=0;j<e;j++){
                float projected=m->wx[i*e+j]*x[j];
                drive+=projected;input_drive+=projected;
            }
            for(size_t j=0;j<w;j++)drive+=m->wh[i*w+j]*s->fast[j]+m->wm[i*w+j]*s->slow[j];
            float gate=m->gate_bias[i]+input_drive;
            for(size_t j=0;j<w;j++)gate+=m->wu[i*w+j]*s->fast[j];
            float g=sigmoid(gate);
            proposal[i]=tanhf(drive); next[i]=(1.0f-g)*s->fast[i]+g*proposal[i]; activity+=fabsf(next[i]-s->fast[i]);
        }
        memcpy(s->fast,next,w*sizeof(float));
        s->steps++;
        float confidence=1.0f/(1.0f+activity/(float)w);
        if(d+1>=2 && threshold<=1.0f && confidence>=threshold)break;
    }
    /* Slow memory advances once per token, independent of recurrent depth. */
    for(size_t i=0;i<w;i++)s->slow[i]=0.995f*s->slow[i]+0.005f*s->fast[i];
    for(size_t k=0;k<m->vocabulary;k++){
        float z=0;for(size_t j=0;j<w;j++)z+=m->head[k*w+j]*s->fast[j];logits[k]=z;
    }
    free(next);free(proposal);return 1;
}
int ama10_step(const AMA10Model *m,AMA10State *s,uint32_t token,float *logits,
               unsigned max_depth,float threshold){
    if(!m||token>=m->vocabulary)return 0;
    float *x=(float*)malloc(m->input_width*sizeof(float));if(!x)return 0;
    for(size_t j=0;j<m->input_width;j++)x[j]=m->embed[j*m->vocabulary+token];
    int ok=ama10_step_features(m,s,x,logits,max_depth,threshold);free(x);return ok;
}
int ama10_step_hidden(const AMA10Model *m,AMA10State *s,uint32_t token,float *logits,
                      float *hidden,unsigned max_depth,float threshold){
    if(!hidden||!ama10_step(m,s,token,logits,max_depth,threshold))return 0;
    memcpy(hidden,s->fast,m->width*sizeof(float));return 1;
}

#define AMA_PARAMS 8
static void parameter_views(AMA10Model *m, float **p, size_t *n) {
    p[0]=m->wx; n[0]=m->width*m->input_width;
    p[1]=m->wh; n[1]=m->width*m->width;
    p[2]=m->wm; n[2]=m->width*m->width;
    p[3]=m->wu; n[3]=m->width*m->width;
    p[4]=m->bias; n[4]=m->width;
    p[5]=m->gate_bias; n[5]=m->width;
    p[6]=m->embed; n[6]=m->input_width*m->vocabulary;
    p[7]=m->head; n[7]=m->vocabulary*m->width;
}
int ama10_optimizer_init(AMA10Optimizer *o,const AMA10Model *m) {
    if(!o||!m)return 0;
    memset(o,0,sizeof(*o));
    o->first_moment=(float**)calloc(AMA_PARAMS,sizeof(float*));
    o->second_moment=(float**)calloc(AMA_PARAMS,sizeof(float*));
    if(!o->first_moment||!o->second_moment){ama10_optimizer_free(o);return 0;}
    o->parameter_count=AMA_PARAMS;
    size_t n[AMA_PARAMS];float *p[AMA_PARAMS];parameter_views((AMA10Model*)m,p,n);
    for(size_t k=0;k<AMA_PARAMS;k++){
        if(n[k]>SIZE_MAX/sizeof(float)){ama10_optimizer_free(o);return 0;}
        o->first_moment[k]=(float*)calloc(n[k],sizeof(float));
        o->second_moment[k]=(float*)calloc(n[k],sizeof(float));
        if(!o->first_moment[k]||!o->second_moment[k]){ama10_optimizer_free(o);return 0;}
    }
    return 1;
}
int ama10_optimizer_set_algorithm(AMA10Optimizer *o,uint32_t algorithm){
    if(!o||!o->first_moment||!o->second_moment||o->step||
       (algorithm!=AMA10_OPT_ADAMW&&algorithm!=AMA10_OPT_MUON))return 0;
    o->algorithm=algorithm;return 1;
}
int ama10_optimizer_resize_vocabulary(AMA10Optimizer *o,const AMA10Model *old_m,const AMA10Model *new_m){
    if(!o||!o->first_moment||!o->second_moment||!old_m||!new_m||
       old_m->vocabulary>=new_m->vocabulary||old_m->width!=new_m->width||
       old_m->input_width!=new_m->input_width||o->parameter_count!=AMA_PARAMS||
       (o->algorithm!=AMA10_OPT_ADAMW&&o->algorithm!=AMA10_OPT_MUON))return 0;
    AMA10Optimizer grown={0};
    if(!ama10_optimizer_init(&grown,new_m))return 0;
    for(size_t k=0;k<6;k++){
        size_t n=k==0?old_m->width*old_m->input_width:
            k<=3?old_m->width*old_m->width:old_m->width;
        memcpy(grown.first_moment[k],o->first_moment[k],n*sizeof(float));
        memcpy(grown.second_moment[k],o->second_moment[k],n*sizeof(float));
    }
    for(size_t row=0;row<old_m->input_width;row++){
        size_t src=row*old_m->vocabulary,dst=row*new_m->vocabulary;
        memcpy(grown.first_moment[6]+dst,o->first_moment[6]+src,old_m->vocabulary*sizeof(float));
        memcpy(grown.second_moment[6]+dst,o->second_moment[6]+src,old_m->vocabulary*sizeof(float));
    }
    size_t head_count=old_m->vocabulary*old_m->width;
    memcpy(grown.first_moment[7],o->first_moment[7],head_count*sizeof(float));
    memcpy(grown.second_moment[7],o->second_moment[7],head_count*sizeof(float));
    grown.algorithm=o->algorithm;grown.step=o->step;
    ama10_optimizer_free(o);*o=grown;return 1;
}
void ama10_optimizer_free(AMA10Optimizer *o){
    if(!o)return;
    if(o->first_moment)for(size_t i=0;i<o->parameter_count;i++)free(o->first_moment[i]);
    if(o->second_moment)for(size_t i=0;i<o->parameter_count;i++)free(o->second_moment[i]);
    free(o->first_moment);free(o->second_moment);memset(o,0,sizeof(*o));
}

int ama10_gradients_init(AMA10Gradients *g,const AMA10Model *m){
    if(!g||!m)return 0;
    memset(g,0,sizeof(*g));
    float *unused[AMA_PARAMS];size_t sizes[AMA_PARAMS];
    parameter_views((AMA10Model*)m,unused,sizes);
    g->values=(float**)calloc(AMA_PARAMS,sizeof(float*));
    if(!g->values)return 0;
    g->parameter_count=AMA_PARAMS;g->width=m->width;
    g->input_width=m->input_width;g->vocabulary=m->vocabulary;
    for(size_t k=0;k<AMA_PARAMS;k++){
        if(sizes[k]>SIZE_MAX/sizeof(float))goto fail;
        g->values[k]=(float*)calloc(sizes[k],sizeof(float));
        if(!g->values[k])goto fail;
    }
    return 1;
fail:
    ama10_gradients_free(g);return 0;
}
void ama10_gradients_reset(AMA10Gradients *g){
    if(!g||!g->values)return;
    float *unused[AMA_PARAMS];size_t sizes[AMA_PARAMS];
    AMA10Model shape={0};shape.width=g->width;shape.input_width=g->input_width;
    shape.vocabulary=g->vocabulary;parameter_views(&shape,unused,sizes);
    for(size_t k=0;k<g->parameter_count;k++)if(g->values[k])memset(g->values[k],0,sizes[k]*sizeof(float));
    g->tokens=0;
}
void ama10_gradients_free(AMA10Gradients *g){
    if(!g)return;
    if(g->values)for(size_t k=0;k<g->parameter_count;k++)free(g->values[k]);
    free(g->values);memset(g,0,sizeof(*g));
}

float ama10_learning_rate(float base,uint64_t update,float decay){
    if(!isfinite(base)||base<=0.0f||!update||!isfinite(decay)||decay<0.0f)return 0.0f;
    double denominator=sqrt(1.0+(double)decay*(double)(update-1));
    double rate=(double)base/denominator;
    return isfinite(rate)&&rate>0.0?(float)rate:0.0f;
}

static int muon_orthogonalize(const float *input,size_t rows,size_t cols,float *output){
    const float a=3.4445f,b=-4.7750f,c=2.0315f;
    size_t r=rows<cols?rows:cols,q=rows<cols?cols:rows;
    if(!rows||!cols||r>SIZE_MAX/r||r*q>SIZE_MAX/sizeof(float)||r*r>SIZE_MAX/sizeof(float))return 0;
    float *x=(float*)malloc(r*q*sizeof(float)),*next=(float*)malloc(r*q*sizeof(float));
    float *gram=(float*)malloc(r*r*sizeof(float)),*gram2=(float*)malloc(r*r*sizeof(float));
    if(!x||!next||!gram||!gram2){free(x);free(next);free(gram);free(gram2);return 0;}
    int transposed=rows>cols;
    double norm2=0;
    for(size_t i=0;i<r;i++)for(size_t j=0;j<q;j++){
        float v=transposed?input[j*cols+i]:input[i*cols+j];
        x[i*q+j]=v;norm2+=(double)v*v;
    }
    if(!isfinite(norm2)){free(x);free(next);free(gram);free(gram2);return 0;}
    float inv_norm=(float)(1.0/(sqrt(norm2)+1.0e-7));
    for(size_t i=0;i<r*q;i++)x[i]*=inv_norm;
    for(unsigned step=0;step<5;step++){
        for(size_t i=0;i<r;i++)for(size_t j=0;j<r;j++){
            double sum=0;for(size_t k=0;k<q;k++)sum+=(double)x[i*q+k]*x[j*q+k];
            gram[i*r+j]=(float)sum;
        }
        for(size_t i=0;i<r;i++)for(size_t j=0;j<r;j++){
            double sum=0;for(size_t k=0;k<r;k++)sum+=(double)gram[i*r+k]*gram[k*r+j];
            gram2[i*r+j]=(float)sum;
        }
        for(size_t i=0;i<r;i++)for(size_t j=0;j<q;j++){
            double ax=0,a2x=0;
            for(size_t k=0;k<r;k++){ax+=(double)gram[i*r+k]*x[k*q+j];a2x+=(double)gram2[i*r+k]*x[k*q+j];}
            next[i*q+j]=a*x[i*q+j]+b*(float)ax+c*(float)a2x;
        }
        float *swap=x;x=next;next=swap;
    }
    if(transposed){for(size_t i=0;i<rows;i++)for(size_t j=0;j<cols;j++)output[i*cols+j]=x[j*q+i];}
    else memcpy(output,x,rows*cols*sizeof(float));
    free(x);free(next);free(gram);free(gram2);return 1;
}

int ama10_optimizer_apply(AMA10Model *m,AMA10Optimizer *o,AMA10Gradients *g,float lr,float wd){
    const float beta1=0.9f,beta2=0.999f,eps=1.0e-8f;
    if(!m||!o||!g||!g->values||!g->tokens||g->parameter_count!=AMA_PARAMS||
       g->width!=m->width||g->input_width!=m->input_width||g->vocabulary!=m->vocabulary||
       !o->first_moment||!o->second_moment||o->parameter_count!=AMA_PARAMS||
       (o->algorithm!=AMA10_OPT_ADAMW&&o->algorithm!=AMA10_OPT_MUON)||
       !isfinite(lr)||lr<=0||!isfinite(wd)||wd<0||o->step==UINT64_MAX)return 0;
    float *params[AMA_PARAMS];size_t sizes[AMA_PARAMS];parameter_views(m,params,sizes);
    double norm2=0;
    for(size_t k=0;k<AMA_PARAMS;k++)for(size_t i=0;i<sizes[k];i++){
        double avg=(double)g->values[k][i]/(double)g->tokens;
        if(!isfinite(avg))return 0;
        norm2+=avg*avg;
    }
    if(!isfinite(norm2))return 0;
    float scale=norm2>1.0?(float)(1.0/sqrt(norm2)):1.0f;
    uint64_t next_step=o->step+1;
    double c1=1.0-pow((double)beta1,(double)next_step);
    double c2=1.0-pow((double)beta2,(double)next_step);
    float *muon_direction[4]={0};size_t matrix_sizes[4]={sizes[0],sizes[1],sizes[2],sizes[3]};
    if(o->algorithm==AMA10_OPT_MUON){
        for(size_t k=0;k<4;k++){
            if(matrix_sizes[k]>SIZE_MAX/sizeof(float))goto muon_fail;
            float *direction=(float*)malloc(matrix_sizes[k]*sizeof(float));
            if(!direction)goto muon_fail;
            muon_direction[k]=direction;
            float *buffer=o->first_moment[k];
            for(size_t i=0;i<matrix_sizes[k];i++){
                float grad=(float)((double)g->values[k][i]/(double)g->tokens)*scale;
                buffer[i]=0.95f*buffer[i]+0.05f*grad;
                direction[i]=0.05f*grad+0.95f*buffer[i];
            }
            size_t rows=k==0?m->width:m->width;
            size_t cols=k==0?m->input_width:m->width;
            if(!muon_orthogonalize(direction,rows,cols,direction))goto muon_fail;
        }
        for(size_t k=0;k<4;k++){
            size_t rows=m->width,cols=k==0?m->input_width:m->width;
            float shape_scale=rows>cols?sqrtf((float)rows/(float)cols):1.0f;
            for(size_t i=0;i<matrix_sizes[k];i++)
                params[k][i]=params[k][i]*(1.0f-lr*wd)-lr*shape_scale*muon_direction[k][i];
            free(muon_direction[k]);muon_direction[k]=NULL;
        }
    }
    for(size_t k=0;k<AMA_PARAMS;k++)for(size_t i=0;i<sizes[k];i++){
        if(o->algorithm==AMA10_OPT_MUON&&k<4)continue;
        float group_lr=o->algorithm==AMA10_OPT_MUON?lr*0.015f:lr;
        float grad=(float)((double)g->values[k][i]/(double)g->tokens)*scale;
        float *a=&o->first_moment[k][i],*b=&o->second_moment[k][i];
        *a=beta1**a+(1.0f-beta1)*grad;*b=beta2**b+(1.0f-beta2)*grad*grad;
        float update=(float)(((double)*a/c1)/(sqrt((double)*b/c2)+eps));
        params[k][i]-=group_lr*(update+wd*params[k][i]);
    }
    o->step=next_step;ama10_gradients_reset(g);return 1;
muon_fail:
    for(size_t k=0;k<4;k++)free(muon_direction[k]);
    return 0;
}

static int train_sequence_internal(AMA10Model *m,AMA10Optimizer *opt,AMA10State *state,
                         const uint32_t *input,const float *input_features,
                         const unsigned char *feature_mask,const unsigned char *loss_mask,
                         const uint32_t *target,const float *soft_targets,size_t len,
                         unsigned depth,float halt_threshold,float lr,float wd,
                         AMA10Gradients *accumulator,float *feature_gradients,
                         float *mean_loss,float *mean_depth,void *opencl_handle){
    void *opencl=opencl_handle;
    const float rho=0.995f;
    if(!m||!opt||!m->width||!m->input_width||!m->vocabulary||
       (!input&&!input_features)||(!input&&feature_mask)||(input&&input_features&&!feature_mask)||
       (feature_mask&&!input_features)||(!target&&!soft_targets)||!len||!depth||
       (feature_gradients&&!input_features)||(input_features&&!feature_gradients)||
       (halt_threshold>=0.0f&&(!isfinite(halt_threshold)||halt_threshold>2.0f))||
       (!accumulator&&(!isfinite(lr)||lr<=0||!isfinite(wd)||wd<0))||
       opt->parameter_count!=AMA_PARAMS||!opt->first_moment||!opt->second_moment||
       (accumulator&&(!accumulator->values||accumulator->parameter_count!=AMA_PARAMS||
        accumulator->width!=m->width||accumulator->input_width!=m->input_width||
        accumulator->vocabulary!=m->vocabulary))||
       (state&&(!state->fast||!state->slow||state->width!=m->width)))return 0;
    size_t w=m->width,e=m->input_width,v=m->vocabulary;
    if(soft_targets&&(len>SIZE_MAX/v||len*v>SIZE_MAX/sizeof(float)))return 0;
    if(len>SIZE_MAX/e||len*e>SIZE_MAX/sizeof(float)||len>SIZE_MAX/w||len*w>SIZE_MAX/sizeof(float))return 0;
    size_t supervised_tokens=0;
    for(size_t t=0;t<len;t++)if(!loss_mask||loss_mask[t]){
        if(soft_targets){double mass=0.0;for(size_t k=0;k<v;k++){
            float q=soft_targets[t*v+k];if(!isfinite(q)||q<0.0f)return 0;mass+=q;
        }if(fabs(mass-1.0)>1e-4)return 0;}
        else if(target[t]>=v)return 0;
        supervised_tokens++;
    }
    if(!supervised_tokens||(accumulator&&accumulator->tokens>UINT64_MAX-supervised_tokens))return 0;
    if(feature_gradients)memset(feature_gradients,0,len*e*sizeof(float));
    if(len>SIZE_MAX/depth)return 0;
    if(len>UINT_MAX/depth||(state&&len>(size_t)(UINT_MAX-state->steps)/depth))return 0;
    size_t cells=len*depth;
    if(cells>SIZE_MAX/8 || cells*8>SIZE_MAX/w || cells*8*w>SIZE_MAX/sizeof(float) ||
       len>SIZE_MAX/v || len*v>SIZE_MAX/sizeof(float))return 0;
#ifdef AMA10_ENABLE_OPENCL
    float *cache=opencl?NULL:(float*)calloc(cells*8*w,sizeof(float));
#else
    float *cache=(float*)calloc(cells*8*w,sizeof(float));
#endif
    float *logits=(float*)malloc(len*v*sizeof(float));
    float *input_vectors=NULL,*input_projection=NULL;
#ifdef AMA10_ENABLE_OPENCL
    if(!opencl){input_vectors=(float*)malloc(len*e*sizeof(float));input_projection=(float*)malloc(len*w*sizeof(float));}
#else
    input_vectors=(float*)malloc(len*e*sizeof(float));input_projection=(float*)malloc(len*w*sizeof(float));
#endif
    float *head_hidden=NULL,*head_errors=NULL,*head_gradient=NULL,*head_hidden_gradient=NULL;
#ifdef AMA10_ENABLE_OPENCL
    if(opencl){
        head_hidden=(float*)malloc(len*w*sizeof(float));
        head_errors=(float*)calloc(len*v,sizeof(float));
        head_gradient=(float*)malloc(v*w*sizeof(float));
        head_hidden_gradient=(float*)malloc(len*w*sizeof(float));
    }
#endif
    unsigned *used_depth=(unsigned*)malloc(len*sizeof(unsigned));
    float *grads[AMA_PARAMS]={0};size_t pn[AMA_PARAMS];float *params[AMA_PARAMS];
    parameter_views(m,params,pn);
    for(size_t k=0;k<AMA_PARAMS;k++)grads[k]=(float*)calloc(pn[k],sizeof(float));
    float *dh=(float*)calloc(w,sizeof(float)),*dm=(float*)calloc(w,sizeof(float));
    float *dh0=(float*)calloc(w,sizeof(float)),*dm0=(float*)calloc(w,sizeof(float));
    float *dpre=(float*)malloc(w*sizeof(float)),*dgate=(float*)malloc(w*sizeof(float));
    if((!cache&&!opencl)||!logits||!used_depth||!dh||!dm||!dh0||!dm0||!dpre||!dgate||
#ifdef AMA10_ENABLE_OPENCL
       ((!opencl)&&(!input_vectors||!input_projection))||
#else
       (!input_vectors||!input_projection)||
#endif
#ifdef AMA10_ENABLE_OPENCL
       (opencl&&(!head_hidden||!head_errors||!head_gradient||!head_hidden_gradient))||
#endif
       0)goto fail;
    for(size_t k=0;k<AMA_PARAMS;k++)if(!grads[k])goto fail;
#define C(t,d,f,i) cache[((((t)*(size_t)depth+(d))*8+(f))*w)+(i)]
    float loss=0.0f;
#ifdef AMA10_ENABLE_OPENCL
    if(opencl){
        for(size_t t=0;t<len;t++){if(input[t]>=v)goto fail;used_depth[t]=depth;}
        AMA10OpenCL *backend=(AMA10OpenCL*)opencl;
        const float *initial_fast=state?state->fast:dh;
        const float *initial_slow=state?state->slow:dm;
        if(!ama10_opencl_train_forward(backend,m,input,len,depth,initial_fast,initial_slow,
                                       head_hidden,dm))goto fail;
    }else
#endif
    for(size_t t=0;t<len;t++){
        int use_feature=input_features&&(!feature_mask||feature_mask[t]);
        if(!use_feature&&(!input||input[t]>=v))goto fail;
        for(size_t j=0;j<e;j++){
            float x=use_feature?input_features[t*e+j]:m->embed[j*v+input[t]];
            if(use_feature&&!isfinite(x))goto fail;
            input_vectors[t*e+j]=x;
        }
        for(size_t i=0;i<w;i++){
            float projected=0.0f;
            for(size_t j=0;j<e;j++)projected+=m->wx[i*e+j]*input_vectors[t*e+j];
            input_projection[t*w+i]=projected;
        }
        used_depth[t]=depth;
        for(unsigned d=0;d<depth;d++){
            for(size_t i=0;i<w;i++){
                unsigned previous_depth=t?used_depth[t-1]:0;
                C(t,d,0,i)=d?C(t,d-1,6,i):(t?C(t-1,previous_depth-1,6,i):(state?state->fast[i]:0.0f));
                C(t,d,1,i)=d?C(t,d-1,1,i):(t?C(t-1,previous_depth-1,7,i):(state?state->slow[i]:0.0f));
            }
            for(size_t i=0;i<w;i++){
                float z=m->bias[i]+input_projection[t*w+i];
                float q=m->gate_bias[i]+input_projection[t*w+i];
                for(size_t j=0;j<w;j++){
                    z+=m->wh[i*w+j]*C(t,d,0,j)+m->wm[i*w+j]*C(t,d,1,j);
                    q+=m->wu[i*w+j]*C(t,d,0,j);
                }
                float proposal=tanhf(z),gate=sigmoid(q);
                C(t,d,2,i)=z;C(t,d,3,i)=proposal;C(t,d,4,i)=q;C(t,d,5,i)=gate;
                C(t,d,6,i)=(1.0f-gate)*C(t,d,0,i)+gate*proposal;
                C(t,d,7,i)=C(t,d,1,i);
            }
            if(halt_threshold>=0.0f&&d+1>=2){
                float activity=0.0f;
                for(size_t i=0;i<w;i++)activity+=fabsf(C(t,d,6,i)-C(t,d,0,i));
                float confidence=1.0f/(1.0f+activity/(float)w);
                if(confidence>=halt_threshold){used_depth[t]=d+1;break;}
            }
        }
        for(size_t i=0;i<w;i++)C(t,used_depth[t]-1,7,i)=rho*C(t,used_depth[t]-1,1,i)+(1.0f-rho)*C(t,used_depth[t]-1,6,i);
        if(opencl)continue;
        if(loss_mask&&!loss_mask[t])continue;
        float mx=-INFINITY;
        for(size_t k=0;k<v;k++){
            float z=0;for(size_t j=0;j<w;j++)z+=m->head[k*w+j]*C(t,used_depth[t]-1,6,j);
            logits[t*v+k]=z;if(z>mx)mx=z;
        }
        double sum=0;for(size_t k=0;k<v;k++)sum+=exp((double)logits[t*v+k]-(double)mx);
        if(!(sum>0)||!isfinite(sum))goto fail;
        double log_sum=log(sum);
        if(soft_targets){for(size_t k=0;k<v;k++)if(soft_targets[t*v+k]>0.0f)
            loss-=soft_targets[t*v+k]*(float)((double)logits[t*v+k]-(double)mx-log_sum);}
        else loss+=-(float)((double)logits[t*v+target[t]]-(double)mx-log_sum);
        for(size_t k=0;k<v;k++)logits[t*v+k]=(float)(exp((double)logits[t*v+k]-(double)mx)/sum);
    }
#ifdef AMA10_ENABLE_OPENCL
    if(opencl){
        if(!ama10_opencl_head_forward(opencl,m->head,head_hidden,len,logits))goto fail;
        for(size_t t=0;t<len;t++)if(!loss_mask||loss_mask[t]){
            float mx=logits[t*v];for(size_t k=1;k<v;k++)if(logits[t*v+k]>mx)mx=logits[t*v+k];
            double sum=0;for(size_t k=0;k<v;k++)sum+=exp((double)logits[t*v+k]-(double)mx);
            if(!(sum>0)||!isfinite(sum))goto fail;
            if(soft_targets){for(size_t k=0;k<v;k++)if(soft_targets[t*v+k]>0.0f)
                loss-=soft_targets[t*v+k]*(float)((double)logits[t*v+k]-(double)mx-log(sum));}
            else loss+=-(float)((double)logits[t*v+target[t]]-(double)mx-log(sum));
            for(size_t k=0;k<v;k++)head_errors[t*v+k]=(float)(exp((double)logits[t*v+k]-(double)mx)/sum)-
                (soft_targets?soft_targets[t*v+k]:(k==target[t]?1.0f:0.0f));
        }
        if(!ama10_opencl_head_backward(opencl,m->head,head_hidden,head_errors,len,head_gradient,head_hidden_gradient))goto fail;
        memcpy(grads[7],head_gradient,v*w*sizeof(float));
    }
#endif
    /* Reverse time only after the complete unrolled trajectory has been recorded. */
#ifdef AMA10_ENABLE_OPENCL
    if(opencl){
        AMA10OpenCL *backend=(AMA10OpenCL*)opencl;
        if(!ama10_opencl_bptt_begin(backend,m,cache,input,len,depth,head_hidden_gradient))goto fail;
        for(size_t revt=len;revt>0;revt--){size_t t=revt-1;
            for(unsigned rev=used_depth[t];rev>0;rev--){unsigned d=rev-1;
                if(!ama10_opencl_bptt_step(backend,t,d,d+1==used_depth[t]))goto fail;
            }
        }
        if(!ama10_opencl_bptt_finish(backend,&(AMA10Gradients){.values=grads,.parameter_count=AMA_PARAMS,
            .width=w,.input_width=e,.vocabulary=v}))goto fail;
    }else
#endif
    for(size_t revt=len;revt>0;revt--){size_t t=revt-1;
        if(!loss_mask||loss_mask[t]){
            for(size_t k=0;k<v;k++){
                float dz=logits[t*v+k]-(soft_targets?soft_targets[t*v+k]:(k==target[t]?1.0f:0.0f));
                for(size_t j=0;j<w;j++){
                    grads[7][k*w+j]+=dz*C(t,used_depth[t]-1,6,j);
                    dh[j]+=m->head[k*w+j]*dz;
                }
            }
        }
        for(unsigned rev=used_depth[t];rev>0;rev--){unsigned d=rev-1;
            memset(dh0,0,w*sizeof(float));memset(dm0,0,w*sizeof(float));
            for(size_t i=0;i<w;i++){
                float h=C(t,d,0,i),proposal=C(t,d,3,i),gate=C(t,d,5,i);
                int final_depth=(d+1==used_depth[t]);
                float dnew=dh[i]+(final_depth?(1.0f-rho)*dm[i]:0.0f);
                dm0[i]+=final_depth?rho*dm[i]:dm[i];
                float dp=dnew*gate;
                float dg=dnew*(proposal-h);
                dpre[i]=dp*(1.0f-proposal*proposal);
                dgate[i]=dg*gate*(1.0f-gate);
                dh0[i]+=dnew*(1.0f-gate);
                grads[4][i]+=dpre[i];grads[5][i]+=dgate[i];
            }
            for(size_t i=0;i<w;i++)for(size_t j=0;j<w;j++){
                grads[1][i*w+j]+=dpre[i]*C(t,d,0,j);
                grads[2][i*w+j]+=dpre[i]*C(t,d,1,j);
                grads[3][i*w+j]+=dgate[i]*C(t,d,0,j);
                dh0[j]+=m->wh[i*w+j]*dpre[i]+m->wu[i*w+j]*dgate[i];
                dm0[j]+=m->wm[i*w+j]*dpre[i];
            }
            for(size_t i=0;i<w;i++)for(size_t j=0;j<e;j++){
                size_t ix=i*e+j;float input_grad=dpre[i]+dgate[i];
                int use_feature=input_features&&(!feature_mask||feature_mask[t]);
                float x=input_vectors[t*e+j];
                grads[0][ix]+=input_grad*x;
                if(use_feature)feature_gradients[t*e+j]+=m->wx[ix]*input_grad;
                else grads[6][j*v+input[t]]+=m->wx[ix]*input_grad;
            }
            memcpy(dh,dh0,w*sizeof(float));memcpy(dm,dm0,w*sizeof(float));
        }
    }
    if(feature_gradients)
        for(size_t i=0;i<len*e;i++)feature_gradients[i]/=(float)supervised_tokens;
    loss/=(float)supervised_tokens;
    {
        unsigned total_steps=0;for(size_t t=0;t<len;t++)total_steps+=used_depth[t];
        if(mean_depth)*mean_depth=(float)total_steps/(float)len;
        if(state){
#ifdef AMA10_ENABLE_OPENCL
        if(opencl){memcpy(state->fast,head_hidden+(len-1)*w,w*sizeof(float));memcpy(state->slow,dm,w*sizeof(float));}
        else{
#endif
        memcpy(state->fast,&cache[(((len-1)*depth+(used_depth[len-1]-1))*8+6)*w],w*sizeof(float));
        memcpy(state->slow,&cache[(((len-1)*depth+(used_depth[len-1]-1))*8+7)*w],w*sizeof(float));
#ifdef AMA10_ENABLE_OPENCL
        }
#endif
        state->steps+=total_steps;
        }
    }
    if(accumulator){
        for(size_t k=0;k<AMA_PARAMS;k++)for(size_t i=0;i<pn[k];i++)
            accumulator->values[k][i]+=grads[k][i];
        accumulator->tokens+=(uint64_t)supervised_tokens;
    }else{
        AMA10Gradients update={0};update.values=grads;update.parameter_count=AMA_PARAMS;
        update.width=m->width;update.input_width=m->input_width;update.vocabulary=m->vocabulary;
        update.tokens=(uint64_t)supervised_tokens;
        if(!ama10_optimizer_apply(m,opt,&update,lr,wd))goto fail;
    }
    if(mean_loss)*mean_loss=loss;
    free(cache);free(logits);free(input_vectors);free(input_projection);free(used_depth);free(dh);free(dm);free(dh0);free(dm0);free(dpre);free(dgate);free(head_hidden);free(head_errors);free(head_gradient);free(head_hidden_gradient);
    for(size_t k=0;k<AMA_PARAMS;k++)free(grads[k]);
    return 1;
fail:
    free(cache);free(logits);free(input_vectors);free(input_projection);free(used_depth);free(dh);free(dm);free(dh0);free(dm0);free(dpre);free(dgate);free(head_hidden);free(head_errors);free(head_gradient);free(head_hidden_gradient);
    for(size_t k=0;k<AMA_PARAMS;k++)free(grads[k]);
    return 0;
#undef C
}

int ama10_train_sequence(AMA10Model *m,AMA10Optimizer *opt,AMA10State *state,const uint32_t *input,
                         const uint32_t *target,size_t len,unsigned depth,
                         float lr,float wd,float *mean_loss){
    return train_sequence_internal(m,opt,state,input,NULL,NULL,NULL,target,NULL,len,depth,-1.0f,lr,wd,NULL,NULL,mean_loss,NULL,NULL);
}

int ama10_train_sequence_adaptive(AMA10Model *m,AMA10Optimizer *opt,AMA10State *state,
                                  const uint32_t *input,const uint32_t *target,size_t len,
                                  unsigned max_depth,float halt_threshold,float lr,float wd,
                                  float *mean_loss,float *mean_depth){
    if(!isfinite(halt_threshold)||halt_threshold<0.0f)return 0;
    return train_sequence_internal(m,opt,state,input,NULL,NULL,NULL,target,NULL,len,max_depth,halt_threshold,
                                   lr,wd,NULL,NULL,mean_loss,mean_depth,NULL);
}

int ama10_train_sequence_accumulate(AMA10Model *m,AMA10Optimizer *opt,AMA10State *state,
                                    const uint32_t *input,const uint32_t *target,size_t len,
                                    unsigned depth,AMA10Gradients *gradients,float *mean_loss){
    return train_sequence_internal(m,opt,state,input,NULL,NULL,NULL,target,NULL,len,depth,-1.0f,0.0f,0.0f,
                                   gradients,NULL,mean_loss,NULL,NULL);
}

int ama10_train_feature_sequence(AMA10Model *m,AMA10Optimizer *opt,AMA10State *state,
                                 const float *input_features,const uint32_t *target,
                                 size_t len,unsigned depth,float lr,float wd,
                                 float *feature_gradients,float *mean_loss){
    if(!feature_gradients)return 0;
    return train_sequence_internal(m,opt,state,NULL,input_features,NULL,NULL,target,NULL,len,depth,-1.0f,
                                   lr,wd,NULL,feature_gradients,mean_loss,NULL,NULL);
}

int ama10_train_multimodal_sequence(AMA10Model *m,AMA10Optimizer *opt,AMA10State *state,
                                    const uint32_t *token_inputs,const float *feature_inputs,
                                    const unsigned char *feature_mask,const unsigned char *loss_mask,
                                    const uint32_t *targets,size_t length,unsigned depth,float lr,
                                    float wd,float *feature_gradients,float *mean_loss){
    return train_sequence_internal(m,opt,state,token_inputs,feature_inputs,feature_mask,loss_mask,
                                   targets,NULL,length,depth,-1.0f,lr,wd,NULL,feature_gradients,
                                   mean_loss,NULL,NULL);
}
int ama10_train_soft_multimodal_sequence(AMA10Model *m,AMA10Optimizer *opt,AMA10State *state,
                                    const uint32_t *token_inputs,const float *feature_inputs,
                                    const unsigned char *feature_mask,const unsigned char *loss_mask,
                                    const float *soft_targets,size_t length,unsigned depth,
                                    float lr,float wd,float *feature_gradients,float *mean_loss){
    if(!soft_targets)return 0;
    return train_sequence_internal(m,opt,state,token_inputs,feature_inputs,feature_mask,loss_mask,
        NULL,soft_targets,length,depth,-1.0f,lr,wd,NULL,feature_gradients,mean_loss,NULL,NULL);
}

int ama10_train_sequence_accumulate_opencl(AMA10Model *m,AMA10Optimizer *opt,AMA10State *state,
                                    const uint32_t *input,const uint32_t *target,size_t len,
                                    unsigned depth,AMA10Gradients *gradients,
                                    struct AMA10OpenCL *backend,float *mean_loss){
#ifdef AMA10_ENABLE_OPENCL
    if(!backend)return 0;
    return train_sequence_internal(m,opt,state,input,NULL,NULL,NULL,target,NULL,len,depth,-1.0f,
                                   0.0f,0.0f,gradients,NULL,mean_loss,NULL,backend);
#else
    (void)m;(void)opt;(void)state;(void)input;(void)target;(void)len;(void)depth;
    (void)gradients;(void)backend;(void)mean_loss;return 0;
#endif
}

static int eval_loss_with_halting(const AMA10Model *m,const uint32_t *input,
                    const uint32_t *target,size_t len,unsigned depth,
                    float halt_threshold,double *mean_loss,double *mean_depth){
    if(!m||!input||!target||!len||!depth||!mean_loss||
       !isfinite(halt_threshold)||halt_threshold<0.0f||halt_threshold>2.0f)return 0;
    AMA10State state;
    if(!ama10_state_init(&state,m->width))return 0;
    float *logits=(float*)malloc(m->vocabulary*sizeof(float));
    if(!logits){ama10_state_free(&state);return 0;}
    double total=0,steps=0;
    for(size_t i=0;i<len;i++){
        unsigned before=state.steps;
        if(target[i]>=m->vocabulary||!ama10_step(m,&state,input[i],logits,depth,halt_threshold)){
            free(logits);ama10_state_free(&state);return 0;
        }
        steps+=(double)(state.steps-before);
        float mx=logits[0];
        for(size_t k=1;k<m->vocabulary;k++)if(logits[k]>mx)mx=logits[k];
        double sum=0;
        for(size_t k=0;k<m->vocabulary;k++)sum+=exp((double)logits[k]-(double)mx);
        if(!(sum>0)||!isfinite(sum)){free(logits);ama10_state_free(&state);return 0;}
        total+=(double)mx+log(sum)-(double)logits[target[i]];
    }
    free(logits);ama10_state_free(&state);*mean_loss=total/(double)len;
    if(mean_depth)*mean_depth=steps/(double)len;
    return 1;
}

int ama10_eval_loss(const AMA10Model *m,const uint32_t *input,const uint32_t *target,
                    size_t len,unsigned depth,double *mean_loss){
    return eval_loss_with_halting(m,input,target,len,depth,2.0f,mean_loss,NULL);
}

int ama10_eval_loss_adaptive(const AMA10Model *m,const uint32_t *input,
                             const uint32_t *target,size_t len,unsigned max_depth,
                             float halt_threshold,double *mean_loss,double *mean_depth){
    if(!mean_depth)return 0;
    return eval_loss_with_halting(m,input,target,len,max_depth,halt_threshold,
                                  mean_loss,mean_depth);
}

int ama10_eval_feature_loss(const AMA10Model *m,const float *input_features,
                            const uint32_t *target,size_t len,unsigned depth,
                            double *mean_loss){
    if(!m||!m->input_width||!input_features||!target||!len||!depth||!mean_loss||
       len>SIZE_MAX/m->input_width)return 0;
    AMA10State state={0};
    if(!ama10_state_init(&state,m->width))return 0;
    float *logits=(float*)malloc(m->vocabulary*sizeof(float));
    if(!logits){ama10_state_free(&state);return 0;}
    double total=0;
    for(size_t i=0;i<len;i++){
        if(target[i]>=m->vocabulary||
           !ama10_step_features(m,&state,input_features+i*m->input_width,logits,depth,2.0f)){
            free(logits);ama10_state_free(&state);return 0;
        }
        float mx=logits[0];
        for(size_t k=1;k<m->vocabulary;k++)if(logits[k]>mx)mx=logits[k];
        double sum=0;
        for(size_t k=0;k<m->vocabulary;k++)sum+=exp((double)logits[k]-(double)mx);
        if(!(sum>0)||!isfinite(sum)){free(logits);ama10_state_free(&state);return 0;}
        total+=(double)mx+log(sum)-(double)logits[target[i]];
    }
    free(logits);ama10_state_free(&state);*mean_loss=total/(double)len;return 1;
}
