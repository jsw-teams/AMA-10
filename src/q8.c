#include "ama10_q8.h"
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int alloc_i8(int8_t**p,size_t n){if(!n||n>SIZE_MAX/sizeof(int8_t))return 0;*p=(int8_t*)malloc(n);return *p!=NULL;}
static int alloc_f(float**p,size_t n){if(!n||n>SIZE_MAX/sizeof(float))return 0;*p=(float*)malloc(n*sizeof(float));return *p!=NULL;}
void ama10_q8_model_free(AMA10Q8Model*m){if(!m)return;free(m->wx);free(m->wh);free(m->wm);free(m->wu);free(m->embed);free(m->head);free(m->wx_scale);free(m->wh_scale);free(m->wm_scale);free(m->wu_scale);free(m->embed_scale);free(m->head_scale);free(m->bias);free(m->gate_bias);memset(m,0,sizeof(*m));}
static int quant_rows(const float*x,int8_t*q,float*s,size_t rows,size_t cols){
    for(size_t r=0;r<rows;r++){float max=0;for(size_t c=0;c<cols;c++){float v=x[r*cols+c];if(!isfinite(v))return 0;if(fabsf(v)>max)max=fabsf(v);}float scale=max>0?max/127.0f:1.0f;s[r]=scale;
        for(size_t c=0;c<cols;c++){long z=lroundf(x[r*cols+c]/scale);if(z>127)z=127;if(z< -127)z=-127;q[r*cols+c]=(int8_t)z;}}
    return 1;
}
static int q8_alloc(AMA10Q8Model*m,size_t v,size_t w,size_t e){
    if(!v||!w||!e||v>UINT32_MAX||w>UINT32_MAX||e>UINT32_MAX||w>SIZE_MAX/e||w>SIZE_MAX/w||e>SIZE_MAX/v||v>SIZE_MAX/w)return 0;
    memset(m,0,sizeof(*m));m->width=w;m->input_width=e;m->vocabulary=v;
    return alloc_i8(&m->wx,w*e)&&alloc_i8(&m->wh,w*w)&&alloc_i8(&m->wm,w*w)&&alloc_i8(&m->wu,w*w)&&alloc_i8(&m->embed,e*v)&&alloc_i8(&m->head,v*w)&&
      alloc_f(&m->wx_scale,w)&&alloc_f(&m->wh_scale,w)&&alloc_f(&m->wm_scale,w)&&alloc_f(&m->wu_scale,w)&&alloc_f(&m->embed_scale,e)&&alloc_f(&m->head_scale,v)&&alloc_f(&m->bias,w)&&alloc_f(&m->gate_bias,w);
}
int ama10_q8_model_from_f32(const AMA10Model*s,AMA10Q8Model*q){
    if(!s||!q||q->wx||!s->wx||!s->wh||!s->wm||!s->wu||!s->embed||!s->head||!s->bias||!s->gate_bias||
       !q8_alloc(q,s->vocabulary,s->width,s->input_width)){ama10_q8_model_free(q);return 0;}
    size_t w=s->width,e=s->input_width,v=s->vocabulary;
    if(!quant_rows(s->wx,q->wx,q->wx_scale,w,e)||!quant_rows(s->wh,q->wh,q->wh_scale,w,w)||!quant_rows(s->wm,q->wm,q->wm_scale,w,w)||!quant_rows(s->wu,q->wu,q->wu_scale,w,w)||
       !quant_rows(s->embed,q->embed,q->embed_scale,e,v)||!quant_rows(s->head,q->head,q->head_scale,v,w)){ama10_q8_model_free(q);return 0;}
    for(size_t i=0;i<w;i++){if(!isfinite(s->bias[i])||!isfinite(s->gate_bias[i])){ama10_q8_model_free(q);return 0;}q->bias[i]=s->bias[i];q->gate_bias[i]=s->gate_bias[i];}
    return 1;
}
static uint64_t hash_bytes(uint64_t h,const void*p,size_t n){const unsigned char*b=(const unsigned char*)p;for(size_t i=0;i<n;i++){h^=b[i];h*=UINT64_C(1099511628211);}return h;}
static uint64_t hash_u32(uint64_t h,uint32_t v){unsigned char b[4]={(unsigned char)v,(unsigned char)(v>>8),(unsigned char)(v>>16),(unsigned char)(v>>24)};return hash_bytes(h,b,4);}
static uint64_t hash_float(uint64_t h,const float*p,size_t n){for(size_t i=0;i<n;i++){uint32_t b;memcpy(&b,&p[i],4);h=hash_u32(h,b);}return h;}
static uint64_t q8_hash(const AMA10Q8Model*m){uint64_t h=UINT64_C(14695981039346656037);h=hash_u32(h,(uint32_t)m->vocabulary);h=hash_u32(h,(uint32_t)m->width);h=hash_u32(h,(uint32_t)m->input_width);
 size_t w=m->width,e=m->input_width,v=m->vocabulary;const int8_t*q[]={m->wx,m->wh,m->wm,m->wu,m->embed,m->head};size_t n[]={w*e,w*w,w*w,w*w,e*v,v*w};for(size_t i=0;i<6;i++)h=hash_bytes(h,q[i],n[i]);
 const float*s[]={m->wx_scale,m->wh_scale,m->wm_scale,m->wu_scale,m->embed_scale,m->head_scale,m->bias,m->gate_bias};size_t sn[]={w,w,w,w,e,v,w,w};for(size_t i=0;i<8;i++)h=hash_float(h,s[i],sn[i]);return h;}
static int put_u32(FILE*f,uint32_t x){unsigned char b[4]={(unsigned char)x,(unsigned char)(x>>8),(unsigned char)(x>>16),(unsigned char)(x>>24)};return fwrite(b,1,4,f)==4;}
static int get_u32(FILE*f,uint32_t*x){unsigned char b[4];if(fread(b,1,4,f)!=4)return 0;*x=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);return 1;}
static int put_floats(FILE*f,const float*x,size_t n){for(size_t i=0;i<n;i++){uint32_t b;memcpy(&b,&x[i],4);if(!put_u32(f,b))return 0;}return 1;}
static int get_floats(FILE*f,float*x,size_t n,int positive){for(size_t i=0;i<n;i++){uint32_t b;if(!get_u32(f,&b))return 0;memcpy(&x[i],&b,4);if(!isfinite(x[i])||(positive&&x[i]<=0))return 0;}return 1;}
int ama10_q8_model_save(const AMA10Q8Model*m,const char*path){
    if(!m||!m->wx||!m->wh||!m->wm||!m->wu||!m->embed||!m->head||!m->wx_scale||!m->wh_scale||!m->wm_scale||!m->wu_scale||!m->embed_scale||!m->head_scale||!m->bias||!m->gate_bias||!path||sizeof(float)!=4||FLT_RADIX!=2||FLT_MANT_DIG!=24||FLT_MAX_EXP!=128)return 0;
    size_t w=m->width,e=m->input_width,v=m->vocabulary;
    size_t pn[]={w*e,w*w,w*w,w*w,e*v,v*w};const int8_t*q[]={m->wx,m->wh,m->wm,m->wu,m->embed,m->head};const float*s[]={m->wx_scale,m->wh_scale,m->wm_scale,m->wu_scale,m->embed_scale,m->head_scale};size_t sn[]={w,w,w,w,e,v};
    size_t len=strlen(path);if(len>SIZE_MAX-5)return 0;char*tmp=(char*)malloc(len+5),*bak=(char*)malloc(len+5);if(!tmp||!bak){free(tmp);free(bak);return 0;}memcpy(tmp,path,len);memcpy(tmp+len,".tmp",5);memcpy(bak,path,len);memcpy(bak+len,".bak",5);FILE*f=fopen(tmp,"wb");if(!f){free(tmp);free(bak);return 0;}
    const unsigned char magic[8]={'A','M','A','1','0','Q','8',1};int ok=fwrite(magic,1,8,f)==8&&put_u32(f,1)&&put_u32(f,(uint32_t)v)&&put_u32(f,(uint32_t)w)&&put_u32(f,(uint32_t)e);
    for(size_t i=0;i<6&&ok;i++)ok=fwrite(q[i],1,pn[i],f)==pn[i];
    for(size_t i=0;i<6&&ok;i++)ok=put_floats(f,s[i],sn[i]);
    if(ok)ok=put_floats(f,m->bias,w)&&put_floats(f,m->gate_bias,w);
    uint32_t low=0,high=0;uint64_t hash=q8_hash(m);low=(uint32_t)hash;high=(uint32_t)(hash>>32);if(ok)ok=put_u32(f,low)&&put_u32(f,high);if(fflush(f)!=0)ok=0;if(fclose(f)!=0)ok=0;
    int had_backup=0;
    if(ok){(void)remove(bak);if(rename(path,bak)==0)had_backup=1;else if(errno!=ENOENT)ok=0;}
    if(ok&&rename(tmp,path)!=0)ok=0;
    if(!ok){(void)remove(tmp);if(had_backup)(void)rename(bak,path);}else if(had_backup)(void)remove(bak);
    free(tmp);free(bak);return ok;
}
int ama10_q8_model_load(AMA10Q8Model*m,const char*path){
    if(!m||m->wx||!path)return 0;
    FILE*f=fopen(path,"rb");if(!f)return 0;if(fseek(f,0,SEEK_END)!=0){fclose(f);return 0;}long size=ftell(f);if(size<32||fseek(f,0,SEEK_SET)!=0){fclose(f);return 0;}
    unsigned char magic[8],expected[8]={'A','M','A','1','0','Q','8',1};uint32_t version=0,v=0,w=0,e=0;int ok=fread(magic,1,8,f)==8&&!memcmp(magic,expected,8)&&get_u32(f,&version)&&version==1&&get_u32(f,&v)&&get_u32(f,&w)&&get_u32(f,&e);
    AMA10Q8Model q={0};if(ok)ok=q8_alloc(&q,v,w,e);size_t pn[6]={0},sn[6]={0};int8_t*weights[6]={0};float*scales[6]={0};if(ok){pn[0]=w*e;pn[1]=pn[2]=pn[3]=w*w;pn[4]=e*v;pn[5]=v*w;sn[0]=sn[1]=sn[2]=sn[3]=w;sn[4]=e;sn[5]=v;weights[0]=q.wx;weights[1]=q.wh;weights[2]=q.wm;weights[3]=q.wu;weights[4]=q.embed;weights[5]=q.head;scales[0]=q.wx_scale;scales[1]=q.wh_scale;scales[2]=q.wm_scale;scales[3]=q.wu_scale;scales[4]=q.embed_scale;scales[5]=q.head_scale;}
    uint64_t values=0;if(ok){values=0;for(size_t i=0;i<6;i++)values+=(uint64_t)pn[i]+(uint64_t)sn[i]*4;values+=(uint64_t)w*8+8;if(values>(uint64_t)LONG_MAX-24||(uint64_t)size!=24+values)ok=0;}
    for(size_t i=0;i<6&&ok;i++)ok=fread(weights[i],1,pn[i],f)==pn[i];
    for(size_t i=0;i<6&&ok;i++)ok=get_floats(f,scales[i],sn[i],1);
    if(ok)ok=get_floats(f,q.bias,w,0)&&get_floats(f,q.gate_bias,w,0);
    uint32_t lo=0,hi=0;if(ok)ok=get_u32(f,&lo)&&get_u32(f,&hi);if(fclose(f)!=0)ok=0;if(ok){uint64_t expected_hash=(uint64_t)lo|((uint64_t)hi<<32);ok=q8_hash(&q)==expected_hash;}
    if(ok){*m=q;}else ama10_q8_model_free(&q);return ok;
}
size_t ama10_q8_model_memory_bytes(const AMA10Q8Model*m){
    if(!m||!m->wx)return 0;
    size_t w=m->width,e=m->input_width,v=m->vocabulary;
    if(!w||!e||!v||w>SIZE_MAX/e||w>SIZE_MAX/w||e>SIZE_MAX/v||v>SIZE_MAX/w||w*w>SIZE_MAX/3)return 0;
    size_t sum=w*e,parts[]={3*w*w,e*v,v*w};
    for(size_t i=0;i<3;i++){if(sum>SIZE_MAX-parts[i])return 0;sum+=parts[i];}
    if(6*w>SIZE_MAX-e||6*w+e>SIZE_MAX-v)return 0;
    size_t floats=6*w+e+v;
    if(floats>SIZE_MAX/sizeof(float)||sum>SIZE_MAX-floats*sizeof(float))return 0;
    return sum+floats*sizeof(float);
}
int ama10_q8_state_init(AMA10State*s,size_t w){return ama10_state_init(s,w);}
static int step( const AMA10Q8Model*m,AMA10State*s,const float*x,float*logits,unsigned max_depth,float threshold){
    if(!m||!s||!x||!logits||!s->fast||!s->slow||s->width!=m->width||!max_depth||!isfinite(threshold)||threshold<0||threshold>2)return 0;
    size_t w=m->width,e=m->input_width;
    for(size_t j=0;j<e;j++)if(!isfinite(x[j]))return 0;
    float*next=(float*)malloc(w*sizeof(float)),*proposal=(float*)malloc(w*sizeof(float));if(!next||!proposal){free(next);free(proposal);return 0;}
    for(unsigned d=0;d<max_depth;d++){float activity=0;for(size_t i=0;i<w;i++){float drive=m->bias[i],input=0,wxsum=0;
        for(size_t j=0;j<e;j++)wxsum+=(float)m->wx[i*e+j]*x[j];
        drive+=m->wx_scale[i]*wxsum;input=m->wx_scale[i]*wxsum;
        float whsum=0,wmsum=0,wusum=0;for(size_t j=0;j<w;j++){whsum+=(float)m->wh[i*w+j]*s->fast[j];wmsum+=(float)m->wm[i*w+j]*s->slow[j];wusum+=(float)m->wu[i*w+j]*s->fast[j];}
        drive+=m->wh_scale[i]*whsum+m->wm_scale[i]*wmsum;float gate=m->gate_bias[i]+input+m->wu_scale[i]*wusum;float g=gate>=0?1.0f/(1.0f+expf(-gate)):expf(gate)/(1.0f+expf(gate));proposal[i]=tanhf(drive);next[i]=(1-g)*s->fast[i]+g*proposal[i];activity+=fabsf(next[i]-s->fast[i]);}
        memcpy(s->fast,next,w*sizeof(float));for(size_t i=0;i<w;i++)s->slow[i]=0.995f*s->slow[i]+0.005f*s->fast[i];s->steps++;float confidence=1.0f/(1.0f+activity/(float)w);if(d+1>=2&&threshold<=1.0f&&confidence>=threshold)break;}
    for(size_t k=0;k<m->vocabulary;k++){float sum=0;for(size_t j=0;j<w;j++)sum+=(float)m->head[k*w+j]*s->fast[j];logits[k]=m->head_scale[k]*sum;}
    free(next);free(proposal);return 1;
}
int ama10_q8_step_features(const AMA10Q8Model*m,AMA10State*s,const float*x,float*z,unsigned depth,float threshold){return step(m,s,x,z,depth,threshold);}
int ama10_q8_step(const AMA10Q8Model*m,AMA10State*s,uint32_t token,float*z,unsigned depth,float threshold){if(!m||token>=m->vocabulary)return 0;float*x=(float*)malloc(m->input_width*sizeof(float));if(!x)return 0;for(size_t j=0;j<m->input_width;j++)x[j]=m->embed_scale[j]*(float)m->embed[j*m->vocabulary+token];int ok=step(m,s,x,z,depth,threshold);free(x);return ok;}
