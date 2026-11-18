#include "checkpoint.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AMA10_PARAMETER_GROUPS 8
static int write_u32(FILE *f,uint32_t x){
    unsigned char b[4]={(unsigned char)x,(unsigned char)(x>>8),(unsigned char)(x>>16),(unsigned char)(x>>24)};
    return fwrite(b,1,4,f)==4;
}
static int read_u32(FILE *f,uint32_t *x){
    unsigned char b[4];if(fread(b,1,4,f)!=4)return 0;
    *x=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);return 1;
}
static int write_u64(FILE *f,uint64_t x){
    return write_u32(f,(uint32_t)x)&&write_u32(f,(uint32_t)(x>>32));
}
static int read_u64(FILE *f,uint64_t *x){
    uint32_t lo,hi;if(!read_u32(f,&lo)||!read_u32(f,&hi))return 0;
    *x=(uint64_t)lo|((uint64_t)hi<<32);return 1;
}
static int write_f32(FILE *f,float x){uint32_t bits;memcpy(&bits,&x,4);return write_u32(f,bits);}
static int read_f32(FILE *f,float *x){uint32_t bits;if(!read_u32(f,&bits))return 0;memcpy(x,&bits,4);return isfinite(*x);}
static int write_floats(FILE *f,const float *p,size_t n){
    for(size_t i=0;i<n;i++) if(!write_f32(f,p[i])) return 0;
    return 1;
}
static int read_floats(FILE *f,float *p,size_t n){
    for(size_t i=0;i<n;i++) if(!read_f32(f,&p[i])) return 0;
    return 1;
}
static void parameter_views(const AMA10Model *m,float **p,size_t *n){
    size_t w=m->width,e=m->input_width,v=m->vocabulary;
    p[0]=m->wx;n[0]=w*e;p[1]=m->wh;n[1]=w*w;p[2]=m->wm;n[2]=w*w;
    p[3]=m->wu;n[3]=w*w;p[4]=m->bias;n[4]=w;p[5]=m->gate_bias;n[5]=w;
    p[6]=m->embed;n[6]=e*v;p[7]=m->head;n[7]=v*w;
}
static int write_payload(FILE *f,const AMA10Model *m,const AMA10Tokenizer *t,
                         const AMA10Optimizer *o,const AMA10State *s,
                         const AMA10CheckpointProgress *p){
    static const unsigned char magic[8]={'A','M','A','1','0','C','K','8'};
    int ok=fwrite(magic,1,8,f)==8&&write_u32(f,8)&&
        write_u32(f,(uint32_t)m->vocabulary)&&write_u32(f,(uint32_t)m->width)&&
        write_u32(f,(uint32_t)m->input_width)&&write_u32(f,(uint32_t)t->vocabulary_size)&&
        write_u32(f,p->sequence_length)&&write_u32(f,p->depth)&&
        write_u32(f,p->accumulation_steps)&&write_u32(f,s->steps)&&
        write_u64(f,p->train_hash)&&write_u64(f,p->validation_hash)&&
        write_u64(f,p->tokenizer_sample_bytes)&&
        write_u64(f,p->completed_updates)&&write_u64(f,p->cursor)&&write_u64(f,o->step)&&
        write_u32(f,o->algorithm)&&
        write_u32(f,p->target_vocabulary)&&write_f32(f,p->learning_rate)&&
        write_f32(f,p->learning_rate_decay)&&write_f32(f,p->weight_decay)&&
        write_u32(f,p->sampler_algorithm)&&write_u64(f,p->sampler_state);
    float *params[AMA10_PARAMETER_GROUPS];size_t sizes[AMA10_PARAMETER_GROUPS];parameter_views(m,params,sizes);
    for(size_t i=0;i<AMA10_PARAMETER_GROUPS&&ok;i++)ok=write_floats(f,params[i],sizes[i]);
    for(size_t i=256;i<t->vocabulary_size&&ok;i++){
        if(t->tokens[i].length>UINT32_MAX){ok=0;break;}
        ok=write_u32(f,t->tokens[i].special)&&write_u32(f,t->tokens[i].left)&&write_u32(f,t->tokens[i].right)&&
           write_u32(f,(uint32_t)t->tokens[i].length)&&
           fwrite(t->tokens[i].bytes,1,t->tokens[i].length,f)==t->tokens[i].length;
    }
    if(ok)ok=write_floats(f,s->fast,m->width)&&write_floats(f,s->slow,m->width);
    for(size_t k=0;k<AMA10_PARAMETER_GROUPS&&ok;k++)ok=write_floats(f,o->first_moment[k],sizes[k]);
    for(size_t k=0;k<AMA10_PARAMETER_GROUPS&&ok;k++)ok=write_floats(f,o->second_moment[k],sizes[k]);
    return ok;
}
int ama10_checkpoint_save(const char *path,const AMA10Model *m,const AMA10Tokenizer *t,
                          const AMA10Optimizer *o,const AMA10State *s,
                          const AMA10CheckpointProgress *p){
    if(!path||!m||!t||!o||!s||!p||m->vocabulary!=t->vocabulary_size||
       m->width!=s->width||o->parameter_count!=AMA10_PARAMETER_GROUPS||
       (o->algorithm!=AMA10_OPT_ADAMW&&o->algorithm!=AMA10_OPT_MUON)||
       p->sequence_length==0||p->depth==0||p->accumulation_steps==0||p->tokenizer_sample_bytes==0||
       p->sampler_algorithm>1||(p->sampler_algorithm==1&&!p->sampler_state)||
       p->target_vocabulary<t->vocabulary_size||
       !isfinite(p->learning_rate)||p->learning_rate<=0||
       !isfinite(p->learning_rate_decay)||p->learning_rate_decay<0||
       !isfinite(p->weight_decay)||p->weight_decay<0)return 0;
    size_t path_len=strlen(path);if(path_len>SIZE_MAX-5)return 0;
    char *tmp=(char*)malloc(path_len+5),*backup=(char*)malloc(path_len+5);
    if(!tmp||!backup){free(tmp);free(backup);return 0;}
    memcpy(tmp,path,path_len);memcpy(tmp+path_len,".tmp",5);
    memcpy(backup,path,path_len);memcpy(backup+path_len,".bak",5);
    FILE *f=fopen(tmp,"wb");if(!f){free(tmp);free(backup);return 0;}
    int ok=write_payload(f,m,t,o,s,p);
    if(fflush(f)!=0)ok=0;
    if(fclose(f)!=0)ok=0;
    int had_backup=0;
    if(ok){
        (void)remove(backup);
        if(rename(path,backup)==0)had_backup=1;
        else if(errno!=ENOENT)ok=0;
    }
    if(ok&&rename(tmp,path)!=0)ok=0;
    if(!ok){
        (void)remove(tmp);
        if(had_backup)(void)rename(backup,path);
    }else if(had_backup)(void)remove(backup);
    free(tmp);free(backup);return ok;
}
static int read_payload(FILE *f,AMA10Model *m,AMA10Tokenizer *t,AMA10Optimizer *o,
                        AMA10State *s,AMA10CheckpointProgress *p){
    unsigned char magic[8];
    uint32_t version=0,v=0,w=0,e=0,tv=0,seq=0,depth=0,accumulation=0,state_steps=0,target=0,algorithm=0;
    uint64_t completed=0,cursor=0,optimizer_step=0;
    AMA10CheckpointProgress progress={0};
    int ok=fread(magic,1,8,f)==8&&read_u32(f,&version)&&
        ((version==6&&memcmp(magic,"AMA10CK6",8)==0)||(version==7&&memcmp(magic,"AMA10CK7",8)==0)||
         (version==8&&memcmp(magic,"AMA10CK8",8)==0))&&
        read_u32(f,&v)&&read_u32(f,&w)&&read_u32(f,&e)&&read_u32(f,&tv)&&
        read_u32(f,&seq)&&read_u32(f,&depth)&&read_u32(f,&accumulation)&&read_u32(f,&state_steps)&&
        read_u64(f,&progress.train_hash)&&read_u64(f,&progress.validation_hash)&&
        read_u64(f,&progress.tokenizer_sample_bytes)&&
        read_u64(f,&completed)&&read_u64(f,&cursor)&&read_u64(f,&optimizer_step)&&
        read_u32(f,&algorithm)&&
        read_u32(f,&target)&&read_f32(f,&progress.learning_rate)&&
        read_f32(f,&progress.learning_rate_decay)&&read_f32(f,&progress.weight_decay);
    if(ok&&version>=8)ok=read_u32(f,&progress.sampler_algorithm)&&read_u64(f,&progress.sampler_state);
    if(!ok||!v||!w||!e||tv!=v||seq==0||depth==0||accumulation==0||
       progress.sampler_algorithm>1||(progress.sampler_algorithm==1&&!progress.sampler_state)||
       progress.tokenizer_sample_bytes==0||target<tv||completed!=optimizer_step||
       !isfinite(progress.learning_rate)||
       progress.learning_rate<=0||!isfinite(progress.learning_rate_decay)||progress.learning_rate_decay<0||
       !isfinite(progress.weight_decay)||progress.weight_decay<0||
       (algorithm!=AMA10_OPT_ADAMW&&algorithm!=AMA10_OPT_MUON))return 0;
    if(!ama10_model_init(m,v,w,e,UINT64_C(1))||
       !ama10_tokenizer_train(t,NULL,0,tv)||!ama10_optimizer_init(o,m)||!ama10_state_init(s,w))return 0;
    m->vocabulary=v;m->width=w;m->input_width=e;
    float *params[AMA10_PARAMETER_GROUPS];size_t sizes[AMA10_PARAMETER_GROUPS];parameter_views(m,params,sizes);
    for(size_t k=0;k<AMA10_PARAMETER_GROUPS&&ok;k++)ok=read_floats(f,params[k],sizes[k]);
    for(uint32_t i=256;i<tv&&ok;i++){
        uint32_t special=0,left=0,right=0,length=0;
        if((version>=7&&!read_u32(f,&special))||!read_u32(f,&left)||!read_u32(f,&right)||!read_u32(f,&length)||
           special>1||length==0||(special?(left!=UINT32_MAX||right!=UINT32_MAX):(left>=i||right>=i))){ok=0;break;}
        size_t ll=0,rl=0;if(!special){ll=t->tokens[left].length;rl=t->tokens[right].length;
            if(ll>SIZE_MAX-rl||ll+rl!=length){ok=0;break;}}
        unsigned char *bytes=(unsigned char*)malloc(length);
        if(!bytes){ok=0;break;}
        if(fread(bytes,1,length,f)!=length||(!special&&(memcmp(bytes,t->tokens[left].bytes,ll)!=0||
           memcmp(bytes+ll,t->tokens[right].bytes,rl)!=0))){free(bytes);ok=0;break;}
        t->tokens[i].left=left;t->tokens[i].right=right;t->tokens[i].length=length;t->tokens[i].bytes=bytes;t->tokens[i].special=special;
        t->vocabulary_size++;
    }
    if(ok)ok=read_floats(f,s->fast,w)&&read_floats(f,s->slow,w);
    for(size_t k=0;k<AMA10_PARAMETER_GROUPS&&ok;k++)ok=read_floats(f,o->first_moment[k],sizes[k]);
    for(size_t k=0;k<AMA10_PARAMETER_GROUPS&&ok;k++)ok=read_floats(f,o->second_moment[k],sizes[k]);
    if(ok&&fgetc(f)!=EOF)ok=0;
    if(!ok)return 0;
    progress.completed_updates=completed;progress.cursor=cursor;progress.sequence_length=seq;
    progress.depth=depth;progress.accumulation_steps=accumulation;
    progress.target_vocabulary=target;progress.optimizer_algorithm=algorithm;
    s->steps=state_steps;o->step=optimizer_step;o->algorithm=algorithm;
    if(p) *p=progress;
    return 1;
}
static int load_try(const char *path,AMA10Model *m,AMA10Tokenizer *t,AMA10Optimizer *o,
                    AMA10State *s,AMA10CheckpointProgress *p){
    FILE *f=fopen(path,"rb");if(!f)return 0;
    AMA10Model model={0};AMA10Tokenizer tokenizer={0};AMA10Optimizer optimizer={0};AMA10State state={0};
    AMA10CheckpointProgress progress={0};int ok=read_payload(f,&model,&tokenizer,&optimizer,&state,&progress);
    if(fclose(f)!=0)ok=0;
    if(ok){
        ama10_model_free(m);ama10_tokenizer_free(t);ama10_optimizer_free(o);ama10_state_free(s);
        *m=model;*t=tokenizer;*o=optimizer;*s=state;if(p)*p=progress;
    }else{
        ama10_model_free(&model);ama10_tokenizer_free(&tokenizer);ama10_optimizer_free(&optimizer);ama10_state_free(&state);
    }
    return ok;
}
int ama10_checkpoint_load(const char *path,AMA10Model *m,AMA10Tokenizer *t,AMA10Optimizer *o,
                          AMA10State *s,AMA10CheckpointProgress *p){
    if(!path||!m||!t||!o||!s)return 0;
    if(load_try(path,m,t,o,s,p))return 1;
    size_t n=strlen(path);if(n>SIZE_MAX-5)return 0;
    char *backup=(char*)malloc(n+5);if(!backup)return 0;
    memcpy(backup,path,n);memcpy(backup+n,".bak",5);
    int ok=load_try(backup,m,t,o,s,p);free(backup);return ok;
}
uint64_t ama10_data_hash(const unsigned char *data,size_t n){
    uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<n;i++){h^=data[i];h*=UINT64_C(1099511628211);}
    return h;
}
