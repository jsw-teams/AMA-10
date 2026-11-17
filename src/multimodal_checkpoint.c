#include "multimodal_checkpoint.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint32_t mm_get_u32(const unsigned char b[4]){
    return (uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);
}
static int mm_write_u32(FILE *f,uint32_t x){
    unsigned char b[4]={(unsigned char)x,(unsigned char)(x>>8),(unsigned char)(x>>16),(unsigned char)(x>>24)};
    return fwrite(b,1,4,f)==4;
}
static int mm_read_u32(FILE *f,uint32_t *x){
    unsigned char b[4];if(fread(b,1,4,f)!=4)return 0;*x=mm_get_u32(b);return 1;
}
static int mm_write_u64(FILE *f,uint64_t x){return mm_write_u32(f,(uint32_t)x)&&mm_write_u32(f,(uint32_t)(x>>32));}
static int mm_read_u64(FILE *f,uint64_t *x){
    uint32_t lo,hi;if(!mm_read_u32(f,&lo)||!mm_read_u32(f,&hi))return 0;
    *x=(uint64_t)lo|((uint64_t)hi<<32);return 1;
}
static char *mm_suffix(const char *path,const char *suffix){
    size_t a=strlen(path),b=strlen(suffix);if(a>SIZE_MAX-b-1)return NULL;
    char *p=(char*)malloc(a+b+1);if(!p)return NULL;
    memcpy(p,path,a);memcpy(p+a,suffix,b+1);return p;
}
static int mm_file_length(FILE *f,uint64_t *length){
    if(fseek(f,0,SEEK_END)!=0)return 0;
    long n=ftell(f);if(n<0||fseek(f,0,SEEK_SET)!=0)return 0;
    *length=(uint64_t)n;return 1;
}
static int mm_hash_file(const char *path,uint64_t *length,uint64_t *hash){
    FILE *f=fopen(path,"rb");if(!f)return 0;
    int ok=mm_file_length(f,length);uint64_t h=UINT64_C(14695981039346656037);
    unsigned char buf[65536];size_t n;
    while(ok&&(n=fread(buf,1,sizeof(buf),f))!=0)
        for(size_t i=0;i<n;i++){h^=buf[i];h*=UINT64_C(1099511628211);}
    if(ferror(f))ok=0;
    if(fclose(f)!=0)ok=0;
    if(ok)*hash=h;
    return ok;
}
static int mm_copy(FILE *src,FILE *dst,uint64_t count,uint64_t *hash){
    unsigned char buf[65536];uint64_t h=UINT64_C(14695981039346656037);
    while(count){
        size_t want=count>sizeof(buf)?sizeof(buf):(size_t)count;
        size_t n=fread(buf,1,want,src);if(n!=want)return 0;
        if(dst&&fwrite(buf,1,n,dst)!=n)return 0;
        for(size_t i=0;i<n;i++){h^=buf[i];h*=UINT64_C(1099511628211);}
        count-=(uint64_t)n;
    }
    if(hash)*hash=h;
    return 1;
}
static int mm_atomic_replace(const char *temp,const char *path,const char *backup){
    int had_backup=0,ok=1;
    (void)remove(backup);
    if(rename(path,backup)==0)had_backup=1;
    else if(errno!=ENOENT)ok=0;
    if(ok&&rename(temp,path)!=0)ok=0;
    if(!ok){
        (void)remove(temp);
        if(had_backup)(void)rename(backup,path);
    }else if(had_backup)(void)remove(backup);
    return ok;
}

int ama10_multimodal_checkpoint_save(const char *path,const AMA10Model *model,
                                     const AMA10Tokenizer *tokenizer,
                                     const AMA10Optimizer *optimizer,
                                     const AMA10State *state,
                                     const AMA10CheckpointProgress *progress,
                                     const AMA10Encoder *encoder){
    if(!path||!model||!tokenizer||!optimizer||!state||!progress||!encoder||
       encoder->output_width!=model->input_width||!encoder->input_width||!encoder->weights)return 0;
    char *core=mm_suffix(path,".stage-core"),*enc=mm_suffix(path,".stage-encoder");
    char *temp=mm_suffix(path,".tmp"),*backup=mm_suffix(path,".bak");
    if(!core||!enc||!temp||!backup){free(core);free(enc);free(temp);free(backup);return 0;}
    (void)remove(core);(void)remove(enc);(void)remove(temp);
    int ok=ama10_checkpoint_save(core,model,tokenizer,optimizer,state,progress)&&
           ama10_encoder_save(encoder,enc);
    uint64_t core_len=0,enc_len=0,core_hash=0,enc_hash=0;
    if(ok)ok=mm_hash_file(core,&core_len,&core_hash)&&mm_hash_file(enc,&enc_len,&enc_hash);
    FILE *out=ok?fopen(temp,"wb"):NULL;
    FILE *cf=NULL,*ef=NULL;
    if(ok&&!out)ok=0;
    if(ok){cf=fopen(core,"rb");ef=fopen(enc,"rb");}
    if(ok&&(!cf||!ef))ok=0;
    static const unsigned char magic[8]={'A','M','A','1','0','M','M','1'};
    if(ok)ok=fwrite(magic,1,8,out)==8&&mm_write_u32(out,1)&&
        mm_write_u64(out,core_len)&&mm_write_u64(out,enc_len)&&
        mm_write_u64(out,core_hash)&&mm_write_u64(out,enc_hash)&&
        mm_copy(cf,out,core_len,NULL)&&mm_copy(ef,out,enc_len,NULL);
    if(cf&&fclose(cf)!=0)ok=0;
    if(ef&&fclose(ef)!=0)ok=0;
    if(out){if(fflush(out)!=0)ok=0;if(fclose(out)!=0)ok=0;}
    if(ok)ok=mm_atomic_replace(temp,path,backup);
    if(!ok)(void)remove(temp);
    (void)remove(core);(void)remove(enc);
    free(core);free(enc);free(temp);free(backup);return ok;
}

int ama10_multimodal_checkpoint_load(const char *path,AMA10Model *model,
                                     AMA10Tokenizer *tokenizer,AMA10Optimizer *optimizer,
                                     AMA10State *state,AMA10CheckpointProgress *progress,
                                     AMA10Encoder *encoder){
    if(!path||!model||!tokenizer||!optimizer||!state||!progress||!encoder)return 0;
    FILE *f=fopen(path,"rb");if(!f)return 0;
    uint64_t file_len=0,core_len=0,enc_len=0,core_hash=0,enc_hash=0;
    int ok=mm_file_length(f,&file_len);unsigned char magic[8],expected[8]={'A','M','A','1','0','M','M','1'};
    uint32_t version=0;
    if(ok)ok=fread(magic,1,8,f)==8&&memcmp(magic,expected,8)==0&&mm_read_u32(f,&version)&&
        version==1&&mm_read_u64(f,&core_len)&&mm_read_u64(f,&enc_len)&&
        mm_read_u64(f,&core_hash)&&mm_read_u64(f,&enc_hash);
    if(ok&&(core_len>UINT64_MAX-44||enc_len>UINT64_MAX-44-core_len||
            44+core_len+enc_len!=file_len))ok=0;
    char *core=mm_suffix(path,".load-core"),*enc=mm_suffix(path,".load-encoder");
    if(!core||!enc)ok=0;
    FILE *cf=NULL,*ef=NULL;
    if(ok){cf=fopen(core,"wb");ef=fopen(enc,"wb");if(!cf||!ef)ok=0;}
    uint64_t actual_core=0,actual_enc=0;
    if(ok)ok=mm_copy(f,cf,core_len,&actual_core)&&mm_copy(f,ef,enc_len,&actual_enc)&&
        actual_core==core_hash&&actual_enc==enc_hash&&fgetc(f)==EOF;
    if(cf&&fclose(cf)!=0)ok=0;
    if(ef&&fclose(ef)!=0)ok=0;
    if(fclose(f)!=0)ok=0;
    AMA10Model loaded_model={0};AMA10Tokenizer loaded_tokenizer={0};
    AMA10Optimizer loaded_optimizer={0};AMA10State loaded_state={0};
    AMA10Encoder loaded_encoder={0};AMA10CheckpointProgress loaded_progress={0};
    if(ok)ok=ama10_checkpoint_load(core,&loaded_model,&loaded_tokenizer,&loaded_optimizer,
                                   &loaded_state,&loaded_progress)&&
             ama10_encoder_load(&loaded_encoder,enc)&&
             loaded_encoder.output_width==loaded_model.input_width;
    if(ok){
        ama10_model_free(model);ama10_tokenizer_free(tokenizer);ama10_optimizer_free(optimizer);
        ama10_state_free(state);ama10_encoder_free(encoder);
        *model=loaded_model;*tokenizer=loaded_tokenizer;*optimizer=loaded_optimizer;
        *state=loaded_state;*encoder=loaded_encoder;*progress=loaded_progress;
        memset(&loaded_model,0,sizeof(loaded_model));memset(&loaded_tokenizer,0,sizeof(loaded_tokenizer));
        memset(&loaded_optimizer,0,sizeof(loaded_optimizer));memset(&loaded_state,0,sizeof(loaded_state));
        memset(&loaded_encoder,0,sizeof(loaded_encoder));
    }
    ama10_model_free(&loaded_model);ama10_tokenizer_free(&loaded_tokenizer);
    ama10_optimizer_free(&loaded_optimizer);ama10_state_free(&loaded_state);
    ama10_encoder_free(&loaded_encoder);
    if(core)(void)remove(core);
    if(enc)(void)remove(enc);
    free(core);free(enc);return ok;
}
