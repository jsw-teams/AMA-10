#include "dataset.h"
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#else
#include <sys/types.h>
#endif

#define DATA_HEADER_SIZE 56u
#define STREAM_CHUNK_BYTES 65536u
#define FNV_OFFSET UINT64_C(14695981039346656037)
#define FNV_PRIME UINT64_C(1099511628211)

static int seek64(FILE *f,uint64_t offset){
#ifdef _WIN32
    if(offset>(uint64_t)INT64_MAX)return 0;
    return _fseeki64(f,(__int64)offset,SEEK_SET)==0;
#else
    if(offset>(uint64_t)INT64_MAX)return 0;
    return fseeko(f,(off_t)offset,SEEK_SET)==0;
#endif
}
static int file_size64(FILE *f,uint64_t *size){
#ifdef _WIN32
    if(_fseeki64(f,0,SEEK_END)!=0)return 0;
    __int64 n=_ftelli64(f);if(n<0)return 0;*size=(uint64_t)n;return 1;
#else
    if(fseeko(f,0,SEEK_END)!=0)return 0;
    off_t n=ftello(f);if(n<0)return 0;*size=(uint64_t)n;return 1;
#endif
}
static void hash_bytes(uint64_t *h,const unsigned char *p,size_t n){
    for(size_t i=0;i<n;i++){*h^=p[i];*h*=FNV_PRIME;}
}
static void hash_u32(uint64_t *h,uint32_t x){
    unsigned char b[4]={(unsigned char)x,(unsigned char)(x>>8),
        (unsigned char)(x>>16),(unsigned char)(x>>24)};hash_bytes(h,b,4);
}
static void hash_u64(uint64_t *h,uint64_t x){hash_u32(h,(uint32_t)x);hash_u32(h,(uint32_t)(x>>32));}
static int write_u32(FILE *f,uint32_t x){
    unsigned char b[4]={(unsigned char)x,(unsigned char)(x>>8),
        (unsigned char)(x>>16),(unsigned char)(x>>24)};return fwrite(b,1,4,f)==4;
}
static int read_u32(FILE *f,uint32_t *x){
    unsigned char b[4];if(fread(b,1,4,f)!=4)return 0;
    *x=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);return 1;
}
static int write_u64(FILE *f,uint64_t x){return write_u32(f,(uint32_t)x)&&write_u32(f,(uint32_t)(x>>32));}
static int read_u64(FILE *f,uint64_t *x){
    uint32_t lo,hi;if(!read_u32(f,&lo)||!read_u32(f,&hi))return 0;
    *x=(uint64_t)lo|((uint64_t)hi<<32);return 1;
}
int ama10_file_fingerprint(const char *path,uint64_t *hash,uint64_t *byte_count){
    if(!path||!hash||!byte_count)return 0;
    FILE *f=fopen(path,"rb");if(!f)return 0;
    unsigned char buffer[65536];uint64_t h=FNV_OFFSET,n=0;size_t got=0;int ok=1;
    while((got=fread(buffer,1,sizeof(buffer),f))>0){
        if(n>UINT64_MAX-got){ok=0;break;}n+=(uint64_t)got;hash_bytes(&h,buffer,got);
    }
    if(ferror(f))ok=0;
    if(fclose(f)!=0)ok=0;
    if(!ok)return 0;
    *hash=h;*byte_count=n;return 1;
}
uint64_t ama10_tokenizer_fingerprint(const AMA10Tokenizer *t){
    if(!t||!t->tokens||t->vocabulary_size<256)return 0;
    uint64_t h=FNV_OFFSET;hash_u64(&h,(uint64_t)t->vocabulary_size);
    for(size_t i=256;i<t->vocabulary_size;i++){
        hash_u32(&h,t->tokens[i].special);
        hash_u32(&h,t->tokens[i].left);hash_u32(&h,t->tokens[i].right);
        hash_u64(&h,(uint64_t)t->tokens[i].length);
        hash_bytes(&h,t->tokens[i].bytes,t->tokens[i].length);
    }
    return h;
}
static int write_ids(FILE *f,const uint32_t *ids,size_t count,uint32_t width){
    for(size_t i=0;i<count;i++){
        if(width==2){unsigned char b[2]={(unsigned char)ids[i],(unsigned char)(ids[i]>>8)};
            if(fwrite(b,1,2,f)!=2)return 0;
        }else if(!write_u32(f,ids[i]))return 0;
    }
    return 1;
}
static int write_header(FILE *f,uint32_t token_width,uint32_t vocab,uint64_t raw_bytes,
                        uint64_t token_count,uint64_t raw_hash,uint64_t tokenizer_hash){
    static const unsigned char magic[8]={'A','M','A','1','0','D','S','1'};
    return fwrite(magic,1,8,f)==8&&write_u32(f,1)&&write_u32(f,token_width)&&
        write_u32(f,vocab)&&write_u32(f,0)&&write_u64(f,raw_bytes)&&
        write_u64(f,token_count)&&write_u64(f,raw_hash)&&write_u64(f,tokenizer_hash);
}
int ama10_token_dataset_build(const char *raw_path,const char *cache_path,
                              const AMA10Tokenizer *t,uint64_t expected_hash,uint64_t expected_bytes){
    if(!raw_path||!cache_path||!t||!t->tokens||t->vocabulary_size<256||
       t->vocabulary_size>UINT32_MAX||expected_bytes==0)return 0;
    uint32_t token_width=t->vocabulary_size<=65536?2u:4u;
    size_t max_token=1;
    for(size_t i=0;i<t->vocabulary_size;i++)if(t->tokens[i].length>max_token)max_token=t->tokens[i].length;
    uint64_t tok_hash=ama10_tokenizer_fingerprint(t);if(!tok_hash)return 0;
    if(expected_bytes>UINT64_MAX-8||strlen(cache_path)>SIZE_MAX-5)return 0;
    size_t path_len=strlen(cache_path);char *tmp=(char*)malloc(path_len+5),*bak=(char*)malloc(path_len+5);
    if(!tmp||!bak){free(tmp);free(bak);return 0;}
    memcpy(tmp,cache_path,path_len);memcpy(tmp+path_len,".tmp",5);
    memcpy(bak,cache_path,path_len);memcpy(bak+path_len,".bak",5);
    FILE *input=fopen(raw_path,"rb"),*output=fopen(tmp,"wb");
    if(!input||!output){if(input)fclose(input);if(output)fclose(output);free(tmp);free(bak);return 0;}
    unsigned char zeros[DATA_HEADER_SIZE]={0};
    int ok=fwrite(zeros,1,sizeof(zeros),output)==sizeof(zeros);
    size_t capacity=STREAM_CHUNK_BYTES,valid=0;unsigned char *buffer=(unsigned char*)malloc(capacity);
    if(!buffer)ok=0;
    uint64_t raw_hash=FNV_OFFSET,raw_read=0,token_count=0;
    int eof=0;
    while(ok&&(!eof||valid)){
        if(!eof){
            if(valid>SIZE_MAX-STREAM_CHUNK_BYTES){ok=0;break;}
            size_t needed=valid+STREAM_CHUNK_BYTES;
            if(needed>capacity){unsigned char *larger=(unsigned char*)realloc(buffer,needed);
                if(!larger){ok=0;break;}buffer=larger;capacity=needed;}
            size_t got=fread(buffer+valid,1,STREAM_CHUNK_BYTES,input);
            if(got){hash_bytes(&raw_hash,buffer+valid,got);raw_read+=(uint64_t)got;valid+=got;}
            if(got<STREAM_CHUNK_BYTES){if(ferror(input)){ok=0;break;}eof=1;}
        }
        if(!valid)continue;
        uint32_t *ids=NULL;size_t ids_count=0;
        if(!ama10_tokenizer_encode(t,buffer,valid,&ids,&ids_count)){ok=0;break;}
        size_t safe=eof?valid:(valid>max_token?valid-max_token:0),bytes=0,emit_count=0;
        while(emit_count<ids_count&&bytes+ t->tokens[ids[emit_count]].length<=safe){
            bytes+=t->tokens[ids[emit_count]].length;emit_count++;
        }
        if(eof){emit_count=ids_count;bytes=valid;}
        if(emit_count>0){
            if(token_count>UINT64_MAX-emit_count||!write_ids(output,ids,emit_count,token_width))ok=0;
            else token_count+=(uint64_t)emit_count;
            if(bytes<valid)memmove(buffer,buffer+bytes,valid-bytes);
            valid-=bytes;
        }
        free(ids);
        if(!emit_count&&!eof&&valid>SIZE_MAX-STREAM_CHUNK_BYTES){ok=0;break;}
    }
    free(buffer);
    if(fclose(input)!=0)ok=0;
    if(raw_read!=expected_bytes||raw_hash!=expected_hash||token_count==0)ok=0;
    if(ok&&(!seek64(output,0)||!write_header(output,token_width,(uint32_t)t->vocabulary_size,
                                              raw_read,token_count,raw_hash,tok_hash)))ok=0;
    if(fflush(output)!=0)ok=0;
    if(fclose(output)!=0)ok=0;
    int had_backup=0;
    if(ok){
        (void)remove(bak);
        if(rename(cache_path,bak)==0)had_backup=1;
        else if(errno!=ENOENT)ok=0;
    }
    if(ok&&rename(tmp,cache_path)!=0)ok=0;
    if(!ok){(void)remove(tmp);if(had_backup)(void)rename(bak,cache_path);}
    else if(had_backup)(void)remove(bak);
    free(tmp);free(bak);return ok;
}
int ama10_token_dataset_open(AMA10TokenDataset *d,const char *path,const AMA10Tokenizer *t,
                             uint64_t expected_hash,uint64_t expected_bytes){
    if(!d||!path||!t)return 0;
    memset(d,0,sizeof(*d));FILE *f=fopen(path,"rb");if(!f)return 0;
    unsigned char magic[8],expected[8]={'A','M','A','1','0','D','S','1'};
    uint32_t version=0,width=0,vocab=0,reserved=0;uint64_t raw_bytes=0,count=0,raw_hash=0,tok_hash=0,size=0;
    int ok=fread(magic,1,8,f)==8&&memcmp(magic,expected,8)==0&&read_u32(f,&version)&&version==1&&
        read_u32(f,&width)&&(width==2||width==4)&&read_u32(f,&vocab)&&read_u32(f,&reserved)&&
        read_u64(f,&raw_bytes)&&read_u64(f,&count)&&read_u64(f,&raw_hash)&&read_u64(f,&tok_hash);
    uint64_t expected_tok_hash=ama10_tokenizer_fingerprint(t);
    if(!ok||reserved||vocab!=t->vocabulary_size||raw_bytes!=expected_bytes||
       raw_hash!=expected_hash||tok_hash!=expected_tok_hash||!count||
       (width==2&&vocab>65536)||count>(UINT64_MAX-DATA_HEADER_SIZE)/width||
       !file_size64(f,&size)||size!=DATA_HEADER_SIZE+count*width){fclose(f);return 0;}
    d->file=f;d->token_count=count;d->raw_bytes=raw_bytes;d->raw_hash=raw_hash;
    d->tokenizer_hash=tok_hash;d->vocabulary=vocab;d->token_width=width;return 1;
}
int ama10_token_dataset_build_or_open(AMA10TokenDataset *d,const char *raw_path,const char *cache_path,
                                      const AMA10Tokenizer *t,uint64_t hash,uint64_t bytes){
    if(ama10_token_dataset_open(d,cache_path,t,hash,bytes))return 1;
    if(!ama10_token_dataset_build(raw_path,cache_path,t,hash,bytes))return 0;
    return ama10_token_dataset_open(d,cache_path,t,hash,bytes);
}
int ama10_token_dataset_read(AMA10TokenDataset *d,uint64_t cursor,size_t count,uint32_t *tokens){
    if(!d||!d->file||(!tokens&&count)||cursor>d->token_count||count>d->token_count-cursor||
       cursor>(UINT64_MAX-DATA_HEADER_SIZE)/d->token_width)return 0;
    uint64_t offset=DATA_HEADER_SIZE+cursor*d->token_width;
    if(!seek64(d->file,offset))return 0;
    for(size_t i=0;i<count;i++){
        uint32_t token=0;
        if(d->token_width==2){unsigned char b[2];if(fread(b,1,2,d->file)!=2)return 0;
            token=(uint32_t)b[0]|((uint32_t)b[1]<<8);
        }else if(!read_u32(d->file,&token))return 0;
        if(token>=d->vocabulary)return 0;
        tokens[i]=token;
    }
    return 1;
}
void ama10_token_dataset_close(AMA10TokenDataset *d){
    if(!d)return;
    if(d->file)fclose(d->file);
    memset(d,0,sizeof(*d));
}
