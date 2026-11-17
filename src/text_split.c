#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#define READ_BUFFER 65536u
#define SPLIT_HASH_OFFSET UINT64_C(14695981039346656037)
#define SPLIT_HASH_PRIME UINT64_C(1099511628211)

static uint64_t hash_doc(const char *p,size_t n) {
    uint64_t h=SPLIT_HASH_OFFSET;size_t i;for(i=0;i<n;i++){h^=(unsigned char)p[i];h*=SPLIT_HASH_PRIME;}return h;
}

static int mkdir_parent(const char *path) {
#ifdef _WIN32
    char *copy=_strdup(path),*p;if(!copy)return 0;
    for(p=copy+1;*p;p++)if(*p=='/'||*p=='\\'){
        char hold=*p;*p=0;if(!CreateDirectoryA(copy,NULL)&&GetLastError()!=ERROR_ALREADY_EXISTS){free(copy);return 0;}*p=hold;
    }
    free(copy);return 1;
#else
    (void)path;return 1;
#endif
}

static int commit_file(const char *partial,const char *final) {
#ifdef _WIN32
    return MoveFileExA(partial,final,MOVEFILE_WRITE_THROUGH)!=0;
#else
    return rename(partial,final)==0;
#endif
}

int ama10_split_documents(const char *input,const char *train_path,const char *validation_path,unsigned validation_percent) {
    FILE *in=NULL,*train=NULL,*validation=NULL;char *train_tmp=NULL,*validation_tmp=NULL,*doc=NULL;
    size_t doc_len=0,doc_cap=0;unsigned char buffer[READ_BUFFER];size_t got,i;
    unsigned long long train_docs=0,val_docs=0,train_bytes=0,val_bytes=0;int previous_lf=0,ok=0;
    if(!input||!train_path||!validation_path||validation_percent<1||validation_percent>49||
       strcmp(train_path,validation_path)==0||!mkdir_parent(train_path)||!mkdir_parent(validation_path))return 0;
    train_tmp=(char *)malloc(strlen(train_path)+9);validation_tmp=(char *)malloc(strlen(validation_path)+9);
    if(!train_tmp||!validation_tmp)goto done;
    sprintf(train_tmp,"%s.partial",train_path);sprintf(validation_tmp,"%s.partial",validation_path);
    if(GetFileAttributesA(train_path)!=INVALID_FILE_ATTRIBUTES||GetFileAttributesA(validation_path)!=INVALID_FILE_ATTRIBUTES)goto done;
    (void)DeleteFileA(train_tmp);(void)DeleteFileA(validation_tmp);
    in=fopen(input,"rb");train=fopen(train_tmp,"wb");validation=fopen(validation_tmp,"wb");
    if(!in||!train||!validation)goto done;
    while((got=fread(buffer,1,sizeof(buffer),in))>0){
        for(i=0;i<got;i++){
            char c=(char)buffer[i];
            if(c=='\n'&&previous_lf){
                FILE *dst;size_t n=doc_len;if(n&&doc[n-1]=='\n')n--;
                if(n){uint64_t h=hash_doc(doc,n);dst=(h%100)<validation_percent?validation:train;
                    if(fwrite(doc,1,n,dst)!=n||fwrite("\n\n",1,2,dst)!=2)goto done;
                    if(dst==validation){val_docs++;val_bytes+=n+2;}else{train_docs++;train_bytes+=n+2;}}
                doc_len=0;previous_lf=0;continue;
            }
            if(doc_len==doc_cap){size_t next=doc_cap?doc_cap*2:8192;char *larger;
                if(next<doc_cap){goto done;}larger=(char *)realloc(doc,next);if(!larger)goto done;doc=larger;doc_cap=next;}
            doc[doc_len++]=c;previous_lf=(c=='\n');
        }
    }
    if(ferror(in))goto done;
    if(doc_len){size_t n=doc_len;FILE *dst;if(n&&doc[n-1]=='\n')n--;
        if(n){dst=(hash_doc(doc,n)%100)<validation_percent?validation:train;
            if(fwrite(doc,1,n,dst)!=n||fwrite("\n\n",1,2,dst)!=2)goto done;
            if(dst==validation){val_docs++;val_bytes+=n+2;}else{train_docs++;train_bytes+=n+2;}}}
    if(fflush(train)!=0||fflush(validation)!=0)goto done;
    if(fclose(train)!=0){train=NULL;goto done;}train=NULL;
    if(fclose(validation)!=0){validation=NULL;goto done;}validation=NULL;
    if(fclose(in)!=0){in=NULL;goto done;}in=NULL;
    if(!commit_file(train_tmp,train_path)||!commit_file(validation_tmp,validation_path))goto done;
    printf("documents_train=%llu documents_validation=%llu train_bytes=%llu validation_bytes=%llu validation_percent=%u\n",
        train_docs,val_docs,train_bytes,val_bytes,validation_percent);ok=1;
done:
    if(in)fclose(in);
    if(train)fclose(train);
    if(validation)fclose(validation);
    if(!ok){if(train_tmp)DeleteFileA(train_tmp);if(validation_tmp)DeleteFileA(validation_tmp);}
    free(train_tmp);free(validation_tmp);free(doc);return ok;
}

#ifdef AMA10_TEXT_SPLIT_CLI
int main(int argc,char **argv) {
    char *end=NULL;unsigned long p;
    if(argc!=5){fprintf(stderr,"Usage: %s <processed.txt> <train.txt> <validation.txt> <validation_percent=1..49>\n",argv[0]);return 2;}
    p=strtoul(argv[4],&end,10);if(!end||*end||p<1||p>49){fputs("validation_percent must be 1..49.\n",stderr);return 2;}
    return ama10_split_documents(argv[1],argv[2],argv[3],(unsigned)p)?0:1;
}
#endif
