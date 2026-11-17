#include "eval.h"
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int read_file(const char *path,unsigned char **bytes,size_t *length){
    FILE *f=fopen(path,"rb");if(!f)return 0;
    if(fseek(f,0,SEEK_END)!=0){fclose(f);return 0;}
    long n=ftell(f);if(n<0||fseek(f,0,SEEK_SET)!=0){fclose(f);return 0;}
#if SIZE_MAX < LONG_MAX
    if((unsigned long)n>SIZE_MAX-1){fclose(f);return 0;}
#endif
    unsigned char *p=(unsigned char*)malloc((size_t)n+1);if(!p){fclose(f);return 0;}
    int ok=fread(p,1,(size_t)n,f)==(size_t)n;
    if(fclose(f)!=0)ok=0;
    if(!ok){free(p);return 0;}
    *bytes=p;*length=(size_t)n;return 1;
}
int main(int argc,char **argv){
    if(argc!=3){fprintf(stderr,"usage: %s <reference.txt> <hypothesis.txt>\n",argv[0]);return 2;}
    unsigned char *reference=NULL,*hypothesis=NULL;size_t reference_bytes=0,hypothesis_bytes=0;
    double wer=0.0,cer=0.0;uint64_t word_errors=0,words=0,char_errors=0,characters=0;
    int ok=read_file(argv[1],&reference,&reference_bytes)&&
        read_file(argv[2],&hypothesis,&hypothesis_bytes)&&
        ama10_text_error_rates(reference,reference_bytes,hypothesis,hypothesis_bytes,
            &wer,&cer,&word_errors,&words,&char_errors,&characters);
    if(ok)printf("WER=%.6f (%llu/%llu) CER=%.6f (%llu/%llu)\n",wer,
        (unsigned long long)word_errors,(unsigned long long)words,cer,
        (unsigned long long)char_errors,(unsigned long long)characters);
    else fprintf(stderr,"text scoring failed (invalid UTF-8 or file error)\n");
    free(reference);free(hypothesis);return ok?0:1;
}
