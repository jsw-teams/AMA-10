#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void){
    static const unsigned char unit[]={
        'A','M','A',' ','d','a','t','a',' ','s','t','r','e','a','m',' ',
        0xe4,0xbd,0xa0,0xe5,0xa5,0xbd,' ',0xf0,0x9f,0x99,0x82,' ',0};
    const size_t repeats=3500,raw_size=sizeof(unit)*repeats;
    unsigned char *raw=(unsigned char*)malloc(raw_size);
    if(!raw)return 1;
    for(size_t i=0;i<repeats;i++)memcpy(raw+i*sizeof(unit),unit,sizeof(unit));
    FILE *f=fopen("build/dataset.raw","wb");
    if(!f||fwrite(raw,1,raw_size,f)!=raw_size||fclose(f)!=0)return 2;
    AMA10Tokenizer tokenizer={0};AMA10TokenDataset dataset={0};
    uint32_t *expected=NULL;size_t expected_count=0;uint64_t hash=0,bytes=0;
    if(!ama10_tokenizer_train(&tokenizer,raw,raw_size,320)||
       !ama10_tokenizer_encode(&tokenizer,raw,raw_size,&expected,&expected_count)||
       !ama10_file_fingerprint("build/dataset.raw",&hash,&bytes)||bytes!=raw_size||
       !ama10_token_dataset_build_or_open(&dataset,"build/dataset.raw","build/dataset.ids",
                                          &tokenizer,hash,bytes)||dataset.token_count!=expected_count){
        fprintf(stderr,"streamed token dataset creation failed\n");return 3;
    }
    uint32_t block[997];uint64_t cursor=0;
    while(cursor<dataset.token_count){
        size_t count=(size_t)(dataset.token_count-cursor);
        if(count>997)count=997;
        if(!ama10_token_dataset_read(&dataset,cursor,count,block)||
           memcmp(block,expected+(size_t)cursor,count*sizeof(uint32_t))){
            fprintf(stderr,"streamed token IDs differ at cursor %llu\n",(unsigned long long)cursor);return 4;
        }
        cursor+=count;
    }
    if(ama10_token_dataset_read(&dataset,dataset.token_count,1,block))return 5;
    printf("streamed token cache preserved %zu exact BPE IDs across raw chunk boundaries\n",expected_count);
    ama10_token_dataset_close(&dataset);ama10_tokenizer_free(&tokenizer);
    free(expected);free(raw);return 0;
}
