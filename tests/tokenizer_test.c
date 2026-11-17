#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int reference_encode(const AMA10Tokenizer *t,const unsigned char *text,size_t length,
                           uint32_t **out,size_t *count){
    uint32_t *ids=(uint32_t*)malloc((length?length:1)*sizeof(*ids));
    if(!ids)return 0;
    for(size_t i=0;i<length;i++)ids[i]=text[i];
    size_t active=length;
    for(size_t merge=256;merge<t->vocabulary_size;merge++){
        if(t->tokens[merge].special)continue;
        size_t write=0;uint32_t left=t->tokens[merge].left,right=t->tokens[merge].right;
        for(size_t read=0;read<active;){
            if(read+1<active&&ids[read]==left&&ids[read+1]==right){ids[write++]=(uint32_t)merge;read+=2;}
            else ids[write++]=ids[read++];
        }
        active=write;
    }
    *out=ids;*count=active;return 1;
}

int main(void){
    static const unsigned char unit[]={
        'T','h','e',' ','q','u','i','c','k',' ','b','r','o','w','n',' ','f','o','x',' ',
        0xe4,0xbd,0xa0,0xe5,0xa5,0xbd,' ',0xf0,0x9f,0x99,0x82,' ',
        0xf0,0x9f,0x91,0xa9,0xe2,0x80,0x8d,0xf0,0x9f,0x94,0xac,0x20,0};
    const size_t repeats=16,unit_len=sizeof(unit);
    unsigned char *corpus=(unsigned char*)malloc(unit_len*repeats);
    if(!corpus)return 1;
    for(size_t i=0;i<repeats;i++)memcpy(corpus+i*unit_len,unit,unit_len);
    AMA10Tokenizer a={0},b={0},loaded={0};
    if(!ama10_tokenizer_train(&a,corpus,unit_len*repeats,300))return 2;
    if(a.vocabulary_size<=256||a.vocabulary_size>300)return 3;
    if(!ama10_tokenizer_train(&b,corpus,unit_len*repeats,300)||b.vocabulary_size!=a.vocabulary_size)return 4;
    for(size_t i=256;i<a.vocabulary_size;i++)
        if(a.tokens[i].left!=b.tokens[i].left||a.tokens[i].right!=b.tokens[i].right)return 5;
    size_t old_vocabulary=a.vocabulary_size;
    AMA10Token *old_tokens=(AMA10Token*)calloc(old_vocabulary,sizeof(AMA10Token));
    if(!old_tokens)return 12;
    for(size_t i=256;i<old_vocabulary;i++){
        old_tokens[i].left=a.tokens[i].left;old_tokens[i].right=a.tokens[i].right;
        old_tokens[i].length=a.tokens[i].length;
        old_tokens[i].bytes=(unsigned char*)malloc(old_tokens[i].length);
        if(!old_tokens[i].bytes)return 12;
        memcpy(old_tokens[i].bytes,a.tokens[i].bytes,old_tokens[i].length);
    }
    static const unsigned char extension_unit[]={'z','q','v'};
    unsigned char extension_text[sizeof(extension_unit)*16];
    for(size_t i=0;i<16;i++)memcpy(extension_text+i*sizeof(extension_unit),extension_unit,sizeof(extension_unit));
    if(!ama10_tokenizer_extend(&a,extension_text,sizeof(extension_text),320)||
       !ama10_tokenizer_extend(&b,extension_text,sizeof(extension_text),320)||
       a.vocabulary_size!=b.vocabulary_size||a.vocabulary_size<=old_vocabulary)return 13;
    for(size_t i=256;i<old_vocabulary;i++)
        if(old_tokens[i].left!=a.tokens[i].left||old_tokens[i].right!=a.tokens[i].right||
           old_tokens[i].length!=a.tokens[i].length||memcmp(old_tokens[i].bytes,a.tokens[i].bytes,a.tokens[i].length)||
           a.tokens[i].left!=b.tokens[i].left||a.tokens[i].right!=b.tokens[i].right)return 14;
    for(size_t i=old_vocabulary;i<a.vocabulary_size;i++)
        if(a.tokens[i].left!=b.tokens[i].left||a.tokens[i].right!=b.tokens[i].right)return 15;
    uint32_t *ids=NULL,*ids2=NULL;size_t count=0,count2=0;
    if(!ama10_tokenizer_encode(&a,unit,unit_len,&ids,&count))return 6;
    uint32_t *reference=NULL;size_t reference_count=0;
    if(!reference_encode(&a,unit,unit_len,&reference,&reference_count)||
       count!=reference_count||memcmp(ids,reference,count*sizeof(uint32_t)))return 16;
    free(reference);
    if(!reference_encode(&a,corpus,unit_len*repeats,&reference,&reference_count))return 17;
    uint32_t *corpus_ids=NULL;size_t corpus_count=0;
    if(!ama10_tokenizer_encode(&a,corpus,unit_len*repeats,&corpus_ids,&corpus_count)||
       corpus_count!=reference_count||memcmp(corpus_ids,reference,corpus_count*sizeof(uint32_t)))return 18;
    free(reference);free(corpus_ids);
    size_t long_len=128u*1024u;
    unsigned char *long_text=(unsigned char*)malloc(long_len);
    if(!long_text)return 19;
    for(size_t i=0;i<long_len;i++)long_text[i]=corpus[i%(unit_len*repeats)];
    if(!reference_encode(&a,long_text,long_len,&reference,&reference_count))return 20;
    corpus_ids=NULL;corpus_count=0;
    if(!ama10_tokenizer_encode(&a,long_text,long_len,&corpus_ids,&corpus_count)||
       corpus_count!=reference_count||memcmp(corpus_ids,reference,corpus_count*sizeof(uint32_t)))return 21;
    free(reference);free(corpus_ids);free(long_text);
    long_text=(unsigned char*)malloc(long_len);
    if(!long_text)return 22;
    uint32_t random_state=UINT32_C(0x6d2b79f5);
    for(size_t i=0;i<long_len;i++){random_state^=random_state<<13;random_state^=random_state>>17;random_state^=random_state<<5;long_text[i]=(unsigned char)random_state;}
    if(!reference_encode(&a,long_text,long_len,&reference,&reference_count))return 23;
    corpus_ids=NULL;corpus_count=0;
    if(!ama10_tokenizer_encode(&a,long_text,long_len,&corpus_ids,&corpus_count)||
       corpus_count!=reference_count||memcmp(corpus_ids,reference,corpus_count*sizeof(uint32_t)))return 24;
    free(reference);free(corpus_ids);free(long_text);
    unsigned char *decoded=NULL;size_t decoded_len=0;
    if(!ama10_tokenizer_decode(&a,ids,count,&decoded,&decoded_len)||decoded_len!=unit_len||memcmp(decoded,unit,unit_len))return 7;
    if(!ama10_tokenizer_save(&a,"build/tokenizer.test"))return 8;
    if(!ama10_tokenizer_load(&loaded,"build/tokenizer.test"))return 9;
    if(!ama10_tokenizer_encode(&loaded,unit,unit_len,&ids2,&count2)||count!=count2||memcmp(ids,ids2,count*sizeof(uint32_t)))return 10;
    if(!ama10_tokenizer_decode(&loaded,ids2,count2,&decoded,&decoded_len)||decoded_len!=unit_len||memcmp(decoded,unit,unit_len))return 11;
    {
        static const unsigned char image_token[]="<|image_code_000|>";
        static const unsigned char eos_token[]="<|image_eos|>";
        uint32_t image_id=0,eos_id=0;
        if(!ama10_tokenizer_add_special(&a,image_token,sizeof(image_token)-1,&image_id)||
           !ama10_tokenizer_add_special(&a,eos_token,sizeof(eos_token)-1,&eos_id)||
           image_id<old_vocabulary||eos_id!=image_id+1||!a.tokens[image_id].special||!a.tokens[eos_id].special)return 25;
        uint32_t *special_text_ids=NULL;size_t special_text_count=0;
        if(!ama10_tokenizer_encode(&a,image_token,sizeof(image_token)-1,&special_text_ids,&special_text_count))return 26;
        for(size_t i=0;i<special_text_count;i++)if(special_text_ids[i]==image_id)return 27;
        free(special_text_ids);
        if(!ama10_tokenizer_save(&a,"build/tokenizer-special.test")||
           !ama10_tokenizer_load(&loaded,"build/tokenizer-special.test")||
           loaded.vocabulary_size!=a.vocabulary_size||!loaded.tokens[image_id].special||
           loaded.tokens[eos_id].special!=1||loaded.tokens[image_id].length!=sizeof(image_token)-1||
           memcmp(loaded.tokens[image_id].bytes,image_token,sizeof(image_token)-1))return 28;
    }
    free(corpus);free(ids);free(ids2);free(decoded);
    for(size_t i=256;i<old_vocabulary;i++)free(old_tokens[i].bytes);
    free(old_tokens);
    ama10_tokenizer_free(&a);ama10_tokenizer_free(&b);ama10_tokenizer_free(&loaded);
    puts("BPE parity and round-trip; reserved image/EOS IDs remain outside byte encoding and survive save/load: ok");return 0;
}
