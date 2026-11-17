#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint64_t key; size_t count; } PairSlot;
typedef struct { uint32_t rank; size_t left, right, left_version, right_version; } EncodeCandidate;
typedef struct { EncodeCandidate *items; size_t count, capacity; } EncodeHeap;
static uint64_t pair_hash(uint64_t x);

static int candidate_less(const EncodeCandidate *a,const EncodeCandidate *b){
    return a->rank<b->rank||(a->rank==b->rank&&a->left<b->left);
}
static int heap_push(EncodeHeap *h,EncodeCandidate value){
    if(h->count==h->capacity){
        size_t cap=h->capacity?h->capacity*2:256;
        if(cap<h->capacity||cap>SIZE_MAX/sizeof(*h->items))return 0;
        EncodeCandidate *p=(EncodeCandidate*)realloc(h->items,cap*sizeof(*p));
        if(!p)return 0;
        h->items=p;h->capacity=cap;
    }
    size_t i=h->count++;
    while(i){size_t parent=(i-1)/2;if(!candidate_less(&value,&h->items[parent]))break;
        h->items[i]=h->items[parent];i=parent;}
    h->items[i]=value;return 1;
}
static EncodeCandidate heap_pop(EncodeHeap *h){
    EncodeCandidate top=h->items[0],last=h->items[--h->count];size_t i=0;
    while(i<h->count){size_t child=i*2+1;if(child>=h->count)break;
        if(child+1<h->count&&candidate_less(&h->items[child+1],&h->items[child]))child++;
        if(!candidate_less(&h->items[child],&last))break;
        h->items[i]=h->items[child];i=child;}
    if(h->count)h->items[i]=last;
    return top;
}
static int pair_rank(const PairSlot *table,size_t capacity,uint64_t key,uint32_t *rank){
    size_t slot=(size_t)pair_hash(key)&(capacity-1);
    while(table[slot].key!=UINT64_MAX){
        if(table[slot].key==key){*rank=(uint32_t)table[slot].count;return 1;}
        slot=(slot+1)&(capacity-1);
    }
    return 0;
}
static int queue_pair(EncodeHeap *heap,const PairSlot *table,size_t table_capacity,
                      const size_t *next,const size_t *version,const uint32_t *ids,size_t left){
    size_t right=next[left];uint32_t rank;
    if(right==SIZE_MAX)return 1;
    uint64_t key=((uint64_t)ids[left]<<32)|(uint64_t)ids[right];
    if(!pair_rank(table,table_capacity,key,&rank))return 1;
    EncodeCandidate c={rank,left,right,version[left],version[right]};
    return heap_push(heap,c);
}

static int reserve(AMA10Tokenizer *t,size_t count){
    if(count<=t->capacity)return 1;
    if(count>SIZE_MAX/sizeof(AMA10Token))return 0;
    AMA10Token *p=(AMA10Token*)realloc(t->tokens,count*sizeof(AMA10Token));
    if(!p)return 0;
    memset(p+t->capacity,0,(count-t->capacity)*sizeof(AMA10Token));
    t->tokens=p;t->capacity=count;return 1;
}
static int init_bytes(AMA10Tokenizer *t,size_t capacity){
    memset(t,0,sizeof(*t));
    if(capacity<256||!reserve(t,capacity))return 0;
    for(size_t i=0;i<256;i++){
        t->tokens[i].bytes=(unsigned char*)malloc(1);
        if(!t->tokens[i].bytes){ama10_tokenizer_free(t);return 0;}
        t->tokens[i].bytes[0]=(unsigned char)i;t->tokens[i].length=1;
    }
    t->vocabulary_size=256;return 1;
}
void ama10_tokenizer_free(AMA10Tokenizer *t){
    if(!t)return;
    for(size_t i=0;i<t->vocabulary_size;i++)free(t->tokens[i].bytes);
    free(t->tokens);memset(t,0,sizeof(*t));
}
static uint64_t pair_hash(uint64_t x){
    x^=x>>30;x*=UINT64_C(0xbf58476d1ce4e5b9);
    x^=x>>27;x*=UINT64_C(0x94d049bb133111eb);return x^(x>>31);
}
static int pair_table_capacity(size_t length,size_t *out){
    if(length>SIZE_MAX/2)return 0;
    size_t need=length*2,cap=16;
    while(cap<need){if(cap>SIZE_MAX/2)return 0;cap*=2;}
    if(cap>SIZE_MAX/sizeof(PairSlot))return 0;
    *out=cap;return 1;
}
static int append_merge(AMA10Tokenizer *t,uint32_t a,uint32_t b){
    if(t->vocabulary_size>=UINT32_MAX||a>=t->vocabulary_size||b>=t->vocabulary_size)return 0;
    size_t al=t->tokens[a].length,bl=t->tokens[b].length;
    if(al>SIZE_MAX-bl)return 0;
    size_t n=al+bl;unsigned char *bytes=(unsigned char*)malloc(n?n:1);
    if(!bytes)return 0;
    memcpy(bytes,t->tokens[a].bytes,al);memcpy(bytes+al,t->tokens[b].bytes,bl);
    size_t id=t->vocabulary_size;
    t->tokens[id].left=a;t->tokens[id].right=b;t->tokens[id].bytes=bytes;t->tokens[id].length=n;t->tokens[id].special=0;
    t->vocabulary_size++;return 1;
}
int ama10_tokenizer_add_special(AMA10Tokenizer *t,const unsigned char *bytes,size_t length,uint32_t *token_id){
    if(!t||!t->tokens||!bytes||!length||t->vocabulary_size<256||t->vocabulary_size>=UINT32_MAX)return 0;
    for(size_t i=256;i<t->vocabulary_size;i++)if(t->tokens[i].special&&t->tokens[i].length==length&&
        memcmp(t->tokens[i].bytes,bytes,length)==0)return 0;
    if(!reserve(t,t->vocabulary_size+1))return 0;
    unsigned char *copy=(unsigned char*)malloc(length);if(!copy)return 0;memcpy(copy,bytes,length);
    size_t id=t->vocabulary_size;t->tokens[id].left=UINT32_MAX;t->tokens[id].right=UINT32_MAX;
    t->tokens[id].bytes=copy;t->tokens[id].length=length;t->tokens[id].special=1;t->vocabulary_size++;
    if(token_id)*token_id=(uint32_t)id;
    return 1;
}
int ama10_tokenizer_train(AMA10Tokenizer *t,const unsigned char *text,size_t length,size_t target){
    if(!t||(!text&&length)||target<256||target>UINT32_MAX||length>SIZE_MAX/sizeof(uint32_t))return 0;
    ama10_tokenizer_free(t);
    if(!init_bytes(t,target))return 0;
    if(length<2||target==256)return 1;
    uint32_t *ids=(uint32_t*)malloc(length*sizeof(uint32_t));
    size_t table_cap;
    if(!ids||!pair_table_capacity(length,&table_cap)){free(ids);ama10_tokenizer_free(t);return 0;}
    PairSlot *table=(PairSlot*)malloc(table_cap*sizeof(PairSlot));
    if(!table){free(ids);ama10_tokenizer_free(t);return 0;}
    for(size_t i=0;i<length;i++)ids[i]=text[i];
    size_t active=length;
    while(t->vocabulary_size<target&&active>1){
        for(size_t i=0;i<table_cap;i++){table[i].key=UINT64_MAX;table[i].count=0;}
        for(size_t i=0;i+1<active;i++){
            uint64_t key=((uint64_t)ids[i]<<32)|(uint64_t)ids[i+1];
            size_t slot=(size_t)pair_hash(key)&(table_cap-1);
            while(table[slot].key!=UINT64_MAX&&table[slot].key!=key)slot=(slot+1)&(table_cap-1);
            if(table[slot].key==UINT64_MAX)table[slot].key=key;
            table[slot].count++;
        }
        uint64_t best=UINT64_MAX;size_t best_count=1;
        for(size_t i=0;i<table_cap;i++)if(table[i].key!=UINT64_MAX&&
            (table[i].count>best_count||(table[i].count==best_count&&table[i].key<best))){
            best=table[i].key;best_count=table[i].count;
        }
        if(best==UINT64_MAX||best_count<2)break;
        uint32_t left=(uint32_t)(best>>32),right=(uint32_t)best;
        if(!append_merge(t,left,right)){free(table);free(ids);ama10_tokenizer_free(t);return 0;}
        uint32_t merged=(uint32_t)(t->vocabulary_size-1);size_t write=0;
        for(size_t read=0;read<active;){
            if(read+1<active&&ids[read]==left&&ids[read+1]==right){ids[write++]=merged;read+=2;}
            else ids[write++]=ids[read++];
        }
        active=write;
    }
    free(table);free(ids);return 1;
}
int ama10_tokenizer_extend(AMA10Tokenizer *t,const unsigned char *text,size_t length,size_t target){
    if(!t||!t->tokens||t->vocabulary_size<256||(!text&&length)||
       target<t->vocabulary_size||target>UINT32_MAX||length>SIZE_MAX/sizeof(uint32_t))return 0;
    if(target==t->vocabulary_size||length<2)return 1;
    if(!reserve(t,target))return 0;
    size_t original_vocabulary=t->vocabulary_size;
    uint32_t *ids=NULL;size_t active=0;
    if(!ama10_tokenizer_encode(t,text,length,&ids,&active))return 0;
    size_t table_cap;
    if(!pair_table_capacity(active,&table_cap)){free(ids);return 0;}
    PairSlot *table=(PairSlot*)malloc(table_cap*sizeof(PairSlot));
    if(!table){free(ids);return 0;}
    while(t->vocabulary_size<target&&active>1){
        for(size_t i=0;i<table_cap;i++){table[i].key=UINT64_MAX;table[i].count=0;}
        for(size_t i=0;i+1<active;i++){
            uint64_t key=((uint64_t)ids[i]<<32)|(uint64_t)ids[i+1];
            size_t slot=(size_t)pair_hash(key)&(table_cap-1);
            while(table[slot].key!=UINT64_MAX&&table[slot].key!=key)slot=(slot+1)&(table_cap-1);
            if(table[slot].key==UINT64_MAX)table[slot].key=key;
            table[slot].count++;
        }
        uint64_t best=UINT64_MAX;size_t best_count=1;
        for(size_t i=0;i<table_cap;i++)if(table[i].key!=UINT64_MAX&&
            (table[i].count>best_count||(table[i].count==best_count&&table[i].key<best))){
            best=table[i].key;best_count=table[i].count;
        }
        if(best==UINT64_MAX||best_count<2)break;
        uint32_t left=(uint32_t)(best>>32),right=(uint32_t)best;
        if(!append_merge(t,left,right)){
            while(t->vocabulary_size>original_vocabulary){
                t->vocabulary_size--;free(t->tokens[t->vocabulary_size].bytes);
                memset(&t->tokens[t->vocabulary_size],0,sizeof(AMA10Token));
            }
            free(table);free(ids);return 0;
        }
        uint32_t merged=(uint32_t)(t->vocabulary_size-1);size_t write=0;
        for(size_t read=0;read<active;){
            if(read+1<active&&ids[read]==left&&ids[read+1]==right){ids[write++]=merged;read+=2;}
            else ids[write++]=ids[read++];
        }
        active=write;
    }
    free(table);free(ids);return 1;
}
int ama10_tokenizer_encode(const AMA10Tokenizer *t,const unsigned char *text,size_t length,
                           uint32_t **out,size_t *count){
    if(!t||!t->tokens||(!text&&length)||!out||!count||length>SIZE_MAX/sizeof(uint32_t)||
       t->vocabulary_size<256||t->vocabulary_size>UINT32_MAX)return 0;
    *out=NULL;*count=0;
    if(length==0){uint32_t *empty=(uint32_t*)malloc(sizeof(*empty));if(!empty)return 0;*out=empty;return 1;}
    size_t merges=t->vocabulary_size-256,table_capacity;
    if(!pair_table_capacity(merges?merges:1,&table_capacity)||
       length>SIZE_MAX/sizeof(size_t)||length>SIZE_MAX/sizeof(uint32_t))return 0;
    PairSlot *table=(PairSlot*)malloc(table_capacity*sizeof(*table));
    uint32_t *ids=(uint32_t*)malloc(length*sizeof(*ids));
    size_t *previous=(size_t*)malloc(length*sizeof(*previous));
    size_t *next=(size_t*)malloc(length*sizeof(*next));
    size_t *version=(size_t*)calloc(length,sizeof(*version));
    EncodeHeap heap={0};
    if(!table||!ids||!previous||!next||!version){free(table);free(ids);free(previous);free(next);free(version);return 0;}
    for(size_t i=0;i<table_capacity;i++){table[i].key=UINT64_MAX;table[i].count=0;}
    for(size_t merge=256;merge<t->vocabulary_size;merge++){
        if(t->tokens[merge].special)continue;
        uint64_t key=((uint64_t)t->tokens[merge].left<<32)|(uint64_t)t->tokens[merge].right;
        size_t slot=(size_t)pair_hash(key)&(table_capacity-1);
        while(table[slot].key!=UINT64_MAX&&table[slot].key!=key)slot=(slot+1)&(table_capacity-1);
        if(table[slot].key==UINT64_MAX){table[slot].key=key;table[slot].count=merge;}
    }
    for(size_t i=0;i<length;i++){
        ids[i]=text[i];previous[i]=i?i-1:SIZE_MAX;next[i]=i+1<length?i+1:SIZE_MAX;
    }
    int ok=1;
    for(size_t i=0;i+1<length&&ok;i++)ok=queue_pair(&heap,table,table_capacity,next,version,ids,i);
    while(ok&&heap.count){
        EncodeCandidate c=heap_pop(&heap);
        if(c.left>=length||c.right>=length||next[c.left]!=c.right||
           version[c.left]!=c.left_version||version[c.right]!=c.right_version)continue;
        size_t left=c.left,right=c.right,after=next[right],before=previous[left];
        if(version[left]==SIZE_MAX||version[right]==SIZE_MAX||(after!=SIZE_MAX&&version[after]==SIZE_MAX)){ok=0;break;}
        ids[left]=c.rank;next[left]=after;version[left]++;
        previous[right]=SIZE_MAX;next[right]=SIZE_MAX;version[right]++;
        if(after!=SIZE_MAX){previous[after]=left;version[after]++;}
        if(before!=SIZE_MAX)ok=queue_pair(&heap,table,table_capacity,next,version,ids,before);
        if(ok)ok=queue_pair(&heap,table,table_capacity,next,version,ids,left);
        if(ok&&after!=SIZE_MAX)ok=queue_pair(&heap,table,table_capacity,next,version,ids,after);
    }
    free(table);free(previous);free(version);free(heap.items);
    if(!ok){free(next);free(ids);return 0;}
    /* The linked list is rooted at index zero; compact surviving IDs in place. */
    size_t active=0;
    for(size_t i=0;i!=SIZE_MAX;){ids[active++]=ids[i];i=next[i];}
    free(next);*out=ids;*count=active;return 1;
}
int ama10_tokenizer_decode(const AMA10Tokenizer *t,const uint32_t *ids,size_t count,
                           unsigned char **out,size_t *length){
    if(!t||!t->tokens||(!ids&&count)||!out||!length)return 0;
    size_t n=0;
    for(size_t i=0;i<count;i++){
        if(ids[i]>=t->vocabulary_size||t->tokens[ids[i]].length>SIZE_MAX-n)return 0;
        n+=t->tokens[ids[i]].length;
    }
    unsigned char *bytes=(unsigned char*)malloc(n?n:1);if(!bytes)return 0;
    size_t pos=0;for(size_t i=0;i<count;i++){
        size_t part=t->tokens[ids[i]].length;
        memcpy(bytes+pos,t->tokens[ids[i]].bytes,part);pos+=part;
    }
    *out=bytes;*length=n;return 1;
}
static int write_u32(FILE *f,uint32_t x){
    unsigned char b[4]={(unsigned char)x,(unsigned char)(x>>8),(unsigned char)(x>>16),(unsigned char)(x>>24)};
    return fwrite(b,1,4,f)==4;
}
static int read_u32(FILE *f,uint32_t *x){
    unsigned char b[4];if(fread(b,1,4,f)!=4)return 0;
    *x=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);return 1;
}
int ama10_tokenizer_save(const AMA10Tokenizer *t,const char *path){
    if(!t||!t->tokens||t->vocabulary_size<256||t->vocabulary_size>UINT32_MAX||!path)return 0;
    FILE *f=fopen(path,"wb");if(!f)return 0;
    static const unsigned char magic[8]={'A','M','A','1','0','T','K','2'};
    int ok=fwrite(magic,1,8,f)==8&&write_u32(f,2)&&write_u32(f,(uint32_t)t->vocabulary_size);
    for(size_t i=256;i<t->vocabulary_size&&ok;i++){
        if(t->tokens[i].length>UINT32_MAX||t->tokens[i].special>1){ok=0;break;}
        ok=write_u32(f,t->tokens[i].special)&&write_u32(f,t->tokens[i].left)&&write_u32(f,t->tokens[i].right)&&
           write_u32(f,(uint32_t)t->tokens[i].length)&&
           fwrite(t->tokens[i].bytes,1,t->tokens[i].length,f)==t->tokens[i].length;
    }
    if(fclose(f)!=0)ok=0;
    return ok;
}
int ama10_tokenizer_load(AMA10Tokenizer *t,const char *path){
    if(!t||!path)return 0;
    FILE *f=fopen(path,"rb");if(!f)return 0;
    if(fseek(f,0,SEEK_END)!=0){fclose(f);return 0;}
    long file_size=ftell(f);if(file_size<16||fseek(f,0,SEEK_SET)!=0){fclose(f);return 0;}
    unsigned char magic[8];uint32_t version=0,vocab=0;
    static const unsigned char expected_v1[8]={'A','M','A','1','0','T','K','1'};
    static const unsigned char expected_v2[8]={'A','M','A','1','0','T','K','2'};
    int ok=fread(magic,1,8,f)==8&&read_u32(f,&version)&&
           ((version==1&&!memcmp(magic,expected_v1,8))||(version==2&&!memcmp(magic,expected_v2,8)))&&
           read_u32(f,&vocab)&&vocab>=256;
    if(ok&&(uint64_t)(vocab-256)>(uint64_t)(file_size-16)/12)ok=0;
    AMA10Tokenizer tmp={0};
    if(ok)ok=init_bytes(&tmp,(size_t)vocab);
    for(uint32_t i=256;i<vocab&&ok;i++){
        uint32_t special=0,left,right,n;
        if((version==2&&!read_u32(f,&special))||!read_u32(f,&left)||!read_u32(f,&right)||!read_u32(f,&n)||
           special>1||n==0||(special?(left!=UINT32_MAX||right!=UINT32_MAX):(left>=i||right>=i))){ok=0;break;}
        if(!special){size_t expected_length=tmp.tokens[left].length;
            if(tmp.tokens[right].length>SIZE_MAX-expected_length||expected_length+tmp.tokens[right].length!=n){ok=0;break;}}
        unsigned char *bytes=(unsigned char*)malloc(n);
        if(!bytes){ok=0;break;}
        if(fread(bytes,1,n,f)!=n||(!special&&(memcmp(bytes,tmp.tokens[left].bytes,tmp.tokens[left].length)!=0||
           memcmp(bytes+tmp.tokens[left].length,tmp.tokens[right].bytes,tmp.tokens[right].length)!=0))){
            free(bytes);ok=0;break;
        }
        tmp.tokens[i].left=left;tmp.tokens[i].right=right;tmp.tokens[i].length=n;tmp.tokens[i].bytes=bytes;tmp.tokens[i].special=special;
        tmp.vocabulary_size++;
    }
    if(ok&&fgetc(f)!=EOF)ok=0;
    if(fclose(f)!=0)ok=0;
    if(ok){ama10_tokenizer_free(t);*t=tmp;}else ama10_tokenizer_free(&tmp);
    return ok;
}
