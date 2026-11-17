#include "eval.h"
#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {uint64_t hash;unsigned char used;} NgramSlot;
static uint64_t mix_ngram(const uint32_t *ids){
    uint64_t h=UINT64_C(14695981039346656037);
    for(size_t j=0;j<4;j++)for(unsigned b=0;b<4;b++){
        h^=(ids[j]>>(8*b))&255u;h*=UINT64_C(1099511628211);
    }
    return h;
}
static size_t read_codepoint(const unsigned char *p,size_t remaining,uint32_t *cp){
    unsigned char a=p[0];
    if(a<0x80){*cp=a;return 1;}
    unsigned need;uint32_t value,min;
    if(a>=0xc2&&a<=0xdf){need=2;value=a&0x1f;min=0x80;}
    else if(a>=0xe0&&a<=0xef){need=3;value=a&0x0f;min=0x800;}
    else if(a>=0xf0&&a<=0xf4){need=4;value=a&0x07;min=0x10000;}
    else return 0;
    if(remaining<need)return 0;
    for(unsigned i=1;i<need;i++){
        if((p[i]&0xc0)!=0x80)return 0;
        value=(value<<6)|(p[i]&0x3f);
    }
    if(value<min||value>0x10ffff||(value>=0xd800&&value<=0xdfff))return 0;
    *cp=value;return need;
}
typedef struct {size_t start,length;} TextSpan;
static int normalize_text(const unsigned char *src,size_t n,unsigned char **out,size_t *out_n){
    if((!src&&n)||!out||!out_n||n==SIZE_MAX)return 0;
    unsigned char *dst=(unsigned char*)malloc(n+1);if(!dst)return 0;
    size_t used=0;int pending_space=0;
    for(size_t i=0;i<n;){
        unsigned char c=src[i];
        if(c<0x80){
            i++;
            if(isspace(c)){pending_space=used>0;continue;}
            if(isalnum(c)){
                if(pending_space)dst[used++]=' ';
                pending_space=0;dst[used++]=(unsigned char)tolower(c);
            }
            continue;
        }
        uint32_t cp=0;size_t width=read_codepoint(src+i,n-i,&cp);
        if(!width){free(dst);return 0;}
        if(pending_space)dst[used++]=' ';
        pending_space=0;memcpy(dst+used,src+i,width);used+=width;i+=width;
    }
    dst[used]='\0';*out=dst;*out_n=used;return 1;
}
static int split_words(const unsigned char *s,size_t n,TextSpan **out,size_t *count){
    size_t words=0;int in_word=0;
    for(size_t i=0;i<n;i++){if(s[i]==' '){in_word=0;}else if(!in_word){words++;in_word=1;}}
    if(words>SIZE_MAX/sizeof(TextSpan))return 0;
    TextSpan *spans=words?(TextSpan*)malloc(words*sizeof(*spans)):NULL;
    if(words&&!spans)return 0;
    size_t index=0,i=0;
    while(i<n){while(i<n&&s[i]==' ')i++;if(i==n)break;
        size_t start=i;while(i<n&&s[i]!=' ')i++;
        spans[index++]=(TextSpan){start,i-start};}
    *out=spans;*count=index;return 1;
}
static int words_equal(const unsigned char *a,TextSpan x,const unsigned char *b,TextSpan y){
    return x.length==y.length&&!memcmp(a+x.start,b+y.start,x.length);
}
static int word_distance(const unsigned char *a,const TextSpan *x,size_t nx,
                         const unsigned char *b,const TextSpan *y,size_t ny,size_t *distance){
    size_t nr=nx,nc=ny;int swapped=0;
    if(nc>nr){nr=ny;nc=nx;swapped=1;}
    if(nc==SIZE_MAX||nc+1>SIZE_MAX/sizeof(size_t))return 0;
    size_t *previous=(size_t*)malloc((nc+1)*sizeof(size_t));
    size_t *current=(size_t*)malloc((nc+1)*sizeof(size_t));
    if(!previous||!current){free(previous);free(current);return 0;}
    for(size_t j=0;j<=nc;j++)previous[j]=j;
    for(size_t i=1;i<=nr;i++){
        current[0]=i;
        for(size_t j=1;j<=nc;j++){
            int equal=swapped?words_equal(b,y[i-1],a,x[j-1]):words_equal(a,x[i-1],b,y[j-1]);
            size_t deletion=previous[j]+1,insertion=current[j-1]+1;
            size_t substitution=previous[j-1]+(equal?0u:1u);
            size_t best=deletion<insertion?deletion:insertion;
            current[j]=best<substitution?best:substitution;
        }
        size_t *tmp=previous;previous=current;current=tmp;
    }
    *distance=previous[nc];free(previous);free(current);return 1;
}
static int codepoints(const unsigned char *s,size_t n,uint32_t **out,size_t *count){
    size_t total=0;for(size_t i=0;i<n;){uint32_t cp;size_t w=read_codepoint(s+i,n-i,&cp);if(!w)return 0;i+=w;total++;}
    if(total>SIZE_MAX/sizeof(uint32_t))return 0;
    uint32_t *items=total?(uint32_t*)malloc(total*sizeof(*items)):NULL;if(total&&!items)return 0;
    size_t at=0;for(size_t i=0;i<n;){uint32_t cp;size_t w=read_codepoint(s+i,n-i,&cp);items[at++]=cp;i+=w;}
    *out=items;*count=total;return 1;
}
static int sequence_distance(const uint32_t *a,size_t na,const uint32_t *b,size_t nb,size_t *distance){
    if(nb>na){const uint32_t *t=a;a=b;b=t;size_t n=na;na=nb;nb=n;}
    if(nb==SIZE_MAX||nb+1>SIZE_MAX/sizeof(size_t))return 0;
    size_t *prev=(size_t*)malloc((nb+1)*sizeof(size_t)),*curr=(size_t*)malloc((nb+1)*sizeof(size_t));
    if(!prev||!curr){free(prev);free(curr);return 0;}
    for(size_t j=0;j<=nb;j++)prev[j]=j;
    for(size_t i=1;i<=na;i++){curr[0]=i;for(size_t j=1;j<=nb;j++){
        size_t d=prev[j]+1,in=curr[j-1]+1,s=prev[j-1]+(a[i-1]==b[j-1]?0u:1u);
        curr[j]=d<in?(d<s?d:s):(in<s?in:s);}
        size_t *t=prev;prev=curr;curr=t;}
    *distance=prev[nb];free(prev);free(curr);return 1;
}
int ama10_text_error_rates(const unsigned char *reference,size_t reference_bytes,
                           const unsigned char *hypothesis,size_t hypothesis_bytes,
                           double *wer,double *cer,uint64_t *word_errors,
                           uint64_t *reference_words,uint64_t *character_errors,
                           uint64_t *reference_characters){
    if((!reference&&reference_bytes)||(!hypothesis&&hypothesis_bytes)||!wer||!cer)return 0;
    unsigned char *r=NULL,*h=NULL;size_t rn=0,hn=0;TextSpan *rw=NULL,*hw=NULL;
    size_t nrw=0,nhw=0;uint32_t *rc=NULL,*hc=NULL;size_t nrc=0,nhc=0,wd=0,cd=0;
    int ok=normalize_text(reference,reference_bytes,&r,&rn)&&
        normalize_text(hypothesis,hypothesis_bytes,&h,&hn)&&
        split_words(r,rn,&rw,&nrw)&&split_words(h,hn,&hw,&nhw)&&
        codepoints(r,rn,&rc,&nrc)&&codepoints(h,hn,&hc,&nhc)&&
        word_distance(r,rw,nrw,h,hw,nhw,&wd)&&sequence_distance(rc,nrc,hc,nhc,&cd);
    if(ok){*wer=nrw?(double)wd/(double)nrw:(nhw?1.0:0.0);*cer=nrc?(double)cd/(double)nrc:(nhc?1.0:0.0);
        if(word_errors)*word_errors=(uint64_t)wd;
        if(reference_words)*reference_words=(uint64_t)nrw;
        if(character_errors)*character_errors=(uint64_t)cd;
        if(reference_characters)*reference_characters=(uint64_t)nrc;}
    free(r);free(h);free(rw);free(hw);free(rc);free(hc);return ok;
}
static void count_script(AMA10GenerationMetrics *m,uint32_t cp){
    m->codepoints++;
    if((cp>=0x0041&&cp<=0x024f)||(cp>=0x1e00&&cp<=0x1eff))m->latin++;
    else if((cp>=0x3400&&cp<=0x4dbf)||(cp>=0x4e00&&cp<=0x9fff)||
            (cp>=0x20000&&cp<=0x2fa1f))m->cjk++;
    else if((cp>=0x3040&&cp<=0x30ff)||(cp>=0x31f0&&cp<=0x31ff))m->kana++;
    else if((cp>=0xac00&&cp<=0xd7af)||(cp>=0x1100&&cp<=0x11ff))m->hangul++;
    else if((cp>=0x0400&&cp<=0x052f)||(cp>=0x2de0&&cp<=0x2dff))m->cyrillic++;
    else if((cp>=0x0600&&cp<=0x08ff)||(cp>=0xfb50&&cp<=0xfdff)||(cp>=0xfe70&&cp<=0xfeff))m->arabic++;
    else if(cp>=0x0900&&cp<=0x097f)m->devanagari++;
    else if(cp>=0x0980&&cp<=0x09ff)m->bengali++;
    else m->other++;
}
int ama10_generation_metrics(const AMA10Tokenizer *t,const uint32_t *ids,size_t n,
                             const unsigned char *bytes,size_t byte_count,
                             AMA10GenerationMetrics *m){
    if(!t||!t->tokens||(!ids&&n)||(!bytes&&byte_count)||!m)return 0;
    memset(m,0,sizeof(*m));
    unsigned char *seen=(unsigned char*)calloc(t->vocabulary_size,1);
    if(!seen)return 0;
    size_t unique=0;
    for(size_t i=0;i<n;i++){
        if(ids[i]>=t->vocabulary_size){free(seen);return 0;}
        if(!seen[ids[i]]){seen[ids[i]]=1;unique++;}
    }
    free(seen);m->distinct_token_ratio=n?(double)unique/(double)n:0.0;
    size_t grams=n>=4?n-3:0,capacity=8;
    if(grams>SIZE_MAX/2)return 0;
    while(capacity<grams*2){if(capacity>SIZE_MAX/2)return 0;capacity*=2;}
    NgramSlot *table=(NgramSlot*)calloc(capacity,sizeof(*table));
    if(!table)return 0;
    size_t repeated=0;
    for(size_t i=0;i<grams;i++){
        uint64_t h=mix_ngram(ids+i);size_t slot=(size_t)h&(capacity-1);
        while(table[slot].used&&table[slot].hash!=h)slot=(slot+1)&(capacity-1);
        if(table[slot].used)repeated++;
        else{table[slot].used=1;table[slot].hash=h;}
    }
    free(table);m->repeated_fourgram_ratio=grams?(double)repeated/(double)grams:0.0;
    for(size_t offset=0;offset<byte_count;){
        uint32_t cp=0;size_t width=read_codepoint(bytes+offset,byte_count-offset,&cp);
        if(width){count_script(m,cp);offset+=width;}
        else{m->invalid_utf8_bytes++;offset++;}
    }
    return 1;
}
