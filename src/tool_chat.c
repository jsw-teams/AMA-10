#include "ama10.h"
#include "checkpoint.h"
#include "dataset.h"
#include "tools.h"
#include "tokenizer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct{unsigned left;unsigned char lo,hi;} Utf8;
static int utf8_byte(Utf8 *s,unsigned char c){if(!s->left){if(c<0x80)return 1;if(c>=0xc2&&c<=0xdf){s->left=1;s->lo=0x80;s->hi=0xbf;return 1;}if(c==0xe0){s->left=2;s->lo=0xa0;s->hi=0xbf;return 1;}if((c>=0xe1&&c<=0xec)||(c>=0xee&&c<=0xef)){s->left=2;s->lo=0x80;s->hi=0xbf;return 1;}if(c==0xed){s->left=2;s->lo=0x80;s->hi=0x9f;return 1;}if(c==0xf0){s->left=3;s->lo=0x90;s->hi=0xbf;return 1;}if(c>=0xf1&&c<=0xf3){s->left=3;s->lo=0x80;s->hi=0xbf;return 1;}if(c==0xf4){s->left=3;s->lo=0x80;s->hi=0x8f;return 1;}return 0;}if(c<s->lo||c>s->hi)return 0;s->left--;s->lo=0x80;s->hi=0xbf;return 1;}
static int token_ok(const AMA10Tokenizer *t,const Utf8 *prefix,size_t id,int complete){Utf8 s=*prefix;for(size_t i=0;i<t->tokens[id].length;i++)if(!utf8_byte(&s,t->tokens[id].bytes[i]))return 0;return !complete||!s.left;}
static int append(unsigned char **buf,size_t *len,size_t *cap,const void *data,size_t n){if(n>SIZE_MAX-*len-1)return 0;size_t need=*len+n+1;if(need>*cap){size_t c=*cap?*cap:1024;while(c<need){if(c>SIZE_MAX/2){c=need;break;}c*=2;}unsigned char *p=(unsigned char*)realloc(*buf,c);if(!p)return 0;*buf=p;*cap=c;}memcpy(*buf+*len,data,n);*len+=n;(*buf)[*len]=0;return *len<=2*1024*1024;}
static int read_prompt(const char *path,unsigned char **buf,size_t *len){FILE *f=fopen(path,"rb");if(!f)return 0;if(fseek(f,0,SEEK_END)!=0){fclose(f);return 0;}long n=ftell(f);if(n<=0||n>1024*1024||fseek(f,0,SEEK_SET)!=0){fclose(f);return 0;}*buf=(unsigned char*)malloc((size_t)n+1);if(!*buf){fclose(f);return 0;}int ok=fread(*buf,1,(size_t)n,f)==(size_t)n;if(fclose(f)!=0)ok=0;if(!ok){free(*buf);*buf=NULL;return 0;}(*buf)[n]=0;*len=(size_t)n;return 1;}
static int one_turn(const AMA10Model *m,const AMA10Tokenizer *t,const unsigned char *context,size_t context_len,size_t max_tokens,float penalty,unsigned char **generated,size_t *generated_len){
    static const char end_marker[]="<|end_tool_call|>";
    uint32_t *ids=NULL,*out_ids=(uint32_t*)malloc(max_tokens*sizeof(uint32_t));size_t *freq=(size_t*)calloc(m->vocabulary,sizeof(size_t));size_t ni=0;float *z=(float*)malloc(m->vocabulary*sizeof(float));AMA10State s={0};int ok=0;
    if(!out_ids||!freq||!z||!ama10_tokenizer_encode(t,context,context_len,&ids,&ni)||!ni||!ama10_state_init(&s,m->width))goto done;
    for(size_t i=0;i<ni;i++)if(!ama10_step(m,&s,ids[i],z,2,2.0f))goto done;
    Utf8 utf={0,0x80,0xbf};for(size_t i=0;i<max_tokens;i++){size_t best=m->vocabulary;float score=-INFINITY;for(size_t k=0;k<m->vocabulary;k++){float candidate=z[k]-(float)freq[k]*penalty;if(candidate>score&&token_ok(t,&utf,k,i+1==max_tokens)){best=k;score=candidate;}}if(best==m->vocabulary)goto done;out_ids[i]=(uint32_t)best;if(freq[best]<SIZE_MAX)freq[best]++;for(size_t b=0;b<t->tokens[best].length;b++)if(!utf8_byte(&utf,t->tokens[best].bytes[b]))goto done;if(!ama10_step(m,&s,(uint32_t)best,z,2,2.0f))goto done;
        if(i>0){unsigned char *tmp=NULL;size_t bytes=0;if(!ama10_tokenizer_decode(t,out_ids,i+1,&tmp,&bytes))goto done;int end=bytes>=sizeof(end_marker)-1&&!memcmp(tmp+bytes-(sizeof(end_marker)-1),end_marker,sizeof(end_marker)-1);free(tmp);if(end){if(!ama10_tokenizer_decode(t,out_ids,i+1,generated,generated_len))goto done;ok=1;break;}}
    }
    if(!ok){if(!ama10_tokenizer_decode(t,out_ids,max_tokens,generated,generated_len))goto done;ok=1;}
done:free(ids);free(out_ids);free(freq);free(z);ama10_state_free(&s);return ok;
}
int main(int argc,char **argv){static const char assistant_label[]="\nAssistant: ",tool_tail[]="<|end_tool_result|>\nAssistant:";if(argc<5||argc>8){fprintf(stderr,"usage: %s <checkpoint.ama> <tokenizer.tok> <prompt.txt> <transcript.txt> [max-turns=3] [tokens-per-turn=128] [repetition-penalty=0.5]\n",argv[0]);return 2;}size_t turns=3,tokens=128;float penalty=0.5f;char *end=NULL;if(argc>=6){unsigned long n=strtoul(argv[5],&end,10);if(!end||*end||n<1||n>8)return 2;turns=n;}if(argc>=7){unsigned long n=strtoul(argv[6],&end,10);if(!end||*end||n<1||n>512)return 2;tokens=n;}if(argc==8){penalty=strtof(argv[7],&end);if(!end||*end||!isfinite(penalty)||penalty<0||penalty>10)return 2;}
    AMA10Model m={0};AMA10Tokenizer t={0};AMA10Optimizer o={0};AMA10State s={0};AMA10CheckpointProgress p={0};unsigned char *context=NULL,*transcript=NULL;size_t context_len=0,transcript_len=0,context_cap=0,transcript_cap=0;int ok=0;
    if(!read_prompt(argv[3],&context,&context_len)||!append(&transcript,&transcript_len,&transcript_cap,context,context_len))goto done;
    int bundled=ama10_checkpoint_load(argv[1],&m,&t,&o,&s,&p);if(!bundled&&(!ama10_model_load(&m,argv[1])||!ama10_tokenizer_load(&t,argv[2])||m.vocabulary!=t.vocabulary_size||!ama10_state_init(&s,m.width)))goto done;
    if(bundled){AMA10Tokenizer side={0};if(!ama10_tokenizer_load(&side,argv[2])||ama10_tokenizer_fingerprint(&side)!=ama10_tokenizer_fingerprint(&t)){ama10_tokenizer_free(&side);goto done;}ama10_tokenizer_free(&side);ama10_optimizer_free(&o);}
    for(size_t turn=0;turn<turns;turn++){unsigned char *reply=NULL;size_t reply_len=0;if(!one_turn(&m,&t,context,context_len,tokens,penalty,&reply,&reply_len)){free(reply);goto done;}if(!append(&transcript,&transcript_len,&transcript_cap,assistant_label,sizeof(assistant_label)-1)||!append(&transcript,&transcript_len,&transcript_cap,reply,reply_len)||!append(&context,&context_len,&context_cap,assistant_label,sizeof(assistant_label)-1)||!append(&context,&context_len,&context_cap,reply,reply_len)){free(reply);goto done;}
        char *call=strstr((char*)reply,"<|tool_call|>");if(!call){free(reply);ok=1;break;}char name[64],result[8192];if(!ama10_tool_dispatch(reply,reply_len,name,sizeof(name),result,sizeof(result))){fprintf(stderr,"assistant emitted a malformed or unsupported tool call\n");free(reply);goto done;}free(reply);char prefix[256];int n=snprintf(prefix,sizeof(prefix),"\n<|tool_result|>%s\n[untrusted tool output]\n",name);if(n<0||(size_t)n>=sizeof(prefix)||!append(&transcript,&transcript_len,&transcript_cap,prefix,(size_t)n)||!append(&transcript,&transcript_len,&transcript_cap,result,strlen(result))||!append(&transcript,&transcript_len,&transcript_cap,tool_tail,sizeof(tool_tail)-1)||!append(&context,&context_len,&context_cap,prefix,(size_t)n)||!append(&context,&context_len,&context_cap,result,strlen(result))||!append(&context,&context_len,&context_cap,tool_tail,sizeof(tool_tail)-1))goto done;
        if(turn+1==turns){fprintf(stderr,"tool turn limit reached before a final answer\n");goto done;}
    }
    if(ok){FILE *f=fopen(argv[4],"wb");if(!f)ok=0;else{if(fwrite(transcript,1,transcript_len,f)!=transcript_len||fclose(f)!=0)ok=0;}}
    if(ok)printf("tool_chat_turns_limit=%zu transcript_bytes=%zu output=%s\n",turns,transcript_len,argv[4]);
done:if(!ok)fprintf(stderr,"tool chat failed\n");free(context);free(transcript);ama10_state_free(&s);ama10_optimizer_free(&o);ama10_model_free(&m);ama10_tokenizer_free(&t);return ok?0:1;}
