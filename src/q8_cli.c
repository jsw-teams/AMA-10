#include "ama10_q8.h"
#include "checkpoint.h"
#include "eval.h"
#include "tokenizer.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct{unsigned remaining;unsigned char low,high;} Utf8State;
static int utf8_feed(Utf8State*s,const unsigned char*b,size_t n){for(size_t i=0;i<n;i++){unsigned char c=b[i];if(!s->remaining){if(c<=0x7f)continue;if(c>=0xc2&&c<=0xdf){s->remaining=1;s->low=0x80;s->high=0xbf;}else if(c==0xe0){s->remaining=2;s->low=0xa0;s->high=0xbf;}else if((c>=0xe1&&c<=0xec)||(c>=0xee&&c<=0xef)){s->remaining=2;s->low=0x80;s->high=0xbf;}else if(c==0xed){s->remaining=2;s->low=0x80;s->high=0x9f;}else if(c==0xf0){s->remaining=3;s->low=0x90;s->high=0xbf;}else if(c>=0xf1&&c<=0xf3){s->remaining=3;s->low=0x80;s->high=0xbf;}else if(c==0xf4){s->remaining=3;s->low=0x80;s->high=0x8f;}else return 0;}else{if(c<s->low||c>s->high)return 0;s->remaining--;s->low=0x80;s->high=0xbf;}}return 1;}
static int utf8_valid(const unsigned char*b,size_t n){Utf8State s={0,0x80,0xbf};return utf8_feed(&s,b,n)&&s.remaining==0;}
static int candidate_ok(const AMA10Tokenizer*t,const Utf8State*p,size_t id,int complete){Utf8State s=*p;return id<t->vocabulary_size&&utf8_feed(&s,t->tokens[id].bytes,t->tokens[id].length)&&(!complete||s.remaining==0);}
static int read_all(const char*path,unsigned char**out,size_t*n,size_t cap){FILE*f=fopen(path,"rb");if(!f)return 0;if(fseek(f,0,SEEK_END)!=0){fclose(f);return 0;}long z=ftell(f);if(z<0||(uint64_t)z>cap||fseek(f,0,SEEK_SET)!=0){fclose(f);return 0;}unsigned char*p=(unsigned char*)malloc((size_t)z+1);if(!p){fclose(f);return 0;}int ok=fread(p,1,(size_t)z,f)==(size_t)z;if(fclose(f)!=0)ok=0;if(!ok){free(p);return 0;}*out=p;*n=(size_t)z;return 1;}
static char*suffix(const char*p,const char*s){size_t a=strlen(p),b=strlen(s);if(a>SIZE_MAX-b-1)return NULL;char*q=(char*)malloc(a+b+1);if(q){memcpy(q,p,a);memcpy(q+a,s,b+1);}return q;}
static uint64_t f32_bytes(const AMA10Model*m){uint64_t w=m->width,e=m->input_width,v=m->vocabulary;return 4*(w*e+3*w*w+e*v+v*w+2*w);}
static int quantize(const char*input,const char*output,const char*tokenizer_path){AMA10Model m={0};AMA10Tokenizer tok={0};AMA10Optimizer opt={0};AMA10State st={0};AMA10CheckpointProgress progress={0};AMA10Q8Model q={0};int bundled=ama10_checkpoint_load(input,&m,&tok,&opt,&st,&progress);if(!bundled){if(!ama10_model_load(&m,input))goto fail;if(tokenizer_path&&!ama10_tokenizer_load(&tok,tokenizer_path))goto fail;}
 if(m.vocabulary>UINT32_MAX||m.width>UINT32_MAX||m.input_width>UINT32_MAX||(tok.vocabulary_size&&tok.vocabulary_size!=m.vocabulary)||!ama10_q8_model_from_f32(&m,&q)||!ama10_q8_model_save(&q,output))goto fail;
 if(bundled||tokenizer_path){if(!tok.vocabulary_size||tok.vocabulary_size!=m.vocabulary)goto fail;char*tokout=suffix(output,".tok");if(!tokout)goto fail;int saved=ama10_tokenizer_save(&tok,tokout);free(tokout);if(!saved)goto fail;}
 printf("quantized=Q8_ROW_SYMMETRIC vocabulary=%zu width=%zu input_width=%zu f32_parameter_bytes=%llu q8_resident_bytes=%zu reduction=%.3f checkpoint_update=%llu output=%s\n",m.vocabulary,m.width,m.input_width,(unsigned long long)f32_bytes(&m),ama10_q8_model_memory_bytes(&q),1.0-(double)ama10_q8_model_memory_bytes(&q)/(double)f32_bytes(&m),(unsigned long long)progress.completed_updates,output);
 ama10_q8_model_free(&q);ama10_model_free(&m);ama10_tokenizer_free(&tok);ama10_optimizer_free(&opt);ama10_state_free(&st);return 0;
fail:fprintf(stderr,"Q8 conversion failed\n");ama10_q8_model_free(&q);ama10_model_free(&m);ama10_tokenizer_free(&tok);ama10_optimizer_free(&opt);ama10_state_free(&st);return 1;}
static double nll(const float*z,size_t n,uint32_t y){float mx=z[0];for(size_t i=1;i<n;i++)if(z[i]>mx)mx=z[i];double sum=0;for(size_t i=0;i<n;i++)sum+=exp((double)z[i]-mx);return (double)mx+log(sum)-(double)z[y];}
static int evaluate(const char*model_path,const char*tok_path,const char*text_path){AMA10Q8Model m={0};AMA10Tokenizer t={0};AMA10State s={0};unsigned char*text=NULL;size_t n=0;uint32_t*ids=NULL;size_t count=0;float*logits=NULL;int ok=0;
 if(!ama10_q8_model_load(&m,model_path)||!ama10_tokenizer_load(&t,tok_path)||t.vocabulary_size!=m.vocabulary||!ama10_q8_state_init(&s,m.width)||!read_all(text_path,&text,&n,67108864)||!ama10_tokenizer_encode(&t,text,n,&ids,&count)||count<2)goto done;
 logits=(float*)malloc(m.vocabulary*sizeof(float));if(!logits)goto done;double total=0;for(size_t i=0;i+1<count;i++){if(!ama10_q8_step(&m,&s,ids[i],logits,2,2.0f))goto done;total+=nll(logits,m.vocabulary,ids[i+1]);}
 printf("backend=q8-cpu tokens=%zu token_nll=%.8f q8_resident_bytes=%zu\n",count-1,total/(double)(count-1),ama10_q8_model_memory_bytes(&m));ok=1;
done:if(!ok)fprintf(stderr,"Q8 evaluation failed\n");free(text);free(ids);free(logits);ama10_state_free(&s);ama10_tokenizer_free(&t);ama10_q8_model_free(&m);return ok?0:1;}
static int generate(const char*model_path,const char*tok_path,const char*prompt_path,const char*outpath,size_t max_tokens,float repeat){AMA10Q8Model m={0};AMA10Tokenizer tok={0};AMA10State state={0};unsigned char*prompt=NULL,*bytes=NULL;size_t prompt_len=0,byte_len=0;uint32_t*prompt_ids=NULL,*output_ids=NULL;size_t prompt_count=0;float*logits=NULL;size_t*frequency=NULL;int ok=0;
 if(!ama10_q8_model_load(&m,model_path)||!ama10_tokenizer_load(&tok,tok_path)||tok.vocabulary_size!=m.vocabulary||!ama10_q8_state_init(&state,m.width)||!read_all(prompt_path,&prompt,&prompt_len,1048576)||!utf8_valid(prompt,prompt_len)||!ama10_tokenizer_encode(&tok,prompt,prompt_len,&prompt_ids,&prompt_count)||!prompt_count)goto done;
 logits=(float*)malloc(m.vocabulary*sizeof(float));output_ids=(uint32_t*)malloc((max_tokens?max_tokens:1)*sizeof(uint32_t));frequency=(size_t*)calloc(m.vocabulary,sizeof(size_t));if(!logits||!output_ids||!frequency)goto done;
 for(size_t i=0;i<prompt_count;i++)if(!ama10_q8_step(&m,&state,prompt_ids[i],logits,2,2.0f))goto done;
 Utf8State utf={0,0x80,0xbf};size_t count=0;for(;count<max_tokens;count++){size_t best=m.vocabulary;float best_score=-INFINITY;for(size_t j=0;j<m.vocabulary;j++){float score=logits[j]-(float)frequency[j]*repeat;if(score>best_score&&candidate_ok(&tok,&utf,j,count+1==max_tokens)){best=j;best_score=score;}}if(best==m.vocabulary)goto done;output_ids[count]=(uint32_t)best;if(frequency[best]<SIZE_MAX)frequency[best]++;if(!utf8_feed(&utf,tok.tokens[best].bytes,tok.tokens[best].length)||!ama10_q8_step(&m,&state,(uint32_t)best,logits,2,2.0f))goto done;}
 if(!ama10_tokenizer_decode(&tok,output_ids,count,&bytes,&byte_len)||!utf8_valid(bytes,byte_len))goto done;
 {FILE*f=fopen(outpath,"wb");if(!f)goto done;int wrote=fwrite(bytes,1,byte_len,f)==byte_len;if(fclose(f)!=0)wrote=0;if(!wrote)goto done;}
 {AMA10GenerationMetrics metrics={0};if(!ama10_generation_metrics(&tok,output_ids,count,bytes,byte_len,&metrics))goto done;printf("backend=q8-cpu prompt_tokens=%zu generated_tokens=%zu generated_bytes=%zu distinct_token_ratio=%.4f repeated_4gram_ratio=%.4f q8_resident_bytes=%zu output=%s\n",prompt_count,count,byte_len,metrics.distinct_token_ratio,metrics.repeated_fourgram_ratio,ama10_q8_model_memory_bytes(&m),outpath);}
 ok=1;
done:if(!ok)fprintf(stderr,"Q8 generation failed\n");free(prompt);free(bytes);free(prompt_ids);free(output_ids);free(logits);free(frequency);ama10_state_free(&state);ama10_tokenizer_free(&tok);ama10_q8_model_free(&m);return ok?0:1;}
int main(int argc,char**argv){if(argc>=2&&!strcmp(argv[1],"quantize")){if(argc<4||argc>5){fprintf(stderr,"usage: %s quantize <model.ama|checkpoint.amm> <output.q8> [tokenizer.tok]\n",argv[0]);return 2;}return quantize(argv[2],argv[3],argc==5?argv[4]:NULL);}
 if(argc>=2&&!strcmp(argv[1],"generate")){if(argc<6||argc>8){fprintf(stderr,"usage: %s generate <model.q8> <tokenizer.tok> <prompt.txt> <output.txt> [tokens=128] [repetition-penalty=0]\n",argv[0]);return 2;}char*end=NULL;unsigned long long n=128;float repeat=0;if(argc>=7){errno=0;n=strtoull(argv[6],&end,10);if(errno||end==argv[6]||*end||n<1||n>1000000)return 2;}if(argc==8){errno=0;repeat=strtof(argv[7],&end);if(errno||end==argv[7]||*end||!isfinite(repeat)||repeat<0||repeat>100)return 2;}if(n>SIZE_MAX/sizeof(uint32_t))return 2;return generate(argv[2],argv[3],argv[4],argv[5],(size_t)n,repeat);}
 if(argc==5&&!strcmp(argv[1],"evaluate"))return evaluate(argv[2],argv[3],argv[4]);
 fprintf(stderr,"usage: %s quantize|generate|evaluate ...\n",argv[0]);return 2;}
