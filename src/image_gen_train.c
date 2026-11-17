#include "ama10.h"
#include "ama10_imggen.h"
#include "checkpoint.h"
#include "tokenizer.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct{char *path,*caption,lang[16];int valid;AMA10Image image;uint64_t image_hash;uint32_t *codes;size_t ncode;} Row;
static char *dup(const char*s){size_t n=strlen(s);char*p=(char*)malloc(n+1);if(p)memcpy(p,s,n+1);return p;}
static void rows_free(Row*r,size_t n){for(size_t i=0;i<n;i++){free(r[i].path);free(r[i].caption);free(r[i].codes);ama10_image_free(&r[i].image);}free(r);}
static int read_rows(const char*manifest,Row**out,size_t*count,size_t*nt,size_t*nv){FILE*f=fopen(manifest,"rb");if(!f)return 0;char line[16384];Row*r=NULL;size_t n=0,cap=0,tr=0,va=0;int ok=1;
 while(fgets(line,sizeof(line),f)){size_t z=strlen(line);if(z&&line[z-1]=='\n')line[--z]=0;if(z&&line[z-1]=='\r')line[--z]=0;if(!z)continue;
  char*a=strchr(line,'\t');if(!a){ok=0;break;}*a++=0;char*b=strchr(a,'\t');if(!b){ok=0;break;}*b++=0;char*c=strchr(b,'\t');if(c){*c++=0;if(strchr(c,'\t')){ok=0;break;}}else c="en";
  if(!*line||!*a||(strcmp(b,"train")&&strcmp(b,"validation"))||strlen(c)>=16){ok=0;break;}
  if(n==cap){size_t nc=cap?cap*2:16;Row*q=(Row*)realloc(r,nc*sizeof(*q));if(!q){ok=0;break;}r=q;cap=nc;}
  Row x={0};x.path=dup(line);x.caption=dup(a);x.valid=!strcmp(b,"validation");memcpy(x.lang,c,strlen(c)+1);if(!x.path||!x.caption){free(x.path);free(x.caption);ok=0;break;}r[n++]=x;if(x.valid)va++;else tr++;
 }
 if(ferror(f))ok=0;
 fclose(f);
 if(!ok||!tr||!va){rows_free(r,n);return 0;}
 *out=r;*count=n;*nt=tr;*nv=va;return 1;}
static int join(const char*m,const char*r,char**out){const char*a=strrchr(m,'/'),*b=strrchr(m,'\\'),*s=a;if(b&&(!s||b>s))s=b;size_t d=s?(size_t)(s-m+1):0,n=strlen(r);char*p=(char*)malloc(d+n+1);if(!p)return 0;if(d)memcpy(p,m,d);memcpy(p+d,r,n+1);*out=p;return 1;}
static uint64_t hbytes(uint64_t h,const void*p,size_t n){const unsigned char*b=(const unsigned char*)p;for(size_t i=0;i<n;i++){h^=b[i];h*=UINT64_C(1099511628211);}return h;}
static uint64_t split_hash(Row*r,size_t n,int val){uint64_t h=UINT64_C(14695981039346656037);static const char mark[]="AMA10_TEXT_TO_IMAGE_OVERLAP_VQ64_PATCH16_STRIDE12_COORD8_V5";h=hbytes(h,mark,sizeof(mark)-1);
 for(size_t i=0;i<n;i++)if(r[i].valid==val){h=hbytes(h,r[i].path,strlen(r[i].path));h=hbytes(h,r[i].caption,strlen(r[i].caption));h=hbytes(h,r[i].lang,strlen(r[i].lang));if(r[i].image.rgb){h=hbytes(h,&r[i].image.width,sizeof(r[i].image.width));h=hbytes(h,&r[i].image.height,sizeof(r[i].image.height));h=hbytes(h,r[i].image.rgb,r[i].image.width*r[i].image.height*3);}}
 return h;}
static int tokenize_rows(Row*r,size_t n,AMA10Tokenizer*t,int learn,size_t code_count,size_t*bytes){size_t total=0;for(size_t i=0;i<n;i++)if(!r[i].valid){size_t z=strlen(r[i].caption)+strlen(r[i].lang)+10;if(total>SIZE_MAX-z-1)return 0;total+=z+1;}
 unsigned char*corpus=(unsigned char*)malloc(total);if(!corpus)return 0;size_t at=0;for(size_t i=0;i<n;i++)if(!r[i].valid){int z=snprintf((char*)corpus+at,total-at,"<lang:%s> %s\n",r[i].lang,r[i].caption);if(z<0){free(corpus);return 0;}at+=(size_t)z;}
 int ok=1;if(learn){ok=ama10_tokenizer_train(t,corpus,total,256);for(size_t c=0;c<code_count&&ok;c++){char marker[40];int z=snprintf(marker,sizeof(marker),"<|image_code_%03zu|>",c);ok=z>0&&(size_t)z<sizeof(marker)&&ama10_tokenizer_add_special(t,(const unsigned char*)marker,(size_t)z,NULL);}for(size_t axis=0;axis<2&&ok;axis++)for(size_t pos=0;pos<8&&ok;pos++){char marker[32];int z=snprintf(marker,sizeof(marker),"<|image_%c_%02zu|>",axis?'y':'x',pos);ok=z>0&&(size_t)z<sizeof(marker)&&ama10_tokenizer_add_special(t,(const unsigned char*)marker,(size_t)z,NULL);}static const unsigned char eos[]="<|image_eos|>";if(ok)ok=ama10_tokenizer_add_special(t,eos,sizeof(eos)-1,NULL);}free(corpus);if(!ok||t->vocabulary_size!=273+code_count)return 0;*bytes=total;return 1;}
static int prompt_seq(const Row*r,const AMA10Tokenizer*t,uint32_t**out,size_t*n){size_t len=strlen(r->lang)+strlen(r->caption)+12;unsigned char*b=(unsigned char*)malloc(len);if(!b)return 0;int z=snprintf((char*)b,len,"<lang:%s> %s\n",r->lang,r->caption);if(z<0){free(b);return 0;}size_t sz=(size_t)z;b[sz++]=0;int ok=ama10_tokenizer_encode(t,b,sz,out,n);free(b);return ok;}
static uint32_t coord_token(size_t axis,size_t value,size_t codes){size_t bins=8,base=256+codes+(axis?bins:0);return (uint32_t)(base+value);}
static size_t normalized_coord(size_t position,size_t extent){return extent<2?0:(size_t)llround(7.0*(double)position/(double)(extent-1));}
static int sequence(const Row*r,const AMA10Tokenizer*t,const AMA10ImageCodebook*b,uint32_t**out,size_t*n){uint32_t*p=NULL;size_t pn=0;if(!prompt_seq(r,t,&p,&pn))return 0;size_t total;if(r->ncode>(SIZE_MAX-pn-1)/3){free(p);return 0;}total=pn+3*r->ncode+1;if(total>SIZE_MAX/sizeof(uint32_t)){free(p);return 0;}uint32_t*s=(uint32_t*)malloc(total*sizeof(uint32_t));if(!s){free(p);return 0;}memcpy(s,p,pn*sizeof(uint32_t));free(p);size_t dx=r->image.width>b->patch_width?r->image.width-b->patch_width:0,dy=r->image.height>b->patch_height?r->image.height-b->patch_height:0;size_t gw=dx/b->stride_width+(dx%b->stride_width!=0)+1,gh=dy/b->stride_height+(dy%b->stride_height!=0)+1;if(gw>SIZE_MAX/gh||gw*gh!=r->ncode){free(s);return 0;}for(size_t i=0;i<r->ncode;i++){if(!r->codes[i]||r->codes[i]>b->code_count){free(s);return 0;}size_t x=i%gw,y=i/gw,at=pn+3*i;s[at]=coord_token(1,normalized_coord(y,gh),b->code_count);s[at+1]=coord_token(0,normalized_coord(x,gw),b->code_count);s[at+2]=256+r->codes[i]-1;}s[total-1]=(uint32_t)(t->vocabulary_size-1);*out=s;*n=total;return 1;}
static int prep(Row*r,size_t n,const char*manifest,AMA10Tokenizer*t,int learn,size_t code_count,size_t*bytes){if(!tokenize_rows(r,n,t,learn,code_count,bytes))return 0;for(size_t i=0;i<n;i++){char*p=NULL;if(!join(manifest,r[i].path,&p))return 0;int ok=ama10_image_load(&r[i].image,p);free(p);if(!ok)return 0;r[i].image_hash=UINT64_C(14695981039346656037);r[i].image_hash=hbytes(r[i].image_hash,&r[i].image.width,sizeof(r[i].image.width));r[i].image_hash=hbytes(r[i].image_hash,&r[i].image.height,sizeof(r[i].image.height));r[i].image_hash=hbytes(r[i].image_hash,r[i].image.rgb,r[i].image.width*r[i].image.height*3);}return 1;}
static int first_training_image(const Row*r,size_t index){for(size_t i=0;i<index;i++)if(!r[i].valid&&r[i].image_hash==r[index].image_hash&&r[i].image.width==r[index].image.width&&r[i].image.height==r[index].image.height&&memcmp(r[i].image.rgb,r[index].image.rgb,r[i].image.width*r[i].image.height*3)==0)return 0;return 1;}
static int encode_all(Row*r,size_t n,const AMA10ImageCodebook*b){for(size_t i=0;i<n;i++)if(!ama10_image_encode_codes(&r[i].image,b,&r[i].codes,&r[i].ncode)||!r[i].ncode)return 0;return 1;}
static int eval(const AMA10Model*m,AMA10Tokenizer*t,const AMA10ImageCodebook*b,Row*r,size_t n,double*loss,uint64_t*tokens){double sum=0;uint64_t nt=0;for(size_t i=0;i<n;i++)if(r[i].valid){uint32_t*s=NULL;size_t z=0;double q;if(!sequence(&r[i],t,b,&s,&z)||!ama10_eval_loss(m,s,s+1,z-1,2,&q)){free(s);return 0;}free(s);sum+=q*(z-1);nt+=(uint64_t)(z-1);}if(!nt)return 0;*loss=sum/nt;if(tokens)*tokens=nt;return isfinite(*loss);}
static double ce_range(const float*z,size_t start,size_t n,uint32_t y){float mx=z[start];for(size_t i=1;i<n;i++)if(z[start+i]>mx)mx=z[start+i];double sum=0;for(size_t i=0;i<n;i++)sum+=exp((double)z[start+i]-mx);return (double)mx+log(sum)-z[start+y];}
static int eval_visual(const AMA10Model*m,AMA10Tokenizer*t,const AMA10ImageCodebook*b,AMA10State*state,Row*r,size_t n,double*loss){float*logits=(float*)malloc(m->vocabulary*sizeof(float));if(!logits)return 0;double sum=0;uint64_t count=0;
 for(size_t i=0;i<n;i++)if(r[i].valid){uint32_t*p=NULL;size_t pn=0;if(!prompt_seq(&r[i],t,&p,&pn)){free(logits);return 0;}ama10_state_reset(state);int ok=1;for(size_t j=0;j<pn;j++)if(!ama10_step(m,state,p[j],logits,2,2.0f)){ok=0;break;}free(p);if(!ok){free(logits);return 0;}
  size_t dx=r[i].image.width>b->patch_width?r[i].image.width-b->patch_width:0,dy=r[i].image.height>b->patch_height?r[i].image.height-b->patch_height:0,gw=dx/b->stride_width+(dx%b->stride_width!=0)+1,gh=dy/b->stride_height+(dy%b->stride_height!=0)+1;
  for(size_t j=0;j<r[i].ncode;j++){size_t x=j%gw,y=j/gw;uint32_t yt=coord_token(1,normalized_coord(y,gh),b->code_count),xt=coord_token(0,normalized_coord(x,gw),b->code_count),image_token=256+r[i].codes[j]-1;if(image_token>=t->vocabulary_size-17||!ama10_step(m,state,yt,logits,2,2.0f)||!ama10_step(m,state,xt,logits,2,2.0f)){free(logits);return 0;}sum+=ce_range(logits,256,m->vocabulary-273,r[i].codes[j]-1);count++;if(!ama10_step(m,state,image_token,logits,2,2.0f)){free(logits);return 0;}}}
 free(logits);if(!count)return 0;*loss=sum/(double)count;return isfinite(*loss);}
static char*suffix(const char*p,const char*s){size_t a=strlen(p),b=strlen(s);char*q=(char*)malloc(a+b+1);if(q){memcpy(q,p,a);memcpy(q+a,s,b+1);}return q;}
static uint64_t next_random(uint64_t*x){*x^=*x>>12;*x^=*x<<25;*x^=*x>>27;return *x*UINT64_C(2685821657736338717);}
static void shuffle(size_t*a,size_t n,uint64_t seed,uint64_t epoch){uint64_t x=seed^(epoch*UINT64_C(0x9e3779b97f4a7c15));if(!x)x=1;for(size_t i=n;i>1;i--){size_t j=(size_t)(next_random(&x)%i);size_t t=a[i-1];a[i-1]=a[j];a[j]=t;}}
static int save_all(const char*p,AMA10Model*m,AMA10Tokenizer*t,AMA10Optimizer*o,AMA10State*s,AMA10CheckpointProgress*g,AMA10ImageCodebook*b){char*tp=suffix(p,".tok"),*vp=suffix(p,".vq");if(!tp||!vp){free(tp);free(vp);return 0;}int ok=ama10_checkpoint_save(p,m,t,o,s,g)&&ama10_tokenizer_save(t,tp)&&ama10_image_codebook_save(b,vp);free(tp);free(vp);return ok;}
int main(int argc,char**argv){if(argc<4||argc>6){fprintf(stderr,"usage: %s <manifest.tsv> <updates-total> <checkpoint.ama> [codebook-size=64] [--resume]\n",argv[0]);return 2;}char*e=NULL;errno=0;unsigned long long requested=strtoull(argv[2],&e,10);if(errno||e==argv[2]||*e||!requested)return 2;uint32_t code_count=64;int resume=0;if(argc>=5){if(!strcmp(argv[4],"--resume"))resume=1;else{errno=0;unsigned long parsed=strtoul(argv[4],&e,10);if(errno||e==argv[4]||*e||parsed<2||parsed>254)return 2;code_count=(uint32_t)parsed;}}if(argc==6){if(strcmp(argv[5],"--resume")||resume)return 2;resume=1;}
 Row*r=NULL;size_t n=0,nt=0,nv=0,bytes=0;AMA10Tokenizer tok={0};AMA10Model m={0};AMA10Optimizer opt={0};AMA10State st={0};AMA10CheckpointProgress pr={0};AMA10ImageCodebook book={0};size_t*order=NULL;int ok=0;const char*phase="manifest";if(!read_rows(argv[1],&r,&n,&nt,&nv))goto done;
 phase="checkpoint";if(resume&&!ama10_checkpoint_load(argv[3],&m,&tok,&opt,&st,&pr))goto done;phase="images/tokenizer";if(!prep(r,n,argv[1],&tok,!resume,code_count,&bytes))goto done;uint64_t th=split_hash(r,n,0),vh=split_hash(r,n,1);uint32_t target_vocab=(uint32_t)tok.vocabulary_size;
 phase="codebook";char*vp=suffix(argv[3],".vq");if(!vp)goto done;
 if(resume){ok=ama10_image_codebook_load(&book,vp)&&book.train_hash==th&&book.code_count==code_count&&book.patch_width==16&&book.patch_height==16&&book.stride_width==12&&book.stride_height==12&&book.feature_width==768;}
 else{
  size_t patches=0;
  for(size_t i=0;i<n;i++)if(!r[i].valid&&first_training_image(r,i)){float*p=NULL;size_t np=0,d=0;if(!ama10_image_patchify_overlap(&r[i].image,16,16,12,12,&p,&np,&d)||d!=768||patches>SIZE_MAX-np){free(p);patches=0;break;}patches+=np;free(p);}
  if(patches>=code_count&&patches<=SIZE_MAX/768&&patches*768<=SIZE_MAX/sizeof(float)){
   float*x=(float*)malloc(patches*768*sizeof(float));
   if(x){size_t q=0;for(size_t i=0;i<n;i++)if(!r[i].valid&&first_training_image(r,i)){float*p=NULL;size_t np=0,d=0;if(!ama10_image_patchify_overlap(&r[i].image,16,16,12,12,&p,&np,&d)||d!=768||np>patches-q){free(p);free(x);x=NULL;break;}memcpy(x+q*768,p,np*d*sizeof(float));q+=np;free(p);}
    if(x&&q==patches){ok=ama10_image_codebook_fit(x,patches,768,code_count,16,16,th,&book);if(ok){book.stride_width=12;book.stride_height=12;}free(x);}else if(x)free(x);
   }
  }
 }
 free(vp);if(!ok||!encode_all(r,n,&book))goto done;
 size_t maxseq=0;for(size_t i=0;i<n;i++){uint32_t*p=NULL;size_t z=0;if(!prompt_seq(&r[i],&tok,&p,&z))goto done;free(p);if(r[i].ncode>(SIZE_MAX-z-1)/3)goto done;if(maxseq<z+3*r[i].ncode+1)maxseq=z+3*r[i].ncode+1;}if(maxseq>UINT32_MAX)goto done;
 phase="model";if(!resume){if(!ama10_model_init(&m,tok.vocabulary_size,32,32,UINT64_C(20261118))||!ama10_optimizer_init(&opt,&m)||!ama10_state_init(&st,32))goto done;}if(m.vocabulary!=tok.vocabulary_size||m.width!=32||(resume&&!opt.step))goto done;
 if(resume){if(pr.train_hash!=th||pr.validation_hash!=vh||pr.tokenizer_sample_bytes!=bytes||pr.sequence_length!=maxseq||pr.target_vocabulary!=target_vocab||pr.depth!=2||pr.accumulation_steps!=1||pr.optimizer_algorithm!=opt.algorithm||pr.learning_rate!=0.0001f||pr.learning_rate_decay!=0.01f||pr.weight_decay!=0.01f||pr.cursor!=pr.completed_updates%nt||requested<=pr.completed_updates)goto done;}
 else{pr.train_hash=th;pr.validation_hash=vh;pr.tokenizer_sample_bytes=bytes;pr.sequence_length=(uint32_t)maxseq;pr.depth=2;pr.accumulation_steps=1;pr.target_vocabulary=target_vocab;pr.optimizer_algorithm=opt.algorithm;pr.learning_rate=0.0001f;pr.learning_rate_decay=0.01f;pr.weight_decay=0.01f;}
 double before=0,after=0,vcode_before=0,vcode_after=0;uint64_t valtokens=0;if(!eval(&m,&tok,&book,r,n,&before,&valtokens)||!eval_visual(&m,&tok,&book,&st,r,n,&vcode_before))goto done;order=(size_t*)malloc(nt*sizeof(size_t));if(!order)goto done;size_t q=0;for(size_t i=0;i<n;i++)if(!r[i].valid)order[q++]=i;
 uint64_t active_epoch=UINT64_MAX;for(uint64_t u=pr.completed_updates;u<(uint64_t)requested;u++){uint64_t epoch=u/nt;if(epoch!=active_epoch){shuffle(order,nt,th,epoch);active_epoch=epoch;}size_t i=order[u%nt];uint32_t*s=NULL;size_t z=0;float loss=0;float lr=ama10_learning_rate(0.0001f,u+1,0.01f);ama10_state_reset(&st);if(!sequence(&r[i],&tok,&book,&s,&z)||!ama10_train_sequence(&m,&opt,&st,s,s+1,z-1,2,lr,0.01f,&loss)||!isfinite(loss)){free(s);goto done;}free(s);if((u+1)%100==0){pr.completed_updates=opt.step;pr.cursor=opt.step%nt;if(!save_all(argv[3],&m,&tok,&opt,&st,&pr,&book))goto done;}}
 if(!eval(&m,&tok,&book,r,n,&after,NULL)||!eval_visual(&m,&tok,&book,&st,r,n,&vcode_after))goto done;
 pr.completed_updates=opt.step;pr.cursor=opt.step%nt;phase="save";if(!save_all(argv[3],&m,&tok,&opt,&st,&pr,&book))goto done;
 printf("image_pairs=%zu train=%zu validation=%zu visual_codes=%u patch_size=%ux%u variable_resolution=true updates=%llu validation_token_nll_before=%.6f after=%.6f validation_image_code_nll_before=%.6f after=%.6f checkpoint=%s\n",n,nt,nv,book.code_count,book.patch_width,book.patch_height,(unsigned long long)opt.step,before,after,vcode_before,vcode_after,argv[3]);ok=1;
done:if(!ok)fprintf(stderr,"image generation training failed during %s\n",phase);free(order);rows_free(r,n);ama10_image_codebook_free(&book);ama10_state_free(&st);ama10_optimizer_free(&opt);ama10_model_free(&m);ama10_tokenizer_free(&tok);return ok?0:1;}
