#include "ama10.h"
#include "checkpoint.h"
#include "dataset.h"
#include "lora.h"
#include "tokenizer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {char *p,*r;uint32_t *ids;size_t pn,rn;} Pair;
static void pairs_free(Pair *p,size_t n){for(size_t i=0;i<n;i++){free(p[i].p);free(p[i].r);free(p[i].ids);}free(p);}
static char *unescape(const char *s,size_t n){char *o=(char*)malloc(n+1);if(!o)return NULL;size_t z=0;for(size_t i=0;i<n;i++){if(s[i]!='\\'){o[z++]=s[i];continue;}if(++i==n){free(o);return NULL;}switch(s[i]){case 'n':o[z++]='\n';break;case 'r':o[z++]='\r';break;case 't':o[z++]='\t';break;case '\\':o[z++]='\\';break;default:free(o);return NULL;}}o[z]=0;return o;}
static int read_pairs(const char *path,Pair **out,size_t *n){FILE *f=fopen(path,"rb");if(!f)return 0;Pair *p=NULL;size_t count=0,cap=0;char *line=NULL;size_t lc=0;int ok=1;while(ok){size_t z=0;int c;while((c=fgetc(f))!=EOF&&c!='\n'){if(!c){ok=0;break;}if(z+1>=lc){size_t nc=lc?lc*2:512;char *q=(char*)realloc(line,nc);if(!q){ok=0;break;}line=q;lc=nc;}line[z++]=(char)c;}if(!ok||(c==EOF&&!z))break;if(z&&line[z-1]=='\r')z--;if(!z)continue;char *tab=memchr(line,'\t',z);if(!tab||tab==line||tab==line+z-1||memchr(tab+1,'\t',(size_t)(line+z-tab-1))){ok=0;break;}char *a=unescape(line,(size_t)(tab-line)),*b=unescape(tab+1,(size_t)(line+z-tab-1));if(!a||!b||!a[0]||!b[0]){free(a);free(b);ok=0;break;}if(count==cap){size_t nc=cap?cap*2:16;Pair *q=(Pair*)realloc(p,nc*sizeof(*p));if(!q){free(a);free(b);ok=0;break;}p=q;cap=nc;}p[count++]=(Pair){.p=a,.r=b};if(c==EOF)break;}free(line);if(ferror(f))ok=0;fclose(f);if(!ok||!count){pairs_free(p,count);return 0;}*out=p;*n=count;return 1;}
static int encode_pairs(Pair *p,size_t n,const AMA10Tokenizer *t){for(size_t i=0;i<n;i++){uint32_t *a=NULL,*b=NULL;size_t na=0,nb=0;if(!ama10_tokenizer_encode(t,(unsigned char*)p[i].p,strlen(p[i].p),&a,&na)||!ama10_tokenizer_encode(t,(unsigned char*)p[i].r,strlen(p[i].r),&b,&nb)||!na||!nb||na>SIZE_MAX-nb){free(a);free(b);return 0;}p[i].ids=(uint32_t*)malloc((na+nb)*sizeof(uint32_t));if(!p[i].ids){free(a);free(b);return 0;}memcpy(p[i].ids,a,na*sizeof(uint32_t));memcpy(p[i].ids+na,b,nb*sizeof(uint32_t));p[i].pn=na;p[i].rn=nb;free(a);free(b);}return 1;}
static double forward(const AMA10Model *m,const AMA10Tokenizer *tok,AMA10LoRA *l,const Pair *p,float *ga,float *gb,float *mA,float *vA,float *mB,float *vB,uint64_t *step,float lr){
    (void)tok;
    size_t n=p->pn+p->rn,v=m->vocabulary,r=l->rank,w=l->width;float *z=(float*)malloc(v*sizeof(float)),*h=(float*)malloc(w*sizeof(float)),*c=(float*)malloc(r*sizeof(float)),*dz=(float*)malloc(v*sizeof(float));AMA10State s={0};
    if(!z||!h||!c||!dz||!ama10_state_init(&s,w)){free(z);free(h);free(c);free(dz);return NAN;}double loss=0;uint64_t count=0;
    if(ga&&(!mA||!vA||!mB||!vB)){free(z);free(h);free(c);free(dz);ama10_state_free(&s);return NAN;}
    for(size_t t=0;t+1<n;t++){
        if(!ama10_step_hidden(m,&s,p->ids[t],z,h,2,2.0f)){loss=NAN;break;}ama10_lora_apply(l,h,z);
        if(t+1<p->pn)continue;
        uint32_t target=p->ids[t+1];float mx=z[0];for(size_t k=1;k<v;k++)if(z[k]>mx)mx=z[k];double sum=0;for(size_t k=0;k<v;k++)sum+=exp((double)z[k]-mx);double lz=(double)mx+log(sum);loss+=lz-z[target];count++;
        if(ga){for(size_t j=0;j<v;j++)dz[j]=(float)(exp((double)z[j]-lz)-(j==target?1.0:0.0));for(size_t q=0;q<r;q++){double aq=0;for(size_t j=0;j<w;j++)aq+=(double)l->a[q*w+j]*h[j];c[q]=(float)aq;}
            for(size_t k=0;k<v;k++)for(size_t q=0;q<r;q++)gb[k*r+q]+=l->scale*dz[k]*c[q];
            for(size_t q=0;q<r;q++){double d=0;for(size_t k=0;k<v;k++)d+=(double)dz[k]*l->b[k*r+q];for(size_t j=0;j<w;j++)ga[q*w+j]+=l->scale*(float)d*h[j];}
        }
    }
    if(ga&&count){(*step)++;double b1=1.0-pow(0.9,(double)*step),b2=1.0-pow(0.999,(double)*step);float *par[2]={l->a,l->b},*gr[2]={ga,gb},*m[2]={mA,mB},*vv[2]={vA,vB};size_t sz[2]={r*w,v*r};for(int q=0;q<2;q++)for(size_t i=0;i<sz[q];i++){float g=gr[q][i]/(float)count;m[q][i]=0.9f*m[q][i]+0.1f*g;vv[q][i]=0.999f*vv[q][i]+0.001f*g*g;par[q][i]-=lr*(m[q][i]/(float)b1)/(sqrtf(vv[q][i]/(float)b2)+1e-8f);}}
    free(z);free(h);free(c);free(dz);ama10_state_free(&s);return count?loss/(double)count:NAN;
}
static int score(const AMA10Model *m,const AMA10Tokenizer *t,AMA10LoRA *l,Pair *p,size_t n,double *loss){double sum=0;uint64_t count=0;for(size_t i=0;i<n;i++){double x=forward(m,t,l,&p[i],NULL,NULL,NULL,NULL,NULL,NULL,0,0);if(!isfinite(x))return 0;sum+=x*p[i].rn;count+=p[i].rn;}if(!count)return 0;*loss=sum/count;return 1;}
int main(int argc,char **argv){if(argc<6||argc>8){fprintf(stderr,"usage: %s <base-checkpoint.ama> <train.tsv> <validation.tsv> <updates> <adapter.lora> [rank=8] [learning-rate=0.0003]\n",argv[0]);return 2;}
    char *end=NULL;unsigned long long updates=strtoull(argv[4],&end,10);if(!end||*end||!updates)return 2;size_t rank=8;if(argc>=7){unsigned long q=strtoul(argv[6],&end,10);if(!end||*end||!q)return 2;rank=q;}float lr=3e-4f;if(argc>=8){lr=strtof(argv[7],&end);if(!end||*end||!isfinite(lr)||lr<=0)return 2;}
    AMA10Model m={0};AMA10Tokenizer t={0};AMA10Optimizer o={0};AMA10State s={0};AMA10CheckpointProgress cp={0};Pair *tr=NULL,*va=NULL;size_t nt=0,nv=0;AMA10LoRA l={0};float *ga=NULL,*gb=NULL,*mA=NULL,*vA=NULL,*mB=NULL,*vB=NULL;uint64_t base=0,bytes=0,step=0;int ok=0;
    if(!ama10_checkpoint_load(argv[1],&m,&t,&o,&s,&cp)){fprintf(stderr,"LoRA: base checkpoint load failed\n");goto done;}
    if(!ama10_file_fingerprint(argv[1],&base,&bytes)||!read_pairs(argv[2],&tr,&nt)||!read_pairs(argv[3],&va,&nv)||!encode_pairs(tr,nt,&t)||!encode_pairs(va,nv,&t)){fprintf(stderr,"LoRA: data or tokenizer setup failed\n");goto done;}
    if(!ama10_lora_init(&l,&m,rank,(float)rank/(float)m.width,base,ama10_tokenizer_fingerprint(&t),1234567)){fprintf(stderr,"LoRA: adapter initialization failed\n");goto done;}
    ga=(float*)calloc(rank*m.width,sizeof(float));gb=(float*)calloc(m.vocabulary*rank,sizeof(float));mA=(float*)calloc(rank*m.width,sizeof(float));vA=(float*)calloc(rank*m.width,sizeof(float));mB=(float*)calloc(m.vocabulary*rank,sizeof(float));vB=(float*)calloc(m.vocabulary*rank,sizeof(float));if(!ga||!gb||!mA||!vA||!mB||!vB)goto done;double before=0,after=0;if(!score(&m,&t,&l,va,nv,&before)){fprintf(stderr,"LoRA: initial evaluation failed\n");goto done;}
    for(uint64_t u=0;u<updates;u++){size_t ix=(size_t)(u%nt);memset(ga,0,rank*m.width*sizeof(float));memset(gb,0,m.vocabulary*rank*sizeof(float));double x=forward(&m,&t,&l,&tr[ix],ga,gb,mA,vA,mB,vB,&step,lr);if(!isfinite(x))goto done;if((u+1)%100==0)printf("lora_update=%llu response_nll=%.6f\n",(unsigned long long)(u+1),x);}
    if(!score(&m,&t,&l,va,nv,&after)){fprintf(stderr,"LoRA: final evaluation failed\n");goto done;}
    if(!ama10_lora_save(&l,argv[5])){fprintf(stderr,"LoRA: adapter save failed\n");goto done;}
    {AMA10LoRA v={0};ok=ama10_lora_load(&v,argv[5],&m,base,l.tokenizer_hash);ama10_lora_free(&v);}
    if(ok)printf("lora_pairs_train=%zu validation=%zu updates=%llu rank=%zu learning_rate=%.7g response_nll_before=%.6f after=%.6f adapter=%s\n",nt,nv,updates,rank,(double)lr,before,after,argv[5]);
done:if(!ok)fprintf(stderr,"LoRA training failed\n");free(ga);free(gb);free(mA);free(vA);free(mB);free(vB);pairs_free(tr,nt);pairs_free(va,nv);ama10_lora_free(&l);ama10_state_free(&s);ama10_optimizer_free(&o);ama10_model_free(&m);ama10_tokenizer_free(&t);return ok?0:1;}
