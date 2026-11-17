#include "ama10.h"
#include "checkpoint.h"
#include "dataset.h"
#include "tokenizer.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char *prompt,*response;uint32_t *ids;size_t prompt_count,response_count; } Pair;
static char *decode_field(const char *s,size_t n){
    char *out=(char*)malloc(n+1);if(!out)return NULL;size_t w=0;
    for(size_t i=0;i<n;i++){
        if(s[i]!='\\'){out[w++]=s[i];continue;}
        if(++i>=n){free(out);return NULL;}
        switch(s[i]){case '\\':out[w++]='\\';break;case 't':out[w++]='\t';break;
            case 'n':out[w++]='\n';break;case 'r':out[w++]='\r';break;
            default:free(out);return NULL;}
    }
    out[w]='\0';return out;
}
static void free_pairs(Pair *p,size_t n){if(!p)return;for(size_t i=0;i<n;i++){free(p[i].prompt);free(p[i].response);free(p[i].ids);}free(p);}
static int read_pairs(const char *path,Pair **out,size_t *out_count){
    FILE *f=fopen(path,"rb");if(!f)return 0;Pair *pairs=NULL;size_t count=0,capacity=0;char *line=NULL;size_t linecap=0;int ok=1;
    while(1){
        size_t used=0;int ch;
        while((ch=fgetc(f))!=EOF&&ch!='\n'){
            if(ch==0){ok=0;break;}
            if(used+1>=linecap){size_t next=linecap?linecap*2:256;if(next<linecap){ok=0;break;}char *p=(char*)realloc(line,next);if(!p){ok=0;break;}line=p;linecap=next;}
            line[used++]=(char)ch;
        }
        if(!ok)break;
        if(ch==EOF&&!used)break;
        if(used&&line[used-1]=='\r')used--;
        if(used==0)continue;
        if(used+1>=linecap){char *p=(char*)realloc(line,used+1);if(!p){ok=0;break;}line=p;linecap=used+1;}line[used]='\0';
        char *sep=memchr(line,'\t',used);if(!sep||sep==line||sep==line+used-1||memchr(sep+1,'\t',(size_t)(line+used-sep-1))){ok=0;break;}
        size_t prompt_len=(size_t)(sep-line),response_len=(size_t)(line+used-sep-1);
        char *prompt=decode_field(line,prompt_len),*response=decode_field(sep+1,response_len);
        if(!prompt||!response||!prompt[0]||!response[0]){free(prompt);free(response);ok=0;break;}
        if(count==capacity){size_t next=capacity?capacity*2:16;if(next<capacity||next>SIZE_MAX/sizeof(*pairs)){ok=0;break;}Pair *p=(Pair*)realloc(pairs,next*sizeof(*pairs));if(!p){ok=0;break;}pairs=p;capacity=next;}
        pairs[count]=(Pair){0};pairs[count].prompt=prompt;pairs[count].response=response;count++;
        if(ch==EOF)break;
    }
    if(ferror(f))ok=0;
    free(line);fclose(f);
    if(!ok||!count){free_pairs(pairs,count);return 0;}*out=pairs;*out_count=count;return 1;
}
static int learn_or_encode(Pair *train,size_t nt,Pair *valid,size_t nv,AMA10Tokenizer *tok,int learn,size_t *sample_bytes){
    size_t total=0;
    for(size_t i=0;i<nt;i++){
        size_t a=strlen(train[i].prompt),b=strlen(train[i].response);
        if(a>SIZE_MAX-b-2||total>SIZE_MAX-a-b-2)return 0;
        total+=a+b+2;
    }
    if(learn){unsigned char *sample=(unsigned char*)malloc(total?total:1);if(!sample)return 0;size_t pos=0;
        for(size_t i=0;i<nt;i++){size_t a=strlen(train[i].prompt),b=strlen(train[i].response);memcpy(sample+pos,train[i].prompt,a);pos+=a;sample[pos++]='\n';memcpy(sample+pos,train[i].response,b);pos+=b;sample[pos++]='\n';}
        int ok=ama10_tokenizer_train(tok,sample,total,512);free(sample);if(!ok)return 0;
    }
    if(!tok->vocabulary_size)return 0;
    *sample_bytes=total;
    for(size_t pass=0;pass<2;pass++){Pair *pairs=pass?valid:train;size_t count=pass?nv:nt;
        for(size_t i=0;i<count;i++){
            uint32_t *p=NULL,*r=NULL;size_t pn=0,rn=0;
            if(!ama10_tokenizer_encode(tok,(const unsigned char*)pairs[i].prompt,strlen(pairs[i].prompt),&p,&pn)||
               !ama10_tokenizer_encode(tok,(const unsigned char*)pairs[i].response,strlen(pairs[i].response),&r,&rn)||!pn||!rn||pn>SIZE_MAX-rn){free(p);free(r);return 0;}
            if(pn+rn>SIZE_MAX/sizeof(uint32_t)){free(p);free(r);return 0;}
            pairs[i].ids=(uint32_t*)malloc((pn+rn)*sizeof(uint32_t));if(!pairs[i].ids){free(p);free(r);return 0;}
            memcpy(pairs[i].ids,p,pn*sizeof(uint32_t));memcpy(pairs[i].ids+pn,r,rn*sizeof(uint32_t));
            pairs[i].prompt_count=pn;pairs[i].response_count=rn;free(p);free(r);
        }
    }
    return 1;
}
static double token_nll(const float *logits,size_t v,uint32_t target){
    float m=logits[0];for(size_t k=1;k<v;k++)if(logits[k]>m)m=logits[k];double sum=0;
    for(size_t k=0;k<v;k++)sum+=exp((double)logits[k]-m);
    return (double)m+log(sum)-logits[target];
}
static int teacher_probabilities(const AMA10Model *teacher,const uint32_t *inputs,size_t length,float *probabilities){
    float *logits=(float*)malloc(teacher->vocabulary*sizeof(float));AMA10State state={0};
    if(!logits||!ama10_state_init(&state,teacher->width)){free(logits);return 0;}
    int ok=1;size_t v=teacher->vocabulary;
    for(size_t t=0;t<length&&ok;t++){
        if(!ama10_step(teacher,&state,inputs[t],logits,2,2.0f)){ok=0;break;}
        float maximum=logits[0];for(size_t k=1;k<v;k++)if(logits[k]>maximum)maximum=logits[k];
        double sum=0.0;for(size_t k=0;k<v;k++){double value=exp((double)logits[k]-(double)maximum);probabilities[t*v+k]=(float)value;sum+=value;}
        if(!(sum>0.0)||!isfinite(sum)){ok=0;break;}
        for(size_t k=0;k<v;k++)probabilities[t*v+k]=(float)((double)probabilities[t*v+k]/sum);
    }
    free(logits);ama10_state_free(&state);return ok;
}
static int build_arrays(const Pair *p,uint32_t **inputs,uint32_t **targets,unsigned char **mask,size_t *length){
    if(p->prompt_count>SIZE_MAX-p->response_count)return 0;
    size_t n=p->prompt_count+p->response_count;
    if(n<2||n-1>SIZE_MAX/sizeof(uint32_t))return 0;
    size_t l=n-1;
    uint32_t *in=(uint32_t*)malloc(l*sizeof(uint32_t)),*tar=(uint32_t*)malloc(l*sizeof(uint32_t));
    unsigned char *m=(unsigned char*)calloc(l,1);if(!in||!tar||!m){free(in);free(tar);free(m);return 0;}
    memcpy(in,p->ids,l*sizeof(uint32_t));memcpy(tar,p->ids+1,l*sizeof(uint32_t));
    for(size_t i=p->prompt_count-1;i<l;i++)m[i]=1;
    *inputs=in;*targets=tar;*mask=m;*length=l;return 1;
}
static int eval_pairs(const AMA10Model *model,const Pair *pairs,size_t count,double *mean,uint64_t *targets){
    float *logits=(float*)malloc(model->vocabulary*sizeof(float));if(!logits)return 0;double sum=0;uint64_t nt=0;
    for(size_t i=0;i<count;i++){
        uint32_t *in=NULL,*tar=NULL;unsigned char *mask=NULL;size_t len=0;AMA10State state={0};
        if(!build_arrays(&pairs[i],&in,&tar,&mask,&len)||!ama10_state_init(&state,model->width)){free(in);free(tar);free(mask);free(logits);return 0;}
        for(size_t t=0;t<len;t++){
            if(!ama10_step(model,&state,in[t],logits,2,2.0f)){free(in);free(tar);free(mask);ama10_state_free(&state);free(logits);return 0;}
            if(mask[t]){sum+=token_nll(logits,model->vocabulary,tar[t]);nt++;}
        }
        free(in);free(tar);free(mask);ama10_state_free(&state);
    }
    free(logits);if(!nt)return 0;*mean=sum/(double)nt;if(targets)*targets=nt;return isfinite(*mean);
}
static uint64_t order_rng_next(uint64_t *state){
    uint64_t x=*state;
    x^=x>>12;x^=x<<25;x^=x>>27;*state=x;
    return x*UINT64_C(2685821657736338717);
}
static void shuffle_order(size_t *order,size_t count,uint64_t *rng){
    for(size_t i=count;i>1;i--){
        size_t j=(size_t)(order_rng_next(rng)%(uint64_t)i);
        size_t swap=order[i-1];order[i-1]=order[j];order[j]=swap;
    }
}
int main(int argc,char **argv){
    if(argc<5||argc>12){fprintf(stderr,"usage: %s <train.tsv> <validation.tsv> <additional-updates> <output.ama> [learning-rate] [--init checkpoint.ama] [--preserve-from teacher.ama --preserve-weight 0..1]\n",argv[0]);return 2;}
    char *end=NULL;errno=0;unsigned long long requested=strtoull(argv[3],&end,10);
    if(errno||end==argv[3]||*end||!requested||requested>SIZE_MAX){fprintf(stderr,"invalid update count\n");return 2;}
    float sft_learning_rate=0.00003f;
    float preserve_weight=0.0f;int init=0,have_lr=0;const char *initial_checkpoint=NULL,*preserve_path=NULL;
    for(int i=5;i<argc;i++){
        if(strcmp(argv[i],"--init")==0){if(init||i+1>=argc)return 2;init=1;initial_checkpoint=argv[++i];}
        else if(strcmp(argv[i],"--preserve-from")==0){if(preserve_path||i+1>=argc)return 2;preserve_path=argv[++i];}
        else if(strcmp(argv[i],"--preserve-weight")==0){if(i+1>=argc)return 2;errno=0;preserve_weight=strtof(argv[++i],&end);if(errno||end==argv[i]||*end||!isfinite(preserve_weight)||preserve_weight<0.0f||preserve_weight>1.0f)return 2;}
        else if(!have_lr&&argv[i][0]!='-'){errno=0;sft_learning_rate=strtof(argv[i],&end);if(errno||end==argv[i]||*end||!isfinite(sft_learning_rate)||sft_learning_rate<=0.0f||sft_learning_rate>0.1f){fprintf(stderr,"learning rate must be finite and in (0, 0.1]\n");return 2;}have_lr=1;}
        else return 2;
    }
    if((preserve_weight>0.0f&&(!init||!preserve_path))||(preserve_weight==0.0f&&preserve_path))return 2;
    Pair *train=NULL,*valid=NULL;size_t *order=NULL;size_t nt=0,nv=0,sample_bytes=0,maxseq=0;AMA10Tokenizer tok={0},teacher_tok={0};AMA10Model model={0},teacher={0};AMA10Optimizer opt={0},teacher_opt={0};AMA10State state={0},teacher_state={0};AMA10CheckpointProgress progress={0},teacher_progress={0};
    int ok=0;
    if(!read_pairs(argv[1],&train,&nt)||!read_pairs(argv[2],&valid,&nv))goto done;
    if(init&&!ama10_checkpoint_load(initial_checkpoint,&model,&tok,&opt,&state,&progress))goto done;
    if(!learn_or_encode(train,nt,valid,nv,&tok,!init,&sample_bytes))goto done;
    if(!init){if(!ama10_model_init(&model,tok.vocabulary_size,32,32,UINT64_C(20261120))||
        !ama10_optimizer_init(&opt,&model)||!ama10_state_init(&state,model.width))goto done;}
    if(model.width<8||model.width>4096||model.input_width!=model.width||
       model.vocabulary!=tok.vocabulary_size||
       (init&&opt.step!=progress.completed_updates))goto done;
    if(preserve_weight>0.0f){
        if(!ama10_checkpoint_load(preserve_path,&teacher,&teacher_tok,&teacher_opt,&teacher_state,&teacher_progress)||
           teacher.width!=model.width||teacher.vocabulary!=model.vocabulary||
           ama10_tokenizer_fingerprint(&teacher_tok)!=ama10_tokenizer_fingerprint(&tok))goto done;
        ama10_tokenizer_free(&teacher_tok);ama10_optimizer_free(&teacher_opt);ama10_state_free(&teacher_state);
    }
    for(size_t i=0;i<nt;i++){size_t n=train[i].prompt_count+train[i].response_count-1;if(maxseq<n)maxseq=n;}
    for(size_t i=0;i<nv;i++){size_t n=valid[i].prompt_count+valid[i].response_count-1;if(maxseq<n)maxseq=n;}
    if(maxseq>UINT32_MAX)goto done;
    uint64_t train_hash=0,train_bytes=0,valid_hash=0,valid_bytes=0;
    if(!ama10_file_fingerprint(argv[1],&train_hash,&train_bytes)||!ama10_file_fingerprint(argv[2],&valid_hash,&valid_bytes))goto done;
    if(nt>SIZE_MAX/sizeof(*order))goto done;
    order=(size_t*)malloc(nt*sizeof(*order));if(!order)goto done;
    for(size_t i=0;i<nt;i++)order[i]=i;
    uint64_t order_seed=train_hash^UINT64_C(0x9E3779B97F4A7C15);
    if(!order_seed)order_seed=UINT64_C(0xD1B54A32D192ED03);
    uint64_t order_rng=order_seed;
    progress.train_hash=train_hash;progress.validation_hash=valid_hash;progress.tokenizer_sample_bytes=sample_bytes;
    progress.sequence_length=(uint32_t)maxseq;progress.depth=2;progress.accumulation_steps=1;
    progress.target_vocabulary=(uint32_t)tok.vocabulary_size;progress.optimizer_algorithm=opt.algorithm;
    progress.learning_rate=sft_learning_rate;progress.learning_rate_decay=0.01f;progress.weight_decay=0.01f;
    double before=0,after=0;uint64_t response_tokens=0;
    if(!eval_pairs(&model,valid,nv,&before,&response_tokens))goto done;
    uint64_t start=opt.step;
    if((uint64_t)requested>UINT64_MAX-start)goto done;
    uint64_t end_update=start+(uint64_t)requested;
    for(uint64_t update=start;update<end_update;update++){
        uint64_t local_update=update-start;
        size_t epoch_offset=(size_t)(local_update%(uint64_t)nt);
        if(epoch_offset==0)shuffle_order(order,nt,&order_rng);
        size_t index=order[epoch_offset];uint32_t *in=NULL,*tar=NULL;unsigned char *mask=NULL;size_t len=0;float loss=0;
        if(!build_arrays(&train[index],&in,&tar,&mask,&len)){free(in);free(tar);free(mask);goto done;}
        ama10_state_reset(&state);
        float lr=ama10_learning_rate(sft_learning_rate,update-start+1,0.01f);int trained=0;
        if(preserve_weight>0.0f){
            if(len>SIZE_MAX/model.vocabulary||len*model.vocabulary>SIZE_MAX/sizeof(float)){free(in);free(tar);free(mask);goto done;}
            float *soft=(float*)malloc(len*model.vocabulary*sizeof(float));
            if(!soft||!teacher_probabilities(&teacher,in,len,soft)){free(soft);free(in);free(tar);free(mask);goto done;}
            for(size_t t=0;t<len;t++){
                if(mask[t]){
                    for(size_t k=0;k<model.vocabulary;k++)soft[t*model.vocabulary+k]*=preserve_weight;
                    soft[t*model.vocabulary+tar[t]]+=1.0f-preserve_weight;
                }else{
                    /* Distill the initialized model on prompt positions so SFT
                       updates preserve its prompt and language distributions. */
                    mask[t]=1;
                }
            }
            trained=ama10_train_soft_multimodal_sequence(&model,&opt,&state,in,NULL,NULL,mask,soft,len,2,lr,0.01f,NULL,&loss);
            free(soft);
        }else{
            trained=ama10_train_multimodal_sequence(&model,&opt,&state,in,NULL,NULL,mask,tar,len,2,lr,0.01f,NULL,&loss);
        }
        free(in);free(tar);free(mask);if(!trained||!isfinite(loss))goto done;
        if((update+1)%100==0)printf("sft_update=%llu response_token_nll=%.5f\n",(unsigned long long)(update+1),(double)loss);
    }
    if(!eval_pairs(&model,valid,nv,&after,NULL))goto done;
    progress.completed_updates=opt.step;progress.cursor=progress.completed_updates%(uint64_t)nt;
    ok=ama10_checkpoint_save(argv[4],&model,&tok,&opt,&state,&progress);
    if(ok){size_t n=strlen(argv[4]);if(n>SIZE_MAX-5)ok=0;else{
        char *sidecar=(char*)malloc(n+5);if(!sidecar)ok=0;else{
            memcpy(sidecar,argv[4],n);memcpy(sidecar+n,".tok",5);
            ok=ama10_tokenizer_save(&tok,sidecar);free(sidecar);
        }
    }}
    if(ok){AMA10Model vm={0};AMA10Tokenizer vt={0};AMA10Optimizer vo={0};AMA10State vs={0};AMA10CheckpointProgress vp={0};double persisted=0;
        ok=ama10_checkpoint_load(argv[4],&vm,&vt,&vo,&vs,&vp)&&vp.completed_updates==progress.completed_updates&&
           vo.step==opt.step&&ama10_tokenizer_fingerprint(&vt)==ama10_tokenizer_fingerprint(&tok)&&
           eval_pairs(&vm,valid,nv,&persisted,NULL)&&fabs(persisted-after)<1e-6;
        ama10_state_free(&vs);ama10_optimizer_free(&vo);ama10_model_free(&vm);ama10_tokenizer_free(&vt);
    }
    if(ok)printf("sft_pairs_train=%zu validation=%zu response_tokens=%llu updates=%llu learning_rate=%.7g teacher_preservation_weight=%.4f order=deterministic_epoch_shuffle seed=%016llx response_token_nll_before=%.6f after=%.6f checkpoint=%s\n",
        nt,nv,(unsigned long long)response_tokens,(unsigned long long)opt.step,(double)sft_learning_rate,(double)preserve_weight,(unsigned long long)order_seed,before,after,argv[4]);
done:
    if(!ok)fprintf(stderr,"SFT training failed\n");
    free(order);free_pairs(train,nt);free_pairs(valid,nv);ama10_tokenizer_free(&tok);
    ama10_tokenizer_free(&teacher_tok);ama10_optimizer_free(&opt);ama10_optimizer_free(&teacher_opt);
    ama10_state_free(&state);ama10_state_free(&teacher_state);ama10_model_free(&model);ama10_model_free(&teacher);
    return ok?0:1;
}
