#include "ama10.h"
#include "checkpoint.h"
#include "dataset.h"
#include "tokenizer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int open_data(AMA10TokenDataset *d,const char *path,char *cache,size_t cap,const AMA10Tokenizer *tok){
    uint64_t hash=0,bytes=0;
    if(!ama10_file_fingerprint(path,&hash,&bytes))return 0;
    int n=snprintf(cache,cap,"%s.ama10tok",path);
    return n>0&&(size_t)n<cap&&ama10_token_dataset_build_or_open(d,path,cache,tok,hash,bytes);
}
static int teacher_probs(const AMA10Model *m,const uint32_t *in,size_t n,float *probs){
    AMA10State state={0};float *logits=(float*)malloc(m->vocabulary*sizeof(float));
    if(!logits||!ama10_state_init(&state,m->width)){free(logits);return 0;}
    int ok=1;size_t v=m->vocabulary;
    for(size_t t=0;t<n&&ok;t++){
        if(!ama10_step(m,&state,in[t],logits,2,2.0f)){ok=0;break;}
        float mx=logits[0];for(size_t k=1;k<v;k++)if(logits[k]>mx)mx=logits[k];
        double sum=0;for(size_t k=0;k<v;k++){double x=exp((double)logits[k]-mx);probs[t*v+k]=(float)x;sum+=x;}
        if(!(sum>0)||!isfinite(sum)){ok=0;break;}
        for(size_t k=0;k<v;k++)probs[t*v+k]=(float)(probs[t*v+k]/sum);
    }
    free(logits);ama10_state_free(&state);return ok;
}
static int evaluate(const AMA10Model *teacher,const AMA10Model *student,AMA10TokenDataset *data,double *kl,double *nll){
    size_t v=teacher->vocabulary,n=128;uint32_t *window=(uint32_t*)malloc((n+1)*sizeof(uint32_t));
    uint32_t *in=(uint32_t*)malloc(n*sizeof(uint32_t));float *p=(float*)malloc(n*v*sizeof(float));float *logits=(float*)malloc(v*sizeof(float));
    if(!window||!in||!p||!logits){free(window);free(in);free(p);free(logits);return 0;}
    double sk=0,sn=0;uint64_t count=0;int ok=1;
    for(uint64_t cursor=0;cursor+1<data->token_count;cursor+=n){size_t len=(size_t)((data->token_count-1-cursor)>n?n:(data->token_count-1-cursor));
        if(!len||!ama10_token_dataset_read(data,cursor,len+1,window)){ok=0;break;}
        memcpy(in,window,len*sizeof(uint32_t));if(!teacher_probs(teacher,in,len,p)){ok=0;break;}
        AMA10State ss={0};if(!ama10_state_init(&ss,student->width)){ok=0;break;}
        for(size_t t=0;t<len;t++){
            if(!ama10_step(student,&ss,in[t],logits,2,2.0f)){ok=0;break;}
            float mx=logits[0];for(size_t k=1;k<v;k++)if(logits[k]>mx)mx=logits[k];double z=0;
            for(size_t k=0;k<v;k++)z+=exp((double)logits[k]-mx);
            double logz=(double)mx+log(z);
            for(size_t k=0;k<v;k++)if(p[t*v+k]>0)sk+=(double)p[t*v+k]*(log(p[t*v+k])-((double)logits[k]-logz));
            sn+=logz-logits[window[t+1]];count++;
        }
        ama10_state_free(&ss);if(!ok)break;
    }
    free(window);free(in);free(p);free(logits);if(!ok||!count)return 0;*kl=sk/count;*nll=sn/count;return isfinite(*kl)&&isfinite(*nll);
}
int main(int argc,char **argv){
    if(argc<6||argc>8){fprintf(stderr,"usage: %s <teacher.ama> <train.txt> <validation.txt> <updates> <output.ama> [student-width=16] [learning-rate=0.0001]\n",argv[0]);return 2;}
    char *end=NULL;unsigned long long updates=strtoull(argv[4],&end,10);if(!end||*end||!updates)return 2;
    size_t width=16;if(argc>=7){unsigned long x=strtoul(argv[6],&end,10);if(!end||*end||x<2||x>4096)return 2;width=(size_t)x;}
    float lr=0.0001f;if(argc>=8){lr=strtof(argv[7],&end);if(!end||*end||!isfinite(lr)||lr<=0||lr>0.1f)return 2;}
    AMA10Model teacher={0},student={0};AMA10Tokenizer tok={0};AMA10Optimizer teacher_opt={0},opt={0};AMA10State teacher_state={0},state={0};AMA10CheckpointProgress tp={0},progress={0};
    AMA10TokenDataset train={0},valid={0};char train_cache[4096],valid_cache[4096];int ok=0;uint64_t train_hash=0,train_bytes=0,valid_hash=0,valid_bytes=0;
    if(!ama10_checkpoint_load(argv[1],&teacher,&tok,&teacher_opt,&teacher_state,&tp)){fprintf(stderr,"KD: cannot load teacher checkpoint\n");goto done;}
    if(!open_data(&train,argv[2],train_cache,sizeof(train_cache),&tok)||!open_data(&valid,argv[3],valid_cache,sizeof(valid_cache),&tok)||train.token_count<66||valid.token_count<2){fprintf(stderr,"KD: dataset/tokenizer setup failed train=%llu validation=%llu\n",(unsigned long long)train.token_count,(unsigned long long)valid.token_count);goto done;}
    if(!ama10_file_fingerprint(argv[2],&train_hash,&train_bytes)||!ama10_file_fingerprint(argv[3],&valid_hash,&valid_bytes)){fprintf(stderr,"KD: source fingerprint failed\n");goto done;}
    if(!ama10_model_init(&student,teacher.vocabulary,width,width,UINT64_C(0x4b445f414d413130))||!ama10_optimizer_init(&opt,&student)||!ama10_state_init(&state,width)){fprintf(stderr,"KD: student initialization failed\n");goto done;}
    progress.train_hash=train_hash;progress.validation_hash=valid_hash;progress.tokenizer_sample_bytes=train_bytes;progress.target_vocabulary=(uint32_t)teacher.vocabulary;progress.sequence_length=64;progress.depth=2;progress.accumulation_steps=1;progress.learning_rate=lr;progress.weight_decay=0.01f;
    double kl0=0,nll0=0,kl1=0,nll1=0;if(!evaluate(&teacher,&student,&valid,&kl0,&nll0)){fprintf(stderr,"KD: initial evaluation failed\n");goto done;}
    size_t seq=64,v=teacher.vocabulary;uint32_t *window=(uint32_t*)malloc((seq+1)*sizeof(uint32_t));uint32_t *in=(uint32_t*)malloc(seq*sizeof(uint32_t));float *soft=(float*)malloc(seq*v*sizeof(float));
    if(!window||!in||!soft){free(window);free(in);free(soft);goto done;}
    uint64_t max_cursor=train.token_count-seq-1;
    for(uint64_t u=0;u<updates;u++){
        uint64_t cursor=(u*seq)%max_cursor;
        if(!ama10_token_dataset_read(&train,cursor,seq+1,window)){fprintf(stderr,"KD: training token read failed at update %llu\n",(unsigned long long)u);free(window);free(in);free(soft);goto done;}
        memcpy(in,window,seq*sizeof(uint32_t));if(!teacher_probs(&teacher,in,seq,soft)){fprintf(stderr,"KD: teacher forward failed at update %llu\n",(unsigned long long)u);free(window);free(in);free(soft);goto done;}
        float loss=0;ama10_state_reset(&state);
        if(!ama10_train_soft_multimodal_sequence(&student,&opt,&state,in,NULL,NULL,NULL,soft,seq,2,lr,0.01f,NULL,&loss)||!isfinite(loss)){fprintf(stderr,"KD: student update failed at update %llu\n",(unsigned long long)u);free(window);free(in);free(soft);goto done;}
        if((u+1)%100==0)printf("kd_update=%llu teacher_soft_ce=%.6f\n",u+1,(double)loss);
    }
    free(window);free(in);free(soft);if(!evaluate(&teacher,&student,&valid,&kl1,&nll1)){fprintf(stderr,"KD: final evaluation failed\n");goto done;}
    progress.completed_updates=opt.step;progress.cursor=updates*seq;
    if(!ama10_checkpoint_save(argv[5],&student,&tok,&opt,&state,&progress)){fprintf(stderr,"KD: checkpoint save failed\n");goto done;}
    {size_t n=strlen(argv[5]);char *side=(char*)malloc(n+5);if(!side)goto done;memcpy(side,argv[5],n);memcpy(side+n,".tok",5);ok=ama10_tokenizer_save(&tok,side);free(side);}
    if(ok){AMA10Model vm={0};AMA10Tokenizer vt={0};AMA10Optimizer vo={0};AMA10State vs={0};AMA10CheckpointProgress vp={0};
        ok=ama10_checkpoint_load(argv[5],&vm,&vt,&vo,&vs,&vp)&&vp.completed_updates==updates&&ama10_tokenizer_fingerprint(&vt)==ama10_tokenizer_fingerprint(&tok);
        ama10_state_free(&vs);ama10_optimizer_free(&vo);ama10_model_free(&vm);ama10_tokenizer_free(&vt);}
    if(ok)printf("kd_updates=%llu teacher_width=%zu student_width=%zu validation_tokens=%llu teacher_to_student_kl_before=%.6f after=%.6f student_gold_nll_before=%.6f after=%.6f checkpoint=%s\n",updates,teacher.width,width,(unsigned long long)(valid.token_count-1),kl0,kl1,nll0,nll1,argv[5]);
done:
    if(!ok)fprintf(stderr,"KD training failed\n");
    ama10_token_dataset_close(&train);ama10_token_dataset_close(&valid);
    ama10_state_free(&state);ama10_state_free(&teacher_state);
    ama10_optimizer_free(&opt);ama10_optimizer_free(&teacher_opt);
    ama10_model_free(&student);ama10_model_free(&teacher);ama10_tokenizer_free(&tok);
    return ok?0:1;
}
