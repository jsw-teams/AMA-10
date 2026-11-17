#define _POSIX_C_SOURCE 200809L
#include "ama10.h"
#include "checkpoint.h"
#include "dataset.h"
#include "eval.h"
#include "lora.h"
#include "tokenizer.h"
#ifdef AMA10_ENABLE_OPENCL
#include "opencl_backend.h"
#endif
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <io.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/types.h>
#endif

static double native_wall_seconds(void){
#ifdef _WIN32
    LARGE_INTEGER now,frequency;
    if(!QueryPerformanceCounter(&now)||!QueryPerformanceFrequency(&frequency)||frequency.QuadPart<=0)return 0.0;
    return (double)now.QuadPart/(double)frequency.QuadPart;
#else
    struct timespec now;
    if(clock_gettime(CLOCK_MONOTONIC,&now)!=0)return 0.0;
    return (double)now.tv_sec+(double)now.tv_nsec*1.0e-9;
#endif
}

static int read_all(const char *path,unsigned char **data,size_t *size){
    FILE *f=fopen(path,"rb");if(!f){perror(path);return 0;}
    if(fseek(f,0,SEEK_END)!=0){fclose(f);return 0;}
    long n=ftell(f);if(n<=0||fseek(f,0,SEEK_SET)!=0){fclose(f);return 0;}
    unsigned char *p=(unsigned char*)malloc((size_t)n);
    if(!p){fclose(f);return 0;}
    if(fread(p,1,(size_t)n,f)!=(size_t)n){free(p);fclose(f);return 0;}
    fclose(f);*data=p;*size=(size_t)n;return 1;
}
typedef struct { unsigned remaining; unsigned char low,high; } Utf8State;
static int utf8_feed(Utf8State *s,const unsigned char *bytes,size_t length){
    for(size_t i=0;i<length;i++){
        unsigned char c=bytes[i];
        if(!s->remaining){
            if(c<=0x7f)continue;
            if(c>=0xc2&&c<=0xdf){s->remaining=1;s->low=0x80;s->high=0xbf;}
            else if(c==0xe0){s->remaining=2;s->low=0xa0;s->high=0xbf;}
            else if((c>=0xe1&&c<=0xec)||(c>=0xee&&c<=0xef)){
                s->remaining=2;s->low=0x80;s->high=0xbf;
            }else if(c==0xed){s->remaining=2;s->low=0x80;s->high=0x9f;}
            else if(c==0xf0){s->remaining=3;s->low=0x90;s->high=0xbf;}
            else if(c>=0xf1&&c<=0xf3){s->remaining=3;s->low=0x80;s->high=0xbf;}
            else if(c==0xf4){s->remaining=3;s->low=0x80;s->high=0x8f;}
            else return 0;
        }else{
            if(c<s->low||c>s->high)return 0;
            s->remaining--;s->low=0x80;s->high=0xbf;
        }
    }
    return 1;
}
static int utf8_valid(const unsigned char *bytes,size_t length){
    Utf8State state={0,0x80,0xbf};
    return utf8_feed(&state,bytes,length)&&state.remaining==0;
}
static int utf8_candidate_valid(const AMA10Tokenizer *tokenizer,const Utf8State *prefix,
                                size_t candidate,int require_complete){
    if(!prefix)return 0;
    if(candidate>=tokenizer->vocabulary_size||tokenizer->tokens[candidate].special)return 0;
    Utf8State state=*prefix;
    if(!utf8_feed(&state,tokenizer->tokens[candidate].bytes,tokenizer->tokens[candidate].length))return 0;
    return !require_complete||state.remaining==0;
}
static int seek_file64(FILE *f,uint64_t offset){
#ifdef _WIN32
    if(offset>INT64_MAX)return 0;
    return _fseeki64(f,(__int64)offset,SEEK_SET)==0;
#else
    if(offset>INT64_MAX)return 0;
    return fseeko(f,(off_t)offset,SEEK_SET)==0;
#endif
}
static int read_tokenizer_sample(const char *path,uint64_t file_bytes,uint64_t limit,
                                 unsigned char **data,size_t *size){
    if(!path||!data||!size||!limit||limit>SIZE_MAX)return 0;
    FILE *f=fopen(path,"rb");if(!f){perror(path);return 0;}
    size_t cap=(size_t)limit;if(cap==0)cap=1;
    unsigned char *p=(unsigned char*)malloc(cap);if(!p){fclose(f);return 0;}
    size_t n=0;int ok=1;
    if(file_bytes<=limit){
        n=fread(p,1,(size_t)file_bytes,f);
        if(n!=(size_t)file_bytes||ferror(f))ok=0;
    }else{
        size_t segments=16;
        if(limit<(uint64_t)segments*65)segments=(size_t)((limit+1)/65);
        if(segments<1)segments=1;
        size_t separators=segments-1;
        size_t window=(size_t)((limit-separators)/segments);
        if(window<1)ok=0;
        uint64_t span=file_bytes-(uint64_t)window;
        for(size_t i=0;i<segments&&ok;i++){
            uint64_t start=segments==1?span/2:
                (span/(segments-1))*i+((span%(segments-1))*i)/(segments-1);
            if(!seek_file64(f,start)||fread(p+n,1,window,f)!=window||ferror(f)){ok=0;break;}
            n+=window;
            if(i+1<segments)p[n++]='\n';
        }
    }
    if(fclose(f)!=0)ok=0;
    if(!ok){free(p);return 0;}
    *data=p;*size=n;return 1;
}
static int build_validation_prompt(const char *path,uint64_t file_bytes,
                                   const AMA10Tokenizer *tokenizer,uint32_t ids[64],
                                   size_t *token_count,uint64_t *byte_offset,
                                   unsigned char **prompt_bytes,size_t *prompt_length){
    if(!path||!tokenizer||!ids||!token_count||!byte_offset||!prompt_bytes||!prompt_length||file_bytes<2)return 0;
    *token_count=0;*prompt_bytes=NULL;*prompt_length=0;
    FILE *f=fopen(path,"rb");if(!f){perror(path);return 0;}
    uint64_t offset=file_bytes/2u;int ok=seek_file64(f,offset);
    if(ok&&offset){int c;
        while((c=fgetc(f))!=EOF&&c!='\n'){}
        if(c==EOF)ok=0;
    }
    if(ok){
        long long current;
#ifdef _WIN32
        current=_ftelli64(f);
#else
        current=(long long)ftello(f);
#endif
        if(current<0)ok=0;else offset=(uint64_t)current;
    }
    size_t cap=65536u;unsigned char *line=ok?(unsigned char*)malloc(cap):NULL;
    if(ok&&!line)ok=0;
    size_t length=0,begin=0;uint64_t line_offset=offset;
    while(ok){
        long long current;
#ifdef _WIN32
        current=_ftelli64(f);
#else
        current=(long long)ftello(f);
#endif
        if(current<0){ok=0;break;}line_offset=(uint64_t)current;length=0;
        while(length<cap){int c=fgetc(f);if(c==EOF||c=='\n')break;line[length++]=(unsigned char)c;}
        if(ferror(f)){ok=0;break;}
        if(length&&line[length-1]=='\r')length--;
        begin=0;while(begin<length&&(line[begin]==' '||line[begin]=='\t'||line[begin]=='\r'))begin++;
        if(begin<length)break;
        if(feof(f)){ok=0;break;}
    }
    uint32_t *encoded=NULL;size_t encoded_count=0;
    if(ok&&(begin==length||!ama10_tokenizer_encode(tokenizer,line+begin,length-begin,&encoded,&encoded_count)||!encoded_count))ok=0;
    if(ok){*token_count=encoded_count<64?encoded_count:64;memcpy(ids,encoded,*token_count*sizeof(uint32_t));
        ok=ama10_tokenizer_decode(tokenizer,ids,*token_count,prompt_bytes,prompt_length);}
    if(ok)*byte_offset=line_offset+begin;
    free(encoded);free(line);if(fclose(f)!=0)ok=0;
    if(!ok){free(*prompt_bytes);*prompt_bytes=NULL;*prompt_length=0;*token_count=0;return 0;}
    return 1;
}
static int eval_dataset_region(const AMA10Model *m,AMA10TokenDataset *dataset,
                               size_t count,double *loss){
    if(!count||count==SIZE_MAX)return 0;
    uint32_t *window=(uint32_t*)malloc((count+1)*sizeof(uint32_t));
    uint32_t *in=(uint32_t*)malloc(count*sizeof(uint32_t));
    uint32_t *target=(uint32_t*)malloc(count*sizeof(uint32_t));
    if(!window||!in||!target){free(window);free(in);free(target);return 0;}
    int ok=ama10_token_dataset_read(dataset,0,count+1,window);
    if(ok){for(size_t i=0;i<count;i++){in[i]=window[i];target[i]=window[i+1];}
        ok=ama10_eval_loss(m,in,target,count,2,loss);}
    free(window);free(in);free(target);return ok;
}
static void print_generation_metrics(const AMA10Tokenizer *tokenizer,const uint32_t *tokens,
                                    size_t token_count,const unsigned char *bytes,size_t byte_count){
    AMA10GenerationMetrics metrics;
    if(!ama10_generation_metrics(tokenizer,tokens,token_count,bytes,byte_count,&metrics)){
        fprintf(stderr,"generation metrics unavailable\n");return;
    }
    printf("generation_distinct_token_ratio=%.4f repeated_4gram_ratio=%.4f codepoints=%llu invalid_utf8_bytes=%llu scripts_latin=%llu cjk=%llu kana=%llu hangul=%llu cyrillic=%llu arabic=%llu devanagari=%llu other=%llu\n",
        metrics.distinct_token_ratio,metrics.repeated_fourgram_ratio,
        (unsigned long long)metrics.codepoints,(unsigned long long)metrics.invalid_utf8_bytes,
        (unsigned long long)metrics.latin,(unsigned long long)metrics.cjk,
        (unsigned long long)metrics.kana,(unsigned long long)metrics.hangul,
        (unsigned long long)metrics.cyrillic,(unsigned long long)metrics.arabic,
        (unsigned long long)metrics.devanagari,(unsigned long long)metrics.other);
}
static int train_file(const char *train_path,const char *validation_path,size_t updates,
                      float learning_rate,float learning_rate_decay,size_t target_vocab,size_t accumulation_steps,
                      uint32_t optimizer_algorithm,
                      uint64_t tokenizer_sample_limit,
                      const char *tokenizer_file,
                      const char *checkpoint,int resume,int opencl_adamw,size_t model_width,
                      const char *metrics_csv){
    unsigned char *sample=NULL,*generated_bytes=NULL;
    uint32_t *generated_ids=NULL;
    size_t sample_bytes=0,generated_len=0,generated_count=0;
    uint64_t n=0,nval=0,train_tokens=0,val_tokens=0;
    AMA10Tokenizer tokenizer={0};
    AMA10Model m={0};AMA10Optimizer opt={0};AMA10State state={0};
    AMA10Gradients gradients={0};
#ifdef AMA10_ENABLE_OPENCL
    AMA10OpenCL *update_backend=NULL;
#endif
    AMA10TokenDataset train_dataset={0},validation_dataset={0};
    AMA10CheckpointProgress progress={0};
    FILE *metrics_file=NULL;
    double wall_start=native_wall_seconds(),phase_start=wall_start;
    double tokenizer_seconds=0.0,cache_seconds=0.0,setup_eval_seconds=0.0;
    double update_seconds=0.0,final_eval_seconds=0.0,generation_seconds=0.0;
    uint64_t train_hash=0,validation_hash=0;
    if(!tokenizer_sample_limit||
       !ama10_file_fingerprint(train_path,&train_hash,&n)||
       !ama10_file_fingerprint(validation_path,&validation_hash,&nval)){
        fprintf(stderr,"training or validation file fingerprint failed\n");goto fail;
    }
    if(n<64||nval<2){fprintf(stderr,"train/validation text is too short\n");goto fail;}
    if(!resume){
        if(tokenizer_file){
            if(!ama10_tokenizer_load(&tokenizer,tokenizer_file)||tokenizer.vocabulary_size!=target_vocab){
                fprintf(stderr,"supplied tokenizer could not be loaded or its vocabulary does not match\n");goto fail;
            }
        }else if(!read_tokenizer_sample(train_path,n,tokenizer_sample_limit,&sample,&sample_bytes)||sample_bytes<64||
                 !ama10_tokenizer_train(&tokenizer,sample,sample_bytes,target_vocab)){
            fprintf(stderr,"bounded-sample tokenizer training failed\n");goto fail;
        }
        free(sample);sample=NULL;
    }
    tokenizer_seconds=native_wall_seconds()-phase_start;phase_start=native_wall_seconds();
    if(resume){
        if(!checkpoint||!ama10_checkpoint_load(checkpoint,&m,&tokenizer,&opt,&state,&progress)){
            fprintf(stderr,"training checkpoint could not be loaded\n");goto fail;
        }
        if(progress.train_hash!=train_hash||progress.validation_hash!=validation_hash||
           progress.sequence_length!=24||progress.depth!=2||m.width!=model_width||
           progress.accumulation_steps!=accumulation_steps||progress.target_vocabulary!=target_vocab||
           progress.learning_rate!=learning_rate||progress.weight_decay!=0.01f||
           progress.tokenizer_sample_bytes!=tokenizer_sample_limit||
           progress.optimizer_algorithm!=optimizer_algorithm){
            fprintf(stderr,"resume data or training settings do not match checkpoint\n");goto fail;
        }
        if(progress.learning_rate_decay!=learning_rate_decay){
            fprintf(stderr,"resume learning-rate schedule does not match checkpoint\n");goto fail;
        }
    }
    {
        uint64_t tok_hash=ama10_tokenizer_fingerprint(&tokenizer);
        char train_cache[160],validation_cache[160];
        if(!tok_hash||snprintf(train_cache,sizeof(train_cache),"build/train-%016llx-%016llx.ama10ds",
            (unsigned long long)train_hash,(unsigned long long)tok_hash)>=(int)sizeof(train_cache)||
           snprintf(validation_cache,sizeof(validation_cache),"build/validation-%016llx-%016llx.ama10ds",
            (unsigned long long)validation_hash,(unsigned long long)tok_hash)>=(int)sizeof(validation_cache)||
           !ama10_token_dataset_build_or_open(&train_dataset,train_path,train_cache,&tokenizer,train_hash,n)||
           !ama10_token_dataset_build_or_open(&validation_dataset,validation_path,validation_cache,
                                              &tokenizer,validation_hash,nval)){
            fprintf(stderr,"streaming token dataset build or open failed\n");goto fail;
        }
        train_tokens=train_dataset.token_count;val_tokens=validation_dataset.token_count;
    }
    cache_seconds=native_wall_seconds()-phase_start;phase_start=native_wall_seconds();
    if(train_tokens<64||val_tokens<2){fprintf(stderr,"tokenized train/validation text is too short\n");goto fail;}
    const size_t width=model_width,seq=24,eval_n=512;
    if(!resume&&(!ama10_model_init(&m,tokenizer.vocabulary_size,width,width,UINT64_C(10))||
       !ama10_optimizer_init(&opt,&m)||!ama10_optimizer_set_algorithm(&opt,optimizer_algorithm)||
       !ama10_state_init(&state,width))){
        fprintf(stderr,"model initialization failed\n");goto fail;
    }
    if((accumulation_steps>1||optimizer_algorithm==AMA10_OPT_MUON||opencl_adamw)&&!ama10_gradients_init(&gradients,&m)){
        fprintf(stderr,"gradient accumulator initialization failed\n");goto fail;
    }
#ifdef AMA10_ENABLE_OPENCL
    if(opencl_adamw){update_backend=ama10_opencl_create(&m,"kernels/ama10_recurrent.cl");if(!update_backend){fprintf(stderr,"OpenCL AdamW backend initialization failed\n");goto fail;}}
#else
    if(opencl_adamw){fprintf(stderr,"--opencl-adamw requires a build made with -OpenCL\n");goto fail;}
#endif
    size_t train_eval=train_tokens-1;if(train_eval>eval_n)train_eval=eval_n;
    size_t val_eval=val_tokens-1;if(val_eval>eval_n)val_eval=eval_n;
    double before_train,before_val,after_train,after_val;
    if(!eval_dataset_region(&m,&train_dataset,train_eval,&before_train)||
       !eval_dataset_region(&m,&validation_dataset,val_eval,&before_val)){
        fprintf(stderr,"initial evaluation failed\n");goto fail;
    }
    setup_eval_seconds=native_wall_seconds()-phase_start;phase_start=native_wall_seconds();
    uint64_t start_update=resume?progress.completed_updates:0;
    if(metrics_csv){
        metrics_file=fopen(metrics_csv,resume?"ab":"wb");
        if(!metrics_file){fprintf(stderr,"could not open requested metrics CSV\n");goto fail;}
        if(fseek(metrics_file,0,SEEK_END)!=0){fprintf(stderr,"metrics CSV seek failed\n");goto fail;}
        long metrics_bytes=ftell(metrics_file);
        if(metrics_bytes<0||(!metrics_bytes&&fputs("step,key,value\n",metrics_file)<0)){fprintf(stderr,"metrics CSV header write failed\n");goto fail;}
        if(fprintf(metrics_file,"%llu,train/token_nll_before,%.9g\n%llu,validation/token_nll_before,%.9g\n",
                   (unsigned long long)start_update,before_train,(unsigned long long)start_update,before_val)<0){
            fprintf(stderr,"initial metrics CSV write failed\n");goto fail;
        }
    }
    uint64_t cursor=resume?progress.cursor:0;double running=0;
    if(cursor>=train_tokens) { fprintf(stderr,"checkpoint data cursor is invalid\n");goto fail; }
    if(resume&&(uint64_t)updates<start_update){fprintf(stderr,"requested total updates precede checkpoint\n");goto fail;}
    if(resume) printf("resuming_at_update=%llu cursor=%zu optimizer_step=%llu\n",
        (unsigned long long)progress.completed_updates,cursor,(unsigned long long)opt.step);
    for(uint64_t step=resume?progress.completed_updates:0;step<(uint64_t)updates;step++){
        float update_loss=0.0f;
        float effective_lr=ama10_learning_rate(learning_rate,step+1,learning_rate_decay);
        if(!(effective_lr>0.0f)){fprintf(stderr,"learning-rate schedule produced an invalid rate\n");goto fail;}
        for(size_t micro=0;micro<accumulation_steps;micro++){
            if(cursor+seq>=train_tokens){cursor=0;ama10_state_reset(&state);}
            uint32_t window[seq+1],in[seq],target[seq];
            if(!ama10_token_dataset_read(&train_dataset,cursor,seq+1,window)){
                fprintf(stderr,"training token window read failed at cursor %llu\n",(unsigned long long)cursor);goto fail;
            }
            for(size_t j=0;j<seq;j++){in[j]=window[j];target[j]=window[j+1];}
            float micro_loss=0.0f;
            int trained;
#ifdef AMA10_ENABLE_OPENCL
            if(opencl_adamw)trained=ama10_train_sequence_accumulate_opencl(&m,&opt,&state,in,target,seq,2,
                                                                           &gradients,update_backend,&micro_loss);
            else
#endif
            trained=accumulation_steps==1&&optimizer_algorithm==AMA10_OPT_ADAMW?
                ama10_train_sequence(&m,&opt,&state,in,target,seq,2,effective_lr,0.01f,&micro_loss):
                ama10_train_sequence_accumulate(&m,&opt,&state,in,target,seq,2,&gradients,&micro_loss);
            if(!trained){
                fprintf(stderr,"training failed at update %llu microbatch %zu\n",
                        (unsigned long long)(step+1),micro);goto fail;
            }
            update_loss+=micro_loss;cursor+=seq;
        }
        int update_ok=1;
        if(accumulation_steps>1||optimizer_algorithm==AMA10_OPT_MUON||opencl_adamw){
#ifdef AMA10_ENABLE_OPENCL
            update_ok=opencl_adamw?ama10_opencl_adamw_apply(update_backend,&m,&opt,&gradients,effective_lr,0.01f):
                ama10_optimizer_apply(&m,&opt,&gradients,effective_lr,0.01f);
#else
            update_ok=ama10_optimizer_apply(&m,&opt,&gradients,effective_lr,0.01f);
#endif
        }
        if(!update_ok){
            fprintf(stderr,"AdamW accumulation step failed at update %llu\n",(unsigned long long)(step+1));goto fail;
        }
        float loss=update_loss/(float)accumulation_steps;running+=loss;
        uint64_t completed=step+1;
        if(metrics_file&&fprintf(metrics_file,"%llu,train/chunk_loss,%.9g\n%llu,train/learning_rate,%.9g\n",
                                 (unsigned long long)completed,(double)loss,(unsigned long long)completed,(double)effective_lr)<0){
            fprintf(stderr,"metrics CSV write failed at update %llu\n",(unsigned long long)completed);goto fail;
        }
        if((completed%100==0||completed==(uint64_t)updates)&&checkpoint){
            progress.completed_updates=completed;progress.cursor=cursor;
            progress.train_hash=train_hash;progress.validation_hash=validation_hash;
            progress.tokenizer_sample_bytes=tokenizer_sample_limit;
            progress.sequence_length=(uint32_t)seq;progress.depth=2;
            progress.accumulation_steps=(uint32_t)accumulation_steps;
            progress.target_vocabulary=(uint32_t)target_vocab;
            progress.optimizer_algorithm=optimizer_algorithm;
            progress.learning_rate=learning_rate;progress.weight_decay=0.01f;
            progress.learning_rate_decay=learning_rate_decay;
#ifdef AMA10_ENABLE_OPENCL
            if(opencl_adamw&&!ama10_opencl_adamw_sync(update_backend,&m,&opt)){fprintf(stderr,"OpenCL optimizer moment synchronization failed before checkpoint\n");goto fail;}
#endif
            if(!ama10_checkpoint_save(checkpoint,&m,&tokenizer,&opt,&state,&progress)){
                fprintf(stderr,"training checkpoint save failed at update %llu\n",(unsigned long long)completed);goto fail;
            }
            size_t path_len=strlen(checkpoint);char *tok_path=(char*)malloc(path_len+5);
            if(!tok_path){fprintf(stderr,"tokenizer path allocation failed\n");goto fail;}
            memcpy(tok_path,checkpoint,path_len);memcpy(tok_path+path_len,".tok",5);
            int tok_saved=ama10_tokenizer_save(&tokenizer,tok_path);free(tok_path);
            if(!tok_saved){fprintf(stderr,"tokenizer sidecar save failed\n");goto fail;}
        }
        if(metrics_file&&(completed%100==0||completed==(uint64_t)updates)&&fflush(metrics_file)!=0){fprintf(stderr,"metrics CSV flush failed\n");goto fail;}
        if(completed%100==0||completed==(uint64_t)updates)
            printf("update=%llu learning_rate=%.7g train_chunk_loss=%.5f mean_chunk_loss=%.5f\n",
                   (unsigned long long)completed,effective_lr,loss,running/(double)(completed-start_update));
    }
    update_seconds=native_wall_seconds()-phase_start;phase_start=native_wall_seconds();
#ifdef AMA10_ENABLE_OPENCL
    if(opencl_adamw&&!ama10_opencl_adamw_sync(update_backend,&m,&opt)){fprintf(stderr,"OpenCL optimizer moment synchronization failed\n");goto fail;}
#endif
    if(!eval_dataset_region(&m,&train_dataset,train_eval,&after_train)||
       !eval_dataset_region(&m,&validation_dataset,val_eval,&after_val)){
        fprintf(stderr,"final evaluation failed\n");goto fail;
    }
    final_eval_seconds=native_wall_seconds()-phase_start;phase_start=native_wall_seconds();
    printf("train_bytes=%llu validation_bytes=%llu train_tokens=%llu validation_tokens=%llu tokenizer_sample_budget_bytes=%llu vocabulary=%zu updates=%zu depth=2 width=%zu accumulation=%zu optimizer=%s seed=10 learning_rate=%.7g inverse_sqrt_decay=%.7g\n",
           (unsigned long long)n,(unsigned long long)nval,(unsigned long long)train_tokens,
           (unsigned long long)val_tokens,(unsigned long long)tokenizer_sample_limit,
           tokenizer.vocabulary_size,updates,width,accumulation_steps,
           optimizer_algorithm==AMA10_OPT_MUON?"muon-hybrid":opencl_adamw?"adamw-opencl-hybrid":"adamw",learning_rate,learning_rate_decay);
#ifdef AMA10_ENABLE_OPENCL
    if(opencl_adamw)printf("training_compute=opencl_recurrent_forward_output_head_bptt_adamw+cpu_loss_orchestration device=%s\n",ama10_opencl_device_name(update_backend));
#endif
    if(optimizer_algorithm==AMA10_OPT_MUON)
        printf("muon_hidden_lr=%.7g auxiliary_adamw_lr=%.7g momentum=0.95 nesterov=true newton_schulz_steps=5\n",
               learning_rate,learning_rate*0.015f);
    printf("train_token_nll_before=%.6f train_token_nll_after=%.6f validation_token_nll_before=%.6f validation_token_nll_after=%.6f\n",
           before_train,after_train,before_val,after_val);
    if(metrics_file){
        if(fprintf(metrics_file,"%llu,train/token_nll_after,%.9g\n%llu,validation/token_nll_after,%.9g\n",
                   (unsigned long long)updates,after_train,(unsigned long long)updates,after_val)<0||fflush(metrics_file)!=0){
            fprintf(stderr,"final metrics CSV write failed\n");goto fail;
        }
        fclose(metrics_file);metrics_file=NULL;
    }
    if(checkpoint&&progress.completed_updates!=(uint64_t)updates){
        progress.completed_updates=(uint64_t)updates;progress.cursor=cursor;
        progress.train_hash=train_hash;progress.validation_hash=validation_hash;
        progress.tokenizer_sample_bytes=tokenizer_sample_limit;
        progress.sequence_length=(uint32_t)seq;progress.depth=2;
        progress.accumulation_steps=(uint32_t)accumulation_steps;
        progress.target_vocabulary=(uint32_t)target_vocab;
        progress.optimizer_algorithm=optimizer_algorithm;
        progress.learning_rate=learning_rate;progress.weight_decay=0.01f;
        progress.learning_rate_decay=learning_rate_decay;
        if(!ama10_checkpoint_save(checkpoint,&m,&tokenizer,&opt,&state,&progress)){
            fprintf(stderr,"final checkpoint save failed\n");goto fail;
        }
        size_t path_len=strlen(checkpoint);char *tok_path=(char*)malloc(path_len+5);
        if(!tok_path){fprintf(stderr,"tokenizer path allocation failed\n");goto fail;}
        memcpy(tok_path,checkpoint,path_len);memcpy(tok_path+path_len,".tok",5);
        int tok_saved=ama10_tokenizer_save(&tokenizer,tok_path);free(tok_path);
        if(!tok_saved){fprintf(stderr,"tokenizer sidecar save failed\n");goto fail;}
    }
    /* Greedy continuation from a fixed held-out validation prefix for comparable runs. */
    {
        const char *gen_path="build/generated.bin";FILE *g=fopen(gen_path,"wb");
        float *logits=(float*)malloc(tokenizer.vocabulary_size*sizeof(float));
        if(g&&logits){
            ama10_state_reset(&state);
            size_t prompt=0;uint64_t prompt_start=0;
            uint32_t prompt_ids[64];
            unsigned char *prompt_bytes=NULL;size_t prompt_byte_count=0;
            int prompt_ok=build_validation_prompt(validation_path,nval,&tokenizer,prompt_ids,
                                                   &prompt,&prompt_start,&prompt_bytes,&prompt_byte_count);
            if(prompt_ok){
                FILE *prompt_file=fopen("build/generation_prompt.bin","wb");
                if(!prompt_file||fwrite(prompt_bytes,1,prompt_byte_count,prompt_file)!=prompt_byte_count)prompt_ok=0;
                if(prompt_file&&fclose(prompt_file)!=0)prompt_ok=0;
            }
            free(prompt_bytes);
            for(size_t j=0;j<prompt&&prompt_ok;j++)
                if(!ama10_step(&m,&state,prompt_ids[j],logits,2,2.0f)){prompt_ok=0;break;}
            generated_ids=(uint32_t*)malloc(128*sizeof(uint32_t));
            if(!generated_ids)prompt_ok=0;
            Utf8State generated_utf8={0,0x80,0xbf};
            for(size_t k=0;k<128;k++){
                if(!prompt_ok)break;
                size_t best=tokenizer.vocabulary_size;float best_logit=-INFINITY;
                int require_complete=(k+1==128);
                for(size_t j=0;j<tokenizer.vocabulary_size;j++)
                    if(logits[j]>best_logit&&utf8_candidate_valid(&tokenizer,&generated_utf8,j,require_complete)){
                        best=j;best_logit=logits[j];
                    }
                if(best==tokenizer.vocabulary_size||
                   !utf8_feed(&generated_utf8,tokenizer.tokens[best].bytes,tokenizer.tokens[best].length)){
                    prompt_ok=0;break;
                }
                generated_ids[generated_count++]=(uint32_t)best;
                if(!ama10_step(&m,&state,(uint32_t)best,logits,2,2.0f)){prompt_ok=0;break;}
            }
            if(!ama10_tokenizer_decode(&tokenizer,generated_ids,generated_count,&generated_bytes,&generated_len)||
               fwrite(generated_bytes,1,generated_len,g)!=generated_len)prompt_ok=0;
            if(prompt_ok)print_generation_metrics(&tokenizer,generated_ids,generated_count,
                                                  generated_bytes,generated_len);
            fclose(g);
            if(prompt_ok)printf("generation_prompt_source=validation byte_offset=%llu prompt_tokens=%zu prompt_bytes=%zu\n",
                                (unsigned long long)prompt_start,prompt,prompt_byte_count);
            if(prompt_ok)printf("greedy_tokens=%zu generated_bytes=%zu generation_file=%s\n",generated_count,generated_len,gen_path);
            else fprintf(stderr,"generation output failed\n");
        } else {if(g)fclose(g);fprintf(stderr,"generation output unavailable\n");}
        free(logits);
    }
    generation_seconds=native_wall_seconds()-phase_start;
    {
        double wall=native_wall_seconds()-wall_start;
        uint64_t updates_done=(uint64_t)updates-start_update;
        double token_positions=(double)updates_done*(double)seq*(double)accumulation_steps;
        printf("phase_seconds tokenizer=%.6f token_cache=%.6f setup_and_initial_eval=%.6f training_updates=%.6f final_eval=%.6f generation=%.6f total=%.6f updates_per_second=%.6f token_positions_per_second=%.3f\n",
            tokenizer_seconds,cache_seconds,setup_eval_seconds,update_seconds,final_eval_seconds,
            generation_seconds,wall,update_seconds>0.0?(double)updates_done/update_seconds:0.0,
            update_seconds>0.0?token_positions/update_seconds:0.0);
    }
    ama10_gradients_free(&gradients);
#ifdef AMA10_ENABLE_OPENCL
    ama10_opencl_destroy(update_backend);
#endif
    ama10_token_dataset_close(&train_dataset);
    ama10_token_dataset_close(&validation_dataset);ama10_state_free(&state);ama10_optimizer_free(&opt);
    ama10_model_free(&m);ama10_tokenizer_free(&tokenizer);
    free(sample);free(generated_ids);free(generated_bytes);return 0;
fail:
    if(metrics_file)fclose(metrics_file);
    ama10_gradients_free(&gradients);
#ifdef AMA10_ENABLE_OPENCL
    ama10_opencl_destroy(update_backend);
#endif
    ama10_token_dataset_close(&train_dataset);
    ama10_token_dataset_close(&validation_dataset);ama10_state_free(&state);ama10_optimizer_free(&opt);
    ama10_model_free(&m);ama10_tokenizer_free(&tokenizer);
    free(sample);free(generated_ids);free(generated_bytes);return 1;
}
static int generate_file(const char *model_path,const char *tokenizer_path,const char *prompt_path,
                         const char *output_path,size_t max_tokens,int use_opencl,
                         float repetition_penalty,const char *adapter_path){
    AMA10Model m={0};AMA10Tokenizer tokenizer={0};AMA10State state={0};
    AMA10Optimizer optimizer={0};AMA10CheckpointProgress checkpoint_progress={0};
    AMA10LoRA adapter={0};float *hidden=NULL;
#ifdef AMA10_ENABLE_OPENCL
    AMA10OpenCL *opencl=NULL;
#endif
    unsigned char *prompt=NULL,*generated=NULL;size_t prompt_bytes=0,generated_bytes=0;
    uint32_t *prompt_ids=NULL,*output_ids=NULL;size_t prompt_count=0;
    float *logits=NULL;size_t *frequency=NULL;int rc=1;
    int bundled=ama10_checkpoint_load(model_path,&m,&tokenizer,&optimizer,&state,&checkpoint_progress);
    if(bundled){
        AMA10Tokenizer sidecar={0};
        if(!ama10_tokenizer_load(&sidecar,tokenizer_path)||
           ama10_tokenizer_fingerprint(&sidecar)!=ama10_tokenizer_fingerprint(&tokenizer)){
            ama10_tokenizer_free(&sidecar);fprintf(stderr,"checkpoint tokenizer sidecar does not match\n");goto done;
        }
        ama10_tokenizer_free(&sidecar);ama10_optimizer_free(&optimizer);ama10_state_reset(&state);
    }else if(!ama10_model_load(&m,model_path)||!ama10_tokenizer_load(&tokenizer,tokenizer_path)||
             m.vocabulary!=tokenizer.vocabulary_size||!ama10_state_init(&state,m.width)){
        fprintf(stderr,"model and tokenizer load failed\n");goto done;
    }
    if(m.vocabulary!=tokenizer.vocabulary_size||!read_all(prompt_path,&prompt,&prompt_bytes)||
       !utf8_valid(prompt,prompt_bytes)||
       !ama10_tokenizer_encode(&tokenizer,prompt,prompt_bytes,&prompt_ids,&prompt_count)||!prompt_count){
        fprintf(stderr,"model, tokenizer, or UTF-8 prompt load failed\n");goto done;
    }
    if(adapter_path){uint64_t base_hash=0,base_bytes=0;
        if(use_opencl||!ama10_file_fingerprint(model_path,&base_hash,&base_bytes)||
           !ama10_lora_load(&adapter,adapter_path,&m,base_hash,ama10_tokenizer_fingerprint(&tokenizer))){
            fprintf(stderr,"LoRA adapter does not match this CPU model and tokenizer\n");goto done;
        }
        hidden=(float*)malloc(m.width*sizeof(float));if(!hidden)goto done;
    }
#ifdef AMA10_ENABLE_OPENCL
    if(use_opencl){
        opencl=ama10_opencl_create(&m,"kernels/ama10_recurrent.cl");
        if(!opencl||!ama10_opencl_reset(opencl))goto done;
    }
#else
    if(use_opencl){fprintf(stderr,"this executable was built without OpenCL; use scripts/build.ps1 -OpenCL\n");goto done;}
#endif
    if(max_tokens>SIZE_MAX/sizeof(uint32_t)){fprintf(stderr,"generation length exceeds addressable memory\n");goto done;}
    logits=(float*)malloc(m.vocabulary*sizeof(float));
    output_ids=(uint32_t*)malloc((max_tokens?max_tokens:1)*sizeof(uint32_t));
    frequency=(size_t*)calloc(m.vocabulary,sizeof(size_t));
    if(!logits||!output_ids||!frequency){fprintf(stderr,"generation allocation failed\n");goto done;}
    for(size_t i=0;i<prompt_count;i++)if(adapter_path){
        if(!ama10_step_hidden(&m,&state,prompt_ids[i],logits,hidden,2,2.0f))goto done;
        ama10_lora_apply(&adapter,hidden,logits);
    }else if(
#ifdef AMA10_ENABLE_OPENCL
        (opencl?ama10_opencl_step_stream(opencl,prompt_ids[i],logits,2,2.0f):
#endif
        ama10_step(&m,&state,prompt_ids[i],logits,2,2.0f)
#ifdef AMA10_ENABLE_OPENCL
        )
#endif
        ==0){
        fprintf(stderr,"prompt inference failed at token %zu\n",i);goto done;
    }
    size_t count=0;
    Utf8State generated_utf8={0,0x80,0xbf};
    for(;count<max_tokens;count++){
        size_t best=m.vocabulary;float best_logit=-INFINITY;
        int require_complete=(count+1==max_tokens);
        for(size_t j=0;j<m.vocabulary;j++){
            float score=logits[j]-(float)frequency[j]*repetition_penalty;
            if(score>best_logit&&utf8_candidate_valid(&tokenizer,&generated_utf8,j,require_complete)){
                best=j;best_logit=score;
            }
        }
        if(best==m.vocabulary){fprintf(stderr,"no UTF-8-valid continuation exists at token %zu\n",count);goto done;}
        output_ids[count]=(uint32_t)best;
        if(frequency[best]<SIZE_MAX)frequency[best]++;
        if(!utf8_feed(&generated_utf8,tokenizer.tokens[best].bytes,tokenizer.tokens[best].length)){
            fprintf(stderr,"UTF-8 decoder state failed at token %zu\n",count);goto done;
        }
        if(adapter_path){
            if(!ama10_step_hidden(&m,&state,(uint32_t)best,logits,hidden,2,2.0f)){fprintf(stderr,"decode failed at token %zu\n",count);goto done;}
            ama10_lora_apply(&adapter,hidden,logits);
        }else if(
#ifdef AMA10_ENABLE_OPENCL
           (opencl?ama10_opencl_step_stream(opencl,(uint32_t)best,logits,2,2.0f):
#endif
           ama10_step(&m,&state,(uint32_t)best,logits,2,2.0f)
#ifdef AMA10_ENABLE_OPENCL
           )
#endif
           ==0){
            fprintf(stderr,"decode failed at token %zu\n",count);goto done;
        }
    }
    if(!ama10_tokenizer_decode(&tokenizer,output_ids,count,&generated,&generated_bytes)){
        fprintf(stderr,"token decoding failed\n");goto done;
    }
    print_generation_metrics(&tokenizer,output_ids,count,generated,generated_bytes);
    {
        FILE *f=fopen(output_path,"wb");
        if(!f){perror(output_path);goto done;}
        int ok=fwrite(generated,1,generated_bytes,f)==generated_bytes;
        if(fclose(f)!=0)ok=0;
        if(!ok){fprintf(stderr,"output write failed\n");goto done;}
    }
    printf("prompt_bytes=%zu prompt_tokens=%zu generated_tokens=%zu generated_bytes=%zu backend=%s output=%s\n",
           prompt_bytes,prompt_count,count,generated_bytes,
#ifdef AMA10_ENABLE_OPENCL
           opencl?ama10_opencl_device_name(opencl):"cpu",
#else
           "cpu",
#endif
           output_path);
    rc=0;
done:
    free(prompt);free(generated);free(prompt_ids);free(output_ids);free(logits);free(frequency);free(hidden);ama10_lora_free(&adapter);
    ama10_state_free(&state);ama10_tokenizer_free(&tokenizer);ama10_model_free(&m);
#ifdef AMA10_ENABLE_OPENCL
    ama10_opencl_destroy(opencl);
#endif
    ama10_optimizer_free(&optimizer);
    return rc;
}
static int evaluate_adaptive_dataset(const AMA10Model *model,AMA10TokenDataset *dataset,
                                     unsigned max_depth,float threshold,
                                     size_t reset_period,
                                     double *mean_loss,double *mean_depth,
                                     double *cpu_seconds){
    if(!model||!dataset||dataset->token_count<2||!max_depth||!mean_loss||!mean_depth)return 0;
    clock_t start=clock();
    AMA10State state={0};float *logits=(float*)malloc(model->vocabulary*sizeof(float));
    uint32_t window[513];double loss_sum=0,steps_sum=0;uint64_t predictions=0;
    if(!logits||!ama10_state_init(&state,model->width)){free(logits);ama10_state_free(&state);return 0;}
    for(uint64_t cursor=0;cursor+1<dataset->token_count;){
        size_t count=(size_t)((dataset->token_count-1-cursor)>512?512:(dataset->token_count-1-cursor));
        if(!ama10_token_dataset_read(dataset,cursor,count+1,window))goto fail;
        for(size_t i=0;i<count;i++){
            if(reset_period&&predictions&&predictions%(uint64_t)reset_period==0)
                ama10_state_reset(&state);
            unsigned before=state.steps;
            if(window[i+1]>=model->vocabulary||!ama10_step(model,&state,window[i],logits,max_depth,threshold))goto fail;
            float maximum=-INFINITY;for(size_t j=0;j<model->vocabulary;j++)if(logits[j]>maximum)maximum=logits[j];
            if(!isfinite(maximum))goto fail;
            double exp_sum=0;for(size_t j=0;j<model->vocabulary;j++)exp_sum+=exp((double)logits[j]-maximum);
            if(!(exp_sum>0)||!isfinite(exp_sum))goto fail;
            loss_sum+=(double)maximum+log(exp_sum)-(double)logits[window[i+1]];
            steps_sum+=(double)(state.steps-before);predictions++;
        }
        cursor+=(uint64_t)count;
    }
    free(logits);ama10_state_free(&state);
    if(predictions+1!=dataset->token_count)return 0;
    *mean_loss=loss_sum/(double)predictions;*mean_depth=steps_sum/(double)predictions;
    if(cpu_seconds){clock_t end=clock();*cpu_seconds=start!=(clock_t)-1&&end>=start?
        (double)(end-start)/(double)CLOCKS_PER_SEC:-1.0;}
    return 1;
fail:
    free(logits);ama10_state_free(&state);return 0;
}

static int evaluate_file(const char *model_path,const char *tokenizer_path,const char *data_path,
                         unsigned adaptive_depth,float halt_threshold,size_t reset_period){
    AMA10Model model={0};AMA10Tokenizer tokenizer={0};AMA10State state={0};
    AMA10Optimizer optimizer={0};AMA10CheckpointProgress progress={0};
    AMA10TokenDataset dataset={0};int ok=0;uint32_t optimizer_algorithm=AMA10_OPT_ADAMW;
    uint64_t data_hash=0,raw_bytes=0;
    if(!ama10_file_fingerprint(data_path,&data_hash,&raw_bytes)||!raw_bytes)goto done;
    int bundled=ama10_checkpoint_load(model_path,&model,&tokenizer,&optimizer,&state,&progress);
    if(bundled){
        optimizer_algorithm=optimizer.algorithm;
        AMA10Tokenizer sidecar={0};
        int sidecar_ok=ama10_tokenizer_load(&sidecar,tokenizer_path)&&
            ama10_tokenizer_fingerprint(&sidecar)==ama10_tokenizer_fingerprint(&tokenizer);
        ama10_tokenizer_free(&sidecar);ama10_optimizer_free(&optimizer);
        if(!sidecar_ok)goto done;
        ama10_state_reset(&state);
    }else if(!ama10_model_load(&model,model_path)||!ama10_tokenizer_load(&tokenizer,tokenizer_path)||
             model.vocabulary!=tokenizer.vocabulary_size||!ama10_state_init(&state,model.width))goto done;
    char cache_path[192];uint64_t tokenizer_hash=ama10_tokenizer_fingerprint(&tokenizer);
    if(!tokenizer_hash||snprintf(cache_path,sizeof(cache_path),"build/eval-%016llx-%016llx.ama10ds",
       (unsigned long long)data_hash,(unsigned long long)tokenizer_hash)>=(int)sizeof(cache_path)||
       !ama10_token_dataset_build_or_open(&dataset,data_path,cache_path,&tokenizer,data_hash,raw_bytes)||
       dataset.token_count<2)goto done;
    float *logits=(float*)malloc(model.vocabulary*sizeof(float));
    uint32_t window[513];if(!logits)goto done;
    ama10_state_reset(&state);double total_loss=0;uint64_t predictions=0;clock_t fixed_start=clock();
    for(uint64_t cursor=0;cursor+1<dataset.token_count;){
        size_t count=(size_t)((dataset.token_count-1-cursor)>512?512:(dataset.token_count-1-cursor));
        if(!ama10_token_dataset_read(&dataset,cursor,count+1,window)){free(logits);goto done;}
        for(size_t i=0;i<count;i++){
            if(!ama10_step(&model,&state,window[i],logits,2,2.0f)){free(logits);goto done;}
            float maximum=-INFINITY;
            for(size_t j=0;j<model.vocabulary;j++)if(logits[j]>maximum)maximum=logits[j];
            if(window[i+1]>=model.vocabulary||!isfinite(maximum)){free(logits);goto done;}
            double exp_sum=0;for(size_t j=0;j<model.vocabulary;j++)exp_sum+=exp((double)logits[j]-maximum);
            if(!(exp_sum>0)||!isfinite(exp_sum)){free(logits);goto done;}
            total_loss+=(double)maximum+log(exp_sum)-(double)logits[window[i+1]];
            predictions++;
        }
        cursor+=(uint64_t)count;
    }
    clock_t fixed_end=clock();double fixed_seconds=fixed_start!=(clock_t)-1&&fixed_end>=fixed_start?
        (double)(fixed_end-fixed_start)/(double)CLOCKS_PER_SEC:-1.0;
    free(logits);
    if(predictions+1!=dataset.token_count)goto done;
    double nats_per_byte=total_loss/(double)raw_bytes;
    printf("eval_bytes=%llu eval_tokens=%llu predictions=%llu token_nll=%.8f nats_per_byte=%.8f bits_per_byte=%.8f checkpoint_updates=%llu optimizer=%s\n",
        (unsigned long long)raw_bytes,(unsigned long long)dataset.token_count,
        (unsigned long long)predictions,total_loss/(double)predictions,nats_per_byte,nats_per_byte/0.6931471805599453,
        (unsigned long long)progress.completed_updates,
        bundled?(optimizer_algorithm==AMA10_OPT_MUON?"muon-hybrid":"adamw"):"weights-only");
    if(reset_period){double reset_loss=0,reset_depth=0,reset_seconds=-1.0;
        if(!evaluate_adaptive_dataset(&model,&dataset,2,2.0f,reset_period,&reset_loss,&reset_depth,&reset_seconds))goto done;
        printf("state_reset_period=%zu state_reset_fixed_depth=2 state_reset_token_nll=%.8f state_reset_mean_depth=%.6f state_reset_cpu_seconds=%.6f\n",
            reset_period,reset_loss,reset_depth,reset_seconds);
    }
    if(adaptive_depth){double adaptive_loss=0,mean_depth=0,fixed_max_loss=0,fixed_max_depth=0;
        double adaptive_seconds=-1.0,fixed_max_seconds=-1.0;
        if(!evaluate_adaptive_dataset(&model,&dataset,adaptive_depth,2.0f,0,&fixed_max_loss,&fixed_max_depth,&fixed_max_seconds)||
           !evaluate_adaptive_dataset(&model,&dataset,adaptive_depth,halt_threshold,0,&adaptive_loss,&mean_depth,&adaptive_seconds))goto done;
        printf("fixed_two_step_cpu_seconds=%.6f fixed_max_depth=%u fixed_max_depth_token_nll=%.8f fixed_mean_depth=%.6f fixed_max_depth_cpu_seconds=%.6f\n",
            fixed_seconds,adaptive_depth,fixed_max_loss,fixed_max_depth,fixed_max_seconds);
        printf("adaptive_max_depth=%u halt_threshold=%.6f adaptive_token_nll=%.8f adaptive_mean_depth=%.6f adaptive_cpu_seconds=%.6f\n",
            adaptive_depth,halt_threshold,adaptive_loss,mean_depth,adaptive_seconds);
    }
    ok=1;
done:
    ama10_token_dataset_close(&dataset);ama10_state_free(&state);ama10_optimizer_free(&optimizer);
    ama10_tokenizer_free(&tokenizer);ama10_model_free(&model);
    if(!ok)fprintf(stderr,"full-stream evaluation failed\n");
    return ok?0:1;
}
static int score_eval_logits(const float *logits,size_t vocabulary,uint32_t target,double *loss){
    if(!logits||!vocabulary||target>=vocabulary||!loss)return 0;
    float maximum=logits[0];for(size_t i=1;i<vocabulary;i++)if(logits[i]>maximum)maximum=logits[i];
    if(!isfinite(maximum))return 0;
    double sum=0;for(size_t i=0;i<vocabulary;i++)sum+=exp((double)logits[i]-(double)maximum);
    if(!(sum>0)||!isfinite(sum))return 0;
    *loss=(double)maximum+log(sum)-(double)logits[target];return isfinite(*loss);
}
static int evaluate_document_state(const char *model_path,const char *tokenizer_path,
                                   const char *pages_path){
    AMA10Model model={0};AMA10Tokenizer tokenizer={0};AMA10Optimizer optimizer={0};
    AMA10State carry={0},reset={0};AMA10CheckpointProgress progress={0};
    unsigned char *bytes=NULL;size_t byte_count=0;float *carry_logits=NULL,*reset_logits=NULL;
    uint32_t optimizer_algorithm=AMA10_OPT_ADAMW;uint64_t predictions=0,tokens=0;size_t pages=0;
    double carry_loss=0,reset_loss=0;int ok=0;clock_t eval_start,eval_end;
    int bundled=ama10_checkpoint_load(model_path,&model,&tokenizer,&optimizer,&carry,&progress);
    if(bundled){
        optimizer_algorithm=optimizer.algorithm;AMA10Tokenizer sidecar={0};
        int sidecar_ok=ama10_tokenizer_load(&sidecar,tokenizer_path)&&
            ama10_tokenizer_fingerprint(&sidecar)==ama10_tokenizer_fingerprint(&tokenizer);
        ama10_tokenizer_free(&sidecar);ama10_optimizer_free(&optimizer);
        if(!sidecar_ok)goto done;
        ama10_state_reset(&carry);
    }else if(!ama10_model_load(&model,model_path)||!ama10_tokenizer_load(&tokenizer,tokenizer_path)||
             model.vocabulary!=tokenizer.vocabulary_size||!ama10_state_init(&carry,model.width))goto done;
    if(!ama10_state_init(&reset,model.width)||!read_all(pages_path,&bytes,&byte_count))goto done;
    carry_logits=(float*)malloc(model.vocabulary*sizeof(float));reset_logits=(float*)malloc(model.vocabulary*sizeof(float));
    if(!carry_logits||!reset_logits)goto done;
    eval_start=clock();
    size_t begin=0;
    while(begin<byte_count){
        size_t end=begin;while(end<byte_count&&bytes[end]!=0x0c)end++;
        size_t page_end=end;while(page_end>begin&&(bytes[page_end-1]=='\r'||bytes[page_end-1]=='\n'))page_end--;
        while(begin<page_end&&(bytes[begin]=='\r'||bytes[begin]=='\n'))begin++;
        if(page_end<=begin)goto done;
        uint32_t *ids=NULL;size_t count=0;
        if(!ama10_tokenizer_encode(&tokenizer,bytes+begin,page_end-begin,&ids,&count)||count<2){free(ids);goto done;}
        ama10_state_reset(&reset);
        for(size_t i=0;i<count;i++){
            if(!ama10_step(&model,&carry,ids[i],carry_logits,2,2.0f)||
               !ama10_step(&model,&reset,ids[i],reset_logits,2,2.0f)){free(ids);goto done;}
            if(i+1<count){double lc=0,lr=0;
                if(!score_eval_logits(carry_logits,model.vocabulary,ids[i+1],&lc)||
                   !score_eval_logits(reset_logits,model.vocabulary,ids[i+1],&lr)){free(ids);goto done;}
                carry_loss+=lc;reset_loss+=lr;predictions++;
            }
        }
        tokens+=count;pages++;free(ids);
        if(end==byte_count)begin=byte_count;else begin=end+1;
    }
    eval_end=clock();
    if(!pages||!predictions)goto done;
    printf("document_pages=%zu scored_predictions=%llu tokenized_tokens=%llu checkpoint_updates=%llu optimizer=%s\n",
        pages,(unsigned long long)predictions,(unsigned long long)tokens,
        (unsigned long long)progress.completed_updates,
        bundled?(optimizer_algorithm==AMA10_OPT_MUON?"muon-hybrid":"adamw"):"weights-only");
    printf("document_state_carry_token_nll=%.8f document_reset_token_nll=%.8f paired_cpu_seconds=%.6f\n",
        carry_loss/(double)predictions,reset_loss/(double)predictions,
        eval_start!=(clock_t)-1&&eval_end>=eval_start?
        (double)(eval_end-eval_start)/(double)CLOCKS_PER_SEC:-1.0);
    ok=1;
done:
    free(bytes);free(carry_logits);free(reset_logits);ama10_state_free(&carry);ama10_state_free(&reset);
    ama10_optimizer_free(&optimizer);ama10_tokenizer_free(&tokenizer);ama10_model_free(&model);
    if(!ok)fprintf(stderr,"document state evaluation failed\n");
    return ok?0:1;
}
static int profile_tokenizer_file(const char *tokenizer_path,const char *data_path){
    AMA10Tokenizer tokenizer={0};unsigned char *bytes=NULL,*decoded=NULL;
    uint32_t *ids=NULL;size_t byte_count=0,decoded_count=0,token_count=0;
    if(!ama10_tokenizer_load(&tokenizer,tokenizer_path)||!read_all(data_path,&bytes,&byte_count)||
       !utf8_valid(bytes,byte_count)||
       !ama10_tokenizer_encode(&tokenizer,bytes,byte_count,&ids,&token_count)||
       !ama10_tokenizer_decode(&tokenizer,ids,token_count,&decoded,&decoded_count)||
       decoded_count!=byte_count||memcmp(bytes,decoded,byte_count)!=0){
        fprintf(stderr,"tokenizer profile failed: invalid UTF-8 or non-exact round trip\n");
        free(bytes);free(decoded);free(ids);ama10_tokenizer_free(&tokenizer);return 1;
    }
    uint64_t fallback_tokens=0;size_t max_token_bytes=0;
    for(size_t i=0;i<token_count;i++){
        if(ids[i]<256)fallback_tokens++;
        if(tokenizer.tokens[ids[i]].length>max_token_bytes)max_token_bytes=tokenizer.tokens[ids[i]].length;
    }
    double fallback_fraction=token_count?(double)fallback_tokens/(double)token_count:0.0;
    printf("profile_bytes=%zu tokens=%zu bytes_per_token=%.6f fallback_token_fraction=%.6f learned_token_fraction=%.6f max_token_bytes=%zu vocabulary=%zu roundtrip=exact\n",
        byte_count,token_count,token_count?(double)byte_count/(double)token_count:0.0,
        fallback_fraction,1.0-fallback_fraction,max_token_bytes,tokenizer.vocabulary_size);
    free(bytes);free(decoded);free(ids);ama10_tokenizer_free(&tokenizer);return 0;
}
static int dataset_cache_open(AMA10TokenDataset *dataset,const char *path,uint64_t raw_hash,
                              uint64_t raw_bytes,const AMA10Tokenizer *tokenizer){
    char cache[192];uint64_t tok_hash=ama10_tokenizer_fingerprint(tokenizer);
    return tok_hash&&snprintf(cache,sizeof(cache),"build/train-%016llx-%016llx.ama10ds",
        (unsigned long long)raw_hash,(unsigned long long)tok_hash)<(int)sizeof(cache)&&
        ama10_token_dataset_build_or_open(dataset,path,cache,tokenizer,raw_hash,raw_bytes);
}
static int dataset_cursor_raw_offset(AMA10TokenDataset *dataset,const AMA10Tokenizer *tokenizer,
                                     uint64_t cursor,uint64_t *raw_offset){
    if(!dataset||!tokenizer||!raw_offset||cursor>=dataset->token_count)return 0;
    uint32_t tokens[4096];uint64_t position=0;
    for(uint64_t start=0;start<cursor;){
        size_t count=(size_t)((cursor-start)>4096?4096:(cursor-start));
        if(!ama10_token_dataset_read(dataset,start,count,tokens))return 0;
        for(size_t i=0;i<count;i++){
            size_t len=tokenizer->tokens[tokens[i]].length;
            if(position>UINT64_MAX-len)return 0;
            position+=(uint64_t)len;
        }
        start+=(uint64_t)count;
    }
    *raw_offset=position;return 1;
}
static int dataset_raw_offset_cursor(AMA10TokenDataset *dataset,const AMA10Tokenizer *tokenizer,
                                     uint64_t raw_offset,uint64_t *cursor){
    if(!dataset||!tokenizer||!cursor)return 0;
    if(raw_offset==0){*cursor=0;return 1;}
    uint32_t tokens[4096];uint64_t position=0,count_read=0;
    while(count_read<dataset->token_count){
        size_t count=(size_t)((dataset->token_count-count_read)>4096?4096:
                              (dataset->token_count-count_read));
        if(!ama10_token_dataset_read(dataset,count_read,count,tokens))return 0;
        for(size_t i=0;i<count;i++){
            size_t len=tokenizer->tokens[tokens[i]].length;
            if(position>UINT64_MAX-len)return 0;
            if(position+(uint64_t)len>raw_offset){*cursor=count_read+i;return 1;}
            position+=(uint64_t)len;
            if(position==raw_offset){
                *cursor=count_read+i+1;
                if(*cursor>=dataset->token_count&&dataset->token_count>1)*cursor=dataset->token_count-1;
                return *cursor<dataset->token_count;
            }
        }
        count_read+=(uint64_t)count;
    }
    if(dataset->token_count<2)return 0;
    *cursor=dataset->token_count-1;return 1;
}
static int grow_checkpoint_vocabulary(const char *checkpoint_path,const char *tokenizer_path,
                                      const char *training_path,size_t target_vocabulary,
                                      const char *output_path){
    AMA10Model model={0},old_shape={0};AMA10Tokenizer tokenizer={0},sidecar={0};
    AMA10Optimizer optimizer={0};AMA10State state={0};AMA10CheckpointProgress progress={0};
    AMA10TokenDataset old_dataset={0},new_dataset={0};
    unsigned char *sample=NULL;size_t sample_bytes=0;uint64_t train_hash=0,train_bytes=0;
    int ok=0;size_t old_vocabulary=0;uint64_t cursor_raw_offset=0;
    if(!ama10_checkpoint_load(checkpoint_path,&model,&tokenizer,&optimizer,&state,&progress)||
       !ama10_tokenizer_load(&sidecar,tokenizer_path)||
       ama10_tokenizer_fingerprint(&sidecar)!=ama10_tokenizer_fingerprint(&tokenizer)||
       !ama10_file_fingerprint(training_path,&train_hash,&train_bytes)||
       train_hash!=progress.train_hash||target_vocabulary<=model.vocabulary||
       target_vocabulary>UINT32_MAX||optimizer.algorithm!=progress.optimizer_algorithm){
        fprintf(stderr,"checkpoint, tokenizer, training-file fingerprint, or vocabulary target mismatch\n");goto done;
    }
    old_vocabulary=model.vocabulary;
    if(!dataset_cache_open(&old_dataset,training_path,train_hash,train_bytes,&tokenizer)||
       !dataset_cursor_raw_offset(&old_dataset,&tokenizer,progress.cursor,&cursor_raw_offset)){
        fprintf(stderr,"could not map the old token cursor to source bytes\n");goto done;
    }
    if(!read_tokenizer_sample(training_path,train_bytes,progress.tokenizer_sample_bytes,
                              &sample,&sample_bytes)||sample_bytes<64||
       !ama10_tokenizer_extend(&tokenizer,sample,sample_bytes,target_vocabulary)||
       tokenizer.vocabulary_size<=old_vocabulary){
        fprintf(stderr,"training sample did not support vocabulary growth\n");goto done;
    }
    old_shape.width=model.width;old_shape.input_width=model.input_width;old_shape.vocabulary=old_vocabulary;
    uint64_t seed=ama10_data_hash(sample,sample_bytes)^progress.completed_updates^
                  (uint64_t)tokenizer.vocabulary_size;
    if(seed==0)seed=UINT64_C(0x9e3779b97f4a7c15);
    if(!ama10_model_resize_vocabulary(&model,tokenizer.vocabulary_size,seed)||
       !ama10_optimizer_resize_vocabulary(&optimizer,&old_shape,&model)||
       !dataset_cache_open(&new_dataset,training_path,train_hash,train_bytes,&tokenizer)||
       !dataset_raw_offset_cursor(&new_dataset,&tokenizer,cursor_raw_offset,&progress.cursor)){
        fprintf(stderr,"model, optimizer, or grown token-cache resize failed\n");goto done;
    }
    ama10_state_reset(&state);
    progress.target_vocabulary=(uint32_t)tokenizer.vocabulary_size;
    if(!ama10_checkpoint_save(output_path,&model,&tokenizer,&optimizer,&state,&progress)){
        fprintf(stderr,"grown checkpoint save failed\n");goto done;
    }
    size_t path_length=strlen(output_path);
    if(path_length>SIZE_MAX-5){fprintf(stderr,"tokenizer sidecar path too long\n");goto done;}
    char *sidecar_output=(char*)malloc(path_length+5);
    if(!sidecar_output){fprintf(stderr,"tokenizer sidecar path allocation failed\n");goto done;}
    memcpy(sidecar_output,output_path,path_length);memcpy(sidecar_output+path_length,".tok",5);
    int saved=ama10_tokenizer_save(&tokenizer,sidecar_output);free(sidecar_output);
    if(!saved){fprintf(stderr,"grown tokenizer sidecar save failed\n");goto done;}
    printf("vocabulary_growth_old=%zu requested=%zu actual=%zu checkpoint_updates=%llu optimizer=%s checkpoint=%s\n",
        old_vocabulary,target_vocabulary,tokenizer.vocabulary_size,
        (unsigned long long)progress.completed_updates,
        optimizer.algorithm==AMA10_OPT_MUON?"muon-hybrid":"adamw",output_path);
    ok=1;
done:
    ama10_token_dataset_close(&old_dataset);ama10_token_dataset_close(&new_dataset);
    free(sample);ama10_tokenizer_free(&sidecar);ama10_tokenizer_free(&tokenizer);
    ama10_optimizer_free(&optimizer);ama10_state_free(&state);ama10_model_free(&model);
    return ok?0:1;
}
static void usage(const char *exe){
    fprintf(stderr,"usage:\n  %s train <train-utf8-file> <validation-utf8-file> [total-updates=1000] [learning-rate=0.001] [vocabulary=512] [checkpoint-path] [--width N] [--accumulation N] [--lr-decay D] [--optimizer adamw|muon] [--tokenizer-sample-bytes N] [--tokenizer-file path] [--metrics-csv path] [--resume] [--opencl-adamw]\n  %s generate <model> <tokenizer> <prompt-file> <output-file> [tokens=128] [cpu|opencl] [repetition-penalty=0] [adapter.lora]\n  %s export-safetensors <checkpoint-or-model.safetensors> <output.safetensors>\n  %s evaluate <checkpoint-or-model> <tokenizer> <utf8-data-file> [adaptive-max-depth halt-threshold] [state-reset-period]\n      or: evaluate <checkpoint-or-model> <tokenizer> <utf8-data-file> <state-reset-period>\n  %s evaluate-documents <checkpoint-or-model> <tokenizer> <form-feed-separated-documents.txt>\n  %s profile <tokenizer> <utf8-data-file>\n  %s grow-vocabulary <checkpoint> <tokenizer> <training-file> <target-vocabulary> <output-checkpoint>\n",exe,exe,exe,exe,exe,exe,exe);
}
static int export_safetensors(const char *source,const char *destination){
    AMA10Model model={0};AMA10Tokenizer tokenizer={0};AMA10Optimizer optimizer={0};AMA10State state={0};
    AMA10CheckpointProgress progress={0};
    int loaded=ama10_checkpoint_load(source,&model,&tokenizer,&optimizer,&state,&progress);
    if(!loaded)loaded=ama10_model_load(&model,source);
    int ok=loaded&&ama10_model_save_safetensors(&model,destination);
    if(ok)printf("safetensors_export=ok width=%zu input_width=%zu vocabulary=%zu output=%s\n",
        model.width,model.input_width,model.vocabulary,destination);
    else fprintf(stderr,"SafeTensors export failed\n");
    ama10_state_free(&state);ama10_optimizer_free(&optimizer);ama10_tokenizer_free(&tokenizer);ama10_model_free(&model);
    return ok?0:1;
}
int main(int argc,char **argv){
    if(argc<2){usage(argv[0]);return 2;}
    if(strcmp(argv[1],"evaluate-documents")==0){
        if(argc!=5){usage(argv[0]);return 2;}
        return evaluate_document_state(argv[2],argv[3],argv[4]);
    }
    if(strcmp(argv[1],"export-safetensors")==0){
        if(argc!=4){usage(argv[0]);return 2;}
        return export_safetensors(argv[2],argv[3]);
    }
    if(strcmp(argv[1],"evaluate")==0){
        if(argc!=5&&argc!=6&&argc!=7&&argc!=8){usage(argv[0]);return 2;}
        unsigned adaptive_depth=0;float threshold=2.0f;size_t reset_period=0;
        if(argc==6||argc==8){const char *reset_arg=argc==6?argv[5]:argv[7];char *end=NULL;errno=0;
            unsigned long long parsed=strtoull(reset_arg,&end,10);
            if(errno||!end||*end||parsed==0||parsed>SIZE_MAX){fprintf(stderr,"state reset period must be a positive integer\n");return 2;}
            reset_period=(size_t)parsed;}
        if(argc>=7){char *end=NULL;errno=0;unsigned long parsed=strtoul(argv[5],&end,10);
            if(errno||!end||*end||parsed==0||parsed>UINT_MAX){fprintf(stderr,"invalid adaptive maximum depth\n");return 2;}
            adaptive_depth=(unsigned)parsed;errno=0;threshold=strtof(argv[6],&end);
            if(errno||!end||*end||!isfinite(threshold)||threshold<0.0f||threshold>1.0f){fprintf(stderr,"halt threshold must be between zero and one\n");return 2;}}
        return evaluate_file(argv[2],argv[3],argv[4],adaptive_depth,threshold,reset_period);
    }
    if(strcmp(argv[1],"profile")==0){
        if(argc!=4){usage(argv[0]);return 2;}
        return profile_tokenizer_file(argv[2],argv[3]);
    }
    if(strcmp(argv[1],"grow-vocabulary")==0){
        if(argc!=7){usage(argv[0]);return 2;}
        char *end=NULL;errno=0;unsigned long long target=strtoull(argv[5],&end,10);
        if(errno||!end||*end||target>SIZE_MAX){fprintf(stderr,"invalid target vocabulary size\n");return 2;}
        return grow_checkpoint_vocabulary(argv[2],argv[3],argv[4],(size_t)target,argv[6]);
    }
    if(strcmp(argv[1],"generate")==0){
        if(argc<6||argc>10){usage(argv[0]);return 2;}
        size_t count=128;
        if(argc>=7){char *end=NULL;errno=0;unsigned long long x=strtoull(argv[6],&end,10);
            if(errno||!end||*end||x>SIZE_MAX/sizeof(uint32_t)){fprintf(stderr,"invalid generation length\n");return 2;}count=(size_t)x;}
        int use_opencl=argc>=8&&strcmp(argv[7],"opencl")==0;
        if(argc>=8&&!use_opencl&&strcmp(argv[7],"cpu")!=0){fprintf(stderr,"backend must be cpu or opencl\n");return 2;}
        float repetition_penalty=0.0f;
        if(argc==9){char *end=NULL;errno=0;repetition_penalty=strtof(argv[8],&end);
            if(errno||end==argv[8]||*end||!isfinite(repetition_penalty)||repetition_penalty<0.0f||repetition_penalty>10.0f){
                fprintf(stderr,"repetition penalty must be in [0, 10]\n");return 2;
            }}
        return generate_file(argv[2],argv[3],argv[4],argv[5],count,use_opencl,repetition_penalty,argc==10?argv[9]:NULL);
    }
    if(strcmp(argv[1],"train")!=0||argc<4||argc>24){usage(argv[0]);return 2;}
    size_t updates=1000;
    if(argc>=5){char *end=NULL;errno=0;unsigned long long x=strtoull(argv[4],&end,10);
        if(errno||!end||*end||x==0||x>SIZE_MAX){fprintf(stderr,"invalid update count\n");return 2;}updates=(size_t)x;}
    float lr=0.001f;
    if(argc>=6){char *end=NULL;errno=0;lr=strtof(argv[5],&end);
        if(errno||!end||*end||!isfinite(lr)||lr<=0.0f){fprintf(stderr,"invalid learning rate\n");return 2;}}
    size_t vocab=512;
    if(argc>=7){char *end=NULL;errno=0;unsigned long long x=strtoull(argv[6],&end,10);
        if(errno||!end||*end||x<256||x>UINT32_MAX||x>SIZE_MAX){fprintf(stderr,"invalid tokenizer vocabulary size\n");return 2;}vocab=(size_t)x;}
    size_t accumulation=1,model_width=32;float lr_decay=0.0f;uint64_t tokenizer_sample=UINT64_C(8)*1024u*1024u;
    int resume=0,opencl_adamw=0,have_width=0,have_accumulation=0,have_lr_decay=0,have_tokenizer_sample=0,have_optimizer=0;
    int have_tokenizer_file=0;const char *tokenizer_file=NULL,*metrics_csv=NULL;
    uint32_t optimizer_algorithm=AMA10_OPT_ADAMW;
    for(int i=8;i<argc;i++){
        if(strcmp(argv[i],"--resume")==0){if(resume){fprintf(stderr,"duplicate --resume\n");return 2;}resume=1;}
        else if(strcmp(argv[i],"--opencl-adamw")==0){if(opencl_adamw){fprintf(stderr,"duplicate --opencl-adamw\n");return 2;}opencl_adamw=1;}
        else if(strcmp(argv[i],"--width")==0){if(have_width||i+1>=argc){fprintf(stderr,"invalid --width option\n");return 2;}char *end=NULL;errno=0;unsigned long long x=strtoull(argv[++i],&end,10);if(errno||!end||*end||x<8||x>4096){fprintf(stderr,"model width must be between 8 and 4096\n");return 2;}model_width=(size_t)x;have_width=1;}
        else if(strcmp(argv[i],"--accumulation")==0){
            if(have_accumulation||i+1>=argc){fprintf(stderr,"invalid --accumulation option\n");return 2;}
            char *end=NULL;errno=0;unsigned long long x=strtoull(argv[++i],&end,10);
            if(errno||!end||*end||x==0||x>UINT32_MAX||x>SIZE_MAX){fprintf(stderr,"invalid accumulation count\n");return 2;}
            accumulation=(size_t)x;have_accumulation=1;
        }
        else if(strcmp(argv[i],"--lr-decay")==0){
            if(have_lr_decay||i+1>=argc){fprintf(stderr,"invalid --lr-decay option\n");return 2;}
            char *end=NULL;errno=0;lr_decay=strtof(argv[++i],&end);
            if(errno||!end||*end||!isfinite(lr_decay)||lr_decay<0.0f){fprintf(stderr,"invalid learning-rate decay\n");return 2;}
            have_lr_decay=1;
        }else if(strcmp(argv[i],"--optimizer")==0){
            if(have_optimizer||i+1>=argc){fprintf(stderr,"invalid --optimizer option\n");return 2;}
            const char *name=argv[++i];
            if(strcmp(name,"adamw")==0)optimizer_algorithm=AMA10_OPT_ADAMW;
            else if(strcmp(name,"muon")==0)optimizer_algorithm=AMA10_OPT_MUON;
            else{fprintf(stderr,"optimizer must be adamw or muon\n");return 2;}
            have_optimizer=1;
        }else if(strcmp(argv[i],"--tokenizer-sample-bytes")==0){
            if(have_tokenizer_sample||i+1>=argc){fprintf(stderr,"invalid --tokenizer-sample-bytes option\n");return 2;}
            char *end=NULL;errno=0;unsigned long long x=strtoull(argv[++i],&end,10);
            if(errno||!end||*end||x<64||x>SIZE_MAX){fprintf(stderr,"invalid tokenizer sample size\n");return 2;}
            tokenizer_sample=(uint64_t)x;have_tokenizer_sample=1;
        }else if(strcmp(argv[i],"--tokenizer-file")==0){
            if(have_tokenizer_file||i+1>=argc||!*argv[i+1]){fprintf(stderr,"invalid --tokenizer-file option\n");return 2;}
            tokenizer_file=argv[++i];have_tokenizer_file=1;
        }else if(strcmp(argv[i],"--metrics-csv")==0){
            if(metrics_csv||i+1>=argc){fprintf(stderr,"invalid --metrics-csv option\n");return 2;}
            metrics_csv=argv[++i];if(!*metrics_csv){fprintf(stderr,"metrics CSV path cannot be empty\n");return 2;}
        }else{fprintf(stderr,"unknown training option: %s\n",argv[i]);return 2;}
    }
    if(opencl_adamw&&optimizer_algorithm!=AMA10_OPT_ADAMW){fprintf(stderr,"--opencl-adamw currently supports AdamW only\n");return 2;}
    if(resume&&argc<8){fprintf(stderr,"resume requires a checkpoint path\n");return 2;}
    if(resume&&tokenizer_file){fprintf(stderr,"--tokenizer-file cannot be used with --resume\n");return 2;}
    return train_file(argv[2],argv[3],updates,lr,lr_decay,vocab,accumulation,optimizer_algorithm,tokenizer_sample,
                      tokenizer_file,
                      argc>=8?argv[7]:NULL,resume,opencl_adamw,model_width,metrics_csv);
}
