#include "ama10.h"
#include "ama10_audio.h"
#include "ama10_tts.h"
#include "checkpoint.h"
#include "dataset.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char *path,*caption;char language[16];int validation;float *features;size_t frames; } TTSItem;
static void items_free(TTSItem *a,size_t n){if(!a)return;for(size_t i=0;i<n;i++){free(a[i].path);free(a[i].caption);free(a[i].features);}free(a);}
static char *dupstr(const char *s){size_t n=strlen(s);char *p=(char*)malloc(n+1);if(p)memcpy(p,s,n+1);return p;}
static int add_item(TTSItem **items,size_t *count,size_t *capacity,TTSItem *item){
    if(*count==*capacity){size_t next=*capacity?*capacity*2:16;if(next<*capacity||next>SIZE_MAX/sizeof(**items))return 0;
        TTSItem *p=(TTSItem*)realloc(*items,next*sizeof(**items));if(!p)return 0;*items=p;*capacity=next;}
    (*items)[(*count)++]=*item;memset(item,0,sizeof(*item));return 1;
}
static int read_manifest(const char *path,TTSItem **out,size_t *count,size_t *train,size_t *valid){
    FILE *f=fopen(path,"rb");if(!f)return 0;char line[16384];TTSItem *items=NULL;size_t n=0,cap=0,nt=0,nv=0;int ok=1;
    while(fgets(line,sizeof(line),f)){
        size_t len=strlen(line);if(len&&line[len-1]=='\n')line[--len]='\0';if(len&&line[len-1]=='\r')line[--len]='\0';if(!len)continue;
        char *a=strchr(line,'\t');if(!a){ok=0;break;}*a++='\0';char *b=strchr(a,'\t');if(!b){ok=0;break;}*b++='\0';char *c=strchr(b,'\t');
        if(c){*c++='\0';if(strchr(c,'\t')){ok=0;break;}}else c="en";
        if(!*line||!*a||!*b||!*c||(strcmp(b,"train")&&strcmp(b,"validation"))){ok=0;break;}
        TTSItem item={0};item.path=dupstr(line);item.caption=dupstr(a);item.validation=!strcmp(b,"validation");
        size_t langlen=strlen(c);if(langlen>=sizeof(item.language)){free(item.path);free(item.caption);ok=0;break;}
        memcpy(item.language,c,langlen+1);
        for(char *q=item.caption;*q;q++)if((unsigned char)*q>='a'&&(unsigned char)*q<='z')*q=(char)(*q-'a'+'A');
        if(!item.path||!item.caption||!add_item(&items,&n,&cap,&item)){free(item.path);free(item.caption);ok=0;break;}
        if(items[n-1].validation)nv++;else nt++;
    }
    if(ferror(f))ok=0;
    fclose(f);
    if(!ok||!nt||!nv){items_free(items,n);return 0;}
    *out=items;*count=n;*train=nt;*valid=nv;return 1;
}
static int join_path(const char *manifest,const char *relative,char **out){
    const char *a=strrchr(manifest,'/'),*b=strrchr(manifest,'\\'),*s=a;if(b&&(!s||b>s))s=b;
    size_t dir=s?(size_t)(s-manifest+1):0,n=strlen(relative);if(dir>SIZE_MAX-n-1)return 0;
    char *p=(char*)malloc(dir+n+1);if(!p)return 0;if(dir)memcpy(p,manifest,dir);memcpy(p+dir,relative,n+1);*out=p;return 1;
}
static int prepare(TTSItem *items,size_t count,const char *manifest,AMA10Tokenizer *tokenizer,int learn,size_t *sample_bytes){
    size_t total=0;for(size_t i=0;i<count;i++)if(!items[i].validation){size_t n=strlen(items[i].caption)+strlen(items[i].language)+10;if(total>SIZE_MAX-n-1)return 0;total+=n+1;}
    unsigned char *text=(unsigned char*)malloc(total);if(!text)return 0;size_t p=0;
    for(size_t i=0;i<count;i++)if(!items[i].validation){int n=snprintf((char*)text+p,total-p,"<lang:%s> ",items[i].language);if(n<0)return 0;p+=(size_t)n;size_t m=strlen(items[i].caption);memcpy(text+p,items[i].caption,m);p+=m;text[p++]='\n';}
    int ok=1;if(learn)ok=ama10_tokenizer_train(tokenizer,text,total,256);free(text);if(!ok||tokenizer->vocabulary_size!=256)return 0;*sample_bytes=total;
    for(size_t i=0;i<count;i++){
        char *full=NULL;if(!join_path(manifest,items[i].path,&full))return 0;AMA10Audio audio={0};ok=ama10_audio_load_wav16k(&audio,full);free(full);
        float *mel=NULL;size_t frames=0,width=0;if(!ok||!ama10_audio_log_mel(&audio,&mel,&frames,&width)||width!=40||!frames){free(mel);ama10_audio_free(&audio);return 0;}
        size_t pooled=frames/2+(frames%2!=0);items[i].features=(float*)malloc(pooled*40*sizeof(float));if(!items[i].features){free(mel);ama10_audio_free(&audio);return 0;}
        ok=ama10_audio_align_features(mel,frames,width,pooled,&items[i].features);free(mel);ama10_audio_free(&audio);if(!ok)return 0;items[i].frames=pooled;
    }
    return 1;
}
static uint64_t hash_bytes(uint64_t h,const void *p,size_t n){const unsigned char *b=(const unsigned char*)p;for(size_t i=0;i<n;i++){h^=b[i];h*=UINT64_C(1099511628211);}return h;}
static uint64_t split_hash(const TTSItem *items,size_t n,int validation){
    uint64_t h=UINT64_C(14695981039346656037);static const char marker[]="AMA10_TTS_TEXT_TO_MELVQ64_20MS_BYTE256_V1";h=hash_bytes(h,marker,sizeof(marker)-1);
    for(size_t i=0;i<n;i++)if(items[i].validation==validation){h=hash_bytes(h,items[i].caption,strlen(items[i].caption));unsigned char z=0;h=hash_bytes(h,&z,1);h=hash_bytes(h,items[i].language,strlen(items[i].language));h=hash_bytes(h,items[i].features,items[i].frames*40*sizeof(float));}
    return h;
}
static int prefix_bytes(const TTSItem *item,unsigned char **out,size_t *length){
    char lang[32];int n=snprintf(lang,sizeof(lang),"<lang:%s> ",item->language);size_t c=strlen(item->caption);
    if(n<0||(size_t)n>=sizeof(lang)||(size_t)n>SIZE_MAX-c-2)return 0;
    size_t len=(size_t)n+c+2;unsigned char *p=(unsigned char*)malloc(len);if(!p)return 0;
    memcpy(p,lang,(size_t)n);memcpy(p+n,item->caption,c);p[n+c]='\n';p[n+c+1]=0;*out=p;*length=len;return 1;
}
static int build_sequence(const TTSItem *item,const AMA10Tokenizer *tokenizer,const AMA10TTSCodebook *cb,uint32_t **out,size_t *length){
    unsigned char *bytes=NULL;size_t byte_count=0;uint32_t *prefix=NULL;size_t pn=0;
    if(!prefix_bytes(item,&bytes,&byte_count)||!ama10_tokenizer_encode(tokenizer,bytes,byte_count,&prefix,&pn)){free(bytes);free(prefix);return 0;}free(bytes);
    if(pn>SIZE_MAX-item->frames-1||pn+item->frames+1>SIZE_MAX/sizeof(uint32_t)){free(prefix);return 0;}
    size_t n=pn+item->frames+1;uint32_t *seq=(uint32_t*)malloc(n*sizeof(uint32_t));if(!seq){free(prefix);return 0;}
    memcpy(seq,prefix,pn*sizeof(uint32_t));free(prefix);
    for(size_t t=0;t<item->frames;t++){uint32_t code;if(!ama10_tts_quantize(cb,item->features+t*40,&code)){free(seq);return 0;}seq[pn+t]=code;}
    seq[n-1]=255;*out=seq;*length=n;return 1;
}
static int evaluate(const AMA10Model *model,const AMA10Tokenizer *tok,const AMA10TTSCodebook *cb,const TTSItem *items,size_t n,double *mean,uint64_t *tokens){
    double loss=0.0;uint64_t count=0;for(size_t i=0;i<n;i++)if(items[i].validation){uint32_t *seq=NULL;size_t len=0;double one=0.0;
        if(!build_sequence(&items[i],tok,cb,&seq,&len)||len<2||!ama10_eval_loss(model,seq,seq+1,len-1,2,&one)){free(seq);return 0;}
        free(seq);if(len-1>UINT64_MAX-count)return 0;count+=(uint64_t)(len-1);loss+=one*(double)(len-1);}
    if(!count)return 0;
    *mean=loss/(double)count;if(tokens)*tokens=count;return isfinite(*mean);
}
static uint64_t rng_next(uint64_t *x){*x^=*x>>12;*x^=*x<<25;*x^=*x>>27;return *x*UINT64_C(2685821657736338717);}
static void shuffle(size_t *order,size_t n,uint64_t seed,uint64_t epoch){uint64_t state=seed^(epoch*UINT64_C(0x9e3779b97f4a7c15));if(!state)state=1;
    for(size_t i=n;i>1;i--){size_t j=(size_t)(rng_next(&state)%(uint64_t)i);size_t t=order[i-1];order[i-1]=order[j];order[j]=t;}}
static char *suffix(const char *path,const char *s){size_t a=strlen(path),b=strlen(s);if(a>SIZE_MAX-b-1)return NULL;char *p=(char*)malloc(a+b+1);if(p){memcpy(p,path,a);memcpy(p+a,s,b+1);}return p;}
static int save_all(const char *path,const AMA10Model *model,const AMA10Tokenizer *tok,const AMA10Optimizer *opt,const AMA10State *state,const AMA10CheckpointProgress *progress,const AMA10TTSCodebook *cb){
    char *tp=suffix(path,".tok"),*vp=suffix(path,".vq");if(!tp||!vp){free(tp);free(vp);return 0;}
    int ok=ama10_checkpoint_save(path,model,tok,opt,state,progress)&&ama10_tokenizer_save(tok,tp)&&ama10_tts_codebook_save(cb,vp);free(tp);free(vp);return ok;
}
int main(int argc,char **argv){
    if(argc<4||argc>5){fprintf(stderr,"usage: %s <manifest.tsv> <updates-total> <checkpoint.ama> [--resume]\n",argv[0]);return 2;}
    char *end=NULL;errno=0;unsigned long long requested=strtoull(argv[2],&end,10);if(errno||end==argv[2]||*end||!requested||requested>UINT64_MAX){fprintf(stderr,"updates-total must be positive\n");return 2;}
    int resume=argc==5&&!strcmp(argv[4],"--resume");if(argc==5&&!resume){fprintf(stderr,"unknown option %s\n",argv[4]);return 2;}
    TTSItem *items=NULL;size_t count=0,train_count=0,valid_count=0,sample_bytes=0;AMA10Model model={0};AMA10Tokenizer tokenizer={0};AMA10Optimizer optimizer={0};AMA10State state={0};AMA10CheckpointProgress progress={0};AMA10TTSCodebook cb={0};size_t *order=NULL;int ok=0;const char *phase="manifest";
    if(!read_manifest(argv[1],&items,&count,&train_count,&valid_count))goto done;
    phase="checkpoint load";if(resume&&!ama10_checkpoint_load(argv[3],&model,&tokenizer,&optimizer,&state,&progress))goto done;
    phase="audio and tokenizer preparation";if(!prepare(items,count,argv[1],&tokenizer,!resume,&sample_bytes))goto done;
    uint64_t train_hash=split_hash(items,count,0),valid_hash=split_hash(items,count,1);
    size_t pooled_count=0,max_sequence=0;uint64_t train_text_bytes=0;for(size_t i=0;i<count;i++){
        if(!items[i].validation){size_t caption_bytes=strlen(items[i].caption);if(items[i].frames>SIZE_MAX-pooled_count||caption_bytes>UINT64_MAX-train_text_bytes)goto done;pooled_count+=items[i].frames;train_text_bytes+=(uint64_t)caption_bytes;}
        unsigned char *p=NULL;size_t pn=0;uint32_t *pi=NULL;size_t nic=0; if(!prefix_bytes(&items[i],&p,&pn)||!ama10_tokenizer_encode(&tokenizer,p,pn,&pi,&nic)){free(p);free(pi);goto done;}free(p);free(pi);
        if(nic>SIZE_MAX-items[i].frames-1)goto done;
        size_t seq=nic+items[i].frames+1;if(max_sequence<seq)max_sequence=seq;
    }
    if(max_sequence>UINT32_MAX)goto done;
    phase="codebook load or fit";char *vqpath=suffix(argv[3],".vq");if(!vqpath)goto done;
    if(resume){ok=ama10_tts_codebook_load(&cb,vqpath)&&cb.train_hash==train_hash&&cb.width==40;}
    else{
        float *pooled=(float*)malloc(pooled_count*40*sizeof(float));if(pooled){size_t at=0;for(size_t i=0;i<count;i++)if(!items[i].validation){memcpy(pooled+at*40,items[i].features,items[i].frames*40*sizeof(float));at+=items[i].frames;}
            ok=ama10_tts_fit_codebook(pooled,pooled_count,40,64,train_hash,&cb);free(pooled);}
    }
    free(vqpath);if(!ok)goto done;
    float expected_frames_per_byte=(float)((double)pooled_count/(double)train_text_bytes);
    if(resume&&cb.frames_per_byte>0.0f&&fabsf(cb.frames_per_byte-expected_frames_per_byte)>1.0e-5f)goto done;
    cb.frames_per_byte=expected_frames_per_byte;
    phase="model initialization";if(!resume){if(!ama10_model_init(&model,256,32,32,UINT64_C(20261117))||!ama10_optimizer_init(&optimizer,&model)||!ama10_state_init(&state,32))goto done;}
    if(model.vocabulary!=256||model.width!=32||model.input_width!=32||optimizer.algorithm!=AMA10_OPT_ADAMW||
       (resume&&(progress.train_hash!=train_hash||progress.validation_hash!=valid_hash||progress.tokenizer_sample_bytes!=sample_bytes||progress.sequence_length!=max_sequence||progress.target_vocabulary!=256||progress.depth!=2||progress.accumulation_steps!=1||progress.optimizer_algorithm!=optimizer.algorithm||progress.learning_rate!=0.0001f||progress.learning_rate_decay!=0.01f||progress.weight_decay!=0.01f||progress.cursor!=progress.completed_updates%(uint64_t)train_count||requested<=progress.completed_updates)))goto done;
    if(!resume){progress.train_hash=train_hash;progress.validation_hash=valid_hash;progress.tokenizer_sample_bytes=sample_bytes;progress.sequence_length=(uint32_t)max_sequence;progress.depth=2;progress.accumulation_steps=1;progress.target_vocabulary=256;progress.optimizer_algorithm=optimizer.algorithm;progress.learning_rate=0.0001f;progress.learning_rate_decay=0.01f;progress.weight_decay=0.01f;}
    double before=0.0,after=0.0;uint64_t eval_tokens=0;phase="initial evaluation";if(!evaluate(&model,&tokenizer,&cb,items,count,&before,&eval_tokens))goto done;
    order=(size_t*)malloc(train_count*sizeof(*order));if(!order)goto done;size_t at=0;for(size_t i=0;i<count;i++)if(!items[i].validation)order[at++]=i;
    uint64_t completed=progress.completed_updates,active_epoch=UINT64_MAX;phase="training";
    for(;completed<(uint64_t)requested;completed++){
        uint64_t epoch=completed/(uint64_t)train_count;if(epoch!=active_epoch){shuffle(order,train_count,train_hash,epoch);active_epoch=epoch;}
        uint32_t *seq=NULL;size_t len=0;float loss=0.0f;size_t index=order[(size_t)(completed%(uint64_t)train_count)];
        if(!build_sequence(&items[index],&tokenizer,&cb,&seq,&len)||len<2){free(seq);goto done;}
        ama10_state_reset(&state);float lr=ama10_learning_rate(0.0001f,completed+1,0.01f);
        int trained=ama10_train_sequence(&model,&optimizer,&state,seq,seq+1,len-1,2,lr,0.01f,&loss);free(seq);
        if(!trained||!isfinite(loss))goto done;
        if((completed+1)%100==0){progress.completed_updates=optimizer.step;progress.cursor=progress.completed_updates%(uint64_t)train_count;progress.train_hash=train_hash;progress.validation_hash=valid_hash;progress.tokenizer_sample_bytes=sample_bytes;progress.sequence_length=(uint32_t)max_sequence;progress.depth=2;progress.accumulation_steps=1;progress.target_vocabulary=256;progress.optimizer_algorithm=optimizer.algorithm;progress.learning_rate=0.0001f;progress.learning_rate_decay=0.01f;progress.weight_decay=0.01f;
            if(!save_all(argv[3],&model,&tokenizer,&optimizer,&state,&progress,&cb))goto done;
            printf("tts_update=%llu train_token_nll=%.5f lr=%.7g\n",(unsigned long long)(completed+1),(double)loss,(double)lr);}
    }
    phase="final evaluation";if(!evaluate(&model,&tokenizer,&cb,items,count,&after,NULL))goto done;
    progress.completed_updates=optimizer.step;progress.cursor=progress.completed_updates%(uint64_t)train_count;progress.train_hash=train_hash;progress.validation_hash=valid_hash;progress.tokenizer_sample_bytes=sample_bytes;progress.sequence_length=(uint32_t)max_sequence;progress.depth=2;progress.accumulation_steps=1;progress.target_vocabulary=256;progress.optimizer_algorithm=optimizer.algorithm;progress.learning_rate=0.0001f;progress.learning_rate_decay=0.01f;progress.weight_decay=0.01f;
    phase="checkpoint save and reload";if(!save_all(argv[3],&model,&tokenizer,&optimizer,&state,&progress,&cb))goto done;
    {AMA10Model vm={0};AMA10Tokenizer vt={0};AMA10Optimizer vo={0};AMA10State vs={0};AMA10CheckpointProgress vp={0};AMA10TTSCodebook vc={0};double persisted=0.0;char *vp_path=suffix(argv[3],".vq");
        ok=vp_path&&ama10_checkpoint_load(argv[3],&vm,&vt,&vo,&vs,&vp)&&ama10_tts_codebook_load(&vc,vp_path)&&vp.completed_updates==progress.completed_updates&&vo.step==optimizer.step&&vc.train_hash==train_hash&&evaluate(&vm,&vt,&vc,items,count,&persisted,NULL)&&fabs(persisted-after)<1.0e-6;
        free(vp_path);ama10_tts_codebook_free(&vc);ama10_state_free(&vs);ama10_optimizer_free(&vo);ama10_model_free(&vm);ama10_tokenizer_free(&vt);}
    if(ok)printf("tts_pairs=%zu train=%zu validation=%zu codebook=%u frame_rate=50 train_sample_bytes=%zu validation_tokens=%llu updates=%llu validation_loss_before=%.6f after=%.6f checkpoint=%s\n",count,train_count,valid_count,cb.count,sample_bytes,(unsigned long long)eval_tokens,(unsigned long long)optimizer.step,before,after,argv[3]);
done:
    if(!ok)fprintf(stderr,"TTS training failed during %s\n",phase);
    free(order);items_free(items,count);ama10_tts_codebook_free(&cb);ama10_state_free(&state);ama10_optimizer_free(&optimizer);ama10_model_free(&model);ama10_tokenizer_free(&tokenizer);return ok?0:1;
}
