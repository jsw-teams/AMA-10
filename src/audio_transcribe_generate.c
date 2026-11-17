#include "ama10_encoder.h"
#include "ama10_audio.h"
#include "multimodal_checkpoint.h"
#include "tokenizer.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { unsigned remaining;unsigned char low,high; } Utf8State;
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
typedef struct {
    uint32_t *tokens;size_t length;double blank,nonblank,language_score;Utf8State utf8;
} CTCBeam;
typedef struct { size_t vocabulary;uint64_t *unigrams,*bigrams,*row_totals;uint64_t total; } CTCLanguageModel;
static void language_model_free(CTCLanguageModel *lm){free(lm->unigrams);free(lm->bigrams);free(lm->row_totals);memset(lm,0,sizeof(*lm));}
static int language_model_build(CTCLanguageModel *lm,const AMA10Tokenizer *tokenizer,
                               const char *manifest,const char *language){
    memset(lm,0,sizeof(*lm));lm->vocabulary=tokenizer->vocabulary_size;
    if(!lm->vocabulary||lm->vocabulary>SIZE_MAX/lm->vocabulary||
       lm->vocabulary*lm->vocabulary>SIZE_MAX/sizeof(uint64_t))return 0;
    lm->unigrams=(uint64_t*)calloc(lm->vocabulary,sizeof(uint64_t));
    lm->bigrams=(uint64_t*)calloc(lm->vocabulary*lm->vocabulary,sizeof(uint64_t));
    lm->row_totals=(uint64_t*)calloc(lm->vocabulary,sizeof(uint64_t));
    FILE *f=fopen(manifest,"rb");if(!lm->unigrams||!lm->bigrams||!lm->row_totals||!f){language_model_free(lm);if(f)fclose(f);return 0;}
    char line[16384];int ok=1;
    while(fgets(line,sizeof(line),f)){
        size_t n=strlen(line);if(n&&line[n-1]=='\n')line[--n]='\0';if(n&&line[n-1]=='\r')line[--n]='\0';
        char *tab1=strchr(line,'\t');if(!tab1){ok=0;break;}char *caption=tab1+1;
        char *tab2=strchr(caption,'\t');if(!tab2){ok=0;break;}*tab2++='\0';
        char *tab3=strchr(tab2,'\t');char *row_language="en";
        if(tab3){*tab3++='\0';row_language=tab3;if(strchr(row_language,'\t')){ok=0;break;}}
        if(strcmp(tab2,"train")||strcmp(row_language,language))continue;
        uint32_t *tokens=NULL;size_t count=0;
        if(!*caption||!ama10_tokenizer_encode(tokenizer,(const unsigned char*)caption,
             strlen(caption),&tokens,&count)||!count){free(tokens);ok=0;break;}
        uint32_t previous=UINT32_MAX;
        for(size_t i=0;i<count;i++){
            uint32_t token=tokens[i];if(token>=lm->vocabulary||lm->unigrams[token]==UINT64_MAX||lm->total==UINT64_MAX){ok=0;break;}
            lm->unigrams[token]++;lm->total++;
            if(previous!=UINT32_MAX){size_t at=(size_t)previous*lm->vocabulary+token;
                if(lm->bigrams[at]==UINT64_MAX||lm->row_totals[previous]==UINT64_MAX){ok=0;break;}
                lm->bigrams[at]++;lm->row_totals[previous]++;}
            previous=token;
        }
        free(tokens);if(!ok)break;
    }
    if(ferror(f))ok=0;
    fclose(f);
    if(!ok||!lm->total){language_model_free(lm);return 0;}return 1;
}
static double language_transition(const CTCLanguageModel *lm,uint32_t previous,uint32_t token){
    if(!lm||!lm->unigrams||previous>=lm->vocabulary||token>=lm->vocabulary)return 0.0;
    double v=(double)lm->vocabulary;
    double pair=(double)lm->bigrams[(size_t)previous*lm->vocabulary+token];
    double row=(double)lm->row_totals[previous];
    double uni=(double)lm->unigrams[token];
    return log((pair+1.0)/(row+v))-log((uni+1.0)/((double)lm->total+v));
}
static double beam_logadd(double a,double b){
    if(a==-INFINITY)return b;
    if(b==-INFINITY)return a;
    double m=a>b?a:b;return m+log1p(exp((a>b?b:a)-m));
}
static int beam_add(CTCBeam *items,size_t *count,size_t capacity,const uint32_t *tokens,
                    size_t length,double blank,double nonblank,double language_score,const Utf8State *utf8){
    for(size_t i=0;i<*count;i++)if(items[i].length==length&&
        (!length||!memcmp(items[i].tokens,tokens,length*sizeof(uint32_t)))){
        items[i].blank=beam_logadd(items[i].blank,blank);
        items[i].nonblank=beam_logadd(items[i].nonblank,nonblank);return 1;
    }
    if(*count>=capacity)return 0;
    uint32_t *copy=NULL;
    if(length){copy=(uint32_t*)malloc(length*sizeof(uint32_t));if(!copy)return 0;
        memcpy(copy,tokens,length*sizeof(uint32_t));}
    CTCBeam *b=&items[(*count)++];b->tokens=copy;b->length=length;
    b->blank=blank;b->nonblank=nonblank;b->language_score=language_score;b->utf8=*utf8;return 1;
}
static int beam_score_order(const void *a,const void *b){
    const CTCBeam *x=(const CTCBeam*)a,*y=(const CTCBeam*)b;
    double xs=beam_logadd(x->blank,x->nonblank)+x->language_score;
    double ys=beam_logadd(y->blank,y->nonblank)+y->language_score;
    return xs>ys?-1:xs<ys?1:0;
}
static int ctc_prefix_beam(const AMA10Tokenizer *tokenizer,const float *logits,size_t frames,
                           size_t max_tokens,float blank_penalty,const CTCLanguageModel *lm,
                           double lm_weight,double insertion_bonus,uint32_t **decoded,
                           size_t *decoded_count){
    const size_t beam_width=8,top_classes=16;
    size_t vocab=tokenizer->vocabulary_size,capacity=beam_width*(2*top_classes+1);
    if(!frames||!vocab||frames>SIZE_MAX/vocab)return 0;
    CTCBeam *beam=(CTCBeam*)calloc(beam_width,sizeof(*beam));if(!beam)return 0;
    size_t beam_count=1;beam[0].blank=0.0;beam[0].nonblank=-INFINITY;
    beam[0].utf8=(Utf8State){0,0x80,0xbf};
    for(size_t t=0;t<frames;t++){
        float maxlog=-INFINITY;for(size_t k=0;k<vocab;k++){
            float z=logits[t*vocab+k]-(k==0?blank_penalty:0.0f);
            if(!isfinite(z)){for(size_t i=0;i<beam_count;i++)free(beam[i].tokens);free(beam);return 0;}
            if(z>maxlog)maxlog=z;
        }
        double norm=0.0;for(size_t k=0;k<vocab;k++)norm+=exp((double)(logits[t*vocab+k]-(k==0?blank_penalty:0.0f))-maxlog);
        double lognorm=(double)maxlog+log(norm);
        double blank_logp=(double)(logits[t*vocab] - blank_penalty)-lognorm;
        size_t best_ids[top_classes];double best_values[top_classes];
        for(size_t j=0;j<top_classes;j++){best_ids[j]=0;best_values[j]=-INFINITY;}
        for(size_t k=1;k<vocab;k++){
            double lp=(double)logits[t*vocab+k]-lognorm;
            if(lp<=best_values[top_classes-1])continue;
            size_t at=top_classes-1;while(at&&lp>best_values[at-1]){
                best_values[at]=best_values[at-1];best_ids[at]=best_ids[at-1];at--;
            }
            best_values[at]=lp;best_ids[at]=k;
        }
        CTCBeam *next=(CTCBeam*)calloc(capacity,sizeof(*next));if(!next)goto fail;
        size_t next_count=0;
        for(size_t i=0;i<beam_count;i++){
            CTCBeam *src=&beam[i];double total=beam_logadd(src->blank,src->nonblank);
            if(!beam_add(next,&next_count,capacity,src->tokens,src->length,
                         total+blank_logp,-INFINITY,src->language_score,&src->utf8))goto next_fail;
            for(size_t j=0;j<top_classes;j++){
                uint32_t token=(uint32_t)best_ids[j];double lp=best_values[j];
                if(lp==-INFINITY)continue;
                uint32_t last=src->length?src->tokens[src->length-1]:UINT32_MAX;
                if(token==last){
                    if(!beam_add(next,&next_count,capacity,src->tokens,src->length,
                                 -INFINITY,src->nonblank+lp,src->language_score,&src->utf8))goto next_fail;
                    if(src->blank>-INFINITY&&src->length<max_tokens){
                        Utf8State u=src->utf8;
                        if(utf8_feed(&u,tokenizer->tokens[token].bytes,tokenizer->tokens[token].length)&&
                           (t+1<frames||!u.remaining)){
                            uint32_t *tmp=(uint32_t*)malloc((src->length+1)*sizeof(uint32_t));if(!tmp)goto next_fail;
                            if(src->length)memcpy(tmp,src->tokens,src->length*sizeof(uint32_t));
                            tmp[src->length]=token;
                            int added=beam_add(next,&next_count,capacity,tmp,src->length+1,
                                -INFINITY,src->blank+lp,src->language_score+
                                insertion_bonus+(src->length?lm_weight*language_transition(lm,last,token):0.0),
                                &u);free(tmp);if(!added)goto next_fail;
                        }
                    }
                }else if(src->length<max_tokens){
                    Utf8State u=src->utf8;
                    if(!utf8_feed(&u,tokenizer->tokens[token].bytes,tokenizer->tokens[token].length)||
                       (t+1==frames&&u.remaining))continue;
                    uint32_t *tmp=(uint32_t*)malloc((src->length+1)*sizeof(uint32_t));if(!tmp)goto next_fail;
                    if(src->length)memcpy(tmp,src->tokens,src->length*sizeof(uint32_t));
                    tmp[src->length]=token;
                    int added=beam_add(next,&next_count,capacity,tmp,src->length+1,
                        -INFINITY,total+lp,src->language_score+insertion_bonus+
                        (src->length?lm_weight*language_transition(lm,last,token):0.0),
                        &u);free(tmp);if(!added)goto next_fail;
                }
            }
        }
        for(size_t i=0;i<beam_count;i++)free(beam[i].tokens);
        free(beam);
        qsort(next,next_count,sizeof(*next),beam_score_order);
        if(next_count>beam_width){for(size_t i=beam_width;i<next_count;i++)free(next[i].tokens);next_count=beam_width;}
        beam=next;beam_count=next_count;continue;
next_fail:
        for(size_t i=0;i<next_count;i++)free(next[i].tokens);
        free(next);
        goto fail;
    }
    qsort(beam,beam_count,sizeof(*beam),beam_score_order);
    *decoded=beam[0].tokens;*decoded_count=beam[0].length;beam[0].tokens=NULL;
    for(size_t i=0;i<beam_count;i++)free(beam[i].tokens);
    free(beam);return 1;
fail:
    for(size_t i=0;i<beam_count;i++)free(beam[i].tokens);
    free(beam);return 0;
}
static int write_bytes(const char *path,const unsigned char *bytes,size_t length){
    FILE *f=fopen(path,"wb");if(!f)return 0;
    int ok=fwrite(bytes,1,length,f)==length;
    if(fflush(f)!=0)ok=0;
    if(fclose(f)!=0)ok=0;
    return ok;
}
int main(int argc,char **argv){
    if(argc<5||argc>10){fprintf(stderr,"usage: %s <checkpoint.amm> <audio.wav> <output.txt> <max-tokens> [language] [blank-penalty] [train-manifest.tsv [lm-weight [insertion-bonus]]]\n",argv[0]);return 2;}
    const char *language=argc>=6?argv[5]:"en";
    if(strlen(language)>8){fprintf(stderr,"language code is too long\n");return 2;}
    char *end=NULL;errno=0;unsigned long requested=strtoul(argv[4],&end,10);
    if(errno||end==argv[4]||*end||requested<1||requested>4096){
        fprintf(stderr,"max-tokens must be from 1 to 4096\n");return 2;
    }
    float blank_penalty=argc>=7?strtof(argv[6],&end):0.0f;
    if((argc>=7&&(errno||end==argv[6]||*end))||!isfinite(blank_penalty)||blank_penalty<0.0f||blank_penalty>20.0f){
        fprintf(stderr,"blank penalty must be from 0 to 20\n");return 2;
    }
    double lm_weight=0.0,insertion_bonus=0.0;
    if(argc>=9){errno=0;lm_weight=strtod(argv[8],&end);
        if(errno||end==argv[8]||*end||!isfinite(lm_weight)||lm_weight<0.0||lm_weight>10.0){
            fprintf(stderr,"LM weight must be from 0 to 10\n");return 2;}}
    if(argc>=10){errno=0;insertion_bonus=strtod(argv[9],&end);
        if(errno||end==argv[9]||*end||!isfinite(insertion_bonus)||insertion_bonus< -10.0||insertion_bonus>10.0){
            fprintf(stderr,"insertion bonus must be from -10 to 10\n");return 2;}}
    AMA10Model model={0};AMA10Tokenizer tokenizer={0};AMA10Optimizer optimizer={0};
    AMA10State state={0};AMA10Encoder encoder={0};AMA10CheckpointProgress progress={0};
    CTCLanguageModel lm={0};
    AMA10Audio audio={0};float *patches=NULL,*features=NULL,*logits=NULL;
    uint32_t *generated=NULL,*conditioning=NULL;size_t conditioning_count=0;unsigned char *output=NULL;size_t output_bytes=0;
    int ok=ama10_multimodal_checkpoint_load(argv[1],&model,&tokenizer,&optimizer,
        &state,&progress,&encoder)&&ama10_audio_load_wav16k(&audio,argv[2]);
    if(ok&&argc>=8)ok=language_model_build(&lm,&tokenizer,argv[7],language);
    size_t patch_count=0,patch_width=0;
    if(ok)ok=model.input_width==32&&encoder.input_width==40&&encoder.output_width==model.input_width&&
        ama10_audio_log_mel(&audio,&patches,&patch_count,&patch_width)&&patch_width==encoder.input_width;
    if(ok)ok=patch_count<=SIZE_MAX/encoder.output_width&&
        patch_count*encoder.output_width<=SIZE_MAX/sizeof(float)&&
        patch_count<=SIZE_MAX/model.vocabulary&&
        patch_count*model.vocabulary<=SIZE_MAX/sizeof(float);
    if(ok){
        features=(float*)malloc(patch_count*encoder.output_width*sizeof(float));
        logits=(float*)malloc(patch_count*model.vocabulary*sizeof(float));
        char prefix[32];int prefix_length=snprintf(prefix,sizeof(prefix),"<lang:%s> ",language);
        if(prefix_length<0||(size_t)prefix_length>=sizeof(prefix)||!features||!logits||
           !ama10_tokenizer_encode(&tokenizer,(const unsigned char*)prefix,(size_t)prefix_length,
                                   &conditioning,&conditioning_count)||!conditioning_count||
           !ama10_encoder_forward(&encoder,patches,patch_count,features))ok=0;
    }
    if(ok){
        ama10_state_reset(&state);
        for(size_t i=0;i<conditioning_count&&ok;i++)
            ok=ama10_step(&model,&state,conditioning[i],logits,2,2.0f);
    }
    for(size_t i=0;i<patch_count&&ok;i++)
        ok=ama10_step_features(&model,&state,features+i*encoder.output_width,
                               logits+i*model.vocabulary,2,2.0f);
    size_t generated_count=0;
    if(ok)ok=ctc_prefix_beam(&tokenizer,logits,patch_count,(size_t)requested,
                              blank_penalty,argc>=8?&lm:NULL,lm_weight,insertion_bonus,
                              &generated,&generated_count);
    Utf8State utf8={0,0x80,0xbf};
    for(size_t i=0;i<generated_count&&ok;i++)
        if(!utf8_feed(&utf8,tokenizer.tokens[generated[i]].bytes,
                      tokenizer.tokens[generated[i]].length))ok=0;
    if(ok&&utf8.remaining)ok=0;
    if(ok)ok=ama10_tokenizer_decode(&tokenizer,generated,generated_count,&output,&output_bytes)&&
             write_bytes(argv[3],output,output_bytes);
    if(ok)printf("audio_steps=%zu generated_tokens=%zu bytes=%zu valid_utf8=yes checkpoint_update=%llu output=%s\n",
        patch_count,generated_count,output_bytes,(unsigned long long)progress.completed_updates,argv[3]);
    else fprintf(stderr,"audio-transcription generation failed\n");
    language_model_free(&lm);free(output);free(conditioning);free(generated);free(logits);free(features);free(patches);ama10_audio_free(&audio);
    ama10_encoder_free(&encoder);ama10_state_free(&state);ama10_optimizer_free(&optimizer);
    ama10_tokenizer_free(&tokenizer);ama10_model_free(&model);return ok?0:1;
}
