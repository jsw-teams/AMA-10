#include "ama10.h"
#include "ama10_encoder.h"
#include "ama10_image.h"
#include "multimodal_checkpoint.h"
#include "tokenizer.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *path;
    char *caption;
    char language[16];
    int validation;
    AMA10Image image;
    float *patches;
    size_t patch_count,patch_width,fine_patch_count;
    uint32_t *tokens;
    size_t token_count;
    uint32_t *conditioning_tokens;
    size_t conditioning_count;
} CaptionItem;

static void items_free(CaptionItem *items,size_t count){
    if(!items)return;
    for(size_t i=0;i<count;i++){
        free(items[i].path);free(items[i].caption);free(items[i].patches);
        free(items[i].tokens);free(items[i].conditioning_tokens);ama10_image_free(&items[i].image);
    }
    free(items);
}
static char *copy_string(const char *s){
    size_t n=strlen(s);char *p=(char*)malloc(n+1);
    if(p)memcpy(p,s,n+1);
    return p;
}
static int add_item(CaptionItem **items,size_t *count,size_t *capacity,CaptionItem *item){
    if(*count==*capacity){
        size_t next=*capacity?*capacity*2:8;
        if(next<*capacity||next>SIZE_MAX/sizeof(**items))return 0;
        CaptionItem *p=(CaptionItem*)realloc(*items,next*sizeof(**items));if(!p)return 0;
        *items=p;*capacity=next;
    }
    (*items)[(*count)++]=*item;memset(item,0,sizeof(*item));return 1;
}
static int read_manifest(const char *path,CaptionItem **out,size_t *out_count,
                         size_t *train_count,size_t *validation_count){
    FILE *f=fopen(path,"rb");if(!f)return 0;
    char line[16384];CaptionItem *items=NULL;size_t count=0,capacity=0;
    size_t train=0,validation=0;int ok=1;
    while(fgets(line,sizeof(line),f)){
        size_t n=strlen(line);if(n&&line[n-1]=='\n')line[--n]='\0';
        if(n&&line[n-1]=='\r')line[--n]='\0';
        if(!n)continue;
        char *first=strchr(line,'\t');if(!first){ok=0;break;}*first++='\0';
        char *second=strchr(first,'\t');if(!second){ok=0;break;}*second++='\0';
        char *language=strchr(second,'\t');
        if(language){*language++='\0';if(strchr(language,'\t')){ok=0;break;}}
        else language="en";
        if(!*line||!*first||!*second||!*language){ok=0;break;}
        CaptionItem item={0};item.path=copy_string(line);item.caption=copy_string(first);
        item.validation=!strcmp(second,"validation");
        if(strcmp(second,"train")&&strcmp(second,"validation")){free(item.path);free(item.caption);ok=0;break;}
        size_t language_length=strlen(language);
        if(language_length>=sizeof(item.language)){free(item.path);free(item.caption);ok=0;break;}
        memcpy(item.language,language,language_length+1);
        if(!item.path||!item.caption||!add_item(&items,&count,&capacity,&item)){
            free(item.path);free(item.caption);ok=0;break;
        }
        if(items[count-1].validation)validation++;else train++;
    }
    if(ferror(f))ok=0;
    fclose(f);
    if(!ok||!count||!train||!validation){items_free(items,count);return 0;}
    *out=items;*out_count=count;*train_count=train;*validation_count=validation;return 1;
}
static int join_path(const char *manifest,const char *relative,char **out){
    const char *slash1=strrchr(manifest,'/'),*slash2=strrchr(manifest,'\\');
    const char *slash=slash1;
    if(slash2&&(!slash||slash2>slash))slash=slash2;
    size_t dir=slash?(size_t)(slash-manifest+1):0,n=strlen(relative);
    if(dir>SIZE_MAX-n-1)return 0;
    char *p=(char*)malloc(dir+n+1);if(!p)return 0;
    if(dir)memcpy(p,manifest,dir);
    memcpy(p+dir,relative,n+1);*out=p;return 1;
}
static int prepare_items(CaptionItem *items,size_t count,const char *manifest,
                         AMA10Tokenizer *tokenizer,int learn_tokenizer,
                         size_t *train_bytes){
    size_t total=0;
    for(size_t i=0;i<count;i++)if(!items[i].validation){
        size_t n=strlen(items[i].caption)+strlen(items[i].language)+10;
        if(total>SIZE_MAX-n-1)return 0;
        total+=n+1;
    }
    unsigned char *text=(unsigned char*)malloc(total);if(!text)return 0;
    size_t pos=0;
    for(size_t i=0;i<count;i++)if(!items[i].validation){
        int prefix=snprintf((char*)text+pos,total-pos,"<lang:%s> ",items[i].language);
        if(prefix<0)return 0;
        pos+=(size_t)prefix;
        size_t n=strlen(items[i].caption);memcpy(text+pos,items[i].caption,n);pos+=n;text[pos++]='\n';
    }
    int ok=1;
    if(learn_tokenizer)ok=ama10_tokenizer_train(tokenizer,text,total,512);
    free(text);if(!ok||!tokenizer->vocabulary_size)return 0;
    *train_bytes=total;
    for(size_t i=0;i<count;i++){
        char *full=NULL;
        if(!join_path(manifest,items[i].path,&full))return 0;
        ok=ama10_image_load(&items[i].image,full);free(full);
        if(!ok||!ama10_image_patchify_multiscale(&items[i].image,16,16,12,12,
                                 &items[i].patches,&items[i].patch_count,
                                 &items[i].patch_width,&items[i].fine_patch_count)||
           !items[i].patch_count||items[i].patch_width!=768)return 0;
        const unsigned char *caption=(const unsigned char*)items[i].caption;
        char prefix[32];int prefix_length=snprintf(prefix,sizeof(prefix),"<lang:%s> ",items[i].language);
        if(prefix_length<0||(size_t)prefix_length>=sizeof(prefix)||
           !ama10_tokenizer_encode(tokenizer,(const unsigned char*)prefix,(size_t)prefix_length,
              &items[i].conditioning_tokens,&items[i].conditioning_count)||!items[i].conditioning_count)return 0;
        if(!ama10_tokenizer_encode(tokenizer,caption,strlen(items[i].caption),
                                   &items[i].tokens,&items[i].token_count)||!items[i].token_count)return 0;
    }
    return 1;
}
static double token_nll(const float *logits,size_t vocabulary,uint32_t target){
    float maximum=logits[0];
    for(size_t i=1;i<vocabulary;i++)if(logits[i]>maximum)maximum=logits[i];
    double sum=0.0;
    for(size_t i=0;i<vocabulary;i++)sum+=exp((double)logits[i]-(double)maximum);
    if(!(sum>0.0)||!isfinite(sum))return NAN;
    return (double)maximum+log(sum)-(double)logits[target];
}
static uint64_t hash_update(uint64_t hash,const unsigned char *bytes,size_t count){
    for(size_t i=0;i<count;i++){hash^=bytes[i];hash*=UINT64_C(1099511628211);}
    return hash;
}
static uint64_t hash_split(const CaptionItem *items,size_t count,int validation){
    uint64_t hash=UINT64_C(14695981039346656037);
    for(size_t i=0;i<count;i++)if(items[i].validation==validation){
        hash=hash_update(hash,(const unsigned char*)items[i].caption,strlen(items[i].caption));
        const unsigned char delimiter=0;hash=hash_update(hash,&delimiter,1);
        hash=hash_update(hash,(const unsigned char*)items[i].language,strlen(items[i].language));
        hash=hash_update(hash,&delimiter,1);
        uint64_t dimensions[2]={(uint64_t)items[i].image.width,(uint64_t)items[i].image.height};
        hash=hash_update(hash,(const unsigned char*)dimensions,sizeof(dimensions));
        hash=hash_update(hash,items[i].image.rgb,items[i].image.width*items[i].image.height*3);
    }
    return hash;
}
static int caption_loss(const AMA10Model *model,const AMA10Encoder *encoder,
                        const CaptionItem *item,double *loss){
    AMA10State state={0};float *features=NULL,*logits=NULL;
    if(!ama10_state_init(&state,model->width)||
       item->patch_count>SIZE_MAX/encoder->output_width||
       item->patch_count*encoder->output_width>SIZE_MAX/sizeof(float))goto fail;
    features=(float*)malloc(item->patch_count*encoder->output_width*sizeof(float));
    logits=(float*)malloc(model->vocabulary*sizeof(float));
    if(!features||!logits||!ama10_encoder_forward(encoder,item->patches,item->patch_count,features))goto fail;
    double sum=0.0;
    for(size_t i=0;i<item->patch_count;i++){
        if(!ama10_step_features(model,&state,features+i*encoder->output_width,logits,2,2.0f))goto fail;
        if(i+1==item->patch_count){
        }
    }
    for(size_t i=0;i<item->conditioning_count;i++)
        if(!ama10_step(model,&state,item->conditioning_tokens[i],logits,2,2.0f))goto fail;
    { double value=token_nll(logits,model->vocabulary,item->tokens[0]); if(!isfinite(value))goto fail; sum+=value; }
    for(size_t i=0;i+1<item->token_count;i++){
        if(!ama10_step(model,&state,item->tokens[i],logits,2,2.0f))goto fail;
        double value=token_nll(logits,model->vocabulary,item->tokens[i+1]);
        if(!isfinite(value))goto fail;
        sum+=value;
    }
    *loss=sum/(double)item->token_count;free(features);free(logits);ama10_state_free(&state);return 1;
fail:
    free(features);free(logits);ama10_state_free(&state);return 0;
}
static int evaluate(const AMA10Model *model,const AMA10Encoder *encoder,
                    const CaptionItem *items,size_t count,double *mean,uint64_t *target_count){
    double total=0.0;uint64_t tokens=0;
    for(size_t i=0;i<count;i++)if(items[i].validation){
        double loss;if(!caption_loss(model,encoder,&items[i],&loss))return 0;
        if(items[i].token_count>UINT64_MAX-tokens)return 0;
        total+=loss*(double)items[i].token_count;tokens+=(uint64_t)items[i].token_count;
    }
    if(!tokens)return 0;
    *mean=total/(double)tokens;if(target_count)*target_count=tokens;
    return isfinite(*mean);
}
static int evaluate_language(const AMA10Model *model,const AMA10Encoder *encoder,
                             const CaptionItem *items,size_t count,const char *language,
                             double *mean,uint64_t *target_count){
    double total=0.0;uint64_t tokens=0;
    for(size_t i=0;i<count;i++)if(items[i].validation&&!strcmp(items[i].language,language)){
        double loss;if(!caption_loss(model,encoder,&items[i],&loss))return 0;
        if(items[i].token_count>UINT64_MAX-tokens)return 0;
        total+=loss*(double)items[i].token_count;tokens+=(uint64_t)items[i].token_count;
    }
    if(!tokens)return 0;
    *mean=total/(double)tokens;if(target_count)*target_count=tokens;
    return isfinite(*mean);
}
static int print_language_losses(const AMA10Model *model,const AMA10Encoder *encoder,
                                 const CaptionItem *items,size_t count,const char *phase){
    for(size_t i=0;i<count;i++)if(items[i].validation){
        int first=1;
        for(size_t j=0;j<i;j++)if(items[j].validation&&
            !strcmp(items[j].language,items[i].language)){first=0;break;}
        if(!first)continue;
        double loss;uint64_t tokens;
        if(!evaluate_language(model,encoder,items,count,items[i].language,&loss,&tokens))return 0;
        printf("%s_language=%s caption_token_nll=%.6f tokens=%llu\n",phase,items[i].language,
               loss,(unsigned long long)tokens);
    }
    return 1;
}
int main(int argc,char **argv){
    int resume=argc==6&&!strcmp(argv[4],"--resume");
    if(argc!=4&&!resume){fprintf(stderr,"usage: %s <manifest.tsv> <total-updates> <output-prefix> [--resume input.amm]\n",argv[0]);return 2;}
    char *end=NULL;errno=0;unsigned long long requested=strtoull(argv[2],&end,10);
    if(errno||end==argv[2]||*end||!requested||requested>SIZE_MAX){fprintf(stderr,"invalid update count\n");return 2;}
    CaptionItem *items=NULL;size_t count=0,train_count=0,validation_count=0,train_bytes=0;
    AMA10Tokenizer tokenizer={0};AMA10Model model={0};AMA10Optimizer optimizer={0};
    AMA10Encoder encoder={0};AMA10State state={0};int ok=0;
    AMA10CheckpointProgress progress={0};
    if(!read_manifest(argv[1],&items,&count,&train_count,&validation_count))goto done;
    if(resume&&!ama10_multimodal_checkpoint_load(argv[5],&model,&tokenizer,&optimizer,
                                                  &state,&progress,&encoder))goto done;
    if(!prepare_items(items,count,argv[1],&tokenizer,!resume,&train_bytes))goto done;
    if(!resume){
        if(!ama10_model_init(&model,tokenizer.vocabulary_size,32,32,UINT64_C(20261117))||
           !ama10_optimizer_init(&optimizer,&model)||!ama10_state_init(&state,model.width)||
           !ama10_encoder_init(&encoder,768,32,UINT64_C(20261118)))goto done;
    }
    if(model.width!=32||model.input_width!=32||model.vocabulary!=tokenizer.vocabulary_size||
       encoder.input_width!=768||encoder.output_width!=32||
       (resume&&optimizer.step!=progress.completed_updates))goto done;
    size_t max_sequence=0;
    for(size_t i=0;i<count;i++){
        size_t n=items[i].patch_count;
        if(n>SIZE_MAX-items[i].conditioning_count||
           n+items[i].conditioning_count>SIZE_MAX-items[i].token_count)goto done;
        n+=items[i].conditioning_count+items[i].token_count;
        if(max_sequence<n)max_sequence=n;
    }
    if(max_sequence>UINT32_MAX)goto done;
    uint64_t train_hash=hash_split(items,count,0),validation_hash=hash_split(items,count,1);
    if(resume){
        if(progress.train_hash!=train_hash||progress.validation_hash!=validation_hash||
           progress.tokenizer_sample_bytes!=(uint64_t)train_bytes||
           progress.sequence_length!=(uint32_t)max_sequence||progress.depth!=2||
           progress.accumulation_steps!=1||progress.target_vocabulary!=512||
           progress.optimizer_algorithm!=optimizer.algorithm||optimizer.algorithm!=AMA10_OPT_ADAMW||
           progress.learning_rate!=0.001f||progress.learning_rate_decay!=0.01f||
           progress.weight_decay!=0.01f||progress.cursor!=progress.completed_updates%(uint64_t)train_count||
           requested<=progress.completed_updates)goto done;
    }else{
        progress.train_hash=train_hash;progress.validation_hash=validation_hash;
        progress.tokenizer_sample_bytes=(uint64_t)train_bytes;
        progress.sequence_length=(uint32_t)max_sequence;progress.depth=2;
        progress.accumulation_steps=1;progress.target_vocabulary=512;
        progress.optimizer_algorithm=optimizer.algorithm;progress.learning_rate=0.001f;
        progress.learning_rate_decay=0.01f;progress.weight_decay=0.01f;
    }
    double before=0.0,after=0.0;uint64_t validation_tokens=0,train_tokens=0;
    if(!evaluate(&model,&encoder,items,count,&before,&validation_tokens)||
       !print_language_losses(&model,&encoder,items,count,"validation_before"))goto done;
    size_t train_seen=0;
    for(size_t i=0;i<count;i++)if(!items[i].validation){
        train_seen++;
        if(items[i].token_count>UINT64_MAX-train_tokens)goto done;
        train_tokens+=(uint64_t)items[i].token_count;
    }
    uint64_t completed=progress.completed_updates;
    for(;completed<(uint64_t)requested;completed++){
        size_t pick=(size_t)(completed%(uint64_t)train_seen),index=0;
        for(;index<count;index++)if(!items[index].validation){if(!pick)break;pick--;}
        float loss=0.0f;
        float lr=ama10_learning_rate(0.001f,completed+1,0.01f);
        if(index==count||!ama10_train_image_caption_conditioned(&model,&optimizer,&state,&encoder,
            items[index].patches,items[index].patch_count,items[index].conditioning_tokens,
            items[index].conditioning_count,items[index].tokens,
            items[index].token_count,2,lr,0.01f,&loss)||!isfinite(loss))goto done;
        if((completed+1)%100==0)printf("update=%llu train_loss=%.5f lr=%.7g\n",
            (unsigned long long)(completed+1),(double)loss,(double)lr);
    }
    if(!evaluate(&model,&encoder,items,count,&after,NULL)||
       !print_language_losses(&model,&encoder,items,count,"validation_after"))goto done;
    progress.completed_updates=optimizer.step;
    progress.cursor=progress.completed_updates%(uint64_t)train_seen;
    progress.train_hash=train_hash;
    progress.validation_hash=validation_hash;
    progress.tokenizer_sample_bytes=(uint64_t)train_bytes;
    progress.sequence_length=(uint32_t)max_sequence;
    progress.depth=2;progress.accumulation_steps=1;
    progress.target_vocabulary=512;progress.optimizer_algorithm=optimizer.algorithm;
    progress.learning_rate=0.001f;progress.learning_rate_decay=0.01f;
    progress.weight_decay=0.01f;
    size_t prefix_length=strlen(argv[3]);
    if(prefix_length>SIZE_MAX-5)goto done;
    char *bundle_path=(char*)malloc(prefix_length+5);
    if(!bundle_path)goto done;
    snprintf(bundle_path,prefix_length+5,"%s.amm",argv[3]);
    ok=ama10_multimodal_checkpoint_save(bundle_path,&model,&tokenizer,&optimizer,&state,
                                        &progress,&encoder);
    if(ok){
        AMA10Model verify_model={0};AMA10Tokenizer verify_tokenizer={0};
        AMA10Optimizer verify_optimizer={0};AMA10State verify_state={0};
        AMA10Encoder verify_encoder={0};AMA10CheckpointProgress verify_progress={0};
        double persisted_loss=0.0;
        ok=ama10_multimodal_checkpoint_load(bundle_path,&verify_model,&verify_tokenizer,
            &verify_optimizer,&verify_state,&verify_progress,&verify_encoder)&&
            verify_progress.completed_updates==progress.completed_updates&&
            verify_optimizer.step==optimizer.step&&
            evaluate(&verify_model,&verify_encoder,items,count,&persisted_loss,NULL)&&
            fabs(persisted_loss-after)<1.0e-6;
        ama10_encoder_free(&verify_encoder);ama10_state_free(&verify_state);
        ama10_optimizer_free(&verify_optimizer);ama10_model_free(&verify_model);
        ama10_tokenizer_free(&verify_tokenizer);
        if(ok)printf("reloaded_validation_caption_token_nll=%.6f\n",persisted_loss);
    }
    free(bundle_path);if(!ok)goto done;
    printf("pairs=%zu train=%zu validation=%zu train_caption_bytes=%zu train_tokens=%llu validation_tokens=%llu vocabulary=%zu updates=%llu\n",
        count,train_count,validation_count,train_bytes,(unsigned long long)train_tokens,
        (unsigned long long)validation_tokens,tokenizer.vocabulary_size,requested);
    printf("validation_caption_token_nll_before=%.6f after=%.6f\n",before,after);
    puts("saved one checksummed multimodal checkpoint with model, tokenizer, optimizer, recurrent state, and encoder");
done:
    if(!ok)fprintf(stderr,"image-caption training failed\n");
    ama10_state_free(&state);ama10_encoder_free(&encoder);ama10_optimizer_free(&optimizer);
    ama10_model_free(&model);ama10_tokenizer_free(&tokenizer);items_free(items,count);
    return ok?0:1;
}
