#include "ama10_encoder.h"
#include "ama10_image.h"
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
static int choose_token(const AMA10Tokenizer *tokenizer,const float *logits,
                        const Utf8State *prefix,int require_complete,uint32_t *selected){
    size_t best=tokenizer->vocabulary_size;float best_logit=-INFINITY;
    for(size_t id=0;id<tokenizer->vocabulary_size;id++){
        Utf8State next=*prefix;
        const AMA10Token *token=&tokenizer->tokens[id];
        if(!utf8_feed(&next,token->bytes,token->length)||
           (require_complete&&next.remaining)||logits[id]<=best_logit)continue;
        best_logit=logits[id];best=id;
    }
    if(best==tokenizer->vocabulary_size)return 0;
    *selected=(uint32_t)best;return 1;
}
static int write_bytes(const char *path,const unsigned char *bytes,size_t length){
    FILE *f=fopen(path,"wb");if(!f)return 0;
    int ok=fwrite(bytes,1,length,f)==length;
    if(fflush(f)!=0)ok=0;
    if(fclose(f)!=0)ok=0;
    return ok;
}
int main(int argc,char **argv){
    if(argc!=5&&argc!=6){fprintf(stderr,"usage: %s <checkpoint.amm> <image.ppm> <output.txt> <max-tokens> [language]\n",argv[0]);return 2;}
    const char *language=argc==6?argv[5]:"en";
    if(strlen(language)>8){fprintf(stderr,"language code is too long\n");return 2;}
    char *end=NULL;errno=0;unsigned long requested=strtoul(argv[4],&end,10);
    if(errno||end==argv[4]||*end||requested<1||requested>4096){
        fprintf(stderr,"max-tokens must be from 1 to 4096\n");return 2;
    }
    AMA10Model model={0};AMA10Tokenizer tokenizer={0};AMA10Optimizer optimizer={0};
    AMA10State state={0};AMA10Encoder encoder={0};AMA10CheckpointProgress progress={0};
    AMA10Image image={0};float *patches=NULL,*features=NULL,*logits=NULL;
    uint32_t *generated=NULL,*conditioning=NULL;size_t conditioning_count=0;unsigned char *output=NULL;size_t output_bytes=0;
    int ok=ama10_multimodal_checkpoint_load(argv[1],&model,&tokenizer,&optimizer,
        &state,&progress,&encoder)&&ama10_image_load(&image,argv[2]);
    size_t patch_count=0,patch_width=0,fine_patch_count=0;
    if(ok)ok=model.input_width==32&&encoder.input_width==768&&encoder.output_width==model.input_width&&
        ama10_image_patchify_multiscale(&image,16,16,12,12,&patches,&patch_count,&patch_width,
                                        &fine_patch_count)&&patch_width==encoder.input_width&&
        patch_count<=SIZE_MAX/encoder.output_width&&
        patch_count*encoder.output_width<=SIZE_MAX/sizeof(float);
    if(ok){
        features=(float*)malloc(patch_count*encoder.output_width*sizeof(float));
        logits=(float*)malloc(model.vocabulary*sizeof(float));
        generated=(uint32_t*)malloc((size_t)requested*sizeof(uint32_t));
        char prefix[32];int prefix_length=snprintf(prefix,sizeof(prefix),"<lang:%s> ",language);
        if(prefix_length<0||(size_t)prefix_length>=sizeof(prefix)||!features||!logits||!generated||
           !ama10_tokenizer_encode(&tokenizer,(const unsigned char*)prefix,(size_t)prefix_length,
                                   &conditioning,&conditioning_count)||!conditioning_count||
           !ama10_encoder_forward(&encoder,patches,patch_count,features))ok=0;
    }
    if(ok){
        ama10_state_reset(&state);
        for(size_t i=0;i<patch_count&&ok;i++)
            ok=ama10_step_features(&model,&state,features+i*encoder.output_width,
                                   logits,2,2.0f);
        for(size_t i=0;i<conditioning_count&&ok;i++)
            ok=ama10_step(&model,&state,conditioning[i],logits,2,2.0f);
    }
    Utf8State utf8={0,0x80,0xbf};
    for(size_t i=0;i<(size_t)requested&&ok;i++){
        uint32_t token=0;
        ok=choose_token(&tokenizer,logits,&utf8,i+1==(size_t)requested,&token);
        if(!ok)break;
        AMA10Token *piece=&tokenizer.tokens[token];
        if(!utf8_feed(&utf8,piece->bytes,piece->length)){ok=0;break;}
        generated[i]=token;
        if(i+1<(size_t)requested)ok=ama10_step(&model,&state,token,logits,2,2.0f);
    }
    if(ok&&utf8.remaining)ok=0;
    if(ok)ok=ama10_tokenizer_decode(&tokenizer,generated,(size_t)requested,&output,&output_bytes)&&
             write_bytes(argv[3],output,output_bytes);
    if(ok)printf("patches=%zu fine_patches=%zu pyramid_views=5 generated_tokens=%lu bytes=%zu valid_utf8=yes checkpoint_update=%llu output=%s\n",
        patch_count,fine_patch_count,requested,output_bytes,(unsigned long long)progress.completed_updates,argv[3]);
    else fprintf(stderr,"image-caption generation failed\n");
    free(output);free(conditioning);free(generated);free(logits);free(features);free(patches);ama10_image_free(&image);
    ama10_encoder_free(&encoder);ama10_state_free(&state);ama10_optimizer_free(&optimizer);
    ama10_tokenizer_free(&tokenizer);ama10_model_free(&model);return ok?0:1;
}
