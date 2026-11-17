#include "multimodal_checkpoint.h"
#include "ama10_encoder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int corrupt_last_byte(const char *path){
    FILE *f=fopen(path,"r+b");if(!f)return 0;
    int ok=fseek(f,-1,SEEK_END)==0;int c=ok?fgetc(f):EOF;
    if(c==EOF||fseek(f,-1,SEEK_CUR)!=0||fputc(c^0x5a,f)==EOF)ok=0;
    if(fclose(f)!=0)ok=0;
    return ok;
}
int main(void){
    static const unsigned char sample[]="multimodal checkpoint sample";
    AMA10Tokenizer tokenizer={0},loaded_tokenizer={0};
    AMA10Model model={0},loaded_model={0};
    AMA10Optimizer optimizer={0},loaded_optimizer={0};
    AMA10State state={0},loaded_state={0};
    AMA10Encoder encoder={0},loaded_encoder={0};
    AMA10CheckpointProgress progress={0},loaded_progress={0};
    const uint32_t input[]={1,2,3,4},target[]={2,3,4,5};
    float encoder_input[]={0.1f,0.2f,0.3f,0.4f};
    float encoder_features[2],encoder_gradient[]={0.2f,-0.1f,0.3f,0.05f};
    int ok=ama10_tokenizer_train(&tokenizer,sample,sizeof(sample)-1,270)&&
        ama10_model_init(&model,tokenizer.vocabulary_size,4,4,101)&&
        ama10_optimizer_init(&optimizer,&model)&&ama10_state_init(&state,model.width)&&
        ama10_encoder_init(&encoder,2,4,202)&&
        ama10_train_sequence(&model,&optimizer,&state,input,target,4,2,0.001f,0.01f,NULL);
    if(ok)ok=ama10_encoder_forward(&encoder,encoder_input,2,encoder_features)&&
        ama10_encoder_backward_update(&encoder,encoder_input,encoder_features,
                                      encoder_gradient,2,0.001f,0.01f,NULL);
    progress.completed_updates=optimizer.step;progress.cursor=4;
    progress.train_hash=11;progress.validation_hash=22;progress.tokenizer_sample_bytes=sizeof(sample)-1;
    progress.sequence_length=4;progress.depth=2;progress.accumulation_steps=1;
    progress.target_vocabulary=(uint32_t)tokenizer.vocabulary_size;
    progress.optimizer_algorithm=optimizer.algorithm;progress.learning_rate=0.001f;
    progress.learning_rate_decay=0.01f;progress.weight_decay=0.01f;
    if(ok)ok=ama10_multimodal_checkpoint_save("build/multimodal-checkpoint.test",&model,
        &tokenizer,&optimizer,&state,&progress,&encoder)&&
        ama10_multimodal_checkpoint_load("build/multimodal-checkpoint.test",&loaded_model,
        &loaded_tokenizer,&loaded_optimizer,&loaded_state,&loaded_progress,&loaded_encoder);
    if(ok)ok=loaded_model.vocabulary==model.vocabulary&&
        loaded_optimizer.step==optimizer.step&&loaded_encoder.step==encoder.step&&
        loaded_progress.completed_updates==progress.completed_updates&&
        loaded_progress.cursor==progress.cursor&&
        memcmp(loaded_model.wx,model.wx,model.width*model.input_width*sizeof(float))==0&&
        memcmp(loaded_optimizer.first_moment[0],optimizer.first_moment[0],
               model.width*model.input_width*sizeof(float))==0&&
        memcmp(loaded_state.fast,state.fast,model.width*sizeof(float))==0&&
        memcmp(loaded_encoder.weights,encoder.weights,
               encoder.input_width*encoder.output_width*sizeof(float))==0&&
        memcmp(loaded_encoder.first_moment,encoder.first_moment,
               encoder.input_width*encoder.output_width*sizeof(float))==0;
    float preserved_model=ok?loaded_model.wx[0]:0.0f;
    float preserved_encoder=ok?loaded_encoder.weights[0]:0.0f;
    uint64_t preserved_update=loaded_progress.completed_updates;
    if(ok)ok=corrupt_last_byte("build/multimodal-checkpoint.test")&&
        !ama10_multimodal_checkpoint_load("build/multimodal-checkpoint.test",&loaded_model,
            &loaded_tokenizer,&loaded_optimizer,&loaded_state,&loaded_progress,&loaded_encoder)&&
        loaded_model.wx[0]==preserved_model&&loaded_encoder.weights[0]==preserved_encoder&&
        loaded_progress.completed_updates==preserved_update;
    ama10_encoder_free(&loaded_encoder);ama10_state_free(&loaded_state);
    ama10_optimizer_free(&loaded_optimizer);ama10_model_free(&loaded_model);
    ama10_tokenizer_free(&loaded_tokenizer);
    ama10_encoder_free(&encoder);ama10_state_free(&state);
    ama10_optimizer_free(&optimizer);ama10_model_free(&model);ama10_tokenizer_free(&tokenizer);
    if(!ok){fprintf(stderr,"multimodal atomic checkpoint round trip/corruption test failed\n");return 1;}
    puts("single-file multimodal checkpoint preserves model, optimizer, state, tokenizer and encoder; checksum corruption rejected without replacing loaded state: ok");
    return 0;
}
