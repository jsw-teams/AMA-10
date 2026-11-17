#include "checkpoint.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int same_floats(const float *a,const float *b,size_t n){
    return memcmp(a,b,n*sizeof(float))==0;
}
static int same_models(const AMA10Model *a,const AMA10Model *b){
    size_t w=a->width,e=a->input_width,v=a->vocabulary;
    return a->width==b->width&&a->input_width==b->input_width&&a->vocabulary==b->vocabulary&&
        same_floats(a->wx,b->wx,w*e)&&same_floats(a->wh,b->wh,w*w)&&
        same_floats(a->wm,b->wm,w*w)&&same_floats(a->wu,b->wu,w*w)&&
        same_floats(a->bias,b->bias,w)&&same_floats(a->gate_bias,b->gate_bias,w)&&
        same_floats(a->embed,b->embed,e*v)&&same_floats(a->head,b->head,v*w);
}
static int same_optimizers(const AMA10Optimizer *a,const AMA10Optimizer *b,const AMA10Model *m){
    const size_t n[]={m->width*m->input_width,m->width*m->width,m->width*m->width,
        m->width*m->width,m->width,m->width,m->input_width*m->vocabulary,
        m->vocabulary*m->width};
    if(a->step!=b->step||a->parameter_count!=8||b->parameter_count!=8)return 0;
    for(size_t k=0;k<8;k++)
        if(!same_floats(a->first_moment[k],b->first_moment[k],n[k])||
           !same_floats(a->second_moment[k],b->second_moment[k],n[k]))return 0;
    return 1;
}
static int train_one(AMA10Model *m,AMA10Optimizer *o,AMA10State *s,
                     const uint32_t *in,const uint32_t *target){
    float loss=0;
    return ama10_train_sequence(m,o,s,in,target,4,2,0.003f,0.01f,&loss);
}
int main(void){
    static const unsigned char text[]="checkpoint round trip";
    static const uint32_t input[]={65,66,67,68},target[]={66,67,68,69};
    AMA10Tokenizer tok={0},loaded_tok={0};
    AMA10Model full={0},interrupted={0},resumed={0};
    AMA10Optimizer full_opt={0},interrupted_opt={0},resumed_opt={0};
    AMA10State full_state={0},interrupted_state={0},resumed_state={0};
    AMA10CheckpointProgress p={0},loaded_p={0};
    int result=1;
    if(!ama10_tokenizer_train(&tok,text,sizeof(text)-1,256)||
       !ama10_model_init(&full,256,4,4,1234)||!ama10_model_init(&interrupted,256,4,4,1234)||
       !ama10_optimizer_init(&full_opt,&full)||!ama10_optimizer_init(&interrupted_opt,&interrupted)||
       !ama10_state_init(&full_state,4)||!ama10_state_init(&interrupted_state,4))goto done;
    for(size_t i=0;i<5;i++)if(!train_one(&full,&full_opt,&full_state,input,target))goto done;
    for(size_t i=0;i<2;i++)if(!train_one(&interrupted,&interrupted_opt,&interrupted_state,input,target))goto done;
    p.completed_updates=2;p.cursor=8;p.train_hash=ama10_data_hash(text,sizeof(text)-1);
    p.validation_hash=ama10_data_hash(text,sizeof(text)-1);p.sequence_length=4;p.depth=2;
    p.accumulation_steps=1;
    p.tokenizer_sample_bytes=sizeof(text)-1;
    p.target_vocabulary=256;p.learning_rate=0.003f;p.weight_decay=0.01f;
    p.learning_rate_decay=0.0f;
    if(!ama10_checkpoint_save("build/checkpoint.test",&interrupted,&tok,&interrupted_opt,
                              &interrupted_state,&p)||
       !ama10_checkpoint_load("build/checkpoint.test",&resumed,&loaded_tok,&resumed_opt,
                              &resumed_state,&loaded_p))goto done;
    if(loaded_p.completed_updates!=p.completed_updates||loaded_p.cursor!=p.cursor||
       loaded_p.train_hash!=p.train_hash||loaded_p.validation_hash!=p.validation_hash||
       loaded_p.tokenizer_sample_bytes!=p.tokenizer_sample_bytes||
       loaded_p.sequence_length!=p.sequence_length||loaded_p.depth!=p.depth||
       loaded_p.accumulation_steps!=p.accumulation_steps||
       loaded_p.target_vocabulary!=p.target_vocabulary||
       loaded_p.learning_rate_decay!=p.learning_rate_decay||
       loaded_tok.vocabulary_size!=tok.vocabulary_size)goto done;
    for(size_t i=256;i<tok.vocabulary_size;i++)
        if(tok.tokens[i].left!=loaded_tok.tokens[i].left||tok.tokens[i].right!=loaded_tok.tokens[i].right||
           tok.tokens[i].length!=loaded_tok.tokens[i].length||
           memcmp(tok.tokens[i].bytes,loaded_tok.tokens[i].bytes,tok.tokens[i].length))goto done;
    for(size_t i=2;i<5;i++)
        if(!train_one(&interrupted,&interrupted_opt,&interrupted_state,input,target)||
           !train_one(&resumed,&resumed_opt,&resumed_state,input,target))goto done;
    if(!same_models(&full,&interrupted)||!same_models(&full,&resumed)||
       !same_optimizers(&full_opt,&resumed_opt,&full)||
       !same_floats(full_state.fast,resumed_state.fast,4)||
       !same_floats(full_state.slow,resumed_state.slow,4)||
       full_state.steps!=resumed_state.steps||full_opt.step!=resumed_opt.step)goto done;
    {
        AMA10Model uninterrupted={0},saved_model={0},continued={0};
        AMA10Optimizer uninterrupted_opt={0},saved_opt={0},continued_opt={0};
        AMA10State uninterrupted_state={0},saved_state={0},continued_state={0};
        AMA10Tokenizer grown_tok={0},saved_tok={0};AMA10CheckpointProgress muon_p={0},loaded_muon_p={0};
        int muon_ok=ama10_model_init(&uninterrupted,256,4,4,9876)&&
            ama10_model_init(&saved_model,256,4,4,9876)&&
            ama10_optimizer_init(&uninterrupted_opt,&uninterrupted)&&
            ama10_optimizer_init(&saved_opt,&saved_model)&&
            ama10_optimizer_set_algorithm(&uninterrupted_opt,AMA10_OPT_MUON)&&
            ama10_optimizer_set_algorithm(&saved_opt,AMA10_OPT_MUON)&&
            ama10_state_init(&uninterrupted_state,4)&&ama10_state_init(&saved_state,4);
        if(muon_ok)muon_ok=train_one(&uninterrupted,&uninterrupted_opt,&uninterrupted_state,input,target)&&
            train_one(&saved_model,&saved_opt,&saved_state,input,target);
        static const unsigned char growth_unit[]={'z','q','v'};
        static const unsigned char image_code_marker[]="<|image_code_000|>";
        static const unsigned char image_eos_marker[]="<|image_eos|>";
        unsigned char growth_text[sizeof(growth_unit)*8];
        uint32_t image_code_id=0,image_eos_id=0;
        for(size_t i=0;i<8;i++)memcpy(growth_text+i*sizeof(growth_unit),growth_unit,sizeof(growth_unit));
        if(muon_ok)muon_ok=ama10_tokenizer_train(&grown_tok,text,sizeof(text)-1,256)&&
            ama10_tokenizer_extend(&grown_tok,growth_text,sizeof(growth_text),272)&&grown_tok.vocabulary_size>256&&
            ama10_tokenizer_add_special(&grown_tok,image_code_marker,sizeof(image_code_marker)-1,&image_code_id)&&
            ama10_tokenizer_add_special(&grown_tok,image_eos_marker,sizeof(image_eos_marker)-1,&image_eos_id);
        size_t old_vocab=256;
        AMA10Model old_shape={0};old_shape.width=4;old_shape.input_width=4;old_shape.vocabulary=old_vocab;
        if(muon_ok)muon_ok=ama10_model_resize_vocabulary(&uninterrupted,grown_tok.vocabulary_size,4321)&&
            ama10_model_resize_vocabulary(&saved_model,grown_tok.vocabulary_size,4321)&&
            ama10_optimizer_resize_vocabulary(&uninterrupted_opt,&old_shape,&uninterrupted)&&
            ama10_optimizer_resize_vocabulary(&saved_opt,&old_shape,&saved_model);
        muon_p.completed_updates=1;muon_p.cursor=4;muon_p.train_hash=ama10_data_hash(text,sizeof(text)-1);
        muon_p.validation_hash=muon_p.train_hash;muon_p.sequence_length=4;muon_p.depth=2;
        muon_p.accumulation_steps=1;muon_p.optimizer_algorithm=AMA10_OPT_MUON;
        muon_p.tokenizer_sample_bytes=sizeof(text)-1;muon_p.target_vocabulary=(uint32_t)grown_tok.vocabulary_size;
        muon_p.learning_rate=0.003f;muon_p.learning_rate_decay=0.0f;muon_p.weight_decay=0.01f;
        uint32_t grown_input[]={271,65,66,271},grown_target[]={65,66,271,65};
        grown_input[0]=image_code_id;
        grown_input[3]=image_eos_id;grown_target[2]=image_eos_id;
        if(muon_ok)muon_ok=ama10_checkpoint_save("build/muon-checkpoint.test",&saved_model,&grown_tok,&saved_opt,
                                                  &saved_state,&muon_p)&&
            ama10_checkpoint_load("build/muon-checkpoint.test",&continued,&saved_tok,&continued_opt,
                                  &continued_state,&loaded_muon_p)&&
            continued_opt.algorithm==AMA10_OPT_MUON&&loaded_muon_p.optimizer_algorithm==AMA10_OPT_MUON&&
            saved_tok.vocabulary_size==grown_tok.vocabulary_size&&
            saved_tok.tokens[image_code_id].special&&saved_tok.tokens[image_eos_id].special&&
            saved_tok.tokens[image_code_id].left==UINT32_MAX&&saved_tok.tokens[image_eos_id].right==UINT32_MAX&&
            ama10_train_sequence(&uninterrupted,&uninterrupted_opt,&uninterrupted_state,
                                 grown_input,grown_target,4,2,0.003f,0.01f,NULL)&&
            ama10_train_sequence(&continued,&continued_opt,&continued_state,
                                 grown_input,grown_target,4,2,0.003f,0.01f,NULL)&&
            same_models(&uninterrupted,&continued)&&same_optimizers(&uninterrupted_opt,&continued_opt,&uninterrupted)&&
            same_floats(uninterrupted_state.fast,continued_state.fast,4)&&
            same_floats(uninterrupted_state.slow,continued_state.slow,4);
        ama10_state_free(&uninterrupted_state);ama10_state_free(&saved_state);ama10_state_free(&continued_state);
        ama10_optimizer_free(&uninterrupted_opt);ama10_optimizer_free(&saved_opt);ama10_optimizer_free(&continued_opt);
        ama10_model_free(&uninterrupted);ama10_model_free(&saved_model);ama10_model_free(&continued);
        ama10_tokenizer_free(&grown_tok);ama10_tokenizer_free(&saved_tok);
        if(!muon_ok){fprintf(stderr,"dynamic vocabulary growth or Muon checkpoint continuation failed\n");goto done;}
    }
    puts("checkpoint exact training-resume round trip: ok");result=0;
done:
    if(result)fprintf(stderr,"checkpoint exact training-resume round trip: failed\n");
    ama10_state_free(&full_state);ama10_state_free(&interrupted_state);ama10_state_free(&resumed_state);
    ama10_optimizer_free(&full_opt);ama10_optimizer_free(&interrupted_opt);ama10_optimizer_free(&resumed_opt);
    ama10_model_free(&full);ama10_model_free(&interrupted);ama10_model_free(&resumed);
    ama10_tokenizer_free(&tok);ama10_tokenizer_free(&loaded_tok);
    return result;
}
