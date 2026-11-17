#include "ama10.h"
#include "ama10_encoder.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double projected_objective(const AMA10Encoder *encoder,const float *inputs,
                                  const float *gradient,size_t samples){
    float *features=(float*)malloc(samples*encoder->output_width*sizeof(float));
    if(!features)return NAN;
    if(!ama10_encoder_forward(encoder,inputs,samples,features)){free(features);return NAN;}
    double value=0.0;
    for(size_t i=0;i<samples*encoder->output_width;i++)value+=(double)features[i]*gradient[i];
    free(features);return value;
}

int main(void){
    AMA10Encoder encoder={0},loaded={0};
    const float raw[]={0.2f,-0.4f,0.7f, 0.9f,0.1f,-0.3f};
    const float output_gradient[]={0.3f,-0.7f,0.2f,0.8f};
    float features[4],input_gradients[6];
    if(!ama10_encoder_init(&encoder,3,2,721) ||
       !ama10_encoder_forward(&encoder,raw,2,features))return 1;
    size_t index=0;double best=0.0,numeric_best=0.0;
    for(size_t i=0;i<6;i++){
        float changed[6];memcpy(changed,raw,sizeof(raw));changed[i]+=0.001f;
        double plus=projected_objective(&encoder,changed,output_gradient,2);
        memcpy(changed,raw,sizeof(raw));changed[i]-=0.001f;
        double minus=projected_objective(&encoder,changed,output_gradient,2);
        double grad=(plus-minus)/0.002;
        if(fabs(grad)>best){best=fabs(grad);numeric_best=grad;index=i;}
    }
    float old_weight=encoder.weights[0],old_bias=encoder.bias[0];
    if(!ama10_encoder_backward_update(&encoder,raw,features,output_gradient,2,
                                     0.001f,0.0f,input_gradients)||
       encoder.step!=1||encoder.weights[0]==old_weight||encoder.bias[0]==old_bias||
       fabs(input_gradients[index])<1.0e-6||!isfinite(input_gradients[index])||
       input_gradients[index]*numeric_best<=0.0||
       fabs((double)input_gradients[index]-numeric_best)>0.02*fabs(numeric_best)+1.0e-4)return 2;
    if(!ama10_encoder_save(&encoder,"build/encoder.test")||
       !ama10_encoder_load(&loaded,"build/encoder.test")||loaded.step!=encoder.step||
       loaded.input_width!=encoder.input_width||loaded.output_width!=encoder.output_width||
       memcmp(loaded.weights,encoder.weights,encoder.input_width*encoder.output_width*sizeof(float))||
       memcmp(loaded.first_moment,encoder.first_moment,encoder.input_width*encoder.output_width*sizeof(float)))return 3;
    {
        AMA10Model model={0};AMA10Optimizer opt={0};AMA10Encoder joint={0};
        const float frame_data[]={0.0f,1.0f,0.5f, 1.0f,0.1f,0.0f,
                                  0.2f,0.8f,1.0f, 1.0f,0.0f,0.3f};
        const uint32_t targets[]={1,2,3,4};float loss=0.0f;
        if(!ama10_model_init(&model,8,4,2,9123)||!ama10_optimizer_init(&opt,&model)||
           !ama10_encoder_init(&joint,3,2,2718)||
           !ama10_train_encoded_sequence(&model,&opt,NULL,&joint,frame_data,targets,4,2,
                                         0.001f,0.0f,&loss)||
           opt.step!=1||joint.step!=1||!isfinite(loss)||
           model.wx[0]==0.0f||joint.weights[0]==0.0f){
            fprintf(stderr,"joint encoder and recurrent training failed\n");return 4;
        }
        ama10_encoder_free(&joint);ama10_optimizer_free(&opt);ama10_model_free(&model);
    }
    {
        AMA10Model model={0};AMA10Optimizer opt={0};AMA10Encoder image_encoder={0};
        const float image_frames[]={0.0f,1.0f,0.2f,0.8f, 1.0f,0.0f,0.7f,0.1f};
        float mixed_features[16]={0},mixed_feature_gradients[16]={0};
        const uint32_t token_inputs[]={0,0,1,2},targets[]={0,1,2,3};
        const unsigned char feature_mask[]={1,1,0,0},loss_mask[]={0,1,1,1};float loss=0.0f;
        if(!ama10_model_init(&model,8,4,4,931)||!ama10_optimizer_init(&opt,&model)||
           !ama10_encoder_init(&image_encoder,4,4,1931)||
           !ama10_encoder_forward(&image_encoder,image_frames,2,mixed_features))return 5;
        if(!ama10_train_multimodal_sequence(&model,&opt,NULL,token_inputs,mixed_features,
                                            feature_mask,loss_mask,targets,4,2,0.001f,0.0f,
                                            mixed_feature_gradients,&loss)||opt.step!=1||
           !isfinite(loss))return 6;
        float image_gradient_norm=0.0f;
        for(size_t i=0;i<8;i++)image_gradient_norm+=fabsf(mixed_feature_gradients[i]);
        for(size_t i=8;i<16;i++)if(mixed_feature_gradients[i]!=0.0f)return 7;
        if(!(image_gradient_norm>1.0e-8f)||
           !ama10_encoder_backward_update(&image_encoder,image_frames,mixed_features,
                                          mixed_feature_gradients,2,0.001f,0.0f,NULL)||
           image_encoder.step!=1){
            fprintf(stderr,"masked caption loss did not backpropagate through visual prefix\n");return 8;
        }
        ama10_encoder_free(&image_encoder);ama10_optimizer_free(&opt);ama10_model_free(&model);
    }
    {
        AMA10Model hard={0},soft={0};AMA10Optimizer hard_opt={0},soft_opt={0};
        const uint32_t inputs[]={1,2,3},targets[]={2,3,4};
        const unsigned char mask[]={1,1,1};float q[18]={0},hard_loss=0.0f,soft_loss=0.0f;
        for(size_t t=0;t<3;t++)q[t*6+targets[t]]=1.0f;
        if(!ama10_model_init(&hard,6,4,4,551)||!ama10_model_init(&soft,6,4,4,551)||
           !ama10_optimizer_init(&hard_opt,&hard)||!ama10_optimizer_init(&soft_opt,&soft)||
           !ama10_train_multimodal_sequence(&hard,&hard_opt,NULL,inputs,NULL,NULL,mask,
               targets,3,2,0.001f,0.0f,NULL,&hard_loss)||
           !ama10_train_soft_multimodal_sequence(&soft,&soft_opt,NULL,inputs,NULL,NULL,mask,
               q,3,2,0.001f,0.0f,NULL,&soft_loss)||
           fabsf(hard_loss-soft_loss)>1.0e-6f||
           memcmp(hard.wx,soft.wx,hard.width*hard.input_width*sizeof(float))||
           memcmp(hard.head,soft.head,hard.vocabulary*hard.width*sizeof(float)))return 9;
        ama10_optimizer_free(&hard_opt);ama10_optimizer_free(&soft_opt);
        ama10_model_free(&hard);ama10_model_free(&soft);
    }
    ama10_encoder_free(&encoder);ama10_encoder_free(&loaded);
    puts("trainable C modality encoder forward/backward, finite differences, checkpoint, and joint recurrent update: ok");
    return 0;
}
