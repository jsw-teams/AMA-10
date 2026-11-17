#include "ama10.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static float *parameter_group(AMA10Model *m,int group,size_t *count){
    switch(group){
        case 0:*count=m->width*m->input_width;return m->wx;
        case 1:*count=m->width*m->width;return m->wh;
        case 2:*count=m->width*m->width;return m->wm;
        case 3:*count=m->width*m->width;return m->wu;
        case 4:*count=m->width;return m->bias;
        case 5:*count=m->width;return m->gate_bias;
        case 6:*count=m->input_width*m->vocabulary;return m->embed;
        case 7:*count=m->vocabulary*m->width;return m->head;
        default:return NULL;
    }
}
static int same_model_parameters(AMA10Model *a,AMA10Model *b){
    for(int group=0;group<8;group++){
        size_t n=0,other_n=0;float *x=parameter_group(a,group,&n),*y=parameter_group(b,group,&other_n);
        if(n!=other_n||memcmp(x,y,n*sizeof(float)))return 0;
    }
    return 1;
}

int main(void){
    if(ama10_learning_rate(0.01f,0,0.1f)!=0.0f||
       ama10_learning_rate(0.01f,1,0.1f)!=0.01f||
       !(ama10_learning_rate(0.01f,11,0.1f)<ama10_learning_rate(0.01f,2,0.1f))||
       ama10_learning_rate(0.01f,11,0.0f)!=0.01f)return 30;
    AMA10Model a,b; AMA10State x,y;
    if(!ama10_model_init(&a,17,8,8,123)||!ama10_model_init(&b,17,8,8,123)||!ama10_state_init(&x,8)||!ama10_state_init(&y,8))return 1;
    float p[17],q[17];
    for(uint32_t t=0;t<12;t++){
        if(!ama10_step(&a,&x,t%17,p,5,0.8f)||!ama10_step(&b,&y,t%17,q,5,0.8f))return 2;
        for(int i=0;i<17;i++)if(!isfinite(p[i])||p[i]!=q[i])return 3;
    }
    if(x.steps==0||x.steps>60)return 4;
    if(!ama10_model_save(&a,"build/model.test"))return 22;
    AMA10Model restored={0};AMA10State sa,sb;
    if(!ama10_model_load(&restored,"build/model.test")||restored.vocabulary!=a.vocabulary||
       restored.width!=a.width||!ama10_state_init(&sa,8)||!ama10_state_init(&sb,8))return 23;
    if(!ama10_step(&a,&sa,4,p,3,2.0f)||!ama10_step(&restored,&sb,4,q,3,2.0f))return 24;
    for(int i=0;i<17;i++)if(p[i]!=q[i])return 25;
    ama10_state_free(&sa);ama10_state_free(&sb);ama10_model_free(&restored);
    ama10_state_reset(&x);if(x.steps||x.fast[0]!=0||x.slow[0]!=0)return 5;
    ama10_state_free(&x);ama10_state_free(&y);ama10_model_free(&a);ama10_model_free(&b);
    if(!ama10_model_init(&a,2,8,8,456))return 6;
    AMA10Optimizer opt;
    if(!ama10_optimizer_init(&opt,&a))return 7;
    if(!ama10_state_init(&x,8))return 7;
    const uint32_t in[]={0,1,0,1,0,1,0,1};
    const uint32_t out[]={1,0,1,0,1,0,1,0};
    float first=0,last=0;
    double initial_eval=0,final_eval=0;
    if(!ama10_eval_loss(&a,in,out,8,2,&initial_eval))return 8;
    for(int i=0;i<100;i++){
        ama10_state_reset(&x);
        float loss=0;
        if(!ama10_train_sequence(&a,&opt,&x,in,out,8,2,0.01f,0.0f,&loss))return 8;
        if(i==0)first=loss;
    }
    ama10_state_reset(&x);
    if(!ama10_train_sequence(&a,&opt,&x,in,out,8,2,0.01f,0.0f,&last))return 9;
    if(!(last<first)){fprintf(stderr,"tiny sequence did not overfit: initial %.6f final %.6f\n",first,last);return 10;}
    if(!ama10_eval_loss(&a,in,out,8,2,&final_eval)||!(final_eval<initial_eval))return 11;
    ama10_state_free(&x);ama10_optimizer_free(&opt);ama10_model_free(&a);
    {
        AMA10Model m;AMA10Optimizer muon;AMA10State state;
        const uint32_t mi[]={0,1,2,1,0},mt[]={1,2,1,0,2};float loss=0;
        if(!ama10_model_init(&m,3,5,3,9901)||!ama10_optimizer_init(&muon,&m)||
           !ama10_optimizer_set_algorithm(&muon,AMA10_OPT_MUON)||!ama10_state_init(&state,5))return 31;
        float before=m.wh[0];
        if(!ama10_train_sequence(&m,&muon,&state,mi,mt,5,2,0.001f,0.01f,&loss)||
           muon.step!=1||muon.algorithm!=AMA10_OPT_MUON||!isfinite(loss)||m.wh[0]==before)return 32;
        for(int group=0;group<8;group++){
            size_t count=0;float *params=parameter_group(&m,group,&count);
            for(size_t i=0;i<count;i++)if(!isfinite(params[i]))return 33;
        }
        for(size_t i=0;i<m.input_width*m.vocabulary;i++)
            if(!isfinite(muon.second_moment[6][i]))return 34;
        ama10_state_free(&state);ama10_optimizer_free(&muon);ama10_model_free(&m);
    }
    {
        AMA10Model model,old_shape={0};AMA10Optimizer opt;AMA10State state;
        const uint32_t before_in[]={0,1,2,1},before_target[]={1,2,1,0};float loss=0;
        if(!ama10_model_init(&model,4,5,3,5521)||!ama10_optimizer_init(&opt,&model)||
           !ama10_state_init(&state,5)||
           !ama10_train_sequence(&model,&opt,&state,before_in,before_target,4,2,0.001f,0.0f,&loss))return 35;
        size_t old_vocabulary=model.vocabulary,old_width=model.width,old_input_width=model.input_width;
        float wx=model.wx[3],wh=model.wh[7],bias=model.bias[2];
        float embed_first=model.embed[0],embed_last=model.embed[(old_input_width-1)*old_vocabulary+old_vocabulary-1];
        float head_last=model.head[(old_vocabulary-1)*old_width+old_width-1];
        float first_moment=opt.first_moment[1][7],second_moment=opt.second_moment[1][7];
        uint64_t step=opt.step;
        old_shape.width=old_width;old_shape.input_width=old_input_width;old_shape.vocabulary=old_vocabulary;
        if(!ama10_model_resize_vocabulary(&model,8,78123)||model.vocabulary!=8||
           !ama10_optimizer_resize_vocabulary(&opt,&old_shape,&model)||opt.step!=step||
           model.wx[3]!=wx||model.wh[7]!=wh||model.bias[2]!=bias||model.embed[0]!=embed_first||
           model.embed[(old_input_width-1)*model.vocabulary+old_vocabulary-1]!=embed_last||
           model.head[(old_vocabulary-1)*old_width+old_width-1]!=head_last||
           opt.first_moment[1][7]!=first_moment||opt.second_moment[1][7]!=second_moment)return 36;
        if(opt.first_moment[6][old_vocabulary]!=0.0f||opt.second_moment[6][old_vocabulary]!=0.0f||
           opt.first_moment[7][old_vocabulary*old_width]!=0.0f||
           opt.second_moment[7][old_vocabulary*old_width]!=0.0f)return 37;
        const uint32_t after_in[]={7,0,7,1},after_target[]={0,7,1,7};
        ama10_state_reset(&state);
        if(!ama10_train_sequence(&model,&opt,&state,after_in,after_target,4,2,0.001f,0.0f,&loss)||
           model.vocabulary!=8||opt.step!=step+1||!isfinite(loss))return 38;
        ama10_state_free(&state);ama10_optimizer_free(&opt);ama10_model_free(&model);
    }
    {
        const uint32_t gi[]={0,1,2,1,0},gt[]={1,2,1,0,2};
        double ratios[8],ratio_mean=0.0;
        for(int group=0;group<8;group++){
            AMA10Model m,reference;AMA10Optimizer adam;double plus,minus;float loss;
            if(!ama10_model_init(&m,3,5,5,901)||!ama10_model_init(&reference,3,5,5,901)||!ama10_optimizer_init(&adam,&m))return 12;
            double reference_loss,model_loss;if(!ama10_eval_loss(&reference,gi,gt,5,2,&reference_loss)||!ama10_eval_loss(&m,gi,gt,5,2,&model_loss))return 13;
            if(fabs(model_loss-reference_loss)>1.0e-12){fprintf(stderr,"same-seed models differ: %.9g %.9g\n",model_loss,reference_loss);return 21;}
            if(!ama10_train_sequence(&m,&adam,NULL,gi,gt,5,2,0.0001f,0.0f,&loss))return 16;
            if(fabs((double)loss-reference_loss)>1.0e-6){fprintf(stderr,"forward mismatch: train %.9g eval %.9g\n",loss,reference_loss);return 20;}
            size_t count=0,index=0;float *group_params=parameter_group(&reference,group,&count);
            for(size_t i=1;i<count;i++)if(fabsf(adam.first_moment[group][i])>fabsf(adam.first_moment[group][index]))index=i;
            float *p=&group_params[index],old=*p,epsilon=0.0001f*(1.0f+fabsf(old));*p=old+epsilon;
            if(!ama10_eval_loss(&reference,gi,gt,5,2,&plus))return 13;
            *p=old-epsilon;if(!ama10_eval_loss(&reference,gi,gt,5,2,&minus))return 14;
            *p=old;double deriv=(plus-minus)/(2.0*(double)epsilon);
            double analytic=(double)adam.first_moment[group][index]/0.1;
            if(fabs(deriv)<1.0e-5||!(deriv*analytic>0.0)){
                fprintf(stderr,"gradient mismatch group %d index %zu: finite difference %.9g reverse mode %.9g\n",group,index,deriv,analytic);return 17;
            }
            ratios[group]=analytic/deriv;
            ratio_mean+=ratios[group]/8.0;
            ama10_optimizer_free(&adam);ama10_model_free(&m);ama10_model_free(&reference);
        }
        if(!(ratio_mean>0.0&&ratio_mean<=1.02)){
            fprintf(stderr,"gradient common scale %.6f; group ratios:",ratio_mean);
            for(int group=0;group<8;group++)fprintf(stderr," %.6f",ratios[group]);
            fputc('\n',stderr);return 18;
        }
        for(int group=0;group<8;group++)if(fabs(ratios[group]-ratio_mean)>0.12*ratio_mean){
            fprintf(stderr,"gradient magnitude mismatch group %d: relative scale %.6f versus shared clip scale %.6f\n",group,ratios[group],ratio_mean);return 19;
        }
    }
    {
        AMA10Model token_model,feature_model,reference;
        AMA10Optimizer token_opt,feature_opt;
        AMA10State token_state,feature_state;
        const uint32_t input[]={1,2,3,1},target[]={2,3,4,0};
        float features[16],feature_gradients[16],token_loss=0.0f,feature_loss=0.0f;
        double token_eval=0.0,feature_eval=0.0,plus=0.0,minus=0.0;
        if(!ama10_model_init(&token_model,6,4,4,661)||
           !ama10_model_init(&feature_model,6,4,4,661)||
           !ama10_model_init(&reference,6,4,4,661)||
           !ama10_optimizer_init(&token_opt,&token_model)||
           !ama10_optimizer_init(&feature_opt,&feature_model)||
           !ama10_state_init(&token_state,4)||!ama10_state_init(&feature_state,4))return 39;
        for(size_t t=0;t<4;t++)for(size_t j=0;j<4;j++)
            features[t*4+j]=token_model.embed[j*token_model.vocabulary+input[t]];
        if(!ama10_eval_loss(&token_model,input,target,4,2,&token_eval)||
           !ama10_eval_feature_loss(&feature_model,features,target,4,2,&feature_eval)||
           fabs(token_eval-feature_eval)>1.0e-12||
           !ama10_train_sequence(&token_model,&token_opt,&token_state,input,target,4,2,
                                 0.001f,0.0f,&token_loss)||
           !ama10_train_feature_sequence(&feature_model,&feature_opt,&feature_state,
                                         features,target,4,2,0.001f,0.0f,
                                         feature_gradients,&feature_loss)||
           fabs((double)token_loss-feature_loss)>1.0e-6)return 40;
        for(int group=0;group<8;group++)if(group!=6){
            size_t count_a=0,count_b=0;
            float *a=parameter_group(&token_model,group,&count_a);
            float *b=parameter_group(&feature_model,group,&count_b);
            if(count_a!=count_b||memcmp(a,b,count_a*sizeof(float)))return 41;
        }
        size_t feature_index=0;
        for(size_t i=1;i<16;i++)if(fabsf(feature_gradients[i])>fabsf(feature_gradients[feature_index]))feature_index=i;
        float old=features[feature_index],epsilon=0.001f;
        features[feature_index]=old+epsilon;
        if(!ama10_eval_feature_loss(&reference,features,target,4,2,&plus))return 42;
        features[feature_index]=old-epsilon;
        if(!ama10_eval_feature_loss(&reference,features,target,4,2,&minus))return 43;
        features[feature_index]=old;
        double numeric=(plus-minus)/(2.0*(double)epsilon);
        if(fabs(numeric)<1.0e-6||numeric*feature_gradients[feature_index]<=0.0||
           fabs(numeric-feature_gradients[feature_index])>0.08*fabs(numeric)+0.001){
            fprintf(stderr,"modality input gradient mismatch at %zu: finite difference %.9g reverse mode %.9g\n",
                    feature_index,numeric,feature_gradients[feature_index]);return 44;
        }
        ama10_state_free(&token_state);ama10_state_free(&feature_state);
        ama10_optimizer_free(&token_opt);ama10_optimizer_free(&feature_opt);
        ama10_model_free(&token_model);ama10_model_free(&feature_model);ama10_model_free(&reference);
    }
    {
        AMA10Model fixed,adaptive;AMA10Optimizer fixed_opt,adaptive_opt;
        AMA10State fixed_state,adaptive_state;float fixed_loss=0,adaptive_loss=0,mean_depth=0;
        double fixed_eval=0,adaptive_eval=0,eval_depth=0;
        const uint32_t ai[]={0,1,2,1,0},at[]={1,2,1,0,2};
        if(!ama10_model_init(&fixed,3,5,5,731)||!ama10_model_init(&adaptive,3,5,5,731)||
           !ama10_optimizer_init(&fixed_opt,&fixed)||!ama10_optimizer_init(&adaptive_opt,&adaptive)||
           !ama10_state_init(&fixed_state,5)||!ama10_state_init(&adaptive_state,5))return 22;
        if(!ama10_train_sequence(&fixed,&fixed_opt,&fixed_state,ai,at,5,2,0.0001f,0.0f,&fixed_loss)||
           !ama10_train_sequence_adaptive(&adaptive,&adaptive_opt,&adaptive_state,ai,at,5,4,0.0f,
                                          0.0001f,0.0f,&adaptive_loss,&mean_depth))return 23;
        if(mean_depth!=2.0f||adaptive_state.steps!=10||fixed_state.steps!=10||
           fixed_loss!=adaptive_loss||!same_model_parameters(&fixed,&adaptive)){
            fprintf(stderr,"adaptive minimum-depth path differs from fixed two-step reference\n");return 24;
        }
        if(!ama10_eval_loss(&adaptive,ai,at,5,4,&fixed_eval)||
           !ama10_eval_loss_adaptive(&adaptive,ai,at,5,4,2.0f,&adaptive_eval,&eval_depth)||
           fixed_eval!=adaptive_eval||eval_depth!=4.0||
           !ama10_eval_loss_adaptive(&adaptive,ai,at,5,4,0.0f,&adaptive_eval,&eval_depth)||
           eval_depth!=2.0||!isfinite(adaptive_eval)){
            fprintf(stderr,"adaptive inference evaluation depth/loss parity failed\n");return 45;
        }
        ama10_state_free(&fixed_state);ama10_state_free(&adaptive_state);
        ama10_optimizer_free(&fixed_opt);ama10_optimizer_free(&adaptive_opt);
        ama10_model_free(&fixed);ama10_model_free(&adaptive);
    }
    {
        AMA10Model direct,accumulated;AMA10Optimizer direct_opt,accumulated_opt;
        AMA10State direct_state,accumulated_state;AMA10Gradients gradients;
        float direct_loss=0,accumulated_loss=0;
        const uint32_t ai[]={0,1,2,1,0},at[]={1,2,1,0,2};
        if(!ama10_model_init(&direct,3,5,5,811)||!ama10_model_init(&accumulated,3,5,5,811)||
           !ama10_optimizer_init(&direct_opt,&direct)||!ama10_optimizer_init(&accumulated_opt,&accumulated)||
           !ama10_state_init(&direct_state,5)||!ama10_state_init(&accumulated_state,5)||
           !ama10_gradients_init(&gradients,&accumulated))return 25;
        if(!ama10_train_sequence(&direct,&direct_opt,&direct_state,ai,at,5,2,0.001f,0.0f,&direct_loss)||
           !ama10_train_sequence_accumulate(&accumulated,&accumulated_opt,&accumulated_state,
                                            ai,at,5,2,&gradients,&accumulated_loss))return 26;
        if(gradients.tokens!=5||accumulated_opt.step!=0||direct_loss!=accumulated_loss||
           !ama10_optimizer_apply(&accumulated,&accumulated_opt,&gradients,0.001f,0.0f))return 27;
        if(gradients.tokens!=0||accumulated_opt.step!=1||!same_model_parameters(&direct,&accumulated)){
            fprintf(stderr,"one-chunk gradient accumulation differs from direct update\n");return 28;
        }
        for(size_t k=0;k<8;k++){
            size_t n=0;parameter_group(&direct,(int)k,&n);
            if(memcmp(direct_opt.first_moment[k],accumulated_opt.first_moment[k],n*sizeof(float))||
               memcmp(direct_opt.second_moment[k],accumulated_opt.second_moment[k],n*sizeof(float))){
                fprintf(stderr,"accumulated AdamW moments differ from direct update\n");return 29;
            }
        }
        ama10_gradients_free(&gradients);ama10_state_free(&direct_state);ama10_state_free(&accumulated_state);
        ama10_optimizer_free(&direct_opt);ama10_optimizer_free(&accumulated_opt);
        ama10_model_free(&direct);ama10_model_free(&accumulated);
    }
    {
        AMA10Model depth_model={0};AMA10State depth_one={0},depth_four={0};float logits_one[2],logits_four[2];
        if(!ama10_model_init(&depth_model,2,1,1,71)||!ama10_state_init(&depth_one,1)||
           !ama10_state_init(&depth_four,1))return 30;
        memset(depth_model.wx,0,sizeof(float));memset(depth_model.wh,0,sizeof(float));
        memset(depth_model.wm,0,sizeof(float));memset(depth_model.wu,0,sizeof(float));
        memset(depth_model.bias,0,sizeof(float));depth_model.gate_bias[0]=-100.0f;
        memset(depth_model.embed,0,2*sizeof(float));memset(depth_model.head,0,2*sizeof(float));
        depth_one.fast[0]=depth_four.fast[0]=0.3f;depth_one.slow[0]=depth_four.slow[0]=0.8f;
        if(!ama10_step(&depth_model,&depth_one,0,logits_one,1,2.0f)||
           !ama10_step(&depth_model,&depth_four,0,logits_four,4,2.0f)||
           fabsf(depth_one.fast[0]-depth_four.fast[0])>1e-6f||
           fabsf(depth_one.slow[0]-depth_four.slow[0])>1e-6f||
           fabsf(depth_four.slow[0]-0.7975f)>1e-6f){
            fprintf(stderr,"slow-memory retention changed with recurrent depth\n");return 31;
        }
        ama10_state_free(&depth_one);ama10_state_free(&depth_four);ama10_model_free(&depth_model);
    }
    puts("core forward/reset, depth-invariant slow memory, overfit, token and modality-input gradient checks, adaptive-depth, accumulation parity, and Muon hybrid update: ok");return 0;
}
