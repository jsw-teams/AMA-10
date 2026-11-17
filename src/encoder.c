#include "ama10_encoder.h"
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t encoder_rng(uint64_t *state){
    *state^=*state>>12;*state^=*state<<25;*state^=*state>>27;
    return (uint32_t)((*state*UINT64_C(2685821657736338717))>>32);
}
static int encoder_alloc(float **out,size_t count){
    if(count>SIZE_MAX/sizeof(float))return 0;
    *out=(float*)calloc(count,sizeof(float));return *out!=NULL;
}
int ama10_encoder_init(AMA10Encoder *e,size_t input_width,size_t output_width,uint64_t seed){
    if(!e||!input_width||!output_width||input_width>SIZE_MAX/output_width)return 0;
    if(!seed)seed=UINT64_C(1);
    memset(e,0,sizeof(*e));e->input_width=input_width;e->output_width=output_width;
    size_t weights=input_width*output_width;
    if(!encoder_alloc(&e->weights,weights)||!encoder_alloc(&e->bias,output_width)||
       !encoder_alloc(&e->first_moment,weights)||!encoder_alloc(&e->second_moment,weights)||
       !encoder_alloc(&e->bias_first_moment,output_width)||
       !encoder_alloc(&e->bias_second_moment,output_width)){
        ama10_encoder_free(e);return 0;
    }
    float scale=sqrtf(6.0f/((float)input_width+(float)output_width));
    for(size_t i=0;i<weights;i++)
        e->weights[i]=((float)encoder_rng(&seed)/2147483648.0f-1.0f)*scale;
    return 1;
}
void ama10_encoder_free(AMA10Encoder *e){
    if(!e)return;
    free(e->weights);free(e->bias);free(e->first_moment);free(e->second_moment);
    free(e->bias_first_moment);free(e->bias_second_moment);memset(e,0,sizeof(*e));
}
int ama10_encoder_forward(const AMA10Encoder *e,const float *inputs,size_t samples,float *features){
    if(!e||!e->weights||!e->bias||!inputs||!features||!samples||!e->input_width||
       !e->output_width||samples>SIZE_MAX/e->input_width||samples>SIZE_MAX/e->output_width)
        return 0;
    for(size_t s=0;s<samples;s++){
        for(size_t j=0;j<e->input_width;j++)if(!isfinite(inputs[s*e->input_width+j]))return 0;
        for(size_t i=0;i<e->output_width;i++){
            double value=e->bias[i];
            for(size_t j=0;j<e->input_width;j++)
                value+=(double)e->weights[i*e->input_width+j]*inputs[s*e->input_width+j];
            features[s*e->output_width+i]=tanhf((float)value);
            if(!isfinite(features[s*e->output_width+i]))return 0;
        }
    }
    return 1;
}
int ama10_encoder_backward_update(AMA10Encoder *e,const float *inputs,const float *features,
                                 const float *output_gradients,size_t samples,float lr,float wd,
                                 float *input_gradients){
    const float beta1=0.9f,beta2=0.999f,epsilon=1.0e-8f;
    if(!e||!e->weights||!e->bias||!e->first_moment||!e->second_moment||
       !e->bias_first_moment||!e->bias_second_moment||!inputs||!features||
       !output_gradients||!samples||!isfinite(lr)||lr<=0.0f||!isfinite(wd)||wd<0.0f||
       e->step==UINT64_MAX||samples>SIZE_MAX/e->input_width||samples>SIZE_MAX/e->output_width)
        return 0;
    size_t nweights=e->input_width*e->output_width;
    if(nweights>SIZE_MAX/sizeof(float))return 0;
    float *gw=(float*)calloc(nweights,sizeof(float)),*gb=(float*)calloc(e->output_width,sizeof(float));
    if(!gw||!gb){free(gw);free(gb);return 0;}
    if(input_gradients)memset(input_gradients,0,samples*e->input_width*sizeof(float));
    double norm2=0.0;
    for(size_t s=0;s<samples;s++)for(size_t i=0;i<e->output_width;i++){
        size_t ix=s*e->output_width+i;float y=features[ix],dy=output_gradients[ix];
        if(!isfinite(y)||!isfinite(dy)||fabsf(y)>1.0f){free(gw);free(gb);return 0;}
        float dz=dy*(1.0f-y*y);gb[i]+=dz;
        for(size_t j=0;j<e->input_width;j++){
            float x=inputs[s*e->input_width+j];if(!isfinite(x)){free(gw);free(gb);return 0;}
            gw[i*e->input_width+j]+=dz*x;
            if(input_gradients)input_gradients[s*e->input_width+j]+=
                e->weights[i*e->input_width+j]*dz;
        }
    }
    for(size_t i=0;i<nweights;i++)norm2+=(double)gw[i]*gw[i];
    for(size_t i=0;i<e->output_width;i++)norm2+=(double)gb[i]*gb[i];
    if(!isfinite(norm2)){free(gw);free(gb);return 0;}
    float clip=norm2>1.0?(float)(1.0/sqrt(norm2)):1.0f;
    uint64_t next=e->step+1;
    double correction1=1.0-pow((double)beta1,(double)next);
    double correction2=1.0-pow((double)beta2,(double)next);
    for(size_t i=0;i<nweights;i++){
        float grad=gw[i]*clip,*m=&e->first_moment[i],*v=&e->second_moment[i];
        *m=beta1**m+(1.0f-beta1)*grad;*v=beta2**v+(1.0f-beta2)*grad*grad;
        float update=(float)(((double)*m/correction1)/(sqrt((double)*v/correction2)+epsilon));
        e->weights[i]=e->weights[i]*(1.0f-lr*wd)-lr*update;
    }
    for(size_t i=0;i<e->output_width;i++){
        float grad=gb[i]*clip,*m=&e->bias_first_moment[i],*v=&e->bias_second_moment[i];
        *m=beta1**m+(1.0f-beta1)*grad;*v=beta2**v+(1.0f-beta2)*grad*grad;
        float update=(float)(((double)*m/correction1)/(sqrt((double)*v/correction2)+epsilon));
        e->bias[i]=e->bias[i]*(1.0f-lr*wd)-lr*update;
    }
    e->step=next;free(gw);free(gb);return 1;
}
static int enc_write_u32(FILE *f,uint32_t v){
    unsigned char b[4]={(unsigned char)v,(unsigned char)(v>>8),(unsigned char)(v>>16),(unsigned char)(v>>24)};
    return fwrite(b,1,4,f)==4;
}
static int enc_read_u32(FILE *f,uint32_t *v){
    unsigned char b[4];if(fread(b,1,4,f)!=4)return 0;
    *v=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);return 1;
}
static int enc_write_u64(FILE *f,uint64_t v){return enc_write_u32(f,(uint32_t)v)&&enc_write_u32(f,(uint32_t)(v>>32));}
static int enc_read_u64(FILE *f,uint64_t *v){uint32_t lo,hi;if(!enc_read_u32(f,&lo)||!enc_read_u32(f,&hi))return 0;*v=(uint64_t)lo|((uint64_t)hi<<32);return 1;}
static int enc_write_floats(FILE *f,const float *x,size_t count){
    for(size_t i=0;i<count;i++){uint32_t bits;memcpy(&bits,&x[i],4);if(!enc_write_u32(f,bits))return 0;}return 1;
}
static int enc_read_floats(FILE *f,float *x,size_t count){
    for(size_t i=0;i<count;i++){uint32_t bits;if(!enc_read_u32(f,&bits))return 0;memcpy(&x[i],&bits,4);if(!isfinite(x[i]))return 0;}return 1;
}
int ama10_encoder_save(const AMA10Encoder *e,const char *path){
    if(!e||!path||!e->weights||!e->bias||!e->first_moment||!e->second_moment||
       !e->bias_first_moment||!e->bias_second_moment||!e->input_width||!e->output_width||
       e->input_width>UINT32_MAX||e->output_width>UINT32_MAX||e->input_width>SIZE_MAX/e->output_width)
        return 0;
    size_t path_len=strlen(path);if(path_len>SIZE_MAX-5)return 0;
    char *tmp=(char*)malloc(path_len+5),*backup=(char*)malloc(path_len+5);
    if(!tmp||!backup){free(tmp);free(backup);return 0;}
    memcpy(tmp,path,path_len);memcpy(tmp+path_len,".tmp",5);
    memcpy(backup,path,path_len);memcpy(backup+path_len,".bak",5);
    FILE *f=fopen(tmp,"wb");if(!f){free(tmp);free(backup);return 0;}
    static const unsigned char magic[8]={'A','M','A','1','0','E','N','1'};
    size_t n=e->input_width*e->output_width;
    int ok=fwrite(magic,1,8,f)==8&&enc_write_u32(f,1)&&enc_write_u32(f,(uint32_t)e->input_width)&&
       enc_write_u32(f,(uint32_t)e->output_width)&&enc_write_u64(f,e->step)&&
       enc_write_floats(f,e->weights,n)&&enc_write_floats(f,e->bias,e->output_width)&&
       enc_write_floats(f,e->first_moment,n)&&enc_write_floats(f,e->second_moment,n)&&
       enc_write_floats(f,e->bias_first_moment,e->output_width)&&
       enc_write_floats(f,e->bias_second_moment,e->output_width);
    if(fflush(f)!=0)ok=0;
    if(fclose(f)!=0)ok=0;
    int had_backup=0;
    if(ok){
        (void)remove(backup);
        if(rename(path,backup)==0)had_backup=1;
        else if(errno!=ENOENT)ok=0;
    }
    if(ok&&rename(tmp,path)!=0)ok=0;
    if(!ok){
        (void)remove(tmp);
        if(had_backup)(void)rename(backup,path);
    }else if(had_backup)(void)remove(backup);
    free(tmp);free(backup);return ok;
}
int ama10_encoder_load(AMA10Encoder *e,const char *path){
    if(!e||!path)return 0;
    FILE *f=fopen(path,"rb");if(!f)return 0;
    if(fseek(f,0,SEEK_END)!=0){fclose(f);return 0;}
    long file_size=ftell(f);if(file_size<28||fseek(f,0,SEEK_SET)!=0){fclose(f);return 0;}
    unsigned char magic[8],expected[8]={'A','M','A','1','0','E','N','1'};
    uint32_t version=0,input_width=0,output_width=0;uint64_t step=0;
    int ok=fread(magic,1,8,f)==8&&memcmp(magic,expected,8)==0&&enc_read_u32(f,&version)&&
       version==1&&enc_read_u32(f,&input_width)&&enc_read_u32(f,&output_width)&&
       enc_read_u64(f,&step)&&input_width&&output_width;
    AMA10Encoder temp={0};
    uint64_t n64=(uint64_t)input_width*(uint64_t)output_width;
    uint64_t out64=(uint64_t)output_width;
    if(ok&&(input_width>SIZE_MAX/output_width||n64>(UINT64_MAX-28-out64*12)/12||
            28+n64*12+out64*12!=(uint64_t)file_size))ok=0;
    if(ok)ok=ama10_encoder_init(&temp,input_width,output_width,1);
    if(ok){
        size_t n=(size_t)n64;
        if(ok)ok=enc_read_floats(f,temp.weights,n)&&enc_read_floats(f,temp.bias,output_width)&&
           enc_read_floats(f,temp.first_moment,n)&&enc_read_floats(f,temp.second_moment,n)&&
           enc_read_floats(f,temp.bias_first_moment,output_width)&&
           enc_read_floats(f,temp.bias_second_moment,output_width);
    }
    if(ok&&fgetc(f)!=EOF)ok=0;
    if(fclose(f)!=0)ok=0;
    if(ok){temp.step=step;ama10_encoder_free(e);*e=temp;}else ama10_encoder_free(&temp);
    return ok;
}
int ama10_train_encoded_sequence(AMA10Model *m,AMA10Optimizer *opt,AMA10State *state,
                                 AMA10Encoder *encoder,const float *raw_inputs,
                                 const uint32_t *targets,size_t length,unsigned depth,
                                 float lr,float wd,float *mean_loss){
    if(!m||!encoder||encoder->output_width!=m->input_width||!length||
       length>SIZE_MAX/encoder->input_width||length>SIZE_MAX/encoder->output_width||
       length*encoder->output_width>SIZE_MAX/sizeof(float)||
       length*encoder->input_width>SIZE_MAX/sizeof(float))return 0;
    float *features=(float*)malloc(length*encoder->output_width*sizeof(float));
    float *feature_gradients=(float*)malloc(length*encoder->output_width*sizeof(float));
    if(!features||!feature_gradients){free(features);free(feature_gradients);return 0;}
    int ok=ama10_encoder_forward(encoder,raw_inputs,length,features)&&
       ama10_train_feature_sequence(m,opt,state,features,targets,length,depth,lr,wd,
                                    feature_gradients,mean_loss)&&
       ama10_encoder_backward_update(encoder,raw_inputs,features,feature_gradients,length,lr,wd,NULL);
    free(features);free(feature_gradients);return ok;
}

int ama10_train_feature_conditioned_text(AMA10Model *m,AMA10Optimizer *opt,
                              AMA10State *state,AMA10Encoder *encoder,
                              const float *raw_features,size_t feature_count,
                              const uint32_t *conditioning_tokens,size_t conditioning_count,
                              const uint32_t *text,size_t text_count,
                              unsigned depth,float lr,float wd,float *mean_loss){
    if(!m||!opt||!state||!encoder||!raw_features||!feature_count||!text||
       !text_count||(conditioning_count&&!conditioning_tokens)||
       encoder->output_width!=m->input_width||state->width!=m->width||
       feature_count>SIZE_MAX-conditioning_count||
       feature_count+conditioning_count>SIZE_MAX-text_count)return 0;
    for(size_t i=0;i<text_count;i++)if(text[i]>=m->vocabulary)return 0;
    for(size_t i=0;i<conditioning_count;i++)if(conditioning_tokens[i]>=m->vocabulary)return 0;
    size_t text_start=feature_count+conditioning_count;
    size_t length=text_start+text_count;
    if(length>SIZE_MAX/sizeof(uint32_t)||length>SIZE_MAX/sizeof(unsigned char)||
       length>SIZE_MAX/m->input_width||
       length*m->input_width>SIZE_MAX/sizeof(float))return 0;
    float *features=(float*)calloc(length*m->input_width,sizeof(float));
    float *feature_gradients=(float*)malloc(length*m->input_width*sizeof(float));
    uint32_t *tokens=(uint32_t*)calloc(length,sizeof(uint32_t));
    uint32_t *targets=(uint32_t*)calloc(length,sizeof(uint32_t));
    unsigned char *feature_mask=(unsigned char*)calloc(length,1);
    unsigned char *loss_mask=(unsigned char*)calloc(length,1);
    if(!features||!feature_gradients||!tokens||!targets||!feature_mask||!loss_mask){
        free(features);free(feature_gradients);free(tokens);free(targets);
        free(feature_mask);free(loss_mask);return 0;
    }
    int ok=ama10_encoder_forward(encoder,raw_features,feature_count,features);
    if(ok){
        for(size_t i=0;i<feature_count;i++)feature_mask[i]=1;
        size_t first_text_target=conditioning_count?text_start-1:feature_count-1;
        targets[first_text_target]=text[0];
        loss_mask[first_text_target]=1;
        for(size_t i=0;i<conditioning_count;i++)
            tokens[feature_count+i]=conditioning_tokens[i];
        for(size_t i=0;i<text_count;i++){
            size_t position=text_start+i;
            tokens[position]=text[i];
            if(i+1<text_count){targets[position]=text[i+1];loss_mask[position]=1;}
        }
        ama10_state_reset(state);
        ok=ama10_train_multimodal_sequence(m,opt,state,tokens,features,feature_mask,
                                           loss_mask,targets,length,depth,lr,wd,
                                           feature_gradients,mean_loss);
        if(ok)ok=ama10_encoder_backward_update(encoder,raw_features,features,
            feature_gradients,feature_count,lr,wd,NULL);
    }
    free(features);free(feature_gradients);free(tokens);free(targets);
    free(feature_mask);free(loss_mask);return ok;
}
int ama10_train_aligned_feature_text(AMA10Model *m,AMA10Optimizer *opt,
                              AMA10State *state,AMA10Encoder *encoder,
                              const float *raw_aligned_features,size_t steps,
                              const uint32_t *conditioning_tokens,size_t conditioning_count,
                              const uint32_t *text,size_t text_count,
                              unsigned depth,float lr,float wd,float *mean_loss){
    if(!m||!opt||!state||!encoder||!raw_aligned_features||!steps||!text||!text_count||
       steps!=text_count||(conditioning_count&&!conditioning_tokens)||
       encoder->output_width!=m->input_width||state->width!=m->width||
       steps>SIZE_MAX/encoder->output_width||
       steps*encoder->output_width>SIZE_MAX/sizeof(float)||
       text_count>(SIZE_MAX-conditioning_count)/2)return 0;
    for(size_t i=0;i<text_count;i++)if(text[i]>=m->vocabulary)return 0;
    for(size_t i=0;i<conditioning_count;i++)if(conditioning_tokens[i]>=m->vocabulary)return 0;
    size_t length=conditioning_count+2*text_count;
    if(length>SIZE_MAX/sizeof(uint32_t)||length>SIZE_MAX/sizeof(unsigned char)||
       length>SIZE_MAX/m->input_width||length*m->input_width>SIZE_MAX/sizeof(float))return 0;
    float *encoded=(float*)malloc(steps*encoder->output_width*sizeof(float));
    float *features=(float*)calloc(length*m->input_width,sizeof(float));
    float *sequence_gradients=(float*)malloc(length*m->input_width*sizeof(float));
    float *aligned_gradients=(float*)malloc(steps*m->input_width*sizeof(float));
    uint32_t *tokens=(uint32_t*)calloc(length,sizeof(uint32_t));
    uint32_t *targets=(uint32_t*)calloc(length,sizeof(uint32_t));
    unsigned char *feature_mask=(unsigned char*)calloc(length,1);
    unsigned char *loss_mask=(unsigned char*)calloc(length,1);
    if(!encoded||!features||!sequence_gradients||!aligned_gradients||!tokens||!targets||
       !feature_mask||!loss_mask){
        free(encoded);free(features);free(sequence_gradients);free(aligned_gradients);
        free(tokens);free(targets);free(feature_mask);free(loss_mask);return 0;
    }
    int ok=ama10_encoder_forward(encoder,raw_aligned_features,steps,encoded);
    if(ok){
        for(size_t i=0;i<conditioning_count;i++)tokens[i]=conditioning_tokens[i];
        for(size_t i=0;i<text_count;i++){
            size_t feature_position=conditioning_count+2*i;
            memcpy(features+feature_position*m->input_width,encoded+i*encoder->output_width,
                   m->input_width*sizeof(float));
            feature_mask[feature_position]=1;
            targets[feature_position]=text[i];loss_mask[feature_position]=1;
            tokens[feature_position+1]=text[i];
        }
        ama10_state_reset(state);
        ok=ama10_train_multimodal_sequence(m,opt,state,tokens,features,feature_mask,
            loss_mask,targets,length,depth,lr,wd,sequence_gradients,mean_loss);
        if(ok){
            for(size_t i=0;i<steps;i++){
                size_t feature_position=conditioning_count+2*i;
                memcpy(aligned_gradients+i*m->input_width,
                       sequence_gradients+feature_position*m->input_width,
                       m->input_width*sizeof(float));
            }
            ok=ama10_encoder_backward_update(encoder,raw_aligned_features,encoded,
                aligned_gradients,steps,lr,wd,NULL);
        }
    }
    free(encoded);free(features);free(sequence_gradients);free(aligned_gradients);
    free(tokens);free(targets);free(feature_mask);free(loss_mask);return ok;
}
int ama10_train_image_caption_conditioned(AMA10Model *m,AMA10Optimizer *opt,
                              AMA10State *state,AMA10Encoder *encoder,
                              const float *raw_patches,size_t patch_count,
                              const uint32_t *conditioning_tokens,size_t conditioning_count,
                              const uint32_t *caption,size_t caption_count,
                              unsigned depth,float lr,float wd,float *mean_loss){
    return ama10_train_feature_conditioned_text(m,opt,state,encoder,raw_patches,
        patch_count,conditioning_tokens,conditioning_count,caption,caption_count,
        depth,lr,wd,mean_loss);
}
int ama10_train_image_caption(AMA10Model *m,AMA10Optimizer *opt,AMA10State *state,
                              AMA10Encoder *encoder,const float *raw_patches,
                              size_t patch_count,const uint32_t *caption,
                              size_t caption_count,unsigned depth,float lr,float wd,
                              float *mean_loss){
    return ama10_train_feature_conditioned_text(m,opt,state,encoder,raw_patches,
        patch_count,NULL,0,caption,caption_count,depth,lr,wd,mean_loss);
}
