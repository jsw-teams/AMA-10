#ifndef AMA10_Q8_H
#define AMA10_Q8_H

#include "ama10.h"
#include <stdint.h>

typedef struct {
    size_t width,input_width,vocabulary;
    int8_t *wx,*wh,*wm,*wu,*embed,*head;
    float *wx_scale,*wh_scale,*wm_scale,*wu_scale,*embed_scale,*head_scale;
    float *bias,*gate_bias;
} AMA10Q8Model;

void ama10_q8_model_free(AMA10Q8Model *model);
int ama10_q8_model_from_f32(const AMA10Model *source,AMA10Q8Model *quantized);
int ama10_q8_model_save(const AMA10Q8Model *model,const char *path);
int ama10_q8_model_load(AMA10Q8Model *model,const char *path);
size_t ama10_q8_model_memory_bytes(const AMA10Q8Model *model);
int ama10_q8_state_init(AMA10State *state,size_t width);
int ama10_q8_step_features(const AMA10Q8Model *model,AMA10State *state,
                            const float *features,float *logits,
                            unsigned max_depth,float halt_threshold);
int ama10_q8_step(const AMA10Q8Model *model,AMA10State *state,uint32_t token,
                  float *logits,unsigned max_depth,float halt_threshold);

#endif
