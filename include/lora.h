#ifndef AMA10_LORA_H
#define AMA10_LORA_H
#include "ama10.h"
#include <stdint.h>
typedef struct {
    size_t rank,width,vocabulary;
    float scale;
    uint64_t base_hash,tokenizer_hash;
    float *a; /* rank x width */
    float *b; /* vocabulary x rank */
} AMA10LoRA;
int ama10_lora_init(AMA10LoRA *adapter,const AMA10Model *base,size_t rank,float scale,
                    uint64_t base_hash,uint64_t tokenizer_hash,uint64_t seed);
int ama10_lora_save(const AMA10LoRA *adapter,const char *path);
int ama10_lora_load(AMA10LoRA *adapter,const char *path,const AMA10Model *base,
                    uint64_t base_hash,uint64_t tokenizer_hash);
void ama10_lora_apply(const AMA10LoRA *adapter,const float *hidden,float *logits);
void ama10_lora_free(AMA10LoRA *adapter);
#endif
