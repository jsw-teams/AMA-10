#ifndef AMA10_CHECKPOINT_H
#define AMA10_CHECKPOINT_H

#include "ama10.h"
#include "tokenizer.h"

typedef struct {
    uint64_t completed_updates;
    uint64_t cursor;
    uint64_t train_hash;
    uint64_t validation_hash;
    uint64_t tokenizer_sample_bytes;
    uint32_t sequence_length;
    uint32_t depth;
    uint32_t accumulation_steps;
    uint32_t target_vocabulary;
    uint32_t optimizer_algorithm;
    float learning_rate;
    float learning_rate_decay;
    float weight_decay;
} AMA10CheckpointProgress;

/* Destinations must be zero-initialized before load. Checkpoints replace atomically. */
int ama10_checkpoint_save(const char *path,const AMA10Model *model,
                          const AMA10Tokenizer *tokenizer,const AMA10Optimizer *optimizer,
                          const AMA10State *state,const AMA10CheckpointProgress *progress);
int ama10_checkpoint_load(const char *path,AMA10Model *model,AMA10Tokenizer *tokenizer,
                          AMA10Optimizer *optimizer,AMA10State *state,
                          AMA10CheckpointProgress *progress);
uint64_t ama10_data_hash(const unsigned char *data,size_t length);

#endif
