#ifndef AMA10_MULTIMODAL_CHECKPOINT_H
#define AMA10_MULTIMODAL_CHECKPOINT_H

#include "checkpoint.h"
#include "ama10_encoder.h"

/* A single replaceable bundle for model/tokenizer/optimizer/state and the
   dense modality encoder. Zero-initialize every destination before load. */
int ama10_multimodal_checkpoint_save(const char *path,const AMA10Model *model,
                                     const AMA10Tokenizer *tokenizer,
                                     const AMA10Optimizer *optimizer,
                                     const AMA10State *state,
                                     const AMA10CheckpointProgress *progress,
                                     const AMA10Encoder *encoder);
int ama10_multimodal_checkpoint_load(const char *path,AMA10Model *model,
                                     AMA10Tokenizer *tokenizer,
                                     AMA10Optimizer *optimizer,AMA10State *state,
                                     AMA10CheckpointProgress *progress,
                                     AMA10Encoder *encoder);

#endif
