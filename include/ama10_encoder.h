#ifndef AMA10_ENCODER_H
#define AMA10_ENCODER_H

#include "ama10.h"

/* Trainable dense modality frontend. Inputs are samples x input_width and
   outputs are tanh features with the model's input_width. */
typedef struct {
    size_t input_width;
    size_t output_width;
    float *weights;          /* output_width x input_width */
    float *bias;             /* output_width */
    float *first_moment;     /* AdamW */
    float *second_moment;    /* AdamW */
    float *bias_first_moment;
    float *bias_second_moment;
    uint64_t step;
} AMA10Encoder;

int ama10_encoder_init(AMA10Encoder *encoder,size_t input_width,
                       size_t output_width,uint64_t seed);
void ama10_encoder_free(AMA10Encoder *encoder);
int ama10_encoder_forward(const AMA10Encoder *encoder,const float *inputs,
                          size_t samples,float *features);
/* output_gradients is dL/dfeatures. Updates encoder parameters with clipped
   AdamW; optional input_gradients receives dL/dinputs. */
int ama10_encoder_backward_update(AMA10Encoder *encoder,const float *inputs,
                                  const float *features,const float *output_gradients,
                                  size_t samples,float learning_rate,float weight_decay,
                                  float *input_gradients);
int ama10_encoder_save(const AMA10Encoder *encoder,const char *path);
int ama10_encoder_load(AMA10Encoder *encoder,const char *path);

/* Jointly train the encoder and recurrent model on frame-aligned targets. */
int ama10_train_encoded_sequence(AMA10Model *model,AMA10Optimizer *optimizer,
                                 AMA10State *state,AMA10Encoder *encoder,
                                 const float *raw_inputs,const uint32_t *targets,
                                 size_t length,unsigned depth,float learning_rate,
                                 float weight_decay,float *mean_loss);

/* Image-conditioned causal caption update. Image patches form an unsupervised
   feature prefix; the final image state predicts caption[0], then caption
   tokens predict their successors. Both encoder and recurrent core are trained. */
int ama10_train_image_caption(AMA10Model *model,AMA10Optimizer *optimizer,
                              AMA10State *state,AMA10Encoder *encoder,
                              const float *raw_patches,size_t patch_count,
                              const uint32_t *caption,size_t caption_count,
                              unsigned depth,float learning_rate,float weight_decay,
                              float *mean_loss);
/* Optional text prefix (for example a language tag) is fed after the image;
   its final token predicts the first caption token. */
int ama10_train_image_caption_conditioned(AMA10Model *model,AMA10Optimizer *optimizer,
                              AMA10State *state,AMA10Encoder *encoder,
                              const float *raw_patches,size_t patch_count,
                              const uint32_t *conditioning_tokens,size_t conditioning_count,
                              const uint32_t *caption,size_t caption_count,
                              unsigned depth,float learning_rate,float weight_decay,
                              float *mean_loss);
/* General paired modality-to-text update: an arbitrary dense feature sequence
   and optional text prefix condition causal next-token loss. */
int ama10_train_feature_conditioned_text(AMA10Model *model,AMA10Optimizer *optimizer,
                              AMA10State *state,AMA10Encoder *encoder,
                              const float *raw_features,size_t feature_count,
                              const uint32_t *conditioning_tokens,size_t conditioning_count,
                              const uint32_t *text,size_t text_count,
                              unsigned depth,float learning_rate,float weight_decay,
                              float *mean_loss);
/* Monotonic modality-to-text training. One dense feature context is paired with
   each target token; previous target tokens are interleaved as recurrent input. */
int ama10_train_aligned_feature_text(AMA10Model *model,AMA10Optimizer *optimizer,
                              AMA10State *state,AMA10Encoder *encoder,
                              const float *raw_aligned_features,size_t steps,
                              const uint32_t *conditioning_tokens,size_t conditioning_count,
                              const uint32_t *text,size_t text_count,
                              unsigned depth,float learning_rate,float weight_decay,
                              float *mean_loss);

#endif
