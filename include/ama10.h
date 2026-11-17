#ifndef AMA10_H
#define AMA10_H

#include <stddef.h>
#include <stdint.h>

struct AMA10OpenCL;

/* A small shared recurrent cell with explicit, caller-owned sequence state. */
typedef struct {
    size_t width;
    size_t input_width;
    float *wx;       /* width x input_width */
    float *wh;       /* width x width */
    float *wm;       /* width x width: slow-memory read */
    float *wu;       /* width x width: gated update proposal */
    float *bias;
    float *gate_bias;
    float *embed;    /* input_width x vocabulary */
    float *head;     /* vocabulary x width */
    size_t vocabulary;
} AMA10Model;

typedef struct {
    float **first_moment;
    float **second_moment;
    size_t parameter_count;
    uint64_t step;
    uint32_t algorithm;
} AMA10Optimizer;

enum { AMA10_OPT_ADAMW=0, AMA10_OPT_MUON=1 };

typedef struct {
    float *fast;
    float *slow;
    size_t width;
    unsigned steps;
    float *input_projection; /* reusable inference workspace; not checkpoint state */
    float *scratch_next;
    float *scratch_proposal;
} AMA10State;

typedef struct {
    float **values;
    size_t parameter_count;
    size_t width,input_width,vocabulary;
    uint64_t tokens;
} AMA10Gradients;

int ama10_model_init(AMA10Model *m, size_t vocabulary, size_t width,
                     size_t input_width, uint64_t seed);
int ama10_model_resize_vocabulary(AMA10Model *model,size_t new_vocabulary,uint64_t seed);
void ama10_model_free(AMA10Model *m);
int ama10_model_save(const AMA10Model *m, const char *path);
int ama10_model_save_safetensors(const AMA10Model *m,const char *path);
/* Zero-initialize the destination before loading. */
int ama10_model_load(AMA10Model *m, const char *path);
int ama10_model_load_safetensors(AMA10Model *m,const char *path);
int ama10_state_init(AMA10State *s, size_t width);
void ama10_state_reset(AMA10State *s);
void ama10_state_free(AMA10State *s);
int ama10_step(const AMA10Model *m, AMA10State *s, uint32_t token,
               float *logits, unsigned max_depth, float halt_threshold);
/* Same causal step with the final fast-state vector returned for lightweight adapters. */
int ama10_step_hidden(const AMA10Model *m,AMA10State *s,uint32_t token,float *logits,
                      float *hidden,unsigned max_depth,float halt_threshold);
/* Feed a dense modality feature vector through the same recurrent core/head.
   The feature vector must contain input_width finite values. */
int ama10_step_features(const AMA10Model *m,AMA10State *s,const float *features,
                        float *logits,unsigned max_depth,float halt_threshold);
int ama10_optimizer_init(AMA10Optimizer *opt, const AMA10Model *m);
int ama10_optimizer_set_algorithm(AMA10Optimizer *opt,uint32_t algorithm);
int ama10_optimizer_resize_vocabulary(AMA10Optimizer *optimizer,
                                      const AMA10Model *old_model,
                                      const AMA10Model *new_model);
void ama10_optimizer_free(AMA10Optimizer *opt);
int ama10_gradients_init(AMA10Gradients *gradients,const AMA10Model *model);
void ama10_gradients_reset(AMA10Gradients *gradients);
void ama10_gradients_free(AMA10Gradients *gradients);
float ama10_learning_rate(float base_rate,uint64_t optimizer_update,
                          float inverse_sqrt_decay);
/* Applies the token-mean accumulated gradient, clips its global norm, updates
   AdamW once, then clears the accumulator. */
int ama10_optimizer_apply(AMA10Model *model,AMA10Optimizer *opt,
                      AMA10Gradients *gradients,float learning_rate,
                      float weight_decay);
/* Train one causal chunk; input[t] predicts target[t]. State carries across chunks;
   gradients stop at each call boundary (truncated BPTT). Pass NULL to start at zero. */
int ama10_train_sequence(AMA10Model *m, AMA10Optimizer *opt, AMA10State *state,
                         const uint32_t *input, const uint32_t *target,
                         size_t length, unsigned depth, float learning_rate,
                         float weight_decay, float *mean_loss);
/* Adaptive ACT-style compute: stop after at least two recurrent steps when
   1/(1+mean(abs(h[d+1]-h[d]))) reaches halt_threshold. The discrete stopping
   decision is held fixed during BPTT; gradients cover every executed step. */
int ama10_train_sequence_adaptive(AMA10Model *m, AMA10Optimizer *opt,
                                  AMA10State *state,const uint32_t *input,
                                  const uint32_t *target,size_t length,
                                  unsigned max_depth,float halt_threshold,
                                  float learning_rate,float weight_decay,
                                  float *mean_loss,float *mean_depth);
/* TBPTT sequence update with no parameter update; gradients are summed into the
   caller-owned accumulator and normalized by its total token count at apply. */
int ama10_train_sequence_accumulate(AMA10Model *m,AMA10Optimizer *opt,
                                    AMA10State *state,const uint32_t *input,
                                    const uint32_t *target,size_t length,
                                    unsigned depth,AMA10Gradients *gradients,
                                    float *mean_loss);
int ama10_train_sequence_accumulate_opencl(AMA10Model *m,AMA10Optimizer *opt,
                                    AMA10State *state,const uint32_t *input,
                                    const uint32_t *target,size_t length,
                                    unsigned depth,AMA10Gradients *gradients,
                                    struct AMA10OpenCL *backend,float *mean_loss);
/* Train targets from dense per-step input features (length x input_width).
   feature_gradients receives d(mean_loss)/d(input_features) for the caller's
   modality encoder to backpropagate; model parameters use the normal optimizer. */
int ama10_train_feature_sequence(AMA10Model *m,AMA10Optimizer *opt,AMA10State *state,
                                 const float *input_features,const uint32_t *target,
                                 size_t length,unsigned depth,float learning_rate,
                                 float weight_decay,float *feature_gradients,
                                 float *mean_loss);
/* Mixed token/modality context with optional target mask for prompt-only steps.
   feature_mask[t]!=0 selects input_features[t]; otherwise token_inputs[t] is used.
   loss_mask[t]!=0 (or a NULL mask) trains target[t]. Returned feature gradients
   are normalized by the supervised-target count and zero at token-only steps. */
int ama10_train_multimodal_sequence(AMA10Model *m,AMA10Optimizer *opt,AMA10State *state,
                                    const uint32_t *token_inputs,const float *feature_inputs,
                                    const unsigned char *feature_mask,const unsigned char *loss_mask,
                                    const uint32_t *targets,size_t length,unsigned depth,
                                    float learning_rate,float weight_decay,
                                    float *feature_gradients,float *mean_loss);
/* Train against a normalized soft target distribution at each supervised step.
   This supports sequence criteria (for example CTC forward-backward) that produce
   posterior targets. soft_targets is length x vocabulary, row-major. */
int ama10_train_soft_multimodal_sequence(AMA10Model *m,AMA10Optimizer *opt,AMA10State *state,
                                    const uint32_t *token_inputs,const float *feature_inputs,
                                    const unsigned char *feature_mask,const unsigned char *loss_mask,
                                    const float *soft_targets,size_t length,unsigned depth,
                                    float learning_rate,float weight_decay,
                                    float *feature_gradients,float *mean_loss);
int ama10_eval_loss(const AMA10Model *m, const uint32_t *input,
                    const uint32_t *target, size_t length,
                    unsigned depth, double *mean_loss);
/* Adaptive inference evaluation reports mean recurrent microsteps per token. */
int ama10_eval_loss_adaptive(const AMA10Model *m,const uint32_t *input,
                             const uint32_t *target,size_t length,
                             unsigned max_depth,float halt_threshold,
                             double *mean_loss,double *mean_depth);
int ama10_eval_feature_loss(const AMA10Model *m,const float *input_features,
                            const uint32_t *target,size_t length,unsigned depth,
                            double *mean_loss);

#endif
