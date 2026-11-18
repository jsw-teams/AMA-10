#ifndef AMA10_OPENCL_BACKEND_H
#define AMA10_OPENCL_BACKEND_H

#include "ama10.h"
#include "ama10_affine_scan.h"

typedef struct AMA10OpenCL AMA10OpenCL;

AMA10OpenCL *ama10_opencl_create(const AMA10Model *model,const char *kernel_path);
/* Select the largest square embedding/core width that fits the target device's
   queried global-memory and max-allocation budgets for this train shape. */
size_t ama10_opencl_recommend_width(size_t vocabulary,size_t sequence_length,
                                    unsigned depth,size_t memory_budget_bytes,
                                    size_t *selected_budget_bytes);
/* Batch-independent affine prefix state scan, used by the shared model path. */
int ama10_opencl_affine_scan(AMA10OpenCL *backend,const float *a,const float *b,
                             const float *initial,size_t length,float *states);
/* Batched input projection and affine gate/proposal computation for one whole
   training sequence. Inputs/outputs are sequence-major; model parameters are
   synchronized from the supplied model before execution. */
int ama10_opencl_affine_inputs(AMA10OpenCL *backend,const AMA10Model *model,
                               const float *inputs,size_t length,float *projected,
                               float *gates,float *candidates);
int ama10_opencl_reason_sequence(AMA10OpenCL *backend,const AMA10Model *model,
                                 const float *context,size_t length,unsigned depth,
                                 float *reason_states); /* [length][depth][width], q1..qdepth */
int ama10_opencl_reason_backward(AMA10OpenCL *backend,const float *reason_states,
                                 const float *final_state_gradient,size_t length,
                                 unsigned depth,float *wu_gradient,float *wm_gradient,
                                 float *context_gradient);
int ama10_opencl_affine_backward(AMA10OpenCL *backend,const AMA10Model *model,
                                 const float *inputs,const float *projected,
                                 const float *gates,const float *candidates,
                                 const float *fast,const float *slow,
                                 const float *memory_lambda,const float *fast_lambda,
                                 const float *initial_fast,const float *initial_slow,
                                 size_t length,float *wx_gradient,float *wh_gradient,
                                 float *bias_gradient,float *gate_bias_gradient,
                                 float *input_gradient);
int ama10_opencl_step(AMA10OpenCL *backend,AMA10State *state,uint32_t token,
                      float *logits,unsigned max_depth,float halt_threshold);
/* Device-resident streaming decode. Call reset before a new sequence; recurrent
   state stays on the GPU and only next-token logits return to the host. */
int ama10_opencl_reset(AMA10OpenCL *backend);
int ama10_opencl_step_stream(AMA10OpenCL *backend,uint32_t token,float *logits,
                             unsigned max_depth,float halt_threshold);
/* Hybrid training update: CPU-produced accumulated gradients, OpenCL AdamW.
   Host weights track each successful update; call sync before checkpointing to
   copy device optimizer moments back to the host optimizer object. */
int ama10_opencl_adamw_apply(AMA10OpenCL *backend,AMA10Model *model,
                             AMA10Optimizer *optimizer,AMA10Gradients *gradients,
                             float learning_rate,float weight_decay);
/* GPU cosine/norm statistics, constrained min-norm merge, clipping, and the
   AdamW update for two separately accumulated shared-core task gradients. */
int ama10_opencl_adamw_apply_tasks(AMA10OpenCL *backend,AMA10Model *model,
                             AMA10Optimizer *optimizer,AMA10Gradients *task_a,
                             AMA10Gradients *task_b,float learning_rate,
                             float weight_decay,double *cosine,double *task_a_weight);
int ama10_opencl_adamw_sync(AMA10OpenCL *backend,AMA10Model *model,
                            AMA10Optimizer *optimizer);
/* OpenCL output-head matrix operations used by the recurrent BPTT path. */
int ama10_opencl_head_forward(AMA10OpenCL *backend,const float *head,
                              const float *hidden,size_t length,float *logits);
int ama10_opencl_head_backward(AMA10OpenCL *backend,const float *head,
                               const float *hidden,const float *logit_errors,
                               size_t length,float *head_gradient,
                               float *hidden_gradient);
int ama10_opencl_train_forward(AMA10OpenCL *backend,const AMA10Model *model,
                               const uint32_t *tokens,size_t length,unsigned depth,
                               const float *initial_fast,const float *initial_slow,
                               float *sequence_hidden,float *final_slow);
int ama10_opencl_train_forward_multimodal(AMA10OpenCL *backend,const AMA10Model *model,
                               const uint32_t *tokens,const float *features,
                               const unsigned char *feature_mask,size_t length,unsigned depth,
                               const float *initial_fast,const float *initial_slow,
                               float *sequence_hidden,float *final_slow);
int ama10_opencl_eval_logits(AMA10OpenCL *backend,const AMA10Model *model,
                               const uint32_t *tokens,const float *features,
                               const unsigned char *feature_mask,size_t length,unsigned depth,
                               const float *initial_fast,const float *initial_slow,
                               float *logits,float *final_fast,float *final_slow);
int ama10_opencl_bptt_begin(AMA10OpenCL *backend,const AMA10Model *model,
                           const float *trajectory,const uint32_t *tokens,
                           size_t length,unsigned depth,const float *hidden_gradients);
int ama10_opencl_bptt_step(AMA10OpenCL *backend,size_t time,unsigned layer,
                           int add_hidden_gradient);
int ama10_opencl_bptt_finish(AMA10OpenCL *backend,AMA10Gradients *gradients,
                             float *feature_gradients);
const char *ama10_opencl_device_name(const AMA10OpenCL *backend);
void ama10_opencl_destroy(AMA10OpenCL *backend);

#endif
