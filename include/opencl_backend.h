#ifndef AMA10_OPENCL_BACKEND_H
#define AMA10_OPENCL_BACKEND_H

#include "ama10.h"

typedef struct AMA10OpenCL AMA10OpenCL;

AMA10OpenCL *ama10_opencl_create(const AMA10Model *model,const char *kernel_path);
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
int ama10_opencl_bptt_begin(AMA10OpenCL *backend,const AMA10Model *model,
                           const float *trajectory,const uint32_t *tokens,
                           size_t length,unsigned depth,const float *hidden_gradients);
int ama10_opencl_bptt_step(AMA10OpenCL *backend,size_t time,unsigned layer,
                           int add_hidden_gradient);
int ama10_opencl_bptt_finish(AMA10OpenCL *backend,AMA10Gradients *gradients);
const char *ama10_opencl_device_name(const AMA10OpenCL *backend);
void ama10_opencl_destroy(AMA10OpenCL *backend);

#endif
