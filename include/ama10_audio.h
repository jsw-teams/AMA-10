#ifndef AMA10_AUDIO_H
#define AMA10_AUDIO_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    float *samples;
    size_t sample_count;
    unsigned sample_rate;
    unsigned channels;
} AMA10Audio;

/* Supports RIFF/WAVE PCM16 mono or stereo at 16 kHz. Stereo is averaged. */
int ama10_audio_load_wav16k(AMA10Audio *audio,const char *path);
void ama10_audio_free(AMA10Audio *audio);

/* 25 ms Hamming-windowed frames every 10 ms, zero-padding the final frame. */
int ama10_audio_frameify(const AMA10Audio *audio,float **frames,
                         size_t *frame_count,size_t *frame_width);

/* 40 normalized log-mel power bands from a 512-point FFT; 25 ms windows,
   10 ms hop. The fixed transform is causal per frame and needs no future stats. */
int ama10_audio_log_mel(const AMA10Audio *audio,float **features,
                        size_t *frame_count,size_t *feature_width);

/* Average-pools contiguous frames into a requested number of monotonic steps. */
int ama10_audio_align_features(const float *frames,size_t frame_count,
                               size_t feature_width,size_t steps,float **aligned);

/* CTC forward-backward over frame-major logits. Returns negative log sequence
   probability and per-frame class posteriors in a length x vocabulary array. */
int ama10_audio_ctc_posteriors(const float *logits,size_t frame_count,size_t vocabulary,
                               const uint32_t *tokens,size_t token_count,uint32_t blank_id,
                               float *posteriors,double *negative_log_likelihood);

#endif
