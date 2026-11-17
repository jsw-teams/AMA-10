#ifndef AMA10_TTS_H
#define AMA10_TTS_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t width;
    uint32_t count;
    uint64_t train_hash;
    float frames_per_byte;
    float *centers;
} AMA10TTSCodebook;

int ama10_tts_fit_codebook(const float *frames,size_t frame_count,size_t width,
                           uint32_t requested_count,uint64_t train_hash,
                           AMA10TTSCodebook *codebook);
void ama10_tts_codebook_free(AMA10TTSCodebook *codebook);
int ama10_tts_codebook_save(const AMA10TTSCodebook *codebook,const char *path);
int ama10_tts_codebook_load(AMA10TTSCodebook *codebook,const char *path);
int ama10_tts_quantize(const AMA10TTSCodebook *codebook,const float *frame,
                       uint32_t *code);
/* Codes are one-based codebook IDs; zero is the text/audio separator and 255 EOS. */
int ama10_tts_synthesize_wav16k(const AMA10TTSCodebook *codebook,
                                const uint8_t *codes,size_t code_count,
                                const char *wav_path);

#endif
