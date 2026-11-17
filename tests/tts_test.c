#include "ama10_audio.h"
#include "ama10_tts.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc,char **argv){
    if(argc!=2){fprintf(stderr,"usage: %s <scratch-dir>\n",argv[0]);return 2;}
    float frames[6*40];for(size_t t=0;t<6;t++)for(size_t d=0;d<40;d++)frames[t*40+d]=(float)t*0.2f+(float)d*0.01f;
    AMA10TTSCodebook cb={0},loaded={0};int fit=ama10_tts_fit_codebook(frames,6,40,3,UINT64_C(77),&cb)&&cb.count==3;
    cb.frames_per_byte=2.5f;
    uint32_t q=0;int quant=fit&&ama10_tts_quantize(&cb,frames,&q)&&q>=1&&q<=cb.count;
    char vq[4096],wav[4096];int paths=snprintf(vq,sizeof(vq),"%s/tts-test.vq",argv[1])>0&&snprintf(wav,sizeof(wav),"%s/tts-test.wav",argv[1])>0;
    int persistence=paths&&quant&&ama10_tts_codebook_save(&cb,vq)&&ama10_tts_codebook_load(&loaded,vq)&&loaded.train_hash==77&&loaded.count==3&&fabsf(loaded.frames_per_byte-2.5f)<1.0e-6f;
    uint8_t codes[4]={1,2,3,2};int synth=paths&&persistence&&ama10_tts_synthesize_wav16k(&loaded,codes,4,wav);
    AMA10Audio audio={0};int reload=synth&&ama10_audio_load_wav16k(&audio,wav)&&audio.sample_rate==16000&&audio.sample_count>400;
    if(reload)for(size_t i=0;i<audio.sample_count;i++)if(!isfinite(audio.samples[i])){reload=0;break;}
    int ok=fit&&quant&&persistence&&synth&&reload;
    ama10_audio_free(&audio);ama10_tts_codebook_free(&loaded);ama10_tts_codebook_free(&cb);if(ok){remove(vq);remove(wav);}
    if(!ok){fprintf(stderr,"TTS VQ test failed stages: fit=%d quantize=%d persist=%d synth=%d wav_reload=%d\n",fit,quant,persistence,synth,reload);return 1;}
    puts("TTS codebook fitting, checksum persistence, mel-code quantization, Griffin-Lim synthesis, and PCM16 WAV reload: ok");return 0;
}
