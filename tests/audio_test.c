#include "ama10_audio.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int write_u16(FILE *f,uint16_t x){unsigned char b[2]={(unsigned char)x,(unsigned char)(x>>8)};return fwrite(b,1,2,f)==2;}
static int write_u32(FILE *f,uint32_t x){unsigned char b[4]={(unsigned char)x,(unsigned char)(x>>8),(unsigned char)(x>>16),(unsigned char)(x>>24)};return fwrite(b,1,4,f)==4;}

int main(void){
    FILE *f=fopen("build/audio-fixture.wav","wb");if(!f)return 1;
    int ok=fwrite("RIFF",1,4,f)==4&&write_u32(f,44)&&fwrite("WAVEfmt ",1,8,f)==8&&
        write_u32(f,16)&&write_u16(f,1)&&write_u16(f,2)&&write_u32(f,16000)&&
        write_u32(f,64000)&&write_u16(f,4)&&write_u16(f,16)&&
        fwrite("data",1,4,f)==4&&write_u32(f,8)&&write_u16(f,32767)&&
        write_u16(f,32767)&&write_u16(f,(uint16_t)(int16_t)-32768)&&
        write_u16(f,(uint16_t)(int16_t)-16384);
    if(fclose(f)!=0||!ok)return 2;
    AMA10Audio audio={0};float *frames=NULL;size_t count=0,width=0;
    if(!ama10_audio_load_wav16k(&audio,"build/audio-fixture.wav")||
       audio.sample_rate!=16000||audio.channels!=2||audio.sample_count!=2||
       fabsf(audio.samples[0]-32767.0f/32768.0f)>1e-6f||
       fabsf(audio.samples[1]-(-0.75f))>1e-6f||
       !ama10_audio_frameify(&audio,&frames,&count,&width)||count!=1||width!=400||
       fabsf(frames[0]-audio.samples[0]*0.08f)>1e-6f||fabsf(frames[1]-audio.samples[1]*
          (0.54f-0.46f*cosf(6.2831853071795864769f/399.0f)))>1e-6f||
       fabsf(frames[399])>1e-7f){
        fprintf(stderr,"audio fixture mismatch rate=%u channels=%u samples=%zu width=%zu frames=%zu s0=%g s1=%g f0=%g f1=%g f399=%g\n",
            audio.sample_rate,audio.channels,audio.sample_count,width,count,
            audio.sample_count?audio.samples[0]:0.0f,audio.sample_count>1?audio.samples[1]:0.0f,
            frames?frames[0]:0.0f,frames&&width>1?frames[1]:0.0f,frames&&width>399?frames[399]:0.0f);
        return 3;
    }
    free(frames);ama10_audio_free(&audio);
    {
        AMA10Audio tone={0};float *mel=NULL;size_t mel_frames=0,mel_width=0;
        tone.sample_rate=16000;tone.channels=1;tone.sample_count=800;
        tone.samples=(float*)malloc(tone.sample_count*sizeof(float));if(!tone.samples)return 4;
        for(size_t i=0;i<tone.sample_count;i++)tone.samples[i]=0.5f*sinf(
            6.2831853071795864769f*1000.0f*(float)i/16000.0f);
        if(!ama10_audio_log_mel(&tone,&mel,&mel_frames,&mel_width)||
           mel_frames!=5||mel_width!=40||
           !(mel[2*mel_width+14]>mel[2*mel_width+39]))return 5;
        for(size_t i=0;i<mel_frames*mel_width;i++)if(!isfinite(mel[i]))return 6;
        free(mel);ama10_audio_free(&tone);
    }
    {
        const float frames[]={1,2,3,4,5,6};float *aligned=NULL;
        if(!ama10_audio_align_features(frames,3,2,2,&aligned)||
           aligned[0]!=2||aligned[1]!=3||aligned[2]!=5||aligned[3]!=6)return 7;
        free(aligned);
    }
    {
        float logits[]={0.1f,0.7f,-0.2f, 0.4f,-0.1f,0.8f,
                        -0.3f,0.6f,0.2f, 0.5f,0.3f,-0.4f};
        const uint32_t transcript[]={1,2};float posterior[12];double nll=0.0;
        if(!ama10_audio_ctc_posteriors(logits,4,3,transcript,2,0,posterior,&nll)||
           !(nll>0.0)||!isfinite(nll))return 8;
        for(size_t t=0;t<4;t++){
            double row=posterior[t*3]+posterior[t*3+1]+posterior[t*3+2];
            if(fabs(row-1.0)>1e-5)return 9;
        }
        for(size_t i=0;i<12;i++){
            float old=logits[i],eps=0.001f;double plus=0.0,minus=0.0;
            logits[i]=old+eps;
            if(!ama10_audio_ctc_posteriors(logits,4,3,transcript,2,0,posterior,&plus))return 10;
            logits[i]=old-eps;
            if(!ama10_audio_ctc_posteriors(logits,4,3,transcript,2,0,posterior,&minus))return 11;
            logits[i]=old;
            if(!ama10_audio_ctc_posteriors(logits,4,3,transcript,2,0,posterior,&nll))return 12;
            double maximum=logits[(i/3)*3],sum=0.0;
            for(size_t k=1;k<3;k++)if(logits[(i/3)*3+k]>maximum)maximum=logits[(i/3)*3+k];
            for(size_t k=0;k<3;k++)sum+=exp(logits[(i/3)*3+k]-maximum);
            double derivative=exp(logits[i]-maximum)/sum-posterior[i];
            if(fabs((plus-minus)/(2.0*eps)-derivative)>0.003)return 13;
        }
        {
            float short_logits[6]={0,1,0,0,0,1},short_post[6];double impossible=0.0;
            const uint32_t repeated[]={1,1};
            if(ama10_audio_ctc_posteriors(short_logits,2,3,repeated,2,0,short_post,&impossible))return 14;
        }
    }
    puts("RIFF PCM16 decoding, stereo fold-down, log-mel extraction, and causal frame alignment: ok");
    return 0;
}
