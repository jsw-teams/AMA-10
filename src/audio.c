#include "ama10_audio.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double ctc_logadd(double a,double b){
    if(a==-INFINITY)return b;
    if(b==-INFINITY)return a;
    double m=a>b?a:b;return m+log1p(exp((a>b?b:a)-m));
}
int ama10_audio_ctc_posteriors(const float *logits,size_t tcount,size_t vocab,
                               const uint32_t *tokens,size_t ucount,uint32_t blank,
                               float *post,double *nll){
    if(!logits||!tcount||!vocab||blank>=vocab||!post||!nll||
       (ucount&&!tokens)||ucount>(SIZE_MAX-1)/2)return 0;
    size_t states=2*ucount+1;
    if(tcount>SIZE_MAX/vocab||tcount*vocab>SIZE_MAX/sizeof(float)||
       tcount>SIZE_MAX/states||tcount*states>SIZE_MAX/sizeof(double))return 0;
    for(size_t i=0;i<ucount;i++)if(tokens[i]>=vocab||tokens[i]==blank)return 0;
    uint32_t *labels=(uint32_t*)malloc(states*sizeof(uint32_t));
    double *logp=(double*)malloc(tcount*vocab*sizeof(double));
    double *alpha=(double*)malloc(tcount*states*sizeof(double));
    double *beta=(double*)malloc(tcount*states*sizeof(double));
    if(!labels||!logp||!alpha||!beta){free(labels);free(logp);free(alpha);free(beta);return 0;}
    for(size_t s=0;s<states;s++)labels[s]=(s&1)?tokens[s/2]:blank;
    for(size_t t=0;t<tcount;t++){
        float mx=-INFINITY;for(size_t k=0;k<vocab;k++){
            float x=logits[t*vocab+k];if(!isfinite(x))goto fail;if(x>mx)mx=x;
        }
        double sum=0;for(size_t k=0;k<vocab;k++)sum+=exp((double)logits[t*vocab+k]-mx);
        double norm=(double)mx+log(sum);
        for(size_t k=0;k<vocab;k++)logp[t*vocab+k]=(double)logits[t*vocab+k]-norm;
    }
    for(size_t i=0;i<tcount*states;i++)alpha[i]=-INFINITY;
    alpha[0]=logp[blank];
    if(states>1)alpha[1]=logp[labels[1]];
    for(size_t t=1;t<tcount;t++)for(size_t s=0;s<states;s++){
        double a=alpha[(t-1)*states+s];
        if(s)a=ctc_logadd(a,alpha[(t-1)*states+s-1]);
        if(s>=2&&labels[s]!=blank&&labels[s]!=labels[s-2])
            a=ctc_logadd(a,alpha[(t-1)*states+s-2]);
        alpha[t*states+s]=a+logp[t*vocab+labels[s]];
    }
    for(size_t i=0;i<tcount*states;i++)beta[i]=-INFINITY;
    beta[(tcount-1)*states+states-1]=0.0;
    if(states>1)beta[(tcount-1)*states+states-2]=0.0;
    for(size_t rt=tcount-1;rt>0;rt--){size_t t=rt-1;
        for(size_t s=0;s<states;s++){
            double b=beta[(t+1)*states+s]+logp[(t+1)*vocab+labels[s]];
            if(s+1<states)b=ctc_logadd(b,beta[(t+1)*states+s+1]+logp[(t+1)*vocab+labels[s+1]]);
            if(s+2<states&&labels[s+2]!=blank&&labels[s+2]!=labels[s])
                b=ctc_logadd(b,beta[(t+1)*states+s+2]+logp[(t+1)*vocab+labels[s+2]]);
            beta[t*states+s]=b;
        }
    }
    {
        double z=alpha[(tcount-1)*states+states-1];
        if(states>1)z=ctc_logadd(z,alpha[(tcount-1)*states+states-2]);
        if(!isfinite(z))goto fail;
        memset(post,0,tcount*vocab*sizeof(float));
        for(size_t t=0;t<tcount;t++)for(size_t s=0;s<states;s++)
            post[t*vocab+labels[s]]+=(float)exp(alpha[t*states+s]+beta[t*states+s]-z);
        *nll=-z;
    }
    free(labels);free(logp);free(alpha);free(beta);return 1;
fail:
    free(labels);free(logp);free(alpha);free(beta);return 0;
}

#define WAV_SAMPLE_RATE 16000u
#define AUDIO_FRAME 400u
#define AUDIO_HOP 160u
#define AUDIO_FFT 512u
#define AUDIO_MELS 40u

static uint16_t read_u16(const unsigned char *p){return (uint16_t)p[0]|((uint16_t)p[1]<<8);}
static uint32_t read_u32(const unsigned char *p){return (uint32_t)p[0]|((uint32_t)p[1]<<8)|
    ((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);}
static int read_exact(FILE *f,void *dst,size_t n){return fread(dst,1,n,f)==n;}

void ama10_audio_free(AMA10Audio *audio){
    if(!audio)return;
    free(audio->samples);memset(audio,0,sizeof(*audio));
}

int ama10_audio_load_wav16k(AMA10Audio *audio,const char *path){
    if(!audio||!path)return 0;
    FILE *f=fopen(path,"rb");if(!f)return 0;
    unsigned char header[12];int ok=read_exact(f,header,sizeof(header))&&
        !memcmp(header,"RIFF",4)&&!memcmp(header+8,"WAVE",4);
    uint16_t format=0,channels=0,bits=0,block_align=0;uint32_t sample_rate=0,byte_rate=0;
    unsigned char *pcm=NULL;size_t pcm_size=0;int have_fmt=0,have_data=0;
    while(ok&&(!have_fmt||!have_data)){
        unsigned char chunk[8];
        if(!read_exact(f,chunk,sizeof(chunk))){ok=0;break;}
        uint32_t chunk_size=read_u32(chunk+4);
#if SIZE_MAX < UINT32_MAX
        if(chunk_size>(uint32_t)SIZE_MAX){ok=0;break;}
#endif
        if(chunk_size>(uint32_t)LONG_MAX){ok=0;break;}
        if(!memcmp(chunk,"fmt ",4)){
            if(chunk_size<16||chunk_size>4096){ok=0;break;}
            unsigned char fmt[16];
            if(!read_exact(f,fmt,sizeof(fmt))){ok=0;break;}
            format=read_u16(fmt);channels=read_u16(fmt+2);sample_rate=read_u32(fmt+4);
            byte_rate=read_u32(fmt+8);
            block_align=read_u16(fmt+12);bits=read_u16(fmt+14);have_fmt=1;
            if(fseek(f,(long)(chunk_size-16),SEEK_CUR)!=0){ok=0;break;}
        }else if(!memcmp(chunk,"data",4)){
            if(!chunk_size){ok=0;break;}
            pcm=(unsigned char*)malloc((size_t)chunk_size);
            if(!pcm||!read_exact(f,pcm,(size_t)chunk_size)){ok=0;break;}
            pcm_size=(size_t)chunk_size;have_data=1;
        }else if(fseek(f,(long)chunk_size,SEEK_CUR)!=0){ok=0;break;}
        if(chunk_size&1u)if(fseek(f,1,SEEK_CUR)!=0){ok=0;break;}
    }
    if(fclose(f)!=0)ok=0;
    if(!ok||format!=1||(channels!=1&&channels!=2)||sample_rate!=WAV_SAMPLE_RATE||
       bits!=16||block_align!=channels*2||byte_rate!=sample_rate*(uint32_t)block_align||
       pcm_size%block_align||!pcm_size){
        free(pcm);return 0;
    }
    size_t count=pcm_size/block_align;
    if(count>SIZE_MAX/sizeof(float)){free(pcm);return 0;}
    float *samples=(float*)malloc(count*sizeof(float));if(!samples){free(pcm);return 0;}
    for(size_t i=0;i<count;i++){
        int16_t left=(int16_t)read_u16(pcm+i*block_align);
        float value=(float)left/32768.0f;
        if(channels==2){int16_t right=(int16_t)read_u16(pcm+i*block_align+2);
            value=0.5f*(value+(float)right/32768.0f);}
        samples[i]=value;
    }
    free(pcm);ama10_audio_free(audio);audio->samples=samples;audio->sample_count=count;
    audio->sample_rate=sample_rate;audio->channels=channels;return 1;
}

int ama10_audio_frameify(const AMA10Audio *audio,float **frames,
                         size_t *frame_count,size_t *frame_width){
    if(!audio||!frames||!frame_count||!frame_width||!audio->samples||
       !audio->sample_count||audio->sample_rate!=WAV_SAMPLE_RATE)return 0;
    size_t count=1+(audio->sample_count-1)/AUDIO_HOP;
    if(count>SIZE_MAX/AUDIO_FRAME||count*AUDIO_FRAME>SIZE_MAX/sizeof(float))return 0;
    float *out=(float*)calloc(count*AUDIO_FRAME,sizeof(float));if(!out)return 0;
    for(size_t frame=0;frame<count;frame++)for(size_t j=0;j<AUDIO_FRAME;j++){
        size_t at=frame*AUDIO_HOP+j;
        if(at>=audio->sample_count)break;
        float window=0.54f-0.46f*cosf((float)(6.2831853071795864769*(double)j/
                                               (double)(AUDIO_FRAME-1)));
        out[frame*AUDIO_FRAME+j]=audio->samples[at]*window;
    }
    *frames=out;*frame_count=count;*frame_width=AUDIO_FRAME;return 1;
}

static void fft512(float *real,float *imag){
    size_t j=0;
    for(size_t i=1;i<AUDIO_FFT;i++){
        size_t bit=AUDIO_FFT>>1;
        while(j&bit){j^=bit;bit>>=1;}
        j^=bit;
        if(i<j){float t=real[i];real[i]=real[j];real[j]=t;
            t=imag[i];imag[i]=imag[j];imag[j]=t;}
    }
    for(size_t length=2;length<=AUDIO_FFT;length<<=1){
        float angle=-6.2831853071795864769f/(float)length;
        float step_real=cosf(angle),step_imag=sinf(angle);
        for(size_t base=0;base<AUDIO_FFT;base+=length){
            float wr=1.0f,wi=0.0f;size_t half=length/2;
            for(size_t k=0;k<half;k++){
                size_t even=base+k,odd=even+half;
                float tr=wr*real[odd]-wi*imag[odd];
                float ti=wr*imag[odd]+wi*real[odd];
                float er=real[even],ei=imag[even];
                real[even]=er+tr;imag[even]=ei+ti;
                real[odd]=er-tr;imag[odd]=ei-ti;
                float next_wr=wr*step_real-wi*step_imag;
                wi=wr*step_imag+wi*step_real;wr=next_wr;
            }
        }
    }
}
static float mel_to_hz(float mel){return 700.0f*(expf(mel/1127.01048f)-1.0f);}
int ama10_audio_log_mel(const AMA10Audio *audio,float **features,
                        size_t *frame_count,size_t *feature_width){
    if(!audio||!features||!frame_count||!feature_width||!audio->samples||
       !audio->sample_count||audio->sample_rate!=WAV_SAMPLE_RATE)return 0;
    size_t count=1+(audio->sample_count-1)/AUDIO_HOP;
    if(count>SIZE_MAX/AUDIO_MELS||count*AUDIO_MELS>SIZE_MAX/sizeof(float))return 0;
    float *out=(float*)malloc(count*AUDIO_MELS*sizeof(float));if(!out)return 0;
    const float max_mel=1127.01048f*log1pf(8000.0f/700.0f);
    float edges[AUDIO_MELS+2];size_t bins[AUDIO_MELS+2];
    for(size_t i=0;i<AUDIO_MELS+2;i++){
        edges[i]=max_mel*(float)i/(float)(AUDIO_MELS+1);
        float hz=mel_to_hz(edges[i]);
        size_t bin=(size_t)floorf(((float)AUDIO_FFT+1.0f)*hz/(float)WAV_SAMPLE_RATE);
        bins[i]=bin>AUDIO_FFT/2?AUDIO_FFT/2:bin;
    }
    for(size_t frame=0;frame<count;frame++){
        float real[AUDIO_FFT]={0},imag[AUDIO_FFT]={0};
        for(size_t j=0;j<AUDIO_FRAME;j++){
            size_t at=frame*AUDIO_HOP+j;if(at>=audio->sample_count)break;
            float window=0.54f-0.46f*cosf(6.2831853071795864769f*(float)j/(float)(AUDIO_FRAME-1));
            real[j]=audio->samples[at]*window;
        }
        fft512(real,imag);
        for(size_t m=0;m<AUDIO_MELS;m++){
            size_t left=bins[m],center=bins[m+1],right=bins[m+2];
            float energy=0.0f,weight_sum=0.0f;
            if(center>left)for(size_t k=left;k<center;k++){
                float weight=(float)(k-left)/(float)(center-left);
                float power=(real[k]*real[k]+imag[k]*imag[k])/(float)(AUDIO_FFT*AUDIO_FFT);
                energy+=weight*power;weight_sum+=weight;
            }
            if(right>center)for(size_t k=center;k<=right;k++){
                float weight=(float)(right-k)/(float)(right-center);
                float power=(real[k]*real[k]+imag[k]*imag[k])/(float)(AUDIO_FFT*AUDIO_FFT);
                energy+=weight*power;weight_sum+=weight;
            }
            if(weight_sum>0.0f)energy/=weight_sum;
            float value=(logf(energy+1.0e-8f)+6.0f)*0.25f;
            if(!isfinite(value)){free(out);return 0;}
            out[frame*AUDIO_MELS+m]=value;
        }
    }
    *features=out;*frame_count=count;*feature_width=AUDIO_MELS;return 1;
}

int ama10_audio_align_features(const float *frames,size_t frame_count,
                               size_t feature_width,size_t steps,float **aligned){
    if(!frames||!frame_count||!feature_width||!steps||!aligned||
       frame_count>SIZE_MAX/feature_width||steps>SIZE_MAX/feature_width||
       steps*feature_width>SIZE_MAX/sizeof(float))return 0;
    float *out=(float*)malloc(steps*feature_width*sizeof(float));if(!out)return 0;
    size_t quotient=frame_count/steps,remainder=frame_count%steps;
    for(size_t step=0;step<steps;step++){
        size_t start=step*quotient+(step<remainder?step:remainder);
        size_t next=step+1;
        size_t end=next*quotient+(next<remainder?next:remainder);
        if(end<=start){
            size_t source=start<frame_count?start:frame_count-1;
            memcpy(out+step*feature_width,frames+source*feature_width,
                   feature_width*sizeof(float));
        }else{
            for(size_t d=0;d<feature_width;d++){
                double sum=0.0;
                for(size_t frame=start;frame<end;frame++)sum+=frames[frame*feature_width+d];
                out[step*feature_width+d]=(float)(sum/(double)(end-start));
            }
        }
    }
    *aligned=out;return 1;
}
