#include "ama10_tts.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TTS_MELS 40u
#define TTS_FFT 512u
#define TTS_WINDOW 400u
#define TTS_HOP 160u
#define TTS_SAMPLE_RATE 16000u

static uint64_t tts_hash(const unsigned char *p,size_t n){
    uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<n;i++){h^=p[i];h*=UINT64_C(1099511628211);}return h;
}
static uint32_t tts_u32(const unsigned char b[4]){
    return (uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);
}
static void put_u32(unsigned char *b,uint32_t x){
    b[0]=(unsigned char)x;b[1]=(unsigned char)(x>>8);b[2]=(unsigned char)(x>>16);b[3]=(unsigned char)(x>>24);
}
static int write_u32(FILE *f,uint32_t x){unsigned char b[4];put_u32(b,x);return fwrite(b,1,4,f)==4;}
static int read_u32(FILE *f,uint32_t *x){unsigned char b[4];if(fread(b,1,4,f)!=4)return 0;*x=tts_u32(b);return 1;}
static int write_u64(FILE *f,uint64_t x){return write_u32(f,(uint32_t)x)&&write_u32(f,(uint32_t)(x>>32));}
static int read_u64(FILE *f,uint64_t *x){uint32_t lo,hi;if(!read_u32(f,&lo)||!read_u32(f,&hi))return 0;*x=(uint64_t)lo|((uint64_t)hi<<32);return 1;}

void ama10_tts_codebook_free(AMA10TTSCodebook *c){if(!c)return;free(c->centers);memset(c,0,sizeof(*c));}
int ama10_tts_fit_codebook(const float *frames,size_t frame_count,size_t width,
                           uint32_t requested_count,uint64_t train_hash,
                           AMA10TTSCodebook *out){
    if(!frames||!frame_count||!width||!requested_count||requested_count>254||!out||
       frame_count>SIZE_MAX/width||requested_count>SIZE_MAX/width||
       requested_count*width>SIZE_MAX/sizeof(float))return 0;
    size_t stride=frame_count>8192?(frame_count+8191)/8192:1;
    size_t samples=(frame_count+stride-1)/stride;
    if(samples>UINT32_MAX)return 0;
    uint32_t k=requested_count>(uint32_t)samples?(uint32_t)samples:requested_count;
    float *points=(float*)malloc(samples*width*sizeof(float));
    float *centers=(float*)malloc((size_t)k*width*sizeof(float));
    double *sums=(double*)malloc((size_t)k*width*sizeof(double));
    uint32_t *counts=(uint32_t*)malloc((size_t)k*sizeof(uint32_t));
    if(!points||!centers||!sums||!counts){free(points);free(centers);free(sums);free(counts);return 0;}
    size_t at=0;for(size_t i=0;i<frame_count;i+=stride){
        for(size_t d=0;d<width;d++){float x=frames[i*width+d];if(!isfinite(x)){free(points);free(centers);free(sums);free(counts);return 0;}points[at*width+d]=x;}at++;
    }
    for(uint32_t c=0;c<k;c++)memcpy(centers+(size_t)c*width,points+((size_t)c*samples/k)*width,width*sizeof(float));
    for(unsigned iteration=0;iteration<12;iteration++){
        memset(sums,0,(size_t)k*width*sizeof(double));memset(counts,0,(size_t)k*sizeof(uint32_t));
        for(size_t i=0;i<samples;i++){
            uint32_t best=0;double best_distance=INFINITY;
            for(uint32_t c=0;c<k;c++){
                double dist=0.0;for(size_t d=0;d<width;d++){
                    double delta=(double)points[i*width+d]-centers[(size_t)c*width+d];dist+=delta*delta;
                }
                if(dist<best_distance){best_distance=dist;best=c;}
            }
            if(counts[best]==UINT32_MAX){free(points);free(centers);free(sums);free(counts);return 0;}
            counts[best]++;for(size_t d=0;d<width;d++)sums[(size_t)best*width+d]+=points[i*width+d];
        }
        for(uint32_t c=0;c<k;c++)if(counts[c])for(size_t d=0;d<width;d++)
            centers[(size_t)c*width+d]=(float)(sums[(size_t)c*width+d]/counts[c]);
    }
    free(points);free(sums);free(counts);ama10_tts_codebook_free(out);
    out->width=(uint32_t)width;out->count=k;out->train_hash=train_hash;out->centers=centers;return 1;
}
int ama10_tts_codebook_save(const AMA10TTSCodebook *c,const char *path){
    if(!c||!c->centers||!path||!c->width||c->width>1024||!c->count||c->count>254||
       !isfinite(c->frames_per_byte)||c->frames_per_byte<=0.0f||c->frames_per_byte>100.0f||
       (size_t)c->width>SIZE_MAX/c->count||(size_t)c->width*c->count>SIZE_MAX/sizeof(float))return 0;
    FILE *f=fopen(path,"wb");if(!f)return 0;
    static const unsigned char magic[8]={'A','1','0','T','T','S','V','2'};
    size_t n=(size_t)c->width*c->count*sizeof(float);unsigned char hashbuf[20];
    put_u32(hashbuf,c->width);put_u32(hashbuf+4,c->count);
    for(size_t i=0;i<8;i++)hashbuf[8+i]=(unsigned char)(c->train_hash>>(8*i));
    uint32_t duration_bits=0;memcpy(&duration_bits,&c->frames_per_byte,sizeof(duration_bits));put_u32(hashbuf+16,duration_bits);
    uint64_t hash=tts_hash(hashbuf,sizeof(hashbuf));
    hash=tts_hash((const unsigned char*)c->centers,n)^hash;
    int ok=fwrite(magic,1,8,f)==8&&write_u32(f,2)&&write_u32(f,c->width)&&write_u32(f,c->count)&&
        write_u64(f,c->train_hash)&&write_u32(f,duration_bits)&&write_u64(f,hash)&&fwrite(c->centers,1,n,f)==n;
    if(fflush(f)!=0)ok=0;
    if(fclose(f)!=0)ok=0;
    return ok;
}
int ama10_tts_codebook_load(AMA10TTSCodebook *out,const char *path){
    if(!out||!path)return 0;
    FILE *f=fopen(path,"rb");if(!f)return 0;
    unsigned char magic[8],magic1[8]={'A','1','0','T','T','S','V','1'},magic2[8]={'A','1','0','T','T','S','V','2'};
    uint32_t version=0,width=0,count=0,duration_bits=0;uint64_t train_hash=0,stored=0;float frames_per_byte=0.0f;
    int ok=fread(magic,1,8,f)==8&&read_u32(f,&version)&&((version==1&&!memcmp(magic,magic1,8))||(version==2&&!memcmp(magic,magic2,8)))&&
        read_u32(f,&width)&&read_u32(f,&count)&&read_u64(f,&train_hash);
    if(ok&&version==2){ok=read_u32(f,&duration_bits);memcpy(&frames_per_byte,&duration_bits,sizeof(frames_per_byte));if(!isfinite(frames_per_byte)||frames_per_byte<=0.0f||frames_per_byte>100.0f)ok=0;}
    if(ok)ok=read_u64(f,&stored)&&
        width==TTS_MELS&&count>0&&count<=254&&(size_t)width<=SIZE_MAX/count&&
        (size_t)width*count<=SIZE_MAX/sizeof(float);
    size_t n=ok?(size_t)width*count*sizeof(float):0;float *centers=ok?(float*)malloc(n):NULL;
    if(ok&&(!centers||fread(centers,1,n,f)!=n||fgetc(f)!=EOF))ok=0;
    if(ok){unsigned char hbuf[20]={0};put_u32(hbuf,width);put_u32(hbuf+4,count);
        for(size_t i=0;i<8;i++)hbuf[8+i]=(unsigned char)(train_hash>>(8*i));
        if(version==2)put_u32(hbuf+16,duration_bits);
        uint64_t actual=0;
        if(version==1){unsigned char oldbuf[16];memcpy(oldbuf,hbuf,16);actual=tts_hash(oldbuf,sizeof(oldbuf))^tts_hash((const unsigned char*)centers,n);}
        else actual=tts_hash(hbuf,sizeof(hbuf))^tts_hash((const unsigned char*)centers,n);
        if(actual!=stored)ok=0;
    }
    if(fclose(f)!=0)ok=0;
    if(!ok){free(centers);return 0;}ama10_tts_codebook_free(out);
    out->width=width;out->count=count;out->train_hash=train_hash;out->frames_per_byte=frames_per_byte;out->centers=centers;return 1;
}
int ama10_tts_quantize(const AMA10TTSCodebook *c,const float *frame,uint32_t *code){
    if(!c||!c->centers||!frame||!code||!c->width||!c->count)return 0;
    uint32_t best=0;double best_distance=INFINITY;
    for(uint32_t k=0;k<c->count;k++){double d=0.0;
        for(uint32_t j=0;j<c->width;j++){if(!isfinite(frame[j]))return 0;double x=(double)frame[j]-c->centers[(size_t)k*c->width+j];d+=x*x;}
        if(d<best_distance){best_distance=d;best=k;}}
    *code=best+1;return 1;
}

static void fft(float *r,float *im,int inverse){
    for(size_t i=1,j=0;i<TTS_FFT;i++){
        size_t bit=TTS_FFT>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;
        if(i<j){float t=r[i];r[i]=r[j];r[j]=t;t=im[i];im[i]=im[j];im[j]=t;}}
    for(size_t len=2;len<=TTS_FFT;len<<=1){
        float angle=(inverse?2.0f:-2.0f)*3.14159265358979323846f/(float)len;
        float wr0=cosf(angle),wi0=sinf(angle);
        for(size_t base=0;base<TTS_FFT;base+=len){float wr=1.0f,wi=0.0f;size_t half=len>>1;
            for(size_t j=0;j<half;j++){size_t a=base+j,b=a+half;float tr=wr*r[b]-wi*im[b],ti=wr*im[b]+wi*r[b];
                r[b]=r[a]-tr;im[b]=im[a]-ti;r[a]+=tr;im[a]+=ti;float nw=wr*wr0-wi*wi0;wi=wr*wi0+wi*wr0;wr=nw;}}}
    if(inverse)for(size_t i=0;i<TTS_FFT;i++){r[i]/=(float)TTS_FFT;im[i]/=(float)TTS_FFT;}
}
static float hz_to_mel(float hz){return 1127.01048f*log1pf(hz/700.0f);}
static void make_window(float *w){for(size_t i=0;i<TTS_WINDOW;i++)w[i]=0.54f-0.46f*cosf(6.2831853071795864769f*(float)i/(float)(TTS_WINDOW-1));}
static int render_spectrum(const float *magnitudes,size_t frames,const float *phase,float *wave){
    if(frames>(SIZE_MAX-TTS_WINDOW)/TTS_HOP+1)return 0;
    size_t samples=(frames-1)*TTS_HOP+TTS_WINDOW;
    float *normal=(float*)calloc(samples,sizeof(float));if(!normal)return 0;
    float window[TTS_WINDOW];make_window(window);memset(wave,0,samples*sizeof(float));
    for(size_t t=0;t<frames;t++){
        float real[TTS_FFT]={0},imag[TTS_FFT]={0};
        for(size_t k=0;k<=TTS_FFT/2;k++){
            size_t at=t*(TTS_FFT/2+1)+k;float a=magnitudes[at],p=phase[at];
            real[k]=a*cosf(p);imag[k]=a*sinf(p);
            if(k&&k<TTS_FFT/2){real[TTS_FFT-k]=real[k];imag[TTS_FFT-k]=-imag[k];}
        }
        fft(real,imag,1);
        for(size_t j=0;j<TTS_WINDOW;j++){size_t ix=t*TTS_HOP+j;wave[ix]+=real[j]*window[j];normal[ix]+=window[j]*window[j];}
    }
    for(size_t i=0;i<samples;i++)if(normal[i]>1.0e-6f)wave[i]/=normal[i];
    free(normal);return 1;
}
static int write_wav(const char *path,const float *wave,size_t n){
    if(n>(UINT32_MAX-36u)/2u)return 0;
    double peak=0.0;
    for(size_t i=0;i<n;i++){if(!isfinite(wave[i]))return 0;double a=fabs((double)wave[i]);if(a>peak)peak=a;}
    double scale=peak>0.95?0.95/peak:1.0;FILE *f=fopen(path,"wb");if(!f)return 0;
    int ok=fwrite("RIFF",1,4,f)==4&&write_u32(f,(uint32_t)(36+n*2))&&fwrite("WAVEfmt ",1,8,f)==8&&
        write_u32(f,16)&&fwrite("\x01\0\x01\0",1,4,f)==4&&write_u32(f,TTS_SAMPLE_RATE)&&
        write_u32(f,TTS_SAMPLE_RATE*2)&&fwrite("\x02\0\x10\0data",1,8,f)==8&&write_u32(f,(uint32_t)(n*2));
    for(size_t i=0;i<n&&ok;i++){double x=(double)wave[i]*scale;if(x>0.999969)x=0.999969;if(x< -1.0)x=-1.0;
        int16_t s=(int16_t)lrint(x*32768.0);unsigned char b[2]={(unsigned char)s,(unsigned char)((uint16_t)s>>8)};ok=fwrite(b,1,2,f)==2;}
    if(fflush(f)!=0)ok=0;
    if(fclose(f)!=0)ok=0;
    return ok;
}
int ama10_tts_synthesize_wav16k(const AMA10TTSCodebook *c,const uint8_t *codes,
                                size_t code_count,const char *path){
    if(!c||!c->centers||c->width!=TTS_MELS||!c->count||!codes||!code_count||!path||
       code_count>SIZE_MAX/2||code_count*2>SIZE_MAX/TTS_MELS)return 0;
    size_t frames=code_count*2,bins=TTS_FFT/2+1;
    if(frames>SIZE_MAX/bins||frames*bins>SIZE_MAX/sizeof(float))return 0;
    float *mag=(float*)malloc(frames*bins*sizeof(float));float *phase=(float*)malloc(frames*bins*sizeof(float));
    size_t samples=(frames-1)*TTS_HOP+TTS_WINDOW;float *wave=(float*)malloc(samples*sizeof(float));
    if(!mag||!phase||!wave){free(mag);free(phase);free(wave);return 0;}
    const float max_mel=1127.01048f*log1pf(8000.0f/700.0f);
    uint64_t rng=UINT64_C(0x8f3f73b5cf1c9ade);
    for(size_t t=0;t<frames;t++){
        uint8_t code=codes[t/2];if(!code||code>c->count){free(mag);free(phase);free(wave);return 0;}
        const float *mels=c->centers+(size_t)(code-1)*TTS_MELS;
        for(size_t k=0;k<bins;k++){
            float hz=(float)k*(float)TTS_SAMPLE_RATE/(float)TTS_FFT;
            float pos=hz_to_mel(hz)/max_mel*(float)(TTS_MELS+1)-1.0f;
            if(pos<0.0f)pos=0.0f;
            if(pos>(float)(TTS_MELS-1))pos=(float)(TTS_MELS-1);
            size_t lo=(size_t)pos,hi=lo+1<TTS_MELS?lo+1:lo;float frac=pos-(float)lo;
            float value=mels[lo]*(1.0f-frac)+mels[hi]*frac;
            float power=expf(value*4.0f-6.0f);if(power<1.0e-9f)power=1.0e-9f;if(power>1.0f)power=1.0f;
            mag[t*bins+k]=sqrtf(power)*(float)TTS_FFT;
            rng^=rng>>12;rng^=rng<<25;rng^=rng>>27;uint64_t q=rng*UINT64_C(2685821657736338717);
            phase[t*bins+k]=(float)(q>>40)*(6.2831853071795864769f/16777216.0f);
        }
    }
    int ok=render_spectrum(mag,frames,phase,wave);
    /* Griffin-Lim phase projection improves consistency with overlapping windows. */
    for(unsigned iteration=0;iteration<12&&ok;iteration++){
        float window[TTS_WINDOW];make_window(window);
        float *normal_wave=(float*)malloc(samples*sizeof(float));if(!normal_wave){ok=0;break;}
        memcpy(normal_wave,wave,samples*sizeof(float));
        for(size_t t=0;t<frames;t++){
            float real[TTS_FFT]={0},imag[TTS_FFT]={0};
            for(size_t j=0;j<TTS_WINDOW;j++)real[j]=normal_wave[t*TTS_HOP+j]*window[j];
            fft(real,imag,0);
            for(size_t k=0;k<bins;k++){size_t ix=t*bins+k;phase[ix]=atan2f(imag[k],real[k]);}
        }
        free(normal_wave);ok=render_spectrum(mag,frames,phase,wave);
    }
    if(ok)ok=write_wav(path,wave,samples);
    free(mag);free(phase);free(wave);return ok;
}
