#include "ama10.h"
#include "ama10_tts.h"
#include "checkpoint.h"
#include "tokenizer.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_text(const char *path,unsigned char **out,size_t *length){
    FILE *f=fopen(path,"rb");if(!f)return 0;if(fseek(f,0,SEEK_END)!=0){fclose(f);return 0;}long n=ftell(f);
    if(n<0||n>1048576||fseek(f,0,SEEK_SET)!=0){fclose(f);return 0;}unsigned char *p=(unsigned char*)malloc((size_t)n+1);if(!p){fclose(f);return 0;}
    int ok=fread(p,1,(size_t)n,f)==(size_t)n;if(fclose(f)!=0)ok=0;if(!ok){free(p);return 0;}
    size_t len=(size_t)n;while(len&&(p[len-1]=='\n'||p[len-1]=='\r'||p[len-1]==' '||p[len-1]=='\t'))len--;
    if(!len){free(p);return 0;}p[len]='\0';for(size_t i=0;i<len;i++)if(p[i]>='a'&&p[i]<='z')p[i]=(unsigned char)(p[i]-'a'+'A');
    *out=p;*length=len;return 1;
}
static char *suffix(const char *path,const char *s){size_t a=strlen(path),b=strlen(s);if(a>SIZE_MAX-b-1)return NULL;char *p=(char*)malloc(a+b+1);if(p){memcpy(p,path,a);memcpy(p+a,s,b+1);}return p;}
int main(int argc,char **argv){
    if(argc<4||argc>6){fprintf(stderr,"usage: %s <checkpoint.ama> <text.txt> <output.wav> [language=en] [max-frames=1500]\n",argv[0]);return 2;}
    const char *language=argc>=5?argv[4]:"en";if(strlen(language)>8){fprintf(stderr,"language tag is too long\n");return 2;}
    char *end=NULL;unsigned long max_frames=1500;if(argc==6){errno=0;max_frames=strtoul(argv[5],&end,10);if(errno||end==argv[5]||*end||max_frames<1||max_frames>10000){fprintf(stderr,"max-frames must be 1..10000\n");return 2;}}
    AMA10Model model={0};AMA10Tokenizer tokenizer={0};AMA10Optimizer optimizer={0};AMA10State state={0};AMA10CheckpointProgress progress={0};AMA10TTSCodebook codebook={0};
    unsigned char *text=NULL,*prompt=NULL;uint32_t *inputs=NULL;float *logits=NULL;uint8_t *codes=NULL;size_t text_len=0,prompt_len=0,input_count=0,code_count=0;int ok=0;
    char *vqpath=suffix(argv[1],".vq");if(!vqpath)goto done;
    ok=ama10_checkpoint_load(argv[1],&model,&tokenizer,&optimizer,&state,&progress)&&
       ama10_tts_codebook_load(&codebook,vqpath)&&
       model.vocabulary==256&&tokenizer.vocabulary_size==256&&codebook.width==40;
    free(vqpath);if(!ok)goto done;
    ok=0;
    ok=read_text(argv[2],&text,&text_len);if(!ok)goto done;
    char lang[32];int langlen=snprintf(lang,sizeof(lang),"<lang:%s> ",language);
    if(langlen<0||(size_t)langlen>=sizeof(lang)||(size_t)langlen>SIZE_MAX-text_len-2)goto done;
    prompt_len=(size_t)langlen+text_len+2;prompt=(unsigned char*)malloc(prompt_len);if(!prompt)goto done;
    memcpy(prompt,lang,(size_t)langlen);memcpy(prompt+langlen,text,text_len);prompt[langlen+text_len]='\n';prompt[langlen+text_len+1]=0;
    if(!ama10_tokenizer_encode(&tokenizer,prompt,prompt_len,&inputs,&input_count)||!input_count)goto done;
    logits=(float*)malloc(model.vocabulary*sizeof(float));codes=(uint8_t*)malloc((size_t)max_frames);if(!logits||!codes)goto done;
    ama10_state_reset(&state);
    for(size_t i=0;i<input_count;i++)if(!ama10_step(&model,&state,inputs[i],logits,2,2.0f)){ok=0;goto done;}
    double estimated=(double)text_len*(double)codebook.frames_per_byte;
    size_t target_frames=estimated>(double)max_frames?(size_t)max_frames:(size_t)ceil(estimated);
    if(!target_frames)target_frames=1;
    size_t minimum=target_frames/2;
    const char *stop_reason=target_frames==(size_t)max_frames?"max_frames":"estimated_duration";
    for(size_t frame=0;frame<target_frames;frame++){
        uint32_t best=1;float best_logit=-INFINITY;
        for(uint32_t token=1;token<=codebook.count;token++)if(logits[token]>best_logit){best_logit=logits[token];best=token;}
        if(frame>=minimum&&logits[255]>best_logit){best=255;stop_reason="eos";break;}
        codes[code_count++]=(uint8_t)best;
        if(!ama10_step(&model,&state,best,logits,2,2.0f)){ok=0;goto done;}
    }
    ok=code_count>0&&ama10_tts_synthesize_wav16k(&codebook,codes,code_count,argv[3]);
    if(ok)printf("tts_frames=%zu seconds=%.3f codebook=%u stop=%s checkpoint_update=%llu output=%s\n",code_count,(double)code_count/50.0,codebook.count,stop_reason,(unsigned long long)progress.completed_updates,argv[3]);
done:
    if(!ok)fprintf(stderr,"TTS generation failed\n");
    free(codes);free(logits);free(inputs);free(prompt);free(text);ama10_tts_codebook_free(&codebook);ama10_state_free(&state);ama10_optimizer_free(&optimizer);ama10_model_free(&model);ama10_tokenizer_free(&tokenizer);return ok?0:1;
}
