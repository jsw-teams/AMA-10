#include "ama10_q8.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char**argv){if(argc!=2)return 2;char path[1024];snprintf(path,sizeof(path),"%s/q8-roundtrip.q8",argv[1]);
 AMA10Model f32={0};AMA10Q8Model q8={0},loaded={0},corrupt={0};AMA10State fs={0},qs={0};float *a=NULL,*b=NULL;int ok=0;float max_error=0;
 if(!ama10_model_init(&f32,47,24,16,20261118)||!ama10_q8_model_from_f32(&f32,&q8)||!ama10_q8_model_save(&q8,path)||!ama10_q8_model_save(&q8,path)||
    !ama10_q8_model_load(&loaded,path)||!ama10_state_init(&fs,24)||!ama10_q8_state_init(&qs,24))goto done;
 size_t fbytes=(f32.width*f32.input_width+3*f32.width*f32.width+f32.input_width*f32.vocabulary+f32.vocabulary*f32.width+2*f32.width)*sizeof(float);
 if(!ama10_q8_model_memory_bytes(&loaded)||ama10_q8_model_memory_bytes(&loaded)>=fbytes)goto done;
 a=(float*)malloc(47*sizeof(float));b=(float*)malloc(47*sizeof(float));if(!a||!b)goto done;
 const uint32_t seq[]={3,8,2,15,46,0,29,1,9};double squared=0;
 for(size_t t=0;t<sizeof(seq)/sizeof(seq[0]);t++){if(!ama10_step(&f32,&fs,seq[t],a,2,2.0f)||!ama10_q8_step(&loaded,&qs,seq[t],b,2,2.0f))goto done;
   for(size_t i=0;i<47;i++){float d=fabsf(a[i]-b[i]);if(d>max_error)max_error=d;squared+=(double)d*d;}}
 if(max_error>0.03f||sqrt(squared/(9.0*47.0))>0.005)goto done;
 {FILE*f=fopen(path,"r+b");if(!f||fseek(f,-1,SEEK_END)!=0){if(f)fclose(f);goto done;}int byte=fgetc(f);int changed=byte!=EOF&&fseek(f,-1,SEEK_CUR)==0&&fputc(byte^1,f)!=EOF;if(fclose(f)!=0)changed=0;if(!changed)goto done;}
 if(ama10_q8_model_load(&corrupt,path))goto done;
 ok=1;
done:
 if(!ok)fprintf(stderr,"Q8 conversion, checked save/load, memory reduction, or recurrent parity failed (max logit error %.6g)\n",(double)max_error);
 free(a);free(b);ama10_state_free(&fs);ama10_state_free(&qs);ama10_q8_model_free(&corrupt);ama10_q8_model_free(&loaded);ama10_q8_model_free(&q8);ama10_model_free(&f32);(void)remove(path);
 if(ok)printf("Q8 per-row quantization, checksummed persistence, recurrent-logit parity, and memory reduction: ok (max logit error %.7g, rms %.7g)\n",(double)max_error,sqrt(squared/(9.0*47.0)));
 return ok?0:1;
}
