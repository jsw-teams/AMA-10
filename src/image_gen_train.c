#include "ama10.h"
#include "ama10_imggen.h"
#include "checkpoint.h"
#include "tokenizer.h"
#ifdef AMA10_ENABLE_OPENCL
#include "opencl_backend.h"
#endif
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define IMAGE_CONTEXT_CODES 16u
static double wall_seconds(void){struct timespec t;if(timespec_get(&t,TIME_UTC)!=TIME_UTC)return 0.0;return (double)t.tv_sec+(double)t.tv_nsec*1.0e-9;}
static double cpu_seconds(void){return (double)clock()/(double)CLOCKS_PER_SEC;}

typedef struct{char *path,*caption,lang[16];int valid;AMA10Image image;uint64_t image_hash;uint32_t *codes;size_t ncode;uint32_t *layout_codes;size_t layout_count,layout_width,layout_height;uint32_t context_codes[5];} Row;
static char *dup(const char*s){size_t n=strlen(s);char*p=(char*)malloc(n+1);if(p)memcpy(p,s,n+1);return p;}
static void rows_free(Row*r,size_t n){for(size_t i=0;i<n;i++){free(r[i].path);free(r[i].caption);free(r[i].codes);free(r[i].layout_codes);ama10_image_free(&r[i].image);}free(r);}
static int read_rows(const char*manifest,Row**out,size_t*count,size_t*nt,size_t*nv){FILE*f=fopen(manifest,"rb");if(!f)return 0;char line[16384];Row*r=NULL;size_t n=0,cap=0,tr=0,va=0;int ok=1;
 while(fgets(line,sizeof(line),f)){size_t z=strlen(line);if(z&&line[z-1]=='\n')line[--z]=0;if(z&&line[z-1]=='\r')line[--z]=0;if(!z)continue;
  char*a=strchr(line,'\t');if(!a){ok=0;break;}*a++=0;char*b=strchr(a,'\t');if(!b){ok=0;break;}*b++=0;char*c=strchr(b,'\t');if(c){*c++=0;if(strchr(c,'\t')){ok=0;break;}}else c="en";
  if(!*line||!*a||(strcmp(b,"train")&&strcmp(b,"validation"))||strlen(c)>=16){ok=0;break;}
  if(n==cap){size_t nc=cap?cap*2:16;Row*q=(Row*)realloc(r,nc*sizeof(*q));if(!q){ok=0;break;}r=q;cap=nc;}
  Row x={0};x.path=dup(line);x.caption=dup(a);x.valid=!strcmp(b,"validation");memcpy(x.lang,c,strlen(c)+1);if(!x.path||!x.caption){free(x.path);free(x.caption);ok=0;break;}r[n++]=x;if(x.valid)va++;else tr++;
 }
 if(ferror(f))ok=0;
 fclose(f);
 if(!ok||!tr||!va){rows_free(r,n);return 0;}
 *out=r;*count=n;*nt=tr;*nv=va;return 1;}
static int join(const char*m,const char*r,char**out){const char*a=strrchr(m,'/'),*b=strrchr(m,'\\'),*s=a;if(b&&(!s||b>s))s=b;size_t d=s?(size_t)(s-m+1):0,n=strlen(r);char*p=(char*)malloc(d+n+1);if(!p)return 0;if(d)memcpy(p,m,d);memcpy(p+d,r,n+1);*out=p;return 1;}
static uint64_t hbytes(uint64_t h,const void*p,size_t n){const unsigned char*b=(const unsigned char*)p;for(size_t i=0;i<n;i++){h^=b[i];h*=UINT64_C(1099511628211);}return h;}
static uint64_t split_hash(Row*r,size_t n,int val){uint64_t h=UINT64_C(14695981039346656037);static const char mark[]="AMA10_TEXT_TO_IMAGE_HIERARCHICAL_CONTEXTVQ16_COARSE_LAYOUT_RGB_RVQ2_V24";h=hbytes(h,mark,sizeof(mark)-1);
 for(size_t i=0;i<n;i++)if(r[i].valid==val){h=hbytes(h,r[i].path,strlen(r[i].path));h=hbytes(h,r[i].caption,strlen(r[i].caption));h=hbytes(h,r[i].lang,strlen(r[i].lang));if(r[i].image.rgb){h=hbytes(h,&r[i].image.width,sizeof(r[i].image.width));h=hbytes(h,&r[i].image.height,sizeof(r[i].image.height));h=hbytes(h,r[i].image.rgb,r[i].image.width*r[i].image.height*3);}}
 return h;}
static int tokenize_rows(Row*r,size_t n,AMA10Tokenizer*t,int learn,size_t code_count,size_t context_count,size_t*bytes){size_t total=0;for(size_t i=0;i<n;i++)if(!r[i].valid){size_t z=strlen(r[i].caption)+strlen(r[i].lang)+10;if(total>SIZE_MAX-z-1)return 0;total+=z+1;}
 unsigned char*corpus=(unsigned char*)malloc(total);if(!corpus)return 0;size_t at=0;for(size_t i=0;i<n;i++)if(!r[i].valid){int z=snprintf((char*)corpus+at,total-at,"<lang:%s> %s\n",r[i].lang,r[i].caption);if(z<0){free(corpus);return 0;}at+=(size_t)z;}
 int ok=1;if(learn){ok=ama10_tokenizer_train(t,corpus,total,256);for(size_t c=0;c<code_count&&ok;c++){char marker[40];int z=snprintf(marker,sizeof(marker),"<|image_code_%03zu|>",c);ok=z>0&&(size_t)z<sizeof(marker)&&ama10_tokenizer_add_special(t,(const unsigned char*)marker,(size_t)z,NULL);}for(size_t c=0;c<context_count&&ok;c++){char marker[40];int z=snprintf(marker,sizeof(marker),"<|image_context_code_%03zu|>",c);ok=z>0&&(size_t)z<sizeof(marker)&&ama10_tokenizer_add_special(t,(const unsigned char*)marker,(size_t)z,NULL);}for(size_t v=0;v<6&&ok;v++){char marker[40];int z=snprintf(marker,sizeof(marker),"<|image_global_view_%zu|>",v);ok=z>0&&(size_t)z<sizeof(marker)&&ama10_tokenizer_add_special(t,(const unsigned char*)marker,(size_t)z,NULL);}static const unsigned char eos[]="<|image_eos|>";if(ok)ok=ama10_tokenizer_add_special(t,eos,sizeof(eos)-1,NULL);}free(corpus);if(!ok||t->vocabulary_size!=263+code_count+context_count)return 0;*bytes=total;return 1;}
static int prompt_seq(const Row*r,const AMA10Tokenizer*t,uint32_t**out,size_t*n){size_t len=strlen(r->lang)+strlen(r->caption)+12;unsigned char*b=(unsigned char*)malloc(len);if(!b)return 0;int z=snprintf((char*)b,len,"<lang:%s> %s\n",r->lang,r->caption);if(z<0){free(b);return 0;}size_t sz=(size_t)z;b[sz++]=0;int ok=ama10_tokenizer_encode(t,b,sz,out,n);free(b);return ok;}
static int add_position_features(float*out,size_t width,double x,double y){
 if(!out||width<4||!isfinite(x)||!isfinite(y))return 0;
 memset(out,0,width*sizeof(float));
 size_t axis_width=width/2;
 for(size_t axis=0;axis<2;axis++){
  size_t base=axis*axis_width,slots=axis?width-base:axis_width,pairs=slots/2;
  double position=axis?y:x;
  for(size_t i=0;i<pairs;i++){
   double cycles=pairs==1?1.0:0.5+3.5*(double)i/(double)(pairs-1);
   double phase=6.2831853071795864769*cycles*position;
   out[base+2*i]=(float)sin(phase);out[base+2*i+1]=(float)cos(phase);
  }
 }
 return 1;
}
static uint32_t global_token(size_t view,size_t code_count,size_t context_count){return (uint32_t)(256+code_count+context_count+view);}
typedef struct{uint32_t*token_inputs,*targets,*class_starts,*class_counts;float*features,*feature_gradients;unsigned char*feature_mask,*loss_mask;size_t length,prefix_length;} ImageSequence;
typedef struct{size_t row,tile;} PatchSample;
static void sequence_free(ImageSequence*s){if(!s)return;free(s->token_inputs);free(s->targets);free(s->class_starts);free(s->class_counts);free(s->features);free(s->feature_gradients);free(s->feature_mask);free(s->loss_mask);memset(s,0,sizeof(*s));}
static int sequence(const Row*r,size_t tile,int context_only,const AMA10Tokenizer*t,const AMA10ImageCodebook*b,size_t input_width,ImageSequence*out){
 int all_tiles=context_only==2;if(all_tiles){context_only=0;tile=r->ncode-1;}
 uint32_t*p=NULL;size_t pn=0;if(!out||(!context_only&&tile>=r->ncode)||!r->layout_codes||!r->layout_count||!prompt_seq(r,t,&p,&pn))return 0;
 size_t dx=r->image.width>b->patch_width?r->image.width-b->patch_width:0,dy=r->image.height>b->patch_height?r->image.height-b->patch_height:0;
 size_t gw=dx/b->stride_width+(dx%b->stride_width!=0)+1,gh=dy/b->stride_height+(dy%b->stride_height!=0)+1;
 if(!gw||!gh||gw>SIZE_MAX/gh||(!context_only&&gw*gh!=r->ncode)){free(p);return 0;}
 if(r->layout_count>(SIZE_MAX-1)/2){free(p);return 0;}size_t layout_extra=1+2*r->layout_count;
 size_t raster_stride=(size_t)b->levels+1;
 size_t local_extra=0;if(!context_only){if(tile>(SIZE_MAX-10-(size_t)b->levels)/raster_stride){free(p);return 0;}local_extra=10+tile*raster_stride+(size_t)b->levels;}
 size_t extra=context_only?10:local_extra;if(extra>SIZE_MAX-layout_extra){free(p);return 0;}extra+=layout_extra;
 if(pn>SIZE_MAX-extra){free(p);return 0;}size_t length=pn+extra;
 if(length>SIZE_MAX/input_width||length*input_width>SIZE_MAX/sizeof(float)||length>SIZE_MAX/sizeof(uint32_t)){free(p);return 0;}
 ImageSequence s={0};s.length=length;s.token_inputs=(uint32_t*)calloc(length,sizeof(uint32_t));s.targets=(uint32_t*)calloc(length,sizeof(uint32_t));s.class_starts=(uint32_t*)calloc(length,sizeof(uint32_t));s.class_counts=(uint32_t*)malloc(length*sizeof(uint32_t));s.features=(float*)calloc(length*input_width,sizeof(float));s.feature_gradients=(float*)calloc(length*input_width,sizeof(float));s.feature_mask=(unsigned char*)calloc(length,1);s.loss_mask=(unsigned char*)calloc(length,1);
 if(!s.token_inputs||!s.targets||!s.class_starts||!s.class_counts||!s.features||!s.feature_gradients||!s.feature_mask||!s.loss_mask){free(p);sequence_free(&s);return 0;}
 for(size_t i=0;i<length;i++)s.class_counts[i]=(uint32_t)t->vocabulary_size;
 /* Captions condition visual predictions; image-generation training should not
    spend its loss budget on ordinary next-token language modeling. */
 for(size_t i=0;i<pn;i++)s.token_inputs[i]=p[i];
 s.targets[pn-1]=global_token(0,b->code_count,b->context_code_count);free(p);
 size_t at=pn;for(size_t v=0;v<5;v++){
  uint32_t target=256+b->code_count+r->context_codes[v]-1;s.token_inputs[at]=global_token(v,b->code_count,b->context_code_count);s.targets[at]=target;s.class_starts[at]=256+b->code_count;s.class_counts[at]=b->context_code_count;s.loss_mask[at]=(unsigned char)(context_only!=0);
  s.token_inputs[at+1]=target;s.targets[at+1]=global_token(v+1,b->code_count,b->context_code_count);at+=2;
 }
 s.token_inputs[at++]=global_token(5,b->code_count,b->context_code_count);
 for(size_t k=0;k<r->layout_count;k++){
  size_t pos=at,x=k%r->layout_width,y=k/r->layout_width;
  double nx=r->layout_width<2?0.5:(double)x/(double)(r->layout_width-1);
  double ny=r->layout_height<2?0.5:(double)y/(double)(r->layout_height-1);
  if(!add_position_features(s.features+pos*input_width,input_width,nx,ny)){sequence_free(&s);return 0;}
  s.feature_mask[pos]=1;s.loss_mask[pos]=(unsigned char)(context_only!=0);
  s.class_starts[pos]=256;s.class_counts[pos]=b->code_count;s.targets[pos]=256+r->layout_codes[k]-1;
  s.token_inputs[pos+1]=s.targets[pos];at+=2;
 }
 s.prefix_length=at;
 if(context_only){s.prefix_length=s.length;*out=s;return 1;}
 for(size_t q=0;q<tile;q++){
  size_t pos=at+raster_stride*q,x=q%gw,y=q/gw;double nx=gw<2?0.5:(double)x/(double)(gw-1),ny=gh<2?0.5:(double)y/(double)(gh-1);
  uint32_t base_target=256+r->codes[q*b->levels]-1;
  s.feature_mask[pos]=1;s.targets[pos]=base_target;
  if(all_tiles){s.loss_mask[pos]=1;s.class_starts[pos]=256;s.class_counts[pos]=b->code_count;}
  if(!add_position_features(s.features+pos*input_width,input_width,nx,ny)){sequence_free(&s);return 0;}
  s.token_inputs[pos+1]=base_target;
  if(b->levels==2){uint32_t residual_target=256+r->codes[q*b->levels+1]-1;s.targets[pos+1]=residual_target;s.token_inputs[pos+2]=residual_target;if(all_tiles){s.loss_mask[pos+1]=1;s.class_starts[pos+1]=256;s.class_counts[pos+1]=b->code_count;}}
 }
 size_t pos=at+raster_stride*tile,x=tile%gw,y=tile/gw;double nx=gw<2?0.5:(double)x/(double)(gw-1),ny=gh<2?0.5:(double)y/(double)(gh-1);
 s.feature_mask[pos]=1;s.loss_mask[pos]=1;s.class_starts[pos]=256;s.class_counts[pos]=b->code_count;s.targets[pos]=256+r->codes[tile*b->levels]-1;
 if(!add_position_features(s.features+pos*input_width,input_width,nx,ny)){sequence_free(&s);return 0;}
 if(b->levels==2){s.token_inputs[pos+1]=s.targets[pos];s.loss_mask[pos+1]=1;s.class_starts[pos+1]=256;s.class_counts[pos+1]=b->code_count;s.targets[pos+1]=256+r->codes[tile*b->levels+1]-1;}
 *out=s;return 1;
}
static int train_ranges(AMA10Model*m,AMA10Optimizer*o,AMA10State*state,const ImageSequence*s,
                        unsigned reason_depth,AMA10Gradients*g,void*backend,float*loss){
#ifdef AMA10_ENABLE_OPENCL
 if(backend)return ama10_train_multimodal_accumulate_ranges_opencl(m,o,state,s->token_inputs,
     s->features,s->feature_mask,s->loss_mask,s->targets,s->class_starts,s->class_counts,
     s->length,reason_depth,g,s->feature_gradients,(AMA10OpenCL*)backend,loss);
#else
 (void)backend;
#endif
 return ama10_train_multimodal_accumulate_ranges(m,o,state,s->token_inputs,s->features,
     s->feature_mask,s->loss_mask,s->targets,s->class_starts,s->class_counts,s->length,
     reason_depth,g,s->feature_gradients,loss);
}
static double ce_range(const float*z,size_t start,size_t n,uint32_t y);
static int make_samples(const Row*r,size_t n,PatchSample**out,size_t*count){if(!r||!out||*out||!count)return 0;size_t total=0;for(size_t i=0;i<n;i++)if(!r[i].valid){if(r[i].ncode>SIZE_MAX-total)return 0;total+=r[i].ncode;}if(!total||total>SIZE_MAX/sizeof(PatchSample))return 0;PatchSample*s=(PatchSample*)malloc(total*sizeof(*s));if(!s)return 0;size_t at=0;for(size_t i=0;i<n;i++)if(!r[i].valid)for(size_t tile=0;tile<r[i].ncode;tile++)s[at++]=(PatchSample){i,tile};*out=s;*count=total;return at==total;}
static int prep(Row*r,size_t n,const char*manifest,AMA10Tokenizer*t,int learn,size_t code_count,size_t context_count,size_t*bytes){if(!tokenize_rows(r,n,t,learn,code_count,context_count,bytes))return 0;for(size_t i=0;i<n;i++){char*p=NULL;if(!join(manifest,r[i].path,&p))return 0;int ok=ama10_image_load(&r[i].image,p);free(p);if(!ok)return 0;r[i].image_hash=UINT64_C(14695981039346656037);r[i].image_hash=hbytes(r[i].image_hash,&r[i].image.width,sizeof(r[i].image.width));r[i].image_hash=hbytes(r[i].image_hash,&r[i].image.height,sizeof(r[i].image.height));r[i].image_hash=hbytes(r[i].image_hash,r[i].image.rgb,r[i].image.width*r[i].image.height*3);}return 1;}
static int first_training_image(const Row*r,size_t index){for(size_t i=0;i<index;i++)if(!r[i].valid&&r[i].image_hash==r[index].image_hash&&r[i].image.width==r[index].image.width&&r[i].image.height==r[index].image.height&&memcmp(r[i].image.rgb,r[index].image.rgb,r[i].image.width*r[i].image.height*3)==0)return 0;return 1;}
static int encode_all(Row*r,size_t n,const AMA10ImageCodebook*b){
 if(!b||(b->levels!=1&&b->levels!=2)||!b->centers)return 0;
 const size_t factor=4,code_width=b->latent_width?b->latent_width:b->feature_width;
 float*patches=NULL,*mean=(float*)malloc(code_width*sizeof(float));if(!mean)return 0;
 for(size_t i=0;i<n;i++){
  size_t code_tokens=0;if(!ama10_image_encode_codes(&r[i].image,b,&r[i].codes,&code_tokens)||!code_tokens||code_tokens%b->levels){free(mean);return 0;}r[i].ncode=code_tokens/b->levels;
  size_t dx=r[i].image.width>b->patch_width?r[i].image.width-b->patch_width:0;
  size_t dy=r[i].image.height>b->patch_height?r[i].image.height-b->patch_height:0;
  size_t gw=dx/b->stride_width+(dx%b->stride_width!=0)+1,gh=dy/b->stride_height+(dy%b->stride_height!=0)+1;
  if(!gw||!gh||gw>SIZE_MAX/gh||r[i].ncode!=gw*gh){free(mean);return 0;}
  r[i].layout_width=gw/factor+(gw%factor!=0);r[i].layout_height=gh/factor+(gh%factor!=0);
  if(!r[i].layout_width||r[i].layout_height>SIZE_MAX/r[i].layout_width){free(mean);return 0;}
  r[i].layout_count=r[i].layout_width*r[i].layout_height;
  r[i].layout_codes=(uint32_t*)malloc(r[i].layout_count*sizeof(uint32_t));if(!r[i].layout_codes){free(mean);return 0;}
  for(size_t cy=0;cy<r[i].layout_height;cy++)for(size_t cx=0;cx<r[i].layout_width;cx++){
   memset(mean,0,code_width*sizeof(float));size_t samples=0;
   for(size_t y=cy*factor;y<gh&&y<(cy+1)*factor;y++)for(size_t x=cx*factor;x<gw&&x<(cx+1)*factor;x++){
    uint32_t code=r[i].codes[(y*gw+x)*b->levels];if(!code||code>b->code_count){free(mean);return 0;}
    const float*center=b->centers+(size_t)(code-1)*code_width;for(size_t j=0;j<code_width;j++)mean[j]+=center[j];samples++;
   }
   if(!samples){free(mean);return 0;}for(size_t j=0;j<code_width;j++)mean[j]/=(float)samples;
   double best=INFINITY;uint32_t selected=0;
   for(uint32_t c=0;c<b->code_count;c++){const float*center=b->centers+(size_t)c*code_width;double distance=0.0;
    for(size_t j=0;j<code_width;j++){double delta=(double)mean[j]-center[j];distance+=delta*delta;}
    if(distance<best){best=distance;selected=c+1;}}
   r[i].layout_codes[cy*r[i].layout_width+cx]=selected;
  }
  size_t count=0,fine=0,feature_width=0;patches=NULL;
  if(!ama10_image_patchify_multiscale(&r[i].image,b->patch_width,b->patch_height,b->stride_width,b->stride_height,&patches,&count,&feature_width,&fine)||count!=fine+5||feature_width!=b->feature_width){free(patches);free(mean);return 0;}
  int ok=1;for(size_t v=0;v<5&&ok;v++)ok=ama10_image_context_quantize(b,patches+(fine+v)*feature_width,&r[i].context_codes[v])&&r[i].context_codes[v]<=b->context_code_count;
  free(patches);if(!ok){free(mean);return 0;}
 }
 free(mean);return 1;
}
static double ce_range(const float*z,size_t start,size_t n,uint32_t y){float mx=z[start];for(size_t i=1;i<n;i++)if(z[start+i]>mx)mx=z[start+i];double sum=0;for(size_t i=0;i<n;i++)sum+=exp((double)z[start+i]-mx);return (double)mx+log(sum)-z[start+y];}
static int eval_sequence_logits(const AMA10Model*m,AMA10State*state,const ImageSequence*s,unsigned reason_depth,void*backend,float*logits){
#ifdef AMA10_ENABLE_OPENCL
 if(backend)return ama10_opencl_eval_logits((AMA10OpenCL*)backend,m,s->token_inputs,s->features,
     s->feature_mask,s->length,reason_depth,state->fast,state->slow,logits,state->fast,state->slow);
#else
 (void)backend;
#endif
 for(size_t j=0;j<s->length;j++){
  float*out=logits+j*m->vocabulary;
  if(s->feature_mask[j]){if(!ama10_step_features(m,state,s->features+j*m->input_width,out,reason_depth,2.0f))return 0;}
  else if(!ama10_step(m,state,s->token_inputs[j],out,reason_depth,2.0f))return 0;
 }
 return 1;
}
static int select_eval_backend(const AMA10Model*m,AMA10Tokenizer*t,const AMA10ImageCodebook*b,
                               Row*r,size_t n,unsigned reason_depth,void*gpu_backend,void**selected){
 if(!selected)return 0;
 *selected=gpu_backend;if(!gpu_backend)return 1;
 size_t sample=0;while(sample<n&&!r[sample].valid)sample++;if(sample==n)return 0;
 /* Use an equal-length prefix for CPU/GPU parity and backend timing. Full
    raster evaluation is still performed on the selected backend below; timing
    a long CPU sequence here made automatic width selection prohibitively slow. */
 ImageSequence s={0};if(!sequence(&r[sample],0,2,t,b,m->input_width,&s)||s.length>SIZE_MAX/m->vocabulary){sequence_free(&s);return 0;}
 ImageSequence probe=s;const size_t benchmark_tokens=1024u;if(probe.length>benchmark_tokens)probe.length=benchmark_tokens;
 size_t logits_count=probe.length*m->vocabulary;if(logits_count>SIZE_MAX/sizeof(float)){sequence_free(&s);return 0;}
 float*cpu_logits=(float*)malloc(logits_count*sizeof(float)),*gpu_logits=(float*)malloc(logits_count*sizeof(float));
 AMA10State cpu={0},gpu={0};int ok=cpu_logits&&gpu_logits&&ama10_state_init(&cpu,m->width)&&ama10_state_init(&gpu,m->width);
 if(!ok){free(cpu_logits);free(gpu_logits);ama10_state_free(&cpu);ama10_state_free(&gpu);sequence_free(&s);return 0;}
 double start=wall_seconds();ok=eval_sequence_logits(m,&cpu,&probe,reason_depth,NULL,cpu_logits);double cpu_seconds=wall_seconds()-start;
 start=wall_seconds();if(ok)ok=eval_sequence_logits(m,&gpu,&probe,reason_depth,gpu_backend,gpu_logits);double gpu_seconds=wall_seconds()-start;
 float maximum_error=0.0f;if(ok){for(size_t i=0;i<logits_count;i++){float error=fabsf(cpu_logits[i]-gpu_logits[i]);if(error>maximum_error)maximum_error=error;}
  for(size_t i=0;i<m->width;i++){float fast_error=fabsf(cpu.fast[i]-gpu.fast[i]),slow_error=fabsf(cpu.slow[i]-gpu.slow[i]);if(fast_error>maximum_error)maximum_error=fast_error;if(slow_error>maximum_error)maximum_error=slow_error;}}
 if(ok&&maximum_error<=2.0e-3f&&cpu_seconds<gpu_seconds)*selected=NULL;
 if(ok)fprintf(stderr,"evaluation_backend_selection=benchmark_tokens=%zu cpu_seconds=%.6f opencl_seconds=%.6f parity_max_abs_error=%.7g selected=%s\n",probe.length,cpu_seconds,gpu_seconds,maximum_error,*selected?"opencl":"cpu");
 free(cpu_logits);free(gpu_logits);ama10_state_free(&cpu);ama10_state_free(&gpu);sequence_free(&s);return ok;
}
static int eval_all_images(const AMA10Model*m,AMA10Tokenizer*t,const AMA10ImageCodebook*b,Row*r,size_t n,int validation,unsigned reason_depth,void*backend,int full_vocabulary,double*loss,uint64_t*tokens){double sum=0;uint64_t count=0;AMA10State state={0};if(!ama10_state_init(&state,m->width))return 0;
 for(size_t i=0;i<n;i++)if((int)r[i].valid==validation){ImageSequence s={0};if(!sequence(&r[i],0,2,t,b,m->input_width,&s)||s.length>SIZE_MAX/m->vocabulary){sequence_free(&s);ama10_state_free(&state);return 0;}float*logits=(float*)malloc(s.length*m->vocabulary*sizeof(float));if(!logits){sequence_free(&s);ama10_state_free(&state);return 0;}ama10_state_reset(&state);int ok=eval_sequence_logits(m,&state,&s,reason_depth,backend,logits);if(ok)for(size_t j=s.prefix_length;j<s.length;j++)if(s.loss_mask[j]){size_t begin=full_vocabulary?0:256,classes=full_vocabulary?m->vocabulary:b->code_count;uint32_t target=full_vocabulary?s.targets[j]:s.targets[j]-256;sum+=ce_range(logits+j*m->vocabulary,begin,classes,target);count++;}free(logits);sequence_free(&s);if(!ok){ama10_state_free(&state);return 0;}}
 ama10_state_free(&state);if(!count)return 0;*loss=sum/(double)count;if(tokens)*tokens=count;return isfinite(*loss);}
static int eval(const AMA10Model*m,AMA10Tokenizer*t,AMA10ImageCodebook*b,Row*r,size_t n,unsigned reason_depth,void*backend,double*loss,uint64_t*tokens){return eval_all_images(m,t,b,r,n,1,reason_depth,backend,1,loss,tokens);}
static int eval_visual(const AMA10Model*m,AMA10Tokenizer*t,const AMA10ImageCodebook*b,AMA10State*state,Row*r,size_t n,int validation,unsigned reason_depth,void*backend,double*loss){(void)state;uint64_t count=0;return eval_all_images(m,t,(AMA10ImageCodebook*)b,r,n,validation,reason_depth,backend,0,loss,&count);}
static int eval_context(const AMA10Model*m,AMA10Tokenizer*t,const AMA10ImageCodebook*b,AMA10State*state,Row*r,size_t n,int validation,unsigned reason_depth,void*backend,double*context_loss,double*layout_loss){double context_sum=0,layout_sum=0;uint64_t context_count=0,layout_count=0;size_t base=256+b->code_count;
 for(size_t i=0;i<n;i++)if((int)r[i].valid==validation){ImageSequence s={0};if(!sequence(&r[i],0,1,t,b,m->input_width,&s)||s.length>SIZE_MAX/m->vocabulary){sequence_free(&s);return 0;}float*logits=(float*)malloc(s.length*m->vocabulary*sizeof(float));if(!logits){sequence_free(&s);return 0;}ama10_state_reset(state);int ok=eval_sequence_logits(m,state,&s,reason_depth,backend,logits);if(ok)for(size_t j=0;j<s.length;j++)if(s.loss_mask[j]){if(s.class_starts[j]==256){layout_sum+=ce_range(logits+j*m->vocabulary,256,s.class_counts[j],s.targets[j]-256);layout_count++;}else{context_sum+=ce_range(logits+j*m->vocabulary,s.class_starts[j],s.class_counts[j],s.targets[j]-(uint32_t)base);context_count++;}}free(logits);sequence_free(&s);if(!ok)return 0;}
 if(!context_count||!layout_count)return 0;
 *context_loss=context_sum/(double)context_count;*layout_loss=layout_sum/(double)layout_count;return isfinite(*context_loss)&&isfinite(*layout_loss);}
static char*suffix(const char*p,const char*s){size_t a=strlen(p),b=strlen(s);char*q=(char*)malloc(a+b+1);if(q){memcpy(q,p,a);memcpy(q+a,s,b+1);}return q;}
static uint64_t next_random(uint64_t*x){*x^=*x>>12;*x^=*x<<25;*x^=*x>>27;return *x*UINT64_C(2685821657736338717);}
static void shuffle(size_t*a,size_t n,uint64_t seed,uint64_t epoch){uint64_t x=seed^(epoch*UINT64_C(0x9e3779b97f4a7c15));if(!x)x=1;for(size_t i=n;i>1;i--){size_t j=(size_t)(next_random(&x)%i);size_t t=a[i-1];a[i-1]=a[j];a[j]=t;}}
static int save_all(const char*p,AMA10Model*m,AMA10Tokenizer*t,AMA10Optimizer*o,AMA10State*s,AMA10CheckpointProgress*g,AMA10ImageCodebook*b){char*tp=suffix(p,".tok"),*vp=suffix(p,".vq");if(!tp||!vp){free(tp);free(vp);return 0;}int ok=ama10_checkpoint_save(p,m,t,o,s,g)&&ama10_tokenizer_save(t,tp)&&ama10_image_codebook_save(b,vp);free(tp);free(vp);return ok;}
int main(int argc,char**argv){
 if(argc<4){fprintf(stderr,"usage: %s <manifest.tsv> <updates-total> <checkpoint.ama> [codebook-size=64] [model-width=32|auto] [--model-width N|auto] [--gpu-memory-budget-mb N] [--opencl] [--resume] [--gradient-mode direct|pcgrad|minnorm] [--reason-depth N] [--shared-codebook path.vq]\n",argv[0]);return 2;}
 char*e=NULL;errno=0;unsigned long long requested=strtoull(argv[2],&e,10);
 if(errno||e==argv[2]||*e||!requested||requested>UINT64_MAX/4u)return 2;
 uint32_t code_count=64,reason_depth=2;size_t model_width=32,gpu_memory_budget_bytes=0;const char*shared_codebook_path=NULL;int resume=0,use_opencl=0,automatic_width=0,have_code_count=0,have_model_width=0,gradient_method=0,have_gradient_mode=0,have_reason_depth=0,have_gpu_budget=0;
 for(int arg=4;arg<argc;arg++){
  if(!strcmp(argv[arg],"--resume")){if(resume)return 2;resume=1;continue;}
  if(!strcmp(argv[arg],"--opencl")){if(use_opencl)return 2;use_opencl=1;continue;}
  if(!strcmp(argv[arg],"--shared-codebook")){if(shared_codebook_path||arg+1>=argc||resume)return 2;shared_codebook_path=argv[++arg];continue;}
  if(!strcmp(argv[arg],"--model-width")){
   if(have_model_width||arg+1>=argc)return 2;
   const char*value=argv[++arg];have_model_width=1;
   if(!strcmp(value,"auto")){automatic_width=1;model_width=0;continue;}
   errno=0;unsigned long long parsed_width=strtoull(value,&e,10);if(errno||e==value||*e||!parsed_width||parsed_width>SIZE_MAX)return 2;model_width=(size_t)parsed_width;continue;
  }
  if(!strcmp(argv[arg],"--gpu-memory-budget-mb")){
   if(have_gpu_budget||arg+1>=argc)return 2;
   errno=0;unsigned long long mb=strtoull(argv[++arg],&e,10);size_t unit=(size_t)1024u*1024u;
   if(errno||e==argv[arg]||*e||!mb||mb>SIZE_MAX/unit)return 2;
   gpu_memory_budget_bytes=(size_t)mb*unit;have_gpu_budget=1;continue;
  }
  if(!strcmp(argv[arg],"--gradient-mode")){
   if(have_gradient_mode||arg+1>=argc)return 2;
   const char*mode=argv[++arg];
   if(!strcmp(mode,"pcgrad"))gradient_method=1;else if(!strcmp(mode,"minnorm"))gradient_method=2;else if(strcmp(mode,"direct"))return 2;
   have_gradient_mode=1;continue;
  }
  if(!strcmp(argv[arg],"--reason-depth")){
   if(have_reason_depth||arg+1>=argc)return 2;
   errno=0;unsigned long long parsed_depth=strtoull(argv[++arg],&e,10);
   if(errno||e==argv[arg]||*e||!parsed_depth||parsed_depth>UINT_MAX)return 2;
   reason_depth=(uint32_t)parsed_depth;have_reason_depth=1;continue;
  }
  errno=0;unsigned long long parsed=strtoull(argv[arg],&e,10);
  if(errno||e==argv[arg]||*e||parsed>SIZE_MAX)return 2;
  if(!have_code_count){if(parsed<2||parsed>254)return 2;code_count=(uint32_t)parsed;have_code_count=1;}
  else if(!have_model_width){if(parsed==0)return 2;model_width=(size_t)parsed;have_model_width=1;}else return 2;
 }
 Row*r=NULL;size_t n=0,nt=0,nv=0,bytes=0,sample_count=0;AMA10Tokenizer tok={0};AMA10Model m={0};AMA10Optimizer opt={0};AMA10State st={0};
 AMA10Gradients gradients={0},context_gradients={0},patch_gradients={0};AMA10CheckpointProgress pr={0};AMA10ImageCodebook book={0};size_t*train_order=NULL;PatchSample*samples=NULL;
 void*gpu_backend=NULL,*eval_backend=NULL;
 double setup_wall=0.0,setup_cpu=0.0,initial_eval_wall=0.0,initial_eval_cpu=0.0;
 double train_wall=0.0,train_cpu=0.0,final_eval_wall=0.0,final_eval_cpu=0.0;
 double checkpoint_wall=0.0,checkpoint_cpu=0.0;
 double phase_wall=wall_seconds(),phase_cpu=cpu_seconds();
 int ok=0;double gradient_cosine_sum=0.0,gradient_task_a_weight_sum=0.0;uint64_t gradient_observations=0,gradient_conflicts=0;const char*phase="manifest";if(!read_rows(argv[1],&r,&n,&nt,&nv))goto done;
 if(use_opencl&&!have_model_width)automatic_width=1;
 if(have_gpu_budget&&!automatic_width)return 2;
 if(automatic_width&&(!use_opencl||resume))return 2;
 phase="checkpoint";if(resume&&!ama10_checkpoint_load(argv[3],&m,&tok,&opt,&st,&pr))goto done;phase="images/tokenizer";if(!prep(r,n,argv[1],&tok,!resume,code_count,IMAGE_CONTEXT_CODES,&bytes))goto done;uint64_t codebook_hash=split_hash(r,n,0),th=codebook_hash,vh=split_hash(r,n,1);static const char model_tag[]="STATEFUL_RASTER_RGB_RVQ2_V3_ALL_IMAGE_TARGETS";th=hbytes(th,model_tag,sizeof(model_tag)-1);if(gradient_method==1){static const char tag[]="PCGRAD_GLOBAL_CONTEXT_STATEFUL_RASTER_V1";th=hbytes(th,tag,sizeof(tag)-1);}else if(gradient_method==2){static const char tag[]="MINNORM_GLOBAL_CONTEXT_STATEFUL_RASTER_V1";th=hbytes(th,tag,sizeof(tag)-1);}uint32_t target_vocab=(uint32_t)tok.vocabulary_size;
 phase="codebook";fprintf(stderr,"phase=codebook mode=%s\n",resume?"checkpoint":shared_codebook_path?"shared":"fit");fflush(stderr);char*vp=suffix(argv[3],".vq");if(!vp)goto done;
 if(resume){ok=ama10_image_codebook_load(&book,vp)&&book.train_hash==codebook_hash&&book.levels==2&&!book.latent_width&&book.code_count==code_count&&book.context_code_count==IMAGE_CONTEXT_CODES&&book.patch_width==16&&book.patch_height==16&&book.stride_width==12&&book.stride_height==12&&book.feature_width==768;}
 else if(shared_codebook_path){ok=ama10_image_codebook_load(&book,shared_codebook_path)&&book.train_hash==codebook_hash&&book.levels==2&&!book.latent_width&&book.code_count==code_count&&book.context_code_count==IMAGE_CONTEXT_CODES&&book.patch_width==16&&book.patch_height==16&&book.stride_width==12&&book.stride_height==12&&book.feature_width==768;}
 else{
  size_t patches=0;
  for(size_t i=0;i<n;i++)if(!r[i].valid&&first_training_image(r,i)){float*p=NULL;size_t np=0,d=0;if(!ama10_image_patchify_overlap(&r[i].image,16,16,12,12,&p,&np,&d)||d!=768){free(p);patches=0;break;}size_t take=np>256?256:np;if(patches>SIZE_MAX-take){free(p);patches=0;break;}patches+=take;free(p);}
  if(patches>=code_count&&patches<=SIZE_MAX/768&&patches*768<=SIZE_MAX/sizeof(float)){
   float*x=(float*)malloc(patches*768*sizeof(float));
   if(x){size_t q=0;for(size_t i=0;i<n;i++)if(!r[i].valid&&first_training_image(r,i)){float*p=NULL;size_t np=0,d=0;if(!ama10_image_patchify_overlap(&r[i].image,16,16,12,12,&p,&np,&d)||d!=768){free(p);free(x);x=NULL;break;}size_t take=np>256?256:np;if(take>patches-q){free(p);free(x);x=NULL;break;}for(size_t k=0;k<take;k++){size_t pick=k*np/take;memcpy(x+(q+k)*768,p+pick*d,d*sizeof(float));}q+=take;free(p);}
    if(x&&q==patches){
     ok=ama10_image_codebook_fit(x,patches,768,code_count,16,16,codebook_hash,&book);
     if(ok)ok=ama10_image_codebook_fit_residual(x,patches,&book);
     if(ok){book.stride_width=12;book.stride_height=12;}free(x);
    }else if(x)free(x);
   }
  }
 }
 if(ok&&!resume&&!shared_codebook_path){size_t contexts=0;for(size_t i=0;i<n;i++)if(!r[i].valid&&first_training_image(r,i)){if(contexts>SIZE_MAX-5){ok=0;break;}contexts+=5;}
  if(ok&&(contexts<IMAGE_CONTEXT_CODES||contexts>SIZE_MAX/book.feature_width||contexts*book.feature_width>SIZE_MAX/sizeof(float)))ok=0;
  float*cx=ok?(float*)malloc(contexts*book.feature_width*sizeof(float)):NULL;if(ok&&!cx)ok=0;
  if(ok){size_t at=0;for(size_t i=0;i<n&&ok;i++)if(!r[i].valid&&first_training_image(r,i)){float*all=NULL;size_t count=0,d=0,fine=0;if(!ama10_image_patchify_multiscale(&r[i].image,book.patch_width,book.patch_height,book.stride_width,book.stride_height,&all,&count,&d,&fine)||count!=fine+5||d!=book.feature_width)ok=0;else{memcpy(cx+at*d,all+fine*d,5*d*sizeof(float));at+=5;}free(all);}if(ok)ok=at==contexts&&ama10_image_codebook_fit_context(cx,contexts,IMAGE_CONTEXT_CODES,&book);}
  free(cx);
 }
 free(vp);if(!ok)goto done;ok=0;phase="image quantization";fprintf(stderr,"phase=image_quantization rows=%zu codebook=%u latent_width=%u levels=%u\n",n,book.code_count,book.latent_width,book.levels);fflush(stderr);if(!encode_all(r,n,&book)||!make_samples(r,n,&samples,&sample_count))goto done;
 size_t maxseq=0;for(size_t i=0;i<n;i++){uint32_t*p=NULL;size_t z=0;if(!prompt_seq(&r[i],&tok,&p,&z))goto done;free(p);
  if(r[i].layout_count>(SIZE_MAX-1)/2)goto done;
  if(r[i].layout_count>(SIZE_MAX-28)/2||z>SIZE_MAX-28-2*r[i].layout_count)goto done;
  size_t prefix=z+28+2*r[i].layout_count;
  if(r[i].ncode>(SIZE_MAX-prefix)/(size_t)(book.levels+1))goto done;
  size_t seq=prefix+r[i].ncode*(size_t)(book.levels+1);if(maxseq<seq)maxseq=seq;}if(maxseq>UINT32_MAX)goto done;
 setup_wall=wall_seconds()-phase_wall;setup_cpu=cpu_seconds()-phase_cpu;
#ifdef AMA10_ENABLE_OPENCL
 if(automatic_width){model_width=ama10_opencl_recommend_width(tok.vocabulary_size,maxseq,reason_depth,gpu_memory_budget_bytes,&gpu_memory_budget_bytes);if(!model_width)goto done;double parameters=4.0*(double)model_width*(double)model_width+2.0*(double)tok.vocabulary_size*(double)model_width+2.0*(double)model_width;printf("model_width_selection=auto width=%zu estimated_parameters=%.0f memory_budget_bytes=%zu max_sequence_length=%zu reason_depth=%u\n",model_width,parameters,gpu_memory_budget_bytes,maxseq,reason_depth);}
#else
 (void)gpu_memory_budget_bytes;
#endif
 phase="model";if(!resume){if(!ama10_model_init(&m,tok.vocabulary_size,model_width,model_width,UINT64_C(20261118))||!ama10_optimizer_init(&opt,&m)||!ama10_state_init(&st,model_width))goto done;}if(m.vocabulary!=tok.vocabulary_size||m.width!=model_width||(resume&&!opt.step))goto done;
#ifdef AMA10_ENABLE_OPENCL
 if(use_opencl){phase="OpenCL initialization";gpu_backend=ama10_opencl_create(&m,"kernels/ama10_recurrent.cl");if(!gpu_backend)goto done;}
#else
 if(use_opencl){phase="OpenCL build required";goto done;}
#endif
 eval_backend=gpu_backend;if(!select_eval_backend(&m,&tok,&book,r,n,reason_depth,gpu_backend,&eval_backend))goto done;
 if(resume){if(pr.train_hash!=th||pr.validation_hash!=vh||pr.tokenizer_sample_bytes!=bytes||pr.sequence_length!=maxseq||pr.target_vocabulary!=target_vocab||pr.depth!=reason_depth||pr.accumulation_steps!=2||pr.optimizer_algorithm!=opt.algorithm||pr.learning_rate!=0.0001f||pr.learning_rate_decay!=0.01f||pr.weight_decay!=0.01f||pr.cursor!=pr.completed_updates%nt||requested<=pr.completed_updates)goto done;}
 else{pr.train_hash=th;pr.validation_hash=vh;pr.tokenizer_sample_bytes=bytes;pr.sequence_length=(uint32_t)maxseq;pr.depth=reason_depth;pr.accumulation_steps=2;pr.target_vocabulary=target_vocab;pr.optimizer_algorithm=opt.algorithm;pr.learning_rate=0.0001f;pr.learning_rate_decay=0.01f;pr.weight_decay=0.01f;}
 if(!ama10_gradients_init(&gradients,&m)||!ama10_gradients_init(&context_gradients,&m)||
    !ama10_gradients_init(&patch_gradients,&m))goto done;
 double before=0,after=0,train_code_before=0,train_code_after=0,vcode_before=0,vcode_after=0,train_context_before=0,train_context_after=0,vcontext_before=0,vcontext_after=0,train_layout_before=0,train_layout_after=0,vlayout_before=0,vlayout_after=0;uint64_t valtokens=0;fprintf(stderr,"phase=initial_evaluation\n");fflush(stderr);phase_wall=wall_seconds();phase_cpu=cpu_seconds();if(!eval(&m,&tok,&book,r,n,reason_depth,eval_backend,&before,&valtokens)||!eval_visual(&m,&tok,&book,&st,r,n,0,reason_depth,eval_backend,&train_code_before)||!eval_visual(&m,&tok,&book,&st,r,n,1,reason_depth,eval_backend,&vcode_before)||!eval_context(&m,&tok,&book,&st,r,n,0,reason_depth,eval_backend,&train_context_before,&train_layout_before)||!eval_context(&m,&tok,&book,&st,r,n,1,reason_depth,eval_backend,&vcontext_before,&vlayout_before))goto done;initial_eval_wall=wall_seconds()-phase_wall;initial_eval_cpu=cpu_seconds()-phase_cpu;train_order=(size_t*)malloc(nt*sizeof(size_t));if(!train_order)goto done;for(size_t q=0,at=0;q<n;q++)if(!r[q].valid)train_order[at++]=q;
  fprintf(stderr,"phase=training start=%llu target=%llu backend=%s\n",(unsigned long long)pr.completed_updates,(unsigned long long)requested,gpu_backend?"opencl":"cpu");fflush(stderr);phase_wall=wall_seconds();phase_cpu=cpu_seconds();uint64_t active_epoch=UINT64_MAX,patch_target_exposures=0;for(uint64_t u=pr.completed_updates;u<(uint64_t)requested;u++){
  uint64_t image_epoch=u/nt;if(image_epoch!=active_epoch){shuffle(train_order,nt,codebook_hash,image_epoch);active_epoch=image_epoch;}
   ama10_gradients_reset(&context_gradients);ama10_gradients_reset(&patch_gradients);
   Row*row=&r[train_order[u%nt]];ImageSequence context_sequence={0};float context_loss=0,patch_loss=0;
   ama10_state_reset(&st);int trained=sequence(row,0,1,&tok,&book,m.input_width,&context_sequence)&&context_sequence.length<=UINT_MAX&&train_ranges(&m,&opt,&st,&context_sequence,reason_depth,&context_gradients,gpu_backend,&context_loss)&&isfinite(context_loss);sequence_free(&context_sequence);
   ImageSequence s={0};trained=trained&&sequence(row,0,2,&tok,&book,m.input_width,&s);if(trained){ama10_state_reset(&st);trained=s.length<=UINT_MAX&&train_ranges(&m,&opt,&st,&s,reason_depth,&patch_gradients,gpu_backend,&patch_loss)&&isfinite(patch_loss);if(trained){uint64_t targets=(uint64_t)row->ncode*(uint64_t)book.levels;if(targets>UINT64_MAX-patch_target_exposures)trained=0;else patch_target_exposures+=targets;}}sequence_free(&s);
   if(!trained)goto done;
  float lr=ama10_learning_rate(0.0001f,u+1,0.01f);
  double cosine=0.0,context_weight=0.0;int updated=0,gpu_task_update=0;
#ifdef AMA10_ENABLE_OPENCL
  if(gpu_backend&&gradient_method==2){gpu_task_update=1;updated=ama10_opencl_adamw_apply_tasks((AMA10OpenCL*)gpu_backend,&m,&opt,&context_gradients,&patch_gradients,lr,0.01f,&cosine,&context_weight);}
#endif
  if(!gpu_task_update){
   if(!ama10_gradients_combine_tasks_with_weights(&gradients,&context_gradients,&patch_gradients,gradient_method,&cosine,&context_weight))goto done;
#ifdef AMA10_ENABLE_OPENCL
   if(gpu_backend)updated=ama10_opencl_adamw_apply((AMA10OpenCL*)gpu_backend,&m,&opt,&gradients,lr,0.01f);
   else
#endif
   updated=ama10_optimizer_apply(&m,&opt,&gradients,lr,0.01f);
  }else if(updated)ama10_gradients_reset(&gradients);
  if(!updated)goto done;
  gradient_cosine_sum+=cosine;gradient_task_a_weight_sum+=context_weight;gradient_observations++;if(cosine<0.0)gradient_conflicts++;
  fprintf(stderr,"update=%llu/%llu context_loss=%.6f mean_patch_loss=%.6f task_cosine=%.6f\n",(unsigned long long)opt.step,(unsigned long long)requested,context_loss,patch_loss/4.0f,cosine);fflush(stderr);
  if((u+1)%100==0){pr.completed_updates=opt.step;pr.cursor=opt.step%nt;double checkpoint_start=wall_seconds(),checkpoint_cpu_start=cpu_seconds();
#ifdef AMA10_ENABLE_OPENCL
   if(gpu_backend&&!ama10_opencl_adamw_sync((AMA10OpenCL*)gpu_backend,&m,&opt))goto done;
#endif
   if(!save_all(argv[3],&m,&tok,&opt,&st,&pr,&book)){goto done;}checkpoint_wall+=wall_seconds()-checkpoint_start;checkpoint_cpu+=cpu_seconds()-checkpoint_cpu_start;}}
 train_wall=wall_seconds()-phase_wall;train_cpu=cpu_seconds()-phase_cpu;
 fprintf(stderr,"phase=final_evaluation\n");fflush(stderr);phase_wall=wall_seconds();phase_cpu=cpu_seconds();if(!eval(&m,&tok,&book,r,n,reason_depth,eval_backend,&after,NULL)||!eval_visual(&m,&tok,&book,&st,r,n,0,reason_depth,eval_backend,&train_code_after)||!eval_visual(&m,&tok,&book,&st,r,n,1,reason_depth,eval_backend,&vcode_after)||!eval_context(&m,&tok,&book,&st,r,n,0,reason_depth,eval_backend,&train_context_after,&train_layout_after)||!eval_context(&m,&tok,&book,&st,r,n,1,reason_depth,eval_backend,&vcontext_after,&vlayout_after))goto done;final_eval_wall=wall_seconds()-phase_wall;final_eval_cpu=cpu_seconds()-phase_cpu;
 pr.completed_updates=opt.step;pr.cursor=opt.step%nt;phase="save";
#ifdef AMA10_ENABLE_OPENCL
 if(gpu_backend&&!ama10_opencl_adamw_sync((AMA10OpenCL*)gpu_backend,&m,&opt))goto done;
#endif
 {double checkpoint_start=wall_seconds(),checkpoint_cpu_start=cpu_seconds();if(!save_all(argv[3],&m,&tok,&opt,&st,&pr,&book))goto done;checkpoint_wall+=wall_seconds()-checkpoint_start;checkpoint_cpu+=cpu_seconds()-checkpoint_cpu_start;}
 printf("image_pairs=%zu train=%zu validation=%zu train_patch_samples=%zu conditioning=caption+global_context+coarse_spatial_layout+stateful_raster_codes bptt=full_sequence reason_depth=%u raster_targets_per_update=all_image_levels class_range_ce=true gradient_mode=%s observed_conflict_fraction=%.6f mean_task_gradient_cosine=%.6f mean_context_task_weight=%.6f global_context_codes=5 coarse_layout_factor=4 visual_codes=%u patch_size=%ux%u stride=%ux%u variable_resolution=true updates=%llu patch_target_exposures_this_run=%llu validation_full_vocabulary_visual_nll_before=%.6f after=%.6f training_image_code_nll_before=%.6f after=%.6f validation_image_code_nll_before=%.6f after=%.6f training_context_code_nll_before=%.6f after=%.6f validation_context_code_nll_before=%.6f after=%.6f training_layout_code_nll_before=%.6f after=%.6f validation_layout_code_nll_before=%.6f after=%.6f checkpoint=%s\n",n,nt,nv,sample_count,reason_depth,gradient_method==1?"pcgrad":gradient_method==2?"minnorm":"direct",gradient_observations?(double)gradient_conflicts/(double)gradient_observations:0.0,gradient_observations?gradient_cosine_sum/(double)gradient_observations:0.0,gradient_observations?gradient_task_a_weight_sum/(double)gradient_observations:0.0,book.code_count,book.patch_width,book.patch_height,book.stride_width,book.stride_height,(unsigned long long)opt.step,(unsigned long long)patch_target_exposures,before,after,train_code_before,train_code_after,vcode_before,vcode_after,train_context_before,train_context_after,vcontext_before,vcontext_after,train_layout_before,train_layout_after,vlayout_before,vlayout_after,argv[3]);
 printf("phase_timing_setup_wall=%.3f setup_cpu=%.3f initial_eval_wall=%.3f initial_eval_cpu=%.3f train_wall=%.3f train_cpu=%.3f final_eval_wall=%.3f final_eval_cpu=%.3f checkpoint_wall=%.3f checkpoint_cpu=%.3f backend=%s",setup_wall,setup_cpu,initial_eval_wall,initial_eval_cpu,train_wall,train_cpu,final_eval_wall,final_eval_cpu,checkpoint_wall,checkpoint_cpu,gpu_backend?"opencl":"cpu");
#ifdef AMA10_ENABLE_OPENCL
 if(gpu_backend)printf(" device=%s",ama10_opencl_device_name((AMA10OpenCL*)gpu_backend));
#endif
 printf("\n");ok=1;
done:if(!ok)fprintf(stderr,"image generation training failed during %s\n",phase);
#ifdef AMA10_ENABLE_OPENCL
 ama10_opencl_destroy((AMA10OpenCL*)gpu_backend);
#endif
 ama10_gradients_free(&gradients);ama10_gradients_free(&context_gradients);ama10_gradients_free(&patch_gradients);free(samples);free(train_order);rows_free(r,n);ama10_image_codebook_free(&book);ama10_state_free(&st);ama10_optimizer_free(&opt);ama10_model_free(&m);ama10_tokenizer_free(&tok);return ok?0:1;}
