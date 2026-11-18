#include "ama10_imggen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char**argv){if(argc!=2)return 2;size_t neighbors[8],neighbor_count=ama10_image_spatial_neighbors(12,5,5,neighbors);if(neighbor_count!=8)return 1;for(size_t i=0;i<neighbor_count;i++){if(neighbors[i]>=12)return 1;for(size_t j=0;j<i;j++)if(neighbors[i]==neighbors[j])return 1;}neighbor_count=ama10_image_spatial_neighbors(134,90,53,neighbors);if(neighbor_count!=7)return 1;for(size_t i=0;i<neighbor_count;i++)if(neighbors[i]>=134)return 1;if(ama10_image_spatial_neighbors(0,0,1,neighbors)!=0)return 1;float *patches=(float*)malloc(2*192*sizeof(float));if(!patches)return 1;
 for(size_t p=0;p<2;p++)for(size_t i=0;i<192;i++)patches[p*192+i]=(float)p;
 AMA10ImageCodebook b={0},loaded={0};int ok=ama10_image_codebook_fit(patches,2,192,2,8,8,1234,&b);if(ok)ok=ama10_image_codebook_fit_context(patches,2,2,&b);free(patches);if(!ok)return 1;
 char bookpath[1024],impath[1024];snprintf(bookpath,sizeof(bookpath),"%s/image-gen-test.vq",argv[1]);snprintf(impath,sizeof(impath),"%s/image-gen-test.ppm",argv[1]);
 uint32_t context_code=0;ok=ama10_image_codebook_save(&b,bookpath)&&ama10_image_codebook_load(&loaded,bookpath)&&loaded.train_hash==1234&&loaded.code_count==2&&loaded.context_code_count==2&&loaded.context_centers&&ama10_image_context_quantize(&loaded,loaded.context_centers,&context_code)&&context_code>=1&&context_code<=2;
 AMA10Image input={64,64,(unsigned char*)malloc(64*64*3)},decoded={0},readback={0};if(!input.rgb)ok=0;
 if(input.rgb)for(size_t y=0;y<64;y++)for(size_t x=0;x<64;x++){size_t p=(y*64+x)*3;unsigned char value=(unsigned char)(x<32?25:225);input.rgb[p]=value;input.rgb[p+1]=value;input.rgb[p+2]=value;}
 uint32_t *codes=NULL;size_t count=0;if(ok)ok=ama10_image_encode_codes(&input,&loaded,&codes,&count)&&count==64&&ama10_image_decode_codes(&loaded,codes,count,&decoded)&&ama10_image_save_ppm(&decoded,impath)&&ama10_image_load_ppm(&readback,impath)&&readback.width==64&&readback.height==64;
 if(ok){for(size_t y=0;y<64;y++)for(size_t x=0;x<64;x++){size_t p=(y*64+x)*3;unsigned char value=readback.rgb[p];if((x<32&&value>20)||(x>=32&&value<235)||readback.rgb[p+1]!=value||readback.rgb[p+2]!=value)ok=0;}}
 double psnr=0,ssim=0;if(ok)ok=ama10_image_metrics(&decoded,&readback,&psnr,&ssim)&&psnr>100.0&&ssim>0.999999;
 if(ok){readback.rgb[0]^=255;ok=ama10_image_metrics(&decoded,&readback,&psnr,&ssim)&&psnr<100.0&&ssim<1.0;}
 free(codes);ama10_image_free(&readback);ama10_image_free(&decoded);ama10_image_free(&input);ama10_image_codebook_free(&loaded);ama10_image_codebook_free(&b);
 {
  float samples[2*16*16*3];for(size_t i=0;i<16*16*3;i++){samples[i]=0.0f;samples[16*16*3+i]=1.0f;}
  AMA10ImageCodebook hi={0};AMA10Image source={35,23,(unsigned char*)malloc(35*23*3)},roundtrip={0};uint32_t *grid=NULL;size_t grid_count=0;
  if(!source.rgb)return 1;
  for(size_t y=0;y<23;y++)for(size_t x=0;x<35;x++){unsigned char v=x<18?0:255;size_t at=(y*35+x)*3;source.rgb[at]=source.rgb[at+1]=source.rgb[at+2]=v;}
  ok=ama10_image_codebook_fit(samples,2,16*16*3,2,16,16,5678,&hi)&&
      ama10_image_encode_codes(&source,&hi,&grid,&grid_count)&&grid_count==6&&
      ama10_image_decode_codes_size(&hi,grid,grid_count,35,23,&roundtrip)&&roundtrip.width==35&&roundtrip.height==23;
  free(grid);grid=NULL;ama10_image_free(&source);ama10_image_free(&roundtrip);ama10_image_codebook_free(&hi);
  if(!ok){fprintf(stderr,"variable-resolution image-code grid roundtrip failed\n");return 1;}
  {
   AMA10ImageCodebook high={0};uint32_t *high_codes=(uint32_t*)malloc(2720*sizeof(uint32_t));AMA10Image camera={0};
   if(!high_codes||!ama10_image_codebook_fit(samples,2,16*16*3,2,16,16,9012,&high)){free(high_codes);ama10_image_codebook_free(&high);return 1;}
   for(size_t i=0;i<2720;i++)high_codes[i]=1;
   ok=ama10_image_decode_codes_size(&high,high_codes,2720,1080,640,&camera)&&
      camera.width==1080&&camera.height==640&&camera.rgb[0]<32&&camera.rgb[(640*1080-1)*3]<32;
   free(high_codes);ama10_image_free(&camera);ama10_image_codebook_free(&high);
   if(!ok){fprintf(stderr,"1080x640 image-code output failed\n");return 1;}
  }
 }
 {
  enum{N=8,D=8*8*3};float samples[N*D];AMA10Image source[N]={{0}},decoded={0},decoded_rq={0};
  AMA10ImageCodebook base={0},rq={0},loaded_rq={0};char rqpath[1024];snprintf(rqpath,sizeof(rqpath),"%s/image-gen-rq-test.vq",argv[1]);
  int rq_ok=1;for(size_t i=0;i<N;i++){source[i].width=8;source[i].height=8;source[i].rgb=(unsigned char*)malloc(D);if(!source[i].rgb){rq_ok=0;break;}
   for(size_t j=0;j<D;j++){double q=0.06*(double)i+((i&1)?0.1:-0.1)*(double)(j%9)/8.0;if(q<0)q=0;if(q>1)q=1;unsigned char v=(unsigned char)(q*255.0+0.5);source[i].rgb[j]=v;samples[i*D+j]=(float)v/255.0f;}}
  if(rq_ok)rq_ok=ama10_image_codebook_fit(samples,N,D,2,8,8,777,&base)&&ama10_image_codebook_fit(samples,N,D,2,8,8,777,&rq)&&
     ama10_image_codebook_fit_residual(samples,N,&rq)&&rq.levels==2&&rq.residual_centers&&
     ama10_image_codebook_save(&rq,rqpath)&&ama10_image_codebook_load(&loaded_rq,rqpath)&&loaded_rq.levels==2;
  double mse_base=0,mse_rq=0;
  for(size_t i=0;i<N&&rq_ok;i++){uint32_t *cb=NULL,*cr=NULL;size_t nb=0,nr=0;
   rq_ok=ama10_image_encode_codes(&source[i],&base,&cb,&nb)&&nb==1&&ama10_image_encode_codes(&source[i],&loaded_rq,&cr,&nr)&&nr==2&&
      ama10_image_decode_codes_size(&base,cb,nb,8,8,&decoded)&&ama10_image_decode_codes_size(&loaded_rq,cr,nr,8,8,&decoded_rq);
   if(rq_ok)for(size_t j=0;j<D;j++){double a=(double)source[i].rgb[j]-decoded.rgb[j],b=(double)source[i].rgb[j]-decoded_rq.rgb[j];mse_base+=a*a;mse_rq+=b*b;}
   free(cb);free(cr);ama10_image_free(&decoded);ama10_image_free(&decoded_rq);
  }
  for(size_t i=0;i<N;i++){ama10_image_free(&source[i]);}
  ama10_image_codebook_free(&base);ama10_image_codebook_free(&rq);ama10_image_codebook_free(&loaded_rq);
  if(!rq_ok||(mse_base>1.0e-8&&!(mse_rq<mse_base))||mse_rq>mse_base+1.0e-8){fprintf(stderr,"residual VQ degraded patch reconstruction: base=%g residual=%g\n",mse_base,mse_rq);return 1;}
 }
 if(!ok){fprintf(stderr,"image codebook, residual quantization, serialization, metrics, or PPM round trip failed\n");return 1;}puts("image VQ/RQ fitting, serialization, 2D spatial neighbors, variable-resolution grids, 1080x640 decode, metrics, and PPM roundtrip: ok");return 0;}
