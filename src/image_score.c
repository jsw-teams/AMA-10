#include "ama10_imggen.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char**argv){
 if(argc==5&&strcmp(argv[1],"--reconstruct")==0){
  AMA10Image reference={0},reconstruction={0};AMA10ImageCodebook book={0};uint32_t*codes=NULL;size_t count=0;double psnr=0,ssim=0;
  int ok=ama10_image_load(&reference,argv[2])&&ama10_image_codebook_load(&book,argv[3])&&
    ama10_image_encode_codes(&reference,&book,&codes,&count)&&
    ama10_image_decode_codes_size(&book,codes,count,reference.width,reference.height,&reconstruction)&&
    ama10_image_save_ppm(&reconstruction,argv[4])&&
    ama10_image_metrics(&reference,&reconstruction,&psnr,&ssim);
  if(ok)printf("reconstruction_codes=%zu levels=%u size=%zux%zu psnr=%.6f ssim=%.6f output=%s\n",count,book.levels,reference.width,reference.height,psnr,ssim,argv[4]);
  else fprintf(stderr,"image-code reconstruction failed\n");
  free(codes);ama10_image_free(&reference);ama10_image_free(&reconstruction);ama10_image_codebook_free(&book);return ok?0:1;
 }
 if(argc!=3){fprintf(stderr,"usage: %s <reference.ppm> <candidate.ppm>\n       %s --reconstruct <input-image> <codebook.vq> <output.ppm>\n",argv[0],argv[0]);return 2;}double psnr=0,ssim=0;
 if(!ama10_image_score_ppm(argv[1],argv[2],&psnr,&ssim)){fprintf(stderr,"could not score matching P6 PPM images\n");return 1;}
 if(isinf(psnr))printf("psnr=inf ssim=%.6f\n",ssim);else printf("psnr=%.6f ssim=%.6f\n",psnr,ssim);return 0;}
