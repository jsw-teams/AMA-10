#include "ama10_imggen.h"
#include <math.h>
#include <stdio.h>
int main(int argc,char**argv){if(argc!=3){fprintf(stderr,"usage: %s <reference.ppm> <candidate.ppm>\n",argv[0]);return 2;}double psnr=0,ssim=0;
 if(!ama10_image_score_ppm(argv[1],argv[2],&psnr,&ssim)){fprintf(stderr,"could not score matching P6 PPM images\n");return 1;}
 if(isinf(psnr))printf("psnr=inf ssim=%.6f\n",ssim);else printf("psnr=%.6f ssim=%.6f\n",psnr,ssim);return 0;}
