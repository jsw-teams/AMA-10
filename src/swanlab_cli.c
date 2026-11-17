#include "ama10_swanlab.h"
#include <stdlib.h>
#include <stdio.h>
int main(int argc,char **argv){if(argc<5||argc>6){fprintf(stderr,"usage: %s <workspace> <project> <experiment-name> <metrics.csv> [api-origin]\nCSV columns: step,key,value\nSet SWANLAB_API_KEY before running.\n",argv[0]);return 2;}char error[256];const char *origin=argc==6?argv[5]:getenv("SWANLAB_API_HOST");if(!ama10_swanlab_upload_csv(origin,argv[1],argv[2],argv[3],argv[4],error,sizeof(error))){fprintf(stderr,"%s\n",error[0]?error:"SwanLab metric upload failed");return 1;}return 0;}
