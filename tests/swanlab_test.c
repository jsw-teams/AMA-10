#define _POSIX_C_SOURCE 200809L
#include "ama10_swanlab.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void){
#ifdef _WIN32
    _putenv_s("SWANLAB_API_KEY","");
#else
    unsetenv("SWANLAB_API_KEY");
#endif
    char error[128]={0};
    if(ama10_swanlab_upload_csv("https://api.swanlab.cn","user","project","run","missing.csv",error,sizeof(error)))return 1;
    if(strcmp(error,"SWANLAB_API_KEY is not set")){fprintf(stderr,"unexpected validation error: %s\n",error);return 1;}
    puts("SwanLab uploader rejects missing credentials before making requests: ok");
    return 0;
}
