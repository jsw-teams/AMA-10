#include "ama10_image.h"
#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb/stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

static int ppm_token(FILE *f,char *buffer,size_t capacity){
    int c;
    for(;;){
        c=fgetc(f);
        if(c==EOF)return 0;
        if(isspace((unsigned char)c))continue;
        if(c=='#'){
            do{c=fgetc(f);}while(c!=EOF&&c!='\n'&&c!='\r');
            continue;
        }
        break;
    }
    size_t n=0;
    do{
        if(n+1>=capacity)return 0;
        buffer[n++]=(char)c;
        c=fgetc(f);
    }while(c!=EOF&&!isspace((unsigned char)c));
    if(c==EOF)return 0;
    if(c=='\r'){
        int next=fgetc(f);
        if(next!='\n'&&next!=EOF)ungetc(next,f);
    }
    buffer[n]='\0';return 1;
}
static int ppm_size(const char *token,size_t *value){
    char *end=NULL;errno=0;
    unsigned long long parsed=strtoull(token,&end,10);
    if(errno||end==token||*end||!parsed||parsed>SIZE_MAX)return 0;
    *value=(size_t)parsed;return 1;
}
int ama10_image_load_ppm(AMA10Image *image,const char *path){
    if(!image||!path)return 0;
    FILE *f=fopen(path,"rb");if(!f)return 0;
    char magic[16],width_text[32],height_text[32],maxval_text[32];
    size_t width=0,height=0,maxval=0;
    int ok=ppm_token(f,magic,sizeof(magic))&&strcmp(magic,"P6")==0&&
       ppm_token(f,width_text,sizeof(width_text))&&ppm_size(width_text,&width)&&
       ppm_token(f,height_text,sizeof(height_text))&&ppm_size(height_text,&height)&&
       ppm_token(f,maxval_text,sizeof(maxval_text))&&ppm_size(maxval_text,&maxval)&&maxval==255&&
       width<=SIZE_MAX/height&&width*height<=SIZE_MAX/3;
    AMA10Image temp={0};
    if(ok){
        size_t bytes=width*height*3;
        temp.rgb=(unsigned char*)malloc(bytes);
        if(!temp.rgb)ok=0;
        else if(fread(temp.rgb,1,bytes,f)!=bytes)ok=0;
        temp.width=width;temp.height=height;
    }
    if(fclose(f)!=0)ok=0;
    if(ok){ama10_image_free(image);*image=temp;}
    else ama10_image_free(&temp);
    return ok;
}
int ama10_image_load(AMA10Image *image,const char *path){
    if(!image||!path)return 0;
    FILE *probe=fopen(path,"rb");if(!probe)return 0;
    unsigned char magic[2];size_t got=fread(magic,1,sizeof(magic),probe);
    int closed=fclose(probe)==0;
    if(!closed)return 0;
    if(got==2&&magic[0]=='P'&&magic[1]=='6')return ama10_image_load_ppm(image,path);
    int width=0,height=0,source_channels=0;
    stbi_uc *decoded=stbi_load(path,&width,&height,&source_channels,3);
    if(!decoded||width<=0||height<=0||(size_t)width>SIZE_MAX/(size_t)height||
       (size_t)width*(size_t)height>SIZE_MAX/3){stbi_image_free(decoded);return 0;}
    AMA10Image temp={(size_t)width,(size_t)height,decoded};
    ama10_image_free(image);*image=temp;return 1;
}
void ama10_image_free(AMA10Image *image){
    if(!image)return;
    free(image->rgb);memset(image,0,sizeof(*image));
}
int ama10_image_patchify_overlap(const AMA10Image *image,size_t patch_width,size_t patch_height,
                                 size_t stride_x,size_t stride_y,float **patches,
                                 size_t *patch_count,size_t *feature_width){
    if(!image||!image->rgb||!image->width||!image->height||!patch_width||!patch_height||
       !stride_x||!stride_y||stride_x>patch_width||stride_y>patch_height||!patches||
       !patch_count||!feature_width||patch_width>SIZE_MAX/patch_height||
       patch_width*patch_height>SIZE_MAX/3||*patches)return 0;
    size_t dx=image->width>patch_width?image->width-patch_width:0;
    size_t dy=image->height>patch_height?image->height-patch_height:0;
    size_t grid_width=dx/stride_x+(dx%stride_x!=0)+1;
    size_t grid_height=dy/stride_y+(dy%stride_y!=0)+1;
    if(grid_width>SIZE_MAX/grid_height)return 0;
    size_t count=grid_width*grid_height,features=patch_width*patch_height*3;
    if(count>SIZE_MAX/features||count*features>SIZE_MAX/sizeof(float))return 0;
    float *out=(float*)malloc(count*features*sizeof(float));if(!out)return 0;
    for(size_t py=0;py<grid_height;py++)for(size_t px=0;px<grid_width;px++){
        size_t x0=px*stride_x,y0=py*stride_y;
        if(x0>=image->width)x0=image->width-1;
        if(y0>=image->height)y0=image->height-1;
        size_t x1=x0+patch_width;if(x1>image->width)x1=image->width;
        size_t y1=y0+patch_height;if(y1>image->height)y1=image->height;
        size_t cell_width=x1-x0,cell_height=y1-y0;
        size_t patch_index=py*grid_width+px;
        for(size_t fy=0;fy<patch_height;fy++)for(size_t fx=0;fx<patch_width;fx++){
            size_t sx=x0+(size_t)(((double)fx*(double)cell_width)/(double)patch_width);
            size_t sy=y0+(size_t)(((double)fy*(double)cell_height)/(double)patch_height);
            if(sx>=x1)sx=x1-1;
            if(sy>=y1)sy=y1-1;
            size_t pixel=(sy*image->width+sx)*3;
            size_t offset=patch_index*features+(fy*patch_width+fx)*3;
            out[offset]=(float)image->rgb[pixel]/255.0f;
            out[offset+1]=(float)image->rgb[pixel+1]/255.0f;
            out[offset+2]=(float)image->rgb[pixel+2]/255.0f;
        }
    }
    *patches=out;*patch_count=count;*feature_width=features;return 1;
}
int ama10_image_patchify(const AMA10Image *image,size_t patch_width,size_t patch_height,
                         float **patches,size_t *patch_count,size_t *feature_width){
    return ama10_image_patchify_overlap(image,patch_width,patch_height,patch_width,patch_height,
                                        patches,patch_count,feature_width);
}
int ama10_image_patchify_multiscale(const AMA10Image *image,size_t patch_width,size_t patch_height,
                                    size_t stride_x,size_t stride_y,float **patches,
                                    size_t *patch_count,size_t *feature_width,
                                    size_t *fine_patch_count){
    if(!image||!image->rgb||!patches||*patches||!patch_count||!feature_width||!fine_patch_count||
       patch_width>SIZE_MAX/patch_height||patch_width*patch_height>SIZE_MAX/3)return 0;
    float *fine=NULL;size_t fine_count=0,features=0;
    if(!ama10_image_patchify_overlap(image,patch_width,patch_height,stride_x,stride_y,
                                     &fine,&fine_count,&features))return 0;
    if(fine_count>SIZE_MAX-5||(fine_count+5)>SIZE_MAX/features||
       (fine_count+5)*features>SIZE_MAX/sizeof(float)){free(fine);return 0;}
    float *all=(float*)malloc((fine_count+5)*features*sizeof(float));
    if(!all){free(fine);return 0;}
    memcpy(all,fine,fine_count*features*sizeof(float));free(fine);
    size_t regions[5][4]={{0,0,image->width,image->height},
        {0,0,(image->width+1)/2,(image->height+1)/2},
        {image->width/2,0,image->width,(image->height+1)/2},
        {0,image->height/2,(image->width+1)/2,image->height},
        {image->width/2,image->height/2,image->width,image->height}};
    for(size_t p=0;p<5;p++){
        size_t x0=regions[p][0],y0=regions[p][1],x1=regions[p][2],y1=regions[p][3];
        size_t rw=x1-x0,rh=y1-y0;
        for(size_t y=0;y<patch_height;y++)for(size_t x=0;x<patch_width;x++){
            size_t sx=x0+(size_t)(((uint64_t)x*(uint64_t)rw)/(uint64_t)patch_width);
            size_t sy=y0+(size_t)(((uint64_t)y*(uint64_t)rh)/(uint64_t)patch_height);
            if(sx>=x1)sx=x1-1;
            if(sy>=y1)sy=y1-1;
            size_t src=(sy*image->width+sx)*3;
            size_t dst=(fine_count+p)*features+(y*patch_width+x)*3;
            all[dst]=(float)image->rgb[src]/255.0f;
            all[dst+1]=(float)image->rgb[src+1]/255.0f;
            all[dst+2]=(float)image->rgb[src+2]/255.0f;
        }
    }
    *patches=all;*patch_count=fine_count+5;*feature_width=features;*fine_patch_count=fine_count;return 1;
}
