#include "ama10.h"
#include "ama10_encoder.h"
#include "ama10_image.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void){
    static const char header[]="P6\r\n# 3 by 2 RGB fixture\r\n3 2\r\n255\r\n";
    static const unsigned char pixels[]={
        0,64,255, 255,128,0, 10,20,30,
        40,50,60, 70,80,90, 100,110,120
    };
    FILE *file=fopen("build/image-fixture.ppm","wb");
    if(!file)return 1;
    int written=fwrite(header,1,sizeof(header)-1,file)==sizeof(header)-1&&
        fwrite(pixels,1,sizeof(pixels),file)==sizeof(pixels);
    if(fclose(file)!=0||!written)return 2;
    AMA10Image image={0};float *patches=NULL;size_t patch_count=0,feature_width=0;
    if(!ama10_image_load_ppm(&image,"build/image-fixture.ppm")||
       image.width!=3||image.height!=2||memcmp(image.rgb,pixels,sizeof(pixels))||
       !ama10_image_patchify(&image,2,2,&patches,&patch_count,&feature_width)||
       patch_count!=2||feature_width!=12)return 3;
    if(fabsf(patches[0]-0.0f)>1.0e-7f||fabsf(patches[1]-64.0f/255.0f)>1.0e-7f||
       fabsf(patches[2]-1.0f)>1.0e-7f||fabsf(patches[6]-40.0f/255.0f)>1.0e-7f||
       fabsf(patches[10]-80.0f/255.0f)>1.0e-7f||
       fabsf(patches[12]-10.0f/255.0f)>1.0e-7f||
       fabsf(patches[21]-100.0f/255.0f)>1.0e-7f)return 4;
    {
        AMA10Image camera={1080,640,(unsigned char*)malloc(1080u*640u*3u)};
        float *pyramid=NULL;size_t count=0,width=0,fine=0;
        if(!camera.rgb)return 6;
        for(size_t y=0;y<camera.height;y++)for(size_t x=0;x<camera.width;x++){
            size_t at=(y*camera.width+x)*3;camera.rgb[at]=(unsigned char)(x%256);
            camera.rgb[at+1]=(unsigned char)(y%256);camera.rgb[at+2]=(unsigned char)((x+y)%256);
        }
        if(!ama10_image_patchify_multiscale(&camera,16,16,12,12,&pyramid,&count,&width,&fine)||
           fine!=4770||count!=4775||width!=768||!isfinite(pyramid[0])||
           !isfinite(pyramid[(count-1)*width+width-1]))return 7;
        free(pyramid);ama10_image_free(&camera);
    }
    {
        AMA10Model model={0};AMA10Optimizer optimizer={0};AMA10Encoder encoder={0};
        AMA10State state={0};float loss=0.0f;const uint32_t caption[]={1,2,3};
        const uint32_t language_prefix[]={4,5};
        float aligned_features[36];
        for(size_t i=0;i<36;i++)aligned_features[i]=patches[i%24];
        if(!ama10_model_init(&model,8,4,4,481)||!ama10_optimizer_init(&optimizer,&model)||
           !ama10_state_init(&state,model.width)||
           !ama10_encoder_init(&encoder,feature_width,4,582)||
           !ama10_train_image_caption_conditioned(&model,&optimizer,&state,&encoder,
               patches,patch_count,language_prefix,2,caption,3,2,0.001f,0.0f,&loss)||
           optimizer.step!=1||encoder.step!=1||!isfinite(loss)||
           !ama10_train_image_caption(&model,&optimizer,&state,&encoder,patches,patch_count,
                                      caption,3,2,0.001f,0.0f,&loss)||
           optimizer.step!=2||encoder.step!=2||!isfinite(loss)||
           !ama10_train_aligned_feature_text(&model,&optimizer,&state,&encoder,
               aligned_features,3,language_prefix,2,caption,3,2,0.001f,0.0f,&loss)||
           optimizer.step!=3||encoder.step!=3||!isfinite(loss))return 5;
        ama10_state_free(&state);ama10_encoder_free(&encoder);
        ama10_optimizer_free(&optimizer);ama10_model_free(&model);
    }
    ama10_image_free(&image);free(patches);
    puts("C P6 image decode plus shared-core conditioned, unconditioned, and aligned text updates: ok");
    return 0;
}
