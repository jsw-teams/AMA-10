#include "ama10_imggen.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc,char **argv){
    if(argc!=3)return 2;
    enum{D=8*8*3};float patches[2*D];
    for(size_t i=0;i<D;i++){patches[i]=0.0f;patches[D+i]=1.0f;}
    AMA10ImageCodebook book={0},loaded={0};AMA10Image source={8,8,(unsigned char*)malloc(D)},decoded={0};
    uint32_t *codes=NULL;size_t code_count=0;char path[1024];
    if(!source.rgb)return 1;
    for(size_t i=0;i<D;i++)source.rgb[i]=(unsigned char)(patches[i]*255.0f);
    snprintf(path,sizeof(path),"%s/image-gen-opencl-codec.vq",argv[1]);
    int ok=ama10_image_codebook_fit_opencl(patches,2,D,2,8,8,771,argv[2],&book)&&
        book.latent_width>0&&book.latent_width<D&&
        ama10_image_codebook_save(&book,path)&&ama10_image_codebook_load(&loaded,path)&&
        loaded.latent_width==book.latent_width&&loaded.encoder&&loaded.decoder&&
        ama10_image_encode_codes(&source,&loaded,&codes,&code_count)&&code_count==1&&
        ama10_image_decode_codes_size(&loaded,codes,code_count,8,8,&decoded);
    for(size_t i=0;i<(size_t)loaded.latent_width*loaded.feature_width&&ok;i++)
        if(!isfinite(loaded.encoder[i])||!isfinite(loaded.decoder[i]))ok=0;
    if(!ok)fprintf(stderr,"OpenCL trainable spatial codec fit/serialization/decode failed\n");
    free(codes);
    ama10_image_free(&decoded);ama10_image_free(&source);
    ama10_image_codebook_free(&loaded);ama10_image_codebook_free(&book);
    if(remove(path)!=0)ok=0;
    if(ok)printf("OpenCL spatial latent codec training, latent VQ, versioned persistence and decode: ok\n");
    return ok?0:1;
}
