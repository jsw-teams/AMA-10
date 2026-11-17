#ifndef AMA10_IMAGE_H
#define AMA10_IMAGE_H

#include <stddef.h>

typedef struct {
    size_t width;
    size_t height;
    unsigned char *rgb; /* row-major interleaved RGB, 8 bits per channel */
} AMA10Image;

/* Zero-initialize image before load. Read binary PPM (P6, max value 255).
   No third-party image library required. */
int ama10_image_load_ppm(AMA10Image *image,const char *path);
/* Decode P6 PPM, JPEG, PNG, BMP, or TGA to 8-bit row-major RGB. */
int ama10_image_load(AMA10Image *image,const char *path);
void ama10_image_free(AMA10Image *image);
/* Set *patches to NULL before first use. Split into row-major patches; edge
   patches are resized by nearest neighbor so every pixel is covered. Features
   are RGB [0,1]. The caller owns and frees the returned array. */
int ama10_image_patchify(const AMA10Image *image,size_t patch_width,size_t patch_height,
                         float **patches,size_t *patch_count,size_t *feature_width);
/* Overlapping tiles keep edge and object detail while covering every source pixel. */
int ama10_image_patchify_overlap(const AMA10Image *image,size_t patch_width,size_t patch_height,
                                 size_t stride_x,size_t stride_y,float **patches,
                                 size_t *patch_count,size_t *feature_width);
/* Fine overlapping tiles followed by one scene view and four quadrant views. */
int ama10_image_patchify_multiscale(const AMA10Image *image,size_t patch_width,size_t patch_height,
                                    size_t stride_x,size_t stride_y,float **patches,
                                    size_t *patch_count,size_t *feature_width,
                                    size_t *fine_patch_count);

#endif
