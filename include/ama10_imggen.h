#ifndef AMA10_IMGGEN_H
#define AMA10_IMGGEN_H

#include "ama10_image.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t patch_width, patch_height, stride_width, stride_height;
    uint32_t code_count, feature_width, latent_width, levels, context_code_count;
    uint64_t train_hash;
    float *centers, *residual_centers, *context_centers;
    float *encoder, *encoder_bias, *decoder, *decoder_bias;
} AMA10ImageCodebook;

void ama10_image_codebook_free(AMA10ImageCodebook *book);
int ama10_image_codebook_fit(const float *patches, size_t patch_count,
                             size_t feature_width, uint32_t code_count,
                             uint32_t patch_width, uint32_t patch_height,
                             uint64_t train_hash, AMA10ImageCodebook *book);
/* Fit the trainable spatial codec and latent-space codebook on the OpenCL GPU. */
int ama10_image_codebook_fit_opencl(const float *patches,size_t patch_count,
                             size_t feature_width,uint32_t code_count,
                             uint32_t patch_width,uint32_t patch_height,
                             uint64_t train_hash,const char *kernel_path,
                             AMA10ImageCodebook *book);
/* Add a second nearest-centroid stage fitted to the first-stage residuals. */
int ama10_image_codebook_fit_residual(const float *patches,size_t patch_count,
                                      AMA10ImageCodebook *book);
/* Fit a separate train-only codebook for global and quadrant context patches. */
int ama10_image_codebook_fit_context(const float *patches,size_t patch_count,
                                     uint32_t context_code_count,
                                     AMA10ImageCodebook *book);
int ama10_image_codebook_save(const AMA10ImageCodebook *book, const char *path);
int ama10_image_codebook_load(AMA10ImageCodebook *book, const char *path);
int ama10_image_codebook_quantize(const AMA10ImageCodebook *book,
                                  const float *patch, uint32_t *code);
int ama10_image_context_quantize(const AMA10ImageCodebook *book,
                                 const float *patch,uint32_t *code);
/* Previous raster patches used as position-tagged 2D neighborhood context. */
size_t ama10_image_spatial_neighbors(size_t position,size_t grid_width,size_t grid_height,
                                     size_t neighbors[8]);
int ama10_image_encode_codes(const AMA10Image *image,
                             const AMA10ImageCodebook *book,
                             uint32_t **codes, size_t *code_count);
int ama10_image_decode_codes(const AMA10ImageCodebook *book,
                             const uint32_t *codes, size_t code_count,
                             AMA10Image *image);
/* Decode a variable-size grid of stride-overlapped visual codes. For a residual
   codebook, each patch position has `levels` consecutive code IDs. */
int ama10_image_decode_codes_size(const AMA10ImageCodebook *book,
                                  const uint32_t *codes,size_t code_count,
                                  size_t width,size_t height,AMA10Image *image);
int ama10_image_save_ppm(const AMA10Image *image, const char *path);
/* Global RGB PSNR and non-overlapping 8x8-window SSIM, both on 8-bit pixels. */
int ama10_image_metrics(const AMA10Image *reference, const AMA10Image *candidate,
                        double *psnr, double *ssim);
int ama10_image_score_ppm(const char *reference_path, const char *candidate_path,
                          double *psnr, double *ssim);

#endif
