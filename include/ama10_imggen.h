#ifndef AMA10_IMGGEN_H
#define AMA10_IMGGEN_H

#include "ama10_image.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t patch_width, patch_height, stride_width, stride_height;
    uint32_t code_count, feature_width;
    uint64_t train_hash;
    float *centers;
} AMA10ImageCodebook;

void ama10_image_codebook_free(AMA10ImageCodebook *book);
int ama10_image_codebook_fit(const float *patches, size_t patch_count,
                             size_t feature_width, uint32_t code_count,
                             uint32_t patch_width, uint32_t patch_height,
                             uint64_t train_hash, AMA10ImageCodebook *book);
int ama10_image_codebook_save(const AMA10ImageCodebook *book, const char *path);
int ama10_image_codebook_load(AMA10ImageCodebook *book, const char *path);
int ama10_image_codebook_quantize(const AMA10ImageCodebook *book,
                                  const float *patch, uint32_t *code);
int ama10_image_encode_codes(const AMA10Image *image,
                             const AMA10ImageCodebook *book,
                             uint32_t **codes, size_t *code_count);
int ama10_image_decode_codes(const AMA10ImageCodebook *book,
                             const uint32_t *codes, size_t code_count,
                             AMA10Image *image);
/* Decode a variable-size grid of non-overlapping visual codes. Partial edge
   tiles are cropped to the requested aspect ratio after edge-resized encoding. */
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
