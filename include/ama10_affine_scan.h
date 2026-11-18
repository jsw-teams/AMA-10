#ifndef AMA10_AFFINE_SCAN_H
#define AMA10_AFFINE_SCAN_H

#include <stddef.h>

typedef struct AMA10AffineScanOpenCL AMA10AffineScanOpenCL;

/* Inclusive scan of independent affine state transforms over [batch,time,width].
   The pair at t represents h' = a[t] * h + b[t], elementwise. */
int ama10_affine_scan_cpu(const float *a,const float *b,size_t batch,size_t length,
                          size_t width,float *prefix_a,float *prefix_b);
AMA10AffineScanOpenCL *ama10_affine_scan_opencl_create(const char *kernel_path);
/* Cap the scan scratch working set; zero lets the device allocator choose. */
int ama10_affine_scan_opencl_set_memory_budget(AMA10AffineScanOpenCL *scan,size_t bytes);
int ama10_affine_scan_opencl_run(AMA10AffineScanOpenCL *scan,const float *a,const float *b,
                                 size_t batch,size_t length,size_t width,
                                 float *prefix_a,float *prefix_b);
const char *ama10_affine_scan_opencl_device(const AMA10AffineScanOpenCL *scan);
void ama10_affine_scan_opencl_destroy(AMA10AffineScanOpenCL *scan);

#endif
