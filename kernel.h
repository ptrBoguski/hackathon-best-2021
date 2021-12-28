#ifndef KERNEL_H
#define KERNEL_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

	void __declspec(dllexport) run(char *input_pixels, char *out, int width, int height, float *ms,
                                            int threshold, float gauss_sigma, bool enable_gauss, bool enable_nms, float* sobel_Gx, float* sobel_Gy);
#ifdef __cplusplus
}
#endif

#endif  // KERNEL_H