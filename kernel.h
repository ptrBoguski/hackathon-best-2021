#ifndef KERNEL_H
#define KERNEL_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

	void __declspec(dllexport) run(char* red, char* out, int width, int height);
#ifdef __cplusplus
}
#endif

#endif  // KERNEL_H