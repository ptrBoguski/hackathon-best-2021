#include "kernel.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.718281828459045
#endif
#define GAUSS_RADIUS 2
void gaussMatrix(int radius, float sigma, float *output) {
  int index = 0;
  double der = 1 / (2 * M_PI * sigma * sigma);
  for (int x = -radius; x <= radius; x++) {
    for (int y = -radius; y <= radius; y++) {
      output[index++] = der * pow(M_E, -(x * x + y * y) / (2 * sigma * sigma));
    }
  }
}
__global__ void gauss(unsigned char *input_pixels, unsigned char *out,
                      float *gaussianMatrix, int radius, int width,
                      int height) {
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x;
  for (int x = ix; x < width * height; x += stride * gridDim.x) {
    int pos_x = x % width;
    int pos_y = x / width;
    float sum = 0;
    unsigned char val =
        input_pixels[x * 4] / 3 + input_pixels[x * 4 + 1] / 3 + input_pixels[x * 4 + 2] / 3;
    if (pos_x > radius && pos_x + radius < width && pos_y > radius &&
        pos_y + radius < height) {
      float sum = 0;
      int k_id = 0;
      for (int x2 = -radius; x2 <= radius; x2++) {
        for (int y2 = -radius; y2 <= radius; y2++) {
          float k_value = -(x2 * x2 + y2 * y2 + 1);
          int index = x + y2 * width + x2;
          int local_val = input_pixels[index * 4] / 3 + input_pixels[index * 4 + 1] / 3 +
                          input_pixels[index * 4 + 2] / 3;
          sum += gaussianMatrix[k_id] * local_val;
          k_id++;
        }
      }
      int s = sum;
      val = s > 255 ? 255 : (unsigned char)s;
      out[x * 4 + 0] = val;
      out[x * 4 + 1] = val;
      out[x * 4 + 2] = val;
      out[x * 4 + 3] = 255;
    } else {
      out[x * 4 + 0] = 0;
      out[x * 4 + 1] = 0;
      out[x * 4 + 2] = 0;
      out[x * 4 + 3] = 255;
    }
  }
}
__global__ void sobel(unsigned char *input_pixels, unsigned char *out, float *grad,
                      int width, int height, int threshold, float* sobel_kernel_x, float* sobel_kernel_y) {
  int radius = 1;
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x;

  for (int x = ix; x < width * height; x += stride * gridDim.x) {
    int pos_x = x % width;
    int pos_y = x / width;
    float sum1 = 0;
    float sum2 = 0;
    unsigned char val =
        input_pixels[x * 4] / 3 + input_pixels[x * 4 + 1] / 3 + input_pixels[x * 4 + 2] / 3;
    if (pos_x > GAUSS_RADIUS && pos_x + GAUSS_RADIUS < width && pos_y >  GAUSS_RADIUS &&
        pos_y + GAUSS_RADIUS < height) {
      float dot = 0;
      float sum = 0;
      int k_id = 0;
      for (int x2 = -radius; x2 <= radius; x2++) {
        for (int y2 = -radius; y2 <= radius; y2++) {
          float k_value = -(x2 * x2 + y2 * y2 + 1);
          int index = x + y2 * width + x2;
          int local_val = input_pixels[index * 4] / 3 + input_pixels[index * 4 + 1] / 3 +
                          input_pixels[index * 4 + 2] / 3;
          dot += local_val / k_value;
          sum += k_value;
          sum1 += sobel_kernel_x[k_id] * local_val;
          sum2 += sobel_kernel_y[k_id] * local_val;
          k_id++;
        }
      }
      // val =(char) dot*sum;
      int s = sqrt(sum1 * sum1 + sum2 * sum2);
      grad[x] = atan(sum2 / (sum1 + 0.0001)) * 180 / M_PI;
    
      val = s > 255 ? 255 : (unsigned char)s;
      val = s < threshold ? 0 : s;
      out[x * 4 + 0] = val;
      out[x * 4 + 1] = val;
      out[x * 4 + 2] = val;
      out[x * 4 + 3] = 255;
    } else {

      out[x * 4 + 0] = 0;
      out[x * 4 + 1] = 0;
      out[x * 4 + 2] = 0;
      out[x * 4 + 3] = 255;
    }
  }
  // 0 - B 1 - G 2 - R 3 - Alpha?
}
__global__ void nms(float *grad, unsigned char *pixels, unsigned char *out,
                    int width, int height) {

  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x;
  const int radius = 1;
  for (int x = ix; x < width * height; x += stride * gridDim.x) {
    int pos_x = x % width;
    int pos_y = x / width;
    if (pos_x > radius && pos_x + radius < width && pos_y > radius &&
        pos_y + radius < height) {
      float dot = 0;
      float sum = 0;
      int k_id = 0;
      float local_gradient = grad[x];
      if ((grad[x] >= 0 && grad[x] <= 45) ||
          (grad[x] < -180 + 45 && grad[x] >= -180)) {
        if (pixels[x * 4] >= pixels[(x + 1 ) * 4] &&
            pixels[x * 4] >= pixels[(x - 1 ) * 4]) {
          out[x * 4] = pixels[x * 4];
          out[x * 4 + 1] = pixels[x * 4];
          out[x * 4 + 2] = pixels[x * 4];
          out[x * 4 + 3] = 255;
        } else {
          out[x * 4] = 0;
          out[x * 4 + 1] = 0;
          out[x * 4 + 2] = 0;
          out[x * 4 + 3] = 255;
        }

      } else if ((grad[x] > 45 && grad[x] <= 90) ||
                 (grad[x] < -90 && grad[x] >= -90 - 45)) {
        if (pixels[x * 4] >= pixels[(x - width ) * 4] &&
            pixels[x * 4] >= pixels[(x + width ) * 4]) {
          out[x * 4] = pixels[x * 4];
          out[x * 4 + 1] = pixels[x * 4];
          out[x * 4 + 2] = pixels[x * 4];
          out[x * 4 + 3] = 255;
        } else {
          out[x * 4] = 0;
          out[x * 4 + 1] = 0;
          out[x * 4 + 2] = 0;
          out[x * 4 + 3] = 255;
        }
      } else if ((grad[x] > 90 && grad[x] <= 135) ||
                 (grad[x] < -45 && grad[x] >= -90)) {
        if (pixels[x * 4] >= pixels[(x  + width) * 4] &&
            pixels[x * 4] >= pixels[(x - 1 - width) * 4]) {
          out[x * 4] = pixels[x * 4];
          out[x * 4 + 1] = pixels[x * 4];
          out[x * 4 + 2] = pixels[x * 4];
          out[x * 4 + 3] = 255;
        } else {
          out[x * 4] = 0;
          out[x * 4 + 1] = 0;
          out[x * 4 + 2] = 0;
          out[x * 4 + 3] = 255;
        }
      } else if ((grad[x] >  90 + 45 && grad[x] <= 180) ||
                 (grad[x] < 0 && grad[x] >= -45)) {
        if (pixels[x * 4] >= pixels[(x + 1 + width) * 4] &&
            pixels[x * 4] >= pixels[(x - 1 - width) * 4]) {
          out[x * 4] = pixels[x * 4];
          out[x * 4 + 1] = pixels[x * 4];
          out[x * 4 + 2] = pixels[x * 4];
          out[x * 4 + 3] = 255;
        } else {
          out[x * 4] = 0;
          out[x * 4 + 1] = 0;
          out[x * 4 + 2] = 0;
          out[x * 4 + 3] = 255;
        }
      }
    }
  }
}
void run(char *input_pixels, char *out, int width, int height, float *ms,
         int threshold, float gauss_sigma, bool enable_gauss, bool enable_nms, float* sobel_Gx, float* sobel_Gy) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  unsigned char *r, *gauss_output, *device_output, *sobel_output;
  float *sobel_gradient, *temp;
  cudaMalloc((void **)&r, sizeof(char) * width * height * 4);
  cudaMalloc((void **)&device_output, sizeof(char) * width * height * 4);
  cudaMalloc((void **)&sobel_output, sizeof(char) * width * height * 4);
  cudaMalloc((void **)&sobel_gradient, sizeof(float) * width * height);
  cudaMalloc((void **)&gauss_output, sizeof(char) * width * height * 4);
  temp = new float[width * height];
  int radius = GAUSS_RADIUS;

  int gaussian_elements = (radius * 2 + 1) * (radius * 2 + 1);
  float *gauss_device, *sobel_kernel_Gx, *sobel_kernel_Gy;
  float *gaussian = new float[gaussian_elements];
  cudaMalloc((void **)&gauss_device, sizeof(float) * gaussian_elements);
  cudaMalloc((void **)&sobel_kernel_Gx, sizeof(float) * 9 );
  cudaMalloc((void **)&sobel_kernel_Gy, sizeof(float) * 9 );
  float sigma = gauss_sigma;
  gaussMatrix(radius, sigma, gaussian);
  cudaMemcpy(gauss_device, gaussian, sizeof(float) * gaussian_elements,
             cudaMemcpyHostToDevice);
  cudaMemcpy(sobel_kernel_Gx, sobel_Gx, sizeof(float) * 9,
             cudaMemcpyHostToDevice);
  cudaMemcpy(sobel_kernel_Gy, sobel_Gy, sizeof(float) * 9,
             cudaMemcpyHostToDevice);
  for (int i = 0; i < gaussian_elements; i++) {
    printf("%f ", gaussian[i]);
    if ((i + 1) % (2 * radius + 1) == 0) {
      printf("\n");
    }
  }
  cudaMemcpy(r, input_pixels, sizeof(char) * width * height * 4, cudaMemcpyHostToDevice);
  if(enable_gauss)
    gauss<<<8, 512>>>(r, gauss_output, gauss_device, 2, width, height);
  else
   cudaMemcpy(gauss_output, input_pixels, sizeof(char) * width * height * 4, cudaMemcpyHostToDevice);
  sobel<<<8, 512>>>(gauss_output, sobel_output, sobel_gradient, width, height,
                    threshold, sobel_kernel_Gx, sobel_kernel_Gy);
  if(enable_nms){
    nms<<<8, 512>>>(sobel_gradient, sobel_output, device_output, width, height);
  cudaMemcpy(out, device_output, sizeof(char) * width * height * 4,
             cudaMemcpyDeviceToHost);
  }
  else
   cudaMemcpy(out, sobel_output, sizeof(char) * width * height * 4, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms[0], start, stop);
  printf("processing time: %f\n", ms[0]);
}