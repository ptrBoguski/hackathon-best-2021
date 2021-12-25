#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include <iostream>
#define _USE_MATH_DEFINES
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif
#ifndef M_E
    #define M_E 2.718281828459045
#endif
void gaussMatrix(int radius,float sigma, float *output){
int index = 0;
double der = 1/(2 * M_PI * sigma * sigma);
  for(int x = -radius; x <= radius; x++){
                for(int y = -radius; y <= radius; y++){
                   output[index++] = der * pow(M_E, -(x*x + y*y) / (2 * sigma * sigma));
                  
               }
           }
}
__global__ void gauss(unsigned char *red, unsigned char *out,float* gaussianMatrix, int radius, int width, int height){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x;
    for(int x = ix ; x < width*height;x += stride * gridDim.x){
// int ix = 0;
// int stride = 1;
// for(int x = ix; x < width*height; x += stride){
        int pos_x = x % width;
        int pos_y = x / width;
        float sum = 0;
        unsigned char val =  red[x*4] / 3 + red[x*4 + 1] / 3 + red[x*4 + 2] / 3;
        if(pos_x > radius && pos_x + radius <  width && pos_y > radius && pos_y + radius < height)
        {
            float sum = 0;
            int k_id = 0;
            for(int x2 = -radius; x2 <= radius; x2++){
                for(int y2 = -radius; y2 <= radius; y2++){
                float k_value = -(x2*x2 + y2*y2 + 1);
                int index = x + y2 * width + x2;
                int local_val =  red[index*4] / 3 + red[index*4 + 1] / 3 + red[index*4 + 2] / 3;
                sum +=gaussianMatrix[k_id] * local_val;
                k_id++;
               }
           }
            int s = sum;
            val = (unsigned char) s;
            out[x*4 + 0] =  val;
            out[x*4 + 1] =  val;
            out[x*4 + 2] =  val;
            out[x*4 + 3] = 255;
        }
        else{
            out[x*4 + 0] = 0;
            out[x*4 + 1] = 0;
            out[x*4 + 2] = 0;
            out[x*4 + 3] = 0;
        }
    }
}
__global__ void sobel(unsigned char *red, unsigned char *out, int width, int height){
    int radius = 1;
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x;
   
    for(int x = ix ; x < width*height;x += stride * gridDim.x){
//        int ix = 0;
//        int stride = 1;
//        for(int x = ix ; x < width*height;x += stride){
        int pos_x = x % width;
        int pos_y = x / width;
        float sum1 = 0;
        float sum2 = 0;
        unsigned char val =  red[x*4] / 3 + red[x*4 + 1] / 3 + red[x*4 + 2] / 3;
        float sobel_kernel_x[9] = {1,1,1,0,0,0,-1,-1,-1};
        float sobel_kernel_y[9] = {-1,0,1,-1,0,1,-1,0,1};
        // float sobel_kernel_x[] = {2,1,0,-1,-2,2,1,0,-1,-2,4,2,0,-2,-4,2,1,0,-1,-2,2,1,0,-1,-2};
        // float sobel_kernel_y[] = {2,2,4,2,2,1,1,2,1,1,0,0,0,0,0,-1,-1,-2,-1,-1,-2,-2,-4,-2,-2};
        if(pos_x > radius && pos_x + radius <  width && pos_y > radius && pos_y + radius < height)
        {
            float dot = 0;
            float sum = 0;
            int k_id = 0;
            for(int x2 = -radius; x2 <= radius; x2++){
                for(int y2 = -radius; y2 <= radius; y2++){
                float k_value = -(x2*x2 + y2*y2 + 1);
                int index = x + y2 * width + x2;
                int local_val =  red[index*4] / 3 + red[index*4 + 1] / 3 + red[index*4 + 2] / 3;
                dot += local_val / k_value ;
                sum += k_value;
                sum1 += sobel_kernel_x[k_id] * local_val;
                sum2 += sobel_kernel_y[k_id] * local_val;
                k_id++;
               }
           }
            // val =(char) dot*sum;
           int s = sqrt(sum1*sum1 + sum2*sum2);
           val = s > 255? 255 : (unsigned char) s;
//         val = (unsigned char) sum2;
            out[x*4 + 0] =  val;
            out[x*4 + 1] =  val;
            out[x*4 + 2] =  val;
            out[x*4 + 3] = 255;
        }
        else{

            out[x*4 + 0] = 255;
            out[x*4 + 1] = 255;
            out[x*4 + 2] = 255;
            out[x*4 + 3] = 255;
        }
    }
    // 0 - B 1 - G 2 - R 3 - Alpha?
}
 
void run(char *red, char *out, int width, int height, float * ms){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);                
    unsigned char *r,*gauss_output, *device_output;
    cudaMalloc((void**)&r, sizeof(char) * width * height * 4);
    cudaMalloc((void**)&device_output, sizeof(char) * width * height * 4);
    cudaMalloc((void**)&gauss_output, sizeof(char) * width * height * 4);
    float *gauss_device;
    float gaussian[25];
    cudaMalloc((void**)&gauss_device, sizeof(float) * 25);
    int radius = 2;
    float sigma = 0.6;
    gaussMatrix(radius, sigma, gaussian);
    cudaMemcpy(gauss_device, gaussian, sizeof(float) * 25, cudaMemcpyHostToDevice);
    for(int i = 0; i <25; i++){
    printf("%f ",gaussian[i]);
    if ((i + 1)%5 == 0){
    printf("\n");
    }
    }
    cudaMemcpy(r, red, sizeof(char) * width * height* 4, cudaMemcpyHostToDevice);
    gauss<<<8, 512>>>(r,gauss_output, gauss_device, 2, width, height);
    sobel<<<8,512>>> (gauss_output,device_output,width,height);
    cudaMemcpy(out, device_output, sizeof(char) * width * height * 4, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms[0], start, stop);
    printf("processing time: %f\n", ms[0]);
}