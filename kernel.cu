#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include <iostream>

__global__ void sobel(unsigned char *red, unsigned char *out, int width, int height){
    int radius = 1;
    int ix = threadIdx.x;
    int stride = blockDim.x;
    for(int x = ix ; x < width*height;x += stride){
        int pos_x = x % width;
        int pos_y = x / width;
        float sum1 = 0;
        float sum2 = 0;
        unsigned char val =  red[x*4] / 3 + red[x*4 + 1] / 3 + red[x*4 + 2] / 3;
        float sobel_kernel_x[9] = {1,2,1,0,0,0,-1,-2,-1};
        float sobel_kernel_y[9] = {-1,0,1,-2,0,2,-1,0,1};
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
                unsigned char local_val =  red[index*4] / 3 + red[index*4 + 1] / 3 + red[index*4 + 2] / 3;
                dot += local_val / k_value ;
                sum += k_value;
                sum1 += sobel_kernel_x[k_id] * local_val;
                sum2 += sobel_kernel_y[k_id] * local_val;
                k_id++;
               }
           }
            // val =(char) dot*sum;
           val =(unsigned char)sqrt(sum1*sum1 + sum2*sum2);
           val = val > 70 ? 255 : 0;
            out[x*4 + 0] =  val;
            out[x*4 + 1] =  val;
            out[x*4 + 2] =  val;
            out[x*4 + 3] =  val;
        }
        else{

            out[x*4 + 0] = 255;
            out[x*4 + 1] = 201;
            out[x*4 + 2] = 175;
            out[x*4 + 3] = 158;
        }
    }
    // 0 - B 1 - G 2 - R 3 - Alpha?
}
 
void run(char *red, char *out, int width, int height){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);                
    unsigned char *r, *device_output;
    cudaMalloc((void**)&r, sizeof(char) * width * height * 4);
    cudaMalloc((void**)&device_output, sizeof(char) * width * height * 4);

    // // Transfer data from host to device memory
    cudaMemcpy(r, red, sizeof(char) * width * height* 4, cudaMemcpyHostToDevice);
    // // cVudaMemcpy(g, green, sizeof(int) * width * height, cudaMemcpyHostToDevice);
    // // cudaMemcpy(b, blue, sizeof(int) * width * height, cudaMemcpyHostToDevice);
    sobel<<<8,2048>>>(r,device_output,width,height);
    cudaMemcpy(out, device_output, sizeof(char) * width * height * 4, cudaMemcpyDeviceToHost);
    // for(int i = 0; i < 100; i++){
    //     // std::cout<<(int)device_output[i]<<" ";
    // }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("processing time: %f\n", ms);
}