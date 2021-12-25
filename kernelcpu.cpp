#include <math.h>

extern "C"
{ 
    //TODO 
__declspec(dllexport) void run(char *red, char *out, int width, int height)
{
       int radius = 1;
       int ix = 0;
       int stride = 1;
       for(int x = ix ; x < width*height;x += stride){
           int pos_x = x % width;
           int pos_y = x / width;
           long long sum1 = 0;
           long long sum2 = 0;
           unsigned char val = 0;// red[x*4] / 3 + red[x*4 + 1] / 3 + red[x*4 + 2] / 3;
           float sobel_kernel_x[9] = {1,1,1,0,0,0,-1,-1,-1};
           float sobel_kernel_y[9] = {-1,0,1,-1,0,1,-1,0,1};
           if(pos_x > radius && pos_x + radius <  width && pos_y > radius && pos_y + radius < height)
           {
               float dot = 0;
               float sum = 0;
               int k_id = 0;
               for(int x2 = -radius; x2 <= radius; x2++)
               {
                   for(int y2 = -radius; y2 <= radius; y2++){
                   float k_value = -(x2*x2 + y2*y2 + 1);
                   int index = x + y2 * width + x2;
                   int local_val = (unsigned char) red[index*4] / 3 + (unsigned char) red[index*4 + 1] / 3 + (unsigned char) red[index*4 + 2] / 3;
                   dot += local_val / k_value ;
                   sum += k_value;
                   sum1 += sobel_kernel_x[k_id] * local_val;
                   sum2 += sobel_kernel_y[k_id] * local_val;
                   k_id++;
                  }
              }
              int s = sqrt(sum1*sum1 + sum2*sum2);
              val = s > 255? 255 : (unsigned char) s;
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
}