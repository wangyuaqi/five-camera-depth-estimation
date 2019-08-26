
#include"configuration.h"
#include"util.h"
#include<opencv2/opencv.hpp>
#include"debug.h"
// do slip patch base sgm result
__constant__ int templates[25];
__global__ void
Slip_Image(uint8_t* data,uint8_t* dim_0,uint8_t* slip_result);

//*********gaussian Kernel**********
__global__ void
GaussianFilter(uint8_t* d_in,uint8_t* d_out,int width,int height);
//*********Sobel Kernel*************
__global__ void
SobelInCuda(u_int8_t* dataIn,u_int8_t* dataOut,int imgHeight,int imWidth);
cv::Mat Sobel_Deal(cv::Mat& src_img);
//*********get candidate disparity,and window width ,height;agsv***********
cv::Mat GetCandidate(cv::Mat& disparity_mat,cv::Mat& sobel_mat,cv::Mat& center_mat,cv::Mat& final_negative_mat);
