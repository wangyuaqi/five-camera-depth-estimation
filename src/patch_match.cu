#include"include/patch_match.h"
__global__ void
Slip_Image(uint8_t* data,uint8_t* dim_0,uint8_t* slip_result)
{
    //
    const int y= blockIdx.x;  // y is CTA Identifier
    const int THRid = threadIdx.x; // THRid is Thread Identifier
}

__global__ void
GaussianFilter(uint8_t* d_in,uint8_t* d_out,int width,int height)
{
        int tidx = blockDim.x * blockIdx.x + threadIdx.x;
        int tidy = blockDim.y * blockIdx.y + threadIdx.y;


        int sum = 0;
        int index = 0;


        if (tidx>2 && tidx<width - 2 && tidy>2 && tidy<height - 2)
       {

        for (int m = tidx - 2; m < tidx + 3; m++)
        {
            for (int n = tidy - 2; n < tidy + 3; n++)
            {
                sum += d_in[m*width + n] * templates[index++];
            }
        }
        if (sum / 273<0)
            *(d_out + (tidx)*width + tidy) = 0;
        else if (sum / 273>255)
            *(d_out + (tidx)*width + tidy) = 255;
        else
            *(d_out + (tidx)*width + tidy) = sum / 273;
        }
        else
        {
            *(d_out + (tidx)*width + tidy)=*(d_in + (tidx)*width + tidy);
        }

}

__global__ void
SobelInCuda(uint8_t* dataIn,uint8_t* dataOut,int imgHeight,int imgWidth)
{
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
        int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
        int index = yIndex * imgWidth + xIndex;
        int Gx = 0;
        int Gy = 0;

        if (xIndex > 0 && xIndex < imgWidth - 1 && yIndex > 0 && yIndex < imgHeight - 1)
        {
            Gx = dataIn[(yIndex - 1) * imgWidth + xIndex + 1] + 2 * dataIn[yIndex * imgWidth + xIndex + 1] + dataIn[(yIndex + 1) * imgWidth + xIndex + 1]
                - (dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[yIndex * imgWidth + xIndex - 1] + dataIn[(yIndex + 1) * imgWidth + xIndex - 1]);
            Gy = dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[(yIndex - 1) * imgWidth + xIndex] + dataIn[(yIndex - 1) * imgWidth + xIndex + 1]
                - (dataIn[(yIndex + 1) * imgWidth + xIndex - 1] + 2 * dataIn[(yIndex + 1) * imgWidth + xIndex] + dataIn[(yIndex + 1) * imgWidth + xIndex + 1]);
            dataOut[index] = (abs(Gx) + abs(Gy)) / 2;
        }

}

cv::Mat Sobel_Deal(cv::Mat& src_img)
{
    uint8_t* d_in;
    uint8_t* d_out;
    uint8_t* g_out;
    uint8_t* c_out;
    int imHeight=src_img.rows;
    int imWidth=src_img.cols;
    cudaMalloc((void**)&d_in,imHeight*imWidth*sizeof(uint8_t));
    cudaMalloc((void**)&d_out,imHeight*imWidth*sizeof(uint8_t));
    cudaMalloc((void**)&g_out,imHeight*imWidth*sizeof(uint8_t));
    cudaMalloc((void**)&c_out,imHeight*imWidth*sizeof(uint8_t));
    cudaMemcpy(d_in,src_img.data,imHeight*imWidth*sizeof(uint8_t),cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(32,32);
    dim3 blocksPerGrid;
    blocksPerGrid.x=(imWidth+threadsPerBlock.x-1)/threadsPerBlock.x;
    blocksPerGrid.y=(imHeight+threadsPerBlock.y-1)/threadsPerBlock.y;

    int Gaussian[25] = { 1, 4, 7, 4, 1,
            4, 16, 26, 16, 4,
            7, 26, 41, 26, 7,
            4, 16, 26, 16, 4,
            1, 4, 7, 4, 1 };
    cudaMemcpyToSymbol(templates, Gaussian, 25 * sizeof(int));
    GaussianFilter<<<blocksPerGrid,threadsPerBlock>>>(d_in,g_out,imHeight,imWidth);
    SobelInCuda<<<blocksPerGrid,threadsPerBlock>>>(g_out,d_out,imHeight,imWidth);
    //*******do filter again************
    GaussianFilter<<<blocksPerGrid,threadsPerBlock>>>(d_out,c_out,imHeight,imWidth);
    //**********************************
    cv::Mat result_mat(imHeight,imWidth,CV_8UC1);
    //cudaMemcpy(result_mat.data,d_out,imHeight*imWidth*sizeof(uint8_t),cudaMemcpyDeviceToHost);
    cudaMemcpy(result_mat.data,c_out,imHeight*imWidth*sizeof(uint8_t),cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    return result_mat;

}

//*************get candidate disparity for each point store min_level,max_level*****
cv::Mat GetCandidate(cv::Mat& disparity_mat,cv::Mat& sobel_mat,cv::Mat& center_mat,cv::Mat& final_negative_mat)
{
  int img_width=center_mat.cols;
  int img_height=center_mat.rows;
  cv::Mat r_mat(center_mat.rows,center_mat.cols,CV_8UC4);
  /*uint8_t* dis_in;
  uint8_t* sobel_in;
  uint8_t* neg_in;
  uint8_t* r_out;
  cudaMalloc((void**)&dis_in,img_height*img_width*sizeof(uint8_t));
  cudaMalloc((void**)&sobel_in,img_height*img_width*sizeof(uint8_t));
  cudaMalloc((void**)&neg_in,img_height*img_width*sizeof(uint8_t));
  cudaMalloc((void*)&r_out,4*img_height*img_width*sizeof(uint8_t));*/
  return r_mat;

}
