/**
    This file is part of sgm. (https://github.com/dhernandez0/sgm).

    Copyright (c) 2016 Daniel Hernandez Juarez.

    sgm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    sgm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

**/

#include "disparity_method.h"

static cudaStream_t stream1, stream2, stream3;//, stream4, stream5, stream6, stream7, stream8;
static uint8_t *d_im0;
static uint8_t *d_im1;
static cost_t *d_transform0;
static cost_t *d_transform1;
static uint8_t *d_cost;
static uint8_t *d_disparity;
static uint8_t *d_disparity_filtered_uchar;
static uint8_t *h_disparity;
static uint16_t *d_S;
static uint8_t *d_L0;
static uint8_t *d_L1;
static uint8_t *d_L2;
static uint8_t *d_L3;
static uint8_t *d_L4;
static uint8_t *d_L5;
static uint8_t *d_L6;
static uint8_t *d_L7;
static uint8_t p1, p2;
static bool first_alloc;
static uint32_t cols, rows, size, size_cube_l;

//**********new static********************
static uint8_t* c_im;
static cost_t* center_transform,*l_center_transform,*r_center_transform,*t_center_transform,*b_center_transform;
static cudaStream_t n_stream1,n_stream2,n_stream3,n_stream4;
static uint8_t *t_d_im0;
static uint8_t *t_d_im1;
static uint8_t *l_im0[SELECT_IMAGE_NUM];
static uint8_t *r_im0[SELECT_IMAGE_NUM];
static uint8_t *t_im0[SELECT_IMAGE_NUM];
static uint8_t *b_im0[SELECT_IMAGE_NUM];

static cost_t *l_transform0[SELECT_IMAGE_NUM];
static cost_t *r_transform0[SELECT_IMAGE_NUM];
static cost_t *t_transform0[SELECT_IMAGE_NUM];
static cost_t *b_transform0[SELECT_IMAGE_NUM];

static uint32_t *new_d_cost;
static uint32_t *cal_new_d_cost;
static uint32_t *l_new_d_cost;
static uint32_t *r_new_d_cost;
static uint32_t *t_new_d_cost;
static uint32_t *b_new_d_cost;

static uint32_t *new_d_L0;
static uint32_t *new_d_L1;
static uint32_t *new_d_L2;
static uint32_t *new_d_L3;
static uint32_t *new_d_L4;
static uint32_t *new_d_L5;
static uint32_t *new_d_L6;
static uint32_t *new_d_L7;

//****************************************

void init_disparity_method(const uint8_t _p1, const uint8_t _p2) {
    // We are not using shared memory, use L1
    //CUDA_CHECK_RETURN(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    //CUDA_CHECK_RETURN(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

    // Create streams
    CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));
    CUDA_CHECK_RETURN(cudaStreamCreate(&stream2));
    CUDA_CHECK_RETURN(cudaStreamCreate(&stream3));
    first_alloc = true;
    p1 = _p1;
    p2 = _p2;
    rows = 0;
    cols = 0;
}
//get
cv::Mat compute_disparity_method(cv::Mat left, cv::Mat right, float *elapsed_time_ms, const char* directory, const char* fname) {
    if(cols != left.cols || rows != left.rows) {
        debug_log("WARNING: cols or rows are different");
        if(!first_alloc) {
            debug_log("Freeing memory");
            free_memory();
        }
        first_alloc = false;
        cols = left.cols;
        rows = left.rows;
        size = rows*cols;
        size_cube_l = size*MAX_DISPARITY;
        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_transform0, sizeof(cost_t)*size));

        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_transform1, sizeof(cost_t)*size));

        int size_cube = size*MAX_DISPARITY;
        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_cost, sizeof(uint8_t)*size_cube));

        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_im0, sizeof(uint8_t)*size));

        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_im1, sizeof(uint8_t)*size));

        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_S, sizeof(uint16_t)*size_cube_l));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L0, sizeof(uint8_t)*size_cube_l));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L1, sizeof(uint8_t)*size_cube_l));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L2, sizeof(uint8_t)*size_cube_l));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L3, sizeof(uint8_t)*size_cube_l));
#if PATH_AGGREGATION == 8
        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L4, sizeof(uint8_t)*size_cube_l));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L5, sizeof(uint8_t)*size_cube_l));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L6, sizeof(uint8_t)*size_cube_l));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L7, sizeof(uint8_t)*size_cube_l));
#endif

        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity, sizeof(uint8_t)*size));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity_filtered_uchar, sizeof(uint8_t)*size));
        h_disparity = new uint8_t[size];
    }
    debug_log("Copying images to the GPU");
    CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im0, left.ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im1, right.ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    dim3 block_size;
    block_size.x = 32;
    block_size.y = 32;

    dim3 grid_size;
    grid_size.x = (cols+block_size.x-1) / block_size.x;
    grid_size.y = (rows+block_size.y-1) / block_size.y;

    debug_log("Calling CSCT");
    CenterSymmetricCensusKernelSM2<<<grid_size, block_size, 0, stream1>>>(d_im0, d_im1, d_transform0, d_transform1, rows, cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
    //CenterSymmetricCensusKernelSM2<<<grid_size, block_size, 0, stream1>>>(d_im0, d_im1, d_transform0, d_transform1, rows, cols);

    // Hamming distance
    CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));
    debug_log("Calling Hamming Distance");
    HammingDistanceCostKernel<<<rows, MAX_DISPARITY/2, 0, stream1>>>(d_transform0, d_transform1, d_cost, rows, cols);
    HammingDistanceCostKernel_Z<<<rows, MAX_DISPARITY/2, 0, stream1>>>(d_transform0, d_transform1, d_cost, rows, cols);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }

    // Cost Aggregation
    const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE/WARP_SIZE;
    const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ/WARP_SIZE;

    debug_log("Calling Left to Right");
    CostAggregationKernelLeftToRight<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream2>>>(d_cost, d_L0, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
    debug_log("Calling Right to Left");
    CostAggregationKernelRightToLeft<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream3>>>(d_cost, d_L1, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
    debug_log("Calling Up to Down");
    CostAggregationKernelUpToDown<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L2, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    debug_log("Calling Down to Up");
    CostAggregationKernelDownToUp<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L3, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }

#if PATH_AGGREGATION == 8
    CostAggregationKernelDiagonalDownUpLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L4, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
    CostAggregationKernelDiagonalUpDownLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L5, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }

    CostAggregationKernelDiagonalDownUpRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L6, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
    CostAggregationKernelDiagonalUpDownRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L7, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
#endif
    debug_log("Calling Median Filter");
    MedianFilter3x3<<<(size+MAX_DISPARITY-1)/MAX_DISPARITY, MAX_DISPARITY, 0, stream1>>>(d_disparity, d_disparity_filtered_uchar, rows, cols);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }

    cudaEventRecord(stop, 0);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    cudaEventElapsedTime(elapsed_time_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    debug_log("Copying final disparity to CPU");
    CUDA_CHECK_RETURN(cudaMemcpy(h_disparity, d_disparity_filtered_uchar, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));

    cv::Mat disparity(rows, cols, CV_8UC1, h_disparity);
    return disparity;
}

static void free_memory() {
    CUDA_CHECK_RETURN(cudaFree(d_im0));
    CUDA_CHECK_RETURN(cudaFree(d_im1));
    CUDA_CHECK_RETURN(cudaFree(d_transform0));
    CUDA_CHECK_RETURN(cudaFree(d_transform1));
    CUDA_CHECK_RETURN(cudaFree(d_L0));
    CUDA_CHECK_RETURN(cudaFree(d_L1));
    CUDA_CHECK_RETURN(cudaFree(d_L2));
    CUDA_CHECK_RETURN(cudaFree(d_L3));
#if PATH_AGGREGATION == 8
    CUDA_CHECK_RETURN(cudaFree(d_L4));
    CUDA_CHECK_RETURN(cudaFree(d_L5));
    CUDA_CHECK_RETURN(cudaFree(d_L6));
    CUDA_CHECK_RETURN(cudaFree(d_L7));
#endif
    CUDA_CHECK_RETURN(cudaFree(d_disparity));
    CUDA_CHECK_RETURN(cudaFree(d_disparity_filtered_uchar));
    CUDA_CHECK_RETURN(cudaFree(d_cost));

    delete[] h_disparity;
}

void finish_disparity_method() {
    if(!first_alloc) {
        free_memory();
        CUDA_CHECK_RETURN(cudaStreamDestroy(stream1));
        CUDA_CHECK_RETURN(cudaStreamDestroy(stream2));
        CUDA_CHECK_RETURN(cudaStreamDestroy(stream3));
    }
}
//cv::Mat n_compute_disparity(vector<cv::Mat> l_image,vector<cv::Mat> r_image,cv::Mat center_image,vector<cv::Mat> t_image,vector<cv::Mat> b_image)
cv::Mat n_compute_disparity(vector<cv::Mat>& l_image,vector<cv::Mat>& r_image,cv::Mat center_image,vector<cv::Mat>& t_image,vector<cv::Mat>& b_image,
                            vector<cv::Mat>& top_left_image,vector<cv::Mat>& top_right_image,vector<cv::Mat>& bottom_left_image,vector<cv::Mat>& bottom_right_img,
                            cost_t *l_d_transform,cost_t *r_d_transform,cost_t *t_d_transform,cost_t *b_d_transform,
                            cost_t *top_left_d_transform,cost_t *top_right_d_transform,cost_t *bottom_left_d_transform,cost_t *bottom_right_d_transform,
                            uint8_t *l_pic,uint8_t *r_pic,uint8_t *t_pic,uint8_t *b_pic,
                            uint8_t *top_left_pic,uint8_t *top_right_pic,uint8_t *bottom_left_pic,uint8_t *bottom_right_pic)
{
    std::cout<<"cols:::"<<center_image.cols<<std::endl;
    std::cout<<"cols2:::"<<cols<<std::endl;
    int size_cube;

    cols=0,rows=0;
    //first_alloc=true;
    int image_count=0;

    if(cols!=center_image.cols&&rows!=center_image.rows)
    {
        std::cout<<"copy"<<std::endl;
        //for(;;);
        debug_log("WARNING: cols or rows are different");
        if(!first_alloc) {
            debug_log("Freeing memory");
            std::cout<<"Freeing memory"<<std::endl;
            free_memory();
        }
        first_alloc = false;
        cols = center_image.cols;
        rows = center_image.rows;
        size = rows*cols;
        size_cube_l = size*NEW_MAX_DISPARITY;
        //*******cuda memory malloc***********
        //*************clear device_vector memory
        CUDA_CHECK_RETURN(cudaMalloc((void **)&c_im, sizeof(uint8_t)*size));

        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_transform0, sizeof(cost_t)*size));

        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_transform1, sizeof(cost_t)*size));

        size_cube = size*NEW_MAX_DISPARITY;
        CUDA_CHECK_RETURN(cudaMalloc((void **)&new_d_cost, sizeof(uint32_t)*size_cube));
       /* CUDA_CHECK_RETURN(cudaMalloc((void **)&cal_new_d_cost, 4*IMAGE_NUMBER*sizeof(uint32_t)*size_cube));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&l_new_d_cost, sizeof(uint32_t)*size_cube));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&r_new_d_cost, sizeof(uint32_t)*size_cube));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&t_new_d_cost, sizeof(uint32_t)*size_cube));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&b_new_d_cost, sizeof(uint32_t)*size_cube));*/


        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_cost, sizeof(uint8_t)*size_cube));

        //malloc((void **)&cpu_cost, sizeof(uint32_t)*size_cube);

        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_im0, sizeof(uint8_t)*size));

        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_im1, sizeof(uint8_t)*size));

        CUDA_CHECK_RETURN(cudaMalloc((void **)&t_d_im0, sizeof(uint8_t)*size));

        CUDA_CHECK_RETURN(cudaMalloc((void **)&t_d_im1, sizeof(uint8_t)*size));

        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_S, sizeof(uint16_t)*size_cube_l));
        //CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L0, sizeof(uint8_t)*size_cube_l));
        //CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L1, sizeof(uint8_t)*size_cube_l));
       // CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L2, sizeof(uint8_t)*size_cube_l));
        //CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L3, sizeof(uint8_t)*size_cube_l));

        CUDA_CHECK_RETURN(cudaMalloc((void **)&new_d_L0, sizeof(uint32_t)*size_cube_l));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&new_d_L1, sizeof(uint32_t)*size_cube_l));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&new_d_L2, sizeof(uint32_t)*size_cube_l));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&new_d_L3, sizeof(uint32_t)*size_cube_l));

        //for(;;);
#if PATH_AGGREGATION == 8
        //CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L4, sizeof(uint8_t)*size_cube_l));
        //CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L5, sizeof(uint8_t)*size_cube_l));
        //CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L6, sizeof(uint8_t)*size_cube_l));
        //CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L7, sizeof(uint8_t)*size_cube_l));
        //************************
        CUDA_CHECK_RETURN(cudaMalloc((void **)&new_d_L4, sizeof(uint32_t)*size_cube_l));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&new_d_L5, sizeof(uint32_t)*size_cube_l));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&new_d_L6, sizeof(uint32_t)*size_cube_l));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&new_d_L7, sizeof(uint32_t)*size_cube_l));

#endif

        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity, sizeof(uint8_t)*size));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity_filtered_uchar, sizeof(uint8_t)*size));
        h_disparity = new uint8_t[size];

       /* for(image_count=0;image_count<IMAGE_NUMBER;image_count++)
        {
             CUDA_CHECK_RETURN(cudaMalloc((void **)&l_transform0[image_count], sizeof(cost_t)*size));
             CUDA_CHECK_RETURN(cudaMalloc((void **)&r_transform0[image_count], sizeof(cost_t)*size));
             CUDA_CHECK_RETURN(cudaMalloc((void **)&t_transform0[image_count], sizeof(cost_t)*size));
             CUDA_CHECK_RETURN(cudaMalloc((void **)&b_transform0[image_count], sizeof(cost_t)*size));

             CUDA_CHECK_RETURN(cudaMalloc((void **)&l_im0[image_count], sizeof(uint8_t)*size));
             CUDA_CHECK_RETURN(cudaMalloc((void **)&r_im0[image_count], sizeof(uint8_t)*size));
             CUDA_CHECK_RETURN(cudaMalloc((void **)&t_im0[image_count], sizeof(uint8_t)*size));
             CUDA_CHECK_RETURN(cudaMalloc((void **)&b_im0[image_count], sizeof(uint8_t)*size));

             CUDA_CHECK_RETURN(cudaMalloc((void **)&center_transform, sizeof(cost_t)*size));

        }*/
        CUDA_CHECK_RETURN(cudaMalloc((void **)&center_transform, sizeof(cost_t)*size));
        /*CUDA_CHECK_RETURN(cudaMalloc((void **)&l_center_transform, sizeof(cost_t)*size));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&r_center_transform, sizeof(cost_t)*size));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&t_center_transform, sizeof(cost_t)*size));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&b_center_transform, sizeof(cost_t)*size));*/


    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //cudaEventRecord(start, 0);

    dim3 block_size;
    block_size.x = 32;
    block_size.y = 32;

    dim3 grid_size;
    grid_size.x = (cols+block_size.x-1) / block_size.x;
    grid_size.y = (rows+block_size.y-1) / block_size.y;

    debug_log("Calling CSCT");
    //**********census transform calcute for each group***********
    //int image_count=0;
    CUDA_CHECK_RETURN(cudaStreamCreate(&n_stream1));
    CUDA_CHECK_RETURN(cudaStreamCreate(&n_stream2));
    CUDA_CHECK_RETURN(cudaStreamCreate(&n_stream3));
    CUDA_CHECK_RETURN(cudaStreamCreate(&n_stream4));
    /*for(image_count=0;image_count<IMAGE_NUMBER;image_count++)
    {
        //std::cout<<"begin"<<std::endl;

        CUDA_CHECK_RETURN(cudaMemcpyAsync(l_im0[image_count], l_image[image_count].ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, n_stream1));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(r_im0[image_count], r_image[image_count].ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, n_stream1));

        CenterSymmetricCensusKernelSM2<<<grid_size, block_size, 0, n_stream1>>>(l_im0[image_count], r_im0[image_count], l_transform0[image_count], r_transform0[image_count], rows, cols);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error: %s %d\n", cudaGetErrorString(err), err);
            exit(-1);
        }
        CUDA_CHECK_RETURN(cudaStreamSynchronize(n_stream1));

        CUDA_CHECK_RETURN(cudaMemcpyAsync(t_im0[image_count], t_image[image_count].ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, n_stream2));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(b_im0[image_count], b_image[image_count].ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, n_stream2));

        CenterSymmetricCensusKernelSM2<<<grid_size, block_size, 0, n_stream1>>>(t_im0[image_count], b_im0[image_count],t_transform0[image_count], b_transform0[image_count], rows, cols);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error: %s %d\n", cudaGetErrorString(err), err);
            exit(-1);
        }
        CUDA_CHECK_RETURN(cudaStreamSynchronize(n_stream1));
    }*/
    //std::cout<<"end"<<std::endl;
    CUDA_CHECK_RETURN(cudaMemcpyAsync(c_im, center_image.ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, n_stream1));
    //CUDA_CHECK_RETURN(cudaMalloc((void **)&center_transform, sizeof(cost_t)*size));
    CenterSymmetricCensusKernelSM2<<<grid_size, block_size, 0, n_stream1>>>(c_im, d_im0, center_transform, d_transform1, rows, cols);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(n_stream1));
    cost_t* census_data=new cost_t[size*SELECT_IMAGE_NUM*NEW_MAX_DISPARITY];
    /*CUDA_CHECK_RETURN(cudaMemcpy(census_data, r_d_transform, sizeof(cost_t)*size*IMAGE_NUMBER*NEW_MAX_DISPARITY, cudaMemcpyDeviceToHost));
    for(int s_c=0;s_c<size;s_c++)
       printf("%d\n",census_data[s_c]);*/
    //for(;;);
    //**********finish census transform**************************
    /*cost_t* census_data=new cost_t[size];
    cost_t* t_tr=b_transform0[0];
    cost_t* t_tr1=b_transform0[1];
    cost_t* t_tr2=b_transform0[2];
    cost_t* t_tr3=center_transform;
    CUDA_CHECK_RETURN(cudaMemcpy(census_data, t_tr, sizeof(cost_t)*size, cudaMemcpyDeviceToHost));
    std::cout<<"census_result::"<<census_data[200*512+200]<<std::endl;
    CUDA_CHECK_RETURN(cudaMemcpy(census_data, t_tr1, sizeof(cost_t)*size, cudaMemcpyDeviceToHost));
    std::cout<<"census_result::"<<census_data[200*512+200]<<std::endl;
    CUDA_CHECK_RETURN(cudaMemcpy(census_data, t_tr2, sizeof(cost_t)*size, cudaMemcpyDeviceToHost));
    std::cout<<"census_result::"<<census_data[200*512+200]<<std::endl;
    CUDA_CHECK_RETURN(cudaMemcpy(census_data, t_tr3, sizeof(cost_t)*size, cudaMemcpyDeviceToHost));
    std::cout<<"census_result::"<<census_data[200*512+200]<<std::endl;*/
    //printf("census_result::::::::::%d\n",(census_data));
    //CUDA_CHECK_RETURN(cudaMemcpy(cen_img_data, c_im, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));
    uint8_t* r_img_data=l_im0[0];
    uint8_t* cen_img_data=new uint8_t[size];
    //CUDA_CHECK_RETURN(cudaMemcpy(cen_img_data, r_img_data, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));
    //********census result check*******************************
    //********cost volume calculate*****************************
    //HammingDistanceCostKernel<<<rows, MAX_DISPARITY/2, 0, stream1>>>(d_transform0, d_transform1, d_cost, rows, cols);
   // std::cout<<"rows::"<<rows<<std::endl;
    cost_t z=2;
    /*N_HammingDistanceCostKernel<<<rows,NEW_MAX_DISPARITY/2, 0, n_stream1>>>(l_transform0[0],l_transform0[1],l_transform0[2],l_transform0[3],
            r_transform0[0],r_transform0[1],r_transform0[2],r_transform0[3],
            center_transform,
            t_transform0[0],t_transform0[1],t_transform0[2],t_transform0[3],
            b_transform0[0],b_transform0[1],b_transform0[2],b_transform0[3],
            d_cost,rows,cols);*/
    int image_number = 0;

    dim3 cost_grid_size;
  //  cost_grid_size.x = cols*16;
    cost_grid_size.x = cols*4;
    cost_grid_size.y = rows;

    dim3 cost_block_size;
    cost_block_size.x = SUM_IMAGE_NUM;
    //cost_block_size=0;
    cost_block_size.y = NEW_MAX_DISPARITY/4;
    cudaError_t err ;
    cudaEventRecord(start, 0);
    float *right_baseline_vec,*right_f_vec;
    float *left_baseline_vec, *left_f_vec;

    cudaMallocManaged((void**)&right_baseline_vec,2*sizeof(float));
    cudaMallocManaged((void**)&left_baseline_vec,2*sizeof(float));
    cudaMallocManaged((void**)&right_f_vec,2*sizeof(float));
    cudaMallocManaged((void**)&left_f_vec,2*sizeof(float));
    //cudaMallocManaged((void**)right_baseline_vec,2*sizeof(float));
    left_baseline_vec[1]= 13.699f;
    left_baseline_vec[0]= 7.203f;
    right_baseline_vec[0]= 6.821f;
    right_baseline_vec[1]= 13.724f;

    left_f_vec[0] = 845.92f;
    //left_f_vec[0] = 1680.77f;

    //right_f_vec[0] =


    W_N_HammingDistanceKernel<<<cost_grid_size,cost_block_size, 0, n_stream1>>>(center_transform,l_d_transform,r_d_transform,b_d_transform,t_d_transform,
                                                                                top_left_d_transform,top_right_d_transform,bottom_left_d_transform,bottom_right_d_transform,
    c_im,l_pic,r_pic,b_pic,t_pic,
    top_left_pic,top_right_pic,bottom_left_pic,bottom_right_pic,
    new_d_cost,
    right_f_vec,right_baseline_vec,
    left_f_vec,left_baseline_vec
    );
    /*CUDA_CHECK_RETURN(cudaMemcpyAsync(l_center_transform, center_transform, sizeof(cost_t)*size, cudaMemcpyDeviceToDevice, n_stream1));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(r_center_transform, center_transform, sizeof(cost_t)*size, cudaMemcpyDeviceToDevice, n_stream2));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(t_center_transform, center_transform, sizeof(cost_t)*size, cudaMemcpyDeviceToDevice, n_stream3));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(b_center_transform, center_transform, sizeof(cost_t)*size, cudaMemcpyDeviceToDevice, n_stream4));
    N_N_HammingDistanceKernel<<<cost_grid_size,cost_block_size, 0, n_stream1>>>(l_center_transform,l_d_transform,l_new_d_cost);
    N_N_HammingDistanceKernel<<<cost_grid_size,cost_block_size, 0, n_stream2>>>(r_center_transform,r_d_transform,r_new_d_cost);
    N_N_HammingDistanceKernel<<<cost_grid_size,cost_block_size, 0, n_stream3>>>(t_center_transform,b_d_transform,b_new_d_cost);
    N_N_HammingDistanceKernel<<<cost_grid_size,cost_block_size, 0, n_stream4>>>(b_center_transform,t_d_transform,t_new_d_cost);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    cost_grid_size.x = cols;
    cost_grid_size.y = rows;
    cudaEventRecord(stop, 0);
    SumCost<<<cost_grid_size,cost_block_size, 0, n_stream4>>>(l_new_d_cost,r_new_d_cost,b_new_d_cost,t_new_d_cost,new_d_cost);*/
    // CUDA_CHECK_RETURN(cudaStreamSynchronize(n_stream1));
    err = cudaGetLastError();
   // cudaEventRecord(stop, 0);
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
    // cudaEventRecord(stop, 0);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(n_stream1));
    /*for(image_number=0;image_number<IMAGE_NUMBER;image_number++)
    {
        N_HammingDistanceCostKernel1<<<cost_grid_size,NEW_MAX_DISPARITY/2, 0, n_stream1>>>(l_d_transform,r_d_transform,
                                                                                    center_transform,b_d_transform,t_d_transform,
                                                                                    new_d_cost,rows,cols,image_number+1,SUM_LENGTH*(image_number+1));
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error: %s %d\n", cudaGetErrorString(err), err);
            exit(-1);
        }
        CUDA_CHECK_RETURN(cudaStreamSynchronize(n_stream1));
    }*/
    //std::cout<<"d_cost_example:"<<new_d_cost[(200+512*200)*NEW_MAX_DISPARITY+90]<<std::endl;

   // cudaError_t err = cudaGetLastError();
    /*if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }*/
    //CUDA_CHECK_RETURN(cudaStreamSynchronize(n_stream1));
    /*uint32_t* cpu_cost=new uint32_t[size_cube];
    CUDA_CHECK_RETURN(cudaMemcpy(cpu_cost, new_d_cost, sizeof(uint32_t)*size_cube, cudaMemcpyDeviceToHost));
    uint32_t max_dis=0;
    int min_number,max_number;
    uint32_t min_dis=cpu_cost[0];
    int cost_count;
    int min_count;
    for(cost_count=0;cost_count<size_cube;cost_count++)
    {
        (cpu_cost[cost_count]>max_dis)?max_dis=cpu_cost[cost_count]:max_dis=max_dis;
        (cpu_cost[cost_count]<=min_dis)?min_dis=cpu_cost[cost_count],min_number=cost_count:min_dis=min_dis;
    }
    printf("min_dis::%u\n",min_dis);
    printf("max_dis::%u\n",max_dis);
    printf("min_number%d\n",min_number%NEW_MAX_DISPARITY);
    printf("cost_result::%u\n",cpu_cost[(200+200*512)*NEW_MAX_DISPARITY+90]);
    printf("cost_result::%u\n",cpu_cost[(200+100*512)*NEW_MAX_DISPARITY+0]);
    printf("cost_result::%u\n",cpu_cost[(200+300*512)*NEW_MAX_DISPARITY+90]);*/
    //********cost calculate end********************************
   // std::cout<<"huu:::"<<z<<std::endl;
    //********aggregation**************************************

    const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE/WARP_SIZE;//128/128
    const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ/WARP_SIZE;//128/128
   // std::cout<<"threads number::"<<COSTAGG_BLOCKSIZE_HORIZ<<std::endl;
    //std::cout<<"block number::"<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ<<std::endl;
    debug_log("Calling Left to Right");
    N_CostAggregationKernelLeftToRight<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, n_stream1>>>(new_d_cost, new_d_L0, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, new_d_L0, new_d_L1, new_d_L2, new_d_L3, new_d_L4, new_d_L5, new_d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
    debug_log("Calling Right to Left");
    N_CostAggregationKernelRightToLeft<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, n_stream1>>>(new_d_cost, new_d_L1, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, new_d_L0, new_d_L1, new_d_L2, new_d_L3, new_d_L4, new_d_L5, new_d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
    debug_log("Calling Up to Down");
    N_CostAggregationKernelUpToDown<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, n_stream1>>>(new_d_cost, new_d_L2, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, new_d_L0, new_d_L1, new_d_L2, new_d_L3, new_d_L4, new_d_L5, new_d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    debug_log("Calling Down to Up");
    std::cout<<COSTAGG_BLOCKSIZE<<std::endl;
   // for(;;);
    N_CostAggregationKernelDownToUp<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, n_stream1>>>(new_d_cost, new_d_L3, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, new_d_L0, new_d_L1, new_d_L2, new_d_L3, new_d_L4, new_d_L5, new_d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }

#if PATH_AGGREGATION == 8
    N_CostAggregationKernelDiagonalDownUpLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, n_stream1>>>(new_d_cost, new_d_L4, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, new_d_L0, new_d_L1, new_d_L2, new_d_L3, new_d_L4, new_d_L5, new_d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
    N_CostAggregationKernelDiagonalUpDownLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, n_stream1>>>(new_d_cost, new_d_L5, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, new_d_L0, new_d_L1, new_d_L2, new_d_L3, new_d_L4, new_d_L5, new_d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }

    N_CostAggregationKernelDiagonalDownUpRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, n_stream1>>>(new_d_cost, new_d_L6, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, new_d_L0, new_d_L1, new_d_L2, new_d_L3, new_d_L4, new_d_L5, new_d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
    N_CostAggregationKernelDiagonalUpDownRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, n_stream1>>>(new_d_cost, new_d_L7, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, new_d_L0, new_d_L1, new_d_L2, new_d_L3, new_d_L4, new_d_L5, new_d_L6);
    err = cudaGetLastError();
    //cudaEventRecord(stop, 0);
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
#endif
    debug_log("Calling Median Filter");
    //MedianFilter3x3<<<(size+NEW_MAX_DISPARITY-1)/NEW_MAX_DISPARITY, NEW_MAX_DISPARITY, 0, n_stream1>>>(d_disparity, d_disparity_filtered_uchar, rows, cols);
    MedianFilter3x3<<<(size+NEW_MAX_DISPARITY-1)/NEW_MAX_DISPARITY, NEW_MAX_DISPARITY, 0, n_stream1>>>(d_disparity, d_disparity_filtered_uchar, rows, cols);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }

    cudaEventRecord(stop, 0);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    float elapsed_mas;
    cudaEventElapsedTime(&elapsed_mas, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout<<"time::"<<elapsed_mas<<std::endl;
    debug_log("Copying final disparity to CPU");
    CUDA_CHECK_RETURN(cudaMemcpy(h_disparity, d_disparity_filtered_uchar, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));
   // CUDA_CHECK_RETURN(cudaMemcpy(h_disparity, d_disparity, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));

    //********aggregation end*************************************************
    cv::Mat n_disparity_image(rows,cols,CV_8UC1,h_disparity);
    std::cout<<"************************************************"<<std::endl;
    //imwrite("1.jpg",n_disparity_image);
    //for(;;);
    return n_disparity_image;
}

cv::Mat Image_Warp(cv::Mat& center_pic,
                   vector<cv::Mat>& l_image,
                   vector<cv::Mat>& r_image,
                   vector<cv::Mat>& t_image,
                   vector<cv::Mat>& b_image,
                   vector<cv::Mat>& top_left_image,
                   vector<cv::Mat>& top_right_image,
                   vector<cv::Mat>& bottom_left_image,
                   vector<cv::Mat>& bottom_right_image
                   )
{
    int c_size=IMG_HEIGHT*IMG_WIDTH;
    int w_size=NEW_MAX_DISPARITY*SELECT_IMAGE_NUM;
    cudaStream_t w_stream,w_stream1,w_stream2,w_stream3,w_stream4;
    std::cout<<center_pic.cols<<std::endl;
    uint8_t *c_pic;
    uint8_t *p_pic,*pc_pic=new uint8_t[c_size];
    uint8_t **l_c_pic,**r_c_pic,**t_c_pic,**b_c_pic;
    uint8_t **l_h_pic=(uint8_t **)malloc(sizeof(uint8_t*)*w_size);
    uint8_t **r_h_pic=(uint8_t **)malloc(sizeof(uint8_t*)*w_size);
    uint8_t **t_h_pic=(uint8_t **)malloc(sizeof(uint8_t*)*w_size);
    uint8_t **b_h_pic=(uint8_t **)malloc(sizeof(uint8_t*)*w_size);

    //left right top bottom image origin
    uint8_t **l_h_image=(uint8_t **)malloc(sizeof(uint8_t*)*SELECT_IMAGE_NUM);
    uint8_t **r_h_image=(uint8_t **)malloc(sizeof(uint8_t*)*SELECT_IMAGE_NUM);
    uint8_t **t_h_image=(uint8_t **)malloc(sizeof(uint8_t*)*SELECT_IMAGE_NUM);
    uint8_t **b_h_image=(uint8_t **)malloc(sizeof(uint8_t*)*SELECT_IMAGE_NUM);

    //top_left and top_right, bottom_left,bottom_right pic init

    uint8_t **top_left_h_pic=(uint8_t **)malloc(sizeof(uint8_t*)*w_size);
    uint8_t **top_right_h_pic=(uint8_t **)malloc(sizeof(uint8_t*)*w_size);
    uint8_t **bottom_left_h_pic=(uint8_t **)malloc(sizeof(uint8_t*)*w_size);
    uint8_t **bottom_right_h_pic=(uint8_t **)malloc(sizeof(uint8_t*)*w_size);

    uint8_t **top_left_h_image=(uint8_t **)malloc(sizeof(uint8_t*)*SELECT_IMAGE_NUM);
    uint8_t **top_right_h_image=(uint8_t **)malloc(sizeof(uint8_t*)*SELECT_IMAGE_NUM);
    uint8_t **bottom_left_h_image=(uint8_t **)malloc(sizeof(uint8_t*)*SELECT_IMAGE_NUM);
    uint8_t **bottom_right_h_image=(uint8_t **)malloc(sizeof(uint8_t*)*SELECT_IMAGE_NUM);

    uint8_t *top_left_orign_image,*top_right_orign_image,*bottom_right_orign_image,*bottom_left_orign_image;

    CUDA_CHECK_RETURN(cudaMalloc((void **)&top_left_orign_image, sizeof(uint8_t)*SELECT_IMAGE_NUM*c_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&top_right_orign_image, sizeof(uint8_t)*SELECT_IMAGE_NUM*c_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&bottom_right_orign_image, sizeof(uint8_t)*SELECT_IMAGE_NUM*c_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&bottom_left_orign_image, sizeof(uint8_t)*SELECT_IMAGE_NUM*c_size));
    //********************************************************

    uint8_t *l_orign_image,*r_orign_image,*t_orign_image,*b_orign_image;

    CUDA_CHECK_RETURN(cudaMalloc((void **)&l_orign_image, sizeof(uint8_t)*SELECT_IMAGE_NUM*c_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&r_orign_image, sizeof(uint8_t)*SELECT_IMAGE_NUM*c_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&t_orign_image, sizeof(uint8_t)*SELECT_IMAGE_NUM*c_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&b_orign_image, sizeof(uint8_t)*SELECT_IMAGE_NUM*c_size));
    std::cout<<top_left_image.size()<<" "<<top_right_image.size()<<" "
            <<bottom_left_image.size()<<" "<<bottom_left_image.size()<<std::endl;
    //std::cout<<top_left_orign_image.size(<<std::endl;

    for(int image_c=0;image_c<SELECT_IMAGE_NUM;image_c++)
    {
        CUDA_CHECK_RETURN(cudaMemcpy(l_orign_image+image_c*c_size, l_image[image_c].ptr<uint8_t>(), sizeof(uint8_t)*c_size, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(r_orign_image+image_c*c_size, r_image[image_c].ptr<uint8_t>(), sizeof(uint8_t)*c_size, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(t_orign_image+image_c*c_size, t_image[image_c].ptr<uint8_t>(), sizeof(uint8_t)*c_size, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(b_orign_image+image_c*c_size, b_image[image_c].ptr<uint8_t>(), sizeof(uint8_t)*c_size, cudaMemcpyHostToDevice));

        //top_left,top_right,bottom_left,bottom_right;
        CUDA_CHECK_RETURN(cudaMemcpy(top_left_orign_image+image_c*c_size, top_left_image[image_c].ptr<uint8_t>(), sizeof(uint8_t)*c_size, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(top_right_orign_image+image_c*c_size, top_right_image[image_c].ptr<uint8_t>(), sizeof(uint8_t)*c_size, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(bottom_left_orign_image+image_c*c_size, bottom_left_image[image_c].ptr<uint8_t>(), sizeof(uint8_t)*c_size, cudaMemcpyHostToDevice));
        std::cout<<"begin_1"<<std::endl;
        CUDA_CHECK_RETURN(cudaMemcpy(bottom_right_orign_image+image_c*c_size, bottom_right_image[image_c].ptr<uint8_t>(), sizeof(uint8_t)*c_size, cudaMemcpyHostToDevice));
        std::cout<<"begin"<<std::endl;
    }
    std::cout<<"begin"<<std::endl;

    for(int image_c=0;image_c<SELECT_IMAGE_NUM;image_c++)
    {
        l_h_image[image_c]=l_orign_image+image_c*c_size;
        r_h_image[image_c]=r_orign_image+image_c*c_size;
        t_h_image[image_c]=t_orign_image+image_c*c_size;
        b_h_image[image_c]=b_orign_image+image_c*c_size;

        //top_left,top_right,bottom_left,bottom_right
        top_left_h_image[image_c]=top_left_orign_image+image_c*c_size;
        top_right_h_image[image_c]=top_right_orign_image+image_c*c_size;
        bottom_left_h_image[image_c]=bottom_left_orign_image+image_c*c_size;
        bottom_right_h_image[image_c]=bottom_right_orign_image+image_c*c_size;
    }
    std::cout<<"begin1"<<std::endl;


    uint8_t **l_d_image,**r_d_image,**b_d_image,**t_d_image;

    CUDA_CHECK_RETURN(cudaMalloc((void**)&l_d_image, sizeof(uint8_t*)*SELECT_IMAGE_NUM));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&r_d_image, sizeof(uint8_t*)*SELECT_IMAGE_NUM));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&t_d_image, sizeof(uint8_t*)*SELECT_IMAGE_NUM));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&b_d_image, sizeof(uint8_t*)*SELECT_IMAGE_NUM));

    CUDA_CHECK_RETURN(cudaMemcpy(l_d_image, (void*)l_h_image, sizeof(uint8_t*)*SELECT_IMAGE_NUM, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(r_d_image, (void*)r_h_image, sizeof(uint8_t*)*SELECT_IMAGE_NUM, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(b_d_image, (void*)b_h_image, sizeof(uint8_t*)*SELECT_IMAGE_NUM, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(t_d_image, (void*)t_h_image, sizeof(uint8_t*)*SELECT_IMAGE_NUM, cudaMemcpyHostToDevice));

    //top_left top_right bottom_left bottom_right
    uint8_t **top_left_d_image,**top_right_d_image,**bottom_left_d_image,**bottom_right_d_image;

    CUDA_CHECK_RETURN(cudaMalloc((void**)&top_left_d_image, sizeof(uint8_t*)*SELECT_IMAGE_NUM));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&top_right_d_image, sizeof(uint8_t*)*SELECT_IMAGE_NUM));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&bottom_left_d_image, sizeof(uint8_t*)*SELECT_IMAGE_NUM));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&bottom_right_d_image, sizeof(uint8_t*)*SELECT_IMAGE_NUM));

    CUDA_CHECK_RETURN(cudaMemcpy(top_left_d_image, (void*)top_left_h_image, sizeof(uint8_t*)*SELECT_IMAGE_NUM, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(top_right_d_image, (void*)top_right_h_image, sizeof(uint8_t*)*SELECT_IMAGE_NUM, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(bottom_left_d_image, (void*)bottom_left_h_image, sizeof(uint8_t*)*SELECT_IMAGE_NUM, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(bottom_right_d_image, (void*)bottom_right_h_image, sizeof(uint8_t*)*SELECT_IMAGE_NUM, cudaMemcpyHostToDevice));
    //**********************************left copy end
    uint8_t *l_h_data,*r_h_data,*t_h_data,*b_h_data;
    CUDA_CHECK_RETURN(cudaStreamCreate(&w_stream));
    CUDA_CHECK_RETURN(cudaStreamCreate(&w_stream1));
    CUDA_CHECK_RETURN(cudaStreamCreate(&w_stream2));
    CUDA_CHECK_RETURN(cudaStreamCreate(&w_stream3));
    CUDA_CHECK_RETURN(cudaStreamCreate(&w_stream4));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&c_pic, sizeof(uint8_t)*c_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&p_pic, sizeof(uint8_t)*c_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&l_h_data, sizeof(uint8_t)*c_size*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&r_h_data, sizeof(uint8_t)*c_size*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&b_h_data, sizeof(uint8_t)*c_size*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&t_h_data, sizeof(uint8_t)*c_size*w_size));


    for(int w_c=0;w_c<w_size;w_c++)
    {
        l_h_pic[w_c]=l_h_data+c_size*w_c;
        r_h_pic[w_c]=r_h_data+c_size*w_c;
        t_h_pic[w_c]=t_h_data+c_size*w_c;
        b_h_pic[w_c]=b_h_data+c_size*w_c;

    }

    //CUDA_CHECK_RETURN(cudaMemcpy(h_disparity, d_disparity_filtered_uchar, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));
   // CUDA_CHECK_RETURN(cudaMemcpyAsync(c_pic, center_pic.ptr<uint8_t>(), sizeof(uint8_t)*c_size, cudaMemcpyHostToDevice, w_stream));
    CUDA_CHECK_RETURN(cudaMemcpy(c_pic, center_pic.ptr<uint8_t>(), sizeof(uint8_t)*c_size, cudaMemcpyHostToDevice));

    printf("def1\n");
    CUDA_CHECK_RETURN(cudaMalloc((void**)&l_c_pic, sizeof(uint8_t*)*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&r_c_pic, sizeof(uint8_t*)*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&t_c_pic, sizeof(uint8_t*)*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&b_c_pic, sizeof(uint8_t*)*w_size));

    CUDA_CHECK_RETURN(cudaMemcpy(l_c_pic, (void*)l_h_pic, sizeof(uint8_t*)*w_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(r_c_pic, (void*)r_h_pic, sizeof(uint8_t*)*w_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(b_c_pic, (void*)b_h_pic, sizeof(uint8_t*)*w_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(t_c_pic, (void*)t_h_pic, sizeof(uint8_t*)*w_size, cudaMemcpyHostToDevice));
    printf("%d\n",sizeof(l_c_pic)/sizeof(l_c_pic[0]));
    printf("def\n");

    //top_left top_right bottom_left bottom_right end
    uint8_t *top_left_h_data,*top_right_h_data,*bottom_left_h_data,*bottom_right_h_data;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&top_left_h_data, sizeof(uint8_t)*c_size*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&top_right_h_data, sizeof(uint8_t)*c_size*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&bottom_left_h_data, sizeof(uint8_t)*c_size*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&bottom_right_h_data, sizeof(uint8_t)*c_size*w_size));


    for(int w_c=0;w_c<w_size;w_c++)
    {
        top_left_h_pic[w_c]=top_left_h_data+c_size*w_c;
        top_right_h_pic[w_c]=top_right_h_data+c_size*w_c;
        bottom_left_h_pic[w_c]=bottom_left_h_data+c_size*w_c;
        bottom_right_h_pic[w_c]=bottom_right_h_data+c_size*w_c;
    }

    //CUDA_CHECK_RETURN(cudaMemcpy(h_disparity, d_disparity_filtered_uchar, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));
   // CUDA_CHECK_RETURN(cudaMemcpyAsync(c_pic, center_pic.ptr<uint8_t>(), sizeof(uint8_t)*c_size, cudaMemcpyHostToDevice, w_stream));
    printf("def1\n");
    uint8_t **top_left_c_pic,**top_right_c_pic,**bottom_left_c_pic,**bottom_right_c_pic;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&top_left_c_pic, sizeof(uint8_t*)*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&top_right_c_pic, sizeof(uint8_t*)*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&bottom_left_c_pic, sizeof(uint8_t*)*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&bottom_right_c_pic, sizeof(uint8_t*)*w_size));

    CUDA_CHECK_RETURN(cudaMemcpy(top_left_c_pic, (void*)top_left_h_pic, sizeof(uint8_t*)*w_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(top_right_c_pic, (void*)top_right_h_pic, sizeof(uint8_t*)*w_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(bottom_left_c_pic, (void*)bottom_left_h_pic, sizeof(uint8_t*)*w_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(bottom_right_c_pic, (void*)bottom_right_h_pic, sizeof(uint8_t*)*w_size, cudaMemcpyHostToDevice));
    //***********************************************

    printf("def\n");
    const int thread_index = 32;
    dim3 block_grid;
    block_grid.x=IMG_HEIGHT;
    //block_grid.y=NEW_MAX_DISPARITY*IMAGE_NUMBER;
    block_grid.y=w_size;

    /*block_grid.x = IMG_WIDTH*NEW_MAX_DISPARITY*SELECT_IMAGE_NUM/thread_index;
    block_grid.y = IMG_HEIGHT*NEW_MAX_DISPARITY*SELECT_IMAGE_NUM/thread_index;*/
    dim3 thread_grid;
    //thread_grid.x=IMG_WIDTH;
    thread_grid.x=IMG_WIDTH/2;

    //thread_grid.x = thread_index;
    //thread_grid.y = thread_index;
   // thread_grid.y=1;
    //thread_grid.y=2;
    //thread_grid.y=2;
    printf("def\n");
    cudaEvent_t m_start, m_stop;
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);

    N_ShiftImage<<<block_grid,thread_grid,0>>>(l_d_image,l_c_pic,
                                               r_d_image,r_c_pic,
                                               t_d_image,t_c_pic,
                                               b_d_image,b_c_pic,
                                               top_left_d_image,top_left_c_pic,
                                               top_right_d_image,top_right_c_pic,
                                               bottom_left_d_image,bottom_left_c_pic,
                                               bottom_right_d_image,bottom_right_c_pic);

    // CUDA_CHECK_RETURN(cudaStreamSynchronize(w_stream3));

     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) {
         printf("Error: %s %d\n", cudaGetErrorString(err), err);
         exit(-1);
     }
    // cudaEventRecord(m_stop, 0);
     CUDA_CHECK_RETURN(cudaDeviceSynchronize());

     //*********for tiao shi
     //CUDA_CHECK_RETURN(cudaMemcpy(r_h_data,r_c_pic[0], sizeof(uint8_t)*c_size, cudaMemcpyDeviceToHost));
     cv::Mat warp_mat(IMG_HEIGHT,IMG_WIDTH,CV_8UC1);
     uint8_t *right_host_data=(uint8_t*)malloc(sizeof(uint8_t)*c_size*w_size);
     CUDA_CHECK_RETURN(cudaMemcpy(right_host_data,r_h_data, sizeof(uint8_t)*c_size*w_size, cudaMemcpyDeviceToHost));
     for(int col_count=0;col_count<IMG_HEIGHT;col_count++)
     {
         uchar* warp_data = warp_mat.ptr<uchar>(col_count);
         for(int row_count=0;row_count<IMG_WIDTH;row_count++){
             //uint8_t r_data = *(right_host_data+1);
             //std::cout<<*(r_h_data+1)<<std::endl;
             warp_data[row_count]=right_host_data[0*c_size*w_size+10*c_size+row_count+col_count*IMG_WIDTH];
         }
     }
     cv::imwrite("warp.jpg",warp_mat);
     //**********
    //census
    cost_t **l_d_census,**r_d_census,**t_d_census,**b_d_census;
    cost_t **l_h_census=(cost_t **)malloc(sizeof(cost_t*)*w_size);
    cost_t **r_h_census=(cost_t **)malloc(sizeof(cost_t*)*w_size);
    cost_t **t_h_census=(cost_t **)malloc(sizeof(cost_t*)*w_size);
    cost_t **b_h_census=(cost_t **)malloc(sizeof(cost_t*)*w_size);
    cost_t *l_census_data,*r_census_data,*t_census_data,*b_census_data;

    CUDA_CHECK_RETURN(cudaMalloc((void **)&l_census_data, sizeof(cost_t)*c_size*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&r_census_data, sizeof(cost_t)*c_size*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&b_census_data, sizeof(cost_t)*c_size*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&t_census_data, sizeof(cost_t)*c_size*w_size));
    //top_left top_right bottom_left bottom_right
    cost_t **top_left_d_census,**top_right_d_census,**bottom_left_d_census,**bottom_right_d_census;
    cost_t **top_left_h_census=(cost_t **)malloc(sizeof(cost_t*)*w_size);
    cost_t **top_right_h_census=(cost_t **)malloc(sizeof(cost_t*)*w_size);
    cost_t **bottom_left_h_census=(cost_t **)malloc(sizeof(cost_t*)*w_size);
    cost_t **bottom_right_h_census=(cost_t **)malloc(sizeof(cost_t*)*w_size);
    cost_t *top_left_census_data,*top_right_census_data,*bottom_left_census_data,*bottom_right_census_data;

    CUDA_CHECK_RETURN(cudaMalloc((void **)&top_left_census_data, sizeof(cost_t)*c_size*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&top_right_census_data, sizeof(cost_t)*c_size*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&bottom_left_census_data, sizeof(cost_t)*c_size*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&bottom_right_census_data, sizeof(cost_t)*c_size*w_size));


    for(int w_c=0;w_c<w_size;w_c++)
    {
        top_left_h_census[w_c]=top_left_census_data+c_size*w_c;
        top_right_h_census[w_c]=top_right_census_data+c_size*w_c;
        bottom_left_h_census[w_c]=bottom_left_census_data+c_size*w_c;
        bottom_right_h_census[w_c]=bottom_right_census_data+c_size*w_c;
        l_h_census[w_c]=l_census_data+c_size*w_c;
        r_h_census[w_c]=r_census_data+c_size*w_c;
        t_h_census[w_c]=t_census_data+c_size*w_c;
        b_h_census[w_c]=b_census_data+c_size*w_c;
    }
    CUDA_CHECK_RETURN(cudaMalloc((void**)&l_d_census, sizeof(cost_t*)*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&r_d_census, sizeof(cost_t*)*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&t_d_census, sizeof(cost_t*)*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&b_d_census, sizeof(cost_t*)*w_size));

    CUDA_CHECK_RETURN(cudaMemcpy(l_d_census, (void*)l_h_census, sizeof(uint8_t*)*w_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(r_d_census, (void*)r_h_census, sizeof(uint8_t*)*w_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(b_d_census, (void*)b_h_census, sizeof(uint8_t*)*w_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(t_d_census, (void*)t_h_census, sizeof(uint8_t*)*w_size, cudaMemcpyHostToDevice));


    CUDA_CHECK_RETURN(cudaMalloc((void**)&top_left_d_census, sizeof(cost_t*)*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&top_right_d_census, sizeof(cost_t*)*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&bottom_left_d_census, sizeof(cost_t*)*w_size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&bottom_right_d_census, sizeof(cost_t*)*w_size));

     CUDA_CHECK_RETURN(cudaMemcpy(top_left_d_census, (void*)top_left_h_census, sizeof(uint8_t*)*w_size, cudaMemcpyHostToDevice));
     CUDA_CHECK_RETURN(cudaMemcpy(top_right_d_census, (void*)top_right_h_census, sizeof(uint8_t*)*w_size, cudaMemcpyHostToDevice));
     CUDA_CHECK_RETURN(cudaMemcpy(bottom_left_d_census, (void*)bottom_left_h_census, sizeof(uint8_t*)*w_size, cudaMemcpyHostToDevice));
     CUDA_CHECK_RETURN(cudaMemcpy(bottom_right_d_census, (void*)bottom_right_h_census, sizeof(uint8_t*)*w_size, cudaMemcpyHostToDevice));

     dim3 block_size;
     block_size.x = 32;
     block_size.y = 32;

     dim3 grid_size;
     grid_size.x = w_size*((IMG_WIDTH+block_size.x-1) / block_size.x);
     grid_size.y = (IMG_HEIGHT+block_size.y-1) / block_size.y;
     //("grid_size::%d::%d\n",grid_size.x,w_size);
     //for(;;);
     cudaEventRecord(m_start, 0);
    // for(int m_c=0;m_c<20;m_c++)
     {
     N_CenterSymmetricCnesusKernelSM2<<<grid_size,block_size,0,w_stream1>>>(l_c_pic,r_c_pic,l_d_census,r_d_census);
     CUDA_CHECK_RETURN(cudaStreamSynchronize(w_stream1));
     N_CenterSymmetricCnesusKernelSM2<<<grid_size,block_size,0,w_stream1>>>(t_c_pic,b_c_pic,t_d_census,b_d_census);
     CUDA_CHECK_RETURN(cudaStreamSynchronize(w_stream1));

     N_CenterSymmetricCnesusKernelSM2<<<grid_size,block_size,0,w_stream3>>>(top_left_c_pic,top_right_c_pic,top_left_d_census,top_right_d_census);
     CUDA_CHECK_RETURN(cudaStreamSynchronize(w_stream3));
     N_CenterSymmetricCnesusKernelSM2<<<grid_size,block_size,0,w_stream4>>>(bottom_left_c_pic,bottom_right_c_pic,bottom_left_d_census,bottom_right_d_census);
     CUDA_CHECK_RETURN(cudaStreamSynchronize(w_stream4));


     cudaEventRecord(m_stop, 0);
     }

     float elapsed_mas;
     cudaEventElapsedTime(&elapsed_mas, m_start, m_stop);
     cudaEventDestroy(m_start);
     cudaEventDestroy(m_stop);
     printf("%f\n",elapsed_mas);
     std::cout<<"time::"<<elapsed_mas<<std::endl;
     //for(;;);
     uint8_t *p_image,*p_image1;
     CUDA_CHECK_RETURN(cudaMalloc((void**)&p_image, sizeof(uint8_t)*c_size));
     CUDA_CHECK_RETURN(cudaMalloc((void**)&p_image1, sizeof(uint8_t)*c_size));

     CUDA_CHECK_RETURN(cudaMemcpy(p_image, l_h_data, sizeof(uint8_t)*c_size, cudaMemcpyDeviceToDevice));

     dim3 n_block_size;
     n_block_size.x=32;
     n_block_size.y=32;

     dim3 n_grid_size;
     n_grid_size.x=(IMG_WIDTH+block_size.x-1) / block_size.x;
     n_grid_size.y=(IMG_HEIGHT+block_size.y-1) / block_size.y;
     cost_t *c_transform,*c_transform_1;
     CUDA_CHECK_RETURN(cudaMalloc((void**)&c_transform, sizeof(cost_t)*c_size));
      CUDA_CHECK_RETURN(cudaMalloc((void**)&c_transform_1, sizeof(cost_t)*c_size));
    CenterSymmetricCensusKernelSM2<<<n_grid_size, n_block_size,0,w_stream>>>(p_image, p_image1, c_transform, c_transform_1, IMG_HEIGHT, IMG_WIDTH);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(w_stream));
    cost_t *hc_transform=new cost_t[c_size],*hc_lcensus=new cost_t[c_size];
   // CUDA_CHECK_RETURN(cudaMalloc((void**)&hc_transform, sizeof(cost_t)*c_size));
   // CUDA_CHECK_RETURN(cudaMalloc((void**)&hc_lcensus, sizeof(cost_t)*c_size));
    /*CUDA_CHECK_RETURN(cudaMemcpy(hc_transform, c_transform, sizeof(cost_t)*c_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(hc_lcensus, l_census_data, sizeof(cost_t)*c_size, cudaMemcpyDeviceToHost));*/

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    cv::Mat output_mat=n_compute_disparity(l_image,r_image,center_pic,t_image,b_image,
                                            top_left_image,top_right_image,bottom_left_image,bottom_right_image,
                                            l_census_data,r_census_data,t_census_data,b_census_data,
                                            top_left_census_data,top_right_census_data,bottom_left_census_data,bottom_right_census_data,
                                            l_h_data,r_h_data,t_h_data,b_h_data,
                                            top_left_h_data,top_right_h_data,bottom_left_h_data,bottom_right_h_data);
     return output_mat;



}
void Deal_Out(uint8_t *out_data)
{
    int c_size=IMG_HEIGHT*IMG_WIDTH;
    int w_size=NEW_MAX_DISPARITY*SELECT_IMAGE_NUM;
    uint8_t *l_s_image=new uint8_t[c_size*w_size];
    uint8_t *l_s_image2=new uint8_t[c_size*w_size];
    CUDA_CHECK_RETURN(cudaMemcpy(l_s_image, out_data, sizeof(uint8_t)*c_size*w_size, cudaMemcpyDeviceToHost));
    //CUDA_CHECK_RETURN(cudaMemcpy(l_s_image, out_data, sizeof(uint8_t)*c_size*IMAGE_NUMBER, cudaMemcpyDeviceToHost));
    cv::Mat s_pic(IMG_HEIGHT,IMG_WIDTH,CV_8UC1,l_s_image);


    //CUDA_CHECK_RETURN(cudaMemcpy(l_s_image, out_data, sizeof(uint8_t)*c_size*IMAGE_NUMBER, cudaMemcpyDeviceToHost));
    cv::Mat s_pic2(IMG_HEIGHT,IMG_WIDTH,CV_8UC1,l_s_image+2*(NEW_MAX_DISPARITY)*c_size);

   // CUDA_CHECK_RETURN(cudaMemcpy(l_s_image, out_data+3*c_size, sizeof(uint8_t)*c_size*w_size, cudaMemcpyDeviceToHost));
    //CUDA_CHECK_RETURN(cudaMemcpy(l_s_image, out_data, sizeof(uint8_t)*c_size*IMAGE_NUMBER, cudaMemcpyDeviceToHost));
    //cv::Mat s_pic1(IMG_HEIGHT,IMG_WIDTH,CV_8UC1,l_s_image);
    cv::imshow("shift",s_pic);
     cv::imshow("shift2",s_pic2);

    cv::waitKey(0);
}
;














