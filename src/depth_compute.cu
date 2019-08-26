#include "depth_compute.h"
#include "configuration.h"
#include "image_warp.h"
#include "costs.h"
#include "new_cost_aggregation.h"
#include <stdint.h>
using namespace std;
cv::Mat DepthComputeUtil::DepthCompute(pair<cv::Mat, cv::Mat> &left_left_pair, CameraParam &left_left_param,
                                    pair<cv::Mat, cv::Mat> &left_center_pair, CameraParam &left_center_param,
                                    pair<cv::Mat, cv::Mat> &right_center_pair, CameraParam &right_center_param,
                                    pair<cv::Mat, cv::Mat> &right_right_pair, CameraParam &right_right_param,
                                    float*& left_left_disparity, float*& left_center_disparity,
                                    float*& right_center_disparity, float*& right_right_disparity,
                                    StereoRecitified& stereoRecitified) {


    uint8_t *left_left_data;
    uint8_t *left_left_center_data;
    uint8_t *left_center_data;
    uint8_t *left_center_center_data;
    uint8_t *right_center_data;
    uint8_t *right_center_center_data;
    uint8_t *right_right_data;
    uint8_t *right_right_center_data;

    uint8_t **left_left_warp_data;
    uint8_t **left_center_warp_data;
    uint8_t **right_center_warp_data;
    uint8_t **right_right_warp_data;

    const int image_width = (left_left_pair.first).cols;
    const int image_height = (left_left_pair.first).rows;

    const int origin_image_size = image_width*image_height*sizeof(uint8_t);
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&left_left_data,origin_image_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&left_left_center_data,origin_image_size));

    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&left_center_data,origin_image_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&left_center_center_data,origin_image_size));

    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&right_center_data,origin_image_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&right_center_center_data,origin_image_size));

    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&right_right_data,origin_image_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&right_right_center_data,origin_image_size));

    for(int image_x = 0;image_x<image_width;image_x++){
        for(int image_y = 0;image_y<image_height;image_y++){
            int image_index = image_y*image_width+image_x;
            left_left_data[image_index] = left_left_pair.second.at<uint8_t>(image_y,image_x);
            left_left_center_data[image_index] = left_left_pair.first.at<uint8_t>(image_y,image_x);

            left_center_data[image_index] = left_center_pair.second.at<uint8_t>(image_y,image_x);
            left_center_center_data[image_index] = left_center_pair.first.at<uint8_t>(image_y,image_x);

            right_center_data[image_index] = right_center_pair.second.at<uint8_t>(image_y,image_x);
            right_center_center_data[image_index] = right_center_pair.first.at<uint8_t>(image_y,image_x);

            right_right_data[image_index] = right_right_pair.second.at<uint8_t>(image_y,image_x);
            right_right_center_data[image_index] = right_right_pair.first.at<uint8_t>(image_y,image_x);
        }
    }
    cv::imwrite("4_base.png",right_right_pair.first);

    // allocate warped image memory
    std::cout<<"init"<<std::endl;
    const int dis_num = NEW_MAX_DISPARITY;

    const int warp_image_size = dis_num*sizeof(uint8_t*);
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&left_left_warp_data,warp_image_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&left_center_warp_data,warp_image_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&right_center_warp_data,warp_image_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&right_right_warp_data,warp_image_size));


    for(int dis_iter =0;dis_iter<dis_num;dis_iter++){
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&left_left_warp_data[dis_iter],origin_image_size));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&left_center_warp_data[dis_iter],origin_image_size));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&right_center_warp_data[dis_iter],origin_image_size));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&right_right_warp_data[dis_iter],origin_image_size));
    }
    cudaStream_t n_stream1,n_stream2,n_stream3,n_stream4;
    CUDA_CHECK_RETURN(cudaStreamCreate(&n_stream1));
    CUDA_CHECK_RETURN(cudaStreamCreate(&n_stream2));
    CUDA_CHECK_RETURN(cudaStreamCreate(&n_stream3));
    CUDA_CHECK_RETURN(cudaStreamCreate(&n_stream4));

    const int w_size=NEW_MAX_DISPARITY;
    dim3 block_grid;
    block_grid.x=IMG_HEIGHT;
    block_grid.y=w_size;


    dim3 thread_grid;
    //thread_grid.x=IMG_WIDTH;
    thread_grid.x=IMG_WIDTH/2;

    cudaStream_t w_stream1,w_stream2,w_stream3,w_stream4;
    CUDA_CHECK_RETURN(cudaStreamCreate(&w_stream1));
    CUDA_CHECK_RETURN(cudaStreamCreate(&w_stream2));
    CUDA_CHECK_RETURN(cudaStreamCreate(&w_stream3));
    CUDA_CHECK_RETURN(cudaStreamCreate(&w_stream4));

    cudaError_t err;
    Image_Shift<<<block_grid,thread_grid,0>>>(left_center_data,left_center_warp_data,left_center_disparity,-1,0);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }

    Image_Shift<<<block_grid,thread_grid,0>>>(left_left_data,left_left_warp_data,left_left_disparity,-1,0);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }


    Image_Shift<<<block_grid,thread_grid,0>>>(right_center_data,right_center_warp_data,right_center_disparity,1,0);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }

    Image_Shift<<<block_grid,thread_grid,0>>>(right_right_data,right_right_warp_data,right_right_disparity,1,0);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }

    CUDA_CHECK_RETURN(cudaStreamSynchronize(w_stream1));
    CUDA_CHECK_RETURN(cudaStreamSynchronize(w_stream2));
    CUDA_CHECK_RETURN(cudaStreamSynchronize(w_stream3));
    CUDA_CHECK_RETURN(cudaStreamSynchronize(w_stream4));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    //***********warp image output
    uint8_t *image_data;
    image_data = new uint8_t[image_height*image_width];
    for(int img_y = 0;img_y<image_height;img_y++){
        for(int img_x =0;img_x<image_width;img_x++){
            int img_index = img_x+img_y*image_width;
            image_data[img_index] = right_right_warp_data[115][img_index];
            //image_data[img_index] = left_left_center_data[img_index];
        }
    }
    cv::Mat warp_image(image_height,image_width,CV_8UC1,image_data);
    cv::imwrite("warp.png",warp_image);

    uint8_t *image_data_2;
    image_data_2 = new uint8_t[image_height*image_width];
    for(int img_y = 0;img_y<image_height;img_y++){
        for(int img_x =0;img_x<image_width;img_x++){
            int img_index = img_x+img_y*image_width;
            //image_data[img_index] = left_left_warp_data[120][img_index];
            image_data_2[img_index] = left_left_center_data[img_index];
        }
    }
    cv::Mat warp_image_2(image_height,image_width,CV_8UC1,image_data_2);
    cv::imwrite("warp_2.png",warp_image_2);

    //****************************

    // compute census transform
    cost_t **left_left_warp_census;
    cost_t *left_left_center_census;

    cost_t **left_center_warp_census;
    cost_t *left_center_center_census;

    cost_t **right_center_warp_census;
    cost_t *right_center_center_census;

    cost_t **right_right_warp_census;
    cost_t *right_right_center_census;



    //allocate warp census
    const int image_census_size = image_height*image_width*sizeof(cost_t);
    const int warp_census_size = dis_num*sizeof(cost_t*);

    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&left_left_warp_census,warp_census_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&left_center_warp_census,warp_census_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&right_center_warp_census,warp_census_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&right_right_warp_census,warp_census_size));


    for(int dis_iter =0;dis_iter<dis_num;dis_iter++)
    {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(left_left_warp_census[dis_iter]),image_census_size));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(left_center_warp_census[dis_iter]),image_census_size));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(right_center_warp_census[dis_iter]),image_census_size));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(right_right_warp_census[dis_iter]),image_census_size));
    }

    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&left_left_center_census,image_census_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&left_center_center_census,image_census_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&right_center_center_census,image_census_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&right_right_center_census,image_census_size));



    dim3 census_block_size;
    census_block_size.x = 32;
    census_block_size.y = 32;

    dim3 census_grid_size;
    census_grid_size.x = (NEW_MAX_DISPARITY)*((IMG_WIDTH+census_block_size.x-1) / census_block_size.x);
    census_grid_size.y = (IMG_HEIGHT+census_block_size.y-1) / census_block_size.y;

    N_CenterSymmetricCnesusKernelSM2<<<census_grid_size,census_block_size,0>>>(left_left_warp_data,left_center_warp_data,
                                                                           left_left_warp_census,left_center_warp_census);
    //CUDA_CHECK_RETURN(cudaStreamSynchronize(w_stream1));
    N_CenterSymmetricCnesusKernelSM2<<<census_grid_size,census_block_size,0>>>(right_center_warp_data,right_right_warp_data,
                                                                           right_center_warp_census,right_right_warp_census);

    /*N_CenterSymmetricCnesusKernelSM2<<<census_grid_size,census_block_size,0>>>(right_right_warp_data,right_center_warp_data,
            right_right_warp_census,right_center_warp_census);*/
    //CUDA_CHECK_RETURN(cudaStreamSynchronize(w_stream2));

    // Do center census
    dim3 n_block_size;
    n_block_size.x=32;
    n_block_size.y=32;

    dim3 n_grid_size;
    n_grid_size.x=(IMG_WIDTH+n_block_size.x-1) / n_block_size.x;
    n_grid_size.y=(IMG_HEIGHT+n_block_size.y-1) / n_block_size.y;
    CenterSymmetricCensusKernelSM2<<<n_grid_size, n_block_size,0>>>(left_left_center_data, left_center_center_data,
                                                                        left_left_center_census, left_center_center_census, IMG_HEIGHT, IMG_WIDTH);
    //CUDA_CHECK_RETURN(cudaStreamSynchronize(w_stream3));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    /*CenterSymmetricCensusKernelSM2<<<n_grid_size, n_block_size,0>>>(right_center_center_data, right_right_center_data,
            right_center_center_census, right_right_center_census, IMG_HEIGHT, IMG_WIDTH);*/
    CenterSymmetricCensusKernelSM2<<<n_grid_size, n_block_size,0>>>(right_right_center_data, right_center_center_data,
            right_right_center_census, right_center_center_census, IMG_HEIGHT, IMG_WIDTH);
    //CUDA_CHECK_RETURN(cudaStreamSynchronize(w_stream4));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cv::Mat result_img = Process(left_left_warp_census,left_left_center_census,
                                           left_center_warp_census,left_center_center_census,
                                           right_center_warp_census,right_center_center_census,
                                           right_right_warp_census,right_right_center_census,
                                           image_width,image_height,
                                           stereoRecitified,
                                           left_left_disparity,left_center_disparity,
                                           right_center_disparity,right_right_disparity
                                           );
    return result_img;

}

void TestCostCompute(cost_t **left_left_warp_census,cost_t *left_left_center_census,
                     float *left_left_disparity,
                     int image_width,int image_height,
                     uint32_t*& final_cost_volume
                     )
{
    int dis_num = NEW_MAX_DISPARITY;

    uint32_t max_cost =300;


    for(int image_y =0;image_y<image_height;image_y++){
        for(int image_x = 0;image_x<image_width;image_x++)
        {
            for(int dis_iter = 0;dis_iter<dis_num;dis_iter++) {
                int cost_index = (image_y * image_width + image_x) * dis_num + dis_iter;
                int image_index = (image_y * image_width) + image_x;
                if (image_x + left_left_disparity[dis_iter] >= image_width)
                    final_cost_volume[cost_index] = max_cost;
                else {
                    uint32_t answer = (left_left_center_census[image_index] ^
                                       (left_left_warp_census[dis_iter][image_index])); //Hamming Distance
                    uint32_t dist = 0;
                    while (answer) {
                        ++dist;
                        answer &= answer - 1;
                    }

                    final_cost_volume[cost_index] = dist;
                    //printf("%d::%d::%d\n",dis_iter,dist,final_cost_volume[cost_index]);

                }
            }
        }
    }

}
//for compute the cost volume that needed by cost aggregation
cv::Mat DepthComputeUtil::Process(cost_t **left_left_warp_census, cost_t *left_left_center_census,
                                         cost_t **left_center_warp_census, cost_t *left_center_center_census,
                                         cost_t **right_center_warp_census, cost_t *right_center_center_census,
                                         cost_t **right_right_warp_census, cost_t *right_right_center_census,
                                         int image_width,int image_height,
                                         StereoRecitified& stereoRecitified,
                                         float* left_left_disparity, float* left_center_disparity,
                                         float* right_center_disparity, float* right_right_disparity
                                         ){

    cudaStream_t n_stream1,n_stream2,n_stream3,n_stream4;
    CUDA_CHECK_RETURN(cudaStreamCreate(&n_stream1));
    CUDA_CHECK_RETURN(cudaStreamCreate(&n_stream2));
    CUDA_CHECK_RETURN(cudaStreamCreate(&n_stream3));
    CUDA_CHECK_RETURN(cudaStreamCreate(&n_stream4));

    uint32_t *final_cost_volume;
    int cost_size =NEW_MAX_DISPARITY*IMG_HEIGHT*IMG_WIDTH*sizeof(uint32_t);
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&final_cost_volume,cost_size));

    float *left_left_x,*left_left_y;
    float *left_center_x,*left_center_y;
    float *right_center_x,*right_center_y;
    float *right_right_x,*right_right_y;

    int map_size = image_width*image_height*sizeof(float);
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&left_left_x,map_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&left_left_y,map_size));

    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&left_center_x,map_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&left_center_y,map_size));

    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&right_center_x,map_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&right_center_y,map_size));

    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&right_right_x,map_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&right_right_y,map_size));

    StereoRecitified new_stereoRecitified;
    new_stereoRecitified.Preprocess(0);
    for(int img_x = 0;img_x<image_width;img_x++){
        for(int img_y = 0;img_y<image_height;img_y++){
            int img_index = img_x+img_y*image_width;
            new_stereoRecitified.GetCorrespondCoordinate(img_x,img_y,0,left_left_x[img_index],left_left_y[img_index]);
        }
    }
    new_stereoRecitified.Preprocess(1);
    for(int img_x = 0;img_x<image_width;img_x++){
        for(int img_y = 0;img_y<image_height;img_y++){
            int img_index = img_x+img_y*image_width;
            new_stereoRecitified.GetCorrespondCoordinate(img_x,img_y,1,left_center_x[img_index],left_center_y[img_index]);
        }
    }
    new_stereoRecitified.Preprocess(2);
    for(int img_x = 0;img_x<image_width;img_x++){
        for(int img_y = 0;img_y<image_height;img_y++){
            int img_index = img_x+img_y*image_width;
            new_stereoRecitified.GetCorrespondCoordinate(img_x,img_y,2,right_center_x[img_index],right_center_y[img_index]);
        }
    }
    new_stereoRecitified.Preprocess(3);

    for(int img_x = 0;img_x<image_width;img_x++){
        for(int img_y = 0;img_y<image_height;img_y++) {
            int img_index = img_x+img_y*image_width;
            new_stereoRecitified.GetCorrespondCoordinate(img_x,img_y,3,right_right_x[img_index],right_right_y[img_index]);
        }
    }
    for(int img_y = 0;img_y<image_height;img_y++){
        for (int img_x = 0; img_x < image_width; img_x++) {
            int img_index = img_x + img_y * image_width;

            //std::cout<<left_left_x[img_index]<<" "<<left_center_x[img_index]<<" "<<right_center_x[img_index]<<std::endl;
        }
    }
    //for(;;);
    /*for(int img_x = 0;img_x<image_width;img_x++){
        for(int img_y = 0;img_y<image_height;img_y++){
            int img_index = img_x+img_y*image_width;
            stereoRecitified.GetCorrespondCoordinate(img_x,img_y,0,left_left_x[img_index],left_left_y[img_index]);
            stereoRecitified.GetCorrespondCoordinate(img_x,img_y,1,left_center_x[img_index],left_center_y[img_index]);
            stereoRecitified.GetCorrespondCoordinate(img_x,img_y,2,right_center_x[img_index],right_center_y[img_index]);
            stereoRecitified.GetCorrespondCoordinate(img_x,img_y,3,right_right_x[img_index],right_right_y[img_index]);

        }
    }*/
    //TestCostCompute(left_left_warp_census,left_left_center_census,left_left_disparity,image_width,image_height,final_cost_volume);

    dim3 cost_grid_size;
    //  cost_grid_size.x = cols*16;
    cost_grid_size.x = image_width*4;
    cost_grid_size.y = image_height;

    dim3 cost_block_size;
    cost_block_size.x = 4;
    //cost_block_size=0;
    cost_block_size.y = NEW_MAX_DISPARITY/4;



    /*ComputeCostVolume<<<cost_grid_size, cost_block_size,0>>>(
            left_center_warp_census,left_center_center_census,
            left_left_warp_census,left_left_center_census,
            right_center_warp_census,right_center_center_census,
            right_right_warp_census,right_right_center_census,
            final_cost_volume,
            image_width,image_height,
            left_center_x,left_center_y,
            left_left_x,left_left_y,
            right_center_x,right_center_y,
            right_right_x,right_right_y,
            left_center_disparity,left_left_disparity,
            right_center_disparity,right_right_disparity
    );*/
    ComputeCostVolume<<<cost_grid_size, cost_block_size,0>>>(left_left_warp_census,left_left_center_census,
                                                          left_center_warp_census,left_center_center_census,
                                                          right_center_warp_census,right_center_center_census,
                                                          right_right_warp_census,right_right_center_census,
                                                          final_cost_volume,
                                                          image_width,image_height,
                                                          left_left_x,left_left_y,
                                                          left_center_x,left_center_y,
                                                          right_center_x,right_center_y,
                                                          right_right_x,right_right_y,
                                                          left_left_disparity,left_center_disparity,
                                                          right_center_disparity,right_right_disparity
                                                          );

    cudaError_t  err = cudaGetLastError();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // cudaEventRecord(stop, 0);
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    /*cv::Mat test_result_img(IMG_HEIGHT,IMG_WIDTH,CV_8UC1);
    std::cout<<"Process"<<std::endl;
    int sum_count = 0;
    for(int img_y =0;img_y<IMG_HEIGHT;img_y++){
        for(int img_x = 0;img_x<IMG_WIDTH;img_x++){
            test_result_img.at<uint8_t>(img_y,img_x) = 255;
            uint32_t max_cost = 1000;
            uint8_t r_index =200;
            int img_index = img_x+img_y*IMG_WIDTH;
            for(uint8_t dis_iter =0;dis_iter<NEW_MAX_DISPARITY;dis_iter++){
                if(final_cost_volume[img_index*NEW_MAX_DISPARITY+dis_iter]>300)
                    std::cout<<"Max_value::"<<final_cost_volume[img_index*NEW_MAX_DISPARITY+dis_iter]<<std::endl;
                if(final_cost_volume[img_index*NEW_MAX_DISPARITY+dis_iter]<max_cost) {
                    max_cost = final_cost_volume[img_index * NEW_MAX_DISPARITY + dis_iter];
                    r_index = dis_iter;

                }
            }
            //printf("%d::\n",r_index);

            sum_count++;
            //printf("%d\n",r_index);
            //if(r_index < 30)
            //    std::cout<<r_index<<" "<<max_cost<<" "<<left_left_x[img_index]<<" "<<left_left_y[img_index]<<std::endl;

            test_result_img.at<uint8_t>(img_y,img_x) = r_index;
        }
    }
    std::cout<<"sum count::"<<sum_count<<std::endl;
    return test_result_img;*/

    for(int m =500*NEW_MAX_DISPARITY;m<501*NEW_MAX_DISPARITY;m++)
        std::cout<<"Cost::"<<int(final_cost_volume[m])<<std::endl;

    // Cost Aggregation

    //cost Aggregation Allocate

    uint32_t *new_d_L0;
    uint32_t *new_d_L1;
    uint32_t *new_d_L2;
    uint32_t *new_d_L3;
    uint32_t *new_d_L4;
    uint32_t *new_d_L5;
    uint32_t *new_d_L6;
    uint32_t *new_d_L7;

    uint8_t *d_disparity;

    const int l_cost_size = image_width*image_height*NEW_MAX_DISPARITY;

    CUDA_CHECK_RETURN(cudaMallocManaged((void **)&new_d_L0, sizeof(uint32_t)*l_cost_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void **)&new_d_L1, sizeof(uint32_t)*l_cost_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void **)&new_d_L2, sizeof(uint32_t)*l_cost_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void **)&new_d_L3, sizeof(uint32_t)*l_cost_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void **)&new_d_L4, sizeof(uint32_t)*l_cost_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void **)&new_d_L5, sizeof(uint32_t)*l_cost_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void **)&new_d_L6, sizeof(uint32_t)*l_cost_size));
    CUDA_CHECK_RETURN(cudaMallocManaged((void **)&new_d_L7, sizeof(uint32_t)*l_cost_size));

    CUDA_CHECK_RETURN(cudaMallocManaged((void **)&d_disparity, sizeof(uint8_t)*image_height*image_width));




    const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE/WARP_SIZE;//128/128
    const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ/WARP_SIZE;//128/128
    // std::cout<<"threads number::"<<COSTAGG_BLOCKSIZE_HORIZ<<std::endl;
    //std::cout<<"block number::"<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ<<std::endl;
    debug_log("Calling Left to Right");
    const int rows = image_height;
    const int cols = image_width;

    const int size = image_height*image_width;
    const int size_cube_l = size*NEW_MAX_DISPARITY;
    N_CostAggregationKernelLeftToRight<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, n_stream1>>>(final_cost_volume, new_d_L0, p1, p2, rows, cols, left_left_center_census, left_center_center_census, d_disparity, new_d_L0, new_d_L1, new_d_L2, new_d_L3, new_d_L4, new_d_L5, new_d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
    debug_log("Calling Right to Left");
    N_CostAggregationKernelRightToLeft<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, n_stream1>>>(final_cost_volume, new_d_L1, p1, p2, rows, cols, left_left_center_census, left_center_center_census, d_disparity, new_d_L0, new_d_L1, new_d_L2, new_d_L3, new_d_L4, new_d_L5, new_d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
    debug_log("Calling Up to Down");
    N_CostAggregationKernelUpToDown<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, n_stream1>>>(final_cost_volume, new_d_L2, p1, p2, rows, cols, left_left_center_census, left_center_center_census, d_disparity, new_d_L0, new_d_L1, new_d_L2, new_d_L3, new_d_L4, new_d_L5, new_d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    debug_log("Calling Down to Up");
    std::cout<<COSTAGG_BLOCKSIZE<<std::endl;
    // for(;;);
    N_CostAggregationKernelDownToUp<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, n_stream1>>>(final_cost_volume, new_d_L3, p1, p2, rows, cols, left_left_center_census, left_center_center_census, d_disparity, new_d_L0, new_d_L1, new_d_L2, new_d_L3, new_d_L4, new_d_L5, new_d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }

#if PATH_AGGREGATION == 8
    N_CostAggregationKernelDiagonalDownUpLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, n_stream1>>>(final_cost_volume, new_d_L4, p1, p2, rows, cols, left_left_center_census, left_center_center_census, d_disparity, new_d_L0, new_d_L1, new_d_L2, new_d_L3, new_d_L4, new_d_L5, new_d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
    N_CostAggregationKernelDiagonalUpDownLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, n_stream1>>>(final_cost_volume, new_d_L5, p1, p2, rows, cols, left_left_center_census, left_center_center_census, d_disparity, new_d_L0, new_d_L1, new_d_L2, new_d_L3, new_d_L4, new_d_L5, new_d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }

    N_CostAggregationKernelDiagonalDownUpRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, n_stream1>>>(final_cost_volume, new_d_L6, p1, p2, rows, cols, left_left_center_census, left_center_center_census, d_disparity, new_d_L0, new_d_L1, new_d_L2, new_d_L3, new_d_L4, new_d_L5, new_d_L6);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
    N_CostAggregationKernelDiagonalUpDownRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, n_stream1>>>(final_cost_volume, new_d_L7, p1, p2, rows, cols, left_left_center_census, left_center_center_census, d_disparity, new_d_L0, new_d_L1, new_d_L2, new_d_L3, new_d_L4, new_d_L5, new_d_L6);
    err = cudaGetLastError();
    //cudaEventRecord(stop, 0);
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
        exit(-1);
    }
#endif
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    debug_log("Calling Median Filter");
    //MedianFilter3x3<<<(size+NEW_MAX_DISPARITY-1)/NEW_MAX_DISPARITY, NEW_MAX_DISPARITY, 0, n_stream1>>>(d_disparity, d_disparity_filtered_uchar, rows, cols);

    uint8_t *d_disparity_filtered_uchar;
    CUDA_CHECK_RETURN(cudaMallocManaged((void **)&d_disparity_filtered_uchar, sizeof(uint8_t)*image_width*image_height));

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
    //CUDA_CHECK_RETURN(cudaMemcpy(h_disparity, d_disparity_filtered_uchar, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));

    uint8_t *h_disparity;
    h_disparity = new uint8_t[image_height*image_width];
    cudaMemcpy(h_disparity, d_disparity_filtered_uchar, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_disparity, d_disparity, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost);
    // CUDA_CHECK_RETURN(cudaMemcpy(h_disparity, d_disparity, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));

    //********aggregation end*************************************************
    for(int m =0;m<image_height*image_width;m++)
        std::cout<<int(h_disparity[m])<<std::endl;
    cv::Mat n_disparity_image(rows,cols,CV_8UC1,h_disparity);
    std::cout<<"************************************************"<<std::endl;
    //imwrite("1.jpg",n_disparity_image);
    //for(;;);
    return n_disparity_image;
}