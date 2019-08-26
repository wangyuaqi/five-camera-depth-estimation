#include <iostream>
#include <numeric>
#include <sys/time.h>
#include <vector>
#include <stdlib.h>
#include <typeinfo>
#include <opencv2/opencv.hpp>

#include <numeric>
#include <stdlib.h>
#include <ctime>
#include <sys/types.h>
#include <stdint.h>
#include <linux/limits.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include "disparity_method.h"
#include "stereoRectified.h"
#include "configuration.h"
#include "depth_compute.h"
int main(int argc, char *argv[]){
    if(PATH_AGGREGATION != 4 && PATH_AGGREGATION != 8) {
        std::cerr << "Due to implementation limitations PATH_AGGREGATION must be 4 or 8" << std::endl;
        return -1;
    }
    //const char* directory = argv[1];
    const char* directory = "../input_data/";
    uint8_t p1, p2;
    //p1 = atoi(argv[2]);
    //p2 = atoi(argv[3]);

    p1 = 30;
    p2 = 100;

    //const char* img_dir=argv[1];
    const char* img_dir="../input_data/";
    char left_left_img_dir[PATH_MAX];
    char left_center_img_dir[PATH_MAX];
    char right_center_img_dir[PATH_MAX];
    char right_right_img_dir[PATH_MAX];
    char center_img_dir[PATH_MAX];
    sprintf(left_left_img_dir,"%s%s",img_dir,"1.bmp");
    sprintf(left_center_img_dir,"%s%s",img_dir,"2.bmp");
    sprintf(center_img_dir,"%s%s",img_dir,"3.bmp");
    sprintf(right_center_img_dir,"%s%s",img_dir,"4.bmp");
    sprintf(right_right_img_dir,"%s%s",img_dir,"5.bmp");
    std::cout<<left_left_img_dir<<endl;
    cv::Mat left_left_image = cv::imread(left_left_img_dir,0);
    cv::Mat left_center_image = cv::imread(left_center_img_dir,0);
    cv::Mat center_image = cv::imread(center_img_dir,0);
    cv::Mat right_center_image = cv::imread(right_center_img_dir,0);
    cv::Mat right_right_image = cv::imread(right_right_img_dir,0);
    if(left_left_image.empty()||left_center_image.empty()
      ||center_image.empty()||right_center_image.empty()||right_right_image.empty()) {
        std::cout<<"end"<<endl;
        return -1;
    }
    vector<cv::Mat> left_image={left_center_image,left_left_image};
    vector<cv::Mat> right_image={right_center_image,right_right_image};

    StereoRecitified stereoRecitified;
    std::cout<<"PreProcesss"<<std::endl;

    //stereoRecitified.Preprocess(0);
    //stereoRecitified.Preprocess(1);
    //stereoRecitified.Preprocess(2);
    //stereoRecitified.Preprocess(3);

    std::cout<<"PreProcesss"<<std::endl;
    std::pair<cv::Mat,cv::Mat> left_left_pair(cv::Mat(IMG_HEIGHT,IMG_WIDTH,CV_8UC1),cv::Mat(IMG_HEIGHT,IMG_WIDTH,CV_8UC1));
    std::pair<cv::Mat,cv::Mat> left_center_pair(cv::Mat(IMG_HEIGHT,IMG_WIDTH,CV_8UC1),cv::Mat(IMG_HEIGHT,IMG_WIDTH,CV_8UC1));
    std::pair<cv::Mat,cv::Mat> right_center_pair(cv::Mat(IMG_HEIGHT,IMG_WIDTH,CV_8UC1),cv::Mat(IMG_HEIGHT,IMG_WIDTH,CV_8UC1));
    std::pair<cv::Mat,cv::Mat> right_right_pair(cv::Mat(IMG_HEIGHT,IMG_WIDTH,CV_8UC1),cv::Mat(IMG_HEIGHT,IMG_WIDTH,CV_8UC1));

    //finish the stereoRectify and store in per pair

    std::cout<<left_left_pair.first.cols<<endl;

    cv::Mat n_center_image(IMG_HEIGHT,IMG_WIDTH,CV_8UC1);

    stereoRecitified.Preprocess(1);
    //stereoRecitified.ImageProcess(left_center_pair.first,left_center_pair.second,center_image,left_left_image,1);

    stereoRecitified.ImageProcess(left_center_pair.second,left_center_pair.first,left_center_image,center_image,1);
    cv::imwrite("2.png",left_center_pair.first);
    cv::imwrite("2_1.png",left_center_pair.second);

    stereoRecitified.Preprocess(0);
    //stereoRecitified.ImageProcess(left_left_pair.first,left_left_pair.second,center_image,left_left_image,0);

    stereoRecitified.ImageProcess(left_left_pair.second,left_left_pair.first,left_left_image,center_image,0);
    cv::imwrite("1.png",left_left_pair.first);
    cv::imwrite("1_1.png",left_left_pair.second);

    /*for(int img_y =0;img_y<IMG_HEIGHT;img_y++){
        for(int img_x =0;img_x<IMG_WIDTH;img_x++){

            float  n_x,n_y;
            stereoRecitified.GetCorrespondCoordinate(img_x,img_y,3,n_x,n_y);
            //std::cout<<n_x<<" "<<n_y<<std::endl;
            int m_x = (int)n_x, m_y = (int)n_y;
            if(m_x<IMG_WIDTH&&m_y<IMG_HEIGHT)
                n_center_image.at<uint8_t>(m_y,m_x) = center_image.at<uint8_t>(img_y,img_x);
        }
    }*/
    //for(;;);
    //cv::imwrite("test.png",n_center_image);

    stereoRecitified.Preprocess(2);
    stereoRecitified.ImageProcess(right_center_pair.first,right_center_pair.second,center_image,right_center_image,2);

    //stereoRecitified.ImageProcess(right_center_pair.second,right_center_pair.first,right_center_image,center_image,2);


    cv::imwrite("3.png",right_center_pair.first);
    cv::imwrite("3_1.png",right_center_pair.second);

    stereoRecitified.Preprocess(3);

    stereoRecitified.ImageProcess(right_right_pair.first,right_right_pair.second,center_image,right_right_image,3);
    //stereoRecitified.ImageProcess(right_right_pair.second,right_right_pair.first,right_right_image,center_image,3);
    cv::imwrite("4.png",right_right_pair.first);
    cv::imwrite("4_1.png",right_right_pair.second);
    std::cout<<"end"<<endl;
    float *left_left_disparity,*left_center_disparity,*right_center_disparity,*right_right_disparity;

    const int depth_num = NEW_MAX_DISPARITY;
    cudaMallocManaged((void**)&left_left_disparity,depth_num*sizeof(float));
    cudaMallocManaged((void**)&left_center_disparity,depth_num*sizeof(float));
    cudaMallocManaged((void**)&right_center_disparity,depth_num*sizeof(float));
    cudaMallocManaged((void**)&right_right_disparity,depth_num*sizeof(float));

    CameraParam left_left_param;
    CameraParam left_center_param;
    CameraParam right_center_param;
    CameraParam right_right_param;

    left_left_param.baseline_param = 13.699f;
    left_center_param.baseline_param = 7.203f;
    right_center_param.baseline_param = 6.821f;
    right_right_param.baseline_param = 13.724f;

    left_left_param.f_param =  840.18f;
    left_center_param.f_param =  846.07f;
    right_center_param.f_param =  844.86f;
    right_right_param.f_param =  844.86f;

    const float depth_max = END_DEPTH;
    const float depth_min = BEGIN_DEPTH;
    const float depth_dis = (depth_max-depth_min)/depth_num;
    float cur_depth;
    const float disp_min = 10.0f;
    const float disp_max = 200;
    const float disp_dis = (disp_max-disp_min)/NEW_MAX_DISPARITY;
    for(int depth_iter = 0;depth_iter<depth_num;depth_iter++) {
        //cur_depth = depth_min + depth_iter * depth_dis;
        //left_left_disparity[depth_iter] = depth_min+disp_dis*depth_iter;
        left_center_disparity[depth_iter] = depth_min+disp_dis*depth_iter;
    }
    for(int depth_iter = 0;depth_iter<depth_num;depth_iter++){
        cur_depth = depth_min+depth_iter*depth_dis;
        //left_left_disparity[depth_iter] = depth_iter+80;
        //cur_depth =  (left_left_param.f_param*left_left_param.baseline_param)/left_left_disparity[depth_iter];
        //cur_depth =  (left_center_param.f_param*left_center_param.baseline_param)/left_center_disparity[depth_iter];
        left_left_disparity[depth_iter] = (left_left_param.f_param*left_left_param.baseline_param)/cur_depth;
        left_center_disparity[depth_iter] = (left_center_param.f_param*left_center_param.baseline_param)/cur_depth;
        right_center_disparity[depth_iter] = (right_center_param.f_param*right_center_param.baseline_param)/cur_depth;
        right_right_disparity[depth_iter] = (right_right_param.f_param*right_right_param.baseline_param)/cur_depth;
        //right_right_disparity[depth_iter] = depth_iter+80;

        //std::cout<<"dis::"<<left_left_disparity[depth_iter]<<std::endl;

    }
    std::cout<<"dis::"<<left_left_disparity[0]<<std::endl;
    std::cout<<"compute"<<endl;

    //for(;;);
    /*test_2*/

    /*cv::Mat test_image_2(IMG_HEIGHT,IMG_WIDTH,CV_8UC1);
    StereoRecitified new_stereoRecitified;
    new_stereoRecitified.Preprocess(0);
    for(int img_x = 0;img_x<image_width;img_x++){
        for(int img_y = 0;img_y<image_height;img_y++) {
            int img_index = img_x+img_y*image_width;
            float  n_x,n_y;
            stereoRecitified.GetCorrespondCoordinate(img_x,img_y,3,n_x,n_y);
            //std::cout<<n_x<<" "<<n_y<<std::endl;
            int m_x = (int)n_x, m_y = (int)n_y;
            if(m_x<IMG_WIDTH&&m_y<IMG_HEIGHT)
                test_image_2.at<uint8_t>(m_y,m_x) = center_image.at<uint8_t>(img_y,img_x);
        }
    }
    cv::imwrite("test_2.png",test_image_2);*/
    //*******
    DepthComputeUtil depthComputeUtil(p1,p2);

    cv::Mat result_mat = depthComputeUtil.DepthCompute(left_left_pair,left_left_param,
                                 left_center_pair,left_center_param,
                                 right_center_pair,right_center_param,
                                 right_right_pair,right_right_param,
                                 left_left_disparity,left_center_disparity,
                                 right_center_disparity,right_right_disparity,
                                 stereoRecitified);

    std::cout<<"compute_2"<<endl;
    //cv::imwrite("result_0.png",result_mat);
    //cv::imwrite("result_1.png",result_mat);
    cv::imwrite("result_0_1_2.png",result_mat);


}

/*void OutProcess(cv::Mat& origin_img,cv::Mat& out_img, int img_width,int img_height){
    int img_x = 0,img_y = 0;
    for(img_y =0 ;img_y<image_height;img_y++){
        for(int )
    }

}*/