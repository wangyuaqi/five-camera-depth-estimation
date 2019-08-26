//
// Created by wangyuanqi on 19-8-21.
//

#ifndef SGM_DEPTH_COMPUTE_H
#define SGM_DEPTH_COMPUTE_H
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include "util.h"
#include "configuration.h"
#include "costs.h"
#include "hamming_cost.h"
#include "median_filter.h"
#include "cost_aggregation.h"
#include "debug.h"
//******new include*****
//#include<thrust/host_vector.h>
//#include<thrust/device_vector.h>
#include<vector>
#include"n_hamming_cost.h"
#include"new_cost_aggregation.h"
#include"new_median_filter.h"
#include "stereoRectified.h"

using namespace std;
using namespace cv;
class DepthComputeUtil{

public:

    DepthComputeUtil(uint8_t p1,uint8_t p2):p1(p1),p2(p2){};

    cv::Mat DepthCompute(pair<cv::Mat,cv::Mat>& left_left_image,CameraParam& left_left_param,
                      pair<cv::Mat,cv::Mat>& left_center_image,CameraParam& left_center_param,
                      pair<cv::Mat,cv::Mat>& right_center_image,CameraParam& right_center_param,
                      pair<cv::Mat,cv::Mat>& right_right_image,CameraParam& right_right_param,
                      float*& left_left_disparity,float*& left_center_disparity,
                      float*& right_center_disparity,float*& right_right_disparity,
                      StereoRecitified& stereoRecitified
                      );


private:
    int WriteFilePFM(const Mat &im, string path, float scalef);

    cv::Mat Process(cost_t **left_left_warp_census, cost_t *left_left_center_census,
                           cost_t **left_center_warp_census, cost_t *left_center_center_census,
                           cost_t **right_center_warp_census, cost_t *right_center_center_census,
                           cost_t **right_right_warp_census, cost_t *right_right_center_census,
                           int image_width,int image_height,
                           StereoRecitified& stereoRecitified,
                           float* left_left_disparity, float* left_center_disparity,
                           float* right_center_disparity, float* right_right_disparity
                           );

    uint8_t p1;
    uint8_t p2;

    cv::Mat s_center_mat;
    cv::Mat s_left_left_mat;


};
#endif //SGM_DEPTH_COMPUTE_H
