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

#ifndef DISPARITY_METHOD_H_
#define DISPARITY_METHOD_H_

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
using namespace std;
using namespace cv;
//using namespace thrust;
void init_disparity_method(const uint8_t _p1, const uint8_t _p2);
cv::Mat compute_disparity_method(cv::Mat left, cv::Mat right, float *elapsed_time_ms, const char* directory, const char* fname);

/*                    t_image
 *      l_image       center_image     r_image
 *                    b_image
 */
//cv::Mat n_compute_disparity(vector<cv::Mat> l_image,vector<cv::Mat> r_image,cv::Mat center_mat,vector<cv::Mat> t_image,vector<cv::Mat> b_image);
cv::Mat n_compute_disparity(vector<cv::Mat>& l_image,vector<cv::Mat>& r_image,cv::Mat center_mat,vector<cv::Mat>& t_image,vector<cv::Mat>& b_image,
                            vector<cv::Mat>& top_left_image, vector<cv::Mat>& top_right_image,
                            vector<cv::Mat>& bottom_left_image,vector<cv::Mat>& bottom_right_image,
                            cost_t *l_cost,cost_t *r_cost,cost_t *t_cost,cost_t *b_cost,cost_t *top_left_cost,cost_t *top_right_cost,
                            cost_t *bottom_left_cost, cost_t *bottom_right_cost,
                            uint8_t *l_pic,uint8_t *r_pic,uint8_t *t_img,uint8_t *b_img);
void finish_disparity_method();

cv::Mat Image_Warp(cv::Mat& center_pic,vector<cv::Mat>& l_image,vector<cv::Mat>& r_image,vector<cv::Mat>& t_image,vector<cv::Mat>& b_image,
                   vector<cv::Mat>& top_left_image,vector<cv::Mat>& top_right_image,vector<cv::Mat> & bottom_left_image,vector<cv::Mat>& bottom_right_image);
cv::Mat N_Image_Warp(cv::Mat& center_pic,vector<cv::Mat>& l_image,vector<cv::Mat>& r_image,vector<cv::Mat>& t_image,vector<cv::Mat>& b_image);

static void free_memory();

void Deal_Out(uint8_t *out_data);

int WriteFilePFM(const Mat &im, string path, float scalef);
#endif /* DISPARITY_METHOD_H_ */
