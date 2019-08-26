//
// Created by wangyuanqi on 2019/8/18.
//

#ifndef SGM_STEREORECTIFIED_H
#define SGM_STEREORECTIFIED_H
#include <opencv2/opencv.hpp>
using namespace cv;

class StereoRecitified{
public:
    /*
    */
    void Preprocess(int pair_index);

    /*
     * Input: per couple image and matrix
     * Output: after processed right_img
     */
    void ImageProcess(Mat& img1_rectified, Mat& img2_rectified,
                      cv::Mat center_mat, cv::Mat right_img, int img_index);

    /*
     * Input: origin_x and origin_y correspond to origin_img and
     * Output:
     */
    void GetCorrespondCoordinate(int& origin_x,int& origin_y, int pair_index,
                                 float& correspond_x, float& correspond_y);
private:
    std::vector<cv::Mat> center_img_map_;//for every pair img_map

};

#endif //SGM_STEREORECTIFIED_H

