#include "stereoRectified.h"
#include "stereoRectifyParams.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <string>
using namespace cv;
using namespace std;

void StereoRecitified::Preprocess(int pair_index) // 预处理：读取标定文件，计算重投影矩阵
{
    cameraMatrix_L = getCameraMatrix(singleCalibrate_result_L[pair_index]);
    cameraMatrix_R = getCameraMatrix(singleCalibrate_result_R[pair_index]);
    distCoeffs_L = getDistCoeff(singleCalibrate_result_L[pair_index]);
    distCoeffs_R = getDistCoeff(singleCalibrate_result_R[pair_index]);
    R1 = getR1Matrix(stereoRectifyParams[pair_index]);
    R2 = getR2Matrix(stereoRectifyParams[pair_index]);
    P1 = getP1Matrix(stereoRectifyParams[pair_index]);
    P2 = getP2Matrix(stereoRectifyParams[pair_index]);
    initUndistortRectifyMap(cameraMatrix_L, distCoeffs_L, R1, P1, imageSize, CV_32FC1, mapl1, mapl2);
    initUndistortRectifyMap(cameraMatrix_R, distCoeffs_R, R2, P2, imageSize, CV_32FC1, mapr1, mapr2);
    /*
    if (pair_index < 2)
    {
        flip(mapl1, mapl1, -1);
        flip(mapr1, mapr1, -1);
        flip(mapl2, mapl2, -1);
        flip(mapr2, mapr2, -1);
    }*/
};

void StereoRecitified::GetCorrespondCoordinate(int& origin_x, int& origin_y, int pair_index,
                                               float& correspond_x, float& correspond_y)
{
    correspond_x = mapr1.at<float>(origin_y, origin_x);
    correspond_y = mapr2.at<float>(origin_y, origin_x);
};

void StereoRecitified::ImageProcess(cv::Mat& img1_rectified, cv::Mat& img2_rectified,
                                    cv::Mat center_mat, cv::Mat right_img, int img_index)
{
    remap(center_mat, img1_rectified, mapl1, mapl2, INTER_LINEAR);
    remap(right_img, img2_rectified, mapr1, mapr2, INTER_LINEAR);
};

/*int main()
{
    StereoRecitified test;
    test.Preprocess(1);
    float correspond_x, correspond_y;
    int origin_x = 20;
    int origin_y = 20;
    test.GetCorrespondCoordinate(origin_x, origin_y, 1, correspond_x, correspond_y);
    cout << correspond_x << endl;
    cout << correspond_y << endl;
    cv::Mat img1, img2, img1_rectified, img2_rectified;
    img1 = imread("C:/Users/chen/source/repos/相机校准/testdata/0002.bmp");
    img2 = imread("C:/Users/chen/source/repos/相机校准/testdata/0003.bmp");
    // imshow("test", img2);
    test.ImageProcess(img1_rectified, img2_rectified, img1, img2, 1);
    // imshow("left", img1_rectified);
    // imshow("right", img2_rectified);
    // imwrite("left.png", img1_rectified);
    // imwrite("right.png", img2_rectified);
    waitKey(0);
    return 0;
}*/