//
// Created by wangyuanqi on 2019/8/21.
//

#ifndef SGM_STEREORECTIFYPARAMS_H
#define SGM_STEREORECTIFYPARAMS_H

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

const char* singleCalibrate_result_L[] = { "../calibrate/calibrationresults_right1.txt",
                                           "../calibrate/calibrationresults_right2.txt",
                                           "../calibrate/calibrationresults_center3.txt",
                                           "../calibrate/calibrationresults_center3.txt" }; //
const char* singleCalibrate_result_R[] = { "../calibrate/calibrationresults_center3.txt",
                                           "../calibrate/calibrationresults_center3.txt",
                                           "../calibrate/calibrationresults_right4.txt",
                                           "../calibrate/calibrationresults_right5.txt" }; //

const char* stereoRectifyParams[] = { "../calibrate/stereoRectifyParams13.txt",
                                      "../calibrate/stereoRectifyParams23.txt",
                                      "../calibrate/stereoRectifyParams34.txt",
                                      "../calibrate/stereoRectifyParams35.txt" }; //
Mat cameraMatrix_L = Mat(3, 3, CV_32FC1, Scalar::all(0)); //
Mat cameraMatrix_R = Mat(3, 3, CV_32FC1, Scalar::all(0)); //
Mat distCoeffs_L = Mat(1, 5, CV_32FC1, Scalar::all(0)); //
Mat distCoeffs_R = Mat(1, 5, CV_32FC1, Scalar::all(0)); //
Mat R, T, E, F; //
Mat R1, R2, P1, P2, Q; //
Mat mapl1, mapl2, mapr1, mapr2; //
Mat img1_rectified, img2_rectified; //
Size imageSize = Size(640, 512); //
Rect validRoi[2];

/*
��ȡ�궨���
*/
Mat getCameraMatrix(string path)
{
    Mat cameraMatrix(3, 3, CV_64FC1, Scalar::all(0));
    ifstream infile(path, ios::in);

    string str = "";
    while (str != "camera_matrix")
    {
        infile >> str;
    }

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            infile >> str;

            if (i == 0 && j == 0)
                str = str.substr(1);

            double num = atof(str.data());
            cameraMatrix.at<double>(i, j) = num;
        }
    }

    infile.close();
    return cameraMatrix;
}

Mat getDistCoeff(string path)
{
    Mat distCoeff(1, 5, CV_64FC1, Scalar::all(0));
    ifstream infile(path, ios::in);

    string str = "";
    while (str != "distortion_coefficients")
    {
        infile >> str;
    }

    for (int i = 0; i < 5; i++)
    {
        infile >> str;

        if (i == 0)
            str = str.substr(1);

        double num = atof(str.data());
        distCoeff.at<double>(i) = num;
    }

    infile.close();
    return distCoeff;
}

Mat getR1Matrix(string path)
{
    Mat R1Matrix(3, 3, CV_64FC1, Scalar::all(0));
    ifstream infile(path, ios::in);

    string str = "";
    while (str != "R1")
    {
        infile >> str;
    }

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            infile >> str;

            if (i == 0 && j == 0)
                str = str.substr(1);

            double num = atof(str.data());
            R1Matrix.at<double>(i, j) = num;
        }
    }

    infile.close();
    return R1Matrix;
}

Mat getR2Matrix(string path)
{
    Mat R2Matrix(3, 3, CV_64FC1, Scalar::all(0));
    ifstream infile(path, ios::in);

    string str = "";
    while (str != "R2")
    {
        infile >> str;
    }

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            infile >> str;

            if (i == 0 && j == 0)
                str = str.substr(1);

            double num = atof(str.data());
            R2Matrix.at<double>(i, j) = num;
        }
    }

    infile.close();
    return R2Matrix;
}

Mat getP1Matrix(string path)
{
    Mat P1Matrix(3, 4, CV_64FC1, Scalar::all(0));
    ifstream infile(path, ios::in);

    string str = "";
    while (str != "P1")
    {
        infile >> str;
    }

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            infile >> str;

            if (i == 0 && j == 0)
                str = str.substr(1);

            double num = atof(str.data());
            P1Matrix.at<double>(i, j) = num;
        }
    }

    infile.close();
    return P1Matrix;
}

Mat getP2Matrix(string path)
{
    Mat P2Matrix(3, 4, CV_64FC1, Scalar::all(0));
    ifstream infile(path, ios::in);

    string str = "";
    while (str != "P2")
    {
        infile >> str;
    }

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            infile >> str;

            if (i == 0 && j == 0)
                str = str.substr(1);

            double num = atof(str.data());
            P2Matrix.at<double>(i, j) = num;
        }
    }

    infile.close();
    return P2Matrix;
}

Mat getQMatrix(string path) {
    Mat QMatrix(4, 4, CV_64FC1, Scalar::all(0));
    ifstream infile(path, ios::in);

    string str = "";
    while (str != "Q") {
        infile >> str;
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            infile >> str;

            if (i == 0 && j == 0)
                str = str.substr(1);

            double num = atof(str.data());
            QMatrix.at<double>(i, j) = num;
        }
    }

    infile.close();
    return QMatrix;
}
#endif