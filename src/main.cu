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
#include "include/disparity_method.h"
#include "include/patch_match.h"
//#include "ImageTool.h"

using namespace cv;
using namespace std;
bool directory_exists(const char* dir) {
	DIR* d = opendir(dir);
	bool ok = false;
	if(d) {
	    closedir(d);
	    ok = true;
	}
	return ok;
}

void disparity_errors(cv::Mat estimation, const char* gt_file, int *n, int *n_err) {
	int nlocal = 0;
	int nerrlocal = 0;

	cv::Mat gt_image = cv::imread(gt_file, cv::IMREAD_UNCHANGED);
	if(!gt_image.data) {
		std::cerr << "Couldn't read the file " << gt_file << std::endl;
		exit(EXIT_FAILURE);
	}
	if(estimation.rows != gt_image.rows || estimation.cols != gt_image.cols) {
		std::cerr << "Ground truth must have the same dimesions" << std::endl;
		exit(EXIT_FAILURE);
	}
	const int type = estimation.type();
	const uchar depth = type & CV_MAT_DEPTH_MASK;
	for(int i = 0; i < gt_image.rows; i++) {
		for(int j = 0; j < gt_image.cols; j++) {
			const uint16_t gt = gt_image.at<uint16_t>(i, j);
			if(gt > 0) {
				const float gt_f = ((float)gt)/256.0f;
				float est;
				if(depth == CV_8U) {
					est = (float) estimation.at<uint8_t>(i, j);
				} else {
					est = estimation.at<float>(i, j);
				}
				const float err = fabsf(est-gt_f);
				const float ratio = err/fabsf(gt_f);
				if(err > ABS_THRESH && ratio > REL_THRESH) {
					nerrlocal++;
				}
				nlocal++;
			}
		}
	}
	*n += nlocal;
	*n_err += nerrlocal;
}

bool check_directories_exist(const char* directory, const char* left_dir, const char* right_dir, const char* disparity_dir) {
	char left_dir_sub[PATH_MAX];
	char right_dir_sub[PATH_MAX];
	char disparity_dir_sub[PATH_MAX];
	sprintf(left_dir_sub, "%s/%s", directory, left_dir);
	sprintf(right_dir_sub, "%s/%s", directory, right_dir);
	sprintf(disparity_dir_sub, "%s/%s", directory, disparity_dir);

	return directory_exists(left_dir_sub) && directory_exists(right_dir_sub) && directory_exists(disparity_dir_sub);
}

void LRCheckOut(cv::Mat& left_img,cv::Mat& right_image,cv::Mat_<float>& disp_mat,cv::Mat& dis_result){

    int src_rows = left_img.rows;
    int src_cols = left_img.cols;

    for(int col = 0;col<src_cols;col++) //x
    {
        for(int row =0; row<src_rows;row++){//y
            uchar src_color = left_img.at<uchar>(row,col);

            int right_col = col-disp_mat(row,col);
            if(right_col<0){
                dis_result.at<uchar>(row,col)=0;

            }else{
               uchar right_color = right_image.at<uchar>(row,right_col);
                int sum_dif =0;
                sum_dif+=abs(right_color-src_color);
                //std::cout<<sum_dif<<std::endl;
                if(sum_dif>200)
                    dis_result.at<uchar>(row,col)=0;
            }

        }
    }
}

/*void DepthDeal(cv::Mat_<float>& dis_mat,cv::Mat_<float>& depth_mat){
    const float f_dis = 1692.5;
    const float b_dis = 6.8;

    PointCloud::Ptr cloud(new PointCloud);

    for(int img_width=0; img_width<IMG_WIDTH; img_width++){
        for(int img_height=0; img_height<IMG_HEIGHT; img_height++){
            PointT p;
            p.x = img_width;
            p.y = img_height;
            float depth = f_dis*b_dis/dis_mat(img_height,img_width);
            depth_mat(img_height,img_width) = depth;

            p.y = depth;
            cloud->points.push_back(p);

        }
    }
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;
    pcl::io::savePCDFile("boxes/222.pcd", *cloud);
    cloud->points.clear();
}*/

int main(int argc, char *argv[]) {

    /*if(MAX_DISPARITY != 128) {
		std::cerr << "Due to implementation limitations MAX_DISPARITY must be 128" << std::endl;
		return -1;
    }*/
	if(PATH_AGGREGATION != 4 && PATH_AGGREGATION != 8) {
                std::cerr << "Due to implementation limitations PATH_AGGREGATION must be 4 or 8" << std::endl;
                return -1;
        }
	const char* directory = argv[1];
	uint8_t p1, p2;
	p1 = atoi(argv[2]);
	p2 = atoi(argv[3]);

    const char* img_dir=argv[4];
    char left_img_dir[PATH_MAX];
    char right_img_dir[PATH_MAX];
    char top_img_dir[PATH_MAX];
    char bottom_img_dir[PATH_MAX];
    char center_img_dir[PATH_MAX];
    sprintf(left_img_dir,"%s%s",img_dir,"input_Cam036.png");
    sprintf(right_img_dir,"%s%s",img_dir,"input_Cam044.png");
    sprintf(top_img_dir,"%s%s",img_dir,"input_Cam004.png");
    sprintf(bottom_img_dir,"%s%s",img_dir,"input_Cam076.png");
    sprintf(center_img_dir,"%s%s",img_dir,"input_Cam040.png");

	DIR *dp;
	struct dirent *ep;

	// Directories
	const char* left_dir = "left";
	const char* disparity_dir = "disparities";
	const char* right_dir = "right";
	const char* gt_dir = "gt";


    /*if(!check_directories_exist(directory, left_dir, right_dir, disparity_dir)) {
		std::cerr << "We need <left>, <right> and <disparities> directories" << std::endl;
		exit(EXIT_FAILURE);
    }*/

    std::vector<float> times;

	init_disparity_method(p1, p2);
    {
		// Skip directories

//***************load images begin********************************


        //cv::Mat h_im0 = cv::imread(left_file);
        //******


        //******
//*******************load images end**************
//#if LOG
        //std::cout << "processing: " << left_file << std::endl;
        // ************Compute************
        float elapsed_time_ms;
        float t_elapsed_time_ms;
        //t_disparity=compute_disparity_method(top_rotate_img,bottom_rotate_img,&t_elapsed_time_ms,directory,ep->d_name).clone();


        //***********shift to center omage*******

        //*********compute final disparity map ********
        //final_disparity_im=(c_t_diparity_im+c_l_disparity_im)/2;

       /* Mat final_negative_mat(s_img_height,s_img_width,CV_8UC1,cv::Scalar(0));
        unsigned char* final_n_data=final_negative_mat.data;
        int n_count=0;
        for(c_height=0;c_height<s_img_height;c_height++)
        {
            for(c_width=0;c_width<s_img_width;c_width++)
            {
             if(abs(c_t_dis_data[c_width+c_height*s_img_width]-c_l_dis_data[c_width+c_height*s_img_width])>1)
             {
                 n_count++;
                 final_n_data[c_width+c_height*s_img_width]=255;
             }
            }
        }
        std::cout<<"negative mount(top_dis!=left_dis)::"<<n_count<<std::endl;*/
        //******************sgm end**************************
        //******************sobel center image***************
        //clock_t begin_time,end_time;
        //begin_time=clock();
       /* Mat sobel_mat=Sobel_Deal(center_img);
        unsigned char* sobel_data=sobel_mat.data;
        int sobel_width,sobel_height;
        int sobel_count=0;
        int sobel_disparity;
        int sum_false_count=0;
        for(sobel_height=0;sobel_height<s_img_height;sobel_height++)
        {
            for(sobel_width=0;sobel_width<s_img_width;sobel_width++)
            {
               sobel_disparity=final_dis_data[sobel_width+sobel_height*s_img_width];
               sobel_disparity-=32;
               if(sobel_data[sobel_width+sobel_height*sobel_mat.cols]>10)
               {

                  if(sobel_width+sobel_disparity>=0&&sobel_width+sobel_disparity<s_img_width)
                  {
                      if(abs(center_im_data[sobel_width+sobel_disparity+sobel_height*s_img_width]-right_m_data[sobel_width+sobel_height*s_img_width])>1)
                          sobel_count++;
                      else
                          sobel_data[sobel_width+sobel_height*sobel_mat.cols]=0;
                  }

               }
               else
                  sobel_data[sobel_width+sobel_height*sobel_mat.cols]=0;
               if(abs(center_im_data[sobel_width+sobel_disparity+sobel_height*s_img_width]-right_m_data[sobel_width+sobel_height*s_img_width])>1)
                   sum_false_count++;
              // else
                   //sobel_data[sobel_width+sobel_height*sobel_mat.cols]=0;
            }
        }
        std::cout<<"false dis point number::"<<sum_false_count<<std::endl;
        std::cout<<"sobel point number::"<<sobel_count<<std::endl;*/
        //std::cout<<"soble_time"<<(clock()-begin_time)*1.0/CLOCKS_PER_SEC*1000<<std::endl;
        //******************end 0.1ms******************************
        //******************base final disparity and slobel poc **************************

        char r_im_dir[PATH_MAX];
        char c_im_dir[PATH_MAX];
        vector<cv::Mat> row_l_img;//img in the same row
        vector<cv::Mat> row_r_img;
        vector<cv::Mat> col_t_img;//img in the same col
        vector<cv::Mat> col_b_img;
        int im_count,im_number;
        //row_l_img.push_back(h_im0);
        //col_t_img.push_back(top_img);
        Mat row_mat,col_mat;
        //*****get the same row image********
        /*for(im_count=1;im_count<2*IMAGE_NUMBER;im_count++)
        {
           if(im_count<4)
           {
               sprintf(r_im_dir,"%s%s%d%s",img_dir,"input_Cam0",im_count+36,".png");
               row_mat=cv::imread(r_im_dir,0);
               row_l_img.push_back(row_mat);
           }
           if(im_count>4)
           {
               sprintf(r_im_dir,"%s%s%d%s",img_dir,"input_Cam0",im_count+36,".png");
               row_mat=cv::imread(r_im_dir,0);
               row_r_img.push_back(row_mat);
           }
        }
        row_r_img.push_back(h_im1);*/
        //**********************************
        //*************get the same col image*****
        /*for(im_count=1;im_count<8;im_count++)
        {
            im_number=im_count*9+4;
            if(im_count<4)
            {
                sprintf(c_im_dir,"%s%s%d%s",img_dir,"input_Cam0",im_number,".png");
                col_mat=cv::imread(c_im_dir,0);
                col_t_img.push_back(col_mat);
            }
            if(im_count>4)
            {
                sprintf(c_im_dir,"%s%s%d%s",img_dir,"input_Cam0",im_number,".png");
                col_mat=cv::imread(c_im_dir,0);
                col_b_img.push_back(col_mat);
            }
        }
        col_b_img.push_back(bottom_img);*/

        //******************new image load*************************************************
        vector<cv::Mat> l_image;
        vector<cv::Mat> r_image;
        vector<cv::Mat> t_image;
        vector<cv::Mat> b_image;

        vector<cv::Mat> top_left_image;
        vector<cv::Mat> top_right_image;
        vector<cv::Mat> bottom_left_image;
        vector<cv::Mat> bottom_right_image;

        cv::Mat center_image;
        cv::Size new_size;
        new_size.width=IMG_WIDTH;
        new_size.height=IMG_HEIGHT;
        row_mat =cv::imread("/home/wangyuanqi/code/code/libSGM/image1_1/right/RRR.jpg",1);
        col_mat =cv::imread("/home/wangyuanqi/code/code/libSGM/image1_1/right/RRR.jpg",1);
        for(im_count=0;im_count<2*IMAGE_NUMBER;im_count++)
        {
           if(im_count<IMAGE_NUMBER)
           {   
               sprintf(r_im_dir,"%s%s%d%s",img_dir,"input_Cam0",IMAGE_NUMBER-1-im_count+4*(2*IMAGE_NUMBER+1),".png");
               row_mat=cv::imread(r_im_dir,0);
               //row_mat =cv::imread("/home/wangyuanqi/code/code/libSGM/image1_1/right/RRR.jpg",0);
               std::cout<<"left::"<<IMAGE_NUMBER-1-im_count+4*(2*IMAGE_NUMBER+1)<<std::endl;
               //*****
               cv::resize(row_mat,row_mat,new_size);
               //*****
               l_image.push_back(row_mat);

           }
           if(im_count>=IMAGE_NUMBER)
           {

               sprintf(r_im_dir,"%s%s%d%s",img_dir,"input_Cam0",im_count+4*(2*IMAGE_NUMBER+1)+1,".png");
               row_mat=cv::imread(r_im_dir,0);
               //row_mat =cv::imread("/home/wangyuanqi/code/code/libSGM/image1_1/right/RRR.jpg",0);
               std::cout<<"right::"<<im_count+4*(2*IMAGE_NUMBER+1)+1<<std::endl;
               //*****
               cv::resize(row_mat,row_mat,new_size);
               //*****
               r_image.push_back(row_mat);

           }
        }
        std::cout<<"vector+size::"<<l_image.size()<<std::endl;
        //******************the same row***************************************************
        for(im_count=0;im_count<2*IMAGE_NUMBER;im_count++)
        {
            if(im_count==IMAGE_NUMBER-1)
            {
                im_number=(IMAGE_NUMBER-1-im_count)*(2*IMAGE_NUMBER+1)+IMAGE_NUMBER;
                sprintf(c_im_dir,"%s%s%d%s",img_dir,"input_Cam00",im_number,".png");
                col_mat=cv::imread(c_im_dir,0);
                //row_mat =cv::imread("/home/wangyuanqi/code/code/libSGM/image1_1/right/RRR.jpg",0);
                //std::cout<<col_mat.cols<<std::endl;
                std::cout<<"top::"<<im_number<<std::endl;

                //****
                //cv::transpose(col_mat,col_mat);
                //cv::flip(col_mat,col_mat,1);
                //cv::imwrite("1.jpg",col_mat);
                //*****
                //*****
                cv::resize(col_mat,col_mat,new_size);
                //*****
                t_image.push_back(col_mat);

                continue;
            }
            if(im_count<IMAGE_NUMBER)
            {
                im_number=(IMAGE_NUMBER-im_count-1)*(2*IMAGE_NUMBER+1)+IMAGE_NUMBER;
                sprintf(c_im_dir,"%s%s%d%s",img_dir,"input_Cam0",im_number,".png");
                col_mat=cv::imread(c_im_dir,0);
                //col_mat =cv::imread("/home/wangyuanqi/code/code/libSGM/image1_1/right/RRR.jpg",0);
                //std::cout<<col_mat.cols<<std::endl;
                std::cout<<"top::"<<im_number<<std::endl;

                //*****
                cv::resize(col_mat,col_mat,new_size);
                //*****
                t_image.push_back(col_mat);
            }
            if(im_count>=IMAGE_NUMBER)
            {
                im_number=(im_count+1)*(2*IMAGE_NUMBER+1)+IMAGE_NUMBER;
                sprintf(c_im_dir,"%s%s%d%s",img_dir,"input_Cam0",im_number,".png");
                col_mat=cv::imread(c_im_dir,0);
                //col_mat =cv::imread("/home/wangyuanqi/code/code/libSGM/image1_1/right/RRR.jpg",0);
                if(col_mat.empty())
                    return 0;
                std::cout<<"bottom::"<<c_im_dir<<std::endl;
                //std::cout<<col_mat.cols<<std::endl;
                //*****
                cv::resize(col_mat,col_mat,new_size);
                //*****
                b_image.push_back(col_mat);
            }
        }
        //load left_top image and load right_top image
        for(im_count=1;im_count<2*IMAGE_NUMBER+1;im_count++)
        {
            if(im_count<IMAGE_NUMBER)
            {
                im_number=(IMAGE_NUMBER-im_count)*9+(IMAGE_NUMBER-im_count);
                sprintf(c_im_dir,"%s%s%d%s",img_dir,"input_Cam0",im_number,".png");
                std::cout<<"top_left_image::"<<c_im_dir<<std::endl;
                col_mat=cv::imread(c_im_dir,1);
                //col_mat =cv::imread("/home/wangyuanqi/code/code/libSGM/image1_1/right/RRR.jpg",0);
                cv::resize(col_mat,col_mat,new_size);
                if(col_mat.empty())
                    for(;;);
                top_left_image.push_back(col_mat);
            }
            else if(im_count==IMAGE_NUMBER)
            {
                im_number=(IMAGE_NUMBER-im_count)*9+(IMAGE_NUMBER-im_count);
                sprintf(c_im_dir,"%s%s%d%s",img_dir,"input_Cam00",im_number,".png");
                std::cout<<"top_left_image::"<<c_im_dir<<std::endl;
                col_mat=cv::imread(c_im_dir,1);
                //col_mat =cv::imread("/home/wangyuanqi/code/code/libSGM/image1_1/right/RRR.jpg",0);
                if(col_mat.empty())
                    for(;;);
                cv::resize(col_mat,col_mat,new_size);
                top_left_image.push_back(col_mat);
            }
            else if(im_count>IMAGE_NUMBER)
            {
                im_number=(im_count)*9+(im_count);
                sprintf(c_im_dir,"%s%s%d%s",img_dir,"input_Cam0",im_number,".png");
                std::cout<<"bottom_right_image::"<<c_im_dir<<std::endl;
                col_mat=cv::imread(c_im_dir,1);
                //col_mat =cv::imread("/home/wangyuanqi/code/code/libSGM/image1_1/right/RRR.jpg",0);
                if(col_mat.empty())
                    for(;;);
                cv::resize(col_mat,col_mat,new_size);
                bottom_right_image.push_back(col_mat);
            }
        }
        //load end**
        //load you zuo biao xi
        for(im_count=1;im_count<9;im_count++)
        {
            if(im_count<IMAGE_NUMBER)
            {
                im_number=(IMAGE_NUMBER-im_count)*9+(IMAGE_NUMBER+im_count);
                sprintf(c_im_dir,"%s%s%d%s",img_dir,"input_Cam0",im_number,".png");
                std::cout<<"top_right_image::"<<c_im_dir<<std::endl;
                col_mat=cv::imread(c_im_dir,1);
                //col_mat =cv::imread("/home/wangyuanqi/code/code/libSGM/image1_1/right/RRR.jpg",0);
                if(col_mat.empty())
                    for(;;);
                cv::resize(col_mat,col_mat,new_size);
                top_right_image.push_back(col_mat);
            }
            else if(im_count==IMAGE_NUMBER)
            {
                im_number=(IMAGE_NUMBER-im_count)*9+(IMAGE_NUMBER+im_count);
                sprintf(c_im_dir,"%s%s%d%s",img_dir,"input_Cam00",im_number,".png");
                std::cout<<"top_right_image::"<<c_im_dir<<std::endl;
                col_mat=cv::imread(c_im_dir,1);
                //col_mat =cv::imread("/home/wangyuanqi/code/code/libSGM/image1_1/right/RRR.jpg",0);
                if(col_mat.empty())
                    for(;;);
                cv::resize(col_mat,col_mat,new_size);
                top_right_image.push_back(col_mat);
            }
            else if(im_count>IMAGE_NUMBER)
            {
                im_number=(im_count)*9+(2*IMAGE_NUMBER-im_count);
                sprintf(c_im_dir,"%s%s%d%s",img_dir,"input_Cam0",im_number,".png");
                std::cout<<"bottom_left_image::"<<c_im_dir<<std::endl;
                col_mat=cv::imread(c_im_dir,1);
                //col_mat =cv::imread("/home/wangyuanqi/code/code/libSGM/image1_1/right/RRR.jpg",0);
                if(col_mat.empty())
                    for(;;);
                cv::resize(col_mat,col_mat,new_size);
                bottom_left_image.push_back(col_mat);
            }
        }

        //********************************************
        im_number=IMAGE_NUMBER*(2*IMAGE_NUMBER+1)+IMAGE_NUMBER;
        sprintf(c_im_dir,"%s%s%d%s",img_dir,"input_Cam0",im_number,".png");
        std::cout<<"center::"<<im_number<<std::endl;
        center_image=cv::imread(c_im_dir,0);
        //*****
        cv::resize(center_image,center_image,new_size);
        //*****
        //******warp Image*******
        //cudaSetDevice(1);
        cudaSetDevice(1);
        int s_mount = 0;

        //***   for stereo image to use stereo matching


        cv::Mat right_img;
        //= cv::imread("/home/wangyuanqi/code/code/libSGM/images/right/right1.png",0);
        right_img = cv::imread("/home/wangyuanqi/code/code/sgm_12_30/sgm/example/test_8_3/right5.png",0);
        //right_img = cv::imread("right.png",0);
        cv::resize(right_img,right_img,new_size);
        //center_image=cv::imread("/home/wangyuanqi/code/code/libSGM/images/left/left1.png",0);
        center_image=cv::imread("/home/wangyuanqi/code/code/sgm_12_30/sgm/example/test_8_3/left5.png",0);
        //center_image=cv::imread("left.png",0);

        cv::resize(center_image,center_image,new_size);

        // new_image load for five
        cv::Mat image_1,image_2,image_3,image_4,image_5;
        string image_path ="/home/wangyuanqi/code/code/sgm_12_30/camera_5_code/camera_image/";
        image_1 = cv::imread(image_path+"0001.bmp",0);
        image_2 = cv::imread(image_path+"0002.bmp",0);
        image_3 = cv::imread(image_path+"0003.bmp",0);
        image_4 = cv::imread(image_path+"0004.bmp",0);
        image_5 = cv::imread(image_path+"0005.bmp",0);

        cv::resize(image_1,image_1,new_size);
        cv::resize(image_2,image_2,new_size);
        cv::resize(image_3,image_3,new_size);
        cv::resize(image_4,image_4,new_size);
        cv::resize(image_5,image_5,new_size);

        center_image = image_3.clone();

        l_image.pop_back();
        l_image.pop_back();
        r_image.pop_back();
        r_image.pop_back();


        l_image.push_back(image_1);
        l_image.push_back(image_2);
        r_image.push_back(image_5);
        r_image.push_back(image_4);

        cv::imwrite("result.jpg",center_image);
        if(right_img.empty()||center_image.empty())
            for(;;);
        //r_image.insert(r_image.begin(),right_img);
        //r_image.pop_back();
        //r_image.push_back(right_img);
        //*********************************************
        reverse(l_image.begin(),l_image.end());
        reverse(r_image.begin(),r_image.end());
        reverse(t_image.begin(),t_image.end());
        reverse(b_image.begin(),b_image.end());
        reverse(top_left_image.begin(),top_left_image.end());
        reverse(top_right_image.begin(),top_right_image.end());
        reverse(bottom_left_image.begin(),bottom_left_image.end());
        reverse(bottom_right_image.begin(),bottom_right_image.end());


        for(;s_mount<IMAGE_NUMBER-SELECT_IMAGE_NUM;s_mount++)
        {
            l_image.pop_back();
            r_image.pop_back();
            t_image.pop_back();
            b_image.pop_back();
            top_left_image.pop_back();
            top_right_image.pop_back();
            bottom_left_image.pop_back();
            bottom_right_image.pop_back();
        }
        cv::imwrite("input.jpg",r_image[0]);

        Mat census_image=Image_Warp(center_image,l_image,r_image,t_image,b_image,top_left_image,top_right_image,bottom_left_image,bottom_right_image);
        cv::Mat_<float> result_img(IMG_HEIGHT,IMG_WIDTH,CV_32FC1);
        int max_color=0;
        for(int col_x=0;col_x<center_image.cols;col_x++)
        {
            for(int row_x=0;row_x<center_image.rows;row_x++)
            {
                max_color=max(max_color,census_image.at<uint8_t>(row_x,col_x));
                int color= census_image.at<uint8_t>(row_x,col_x);
                float warp_dis = (END_DIS-BEGIN_DIS)/float(NEW_MAX_DISPARITY);
                result_img(row_x,col_x)=(color-SHIFT_MIDDLE)*warp_dis+BEGIN_DIS;
                census_image.at<uint8_t>(row_x,col_x)*=2;
            }
        }

        LRCheckOut(center_image,right_img,result_img,census_image);
        //WriteFilePFM(result_img,"/home/wangyuanqi/code/code/evaluation-toolkit-master_2/algo_results/wang_method1/disp_maps/rosemary.pfm",1);
        Mat_<float> depth_mat = result_img.clone();
        //DepthDeal(result_img,depth_mat);
        WriteFilePFM(depth_mat,"boxes/test_5/depth.pfm",1);
         WriteFilePFM(result_img,"boxes/test_5/boxes.pfm",1);
        std::cout<<max_color<<std::endl;
       // for(;;);
        //******warp end  *******
        std::cout<<"end"<<std::endl;
        //Mat census_image=n_compute_disparity(l_image,r_image,center_image,t_image,b_image);
        if(!census_image.empty())
        {
            std::cout<<"not empty::"<<census_image.cols<<"::"<<census_image.rows<<std::endl;
             cv::imwrite("boxes/test_5/c_result.png",census_image);
             //write Mat_<float>


             //for(;;);
            Mat x_color;
            cv::applyColorMap(census_image,x_color,cv::COLORMAP_JET);
           // imshow("census",census_image*256/NEW_MAX_DISPARITY);
            //cv::imwrite("c_result.png",census_image*256/NEW_MAX_DISPARITY);
            //cv::imwrite("c_result.png",census_image*2);
           // imshow("census_color",x_color);
            cv::imwrite("boxes/test_5/x_color.jpg",x_color);
            std::cout<<"census_mount"<<census_image.at<uchar>(0,0)<<std::endl;
        }
        else {
            std::cout<<"empty"<<std::endl;
             for(;;);
        }
        //******************end************************************************************
        //*****************begin subpixel disparity****************************************
        //*****************set candidate disparity for each point**************************

        //*****************end*************************************************************
        //*****************end ************************************************************
//#if LOG
        std::cout<< "done" <<std::endl;
//#endif
		times.push_back(elapsed_time_ms);
        //float* ;bsn_err;
       /* if(has_gt) {
			disparity_errors(disparity_im, gt_file, &n, &n_err);
        }*/
//#if WRITE_FILES
    //const int type = disparity_im.type();
    //const uchar depth = type & CV_MAT_DEPTH_MASK;
    //cv::Mat color,disp,top_color,final_color;
    //if(depth == CV_8U) {
        /*cv::normalize(disparity_im*256/MAX_DISPARITY,disp,0.,255.,cv::NORM_MINMAX);
        disparity_im.convertTo(disp,CV_8UC1);
        cv::imwrite(dis_file, disp);
        cv::applyColorMap(disp,color,cv::COLORMAP_RAINBOW);
        cv::imwrite("colog.jpg",color);*/

      //  std::cout<<"type"<<disparity_im.type()<<std::endl;
      //  unsigned char* img_data=census_image.data;
       // int out_data;
      //  out_data=(int)img_data[0];
      //  std::cout<<"data::"<<out_data<<std::endl;
      //  std::ofstream out_stream("result.txt");
      //  int m_height,m_width;
     //   for(m_height=0;m_height<disparity_im.rows;m_height++)
      //  {
        //    for(m_width=0;m_width<disparity_im.cols;m_width++)
        //    {
               // out_stream << (int)img_data[m_height*disparity_im.cols+m_width]+"\t";
           //     out_data=(int)img_data[m_height*disparity_im.cols+m_width];
        //        out_stream <<out_data;
        //        out_stream <<"\t";
        //    }
            //std::cout<<""<<std::endl;
        //    out_stream<<"\n";
       // }
      //  out_stream.close();

      //  cv::imshow("Sobel_result",sobel_mat);
      //  cv::imwrite("disparity.jpg",disparity_im);
      //  cv::imshow("disparity",disparity_im*256/MAX_DISPARITY);
      //  cv::imshow("top_disparity",t_disparity*256/MAX_DISPARITY);
       // cv::imwrite("final_disparity.jpg",final_disparity_im*512/MAX_DISPARITY);
     //   cv::applyColorMap(disparity_im*256/MAX_DISPARITY,color,cv::COLORMAP_JET);
    //    cv::applyColorMap(t_disparity*256/MAX_DISPARITY,top_color,cv::COLORMAP_JET);
      //  cv::applyColorMap(final_disparity_im*256/MAX_DISPARITY,final_color,cv::COLORMAP_JET);
     //   cv::imshow("top_color",top_color);
       // cv::imshow("confidence",final_negative_mat);
        //**********add 7x7 window slip*******
        //unsigned char* final_color_data=final_color.data;
     //   std::cout<<"height::"<<final_color.type()<<std::endl;
        /*for(m_height=0;m_height<final_color.rows;m_height++)
        {
            for(m_width=0;m_width<final_color.cols;m_width++)
            {
                if((m_height+1)%7==0)
                {
                    //std::cout<<"height::"<<final_color.rows<<std::endl;
                    final_color.at<cv::Vec3b>(m_height,m_width)[0]=255;
                    final_color.at<cv::Vec3b>(m_height,m_width)[1]=255;
                    final_color.at<cv::Vec3b>(m_height,m_width)[2]=255;
                }
                if((m_width+1)%7==0)
                {
                    final_color.at<cv::Vec3b>(m_height,m_width)[0]=255;
                    final_color.at<cv::Vec3b>(m_height,m_width)[1]=255;
                    final_color.at<cv::Vec3b>(m_height,m_width)[2]=255;
                }
            }
        }*/
        //************************************
     //   cv::imshow("final_color",final_color);
      //  cv::imwrite("top_color.jpg",top_color);
     //   cv::imwrite("final_color.jpg",final_color);
     //   cv::imwrite("colog.jpg",color);
     //   cv::imshow("color",color);
        //cv::imwrite("disparity_center_l.jpg",c_l_disparity_im*256/MAX_DISPARITY);
        //cv::imwrite("disparity_center_t.jpg",c_t_diparity_im*256/MAX_DISPARITY);
        cv::waitKey(0);
    //}
        /*else {
		cv::Mat disparity16(disparity_im.rows, disparity_im.cols, CV_16UC1);
		for(int i = 0; i < disparity_im.rows; i++) {
			for(int j = 0; j < disparity_im.cols; j++) {
				const float d = disparity_im.at<float>(i, j)*256.0f;
				disparity16.at<uint16_t>(i, j) = (uint16_t) d;
			}
		}
		cv::imwrite(dis_file, disparity16);
    }*/
//#endif
	}
//	closedir(dp);
	finish_disparity_method();

	double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    /*if(has_gt) {
		printf("%f\n", (float) n_err/n);
    } else */
    {
		std::cout << "It took an average of " << mean << " miliseconds, " << 1000.0f/mean << " fps" << std::endl;
	}

	return 0;
}
