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

#include "costs.h"
#include <stdio.h>
#include <device_launch_parameters.h>

__global__ void __launch_bounds__(1024, 2)
CenterSymmetricCensusKernelSM2(const uint8_t *im, const uint8_t *im2, cost_t *transform, cost_t *transform2, const uint32_t rows, const uint32_t cols) {
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	const int idy = blockIdx.y*blockDim.y+threadIdx.y;

    const int win_cols = (32+LEFT*2); // 32+4*2 = 40
    const int win_rows = (32+TOP*2); // 32+3*2 = 38

    /*const int win_cols=7;
    const int win_rows=7;*/
	__shared__ uint8_t window[win_cols*win_rows];
	__shared__ uint8_t window2[win_cols*win_rows];

	const int id = threadIdx.y*blockDim.x+threadIdx.x;
	const int sm_row = id / win_cols;
	const int sm_col = id % win_cols;

	const int im_row = blockIdx.y*blockDim.y+sm_row-TOP;
	const int im_col = blockIdx.x*blockDim.x+sm_col-LEFT;
	const bool boundaries = (im_row >= 0 && im_col >= 0 && im_row < rows && im_col < cols);
    //printf("fii\n");
	window[sm_row*win_cols+sm_col] = boundaries ? im[im_row*cols+im_col] : 0;
	window2[sm_row*win_cols+sm_col] = boundaries ? im2[im_row*cols+im_col] : 0;

	// Not enough threads to fill window and window2
	const int block_size = blockDim.x*blockDim.y;
	if(id < (win_cols*win_rows-block_size)) {
		const int id = threadIdx.y*blockDim.x+threadIdx.x+block_size;
		const int sm_row = id / win_cols;
		const int sm_col = id % win_cols;

		const int im_row = blockIdx.y*blockDim.y+sm_row-TOP;
		const int im_col = blockIdx.x*blockDim.x+sm_col-LEFT;
		const bool boundaries = (im_row >= 0 && im_col >= 0 && im_row < rows && im_col < cols);
		window[sm_row*win_cols+sm_col] = boundaries ? im[im_row*cols+im_col] : 0;
		window2[sm_row*win_cols+sm_col] = boundaries ? im2[im_row*cols+im_col] : 0;
	}

	__syncthreads();
	cost_t census = 0;
	cost_t census2 = 0;
	if(idy < rows && idx < cols) {
			for(int k = 0; k < CENSUS_HEIGHT/2; k++) {
				for(int m = 0; m < CENSUS_WIDTH; m++) {
					const uint8_t e1 = window[(threadIdx.y+k)*win_cols+threadIdx.x+m];
					const uint8_t e2 = window[(threadIdx.y+2*TOP-k)*win_cols+threadIdx.x+2*LEFT-m];
					const uint8_t i1 = window2[(threadIdx.y+k)*win_cols+threadIdx.x+m];
					const uint8_t i2 = window2[(threadIdx.y+2*TOP-k)*win_cols+threadIdx.x+2*LEFT-m];

					const int shft = k*CENSUS_WIDTH+m;
					// Compare to the center
					cost_t tmp = (e1 >= e2);
					// Shift to the desired position
					tmp <<= shft;
					// Add it to its place
					census |= tmp;
					// Compare to the center
					cost_t tmp2 = (i1 >= i2);
					// Shift to the desired position
					tmp2 <<= shft;
					// Add it to its place
					census2 |= tmp2;
				}
			}
			if(CENSUS_HEIGHT % 2 != 0) {
				const int k = CENSUS_HEIGHT/2;
				for(int m = 0; m < CENSUS_WIDTH/2; m++) {
					const uint8_t e1 = window[(threadIdx.y+k)*win_cols+threadIdx.x+m];
					const uint8_t e2 = window[(threadIdx.y+2*TOP-k)*win_cols+threadIdx.x+2*LEFT-m];
					const uint8_t i1 = window2[(threadIdx.y+k)*win_cols+threadIdx.x+m];
					const uint8_t i2 = window2[(threadIdx.y+2*TOP-k)*win_cols+threadIdx.x+2*LEFT-m];

					const int shft = k*CENSUS_WIDTH+m;
					// Compare to the center
					cost_t tmp = (e1 >= e2);
					// Shift to the desired position
					tmp <<= shft;
					// Add it to its place
					census |= tmp;
					// Compare to the center
					cost_t tmp2 = (i1 >= i2);
					// Shift to the desired position
					tmp2 <<= shft;
					// Add it to its place
					census2 |= tmp2;
				}
			}

		transform[idy*cols+idx] = census;
		transform2[idy*cols+idx] = census2;
	}
}

__global__ void
N_CenterSymmetricCnesusKernelSM2( uint8_t **l_im, uint8_t **r_im,cost_t **l_transform,cost_t **r_transform)
{
    const int p_x=(IMG_WIDTH+blockDim.x-1) / blockDim.x;
    //printf("%d\n",p_x);
    const int bx=blockIdx.x/p_x;
    int img_num=bx;
    //if(img_num>=SELECT_IMAGE_NUM*NEW_MAX_DISPARITY)
    //if(img_num == 125)
    //printf("img_num::%d\n",img_num);
    C_CenterSymmetricCensusKernelSM2(l_im[img_num],r_im[img_num],l_transform[img_num],r_transform[img_num],IMG_HEIGHT,IMG_WIDTH);
    //C_CenterSymmetricCensusKernelSM2(t_im[img_num],b_im[img_num],t_transform[img_num],b_transform[img_num],IMG_HEIGHT,IMG_WIDTH);
}

__device__ void
 C_CenterSymmetricCensusKernelSM2( uint8_t *im,  uint8_t *im2, cost_t *transform, cost_t *transform2, const uint32_t rows, const uint32_t cols)
{
    //printf("begin\n");
     const int p_x=(IMG_WIDTH+blockDim.x-1) / blockDim.x;
    const int bx=blockIdx.x%(p_x);
    //if(bx>=16)
     //   printf("bx::%d\n",bx);
    const int idx = bx*blockDim.x+threadIdx.x;
    const int idy = blockIdx.y*blockDim.y+threadIdx.y;

    const int win_cols = (32+LEFT*2); // 32+4*2 = 40
    const int win_rows = (32+TOP*2); // 32+3*2 = 38

    /*const int win_cols=7;
    const int win_rows=7;*/
    __shared__ uint8_t window[win_cols*win_rows];
    __shared__ uint8_t window2[win_cols*win_rows];

    const int id = threadIdx.y*blockDim.x+threadIdx.x;
    const int sm_row = id / win_cols;
    const int sm_col = id % win_cols;

    const int im_row = blockIdx.y*blockDim.y+sm_row-TOP;
    const int im_col = bx*blockDim.x+sm_col-LEFT;
    const bool boundaries = (im_row >= 0 && im_col >= 0 && im_row < rows && im_col < cols);
    //printf("fii\n");
    window[sm_row*win_cols+sm_col] = boundaries ? im[im_row*cols+im_col] : 0;
    window2[sm_row*win_cols+sm_col] = boundaries ? im2[im_row*cols+im_col] : 0;

    // Not enough threads to fill window and window2
    const int block_size = blockDim.x*blockDim.y;
    if(id < (win_cols*win_rows-block_size)) {
        const int id = threadIdx.y*blockDim.x+threadIdx.x+block_size;
        const int sm_row = id / win_cols;
        const int sm_col = id % win_cols;

        const int im_row = blockIdx.y*blockDim.y+sm_row-TOP;
        const int im_col = bx*blockDim.x+sm_col-LEFT;
        const bool boundaries = (im_row >= 0 && im_col >= 0 && im_row < rows && im_col < cols);
        window[sm_row*win_cols+sm_col] = boundaries ? im[im_row*cols+im_col] : 0;
        window2[sm_row*win_cols+sm_col] = boundaries ? im2[im_row*cols+im_col] : 0;
    }

    __syncthreads();
    cost_t census = 0;
    cost_t census2 = 0;
    if(idy < rows && idx < cols) {
            for(int k = 0; k < CENSUS_HEIGHT/2; k++) {
                for(int m = 0; m < CENSUS_WIDTH; m++) {
                    const uint8_t e1 = window[(threadIdx.y+k)*win_cols+threadIdx.x+m];
                    const uint8_t e2 = window[(threadIdx.y+2*TOP-k)*win_cols+threadIdx.x+2*LEFT-m];
                    const uint8_t i1 = window2[(threadIdx.y+k)*win_cols+threadIdx.x+m];
                    const uint8_t i2 = window2[(threadIdx.y+2*TOP-k)*win_cols+threadIdx.x+2*LEFT-m];

                    const int shft = k*CENSUS_WIDTH+m;
                    // Compare to the center
                    cost_t tmp = (e1 >= e2);
                    // Shift to the desired position
                    tmp <<= shft;
                    // Add it to its place
                    census |= tmp;
                    // Compare to the center
                    cost_t tmp2 = (i1 >= i2);
                    // Shift to the desired position
                    tmp2 <<= shft;
                    // Add it to its place
                    census2 |= tmp2;
                }
            }
            if(CENSUS_HEIGHT % 2 != 0) {
                const int k = CENSUS_HEIGHT/2;
                for(int m = 0; m < CENSUS_WIDTH/2; m++) {
                    const uint8_t e1 = window[(threadIdx.y+k)*win_cols+threadIdx.x+m];
                    const uint8_t e2 = window[(threadIdx.y+2*TOP-k)*win_cols+threadIdx.x+2*LEFT-m];
                    const uint8_t i1 = window2[(threadIdx.y+k)*win_cols+threadIdx.x+m];
                    const uint8_t i2 = window2[(threadIdx.y+2*TOP-k)*win_cols+threadIdx.x+2*LEFT-m];

                    const int shft = k*CENSUS_WIDTH+m;
                    // Compare to the center
                    cost_t tmp = (e1 >= e2);
                    // Shift to the desired position
                    tmp <<= shft;
                    // Add it to its place
                    census |= tmp;
                    // Compare to the center
                    cost_t tmp2 = (i1 >= i2);
                    // Shift to the desired position
                    tmp2 <<= shft;
                    // Add it to its place
                    census2 |= tmp2;
                }
            }

        transform[idy*cols+idx] = census;
        transform2[idy*cols+idx] = census2;
    }
}

__global__ void
ShiftImage(uint8_t **orign_im, uint8_t **center_warp,int px,int py)
{
    //printf("begin\n");
    //const int b_x=blockIdx.x;
    const int b_y=blockIdx.y;
    int image_n=b_y/NEW_MAX_DISPARITY;
    int dis_n=b_y-image_n*(NEW_MAX_DISPARITY);
    float d_dis=(dis_n+1-SHIFT_MIDDLE)*WARP_DIS;
    float d_y=py*d_dis*(image_n+1),d_x=px*d_dis*(image_n+1);
    WarpImage(orign_im[image_n],center_warp[b_y],d_x,d_y);
}
__global__ void N_ShiftImage(uint8_t **left_img,uint8_t **left_warp,uint8_t **right_img,uint8_t **right_warp,
                             uint8_t **top_img, uint8_t **top_warp,uint8_t **bottom_img,uint8_t **bottom_warp,
                             uint8_t **top_left_img, uint8_t **top_left_warp, uint8_t **top_right_img, uint8_t **top_right_warp,
                             uint8_t **bottom_left_img, uint8_t **bottom_left_warp, uint8_t **bottom_right_img, uint8_t **bottom_right_warp)
{
    const int b_y=blockIdx.y;
    /*if(b_y>64)
        printf("%d\n",b_y);*/
    const int t_y=threadIdx.y;
    int image_n=b_y/NEW_MAX_DISPARITY;
    int label_n = IMAGE_NUMBER-1-image_n;
    label_n =0;
    //int dis_n=b_y-image_n*(NEW_MAX_DISPARITY);
    int dis_n =  b_y-image_n*(NEW_MAX_DISPARITY);
    //float d_dis=(dis_n+1-SHIFT_MIDDLE)*WARP_DIS;
    //if(dis_n==127)
    //   printf("dis*****************%d\n",dis_n);
    const float warp_dis= ((float(END_DIS-BEGIN_DIS)/NEW_MAX_DISPARITY));
    float d_dis=(dis_n-SHIFT_MIDDLE)*warp_dis+BEGIN_DIS;
    //if(d_dis==WARP_DIS*-63)
    //  printf("d_dis::*****************%f\n",d_dis);
    // float d_y=py*d_dis*(image_n+1),d_x=px*d_dis*(image_n+1);
    int px,py;
    float d_x,d_y;
    //if(t_y==0)
    {
        px=-1,py=0;
        //d_y=py*d_dis*(label_n+1),d_x=px*d_dis*(label_n+1);
        d_y=py*d_dis*(1),d_x=px*(d_dis*(1)+BEGIN_DIS);
        WarpImage(left_img[image_n],left_warp[b_y],d_x,d_y);
    }
    // else if(t_y==1)
    {
        px=1,py=0;
        //d_y=py*d_dis*(label_n+1),d_x=px*d_dis*(label_n+1)+BEGIN_DIS;
        d_y=py*d_dis*(label_n),d_x=px*(d_dis*(1)+BEGIN_DIS);
        /*if(d_x>64.0f)
            printf("%f::%f\n",d_y,d_x);*/
        WarpImage(right_img[image_n],right_warp[b_y],d_x,d_y);
    }
    //else if(t_y==2)
    {
        px=0,py=-1;
        d_y=py*d_dis*(label_n+1),d_x=px*d_dis*(label_n+1);
        WarpImage(top_img[image_n],top_warp[b_y],d_x,d_y);
    }
    //else
    {
        px=0,py=1;
        d_y=py*d_dis*(label_n+1),d_x=px*d_dis*(label_n+1);
        WarpImage(bottom_img[image_n],bottom_warp[b_y],d_x,d_y);
    }

    //top_left
    {
        px=-1, py=-1;
        d_y=py*d_dis*(label_n+1),d_x=px*d_dis*(label_n+1);
        WarpImage(top_left_img[image_n],top_left_warp[b_y],d_x,d_y);
    }
    //top_right
    {
        px=1, py=-1;
        d_y=py*d_dis*(label_n+1),d_x=px*d_dis*(label_n+1);
        WarpImage(top_right_img[image_n],top_right_warp[b_y],d_x,d_y);
    }
    //bottom_left
    {
        px=-1, py=1;
        d_y=py*d_dis*(label_n+1),d_x=px*d_dis*(label_n+1);
        WarpImage(bottom_left_img[image_n],bottom_left_warp[b_y],d_x,d_y);
    }
    //bottom_right
    {
        px=1, py=1;
        d_y=py*d_dis*(label_n+1),d_x=px*d_dis*(label_n+1);
        WarpImage(bottom_right_img[image_n],bottom_right_warp[b_y],d_x,d_y);
    }
}
/*__global__ void N_ShiftImage(uint8_t **left_img,uint8_t **left_warp,uint8_t **right_img,uint8_t **right_warp,
                             uint8_t **top_img, uint8_t **top_warp,uint8_t **bottom_img,uint8_t **bottom_warp,
                             uint8_t **top_left_img, uint8_t **top_left_warp, uint8_t **top_right_img, uint8_t **top_right_warp,
                             uint8_t **bottom_left_img, uint8_t **bottom_left_warp, uint8_t **bottom_right_img, uint8_t **bottom_right_warp)
{
    const int b_y=blockIdx.y;

    const int t_y=threadIdx.y;
    int image_n=b_y/NEW_MAX_DISPARITY;
    int label_n = IMAGE_NUMBER-1-image_n;
    label_n =0;
    //int dis_n=b_y-image_n*(NEW_MAX_DISPARITY);
    int dis_n =  b_y-image_n*(NEW_MAX_DISPARITY);
    //float d_dis=(dis_n+1-SHIFT_MIDDLE)*WARP_DIS;
    //if(dis_n==127)
     //   printf("dis*****************%d\n",dis_n);
    const float warp_dis= WARP_DIS*((float(END_DIS-BEGIN_DIS)/NEW_MAX_DISPARITY));
    float d_dis=(dis_n-SHIFT_MIDDLE)*warp_dis;
    //if(d_dis==WARP_DIS*-63)
      //  printf("d_dis::*****************%f\n",d_dis);
   // float d_y=py*d_dis*(image_n+1),d_x=px*d_dis*(image_n+1);
    int px,py;
    float d_x,d_y;
    //if(t_y==0)
    {
        px=-1,py=0;
        d_y=py*d_dis*(label_n+1),d_x=px*d_dis*(label_n+1);
        WarpImage(left_img[image_n],left_warp[b_y],d_x,d_y);
    }
   // else if(t_y==1)
    {
        px=1,py=0;
        d_y=py*d_dis*(label_n+1),d_x=px*d_dis*(label_n+1)+BEGIN_DIS;

        WarpImage(right_img[image_n],right_warp[b_y],d_x,d_y);
    }
    //else if(t_y==2)
    {
        px=0,py=-1;
        d_y=py*d_dis*(label_n+1),d_x=px*d_dis*(label_n+1);
        WarpImage(top_img[image_n],top_warp[b_y],d_x,d_y);
    }
    //else
    {
        px=0,py=1;
        d_y=py*d_dis*(label_n+1),d_x=px*d_dis*(label_n+1);
        WarpImage(bottom_img[image_n],bottom_warp[b_y],d_x,d_y);
    }

    //top_left
    {
        px=-1, py=-1;
        d_y=py*d_dis*(label_n+1),d_x=px*d_dis*(label_n+1);
        WarpImage(top_left_img[image_n],top_left_warp[b_y],d_x,d_y);
    }
    //top_right
    {
        px=1, py=-1;
        d_y=py*d_dis*(label_n+1),d_x=px*d_dis*(label_n+1);
        WarpImage(top_right_img[image_n],top_right_warp[b_y],d_x,d_y);
    }
    //bottom_left
    {
        px=-1, py=1;
        d_y=py*d_dis*(label_n+1),d_x=px*d_dis*(label_n+1);
        WarpImage(bottom_left_img[image_n],bottom_left_warp[b_y],d_x,d_y);
    }
    //bottom_right
    {
        px=1, py=1;
        d_y=py*d_dis*(label_n+1),d_x=px*d_dis*(label_n+1);
        WarpImage(bottom_right_img[image_n],bottom_right_warp[b_y],d_x,d_y);
    }
}*/
__device__ void
WarpImage(const uint8_t *center_im,uint8_t *n_image,float dx,float dy)
{
  //  const int img_x=blockIdx.x*blockDim.x+threadIdx.x;
  //  const int img_y=blockIdx.y*blockDim.y+threadIdx.y;
    //printf("begin\n");
    int img_x=2*threadIdx.x;
    const int img_y=blockIdx.x;


    float n_x=img_x-dx,n_y=img_y-dy;
    //get the orign x and the orign y
    //printf("size::%d",sizeof(n_image)/sizeof(n_image[0]));
    if(n_x<FLOAT_EPS||n_x>IMG_WIDTH-1||n_y<FLOAT_EPS||n_y>IMG_HEIGHT-1)
        return ;
    int tl_x=(int)n_x,tl_y=(int)n_y,tr_x=n_x+1,tr_y=tl_y;
    int bl_x=tl_x,bl_y=tl_y+1,br_x=bl_x+1,br_y=bl_y;

    uint8_t tl_color=center_im[tl_x+tl_y*IMG_WIDTH];
    uint8_t tr_color=center_im[tr_x+tr_y*IMG_WIDTH];
    uint8_t bl_color=center_im[bl_x+bl_y*IMG_WIDTH];
    uint8_t br_color=center_im[br_x+br_y*IMG_WIDTH];

    uint8_t t_color=(tr_x-n_x)*tl_color+(n_x-tl_x)*tr_color;
    uint8_t b_color=(br_x-n_x)*bl_color+(n_x-bl_x)*br_color;
    int r_d=(br_y-n_y)*t_color+(n_y-tr_y)*b_color;
    uint8_t r_color=(br_y-n_y)*t_color+(n_y-tr_y)*b_color;
    n_image[img_x+img_y*IMG_WIDTH]=r_color;
    //n_image[0]=r_d;

    //fix for big size
    img_x = 2*threadIdx.x+1;
    n_x=img_x-dx,n_y=img_y-dy;
    //get the orign x and the orign y
    //printf("size::%d",sizeof(n_image)/sizeof(n_image[0]));
    if(n_x<FLOAT_EPS||n_x>IMG_WIDTH-1||n_y<FLOAT_EPS||n_y>IMG_HEIGHT-1)
        return ;
    tl_x=(int)n_x,tl_y=(int)n_y,tr_x=n_x+1,tr_y=tl_y;
    bl_x=tl_x,bl_y=tl_y+1,br_x=bl_x+1,br_y=bl_y;

    tl_color=center_im[tl_x+tl_y*IMG_WIDTH];
    tr_color=center_im[tr_x+tr_y*IMG_WIDTH];
    bl_color=center_im[bl_x+bl_y*IMG_WIDTH];
    br_color=center_im[br_x+br_y*IMG_WIDTH];

    t_color=(tr_x-n_x)*tl_color+(n_x-tl_x)*tr_color;
    b_color=(br_x-n_x)*bl_color+(n_x-bl_x)*br_color;
    r_d=(br_y-n_y)*t_color+(n_y-tr_y)*b_color;
    r_color=(br_y-n_y)*t_color+(n_y-tr_y)*b_color;
    n_image[img_x+img_y*IMG_WIDTH]=r_color;

}
