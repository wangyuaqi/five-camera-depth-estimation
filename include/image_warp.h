//
// Created by wangyuanqi on 19-8-21.
//

#ifndef SGM_IMAGE_WARP_H
#define SGM_IMAGE_WARP_H

#include "cost_aggregation.h"
#include "configuration.h"

#include "util.h"
#include <stdint.h>

__global__ void Image_Shift(uint8_t *origin_image, uint8_t **warp_image, float *disp,
                            int px, int py);

__device__ void ImageWarp(const uint8_t *center_im,uint8_t *n_image,float dx,float dy);
__global__ void ComputeCostVolume(cost_t **left_left_census, cost_t *left_left_center_census,
                                  cost_t **left_center_census, cost_t *left_center_center_census,
                                  cost_t **right_center_census, cost_t *right_center_center_census,
                                  cost_t **right_right_census, cost_t *right_right_center_census,
                                  uint32_t *final_cost_volume,
                                  int image_width, int image_height,
                                  float *left_left_x,float *left_left_y,
                                  float *left_center_x,float *left_center_y,
                                  float *right_center_x,float *right_center_y,
                                  float *right__right_x,float *right_right_y,
                                  float *left_left_disparity, float *left_center_disparity,
                                  float *right_center_disparity, float *right_right_disparity

);

#endif //SGM_IMAGE_WARP_H
