/*
 * new hamming cost for light_field
 */

#ifndef N_HAMMING_COST_H_
#define N_HAMMING_COST_H_

#include "configuration.h"
#include "util.h"
#include <stdint.h>

/*__global__ void
N_HammingDistanceCostKernel (  cost_t *l_transform1,cost_t *l_transform2,cost_t *l_transform3,cost_t *l_transform4,
                               cost_t *r_transform1 ,cost_t *r_transform2 ,cost_t *r_transform3 ,cost_t *r_transform4 ,
                               const cost_t* center_transform,
                               cost_t *t_transform1,cost_t *t_transform2,cost_t *t_transform3,cost_t *t_transform4,
                               cost_t *b_transform1,cost_t *b_transform2,cost_t *b_transform3,cost_t *b_transform4,
                               uint8_t *d_cost, const int rows, const int cols,const int image_number,const int sum_disparity);*/
//**************positive disparity***********************
__global__ void
N_HammingDistanceCostKernel1 (  cost_t *l_transform1,
                               cost_t *r_transform1 ,
                               const cost_t* center_transform,
                               cost_t *t_transform1,
                               cost_t *b_transform1,
                               uint32_t *d_cost, const int rows, const int cols,const int image_number,const int sum_disparity);
//**************negative disparity***********************
__global__ void
N_HammingDistanceCostKernel1_Z (  cost_t *l_transform1,
                               cost_t *r_transform1 ,
                               const cost_t* center_transform,
                               cost_t *t_transform1,
                               cost_t *b_transform1,
                               uint8_t *d_cost, const int rows, const int cols,const int image_number,const int sum_disparity);
/************Warp Disparity*****************************/
__global__ void
W_N_HammingDistanceKernel(cost_t* center_transform,
                          cost_t* l_c_transform,cost_t *r_c_transform,cost_t *b_c_transform,cost_t *t_c_transform,
                          cost_t *top_left_c_transform,cost_t *top_right_c_transform, cost_t *bottom_left_c_transform,cost_t *bottom_right_c_transform,
                          uint8_t *c_pic,uint8_t *l_pic,uint8_t *r_pic,uint8_t *b_pic,uint8_t *t_pic,
                          uint8_t *top_left_pic,uint8_t *top_righth_pic,uint8_t *bottom_left_pic, uint8_t *bottom_right_pic,
                          uint32_t *d_cost,
                          float* right_f_vec,float* right_baseline_vec,
                          float* left_f_vec,float* left_baseline_vec
                          );

__global__ void
N_N_HammingDistanceKernel(cost_t* center_transform,
                          cost_t* l_c_transform,
                         // uint8_t *c_pic,
                          uint32_t *d_cost);

__global__ void SumCost(uint32_t *l_cost,uint32_t *r_cost,uint32_t *t_cost,uint32_t *b_cost,uint32_t *sum_cost);
#endif
