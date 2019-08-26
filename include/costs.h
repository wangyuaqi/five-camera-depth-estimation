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

#ifndef COSTS_H_
#define COSTS_H_

#include <stdint.h>
#include "configuration.h"

__global__ void CenterSymmetricCensusKernelSM2(const uint8_t *im, const uint8_t *im2, cost_t *transform, cost_t *transform2, const uint32_t rows, const uint32_t cols);

__global__ void N_CenterSymmetricCnesusKernelSM2(uint8_t **l_im, uint8_t **r_im,cost_t **l_transform,cost_t **r_transform);

__device__ void C_CenterSymmetricCensusKernelSM2( uint8_t *im,  uint8_t *im2, cost_t *transform, cost_t *transform2, const uint32_t rows, const uint32_t cols);

//warp
__global__ void ShiftImage(uint8_t **center_im, uint8_t **center_warp,int px,int py);

__global__ void N_ShiftImage(uint8_t **left_img,uint8_t **left_warp,uint8_t **right_img,uint8_t **right_warp,
                              uint8_t **top_img, uint8_t **top_warp,uint8_t **bottom_img,uint8_t **bottom_warp,
                             uint8_t **top_left_img, uint8_t **top_left_warp, uint8_t **top_right_img, uint8_t **top_right_warp,
                             uint8_t **bottom_left_img, uint8_t **bottom_left_warp, uint8_t **bottom_right_img, uint8_t **bottom_right_warp);
#endif /* COSTS_H_ */
__device__ void WarpImage(const uint8_t *center_im,uint8_t *n_image,float dx,float dy);

