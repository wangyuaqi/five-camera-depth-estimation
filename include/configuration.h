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

#ifndef CONFIGURATION_H_
#define CONFIGURATION_H_

#include <stdint.h>

#define LOG						false
#define WRITE_FILES				true

#define PATH_AGGREGATION	8
#define	MAX_DISPARITY		128//64//128//128//64
#define CENSUS_WIDTH	7//5//4//5//7//5//7//5//3//7  //9
#define CENSUS_HEIGHT		7//5//4//5//7//5//7//5//3//7

#define OCCLUDED_PIXEL		120
#define MISMATCHED_PIXEL	129

#define TOP				(CENSUS_HEIGHT-1)/2
#define LEFT			(CENSUS_WIDTH-1)/2

typedef uint32_t cost_t;
#define MAX_COST		30

#define BLOCK_SIZE					256
#define COSTAGG_BLOCKSIZE			GPU_THREADS_PER_BLOCK
#define COSTAGG_BLOCKSIZE_HORIZ		GPU_THREADS_PER_BLOCK

#define ABS_THRESH 3.0
#define REL_THRESH 0.05
//new define**************************************
#define NEW_MAX_DISPARITY 128
#define IMAGE_NUMBER 4//4   // per group image number
#define SELECT_IMAGE_NUM 1
#define SUM_IMAGE_NUM SELECT_IMAGE_NUM*4
//#define LOAD_IMAGE_NUMBER 9
#define SUM_LENGTH 4  //80*0.05
#define SUM_LENGTH2 8
#define SUM_LENGTH3 12
#define SUM_LENGTH4 16

#define IMG_HEIGHT 512
#define IMG_WIDTH 640
#define FLOAT_EPS 1e-10
#define WARP_DIS 1.0f//0.025f//0.027f
#define BEGIN_DIS 40
#define END_DIS 128
#define SHIFT_MIDDLE 0

#define BEGIN_DEPTH 10
#define END_DEPTH 130

typedef struct PairParam{
    float f_param;
    float baseline_param;
}CameraParam;

#endif /* CONFIGURATION_H_ */
