#    This file is part of sgm. (https://github.com/dhernandez0/sgm).
#
#    Copyright (c) 2016 Daniel Hernandez Juarez.

#    sgm is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    sgm is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 2.4)
project(sgm)
set(OpenCV_DIR /home/wangyuanqi/opencv-3.4.1/new_build/)

find_package(OpenCV 3 REQUIRED)
find_package( CUDA REQUIRED )

#find_package(PCL REQUIRED)

#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

#set(
#    CUDA_NVCC_FLAGS
#    ${CUDA_NVCC_FLAGS};
#    -O3 -lineinfo
#    -gencode=arch=compute_30,code=sm_30
#    -gencode=arch=compute_35,code=sm_35
#    -gencode=arch=compute_50,code=sm_50
#    -gencode=arch=compute_52,code=sm_52
#    -gencode=arch=compute_61,code=sm_61
#    -gencode=arch=compute_70,code=sm_70
#    )
set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")
INCLUDE_DIRECTORIES("include/")
cuda_add_executable(
    sgm
        src/n_hamming_cost.cu src/disparity_method.cu src/debug.cu src/costs.cu src/n_hamming_cost.cu src/new_main.cu src/depth_compute.cu
    src/image_warp.cu src/median_filter.cu src/hamming_cost.cu)
add_executable(sgm include/configuration.h include/new_cost_aggregation.h include/costs.h include/debug.h include/disparity_method.h include/hamming_cost.h include/median_filter.h include/util.h include/patch_match.h include/ImageTool.h src/ImageTool.cpp)
add_executable(sgm src/ImageTool.cpp include/stereoRectified.h src/stereoRectified.cpp src/map.cpp include/stereoRectifyParams.h include/depth_compute.h include/image_warp.h)

target_link_libraries( sgm ${OpenCV_LIBS})
#target_link_libraries( sgm ${PCL_LIBRARIES})

