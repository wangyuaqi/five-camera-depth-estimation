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

#include "hamming_cost.h"

//d_transform0, d_transform1, d_cost, rows, cols
__global__ void
HammingDistanceCostKernel (  const cost_t *d_transform0, const cost_t *d_transform1,
		uint8_t *d_cost, const int rows, const int cols ) {
	//const int Dmax=   blockDim.x;  // Dmax is CTA size
	const int y=      blockIdx.x;  // y is CTA Identifier
	const int THRid = threadIdx.x; // THRid is Thread Identifier
   // printf("blockId::%d\n",y);
    __shared__ cost_t SharedMatch[MAX_DISPARITY];
    __shared__ cost_t SharedBase [MAX_DISPARITY/2];

    //printf("threadId::%d\n",THRid);
    SharedMatch [MAX_DISPARITY/2+THRid] = d_transform1[y*cols+0];  // init position

    int n_iter = 2*cols/MAX_DISPARITY;
	for (int ix=0; ix<n_iter; ix++) {
        const int x = ix*MAX_DISPARITY/2;
        SharedMatch [THRid]      = SharedMatch [THRid + MAX_DISPARITY/2];
        SharedMatch [THRid+MAX_DISPARITY/2] = d_transform1 [y*cols+x+THRid];
		SharedBase  [THRid]      = d_transform0 [y*cols+x+THRid];

		__syncthreads();
        for (int i=0; i<MAX_DISPARITY/2; i++) {
			const cost_t base  = SharedBase [i];
            const cost_t match = SharedMatch[(MAX_DISPARITY/2-1-THRid)+1+i];
            d_cost[(y*cols+x+i)*MAX_DISPARITY+THRid] = 30;
            //*************
            d_cost[(y*cols+x+i)*MAX_DISPARITY+MAX_DISPARITY/2+THRid] = popcount( base ^ match );
		}
		__syncthreads();
	}
	// For images with cols not multiples of MAX_DISPARITY
    const int x = MAX_DISPARITY/2*(2*cols/MAX_DISPARITY);
	const int left = cols-x;
	if(left > 0) {
        SharedMatch [THRid]      = SharedMatch [THRid + MAX_DISPARITY/2];
		if(THRid < left) {
            SharedMatch [THRid+MAX_DISPARITY/2] = d_transform1 [y*cols+x+THRid];
			SharedBase  [THRid]      = d_transform0 [y*cols+x+THRid];
		}

		__syncthreads();
		for (int i=0; i<left; i++) {
			const cost_t base  = SharedBase [i];
            const cost_t match = SharedMatch[(MAX_DISPARITY/2-1-THRid)+1+i];
            d_cost[(y*cols+x+i)*MAX_DISPARITY+THRid] = 300;
            //************
            d_cost[(y*cols+x+i)*MAX_DISPARITY+THRid+MAX_DISPARITY/2] = popcount( base ^ match );

		}
		__syncthreads();
	}
}


//**********
__global__ void
HammingDistanceCostKernel_Z (  const cost_t *d_transform0, const cost_t *d_transform1,
        uint8_t *d_cost, const int rows, const int cols ) {
    //const int Dmax=   blockDim.x;  // Dmax is CTA size
    const int y=      blockIdx.x;  // y is CTA Identifier
    const int THRid = threadIdx.x; // THRid is Thread Identifier
    //printf("blockId::%d\n",y);
    __shared__ cost_t SharedMatch_Z[MAX_DISPARITY];
    __shared__ cost_t SharedBase_Z [MAX_DISPARITY/2];

    //printf("new hamming distance%u::\n",d_transform0[0]);
    //printf("threadId::%d\n",THRid);
    SharedMatch_Z [THRid] = d_transform1[y*cols+cols-1];  // init position
    //int min_pop=100;
    int n_iter = 2*cols/MAX_DISPARITY;
    for (int ix=n_iter; ix>0; ix--) {
        const int x = ix*MAX_DISPARITY/2;
        SharedMatch_Z [THRid+MAX_DISPARITY/2]= SharedMatch_Z [THRid];
        SharedMatch_Z [MAX_DISPARITY/2-1-THRid] = d_transform1 [y*cols+x-THRid-1];
        SharedBase_Z  [MAX_DISPARITY/2-1-THRid]      = d_transform0 [y*cols+x-THRid-1];

        __syncthreads();
        for (int i=0; i<MAX_DISPARITY/2; i++) {
            const cost_t base  = SharedBase_Z [i];
            //const cost_t match = SharedMatch_Z[(MAX_DISPARITY/2-1-THRid)+1+i];
            const cost_t match = SharedMatch_Z[i+THRid];
            //if(min_pop>popcount( base ^ match ))
                //min_pop=popcount( base ^ match );
            d_cost[(y*cols+x-MAX_DISPARITY/2+i)*MAX_DISPARITY+MAX_DISPARITY/2-1-THRid] = popcount( base ^ match );
            //*************
            //d_cost[(y*cols+x-MAX_DISPARITY/2+i)*MAX_DISPARITY+MAX_DISPARITY-THRid] = 40;
            //d_cost[(y*cols+x-MAX_DISPARITY/2+i)*MAX_DISPARITY+MAX_DISPARITY/2+THRid] = 40;
        }
        __syncthreads();
    }
   // printf("min_pop::%d\n",min_pop);
    // For images with cols not multiples of MAX_DISPARITY
    const int x = MAX_DISPARITY/2*(2*cols/MAX_DISPARITY);
    //const int left = cols-x;
    const int left=cols-x;
    /*if(left > 0) {
        SharedMatch_Z [THRid]      = SharedMatch_Z [THRid + MAX_DISPARITY/2];
        if(THRid < left) {
            SharedMatch_Z [THRid+MAX_DISPARITY/2] = d_transform1 [y*cols+x-THRid];
            SharedBase_Z  [THRid]      = d_transform0 [y*cols+x-THRid];
        }

        __syncthreads();
        for (int i=0; i<left; i++) {
            const cost_t base  = SharedBase_Z [i];
            const cost_t match = SharedMatch_Z[(MAX_DISPARITY/2-1-THRid)+1+i];
            d_cost[(y*cols+x+i)*MAX_DISPARITY+THRid] = 300;
            //************
            d_cost[(y*cols+x+i)*MAX_DISPARITY+THRid+MAX_DISPARITY/2] = popcount( base ^ match );
        }
        __syncthreads();
    }*/
    if(left>0)
    {
        SharedMatch_Z [THRid]      = SharedMatch_Z [THRid + MAX_DISPARITY/2];
        if(THRid < left) {
            SharedMatch_Z [THRid+MAX_DISPARITY/2] = d_transform1 [y*cols+x-THRid];
            SharedBase_Z  [THRid]      = d_transform0 [y*cols+x-THRid];
        }

        __syncthreads();
        for (int i=0; i<left; i++) {
            const cost_t base  = SharedBase_Z [i];
            const cost_t match = SharedMatch_Z[(MAX_DISPARITY/2-1-THRid)+1+i];
            //d_cost[(y*cols+x+i)*MAX_DISPARITY+THRid] = 300;
            //************
            d_cost[(y*cols+x+i)*MAX_DISPARITY+THRid+MAX_DISPARITY/2] = popcount( base ^ match );
        }
        __syncthreads();
    }
}
