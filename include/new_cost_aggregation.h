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

#ifndef NEW_COST_AGGREGATION_H_
#define NEW_COST_AGGREGATION_H_

#define ITER_COPY			0
#define ITER_NORMAL			1

#define MIN_COMPUTE			0
#define MIN_NOCOMPUTE		1

#define DIR_UPDOWN			0
#define DIR_DOWNUP			1
#define DIR_LEFTRIGHT		2
#define DIR_RIGHTLEFT		3

template<int add_col, bool recompute, bool join_dispcomputation>
__device__ __forceinline__ void N_CostAggregationGenericIndexesIncrement(int *index, int *index_im, int *col, const int add_index, const int add_imindex) {
    *index += add_index;
    if(recompute || join_dispcomputation) {
        *index_im += add_imindex;
        if(recompute) {
            *col += add_col;
        }
    }
}

template<int add_index, bool recompute, bool join_dispcomputation>
__device__ __forceinline__ void N_CostAggregationDiagonalGenericIndexesIncrement(int *index, int *index_im, int *col, const int cols, const int initial_row, const int i, const int dis) {
    *col += add_index;
    if(add_index > 0 && *col >= cols) {
        *col = 0;
    } else if(*col < 0) {
        *col = cols-1;
    }
    *index = abs(initial_row-i)*cols*MAX_DISPARITY+*col*MAX_DISPARITY+dis;
    if(recompute || join_dispcomputation) {
        *index_im = abs(initial_row-i)*cols+*col;
    }
}

template<class T, int iter_type, int min_type, int dir_type, bool first_iteration, bool recompute, bool join_dispcomputation>
__device__ __forceinline__ void N_CostAggregationGenericIteration(int index, int index_im, int col, uint32_t *old_values, int *old_value1, int *old_value2, int *old_value3, int *old_value4, uint32_t *min_cost, uint32_t *min_cost_p2, uint32_t* d_cost, uint32_t *d_L, const int p1_vector, const int p2_vector, const T *_d_transform0, const T *_d_transform1, const int lane, const int MAX_PAD, const int dis, T *rp0, T *rp1, T *rp2, T *rp3, uint8_t* __restrict__ d_disparity, const uint32_t* d_L0, const uint32_t* d_L1, const uint32_t* d_L2, const uint32_t* d_L3, const uint32_t* d_L4, const uint32_t* d_L5, const uint32_t* d_L6) {
    const T __restrict__ *d_transform0 = _d_transform0;
    const T __restrict__ *d_transform1 = _d_transform1;
    uint32_t costs, next_dis, prev_dis;
    int prev_dis1,prev_dis2,prev_dis3,prev_dis4;
    int next_dis4,next_dis3,next_dis2,next_dis1;
    uint32_t costs_1,costs_2,costs_3,costs_4;
    if(iter_type == ITER_NORMAL) {
        // First shuffle
        /*int prev_dis1 = shfl_up_32(*old_value4, 1);//get last thread old_value4 as the pre_dis1
        if(lane == 0) {
            prev_dis1 = MAX_PAD;
        }

        // Second shuffle
        int next_dis4 = shfl_down_32(*old_value1, 1);

        if(lane == 15) {
                    next_dis4 = MAX_PAD;
        }*/

        // Shift + rotate
        //next_dis = __funnelshift_r(next_dis4, *old_values, 8);
       // next_dis = __byte_perm(*old_values, next_dis4, 0x4321);
        //prev_dis = __byte_perm(*old_values, prev_dis1, 0x2104);

        //******orign font*******
        //******xiugai********
        /*next_dis=shfl_up_32(*old_values, 1);
        prev_dis=shfl_down_32(*old_values, 1);
        if(lane==0)
            prev_dis=MAX_PAD;

        if(lane==127)
        {
            next_dis=MAX_PAD;
            //printf("%u::%d\n",next_dis,MAX_PAD);
        }
       // printf("next_dis::%u::pre_dis::%u\n",next_dis_e,prev_dis_e);

        next_dis = next_dis + p1_vector;//(disparity+1)
        prev_dis = prev_dis + p1_vector;//(disparity-1)*/
        //printf("next_dis::%u::pre_dis::%u::p1::%u::p2::%u\n",next_dis,prev_dis,p1_vector,p2_vector);
        //second xiugai************
        prev_dis1 = shfl_up_32(*old_value4, 1);
        //if(prev_dis1>1000||prev_dis1<0)
            //printf("%d\n",prev_dis1);
        prev_dis2=*old_value1;
        prev_dis3=*old_value2;
        prev_dis4=*old_value3;

        if(lane==0)
        {
            //printf("%d::prev_dis1::%d\n",MAX_PAD,prev_dis1);
            prev_dis1=MAX_PAD;
        }

        prev_dis1+=p1_vector;
        prev_dis2+=p1_vector;
        prev_dis3+=p1_vector;
        prev_dis4+=p1_vector;

        next_dis4 = shfl_down_32(*old_value1, 1);
        //printf("%d\n",next_dis4);
        next_dis3=*old_value4;
        next_dis2=*old_value3;
        next_dis1=*old_value2;

        if(lane==31)
            next_dis4=MAX_PAD;
        next_dis4+=p1_vector;
        next_dis3+=p1_vector;
        next_dis2+=p1_vector;
        next_dis1+=p1_vector;

        //printf("%d::%u\n",next_dis3,*old_value4);

    }
    if(recompute) {
        /*const int dif = col - dis;
        if(dir_type == DIR_LEFTRIGHT) {
            if(lane == 0) {
                // lane = 0 is dis = 0, no need to subtract dis
                *rp0 = d_transform1[index_im];
            }
        } else if(dir_type == DIR_RIGHTLEFT) {
            // First iteration, load D pixels
            if(first_iteration) {
                const uint4 right = reinterpret_cast<const uint4*>(&d_transform1[index_im-dis-3])[0];
                *rp3 = right.x;
                *rp2 = right.y;
                *rp1 = right.z;
                *rp0 = right.w;
            } else if(lane == 31 && dif >= 3) {
                *rp3 = d_transform1[index_im-dis-3];
            }
        } else {


            __shared__ T right_p[128+32];
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int block_imindex = index_im - warp_id + 2;
            const int rp_index = warp_id*WARP_SIZE+lane;
            const int col_cpy = col-warp_id+2;
            right_p[rp_index] = ((col_cpy-(129-rp_index)) >= 0) ? d_transform1[block_imindex-(129-rp_index)] : 0;
            right_p[rp_index+64] = ((col_cpy-(129-rp_index-64)) >= 0) ? d_transform1[block_imindex-(129-rp_index-64)] : 0;
            //right_p[rp_index+128] = ld_gbl_cs(&d_transform1[block_imindex-(129-rp_index-128)]);
            if(warp_id == 0) {
                right_p[128+lane] = ld_gbl_cs(&d_transform1[block_imindex-(129-lane)]);
            }
            __syncthreads();

            const int px = MAX_DISPARITY+warp_id-dis-1;
            *rp0 = right_p[px];
            *rp1 = right_p[px-1];
            *rp2 = right_p[px-2];
            *rp3 = right_p[px-3];
        }
        const T left_pixel = d_transform0[index_im];
        *old_value1 = popcount(left_pixel ^ *rp0);
        *old_value2 = popcount(left_pixel ^ *rp1);
        *old_value3 = popcount(left_pixel ^ *rp2);
        *old_value4 = popcount(left_pixel ^ *rp3);
        if(iter_type == ITER_COPY) {
            *old_values = uchars_to_uint32(*old_value1, *old_value2, *old_value3, *old_value4);
        } else {
            costs = uchars_to_uint32(*old_value1, *old_value2, *old_value3, *old_value4);
        }
        // Prepare for next iteration
        if(dir_type == DIR_LEFTRIGHT) {
            *rp3 = shfl_up_32(*rp3, 1);
        } else if(dir_type == DIR_RIGHTLEFT) {
            *rp0 = shfl_down_32(*rp0, 1);
        }*/
    }//realy use
    else {
        if(iter_type == ITER_COPY) {
            //*old_values = ld_gbl_ca(reinterpret_cast<const uint32_t*>(&d_cost[index]));
            *old_value1 =(ld_gbl_ca((&d_cost[index])));
            *old_value2 =(ld_gbl_ca((&d_cost[index+1])));
            *old_value3 =(ld_gbl_ca((&d_cost[index+2])));
            *old_value4 =(ld_gbl_ca((&d_cost[index+3])));
            /**old_value1=d_cost[index];
            *old_value2=d_cost[index+1];
            *old_value3=d_cost[index+2];
            *old_value4=d_cost[index+3];*/
            //printf("old_values::%u::%d\n",*old_values,index);
        } else {
           // costs = ld_gbl_ca(reinterpret_cast<const uint32_t*>(&d_cost[index]));
            costs_1=ld_gbl_ca((&d_cost[index]));
            costs_2=ld_gbl_ca((&d_cost[index+1]));
            costs_3=ld_gbl_ca((&d_cost[index+2]));
            costs_4=ld_gbl_ca((&d_cost[index+3]));

            /*costs_1=d_cost[index];
            costs_2=d_cost[index+1];
            costs_3=d_cost[index+2];
            costs_4=d_cost[index+3];*/
           // printf("costs::%u::%u::%u::%u\n",costs_1,costs_2,costs_3,costs_4);
        }
    }

    if(iter_type == ITER_NORMAL) {
        /*const uint32_t min1 = __vminu4(*old_values, prev_dis);//old_values:Lr(p-r,d),prev_dis:Lr(p-r,d-1)+P1
        const uint32_t min2 = __vminu4(next_dis, *min_cost_p2);//next_dis:Lr(p-r,d+1)+P1,min(Lr(p-r,i))+P2
        const uint32_t min_prev = __vminu4(min1, min2);//min(....)
        *old_values = costs + (min_prev - *min_cost);//Lr(p,d)=C(p,d)+min(....)-min(Lr(p,k)),min_cost=min(Lr(p,k))*/
        //old_values_1
        //printf("%u::%d::%d::%d::%d\n",*min_cost,*old_value1,*old_value2,*old_value3,*old_value4);
        int min1;
        int min2;
        int min_prev;
        //int z=*old_value1;
        min1=min(*old_value1, prev_dis1);
        //min2=__vminu4((uint32_t)next_dis1, *min_cost_p2);
        min2=min(next_dis1, (int)*min_cost_p2);
        min_prev = min(min1, min2);
        //printf("%d::%u::%u\n",next_dis1,*min_cost_p2,min2);
        /*min1=min(*old_value1, prev_dis1);
        min2=min(next_dis1, *min_cost_p2);
        min_prev = min(min1, min2);*/
        //printf("%d::%d::%d::%d::%u::r::%u\n",*old_value1,prev_dis1,min1,next_dis1,*min_cost_p2,*min_cost);

        *old_value1 = int(costs_1 + ((uint32_t)min_prev - *min_cost));
        //printf("%u::%d::%u\n",((uint32_t)min_prev - *min_cost),min_prev,*min_cost);
        //if(*old_value1>4000||*old_value1<0)
            //printf("%d::%d::%d::%d::%u::::min1::%u::min2::%u::min_orev::%u::r::%u\n",z,prev_dis1,min1,next_dis1,*min_cost_p2,min1,min2,min_prev,*min_cost);
        //************
        //old_values_2
        min1=min(*old_value2, prev_dis2);
        min2=min(next_dis2, (int)*min_cost_p2);
        min_prev = min(min1, min2);
        *old_value2 = int(costs_2 + ((uint32_t)min_prev - *min_cost));
        //if(((uint32_t)min_prev - *min_cost)<90)
       // printf("%u::%d::%u::%u::%d\n",((uint32_t)min_prev - *min_cost),min_prev,*min_cost,costs_2,*old_value2);
        //************
        //old_values_3
        min1=min(*old_value3, prev_dis3);
        min2=min(next_dis3, (int)*min_cost_p2);

        min_prev = min(min1, min2);
        *old_value3 = int(costs_3 + ((uint32_t)min_prev - *min_cost));
        //************
        //old_values_4
        min1=min(*old_value4, prev_dis4);
        min2=min(next_dis4, (int)*min_cost_p2);
        min_prev = min(min1, min2);
        *old_value4 = int(costs_4 + ((uint32_t)min_prev - *min_cost));
        //if(*old_value1<0||*old_value2<0||*old_value3<0||*old_value4<0)
           // printf("end*************************\n");
        //printf("%d::%u\n",*old_value4,costs_4 + (min_prev - *min_cost));
        //************
        //if(*old_value1<0||*old_value2<0||*old_value3<0||*old_value4<0)
         // printf("%d::%d::%d::%d\n",*old_value1,*old_value2,*old_value3,*old_value4);
    }
    if(iter_type == ITER_NORMAL || !recompute) {
        //uint32_to_uchars(*old_values, old_value1, old_value2, old_value3, old_value4);
    }
    //join_discomputation=false not work
    if(join_dispcomputation) {
        /*const uint32_t L0_costs = *((uint32_t*) (d_L0+index));
        const uint32_t L1_costs = *((uint32_t*) (d_L1+index));
        const uint32_t L2_costs = *((uint32_t*) (d_L2+index));
        #if PATH_AGGREGATION == 8
            const uint32_t L3_costs = *((uint32_t*) (d_L3+index));
            const uint32_t L4_costs = *((uint32_t*) (d_L4+index));
            const uint32_t L5_costs = *((uint32_t*) (d_L5+index));
            const uint32_t L6_costs = *((uint32_t*) (d_L6+index));
        #endif

        int l0_x, l0_y, l0_z, l0_w;
        int l1_x, l1_y, l1_z, l1_w;
        int l2_x, l2_y, l2_z, l2_w;
        #if PATH_AGGREGATION == 8
            int l3_x, l3_y, l3_z, l3_w;
            int l4_x, l4_y, l4_z, l4_w;
            int l5_x, l5_y, l5_z, l5_w;
            int l6_x, l6_y, l6_z, l6_w;
        #endif

        uint32_to_uchars(L0_costs, &l0_x, &l0_y, &l0_z, &l0_w);
        uint32_to_uchars(L1_costs, &l1_x, &l1_y, &l1_z, &l1_w);
        uint32_to_uchars(L2_costs, &l2_x, &l2_y, &l2_z, &l2_w);
        #if PATH_AGGREGATION == 8
            uint32_to_uchars(L3_costs, &l3_x, &l3_y, &l3_z, &l3_w);
            uint32_to_uchars(L4_costs, &l4_x, &l4_y, &l4_z, &l4_w);
            uint32_to_uchars(L5_costs, &l5_x, &l5_y, &l5_z, &l5_w);
            uint32_to_uchars(L6_costs, &l6_x, &l6_y, &l6_z, &l6_w);
        #endif*/

        /*#if PATH_AGGREGATION == 8
            const uint16_t val1 = l0_x + l1_x + l2_x + l3_x + l4_x + l5_x + l6_x + *old_value1;
            const uint16_t val2 = l0_y + l1_y + l2_y + l3_y + l4_y + l5_y + l6_y + *old_value2;
            const uint16_t val3 = l0_z + l1_z + l2_z + l3_z + l4_z + l5_z + l6_z + *old_value3;
            const uint16_t val4 = l0_w + l1_w + l2_w + l3_w + l4_w + l5_w + l6_w + *old_value4;
        #else
            const uint16_t val1 = l0_x + l1_x + l2_x + *old_value1;
            const uint16_t val2 = l0_y + l1_y + l2_y + *old_value2;
            const uint16_t val3 = l0_z + l1_z + l2_z + *old_value3;
            const uint16_t val4 = l0_w + l1_w + l2_w + *old_value4;
        #endif

        int min_idx1 = dis;
        uint16_t min1 = val1;
        if(val1 > val2) {
            min1 = val2;
            min_idx1 = dis+1;
        }

        int min_idx2 = dis+2;
        uint16_t min2 = val3;
        if(val3 > val4) {
            min2 = val4;
            min_idx2 = dis+3;
        }

        uint16_t minval = min1;
        int min_idx = min_idx1;
        if(min1 > min2) {
            minval = min2;
            min_idx = min_idx2;
        }

        const int min_warpindex = warpReduceMinIndex(minval, min_idx);
        if(lane == 0) {
            //printf("minval::%u\n",minval);
            //printf("min_idx::%d\n",min_idx);
            d_disparity[index_im] = min_warpindex;
        }*/
        //printf("%u::%u::%u\n",L0_costs,L1_costs,L2_costs);
        //uint32_t l0_x, l0_y, l0_z, l0_w;
        //uint32_t l1_x, l1_y, l1_z, l1_w;
        //uint32_t l2_x, l2_y, l2_z, l2_w;
        /*const uint32_t L0_costs = d_L0[index];
        const uint32_t L1_costs = d_L1[index];
        const uint32_t L2_costs = d_L2[index];*/
        const uint32_t l0_x=d_L0[index],l0_y=d_L0[index+1],l0_z=d_L0[index+2],l0_w=d_L0[index+3];
        const uint32_t l1_x=d_L1[index],l1_y=d_L1[index+1],l1_z=d_L1[index+2],l1_w=d_L1[index+3];
        const uint32_t l2_x=d_L2[index],l2_y=d_L2[index+1],l2_z=d_L2[index+2],l2_w=d_L2[index+3];
        /*l0_x=ld_gbl_ca((&d_L0[index])),l0_y=ld_gbl_ca((&d_L0[index+1])),l0_z=ld_gbl_ca((&d_L0[index+2])),l0_w=ld_gbl_ca((&d_L0[index+3]));
        l1_x=ld_gbl_ca((&d_L1[index])),l1_y=ld_gbl_ca((&d_L1[index+1])),l1_z=ld_gbl_ca((&d_L1[index+2])),l1_w=ld_gbl_ca((&d_L1[index+3]));
        l2_x=ld_gbl_ca((&d_L2[index])),l2_y=ld_gbl_ca((&d_L2[index+1])),l2_z=ld_gbl_ca((&d_L2[index+2])),l2_w=ld_gbl_ca((&d_L2[index+3]));*/
        //printf("%u::%u::%u::%u\n",l0_x,l0_y,l0_z,l0_w);
        #if PATH_AGGREGATION == 8
           /*const uint32_t L3_costs = d_L3[index];
           const uint32_t L4_costs = d_L4[index];
           const uint32_t L5_costs = d_L5[index];
           const uint32_t L6_costs = d_L6[index];*/
           //uint32_t l3_x, l3_y, l3_z, l3_w;
           //uint32_t l4_x, l4_y, l4_z, l4_w;
           //uint32_t l5_x, l5_y, l5_z, l5_w;
           //uint32_t l6_x, l6_y, l6_z, l6_w;
           /*st_gbl_cs(reinterpret_cast<uint32_t*>(&d_L3[index],), l3_x);
           st_gbl_cs(reinterpret_cast<uint32_t*>(&d_L3[index+1],), l3_y);
           st_gbl_cs(reinterpret_cast<uint32_t*>(&d_L3[index+2],), l3_z);
           st_gbl_cs(reinterpret_cast<uint32_t*>(&d_L3[index+3],), l3_w);*/


           const uint32_t l3_x=d_L3[index],l3_y=d_L3[index+1],l3_z=d_L3[index+2],l3_w=d_L3[index+3];
           const uint32_t l4_x=d_L4[index],l4_y=d_L4[index+1],l4_z=d_L4[index+2],l4_w=d_L4[index+3];
           const uint32_t l5_x=d_L5[index],l5_y=d_L5[index+1],l5_z=d_L5[index+2],l5_w=d_L5[index+3];
           const uint32_t l6_x=d_L6[index],l6_y=d_L6[index+1],l6_z=d_L6[index+2],l6_w=d_L6[index+3];
           //printf("%d::%u\n",l3_x,d_L3[index]);
        #endif
        #if PATH_AGGREGATION == 8
           const uint32_t val1 = l0_x + l1_x + l2_x + l3_x + l4_x + l5_x + l6_x + uint32_t(*old_value1);
           const uint32_t val2 = l0_y + l1_y + l2_y + l3_y + l4_y + l5_y + l6_y + uint32_t(*old_value2);
           const uint32_t val3 = l0_z + l1_z + l2_z + l3_z + l4_z + l5_z + l6_z + uint32_t(*old_value3);
           const uint32_t val4 = l0_w + l1_w + l2_w + l3_w + l4_w + l5_w + l6_w + uint32_t(*old_value4);
        #else
           const uint32_t val1 = l0_x + l1_x + l2_x + uint32_t(*old_value1);
           const uint32_t val2 = l0_y + l1_y + l2_y + uint32_t(*old_value2);
           const uint32_t val3 = l0_z + l1_z + l2_z + uint32_t(*old_value3);
           const uint32_t val4 = l0_w + l1_w + l2_w + uint32_t(*old_value4);
        #endif

           int min_idx1 = dis;
           uint32_t min1 = val1;
           if(val1 > val2) {
               min1 = val2;
               min_idx1 = dis+1;
           }

           int min_idx2 = dis+2;
           uint32_t min2 = val3;
           if(val3 > val4) {
               min2 = val4;
               min_idx2 = dis+3;
           }

           uint32_t minval = min1;
           int min_idx = min_idx1;
           if(min1 > min2) {
               minval = min2;
               min_idx = min_idx2;
           }

           const int min_warpindex = warpReduceMinIndex(minval, min_idx);
           //if(minval>1000)
               //printf("min::val::%u\n",minval);
           //const int s_val=warpReduceMin(minval);
           //if(val1>10000||val2>10000||val3>10000||val4>10000)
               //printf("%u::%u::%u::%u::%d::%d\n",val1,val2,val3,val4,s_val,minval);
           //if(minval<=s_val)
               //printf("%d::%d\n",minval,s_val);
           if(lane == 0) {
               //if(index_im/512==0&&index_im%512==0)
                   //printf("minval::%u::col::%d::row::%d\n",min_warpindex,index_im/512,index_im%512);
               //printf("min_idx::%d\n",min_idx);
               d_disparity[index_im] = min_warpindex;
               //d_disparity[index_im]=255;
              // if(min_warpindex>125||min_warpindex<10)
                  // printf("%d::%u::%u::%u::col::%drow::%d\n",min_warpindex,minval,min_idx,*min_cost,index_im%512,index_im/512);
              // printf("min_idx::%u::%d\n",d_disparity[index_im],min_warpindex);
           }
//*****xiugai font*******
    } else {

        //st_gbl_cs(reinterpret_cast<uint32_t*>(&d_L[index]), *old_values);
       /* st_gbl_cs(reinterpret_cast<uint32_t*>(&d_L[index]), uint32_t(*old_value1));
        st_gbl_cs(reinterpret_cast<uint32_t*>(&d_L[index+1]), uint32_t(*old_value2));
        st_gbl_cs(reinterpret_cast<uint32_t*>(&d_L[index+2]), uint32_t(*old_value3));
        st_gbl_cs(reinterpret_cast<uint32_t*>(&d_L[index+3]), uint32_t(*old_value4));*/
        st_gbl_cs((&d_L[index]), uint32_t(*old_value1));
        st_gbl_cs((&d_L[index+1]), uint32_t(*old_value2));
        st_gbl_cs((&d_L[index+2]), uint32_t(*old_value3));
        st_gbl_cs((&d_L[index+3]), uint32_t(*old_value4));
        /*d_L[index]=*old_value1;
        d_L[index+1]=*old_value2;
        d_L[index+2]=*old_value3;
        d_L[index+3]=*old_value4;*/

        //printf("%d::%u\n",*old_value1,d_L[index]);
        //if(*old_value1<0)
       // d_L[index]=*old_values;
    }
    if(min_type == MIN_COMPUTE) {
        //update min cost
        //if(*old_value2>4000||*old_value2<0)
           // printf("%d\n",*old_value1);
        int min_cost_scalar = min(min(*old_value1, *old_value2), min(*old_value3, *old_value4));
        int r=warpReduceMin(min_cost_scalar);
        //printf("%d::%d\n",r,min_cost_scalar);
        //if(min_cost_scalar<r)
            //printf("false\n");
        *min_cost = uint32_t(r);
        *min_cost_p2 = *min_cost + uint32_t(p2_vector);
        //if(*min_cost>1000)
        //printf("%u::%d::%d::%d::%d\n",*min_cost,*old_value1,*old_value2,*old_value3,*old_value4);
       // if(*old_value2>4000)
           // printf("%u\n",*old_value1);
        //*min_cost = uint32_t(warpReduceMin((int)*old_values));
        //*min_cost_p2 = *min_cost + p2_vector;
       // if(min_cost_scalar<0)
          // printf("min::%d::%d::%d::%d\n",*old_value1,*old_value2,*old_value3,*old_value4);
        //if(*min_cost>1000)
        //printf("%u::%u::%d\n",*min_cost,*min_cost_p2,r);
        //xiugai font***

    }
}

template<class T, int add_col, int dir_type, bool recompute, bool join_dispcomputation>
__device__ __forceinline__ void N_CostAggregationGeneric(uint32_t* d_cost, uint32_t *d_L, const int P1, const int P2, const int initial_row, const int initial_col, const int max_iter, const int cols, int add_index, const T *_d_transform0, const T *_d_transform1, const int add_imindex, uint8_t* __restrict__ d_disparity, const uint32_t* d_L0, const uint32_t* d_L1, const uint32_t* d_L2, const uint32_t* d_L3, const uint32_t* d_L4, const uint32_t* d_L5, const uint32_t* d_L6) {
    const int lane = threadIdx.x % WARP_SIZE;
    //const int dis = 4*lane;
    const int dis = lane*4;
    //printf("%d\n",dis);
    //xiugai***font
    //if(dis>123)
       // printf("%d\n",dis);
    int index = initial_row*cols*MAX_DISPARITY+initial_col*MAX_DISPARITY+dis;
    int col, index_im;
    if(recompute || join_dispcomputation) {
        if(recompute) {
            col = initial_col;
        }
        index_im = initial_row*cols+initial_col;
    }

    /*const int MAX_PAD = UCHAR_MAX-P1;
    const uint32_t p1_vector = uchars_to_uint32(P1, P1, P1, P1);
    const uint32_t p2_vector = uchars_to_uint32(P2, P2, P2, P2);*/
    //printf("%d\n",MAX_PAD);
    const int MAX_PAD = 300000-P1;
    const uint32_t p1_vector=uint32_t(P1);
    const uint32_t p2_vector=uint32_t(P2);
    //printf("%u::%u\n",p1_vector,p2_vector);
    //xiugai***font
    int old_value1;
    int old_value2;
    int old_value3;
    int old_value4;
    uint32_t min_cost, min_cost_p2, old_values;
    T rp0, rp1, rp2, rp3;

    if(recompute) {
        /*if(dir_type == DIR_LEFTRIGHT) {
            N_CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            for(int i = 4; i < max_iter-3; i+=4) {
                N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
                N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
                N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
                N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
                N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
                N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
                N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
                N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            }
            N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
        } else if(dir_type == DIR_RIGHTLEFT) {
            N_CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            for(int i = 4; i < max_iter-3; i+=4) {
                N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
                N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
                N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
                N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
                N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
                N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
                N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
                N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            }
            N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
        } else {
            N_CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            for(int i = 1; i < max_iter; i++) {
                N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
                N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            }
            N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
        }*/
    }
    else {
        N_CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);

        for(int i = 1; i < max_iter; i++) {
            N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
        }
        N_CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
        N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}

template<int add_index, class T, int dir_type, bool recompute, bool join_dispcomputation>
__device__ __forceinline__ void N_CostAggregationDiagonalGeneric(uint32_t* d_cost, uint32_t *d_L, const int P1, const int P2, const int initial_row, const int initial_col, const int max_iter, const int col_nomin, const int col_copycost, const int cols, const T *_d_transform0, const T *_d_transform1, uint8_t* __restrict__ d_disparity, const uint32_t* d_L0, const uint32_t* d_L1, const uint32_t* d_L2, const uint32_t* d_L3, const uint32_t* d_L4, const uint32_t* d_L5, const uint32_t* d_L6) {
    const int lane = threadIdx.x % WARP_SIZE;
    //const int dis = 4*lane;
    const int dis = lane*4;
   // printf("%d\n",dis);
    int col = initial_col;
    int index = initial_row*cols*MAX_DISPARITY+initial_col*MAX_DISPARITY+dis;
    int index_im;
    if(recompute || join_dispcomputation) {
        index_im = initial_row*cols+col;
    }
    //const int MAX_PAD = UCHAR_MAX-P1;
    const int MAX_PAD=300000-P1;
    //printf("MAX_PAD::%d\n",MAX_PAD);
    //*****xiugai font****
    /*const uint32_t p1_vector = uchars_to_uint32(P1, P1, P1, P1);
    const uint32_t p2_vector = uchars_to_uint32(P2, P2, P2, P2);*/
    const uint32_t p1_vector=uint32_t(P1);
    const uint32_t p2_vector=uint32_t(P2);
    //if(p1_vector>3000||p2_vector>3000)
        //printf("%u::%u\n",p1_vector,p2_vector);
    //xiugai***font

    int old_value1;
    int old_value2;
    int old_value3;
    int old_value4;
    uint32_t min_cost, min_cost_p2, old_values;
    T rp0, rp1, rp2, rp3;

    N_CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    for(int i = 1; i < max_iter; i++) {
        N_CostAggregationDiagonalGenericIndexesIncrement<add_index, recompute, join_dispcomputation>(&index, &index_im, &col, cols, initial_row, i, dis);

        if(col == col_copycost) {
            N_CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            //printf("col::%d\n",col);
        } else {
            N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            //printf("col::%d\n",col);
        }
    }

    N_CostAggregationDiagonalGenericIndexesIncrement<add_index, recompute, join_dispcomputation>(&index, &index_im, &col, cols, max_iter, initial_row, dis);
    if(col == col_copycost) {
        N_CostAggregationGenericIteration<T, ITER_COPY, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    } else {
        N_CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}

template<class T>

__global__ void N_CostAggregationKernelDiagonalDownUpRightLeft(uint32_t* d_cost, uint32_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* __restrict__ d_disparity, const uint32_t* d_L0, const uint32_t* d_L1, const uint32_t* d_L2, const uint32_t* d_L3, const uint32_t* d_L4, const uint32_t* d_L5, const uint32_t* d_L6) {
    const int initial_col = cols - (blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE)) - 1;
    if(initial_col < cols) {
        const int initial_row = rows-1;
        const int add_index = -1;
        const int col_nomin = 0;
        const int col_copycost = cols-1;
        const int max_iter = rows-1;
        const bool recompute = false;
        const bool join_dispcomputation = false;

        N_CostAggregationDiagonalGeneric<add_index, T, DIR_DOWNUP, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, col_nomin, col_copycost, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}

template<class T>
__global__ void N_CostAggregationKernelDiagonalDownUpLeftRight(uint32_t* d_cost, uint32_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* __restrict__ d_disparity, const uint32_t* d_L0, const uint32_t* d_L1, const uint32_t* d_L2, const uint32_t* d_L3, const uint32_t* d_L4, const uint32_t* d_L5, const uint32_t* d_L6) {
    const int initial_col = cols - (blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE)) - 1;
    //printf("%d\n",initial_col);
    if(initial_col >= 0) {
        const int initial_row = rows-1;
        const int add_index = 1;
        const int col_nomin = cols-1;
        const int col_copycost = 0;
        const int max_iter = rows-1;
        const bool recompute = false;
        const bool join_dispcomputation = false;

        N_CostAggregationDiagonalGeneric<add_index, T, DIR_DOWNUP, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, col_nomin, col_copycost, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}

template<class T>

__global__ void N_CostAggregationKernelDiagonalUpDownRightLeft(uint32_t* d_cost, uint32_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* __restrict__ d_disparity, const uint32_t* d_L0, const uint32_t* d_L1, const uint32_t* d_L2, const uint32_t* d_L3, const uint32_t* d_L4, const uint32_t* d_L5, const uint32_t* d_L6) {
    const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if(initial_col < cols) {
        const int initial_row = 0;
        const int add_index = -1;
        const int col_nomin = 0;
        const int col_copycost = cols-1;
        const int max_iter = rows-1;
        const bool recompute = false;
        const bool join_dispcomputation = PATH_AGGREGATION == 8;

        N_CostAggregationDiagonalGeneric<add_index, T, DIR_UPDOWN, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, col_nomin, col_copycost, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}

template<class T>

__global__ void N_CostAggregationKernelDiagonalUpDownLeftRight(uint32_t* d_cost, uint32_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* __restrict__ d_disparity, const uint32_t* d_L0, const uint32_t* d_L1, const uint32_t* d_L2, const uint32_t* d_L3, const uint32_t* d_L4, const uint32_t* d_L5, const uint32_t* d_L6) {
    const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if(initial_col < cols) {
        const int initial_row = 0;
        const int add_index = 1;
        const int col_nomin = cols-1;
        const int col_copycost = 0;
        const int max_iter = rows-1;
        const bool recompute = false;
        const bool join_dispcomputation = false;

        N_CostAggregationDiagonalGeneric<add_index, T, DIR_UPDOWN, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, col_nomin, col_copycost, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}

template<class T>

__global__ void N_CostAggregationKernelLeftToRight(uint32_t* d_cost, uint32_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* __restrict__ d_disparity, const uint32_t* d_L0, const uint32_t* d_L1, const uint32_t* d_L2, const uint32_t* d_L3, const uint32_t* d_L4, const uint32_t* d_L5, const uint32_t* d_L6) {
    const int initial_row = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if(initial_row < rows) {
        const int initial_col = 0;
        const int add_index = MAX_DISPARITY;
        const int add_imindex = 1;
        const int max_iter = cols-1;
        const int add_col = 1;
        //const bool recompute = true;
        const bool recompute=false;
        const bool join_dispcomputation = false;

        N_CostAggregationGeneric<T, add_col, DIR_LEFTRIGHT, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}

template<class T>

__global__ void N_CostAggregationKernelRightToLeft(uint32_t* d_cost, uint32_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* __restrict__ d_disparity, const uint32_t* d_L0, const uint32_t* d_L1, const uint32_t* d_L2, const uint32_t* d_L3, const uint32_t* d_L4, const uint32_t* d_L5, const uint32_t* d_L6) {
    const int initial_row = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if(initial_row < rows) {
        const int initial_col = cols-1;
        const int add_index = -MAX_DISPARITY;
        const int add_imindex = -1;
        const int max_iter = cols-1;
        const int add_col = -1;
        //const bool recompute = true;
        const bool recompute=false;
        const bool join_dispcomputation = false;

        N_CostAggregationGeneric<T, add_col, DIR_RIGHTLEFT, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}

template<class T>
__global__ void N_CostAggregationKernelDownToUp(uint32_t* d_cost, uint32_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* __restrict__ d_disparity, const uint32_t* d_L0, const uint32_t* d_L1, const uint32_t* d_L2, const uint32_t* d_L3, const uint32_t* d_L4, const uint32_t* d_L5, const uint32_t* d_L6) {
    const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if(initial_col < cols) {
        const int initial_row = rows-1;
        const int add_index = -cols*MAX_DISPARITY;
        const int add_imindex = -cols;
        const int max_iter = rows-1;
        const int add_col = 0;
        const bool recompute = false;
        const bool join_dispcomputation = PATH_AGGREGATION == 4;

        N_CostAggregationGeneric<T, add_col, DIR_DOWNUP, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}

template<class T>
//__launch_bounds__(64, 16)
__global__ void N_CostAggregationKernelUpToDown(uint32_t* d_cost, uint32_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* __restrict__ d_disparity, const uint32_t* d_L0, const uint32_t* d_L1, const uint32_t* d_L2, const uint32_t* d_L3, const uint32_t* d_L4, const uint32_t* d_L5, const uint32_t* d_L6) {
    const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if(initial_col < cols) {
        const int initial_row = 0;
        const int add_index = cols*MAX_DISPARITY;
        const int add_imindex = cols;
        const int max_iter = rows-1;
        const int add_col = 0;
        const bool recompute = false;
        const bool join_dispcomputation = false;

        N_CostAggregationGeneric<T, add_col, DIR_UPDOWN, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}

#endif /* COST_AGGREGATION_H_ */
