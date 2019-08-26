#include "image_warp.h"
#include "configuration.h"


__device__ void ImageWarp(const uint8_t *center_im,uint8_t *n_image,float dx,float dy)
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
// Do Image Shift
__global__ void Image_Shift(uint8_t *origin_image, uint8_t **warp_image,float *dis_list,
                            int px, int py)
{
    const int b_y=blockIdx.y;
    int image_n=b_y;

    int dis_n =  b_y;//current dis_label
    float now_dis = dis_list[dis_n];

    float d_x,d_y;
    d_y=py*now_dis,d_x=px*now_dis;
    ImageWarp(origin_image,warp_image[dis_n],d_x,d_y);
}

__device__ uint32_t CostCompute(float map_center_x, float map_center_y, int image_width, int image_height,
                      cost_t *center_census, cost_t **stereo_census, int dis_index){

    //printf("matrix::%f::%f\n",map_center_x,map_center_y);
    int left_map_x = int(map_center_x);
    int bottom_map_y = int(map_center_y);

    int right_map_x = left_map_x+1;
    int top_map_y = bottom_map_y+1;

    float right_map_x_dis = map_center_x-left_map_x;
    float left_map_x_dis = 1-right_map_x_dis;

    float top_map_y_dis = map_center_y-bottom_map_y;
    //printf("%f:::::%d\n",map_center_y,bottom_map_y);
    float bottom_map_y_dis = 1-top_map_y_dis;
    //float sum_cost = 0;
    int max_cost = 300;
    int t_l_cost=max_cost, t_r_cost=max_cost,b_l_cost=max_cost,b_r_cost=max_cost;
    int census_index;
    if(top_map_y<image_height){
        census_index = top_map_y * image_width+ left_map_x;
        t_l_cost = popcount(center_census[census_index]^stereo_census[dis_index][census_index]);
        if(right_map_x<image_width) {
            census_index = top_map_y * image_width + right_map_x;
            t_r_cost = popcount(center_census[census_index]^stereo_census[dis_index][census_index]);
        }
    }

    census_index = bottom_map_y * image_width+ left_map_x;
    b_l_cost = popcount(center_census[census_index]^stereo_census[dis_index][census_index]);
    if(right_map_x<image_width) {
        census_index = bottom_map_y * image_width + right_map_x;
        b_r_cost = popcount(center_census[census_index]^stereo_census[dis_index][census_index]);
    }

    //printf("%f::%f::%f::%f\n",top_map_y_dis,right_map_x_dis,left_map_x_dis,bottom_map_y_dis);
    float sum_cost = top_map_y_dis*(right_map_x_dis*t_r_cost+left_map_x_dis*t_l_cost)+bottom_map_y_dis*(right_map_x_dis*b_r_cost+left_map_x_dis*b_l_cost);

    uint32_t result_cost = (uint32_t)sum_cost;
    //printf("%f::%d\n",sum_cost,result_cost);
    return result_cost;

}
__global__ void ComputeCostVolume(cost_t **left_left_census, cost_t *left_left_center_census,
                                  cost_t **left_center_census, cost_t *left_center_center_census,
                                  cost_t **right_center_census, cost_t *right_center_center_census,
                                  cost_t **right_right_census, cost_t *right_right_center_census,
                                  uint32_t *final_cost_volume,
                                  int image_width, int image_height,
                                  float *left_left_x,float *left_left_y,
                                  float *left_center_x,float *left_center_y,
                                  float *right_center_x,float *right_center_y,
                                  float *right_right_x,float *right_right_y,
                                  float *left_left_disparity, float *left_center_disparity,
                                  float *right_center_disparity, float *right_right_disparity

)
{
    __shared__ uint32_t costs[128];
    int b_x = blockIdx.x;
    int b_y = blockIdx.y;
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    int image_num=t_x;
    int dis_n = (b_x/image_width)*32+t_y;
    int t_num = t_y;
    int image_w = b_x%image_width;
    int image_h = b_y;
    uint32_t n_result = 0;
    int image_p=image_num;
    const int image_mount = 4;
    //int image_count=image_num%SELECT_IMAGE_NUM;
    //image_count = IMAGE_NUMBER-1-image_count;
    //int t_pixel_count=(dis_n)*IMG_WIDTH*IMG_HEIGHT+image_h*IMG_WIDTH+image_w;

    int center_index = image_h*image_width+image_w;

    cost_t left_left_center_cost,left_center_center_cost;
    cost_t right_center_center_cost,right_right_center_cost;


    int census_image_num = dis_n;

    //int map_center_x,map_center_y;
    //t_pixel_count= dis_num*IMG_WIDTH*IMG_HEIGHT+image_h*IMG_WIDTH+image_w;
    const uint32_t max_cost =300;

    int census_index;
    float map_center_x,map_center_y;

    float now_dis,new_map_x;

    int left_map_x,right_map_x;
    float left_map_x_dis,right_map_x_dis;
    int top_map_y,bottom_map_y;
    float top_map_y_dis, bottom_map_y_dis;
    int left_bottom_cost,right_bottom_cost,left_top_cost,right_top_cost;
    if(image_p==0)
    {
        //*****************
        //left left pair
        map_center_x = left_left_x[center_index];
        map_center_y = left_left_y[center_index];

        /*left_map_x = int(map_center_x);
        bottom_map_y = int(map_center_y);

        right_map_x = left_map_x+1;
        top_map_y = bottom_map_y+1;

        right_map_x_dis = map_center_x-left_map_x;
        left_map_x_dis = 1-left_map_x_dis;

        top_map_y_dis = map_center_y-bottom_map_y;
        bottom_map_y_dis = 1-top_map_y_dis;
        //map_center_x = image_w;
        //map_center_y = image_h;
        census_index = map_center_y*image_width+map_center_x;*/
        now_dis = left_left_disparity[dis_n];
        new_map_x = map_center_x + now_dis;
        if(map_center_x>=image_width||map_center_y>=image_height||new_map_x>image_width) {
            //printf("%d::%d\n",map_center_x,image_width);
            costs[t_num * image_mount + image_num] = max_cost;
        }
        else {

            /*left_left_center_cost = left_left_center_census[census_index];
            costs[t_num * image_mount + image_num] = popcount(
                    left_left_center_cost ^ (left_left_census[census_image_num][census_index]));*/
            costs[t_num * image_mount + image_num]= CostCompute(map_center_x,map_center_y,image_width,image_height,left_left_center_census,left_left_census,census_image_num);
            //printf("%d\n",costs[t_num * SUM_IMAGE_NUM + image_num]);
        }

        //********


    }
    else if(image_p==1)
    {
        //left center pair
        map_center_x = (left_center_x[center_index]);
        map_center_y = (left_center_y[center_index]);
        //map_center_x = image_w;
        //map_center_y = image_h;
        census_index = int(map_center_y)*image_width+int(map_center_x);

        now_dis = left_center_disparity[dis_n];
        new_map_x = map_center_x+now_dis;
        if(int(map_center_x)>=image_width||int(map_center_y)>=image_height||int(new_map_x)>=image_width)
            costs[t_num*SUM_IMAGE_NUM+image_num] = max_cost;
        else{
            /*left_center_center_cost = left_center_center_census[census_index];
            costs[t_num * SUM_IMAGE_NUM + image_num] = popcount(
                    left_center_center_cost ^ left_center_census[dis_n][census_index]);*/
            costs[t_num * image_mount + image_num]= CostCompute(map_center_x,map_center_y,image_width,image_height,left_center_center_census,left_center_census,census_image_num);
        }
    }
    else if(image_p==2)
    {

        //**************
        //right center pair
        map_center_x = (right_center_x[center_index]);
        map_center_y = (right_center_y[center_index]);
        census_index = map_center_y*image_width+map_center_x;

        now_dis = right_center_disparity[dis_n];
        new_map_x = map_center_x-now_dis;

        if(map_center_x>=image_width||map_center_y>=image_height||new_map_x<0)
            costs[t_num*image_mount+image_num] = max_cost;
        else{
            /*right_center_center_cost = right_center_center_census[census_index];
            costs[t_num * image_mount + image_num] = popcount(
                    right_center_center_cost ^ right_center_census[census_image_num][census_index]);*/
            costs[t_num * image_mount + image_num]= CostCompute(map_center_x,map_center_y,image_width,image_height,right_center_center_census,right_center_census,census_image_num);

        }

        //*********************
    }
    else
    {
        // right right pair
        map_center_x = int(right_right_x[center_index]);
        map_center_y = int(right_right_y[center_index]);
        census_index = map_center_y*image_width+map_center_x;

        now_dis = right_right_disparity[dis_n];
        new_map_x = map_center_x-now_dis;

        if(map_center_x>=image_width||map_center_y>=image_height||new_map_x<0.0f)
            costs[t_num*SUM_IMAGE_NUM+image_num] = max_cost;
        else{
            /*right_right_center_cost = right_right_center_census[census_index];
            costs[t_num * SUM_IMAGE_NUM + image_num] = popcount(
                    right_right_center_cost ^ (right_right_census[census_image_num][census_index]));*/
            costs[t_num * image_mount + image_num]= CostCompute(map_center_x,map_center_y,image_width,image_height,right_right_center_census,right_right_census,census_image_num);

        }
    }
    __syncthreads();
    if(image_p==3)
    {

        int e_index=(t_num+1)*image_mount,b_index=t_num*image_mount;
        uint32_t s_result=0;
        //s_result=costs[b_index+2]+costs[b_index];
        //s_result=costs[b_index+2];
        s_result=costs[b_index+2]+costs[b_index+1]+costs[b_index];
        /*for(;b_index<e_index;b_index++)
        {
            //if((b_index%4)!=0&&(b_index%4!=1)&&(b_index%4!=2))
            //printf("dddd%d\n",costs[b_index]);
            s_result=costs[b_index];

        }*/
        //printf("%d\n",s_result);
        final_cost_volume[(image_h*IMG_WIDTH+image_w)*NEW_MAX_DISPARITY+dis_n]=s_result;
    }
    /*int b_index,e_index,h_index;
    if(image_num%4==0)
    {
        h_index=b_index=t_num*SUM_IMAGE_NUM+image_num,e_index=t_num*SUM_IMAGE_NUM+(image_num/4+1)*4;
        uint32_t s_result=0;
        for(;b_index<e_index;b_index++)
        {
            s_result+=costs[b_index];
        }
        costs[h_index]=s_result;

    }
    __syncthreads();
    if(image_num==15)
    {
        e_index=(t_num+1)*SUM_IMAGE_NUM,b_index=t_num*SUM_IMAGE_NUM;
        uint32_t s_result=0;
        for(;b_index<e_index;b_index+=4)
        {
            s_result+=costs[b_index];
        }
                //printf("done\n");
        d_cost[(image_h*IMG_WIDTH+image_w)*NEW_MAX_DISPARITY+dis_n]=s_result;
    }*/
}