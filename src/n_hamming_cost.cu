#include"n_hamming_cost.h"
/*__global__ void
N_HammingDistanceCostKernel (  cost_t *l_transform1,cost_t *l_transform2,cost_t *l_transform3,cost_t *l_transform4,
                               cost_t *r_transform1 ,cost_t *r_transform2 ,cost_t *r_transform3 ,cost_t *r_transform4 ,
                               const cost_t* center_transform,
                               cost_t *t_transform1,cost_t *t_transform2,cost_t *t_transform3,cost_t *t_transform4,
                               cost_t *b_transform1,cost_t *b_transform2,cost_t *b_transform3,cost_t *b_transform4,
                               uint8_t *d_cost, const int rows, const int cols,const int image_number,const int sum_disparity)*/
__global__ void
N_HammingDistanceCostKernel1 (  cost_t *l_transform1,
                                cost_t *r_transform1 ,
                               const cost_t* center_transform,
                               cost_t *t_transform1,
                               cost_t *b_transform1,
                               uint32_t *d_cost, const int rows, const int cols,const int image_number,const int sum_disparity)
{
    const int x=   blockIdx.x;  // the center image pixel x
    const int y=   blockIdx.y;  // the center image pixel y
    const int THRid = threadIdx.x; // represend the cost label
    //l_transfgorm b_transform is right top
    //cost_t center_tran=center_transform[y*cols+x];
    //float disparity=0.05*THRid*image_number;

    float disparity=0.1*(THRid+1)*image_number;
    int l_disparity=int(disparity);
    //float l_r=float(disparity-l_disparity);
    int r_disparity=(disparity-l_disparity>0)?(l_disparity+1):(l_disparity);
    float dif=disparity-l_disparity;
   // int result=0;
    //int n_result=0;
    int l_dif=int((1.0f-dif)*10.0f+0.1f);
    int r_dif=int(dif*10.0f+0.1f);
    const cost_t center_cost=center_transform[x+y*cols];
    const cost_t r=~center_cost;
    //printf("%d::%d\n",l_dif,r_dif);
   // printf("%d\n",popcount(center_cost^r));
    //cost_t r=0;
    /*cost_t l_r_cost=(x+r_disparity)<cols?l_transform1[x+r_disparity+y*cols]:l_transform1[cols-1+y*cols];
    cost_t l_l_cost=(x+l_disparity)<cols?l_transform1[x+l_disparity+y*cols]:l_transform1[cols-1+y*cols];

    cost_t r_r_cost=(x-r_disparity)>=0?r_transform1[x-r_disparity+y*cols]:r_transform1[0+y*cols];
    cost_t r_l_cost=(x-l_disparity)>=0?r_transform1[x-l_disparity+y*cols]:r_transform1[0+y*cols];

    cost_t b_r_cost=(y+r_disparity)<rows?b_transform1[x+(y+r_disparity)*cols]:b_transform1[x+(rows-1)*cols];
    cost_t b_l_cost=(y+l_disparity)<rows?b_transform1[x+(y+l_disparity)*cols]:b_transform1[x+(rows-1)*cols];

    cost_t t_r_cost=(y-r_disparity)>=0?t_transform1[x+(y-r_disparity)*cols]:t_transform1[x+0*cols];
    cost_t t_l_cost=(y-l_disparity)>=0?t_transform1[x+(y-l_disparity)*cols]:t_transform1[x+0*cols];*/

    const cost_t l_r_cost=(x+r_disparity)<cols?l_transform1[x+r_disparity+y*cols]:r;
    const cost_t l_l_cost=(x+l_disparity)<cols?l_transform1[x+l_disparity+y*cols]:r;

    const cost_t r_r_cost=(x-r_disparity)>=0?r_transform1[x-r_disparity+y*cols]:r;
    const cost_t r_l_cost=(x-l_disparity)>=0?r_transform1[x-l_disparity+y*cols]:r;

    const cost_t b_r_cost=(y+r_disparity)<rows?b_transform1[x+(y+r_disparity)*cols]:r;
    const cost_t b_l_cost=(y+l_disparity)<rows?b_transform1[x+(y+l_disparity)*cols]:r;

    const cost_t t_r_cost=(y-r_disparity)>=0?t_transform1[x+(y-r_disparity)*cols]:r;
    const cost_t t_l_cost=(y-l_disparity)>=0?t_transform1[x+(y-l_disparity)*cols]:r;


    const int result=(l_dif*((popcount(center_cost^l_l_cost)+popcount(center_cost^r_l_cost)+popcount(center_cost^b_l_cost)+popcount(center_cost^t_l_cost)))
     +r_dif*((popcount(center_cost^l_r_cost)+popcount(center_cost^r_r_cost)+popcount(center_cost^b_r_cost)+popcount(center_cost^t_r_cost))));
    /*const int result=(l_dif*(popcount(center_cost^b_l_cost)+popcount(center_cost^t_l_cost)))
             +r_dif*(+popcount(center_cost^b_r_cost)+popcount(center_cost^t_r_cost));*/
    /*const int result=(l_dif*(popcount(center_cost^l_l_cost)+popcount(center_cost^r_l_cost)))
             +r_dif*(+popcount(center_cost^r_r_cost)+popcount(center_cost^r_r_cost));*/

    d_cost[(x+y*cols)*NEW_MAX_DISPARITY+NEW_MAX_DISPARITY/2+THRid]+=(result);
    //if(d_cost[(x+y*cols)*NEW_MAX_DISPARITY+NEW_MAX_DISPARITY/2-1-THRid]==0)
     // printf("%u::result::%d::old_id::%d\n",d_cost[(x+y*cols)*NEW_MAX_DISPARITY+NEW_MAX_DISPARITY/2-1-THRid],result,NEW_MAX_DISPARITY/2+THRid);
    //***********positive d_cost finish******************************
    //***********negative d_cost begin*******************************
    /*l_r_cost=(x-r_disparity)>=0?l_transform1[x-r_disparity+y*cols]:l_transform1[0+y*cols];
    l_l_cost=(x-l_disparity)>=0?l_transform1[x-l_disparity+y*cols]:l_transform1[0+y*cols];

    r_r_cost=(x+r_disparity)<cols?r_transform1[x+r_disparity+y*cols]:r_transform1[cols-1+y*cols];
    r_l_cost=(x+l_disparity)<cols?r_transform1[x+l_disparity+y*cols]:r_transform1[cols-1+y*cols];

    b_r_cost=(y-r_disparity)>=0?b_transform1[x+(y-r_disparity)*cols]:b_transform1[x+0*cols];
    b_l_cost=(y-l_disparity)>=0?b_transform1[x+(y-l_disparity)*cols]:b_transform1[x+0*cols];

    t_r_cost=(y+r_disparity)<rows?t_transform1[x+(y+r_disparity)*cols]:t_transform1[x+(rows-1)*cols];
    t_l_cost=(y+l_disparity)<rows?t_transform1[x+(y+l_disparity)*cols]:t_transform1[x+(rows-1)*cols];*/


    const cost_t l_r_cost_2=(x-r_disparity)>=0?l_transform1[x-r_disparity+y*cols]:r;
    const cost_t l_l_cost_2=(x-l_disparity)>=0?l_transform1[x-l_disparity+y*cols]:r;

    const cost_t r_r_cost_2=(x+r_disparity)<cols?r_transform1[x+r_disparity+y*cols]:r;
    const cost_t r_l_cost_2=(x+l_disparity)<cols?r_transform1[x+l_disparity+y*cols]:r;

    const cost_t b_r_cost_2=(y-r_disparity)>=0?b_transform1[x+(y-r_disparity)*cols]:r;
    const cost_t b_l_cost_2=(y-l_disparity)>=0?b_transform1[x+(y-l_disparity)*cols]:r;

    const cost_t t_r_cost_2=(y+r_disparity)<rows?t_transform1[x+(y+r_disparity)*cols]:r;
    const cost_t t_l_cost_2=(y+l_disparity)<rows?t_transform1[x+(y+l_disparity)*cols]:r;

    //int f1=(1.0-dif)*100*((popcount(center_cost^l_l_cost)+popcount(center_cost^r_l_cost)+popcount(center_cost^b_l_cost)+popcount(center_cost^t_l_cost))/4);
    //int f2=dif*100*((popcount(center_cost^l_r_cost)+popcount(center_cost^r_r_cost)+popcount(center_cost^b_r_cost)+popcount(center_cost^t_r_cost))/4);
    const int n_result=(l_dif*((popcount(center_cost^l_l_cost_2)+popcount(center_cost^r_l_cost_2)+popcount(center_cost^b_l_cost_2)+popcount(center_cost^t_l_cost_2)))
             +r_dif*((popcount(center_cost^l_r_cost_2)+popcount(center_cost^r_r_cost_2)+popcount(center_cost^b_r_cost_2)+popcount(center_cost^t_r_cost_2))));
    /*const int n_result=(l_dif*(popcount(center_cost^b_l_cost_2)+popcount(center_cost^t_l_cost_2)))
             +r_dif*(+popcount(center_cost^b_r_cost_2)+popcount(center_cost^t_r_cost_2));*/
   /* const int n_result=(l_dif*(popcount(center_cost^l_l_cost)+popcount(center_cost^r_l_cost)))
             +r_dif*(+popcount(center_cost^r_r_cost)+popcount(center_cost^r_r_cost));*/

    d_cost[(x+y*cols)*NEW_MAX_DISPARITY+NEW_MAX_DISPARITY/2-1-THRid]+=(n_result);
    //if(n_result==0)
      //printf("%u::id::%d\n",d_cost[(x+y*cols)*NEW_MAX_DISPARITY+NEW_MAX_DISPARITY/2+THRid],NEW_MAX_DISPARITY/2+THRid);*/

    //if(d_cost[(x+y*cols)*NEW_MAX_DISPARITY+NEW_MAX_DISPARITY/2+THRid]!=n_result)
    //printf("cost:::%u::%d\n",d_cost[(x+y*cols)*NEW_MAX_DISPARITY+NEW_MAX_DISPARITY/2+THRid],n_result);
}

//****************Warp Disparity***********
__global__ void
W_N_HammingDistanceKernel(cost_t* center_transform,
                          cost_t* l_c_transform,cost_t *r_c_transform,cost_t *b_c_transform,cost_t *t_c_transform,
                          cost_t *top_left_c_transform,cost_t *top_right_c_transform, cost_t *bottom_left_c_transform,cost_t *bottom_right_c_transform,
                          uint8_t *c_pic,uint8_t *l_pic,uint8_t *r_pic,uint8_t *b_pic,uint8_t *t_pic,
                          uint8_t *top_left_pic,uint8_t *top_righth_pic,uint8_t *bottom_left_pic, uint8_t *bottom_right_pic,
                          uint32_t *d_cost,
                          float* right_f_vec,float* right_baseline_vec,
                          float* left_f_vec,float* left_baseline_vec
                          )
{
    // __shared__ uint32_t cost_z[NEW_MAX_DISPARITY*16];
    /*const int b_x=blockIdx.x;
    const int by=blockIdx.y;
    const int dis_n=threadIdx.x;

    const int bx=b_x%IMG_WIDTH;
    const int img_n=(b_x/IMG_WIDTH)/4;
    const int img_p=(b_x/IMG_WIDTH)%4;
   // const int img_n=(b_x/IMG_WIDTH);
    uint32_t n_result=0;
    int dis_count,pixel_count,t_pixel_count;
    pixel_count=by*IMG_WIDTH+bx;
    dis_count=img_n*NEW_MAX_DISPARITY+dis_n;
    t_pixel_count=dis_count*(IMG_HEIGHT*IMG_WIDTH)+pixel_count;
    cost_t center_cost=center_transform[pixel_count];//get the center cost
    int cost_index=pixel_count*NEW_MAX_DISPARITY+dis_n;
    if(img_p==0)
      n_result=popcount(center_cost^l_c_transform[t_pixel_count]);
    else if(img_p==1)
      n_result=popcount(center_cost^r_c_transform[t_pixel_count]);
    else if(img_p==2)
      n_result=popcount(center_cost^t_c_transform[t_pixel_count]);
    else
      n_result=popcount(center_cost^b_c_transform[t_pixel_count]);
     d_cost[cost_index]+=(n_result);*/
    __shared__ uint32_t costs[SUM_IMAGE_NUM*32];
    __shared__ uint32_t new_costs[SUM_IMAGE_NUM*32];
    int b_x=blockIdx.x;
    int b_y=blockIdx.y;
    int t_x=threadIdx.x;
    int t_y=threadIdx.y;
    int image_num=t_x;
    int dis_n=(b_x/IMG_WIDTH)*32+t_y;
    int t_num=t_y;
    int image_w=b_x%IMG_WIDTH;
    int image_h=b_y;
    uint32_t n_result=0;
    int image_p=image_num/SELECT_IMAGE_NUM;
    int image_count=image_num%SELECT_IMAGE_NUM;
    //image_count = IMAGE_NUMBER-1-image_count;
    int t_pixel_count=(dis_n+image_count*NEW_MAX_DISPARITY)*IMG_WIDTH*IMG_HEIGHT+image_h*IMG_WIDTH+image_w;
    cost_t center_cost=center_transform[image_h*IMG_WIDTH+image_w];
    //if(image_num%4!=0&&image_num%4!=1&&image_num%4!=2)

    float now_dis;
    float now_depth = dis_n*(END_DEPTH-BEGIN_DEPTH)/float(NEW_MAX_DISPARITY)+BEGIN_DEPTH;
    //printf("::%f\n",left_baseline_vec[1]);
    //printf("%f\n",now_depth);
    int dis_num;
    if(image_p==0)
    {
        //costs[t_num*SUM_IMAGE_NUM+image_num]=popcount(center_cost^l_c_transform[t_pixel_count]);
        //new_costs[t_num*SUM_IMAGE_NUM+image_num]=popcount(center_cost^top_left_c_transform[t_pixel_count]);
        //printf("::%f\n",left_baseline_vec[0]);
        now_dis = left_baseline_vec[image_count]*left_f_vec[0]/now_depth;
        //printf("::%f\n",now_dis);
        if(now_dis>END_DIS)
            dis_num = NEW_MAX_DISPARITY-1;
        else
            dis_num = (now_dis-BEGIN_DIS)/((END_DIS-BEGIN_DIS)/float(NEW_MAX_DISPARITY));
        t_pixel_count= (dis_num+image_count*NEW_MAX_DISPARITY)*IMG_WIDTH*IMG_HEIGHT+image_h*IMG_WIDTH+image_w;
        costs[t_num*SUM_IMAGE_NUM+image_num]=popcount(center_cost^l_c_transform[t_pixel_count]);

        new_costs[t_num*SUM_IMAGE_NUM+image_num]=0;
    }
    else if(image_p==1)
    {
        now_dis = right_baseline_vec[image_count]*left_f_vec[0]/now_depth;

        if(now_dis>END_DIS)
            dis_num = NEW_MAX_DISPARITY-1;
        else
            dis_num = (now_dis-BEGIN_DIS)/((END_DIS-BEGIN_DIS)/float(NEW_MAX_DISPARITY));
        //if(now_dis<BEGIN_DIS)
        //printf("::%d\n",dis_num);
        t_pixel_count=(dis_num+image_count*NEW_MAX_DISPARITY)*IMG_WIDTH*IMG_HEIGHT+image_h*IMG_WIDTH+image_w;
        costs[t_num*SUM_IMAGE_NUM+image_num]=popcount(center_cost^r_c_transform[t_pixel_count]);
        /*if(image_w>=dis_n*WARP_DIS){
            costs[t_num*SUM_IMAGE_NUM+image_num]=popcount(center_cost^r_c_transform[t_pixel_count]);
        }
        else {
             costs[t_num*SUM_IMAGE_NUM+image_num]=300;
        }*/
        //new_costs[t_num*SUM_IMAGE_NUM+image_num]=popcount(center_cost^top_right_c_transform[t_pixel_count]);
        //costs[t_num*SUM_IMAGE_NUM+image_num] = 0;
        new_costs[t_num*SUM_IMAGE_NUM+image_num] = 0;
    }
    else if(image_p==2)
    {
        //printf("::%f\n",right_baseline_vec[0]);
        //costs[t_num*SUM_IMAGE_NUM+image_num]=popcount(center_cost^t_c_transform[t_pixel_count]);
        //new_costs[t_num*SUM_IMAGE_NUM+image_num]=popcount(center_cost^bottom_left_c_transform[t_pixel_count]);
        costs[t_num*SUM_IMAGE_NUM+image_num]=0;
        new_costs[t_num*SUM_IMAGE_NUM+image_num]=0;
    }
    else
    {
        //printf("::%f\n",right_baseline_vec[0]);
        //costs[t_num*SUM_IMAGE_NUM+image_num]=popcount(center_cost^b_c_transform[t_pixel_count]);
        //new_costs[t_num*SUM_IMAGE_NUM+image_num]=popcount(center_cost^bottom_right_c_transform[t_pixel_count]);
        costs[t_num*SUM_IMAGE_NUM+image_num]=0;
        new_costs[t_num*SUM_IMAGE_NUM+image_num]=0;
    }
    __syncthreads();
    if(image_num==4*SELECT_IMAGE_NUM-1)
    {
        //
        int e_index=(t_num+1)*SUM_IMAGE_NUM,b_index=t_num*SUM_IMAGE_NUM;
        uint32_t s_result=0;
        for(;b_index<e_index;b_index++)
        {
            //if((b_index%4)!=0&&(b_index%4!=1)&&(b_index%4!=2))
            s_result+=costs[b_index];
               // s_result+=new_costs[b_index];
        }
        //s_result = costs[e_index-1];
        //printf("done\n");
        d_cost[(image_h*IMG_WIDTH+image_w)*NEW_MAX_DISPARITY+dis_n]=s_result;
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

__global__ void
N_N_HammingDistanceKernel(cost_t* center_transform,
                          cost_t* l_c_transform,
                         // uint8_t *c_pic,
                          uint32_t *d_cost)
{
    const int b_x=blockIdx.x;
    const int by=blockIdx.y;
    const int dis_n=threadIdx.x;

    const int bx=b_x%IMG_WIDTH;
    const int img_n=(b_x/IMG_WIDTH);
  //  const int img_p=(b_x/IMG_WIDTH)%4;

    uint32_t n_result=0;
    int dis_count,pixel_count,t_pixel_count;
    pixel_count=by*IMG_WIDTH+bx;
    dis_count=img_n*NEW_MAX_DISPARITY+dis_n;
    t_pixel_count=dis_count*(IMG_HEIGHT*IMG_WIDTH)+pixel_count;
    cost_t center_cost=center_transform[pixel_count];//get the center cos
      n_result=popcount(center_cost^l_c_transform[t_pixel_count]);

    //d_cost[(bx+by*IMG_WIDTH)*NEW_MAX_DISPARITY+dis_n]+=10*(n_result);
    d_cost[(bx+by*IMG_WIDTH)*NEW_MAX_DISPARITY+dis_n]+=(n_result);
}

__global__ void SumCost(uint32_t *l_cost,uint32_t *r_cost,uint32_t *t_cost,uint32_t *b_cost,uint32_t *sum_cost)
{
    int bx=blockIdx.x+1;
    int by=blockIdx.y+1;
    int tx=threadIdx.x;
    int t_num=blockDim.x;
    int index=(bx*by-1)*t_num+tx;
    sum_cost[index]=l_cost[index]+r_cost[index]+b_cost[index]+t_cost[index];
}
//****************negative disparity*******
__global__ void
N_HammingDistanceCostKernel1_Z(cost_t *l_transform1,
                               cost_t *r_transform1 ,
                               const cost_t* center_transform,
                               cost_t *t_transform1,
                               cost_t *b_transform1,
                               uint8_t *d_cost, const int rows, const int cols,const int image_number,const int sum_disparity)
{

    const int x=   blockIdx.x;  // the center image pixel x
    const int y=   blockIdx.y;  // the center image pixel y
    const int THRid = threadIdx.x; // represend the cost label

    //l_transfgorm b_transform is right top
    //cost_t center_tran=center_transform[y*cols+x];
    //float disparity=0.05*THRid*image_number;
    float disparity=0.05*THRid*image_number;
    int l_disparity=int(disparity);
    int r_disparity=(disparity-l_disparity>0)?(l_disparity+1):(l_disparity);
    float dif=disparity-l_disparity;
    int result=0;
    int n_result=0;
    int l_dif;
    int r_dif;
    cost_t center_cost=center_transform[x+y*cols];
    /*cost_t l_r_cost=(x+r_disparity)<cols?l_transform1[x+r_disparity+y*cols]:0;
    cost_t l_l_cost=(x+l_disparity)<cols?l_transform1[x+l_disparity+y*cols]:0;

    cost_t r_r_cost=(x-r_disparity)>=0?r_transform1[x-r_disparity+y*cols]:0;
    cost_t r_l_cost=(x-l_disparity)>=0?r_transform1[x-l_disparity+y*cols]:0;

    cost_t b_r_cost=(y+r_disparity)<rows?b_transform1[x+(y+r_disparity)*cols]:0;
    cost_t b_l_cost=(y+l_disparity)<rows?b_transform1[x+(y+l_disparity)*cols]:0;

    cost_t t_r_cost=(y-r_disparity)>=0?t_transform1[x+(y-r_disparity)*cols]:0;
    cost_t t_l_cost=(y-l_disparity)>=0?t_transform1[x+(y-l_disparity)*cols]:0;*/

    cost_t l_r_cost=(x+r_disparity)<cols?l_transform1[x+r_disparity+y*cols]:l_transform1[cols-1+y*cols];
    cost_t l_l_cost=(x+l_disparity)<cols?l_transform1[x+l_disparity+y*cols]:l_transform1[cols-1+y*cols];

    cost_t r_r_cost=(x-r_disparity)>=0?r_transform1[x-r_disparity+y*cols]:r_transform1[0+y*cols];
    cost_t r_l_cost=(x-l_disparity)>=0?r_transform1[x-l_disparity+y*cols]:r_transform1[0+y*cols];

    cost_t b_r_cost=(y+r_disparity)<rows?b_transform1[x+(y+r_disparity)*cols]:b_transform1[x+(rows-1)*cols];
    cost_t b_l_cost=(y+l_disparity)<rows?b_transform1[x+(y+l_disparity)*cols]:b_transform1[x+(rows-1)*cols];

    cost_t t_r_cost=(y-r_disparity)>=0?t_transform1[x+(y-r_disparity)*cols]:t_transform1[x+0*cols];
    cost_t t_l_cost=(y-l_disparity)>=0?t_transform1[x+(y-l_disparity)*cols]:t_transform1[x+0*cols];

    l_dif=int((1.0f-dif)*20.0f+0.1f);
    r_dif=int(dif*20.0f+0.1f);
    result+=(l_dif*((popcount(center_cost^l_l_cost)+popcount(center_cost^r_l_cost)+popcount(center_cost^b_l_cost)+popcount(center_cost^t_l_cost)))
             +r_dif*((popcount(center_cost^l_r_cost)+popcount(center_cost^r_r_cost)+popcount(center_cost^b_r_cost)+popcount(center_cost^t_r_cost))));

    //result/=10;
    //printf("%d\n",result);
   // d_cost[(x+y*cols)*NEW_MAX_DISPARITY+NEW_MAX_DISPARITY/2-1-THRid]=(uint8_t)result;
    result/=8;
    //if(result>255)
       // printf("%d::%d\n",result,result);
    d_cost[(x+y*cols)*NEW_MAX_DISPARITY+NEW_MAX_DISPARITY/2+THRid]=(uint8_t)(result);
    //***********positive d_cost finish******************************
    //***********negative d_cost begin*******************************

    /*l_r_cost=(x-r_disparity)>=0?l_transform1[x-r_disparity+y*cols]:0;
    l_l_cost=(x-l_disparity)>=0?l_transform1[x-l_disparity+y*cols]:0;

    r_r_cost=(x+r_disparity)<cols?r_transform1[x+r_disparity+y*cols]:0;
    r_l_cost=(x+l_disparity)<cols?r_transform1[x+l_disparity+y*cols]:0;

    b_r_cost=(y-r_disparity)>=0?b_transform1[x+(y-r_disparity)*cols]:0;
    b_l_cost=(y-l_disparity)>=0?b_transform1[x+(y-l_disparity)*cols]:0;

    t_r_cost=(y+r_disparity)<rows?t_transform1[x+(y+r_disparity)*cols]:0;
    t_l_cost=(y+l_disparity)<rows?t_transform1[x+(y+l_disparity)*cols]:0;*/
    l_r_cost=(x-r_disparity)>=0?l_transform1[x-r_disparity+y*cols]:l_transform1[0+y*cols];
    l_l_cost=(x-l_disparity)>=0?l_transform1[x-l_disparity+y*cols]:l_transform1[0+y*cols];

    r_r_cost=(x+r_disparity)<cols?r_transform1[x+r_disparity+y*cols]:r_transform1[cols-1+y*cols];
    r_l_cost=(x+l_disparity)<cols?r_transform1[x+l_disparity+y*cols]:r_transform1[cols-1+y*cols];

    b_r_cost=(y-r_disparity)>=0?b_transform1[x+(y-r_disparity)*cols]:b_transform1[x+0*cols];
    b_l_cost=(y-l_disparity)>=0?b_transform1[x+(y-l_disparity)*cols]:b_transform1[x+0*cols];

    t_r_cost=(y+r_disparity)<rows?t_transform1[x+(y+r_disparity)*cols]:t_transform1[x+(rows-1)*cols];
    t_l_cost=(y+l_disparity)<rows?t_transform1[x+(y+l_disparity)*cols]:t_transform1[x+(rows-1)*cols];

    float f1=(1.0-dif)*100*((popcount(center_cost^l_l_cost)+popcount(center_cost^r_l_cost)+popcount(center_cost^b_l_cost)+popcount(center_cost^t_l_cost)));
    float f2=dif*100*((popcount(center_cost^l_r_cost)+popcount(center_cost^r_r_cost)+popcount(center_cost^b_r_cost)+popcount(center_cost^t_r_cost)));


    n_result+=(l_dif*((popcount(center_cost^l_l_cost)+popcount(center_cost^r_l_cost)+popcount(center_cost^b_l_cost)+popcount(center_cost^t_l_cost)))
             +r_dif*((popcount(center_cost^l_r_cost)+popcount(center_cost^r_r_cost)+popcount(center_cost^b_r_cost)+popcount(center_cost^t_r_cost))));

    //n_result+=(f1+f2);
    //n_result/=10;
    //d_cost[(x+y*cols)*NEW_MAX_DISPARITY+NEW_MAX_DISPARITY/2+THRid]=(uint8_t)(n_result);
    //printf("%d::%d::%d\n",int((1.0f-dif)*20.0f+0.1f),int(dif*20.0f+0.1f),dif);
    n_result/=8;
   // if(n_result>255)
     //printf("%d\n",n_result);
    d_cost[(x+y*cols)*NEW_MAX_DISPARITY+NEW_MAX_DISPARITY/2-1-THRid]=(uint8_t)(n_result);

}
