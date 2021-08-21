#include "Brief.h"

using namespace std;
using namespace cv;

//计算方向用SIFT REAL的梯度方向
Mat make_bref(vector<KeyPoint> &kps, Mat gray, u8 angle_mode)
{
    //获取点对
    static const std::vector<cv::Point> pattern = GetPattern();
    static const u16 M = pattern.size();
    u8 val;
	float kp_sin, kp_cos;
    
	#define GET_X(i, index) pattern[index].x
	#define GET_Y(i, index) pattern[index].y
    #define GET_VALUE(i, index, sin_, cos_) gray.at<u8>(kps[i].pt.y + cvRound(GET_X(i, index) * sin_ + GET_Y(i, index) * cos_), kps[i].pt.x + cvRound(GET_X(i, index) * cos_ + GET_Y(i, index) * sin_))

    switch(angle_mode){
        case DEFAULT_GRAY:
            get_gray_angle(gray, kps);
            break;
        case IMPROVE_GRAY:
            get_gray_improve_angle(gray, kps);
            break;
        case COM_GRAY:
            get_coM_angle(gray, kps);
            break;
        case DEFAULT_SIFT:
            get_sift_default_angle(gray, kps);
            break;
        case DEFAULT_FREAK:
            get_freak_angle(gray, kps);
            break;
        case COMBINE_SIFT_GRAY:
            get_sift_gray_angle(gray, kps);
            break;
        case DEFAULT_SIFT_1:
            get_sift_default_1_angle(gray, kps);
            break;
        case DEFAULT_SIFT_2:
            get_sift_default_2_angle(gray, kps);
            break;
        case DEFAULT_SIFT_3:
            get_sift_default_3_angle(gray, kps);
            break;
        case QUICK_SIFT:
            get_sift_quick_angle(gray, kps);
            break;
        default:
            user_main_error("unknown angle mode");
            exit(-1);
            break;
    }

    Mat desc(kps.size(), 32, CV_8UC1);

    for(int kp = 0; kp < kps.size(); kp++)
	{
        kp_cos = (float)cos(kps[kp].angle/180.0f * CV_PI);
        kp_sin = (float)sin(kps[kp].angle/180.0f * CV_PI);
        for (u16 num = 0, count = 0; num < 32; num++, count+=15)
        {
            val  = GET_VALUE(kp, num*16, kp_sin, kp_cos)		  < GET_VALUE(kp, num*16+1, kp_sin, kp_cos);
            val |= (GET_VALUE(kp, num*16+ 2, kp_sin, kp_cos)   < GET_VALUE(kp, num*16 + 3, kp_sin, kp_cos))   << 1;
            val |= (GET_VALUE(kp, num*16+ 4, kp_sin, kp_cos)   < GET_VALUE(kp, num*16 + 5, kp_sin, kp_cos))   << 2;
            int test1 = cvRound(GET_X(kp, num*16+ 6) * kp_sin + GET_Y(kp, num*16+ 6) * kp_cos);
            int test2 = cvRound(GET_X(kp, num*16+ 6) * kp_cos + GET_Y(kp, num*16+ 6) * kp_sin);
            test1 = cvRound(GET_X(kp, num*16+ 7) * kp_sin + GET_Y(kp, num*16+ 7) * kp_cos);
            test2 = cvRound(GET_X(kp, num*16+ 7) * kp_cos + GET_Y(kp, num*16+ 7) * kp_sin);
            val |= (GET_VALUE(kp, num*16+ 6, kp_sin, kp_cos)   < GET_VALUE(kp, num*16 + 7, kp_sin, kp_cos))   << 3;
            val |= (GET_VALUE(kp, num*16+ 8, kp_sin, kp_cos)   < GET_VALUE(kp, num*16 + 9, kp_sin, kp_cos))    << 4;
            val |= (GET_VALUE(kp, num*16+ 10, kp_sin, kp_cos) < GET_VALUE(kp, num*16 + 11, kp_sin, kp_cos)) << 5;
            val |= (GET_VALUE(kp, num*16+ 12, kp_sin, kp_cos) < GET_VALUE(kp, num*16 + 13, kp_sin, kp_cos)) << 6;
            val |= (GET_VALUE(kp, num*16+ 14, kp_sin, kp_cos) < GET_VALUE(kp, num*16 + 15, kp_sin, kp_cos)) << 7;
            desc.at<u8>(kp, num) = val;
        }
    }
    #undef GET_VALUE
	#undef GET_X
	#undef GET_Y
    return desc;
}
