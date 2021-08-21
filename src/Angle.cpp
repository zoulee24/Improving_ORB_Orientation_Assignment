#include "Angle.h"


using namespace std;
using namespace cv;

void get_freak_angle(Mat gray, vector<KeyPoint> &kps)
{
    //获取点对
    static const std::vector<cv::Point> pattern = GetPattern();
    const u16 M = pattern.size();
    #define GET_VALUE_Y(i, point_) gray.at<u8>(kps[i].pt.y+point_.y + 1, kps[i].pt.x+point_.x)-gray.at<u8>(kps[i].pt.y+point_.y - 1, kps[i].pt.x+point_.x)
    #define GET_VALUE_X(i, point_) gray.at<u8>(kps[i].pt.y+point_.y, kps[i].pt.x+point_.x + 1)-gray.at<u8>(kps[i].pt.y+point_.y, kps[i].pt.x+point_.x - 1)
    for (size_t i = 0; i < kps.size(); i++)
    {
        float angle, test_a;
        int sum_x = 0, sum_y = 0;

        for( int num = 0; num < M; num++)
        {
            Point2i point1 = pattern[num];

            sum_x += GET_VALUE_X(i, point1);
            sum_y += GET_VALUE_Y(i, point1);
        }
        kps[i].class_id = i;
        kps[i].angle = fastAtan2(float(sum_y) / M, float(sum_x) / M);
    }

    #undef GET_VALUE_Y
    #undef GET_VALUE_X
}

void get_sift_gray_angle(Mat gray, vector<KeyPoint> &kps)
{
    vector<int> angles;
    vector<KeyPoint> add_kps;
    angles.resize(361, 0);
    int class_id = 0;
    //获取点对
    static const std::vector<cv::Point> pattern = GetPattern();
    const u16 M = pattern.size();
    #define GET_VALUE_Y(i, point_) gray.at<u8>(i->pt.y+point_.y + 1, i->pt.x+point_.x)-gray.at<u8>(i->pt.y+point_.y - 1, i->pt.x+point_.x)
    #define GET_VALUE_X(i, point_) gray.at<u8>(i->pt.y+point_.y, i->pt.x+point_.x + 1)-gray.at<u8>(i->pt.y+point_.y, i->pt.x+point_.x - 1)
    for (auto kp = kps.begin(); kp != kps.end(); kp++, class_id++)
    {
        float angle, test_a;
        int x_cha, y_cha;

        for( int num = 1; num < M; num++)
        {
            Point2i point1 = pattern[num-1];
            Point2i point2 = pattern[num];

            x_cha = GET_VALUE_X(kp, point1) - GET_VALUE_X(kp, point2);
            y_cha = GET_VALUE_Y(kp, point1) - GET_VALUE_Y(kp, point2);
            angles[cvRound(fastAtan2(y_cha, x_cha))]++;
        }
        angle = 0.0f;
        int max_num = -1;
        int sec_max_num = -1, third_max_num = -1;
        float sec_angle, third_angle;
        for (int k = 0; k < 361; k++) {
            if(angles[k]){
                if (angles[k] > max_num)
                {
                    third_max_num = sec_max_num;
                    sec_max_num = max_num;
                    max_num = angles[k];
                    sec_angle = angle;
                    angle = k;
                }
                else if (angles[k] > sec_max_num)
                {
                    third_max_num = sec_max_num;
                    sec_max_num = angles[k];
                    third_angle = sec_angle;
                    sec_angle = k;
                }
            }
            angles[k] = 0;
        }
        kp->angle = angle;
        kp->class_id = class_id;
        if (sec_max_num >= 0.80f * max_num)
        {
            KeyPoint add_kp(*kp);
            add_kp.angle = sec_angle;
            add_kps.push_back(add_kp);
        }
    }

    for (int i = 0; i < add_kps.size(); i++)
    {
        kps.push_back(add_kps[i]);
    }
    #undef GET_VALUE_Y
    #undef GET_VALUE_X
    
}

const std::vector<cv::Point> GetPattern(void)
{
    std::vector<cv::Point> pattern;
    const cv::Point* pattern0 = (const cv::Point*)bit_pattern_31_;	
    std::copy(pattern0, pattern0 + 512, std::back_inserter(pattern));
    return pattern;
}

void get_gray_angle(Mat gray, vector<KeyPoint> &kps)
{
    int up, down;
    for (size_t kpnum = 0; kpnum < kps.size(); kpnum++)
    {
        //m10 是x*value
        int m_01 = 0, m_10 = 0;
        static vector<vector<int> > umax = count_max(15);
        for (int i = 45; i < 61; i++)
        {
            int v_sum = 0;
            int y = umax[i][1];
            for (int x = -umax[i][0]; x <= umax[i][0]; x++)
            {
                up = gray.at<u8>(kps[kpnum].pt.y + y, kps[kpnum].pt.x + x);
                if ( i != 60){
                    down = gray.at<u8>(kps[kpnum].pt.y - y, kps[kpnum].pt.x + x);
                }
                else{
                    down = 0;
                }
                v_sum += (up - down);
                m_10 += x * (up + down);
            }
            m_01 += y * v_sum;
        }
        kps[kpnum].class_id = kpnum;
        kps[kpnum].angle =  fastAtan2((float)m_01 , (float)m_10);
    }
}

void get_coM_angle(Mat gray, vector<KeyPoint> &kps)
{
    static vector<vector<int> > umax = count_max(15);
    float qz;
    for (size_t kpnum = 0; kpnum < kps.size(); kpnum++)
    {
        //m10 是x*value
        float m_01 = 0.0f, m_10 = 0.0f;
        float S = 0.0f;
        int up, down;
        
        for (int i = 45; i < 61; i++)
        {
            float v_sum = 0.0f;
            int y = umax[i][1];
            float R2;
            for (int x = -umax[i][0]; x <= umax[i][0]; x++)
            {
                R2 = x*x + y*y;
                if (R2 < 121.0f){
                    qz = 1 - powf((sqrtf(R2) / 11.0f), 2);
                }
                else
                    qz = 0.0f;
                up = gray.at<u8>(kps[kpnum].pt.y + y, kps[kpnum].pt.x + x);
                if ( i != 44){
                    down = gray.at<u8>(kps[kpnum].pt.y - y, kps[kpnum].pt.x + x);
                }
                else{
                    down = 0;
                }
                v_sum +=  qz * (up - down);
                m_10 +=  qz * x * (up + down);
                S +=  qz * (up + down);
            }
            m_01 += y * v_sum;
        }
		kps[kpnum].class_id = kpnum;
		kps[kpnum].angle =  fastAtan2(m_01 / S, m_10 / S);
	}
}

/*
void get_coM_pro_angle(Mat gray, vector<KeyPoint> &kps)
{
    int up, down;
    vector<float> angles;
    angles.resize(361, 0);
    for (size_t kpnum = 0; kpnum < kps.size(); kpnum++)
    {
        static vector<vector<int> > umax = count_max(15);
        for (int i = 45; i < 61; i++)
        {
            int v_sum = 0;
            int y = umax[i][1];
            for (int x = -umax[i][0]; x <= umax[i][0]; x++)
            {
                up = gray.at<u8>(kps[kpnum].pt.y + y, kps[kpnum].pt.x + x);
                if ( i != 60){
                    down = gray.at<u8>(kps[kpnum].pt.y - y, kps[kpnum].pt.x + x);
                }
                else{
                    down = 0;
                }
                angles[cvRound(fastAtan2(y, x))]++;
                angles[cvRound(fastAtan2(-y, x))]++;
            }
        }
        kps[kpnum].class_id = kpnum;
        kps[kpnum].angle =  fastAtan2((float)m_01 , (float)m_10);
    }
}
*/

void get_gray_improve_angle(Mat gray, vector<KeyPoint> &kps)
{
    for (size_t kpnum = 0; kpnum < kps.size(); kpnum++)
    {
        //m10 是x*value
        int m_01 = 0, m_10 = 0;
        static vector<vector<int> > umax = count_max(15);
        for (int i = 45; i < 61; i++)
        {
            int v_sum = 0;
            int y = umax[i][1];
            for (int x = -umax[i][0]; x <= umax[i][0]; x++)
            {
                int up = gray.at<u8>(kps[kpnum].pt.y + y, kps[kpnum].pt.x + x);
                int down = gray.at<u8>(kps[kpnum].pt.y - y, kps[kpnum].pt.x + x);
                v_sum += (up*up - down*down);
                m_10 += x * (up*up + down*down);
            }
            m_01 += y * v_sum;
        }
        kps[kpnum].class_id = kpnum;
        kps[kpnum].angle =  fastAtan2((float)m_01 , (float)m_10);
    }
}

vector<vector<int> > count_max(int R)
{

    Mat circles(R+1, R+1, CV_8UC1, Scalar(0));
    circle(circles, Point2i(0, 0), R, Scalar(1), 1);
    vector<vector<int> > umax;
    umax.resize((circles.rows - 1) * 4 + 1);
    for (auto &x : umax)
    {
        x.resize(2);
    }
    for (int y = 0; y < circles.rows; y++)
    {
        for (int x = circles.cols; x-- ;)
        {
            if(circles.at<u8>(y, x))
            {
                umax[y][0] = x;
                break;
            }
        }
    }
    for (int y = R, i=0; y--; i++)
    {
        umax[R + 1 + i][0] =  - umax[y][0];
        umax[2 * R + i][0] = - umax[i][0];
        umax[3 * R + i + 1][0] = umax[y][0];
    }
    for (int y = 0, i = R, j = R; y <= R; y++, i--)
    {
        umax[i][1] = -i;
        umax[j + y][1] = -i;
        umax[2 * j + y][1] = y;
        umax[3 * j + i][1] = y;
    }
    return umax;
}

void get_sift_default_angle(Mat gray, vector<KeyPoint> &kps)
{
    //获取点对
    static const std::vector<cv::Point> pattern = GetPattern();
    const u16 M = pattern.size();
    float angle;
    vector<int> test_a;
    vector<KeyPoint> add_kps;
    test_a.resize(361, 0);
    int test_x, test_y;
    #define GET_VALUE_Y(kp, point_) gray.at<u8>(kp->pt.y+point_.y + 1, kp->pt.x+point_.x)-gray.at<u8>(kp->pt.y+point_.y - 1, kp->pt.x+point_.x)
    #define GET_VALUE_X(kp, point_) gray.at<u8>(kp->pt.y+point_.y, kp->pt.x+point_.x + 1)-gray.at<u8>(kp->pt.y+point_.y, kp->pt.x+point_.x - 1)
    int class_id = 0;
    for (auto kp = kps.begin(); kp != kps.end(); kp++, class_id++)
    {
        for( int num = 0; num < M; num ++)
        {
            Point2i point1 = pattern[num];
            test_x = GET_VALUE_X(kp, point1);
            test_y = GET_VALUE_Y(kp, point1);
            angle = cvRound(fastAtan2(test_y, test_x));
            test_a[angle]+=1;
        }
        angle = 0.0f;
        int max_num = -1;
        int sec_max_num = -1;
        float sec_angle;
        for (int i = 0; i < 361; i++) {
            if (test_a[i] > max_num)
            {
                sec_max_num = max_num;
                max_num = test_a[i];
                sec_angle = angle;
                angle = i;
            }
            else if (test_a[i] > sec_max_num)
            {
                sec_max_num = test_a[i];
                sec_angle = i;
            }
        }
        kp->angle = angle;
        kp->class_id = class_id;
        if (sec_max_num >= 0.85f * max_num)
        {
            KeyPoint add_kp(*kp);
            add_kp.angle = sec_angle;
            add_kps.push_back(add_kp);
        }
    }
    #undef GET_VALUE_Y
    #undef GET_VALUE_X
    for (int i = 0; i < add_kps.size(); i++)
    {
        kps.push_back(add_kps[i]);
    }
}

void get_sift_default_1_angle(Mat gray, vector<KeyPoint> &kps)
{
    //获取点对
    static const std::vector<cv::Point> pattern = GetPattern();
    const u16 M = pattern.size();
    float angle;
    vector<int> test_a;
    vector<KeyPoint> add_kps;
    test_a.resize(361, 0);
    int test_x, test_y;
    #define GET_VALUE_Y(kp, point_) gray.at<u8>(kp->pt.y+point_.y + 1, kp->pt.x+point_.x)-gray.at<u8>(kp->pt.y+point_.y - 1, kp->pt.x+point_.x)
    #define GET_VALUE_X(kp, point_) gray.at<u8>(kp->pt.y+point_.y, kp->pt.x+point_.x + 1)-gray.at<u8>(kp->pt.y+point_.y, kp->pt.x+point_.x - 1)
    int class_id = 0;
    for (auto kp = kps.begin(); kp != kps.end(); kp++, class_id++)
    {
        for( int num = 0; num < M; num +=2)
        {
            Point2i point1 = pattern[num];
            test_x = GET_VALUE_X(kp, point1);
            test_y = GET_VALUE_Y(kp, point1);
            angle = cvRound(fastAtan2(test_y, test_x));
            test_a[angle]+=1;
        }
        angle = 0.0f;
        int max_num = -1;
        int sec_max_num = -1;
        float sec_angle;
        for (int i = 0; i < 361; i++) {
            if (test_a[i] > max_num)
            {
                sec_max_num = max_num;
                max_num = test_a[i];
                sec_angle = angle;
                angle = i;
            }
            else if (test_a[i] > sec_max_num)
            {
                sec_max_num = test_a[i];
                sec_angle = i;
            }
        }
        kp->angle = angle;
        kp->class_id = class_id;
        if (sec_max_num >= 0.85f * max_num)
        {
            KeyPoint add_kp(*kp);
            add_kp.angle = sec_angle;
            add_kps.push_back(add_kp);
        }
    }
    #undef GET_VALUE_Y
    #undef GET_VALUE_X
    for (int i = 0; i < add_kps.size(); i++)
    {
        kps.push_back(add_kps[i]);
    }
}

void get_sift_default_2_angle(Mat gray, vector<KeyPoint> &kps)
{
    //获取点对
    static const std::vector<cv::Point> pattern = GetPattern();
    const u16 M = pattern.size();
    float angle;
    vector<int> test_a;
    vector<KeyPoint> add_kps;
    test_a.resize(361, 0);
    int test_x, test_y;
    #define GET_VALUE_Y(kp, point_) gray.at<u8>(kp->pt.y+point_.y + 1, kp->pt.x+point_.x)-gray.at<u8>(kp->pt.y+point_.y - 1, kp->pt.x+point_.x)
    #define GET_VALUE_X(kp, point_) gray.at<u8>(kp->pt.y+point_.y, kp->pt.x+point_.x + 1)-gray.at<u8>(kp->pt.y+point_.y, kp->pt.x+point_.x - 1)
    int class_id = 0;
    for (auto kp = kps.begin(); kp != kps.end(); kp++, class_id++)
    {
        for( int num = 0; num < M; num +=3)
        {
            Point2i point1 = pattern[num];
            test_x = GET_VALUE_X(kp, point1);
            test_y = GET_VALUE_Y(kp, point1);
            angle = cvRound(fastAtan2(test_y, test_x));
            test_a[angle]+=1;
        }
        angle = 0.0f;
        int max_num = -1;
        int sec_max_num = -1;
        float sec_angle;
        for (int i = 0; i < 361; i++) {
            if (test_a[i] > max_num)
            {
                sec_max_num = max_num;
                max_num = test_a[i];
                sec_angle = angle;
                angle = i;
            }
            else if (test_a[i] > sec_max_num)
            {
                sec_max_num = test_a[i];
                sec_angle = i;
            }
        }
        kp->angle = angle;
        kp->class_id = class_id;
        if (sec_max_num >= 0.85f * max_num)
        {
            KeyPoint add_kp(*kp);
            add_kp.angle = sec_angle;
            add_kps.push_back(add_kp);
        }
    }
    #undef GET_VALUE_Y
    #undef GET_VALUE_X
    for (int i = 0; i < add_kps.size(); i++)
    {
        kps.push_back(add_kps[i]);
    }
}

void get_sift_default_3_angle(Mat gray, vector<KeyPoint> &kps)
{
    //获取点对
    static const std::vector<cv::Point> pattern = GetPattern();
    const u16 M = pattern.size();
    float angle;
    vector<int> test_a;
    
    test_a.resize(361, 0);
    int test_x, test_y;
    #define GET_VALUE_Y(kp, point_) gray.at<u8>(kp->pt.y+point_.y + 1, kp->pt.x+point_.x)-gray.at<u8>(kp->pt.y+point_.y - 1, kp->pt.x+point_.x)
    #define GET_VALUE_X(kp, point_) gray.at<u8>(kp->pt.y+point_.y, kp->pt.x+point_.x + 1)-gray.at<u8>(kp->pt.y+point_.y, kp->pt.x+point_.x - 1)
    int class_id = 0;
    // float max_response = 0.0f, min_response = 999.0f, sub;
    for (auto kp = kps.begin(); kp != kps.end(); kp++, class_id++)
    {
        // if ( kp->response > max_response)
        // {
        //     max_response = kp->response;
        // }
        // if ( kp->response <= min_response)
        // {
        //     min_response = kp->response;
        // }
        for( int num = 0; num < M; num +=2)
        {
            Point2i point1 = pattern[num];
            test_x = GET_VALUE_X(kp, point1);
            test_y = GET_VALUE_Y(kp, point1);
            angle = cvRound(fastAtan2(test_y, test_x));
            test_a[angle]+=1;
        }
        kp->class_id = class_id;
    }

    float max_angle = 0.0f, sec_angle = 0.0f;
    int max_num = -1, sec_max_num = -1;
    for (int i = 0; i < 361; i++) {
        if (test_a[i] > max_num)
        {
            // sec_max_num = max_num;
            max_num = test_a[i];
            // sec_angle = max_angle;
            max_angle = i;
        }
        // else if ( test_a[i] > sec_max_num)
        // {
        //     sec_max_num = test_a[i];
        //     sec_angle = i;
        // }
    }
    vector<KeyPoint> add_kps;
    for (auto kp = kps.begin(); kp != kps.end(); kp++)
    {
        kp->angle = max_angle;
        // if (kp->response == min_response)
        // {
        //     KeyPoint add_kp(*kp);
        //     add_kp.angle = max_angle - sec_angle;
        //     add_kps.push_back(add_kp);
        // }
    }
    #undef GET_VALUE_Y
    #undef GET_VALUE_X
    // for(int i =0; i< add_kps.size(); i++)
    // {
    //     kps.push_back(add_kps[i]);
    // }
}

void get_sift_quick_angle(Mat gray, vector<KeyPoint> &kps)
{
    //获取点对
    static const std::vector<cv::Point> pattern = GetPattern();
    static const u16 M = pattern.size();
    float angle;
    vector<int> test_a;
    // vector<KeyPoint> add_kps;
    test_a.resize(361, 0);
    int test_x, test_y;
    #define GET_VALUE_Y(kp, point_) gray.at<u8>(kp->pt.y+point_.y + 1, kp->pt.x+point_.x)-gray.at<u8>(kp->pt.y+point_.y - 1, kp->pt.x+point_.x)
    #define GET_VALUE_X(kp, point_) gray.at<u8>(kp->pt.y+point_.y, kp->pt.x+point_.x + 1)-gray.at<u8>(kp->pt.y+point_.y, kp->pt.x+point_.x - 1)
    int class_id = 0, add;
    for (auto kp = kps.begin(); kp != kps.end(); kp++, class_id++)
    {
        for( int num = 0; num < M; num += 2 )
        {
            Point2i point1 = pattern[num];
            test_x = GET_VALUE_X(kp, point1);
            test_y = GET_VALUE_Y(kp, point1);
            
            angle = fastAtan2(test_y, test_x);
            test_a[cvRound(angle)]+=1;
        }
        float max_angle = 0.0f;
        int max_num = -1;
        for (int i = 0; i < 361; i++) {
            if (test_a[i] > max_num)
            {
                max_num = test_a[i];
                max_angle = i;
            }
            test_a[i] = 0;
        }
        kp->class_id = class_id;
        kp->angle = max_angle;
    }
    #undef GET_VALUE_Y
    #undef GET_VALUE_X
    
}
