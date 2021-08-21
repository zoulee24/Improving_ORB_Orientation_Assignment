#ifndef _TEST_INCLUDE_H
#define _TEST_INCLUDE_H

#include "comment_include.h"
#include "windows.h"
#include "Angle.h"
#include "Brief.h"
#include "fstream"
#include "orb_tree.h"
#include <thread>

using namespace std;
using namespace cv;

enum test_mode{
    V1_EasyMode=1, 
    V1_MediumMode, 
    V1_DifficultMode, 
    V2_EasyMode, 
    V2_MediumMode, 
    V2_DifficultMode, 
    HALL_DifficultMode, 
    FR1XYZ,
    FR1RPY, 
    FR1FLOOR, 
    FR2PIONEERSLAM
};


vector<DMatch> real_match(vector<KeyPoint> keypointsA, vector<KeyPoint> keypointsB, vector<DMatch> final_matches);
vector<DMatch> select_match(vector<DMatch> matches);
void get_img_path(const string filepath, vector<string>  &img_paths);
void compare_two_img(Mat gray1, Mat gray2, vector<long > &match_num, int &fail_num, vector<clock_t> &cost, u8 angle_mode);
std::string getCurrentTimeStr();
int sum_mat(Mat input, Mat gs, const int WindowSize);
Mat generateGaussMask(cv::Size wsize, float sigma);
Mat RectangleGauss(Mat gray, const int WindowSize);
void test(u8 testmode, u8 angle_mode);

#endif // !_TEST_INCLUDE
