#include "comment_include.h"
#include "windows.h"
#include "Angle.h"
#include "Brief.h"
#include "fstream"
#include "orb_tree.h"
#include<iomanip>

using namespace std;
using namespace cv;

vector<DMatch> real_match(vector<KeyPoint> keypointsA, vector<KeyPoint> keypointsB, vector<DMatch> final_matches);
vector<DMatch> select_match(vector<DMatch> matches);
void get_img_path(const string filepath, vector<string>  &img_paths);
void compare_two_img(Mat gray1, Mat gray2, vector<vector<int> > &match_num, vector<int> &fail_num, vector<clock_t> &cost, u8 angle_mode);
std::string getCurrentTimeStr();
int sum_mat(Mat input, Mat gs, const int WindowSize);
Mat generateGaussMask(cv::Size wsize, float sigma);
Mat RectangleGauss(Mat gray, const int WindowSize);

enum test_mode{
    V1_EasyMode=0, 
    V1_MediumMode, 
    V2_EasyMode, 
    V2_MediumMode, 
    HALL_DifficultMode
};

int main(int argc, char const *argv[])
{
    const string filepath1 = "/home/zoulee/DATAS/V1_01_easy/mav0/cam0/";
    const string filepath2 = "/home/zoulee/DATAS/V1_02_medium/mav0/cam0/";
    const string filepath3 = "/home/zoulee/DATAS/MH_04_difficult/mav0/cam0/";
    const string filepath4 = "/home/zoulee/DATAS/V2_01_easy/mav0/cam0/";
    const string filepath5 = "/home/zoulee/DATAS/V2_02_medium/mav0/cam0/";
    string realimgpath;
    u8 testmode = V1_MediumMode;
    switch(testmode)
    {
        case V1_EasyMode:
            cout <<"模式 : V1_EasyMode" << endl; 
            realimgpath = filepath1;
            break;
        case V1_MediumMode:
            cout <<"模式 : V1_MediumMode" << endl; 
            realimgpath = filepath2;
            break;
        case V2_EasyMode:
            cout <<"模式 : V2_EasyMode" << endl; 
            realimgpath = filepath4;
            break;
        case V2_MediumMode:
            cout <<"模式 : V2_MediumMode" << endl; 
            realimgpath = filepath5;
            break;
        case HALL_DifficultMode:
            cout <<"模式 : HALL_DifficultMode" << endl; 
            realimgpath = filepath3;
            break;
    }
    vector<string>  img_paths;
    get_img_path(realimgpath, img_paths);
    size_t img_size = img_paths.size();
    vector<vector<int> > match_num;
    vector<int > fail_num;
    vector<clock_t> time_cost;

    u8 angle_mode = DEFAULT_SIFT;

    for(int i = 1; i < img_size; i++)
    {
        if(i % 500 == 0)
            cout << "进度-----------------" << int(float(i) / img_size * 100) << "%" << endl;
        cv::Mat image1 = cv::imread(img_paths[i - 1], 0);
        cv::Mat image2 = cv::imread(img_paths[i], 0);
        compare_two_img(image1, image2, match_num, fail_num, time_cost, angle_mode);
    }

    waitKey(100);
    destroyAllWindows();

    long orb_real_match=0, test_real_match=0;
    long orb_match=0, test_match=0;
    long orb_true_match=0, test_true_match=0;

    const int N = match_num[0].size();
    float max_acc_orb = 0.0f, max_acc_test = 0.0f;
    float min_acc_orb = 100.0f, min_acc_test = 100.0f;
    float acc;
    for(int i = 0; i < N; i++)
    {
        orb_real_match += match_num[0][i];
        test_real_match += match_num[1][i];
        orb_match += match_num[2][i];
        test_match += match_num[3][i];
        orb_true_match += match_num[4][i];
        test_true_match += match_num[5][i];
        //orb acc
        acc = match_num[4][i] / double(match_num[2][i]) * 100.0f;
        if(acc > max_acc_orb)
            max_acc_orb = acc;
        if(acc < min_acc_orb)
            min_acc_orb = acc;
        acc = match_num[5][i] / double(match_num[3][i]) * 100.0f;
        if(acc > max_acc_test)
            max_acc_test = acc;
        if(acc < min_acc_test)
            min_acc_test = acc;
    }
    
    cout << "ORB::平均匹配个数 \t= " << orb_match / double(N) << endl;
    cout << "ORB::平均正确匹配个数 = " << orb_true_match / double(N) << endl;
    cout << "ORB::跟踪失败次数\t = " <<  fail_num[0] << endl;
    cout << "ORB::平均正确率\t = " << orb_true_match / double(orb_match ) * 100 << "%" << endl;
    cout << "ORB::最低正确率\t = " << min_acc_orb << endl;
    cout << "ORB::最高正确率\t = " << max_acc_orb << endl;
    cout << "ORB::计算1万个描述子耗时" << double(time_cost[0]) / (time_cost[3]/10000.0) / CLOCKS_PER_SEC << " s" << endl;

    cout << "TEST::平均匹配个数 \t= " << test_match / double(N) << endl;
    cout << "TEST::平均正确匹配个数 = " << test_true_match / double(N) << endl;
    cout << "TEST::跟踪失败次数\t = " <<  fail_num[1] << endl;
    cout << "TEST::平均正确率\t = " << test_true_match / double(test_match ) * 100 << "%" << endl;
    cout << "TEST::最低正确率\t = " << min_acc_test << endl;
    cout << "TEST::最高正确率\t = " << max_acc_test << endl;
    cout << "TEST::计算1万个描述子耗时" << (double(time_cost[1]) / (time_cost[4]/10000.0)) / CLOCKS_PER_SEC << " s"<< endl;

    fstream F;
    string txt_path = "/home/zoulee/c++_codes/2021newlunwen/test_res.txt";
    F.open(txt_path, ios::app);

    string time = getCurrentTimeStr();
    F << time;
    switch(testmode)
    {
        case V1_EasyMode:
            F << "\t[V1_EasyMode]" << endl;
            break;
        case V1_MediumMode:
            F << "\t[V1_MediumMode]" << endl;
            break;
        case V2_EasyMode:
            F << "\t[V2_EasyMode]" << endl;
            break;
        case V2_MediumMode:
            F << "\t[V2_MediumMode]" << endl;
            break;
        case HALL_DifficultMode:
            F << "\t[HALL_DifficultMode]" << endl;
            break;
    }
    F << "\t[ORB]" << endl;
    F << "\t\t" << "平均匹配个数 \t= " << orb_match / float(N) << endl;
    F << "\t\t" << "平均正确匹配个数 = " << orb_true_match / float(N) << endl;
    F << "\t\t" << "跟踪失败次数 = " <<  fail_num[0] << endl;
    F << "\t\t" << "平均正确率\t = " <<  orb_true_match / float(orb_match ) * 100 << "%" << endl;
    F << "\t\t" << "最低正确率\t = " << min_acc_orb << endl;
    F << "\t\t" << "最高正确率\t = " << max_acc_orb << endl;    
    F << "\t\t" << "计算1万个描述子耗时" << double(time_cost[0]) / (time_cost[3]/10000.0) / CLOCKS_PER_SEC << " s"<< endl;
    switch(angle_mode){
        case DEFAULT_GRAY:
            F << "\t[DEFAULT_GRAY]" << endl;
            break;
        case IMPROVE_GRAY:
            F << "\t[IMPROVE_GRAY]" << endl;
            break;
        case DEFAULT_SIFT:
            F << "\t[DEFAULT_SIFT]" << endl;
            break;
        case DEFAULT_FREAK:
            F << "\t[DEFAULT_FREAK]" << endl;
            break;
        case COMBINE_SIFT_GRAY:
            F << "\t[COMBINE_SIFT_GRAY]" << endl;
            break;
    }
    F << "\t\t" << "平均匹配个数 \t= " << test_match / float(N) << endl;
    F << "\t\t" << "平均正确匹配个数 = " << test_true_match / float(N) << endl;
    F << "\t\t" << "跟踪失败次数\t = " <<  fail_num[1] << endl;
    F << "\t\t" << "平均正确率\t = " << test_true_match / float(test_match ) * 100 << "%" << endl;
    F << "\t\t" << "最低正确率\t = " << min_acc_test << endl;
    F << "\t\t" << "最高正确率\t = " << max_acc_test << endl;
    F << "\t\t" << "计算1万个描述子耗时" << (double(time_cost[1]) / (time_cost[4]/10000.0)) / CLOCKS_PER_SEC << " s"<< endl;

    F.close();

    return 0;
}

std::string getCurrentTimeStr()
{
  time_t t = time(NULL);
  char ch[64] = {0};
  char result[100] = {0};
  strftime(ch, sizeof(ch) - 1, "[%Y-%m-%d\t%H:%M:%S]", localtime(&t));
  sprintf(result, "%s", ch);
  return std::string(result);
}

//0:orb_real_match个数
//1:test_real_match个数
//2:orb_match个数
//3:test_match个数
//4:orb_true_match个数
//5:test_true_match个数
void compare_two_img(Mat gray1, Mat gray2, vector<vector<int> > &match_num, vector<int> &fail_num, vector<clock_t> &cost, u8 angle_mode)
{
    if(match_num.empty())
        match_num.resize(6);
    if(fail_num.empty())
        fail_num.resize(2, 0);
    //0 cost: orb time cost
    //1 cost: test time cost
    //2 cost: count times
    //3 orb: kps num
    //4 test: kps num
    if(cost.empty())
        cost.resize(5, 0);
    // cv::resize(imageL, imageL, cv::Size(640, 480));
    if(!gray1.data)
    {
        cout << "Image 1 Error!" << endl;
        exit(1);
    }
    if(!gray2.data)
    {
        cout << "Image 2 Error!" << endl;
        exit(1);
    }
    if(gray1.channels() == 3)
        cv::cvtColor(gray1, gray1, cv::COLOR_BGR2GRAY);
    if(gray2.channels() == 3)
        cv::cvtColor(gray2, gray2, cv::COLOR_BGR2GRAY);

    vector<KeyPoint> kps_1, kps_2;
    kps_1 = Fast_win(gray1);
    kps_2 = Fast_win(gray2);

    Mat gs_1, gs_2;
    
    // gs_1 = RectangleGauss(gray1, 7);
    // gs_2 = RectangleGauss(gray2, 7);
    GaussianBlur(gray1, gs_1, Size(7, 7), 1.5, 1.5, BORDER_REFLECT);
    GaussianBlur(gray2, gs_2, Size(7, 7), 1.5, 1.5, BORDER_REFLECT);

    ZOULEE::make_tree(gs_1, kps_1, 800);
    ZOULEE::make_tree(gs_2, kps_2, 800);

    vector<KeyPoint> kps_1_copy(kps_1), kps_2_copy(kps_2);

    clock_t start, end;
    start = clock();
    Mat desc_orb1 = make_bref(kps_1_copy, gs_1, DEFAULT_GRAY);
    end = clock();
    cost[0] += end - start;
    start = clock();
    Mat desc_orb2 = make_bref(kps_2_copy, gs_2, DEFAULT_GRAY);
    end = clock();
    cost[0] += end - start;
    cost[3] += kps_1_copy.size() + kps_2_copy.size();

    start = clock();
    Mat test_desc1 = make_bref(kps_1, gs_1, angle_mode);
    end = clock();
    cost[1] += end - start;
    start = clock();
    Mat test_desc2 = make_bref(kps_2, gs_2, angle_mode);
    end = clock();
    cost[1] += end - start;
    cost[4] += kps_1.size() + kps_2.size();
    cost[2] += 2;

    vector<DMatch> orb_matches, real_matches1, real_matches2, test_matches;
    BFMatcher matcher (NORM_HAMMING, true);
    matcher.match(desc_orb1, desc_orb2, orb_matches);
    matcher.match(test_desc1, test_desc2, test_matches);
    
    vector<DMatch> next_orb, next_test;
    next_orb = select_match(orb_matches);
    next_test = select_match(test_matches);

    int class_id;
    for (int i = 0; i < next_test.size() - 1; i++)
    {
        class_id = kps_1[next_test[i].queryIdx].class_id;
        for (auto j = next_test.begin() + i + 1; j != next_test.end(); j++)
        {
            if (kps_1[j->queryIdx].class_id == class_id)
            {
                next_test.erase(j);
                break;
            }
        }
    }

    real_matches1 = real_match(kps_1_copy, kps_2_copy, orb_matches);
    real_matches2 = real_match(kps_1, kps_2, test_matches);

    if(real_matches1.size() < 15)
    {
        fail_num[0]++;
    }
    if (real_matches2.size() < 15)
    {
        fail_num[1]++;
    }
    
    match_num[0].push_back(real_matches1.size());
    match_num[1].push_back(real_matches2.size());
    
    match_num[2].push_back(next_orb.size());
    match_num[3].push_back(next_test.size());

    int test_true_num = 0, orb_true_num = 0;
    for(int i = 0; i < next_test.size(); i++) {
        for (int j = 0; j < real_matches2.size(); j++)
        {
            if(next_test[i].queryIdx == real_matches2[j].queryIdx)
            {
                if(next_test[i].trainIdx == real_matches2[j].trainIdx)
                {
                    test_true_num++;
                }
            }
        }
    }
    for(int i = 0; i < next_orb.size(); i++) {
        for (int j = 0; j < real_matches1.size(); j++)
        {
            if(next_orb[i].queryIdx == real_matches1[j].queryIdx)
            {
                if(next_orb[i].trainIdx == real_matches1[j].trainIdx)
                {
                    orb_true_num++;
                }
            }
        }
    }
    match_num[4].push_back(orb_true_num);
    match_num[5].push_back(test_true_num);
    
    // Mat img_match1, img_match2, img_orb_match, img_test_match;

    // drawMatches(gray1, kps_1_copy, gray2, kps_2_copy, next_orb, img_orb_match);
    // imshow("ORB匹配点对", img_orb_match);

    // drawMatches(gray1, kps_1, gray2, kps_2, real_matches2, img_match2);
    // imshow("TEST REAL匹配点对", img_match2);

    // drawMatches(gray1, kps_1_copy, gray2, kps_2_copy, real_matches1, img_match1);
    // imshow("ORB REAL匹配点对", img_match1);

    // drawMatches(gray1, kps_1, gray2, kps_2, next_test, img_test_match);
    // imshow("TEST匹配点对", img_test_match);

    // waitKey(1);
}

void get_img_path(const string filepath, vector<string>  &img_paths)
{
    ifstream img_path_datas(filepath+"data.csv");
    if(!img_path_datas.is_open()) {
        cout << "Error" << endl;
        exit(1);
    }
    string datas;
    vector<long>times;
    getline(img_path_datas, datas);
    while (!img_path_datas.eof()){
        getline(img_path_datas, datas);
		stringstream ss(datas);
        getline(ss, datas, ',');
        long time = atol(datas.c_str());
        if(time){
            times.push_back(time);
            img_paths.push_back(filepath+"data/"+datas+".png");
        }
    }
    img_path_datas.close();
    cout <<  "img_paths size: " << img_paths.size() << endl;
    cout <<  "get_img_path finished " << endl;
}

vector<DMatch> select_match(vector<DMatch> matches)
{
    vector<DMatch> good_matches;
    size_t N = matches.size();
    float max_dist = 0.0f, min_dist = 256.0f, dist;
    for (int i = 0; i < N; i++)
    {
        dist = matches[i].distance;
        if ( dist > max_dist)
            max_dist = dist;
        if ( dist < min_dist)
            min_dist = dist;
    }
    // cout << "最小距离/最大距离" << min_dist << "/" << max_dist << endl;
    for (int i = 0; i < N; i++)
    {
        dist = matches[i].distance;
        if ( dist < 2 * min_dist || dist < 40.0f){
            good_matches.push_back(matches[i]);
        }
    }
    return good_matches;
}

vector<DMatch> real_match(vector<KeyPoint> keypointsA, vector<KeyPoint> keypointsB, vector<DMatch> final_matches)
{
    if(keypointsA.empty())
    {
        cout << "No keypoints 1 found" << endl;
        exit(1);
    }
    if(keypointsB.empty())
    {
        cout << "No keypoints 2 found" << endl;
        exit(1);
    }
    vector<Point2f>querymatchespoints, trainmatchespoints;
    vector<KeyPoint>p1,p2;
    for(int i=0;i<final_matches.size();i++)
    {
        p1.push_back(keypointsA[final_matches[i].queryIdx]);
        p2.push_back(keypointsB[final_matches[i].trainIdx]);
        querymatchespoints.push_back(p1[i].pt);
        trainmatchespoints.push_back(p2[i].pt);
    }
    // cout<<querymatchespoints[1]<<" and "<<trainmatchespoints[1]<<endl;
    vector<uchar>status;
    Mat h = findHomography(querymatchespoints, trainmatchespoints, RANSAC, 3, status, 20000 );
    int index = 0; 
    vector<DMatch> sfinal_matches;
    for(int i=0;i<final_matches.size();i++)
    {	
        cout<<status[i];
        if(status[i]!=0)
        {
        sfinal_matches.push_back(final_matches[i]);
        index++;
        }
    }
    return sfinal_matches;
}

Mat generateGaussMask(cv::Size wsize, float sigma)
{
    cv::Mat Mask;
	Mask.create(wsize, CV_32F);
	int h = wsize.height;
	int w = wsize.width;
	int center_h = (h - 1) / 2;
	int center_w = (w - 1) / 2;
	float sum = 0.0;
	float x, y;
	for (int i = 0; i < h; ++i){
		y = pow(i - center_h, 2);
		for (int j = 0; j < w; ++j){
			x = pow(j - center_w, 2);
			//因为最后都要归一化的，常数部分可以不计算，也减少了运算量
			float g = exp(-(x + y) / (2 * sigma*sigma));
			Mask.at<float>(i, j) = g;
			sum += g;
		}
	}
    return Mask / sum;
}

Mat RectangleGauss(Mat gray, const int WindowSize)
{
    if (!gray.data)
    {
        cout << "WARNING: no img" << endl;
        exit(1);
    }
    else if (gray.channels() > 1)
    {
        cout << "WARNING: not gray" << endl;
        exit(1);
    }
    Mat output;
    output.create(Size(gray.cols, gray.rows), CV_8UC1);
    const Mat Gs = generateGaussMask(Size(WindowSize, WindowSize), 1.5f);
    const int R = WindowSize / 2;

    Mat gray_large;
    copyMakeBorder(gray, gray_large, R, R , R, R, BORDER_REFLECT);
    const int imgX = gray_large.cols, imgY = gray_large.rows;
    const int xCount = imgX -  R, yCount = imgY -  R;
    
    for(int winY= R; winY < yCount; winY++)
    {
        for(int winX = R; winX < xCount; winX++) 
        {
            Mat win = gray_large.colRange(winX - R, winX + R+1).rowRange(winY - R, winY + R+1);
            int a = sum_mat(win, Gs, WindowSize);
            output.at<u8>(winY - R, winX - R) = a;
        }
    }
    return output;
}

int sum_mat(Mat input, Mat gs, const int WindowSize)
{
    float sum=0;
    for(int y = 0; y < WindowSize; y++)
    {
        for(int x = 0; x < WindowSize; x++)
        {
            sum += gs.at<float>(y, x) * input.at<u8>(y, x);
        }
    }
    return max(0, min(cvRound(sum), 255));
}
