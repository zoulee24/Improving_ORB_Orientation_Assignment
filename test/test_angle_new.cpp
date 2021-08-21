#include "test_include.h"

using namespace std;
using namespace cv;


#define UVIEWABLE

int main(int argc, char const *argv[])
{
    string tm1, am1, tm2, am2;
    if (argc != 3)
    {
        for (int i = 0; i < argc; i++)
        {
            cout << "输入" << argv[i] << endl;
        }
        
        std::cout << "请输入 测试地图ID, 测试角度ID" << endl;
        return 0;
    }


    am1 = argv[1];
    tm1 = argv[2];

    u8 testmode = atoi(am1.c_str());
    u8 angle_mode = atoi(tm1.c_str());
    test(testmode, angle_mode);


    return 0;
}

void test(u8 testmode, u8 angle_mode)
{
    const string filepath1 = "/home/zoulee/DATAS/V1_01_easy/mav0/cam0/";
    const string filepath2 = "/home/zoulee/DATAS/V1_02_medium/mav0/cam0/";
    const string filepath3 = "/home/zoulee/DATAS/MH_04_difficult/mav0/cam0/";
    const string filepath4 = "/home/zoulee/DATAS/V2_01_easy/mav0/cam0/";
    const string filepath5 = "/home/zoulee/DATAS/V2_02_medium/mav0/cam0/";
    const string filepath6 = "/home/zoulee/DATAS/V1_03_difficult/mav0/cam0/";
    const string filepath7 = "/home/zoulee/DATAS/V2_03_difficult/mav0/cam0/";
    const string filepath8 = "/home/zoulee/DATAS/fr1_xyz/";
    const string filepath9 = "/home/zoulee/DATAS/fr1_rpy/";
    const string filepath10 = "/home/zoulee/DATAS/fr1_floor/";
    const string filepath11 = "/home/zoulee/DATAS/fr2_pioneer_slam/";
    string realimgpath;
    vector<string>  img_paths;

    fstream F;
    string txt_path = "./实验数据-1.txt";
    F.open(txt_path, ios::app);
    string time = getCurrentTimeStr();
    F << time;

    switch(testmode)
    {
        case V1_EasyMode:
            cout <<"简单模式 : V1_EasyMode" << endl; 
            F << "\t[EuRoc]\t[V1_EasyMode]" << endl;
            realimgpath = filepath1;
            break;
        case V1_MediumMode:
            cout <<"中等模式 : V1_MediumMode" << endl; 
            F << "\t[EuRoc]\t[V1_MediumMode]" << endl;
            realimgpath = filepath2;
            break;
        case V1_DifficultMode:
            F << "\t[EuRoc]\t[V1_DifficultMode]" << endl;
            cout <<"困难模式 : V1_DifficultMode" << endl; 
            realimgpath = filepath6;
            break;
        case V2_EasyMode:
            F << "\t[EuRoc]\t[V2_EasyMode]" << endl;
            cout <<"简单模式 : V2_EasyMode" << endl; 
            realimgpath = filepath4;
            break;
        case V2_MediumMode:
            F << "\t[EuRoc]\t[V2_MediumMode]" << endl;
            cout <<"中等模式 : V2_MediumMode" << endl; 
            realimgpath = filepath5;
            break;
        case V2_DifficultMode:
            F << "\t[EuRoc]\t[V2_DifficultMode]" << endl;
            cout <<"中等模式 : V2_DifficultMode" << endl; 
            realimgpath = filepath7;
            break;
        case HALL_DifficultMode:
            cout <<"困难模式 : HALL_DifficultMode" << endl; 
            F << "\t[EuRoc]\t[HALL_DifficultMode]" << endl;
            realimgpath = filepath3;
            break;
        case FR1XYZ:
            cout <<"旋转模式 : FR1XYZ" << endl; 
            F << "\t[TUM]\t[FR1XYZ]" << endl;
            realimgpath = filepath8;
            break;
        case FR1RPY:
            cout <<"旋转模式 : FR1RPY" << endl; 
            F << "\t[TUM]\t[FR1RPY]" << endl;
            realimgpath = filepath9;
            break;
        case FR1FLOOR:
            cout <<"旋转模式 : FR1FLOOR" << endl; 
            F << "\t[TUM]\t[FR1FLOOR]" << endl;
            realimgpath = filepath10;
            break;
        case FR2PIONEERSLAM:
            cout <<"旋转模式 : FR2PIONEERSLAM" << endl; 
            F << "\t[TUM]\t[FR2PIONEERSLAM]" << endl;
            realimgpath = filepath11;
            break;
            
    }
    
    switch(angle_mode){
        case DEFAULT_GRAY:
            F << "\t[ORB]" << endl;
            break;
        case IMPROVE_GRAY:
            F << "\t[IMPROVE_GRAY]" << endl;
            break;
        case COM_GRAY:
            F << "\t[COM_GRAY]" << endl;
            break;
        case DEFAULT_SIFT:
            F << "\t[DEFAULT_SIFT]" << endl;
            break;
        case  DEFAULT_FREAK:
            F << "\t[DEFAULT_FREAK]" << endl;
            break;
        case COMBINE_SIFT_GRAY:
            F << "\t[COMBINE_SIFT_GRAY]" << endl;
            break;
        case DEFAULT_SIFT_1:
            F << "\t[DEFAULT_SIFT_1]" << endl;
            break;
        case DEFAULT_SIFT_2:
            F << "\t[DEFAULT_SIFT_2]" << endl;
            break;
        case DEFAULT_SIFT_3:
            F << "\t[DEFAULT_SIFT_3]" << endl;
            break;
        case QUICK_SIFT:
            F << "\t[QUICK_SIFT]" << endl;
            break;
    }
    
    int N = 0;
    vector<long > match_num;
    int fail_num = 0;
    vector<clock_t> time_cost;
    try{
        get_img_path(realimgpath, img_paths);
        size_t img_size = img_paths.size();
        for(int i = 1; i < img_size; i++)
        {
            if(i % 500 == 0)
                cout << "进度-----------------" << int(float(i) / img_size * 100) << "%" << endl;
            cv::Mat image1 = cv::imread(img_paths[i - 1], 0);
            cv::Mat image2 = cv::imread(img_paths[i], 0);

            #ifdef VIEWABLE
            cout << img_paths[i - 1] << "\t" << img_paths[i] << endl;
            #endif // VIEWABLE
            compare_two_img(image1, image2, match_num, fail_num, time_cost, angle_mode);
            N++;
        }
        destroyAllWindows();
    }
    catch(...) {
        F << "================================" << endl;
        F << "==============Error==============" << endl;
        F << "================================" << endl;
        F.close();
    }
    
    cout << "平均匹配个数 \t= " << match_num[1] / double(N) << endl;
    cout << "平均正确匹配个数 = " << match_num[2] / double(N) << endl;
    cout << "跟踪失败次数\t = " << fail_num << endl;
    cout << "平均正确率\t = " << match_num[2] / double(match_num[1]) * 100 << "%" << endl;
    cout << "计算1万个描述子耗时" << double(time_cost[0]) / (time_cost[2]/10000.0) / CLOCKS_PER_SEC << " s" << endl;
    cout << "总耗时" << double(time_cost[3]) / CLOCKS_PER_SEC << " s" << endl;

    F << "\t\t" << "平均匹配个数 \t= " << match_num[1] / double(N) << endl;
    F << "\t\t" << "平均正确匹配个数 = " << match_num[2] / double(N) << endl;
    F << "\t\t" << "跟踪失败次数\t = " << fail_num << endl;
    F << "\t\t" << "平均正确率\t = "  << match_num[2] / double(match_num[1]) * 100 << "%" << endl;
    F << "\t\t" << "计算1万个描述子耗时" << double(time_cost[0]) / (time_cost[2]/10000.0) / CLOCKS_PER_SEC << " s" << endl;
    F << "\t\t" << "总耗时" << double(time_cost[3]) / CLOCKS_PER_SEC << " s" << endl;

    F.close();
}

void compare_two_img(Mat gray1, Mat gray2, vector<long > &match_num, int &fail_num, vector<clock_t> &cost, u8 angle_mode)
{
//0:real_match个数
//1:select match个数
//2:true_match个数
    if(match_num.empty())
        match_num.resize(3, 0);
    //0     orb time cost
    //1     count times
    //2     kps num
    //3     all time
    if(cost.empty())
        cost.resize(4, 0);
        
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

    clock_t all_start , all_end;
    all_start = clock();

    vector<KeyPoint> kps_1, kps_2;
    kps_1 = Fast_win(gray1);
    kps_2 = Fast_win(gray2);
    
    // ZOULEE::make_tree(gray1, kps_1, 800);
    // ZOULEE::make_tree(gray2, kps_2, 800);

    Mat gs_1, gs_2;
    GaussianBlur(gray1, gs_1, Size(7, 7), 1.5, 1.5, BORDER_REFLECT);
    GaussianBlur(gray2, gs_2, Size(7, 7), 1.5, 1.5, BORDER_REFLECT);

    clock_t start, end;
    start = clock();
    Mat desc_1 = make_bref(kps_1, gs_1, angle_mode);
    Mat desc_2 = make_bref(kps_2, gs_2, angle_mode);
    end = clock();
    all_end = end;
    cost[0] += end - start;
    cost[1] += 2;
    cost[2] += kps_1.size() + kps_2.size();
    cost[3] += all_end - all_start;

    all_start = clock();
    vector<DMatch> matches, real_matches;
    BFMatcher matcher (NORM_HAMMING, true);
    matcher.match(desc_1, desc_2, matches);
    
    vector<DMatch> next_matches;
    next_matches = select_match(matches);
    all_end = clock();
    cost[3] += all_end - all_start;

    int class_id_1, class_id_2;
    for (int i = 0; i < next_matches.size() - 1; i++)
    {
        class_id_1 = kps_1[next_matches[i].queryIdx].class_id;
        class_id_2 = kps_2[next_matches[i].trainIdx].class_id;
        for (auto j = next_matches.begin() + i + 1; j != next_matches.end(); j++)
        {
            if (kps_1[j->queryIdx].class_id == class_id_1)
            {
                next_matches.erase(j);
                break;
            }
            if (kps_2[j->trainIdx].class_id == class_id_2)
            {
                next_matches.erase(j);
                break;
            }
        }
    }

    real_matches = real_match(kps_1, kps_2, matches);
    int last_fail_num = fail_num;

    if(real_matches.size() < 15)
    {
        fail_num++;
    }
    else{
        match_num[0] += real_matches.size();
    }
    match_num[1] += next_matches.size();

    int true_num = 0;
    for(int i = 0; i < next_matches.size(); i++) {
        for (int j = 0; j < real_matches.size(); j++)
        {
            if(next_matches[i].queryIdx == real_matches[j].queryIdx)
            {
                if(next_matches[i].trainIdx == real_matches[j].trainIdx)
                {
                    true_num++;
                }
            }
        }
    }
    match_num[2] += true_num;
    
    Mat img_match, img_real_match;

    #ifdef VIEWABLE
    drawMatches(gray1, kps_1, gray2, kps_2, next_matches, img_match);
    imshow("匹配点对", img_match);

    drawMatches(gray1, kps_1, gray2, kps_2, real_matches, img_real_match);
    imshow("REAL匹配点对", img_real_match);
    if(last_fail_num != fail_num)
    {

        cout << "kps_2[1].angle = " << kps_2[1].angle << "\tkps_2[2].angle = " << kps_2[2].angle << endl;
        waitKey(0);
    }
    waitKey(1);
    #endif // VIEWABLE
}

void get_img_path(const string filepath, vector<string>  &img_paths)
{
    ifstream img_path_datas(filepath+"data.csv");
    if(img_path_datas.is_open()) {
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
    }
    else
    {
        img_path_datas.open(filepath + "rgb.txt");
        if(!img_path_datas.is_open()) {
            cout << "img path Error!" << endl;
            cout << "Error: " << filepath + "rgb.txt" << endl;
            exit(1);
        }
        string datas;
        vector<long>times;
        getline(img_path_datas, datas);
        while (!img_path_datas.eof()){
            getline(img_path_datas, datas);
            stringstream ss(datas);
            getline(ss, datas, ' ');
            if(!datas.empty())
                img_paths.push_back(filepath+"rgb/"+datas + ".png");
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

std::string getCurrentTimeStr()
{
  time_t t = time(NULL);
  char ch[64] = {0};
  char result[100] = {0};
  strftime(ch, sizeof(ch) - 1, "[%Y-%m-%d\t%H:%M:%S]", localtime(&t));
  sprintf(result, "%s", ch);
  return std::string(result);
}
