#include "windows.h"

using namespace std;
using namespace cv;

//窗口计算关键点
vector<KeyPoint > Fast_win(Mat &gray)
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
    copyMakeBorder(gray, gray, 18, 18, 18, 18, BORDER_REFLECT);
    const int WindowSize = 30;
    const int imgX = gray.cols - 15, imgY = gray.rows - 15;
    int minX, maxX, minY, maxY;
    const int xCount = ((imgX - 15) % WindowSize) == 0 ?  (imgX - 15) / WindowSize : ((imgX - 15) / WindowSize) + 1, yCount = ((imgY - 15) % WindowSize) == 0 ?  (imgY - 15) / WindowSize : ((imgY - 15) / WindowSize) + 1;
    vector<KeyPoint > kps;
    for(int winY= 0; winY < yCount; winY++)
    {
        minY = winY * WindowSize + 15;
        maxY = minY + WindowSize;
        if(maxY > imgY)
        {
            maxY = imgY;
        }
        for(int winX = 0; winX < xCount; winX++) 
        {
            minX = winX * WindowSize + 15;
            maxX = minX + WindowSize;
            if(maxX > imgX)
            {
                maxX = imgX;
            }
            Mat win = gray.colRange(minX, maxX).rowRange(minY, maxY);
            // float th = ContrastRatio(win), threshold;
            // if ( th < 250){
            //     threshold = th * 0.9f + 7.2f  < 4 ? 4 : th * 0.9f + 7.2f;
            // }
            // else{
            //     threshold = 7.2f * log(th) + 4.3f;
            // }
            vector<KeyPoint > kps_temp;
            // FAST(win, kps_temp, threshold);
            FAST(win, kps_temp, 20);
            if (kps_temp.empty())
                FAST(win, kps_temp, 10);
            if ( !kps_temp.empty())
            {
                for(auto kp = kps_temp.begin();kp != kps_temp.end(); kp++)
                {
                    kp->pt.x += minX;
                    kp->pt.y += minY;
                    kps.push_back(*kp);
                }
            }
        }
    }
    // cout  << "kps: " << kps.size() << endl;
    return kps;
}
//计算对比度
float ContrastRatio(Mat anko)
{
    Mat mi;
    int row = anko.rows;
    int col = anko.cols;
    mi.create(row, col, anko.type());
    int gray[256] = { 0 };
    float gray_prob[256] = { 0.0f };
    int num = 0;
    int max_gray_value = 0, min_gray_value = 256;
    for (int i = 0; i < row-1; i++)
    {
        uchar *p = mi.ptr<uchar>(i);
        for (int j = 0; j < col; j++)
        {   
            if(max_gray_value < anko.at<u8>(i, j))
            {
                max_gray_value = anko.at<u8>(i, j);
            }
            else if(min_gray_value > anko.at<u8>(i, j))
            {
                min_gray_value = anko.at<u8>(i, j);
            }
            mi.ptr<uchar>(i)[j] = abs(anko.ptr<uchar>(i+1)[j] - anko.ptr<uchar>(i)[j]);
            int value = p[j];
            gray[value]++;
            num++;
        }
    }
    float CON = 0;
    for (int i = 0; i < 256; i++)
    {
        gray_prob[i] = ((float)gray[i] / num);
        CON += i*i*gray_prob[i];
    }
    return CON;
}
