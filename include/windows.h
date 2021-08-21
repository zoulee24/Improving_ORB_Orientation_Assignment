#ifndef _WINDOWS_H
#define _WINDOWS_H
#include "comment_include.h"

using namespace std;
using namespace cv;

//窗口计算关键点
vector<KeyPoint > Fast_win(Mat &gray);
float ContrastRatio(Mat anko);

#endif // !_WINDOWS_H

