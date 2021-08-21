#ifndef _ZOULEE_INCLUDE_H
#define _ZOULEE_INCLUDE_H

#include "opencv2/core/core.hpp"
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

typedef uint8_t        			  u8;
typedef uint16_t    			 u16;
typedef uint32_t				 u32;
typedef uint64_t				 u64;
typedef uint_fast8_t          u_f_8;
typedef uint_fast16_t       u_f_16;
typedef uint_fast32_t       u_f_32;
typedef uint_fast64_t       u_f_64;
typedef int8_t         		    	int8;
typedef int16_t     			  int16;
typedef int32_t       			  int32;

typedef int_fast8_t             int_f_8;
typedef int_fast16_t           int_f_16;
typedef int_fast32_t        int_f_32;
typedef int_fast64_t        int_f_64;

#define USER_MAIN_MSG
#define USER_MAIN_DEBUG
#define USER_MAIN_ERROR

#ifdef USER_MAIN_MSG
#define user_main_printf(format, ...) printf( format "\r\n", ##__VA_ARGS__)
#define user_main_info(format, ...) printf("[main]info:" format "\r\n", ##__VA_ARGS__)
#else
#define user_main_printf(format, ...)
#define user_main_info(format, ...)
#endif

#ifdef USER_MAIN_DEBUG
#define user_main_debug(format, ...) printf("[main]debug:" format "\r\n", ##__VA_ARGS__)
#else
#define user_main_debug(format, ...)
#endif

#ifdef USER_MAIN_ERROR
#define user_main_error(format, ...) printf("[main]error:" format "\r\n",##__VA_ARGS__)
#else
#define user_main_error(format, ...)
#endif

//内参矩阵
//Internal parameter matrix
//4624 x 3472
// const cv::Mat Internal_matrix = (cv::Mat_<double>(3, 3)<<4064.539918250539, 0.0, 2141.687219148065, 
//                                                                                                                         0.0, 4047.832554235512, 1652.072211112112, 
//                                                                                                                             0.0, 0.0, 1.0);
//640 x 480
 const cv::Mat Internal_matrix = (cv::Mat_<double>(3, 3)<<1421.371528580428, 0, 308.4888698211675, 
                                                                                                                            0.0, 1390.239557801079, 180.4334358290608, 
                                                                                                                            0.0, 0.0, 1.0);

// 畸变系数
//Distortion coefficient
//4624 x 3472
// const cv::Mat distCoeffs = (cv::Mat_<double>(1, 5)<<-0.09030663262807706, 0.0672808339370711, 0.007395122768881658, -0.006491803492545443, -1.279202211850705);
//640 x 480
const cv::Mat distCoeffs = (cv::Mat_<double>(1, 5)<<0.2547322410078418, -1.972376176046268, 0.1866043674965747, 0.00158570350995139, -852.5884204386055);

const u16 focal_length = 543; // 焦距

void drawkey_vector_l(std::vector<cv::KeyPoint> keypoints);
void drawkey_vector_r(std::vector<cv::KeyPoint> keypoints);

#endif // !_ZOULEE_INCLUDE_H
