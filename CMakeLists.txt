cmake_minimum_required(VERSION 2.8)
project(test1)

set( CMAKE_CXX_FLAGS "-std=c++11" )
# set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_BUILD_TYPE "Release" )

find_package( OpenCV 4 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${PROJECT_SOURCE_DIR}/include )

add_library(lw_lib
./src/Angle.cpp
./src/windows.cpp
./src/Brief.cpp
./src/orb_tree.cpp
)


add_executable(test1 ./test/test_angle_new.cpp) # 生成可执行文件

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
target_link_libraries(${PROJECT_NAME} lw_lib )