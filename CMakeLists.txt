#project
cmake_minimum_required(VERSION 2.8)
project(sample)

SET(CMAKE_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/)

#link
find_package(OpenCV REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(Caffe REQUIRED)
find_package(cvsba REQUIRED)

#set build option
SET(CMAKE_BUILD_TYPE "Debug") 
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -std=c++0x")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wno-deprecated -g -ggdb -std=c++0x")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
SET(CMAKE_CXX_FLAGS "-msse3")

#cpu_tsdf
#link_directories(${PROJECT_SOURCE_DIR}/so)
#include_directories(${PROJECT_SOURCE_DIR}/include)

#include directory
set(SRC_DIR src)
include_directories("${SRC_DIR}")
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${Caffe_INCLUDE_DIRS})

# sources
FILE(GLOB SRC_FILES "src/*.cpp")

add_definitions(${Caffe_DEFINITIONS})
add_definitions(${PCL_DEFINITIONS})

link_directories(${PCL_LIBRARY_DIRS})

#sample program
add_executable(sample sample.cpp ${SRC_FILES})
target_link_libraries(sample ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Caffe_LIBRARIES} ${cvsba_LIBS})

add_executable(segnet segnetOnly.cpp ${SRC_FILES})
target_link_libraries(segnet ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Caffe_LIBRARIES})

add_executable(crf crf.cpp ${SRC_FILES})
target_link_libraries(crf ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Caffe_LIBRARIES})

add_executable(vo sample_3_17_best.cpp ${SRC_FILES})
target_link_libraries(vo ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Caffe_LIBRARIES})

add_executable(inin segnetOnly_inin.cpp ${SRC_FILES})
target_link_libraries(inin ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Caffe_LIBRARIES})
