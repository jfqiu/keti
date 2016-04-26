#pragma once

//std
#include <iostream>
#include <dirent.h>
#include <string.h>

//thread
#include <thread>    
#include <sys/time.h>
#include <time.h>
#include <unistd.h> //time sleep

//pcl
#include <boost/thread/thread.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/crf_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/poisson.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/io/vtk_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/surface/vtk_smoothing/vtk_mesh_smoothing_laplacian.h>

//crf
#include "crf_refine.hpp"

//sba
#include <cvsba/cvsba.h>

//vo
#include "basicStructure.hpp"
#include "vo_stereo.hpp"
#include "uvdisparity.hpp"
#include "matrix_.h"

//segnet
#include "segnet.hpp"

//fusion
#include <queue>

// FOREGROUND 
#define RST  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#define FRED(x) KRED x RST
#define FGRN(x) KGRN x RST
#define FYEL(x) KYEL x RST
#define FBLU(x) KBLU x RST
#define FMAG(x) KMAG x RST
#define FCYN(x) KCYN x RST
#define FWHT(x) KWHT x RST

#define BOLD(x)	"\x1B[1m" x RST
#define UNDL(x)	"\x1B[4m" x RST

struct Fusion {
	cv::Mat depth;
	cv::Mat mov;
	cv::Mat roi;
	cv::Mat xyz;
	cv::Mat rgb;
	std::vector<Prediction> preL;
	std::vector<Prediction> preR;
	Matrix_ rt;
};

//creates a PCL visualizer
boost::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizer(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud);

//number of files
int numFiles(std::string rgb_dir);

//set calibration paramerers for odometry
void setVO(VisualOdometryStereo::parameters &param, double, double, double, double, double);

//filter
void filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudRGB, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudRGB_filtered_);

//post-process
void post_process(pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr);

//display
void display(pcl::PointCloud<pcl::PointXYZRGB>::Ptr);

//lut
void pclut(CloudLT::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr);

//3d reconstruction thread
void reconstruction(cv::Mat& img_lc, cv::Mat& img_rc, cv::Mat& img_rp, cv::Mat& img_lp, cv::Mat& disp_sgbm, cv::Mat& xyz,
                    double f, double c_u, double c_v, double base, ROI3D roi_, 
		    cv::Mat& disp_show_sgbm, VisualOdometryStereo& viso, cv::Mat& roi_mask, 
		    cv::Mat& ground_mask, double& pitch1, double& pitch2, cv::Mat& moving_mask, Matrix_& pose, UVDisparity& uv_disparity, bool& success);

//segmentation
void semantic(cv::Mat& img_rgb, Classifier& classifier, int img_cols, int cropCols, int img_rows, int cropRows,
		int cropCols2, cv::Mat& segnet, cv::Mat& colormap, std::vector<Prediction> & predictionsL, std::vector<Prediction> & predictionsR);

//mesh
void postMesh(pcl::PointCloud<pcl::PointXYZRGB>::Ptr);










