#include <pcl/io/pcd_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>

#include <pcl/segmentation/crf_segmentation.h>
#include <pcl/features/normal_3d.h>

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

typedef pcl::PointCloud<pcl::PointXYZRGB> CloudT;
typedef pcl::PointCloud<pcl::PointXYZRGBL> CloudLT;

void compute (const CloudT::Ptr &cloud, 
         const CloudLT::Ptr &anno,
         float normal_radius_search,
         float leaf_x, float leaf_y, float leaf_z,
         CloudLT::Ptr &out);
