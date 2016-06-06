#include "crf_refine.hpp"

void compute (const CloudT::Ptr &cloud, 
         const CloudLT::Ptr &anno,
         float normal_radius_search,
         float leaf_x, float leaf_y, float leaf_z,
         CloudLT::Ptr &out)
{
	TicToc tt;
	tt.tic ();

	print_highlight ("Computing ");

	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals (new pcl::PointCloud<pcl::PointNormal>);
	cloud_normals->width = cloud->width;
	cloud_normals->height = cloud->height;
	cloud_normals->points.resize (cloud->points.size ());
	for (size_t i = 0; i < cloud->points.size (); i++)
	{
		cloud_normals->points[i].x = cloud->points[i].x;
		cloud_normals->points[i].y = cloud->points[i].y;
		cloud_normals->points[i].z = cloud->points[i].z;
	}

	// estimate surface normals
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::PointNormal> ne;
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
	ne.setSearchMethod (tree);
	ne.setInputCloud (cloud);
	//ne.setRadiusSearch (normal_radius_search);
	ne.setKSearch(15);
	ne.compute (*cloud_normals);

	/** \brief Set the appearanche kernel parameters.
	* \param[in] sx standard deviation x
	* \param[in] sy standard deviation y
	* \param[in] sz standard deviation z
	* \param[in] sr standard deviation red
	* \param[in] sg standard deviation green
	* \param[in] sb standard deviation blue
	* \param[in] w weight
	*/

	pcl::CrfSegmentation<pcl::PointXYZRGB> crf;
	crf.setInputCloud (cloud);
	crf.setNormalCloud (cloud_normals);
	crf.setAnnotatedCloud (anno);
	crf.setVoxelGridLeafSize (leaf_x, leaf_y, leaf_z);
	crf.setSmoothnessKernelParameters (3, 3, 3, 2.5); //2.5
	crf.setAppearanceKernelParameters (30, 30, 30, 20, 20, 20, 4.0); //3.5
	crf.setSurfaceKernelParameters (20, 20, 20, 0.3f, 0.3f, 0.3f, 1.0); //1.0

	crf.setNumberOfIterations (10);
	crf.segmentPoints (*out);

	print_info ("[done, "); 
	print_value ("%g", tt.toc ()); 
	print_info (" ms : "); print_value ("%d", out->width * out->height); 
	print_info (" points]\n");
}
