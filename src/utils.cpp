#include "utils.h"

int numFiles(std::string rgb_dir)
{
	//number of files(count)
	char directory[256]; strcpy(directory, rgb_dir.c_str()); 
	struct dirent *de;
	DIR *dir = opendir(directory);
	int count = 0;	
	while ((de = readdir(dir))) ++count;
	closedir(dir);
	return count;
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizer (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor (255, 255, 255);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "reconstruction");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "reconstruction");
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters ();
	return (viewer);
}


void setVO(VisualOdometryStereo::parameters &param, double f, double c_u, double c_v, double base, double inlier_threshold)
{
	
	param.calib.f  = f;      param.calib.cu = c_u;
	param.calib.cv = c_v;    param.base     = base;	
	param.inlier_threshold = inlier_threshold;
}

void filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudRGB, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudRGB_filtered_)
{
	//outlier removal
	//printf("before outlier removal: %d\n", (int) cloudRGB->points.size());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudRGBfiltered(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor_;
	sor_.setInputCloud(cloudRGB);
	sor_.setMeanK(50);
	sor_.setStddevMulThresh(1.0);
	sor_.filter(*cloudRGBfiltered);
	//printf("after outlier removal: %d\n", (int) cloudRGBfiltered->points.size());

	//voxel grid filter
	////printf("before filted: %d\n", (int) cloudRGBfiltered->points.size());
	pcl::VoxelGrid<pcl::PointXYZRGB> sor;
	sor.setInputCloud (cloudRGBfiltered);
	sor.setLeafSize (0.1f, 0.1f, 0.1f);
	sor.filter (*cloudRGB_filtered_);
	printf("after filted: %d\n", (int) cloudRGB_filtered_->points.size());
}

void post_process(pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudRGB_filtered)
{
	//outlier removal
	//printf("before outlier removal: %d\n", (int) point_cloud_ptr->points.size());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor_;
	sor_.setInputCloud(point_cloud_ptr);
	sor_.setMeanK(50);
	sor_.setStddevMulThresh(1.0);
	sor_.filter(*point_cloud_ptr_filtered);
	//printf("after outlier removal: %d\n", (int) point_cloud_ptr_filtered->points.size());

	//voxel grid filter
	//printf("before filted: %d\n", (int) point_cloud_ptr_filtered->points.size());
	pcl::VoxelGrid<pcl::PointXYZRGB> sor;
	sor.setInputCloud (point_cloud_ptr_filtered);
	sor.setLeafSize (0.2f, 0.2f, 0.2f);
	sor.filter (*cloudRGB_filtered);
	//printf("after filted: %d\n", (int) cloudRGB_filtered->points.size());
}

void display(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudRGB_filtered)
{
	//display
	cloudRGB_filtered->width = (int) cloudRGB_filtered->points.size();
	cloudRGB_filtered->height = 1;
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor (255, 255, 255);
	viewer->initCameraParameters ();
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloudRGB_filtered);
	viewer->addPointCloud<pcl::PointXYZRGB> (cloudRGB_filtered, rgb, "reconstruction");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "reconstruction");
	std::cout << BOLD(FCYN("The total length: ")) << cloudRGB_filtered->width << " points.\n";
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void pclut(CloudLT::Ptr out, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudRGB)
{
	pcl::PointXYZRGB point;
	uchar pr, pg, pb;
	for (size_t i = 0; i < out->points.size(); ++i)
	{
		point.x = out->points[i].x; 
		point.y = out->points[i].y; 
		point.z = out->points[i].z;
		switch (out->points[i].label)
		{
			case 0: pb=128; pg=128; pr=128; break; //sky
			case 1: pb=0; pg=0; pr=128; break; 
			case 2: pb=128; pg=192; pr=192; break; 
			case 3: pb=0; pg=69; pr=255; break; 
			case 4: pb=128; pg=64; pr=128; break;
			case 5: pb=222; pg=40; pr=60; break; 
			case 6: pb=0; pg=128; pr=128; break; 
			case 7: pb=128; pg=128; pr=192; break; 
			case 8: pb=128; pg=64; pr=64; break; 
			case 9: pb=128; pg=0; pr=64; break; //car
			case 10: pb=0; pg=64; pr=64; break; //pedestrian
			case 11: pb=192; pg=128; pr=0; break; //cyclist
		}
		uint32_t rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
		point.rgb = *reinterpret_cast<float*>(&rgb);
		cloudRGB->points.push_back(point);
	}
}


void reconstruction(cv::Mat& img_lc, cv::Mat& img_rc, cv::Mat& img_rp, cv::Mat& img_lp, cv::Mat& disp_sgbm, cv::Mat& xyz,
                    double f, double c_u, double c_v, double base, ROI3D roi_, 
		    cv::Mat& disp_show_sgbm, VisualOdometryStereo& viso, cv::Mat& roi_mask, 
		    cv::Mat& ground_mask, double& pitch1, double& pitch2, cv::Mat& moving_mask, Matrix_& pose, UVDisparity& uv_disparity, bool& success)
{
	success = false;

	//detect and matching feature points circlely in four images
	QuadFeatureMatch* quadmatcher = new QuadFeatureMatch(img_lc,img_rc,img_lp,img_rp,true);
	quadmatcher->init(DET_GFTT,DES_SIFT);
	quadmatcher->detectFeature();
	quadmatcher->circularMatching();

	//visual odometry valid
	if(viso.Process(*quadmatcher)==true)
	{
		//get ego-motion matrix (6DOF)
		cv::Mat motion;
		motion = viso.getMotion();

		//computing disparity image (SGBM or BM method) and 3D reconstruction by triangulation
		calDisparity_SGBM(img_lc, img_rc, disp_sgbm, disp_show_sgbm);
		triangulate10D(img_lc, disp_sgbm, xyz, f, c_u, c_v, base, roi_);

		//analyzing the stereo image by U-V disparity images (here I put moving object detection result)
		moving_mask = uv_disparity.Process(img_lc, disp_sgbm, viso, xyz, roi_mask, ground_mask, pitch1, pitch2);
		cv::imshow("roi", roi_mask);

		//visual odometry
		Matrix_ M = Matrix_::eye(4);
		for (int32_t i=0; i<4; ++i)
			for (int32_t j=0; j<4; ++j)
				M.val[i][j] = motion.at<double>(i,j);
		pose = pose * Matrix_::inv(M);
		success = true;
	}
	delete quadmatcher;
}

void semantic(cv::Mat& img_rgb, Classifier& classifier, int img_cols, int cropCols, int img_rows, int cropRows,
		int cropCols2, cv::Mat& segnet, cv::Mat& colormap, std::vector<Prediction> & predictionsL, std::vector<Prediction> & predictionsR)
{
		//img_rgb crop
		cv::Rect croprgbL(img_cols/2-cropCols,img_rows-cropRows,cropCols,cropRows);
		cv::Rect croprgbR(img_cols/2,img_rows-cropRows,cropCols,cropRows);
		cv::Mat img_rgbL, img_rgbR;
		img_rgb(croprgbL).copyTo(img_rgbL);
		img_rgb(croprgbR).copyTo(img_rgbR);
		cv::resize(img_rgbL, img_rgbL, Size(cropCols,cropRows));
		cv::resize(img_rgbR, img_rgbR, Size(cropCols,cropRows));

		//prediction
		predictionsL = classifier.Classify(img_rgbL);
		predictionsR = classifier.Classify(img_rgbR);
		for (int i = 0; i < cropRows; ++i)
		{	
			uchar* segnet_ptr = segnet.ptr<uchar>(i);
			for (int j = 0; j < cropCols; ++j)
			{
				segnet_ptr[j*3] = predictionsL[i*cropCols+j].second;
				segnet_ptr[j*3+1] = predictionsL[i*cropCols+j+1].second;
				segnet_ptr[j*3+2] = predictionsL[i*cropCols+j+2].second;

				segnet_ptr[(j+cropCols)*3] = predictionsR[i*cropCols+j].second;
				segnet_ptr[(j+cropCols)*3+1] = predictionsR[i*cropCols+j+1].second;
				segnet_ptr[(j+cropCols)*3+2] = predictionsR[i*cropCols+j+2].second;
			}
		}

		//middle filter
		int boxWidth = 2;
		for (int i = 0; i < cropRows; ++i)
		{	
			uchar* segnet_ptr = segnet.ptr<uchar>(i);
			uchar* maxLabel = new uchar[boxWidth*2];
			memset(maxLabel, 0, sizeof(uchar)*boxWidth*2);
			for (int j = 0; j < cropCols2; ++j)
			{
				if (j>=cropCols-boxWidth && j<cropCols+boxWidth)
				{
					int index = j - (cropCols - boxWidth); 
					maxLabel[index] = segnet_ptr[j*3];
				}
			}
			uchar maxlab = 0;
			for (int m = 1; m < boxWidth*2; m++)
			{
				if (maxLabel[maxlab] < maxLabel[m]) maxlab = m;
			}
			
			for (int k = cropCols-boxWidth; k < cropCols+boxWidth; k++)
			{
				segnet_ptr[k*3] = maxLabel[maxlab];
				segnet_ptr[k*3+1] = maxLabel[maxlab];
				segnet_ptr[k*3+2] = maxLabel[maxlab];
			}
			delete[] maxLabel;
		}

		//LUT
		cv::LUT(segnet, colormap, segnet);
}

void postMesh(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudRGB_filtered_)
{
	TicToc tt;
	tt.tic ();

	printf("start meshing...\n");
	// Normal estimation*
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
	tree->setInputCloud (cloudRGB_filtered_);
	n.setInputCloud (cloudRGB_filtered_);
	n.setSearchMethod (tree);
	n.setKSearch (100);
	n.compute (*normals);

	// Concatenate the XYZRGB and normal fields*
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::concatenateFields (*cloudRGB_filtered_, *normals, *cloud_with_normals);

	// Create search tree*
	pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
	tree2->setInputCloud (cloud_with_normals);

	// Initialize objects
	pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;
	pcl::PolygonMesh triangles;

	// Set the maximum distance between connected points (maximum edge length)
	gp3.setMu (100); //search maxmimum distance
	gp3.setMaximumNearestNeighbors (100);
	gp3.setSearchRadius (2500);
	gp3.setMinimumAngle(M_PI/90); // 10 degrees
	gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
	gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
	gp3.setNormalConsistency(false);

	// Get result
	gp3.setInputCloud (cloud_with_normals);
	gp3.setSearchMethod (tree2);
	gp3.reconstruct (triangles);

	print_info ("[Meshing done, "); 
	print_value ("%g", tt.toc ()); 
	print_info (" ms]\n"); 

	pcl::visualization::PCLVisualizer::Ptr viewer_mesh;
	viewer_mesh.reset(new pcl::visualization::PCLVisualizer);    
	viewer_mesh->addPolygonMesh(triangles, "polygon");
	viewer_mesh->spin();
}

