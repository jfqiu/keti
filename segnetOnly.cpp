#include "utils.h"

#include <pcl/io/png_io.h>
//stereo params
double f     = 707.0912; //focal length in pixels
double c_u   = 601.8873;  //principal point (u-coordinate) in pixels
double c_v   = 183.1104;  //principal point (v-coordinate) in pixels
double base  = 0.537904488; //baseline in meters
double inlier_threshold = 6.0f; //RANSAC parameter for classifying inlier and outlier

//roi space
int roix = 30;
int roiy = 30; //0006-10
int roiz = 30;

//colormap
string colorfile = "/home/inin/SegNet/caffe-segnet/examples/segnet/color.png";

//size
int cropRows = 360;
int cropCols = 480;
int cropCols2 = 960;

string rgb_dirL = "/media/inin/data/tracking dataset/training/image_02/0010/"; //tt0006 tr0013 tt0011 tt0003  tr0009 173-213
string rgb_dirR = "/media/inin/data/tracking dataset/training/image_03/0010/";
//data directory
//string rgb_dirL = "/media/inin/data/sequences/07/image_2/";
//string rgb_dirR = "/media/inin/data/sequences/07/image_3/";
//string velodyne_dir = "/media/inin/data/dataset/sequences/07/velodyne/";

//crf param
float  default_leaf_size = 0.02; //0.05f 0.03f is also valid
double default_feature_threshold = 5.0; //5.0
double default_normal_radius_search = 0.5; //0.03


int main() 
{
	ParameterReader pd("../data/poses/07.txt");
	//set visual odometry parameters
	VisualOdometryStereo::parameters param; 
	setVO(param, f, c_u, c_v, base, inlier_threshold);
	VisualOdometryStereo viso(param);
	ROI3D roi_(roix,roiy,roiz); //set the ROI region for the field of view
	Classifier classifier; //load network
	cv::Mat colormap = cv::imread(colorfile, 1); //look-up table
	//int count = numFiles(rgb_dirL); //frame list

	Matrix_ pose = Matrix_::eye(4); //visual odometry
	Matrix_ gtpose_ = Matrix_::eye(4); //visual odometry

	//initialize the parameters of UVDisparity
	CalibPars calib_(f,c_u,c_v,base);
	UVDisparity uv_disparity;
	uv_disparity.SetCalibPars(calib_); uv_disparity.SetROI3D(roi_);
	uv_disparity.SetOutThreshold(6.0f); uv_disparity.SetInlierTolerance(3);
	uv_disparity.SetMinAdjustIntense(20);

	//fusion
	cv::Mat key_moving_mask, key_roi_mask, key_xyz, key_disp_sgbm, key_img_rgb, key_moving_mask_before;
	std::vector<Prediction> key_preL;
	std::vector<Prediction> key_preR;
	std::vector<Fusion> vec;	
	Matrix_ key_pose = Matrix_::eye(4);
	size_t keyFrameT = 0;
	//size_t fusionLen = 1; double translationT = 3; double rotationT = 5; double RT_Threshold = 5;//translation 10_m  rotation 5_degree
	size_t fusionLen = 1; double translationT = 1; double rotationT = 1; double RT_Threshold = 1;//translation 10_m  rotation 5_degree

	//pcl
    pcl::visualization::CloudViewer rgbviewer( "rgbviewer" );

	/*********************** loop **********************/
	//time elapse
	struct timeval t_start, t_end;
	long seconds, useconds;
	double duration;
	gettimeofday(&t_start, NULL);    

	for (int n = 0; n < 120; ++n)
	{
		//variables
		cv::Mat moving_mask, roi_mask, xyz, disp_sgbm, disp_show_sgbm, ground_mask;
		cv::Mat img_lc,img_lp,img_rc,img_rp,img_rgb,img_semantic;
		Matrix_ poseChanged = Matrix_::eye(4);
		cv::Mat segnet(cropRows, cropCols2, CV_8UC3, cv::Scalar(0,0,0));
		double pitch1, pitch2;
		bool success = false;
		std::vector<Prediction> predictionsL;
		std::vector<Prediction> predictionsR;

		//read consecutive stereo images for 3d reconstruction
		char base_name_p[256], base_name_c[256];
		sprintf(base_name_p, "%06d.png", n);
		sprintf(base_name_c, "%06d.png", n+1);

		string imgpath_lp = rgb_dirL + base_name_p; string imgpath_rp = rgb_dirR + base_name_p;
		string imgpath_lc = rgb_dirL + base_name_c; string imgpath_rc = rgb_dirR + base_name_c;
		img_lc = cv::imread(imgpath_lc,0); img_lp = cv::imread(imgpath_lp,0);
		img_rc = cv::imread(imgpath_rc,0); img_rp = cv::imread(imgpath_rp,0);
		img_rgb = cv::imread(imgpath_lc,1);

		int img_rows = img_lc.rows;
		int img_cols = img_lc.cols;
	
		/**************** 3D reconstruction(260ms) ***************/
		QuadFeatureMatch* quadmatcher = new QuadFeatureMatch(img_lc,img_rc,img_lp,img_rp,true);
		quadmatcher->init(DET_GFTT,DES_SIFT);
		quadmatcher->detectFeature();
		quadmatcher->circularMatching();

		/**************** Threads for Semantic(180ms) **************/ 
		std::thread foo(semantic, std::ref(img_rgb), std::ref(classifier), img_cols, cropCols, img_rows, cropRows, cropCols2,
					std::ref(segnet), std::ref(colormap), std::ref(predictionsL), std::ref(predictionsR));

		//visual odometry valid
		if(viso.Process(*quadmatcher)==true)
		{
			//get ego-motion matrix (6DOF)
			cv::Mat motion;
			motion = viso.getMotion();

			//computing disparity image (SGBM or BM method) and 3D reconstruction by triangulation

			//calDisparity(img_lc, img_rc, disp_sgbm);
			//applyColorMap(disp_sgbm, disp_show_sgbm, COLORMAP_JET); 
			//cv::imshow("Disparity", disp_show_sgbm);
			//disp_sgbm.convertTo(disp_sgbm, CV_16SC1, 16.0f); 

			calDisparity_SGBM(img_lc, img_rc, disp_sgbm, disp_show_sgbm);
			triangulate10D(img_lc, disp_sgbm, xyz, f, c_u, c_v, base, roi_);

			//analyzing the stereo image by U-V disparity images (here I put moving object detection result)
			moving_mask = uv_disparity.Process(img_lc, disp_sgbm, viso, xyz, roi_mask, ground_mask, pitch1, pitch2);
			//cv::imshow("roi", roi_mask);

			//visual odometry
			Matrix_ M = Matrix_::eye(4);
			for (int32_t i=0; i<4; ++i)
				for (int32_t j=0; j<4; ++j)
					M.val[i][j] = motion.at<double>(i,j);
			poseChanged = Matrix_::inv(M);
			pose = pose * poseChanged;
			key_pose = key_pose * poseChanged;
			
			//gt
			cv::Mat gtpose;
			Matrix_ gt_ = Matrix_::eye(4); 
			pd.getData( n+1, gtpose );
			for (int32_t i=0; i<4; ++i)
				for (int32_t j=0; j<4; ++j)
					gt_.val[i][j] = gtpose.at<double>(i,j);
			//pose = gt_;
			gtpose_ = gt_;

			success = true;
		}
		delete quadmatcher;

		/**************** Semantic segmentation ***************/
		foo.join();
		//cv::imshow("disparity", disp_show_sgbm);
		if (!success) continue;

		/****************** Fusion ******************/
		//add new item
		if (vec.size() == fusionLen)
			vec.erase(vec.begin());
		Fusion item;
		disp_sgbm.copyTo(item.depth);
		moving_mask.copyTo(item.mov);
		roi_mask.copyTo(item.roi);
		xyz.copyTo(item.xyz);
		img_rgb.copyTo(item.rgb);
		item.preL = predictionsL;
		item.preR = predictionsR;
		item.rt = poseChanged;
		vec.push_back(item);	

		float rt_change = key_pose.l2norm();
		if (rt_change > RT_Threshold)
		{
			cv::Mat temp_moving_mask, temp_roi_mask, temp_xyz, temp_disp_sgbm;
			std::vector<Prediction> temp_preL, temp_preR;
			Matrix_ RT = Matrix_::eye(4);			
			vec[vec.size()-1].depth.copyTo(key_disp_sgbm);		
			vec[vec.size()-1].mov.copyTo(key_moving_mask);		
			vec[vec.size()-1].roi.copyTo(key_roi_mask);		
			vec[vec.size()-1].xyz.copyTo(key_xyz);		
			vec[vec.size()-1].rgb.copyTo(key_img_rgb);
			key_preL = vec[vec.size()-1].preL;
			key_preR = vec[vec.size()-1].preR;
			//fusion loop
			for (size_t q = 0; q < vec.size()-1; q++)
			{		
				//fusion per frame	
				for (size_t k = q+1; k <= vec.size()-1; k++)
					RT = RT * vec[k].rt;
				RT = Matrix_::inv(RT);
				//std::cout << "RT: " << RT << std::endl; 
				vec[q].depth.copyTo(temp_disp_sgbm);		
				vec[q].mov.copyTo(temp_moving_mask);		
				vec[q].roi.copyTo(temp_roi_mask);		
				vec[q].xyz.copyTo(temp_xyz);	
				temp_preL = vec[q].preL;
				temp_preR = vec[q].preR;

				for (int i = 0; i < img_rows; i++)
				{
					//other frame
					uchar* temp_moving_ptr = temp_moving_mask.ptr<uchar>(i);
					const float* temp_recons_ptr = temp_xyz.ptr<float>(i);
					const short* temp_depth_ptr = temp_disp_sgbm.ptr<short>(i);
					const uchar* temp_roi_ptr = temp_roi_mask.ptr<uchar>(i);
					for (int j = 0; j < img_cols; j++)
					{
						//other frame validation
		    			const short d = temp_depth_ptr[j];
						const uchar r = temp_roi_ptr[j];
						if (fabs(d)<=FLT_EPSILON || r==0) continue;

						//projection
						const float x = temp_recons_ptr[10*j]*RT.val[0][0] + temp_recons_ptr[10*j+1]*RT.val[0][1] + temp_recons_ptr[10*j+2]*RT.val[0][2] + RT.val[0][3];
						const float y = temp_recons_ptr[10*j]*RT.val[1][0] + temp_recons_ptr[10*j+1]*RT.val[1][1] + temp_recons_ptr[10*j+2]*RT.val[1][2] + RT.val[1][3];
						const float z = temp_recons_ptr[10*j]*RT.val[2][0] + temp_recons_ptr[10*j+1]*RT.val[2][1] + temp_recons_ptr[10*j+2]*RT.val[2][2] + RT.val[2][3];
						const int u = int(x*f/z + c_u);
						const int v = int(y*f/z + c_v);
					
						//boundary validation
						if (u<0 || u>=img_cols || v<0 || v>=img_rows) continue;

						//key frame
						uchar* key_moving_ptr = key_moving_mask.ptr<uchar>(v);
						float* key_recons_ptr = key_xyz.ptr<float>(v);
						const short* key_depth_ptr = key_disp_sgbm.ptr<short>(v);
						const uchar* key_roi_ptr = key_roi_mask.ptr<uchar>(v);

						//key frame validation
						const short key_d = key_depth_ptr[u];
						const uchar key_r = key_roi_ptr[u];
						if (fabs(key_d)<=FLT_EPSILON || key_r==0) continue;

						//depth fusion
						key_recons_ptr[10*u] = (key_recons_ptr[10*u]+x) * 0.5;
						key_recons_ptr[10*u+1] = (key_recons_ptr[10*u+1]+y) * 0.5;
						key_recons_ptr[10*u+2] = (key_recons_ptr[10*u+2]+z) * 0.5;

					}
				}
			}
			keyFrameT ++;
			key_pose = Matrix_::eye(4);
		}
		else 
		{
			continue;
		}

		/**************** Integration ***************/
		//pcl
		float px, py, pz;
		uchar pr, pg, pb;
		uint32_t rgb;
		pcl::PointXYZRGB pointRGB;
		pcl::PointXYZRGBL pointRGBL;
		CloudT::Ptr cloud(new CloudT);
		CloudLT::Ptr cloud_anno(new CloudLT); 

		double minDisparity = FLT_MAX;
		cv::minMaxIdx(disp_sgbm, &minDisparity, 0, 0, 0 );
		for (int v = 0; v < img_rows; ++v)
		{
			const uchar* moving_ptr = moving_mask.ptr<uchar>(v);
			const uchar* roi_ptr = roi_mask.ptr<uchar>(v);
			const short* disparity_ptr = disp_sgbm.ptr<short>(v);
			const float* recons_ptr = xyz.ptr<float>(v);
			const uchar* rgb_ptr = img_rgb.ptr<uchar>(v);

			for (int u = 0; u < img_cols; ++u)
			{
				if (v>=img_rows-cropRows && v<img_rows && u>=img_cols/2-cropCols && u<img_cols/2+cropCols)
				{
    				short d = disparity_ptr[u];
    				//if (fabs(d)>FLT_EPSILON && roi_ptr[u]!=0 && moving_ptr[u]!=255) //remove moving objects and outside the ROI
    				if (fabs(d)>FLT_EPSILON && roi_ptr[u]!=0) //remove moving objects and outside the ROI
					{
						//3d points
						//px = recons_ptr[10*u] * pose.val[0][0] + recons_ptr[10*u+1] * pose.val[0][1] + recons_ptr[10*u+2] * pose.val[0][2] + pose.val[0][3];
						//py = recons_ptr[10*u] * pose.val[1][0] + recons_ptr[10*u+1] * pose.val[1][1] + recons_ptr[10*u+2] * pose.val[1][2] + pose.val[1][3];
						//pz = recons_ptr[10*u] * pose.val[2][0] + recons_ptr[10*u+1] * pose.val[2][1] + recons_ptr[10*u+2] * pose.val[2][2] + pose.val[2][3];

						double pw = base/(1.0*static_cast<double>(d));
						px = ((static_cast<double>(u)-c_u)*pw)*16.0f;
						py = ((static_cast<double>(v)-c_v)*pw)*16.0f;
						pz = (f*pw)*16.0f;

	            		if (fabs(d-minDisparity) <= FLT_EPSILON)
	            			continue;
	            		if (fabs(px)>roix || fabs(py)>roiy || fabs(pz)>roiz)
	            			continue;

						pb = rgb_ptr[u*3]; 
						pg = rgb_ptr[u*3+1]; 
						pr = rgb_ptr[u*3+2]; 
						rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
						pointRGB.x = px;
						pointRGB.y = py;
						pointRGB.z = pz;
						pointRGB.rgb = *reinterpret_cast<float*>(&rgb);
				
						//semantic points
						int i = v-(img_rows-cropRows);
						int j = u-(img_cols/2-cropCols);
						const uchar* segnet_ptr = segnet.ptr<uchar>(i);
						pb = segnet_ptr[j*3]; 
						pg = segnet_ptr[j*3+1]; 
						pr = segnet_ptr[j*3+2];
						rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
						pointRGBL.x = px;
						pointRGBL.y = py;
						pointRGBL.z = pz;
						pointRGBL.rgb = *reinterpret_cast<float*>(&rgb);

						if (u < img_cols/2)
							pointRGBL.label = predictionsL[i*cropCols+j].second; 
						else 
							pointRGBL.label = predictionsR[i*cropCols+j].second;

						if (pb == pg && pg == pr) continue;
						cloud->points.push_back(pointRGB);
						cloud_anno->points.push_back(pointRGBL);

					}
				}
			}
		} 

		/************** 3d CRF mapInference(720ms) *************/
		
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
		pclut(cloud_anno, tmpcloud);
		//rgbviewer.showCloud(tmpcloud, "rgbviewer");
		//while( !rgbviewer.wasStopped() ) {}
		
		
		float normal_radius_search = static_cast<float>(default_normal_radius_search);
		float leaf_x = default_leaf_size, leaf_y = default_leaf_size, leaf_z = default_leaf_size;
		CloudLT::Ptr crfCloud(new CloudLT); compute(cloud, cloud_anno, normal_radius_search, leaf_x, leaf_y, leaf_z, crfCloud);
		*cloud_anno = *crfCloud;
		
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpcloud_(new pcl::PointCloud<pcl::PointXYZRGB>);
		pclut(crfCloud, tmpcloud_);
		//rgbviewer.showCloud(tmpcloud_, "crfviewer");
		//while( !rgbviewer.wasStopped() ) {}


		/********************* refine segnet ********************/
		cv::Mat img(cropRows,cropCols2,CV_8UC3,cv::Scalar(0,0,0));
		for (size_t j = 0; j < cloud_anno->points.size(); j++)
		{
			//double x = (-cloud_anno->points[j].x) * poseinv.val[0][0] + (-cloud_anno->points[j].y) * poseinv.val[0][1] + cloud_anno->points[j].z * poseinv.val[0][2] + poseinv.val[0][3];
			//double y = (-cloud_anno->points[j].x) * poseinv.val[1][0] + (-cloud_anno->points[j].y) * poseinv.val[1][1] + cloud_anno->points[j].z * poseinv.val[1][2] + poseinv.val[1][3];
			//double z = (-cloud_anno->points[j].x) * poseinv.val[2][0] + (-cloud_anno->points[j].y) * poseinv.val[2][1] + cloud_anno->points[j].z * poseinv.val[2][2] + poseinv.val[2][3];
			double x = cloud_anno->points[j].x + 0.06;
			double y = cloud_anno->points[j].y;
			double z = cloud_anno->points[j].z;

			int u = (x * (f / z) + c_u);
			int v = (y * (f / z) + c_v);

			u -= (img_cols/2 - cropCols);
			v -= (img_rows - cropRows);

			if (u>=0 && u<img.cols && v>=0 && v<img.rows)
			{
				uchar* imgptr = img.ptr<uchar>(v);
				imgptr[u*3] = cloud_anno->points[j].label;
				imgptr[u*3+1] = cloud_anno->points[j].label;
				imgptr[u*3+2] = cloud_anno->points[j].label;
			}
		}
		//std::cout << BOLDYELLOW"Offset: " << img_rows - cropRows << RESET"" << std::endl;
		cv::Mat img_raw = img.clone();
		cv::LUT(img, colormap, img);

		for (int i=0;i<img.rows;i++)
		{
			for (int j=0;j<img.cols;j++)
			{
				uchar* imgptr = img.ptr<uchar>(i);
				if (imgptr[j*3]==imgptr[j*3+1]&& imgptr[j*3+1]==imgptr[j*3+2])
				{	
					imgptr[j*3] = 0;
					imgptr[j*3+1] = 0;
					imgptr[j*3+2] = 0;
				}
			}
		}


		//save pictures
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_(new pcl::PointCloud<pcl::PointXYZRGB>);
		for (size_t j = 0; j < cloud_anno->points.size(); j++)
		{
			Matrix_ poseinv = Matrix_::inv(pose);
			double x = cloud_anno->points[j].x;
			double y = cloud_anno->points[j].x;
			double z = cloud_anno->points[j].x;

			if ( (cloud_anno->points[j].b==128 && cloud_anno->points[j].g==0 && cloud_anno->points[j].r==64) ||
				 (cloud_anno->points[j].b==0 && cloud_anno->points[j].g==64 && cloud_anno->points[j].r==64) ||
				 (cloud_anno->points[j].b==192 && cloud_anno->points[j].g==128 && cloud_anno->points[j].r==0) )
				continue;	
			else
			{
				pcl::PointXYZRGB pointRGB;
				pointRGB.x = x;
				pointRGB.y = y;
				pointRGB.z = z;
				pointRGB.r = cloud_anno->points[j].r;
				pointRGB.g = cloud_anno->points[j].g;
				pointRGB.b = cloud_anno->points[j].b;
				cloud_->points.push_back(pointRGB);
			}

		}

		cv::imshow("img",img);
		cv::imshow("segnet", segnet);
		cv::waitKey(1);

		char pcdname[256];
		cloud->width = (int) cloud->points.size();
		cloud->height = 1;
		cloud_->width = (int) cloud_->points.size();
		cloud_->height = 1;
		tmpcloud->width = (int) tmpcloud->points.size();
		tmpcloud->height = 1;
		sprintf(pcdname, "png/3dReconstrcution%d.pcd", n+1);
		pcl::io::savePCDFileASCII (pcdname, *cloud);
		sprintf(pcdname, "png/3dBeforeSemantic%d.pcd", n+1);
		pcl::io::savePCDFileASCII (pcdname, *tmpcloud);
		sprintf(pcdname, "png/3dSemanrtic%d.pcd", n+1);
		pcl::io::savePCDFileASCII (pcdname, *cloud_anno);
		sprintf(pcdname, "png/3dSemanrticRemoveMoving%d.pcd", n+1);
		pcl::io::savePCDFileASCII (pcdname, *cloud_);

		char savepic[256];
		sprintf(savepic, "png/rgb%d.bmp", n+1);
		cv::imwrite(savepic, img_rgb);
		sprintf(savepic, "png/disparity%d.bmp", n+1);
		cv::imwrite(savepic, disp_show_sgbm);
		sprintf(savepic, "png/segnet%d.bmp", n+1);
		cv::imwrite(savepic, segnet);
		sprintf(savepic, "png/refineSegnet%d.bmp", n+1);
		cv::imwrite(savepic, img);
		sprintf(savepic, "png/key_moving%d.bmp", n+1);
		cv::imwrite(savepic, key_moving_mask);

		//std::cout << BOLDYELLOW"Pointcloud: " << pointCloudNum << RESET"" << std::endl;
	}
 
	//time elapse
	gettimeofday(&t_end, NULL);
	seconds  = t_end.tv_sec  - t_start.tv_sec;
	useconds = t_end.tv_usec - t_start.tv_usec;
	duration = seconds*1000.0 + useconds/1000.0;

	while( !rgbviewer.wasStopped() )
    {
        
    }

	return 0;
}



/*
int main(int argc, char** argv) 
{
	// Load network
	string model_file = "/home/inin/SegNet/caffe-segnet/examples/segnet/models/segnet_model_driving_webdemo.prototxt";
	string trained_file = "/home/inin/SegNet/caffe-segnet/examples/segnet/models/segnet_weights_driving_webdemo.caffemodel";
	string label_file = "/home/inin/SegNet/caffe-segnet/examples/segnet/semantic12.txt";
	string colorfile = "/home/inin/SegNet/caffe-segnet/examples/segnet/color.png";
	Classifier classifier;

	// Load image
	cv::Mat img = cv::imread("/home/inin/L_755.png", 1);
	cv::Mat color = cv::imread(colorfile, 1);
	cv::Mat resizeImg;
	cv::resize(img, resizeImg, Size(480,360));

	// Prediction
	std::vector<Prediction> predictions = classifier.Classify(resizeImg);


	// Display segnet
	cv::Mat segnet(resizeImg.size(), CV_8UC3, Scalar(0,0,0));
	for (int i = 0; i < resizeImg.rows; ++i)
	{	
		uchar* segnet_ptr = segnet.ptr<uchar>(i);
		for (int j = 0; j < resizeImg.cols; ++j)
		{
			segnet_ptr[j*3+0] = predictions[i*resizeImg.cols+j].second;
			segnet_ptr[j*3+1] = predictions[i*resizeImg.cols+j].second;
			segnet_ptr[j*3+2] = predictions[i*resizeImg.cols+j].second;
		}
	}
	cv::LUT(segnet, color, segnet);

	cv::resize(segnet, segnet, Size(img.cols,img.rows));
	imshow("img",segnet);
	imwrite("segnet.png", segnet);
	waitKey(0);

	return 0;
}
*/