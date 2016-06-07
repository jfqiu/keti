#include "utils.h"

#include <pcl/io/png_io.h>
//stereo params
double f     = 707.0912; //focal length in pixels
double c_u   = 601.8873;  //principal point (u-coordinate) in pixels
double c_v   = 183.1104;  //principal point (v-coordinate) in pixels
double base  = 0.537904488; //baseline in meters
double inlier_threshold = 6.0f; //RANSAC parameter for classifying inlier and outlier

//roi space
int roix = 15;
int roiy = 20; //0006-10
int roiz = 30;

//colormap
string colorfile = "/home/inin/SegNet/caffe-segnet/examples/segnet/color.png";

//size
int cropRows = 360;
int cropCols = 480;
int cropCols2 = 960;

//data directory
string rgb_dirL = "/media/inin/data/sequences/07/image_2/";
string rgb_dirR = "/media/inin/data/sequences/07/image_3/";
string velodyne_dir = "/media/inin/data/dataset/sequences/05/velodyne/";

//crf param
float  default_leaf_size = 0.02; //0.05f 0.03f is also valid
double default_feature_threshold = 5.0; //5.0
double default_normal_radius_search = 0.5; //0.03


int main() 
{
	//set visual odometry parameters
	VisualOdometryStereo::parameters param; 
	setVO(param, f, c_u, c_v, base, inlier_threshold);
	VisualOdometryStereo viso(param);
	ROI3D roi_(roix,roiy,roiz); //set the ROI region for the field of view
	Classifier classifier; //load network
	cv::Mat colormap = cv::imread(colorfile, 1); //look-up table
	//int count = numFiles(rgb_dirL); //frame list

	//initialize the parameters of UVDisparity
	CalibPars calib_(f,c_u,c_v,base);
	UVDisparity uv_disparity;
	uv_disparity.SetCalibPars(calib_); uv_disparity.SetROI3D(roi_);
	uv_disparity.SetOutThreshold(6.0f); uv_disparity.SetInlierTolerance(3);
	uv_disparity.SetMinAdjustIntense(20);

	//pcl
    pcl::visualization::CloudViewer rgbviewer( "rgbviewer" );

	/*********************** loop **********************/
	//time elapse
	struct timeval t_start, t_end;
	long seconds, useconds;
	double duration;
	gettimeofday(&t_start, NULL);    

    DIR *dp;
    struct dirent *dirp;
    char dirname[] = "/home/inin/SegNet/caffe-segnet/examples/segnet/data/Validation/GT/";
    vector<string> gt_file_names;
    if ((dp = opendir(dirname))==NULL) 
    {
        perror("opendir error");
        exit(1);
    }
    while ((dirp=readdir(dp))!=NULL)
    {
        if((strcmp(dirp->d_name,".")==0)||(strcmp(dirp->d_name,"..")==0))
            continue;
         gt_file_names.push_back(dirp->d_name);
	}

	for (int number = 0; number < gt_file_names.size(); ++number)
	{
		int n = (gt_file_names[number][2]-'0') * 1000 + (gt_file_names[number][3]-'0') * 100 + (gt_file_names[number][4]-'0') * 10 + (gt_file_names[number][5]-'0');

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
		char base_name_p[256], base_name_c[256], base_name_s[256];
		sprintf(base_name_p, "%06d.png", n-1);
		sprintf(base_name_c, "%06d.png", n);
		sprintf(base_name_s, "%06d.png", n);

		string imgpath_lp = rgb_dirL + base_name_p; string imgpath_rp = rgb_dirR + base_name_p;
		string imgpath_lc = rgb_dirL + base_name_c; string imgpath_rc = rgb_dirR + base_name_c;
		string semantic_path = string(dirname) + base_name_s;
		img_lc = cv::imread(imgpath_lc,0); img_lp = cv::imread(imgpath_lp,0);
		img_rc = cv::imread(imgpath_rc,0); img_rp = cv::imread(imgpath_rp,0);
		img_rgb = cv::imread(imgpath_lc,1);
		img_semantic = cv::imread(semantic_path, 1);
		cout << semantic_path << endl;
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

			calDisparity_SGBM(img_lc, img_rc, disp_sgbm, disp_show_sgbm);
			triangulate10D(img_lc, disp_sgbm, xyz, f, c_u, c_v, base, roi_);

			//analyzing the stereo image by U-V disparity images (here I put moving object detection result)
			moving_mask = uv_disparity.Process(img_lc, disp_sgbm, viso, xyz, roi_mask, ground_mask, pitch1, pitch2);

			success = true;
		}
		delete quadmatcher;

		/**************** Semantic segmentation ***************/
		foo.join();
		//cv::imshow("disparity", disp_show_sgbm);
		if (!success) continue;

		/**************** Integration ***************/
		//pcl
		float px, py, pz;
		uchar pr, pg, pb;
		uint32_t rgb;
		pcl::PointXYZRGB pointRGB;
		pcl::PointXYZRGBL pointRGBL;
		CloudT::Ptr cloud(new CloudT);
		CloudLT::Ptr cloud_anno(new CloudLT);   
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
						px = recons_ptr[10*u];
						py = recons_ptr[10*u+1];
						pz = recons_ptr[10*u+2];

						pb = rgb_ptr[u*3]; 
						pg = rgb_ptr[u*3+1]; 
						pr = rgb_ptr[u*3+2]; 
						rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
						pointRGB.x = -px;
						pointRGB.y = -py;
						pointRGB.z =  pz;
						pointRGB.rgb = *reinterpret_cast<float*>(&rgb);
						

						//semantic points
						int i = v-(img_rows-cropRows);
						int j = u-(img_cols/2-cropCols);
						const uchar* segnet_ptr = segnet.ptr<uchar>(i);
						pb = segnet_ptr[j*3]; 
						pg = segnet_ptr[j*3+1]; 
						pr = segnet_ptr[j*3+2];
						rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
						pointRGBL.x = -px;
						pointRGBL.y = -py;
						pointRGBL.z =  pz;
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
		rgbviewer.showCloud(tmpcloud, "rgbviewer");
		//while( !rgbviewer.wasStopped() ) {}
		
		
		float normal_radius_search = static_cast<float>(default_normal_radius_search);
		float leaf_x = default_leaf_size, leaf_y = default_leaf_size, leaf_z = default_leaf_size;
		CloudLT::Ptr crfCloud(new CloudLT); compute(cloud, cloud_anno, normal_radius_search, leaf_x, leaf_y, leaf_z, crfCloud);
		*cloud_anno = *crfCloud;
		
		//pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpcloud_(new pcl::PointCloud<pcl::PointXYZRGB>);
		//pclut(crfCloud, tmpcloud_);
		//rgbviewer.showCloud(tmpcloud_, "crfviewer");
		//while( !rgbviewer.wasStopped() ) {}

		/********************* refine segnet ********************/
		cv::Mat img(360,960,CV_8UC3,cv::Scalar(0,0,0));
		for (size_t j = 0; j < cloud_anno->points.size(); j++)
		{
			//double x = (-cloud_anno->points[j].x) * poseinv.val[0][0] + (-cloud_anno->points[j].y) * poseinv.val[0][1] + cloud_anno->points[j].z * poseinv.val[0][2] + poseinv.val[0][3];
			//double y = (-cloud_anno->points[j].x) * poseinv.val[1][0] + (-cloud_anno->points[j].y) * poseinv.val[1][1] + cloud_anno->points[j].z * poseinv.val[1][2] + poseinv.val[1][3];
			//double z = (-cloud_anno->points[j].x) * poseinv.val[2][0] + (-cloud_anno->points[j].y) * poseinv.val[2][1] + cloud_anno->points[j].z * poseinv.val[2][2] + poseinv.val[2][3];
			double x = -cloud_anno->points[j].x;
			double y = -cloud_anno->points[j].y;
			double z =  cloud_anno->points[j].z;

			int u = (x * f / z + c_u);
			int v = (y * f / z + c_v);

			u -= (img_cols/2 - cropCols);
			v -= (img_rows - cropRows);
			//v -= 100;

			if (u>=0 && u<960 && v>=0 && v<360)
			{
				uchar* imgptr = img.ptr<uchar>(v);
				imgptr[u*3] = cloud_anno->points[j].label;
				imgptr[u*3+1] = cloud_anno->points[j].label;
				imgptr[u*3+2] = cloud_anno->points[j].label;
			}
		}
		//std::cout << BOLDYELLOW"Offset: " << img_rows - cropRows << RESET"" << std::endl;
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
		cv::imshow("img",img);
		cv::imshow("segnet", segnet);

		// Semantic Evaluation
		cv::Rect crop(img_cols/2-cropCols,img_rows-cropRows,cropCols*2,cropRows);
		cv::Mat crop_gt;
		img_semantic(crop).copyTo(crop_gt);
		cv::resize(crop_gt, crop_gt, Size(960,360));

		cv::imshow("gt", crop_gt);
		cv::waitKey(0);

		double pave_s=0, building_s=0, vege_s=0, car_s=0, road_s=0, fence_s=0, pole_s=0; // intersection
		double pave_r=0, building_r=0, vege_r=0, car_r=0, road_r=0, fence_r=0, pole_r=0;

		double pave_ss=0, building_ss=0, vege_ss=0, car_ss=0, road_ss=0, fence_ss=0, pole_ss=0; //test
		double pave_rr=0, building_rr=0, vege_rr=0, car_rr=0, road_rr=0, fence_rr=0, pole_rr=0;

		double pave=0, building=0, vege=0, car=0, road=0, fence=0, pole=0; //gt
		for (int i=0; i<crop_gt.rows; i++)
		{
			uchar* imgptr = img.ptr<uchar>(i);
			uchar* segptr = segnet.ptr<uchar>(i);
			uchar* gtptr = crop_gt.ptr<uchar>(i);

			for (int j=0; j<crop_gt.cols; j++)
			{
				uchar pb_gt = gtptr[j*3], pg_gt = gtptr[j*3+1], pr_gt = gtptr[j*3+2];
				uchar pb_refine = imgptr[j*3], pg_refine = imgptr[j*3+1], pr_refine = imgptr[j*3+2];
				uchar pb_segnet = segptr[j*3], pg_segnet = segptr[j*3+1], pr_segnet = segptr[j*3+2];

				if (pb_gt==0 && pg_gt==0 && pr_gt==128) 
				{
					if (pb_segnet==0 && pg_segnet==0 && pr_segnet==128) building_s+=1;
					if (pb_refine==0 && pg_refine==0 && pr_refine==128) building_r+=1;
					building+=1;
				}
				if (pb_gt==192 && pg_gt==0 && pr_gt==0) 
				{
					if (pb_segnet==222 && pg_segnet==40 && pr_segnet==60) pave_s+=1;
					if (pb_refine==222 && pg_refine==40 && pr_refine==60) pave_r+=1;
					pave+=1;
				}
				if (pb_gt==0 && pg_gt==128 && pr_gt==128) 
				{
					if (pb_segnet==0 && pg_segnet==128 && pr_segnet==128) vege_s+=1;
					if (pb_refine==0 && pg_refine==128 && pr_refine==128) vege_r+=1;
					vege+=1;
				}
				if (pb_gt==128 && pg_gt==0 && pr_gt==64) 
				{
					if (pb_segnet==128 && pg_segnet==0 && pr_segnet==64) car_s+=1;
					if (pb_refine==128 && pg_refine==0 && pr_refine==64) car_r+=1;
					car+=1;
				}
				if (pb_gt==128 && pg_gt==64 && pr_gt==128) 
				{
					if (pb_segnet==128 && pg_segnet==64 && pr_segnet==128) road_s+=1;
					if (pb_refine==128 && pg_refine==64 && pr_refine==128) road_r+=1;
					road+=1;
				}
				if (pb_gt==128 && pg_gt==192 && pr_gt==192) 
				{
					if (pb_segnet==128 && pg_segnet==192 && pr_segnet==192) pole_s+=1;
					if (pb_refine==128 && pg_refine==192 && pr_refine==192) pole_r+=1;
					pole+=1;
				}
				if (pb_gt==128 && pg_gt==64 && pr_gt==64) 
				{
					if (pb_segnet==128 && pg_segnet==64 && pr_segnet==64) fence_s+=1;
					if (pb_refine==128 && pg_refine==64 && pr_refine==64) fence_r+=1;
					fence+=1;
				}

				// another
				if (pb_segnet==0 && pg_segnet==0 && pr_segnet==128) building_ss+=1;
				if (pb_refine==0 && pg_refine==0 && pr_refine==128) building_rr+=1;

				if (pb_segnet==222 && pg_segnet==40 && pr_segnet==60) pave_ss+=1;
				if (pb_refine==222 && pg_refine==40 && pr_refine==60) pave_rr+=1;

				if (pb_segnet==0 && pg_segnet==128 && pr_segnet==128) vege_ss+=1;
				if (pb_refine==0 && pg_refine==128 && pr_refine==128) vege_rr+=1;

				if (pb_segnet==128 && pg_segnet==0 && pr_segnet==64) car_ss+=1;
				if (pb_refine==128 && pg_refine==0 && pr_refine==64) car_rr+=1;

				if (pb_segnet==128 && pg_segnet==64 && pr_segnet==128) road_ss+=1;
				if (pb_refine==128 && pg_refine==64 && pr_refine==128) road_rr+=1;

				if (pb_segnet==128 && pg_segnet==64 && pr_segnet==64) fence_ss+=1;
				if (pb_refine==128 && pg_refine==64 && pr_refine==64) fence_rr+=1;

				if (pb_segnet==128 && pg_segnet==192 && pr_segnet==192) pole_ss+=1;
				if (pb_refine==128 && pg_refine==192 && pr_refine==192) pole_rr+=1;
			}
		}
		cout << fence_ss << "," << fence << "," << fence_s << "," << fence_r << "," << fence_rr << endl;
		double acc_pave_s = pave_s/(pave+pave_ss-pave_s);
		double acc_building_s = building_s/(building+building_ss-building_s);
		double acc_vege_s = vege_s/(vege+vege_ss-vege_s);
		double acc_car_s = car_s/(car+car_ss-car_s);
		double acc_road_s = road_s/(road+road_ss-road_s);
		double acc_fence_s = fence_s/(fence+fence_ss-fence_s);
		double acc_pole_s = pole_s/(pole+pole_ss-pole_s);

		double acc_pave_r = pave_r/(pave+pave_r-pave_r);
		double acc_building_r = building_r/(building+building_rr-building_r);
		double acc_vege_r = vege_r/(vege+vege_rr-vege_r);
		double acc_car_r = car_r/(car+car_rr-car_r);
		double acc_road_r = road_r/(road+road_rr-road_r);
		double acc_fence_r = fence_r/(fence+fence_rr-fence_r);
		double acc_pole_r = pole_r/(pole+pole_rr-pole_r);
		std::cout << BOLDYELLOW"Average Accuracy: " << std::endl << 
			"pave: " << acc_pave_s << "," << acc_pave_r << std::endl <<
			"building: " << acc_building_s << "," << acc_building_r << std::endl <<
			"vege: " << acc_vege_s << "," << acc_vege_r << std::endl <<
			"car: " << acc_car_s << "," << acc_car_r << std::endl <<
			"road: " << acc_road_s << "," << acc_road_r << std::endl <<
			"fence: " << acc_fence_s << "," << acc_fence_r << std::endl << 
			"pole: " << acc_pole_s << "," << acc_pole_r << std::endl <<
		RESET"" << std::endl;	

		double sum = 0.0, sum_s = 0.0, sum_r;
		sum = pave + building + vege + car + road + fence + pole;
		sum_s = pave_s + building_s + vege_s + car_s + road_s + fence_s + pole_s;
		sum_r = pave_r + building_r + vege_r + car_r + road_r + fence_r + pole_r;

		double sum_acc_s = acc_pave_s + acc_building_s + acc_vege_s + acc_car_s + acc_road_s + acc_fence_s + acc_pole_s;
		double sum_acc_r = acc_pave_r + acc_building_r + acc_vege_r + acc_car_r + acc_road_r + acc_fence_r + acc_pole_r;
		double average_s = sum_acc_s / 7.0;
		double average_r = sum_acc_r / 7.0;

		std::cout << BOLDYELLOW"Average Accuracy: " << average_s << "," << average_r << std::endl << RESET"" << std::endl;
		std::cout << BOLDYELLOW"Global Accuracy: " << sum_s/sum << "," << sum_r/sum << std::endl << RESET"" << std::endl;

/*
		// semantic cues remove moving error
		cv::imshow("key_moving_mask_before", key_moving_mask);
		key_moving_mask.copyTo(key_moving_mask_before);
		int cues = 0;
		for (int v = 0; v < img_rows; v++)
		{
			uchar* key_moving_ptr = key_moving_mask.ptr<uchar>(v);
			for (int u = 0; u < img_cols; u++)
			{
				if (v>=img_rows-cropRows && v<img_rows && u>=img_cols/2-cropCols && u<img_cols/2+cropCols)
				{
					int i = v-(img_rows-cropRows);
					int j = u-(img_cols/2-cropCols);
					//get semantic cues
					uchar* imgptr = img.ptr<uchar>(i);
					cues = imgptr[3*j];
					//validation
					if (key_moving_ptr[u]==255 && cues<9) key_moving_ptr[u] = 0;
				}
				else key_moving_ptr[u] = 0;
			}
		}
		cv::imshow("key_moving_mask_refine", key_moving_mask);
		// refine semantic
		cv::LUT(img, colormap, img);
		for (int i=0;i<360;i++)
		{
			for (int j=0;j<960;j++)
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
		cv::imshow("img",img);
		//cv::waitKey(0);

		// refine_semantic_moving
		cv::Mat img_moving;
		img.copyTo(img_moving);
		for (int v = 0; v < img_rows; v++)
		{
			uchar* key_moving_ptr = key_moving_mask.ptr<uchar>(v);
			for (int u = 0; u < img_cols; u++)
			{
				if (v>=img_rows-cropRows && v<img_rows && u>=img_cols/2-cropCols && u<img_cols/2+cropCols)
				{
					int i = v-(img_rows-cropRows);
					int j = u-(img_cols/2-cropCols);
					//get semantic cues
					uchar* img_moving_ptr = img_moving.ptr<uchar>(i);
					//validation
					//if (key_moving_ptr[u]==255) 
					if ( (img_moving_ptr[j*3]==128 && img_moving_ptr[j*3+1]==0 && img_moving_ptr[j*3+2]==64) ||
						 (img_moving_ptr[j*3]==0 && img_moving_ptr[j*3+1]==64 && img_moving_ptr[j*3+2]==64) ||
						 (img_moving_ptr[j*3]==192 && img_moving_ptr[j*3+1]==128 && img_moving_ptr[j*3+2]==0) )
					{
						img_moving_ptr[j*3] = 0;
						img_moving_ptr[j*3+1] = 0;
						img_moving_ptr[j*3+2] = 0;
					}
				}
			}
		}
		cv::imshow("img_moving",img_moving);


		//save pictures
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_(new pcl::PointCloud<pcl::PointXYZRGB>);
		for (size_t j = 0; j < cloud_anno->points.size(); j++)
		{
			Matrix_ poseinv = Matrix_::inv(pose);
			double x = -cloud_anno->points[j].x * poseinv.val[0][0] + -cloud_anno->points[j].y * poseinv.val[0][1] + cloud_anno->points[j].z * poseinv.val[0][2] + poseinv.val[0][3];
			double y = -cloud_anno->points[j].x * poseinv.val[1][0] + -cloud_anno->points[j].y * poseinv.val[1][1] + cloud_anno->points[j].z * poseinv.val[1][2] + poseinv.val[1][3];
			double z = -cloud_anno->points[j].x * poseinv.val[2][0] + -cloud_anno->points[j].y * poseinv.val[2][1] + cloud_anno->points[j].z * poseinv.val[2][2] + poseinv.val[2][3];

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

		char pcdname[256];
		cloud->width = (int) cloud->points.size();
		cloud->height = 1;
		cloud_->width = (int) cloud_->points.size();
		cloud_->height = 1;
		sprintf(pcdname, "png/3dReconstrcution%d.pcd", n);
		pcl::io::savePCDFileASCII (pcdname, *cloud);
		sprintf(pcdname, "png/3dSemanrtic%d.pcd", n);
		pcl::io::savePCDFileASCII (pcdname, *cloud_);

		char savepic[256];
		sprintf(savepic, "png/rgb%d.bmp", n);
		cv::imwrite(savepic, img_rgb);
		sprintf(savepic, "png/disparity%d.bmp", n);
		cv::imwrite(savepic, disp_show_sgbm);
		sprintf(savepic, "png/segnet%d.bmp", n);
		cv::imwrite(savepic, segnet);
		sprintf(savepic, "png/refine%d.bmp", n);
		cv::imwrite(savepic, img);
		sprintf(savepic, "png/refinemovingsemantic%d.bmp", n);
		cv::imwrite(savepic, img_moving);
		sprintf(savepic, "png/before_moving%d.bmp", n);
		cv::imwrite(savepic, key_moving_mask_before);
		sprintf(savepic, "png/refine_moving%d.bmp", n);
		cv::imwrite(savepic, key_moving_mask);
*/
		//std::cout << BOLDYELLOW"Pointcloud: " << pointCloudNum << RESET"" << std::endl;
	}
 
	//time elapse
	gettimeofday(&t_end, NULL);
	seconds  = t_end.tv_sec  - t_start.tv_sec;
	useconds = t_end.tv_usec - t_start.tv_usec;
	duration = seconds*1000.0 + useconds/1000.0;
	//std::cout << BOLDMAGENTA"Average time: " << duration/count << " ms" << RESET" " << std::endl;
	while( !rgbviewer.wasStopped() )
    {
        
    }

	return 0;
}

