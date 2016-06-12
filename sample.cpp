#include "utils.h"

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

//data directory
string rgb_dirL = "/media/inin/data/data_scene_flow/training/image_2/";
string rgb_dirR = "/media/inin/data/data_scene_flow/training/image_3/";
string gt_dir = "/media/inin/data/data_scene_flow/training/obj_map/";

//crf param
float  default_leaf_size = 0.02; //0.05f 0.03f is also valid
double default_feature_threshold = 5.0; //5.0
double default_normal_radius_search = 0.5; //0.03


/*************** 413.94 ms ***************/

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

	/*********************** loop **********************/
	//time elapse
	struct timeval t_start, t_end;
	long seconds, useconds;
	double duration;
	gettimeofday(&t_start, NULL);    

	double positive = 0;
	int num = 0;
	for (int n = 0; n < 200; n++)
	{
		//variables
		cv::Mat moving_mask, roi_mask, xyz, disp_sgbm, disp_show_sgbm, ground_mask;
		cv::Mat img_lc, img_lp, img_rc, img_rp, img_rgb, img_motion_gt;
		cv::Mat segnet(cropRows, cropCols2, CV_8UC3, cv::Scalar(0,0,0));
		double pitch1, pitch2;
		bool success = false;
		std::vector<Prediction> predictionsL;
		std::vector<Prediction> predictionsR;

		//read consecutive stereo images for 3d reconstruction
		char base_name_p[256], base_name_c[256], base_name_m[256];
		sprintf(base_name_p, "%06d_11.png", n);
		sprintf(base_name_c, "%06d_10.png", n);
		sprintf(base_name_m, "%06d_10.png", n);

		string imgpath_lp = rgb_dirL + base_name_p; string imgpath_rp = rgb_dirR + base_name_p;
		string imgpath_lc = rgb_dirL + base_name_c; string imgpath_rc = rgb_dirR + base_name_c;
		string motion_path = string(gt_dir) + base_name_m;
		img_lc = cv::imread(imgpath_lc,0); img_lp = cv::imread(imgpath_lp,0);
		img_rc = cv::imread(imgpath_rc,0); img_rp = cv::imread(imgpath_rp,0);
		img_rgb = cv::imread(imgpath_lc,1);
		img_motion_gt = cv::imread(motion_path, 0);

		int img_rows = img_lc.rows;
		int img_cols = img_lc.cols;
	
		cv::Mat img_rgb_crop;
		cv::Rect imgcrop(img_rgb.cols/2-480,img_rgb.rows-360,960,360);
		img_rgb(imgcrop).copyTo(img_rgb_crop);

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
		if (!success) continue;

		/**************** Integration ***************/
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
		float normal_radius_search = static_cast<float>(default_normal_radius_search);
		float leaf_x = default_leaf_size, leaf_y = default_leaf_size, leaf_z = default_leaf_size;
		//CloudLT::Ptr crfCloud(new CloudLT); compute(cloud, cloud_anno, normal_radius_search, leaf_x, leaf_y, leaf_z, crfCloud);
		//*cloud_anno = *crfCloud;

		/********************* refine segnet ********************/
		cv::Mat img(cropRows, cropCols2, CV_8UC3, cv::Scalar(0,0,0));
		for (size_t j = 0; j < cloud_anno->points.size(); j++)
		{ 
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

		// motion_gt_crop
		for (int i=0; i<img_motion_gt.rows; i++)
		{
			uchar* img_motion_gt_ptr = img_motion_gt.ptr<uchar>(i);
			for (int j=0; j<img_motion_gt.cols; j++)
			{
				if (img_motion_gt_ptr[j] > 0)
				{	
					img_motion_gt_ptr[j] = 255;
				}
			}
		}
		cv::Mat motion_gt_crop;
		cv::Mat motion_test_crop;
		cv::Rect crop(moving_mask.cols/2-480,moving_mask.rows-360,960,360);
		moving_mask(crop).copyTo(motion_test_crop);
		cv::Rect crop_motion(img_motion_gt.cols/2-480,img_motion_gt.rows-360,960,360);
		img_motion_gt(crop_motion).copyTo(motion_gt_crop);


		// get gray_motion_mask
		cv::Mat gray_semantic_mask;
		for (int i = 0; i < img.rows; i++)
		{
			uchar* img_ptr = img.ptr<uchar>(i);
			uchar* gray_semantic_mask_ptr = gray_semantic_mask.ptr<uchar>(i);
			for (int j = 0; j < img.cols; j++)
			{
				uchar pb = img_ptr[j*3];
				uchar pg = img_ptr[j*3+1];
				uchar pr = img_ptr[j*3+2];
				if ( (pb==128 && pg==0 && pr==64) ||
					 (pb==0 && pg==64 && pr==64) ||
					 (pb==192 && pg==128 && pr==0) )
				{
				}
				else
				{
					img_ptr[j*3] = 0;
					img_ptr[j*3+1] = 0;
					img_ptr[j*3+2] = 0;
				}
			}
		}
		cv::cvtColor(img, gray_semantic_mask, CV_BGR2GRAY); 
		cv::threshold(gray_semantic_mask, gray_semantic_mask, 0, 255, THRESH_BINARY);

		// get contours  
		std::vector<std::vector<cv::Point> > contours; 
		cv::findContours(gray_semantic_mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		cv::Mat semantic_motion_mask(img.size(), CV_8U, cv::Scalar(0)); 
		//cv::drawContours(semantic_motion_mask, contours, -1, cv::Scalar(255), 2); 

		int area_thres = 100;
		double overlay_portion_thres = 0.14;
		std::vector<cv::Mat> semantic_motion_result_masks;
		for (int i = 0; i < contours.size(); i++)
		{
			cv::Mat semantic_potential_mask = cv::Mat::zeros(img.size(), img.type());

			//cv::Mat tmp_mask(img.size(), CV_8U, cv::Scalar(0));
			//cv::drawContours(tmp_mask, contours, i, cv::Scalar(255,255,255), CV_FILLED, 8);
			//char all_mask_name[256];
			//sprintf(all_mask_name, "png/masks%d.bmp", i);
			//cv::imwrite(all_mask_name, tmp_mask);

			//std::cout << "area_" << i << ": " << cv::contourArea(contours[i]) << std::endl;
			if (cv::contourArea(contours[i]) > area_thres)
			{
				cv::drawContours(semantic_potential_mask, contours, i, cv::Scalar(255,255,255), CV_FILLED, 8);
				int overlay_count = 0;
				int mask_count = 1;
				for (int v = 0; v < img.rows; v++)
				{
					uchar* motion_test_crop_ptr = motion_test_crop.ptr<uchar>(v);
					uchar* semantic_potential_mask_ptr = semantic_potential_mask.ptr<uchar>(v);
					for (int u = 0; u < img.cols; u++)
					{
						if (semantic_potential_mask_ptr[u*3] == 255) 
						{
							mask_count ++;
							if (motion_test_crop_ptr[u] == 255)
							{
								overlay_count ++;
							}
						}
					}
				}
				double overlay_portion = overlay_count * 1.0f / mask_count;
				//std::cout << "area_portion_" << i << ": " << overlay_portion << "  area: " << cv::contourArea(contours[i]) << std::endl;
				if (overlay_portion > overlay_portion_thres)
				{
					semantic_motion_result_masks.push_back(semantic_potential_mask);
				}
			}
		}
		cv::Mat potential_semantic_motion_result_masks = cv::Mat::zeros(img.size(), img.type());
		if (semantic_motion_result_masks.size() > 0)	
		{
			//printf("There are %ld potential semantic masks.\n", semantic_motion_result_masks.size());
			for (int i = 0; i < semantic_motion_result_masks.size(); i++)
			{
				potential_semantic_motion_result_masks += semantic_motion_result_masks[i];
			}
		}
		else 
		{
			//printf("There are no potential semantic masks.\n");
			continue;
		}

		// Preparation color_semantic(semantic) -- potential_semantic_motion_result_masks(motion) -- img_rgb_crop(rgb)
		cv::Mat color_semantic(img.size(), CV_8UC3, cv::Scalar(0,0,0));
		for (int i = 0; i < img.rows; i++)
		{
			uchar* img_ptr = img.ptr<uchar>(i);
			uchar* potential_semantic_motion_result_masks_ptr = potential_semantic_motion_result_masks.ptr<uchar>(i);
			uchar* color_semantic_ptr = color_semantic.ptr<uchar>(i);
			for (int j = 0; j < img.cols; j++)
			{
				if (potential_semantic_motion_result_masks_ptr[j*3] == 255)
				{
					color_semantic_ptr[j*3] = img_ptr[j*3];
					color_semantic_ptr[j*3+1] = img_ptr[j*3+1];
					color_semantic_ptr[j*3+2] = img_ptr[j*3+2];
				}
			}
		}

		// 2d crf refine (use semantic image to refine motion image)


		// evaluation
		double tp = 0;
		double tn = 0;
		double fp = 0;
		double fn = 0;
		for (int i = 0; i < motion_gt_crop.rows; i++)
		{
			uchar* gt_ptr = motion_gt_crop.ptr<uchar>(i);
			uchar* result_ptr = potential_semantic_motion_result_masks.ptr<uchar>(i);
			for (int j = 0; j < motion_gt_crop.cols; j++)
			{
				if (gt_ptr[j] == 255)
				{
					if (result_ptr[j*3] == 255)
						tp ++;
					else
						fn ++;
				}
				else
				{
					if (result_ptr[j*3] == 0)
						tn ++;
					else
						fp ++;
				}
			}
		}
		double IOU = tp/(tp+fp+fn);
		if (IOU > 0)
		{
			positive += IOU;
			num ++;
		}
		else continue;
		printf("IOU: %f\n", IOU);

		cv::imshow("img_rgb_crop", img_rgb_crop);
		cv::imshow("segnet", segnet);
		cv::imshow("refine_segnet", img);
		cv::imshow("gray_semantic_mask", gray_semantic_mask);
		cv::imshow("motion_test_crop", motion_test_crop);
		cv::imshow("motion_gt_crop", motion_gt_crop);
		cv::imshow("color_semantic", color_semantic);
		cv::imshow("potential_semantic_motion_result_masks", potential_semantic_motion_result_masks);
		cv::waitKey(1);
/*
		char pcdname[256];
		sprintf(pcdname, "png/3dBeforeSemantic%d.pcd", n);
		tmpcloud->width = (int) tmpcloud->points.size();
		tmpcloud->height = 1;
		pcl::io::savePCDFileASCII (pcdname, *tmpcloud);

		char pcdname[256];
		cloud->width = (int) cloud->points.size();
		cloud->height = 1;
		cloud_->width = (int) cloud_->points.size();
		cloud_->height = 1;
		sprintf(pcdname, "png/3dReconstrcution%d.pcd", n);
		pcl::io::savePCDFileASCII (pcdname, *cloud);
		sprintf(pcdname, "png/3dBeforeSemantic%d.pcd", n);
		pcl::io::savePCDFileASCII (pcdname, *tmpcloud);
		sprintf(pcdname, "png/3dSemanrtic%d.pcd", n);
		pcl::io::savePCDFileASCII (pcdname, *cloud_anno);
		sprintf(pcdname, "png/3dSemanrticRemoveMoving%d.pcd", n);
		pcl::io::savePCDFileASCII (pcdname, *cloud_);

		char savepic[256];
		sprintf(savepic, "png/rgb%d.bmp", n);
		cv::imwrite(savepic, img_rgb);
		sprintf(savepic, "png/disparity%d.bmp", n);
		cv::imwrite(savepic, disp_show_sgbm);
		sprintf(savepic, "png/segnet%d.bmp", n);
		cv::imwrite(savepic, segnet);
		sprintf(savepic, "png/refineSegnet%d.bmp", n);
		cv::imwrite(savepic, img);
		sprintf(savepic, "png/key_moving%d.bmp", n);
		cv::imwrite(savepic, key_moving_mask);
		sprintf(savepic, "png/gt%d.bmp", n);
		cv::imwrite(savepic, crop_gt);
*/
		//std::cout << BOLDYELLOW"Pointcloud: " << pointCloudNum << RESET"" << std::endl;
	}
 
	//time elapse
	gettimeofday(&t_end, NULL);
	seconds  = t_end.tv_sec  - t_start.tv_sec;
	useconds = t_end.tv_usec - t_start.tv_usec;
	duration = seconds*1000.0 + useconds/1000.0;
	std::cout << BOLDMAGENTA"Average time: " << duration/200 << " ms  " << positive/num << RESET" " << std::endl;

	return 0;
}

