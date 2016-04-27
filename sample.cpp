#include "utils.h"

//stereo params
//double f     = 721.5377; //focal length in pixels
//double c_u   = 609.5593;  //principal point (u-coordinate) in pixels
//double c_v   = 172.8540;  //principal point (v-coordinate) in pixels
//double base  = 0.537150588; //baseline in meters
//double inlier_threshold = 6.0f; //RANSAC parameter for classifying inlier and outlier

double f     = 707.0912; //focal length in pixels
double c_u   = 601.8873;  //principal point (u-coordinate) in pixels
double c_v   = 183.1104;  //principal point (v-coordinate) in pixels
double base  = 0.537904488; //baseline in meters
double inlier_threshold = 6.0f; //RANSAC parameter for classifying inlier and outlier

//pcl params
float  default_leaf_size = 0.2f; //0.005f
double default_feature_threshold = 5.0; //5.0
double default_normal_radius_search = 0.03; //0.03

//roi space
int roix = 15;
int roiy = 5;
int roiz = 30;

//colormap
string colorfile = "/home/inin/SegNet/caffe-segnet/examples/segnet/color.png";

//data directory
//string rgb_dirL = "/media/inin/data/tracking dataset/testing/image_02/0006/"; //tt0006 tr0013 tt0011 tt0003  tr0009 173-213
//string rgb_dirR = "/media/inin/data/tracking dataset/testing/image_03/0006/";
string rgb_dirL = "/media/inin/data/sequences/05/image_2/";
string rgb_dirR = "/media/inin/data/sequences/05/image_3/";
//00(segnet-bad) 05(vo-bad-xiaodao) 07(confuse-xiaodao) 12(gaosu) 13(xiaodao) 16(kuandao) 18(segnet-bad) 19(segnet-confuse) 21(gaosu)

//size
int cropRows = 360;
int cropCols = 480;
int cropCols2 = 960;

//update voxel
const int gridScale = 1;
const int gridWidth = gridScale*500;
const int gridHeight = gridScale*5;

/*
 	            * (z)
	          *
	        *
          *
	     * * * * * * (x)
	     *
         *
         *
         (y)
*/

int main() 
{
	// ground pose
	ParameterReader pd("../data/poses/05.txt");
	//pd.getData( index, gtpose );

	//set visual odometry parameters
	VisualOdometryStereo::parameters param; 
	setVO(param, f, c_u, c_v, base, inlier_threshold);
	VisualOdometryStereo viso(param);
	ROI3D roi_(roix,roiy,roiz); //set the ROI region for the field of view
	Classifier classifier; //load network
	cv::Mat colormap = cv::imread(colorfile, 1); //look-up table
	Matrix_ pose = Matrix_::eye(4); //visual odometry
	//int count = numFiles(rgb_dirL); //frame list

	//initialize the parameters of UVDisparity
	CalibPars calib_(f,c_u,c_v,base);
	UVDisparity uv_disparity;
	uv_disparity.SetCalibPars(calib_); uv_disparity.SetROI3D(roi_);
	uv_disparity.SetOutThreshold(6.0f); uv_disparity.SetInlierTolerance(3);
	uv_disparity.SetMinAdjustIntense(20);

	//pcl
    pcl::visualization::CloudViewer voviewer( "voviewer" );
    //pcl::visualization::CloudViewer rgbviewer( "rgbviewer" );
    //pcl::visualization::CloudViewer crfviewer( "crfviewer" );
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr vooutput (new pcl::PointCloud<pcl::PointXYZRGB>);

	//fusion params
	cv::Mat key_moving_mask, key_roi_mask, key_xyz, key_disp_sgbm, key_img_rgb;
	std::vector<Prediction> key_preL;
	std::vector<Prediction> key_preR;
	std::vector<Fusion> vec;	
	Matrix_ key_pose = Matrix_::eye(4);
	size_t keyFrameT = 0;
	size_t fusionLen = 1; double translationT = 3; double rotationT = 5; double RT_Threshold = 5;//translation 10_m  rotation 5_degree

	//hash table
	size_t pointCloudNum = 0;
	hash_table* hash = new hash_table[gridWidth * gridWidth * gridHeight * 8];
	memset(hash, 0, sizeof(hash_table));

	/*********************** loop **********************/
	//time elapse
	struct timeval t_start, t_end;
	long seconds, useconds;
	double duration;
	gettimeofday(&t_start, NULL);    

	int count = 200;  
	for (int n = 0; n < count; ++n)
	{
		//variables
		cv::Mat moving_mask, roi_mask, xyz, disp_sgbm, disp_show_sgbm, ground_mask;
		cv::Mat img_lc,img_lp,img_rc,img_rp,img_rgb;
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
			/*
			//gt
			cv::Mat gtpose;
			Matrix_ gt_ = Matrix_::eye(4); 
			pd.getData( n+1, gtpose );
			for (int32_t i=0; i<4; ++i)
				for (int32_t j=0; j<4; ++j)
					gt_.val[i][j] = gtpose.at<double>(i,j);
			pose = gt_;
			*/
			success = true;
		}
		delete quadmatcher;

		/**************** Semantic segmentation ***************/
		foo.join();
		imshow("segnet", segnet);
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

		//build key frame
		double thetax = atan2(key_pose.val[2][1], key_pose.val[2][2]) * 180.0 / M_PI;	
		double thetay = atan2(-key_pose.val[2][0], sqrt(key_pose.val[2][1]*key_pose.val[2][1] + key_pose.val[2][2]*key_pose.val[2][2])) * 180.0 / M_PI;	
		double thetaz = atan2(key_pose.val[1][0], key_pose.val[0][0]) * 180.0 / M_PI;
		double translationx = key_pose.val[0][3];		
		double translationy = key_pose.val[1][3];		
		double translationz = key_pose.val[2][3];	
		double r_change = sqrt( (thetax*thetax) + (thetay*thetay) + (thetaz*thetaz) );
		double t_change = sqrt( (translationx*translationx) + (translationy*translationy) + (translationz*translationz) );
		float rt_change = key_pose.l2norm();
		//printf("size[%zu] [%f,%f,%f] [%f,%f,%f]\n", vec.size(),thetax,thetay,thetaz,translationx,translationy,translationz);
		//std::cout << BOLDWHITE"[r_key, t_key, rt_key]: " << BOLDGREEN" " << r_change << " " << t_change << BOLDYELLOW" " << rt_change << RESET" " << std::endl;
		//if (fabs(thetax)+fabs(thetay)+fabs(thetaz)>rotationT || fabs(translationx)+fabs(translationy)+fabs(translationz)>translationT)
		//if (r_change>rotationT || t_change>translationT)
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

						/*
						//moving fusion
						if (temp_moving_ptr[j] == 255) 
						{
							key_moving_ptr[u] = 255;
							//use semantic cues
							if (i>=img_rows-cropRows && i<img_rows && j>=img_cols/2-cropCols && j<img_cols/2+cropCols)
							{
								int label = 0;
								int r = i - (img_rows-cropRows);
								int c = j - (img_cols/2-cropCols);
								if (j < img_cols/2) label = temp_preL[r*cropCols+c].second;
								else label = temp_preR[r*cropCols+c].second;

								if (temp_moving_ptr[j]==255 && label<9) key_moving_ptr[u] = 0;
							}
						}
						*/
					}
				}
			}
			// semantic cues remove moving error
			cv::imshow("key_moving_mask", key_moving_mask);
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
						if (u < img_cols/2) cues = predictionsL[i*cropCols+j].second; 
						else cues = predictionsR[i*cropCols+j].second;
						//validation
						if (key_moving_ptr[u]==255 && cues<9) key_moving_ptr[u] = 0;
					}
					else key_moving_ptr[u] = 0;
				}
			}
			// dilate
			int dilate_type = MORPH_RECT;
			int dilate_ele_size = 3;
			Mat ele = getStructuringElement(dilate_type, Size(2*dilate_ele_size+1, 2*dilate_ele_size+1), Point(dilate_ele_size, dilate_ele_size));
			dilate(key_moving_mask, key_moving_mask, ele);
			cv::imshow("refine_key_moving_mask", key_moving_mask);

			keyFrameT ++;
			key_pose = Matrix_::eye(4);
		}
		else continue;
/*
		//save pictures
		char savepic[256];
		sprintf(savepic, "rgb%d.jpg", n);
		cv::imwrite(savepic, img_rgb);
		sprintf(savepic, "disparity%d.jpg", n);
		cv::imwrite(savepic, disp_show_sgbm);
		sprintf(savepic, "segnet%d.jpg", n);
		cv::imwrite(savepic, segnet);
*/
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
			const uchar* moving_ptr = key_moving_mask.ptr<uchar>(v);
			const uchar* roi_ptr = key_roi_mask.ptr<uchar>(v);
			const short* disparity_ptr = key_disp_sgbm.ptr<short>(v);
			const float* recons_ptr = key_xyz.ptr<float>(v);
			const uchar* rgb_ptr = key_img_rgb.ptr<uchar>(v);

			for (int u = 0; u < img_cols; ++u)
			{
				if (v>=img_rows-cropRows && v<img_rows && u>=img_cols/2-cropCols && u<img_cols/2+cropCols)
				{
    				short d = disparity_ptr[u];
    				if (fabs(d)>FLT_EPSILON && roi_ptr[u]!=0 && moving_ptr[u]!=255) //remove moving objects and outside the ROI
					{
						//3d points
						px = recons_ptr[10*u] * pose.val[0][0] + recons_ptr[10*u+1] * pose.val[0][1] + recons_ptr[10*u+2] * pose.val[0][2] + pose.val[0][3];
						py = recons_ptr[10*u] * pose.val[1][0] + recons_ptr[10*u+1] * pose.val[1][1] + recons_ptr[10*u+2] * pose.val[1][2] + pose.val[1][3];
						pz = recons_ptr[10*u] * pose.val[2][0] + recons_ptr[10*u+1] * pose.val[2][1] + recons_ptr[10*u+2] * pose.val[2][2] + pose.val[2][3];
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
		/*
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
		pclut(cloud_anno, tmpcloud);
		rgbviewer.showCloud(tmpcloud, "rgbviewer");
		while( !rgbviewer.wasStopped() ) {}

		float normal_radius_search = static_cast<float>(default_normal_radius_search);
		float leaf_x = default_leaf_size, leaf_y = default_leaf_size, leaf_z = default_leaf_size;
		CloudLT::Ptr crfCloud(new CloudLT); compute(cloud, cloud_anno, normal_radius_search, leaf_x, leaf_y, leaf_z, crfCloud);
		*cloud_anno = *crfCloud;

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpcloud_(new pcl::PointCloud<pcl::PointXYZRGB>);
		pclut(crfCloud, tmpcloud_);
		rgbviewer.showCloud(tmpcloud_, "crfviewer");
		while( !rgbviewer.wasStopped() ) {}
		*/

		/********************* hash update  ********************/
		for (size_t j = 0; j < cloud_anno->points.size(); j++)
		{
			//scalable
			int key_x = int( cloud_anno->points[j].x * gridScale ) + (gridWidth-100*gridScale); 
			int key_y = int( cloud_anno->points[j].y * gridScale ) + gridHeight; 
			int key_z = int( cloud_anno->points[j].z * gridScale ) + (gridWidth-100*gridScale); 
			if (fabs(key_x)>=gridWidth*2 || fabs(key_y)>=gridHeight*2 || fabs(key_z)>=gridWidth*2) continue;

			//hash function
			size_t key = (key_x*gridWidth*gridHeight*4) + (key_z*gridHeight*2) + (key_y);
			if (hash[key].status)
			{
				//update position
				hash[key].pt.x = (hash[key].pt.x + key_x) * 0.5;
				hash[key].pt.y = (hash[key].pt.y + key_y) * 0.5;
				hash[key].pt.z = (hash[key].pt.z + key_z) * 0.5;
			}
			else
			{
				pointCloudNum ++;
				//update position
				hash[key].pt.x = key_x;
				hash[key].pt.y = key_y;
				hash[key].pt.z = key_z;
			}
			//update label and occupancy status
			hash[key].label[cloud_anno->points[j].label] ++;
			hash[key].status ++;
		}

		// vo cloud
		pcl::PointXYZRGB vopoint;
		vopoint.x = pose.val[0][3];
		vopoint.y = pose.val[1][3];
		vopoint.z = pose.val[2][3];
		vopoint.r = 255;
		vopoint.g = 0;
		vopoint.b = 0;
		vooutput->push_back(vopoint);
		voviewer.showCloud(vooutput);
		

		std::cout << BOLDGREEN"Updated[" << n+1 << "]" << BOLDBLUE" (" << keyFrameT << ")" << RESET" " << std::endl;
		//std::cout << BOLDMAGENTA"pose: " << pose.val[0][3] << "," << pose.val[1][3] << "," << pose.val[2][3] << std::endl;
		//std::cout << BOLDYELLOW"Pointcloud: " << pointCloudNum << RESET"" << std::endl;
	}
	/*
	while( !rgbviewer.wasStopped() )
    {
        
    }
	*/
	while( !voviewer.wasStopped() )
    {
        
    }

























 
	//time elapse
	gettimeofday(&t_end, NULL);
	seconds  = t_end.tv_sec  - t_start.tv_sec;
	useconds = t_end.tv_usec - t_start.tv_usec;
	duration = seconds*1000.0 + useconds/1000.0;
	std::cout << BOLDMAGENTA"Average time: " << duration/count << " ms" << RESET" " << std::endl;

	//display
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointXYZRGB point;
	uchar pr, pg, pb;
	for (size_t i = 0; i < gridWidth * gridWidth * gridHeight * 8; ++i)
	{
		if (hash[i].status < 1) continue;	
		point.x = hash[i].pt.x; 
		point.y = hash[i].pt.y; 
		point.z = hash[i].pt.z;
		int maxIndex = 0;
		for (int k = 1; k < 12; k++)
			if (hash[i].label[maxIndex] <= hash[i].label[k])
				maxIndex = k;
		switch (maxIndex)
		{
			//case 0: pb=128; pg=128; pr=128; break; //sky
			case 1: pb=0; pg=0; pr=128; break; //building 
			case 2: pb=128; pg=192; pr=192; break; 
			case 3: pb=0; pg=69; pr=255; break; 
			case 4: pb=128; pg=64; pr=128; break;
			case 5: pb=222; pg=40; pr=60; break; 
			case 6: pb=0; pg=128; pr=128; break; 
			case 7: pb=128; pg=128; pr=192; break; 
			case 8: pb=128; pg=64; pr=64; break; 
			case 9: pb=128; pg=0; pr=64; break; 
			case 10: pb=0; pg=64; pr=64; break;
			case 11: pb=192; pg=128; pr=0; break; 
		}
		uint32_t rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
		point.rgb = *reinterpret_cast<float*>(&rgb);
		point_cloud_ptr->points.push_back(point);
	}

	//outlier removal
	printf("before outlier removal: %d\n", (int) point_cloud_ptr->points.size());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor_;
	sor_.setInputCloud(point_cloud_ptr);
	sor_.setMeanK(50);
	sor_.setStddevMulThresh(1.0);
	sor_.filter(*point_cloud_ptr_filtered);
	printf("after outlier removal: %d\n", (int) point_cloud_ptr_filtered->points.size());

	//create visualizer
	point_cloud_ptr_filtered->width = (int) point_cloud_ptr_filtered->points.size();
	point_cloud_ptr_filtered->height = 1;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud_ptr_filtered);
	viewer->addPointCloud<pcl::PointXYZRGB> (point_cloud_ptr_filtered, rgb, "reconstruction");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "reconstruction");

	print_highlight("The total length: "); print_value("%d ", point_cloud_ptr_filtered->width); printf("frames\n");

	//main loop
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

	return 0;
}

