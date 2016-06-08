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

//data directory
string rgb_dirL = "/media/inin/data/sequences/07/image_2/";
string rgb_dirR = "/media/inin/data/sequences/07/image_3/";
string velodyne_dir = "/media/inin/data/dataset/sequences/07/velodyne/";

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
	size_t fusionLen = 1; double translationT = 31; double rotationT = 1; double RT_Threshold = 1;//translation 10_m  rotation 5_degree

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

	double depth_accuracy = 0;
	double depth_thres = 1.0;
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
		char base_name_p[256], base_name_c[256], base_name_s[256], base_name_v[256];
		sprintf(base_name_p, "%06d.png", n-1);
		sprintf(base_name_c, "%06d.png", n);
		sprintf(base_name_s, "%06d.png", n);
		sprintf(base_name_v, "%06d.bin", n);

		string imgpath_lp = rgb_dirL + base_name_p; string imgpath_rp = rgb_dirR + base_name_p;
		string imgpath_lc = rgb_dirL + base_name_c; string imgpath_rc = rgb_dirR + base_name_c;
		string semantic_path = string(dirname) + base_name_s;
		string velodyne_infile = velodyne_dir + base_name_v;
		img_lc = cv::imread(imgpath_lc,0); img_lp = cv::imread(imgpath_lp,0);
		img_rc = cv::imread(imgpath_rc,0); img_rp = cv::imread(imgpath_rp,0);
		img_rgb = cv::imread(imgpath_lc,1);
		img_semantic = cv::imread(semantic_path, 1);

		fstream input(velodyne_infile.c_str(), ios::in | ios::binary);
		if (!input.good()) {
			cerr << "Could not read velodyne file: " << velodyne_infile << endl;
			exit(EXIT_FAILURE);
		}

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

		//build key frame
		//double thetax = atan2(key_pose.val[2][1], key_pose.val[2][2]) * 180.0 / M_PI;	
		//double thetay = atan2(-key_pose.val[2][0], sqrt(key_pose.val[2][1]*key_pose.val[2][1] + key_pose.val[2][2]*key_pose.val[2][2])) * 180.0 / M_PI;	
		//double thetaz = atan2(key_pose.val[1][0], key_pose.val[0][0]) * 180.0 / M_PI;
		//double translationx = key_pose.val[0][3];		
		//double translationy = key_pose.val[1][3];		
		//double translationz = key_pose.val[2][3];	
		//double r_change = sqrt( (thetax*thetax) + (thetay*thetay) + (thetaz*thetaz) );
		//double t_change = sqrt( (translationx*translationx) + (translationy*translationy) + (translationz*translationz) );
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
			/*
			// semantic cues remove moving error
			cv::imshow("key_moving_mask_before", key_moving_mask);
			key_moving_mask_before = key_moving_mask;
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
			}*/
			
			// dilate
			//int dilate_type = MORPH_RECT;
			//int dilate_ele_size = 3;
			//Mat ele = getStructuringElement(dilate_type, Size(2*dilate_ele_size+1, 2*dilate_ele_size+1), Point(dilate_ele_size, dilate_ele_size));
			//dilate(key_moving_mask, key_moving_mask, ele);
			//cv::imshow("refine_key_moving_mask", key_moving_mask);

			keyFrameT ++;
			key_pose = Matrix_::eye(4);
		}
		else 
		{
			input.close();
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
		
		//pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
		//pclut(cloud_anno, tmpcloud);
		//rgbviewer.showCloud(tmpcloud, "rgbviewer");
		//while( !rgbviewer.wasStopped() ) {}
		
		
		float normal_radius_search = static_cast<float>(default_normal_radius_search);
		float leaf_x = default_leaf_size, leaf_y = default_leaf_size, leaf_z = default_leaf_size;
		CloudLT::Ptr crfCloud(new CloudLT); compute(cloud, cloud_anno, normal_radius_search, leaf_x, leaf_y, leaf_z, crfCloud);
		*cloud_anno = *crfCloud;
		
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpcloud_(new pcl::PointCloud<pcl::PointXYZRGB>);
		pclut(crfCloud, tmpcloud_);
		rgbviewer.showCloud(tmpcloud_, "crfviewer");
		//while( !rgbviewer.wasStopped() ) {}

		/****************3d evaluation******************/
		cv::Mat projection(cropRows, cropCols2, CV_32F, cv::Scalar(0));
		for (size_t j = 0; j < cloud_anno->points.size(); j++)
		{
			double x = cloud_anno->points[j].x + 0.06;
			double y = cloud_anno->points[j].y;
			double z = cloud_anno->points[j].z;

			int u = (x * (f / z) + c_u);
			int v = (y * (f / z) + c_v);

			u -= (img_cols/2 - cropCols);
			v -= (img_rows - cropRows);

			if (u>=0 && u<cropCols2 && v>=0 && v<cropRows)
			{
				float* projection_ptr = projection.ptr<float>(v);
				projection_ptr[u] = z;
			}
		}

		// get gt
		input.seekg(0, ios::beg);
		cv::Mat gt = cv::Mat(cropRows, cropCols2, CV_32F, cv::Scalar(0)); 
		//pcl::PointCloud<PointXYZI>::Ptr points (new pcl::PointCloud<PointXYZI>);
		for (int i=0; input.good() && !input.eof(); i++) 
		{
			pcl::PointXYZI point;
			input.read((char *) &point.x, 3*sizeof(float));
			input.read((char *) &point.intensity, sizeof(float));

			// calib_velodyne_to_cam0
			float x = point.x * 7.533745e-03 + point.y * -9.999714e-01 + point.z * -6.166020e-04 - 4.069766e-03; 
			float y = point.x * 1.480249e-02 + point.y * 7.280733e-04  + point.z * -9.998902e-01 - 7.631618e-02;
			float z = point.x * 9.998621e-01 + point.y * 7.523790e-03 + point.z * 1.480755e-02 - 2.717806e-01;
			//points->push_back(point);

			// calib_velodyne_to_cam0_to_cam0plane
			int u = (x * f / z + c_u);
			int v = (y * f / z + c_v);

			u -= (img_cols/2 - cropCols);
			v -= (img_rows - cropRows);

    		if (fabs(x)>roix || fabs(y)>roiy || fabs(z)>roiz)
    			continue;
			if (u>=0 && u<cropCols2 && v>=0 && v<cropRows)
			{
				float* gt_ptr = gt.ptr<float>(v);
				gt_ptr[u] = z;
			}
		}

		// correctness evaluation
	    double correct = 0.0;
	    double sum_depth = 0.0;
		cv::Mat result = cv::Mat(cropRows, cropCols2, CV_32F, cv::Scalar(0)); 
	    for (int i = 0; i < disp_sgbm.rows; i++)
	    {
			float* projection_ptr = projection.ptr<float>(i);
	        float* gt_ptr = gt.ptr<float>(i);
	        float* result_ptr = result.ptr<float>(i);
	        for (int j = 0; j < disp_sgbm.cols; j++)
	        {
	            float depth = projection_ptr[j];
	            float depth_gt = gt_ptr[j];
	            if (depth==0 || depth_gt<0.1) continue;

	            if (fabs(depth-depth_gt) < depth_thres) 
	            	correct += 1.0;
	            sum_depth += 1.0;

	            result_ptr[j] = 255;
	        }
	    }
		std::cout << "Reconstruction Evaluation[" << n+1 << "]" << " (" << correct << "/" << sum_depth << ")" << correct/sum_depth << std::endl;
		depth_accuracy += correct/sum_depth;


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


		// Semantic Evaluation
		cv::Rect crop(img_cols/2-cropCols,img_rows-cropRows,cropCols*2,cropRows);
		cv::Mat crop_gt;
		img_semantic(crop).copyTo(crop_gt);
		cv::resize(crop_gt, crop_gt, Size(960,360));

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

				if (pb_refine==0 && pg_refine==0 && pr_refine==0) continue;

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
					if (pb_segnet==128 && pg_segnet==64 && pr_segnet==64) 
						fence_s += 1;
					if (pb_refine==128 && pg_refine==64 && pr_refine==64) 
						fence_r += 1;
					fence += 1;
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

		std::cout << BOLDYELLOW"Average Semantic Accuracy: " << average_s << "," << average_r << std::endl << RESET"" << std::endl;
		std::cout << BOLDYELLOW"Global Semantic Accuracy: " << sum_s/sum << "," << sum_r/sum << std::endl << RESET"" << std::endl;



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

		imshow("projection", projection);
		imshow("gt", gt);
		imshow("result", result);

		cv::imshow("img",img);
		cv::imshow("segnet", segnet);
		cv::imshow("gt", crop_gt);
		cv::waitKey(1);

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
/*
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
	cout << "Depth Accuracy [" << depth_thres << "]: " << depth_accuracy/gt_file_names.size() << endl;
 
	//time elapse
	gettimeofday(&t_end, NULL);
	seconds  = t_end.tv_sec  - t_start.tv_sec;
	useconds = t_end.tv_usec - t_start.tv_usec;
	duration = seconds*1000.0 + useconds/1000.0;
	//std::cout << BOLDMAGENTA"Average time: " << duration/count << " ms" << RESET" " << std::endl;
	//while( !rgbviewer.wasStopped() )
    {
        
    }

	return 0;
}

