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
int roiy = 5; //0006-10
int roiz = 30;

//colormap
string colorfile = "/home/inin/SegNet/caffe-segnet/examples/segnet/color.png";

//size
int cropRows = 360;
int cropCols = 480;
int cropCols2 = 960;

//data directory
//string rgb_dirL = "/media/inin/data/tracking dataset/testing/image_02/0006/"; //tt0006 tr0013 tt0011 tt0003  tr0009 173-213
//string rgb_dirR = "/media/inin/data/tracking dataset/testing/image_03/0006/";
string rgb_dirL = "/media/inin/data/sequences/05/image_2/";
string rgb_dirR = "/media/inin/data/sequences/05/image_3/";
string velodyne_dir = "/media/inin/data/dataset/sequences/05/velodyne/";
//00(segnet-bad) 05(good-xiaodao) 07(confuse-xiaodao) 12(gaosu) 13(xiaodao) 16(kuandao) 18(segnet-bad) 19(segnet-confuse) 21(gaosu)

//update voxel
const int gridScale = 4; //05-2 0006-10
const int gridWidth = gridScale*200; //05-200 0006-80
const int gridHeight = gridScale*15; //05-15 0006-5

//crf param
float  default_leaf_size = 0.02; //0.05f 0.03f is also valid
double default_feature_threshold = 5.0; //5.0
double default_normal_radius_search = 0.5; //0.03

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
	Matrix_ gtpose_ = Matrix_::eye(4); //visual odometry
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
	cv::Mat key_moving_mask, key_roi_mask, key_xyz, key_disp_sgbm, key_img_rgb, key_moving_mask_before;
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

	int count = 1200;   //05-1200 07-1085 0006-50
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
		char base_name_p[256], base_name_c[256], base_name_v[256];
		sprintf(base_name_p, "%06d.png", n);
		sprintf(base_name_c, "%06d.png", n+1);
		sprintf(base_name_v, "%06d.bin", n+1);

		string imgpath_lp = rgb_dirL + base_name_p; string imgpath_rp = rgb_dirR + base_name_p;
		string imgpath_lc = rgb_dirL + base_name_c; string imgpath_rc = rgb_dirR + base_name_c;
		string velodyne_infile = velodyne_dir + base_name_v;
		img_lc = cv::imread(imgpath_lc,0); img_lp = cv::imread(imgpath_lp,0);
		img_rc = cv::imread(imgpath_rc,0); img_rp = cv::imread(imgpath_rp,0);
		img_rgb = cv::imread(imgpath_lc,1);

		// load point cloud
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
		cv::imshow("segnet", segnet);
		cv::imshow("disparity", disp_show_sgbm);
		if (!success) continue;

		//char path[255];
		//sprintf(path,"pic/segnet%d.jpg",n);
		//cv::imwrite(path,segnet);

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
    				//if (fabs(d)>FLT_EPSILON && roi_ptr[u]!=0 && moving_ptr[u]!=255) //remove moving objects and outside the ROI
    				if (fabs(d)>FLT_EPSILON && roi_ptr[u]!=0) //remove moving objects and outside the ROI
					{
						//3d points
						px = recons_ptr[10*u] * pose.val[0][0] + recons_ptr[10*u+1] * pose.val[0][1] + recons_ptr[10*u+2] * pose.val[0][2] + pose.val[0][3];
						py = recons_ptr[10*u] * pose.val[1][0] + recons_ptr[10*u+1] * pose.val[1][1] + recons_ptr[10*u+2] * pose.val[1][2] + pose.val[1][3];
						pz = recons_ptr[10*u] * pose.val[2][0] + recons_ptr[10*u+1] * pose.val[2][1] + recons_ptr[10*u+2] * pose.val[2][2] + pose.val[2][3];
						//px = recons_ptr[10*u];
						//py = recons_ptr[10*u+1];
						//pz = recons_ptr[10*u+2];
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
		*/

		
		float normal_radius_search = static_cast<float>(default_normal_radius_search);
		float leaf_x = default_leaf_size, leaf_y = default_leaf_size, leaf_z = default_leaf_size;
		CloudLT::Ptr crfCloud(new CloudLT); compute(cloud, cloud_anno, normal_radius_search, leaf_x, leaf_y, leaf_z, crfCloud);
		*cloud_anno = *crfCloud;

		
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpcloud_(new pcl::PointCloud<pcl::PointXYZRGB>);
		pclut(crfCloud, tmpcloud_);
		//rgbviewer.showCloud(tmpcloud_, "crfviewer");
		//while( !rgbviewer.wasStopped() ) {}



//		char pathname[256];
//		sprintf(pathname, "pcd/pcl%d.ply", n);
//		pcl::PLYWriter writer;	
//		tmpcloud_->width = (int) tmpcloud_->points.size();
//		tmpcloud_->height = 1;
// 		writer.write (pathname, *tmpcloud_);

/*
		// get velodyne points
		input.seekg(0, ios::beg);
		cv::Mat disp_gt = cv::Mat(disp_sgbm.size(), CV_32F, cv::Scalar(0)); 
		pcl::PointCloud<PointXYZI>::Ptr points (new pcl::PointCloud<PointXYZI>);
		for (int i=0; input.good() && !input.eof(); i++) 
		{
			PointXYZI point;
			input.read((char *) &point.x, 3*sizeof(float));
			input.read((char *) &point.intensity, sizeof(float));
			//points->push_back(point);

			// calib_velodyne_to_cam0
			float x = point.x * 7.533745e-03 + point.y * -9.999714e-01 + point.z * -6.166020e-04 - 4.069766e-03; 
			float y = point.x * 1.480249e-02 + point.y * 7.280733e-04  + point.z * -9.998902e-01 - 7.631618e-02;
			float z = point.x * 9.998621e-01 + point.y * 7.523790e-03 + point.z * 1.480755e-02 - 2.717806e-01;
			//7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03 
			//1.480249e-02 7.280733e-04 -9.998902e-01 -7.631618e-02
			//9.998621e-01 7.523790e-03 1.480755e-02 -2.717806e-01

			// calib_velodyne_to_cam0_to_cam0plane
			int u = (x * 7.070912000000e+02 / z + 6.018873000000e+02);
			int v = (y * 7.070912000000e+02 / z + 1.831104000000e+02);
			//7.070912000000e+02 0.000000000000e+00 6.018873000000e+02 0.000000000000e+00
		    //0.000000000000e+00 7.070912000000e+02 1.831104000000e+02 0.000000000000e+00
		    //0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
			if (fabs(x)>roix || fabs(y)>roiy || fabs(z)>roiz) continue;
			if (u>=0 && u<disp_sgbm.cols && v>=0 && v<disp_sgbm.rows)
			{
				float* disp_ptr = disp_gt.ptr<float>(v);
				disp_ptr[u] = z;
			}
		}

	    // projection_cam1_to_cam0
		cv::Mat disp_calib = cv::Mat(disp_sgbm.size(), CV_32F, cv::Scalar(0)); 
	    for (int i = 0; i<disp_sgbm.rows; i++)
	    {
	        const float* recons_ptr = key_xyz.ptr<float>(i);
	        for (int j = 0; j<disp_sgbm.cols; j++)
	        {
	        	float px = recons_ptr[10*j] + 0.06;
	        	float py = recons_ptr[10*j+1];
	        	float pz = recons_ptr[10*j+2];
				int u = (px * 7.070912000000e+02 / pz + 6.018873000000e+02);
				int v = (py * 7.070912000000e+02 / pz + 1.831104000000e+02);
				if (fabs(px)>roix || fabs(py)>roiy || fabs(pz)>roiz) continue;

				if (u>=0 && u<disp_sgbm.cols && v>=0 && v<disp_sgbm.rows)
				{
					float* disp_calib_ptr = disp_calib.ptr<float>(v);
					disp_calib_ptr[u] = pz;
				}
	        }
	    }

	    // correctness evaluation
	    double correct = 0.0;
	    double sum = 0.0;
	    for (int i = 0; i < disp_sgbm.rows; i++)
	    {
			float* disp_calib_ptr = disp_calib.ptr<float>(i);
	        float* disp_gt_ptr = disp_gt.ptr<float>(i);
	        for (int j = 0; j < disp_sgbm.cols; j++)
	        {
	            float depth = disp_calib_ptr[j];
	            float depth_gt = disp_gt_ptr[j];

	            if (fabs(depth-depth_gt) < 32.0) 
	            	correct += 1.0;
	            sum += 1.0;
	        }
	    }
	    cv::imshow("disp_gt", disp_gt);
		std::cout << BOLDYELLOW"Reconstruction Evaluation[" << n+1 << "]" << BOLDBLUE" (" << correct << "/" << sum << ")" << correct/sum << RESET" " << std::endl;
*/
		input.close();

		/********************* refine segnet ********************/
		cv::Mat img(360,960,CV_8UC3,cv::Scalar(0,0,0));
		for (size_t j = 0; j < cloud_anno->points.size(); j++)
		{
			Matrix_ poseinv = Matrix_::inv(pose);
			double x = (-cloud_anno->points[j].x) * poseinv.val[0][0] + (-cloud_anno->points[j].y) * poseinv.val[0][1] + cloud_anno->points[j].z * poseinv.val[0][2] + poseinv.val[0][3];
			double y = (-cloud_anno->points[j].x) * poseinv.val[1][0] + (-cloud_anno->points[j].y) * poseinv.val[1][1] + cloud_anno->points[j].z * poseinv.val[1][2] + poseinv.val[1][3];
			double z = (-cloud_anno->points[j].x) * poseinv.val[2][0] + (-cloud_anno->points[j].y) * poseinv.val[2][1] + cloud_anno->points[j].z * poseinv.val[2][2] + poseinv.val[2][3];

			int u = (x * f / z + c_u);
			int v = (y * f / z + c_v);
			//int u = (-cloud_anno->points[j].x * f / cloud_anno->points[j].z + c_u);
			//int v = (-cloud_anno->points[j].y * f / cloud_anno->points[j].z + c_v);
			u -= (img_cols/2 - cropCols);
			v -= (img_rows - cropRows);
			if (u>=0 && u<960 && v>=0 && v<360)
			{
				uchar* imgptr = img.ptr<uchar>(v);
				imgptr[u*3] = cloud_anno->points[j].label;
				imgptr[u*3+1] = cloud_anno->points[j].label;
				imgptr[u*3+2] = cloud_anno->points[j].label;
			}
		}

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

		/********************* hash update  ********************/
		for (size_t j = 0; j < cloud_anno->points.size(); j++)
		{

			//scalable
			int key_x = int( cloud_anno->points[j].x * gridScale ) + 280*gridScale; //05-280 0006-280
			int key_y = int( cloud_anno->points[j].y * gridScale ) + 15*gridScale;  //05-15 0006-15
			int key_z = int( cloud_anno->points[j].z * gridScale ) + 20*gridScale;  //05-20 0006-20

			if (key_x < 0 || key_y < 0 || key_z < 0 ||
					fabs(key_x)>=gridWidth*2 || fabs(key_y)>=gridHeight*2 || fabs(key_z)>=gridWidth*2) continue;

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
		vopoint.r = 0;
		vopoint.g = 0;
		vopoint.b = 255;
		vooutput->push_back(vopoint);

		vopoint.x = gtpose_.val[0][3];
		vopoint.y = gtpose_.val[1][3];
		vopoint.z = gtpose_.val[2][3];
		vopoint.r = 255;
		vopoint.g = 0;
		vopoint.b = 0;
		vooutput->push_back(vopoint);

		voviewer.showCloud(vooutput);
		
		std::cout << BOLDGREEN"Updated[" << n+1 << "]" << BOLDBLUE" (" << keyFrameT << ")" << RESET" " << std::endl;
		std::cout << BOLDMAGENTA"pose: " << pose.val[0][3] << "," << pose.val[1][3] << "," << pose.val[2][3] << std::endl;
		std::cout << BOLDMAGENTA"gtpose_: " << gtpose_.val[0][3] << "," << gtpose_.val[1][3] << "," << gtpose_.val[2][3] << std::endl;
		//std::cout << BOLDYELLOW"Pointcloud: " << pointCloudNum << RESET"" << std::endl;
	}
 
	//time elapse
	gettimeofday(&t_end, NULL);
	seconds  = t_end.tv_sec  - t_start.tv_sec;
	useconds = t_end.tv_usec - t_start.tv_usec;
	duration = seconds*1000.0 + useconds/1000.0;
	std::cout << BOLDMAGENTA"Average time: " << duration/count << " ms" << RESET" " << std::endl;

	std::cout << BOLDYELLOW"Waiting to Reconstruction..." << RESET" " << std::endl;
	
	//while( !rgbviewer.wasStopped() )
    {
        
    }

	vooutput->width = (int) vooutput->points.size();
	vooutput->height = 1;
	while( !voviewer.wasStopped() )
    {
        
    }
	pcl::io::savePCDFileASCII ("voresult.pcd", *vooutput);



































	//display
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointXYZRGB point;
	uchar pr, pg, pb;
	size_t bar = 0;
	size_t progress = 1;
	size_t total = gridWidth * gridWidth * gridHeight * 8 / 100;
	for (size_t i = 0; i < gridWidth * gridWidth * gridHeight * 8; ++i)
	{
		bar ++;
		if (bar/total == progress)
		{
			progress ++;
			std::cout << BOLDYELLOW"Progress bar: " << bar/total << "%" << RESET" " << std::endl; 
		}

		if (hash[i].status < 1) continue;	
		point.x = hash[i].pt.x; 
		point.y = hash[i].pt.y; 
		point.z = hash[i].pt.z;
		int maxIndex = 0;
		for (int k = 1; k < 12; k++)
			if (hash[i].label[maxIndex] <= hash[i].label[k])
				maxIndex = k;
		if (maxIndex >= 9) continue;
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
	std::cout << BOLDMAGENTA"Before outlier removal: " << point_cloud_ptr->points.size() << RESET" " << std::endl;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor_;
	sor_.setInputCloud(point_cloud_ptr);
	sor_.setMeanK(50);
	sor_.setStddevMulThresh(1.0);
	sor_.filter(*point_cloud_ptr_filtered);
	std::cout << BOLDMAGENTA"After outlier removal: " << point_cloud_ptr_filtered->points.size() << RESET" " << std::endl;

	//create visualizer
	point_cloud_ptr_filtered->width = (int) point_cloud_ptr_filtered->points.size();
	point_cloud_ptr_filtered->height = 1;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud_ptr_filtered);
	viewer->addPointCloud<pcl::PointXYZRGB> (point_cloud_ptr_filtered, rgb, "reconstruction");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "reconstruction");

	std::cout << BOLDWHITE"The total length:" << BOLDBLUE" " << point_cloud_ptr_filtered->width << BOLDWHITE" points" << RESET" " << std::endl;

	//main loop
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

	/********* mesh ********/
	//postMesh(point_cloud_ptr_filtered);

	/********* save ********/  
	//PCDWriter w;
  	//w.writeBinaryCompressed ("result.pcd", *point_cloud_ptr_filtered);
	pcl::io::savePCDFileASCII ("result.pcd", *point_cloud_ptr_filtered);

	return 0;
}

