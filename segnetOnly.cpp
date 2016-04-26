#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <dirent.h>
#include <thread>

#include "segnet.hpp"

int main(int argc, char** argv) 
{
	// Load network
	string model_file = "/home/inin/SegNet/caffe-segnet/examples/segnet/models/segnet_model_driving_webdemo.prototxt";
	string trained_file = "/home/inin/SegNet/caffe-segnet/examples/segnet/models/segnet_weights_driving_webdemo.caffemodel";
	string label_file = "/home/inin/SegNet/caffe-segnet/examples/segnet/semantic12.txt";
	Classifier classifier;

	// Load image
	string rgbfile = "/media/inin/data/tracking dataset/testing/image_02/0006/";
	char directory[256]; strcpy(directory, rgbfile.c_str()); 
	struct dirent *de;
	DIR *dir = opendir(directory);
	int count = 0;	
	while (de = readdir(dir)) ++count;
	closedir(dir);

	for (int n = 0; n < count; n++)
	{
		// Counting time
		double time = double(getTickCount());

		// Load image
		char base_name[256]; sprintf(base_name,"%06d.png", n);
		string rgbfile_ = rgbfile + base_name;
		cv::Mat img = cv::imread(rgbfile_, 1);
		CHECK(!img.empty()) << "Unable to decode image " << rgbfile_;
		printf("Load image successfully.\n");
		int img_rows = img.rows;
		int img_cols = img.cols;
		int rows = 360;
		int cols = 480;

		// Split image
		cv::Rect roil(img_cols/2-cols,0,cols,rows);
		cv::Rect roir(img_cols/2,0,cols,rows);
		cv::Mat croppedl;
		cv::Mat croppedr;
		img(roil).copyTo(croppedl);
		img(roir).copyTo(croppedr);
		cv::resize(croppedl, croppedl, Size(cols,rows));
		cv::resize(croppedr, croppedr, Size(cols,rows));
	
		// Prediction
		std::vector<Prediction> predictionsl = classifier.Classify(croppedl);
		std::vector<Prediction> predictionsr = classifier.Classify(croppedr);

		// Display segnet
		string colorfile = "/home/inin/SegNet/caffe-segnet/examples/segnet/color.png";
		cv::Mat color = cv::imread(colorfile, 1);
		cv::Mat segnetl(croppedl.size(), CV_8UC3, Scalar(0,0,0));
		cv::Mat segnetr(croppedr.size(), CV_8UC3, Scalar(0,0,0));
		for (int i = 0; i < rows; ++i)
		{	
			uchar* segnetl_ptr = segnetl.ptr<uchar>(i);
			uchar* segnetr_ptr = segnetr.ptr<uchar>(i);
			for (int j = 0; j < cols; ++j)
			{
				segnetl_ptr[j*3+0] = predictionsl[i*cols+j].second;
				segnetl_ptr[j*3+1] = predictionsl[i*cols+j].second;
				segnetl_ptr[j*3+2] = predictionsl[i*cols+j].second;

				segnetr_ptr[j*3+0] = predictionsr[i*cols+j].second;
				segnetr_ptr[j*3+1] = predictionsr[i*cols+j].second;
				segnetr_ptr[j*3+2] = predictionsr[i*cols+j].second;
			}
			
		}
		cv::LUT(segnetl, color, segnetl);
		cv::LUT(segnetr, color, segnetr);

		cv::Mat segnet;
		cv::Mat segimg;
		cv::hconcat(segnetl, segnetr, segnet);
		cv::hconcat(croppedl, croppedr, segimg);

		// Counting time
		time = double(getTickCount() - time) / getTickFrequency() * 1000;
		cout << "cost : " << time << "ms" << endl;

		Mat result;
		addWeighted(segnet, 0.7, segimg, 1.0, 0, result);
		imshow("segnet", segnet);
		imshow("addweight", result);
		waitKey(1);
	}

	return 0;
}


