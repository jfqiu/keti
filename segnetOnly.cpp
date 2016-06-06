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
	string rgbfile = "/home/inin/SegNet/caffe-segnet/examples/segnet/lh-intersection/";
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
		char base_name[256]; sprintf(base_name,"%06d.jpg", n);
		string rgbfile_ = rgbfile + base_name;
		cv::Mat img = cv::imread(rgbfile_, 1);
		CHECK(!img.empty()) << "Unable to decode image " << rgbfile_;
		printf("Load image successfully.\n");
		int img_rows = img.rows;
		int img_cols = img.cols;
		int rows = 360;
		int cols = 480;

		// resize image
		cv::resize(img, img, Size(cols,rows));
	
		// Prediction
		std::vector<Prediction> predictions = classifier.Classify(img);

		// Display segnet
		string colorfile = "/home/inin/SegNet/caffe-segnet/examples/segnet/color.png";
		cv::Mat color = cv::imread(colorfile, 1);
		cv::Mat segnet(img.size(), CV_8UC3, Scalar(0,0,0));
		for (int i = 0; i < rows; ++i)
		{	
			uchar* segnet_ptr = segnet.ptr<uchar>(i);
			for (int j = 0; j < cols; ++j)
			{
				segnet_ptr[j*3+0] = predictions[i*cols+j].second;
				segnet_ptr[j*3+1] = predictions[i*cols+j].second;
				segnet_ptr[j*3+2] = predictions[i*cols+j].second;
			}
			
		}
		cv::LUT(segnet, color, segnet);

		// Counting time
		time = double(getTickCount() - time) / getTickFrequency() * 1000;
		cout << "cost : " << time << "ms" << endl;

		Mat result;
		addWeighted(segnet, 1.0, img, 0.5, 0, result);
		imshow("img", img);
		imshow("segnet", segnet);
		imshow("addweight", result);

		waitKey(0);
		char name[256];
		//sprintf(name, "segnet_lh/%dimg.bmp", n);
		//imwrite(name, img);
		//sprintf(name, "segnet_lh/%dsegnet.bmp", n);
		//imwrite(name, segnet);
		//sprintf(name, "segnet_lh/%dresult.bmp", n);
		//imwrite(name, result);
	}

	return 0;
}


