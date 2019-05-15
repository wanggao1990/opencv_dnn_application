
#include <iostream>
#include <fstream>
#include <sstream>

#include "opencv2/opencv.hpp"
#include "opencv2/dnn/layer.details.hpp"

#include "BNLayer.h"
#include "UpsampleLayer.h"

using namespace cv;

int main()
{

	try {

		// REGISTER_LAYER
		CV_DNN_REGISTER_LAYER_CLASS(BN, BNLayer);
		CV_DNN_REGISTER_LAYER_CLASS(Upsample, UpsampleLayer);

		float scale = 1;
		cv::Scalar mean = Scalar{ 0,0,0 };
		bool swapRB = false;
		int inpWidth = 1024;
		int inpHeight = 512;

		std::string modelPath = R"(..\model\cityscapes_weights.caffemodel)";
		std::string configPath = R"(..\model\data\3thdModel\ENet\enet_deploy_final.prototxt)";

		String framework = "";
		int backendId = cv::dnn::DNN_BACKEND_OPENCV;
		int targetId = cv::dnn::DNN_TARGET_CPU;

		cv::dnn::Net net = cv::dnn::readNet(modelPath, configPath, framework);
		net.setPreferableBackend(backendId);
		net.setPreferableTarget(targetId);

		// Create a window
		static const std::string kWinName = "Deep learning semantic segmentation in OpenCV";

		Mat frame = imread(R"(../../data/3thdModel/ENet/strasbourg_000001_033027.jpg)");

		if (frame.empty()) {
			std::cerr << "cant load image" << std::endl;
		}

		Mat blob;
		cv::dnn::blobFromImage(frame, blob, scale, Size(inpWidth, inpHeight), mean, swapRB, false);
		net.setInput(blob);

		Mat score = net.forward();

		Mat segm;

		cv::Mat class_each_row(score.size[1], score.size[2] * score.size[3], CV_32FC1, (float *)(score.data));
		cv::Point maxId;
		double maxValue;
		cv::Mat prediction_map(score.size[2], score.size[3], CV_8UC1, Scalar(0));
		for (int i = 0; i < class_each_row.cols; i++) {
			minMaxLoc(class_each_row.col(i), 0, &maxValue, 0, &maxId);
			prediction_map.at<uchar>(i) = maxId.y;
		}
		cv::cvtColor(prediction_map, prediction_map, cv::COLOR_GRAY2BGR);
		std::string lut = R"(..\data\model\cityscapes19.png)";
		cv::Mat label_colours = cv::imread(lut, 1);
		cv::cvtColor(label_colours, label_colours, cv::COLOR_RGB2BGR);
		LUT(prediction_map, label_colours, segm);

		// Put efficiency information.
		std::vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		std::string label = format("Inference time: %.2f ms", t);
		putText(segm, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

		imshow(kWinName, segm);
		waitKey();
	}
	catch (std::exception & e) {
		std::cerr << e.what() << std::endl;
	}
}
