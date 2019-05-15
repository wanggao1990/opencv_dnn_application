#pragma once
#include "opencv2/opencv.hpp"

#include <iostream>
#include <string>

class BNLayer : public cv::dnn::Layer
{
public:
	BNLayer(const cv::dnn::LayerParams &params) : cv::dnn::Layer(params)
	{		
		bn_mode = params.get<std::string>("bn_mode");

		scale = params.get<float>("scale");
		shift = params.get<float>("shift");

		assert(params.blobs.size() == 2);

		std::cout << cv::format("%s[%s]:\n",params.name.c_str(),params.type.c_str()) <<  params << std::endl;
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new BNLayer(params));
	}


	virtual bool getMemoryShapes(const std::vector<std::vector<int> > &inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> > &outputs,
		std::vector<std::vector<int> > &internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);

		outputs.assign(1, inputs[0]);

		N_ = inputs[0][0];
		C_ = inputs[0][1];
		H_ = inputs[0][2];
		W_ = inputs[0][3];

		return false;
	}

	virtual void forward(cv::InputArrayOfArrays inputs_arr,
		cv::OutputArrayOfArrays outputs_arr,
		cv::OutputArrayOfArrays internals_arr) CV_OVERRIDE
	{
		std::vector<cv::Mat> inputs, outputs;
		inputs_arr.getMatVector(inputs);
		outputs_arr.getMatVector(outputs);

		cv::Mat& inpBlob = inputs[0];
		cv::Mat& outBlob = outputs[0];

		if (bn_mode == "INFERENCE") {

			//input.copyTo(output);
			//return;

			if (!this->blobs.empty()) {
				std::cout << "Skipping parameter initialization" << std::endl;
			}
			else {
				this->blobs.clear();
				this->blobs.push_back(cv::Mat(std::vector<int>{1, C_, 1, 1}, CV_32F, cv::Scalar(scale)));
				this->blobs.push_back(cv::Mat(std::vector<int>{1, C_, 1, 1}, CV_32F, cv::Scalar(shift)));
			}

			CV_Assert(blobs.size() >= 2);
			CV_Assert(inputs.size() == 1);

			CV_Assert(inpBlob.dims == 2 || inpBlob.dims == 4);
			int rows = inpBlob.dims > 2 ? inpBlob.size[2] : 1;
			int cols = inpBlob.dims > 2 ? inpBlob.size[3] : 1;

			//cv::TickMeter tk;
			//tk.start();

			for (size_t ii = 0; ii < outputs.size(); ii++) {   // 1
				Mat &outBlob = outputs[ii];

				for (int num = 0; num < outBlob.size[0]; num++) {   // N
				//	for (int c = 0; c < outBlob.size[1]; n++) {    // C
					cv::parallel_for_(cv::Range(0, outBlob.size[1]),[&](auto& r){
						for (int c = r.start; c < r.end; ++c) {
							float w = *blobs[0].ptr<float>(0, c);   // 1*C*1*1
							float b = *blobs[1].ptr<float>(0, c);
							cv::Mat inpBlobPlane(rows, cols, CV_32F, inpBlob.ptr<float>(num, c));
							cv::Mat outBlobPlane(rows, cols, CV_32F, outBlob.ptr<float>(num, c));
							inpBlobPlane.convertTo(outBlobPlane, CV_32F, w, b);
						}
					});
				}
			}
			//tk.stop();
			//std::cout << cv::format("%s:   %.2f ms", this->name,tk.getTimeMilli()) << std::endl;

			return;
		}
	}

private:
	std::string bn_mode = "LEARN";
	float scale;
	float shift;
	
	// dimension
	mutable int N_;
	mutable int C_;
	mutable int H_;
	mutable int W_;
	
	// eps
	mutable float var_eps_;
};
