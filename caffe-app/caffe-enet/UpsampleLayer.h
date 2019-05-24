#pragma once


#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>

class UpsampleLayer : public cv::dnn::Layer
{
public:
	UpsampleLayer(const cv::dnn::LayerParams &params) : cv::dnn::Layer(params)
	{
		if (params.has("upsample_h") && params.has("upsample_w")) {
			upsample_h_ = params.get<int>("upsample_h");
			upsample_w_ = params.get<int>("upsample_w");
			assert(upsample_h_ >= 1);
			assert(upsample_w_ >= 1);
		}
		else {
			if (!params.has("scale_h")) {
				scale_h_ = scale_w_ = params.get<int>("scale");
				assert(scale_h_ >= 1);
			}
			else {
				scale_h_ = params.get<int>("scale_h");
				scale_w_ = params.get<int>("scale_w");
				assert(scale_h_ >= 1);
				assert(scale_w_ >= 1);
			}
			pad_out_h_ = params.has("pad_out_h") ? params.get<std::string>("pad_out_h") == "true" : pad_out_h_;
			pad_out_w_ = params.has("pad_out_w") ? params.get<std::string>("pad_out_w") == "true" : pad_out_w_;

			assert(!pad_out_h_ || scale_h_ == 2);
			assert(!pad_out_w_ || scale_w_ == 2);

			upsample_h_ = upsample_w_ = -1;
		}

		std::cout << cv::format("%s[%s]:\n", params.name.c_str(), params.type.c_str()) << params << std::endl;
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new UpsampleLayer(params));
	}

	virtual bool getMemoryShapes(const std::vector<std::vector<int> > &inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> > &outputs,
		std::vector<std::vector<int> > &internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);

		assert(4 == inputs[0].size());assert(4 == inputs[1].size());
		assert(inputs[0][0] == inputs[1][0]);
		assert(inputs[0][1] == inputs[1][1]);
		assert(inputs[0][2] == inputs[1][2]);
		assert(inputs[0][3] == inputs[1][3]);


		if (upsample_h_ <= 0 || upsample_w_ <= 0) {
			upsample_h_ = inputs[0][2] * scale_h_ - int(pad_out_h_);
			upsample_w_ = inputs[0][3] * scale_w_ - int(pad_out_w_);
		}

		std::vector<int> outShape(4);
		outShape[0] = inputs[0][0];
		outShape[1] = inputs[0][1];
		outShape[2] = upsample_h_;
		outShape[3] = upsample_w_;
		outputs.assign(1, outShape);

		channels_ = inputs[0][1];
		height_ =	inputs[0][2];
		width_ =	inputs[0][3];

		return false;
	}

	virtual void forward(cv::InputArrayOfArrays inputs_arr,
		cv::OutputArrayOfArrays outputs_arr,
		cv::OutputArrayOfArrays internals_arr) CV_OVERRIDE
	{
		std::vector<cv::Mat> inputs, outputs, internals;
		inputs_arr.getMatVector(inputs);   // 2
		outputs_arr.getMatVector(outputs); // 1

		cv::Mat& inp1 = inputs[0];
		cv::Mat& inp2 = inputs[1];
		cv::Mat& out = outputs[0];		

		out = 0;

		//cv::TickMeter tk;
		//tk.start();

		for (int n = 0; n < inp1.size[0]; ++n) {
			cv::parallel_for_(cv::Range(0, inp1.size[1]), [&](const cv::Range& range) {
				for (int c = range.start; c < range.end; ++c) {

					const float *inp1_ch_ptr = inp1.ptr<float>(n, c);
					const float *inp2_ch_ptr = inp2.ptr<float>(n, c);
					float *out_ch_ptr = out.ptr<float>(n, c);

					for (int i = 0; i < height_ * width_; ++i) {
						const int idx = static_cast<int>(inp2_ch_ptr[i]);
						if (idx >= upsample_h_ * upsample_w_) {
							std::cout << "upsample top index " << idx
								<< " out of range - check scale settings match input pooling layer's downsample setup";
						}
						out_ch_ptr[idx] = inp1_ch_ptr[i];
					}
				}
			});
		}

		//tk.stop();
		//std::cout << cv::format("%s:  %.2f ms", this->name, tk.getTimeMilli()) << std::endl;
	}

private:
	int scale_h_;
	int scale_w_;
	bool pad_out_h_ = false;
	bool pad_out_w_ = false;

	mutable int upsample_h_, upsample_w_;

	mutable int channels_;
	mutable int height_;
	mutable int width_;
};
