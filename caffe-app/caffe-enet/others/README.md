# comparision outputs blob binary file between opencv implemention and caffe-enet

When I firstlt implent the upsamlpe and bn layer and run test code, then opencv dnn module output diffs from caffe-ent. I added some codes in both caffe-ent and opencv test cpp file to find what's happend.

This code just for memorizing my fist tial.

## caffe-ent
```c++
	...
	
	cv::Mat score = cv::Mat(4, sz, CV_32F, const_cast<float *>(output_layer->cpu_data())).clone();   // 4D 

	auto& layerNames = net_->layer_names();
	for (auto& blobName : net_->blob_names())
	{
		if (std::find(layerNames.cbegin(), layerNames.cend(), blobName) == layerNames.end())
			continue;

		auto& blob = net_->blob_by_name(blobName);
		int cnt = blob->count();
		auto& sp = blob->shape();
		
		int ssz[] = { sp[0],sp[1],sp[2],sp[3]};
		std::string fileName = R"(E:\Microsoft Visual Studio 2015\OpenCv400Test\data\3thdModel\ENet\caffe_out\)" + blobName;
		FILE *f = fopen(fileName.c_str(), "wb");
		fwrite(ssz, sizeof(int), 4, f);		// sp
		fwrite(&cnt, sizeof(int), 1, f);	// cnt blob
		fwrite((void*)blob->cpu_data(), sizeof(float), cnt, f);  
		fclose(f);

		std::cout << cv::format("write %s [%d]", blobName.c_str(), cnt* sizeof(float) + sizeof(float)*5) << std::endl;
	}
	
	...
```


## opencv 
```c++
	...
	
	std::vector<cv::Mat> outBlobs;
	net.forward(outBlobs,layerNames);
	Mat score = outBlobs[outBlobs.size() - 1];

	for (int i = 0; i < score.size[1]; ++i) {
		cv::Mat img(score.size[2],score.size[3],CV_32F, score.ptr<float>(0,i));
	}

	// last layer output
	FILE *f = fopen(R"(E:\Microsoft Visual Studio 2015\OpenCv400Test\data\3thdModel\ENet\caffe_out\caffe_out.bin)", "rb");
	int sz[4] = {};
	fread((void *)sz, sizeof(int), 4, f);
	cv::Mat score_caffe = cv::Mat(4, sz, CV_32F);
	fread((void *)(score_caffe.data), sizeof(float), sz[2]*sz[3], f);
	fclose(f);
	for (int i = 0; i < score.size[1]; ++i) {
		Mat sc = Mat(score.size[2], score.size[3], CV_32F, score.ptr<float>(0, i));
		Mat sc_caffe = Mat(score.size[2], score.size[3], CV_32F, score_caffe.ptr<float>(0, i));
		float s = cv::sum(sc - sc_caffe)[0];
	}

	///// each layer
	for (int j = 0; j < layerNames.size();++j) {
		auto& layerName = layerNames[j];
	
		std::string fileName = R"(..\..\data\3thdModel\ENet\caffe_out\)" + layerName;
		FILE *f = fopen(fileName.c_str(), "rb");
		int sz[4] = {},cnt;
		fread((void *)sz, sizeof(int), 4, f);
		fread(      &cnt, sizeof(int), 1, f);
		cv::Mat score_caffe = cv::Mat(4, sz, CV_32F);
		fread((void *)(score_caffe.data), sizeof(float), cnt, f);
		fclose(f);

		float s = 0;
		auto& outBlob = outBlobs[j];

		assert(outBlob.size[0] == score_caffe.size[0]);
		assert(outBlob.size[1] == score_caffe.size[1]);
		assert(outBlob.size[2] == score_caffe.size[2]);
		assert(outBlob.size[3] == score_caffe.size[3]);

		for (int i = 0; i < outBlob.size[1]; ++i) {
			Mat sc_opencv =  Mat(   outBlob.size[2],      outBlob.size[3], CV_32F,     outBlob.ptr<float>(0, i));
			Mat sc_caffe =   Mat(score_caffe.size[2], score_caffe.size[3], CV_32F, score_caffe.ptr<float>(0, i));
			s = cv::sum(sc_opencv - sc_caffe)[0];
		}
	}

	...
```