/*
 * FaceDetect.h
 *
 *  Created on: Apr 8, 2014
 *      Author: trevor
 */

#ifndef FACEDETECT_H_
#define FACEDETECT_H_

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <map>

#include "GPUNet.h"

class FaceDetect {
public:
	FaceDetect(GPUNet *gnet, std::string keysf);
	~FaceDetect();

	bool open_stream(cv::VideoCapture cap, int dev_id);

	void load_keys(std::string keysf);
	void loop();


private:
	GPUNet *gnet;
	std::map<std::string, std::string> keymap;

};

#endif /* FACEDETECT_H_ */
