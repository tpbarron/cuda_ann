/*
 * FaceDetect.h
 *
 *  Created on: Apr 8, 2014
 *      Author: trevor
 */

#ifndef FACEDETECT_H_
#define FACEDETECT_H_

#include <map>

#include "GPUNet.h"

class FaceDetect {
public:
	FaceDetect(GPUNet *gnet, std::string keysf);
	~FaceDetect();

	void load_keys(std::string keysf);
	void loop();


private:
	GPUNet *gnet;
	std::map<std::string, std::string> keymap;
};

#endif /* FACEDETECT_H_ */
