/*
 * FaceDetect.cpp
 *
 *  Created on: Apr 8, 2014
 *      Author: trevor
 */

#include "FaceDetect.h"

FaceDetect::FaceDetect(GPUNet *gnet, std::string keysf) {
	FaceDetect::gnet = gnet;
}

FaceDetect::~FaceDetect() {
	// TODO Auto-generated destructor stub
}

/*
 * take video input
 * process frame
 * look for face
 * process face
 * run input through network
 * check output
 * If output in keymap, display value for key
 */
void FaceDetect::loop() {

}

