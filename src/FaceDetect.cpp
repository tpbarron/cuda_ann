/*
 * FaceDetect.cpp
 *
 *  Created on: Apr 8, 2014
 *      Author: trevor
 */


//#include <iostream>

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
	cv::VideoCapture cap;
	cv::CascadeClassifier haar_cascade;
	open_stream(cap, 0);
	haar_cascade.load("haarcascade_frontalface_default.xml");
	cv::Mat frame;
	while (true) {
		cap >> frame;
		//Mat original = cv::frame.clone();
		cv::Mat gray;
		cvtColor(frame, gray, CV_BGR2GRAY);
		// Find the faces in the frame:
		cv::vector< cv::Rect_<int> > faces;
		haar_cascade.detectMultiScale(gray, faces);
		for(int i = 0; i < faces.size(); i++) {
			cv::Rect face_i = faces[i];
			// Crop the face from the image. So simple with OpenCV C++:
			//Mat face = gray(face_i);
			cv::rectangle(frame, face_i, CV_RGB(0, 255,0), 1);
			//string box_text = format("Prediction = %d", prediction);
			// Calculate the position for annotated text (make sure we don't
			// put illegal values in there):
			//int pos_x = std::max(face_i.tl().x - 10, 0);
			//int pos_y = std::max(face_i.tl().y - 10, 0);
			// And now put it into the image:
			//putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
			break;
		}

		cv::imshow("face_recognizer",frame);
		char key = (char) cv::waitKey(20);
		if (key == 27)
			break;
	}
}

bool FaceDetect::open_stream(cv::VideoCapture cap, int dev_id) {
	cap.open(dev_id);
	// Check if we can use this device at all:
	if(!cap.isOpened()) {
		std::cerr << "Capture Device ID " << dev_id << "cannot be opened." << std::endl;
		return false;
	}
	std::cout << "Device opened " << dev_id << std::endl;
	return true;
}
