/*
 * GPUNetSettings.h
 *
 *  Created on: Mar 30, 2014
 *      Author: trevor
 */

#ifndef GPUNETSETTINGS_H_
#define GPUNETSETTINGS_H_

namespace GPUNetSettings {

	enum NetworkStructure {
		STANDARD = 0,
		GPU_ARCH_OPT = 1
	};

	//defaults
	const float GPU_LEARNING_RATE = 0.7;
	const float GPU_MOMENTUM = 0.9;
	const long GPU_MAX_EPOCHS = 1500;
	const int GPU_DESIRED_ACCURACY = 90;
	const int GPU_DESIRED_MSE = 0.001;
	const bool GPU_USE_BATCH = false;

}

#endif /* GPUNETSETTINGS_H_ */
