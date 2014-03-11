/*
 * FeatureVector.h
 *
 *  Created on: Dec 17, 2013
 *      Author: trevor
 */

#ifndef FEATUREVECTOR_H_
#define FEATUREVECTOR_H_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

/**
 * A @FeatureVector holds float pointers to inputs and targets
 */
class FeatureVector {

public:
	float *input;
	float *target;

	CUDA_CALLABLE_MEMBER FeatureVector(float *i, float *t) {
		input = i;
		target = t;
	}

	CUDA_CALLABLE_MEMBER ~FeatureVector() {
		delete[] input;
		delete[] target;
	}
};



#endif /* FEATUREVECTOR_H_ */
