/*
 * TrainingDataSet.h
 *
 *  Created on: Dec 17, 2013
 *      Author: trevor
 */

#ifndef TRAININGDATASET_H_
#define TRAININGDATASET_H_


class TrainingDataSet {

public:

	/**
	 * To make copies and computation on the GPU easier
	 * store patterns in 1D array
	 * [input1][output1][input2][output2]...[inputn][outputn]
	 */
	float* training_set;
	float* generalization_set;
	float* validation_set;

	TrainingDataSet() {
		n_input = 0;
		n_target = 0;
		n_patterns = 0;
		n_training = 0;
		n_generalization = 0;
		n_validation = 0;
		fpp = 0;
		training_set = NULL;
		generalization_set = NULL;
		validation_set = NULL;
	}

	int floats_per_pattern() {
		return (n_input+1)+n_target;
	}
	int n_patterns, n_training, n_generalization, n_validation;
	int n_input, n_target;
	int fpp;

};



#endif /* TRAININGDATASET_H_ */
