/*
 * TrainingDataSet.h
 *
 *  Created on: Dec 17, 2013
 *      Author: trevor
 */

#ifndef TRAININGDATASET_H_
#define TRAININGDATASET_H_

#include <thrust/host_vector.h>

class TrainingDataSet {

public:

	thrust::host_vector<FeatureVector*> training_set;
	thrust::host_vector<FeatureVector*> generalization_set;
	thrust::host_vector<FeatureVector*> validation_set;

	TrainingDataSet() {
		n = 0;
	}

	void clear() {
		training_set.clear();
		generalization_set.clear();
		validation_set.clear();
	}

	/**
	 * Return the number of total patterns in all sets
	 */
	size_t size() {
		return n;
	}

	void set_size(size_t s) {
		n = s;
	}

private:
	size_t n;

};



#endif /* TRAININGDATASET_H_ */
