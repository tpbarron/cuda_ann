/*
 * NetData.h
 *
 *  Created on: Dec 15, 2013
 *      Author: trevor
 */

#ifndef NETDATA_H_
#define NETDATA_H_

#include <vector>
#include <thrust/host_vector.h>
#include <cstdlib>
#include <string>
#include "FeatureVector.h"
#include "TrainingDataSet.h"

class NetData {

public:

	NetData();
	~NetData();

	size_t num_feature_vecs();
	int num_inputs();
	int num_targets();

	bool load_file(std::string fname);
	TrainingDataSet* get_training_dataset();

	void print_loaded_patterns();

private:

	// storage
	std::vector<FeatureVector*> data;
	int n_inputs;
	int n_targets;

	int training_data_end_index;

	//reference to training set
	TrainingDataSet tset;
};

#endif /* NETDATA_H_ */
