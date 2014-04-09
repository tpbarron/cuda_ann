/*
 * NetData.h
 *
 *  Created on: Dec 15, 2013
 *      Author: trevor
 */

#ifndef NETDATA_H_
#define NETDATA_H_

#include <cstdlib>
#include <string>
#include <vector>

#include <thrust/host_vector.h>

#include "TrainingDataSet.h"
#include "FeatureVector.h"


class NetData {

public:

	NetData(float tpct);
	~NetData();

	size_t num_feature_vecs();
	unsigned int num_inputs();
	unsigned int num_targets();

	bool load_file(std::string fname);

	void add_to_dataset(float* set, int floats_per_pattern, int p, int i);
	TrainingDataSet* get_training_dataset();

	void print_loaded_patterns();
	void print_loaded_patterns_flatted();

	std::vector<FeatureVector*> data;
private:

	// storage

	unsigned int n_patterns;
	unsigned int n_inputs;
	unsigned int n_targets;
	float t_set_pct;

	//reference to training set
	TrainingDataSet tset;
};

#endif /* NETDATA_H_ */
