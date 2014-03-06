/*
 * Net.h
 *
 *  Created on: Dec 9, 2013
 *      Author: trevor
 */

#ifndef NET_H_
#define NET_H_

#include <string>
#include <vector>
#include <cmath>
#include "NetData.h"

class Net {

public:
	Net(int ni, int nh, int no);
	~Net();

	//num neurons
	int n_input, n_hidden, n_output;

	//neurons
	float* inputNeurons;
	float* hiddenNeurons;
	float* outputNeurons;

	float* wInputHidden;
	float* wHiddenOutput;

	float get_set_mse(thrust::host_vector<FeatureVector*> set);
	float get_set_accuracy(thrust::host_vector<FeatureVector*> set);
	void feed_forward(float *input);
	int clamp_output(float a);

	float get_ih_weight(int i, int h);
	float get_ho_weight(int h, int o);
	void set_ih_weight(int i, int h, float w);
	void set_ho_weight(int h, int o, float w);

	void set_learning_rate(int l);
	void train_on_file(std::string fname, int max_epochs, int epochs_btwn_reports,
			float desired_error);
	void save(std::string fname);
	void print_network();

private:

	void init_weights();
	float sum_sqrs_error(float* target, float* output);
	inline float activation(float x) {
		//return 2*(1.0 / (1 + exp(-x))) - 1;
		return 1.0 / (1 + exp(-x));
	}
	float* feed_forward_input(float *input);

	float get_random_range(float min, float max);


};

#endif /* Net_H_ */
