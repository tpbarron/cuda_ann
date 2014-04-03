/*
 * NetTrainer.h
 *
 *  Created on: Dec 16, 2013
 *      Author: trevor
 */

#ifndef NETTRAINER_H_
#define NETTRAINER_H_

#include <vector>
#include "Net.h"

//defaults
const float CPU_LEARNING_RATE = 0.7;
const float CPU_MOMENTUM = 0.9;
const long CPU_MAX_EPOCHS = 1500;
const int CPU_DESIRED_ACCURACY = 90;
const int CPU_DESIRED_MSE = 0.001;


class NetTrainer {
public:

	NetTrainer(Net *net);
	~NetTrainer();

	void set_learning_rate(float lr);
	void set_momentum(float m);

	void set_training_params(float lr, float m);
	void set_max_epochs(int max_epochs);
	void set_desired_accuracy(float acc);
	void set_stopping_conds(int me, float acc);
	void train_net(TrainingDataSet *tset);

	void backprop(float* targets);
	void update_weights();

private:

	Net *net;

	long epoch;
	long max_epochs;

	float l_rate;
	float momentum;

	float desired_acc;

	float trainingSetAccuracy;
	float validationSetAccuracy;
	float generalizationSetAccuracy;
	float trainingSetMSE;
	float validationSetMSE;
	float generalizationSetMSE;

	float** deltaInputHidden;
	float** deltaHiddenOutput;

	float* hiddenErrorGradients;
	float* outputErrorGradients;

	inline float get_output_error_gradient(float target, float output) {
		return output * (1 - output) * (target - output);
	}
	float get_hidden_error_gradient(int j);
	void run_training_epoch(TrainingDataSet *tset);

};

#endif /* NETTRAINER_H_ */
