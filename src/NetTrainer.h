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
const bool CPU_USE_BATCH = false;


class NetTrainer {
public:

	NetTrainer(Net *net);
	~NetTrainer();

	void set_learning_rate(float lr);
	void set_momentum(float m);

	void set_training_params(float lr, float m, bool b);
	void set_max_epochs(int max_epochs);
	void set_desired_accuracy(float acc);
	void set_stopping_conds(int me, float acc);
	void train_net(TrainingDataSet *tset);

	void backprop(float* targets);
	void rprop(float* targets);
	void update_weights();

	void run_training_epoch(TrainingDataSet *tset);

private:

	Net *net;

	long epoch;
	long max_epochs;

	bool batching;
	float delta_min, delta_max;

	clock_t start, stop;

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
	float** rpropDeltaInputHidden;
	float** rpropDeltaHiddenOutput;
	float** rpropGradientInputHidden;
	float** rpropGradientHiddenOutput;
	float** rpropLastGradientInputHidden;
	float** rpropLastGradientHiddenOutput;

	float** last_wt_update_input_hidden;
	float** last_wt_update_hidden_output;

	float* hiddenErrorGradients;
	float* outputErrorGradients;

	float d_max, d_min, rate_plus, rate_minus;

	inline float get_output_error_gradient(float target, float output) {
		return output * (1 - output) * (target - output);
	}
	inline int sign();

	float get_hidden_error_gradient(int j);
	//void run_training_epoch(TrainingDataSet *tset);

	float update_weight_rpropp(float** gradients, float** last_gradients,
			float** deltas, float** last_wt_changes, int i, int j);

	float min(float a, float b);
	float max(float a, float b);

};

#endif /* NETTRAINER_H_ */
