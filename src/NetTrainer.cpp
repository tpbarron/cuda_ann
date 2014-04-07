/*
 * NetTrainer.cpp
 *
 *  Created on: Dec 16, 2013
 *      Author: trevor
 */

#include <iostream>
#include <math.h>

#include "NetTrainer.h"

NetTrainer::NetTrainer(Net *net) {

	NetTrainer::net = net;
	NetTrainer::max_epochs = CPU_MAX_EPOCHS;
	NetTrainer::l_rate = CPU_LEARNING_RATE;
	NetTrainer::momentum = CPU_MOMENTUM;
	NetTrainer::desired_acc = CPU_DESIRED_ACCURACY;
	NetTrainer::batching = CPU_USE_BATCH;

	epoch = 0;
	trainingSetAccuracy = 0;
	validationSetAccuracy = 0;
	generalizationSetAccuracy = 0;
	trainingSetMSE = 0;
	validationSetMSE = 0;
	generalizationSetMSE = 0;

	d_min = .000001;
	d_max = 50;
	rate_minus = .5;
	rate_plus = 1.2;

	delta_min = -.1;
	delta_max = .1;

	start = 0;
	stop = 0;

	//create delta arrays, include biasm 	//rprop deltas
	NetTrainer::deltaInputHidden = new float*[net->n_input+1];
	NetTrainer::rpropDeltaInputHidden = new float*[net->n_input+1];
	for (int i = 0; i <= net->n_input; ++i) {
		NetTrainer::deltaInputHidden[i] = new float[net->n_hidden];
		NetTrainer::rpropDeltaInputHidden[i] = new float[net->n_hidden];
		for (int j = 0; j < net->n_hidden; ++j) {
			NetTrainer::deltaInputHidden[i][j] = 0;
			NetTrainer::rpropDeltaInputHidden[i][j] = 0;
		}
	}

	NetTrainer::deltaHiddenOutput = new float*[net->n_hidden+1];
	NetTrainer::rpropDeltaHiddenOutput = new float*[net->n_hidden+1];
	for (int i = 0; i <= net->n_hidden; ++i) {
		NetTrainer::deltaHiddenOutput[i] = new float[net->n_output];
		NetTrainer::rpropDeltaHiddenOutput[i] = new float[net->n_output];
		for (int j = 0; j < net->n_output; ++j) {
			NetTrainer::deltaHiddenOutput[i][j] = 0;
			NetTrainer::rpropDeltaHiddenOutput[i][j] = 0;
		}
	}

	//error gradients
	NetTrainer::hiddenErrorGradients = new float[net->n_hidden+1];
	for (int i = 0; i <= net->n_hidden; ++i)
		NetTrainer::hiddenErrorGradients[i] = 0;
	NetTrainer::outputErrorGradients = new float[net->n_output+1];
	for (int i = 0; i <= net->n_output; ++i)
		NetTrainer::outputErrorGradients[i] = 0;
}

NetTrainer::~NetTrainer() {
	//delete deltas
	for (int i = 0; i <= net->n_input; ++i) {
		delete[] NetTrainer::deltaInputHidden[i];
		delete[] NetTrainer::rpropDeltaInputHidden[i];
	}
	delete[] NetTrainer::deltaInputHidden;
	delete[] NetTrainer::rpropDeltaInputHidden;

	for (int i = 0; i <= net->n_hidden; ++i) {
		delete[] NetTrainer::deltaHiddenOutput[i];
		delete[] NetTrainer::rpropDeltaHiddenOutput[i];
	}
	delete[] NetTrainer::deltaHiddenOutput;
	delete[] NetTrainer::rpropDeltaHiddenOutput;

	//error gradients
	delete[] NetTrainer::hiddenErrorGradients;
	delete[] NetTrainer::outputErrorGradients;
}

/*
 * ------------ public --------------
 */

void NetTrainer::set_learning_rate(float lr) {
	l_rate = lr;
}

void NetTrainer::set_momentum(float m) {
	momentum = m;
}

void NetTrainer::set_training_params(float lr, float m, bool b) {
	l_rate = lr;
	momentum = m;
	batching = b;
}

void NetTrainer::set_max_epochs(int me) {
	max_epochs = me;
}

void NetTrainer::set_desired_accuracy(float acc) {
	desired_acc = acc;
}

void NetTrainer::set_stopping_conds(int me, float acc) {
	max_epochs = me;
	desired_acc = acc;
}

void NetTrainer::train_net(TrainingDataSet *tset) {
	std::cout << std::endl << " Neural Network Training Starting: " << std::endl
			<< "==========================================================================" << std::endl
			<< " LR: " << l_rate << ", Momentum: " << momentum << ", Max Epochs: " << max_epochs << std::endl
			<< " " << net->n_input << " Input Neurons, " << net->n_hidden << " Hidden Neurons, " << net->n_output << " Output Neurons" << std::endl
			<< "==========================================================================" << std::endl << std::endl;

	//reset epoch and log counters
	epoch = 0;

	//train network using training dataset for training and generalization dataset for testing
	while ((trainingSetAccuracy < desired_acc || generalizationSetAccuracy < desired_acc) && epoch < max_epochs) {
		//store previous accuracy
		float previousTAccuracy = trainingSetAccuracy;
		float previousGAccuracy = generalizationSetAccuracy;

		//use training set to train network
		run_training_epoch(tset);

		//get generalization set accuracy and MSE
		generalizationSetAccuracy = net->get_set_accuracy(tset->generalization_set, tset->n_generalization, tset->fpp);
		generalizationSetMSE = net->get_set_mse(tset->generalization_set, tset->n_generalization, tset->fpp);

		//print out change in training /generalization accuracy (only if a change is greater than a percent)
		if (ceil(previousTAccuracy) != ceil(trainingSetAccuracy) || ceil(previousGAccuracy) != ceil(generalizationSetAccuracy)) {
			std::cout << "Epoch: " << epoch;
			std::cout << "; test set acc:" << trainingSetAccuracy << "%, MSE: " << trainingSetMSE;
			std::cout << ";\tgset acc:" << generalizationSetAccuracy << "%, MSE: " << generalizationSetMSE << std::endl;
		}

		if (batching)
			update_weights();

		//once training set is complete increment epoch
		epoch++;

	}//end while

	//get validation set accuracy and MSE
	validationSetAccuracy = net->get_set_accuracy(tset->validation_set, tset->n_validation, tset->fpp);
	validationSetMSE = net->get_set_mse(tset->validation_set, tset->n_validation, tset->fpp);

	//out validation accuracy and MSE
	std::cout << std::endl << "Training complete. Elapsed epochs: " << epoch << std::endl;
	std::cout << "\tValidation set accuracy: " << validationSetAccuracy << std::endl;
	std::cout << "\tValidation set MSE: " << validationSetMSE << std::endl << std::endl;
}


/*
 * ------------- private -----------------
 */


void NetTrainer::run_training_epoch(TrainingDataSet *tset) {
	start = clock();
	//incorrect patterns
	int incorrectPatterns = 0;
	float mse = 0;

	//for every training pattern
	for (int tp = 0; tp < tset->n_training; tp++) {
		//feed inputs through network and backpropagate errors

		float* input = &(tset->training_set[tp*tset->fpp]);
		float* target = &(tset->training_set[tp*tset->fpp+tset->n_input+1]);
		net->feed_forward(input);
		//backprop(target);
		rprop(target);

		//pattern correct flag
		bool patternCorrect = true;

		//check all outputs from neural network against desired values
		for (int k = 0; k < net->n_output; k++) {
			//pattern incorrect if desired and output differ
			//std::cout << "orig: " << net->outputNeurons[k] << ", clamped: " << net->clamp_output(net->outputNeurons[k]) << ", target: " << feature_vecs[tp]->target[k] << std::endl;
			if (net->clamp_output(net->outputNeurons[k]) != net->clamp_output(target[k]))
				patternCorrect = false;

			//calculate MSE
			mse += pow((net->outputNeurons[k] - target[k]), 2);
		}

		//if pattern is incorrect add to incorrect count
		if (!patternCorrect)
			incorrectPatterns++;

	}//end for

	//update training accuracy and MSE
	trainingSetAccuracy = 100 - ((float)incorrectPatterns/tset->n_training * 100);
	trainingSetMSE = mse / (net->n_output * tset->n_training);
	stop = clock();
	std::cout << "Epoch time: " << ((float)stop-start)/CLOCKS_PER_SEC << std::endl;
}

//float NetTrainer::get_output_error_gradient(float target, float output) {
//	return output * (1 - output) * (target - output);
//}

float NetTrainer::get_hidden_error_gradient(int j) {
	//get sum of hidden->output weights * output error gradients
	float weightedSum = 0;
	for (int k = 0; k < net->n_output; ++k)
		weightedSum += net->get_ho_weight(j, k) * outputErrorGradients[k];

	//return error gradient
	return net->hiddenNeurons[j] * (1 - net->hiddenNeurons[j]) * weightedSum;
}

void NetTrainer::backprop(float* targets) {
	//modify deltas between hidden and output layers
	for (int k = 0; k < net->n_output; ++k) {
		//std::cout << "targets["<<k<<"]="<<targets[k] << std::endl;
		//get error gradient for every output node
		outputErrorGradients[k] = NetTrainer::get_output_error_gradient(targets[k], net->outputNeurons[k]);

		//for all nodes in hidden layer and bias neuron
		for (int j = 0; j <= net->n_hidden; ++j) {
			//calculate change in weight
			deltaHiddenOutput[j][k] = l_rate * net->hiddenNeurons[j] * outputErrorGradients[k] + momentum * deltaHiddenOutput[j][k];
		}
	}

	/* --- prints ------ */
	/*std::cout << "output error gradients: ";
	for (int k = 0; k < net->n_output; ++k) {
		std::cout << "[g" << k << ": " << outputErrorGradients[k] << "], ";
	}
	std::cout << std::endl;

	std::cout << "deltas hidden output: ";
	for (int k = 0; k < net->n_output; ++k) {
		for (int j = 0; j <= net->n_hidden; ++j) {
			std::cout << "[d" << j << k << ": " << deltaHiddenOutput[j][k] << "], ";
		}
	}
	std::cout << std::endl;*/


	//modify deltas between input and hidden layers
	for (int j = 0; j < net->n_hidden; ++j) {
		//get error gradient for every hidden node
		hiddenErrorGradients[j] = get_hidden_error_gradient(j);
		//for all nodes in input layer and bias neuron
		for (int i = 0; i <= net->n_input; ++i) {
			//calculate change in weight
			deltaInputHidden[i][j] = l_rate * net->inputNeurons[i] * hiddenErrorGradients[j] + momentum * deltaInputHidden[i][j];
		}
	}

	/* --- prints ------ */
	/*std::cout << "hidden error gradients: ";
	for (int k = 0; k < net->n_hidden; ++k) {
		std::cout << "[g" << k << ": " << hiddenErrorGradients[k] << "], ";
	}
	std::cout << std::endl;

	std::cout << "deltas input hidden: ";
	for (int k = 0; k < net->n_hidden; ++k) {
		for (int j = 0; j <= net->n_input; ++j) {
			std::cout << "[d" << j << k << ": " << deltaInputHidden[j][k] << "], ";
		}
	}
	std::cout << std::endl;*/


	//stochastic learning update the weights immediately
	if (!batching)
		update_weights();
}


void NetTrainer::rprop(float* targets) {
	//modify deltas between hidden and output layers
	for (int k = 0; k < net->n_output; ++k) {
		//std::cout << "targets["<<k<<"]="<<targets[k] << std::endl;
		//get error gradient for every output node
		float prevOutputErrorGradient = outputErrorGradients[k];
		outputErrorGradients[k] = NetTrainer::get_output_error_gradient(targets[k], net->outputNeurons[k]);

		float v = prevOutputErrorGradient * outputErrorGradients[k];
		std::cout << "v out = " << v << " ";
		int sign = (outputErrorGradients[k] > 0) - (outputErrorGradients[k] < 0);
		//for all nodes in hidden layer and bias neuron
		for (int j = 0; j <= net->n_hidden; ++j) {
			if (v > 0.00001) {
				rpropDeltaHiddenOutput[j][k] = std::min(rpropDeltaHiddenOutput[j][k]*rate_plus, d_max);
			} else if (v < -0.00001) {
				rpropDeltaHiddenOutput[j][k] = std::max(rpropDeltaHiddenOutput[j][k]*rate_minus, d_min);
			} else {
				rpropDeltaHiddenOutput[j][k] = outputErrorGradients[k];
			}
			net->set_ho_weight(j, k, net->get_ho_weight(j, k) - sign * rpropDeltaHiddenOutput[j][k]);
		}
		/*for (int j = 0; j <= net->n_hidden; ++j) {
			if (v > 0.00001) {
				rpropDeltaHiddenOutput[j][k] = std::min(rpropDeltaHiddenOutput[j][k]*rate_plus, d_max);
				deltaHiddenOutput[j][k] = -1*sign*rpropDeltaHiddenOutput[j][k];
				net->set_ho_weight(j, k, net->get_ho_weight(j, k)+deltaHiddenOutput[j][k]);
			} else if (v < -0.00001) {
				rpropDeltaHiddenOutput[j][k] = std::max(rpropDeltaHiddenOutput[j][k]*rate_minus, d_min);
				net->set_ho_weight(j, k, net->get_ho_weight(j, k)-deltaHiddenOutput[j][k]);
				outputErrorGradients[k] = 0;
			} else {
				deltaHiddenOutput[j][k] = -1*sign*rpropDeltaHiddenOutput[j][k];
				net->set_ho_weight(j, k, net->get_ho_weight(j, k)+deltaHiddenOutput[j][k]);
			}
		}*/
	}
	std::cout << std::endl;
	/* --- prints ------ */
	/*std::cout << "output error gradients: ";
	for (int k = 0; k < net->n_output; ++k) {
		std::cout << "[g" << k << ": " << outputErrorGradients[k] << "], ";
	}
	std::cout << std::endl;

	std::cout << "deltas hidden output: ";
	for (int k = 0; k < net->n_output; ++k) {
		for (int j = 0; j <= net->n_hidden; ++j) {
			std::cout << "[d" << j << k << ": " << deltaHiddenOutput[j][k] << "], ";
		}
	}
	std::cout << std::endl;*/


	//modify deltas between input and hidden layers
	for (int j = 0; j < net->n_hidden; ++j) {
		//get error gradient for every hidden node
		//for all nodes in input layer and bias neuron
		float prevHiddenErrorGradient = hiddenErrorGradients[j];
		hiddenErrorGradients[j] = get_hidden_error_gradient(j);

		float v = prevHiddenErrorGradient * hiddenErrorGradients[j];
		std::cout << "v hid = " << v << " ";
		int sign = (hiddenErrorGradients[j] > 0) - (hiddenErrorGradients[j] < 0);
		for (int i = 0; i <= net->n_input; ++i) {
			//calculate change in weight
			if (v > 0.00001) {
				rpropDeltaInputHidden[i][j] = std::min(rpropDeltaInputHidden[i][j]*rate_plus, d_max);
			} else if (v < -0.00001) {
				rpropDeltaInputHidden[i][j] = std::max(rpropDeltaInputHidden[i][j]*rate_minus, d_min);
			} else {
				rpropDeltaInputHidden[i][j] = hiddenErrorGradients[j];
			}
			net->set_ih_weight(i, j, net->get_ih_weight(i, j) - sign * rpropDeltaInputHidden[i][j]);
		}

		/*for (int i = 0; i <= net->n_input; ++i) {
			//calculate change in weight
			if (v > 0.00001) {
				rpropDeltaInputHidden[i][j] = std::min(rpropDeltaInputHidden[i][j]*rate_plus, d_max);
				deltaInputHidden[i][j] = -1*sign*rpropDeltaInputHidden[i][j];
				net->set_ih_weight(i, j, net->get_ih_weight(i, j)+deltaInputHidden[i][j]);
			} else if (v < -0.00001) {
				rpropDeltaInputHidden[i][j] = std::max(rpropDeltaInputHidden[i][j]*rate_minus, d_min);
				net->set_ih_weight(i, j, net->get_ih_weight(i, j)-deltaInputHidden[i][j]);
				hiddenErrorGradients[j] = 0;
			} else {
				deltaInputHidden[i][j] = -1*sign*rpropDeltaInputHidden[i][j];
				net->set_ih_weight(i, j, net->get_ih_weight(i, j)+deltaInputHidden[i][j]);
			}
		}*/
	}
	std::cout << std::endl;

	/* --- prints ------ */
	/*std::cout << "hidden error gradients: ";
	for (int k = 0; k < net->n_hidden; ++k) {
		std::cout << "[g" << k << ": " << hiddenErrorGradients[k] << "], ";
	}
	std::cout << std::endl;

	std::cout << "deltas input hidden: ";
	for (int k = 0; k < net->n_hidden; ++k) {
		for (int j = 0; j <= net->n_input; ++j) {
			std::cout << "[d" << j << k << ": " << deltaInputHidden[j][k] << "], ";
		}
	}
	std::cout << std::endl;*/


	//stochastic learning update the weights immediately
	//update_weights();
}

void NetTrainer::update_weights() {
	//input -> hidden weights
	for (int i = 0; i <= net->n_input; i++) {
		for (int j = 0; j < net->n_hidden; j++) {
			//std::cout << "h_weight(" << i << ", " << j << ") = " << net->get_ih_weight(i,j) << ", h_deltas(" << i << ", " << j << ") = " << deltaInputHidden[i][j] << std::endl;
			float d = deltaInputHidden[i][j];
			if (batching) {
				if (d > delta_max) {
					net->set_ih_weight(i, j, net->get_ih_weight(i, j) + delta_max);
				} else if (d < delta_min) {
					net->set_ih_weight(i, j, net->get_ih_weight(i, j) + delta_min);
				} else {
					net->set_ih_weight(i, j, net->get_ih_weight(i, j) + d);
				}
				deltaInputHidden[i][j] = 0;
			} else {
				net->set_ih_weight(i, j, net->get_ih_weight(i, j) + d);
			}
		}
	}

	//hidden -> output weights
	for (int j = 0; j <= net->n_hidden; j++) {
		for (int k = 0; k < net->n_output; k++) {
			float d = deltaHiddenOutput[j][k];
			if (batching) {
				if (d > delta_max) {
					net->set_ho_weight(j, k, net->get_ho_weight(j, k) + delta_max);
				} else if (d < delta_min) {
					net->set_ho_weight(j, k, net->get_ho_weight(j, k) + delta_min);
				} else {
					net->set_ho_weight(j, k, net->get_ho_weight(j, k) + d);
				}
				deltaHiddenOutput[j][k] = 0;
			} else {
				net->set_ho_weight(j, k, net->get_ho_weight(j, k) + deltaHiddenOutput[j][k]);
			}
		}
	}
}

