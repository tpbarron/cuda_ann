/*
 * ANN.cpp
 *
 *  Created on: Dec 9, 2013
 *      Author: trevor
 */

#include <math.h>
#include <stdlib.h>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "Net.h"


Net::Net(int ni, int nh, int no) {

	Net::n_input = ni;
	Net::n_hidden = nh;
	Net::n_output = no;

	//allocate neuron arrays
	Net::inputNeurons = new float[ni + 1];
	Net::hiddenNeurons = new float[nh + 1];
	Net::outputNeurons = new float[no];

	//Initialize nodes
	for (int i = 0; i < ni; ++i)
		Net::inputNeurons[i] = 0;
	//bias neuron, the last neuron in the lists
	Net::inputNeurons[ni] = 1;


	for (int i = 0; i < nh; ++i)
		Net::hiddenNeurons[i] = 0;
	//bias neuron
	Net::hiddenNeurons[nh] = 1;

	for (int i = 0; i < no; ++i)
		Net::outputNeurons[i] = 0;

	//initialize weight lists including bias node
	Net::wInputHidden = new float[(ni+1)*(nh)];
	for (int i = 0; i < (ni+1)*(nh); ++i)
		Net::wInputHidden[i] = 0;

	Net::wHiddenOutput = new float[(nh+1)*(no)];
	for (int i = 0; i < (nh+1)*(no); ++i)
		Net::wHiddenOutput[i] = 0;


	init_weights();
}

Net::~Net() {
	//delete neurons
	delete[] Net::inputNeurons;
	delete[] Net::hiddenNeurons;
	delete[] Net::outputNeurons;

	//delete weights
	delete[] Net::wInputHidden;
	delete[] Net::wHiddenOutput;
}

/*
 * ------------- public -----------------------
 */


/*
 * expects a file of normalized data
 *
 * examples size_input size_output
 * input
 * output
 * input
 * output
 * ....
 */
void Net::train_on_file(std::string fname, int max_epochs, int epochs_btwn_reports,
		float desired_error) {


}


void Net::save(std::string fname) {

}


void Net::print_network() {
	std::cout << "input: ";
	for (int i = 0; i <= Net::n_input; ++i) {
		std::cout << "[" << i << ": " << Net::inputNeurons[i] << "], ";
	}
	std::cout << std::endl;
	for (int i = 0; i <= Net::n_input; ++i) {
		for (int j = 0; j < Net::n_hidden; ++j) {
			std::cout << "[w" << i << j << ": " << Net::get_ih_weight(i, j) << "], ";
		}
	}
	std::cout << std::endl;

	std::cout << "hidden: ";
	for (int i = 0; i <= Net::n_hidden; ++i) {
		std::cout << "[" << i << ": " << Net::hiddenNeurons[i] << "], ";
	}
	std::cout << std::endl;
	for (int i = 0; i <= Net::n_hidden; ++i) {
		for (int j = 0; j < Net::n_output; ++j) {
			std::cout << "[w" << i << j << ": " << Net::get_ho_weight(i, j) << "], ";
		}
	}
	std::cout << std::endl;

	std::cout << "output: ";
	for (int i = 0; i < Net::n_output; ++i) {
		std::cout << "[" << i << ": " << Net::outputNeurons[i] << "], ";
	}

	std::cout << std::endl << std::endl;
}


/*
 * Note: This weight storage format doesn't work as well with
 * non-fully connected networks. But it is easier in CUDA to work with
 * 1D arrays.
 */
float Net::get_ih_weight(int i, int h) {
	return Net::wInputHidden[Net::n_hidden*i + h];
}

float Net::get_ho_weight(int h, int o) {
	return Net::wHiddenOutput[Net::n_output*h + o];
}

void Net::set_ih_weight(int i, int h, float w) {
	Net::wInputHidden[Net::n_hidden*i + h] = w;
}

void Net::set_ho_weight(int h, int o, float w) {
	Net::wHiddenOutput[Net::n_output*h + o] = w;
}



/*
 * -------------- private ----------------------
 */

float Net::get_random_range(float min, float max) {
	float r = (float)rand() / (float)RAND_MAX;
	return min + r * (max - min);
}



/*
 * http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
 */
void Net::init_weights() {
	std::cout << "initializing weights\n";
	float rh = 1.0 / sqrt((float)Net::n_input);
	float ro = 1.0 / sqrt((float)Net::n_hidden);

	for (int i = 0; i < (n_input+1)*(n_hidden); ++i) {
		Net::wInputHidden[i] = Net::get_random_range(-rh, rh);
		//std::cout << "Net::wInputHidden["<<i <<"] = " << Net::wInputHidden[i] << std::endl;
	}

	for (int i = 0; i < (n_hidden+1)*(n_output); ++i)
		Net::wHiddenOutput[i] = Net::get_random_range(-ro, ro);
}


float Net::get_set_mse(thrust::host_vector<FeatureVector*> set) {
	float mse = 0;
	//for every training input array
	for ( int tp = 0; tp < (int) set.size(); tp++) {
		//feed inputs through network and backpropagate errors
		feed_forward(set[tp]->input);

		//check all outputs against desired output values
		for ( int k = 0; k < n_output; k++ ) {
			//sum all the MSEs together
			mse += pow((outputNeurons[k] - set[tp]->target[k]), 2);
		}
	}

	return mse / (n_output * set.size());
}


float Net::get_set_accuracy(thrust::host_vector<FeatureVector*> set) {
	float incorrectResults = 0;

	//for every training input array
	for (int tp = 0; tp < (int) set.size(); tp++) {
		//feed inputs through network and backpropagate errors
		feed_forward(set[tp]->input);

		//correct pattern flag
		bool correctResult = true;

		//check all outputs against desired output values
		for (int k = 0; k < n_output; ++k) {
			//set flag to false if desired and output differ
			if (clamp_output(outputNeurons[k]) != set[tp]->target[k])
				correctResult = false;
		}

		//inc training error for a incorrect result
		if (!correctResult)
			incorrectResults++;

	}

	//calculate error and return as percentage
	return 100 - (incorrectResults/set.size() * 100);
}


/*
 * sigmoid activation function
 */
//in header
//inline float Net::activation(float x) {
	//return 1.0 / (1 + exp(-x));
//}


int Net::clamp_output(float a) {
	if (a < 0.1) {
		return 0;
	} else if (a > 0.9) {
		return 1;
	} else {
		return -1;
	}
}

/*int Net::clamp_output(float a) {
	if (a < -0.8) {
		return -1;
	} else if (a > 0.8) {
		return 1;
	} else {
		return 0;
	}
}*/


float* Net::feed_forward_input(float *input) {
	feed_forward(input);
	float *results = new float[Net::n_output];
	for (int i = 0; i < Net::n_output; ++i)
		results[i] = Net::outputNeurons[i];
	return results;
}

void Net::feed_forward(float *input) {
	//set input neurons
	for (int i = 0; i < Net::n_input; ++i) {
		Net::inputNeurons[i] = input[i];
	}
	//calc hidden layer vals
	for (int j = 0; j < Net::n_hidden; ++j) {
		Net::hiddenNeurons[j] = 0;

		//weighted sum of pattern and bias neuron
		for (int i = 0; i <= Net::n_input; ++i) {
			//std::cout << "Net::get_ih_weights(" << i << ", " << j << ") = " << Net::get_ih_weight(i, j) << std::endl;
			Net::hiddenNeurons[j] += Net::inputNeurons[i] * Net::get_ih_weight(i, j); //Net::wInputHidden[i][j];
		}

		Net::hiddenNeurons[j] = activation(Net::hiddenNeurons[j]);
	}

	//calc output layer
	for (int k = 0; k < Net::n_output; ++k) {
		Net::outputNeurons[k] = 0;

		//weighted sum
		for (int j = 0; j <= Net::n_hidden; ++j) {
			//std::cout << "[" << j << "," << k << "], (" << Net::hiddenNeurons[j] << ")*(" << Net::wHiddenOutput[j][k] << ")\n";
			Net::outputNeurons[k] += Net::hiddenNeurons[j] * Net::get_ho_weight(j, k); //Net::wHiddenOutput[j][k];
		}

		//std::cout << Net::outputNeurons[k] << std::endl;
		Net::outputNeurons[k] = activation(Net::outputNeurons[k]);
	}
}


