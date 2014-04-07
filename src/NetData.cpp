/*
 * NetData.cpp
 *
 *  Created on: Dec 15, 2013
 *      Author: trevor
 */

#include <algorithm>
#include <fstream>
#include <math.h>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "NetData.h"

NetData::NetData(float tpct) {
	t_set_pct = tpct;
	n_patterns = 0;
	n_inputs = 0;
	n_targets = 0;
}

NetData::~NetData() {
	for (unsigned int i = 0; i < data.size(); ++i)
		delete data[i];
	data.clear();
}


/*
 * ---------- public ------------
 */
bool NetData::load_file(std::string fname) {
	std::ifstream in(fname.c_str());
	if (in.is_open()) {
		std::string line;

		//TODO: is there a way to load these straight into a pointer array and still do a random shuffle?

		//read info line
		getline(in, line);
		std::vector<std::string> res;
		boost::split(res, line, boost::is_space());
		n_patterns = boost::lexical_cast<unsigned int>(res[0]);
		n_inputs = boost::lexical_cast<unsigned int>(res[1]);
		n_targets = boost::lexical_cast<unsigned int>(res[2]);

		std::cout << line << std::endl;
		for (int i = 0; i < n_patterns; ++i) {
			float *inputs = new float[n_inputs];
			float *targets = new float[n_targets];

			//inputs
			getline(in, line);
			boost::trim(line);
			boost::split(res, line, boost::is_space());
			for (unsigned int j = 0; j < res.size(); ++j) {
				float f = boost::lexical_cast<float>(res[j]);
				inputs[j] = f;
			}

			//targets
			getline(in, line);
			boost::trim(line);
			boost::split(res, line, boost::is_space());
			for (unsigned int j = 0; j < res.size(); ++j) {
				float f = boost::lexical_cast<float>(res[j]);
				targets[j] = f;
			}

			data.push_back(new FeatureVector(inputs, targets));
		}

		random_shuffle(data.begin(), data.end());

		//split data set
		int t_size = (int) (t_set_pct * data.size());
		//int g_size = (int) (ceil(0.2 * data.size()));
		int v_size = (int) (data.size() - t_size);
		//int t_size = (int) (1.0 * data.size()), g_size =0, v_size = 0;
		tset.n_input = n_inputs;
		tset.n_target = n_targets;
		tset.fpp = tset.floats_per_pattern();

		//allocate memory in data partitions and copy memory
		int floats_per_pattern = tset.floats_per_pattern();
		int num_floats_training = t_size*floats_per_pattern;
		//int num_floats_generalization = g_size*floats_per_pattern;
		int num_floats_validation = v_size*floats_per_pattern;


		tset.training_set = new float[num_floats_training];
		//tset.generalization_set = new float[num_floats_generalization];
		tset.validation_set = new float[num_floats_validation];
		tset.n_patterns = data.size();
		tset.n_training = t_size;
		//tset.n_generalization = g_size;
		tset.n_validation = v_size;

		//training set
		for (int i = 0; i < t_size; ++i) {
			add_to_dataset(tset.training_set, floats_per_pattern, i, i);
		}

		//generalization set
		//for (int i = 0; i < g_size; ++i) {
		//	add_to_dataset(tset.generalization_set, floats_per_pattern, i);
		//}

		//validation set
		for (int i = 0; i < v_size; i++) {
			add_to_dataset(tset.validation_set, floats_per_pattern, i, i+t_size);
		}

		//print success
		std::cout << "Data file: " << fname << std::endl;
		std::cout << "Read complete: " << data.size() << " patterns loaded"  << std::endl;
		std::cout << "Test set: " << tset.n_training << ", generalization set: " << tset.n_generalization << ", validation set: " << tset.n_validation << std::endl;

		//close file
		in.close();
		return true;
	} else {
		std::cout << "Could not open data file...\n";
	}
	return false;
}

void NetData::add_to_dataset(float* set, int floats_per_pattern, int p, int i) {
	//Add the ith pattern to dataset set at position p
	int pos;
	for (int j = 0; j < n_inputs; ++j) {
		set[p*floats_per_pattern+j] = data[i]->input[j];
	}
	set[p*floats_per_pattern+n_inputs] = 1; //set bias
	for (int k = 0; k < n_targets; ++k) {
		//std::cout << data[i]->target[k] << std::endl;
		set[p*floats_per_pattern+n_inputs+1+k] = data[i]->target[k];
	}
}

TrainingDataSet* NetData::get_training_dataset() {
	return &tset;
}

size_t NetData::num_feature_vecs() {
	return data.size();
}

unsigned int NetData::num_inputs() {
	return n_inputs;
}

unsigned int NetData::num_targets() {
	return n_targets;
}

void NetData::print_loaded_patterns() {
	for (unsigned int i = 0; i < tset.n_training; ++i) {
		FeatureVector *tp = data[i];
		for (int j = 0; j < n_inputs; ++j) {
			std::cout << (*tp).input[j] << " ";
		}
		std::cout << ", ";
		for (int k = 0; k < n_targets; ++k) {
			std::cout <<(*tp).target[k] << " ";
		}
		std::cout << std::endl;
	}
}

void NetData::print_loaded_patterns_flatted() {
	std::cout << "training set" << std::endl;
	for (unsigned int i = 0; i < tset.n_training; ++i) {
		int fpp = tset.floats_per_pattern();
		std::cout << "input: ";
		for(int j = 0; j < tset.n_input; j++) {
			int p = i*fpp+j;
			std::cout << tset.training_set[p] << " ";
		}
		std::cout << ", bias: " << tset.training_set[i*fpp+tset.n_input] << "\n";
		std::cout << "target: ";
		for(int j = 0; j < tset.n_target; j++) {
			int p = i*fpp+tset.n_input+1+j;
			std::cout << tset.training_set[p] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "validation set" << std::endl;
	for (unsigned int i = 0; i < tset.n_validation; ++i) {
		int fpp = tset.floats_per_pattern();
		std::cout << "input: ";
		for(int j = 0; j < tset.n_input; j++) {
			int p = i*fpp+j;
			std::cout << tset.validation_set[p] << " ";
		}
		std::cout << ", bias: " << tset.validation_set[i*fpp+tset.n_input] << "\n";
		std::cout << "target: ";
		for(int j = 0; j < tset.n_target; j++) {
			int p = i*fpp+tset.n_input+1+j;
			std::cout << tset.validation_set[p] << " ";
		}
		std::cout << std::endl;
	}
}

/*
 * ----------- private --------------
 */
