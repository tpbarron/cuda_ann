/*
 * NetData.cpp
 *
 *  Created on: Dec 15, 2013
 *      Author: trevor
 */

#include "NetData.h"
#include <fstream>
#include <algorithm>
#include <math.h>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

NetData::NetData() {
	n_inputs = 0;
	n_targets = 0;
	training_data_end_index = 0;
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

		//read info line
		getline(in, line);
		std::vector<std::string> res;
		boost::split(res, line, boost::is_space());
		int num_vecs = boost::lexical_cast<int>(res[0]);
		n_inputs = boost::lexical_cast<int>(res[1]);
		n_targets = boost::lexical_cast<int>(res[2]);

		std::cout << line << std::endl;
		for (int i = 0; i < num_vecs; ++i) {

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
		training_data_end_index = (int) (1.0 * data.size());

		//training_data_end_index = (int) (0.6 * data.size());
		//int gSize = (int) (ceil(0.2 * data.size()));
		//int vSize = (int) (data.size() - training_data_end_index - gSize);

		//training set
		for (int i = 0; i < training_data_end_index; ++i)
			tset.training_set.push_back(data[i]);

		//generalization set
		/*for (int i = training_data_end_index; i < training_data_end_index + gSize; ++i)
			tset.generalization_set.push_back(data[i]);

		//validation set
		for (int i = training_data_end_index + gSize; i < (int)data.size(); i++)
			tset.validation_set.push_back(data[i]);*/

		tset.set_size(data.size());

		//print success
		std::cout << "Data file: " << fname << std::endl;
		std::cout << "Read complete: " << data.size() << " patterns loaded"  << std::endl;

		//close file
		in.close();

		return true;
	} else {
		std::cout << "Could not open data file...\n";
	}
	return false;
}

TrainingDataSet* NetData::get_training_dataset() {
	return &tset;
}

size_t NetData::num_feature_vecs() {
	return data.size();
}

int NetData::num_inputs() {
	return n_inputs;
}

int NetData::num_targets() {
	return n_targets;
}

void NetData::print_loaded_patterns() {
	for (unsigned int i = 0; i < tset.training_set.size(); ++i) {
		FeatureVector *tp = tset.training_set[i];
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

/*
 * ----------- private --------------
 */
