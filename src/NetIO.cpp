/*
 * NetIO.cpp
 *
 *  Created on: Mar 23, 2014
 *      Author: trevor
 */

#include "NetIO.h"
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>

NetIO::NetIO() {}

NetIO::~NetIO() {}

bool NetIO::read_net(std::string fname) {
	std::ifstream in(fname.c_str());
	if (in.is_open()) {
		// num epochs
		epoch = get_next_long(in);
		max_epochs = get_next_long(in);
		net_type = (GPUNet::NetworkStructure)get_next_int(in);

		//skip n_layers
		get_next_int(in);
		n_input = get_next_int(in);
		n_hidden = get_next_int(in);
		n_output = get_next_int(in);

		l_rate = get_next_float(in);
		momentum = get_next_float(in);
		desired_acc = get_next_float(in);
		trainingSetAccuracy = get_next_float(in);
		generalizationSetAccuracy = get_next_float(in);
		validationSetAccuracy = get_next_float(in);
		trainingSetMSE = get_next_float(in);
		generalizationSetMSE = get_next_float(in);
		validationSetMSE = get_next_float(in);

		float *ih_weights = get_next_list(in);
		float *ho_weights = get_next_list(in);

		delete[] ih_weights;
		delete[] ho_weights;
	} else {
		std::cerr << "Could not read net file!" << std::endl;
		return false;
	}
	return true;
}


/*
 * transfer weights back to host
 * write important data (num_epochs, layers, nodes/layer, l_rate, momentum, max_epochs, desired_acc, current mse, current acc)
 *
 */
bool NetIO::write_net(std::string fname) {
	std::ofstream of(fname.c_str());
	if (of.is_open()) {
		of << "num_epochs=" << epoch << "\n";
		of << "max_epochs=" << max_epochs << "\n";
		of << "net_type=" << net_type << "\n";
		of << "num_layers=" << 3 << "\n";
		of << "n_layer_0=" << n_input << "\n";
		of << "n_layer_1=" << n_hidden << "\n";
		of << "n_layer_2=" << n_output << "\n";
		of << "l_rate=" << l_rate << "\n";
		of << "momentum=" << momentum << "\n";
		of << "desired_acc=" << desired_acc << "\n";
		of << "tset_acc=" << trainingSetAccuracy << "\n";
		of << "gset_acc=" << generalizationSetAccuracy << "\n";
		of << "vset_acc=" << validationSetAccuracy << "\n";
		of << "tset_mse=" << trainingSetMSE << "\n";
		of << "gset_mse=" << generalizationSetMSE << "\n";
		of << "vset_mse=" << validationSetMSE << "\n";
		of << "weights_ih=";
		for (int i = 0, l = (n_input+1)*n_hidden; i < l; ++i) {
			of << ih_weights[i];
			if (i != l-1)
				of << ",";
		}
		of << "\n";
		of << "weights_ho=";
		for (int i = 0, l = (n_hidden+1)*n_output; i < l; ++i) {
			of << ho_weights[i];
			if (i != l-1)
				of << ",";
		}

		of.flush();
		of.close();
	} else {
		std::cerr << "Could not write file!" << std::endl;
		return false;
	}
	return true;
}


int NetIO::get_next_int(std::ifstream &in) {
	std::string line;
	std::getline(in, line);
	std::vector<std::string> res;
	boost::split(res, line, boost::is_any_of("="));
	return boost::lexical_cast<int>(res[1]);
}

long NetIO::get_next_long(std::ifstream &in) {
	std::string line;
	std::getline(in, line);
	std::vector<std::string> res;
	boost::split(res, line, boost::is_any_of("="));
	return boost::lexical_cast<long>(res[1]);
}

float NetIO::get_next_float(std::ifstream &in) {
	std::string line;
	std::getline(in, line);
	std::vector<std::string> res;
	boost::split(res, line, boost::is_any_of("="));
	return boost::lexical_cast<float>(res[1]);
}

float* NetIO::get_next_list(std::ifstream &in) {
	std::string line;
	std::getline(in, line);
	std::vector<std::string> res;
	boost::split(res, line, boost::is_any_of("="));
	std::vector<std::string> list;
	boost::split(list, res[1], boost::is_any_of(", "));

	float *fl_list = new float[list.size()];
	//just overwrite random GPU values
	for (size_t i = 0; i < list.size(); ++i) {
		fl_list[i] = boost::lexical_cast<float>(list[i]);
	}
	return fl_list;
}
