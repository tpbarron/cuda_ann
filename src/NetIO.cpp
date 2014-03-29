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

NetIO::NetIO() {
	epoch = 0, max_epochs = 0;
	net_type = GPUNet::STANDARD;

	n_input = 0, n_hidden = 0, n_output = 0;

	l_rate = 0, momentum = 0, desired_acc = 0;
	trainingSetAccuracy = 0, generalizationSetAccuracy = 0, validationSetAccuracy = 0;
	trainingSetMSE = 0, generalizationSetMSE = 0, validationSetMSE = 0;

	ih_weights = NULL, ho_weights = NULL;
}

NetIO::~NetIO() {
	delete[] ih_weights;
	delete[] ho_weights;
}

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

		ih_weights = get_next_list(in);
		ho_weights = get_next_list(in);

		std::cout << "NetIO: Closing file" <<std::endl;
		in.close();
	} else {
		std::cerr << "NetIO: Could not read net file!" << std::endl;
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
	std::cout << "NetIO: name=" << fname << std::endl;
	std::ofstream of(fname.c_str());
	std::cout << "NetIO: Is good: " << of.good() << std::endl;
	std::cout << "NetIO: ofstream initialized" <<std::endl;

	if (of.is_open()) {
		std::cout << "NetIO: file open" << std::endl;

		of << "num_epochs=" << epoch << std::endl;
		of << "max_epochs=" << max_epochs << std::endl;
		of << "net_type=" << net_type << std::endl;
		of << "num_layers=" << 3 << std::endl;
		of << "n_layer_0=" << n_input << std::endl;
		of << "n_layer_1=" << n_hidden << std::endl;
		of << "n_layer_2=" << n_output << std::endl;
		of << "l_rate=" << l_rate << std::endl;
		of << "momentum=" << momentum << std::endl;
		of << "desired_acc=" << desired_acc << std::endl;
		of << "tset_acc=" << trainingSetAccuracy << std::endl;
		of << "gset_acc=" << generalizationSetAccuracy << std::endl;
		of << "vset_acc=" << validationSetAccuracy << std::endl;
		of << "tset_mse=" << trainingSetMSE << std::endl;
		of << "gset_mse=" << generalizationSetMSE << std::endl;
		of << "vset_mse=" << validationSetMSE << std::endl;
		of << "weights_ih=";
		for (int i = 0, l = (n_input+1)*n_hidden; i < l; ++i) {
			of << ih_weights[i];
			if (i != l-1)
				of << ",";
		}
		of << std::endl;
		of << "weights_ho=";
		for (int i = 0, l = (n_hidden+1)*n_output; i < l; ++i) {
			of << ho_weights[i];
			if (i != l-1)
				of << ",";
		}

		of.flush();
		of.close();
		std::cout << "NetIO: Closed file" <<std::endl;
	} else {
		std::cerr << "NetIO: Could not write file!" << std::endl;
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
