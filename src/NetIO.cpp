/*
 * NetIO.cpp
 *
 *  Created on: Mar 23, 2014
 *      Author: trevor
 */

#include <fstream>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "NetIO.h"


NetIO::NetIO() {
	gnet = NULL;
}

NetIO::~NetIO() {}

bool NetIO::read_net(std::string fname) {
	std::ifstream in(fname.c_str());
	if (in.is_open()) {
		// num epochs
		int epoch = get_next_long(in);
		get_next_long(in);
		//gnet->max_epochs = get_next_long(in);

		if (get_next_string(in) == "STANDARD") {
			gnet->net_type = GPUNetSettings::STANDARD;
		} else {
			gnet->net_type = GPUNetSettings::GPU_ARCH_OPT;
		}
		//skip n_layers
		get_next_int(in);
		gnet->n_input = get_next_int(in);
		gnet->n_hidden = get_next_int(in);
		gnet->n_output = get_next_int(in);

		//both needed number of nodes per layer
		gnet->init_vars();
		gnet->alloc_dev_mem();

		gnet->epoch = epoch;
		gnet->l_rate = get_next_float(in);
		gnet->momentum = get_next_float(in);
		gnet->desired_acc = get_next_float(in);
		gnet->trainingSetAccuracy = get_next_float(in);
		gnet->generalizationSetAccuracy = get_next_float(in);
		gnet->validationSetAccuracy = get_next_float(in);
		gnet->trainingSetMSE = get_next_float(in);
		gnet->generalizationSetMSE = get_next_float(in);
		gnet->validationSetMSE = get_next_float(in);

		gnet->h_ih_weights = get_next_list(in);
		gnet->h_ho_weights = get_next_list(in);
		// get weights
		CUDA_CHECK_RETURN(cudaMemcpy(gnet->d_ih_weights, gnet->h_ih_weights, (gnet->n_input+1)*gnet->n_hidden*sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(gnet->d_ho_weights, gnet->h_ho_weights, (gnet->n_hidden+1)*gnet->n_output*sizeof(float), cudaMemcpyHostToDevice));

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
 */
bool NetIO::write_net(std::string fname) {
	std::ofstream of(fname.c_str());

	if (of.is_open()) {
		CUDA_CHECK_RETURN(cudaMemcpy(gnet->h_ih_weights, gnet->d_ih_weights, (gnet->n_input+1)*(gnet->n_hidden)*sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpy(gnet->h_ho_weights, gnet->d_ho_weights, (gnet->n_hidden+1)*(gnet->n_output)*sizeof(float), cudaMemcpyDeviceToHost));

		of << "num_epochs=" << gnet->epoch << std::endl;
		of << "max_epochs=" << gnet->max_epochs << std::endl;

		if (gnet->net_type == GPUNetSettings::STANDARD) {
			of << "net_type=" << "STANDARD" << std::endl;
			//std::cout << "Standard type" << std::endl;
		} else {
			of << "net_type=" << "GPU_ARCH_OPT" << std::endl;
			//std::cout << "Optimized type" << std::endl;
		}
		of << "num_layers=" << 3 << std::endl;
		of << "n_layer_0=" << gnet->n_input << std::endl;
		of << "n_layer_1=" << gnet->n_hidden << std::endl;
		of << "n_layer_2=" << gnet->n_output << std::endl;
		of << "l_rate=" << gnet->l_rate << std::endl;
		of << "momentum=" << gnet->momentum << std::endl;
		of << "desired_acc=" << gnet->desired_acc << std::endl;

		//if (gnet->trainingSetAccuracy == 0 || gnet->trainingSetMSE == 0) {
		//	std::cerr << "Incorrect stats. Exiting." << std::endl;
		//	exit(0);
		//}

		of << "tset_acc=" << gnet->trainingSetAccuracy << std::endl;
		of << "gset_acc=" << gnet->generalizationSetAccuracy << std::endl;
		of << "vset_acc=" << gnet->validationSetAccuracy << std::endl;
		of << "tset_mse=" << gnet->trainingSetMSE << std::endl;
		of << "gset_mse=" << gnet->generalizationSetMSE << std::endl;
		of << "vset_mse=" << gnet->validationSetMSE << std::endl;

		of << "weights_ih=";
		for (int i = 0, l = (gnet->n_input+1)*gnet->n_hidden; i < l; ++i) {
			of << gnet->h_ih_weights[i];
			if (i != l-1)
				of << ",";
		}
		of << std::endl;
		of << "weights_ho=";
		for (int i = 0, l = (gnet->n_hidden+1)*gnet->n_output; i < l; ++i) {
			of << gnet->h_ho_weights[i];
			if (i != l-1)
				of << ",";
		}

		of.flush();
		of.close();
	} else {
		std::cerr << "NetIO: Could not write file!" << std::endl;
		return false;
	}
	return true;
}


void NetIO::set_gnet(GPUNet *g) {
	gnet = g;
}

std::string NetIO::get_next_string(std::ifstream &in) {
	std::string line;
	std::getline(in, line);
	std::vector<std::string> res;
	boost::split(res, line, boost::is_any_of("="));
	return res[1];
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
