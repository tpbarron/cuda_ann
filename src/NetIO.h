/*
 * NetIO.h
 *
 *  Created on: Mar 23, 2014
 *      Author: trevor
 */

#ifndef NETIO_H_
#define NETIO_H_

#include "GPUNet.h"
#include <string>

class NetIO {
public:
	NetIO();
	~NetIO();

	bool read_net(std::string fname);
	bool write_net(std::string fname);


	long epoch, max_epochs;
	GPUNet::NetworkStructure net_type;

	int n_input, n_hidden, n_output;

	float l_rate, momentum, desired_acc;
	float trainingSetAccuracy, generalizationSetAccuracy, validationSetAccuracy;
	float trainingSetMSE, generalizationSetMSE, validationSetMSE;

	float *ih_weights, *ho_weights;

private:
	int get_next_int(std::ifstream &in);
	long get_next_long(std::ifstream &in);
	float get_next_float(std::ifstream &in);
	float* get_next_list(std::ifstream &in);
};

#endif /* NETIO_H_ */
