/*
 * NetIO.h
 *
 * Created on: Mar 23, 2014
 * Author: trevor
 */

#ifndef NETIO_H_
#define NETIO_H_

#include <string>

#include "GPUNet.h"
#include "GPUNetSettings.h"


class GPUNet;
class NetIO {
public:
	NetIO();
	~NetIO();

	bool read_net(std::string fname);
	bool write_net(std::string fname);
	void set_gnet(GPUNet *g);

private:
	GPUNet *gnet;

	std::string get_next_string(std::ifstream &in);
	int get_next_int(std::ifstream &in);
	long get_next_long(std::ifstream &in);
	float get_next_float(std::ifstream &in);
	float* get_next_list(std::ifstream &in);
};

#endif /* NETIO_H_ */
